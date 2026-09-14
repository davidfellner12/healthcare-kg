"""
src/gnn/train_graphsage.py

Trains a GraphSAGE model (PyTorch Geometric) to predict demographic
risk scores for districts, supporting what-if analysis for
hypothetical facility placements (LO3).

Node types:  District, HealthcareFacility, Stop
Edge types:
  - facility → district   (inDistrict)
  - facility → stop       (nearestStop)
  - district → district   (geographic adjacency / shared border)

Node features:
  District:  [population, age65plusPct, age0to4Pct, gpPer1000,
               carOwnershipPct, medianIncome (normalised)]
  Facility:  [type_one_hot (hospital/GP/pharmacy), bedCount]
  Stop:      [lat, lon (normalised)]

Target (district nodes): vulnerabilityScore ∈ [0, 1]

Output: models/graphsage/
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODELS_DIR, NAMESPACES, GRAPHSAGE_CONFIG, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "graphsage"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class GraphSAGE:
    """
    Lazily-defined at import time via _make_graphsage_class() once torch is
    available -- see module bottom. Placeholder so `from ... import GraphSAGE`
    resolves even before training has run in this process.
    """
    pass


def _make_graphsage_class():
    """Build the real torch.nn.Module GraphSAGE class (needs torch + PyG)."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from config.settings import GRAPHSAGE_CONFIG as cfg

    class _GraphSAGE(torch.nn.Module):
        def __init__(self, in_ch, hidden_ch, out_ch=1, num_layers=2, dropout=0.3):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_ch, hidden_ch, aggr=cfg["aggr"]))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_ch, hidden_ch, aggr=cfg["aggr"]))
            self.convs.append(SAGEConv(hidden_ch, out_ch, aggr=cfg["aggr"]))
            self.dropout = dropout

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index).relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.convs[-1](x, edge_index).squeeze(-1).sigmoid()

    return _GraphSAGE


try:
    GraphSAGE = _make_graphsage_class()
except ImportError:
    pass  # torch / torch_geometric not installed; placeholder class stays


def build_node_features() -> dict:
    """
    Build node feature matrices for districts, facilities, stops.
    Returns a dict with arrays ready for PyG Data objects.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # District features
    csv = RAW_DIR / "demographics.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        from src.ingestion.demographics_ingestion import SAMPLE_DISTRICTS
        df = pd.DataFrame(SAMPLE_DISTRICTS)

    feat_cols = ["population", "age_65plus_pct", "age_0_4_pct",
                 "car_ownership_pct", "median_income"]
    X_dist = df[feat_cols].values.astype(float)

    # Add GP-per-1000
    gp_per_1000 = (df["gp_count"].values / df["population"].values * 1000).reshape(-1, 1)
    X_dist = np.hstack([X_dist, gp_per_1000])

    scaler = MinMaxScaler()
    X_dist = scaler.fit_transform(X_dist)

    # Vulnerability labels
    elder    = df["age_65plus_pct"].values / 30.0
    young    = df["age_0_4_pct"].values    / 10.0
    nocar    = 1.0 - df["car_ownership_pct"].values / 100.0
    low_inc  = 1.0 - df["median_income"].values / 50000.0
    y = (0.4 * elder + 0.2 * young + 0.25 * nocar + 0.15 * low_inc).clip(0, 1)

    district_ids = df["district_id"].tolist()

    return {
        "X_district": X_dist.astype(np.float32),
        "y_district": y.astype(np.float32),
        "district_ids": district_ids,
        "n_districts": len(df),
    }


def build_graph_edges(node_data: dict) -> dict:
    """
    Build edge index for PyG from geographic adjacency + facility links.
    Adjacency defined by shared-prefix of district IDs (same state).
    """
    district_ids = node_data["district_ids"]
    id_to_idx = {did: i for i, did in enumerate(district_ids)}

    edges_src, edges_dst = [], []

    # Intra-state adjacency (all districts in same state connect)
    state_groups: dict = {}
    for did in district_ids:
        state = did[:4]   # "AT-9", "AT-4", "AT-7"
        state_groups.setdefault(state, []).append(did)

    for dids in state_groups.values():
        for i, di in enumerate(dids):
            for dj in dids[i+1:]:
                ii, jj = id_to_idx[di], id_to_idx[dj]
                edges_src += [ii, jj]
                edges_dst += [jj, ii]

    return {
        "edge_index_src": edges_src,
        "edge_index_dst": edges_dst,
    }


def train_graphsage(node_data: dict, edge_data: dict) -> dict:
    """Train GraphSAGE; returns metrics dict."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv

        cfg = GRAPHSAGE_CONFIG
        torch.manual_seed(cfg["random_seed"])

        X = torch.tensor(node_data["X_district"])
        y = torch.tensor(node_data["y_district"])
        n = node_data["n_districts"]

        edge_index = torch.tensor(
            [edge_data["edge_index_src"], edge_data["edge_index_dst"]],
            dtype=torch.long
        )

        # Train/val/test masks
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["random_seed"]))
        train_size = int(0.6 * n)
        val_size   = int(0.2 * n)

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)

        train_mask[idx[:train_size]]                  = True
        val_mask  [idx[train_size:train_size+val_size]] = True
        test_mask [idx[train_size+val_size:]]         = True

        data = Data(x=X, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        # GraphSAGE model class is defined once at module level (see
        # _make_graphsage_class above) so predict_risk_scores.py can import
        # and reload it for what-if inference without retraining.
        model = GraphSAGE(
            in_ch=X.shape[1],
            hidden_ch=cfg["hidden_channels"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"]
        )

        best_val_loss = float("inf")
        best_state = None

        log.info(f"Training GraphSAGE: {cfg['epochs']} epochs …")
        for epoch in range(cfg["epochs"]):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = F.mse_loss(out[data.val_mask], data.y[data.val_mask]).item()
                log.info(f"  Epoch {epoch:>4}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), MODEL_DIR / "graphsage_weights.pt")

        # Test MAE
        model.eval()
        with torch.no_grad():
            preds = model(data.x, data.edge_index)
            mae   = F.l1_loss(preds[data.test_mask], data.y[data.test_mask]).item()
        log.info(f"Test MAE: {mae:.4f}")

        # Save predictions
        all_preds = preds.detach().numpy()
        pred_records = [
            {"district_id": did, "predicted_risk_score": float(all_preds[i]),
             "true_score": float(node_data["y_district"][i])}
            for i, did in enumerate(node_data["district_ids"])
        ]
        with open(MODEL_DIR / "risk_scores.json", "w") as f:
            json.dump(pred_records, f, indent=2)

        metrics = {"test_mae": mae, "best_val_loss": best_val_loss}
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    except ImportError as e:
        log.error(f"PyTorch Geometric not available: {e}")
        log.info("Install: pip install torch-geometric")
        # Save placeholder
        placeholder = {"status": "not_trained", "reason": str(e)}
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(placeholder, f)
        return placeholder


def main():
    log.info("=== GraphSAGE Training Pipeline ===")
    node_data = build_node_features()
    edge_data = build_graph_edges(node_data)
    log.info(f"Graph: {node_data['n_districts']} nodes, "
             f"{len(edge_data['edge_index_src'])} directed edges")
    metrics = train_graphsage(node_data, edge_data)
    log.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
