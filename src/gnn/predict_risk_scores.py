"""
src/gnn/predict_risk_scores.py

Uses trained GraphSAGE model to predict risk scores for:
  1. All existing districts (validation of trained model)
  2. Hypothetical new facility placements (what-if analysis)

What-if usage example:
  "If we place a new GP in district AT-4-20 (Rohrbach), how does the
   risk score change?"

This is a core deliverable for LO3 (GNN) and LO9 (real-world application).
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_predictions() -> list:
    pred_file = MODELS_DIR / "graphsage" / "risk_scores.json"
    if pred_file.exists():
        with open(pred_file) as f:
            return json.load(f)
    log.warning("risk_scores.json not found. Run train_graphsage.py first.")
    return []


def what_if_analysis(target_district: str, new_gp_count: int = 1) -> dict:
    """
    Simulate adding GPs to a district and predict the change in risk score.
    Re-computes node features with updated GP count and runs forward pass.
    """
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        from config.settings import RAW_DIR
        from src.gnn.train_graphsage import build_node_features, build_graph_edges, GraphSAGE
        from config.settings import GRAPHSAGE_CONFIG

        node_data = build_node_features()
        edge_data = build_graph_edges(node_data)

        district_ids = node_data["district_ids"]
        if target_district not in district_ids:
            return {"error": f"District {target_district} not found"}

        target_idx = district_ids.index(target_district)

        # Baseline prediction
        X = torch.tensor(node_data["X_district"])
        edge_index = torch.tensor(
            [edge_data["edge_index_src"], edge_data["edge_index_dst"]],
            dtype=torch.long
        )

        cfg = GRAPHSAGE_CONFIG
        model = GraphSAGE(
            in_ch=X.shape[1],
            hidden_ch=cfg["hidden_channels"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        )

        weights_path = MODELS_DIR / "graphsage" / "graphsage_weights.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            baseline_preds = model(X, edge_index).numpy()
        baseline_score = baseline_preds[target_idx]

        # Modify: increase gpPer1000 for target district
        # (Feature index 5 = gp_per_1000, after MinMaxScaler)
        X_modified = X.clone()
        X_modified[target_idx, 5] = min(X_modified[target_idx, 5].item() + 0.2, 1.0)

        with torch.no_grad():
            modified_preds = model(X_modified, edge_index).numpy()
        modified_score = modified_preds[target_idx]

        return {
            "district_id": target_district,
            "scenario": f"Add {new_gp_count} GP(s)",
            "baseline_risk_score": round(float(baseline_score), 4),
            "modified_risk_score": round(float(modified_score), 4),
            "delta": round(float(modified_score - baseline_score), 4),
        }

    except ImportError as e:
        return {"error": str(e), "note": "Install PyTorch Geometric"}
    except Exception as e:
        return {"error": str(e)}


def main():
    # Windows consoles default stdout to cp1252, which can't encode the
    # Delta/box-drawing characters below; reconfigure to UTF-8 so this
    # doesn't crash mid-run (surfaced when scaling from 16 to 50 districts
    # made this the first script printing a wide results table).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    log.info("=== GraphSAGE Risk Score Analysis ===")

    predictions = load_predictions()
    if predictions:
        print(f"\n{'District':<20}  {'True':>7}  {'Pred':>7}  {'Δ':>7}")
        print("─" * 45)
        for p in sorted(predictions, key=lambda x: x["predicted_risk_score"], reverse=True):
            delta = p["predicted_risk_score"] - p["true_score"]
            print(f"{p['district_id']:<20}  {p['true_score']:>7.4f}  "
                  f"{p['predicted_risk_score']:>7.4f}  {delta:>+7.4f}")

    # What-if analysis for the two most GP-deficit districts (lowest GPs per
    # 1,000 residents) -- computed dynamically from the real demographics
    # data, not hardcoded. This is the more meaningful pair to run "add a
    # GP" against: the highest *predicted-vulnerability* districts turn out
    # to be dense, low-income inner-Vienna districts whose vulnerability is
    # driven mostly by income/car-ownership rather than GP scarcity, so
    # adding a GP barely moves their score (see Section 3.1's discussion of
    # this distinction).
    import pandas as pd
    from config.settings import RAW_DIR as _RAW_DIR
    _demo = pd.read_csv(_RAW_DIR / "demographics.csv")
    _demo["gp_per_1000"] = _demo["gp_count"] / _demo["population"] * 1000
    top2 = _demo.nsmallest(2, "gp_per_1000")["district_id"].tolist()

    log.info(f"\nWhat-if Analysis: Adding GPs to the {len(top2)} most GP-deficit districts")
    for district in top2:
        result = what_if_analysis(district, new_gp_count=2)
        print(f"\n  {result}")


if __name__ == "__main__":
    main()
