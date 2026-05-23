"""
src/gnn/build_graph.py

Constructs the heterogeneous graph data object for PyTorch Geometric
from the RDF KG and demographic data.

Run this before train_graphsage.py to inspect graph statistics.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    from src.gnn.train_graphsage import build_node_features, build_graph_edges

    log.info("=== Graph Construction ===")
    node_data = build_node_features()
    edge_data = build_graph_edges(node_data)

    n_nodes = node_data["n_districts"]
    n_edges = len(edge_data["edge_index_src"])
    feat_dim = node_data["X_district"].shape[1]

    log.info(f"Nodes (districts):   {n_nodes}")
    log.info(f"Feature dimensions:  {feat_dim}")
    log.info(f"Directed edges:      {n_edges}")
    log.info(f"Avg degree:          {n_edges / n_nodes:.1f}")

    vuln_mean = node_data["y_district"].mean()
    vuln_std  = node_data["y_district"].std()
    log.info(f"Target (vuln score): mean={vuln_mean:.3f}  std={vuln_std:.3f}")

    # Save graph stats
    import json
    stats = {
        "n_district_nodes": n_nodes,
        "n_features": feat_dim,
        "n_directed_edges": n_edges,
        "avg_degree": round(n_edges / n_nodes, 2),
        "target_mean": round(float(vuln_mean), 4),
        "target_std": round(float(vuln_std), 4),
        "district_ids": node_data["district_ids"],
    }
    out = MODELS_DIR / "graphsage" / "graph_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Graph stats saved to {out}")


if __name__ == "__main__":
    main()
