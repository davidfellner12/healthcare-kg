"""
scripts/robustness_check.py

Re-trains TransE and GraphSAGE across multiple random seeds (in addition
to the official seed=42 run used everywhere else in the pipeline/report)
to check how much the headline metrics actually vary run-to-run. Writes
its own separate models/robustness_check.json rather than touching
models/transe/ or models/graphsage/, so it never affects the "official"
numbers quoted elsewhere in the report.

Run after scripts/build_kg.py (needs the ingested RDF + CSVs).
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import MODELS_DIR, TRANSE_CONFIG, GRAPHSAGE_CONFIG

SEEDS = [42, 1, 7, 123, 2024]


def transe_run(seed: int) -> dict:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    from src.embeddings.train_transe import extract_triples_from_rdf

    cfg = TRANSE_CONFIG
    triples = extract_triples_from_rdf()
    tf = TriplesFactory.from_labeled_triples(triples=np.array(triples, dtype=str),
                                              create_inverse_triples=True)
    training, validation, testing = tf.split(
        [cfg["train_split"], cfg["val_split"], 1.0 - cfg["train_split"] - cfg["val_split"]],
        random_state=seed,
    )
    t0 = time.time()
    result = pipeline(
        training=training, validation=validation, testing=testing,
        model="TransE",
        model_kwargs={"embedding_dim": cfg["embedding_dim"], "scoring_fct_norm": cfg["scoring_fct_norm"]},
        training_kwargs={"num_epochs": cfg["epochs"], "batch_size": cfg["batch_size"]},
        optimizer="Adam", optimizer_kwargs={"lr": cfg["learning_rate"]},
        loss="MarginRankingLoss", loss_kwargs={"margin": cfg["margin"]},
        random_seed=seed, device="cpu",
    )
    m = result.metric_results.to_dict()["both"]["realistic"]
    return {
        "seed": seed,
        "hits_at_1": m["hits_at_1"], "hits_at_10": m["hits_at_10"],
        "mrr": m["inverse_harmonic_mean_rank"],
        "seconds": round(time.time() - t0, 1),
    }


def graphsage_run(seed: int) -> dict:
    import torch
    import torch.nn.functional as F
    from src.gnn.train_graphsage import build_node_features, build_graph_edges, GraphSAGE

    cfg = GRAPHSAGE_CONFIG
    torch.manual_seed(seed)
    node_data = build_node_features()
    edge_data = build_graph_edges(node_data)
    X = torch.tensor(node_data["X_district"])
    y = torch.tensor(node_data["y_district"])
    n = node_data["n_districts"]
    edge_index = torch.tensor([edge_data["edge_index_src"], edge_data["edge_index_dst"]], dtype=torch.long)

    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_size, val_size = int(0.6 * n), int(0.2 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[idx[:train_size]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[idx[train_size:train_size + val_size]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[idx[train_size + val_size:]] = True

    model = GraphSAGE(in_ch=X.shape[1], hidden_ch=cfg["hidden_channels"],
                       num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val, best_state = float("inf"), None
    for epoch in range(cfg["epochs"]):
        model.train(); optimizer.zero_grad()
        out = model(X, edge_index)
        loss = F.mse_loss(out[train_mask], y[train_mask])
        loss.backward(); optimizer.step()
        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(out[val_mask], y[val_mask]).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X, edge_index)
        mae = F.l1_loss(preds[test_mask], y[test_mask]).item()
    return {"seed": seed, "test_mae": mae}


def main():
    print(f"=== Robustness check across {len(SEEDS)} seeds ===")
    transe_results, gsage_results = [], []
    for seed in SEEDS:
        print(f"-- TransE seed={seed} --")
        transe_results.append(transe_run(seed))
        print(transe_results[-1])
    for seed in SEEDS:
        print(f"-- GraphSAGE seed={seed} --")
        gsage_results.append(graphsage_run(seed))
        print(gsage_results[-1])

    def summarize(rows, key):
        vals = [r[key] for r in rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}

    summary = {
        "seeds": SEEDS,
        "transe": {
            "runs": transe_results,
            "hits_at_10": summarize(transe_results, "hits_at_10"),
            "mrr": summarize(transe_results, "mrr"),
        },
        "graphsage": {
            "runs": gsage_results,
            "test_mae": summarize(gsage_results, "test_mae"),
        },
    }
    out = MODELS_DIR / "robustness_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")
    print("TransE Hits@10:", summary["transe"]["hits_at_10"])
    print("TransE MRR:", summary["transe"]["mrr"])
    print("GraphSAGE test MAE:", summary["graphsage"]["test_mae"])


if __name__ == "__main__":
    main()
