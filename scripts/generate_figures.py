"""
scripts/generate_figures.py

Regenerates the four data-driven figures used in the portfolio report from
the actual pipeline outputs (models/transe, models/graphsage, data/rdf).
Run after the full pipeline (scripts/build_kg.py) so the figures reflect
the current data.

Output: docs/figures/{transe_loss,graphsage_fit,graphsage_whatif,vulnerability_by_district}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})


def transe_loss():
    r = json.load(open(ROOT / "models" / "transe" / "results.json", encoding="utf-8"))
    losses = r["losses"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(1, len(losses) + 1), losses, color="#1a365d", linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Margin-ranking loss")
    ax.set_title("TransE training loss (200 epochs, 218 entities / 18 relations / 2,401 triples)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "transe_loss.png")
    plt.close(fig)
    print(f"transe_loss.png  (final loss={losses[-1]:.4f}, {len(losses)} epochs)")


def graphsage_fit():
    preds = json.load(open(ROOT / "models" / "graphsage" / "risk_scores.json", encoding="utf-8"))
    preds = sorted(preds, key=lambda p: p["true_score"])
    ids = [p["district_id"] for p in preds]
    true = [p["true_score"] for p in preds]
    pred = [p["predicted_risk_score"] for p in preds]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = range(len(ids))
    ax.plot(x, true, "o-", color="#1a365d", label="True vulnerability score", markersize=3, linewidth=1)
    ax.plot(x, pred, "o-", color="#e53e3e", label="GraphSAGE prediction", markersize=3, linewidth=1, alpha=0.85)
    ax.set_xticks(list(x)[::2])
    ax.set_xticklabels([ids[i] for i in x][::2], rotation=90, fontsize=6)
    ax.set_ylabel("Vulnerability score [0,1]")
    ax.set_title(f"GraphSAGE prediction vs. true vulnerability score, all {len(ids)} districts "
                 f"(test MAE = {json.load(open(ROOT/'models'/'graphsage'/'metrics.json', encoding='utf-8'))['test_mae']:.4f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "graphsage_fit.png")
    plt.close(fig)
    print("graphsage_fit.png")


def graphsage_whatif():
    from src.gnn.predict_risk_scores import what_if_analysis
    demo = pd.read_csv(ROOT / "data" / "raw" / "demographics.csv")
    demo["gp_per_1000"] = demo["gp_count"] / demo["population"] * 1000
    targets = demo.nsmallest(2, "gp_per_1000")["district_id"].tolist()
    results = [what_if_analysis(d, new_gp_count=2) for d in targets]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = range(len(results))
    width = 0.35
    baseline = [r["baseline_risk_score"] for r in results]
    modified = [r["modified_risk_score"] for r in results]
    labels = [r["district_id"] for r in results]
    ax.bar([i - width / 2 for i in x], baseline, width, label="Baseline", color="#274472")
    ax.bar([i + width / 2 for i in x], modified, width, label="+2 GPs", color="#38a169")
    for i, r in enumerate(results):
        ax.annotate(f"{r['delta']:+.4f}", (i, max(r['baseline_risk_score'], r['modified_risk_score']) + 0.005),
                    ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Predicted vulnerability score")
    ax.set_title("What-if: predicted vulnerability before/after adding 2 GPs\n(2 most GP-deficit districts)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "graphsage_whatif.png")
    plt.close(fig)
    print("graphsage_whatif.png", results)


def vulnerability_by_district():
    from rdflib import Graph, Namespace
    g = Graph()
    g.parse(ROOT / "data" / "rdf" / "demographics.ttl", format="turtle")
    g.parse(ROOT / "data" / "rdf" / "vulnerability.ttl", format="turtle")

    q = """
    PREFIX hkg: <http://healthcare-kg.at/ontology#>
    SELECT ?d ?name ?vuln ?risk WHERE {
        ?d a hkg:District ; hkg:districtName ?name ; hkg:vulnerabilityScore ?vuln ; hkg:accessRisk ?risk .
    }
    """
    rows = []
    for r in g.query(q):
        rows.append({"id": str(r.d).split("/")[-1], "name": str(r.name), "vuln": float(r.vuln),
                     "risk": str(r.risk).split("#")[-1]})
    rows.sort(key=lambda r: r["vuln"], reverse=True)

    colors = {"HighRisk": "#e53e3e", "MediumRisk": "#ed8936", "LowRisk": "#38a169"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = range(len(rows))
    ax.barh(y, [r["vuln"] for r in rows], color=[colors[r["risk"]] for r in rows])
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["id"] for r in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("Vulnerability score")
    ax.set_title(f"Materialised accessRisk / vulnerabilityScore, all {len(rows)} districts (V1-V4 rule output)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "vulnerability_by_district.png")
    plt.close(fig)
    n_high = sum(1 for r in rows if r["risk"] == "HighRisk")
    n_med = sum(1 for r in rows if r["risk"] == "MediumRisk")
    n_low = sum(1 for r in rows if r["risk"] == "LowRisk")
    print(f"vulnerability_by_district.png  (HighRisk={n_high}, MediumRisk={n_med}, LowRisk={n_low})")


if __name__ == "__main__":
    transe_loss()
    graphsage_fit()
    graphsage_whatif()
    vulnerability_by_district()
