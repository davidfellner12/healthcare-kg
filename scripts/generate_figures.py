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
    """
    Scatter of predicted vs. true vulnerability score, one point per
    district, against the y=x "perfect prediction" line. This replaces an
    earlier version (both series plotted against a district-ID x-axis with
    50 tiny tick labels) that was hard to read at a glance -- a scatter
    against the diagonal is the standard way to show model fit and needs no
    per-district labels to be immediately interpretable, while still
    calling out the districts a reader would actually want to identify.
    """
    import pandas as pd
    preds = json.load(open(ROOT / "models" / "graphsage" / "risk_scores.json", encoding="utf-8"))
    demo = pd.read_csv(ROOT / "data" / "raw" / "demographics.csv").set_index("district_id")
    mae = json.load(open(ROOT / "models" / "graphsage" / "metrics.json", encoding="utf-8"))["test_mae"]

    true = [p["true_score"] for p in preds]
    pred = [p["predicted_risk_score"] for p in preds]
    lo, hi = min(true + pred) - 0.02, max(true + pred) + 0.02

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([lo, hi], [lo, hi], "--", color="#a0aec0", linewidth=1.3, label="Perfect prediction (y = x)")
    ax.scatter(true, pred, s=42, color="#1a365d", alpha=0.75, zorder=3, label="District")

    # Label the 3 highest and 1 lowest true-vulnerability districts by name,
    # so a reader can immediately tie the scatter back to real places.
    # Offsets are staggered manually because the 3 highest districts sit
    # close together in one corner and would otherwise overlap.
    by_true = sorted(preds, key=lambda p: p["true_score"], reverse=True)
    to_label = [(by_true[0], (8, 10)), (by_true[1], (8, -4)), (by_true[2], (8, -18)),
                (by_true[-1], (8, 4))]
    for p, offset in to_label:
        name = str(demo.loc[p["district_id"], "name"]).split(",")[-1].strip() or p["district_id"]
        ax.annotate(f"{name} ({p['district_id']})", (p["true_score"], p["predicted_risk_score"]),
                    textcoords="offset points", xytext=offset, fontsize=8, color="#2d3748")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True vulnerability score (rule-computed, Section 4.1)")
    ax.set_ylabel("GraphSAGE-predicted vulnerability score")
    ax.set_title(f"GraphSAGE prediction accuracy, all {len(preds)} districts\n"
                 f"(test MAE = {mae:.4f} on a [0,1]-scaled score)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "graphsage_fit.png")
    plt.close(fig)
    print("graphsage_fit.png")


def graphsage_whatif():
    """
    Before/after bars for the what-if scenario, redesigned for a
    non-technical reader: full district names instead of just IDs, a
    zoomed y-axis so the actual before/after difference is visible as a
    bar-height change (a 0-based axis made both bars look identical), and
    an explicit second panel giving the same two bars on the full [0,1]
    scale for honest context -- a reader gets both "yes, it moved" and
    "here is how small that move is against the full range" without either
    view hiding the other.
    """
    from src.gnn.predict_risk_scores import what_if_analysis
    demo = pd.read_csv(ROOT / "data" / "raw" / "demographics.csv")
    demo["gp_per_1000"] = demo["gp_count"] / demo["population"] * 1000
    all_preds = json.load(open(ROOT / "models" / "graphsage" / "risk_scores.json", encoding="utf-8"))
    score_lo = min(p["true_score"] for p in all_preds)
    score_hi = max(p["true_score"] for p in all_preds)

    targets = demo.nsmallest(2, "gp_per_1000")["district_id"].tolist()
    results = [what_if_analysis(d, new_gp_count=2) for d in targets]
    names = {r["district_id"]: str(demo.set_index("district_id").loc[r["district_id"], "name"])
             for r in results}
    labels = [f"{names[r['district_id']]}\n({r['district_id']})" for r in results]
    baseline = [r["baseline_risk_score"] for r in results]
    modified = [r["modified_risk_score"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    x = range(len(results))
    width = 0.32

    # Left panel: zoomed in, so the actual change is visible.
    zoom_lo = min(baseline + modified) - 0.006
    zoom_hi = max(baseline + modified) + 0.012
    ax1.bar([i - width / 2 for i in x], baseline, width, label="Baseline", color="#274472", zorder=3)
    ax1.bar([i + width / 2 for i in x], modified, width, label="After adding 2 GPs", color="#38a169", zorder=3)
    for i, r in enumerate(results):
        pct = 100 * r["delta"] / r["baseline_risk_score"]
        top = max(r["baseline_risk_score"], r["modified_risk_score"])
        ax1.annotate(f"{r['delta']:+.4f} ({pct:+.2f}%)", (i, top + (zoom_hi - zoom_lo) * 0.03),
                     ha="center", fontsize=8.5, fontweight="bold", color="#1a202c")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(zoom_lo, zoom_hi)
    ax1.set_ylabel("Predicted vulnerability score (zoomed in)")
    ax1.set_title("The change itself", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2, axis="y", zorder=0)

    # Right panel: same bars on the full [0,1]-relevant scale, for context.
    ax2.axhspan(score_lo, score_hi, color="#edf2f7", zorder=0)
    ax2.bar([i - width / 2 for i in x], baseline, width, color="#274472", zorder=3)
    ax2.bar([i + width / 2 for i in x], modified, width, color="#38a169", zorder=3)
    ax2.set_xticks(list(x)); ax2.set_xticklabels([r["district_id"] for r in results], fontsize=9.5)
    ax2.set_ylim(0, 0.7)
    ax2.set_ylabel("Same bars, full scale")
    ax2.set_title(f"In context (shaded = full\nacross-district range\n"
                  f"{score_lo:.2f}–{score_hi:.2f})", fontsize=10)
    ax2.grid(alpha=0.2, axis="y", zorder=0)

    fig.suptitle("What-if: adding 2 GPs to the 2 most GP-deficit districts", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "graphsage_whatif.png", bbox_inches="tight")
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
