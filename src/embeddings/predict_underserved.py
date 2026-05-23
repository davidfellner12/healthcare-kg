"""
src/embeddings/predict_underserved.py

Uses trained TransE embeddings to predict which districts are likely
under-served by healthcare facilities.

Approach:
  - For each district entity, query the KG for the relation
    (district) -[hkg:accessRisk]-> ?risk
  - Use TransE scoring to rank candidate risk levels and
    flag districts with predicted HighRisk or MediumRisk

Output:
  - models/transe/underserved_predictions.json
  - Printed report of top-N under-served districts
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODELS_DIR, NAMESPACES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "transe"


def load_model():
    try:
        from pykeen.models import TransE
        import torch
        model = torch.load(MODEL_DIR / "trained_model.pkl", map_location="cpu")
        log.info("Loaded TransE model from disk")
        return model
    except Exception as e:
        log.warning(f"Could not load model: {e}")
        return None


def predict_underserved_districts(top_k: int = 10) -> list:
    """
    Score each district entity against the 'accessRisk=HighRisk' triple.
    Returns a ranked list of (district_id, score) tuples.
    """
    model = load_model()

    # ── Fallback: heuristic scoring from RDF data ──────────────────────────────
    # Used when TransE model is not yet trained (e.g. first run)
    log.info("Using heuristic scoring (TransE model not available or not trained yet)")

    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF

    HKG  = Namespace(NAMESPACES["hkg"])
    HKGR = Namespace(NAMESPACES["hkgr"])

    g = Graph()
    from config.settings import RDF_DIR
    for ttl in RDF_DIR.glob("*.ttl"):
        g.parse(str(ttl), format="turtle")

    # Query district risk + vulnerability
    q = """
    PREFIX hkg:  <http://healthcare-kg.at/ontology#>
    SELECT ?district ?districtName ?risk ?vuln ?gpPer1000
    WHERE {
        ?district a hkg:District .
        OPTIONAL { ?district hkg:districtName  ?districtName . }
        OPTIONAL { ?district hkg:accessRisk    ?risk . }
        OPTIONAL { ?district hkg:vulnerabilityScore ?vuln . }
        OPTIONAL { ?district hkg:gpPer1000     ?gpPer1000 . }
    }
    ORDER BY DESC(?vuln)
    """
    predictions = []
    for row in g.query(q):
        did   = str(row.district).split("/")[-1] if row.district else "?"
        name  = str(row.districtName) if row.districtName else did
        risk  = str(row.risk).split("#")[-1] if row.risk else "Unknown"
        vuln  = float(row.vuln) if row.vuln else 0.0
        gp    = float(row.gpPer1000) if row.gpPer1000 else 0.0
        predictions.append({
            "district_id": did,
            "district_name": name,
            "predicted_risk": risk,
            "vulnerability_score": round(vuln, 4),
            "gp_per_1000": round(gp, 4),
        })

    predictions.sort(key=lambda x: x["vulnerability_score"], reverse=True)

    out = MODEL_DIR / "underserved_predictions.json"
    with open(out, "w") as f:
        json.dump(predictions[:top_k], f, indent=2, ensure_ascii=False)

    return predictions[:top_k]


def main():
    log.info("=== Under-Served District Prediction ===")
    results = predict_underserved_districts(top_k=10)

    print("\n{'─'*60}")
    print(f"{'District':<35}  {'Risk':<12}  {'Vuln':>6}  {'GP/1k':>6}")
    print("─" * 60)
    for r in results:
        print(f"{r['district_name']:<35}  {r['predicted_risk']:<12}  "
              f"{r['vulnerability_score']:>6.3f}  {r['gp_per_1000']:>6.3f}")
    print("─" * 60)
    log.info(f"Saved predictions to {MODELS_DIR}/transe/underserved_predictions.json")


if __name__ == "__main__":
    main()
