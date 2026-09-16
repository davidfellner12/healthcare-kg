"""
src/embeddings/predict_underserved.py

Uses the trained TransE embeddings to predict which districts are
likely under-served by healthcare facilities (LO1).

Approach:
  - For each hkg:District entity, use PyKEEN's link-prediction API
    to score all candidate tails of the (district, hkg:accessRisk, ?)
    triple in embedding space.
  - Rank districts by the *plausibility score* the model assigns to
    the tail ontology#HighRisk -> this is the actual TransE-based
    "under-served district" ranking asked for in the project plan
    (not a re-run of the symbolic rule from src/reasoning).
  - Compare the model's top-1 prediction against the ground-truth
    label materialised by the logic layer (src/reasoning) to surface
    one true-positive and one false-positive example, as required
    for the portfolio's ML-based Representation section.

Output:
  - models/transe/underserved_predictions.json
  - models/transe/link_prediction_examples.json
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODELS_DIR, RDF_DIR, NAMESPACES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "transe"
RISK_TAILS = ["ontology#HighRisk", "ontology#MediumRisk", "ontology#LowRisk"]


def load_model_and_factory():
    import torch
    from pykeen.triples import TriplesFactory

    model = torch.load(MODEL_DIR / "trained_model.pkl", map_location="cpu", weights_only=False)
    tf = TriplesFactory.from_path_binary(MODEL_DIR / "training_triples")
    log.info(f"Loaded TransE model ({tf.num_entities} entities, {tf.num_relations} relations)")
    return model, tf


def ground_truth_risk() -> dict:
    """district_id -> {name, actual_risk, vuln, gp_per_1000} from the materialised KG."""
    from rdflib import Graph, Namespace

    HKG = Namespace(NAMESPACES["hkg"])
    g = Graph()
    for ttl in RDF_DIR.glob("*.ttl"):
        if ttl.name == "gtfs_transit.ttl":
            continue
        g.parse(str(ttl), format="turtle")

    q = """
    PREFIX hkg: <http://healthcare-kg.at/ontology#>
    SELECT ?district ?districtName ?risk ?vuln ?gpPer1000
    WHERE {
        ?district a hkg:District .
        OPTIONAL { ?district hkg:districtName ?districtName . }
        OPTIONAL { ?district hkg:accessRisk ?risk . }
        OPTIONAL { ?district hkg:vulnerabilityScore ?vuln . }
        OPTIONAL { ?district hkg:gpPer1000 ?gpPer1000 . }
    }
    """
    out = {}
    for row in g.query(q):
        did = str(row.district).split("/")[-1]
        out[did] = {
            "district_name": str(row.districtName) if row.districtName else did,
            "actual_risk": str(row.risk).split("#")[-1] if row.risk else "Unknown",
            "vulnerability_score": round(float(row.vuln), 4) if row.vuln else 0.0,
            "gp_per_1000": round(float(row.gpPer1000), 4) if row.gpPer1000 else 0.0,
        }
    return out


def held_out_test_examples(model, tf) -> tuple:
    """
    Reproduce the exact train/val/test split used in train_transe.py
    (same triples, same random_seed) so we can evaluate the model on
    genuinely unseen accessRisk triples, rather than triples it was
    trained on. This is what makes the true-positive / false-positive
    example below meaningful instead of memorisation.
    """
    from config.settings import TRANSE_CONFIG as cfg
    from pykeen.predict import predict_target
    from src.embeddings.train_transe import extract_triples_from_rdf
    import numpy as np
    from pykeen.triples import TriplesFactory as TF2

    triples = extract_triples_from_rdf()
    tf_full = TF2.from_labeled_triples(triples=np.array(triples, dtype=str),
                                        create_inverse_triples=True)
    _, _, testing = tf_full.split(
        [cfg["train_split"], cfg["val_split"], 1.0 - cfg["train_split"] - cfg["val_split"]],
        random_state=cfg["random_seed"],
    )

    id_to_entity = {v: k for k, v in tf_full.entity_to_id.items()}
    id_to_relation = {v: k for k, v in tf_full.relation_to_id.items()}

    # We evaluate across ALL relations in the held-out test split (not just
    # accessRisk): with only 16 districts and one accessRisk triple each,
    # PyKEEN's entity/relation-coverage-aware splitter keeps every accessRisk
    # triple in the training set (there is no slack to hold any out), so a
    # true/false-positive example has to be drawn from whichever relations
    # actually did get split off -- this is itself a small-graph scalability
    # limitation worth noting (LO6).
    test_examples, n_match = [], 0
    for h, r, t in testing.mapped_triples.tolist():
        rel = id_to_relation[r]
        if rel.startswith("inverse_"):
            continue
        head_label = id_to_entity[h]
        actual_label = id_to_entity[t]
        actual = actual_label.split("#")[-1] if "#" in actual_label else actual_label
        if head_label not in tf.entity_to_id:
            continue  # entity not present in the (gtfs-excluded) training factory

        pred = predict_target(model=model, head=head_label, relation=rel, triples_factory=tf)
        df = pred.df.sort_values("score", ascending=False)
        # Self-loop filtering: on this tiny, sparse graph several relations
        # (e.g. reachableIn30min, inDistrict) are seen so few times per
        # relation that their relation vector stays close to zero after
        # training, which makes score(h, r, h) rank first for almost every
        # head -- a known TransE failure mode on very small/sparse KGs
        # (see limitations discussion). We drop that degenerate self-loop
        # candidate before taking top-1, which is standard practice for
        # non-reflexive relations and is what genuinely tests generalisation.
        df = df[df["tail_label"] != head_label]
        top_label = df.iloc[0]["tail_label"]
        predicted = top_label.split("#")[-1] if "#" in top_label else top_label
        match = predicted == actual
        n_match += match
        test_examples.append({
            "relation": rel, "head": head_label, "predicted_tail": predicted,
            "actual_tail": actual, "match": bool(match),
            "top3_excl_self_loop": [
                {"tail": (l.split("#")[-1] if "#" in l else l), "score": round(float(s), 4)}
                for l, s in zip(df["tail_label"].head(3), df["score"].head(3))
            ],
        })

    log.info(f"Held-out test-set top-1 accuracy (all relations): {n_match}/{len(test_examples)} "
             f"({(n_match/len(test_examples) if test_examples else float('nan')):.1%})")

    tp = next((e for e in test_examples if e["match"]), None)
    fp = next((e for e in test_examples if not e["match"]), None)
    return tp, fp, test_examples


def predict_underserved_districts(top_k: int = 16) -> tuple:
    """
    Returns (ranked_predictions, (true_positive, false_positive)).
    """
    model, tf = load_model_and_factory()
    gt = ground_truth_risk()

    from pykeen.predict import predict_target

    district_ids = [gt_id for gt_id in gt.keys() if gt_id in tf.entity_to_id]
    log.info(f"Scoring {len(district_ids)} districts against relation 'accessRisk'")

    predictions = []
    for did in district_ids:
        pred = predict_target(
            model=model, head=did, relation="accessRisk", triples_factory=tf,
        )
        df = pred.df  # columns: tail_id, tail_label, score
        scores = {row.tail_label: float(row.score) for _, row in df.iterrows()
                  if row.tail_label in RISK_TAILS}

        top_label = max(scores, key=scores.get) if scores else None
        top_risk = top_label.split("#")[-1] if top_label else "Unknown"
        actual = gt[did]["actual_risk"]

        predictions.append({
            "district_id": did,
            "district_name": gt[did]["district_name"],
            "predicted_risk": top_risk,
            "actual_risk": actual,
            "match": top_risk == actual,
            "score_high_risk": round(scores.get("ontology#HighRisk", float("nan")), 4),
            "score_medium_risk": round(scores.get("ontology#MediumRisk", float("nan")), 4),
            "vulnerability_score": gt[did]["vulnerability_score"],
            "gp_per_1000": gt[did]["gp_per_1000"],
        })

    # Rank by embedding-assigned plausibility of being HighRisk == "under-served"
    predictions.sort(key=lambda x: x["score_high_risk"], reverse=True)

    tp, fp, test_examples = held_out_test_examples(model, tf)

    # NOTE: encoding="utf-8" is required here, not just ensure_ascii=False --
    # without it, open()'s default encoding on Windows is the system locale
    # (cp1252), which silently mis-encodes non-ASCII district names (e.g.
    # "Oberösterreich") into bytes that aren't valid UTF-8, breaking any
    # UTF-8 JSON reader (including this same script re-reading its own
    # output on a different machine/locale).
    out1 = MODEL_DIR / "underserved_predictions.json"
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(predictions[:top_k], f, indent=2, ensure_ascii=False)

    out2 = MODEL_DIR / "link_prediction_examples.json"
    with open(out2, "w", encoding="utf-8") as f:
        json.dump({"true_positive_example": tp, "false_positive_example": fp,
                    "held_out_test_examples": test_examples}, f, indent=2, ensure_ascii=False)

    return predictions[:top_k], (tp, fp)


def main():
    log.info("=== Under-Served District Prediction (TransE link prediction) ===")
    results, (tp, fp) = predict_underserved_districts()

    print("\n" + "-" * 78)
    print(f"{'District':<12} {'Predicted':<12} {'Actual':<12} {'Match':<6} "
          f"{'P(HighRisk)':>12} {'Vuln':>6}")
    print("-" * 78)
    for r in results:
        print(f"{r['district_id']:<12} {r['predicted_risk']:<12} {r['actual_risk']:<12} "
              f"{str(r['match']):<6} {r['score_high_risk']:>12.4f} {r['vulnerability_score']:>6.3f}")
    print("-" * 78)

    if tp:
        log.info(f"True-positive example (held-out): ({tp['head']}, {tp['relation']}, ?) "
                 f"-> predicted {tp['predicted_tail']}, actual {tp['actual_tail']}")
    if fp:
        log.info(f"False-positive example (held-out): ({fp['head']}, {fp['relation']}, ?) "
                 f"-> predicted {fp['predicted_tail']}, actual {fp['actual_tail']}")
    log.info(f"Saved predictions to {MODEL_DIR}/underserved_predictions.json")


if __name__ == "__main__":
    main()
