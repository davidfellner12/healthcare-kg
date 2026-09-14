"""
src/embeddings/train_transe.py

Trains a TransE KG embedding model on the healthcare KG using PyKEEN.
The model learns entity/relation embeddings that enable link prediction,
specifically to identify potentially under-served districts (LO1).

Workflow:
  1. Extract (head, relation, tail) triples from RDF files
  2. Build PyKEEN TriplesFactory
  3. Train TransE with configured hyperparameters
  4. Save model + entity/relation mappings for later inference

Output: models/transe/
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RDF_DIR, MODELS_DIR, NAMESPACES, TRANSE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "transe"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Triples extraction from RDF ───────────────────────────────────────────────

def extract_triples_from_rdf() -> list:
    """
    Convert RDF graph to (head, relation, tail) string tuples.
    Filters to relations relevant for link prediction.
    """
    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF, RDFS

    HKG  = Namespace(NAMESPACES["hkg"])
    HKGR = Namespace(NAMESPACES["hkgr"])

    # gtfs_transit.ttl is excluded here: its ~4,900 individual stop/route
    # entities are singletons under RELEVANT_PREDICATES (only rdf:type +
    # rdfs:label edges), so including them starves PyKEEN's train/val/test
    # entity-coverage split without adding any link-prediction signal.
    # The facility<->stop connection that matters (nearestStop) already
    # comes from facility_stop_links.ttl.
    g = Graph()
    for ttl in RDF_DIR.glob("*.ttl"):
        if ttl.name == "gtfs_transit.ttl":
            continue
        log.info(f"  Loading {ttl.name}")
        g.parse(str(ttl), format="turtle")

    log.info(f"Loaded graph: {len(g)} triples")

    # Relations to include in embedding training
    RELEVANT_PREDICATES = {
        str(HKG.inDistrict),
        str(HKG.inState),
        str(HKG.nearestStop),
        str(HKG.reachableIn15min),
        str(HKG.reachableIn30min),
        str(HKG.reachableIn60min),
        str(HKG.accessRisk),
        str(HKG.gpDeficit),
        str(RDF.type),
    }

    triples = []
    for s, p, o in g:
        ps = str(p)
        if ps in RELEVANT_PREDICATES:
            # Use short local names where possible
            head = str(s).split("/")[-1]
            rel  = ps.split("#")[-1] if "#" in ps else ps.split("/")[-1]
            tail = str(o).split("/")[-1] if str(o).startswith("http") else str(o)
            triples.append((head, rel, tail))

    log.info(f"Extracted {len(triples)} triples for embedding")
    return triples


def build_triples_factory(triples: list):
    """Build a PyKEEN TriplesFactory from extracted triples."""
    try:
        from pykeen.triples import TriplesFactory
        import numpy as np

        tf = TriplesFactory.from_labeled_triples(
            triples=np.array(triples, dtype=str),
            create_inverse_triples=True,
        )
        log.info(f"TriplesFactory: {tf.num_entities} entities, "
                 f"{tf.num_relations} relations, {tf.num_triples} triples")
        return tf
    except ImportError:
        log.warning("PyKEEN not installed; saving triples as JSON for manual training")
        out = MODEL_DIR / "triples.json"
        with open(out, "w") as f:
            json.dump(triples, f)
        log.info(f"Saved triples to {out}")
        return None


def train_transe(tf):
    """Train TransE using PyKEEN pipeline."""
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory

        cfg = TRANSE_CONFIG
        training, validation, testing = tf.split(
            [cfg["train_split"], cfg["val_split"],
             1.0 - cfg["train_split"] - cfg["val_split"]],
            random_state=cfg["random_seed"]
        )

        log.info("Starting TransE training …")
        result = pipeline(
            training=training,
            validation=validation,
            testing=testing,
            model="TransE",
            model_kwargs={
                "embedding_dim": cfg["embedding_dim"],
                "scoring_fct_norm": cfg["scoring_fct_norm"],
            },
            training_kwargs={
                "num_epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
            },
            optimizer="Adam",
            optimizer_kwargs={"lr": cfg["learning_rate"]},
            loss="MarginRankingLoss",
            loss_kwargs={"margin": cfg["margin"]},
            random_seed=cfg["random_seed"],
            device="cpu",
        )

        result.save_to_directory(str(MODEL_DIR))
        log.info(f"TransE model saved to {MODEL_DIR}")

        # Log key metrics
        metrics = result.metric_results.to_dict()
        realistic = metrics.get("both", {}).get("realistic", {})
        log.info(f"  Hits@10: {realistic.get('hits_at_10', float('nan')):.4f}")
        log.info(f"  MRR:     {realistic.get('inverse_harmonic_mean_rank', float('nan')):.4f}")

        return result

    except ImportError as e:
        log.error(f"PyKEEN import failed: {e}")
        log.info("Install with: pip install pykeen")
        # Save a placeholder metrics file so downstream scripts don't crash
        placeholder = {"status": "not_trained", "reason": str(e)}
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(placeholder, f)
        return None


def main():
    log.info("=== TransE Training Pipeline ===")
    triples = extract_triples_from_rdf()

    if not triples:
        log.error("No triples extracted. Run ingestion pipeline first.")
        return

    tf = build_triples_factory(triples)
    if tf is None:
        log.warning("TriplesFactory not built; check PyKEEN installation")
        return

    train_transe(tf)


if __name__ == "__main__":
    main()
