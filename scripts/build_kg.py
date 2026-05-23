"""
scripts/build_kg.py

Master pipeline script — runs all ingestion, reasoning, and
ML training steps in sequence.

Usage:
    python scripts/build_kg.py [--skip-ml]

Steps:
  1. Ingest healthcare facilities → RDF
  2. Ingest GTFS transit data     → RDF
  3. Ingest demographic data      → RDF
  4. Link facilities to stops     → RDF
  5. Materialise reachability     → RDF
  6. Materialise vulnerability    → RDF
  7. Train TransE embeddings      (skippable)
  8. Train GraphSAGE GNN          (skippable)
"""

import sys
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_kg")


def run_step(name: str, fn):
    log.info(f"\n{'='*60}")
    log.info(f"  STEP: {name}")
    log.info(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        log.info(f"  ✓ {name} completed in {time.time()-t0:.1f}s")
    except Exception as e:
        log.error(f"  ✗ {name} FAILED: {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ml", action="store_true",
                        help="Skip TransE and GraphSAGE training")
    parser.add_argument("--skip-ingestion", action="store_true",
                        help="Skip data ingestion (use cached RDF files)")
    args = parser.parse_args()

    if not args.skip_ingestion:
        from src.ingestion.healthcare_ingestion import main as ingest_healthcare
        run_step("1. Healthcare facility ingestion", ingest_healthcare)

        from src.ingestion.gtfs_ingestion import main as ingest_gtfs
        run_step("2. GTFS transit ingestion", ingest_gtfs)

        from src.ingestion.demographics_ingestion import main as ingest_demo
        run_step("3. Demographics ingestion", ingest_demo)

        from src.ingestion.link_facilities_to_stops import main as link_stops
        run_step("4. Facility → Stop geo-linking", link_stops)

    from src.reasoning.materialize_reachability import main as materialize
    run_step("5. Reachability + vulnerability materialisation", materialize)

    if not args.skip_ml:
        from src.embeddings.train_transe import main as train_transe
        run_step("6. TransE embedding training", train_transe)

        from src.embeddings.predict_underserved import main as predict
        run_step("7. Under-served zone prediction", predict)

        from src.gnn.train_graphsage import main as train_gnn
        run_step("8. GraphSAGE risk score training", train_gnn)
    else:
        log.info("Skipping ML steps (--skip-ml)")

    log.info("\n" + "="*60)
    log.info("  KG BUILD COMPLETE")
    log.info("  Start the API: python src/api/app.py")
    log.info("="*60)


if __name__ == "__main__":
    main()
