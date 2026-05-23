"""
src/reasoning/load_triplestore.py

Loads all RDF Turtle files into the Oxigraph (or Jena) triplestore.
Also supports applying SPARQL UPDATE patch files.

Usage:
  # Load all TTL files into Oxigraph store
  python src/reasoning/load_triplestore.py

  # Apply a SPARQL patch (from kg_evolution.py)
  python src/reasoning/load_triplestore.py --apply-patch data/rdf/patch_remove_KA001_*.sparql
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RDF_DIR, NAMESPACES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_into_rdflib():
    """In-memory load using rdflib (no server required)."""
    from rdflib import Graph, Namespace

    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    total = 0
    for ttl in sorted(RDF_DIR.glob("*.ttl")):
        before = len(g)
        g.parse(str(ttl), format="turtle")
        added = len(g) - before
        log.info(f"  {ttl.name}: +{added} triples (total: {len(g)})")
        total += added

    log.info(f"Loaded {total} triples into in-memory graph")
    return g


def apply_patch(patch_file: Path, graph):
    """Apply a SPARQL UPDATE patch to the graph."""
    query = patch_file.read_text()
    log.info(f"Applying patch: {patch_file.name}")
    graph.update(query)
    log.info("Patch applied")
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-patch", metavar="FILE", help="Apply a .sparql UPDATE patch")
    args = parser.parse_args()

    g = load_into_rdflib()

    if args.apply_patch:
        patch_path = Path(args.apply_patch)
        if not patch_path.exists():
            log.error(f"Patch file not found: {patch_path}")
            sys.exit(1)
        g = apply_patch(patch_path, g)
        log.info(f"Graph after patch: {len(g)} triples")

    log.info("Triplestore ready. Use src/api/app.py to query via REST/SPARQL.")


if __name__ == "__main__":
    main()
