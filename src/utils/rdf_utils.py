"""
src/utils/rdf_utils.py

Helper functions for loading, merging, and querying RDF graphs.
"""

import logging
from pathlib import Path
from typing import List, Optional
from rdflib import Graph, ConjunctiveGraph, Namespace
from rdflib.plugins.sparql import prepareQuery

from config.settings import RDF_DIR, NAMESPACES

log = logging.getLogger(__name__)


def load_graph(paths: Optional[List[Path]] = None, fmt: str = "turtle") -> Graph:
    """
    Load one or more Turtle files into a single in-memory RDF graph.
    Defaults to all .ttl files in data/rdf/.
    """
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    if paths is None:
        paths = list(RDF_DIR.glob("*.ttl"))

    for p in paths:
        if p.exists():
            log.info(f"Loading {p} …")
            g.parse(str(p), format=fmt)
        else:
            log.warning(f"File not found: {p}")

    log.info(f"Total triples in graph: {len(g)}")
    return g


def merge_graphs(graphs: List[Graph]) -> Graph:
    """Merge multiple rdflib Graphs into one."""
    merged = Graph()
    for g in graphs:
        for triple in g:
            merged.add(triple)
    return merged


def sparql_query(graph: Graph, query_str: str):
    """Run a SPARQL SELECT query and return results as a list of dicts."""
    results = []
    for row in graph.query(query_str):
        results.append({str(var): str(val) for var, val in zip(row.labels, row)})
    return results


def sparql_update(graph: Graph, update_str: str) -> Graph:
    """Run a SPARQL INSERT/DELETE update in-place and return graph."""
    graph.update(update_str)
    return graph


def count_triples_by_type(graph: Graph) -> dict:
    """Return a dict of {rdf:type URI -> count} for quick inspection."""
    q = """
    SELECT ?type (COUNT(?s) AS ?count)
    WHERE { ?s a ?type }
    GROUP BY ?type
    ORDER BY DESC(?count)
    """
    return {r["type"]: int(r["count"]) for r in sparql_query(graph, q)}
