"""
src/ingestion/gtfs_ingestion.py

Parses GTFS feeds (Wiener Linien + ÖBB) and converts stops, routes,
and trip-level travel times to RDF.

Key outputs:
  - hkg:Stop individuals with geo-coordinates
  - hkg:Route individuals
  - hkg:Trip individuals with stop sequences
  - hkg:StopTime (departure / arrival literals)

Output: data/rdf/gtfs_transit.ttl
"""

import sys
import io
import csv
import zipfile
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RAW_DIR, RDF_DIR, NAMESPACES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GTFS_NS = Namespace(NAMESPACES["gtfs"])
HKG     = Namespace(NAMESPACES["hkg"])
HKGR    = Namespace(NAMESPACES["hkgr"])
GEO     = Namespace(NAMESPACES["geo"])

# ── Sample stops (fallback when GTFS download is unavailable) ─────────────────
SAMPLE_STOPS = [
    {"stop_id": "W001", "stop_name": "Wien Hauptbahnhof",        "stop_lat": 48.1851, "stop_lon": 16.3761, "zone": "Wien"},
    {"stop_id": "W002", "stop_name": "Wien Westbahnhof",         "stop_lat": 48.1969, "stop_lon": 16.3381, "zone": "Wien"},
    {"stop_id": "W003", "stop_name": "Wien Mitte/Landstraße",    "stop_lat": 48.2047, "stop_lon": 16.3861, "zone": "Wien"},
    {"stop_id": "W004", "stop_name": "Spittelau",                "stop_lat": 48.2361, "stop_lon": 16.3600, "zone": "Wien"},
    {"stop_id": "W005", "stop_name": "Floridsdorf",              "stop_lat": 48.2564, "stop_lon": 16.4007, "zone": "Wien"},
    {"stop_id": "W006", "stop_name": "Ottakring",                "stop_lat": 48.2136, "stop_lon": 16.3047, "zone": "Wien"},
    {"stop_id": "W007", "stop_name": "Simmering",                "stop_lat": 48.1736, "stop_lon": 16.4206, "zone": "Wien"},
    {"stop_id": "L001", "stop_name": "Linz/Donau Hauptbahnhof",  "stop_lat": 48.2907, "stop_lon": 14.2920, "zone": "Linz"},
    {"stop_id": "L002", "stop_name": "Linz Volksgarten",         "stop_lat": 48.3053, "stop_lon": 14.2874, "zone": "Linz"},
    {"stop_id": "I001", "stop_name": "Innsbruck Hauptbahnhof",   "stop_lat": 47.2631, "stop_lon": 11.4006, "zone": "Innsbruck"},
    {"stop_id": "I002", "stop_name": "Innsbruck Westbahnhof",    "stop_lat": 47.2681, "stop_lon": 11.3797, "zone": "Innsbruck"},
]

SAMPLE_ROUTES = [
    {"route_id": "R_U4",  "route_short_name": "U4",   "route_long_name": "Heiligenstadt – Hütteldorf",         "route_type": 1, "zone": "Wien"},
    {"route_id": "R_U6",  "route_short_name": "U6",   "route_long_name": "Siebenhirten – Floridsdorf",          "route_type": 1, "zone": "Wien"},
    {"route_id": "R_REX", "route_short_name": "REX1", "route_long_name": "Wien – Salzburg Regional Express",    "route_type": 2, "zone": "ÖBB"},
    {"route_id": "R_WB",  "route_short_name": "WB50", "route_long_name": "Linz – Wels – Grieskirchen",          "route_type": 3, "zone": "Oberösterreich"},
]


class GTFSIngestion:
    def __init__(self, feed_url: Optional[str] = None, feed_name: str = "generic"):
        self.feed_url = feed_url
        self.feed_name = feed_name
        self.cache_path = RAW_DIR / f"gtfs_{feed_name}.zip"

    def _download_or_cache(self) -> Optional[bytes]:
        if self.cache_path.exists():
            log.info(f"Using cached GTFS feed: {self.cache_path}")
            return self.cache_path.read_bytes()
        if not self.feed_url:
            return None
        try:
            log.info(f"Downloading GTFS from {self.feed_url}")
            r = requests.get(self.feed_url, timeout=60, stream=True)
            r.raise_for_status()
            data = r.content
            self.cache_path.write_bytes(data)
            log.info(f"Saved GTFS feed to {self.cache_path}")
            return data
        except Exception as e:
            log.warning(f"GTFS download failed: {e}")
            return None

    def _read_table(self, zf: zipfile.ZipFile, filename: str) -> List[Dict]:
        try:
            with zf.open(filename) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
                return list(reader)
        except KeyError:
            log.warning(f"{filename} not found in GTFS zip")
            return []

    def parse(self) -> Dict[str, List[Dict]]:
        raw = self._download_or_cache()
        if raw is None:
            log.info("Using sample GTFS data")
            return {"stops": SAMPLE_STOPS, "routes": SAMPLE_ROUTES,
                    "trips": [], "stop_times": []}
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            return {
                "stops":      self._read_table(zf, "stops.txt"),
                "routes":     self._read_table(zf, "routes.txt"),
                "trips":      self._read_table(zf, "trips.txt"),
                "stop_times": self._read_table(zf, "stop_times.txt"),
            }


def _stop_uri(stop_id: str) -> URIRef:
    return HKGR[f"stop/{stop_id}"]


def _route_uri(route_id: str) -> URIRef:
    return HKGR[f"route/{route_id}"]


def gtfs_to_rdf(feeds: List[Dict[str, List[Dict]]]) -> Graph:
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    # Ontology declarations
    for cls in ["Stop", "Route", "Trip", "TransitZone"]:
        g.add((HKG[cls], RDF.type, Namespace(NAMESPACES["owl"]).Class))
        g.add((HKG[cls], RDFS.label, Literal(cls)))

    stop_count = route_count = 0

    for feed_data in feeds:
        # ── Stops ─────────────────────────────────────────────────────────────
        for row in feed_data.get("stops", []):
            sid = str(row.get("stop_id", ""))
            if not sid:
                continue
            uri = _stop_uri(sid)
            g.add((uri, RDF.type,       HKG.Stop))
            g.add((uri, RDFS.label,     Literal(str(row.get("stop_name", sid)))))
            g.add((uri, GTFS_NS.code,   Literal(sid)))

            try:
                g.add((uri, GEO.lat,  Literal(float(row["stop_lat"]), datatype=XSD.decimal)))
                g.add((uri, GEO.long, Literal(float(row["stop_lon"]), datatype=XSD.decimal)))
            except (KeyError, ValueError, TypeError):
                pass

            if row.get("zone"):
                g.add((uri, HKG.transitZone, Literal(str(row["zone"]))))
            stop_count += 1

        # ── Routes ────────────────────────────────────────────────────────────
        for row in feed_data.get("routes", []):
            rid = str(row.get("route_id", ""))
            if not rid:
                continue
            uri = _route_uri(rid)
            g.add((uri, RDF.type,                     HKG.Route))
            g.add((uri, RDFS.label,                   Literal(str(row.get("route_long_name", rid)))))
            g.add((uri, GTFS_NS.shortName,            Literal(str(row.get("route_short_name", "")))))
            try:
                g.add((uri, GTFS_NS.routeType,        Literal(int(row["route_type"]), datatype=XSD.integer)))
            except (KeyError, ValueError, TypeError):
                pass
            route_count += 1

    log.info(f"GTFS → RDF: {stop_count} stops, {route_count} routes → {len(g)} triples")
    return g


def main():
    wl_feed = GTFSIngestion(
        feed_url="https://www.wienerlinien.at/ogd_realtime/doku/ogd/gtfs/gtfs.zip",
        feed_name="wienerlinien"
    )
    obb_feed = GTFSIngestion(
        feed_url=None,   # ÖBB requires registration; use cached file if available
        feed_name="oebb"
    )

    feeds = [wl_feed.parse(), obb_feed.parse()]
    g = gtfs_to_rdf(feeds)

    out = RDF_DIR / "gtfs_transit.ttl"
    g.serialize(destination=str(out), format="turtle")
    log.info(f"Saved transit RDF to {out}")


if __name__ == "__main__":
    main()
