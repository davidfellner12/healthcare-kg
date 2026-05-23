"""
src/ingestion/link_facilities_to_stops.py

Geo-links healthcare facilities to their nearest GTFS transit stop.
Writes hkg:nearestStop triples to data/rdf/facility_stop_links.ttl.

Uses the Haversine formula; walks up to MAX_STOP_DISTANCE_KM.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RAW_DIR, RDF_DIR, NAMESPACES, MAX_STOP_DISTANCE_KM
from src.utils.geo_utils import haversine_km

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HKG  = Namespace(NAMESPACES["hkg"])
HKGR = Namespace(NAMESPACES["hkgr"])

# Inline sample data (mirrors what healthcare_ingestion & gtfs_ingestion produce)
FACILITY_COORDS = [
    {"id": "KA001", "lat": 48.2196, "lon": 16.3564},
    {"id": "KA002", "lat": 48.1861, "lon": 16.2828},
    {"id": "KA003", "lat": 48.1600, "lon": 14.0300},
    {"id": "KA004", "lat": 48.3069, "lon": 14.2858},
    {"id": "KA005", "lat": 47.2682, "lon": 11.3923},
    {"id": "GP001", "lat": 48.2083, "lon": 16.3731},
    {"id": "GP002", "lat": 48.2206, "lon": 16.4134},
    {"id": "GP003", "lat": 48.3120, "lon": 14.2720},
    {"id": "GP004", "lat": 47.2650, "lon": 11.4000},
]

STOP_COORDS = [
    {"stop_id": "W001", "stop_lat": 48.1851, "stop_lon": 16.3761},
    {"stop_id": "W002", "stop_lat": 48.1969, "stop_lon": 16.3381},
    {"stop_id": "W003", "stop_lat": 48.2047, "stop_lon": 16.3861},
    {"stop_id": "W004", "stop_lat": 48.2361, "stop_lon": 16.3600},
    {"stop_id": "W005", "stop_lat": 48.2564, "stop_lon": 16.4007},
    {"stop_id": "L001", "stop_lat": 48.2907, "stop_lon": 14.2920},
    {"stop_id": "I001", "stop_lat": 47.2631, "stop_lon": 11.4006},
]


def find_nearest_stop(fac_lat: float, fac_lon: float, stops: list) -> tuple:
    """Return (stop_id, distance_km) of nearest stop."""
    best_id, best_dist = None, float("inf")
    for s in stops:
        d = haversine_km(fac_lat, fac_lon, float(s["stop_lat"]), float(s["stop_lon"]))
        if d < best_dist:
            best_dist = d
            best_id = s["stop_id"]
    return best_id, best_dist


def build_links(facilities: list, stops: list, max_km: float = MAX_STOP_DISTANCE_KM) -> Graph:
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    linked = 0
    for fac in facilities:
        fid = str(fac["id"])
        fac_uri = HKGR[f"facility/{fid}"]
        sid, dist = find_nearest_stop(float(fac["lat"]), float(fac["lon"]), stops)
        if dist <= max_km:
            stop_uri = HKGR[f"stop/{sid}"]
            g.add((fac_uri, HKG.nearestStop,         stop_uri))
            g.add((fac_uri, HKG.nearestStopDistKm,   Literal(round(dist, 4), datatype=XSD.decimal)))
            linked += 1
            log.debug(f"  {fid} → stop {sid} ({dist:.3f} km)")
        else:
            log.warning(f"  {fid}: nearest stop {sid} is {dist:.2f} km — beyond threshold")

    log.info(f"Linked {linked}/{len(facilities)} facilities to transit stops")
    return g


def main():
    # Try to read from processed CSVs if they exist, else use sample data
    fac_csv  = RAW_DIR / "healthcare_facilities.csv"
    stop_csv = RAW_DIR / "gtfs_stops.csv"

    if fac_csv.exists():
        facilities = pd.read_csv(fac_csv)[["id", "lat", "lon"]].dropna().to_dict("records")
    else:
        facilities = FACILITY_COORDS

    if stop_csv.exists():
        stops = pd.read_csv(stop_csv).to_dict("records")
    else:
        stops = STOP_COORDS

    g   = build_links(facilities, stops)
    out = RDF_DIR / "facility_stop_links.ttl"
    g.serialize(destination=str(out), format="turtle")
    log.info(f"Saved facility–stop links to {out}")


if __name__ == "__main__":
    main()
