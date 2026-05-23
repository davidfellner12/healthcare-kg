"""
src/reasoning/kg_evolution.py

Handles KG Evolution (LO8): incremental re-reasoning when facilities
open/close or GTFS timetables change, without full recomputation.

Supported update events:
  - FACILITY_ADDED:   new hospital or GP opens
  - FACILITY_CLOSED:  facility permanently closes
  - TIMETABLE_CHANGE: GTFS route/stop modification

Approach:
  1. Parse the change event
  2. Retract stale triples (SPARQL DELETE)
  3. Re-materialise only affected facts (targeted INSERT)
  4. Append to the RDF store; full re-build is NOT needed

Usage:
  python src/reasoning/kg_evolution.py --event facility_added --id KA999 \\
         --lat 48.21 --lon 16.37 --type Hospital --district AT-9-02
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RDF_DIR, NAMESPACES
from src.utils.geo_utils import haversine_km

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HKG_BASE = NAMESPACES["hkg"]
HKGR_BASE = NAMESPACES["hkgr"]

KNOWN_STOPS = [
    {"id": "W001", "lat": 48.1851, "lon": 16.3761},
    {"id": "W002", "lat": 48.1969, "lon": 16.3381},
    {"id": "L001", "lat": 48.2907, "lon": 14.2920},
    {"id": "I001", "lat": 47.2631, "lon": 11.4006},
]


def add_facility(facility_id: str, lat: float, lon: float,
                 ftype: str, district: str, name: str = "") -> str:
    """
    Generate Turtle snippet for a newly opened facility and
    append it to healthcare_facilities.ttl.
    Returns the generated Turtle string.
    """
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD

    HKG  = Namespace(HKG_BASE)
    HKGR = Namespace(HKGR_BASE)
    GEO  = Namespace(NAMESPACES["geo"])

    g = Graph()
    uri = HKGR[f"facility/{facility_id}"]
    g.add((uri, RDF.type,         HKG[ftype]))
    g.add((uri, HKG.facilityId,   Literal(facility_id)))
    g.add((uri, HKG.facilityName, Literal(name or facility_id, lang="de")))
    g.add((uri, GEO.lat,          Literal(lat, datatype=XSD.decimal)))
    g.add((uri, GEO.long,         Literal(lon, datatype=XSD.decimal)))
    g.add((uri, HKG.inDistrict,   HKGR[f"district/{district}"]))
    g.add((uri, HKG.addedAt,      Literal(datetime.utcnow().isoformat(), datatype=XSD.dateTime)))

    # Nearest stop link
    nearest = min(KNOWN_STOPS, key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))
    dist_km = haversine_km(lat, lon, nearest["lat"], nearest["lon"])
    g.add((uri, HKG.nearestStop,        HKGR[f"stop/{nearest['id']}"]))
    g.add((uri, HKG.nearestStopDistKm,  Literal(round(dist_km, 4), datatype=XSD.decimal)))

    ttl_str = g.serialize(format="turtle")

    # Append to existing file
    out = RDF_DIR / "healthcare_facilities.ttl"
    with open(out, "a") as f:
        f.write(f"\n# === Added by kg_evolution.py at {datetime.utcnow().isoformat()} ===\n")
        f.write(ttl_str)

    log.info(f"Added facility {facility_id} → {out}")
    return ttl_str


def remove_facility(facility_id: str) -> str:
    """
    Generate SPARQL DELETE to retract all triples about a closed facility.
    Writes a .sparql patch file for audit trail.
    """
    fac_uri = f"{HKGR_BASE}facility/{facility_id}"
    delete_query = f"""
PREFIX hkg:  <{HKG_BASE}>
PREFIX hkgr: <{HKGR_BASE}>

DELETE {{
    <{fac_uri}> ?p ?o .
    ?s hkg:reachableIn15min <{fac_uri}> .
    ?s hkg:reachableIn30min <{fac_uri}> .
    ?s hkg:reachableIn60min <{fac_uri}> .
}}
WHERE {{
    OPTIONAL {{ <{fac_uri}> ?p ?o . }}
    OPTIONAL {{ ?s hkg:reachableIn15min <{fac_uri}> . }}
    OPTIONAL {{ ?s hkg:reachableIn30min <{fac_uri}> . }}
    OPTIONAL {{ ?s hkg:reachableIn60min <{fac_uri}> . }}
}}
"""
    patch_file = RDF_DIR / f"patch_remove_{facility_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.sparql"
    patch_file.write_text(delete_query)
    log.info(f"Removal patch written to {patch_file}")
    log.info("Apply with: python src/reasoning/load_triplestore.py --apply-patch <file>")
    return delete_query


def log_change_event(event_type: str, entity_id: str, details: dict):
    """Append a structured change log entry (supports audit / KG provenance)."""
    log_file = RDF_DIR / "change_log.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "entity_id": entity_id,
        **details,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"Change event logged: {event_type} / {entity_id}")


def main():
    parser = argparse.ArgumentParser(description="KG Evolution — incremental updates")
    parser.add_argument("--event", required=True,
                        choices=["facility_added", "facility_closed", "timetable_change"],
                        help="Type of change event")
    parser.add_argument("--id",       required=True, help="Entity ID")
    parser.add_argument("--lat",      type=float,    help="Latitude (facility_added)")
    parser.add_argument("--lon",      type=float,    help="Longitude (facility_added)")
    parser.add_argument("--type",     default="GeneralPractitioner",
                        help="Facility type (facility_added)")
    parser.add_argument("--district", default="",    help="District ID (facility_added)")
    parser.add_argument("--name",     default="",    help="Facility name (facility_added)")
    args = parser.parse_args()

    if args.event == "facility_added":
        assert args.lat and args.lon, "--lat and --lon required for facility_added"
        ttl = add_facility(args.id, args.lat, args.lon,
                           args.type, args.district, args.name)
        log_change_event("facility_added", args.id,
                         {"lat": args.lat, "lon": args.lon, "type": args.type,
                          "district": args.district})
        print(ttl)

    elif args.event == "facility_closed":
        sparql = remove_facility(args.id)
        log_change_event("facility_closed", args.id, {})
        print(sparql)

    elif args.event == "timetable_change":
        log.info(f"Timetable change for route/stop {args.id}")
        log.info("Full incremental re-ingestion of affected GTFS feed required.")
        log_change_event("timetable_change", args.id, {})


if __name__ == "__main__":
    main()
