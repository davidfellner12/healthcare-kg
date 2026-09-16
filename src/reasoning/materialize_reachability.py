"""
src/reasoning/materialize_reachability.py

Materialises transit-based reachability facts and vulnerability scores
into the Knowledge Graph using SPARQL INSERT rules.

Because full GTFS timetable reasoning is expensive, we pre-compute
approximate travel times using:
  1. Walking time from facility → nearest stop (Haversine + 4.5 km/h)
  2. Estimated transit time between stops (distance / average speed)
  3. Walking time from destination stop → district centroid

This is the "batch pre-computation" described in LO5/LO6.

Outputs:
  data/rdf/reachability.ttl
  data/rdf/vulnerability.ttl
"""

import sys
import math
import logging
from pathlib import Path
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RDF_DIR, RAW_DIR, NAMESPACES, REACHABILITY_THRESHOLDS, VULNERABILITY_WEIGHTS
from src.utils.rdf_utils import load_graph, sparql_query
from src.utils.geo_utils import haversine_km, walking_time_minutes

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HKG  = Namespace(NAMESPACES["hkg"])
HKGR = Namespace(NAMESPACES["hkgr"])

TRANSIT_SPEED_KMH = 30.0   # average urban transit speed


def compute_transit_time(fac_lat, fac_lon, stop_lat, stop_lon,
                          dest_lat, dest_lon) -> float:
    """
    Approximate total door-to-stop-to-district travel time (minutes).
    walk_to_stop + transit + (no walk from stop modeled here — 
    district centroid used as proxy for stop).
    """
    walk_mins    = walking_time_minutes(haversine_km(fac_lat, fac_lon, stop_lat, stop_lon))
    transit_km   = haversine_km(stop_lat, stop_lon, dest_lat, dest_lon)
    transit_mins = (transit_km / TRANSIT_SPEED_KMH) * 60
    return walk_mins + transit_mins


# ── Small hardcoded fallbacks (only used if the CSVs from the ingestion ------
# stage are not present, e.g. running this module standalone before any
# ingestion has run) -----------------------------------------------------------
_FALLBACK_DISTRICT_CENTROIDS = {
    "AT-9-01": (48.2082, 16.3738), "AT-9-13": (48.1833, 16.2833),
    "AT-4-01": (48.3069, 14.2858), "AT-7-01": (47.2682, 11.3923),
}
_FALLBACK_FACILITY_DATA = [
    {"id": "KA001", "lat": 48.2196, "lon": 16.3564, "district": "AT-9-01"},
    {"id": "KA002", "lat": 48.1861, "lon": 16.2828, "district": "AT-9-13"},
]
_FALLBACK_STOP_DATA = [
    {"id": "W001", "lat": 48.1851, "lon": 16.3761},
    {"id": "L001", "lat": 48.2907, "lon": 14.2920},
    {"id": "I001", "lat": 47.2631, "lon": 11.4006},
]


def _load_district_centroids() -> dict:
    """district_id -> (lat, lon), sourced from demographics.csv when present."""
    import pandas as pd
    csv = RAW_DIR / "demographics.csv"
    if not csv.exists():
        log.warning("demographics.csv not found; using tiny fallback centroid set")
        return _FALLBACK_DISTRICT_CENTROIDS
    df = pd.read_csv(csv)
    return {str(r["district_id"]): (float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()}


def _load_facility_data() -> list:
    """[{id, lat, lon, district}], sourced from healthcare_facilities.csv when present."""
    import pandas as pd
    csv = RAW_DIR / "healthcare_facilities.csv"
    if not csv.exists():
        log.warning("healthcare_facilities.csv not found; using tiny fallback facility set")
        return _FALLBACK_FACILITY_DATA
    df = pd.read_csv(csv)[["id", "lat", "lon", "district"]].dropna(subset=["lat", "lon", "district"])
    return df.to_dict("records")


def _load_stop_data() -> list:
    """[{id, lat, lon}], sourced from gtfs_stops.csv (real WL stops + illustrative
    regional hubs, see gtfs_ingestion.py) when present."""
    import pandas as pd
    csv = RAW_DIR / "gtfs_stops.csv"
    if not csv.exists():
        log.warning("gtfs_stops.csv not found; using tiny fallback stop set")
        return _FALLBACK_STOP_DATA
    df = pd.read_csv(csv)
    return [{"id": r["stop_id"], "lat": float(r["stop_lat"]), "lon": float(r["stop_lon"])}
            for _, r in df.iterrows()]


def materialize_reachability() -> Graph:
    """
    For each (facility, district) pair, compute approximate travel time
    and emit reachability triples for each threshold.
    """
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    facility_data = _load_facility_data()
    stop_data = _load_stop_data()
    district_centroids = _load_district_centroids()
    log.info(f"Reachability inputs: {len(facility_data)} facilities, {len(stop_data)} stops, "
             f"{len(district_centroids)} district centroids")

    triples_added = 0
    for fac in facility_data:
        fac_uri = HKGR[f"facility/{fac['id']}"]
        # Find nearest stop
        nearest_stop = min(stop_data,
                           key=lambda s: haversine_km(fac["lat"], fac["lon"], s["lat"], s["lon"]))

        for did, (dlat, dlon) in district_centroids.items():
            total_mins = compute_transit_time(
                fac["lat"], fac["lon"],
                nearest_stop["lat"], nearest_stop["lon"],
                dlat, dlon
            )
            district_uri = HKGR[f"district/{did}"]

            for threshold in REACHABILITY_THRESHOLDS:
                if total_mins <= threshold:
                    prop = HKG[f"reachableIn{threshold}min"]
                    g.add((fac_uri, prop, district_uri))
                    g.add((fac_uri, HKG.travelTimeToDistrict,
                           Literal(round(total_mins, 2), datatype=XSD.decimal)))
                    triples_added += 1

    # NOTE: triples_added counts satisfied (facility, threshold) pairs, but
    # each one emits *two* RDF triples (the reachableInXmin edge and a
    # travelTimeToDistrict literal), so the true graph size is roughly
    # double this -- log both rather than the undercounted figure alone.
    log.info(f"Materialised {triples_added} (facility, threshold) reachability facts "
             f"-> {len(g)} RDF triples")
    return g


def materialize_vulnerability() -> Graph:
    """
    Compute age-weighted vulnerability scores per district and
    emit risk classification triples.
    """
    import pandas as pd
    from config.settings import RAW_DIR

    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    csv = RAW_DIR / "demographics.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        from src.ingestion.demographics_ingestion import SAMPLE_DISTRICTS
        df = pd.DataFrame(SAMPLE_DISTRICTS)

    for _, row in df.iterrows():
        did  = str(row["district_id"])
        duri = HKGR[f"district/{did}"]
        pop  = int(row["population"])

        # Normalised sub-scores (0–1)
        elder_score  = min(float(row["age_65plus_pct"]) / 30.0, 1.0)
        young_score  = min(float(row["age_0_4_pct"])    / 10.0, 1.0)
        nocar_score  = min(float(row["car_ownership_pct"]) / 100.0, 1.0)  # invert below
        income_score = 1.0 - min(float(row["median_income"]) / 50000.0, 1.0)
        gp_score     = 1.0 - min(float(row["gp_count"]) / (pop / 1000.0 * 1.2), 1.0)  # shortage proxy
        nocar_score  = 1.0 - nocar_score   # high car ownership = LOW vulnerability

        vuln = (
            VULNERABILITY_WEIGHTS["age_over_65_pct"]  * elder_score
          + VULNERABILITY_WEIGHTS["age_under_5_pct"]  * young_score
          + VULNERABILITY_WEIGHTS["no_car_pct"]       * nocar_score
          + VULNERABILITY_WEIGHTS["low_income_pct"]   * income_score
        )
        vuln = round(min(vuln, 1.0), 4)

        g.add((duri, HKG.vulnerabilityScore, Literal(vuln, datatype=XSD.decimal)))

        # GP deficit flag
        gp_per_1000 = int(row["gp_count"]) / pop * 1000
        g.add((duri, HKG.gpPer1000, Literal(round(gp_per_1000, 4), datatype=XSD.decimal)))
        if gp_per_1000 < 0.6:
            g.add((duri, HKG.gpDeficit, Literal(True, datatype=XSD.boolean)))

        # Risk classification
        elder_pct = float(row["age_65plus_pct"])
        if vuln >= 0.6 or elder_pct > 20.0:
            risk = HKG.HighRisk
        elif vuln >= 0.35 or elder_pct > 18.0:
            risk = HKG.MediumRisk
        else:
            risk = HKG.LowRisk
        g.add((duri, HKG.accessRisk, risk))

        log.debug(f"  {did}: vuln={vuln:.3f}  gpPer1000={gp_per_1000:.3f}  risk={risk.split('#')[-1]}")

    log.info(f"Materialised vulnerability for {len(df)} districts → {len(g)} triples")
    return g


def main():
    g_reach = materialize_reachability()
    out1 = RDF_DIR / "reachability.ttl"
    g_reach.serialize(destination=str(out1), format="turtle")
    log.info(f"Saved reachability RDF to {out1}")

    g_vuln = materialize_vulnerability()
    out2 = RDF_DIR / "vulnerability.ttl"
    g_vuln.serialize(destination=str(out2), format="turtle")
    log.info(f"Saved vulnerability RDF to {out2}")


if __name__ == "__main__":
    main()
