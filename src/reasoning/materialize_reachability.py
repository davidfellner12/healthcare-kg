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
from config.settings import RDF_DIR, NAMESPACES, REACHABILITY_THRESHOLDS, VULNERABILITY_WEIGHTS
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


# ── Approximate district centroids ────────────────────────────────────────────
DISTRICT_CENTROIDS = {
    "AT-9-01": (48.2090, 16.3700), "AT-9-02": (48.2200, 16.4100),
    "AT-9-03": (48.2000, 16.3900), "AT-9-10": (48.1750, 16.3800),
    "AT-9-11": (48.1700, 16.4200), "AT-9-13": (48.1880, 16.2820),
    "AT-9-21": (48.2580, 16.3990), "AT-9-23": (48.1550, 16.3090),
    "AT-4-10": (48.3060, 14.2870), "AT-4-15": (48.1620, 14.0180),
    "AT-4-18": (48.2500, 14.6500), "AT-4-20": (48.5600, 13.9900),
    "AT-7-01": (47.2682, 11.3923), "AT-7-02": (47.2600, 11.4500),
    "AT-7-05": (47.5800, 12.1650), "AT-7-07": (46.8300, 12.7600),
}

FACILITY_DATA = [
    {"id": "KA001", "lat": 48.2196, "lon": 16.3564, "district": "AT-9-01"},
    {"id": "KA002", "lat": 48.1861, "lon": 16.2828, "district": "AT-9-13"},
    {"id": "KA003", "lat": 48.1600, "lon": 14.0300, "district": "AT-4-15"},
    {"id": "KA004", "lat": 48.3069, "lon": 14.2858, "district": "AT-4-10"},
    {"id": "KA005", "lat": 47.2682, "lon": 11.3923, "district": "AT-7-01"},
    {"id": "GP001", "lat": 48.2083, "lon": 16.3731, "district": "AT-9-01"},
    {"id": "GP002", "lat": 48.2206, "lon": 16.4134, "district": "AT-9-02"},
    {"id": "GP003", "lat": 48.3120, "lon": 14.2720, "district": "AT-4-10"},
    {"id": "GP004", "lat": 47.2650, "lon": 11.4000, "district": "AT-7-01"},
]

STOP_DATA = [
    {"id": "W001", "lat": 48.1851, "lon": 16.3761},
    {"id": "W002", "lat": 48.1969, "lon": 16.3381},
    {"id": "W003", "lat": 48.2047, "lon": 16.3861},
    {"id": "W004", "lat": 48.2361, "lon": 16.3600},
    {"id": "L001", "lat": 48.2907, "lon": 14.2920},
    {"id": "I001", "lat": 47.2631, "lon": 11.4006},
]


def materialize_reachability() -> Graph:
    """
    For each (facility, district) pair, compute approximate travel time
    and emit reachability triples for each threshold.
    """
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    triples_added = 0
    for fac in FACILITY_DATA:
        fac_uri = HKGR[f"facility/{fac['id']}"]
        # Find nearest stop
        nearest_stop = min(STOP_DATA,
                           key=lambda s: haversine_km(fac["lat"], fac["lon"], s["lat"], s["lon"]))

        for did, (dlat, dlon) in DISTRICT_CENTROIDS.items():
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

    log.info(f"Materialised {triples_added} reachability triples")
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
