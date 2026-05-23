"""
src/ingestion/demographics_ingestion.py

Loads district-level demographic data from Statistics Austria and
converts it to RDF.

Variables per district:
  - Total population
  - Age groups (< 5, 5–14, 15–64, 65+)
  - Car ownership rate
  - Income proxy (median household income)
  - Existing GP count (for GP-per-1000 calculation)

Output: data/rdf/demographics.ttl
"""

import sys
import logging
import requests
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RAW_DIR, RDF_DIR, NAMESPACES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HKG  = Namespace(NAMESPACES["hkg"])
HKGR = Namespace(NAMESPACES["hkgr"])
DCT  = Namespace(NAMESPACES["dct"])

# ── Synthetic district data (replace with real Statistik Austria CSV) ─────────
SAMPLE_DISTRICTS = [
    # Wien districts (Bezirke)
    {"district_id": "AT-9-01", "name": "Wien 1., Innere Stadt",   "state": "Wien",          "population": 16500,  "age_0_4_pct": 3.2,  "age_65plus_pct": 14.1, "gp_count": 18,  "car_ownership_pct": 28.0, "median_income": 38000},
    {"district_id": "AT-9-02", "name": "Wien 2., Leopoldstadt",   "state": "Wien",          "population": 105000, "age_0_4_pct": 5.8,  "age_65plus_pct": 15.2, "gp_count": 85,  "car_ownership_pct": 32.0, "median_income": 27000},
    {"district_id": "AT-9-03", "name": "Wien 3., Landstraße",     "state": "Wien",          "population": 93000,  "age_0_4_pct": 4.9,  "age_65plus_pct": 17.8, "gp_count": 72,  "car_ownership_pct": 35.0, "median_income": 30000},
    {"district_id": "AT-9-10", "name": "Wien 10., Favoriten",     "state": "Wien",          "population": 215000, "age_0_4_pct": 7.2,  "age_65plus_pct": 13.5, "gp_count": 120, "car_ownership_pct": 45.0, "median_income": 22000},
    {"district_id": "AT-9-11", "name": "Wien 11., Simmering",     "state": "Wien",          "population": 105000, "age_0_4_pct": 6.8,  "age_65plus_pct": 14.0, "gp_count": 55,  "car_ownership_pct": 55.0, "median_income": 24000},
    {"district_id": "AT-9-13", "name": "Wien 13., Hietzing",      "state": "Wien",          "population": 55000,  "age_0_4_pct": 4.1,  "age_65plus_pct": 22.3, "gp_count": 40,  "car_ownership_pct": 68.0, "median_income": 42000},
    {"district_id": "AT-9-21", "name": "Wien 21., Floridsdorf",   "state": "Wien",          "population": 165000, "age_0_4_pct": 6.5,  "age_65plus_pct": 14.8, "gp_count": 90,  "car_ownership_pct": 62.0, "median_income": 28000},
    {"district_id": "AT-9-23", "name": "Wien 23., Liesing",       "state": "Wien",          "population": 105000, "age_0_4_pct": 5.9,  "age_65plus_pct": 18.5, "gp_count": 62,  "car_ownership_pct": 71.0, "median_income": 33000},
    # Oberösterreich
    {"district_id": "AT-4-10", "name": "Linz",                    "state": "Oberösterreich","population": 210000, "age_0_4_pct": 5.3,  "age_65plus_pct": 17.2, "gp_count": 140, "car_ownership_pct": 58.0, "median_income": 30000},
    {"district_id": "AT-4-15", "name": "Wels-Land",               "state": "Oberösterreich","population": 73000,  "age_0_4_pct": 5.8,  "age_65plus_pct": 16.0, "gp_count": 38,  "car_ownership_pct": 78.0, "median_income": 26000},
    {"district_id": "AT-4-18", "name": "Perg",                    "state": "Oberösterreich","population": 67000,  "age_0_4_pct": 6.1,  "age_65plus_pct": 18.3, "gp_count": 28,  "car_ownership_pct": 85.0, "median_income": 24000},
    {"district_id": "AT-4-20", "name": "Rohrbach",                "state": "Oberösterreich","population": 57000,  "age_0_4_pct": 5.4,  "age_65plus_pct": 21.0, "gp_count": 18,  "car_ownership_pct": 90.0, "median_income": 22000},
    # Tirol
    {"district_id": "AT-7-01", "name": "Innsbruck-Stadt",         "state": "Tirol",         "population": 135000, "age_0_4_pct": 4.8,  "age_65plus_pct": 16.5, "gp_count": 92,  "car_ownership_pct": 52.0, "median_income": 32000},
    {"district_id": "AT-7-02", "name": "Innsbruck-Land",          "state": "Tirol",         "population": 176000, "age_0_4_pct": 6.0,  "age_65plus_pct": 16.9, "gp_count": 95,  "car_ownership_pct": 76.0, "median_income": 30000},
    {"district_id": "AT-7-05", "name": "Kufstein",                "state": "Tirol",         "population": 95000,  "age_0_4_pct": 6.3,  "age_65plus_pct": 17.1, "gp_count": 45,  "car_ownership_pct": 80.0, "median_income": 28000},
    {"district_id": "AT-7-07", "name": "Lienz",                   "state": "Tirol",         "population": 48000,  "age_0_4_pct": 5.2,  "age_65plus_pct": 23.5, "gp_count": 18,  "car_ownership_pct": 87.0, "median_income": 21000},
]


def fetch_demographics() -> pd.DataFrame:
    csv_cache = RAW_DIR / "demographics.csv"
    if csv_cache.exists():
        log.info(f"Loading cached demographics from {csv_cache}")
        return pd.read_csv(csv_cache)

    log.info("Using sample demographics data (Statistik Austria download requires account)")
    df = pd.DataFrame(SAMPLE_DISTRICTS)
    df.to_csv(csv_cache, index=False)
    return df


def _district_uri(district_id: str) -> URIRef:
    return HKGR[f"district/{district_id}"]


def demographics_to_rdf(df: pd.DataFrame) -> Graph:
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    # ── Ontology classes + properties ─────────────────────────────────────────
    g.add((HKG.District,       RDF.type,    OWL.Class))
    g.add((HKG.District,       RDFS.label,  Literal("District", lang="en")))

    datatype_props = {
        "districtId":        ("District identifier", XSD.string),
        "districtName":      ("District name", XSD.string),
        "stateName":         ("Federal state", XSD.string),
        "population":        ("Total population", XSD.integer),
        "age0to4pct":        ("Population aged 0–4 (%)", XSD.decimal),
        "age65plusPct":      ("Population aged 65+ (%)", XSD.decimal),
        "gpCount":           ("Number of general practitioners", XSD.integer),
        "gpPer1000":         ("GPs per 1,000 residents", XSD.decimal),
        "carOwnershipPct":   ("Car ownership rate (%)", XSD.decimal),
        "medianIncome":      ("Median household income (EUR)", XSD.decimal),
        "vulnerabilityScore":("Computed vulnerability score", XSD.decimal),
    }
    for prop, (label, _) in datatype_props.items():
        g.add((HKG[prop], RDF.type,    OWL.DatatypeProperty))
        g.add((HKG[prop], RDFS.label,  Literal(label, lang="en")))
        g.add((HKG[prop], RDFS.domain, HKG.District))

    # ── Individuals ───────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        did = str(row["district_id"])
        uri = _district_uri(did)

        g.add((uri, RDF.type,            HKG.District))
        g.add((uri, HKG.districtId,      Literal(did)))
        g.add((uri, HKG.districtName,    Literal(str(row["name"]), lang="de")))
        g.add((uri, RDFS.label,          Literal(str(row["name"]), lang="de")))
        g.add((uri, HKG.stateName,       Literal(str(row["state"]), lang="de")))
        g.add((uri, HKG.population,      Literal(int(row["population"]), datatype=XSD.integer)))
        g.add((uri, HKG.age0to4pct,      Literal(float(row["age_0_4_pct"]), datatype=XSD.decimal)))
        g.add((uri, HKG.age65plusPct,    Literal(float(row["age_65plus_pct"]), datatype=XSD.decimal)))
        g.add((uri, HKG.gpCount,         Literal(int(row["gp_count"]), datatype=XSD.integer)))
        g.add((uri, HKG.carOwnershipPct, Literal(float(row["car_ownership_pct"]), datatype=XSD.decimal)))
        g.add((uri, HKG.medianIncome,    Literal(float(row["median_income"]), datatype=XSD.decimal)))

        # Computed: GP-per-1000
        gp_per_1000 = round(int(row["gp_count"]) / int(row["population"]) * 1000, 4)
        g.add((uri, HKG.gpPer1000, Literal(gp_per_1000, datatype=XSD.decimal)))

    log.info(f"Generated {len(g)} demographic RDF triples for {len(df)} districts")
    return g


def main():
    df = fetch_demographics()
    g  = demographics_to_rdf(df)
    out = RDF_DIR / "demographics.ttl"
    g.serialize(destination=str(out), format="turtle")
    log.info(f"Saved demographics RDF to {out}")


if __name__ == "__main__":
    main()
