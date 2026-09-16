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
  - Approximate geographic centroid (lat/lon)

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
GEO  = Namespace(NAMESPACES["geo"])

# ── District data ──────────────────────────────────────────────────────────
# Self-constructed / illustrative: district names, official Vienna Bezirk
# numbering, and the OÖ/Tirol political-district (Bezirk) names are real
# Austrian administrative geography; population figures are realistic
# order-of-magnitude estimates (not exact Statistik Austria table extracts),
# and the demographic/economic indicators are internally-consistent synthetic
# values generated per district from a small set of region-type profiles
# (urban core / dense / outer / suburban / regional city / rural / alpine
# rural) -- see scripts/gen_districts.py used to derive them. Total modelled
# population (~4.16M) is deliberately close to the real combined population
# of Wien + Oberösterreich + Tirol (~4.2M) as a sanity check, but no single
# row should be read as an official statistic.
SAMPLE_DISTRICTS = [
    # ---- Wien: all 23 Gemeindebezirke (real official numbering 1-23) ------
    {"district_id": "AT-9-01", "name": "Wien 1., Innere Stadt",           "state": "Wien", "population": 16700,  "age_0_4_pct": 3.0, "age_65plus_pct": 14.3, "gp_count": 31,  "car_ownership_pct": 26.6, "median_income": 43600, "lat": 48.2082, "lon": 16.3738},
    {"district_id": "AT-9-02", "name": "Wien 2., Leopoldstadt",          "state": "Wien", "population": 105000, "age_0_4_pct": 6.0, "age_65plus_pct": 18.1, "gp_count": 94,  "car_ownership_pct": 32.7, "median_income": 24600, "lat": 48.2145, "lon": 16.3906},
    {"district_id": "AT-9-03", "name": "Wien 3., Landstraße",            "state": "Wien", "population": 94000,  "age_0_4_pct": 5.9, "age_65plus_pct": 17.2, "gp_count": 56,  "car_ownership_pct": 28.3, "median_income": 26600, "lat": 48.1980, "lon": 16.3900},
    {"district_id": "AT-9-04", "name": "Wien 4., Wieden",                "state": "Wien", "population": 32500,  "age_0_4_pct": 3.4, "age_65plus_pct": 16.2, "gp_count": 46,  "car_ownership_pct": 21.7, "median_income": 38300, "lat": 48.1929, "lon": 16.3672},
    {"district_id": "AT-9-05", "name": "Wien 5., Margareten",            "state": "Wien", "population": 56000,  "age_0_4_pct": 6.1, "age_65plus_pct": 17.3, "gp_count": 53,  "car_ownership_pct": 42.0, "median_income": 26500, "lat": 48.1877, "lon": 16.3572},
    {"district_id": "AT-9-06", "name": "Wien 6., Mariahilf",             "state": "Wien", "population": 30500,  "age_0_4_pct": 4.2, "age_65plus_pct": 16.5, "gp_count": 49,  "car_ownership_pct": 21.2, "median_income": 38800, "lat": 48.1966, "lon": 16.3517},
    {"district_id": "AT-9-07", "name": "Wien 7., Neubau",                "state": "Wien", "population": 32000,  "age_0_4_pct": 3.3, "age_65plus_pct": 14.7, "gp_count": 65,  "car_ownership_pct": 20.7, "median_income": 43100, "lat": 48.2020, "lon": 16.3467},
    {"district_id": "AT-9-08", "name": "Wien 8., Josefstadt",            "state": "Wien", "population": 25000,  "age_0_4_pct": 3.8, "age_65plus_pct": 13.4, "gp_count": 50,  "car_ownership_pct": 23.4, "median_income": 43900, "lat": 48.2100, "lon": 16.3467},
    {"district_id": "AT-9-09", "name": "Wien 9., Alsergrund",            "state": "Wien", "population": 45000,  "age_0_4_pct": 5.1, "age_65plus_pct": 16.8, "gp_count": 38,  "car_ownership_pct": 33.2, "median_income": 26000, "lat": 48.2233, "lon": 16.3550},
    {"district_id": "AT-9-10", "name": "Wien 10., Favoriten",            "state": "Wien", "population": 211000, "age_0_4_pct": 6.2, "age_65plus_pct": 15.3, "gp_count": 115, "car_ownership_pct": 42.3, "median_income": 24900, "lat": 48.1700, "lon": 16.3800},
    {"district_id": "AT-9-11", "name": "Wien 11., Simmering",            "state": "Wien", "population": 108000, "age_0_4_pct": 6.3, "age_65plus_pct": 15.3, "gp_count": 51,  "car_ownership_pct": 43.4, "median_income": 22800, "lat": 48.1667, "lon": 16.4333},
    {"district_id": "AT-9-12", "name": "Wien 12., Meidling",             "state": "Wien", "population": 96000,  "age_0_4_pct": 4.6, "age_65plus_pct": 16.1, "gp_count": 71,  "car_ownership_pct": 31.5, "median_income": 28400, "lat": 48.1783, "lon": 16.3350},
    {"district_id": "AT-9-13", "name": "Wien 13., Hietzing",             "state": "Wien", "population": 54000,  "age_0_4_pct": 5.7, "age_65plus_pct": 18.8, "gp_count": 46,  "car_ownership_pct": 58.9, "median_income": 35300, "lat": 48.1833, "lon": 16.2833},
    {"district_id": "AT-9-14", "name": "Wien 14., Penzing",              "state": "Wien", "population": 106000, "age_0_4_pct": 5.2, "age_65plus_pct": 19.2, "gp_count": 89,  "car_ownership_pct": 60.3, "median_income": 31100, "lat": 48.2000, "lon": 16.2833},
    {"district_id": "AT-9-15", "name": "Wien 15., Rudolfsheim-Fünfhaus", "state": "Wien", "population": 80000,  "age_0_4_pct": 6.3, "age_65plus_pct": 14.2, "gp_count": 66,  "car_ownership_pct": 30.0, "median_income": 29900, "lat": 48.1917, "lon": 16.3333},
    {"district_id": "AT-9-16", "name": "Wien 16., Ottakring",            "state": "Wien", "population": 103000, "age_0_4_pct": 6.4, "age_65plus_pct": 14.8, "gp_count": 97,  "car_ownership_pct": 37.3, "median_income": 28000, "lat": 48.2136, "lon": 16.3047},
    {"district_id": "AT-9-17", "name": "Wien 17., Hernals",              "state": "Wien", "population": 57000,  "age_0_4_pct": 5.4, "age_65plus_pct": 21.2, "gp_count": 49,  "car_ownership_pct": 63.7, "median_income": 31800, "lat": 48.2333, "lon": 16.3167},
    {"district_id": "AT-9-18", "name": "Wien 18., Währing",              "state": "Wien", "population": 50000,  "age_0_4_pct": 4.8, "age_65plus_pct": 21.2, "gp_count": 49,  "car_ownership_pct": 69.6, "median_income": 38200, "lat": 48.2350, "lon": 16.3333},
    {"district_id": "AT-9-19", "name": "Wien 19., Döbling",              "state": "Wien", "population": 72000,  "age_0_4_pct": 4.0, "age_65plus_pct": 20.8, "gp_count": 52,  "car_ownership_pct": 72.8, "median_income": 29200, "lat": 48.2500, "lon": 16.3450},
    {"district_id": "AT-9-20", "name": "Wien 20., Brigittenau",          "state": "Wien", "population": 89000,  "age_0_4_pct": 6.3, "age_65plus_pct": 18.6, "gp_count": 57,  "car_ownership_pct": 31.2, "median_income": 30700, "lat": 48.2333, "lon": 16.3833},
    {"district_id": "AT-9-21", "name": "Wien 21., Floridsdorf",          "state": "Wien", "population": 168000, "age_0_4_pct": 6.3, "age_65plus_pct": 15.3, "gp_count": 84,  "car_ownership_pct": 49.9, "median_income": 21000, "lat": 48.2564, "lon": 16.4007},
    {"district_id": "AT-9-22", "name": "Wien 22., Donaustadt",           "state": "Wien", "population": 197000, "age_0_4_pct": 7.4, "age_65plus_pct": 14.3, "gp_count": 115, "car_ownership_pct": 46.8, "median_income": 23100, "lat": 48.2333, "lon": 16.4667},
    {"district_id": "AT-9-23", "name": "Wien 23., Liesing",              "state": "Wien", "population": 106000, "age_0_4_pct": 4.2, "age_65plus_pct": 18.6, "gp_count": 94,  "car_ownership_pct": 65.4, "median_income": 32400, "lat": 48.1333, "lon": 16.2833},

    # ---- Oberösterreich: 15 Bezirke + 3 Statutarstädte (18 total) --------
    {"district_id": "AT-4-01", "name": "Linz",                    "state": "Oberösterreich", "population": 208000, "age_0_4_pct": 5.0, "age_65plus_pct": 19.0, "gp_count": 266, "car_ownership_pct": 52.7, "median_income": 28700, "lat": 48.3069, "lon": 14.2858},
    {"district_id": "AT-4-02", "name": "Wels",                    "state": "Oberösterreich", "population": 63000,  "age_0_4_pct": 5.6, "age_65plus_pct": 18.5, "gp_count": 85,  "car_ownership_pct": 54.5, "median_income": 30400, "lat": 48.1575, "lon": 14.0289},
    {"district_id": "AT-4-03", "name": "Steyr",                   "state": "Oberösterreich", "population": 38000,  "age_0_4_pct": 5.6, "age_65plus_pct": 18.6, "gp_count": 50,  "car_ownership_pct": 53.6, "median_income": 29100, "lat": 48.0396, "lon": 14.4212},
    {"district_id": "AT-4-04", "name": "Linz-Land",               "state": "Oberösterreich", "population": 145000, "age_0_4_pct": 4.3, "age_65plus_pct": 20.3, "gp_count": 114, "car_ownership_pct": 59.6, "median_income": 38500, "lat": 48.2200, "lon": 14.2500},
    {"district_id": "AT-4-05", "name": "Wels-Land",               "state": "Oberösterreich", "population": 74000,  "age_0_4_pct": 5.6, "age_65plus_pct": 19.3, "gp_count": 40,  "car_ownership_pct": 76.9, "median_income": 27000, "lat": 48.1500, "lon": 13.9700},
    {"district_id": "AT-4-06", "name": "Vöcklabruck",             "state": "Oberösterreich", "population": 132000, "age_0_4_pct": 4.8, "age_65plus_pct": 21.4, "gp_count": 93,  "car_ownership_pct": 78.6, "median_income": 26600, "lat": 48.0089, "lon": 13.6553},
    {"district_id": "AT-4-07", "name": "Braunau am Inn",          "state": "Oberösterreich", "population": 101000, "age_0_4_pct": 4.5, "age_65plus_pct": 19.4, "gp_count": 72,  "car_ownership_pct": 79.8, "median_income": 25400, "lat": 48.2569, "lon": 13.0364},
    {"district_id": "AT-4-08", "name": "Urfahr-Umgebung",         "state": "Oberösterreich", "population": 78000,  "age_0_4_pct": 4.4, "age_65plus_pct": 17.4, "gp_count": 72,  "car_ownership_pct": 66.8, "median_income": 41900, "lat": 48.3600, "lon": 14.2200},
    {"district_id": "AT-4-09", "name": "Gmunden",                 "state": "Oberösterreich", "population": 99000,  "age_0_4_pct": 4.5, "age_65plus_pct": 21.7, "gp_count": 32,  "car_ownership_pct": 88.6, "median_income": 28600, "lat": 47.9186, "lon": 13.7994},
    {"district_id": "AT-4-10", "name": "Ried im Innkreis",        "state": "Oberösterreich", "population": 60000,  "age_0_4_pct": 4.7, "age_65plus_pct": 22.1, "gp_count": 29,  "car_ownership_pct": 84.6, "median_income": 27000, "lat": 48.2100, "lon": 13.4890},
    {"district_id": "AT-4-11", "name": "Grieskirchen",            "state": "Oberösterreich", "population": 65000,  "age_0_4_pct": 4.6, "age_65plus_pct": 22.3, "gp_count": 34,  "car_ownership_pct": 76.6, "median_income": 24000, "lat": 48.2333, "lon": 13.8333},
    {"district_id": "AT-4-12", "name": "Steyr-Land",              "state": "Oberösterreich", "population": 60000,  "age_0_4_pct": 5.0, "age_65plus_pct": 20.7, "gp_count": 40,  "car_ownership_pct": 83.1, "median_income": 23400, "lat": 47.9800, "lon": 14.3800},
    {"district_id": "AT-4-13", "name": "Freistadt",               "state": "Oberösterreich", "population": 66000,  "age_0_4_pct": 5.6, "age_65plus_pct": 20.0, "gp_count": 35,  "car_ownership_pct": 79.6, "median_income": 27900, "lat": 48.5100, "lon": 14.5000},
    {"district_id": "AT-4-14", "name": "Perg",                    "state": "Oberösterreich", "population": 67000,  "age_0_4_pct": 5.0, "age_65plus_pct": 20.8, "gp_count": 50,  "car_ownership_pct": 81.7, "median_income": 25400, "lat": 48.2500, "lon": 14.6333},
    {"district_id": "AT-4-15", "name": "Rohrbach",                "state": "Oberösterreich", "population": 57000,  "age_0_4_pct": 5.7, "age_65plus_pct": 19.4, "gp_count": 41,  "car_ownership_pct": 82.8, "median_income": 27200, "lat": 48.5667, "lon": 13.9958},
    {"district_id": "AT-4-16", "name": "Kirchdorf an der Krems",  "state": "Oberösterreich", "population": 55000,  "age_0_4_pct": 4.8, "age_65plus_pct": 23.1, "gp_count": 20,  "car_ownership_pct": 87.9, "median_income": 28100, "lat": 47.9167, "lon": 14.1333},
    {"district_id": "AT-4-17", "name": "Schärding",               "state": "Oberösterreich", "population": 57000,  "age_0_4_pct": 4.8, "age_65plus_pct": 21.5, "gp_count": 29,  "car_ownership_pct": 77.4, "median_income": 23700, "lat": 48.4500, "lon": 13.4333},
    {"district_id": "AT-4-18", "name": "Eferding",                "state": "Oberösterreich", "population": 33000,  "age_0_4_pct": 4.5, "age_65plus_pct": 19.5, "gp_count": 15,  "car_ownership_pct": 86.8, "median_income": 24000, "lat": 48.3081, "lon": 14.0311},

    # ---- Tirol: all 9 political districts --------------------------------
    {"district_id": "AT-7-01", "name": "Innsbruck-Stadt", "state": "Tirol", "population": 132000, "age_0_4_pct": 5.6, "age_65plus_pct": 17.2, "gp_count": 194, "car_ownership_pct": 54.5, "median_income": 29200, "lat": 47.2682, "lon": 11.3923},
    {"district_id": "AT-7-02", "name": "Innsbruck-Land",  "state": "Tirol", "population": 186000, "age_0_4_pct": 5.9, "age_65plus_pct": 17.6, "gp_count": 155, "car_ownership_pct": 57.5, "median_income": 34200, "lat": 47.2833, "lon": 11.5000},
    {"district_id": "AT-7-03", "name": "Kufstein",        "state": "Tirol", "population": 112000, "age_0_4_pct": 5.2, "age_65plus_pct": 24.3, "gp_count": 49,  "car_ownership_pct": 79.6, "median_income": 31500, "lat": 47.5833, "lon": 12.1667},
    {"district_id": "AT-7-04", "name": "Schwaz",          "state": "Tirol", "population": 89000,  "age_0_4_pct": 4.8, "age_65plus_pct": 20.8, "gp_count": 44,  "car_ownership_pct": 87.9, "median_income": 31600, "lat": 47.3500, "lon": 11.7167},
    {"district_id": "AT-7-05", "name": "Kitzbühel",       "state": "Tirol", "population": 62000,  "age_0_4_pct": 5.0, "age_65plus_pct": 22.6, "gp_count": 23,  "car_ownership_pct": 87.0, "median_income": 27700, "lat": 47.4467, "lon": 12.3925},
    {"district_id": "AT-7-06", "name": "Imst",            "state": "Tirol", "population": 59000,  "age_0_4_pct": 5.2, "age_65plus_pct": 20.9, "gp_count": 28,  "car_ownership_pct": 89.6, "median_income": 29900, "lat": 47.2417, "lon": 10.7333},
    {"district_id": "AT-7-07", "name": "Landeck",         "state": "Tirol", "population": 45000,  "age_0_4_pct": 5.0, "age_65plus_pct": 24.4, "gp_count": 22,  "car_ownership_pct": 81.7, "median_income": 29800, "lat": 47.1400, "lon": 10.5667},
    {"district_id": "AT-7-08", "name": "Reutte",          "state": "Tirol", "population": 33000,  "age_0_4_pct": 5.1, "age_65plus_pct": 22.2, "gp_count": 14,  "car_ownership_pct": 81.6, "median_income": 31700, "lat": 47.4833, "lon": 10.7167},
    {"district_id": "AT-7-09", "name": "Lienz",           "state": "Tirol", "population": 49000,  "age_0_4_pct": 4.8, "age_65plus_pct": 23.9, "gp_count": 19,  "car_ownership_pct": 86.6, "median_income": 25200, "lat": 46.8300, "lon": 12.7600},
]


def fetch_demographics() -> pd.DataFrame:
    csv_cache = RAW_DIR / "demographics.csv"
    if csv_cache.exists():
        log.info(f"Loading cached demographics from {csv_cache}")
        return pd.read_csv(csv_cache)

    log.info("Using self-constructed illustrative demographics data "
             "(covering all 50 districts of Wien/Oberösterreich/Tirol; "
             "Statistik Austria's regional-statistics download requires an account)")
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

        # Approximate geographic centroid, used by the reasoning layer to
        # compute facility -> district travel times (materialize_reachability.py)
        if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
            g.add((uri, GEO.lat,  Literal(float(row["lat"]), datatype=XSD.decimal)))
            g.add((uri, GEO.long, Literal(float(row["lon"]), datatype=XSD.decimal)))

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
