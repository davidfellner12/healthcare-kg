"""
src/ingestion/healthcare_ingestion.py

Downloads (or loads from local cache) Austrian healthcare facility data
from Gesundheit Österreich (GÖG) / data.gv.at and converts it to RDF.

Facility types handled:
  - Hospitals (Krankenanstalten)
  - General Practitioners (Allgemeinmediziner / Kassenärzte)
  - Pharmacies (Apotheken)

Output: data/rdf/healthcare_facilities.ttl
"""

import sys
import json
import logging
from pathlib import Path
import requests
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RAW_DIR, RDF_DIR, NAMESPACES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Namespaces ────────────────────────────────────────────────────────────────
HKG  = Namespace(NAMESPACES["hkg"])
HKGR = Namespace(NAMESPACES["hkgr"])
GEO  = Namespace(NAMESPACES["geo"])
DCT  = Namespace(NAMESPACES["dct"])
SCH  = Namespace(NAMESPACES["schema"])

# ── Sample synthetic data (used when live download is unavailable) ─────────────
# Real project: replace with live download / CSV from data.gv.at
SAMPLE_HOSPITALS = [
    {"id": "KA001", "name": "Allgemeines Krankenhaus Wien",       "type": "Hospital",           "district": "AT-9-01", "lat": 48.2196, "lon": 16.3564, "beds": 1900, "state": "Wien"},
    {"id": "KA002", "name": "Krankenhaus Hietzing",              "type": "Hospital",           "district": "AT-9-13", "lat": 48.1861, "lon": 16.2828, "beds": 830,  "state": "Wien"},
    {"id": "KA003", "name": "Klinikum Wels-Grieskirchen",        "type": "Hospital",           "district": "AT-4-15", "lat": 48.1600, "lon": 14.0300, "beds": 1200, "state": "Oberösterreich"},
    {"id": "KA004", "name": "Kepler Universitätsklinikum Linz",  "type": "Hospital",           "district": "AT-4-10", "lat": 48.3069, "lon": 14.2858, "beds": 1800, "state": "Oberösterreich"},
    {"id": "KA005", "name": "Landeskrankenhaus Innsbruck",       "type": "Hospital",           "district": "AT-7-01", "lat": 47.2682, "lon": 11.3923, "beds": 1600, "state": "Tirol"},
    {"id": "GP001", "name": "Dr. Maria Huber – Allgemeinmedizin","type": "GeneralPractitioner","district": "AT-9-01", "lat": 48.2083, "lon": 16.3731, "beds": None,  "state": "Wien"},
    {"id": "GP002", "name": "Dr. Thomas Gruber – Hausarzt",      "type": "GeneralPractitioner","district": "AT-9-02", "lat": 48.2206, "lon": 16.4134, "beds": None,  "state": "Wien"},
    {"id": "GP003", "name": "Dr. Eva Rainer – Allgemeinmedizin", "type": "GeneralPractitioner","district": "AT-4-10", "lat": 48.3120, "lon": 14.2720, "beds": None,  "state": "Oberösterreich"},
    {"id": "GP004", "name": "Dr. Stefan Mair – Hausarzt",        "type": "GeneralPractitioner","district": "AT-7-01", "lat": 47.2650, "lon": 11.4000, "beds": None,  "state": "Tirol"},
    {"id": "PH001", "name": "Apotheke zum Weißen Engel",         "type": "Pharmacy",           "district": "AT-9-01", "lat": 48.2100, "lon": 16.3690, "beds": None,  "state": "Wien"},
    {"id": "PH002", "name": "Dom-Apotheke Linz",                 "type": "Pharmacy",           "district": "AT-4-10", "lat": 48.3050, "lon": 14.2860, "beds": None,  "state": "Oberösterreich"},
]


def fetch_healthcare_data() -> pd.DataFrame:
    """
    Attempt to download from data.gv.at; fall back to sample data.
    In the real project, replace the URL with the exact CSV endpoint.
    """
    csv_cache = RAW_DIR / "healthcare_facilities.csv"
    if csv_cache.exists():
        log.info(f"Loading cached healthcare data from {csv_cache}")
        return pd.read_csv(csv_cache)

    # ---- attempt live download -----------------------------------------------
    try:
        url = "https://data.gv.at/katalog/api/3/action/datastore_search?resource_id=krankenanstalten&limit=5000"
        log.info(f"Downloading healthcare data from {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        records = r.json()["result"]["records"]
        df = pd.DataFrame(records)
        df.to_csv(csv_cache, index=False)
        log.info(f"Saved {len(df)} records to {csv_cache}")
        return df
    except Exception as e:
        log.warning(f"Live download failed ({e}); using sample data")

    df = pd.DataFrame(SAMPLE_HOSPITALS)
    df.to_csv(csv_cache, index=False)
    return df


def _facility_uri(facility_id: str) -> URIRef:
    return HKGR[f"facility/{facility_id}"]


def _district_uri(district_code: str) -> URIRef:
    return HKGR[f"district/{district_code}"]


def facilities_to_rdf(df: pd.DataFrame) -> Graph:
    g = Graph()

    # Bind prefixes
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))

    # ── Ontology classes ───────────────────────────────────────────────────────
    for cls in ["HealthcareFacility", "Hospital", "GeneralPractitioner", "Pharmacy", "District"]:
        g.add((HKG[cls], RDF.type, OWL.Class))
        g.add((HKG[cls], RDFS.label, Literal(cls, lang="en")))

    g.add((HKG.Hospital,            RDFS.subClassOf, HKG.HealthcareFacility))
    g.add((HKG.GeneralPractitioner, RDFS.subClassOf, HKG.HealthcareFacility))
    g.add((HKG.Pharmacy,            RDFS.subClassOf, HKG.HealthcareFacility))

    # ── Properties ────────────────────────────────────────────────────────────
    for prop, label in [
        ("facilityId",  "Facility ID"),
        ("facilityName","Facility Name"),
        ("bedCount",    "Number of beds"),
        ("inDistrict",  "Located in district"),
        ("inState",     "Located in federal state"),
        ("hasStop",     "Nearest GTFS stop"),
    ]:
        g.add((HKG[prop], RDF.type, OWL.DatatypeProperty if prop not in ("inDistrict", "hasStop") else OWL.ObjectProperty))
        g.add((HKG[prop], RDFS.label, Literal(label, lang="en")))

    # ── Individuals ───────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        fid  = str(row.get("id", row.get("facilityId", "UNKNOWN")))
        uri  = _facility_uri(fid)
        ftype = str(row.get("type", "HealthcareFacility"))

        g.add((uri, RDF.type,         HKG[ftype]))
        g.add((uri, HKG.facilityId,   Literal(fid)))
        g.add((uri, HKG.facilityName, Literal(str(row.get("name", "")), lang="de")))
        g.add((uri, RDFS.label,       Literal(str(row.get("name", "")), lang="de")))

        if pd.notna(row.get("lat")):
            g.add((uri, GEO.lat, Literal(float(row["lat"]), datatype=XSD.decimal)))
        if pd.notna(row.get("lon")):
            g.add((uri, GEO.long, Literal(float(row["lon"]), datatype=XSD.decimal)))

        if pd.notna(row.get("beds")):
            g.add((uri, HKG.bedCount, Literal(int(row["beds"]), datatype=XSD.integer)))

        if pd.notna(row.get("district")):
            g.add((uri, HKG.inDistrict, _district_uri(str(row["district"]))))

        if pd.notna(row.get("state")):
            g.add((uri, HKG.inState, Literal(str(row["state"]), lang="de")))

    log.info(f"Generated {len(g)} RDF triples from {len(df)} facilities")
    return g


def main():
    df  = fetch_healthcare_data()
    g   = facilities_to_rdf(df)
    out = RDF_DIR / "healthcare_facilities.ttl"
    g.serialize(destination=str(out), format="turtle")
    log.info(f"Saved RDF to {out}")


if __name__ == "__main__":
    main()
