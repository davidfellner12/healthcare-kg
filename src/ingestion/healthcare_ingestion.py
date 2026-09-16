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

# ── Self-constructed / illustrative facility data (used when the live -----
# data.gv.at download is unavailable, which is the case in this environment)
# ─────────────────────────────────────────────────────────────────────────
# Hospitals below are REAL Austrian institutions (real names, approximate
# real coordinates) -- their bed counts are order-of-magnitude estimates,
# not sourced from an official register. GPs (one representative per
# modelled district) and pharmacies use fictional names/practitioners at
# realistic locations, since no public per-practice registry was available;
# this mirrors the honesty declaration already given for the demographics
# side of the KG (see demographics_ingestion.py).
SAMPLE_HOSPITALS = [
    # ---- Real hospitals: Wien (8) ------------------------------------------
    {"id": "KA001", "name": "Allgemeines Krankenhaus Wien (AKH)",           "type": "Hospital", "district": "AT-9-09", "lat": 48.2196, "lon": 16.3564, "beds": 1900, "state": "Wien"},
    {"id": "KA002", "name": "Klinik Hietzing",                              "type": "Hospital", "district": "AT-9-13", "lat": 48.1861, "lon": 16.2828, "beds": 830,  "state": "Wien"},
    {"id": "KA003", "name": "Klinik Favoriten",                             "type": "Hospital", "district": "AT-9-10", "lat": 48.1650, "lon": 16.3800, "beds": 1140, "state": "Wien"},
    {"id": "KA004", "name": "Klinik Floridsdorf",                           "type": "Hospital", "district": "AT-9-21", "lat": 48.2600, "lon": 16.4100, "beds": 785,  "state": "Wien"},
    {"id": "KA005", "name": "Klinik Donaustadt",                            "type": "Hospital", "district": "AT-9-22", "lat": 48.2350, "lon": 16.4600, "beds": 785,  "state": "Wien"},
    {"id": "KA006", "name": "Klinik Ottakring",                             "type": "Hospital", "district": "AT-9-16", "lat": 48.2150, "lon": 16.3000, "beds": 830,  "state": "Wien"},
    {"id": "KA007", "name": "Krankenhaus der Barmherzigen Brüder Wien",     "type": "Hospital", "district": "AT-9-02", "lat": 48.2270, "lon": 16.3800, "beds": 380,  "state": "Wien"},
    {"id": "KA008", "name": "St. Anna Kinderspital",                        "type": "Hospital", "district": "AT-9-09", "lat": 48.2200, "lon": 16.3600, "beds": 130,  "state": "Wien"},
    # ---- Real hospitals: Oberösterreich (5) --------------------------------
    {"id": "KA009", "name": "Kepler Universitätsklinikum Linz",             "type": "Hospital", "district": "AT-4-01", "lat": 48.3069, "lon": 14.2858, "beds": 1800, "state": "Oberösterreich"},
    {"id": "KA010", "name": "Klinikum Wels-Grieskirchen",                   "type": "Hospital", "district": "AT-4-02", "lat": 48.1600, "lon": 14.0300, "beds": 1200, "state": "Oberösterreich"},
    {"id": "KA011", "name": "LKH Steyr",                                    "type": "Hospital", "district": "AT-4-03", "lat": 48.0396, "lon": 14.4212, "beds": 640,  "state": "Oberösterreich"},
    {"id": "KA012", "name": "Salzkammergut-Klinikum Gmunden",               "type": "Hospital", "district": "AT-4-09", "lat": 47.9186, "lon": 13.7994, "beds": 340,  "state": "Oberösterreich"},
    {"id": "KA013", "name": "Innviertel-Klinikum Ried",                     "type": "Hospital", "district": "AT-4-10", "lat": 48.2100, "lon": 13.4890, "beds": 460,  "state": "Oberösterreich"},
    # ---- Real hospitals: Tirol (4) ------------------------------------------
    {"id": "KA014", "name": "Landeskrankenhaus Innsbruck (Tirol Kliniken)", "type": "Hospital", "district": "AT-7-01", "lat": 47.2682, "lon": 11.3923, "beds": 1600, "state": "Tirol"},
    {"id": "KA015", "name": "Bezirkskrankenhaus Kufstein",                  "type": "Hospital", "district": "AT-7-03", "lat": 47.5833, "lon": 12.1667, "beds": 340,  "state": "Tirol"},
    {"id": "KA016", "name": "Bezirkskrankenhaus Schwaz",                    "type": "Hospital", "district": "AT-7-04", "lat": 47.3500, "lon": 11.7167, "beds": 220,  "state": "Tirol"},
    {"id": "KA017", "name": "Bezirkskrankenhaus Lienz",                     "type": "Hospital", "district": "AT-7-09", "lat": 46.8300, "lon": 12.7600, "beds": 280,  "state": "Tirol"},

    # ---- General practitioners: one per district (50) --------------------
    {"id": "GP001", "name": "Dr. Maria Mayer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-01", "lat": 48.2146, "lon": 16.3725, "beds": None, "state": "Wien"},
    {"id": "GP002", "name": "Dr. Thomas Aigner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-02", "lat": 48.2165, "lon": 16.3829, "beds": None, "state": "Wien"},
    {"id": "GP003", "name": "Dr. Eva Baumgartner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-03", "lat": 48.2078, "lon": 16.3929, "beds": None, "state": "Wien"},
    {"id": "GP004", "name": "Dr. Stefan Wimmer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-04", "lat": 48.2012, "lon": 16.3703, "beds": None, "state": "Wien"},
    {"id": "GP005", "name": "Dr. Anna Pfeiffer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-05", "lat": 48.1913, "lon": 16.3585, "beds": None, "state": "Wien"},
    {"id": "GP006", "name": "Dr. Michael Ebner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-06", "lat": 48.2063, "lon": 16.3473, "beds": None, "state": "Wien"},
    {"id": "GP007", "name": "Dr. Julia Peer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-07", "lat": 48.2064, "lon": 16.351, "beds": None, "state": "Wien"},
    {"id": "GP008", "name": "Dr. Andreas Wagner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-08", "lat": 48.2181, "lon": 16.3483, "beds": None, "state": "Wien"},
    {"id": "GP009", "name": "Dr. Sabine Fischer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-09", "lat": 48.2243, "lon": 16.3544, "beds": None, "state": "Wien"},
    {"id": "GP010", "name": "Dr. Georg Brunner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-10", "lat": 48.1705, "lon": 16.3779, "beds": None, "state": "Wien"},
    {"id": "GP011", "name": "Dr. Petra Auer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-11", "lat": 48.1706, "lon": 16.4262, "beds": None, "state": "Wien"},
    {"id": "GP012", "name": "Dr. Christian Wagner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-12", "lat": 48.1798, "lon": 16.3381, "beds": None, "state": "Wien"},
    {"id": "GP013", "name": "Dr. Claudia Fellner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-13", "lat": 48.1752, "lon": 16.2897, "beds": None, "state": "Wien"},
    {"id": "GP014", "name": "Dr. Martin Angerer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-14", "lat": 48.1921, "lon": 16.2741, "beds": None, "state": "Wien"},
    {"id": "GP015", "name": "Dr. Sandra Gruber – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-15", "lat": 48.1982, "lon": 16.3282, "beds": None, "state": "Wien"},
    {"id": "GP016", "name": "Dr. Wolfgang Leitner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-16", "lat": 48.2204, "lon": 16.305, "beds": None, "state": "Wien"},
    {"id": "GP017", "name": "Dr. Karin Egger – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-17", "lat": 48.2409, "lon": 16.326, "beds": None, "state": "Wien"},
    {"id": "GP018", "name": "Dr. Bernhard Eder – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-18", "lat": 48.2372, "lon": 16.3249, "beds": None, "state": "Wien"},
    {"id": "GP019", "name": "Dr. Elisabeth Schneider – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-19", "lat": 48.24, "lon": 16.3391, "beds": None, "state": "Wien"},
    {"id": "GP020", "name": "Dr. Franz Lang – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-20", "lat": 48.2407, "lon": 16.3748, "beds": None, "state": "Wien"},
    {"id": "GP021", "name": "Dr. Monika Wieser – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-21", "lat": 48.2602, "lon": 16.4017, "beds": None, "state": "Wien"},
    {"id": "GP022", "name": "Dr. Peter Huber – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-22", "lat": 48.2425, "lon": 16.4572, "beds": None, "state": "Wien"},
    {"id": "GP023", "name": "Dr. Barbara Lechner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-9-23", "lat": 48.1366, "lon": 16.2901, "beds": None, "state": "Wien"},
    {"id": "GP024", "name": "Dr. Josef Schmid – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-01", "lat": 48.3018, "lon": 14.2786, "beds": None, "state": "Oberösterreich"},
    {"id": "GP025", "name": "Dr. Ingrid Weber – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-02", "lat": 48.157, "lon": 14.0369, "beds": None, "state": "Oberösterreich"},
    {"id": "GP026", "name": "Dr. Klaus Wolf – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-03", "lat": 48.0307, "lon": 14.4234, "beds": None, "state": "Oberösterreich"},
    {"id": "GP027", "name": "Dr. Ursula Kaltenbrunner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-04", "lat": 48.2297, "lon": 14.2466, "beds": None, "state": "Oberösterreich"},
    {"id": "GP028", "name": "Dr. Herbert Kirchner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-05", "lat": 48.1415, "lon": 13.9655, "beds": None, "state": "Oberösterreich"},
    {"id": "GP029", "name": "Dr. Brigitte Neumann – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-06", "lat": 48.0075, "lon": 13.6473, "beds": None, "state": "Oberösterreich"},
    {"id": "GP030", "name": "Dr. Manfred Hofer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-07", "lat": 48.2528, "lon": 13.0284, "beds": None, "state": "Oberösterreich"},
    {"id": "GP031", "name": "Dr. Renate Winkler – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-08", "lat": 48.3509, "lon": 14.2253, "beds": None, "state": "Oberösterreich"},
    {"id": "GP032", "name": "Dr. Helmut Reiter – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-09", "lat": 47.9279, "lon": 13.8037, "beds": None, "state": "Oberösterreich"},
    {"id": "GP033", "name": "Dr. Gabriele Binder – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-10", "lat": 48.2178, "lon": 13.4895, "beds": None, "state": "Oberösterreich"},
    {"id": "GP034", "name": "Dr. Kurt Wallner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-11", "lat": 48.2417, "lon": 13.8295, "beds": None, "state": "Oberösterreich"},
    {"id": "GP035", "name": "Dr. Doris Holzer – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-12", "lat": 47.9715, "lon": 14.3786, "beds": None, "state": "Oberösterreich"},
    {"id": "GP036", "name": "Dr. Alexander Scharf – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-13", "lat": 48.502, "lon": 14.4952, "beds": None, "state": "Oberösterreich"},
    {"id": "GP037", "name": "Dr. Susanne Berger – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-14", "lat": 48.2594, "lon": 14.6432, "beds": None, "state": "Oberösterreich"},
    {"id": "GP038", "name": "Dr. Werner Pichler – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-15", "lat": 48.5661, "lon": 13.9928, "beds": None, "state": "Oberösterreich"},
    {"id": "GP039", "name": "Dr. Christine Mayr – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-16", "lat": 47.9238, "lon": 14.1328, "beds": None, "state": "Oberösterreich"},
    {"id": "GP040", "name": "Dr. Erwin Maier – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-17", "lat": 48.4503, "lon": 13.44, "beds": None, "state": "Oberösterreich"},
    {"id": "GP041", "name": "Dr. Silvia Riedl – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-4-18", "lat": 48.3001, "lon": 14.0266, "beds": None, "state": "Oberösterreich"},
    {"id": "GP042", "name": "Dr. Rudolf Gasser – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-01", "lat": 47.271, "lon": 11.3843, "beds": None, "state": "Tirol"},
    {"id": "GP043", "name": "Dr. Martina Unterberger – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-02", "lat": 47.2845, "lon": 11.4982, "beds": None, "state": "Tirol"},
    {"id": "GP044", "name": "Dr. Walter Steiner – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-03", "lat": 47.5738, "lon": 12.163, "beds": None, "state": "Tirol"},
    {"id": "GP045", "name": "Dr. Andrea Moser – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-04", "lat": 47.3542, "lon": 11.7214, "beds": None, "state": "Tirol"},
    {"id": "GP046", "name": "Dr. Gerhard Fuchs – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-05", "lat": 47.4401, "lon": 12.3926, "beds": None, "state": "Tirol"},
    {"id": "GP047", "name": "Dr. Birgit Schwarz – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-06", "lat": 47.2415, "lon": 10.727, "beds": None, "state": "Tirol"},
    {"id": "GP048", "name": "Dr. Johann Haas – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-07", "lat": 47.1368, "lon": 10.5607, "beds": None, "state": "Tirol"},
    {"id": "GP049", "name": "Dr. Nicole Kofler – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-08", "lat": 47.4933, "lon": 10.7085, "beds": None, "state": "Tirol"},
    {"id": "GP050", "name": "Dr. Karl Rieder – Allgemeinmedizin", "type": "GeneralPractitioner", "district": "AT-7-09", "lat": 46.8383, "lon": 12.7634, "beds": None, "state": "Tirol"},

    # ---- Pharmacies: 20 in the larger population centres -------------------
    {"id": "PH001", "name": "Apotheke zum Weissen Engel", "type": "Pharmacy", "district": "AT-9-01", "lat": 48.2006, "lon": 16.3828, "beds": None, "state": "Wien"},
    {"id": "PH002", "name": "Dom-Apotheke", "type": "Pharmacy", "district": "AT-9-10", "lat": 48.1691, "lon": 16.3735, "beds": None, "state": "Wien"},
    {"id": "PH003", "name": "Stadt-Apotheke", "type": "Pharmacy", "district": "AT-9-21", "lat": 48.256, "lon": 16.3949, "beds": None, "state": "Wien"},
    {"id": "PH004", "name": "Rathaus-Apotheke", "type": "Pharmacy", "district": "AT-9-22", "lat": 48.2324, "lon": 16.4685, "beds": None, "state": "Wien"},
    {"id": "PH005", "name": "Marien-Apotheke", "type": "Pharmacy", "district": "AT-9-16", "lat": 48.2195, "lon": 16.2986, "beds": None, "state": "Wien"},
    {"id": "PH006", "name": "Sonnen-Apotheke", "type": "Pharmacy", "district": "AT-9-11", "lat": 48.166, "lon": 16.4273, "beds": None, "state": "Wien"},
    {"id": "PH007", "name": "Adler-Apotheke", "type": "Pharmacy", "district": "AT-9-12", "lat": 48.1692, "lon": 16.3371, "beds": None, "state": "Wien"},
    {"id": "PH008", "name": "Bahnhof-Apotheke", "type": "Pharmacy", "district": "AT-9-14", "lat": 48.1949, "lon": 16.2769, "beds": None, "state": "Wien"},
    {"id": "PH009", "name": "Central-Apotheke", "type": "Pharmacy", "district": "AT-9-23", "lat": 48.1295, "lon": 16.2807, "beds": None, "state": "Wien"},
    {"id": "PH010", "name": "Löwen-Apotheke", "type": "Pharmacy", "district": "AT-4-01", "lat": 48.3068, "lon": 14.2863, "beds": None, "state": "Oberösterreich"},
    {"id": "PH011", "name": "Hirsch-Apotheke", "type": "Pharmacy", "district": "AT-4-02", "lat": 48.1561, "lon": 14.0254, "beds": None, "state": "Oberösterreich"},
    {"id": "PH012", "name": "Kronen-Apotheke", "type": "Pharmacy", "district": "AT-4-03", "lat": 48.0446, "lon": 14.4187, "beds": None, "state": "Oberösterreich"},
    {"id": "PH013", "name": "Neustadt-Apotheke", "type": "Pharmacy", "district": "AT-4-04", "lat": 48.2154, "lon": 14.2419, "beds": None, "state": "Oberösterreich"},
    {"id": "PH014", "name": "Berg-Apotheke", "type": "Pharmacy", "district": "AT-4-06", "lat": 48.0161, "lon": 13.6647, "beds": None, "state": "Oberösterreich"},
    {"id": "PH015", "name": "See-Apotheke", "type": "Pharmacy", "district": "AT-4-07", "lat": 48.2534, "lon": 13.0379, "beds": None, "state": "Oberösterreich"},
    {"id": "PH016", "name": "Tal-Apotheke", "type": "Pharmacy", "district": "AT-7-01", "lat": 47.2684, "lon": 11.3913, "beds": None, "state": "Tirol"},
    {"id": "PH017", "name": "Linden-Apotheke", "type": "Pharmacy", "district": "AT-7-02", "lat": 47.2853, "lon": 11.5078, "beds": None, "state": "Tirol"},
    {"id": "PH018", "name": "Rosen-Apotheke", "type": "Pharmacy", "district": "AT-7-03", "lat": 47.5884, "lon": 12.1684, "beds": None, "state": "Tirol"},
    {"id": "PH019", "name": "Post-Apotheke", "type": "Pharmacy", "district": "AT-7-04", "lat": 47.3442, "lon": 11.7192, "beds": None, "state": "Tirol"},
    {"id": "PH020", "name": "Markt-Apotheke", "type": "Pharmacy", "district": "AT-7-09", "lat": 46.8298, "lon": 12.7587, "beds": None, "state": "Tirol"},
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
        log.warning(f"Live download failed ({e}); using self-constructed illustrative facility "
                    f"data ({len(SAMPLE_HOSPITALS)} facilities: real named hospitals + one "
                    f"representative GP per modelled district + selected pharmacies)")

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
