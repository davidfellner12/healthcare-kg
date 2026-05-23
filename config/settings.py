"""
config/settings.py
Central configuration for the Healthcare KG project.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RDF_DIR = DATA_DIR / "rdf"
MODELS_DIR = BASE_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, RDF_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── RDF Namespaces ─────────────────────────────────────────────────────────────
NAMESPACES = {
    "hkg":   "http://healthcare-kg.at/ontology#",
    "hkgr":  "http://healthcare-kg.at/resource/",
    "schema": "https://schema.org/",
    "geo":   "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "gtfs":  "http://vocab.gtfs.org/terms#",
    "dct":   "http://purl.org/dc/terms/",
    "rdfs":  "http://www.w3.org/2000/01/rdf-schema#",
    "owl":   "http://www.w3.org/2002/07/owl#",
    "xsd":   "http://www.w3.org/2001/XMLSchema#",
}

# ── Triplestore ────────────────────────────────────────────────────────────────
TRIPLESTORE_TYPE = "oxigraph"          # or "jena"
OXIGRAPH_PATH = str(DATA_DIR / "oxigraph_store")
SPARQL_ENDPOINT = "http://localhost:7878/query"
SPARQL_UPDATE_ENDPOINT = "http://localhost:7878/update"

# ── Geographic Focus ───────────────────────────────────────────────────────────
FOCUS_STATES = ["Wien", "Oberösterreich", "Tirol"]

# Bounding boxes (min_lon, min_lat, max_lon, max_lat)
BBOX = {
    "Wien":          (16.18, 48.12, 16.58, 48.33),
    "Oberösterreich":(13.20, 47.50, 15.00, 48.80),
    "Tirol":         (10.40, 46.70, 12.95, 47.75),
    "Austria":       (9.53,  46.37, 17.17, 49.02),
}

# ── Reachability Thresholds (minutes) ──────────────────────────────────────────
REACHABILITY_THRESHOLDS = [15, 30, 60]   # materialised for each district
WALKING_SPEED_KMH = 4.5
CAR_SPEED_KMH = 50.0
MAX_TRANSFER_COUNT = 1                   # direct + single-transfer

# ── Facility Linking ───────────────────────────────────────────────────────────
MAX_STOP_DISTANCE_KM = 5.0              # max walking distance to nearest stop

# ── TransE (PyKEEN) ────────────────────────────────────────────────────────────
TRANSE_CONFIG = {
    "model": "TransE",
    "embedding_dim": 128,
    "scoring_fct_norm": 1,
    "epochs": 200,
    "batch_size": 512,
    "learning_rate": 0.001,
    "loss": "marginranking",
    "margin": 1.0,
    "random_seed": 42,
    "train_split": 0.8,
    "val_split": 0.1,
}

# ── GraphSAGE (PyTorch Geometric) ─────────────────────────────────────────────
GRAPHSAGE_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "epochs": 150,
    "learning_rate": 0.005,
    "weight_decay": 5e-4,
    "batch_size": 256,
    "random_seed": 42,
    "aggr": "mean",
}

# ── Vulnerability Weights ─────────────────────────────────────────────────────
# Used in age-weighted vulnerability score computation
VULNERABILITY_WEIGHTS = {
    "age_over_65_pct":  0.40,
    "age_under_5_pct":  0.20,
    "no_car_pct":       0.25,
    "low_income_pct":   0.15,
}

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# ── Data Source URLs ──────────────────────────────────────────────────────────
DATA_SOURCES = {
    "wienerlinien_gtfs": "https://www.wienerlinien.at/ogd_realtime/doku/ogd/gtfs/gtfs.zip",
    "goeg_hospitals":    "https://data.gv.at/katalog/dataset/krankenanstalten",  # landing page
    "statistik_austria": "https://www.statistik.at/en/databases/regional-statistics",
}
