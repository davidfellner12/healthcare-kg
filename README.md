# Knowledge Graph-Based Healthcare Accessibility & Demographic Risk Intelligence

> **Course Project** — Extended Track (6 ECTS) | David Fellner  
> Approved: 21 April 2026

---

## Overview

This project builds a **three-component Knowledge Graph system** that integrates:
- Austrian healthcare facility data (Gesundheit Österreich / data.gv.at)
- Public transit timetables (GTFS — Wiener Linien + ÖBB)
- District-level demographic statistics (Statistics Austria)

It enables joint reasoning over **who can reach healthcare services**, **how quickly**, and **which population groups are most at risk**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Service & Infrastructure Layer                │
│  Flask REST API  ·  SPARQL Endpoint  ·  Map-based Demo UI       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  KG Embedding & GNN Layer                        │
│     TransE (under-served zone prediction via PyKEEN)            │
│     GraphSAGE (demographic risk scores via PyG)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Logic & Reasoning Layer                       │
│   Datalog/SPARQL rules · Reachability materialization           │
│   GP-per-1000 ratios · Age-weighted vulnerability scores        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    RDF Triplestore (Oxigraph)                    │
│  Healthcare facilities · Transit stops · Demographics · Links   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

> **Windows:** Do NOT run `pip install -r requirements.txt` directly.  
> `torch` must be installed before `pykeen` and `torch-geometric`.

**Windows:**
```bat
install.bat
```

**Linux / Mac:**
```bash
bash install.sh
```

### 2. Ingest data & build Knowledge Graph

```bash
# Download + convert all data sources to RDF
python scripts/build_kg.py

# Or run individual ingestion steps:
python src/ingestion/healthcare_ingestion.py
python src/ingestion/gtfs_ingestion.py
python src/ingestion/demographics_ingestion.py
python src/ingestion/link_facilities_to_stops.py
```

### 3. Load KG into triplestore & run reasoning

```bash
python src/reasoning/load_triplestore.py
python src/reasoning/materialize_reachability.py
python src/reasoning/compute_vulnerability.py
```

### 4. Train KG Embeddings (TransE)

```bash
python src/embeddings/train_transe.py
python src/embeddings/predict_underserved.py
```

### 5. Train GNN (GraphSAGE)

```bash
python src/gnn/build_graph.py
python src/gnn/train_graphsage.py
python src/gnn/predict_risk_scores.py
```

### 6. Start API & Demo

```bash
python src/api/app.py
# Visit http://localhost:5000
```

---

## Project Structure

```
healthcare-kg/
├── README.md
├── requirements.txt
├── config/
│   └── settings.py              # Central config (paths, endpoints, thresholds)
├── data/
│   ├── raw/                     # Downloaded raw data (gitignored)
│   ├── processed/               # Cleaned CSV/JSON intermediate files
│   └── rdf/                     # Generated .ttl / .nt RDF files
├── scripts/
│   └── build_kg.py              # One-shot pipeline runner
├── src/
│   ├── ingestion/               # ETL for each data source
│   │   ├── healthcare_ingestion.py
│   │   ├── gtfs_ingestion.py
│   │   ├── demographics_ingestion.py
│   │   └── link_facilities_to_stops.py
│   ├── reasoning/               # Datalog/SPARQL rules + materialization
│   │   ├── load_triplestore.py
│   │   ├── materialize_reachability.py
│   │   ├── compute_vulnerability.py
│   │   └── rules/
│   │       ├── reachability.sparql
│   │       └── vulnerability.sparql
│   ├── embeddings/              # TransE via PyKEEN
│   │   ├── train_transe.py
│   │   ├── predict_underserved.py
│   │   └── evaluate_embeddings.py
│   ├── gnn/                     # GraphSAGE via PyTorch Geometric
│   │   ├── build_graph.py
│   │   ├── train_graphsage.py
│   │   ├── predict_risk_scores.py
│   │   └── evaluate_gnn.py
│   ├── api/                     # Flask REST + SPARQL proxy
│   │   ├── app.py
│   │   ├── routes/
│   │   │   ├── accessibility.py
│   │   │   ├── risk.py
│   │   │   └── sparql_proxy.py
│   │   └── templates/
│   │       └── map_demo.html
│   └── utils/
│       ├── geo_utils.py         # Haversine distance, stop linking
│       ├── rdf_utils.py         # RDFLib helpers
│       └── gtfs_parser.py       # GTFS zip parsing utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_kg_analysis.ipynb
│   ├── 03_embeddings_analysis.ipynb
│   └── 04_gnn_analysis.ipynb
├── tests/
│   ├── test_ingestion.py
│   ├── test_reasoning.py
│   ├── test_embeddings.py
│   └── test_api.py
└── docs/
    ├── ontology.md              # RDF schema & namespace docs
    ├── sparql_examples.md       # Example SPARQL queries
    └── api_reference.md         # REST API docs
```

---

## Learning Outcomes Coverage

| LO | Topic | Component |
|----|-------|-----------|
| LO1 | KG Embeddings | `src/embeddings/` — TransE link prediction |
| LO2 | Logical Knowledge | `src/reasoning/rules/` — Datalog/SPARQL rules |
| LO3 | Graph Neural Networks | `src/gnn/` — GraphSAGE risk scores |
| LO4 | Data Models | `docs/ontology.md` — RDF vs property graph trade-offs |
| LO5 | Architectures | Batch reasoning + incremental update pipeline |
| LO6 | Scalable Reasoning | Pre-computation, indexing, named graphs |
| LO7 | KG Creation | `src/ingestion/` — ETL from GÖG + Statistik Austria + GTFS |
| LO8 | KG Evolution | Incremental re-reasoning on facility/timetable changes |
| LO9 | Real-World Applications | Public health accessibility planning |
| LO10 | Financial KGs | Insurance risk, public health budget allocation |
| LO11 | Services | SPARQL endpoint + REST API |
| LO12 | Connections | KG–ML–AI integration in public health |

---

## Data Sources

| Source | Format | Status |
|--------|--------|--------|
| Wiener Linien GTFS | GTFS ZIP | **Real, live-downloaded**: https://www.wienerlinien.at/ogd_realtime/doku/ogd/gtfs/gtfs.zip (4,268 stops, 693 routes) |
| Gesundheit Österreich (GÖG) | CSV/JSON | Live endpoint (data.gv.at) returns 404 from this environment; falls back to `SAMPLE_HOSPITALS` in `src/ingestion/healthcare_ingestion.py` — 87 facilities: 17 real, named hospitals at real approximate coordinates, plus 50 GPs (one per modelled district) and 20 pharmacies with fictional practitioners/names at realistic locations |
| ÖBB GTFS | GTFS ZIP | Requires registration, unavailable in this environment; Oberösterreich/Tirol facility-stop linking instead uses each regional capital's real main train station as a labelled illustrative "hub" stop (`gtfs_ingestion.py`) |
| Statistik Austria Districts | CSV | Regional-statistics download requires an account; falls back to `SAMPLE_DISTRICTS` in `src/ingestion/demographics_ingestion.py` — all 50 official political districts of Wien (23) / Oberösterreich (18) / Tirol (9), with real names/numbering/coordinates but internally-consistent illustrative demographic values (not exact official figures) |

See `docs/figures/` and the portfolio report (Section 2.1) for the full honesty framing and the resulting findings.

---

## Scope

- **Geographic focus**: full coverage of all political districts of Vienna, Upper Austria (OÖ), and Tyrol (50 districts, ~4.16M modelled population vs. Austria's real ~4.2M for these three states)
- **Transit**: Direct + single-transfer connections (full multi-modal is stretch goal)
- **Embeddings**: PyKEEN (TransE) and PyTorch Geometric (GraphSAGE) — no custom architectures
- **API**: Lightweight Flask; production hardening out of scope
- **KG Evolution**: Incremental re-reasoning; no real-time GTFS streaming
