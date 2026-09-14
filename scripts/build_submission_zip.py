"""
scripts/build_submission_zip.py

Packages the code + small reproducibility artifacts into the single
submission ZIP the course asks for, with the four numbered subfolders
from the portfolio guidance ("2 - construction", "3 - ML", "4 - logic",
"5 - reflection"). The large, publicly-downloadable GTFS zip and its
derived gtfs_transit.ttl are deliberately left out (a stable public
link is cited in the report instead, per the guidance for public
datasets); everything self-constructed (facility/demographics CSVs,
the small derived .ttl files) is included.

Run from the repo root: python scripts/build_submission_zip.py
Output: <repo>/../Report/Fellner_David_KG_Portfolio_submission.zip
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parents[1] / "Report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE = ROOT.parents[0] / "_submission_stage"
ZIP_PATH = OUT_DIR / "Fellner_David_KG_Portfolio_submission.zip"

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)


def copy(src, dst):
    dst = STAGE / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if (ROOT / src).is_dir():
        shutil.copytree(ROOT / src, dst, dirs_exist_ok=True,
                         ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(ROOT / src, dst)


# ---- "2 - construction": ingestion, triplestore loading, service layer ---
for f in ["src/ingestion", "src/utils", "src/reasoning/load_triplestore.py",
          "config", "src/api", "scripts/build_kg.py", "docs/ontology.md",
          "data/raw/healthcare_facilities.csv", "data/raw/demographics.csv",
          "data/rdf/healthcare_facilities.ttl", "data/rdf/demographics.ttl",
          "data/rdf/facility_stop_links.ttl"]:
    copy(f, f"2 - construction/{f}")

# ---- "3 - ML": embeddings + GNN code and their (small) outputs -----------
for f in ["src/embeddings", "src/gnn",
          "models/transe/results.json", "models/transe/metadata.json",
          "models/transe/underserved_predictions.json", "models/transe/link_prediction_examples.json",
          "models/graphsage/metrics.json", "models/graphsage/risk_scores.json",
          "models/graphsage/graph_stats.json", "docs/figures"]:
    copy(f, f"3 - ML/{f}")

# ---- "4 - logic": SPARQL rules, reasoning script, KG evolution ----------
for f in ["src/reasoning/materialize_reachability.py", "src/reasoning/kg_evolution.py",
          "src/reasoning/rules", "docs/sparql_examples.md",
          "data/rdf/reachability.ttl", "data/rdf/vulnerability.ttl"]:
    copy(f, f"4 - logic/{f}")
# KG-evolution audit trail (change log + the one retraction patch generated in the demo)
for f in (ROOT / "data" / "rdf").glob("change_log.jsonl"):
    copy(f.relative_to(ROOT), f"4 - logic/{f.relative_to(ROOT)}")
for f in (ROOT / "data" / "rdf").glob("patch_remove_*.sparql"):
    copy(f.relative_to(ROOT), f"4 - logic/{f.relative_to(ROOT)}")

# ---- "5 - reflection": no new code; service-layer evidence for LO11 -----
(STAGE / "5 - reflection").mkdir(parents=True, exist_ok=True)
(STAGE / "5 - reflection" / "api_smoke_test_evidence.md").write_text(
    "# Service smoke-test evidence (Section 5.1)\n\n"
    "Endpoints exercised against the live KG (src/api/app.py, duplicated under "
    "'2 - construction/src/api'):\n\n"
    "```\n"
    "GET /api/risk/AT-7-07\n"
    '[{"gpDeficit": "true", "gpPer1000": "0.375", "name": "Lienz", "pop": "48000", '
    '"risk": "HighRisk", "vuln": "0.5368"}]\n\n'
    "GET /api/underserved  (top of ranked list)\n"
    '[{"district_id": "AT-7-07", "district_name": "Lienz", "predicted_risk": "HighRisk", '
    '"actual_risk": "HighRisk", "match": true, ...}, ...]\n\n'
    "GET /api/accessibility/AT-9-01  (first result)\n"
    '[{"fac": ".../facility/KA001", "name": "Allgemeines Krankenhaus Wien", '
    '"reachableIn15min": true, "reachableIn30min": true, "reachableIn60min": true, '
    '"type": "Hospital"}, ...]\n\n'
    "POST /api/sparql  {\"query\": \"PREFIX hkg: <http://healthcare-kg.at/ontology#> "
    "SELECT (COUNT(*) as ?c) WHERE { ?d a hkg:District }\"}\n"
    '{"count": 1, "results": [{"c": "16"}]}\n'
    "```\n", encoding="utf-8"
)

# ---- top level: README, requirements, tests, install scripts ------------
for f in ["README.md", "requirements.txt", "install.bat", "install.sh", "tests"]:
    copy(f, f)

# ---- full-source: the complete, unsplit repo, for actually RUNNING it ---
# (the "2 - construction" / "3 - ML" / "4 - logic" / "5 - reflection" split
# above exists to satisfy the portfolio's per-section folder convention for
# review; scripts reference each other via sys.path assuming the single
# repo root, so real execution should happen against this copy instead.)
full_src = STAGE / "full-source"
shutil.copytree(ROOT, full_src, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", "data", "models", ".idea"))
(full_src / "data" / "raw").mkdir(parents=True, exist_ok=True)
(full_src / "data" / "rdf").mkdir(parents=True, exist_ok=True)
for f in ["data/raw/healthcare_facilities.csv", "data/raw/demographics.csv"]:
    shutil.copy2(ROOT / f, full_src / f)

readme = STAGE / "readme.md"
readme.write_text(
    "# Submission ZIP — Healthcare KG Portfolio\n\n"
    "Two views of the same code are included:\n\n"
    "- `2 - construction`, `3 - ML`, `4 - logic`, `5 - reflection` — copies of just the files "
    "relevant to each correspondingly-numbered report section, per the portfolio's folder "
    "convention. These are for review, matching what each section of the report discusses.\n"
    "- `full-source/` — the complete, runnable repository (single root, all imports resolve). "
    "**Use this copy to actually execute anything** — the split folders above reference each "
    "other via `sys.path` assuming one repo root, so running a script from inside e.g. "
    "`4 - logic/` directly will not find `config/`.\n\n"
    "## What's NOT included, and why\n"
    "- `data/raw/gtfs_wienerlinien.zip` (65 MB) and the derived `data/rdf/gtfs_transit.ttl` "
    "(916 KB) are left out of every copy: this is public data with a stable link "
    "(https://www.wienerlinien.at/ogd_realtime/doku/ogd/gtfs/gtfs.zip), cited in the report, "
    "per the guidance that public datasets don't need to be re-shipped. Re-running "
    "`full-source/src/ingestion/gtfs_ingestion.py` re-downloads it.\n"
    "- Trained model binaries (`trained_model.pkl`, `training_triples/`, `graphsage_weights."
    "pt`) are regenerated by the commands below in well under a minute and are not needed to "
    "re-derive the report's numbers from the included JSON outputs.\n\n"
    "## How to run everything from scratch\n"
    "```\n"
    "cd full-source\n"
    "install.bat                                      # or install.sh on Linux/Mac\n"
    "python src/ingestion/healthcare_ingestion.py\n"
    "python src/ingestion/gtfs_ingestion.py           # downloads the public GTFS feed live\n"
    "python src/ingestion/demographics_ingestion.py\n"
    "python src/ingestion/link_facilities_to_stops.py\n"
    "python src/reasoning/load_triplestore.py\n"
    "python src/reasoning/materialize_reachability.py\n"
    "python src/embeddings/train_transe.py\n"
    "python src/embeddings/predict_underserved.py\n"
    "python src/gnn/train_graphsage.py\n"
    "python src/gnn/predict_risk_scores.py\n"
    "python src/api/app.py                            # http://localhost:5000\n\n"
    "# KG-evolution demo (report Section 4.2):\n"
    "python src/reasoning/kg_evolution.py --event facility_added --id KA999 --lat 48.30 "
    "--lon 14.29 --type Hospital --district AT-4-10 --name \"Neues Krankenhaus Linz\"\n"
    "python src/reasoning/kg_evolution.py --event facility_closed --id KA002\n"
    "python src/reasoning/load_triplestore.py --apply-patch data/rdf/patch_remove_KA002_*.sparql\n\n"
    "# Tests:\n"
    "python -m pytest tests/ -v\n"
    "```\n",
    encoding="utf-8"
)

# ---- zip it up ------------------------------------------------------------
if ZIP_PATH.exists():
    ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in STAGE.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(STAGE))

shutil.rmtree(STAGE)
size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"Wrote {ZIP_PATH} ({size_mb:.1f} MB)")
