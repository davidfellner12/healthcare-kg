"""
tests/test_api.py

Integration tests for the Flask REST API.
Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    """Build KG in memory and return Flask test client."""
    # Run ingestion + reasoning before starting the API
    from src.ingestion.healthcare_ingestion import main as ingest_hc
    from src.ingestion.demographics_ingestion import main as ingest_demo
    from src.ingestion.link_facilities_to_stops import main as link_stops
    from src.reasoning.materialize_reachability import main as materialize

    ingest_hc()
    ingest_demo()
    link_stops()
    materialize()

    from src.api.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Healthcare KG" in resp.data


def test_districts_endpoint(client):
    resp = client.get("/api/districts")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_facilities_endpoint(client):
    resp = client.get("/api/facilities")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_accessibility_endpoint(client):
    resp = client.get("/api/accessibility/AT-9-01")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)


def test_risk_endpoint(client):
    resp = client.get("/api/risk/AT-7-07")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)


def test_sparql_proxy(client):
    query = """
    PREFIX hkg: <http://healthcare-kg.at/ontology#>
    SELECT ?d WHERE { ?d a hkg:District }
    LIMIT 3
    """
    resp = client.post(
        "/api/sparql",
        json={"query": query},
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "results" in body
    assert body["count"] <= 3


def test_sparql_proxy_missing_query(client):
    resp = client.post("/api/sparql", json={}, content_type="application/json")
    assert resp.status_code == 400
