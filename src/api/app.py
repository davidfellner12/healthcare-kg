"""
src/api/app.py  —  Healthcare KG REST API
Queries rewritten to avoid BIND inside UNION (pyparsing version sensitivity).
"""

import sys
import json
import logging
from pathlib import Path
from functools import lru_cache

from flask import Flask, jsonify, request, render_template_string, abort
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import API_HOST, API_PORT, API_DEBUG, NAMESPACES, RDF_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

FACILITY_TYPES = ["Hospital", "GeneralPractitioner", "Pharmacy"]


@lru_cache(maxsize=1)
def get_graph():
    from rdflib import Graph, Namespace
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))
    for ttl in sorted(RDF_DIR.glob("*.ttl")):
        log.info(f"Loading {ttl.name} …")
        g.parse(str(ttl), format="turtle")
    log.info(f"KG loaded: {len(g)} triples")
    return g


def run_query(query: str) -> list:
    g = get_graph()
    rows = []
    for r in g.query(query):
        rows.append({str(v): (str(val) if val is not None else None)
                     for v, val in zip(r.labels, r)})
    return rows


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Healthcare KG — Accessibility Demo</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, sans-serif; display: flex; flex-direction: column; height: 100vh; }
    #header { padding: .75rem 1rem; background: #1a365d; color: white; }
    #header h1 { font-size: 1.1rem; }
    #header p  { font-size: .8rem; opacity: .8; margin-top: .2rem; }
    #map { flex: 1; }
    #legend { display: flex; gap: 1.2rem; padding: .5rem 1rem; background: #f0f4f8; font-size: .8rem; align-items: center; flex-wrap: wrap; }
    .dot { width: 13px; height: 13px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }
  </style>
</head>
<body>
  <div id="header">
    <h1>Healthcare Accessibility &amp; Risk — Austria Knowledge Graph</h1>
    <p>Districts coloured by risk level. Click any marker for details.</p>
  </div>
  <div id="map"></div>
  <div id="legend">
    <b>Districts:</b>
    <span><span class="dot" style="background:#e53e3e"></span>High Risk</span>
    <span><span class="dot" style="background:#ed8936"></span>Medium Risk</span>
    <span><span class="dot" style="background:#38a169"></span>Low Risk</span>
    <span style="margin-left:1rem"><b>Facilities:</b></span>
    <span>🏥 Hospital</span><span>👨‍⚕️ GP</span><span>💊 Pharmacy</span>
  </div>
  <script>
    const map = L.map('map').setView([47.8, 13.5], 7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 18
    }).addTo(map);

    fetch('/api/districts').then(r => r.json()).then(data => {
      data.forEach(d => {
        if (d.lat == null) return;
        const risk  = d.risk || 'Unknown';
        const color = risk === 'HighRisk' ? '#e53e3e' : risk === 'MediumRisk' ? '#ed8936' : '#38a169';
        const vuln  = d.vuln  ? parseFloat(d.vuln).toFixed(3)  : '—';
        const gp    = d.gpPer1000 ? parseFloat(d.gpPer1000).toFixed(3) : '—';
        L.circleMarker([d.lat, d.lon], {
          radius: 15, color: '#555', weight: 1, fillColor: color, fillOpacity: 0.55
        }).addTo(map).bindPopup(
          `<b>${d.name || d.district_id}</b><br>
           Risk: <b style="color:${color}">${risk}</b><br>
           Vulnerability score: ${vuln}<br>
           GPs / 1,000 residents: ${gp}<br>
           Population: ${d.pop || '—'}`
        );
      });
    }).catch(e => console.error('Districts:', e));

    fetch('/api/facilities').then(r => r.json()).then(data => {
      data.forEach(f => {
        if (f.lat == null) return;
        const icon = f.type === 'Hospital' ? '🏥' : f.type === 'GeneralPractitioner' ? '👨‍⚕️' : '💊';
        const div  = L.divIcon({ html: `<div style="font-size:22px;line-height:1">${icon}</div>`,
                                  className: '', iconAnchor: [11, 11] });
        L.marker([parseFloat(f.lat), parseFloat(f.lon)], { icon: div })
         .addTo(map)
         .bindPopup(`${icon} <b>${f.name || f.fac}</b><br><i>${f.type}</i>`);
      });
    }).catch(e => console.error('Facilities:', e));
  </script>
</body>
</html>"""
    return render_template_string(html)


@app.route("/api/districts")
def list_districts():
    q = """
    PREFIX hkg: <http://healthcare-kg.at/ontology#>
    PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    SELECT ?district ?name ?risk ?vuln ?gpPer1000 ?pop ?lat ?lon
    WHERE {
        ?district a hkg:District .
        OPTIONAL { ?district hkg:districtName       ?name . }
        OPTIONAL { ?district hkg:accessRisk         ?risk . }
        OPTIONAL { ?district hkg:vulnerabilityScore ?vuln . }
        OPTIONAL { ?district hkg:gpPer1000          ?gpPer1000 . }
        OPTIONAL { ?district hkg:population         ?pop . }
        OPTIONAL { ?district geo:lat                ?lat . }
        OPTIONAL { ?district geo:long               ?lon . }
    }
    """
    rows = run_query(q)
    for r in rows:
        did = r.get("district", "").split("/")[-1]
        r["district_id"] = did
        r["risk"] = r["risk"].split("#")[-1] if r.get("risk") else None
    return jsonify(rows)


@app.route("/api/facilities")
def list_facilities():
    """
    Query each facility type separately to avoid BIND-inside-UNION
    (triggers a pyparsing bug in some rdflib installations).
    """
    HKG = "http://healthcare-kg.at/ontology#"
    GEO = "http://www.w3.org/2003/01/geo/wgs84_pos#"
    results = []

    for ftype in FACILITY_TYPES:
        q = f"""
        PREFIX hkg: <{HKG}>
        PREFIX geo: <{GEO}>
        SELECT ?fac ?name ?lat ?lon ?district
        WHERE {{
            ?fac a hkg:{ftype} .
            OPTIONAL {{ ?fac hkg:facilityName ?name . }}
            OPTIONAL {{ ?fac geo:lat          ?lat  . }}
            OPTIONAL {{ ?fac geo:long         ?lon  . }}
            OPTIONAL {{ ?fac hkg:inDistrict   ?district . }}
        }}
        """
        for row in run_query(q):
            row["type"] = ftype
            results.append(row)

    return jsonify(results)


@app.route("/api/accessibility/<district_id>")
def accessibility(district_id: str):
    duri = f"http://healthcare-kg.at/resource/district/{district_id}"
    HKG  = "http://healthcare-kg.at/ontology#"
    GEO  = "http://www.w3.org/2003/01/geo/wgs84_pos#"
    results = []

    for ftype in FACILITY_TYPES:
        q = f"""
        PREFIX hkg: <{HKG}>
        PREFIX geo: <{GEO}>
        SELECT ?fac ?name ?lat ?lon
        WHERE {{
            ?fac a hkg:{ftype} .
            OPTIONAL {{ ?fac hkg:facilityName ?name . }}
            OPTIONAL {{ ?fac geo:lat          ?lat  . }}
            OPTIONAL {{ ?fac geo:long         ?lon  . }}
        }}
        """
        g = get_graph()
        for row in run_query(q):
            furi = row.get("fac", "")
            row["type"] = ftype
            # NOTE: bool(g.query(ask)) reads the ASK result correctly. An
            # earlier version wrapped it as bool(list(g.query(ask))) instead,
            # which is always True for an ASK query -- list(result) yields a
            # single-element list ([True] or [False]) and a non-empty list is
            # always truthy, so every facility was reported reachable from
            # every district regardless of the real answer. This surfaced
            # once districts/facilities covered three states rather than a
            # small Vienna-only sample, where the bug's effect (every distant
            # facility marked reachable) became obviously wrong.
            row["reachableIn15min"] = bool(g.query(
                f"ASK {{ <{furi}> <{HKG}reachableIn15min> <{duri}> }}"))
            row["reachableIn30min"] = bool(g.query(
                f"ASK {{ <{furi}> <{HKG}reachableIn30min> <{duri}> }}"))
            row["reachableIn60min"] = bool(g.query(
                f"ASK {{ <{furi}> <{HKG}reachableIn60min> <{duri}> }}"))
            if any([row["reachableIn15min"], row["reachableIn30min"], row["reachableIn60min"]]):
                results.append(row)

    return jsonify(results)


@app.route("/api/risk/<district_id>")
def risk(district_id: str):
    duri = f"http://healthcare-kg.at/resource/district/{district_id}"
    q = f"""
    PREFIX hkg: <http://healthcare-kg.at/ontology#>
    SELECT ?risk ?vuln ?gpPer1000 ?gpDeficit ?name ?pop
    WHERE {{
        <{duri}> a hkg:District .
        OPTIONAL {{ <{duri}> hkg:districtName       ?name      . }}
        OPTIONAL {{ <{duri}> hkg:accessRisk         ?risk      . }}
        OPTIONAL {{ <{duri}> hkg:vulnerabilityScore ?vuln      . }}
        OPTIONAL {{ <{duri}> hkg:gpPer1000          ?gpPer1000 . }}
        OPTIONAL {{ <{duri}> hkg:gpDeficit          ?gpDeficit . }}
        OPTIONAL {{ <{duri}> hkg:population         ?pop       . }}
    }}
    """
    rows = run_query(q)
    for r in rows:
        r["risk"] = r["risk"].split("#")[-1] if r.get("risk") else None
    return jsonify(rows)


@app.route("/api/underserved")
def underserved():
    pred_file = MODELS_DIR / "transe" / "underserved_predictions.json"
    if pred_file.exists():
        with open(pred_file) as f:
            return jsonify(json.load(f))
    q = """
    PREFIX hkg: <http://healthcare-kg.at/ontology#>
    SELECT ?district ?name ?vuln ?gpPer1000
    WHERE {
        ?district a hkg:District ;
                  hkg:accessRisk hkg:HighRisk .
        OPTIONAL { ?district hkg:districtName       ?name      . }
        OPTIONAL { ?district hkg:vulnerabilityScore ?vuln      . }
        OPTIONAL { ?district hkg:gpPer1000          ?gpPer1000 . }
    }
    ORDER BY DESC(?vuln)
    """
    return jsonify(run_query(q))


@app.route("/api/sparql", methods=["POST"])
def sparql_proxy():
    body  = request.get_json(silent=True) or {}
    query = body.get("query", "")
    if not query:
        abort(400, "Missing 'query' field in JSON body")
    try:
        return jsonify({"results": run_query(query), "count": len(run_query(query))})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    log.info(f"Starting Healthcare KG API on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
