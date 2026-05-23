# REST API Reference

Base URL: `http://localhost:5000`

All endpoints return JSON.

---

## GET /api/districts

Returns all districts with risk classification and vulnerability scores.

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `district` | string | Full URI of district resource |
| `name` | string | District name (German) |
| `risk` | string | `HighRisk`, `MediumRisk`, or `LowRisk` |
| `vuln` | decimal | Vulnerability score ∈ [0,1] |
| `gpPer1000` | decimal | GPs per 1,000 residents |
| `pop` | integer | Total population |
| `lat`, `lon` | decimal | Approximate centroid coordinates |

**Example:**
```bash
curl http://localhost:5000/api/districts
```

---

## GET /api/facilities

Returns all healthcare facilities with type and coordinates.

**Example:**
```bash
curl http://localhost:5000/api/facilities
```

---

## GET /api/accessibility/{district_id}

Returns which facilities are reachable from a given district within 15, 30, and 60 minutes.

**Path parameter:** `district_id` — district code, e.g. `AT-9-10`

**Example:**
```bash
curl http://localhost:5000/api/accessibility/AT-9-10
```

---

## GET /api/risk/{district_id}

Returns the risk profile for a specific district.

**Example:**
```bash
curl http://localhost:5000/api/risk/AT-7-07
```

**Response example:**
```json
[{
  "name": "Lienz",
  "risk": "HighRisk",
  "vuln": "0.6820",
  "gpPer1000": "0.3750",
  "gpDeficit": "true",
  "pop": "48000"
}]
```

---

## GET /api/underserved

Returns the top under-served districts ranked by vulnerability score.
Uses TransE predictions if the model has been trained; otherwise falls
back to SPARQL-based heuristics.

**Example:**
```bash
curl http://localhost:5000/api/underserved
```

---

## POST /api/sparql

Execute arbitrary SPARQL SELECT queries against the loaded KG.

**Request body:**
```json
{
  "query": "SELECT ?d ?name WHERE { ?d a <http://healthcare-kg.at/ontology#District> ; <http://healthcare-kg.at/ontology#districtName> ?name . }"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/sparql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT ?d WHERE { ?d a <http://healthcare-kg.at/ontology#District> }"}'
```

**Response:**
```json
{
  "results": [{"d": "http://healthcare-kg.at/resource/district/AT-9-01"}, ...],
  "count": 16
}
```

---

## GET /

Interactive map-based demo (HTML page). Opens in browser; shows district risk
as colour-coded circles (red = High, orange = Medium, green = Low) and facility
markers.
