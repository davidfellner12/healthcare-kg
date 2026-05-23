# Healthcare KG — Ontology & Data Model

## Namespaces

| Prefix | URI | Usage |
|--------|-----|-------|
| `hkg:` | `http://healthcare-kg.at/ontology#` | Classes and properties |
| `hkgr:` | `http://healthcare-kg.at/resource/` | Individual entities |
| `geo:` | `http://www.w3.org/2003/01/geo/wgs84_pos#` | Latitude/longitude |
| `gtfs:` | `http://vocab.gtfs.org/terms#` | GTFS transit concepts |
| `dct:` | `http://purl.org/dc/terms/` | Dublin Core metadata |
| `schema:` | `https://schema.org/` | Schema.org terms |

---

## Class Hierarchy

```
owl:Thing
├── hkg:HealthcareFacility
│   ├── hkg:Hospital
│   ├── hkg:GeneralPractitioner
│   └── hkg:Pharmacy
├── hkg:District
├── hkg:Stop
├── hkg:Route
└── hkg:Trip
```

---

## Properties

### Facility properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `hkg:facilityId` | Facility | xsd:string | Unique identifier |
| `hkg:facilityName` | Facility | xsd:string | Name (lang=de) |
| `hkg:bedCount` | Hospital | xsd:integer | Number of beds |
| `hkg:inDistrict` | Facility | District | Located in district |
| `hkg:inState` | Facility | xsd:string | Federal state name |
| `hkg:nearestStop` | Facility | Stop | Nearest GTFS stop |
| `hkg:nearestStopDistKm` | Facility | xsd:decimal | Walking distance (km) |
| `hkg:addedAt` | Facility | xsd:dateTime | KG evolution timestamp |

### Reachability properties (materialised by rules)

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `hkg:reachableIn15min` | Facility | District | District reachable ≤15 min |
| `hkg:reachableIn30min` | Facility | District | District reachable ≤30 min |
| `hkg:reachableIn60min` | Facility | District | District reachable ≤60 min |
| `hkg:travelTimeToDistrict` | Facility | xsd:decimal | Approx travel time (min) |

### District demographic properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `hkg:districtId` | District | xsd:string | Official district code |
| `hkg:districtName` | District | xsd:string | Name (lang=de) |
| `hkg:stateName` | District | xsd:string | Federal state (lang=de) |
| `hkg:population` | District | xsd:integer | Total population |
| `hkg:age0to4pct` | District | xsd:decimal | Share aged 0–4 (%) |
| `hkg:age65plusPct` | District | xsd:decimal | Share aged 65+ (%) |
| `hkg:gpCount` | District | xsd:integer | Number of GPs |
| `hkg:gpPer1000` | District | xsd:decimal | GPs per 1,000 residents |
| `hkg:carOwnershipPct` | District | xsd:decimal | Car ownership rate (%) |
| `hkg:medianIncome` | District | xsd:decimal | Median household income (EUR) |

### Vulnerability properties (materialised by rules)

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `hkg:vulnerabilityScore` | District | xsd:decimal | Weighted score ∈ [0, 1] |
| `hkg:gpDeficit` | District | xsd:boolean | GP-per-1000 < 0.6 |
| `hkg:accessRisk` | District | Risk class | `hkg:HighRisk` / `MediumRisk` / `LowRisk` |

### Transit properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `geo:lat` | Stop | xsd:decimal | Latitude |
| `geo:long` | Stop | xsd:decimal | Longitude |
| `gtfs:code` | Stop | xsd:string | GTFS stop_id |
| `gtfs:shortName` | Route | xsd:string | Route short name |
| `gtfs:routeType` | Route | xsd:integer | GTFS route type code |
| `hkg:transitZone` | Stop | xsd:string | Fare zone |

---

## RDF Store Trade-offs (LO4)

### RDF Triplestore (chosen)

**Pros:**
- Native SPARQL support for complex graph queries
- Built-in reasoning/inference hooks (RDFS, OWL)
- Standard interchange format (Turtle, N-Triples)
- Federated query support (SPARQL 1.1)
- Well-suited for heterogeneous schema (healthcare + transit + demographics)

**Cons:**
- Slower write throughput than property graphs
- Limited support for graph algorithms (PageRank, community detection)
- Verbose serialisation

### Property Graph (neo4j / alternatives)

**Pros:**
- Efficient traversal algorithms (Cypher, GDS library)
- Better native GNN integration (via neo4j-graphdatascience)
- Faster bulk import

**Cons:**
- No native RDF/SPARQL support
- Schema must be pre-defined
- Federated queries require custom tooling

**Decision:** RDF triplestore selected because the project requires
SPARQL-based rule materialisation (LO2, LO6) and standard semantic
web interchange. PyG is used for the GNN layer independently.

---

## Named Graphs

| Graph URI | Contents |
|-----------|----------|
| `hkgr:graph/healthcare` | Facility instances |
| `hkgr:graph/transit` | GTFS stop/route instances |
| `hkgr:graph/demographics` | District demographic data |
| `hkgr:graph/reachability` | Materialised reachability facts |
| `hkgr:graph/vulnerability` | Materialised risk classifications |
| `hkgr:graph/changes` | KG evolution provenance |
