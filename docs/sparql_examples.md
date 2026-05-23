# SPARQL Query Examples

All queries assume the following prefix declarations:

```sparql
PREFIX hkg:  <http://healthcare-kg.at/ontology#>
PREFIX hkgr: <http://healthcare-kg.at/resource/>
PREFIX geo:  <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
```

---

## 1. List all hospitals with bed count and district

```sparql
SELECT ?hospital ?name ?beds ?district
WHERE {
    ?hospital a hkg:Hospital ;
              hkg:facilityName ?name ;
              hkg:inDistrict   ?district .
    OPTIONAL { ?hospital hkg:bedCount ?beds . }
}
ORDER BY DESC(?beds)
```

---

## 2. Districts with high vulnerability and GP deficit

```sparql
SELECT ?district ?name ?vuln ?gpPer1000
WHERE {
    ?district a hkg:District ;
              hkg:vulnerabilityScore ?vuln ;
              hkg:gpPer1000 ?gpPer1000 ;
              hkg:gpDeficit "true"^^xsd:boolean .
    OPTIONAL { ?district hkg:districtName ?name . }
    FILTER (?vuln > 0.5)
}
ORDER BY DESC(?vuln)
```

---

## 3. Facilities reachable within 30 minutes from district Lienz

```sparql
SELECT ?facility ?name ?type
WHERE {
    ?facility hkg:reachableIn30min hkgr:district/AT-7-07 ;
              a ?type ;
              hkg:facilityName ?name .
    ?type rdfs:subClassOf* hkg:HealthcareFacility .
}
ORDER BY ?type
```

---

## 4. GP-per-1000 across all Vienna districts (for bar chart)

```sparql
SELECT ?name ?gpPer1000
WHERE {
    ?district a hkg:District ;
              hkg:stateName "Wien"@de ;
              hkg:gpPer1000 ?gpPer1000 .
    OPTIONAL { ?district hkg:districtName ?name . }
}
ORDER BY ?gpPer1000
```

---

## 5. High-risk districts NOT reachable to any hospital in 60 min

```sparql
SELECT ?district ?name ?risk
WHERE {
    ?district a hkg:District ;
              hkg:accessRisk hkg:HighRisk .
    OPTIONAL { ?district hkg:districtName ?name . }
    FILTER NOT EXISTS {
        ?hospital a hkg:Hospital ;
                  hkg:reachableIn60min ?district .
    }
}
```

---

## 6. Nearest stop for each facility and walking distance

```sparql
SELECT ?facility ?facilityName ?stop ?stopName ?distKm
WHERE {
    ?facility a ?t ;
              hkg:facilityName ?facilityName ;
              hkg:nearestStop ?stop ;
              hkg:nearestStopDistKm ?distKm .
    ?stop rdfs:label ?stopName .
    ?t rdfs:subClassOf* hkg:HealthcareFacility .
}
ORDER BY ?distKm
```

---

## 7. Count of facilities per district per type

```sparql
SELECT ?district ?districtName ?type (COUNT(?f) AS ?count)
WHERE {
    ?f a ?type ;
       hkg:inDistrict ?district .
    ?type rdfs:subClassOf* hkg:HealthcareFacility .
    OPTIONAL { ?district hkg:districtName ?districtName . }
}
GROUP BY ?district ?districtName ?type
ORDER BY ?district ?type
```

---

## 8. All triples about a specific facility

```sparql
DESCRIBE hkgr:facility/KA001
```

---

## 9. Risk classification distribution across states

```sparql
SELECT ?state ?risk (COUNT(?d) AS ?count)
WHERE {
    ?d a hkg:District ;
       hkg:stateName ?state ;
       hkg:accessRisk ?risk .
}
GROUP BY ?state ?risk
ORDER BY ?state ?risk
```

---

## 10. Compare vulnerability scores: Vienna vs Tyrol

```sparql
SELECT ?state (AVG(?vuln) AS ?avgVuln) (MIN(?vuln) AS ?minVuln) (MAX(?vuln) AS ?maxVuln)
WHERE {
    ?d a hkg:District ;
       hkg:stateName ?state ;
       hkg:vulnerabilityScore ?vuln .
    FILTER (?state IN ("Wien"@de, "Tirol"@de))
}
GROUP BY ?state
```
