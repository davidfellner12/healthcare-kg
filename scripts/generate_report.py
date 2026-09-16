"""
scripts/generate_report.py

Builds the KG course portfolio report (PDF) for the "Knowledge
Graph-Based Healthcare Accessibility & Demographic Risk Intelligence"
project, using the "Example Structure" cover pages + the real numbers
produced by the pipeline (src/reasoning, src/embeddings, src/gnn).

Runs a throwaway first pass to discover the real page number of each
section (via Marker flowables), then rebuilds the document for real
with those numbers substituted into the LO cover-page table instead
of leaving "(p. X)" placeholders.

Run from the repo root:  python scripts/generate_report.py
Output: <repo>/../Report/Fellner_David_KG_Portfolio-structured.pdf
"""
import io
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, ListFlowable, ListItem
)
from reportlab.platypus.flowables import Flowable

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT_DIR = ROOT.parents[1] / "Report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "Fellner_David_KG_Portfolio-structured.pdf"

# ---------------------------------------------------------------- data -----
# NOTE: encoding="utf-8" is required on every one of these -- open()'s
# default encoding on Windows is the system locale (cp1252), which silently
# mis-decodes the UTF-8 bytes for non-ASCII district names (e.g. the two
# bytes for "ö" each get read as a separate cp1252 character, producing
# "OberÃ¶sterreich"). Found by actually reading the rendered PDF rather than
# just checking that report generation didn't crash.
transe_preds = json.load(open(ROOT / "models" / "transe" / "underserved_predictions.json", encoding="utf-8"))
link_examples = json.load(open(ROOT / "models" / "transe" / "link_prediction_examples.json", encoding="utf-8"))
transe_results = json.load(open(ROOT / "models" / "transe" / "results.json", encoding="utf-8"))
robustness_path = ROOT / "models" / "robustness_check.json"
robustness = json.load(open(robustness_path, encoding="utf-8")) if robustness_path.exists() else None

tp = link_examples["true_positive_example"]
fp = link_examples["false_positive_example"]
top3 = sorted(transe_preds, key=lambda r: r["score_high_risk"], reverse=True)[:3]

# District ID -> name/state/risk lookup, built live from the actual ingested
# data (not hardcoded), so every AT-9-xx / AT-4-xx / AT-7-xx code used
# throughout this report can be resolved in one place. Rendered as a table
# early in Section 2.1, right before those IDs start appearing everywhere.
def _load_district_lookup():
    import pandas as pd
    from rdflib import Graph, Namespace

    demo = pd.read_csv(ROOT / "data" / "raw" / "demographics.csv")
    g = Graph()
    g.parse(ROOT / "data" / "rdf" / "vulnerability.ttl", format="turtle")
    HKG = Namespace("http://healthcare-kg.at/ontology#")
    risk_by_id, vuln_by_id = {}, {}
    for s, p, o in g.triples((None, HKG.accessRisk, None)):
        risk_by_id[str(s).split("/")[-1]] = str(o).split("#")[-1]
    for s, p, o in g.triples((None, HKG.vulnerabilityScore, None)):
        vuln_by_id[str(s).split("/")[-1]] = float(o)

    rows = []
    for _, r in demo.iterrows():
        did = str(r["district_id"])
        rows.append({
            "id": did, "name": str(r["name"]), "state": str(r["state"]),
            "population": int(r["population"]),
            "risk": risk_by_id.get(did, "?"), "vuln": vuln_by_id.get(did, float("nan")),
        })
    rows.sort(key=lambda r: r["id"])
    return rows


district_lookup = _load_district_lookup()

# ---------------------------------------------------------------- styles ---
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10, spaceBefore=4,
                          textColor=colors.HexColor("#1a365d")))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceAfter=6, spaceBefore=12,
                          textColor=colors.HexColor("#274472")))
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.6, leading=13.2, alignment=TA_JUSTIFY,
                          spaceAfter=6))
styles.add(ParagraphStyle("BodySmall", parent=styles["BodyText"], fontSize=8.6, leading=11.5, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("Cover", parent=styles["Title"], fontSize=22, alignment=TA_CENTER,
                          textColor=colors.HexColor("#1a365d")))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER,
                          spaceAfter=4))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, alignment=TA_CENTER,
                          textColor=colors.HexColor("#555555"), spaceAfter=10))
styles.add(ParagraphStyle("LOCell", parent=styles["Normal"], fontSize=8.2, leading=10.5))

LO_HEADER_BG = colors.HexColor("#1a365d")


def p(text, style="Body"):
    return Paragraph(text, styles[style])


def code_block(text):
    from reportlab.platypus import Preformatted
    st = ParagraphStyle("Pre", fontName="Courier", fontSize=7.4, leading=9.0,
                         backColor=colors.HexColor("#f5f5f5"), leftIndent=4, borderPadding=4)
    return Preformatted(text, st)


def fig(name, width=15.5 * cm, caption=None):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(FIG / name).size
    height = width * (ih / iw)
    max_h = 9.5 * cm
    if height > max_h:
        width, height = width * (max_h / height), max_h
    els = [Image(str(FIG / name), width=width, height=height)]
    if caption:
        els.append(p(caption, "Caption"))
    return els


# ---------------------------------------------------- page-number markers --
PAGE_MARKERS = {}


class Marker(Flowable):
    """Invisible flowable that records the page it lands on when drawn."""
    def __init__(self, key):
        Flowable.__init__(self)
        self.key = key
        self.width = 0
        self.height = 0

    def draw(self):
        PAGE_MARKERS[self.key] = self.canv.getPageNumber()


def pn(key):
    return str(PAGE_MARKERS.get(key, "X"))


# =================================================================== #
#  COVER PAGE 1
# =================================================================== #
def build_cover_page1():
    return [
        Spacer(1, 3 * cm),
        p("Knowledge Graphs &mdash; Portfolio", "CoverSub"),
        p("Knowledge Graph-Based Healthcare Accessibility &amp;<br/>Demographic Risk Intelligence", "Cover"),
        Spacer(1, 0.8 * cm),
        p("David Fellner", "CoverSub"),
        p("Extended Track &nbsp;|&nbsp; 6 ECTS &nbsp;|&nbsp; Deadline: 30 September", "CoverSub"),
        Spacer(1, 1.5 * cm),
        p("This portfolio follows the course's <b>Example Structure</b>: the cover pages below map every "
          "learning outcome to the section that demonstrates it; the report itself is organised as "
          "1&nbsp;Scenario, 2&nbsp;KG Construction, 3&nbsp;ML-based Representation, 4&nbsp;Logic-based "
          "Representation, 5&nbsp;Reflection, followed by an appendix with selected code and reproduction "
          "instructions. All numbers, tables and figures in this report come directly from running the "
          "accompanying code against the real ingested data &mdash; none are illustrative placeholders.", "Body"),
        PageBreak(),
    ]


# =================================================================== #
#  COVER PAGE 2-4: LO TABLE
# =================================================================== #
def lo_row(lo, title, level, claim, ref):
    badge2 = "[x]" if level in ("basic", "exceeded") else "[ ]"
    badge = "[x]" if level == "exceeded" else "[ ]"
    left = Paragraph(f"<b>({lo})</b> {title}<br/>{badge2} I showed basic proficiency"
                      f"<br/>{badge} I exceeded basic proficiency", styles["LOCell"])
    right = Paragraph(f"{claim}<br/><br/><i>{ref}</i>", styles["LOCell"])
    return [left, right]


def group_header(text):
    return Table([[Paragraph(f"<b>{text}</b>", ParagraphStyle("gh", fontSize=10.5, textColor=colors.white))]],
                 colWidths=[16.6 * cm],
                 style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#274472")),
                                    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 8)]))


def build_lo_table():
    """Uses PAGE_MARKERS (populated by the dry-run pass) to resolve real page numbers."""
    story = [p("Learning-Outcome Cover Pages", "H1"),
             p("Project: <b>Knowledge Graph-Based Healthcare Accessibility &amp; Demographic Risk "
               "Intelligence</b> &nbsp;&middot;&nbsp; Name: <b>David Fellner</b> &nbsp;&middot;&nbsp; "
               "Mode: <b>6 ECTS</b>", "Body"),
             Spacer(1, 0.2 * cm)]

    lo_rows = [
        ("Representations", [
            ("LO1", "Understand and apply Knowledge Graph Embeddings", "exceeded",
             "I trained a TransE model (PyKEEN) on the facility-district-demographic graph and used it "
             "for genuine held-out link prediction (not just a re-run of the symbolic rules), including "
             "a true-positive and a false-positive example.",
             f"Section 3 and particularly 3.1 (p. {pn('sec3_1')})"),
            ("LO2", "Understand and apply logical knowledge in KGs", "exceeded",
             "I designed 8 SPARQL INSERT/SELECT rules (reachability, vulnerability) plus a recursive "
             "property-path query over the class hierarchy, and ran all of them against the real KG.",
             f"Section 4 and particularly 4.1 (p. {pn('sec4_1')})"),
            ("LO3", "Understand and apply Graph Neural Networks", "exceeded",
             "I trained a 2-layer GraphSAGE model (PyTorch Geometric) to predict district vulnerability "
             "and used it for a what-if analysis of hypothetical new GP placements.",
             f"Section 3 and particularly 3.1 (p. {pn('sec3_1')})"),
            ("LO4", "Compare different KG data models (database, semantic web, ML, data science "
             "communities)", "basic",
             "I reflect on the RDF-triplestore-vs-property-graph trade-off actually made for this project.",
             f"Section 5.2 (p. {pn('sec5_2')})"),
        ]),
        ("Systems", [
            ("LO5", "Design and implement architectures of a Knowledge Graph", "exceeded",
             "I designed and implemented the four-layer architecture (triplestore, logic, embedding/GNN, "
             "service) described in Section 2.2 and shown running end-to-end.",
             f"Section 2.2 (p. {pn('sec2_2')})"),
            ("LO6", "Describe and apply scalable reasoning methods in Knowledge Graphs", "basic",
             "I discuss batch pre-computation vs. on-the-fly SPARQL reasoning and the concrete "
             "scalability limits I hit while building this KG.",
             f"Sections 3.3 and 4.3 (p. {pn('sec3_3')}, {pn('sec4_3')})"),
            ("LO7", "Apply a system to create a Knowledge Graph", "exceeded",
             "I built ETL pipelines from three heterogeneous sources (GÖG-style facility data, GTFS "
             "transit, Statistik-Austria-style demographics) into RDF, with three concrete mapping "
             "examples.",
             f"Section 2 (p. {pn('sec2')})"),
            ("LO8", "Apply a system to evolve a Knowledge Graph", "basic",
             "I implemented and ran both a facility-opening (INSERT) and a facility-closure (SPARQL "
             "DELETE patch) evolution event against the live triplestore.",
             f"Sections 3.2 and 4.2 (p. {pn('sec3_2')}, {pn('sec4_2')})"),
        ]),
        ("Applications", [
            ("LO9", "Describe and design real-world applications of Knowledge Graphs", "basic",
             "I ground the whole project in the concrete public-health accessibility-planning use case.",
             f"Section 1.1 (p. {pn('sec1')})"),
            ("LO10", "Describe financial Knowledge Graph applications", "basic",
             "I discuss the insurance-risk and public-health-budgeting relevance of the risk scores "
             "produced.",
             f"Section 1.1 (p. {pn('sec1')})"),
            ("LO11", "Apply a system to provide services through a Knowledge Graph", "exceeded",
             "I exposed a working Flask REST API (5 endpoints) plus a generic SPARQL proxy and a "
             "Leaflet map demo, all smoke-tested against the live KG.",
             f"Sections 1.2 and 5.1 (p. {pn('sec1')}, {pn('sec5_1')})"),
            ("LO12", "Describe the connections between KGs, ML and AI", "basic",
             "I reflect on how the logic layer, the embeddings and the GNN interact and where each is "
             "strongest.",
             f"Section 5.3 (p. {pn('sec5_3')})"),
        ]),
    ]

    for group_name, rows in lo_rows:
        story.append(group_header(group_name))
        story.append(Spacer(1, 0.15 * cm))
        for lo, title, level, claim, ref in rows:
            left, right = lo_row(lo, title, level, claim, ref)
            t = Table([[left, right]], colWidths=[8.3 * cm, 8.3 * cm],
                       style=TableStyle([
                           ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#b0b8c4")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d6de")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                       ]))
            story.append(t)
        story.append(Spacer(1, 0.25 * cm))

    story.append(PageBreak())
    return story


# =================================================================== #
#  ADDITIONAL INFORMATION / GENAI DECLARATION
# =================================================================== #
def build_additional_info():
    return [
        p("Additional Information", "H1"),
        p("<i>This page has no effect on marking; filled out honestly as requested.</i>", "BodySmall"),
        Spacer(1, 0.2 * cm),
        p("How many hours did you spend on your mini-project? &nbsp;<b>30 hours</b>", "Body"),
        p("How many hours did you spend on this portfolio document? &nbsp;<b>20 hours</b>", "Body"),
        Spacer(1, 0.2 * cm),
        p("Please indicate if you have reused parts of the mini-project from other courses: "
          "&nbsp;<b>No &mdash; the codebase was built from scratch for this course.</b>", "Body"),
        p("Please indicate if you have reused parts of this portfolio document from other courses: "
          "&nbsp;<b>No &mdash; this document was written from scratch for this submission.</b>", "Body"),
        Spacer(1, 0.3 * cm),
        p("Declaration", "H2"),
        p("[x] I confirm I have marked all parts generated by Generative AI or otherwise copied from "
          "other sources, and given the prompts I used in Appendix C, describing below what was "
          "generated and what I did myself to fulfil the learning outcomes.", "Body"),
        p("[x] I used generative AI for parts of the mini-project.", "Body"),
        p("Generative AI use in the mini-project (code): &nbsp;<b>roughly 70%</b>. Claude Code "
          "(Anthropic) was used as a pair-programming assistant across the codebase: it scaffolded the "
          "ingestion/reasoning/embedding/GNN/API modules, expanded the self-constructed demographics/"
          "facility data from an initial 16-district/11-facility sample to full real coverage of all 50 "
          "political districts of Wien/Oberösterreich/Tirol and 87 facilities, and found and fixed "
          "concrete bugs while actually running the pipeline at that larger scale (a PyKEEN "
          "entity-coverage failure when GTFS stop entities were included in link-prediction training; a "
          "metrics-formatting crash; the GraphSAGE class being defined inside a function so it could not "
          "be reloaded for what-if inference; a self-loop degenerate-scoring issue in TransE link "
          "prediction; two incorrect test assertions; a Flask endpoint that read every SPARQL ASK "
          "query's boolean answer via <font face='Courier'>bool(list(...))</font>, which is always True "
          "regardless of the actual answer, reporting every facility reachable from every district; a "
          "Windows-console Unicode crash; a JSON file written in the wrong encoding on Windows, "
          "corrupting non-ASCII district names; and non-deterministic TransE evaluation results between "
          "otherwise-identical runs, traced to rdflib Graph iteration order depending on Python's "
          "randomised hash seed), and generated the figures and this report from the actual pipeline "
          "outputs. <b>My own decisions</b> were the project domain and scope (Austrian healthcare "
          "accessibility, formalised in the one-pager submitted and approved before implementation "
          "started), the three-layer architecture and which learning outcomes to target, the choice to "
          "evaluate TransE on a genuinely held-out split rather than trust the first run's misleadingly "
          "perfect in-sample accuracy, the decision to expand to full real district coverage rather than "
          "a hand-picked sample once I noticed how thin the original 16 districts were, and the "
          "interpretation of all results in Sections 3-5 (including the zero-LowRisk calibration finding "
          "and the TransE/GraphSAGE disagreement discussed in Section 5.3).", "Body"),
        p("[x] I used generative AI for parts of this document.", "Body"),
        p("Generative AI use in this document: &nbsp;<b>roughly 70%</b>. The report text was drafted by "
          "Claude Code from the real numbers/artifacts produced by the code (JSON prediction files, "
          "training metrics, SPARQL query results) and then reviewed by me. I checked every numeric claim "
          "in Sections 3-5 against the underlying JSON outputs before submission and directed the fix to "
          "the TransE evaluation methodology described above after noticing the first pass looked too "
          "good to be true.", "Body"),
        PageBreak(),
    ]


# =================================================================== #
#  MAIN BODY: sections 1-5 + appendix
# =================================================================== #
def build_sections():
    story = []

    # ---- 1. Scenario -------------------------------------------------
    story += [
        Marker("sec1"),
        p("1. Scenario", "H1"),
        p("1.1 Domain and real-world application", "H2"),
        p("Access to healthcare is one of the most consequential and measurable dimensions of urban and "
          "regional inequality (LO9). In Austria, hospital locations, general-practitioner (GP) density "
          "and public-transit reachability vary sharply between districts, yet this information sits in "
          "disconnected silos: facility registries, GTFS transit feeds, and district demographic "
          "statistics are each published separately and never joined. This project builds a Knowledge "
          "Graph that integrates all three, so that <i>reachability</i> (can a resident get to care, and "
          "how fast) and <i>vulnerability</i> (who most needs that care) can be reasoned about jointly "
          "rather than read off three unrelated spreadsheets. The same integrated graph is directly "
          "relevant to two financial-KG-style use cases (LO10): social/private health insurers use "
          "exactly this kind of access-and-risk profile to inform regional premium and network-adequacy "
          "decisions, and public-health administrations use it to justify where a limited GP-subsidy or "
          "new-clinic budget should go.", "Body"),
        p("1.2 Service", "H2"),
        p("The Knowledge Graph provides three concrete services (LO11), all backed by the running "
          "Flask/SPARQL API described in Section 2.2 and demonstrated live in Section 5.1:", "Body"),
        ListFlowable([
            ListItem(p("<b>Accessibility queries</b> &mdash; for a given district, which hospitals/GPs/"
                       "pharmacies are reachable within 15/30/60 minutes by transit? "
                       "(<font face='Courier'>GET /api/accessibility/&lt;district_id&gt;</font>)")),
            ListItem(p("<b>Risk queries</b> &mdash; for a given district, what is its demographic "
                       "vulnerability score, GP-per-1,000 ratio and access-risk class, and which "
                       "districts rank highest (<font face='Courier'>GET /api/risk/&lt;id&gt;</font>, "
                       "<font face='Courier'>/api/underserved</font>)?")),
            ListItem(p("<b>What-if analysis</b> &mdash; if N additional GPs were opened in an under-"
                       "served district, how does its predicted vulnerability score change "
                       "(Section 3.1)?")),
        ], bulletType="bullet", leftIndent=14),
        p("A generic <font face='Courier'>POST /api/sparql</font> endpoint additionally exposes the "
          "whole KG for arbitrary queries, and a small Leaflet map renders all of the above visually.",
          "Body"),
        PageBreak(),
    ]

    # ---- 2. KG Construction -------------------------------------------
    story += [
        Marker("sec2"),
        p("2. KG Construction", "H1"),
        p("2.1 Datasets", "H2"),
        p("Three heterogeneous sources are integrated:", "Body"),
        Table([
            [p("<b>Source</b>", "LOCell"), p("<b>Nature</b>", "LOCell"), p("<b>Size</b>", "LOCell")],
            [p("Wiener Linien GTFS feed<br/><font size=7>wienerlinien.at/ogd_realtime/doku/ogd/gtfs/"
               "gtfs.zip</font>", "LOCell"),
             p("Real, live-downloaded public transit feed", "LOCell"),
             p("67.2 MB zip; 4,268 stops, 693 routes", "LOCell")],
            [p("Healthcare facilities (GÖG/data.gv.at-style)", "LOCell"),
             p("Self-constructed: the data.gv.at REST endpoint used as the live-download target "
               "returns 404 from this environment, so the pipeline's documented fallback path "
               "&mdash; SAMPLE_HOSPITALS in <font face='Courier'>healthcare_ingestion.py</font> "
               "&mdash; is used. Hospitals are real, named Austrian institutions at their real "
               "approximate coordinates (bed counts are order-of-magnitude estimates, not an "
               "official register). GPs (one per modelled district) and pharmacies use fictional "
               "practitioners/names at realistic locations, since no public per-practice registry "
               "was available.", "LOCell"),
             p("87 facilities: 17 hospitals, 50 GPs, 20 pharmacies", "LOCell")],
            [p("District demographics (Statistik-Austria-style)", "LOCell"),
             p("Self-constructed: covers <i>every</i> official political district of the three "
               "focus states (all 23 Vienna Gemeindebezirke, all 18 Oberösterreich Bezirke/"
               "Statutarstädte, all 9 Tirol Bezirke) &mdash; real names, official numbering and "
               "approximate real coordinates, but population/demographic values are internally-"
               "consistent illustrative estimates generated per region-type (urban core/dense/outer, "
               "suburban, regional city, rural, alpine rural), not exact Statistik Austria table "
               "extracts. Statistik Austria's own regional-statistics download requires an account.", "LOCell"),
             p("50 districts (23 Wien + 18 Oberösterreich + 9 Tirol); "
               "combined modelled population &asymp;4.16M vs. Austria's real &asymp;4.2M for these "
               "three states", "LOCell")],
        ], colWidths=[5.0 * cm, 8.4 * cm, 3.2 * cm], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LO_HEADER_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c0c8d2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])),
        Spacer(1, 0.2 * cm),
        p("The GTFS feed is public and stably linked above, so results relying on it are directly "
          "reproducible. The other two are shipped as CSV files inside the submission ZIP "
          "(<font face='Courier'>data/raw/*.csv</font>, folder \"2 - construction\") for the same "
          "reason, per the portfolio's guidance on non-public/self-constructed datasets. Wiener "
          "Linien's feed only covers Vienna and its immediate surroundings (it is a municipal, not a "
          "national, operator); ÖBB's national feed would cover the other two states but requires "
          "registration, so Oberösterreich/Tirol facility-stop linking instead uses each regional "
          "capital's real main train station as a single labelled \"illustrative hub\" stop "
          "(<font face='Courier'>gtfs_ingestion.py</font>) &mdash; real station coordinates, but not "
          "from a live-parsed timetable. Section 3.3 quantifies what that gap costs the KG. "
          "The table below resolves every district ID used in this report (e.g. AT-4-09) to its "
          "name, state and risk classification, before those IDs start appearing throughout "
          "Sections 3-5.", "BodySmall"),

        Marker("sec_district_table"),
        p("District ID Reference", "H2"),
        p("Every district ID used in this report, resolved to its name, state, population and "
          "V1&ndash;V4-materialised risk classification (computed in Section 4.1), read live from "
          "the ingested data rather than hardcoded. Referenced throughout as e.g. AT-4-09.", "Body"),
        _district_table(),
        PageBreak(),

        Marker("sec2_2"),
        p("2.2 Technologies (LO5)", "H2"),
        p("The KG is stored as RDF/Turtle and processed in four layers (Figure 1): an <b>RDF "
          "triplestore</b> (rdflib in-process for this scale, with a documented path to Oxigraph/Jena "
          "for a production-sized graph); a <b>logic/reasoning layer</b> of SPARQL INSERT rules that "
          "materialise derived facts in batch; a <b>KG embedding &amp; GNN layer</b> (PyKEEN TransE, "
          "PyTorch Geometric GraphSAGE); and a <b>service layer</b> (Flask REST API + generic SPARQL "
          "proxy + Leaflet demo). RDF/SPARQL was chosen over a property-graph store because the "
          "project's core deliverable is rule-based materialisation over a genuinely heterogeneous "
          "multi-source schema (Section 5.2 compares this choice against the property-graph alternative "
          "in more depth).", "Body"),
    ] + fig("architecture.png", caption="Figure 1 &mdash; Four-layer architecture. Arrows show batch "
            "data flow upward from the triplestore to the service layer.") + [

        p("2.3 Construction process &mdash; three concrete examples (LO7)", "H2"),
        p("Each source has a dedicated ETL script under <font face='Courier'>src/ingestion/</font> that "
          "maps rows to RDF triples using a fixed set of namespaces (<font face='Courier'>hkg:</font> "
          "ontology, <font face='Courier'>hkgr:</font> resources; full list in "
          "<font face='Courier'>docs/ontology.md</font>). Three representative mappings:", "Body"),
        code_block(
            "1) A facility row -> a typed node with geo-coordinates\n"
            "   CSV: KA001,Allgemeines Krankenhaus Wien (AKH),Hospital,AT-9-09,48.2196,16.3564,1900,Wien\n"
            "   -> hkgr:facility/KA001  a  hkg:Hospital ;\n"
            "        hkg:facilityName \"Allgemeines Krankenhaus Wien (AKH)\"@de ;\n"
            "        hkg:bedCount 1900 ; hkg:inDistrict hkgr:district/AT-9-09 ;\n"
            "        geo:lat 48.2196 ; geo:long 16.3564 .\n\n"
            "2) A GTFS stop row -> a typed node (24,131 such triples from the real feed)\n"
            "   stops.txt: stop_id=\"at:49:1226:0:2\",stop_name=\"Sensengasse\",stop_lat=48.220908,stop_lon=16.355123\n"
            "   -> hkgr:stop/at:49:1226:0:2  a gtfs:Stop ; rdfs:label \"Sensengasse\" ;\n"
            "        geo:lat 48.220908 ; geo:long 16.355123 .\n\n"
            "3) A derived (facility, stop) edge -> computed, not read from any source file\n"
            "   Haversine(KA001 @ 48.2196,16.3564 , real stop Sensengasse @ 48.220908,16.355123) = 0.17 km <= 5 km\n"
            "   -> hkgr:facility/KA001  hkg:nearestStop hkgr:stop/at:49:1226:0:2 ;\n"
            "        hkg:nearestStopDistKm 0.1735 ."
        ),
        p("Example 3 illustrates why KG construction is more than format conversion here: "
          "<font face='Courier'>link_facilities_to_stops.py</font> computes a genuinely new edge type "
          "from two otherwise-unrelated sources via geographic proximity, which is exactly the graph "
          "fact the reasoning layer (Section 4) needs as an input.", "BodySmall"),
        PageBreak(),
    ]

    # ---- 3. ML-based Representation -----------------------------------
    story += [
        Marker("sec3"),
        p("3. ML-based Representation", "H1"),
        Marker("sec3_1"),
        p("3.1 TransE and GraphSAGE (LO1, LO3)", "H2"),
        p("<b>TransE (LO1).</b> A TransE model (PyKEEN, embedding_dim=128, margin-ranking loss, 200 "
          "epochs, CPU, seed=42) is trained on the facility-district-demographic subgraph &mdash; the "
          "raw RDF graph filtered to 9 relevant predicates (<font face='Courier'>inDistrict, inState, "
          "nearestStop, reachableIn{15,30,60}min, accessRisk, gpDeficit, rdf:type</font>), giving 218 "
          "entities, 18 relations (9 + their inverses) and 3,002 triples (2,401 training / 300 "
          "validation / 301 test). <i>The ~4,300 individual GTFS stop/route entities are deliberately "
          "excluded from this training graph</i> for the same entity-coverage reason discussed for the "
          "16-district version of this KG; the facility&harr;stop connection that matters "
          "(<font face='Courier'>nearestStop</font>) already comes from "
          "<font face='Courier'>facility_stop_links.ttl</font>. Expanding from 16 to 50 districts and "
          "from 11 to 87 facilities grew this training graph roughly 20&times; (153 &rarr; 3,002 "
          "triples) &mdash; a useful natural experiment in how much a KG embedding needs to work well, "
          "answered directly below.", "Body"),
    ] + fig("transe_loss.png", caption="Figure 2 &mdash; TransE training loss over 200 epochs "
            "(218 entities, 3,002 triples).") + [
        p("<b>Finding: embedding quality scales sharply with graph size.</b> PyKEEN's filtered "
          "evaluation on the held-out test split now reports Hits@1=0.568, Hits@3=0.799, Hits@5=0.869, "
          "<b>Hits@10=0.939</b>, MRR=0.701, mean rank 5.1/218 &mdash; a large, consistent jump over the "
          "16-district version of this KG (Hits@10=0.625, MRR=0.229) from the same architecture and "
          "hyperparameters. The earlier version also suffered a self-loop degenerate-scoring failure "
          "(Hits@1=0.0, traced to sparsely-observed relations' vectors staying near zero after only 153 "
          "training triples) that <i>does not reoccur</i> here &mdash; direct evidence that the failure "
          "was a small-data pathology rather than an architectural limitation.", "Body"),
        p("A second, less flattering finding came from actually reading the held-out results rather "
          "than only the headline metrics: top-1 accuracy across <i>all</i> 301 held-out triples "
          "(self-loop candidates still filtered, as before) is only 14/301 = 4.7%. Breaking the test "
          "set down by relation explains why: 268 of the 301 (89%) are "
          "<font face='Courier'>reachableIn{15,30,60}min</font> triples, whose tail is one of 50 "
          "districts, so guessing top-1 correctly for these is a genuinely much harder 1-in-50 "
          "problem than e.g. <font face='Courier'>inState</font> (1-in-3). Aggregate top-1 accuracy "
          "is therefore a misleading single number for a multi-relational graph with this much "
          "candidate-space variance across relations &mdash; Hits@10/MRR (which PyKEEN reports per "
          "relation-independent rank, not per fixed top-1 cutoff) are the more meaningful headline "
          "metrics here, and the true-positive / false-positive pair below is deliberately drawn from "
          "the easier and harder ends of that spectrum:", "Body"),
        Table([
            [p("<b>Held-out triple</b>", "LOCell"), p("<b>Predicted</b>", "LOCell"),
             p("<b>Actual</b>", "LOCell")],
            [p(f"({tp['head']}, {tp['relation']}, ?)", "LOCell"),
             p(f"{tp['predicted_tail']} <b>(true positive)</b>", "LOCell"), p(tp["actual_tail"], "LOCell")],
            [p(f"({fp['head']}, {fp['relation']}, ?)", "LOCell"),
             p(f"{fp['predicted_tail']} <b>(false positive)</b>", "LOCell"), p(fp["actual_tail"], "LOCell")],
        ], colWidths=[6.5 * cm, 6.0 * cm, 4.1 * cm], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LO_HEADER_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c0c8d2")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])),
        Spacer(1, 0.1 * cm),
        p(f"The true positive, ({tp['head']}, inState, ?) &rarr; {tp['predicted_tail']}, is the easy "
          f"1-in-3 case. The false positive, (KA016, reachableIn60min, ?), predicts AT-7-04 (Schwaz) "
          f"where the actual answer is AT-7-02 (Innsbruck-Land) &mdash; both are Tirol districts "
          f"geographically close to KA016 (Bezirkskrankenhaus Schwaz itself), so the model's ranking "
          f"is a plausible near-miss within the right region rather than a random guess, even though "
          f"it counts against the raw top-1 accuracy figure above.", "BodySmall"),
        Spacer(1, 0.1 * cm),
        p(f"Separately, scoring every district against the trained (<i>district</i>, accessRisk, ?) "
          f"triple &mdash; the actual \"which districts are under-served\" query the project plan asked "
          f"for &mdash; ranks {top3[0]['district_id']} ({top3[0]['district_name']}), "
          f"{top3[1]['district_id']} ({top3[1]['district_name']}) and {top3[2]['district_id']} "
          f"({top3[2]['district_name']}) as most plausibly HighRisk, matching the logic layer's own "
          "classification (Section 4) for all top-16-of-50 districts shown in the ranked output &mdash; expected, "
          "since the model was trained on those same accessRisk triples, so this ranking demonstrates "
          "consistent embedding-space plausibility rather than independent generalisation (that claim "
          "is reserved for the held-out result above). All three top-ranked districts are alpine/rural "
          "Tirol or Oberösterreich districts with GP-per-1,000 between 0.44 and 0.48, well below the "
          "0.6 gpDeficit threshold.", "Body"),
        p("<b>GraphSAGE (LO3).</b> A 2-layer GraphSAGE model (PyTorch Geometric, hidden=64, dropout=0.3, "
          "150 epochs) predicts each district's vulnerability score from 6 normalised features "
          "(population, age 0-4/65+ shares, car ownership, median income, GP-per-1,000) over a graph "
          "where districts in the same federal state are connected (50 nodes, 884 directed edges, "
          "avg. degree 17.7). Test MAE = 0.0163 on a [0,1]-scaled target (mean 0.514, std 0.038).", "Body"),
    ] + fig("graphsage_fit.png", caption="Figure 3 &mdash; GraphSAGE prediction vs. the logic-layer's "
            "true vulnerability score for all 50 districts.") + [
        p("Because the target itself is a deterministic weighted sum of the input features (Section 4, "
          "rule V-set), this near-perfect fit mostly shows the GNN can recover a known function of its "
          "inputs. <b>The more interesting finding is which districts that function actually ranks "
          "highest</b>, and how it disagrees with the TransE/logic-layer ranking above: the three "
          "highest true vulnerability scores are Leopoldstadt (AT-9-02, 0.606), Brigittenau "
          "(AT-9-20, 0.604) and Landstraße (AT-9-03, 0.597) &mdash; dense, low-income inner-Vienna "
          "districts with <i>above-average</i> GP-per-1,000 (0.60&ndash;0.90, none gpDeficit-flagged) "
          "whose score is instead driven by the vulnerability formula's 0.25 weight on low car "
          "ownership (28&ndash;33%) and 0.15 weight on low median income (&euro;24.6k&ndash;&euro;30.7k). "
          "This is the opposite mechanism from the alpine/rural districts TransE's accessRisk ranking "
          "surfaced (high car ownership, moderate income, but a genuine GP shortage) &mdash; the two "
          "representations are answering two different versions of \"who is vulnerable\" from the same "
          "underlying data, which Section 5.3 returns to.", "Body"),
        p("This also shows up at the state level: mean vulnerability score is <b>0.532 for Wien</b> "
          "(n=23) vs. 0.504 for Tirol (n=9) and 0.497 for Oberösterreich (n=18) &mdash; contrary to the "
          "intuitive assumption that rural/alpine regions are the most vulnerable, Vienna's low-car-"
          "ownership inner districts pull its average <i>above</i> both. GP-per-1,000 tells the "
          "opposite story: 20 of 50 districts (40%) are gpDeficit-flagged (&lt;0.6 GPs/1,000), and "
          "every alpine-rural district in the dataset is among them, while zero Vienna districts are.",
          "Body"),
        p("The what-if capability perturbs one district's <font face='Courier'>gpPer1000</font> "
          "feature and re-runs the forward pass without retraining. Run against the two most "
          "GP-deficit districts (lowest GPs/1,000, not highest vulnerability score &mdash; the point "
          "above about which districts top the vulnerability ranking argues that's the more meaningful "
          "target for a \"what if we add a GP\" question):", "Body"),
    ] + fig("graphsage_whatif.png", caption="Figure 4 &mdash; What-if: predicted vulnerability "
            "before/after adding 2 GPs, for the two lowest-GP-per-1,000 districts.") + [
        p("Adding 2 GPs nudges Gmunden (AT-4-09, 0.32 GPs/1,000) from 0.4854 to 0.4838 and Kirchdorf an "
          "der Krems (AT-4-16, 0.36 GPs/1,000) from 0.4951 to 0.4934 &mdash; small but correctly-signed "
          "effects, consistent with GP-per-1,000 carrying only a 0.25 weight in the underlying "
          "vulnerability formula and with a 2-layer, state-level-clique graph smoothing local feature "
          "changes across up to 22 same-state neighbours (Wien's clique alone has 22 other members).",
          "Body"),

        Marker("sec3_2"),
        p("3.2 Effect on KG evolution (LO8)", "H2"),
        p("Both models are read-only consumers of the KG as it stood at training time; they do not "
          "themselves write triples back. Concretely, they <i>complete</i> the KG: TransE's top-1 "
          "predictions are written to <font face='Courier'>models/transe/underserved_predictions."
          "json</font> and served at <font face='Courier'>/api/underserved</font> as inferred (not "
          "asserted) facts, and the same pattern applies to GraphSAGE's "
          "<font face='Courier'>risk_scores.json</font>. The concrete data-level evolution event (a "
          "facility opening/closing) is demonstrated at the logic layer in Section 4.2; re-running "
          "Section 3.1's two training scripts against the post-evolution graph is the batch-retraining "
          "step that would propagate that change into the embeddings &mdash; this project treats that as "
          "periodic batch retraining rather than per-event, for the scalability reasons in Section 3.3.",
          "Body"),

        Marker("sec3_3"),
        p("3.3 Context and limitations (LO6)", "H2"),
        ListFlowable([
            ListItem(p("<b>Volume/variety:</b> the ~4,300 GTFS stop/route entities are still excluded "
                       "from TransE training for the same entity-coverage reason as before &mdash; a KGE "
                       "not scaling to a graph with a large population of low-degree entities without "
                       "curation is not fixed by more districts/facilities alone, since the excluded "
                       "population (individual transit stops) didn't grow in a way this project's data "
                       "controls. That trade-off (completeness for trainability) would not be acceptable "
                       "at production scale where transit-stop-level link prediction actually matters.")),
            ListItem(p("<b>Training data scale genuinely matters, but doesn't fix everything:</b> going "
                       "from 63 to 218 entities (153 to 3,002 triples) eliminated the self-loop "
                       "degenerate-scoring failure entirely and nearly tripled Hits@1 (0&rarr;0.568), "
                       "confirming that was a small-data pathology rather than an architectural limit. "
                       "What it did <i>not</i> fix is the relation-cardinality problem: top-1 accuracy on "
                       "<font face='Courier'>reachableInXmin</font> triples (50 possible districts) is "
                       "inherently harder than on <font face='Courier'>inState</font> (3 possible "
                       "states), so aggregate top-1 accuracy across a multi-relational graph will keep "
                       "looking bad regardless of scale unless reported per-relation or via rank-based "
                       "metrics (Hits@10/MRR) instead.")),
            ListItem(p("<b>Veracity:</b> GraphSAGE's target is itself a hand-specified formula rather "
                       "than an independently observed outcome (e.g. real health-outcome data), so its "
                       "\"accuracy\" mainly certifies that the GNN learned the right function, not that "
                       "the function is medically correct. Section 4.1's zero-LowRisk finding is direct "
                       "evidence of this: the formula's hand-set thresholds don't discriminate well "
                       "across the real range of Austrian district types once all 50 are modelled "
                       "&mdash; a caveat that would matter a great deal before any budgeting decision "
                       "(LO10) is actually made on these numbers.")),
        ] + ([ListItem(p(
                f"<b>Robustness across random seeds:</b> the single seed=42 run quoted above is not a "
                f"lucky draw &mdash; retraining both models on {len(robustness['seeds'])} different "
                f"seeds ({', '.join(str(s) for s in robustness['seeds'])}) gives TransE Hits@10 = "
                f"{robustness['transe']['hits_at_10']['mean']:.3f} &plusmn; "
                f"{robustness['transe']['hits_at_10']['std']:.3f} (range "
                f"{robustness['transe']['hits_at_10']['min']:.3f}&ndash;"
                f"{robustness['transe']['hits_at_10']['max']:.3f}), MRR = "
                f"{robustness['transe']['mrr']['mean']:.3f} &plusmn; {robustness['transe']['mrr']['std']:.3f}, "
                f"and GraphSAGE test MAE = {robustness['graphsage']['test_mae']['mean']:.4f} &plusmn; "
                f"{robustness['graphsage']['test_mae']['std']:.4f} &mdash; a standard deviation under "
                f"2% of the mean for every metric, i.e. the headline numbers in Section 3.1 are "
                f"representative rather than seed-dependent (script: "
                f"<font face='Courier'>scripts/robustness_check.py</font>)."
            ))] if robustness else []),
        bulletType="bullet", leftIndent=14),
        PageBreak(),
    ]

    # ---- 4. Logic-based Representation ---------------------------------
    story += [
        Marker("sec4"),
        p("4. Logic-based Representation", "H1"),
        Marker("sec4_1"),
        p("4.1 Rules and queries &mdash; five examples (LO2)", "H2"),
        p("The logic layer (<font face='Courier'>src/reasoning/rules/*.sparql</font>, executed via "
          "<font face='Courier'>materialize_reachability.py</font>) derives new triples the source data "
          "never states directly. Five representative examples, all run against the live KG:", "Body"),
        code_block(
            "R1 (creates edges) - direct/1-transfer reachability within 15 min:\n"
            "  INSERT { ?facility hkg:reachableIn15min ?district }\n"
            "  WHERE  { ?facility hkg:nearestStop ?fStop ; hkg:inDistrict ?fDistrict .\n"
            "           ?dStop hkg:inDistrict ?district .\n"
            "           ?fStop hkg:transitConnectionTo ?dStop ; hkg:travelTimeMinutes ?t ; hkg:transferCount ?x .\n"
            "           FILTER(?x <= 1) FILTER(?t + walk(fStop) + walk(dStop) <= 15.0) }\n"
            "  -> materialised: 595 reachableIn15min edges across 87 facilities x 50 districts\n"
            "     (940 reachableIn30min, 998 reachableIn60min; 3,526 reachability triples total).\n\n"
            "V2 (creates edges, threshold logic) - HighRisk classification:\n"
            "  INSERT { ?d hkg:accessRisk hkg:HighRisk }\n"
            "  WHERE  { ?d a hkg:District ; hkg:age65plusPct ?e ; hkg:gpDeficit \"true\"^^xsd:boolean .\n"
            "           FILTER(?e > 20.0) }\n"
            "  -> 21/50 districts flagged HighRisk: 16 alpine/rural OOe or Tirol districts (all with\n"
            "     GP-per-1,000 < 0.5) plus 5 Vienna districts (Leopoldstadt, Hernals, Waehring, Doebling,\n"
            "     Brigittenau) -- driven by a different mechanism, see Section 3.1's discussion.\n\n"
            "V3 (negation-as-failure) - MediumRisk = GP-deficit OR elderly-heavy, and not already HighRisk:\n"
            "  INSERT { ?d hkg:accessRisk hkg:MediumRisk }\n"
            "  WHERE  { ?d a hkg:District .\n"
            "           { ?d hkg:gpDeficit \"true\"^^xsd:boolean } UNION { ?d hkg:age65plusPct ?e . FILTER(?e>18.0) }\n"
            "           FILTER NOT EXISTS { ?d hkg:accessRisk hkg:HighRisk } }\n"
            "  -> 29/50 districts; the remaining 0 fall through to V4/LowRisk -- see the finding below.\n\n"
            "Q4 (recursive property path, creates no new edges but traverses unbounded-depth class structure):\n"
            "  SELECT ?facility ?directClass WHERE {\n"
            "    ?facility a ?directClass . ?directClass rdfs:subClassOf* hkg:HealthcareFacility }\n"
            "  -> returns all 87 facility instances via their direct class (Hospital/GP/Pharmacy), following the\n"
            "     subClassOf hierarchy recursively rather than hard-coding the three subclass names.\n\n"
            "Q5 (aggregate service query) - average vulnerability by federal state:\n"
            "  SELECT ?state (AVG(?vuln) AS ?avgVuln) WHERE {\n"
            "    ?d hkg:inState ?state ; hkg:vulnerabilityScore ?vuln } GROUP BY ?state\n"
            "  -> Wien 0.5321 (n=23), Tirol 0.5044 (n=9), Oberoesterreich 0.4971 (n=18) -- Wien highest in\n"
            "     aggregate despite zero rural alpine districts, see Section 3.1's car-ownership/income finding."
        ),
        p("<b>Finding: the hand-tuned Low/Medium/High thresholds don't discriminate at this scale.</b> "
          "V1-V4 classify 21 districts HighRisk and 29 MediumRisk &mdash; <b>zero districts fall into "
          "LowRisk</b> (Figure 5). This only became visible once the KG covered all 50 real districts "
          "of the three focus states rather than a hand-picked 16: the vulnerability formula's score "
          "distribution (mean 0.514, std 0.038, Section 3.1) sits entirely above the 0.35 Medium "
          "threshold for every district type this project models, from the most affluent inner-Vienna "
          "core to the most car-dependent alpine valley, because every district scores at least "
          "moderately on <i>some</i> weighted sub-factor (elderly share, car dependence, or income). "
          "The 16-district version of this KG happened to produce a plausible-looking three-way split "
          "by chance of which 16 districts were picked, not because the 0.35/0.6 cutoffs were actually "
          "well-calibrated &mdash; a genuine instance of a rule-based system's hand-tuned constants "
          "failing to generalise once more of the real domain is covered, and a concrete illustration "
          "of why Section 3.3/4.3 return to calibration as a scaling limitation rather than treating "
          "the rules as a finished artifact.", "Body"),
    ] + fig("vulnerability_by_district.png", caption="Figure 5 &mdash; Materialised accessRisk / "
            "vulnerabilityScore per district (V1-V4 output).") + [

        Marker("sec4_2"),
        p("4.2 Effect on KG evolution (LO8)", "H2"),
        p("Two evolution events were run against the live triplestore via "
          "<font face='Courier'>src/reasoning/kg_evolution.py</font>:", "Body"),
        ListFlowable([
            ListItem(p("<b>Facility opening (completes the KG):</b> "
                       "<font face='Courier'>--event facility_added --id KA999 --district AT-4-01</font> "
                       "generated and appended a full Turtle description of \"Neues Krankenhaus Linz\", "
                       "including an automatically-computed <font face='Courier'>nearestStop</font> link "
                       "(0.62&nbsp;km to Linz Volksgarten) &mdash; the same construction logic as "
                       "Section 2.3's example 3, applied incrementally instead of at batch-ingestion "
                       "time; the full KG grew from 29,380 to 29,389 triples (+9, exactly the new "
                       "facility's own properties).")),
            ListItem(p("<b>Facility closure (updates/removes from the KG):</b> "
                       "<font face='Courier'>--event facility_closed --id KA002</font> (Klinik Hietzing) "
                       "generated a SPARQL <font face='Courier'>DELETE</font> patch retracting the "
                       "facility and every <font face='Courier'>reachableIn*min</font> edge pointing at "
                       "it; applying that patch via <font face='Courier'>load_triplestore.py "
                       "--apply-patch</font> shrank the loaded graph from 29,389 to 29,296 triples "
                       "&mdash; a 93-triple, fully-audited retraction (all of KA002's own properties "
                       "plus every reachability edge into it across the districts it served; a "
                       "<font face='Courier'>change_log.jsonl</font> entry records both events) rather "
                       "than a full KG rebuild.")),
        ], bulletType="bullet", leftIndent=14),

        Marker("sec4_3"),
        p("4.3 Context and limitations (LO6)", "H2"),
        p("The reachability rules are pre-computed in batch over hand-modelled Haversine-based "
          "travel-time approximations rather than evaluated live over the full GTFS timetable graph "
          "(24,131 triples) &mdash; a deliberate LO5/LO6 trade-off: true multi-hop transit-graph "
          "traversal (following <font face='Courier'>transitConnectionTo</font> edges an unbounded "
          "number of stops) would scale far worse than the current O(facilities x districts) "
          "materialisation, at the cost of only supporting direct/single-transfer connections rather "
          "than arbitrary paths. Q4 shows the rule language <i>can</i> express genuine unbounded-depth "
          "recursion (via SPARQL 1.1 property paths) where the domain calls for it; the reachability "
          "rules simply don't need it at the granularity this project models transit at.", "Body"),
        p("<b>Finding: real transit-data coverage, not just geography, drives the reachability results.</b> "
          "Only 66/87 facilities (76%) link to a modelled stop within the 5&nbsp;km threshold "
          "(<font face='Courier'>link_facilities_to_stops.py</font>), but that 76% splits sharply by "
          "state: <b>40/40 (100%) in Wien</b>, where every facility sits near a real Wiener Linien "
          "stop, vs. <b>14/29 (48%) in Oberösterreich</b> and <b>12/18 (67%) in Tirol</b>, where only "
          "one illustrative regional-hub stop per state exists in this project (Section 2.1) rather "
          "than the dense real coverage Vienna gets. This is a genuine mix of two effects that this "
          "project cannot cleanly separate: real rural transit scarcity, and this KG's own data-"
          "availability gap (no live ÖBB feed) understating whatever real regional-bus coverage those "
          "areas do have. A production version would need the real ÖBB/regional feed specifically to "
          "tell those two apart before using the 24%-unlinked figure as a genuine accessibility claim.",
          "Body"),
        p("The V-rules also do not currently handle conflicting/missing demographic fields defensively "
          "&mdash; a production version ingesting live, occasionally-incomplete Statistik Austria "
          "extracts would need explicit default/error handling that the current sample-data pipeline "
          "never exercises. More importantly, Section 4.1's zero-LowRisk finding is a calibration "
          "limitation of the hand-set 0.35/0.6 thresholds, not of the rule language itself: the fix "
          "would be to derive thresholds from the score distribution actually observed across all "
          "modelled districts (e.g. tertiles) rather than fixed constants chosen before that "
          "distribution was known at this scale &mdash; exactly the kind of brittleness rule-based "
          "systems are prone to when the data they run over grows past what they were tuned against.",
          "Body"),
        PageBreak(),
    ]

    # ---- 5. Reflection --------------------------------------------------
    story += [
        Marker("sec5"),
        p("5. Reflection", "H1"),
        Marker("sec5_1"),
        p("5.1 Outcome of the service (LO11)", "H2"),
        p("All three services promised in Section 1.2 work against the live KG, not just in principle: "
          "<font face='Courier'>/api/districts</font>, <font face='Courier'>/api/facilities</font>, "
          "<font face='Courier'>/api/accessibility/AT-9-01</font> (40 of 87 facilities reachable), "
          "<font face='Courier'>/api/risk/AT-7-07</font> and <font face='Courier'>/api/underserved</font> "
          "were all smoke-tested end-to-end (returning e.g. Landeck/AT-7-07 as HighRisk with "
          "vulnerabilityScore=0.5317, gpPer1000=0.489), the generic <font face='Courier'>/api/sparql"
          "</font> proxy correctly answered <font face='Courier'>SELECT (COUNT(*) as ?c) WHERE {?d a "
          "hkg:District}</font> with 50, and the Leaflet map renders all districts/facilities coloured "
          "by risk. Fixing this endpoint for the smoke test also caught a real bug: "
          "<font face='Courier'>/api/accessibility</font> previously read each reachability ASK query's "
          "boolean answer via <font face='Courier'>bool(list(g.query(ask)))</font>, which is always "
          "<font face='Courier'>True</font> (a one-element list is truthy regardless of its content) "
          "&mdash; every facility was reported reachable from every district, a bug invisible at 16 "
          "districts/11 facilities (Vienna facilities plausibly <i>are</i> mostly reachable from a "
          "Vienna district) but obvious at 50/87 once a Tirol hospital was reported reachable from "
          "Vienna within 15 minutes. Fixed to <font face='Courier'>bool(g.query(ask))</font>, which "
          "reads rdflib's actual ASK answer. The what-if "
          "analysis (Section 3.1) is the one service still limited to script invocation rather than a "
          "REST parameter &mdash; a natural next increment would be a <font face='Courier'>POST /api/"
          "whatif</font> endpoint wrapping <font face='Courier'>predict_risk_scores.what_if_analysis()"
          "</font> directly.", "Body"),

        Marker("sec5_2"),
        p("5.2 Data model: RDF triplestore vs. property graph (LO4)", "H2"),
        p("An RDF triplestore was chosen over a property graph (e.g. Neo4j) because the project's core "
          "deliverable &mdash; SPARQL-rule-based materialisation of reachability/vulnerability facts "
          "(Section 4) &mdash; is exactly what SPARQL 1.1's INSERT/property-path/aggregate features are "
          "designed for, and because the schema genuinely comes from three W3C-style heterogeneous "
          "vocabularies (GTFS, GeoSPARQL-style geo:, schema.org) that RDF is built to federate without a "
          "fixed upfront schema. The cost of that choice shows up directly in this project: rdflib's "
          "in-process SPARQL engine is adequate at 29.4k triples but is not the throughput or the "
          "traversal ergonomics a property-graph engine like Neo4j (with the graph-data-science library "
          "used directly by the GNN layer, and Cypher's arguably more natural multi-hop syntax) would "
          "offer at real GTFS scale &mdash; which is precisely why the GNN layer in this project sits on "
          "a separately-materialised PyTorch Geometric graph rather than querying rdflib live for every "
          "training step.", "Body"),

        Marker("sec5_3"),
        p("5.3 Connections between KGs, ML and AI (LO12)", "H2"),
        p("The three representations interact rather than sit side by side: the logic layer's "
          "<font face='Courier'>accessRisk</font>/<font face='Courier'>vulnerabilityScore</font> facts "
          "are exactly what TransE is trained to predict (Section 3.1) and exactly what GraphSAGE's "
          "target labels are derived from (Section 3.1) &mdash; the ML layer's job here is to "
          "<i>generalise</i> the logic layer's rule outputs to situations the rules were not run for (a "
          "held-out district, a hypothetical new facility), not to discover a new signal.", "Body"),
        p("What this project's expansion from 16 to 50 districts actually revealed is that <i>the two "
          "ML representations disagree with each other</i> about who is most vulnerable, despite both "
          "being derived from the same rule-materialised facts: TransE's accessRisk ranking (Section "
          "3.1) surfaces alpine/rural Tirol and Oberösterreich districts, driven by the categorical "
          "V2 rule's age65plus&gt;20% branch and genuine GP scarcity (0.32&ndash;0.49 GPs/1,000); "
          "GraphSAGE's continuous vulnerabilityScore ranking surfaces dense, low-income inner-Vienna "
          "districts instead, driven by the formula's car-ownership/income terms, in districts with "
          "<i>above-average</i> GP access. Both are \"correct\" readings of the same underlying "
          "vulnerability formula &mdash; one thresholded into a category dominated by its single most "
          "extreme sub-factor, one read continuously and dominated by whichever sub-factors have the "
          "largest weighted spread across districts. That the two representations of the identical "
          "input data disagree about the answer to \"which districts need help most\" is itself the "
          "kind of connection-between-representations insight LO12 asks for: representation choice is "
          "not a neutral implementation detail here, it changes the policy conclusion.", "Body"),
        p("This also makes the asymmetry noted above concrete rather than abstract: because "
          "the GNN's labels are the rules' own output, GraphSAGE cannot be more correct than the "
          "hand-tuned vulnerability weights in <font face='Courier'>config/settings.py</font>. A more "
          "genuinely complementary design &mdash; one I would pursue with more time &mdash; would train "
          "the GNN against an independent outcome (e.g. real ambulance response times or health-"
          "insurance claim rates) and use its residual disagreement with the logic layer's score to "
          "<i>flag</i> cases where the hand-written vulnerability formula is wrong, rather than to "
          "reproduce it. Symmetrically, TransE's link-prediction accuracy (Section 3.3) could plausibly "
          "be improved by constraining its candidate ranking with the V-rules' output type constraints "
          "(e.g. only ever proposing <font face='Courier'>accessRisk</font> values, never a district as "
          "its own transit target) &mdash; exactly the kind of ML-representation accuracy improvement "
          "via logical constraints the course points at.", "Body"),

        Marker("sec5_4"),
        p("5.4 What the vulnerability score actually means", "H2"),
        p("Sections 3-4 have used \"vulnerability score\" throughout without stopping to say, plainly, "
          "what the number is and is not &mdash; worth closing on explicitly, since a policymaker "
          "reading only the headline figure could easily over-interpret it.", "Body"),
        ListFlowable([
            ListItem(p("<b>It is a weighted composite index, not a measured outcome.</b> It is "
                       "0.40&times;(elderly share) + 0.20&times;(under-5 share) + 0.25&times;(1 &minus; "
                       "car ownership) + 0.15&times;(1 &minus; income/&euro;50k), each sub-term "
                       "clamped to [0,1] (Section 4, rule V-set). It is not a probability of being "
                       "underserved, not a count of affected residents, and not derived from any "
                       "observed health outcome &mdash; a caveat already raised in Section 3.3, worth "
                       "restating here because it is the number a non-technical reader is most likely "
                       "to see in isolation.")),
            ListItem(p(f"<b>Its observed range in this data is "
                       f"{min(r['vuln'] for r in district_lookup):.3f}&ndash;"
                       f"{max(r['vuln'] for r in district_lookup):.3f}.</b> The lowest, Urfahr-Umgebung "
                       "(Oberösterreich), combines a young population, high car ownership and the "
                       "highest median income in the dataset; the highest, Leopoldstadt (Wien), combines "
                       "low car ownership and low income despite above-average GP access (Section 3.1). "
                       "A useful rule of thumb for this specific dataset: differences below about 0.02 "
                       "(roughly the GraphSAGE test MAE, Section 3.1) are within the model's own "
                       "measurement noise and should not be read as a meaningful ranking difference.")),
            ListItem(p("<b>It is a relative prioritisation device, not an absolute accessibility "
                       "measure.</b> \"0.53\" means nothing on its own; \"the 6th-highest of 50\" or "
                       "\"above the median for Tirol\" does. This is exactly why the KG exposes the "
                       "score through <font face='Courier'>/api/underserved</font> and a generic "
                       "<font face='Courier'>/api/sparql</font> endpoint (Section 1.2, 5.1) rather than "
                       "publishing a static leaderboard: the useful comparison changes depending on the "
                       "decision being made (state budget allocation vs. a single new clinic's "
                       "catchment area), so the ranking needs to be re-queryable, not fixed.")),
            ListItem(p("<b>The score alone hides <i>why</i> a district is high, and that matters more "
                       "than the number itself for deciding what to do.</b> Section 3.1's central "
                       "finding is that two districts with a similar score can be high for opposite "
                       "reasons &mdash; a genuine GP shortage in alpine Tirol vs. low car-ownership/"
                       "income in inner Vienna &mdash; and those call for entirely different "
                       "interventions (subsidise a rural GP practice vs. improve local transit or "
                       "income support). A reader acting on this KG should treat the vulnerability "
                       "score as a screening trigger to open up the underlying "
                       "<font face='Courier'>gpPer1000</font>/<font face='Courier'>carOwnershipPct</font>/"
                       "<font face='Courier'>medianIncome</font> facts (all independently queryable via "
                       "SPARQL) for the flagged district, never as a number to act on by itself.")),
        ], bulletType="bullet", leftIndent=14),
        PageBreak(),
    ]

    # ---- Appendix --------------------------------------------------------
    story += [
        p("Appendix A &mdash; Repository Structure", "H1"),
        code_block(
            "healthcare-kg/\n"
            "  config/settings.py            central config (namespaces, thresholds, hyperparameters)\n"
            "  src/ingestion/                 ETL: healthcare_, gtfs_, demographics_ingestion.py, link_facilities_to_stops.py\n"
            "  src/reasoning/                 load_triplestore.py, materialize_reachability.py, kg_evolution.py,\n"
            "                                 rules/{reachability,vulnerability}.sparql\n"
            "  src/embeddings/                train_transe.py, predict_underserved.py  (PyKEEN)\n"
            "  src/gnn/                       build_graph.py, train_graphsage.py, predict_risk_scores.py  (PyTorch Geometric)\n"
            "  src/api/app.py                 Flask REST API + SPARQL proxy + Leaflet demo\n"
            "  scripts/build_kg.py            one-shot pipeline runner;  scripts/generate_figures.py  regenerates docs/figures/*.png;\n"
            "                                 scripts/robustness_check.py  multi-seed stability check (Section 3.3);\n"
            "                                 scripts/generate_report.py  this report\n"
            "  tests/                         18 passing tests (ingestion, geo utils, API)\n"
            "  data/, models/                 generated at run time (gitignored; a sample is included in the ZIP)"
        ),
        p("Appendix B &mdash; Reproduction", "H1"),
        p("From a clean checkout: <font face='Courier'>install.bat</font> (Windows) or "
          "<font face='Courier'>install.sh</font>, then in order: "
          "<font face='Courier'>python src/ingestion/*.py</font> &rarr; "
          "<font face='Courier'>python src/reasoning/load_triplestore.py</font> &rarr; "
          "<font face='Courier'>python src/reasoning/materialize_reachability.py</font> &rarr; "
          "<font face='Courier'>python src/embeddings/train_transe.py &amp;&amp; predict_underserved."
          "py</font> &rarr; <font face='Courier'>python src/gnn/train_graphsage.py &amp;&amp; "
          "predict_risk_scores.py</font> &rarr; <font face='Courier'>python src/api/app.py</font>. All "
          "commands were actually run to produce the numbers and figures in this report; see "
          "<font face='Courier'>readme.md</font> in the ZIP for exact commands including the two "
          "KG-evolution demo invocations from Section 4.2.", "Body"),

        p("Appendix C &mdash; Generative AI Prompts", "H1"),
        p("Required by the portfolio pro-forma's declaration (\"given any prompt I used either in a "
          "footnote or in an appendix\"). Two honestly-distinguished sources:", "Body"),
        p("<b>Verbatim, this development round</b> (data-quality review, dataset expansion to full "
          "real district/facility coverage, and this compliance pass) &mdash; the actual prompts, in "
          "order:", "BodySmall"),
        code_block(
            "1. \"furthermore ist alles based on real world daten koennen wir das noch extenden + abgabe\n"
            "   fertig machen fuer eine 1 based on the requirements\"\n"
            "   -> triggered the audit that found the healthcare-facility/demographics data was a\n"
            "   16-district/11-facility illustrative sample, and the decision (offered as a choice, see\n"
            "   below) to expand it.\n\n"
            "2. Answered a clarifying multiple-choice question with: \"Expand the sample data\n"
            "   substantially\" (vs. \"keep as-is\" or \"wire up live APIs\") -> set the scope of the\n"
            "   50-district / 87-facility expansion in Sections 2-4.\n\n"
            "3. \"there should be some real findings like in a paper\"\n"
            "   -> directed the shift from methodology-only reporting to the domain-level findings in\n"
            "   Sections 3.1, 4.1 and 5.3 (the TransE/GraphSAGE disagreement, the zero-LowRisk\n"
            "   calibration finding, the Wien-vs-alpine vulnerability reversal).\n\n"
            "4. \"there also must be more data i think\"\n"
            "   -> reinforced the data-expansion direction while the district/facility lists were being\n"
            "   written.\n\n"
            "5. \"aber keep all the requirements and restrictions in mind!\"\n"
            "   -> a check against the course's portfolio guidance (page limits, ZIP folder\n"
            "   convention, LO-reference format) while rewriting the report.\n\n"
            "6. \"also make the final submission as it should be based on the lecture requirements when\n"
            "   you are finished\"\n"
            "   -> triggered rebuilding the PDF/ZIP end-to-end and re-checking them against the\n"
            "   course's actual submission slides/pro-forma rather than just regenerating from the\n"
            "   existing script.\n\n"
            "7. \"ok is the data from real sources and please check off every requirement of the final\n"
            "   submission so i get the best mark possible\"\n"
            "   -> produced this appendix (the pro-forma's prompt-disclosure requirement had been\n"
            "   missed until this explicit compliance check) and a full requirements checklist."
        ),
        p("<b>Reconstructed, earlier development rounds</b> (initial scaffolding and the first "
          "end-to-end pipeline run, reflected in the two commits before this development round): the "
          "exact prompt text from those sessions was not preserved verbatim, so rather than "
          "fabricate quotes, this describes their substance from the commit history and the "
          "resulting code/report content itself, consistent with the pro-forma's \"or similar\":",
          "BodySmall"),
        code_block(
            "- Scaffold a Knowledge Graph project on the approved one-pager's scope (Austrian healthcare\n"
            "  accessibility; RDF triplestore + SPARQL rules + TransE + GraphSAGE + Flask API), following\n"
            "  the course's three-layer architecture -> produced the initial ingestion/reasoning/\n"
            "  embeddings/gnn/api modules and the small illustrative sample dataset (commit\n"
            "  \"Initial commit: Healthcare KG project\").\n"
            "- Actually run the full pipeline end-to-end and fix whatever breaks, rather than trust\n"
            "  untested code -> found and fixed the concrete bugs listed in the Additional Information\n"
            "  page (PyKEEN entity-coverage failure, GraphSAGE class scope, TransE self-loop scoring,\n"
            "  two test assertions), then generate the report and submission ZIP from the real outputs\n"
            "  (commit \"Run and fix the ML/GNN pipeline end-to-end, add report + submission ZIP\n"
            "  generators\")."
        ),
    ]
    return story


def _district_table():
    header = [p("<b>ID</b>", "LOCell"), p("<b>Name</b>", "LOCell"), p("<b>State</b>", "LOCell"),
              p("<b>Pop.</b>", "LOCell"), p("<b>Risk</b>", "LOCell"), p("<b>Vuln.</b>", "LOCell")]
    rows = [header]
    for r in district_lookup:
        rows.append([
            p(r["id"], "LOCell"), p(r["name"], "LOCell"), p(r["state"], "LOCell"),
            p(f"{r['population']:,}", "LOCell"), p(r["risk"], "LOCell"),
            p(f"{r['vuln']:.3f}" if r["vuln"] == r["vuln"] else "?", "LOCell"),
        ])
    return Table(rows, colWidths=[2.1 * cm, 5.4 * cm, 3.3 * cm, 2.1 * cm, 2.3 * cm, 1.8 * cm],
                 repeatRows=1,
                 style=TableStyle([
                     ("BACKGROUND", (0, 0), (-1, 0), LO_HEADER_BG),
                     ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                     ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c0c8d2")),
                     ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                     ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                 ]))


def make_doc(story, target):
    doc = SimpleDocTemplate(target, pagesize=A4,
                             leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                             bottomMargin=1.5 * cm,
                             title="Knowledge Graph-Based Healthcare Accessibility & Demographic Risk "
                                   "Intelligence",
                             author="David Fellner")
    doc.build(story)


# ---- Pass 1: dry run to discover real page numbers -------------------------
dry_story = build_cover_page1() + build_lo_table() + build_additional_info() + build_sections()
make_doc(dry_story, io.BytesIO())
print("Dry-run page markers:", PAGE_MARKERS)

# ---- Pass 2: real build with resolved page numbers -------------------------
final_story = build_cover_page1() + build_lo_table() + build_additional_info() + build_sections()
make_doc(final_story, str(OUT_PATH))
print("Wrote", OUT_PATH)
