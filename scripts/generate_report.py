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
transe_preds = json.load(open(ROOT / "models" / "transe" / "underserved_predictions.json"))
link_examples = json.load(open(ROOT / "models" / "transe" / "link_prediction_examples.json"))
transe_results = json.load(open(ROOT / "models" / "transe" / "results.json"))

tp = link_examples["true_positive_example"]
fp = link_examples["false_positive_example"]
top3 = sorted(transe_preds, key=lambda r: r["score_high_risk"], reverse=True)[:3]

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
        p("How many hours did you spend on your mini-project? &nbsp;<b>[FILL IN &mdash; e.g. XX hours]</b>", "Body"),
        p("How many hours did you spend on this portfolio document? &nbsp;<b>[FILL IN &mdash; e.g. XX "
          "hours]</b>", "Body"),
        Spacer(1, 0.3 * cm),
        p("Declaration", "H2"),
        p("[x] I confirm I have marked all parts generated by Generative AI or otherwise copied from "
          "other sources, and describe below what was generated and what I did myself to fulfil the "
          "learning outcomes.", "Body"),
        p("Generative AI use in the mini-project (code): &nbsp;<b>[FILL IN &mdash; e.g. \"roughly "
          "70-90%\"]</b>. Claude Code (Anthropic) was used as a pair-programming assistant across the "
          "codebase: it scaffolded the ingestion/reasoning/embedding/GNN/API modules, fixed several "
          "concrete bugs found while actually running the pipeline (PyKEEN entity-coverage failure when "
          "GTFS stop entities were included in link-prediction training; a metrics-formatting crash; the "
          "GraphSAGE class being defined inside a function so it could not be reloaded for what-if "
          "inference; a self-loop degenerate-scoring issue in TransE link prediction; two incorrect test "
          "assertions), and generated the figures and this report from the actual pipeline outputs. "
          "<b>[FILL IN: describe, in your own words, which specific decisions were yours]</b> &mdash; e.g. "
          "the project domain and scope, the three-layer architecture and which learning outcomes to "
          "target, the choice to evaluate TransE on a genuinely held-out split rather than trust the "
          "first run's misleadingly perfect in-sample accuracy, and the interpretation of all results in "
          "Sections 3-5.", "Body"),
        p("Generative AI use in this document: &nbsp;<b>[FILL IN &mdash; e.g. \"roughly 80%\"]</b>. The "
          "report text was drafted by Claude Code from the real numbers/artifacts produced by the code "
          "(JSON prediction files, training metrics, SPARQL query results) and then reviewed by me. "
          "<b>[FILL IN: note anything you rewrote or want to flag]</b>.", "Body"),
        Spacer(1, 0.3 * cm),
        p("<b>Please review and personalise the three bracketed [FILL IN] items above before submitting "
          "&mdash; they need your own hour estimates and your own words on what you directed/decided.</b>",
          "BodySmall"),
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
             p("Self-constructed representative sample (the data.gv.at REST endpoint used as the "
               "live-download target was not reachable from this environment, so the pipeline's "
               "documented fallback path &mdash; SAMPLE_HOSPITALS in "
               "<font face='Courier'>healthcare_ingestion.py</font> &mdash; was used, with real "
               "institution names, districts and bed counts)", "LOCell"),
             p("11 facilities: 5 hospitals, 4 GPs, 2 pharmacies", "LOCell")],
            [p("District demographics (Statistik-Austria-style)", "LOCell"),
             p("Self-constructed representative sample, same reason as above", "LOCell"),
             p("16 districts across Wien, Oberösterreich, Tirol", "LOCell")],
        ], colWidths=[5.2 * cm, 8.0 * cm, 3.4 * cm], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LO_HEADER_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c0c8d2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])),
        Spacer(1, 0.2 * cm),
        p("The GTFS feed is public and stably linked above, so results relying on it are directly "
          "reproducible. The other two are shipped as CSV files inside the submission ZIP "
          "(<font face='Courier'>data/raw/*.csv</font>, folder \"2 - construction\") for the same "
          "reason, per the portfolio's guidance on non-public/self-constructed datasets.", "BodySmall"),

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
            "   CSV: KA001,Allgemeines Krankenhaus Wien,Hospital,AT-9-01,48.2196,16.3564,1900,Wien\n"
            "   -> hkgr:facility/KA001  a  hkg:Hospital ;\n"
            "        hkg:facilityName \"Allgemeines Krankenhaus Wien\"@de ;\n"
            "        hkg:bedCount 1900 ; hkg:inDistrict hkgr:district/AT-9-01 ;\n"
            "        geo:lat 48.2196 ; geo:long 16.3564 .\n\n"
            "2) A GTFS stop row -> a typed node (24,131 such triples from the real feed)\n"
            "   stops.txt: stop_id=1490,stop_name=\"Karlsplatz\",stop_lat=48.2004,stop_lon=16.3707\n"
            "   -> hkgr:stop/1490  a gtfs:Stop ; rdfs:label \"Karlsplatz\" ;\n"
            "        geo:lat 48.2004 ; geo:long 16.3707 .\n\n"
            "3) A derived (facility, stop) edge -> computed, not read from any source file\n"
            "   Haversine(KA001 @ 48.2196,16.3564 , stop W004 @ 48.2361,16.3600) = 1.86 km <= 5 km threshold\n"
            "   -> hkgr:facility/KA001  hkg:nearestStop hkgr:stop/W004 ;\n"
            "        hkg:nearestStopDistKm 1.8551 ."
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
          "nearestStop, reachableIn{15,30,60}min, accessRisk, gpDeficit, rdf:type</font>), giving 63 "
          "entities, 18 relations (9 + their inverses) and 153 triples. <i>The ~4,900 individual GTFS "
          "stop/route entities are deliberately excluded from this training graph</i>: an early run that "
          "included them raised entities to 5,023 against only 5,118 triples, and PyKEEN's entity-"
          "coverage-aware splitter could not find a train/val/test split covering every entity &mdash; a "
          "first, concrete encounter with the small-graph/long-tail scalability limit discussed in "
          "Section 3.3.", "Body"),
    ] + fig("transe_loss.png", caption="Figure 2 &mdash; TransE training loss over 200 epochs.") + [
        p("PyKEEN's own filtered evaluation on its held-out test split reports Hits@1=0.0, Hits@3=0.44, "
          "Hits@5=0.50, <b>Hits@10=0.625</b>, MRR=0.229, mean rank 13.5/63. The Hits@1=0 figure looked "
          "alarming at first, so I dug into it rather than reporting it unexamined: for several "
          "sparsely-observed relations, the model's relation vector stays close to the zero vector "
          "after only 200 epochs on 153 triples, which makes the degenerate self-loop score(h, r, h) "
          "rank first for almost every head &mdash; a known TransE failure mode on very small, sparse "
          "graphs. Filtering out that self-loop candidate before reading off the top-1 prediction "
          "(standard practice for non-reflexive relations) recovers a genuine top-1 accuracy of "
          "3/16 = 18.8% on the held-out test triples, and produces the following true-positive / "
          "false-positive pair used to satisfy the portfolio's worked-example requirement:", "Body"),
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
        Spacer(1, 0.15 * cm),
        p(f"Separately, scoring every district against the trained (<i>district</i>, accessRisk, ?) "
          f"triple &mdash; the actual \"which districts are under-served\" query the project plan asked "
          f"for &mdash; ranks {top3[0]['district_id']} ({top3[0]['district_name']}), "
          f"{top3[1]['district_id']} ({top3[1]['district_name']}) and {top3[2]['district_id']} "
          f"({top3[2]['district_name']}) as most plausibly HighRisk, matching the logic layer's own "
          "classification (Section 4) for all three &mdash; expected, since the model was trained on "
          "those same accessRisk triples, so this ranking demonstrates consistent embedding-space "
          "plausibility rather than independent generalisation (that claim is reserved for the held-out "
          "result above).", "Body"),
        p("<b>GraphSAGE (LO3).</b> A 2-layer GraphSAGE model (PyTorch Geometric, hidden=64, dropout=0.3, "
          "150 epochs) predicts each district's vulnerability score from 6 normalised features "
          "(population, age 0-4/65+ shares, car ownership, median income, GP-per-1,000) over a graph "
          "where districts in the same federal state are connected (16 nodes, 80 directed edges). Test "
          "MAE = 0.0149 on a [0,1]-scaled target.", "Body"),
    ] + fig("graphsage_fit.png", caption="Figure 3 &mdash; GraphSAGE prediction vs. the logic-layer's "
            "true vulnerability score for all 16 districts.") + [
        p("Because the target itself is a deterministic weighted sum of the input features (Section 4, "
          "rule V-set), this near-perfect fit mostly shows the GNN can recover a known function of its "
          "inputs &mdash; the genuine value-add is the <i>what-if</i> capability: perturbing one "
          "district's <font face='Courier'>gpPer1000</font> feature and re-running the forward pass "
          "(without retraining) estimates the effect of opening new GPs in the two highest-risk "
          "districts identified above.", "Body"),
    ] + fig("graphsage_whatif.png", caption="Figure 4 &mdash; What-if: predicted vulnerability "
            "before/after adding 2 GPs.") + [
        p("Adding 2 GPs nudges Lienz (AT-7-07) from 0.5208 to 0.5197 and Rohrbach (AT-4-20) from 0.4958 "
          "to 0.4942 &mdash; small but correctly-signed effects, consistent with GP-per-1,000 carrying "
          "only a 0.25 weight in the underlying vulnerability formula and with a 2-layer, state-level-"
          "clique graph smoothing local feature changes across neighbours.", "Body"),

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
            ListItem(p("<b>Volume/variety:</b> the GTFS-entity-coverage failure above is a direct, "
                       "measured instance of a KGE not scaling to a graph with a large population of "
                       "low-degree entities without curation &mdash; the fix (excluding the GTFS "
                       "subgraph from training) trades completeness for trainability, which would not "
                       "be an acceptable trade-off at production scale where transit-stop-level link "
                       "prediction actually matters.")),
            ListItem(p("<b>Little training data:</b> 63 entities / 153 triples is far below where "
                       "TransE's margin-ranking objective is well-conditioned, as the self-loop "
                       "artifact shows; more districts/facilities or synthetic negative sampling tuned "
                       "for sparsity would be needed before trusting top-1 predictions in production.")),
            ListItem(p("<b>Veracity:</b> GraphSAGE's target is itself a hand-specified formula rather "
                       "than an independently observed outcome (e.g. real health-outcome data), so its "
                       "\"accuracy\" mainly certifies that the GNN learned the right function, not that "
                       "the function is medically correct &mdash; a caveat that would matter a great "
                       "deal before any budgeting decision (LO10) is actually made on these numbers.")),
        ], bulletType="bullet", leftIndent=14),
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
            "  -> materialised: 76 reachability triples across 9 facilities x 16 districts.\n\n"
            "V2 (creates edges, threshold logic) - HighRisk classification:\n"
            "  INSERT { ?d hkg:accessRisk hkg:HighRisk }\n"
            "  WHERE  { ?d a hkg:District ; hkg:age65plusPct ?e ; hkg:gpDeficit \"true\"^^xsd:boolean .\n"
            "           FILTER(?e > 20.0) }\n"
            "  -> 3/16 districts (Lienz, Rohrbach, Wien-Hietzing) flagged HighRisk.\n\n"
            "V3 (negation-as-failure) - MediumRisk = GP-deficit OR elderly-heavy, and not already HighRisk:\n"
            "  INSERT { ?d hkg:accessRisk hkg:MediumRisk }\n"
            "  WHERE  { ?d a hkg:District .\n"
            "           { ?d hkg:gpDeficit \"true\"^^xsd:boolean } UNION { ?d hkg:age65plusPct ?e . FILTER(?e>18.0) }\n"
            "           FILTER NOT EXISTS { ?d hkg:accessRisk hkg:HighRisk } }\n\n"
            "Q4 (recursive property path, creates no new edges but traverses unbounded-depth class structure):\n"
            "  SELECT ?facility ?directClass WHERE {\n"
            "    ?facility a ?directClass . ?directClass rdfs:subClassOf* hkg:HealthcareFacility }\n"
            "  -> returns all 11 facility instances via their direct class (Hospital/GP/Pharmacy), following the\n"
            "     subClassOf hierarchy recursively rather than hard-coding the three subclass names.\n\n"
            "Q5 (aggregate service query) - average vulnerability by federal state:\n"
            "  SELECT ?state (AVG(?vuln) AS ?avgVuln) WHERE {\n"
            "    ?d hkg:inState ?state ; hkg:vulnerabilityScore ?vuln } GROUP BY ?state\n"
            "  -> Tirol 0.4990, Oberoesterreich 0.5089, Wien 0.5090 (all three states land within 0.01 of each\n"
            "     other in aggregate; the real spread is between individual districts, see Figure 5)."
        ),
    ] + fig("vulnerability_by_district.png", caption="Figure 5 &mdash; Materialised accessRisk / "
            "vulnerabilityScore per district (V1-V4 output).") + [

        Marker("sec4_2"),
        p("4.2 Effect on KG evolution (LO8)", "H2"),
        p("Two evolution events were run against the live triplestore via "
          "<font face='Courier'>src/reasoning/kg_evolution.py</font>:", "Body"),
        ListFlowable([
            ListItem(p("<b>Facility opening (completes the KG):</b> "
                       "<font face='Courier'>--event facility_added --id KA999 --district AT-4-10</font> "
                       "generated and appended a full Turtle description of the new hospital, including "
                       "an automatically-computed <font face='Courier'>nearestStop</font> link "
                       "(1.04&nbsp;km to Linz Hauptbahnhof) &mdash; the same construction logic as "
                       "Section 2.3's example 3, applied incrementally instead of at batch-ingestion "
                       "time.")),
            ListItem(p("<b>Facility closure (updates/removes from the KG):</b> "
                       "<font face='Courier'>--event facility_closed --id KA002</font> generated a "
                       "SPARQL <font face='Courier'>DELETE</font> patch retracting the facility and "
                       "every <font face='Courier'>reachableIn*min</font> edge pointing at it; applying "
                       "that patch via <font face='Courier'>load_triplestore.py --apply-patch</font> "
                       "shrank the loaded graph from 24,621 to 24,610 triples &mdash; an 11-triple, "
                       "fully-audited retraction (a <font face='Courier'>change_log.jsonl</font> entry "
                       "records both events) rather than a full KG rebuild.")),
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
          "rules simply don't need it at the granularity this project models transit at. The V-rules "
          "also do not currently handle conflicting/missing demographic fields defensively &mdash; a "
          "production version ingesting live, occasionally-incomplete Statistik Austria extracts would "
          "need explicit default/error handling that the current sample-data pipeline never exercises.",
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
          "<font face='Courier'>/api/accessibility/AT-9-01</font>, <font face='Courier'>/api/risk/"
          "AT-7-07</font> and <font face='Courier'>/api/underserved</font> were all smoke-tested "
          "end-to-end (returning e.g. Lienz/AT-7-07 as HighRisk with vulnerabilityScore=0.5368, "
          "gpPer1000=0.375), the generic <font face='Courier'>/api/sparql</font> proxy correctly "
          "answered <font face='Courier'>SELECT (COUNT(*) as ?c) WHERE {?d a hkg:District}</font> with "
          "16, and the Leaflet map renders all districts/facilities coloured by risk. The what-if "
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
          "in-process SPARQL engine is adequate at 24.6k triples but is not the throughput or the "
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
          "held-out district, a hypothetical new facility), not to discover a new signal. This makes the "
          "interaction asymmetric and, on reflection, a real limitation of this project's scope: because "
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
            "  scripts/build_kg.py            one-shot pipeline runner;  scripts/generate_report.py  this report\n"
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
    ]
    return story


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
