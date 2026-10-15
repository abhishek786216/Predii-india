import csv
import html
import io
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.llm_extractor import generate_answer
from src.main import DEFAULT_PDF_PATH, index_pdf
from src.retriever import Retriever
from src.vector_store import VectorStore


app = Flask(__name__)
ARCHITECTURE_PATH = Path(__file__).with_name("architecture.drawio")
_index_lock = threading.Lock()
_indexed_pdf = None
_indexed_store = None
_indexed_stats = None


def get_indexed_store(pdf_path):
    """Reuse the expensive manual index for subsequent questions."""
    global _indexed_pdf, _indexed_store, _indexed_stats

    resolved_path = pdf_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    signature = (str(resolved_path), resolved_path.stat().st_mtime_ns)

    with _index_lock:
        if _indexed_pdf != signature:
            store = VectorStore()
            existing_chunks = store.collection.count()
            if existing_chunks:
                pages, chunks = 0, existing_chunks
            else:
                pages, chunks = index_pdf(store, resolved_path)
            _indexed_pdf = signature
            _indexed_store = store
            _indexed_stats = {"pages": pages, "chunks": chunks}

        return _indexed_store, _indexed_stats


def read_architecture_boxes():
    root = ET.parse(ARCHITECTURE_PATH).getroot()
    boxes = []
    for cell in root.findall(".//mxCell"):
        if cell.get("vertex") != "1" or cell.get("id") in {"title", "subtitle"}:
            continue
        raw_value = html.unescape(cell.get("value", ""))
        lines = [
            re.sub(r"<[^>]+>", "", line).strip()
            for line in raw_value.split("<br>")
        ]
        lines = [line for line in lines if line]
        if not lines:
            continue
        geometry = cell.find("mxGeometry")
        style = cell.get("style", "")
        fill_color = re.search(r"fillColor=([^;]+)", style)
        boxes.append(
            {
                "id": cell.get("id", ""),
                "title": lines[0],
                "details": lines[1:],
                "fill": fill_color.group(1) if fill_color else "#e8f2ff",
                "x": float(geometry.get("x", "0")) if geometry is not None else 0,
                "y": float(geometry.get("y", "0")) if geometry is not None else 0,
                "width": float(geometry.get("width", "200")) if geometry is not None else 200,
                "height": float(geometry.get("height", "100")) if geometry is not None else 100,
            }
        )
    boxes.sort(key=lambda box: (box["y"], box["x"]))
    return boxes


def as_csv(records):
    output = io.StringIO(newline="")
    fields = [
        "specification_name",
        "value",
        "unit",
        "vehicle_model",
        "part_number",
        "source_page",
        "source_text",
        "confidence",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/architecture")
def architecture_page():
    try:
        boxes = read_architecture_boxes()
        by_id = {box["id"]: box for box in boxes}
        groups = [
            {
                "number": "01",
                "label": "INGESTION",
                "description": "Turn the service manual into reliable searchable text.",
                "boxes": [by_id[box_id] for box_id in
                          ("input", "extract", "condition", "keep", "ocr", "compare", "merge")],
            },
            {
                "number": "02",
                "label": "KNOWLEDGE LAYER",
                "description": "Clean, split, embed, and store the manual evidence.",
                "boxes": [by_id[box_id] for box_id in ("clean", "chunk", "embed", "chroma")],
            },
            {
                "number": "03",
                "label": "QUESTION TO EVIDENCE",
                "description": "Find the most relevant chunks for a user question.",
                "boxes": [by_id[box_id] for box_id in ("query", "qembed", "retrieve")],
            },
            {
                "number": "04",
                "label": "TRUSTED OUTPUT",
                "description": "Generate, validate, and return auditable specifications.",
                "boxes": [by_id[box_id] for box_id in ("llm", "validate", "output")],
            },
        ]
        return render_template("architecture.html", groups=groups)
    except (ET.ParseError, OSError, KeyError) as exc:
        app.logger.exception("Architecture page could not be loaded")
        return jsonify(error=f"Could not load architecture page: {exc}"), 500


@app.get("/api/architecture")
def architecture():
    """Return the diagram's process boxes as data for the interactive UI."""
    try:
        boxes = read_architecture_boxes()
        return jsonify(
            title="Predii India RAG Architecture",
            subtitle="Interactive view generated from architecture.drawio",
            source=str(ARCHITECTURE_PATH.name),
            boxes=boxes,
        )
    except (ET.ParseError, OSError) as exc:
        app.logger.exception("Architecture diagram could not be loaded")
        return jsonify(error=f"Could not load architecture diagram: {exc}"), 500


@app.get("/architecture.svg")
def architecture_svg():
    try:
        boxes = {box["id"]: box for box in read_architecture_boxes()}
        positions = {
            "input": (60, 155), "extract": (390, 155), "condition": (720, 155),
            "keep": (390, 375), "ocr": (720, 375), "compare": (1050, 375),
            "merge": (1050, 565), "clean": (1050, 755), "chunk": (720, 755),
            "embed": (390, 755), "chroma": (60, 755), "query": (60, 1010),
            "qembed": (390, 1010), "retrieve": (720, 1010), "llm": (1050, 1010),
            "validate": (720, 1240), "output": (1050, 1240),
        }
        card_width, card_height = 270, 145

        def wrap(text, limit=31):
            lines, current = [], ""
            for word in text.split():
                if current and len(current) + len(word) + 1 > limit:
                    lines.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            if current:
                lines.append(current)
            return lines[:4]

        rendered_boxes = []
        for box_id, (x, y) in positions.items():
            box = boxes[box_id]
            detail = " ".join(box["details"])
            detail_lines = wrap(html.unescape(detail))
            detail_svg = "".join(
                f'<tspan x="{x + 22}" dy="{18 if index else 0}">'
                f'{html.escape(line)}</tspan>'
                for index, line in enumerate(detail_lines)
            )
            rendered_boxes.append(
                f'<g class="stage" tabindex="0"><title>{html.escape(box["title"])} — '
                f'{html.escape(detail)}</title><rect x="{x}" y="{y}" width="{card_width}" '
                f'height="{card_height}" rx="18" fill="{box["fill"]}" fill-opacity=".18" '
                f'stroke="{box["fill"]}" stroke-width="2"/>'
                f'<rect x="{x}" y="{y}" width="7" height="{card_height}" rx="4" '
                f'fill="{box["fill"]}"/><text x="{x + 22}" y="{y + 38}" '
                f'class="stage-title">{html.escape(box["title"])}</text>'
                f'<text x="{x + 22}" y="{y + 72}" class="stage-detail">{detail_svg}</text></g>'
            )

        def connector(source, target, label=""):
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            start_x, start_y = x1 + card_width / 2, y1 + card_height
            end_x, end_y = x2 + card_width / 2, y2
            if y1 == y2:
                start_x, start_y = x1 + card_width, y1 + card_height / 2
                end_x, end_y = x2, y2 + card_height / 2
            label_svg = (
                f'<text x="{(start_x + end_x) / 2}" y="{(start_y + end_y) / 2 - 8}" '
                f'class="connector-label">{html.escape(label)}</text>' if label else ""
            )
            return (
                f'<path d="M {start_x} {start_y} L {end_x} {end_y}" '
                f'class="connector" marker-end="url(#arrow)"/>{label_svg}'
            )

        connections = [
            ("input", "extract", ""), ("extract", "condition", ""),
            ("condition", "keep", "PDF text is sufficient"), ("condition", "ocr", "Needs OCR"),
            ("ocr", "compare", ""), ("compare", "merge", ""), ("keep", "clean", ""),
            ("merge", "clean", ""), ("clean", "chunk", ""), ("chunk", "embed", ""),
            ("embed", "chroma", ""), ("chroma", "query", "User asks a question"),
            ("query", "qembed", ""), ("qembed", "retrieve", ""), ("retrieve", "llm", ""),
            ("llm", "validate", ""), ("validate", "output", ""),
        ]
        rendered_connections = "".join(connector(*connection) for connection in connections)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1380 1450"
            role="img" aria-labelledby="title desc">
          <title id="title">Predii India RAG architecture</title>
          <desc id="desc">Hover over a stage for its responsibility. The diagram shows the
            document ingestion, retrieval, and structured output pipeline.</desc>
          <style>
            text {{ font-family: Inter, Arial, sans-serif; }}
            .stage {{ cursor: pointer; outline: none; }}
            .stage:hover rect:first-of-type, .stage:focus rect:first-of-type {{
              stroke: #58e6d2; stroke-width: 4; fill-opacity: .34;
              filter: drop-shadow(0 8px 12px #0008);
            }}
            .stage-title {{ fill: #edf3ff; font-size: 16px; font-weight: 800; }}
            .stage-detail {{ fill: #aab9d1; font-size: 12px; }}
            .connector {{ fill: none; stroke: #496887; stroke-width: 2;
              stroke-dasharray: 7 7; opacity: .8; }}
            .connector-label {{ fill: #58e6d2; font-size: 11px; font-weight: 700;
              text-anchor: middle; }}
          </style>
          <defs>
            <linearGradient id="background" x2="1" y2="1">
              <stop stop-color="#102847"/><stop offset=".55" stop-color="#07111f"/>
              <stop offset="1" stop-color="#040812"/>
            </linearGradient>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4"
              orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#58e6d2"/></marker>
          </defs>
          <rect width="1380" height="1450" fill="url(#background)"/>
          <circle cx="1260" cy="80" r="220" fill="#9b7bff" opacity=".09"/>
          <circle cx="80" cy="1320" r="250" fill="#58e6d2" opacity=".06"/>
          <text x="60" y="54" fill="#58e6d2" font-size="28" font-weight="900">
            PREDII INDIA | VEHICLE INTELLIGENCE ARCHITECTURE
          </text>
          <text x="60" y="84" fill="#91a0bc" font-size="15">
            A guided map of ingestion, semantic retrieval, and grounded generation
          </text>
          <text x="60" y="120" fill="#58e6d2" font-size="11" font-weight="800"
            letter-spacing="2">01 · INGESTION</text>
          <text x="60" y="975" fill="#58e6d2" font-size="11" font-weight="800"
            letter-spacing="2">02 · QUESTION TO EVIDENCE</text>
          <text x="60" y="1205" fill="#58e6d2" font-size="11" font-weight="800"
            letter-spacing="2">03 · TRUSTED OUTPUT</text>
          {rendered_connections}
          {"".join(rendered_boxes)}
        </svg>"""
        return app.response_class(svg, mimetype="image/svg+xml")
    except (ET.ParseError, OSError, ValueError) as exc:
        app.logger.exception("Architecture SVG could not be generated")
        return jsonify(error=f"Could not generate architecture SVG: {exc}"), 500


@app.post("/api/query")
def query():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query", "")).strip()
    output_format = payload.get("format", "json")
    top_k = int(payload.get("top_k", 10))
    pdf_path = Path(payload.get("pdf", str(DEFAULT_PDF_PATH)))

    if not question:
        return jsonify(error="Enter a question."), 400
    if output_format not in {"json", "csv"}:
        return jsonify(error="Format must be json or csv."), 400
    if not 1 <= top_k <= 20:
        return jsonify(error="top_k must be between 1 and 20."), 400

    try:
        store, indexed = get_indexed_store(pdf_path)

        results = Retriever(store).retrieve(question, top_k=top_k)
        records = generate_answer(question, results)
        response = {
            "query": question,
            "format": output_format,
            "records": records,
            "retrieved_chunks": len(results),
            "indexed": indexed,
        }
        if output_format == "csv":
            response["content"] = as_csv(records)
        return jsonify(response)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Query failed")
        return jsonify(error="The query failed. Check the server log."), 500


if __name__ == "__main__":
    # Avoid the debug watchdog interrupting long indexing and API requests.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
