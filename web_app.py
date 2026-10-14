import csv
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.llm_extractor import generate_answer
from src.main import DEFAULT_PDF_PATH, index_pdf
from src.retriever import Retriever
from src.vector_store import VectorStore


app = Flask(__name__)


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


@app.post("/api/query")
def query():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query", "")).strip()
    output_format = payload.get("format", "json")
    top_k = int(payload.get("top_k", 5))
    should_index = bool(payload.get("index", False))
    pdf_path = Path(payload.get("pdf", str(DEFAULT_PDF_PATH)))

    if not question:
        return jsonify(error="Enter a question."), 400
    if output_format not in {"json", "csv"}:
        return jsonify(error="Format must be json or csv."), 400
    if not 1 <= top_k <= 20:
        return jsonify(error="top_k must be between 1 and 20."), 400

    try:
        store = VectorStore()
        indexed = None
        if should_index:
            pages, chunks = index_pdf(store, pdf_path)
            indexed = {"pages": pages, "chunks": chunks}

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
    app.run(debug=True)
