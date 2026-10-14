import argparse
import csv
import json
import sys
from pathlib import Path

from src.chunker import create_chunks
from src.pdf_extractor import extract_pdf_text
from src.llm_extractor import generate_answer
from src.ocr import extract_with_ocr
from src.retriever import Retriever
from src.vector_store import VectorStore


DEFAULT_PDF_PATH = Path("data/sample-service-manual.pdf")


def configure_output():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def index_pdf(vector_store, pdf_path):
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = extract_pdf_text(str(pdf_path))
    pages = extract_with_ocr(str(pdf_path), pages)
    chunks = create_chunks(pages)
    vector_store.add_chunks(chunks)

    return len(pages), len(chunks)


OUTPUT_DIR = Path("outputs")


def write_output(specifications, output_format, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        output_path.write_text(
            json.dumps(specifications, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    fieldnames = [
        "specification_name",
        "value",
        "unit",
        "vehicle_model",
        "part_number",
        "source_page",
        "source_text",
        "confidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(specifications)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index a service manual and search its specifications."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Path to the service manual PDF.",
    )
    parser.add_argument(
        "--query",
        help="Question or specification to search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matching chunks to return.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Search the existing Chroma collection without indexing a PDF.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path. Defaults to outputs/specifications.<format>.",
    )
    return parser.parse_args()


def main():
    configure_output()
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    if not args.query:
        raise ValueError("Provide a search question with --query.")

    vector_store = VectorStore()

    if not args.no_index:
        page_count, chunk_count = index_pdf(vector_store, args.pdf)
        print(
            f"Indexed {chunk_count} chunks from {page_count} pages "
            f"into ChromaDB."
        )

    retriever = Retriever(vector_store)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    answer = generate_answer(args.query, results)
    output_path = args.output or (
        OUTPUT_DIR / f"specifications.{args.format}"
    )
    write_output(answer, args.format, output_path)
    print(f"Wrote {len(answer)} specifications to {output_path}")


if __name__ == "__main__":
    main()