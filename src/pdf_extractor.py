import pymupdf


def extract_pdf_text(pdf_path):
    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            pages.append({
                "page": page_number,
                "text": text,
                "method": "pdf_text"
            })

    return pages