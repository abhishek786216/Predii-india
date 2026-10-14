import re
import pymupdf
import pytesseract

from PIL import Image
from io import BytesIO


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.]", "", text)

    return text.strip()


def extract_ocr_text(page):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image = Image.open(
        BytesIO(pix.tobytes("png"))
    )

    return pytesseract.image_to_string(image).strip()


def extract_with_ocr(pdf_path, pages):
    with pymupdf.open(pdf_path) as doc:

        for item in pages:
            pdf_text = item["text"].strip()

            # Skip OCR when the PDF text appears sufficient.
            if len(pdf_text) >= 100:
                continue

            page = doc[item["page"] - 1]
            ocr_text = extract_ocr_text(page)

            # If OCR produces no text, keep the original.
            if not ocr_text:
                continue

            pdf_normalized = normalize_text(pdf_text)
            ocr_normalized = normalize_text(ocr_text)

            # Avoid duplicating identical text.
            if pdf_normalized == ocr_normalized:
                continue

            # If PDF text is empty, use OCR.
            if not pdf_text:
                item["text"] = ocr_text
                item["method"] = "ocr"
                continue

            # If OCR contains all the original text,
            # prefer the OCR result.
            if pdf_normalized in ocr_normalized:
                item["text"] = ocr_text
                item["method"] = "ocr"
                continue

            # If PDF text contains all the OCR text,
            # keep the original PDF text.
            if ocr_normalized in pdf_normalized:
                continue

            # If the results differ, preserve both
            # for subsequent review.
            item["text"] = (
                pdf_text + "\n" + ocr_text
            )
            item["method"] = "combined"

    return pages