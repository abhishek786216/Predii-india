from src.text_cleaner import clean_text


def split_text(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(pages, chunk_size=1000, overlap=200):
    all_chunks = []

    for page in pages:
        text = clean_text(page["text"])

        if not text:
            continue

        page_chunks = split_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in page_chunks:
            all_chunks.append({
                "text": chunk,
                "page": page["page"],
                "method": page["method"]
            })

    return all_chunks