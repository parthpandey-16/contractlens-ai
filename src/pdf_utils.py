from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all readable text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)

    return "\n".join(pages_text).strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """Split text into overlapping chunks for retrieval."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks