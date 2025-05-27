import fitz  # PyMuPDF
import random

def extract_pdf_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def get_random_chunk(text, chunk_size=1000):
    if len(text) <= chunk_size:
        return text
    start = random.randint(0, len(text) - chunk_size)
    return text[start:start + chunk_size]
