"""
PDF Processing Module
Extracts text from uploaded PDF files, chunks it for RAG ingestion.
"""

import os
import re
from typing import List, Tuple

import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")

    full_text = "\n\n".join(text_parts)
    if not full_text.strip():
        raise ValueError(f"No text found in {pdf_path}. The PDF may be image-based.")
    return full_text


def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, fix encoding artifacts."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page markers if not needed downstream
    text = re.sub(r'--- Page \d+ ---\n', '', text)
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for RAG ingestion.
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            chunks.append(chunk_text_str)
            # Keep overlap
            overlap_words = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_count + s_len > overlap:
                    break
                overlap_words.insert(0, s)
                overlap_count += s_len
            current_chunk = overlap_words
            current_length = overlap_count

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_pdf(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Full pipeline: extract → clean → chunk.
    Returns (full_text, chunks).
    """
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    return cleaned, chunks


def process_multiple_pdfs(pdf_paths: List[str]) -> Tuple[str, List[str]]:
    """Process multiple PDFs and combine results."""
    all_text = []
    all_chunks = []
    for path in pdf_paths:
        text, chunks = process_pdf(path)
        all_text.append(f"=== Document: {os.path.basename(path)} ===\n{text}")
        all_chunks.extend(chunks)
    return "\n\n".join(all_text), all_chunks
