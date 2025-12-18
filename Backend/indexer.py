#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KB Indexer
==========
Indexes DOCX, PDF, and image sources into token-aware chunks aligned 1-to-1
with the kb_chunks schema.

Features:
- Stable hash-based chunk IDs
- Namespace (ns) support
- Incremental re-indexing (skips unchanged files)
- Token-aware chunking (tiktoken optional)
- OCR caching for images
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---------- OPTIONAL TOKENIZATION ----------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def encode_text(text: str) -> List[int]:
        return _ENC.encode(text)

    def decode_tokens(tokens: List[int]) -> str:
        return _ENC.decode(tokens)

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except Exception:
    _ENC = None

    def encode_text(text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode_tokens(tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)

    def count_tokens(text: str) -> int:
        return len(text)  # char fallback


# ---------- FILE READERS ----------
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ---------- CONFIG ----------
KB_INPUT_DIRS = ["kb", "data/kb_extra"]
OUTPUT_DIR = Path("data/kb_index")
OCR_CACHE_DIR = Path("data/ocr_cache")
STATE_FILE = OUTPUT_DIR / "index_state.json"

DEFAULT_NAMESPACE = "default"

CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 50

SUPPORTED_EXT = {".docx", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------- UTIL ----------
def log(msg: str):
    print(f"[indexer] {msg}")


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def file_fingerprint(fp: Path) -> str:
    stat = fp.stat()
    raw = f"{fp.resolve()}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_state() -> Dict[str, str]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: Dict[str, str]):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def stable_chunk_id(ns: str, source: str, chunk_index: int, content: str) -> str:
    raw = f"{ns}|{source}|{chunk_index}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------- CHUNKING ----------
def split_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not _ENC:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += max(chunk_size - overlap, 1)
        return chunks

    tokens = encode_text(text)
    chunks, start, n = [], 0, len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(decode_tokens(tokens[start:end]))
        if end == n:
            break
        start += max(chunk_size - overlap, 1)
    return chunks


# ---------- EXTRACTORS ----------
def infer_title(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0][:120] if lines else None


def extract_docx(fp: Path) -> Tuple[str, Dict]:
    doc = DocxDocument(fp)
    parts, headings = [], []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        parts.append(txt)
        if p.style and "heading" in p.style.name.lower():
            headings.append(txt)
    text = normalize_whitespace("\n".join(parts))
    return text, {
        "type": "docx",
        "title": infer_title(text),
        "headings": headings,
    }


def extract_pdf(fp: Path) -> Tuple[str, Dict]:
    reader = PdfReader(str(fp))
    pages = []
    for p in reader.pages:
        try:
            pages.append(normalize_whitespace(p.extract_text() or ""))
        except Exception:
            pages.append("")
    text = normalize_whitespace("\n\n".join(pages))
    return text, {
        "type": "pdf",
        "title": infer_title(text),
    }


def ocr_image(fp: Path) -> str:
    cache_fp = OCR_CACHE_DIR / f"{fp.stem}_{abs(hash(fp))}.txt"
    if cache_fp.exists():
        return cache_fp.read_text(encoding="utf-8")

    if not OCR_AVAILABLE:
        return ""

    try:
        text = pytesseract.image_to_string(Image.open(fp))
        text = normalize_whitespace(text)
        cache_fp.write_text(text, encoding="utf-8")
        return text
    except Exception:
        return ""


def extract_image(fp: Path) -> Tuple[str, Dict]:
    text = ocr_image(fp)
    return text, {
        "type": "image",
        "title": infer_title(text) if len(text) > 40 else None,
    }


# ---------- PIPELINE ----------
def collect_files() -> List[Path]:
    files = []
    for d in KB_INPUT_DIRS:
        base = Path(d)
        if base.exists():
            files.extend(fp for fp in base.rglob("*") if fp.suffix.lower() in SUPPORTED_EXT)
    return sorted(files)


def index(namespace: str = DEFAULT_NAMESPACE):
    ensure_dirs()
    prev_state = load_state()
    new_state = {}

    all_records: List[Dict] = []

    files = collect_files()
    log(f"Scanning {len(files)} files")

    for fp in files:
        fingerprint = file_fingerprint(fp)
        new_state[str(fp)] = fingerprint

        if prev_state.get(str(fp)) == fingerprint:
            log(f"Skipping unchanged: {fp.name}")
            continue

        log(f"Indexing: {fp}")
        ext = fp.suffix.lower()

        if ext == ".docx":
            text, meta = extract_docx(fp)
        elif ext == ".pdf":
            text, meta = extract_pdf(fp)
        else:
            text, meta = extract_image(fp)

        if not text:
            continue

        chunks = split_tokens(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

        for i, chunk in enumerate(chunks):
            chunk_id = stable_chunk_id(namespace, str(fp), i, chunk)

            record = {
                # ---- kb_chunks schema alignment ----
                "ns": namespace,
                "id": chunk_id,
                "title": meta.get("title"),
                "text": chunk,
                # ---- optional / downstream ----
                "source": str(fp),
                "chunk_index": i,
                "chunk_count": len(chunks),
                "tokens_estimate": count_tokens(chunk),
                "created_at": int(time.time()),
            }
            all_records.append(record)

    out_fp = OUTPUT_DIR / f"kb_chunks_{namespace}.jsonl"
    with open(out_fp, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_state(new_state)
    log(f"Indexed {len(all_records)} chunks → {out_fp}")
    log("Done.")


if __name__ == "__main__":
    index()
