"""Ingest PDFs/TXT from data/raw -> chunk -> embed -> Chroma.

Usage:  python src/ingest.py [--reset]
"""
import argparse
import json
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

from config import settings
from embeddings import embed

RAW = Path("data/raw")


def load_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


# Contextual chunk headers: every chunk starts with which document and device it is from,
# so BM25, the embeddings, the re-ranker and the LLM can tell e.g. MIMOSA from its
# predicate ODISsey, or Derma-2 from Med-Hot, even when a chunk is a flattened table.
DEVICE_PATTERNS = [
    r"Trade/Device Name:\s*(.+?)\s+(?:Regulation Number|Regulatory Class|Dated)",
    r"Trade Name:\s*(.+?)\s+(?:Common Name|Generic|Classification|Regulation)",
    r"Proprietary Name:\s*(.+?)\s+(?:Common Name|Classification|-|Submittal)",
    r"Device Name\s+(.+?)\s+Indications for Use",
]
OVERRIDES = RAW / "devices.json"  # optional {"K073656": "OxyVu-1 (Hypermed)"} to fix a name


def device_name(path: Path, pages: list[tuple[int, str]]) -> str:
    if OVERRIDES.exists():
        name = json.loads(OVERRIDES.read_text()).get(path.stem)
        if name:
            return name
    text = " ".join(" ".join(t.split()) for _, t in pages[:6])
    for pat in DEVICE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace("™", "").replace("TM", "").strip()[:60]
    return path.stem


def chunk(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return [c for c in out if len(c) > 50]


def main(reset: bool) -> None:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    if reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass
    col = client.get_or_create_collection(settings.collection, metadata={"hnsw:space": "cosine"})

    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise SystemExit(f"No documents in {RAW}/ - see README 'Data'.")

    total = 0
    for f in files:
        ids, docs, metas = [], [], []
        pages = load_pages(f)
        device = device_name(f, pages)
        header = f"[{f.stem} | {device}] "
        for page, text in pages:
            for j, c in enumerate(chunk(text, settings.chunk_size, settings.chunk_overlap)):
                ids.append(f"{f.stem}-p{page}-c{j}")
                docs.append(header + c)
                metas.append({"source": f.name, "page": page, "device": device})
        if docs:
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
            total += len(docs)
            print(f"{f.name}: {len(docs)} chunks  [{device}]")
    print(f"Done - {total} chunks in collection '{settings.collection}'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    main(ap.parse_args().reset)
