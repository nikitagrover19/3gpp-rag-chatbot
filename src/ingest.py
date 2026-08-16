"""Ingest 3GPP documents into structure-aware chunks."""
import os
import re
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, CHUNKS_PATH, INDEX_DIR, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS

# Matches lines like "5.5.1.2.3 Initial registration accepted by the network"
CLAUSE_HEADER_RE = re.compile(r"^(\d+(?:\.\d+){0,5})\s+(.+)$")

# Word exports ToC entries with a trailing tab and page number.
TOC_ENTRY_RE = re.compile(r"\t\d+\s*$")

# Matches "3GPP TS 24.501 - Non-Access-Stratum ..." as the doc title line
DOC_TITLE_RE = re.compile(r"^3GPP\s+(TS|TR)\s+([\d.]+)\s*-\s*(.+)$")

TOC_TRAILING_PAGE_NUM_RE = re.compile(r"\t\d+\s*$")


def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx_file(path):
    """3GPP's real distribution format is .docx inside a .zip. python-docx
    reads paragraph text; tables (used for some annexes) are skipped for
    simplicity — extend this if you need annex table content indexed too."""
    import docx
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def split_into_clauses(raw_text, doc_id, doc_title):
    """Split a document's text into (clause_number, clause_title, body) tuples."""
    lines = raw_text.splitlines()
    clauses = []
    current = None

    for line in lines:
        if TOC_ENTRY_RE.search(line):
            continue  # skip table-of-contents lines entirely — not real clause content
        stripped = line.strip()
        m = CLAUSE_HEADER_RE.match(stripped)
        # Reject likely false-positive lines from document front matter.
        if m and m.group(2)[:1].islower():
            m = None
        if m and len(m.group(2)) < 120:  # header lines are short; avoids false positives on body text
            if TOC_TRAILING_PAGE_NUM_RE.search(stripped):
                # Table-of-contents entry, not a real clause header — skip it
                # entirely rather than starting a bogus clause.
                continue
            if current:
                clauses.append(current)
            current = {
                "clause_number": m.group(1),
                "clause_title": m.group(2).strip(),
                "body_lines": [],
            }
        else:
            if current is not None:
                current["body_lines"].append(line)
            # lines before the first detected clause header are dropped (title page/preamble)

    if current:
        clauses.append(current)

    return clauses


def sub_split_long_clause(text, max_chars, overlap):
    """Fallback fixed-size splitter, only used when a single clause is too long."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return parts


def chunk_document(filepath):
    filename = os.path.basename(filepath)
    if filepath.lower().endswith(".pdf"):
        raw_text = read_pdf_file(filepath)
    elif filepath.lower().endswith(".docx"):
        raw_text = read_docx_file(filepath)
    else:
        raw_text = read_text_file(filepath)

    doc_title = filename
    first_line = raw_text.strip().splitlines()[0] if raw_text.strip() else ""
    m = DOC_TITLE_RE.match(first_line.strip())
    if m:
        doc_title = f"TS {m.group(2)} - {m.group(3)}"

    clauses = split_into_clauses(raw_text, filename, doc_title)

    chunks = []
    for clause in clauses:
        body = "\n".join(clause["body_lines"]).strip()
        if not body:
            continue
        sub_parts = sub_split_long_clause(body, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        for i, part in enumerate(sub_parts):
            chunk_id = f"{filename}::{clause['clause_number']}"
            if len(sub_parts) > 1:
                chunk_id += f"::part{i+1}"
            chunks.append({
                "chunk_id": chunk_id,
                "source_doc": doc_title,
                "source_file": filename,
                "clause_number": clause["clause_number"],
                "clause_title": clause["clause_title"],
                "text": f"{clause['clause_number']} {clause['clause_title']}\n{part}",
            })
    return chunks


def run_ingestion():
    os.makedirs(INDEX_DIR, exist_ok=True)
    all_chunks = []
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".txt", ".pdf", ".docx"))]
    if not files:
        print(f"No .txt, .pdf, or .docx files found in {DATA_DIR}. "
              f"Unzip a downloaded 3GPP spec (e.g. 24501-id0.zip) and drop the .docx file here.")
        return []

    for fname in sorted(files):
        fpath = os.path.join(DATA_DIR, fname)
        chunks = chunk_document(fpath)
        print(f"  {fname}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks written to {CHUNKS_PATH}")
    return all_chunks


if __name__ == "__main__":
    print(f"Ingesting documents from {DATA_DIR} ...")
    run_ingestion()
