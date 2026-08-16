"""Build a local FAISS index from the ingested text chunks."""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNKS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, INDEX_DIR


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index():
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    os.makedirs(INDEX_DIR, exist_ok=True)
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found — run src/ingest.py first.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (downloads once, then cached locally)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"FAISS index with {index.ntotal} vectors written to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    build_index()
