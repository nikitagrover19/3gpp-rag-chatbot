"""Retrieve relevant chunks using FAISS and a minimum similarity threshold."""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNKS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, TOP_K, MIN_SIMILARITY

_model = None
_index = None
_chunks = None


def _lazy_load():
    global _model, _index, _chunks
    if _model is None:
        import faiss
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _index = faiss.read_index(FAISS_INDEX_PATH)
        _chunks = []
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                _chunks.append(json.loads(line))


def retrieve(query, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    """Return relevant chunks above the similarity threshold."""
    _lazy_load()
    import numpy as np

    q_emb = _model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, idxs = _index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        if score < min_similarity:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    hits = retrieve(args.query, top_k=args.top_k)
    if not hits:
        print("No sufficiently relevant chunks found.")
    for h in hits:
        print(f"[{h['score']:.3f}] {h['source_doc']} § {h['clause_number']} {h['clause_title']}")
