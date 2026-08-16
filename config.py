"""
Central configuration for the 3GPP RAG chatbot.
Change values here rather than hunting through the codebase.
"""
import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_3gpp")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index_store")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# --- Chunking ---
# 3GPP specs are hierarchically numbered (e.g. 5.5.1.2.3). We chunk on clause
# boundaries first (structure-aware) and only fall back to fixed-size splitting
# for clauses that are unusually long. This preserves semantic + citation
# integrity far better than naive fixed-length windows.
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# --- Embedding model ---
# BGE-M3: free, local, runs fine on CPU, and scores meaningfully higher on
# MTEB retrieval benchmarks (~63.0) than smaller models like all-MiniLM-L6-v2
# (~56). Retrieval quality is the single biggest lever on final answer
# quality in a RAG pipeline, so it's worth the extra ~2GB download / slightly
# higher per-chunk latency versus MiniLM.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# --- Retrieval ---
TOP_K = 5
# Minimum cosine similarity for a chunk to be considered "relevant enough" to
# use. Chunks below this are dropped. This is one of the main anti-
# hallucination levers: if nothing is relevant, we refuse rather than guess.
MIN_SIMILARITY = 0.30

# --- Generation ---
# Primary backend: openai/gpt-oss-120b via Groq's free API tier. No cost, no
# credit card. Originally configured with qwen/qwen3-32b, but Groq
# deprecated that model; gpt-oss-120b is their suggested replacement and
# performed well in testing (see README Results). No RAGAS or other formal
# faithfulness benchmark was run for this specific model — the eval numbers
# in README.md are this project's own measurement, not a third-party score.
#
# Fallback backend: "local_hf" runs a small model fully offline, zero setup,
# for anyone evaluating this project without wanting to create an API key.
GENERATION_BACKEND = "groq"  # "groq" | "local_hf"

GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"  # set this in your shell, never hardcode the key

LOCAL_LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # offline fallback, runs on CPU (slowly) or GPU
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.1  # low temperature = more deterministic, less hallucination-prone

REFUSAL_MESSAGE = (
    "I couldn't find this in the provided 3GPP documentation, so I won't "
    "guess. Please rephrase, or supply the relevant spec section."
)
