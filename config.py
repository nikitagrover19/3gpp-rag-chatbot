"""Central configuration for the 3GPP RAG chatbot."""
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_3gpp")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index_store")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Preserve clause boundaries before applying fixed-size splitting.
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# Local embedding model used for retrieval.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5
# Drop chunks below this cosine similarity threshold.
MIN_SIMILARITY = 0.30

# Generation backend and model.
GENERATION_BACKEND = "groq"  # "groq" | "local_hf"

GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"

LOCAL_LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.1  # low temperature = more deterministic, less hallucination-prone

REFUSAL_MESSAGE = (
    "I couldn't find this in the provided 3GPP documentation, so I won't "
    "guess. Please rephrase, or supply the relevant spec section."
)
