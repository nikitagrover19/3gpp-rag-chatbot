"""
Generation layer: turns (question, retrieved chunks) into a grounded answer.

Anti-hallucination design choices (this is the core of the assignment):

1. Retrieval gate: if retriever.py returns zero chunks (nothing above
   MIN_SIMILARITY), we never even call the LLM — we return REFUSAL_MESSAGE
   directly. The model can't hallucinate an answer it's never asked to produce.

2. Strict grounding instruction: the system prompt explicitly forbids using
   outside/parametric knowledge and requires every claim to be traceable to
   a provided chunk.

3. Mandatory citation: the model must tag each claim with the clause number
   it came from (e.g. "[TS 24.501 § 5.5.1.2.3]"). This does double duty —
   it gives the evaluator a way to verify claims against the source, and the
   act of having to name a specific clause measurably suppresses free-form
   fabrication compared to unattributed generation.

4. Explicit "insufficient information" instruction: the model is told, in
   the system prompt, that saying "not covered in the provided documents" is
   a fully acceptable and preferred answer when the chunks don't fully
   answer the question — this removes the implicit pressure most chat models
   have to always produce a confident, complete answer.

5. Low temperature: deterministic decoding reduces variance and speculative
   phrasing.

6. Post-hoc citation check (see evaluate.py): every clause number cited in
   the answer is checked against the clause numbers actually present in the
   retrieved context; uncited/invented clause numbers are flagged.

Backend is pluggable: GENERATION_BACKEND = "local_hf" runs a small
instruction-tuned model fully locally and for free. Set it to "api" and
fill in call_api_backend() to use a free-tier hosted API instead (Groq,
Google AI Studio, etc.) if you want faster responses without a local GPU.
"""
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GENERATION_BACKEND, LOCAL_LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE,
    REFUSAL_MESSAGE, GROQ_MODEL_NAME, GROQ_API_KEY_ENV_VAR,
)

SYSTEM_PROMPT = """You are a technical assistant answering questions strictly from the provided 3GPP specification excerpts.

Rules you must follow:
1. Use ONLY the information in the provided context. Do not use outside knowledge, even if you are confident it is correct.
2. Every factual claim must be immediately followed by a citation to the clause it came from, in the format [<source_doc> § <clause_number>].
3. If the context does not fully answer the question, say so explicitly (e.g. "The provided documents do not cover X"). A partial or "insufficient information" answer is preferred over a fabricated complete one.
4. Do not invent clause numbers, section titles, or document names that are not present in the context.
5. Be concise and technically precise."""

_local_pipeline = None


def _format_context(chunks):
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['source_doc']} § {c['clause_number']} — {c['clause_title']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _build_user_prompt(question, chunks):
    context = _format_context(chunks)
    return f"""Context (retrieved 3GPP excerpts):

{context}

---

Question: {question}

Answer using only the context above, with clause citations for every claim."""


def _lazy_load_local_model():
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import pipeline
        print(f"Loading local generation model: {LOCAL_LLM_MODEL_NAME} (first run downloads it, then it's cached)")
        _local_pipeline = pipeline(
            "text-generation",
            model=LOCAL_LLM_MODEL_NAME,
            trust_remote_code=True,
        )
    return _local_pipeline


def call_local_backend(question, chunks):
    pipe = _lazy_load_local_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, chunks)},
    ]
    out = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=TEMPERATURE > 0,
    )
    generated = out[0]["generated_text"]
    # transformers chat pipelines return the full conversation; take the last assistant turn
    if isinstance(generated, list):
        return generated[-1]["content"]
    return str(generated)


_groq_client = None


def _lazy_load_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get(GROQ_API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(
                f"{GROQ_API_KEY_ENV_VAR} is not set. Get a free key at "
                f"https://console.groq.com and export it, e.g.\n"
                f"  export {GROQ_API_KEY_ENV_VAR}=your_key_here"
            )
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def call_groq_backend(question, chunks):
    """
    Qwen3-32B via Groq's free API tier. Groq's SDK is OpenAI-compatible.
    No cost, no credit card required for the free tier.
    """
    client = _lazy_load_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, chunks)},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


def call_api_backend(question, chunks):
    """Dispatcher kept for naming symmetry with call_local_backend; routes to Groq."""
    return call_groq_backend(question, chunks)


def extract_cited_clauses(answer_text):
    """Pulls out every clause number cited in the answer, e.g. '5.5.1.2.3' from '[TS 24.501 § 5.5.1.2.3]'.
    Requires at least one sub-level (X.Y) — retrieved chunks are always
    leaf-level clauses like '6.2.1' or '5.15.1', never a bare top-level
    number like '6'. Without this, an informal in-prose reference to a
    broader section (e.g. "as described in §6...") gets misread as a
    citation to a specific clause that was never actually retrieved,
    producing a false "invented citation" flag."""
    return set(re.findall(r"§\s*(\d+\.\d+(?:\.\d+)*)", answer_text))


def generate_answer(question, chunks):
    """
    Top-level entry point. Returns a dict with the answer text plus
    groundedness metadata for evaluation.
    """
    if not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "refused": True,
            "cited_clauses": set(),
            "retrieved_clauses": set(),
            "uncited_or_invented_clauses": set(),
        }

    if GENERATION_BACKEND == "local_hf":
        answer = call_local_backend(question, chunks)
    elif GENERATION_BACKEND == "groq":
        answer = call_groq_backend(question, chunks)
    else:
        raise ValueError(f"Unknown GENERATION_BACKEND: {GENERATION_BACKEND}")

    retrieved_clauses = {c["clause_number"] for c in chunks}
    cited_clauses = extract_cited_clauses(answer)
    invented = cited_clauses - retrieved_clauses  # citations to clauses we never actually retrieved

    return {
        "answer": answer,
        "refused": False,
        "cited_clauses": cited_clauses,
        "retrieved_clauses": retrieved_clauses,
        "uncited_or_invented_clauses": invented,
    }