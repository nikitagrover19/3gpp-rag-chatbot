"""
Generation layer: turns (question, retrieved chunks) into a grounded answer.

Anti-hallucination design choices:

1. Retrieval gate: if retriever.py returns zero chunks (nothing above
   MIN_SIMILARITY), we never even call the LLM — we return REFUSAL_MESSAGE
   directly.

2. Strict grounding instruction: the system prompt forbids outside/
   parametric knowledge and requires every claim to cite a provided chunk.

3. Mandatory citation format: [<source_doc> § <clause_number>].

4. Explicit "insufficient information" instruction: the system prompt states
   that declining to answer is preferred over a fabricated complete answer.

5. Low temperature: more deterministic decoding.

6. Post-hoc citation audit (extract_cited_clauses, below): every clause
   number cited in the answer is checked against the clause numbers actually
   retrieved; a citation to a clause never retrieved is flagged as invented.

7. Self-correction (_attempt_self_correction, below): when the audit flags
   an invented citation, the answer is sent back to the model with the
   specific problem named and a revision is requested, before showing
   anything to the caller. Falls back to the original answer plus a warning
   if the correction doesn't resolve it.

Backend is pluggable via GENERATION_BACKEND in config.py:
- "groq" (default): openai/gpt-oss-120b via the Groq API, free tier,
  requires GROQ_API_KEY.
- "local_hf": a small model (LOCAL_LLM_MODEL_NAME) run fully offline via
  transformers, no API key required, lower answer quality.
Both dispatch through _call_backend(), below, so the grounding/audit logic
above is identical regardless of which backend is active.
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
2. Every factual claim must be immediately followed by a citation in EXACTLY this format: [<source_doc> § <clause_number>] — for example [24501-id0.docx § 5.5.1.2.3]. The clause number always comes AFTER the § symbol, never before it, and the source document name (not a description of the clause) always comes first.
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


def call_local_backend(messages):
    pipe = _lazy_load_local_model()
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


def call_groq_backend(messages, max_tokens=None):
    """
    openai/gpt-oss-120b via Groq's free API tier. Groq's SDK is OpenAI-compatible.
    No cost, no credit card required for the free tier.

    max_tokens is overridable per-call: gpt-oss-120b is a reasoning model —
    its internal reasoning tokens count against the completion budget, and a
    longer conversation (e.g. the self-correction round-trip, which includes
    the full original exchange plus new instructions) can burn the whole
    budget on reasoning and leave nothing for the actual final answer,
    silently returning an empty string. The correction call passes a larger
    budget to avoid this.
    """
    client = _lazy_load_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens or MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def call_api_backend(messages):
    """Dispatcher kept for naming symmetry with call_local_backend; routes to Groq."""
    return call_groq_backend(messages)


def _call_backend(messages, max_tokens=None):
    """Single dispatch point so both the initial generation and the
    self-correction round-trip (below) go through the same backend logic."""
    if GENERATION_BACKEND == "local_hf":
        return call_local_backend(messages)
    elif GENERATION_BACKEND == "groq":
        return call_groq_backend(messages, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown GENERATION_BACKEND: {GENERATION_BACKEND}")


CORRECTION_PROMPT_TEMPLATE = """Your previous answer cited the following clause number(s), but they do NOT appear anywhere in the provided context: {invented_clauses}

This means you cited a source you were not actually given — likely because the claim came from your own general knowledge rather than the retrieved excerpts, even though it may be factually true in general.

Revise your answer:
- Remove or rewrite any claim that relied on that invented citation, UNLESS the same fact is also supported by a citation to a clause that IS in the provided context (in which case keep the claim but cite only the real clause).
- Do not introduce any new clause numbers that aren't in the original context.
- Keep everything else in your answer that was already correctly grounded.

Provide the complete revised answer."""


def _attempt_self_correction(messages, previous_answer, invented_clauses, retrieved_clauses):
    """
    One-shot self-correction round-trip: when the post-hoc audit finds a
    citation to a clause that was never retrieved, send the flawed answer
    back to the model with the specific problem named and ask it to revise
    — rather than silently showing the user an answer known to contain a
    fabricated citation. Capped at one attempt (no retry loop) to bound
    latency/cost; if the correction still isn't clean, we fall back to
    surfacing the warning rather than looping indefinitely.

    Requests a larger token budget than the initial call (see call_groq_backend)
    since the correction round-trip includes the full prior exchange and is
    more prone to truncation. A blank/whitespace-only response is treated as
    a failed correction, never as "no invented clauses" — an empty string
    trivially contains no invalid citations, which would otherwise be
    misread as success and silently show the user nothing.
    """
    correction_messages = messages + [
        {"role": "assistant", "content": previous_answer},
        {"role": "user", "content": CORRECTION_PROMPT_TEMPLATE.format(
            invented_clauses=", ".join(sorted(invented_clauses))
        )},
    ]
    corrected_answer = _call_backend(correction_messages, max_tokens=MAX_NEW_TOKENS + 400)

    if not corrected_answer.strip():
        # Treat a blank correction response as an outright failure — return
        # sentinel values the caller recognizes as "correction did not
        # produce a usable answer" rather than a false "success".
        return previous_answer, extract_cited_clauses(previous_answer), invented_clauses

    corrected_cited = extract_cited_clauses(corrected_answer)
    corrected_invented = corrected_cited - retrieved_clauses
    return corrected_answer, corrected_cited, corrected_invented


CITATION_BRACKET_RE = re.compile(r"\[[^\]]*§[^\]]*\]")
CLAUSE_NUMBER_RE = re.compile(r"\d+\.\d+(?:\.\d+)*")


def extract_cited_clauses(answer_text):
    """Pulls out every clause number cited in the answer.

    The system prompt asks for the format '[<source_doc> § <clause_number>]',
    e.g. '[TS 24.501 § 5.5.1.2.3]' — but the model doesn't always follow that
    exact ordering. In practice it sometimes writes
    '[5.5.1.2.3 § <paraphrased description>]' instead, putting the clause
    number BEFORE the § symbol with descriptive text after it. A regex that
    only looks for digits immediately following '§' misses that case
    entirely — silently, with no error — which means a real invented
    citation can slip past the audit undetected simply because of citation
    formatting drift, not because it was actually grounded.

    To avoid depending on the model's formatting being exactly right, this
    instead finds every bracketed span containing a '§' (the citation marker
    itself, which the model does reliably include) and pulls out ANY
    clause-number-shaped token inside that span, regardless of which side of
    '§' it's on. Requires at least one sub-level (X.Y) — retrieved chunks are
    always leaf-level clauses like '6.2.1' or '5.15.1', never a bare
    top-level number like '6' — so an informal in-prose reference to a
    broader section ("as described in §6...", which won't be inside a
    bracket at all) still isn't misread as a citation.
    """
    clauses = set()
    for bracket in CITATION_BRACKET_RE.findall(answer_text):
        clauses.update(CLAUSE_NUMBER_RE.findall(bracket))
    return clauses


def generate_answer(question, chunks):
    """
    Top-level entry point. Returns a dict with the answer text plus
    groundedness metadata for evaluation.

    If the first-pass answer contains an invented citation (a clause number
    not present in the retrieved context), this triggers a one-shot
    self-correction round-trip (see _attempt_self_correction) rather than
    just flagging the problem and returning the flawed answer as-is. The
    result dict reports whether correction was needed/applied so callers
    (evaluate.py, app.py) can be transparent about it.
    """
    if not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "refused": True,
            "cited_clauses": set(),
            "retrieved_clauses": set(),
            "uncited_or_invented_clauses": set(),
            "correction_attempted": False,
            "correction_succeeded": None,
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, chunks)},
    ]
    answer = _call_backend(messages)

    retrieved_clauses = {c["clause_number"] for c in chunks}
    cited_clauses = extract_cited_clauses(answer)
    invented = cited_clauses - retrieved_clauses  # citations to clauses we never actually retrieved

    correction_attempted = False
    correction_succeeded = None

    if invented:
        correction_attempted = True
        corrected_answer, corrected_cited, corrected_invented = _attempt_self_correction(
            messages, answer, invented, retrieved_clauses
        )
        correction_succeeded = not corrected_invented  # True if the revision is now clean
        if correction_succeeded:
            # use the corrected answer — the fabricated citation is gone
            answer, cited_clauses, invented = corrected_answer, corrected_cited, corrected_invented
        # if correction_succeeded is False, we deliberately keep the ORIGINAL
        # answer/invented-set rather than the possibly-still-flawed revision,
        # so the groundedness warning downstream reflects a known-accurate
        # diagnosis rather than a second unverified guess.

    return {
        "answer": answer,
        "refused": False,
        "cited_clauses": cited_clauses,
        "retrieved_clauses": retrieved_clauses,
        "uncited_or_invented_clauses": invented,
        "correction_attempted": correction_attempted,
        "correction_succeeded": correction_succeeded,
    }
