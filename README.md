# 3GPP RAG Chatbot

A Retrieval-Augmented Generation chatbot for answering questions over 3GPP
standards documentation. Retrieval and embeddings run locally; generation
uses a free Groq API by default, with a fully local, no-API-key fallback
available. The pipeline is designed to prefer a grounded refusal over an
answer that cannot be supported by the retrieved documents.

## Architecture

```
3GPP documents
      |
      v
   ingest.py
(clause-aware chunking)
      |
      v
 build_index.py
(embeddings + FAISS)
      |
      v
   retriever.py
(semantic search + similarity threshold)
      |
      v
   generator.py
(grounded generation + citation check)
      |
      v
    answer
```

`rag_pipeline.py` connects retrieval and generation, `app.py` provides the
CLI, and `evaluate.py` runs the evaluation set.

## Design choices

### Clause-aware chunking

3GPP documents use hierarchical clause numbering (e.g. `5.5.1.2.3`), so
`ingest.py` detects clause headers and chunks on clause boundaries rather
than splitting by character count. Fixed-length chunking would routinely
cut a clause in half, handing the model incomplete context.

Long clauses are sub-split only when they exceed a length threshold, with
overlap to preserve continuity. Every chunk retains its source document,
clause number, and clause title, which is what lets the generator cite a
specific clause (`[24501-id0.docx § 5.5.1.2.3]`) instead of a vague
reference to "the document."

### Retrieval

Embeddings use `BAAI/bge-m3` (local, free, runs on CPU) with a FAISS
flat index for cosine-similarity search. Retrieved chunks are filtered by a
minimum similarity threshold (`MIN_SIMILARITY` in `config.py`); if nothing
clears it, retrieval returns empty on purpose. If no relevant chunks are
retrieved, the generator receives no context and the system returns a
refusal instead of answering from unrelated context or general knowledge.

### Generation and grounding

The system prompt instructs the model to use only the retrieved context and
to cite the source clause for every claim, and states explicitly that
declining to answer is preferred over a fabricated complete answer.

Generated answers are checked after generation: every clause number the
model cites is compared against the clause numbers actually retrieved
(`extract_cited_clauses()` in `generator.py`). A citation to a clause that
was never retrieved is flagged as invented. When this happens, the pipeline
sends the answer back to the model with the specific problem named and asks
for a correction (`_attempt_self_correction()`), falling back to the
original answer plus a warning if the correction doesn't resolve it.

This grounding/audit logic is independent of the generation backend.

### Backend

`GENERATION_BACKEND` in `config.py` defaults to `"groq"`, using
`openai/gpt-oss-120b` via the Groq API (free tier, requires an API key —
see Setup below). `GENERATION_BACKEND = "local_hf"` runs a small model
(`microsoft/Phi-3-mini-4k-instruct`) fully offline instead, with no API key
required, at lower answer quality. The grounding and citation-audit logic
is identical either way.

## Evaluation

`eval_set.json` has 15 questions across 9 categories: basic factual
questions, out-of-scope questions, a vendor-specific question (asks about
Mavenir's own implementation, which no 3GPP spec covers), a prompt-injection
attempt, a multi-hop question spanning two documents, an ambiguously-phrased
question, two "near-miss" questions (details that sound plausible but
aren't in the spec), and a question about a clause the real spec marks
`Void`.

`evaluate.py` checks three things:
- **Refusal accuracy** — does the system decline out-of-scope questions
  rather than fabricate an answer?
- **Citation groundedness** — does every cited clause number appear in the
  retrieved context?
- **Retrieval hit rate** — did the expected clause get retrieved? This is a
  diagnostic metric, not a hallucination metric, and the `expected_clause`
  values are hand-verified from manual testing rather than a formally
  scored ground truth set.

Two categories (`prompt_injection`, `deprecated_clause_handling`) test *how*
the model responds, not just pass/fail, so `evaluate.py` prints those
answers in full for manual review.

## Results

![Evaluation Results](results_assets/eval_results.png)

Run against the ingested corpus (8,266 chunks across TS 24.501, TS 23.501,
TS 23.502) with `openai/gpt-oss-120b` via Groq:

| Metric | Result |
|---|---:|
| Refusal accuracy | 6/6 (100%) |
| Citation groundedness | 9/9 (100%) |
| Self-correction (invented citations auto-fixed) | 2/2 (100%) |
| Retrieval hit rate (diagnostic) | 4/9 (44%) |

### Debugging a retrieval failure

During testing, the model repeatedly cited clause `9.11.3.4` in answers
about the "5GS mobile identity" IE, even though that clause was never in
the retrieved context. The citation audit flagged the mismatch every time:

```
[groundedness warning] cited clause(s) not found in retrieved context: {'9.11.3.4'}
```

The repeated appearance of the same specific clause number suggested this
was worth investigating as a possible retrieval or ingestion issue rather
than assuming it was arbitrary fabrication. Grepping the raw `.docx`
directly (bypassing the chunker) found that clause 9.11.3.4 was genuinely
present in the source document, with content matching what the model had
produced.

The root cause was in the clause-header filter in `ingest.py`: a check
meant to reject front-matter boilerplate was rejecting any candidate clause
title that didn't start with an uppercase letter. Many real 3GPP
information-element names start with a digit ("5GS mobile identity",
"5G-GUTI"), so the filter was silently dropping those clauses across the
whole corpus — re-ingestion after the fix recovered 64 previously-dropped
chunks, not just this one clause.

After the fix and a rebuild, the same question retrieves clause 9.11.3.4
directly (top result, 0.71 similarity) and cites it correctly, with no
warning. This case is documented in
`tests/test_chunking.py::test_digit_led_clause_titles_are_kept`.

This wasn't purely a model hallucination — the model had accurate knowledge
from training, but the pipeline had failed to put the matching source
material into context. The audit caught the symptom; tracing it back
distinguished a retrieval bug from genuine fabrication and led to a fix
rather than just a logged warning.

Self-correction was also tested on this exact case, before the fix: it did
not resolve it (the model repeated the same citation even after being told
it was invalid), and the system reported that failure explicitly rather
than silently keeping the flawed answer:

```
[groundedness warning] cited clause(s) not found in retrieved context,
and automatic correction did not fully resolve it: {'9.11.3.4'}
```

### Void-clause case

One eval case asks about clause `5.4.4.2` in TS 23.501, which the real spec
marks `Void`. The model reports that the provided context doesn't cover it,
which is correct — but retrieval doesn't actually surface that clause for
this query, likely because a `Void` clause has almost no text to embed
against. The refusal is correct, but for a retrieval-recall reason rather
than the model having seen and reported the clause as void.

### Bugs found and fixed

Covered by regression tests in `tests/test_chunking.py`:
1. **Table of Contents mis-parsed as clause content.** Word-exported ToC
   lines were read as real clauses. Fixed by detecting the trailing
   tab+page-number pattern unique to ToC lines.
2. **Front-matter boilerplate false positive.** A line from the "version
   numbering convention" boilerplate was matched as a clause header because
   it starts with a digit. Fixed by rejecting titles starting with a
   lowercase letter.
3. **Citation-format sensitivity.** The citation-extraction regex only
   matched clause numbers immediately after `§`, but the model doesn't
   always follow the `[<doc> § <clause>]` ordering — it sometimes writes
   `[<clause> § <description>]` instead, which the original regex missed
   entirely. This is how the `9.11.3.4` citation went undetected for two
   full test rounds. Fixed by extracting any clause-number-shaped token
   from within any bracketed span containing `§`, regardless of order.
4. **Digit-led clause titles dropped.** The fix for bug 2 above was too
   broad and caused the `9.11.3.4` retrieval failure described above. Fixed
   by checking specifically for a leading lowercase letter.

### Evidence

![Eval suite run — questions and pass/fail](results_assets/eval_summary_run1.png)

![Eval suite run — summary and category breakdown](results_assets/eval_summary_run2.png)

![Clean answer after the chunking fix](results_assets/mobile_identity_clean_answer.png)

## Getting 3GPP specs

Specs are distributed at `https://www.3gpp.org/ftp/Specs/latest/<Release>/<series>_series/<spec>-<version>.zip`.
Each zip contains a `.docx` file, not a PDF — `ingest.py` reads `.docx`
directly.

```bash
curl -L -A "Mozilla/5.0" -O https://www.3gpp.org/ftp/specs/latest/Rel-18/24_series/24501-id0.zip
curl -L -A "Mozilla/5.0" -O https://www.3gpp.org/ftp/specs/latest/Rel-18/23_series/23501-ic0.zip
curl -L -A "Mozilla/5.0" -O https://www.3gpp.org/ftp/specs/latest/Rel-18/23_series/23502-ie0.zip
unzip 24501-id0.zip -d data/sample_3gpp/
unzip 23501-ic0.zip -d data/sample_3gpp/
unzip 23502-ie0.zip -d data/sample_3gpp/
```

Check the live directory listing for the current version suffix before
downloading — version suffixes change when a spec is revised.

## Setup

```bash
pip install -r requirements.txt

# Free Groq API key, no credit card: https://console.groq.com
export GROQ_API_KEY=your_key_here

python src/ingest.py          # chunk the documents in data/sample_3gpp/
python src/build_index.py     # build embeddings + FAISS index (downloads BGE-M3 once)
python app.py                 # interactive chat
python src/evaluate.py        # run the eval suite
python tests/test_chunking.py # offline unit tests, no ML deps required
```

To run without an API key, set `GENERATION_BACKEND = "local_hf"` in
`config.py` and install `transformers`/`torch` (commented out in
`requirements.txt` by default).

## Known limitations

- **Corpus scope**: 3 specs (24.501, 23.501, 23.502) rather than the full
  3GPP corpus. This was a deliberate choice — a focused, cross-referenced
  corpus gives higher retrieval precision than a broader but shallower one.
- **BGE-M3 embedding time**: ~3 hours on CPU for the full corpus. A smaller
  model (`bge-small-en-v1.5` or `all-MiniLM-L6-v2`) would be faster to
  iterate with, at some cost to retrieval quality.
- No re-ranking stage — a cross-encoder re-ranker over the top-k results
  would likely improve precision, especially as the corpus grows.
- Retrieval hit rate (44%) is diagnostic, not a hallucination metric, and
  `expected_clause` values are hand-verified rather than a formal ground
  truth set; some questions have relevant content spanning more than one
  clause.
- Table content in `.docx` files is not extracted — `read_docx_file()`
  reads paragraph text only, so parameter tables in some annexes are not
  indexed.
