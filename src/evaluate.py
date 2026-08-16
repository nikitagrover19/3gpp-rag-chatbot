"""
Evaluation harness. Runs eval_set.json through the pipeline and reports:

1. Refusal accuracy: for questions marked "answerable: false" (i.e. genuinely
   not covered by the ingested corpus), did the system correctly refuse
   instead of fabricating an answer? This is the primary hallucination
   metric requested by the assignment.
2. Groundedness: for answerable questions, did every cited clause number
   actually appear in the retrieved context (no invented citations)?
3. Retrieval hit: did the expected clause show up among the retrieved chunks?

This is intentionally a lightweight, deterministic eval (no LLM-as-judge)
so it's cheap to run and easy to explain in an interview.
"""
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_pipeline import answer_question

EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_set.json")

# Groq's free tier caps tokens-per-minute (8000 TPM as of writing). Each RAG
# call sends ~5 chunks of context plus the system prompt, which can be
# 1500-2500+ tokens per request — firing eval cases back-to-back can exceed
# the limit within 3-4 calls. This pause keeps us comfortably under it.
SECONDS_BETWEEN_CALLS = 15

# Phrases the model uses (per the system prompt's instruction) to signal
# "this isn't covered by the provided context" in its own words. A single
# hardcoded phrase is too brittle — real refusals are phrased many ways
# ("do not contain", "cannot be answered", "not specified", etc.) — so we
# check against this broader set instead of requiring an exact match.
REFUSAL_PHRASES = [
    "not covered", "does not contain", "do not contain", "cannot be answered",
    "does not specify", "do not specify", "no information", "not specified",
    "not addressed", "does not describe", "do not describe", "not available",
    "does not provide", "do not provide", "not mentioned", "does not mention",
]


# Cases in these categories test qualitative behavior (HOW the model refuses
# or handles an edge case, not just whether it refuses) — the automated
# PASS/FAIL is still computed, but flagged here as worth a manual read too.
CATEGORIES_NEEDING_MANUAL_REVIEW = {"prompt_injection", "deprecated_clause_handling"}


def run_eval():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    n = len(cases)
    correct_refusals = 0
    total_unanswerable = 0
    grounded_answers = 0
    total_answerable = 0
    retrieval_hits = 0
    corrections_attempted = 0
    corrections_succeeded = 0
    category_results = {}  # category -> [pass_count, total_count]
    manual_review_answers = []  # (question, category, answer) for the printed appendix

    print(f"Running {n} eval cases...\n")

    for i, case in enumerate(cases):
        q = case["question"]
        category = case.get("category", "uncategorized")
        print(f"Q: {q}")
        result = answer_question(q, verbose=False)

        if i < len(cases) - 1:  # no need to wait after the last case
            time.sleep(SECONDS_BETWEEN_CALLS)

        category_results.setdefault(category, [0, 0])
        category_results[category][1] += 1

        if result.get("correction_attempted"):
            corrections_attempted += 1
            if result["correction_succeeded"]:
                corrections_succeeded += 1
                print("  [self-correction triggered and succeeded — invented citation was auto-fixed]")
            else:
                print("  [self-correction triggered but did not fully resolve the issue]")

        passed = False
        if not case["answerable"]:
            total_unanswerable += 1
            answer_lower = result["answer"].lower()
            if result["refused"] or any(phrase in answer_lower for phrase in REFUSAL_PHRASES):
                correct_refusals += 1
                passed = True
                print("  -> correctly refused / flagged as not covered. PASS")
            else:
                print("  -> [FAIL] answered a question the corpus does not cover:")
                print(f"     {result['answer'][:200]}")
        else:
            total_answerable += 1
            if result["retrieved_clauses"] and case["expected_clause"] in result["retrieved_clauses"]:
                retrieval_hits += 1
            if not result["uncited_or_invented_clauses"]:
                grounded_answers += 1
                passed = True
                print("  -> answered with grounded citations. PASS")
            else:
                print(f"  -> [FAIL] invented/uncited clauses: {result['uncited_or_invented_clauses']}")

        if passed:
            category_results[category][0] += 1

        if category in CATEGORIES_NEEDING_MANUAL_REVIEW:
            manual_review_answers.append((q, category, result["answer"]))

        print()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if total_unanswerable:
        print(f"Refusal accuracy (hallucination guard): {correct_refusals}/{total_unanswerable} "
              f"({100*correct_refusals/total_unanswerable:.0f}%)")
    if total_answerable:
        print(f"Citation groundedness:                  {grounded_answers}/{total_answerable} "
              f"({100*grounded_answers/total_answerable:.0f}%)")
        print(f"Retrieval hit rate:                      {retrieval_hits}/{total_answerable} "
              f"({100*retrieval_hits/total_answerable:.0f}%)")
    if corrections_attempted:
        print(f"Self-correction:                        {corrections_succeeded}/{corrections_attempted} "
              f"invented citations automatically fixed")

    print("\n" + "=" * 50)
    print("BY CATEGORY")
    print("=" * 50)
    for category, (passed, total) in sorted(category_results.items()):
        print(f"{category:35s} {passed}/{total} ({100*passed/total:.0f}%)")

    if manual_review_answers:
        print("\n" + "=" * 50)
        print("MANUAL REVIEW RECOMMENDED (qualitative, not just pass/fail)")
        print("=" * 50)
        for q, category, answer in manual_review_answers:
            print(f"\n[{category}] {q}")
            print(f"  -> {answer}")


if __name__ == "__main__":
    run_eval()
