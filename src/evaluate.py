"""Evaluate retrieval, groundedness, and refusal behavior of the RAG pipeline."""
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_pipeline import answer_question

EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_set.json")

# Keep eval requests below the Groq free-tier token limit.
SECONDS_BETWEEN_CALLS = 15

# Match common ways the model indicates that the corpus does not cover a question.
REFUSAL_PHRASES = [
    "not cover", "does not contain", "do not contain", "cannot be answered",
    "does not specify", "do not specify", "no information", "not specified",
    "not addressed", "does not describe", "do not describe", "not available",
    "does not provide", "do not provide", "not mentioned", "does not mention",
]


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
    category_results = {} 
    manual_review_answers = [] 

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
