"""Run retrieval and generation for a user question."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import retrieve
from src.generator import generate_answer


def answer_question(question, top_k=None, verbose=True):
    kwargs = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    chunks = retrieve(question, **kwargs)

    if verbose:
        if chunks:
            print(f"Retrieved {len(chunks)} relevant chunk(s):")
            for c in chunks:
                print(f"  [{c['score']:.3f}] {c['source_doc']} § {c['clause_number']} {c['clause_title']}")
        else:
            print("No chunks cleared the relevance threshold — refusing rather than guessing.")

    result = generate_answer(question, chunks)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    result = answer_question(args.question)
    print("\n--- ANSWER ---")
    print(result["answer"])
    if result["uncited_or_invented_clauses"]:
        print(f"\n[WARNING] Model cited clause(s) not present in retrieved context: "
              f"{result['uncited_or_invented_clauses']}")
