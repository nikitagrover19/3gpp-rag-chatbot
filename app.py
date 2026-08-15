"""
Simple CLI chat loop for the 3GPP RAG chatbot.

Usage:
    python app.py
"""
import sys
from src.rag_pipeline import answer_question


def main():
    print("=" * 60)
    print("3GPP RAG Chatbot (grounded, local, free)")
    print("Ask a question about the ingested 3GPP specs. Ctrl+C to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if not question:
            continue

        result = answer_question(question)
        print("\nBot:", result["answer"])
        if result["uncited_or_invented_clauses"]:
            print(f"\n[groundedness warning] cited clause(s) not found in retrieved "
                  f"context: {result['uncited_or_invented_clauses']}")


if __name__ == "__main__":
    main()
