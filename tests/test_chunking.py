"""
Offline unit tests for clause-aware chunking (no ML dependencies required).
Run with: python -m pytest tests/test_chunking.py -v
    or:   python tests/test_chunking.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingest import split_into_clauses, sub_split_long_clause


SAMPLE_TEXT = """3GPP TS 24.501 - Sample

5.5.1 Registration procedure

5.5.1.1 General
This is the general text for the registration procedure.
It spans multiple lines.

5.5.1.2 Initial registration
This describes initial registration in detail.
"""


def test_clause_split_count():
    clauses = split_into_clauses(SAMPLE_TEXT, "test.txt", "TS 24.501")
    assert len(clauses) == 3, f"expected 3 clauses, got {len(clauses)}"


def test_clause_numbers_correct():
    clauses = split_into_clauses(SAMPLE_TEXT, "test.txt", "TS 24.501")
    numbers = [c["clause_number"] for c in clauses]
    assert numbers == ["5.5.1", "5.5.1.1", "5.5.1.2"], numbers


def test_clause_body_not_empty():
    clauses = split_into_clauses(SAMPLE_TEXT, "test.txt", "TS 24.501")
    for c in clauses:
        assert "".join(c["body_lines"]).strip() != "" or c["clause_number"] == "5.5.1"


def test_long_clause_gets_subsplit():
    long_text = "x" * 3000
    parts = sub_split_long_clause(long_text, max_chars=1200, overlap=150)
    assert len(parts) > 1
    assert all(len(p) <= 1200 for p in parts)


def test_short_clause_not_split():
    short_text = "short body"
    parts = sub_split_long_clause(short_text, max_chars=1200, overlap=150)
    assert parts == [short_text]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
