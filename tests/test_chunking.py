"""
Offline unit tests for clause-aware chunking (no ML dependencies required).
Run with: python -m pytest tests/test_chunking.py -v
    or:   python tests/test_chunking.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingest import split_into_clauses, sub_split_long_clause, TOC_ENTRY_RE


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


TOC_SAMPLE = """3GPP TS 23.501 - Sample

4.2.5\tData Storage architectures\t50
4.2.5a\tRadio Capabilities Signalling optimisation\t52
5.2.2\tNetwork selection\t93

5.5.1 Registration procedure

5.5.1.1 General
Real body text for the registration procedure clause.
"""


def test_toc_entries_are_filtered_out():
    clauses = split_into_clauses(TOC_SAMPLE, "test.txt", "TS 23.501")
    numbers = [c["clause_number"] for c in clauses]
    # only the two real body clauses should survive; ToC lines must not appear
    assert numbers == ["5.5.1", "5.5.1.1"], numbers


def test_toc_entry_regex_detects_page_number_lines():
    assert TOC_ENTRY_RE.search("4.2.5\tData Storage architectures\t50")
    assert not TOC_ENTRY_RE.search("5.5.1.1 General")


BOILERPLATE_SAMPLE = """3GPP TS 23.501 - Sample

Version x.y.z
where:
x the first digit:
1 presented to TSG for information;
2 presented to TSG for approval;
3 or greater indicates TSG approved document under change control.
y the second digit is incremented for all changes of substance.

1 Scope
The present document defines the Stage 2 system architecture.
"""


def test_lowercase_continuation_lines_not_treated_as_clauses():
    clauses = split_into_clauses(BOILERPLATE_SAMPLE, "test.txt", "TS 23.501")
    numbers = [c["clause_number"] for c in clauses]
    # only the real "1 Scope" heading should be picked up; the boilerplate
    # "3 or greater indicates..." line must be rejected (lowercase continuation)
    assert numbers == ["1"], numbers


DIGIT_LED_TITLE_SAMPLE = """3GPP TS 24.501 - Sample

9.11.3.3\t5GS identity type
Body text for identity type.

9.11.3.4\t5GS mobile identity
The purpose of the 5GS mobile identity information element is to provide either the SUCI, the 5G-GUTI, or other identity.
The 5GS mobile identity is a type 6 information element with a minimum length of 4.

9.11.3.5\t5GS network feature support
Body text for network feature support.
"""


def test_digit_led_clause_titles_are_kept():
    """Regression test: a prior version of the false-positive filter used
    'not title[0].isupper()' to reject boilerplate continuation lines, but
    that ALSO rejected real clause titles starting with a digit (e.g.
    "5GS mobile identity", "5G-GUTI") since a digit is neither upper nor
    lower case in the sense the filter cared about — '5'.isupper() is
    False, so these were incorrectly treated as boilerplate and silently
    dropped from the corpus. This was traced end-to-end from a real
    hallucinated citation during manual testing (the model correctly
    "remembered" clause 9.11.3.4 from training data, but it had been
    silently excluded from the ingested corpus by this bug, so the
    citation-groundedness audit correctly flagged it as unretrieved)."""
    clauses = split_into_clauses(DIGIT_LED_TITLE_SAMPLE, "test.txt", "TS 24.501")
    numbers = [c["clause_number"] for c in clauses]
    assert numbers == ["9.11.3.3", "9.11.3.4", "9.11.3.5"], numbers
    mobile_identity = next(c for c in clauses if c["clause_number"] == "9.11.3.4")
    assert mobile_identity["clause_title"] == "5GS mobile identity"
    assert "SUCI" in "\n".join(mobile_identity["body_lines"])


def test_citation_extraction_requires_sub_level():
    from src.generator import extract_cited_clauses
    # real citations to actual retrieved clauses (always multi-level) are captured
    assert extract_cited_clauses("[doc § 6.2.1]") == {"6.2.1"}
    assert extract_cited_clauses("[doc § 5.15.1] and [doc § 5.5.1.2.2]") == {"5.15.1", "5.5.1.2.2"}
    # informal bare top-level references in prose are NOT mistaken for citations
    assert extract_cited_clauses("as described in §6 of the architecture") == set()


def test_citation_extraction_handles_reordered_format():
    """Regression test: the model doesn't always follow the requested
    '[<source_doc> § <clause_number>]' format — it sometimes writes
    '[<clause_number> § <description>]' instead, putting the clause number
    BEFORE the § symbol. A real invented citation slipped past the audit
    undetected in manual testing because of exactly this formatting drift.
    extract_cited_clauses must catch clause numbers on either side of §."""
    from src.generator import extract_cited_clauses
    # clause number before §, with a description after (the format that broke detection)
    assert extract_cited_clauses("[6.2.1 § some description text]") == {"6.2.1"}
    # mixed formats in the same answer
    assert extract_cited_clauses("[doc § 5.15.1] and [5.5.1.2.2 § description]") == {"5.15.1", "5.5.1.2.2"}
    # bare single-digit before § still correctly rejected (no sub-level)
    assert extract_cited_clauses("[6 § some description]") == set()


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
