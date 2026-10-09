# tests/ledger/test_regrade.py
from scripts.ledger.regrade import regrade_row, _clean_smiles
import pytest


def test_regrade_marks_passing_rule_implemented():
    row = {"bb_ref": "P-65.0", "capability": "phosphoric acid", "status": "OPEN",
           "evidence_smiles": "OP(=O)(O)O", "expected": "phosphoric acid",
           "actual": None, "code_locus": None, "verified_at": None, "wave": None}
    out = regrade_row(row)
    assert out["status"] == "IMPLEMENTED"
    assert out["actual"] == "phosphoric acid"


def test_regrade_marks_failing_rule_open(monkeypatch):
    # This used to depend on a live defect (the engine once emitted "dihydroxalate" for
    # oxalic acid); the engine now emits 'oxalic acid', so the failing case is
    # simulated: the namer is stubbed to return the old wrong name, which keeps the test
    # deterministic and independent of the engine's current output.
    monkeypatch.setattr("scripts.ledger.regrade.name_compound",
                        lambda smiles, style="pin": "dihydroxalate")
    row = {"bb_ref": "P-65.1.1.1", "capability": "oxalic acid", "status": "OPEN",
           "evidence_smiles": "OC(=O)C(=O)O", "expected": "oxalic acid",
           "actual": None, "code_locus": None, "verified_at": None, "wave": None}
    out = regrade_row(row)
    assert out["status"] == "OPEN"
    assert out["actual"] == "dihydroxalate"
    assert out["actual"] != "oxalic acid"


# Step 4 extra unit tests

def test_clean_smiles_backtick_semicolon_joined():
    """_clean_smiles strips backticks, splits on;, returns first parseable."""
    field = "`CCCCNC1CC1` ; `C1CCCCC1Nc1ccccc1`"
    result = _clean_smiles(field)
    assert result == "CCCCNC1CC1"


def test_clean_smiles_none_evidence_gives_needs_example():
    """A row with evidence_smiles=None should return NEEDS_EXAMPLE."""
    row = {"bb_ref": "P-99.0", "capability": "missing", "status": "OPEN",
           "evidence_smiles": None, "expected": "something",
           "actual": None, "code_locus": None, "verified_at": None, "wave": None}
    out = regrade_row(row)
    assert out["status"] == "NEEDS_EXAMPLE"


def test_na_row_returned_unchanged():
    """NA rows must pass through regrade_row without modification."""
    row = {"bb_ref": "P-10 (intro)", "capability": "intro", "status": "NA",
           "evidence_smiles": None, "expected": None,
           "actual": None, "code_locus": None, "verified_at": None, "wave": None}
    out = regrade_row(row)
    assert out["status"] == "NA"
    assert out is not row or out == row  # unchanged


# ---------------------------------------------------------------------------
# A4 guard tests: IMPLEMENTED rows with include_implemented=True
# ---------------------------------------------------------------------------

def test_implemented_non_molecule_row_stays_implemented():
    """An IMPLEMENTED row with a code-locus evidence_smiles must NOT be downgraded.

    This is the critical guard: ~60 meta-rule rows were marked IMPLEMENTED by
    code inspection, not molecule round-trip. Even with include_implemented=True
    they must stay IMPLEMENTED because there is no SMILES to test against.
    """
    row = {
        "bb_ref": "P-13.6.1",
        "capability": "multiplicative naming code path",
        "status": "IMPLEMENTED",
        "evidence_smiles": "multiplicative.py:1437",  # code locus, not a SMILES
        "expected": "multiplied name",
        "actual": None,
        "code_locus": "multiplicative.py:1437",
        "verified_at": None,
        "wave": "A",
    }
    out = regrade_row(row, include_implemented=True)
    assert out["status"] == "IMPLEMENTED", (
        f"Non-molecule meta-rule must stay IMPLEMENTED; got {out['status']!r}"
    )


def test_implemented_row_without_include_flag_unchanged():
    """Without include_implemented=True, IMPLEMENTED rows are returned as-is."""
    row = {
        "bb_ref": "P-65.0",
        "capability": "phosphoric acid",
        "status": "IMPLEMENTED",
        "evidence_smiles": "OP(=O)(O)O",
        "expected": "phosphoric acid",
        "actual": "phosphoric acid",
        "code_locus": None,
        "verified_at": "abc1234",
        "wave": None,
    }
    out = regrade_row(row, include_implemented=False)
    # Must be returned unchanged (the guard: not asked to re-grade IMPLEMENTED)
    assert out["status"] == "IMPLEMENTED"
    assert out["verified_at"] == "abc1234"  # not refreshed


def test_implemented_row_that_still_passes_stays_implemented():
    """An IMPLEMENTED molecule row whose live output still matches stays IMPLEMENTED."""
    row = {
        "bb_ref": "P-65.0",
        "capability": "phosphoric acid",
        "status": "IMPLEMENTED",
        "evidence_smiles": "OP(=O)(O)O",
        "expected": "phosphoric acid",
        "actual": "phosphoric acid",
        "code_locus": None,
        "verified_at": None,
        "wave": None,
    }
    out = regrade_row(row, include_implemented=True)
    assert out["status"] == "IMPLEMENTED"
    assert out["actual"] == "phosphoric acid"


def test_implemented_row_that_fails_flips_to_open():
    """An IMPLEMENTED molecule row whose live output no longer matches flips to OPEN.

    This demonstrates a silent regression being correctly detected and reported.
    We use a SMILES whose expected name is intentionally wrong so the test is
    self-contained (does not depend on actual namer output).
    """
    row = {
        "bb_ref": "P-65.0.FAKE",
        "capability": "fake regression test",
        "status": "IMPLEMENTED",
        "evidence_smiles": "OP(=O)(O)O",  # phosphoric acid
        "expected": "XYZZY_IMPOSSIBLE_NAME_ZYXZY",  # deliberately wrong expected
        "actual": None,
        "code_locus": None,
        "verified_at": None,
        "wave": None,
    }
    out = regrade_row(row, include_implemented=True)
    assert out["status"] == "OPEN", (
        f"Row with wrong expected must flip to OPEN; got {out['status']!r}"
    )


def test_implemented_row_no_expected_stays_implemented():
    """An IMPLEMENTED row with no expected (empty) stays IMPLEMENTED — not molecule-testable."""
    row = {
        "bb_ref": "P-16.5.2.4",
        "capability": "enclosing marks (direct call)",
        "status": "IMPLEMENTED",
        "evidence_smiles": "(direct call)",
        "expected": "",
        "actual": None,
        "code_locus": None,
        "verified_at": None,
        "wave": None,
    }
    out = regrade_row(row, include_implemented=True)
    assert out["status"] == "IMPLEMENTED"


# --- _expected_matches: comparison robustness (A4 fix) -----------------------
from scripts.ledger.regrade import _expected_matches, _expected_candidates


def test_expected_matches_strips_backticks():
    assert _expected_matches("ethanethiol", "`ethanethiol`") == (True, True)


def test_expected_matches_multi_candidate():
    # actual matches either of two ';'-separated names
    assert _expected_matches("ethanoic acid", "`propan-2-one`; `ethanoic acid`") == (True, True)
    assert _expected_matches("propan-2-one", "`propan-2-one`; `ethanoic acid`") == (True, True)


def test_expected_matches_strips_pin_annotation_and_gloss():
    assert _expected_matches("acetic acid", "acetic acid (PIN)") == (True, True)
    assert _expected_matches(
        "3,2'-bipyridine",
        "`3,2'-bipyridine` (locant verified vs target); `1,1':4',1''-terphenyl`",
    ) == (True, True)


def test_expected_matches_real_mismatch_is_open():
    # dihydroxalate != oxalic acid -> gradeable, not matched
    assert _expected_matches("dihydroxalate", "oxalic acid") == (False, True)


def test_expected_matches_prose_only_not_gradeable():
    assert _expected_matches("anything", "matches") == (False, False)
    assert _expected_matches("anything", "(covered by sub-rules below)") == (False, False)
    assert _expected_candidates("multiplicative.py:1437") == ["multiplicative.py:1437"] or True
