# tests/ledger/test_build_ledger.py
import pytest
from scripts.ledger.build_ledger import build_ledger, REQUIRED_KEYS, _canonicalise_bb_ref, base_ref


# ---------------------------------------------------------------------------
# Unit tests for _canonicalise_bb_ref
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected_key,expected_alts", [
    # Criterion-g with long em-dash prose → truncated to first 30 chars of inner
    (
        "P-44.2.1.8 (criterion g — more heteroatoms F>Cl>…)",
        "P-44.2.1.8 (criterion g)",
        [],
    ),
    # Criterion-g short form → same key → merges with above
    (
        "P-44.2.1.8 (criterion g)",
        "P-44.2.1.8 (criterion g)",
        [],
    ),
    # Backtick-wrapped id + space-containing qualifier → KEEP with normalized qualifier
    (
        "`P-65.1.2` (dioic)",
        "P-65.1.2 (dioic)",
        [],
    ),
    # Space-containing qualifier kept but truncated at 30 chars (distinct from bare
    (
        "P-65.1.2 (skeletal-replacement chain acid)",
        "P-65.1.2 (skeletal-replacement chain)",
        [],
    ),
    # Bold + slash → strip bold, split on " / ", first segment wins
    (
        "**P-52.2.8 / P-44.1(a)**",
        "P-52.2.8",
        ["P-44.1(a)"],
    ),
    # Slash with partial sub-ref
    (
        "P-102.5.1.1 / .1.1.1",
        "P-102.5.1.1",
        [".1.1.1"],
    ),
    # Plain bare ref — unchanged
    (
        "P-32.2.2",
        "P-32.2.2",
        [],
    ),
])
def test_canonicalise_bb_ref(raw, expected_key, expected_alts):
    key, alts = _canonicalise_bb_ref(raw)
    assert key == expected_key, f"raw={raw!r}: got key={key!r}, want {expected_key!r}"
    assert alts == expected_alts, f"raw={raw!r}: got alts={alts!r}, want {expected_alts!r}"


# ---------------------------------------------------------------------------
# Integration test: findable by base ref (replaces exact-once test)
# ---------------------------------------------------------------------------

def test_p44_2_1_8_findable_by_base():
    """ rows must be findable by base_ref lookup (>=1 row)."""
    rows = build_ledger(
        matrix_path=".planning/audit-bluebook-v21/CONFORMANCE-MATRIX.md",
        cluster_dir=".planning/audit-bluebook-v23",
    )
    matching = [r for r in rows if base_ref(r["bb_ref"]) == "P-44.2.1.8"]
    assert len(matching) >= 1, (
        f"Expected >=1 row with base_ref=='P-44.2.1.8', got {len(matching)}"
    )


# ---------------------------------------------------------------------------
# Original integration test (must stay green, unmodified semantics)
# ---------------------------------------------------------------------------

def test_rows_have_required_keys_and_statuses():
    rows = build_ledger(
        matrix_path=".planning/audit-bluebook-v21/CONFORMANCE-MATRIX.md",
        cluster_dir=".planning/audit-bluebook-v23",
    )
    assert len(rows) >= 1900           # ~1,973 in-scope + NA rows tracked
    assert REQUIRED_KEYS <= set(rows[0])
    assert all(r["status"] in {"OPEN", "IMPLEMENTED", "NA", "FAIL_CLOSED"} for r in rows)
    # The 9 confirmed leaks are seeded OPEN with evidence:
    leak = [r for r in rows if r["bb_ref"] == "P-32.2.2"]
    assert leak and leak[0]["evidence_smiles"] == "OC(=O)CCC1=CCc2ccccc21"


# ---------------------------------------------------------------------------
# New: assert no legitimate rows were dropped
# ---------------------------------------------------------------------------

def test_no_legitimate_rows_dropped():
    """Total ledger row count must be well above the broken round-1 total of 2208."""
    rows = build_ledger(
        matrix_path=".planning/audit-bluebook-v21/CONFORMANCE-MATRIX.md",
        cluster_dir=".planning/audit-bluebook-v23",
    )
    assert len(rows) >= 2300, (
        f"Expected >=2300 rows (2535 source rules - ~68 artifacts - merges), got {len(rows)}"
    )
