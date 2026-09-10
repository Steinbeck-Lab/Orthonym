"""
tests/ledger/test_fill_examples_2b.py — Unit tests for fill_examples_2b.py.

Tests (RED → GREEN):
  1. Mechanism A picks the SMILES-bearing sibling unit row for a base_ref with
     multiple sub-rows; it prefers the EXACT rule_id match over a base_ref
     fallback for a row that has a SMILES.
  2. OPSIN name→SMILES helper returns a SMILES for 'spiro[4.5]decane' and
     returns None for 'yohimban' (guard: skip if no JVM or OPSIN jar found).
  3. A prose-only candidate is skipped (not passed to OPSIN / not used as a name).
  4. batch_opsin returns a parallel list (same length as input, None for failures).
"""

from __future__ import annotations

import pytest
from typing import Optional

from scripts.ledger.fill_examples_2b import (
    OPSIN_JAR,
    _mech_a_recover,
    _is_prose,
    _gather_name_candidates,
    batch_opsin,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal NEEDS_EXAMPLE ledger row
# ---------------------------------------------------------------------------

def _row(bb_ref: str, expected=None, evidence_smiles=None, capability=""):
    return {
        "bb_ref": bb_ref,
        "capability": capability,
        "status": "NEEDS_EXAMPLE",
        "wave": None,
        "evidence_smiles": evidence_smiles,
        "expected": expected,
        "actual": None,
        "code_locus": None,
        "verified_at": None,
    }


# ---------------------------------------------------------------------------
# 1. Mechanism A: sibling sub-row SMILES recovery
# ---------------------------------------------------------------------------

class TestMechanismA:
    """Mechanism A picks the correct sub-row when multiple sub-rows share a base_ref."""

    def test_exact_match_preferred_over_base_ref(self):
        """
        Given two source rows that share the same base_ref (P-44.2):
          - 'P-44.2 (criterion a)' has NO SMILES
          - 'P-44.2 (criterion b)' HAS a SMILES and an expected
        The ledger row for 'P-44.2 (criterion b)' should recover the SMILES
        via the exact match path, not be blocked by the SMILES-less 'criterion a' row.
        """
        smiles_b = "CCO"
        exact_index = {
            "P-44.2 (criterion a)": [
                {
                    "rule_id": "P-44.2 (criterion a)",
                    "test_smiles_cell": "—",
                    "expected_cell": "—",
                    "resolved_smiles": None,
                }
            ],
            "P-44.2 (criterion b)": [
                {
                    "rule_id": "P-44.2 (criterion b)",
                    "test_smiles_cell": f"`{smiles_b}`",
                    "expected_cell": "`ethanol`",
                    "resolved_smiles": smiles_b,
                }
            ],
        }
        base_index = {
            "P-44.2": [
                # criterion a first (no SMILES)
                {
                    "rule_id": "P-44.2 (criterion a)",
                    "test_smiles_cell": "—",
                    "expected_cell": "—",
                    "resolved_smiles": None,
                },
                # criterion b second (has SMILES)
                {
                    "rule_id": "P-44.2 (criterion b)",
                    "test_smiles_cell": f"`{smiles_b}`",
                    "expected_cell": "`ethanol`",
                    "resolved_smiles": smiles_b,
                },
            ]
        }

        row = _row("P-44.2 (criterion b)")
        result = _mech_a_recover(row, exact_index, base_index)

        assert result is not None, "Mechanism A should recover the SMILES from the exact sub-row"
        recovered_smiles, recovered_expected = result
        assert recovered_smiles == smiles_b
        assert recovered_expected == "ethanol"

    def test_falls_back_to_base_ref_when_no_exact_match(self):
        """
        A row whose bb_ref has no exact entry falls back to base_ref, picking the
        first entry with a SMILES.
        """
        smiles_c = "C1CCCCC1"
        exact_index: dict = {}
        base_index = {
            "P-33.1": [
                {
                    "rule_id": "P-33.1 (header)",
                    "test_smiles_cell": "—",
                    "expected_cell": "—",
                    "resolved_smiles": None,
                },
                {
                    "rule_id": "P-33.1 (example)",
                    "test_smiles_cell": f"`{smiles_c}`",
                    "expected_cell": "`cyclohexane`",
                    "resolved_smiles": smiles_c,
                },
            ]
        }

        row = _row("P-33.1 (sub-case)")
        result = _mech_a_recover(row, exact_index, base_index)

        assert result is not None, "Mechanism A should fall back to base_ref and find the SMILES"
        recovered_smiles, recovered_expected = result
        assert recovered_smiles == smiles_c
        assert recovered_expected == "cyclohexane"

    def test_returns_none_when_no_smiles_available(self):
        """Row with no SMILES in any source entry returns None."""
        exact_index = {
            "P-99.1": [
                {
                    "rule_id": "P-99.1",
                    "test_smiles_cell": "—",
                    "expected_cell": "—",
                    "resolved_smiles": None,
                }
            ]
        }
        base_index = {"P-99.1": exact_index["P-99.1"]}

        row = _row("P-99.1")
        result = _mech_a_recover(row, exact_index, base_index)
        assert result is None


# ---------------------------------------------------------------------------
# 2. Mechanism B: OPSIN name helper
# ---------------------------------------------------------------------------

# Skip if no JVM or OPSIN jar available
_opsin_available = OPSIN_JAR is not None

@pytest.mark.skipif(not _opsin_available, reason="OPSIN jar not found — skipping OPSIN tests")
class TestMechanismBOpsin:
    def test_spiro_decane_returns_smiles(self):
        """OPSIN can parse 'spiro[4.5]decane' and return a valid SMILES."""
        results = batch_opsin(["spiro[4.5]decane"])
        assert len(results) == 1
        smi = results[0]
        assert smi is not None, "OPSIN should parse 'spiro[4.5]decane'"
        # RDKit should agree it is 10 carbons
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            assert mol.GetNumAtoms() == 10
        except ImportError:
            pass  # RDKit not installed — trust OPSIN

    def test_yohimban_returns_none(self):
        """OPSIN cannot parse the trivial NP name 'yohimban'; should return None."""
        results = batch_opsin(["yohimban"])
        assert len(results) == 1
        assert results[0] is None, "OPSIN should NOT be able to parse 'yohimban'"

    def test_batch_mixed(self):
        """batch_opsin returns a same-length list with None for unparseable names."""
        names = ["spiro[4.5]decane", "yohimban", "benzene"]
        results = batch_opsin(names)
        assert len(results) == 3
        assert results[0] is not None  # spiro[4.5]decane
        assert results[1] is None       # yohimban
        assert results[2] is not None   # benzene

    def test_empty_input(self):
        """batch_opsin with empty list returns empty list."""
        assert batch_opsin([]) == []


# ---------------------------------------------------------------------------
# 3. Prose candidate is skipped
# ---------------------------------------------------------------------------

class TestProseFilter:
    def test_prose_markers_rejected(self):
        """_is_prose returns True for various prose/code strings."""
        assert _is_prose("substitutive form acceptable") is True
        assert _is_prose("PIN) annotation") is True
        assert _is_prose("cross-ref to P-26") is True
        assert _is_prose("_try_algorithmic_fusion_name") is True
        assert _is_prose("build_systematic_fusion_name('isoquinoline','benzene','[g]')") is True
        assert _is_prose("(as above)") is True
        assert _is_prose("via mechanism A") is True

    def test_chemical_names_not_rejected(self):
        """_is_prose returns False for genuine IUPAC names."""
        assert _is_prose("spiro[4.5]decane") is False
        assert _is_prose("anthra[2,3-b]furan") is False
        assert _is_prose("benzene") is False
        assert _is_prose("21H-biline") is False
        assert _is_prose("porphyrin") is False

    def test_gather_name_candidates_skips_prose(self):
        """_gather_name_candidates excludes prose-only cells."""
        row = _row(
            bb_ref="P-25.3.1.3 (general)",
            evidence_smiles="`build_systematic_fusion_name('isoquinoline','benzene','[g]')`",
            expected="`benzo[g]isoquinoline`",
            capability="Name = attached-prefix(es) + fusion descriptor + parent",
        )
        candidates = _gather_name_candidates(row)
        # 'benzo[g]isoquinoline' should appear (it's a valid name)
        assert "benzo[g]isoquinoline" in candidates
        # The Python call string should NOT appear
        for c in candidates:
            assert "build_systematic_fusion_name" not in c

    def test_gather_name_candidates_empty_row(self):
        """Row with all-None/prose fields yields an empty or prose-free candidate list."""
        row = _row(
            bb_ref="P-25.3.2.4 (header)",
            evidence_smiles=None,
            expected=None,
            capability="Seniority criteria apply if a choice remains.",
        )
        candidates = _gather_name_candidates(row)
        # No names — capability is pure prose
        for c in candidates:
            assert not _is_prose(c), f"Prose candidate leaked: {c!r}"
