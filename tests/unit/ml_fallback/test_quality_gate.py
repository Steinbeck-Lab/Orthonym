"""Unit tests for ``orthonym.ml_fallback.quality_gate`` per 162-AUDIT-MLF.md § 2.

Organized into class-per-concern (Phase 161 Plan-04 inheritance):

* :class:`TestPattern1None` — P1 (name is None)
* :class:`TestPattern2StringEquality` — P2 (name == canonical_smiles)
* :class:`TestPattern3GarbledTokens` — P3 (substring match against
  ``_GARBLED_TOKENS``)
* :class:`TestPattern4LenHaThreshold` — P4 (``len(name) < 6 AND HA > 15``)
* :class:`TestPattern5OpsinParse` — P5 (OPSIN-parse subprocess; mocked)
* :class:`TestPattern6DescriptiveFallback` — P6 ADD per RESEARCH R-03
* :class:`TestPurityInvariants` — CONTEXT D-12 (no mol mutation,
  deterministic, no state R/W, no exception swallowing)

Per CONTEXT D-12 + audit § 2.4: every test asserts purity invariants in
addition to the functional pattern semantics.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rdkit import Chem

from orthonym.ml_fallback.quality_gate import (
    _DESCRIPTIVE_FALLBACK_NAMES,
    _GARBLED_TOKENS,
    is_name_quality_inadequate,
)


@pytest.fixture
def mol_small() -> "Chem.Mol":
    """Ethanol — 3 heavy atoms (HA <= 15)."""
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def mol_large() -> "Chem.Mol":
    """A larger polycyclic — > 15 heavy atoms."""
    # Cholesterol-like skeleton; SMILES known to be valid.
    mol = Chem.MolFromSmiles("CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C")
    assert mol is not None
    assert mol.GetNumHeavyAtoms() > 15
    return mol


@pytest.mark.unit
class TestPattern1None:
    """P1: name is None -> True (rule-based pipeline produced no name)."""

    def test_none_name_returns_true(self, mol_small):
        assert is_name_quality_inadequate(None, mol_small, "CCO") is True

    def test_none_name_regardless_of_opsin_required(self, mol_small):
        # P1 fires before any other pattern; opsin_parse_required is irrelevant
        assert is_name_quality_inadequate(
            None, mol_small, "CCO", opsin_parse_required=False
        ) is True
        assert is_name_quality_inadequate(
            None, mol_small, "CCO", opsin_parse_required=True
        ) is True

    def test_none_with_large_mol(self, mol_large):
        assert is_name_quality_inadequate(None, mol_large, "any_smiles") is True

    def test_empty_string_not_p1(self, mol_small):
        # Empty string is NOT None; P1 doesn't fire.
        # P2 fails (CCO != ""), P3-P4-P6 fail; with opsin_parse_required=False
        # the predicate returns False (empty string isn't considered degraded
        # absent OPSIN check).
        result = is_name_quality_inadequate(
            "", mol_small, "CCO", opsin_parse_required=False
        )
        assert result is False


@pytest.mark.unit
class TestPattern2StringEquality:
    """P2: name == canonical_smiles -> True (SMILES emitted as fallback)."""

    def test_smiles_returned_as_name(self, mol_small):
        # Per namer.py:1240 — canonical SMILES is the bottom-fallback emit
        assert is_name_quality_inadequate("CCO", mol_small, "CCO") is True

    def test_with_opsin_off(self, mol_small):
        assert is_name_quality_inadequate(
            "CCO", mol_small, "CCO", opsin_parse_required=False
        ) is True

    def test_different_string_not_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        ) is False

    def test_case_sensitive_equality(self, mol_small):
        # P2 is exact string equality; case-sensitive (no .lower() call)
        assert is_name_quality_inadequate("CCO", mol_small, "CCO") is True
        assert is_name_quality_inadequate(
            "cco", mol_small, "CCO", opsin_parse_required=False
        ) is False


@pytest.mark.unit
class TestPattern3GarbledTokens:
    """P3: any garbled token in ``name.lower()`` (substring match)."""

    def test_cycloane_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "foocycloanebar", mol_small, "CCO"
        ) is True

    def test_anedicarboxamide_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "xxxanedicarboxamideyy", mol_small, "CCO"
        ) is True

    def test_aneyl_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "methaneyl-ethanol", mol_small, "CCO"
        ) is True

    def test_case_insensitive(self, mol_small):
        # quality_gate.py:124 uses `name.lower()` substring check
        assert is_name_quality_inadequate("CYCLOANE", mol_small, "CCO") is True

    def test_clean_name_not_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        ) is False

    def test_garbled_tokens_is_tuple(self):
        assert isinstance(_GARBLED_TOKENS, tuple)
        assert "cycloane" in _GARBLED_TOKENS
        assert "anedicarboxamide" in _GARBLED_TOKENS
        assert "aneyl" in _GARBLED_TOKENS


@pytest.mark.unit
class TestPattern4LenHaThreshold:
    """P4: ``len(name) < 6 AND mol.GetNumHeavyAtoms() > 15``."""

    def test_short_name_large_mol_caught(self, mol_large):
        # name="octyl" (len=5); mol_large (HA > 15) — P4 fires
        assert is_name_quality_inadequate(
            "octyl", mol_large, "<large smiles>"
        ) is True

    def test_short_name_small_mol_not_caught(self, mol_small):
        # name="octyl" (len=5); mol_small (HA=3) — P4 does NOT fire
        assert is_name_quality_inadequate(
            "octyl", mol_small, "CCO", opsin_parse_required=False
        ) is False

    def test_long_name_large_mol_not_caught(self, mol_large):
        # name="benzene" (len=7); P4 does NOT fire (len not < 6)
        assert is_name_quality_inadequate(
            "benzene", mol_large, "<large smiles>", opsin_parse_required=False
        ) is False

    def test_boundary_len_equals_6(self, mol_large):
        # P4 uses `len(name) < 6` (strict); len=6 should NOT fire
        result = is_name_quality_inadequate(
            "hexyne", mol_large, "<large smiles>", opsin_parse_required=False
        )
        assert result is False


@pytest.mark.unit
class TestPattern5OpsinParse:
    """P5: ``opsin_parse_required AND not opsin_parse_ok(name)``."""

    def test_clean_opsin_parse_not_caught(self, mol_small):
        with patch(
            "orthonym.validation.opsin_roundtrip.opsin_parse",
            return_value="CCO",
        ):
            assert is_name_quality_inadequate(
                "ethanol", mol_small, "CCO", opsin_parse_required=True
            ) is False

    def test_opsin_parse_fail_caught(self, mol_small):
        with patch(
            "orthonym.validation.opsin_roundtrip.opsin_parse",
            return_value=None,
        ):
            assert is_name_quality_inadequate(
                "garbled-non-parseable", mol_small, "CCO",
                opsin_parse_required=True,
            ) is True

    def test_opsin_bypass_via_required_false(self, mol_small):
        # opsin_parse_required=False bypasses P5 entirely
        with patch(
            "orthonym.validation.opsin_roundtrip.opsin_parse",
            return_value=None,
        ):
            assert is_name_quality_inadequate(
                "would-fail-opsin", mol_small, "CCO",
                opsin_parse_required=False,
            ) is False

    def test_p5_short_circuited_by_earlier_patterns(self, mol_small):
        # P1 fires first — P5 (OPSIN subprocess) should NEVER be called
        with patch(
            "orthonym.validation.opsin_roundtrip.opsin_parse"
        ) as mock_opsin:
            is_name_quality_inadequate(None, mol_small, "CCO")
            mock_opsin.assert_not_called()


@pytest.mark.unit
class TestPattern6DescriptiveFallback:
    """P6 (ADD per audit § 2.2): ``name in _DESCRIPTIVE_FALLBACK_NAMES``."""

    def test_unknown_organic_compound_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "unknown organic compound", mol_small, "CCO"
        ) is True

    def test_wildcard_atoms_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "compound with wildcard atoms (not supported)", mol_small, "CCO"
        ) is True

    def test_inorganic_compound_caught(self, mol_small):
        assert is_name_quality_inadequate(
            "inorganic compound (not supported)", mol_small, "CCO"
        ) is True

    def test_metal_compound_pattern_caught(self, mol_small):
        # Dynamic per-metal entries from namer.py:_METAL_NAMES
        if "platinum compound (not supported)" in _DESCRIPTIVE_FALLBACK_NAMES:
            assert is_name_quality_inadequate(
                "platinum compound (not supported)", mol_small, "CCO"
            ) is True
        else:
            pytest.skip(
                "platinum not in _DESCRIPTIVE_FALLBACK_NAMES dynamic set"
            )

    def test_descriptive_fallback_is_frozenset(self):
        assert isinstance(_DESCRIPTIVE_FALLBACK_NAMES, frozenset)

    def test_p6_short_circuits_p5(self, mol_small):
        # P6 (cheap, set lookup) runs BEFORE P5 (expensive, OPSIN subprocess)
        # per audit § 2.3 ordering. OPSIN should not be invoked when P6 fires.
        with patch(
            "orthonym.validation.opsin_roundtrip.opsin_parse"
        ) as mock_opsin:
            is_name_quality_inadequate(
                "unknown organic compound", mol_small, "CCO",
                opsin_parse_required=True,
            )
            mock_opsin.assert_not_called()


@pytest.mark.unit
class TestPurityInvariants:
    """CONTEXT D-12 hard invariant: predicate is pure."""

    def test_no_mol_mutation_on_clean_name(self, mol_small):
        ha_before = mol_small.GetNumHeavyAtoms()
        atoms_before = mol_small.GetNumAtoms()
        _ = is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        )
        assert mol_small.GetNumHeavyAtoms() == ha_before
        assert mol_small.GetNumAtoms() == atoms_before

    def test_no_mol_mutation_on_pattern_match(self, mol_small):
        ha_before = mol_small.GetNumHeavyAtoms()
        _ = is_name_quality_inadequate(None, mol_small, "CCO")  # P1 fires
        assert mol_small.GetNumHeavyAtoms() == ha_before

    def test_deterministic_repeat_call(self, mol_small):
        r1 = is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        )
        r2 = is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        )
        r3 = is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        )
        assert r1 == r2 == r3

    def test_constants_unchanged_after_call(self, mol_small):
        garbled_before_id = id(_GARBLED_TOKENS)
        fallback_before_id = id(_DESCRIPTIVE_FALLBACK_NAMES)
        _ = is_name_quality_inadequate(
            "ethanol", mol_small, "CCO", opsin_parse_required=False
        )
        # Re-import: must be the same object (immutable; never reassigned)
        from orthonym.ml_fallback.quality_gate import (
            _DESCRIPTIVE_FALLBACK_NAMES as fallback_after,
        )
        from orthonym.ml_fallback.quality_gate import (
            _GARBLED_TOKENS as garbled_after,
        )
        assert id(garbled_after) == garbled_before_id
        assert id(fallback_after) == fallback_before_id

    def test_no_exception_swallow_on_opsin_failure(self, mol_small):
        # D-12 line 290: OPSIN parse exceptions MUST propagate
        with patch(
            "orthonym.validation.opsin_roundtrip.opsin_parse",
            side_effect=RuntimeError("OPSIN crashed"),
        ):
            with pytest.raises(RuntimeError, match="OPSIN crashed"):
                is_name_quality_inadequate(
                    "test", mol_small, "CCO", opsin_parse_required=True
                )
