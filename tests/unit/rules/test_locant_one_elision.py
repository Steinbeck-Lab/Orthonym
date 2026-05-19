"""
Unit tests for centralized locant-1 elision logic.

Tests the should_omit_locant_one() function from naming_utils.py,
which is the single source of truth for all locant-1 omission
decisions per IUPAC P-14.3.4.
"""

import pytest

from orthonym.assembly.naming_utils import should_omit_locant_one, TERMINAL_FG_TYPES


# ============================================================================
# Suffix context tests
# ============================================================================


class TestSuffixContext:
    """Tests for context='suffix' -- principal group locant elision."""

    def test_terminal_carboxylic_acid_omits_locant(self):
        """Carboxylic acid is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="carboxylic_acid") is True

    def test_terminal_aldehyde_omits_locant(self):
        """Aldehyde is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="aldehyde") is True

    def test_terminal_nitrile_omits_locant(self):
        """Nitrile is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="nitrile") is True

    def test_terminal_primary_amide_omits_locant(self):
        """Primary amide is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="primary_amide") is True

    def test_terminal_acid_chloride_omits_locant(self):
        """Acid chloride is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="acid_chloride") is True

    def test_terminal_thioic_S_acid_omits_locant(self):
        """Thioic S-acid is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="thioic_S_acid") is True

    def test_terminal_dithioic_acid_omits_locant(self):
        """Dithioic acid is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="dithioic_acid") is True

    def test_terminal_carbamic_acid_omits_locant(self):
        """Carbamic acid is terminal: locant-1 implicit."""
        assert should_omit_locant_one(context="suffix", fg_type="carbamic_acid") is True

    def test_nonterminal_alcohol_keeps_locant(self):
        """Alcohol is not terminal: locant-1 must be explicit."""
        assert should_omit_locant_one(context="suffix", fg_type="alcohol") is False

    def test_nonterminal_ketone_keeps_locant(self):
        """Ketone is not terminal: locant-1 must be explicit."""
        assert should_omit_locant_one(context="suffix", fg_type="ketone") is False

    def test_nonterminal_amine_keeps_locant(self):
        """Amine is not terminal: locant-1 must be explicit."""
        assert should_omit_locant_one(context="suffix", fg_type="amine") is False

    def test_empty_fg_type_keeps_locant(self):
        """Empty fg_type should not trigger terminal group omission."""
        assert should_omit_locant_one(context="suffix", fg_type="") is False

    def test_methane_derivative_suffix_omits(self):
        """Methane derivative (chain_length=1): always omit regardless of fg_type."""
        assert should_omit_locant_one(context="suffix", chain_length=1, fg_type="alcohol") is True


# ============================================================================
# Prefix context tests
# ============================================================================


class TestPrefixContext:
    """Tests for context='prefix' -- substituent locant elision."""

    def test_methane_derivative_prefix_omits(self):
        """Methane derivative (chain_length=1): always omit for prefix."""
        assert should_omit_locant_one(context="prefix", chain_length=1) is True

    def test_monosubstituted_carbocyclic_ring_omits(self):
        """Monosubstituted carbocyclic ring: omit locant-1."""
        assert should_omit_locant_one(
            context="prefix",
            is_ring=True,
            is_heterocyclic=False,
            is_monosubstituted=True,
        ) is True

    def test_monosubstituted_heterocyclic_ring_keeps(self):
        """Monosubstituted heterocyclic ring: keep locant (position matters)."""
        assert should_omit_locant_one(
            context="prefix",
            is_ring=True,
            is_heterocyclic=True,
            is_monosubstituted=True,
        ) is False

    def test_polysubstituted_ring_keeps(self):
        """Polysubstituted ring: keep locants."""
        assert should_omit_locant_one(
            context="prefix",
            is_ring=True,
            is_heterocyclic=False,
            is_monosubstituted=False,
        ) is False

    def test_monosubstituted_chain_at_pos1_omits(self):
        """Monosubstituted chain at position 1: omit locant."""
        assert should_omit_locant_one(
            context="prefix",
            is_ring=False,
            is_monosubstituted=True,
            chain_length=5,
        ) is True

    def test_polysubstituted_chain_keeps(self):
        """Multiple substituents on chain: keep locants."""
        assert should_omit_locant_one(
            context="prefix",
            is_ring=False,
            is_monosubstituted=False,
            chain_length=5,
        ) is False

    def test_polysubstituted_heterocyclic_ring_keeps(self):
        """Polysubstituted heterocyclic ring: keep locants."""
        assert should_omit_locant_one(
            context="prefix",
            is_ring=True,
            is_heterocyclic=True,
            is_monosubstituted=False,
        ) is False


# ============================================================================
# Bond context tests
# ============================================================================


class TestBondContext:
    """Tests for context='bond' -- unsaturation bond locant elision."""

    def test_two_carbon_chain_omits_bond_locant(self):
        """Ethene/ethyne: bond locant omitted (only one position)."""
        assert should_omit_locant_one(context="bond", chain_length=2) is True

    def test_three_carbon_chain_keeps_bond_locant(self):
        """Propene: bond locant needed (could be prop-1-ene or prop-2-ene)."""
        assert should_omit_locant_one(context="bond", chain_length=3) is False

    def test_mono_cycloalkene_omits_bond_locant(self):
        """Cyclohexene: single double bond in ring, locant omitted."""
        assert should_omit_locant_one(
            context="bond",
            is_ring=True,
            is_monosubstituted=True,  # single double bond
        ) is True

    def test_methane_derivative_bond_omits(self):
        """Chain_length=1 with bond context: still omit."""
        assert should_omit_locant_one(context="bond", chain_length=1) is True


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge cases for should_omit_locant_one."""

    def test_chain_length_zero_default(self):
        """chain_length=0 with prefix context: should not omit."""
        assert should_omit_locant_one(context="prefix", chain_length=0) is False

    def test_all_defaults_returns_false(self):
        """With all defaults (empty context "prefix"), returns False."""
        assert should_omit_locant_one(context="prefix") is False

    def test_keyword_only_args(self):
        """Verify arguments are keyword-only (asterisk before first param)."""
        # This should raise TypeError if positional args are used
        with pytest.raises(TypeError):
            should_omit_locant_one("suffix")  # type: ignore


# ============================================================================
# TERMINAL_FG_TYPES constant tests
# ============================================================================


class TestTerminalFGTypes:
    """Tests for the TERMINAL_FG_TYPES frozenset."""

    def test_contains_all_terminal_groups(self):
        """TERMINAL_FG_TYPES must contain all terminal group types.

        Pre-Phase-163: 13 entries. Phase 163 + 163.1 added 9 chalcogen-replacement
        terminal groups per IUPAC P-66.6.3 + P-66.1.4.1.1:
          - chalcogen acids (6): selenoic_Se_acid, selenoic_O_acid, diselenoic_acid,
            telluroic_Te_acid, telluroic_O_acid, ditelluroic_acid
          - chalcogen amides (3): thioamide, selenoamide, telluroamide
          - chalcogen aldehydes (3): thioaldehyde, selenoaldehyde, telluroaldehyde
        Total: 13 + 12 = 25 entries.
        """
        expected = {
            "carboxylic_acid", "aldehyde", "nitrile",
            "primary_amide", "secondary_amide", "tertiary_amide",
            "acid_chloride", "acid_bromide", "acid_fluoride",
            "thioic_S_acid", "thioic_O_acid", "dithioic_acid",
            "carbamic_acid",
            # Phase 163 FRN-A chalcogen acids
            "selenoic_Se_acid", "selenoic_O_acid", "diselenoic_acid",
            "telluroic_Te_acid", "telluroic_O_acid", "ditelluroic_acid",
            # Phase 163 FRN-B chalcogen amides
            "thioamide", "selenoamide", "telluroamide",
            # Phase 163 FRN-C chalcogen aldehydes
            "thioaldehyde", "selenoaldehyde", "telluroaldehyde",
        }
        assert TERMINAL_FG_TYPES == expected

    def test_is_frozenset(self):
        """TERMINAL_FG_TYPES must be a frozenset (immutable)."""
        assert isinstance(TERMINAL_FG_TYPES, frozenset)

    def test_terminal_count_after_phase_163(self):
        """TERMINAL_FG_TYPES must contain exactly 25 entries post-Phase 163 (13 baseline + 12 chalcogen)."""
        assert len(TERMINAL_FG_TYPES) == 25
