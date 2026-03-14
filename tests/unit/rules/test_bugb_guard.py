"""BUG-B guard verification tests.

Tests that the _BRANCH_HANDLED_FGS set in polyfunctional.py only contains
FG types whose prefixes are reliably emitted by the substituent naming path
when the FG is on a small (1-3C) branch.

IUPAC P-59.1(a): all non-principal functional groups must appear as prefixes.
The BUG-B guard filters FGs from the polyfunctional prefix loop when they are
on small substituent branches, trusting that substituent naming will emit the
FG prefix (e.g., "hydroxymethyl" for -CH2OH). If substituent naming does NOT
emit the prefix, the FG silently drops from the name.

These tests verify:
1. Each FG type in _BRANCH_HANDLED_FGS produces correct output via substituent
   naming (no FG prefix loss)
2. No double-naming occurs (FG prefix appears exactly once, not twice)
3. FG types NOT in the set still produce correct polyfunctional prefixes
"""

import pytest
from orthonym.namer import name_compound


class TestHalogensOnSmallBranches:
    """Halogens on 1-3C branches: substituent naming always produces halomethyl etc."""

    def test_fluoro_on_1c_branch(self):
        """Fluoro on -CH2F branch should appear as '(fluoromethyl)' via substituent naming."""
        result = name_compound("FCC(CCC)C(=O)O")
        assert "fluoro" in result.lower(), (
            f"Fluoro prefix missing from name: {result}"
        )

    def test_chloro_on_1c_branch(self):
        """Chloro on -CH2Cl branch should appear in name."""
        result = name_compound("ClCC(CCC)C(=O)O")
        assert "chloro" in result.lower(), (
            f"Chloro prefix missing from name: {result}"
        )

    def test_bromo_on_1c_branch(self):
        """Bromo on -CH2Br branch should appear in name."""
        result = name_compound("BrCC(CCCC)C(=O)O")
        assert "bromo" in result.lower(), (
            f"Bromo prefix missing from name: {result}"
        )

    def test_iodo_on_1c_branch(self):
        """Iodo on -CH2I branch should appear in name."""
        result = name_compound("ICC(CCC)C(=O)O")
        assert "iodo" in result.lower(), (
            f"Iodo prefix missing from name: {result}"
        )

    def test_no_double_naming_halogen(self):
        """Halogen on small branch should appear once, not as both substituent AND standalone prefix."""
        result = name_compound("ClCC(CCC)C(=O)O")
        # Should be "2-(chloromethyl)pentanoic acid", not "2-chloro-2-(chloromethyl)pentanoic acid"
        chloro_count = result.lower().count("chloro")
        assert chloro_count == 1, (
            f"Double-naming: 'chloro' appears {chloro_count} times in: {result}"
        )


class TestAlcoholsOnSmallBranches:
    """Alcohols on 1-3C branches: verify hydroxy prefix is present."""

    def test_primary_alcohol_on_1c_branch(self):
        """Primary -OH on -CH2OH branch should produce 'hydroxymethyl' or 'hydroxy' prefix."""
        result = name_compound("OCC(CCC)C(=O)O")
        assert "hydroxy" in result.lower(), (
            f"Hydroxy prefix missing from name: {result}"
        )

    def test_secondary_alcohol_on_2c_branch(self):
        """Secondary -OH on -CHOH-CH3 branch should produce hydroxy prefix."""
        result = name_compound("OC(C)C(CCC)C(=O)O")
        assert "hydroxy" in result.lower(), (
            f"Hydroxy prefix missing from name: {result}"
        )

    def test_no_double_naming_alcohol(self):
        """Alcohol on small branch: hydroxy should not appear redundantly."""
        result = name_compound("OCC(CCC)C(=O)O")
        # "2-(hydroxymethyl)pentanoic acid" -- hydroxy appears once inside substituent name
        hydroxy_count = result.lower().count("hydroxy")
        assert hydroxy_count == 1, (
            f"Double-naming: 'hydroxy' appears {hydroxy_count} times in: {result}"
        )


class TestAminesOnSmallBranches:
    """Amines on 1-3C branches: verify amino prefix is present."""

    def test_primary_amine_on_1c_branch(self):
        """Primary -NH2 on -CH2NH2 branch should produce 'aminomethyl' or 'amino' prefix."""
        result = name_compound("NCC(CCC)C(=O)O")
        assert "amino" in result.lower(), (
            f"Amino prefix missing from name: {result}"
        )

    def test_no_double_naming_amine(self):
        """Amine on small branch: amino should not appear redundantly."""
        result = name_compound("NCC(CCC)C(=O)O")
        amino_count = result.lower().count("amino")
        assert amino_count == 1, (
            f"Double-naming: 'amino' appears {amino_count} times in: {result}"
        )


class TestPolyfunctionalCompleteness:
    """Polyfunctional molecules: all non-principal FGs must appear as prefixes."""

    def test_hydroxy_plus_oxo_plus_acid(self):
        """Molecule with hydroxyl + aldehyde + carboxylic acid."""
        result = name_compound("OCC(CC(=O)O)CC=O")
        assert "hydroxy" in result.lower(), (
            f"Hydroxy prefix missing in polyfunctional name: {result}"
        )
        assert "oxo" in result.lower(), (
            f"Oxo prefix missing in polyfunctional name: {result}"
        )

    def test_amino_plus_acid(self):
        """Molecule with amine + carboxylic acid on chain."""
        result = name_compound("NC(CC)C(=O)O")
        assert "amino" in result.lower(), (
            f"Amino prefix missing: {result}"
        )

    def test_hydroxy_on_ring_plus_acid(self):
        """Cyclohexane with hydroxyl + carboxylic acid."""
        result = name_compound("OC1CCCCC1C(=O)O")
        assert "hydroxy" in result.lower(), (
            f"Hydroxy prefix missing on ring compound: {result}"
        )

    def test_multi_amino_diacid(self):
        """Multiple amino groups + diacid."""
        result = name_compound("NCC(CC(N)C(=O)O)C(=O)O")
        assert "amino" in result.lower(), (
            f"Amino prefix missing in multi-amino compound: {result}"
        )

    def test_hydroxy_and_aldehyde_on_chain(self):
        """Hydroxyl + aldehyde on chain with acid as principal group."""
        result = name_compound("OC(CC=O)CC(=O)O")
        assert "hydroxy" in result.lower(), (
            f"Hydroxy prefix missing: {result}"
        )
        assert "oxo" in result.lower(), (
            f"Oxo prefix missing: {result}"
        )
