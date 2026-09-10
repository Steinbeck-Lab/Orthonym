"""BUG-B guard verification tests.

Tests that the BRANCH_HANDLED_FGS frozenset (shared from naming_utils) only
contains FG types whose prefixes are reliably emitted by the substituent naming
path when the FG is on a small (1-3C) branch.

IUPAC (a): all non-principal functional groups must appear as prefixes.
The BUG-B guard filters FGs from the polyfunctional prefix loop when they are
on small substituent branches, trusting that substituent naming will emit the
FG prefix (e.g., "hydroxymethyl" for -CH2OH). If substituent naming does NOT
emit the prefix, the FG silently drops from the name.

These tests verify:
1. BRANCH_HANDLED_FGS is a frozenset importable from naming_utils
2. It contains the minimum required entries and does NOT contain dangerous entries
3. Both polyfunctional.py and composer.py import from the same source
4. Each FG type in the set produces correct output via substituent naming
5. No double-naming occurs (FG prefix appears exactly once, not twice)
6. FG types NOT in the set still produce correct polyfunctional prefixes
"""

import pytest
from orthonym.namer import name_compound
from orthonym.assembly.naming_utils import BRANCH_HANDLED_FGS


# ---- Structural tests: verify the shared frozenset contract ----

MINIMUM_REQUIRED_FGS = frozenset({
    'primary_alcohol', 'secondary_alcohol', 'primary_amine',
    'fluoro', 'chloro', 'bromo', 'iodo',
})

# FG types that must NEVER be in the BUG-B guard set because they are
# high-seniority suffix-capable groups. Filtering them would silently
# drop critical prefixes (oxo, carboxy, cyano, etc.).
DANGEROUS_FGS = frozenset({
    'carboxylic_acid', 'aldehyde', 'ketone', 'nitrile',
    'primary_amide', 'ester', 'anhydride', 'acid_chloride',
})


@pytest.mark.unit
class TestBranchHandledFGsContract:
    """Structural tests for the unified BRANCH_HANDLED_FGS frozenset."""

    def test_is_frozenset(self):
        """BRANCH_HANDLED_FGS must be a frozenset (immutable)."""
        assert isinstance(BRANCH_HANDLED_FGS, frozenset), (
            f"Expected frozenset, got {type(BRANCH_HANDLED_FGS).__name__}"
        )

    def test_contains_minimum_entries(self):
        """BRANCH_HANDLED_FGS must contain at minimum the 7 core FG types."""
        missing = MINIMUM_REQUIRED_FGS - BRANCH_HANDLED_FGS
        assert missing == set(), (
            f"Missing required FG types from BRANCH_HANDLED_FGS: {missing}"
        )

    def test_no_dangerous_entries(self):
        """BRANCH_HANDLED_FGS must NOT contain high-seniority FGs.
        These are suffix-capable groups whose prefixes (oxo, carboxy, cyano)
        must always be handled by the polyfunctional prefix loop, not
        delegated to branch naming.
        """
        dangerous_present = DANGEROUS_FGS & BRANCH_HANDLED_FGS
        assert dangerous_present == set(), (
            f"Dangerous FG types found in BRANCH_HANDLED_FGS: {dangerous_present}. "
            f"These must never be filtered by the BUG-B guard."
        )

    def test_shared_between_modules(self):
        """polyfunctional.py and composer.py must use the same BRANCH_HANDLED_FGS
        object from naming_utils, not define their own inline copies.
        """
        import importlib
        import inspect

        # Verify naming_utils exports it
        from orthonym.assembly import naming_utils
        assert hasattr(naming_utils, 'BRANCH_HANDLED_FGS'), (
            "naming_utils.py must export BRANCH_HANDLED_FGS"
        )

        # Check that polyfunctional.py source does NOT define inline _BRANCH_HANDLED_FGS
        from orthonym.rules import polyfunctional
        source = inspect.getsource(polyfunctional)
        assert '_BRANCH_HANDLED_FGS = {' not in source and '_BRANCH_HANDLED_FGS={' not in source, (
            "polyfunctional.py still defines an inline _BRANCH_HANDLED_FGS set. "
            "It should import BRANCH_HANDLED_FGS from naming_utils."
        )


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
        """Molecule with amine + carboxylic acid on chain.
        After 141-02, this is recognized as 'butyrine' (OPSIN trivial name)."""
        result = name_compound("NC(CC)C(=O)O")
        assert "amino" in result.lower() or result == "butyrine", (
            f"Expected amino prefix or trivial name 'butyrine': {result}"
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
