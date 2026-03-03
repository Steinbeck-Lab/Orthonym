"""
Integration tests for Phase 86: Early Return Handler Retrofit.

Tests the shared _integrate_universal_prefixes() helper and verifies
that retrofitted handlers (acid halide, lactone, lactam, ether) correctly
discover and name substituents via the universal pipeline.

Reference: IUPAC 2013 Blue Book, P-31.1 (detachable prefixes)
"""

import pytest
from rdkit import Chem


# ============================================================================
# Helper function tests: _integrate_universal_prefixes()
# ============================================================================


class TestIntegrateUniversalPrefixes:
    """Tests for the _integrate_universal_prefixes() helper in composer.py."""

    def _get_helper(self):
        """Import the helper function."""
        from orthonym.assembly.composer import _integrate_universal_prefixes
        return _integrate_universal_prefixes

    def test_chain_parent_with_methyl(self):
        """Chain parent (propane) with a methyl substituent at C-2 produces '2-methyl' prefix."""
        helper = self._get_helper()
        # 2-methylpropane: CC(C)C
        mol = Chem.MolFromSmiles("CC(C)C")
        assert mol is not None
        # Principal chain: atoms 0, 1, 3 (the longest chain C-C-C)
        # Atom 2 is the methyl substituent at C-2
        # Find the chain: index 0, 1, 3
        principal_chain = [0, 1, 3]
        parent_atoms = set(principal_chain)

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
        )
        assert "methyl" in result
        assert "2" in result

    def test_ring_parent_with_chloro(self):
        """Ring parent (cyclohexane) with Cl substituent produces locanted prefix."""
        helper = self._get_helper()
        # Chlorocyclohexane: ClC1CCCCC1
        mol = Chem.MolFromSmiles("ClC1CCCCC1")
        assert mol is not None
        # Ring atoms: 1, 2, 3, 4, 5, 6
        ring_atoms = set([1, 2, 3, 4, 5, 6])
        oriented_ring = [1, 2, 3, 4, 5, 6]

        result = helper(
            mol, ring_atoms,
            parent_type="ring",
            oriented_ring=oriented_ring,
        )
        assert "chloro" in result

    def test_no_substituents_returns_empty(self):
        """Parent with no substituents returns empty string."""
        helper = self._get_helper()
        # Propane: CCC
        mol = Chem.MolFromSmiles("CCC")
        assert mol is not None
        principal_chain = [0, 1, 2]
        parent_atoms = set(principal_chain)

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
        )
        assert result == ""

    def test_exclude_atoms_skips_excluded(self):
        """Excluded atoms are not discovered as substituents."""
        helper = self._get_helper()
        # Acetyl chloride: CC(=O)Cl
        mol = Chem.MolFromSmiles("CC(=O)Cl")
        assert mol is not None
        # Chain: atoms 0, 1 (C-C chain)
        # Atom 2 = O (=O), Atom 3 = Cl
        # Both are consumed by acid halide -- exclude them
        principal_chain = [0, 1]
        parent_atoms = set(principal_chain)
        exclude_atoms = {2, 3}  # O and Cl

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
            exclude_atoms=exclude_atoms,
        )
        assert result == ""

    def test_multiple_substituents_alphabetical(self):
        """Multiple different substituents are alphabetically sorted."""
        helper = self._get_helper()
        # 2-chloro-3-methylbutane: CC(Cl)C(C)C -- but let's use a cleaner case
        # 1-bromo-2-chloroethane: BrCCCl
        mol = Chem.MolFromSmiles("BrCCCl")
        assert mol is not None
        # Chain: atoms 1, 2
        principal_chain = [1, 2]
        parent_atoms = set(principal_chain)

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
        )
        # Should contain both bromo and chloro, with bromo before chloro alphabetically
        assert "bromo" in result
        assert "chloro" in result
        bromo_pos = result.index("bromo")
        chloro_pos = result.index("chloro")
        assert bromo_pos < chloro_pos, f"bromo should come before chloro alphabetically, got: {result}"


# ============================================================================
# Acid halide handler tests (Task 2)
# ============================================================================


class TestAcidHalideRetrofit:
    """Tests for acid halide handler with universal pipeline."""

    def test_3_methylbutanoyl_chloride(self):
        """3-methylbutanoyl chloride: CC(C)CC(=O)Cl"""
        from orthonym.namer import name_compound
        result = name_compound("CC(C)CC(=O)Cl")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "butanoyl" in result.lower(), f"Expected 'butanoyl' in '{result}'"
        assert "chloride" in result.lower(), f"Expected 'chloride' in '{result}'"

    def test_2_chloroacetyl_bromide(self):
        """2-chloroacetyl bromide: ClCC(=O)Br -- halogen substituent on chain."""
        from orthonym.namer import name_compound
        result = name_compound("ClCC(=O)Br")
        assert result is not None
        assert "chloro" in result.lower(), f"Expected 'chloro' in '{result}'"
        assert "bromide" in result.lower(), f"Expected 'bromide' in '{result}'"


# ============================================================================
# Lactone handler tests (Task 2)
# ============================================================================


class TestLactoneRetrofit:
    """Tests for lactone handler with universal pipeline."""

    def test_3_methyloxolan_2_one(self):
        """3-methyloxolan-2-one (gamma-butyrolactone with methyl): CC1CCOC1=O"""
        from orthonym.namer import name_compound
        result = name_compound("CC1CCOC1=O")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "oxolan" in result.lower(), f"Expected 'oxolan' in '{result}'"

    def test_5_ethyloxan_2_one(self):
        """5-ethyloxan-2-one (delta-valerolactone with ethyl): CCC1CCOC(=O)C1"""
        from orthonym.namer import name_compound
        result = name_compound("CCC1CCOC(=O)C1")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "oxan" in result.lower(), f"Expected 'oxan' in '{result}'"


# ============================================================================
# Lactam handler tests (Task 2)
# ============================================================================


class TestLactamRetrofit:
    """Tests for lactam handler with universal pipeline."""

    def test_1_methylpyrrolidin_2_one(self):
        """1-methylpyrrolidin-2-one (N-methylpyrrolidone): CN1CCCC1=O"""
        from orthonym.namer import name_compound
        result = name_compound("CN1CCCC1=O")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "pyrrolidin" in result.lower(), f"Expected 'pyrrolidin' in '{result}'"

    def test_3_ethylpiperidin_2_one(self):
        """3-ethylpiperidin-2-one: CCC1CCCNC1=O"""
        from orthonym.namer import name_compound
        result = name_compound("CCC1CCCNC1=O")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "piperidin" in result.lower(), f"Expected 'piperidin' in '{result}'"


# ============================================================================
# Ether bare oxy fix tests (Task 2)
# ============================================================================


class TestEtherOxyFix:
    """Tests for ether naming -- no bare 'oxy' prefix."""

    def test_simple_methoxy(self):
        """Methoxy: COc1ccccc1 -> methoxybenzene (anisole retained)."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        result = _alcohol_to_alkoxy("methanol")
        assert result == "methoxy"

    def test_simple_ethoxy(self):
        """Ethoxy: ethanol -> ethoxy."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        result = _alcohol_to_alkoxy("ethanol")
        assert result == "ethoxy"

    def test_no_bare_oxy(self):
        """_alcohol_to_alkoxy never returns bare 'oxy' for organic fragments."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        # Test complex alcohol names that might fail
        test_names = ["cyclopentanol", "cyclohexanol", "phenol"]
        for name in test_names:
            result = _alcohol_to_alkoxy(name)
            if result is not None:
                assert result != "oxy", f"Bare 'oxy' returned for '{name}'"
                assert result.endswith("oxy"), f"Expected alkoxy form for '{name}', got '{result}'"


# ============================================================================
# Ester handler tests (Plan 02, Task 1)
# ============================================================================


class TestEsterRetrofit:
    """Tests for ester handler with universal pipeline (acid-side substituents)."""

    def test_methyl_3_methylbutanoate(self):
        """Branched acid chain: methyl 3-methylbutanoate -- CC(C)CC(=O)OC."""
        from orthonym.namer import name_compound
        result = name_compound("CC(C)CC(=O)OC")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' prefix in '{result}'"
        assert "butanoate" in result.lower(), f"Expected 'butanoate' in '{result}'"
        # The 3-methyl substituent on the acid chain must be present
        assert "3-methyl" in result.lower(), f"Expected '3-methyl' locanted prefix in '{result}'"

    def test_methyl_2_chloropropanoate(self):
        """Halogen substituent on acid chain: methyl 2-chloropropanoate."""
        from orthonym.namer import name_compound
        result = name_compound("CC(Cl)C(=O)OC")
        assert result is not None
        assert "chloro" in result.lower(), f"Expected 'chloro' in '{result}'"
        assert "propanoate" in result.lower(), f"Expected 'propanoate' in '{result}'"

    def test_simple_ethyl_acetate_no_regression(self):
        """Simple ester without substituents: ethyl acetate still works."""
        from orthonym.namer import name_compound
        result = name_compound("CC(=O)OCC")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "acetate" in result.lower(), f"Expected 'acetate' in '{result}'"

    def test_methyl_propanoate_no_regression(self):
        """Simple 3-carbon acid ester: methyl propanoate still works."""
        from orthonym.namer import name_compound
        result = name_compound("CCC(=O)OC")
        assert result is not None
        assert "propanoate" in result.lower(), f"Expected 'propanoate' in '{result}'"

    def test_methyl_2_ethylbutanoate(self):
        """Multiple substituents on acid chain: methyl 2-ethylbutanoate."""
        from orthonym.namer import name_compound
        result = name_compound("CCC(CC)C(=O)OC")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "butanoate" in result.lower(), f"Expected 'butanoate' in '{result}'"
