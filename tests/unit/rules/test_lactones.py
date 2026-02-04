"""
Tests for monocyclic lactone detection and naming.

Monocyclic lactones are cyclic esters that use heterocyclic replacement
nomenclature with an -one suffix:
- beta-propiolactone (4-membered) -> oxetan-2-one
- gamma-butyrolactone (5-membered) -> oxolan-2-one
- delta-valerolactone (6-membered) -> oxan-2-one
- epsilon-caprolactone (7-membered) -> oxepan-2-one

IUPAC Rule: The ring oxygen is position 1, the carbonyl carbon is position 2.
The parent heterocyclic name has its terminal 'e' elided before '-one'.
"""

import pytest
from rdkit import Chem

from orthonym.rules.lactones import (
    is_monocyclic_lactone,
    name_monocyclic_lactone,
    name_lactone_ring,
)


# ===========================================================================
# Detection tests: is_monocyclic_lactone
# ===========================================================================

class TestIsMonocyclicLactone:
    """Test lactone detection for various ring sizes and non-lactone controls."""

    def test_gamma_butyrolactone_detected(self):
        """gamma-butyrolactone (5-membered) should be detected as a lactone."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert result["ring_size"] == 5

    def test_delta_valerolactone_detected(self):
        """delta-valerolactone (6-membered) should be detected as a lactone."""
        mol = Chem.MolFromSmiles("O=C1CCCCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert result["ring_size"] == 6

    def test_beta_propiolactone_detected(self):
        """beta-propiolactone (4-membered) should be detected as a lactone."""
        mol = Chem.MolFromSmiles("O=C1CCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert result["ring_size"] == 4

    def test_epsilon_caprolactone_detected(self):
        """epsilon-caprolactone (7-membered) should be detected as a lactone."""
        mol = Chem.MolFromSmiles("O=C1CCCCCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert result["ring_size"] == 7

    def test_returns_carbonyl_idx(self):
        """Detection result should include the carbonyl carbon index."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert "carbonyl_idx" in result

    def test_returns_ester_o_idx(self):
        """Detection result should include the ester (ring) oxygen index."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert "ester_O_idx" in result

    def test_returns_ring_atoms(self):
        """Detection result should include ring atom indices."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        result = is_monocyclic_lactone(mol)
        assert result is not None
        assert "ring_atoms" in result
        assert len(result["ring_atoms"]) == 5

    # ---- Negative tests: non-lactones ----

    def test_tetrahydrofuran_not_lactone(self):
        """THF has no carbonyl -> not a lactone."""
        mol = Chem.MolFromSmiles("C1CCOC1")
        result = is_monocyclic_lactone(mol)
        assert result is None

    def test_oxetane_not_lactone(self):
        """Oxetane (no C=O) should not be detected."""
        mol = Chem.MolFromSmiles("C1COC1")
        result = is_monocyclic_lactone(mol)
        assert result is None

    def test_oxirane_not_lactone(self):
        """Oxirane (epoxide, 3-membered, no C=O) should not be detected."""
        mol = Chem.MolFromSmiles("C1CO1")
        result = is_monocyclic_lactone(mol)
        assert result is None

    def test_acyclic_ester_not_lactone(self):
        """Methyl acetate (acyclic ester) should not be detected."""
        mol = Chem.MolFromSmiles("CC(=O)OC")
        result = is_monocyclic_lactone(mol)
        assert result is None

    def test_none_mol_returns_none(self):
        """None molecule input should return None."""
        result = is_monocyclic_lactone(None)
        assert result is None

    def test_cyclohexanone_not_lactone(self):
        """Cyclohexanone has C=O but no ring oxygen -> not a lactone."""
        mol = Chem.MolFromSmiles("O=C1CCCCC1")
        result = is_monocyclic_lactone(mol)
        assert result is None

    def test_pyran_not_lactone(self):
        """Oxane (tetrahydropyran) has ring O but no C=O -> not a lactone."""
        mol = Chem.MolFromSmiles("C1CCOCC1")
        result = is_monocyclic_lactone(mol)
        assert result is None


# ===========================================================================
# Naming helper: name_lactone_ring
# ===========================================================================

class TestNameLactoneRing:
    """Test the ring-size to name mapping with vowel elision and -2-one suffix."""

    def test_4_membered(self):
        assert name_lactone_ring(4) == "oxetan-2-one"

    def test_5_membered(self):
        assert name_lactone_ring(5) == "oxolan-2-one"

    def test_6_membered(self):
        assert name_lactone_ring(6) == "oxan-2-one"

    def test_7_membered(self):
        assert name_lactone_ring(7) == "oxepan-2-one"

    def test_unsupported_ring_size_returns_none(self):
        """Ring sizes outside 4-7 return None (3-membered and 8+ uncommon)."""
        assert name_lactone_ring(2) is None
        assert name_lactone_ring(11) is None

    def test_3_membered_ring(self):
        """3-membered lactone (oxiran-2-one) should work if requested."""
        result = name_lactone_ring(3)
        # 3-membered is oxirane -> oxiran-2-one
        assert result == "oxiran-2-one"

    def test_vowel_elision_applied(self):
        """All names should show vowel elision: terminal 'e' removed before '-one'."""
        # oxetane -> oxetan, oxolane -> oxolan, oxane -> oxan, oxepane -> oxepan
        for size in [4, 5, 6, 7]:
            name = name_lactone_ring(size)
            assert name is not None
            # Should NOT contain the unelided form
            assert "ane-2-one" not in name, f"Size {size}: {name} should have vowel elision"
            assert "ene-2-one" not in name, f"Size {size}: {name} should have vowel elision"


# ===========================================================================
# Full naming: name_monocyclic_lactone
# ===========================================================================

class TestNameMonocyclicLactone:
    """Test end-to-end lactone naming from SMILES."""

    def test_gamma_butyrolactone(self):
        """O=C1CCCO1 -> oxolan-2-one"""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        assert name_monocyclic_lactone(mol) == "oxolan-2-one"

    def test_delta_valerolactone(self):
        """O=C1CCCCO1 -> oxan-2-one"""
        mol = Chem.MolFromSmiles("O=C1CCCCO1")
        assert name_monocyclic_lactone(mol) == "oxan-2-one"

    def test_beta_propiolactone(self):
        """O=C1CCO1 -> oxetan-2-one"""
        mol = Chem.MolFromSmiles("O=C1CCO1")
        assert name_monocyclic_lactone(mol) == "oxetan-2-one"

    def test_epsilon_caprolactone(self):
        """O=C1CCCCCO1 -> oxepan-2-one"""
        mol = Chem.MolFromSmiles("O=C1CCCCCO1")
        assert name_monocyclic_lactone(mol) == "oxepan-2-one"

    def test_non_lactone_returns_none(self):
        """THF (no carbonyl) should return None."""
        mol = Chem.MolFromSmiles("C1CCOC1")
        assert name_monocyclic_lactone(mol) is None

    def test_none_mol_returns_none(self):
        """None molecule should return None."""
        assert name_monocyclic_lactone(None) is None

    def test_acyclic_ester_returns_none(self):
        """Acyclic ester should return None."""
        mol = Chem.MolFromSmiles("CC(=O)OC")
        assert name_monocyclic_lactone(mol) is None


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestLactoneEdgeCases:
    """Edge cases and boundary conditions for lactone naming."""

    def test_canonical_smiles_variations(self):
        """Different SMILES representations of gamma-butyrolactone should work."""
        smiles_variants = [
            "O=C1CCCO1",
            "C1CC(=O)OC1",
            "O=C1OCC1",  # reversed
        ]
        for smi in smiles_variants:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Should either be detected or not, but if detected, name correctly
                info = is_monocyclic_lactone(mol)
                if info is not None:
                    name = name_monocyclic_lactone(mol)
                    # All valid 5-membered lactone SMILES should give oxolan-2-one
                    assert name == "oxolan-2-one", f"SMILES {smi} gave {name}"

    def test_carbonyl_oxygen_not_in_ring(self):
        """The exocyclic =O should not be counted as a ring atom."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        info = is_monocyclic_lactone(mol)
        assert info is not None
        # The carbonyl O (exocyclic) should not be in ring_atoms
        assert info["carbonyl_O_idx"] not in info["ring_atoms"]
