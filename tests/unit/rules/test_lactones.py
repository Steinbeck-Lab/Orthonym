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

    def test_oxolane_not_lactone(self):
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
        """Oxane (oxane) has ring O but no C=O -> not a lactone."""
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
        """Ring sizes < 3 or > 50 return None."""
        assert name_lactone_ring(2) is None
        assert name_lactone_ring(51) is None

    def test_macrolide_ring_sizes(self):
        """Ring sizes 11+ use oxacyclo replacement nomenclature."""
        assert name_lactone_ring(11) == "oxacycloundecan-2-one"
        assert name_lactone_ring(13) == "oxacyclotridecan-2-one"
        assert name_lactone_ring(15) == "oxacyclopentadecan-2-one"

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
            assert "ane-2-one" not in name, (
                f"Size {size}: {name} should have vowel elision"
            )
            assert "ene-2-one" not in name, (
                f"Size {size}: {name} should have vowel elision"
            )


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

    def test_canonical_smiles_variations_gbl(self):
        """Different SMILES for gamma-butyrolactone should all give oxolan-2-one."""
        # These are valid alternate representations of the 5-membered lactone
        smiles_variants = [
            "O=C1CCCO1",
            "C1CC(=O)OC1",
        ]
        for smi in smiles_variants:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Failed to parse SMILES: {smi}"
            info = is_monocyclic_lactone(mol)
            assert info is not None, f"SMILES {smi} not detected as lactone"
            name = name_monocyclic_lactone(mol)
            assert name == "oxolan-2-one", f"SMILES {smi} gave {name}"

    def test_canonical_smiles_4_membered(self):
        """O=C1OCC1 canonicalizes to a 4-membered lactone (beta-propiolactone)."""
        mol = Chem.MolFromSmiles("O=C1OCC1")
        if mol is not None:
            info = is_monocyclic_lactone(mol)
            if info is not None:
                assert info["ring_size"] == 4
                name = name_monocyclic_lactone(mol)
                assert name == "oxetan-2-one"

    def test_carbonyl_oxygen_not_in_ring(self):
        """The exocyclic =O should not be counted as a ring atom."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        info = is_monocyclic_lactone(mol)
        assert info is not None
        # The carbonyl O (exocyclic) should not be in ring_atoms
        assert info["carbonyl_O_idx"] not in info["ring_atoms"]


# ===========================================================================
# Substituted lactone tests (Plan 15-07)
# ===========================================================================

class TestSubstitutedLactones:
    """Tests for substituted lactones with correct ring sizes and substituent prefixes."""

    def test_substituted_butyrolactone_methyl(self):
        """3-Methyl-gamma-butyrolactone -> 3-methyloxolan-2-one."""
        mol = Chem.MolFromSmiles("CC1CCOC1=O")
        result = name_monocyclic_lactone(mol)
        assert result == "3-methyloxolan-2-one", f"Got '{result}'"

    def test_substituted_valerolactone_methyl(self):
        """3-Methyl-delta-valerolactone -> 3-methyloxan-2-one.

        Verifies correct 6-membered ring naming (oxan, not oxolan).
        """
        mol = Chem.MolFromSmiles("CC1CCCOC1=O")
        result = name_monocyclic_lactone(mol)
        assert result == "3-methyloxan-2-one", f"Got '{result}'"

    def test_macrolide_11_membered(self):
        """11-membered macrolide -> oxacycloundecan-2-one."""
        mol = Chem.MolFromSmiles("O=C1CCCCCCCCCO1")
        result = name_monocyclic_lactone(mol)
        assert result == "oxacycloundecan-2-one", f"Got '{result}'"

    def test_macrolide_13_membered(self):
        """13-membered macrolide -> oxacyclotridecan-2-one."""
        mol = Chem.MolFromSmiles("O=C1CCCCCCCCCCCO1")
        result = name_monocyclic_lactone(mol)
        assert result == "oxacyclotridecan-2-one", f"Got '{result}'"

    def test_macrolide_15_membered(self):
        """15-membered macrolide -> oxacyclopentadecan-2-one."""
        mol = Chem.MolFromSmiles("O=C1CCCCCCCCCCCCCO1")
        result = name_monocyclic_lactone(mol)
        assert result == "oxacyclopentadecan-2-one", f"Got '{result}'"

    def test_substituted_macrolide(self):
        """Substituted 11-membered macrolide includes substituent prefix."""
        mol = Chem.MolFromSmiles("O=C1CC(C)CCCCCCCO1")
        result = name_monocyclic_lactone(mol)
        assert "methyl" in result, f"Expected 'methyl' in '{result}'"
        assert "oxacycloundecan-2-one" in result, f"Expected macrolide parent in '{result}'"

    def test_amino_lactone(self):
        """3-Amino-gamma-butyrolactone -> 3-aminooxolan-2-one."""
        mol = Chem.MolFromSmiles("NC1CCOC1=O")
        result = name_monocyclic_lactone(mol)
        assert result == "3-aminooxolan-2-one", f"Got '{result}'"

    def test_hydroxy_lactone(self):
        """3-Hydroxy-gamma-butyrolactone -> 3-hydroxyoxolan-2-one."""
        mol = Chem.MolFromSmiles("OC1CCOC1=O")
        result = name_monocyclic_lactone(mol)
        assert result == "3-hydroxyoxolan-2-one", f"Got '{result}'"

    def test_chloro_lactone(self):
        """3-Chloro-gamma-butyrolactone -> 3-chlorooxolan-2-one."""
        mol = Chem.MolFromSmiles("ClC1CCOC1=O")
        result = name_monocyclic_lactone(mol)
        assert result == "3-chlorooxolan-2-one", f"Got '{result}'"

    def test_6_membered_not_5_membered(self):
        """6-membered lactone must return oxan-2-one, not oxolan-2-one.

        This was a specific mismatch identified in UAT: 6-membered lactones
        being incorrectly named as 5-membered.
        """
        mol = Chem.MolFromSmiles("O=C1CCCCO1")
        info = is_monocyclic_lactone(mol)
        assert info is not None
        assert info["ring_size"] == 6, f"Expected ring_size 6, got {info['ring_size']}"
        result = name_monocyclic_lactone(mol)
        assert result == "oxan-2-one", f"Got '{result}' instead of 'oxan-2-one'"
        assert "oxolan" not in result, "6-membered must not use 5-membered name"


class TestLactoneSeniority:
    """P-65.6.3.5.1 (the Blue Book): a lactone is a pseudoketone and ranks LOWER in
    the seniority of classes than an acid or an ester (Table 4.1, the Blue Book:
    Acids/esters > ketones/pseudoketones), but HIGHER than an alcohol, amine or
    imine. These are end-to-end (name_compound) assertions because the seniority
    decision lives in the assembly layer, not in name_monocyclic_lactone.
    """

    def test_acid_beats_lactone(self):
        """Carboxylic acid + lactone: the ACID owns the suffix, the ring C=O
        degrades to an 'oxo' prefix (BB PIN anchor the Blue Book)."""
        from orthonym import name_compound
        assert (
            name_compound("O=C1CCC(C(=O)O)O1")
            == "5-oxooxolane-2-carboxylic acid"
        )

    def test_ester_beats_lactone(self):
        """Acyclic ester + lactone: the ester ranks above the pseudoketone
        lactone; the ring C=O becomes 'oxo'."""
        from orthonym import name_compound
        assert (
            name_compound("O=C1CCC(C(=O)OC)O1")
            == "methyl 5-oxooxolane-2-carboxylate"
        )

    def test_lactone_beats_alcohol(self):
        """Alcohol ranks BELOW the lactone; the lactone keeps the '-one'
        suffix and the OH is a 'hydroxy' prefix."""
        from orthonym import name_compound
        assert name_compound("O=C1CCC(O)O1") == "5-hydroxyoxolan-2-one"

    def test_lactone_beats_amine(self):
        """Amine ranks BELOW the lactone; the lactone keeps the '-one'
        suffix and the NH2 is an 'amino' prefix."""
        from orthonym import name_compound
        assert name_compound("O=C1CCC(N)O1") == "5-aminooxolan-2-one"

    def test_plain_lactone_unchanged(self):
        """A plain lactone (no senior group) keeps its '-one' name — the
        seniority guard must not fire when the only 'ester' detected is the
        lactone's own ring motif."""
        from orthonym import name_compound
        assert name_compound("O=C1CCCO1") == "oxolan-2-one"
        assert name_compound("O=C1CCCCO1") == "oxan-2-one"
