"""
Regression tests for peptide guard and ester parent selection fix (Plan 15-05).

Bug 1: Peptides were being flattened to simple amino acids
  - Gly-Gly dipeptide -> "2-aminobutanoic acid" (WRONG, 4C total)
  - Ala-Gly dipeptide -> "2-aminopentanoic acid" (WRONG, 5C total)

Bug 2: Ring-containing acid portions of esters were linearized
  - Methyl benzoate -> "methyl heptanoate" (WRONG, 6 ring C + 1 = 7)
  - Methyl indole-3-carboxylate -> "methyl nonanoate" (WRONG)
"""

import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.rules.amino_acids import (
    count_peptide_bonds,
    is_peptide,
    detect_amino_acid,
    name_amino_acid,
)
from orthonym.rules.esters import (
    acid_fragment_has_ring,
    get_ring_acid_name,
    parse_ester_fragments,
    name_ester,
    find_ester_match,
)


# ============================================================================
# Bug 1: Peptide Guard Tests
# ============================================================================


class TestPeptideBondDetection:
    """Test peptide bond SMARTS detection."""

    def test_dipeptide_gly_gly_has_peptide_bond(self):
        """Gly-Gly dipeptide has 1 peptide bond."""
        mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)O")
        assert count_peptide_bonds(mol) == 1

    def test_tripeptide_gly_gly_gly_has_two_peptide_bonds(self):
        """Gly-Gly-Gly tripeptide has 2 peptide bonds."""
        mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)NCC(=O)O")
        assert count_peptide_bonds(mol) == 2

    def test_simple_glycine_no_peptide_bond(self):
        """Glycine (simple amino acid) has no peptide bonds."""
        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert count_peptide_bonds(mol) == 0

    def test_alanine_no_peptide_bond(self):
        """Alanine (simple amino acid) has no peptide bonds."""
        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        assert count_peptide_bonds(mol) == 0

    def test_asparagine_no_peptide_bond(self):
        """Asparagine has amide side chain (primary -C(=O)NH2), NOT peptide bond."""
        mol = Chem.MolFromSmiles("NC(CC(N)=O)C(=O)O")
        assert count_peptide_bonds(mol) == 0

    def test_glutamine_no_peptide_bond(self):
        """Glutamine has amide side chain (primary -C(=O)NH2), NOT peptide bond."""
        mol = Chem.MolFromSmiles("NC(CCC(N)=O)C(=O)O")
        assert count_peptide_bonds(mol) == 0

    def test_is_peptide_dipeptide(self):
        """Dipeptide correctly identified as peptide."""
        mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)O")
        assert is_peptide(mol) is True

    def test_is_peptide_simple_amino_acid(self):
        """Simple amino acid is NOT a peptide."""
        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert is_peptide(mol) is False


class TestPeptideGuardInNaming:
    """Test that peptides are NOT named as simple amino acids."""

    def test_dipeptide_not_named_as_amino_acid(self):
        """Gly-Gly should NOT be named '2-aminobutanoic acid'."""
        mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)O")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        result = name_amino_acid(mol, canonical)
        assert result is None, f"Dipeptide incorrectly named as amino acid: {result}"

    def test_tripeptide_not_named_as_amino_acid(self):
        """Gly-Gly-Gly should NOT be named '2-aminohexanoic acid'."""
        mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)NCC(=O)O")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        result = name_amino_acid(mol, canonical)
        assert result is None, f"Tripeptide incorrectly named as amino acid: {result}"

    def test_glycine_still_named_correctly(self):
        """Glycine should still return 'glycine'."""
        result = name_compound("NCC(=O)O")
        assert result == "glycine"

    def test_alanine_still_named_correctly(self):
        """Alanine should still return 'alanine'."""
        result = name_compound("CC(N)C(=O)O")
        assert result == "alanine"

    def test_n_methylglycine_still_works(self):
        """N-methylglycine should still be named correctly -- systematically.

        Was ``== "sarcosine"``, stale since: 'sarcosine' has 0 Blue Book
        hits and is in neither retained table; "Systematic
        substitutive names" (the Blue Book),:54251: "When not denoted by a
        retained name, amino acids receive systematic substitutive names...". User
        decision A (2026-09-26) states the class: an amino acid substituted on its
        nitrogen takes the systematic substitutive name. The deny row
        (data/iupac_2013_pin_list.json 'sarcosine') names the replacement; OPSIN 2.9.0
        full-InChIKey round trip checked below, outside the engine.
        """
        from tests.support.rt_assert import assert_full_rt
        result = name_compound("CNCC(=O)O")
        assert result == "(methylamino)acetic acid"
        assert_full_rt(result, "CNCC(=O)O")

    def test_dipeptide_does_not_produce_wrong_carbon_count(self):
        """Ala-Gly dipeptide should NOT produce '2-aminopentanoic acid' (5C)."""
        result = name_compound("CC(N)C(=O)NCC(=O)O")
        assert "aminopentanoic" not in result, f"Dipeptide linearized: {result}"

    def test_tripeptide_does_not_produce_wrong_carbon_count(self):
        """Gly-Gly-Gly should NOT produce '2-aminohexanoic acid' (6C)."""
        result = name_compound("NCC(=O)NCC(=O)NCC(=O)O")
        assert "aminohexanoic" not in result, f"Tripeptide linearized: {result}"

    def test_asparagine_still_works(self):
        """Asparagine (has amide side chain) should still be named correctly."""
        result = name_compound("NC(CC(N)=O)C(=O)O")
        assert result == "asparagine"

    def test_glutamine_still_works(self):
        """Glutamine (has amide side chain) should still be named correctly."""
        result = name_compound("NC(CCC(N)=O)C(=O)O")
        assert result == "glutamine"


# ============================================================================
# Bug 2: Ester Ring Detection Tests
# ============================================================================


class TestAcidFragmentRingDetection:
    """Test that ring atoms in acid fragments are detected."""

    def test_methyl_benzoate_acid_has_ring(self):
        """Acid fragment of methyl benzoate contains benzene ring."""
        mol = Chem.MolFromSmiles("COC(=O)c1ccccc1")
        match = find_ester_match(mol)
        acid_atoms, _ = parse_ester_fragments(mol, match)
        assert acid_fragment_has_ring(mol, acid_atoms) is True

    def test_methyl_acetate_acid_no_ring(self):
        """Acid fragment of methyl acetate has no ring."""
        mol = Chem.MolFromSmiles("CC(=O)OC")
        match = find_ester_match(mol)
        acid_atoms, _ = parse_ester_fragments(mol, match)
        assert acid_fragment_has_ring(mol, acid_atoms) is False

    def test_ethyl_propanoate_acid_no_ring(self):
        """Acid fragment of ethyl propanoate has no ring."""
        mol = Chem.MolFromSmiles("CCC(=O)OCC")
        match = find_ester_match(mol)
        acid_atoms, _ = parse_ester_fragments(mol, match)
        assert acid_fragment_has_ring(mol, acid_atoms) is False


class TestRingAcidNaming:
    """Test naming of ring-containing acid fragments."""

    def test_benzoate_acid_named_benzoic(self):
        """Benzene + COOH acid fragment named 'benzoic'."""
        mol = Chem.MolFromSmiles("COC(=O)c1ccccc1")
        match = find_ester_match(mol)
        acid_atoms, _ = parse_ester_fragments(mol, match)
        name = get_ring_acid_name(mol, acid_atoms)
        assert name == "benzoic"

    def test_cyclopentane_acid_named_carboxylic(self):
        """Cyclopentane + COOH acid fragment named correctly."""
        mol = Chem.MolFromSmiles("COC(=O)C1CCCC1")
        match = find_ester_match(mol)
        acid_atoms, _ = parse_ester_fragments(mol, match)
        name = get_ring_acid_name(mol, acid_atoms)
        assert name == "cyclopentanecarboxylic"

    def test_cyclohexane_acid_named_carboxylic(self):
        """Cyclohexane + COOH acid fragment named correctly."""
        mol = Chem.MolFromSmiles("COC(=O)C1CCCCC1")
        match = find_ester_match(mol)
        acid_atoms, _ = parse_ester_fragments(mol, match)
        name = get_ring_acid_name(mol, acid_atoms)
        assert name == "cyclohexanecarboxylic"


class TestEsterRingNaming:
    """Test complete ester naming with ring-containing acids."""

    def test_methyl_benzoate(self):
        """Methyl benzoate: NOT 'methyl heptanoate'."""
        result = name_compound("COC(=O)c1ccccc1")
        assert result == "methyl benzoate"

    def test_ethyl_benzoate(self):
        """Ethyl benzoate: NOT 'ethyl heptanoate'."""
        result = name_compound("CCOC(=O)c1ccccc1")
        assert result == "ethyl benzoate"

    def test_methyl_cyclopentanecarboxylate(self):
        """Methyl cyclopentanecarboxylate: ring acid named correctly."""
        result = name_compound("COC(=O)C1CCCC1")
        assert result == "methyl cyclopentanecarboxylate"

    def test_methyl_cyclohexanecarboxylate(self):
        """Methyl cyclohexanecarboxylate: ring acid named correctly."""
        result = name_compound("COC(=O)C1CCCCC1")
        assert result == "methyl cyclohexanecarboxylate"

    def test_methyl_indole_carboxylate_not_linearized(self):
        """Methyl indole-3-carboxylate: NOT 'methyl nonanoate'."""
        result = name_compound("COC(=O)c1c[nH]c2ccccc12")
        assert "nonanoate" not in result, f"Fused ring ester linearized: {result}"
        assert "nonan" not in result, f"Fused ring ester linearized: {result}"


class TestSimpleEstersStillWork:
    """Verify simple esters are not broken by ring detection guards."""

    def test_methyl_acetate(self):
        """Methyl acetate should still work."""
        assert name_compound("CC(=O)OC") == "methyl acetate"

    def test_ethyl_propanoate(self):
        """Ethyl propanoate should still work."""
        assert name_compound("CCC(=O)OCC") == "ethyl propanoate"

    def test_methyl_formate(self):
        """Methyl formate should still work."""
        assert name_compound("COC=O") == "methyl formate"

    def test_ethyl_acetate(self):
        """Ethyl acetate should still work."""
        assert name_compound("CC(=O)OCC") == "ethyl acetate"


class TestPolycyclicEsterRouting:
    """Test that polycyclic esters are not routed to simple ring ester naming."""

    def test_cyclohexyl_acetate_still_works(self):
        """Cyclohexyl acetate (simple ring ester) should still work.

         a phase: the assertion now matches this test's own title. It read
        `"acetyloxy" in result or "cyclohexane" in result` — the substitutive
        form — which is neither of the words in `cyclohexyl acetate`, the
        functional-class PIN required by. The routing property the
        class is here to protect (a SIMPLE ring ester must not fall into the
        polycyclic path) is asserted more sharply by naming it exactly.
        """
        assert name_compound("CC(=O)OC1CCCCC1") == "cyclohexyl acetate"

    def test_fused_ring_ester_not_simple_cycloalkane(self):
        """Fused ring ester should NOT be named as simple cycloalkane."""
        # Methyl indole-3-carboxylate - fused system
        result = name_compound("COC(=O)c1c[nH]c2ccccc12")
        assert "cyclohexane" not in result
        assert "cyclopentane" not in result
