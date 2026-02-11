"""
TDD tests for bond cleavage detection in the decomposition engine.

Tests that find_cleavable_bonds() correctly identifies ester, amide,
and glycosidic bonds while excluding lactones, lactams, carbamates,
and ureas.
"""

import pytest
from rdkit import Chem

from orthonym.decomposition.bond_cleavage import find_cleavable_bonds


# ---------------------------------------------------------------------------
# Ester bond detection
# ---------------------------------------------------------------------------

class TestEsterBondDetection:
    """Tests for ester C(=O)-O bond detection."""

    def test_ethyl_acetate_one_ester_bond(self):
        """CC(=O)OCC (ethyl acetate) should have 1 ester bond."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 1

    def test_diester_two_bonds(self):
        """CC(=O)OCC(=O)OC has 2 ester bonds."""
        mol = Chem.MolFromSmiles("CC(=O)OCC(=O)OC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 2

    def test_methyl_hexanoate_one_ester_bond(self):
        """CCCCCC(=O)OC (methyl hexanoate) should have 1 ester bond."""
        mol = Chem.MolFromSmiles("CCCCCC(=O)OC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 1

    def test_two_esters_in_chain(self):
        """CC(=O)OCCC(=O)OCC has 2 ester bonds."""
        mol = Chem.MolFromSmiles("CC(=O)OCCC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 2

    def test_ester_bond_has_acid_atom(self):
        """Ester bond info should include acid_atom (carbonyl C)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 1
        bond = ester_bonds[0]
        assert "acid_atom" in bond
        # The acid_atom should be the carbonyl carbon
        acid = mol.GetAtomWithIdx(bond["acid_atom"])
        assert acid.GetSymbol() == "C"

    def test_ester_bond_has_alkyl_atom(self):
        """Ester bond info should include alkyl_atom."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 1
        bond = ester_bonds[0]
        assert "alkyl_atom" in bond


# ---------------------------------------------------------------------------
# Lactone exclusion
# ---------------------------------------------------------------------------

class TestLactoneExclusion:
    """Tests that cyclic esters (lactones) are excluded."""

    def test_gamma_butyrolactone_no_bonds(self):
        """O=C1CCCO1 (gamma-butyrolactone) should have 0 cleavable bonds."""
        mol = Chem.MolFromSmiles("O=C1CCCO1")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 0

    def test_delta_valerolactone_no_bonds(self):
        """O=C1CCCCO1 (delta-valerolactone) should have 0 cleavable bonds."""
        mol = Chem.MolFromSmiles("O=C1CCCCO1")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 0


# ---------------------------------------------------------------------------
# Amide bond detection
# ---------------------------------------------------------------------------

class TestAmideBondDetection:
    """Tests for amide C(=O)-N bond detection."""

    def test_n_methylacetamide_one_amide_bond(self):
        """CC(=O)NC (N-methylacetamide) should have 1 amide bond."""
        mol = Chem.MolFromSmiles("CC(=O)NC")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) == 1

    def test_nn_dimethylacetamide_one_amide_bond(self):
        """CC(=O)N(C)C (N,N-dimethylacetamide) should have 1 amide bond."""
        mol = Chem.MolFromSmiles("CC(=O)N(C)C")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) == 1

    def test_acetamide_primary_one_amide_bond(self):
        """CC(=O)N (acetamide) should have 1 amide bond."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) == 1

    def test_amide_bond_has_amine_atom(self):
        """Amide bond info should include amine_atom (N)."""
        mol = Chem.MolFromSmiles("CC(=O)NC")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) == 1
        bond = amide_bonds[0]
        assert "amine_atom" in bond
        # The amine_atom should be nitrogen
        n_atom = mol.GetAtomWithIdx(bond["amine_atom"])
        assert n_atom.GetSymbol() == "N"


# ---------------------------------------------------------------------------
# Lactam exclusion
# ---------------------------------------------------------------------------

class TestLactamExclusion:
    """Tests that cyclic amides (lactams) are excluded."""

    def test_gamma_butyrolactam_no_bonds(self):
        """O=C1CCCN1 (2-pyrrolidone) should have 0 cleavable amide bonds."""
        mol = Chem.MolFromSmiles("O=C1CCCN1")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) == 0

    def test_delta_valerolactam_no_bonds(self):
        """O=C1CCCCN1 (delta-valerolactam) should have 0 cleavable amide bonds."""
        mol = Chem.MolFromSmiles("O=C1CCCCN1")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) == 0


# ---------------------------------------------------------------------------
# Carbamate exclusion
# ---------------------------------------------------------------------------

class TestCarbamateExclusion:
    """Tests that carbamate C-N bonds are not detected as standalone amides."""

    def test_carbamate_c_o_detected_as_ester_or_carbamate(self):
        """CCOC(=O)NC (ethyl methylcarbamate): the C-O should be detected
        but the C-N should NOT be a standalone amide."""
        mol = Chem.MolFromSmiles("CCOC(=O)NC")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        # The C-N bond in a carbamate should NOT appear as a standalone amide
        assert len(amide_bonds) == 0

    def test_carbamate_detected_as_carbamate_type(self):
        """CCOC(=O)NC: the carbamate should be detected as type 'carbamate'."""
        mol = Chem.MolFromSmiles("CCOC(=O)NC")
        bonds = find_cleavable_bonds(mol)
        carbamate_bonds = [b for b in bonds if b["type"] == "carbamate"]
        # Should detect at least one carbamate bond
        assert len(carbamate_bonds) >= 1


# ---------------------------------------------------------------------------
# Bond info structure
# ---------------------------------------------------------------------------

class TestBondInfoStructure:
    """Tests for the structure of returned bond info dicts."""

    def test_bond_info_has_required_keys(self):
        """Each bond info dict must have bond_idx, type, and match."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) >= 1
        bond = bonds[0]
        assert "bond_idx" in bond
        assert "type" in bond
        assert "match" in bond

    def test_bond_idx_is_valid(self):
        """bond_idx should be a valid RDKit bond index."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) >= 1
        bond = bonds[0]
        rdkit_bond = mol.GetBondWithIdx(bond["bond_idx"])
        assert rdkit_bond is not None

    def test_no_bonds_in_simple_alkane(self):
        """CCCCCC (hexane) has no cleavable bonds."""
        mol = Chem.MolFromSmiles("CCCCCC")
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) == 0

    def test_no_bonds_in_simple_alcohol(self):
        """CCO (ethanol) has no cleavable bonds."""
        mol = Chem.MolFromSmiles("CCO")
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) == 0
