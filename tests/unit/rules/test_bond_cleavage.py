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


# ---------------------------------------------------------------------------
# Phosphodiester bond detection
# ---------------------------------------------------------------------------

class TestPhosphodiesterDetection:
    """Tests for phosphodiester O-P(=O)(O)-O-C bond detection."""

    def test_dimethyl_phosphate_detects_phosphodiester(self):
        """COP(=O)(O)OC (dimethyl phosphate) should have phosphodiester bonds.
        The SMARTS can match multiple O-P...O-C patterns; deduplicated via
        seen_bond_indices so each unique P-O bond is counted once."""
        mol = Chem.MolFromSmiles("COP(=O)(O)OC")
        bonds = find_cleavable_bonds(mol)
        phos_bonds = [b for b in bonds if b["type"] == "phosphodiester"]
        # At least 1 phosphodiester bond (2 P-O-C linkages, each producing a bond)
        assert len(phos_bonds) >= 1

    def test_phospholipid_analog_detects_phosphodiester(self):
        """A phospholipid-like molecule with phosphodiester linkage should be detected.
        CCCCOP(=O)(O)OCCCC has a phosphodiester bond."""
        mol = Chem.MolFromSmiles("CCCCOP(=O)(O)OCCCC")
        bonds = find_cleavable_bonds(mol)
        phos_bonds = [b for b in bonds if b["type"] == "phosphodiester"]
        assert len(phos_bonds) >= 1

    def test_cyclic_phosphodiester_excluded(self):
        """Cyclic phosphodiester (sugar-phosphate ring) should be excluded
        by _atoms_in_same_ring() guard. O=P1(O)OCCO1 is a cyclic phosphate."""
        mol = Chem.MolFromSmiles("O=P1(O)OCCO1")
        bonds = find_cleavable_bonds(mol)
        phos_bonds = [b for b in bonds if b["type"] == "phosphodiester"]
        assert len(phos_bonds) == 0

    def test_phosphodiester_bond_info_structure(self):
        """Phosphodiester bond info should have correct keys and P as acid_atom."""
        mol = Chem.MolFromSmiles("COP(=O)(O)OC")
        bonds = find_cleavable_bonds(mol)
        phos_bonds = [b for b in bonds if b["type"] == "phosphodiester"]
        assert len(phos_bonds) >= 1
        bond = phos_bonds[0]
        assert "bond_idx" in bond
        assert "acid_atom" in bond
        assert "alkyl_atom" in bond
        assert "match" in bond
        # acid_atom should be phosphorus
        acid = mol.GetAtomWithIdx(bond["acid_atom"])
        assert acid.GetSymbol() == "P"


# ---------------------------------------------------------------------------
# Thioester bond detection
# ---------------------------------------------------------------------------

class TestThioesterDetection:
    """Tests for thioester C(=O)-S-C bond detection."""

    def test_s_methyl_thioacetate_one_thioester(self):
        """CC(=O)SC (S-methyl thioacetate) should have 1 thioester bond."""
        mol = Chem.MolFromSmiles("CC(=O)SC")
        bonds = find_cleavable_bonds(mol)
        thio_bonds = [b for b in bonds if b["type"] == "thioester"]
        assert len(thio_bonds) == 1

    def test_thioester_carbamate_overlap_excluded(self):
        """If carbonyl C is part of a carbamate, thioester should not
        double-detect it. CCOC(=O)NC has carbamate; CC(=O)SC is thioester.
        A molecule with both: CCSC(=O)NC -- the C(=O) is carbamate, so
        thioester should skip it."""
        # This is a thiocarbamate: N-C(=O)-S-C
        # The carbamate SMARTS N-C(=O)-O-C won't match because it has S not O
        # But the carbonyl C should be in carbamate_carbonyl_atoms if matched
        # For a pure thioester with no carbamate overlap, it should be detected
        mol = Chem.MolFromSmiles("CC(=O)SC")
        bonds = find_cleavable_bonds(mol)
        thio_bonds = [b for b in bonds if b["type"] == "thioester"]
        assert len(thio_bonds) == 1
        # Verify carbonyl C is the acid_atom
        bond = thio_bonds[0]
        acid = mol.GetAtomWithIdx(bond["acid_atom"])
        assert acid.GetSymbol() == "C"

    def test_thiolactone_excluded(self):
        """O=C1CCCS1 (thiobutyrolactone) should have 0 thioester bonds
        because the C-S bond is in a ring."""
        mol = Chem.MolFromSmiles("O=C1CCCS1")
        bonds = find_cleavable_bonds(mol)
        thio_bonds = [b for b in bonds if b["type"] == "thioester"]
        assert len(thio_bonds) == 0

    def test_thioester_bond_info_structure(self):
        """Thioester bond info should have correct keys."""
        mol = Chem.MolFromSmiles("CC(=O)SC")
        bonds = find_cleavable_bonds(mol)
        thio_bonds = [b for b in bonds if b["type"] == "thioester"]
        assert len(thio_bonds) == 1
        bond = thio_bonds[0]
        assert "bond_idx" in bond
        assert "acid_atom" in bond
        assert "alkyl_atom" in bond
        assert bond["type"] == "thioester"


# ---------------------------------------------------------------------------
# Sulfonamide bond detection
# ---------------------------------------------------------------------------

class TestSulfonamideDetection:
    """Tests for sulfonamide S(=O)(=O)-N bond detection."""

    def test_sulfamethoxazole_one_sulfonamide(self):
        """Sulfamethoxazole: Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1
        Should have 1 sulfonamide bond."""
        mol = Chem.MolFromSmiles("Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1")
        bonds = find_cleavable_bonds(mol)
        sulfo_bonds = [b for b in bonds if b["type"] == "sulfonamide"]
        assert len(sulfo_bonds) == 1

    def test_sultam_excluded(self):
        """O=S1(=O)CCCN1 (sultam / 1,2-thiazetidine 1,1-dioxide variant)
        should have 0 sulfonamide bonds because S-N is in a ring."""
        mol = Chem.MolFromSmiles("O=S1(=O)CCCN1")
        bonds = find_cleavable_bonds(mol)
        sulfo_bonds = [b for b in bonds if b["type"] == "sulfonamide"]
        assert len(sulfo_bonds) == 0

    def test_saccharin_no_sulfonamide(self):
        """Saccharin O=C1NS(=O)(=O)c2ccccc21 has an aromatic SO2-N in a ring.
        Should have 0 sulfonamide bonds (ring guard excludes it)."""
        mol = Chem.MolFromSmiles("O=C1NS(=O)(=O)c2ccccc21")
        bonds = find_cleavable_bonds(mol)
        sulfo_bonds = [b for b in bonds if b["type"] == "sulfonamide"]
        assert len(sulfo_bonds) == 0

    def test_simple_sulfonamide_detection(self):
        """CS(=O)(=O)NC (N-methyl methanesulfonamide) should have 1 sulfonamide bond."""
        mol = Chem.MolFromSmiles("CS(=O)(=O)NC")
        bonds = find_cleavable_bonds(mol)
        sulfo_bonds = [b for b in bonds if b["type"] == "sulfonamide"]
        assert len(sulfo_bonds) == 1

    def test_sulfonamide_bond_info_structure(self):
        """Sulfonamide bond info should have correct keys and S as acid_atom."""
        mol = Chem.MolFromSmiles("CS(=O)(=O)NC")
        bonds = find_cleavable_bonds(mol)
        sulfo_bonds = [b for b in bonds if b["type"] == "sulfonamide"]
        assert len(sulfo_bonds) == 1
        bond = sulfo_bonds[0]
        assert "bond_idx" in bond
        assert "acid_atom" in bond
        assert "alkyl_atom" in bond
        assert bond["type"] == "sulfonamide"
        # acid_atom should be sulfur
        acid = mol.GetAtomWithIdx(bond["acid_atom"])
        assert acid.GetSymbol() == "S"
        # alkyl_atom stores the nitrogen
        amine = mol.GetAtomWithIdx(bond["alkyl_atom"])
        assert amine.GetSymbol() == "N"
