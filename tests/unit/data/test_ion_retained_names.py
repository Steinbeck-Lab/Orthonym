"""Unit tests for ion retained names data."""
import pytest
from rdkit import Chem
from src.orthonym.data.ion_retained_names import (
    RETAINED_ANIONS,
    RETAINED_CATIONS,
    INORGANIC_CATIONS,
    INORGANIC_ANIONS,
    get_anion_name,
    get_cation_name,
    get_ion_name,
)


class TestRetainedAnions:
    """Test organic anion name lookup."""

    def test_acetate(self):
        """Test acetate lookup."""
        assert get_anion_name('CC(=O)[O-]') == 'acetate'

    def test_formate(self):
        """Test formate lookup."""
        assert get_anion_name('O=C[O-]') == 'formate'

    def test_benzoate(self):
        """Test benzoate lookup."""
        assert get_anion_name('O=C([O-])c1ccccc1') == 'benzoate'

    def test_methoxide(self):
        """Test methoxide lookup."""
        assert get_anion_name('C[O-]') == 'methoxide'

    def test_ethoxide(self):
        """Test ethoxide lookup."""
        assert get_anion_name('CC[O-]') == 'ethoxide'

    def test_phenoxide(self):
        """Test phenoxide (phenolate) lookup."""
        assert get_anion_name('[O-]c1ccccc1') == 'phenoxide'

    def test_methanide(self):
        """Test methanide (carbanion) lookup."""
        assert get_anion_name('[CH3-]') == 'methanide'

    def test_unknown_returns_none(self):
        """Test unknown anion returns None."""
        assert get_anion_name('CCCC[O-]') is None

    def test_non_canonical_smiles(self):
        """Test non-canonical SMILES is canonicalized."""
        # Different representation of acetate
        name = get_anion_name('[O-]C(=O)C')
        assert name == 'acetate'


class TestRetainedCations:
    """Test organic cation name lookup."""

    def test_ammonium(self):
        """Test ammonium lookup."""
        assert get_cation_name('[NH4+]') == 'ammonium'

    def test_methylammonium(self):
        """Test methylammonium lookup."""
        assert get_cation_name('C[NH3+]') == 'methylammonium'

    def test_ethylammonium(self):
        """Test ethylammonium lookup."""
        assert get_cation_name('CC[NH3+]') == 'ethylammonium'

    def test_tetramethylammonium(self):
        """Test tetramethylammonium lookup."""
        assert get_cation_name('C[N+](C)(C)C') == 'tetramethylammonium'

    def test_methylium(self):
        """Test methylium (carbocation) lookup."""
        assert get_cation_name('[CH3+]') == 'methylium'

    def test_oxonium(self):
        """Test oxonium lookup."""
        assert get_cation_name('[OH3+]') == 'oxonium'

    def test_unknown_returns_none(self):
        """Test unknown cation returns None."""
        assert get_cation_name('CCCC[NH3+]') is None


class TestInorganicCations:
    """Test inorganic cation lookup."""

    def test_sodium_in_cations(self):
        """Test sodium is in INORGANIC_CATIONS."""
        assert '[Na+]' in INORGANIC_CATIONS
        assert INORGANIC_CATIONS['[Na+]'] == 'sodium'

    def test_potassium(self):
        """Test potassium lookup."""
        assert get_cation_name('[K+]') == 'potassium'

    def test_lithium(self):
        """Test lithium lookup."""
        assert get_cation_name('[Li+]') == 'lithium'

    def test_calcium(self):
        """Test calcium lookup."""
        assert get_cation_name('[Ca+2]') == 'calcium'

    def test_magnesium(self):
        """Test magnesium lookup."""
        assert get_cation_name('[Mg+2]') == 'magnesium'

    def test_zinc(self):
        """Test zinc lookup."""
        assert get_cation_name('[Zn+2]') == 'zinc'

    def test_silver(self):
        """Test silver lookup."""
        assert get_cation_name('[Ag+]') == 'silver'

    def test_iron_ii(self):
        """Test iron(II) lookup."""
        assert get_cation_name('[Fe+2]') == 'iron(II)'

    def test_iron_iii(self):
        """Test iron(III) lookup."""
        assert get_cation_name('[Fe+3]') == 'iron(III)'


class TestInorganicAnions:
    """Test inorganic anion lookup."""

    def test_chloride_in_anions(self):
        """Test chloride is in INORGANIC_ANIONS."""
        assert '[Cl-]' in INORGANIC_ANIONS
        assert INORGANIC_ANIONS['[Cl-]'] == 'chloride'

    def test_bromide(self):
        """Test bromide lookup."""
        assert get_anion_name('[Br-]') == 'bromide'

    def test_iodide(self):
        """Test iodide lookup."""
        assert get_anion_name('[I-]') == 'iodide'

    def test_fluoride(self):
        """Test fluoride lookup."""
        assert get_anion_name('[F-]') == 'fluoride'

    def test_hydroxide(self):
        """Test hydroxide lookup."""
        assert get_anion_name('[OH-]') == 'hydroxide'

    def test_cyanide(self):
        """Test cyanide lookup."""
        assert get_anion_name('[C-]#N') == 'cyanide'


class TestGetIonName:
    """Test unified get_ion_name function."""

    def test_cation_lookup(self):
        """Test get_ion_name finds cations."""
        assert get_ion_name('[Na+]') == 'sodium'
        assert get_ion_name('[NH4+]') == 'ammonium'

    def test_anion_lookup(self):
        """Test get_ion_name finds anions."""
        assert get_ion_name('[Cl-]') == 'chloride'
        assert get_ion_name('CC(=O)[O-]') == 'acetate'

    def test_unknown_returns_none(self):
        """Test unknown ion returns None."""
        assert get_ion_name('CCCCCCCC[O-]') is None


class TestDataIntegrity:
    """Test data module integrity."""

    def test_minimum_anion_entries(self):
        """Test RETAINED_ANIONS has minimum entries."""
        assert len(RETAINED_ANIONS) >= 8

    def test_minimum_cation_entries(self):
        """Test RETAINED_CATIONS has minimum entries."""
        assert len(RETAINED_CATIONS) >= 5

    def test_minimum_inorganic_cations(self):
        """Test INORGANIC_CATIONS has minimum entries."""
        assert len(INORGANIC_CATIONS) >= 5

    def test_minimum_inorganic_anions(self):
        """Test INORGANIC_ANIONS has minimum entries."""
        assert len(INORGANIC_ANIONS) >= 5

    def test_total_ion_count(self):
        """Test total ion entries exceeds 20."""
        total = (len(RETAINED_ANIONS) + len(RETAINED_CATIONS) +
                 len(INORGANIC_CATIONS) + len(INORGANIC_ANIONS))
        assert total >= 20

    def test_all_smiles_are_valid(self):
        """Test all SMILES in dictionaries are valid."""
        all_smiles = (list(RETAINED_ANIONS.keys()) +
                      list(RETAINED_CATIONS.keys()) +
                      list(INORGANIC_CATIONS.keys()) +
                      list(INORGANIC_ANIONS.keys()))

        for smiles in all_smiles:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES: {smiles}"


class TestCanonicalSmilesConsistency:
    """Test SMILES canonicalization consistency."""

    def test_anion_smiles_are_canonical(self):
        """Test RETAINED_ANIONS keys are canonical SMILES."""
        for smiles in RETAINED_ANIONS.keys():
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol)
            # Keys should match their canonical form
            assert smiles == canonical, f"Non-canonical: {smiles} -> {canonical}"

    def test_cation_smiles_are_canonical(self):
        """Test RETAINED_CATIONS keys are canonical SMILES."""
        for smiles in RETAINED_CATIONS.keys():
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol)
            assert smiles == canonical, f"Non-canonical: {smiles} -> {canonical}"

    def test_inorganic_cation_smiles_are_canonical(self):
        """Test INORGANIC_CATIONS keys are canonical SMILES."""
        for smiles in INORGANIC_CATIONS.keys():
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol)
            assert smiles == canonical, f"Non-canonical: {smiles} -> {canonical}"

    def test_inorganic_anion_smiles_are_canonical(self):
        """Test INORGANIC_ANIONS keys are canonical SMILES."""
        for smiles in INORGANIC_ANIONS.keys():
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol)
            assert smiles == canonical, f"Non-canonical: {smiles} -> {canonical}"
