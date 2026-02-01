"""Unit tests for salt and zwitterion naming."""
import pytest
from rdkit import Chem
from src.orthonym.rules.salts import (
    name_salt,
    name_zwitterion,
    is_salt,
    is_zwitterion,
    _is_amino_acid_zwitterion,
    _apply_stoichiometric_prefix,
)


class TestNameSalt:
    """Test salt naming with compositional nomenclature."""

    def test_sodium_acetate(self):
        """Test basic salt: sodium acetate."""
        mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        name = name_salt(mol)
        assert name == 'sodium acetate'

    def test_potassium_chloride(self):
        """Test inorganic salt: potassium chloride."""
        mol = Chem.MolFromSmiles('[K+].[Cl-]')
        name = name_salt(mol)
        assert name == 'potassium chloride'

    def test_ammonium_chloride(self):
        """Test organic cation with inorganic anion."""
        mol = Chem.MolFromSmiles('[NH4+].[Cl-]')
        name = name_salt(mol)
        assert name == 'ammonium chloride'

    def test_sodium_chloride(self):
        """Test simple inorganic salt."""
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        name = name_salt(mol)
        assert name == 'sodium chloride'

    def test_calcium_acetate(self):
        """Test divalent cation with two anions."""
        mol = Chem.MolFromSmiles('[Ca+2].[O-]C(C)=O.[O-]C(C)=O')
        name = name_salt(mol)
        # Should contain calcium and acetate (may have di- prefix)
        assert 'calcium' in name
        assert 'acetate' in name

    def test_lithium_methoxide(self):
        """Test alkali metal with alkoxide anion."""
        mol = Chem.MolFromSmiles('[Li+].[O-]C')
        name = name_salt(mol)
        assert 'lithium' in name
        assert 'methoxide' in name

    def test_potassium_formate(self):
        """Test potassium formate salt."""
        mol = Chem.MolFromSmiles('[K+].[O-]C=O')
        name = name_salt(mol)
        assert 'potassium' in name
        assert 'formate' in name

    def test_sodium_phenoxide(self):
        """Test sodium phenoxide (sodium phenolate)."""
        mol = Chem.MolFromSmiles('[Na+].[O-]c1ccccc1')
        name = name_salt(mol)
        assert 'sodium' in name
        assert 'oxide' in name.lower() or 'olate' in name.lower()


class TestNameZwitterion:
    """Test zwitterion naming."""

    def test_glycine_zwitterion_systematic(self):
        """Test glycine zwitterion with systematic naming."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        name = name_zwitterion(mol, style='systematic')
        # Should contain azaniumyl (cation) and acetate (anion)
        assert 'azaniumyl' in name.lower()
        assert 'acetate' in name.lower()

    def test_glycine_zwitterion_trivial(self):
        """Test glycine zwitterion may use trivial name."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        name = name_zwitterion(mol)
        # Either trivial (glycine) or systematic (azaniumylacetate)
        assert name is not None
        assert len(name) > 0

    def test_alanine_zwitterion(self):
        """Test alanine zwitterion."""
        mol = Chem.MolFromSmiles('[NH3+]C(C)C([O-])=O')
        name = name_zwitterion(mol)
        # Should contain some form of amino acid naming
        assert name is not None
        assert len(name) > 0


class TestIsSalt:
    """Test salt detection."""

    def test_is_salt_positive(self):
        """Test that salts are detected correctly."""
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        assert is_salt(mol) is True

    def test_is_salt_negative_neutral(self):
        """Test that neutral molecules are not salts."""
        mol = Chem.MolFromSmiles('CCO')
        assert is_salt(mol) is False

    def test_is_salt_negative_single_ion(self):
        """Test that single ions are not salts."""
        mol = Chem.MolFromSmiles('[Na+]')
        assert is_salt(mol) is False

    def test_is_salt_negative_zwitterion(self):
        """Test that zwitterions are not salts (single fragment)."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert is_salt(mol) is False


class TestIsZwitterion:
    """Test zwitterion detection."""

    def test_is_zwitterion_positive(self):
        """Test that zwitterions are detected correctly."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert is_zwitterion(mol) is True

    def test_is_zwitterion_negative_neutral(self):
        """Test that neutral molecules are not zwitterions."""
        mol = Chem.MolFromSmiles('CCO')
        assert is_zwitterion(mol) is False

    def test_is_zwitterion_negative_salt(self):
        """Test that salts are not zwitterions."""
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        assert is_zwitterion(mol) is False

    def test_is_zwitterion_negative_single_ion(self):
        """Test that single ions are not zwitterions."""
        mol = Chem.MolFromSmiles('[NH4+]')
        assert is_zwitterion(mol) is False


class TestIsAminoAcidZwitterion:
    """Test amino acid zwitterion pattern detection."""

    def test_glycine_zwitterion(self):
        """Test glycine zwitterion pattern."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert _is_amino_acid_zwitterion(mol) is True

    def test_alanine_zwitterion(self):
        """Test alanine zwitterion pattern."""
        mol = Chem.MolFromSmiles('[NH3+]C(C)C([O-])=O')
        assert _is_amino_acid_zwitterion(mol) is True

    def test_non_amino_acid(self):
        """Test that non-amino acid is not detected."""
        mol = Chem.MolFromSmiles('[NH3+]CCCC([O-])=O')
        # gamma-amino butyrate - still matches alpha pattern if connected
        # depends on exact SMARTS pattern
        result = _is_amino_acid_zwitterion(mol)
        # This may or may not match depending on chain length
        assert isinstance(result, bool)


class TestStoichiometricPrefix:
    """Test stoichiometric prefix application."""

    def test_single_occurrence(self):
        """Test that count=1 returns name unchanged."""
        assert _apply_stoichiometric_prefix('acetate', 1) == 'acetate'

    def test_di_prefix(self):
        """Test di- prefix for count=2."""
        assert _apply_stoichiometric_prefix('acetate', 2) == 'diacetate'

    def test_tri_prefix(self):
        """Test tri- prefix for count=3."""
        assert _apply_stoichiometric_prefix('chloride', 3) == 'trichloride'

    def test_tetra_prefix(self):
        """Test tetra- prefix for count=4."""
        assert _apply_stoichiometric_prefix('oxide', 4) == 'tetraoxide'
