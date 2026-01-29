"""Tests for amino acid naming (POLY-07).

Tests both trivial names for standard amino acids and systematic names
for non-standard amino acids.
"""
import pytest
from orthonym import name_compound


class TestStandardAminoAcids:
    """Test trivial names for standard amino acids."""

    def test_glycine(self):
        """Glycine is achiral, simplest amino acid."""
        assert name_compound("NCC(=O)O") == "glycine"

    def test_alanine(self):
        """Alanine: 2-aminopropanoic acid."""
        assert name_compound("CC(N)C(=O)O") == "alanine"

    def test_valine(self):
        """Valine: 2-amino-3-methylbutanoic acid."""
        assert name_compound("CC(C)C(N)C(=O)O") == "valine"

    def test_leucine(self):
        """Leucine: 2-amino-4-methylpentanoic acid."""
        assert name_compound("CC(C)CC(N)C(=O)O") == "leucine"

    def test_isoleucine(self):
        """Isoleucine: 2-amino-3-methylpentanoic acid."""
        assert name_compound("CCC(C)C(N)C(=O)O") == "isoleucine"

    def test_serine(self):
        """Serine: 2-amino-3-hydroxypropanoic acid."""
        result = name_compound("NC(CO)C(=O)O")
        assert result == "serine"

    def test_cysteine(self):
        """Cysteine: 2-amino-3-mercaptopropanoic acid."""
        result = name_compound("NC(CS)C(=O)O")
        assert result == "cysteine"

    def test_methionine(self):
        """Methionine: 2-amino-4-(methylthio)butanoic acid."""
        result = name_compound("CSCC(N)C(=O)O")
        assert result == "methionine"

    def test_phenylalanine(self):
        """Phenylalanine: 2-amino-3-phenylpropanoic acid."""
        result = name_compound("NC(Cc1ccccc1)C(=O)O")
        assert result == "phenylalanine"

    def test_proline(self):
        """Proline: imino acid, cyclic amino acid."""
        result = name_compound("OC(=O)C1CCCN1")
        assert result == "proline"

    def test_threonine(self):
        """Threonine: 2-amino-3-hydroxybutanoic acid."""
        result = name_compound("CC(O)C(N)C(=O)O")
        assert result == "threonine"

    def test_tyrosine(self):
        """Tyrosine: 2-amino-3-(4-hydroxyphenyl)propanoic acid."""
        result = name_compound("NC(Cc1ccc(O)cc1)C(=O)O")
        assert result == "tyrosine"

    def test_tryptophan(self):
        """Tryptophan: contains indole ring."""
        result = name_compound("NC(Cc1c[nH]c2ccccc12)C(=O)O")
        assert result == "tryptophan"


class TestAcidicAminoAcids:
    """Test acidic amino acids and their amides."""

    def test_aspartic_acid(self):
        """Aspartic acid: 2-aminobutanedioic acid."""
        result = name_compound("NC(CC(=O)O)C(=O)O")
        assert result == "aspartic acid"

    def test_glutamic_acid(self):
        """Glutamic acid: 2-aminopentanedioic acid."""
        result = name_compound("NC(CCC(=O)O)C(=O)O")
        assert result == "glutamic acid"

    def test_asparagine(self):
        """Asparagine: 2-amino-3-carbamoylpropanoic acid."""
        result = name_compound("NC(CC(N)=O)C(=O)O")
        assert result == "asparagine"

    def test_glutamine(self):
        """Glutamine: 2-amino-4-carbamoylbutanoic acid."""
        result = name_compound("NC(CCC(N)=O)C(=O)O")
        assert result == "glutamine"


class TestBasicAminoAcids:
    """Test basic amino acids."""

    def test_lysine(self):
        """Lysine: 2,6-diaminohexanoic acid."""
        result = name_compound("NCCCCC(N)C(=O)O")
        assert result == "lysine"

    def test_arginine(self):
        """Arginine: contains guanidino group."""
        result = name_compound("NC(CCCNC(N)=N)C(=O)O")
        assert result == "arginine"

    def test_histidine(self):
        """Histidine: contains imidazole ring."""
        result = name_compound("NC(Cc1cnc[nH]1)C(=O)O")
        assert result == "histidine"


class TestNonStandardAminoAcids:
    """Test non-standard amino acids."""

    def test_sarcosine(self):
        """Sarcosine: N-methylglycine."""
        result = name_compound("CNCC(=O)O")
        assert result == "sarcosine"

    def test_ornithine(self):
        """Ornithine: 2,5-diaminopentanoic acid."""
        result = name_compound("NCCCC(N)C(=O)O")
        assert result == "ornithine"

    def test_gaba(self):
        """GABA: 4-aminobutanoic acid (not alpha-amino acid)."""
        result = name_compound("NCCCC(=O)O")
        # GABA is in our NON_STANDARD dict as "4-aminobutanoic acid"
        assert result == "4-aminobutanoic acid"


class TestAminoAcidDetection:
    """Test amino acid detection logic."""

    def test_not_amino_acid_acid_only(self):
        """Simple carboxylic acid is not an amino acid."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import detect_amino_acid

        # Simple acid - no amino group
        mol = Chem.MolFromSmiles("CC(=O)O")
        assert detect_amino_acid(mol) is False

    def test_not_amino_acid_amine_only(self):
        """Simple amine is not an amino acid."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import detect_amino_acid

        # Simple amine - no acid
        mol = Chem.MolFromSmiles("CCN")
        assert detect_amino_acid(mol) is False

    def test_is_amino_acid_glycine(self):
        """Glycine is detected as amino acid."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import detect_amino_acid

        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert detect_amino_acid(mol) is True

    def test_is_amino_acid_alanine(self):
        """Alanine is detected as amino acid."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import detect_amino_acid

        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        assert detect_amino_acid(mol) is True

    def test_proline_detected(self):
        """Proline (cyclic) is detected as amino acid."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import detect_amino_acid

        mol = Chem.MolFromSmiles("OC(=O)C1CCCN1")
        assert detect_amino_acid(mol) is True


class TestSystematicAminoAcidNaming:
    """Test systematic naming for amino acids not in lookup."""

    def test_2_aminopropanoic_acid_systematic(self):
        """Test systematic naming function directly."""
        from src.orthonym.rules.amino_acids import _name_amino_acid_systematic
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        systematic = _name_amino_acid_systematic(mol)
        assert systematic == "2-aminopropanoic acid"

    def test_2_aminobutanoic_acid_systematic(self):
        """Test 4-carbon amino acid systematic."""
        from src.orthonym.rules.amino_acids import _name_amino_acid_systematic
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCC(N)C(=O)O")
        systematic = _name_amino_acid_systematic(mol)
        assert systematic == "2-aminobutanoic acid"


class TestAminoAcidDataModule:
    """Test the amino acid data module functions."""

    def test_get_amino_acid_name_glycine(self):
        """get_amino_acid_name returns glycine."""
        from src.orthonym.data.amino_acids import get_amino_acid_name
        assert get_amino_acid_name("NCC(=O)O") == "glycine"

    def test_get_amino_acid_name_unknown(self):
        """get_amino_acid_name returns None for unknown."""
        from src.orthonym.data.amino_acids import get_amino_acid_name
        assert get_amino_acid_name("CCCCC") is None

    def test_is_standard_amino_acid(self):
        """is_standard_amino_acid works correctly."""
        from src.orthonym.data.amino_acids import is_standard_amino_acid
        assert is_standard_amino_acid("NCC(=O)O") is True
        assert is_standard_amino_acid("CC(N)C(=O)O") is True
        assert is_standard_amino_acid("CCCCC") is False


class TestNSubstitutedAminoAcidDetection:
    """Test N-substituted amino acid detection."""

    def test_sarcosine_is_n_substituted(self):
        """Sarcosine (N-methylglycine) should be detected as N-substituted."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import is_n_substituted_amino_acid

        mol = Chem.MolFromSmiles("CNCC(=O)O")
        assert is_n_substituted_amino_acid(mol) is True

    def test_glycine_not_n_substituted(self):
        """Glycine is not N-substituted."""
        from rdkit import Chem
        from src.orthonym.rules.amino_acids import is_n_substituted_amino_acid

        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert is_n_substituted_amino_acid(mol) is False


class TestIntegrationWithNameCompound:
    """Test that name_compound correctly handles amino acids."""

    def test_amino_acid_before_polyfunctional(self):
        """Amino acid naming should take precedence over polyfunctional."""
        # Glycine would be "2-aminoacetic acid" via polyfunctional
        # but should return "glycine" via trivial name
        assert name_compound("NCC(=O)O") == "glycine"

    def test_amino_acid_through_full_pipeline(self):
        """Multiple amino acids work through the full pipeline."""
        assert name_compound("NCC(=O)O") == "glycine"
        assert name_compound("CC(N)C(=O)O") == "alanine"
        assert name_compound("NC(Cc1ccccc1)C(=O)O") == "phenylalanine"

    def test_non_amino_acid_not_affected(self):
        """Regular compounds still work correctly."""
        # Simple acid
        assert name_compound("CC(=O)O") == "acetic acid"
        # Simple amine
        assert name_compound("CCN") == "ethylamine"
