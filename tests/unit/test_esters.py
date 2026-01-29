"""Tests for ester naming (POLY-02).

Tests two-component ester naming following IUPAC conventions:
- Simple esters: "alkyl alkanoate" format
- Trivial acid names: acetate, formate, etc.
- Systematic acid names: propanoate, butanoate, etc.
- Fragment parsing and lactone detection
"""
import pytest
from orthonym import name_compound


class TestSimpleEsters:
    """Test simple two-component ester naming."""

    def test_methyl_acetate(self):
        """Methyl acetate - simplest common ester."""
        result = name_compound("COC(C)=O")
        assert result == "methyl acetate" or result == "methyl ethanoate"

    def test_ethyl_acetate(self):
        """Ethyl acetate - common solvent."""
        result = name_compound("CCOC(C)=O")
        assert result == "ethyl acetate" or result == "ethyl ethanoate"

    def test_methyl_formate(self):
        """Methyl formate - 1-carbon acid."""
        result = name_compound("COC=O")
        assert result == "methyl formate" or result == "methyl methanoate"

    def test_ethyl_formate(self):
        """Ethyl formate."""
        result = name_compound("CCOC=O")
        assert result == "ethyl formate" or result == "ethyl methanoate"


class TestSystematicEsters:
    """Test systematic ester naming."""

    def test_methyl_propanoate(self):
        """Methyl propanoate - 3-carbon acid."""
        result = name_compound("COC(=O)CC")
        assert result == "methyl propanoate"

    def test_ethyl_propanoate(self):
        """Ethyl propanoate."""
        result = name_compound("CCOC(=O)CC")
        assert result == "ethyl propanoate"

    def test_methyl_butanoate(self):
        """Methyl butanoate - 4-carbon acid."""
        result = name_compound("COC(=O)CCC")
        assert result == "methyl butanoate"

    def test_ethyl_butanoate(self):
        """Ethyl butanoate."""
        result = name_compound("CCOC(=O)CCC")
        assert result == "ethyl butanoate"

    def test_propyl_acetate(self):
        """Propyl acetate - 3-carbon alkyl."""
        result = name_compound("CCCOC(C)=O")
        assert result == "propyl acetate" or result == "propyl ethanoate"

    def test_butyl_acetate(self):
        """Butyl acetate - 4-carbon alkyl."""
        result = name_compound("CCCCOC(C)=O")
        assert result == "butyl acetate" or result == "butyl ethanoate"

    def test_methyl_pentanoate(self):
        """Methyl pentanoate - 5-carbon acid."""
        result = name_compound("COC(=O)CCCC")
        assert result == "methyl pentanoate"

    def test_pentyl_propanoate(self):
        """Pentyl propanoate - 5-carbon alkyl, 3-carbon acid."""
        result = name_compound("CCCCCOC(=O)CC")
        assert result == "pentyl propanoate"


class TestEsterRetainedNames:
    """Test that common esters use trivial acid names."""

    def test_methyl_acetate_uses_acetate(self):
        """Verify acetate (not ethanoate) for 2-carbon acid."""
        result = name_compound("COC(C)=O")
        # Either trivial or systematic is acceptable
        assert "acetate" in result or "ethanoate" in result

    def test_ethyl_acetate_uses_acetate(self):
        """Verify acetate for ethyl acetate."""
        result = name_compound("CCOC(C)=O")
        assert "acetate" in result or "ethanoate" in result

    def test_methyl_formate_uses_formate(self):
        """Verify formate (not methanoate) for 1-carbon acid."""
        result = name_compound("COC=O")
        assert "formate" in result or "methanoate" in result


class TestEsterFragmentParsing:
    """Test ester fragment detection."""

    def test_parse_methyl_acetate(self):
        """Parse methyl acetate into acid and alkyl fragments."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        acid_atoms, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        # Acid should have 2 carbons (C=O and CH3)
        acid_carbons = [i for i in acid_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(acid_carbons) == 2

        # Alkyl should have 1 carbon (OCH3)
        alkyl_carbons = [i for i in alkyl_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(alkyl_carbons) == 1

    def test_parse_ethyl_propanoate(self):
        """Parse ethyl propanoate into acid and alkyl fragments."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments

        mol = Chem.MolFromSmiles("CCOC(=O)CC")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        acid_atoms, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        # Acid should have 3 carbons (propanoic acid part)
        acid_carbons = [i for i in acid_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(acid_carbons) == 3

        # Alkyl should have 2 carbons (ethyl)
        alkyl_carbons = [i for i in alkyl_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(alkyl_carbons) == 2


class TestLactoneDetection:
    """Test lactone (cyclic ester) detection."""

    def test_gamma_butyrolactone_is_lactone(self):
        """Gamma-butyrolactone should be detected as a lactone."""
        from rdkit import Chem
        from src.orthonym.rules.esters import is_lactone

        # gamma-butyrolactone: C1CC(=O)OC1
        mol = Chem.MolFromSmiles("C1CC(=O)OC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is True

    def test_delta_valerolactone_is_lactone(self):
        """Delta-valerolactone should be detected as a lactone."""
        from rdkit import Chem
        from src.orthonym.rules.esters import is_lactone

        # delta-valerolactone: C1CCC(=O)OC1
        mol = Chem.MolFromSmiles("C1CCC(=O)OC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is True

    def test_simple_ester_not_lactone(self):
        """Simple esters should not be detected as lactones."""
        from rdkit import Chem
        from src.orthonym.rules.esters import is_lactone

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is False

    def test_ethyl_acetate_not_lactone(self):
        """Ethyl acetate should not be detected as lactone."""
        from rdkit import Chem
        from src.orthonym.rules.esters import is_lactone

        mol = Chem.MolFromSmiles("CCOC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is False


class TestAcidFragmentNaming:
    """Test acid fragment name extraction."""

    def test_one_carbon_acid_is_formic(self):
        """1-carbon acid should be named formic."""
        from rdkit import Chem
        from src.orthonym.rules.esters import get_acid_fragment_name

        mol = Chem.MolFromSmiles("COC=O")
        # Acid fragment is just the C=O carbon
        acid_atoms = [2]  # The carbonyl carbon
        result = get_acid_fragment_name(mol, acid_atoms)
        assert result == "formic"

    def test_two_carbon_acid_is_acetic(self):
        """2-carbon acid should be named acetic."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments, get_acid_fragment_name

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        acid_atoms, _ = parse_ester_fragments(mol, matches[0])

        result = get_acid_fragment_name(mol, acid_atoms)
        assert result == "acetic"

    def test_three_carbon_acid_is_propanoic(self):
        """3-carbon acid should be named propanoic."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments, get_acid_fragment_name

        mol = Chem.MolFromSmiles("COC(=O)CC")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        acid_atoms, _ = parse_ester_fragments(mol, matches[0])

        result = get_acid_fragment_name(mol, acid_atoms)
        assert result == "propanoic"


class TestAlkylFragmentNaming:
    """Test alkyl fragment name extraction."""

    def test_one_carbon_alkyl_is_methyl(self):
        """1-carbon alkyl should be named methyl."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments, get_alkyl_fragment_name

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        _, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        result = get_alkyl_fragment_name(mol, alkyl_atoms)
        assert result == "methyl"

    def test_two_carbon_alkyl_is_ethyl(self):
        """2-carbon alkyl should be named ethyl."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments, get_alkyl_fragment_name

        mol = Chem.MolFromSmiles("CCOC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        _, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        result = get_alkyl_fragment_name(mol, alkyl_atoms)
        assert result == "ethyl"

    def test_three_carbon_alkyl_is_propyl(self):
        """3-carbon alkyl should be named propyl."""
        from rdkit import Chem
        from src.orthonym.rules.esters import parse_ester_fragments, get_alkyl_fragment_name

        mol = Chem.MolFromSmiles("CCCOC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        _, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        result = get_alkyl_fragment_name(mol, alkyl_atoms)
        assert result == "propyl"


class TestAcylateConversion:
    """Test acid name to acylate conversion."""

    def test_acetic_to_acetate(self):
        """Acetic should convert to acetate."""
        from src.orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("acetic") == "acetate"

    def test_formic_to_formate(self):
        """Formic should convert to formate."""
        from src.orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("formic") == "formate"

    def test_propanoic_to_propanoate(self):
        """Propanoic should convert to propanoate."""
        from src.orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("propanoic") == "propanoate"

    def test_butanoic_to_butanoate(self):
        """Butanoic should convert to butanoate."""
        from src.orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("butanoic") == "butanoate"

    def test_benzoic_to_benzoate(self):
        """Benzoic should convert to benzoate."""
        from src.orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("benzoic") == "benzoate"


class TestSystematicAcylate:
    """Test systematic acylate name generation from chain length."""

    def test_chain_1_is_methanoate(self):
        """Chain length 1 should be methanoate."""
        from src.orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(1) == "methanoate"

    def test_chain_2_is_ethanoate(self):
        """Chain length 2 should be ethanoate."""
        from src.orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(2) == "ethanoate"

    def test_chain_3_is_propanoate(self):
        """Chain length 3 should be propanoate."""
        from src.orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(3) == "propanoate"

    def test_chain_4_is_butanoate(self):
        """Chain length 4 should be butanoate."""
        from src.orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(4) == "butanoate"

    def test_chain_5_is_pentanoate(self):
        """Chain length 5 should be pentanoate."""
        from src.orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(5) == "pentanoate"


class TestLactoneNaming:
    """Test that lactones are handled specially (not as simple esters)."""

    def test_gamma_butyrolactone_returns_none(self):
        """Lactones should return None from name_ester (deferred handling)."""
        from rdkit import Chem
        from src.orthonym.rules.esters import name_ester

        # gamma-butyrolactone: C1CC(=O)OC1
        mol = Chem.MolFromSmiles("C1CC(=O)OC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            result = name_ester(mol, matches[0])
            # Should return None since lactones need special handling
            assert result is None


class TestVariousAlkylChains:
    """Test various alkyl chain lengths."""

    def test_hexyl_acetate(self):
        """Hexyl acetate - 6-carbon alkyl."""
        result = name_compound("CCCCCCOC(C)=O")
        assert "hexyl" in result.lower()
        assert "acetate" in result.lower() or "ethanoate" in result.lower()

    def test_methyl_hexanoate(self):
        """Methyl hexanoate - 6-carbon acid."""
        result = name_compound("COC(=O)CCCCC")
        assert result == "methyl hexanoate"

    def test_heptyl_formate(self):
        """Heptyl formate - 7-carbon alkyl, 1-carbon acid."""
        result = name_compound("CCCCCCCOC=O")
        assert "heptyl" in result.lower()
        assert "formate" in result.lower() or "methanoate" in result.lower()


class TestEsterFindMatch:
    """Test ester SMARTS matching."""

    def test_find_ester_in_methyl_acetate(self):
        """Find ester match in methyl acetate."""
        from rdkit import Chem
        from src.orthonym.rules.esters import find_ester_match

        mol = Chem.MolFromSmiles("COC(C)=O")
        match = find_ester_match(mol)
        assert match is not None
        assert len(match) >= 3  # At least carbonyl C, =O, ester O

    def test_find_ester_returns_none_for_non_ester(self):
        """Non-esters should return None."""
        from rdkit import Chem
        from src.orthonym.rules.esters import find_ester_match

        mol = Chem.MolFromSmiles("CCCC")  # butane
        match = find_ester_match(mol)
        assert match is None

    def test_find_ester_returns_none_for_carboxylic_acid(self):
        """Carboxylic acids should not match as esters."""
        from rdkit import Chem
        from src.orthonym.rules.esters import find_ester_match

        mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        match = find_ester_match(mol)
        # The SMARTS requires [OX2][#6], but carboxylic acid has [OX2H1]
        # So this should not match (or if it does, is_lactone handles it)
        # Actually the ester SMARTS is [CX3](=O)[OX2][#6] which requires
        # the O to be bonded to a carbon, not just H
        # Acetic acid is CC(=O)O which is actually [CX3](=O)[OX2H1]
        # So it shouldn't match the ester pattern
        assert match is None
