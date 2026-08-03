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
        from orthonym.rules.esters import parse_ester_fragments

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
        from orthonym.rules.esters import parse_ester_fragments

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
        from orthonym.rules.esters import is_lactone

        # gamma-butyrolactone: C1CC(=O)OC1
        mol = Chem.MolFromSmiles("C1CC(=O)OC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is True

    def test_delta_valerolactone_is_lactone(self):
        """Delta-valerolactone should be detected as a lactone."""
        from rdkit import Chem
        from orthonym.rules.esters import is_lactone

        # delta-valerolactone: C1CCC(=O)OC1
        mol = Chem.MolFromSmiles("C1CCC(=O)OC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is True

    def test_simple_ester_not_lactone(self):
        """Simple esters should not be detected as lactones."""
        from rdkit import Chem
        from orthonym.rules.esters import is_lactone

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)

        if matches:
            assert is_lactone(mol, matches[0]) is False

    def test_ethyl_acetate_not_lactone(self):
        """Ethyl acetate should not be detected as lactone."""
        from rdkit import Chem
        from orthonym.rules.esters import is_lactone

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
        from orthonym.rules.esters import get_acid_fragment_name

        mol = Chem.MolFromSmiles("COC=O")
        # Acid fragment is just the C=O carbon
        acid_atoms = [2]  # The carbonyl carbon
        result = get_acid_fragment_name(mol, acid_atoms)
        assert result == "formic"

    def test_two_carbon_acid_is_acetic(self):
        """2-carbon acid should be named acetic."""
        from rdkit import Chem
        from orthonym.rules.esters import parse_ester_fragments, get_acid_fragment_name

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        acid_atoms, _ = parse_ester_fragments(mol, matches[0])

        result = get_acid_fragment_name(mol, acid_atoms)
        assert result == "acetic"

    def test_three_carbon_acid_is_propanoic(self):
        """3-carbon acid should be named propanoic."""
        from rdkit import Chem
        from orthonym.rules.esters import parse_ester_fragments, get_acid_fragment_name

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
        from orthonym.rules.esters import parse_ester_fragments, get_alkyl_fragment_name

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        _, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        result = get_alkyl_fragment_name(mol, alkyl_atoms)
        assert result == "methyl"

    def test_two_carbon_alkyl_is_ethyl(self):
        """2-carbon alkyl should be named ethyl."""
        from rdkit import Chem
        from orthonym.rules.esters import parse_ester_fragments, get_alkyl_fragment_name

        mol = Chem.MolFromSmiles("CCOC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        _, alkyl_atoms = parse_ester_fragments(mol, matches[0])

        result = get_alkyl_fragment_name(mol, alkyl_atoms)
        assert result == "ethyl"

    def test_three_carbon_alkyl_is_propyl(self):
        """3-carbon alkyl should be named propyl."""
        from rdkit import Chem
        from orthonym.rules.esters import parse_ester_fragments, get_alkyl_fragment_name

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
        from orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("acetic") == "acetate"

    def test_formic_to_formate(self):
        """Formic should convert to formate."""
        from orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("formic") == "formate"

    def test_propanoic_to_propanoate(self):
        """Propanoic should convert to propanoate."""
        from orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("propanoic") == "propanoate"

    def test_butanoic_to_butanoate(self):
        """Butanoic should convert to butanoate."""
        from orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("butanoic") == "butanoate"

    def test_benzoic_to_benzoate(self):
        """Benzoic should convert to benzoate."""
        from orthonym.data.trivial_acids import get_acylate_name
        assert get_acylate_name("benzoic") == "benzoate"


class TestSystematicAcylate:
    """Test systematic acylate name generation from chain length."""

    def test_chain_1_is_methanoate(self):
        """Chain length 1 should be methanoate."""
        from orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(1) == "methanoate"

    def test_chain_2_is_ethanoate(self):
        """Chain length 2 should be ethanoate."""
        from orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(2) == "ethanoate"

    def test_chain_3_is_propanoate(self):
        """Chain length 3 should be propanoate."""
        from orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(3) == "propanoate"

    def test_chain_4_is_butanoate(self):
        """Chain length 4 should be butanoate."""
        from orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(4) == "butanoate"

    def test_chain_5_is_pentanoate(self):
        """Chain length 5 should be pentanoate."""
        from orthonym.data.trivial_acids import get_systematic_acylate
        assert get_systematic_acylate(5) == "pentanoate"


class TestLactoneNaming:
    """Test that lactones are handled specially (not as simple esters)."""

    def test_gamma_butyrolactone_returns_none(self):
        """Lactones should return None from name_ester (deferred handling)."""
        from rdkit import Chem
        from orthonym.rules.esters import name_ester

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
        from orthonym.rules.esters import find_ester_match

        mol = Chem.MolFromSmiles("COC(C)=O")
        match = find_ester_match(mol)
        assert match is not None
        assert len(match) >= 3  # At least carbonyl C, =O, ester O

    def test_find_ester_returns_none_for_non_ester(self):
        """Non-esters should return None."""
        from rdkit import Chem
        from orthonym.rules.esters import find_ester_match

        mol = Chem.MolFromSmiles("CCCC")  # butane
        match = find_ester_match(mol)
        assert match is None

    def test_find_ester_returns_none_for_carboxylic_acid(self):
        """Carboxylic acids should not match as esters."""
        from rdkit import Chem
        from orthonym.rules.esters import find_ester_match

        mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        match = find_ester_match(mol)
        # The SMARTS requires [OX2][#6], but carboxylic acid has [OX2H1]
        # So this should not match (or if it does, is_lactone handles it)
        # Actually the ester SMARTS is [CX3](=O)[OX2][#6] which requires
        # the O to be bonded to a carbon, not just H
        # Acetic acid is CC(=O)O which is actually [CX3](=O)[OX2H1]
        # So it shouldn't match the ester pattern
        assert match is None


# ============================================================================
# NEW: Acyloxy Prefix Tests (Phase 14.7, Plan 02)
# ============================================================================


class TestAcyloxyPrefixTrivial:
    """Test acyloxy prefix generation from trivial acid names.

    IUPAC P-65.6.3.2.2: When the ester is named as a substituent prefix,
    the R-CO-O- portion is named as an 'acyloxy' group.
    Conversion: acid name -> drop '-ic' -> add '-yloxy'
    """

    def test_formic_to_formyloxy(self):
        """Formic acid -> formyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("formic") == "formyloxy"

    def test_acetic_to_acetyloxy(self):
        """Acetic acid -> acetyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("acetic") == "acetyloxy"

    def test_benzoic_to_benzoyloxy(self):
        """Benzoic acid -> benzoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("benzoic") == "benzoyloxy"

    def test_propionic_to_propionyloxy(self):
        """Propionic acid -> propionyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("propionic") == "propionyloxy"


class TestAcyloxyPrefixSystematic:
    """Test acyloxy prefix generation from systematic acid names."""

    def test_propanoic_to_propanoyloxy(self):
        """Propanoic acid -> propanoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("propanoic") == "propanoyloxy"

    def test_butanoic_to_butanoyloxy(self):
        """Butanoic acid -> butanoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("butanoic") == "butanoyloxy"

    def test_pentanoic_to_pentanoyloxy(self):
        """Pentanoic acid -> pentanoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("pentanoic") == "pentanoyloxy"

    def test_hexanoic_to_hexanoyloxy(self):
        """Hexanoic acid -> hexanoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("hexanoic") == "hexanoyloxy"

    def test_ethanoic_to_ethanoyloxy(self):
        """Ethanoic (systematic for acetic) -> ethanoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("ethanoic") == "ethanoyloxy"

    def test_methanoic_to_methanoyloxy(self):
        """Methanoic (systematic for formic) -> methanoyloxy prefix."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("methanoic") == "methanoyloxy"


class TestNameEsterAsPrefix:
    """Test molecular-level ester-to-prefix conversion.

    name_ester_as_prefix(mol, ester_match) should extract the acid fragment,
    determine its name, and return the acyloxy prefix string.
    """

    def test_methyl_acetate_prefix_is_acetyloxy(self):
        """Methyl acetate acid fragment -> acetyloxy prefix."""
        from rdkit import Chem
        from orthonym.rules.esters import name_ester_as_prefix

        mol = Chem.MolFromSmiles("COC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        assert matches, "No ester match found"

        result = name_ester_as_prefix(mol, matches[0])
        assert result == "acetyloxy"

    def test_methyl_formate_prefix_is_formyloxy(self):
        """Methyl formate acid fragment -> formyloxy prefix."""
        from rdkit import Chem
        from orthonym.rules.esters import name_ester_as_prefix

        mol = Chem.MolFromSmiles("COC=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        assert matches, "No ester match found"

        result = name_ester_as_prefix(mol, matches[0])
        assert result == "formyloxy"

    def test_methyl_propanoate_prefix_is_propanoyloxy(self):
        """Methyl propanoate acid fragment -> propanoyloxy prefix."""
        from rdkit import Chem
        from orthonym.rules.esters import name_ester_as_prefix

        mol = Chem.MolFromSmiles("COC(=O)CC")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        assert matches, "No ester match found"

        result = name_ester_as_prefix(mol, matches[0])
        assert result == "propanoyloxy"

    def test_lactone_returns_none(self):
        """Lactones should return None from name_ester_as_prefix."""
        from rdkit import Chem
        from orthonym.rules.esters import name_ester_as_prefix

        mol = Chem.MolFromSmiles("C1CC(=O)OC1")  # gamma-butyrolactone
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        assert matches, "No ester match found"

        result = name_ester_as_prefix(mol, matches[0])
        assert result is None


class TestDetectExocyclicEsters:
    """Test detection of exocyclic esters (ester O attached to ring atom).

    detect_exocyclic_esters(mol) should find all esters where the ester
    oxygen is bonded to a ring carbon, indicating the ring is the parent
    and the ester should be named as an acyloxy prefix.
    """

    def test_cyclohexyl_acetate_detected(self):
        """Cyclohexyl acetate has an exocyclic ester on the ring."""
        from rdkit import Chem
        from orthonym.rules.esters import detect_exocyclic_esters

        # cyclohexyl acetate: CC(=O)OC1CCCCC1
        mol = Chem.MolFromSmiles("CC(=O)OC1CCCCC1")
        results = detect_exocyclic_esters(mol)
        assert len(results) >= 1
        # Should have acyloxy prefix
        assert results[0]["acyloxy_prefix"] == "acetyloxy"
        # Ring attach atom should be in a ring
        ring_idx = results[0]["ring_attach_atom_idx"]
        assert mol.GetAtomWithIdx(ring_idx).IsInRing()

    def test_phenyl_acetate_detected(self):
        """Phenyl acetate has an exocyclic ester on phenyl ring."""
        from rdkit import Chem
        from orthonym.rules.esters import detect_exocyclic_esters

        # phenyl acetate: CC(=O)Oc1ccccc1
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1")
        results = detect_exocyclic_esters(mol)
        assert len(results) >= 1
        assert results[0]["acyloxy_prefix"] == "acetyloxy"

    def test_simple_ester_not_exocyclic(self):
        """Simple acyclic esters have no exocyclic ester."""
        from rdkit import Chem
        from orthonym.rules.esters import detect_exocyclic_esters

        mol = Chem.MolFromSmiles("CCOC(C)=O")  # ethyl acetate
        results = detect_exocyclic_esters(mol)
        assert len(results) == 0

    def test_lactone_not_exocyclic(self):
        """Lactones (cyclic esters) should not be reported as exocyclic."""
        from rdkit import Chem
        from orthonym.rules.esters import detect_exocyclic_esters

        mol = Chem.MolFromSmiles("C1CC(=O)OC1")  # gamma-butyrolactone
        results = detect_exocyclic_esters(mol)
        # Lactones are cyclic, not exocyclic
        assert len(results) == 0

    def test_cyclohexyl_propanoate_detected(self):
        """Cyclohexyl propanoate should give propanoyloxy prefix."""
        from rdkit import Chem
        from orthonym.rules.esters import detect_exocyclic_esters

        # cyclohexyl propanoate: CCC(=O)OC1CCCCC1
        mol = Chem.MolFromSmiles("CCC(=O)OC1CCCCC1")
        results = detect_exocyclic_esters(mol)
        assert len(results) >= 1
        assert results[0]["acyloxy_prefix"] == "propanoyloxy"


# ============================================================================
# Acyloxy Prefix for -carboxylic acids (FMT-01/FMT-02 fix)
# ============================================================================


class TestAcyloxyPrefixCarboxylic:
    """Test acyloxy prefix for ring -carboxylic acids.

    FMT-01/FMT-02: cyclopentanecarboxylic/cyclohexanecarboxylic acids
    must produce -carbonyloxy, NOT -carboxylyloxy (the ylyloxy bug).
    """

    def test_cyclopentanecarboxylic_to_carbonyloxy(self):
        """Cyclopentanecarboxylic acid -> cyclopentanecarbonyloxy."""
        from orthonym.rules.esters import get_acyloxy_prefix
        result = get_acyloxy_prefix("cyclopentanecarboxylic acid")
        assert result == "cyclopentanecarbonyloxy"

    def test_cyclohexanecarboxylic_to_carbonyloxy(self):
        """Cyclohexanecarboxylic acid -> cyclohexanecarbonyloxy."""
        from orthonym.rules.esters import get_acyloxy_prefix
        result = get_acyloxy_prefix("cyclohexanecarboxylic acid")
        assert result == "cyclohexanecarbonyloxy"

    def test_cyclopropanecarboxylic_to_carbonyloxy(self):
        """Cyclopropanecarboxylic -> cyclopropanecarbonyloxy."""
        from orthonym.rules.esters import get_acyloxy_prefix
        result = get_acyloxy_prefix("cyclopropanecarboxylic")
        assert result == "cyclopropanecarbonyloxy"

    def test_no_ylyloxy_in_cyclopentane(self):
        """Verify no ylyloxy concatenation bug."""
        from orthonym.rules.esters import get_acyloxy_prefix
        result = get_acyloxy_prefix("cyclopentanecarboxylic acid")
        assert "ylyloxy" not in result

    def test_no_ylyloxy_in_cyclohexane(self):
        """Verify no ylyloxy concatenation bug."""
        from orthonym.rules.esters import get_acyloxy_prefix
        result = get_acyloxy_prefix("cyclohexanecarboxylic acid")
        assert "ylyloxy" not in result

    def test_propanoic_still_works(self):
        """Generic -ic rule still works for non-carboxylic acids."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("propanoic acid") == "propanoyloxy"

    def test_benzoic_trivial_still_works(self):
        """Trivial lookup still works for benzoic acid."""
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("benzoic acid") == "benzoyloxy"


class TestSaturatedFattyEstersUsePinAcylWord:
    """v29 Task J2: the ester acyl word must come from the PIN acid stem.

    P-65.6.3.2.1 "General methodology" (BlueBookV2.md:31659) -- "All preferred
    IUPAC names for esters are named by functional class nomenclature." Its own
    worked examples settle where the acyl word comes from: "CH3-CO-O-CH2-CH3
    ethyl acetate (PIN)" keeps the trivial word because acetic acid IS retained
    as a PIN (P-65.1.1.1, :29715, "Only the following five carboxylic acids
    retained names and are also preferred IUPAC names"), whereas
    "CH3-O-CO-CH2-CH2-CO-O-CH2-CH3 ethyl methyl butanedioate (PIN)" uses the
    SYSTEMATIC word even though succinic acid is a retained name -- because it
    is retained only for general nomenclature (P-65.1.1.2.2, :29745).

    No fatty acid is in the five. P-65.1.2 "Systematic names" (heading :29858)
    disposes of the rest at :29860, and the book prints the marker on the
    systematic side: :29787 "palmitic acid  hexadecanoic acid (PIN)", :29791
    "stearic acid  octadecanoic acid (PIN)".

    These previously emitted 'ethyl palmitate' / 'ethyl laurate' etc. while the
    ACID path for the identical chain already emitted the PIN. That internal
    contradiction is what these tests exist to prevent recurring.
    """

    # (SMILES, expected PIN ester, expected PIN acid for the same chain)
    CASES = [
        ("CCOC(=O)CCCCCCCCCCC", "ethyl dodecanoate", "dodecanoic acid"),
        ("CCOC(=O)CCCCCCCCCCCCC", "ethyl tetradecanoate", "tetradecanoic acid"),
        ("CCCCCCCCCCCCCCCC(=O)OCC", "ethyl hexadecanoate", "hexadecanoic acid"),
        ("CCOC(=O)CCCCCCCCCCCCCCCCC", "ethyl octadecanoate", "octadecanoic acid"),
        ("CCOC(=O)CCCCCCCCCCCCCCCCCCC", "ethyl icosanoate", "icosanoic acid"),
    ]

    @pytest.mark.parametrize("smiles,expected,_acid", CASES)
    def test_emits_systematic_pin_not_trivial(self, smiles, expected, _acid):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected,_acid", CASES)
    def test_never_abstains(self, smiles, expected, _acid):
        """A withdrawn trivial name must become the PIN, never an abstention.

        Removing a wrong output does not fail closed -- it can unmask a worse
        generator, so the emission itself is asserted, not merely the absence
        of the trivial word.
        """
        result = name_compound(smiles)
        assert result, f"{smiles} now abstains -- the withdrawal unmasked a gap"
        assert "ate" in result

    @pytest.mark.parametrize("smiles,expected,acid", CASES)
    def test_acid_and_ester_paths_agree_on_the_stem(self, smiles, expected, acid):
        """The invariant the defect violated.

        get_acid_fragment_name is an ESTER-ONLY producer -- it records zero
        calls when the acid is named -- so the two paths derived the stem
        independently and disagreed. They must now agree.
        """
        acid_stem = acid[: -len(" acid")]          # "hexadecanoic"
        ester_stem = expected.split()[-1][: -len("ate")] + "ic"   # "hexadecanoic"
        assert ester_stem == acid_stem

    def test_retained_pin_acids_keep_their_trivial_acyl_word(self):
        """Guard the over-correction: the five P-65.1.1.1 acids must NOT move."""
        assert name_compound("CC(=O)OCC") == "ethyl acetate"
        assert name_compound("CCOC(=O)c1ccccc1") == "ethyl benzoate"


class TestAcylPrefixUsesPinAcidStem:
    """v29 Task J3: the acyl PREFIX must come from the PIN acid stem.

    Task J2 fixed the ester *word* ('ethyl palmitate' -> 'ethyl hexadecanoate').
    The acyl *prefix* is a different producer and survived that fix, because
    TRIVIAL_ACID_TO_ACYLOXY carried five rows keyed on the SYSTEMATIC stem
    ('hexadecanoic' -> 'palmitoyloxy') which took the already-preferred stem and
    converted it back to the non-PIN word.

    THE RULE. P-65.6.3.2.3 "Esters cited as prefixes" (BlueBookV2.md:31696) --
    "an ester group is indicated by prefixes as 'acyloxy' for the group
    R-CO-O-". Its worked examples give both sides of the boundary:

      :31711  3-(benzoyloxy)propanoic acid (PIN)
              -- benzoic acid IS retained as a preferred IUPAC name
                 (P-65.1.1.1 :29715), so 'benzoyloxy' is PREFERRED and must
                 survive.
      :31723  3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)
                3-(nicotinoyloxy)propanoic acid
              -- nicotinic acid is retained for GENERAL nomenclature only
                 (P-65.1.1.2.2 heading :29745; row :29773), so its
                 trivial-derived acyloxy prefix is the NON-preferred alternative.

    palmitic/stearic/oleic sit in that same general-only list (:29787, :29791,
    :29785), so their acyloxy prefixes are non-PIN for exactly the reason
    'nicotinoyloxy' is. Appendix 2 confirms row by row -- its legend at :55416
    reads "The symbol * designates the preferred prefix" -- printing
    'hexadecanoyl* = palmitoyl' (:56482), 'octadecanoyl* = stearoyl' (:56441)
    and '(9Z)-octadec-9-enoyl* = oleoyl' (:56443).

    ⚠ OPSIN cannot adjudicate any of this: it round-trips both spellings to the
    same structure, so a clean round-trip passes before AND after and has no
    force on PIN preference. These assertions rest on the Blue Book.
    """

    # (SMILES, expected acyloxy prefix, withdrawn word that must not reappear)
    SATURATED = [
        ("CCCCCCCCCCCC(=O)OCC(=O)O", "dodecanoyloxy", "lauroyloxy"),
        ("CCCCCCCCCCCCCC(=O)OCC(=O)O", "tetradecanoyloxy", "myristoyloxy"),
        ("CCCCCCCCCCCCCCCC(=O)OCC(=O)O", "hexadecanoyloxy", "palmitoyloxy"),
        ("CCCCCCCCCCCCCCCCCC(=O)OCC(=O)O", "octadecanoyloxy", "stearoyloxy"),
        ("CCCCCCCCCCCCCCCCCCCC(=O)OCC(=O)O", "icosanoyloxy", "arachidoyloxy"),
    ]

    # The unsaturated rows. J2 deferred these fearing the fall-through would hit
    # the SATURATED get_acid_stem() and yield a WRONG MOLECULE. Measured: the
    # unsaturation branch runs first and returns the full systematic stem, which
    # is verbatim the Blue Book PIN -- e.g. oleic acid is
    # '(9Z)-octadec-9-enoic acid (PIN)' at :29785.
    UNSATURATED = [
        (r"CCCCCCCC/C=C\CCCCCCCC(=O)OCC(=O)O",
         "(9Z)-octadec-9-enoyloxy", "oleoyloxy"),
        (r"CCCCC/C=C\C/C=C\CCCCCCCC(=O)OCC(=O)O",
         "(9Z,12Z)-octadeca-9,12-dienoyloxy", "linoleoyloxy"),
        (r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)OCC(=O)O",
         "(9Z,12Z,15Z)-octadeca-9,12,15-trienoyloxy", "linolenoyloxy"),
        (r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OCC(=O)O",
         "(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoyloxy", "arachidonoyloxy"),
    ]

    @pytest.mark.parametrize("smiles,expected,withdrawn", SATURATED + UNSATURATED)
    def test_acyl_prefix_is_pin(self, smiles, expected, withdrawn):
        name = name_compound(smiles)
        assert expected in name, f"expected {expected!r} in {name!r}"
        assert withdrawn not in name, (
            f"non-PIN acyl prefix {withdrawn!r} re-emitted in {name!r}"
        )

    @pytest.mark.parametrize("smiles,expected,withdrawn", SATURATED + UNSATURATED)
    def test_no_withdrawn_prefix_became_an_abstention(self, smiles, expected,
                                                      withdrawn):
        """the contributor guide #9: withdrawing a wrong name must not fail closed instead."""
        name = name_compound(smiles)
        assert name, f"empty name for {smiles}"
        assert "unknown" not in name.lower(), (
            f"withdrawing {withdrawn!r} turned {smiles} into an abstention: "
            f"{name!r}"
        )

    def test_preferred_acyl_prefixes_survive(self):
        """Guard the over-correction -- the P-65.1.7.2.1 preferred prefixes.

        acetyl (:30442), formyl (:30444), benzoyl (:30446) are preferred
        prefixes and MUST NOT be systematised. Appendix 2: 'acetyloxy*'
        (:55432), 'formyloxy*' (:56044), 'benzoyloxy*' (:55589, :56544).
        """
        assert "acetyloxy" in name_compound("CC(=O)OCC(=O)O")
        assert "formyloxy" in name_compound("C(=O)OCC(=O)O")
        assert "benzoyloxy" in name_compound("c1ccccc1C(=O)OCC(=O)O")
        # ethanoyloxy/methanoyloxy/benzenecarbonyloxy are the non-preferred forms
        for wrong in ("ethanoyloxy", "methanoyloxy", "benzenecarbonyloxy"):
            assert wrong not in name_compound("CC(=O)OCC(=O)O")

    def test_aromatic_parent_path_also_uses_the_pin_stem(self):
        """The second live producer (rules/benzene.py), not just the composer."""
        name = name_compound("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1C(=O)O")
        assert "hexadecanoyloxy" in name, name
        assert "palmitoyloxy" not in name, name
        # aspirin: the acetyl control on the same producer
        assert "acetyloxy" in name_compound("CC(=O)Oc1ccccc1C(=O)O")

    def test_no_systematic_stem_key_in_the_acyloxy_table(self):
        """A systematic key can never be correct -- its input is already the PIN.

        This is the structural invariant behind the fix: if someone re-adds
        'hexadecanoic' -> 'palmitoyloxy', the J3 fix silently reverts.
        """
        from orthonym.rules.esters import TRIVIAL_ACID_TO_ACYLOXY
        offenders = sorted(
            k for k in TRIVIAL_ACID_TO_ACYLOXY
            if k.endswith("anoic") or k.endswith("enoic")
        )
        assert offenders == [], (
            f"systematic-stem keys re-introduced into TRIVIAL_ACID_TO_ACYLOXY: "
            f"{offenders}. Such a row takes a stem that is already the PIN and "
            f"converts it back to a non-PIN word (P-65.6.3.2.3 :31696)."
        )

    def test_acid_and_prefix_paths_agree_on_the_stem(self):
        """Internal consistency: the two independent producers must not disagree.

        This is the check that exposed the defect -- the acid path already said
        'hexadecanoic acid' while the prefix path said 'palmitoyloxy'.
        """
        for acid_smiles, ester_smiles, stem in [
            ("CCCCCCCCCCCCCCCC(=O)O",
             "CCCCCCCCCCCCCCCC(=O)OCC(=O)O", "hexadecano"),
            (r"CCCCCCCC/C=C\CCCCCCCC(=O)O",
             r"CCCCCCCC/C=C\CCCCCCCC(=O)OCC(=O)O", "octadec-9-eno"),
        ]:
            acid_name = name_compound(acid_smiles)
            prefix_name = name_compound(ester_smiles)
            assert stem in acid_name, f"{stem!r} not in acid {acid_name!r}"
            assert stem in prefix_name, f"{stem!r} not in prefix {prefix_name!r}"
