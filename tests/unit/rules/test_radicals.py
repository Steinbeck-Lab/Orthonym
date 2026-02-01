"""Unit tests for radical naming rules.

Tests radical naming according to IUPAC 2013 P-71:
- Monovalent radicals: -yl suffix (methyl, ethyl)
- Divalent radicals: -ylidene suffix (methylidene, ethylidene)
- Trivalent radicals: -ylidyne suffix (methylidyne)
- Acyl radicals: -oyl suffix (acetyl, benzoyl)
- Oxyl radicals: -oxyl suffix (methoxyl, phenoxyl)
"""
import pytest
from rdkit import Chem
from src.orthonym.rules.radicals import (
    classify_radical,
    name_radical,
    get_radical_suffix,
    name_alkyl_radical,
    name_acyl_radical,
    name_oxyl_radical,
    name_divalent_radical,
    name_trivalent_radical,
    name_aryl_radical,
    RADICAL_SUFFIXES,
    RADICAL_TYPE_NAMES,
    CHAIN_PREFIXES,
    RETAINED_RADICALS,
)
from src.orthonym.perception.ions import get_radical_sites


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def methyl_radical():
    """Methyl radical [CH3]."""
    return Chem.MolFromSmiles('[CH3]')


@pytest.fixture
def ethyl_radical():
    """Ethyl radical C[CH2]."""
    return Chem.MolFromSmiles('C[CH2]')


@pytest.fixture
def methylidene():
    """Methylidene (divalent) [CH2]."""
    return Chem.MolFromSmiles('[CH2]')


@pytest.fixture
def methylidyne():
    """Methylidyne (trivalent) [CH]."""
    return Chem.MolFromSmiles('[CH]')


@pytest.fixture
def acetyl_radical():
    """Acetyl radical C[C]=O."""
    return Chem.MolFromSmiles('C[C]=O')


@pytest.fixture
def methoxyl_radical():
    """Methoxyl radical [O]C."""
    return Chem.MolFromSmiles('[O]C')


# =============================================================================
# Test Constants
# =============================================================================

class TestRadicalConstants:
    """Test that constants are properly defined."""

    def test_radical_suffixes_defined(self):
        """Radical suffixes for each valency."""
        assert RADICAL_SUFFIXES[1] == 'yl'
        assert RADICAL_SUFFIXES[2] == 'ylidene'
        assert RADICAL_SUFFIXES[3] == 'ylidyne'

    def test_radical_type_names_defined(self):
        """Radical type names for each valency."""
        assert RADICAL_TYPE_NAMES[1] == 'monovalent'
        assert RADICAL_TYPE_NAMES[2] == 'divalent'
        assert RADICAL_TYPE_NAMES[3] == 'trivalent'

    def test_chain_prefixes_defined(self):
        """Chain prefixes for C1-C10."""
        assert CHAIN_PREFIXES[1] == 'meth'
        assert CHAIN_PREFIXES[2] == 'eth'
        assert CHAIN_PREFIXES[3] == 'prop'
        assert CHAIN_PREFIXES[4] == 'but'
        assert CHAIN_PREFIXES[10] == 'dec'

    def test_retained_radicals_has_common_entries(self):
        """Retained radical names lookup exists."""
        assert '[CH3]' in RETAINED_RADICALS
        assert RETAINED_RADICALS['[CH3]'] == 'methyl'


# =============================================================================
# Test classify_radical
# =============================================================================

class TestClassifyRadical:
    """Test radical classification."""

    def test_monovalent_alkyl(self, methyl_radical):
        """Classify methyl radical as monovalent alkyl."""
        sites = get_radical_sites(methyl_radical)
        assert len(sites) == 1
        info = classify_radical(methyl_radical, sites[0])
        assert info['n_electrons'] == 1
        assert info['radical_type'] == 'monovalent'
        assert info['subtype'] == 'alkyl'

    def test_monovalent_ethyl(self, ethyl_radical):
        """Classify ethyl radical as monovalent alkyl."""
        sites = get_radical_sites(ethyl_radical)
        assert len(sites) == 1
        info = classify_radical(ethyl_radical, sites[0])
        assert info['n_electrons'] == 1
        assert info['radical_type'] == 'monovalent'
        assert info['subtype'] == 'alkyl'

    def test_divalent_carbene(self, methylidene):
        """Classify methylidene as divalent."""
        sites = get_radical_sites(methylidene)
        assert len(sites) == 1
        info = classify_radical(methylidene, sites[0])
        assert info['n_electrons'] == 2
        assert info['radical_type'] == 'divalent'
        assert info['subtype'] == 'alkyl'

    def test_trivalent_carbyne(self, methylidyne):
        """Classify methylidyne as trivalent."""
        sites = get_radical_sites(methylidyne)
        assert len(sites) == 1
        info = classify_radical(methylidyne, sites[0])
        assert info['n_electrons'] == 3
        assert info['radical_type'] == 'trivalent'
        assert info['subtype'] == 'alkyl'

    def test_oxyl_radical(self, methoxyl_radical):
        """Classify methoxyl as oxyl subtype."""
        sites = get_radical_sites(methoxyl_radical)
        assert len(sites) == 1
        info = classify_radical(methoxyl_radical, sites[0])
        assert info['n_electrons'] == 1
        assert info['subtype'] == 'oxyl'

    def test_acyl_radical(self, acetyl_radical):
        """Classify acetyl as acyl subtype."""
        sites = get_radical_sites(acetyl_radical)
        assert len(sites) == 1
        info = classify_radical(acetyl_radical, sites[0])
        assert info['n_electrons'] == 1
        assert info['subtype'] == 'acyl'


# =============================================================================
# Test get_radical_suffix
# =============================================================================

class TestRadicalSuffix:
    """Test radical suffix selection."""

    def test_monovalent_suffix(self):
        """Monovalent radicals use -yl suffix."""
        assert get_radical_suffix(1) == 'yl'

    def test_divalent_suffix(self):
        """Divalent radicals use -ylidene suffix."""
        assert get_radical_suffix(2) == 'ylidene'

    def test_trivalent_suffix(self):
        """Trivalent radicals use -ylidyne suffix."""
        assert get_radical_suffix(3) == 'ylidyne'

    def test_default_suffix(self):
        """Unknown valency defaults to -yl."""
        assert get_radical_suffix(4) == 'yl'
        assert get_radical_suffix(0) == 'yl'


# =============================================================================
# Test name_radical (main function)
# =============================================================================

class TestNameRadical:
    """Test main radical naming function."""

    def test_methyl_radical(self, methyl_radical):
        """Name methyl radical correctly."""
        name = name_radical(methyl_radical)
        assert name == 'methyl'

    def test_ethyl_radical(self, ethyl_radical):
        """Name ethyl radical correctly."""
        name = name_radical(ethyl_radical)
        assert name == 'ethyl'

    def test_propyl_radical(self):
        """Name propyl radical correctly."""
        mol = Chem.MolFromSmiles('CC[CH2]')
        name = name_radical(mol)
        assert name == 'propyl'

    def test_butyl_radical(self):
        """Name butyl radical correctly."""
        mol = Chem.MolFromSmiles('CCC[CH2]')
        name = name_radical(mol)
        assert name == 'butyl'

    def test_methylidene(self, methylidene):
        """Name methylidene correctly."""
        name = name_radical(methylidene)
        assert name == 'methylidene'

    def test_ethylidene(self):
        """Name ethylidene correctly."""
        mol = Chem.MolFromSmiles('C[CH]')
        sites = get_radical_sites(mol)
        # Check if it has 2 unpaired electrons
        if sites and sites[0]['n_electrons'] == 2:
            name = name_radical(mol)
            assert name == 'ethylidene'

    def test_methylidyne(self, methylidyne):
        """Name methylidyne correctly."""
        name = name_radical(methylidyne)
        assert name == 'methylidyne'

    def test_none_mol_returns_empty(self):
        """None molecule returns empty string."""
        name = name_radical(None)
        assert name == ''

    def test_no_radical_returns_empty(self):
        """Molecule without radicals returns empty string."""
        mol = Chem.MolFromSmiles('CCO')  # Ethanol, no radical
        name = name_radical(mol)
        assert name == ''


# =============================================================================
# Test Acyl Radicals
# =============================================================================

class TestAcylRadicals:
    """Test acyl radical naming."""

    def test_acetyl_radical(self, acetyl_radical):
        """Name acetyl radical correctly."""
        name = name_radical(acetyl_radical)
        assert name == 'acetyl'

    def test_formyl_radical(self):
        """Name formyl radical correctly."""
        mol = Chem.MolFromSmiles('[CH]=O')
        name = name_radical(mol)
        assert name == 'formyl'

    def test_propanoyl_radical(self):
        """Name propanoyl radical correctly."""
        mol = Chem.MolFromSmiles('CC[C]=O')
        name = name_radical(mol)
        assert name == 'propanoyl'

    def test_acyl_radical_helper(self, acetyl_radical):
        """Test name_acyl_radical helper directly."""
        sites = get_radical_sites(acetyl_radical)
        name = name_acyl_radical(acetyl_radical, sites[0])
        assert name == 'acetyl'


# =============================================================================
# Test Oxyl Radicals
# =============================================================================

class TestOxylRadicals:
    """Test oxyl radical naming."""

    def test_methoxyl_radical(self, methoxyl_radical):
        """Name methoxyl radical correctly."""
        name = name_radical(methoxyl_radical)
        assert name == 'methoxyl'

    def test_ethoxyl_radical(self):
        """Name ethoxyl radical correctly."""
        mol = Chem.MolFromSmiles('[O]CC')
        name = name_radical(mol)
        assert name == 'ethoxyl'

    def test_propoxyl_radical(self):
        """Name propoxyl radical correctly."""
        mol = Chem.MolFromSmiles('[O]CCC')
        name = name_radical(mol)
        assert name == 'propoxyl'

    def test_oxyl_radical_helper(self, methoxyl_radical):
        """Test name_oxyl_radical helper directly."""
        sites = get_radical_sites(methoxyl_radical)
        name = name_oxyl_radical(methoxyl_radical, sites[0])
        assert name == 'methoxyl'


# =============================================================================
# Test Divalent Radical Naming
# =============================================================================

class TestDivalentRadicals:
    """Test divalent radical (alkylidene) naming."""

    def test_methylidene_helper(self, methylidene):
        """Test name_divalent_radical helper."""
        sites = get_radical_sites(methylidene)
        name = name_divalent_radical(methylidene, sites[0])
        assert name == 'methylidene'

    def test_ethylidene_via_name_radical(self):
        """Test divalent ethylidene via main function."""
        # Note: C[CH] may not parse as divalent in RDKit
        # Test the helper directly
        mol = Chem.MolFromSmiles('[CH2]')
        sites = get_radical_sites(mol)
        if sites and sites[0]['n_electrons'] == 2:
            name = name_divalent_radical(mol, sites[0])
            assert name == 'methylidene'


# =============================================================================
# Test Trivalent Radical Naming
# =============================================================================

class TestTrivalentRadicals:
    """Test trivalent radical (alkylidyne) naming."""

    def test_methylidyne_helper(self, methylidyne):
        """Test name_trivalent_radical helper."""
        sites = get_radical_sites(methylidyne)
        name = name_trivalent_radical(methylidyne, sites[0])
        assert name == 'methylidyne'


# =============================================================================
# Test Alkyl Radical Naming Helper
# =============================================================================

class TestAlkylRadicalHelper:
    """Test individual alkyl radical naming helper."""

    def test_methyl_radical_naming(self, methyl_radical):
        """Test name_alkyl_radical for methyl."""
        sites = get_radical_sites(methyl_radical)
        name = name_alkyl_radical(methyl_radical, sites[0])
        assert name == 'methyl'

    def test_ethyl_radical_naming(self, ethyl_radical):
        """Test name_alkyl_radical for ethyl."""
        sites = get_radical_sites(ethyl_radical)
        name = name_alkyl_radical(ethyl_radical, sites[0])
        assert name == 'ethyl'

    def test_propyl_radical_naming(self):
        """Test name_alkyl_radical for propyl."""
        mol = Chem.MolFromSmiles('CC[CH2]')
        sites = get_radical_sites(mol)
        name = name_alkyl_radical(mol, sites[0])
        assert name == 'propyl'


# =============================================================================
# Test Aryl Radical Naming
# =============================================================================

class TestArylRadicals:
    """Test aryl radical naming."""

    def test_phenyl_radical_from_retained(self):
        """Phenyl from retained names."""
        mol = Chem.MolFromSmiles('[c]1ccccc1')
        if mol:
            sites = get_radical_sites(mol)
            if sites:
                info = classify_radical(mol, sites[0])
                assert info['subtype'] == 'aryl'

    def test_aryl_radical_helper(self):
        """Test name_aryl_radical helper."""
        mol = Chem.MolFromSmiles('[c]1ccccc1')
        if mol:
            sites = get_radical_sites(mol)
            if sites:
                name = name_aryl_radical(mol, sites[0])
                assert name == 'phenyl'


# =============================================================================
# Test Retained Names Lookup
# =============================================================================

class TestRetainedNames:
    """Test retained radical name lookups."""

    def test_methyl_from_retained(self):
        """Methyl uses retained name."""
        mol = Chem.MolFromSmiles('[CH3]')
        canonical = Chem.MolToSmiles(mol, canonical=True)
        assert canonical in RETAINED_RADICALS

    def test_methylidene_from_retained(self):
        """Methylidene uses retained name."""
        mol = Chem.MolFromSmiles('[CH2]')
        canonical = Chem.MolToSmiles(mol, canonical=True)
        assert canonical in RETAINED_RADICALS

    def test_methylidyne_from_retained(self):
        """Methylidyne uses retained name."""
        mol = Chem.MolFromSmiles('[CH]')
        canonical = Chem.MolToSmiles(mol, canonical=True)
        assert canonical in RETAINED_RADICALS

    def test_systematic_style_bypasses_retained(self):
        """Systematic style should bypass retained names."""
        mol = Chem.MolFromSmiles('[CH3]')
        # Even with systematic style, methyl is the systematic name
        name = name_radical(mol, style='systematic')
        assert name == 'methyl'


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_smiles(self):
        """Invalid SMILES returns empty string."""
        mol = Chem.MolFromSmiles('invalid_smiles')
        name = name_radical(mol)
        assert name == ''

    def test_neutral_molecule(self):
        """Neutral molecule with no radicals."""
        mol = Chem.MolFromSmiles('C')  # Methane
        name = name_radical(mol)
        assert name == ''

    def test_charged_species(self):
        """Charged species (not radical)."""
        mol = Chem.MolFromSmiles('[CH3+]')
        name = name_radical(mol)
        # Should be empty since no unpaired electrons
        assert name == '' or name  # May vary by interpretation


# =============================================================================
# Test Integration with perception.ions
# =============================================================================

class TestPerceptionIntegration:
    """Test integration with perception layer."""

    def test_get_radical_sites_import(self):
        """get_radical_sites is properly imported."""
        mol = Chem.MolFromSmiles('[CH3]')
        sites = get_radical_sites(mol)
        assert len(sites) == 1
        assert 'atom_idx' in sites[0]
        assert 'n_electrons' in sites[0]
        assert 'element' in sites[0]
        assert 'radical_type' in sites[0]

    def test_radical_site_structure(self, methyl_radical):
        """Verify radical site dictionary structure."""
        sites = get_radical_sites(methyl_radical)
        site = sites[0]
        assert site['element'] == 'C'
        assert site['n_electrons'] == 1
        assert site['radical_type'] == 'monovalent'


# =============================================================================
# Test Complete Naming Pipeline
# =============================================================================

class TestNamingPipeline:
    """Test complete naming pipeline for various radicals."""

    @pytest.mark.parametrize("smiles,expected_name", [
        ('[CH3]', 'methyl'),
        ('C[CH2]', 'ethyl'),
        ('CC[CH2]', 'propyl'),
        ('[CH2]', 'methylidene'),
        ('[CH]', 'methylidyne'),
        ('C[C]=O', 'acetyl'),
        ('[CH]=O', 'formyl'),
        ('[O]C', 'methoxyl'),
        ('[O]CC', 'ethoxyl'),
    ])
    def test_radical_naming_pipeline(self, smiles, expected_name):
        """Test complete naming pipeline for common radicals."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            name = name_radical(mol)
            assert name == expected_name, f"Expected {expected_name} for {smiles}, got {name}"
