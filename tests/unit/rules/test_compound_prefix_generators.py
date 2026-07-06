"""Unit tests for compound prefix generators (sulfoxide, sulfone, thioether)
and static PREFIX_FORMS entries (acid halides, phosphonooxy).

Phase 80, Plan 01: Fill missing PREFIX_FORMS entries and add dynamic
compound prefix generation for S-bearing functional groups.
"""

import pytest
from rdkit import Chem

from orthonym.rules.polyfunctional import get_fg_prefix_form
from orthonym.rules.seniority import PREFIX_FORMS
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.namer import name_compound


# ============================================================================
# Helpers
# ============================================================================

def _get_fg_prefix(smiles, fg_name, chain_indices=None):
    """Helper: detect FG in a SMILES, return the prefix from get_fg_prefix_form."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    fgs = detect_functional_groups(mol)
    matches = fgs.get(fg_name, [])
    if not matches:
        return None
    if chain_indices is None:
        # Use atoms 0..3 as a rough principal chain (acid end)
        chain_indices = list(range(min(4, mol.GetNumAtoms())))
    return get_fg_prefix_form(fg_name, mol, matches[0], chain_indices)


# ============================================================================
# TestSulfinylPrefixGenerator
# ============================================================================

class TestSulfinylPrefixGenerator:
    """Tests for _get_sulfinyl_prefix (sulfoxide as non-principal group)."""

    @pytest.mark.unit
    def test_methylsulfinyl_on_acid(self):
        """Methyl sulfoxide on ethanoic acid -> methanesulfinyl (Wave2 T3b PIN acid-stem)."""
        prefix = _get_fg_prefix('OC(=O)CS(=O)C', 'sulfoxide')
        assert prefix is not None, "sulfoxide prefix should not be None"
        assert 'sulfinyl' in prefix, f"Expected 'sulfinyl' in '{prefix}'"
        assert prefix == 'methanesulfinyl', f"Expected 'methanesulfinyl', got '{prefix}'"

    @pytest.mark.unit
    def test_ethylsulfinyl_on_acid(self):
        """Ethyl sulfoxide on ethanoic acid -> ethanesulfinyl (Wave2 T3b PIN acid-stem)."""
        prefix = _get_fg_prefix('OC(=O)CS(=O)CC', 'sulfoxide')
        assert prefix is not None
        assert prefix == 'ethanesulfinyl', f"Expected 'ethanesulfinyl', got '{prefix}'"

    @pytest.mark.unit
    def test_sulfinyl_on_aromatic(self):
        """Methyl sulfoxide on benzoic acid -> prefix contains sulfinyl."""
        prefix = _get_fg_prefix('OC(=O)c1ccc(S(=O)C)cc1', 'sulfoxide')
        # May be None for ring-attached FGs (handled by benzene.py)
        # but the generator itself should produce a valid result if called directly
        mol = Chem.MolFromSmiles('OC(=O)c1ccc(S(=O)C)cc1')
        fgs = detect_functional_groups(mol)
        so_atoms = fgs.get('sulfoxide', [])
        if so_atoms:
            from orthonym.rules.polyfunctional import _get_sulfinyl_prefix
            # Use a chain that includes ring atoms
            result = _get_sulfinyl_prefix(mol, so_atoms[0], list(range(mol.GetNumAtoms())))
            assert result is not None, "sulfinyl prefix should work for aromatic case"
            assert 'sulfinyl' in result


# ============================================================================
# TestSulfonylPrefixGenerator
# ============================================================================

class TestSulfonylPrefixGenerator:
    """Tests for _get_sulfonyl_prefix (sulfone as non-principal group)."""

    @pytest.mark.unit
    def test_methylsulfonyl_on_acid(self):
        """Methyl sulfone on ethanoic acid -> methanesulfonyl (Wave2 T3b PIN acid-stem)."""
        prefix = _get_fg_prefix('OC(=O)CS(=O)(=O)C', 'sulfone')
        assert prefix is not None, "sulfone prefix should not be None"
        assert 'sulfonyl' in prefix, f"Expected 'sulfonyl' in '{prefix}'"
        assert prefix == 'methanesulfonyl', f"Expected 'methanesulfonyl', got '{prefix}'"

    @pytest.mark.unit
    def test_ethylsulfonyl_on_acid(self):
        """Ethyl sulfone on ethanoic acid -> ethanesulfonyl (Wave2 T3b PIN acid-stem)."""
        prefix = _get_fg_prefix('OC(=O)CS(=O)(=O)CC', 'sulfone')
        assert prefix is not None
        assert prefix == 'ethanesulfonyl', f"Expected 'ethanesulfonyl', got '{prefix}'"

    @pytest.mark.unit
    def test_sulfonyl_on_aromatic(self):
        """Methyl sulfone on benzoic acid -> prefix contains sulfonyl."""
        mol = Chem.MolFromSmiles('OC(=O)c1ccc(S(=O)(=O)C)cc1')
        fgs = detect_functional_groups(mol)
        so2_atoms = fgs.get('sulfone', [])
        if so2_atoms:
            from orthonym.rules.polyfunctional import _get_sulfonyl_prefix
            result = _get_sulfonyl_prefix(mol, so2_atoms[0], list(range(mol.GetNumAtoms())))
            assert result is not None
            assert 'sulfonyl' in result


# ============================================================================
# TestSulfanylPrefixGenerator
# ============================================================================

class TestSulfanylPrefixGenerator:
    """Tests for _get_sulfanyl_prefix (thioether as non-principal group)."""

    @pytest.mark.unit
    def test_methylsulfanyl_on_acid(self):
        """Methyl thioether on ethanoic acid -> methylsulfanyl."""
        prefix = _get_fg_prefix('OC(=O)CSC', 'thioether')
        assert prefix is not None, "thioether prefix should not be None"
        assert 'sulfanyl' in prefix, f"Expected 'sulfanyl' in '{prefix}'"
        assert prefix == 'methylsulfanyl', f"Expected 'methylsulfanyl', got '{prefix}'"

    @pytest.mark.unit
    def test_ethylsulfanyl_on_acid(self):
        """Ethyl thioether on ethanoic acid -> ethylsulfanyl."""
        prefix = _get_fg_prefix('OC(=O)CSCC', 'thioether')
        assert prefix is not None
        assert prefix == 'ethylsulfanyl', f"Expected 'ethylsulfanyl', got '{prefix}'"


# ============================================================================
# TestAcidHalidePrefixForms
# ============================================================================

class TestAcidHalidePrefixForms:
    """Tests for static PREFIX_FORMS entries for acid halides."""

    @pytest.mark.unit
    def test_acid_chloride_prefix(self):
        # ERRATA-09 (P-29.1.2): chlorocarbonyl -> carbonochloridoyl
        assert PREFIX_FORMS.get('acid_chloride') == 'carbonochloridoyl', \
            f"Expected 'carbonochloridoyl', got '{PREFIX_FORMS.get('acid_chloride')}'"

    @pytest.mark.unit
    def test_acid_bromide_prefix(self):
        assert PREFIX_FORMS.get('acid_bromide') == 'bromocarbonyl', \
            f"Expected 'bromocarbonyl', got '{PREFIX_FORMS.get('acid_bromide')}'"

    @pytest.mark.unit
    def test_acid_fluoride_prefix(self):
        assert PREFIX_FORMS.get('acid_fluoride') == 'fluorocarbonyl', \
            f"Expected 'fluorocarbonyl', got '{PREFIX_FORMS.get('acid_fluoride')}'"


# ============================================================================
# TestPhosphateMonoesterPrefixForm
# ============================================================================

class TestPhosphateMonoesterPrefixForm:
    """Test for phosphate monoester PREFIX_FORMS entry."""

    @pytest.mark.unit
    def test_phosphate_monoester_prefix(self):
        assert PREFIX_FORMS.get('phosphate_monoester') == 'phosphonooxy', \
            f"Expected 'phosphonooxy', got '{PREFIX_FORMS.get('phosphate_monoester')}'"


# ============================================================================
# TestNoRegressions
# ============================================================================

class TestNoRegressions:
    """Regression tests for existing polyfunctional naming."""

    @pytest.mark.unit
    def test_hydroxy_acid(self):
        """2-hydroxyethanoic acid must not regress."""
        result = name_compound('OCC(=O)O')
        assert result == '2-hydroxyethanoic acid', f"Got '{result}'"

    @pytest.mark.unit
    def test_keto_acid(self):
        """2-oxopropanoic acid must not regress."""
        result = name_compound('CC(=O)C(=O)O')
        assert result == '2-oxopropanoic acid', f"Got '{result}'"

    @pytest.mark.unit
    def test_hydroxy_propanoic_acid(self):
        """3-hydroxypropanoic acid must not regress."""
        result = name_compound('OCCC(=O)O')
        assert result == '3-hydroxypropanoic acid', f"Got '{result}'"

    @pytest.mark.unit
    def test_methoxy_acid(self):
        """3-methoxypropanoic acid must not regress."""
        result = name_compound('COCCC(=O)O')
        assert result == '3-methoxypropanoic acid', f"Got '{result}'"

    @pytest.mark.unit
    def test_dimethyl_sulfoxide_substitutive_pin(self):
        """Wave2 T3b (P-63.6): the PIN is substitutive '(methanesulfinyl)methane'
        (BB 46154 verbatim); 'dimethyl sulfoxide' demoted to --trivial."""
        result = name_compound('CS(=O)C')
        assert result == '(methanesulfinyl)methane', f"Got '{result}'"

    @pytest.mark.unit
    def test_dimethyl_sulfone_substitutive_pin(self):
        """Wave2 T3b (P-63.6): PIN '(methanesulfonyl)methane' (parallel to BB
        28115 '(ethanesulfonyl)ethane'); functional class demoted to --trivial."""
        result = name_compound('CS(=O)(=O)C')
        assert result == '(methanesulfonyl)methane', f"Got '{result}'"

    @pytest.mark.unit
    def test_sulfinyl_prefix_in_name(self):
        """Sulfoxide as non-principal should produce sulfinyl prefix in name."""
        result = name_compound('OC(=O)CS(=O)C')
        assert 'sulfinyl' in result, f"Expected 'sulfinyl' in '{result}'"
        assert 'sulfanyl' not in result, f"Should not have 'sulfanyl' in '{result}' (double-naming bug)"

    @pytest.mark.unit
    def test_sulfonyl_prefix_in_name(self):
        """Sulfone as non-principal should produce sulfonyl prefix in name."""
        result = name_compound('OC(=O)CS(=O)(=O)C')
        assert 'sulfonyl' in result, f"Expected 'sulfonyl' in '{result}'"
        assert 'sulfanyl' not in result, f"Should not have 'sulfanyl' in '{result}' (double-naming bug)"

    @pytest.mark.unit
    def test_sulfanyl_prefix_in_name(self):
        """Thioether as non-principal should produce sulfanyl prefix in name."""
        result = name_compound('OC(=O)CSC')
        assert 'sulfanyl' in result, f"Expected 'sulfanyl' in '{result}'"

    @pytest.mark.unit
    def test_no_double_prefix_sulfoxide(self):
        """Sulfoxide compound should NOT have both sulfinyl and sulfanyl prefixes."""
        result = name_compound('OC(=O)CS(=O)C')
        # Should be "2-methylsulfinylethanoic acid", NOT "2-methylsulfanyl-2-methylsulfinylethanoic acid"
        assert result.count('methyl') <= 1 or 'dimethyl' in result, \
            f"Double-naming detected in '{result}'"
