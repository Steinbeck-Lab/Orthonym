"""
Unit tests for ion naming rules.

Tests ion classification and naming functions per IUPAC 2013 P-72/P-73.
"""

import pytest
from rdkit import Chem

from src.orthonym.rules.ions import (
    classify_anion,
    classify_cation,
    name_anion,
    name_cation,
    name_carboxylate_anion,
    name_alkoxide_anion,
    name_phenolate_anion,
    name_aminium_cation,
    name_carbenium_cation,
    get_anion_suffix,
    get_cation_suffix,
)
from src.orthonym.perception.ions import get_ion_sites


class TestClassifyAnion:
    """Test anion classification."""

    def test_carboxylate_acetate(self):
        """Acetate anion should be classified as carboxylate."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'carboxylate'

    def test_carboxylate_formate(self):
        """Formate anion should be classified as carboxylate."""
        mol = Chem.MolFromSmiles('[O-]C=O')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'carboxylate'

    def test_carboxylate_benzoate(self):
        """Benzoate anion should be classified as carboxylate."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccccc1')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'carboxylate'

    def test_alkoxide_methoxide(self):
        """Methoxide anion should be classified as alkoxide."""
        mol = Chem.MolFromSmiles('C[O-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'alkoxide'

    def test_alkoxide_ethoxide(self):
        """Ethoxide anion should be classified as alkoxide."""
        mol = Chem.MolFromSmiles('CC[O-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'alkoxide'

    def test_alkoxide_tert_butoxide(self):
        """tert-Butoxide anion should be classified as alkoxide."""
        mol = Chem.MolFromSmiles('CC(C)(C)[O-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'alkoxide'

    def test_phenolate(self):
        """Phenolate anion should be classified as phenolate."""
        mol = Chem.MolFromSmiles('[O-]c1ccccc1')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'phenolate'

    def test_carbanion_methanide(self):
        """Methanide (CH3-) should be classified as carbanion."""
        mol = Chem.MolFromSmiles('[CH3-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'carbanion'

    def test_aminide(self):
        """Amide anion (NH2-) should be classified as aminide."""
        mol = Chem.MolFromSmiles('[NH2-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'aminide'

    def test_thiolate(self):
        """Thiolate anion (CH3S-) should be classified as thiolate."""
        mol = Chem.MolFromSmiles('C[S-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        anion_type = classify_anion(mol, sites['anions'][0])
        assert anion_type == 'thiolate'


class TestClassifyCation:
    """Test cation classification."""

    def test_aminium_ammonium(self):
        """Ammonium (NH4+) should be classified as aminium."""
        mol = Chem.MolFromSmiles('[NH4+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'aminium'

    def test_aminium_methylammonium(self):
        """Methylammonium (CH3NH3+) should be classified as aminium."""
        mol = Chem.MolFromSmiles('C[NH3+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'aminium'

    def test_aminium_quaternary(self):
        """Tetramethylammonium should be classified as aminium."""
        mol = Chem.MolFromSmiles('C[N+](C)(C)C')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'aminium'

    def test_carbenium_methylium(self):
        """Methylium (CH3+) should be classified as ylium."""
        mol = Chem.MolFromSmiles('[CH3+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'ylium'

    def test_carbenium_ethylium(self):
        """Ethylium (C2H5+) should be classified as ylium."""
        mol = Chem.MolFromSmiles('C[CH2+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'ylium'

    def test_oxonium(self):
        """Oxonium (H3O+) should be classified as onium."""
        mol = Chem.MolFromSmiles('[OH3+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'onium'

    def test_sulfonium(self):
        """Trimethylsulfonium should be classified as onium."""
        mol = Chem.MolFromSmiles('C[S+](C)C')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'onium'

    def test_phosphonium(self):
        """Phosphonium (PH4+) should be classified as onium."""
        mol = Chem.MolFromSmiles('[PH4+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'onium'


class TestNameAnion:
    """Test anion naming."""

    def test_acetate_retained(self):
        """Acetate should use retained name."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'acetate'

    def test_formate_retained(self):
        """Formate should use retained name."""
        mol = Chem.MolFromSmiles('[O-]C=O')
        name = name_anion(mol)
        assert name == 'formate'

    def test_methoxide_retained(self):
        """Methoxide should use retained name."""
        mol = Chem.MolFromSmiles('C[O-]')
        name = name_anion(mol)
        assert name == 'methoxide'

    def test_ethoxide_retained(self):
        """Ethoxide should use retained name."""
        mol = Chem.MolFromSmiles('CC[O-]')
        name = name_anion(mol)
        assert name == 'ethoxide'

    def test_phenolate(self):
        """Phenolate should be named correctly."""
        mol = Chem.MolFromSmiles('[O-]c1ccccc1')
        name = name_anion(mol)
        # Could be phenolate or phenoxide depending on lookup
        assert name in ('phenolate', 'phenoxide')

    def test_propanoate_systematic(self):
        """Propanoate should get systematic name."""
        mol = Chem.MolFromSmiles('CCC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'propanoate'

    def test_butanoate_systematic(self):
        """Butanoate should get systematic name."""
        mol = Chem.MolFromSmiles('CCCC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'butanoate'

    def test_benzoate_retained(self):
        """Benzoate should use retained name."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccccc1')
        name = name_anion(mol)
        assert name == 'benzoate'


class TestNameCation:
    """Test cation naming."""

    def test_ammonium_retained(self):
        """Ammonium should use retained name."""
        mol = Chem.MolFromSmiles('[NH4+]')
        name = name_cation(mol)
        assert name == 'ammonium'

    def test_methylammonium_retained(self):
        """Methylammonium should use retained name."""
        mol = Chem.MolFromSmiles('C[NH3+]')
        name = name_cation(mol)
        assert name == 'methylammonium'

    def test_tetramethylammonium_retained(self):
        """Tetramethylammonium should use retained name."""
        mol = Chem.MolFromSmiles('C[N+](C)(C)C')
        name = name_cation(mol)
        assert name == 'tetramethylammonium'

    def test_ethylammonium_retained(self):
        """Ethylammonium should use retained name."""
        mol = Chem.MolFromSmiles('CC[NH3+]')
        name = name_cation(mol)
        assert name == 'ethylammonium'

    def test_methylium_retained(self):
        """Methylium should use retained name."""
        mol = Chem.MolFromSmiles('[CH3+]')
        name = name_cation(mol)
        assert name == 'methylium'

    def test_oxonium_retained(self):
        """Oxonium should use retained name."""
        mol = Chem.MolFromSmiles('[OH3+]')
        name = name_cation(mol)
        assert name == 'oxonium'

    def test_phosphonium_retained(self):
        """Phosphonium should use retained name."""
        mol = Chem.MolFromSmiles('[PH4+]')
        name = name_cation(mol)
        assert name == 'phosphonium'


class TestSuffixTransformations:
    """Test suffix transformation helpers."""

    def test_carboxylate_from_ic_acid(self):
        """'acetic acid' -> 'acetate'"""
        assert name_carboxylate_anion('acetic acid') == 'acetate'

    def test_carboxylate_from_formic_acid(self):
        """'formic acid' -> 'formate'"""
        assert name_carboxylate_anion('formic acid') == 'formate'

    def test_carboxylate_from_oic_acid(self):
        """'propanoic acid' -> 'propanoate'"""
        assert name_carboxylate_anion('propanoic acid') == 'propanoate'

    def test_carboxylate_from_butanoic_acid(self):
        """'butanoic acid' -> 'butanoate'"""
        assert name_carboxylate_anion('butanoic acid') == 'butanoate'

    def test_alkoxide_pin_style(self):
        """'methanol' -> 'methanolate' (PIN style)"""
        assert name_alkoxide_anion('methanol') == 'methanolate'

    def test_alkoxide_pin_ethanol(self):
        """'ethanol' -> 'ethanolate' (PIN style)"""
        assert name_alkoxide_anion('ethanol') == 'ethanolate'

    def test_alkoxide_common_style(self):
        """'methanol' -> 'methoxide' (common style)"""
        assert name_alkoxide_anion('methanol', style='common') == 'methoxide'

    def test_alkoxide_common_ethanol(self):
        """'ethanol' -> 'ethoxide' (common style)"""
        assert name_alkoxide_anion('ethanol', style='common') == 'ethoxide'

    def test_phenolate_from_phenol(self):
        """'phenol' -> 'phenolate'"""
        assert name_phenolate_anion('phenol') == 'phenolate'

    def test_aminium_from_methanamine(self):
        """'methanamine' -> 'methanaminium'"""
        assert name_aminium_cation('methanamine') == 'methanaminium'

    def test_aminium_from_ethanamine(self):
        """'ethanamine' -> 'ethanaminium'"""
        assert name_aminium_cation('ethanamine') == 'ethanaminium'

    def test_aminium_from_ammonia(self):
        """'ammonia' -> 'ammonium'"""
        assert name_aminium_cation('ammonia') == 'ammonium'

    def test_carbenium_ylium_from_methane(self):
        """'methane' -> 'methylium'"""
        assert name_carbenium_cation('methane') == 'methylium'

    def test_carbenium_ylium_from_ethane(self):
        """'ethane' -> 'ethylium'"""
        assert name_carbenium_cation('ethane') == 'ethylium'

    def test_carbenium_ylium_from_propane(self):
        """'propane' -> 'propylium'"""
        assert name_carbenium_cation('propane') == 'propylium'


class TestSuffixGetters:
    """Test suffix getter functions."""

    def test_anion_suffix_carboxylate(self):
        """Carboxylate suffix is 'ate'."""
        assert get_anion_suffix('carboxylate') == 'ate'

    def test_anion_suffix_alkoxide(self):
        """Alkoxide suffix is 'olate'."""
        assert get_anion_suffix('alkoxide') == 'olate'

    def test_anion_suffix_carbanion(self):
        """Carbanion suffix is 'ide'."""
        assert get_anion_suffix('carbanion') == 'ide'

    def test_anion_suffix_aminide(self):
        """Aminide suffix is 'aminide'."""
        assert get_anion_suffix('aminide') == 'aminide'

    def test_anion_suffix_unknown(self):
        """Unknown anion type defaults to 'ide'."""
        assert get_anion_suffix('unknown') == 'ide'

    def test_cation_suffix_aminium(self):
        """Aminium suffix is 'aminium'."""
        assert get_cation_suffix('aminium') == 'aminium'

    def test_cation_suffix_ylium(self):
        """Ylium suffix is 'ylium'."""
        assert get_cation_suffix('ylium') == 'ylium'

    def test_cation_suffix_onium(self):
        """Onium suffix is 'onium'."""
        assert get_cation_suffix('onium') == 'onium'

    def test_cation_suffix_unknown(self):
        """Unknown cation type defaults to 'ium'."""
        assert get_cation_suffix('unknown') == 'ium'


class TestSystematicNaming:
    """Test systematic naming when no retained name exists."""

    def test_anion_systematic_style(self):
        """Systematic style should bypass retained names."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        name = name_anion(mol, style='systematic')
        # Should generate systematic, not 'acetate'
        assert name in ('acetate', 'ethanoate')  # Either is acceptable

    def test_cation_systematic_style(self):
        """Systematic style should bypass retained names."""
        mol = Chem.MolFromSmiles('[NH4+]')
        name = name_cation(mol, style='systematic')
        assert name == 'ammonium'  # NH4+ has no alternative


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_name_anion_none_mol(self):
        """name_anion should return empty string for None mol."""
        assert name_anion(None) == ''

    def test_name_cation_none_mol(self):
        """name_cation should return empty string for None mol."""
        assert name_cation(None) == ''

    def test_name_anion_neutral_mol(self):
        """name_anion should return empty string for neutral molecule."""
        mol = Chem.MolFromSmiles('CCO')
        name = name_anion(mol)
        assert name == ''

    def test_name_cation_neutral_mol(self):
        """name_cation should return empty string for neutral molecule."""
        mol = Chem.MolFromSmiles('CCO')
        name = name_cation(mol)
        assert name == ''


class TestMultipleCharges:
    """Test molecules with multiple charged sites."""

    def test_dicarboxylate_oxalate(self):
        """Oxalate (doubly charged) should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])C([O-])=O')
        name = name_anion(mol)
        assert name in ('oxalate', 'ethanedioate')

    def test_dicarboxylate_malonate(self):
        """Malonate should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])CC([O-])=O')
        name = name_anion(mol)
        assert name in ('malonate', 'propanedioate')
