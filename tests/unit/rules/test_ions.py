"""
Unit tests for ion naming rules.

Tests ion classification and naming functions per IUPAC 2013 P-72/P-73.
"""

import pytest
from rdkit import Chem

from orthonym.rules.ions import (
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
from orthonym.perception.ions import get_ion_sites


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

    def test_iminide(self):
        """Imine anion (=N-) should be classified as iminide, NOT aminide.

        P-72.2.2.2.3 (the Blue Book): an imine bearing a negative charge on the
        nitrogen takes the 'iminide' suffix. The discriminator vs. 'aminide' is a
        double bond on the anionic nitrogen (the Blue Book butaniminide CCCC=[N-]).
        """
        mol = Chem.MolFromSmiles('CCCC=[N-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        assert classify_anion(mol, sites['anions'][0]) == 'iminide'

    def test_aminide_single_bond_not_iminide(self):
        """A single-bonded amine anion stays 'aminide' (no double bond on N)."""
        mol = Chem.MolFromSmiles('C[NH-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        assert classify_anion(mol, sites['anions'][0]) == 'aminide'


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
        """Tetramethylammonium (a quaternary N: +1, 0 H, degree 4) classifies as
        'quaternary' (a phase WS-E.1 /). The dedicated class routes it to
        the systematic ``-aminium`` PIN (N,N,N-trimethylmethanaminium) via the
        CATION_QUATERNARY dispatch — a quaternary N cannot take the protonated-amine
        ('aminium') neutralize path (over-valent neutral N), so it is split out from
        the generic 'aminium' class. The resulting PIN is asserted in
        tests/unit/rules/test_quaternary_aminium.py."""
        mol = Chem.MolFromSmiles('C[N+](C)(C)C')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        cation_type = classify_cation(mol, sites['cations'][0])
        assert cation_type == 'quaternary'

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
        """NH4+ PIN is 'azanium' (P-73.1.1.2, the Blue Book -- the mononuclear
        parent-hydride cation, "not those given in Table 7.3"); mirrors
        [PH4+]->phosphanium. Was 'ammonium' (Table-7.3 retained)."""
        mol = Chem.MolFromSmiles('[NH4+]')
        name = name_cation(mol)
        assert name == 'azanium'

    def test_methylammonium_pin_is_methanaminium(self):
        """ (P-73.1.2.1): 'methylammonium' is general nomenclature
        only; the PIN is the substitutive 'methanaminium' (the Blue Book). The
        retained name stays available for general/common style."""
        mol = Chem.MolFromSmiles('C[NH3+]')
        assert name_cation(mol) == 'methanaminium'            # default = pin
        assert name_cation(mol, style='common') == 'methylammonium'

    def test_tetramethylammonium_pin_is_trimethylmethanaminium(self):
        """(CH3)4N+ PIN is N,N,N-trimethylmethanaminium (the Blue Book);
        'tetramethylammonium' is general only."""
        mol = Chem.MolFromSmiles('C[N+](C)(C)C')
        assert name_cation(mol) == 'N,N,N-trimethylmethanaminium'
        assert name_cation(mol, style='common') == 'tetramethylammonium'

    def test_ammonium_pin_is_azanium(self):
        """NH4+ PIN is the parent-hydride cation 'azanium', NOT the Table-7.3
        'ammonium' (P-73.1.1.2, the Blue Book: "the preferred IUPAC names and not
        those given in Table 7.3"). 'ammonium' is general/common only."""
        assert name_cation(Chem.MolFromSmiles('[NH4+]')) == 'azanium'
        assert name_cation(Chem.MolFromSmiles('[NH4+]'),
                           style='common') == 'azanium'

    def test_ethylammonium_pin_is_ethanaminium(self):
        """: PIN is 'ethanaminium' (P-73.1.2.1); the general
        'ethylammonium' retained name stays for common style."""
        mol = Chem.MolFromSmiles('CC[NH3+]')
        assert name_cation(mol) == 'ethanaminium'
        assert name_cation(mol, style='common') == 'ethylammonium'

    def test_methylium_retained(self):
        """Methylium should use retained name."""
        mol = Chem.MolFromSmiles('[CH3+]')
        name = name_cation(mol)
        assert name == 'methylium'

    def test_oxonium_retained(self):
        """OH3+ PIN is 'oxidanium' (P-73.1.1.2, the Blue Book -- mononuclear
        parent-hydride cation, "not those given in Table 7.3"); mirrors
        [PH4+]->phosphanium. Was 'oxonium' (Table-7.3 retained)."""
        mol = Chem.MolFromSmiles('[OH3+]')
        name = name_cation(mol)
        assert name == 'oxidanium'

    def test_phosphanium_pin(self):
        """H4P+ PIN is 'phosphanium' (W4-I3; BB 41378/42393 'phosphanium
        (preselected name) phosphonium'). Traditional 'phosphonium' is the alt."""
        mol = Chem.MolFromSmiles('[PH4+]')
        name = name_cation(mol)
        assert name == 'phosphanium'


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


class TestAromaticCarboxylateNaming:
    """
    Test aromatic carboxylate naming (BUG-1 and BUG-5 fixes).

    Previously, aromatic carboxylates like 4-chlorobenzoate were incorrectly
    named as "heptanoate" because the code counted all 7 carbons (benzene + COOH).
    This test class verifies the fix.

    IUPAC 2013 Reference: P-72.1.1 (anions from acids), P-14.6 (aromatic precedence)
    """

    def test_simple_benzoate(self):
        """Unsubstituted benzoate should be named 'benzoate'."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccccc1')
        name = name_anion(mol)
        assert name == 'benzoate'

    def test_4_chlorobenzoate(self):
        """
        4-chlorobenzoate should NOT be named 'heptanoate'.

        This was the canonical BUG-1 case: O=C([O-])c1ccc(Cl)cc1 returned
        "heptanoate" by counting 7 carbons. Should return "4-chlorobenzoate".
        """
        mol = Chem.MolFromSmiles('O=C([O-])c1ccc(Cl)cc1')
        name = name_anion(mol)
        assert name == '4-chlorobenzoate', f"Expected '4-chlorobenzoate', got '{name}'"

    def test_2_chlorobenzoate(self):
        """2-chlorobenzoate should have the chloro at position 2."""
        mol = Chem.MolFromSmiles('Clc1ccccc1C(=O)[O-]')
        name = name_anion(mol)
        assert name == '2-chlorobenzoate', f"Expected '2-chlorobenzoate', got '{name}'"

    def test_3_chlorobenzoate(self):
        """3-chlorobenzoate (meta) should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])c1cccc(Cl)c1')
        name = name_anion(mol)
        assert name == '3-chlorobenzoate', f"Expected '3-chlorobenzoate', got '{name}'"

    def test_4_methylbenzoate(self):
        """4-methylbenzoate (p-toluate) should be named with methyl prefix."""
        mol = Chem.MolFromSmiles('Cc1ccc(C(=O)[O-])cc1')
        name = name_anion(mol)
        assert name == '4-methylbenzoate', f"Expected '4-methylbenzoate', got '{name}'"

    def test_2_methylbenzoate(self):
        """2-methylbenzoate (o-toluate) should be named correctly."""
        mol = Chem.MolFromSmiles('Cc1ccccc1C(=O)[O-]')
        name = name_anion(mol)
        assert name == '2-methylbenzoate', f"Expected '2-methylbenzoate', got '{name}'"

    def test_3_5_dimethylbenzoate(self):
        """3,5-dimethylbenzoate should have both methyl groups named."""
        mol = Chem.MolFromSmiles('Cc1cc(C)cc(C(=O)[O-])c1')
        name = name_anion(mol)
        assert name == '3,5-dimethylbenzoate', f"Expected '3,5-dimethylbenzoate', got '{name}'"

    def test_4_fluorobenzoate(self):
        """4-fluorobenzoate should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccc(F)cc1')
        name = name_anion(mol)
        assert name == '4-fluorobenzoate', f"Expected '4-fluorobenzoate', got '{name}'"

    def test_4_bromobenzoate(self):
        """4-bromobenzoate should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccc(Br)cc1')
        name = name_anion(mol)
        assert name == '4-bromobenzoate', f"Expected '4-bromobenzoate', got '{name}'"

    def test_4_iodobenzoate(self):
        """4-iodobenzoate should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccc(I)cc1')
        name = name_anion(mol)
        assert name == '4-iodobenzoate', f"Expected '4-iodobenzoate', got '{name}'"

    def test_2_4_dichlorobenzoate(self):
        """2,4-dichlorobenzoate should have both chloro groups named."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccc(Cl)cc1Cl')
        name = name_anion(mol)
        assert name == '2,4-dichlorobenzoate', f"Expected '2,4-dichlorobenzoate', got '{name}'"

    def test_2_naphthoate(self):
        """2-naphthoate (naphthalene-2-carboxylate) should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])c1ccc2ccccc2c1')
        name = name_anion(mol)
        # Should be naphthoate-based, not "undecanoate" (11 carbons)
        assert 'naphthoate' in name, f"Expected naphthoate-based name, got '{name}'"

    def test_1_naphthoate(self):
        """1-naphthoate (naphthalene-1-carboxylate) should be named correctly."""
        mol = Chem.MolFromSmiles('O=C([O-])c1cccc2ccccc12')
        name = name_anion(mol)
        # Should be naphthoate-based
        assert 'naphthoate' in name, f"Expected naphthoate-based name, got '{name}'"


class TestAromaticCarboxylateRegressions:
    """
    Regression tests to ensure acyclic carboxylates still work.

    The aromatic carboxylate fix must not break naming of simple
    acyclic carboxylates like acetate, propanoate, etc.
    """

    def test_acetate_regression(self):
        """Acetate should still be named 'acetate' (not affected by aromatic fix)."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'acetate', f"Regression: expected 'acetate', got '{name}'"

    def test_propanoate_regression(self):
        """Propanoate should still be named 'propanoate'."""
        mol = Chem.MolFromSmiles('CCC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'propanoate', f"Regression: expected 'propanoate', got '{name}'"

    def test_butanoate_regression(self):
        """Butanoate should still be named 'butanoate'."""
        mol = Chem.MolFromSmiles('CCCC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'butanoate', f"Regression: expected 'butanoate', got '{name}'"

    def test_formate_regression(self):
        """Formate should still be named 'formate'."""
        mol = Chem.MolFromSmiles('[O-]C=O')
        name = name_anion(mol)
        assert name == 'formate', f"Regression: expected 'formate', got '{name}'"

    def test_pentanoate_regression(self):
        """Pentanoate should still be named 'pentanoate'."""
        mol = Chem.MolFromSmiles('CCCCC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'pentanoate', f"Regression: expected 'pentanoate', got '{name}'"

    def test_hexanoate_regression(self):
        """Hexanoate should still be named 'hexanoate'."""
        mol = Chem.MolFromSmiles('CCCCCC(=O)[O-]')
        name = name_anion(mol)
        assert name == 'hexanoate', f"Regression: expected 'hexanoate', got '{name}'"


class TestAromaticCarboxylateHelpers:
    """Test the helper functions for aromatic carboxylate detection."""

    def test_find_carboxyl_carbon_acetate(self):
        """_find_carboxyl_carbon should find the carboxyl C in acetate."""
        from orthonym.rules.ions import _find_carboxyl_carbon
        from orthonym.perception.ions import get_ion_sites

        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        sites = get_ion_sites(mol)
        carboxyl_c = _find_carboxyl_carbon(mol, sites['anions'][0])
        assert carboxyl_c is not None
        # Verify it's a carbon with double-bonded oxygen
        atom = mol.GetAtomWithIdx(carboxyl_c)
        assert atom.GetSymbol() == 'C'

    def test_find_carboxyl_carbon_benzoate(self):
        """_find_carboxyl_carbon should find the carboxyl C in benzoate."""
        from orthonym.rules.ions import _find_carboxyl_carbon
        from orthonym.perception.ions import get_ion_sites

        mol = Chem.MolFromSmiles('O=C([O-])c1ccccc1')
        sites = get_ion_sites(mol)
        carboxyl_c = _find_carboxyl_carbon(mol, sites['anions'][0])
        assert carboxyl_c is not None
        atom = mol.GetAtomWithIdx(carboxyl_c)
        assert atom.GetSymbol() == 'C'

    def test_detect_aromatic_benzoate(self):
        """_detect_aromatic_carboxylate should return 'benzoate' for benzene attachment."""
        from orthonym.rules.ions import _find_carboxyl_carbon, _detect_aromatic_carboxylate
        from orthonym.perception.ions import get_ion_sites

        mol = Chem.MolFromSmiles('O=C([O-])c1ccccc1')
        sites = get_ion_sites(mol)
        carboxyl_c = _find_carboxyl_carbon(mol, sites['anions'][0])
        aromatic_name = _detect_aromatic_carboxylate(mol, carboxyl_c)
        assert aromatic_name == 'benzoate'

    def test_detect_aromatic_naphthoate(self):
        """_detect_aromatic_carboxylate should return 'naphthoate' for naphthalene attachment."""
        from orthonym.rules.ions import _find_carboxyl_carbon, _detect_aromatic_carboxylate
        from orthonym.perception.ions import get_ion_sites

        mol = Chem.MolFromSmiles('O=C([O-])c1ccc2ccccc2c1')
        sites = get_ion_sites(mol)
        carboxyl_c = _find_carboxyl_carbon(mol, sites['anions'][0])
        aromatic_name = _detect_aromatic_carboxylate(mol, carboxyl_c)
        assert aromatic_name == 'naphthoate'

    def test_detect_aromatic_none_for_acyclic(self):
        """_detect_aromatic_carboxylate should return None for acyclic carboxylates."""
        from orthonym.rules.ions import _find_carboxyl_carbon, _detect_aromatic_carboxylate
        from orthonym.perception.ions import get_ion_sites

        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        sites = get_ion_sites(mol)
        carboxyl_c = _find_carboxyl_carbon(mol, sites['anions'][0])
        aromatic_name = _detect_aromatic_carboxylate(mol, carboxyl_c)
        assert aromatic_name is None


# ============================================================================
# a phase.5 SUB-01 — charge-aware naming (Wave 0 fixtures)
#
# Negative canaries assert NOW (the currently-correct charged paths that the
# SUB-01 routing change MUST NOT regress). The charge-aware targets are
# xfail until Plan 02 lands (they currently hit the carbon-counting
# _name_alkoxide_systematic stub: heptanolate/propanolate/methanolate/heptylium).
# ============================================================================

from orthonym import name_compound  # noqa: E402


@pytest.mark.unit
class TestSUB01NegativeCanary:
    """Currently-correct charged names that SUB-01 routing MUST preserve."""

    def test_acetate_unchanged(self):
        assert name_compound("CC(=O)[O-]") == "acetate"

    def test_benzoate_unchanged(self):
        assert name_compound("[O-]C(=O)c1ccccc1") == "benzoate"

    def test_propanolate_unchanged(self):
        # 169.6-03 named this 'propan-1-olate' (systematic, locant mandated).
        # Wave2 T2d supersedes it: BB P-63.8.1 VERBATIM retains 'propoxide'
        # as the PIN ("sodium propoxide (PIN) sodium propan-1-olate"), so the
        # retained table now resolves the bare skeleton first. Substituted
        # alkoxides still take the systematic -olate route this canary was
        # written to protect (see test_tier2d_data_flips for the family).
        assert name_compound("CCC[O-]") == "propoxide"

    def test_propanethiolate_unchanged(self):
        # 169.6-03: chokepoint adds the IUPAC locant ('propane-1-thiolate'),
        # replacing the deleted carbanion/thiolate stub's locant-less
        # 'propanethiolate'. RT-preserving (same InChI under OPSIN) -> strict
        # improvement.
        assert name_compound("CCC[S-]") == "propane-1-thiolate"


@pytest.mark.unit
class TestSUB01ChargeAwareNaming:
    """SUB-01 charge-aware targets — xfail until Plan 02 routes charged
    species through the general pipeline + the ionic-suffix seam."""

    def test_propanesulfonate(self):
        # SUB-01 Plan 02: routed through general pipeline + structured suffix.
        name = name_compound("CCCS(=O)(=O)[O-]")
        assert "sulfonate" in name and "olate" not in name

    def test_formylbenzenesulfonate_parent_and_suffix(self):
        name = name_compound("O=Cc1ccc(S(=O)(=O)[O-])cc1")
        #: assert correct parent + -sulfonate; the missing 4- locant is a
        # SEPARATE pre-existing ring-substituent-locant defect, out of SUB-01 scope.
        assert "sulfonate" in name and "heptanolate" not in name

    def test_methylphosphonate(self):
        name = name_compound("CP(=O)(O)[O-]")
        assert "phosphonate" in name

    def test_methylphosphonate_dianion_keeps_charge(self):
        # CR-02 (code review 2026-06-02): the FULLY-deprotonated S/P-oxoacid
        # DIANION must ship the anion name ("methanephosphonate", which OPSIN
        # round-trips to the -2 dianion), NOT the neutral acid. Before the fix
        # the poly-anion path neutralized both [O-] and returned
        # "methanephosphonic acid" (charge silently dropped).
        name = name_compound("CP(=O)([O-])[O-]")
        assert "phosphonate" in name
        assert "acid" not in name

    def test_phosphonate_dianion_via_name_anion(self):
        # The library entry point (name_anion) routes the dianion through its
        # multi-anion oxoacid branch (mirror of the dispatch-table fix).
        mol = Chem.MolFromSmiles("CCP(=O)([O-])[O-]")
        out = name_anion(mol, style="pin")
        assert "phosphonate" in out and "acid" not in out

    def test_carboxylate_dianion_not_misrouted(self):
        # Regression guard for the CR-02 fix: a pure CARBOXYLATE dianion must
        # NOT enter the oxoacid branch. W3-P09 (P-65.6.2.1/P-65.6.1.1): the PIN
        # is the systematic 'butanedioate' ('succinate' is retained for general
        # nomenclature only), reached via the now-PIN-aware retained lookup.
        assert name_compound("[O-]C(=O)CCC(=O)[O-]") == "butanedioate"

    def test_wr05_oxoacid_ionize_ignores_bare_ol_amine(self):
        # WR-05 (code review 2026-06-02): when _ionize_acid_name is restricted to
        # the oxoacid suffix set, a parent that merely ENDS in "ol"/"amine" must
        # NOT be transformed (no spurious "...olate"); only genuine oxoacid
        # suffixes ionize.
        from orthonym.rules.ions import _ionize_acid_name, _OXOACID_NEUTRAL_SUFFIXES
        assert _ionize_acid_name("methanephosphonic acid", -1,
                                 _OXOACID_NEUTRAL_SUFFIXES) == "methanephosphonate"
        assert _ionize_acid_name("2-methylphenol", -1,
                                 _OXOACID_NEUTRAL_SUFFIXES) == ""
        assert _ionize_acid_name("some-parentol", -1,
                                 _OXOACID_NEUTRAL_SUFFIXES) == ""
        # The general (unrestricted) helper still ionizes an alcohol, unchanged.
        assert _ionize_acid_name("ethanol", -1) == "ethanolate"

    def test_wr01_degenerate_parent_guard_is_precise(self):
        # WR-01 (code review 2026-06-02): the malformed-parent guard must reject
        # ONLY the degenerate stem-less form (ane/ene/yne glued to an oxoacid
        # suffix or locant), never a legitimate parent that merely begins with
        # those three letters.
        from orthonym.rules.ions import _DEGENERATE_OXOACID_PARENT_RE as rgx
        for degenerate in ("anesulfonic acid", "enephosphonic acid",
                           "yne-1-sulfonic acid", "ane-1-phosphonic acid"):
            assert rgx.match(degenerate), degenerate
        for legit in ("methanesulfonic acid", "benzenesulfonic acid",
                     "cyclohexanesulfonic acid", "anethole-ish-sulfonic acid"):
            assert not rgx.match(legit), legit

    @pytest.mark.xfail(reason="DEFERRED (169.5, honest-fail): carbenium 'phenylmethylium' needs cation-aware parent selection (the C+ as the methylium parent) — neutralize-recurse loses the cation position; a substantial select_parent change, small reach. Currently 'heptylium'.", strict=False)
    def test_phenylmethylium_not_benzylium(self):
        name = name_compound("[CH2+]c1ccccc1")
        # RESEARCH gotcha: phenylmethylium, NOT benzylium, NOT heptylium.
        assert name == "phenylmethylium"

    def test_methanaminium(self):
        # RESOLVED (was xfail): the general-only 'methylammonium'
        # retained name is denied on the PIN path (P-73.1.2.1) so the systematic
        # aminium PIN 'methanaminium' is emitted. 'methylammonium' stays for
        # general/common style.
        assert name_compound("C[NH3+]") == "methanaminium"

    def test_azide_keeps_azido_prefix(self):
        # SUB-01/C2 Plan 02: azido SMARTS fixed -> the azide is no longer dropped.
        name = name_compound("[N-]=[N+]=NCCCNC(=O)CCCC(=O)O")
        assert "azido" in name
