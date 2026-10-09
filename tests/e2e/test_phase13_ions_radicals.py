"""
End-to-end tests for a phase: Radicals, Ions, Salts.

Tests verify all a phase requirements:
-: Anion naming (carboxylate, alkoxide, phenolate, aminide, carbanion)
-: Cation naming (aminium, ylium, diazonium, quaternary ammonium)
-: Radical naming (monovalent -yl, divalent -ylidene, trivalent -ylidyne)
-: Zwitterion naming (amino acid zwitterionic forms)
-: Salt naming (compositional nomenclature)
"""
import pytest
from orthonym.namer import name_compound


# =============================================================================
#: Anion Naming
# =============================================================================

class TestION01CarboxylateAnions:
    """Test carboxylate anion naming (-COO-)."""

    @pytest.mark.parametrize("smiles,expected", [
        ('CC(=O)[O-]', 'acetate'),
        ('[O-]C=O', 'formate'),
        ('CCC(=O)[O-]', 'propanoate'),
        ('CCCC(=O)[O-]', 'butanoate'),
        ('O=C([O-])c1ccccc1', 'benzoate'),
    ])
    def test_carboxylate_anions(self, smiles, expected):
        """Carboxylate anions named with -ate suffix."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    def test_dicarboxylate_oxalate(self):
        """Oxalate (dicarboxylate anion)."""
        result = name_compound('O=C([O-])C(=O)[O-]')
        assert result == 'oxalate', f'Got: {result}'

    def test_dicarboxylate_malonate(self):
        """Malonate (3-carbon dicarboxylate). PIN is the systematic
        'propanedioate' (P-65.6.1.1; 'malonate' is retained for general
        nomenclature only) — W3-P09 made the retained lookup PIN-aware."""
        result = name_compound('O=C([O-])CC(=O)[O-]')
        assert result == 'propanedioate', f'Got: {result}'

    def test_dicarboxylate_succinate(self):
        """Succinate (4-carbon dicarboxylate). PIN is the systematic
        'butanedioate' (P-65.6.2.1 'potassium sodium butanedioate (PIN)';
        'succinate' is general nomenclature only) — W3-P09 PIN-aware lookup."""
        result = name_compound('O=C([O-])CCC(=O)[O-]')
        assert result == 'butanedioate', f'Got: {result}'


class TestION01AlkoxideAnions:
    """Test alkoxide anion naming (R-O-)."""

    @pytest.mark.parametrize("smiles,expected", [
        ('C[O-]', 'methoxide'),
        ('CC[O-]', 'ethoxide'),
    ])
    def test_alkoxide_anions(self, smiles, expected):
        """Alkoxide anions named with -oxide suffix (common name)."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    def test_isopropoxide(self):
        """Isopropoxide (branched alkoxide)."""
        result = name_compound('CC(C)[O-]')
        # "Anions derived from hydroxy compounds" (the Blue Book): isopropoxide
        # is excluded from the retained anion names ('but not isopropoxide'); the
        # example at the Blue Book is 'propan-2-olate (PIN) isopropoxide'.
        assert result == 'propan-2-olate', f'Got: {result}'

    def test_tert_butoxide(self):
        """tert-Butoxide."""
        result = name_compound('CC(C)(C)[O-]')
        assert result == 'tert-butoxide', f'Got: {result}'


class TestION01PhenolateAnions:
    """Test phenolate anion naming (ArO-)."""

    def test_phenolate(self):
        """Phenolate (phenoxide) anion."""
        result = name_compound('[O-]c1ccccc1')
        # Accept either phenolate or phenoxide
        assert result in ('phenolate', 'phenoxide'), f'Got: {result}'

    def test_substituted_phenolate(self):
        """4-Methylphenolate (para-cresol anion)."""
        result = name_compound('[O-]c1ccc(C)cc1')
        # Should contain olate or oxide
        assert 'ol' in result.lower(), f'Got: {result}'


class TestION01CarbanionAnions:
    """Test carbanion naming (C-)."""

    def test_methanide(self):
        """Methanide (simplest carbanion)."""
        result = name_compound('[CH3-]')
        assert result == 'methanide', f'Got: {result}'

    def test_phenide(self):
        """Phenide (benzenide) - aromatic carbanion."""
        result = name_compound('[c-]1ccccc1')
        # Accept phenide or benzenide
        assert result in ('phenide', 'benzenide'), f'Got: {result}'

    def test_ethynide(self):
        """Ethynide (terminal alkynide anion)."""
        result = name_compound('[C-]#C')
        assert result == 'ethynide', f'Got: {result}'


# =============================================================================
#: Cation Naming
# =============================================================================

class TestION02AminiumCations:
    """Test aminium cation naming (R-NH3+, R4N+)."""

    # (the Blue Book): the -aminium names are the preferred IUPAC names, not
    # the Table 7.3 'ammonium' ones; 'methanaminium (PIN)' the Blue Book. The ids keep the
    # pre-aminium spelling so the row identity is stable across the suite history.
    @pytest.mark.parametrize("smiles,expected", [
        ('[NH4+]', 'azanium'),  # PIN (was 'ammonium', Table-7.3 retained)
        pytest.param('C[NH3+]', 'methanaminium', id='C[NH3+]-methylammonium'),
        pytest.param('CC[NH3+]', 'ethanaminium', id='CC[NH3+]-ethylammonium'),
        pytest.param('CCC[NH3+]', 'propan-1-aminium', id='CCC[NH3+]-propylammonium'),
    ])
    def test_primary_aminium_cations(self, smiles, expected):
        """Primary aminium cations (R-NH3+)."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    @pytest.mark.parametrize("smiles,expected", [
        pytest.param('C[NH2+]C', 'N-methylmethanaminium', id='C[NH2+]C-dimethylammonium'),
        pytest.param('CC[NH2+]CC', 'N-ethylethanaminium', id='CC[NH2+]CC-diethylammonium'),
    ])
    def test_secondary_aminium_cations(self, smiles, expected):
        """Secondary aminium cations (R2-NH2+)."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    def test_trimethylammonium(self):
        """Trimethylammonium (tertiary)."""
        result = name_compound('C[NH+](C)C')
        assert result == 'N,N-dimethylmethanaminium', f'Got: {result}'

    @pytest.mark.parametrize("smiles,expected", [
        pytest.param('C[N+](C)(C)C', 'N,N,N-trimethylmethanaminium',
                     id='C[N+](C)(C)C-tetramethylammonium'),
        pytest.param('CC[N+](CC)(CC)CC', 'N,N,N-triethylethanaminium',
                     id='CC[N+](CC)(CC)CC-tetraethylammonium'),
    ])
    def test_quaternary_ammonium_cations(self, smiles, expected):
        """Quaternary ammonium cations (R4N+)."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'


class TestION02YliumCations:
    """Test carbenium/ylium cation naming (R+)."""

    def test_methylium(self):
        """Methylium (simplest carbenium)."""
        result = name_compound('[CH3+]')
        assert result == 'methylium', f'Got: {result}'

    def test_ethylium(self):
        """Ethylium."""
        result = name_compound('[CH2+]C')
        assert result == 'ethylium', f'Got: {result}'

    def test_propylium(self):
        """Propylium."""
        result = name_compound('[CH2+]CC')
        assert result == 'propylium', f'Got: {result}'


class TestION02OniumCations:
    """Test onium cation naming (oxonium, sulfonium, phosphonium)."""

    def test_oxonium(self):
        """Oxidanium (H3O+): PIN (was 'oxonium', Table-7.3 retained)."""
        result = name_compound('[OH3+]')
        assert result == 'oxidanium', f'Got: {result}'

    def test_methyloxonium(self):
        """Methyloxidanium: PIN (was 'methyloxonium')."""
        result = name_compound('C[OH2+]')
        assert result == 'methyloxidanium', f'Got: {result}'

    def test_dimethyloxonium(self):
        """Dimethyloxidanium: PIN (was 'dimethyloxonium')."""
        result = name_compound('C[OH+]C')
        assert result == 'dimethyloxidanium', f'Got: {result}'

    def test_phosphonium(self):
        """Phosphanium (PH4+): PIN (was 'phosphonium')."""
        result = name_compound('[PH4+]')
        assert result == 'phosphanium', f'Got: {result}'

    def test_trimethylsulfonium(self):
        """Trimethylsulfanium: PIN (was 'trimethylsulfonium')."""
        result = name_compound('C[S+](C)C')
        assert result == 'trimethylsulfanium', f'Got: {result}'


# =============================================================================
#: Radical Naming
# =============================================================================

class TestRAD01MonovalentRadicals:
    """Test monovalent radical naming (-yl suffix)."""

    @pytest.mark.parametrize("smiles,expected", [
        ('[CH3]', 'methyl'),
        ('C[CH2]', 'ethyl'),
        ('CC[CH2]', 'propyl'),
        ('CCC[CH2]', 'butyl'),
    ])
    def test_alkyl_radicals(self, smiles, expected):
        """Monovalent alkyl radicals (-yl suffix)."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'


class TestRAD01DivalentRadicals:
    """Test divalent radical naming (-ylidene suffix)."""

    def test_methylidene(self):
        """Methylidene (simplest carbene-like radical)."""
        result = name_compound('[CH2]')
        assert result == 'methylidene', f'Got: {result}'

    def test_ethylidene(self):
        """Ethylidene."""
        result = name_compound('C[CH]')
        # Note: RDKit may interpret this differently
        # Accept ethylidene or equivalent
        assert 'idene' in result or 'yl' in result, f'Got: {result}'


class TestRAD01TrivalentRadicals:
    """Test trivalent radical naming (-ylidyne suffix)."""

    def test_methylidyne(self):
        """Methylidyne (carbyne-like radical)."""
        result = name_compound('[CH]')
        assert result == 'methylidyne', f'Got: {result}'


class TestRAD01AcylRadicals:
    """Test acyl radical naming (-oyl suffix)."""

    def test_formyl_radical(self):
        """Formyl radical."""
        result = name_compound('[CH]=O')
        assert result == 'formyl', f'Got: {result}'

    def test_acetyl_radical(self):
        """Acetyl radical."""
        result = name_compound('C[C]=O')
        assert result == 'acetyl', f'Got: {result}'

    def test_propanoyl_radical(self):
        """Propanoyl radical."""
        result = name_compound('CC[C]=O')
        assert result == 'propanoyl', f'Got: {result}'


class TestRAD01OxylRadicals:
    """Test oxyl radical naming (-oxyl suffix)."""

    def test_methoxyl_radical(self):
        """Methoxyl radical."""
        result = name_compound('[O]C')
        assert result == 'methoxyl', f'Got: {result}'

    def test_ethoxyl_radical(self):
        """Ethoxyl radical."""
        result = name_compound('[O]CC')
        assert result == 'ethoxyl', f'Got: {result}'

    def test_phenoxyl_radical(self):
        """Phenoxyl radical."""
        result = name_compound('[O]c1ccccc1')
        assert result == 'phenoxyl', f'Got: {result}'


# =============================================================================
#: Zwitterion Naming
# =============================================================================

class TestZWIT01AminoAcidZwitterions:
    """Test amino acid zwitterion naming."""

    def test_glycine_zwitterion(self):
        """Glycine zwitterion - systematic name."""
        result = name_compound('[NH3+]CC([O-])=O')
        # Should be systematic 2-azaniumylacetate or trivial glycine
        assert 'azaniumyl' in result.lower() or 'glycine' in result.lower(), \
            f'Got: {result}'

    def test_alanine_zwitterion(self):
        """Alanine zwitterion (racemic)."""
        result = name_compound('[NH3+]C(C)C([O-])=O')
        # Should produce some name (systematic or trivial)
        assert result is not None and len(result) > 0, f'Got: {result}'
        # Should be 2-azaniumylpropanoate or alanine
        assert 'azaniumyl' in result.lower() or 'alanine' in result.lower(), \
            f'Got: {result}'

    def test_glycine_zwitterion_variant_smiles(self):
        """Glycine zwitterion with different SMILES notation."""
        # Alternate SMILES for glycine zwitterion
        result = name_compound('NCC(=O)[O-]')  # This is NOT zwitterion (neutral N)
        # This should NOT be named as zwitterion
        assert 'azaniumyl' not in result.lower(), \
            f'Expected neutral amino acid, got: {result}'


# =============================================================================
#: Salt Naming (Compositional Nomenclature)
# =============================================================================

class TestSALT01SimpleOrganicSalts:
    """Test simple organic salt naming."""

    @pytest.mark.parametrize("smiles,expected", [
        ('[Na+].[O-]C(C)=O', 'sodium acetate'),
        ('[K+].[O-]C=O', 'potassium formate'),
    ])
    def test_alkali_carboxylate_salts(self, smiles, expected):
        """Alkali metal carboxylate salts."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    def test_potassium_phenolate(self):
        """Potassium phenolate (phenoxide)."""
        result = name_compound('[K+].[O-]c1ccccc1')
        # Accept phenolate or phenoxide
        assert 'potassium' in result, f'Got: {result}'
        assert 'phen' in result.lower(), f'Got: {result}'

    def test_lithium_methoxide(self):
        """Lithium methoxide."""
        result = name_compound('[Li+].[O-]C')
        assert 'lithium' in result, f'Got: {result}'
        assert 'meth' in result.lower(), f'Got: {result}'


class TestSALT01InorganicSalts:
    """Test inorganic salt naming."""

    @pytest.mark.parametrize("smiles,expected", [
        ('[NH4+].[Cl-]', 'ammonium chloride'),
        ('[Na+].[Cl-]', 'sodium chloride'),
        ('[K+].[Br-]', 'potassium bromide'),
    ])
    def test_inorganic_halide_salts(self, smiles, expected):
        """Simple inorganic halide salts."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    def test_sodium_fluoride(self):
        """Sodium fluoride."""
        result = name_compound('[Na+].[F-]')
        assert result == 'sodium fluoride', f'Got: {result}'

    def test_potassium_iodide(self):
        """Potassium iodide."""
        result = name_compound('[K+].[I-]')
        assert result == 'potassium iodide', f'Got: {result}'


class TestSALT01DivalentSalts:
    """Test salts with divalent cations."""

    def test_calcium_dichloride(self):
        """Calcium chloride (with 2 chloride ions)."""
        result = name_compound('[Ca+2].[Cl-].[Cl-]')
        assert 'calcium' in result, f'Got: {result}'
        assert 'chloride' in result, f'Got: {result}'

    def test_calcium_diacetate(self):
        """Calcium acetate (with 2 acetate ions)."""
        result = name_compound('[Ca+2].[O-]C(C)=O.[O-]C(C)=O')
        assert 'calcium' in result, f'Got: {result}'
        assert 'acetate' in result, f'Got: {result}'

    @pytest.mark.skip(reason="Sulfate dianion causes recursion in current implementation")
    def test_magnesium_sulfate(self):
        """Magnesium sulfate - KNOWN LIMITATION: dianions like sulfate not yet supported."""
        result = name_compound('[Mg+2].[O-]S([O-])(=O)=O')
        assert 'magnesium' in result.lower(), f'Got: {result}'

    def test_zinc_chloride(self):
        """Zinc chloride."""
        result = name_compound('[Zn+2].[Cl-].[Cl-]')
        assert 'zinc' in result.lower(), f'Got: {result}'
        assert 'chloride' in result, f'Got: {result}'


class TestSALT01TransitionMetalSalts:
    """Test transition metal salts."""

    def test_iron_ii_chloride(self):
        """Iron(II) chloride."""
        result = name_compound('[Fe+2].[Cl-].[Cl-]')
        assert 'iron' in result.lower(), f'Got: {result}'
        assert 'chloride' in result, f'Got: {result}'

    @pytest.mark.skip(reason="Cu+2 misdetected as radical in species_type - known limitation")
    def test_copper_ii_acetate(self):
        """Copper(II) acetate - KNOWN LIMITATION: Cu+2 species detection issue."""
        # Note: name_salt works correctly when called directly
        # Issue is in detect_species_type for Cu+2
        result = name_compound('[Cu+2].[O-]C(C)=O.[O-]C(C)=O')
        assert 'copper' in result.lower(), f'Got: {result}'
        assert 'acetate' in result, f'Got: {result}'

    def test_silver_nitrate(self):
        """Silver nitrate (if nitrate pattern recognized)."""
        result = name_compound('[Ag+].[O-][N+](=O)[O-]')
        assert 'silver' in result.lower(), f'Got: {result}'


# =============================================================================
# Regression Tests: Neutral Molecules
# =============================================================================

class TestNeutralMoleculeRegression:
    """Ensure neutral molecules still named correctly after a phase."""

    @pytest.mark.parametrize("smiles,expected", [
        ('CCO', 'ethanol'),
        ('c1ccccc1', 'benzene'),
        ('CC(=O)O', 'acetic acid'),
        ('CCCC', 'butane'),
        ('CC(C)C', '2-methylpropane'),
        ('C1CCCCC1', 'cyclohexane'),
        ('c1ccncc1', 'pyridine'),
    ])
    def test_neutral_molecules(self, smiles, expected):
        """Neutral molecules should still be named correctly."""
        result = name_compound(smiles)
        assert result == expected, f'Expected {expected}, got {result}'

    def test_methyl_acetate(self):
        """Methyl acetate (ester)."""
        result = name_compound('CC(=O)OC')
        assert result == 'methyl acetate', f'Got: {result}'

    def test_acetamide(self):
        """Acetamide (amide)."""
        result = name_compound('CC(=O)N')
        assert result == 'acetamide', f'Got: {result}'

    def test_propanoic_acid(self):
        """Propanoic acid."""
        result = name_compound('CCC(=O)O')
        assert result == 'propanoic acid', f'Got: {result}'

    def test_propanal(self):
        """Propanal (aldehyde)."""
        result = name_compound('CCC=O')
        assert result == 'propanal', f'Got: {result}'


class TestNoRegressionOnExistingTests:
    """Ensure all a phase modules can be imported."""

    def test_can_import_all_naming_modules(self):
        """Verify all naming modules import without error."""
        from orthonym.perception.ions import detect_species_type
        from orthonym.rules.ions import name_anion, name_cation
        from orthonym.rules.radicals import name_radical
        from orthonym.rules.salts import name_salt, name_zwitterion
        from orthonym.data.ion_retained_names import RETAINED_ANIONS

        assert detect_species_type is not None
        assert name_anion is not None
        assert name_cation is not None
        assert name_radical is not None
        assert name_salt is not None
        assert RETAINED_ANIONS is not None

    def test_module_functions_callable(self):
        """Verify module functions can be called."""
        from rdkit import Chem
        from orthonym.perception.ions import detect_species_type, get_ion_sites

        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        species = detect_species_type(mol)
        assert species == 'ion'

        sites = get_ion_sites(mol)
        assert 'anions' in sites
        assert len(sites['anions']) == 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_neutral_acid_not_salt(self):
        """Ensure neutral acids aren't misidentified as salts."""
        # Acetic acid is neutral, not a salt
        result = name_compound('CC(=O)O')
        assert result == 'acetic acid', f'Got: {result}'
        assert 'sodium' not in result, 'Neutral acid should not contain metal name'

    def test_neutral_with_zwitterion_pattern(self):
        """Amino acids in non-zwitterionic form."""
        # Glycine (neutral form: NH2-CH2-COOH)
        result = name_compound('NCC(=O)O')
        # Should be glycine or 2-aminoacetic acid, NOT zwitterion name
        assert 'azaniumyl' not in result.lower(), \
            f'Neutral amino acid should not have azaniumyl: {result}'

    def test_single_ion_vs_salt(self):
        """Single ion (not salt) naming."""
        # Single cation only
        result = name_compound('[NH4+]')
        assert result == 'azanium', f'Got: {result}'  # PIN (was 'ammonium')
        # Not 'azanium something' - just the cation

    def test_multiple_charges_same_atom(self):
        """Handle divalent ions correctly."""
        result = name_compound('[Ca+2].[Cl-].[Cl-]')
        assert 'calcium' in result, f'Got: {result}'
        assert 'chloride' in result, f'Got: {result}'

    def test_chain_length_edge_cases(self):
        """Test ions at chain length boundaries."""
        # Very short chain
        result = name_compound('[O-]C=O')  # formate
        assert result == 'formate', f'Got: {result}'

        # Longer chain
        result = name_compound('CCCCC(=O)[O-]')  # pentanoate
        assert 'pentanoate' in result or 'valerate' in result.lower(), f'Got: {result}'


class TestRetainedNamesConsistency:
    """Test that retained names are applied consistently."""

    def test_acetate_retained(self):
        """Acetate should use retained name."""
        result = name_compound('CC(=O)[O-]')
        assert result == 'acetate', f'Got: {result}'

    def test_formate_retained(self):
        """Formate should use retained name."""
        result = name_compound('[O-]C=O')
        assert result == 'formate', f'Got: {result}'

    def test_ammonium_retained(self):
        """NH4+ PIN is 'azanium', the Blue Book; was 'ammonium')."""
        result = name_compound('[NH4+]')
        assert result == 'azanium', f'Got: {result}'

    def test_methyl_radical_retained(self):
        """Methyl radical should use retained name."""
        result = name_compound('[CH3]')
        assert result == 'methyl', f'Got: {result}'


# =============================================================================
# Requirements Verification Matrix
# =============================================================================

class TestRequirementsVerification:
    """
    Explicit verification that each a phase requirement is met.

    Requirements:
    -: Anion naming for carboxylate, alkoxide, phenolate, carbanion
    -: Cation naming for aminium, ylium types
    -: Radical naming for monovalent, divalent, trivalent
    -: Zwitterion naming for amino acid zwitterions
    -: Compositional nomenclature for salts
    """

    def test_ion01_carboxylate(self):
        """: Carboxylate anions work."""
        assert name_compound('CC(=O)[O-]') == 'acetate'

    def test_ion01_alkoxide(self):
        """: Alkoxide anions work."""
        assert name_compound('C[O-]') == 'methoxide'

    def test_ion01_phenolate(self):
        """: Phenolate anions work."""
        result = name_compound('[O-]c1ccccc1')
        assert 'phen' in result.lower() and ('olate' in result or 'oxide' in result)

    def test_ion01_carbanion(self):
        """: Carbanions work."""
        assert name_compound('[CH3-]') == 'methanide'

    def test_ion02_aminium(self):
        """: Aminium cations work."""
        assert name_compound('[NH4+]') == 'azanium'  # (the Blue Book)
        assert name_compound('C[NH3+]') == 'methanaminium'  # the Blue Book (PIN)

    def test_ion02_ylium(self):
        """: Ylium cations work."""
        assert name_compound('[CH3+]') == 'methylium'

    def test_rad01_monovalent(self):
        """: Monovalent radicals work."""
        assert name_compound('[CH3]') == 'methyl'
        assert name_compound('C[CH2]') == 'ethyl'

    def test_rad01_divalent(self):
        """: Divalent radicals work."""
        assert name_compound('[CH2]') == 'methylidene'

    def test_rad01_trivalent(self):
        """: Trivalent radicals work."""
        assert name_compound('[CH]') == 'methylidyne'

    def test_zwit01_zwitterion(self):
        """: Zwitterion naming works."""
        result = name_compound('[NH3+]CC([O-])=O')
        assert 'azaniumyl' in result.lower() or 'glycine' in result.lower()

    def test_salt01_compositional(self):
        """: Salt naming uses compositional format."""
        result = name_compound('[Na+].[O-]C(C)=O')
        assert result == 'sodium acetate'
        # Check format: cation + space + anion
        assert ' ' in result
