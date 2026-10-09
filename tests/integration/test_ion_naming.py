"""Integration tests for complete ion naming pipeline."""
import pytest
from orthonym.namer import name_compound


class TestAnionNaming:
    """Test anion naming through main API."""

    def test_acetate(self):
        """Test acetate anion naming."""
        assert name_compound('CC(=O)[O-]') == 'acetate'

    def test_formate(self):
        """Test formate anion naming."""
        name = name_compound('[O-]C=O')
        assert name == 'formate'

    def test_methoxide(self):
        """Test methoxide anion naming."""
        assert name_compound('C[O-]') == 'methoxide'

    def test_ethoxide(self):
        """Test ethoxide anion naming."""
        assert name_compound('CC[O-]') == 'ethoxide'

    def test_phenolate(self):
        """Test phenolate anion naming."""
        name = name_compound('[O-]c1ccccc1')
        # Should be phenoxide or phenolate
        assert 'phen' in name.lower()

    def test_propanoate(self):
        """Test propanoate anion naming."""
        name = name_compound('CCC(=O)[O-]')
        assert 'propano' in name.lower() or name == 'propanoate'


class TestCationNaming:
    """Test cation naming through main API."""

    def test_ammonium(self):
        """NH4+ PIN is 'azanium', the Blue Book; was 'ammonium')."""
        assert name_compound('[NH4+]') == 'azanium'

    # 'Cation and anion names' (the Blue Book): R4N+ salts are named by
    # "(1) adding the suffix 'ium' to the name of the amine or imine, [...] (3) by
    # substituting the parent hydride 'ammonium', NH4+, for quaternary salts only.
    # Method (1) leads to preferred IUPAC names." (examples: 'methanaminium chloride
    # (PIN)',:26672; 'N,N,N-trimethylmethanaminium iodide (PIN)',:26683). The
    # substitutive aminium names replaced the '(alkyl)ammonium' forms in commit 12541211e.

    def test_methylammonium(self):
        """CH3-NH3+ is 'methanaminium' (PIN), not 'methylammonium'."""
        assert name_compound('C[NH3+]') == 'methanaminium'

    def test_dimethylammonium(self):
        """(CH3)2NH2+ is 'N-methylmethanaminium' (PIN), not 'dimethylammonium'."""
        assert name_compound('C[NH2+]C') == 'N-methylmethanaminium'

    def test_trimethylammonium(self):
        """(CH3)3NH+ is 'N,N-dimethylmethanaminium' (PIN), not 'trimethylammonium'."""
        assert name_compound('C[NH+](C)C') == 'N,N-dimethylmethanaminium'


class TestSaltNaming:
    """Test salt naming through main API."""

    def test_sodium_acetate(self):
        """Test sodium acetate salt naming."""
        name = name_compound('[Na+].[O-]C(C)=O')
        assert name == 'sodium acetate'

    def test_potassium_chloride(self):
        """Test potassium chloride salt naming."""
        name = name_compound('[K+].[Cl-]')
        assert name == 'potassium chloride'

    def test_sodium_chloride(self):
        """Test sodium chloride salt naming."""
        name = name_compound('[Na+].[Cl-]')
        assert name == 'sodium chloride'

    def test_ammonium_chloride(self):
        """Test ammonium chloride salt naming."""
        name = name_compound('[NH4+].[Cl-]')
        assert name == 'ammonium chloride'

    def test_lithium_bromide(self):
        """Test lithium bromide salt naming."""
        name = name_compound('[Li+].[Br-]')
        assert 'lithium' in name
        assert 'bromide' in name

    def test_potassium_phenolate(self):
        """Test potassium phenolate salt naming."""
        name = name_compound('[K+].[O-]c1ccccc1')
        assert 'potassium' in name
        # May be phenoxide or phenolate
        assert 'phen' in name.lower()


class TestRadicalNaming:
    """Test radical naming through main API."""

    def test_methyl_radical(self):
        """Test methyl radical naming."""
        name = name_compound('[CH3]')
        assert name == 'methyl'

    def test_ethyl_radical(self):
        """Test ethyl radical naming."""
        name = name_compound('C[CH2]')
        assert name == 'ethyl'

    def test_propyl_radical(self):
        """Test propyl radical naming."""
        name = name_compound('CC[CH2]')
        assert name == 'propyl'


class TestZwitterionNaming:
    """Test zwitterion naming through main API."""

    def test_glycine_zwitterion(self):
        """Test glycine zwitterion naming."""
        name = name_compound('[NH3+]CC([O-])=O')
        # Either 'glycine' or '2-azaniumylacetate' or similar
        assert name is not None
        assert len(name) > 0

    def test_alanine_zwitterion(self):
        """Test alanine zwitterion naming."""
        name = name_compound('[NH3+]C(C)C([O-])=O')
        assert name is not None
        assert len(name) > 0


class TestNeutralMoleculeRegression:
    """Ensure neutral molecules still work correctly after ion routing."""

    def test_ethanol(self):
        """Test ethanol naming still works."""
        assert name_compound('CCO') == 'ethanol'

    def test_methanol(self):
        """Test methanol naming still works."""
        assert name_compound('CO') == 'methanol'

    def test_benzene(self):
        """Test benzene naming still works."""
        assert name_compound('c1ccccc1') == 'benzene'

    def test_acetic_acid(self):
        """Test acetic acid naming still works."""
        assert name_compound('CC(=O)O') == 'acetic acid'

    def test_propane(self):
        """Test propane naming still works."""
        assert name_compound('CCC') == 'propane'

    def test_butane(self):
        """Test butane naming still works."""
        assert name_compound('CCCC') == 'butane'

    def test_cyclohexane(self):
        """Test cyclohexane naming still works."""
        assert name_compound('C1CCCCC1') == 'cyclohexane'

    def test_toluene(self):
        """Test toluene naming still works."""
        assert name_compound('Cc1ccccc1') == 'toluene'


class TestSpeciesTypeDetection:
    """Test that species type detection routes correctly."""

    def test_salt_detection(self):
        """Test that salts are routed correctly."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        assert detect_species_type(mol) == 'salt'

    def test_ion_detection(self):
        """Test that single ions are detected correctly."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        mol = Chem.MolFromSmiles('[NH4+]')
        assert detect_species_type(mol) == 'ion'

    def test_zwitterion_detection(self):
        """Test that zwitterions are detected correctly."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert detect_species_type(mol) == 'zwitterion'

    def test_radical_detection(self):
        """Test that radicals are detected correctly."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        mol = Chem.MolFromSmiles('[CH3]')
        assert detect_species_type(mol) == 'radical'

    def test_neutral_detection(self):
        """Test that neutral molecules are detected correctly."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        mol = Chem.MolFromSmiles('CCO')
        assert detect_species_type(mol) == 'neutral'


class TestMolecularFeaturesIonFields:
    """Test that MolecularFeatures has ion/radical fields populated."""

    def test_salt_features(self):
        """Test that salt features are populated."""
        from orthonym.namer import Orthonym
        from rdkit import Chem

        namer = Orthonym()
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        smiles = '[Na+].[Cl-]'
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, smiles, canonical)

        assert features.species_type == 'salt'
        assert features.ion_sites is not None
        assert 'cations' in features.ion_sites
        assert 'anions' in features.ion_sites

    def test_radical_features(self):
        """Test that radical features are populated."""
        from orthonym.namer import Orthonym
        from rdkit import Chem

        namer = Orthonym()
        mol = Chem.MolFromSmiles('[CH3]')
        smiles = '[CH3]'
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, smiles, canonical)

        assert features.species_type == 'radical'
        assert features.radical_sites is not None
        assert len(features.radical_sites) > 0

    def test_neutral_features(self):
        """Test that neutral molecule features have default values."""
        from orthonym.namer import Orthonym
        from rdkit import Chem

        namer = Orthonym()
        mol = Chem.MolFromSmiles('CCO')
        smiles = 'CCO'
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, smiles, canonical)

        assert features.species_type == 'neutral'
        assert features.total_charge == 0
