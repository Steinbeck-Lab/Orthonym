"""Integration tests for complete ion naming pipeline."""
import pytest
from orthonym.namer import name_compound


class TestAnionNaming:
    """Test anion naming through main API."""

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

    # 'Cation and anion names' (the Blue Book): R4N+ salts are named by
    # "(1) adding the suffix 'ium' to the name of the amine or imine, [...] (3) by
    # substituting the parent hydride 'ammonium', NH4+, for quaternary salts only.
    # Method (1) leads to preferred IUPAC names." (examples: 'methanaminium chloride
    # (PIN)',:26672; 'N,N,N-trimethylmethanaminium iodide (PIN)',:26683). The
    # substitutive aminium names replaced the '(alkyl)ammonium' forms in commit 12541211e.

    def test_dimethylammonium(self):
        """(CH3)2NH2+ is 'N-methylmethanaminium' (PIN), not 'dimethylammonium'."""
        assert name_compound('C[NH2+]C') == 'N-methylmethanaminium'

    def test_trimethylammonium(self):
        """(CH3)3NH+ is 'N,N-dimethylmethanaminium' (PIN), not 'trimethylammonium'."""
        assert name_compound('C[NH+](C)C') == 'N,N-dimethylmethanaminium'


class TestSaltNaming:
    """Test salt naming through main API."""

    def test_sodium_chloride(self):
        """Test sodium chloride salt naming."""
        name = name_compound('[Na+].[Cl-]')
        assert name == 'sodium chloride'

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


@pytest.mark.parametrize(
    "smiles, expected",
    [
        pytest.param("CCS(=O)(=O)[O-]", "ethanesulfonate", id="CCS(=O)(=O)[O-]"),
    ],
)
def test_merged_ion_names(smiles, expected):
    """Names that development-era snapshot tests pinned and that no other test asserts.

    These (SMILES, name) pairs were asserted only by test files removed in the test
    slimming change; they move here so the names stay pinned by a rule-level test.
    Removed tests this replaces:
      - tests/unit/test_perception_171_gold_tripwires.py::test_def3_monosubstituted_ethane_sulfonate_still_elides (1)
    Rules cited by those tests:.
    """
    from orthonym import name_compound
    assert name_compound(smiles) == expected
