"""Unit tests for ion/radical perception."""
import pytest
from rdkit import Chem
from orthonym.perception.ions import (
    detect_species_type,
    get_ion_sites,
    get_radical_sites,
    parse_salt_fragments,
)


class TestDetectSpeciesType:
    """Test species type detection."""

    def test_neutral_molecule(self):
        """Test neutral organic molecule detection."""
        mol = Chem.MolFromSmiles('CCO')
        assert detect_species_type(mol) == 'neutral'

    def test_neutral_benzene(self):
        """Test neutral aromatic compound."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert detect_species_type(mol) == 'neutral'

    def test_simple_cation(self):
        """Test ammonium cation detection."""
        mol = Chem.MolFromSmiles('[NH4+]')
        assert detect_species_type(mol) == 'ion'

    def test_organic_cation(self):
        """Test methylammonium cation."""
        mol = Chem.MolFromSmiles('C[NH3+]')
        assert detect_species_type(mol) == 'ion'

    def test_simple_anion(self):
        """Test acetate anion detection."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        assert detect_species_type(mol) == 'ion'

    def test_carboxylate_anion(self):
        """Test formate anion."""
        mol = Chem.MolFromSmiles('[O-]C=O')
        assert detect_species_type(mol) == 'ion'

    def test_zwitterion(self):
        """Test amino acid zwitterion (glycine)."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert detect_species_type(mol) == 'zwitterion'

    def test_zwitterion_betaine(self):
        """Test betaine-type zwitterion."""
        mol = Chem.MolFromSmiles('C[N+](C)(C)CC([O-])=O')
        assert detect_species_type(mol) == 'zwitterion'

    def test_salt(self):
        """Test sodium acetate salt."""
        mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        assert detect_species_type(mol) == 'salt'

    def test_salt_potassium_chloride(self):
        """Test potassium chloride."""
        mol = Chem.MolFromSmiles('[K+].[Cl-]')
        assert detect_species_type(mol) == 'salt'

    def test_salt_calcium_acetate(self):
        """Test calcium acetate (1:2 salt)."""
        mol = Chem.MolFromSmiles('[Ca+2].CC([O-])=O.CC([O-])=O')
        assert detect_species_type(mol) == 'salt'

    def test_radical(self):
        """Test methyl radical detection."""
        mol = Chem.MolFromSmiles('[CH3]')
        assert detect_species_type(mol) == 'radical'

    def test_radical_ethyl(self):
        """Test ethyl radical."""
        mol = Chem.MolFromSmiles('C[CH2]')
        assert detect_species_type(mol) == 'radical'

    def test_radical_phenyl(self):
        """Test phenyl radical."""
        mol = Chem.MolFromSmiles('[c]1ccccc1')
        assert detect_species_type(mol) == 'radical'

    def test_none_input(self):
        """Test None input returns neutral."""
        assert detect_species_type(None) == 'neutral'


class TestGetIonSites:
    """Test ion site detection."""

    def test_ammonium_cation(self):
        """Test ammonium cation site detection."""
        mol = Chem.MolFromSmiles('[NH4+]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        assert len(sites['anions']) == 0
        assert sites['cations'][0]['element'] == 'N'
        assert sites['cations'][0]['charge'] == 1
        assert sites['cations'][0]['n_hydrogens'] == 4

    def test_acetate_anion(self):
        """Test acetate anion site detection."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 1
        assert len(sites['cations']) == 0
        assert sites['anions'][0]['element'] == 'O'
        assert sites['anions'][0]['charge'] == -1

    def test_zwitterion_sites(self):
        """Test zwitterion has both cation and anion sites."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        assert len(sites['anions']) == 1
        assert sites['cations'][0]['element'] == 'N'
        assert sites['anions'][0]['element'] == 'O'

    def test_dication(self):
        """Test divalent cation (calcium)."""
        mol = Chem.MolFromSmiles('[Ca+2]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        assert sites['cations'][0]['charge'] == 2
        assert sites['cations'][0]['element'] == 'Ca'

    def test_dianion(self):
        """Test dianion (oxalate)."""
        mol = Chem.MolFromSmiles('[O-]C(=O)C(=O)[O-]')
        sites = get_ion_sites(mol)
        assert len(sites['anions']) == 2
        assert all(s['charge'] == -1 for s in sites['anions'])

    def test_neutral_no_sites(self):
        """Test neutral molecule has no ion sites."""
        mol = Chem.MolFromSmiles('CCO')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 0
        assert len(sites['anions']) == 0

    def test_none_input(self):
        """Test None input returns empty dict."""
        sites = get_ion_sites(None)
        assert sites == {'cations': [], 'anions': []}


class TestGetRadicalSites:
    """Test radical site detection."""

    def test_methyl_radical(self):
        """Test methyl radical detection."""
        mol = Chem.MolFromSmiles('[CH3]')
        sites = get_radical_sites(mol)
        assert len(sites) == 1
        assert sites[0]['element'] == 'C'
        assert sites[0]['n_electrons'] == 1
        assert sites[0]['radical_type'] == 'monovalent'

    def test_ethyl_radical(self):
        """Test ethyl radical."""
        mol = Chem.MolFromSmiles('C[CH2]')
        sites = get_radical_sites(mol)
        assert len(sites) == 1
        assert sites[0]['element'] == 'C'
        assert sites[0]['radical_type'] == 'monovalent'

    def test_nitrogen_radical(self):
        """Test nitrogen-centered radical."""
        mol = Chem.MolFromSmiles('[NH2]')
        sites = get_radical_sites(mol)
        assert len(sites) == 1
        assert sites[0]['element'] == 'N'

    def test_oxygen_radical(self):
        """Test oxygen radical (hydroxyl)."""
        mol = Chem.MolFromSmiles('[OH]')
        sites = get_radical_sites(mol)
        assert len(sites) == 1
        assert sites[0]['element'] == 'O'

    def test_carbene_divalent(self):
        """Test carbene (divalent radical)."""
        mol = Chem.MolFromSmiles('[CH2]')
        sites = get_radical_sites(mol)
        assert len(sites) == 1
        assert sites[0]['n_electrons'] == 2
        assert sites[0]['radical_type'] == 'divalent'

    def test_neutral_no_radicals(self):
        """Test neutral molecule has no radical sites."""
        mol = Chem.MolFromSmiles('CCO')
        sites = get_radical_sites(mol)
        assert len(sites) == 0

    def test_none_input(self):
        """Test None input returns empty list."""
        sites = get_radical_sites(None)
        assert sites == []


class TestParseSaltFragments:
    """Test salt fragment parsing."""

    def test_sodium_acetate(self):
        """Test sodium acetate parsing."""
        mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        frags = parse_salt_fragments(mol)
        assert len(frags['cations']) == 1
        assert len(frags['anions']) == 1
        assert frags['cations'][0]['charge'] == 1
        assert frags['anions'][0]['charge'] == -1

    def test_potassium_chloride(self):
        """Test potassium chloride parsing."""
        mol = Chem.MolFromSmiles('[K+].[Cl-]')
        frags = parse_salt_fragments(mol)
        assert len(frags['cations']) == 1
        assert len(frags['anions']) == 1
        assert '[K+]' in frags['cations'][0]['smiles']
        assert '[Cl-]' in frags['anions'][0]['smiles']

    def test_calcium_chloride(self):
        """Test calcium chloride (1:2 stoichiometry)."""
        mol = Chem.MolFromSmiles('[Ca+2].[Cl-].[Cl-]')
        frags = parse_salt_fragments(mol)
        assert len(frags['cations']) == 1
        assert len(frags['anions']) == 2
        assert frags['cations'][0]['charge'] == 2

    def test_ammonium_sulfate(self):
        """Test ammonium sulfate (2:1 stoichiometry)."""
        mol = Chem.MolFromSmiles('[NH4+].[NH4+].O=S([O-])([O-])=O')
        frags = parse_salt_fragments(mol)
        assert len(frags['cations']) == 2
        assert len(frags['anions']) == 1

    def test_fragment_smiles(self):
        """Test fragment SMILES are canonical."""
        mol = Chem.MolFromSmiles('[Na+].CC([O-])=O')
        frags = parse_salt_fragments(mol)
        # Check that SMILES are present and non-empty
        assert len(frags['cations'][0]['smiles']) > 0
        assert len(frags['anions'][0]['smiles']) > 0

    def test_fragment_mol_objects(self):
        """Test fragment mol objects are valid."""
        mol = Chem.MolFromSmiles('[Na+].CC([O-])=O')
        frags = parse_salt_fragments(mol)
        # Check mol objects exist and are valid
        assert frags['cations'][0]['mol'] is not None
        assert frags['anions'][0]['mol'] is not None

    def test_single_molecule_no_fragments(self):
        """Test single ion (not a salt) returns single entry."""
        mol = Chem.MolFromSmiles('[NH4+]')
        frags = parse_salt_fragments(mol)
        assert len(frags['cations']) == 1
        assert len(frags['anions']) == 0

    def test_none_input(self):
        """Test None input returns empty dict."""
        frags = parse_salt_fragments(None)
        assert frags == {'cations': [], 'anions': [], 'neutrals': []}


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_radical_cation(self):
        """Test radical cation (both radical and charged)."""
        # Methyl radical cation
        mol = Chem.MolFromSmiles('[CH3+]')
        # This should be detected as ion (charge takes priority for naming)
        species_type = detect_species_type(mol)
        # Note: RDKit treats [CH3+] as a cation without radical electrons
        assert species_type == 'ion'

    def test_complex_salt(self):
        """Test complex salt with multiple ions."""
        mol = Chem.MolFromSmiles('[Na+].[Na+].[O-]C(=O)C([O-])=O')
        frags = parse_salt_fragments(mol)
        assert len(frags['cations']) == 2
        # Oxalate dianion is one fragment
        assert len(frags['anions']) == 1

    def test_hybridization_info(self):
        """Test hybridization is reported correctly."""
        mol = Chem.MolFromSmiles('[NH4+]')
        sites = get_ion_sites(mol)
        assert 'SP3' in sites['cations'][0]['hybridization']

    def test_metal_cation(self):
        """Test transition metal cation."""
        mol = Chem.MolFromSmiles('[Fe+3]')
        sites = get_ion_sites(mol)
        assert len(sites['cations']) == 1
        assert sites['cations'][0]['charge'] == 3
        assert sites['cations'][0]['element'] == 'Fe'
