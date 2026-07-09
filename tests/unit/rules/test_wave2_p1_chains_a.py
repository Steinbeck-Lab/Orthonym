import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.rules.radicals import name_radical


@pytest.mark.unit
class TestWave2P1ChainsAVerify:
    """Rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        ("[PH3]", "phosphane"),                              # P-52.1.1
        ("[SnH3]O[SnH2]O[SnH3]", "tristannoxane"),           # P-52.1.3
        ("N1CCOCCCCCCCC1", "1-oxa-4-azacyclododecane"),      # P-22.2.3.2.3
        ("C[C@H](Cl)[C@@H](Cl)C", "(2S,3S)-2,3-dichlorobutane"),  # P-44.4.1.12
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP22SingleHeteroatomLocantElision:
    def test_thiacyclododecane_omits_locant_1(self):
        # P-22.2.3.2.1: single heteroatom -> locant '1' omitted
        assert name_compound("S1CCCCCCCCCCC1") == "thiacyclododecane"


@pytest.mark.unit
class TestP15ReplacementMorpholine:
    @pytest.mark.parametrize("smiles,expected", [
        ("S1CCNCC1", "thiomorpholine"),      # S-for-O morpholine (PIN)
        ("O1CCSCC1", "1,4-oxathiane"),        # the ring the OLD key wrongly named thiomorpholine
        ("[Se]1CCNCC1", "selenomorpholine"),  # Se-for-O morpholine (PIN)
    ])
    def test_chalcogen_morpholine(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP54HWPhosphorusPartialSat:
    def test_dihydrophosphole(self):
        # P-54.4.1: 2,3-dihydro-1H-phosphole (PIN)
        # NOTE: the plan's evidence SMILES "C1=CCP1" is a 4-membered ring
        # (phosphete-derived); the actual 2,3-dihydro-1H-phosphole is the
        # 5-membered "P1CCC=C1" (OPSIN-verified). Corrected here per
        # reproduce-first CODE-level divergence.
        assert name_compound("P1CCC=C1") == "2,3-dihydro-1H-phosphole"


@pytest.mark.unit
class TestP41RadicalMostSenior:
    def test_carboxyethyl_radical(self):
        # P-41 cls 1: radical senior to acid -> acid demoted to 'carboxy' prefix,
        # free valence is C-1 of the ethyl point-of-attachment chain.
        mol = Chem.MolFromSmiles("[CH2]CC(=O)O")
        assert name_radical(mol) == "2-carboxyethyl"
