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


@pytest.mark.unit
class TestP44SeniorAtomParent:
    @pytest.mark.parametrize("smiles,expected", [
        ("C[PH][SiH3]", "methyl(silyl)phosphane"),  # P>Si>C: P is the senior parent atom
        ("[PH2]N", "phosphanamine"),                  # N senior heterane class (P-41 cls 21)
    ])
    def test_senior_atom_parent(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP64RingKetonicSuffixSeniority:
    def test_thiazolidinone_structural(self):
        # P-64.6.2 root heal: the lactam handler was DROPPING the ring S
        # (O=C1CSCN1 -> 'pyrrolidin-2-one', SELF-01-suppressed). The narrowed
        # lactam predicate + the ring-ketone '-one' suffix now name it
        # structurally correctly (OPSIN-RT verified). The PIN form carries the
        # '1,3-' heteroatom locants (see xfail below); this asserts the shipped
        # structural heal (RT-OK).
        assert name_compound("O=C1CSCN1") == "thiazolidin-4-one"

    @pytest.mark.xfail(reason="P-64.6.2 follow-up: retained-name parent "
                              "(thiazolidine) omits heteroatom locants when "
                              "suffixed; PIN needs '1,3-'. Shipped RT-OK "
                              "'thiazolidin-4-one'. Broad locant-citation fix "
                              "for retained multi-heteroatom saturated parents "
                              "deferred.")
    def test_thiazolidinone_exact_pin(self):
        assert name_compound("O=C1CSCN1") == "1,3-thiazolidin-4-one"

    @pytest.mark.xfail(reason="P-64.6.2 sulfanylidene follow-up: the ring C=S "
                              "thiocarbonyl co-occurring with a ring amide C=O "
                              "mis-perceives (N-substituentcyclopentane... "
                              "garbage, SELF-01-suppressed -> unknown). Deep "
                              "multi-path thiocarbonyl+ring-amide defect, "
                              "beyond this row's scope; deferred.")
    def test_thiazolidine_thione_one(self):
        # P-64.6.2: C=O senior to C=S -> ring C=O is the '-one' suffix,
        # ring C=S is the 'sulfanylidene' prefix.
        assert name_compound("S=C1NC(=O)CS1") == "2-sulfanylidene-1,3-thiazolidin-4-one"
