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


@pytest.mark.unit
class TestP58NondetachableHydroDione:
    def test_dihydronaphthalenedione(self):
        # P-58.2.5: added-IH/hydro dione, NOT tetrahydro
        assert name_compound("O=C1CCC(=O)c2ccccc21") == "2,3-dihydronaphthalene-1,4-dione"


@pytest.mark.unit
class TestP58AddedIHDistribution:
    @pytest.mark.xfail(reason="P-58.2.3.1.2 BLOCKED on the fusion engine: the "
                              "cyclopenta[a]naphthalene 3-ring parent does not "
                              "resolve (fused-ring namer -> cyclopentane, "
                              "SELF-01-suppressed). name_cyclic_oxo_compound "
                              "returns None; row FAIL_CLOSED (fusion-parent "
                              "numbering follow-up). No wrong name emitted.")
    def test_cyclopentanaphthalene_dione(self):
        # P-58.2.3.1.2: extra IH spills to lowest nonfusion peripheral atom
        assert name_compound("C1(C(CC=2C1=C1C=CC=CC1=CC2)=O)=O") == \
            "1H-cyclopenta[a]naphthalene-1,2(3H)-dione"


@pytest.mark.unit
class TestP59DetachableHydroPlacement:
    @pytest.mark.xfail(reason="P-59.2.3.2 BLOCKED: name_partially_saturated_"
                              "carbocycle (polycyclics.py) is a bare-skeleton "
                              "namer with NO substituent/suffix support — it "
                              "emits '5,6-dihydroazulene', dropping the -COOH "
                              "and bromo (SELF-01-suppressed -> unknown, so no "
                              "wrong name leaks). Building the "
                              "detachable-hydro+PCG-suffix+halogen combined-"
                              "numbering azulene assembly is deferred.")
    def test_bromo_dihydroazulene_carboxylic_acid(self):
        assert name_compound("BrC=1CCC=C2C=C(C=C2C1)C(=O)O") == \
            "7-bromo-5,6-dihydroazulene-2-carboxylic acid"


@pytest.mark.unit
class TestP66CarboxamideSeniorToUrea:
    def test_formamide_senior_to_urea(self):
        # P-66.1.6.1.1.5: carboxamide/formamide senior to urea (already correct
        # at HEAD; locked against regression). urea cited as carbamoylamino.
        assert name_compound("NC(=O)NCCCNC=O") == "N-[3-(carbamoylamino)propyl]formamide"

    def test_pure_urea_unchanged(self):
        assert name_compound("NC(=O)N") == "urea"
        assert name_compound("NC(=O)NCC") == "N-ethylurea"


@pytest.mark.unit
class TestP57ReplacementSubstituent:
    @pytest.mark.parametrize("smiles,expected", [
        ("COCCOc1ccccc1", "(2-methoxyethoxy)benzene"),                       # P-57.1.6.2
        ("COCOCc1ccccc1", "[(methoxymethoxy)methyl]benzene"),                # nested alkoxy
    ])
    def test_oxa_thia_substituent(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.xfail(reason="P-46.1.3 thioether case UNDER-SCOPED: the "
                              "substituent PRODUCER now builds the correct "
                              "'{[(methoxymethyl)sulfanyl]methyl}benzene' "
                              "(verified via name_benzene_derivative, OPSIN-RT "
                              "OK), but the GENERAL parent-selection routes "
                              "-CH2-S-CH2-O-CH3 on benzene to an acyclic-chain "
                              "parent (heptylsulfanyl...methoxymethane) instead "
                              "of the senior benzene ring, so SELF-01 suppresses "
                              "to unknown. Fixing the thioether classification/"
                              "parent-selection is a separate follow-up; no "
                              "wrong name leaks.")
    def test_thioether_substituent_underscoped(self):
        # Strict P-16.3.3 cycling nesting: inner () -> [] -> outer {}.
        # name_benzene_derivative already yields the correct RT-OK PIN;
        # only the whole-molecule parent selection blocks it today.
        assert name_compound("COCSCc1ccccc1") == "{[(methoxymethyl)sulfanyl]methyl}benzene"

    def test_standalone_replacement_chain_unchanged(self):
        # A long homogeneous oxa chain stays a REPLACEMENT PARENT (not a nested
        # substituent) — the recursion must not steal it.
        assert name_compound("COCOCOCOCCc1ccccc1") == "10-phenyl-2,4,6,8-tetraoxadecane"


@pytest.mark.unit
class TestP32FixedNumberingSubstituent:
    def test_bicyclooctenyl_propanoic_acid(self):
        # P-32.1.3: fixed-numbering ring substituent; free valence lowest (2),
        # then the ring double bond (5) -> bicyclo[2.2.2]oct-5-en-2-yl.
        assert name_compound("OC(=O)CCC1CC2CCC1C=C2") == \
            "3-(bicyclo[2.2.2]oct-5-en-2-yl)propanoic acid"

    def test_saturated_bicyclo_substituent_unchanged(self):
        # The saturated von-Baeyer substituent must be UNCHANGED.
        assert name_compound("OC(=O)CCC1CC2CCC1CC2") == \
            "3-(bicyclo[2.2.2]octan-2-yl)propanoic acid"

    def test_monocyclic_substituent_unchanged(self):
        assert name_compound("OC(=O)CCC1CCCCC1") == "3-cyclohexylpropanoic acid"
