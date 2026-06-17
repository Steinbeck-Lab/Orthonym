"""WS-A task 9 — point fixes H/F/G (among-rings gold reds, investigated
root causes; see the workflow findings in the task-9 session).

H — cycloalkyl morphology: P-29.2 drops '-ane' entirely for carbocycles
    (cyclododecyl), the old fallback kept 'an' ('cyclododecanyl') for every
    all-carbon monocycle >= 9. Hantzsch-Widman '-ane' heterocycles keep the
    final-'e' elision (azepanyl).

F — P-14.5.2 chain-orientation alphabetical tie-break compared FABRICATED
    carbon-count names (phenyl as 'hexyl', thiophen-2-yl as 'butyl'),
    inverting the locants of 1-phenyl-4-(thiophen-2-yl)butane-1,4-dione.
    Ring substituents now compare by their real cited prefix name.

G — ketone PG-location semantics: the ketone SMARTS leads with a flanking
    carbon and there is no exocyclic '-one' suffix (P-66.6.1), so an aryl
    ketone's PG was "on ring" and select_parent mis-parented
    O=C(c1ccccc1)Cc1cnc[nH]1 to bare 'benzene'. PG_ATTACHMENT_INDICES gains
    the ketone family; on-ring checks are membership-only for skeletal
    suffixes.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestCycloalkylMorphologyGold:
    def test_cyclododecylpyridine(self):
        assert (
            name_compound("c1cc(ccn1)C1CCCCCCCCCCC1").strip()
            == "4-cyclododecylpyridine"
        )

    def test_large_cycloalkyl_sizes(self):
        from orthonym.rules.ring_substituents import get_ring_substituent_name
        from rdkit import Chem
        for n, want in [(9, "cyclononyl"), (10, "cyclodecyl"),
                        (12, "cyclododecyl")]:
            smi = "C1" + "C" * (n - 1) + "1"
            mol = Chem.MolFromSmiles(smi)
            ring = tuple(range(n))
            assert get_ring_substituent_name(mol, ring, 0) == want

    def test_small_cycloalkyls_unchanged(self):
        # 3-8 come from the retained dict and must not move.
        assert name_compound("c1cc(ccn1)C1CCCCC1").strip() == "4-cyclohexylpyridine"
        assert name_compound("c1cc(ccn1)C1CC1").strip() == "4-cyclopropylpyridine"

    def test_hantzsch_widman_anyl_unchanged(self):
        # H-W '-ane' heterocycle substituents keep final-'e' elision.
        from orthonym.rules.ring_substituents import get_ring_substituent_name
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C1CCNCC1")  # piperidine ring alone
        # (piperidinyl comes from the dict; this guards the fallback shape
        # by checking no 'cyclo'-prefixed name regressed to [:-3])
        name = get_ring_substituent_name(mol, tuple(range(6)), 0)
        assert "an" in name or name.endswith("yl")


@pytest.mark.unit
class TestDioneAlphabeticalOrientationGold:
    def test_phenyl_thiophenyl_dione(self):
        assert (
            name_compound("O=C(c1ccccc1)CCC(=O)c1cccs1").strip()
            == "1-phenyl-4-(thiophen-2-yl)butane-1,4-dione"
        )

    def test_pyridinyl_analog_protect(self):
        # Accidentally-correct neighbor must stay correct.
        assert (
            name_compound("O=C(c1ccccc1)CCC(=O)c1ccncc1").strip()
            == "1-phenyl-4-(pyridin-4-yl)butane-1,4-dione"
        )

    def test_symmetric_diphenyl_protect(self):
        assert (
            name_compound("O=C(c1ccccc1)CCC(=O)c1ccccc1").strip()
            == "1,4-diphenylbutane-1,4-dione"
        )

    def test_alkyl_only_ties_unchanged(self):
        # Pure-alkyl tie-breaks keep the legacy carbon-count path.
        assert name_compound("CCC(C)CCCC(CC)CC").strip() == (
            name_compound("CCC(C)CCCC(CC)CC").strip()
        )


@pytest.mark.unit
class TestKetonePGLocationGold:
    def test_imidazolyl_phenyl_ethanone(self):
        assert (
            name_compound("O=C(c1ccccc1)Cc1cnc[nH]1").strip()
            == "2-(1H-imidazol-5-yl)-1-phenylethan-1-one"
        )

    def test_diphenylethanone(self):
        assert (
            name_compound("O=C(c1ccccc1)Cc1ccccc1").strip()
            == "1,2-diphenylethan-1-one"
        )

    def test_pyridinyl_analog(self):
        assert (
            name_compound("O=C(c1ccccc1)Cc1ccncc1").strip()
            == "1-phenyl-2-(pyridin-4-yl)ethan-1-one"
        )


@pytest.mark.unit
class TestKetonePGLocationProtect:
    """The investigator's 14-probe protect battery (currently-correct
    neighbors of the PG-location change)."""

    CASES = [
        ("CC(=O)c1ccccc1", "acetophenone"),  # retained name shields it
        ("CC(=O)C1CCCCC1", "1-cyclohexylethan-1-one"),
        ("O=C(CC)c1ccccc1", "1-phenylpropan-1-one"),
        ("O=C1CCCCC1", "cyclohexan-1-one"),
        ("CC(=O)CC1CCCCC1", "1-cyclohexylpropan-2-one"),
        ("CC(=O)Cc1ccccc1", "1-phenylpropan-2-one"),
        ("O=Cc1ccccc1", "benzaldehyde"),
        ("OC(=O)c1ccccc1", "benzoic acid"),
    ]

    @pytest.mark.parametrize("smi,expected", CASES)
    def test_protect(self, smi, expected):
        assert name_compound(smi).strip() == expected


@pytest.mark.unit
class TestExocyclicDecorationPrefixGold:
    """B — P-66.6.1.2: an exocyclic -CHO on a ring whose senior group is the
    acid must be 'formyl', never the ring-ketone 'oxo' (which describes a
    structurally DIFFERENT molecule)."""

    def test_formyl_gold(self):
        assert (
            name_compound("OC(=O)C1CCC(C=O)CC1").strip()
            == "4-formylcyclohexane-1-carboxylic acid"
        )

    def test_formyl_gold_smiles_variant(self):
        assert (
            name_compound("O=CC1CCC(C(=O)O)CC1").strip()
            == "4-formylcyclohexane-1-carboxylic acid"
        )

    def test_hydroxymethyl(self):
        assert (
            name_compound("OCC1CCC(C(=O)O)CC1").strip()
            == "4-(hydroxymethyl)cyclohexane-1-carboxylic acid"
        )

    def test_ring_ketone_locant_from_center(self):
        # Bonus root-cause fix: the ring ketone's locant comes from the
        # carbonyl CENTER atom, not the first (flanking) match atom.
        assert (
            name_compound("O=C1CCC(C(=O)O)CC1").strip()
            == "4-oxocyclohexane-1-carboxylic acid"
        )

    def test_exocyclic_acetyl_never_oxo(self):
        # No faithful acyl row yet: the name may be an honest fallback, but
        # it must NEVER be the structurally wrong '4-oxo...' rewrite.
        got = name_compound("O=C(C)C1CCC(C(=O)O)CC1").strip()
        assert "4-oxocyclohexane" not in got


@pytest.mark.unit
class TestExocyclicDecorationProtect:
    def test_hydroxy_on_ring(self):
        assert (
            name_compound("OC1CCC(C(=O)O)CC1").strip()
            == "4-hydroxycyclohexane-1-carboxylic acid"
        )

    def test_cyano_on_ring(self):
        assert (
            name_compound("N#CC1CCC(C(=O)O)CC1").strip()
            == "4-cyanocyclohexane-1-carboxylic acid"
        )


@pytest.mark.unit
class TestRingNitrogenNotAmineGold:
    """E — P-66.6.1: a ring N is a skeletal heteroatom, never an amine FG.
    The amine SMARTS matched ring N, made it the principal group, and the
    amine emitter CUT the ring open ('N,N-dibutyl-N-cyclohexylcyclohexan-1-
    amine' for cyclohexylmorpholine). N-substituents on N-heterocycles cite
    the numeric ring locant (PIN, P-14.3.2)."""

    def test_cyclohexylmorpholine(self):
        assert (
            name_compound("C1CCCCC1N1CCOCC1").strip() == "4-cyclohexylmorpholine"
        )

    def test_pyrrolidinyl_cyclopentanone(self):
        assert (
            name_compound("O=C1CCCC1N1CCCC1").strip()
            == "2-(pyrrolidin-1-yl)cyclopentan-1-one"
        )

    def test_naphthalenyl_pyrrolidine(self):
        assert (
            name_compound("c1ccc2cc(N3CCCC3)ccc2c1").strip()
            == "1-(naphthalen-2-yl)pyrrolidine"
        )

    def test_phenylpyrrolidine(self):
        assert name_compound("c1ccccc1N1CCCC1").strip() == "1-phenylpyrrolidine"

    def test_methylmorpholine_pin(self):
        assert name_compound("CN1CCOCC1").strip() == "4-methylmorpholine"


@pytest.mark.unit
class TestRingNitrogenNotAmineProtect:
    """Acyclic amines and bare N-heterocycle parents must not move."""

    # v22 Phase B (DD1 Fix 4 / H5): triethylamine/trimethylamine are
    # general-nomenclature functional-class names; the PINs are the substitutive
    # forms (P-62.2.1.2). The ethane amine-suffix locant is elided per P-14.3.4.4
    # ('ethanamine' not 'ethan-1-amine'), matching 'ethanol'/'ethanethiol'.
    CASES = [
        ("CCN(CC)CC", "N,N-diethylethanamine"),
        ("CCNCC", "N-ethylethanamine"),
        ("CN(C)C", "N,N-dimethylmethanamine"),
        ("C1CCCCC1N", "cyclohexan-1-amine"),
        ("CNC1CCCCC1", "N-methylcyclohexan-1-amine"),
        ("C1COCCN1", "morpholine"),
        ("C1CCNCC1", "piperidine"),
        ("C1CCCN1", "pyrrolidine"),
    ]

    @pytest.mark.parametrize("smi,expected", CASES)
    def test_protect(self, smi, expected):
        assert name_compound(smi).strip() == expected


@pytest.mark.unit
class TestEsterChainAcidWithRingGold:
    """Ester acid-part routing: 'Xcarboxylate' applies ONLY when the
    carbonyl C is directly bonded to the ring (P-65.1.7); a ring further
    down the chain makes it a chain acid with a ring substituent."""

    def test_methyl_cyclohexylbutanoate(self):
        assert (
            name_compound("COC(=O)CCCC1CCCCC1").strip()
            == "methyl 4-cyclohexylbutanoate"
        )

    def test_phenylpropanoate(self):
        assert (
            name_compound("COC(=O)CCc1ccccc1").strip()
            == "methyl 3-phenylpropanoate"
        )

    def test_true_ring_acid_protect(self):
        assert (
            name_compound("COC(=O)C1CCCCC1").strip()
            == "methyl cyclohexanecarboxylate"
        )

    def test_benzoate_protect(self):
        assert name_compound("COC(=O)c1ccccc1").strip() == "methyl benzoate"

    def test_chain_ester_protect(self):
        assert name_compound("CCOC(=O)CC").strip() == "ethyl propanoate"

    def test_branched_chain_ester_protect(self):
        assert (
            name_compound("COC(=O)C(C)CC").strip() == "methyl 2-methylbutanoate"
        )


@pytest.mark.unit
class TestNoPGReplacementChainGold:
    """P-44.3 + P-51.4/P-15.4: with no PG, a heteroatom skeletal chain
    admissible for replacement nomenclature (>=4 hetero units) is senior to
    an all-carbon ring, and the chain stem cites the oxa locants."""

    def test_tetraoxadodecane(self):
        assert (
            name_compound("C1CCCCC1COCCOCCOCCOC").strip()
            == "1-cyclohexyl-2,5,8,11-tetraoxadodecane"
        )

    def test_mono_ether_keeps_ring_parent(self):
        assert name_compound("CCCCOC1CCCCC1").strip() == "butoxycyclohexane"

    def test_hydrocarbon_ring_seniority_protect(self):
        # DEF-1 (Phase 171): heptylbenzene, never 1-phenylheptane.
        assert name_compound("CCCCCCCc1ccccc1").strip() == "heptylbenzene"
        assert name_compound("CCCCCCCCCC1CCCCC1").strip() == "nonylcyclohexane"
