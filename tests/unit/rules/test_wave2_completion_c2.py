"""Wave-2 completion pass C2 — PS/NC investigation-group builds.

All expected names OPSIN-RT verified. Covers:
  * organometallic silane ligand extension: O-attached (R-oxy) + branched
    alkyl ligands via the substituent chokepoint, with join enclosing marks
    (P-44.1.2 Si-senior parents).
  * decorated-aryloxy recognizer + locant-blind decomposition-join decline
    (P-45.2.1 2-(3-cyanophenoxy)-4-(propan-2-yl)benzonitrile BB verbatim).
  * isothiocyanate/isocyanate attach-via-N guard + enclosing marks
    (P-15.2.1.1 (isothiocyanatomethyl)benzene BB verbatim).
  * O-attach recursion in name_substituent_fragment (one-level nested
    alkoxy; deeper nesting fails closed).
  * carbonic-family composite-N exact-SMILES rows (P-66.4.1.5/.2.2/.3.3).
  * isocyanide prefix-only principal (P-61.9 isocyanomethane).
  * nitroso N-branch recognizer (P-61.5.2 N-methyl-N-nitrosourea BB verbatim).
  * mixed primary+secondary diamine union (P-62.2.4.1.2).
"""

import pytest

from orthonym.namer import name_compound


@pytest.fixture()
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
class TestSilaneLigands:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)(C)[Si](C)(C)OCC1CO1",
         "(tert-butyl)di(methyl)(oxiranylmethoxy)silane"),
        ("[Si](C)(C)(C)OCC1CO1", "tri(methyl)(oxiranylmethoxy)silane"),
        ("CO[Si](C)(C)C", "(methoxy)tri(methyl)silane"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CC[Si](C)(C)C", "ethyltrimethylsilane"),
        ("C[Si](C)(C)C", "tetramethylsilane"),
        ("C[Si](C)(C)O", "trimethylsilanol"),   # G2 COV-02 suffix diversion
        ("CC[Sn](CC)(CC)CC", "tetraethylstannane"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestAryloxyAndEtherJoins:
    def test_dicyano_phenoxy_bb_verbatim(self):
        assert (name_compound("N#Cc1ccc(C(C)C)cc1Oc1cccc(C#N)c1")
                == "2-(3-cyanophenoxy)-4-(propan-2-yl)benzonitrile")

    def test_locanted_ether_on_positional_parent(self):
        # The decomposition join declined; the benzene handler emits the
        # locanted form (was unlocanted '2-methoxybenzonitrile'-class bug).
        assert name_compound("N#Cc1ccccc1OC") == "2-methoxybenzonitrile"

    @pytest.mark.parametrize("smiles,expected", [
        ("COc1ccccc1", "methoxybenzene"),     # RET-01 policy gold path
        ("CCOc1ccccc1", "ethoxybenzene"),
        ("COC1CCCCC1", "methoxycyclohexane"),  # position-invariant cycloalkane
    ])
    def test_invariant_parents_still_join(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestIsothiocyanatoMethyl:
    def test_benzyl_isothiocyanate_bb_verbatim(self):
        assert (name_compound("C(c1ccccc1)N=C=S")
                == "(isothiocyanatomethyl)benzene")

    def test_benzyl_isocyanate(self):
        assert (name_compound("C(c1ccccc1)N=C=O")
                == "(isocyanatomethyl)benzene")

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc(cc1)N=C=S", "phenyl isothiocyanate"),  # aryl keeps FC form
        ("CCN=C=S", "isothiocyanatoethane"),
        ("CCN=C=O", "isocyanatoethane"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestNestedAlkoxy:
    @pytest.mark.parametrize("smiles,expected", [
        ("COCOCC", "methoxymethoxyethane"),
        ("CCCC(CCC)OCOC", "4-methoxymethoxyheptane"),
        ("CCCCCCCCC(OC)CCCCCCCC", "9-methoxyheptadecane"),
    ])
    def test_one_level_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_depth_two_heals(self, _validity_gate_on):
        # Wave2-P1ChainsA Task 12 (P-57.1.6.2 / P-46.1.3): the get_alkoxy_prefix
        # + _name_ether_substituted_chain recursion now expresses depth-2 nesting
        # for an ether-bearing R side. -O-CH2-O-CH2-O-CH3 on heptadecane-C9 ->
        # '9-[(methoxymethoxy)methoxy]heptadecane' (OPSIN-RT verified). Was
        # fail-closed before Task 12; the correct name now round-trips.
        assert name_compound("C(OCOCOC)(CCCCCCCC)CCCCCCCC") == \
            "9-[(methoxymethoxy)methoxy]heptadecane"


@pytest.mark.unit
class TestCompositeNAdditives:
    @pytest.mark.parametrize("smiles,expected", [
        ("N=C(N)SSC(N)=N", "carbamimidic dithioperoxyanhydride"),
        ("C(N)(N)=NN", "carbonohydrazonic diamide"),
        ("N(N)C(NN)=NN", "hydrazinecarbohydrazonohydrazide"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestIsocyanideParents:
    @pytest.mark.parametrize("smiles,expected", [
        ("[C-]#[N+]C", "isocyanomethane"),
        ("[C-]#[N+]CC", "isocyanoethane"),
        ("CC(C)[N+]#[C-]", "2-isocyanopropane"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_nitrile_untouched(self):
        assert name_compound("N#CC") == "acetonitrile"


@pytest.mark.unit
class TestNitrosoUrea:
    def test_bb_verbatim(self):
        assert name_compound("CN(N=O)C(N)=O") == "N-methyl-N-nitrosourea"

    def test_dimethyl_sibling(self):
        assert (name_compound("O=NN(C)C(=O)NC")
                == "N,N'-dimethyl-N-nitrosourea")

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(N)=O", "urea"),
        ("CNC(N)=O", "N-methylurea"),
        ("CNC(=O)NC", "N,N'-dimethylurea"),
    ])
    def test_urea_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestMixedDiamines:
    @pytest.mark.parametrize("smiles,expected", [
        ("CNCCCN", "N-methylpropane-1,3-diamine"),
        ("CNCCN", "N-methylethane-1,2-diamine"),
        ("CN(C)CCN", "N,N-dimethylethane-1,2-diamine"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("NCCCN", "propane-1,3-diamine"),
        ("NCCN", "ethane-1,2-diamine"),
        ("CNCCNC", "N,N'-dimethylethane-1,2-diamine"),
        ("CCNCCCNC", "N-ethyl-N'-methylpropane-1,3-diamine"),
        ("CN(C)CCN(C)C", "N,N,N',N'-tetramethylethane-1,2-diamine"),
        ("CCN", "ethanamine"),
        ("CNC", "N-methylmethanamine"),
        ("NCCO", "2-aminoethan-1-ol"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_complex_polyamine_fails_closed(self, _validity_gate_on):
        # NC-10 (P-62.2.4.1.3 parent-choice cascade) is deferred — the
        # jar-armed gate must suppress the raw candidate (never the
        # double-counted wrong name reaching users).
        assert "unknown" in name_compound("NCNCCN")
