"""Wave-2 completion pass C2 — PS/NC investigation-group builds.

All expected names OPSIN-RT verified. Covers:
  * organometallic silane ligand extension: O-attached (R-oxy) + branched
    alkyl ligands via the substituent chokepoint, with join enclosing marks
     Si-senior parents).
  * decorated-aryloxy recognizer + locant-blind decomposition-join decline
     2-(3-cyanophenoxy)-4-(propan-2-yl)benzonitrile BB verbatim).
  * isothiocyanate/isocyanate attach-via-N guard + enclosing marks
     (isothiocyanatomethyl)benzene BB verbatim).
  * O-attach recursion in name_substituent_fragment (one-level nested
    alkoxy; deeper nesting fails closed).
  * carbonic-family composite-N exact-SMILES rows /.2.2/.3.3).
  * isocyanide prefix-only principal isocyanomethane).
  * nitroso N-branch recognizer N-methyl-N-nitrosourea BB verbatim).
  * mixed primary+secondary diamine union.
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
    # (the Blue Book, "For mononuclear parent hydrides with two
    # or more substituents the first cited substituent never has enclosing marks
    # unless it includes a locant. The second and further substituents are each
    # enclosed with parentheses even for simple substituents. When the simple
    # substituent groups are accompanied by multiplicative prefixes such as 'di'
    # and 'tri', the multiplicative prefixes are not included in the parentheses."):
    # 'trichloro(iodomethyl)silane (PIN)':25870, '[2-(1,3-dioxolan-2-yl)ethyl]
    # tri(methyl)silane (PIN)':35326, 'tert-butyldi(methyl)phosphane (PIN)':16286.
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)(C)[Si](C)(C)OCC1CO1",
         "tert-butyldi(methyl)(oxiranylmethoxy)silane"),
        ("[Si](C)(C)(C)OCC1CO1", "trimethyl(oxiranylmethoxy)silane"),
        ("CO[Si](C)(C)C", "methoxytri(methyl)silane"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CC[Si](C)(C)C", "ethyltri(methyl)silane"),   #:7272
        ("C[Si](C)(C)C", "tetramethylsilane"),
        ("C[Si](C)(C)O", "trimethylsilanol"),   # G2 suffix diversion
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
        ("COc1ccccc1", "anisole"),     # a review RISK 7: bare anisole IS the PIN (the Blue Book)
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
        # (the Blue Book) 'C6H5-NCS isothiocyanatobenzene (PIN) phenyl
        # isothiocyanate': the aryl form is substitutive too (leads L7 / 43c).
        ("c1ccc(cc1)N=C=S", "isothiocyanatobenzene"),
        ("CCN=C=S", "isothiocyanatoethane"),
        ("CCN=C=O", "isocyanatoethane"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestNestedAlkoxy:
    @pytest.mark.parametrize("smiles,expected", [
        # Slice S3, Task S3.1b: a substituted contracted alkoxy is a compound prefix and takes
        # parentheses the Blue Book,:15762,:7232;
        # '1-(chloromethoxy)-4-nitrobenzene (PIN)':27711), as the depth-two name below and
        # the gold '[(methoxymethoxy)methyl]benzene' already enclose it.
        ("COCOCC", "(methoxymethoxy)ethane"),
        # (the Blue Book): the compound prefix is enclosed
        # ('5-(methoxymethyl)oxolan-2-yl',:54971)
        ("CCCC(CCC)OCOC", "4-(methoxymethoxy)heptane"),
        ("CCCCCCCCC(OC)CCCCCCCC", "9-methoxyheptadecane"),
    ])
    def test_one_level_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_depth_two_heals(self, _validity_gate_on):
        # Wave2-P1ChainsA Task 12 /: the get_alkoxy_prefix
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
        # Monosubstituted urea omits the italic-N locant,:2943);
        # the disubstituted form keeps both (:33327).
        ("CNC(N)=O", "methylurea"),
        ("CNC(=O)NC", "N,N'-dimethylurea"),
    ])
    def test_urea_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestMixedDiamines:
    @pytest.mark.parametrize("smiles,expected", [
        # a phase Thread A, the Blue Book) + the low-locant
        # tie-break /: a single N-substituent on a
        # symmetric terminal diamine takes numeric-superscript N1 (the reversal
        # is a legal tie-break that gives the cited prefix the lowest locant),
        # not the non-lowest N2/N3 the old numbering left. Each still OPSIN
        # round-trips to the same molecule (N1 == N2 by symmetry).
        ("CNCCCN", "N1-methylpropane-1,3-diamine"),
        ("CNCCN", "N1-methylethane-1,2-diamine"),
        ("CN(C)CCN", "N1,N1-dimethylethane-1,2-diamine"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("NCCCN", "propane-1,3-diamine"),
        ("NCCN", "ethane-1,2-diamine"),
        # a phase Thread A, the Blue Book): simple
        # polyamines use numeric-superscript italic-N locants, not bare primes.
        ("CNCCNC", "N1,N2-dimethylethane-1,2-diamine"),
        ("CCNCCCNC", "N1-ethyl-N3-methylpropane-1,3-diamine"),
        ("CN(C)CCN(C)C", "N1,N1,N2,N2-tetramethylethane-1,2-diamine"),
        ("CCN", "ethanamine"),
        ("CNC", "N-methylmethanamine"),
        ("NCCO", "2-aminoethan-1-ol"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_complex_polyamine_now_heals(self, _validity_gate_on):
        # parent-choice cascade) HEALED in Wave-3: the
        # polyamine parent-selection now emits the exact Blue Book PIN
        # (BBv2 L26381 'N1-(aminomethyl)ethane-1,2-diamine (PIN)'); OPSIN-RT
        # clean. Was previously fail-closed (deferred) — no longer.
        assert name_compound("NCNCCN") == "N1-(aminomethyl)ethane-1,2-diamine"
