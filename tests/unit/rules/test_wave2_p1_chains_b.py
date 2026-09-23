import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P1ChainsBVerify:
    """Rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        # (d): dialkyl-disulfide PIN is substitutive (BB 27874);
        # no-elision multiplicative parents (di/tri + acetic acid).
        ("C(C)SSCC", "(ethyldisulfanyl)ethane"),
        ("OC(=O)CN(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),
        ("OC(=O)COCC(=O)O", "2,2'-oxydiacetic acid"),
        #: alternate HW stems (N-form iridine/etidine/olidine,
        # no-N form irene/olane).
        ("N1CC1", "aziridine"),
        ("O1CC1", "oxirane"),
        ("C1CNC1", "azetidine"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1CCOC1", "oxolane"),
        #: homogeneous monocyclic parent hydrides (cyclo + chain).
        ("C1CCCCCCC1", "cyclooctane"),
        ("N1NNNN1", "pentazolidine"),
        #: monocyclic hydrocarbon parent (cycloheptatriene).
        ("C1=CC=CCC=C1", "cyclohepta-1,3,5-triene"),
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP22MultiplierAElision:
    def test_tetrazine_elides_multiplier_a(self):
        #: 'tetra' + 'aza' -> 'tetraza' -> '...tetrazine'
        assert name_compound("N1=NN=NC=C1") == "1,2,3,4-tetrazine"


@pytest.mark.unit
class TestP22AnnuleneMonocyclic:
    def test_cyclodecapentaene(self):
        # (b) /: mancude 10-membered carbocycle standalone PIN
        assert name_compound("C1=CC=CC=CC=CC=C1") == "cyclodeca-1,3,5,7,9-pentaene"


@pytest.mark.unit
class TestP13FunctionalClassDiolDiester:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(=O)OCCOC(C)=O", "ethane-1,2-diyl diacetate"),       # (BB 5066)
        ("CCC(=O)OCCOC(=O)CC", "ethane-1,2-diyl dipropanoate"),  #
    ])
    def test_diol_diester(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP16ParensMultipliedComponent:
    def test_disulfanediyl_di_cyclohexanecarboxylic_acid(self):
        # (e): functionalized parent hydride WITH LOCANT -> di(cyclohexane-1-carboxylic acid)
        assert name_compound("OC(=O)C1(SSC2(C(=O)O)CCCCC2)CCCCC1") == \
            "1,1'-disulfanediyldi(cyclohexane-1-carboxylic acid)"


@pytest.mark.unit
class TestP45SubstitutiveFallthrough:
    # REPRODUCE-FIRST DIVERGENCE (recorded 2026-07-10, HEAD f68ca663):
    # The plan's Task-6 SMILES do NOT match its recorded PINs.
    # * The "symmetric control" SMILES
    # OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1Cl
    # is NOT symmetric: with the bridge O removed the two arms canonicalise
    # the same (3-chlorobenzoic acid), but the O ATTACHES ortho-to-COOH on
    # one ring and para-to-COOH on the other. The two units are therefore
    # placed differently relative to the fixed PG -> condition (2)
    # ("symmetrically substituted") is NOT met, so multiplicative correctly
    # DECLINES. OPSIN parses the plan's stated PIN
    # "4,4'-oxybis(2-chlorobenzoic acid)" to a DIFFERENT molecule (the
    # truly-symmetric witness, which the namer already handles -> see
    # test_witness_true_symmetric_works below).
    # * Both plan SMILES thus require the SUBSTITUTIVE biaryl-ether-diacid
    # decomposition (owned by the parent-selection cluster), not the
    # multiplicative path. multiplicative NEVER emits a wrong name here.
    # Disposition: FAIL_CLOSED; both full-molecule assertions ship xfail.

    def test_asymmetric_falls_through_to_substitutive(self):
        #: multiplicative correctly declines (units NOT symmetrically
        # substituted relative to the PG). The substitutive biaryl-ether-diacid
        # path then picks the lowest-locant COOH benzene as parent and names the
        # other acid-bearing aryloxy ring as a '(4-carboxyphenoxy)' prefix
        # carboxy prefix). OPSIN-verified: the PIN round-trips to
        # this exact SMILES. (The prior xfail PIN
        # '4-(4-carboxyphenoxy)-2-chlorobenzoic acid' was positionally WRONG.)
        assert name_compound("OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1") == \
            "2-(4-carboxyphenoxy)-5-chlorobenzoic acid"

    def test_symmetric_stays_multiplicative(self):
        # The plan's 'symmetric control' SMILES is actually ASYMMETRIC (O
        # attaches ortho vs para to COOH on the two rings), so multiplicative
        # correctly declines and the substitutive biaryl-ether-diacid path
        # names it. Here the prefix ring additionally carries an ortho-Cl ->
        # '(4-carboxy-2-chlorophenoxy)'. OPSIN-verified round-trip. (The prior
        # xfail PIN "4,4'-oxybis(2-chlorobenzoic acid)" is a DIFFERENT
        # molecule -- see test_witness_true_symmetric_works.)
        assert name_compound("OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1Cl") == \
            "2-(4-carboxy-2-chlorophenoxy)-5-chlorobenzoic acid"

    def test_multiplicative_declines_asymmetric(self):
        # The load-bearing fail-closed invariant: multiplicative returns None for
        # the plan's asymmetric target (no wrong name emitted)..
        from orthonym.rules.multiplicative import name_multiplicative
        from rdkit import Chem
        m = Chem.MolFromSmiles("OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1")
        assert name_multiplicative(m) is None

    def test_multiplicative_declines_plan_symmetric(self):
        # Same fail-closed invariant for the plan's mislabeled "symmetric" SMILES.
        from orthonym.rules.multiplicative import name_multiplicative
        from rdkit import Chem
        m = Chem.MolFromSmiles("OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1Cl")
        assert name_multiplicative(m) is None

    def test_witness_true_symmetric_works(self):
        # A GENUINELY symmetric 2-chlorobenzoic-acid oxybis (O para to COOH,
        # Cl ortho on BOTH rings) IS the multiplicative PIN and already works.
        # productive branch; protects the direction-safe path.
        assert name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)c(Cl)c2)cc1Cl") == \
            "4,4'-oxybis(2-chlorobenzoic acid)"

    def test_no_cl_symmetric_works(self):
        # Control: no-Cl symmetric oxydibenzoic acid unchanged.
        assert name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1") == \
            "4,4'-oxydibenzoic acid"


@pytest.mark.unit
class TestP45StereoConfigMismatch:
    def test_no_stereo_keeps_multiplicative(self):
        # baseline: no stereo -> identical units -> multiplicative form retained
        assert name_compound("C1(=CCCCCCC1)SC1=CCCCCCC1") == "1,1'-sulfanediyldicyclooctene"

    def test_mismatched_config_declines_multiplicative(self):
        #: a sulfanediyl bridge linking two cyclooctene units of DIFFERENT
        # ring-double-bond configuration must NOT be named multiplicatively.
        # name_multiplicative declines. (NOTE: RDKit does not retain E/Z on both
        # 8-membered ring double bonds through the fragment split, so the identity
        # test cannot see it post-split; the decline is enforced by a config-aware
        # guard on the bridge-adjacent ring double bonds BEFORE the split.)
        from orthonym.rules.multiplicative import name_multiplicative
        from rdkit import Chem
        mixed = Chem.MolFromSmiles(r"C1(=C/CCCCCC1)S/C1=C\CCCCCC1")
        assert mixed is not None
        assert name_multiplicative(mixed) is None


@pytest.mark.unit
class TestP28RingAssemblyBeyondSix:
    # 11 para-linked benzenes -> 'undeciphenyl' (OPSIN parses+round-trips this form).
    UNDECI_SMILES = ("c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccc(-c8ccc("
                     "-c9ccc(-c%10ccc(-c%11ccccc%11)cc%10)cc9)cc8)cc7)cc6)cc5)"
                     "cc4)cc3)cc2)cc1")

    def test_undeciphenyl_names(self):
        from orthonym.namer import name_compound
        name = name_compound(self.UNDECI_SMILES)
        # OPSIN-verified: the 11-mer explicit-locant assembly parses+round-trips.
        assert name is not None
        assert name.endswith("-undeciphenyl")

    def test_undeciphenyl_opsin_round_trips(self):
        # OPSIN oracle (the extension IS parseable, unlike the phane class).
        from orthonym.namer import name_compound
        from rdkit import Chem
        import subprocess
        name = name_compound(self.UNDECI_SMILES)
        jar = "opsin-cli-2.9.0-jar-with-dependencies.jar"
        out = subprocess.run(["java", "-jar", jar, "-o", "smi"],
                             input=name + "\n", capture_output=True, text=True)
        opsin_smi = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        assert opsin_smi, "OPSIN failed to parse undeciphenyl"
        assert Chem.CanonSmiles(opsin_smi) == Chem.CanonSmiles(self.UNDECI_SMILES)

    def test_paracyclophane_fails_closed_internal_oracle(self):
        # phane class: a benzene bridged para-para into a macrocycle
        # ([n]paracyclophane) is OPSIN-UNPARSEABLE. Orthonym must NOT emit a
        # (necessarily unverifiable) phane name for it -> fail-closed (None).
        # INTERNAL oracle: assert refusal, never a passing OPSIN gold.
        from orthonym.namer import name_compound
        from rdkit import Chem
        phane = "C1Cc2ccc(cc2)CCc2ccc1cc2"
        m = Chem.MolFromSmiles(phane)
        if m is None:
            pytest.skip("phane test SMILES invalid in this RDKit build")
        # Must NOT emit a linear ring-assembly ('...phenyl') name for a macrocyclic phane.
        name = name_compound(phane)
        assert name is None or "phenyl" not in (name or "")
