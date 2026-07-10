import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P1ChainsBVerify:
    """Rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        # P-16.7.2(d): dialkyl-disulfide PIN is substitutive (BB 27874);
        # no-elision multiplicative parents (di/tri + acetic acid).
        ("C(C)SSCC", "(ethyldisulfanyl)ethane"),
        ("OC(=O)CN(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),
        ("OC(=O)COCC(=O)O", "2,2'-oxydiacetic acid"),
        # P-22.2.2.1.5: alternate HW stems (N-form iridine/etidine/olidine,
        # no-N form irene/olane).
        ("N1CC1", "aziridine"),
        ("O1CC1", "oxirane"),
        ("C1CNC1", "azetidine"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1CCOC1", "oxolane"),
        # P-22.2.5: homogeneous monocyclic parent hydrides (cyclo + chain).
        ("C1CCCCCCC1", "cyclooctane"),
        ("N1NNNN1", "pentazolidine"),
        # P-25.3.2.1.1: monocyclic hydrocarbon parent (cycloheptatriene).
        ("C1=CC=CCC=C1", "cyclohepta-1,3,5-triene"),
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP22MultiplierAElision:
    def test_tetrazine_elides_multiplier_a(self):
        # P-22.2.2.1.2: 'tetra' + 'aza' -> 'tetraza' -> '...tetrazine'
        assert name_compound("N1=NN=NC=C1") == "1,2,3,4-tetrazine"


@pytest.mark.unit
class TestP22AnnuleneMonocyclic:
    def test_cyclodecapentaene(self):
        # P-22.1.2(b) / P-25.3.2.1.1: mancude 10-membered carbocycle standalone PIN
        assert name_compound("C1=CC=CC=CC=CC=C1") == "cyclodeca-1,3,5,7,9-pentaene"


@pytest.mark.unit
class TestP13FunctionalClassDiolDiester:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(=O)OCCOC(C)=O", "ethane-1,2-diyl diacetate"),       # P-13.6.2 (BB 5066)
        ("CCC(=O)OCCOC(=O)CC", "ethane-1,2-diyl dipropanoate"),  # P-13.6.2
    ])
    def test_diol_diester(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP16ParensMultipliedComponent:
    def test_disulfanediyl_di_cyclohexanecarboxylic_acid(self):
        # P-16.3.4(e): functionalized parent hydride WITH LOCANT -> di(cyclohexane-1-carboxylic acid)
        assert name_compound("OC(=O)C1(SSC2(C(=O)O)CCCCC2)CCCCC1") == \
            "1,1'-(disulfanediyl)di(cyclohexane-1-carboxylic acid)"


@pytest.mark.unit
class TestP45SubstitutiveFallthrough:
    # REPRODUCE-FIRST DIVERGENCE (recorded 2026-07-10, HEAD f68ca663):
    # The plan's Task-6 SMILES do NOT match its recorded PINs.
    #   * The "symmetric control" SMILES
    #       OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1Cl
    #     is NOT symmetric: with the bridge O removed the two arms canonicalise
    #     the same (3-chlorobenzoic acid), but the O ATTACHES ortho-to-COOH on
    #     one ring and para-to-COOH on the other. The two units are therefore
    #     placed differently relative to the fixed PG -> P-45.1.1 condition (2)
    #     ("symmetrically substituted") is NOT met, so multiplicative correctly
    #     DECLINES. OPSIN parses the plan's stated PIN
    #     "4,4'-oxybis(2-chlorobenzoic acid)" to a DIFFERENT molecule (the
    #     truly-symmetric witness, which the namer already handles -> see
    #     test_witness_true_symmetric_works below).
    #   * Both plan SMILES thus require the SUBSTITUTIVE biaryl-ether-diacid
    #     decomposition (owned by the parent-selection cluster), not the
    #     multiplicative path. multiplicative NEVER emits a wrong name here.
    # Disposition: FAIL_CLOSED; both full-molecule assertions ship xfail.

    @pytest.mark.xfail(reason="P-45.1.1: multiplicative correctly declines (units "
                              "not symmetrically substituted relative to the PG); "
                              "the substitutive biaryl-ether-diacid decomposition "
                              "is owned by the parent-selection cluster (follow-up). "
                              "multiplicative never emits a wrong name -- fail-closed.")
    def test_asymmetric_falls_through_to_substitutive(self):
        assert name_compound("OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1") == \
            "4-(4-carboxyphenoxy)-2-chlorobenzoic acid"

    @pytest.mark.xfail(reason="P-45.1.1: the plan's 'symmetric control' SMILES is "
                              "actually asymmetric (O attaches ortho vs para to COOH "
                              "on the two rings); multiplicative correctly declines. "
                              "Substitutive fallthrough owned by parent-selection "
                              "cluster (follow-up). fail-closed.")
    def test_symmetric_stays_multiplicative(self):
        assert name_compound("OC(=O)c1cc(Cl)ccc1Oc1ccc(C(=O)O)cc1Cl") == \
            "4,4'-oxybis(2-chlorobenzoic acid)"

    def test_multiplicative_declines_asymmetric(self):
        # The load-bearing fail-closed invariant: multiplicative returns None for
        # the plan's asymmetric target (no wrong name emitted). P-45.1.1.
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
        # P-45.1.1 productive branch; protects the direction-safe path.
        assert name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)c(Cl)c2)cc1Cl") == \
            "4,4'-oxybis(2-chlorobenzoic acid)"

    def test_no_cl_symmetric_works(self):
        # Control: no-Cl symmetric oxydibenzoic acid unchanged.
        assert name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1") == \
            "4,4'-oxydibenzoic acid"
