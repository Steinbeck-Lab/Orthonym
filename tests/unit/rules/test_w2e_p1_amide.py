"""Wave-2 plan P1AM (2026-07-09) — C1 amide/amidine/polyfunctional rows.

Every expected PIN in this file was OPSIN-2.9.0-parse-verified and
canonical-matched against the evidence SMILES during planning.
"""

import pytest

from orthonym.namer import name_compound
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "NC(=N)N(C)C(=N)N",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



@pytest.fixture()
def _validity_gate_on(monkeypatch):
    """Turn the OPSIN validity gate  back ON — conftest disables it
    for unit-test speed. Required to assert end-to-end fail-closed behavior."""
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
class TestT2CarbonicFamilyParents:
    """ (BB 32675) + (BB 38623: 'The systematic
    name is the preferred IUPAC name') + (BB 34480)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NNC(=O)N", "hydrazinecarboxamide"),          # was 'semicarbazide'
        ("NNC(=N)NN", "hydrazinecarboximidohydrazide"),  # was unknown
        ("NC(=N)NN", "hydrazinecarboximidamide"),      # was unknown
    ])
    def test_pins(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    def test_protect_carbonohydrazonic_diamide_unchanged(self):
        # existing exact row in the same dict must keep working
        assert name_compound("NC(=NN)N") == "carbonohydrazonic diamide"


@pytest.mark.unit
class TestT3AromaticCarboximidamideProtect:
    """P-66.4.1.1 (BB 34173/34393). Already healed at HEAD — protect pins."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=N)c1ccccc1", "benzenecarboximidamide"),
        ("C(=N)(Nc1ccccc1)c1ccccc1", "N-phenylbenzenecarboximidamide"),
    ])
    def test_protect(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT4SulfonimidamideRingAndSe:
    """ (BB 34173): S/Se/Te imidamide suffixes; ring parent form."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N=S(N)(=O)c1ccccc1", "benzenesulfonimidamide"),
        ("C[Se](=N)N", "methaneseleninimidamide"),
        ("C[Se](=N)(=O)N", "methaneselenonimidamide"),
    ])
    def test_heals(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=N)(=O)N", "methanesulfonimidamide"),
        ("CS(=N)N", "methanesulfinimidamide"),
    ])
    def test_protect_chain_s_forms(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT5Biguanide:
    """ (BB 34298): condensed guanidines are named as diamides
    of imidodicarbonimidic acid; 'biguanide' no longer recommended."""

    def test_bare_biguanide_pin(self):
        assert name_compound("NC(=N)NC(=N)N") == "imidodicarbonimidic diamide"

    def test_substituted_keeps_rt_valid_general_name(self):
        # The Blue Book's own row for this molecule (the Blue Book):
        # 'H2N-C(=NH)-NH-C(=N-CH2-CH3)-N(C6H5)2 N'1-ethyl-N1,N1-diphenyl
        # imidodicarbonimidic diamide (PIN)'; (:34298) "The names
        # biguanide, triguanide, etc., are no longer recommended." The
        # substituted diamide is built now (composer._condensed_guanidine_name);
        # it used to ship the guanidine-form general name at pin_verified
        # (TRIAGE g8 C20). OPSIN 2.9.0 full-InChIKey and canonical SMILES EXACT.
        assert name_compound("CCN=C(NC(N)=N)N(c1ccccc1)c1ccccc1") == \
            "N'1-ethyl-N1,N1-diphenylimidodicarbonimidic diamide"

    @pytest.mark.parametrize("smiles,expected", [
        ("CNC(=N)NC(=N)N", "N1-methylimidodicarbonimidic diamide"),
        ("CN(C)C(=N)NC(=N)N", "N1,N1-dimethylimidodicarbonimidic diamide"),
        ("N=C(N)NC(=N)Nc1ccc(Cl)cc1",
         "N1-(4-chlorophenyl)imidodicarbonimidic diamide"),
    ])
    def test_substituted_condensed_guanidine_pin(self, smiles, expected):
        # Same shape and locants as the BB row above (amino N of C-1 = N1).
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.opsin_gate
    def test_central_n_substituted_stays_below_pin_tier(self):
        # A substituent on the central imido N is not built (its locant is not
        # shown in the BB): the guanidine-form name ships RT-exact, below
        # pin_verified,:34298).
        from orthonym import Orthonym
        from tests.support.rt_assert import name_is_rt_exact
        smi = "NC(=N)N(C)C(=N)N"
        r = _dt_row(smi)
        assert r["tier"] != "pin_verified", r
        assert name_is_rt_exact(r["name"], smi), r

    def test_protect_plain_guanidine(self):
        assert name_compound("NC(=N)N") == "guanidine"


@pytest.mark.unit
class TestT6AmidrazonePrefixes:
    """ (BB 34490) + (BB 34498): chain-terminal
    amidrazone C stays IN the chain (hydrazinyl+imino / amino+hydrazinylidene);
    ring-attached keeps the full acyl prefix (hydrazinecarboximidoyl)."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB 34494 verbatim PIN:
        ("N=C(NN)CC(=O)O", "3-hydrazinyl-3-iminopropanoic acid"),
        # chain-end pattern (OPSIN-verified):
        ("NC(=NN)CC(=O)O", "3-amino-3-hydrazinylidenepropanoic acid"),
        ("NN=C(N)CCC(=O)O", "4-amino-4-hydrazinylidenebutanoic acid"),
        # BB 34496 verbatim PIN (ring parent -> acyl prefix retained):
        ("NNC(=N)c1cccc(C(=O)O)c1", "3-(hydrazinecarboximidoyl)benzoic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # pass-D pins share the hydrazonamide FG machinery — protect:
        ("CC(=NN)NCCC(=O)O", "3-(ethanehydrazonamido)propanoic acid"),
        ("CC(=NN)N", "ethanehydrazonamide"),
        # pass-D chain-amidine pins share _TERMINAL_C_FGS + the amidine
        # block — protect:
        ("CCN=C(CCC(=O)OC)N(C)C",
         "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"),
    ])
    def test_protect_shared_machinery(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT7ImidohydrazideFamily:
    """P-66.4.2.1 (BB 34430): the R-C(=NH)-NH-NH2 amidrazone tautomer takes
    the 'imidohydrazide'/'carboximidohydrazide' suffix; P-14.3.4.1 (BB 2877):
    no terminal locants. Family reps per WAVE2-BUILD-PLAN-ALL §2a."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N=CNN", "methanimidohydrazide"),
        ("CC(=N)NN", "ethanimidohydrazide"),
    ])
    def test_heals(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # family reps already healed elsewhere — protect:
        ("NC(=N)SSC(=N)N", "carbamimidic dithioperoxyanhydride"),
        ("NNS(=NN)c1ccccc1", "benzenesulfinohydrazonohydrazide"),
        ("CC(=NN)NN", "ethanehydrazonohydrazide"),
    ])
    def test_protect_family_reps(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT8SulfinoSulfonoHydrazonamido:
    """ (BB 34540 verbatim PIN) + (BB 34617):
    S(=N-NH2) N-attached branches take the e->o amide-name prefix."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB 34540 verbatim:
        ("OC(=O)c1ccc(NS(=NN)c2ccccc2)cc1",
         "4-(benzenesulfinohydrazonamido)benzoic acid"),
        # sulfono analogue (OPSIN-verified during planning):
        ("O=S(=NN)(Nc1ccc(C(=O)O)cc1)c1ccccc1",
         "4-(benzenesulfonohydrazonamido)benzoic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    def test_protect_parent_direction(self):
        assert name_compound("NNS(=NN)c1ccccc1") == \
            "benzenesulfinohydrazonohydrazide"


@pytest.mark.unit
class TestT9ComplexPolyamines:
    """ (BB 26375): senior parent DIAMINE retained; other amine
    N demoted into N-substituent branches; numeric N-locant tags."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB verbatim PINs:
        ("NCCNCN", "N1-(aminomethyl)ethane-1,2-diamine"),
        ("CN(C)CCN(C)CCN",
         "N1-(2-aminoethyl)-N1,N2,N2-trimethylethane-1,2-diamine"),
        # same class, OPSIN-verified during planning:
        ("NCCNCCN", "N1-(2-aminoethyl)ethane-1,2-diamine"),
    ])
    def test_heals(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # existing 2-N path must stay byte-identical (primed style):
        ("NCCN", "ethane-1,2-diamine"),
        ("CNCCN", "N-methylethane-1,2-diamine"),
    ])
    def test_protect_simple_diamines(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT10Am2AcylChainSubstituents:
    """, BB 33576 verbatim): the off-chain amide
    block must express acyl-chain substituents or decline."""

    def test_heals(self):
        assert name_compound("CN(CC(O)CO)C(=O)CN") == \
            "2-amino-N-(2,3-dihydroxypropyl)-N-methylacetamide"

    @pytest.mark.parametrize("smiles,expected", [
        # T5b's existing clean-coverage case must not regress — the plain
        # off-chain amide with NO acyl-chain substituents:
        ("CCCC(NC(C)=O)CC", "N-(hexan-3-yl)acetamide"),
    ])
    def test_protect_plain_off_chain(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT11Am2PoolDiscard:
    """: the correct amide candidate must win the pool; the
    polyfunctional double-express (amide N named twice) must not be
    emitted for ANY input (structure-wrong)."""

    def test_heals(self):
        assert name_compound("NCC(=O)N(C)C") == "2-amino-N,N-dimethylacetamide"

    @pytest.mark.parametrize("smiles,expected", [
        # neighbouring amide pins that exercise the same pool branch:
        ("CC(=O)N(C)C", "N,N-dimethylacetamide"),
        ("CC(=O)NC", "N-methylacetamide"),
    ])
    def test_protect_amide_pool(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected


@pytest.mark.unit
class TestT12GeminalDicarboximidamide:
    """ /: geminal ring diamide/diimidamide."""

    def test_geminal_diamide_base(self):
        # step (a): the {1,1} locant dedup fix (BUILT, OPSIN-RT verified)
        assert name_compound("NC(=O)C1(C(N)=O)CCCCC1") == \
            "cyclohexane-1,1-dicarboxamide"

    def test_geminal_dicarboximidamide_substituted(self):
        # step (b) BUILT (W2E-D4, /: per-group
        # primed-N superscript locant subsystem (N''1-ethyl / N1,N1-dimethyl
        # with load-bearing priming + lowest-locant group assignment).
        assert name_compound("CCNC(=N)C1(C(=N)N(C)C)CCCCC1") == \
            "N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide"

    def test_substituted_stays_fail_closed(self, _validity_gate_on):
        # W2E-D4: now BUILT — the substituted geminal dicarboximidamide names
        # correctly and round-trips (accepts it). Verifies the built
        # class survives the end-to-end validity gate (not a wrong name).
        assert name_compound("CCNC(=N)C1(C(=N)N(C)C)CCCCC1") == \
            "N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide"

    @pytest.mark.parametrize("smiles", [
        # OUT-OF-CLASS forms must still fail closed (never a wrong name):
        # mixed amide/imidamide (one C=O, one C=N) on the geminal carbon.
        "NC(=O)C1(C(=N)N(C)C)CCCCC1",
    ])
    def test_out_of_class_fails_closed(self, _validity_gate_on, smiles):
        # Mixed carboxamide+carboximidamide is not the built (di)imidamide
        # class; must not emit the N-superscript dicarboximidamide name.
        assert _dt_name_compound(smiles) != \
            "N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide"

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=N)C1CCCCC1", "cyclohexanecarboximidamide"),  # mono form OK at HEAD
    ])
    def test_protect_mono(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected
