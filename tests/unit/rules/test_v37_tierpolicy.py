""" TIER-POLICY — promote RT-verified general-engine rescues to the COMPLETE tier.

WHAT THIS FILE GUARDS
---------------------
A trace-before-code pass (a project rule; internal notes
CORESTEREO addendum + this task's report) established that ~88/150 ring+stereo
abstainers ship a FULL-InChIKey-RT-verified name at the BEST-EFFORT tier
(`gf=T, gfu=T`) but ABSTAIN at the verified COMPLETE tier (`gf=T, gfu=F`).

The REAL demotion gate is NOT the three sites the task brief named
(`namer.py:3234/3272/3294`); those never fire for these witnesses (verified by a
live monkeypatch trace). The gate is the ** coverage-by-construction handoff**
inside `_try_general_engine_recovery` (`namer.py:3760`,
`if not self._general_fallback_unverified: return None`), with the same
best-effort-only pattern shared by the two abstain-rescues `namer.py:3234`
(`_try_demote_senior_group_rescue`) and `:3272`
(`_try_alternate_parent_rescue`, whose internal guard is `namer.py:4091`).

THE FIX (precision-safe): each of those three abstain-rescues is re-gated from
`self._general_fallback_unverified` to `self._general_fallback`, so it also runs
at the COMPLETE tier. Every one already RT-gates its own candidate
(`_stereo_emit_decision` + `_rt_match`, or `opsin_roundtrip_check`), and at the
COMPLETE tier `_stereo_emit_decision` returns `(True, False)` only for a
FULL-stereo name and `(False, False)` (abstain) otherwise — so the RT gate is a
FULL-InChIKey compare (constitution+stereo+charge). A rescue whose RT fails
still returns None → the molecule still abstains at COMPLETE. Nothing that is
not RT-verified is promoted; best-effort behaviour is unchanged; the default
(PIN) tier never runs these rescues (`_general_fallback` is False there) so it is
byte-identical.

The np-systematic DOWNGRADE at `namer.py:3294` is DELIBERATELY NOT promoted: it
replaces a retained natural-product-parent PIN (`ursane`, `hopane`,...) with a
systematic von-Baeyer form, which would REGRESS the PIN at the complete tier
(governing priority #1). It stays best-effort-only.

Run ONLY this file (the whole suite deadlocks on an OPSIN pipe):
    .venv/bin/python -m pytest tests/unit/rules/test_v37_tierpolicy.py -q
"""
import pytest
from orthonym.wallclock import wall_clock_limit
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym, errors
from orthonym.jvm_budget import jvm_slots
from tests.support.jars import jar_or_skip
from orthonym.assembly.retained_substitution import OpsinOracle
import orthonym.assembly.t4_coverage as t4_coverage

# These assertions are ABOUT the RT/validity gate (full-InChIKey round-trip), so
# the OPSIN gate must be ON exactly as in production (the suite disables it by
# default — a green-but-blind trap for gate tests; see tests/conftest.py).
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module", autouse=True)
def _jvm_slot():
    with jvm_slots(1, purpose="test_v37_tierpolicy"):
        yield


@pytest.fixture(scope="module")
def _oracle():
    return OpsinOracle(opsin_jar=jar_or_skip())


def _complete(smi):
    """The verified COMPLETE tier: RT-verified engine/rescue names, NO T4
    unverified ship (gf=T, gfu=F, aag=T). The 0-wrong shipping tier."""
    with wall_clock_limit(120):
        return Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=False, allow_aromatic_general=True,
        ).name(smi)


def _best_effort(smi):
    with wall_clock_limit(120):
        return Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=True, allow_aromatic_general=True,
        ).name(smi)


def _pin(smi):
    with wall_clock_limit(120):
        return Orthonym(style="pin").name(smi)


def _ik(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m is not None else None


def _rt_key(oracle, name):
    """Full isomeric InChIKey OPSIN round-trips ``name`` to, or None."""
    if not name or errors.is_failure_name(name):
        return None
    s = oracle.name_to_smiles(name)
    if not s:
        return None
    m = Chem.MolFromSmiles(s)
    return inchi.MolToInchiKey(m) if m is not None else None


# Witnesses (VERIFIED fresh 2026-08-26): each ABSTAINS at the COMPLETE tier today
# and ships a FULL-InChIKey-RT-verified name at the BEST-EFFORT tier, routed
# through the T4-handoff (`namer.py:3760`). After the fix each ships at COMPLETE.
_WITNESSES = [
    "CC[C@H](C)C[C@@H](c1nc(C(=O)O)c(-c2c[nH]c3ccccc23)o1)N(C)C",
    "CCCCNC(=O)[C@H](C)NC(=O)[C@H](N)c1ccccc1",
    "Nc1ccc([C@@H]2O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]2O)cc1",
    "C[C@H](CCC(=O)NCC(=O)[O-])[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@H]"
    "(O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C",
]

# A no-stereo molecule that ALSO routes through the T4-handoff and abstains at
# COMPLETE without it — used for the negative control so the stereo gate can never
# be the reason for an abstain (isolates the RT gate). Breadth job 1: was
# haloperidol, which the main pipeline now names itself (RT-exact, PIN tier
# '4-[4-(4-chlorophenyl)-4-hydroxypiperidin-1-yl]-1-(4-fluorophenyl)butan-1-one'),
# so the handoff is no longer reached for it; this a dev split row (a chloro
# fentanyl analogue) still needs it (measured: PIN tier abstains, COMPLETE names it,
# COMPLETE abstains with name_t4_complete sabotaged and the rescues switched off).
_NOSTEREO_T4 = "CCC(=O)N(c1ccc(Cl)cc1)C1CCN(CCc2ccccc2)CC1"


class TestWitnessesPromotedToComplete:
    """The lever: an RT-verified general-engine rescue now ships at the COMPLETE
    tier (was abstain), and it denotes the RIGHT molecule (full InChIKey)."""

    @pytest.mark.parametrize("smi", _WITNESSES)
    def test_witness_ships_rt_exact_at_complete(self, smi, _oracle):
        name = _complete(smi)
        assert not errors.is_failure_name(name), (
            f"COMPLETE tier still abstains on a promotable witness: {smi!r}")
        # 0-wrong: the promoted name must full-InChIKey round-trip to the input.
        assert _rt_key(_oracle, name) == _ik(smi), (
            f"promoted a name that does NOT round-trip to the input: {name!r}")


class TestUnverifiedRescueStillAbstains:
    """NEGATIVE CONTROL (mandatory): promotion adopts a T4-handoff candidate
    ONLY when it round-trips. A candidate that names a DIFFERENT molecule is
    REJECTED by the RT gate → the molecule still abstains at COMPLETE. The RT
    gate — not the tier flag — is the precision moat; promotion did not relax
    it."""

    def test_rt_failing_t4_candidate_abstains_at_complete(self, monkeypatch):
        # Force the producer to emit a VALID name for a DIFFERENT molecule.
        # Haloperidol has no stereocentres, so the stereo gate passes and only
        # the constitution RT gate can reject — isolating the precision guard.
        # Disable the two sibling abstain-rescues so ONLY the promoted handoff
        # is in play (otherwise they would legitimately name the molecule and
        # mask what this test isolates).
        monkeypatch.setattr(
            t4_coverage, "name_t4_complete",
            lambda mol, feats, *a, **k: "benzene")
        monkeypatch.setattr(
            Orthonym, "_try_demote_senior_group_rescue",
            lambda self, s: None)
        monkeypatch.setattr(
            Orthonym, "_try_alternate_parent_rescue",
            lambda self, s: None)
        # Breadth job 1 added a third sibling: the PIN tier's re-run with the
        # promoted ring-substituent producers (Orthonym._name_with_pin_promotion),
        # which names haloperidol correctly (RT-exact) and so would mask what this
        # test isolates. Switched off here for the same reason.
        monkeypatch.setattr(
            Orthonym, "_pin_promotion_eligible",
            lambda self: False)
        # The general chain engine cites amide N-substituents with the locant N,
        # so its inline attempt now names the molecule correctly (RT-exact
        # 'N-(4-chlorophenyl)-N-[1-(2-phenylethyl)piperidin-4-yl]propanamide')
        # before the handoff is reached. It is turned off here too, so the
        # handoff -- the path under test -- is the one that runs.
        from orthonym.assembly import general_engine
        monkeypatch.setattr(
            general_engine, "name_general",
            lambda mol, feats, *a, **k: None)
        name = _complete(_NOSTEREO_T4)
        assert errors.is_failure_name(name), (
            f"COMPLETE tier SHIPPED an RT-failing T4 candidate: {name!r}")

    def test_wrong_candidate_never_ships_as_wrong_molecule(self, monkeypatch, _oracle):
        # Broader 0-wrong guard: even with the producer sabotaged, the
        # COMPLETE tier NEVER ships the wrong molecule — it abstains or finds a
        # correct name by another (RT-verified) path.
        wrong_ik = _ik("c1ccccc1")
        monkeypatch.setattr(
            t4_coverage, "name_t4_complete",
            lambda mol, feats, *a, **k: "benzene")
        name = _complete(_NOSTEREO_T4)
        k = _rt_key(_oracle, name)
        assert k != wrong_ik, (
            f"COMPLETE tier SHIPPED the RT-failing wrong candidate: {name!r}")
        if not errors.is_failure_name(name):
            assert k == _ik(_NOSTEREO_T4), (
                f"COMPLETE tier shipped a wrong molecule: {name!r}")

    def test_unpatched_nostereo_witness_ships_at_complete(self, _oracle):
        # Positive bracket: with the REAL (RT-verified) name, the same input
        # ships at COMPLETE — so the negative control's abstain is the RT gate
        # rejecting the wrong candidate, not the input being unnameable.
        name = _complete(_NOSTEREO_T4)
        assert not errors.is_failure_name(name)
        assert _rt_key(_oracle, name) == _ik(_NOSTEREO_T4)


class TestDefaultPinTierByteIdentical:
    """PIN never regresses: the default (PIN) tier never runs these rescues
    (`_general_fallback` is False), so its output is byte-identical."""

    _CANARIES = [
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)O", "acetic acid"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",
         "2-[4-(2-methylpropyl)phenyl]propanoic acid"),
    ]

    @pytest.mark.parametrize("smi,expected", _CANARIES)
    def test_default_tier_name_unchanged(self, smi, expected):
        assert _pin(smi) == expected


class TestCompleteTierDeterminism:
    """A promoted name is deterministic across fresh namings (atom-order
    stable)."""

    @pytest.mark.parametrize("smi", _WITNESSES[:2])
    def test_promoted_name_is_deterministic(self, smi):
        n1 = _complete(smi)
        n2 = _complete(smi)
        assert not errors.is_failure_name(n1)
        assert n1 == n2
