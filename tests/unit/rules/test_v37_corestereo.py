""" CORESTEREO — complex-ring / von-Baeyer CORE stereo completion.

WHAT THIS FILE GUARDS (and why there is no new production code)
--------------------------------------------------------------
The assigned lever was: "for a complex_ring / von-Baeyer core whose
constitution RT-matches but whose name LACKS the stereodescriptors the input
has, COMPLETE the stereo." A trace-before-code pass (a project rule) over 550
census abstainers (internal notes, CORESTEREO
addendum) found the lever is **OFF-PATH — 0 addressable targets**:

  * The stereo-completion machinery is ALREADY wired and load-bearing:
    `handlers/tier_a_ring.py:546` (complex_ring), `assembly/general_engine.py:773`
    (cage/chain, ST.1), `assembly/composer.py:4625`, `rules/polycyclic.py:3385`
    all route parent-scope stereo through `inject_stereo_reanchored_rt_gated`,
    which computes CIP descriptors via the canonical `assign_stereochemistry`
    path and renders them RT-gated on the FULL InChIKey.
  * Where a complex-ring / von-Baeyer core has a CORRECT constitution, the
    descriptors are computed AND rendered AND ship (the canaries below).
  * Where a complex_ring candidate abstains, the cause is NOT a dropped
    descriptor: over 400 candidates, 36 produced a complex_ring candidate and
    **31/32 of the abstaining ones carry a CONSTITUTION defect** (substituent
    atom-drop, glycan-drop, acyloxy mis-naming as "butyl"/"propyl"/"ethoxy"),
    with the residual being a charge-layer or OPSIN-unparseable pseudoasymmetric
    layer — none fixable by "rendering descriptors." Those are the acyloxy /
    ringsubst / glycan / charge levers, not stereo completion.

So per a project rule (a measurement that refutes must not be coded around) this
file ships NO new handler; it CODIFIES the finding as regression guards:
  1. decorated complex-ring / von-Baeyer cores with correct constitution keep
     completing their stereo (byte-identical + full-RT) — guards the reanchor;
  2. a wrong stereoisomer is NEVER rendered (diastereomer negative control);
  3. a constitution-defect complex_ring case abstains, never ships a wrong
     molecule (0-wrong).

Run ONLY this file (the whole suite deadlocks on an OPSIN pipe):
    .venv/bin/python -m pytest tests/unit/rules/test_v37_corestereo.py -q
"""
import signal
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym, errors
from orthonym.jvm_budget import jvm_slots
from tests.support.jars import jar_or_skip
from orthonym.assembly.retained_substitution import OpsinOracle

# These assertions are ABOUT the RT/validity gate (full-InChIKey round-trip), so
# the OPSIN gate must be ON exactly as in production (the suite disables it by
# default — a green-but-blind trap for gate tests; see tests/conftest.py).
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module", autouse=True)
def _jvm_slot():
    with jvm_slots(1, purpose="test_v37_corestereo"):
        yield


@pytest.fixture(scope="module")
def _oracle():
    return OpsinOracle(opsin_jar=jar_or_skip())


def _alarm(seconds=90):
    def _raise(sig, frm):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)


def _complete(smi):
    """The verified best-effort ('complete') tier: RT-verified engine names,
    NO unverified opt-in (gf=T, gfu=F, aag=T). This is the 0-wrong shipping
    tier for systematic ring names."""
    _alarm()
    try:
        return Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=False, allow_aromatic_general=True,
        ).name(smi)
    finally:
        signal.alarm(0)


def _pin(smi):
    _alarm()
    try:
        return Orthonym(style="pin").name(smi)
    finally:
        signal.alarm(0)


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


# Correct-stereo canaries: decorated complex-ring / von-Baeyer cores whose full
# stereo the pipeline already COMPLETES + RT-verifies at the 'complete' tier.
# (VERIFIED stable across fresh processes 2026-08-25, HEAD 3591e3b5.)
_CANARIES = [
    (
        "CN1[C@H]2CC[C@@H]1[C@@H](C(=O)O)[C@@H](OC(=O)c1ccccc1)C2",
        "(1R,2R,3S,5S)-3-(benzoyloxy)-8-methyl-8-azabicyclo[3.2.1]"
        "octane-2-carboxylic acid",
    ),
    (
        "COCCC1=C2C(C)(C)[C@H]3CC[C@@]2(C3)C(C)(C)CC1",
        "(1S,8R)-4-(2-methoxyethyl)-2,2,7,7-tetramethyltricyclo"
        "[6.2.1.0^3,8]undec-3-ene",
    ),
]


class TestCoreStereoAlreadyCompletes:
    """The stereo IS rendered on complex-ring / von-Baeyer cores with a correct
    constitution — the reanchor completes it, byte-identical + full-RT. Guards
    against a regression in the tier_a_ring / general_engine stereo routing."""

    @pytest.mark.parametrize("smi,expected", _CANARIES)
    def test_decorated_core_ships_full_stereo(self, smi, expected, _oracle):
        name = _complete(smi)
        assert not errors.is_failure_name(name)
        # byte-identical to the reference full-stereo name
        assert name == expected
        # and it denotes the RIGHT stereoisomer (full InChIKey round-trips)
        assert _rt_key(_oracle, name) == _ik(smi)

    @pytest.mark.parametrize("smi,expected", _CANARIES)
    def test_core_stereo_is_deterministic(self, smi, expected):
        # atom-order stable: two fresh namings agree.
        assert _complete(smi) == _complete(smi) == expected


class TestWrongStereoNeverRendered:
    """NEGATIVE CONTROL (mandatory): the completion is input-faithful — it
    renders the descriptor matching the ACTUAL input configuration, and a
    diastereomer gets a DISTINCT, self-consistent name. A wrong stereoisomer is
    never emitted (the reanchor's full-InChIKey RT gate is load-bearing)."""

    # tropane benzoate and its C3 epimer (the only difference is one @/@@).
    D1 = "CN1[C@H]2CC[C@@H]1[C@@H](C(=O)O)[C@@H](OC(=O)c1ccccc1)C2"
    D2 = "CN1[C@H]2CC[C@@H]1[C@@H](C(=O)O)[C@H](OC(=O)c1ccccc1)C2"

    def test_diastereomers_get_distinct_self_faithful_names(self, _oracle):
        assert _ik(self.D1) != _ik(self.D2)          # genuinely different inputs
        n1, n2 = _complete(self.D1), _complete(self.D2)
        assert not errors.is_failure_name(n1)
        assert not errors.is_failure_name(n2)
        assert n1 != n2                              # distinct names
        # each name round-trips to its OWN input, never to the other epimer
        assert _rt_key(_oracle, n1) == _ik(self.D1)
        assert _rt_key(_oracle, n1) != _ik(self.D2)
        assert _rt_key(_oracle, n2) == _ik(self.D2)
        assert _rt_key(_oracle, n2) != _ik(self.D1)


class TestConstitutionDefectNeverShipsWrong:
    """The dominant complex_ring abstain cause is a CONSTITUTION defect (the
    handler mis-names / drops a substituent), NOT stereo omission. Such a case
    must abstain (or ship the RIGHT constitution) at the PIN tier — never a
    wrong molecule. 0-wrong guard."""

    # cephalosporin: the complex_ring candidate drops the S-CH2-C#N thioether and
    # renames the amide substituent, so the constitution does not round-trip ->
    # suppresses at the PIN tier -> abstain (never a wrong-molecule ship).
    SMI = "CO[C@@]1(NC(=O)CSCC#N)C(=O)N2C(C(=O)[O-])=C(CSc3nnnn3C)CS[C@@H]21"

    def test_pin_tier_abstains_never_wrong(self, _oracle):
        name = _pin(self.SMI)
        if errors.is_failure_name(name):
            return  # clean abstain — correct 0-wrong outcome
        # if it DID ship, it must be the right constitution (block-1 InChIKey)
        k = _rt_key(_oracle, name)
        assert k is not None, f"shipped an OPSIN-unparseable name: {name!r}"
        in_b1 = _ik(self.SMI).split("-")[0]
        assert k.split("-")[0] == in_b1, (
            f"shipped a WRONG constitution: {name!r}")
