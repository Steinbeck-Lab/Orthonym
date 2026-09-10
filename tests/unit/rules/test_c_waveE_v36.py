""" Milestone-C Wave E — three independent tasks.

Task 1: 0-wrong hardening of the stereo fail-path. A stereo descriptor that cannot
        be verified (full name does not OPSIN-round-trip) must be OMITTED (ship the
        constitution-correct stereo-stripped flat name) rather than shipped WRONG.
Task 2: von-Baeyer dependent-bridge residual + detect_bridged_fused detector gap.
Task 3: the spiro-von-Baeyer constitutional defects (RT-gated; abstain if unfixable).

Run ONLY this file (whole-suite deadlocks on an OPSIN pipe):
    .venv/bin/python -m pytest tests/unit/rules/test_c_waveE_v36.py -q
"""
import signal
import pytest

from orthonym.jvm_budget import jvm_slots

# Run every behavioural test with the OPSIN validity gate ENABLED, i.e. exactly as
# production names (the suite disables the gate by default — a green-but-blind trap
# for a test written *about* gate behaviour; see tests/conftest.py). The Task-1
# leak witnesses SHIP a malformed pseudoasymmetric von-Baeyer stereo descriptor
# ONLY with the gate on (via the namer's BBR-GATE stereo carve-out), so the gate
# is load-bearing for these tests.
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module", autouse=True)
def _jvm_slot():
    with jvm_slots(1, purpose="test_c_waveE_v36"):
        yield


class _Timeout(Exception):
    pass


def _alarm(seconds=60):
    def _raise(sig, frm):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)


def _name(smiles):
    from orthonym import name_compound
    _alarm(60)
    try:
        return name_compound(smiles)
    finally:
        signal.alarm(0)


def _rt(smiles, name):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    _alarm(60)
    try:
        return opsin_roundtrip_check(smiles, name)["passed"]
    finally:
        signal.alarm(0)


def _constitution_rt(smiles, name):
    """True iff OPSIN parses *name* to a structure whose InChIKey block-1
    (constitution: formula + connectivity + mobile-H) matches *smiles*. This is
    the sanctioned stereo-OMISSION degrade (feedback_stereo_omission_is_not_wrong
    _molecule): the name is constitutionally correct with stereo omitted."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey
    _alarm(60)
    try:
        osmi = opsin_parse(name)
    finally:
        signal.alarm(0)
    if not osmi:
        return False
    mi = Chem.MolFromSmiles(smiles)
    mo = Chem.MolFromSmiles(osmi)
    if mi is None or mo is None:
        return False
    return MolToInchiKey(mi).split("-")[0] == MolToInchiKey(mo).split("-")[0]


# =====================================================================
# TASK 1 — stereo fail-path: unverifiable stereo is OMITTED, not shipped wrong
# =====================================================================

def test_task1_failpath_unit_returns_stripped_flat(monkeypatch):
    """When candidate A injects stereo but neither A nor a re-anchored B
    round-trips, the function must return the STEREO-STRIPPED FLAT base name,
    never candidate A (the wrong-stereo name)."""
    import orthonym.rules.stereochemistry as S
    import orthonym.validation.opsin_roundtrip as O
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")

    def fake_inject(base_name, m, bm, **kw):
        return "(1R)-" + base_name  # always injects a (fake) descriptor

    monkeypatch.setattr(S, "inject_stereo_from_locant_map", fake_inject)
    monkeypatch.setattr(O, "opsin_roundtrip_check", lambda s, n, **k: {"passed": False})
    monkeypatch.setattr(O, "opsin_atom_locant_map", lambda n, m: None)

    out = S.inject_stereo_reanchored_rt_gated("ethanol", mol, {0: 1})
    assert out == "ethanol"  # stereo omitted (was "(1R)-ethanol" before the fix)


def test_task1_candidate_a_rt_passes_byte_identical(monkeypatch):
    """When candidate A round-trips, it is returned UNCHANGED (byte-identical to
    the plain injector) — the fix must not disturb the passing path."""
    import orthonym.rules.stereochemistry as S
    import orthonym.validation.opsin_roundtrip as O
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    monkeypatch.setattr(S, "inject_stereo_from_locant_map",
                        lambda bn, m, bm, **k: "(1R)-" + bn)
    monkeypatch.setattr(O, "opsin_roundtrip_check", lambda s, n, **k: {"passed": True})

    out = S.inject_stereo_reanchored_rt_gated("ethanol", mol, {0: 1})
    assert out == "(1R)-ethanol"


@pytest.mark.parametrize("smiles,head_leak", [
    ("C1C[C@H]2C[C@H](C2)O1", "(1r,5s)-2-oxabicyclo[3.1.1]heptane"),
    ("C[C@]12CC[C@H](C1)C2", "(1r,4r)-1-methylbicyclo[2.1.1]hexane"),
    ("[C@@H]12CC[C@H](CC1)C2", "(1r,4r)-bicyclo[2.2.1]heptane"),
])
def test_task1_polycyclic_leak_no_unverifiable_stereo_ships(smiles, head_leak):
    """0-wrong hardening (gate ON = production). At HEAD each of these von-Baeyer
    inputs SHIPS a pseudoasymmetric stereo descriptor (``head_leak``) that OPSIN
    cannot round-trip — a residual 0-wrong leak reaching T1 via the BBR-GATE
    stereo carve-out. After the fix the injector returns the stereo-STRIPPED flat
    name; the default-tier SELF-01 gate (namer.py:1089) then abstains rather than
    ship a less-specific molecule. The invariant either way: NO name carrying a
    stereo descriptor that fails to round-trip may ship (abstain, or a fully
    round-tripping name, are both acceptable — a malformed-stereo ship is not)."""
    from orthonym.rules.stereochemistry import strip_stereo
    n = _name(smiles)
    # The exact HEAD leak string must no longer be emitted.
    assert n != head_leak, f"still ships the HEAD leak: {n!r}"
    # Core 0-wrong invariant: a shipped name is either stereo-free or full-RTs.
    assert strip_stereo(n) == n or _rt(smiles, n), \
        f"ships an unverifiable stereo descriptor: {n!r}"
    # When a constitution name IS emitted (not abstain), it must be correct.
    if not n.startswith("unknown") and strip_stereo(n) == n:
        assert _constitution_rt(smiles, n), f"constitution mismatch: {n!r}"


# =====================================================================
# TASK 2 — von-Baeyer force-add residual + detect_bridged_fused gap (NAMED BLOCKER)
# Both are out of the namer's construction reach and correctly ABSTAIN today. These
# guards lock 0-wrong: each stays abstain-or-RT, never a wrong-molecule ship, so a
# future coverage attempt cannot silently regress. See V36-WAVEE-TASK2-FINDING.md.
# =====================================================================

def test_task2b_detect_bridged_fused_correctly_false():
    """The (b) macrocycle's two 6-rings are NOT ortho-fused to each other, so it is
    not a bridged-FUSED system: detect_bridged_fused must return False, and
    name_bridged_fused_system must decline (None) — a detector fix would deliver zero
    breadth (routing there still abstains)."""
    from rdkit import Chem
    from orthonym.rules.bridged_fused import (
        detect_bridged_fused, name_bridged_fused_system,
    )
    m = Chem.MolFromSmiles("C1=NC2CCCCCCCCOC3C=NC(CCCCCCCCOC1CC2)CC3")
    assert detect_bridged_fused(m) is False
    assert name_bridged_fused_system(m) is None


@pytest.mark.parametrize("smiles", [
    # (b) bridged-fused / macrocyclic-bridged witness
    "C1=NC2CCCCCCCCOC3C=NC(CCCCCCCCOC1CC2)CC3",
    # (a) in-scope force-add-branch witnesses (aromatic-fused-bridged alkaloid cage,
    # macrocyclic cyclophane) — genuinely out of von-Baeyer reach
    "CN1CC[C@]23C(=O)C[C@H]4C(=CCO[C@H]5CC(=O)N(c6cc(O)ccc62)[C@H]3[C@H]54)C1",
    "c1cc2cc(c1)Oc1ccc(cc1)C[C@@H]1NCCc3ccc(cc31)Oc1cccc3c1[C@@H](C2)NCC3",
])
def test_task2_zero_wrong_abstain_or_roundtrip(smiles):
    """0-wrong guard: an out-of-reach complex polycyclic must either abstain OR emit a
    name that round-trips — never a wrong-molecule name."""
    n = _name(smiles)
    assert n.startswith("unknown") or _rt(smiles, n), \
        f"shipped a non-round-tripping name for an out-of-reach system: {n!r}"


# =====================================================================
# TASK 3 — spiro-von-Baeyer constitutional defects (NAMED BLOCKER, bounded follow-on)
# The built names are OPSIN-unparseable / wrong-constitution; abstains on them
# (0-wrong holds). None is a bounded von-Baeyer numbering/bridge fix — each needs a full
# spiro-component reconstruction. These guards lock 0-wrong. See V36-WAVEE-TASK3-FINDING.md.
# =====================================================================

@pytest.mark.parametrize("smiles", [
    # (1) oxine-lactone HW-misclassification signature
    "CC1=CC[C@]2(OC[C@]34CCC5=C(CC[C@@H]6C(=C5)C=CC(=O)OC6(C)C)[C@]3(C)CC[C@@H]4[C@@H]2C)OC1=O",
    # (2) pyro/pyrano fusion-component signature
    "COCc1[nH]nc2c1C1(CCSCC1)C(C#N)=C(N)O2",
    # (3) naphtho-dioxine peri-fused-as-von-Baeyer signature
    "O=C1CC[C@@H](O)[C@@H]2C1=CCCC21Oc2cccc3cccc(c23)O1",
])
def test_task3_spiro_vb_defect_zero_wrong(smiles):
    """0-wrong guard: a spiro-von-Baeyer input whose built component name is OPSIN-
    unparseable / wrong-constitution must ABSTAIN (or emit a name that round-trips) —
    never ship the malformed built name. A future construction fix must keep this."""
    n = _name(smiles)
    assert n.startswith("unknown") or _rt(smiles, n), \
        f"shipped a non-round-tripping spiro-VB name: {n!r}"


@pytest.mark.parametrize("smiles,expected", [
    # stereo-bearing von-Baeyer PINs whose full name DOES round-trip: byte-identical
    # (composer bicyclo path — _assemble_complete_bicyclo_name)
    ("CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2", "(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one"),
    ("C[C@H]1CC[C@@H]2CC[C@H]1C2", "(1S,2S,5R)-2-methylbicyclo[3.2.1]octane"),
    # non-stereo von-Baeyer canaries: unaffected
    ("OC1CC2CCC1C2", "bicyclo[2.2.1]heptan-2-ol"),
    ("CC12CCC(CC1)C2", "1-methylbicyclo[2.2.1]heptane"),
    # stereo tricyclo+ PINs that round-trip (polycyclic.py path —
    # name_polycyclic_complete): the reroute must stay byte-identical here too.
    ("C[C@H]1CC2CC3CC1CC(C2)(C3)", "(2S)-2-methyltricyclo[4.3.1.1^4,8]undecane"),
    ("O[C@H]1CC2CC3CC1CC(C2)C3", "(2S)-tricyclo[4.3.1.1^4,8]undecan-2-ol"),
    ("C1C2CC3CC1CC(C2)C3", "adamantane"),
])
def test_task1_canary_byte_identical(smiles, expected):
    assert _name(smiles) == expected
