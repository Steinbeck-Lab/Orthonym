"""v26 P2: aromatic fused-cage von-Baeyer polyene + descriptor edge-audit.

Tests two changes in ``rules/vonbaeyer_universal.py``:

1. **Lifted mancude refusal** (behind ``allow_mancude`` / the ``--emit-tier
   complete`` ``allow_aromatic_general`` flag): a fused cage carrying aromatic
   ring atoms is kekulized and emitted as an explicit von-Baeyer polyene
   (P-23 unsaturation) instead of being refused. The DEFAULT / PIN path
   (``allow_mancude=False``) still refuses -> byte-identical. PIN retained /
   fusion names (quinoline, indole, ...) name via the PIN path first and are
   UNCHANGED under both ``pin`` and ``complete`` (P2 only fires on abstention).

2. **Descriptor edge-audit** (``audit_von_baeyer_descriptor``): a Java-free
   structural floor -- the descriptor + numbering must assert exactly the
   molecular ring/bridge bond set or the cage is discarded (fail-closed). This
   is independent of OPSIN, which fails OPEN without Java.

Harness (mirrors ``tests/unit/assembly/test_v26_p1_monocycle_engine.py``):
the conftest force-disables the production SELF-01 gate suite-wide and its
``opsin_to_smiles`` fixture is broken, so (a) direct-engine tests call
``name_general_ring`` / ``analyze_cage_universal`` and assert exact descriptor
strings + a complete E1 atom partition (deterministic, gate-independent);
(b) full-namer RT tests re-enable SELF-01 via ``production_gate`` (skips
without Java) and round-trip via a private stdin-batch OPSIN helper.
"""
import glob
import shutil
import subprocess

import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name
from orthonym.assembly.general_engine import name_general_ring
from orthonym.rules.vonbaeyer_universal import (
    analyze_cage_universal, audit_von_baeyer_descriptor,
)
from orthonym.rules.polycyclic import VonBaeyerAnalyzer
from orthonym.validation.e1_certificate import verify_certificate


pytestmark = pytest.mark.unit


# (SMILES, expected complete-tier von-Baeyer polyene). Each is an aromatic /
# mancude FUSED cage OUTSIDE the retained/fusion PIN catalog: the default pin
# path ABSTAINS, complete emits a VB polyene that OPSIN round-trips.
AROMATIC_FUSED_CASES = [
    ("C1=CC=CC2=CC=CC=CC=C12",          # heptalene
     "bicyclo[6.4.0]dodeca-1,3,5,7,9,11-hexaene"),
    ("C1=Cc2cc3ccccc3cc2C1",            # as-indacene
     "tricyclo[7.4.0.0^3,7]trideca-1(13),2,5,7,9,11-hexaene"),
    ("O1C=CC=CC=Cc2ccccc21",            # 1-benzoxonine (O fused to benzene)
     "8-oxabicyclo[7.4.0]trideca-1(13),2,4,6,9,11-hexaene"),
    ("C1=CC=Cc2ccccc2C1",               # benzocycloheptene (benzene + 7-ring enes)
     "bicyclo[5.4.0]undeca-1(11),3,5,7,9-pentaene"),
    ("C1CCCc2ccccc2C1",                 # benzosuberane (benzene + saturated 7-ring)
     "bicyclo[5.4.0]undeca-1(11),7,9-triene"),
    ("S1C=CC=CC=Cc2ccccc21",            # 1-benzothionine (S fused to benzene)
     "8-thiabicyclo[7.4.0]trideca-1(13),2,4,6,9,11-hexaene"),
    ("C1=CC=CC=Cc2ccccc2C1",            # benzocyclooctene
     "bicyclo[7.4.0]trideca-1(13),2,4,6,9,11-hexaene"),
]

# Saturated / isolated-ene cages named by the von-Baeyer VALID tier already:
# complete MUST equal pin (the audit runs on the valid tier too; no regression).
UNCHANGED_SATURATED = [
    "C1CC2CCC1CC2",           # bicyclo[2.2.2]octane (von-Baeyer valid tier)
    "C1CC2CCC1C2",            # norbornane / bicyclo[2.2.1]heptane
    "C1CCC2CCCCC2C1",         # decalin (decahydronaphthalene via PIN fusion)
]

# PIN retained / fusion aromatic names: name via the PIN path FIRST, so complete
# MUST equal pin (P2 never downgrades a PIN name to a von-Baeyer polyene).
UNCHANGED_PIN_FUSION = [
    "c1ccc2ccccc2c1",          # naphthalene
    "c1ccc2[nH]ccc2c1",        # 1H-indole -> indole
    "c1ccc2ncccc2c1",          # quinoline
    "c1ccc2ncncc2c1",          # quinazoline
    "c1ccc2c(c1)oc1ccccc12",   # dibenzofuran
]

# Fail-closed: the von-Baeyer ENGINE (name_general_ring) must REFUSE (None) --
# never a wrong name. (The full namer may still name these via the PIN path.)
FAIL_CLOSED_ENGINE = {
    "spiro / <2 bridgeheads (spiro[5.5]undecane)": "C1CCC2(CCCCC2)CC1",
    "charged aromatic fused (acridinium)": "C[n+]1c2ccccc2cc2ccccc21",
    "spiro co-ring (1,4-dioxaspiro[4.4]nonane)": "C1CC2(CC1)OCCO2",
    "lone monocycle (benzene, not a cage)": "c1ccccc1",
    # v29 Phase 2 T3b re-derived these two. Both expectations stand -- the engine
    # must refuse -- but the reasons recorded here were wrong on the numbers:
    # coronene is 24 cage atoms / 7 rings and ovalene 34 cage atoms / 10 rings,
    # so NEITHER exceeds MAX_CAGE_ATOMS = 40 and coronene does not exceed
    # MAX_CAGE_RINGS = 8 either. The Blue Book reason they must refuse is that
    # both are RETAINED fused-ring hydrocarbon parent components -- P-25.1
    # Table 2.7 "Retained names for hydrocarbon parent ring components", with
    # "coronene (PIN)" at BlueBookV2.md:11346 and both listed at P-25.8.2
    # (":14776") -- so the retained fusion name IS the PIN and a von-Baeyer
    # polyene construction can never be preferred for them.
    # The operative code cause is the structural floor, not a cap: both fail
    # audit_von_baeyer_descriptor ("descriptor edge-audit failed; refuse"), and
    # they still refuse with both caps set to 200. Ovalene's ring count (10) does
    # trip the ring cap first, but its refusal is over-determined.
    "retained fusion PIN; fails the descriptor edge-audit (coronene)":
        "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",
    "retained fusion PIN; 10 rings > cap AND fails the edge-audit (ovalene)":
        "c1cc2ccc3ccc4ccc5ccc6ccc7ccc8ccc1c1c2c3c4c2c5c6c7c8c12",
}


# --------------------------------------------------------------------------
# Fixtures / helpers (mirror the P1 test module)
# --------------------------------------------------------------------------
def _find_opsin_jar():
    for pat in ("opsin-cli-*-jar-with-dependencies.jar", "opsin-cli-*.jar",
                "opsin.jar"):
        m = glob.glob(pat)
        if m:
            return m[0]
    return None


@pytest.fixture
def production_gate(monkeypatch):
    """Re-enable the production SELF-01 OPSIN validity gate (the suite autouse
    fixture disables it). Skips when Java/OPSIN are unavailable."""
    if not shutil.which("java") or _find_opsin_jar() is None:
        pytest.skip("OPSIN/Java not available for production-gate semantics")
    monkeypatch.setattr(_namer_mod, "_DISABLE_VALIDITY_GATE", False)


@pytest.fixture(scope="module")
def opsin_roundtrip():
    """name -> canonical SMILES via ONE OPSIN JVM (stdin batch). Skips without
    Java/OPSIN. The conftest ``opsin_to_smiles`` fixture is broken (passes the
    name as a positional CLI arg -> read as a file path -> always None)."""
    jar = _find_opsin_jar()
    if not shutil.which("java") or jar is None:
        pytest.skip("OPSIN/Java not available")
    names = [name for _s, name in AROMATIC_FUSED_CASES]
    proc = subprocess.run(
        ["java", "-jar", jar, "-r", "-o", "smi"],
        input="\n".join(names) + "\n", capture_output=True, text=True,
        timeout=300)
    lines = proc.stdout.split("\n")
    out = {}
    for i, name in enumerate(names):
        smi = lines[i].strip() if i < len(lines) else ""
        out[name] = smi or None
    return out


def _engine_ring(smiles, flag=True):
    """Run name_general_ring directly with the flag on/off. Returns
    (mol, GeneralEngineResult|None) -- deterministic, no Java."""
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=flag)
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return mol, name_general_ring(mol, feats, allow_aromatic_general=flag)


# --------------------------------------------------------------------------
# Direct-engine: exact von-Baeyer polyene string + complete E1 partition
# (deterministic; gate-independent).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", AROMATIC_FUSED_CASES)
def test_engine_exact_string_and_e1(smiles, expected):
    mol, res = _engine_ring(smiles)
    assert res is not None, f"engine refused aromatic cage {smiles!r}"
    assert res.name == expected, f"{smiles}: {res.name!r} != {expected!r}"
    verdict = verify_certificate(mol, res)
    assert verdict.ok, f"E1 failed for {smiles}: {verdict.reason}"


# --------------------------------------------------------------------------
# The lifted refusal: analyze_cage_universal marks the cage mancude and yields
# a descriptor only when allow_mancude=True; None (byte-identical) when False.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", AROMATIC_FUSED_CASES)
def test_mancude_flag_gates_primitive(smiles, _expected):
    mol = Chem.MolFromSmiles(smiles)
    assert analyze_cage_universal(mol, allow_mancude=False) is None, (
        "default primitive must refuse the aromatic cage (byte-identical)")
    cage = analyze_cage_universal(mol, allow_mancude=True)
    assert cage is not None, "allow_mancude must lift the aromatic refusal"
    assert cage.is_mancude is True
    assert cage.descriptor.startswith(("bicyclo[", "tricyclo[", "tetracyclo[",
                                       "pentacyclo["))


# --------------------------------------------------------------------------
# Descriptor edge-audit: known-good True, known-bad (dropped bridge / duplicated
# locant) False. Direct unit test of the structural floor.
# --------------------------------------------------------------------------
def test_edge_audit_known_good_and_bad():
    mol = Chem.MolFromSmiles("C1CC2CCC1C2")   # norbornane, bicyclo[2.2.1]heptane
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    desc = VonBaeyerAnalyzer().analyze(mol, ring)
    assert desc.descriptor_string == "bicyclo[2.2.1]"

    # GOOD: the true decomposition asserts exactly the molecular bond set.
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, desc.bridge_info_list) is True

    # BAD 1: drop the last bridge -> a real ring bond goes un-asserted.
    dropped = list(desc.bridge_info_list[:-1])
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, dropped) is False

    # BAD 2: a numbering that maps two graph-adjacent atoms to one locant.
    bad_num = dict(desc.numbering)
    ks = sorted(bad_num)
    bad_num[ks[0]] = bad_num[ks[1]]
    assert audit_von_baeyer_descriptor(
        mol, ring, bad_num, desc.bridge_info_list) is False


def test_edge_audit_accepts_aromatic_cage():
    """The audit must ACCEPT a correct mancude cage (else it would only lose
    coverage). Every emitted case must pass it."""
    for smiles, _expected in AROMATIC_FUSED_CASES:
        mol = Chem.MolFromSmiles(smiles)
        cage = analyze_cage_universal(mol, allow_mancude=True)
        assert cage is not None, f"cage refused pre-audit for {smiles!r}"


# --------------------------------------------------------------------------
# Fail-closed: the engine refuses out-of-scope cages (never a wrong name).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("desc,smiles", list(FAIL_CLOSED_ENGINE.items()))
def test_engine_fail_closed(desc, smiles):
    _mol, res = _engine_ring(smiles)
    assert res is None, f"expected engine refusal for {desc}, got {res.name!r}"


# --------------------------------------------------------------------------
# Flag OFF: the engine is inert on an aromatic cage (default byte-identity).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", AROMATIC_FUSED_CASES)
def test_engine_flag_off_inert(smiles, _expected):
    _mol, res = _engine_ring(smiles, flag=False)
    assert res is None, f"flag-off engine must refuse aromatic cage {smiles!r}"


# --------------------------------------------------------------------------
# SELF-01: every emitted von-Baeyer polyene OPSIN-parses back to the input.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", AROMATIC_FUSED_CASES)
def test_names_opsin_roundtrip(smiles, expected, opsin_roundtrip):
    back = opsin_roundtrip.get(expected)
    assert back, f"OPSIN could not parse {expected!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{expected!r} round-trips to {back!r}, not {smiles!r}")


# --------------------------------------------------------------------------
# Reproduce-first: the aromatic-fused cases ABSTAIN under the default pin path.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", AROMATIC_FUSED_CASES)
def test_pin_abstains(smiles, _expected, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected default-pin abstention, got {out!r}")


# --------------------------------------------------------------------------
# complete-tier full namer emits the von-Baeyer polyene (PIN abstained first).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", AROMATIC_FUSED_CASES)
def test_complete_tier_emits(smiles, expected, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


# --------------------------------------------------------------------------
# No regression: valid-tier saturated cages are byte-identical (complete==pin).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", UNCHANGED_SATURATED)
def test_valid_tier_unchanged(smiles, production_gate):
    canon = Chem.CanonSmiles(smiles)
    pin = Orthonym(style="pin").name(canon)
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True).name(canon)
    assert comp == pin, f"{smiles}: complete {comp!r} != pin {pin!r}"


# --------------------------------------------------------------------------
# No regression: PIN retained/fusion aromatic names unchanged (complete==pin).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", UNCHANGED_PIN_FUSION)
def test_pin_fusion_names_unchanged(smiles, production_gate):
    canon = Chem.CanonSmiles(smiles)
    pin = Orthonym(style="pin").name(canon)
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True).name(canon)
    assert comp == pin, f"{smiles}: complete {comp!r} != pin {pin!r}"
    assert pin and not is_failure_name(pin), f"{smiles}: pin should name it"
