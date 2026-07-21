"""v26 P1: general LONE-monocycle aromatic/hetero ring engine.

Tests ``general_engine.name_general_monocycle`` and its wiring into the
``--emit-tier complete`` full namer (``Orthonym(general_fallback=True,
allow_aromatic_general=True)``).

Root cause fixed: a bare monocyclic ring (benzene, pyridine, thiophene, ...)
names fine, but the substituted form often abstains under the default composer
because its per-class substituent identifier has a finite hand-built
vocabulary. P1 routes the substituents through the never-None universal
recursion (``substituent_enumerator.name_substituent``) instead. The
demonstrated NEWCOV win is benzene aryl-ethers (``-OCF3`` / ``-OCH2CF3`` / ...),
which the ``benzene.py`` vocab drops but the recursion + SELF-01 name faithfully.

Reproduce-first (confirmed 2026-07-20, production OPSIN gate on): under the
DEFAULT pin path every NEWCOV SMILES below returns ``is_failure_name``
(``unknown organic compound``); under ``complete`` they emit an
OPSIN-round-tripping name.

Every emission is fail-closed: E1 atom partition + SELF-01 OPSIN round-trip.
The engine returns None on anything outside scope (charged / fused-cage parent /
tier-5-unnameable substituent), never a wrong name.

NOTE on the harness: ``conftest._disable_opsin_validity_gate_for_tests``
force-disables the production SELF-01 gate for the whole suite (tests assert
raw output). The full-namer cases here re-enable it via the ``production_gate``
fixture (skipped when Java/OPSIN are unavailable), so they exercise real
production semantics. The direct-engine cases are deterministic and
gate-independent.
"""
import glob
import shutil
import subprocess

import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name
from orthonym.assembly.general_engine import name_general_monocycle
from orthonym.validation.e1_certificate import verify_certificate


pytestmark = pytest.mark.unit


# (SMILES, expected complete-tier name) — pin ABSTAINS, complete EMITS, RT-OK.
# All are benzene aryl-ethers: the concrete P1 coverage win (benzene vocab miss).
NEWCOV_CASES = [
    ("FC(F)(F)Oc1ccccc1", "1-trifluoromethoxybenzene"),
    ("FC(F)(F)COc1ccccc1", "1-(2,2,2-trifluoroethoxy)benzene"),
    ("FC(F)Oc1ccccc1", "1-difluoromethoxybenzene"),
    ("FC(F)(F)C(F)(F)Oc1ccccc1", "1-(1,1,2,2,2-pentafluoroethoxy)benzene"),
    ("ClCCOc1ccccc1", "1-(2-chloroethoxy)benzene"),
    ("FCCOc1ccccc1", "1-(2-fluoroethoxy)benzene"),
    ("FC(F)(F)CCOc1ccccc1", "1-(3,3,3-trifluoropropoxy)benzene"),
    ("FC(F)(F)COc1ccc(F)cc1F", "2,4-difluoro-1-(2,2,2-trifluoroethoxy)benzene"),
]

# Heteroarene / carbocycle parents named DIRECTLY by the monocycle engine
# (the default path also names most of these; here we prove the engine's own
# parent-name + numbering + suffix machinery is correct across ring classes).
DIRECT_ENGINE_CASES = [
    ("Clc1ccncc1", "4-chloropyridine"),
    ("Cc1ccccn1", "2-methylpyridine"),
    ("Clc1cccs1", "2-chlorothiophene"),
    ("Cc1ccco1", "2-methylfuran"),
    ("Clc1ccc[nH]1", "2-chloro-1H-pyrrole"),
    ("Cc1cnc[nH]1", "5-methyl-1H-imidazole"),
    ("Cc1ccn[nH]1", "5-methyl-1H-pyrazole"),
    ("Cc1ccon1", "3-methyl-1,2-oxazole"),
    ("Cc1cscn1", "4-methyl-1,3-thiazole"),
    ("Cc1cncnc1", "5-methylpyrimidine"),
    ("Cc1ccnnc1", "4-methylpyridazine"),
    ("Cc1cnccn1", "2-methylpyrazine"),
    ("C1CCCCC1C1CCCCC1", "1-cyclohexylcyclohexane"),
    ("c1ccc(cc1)C1CCCCC1", "1-cyclohexylbenzene"),
    ("O=C1CCCCC1", "cyclohexan-1-one"),
    ("Oc1ccncc1", "pyridin-4-ol"),
    ("OC(=O)c1ccncc1", "pyridine-4-carboxylic acid"),
]

# Every (SMILES, name) whose emitted name must OPSIN-round-trip to the input.
ALL_RT_CASES = NEWCOV_CASES + DIRECT_ENGINE_CASES

# Fail-closed: the monocycle path must REFUSE (return None). Never a wrong name.
FAIL_CLOSED = {
    "charged (pyridin-1-ium)": "C[n+]1ccccc1",
    "fused cage (naphthalene)": "c1ccc2ccccc2c1",
    "tier-5 ring-on-ring substituent": "c1ccc(cc1)C1CCC(CC1)C1CCCCC1",
    # v26 P1 critical-defect fix: a spiro co-ring is perceived as a
    # substituent that attaches to the parent ring's spiro atom at TWO
    # points; naming it via name_substituent's single-attachment recursion
    # silently drops the ring-closure bond and mis-names it as a linear
    # alkyl chain (WRONG STRUCTURE). Must fail closed instead.
    "spiro co-ring (1,4-dioxaspiro[4.4]nonane)": "C1CC2(CC1)OCCO2",
    "spiro co-ring + carboxylic acid (spiro[5.5]undecane)":
        "O=C(O)C1CCC2(CCCCC2)CC1",
}

# Already-working substituted forms: complete MUST equal pin (P1 only fires on
# PIN abstention; the PIN path runs first and is never downgraded).
UNCHANGED_CASES = [
    "Cc1ccccc1",                 # toluene
    "c1ccncc1",                  # pyridine
    "[O-][N+](=O)c1ccccc1",      # nitrobenzene
    "Cc1ccccn1",                 # 2-methylpyridine
    "FC(F)(F)COc1ccccn1",        # 2-(2,2,2-trifluoroethoxy)pyridine
]


# --------------------------------------------------------------------------
# Fixtures / helpers
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
    """Re-enable the production SELF-01 OPSIN validity gate for the full-namer
    tests (the suite autouse-fixture disables it). Skips when Java/OPSIN are
    unavailable -- without the gate, the default path ships unverified names
    and the reproduce/emit semantics do not hold."""
    if not shutil.which("java") or _find_opsin_jar() is None:
        pytest.skip("OPSIN/Java not available for production-gate semantics")
    monkeypatch.setattr(_namer_mod, "_DISABLE_VALIDITY_GATE", False)


@pytest.fixture(scope="module")
def opsin_roundtrip():
    """name -> canonical SMILES via ONE OPSIN JVM (stdin batch). Skips when
    Java/OPSIN are unavailable. The conftest ``opsin_to_smiles`` fixture is not
    used: it passes the name as a positional CLI arg, which OPSIN reads as a
    file path (always None)."""
    jar = _find_opsin_jar()
    if not shutil.which("java") or jar is None:
        pytest.skip("OPSIN/Java not available")
    names = [name for _s, name in ALL_RT_CASES]
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


def _direct_engine(smiles):
    """Run name_general_monocycle directly with the flag on. Returns
    (mol, GeneralEngineResult|None) -- deterministic, no Java."""
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=True)
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return mol, name_general_monocycle(mol, feats, allow_aromatic_general=True)


# --------------------------------------------------------------------------
# The engine produces the exact expected string with a valid E1 partition
# (deterministic; gate-independent).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", NEWCOV_CASES + DIRECT_ENGINE_CASES)
def test_engine_exact_string_and_e1(smiles, expected):
    mol, res = _direct_engine(smiles)
    assert res is not None, f"engine refused {smiles!r}"
    assert res.name == expected, f"{smiles}: {res.name!r} != {expected!r}"
    verdict = verify_certificate(mol, res)
    assert verdict.ok, f"E1 failed for {smiles}: {verdict.reason}"


# --------------------------------------------------------------------------
# SELF-01: every emitted name OPSIN-parses back to the input structure.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", ALL_RT_CASES)
def test_names_opsin_roundtrip(smiles, expected, opsin_roundtrip):
    back = opsin_roundtrip.get(expected)
    assert back, f"OPSIN could not parse {expected!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{expected!r} round-trips to {back!r}, not {smiles!r}")


# --------------------------------------------------------------------------
# Reproduce-first: the NEWCOV cases ABSTAIN under the default pin path.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", NEWCOV_CASES)
def test_newcov_pin_abstains(smiles, _expected, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected default-pin abstention, got {out!r}")


# --------------------------------------------------------------------------
# complete-tier full namer emits the NEWCOV names (PIN abstained first).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", NEWCOV_CASES)
def test_complete_tier_emits(smiles, expected, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


# --------------------------------------------------------------------------
# Fail-closed: the monocycle path refuses out-of-scope inputs.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("desc,smiles", list(FAIL_CLOSED.items()))
def test_monocycle_fail_closed(desc, smiles):
    _mol, res = _direct_engine(smiles)
    assert res is None, f"expected refusal for {desc}, got {res.name!r}"


def test_complete_tier_abstains_on_tier5_substituent(production_gate):
    """A molecule whose only monocycle path needs a tier-5 (unnameable)
    substituent must ABSTAIN under complete -- never a wrong name."""
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles("c1ccc(cc1)C1CCC(CC1)C1CCCCC1"))
    assert (not out) or is_failure_name(out), f"expected abstention, got {out!r}"


def test_complete_tier_names_spiro_acid(production_gate):
    """v26 P1 critical-defect regression, UPDATED by v27 P3: a substituted spiro
    compound must NEVER be the wrong linear-alkyl mis-name (was:
    '1-pentylcyclohexane-4-carboxylic acid'). v26 P1 kept it abstaining; v27 P3's
    general spiro engine now names it CORRECTLY -- assert the round-tripping PIN
    (the intent 'never the mis-name' is satisfied even better)."""
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles("O=C(O)C1CCC2(CCCCC2)CC1"))
    assert out == "spiro[5.5]undecane-3-carboxylic acid", (
        f"expected the P3 spiro PIN, got {out!r}")


# --------------------------------------------------------------------------
# Byte-identity: the flag OFF makes the monocycle path inert.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", NEWCOV_CASES)
def test_flag_off_monocycle_inert(smiles, _expected):
    nm = Orthonym(style="pin", general_fallback=True)  # allow_aromatic_general=False
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    assert name_general_monocycle(
        mol, feats, allow_aromatic_general=False) is None


@pytest.mark.parametrize("smiles,_expected", NEWCOV_CASES)
def test_flag_off_full_namer_abstains(smiles, _expected, production_gate):
    # With allow_aromatic_general OFF, complete-style namer still abstains on
    # the NEWCOV cases (monocycle path inert; cage/chain paths do not fire).
    nm = Orthonym(style="pin", general_fallback=True)
    assert is_failure_name(nm.name(Chem.CanonSmiles(smiles)))


# --------------------------------------------------------------------------
# Already-working substituted forms are UNCHANGED under complete (PIN first).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", UNCHANGED_CASES)
def test_already_working_unchanged(smiles, production_gate):
    canon = Chem.CanonSmiles(smiles)
    pin = Orthonym(style="pin").name(canon)
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True).name(canon)
    assert comp == pin, f"{smiles}: complete {comp!r} != pin {pin!r}"
