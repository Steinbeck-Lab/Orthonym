"""v26 P5: charged general path under ``complete`` (BB P-73 / P-74).

``general_engine._common_refusal`` refuses any net-charged molecule ("net charge
(G3 scope)"). P5 lifts that refusal ONLY under ``complete``
(``allow_aromatic_general`` -> ``allow_charged``) and appends a charge suffix on
the ALREADY-NUMBERED general parent:

  * ``-ylium``  cation by loss of H- from a skeletal carbon      (P-73.2.2.1.1)
  * ``-ium``    protonated / substituted skeletal heteroatom      (P-73.1)
  * ``-ide``    anion by loss of H+ from a skeletal atom          (P-72.2.2.1)
  * ``-uide``   anion by addition of H- to a skeletal atom        (P-72.3)

The charge suffix's locant is the ACTUAL charged atom's parent locant, so the
descriptor is structurally faithful on its own; SELF-01 (OPSIN round-trip) is
the backstop but fails OPEN without Java, hence the structural fail-closed rules.

Reproduce-first (confirmed 2026-07-21, production OPSIN gate on):
  - ACCEPTANCE cases below: pin (default) ABSTAINS; ``complete`` emits an
    OPSIN-round-tripping charged name.
  - FAIL-CLOSED cases: ``complete`` abstains (never a wrong / charge-dropped
    name) -- charge on a substituent, a radical-cation, an unexpressible charge.

Harness mirrors test_v26_p3_failclosed_routing: ``conftest`` force-disables the
production SELF-01 gate for the whole suite; the full-namer cases here re-enable
it via ``production_gate`` (skipped without Java/OPSIN); the engine-direct and
byte-identity cases are deterministic and gate-independent.
"""
import glob
import shutil
import subprocess

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, is_failure_name
from orthonym.assembly.general_engine import name_general
from orthonym.validation import binding_spine as bs
from orthonym.validation.binding_spine import BindingSpine, verify_spine
from orthonym.validation.e1_certificate import verify_certificate


pytestmark = pytest.mark.unit


# ACCEPTANCE: pin ABSTAINS; complete emits this exact RT-OK charged name.
# (SMILES, complete-tier name). Categories: heteroarene-ammonium (monocycle),
# von-Baeyer cage cation, chain carbanion (carboxylate-free simple organic ion).
ACCEPT_CASES = [
    ("[SiH3][n+]1ccccc1", "1-silylpyridin-1-ium"),
    ("[GeH3][n+]1ccccc1", "1-germylpyridin-1-ium"),
    # v31 change-asserted-value: the aza-cage heteroatom takes the LOWEST locant
    # (P-31.1.4: among valid von-Baeyer numberings of the equal-bridge
    # bicyclo[4.4.0] cage, heteroatoms get lowest locants). N is separated from a
    # bridgehead by one carbon, so its lowest attainable locant is 3, NOT 4. The
    # old `4-silyl-4-aza` expectation was a stale higher-locant numbering; the
    # engine now emits the lowest-locant form (deterministic across spellings,
    # OPSIN round-trips to the input).
    ("[SiH3][n+]1ccc2ccccc2c1",
     "3-silyl-3-azabicyclo[4.4.0]deca-1(10),2,4,6,8-pentaen-3-ium"),
    ("[SiH3]C[CH-]C", "1-silylpropan-2-ide"),
    ("FC(F)(F)[CH-]C", "1,1,1-trifluoropropan-2-ide"),
    # Multi-charge dication: the multiplied suffix ("-1,4-diium") begins with
    # the consonant 'd' (di-), NOT a vowel -- P-16.7.1(a) requires the parent's
    # terminal 'e' be RETAINED ("...diazine-1,4-diium", not "diazin-1,4-diium").
    # Regression case for the elision-conditional fix in _append_charge_suffix.
    ("C[n+]1cc[n+](C)cc1", "1,4-dimethyl-1,4-diazine-1,4-diium"),
]

# FAIL-CLOSED: complete must ABSTAIN (never a wrong / charge-dropped name).
FAILCLOSED_CASES = [
    "[CH2+]Cc1ccccc1",       # charge on a substituent carbon (parent = ring)
    "[CH2+]C[CH2]",          # radical-cation (radical refusal stays)
    "[SiH3][CH-]c1ccccc1",   # charge not expressible as a parent suffix
]

_RT_NAMES = dict(ACCEPT_CASES)


# --------------------------------------------------------------------------
# Fixtures (mirror test_v26_p3_failclosed_routing)
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
    """Re-enable the production SELF-01 OPSIN validity gate (the suite disables
    it). Skips when Java/OPSIN are unavailable."""
    import orthonym.namer as _namer_mod
    if not shutil.which("java") or _find_opsin_jar() is None:
        pytest.skip("OPSIN/Java not available for production-gate semantics")
    monkeypatch.setattr(_namer_mod, "_DISABLE_VALIDITY_GATE", False)


@pytest.fixture(scope="module")
def opsin_roundtrip():
    """name -> canonical SMILES via ONE OPSIN JVM (stdin batch)."""
    jar = _find_opsin_jar()
    if not shutil.which("java") or jar is None:
        pytest.skip("OPSIN/Java not available")
    names = list(_RT_NAMES.values())
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


def _engine_name(smiles, *, allow_aromatic_general):
    """name_general on a perceived charged mol (no Java). Returns the emitted
    name or None (the engine fail-closed)."""
    canon = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(canon)
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=allow_aromatic_general)
    feats = nm._perceive(mol, canon, canon)
    nm._classify(feats)
    eng = name_general(mol, feats,
                       allow_aromatic_general=allow_aromatic_general)
    if eng is None:
        return None
    # E1 must accept the charged partition ONLY under complete.
    if not verify_certificate(mol, eng,
                              allow_charged=allow_aromatic_general).ok:
        return None
    return eng.name


# --------------------------------------------------------------------------
# PIN BYTE-IDENTITY (the sharpest constraint): with allow_aromatic_general OFF
# the net-charge refusal still fires -> the engine is a strict no-op (None) for
# every P5 case, so the PIN/default path is byte-identical. Deterministic, no
# Java. verify_certificate also still rejects charge by default.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_name", ACCEPT_CASES)
def test_pin_engine_inert_without_flag(smiles, _name):
    assert _engine_name(smiles, allow_aromatic_general=False) is None


@pytest.mark.parametrize("smiles,expected", ACCEPT_CASES)
def test_charge_claims_are_declared_in_the_binding_spine(smiles, expected):
    """Phase 0c Task 2: the charge-suffix producer now threads
    ``charge_atom_ids`` through a role='charge' binding, so P3's
    charge-totality proof stops being a permanent ``CHARGE_UNVERIFIED`` warn
    for every net-charged emission. The emitted NAME is unchanged (asserted
    against the SAME pinned ``ACCEPT_CASES`` string every other test in this
    file uses) -- only the binding ledger gained a new, E1-invisible entry.
    """
    got = _engine_name(smiles, allow_aromatic_general=True)
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"

    canon = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(canon)
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=True)
    feats = nm._perceive(mol, canon, canon)
    nm._classify(feats)
    eng = name_general(mol, feats, allow_aromatic_general=True)
    assert eng is not None and eng.name == expected

    spine = BindingSpine.from_token_bindings(eng.bindings)
    proof = verify_spine(mol, spine, eng.name, mode="audit",
                         allow_charged=True)
    assert proof.stats["charge_claims_declared"] > 0, (
        f"{smiles}: charge_atom_ids not threaded onto the binding spine")
    assert bs.CHARGE_UNVERIFIED not in proof.codes()
    assert proof.ok, proof.findings


# --------------------------------------------------------------------------
# Phase 0c Task 4 Part C: the T4 wiring now promotes P8's stereo axis and
# P3's CHARGE_UNVERIFIED to error severity (``escalate=STRICT_STEREO_CHARGE_AXES``,
# ``mode`` stays "audit"). These two ACCEPT_CASES are the task's named charge
# witnesses -- confirm the promotion does not touch them: their
# ``charge_atom_ids`` are already threaded (Task 2a), so CHARGE_UNVERIFIED
# never fires for them regardless of severity, the proof stays ``ok``, and
# the emitted name is unchanged.
# --------------------------------------------------------------------------
_TASK4_CHARGE_WITNESSES = [
    ("C[n+]1cc[n+](C)cc1", "1,4-dimethyl-1,4-diazine-1,4-diium"),
    ("[SiH3]C[CH-]C", "1-silylpropan-2-ide"),
]


@pytest.mark.parametrize("smiles,expected", _TASK4_CHARGE_WITNESSES)
def test_charge_witnesses_still_verify_clean_under_task4_strict_axes(
        smiles, expected):
    got = _engine_name(smiles, allow_aromatic_general=True)
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"

    canon = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(canon)
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=True)
    feats = nm._perceive(mol, canon, canon)
    nm._classify(feats)
    eng = name_general(mol, feats, allow_aromatic_general=True)
    assert eng is not None and eng.name == expected

    spine = BindingSpine.from_token_bindings(
        eng.bindings,
        stereo_atom_to_locant=getattr(eng, 'stereo_atom_to_locant', None))
    proof = verify_spine(mol, spine, eng.name, mode="audit",
                         allow_charged=True,
                         escalate=bs.STRICT_STEREO_CHARGE_AXES)
    assert bs.CHARGE_UNVERIFIED not in proof.codes()
    assert proof.ok, proof.findings


def test_e1_certificate_rejects_charge_by_default():
    smi = Chem.CanonSmiles("[SiH3][n+]1ccccc1")
    mol = Chem.MolFromSmiles(smi)
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=True)
    feats = nm._perceive(mol, smi, smi)
    nm._classify(feats)
    eng = name_general(mol, feats, allow_aromatic_general=True)
    assert eng is not None
    assert verify_certificate(mol, eng, allow_charged=False).ok is False
    assert verify_certificate(mol, eng, allow_charged=True).ok is True


# --------------------------------------------------------------------------
# ENGINE-DIRECT EMISSION under complete (deterministic, no Java): exact strings.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", ACCEPT_CASES)
def test_engine_emits_charged_name(smiles, expected):
    got = _engine_name(smiles, allow_aromatic_general=True)
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"


# --------------------------------------------------------------------------
# FAIL-CLOSED (deterministic, no Java): the engine returns None -> abstain.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", FAILCLOSED_CASES)
def test_engine_fails_closed(smiles):
    assert _engine_name(smiles, allow_aromatic_general=True) is None


# --------------------------------------------------------------------------
# FULL-NAMER complete tier: pin ABSTAINS, complete emits the expected name.
# Needs the production gate (else the suite ships unverified names).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", ACCEPT_CASES)
def test_pin_abstains(smiles, expected, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected pin abstention for {smiles!r}, got {out!r}")


@pytest.mark.parametrize("smiles,expected", ACCEPT_CASES)
def test_complete_emits(smiles, expected, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


@pytest.mark.parametrize("smiles", FAILCLOSED_CASES)
def test_complete_fails_closed(smiles, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert is_failure_name(out), f"{smiles}: expected abstention, got {out!r}"


# --------------------------------------------------------------------------
# SELF-01: every name complete emits round-trips to the input structure.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,name", sorted(_RT_NAMES.items()))
def test_emitted_names_roundtrip(smiles, name, opsin_roundtrip):
    back = opsin_roundtrip.get(name)
    assert back, f"OPSIN could not parse {name!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{name!r} round-trips to {back!r}, not {smiles!r}")
