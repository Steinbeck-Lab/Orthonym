""": charged general path under ``complete`` (BB /.

``general_engine._common_refusal`` refuses any net-charged molecule ("net charge
(G3 scope)"). P5 lifts that refusal ONLY under ``complete``
(``allow_aromatic_general`` -> ``allow_charged``) and appends a charge suffix on
the ALREADY-NUMBERED general parent:

  * ``-ylium`` cation by loss of H- from a skeletal carbon
  * ``-ium`` protonated / substituted skeletal heteroatom
  * ``-ide`` anion by loss of H+ from a skeletal atom
  * ``-uide`` anion by addition of H- to a skeletal atom

The charge suffix's locant is the ACTUAL charged atom's parent locant, so the
descriptor is structurally faithful on its own; (OPSIN round-trip) is
the backstop but fails OPEN without Java, hence the structural fail-closed rules.

Reproduce-first (confirmed 2026-07-21, production OPSIN gate on; pin-emit
split 2026-08-21):
  - PIN_EMIT_CASES: the three silyl/germyl heteroarene-ammonium rows ARE PINs
     + and EMIT at the default pin tier.
  - PIN_ABSTAIN_CASES: pin (default) ABSTAINS; ``complete`` emits an
    OPSIN-round-tripping charged name.
  - FAIL-CLOSED cases: the engine abstains (never a wrong / charge-dropped
    name) -- charge on a substituent, a radical-cation, an unexpressible charge.
    Since 2026-09-26 (decision D-b) the full ``complete`` tier names the first
    two through another producer, RT-exact; it still abstains on the third.

Harness mirrors test_v26_p3_failclosed_routing: ``conftest`` force-disables the
production gate for the whole suite; the full-namer cases here re-enable
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
from tests.support.jars import jar_or_none
from tests.support.rt_assert import assert_full_rt


pytestmark = pytest.mark.unit


# Two expectation tables, split  because the FULL NAMER and the
# ENGINE-DIRECT (``name_general``) call paths legitimately produce different --
# each individually correct -- strings for the fused-ring cation. See
# internal notes (PIN status, engine gap):
# * FULL_NAMER_CASES -- what ``Orthonym.name`` actually ships. The three
# heteroarene-ammonium rows are PINs, whose own worked example is
# ``1-methylpyridin-1-ium (PIN)``; silyl/germyl are PIN-eligible preselected
# prefixes,, so they EMIT at the default pin tier -- they do NOT
# abstain. For the fused-ring cation the production ``ions.py`` onium
# router supplies the retained ``isoquinolin-2-ium`` form /
#: a fusion name is preferred over a skeletal-replacement 'a'
# name) and wins at every tier.
# * ENGINE_DIRECT_CASES -- what ``name_general`` emits when called in
# isolation, bypassing the onium router. ``name_general_ring`` has no
# retained-fused-ring preference, so it emits the von-Baeyer /
# skeletal-replacement form for the fused-ring cation. That string is NEVER
# shipped by the full namer (the onium router intercepts first), so it is
# engine-isolation-only and 0-wrong-safe. Every other row is identical
# between the two tables.

# Full-namer PIN-tier emissions: these ARE PINs and emit at the DEFAULT pin tier.
PIN_EMIT_CASES = [
    ("[SiH3][n+]1ccccc1", "1-silylpyridin-1-ium"),
    ("[GeH3][n+]1ccccc1", "1-germylpyridin-1-ium"),
    # Fused-ring cation: the full namer ships the retained fusion name
    # ``2-silylisoquinolin-2-ium`` /, worked example:42203),
    # via the ions.py onium router recursing on the neutral bare ring.
    ("[SiH3][n+]1ccc2ccccc2c1", "2-silylisoquinolin-2-ium"),
]

# Full-namer cases that abstain at pin and emit only at the ``complete`` tier.
PIN_ABSTAIN_CASES = [
    ("[SiH3]C[CH-]C", "1-silylpropan-2-ide"),
    ("FC(F)(F)[CH-]C", "1,1,1-trifluoropropan-2-ide"),
    # Multi-charge dication: the multiplied suffix ("-1,4-diium") begins with
    # the consonant 'd' (di-), NOT a vowel -- (a) requires the parent's
    # terminal 'e' be RETAINED ("...diazine-1,4-diium", not "diazin-1,4-diium").
    # Regression case for the elision-conditional fix in _append_charge_suffix.
    ("C[n+]1cc[n+](C)cc1", "1,4-dimethyl-1,4-diazine-1,4-diium"),
]

# What the full namer ships across both tiers (pin-emit + complete-only).
FULL_NAMER_CASES = PIN_EMIT_CASES + PIN_ABSTAIN_CASES

# Engine-direct (``name_general``) emissions: identical to FULL_NAMER_CASES
# except the fused-ring cation, which is the von-Baeyer / skeletal-replacement
# form (engine-isolation-only, never shipped in production -- see the header).
ENGINE_DIRECT_CASES = [
    ("[SiH3][n+]1ccccc1", "1-silylpyridin-1-ium"),
    ("[GeH3][n+]1ccccc1", "1-germylpyridin-1-ium"),
    # change-asserted-value: the aza-cage heteroatom takes the LOWEST locant
    #: among valid von-Baeyer numberings of the equal-bridge
    # bicyclo[4.4.0] cage, heteroatoms get lowest locants). N is separated from a
    # bridgehead by one carbon, so its lowest attainable locant is 3, NOT 4. The
    # old `4-silyl-4-aza` expectation was a stale higher-locant numbering; the
    # engine now emits the lowest-locant form (deterministic across spellings,
    # OPSIN round-trips to the input).
    ("[SiH3][n+]1ccc2ccccc2c1",
     "3-silyl-3-azabicyclo[4.4.0]deca-1(10),2,4,6,8-pentaen-3-ium"),
    ("[SiH3]C[CH-]C", "1-silylpropan-2-ide"),
    ("FC(F)(F)[CH-]C", "1,1,1-trifluoropropan-2-ide"),
    ("C[n+]1cc[n+](C)cc1", "1,4-dimethyl-1,4-diazine-1,4-diium"),
]

# FAIL-CLOSED at the ENGINE (``name_general`` returns None; never a wrong /
# charge-dropped name).
FAILCLOSED_CASES = [
    "[CH2+]Cc1ccccc1",       # charge on a substituent carbon (parent = ring)
    "[CH2+]C[CH2]",          # radical-cation (the engine's radical refusal stays)
    "[SiH3][CH-]c1ccccc1",   # charge not expressible as a parent suffix
]

# The full complete tier still abstains on this one.
COMPLETE_FAILCLOSED_CASES = ["[SiH3][CH-]c1ccccc1"]

# The engine still refuses these two, but the full complete tier now NAMES them
# through another producer. Pre-existing-failures plan, Task 9 (TRIAGE.csv rows
# 120, 121; they were test_complete_fails_closed[...]), user decision D-b (plan
# 'User decisions (answered 2026-09-24)'): "D-b -> the policy wins: where a
# wider-tier name round-trips EXACTLY, change the test to assert an exact
# round-trip." Both names are OPSIN 2.9.0 RT exact (full InChIKey; the radical
# with OPSIN -r): 'propan-3-ylium-1-yl' (spelling as "Radical ions derived
# from parent hydrides", the Blue Book) and
# '1-(ethan-2-ylium-1-yl)cyclohexa-1,3,5-triene' (RT exact, but benzene is
# spelled as cyclohexatriene: a best-effort name, not a PIN). The test asserts
# the round trip, not those spellings.
COMPLETE_RT_CASES = ["[CH2+]Cc1ccccc1", "[CH2+]C[CH2]"]

# Round-trip the names the full namer actually ships.
_RT_NAMES = dict(FULL_NAMER_CASES)


# --------------------------------------------------------------------------
# Fixtures (mirror test_v26_p3_failclosed_routing)
# --------------------------------------------------------------------------
def _find_opsin_jar():
    """The pinned OPSIN jar via orthonym.jars (tests.support.jars), or None."""
    return jar_or_none()


@pytest.fixture
def production_gate(monkeypatch):
    """Re-enable the production OPSIN validity gate (the suite disables
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
@pytest.mark.parametrize("smiles,_name", ENGINE_DIRECT_CASES)
def test_pin_engine_inert_without_flag(smiles, _name):
    assert _engine_name(smiles, allow_aromatic_general=False) is None


@pytest.mark.parametrize("smiles,expected", ENGINE_DIRECT_CASES)
def test_charge_claims_are_declared_in_the_binding_spine(smiles, expected):
    """a phase Task 2: the charge-suffix producer now threads
    ``charge_atom_ids`` through a role='charge' binding, so P3's
    charge-totality proof stops being a permanent ``CHARGE_UNVERIFIED`` warn
    for every net-charged emission. The emitted NAME is unchanged (asserted
    against the SAME pinned ``ENGINE_DIRECT_CASES`` string the engine-direct
    tests use) -- only the binding ledger gained a new, E1-invisible entry.
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
# a phase Task 4 Part C: the wiring now promotes P8's stereo axis and
# P3's CHARGE_UNVERIFIED to error severity (``escalate=STRICT_STEREO_CHARGE_AXES``,
# ``mode`` stays "audit"). These two charge cases are the task's named charge
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
@pytest.mark.parametrize("smiles,expected", ENGINE_DIRECT_CASES)
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
# FULL-NAMER pin tier: PIN_EMIT_CASES emit their PIN, PIN_ABSTAIN_CASES abstain.
# FULL-NAMER complete tier: every full-namer case emits its expected name.
# Needs the production gate (else the suite ships unverified names).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", PIN_EMIT_CASES)
def test_pin_emits(smiles, expected, production_gate):
    """: these three are PINs, so the DEFAULT pin tier emits them rather
    than abstaining. `1-silylpyridin-1-ium`/`1-germylpyridin-1-ium` by direct
    analogy to `1-methylpyridin-1-ium (PIN)`; silyl/germyl are
    PIN-eligible preselected prefixes. The fused-ring cation ships
    the retained `2-silylisoquinolin-2-ium` /, worked example
    :42203) via the production ions.py onium router, NOT the P5 complete-tier
    lever. See internal notes."""
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: pin gave {out!r} != {expected!r}"


@pytest.mark.parametrize("smiles,expected", PIN_ABSTAIN_CASES)
def test_pin_abstains(smiles, expected, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected pin abstention for {smiles!r}, got {out!r}")


@pytest.mark.parametrize("smiles,expected", FULL_NAMER_CASES)
def test_complete_emits(smiles, expected, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


@pytest.mark.parametrize("smiles", COMPLETE_FAILCLOSED_CASES)
def test_complete_fails_closed(smiles, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert is_failure_name(out), f"{smiles}: expected abstention, got {out!r}"


@pytest.mark.parametrize("smiles", COMPLETE_RT_CASES)
def test_complete_names_rt_exact(smiles, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    canon = Chem.CanonSmiles(smiles)
    assert_full_rt(comp.name(canon), canon, what="complete tier: ")


# --------------------------------------------------------------------------
#: every name complete emits round-trips to the input structure.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,name", sorted(_RT_NAMES.items()))
def test_emitted_names_roundtrip(smiles, name, opsin_roundtrip):
    back = opsin_roundtrip.get(name)
    assert back, f"OPSIN could not parse {name!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{name!r} round-trips to {back!r}, not {smiles!r}")
