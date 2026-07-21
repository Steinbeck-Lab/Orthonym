"""v26 P6: retained-name preference over the general fallback (PIN-QUALITY
GUARDRAIL, ``complete`` tier only; Heritage A3).

P6 is 0-new-coverage BY DESIGN (V26-MILESTONE-DESIGN.md P6). It sits inside
the general-engine late-recovery (``namer.py:_try_general_engine_recovery``):
once the engine has already built a candidate von-Baeyer/replacement name
(``eng``/``cand``) and it cleared the E1 certificate, a structural recognizer
pass runs the EXISTING retained/fused-heterocycle catalogs
(``data.fused_heterocycles.get_fused_heterocycle_name``,
``data.retained_names.get_retained_name``, the ``heterocycles.py`` monocyclic
retained lookup) on the WHOLE input molecule. If one matches AND the name it
returns independently round-trips (SELF-01), that name replaces the general
``cand``; otherwise the general ``cand`` ships unchanged (fail-closed). This
is a PREFERENCE over an emission the engine already produced, never a
precondition to naming -- if the general engine could not name the molecule
at all (``eng is None``), P6 does not intervene (see NOT-APPLICABLE below).

REPRODUCE-FIRST (2026-07-21, production OPSIN gate on; `.venv/bin/python`,
bounded batches of <=40 molecules per process per the P3 hang-hazard rule):
swept ALL 184 entries of ``FUSED_HETEROCYCLE_DATA`` (the bare, unsubstituted
catalog molecules) through ``Orthonym(style="pin")``. Because
``get_fused_heterocycle_name``/``get_retained_name`` are ALREADY consulted
early on the PIN path itself (``fused_rings.py:1288``, ``heterocycles.py:
1106``), the overwhelming majority (180/184) are already named correctly at
``pin`` tier -- there is no PIN abstention for them to begin with, so P6 has
nothing to improve there (matches the design doc's own caveat: "the PIN path
may already catch all retained names before abstaining"). Exactly 4 bare
catalog molecules DO abstain at ``pin``:

  * ``benzo[f]quinoline`` (``c1ccc2c(c1)ccc1ncccc12``) -- pin abstains;
    ``complete`` (pre-P6) ships the ugly tricyclic von-Baeyer cage
    ``6-azatricyclo[8.4.0.0^2,7]tetradeca-1(14),2,4,6,8,10,12-heptaene``.
    ``get_fused_heterocycle_name`` returns ``'benzo[f]quinoline'``, which
    OPSIN-round-trips to the SAME structure -> real ACCEPT case. WHY pin
    abstains (PIN-PATH CONFOUND, traced via a direct probe): the SAME
    canonical SMILES also keys ``ALL_RETAINED_NAMES`` (consulted elsewhere
    on the PIN path, e.g. ``ring_assemblies.py``) to the WRONG, non-isomeric
    ``'benzo[h]quinoline'``; production's real SELF-01 gate silently
    suppresses that wrong PIN-path candidate to abstention -- so "pin
    abstains" here is itself a symptom of the SAME ``data/retained_names.py``
    defect the DATA-QUALITY finding below documents (out of P6 scope to
    fix). This means ``Orthonym(style="pin").name(...)`` for this molecule
    is NOT a safe Java-independent test double (see the note on
    ``test_preference_fires_under_complete``): with the module-level gate
    off (this suite's conftest default), ``pin.name()`` ships the WRONG
    ``'benzo[h]quinoline'`` directly, never reaching G2 recovery at all --
    only the REAL gate (``production_gate``) reproduces genuine abstention.
  * ``quinolizidine`` (``C1CCN2CCCCC2C1``) -- pin abstains (same PIN-PATH
    CONFOUND: ``get_retained_name`` on this canonical SMILES wrongly
    resolves to ``'decahydroisoquinoline'``, a different connectivity);
    BOTH ``valid`` and ``complete`` (pre-P6) ship the saturated bicyclic
    von-Baeyer name ``1-azabicyclo[4.4.0]decane``.
    ``get_fused_heterocycle_name`` returns ``'quinolizidine'``,
    OPSIN-round-trip-verified -> real ACCEPT case (P6 is gated to
    ``complete`` only per brief scope, so ``valid`` keeps shipping the
    von-Baeyer name -- a documented, deliberate residual, not a P6
    regression).
  * ``[1,3,2]benzodioxathiole`` (``c1ccc2c(c1)O[SH2]O2``, the lambda4-sulfur
    spiro-component ring) -- pin abstains; ``complete`` (pre-P6) ships
    ``7,9-dioxa-8lambda4-thiabicyclo[4.3.0]nona-1,3,5-triene``.
    ``get_fused_heterocycle_name`` DOES match and returns
    ``'[1,3,2]benzodioxathiole'`` -- but that bare name is only valid EMBEDDED
    in a larger spiro PIN with an external lambda-locant citing the
    hypervalent sulfur (see the catalog comment at
    ``data/fused_heterocycles.py:127-137``); standalone, OPSIN parses it to
    the ordinary divalent-sulfur isomer (``c1ccc2c(c1)OSO2``), a DIFFERENT
    structure. This is a real, naturally-occurring FAIL-CLOSED case: the
    recognizer fires, but its candidate does not round-trip, so P6 must keep
    the (uglier but structurally faithful) general name.
  * ``9H-fluoren-9-one`` (``O=C1c2ccccc2-c2ccccc21``) -- pin AND ``complete``
    (pre-P6) both abstain (the general engine cannot build any candidate for
    this cross-conjugated fused-ketone system at all: ``eng is None``). P6
    correctly does NOT intervene here (there is no ``cand`` to prefer over --
    this is the "not a precondition to naming" boundary from the design doc;
    rescuing this molecule would be NEW coverage, out of P6's 0-new-coverage
    scope). Documented, not exercised as an accept/fail-closed case.

DATA-QUALITY SIDE FINDING (documented, not fixed here -- out of P6 scope,
which only REUSES existing recognizers, never edits their catalogs): for
BOTH ACCEPT cases above, ``data.retained_names.get_retained_name`` on the
SAME canonical SMILES returns a DIFFERENT, non-isomeric name than
``get_fused_heterocycle_name`` -- ``benzo[h]quinoline`` (does not
OPSIN-round-trip to the input) and ``decahydroisoquinoline`` (does not
OPSIN-round-trip to the input) respectively. ``_retained_structural_
preference`` therefore tries ``get_fused_heterocycle_name`` FIRST and
short-circuits on a hit; this ordering is load-bearing (swapping it would
surface the bad ``ALL_RETAINED_NAMES`` entries), and is additionally backed
by the unconditional SELF-01 gate at the call site.

Harness mirrors test_v26_p5_charged_general.py: conftest force-disables the
production SELF-01 gate for the whole suite; ``production_gate`` re-enables
it (skipped without Java/OPSIN) for the full-namer cases; the
recognizer-direct and constructor-flag-wired cases are deterministic and
gate-independent (using ``_disable_opsin_validity_gate=True``, the
documented "test/internal mode" of the G2/P6 ladder, or a monkeypatched
oracle for the synthetic fail-closed proof).
"""
import glob
import shutil
import subprocess

import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name


pytestmark = pytest.mark.unit


# Real ACCEPT cases: pin ABSTAINS; complete (pre-P6) ships the von-Baeyer name;
# complete (post-P6) ships the retained name instead (SELF-01-verified).
ACCEPT_CASES = [
    ("c1ccc2c(c1)ccc1ncccc12", "benzo[f]quinoline",
     "6-azatricyclo[8.4.0.0^2,7]tetradeca-1(14),2,4,6,8,10,12-heptaene"),
    ("C1CCN2CCCCC2C1", "quinolizidine", "1-azabicyclo[4.4.0]decane"),
]

# Real FAIL-CLOSED case: the recognizer fires (finds a name), but that name
# does NOT independently round-trip -> complete must keep the general name,
# unchanged by P6.
FAILCLOSED_CASES = [
    ("c1ccc2c(c1)O[SH2]O2", "[1,3,2]benzodioxathiole",
     "7,9-dioxa-8lambda4-thiabicyclo[4.3.0]nona-1,3,5-triene"),
]

# Retained names the PIN path ALREADY emits (P6 must never touch these).
CONTROL_CASES = [
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("c1ccc2ncccc2c1", "quinoline"),
    ("c1ccc2ncncc2c1", "quinazoline"),
    ("c1ncc2nc[nH]c2n1", "9H-purine"),
]

_RT_NAMES = {name: smi for smi, name, _ in ACCEPT_CASES}


# --------------------------------------------------------------------------
# Fixtures (mirror test_v26_p5_charged_general.py)
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
    if not shutil.which("java") or _find_opsin_jar() is None:
        pytest.skip("OPSIN/Java not available for production-gate semantics")
    monkeypatch.setattr(_namer_mod, "_DISABLE_VALIDITY_GATE", False)


@pytest.fixture(scope="module")
def opsin_roundtrip():
    """name -> canonical SMILES via ONE OPSIN JVM (stdin batch)."""
    jar = _find_opsin_jar()
    if not shutil.which("java") or jar is None:
        pytest.skip("OPSIN/Java not available")
    names = list(_RT_NAMES.keys())
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


# --------------------------------------------------------------------------
# RECOGNIZER-DIRECT (deterministic, no Java): _retained_structural_preference
# is a pure structural lookup over the molecule graph.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected,_general", ACCEPT_CASES)
def test_recognizer_finds_retained_name(smiles, expected, _general):
    nm = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    got = nm._retained_structural_preference(mol)
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"


@pytest.mark.parametrize("smiles,expected,_general", FAILCLOSED_CASES)
def test_recognizer_finds_a_name_even_though_it_will_fail_closed(
        smiles, expected, _general):
    """The recognizer itself has no SELF-01 awareness -- it is expected to
    return the catalog name here; the CALLER (_try_general_engine_recovery)
    is what fails closed when this name does not round-trip standalone."""
    nm = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    got = nm._retained_structural_preference(mol)
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"


@pytest.mark.parametrize("smiles,expected", CONTROL_CASES)
def test_recognizer_agrees_with_pin_path_on_controls(smiles, expected):
    nm = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    got = nm._retained_structural_preference(mol)
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"


def test_recognizer_ordering_avoids_the_bad_retained_names_entry():
    """DATA-QUALITY side finding (reproduce-first): get_retained_name on
    benzo[f]quinoline's canonical SMILES returns the WRONG (non-isomeric)
    'benzo[h]quinoline'. The recognizer must try get_fused_heterocycle_name
    FIRST so this bad entry is never even consulted."""
    from orthonym.data.retained_names import get_retained_name
    canon = Chem.CanonSmiles("c1ccc2c(c1)ccc1ncccc12")
    assert get_retained_name(canon) == "benzo[h]quinoline"  # documents the bug
    nm = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(canon)
    assert nm._retained_structural_preference(mol) == "benzo[f]quinoline"


def test_recognizer_returns_none_when_nothing_matches():
    nm = Orthonym(style="pin")
    mol = Chem.MolFromSmiles("CCCCCCCCCCC")  # plain undecane, no ring
    assert nm._retained_structural_preference(mol) is None


# --------------------------------------------------------------------------
# CONSTRUCTOR-WIRED (deterministic, no Java): use the documented
# `_disable_opsin_validity_gate=True` test/internal mode of the ladder so the
# preference-fires / inert-without-flag behavior is provable without OPSIN.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected,_general", ACCEPT_CASES)
def test_preference_fires_under_complete(smiles, expected, _general):
    """Calls ``_try_general_engine_recovery`` DIRECTLY (not ``.name()``) --
    see the module docstring PIN-PATH CONFOUND note: under the test suite's
    conftest-default gate-off, ``.name()``'s EARLIER pipeline stage for these
    two molecules independently surfaces the SAME bad ``ALL_RETAINED_NAMES``
    entry this phase works around (unsuppressed only because the real
    production gate is off), so it is not a safe proxy for the G2/P6 ladder
    in isolation. Calling the recovery method directly exercises P6's own
    logic without that pre-existing, unrelated confound."""
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True,
                     _disable_opsin_validity_gate=True)
    got = comp._try_general_engine_recovery(Chem.CanonSmiles(smiles))
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"


@pytest.mark.parametrize("smiles,expected,general_name", ACCEPT_CASES)
def test_preference_inert_without_allow_aromatic_general(
        smiles, expected, general_name, monkeypatch):
    """`valid` tier (general_fallback=True, allow_aromatic_general=False)
    must NEVER consult the preference pass -- P6 is complete-tier only."""
    calls = []
    monkeypatch.setattr(
        Orthonym, "_retained_structural_preference",
        lambda self, mol: calls.append(mol) or expected)
    valid = Orthonym(style="pin", general_fallback=True,
                      allow_aromatic_general=False,
                      _disable_opsin_validity_gate=True)
    valid.name(Chem.CanonSmiles(smiles))
    assert calls == [], (
        "the retained preference pass must not run under valid tier "
        "(allow_aromatic_general=False)")


def test_preference_inert_without_general_fallback():
    """`pin` tier (general_fallback=False) never reaches the late-recovery
    method at all -- the sharpest byte-identity guarantee."""
    pin = Orthonym(style="pin")
    assert pin._try_general_engine_recovery(
        Chem.CanonSmiles("C1CCN2CCCCC2C1")) is None


# --------------------------------------------------------------------------
# FAIL-CLOSED (synthetic/forced, deterministic, no Java): a recognizer HIT
# whose candidate name does not round-trip must never override `cand`. This
# proves the mechanism even independent of the real-molecule case above.
# --------------------------------------------------------------------------
def test_preference_fails_closed_when_candidate_mismatches(monkeypatch):
    """Direct recovery-method call (see the PIN-PATH CONFOUND note on
    ``test_preference_fires_under_complete``)."""
    smi, _expected, general_name = ACCEPT_CASES[0]
    canon = Chem.CanonSmiles(smi)
    monkeypatch.setattr(
        Orthonym, "_retained_structural_preference",
        lambda self, mol: "totally-fabricated-name")
    monkeypatch.setattr(_namer_mod, "_validity_gate_jar_present",
                        lambda: True)

    def _fake_oracle(name):
        if name == "totally-fabricated-name":
            return "C"  # deliberately the WRONG structure
        if name == general_name:
            return canon  # the general name verifies normally (real result)
        return None

    monkeypatch.setattr(_namer_mod, "_validity_gate_name_to_smiles",
                        _fake_oracle)
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    got = comp._try_general_engine_recovery(canon)
    assert got == general_name, (
        f"a non-round-tripping retained candidate must never override the "
        f"general name: got {got!r}, expected {general_name!r}")


def test_preference_fails_closed_when_jar_absent(monkeypatch):
    """Direct recovery-method call (see the PIN-PATH CONFOUND note on
    ``test_preference_fires_under_complete``)."""
    smi, _expected, general_name = ACCEPT_CASES[0]
    canon = Chem.CanonSmiles(smi)
    monkeypatch.setattr(
        Orthonym, "_retained_structural_preference",
        lambda self, mol: "totally-fabricated-name")
    monkeypatch.setattr(_namer_mod, "_validity_gate_jar_present",
                        lambda: False)
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True,
                     general_fallback_unverified=True)
    got = comp._try_general_engine_recovery(canon)
    assert got == general_name, (
        f"no-JAR must never ship an unverified retained guess: got {got!r}, "
        f"expected {general_name!r}")


# --------------------------------------------------------------------------
# FULL-NAMER, real molecules, production gate (needs Java/OPSIN).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected,_general", ACCEPT_CASES)
def test_pin_abstains_accept_cases(smiles, expected, _general, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected pin abstention for {smiles!r}, got {out!r}")


@pytest.mark.parametrize("smiles,expected,_general", ACCEPT_CASES)
def test_complete_prefers_retained_name(smiles, expected, _general,
                                        production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


def test_valid_tier_keeps_von_baeyer_name_documented_scope(production_gate):
    """P6 is complete-tier only (brief scope): `valid` keeps shipping the
    von-Baeyer name for quinolizidine -- a documented, deliberate residual,
    not a regression this phase introduces."""
    smi, _expected, general_name = ACCEPT_CASES[1]  # quinolizidine
    valid = Orthonym(style="pin", general_fallback=True,
                      allow_aromatic_general=False)
    out = valid.name(Chem.CanonSmiles(smi))
    assert out == general_name


@pytest.mark.parametrize("smiles,retained_candidate,general_name",
                         FAILCLOSED_CASES)
def test_pin_abstains_failclosed_case(smiles, retained_candidate,
                                      general_name, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected pin abstention for {smiles!r}, got {out!r}")


@pytest.mark.parametrize("smiles,retained_candidate,general_name",
                         FAILCLOSED_CASES)
def test_complete_keeps_general_name_on_failclosed(
        smiles, retained_candidate, general_name, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == general_name, (
        f"{smiles}: expected the general name {general_name!r} to be kept "
        f"(retained candidate {retained_candidate!r} fails SELF-01 "
        f"standalone), got {out!r}")


@pytest.mark.parametrize("smiles,expected", CONTROL_CASES)
def test_control_cases_unchanged_under_pin_and_complete(smiles, expected,
                                                        production_gate):
    pin = Orthonym(style="pin")
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    canon = Chem.CanonSmiles(smiles)
    p = pin.name(canon)
    c = comp.name(canon)
    assert p == expected, f"{smiles}: pin gave {p!r} != {expected!r}"
    assert c == expected, f"{smiles}: complete gave {c!r} != {expected!r}"


# --------------------------------------------------------------------------
# SELF-01: the names P6 prefers round-trip to the input structure.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,name", sorted(
    ((smi, name) for smi, name, _ in ACCEPT_CASES)))
def test_preferred_names_roundtrip(smiles, name, opsin_roundtrip):
    back = opsin_roundtrip.get(name)
    assert back, f"OPSIN could not parse {name!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{name!r} round-trips to {back!r}, not {smiles!r}")
