""": multi-fragment (adduct/solvate/co-crystal) split-name-join under
``complete``.

Root cause addressed: ``general_engine._common_refusal`` (and every single-
component handler) refuses ANY multi-fragment input
(``len(Chem.GetMolFrags(mol)) > 1``), so a neutral multi-component molecule
whose ONLY unnameable part is a general-engine-only fragment (e.g. a silyl-
heteroarene) abstains entirely -- even though every individual component is,
in fact, nameable.

P4 lifts this ONLY under the ``complete`` tier
(``general_fallback=True`` AND ``allow_aromatic_general=True``, wired via
``Orthonym._try_general_engine_recovery`` ->
``Orthonym._name_multifragment_complete``,
``src/orthonym/namer.py:2120``): split the mol into connected components,
name each NEUTRAL component through the ``complete``-tier single-component
pipeline (a fresh ``Orthonym`` instance, per the existing fresh-
instance pattern), order + join them via the existing adduct
assembler (``rules.adducts.name_adduct`` / ``_name_component``,
``src/orthonym/rules/adducts.py``), then run the SAME whole-string
gate the rest of the recovery path already applies. Fail-closed: if ANY
component is charged (P5 owns charged multi-fragment input) or unnameable by
ANY tier, or the assembled string fails, the whole recovery declines
(``None`` -> the pre-existing abstention/descriptive-fallback is kept
byte-identical) -- P4 never drops, placeholders, or partially joins a
component.

Reproduce-first (confirmed 2026-07-21, production OPSIN gate ON, single
bounded probe process -- see ``scripts/diagnose.py``-style batched-JVM RT):
  - COVERAGE cases: pin abstains cleanly (``unknown organic compound``);
    complete emits the per-component split-name-join and it round-trips.
  - BYTE-IDENTICAL cases: wherever the PIN path already names the
    multi-fragment input (existing handler on two PIN-nameable
    organics, the salt handler on a charged pair, or the frozen space-join
    on identical-component input), complete reproduces it EXACTLY -- P4 is a
    strict no-op there.
  - FAIL-CLOSED cases: a 3-fragment charged mixture the salt router can't
    resolve (``[NH4+].[Cl-].c1ccccc1``) and a neutral-but-unnameable
    organoiron component (``diphenyliron``) both stay byte-identical
    failure signals under complete -- never a wrong/partial joined name.

NOTE on the harness: ``conftest`` force-disables the production gate
for the whole suite (tests assert raw output). The full-namer cases here
re-enable it via ``production_gate`` (skipped without Java/OPSIN). The
direct ``rules.adducts`` function-level cases are deterministic and
gate-independent.
"""
import glob
import shutil
import subprocess

import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name
from orthonym.rules.adducts import _name_component, name_adduct


pytestmark = pytest.mark.unit


# COVERAGE: pin abstains cleanly (multi-fragment refusal, G3 scope); complete
# emits the per-component complete-tier split-name-join. (SMILES, expected
# complete name.)
COVERAGE_CASES = [
    ("[SiH3]c1ccccn1.O", "2-silylpyridine—water (1/1)"),
    # change-asserted-value: the aza-cage heteroatom takes the LOWEST locant
    # heteroatoms-lowest over the equal-bridge bicyclo[4.4.0]). Here N
    # is adjacent to a bridgehead, so it attains locant 2 and the silyl carbon 3;
    # the old `4-silyl-5-aza` was a stale higher-locant numbering. Engine now
    # emits `3-silyl-2-aza…` (deterministic; OPSIN round-trips). The pyridine
    # component (monocyclic) is unchanged.
    ("[SiH3]c1ccccn1.[SiH3]c1ccc2ccccc2n1",
     "3-silyl-2-azabicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene"
     "—2-silylpyridine (1/1)"),
]

# BYTE-IDENTICAL: the PIN path ALREADY names these (existing adduct
# handler on two PIN-nameable organics, the salt handler on a charged pair,
# the frozen space-join on identical-component input); complete MUST equal
# pin EXACTLY -- P4 must be a strict no-op whenever the PIN path succeeds.
BYTE_IDENTICAL_CASES = [
    ("c1ccccc1.O", "benzene—water (1/1)"),
    ("CCO.O", "ethanol—water (1/1)"),
    ("OCCO.OCCO", "ethane-1,2-diol ethane-1,2-diol"),
    ("CC(=O)[O-].[Na+]", "sodium acetate"),
]

# FAIL-CLOSED: neither pin nor complete emits a name -- P4 must NEVER ship a
# wrong/partial joined name.
# - a 3-fragment charged mixture (name_adduct's own charge != 0 refusal;
# the salt router also can't resolve NH4+/Cl-/benzene, so late recovery
# IS invoked and correctly declines rather than joining a partial name);
# - a neutral-but-unnameable component (diphenyliron has no organoiron
# support anywhere in the pipeline -- 'iron compound (not supported)' --
# so `_name_component` fails closed and `name_adduct` refuses the WHOLE
# adduct rather than dropping/placeholder-ing the Fe component).
FAIL_CLOSED_CASES = [
    ("[NH4+].[Cl-].c1ccccc1", "unknown organic compound"),
    ("[Fe](c1ccccc1)c1ccccc1.O", "iron compound (not supported)"),
]

# Single-fragment input must be completely unaffected by the P4 branch (the
# pre-existing `else` path in `_try_general_engine_recovery`).
SINGLE_FRAGMENT_UNCHANGED_CASES = [
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
]

# Names emitted under complete that must OPSIN-round-trip to their input.
_RT_NAMES = dict(COVERAGE_CASES + BYTE_IDENTICAL_CASES)


# --------------------------------------------------------------------------
# Fixtures (pattern identical to test_v26_p3_failclosed_routing.py)
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
    """Re-enable the production OPSIN validity gate (the suite disables
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


def _pin():
    return Orthonym(style="pin")


def _complete():
    return Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)


# --------------------------------------------------------------------------
# PIN abstains cleanly on the general-only-component cases. NOTE: this needs
# the production_gate: for the second case
# ('[SiH3]c1ccccn1.[SiH3]c1ccc2ccccc2n1'), one component
# ('[SiH3]c1ccc2ccccc2n1') is a P3 fused-heterocycle group-DROP case (the
# silyl substituent is dropped, yielding the bare wrong name 'quinoline');
# only the production gate catches that and turns it into a clean
# abstention (reproduced: with the gate off, raw pin emits 'quinoline').
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", COVERAGE_CASES)
def test_pin_abstains_on_general_only_component(smiles, _expected,
                                                production_gate):
    out = _pin().name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected pin abstention for {smiles!r}, got {out!r}")


# --------------------------------------------------------------------------
# complete emits the expected per-component split-name-join (needs the
# production gate: the emission must be -verified, matching the P3
# full-namer pattern).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", COVERAGE_CASES)
def test_complete_emits_split_name_join(smiles, expected, production_gate):
    out = _complete().name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


# --------------------------------------------------------------------------
# PIN BYTE-IDENTITY: wherever the PIN path already names a multi-fragment
# input, complete reproduces it EXACTLY.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", BYTE_IDENTICAL_CASES)
def test_pin_matches_expected(smiles, expected, production_gate):
    out = _pin().name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: pin gave {out!r} != {expected!r}"


@pytest.mark.parametrize("smiles,expected", BYTE_IDENTICAL_CASES)
def test_complete_byte_identical_to_pin(smiles, expected, production_gate):
    pin_out = _pin().name(Chem.CanonSmiles(smiles))
    comp_out = _complete().name(Chem.CanonSmiles(smiles))
    assert comp_out == pin_out == expected, (
        f"{smiles}: pin={pin_out!r} complete={comp_out!r} "
        f"expected={expected!r}")


# --------------------------------------------------------------------------
# FAIL-CLOSED: complete NEVER emits a wrong/partial name for a charged
# multi-fragment mixture or an unnameable neutral component -- it stays
# byte-identical to the pin abstention/descriptive fallback.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected_failure", FAIL_CLOSED_CASES)
def test_fail_closed_never_ships_partial_name(smiles, expected_failure,
                                              production_gate):
    pin_out = _pin().name(Chem.CanonSmiles(smiles))
    comp_out = _complete().name(Chem.CanonSmiles(smiles))
    assert is_failure_name(pin_out), (
        f"{smiles}: expected pin failure signal, got {pin_out!r}")
    assert is_failure_name(comp_out), (
        f"{smiles}: expected complete failure signal, got {comp_out!r}")
    assert comp_out == pin_out == expected_failure, (
        f"{smiles}: pin={pin_out!r} complete={comp_out!r} "
        f"expected={expected_failure!r}")


# --------------------------------------------------------------------------
# Single-fragment input is untouched by the P4 branch.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", SINGLE_FRAGMENT_UNCHANGED_CASES)
def test_single_fragment_path_unaffected(smiles, expected, production_gate):
    pin_out = _pin().name(Chem.CanonSmiles(smiles))
    comp_out = _complete().name(Chem.CanonSmiles(smiles))
    assert pin_out == comp_out == expected, (
        f"{smiles}: pin={pin_out!r} complete={comp_out!r} "
        f"expected={expected!r}")


# --------------------------------------------------------------------------
# 'valid' tier (general_fallback=True, allow_aromatic_general=False) must
# NOT engage the P4 multi-fragment recovery -- only 'complete'
# (allow_aromatic_general=True) does. Deterministic (no Java): pins the
# `if not self._allow_aromatic_general: return None` gate in
# `_name_multifragment_complete` (namer.py:2132).
# --------------------------------------------------------------------------
def test_multifragment_recovery_is_complete_tier_only():
    valid_tier = Orthonym(style="pin", general_fallback=True,
                          allow_aromatic_general=False)
    out = valid_tier.name(Chem.CanonSmiles("[SiH3]c1ccccn1.O"))
    assert (not out) or is_failure_name(out), (
        f"'valid' tier must not run the complete-only multi-fragment "
        f"recovery, got {out!r}")


# --------------------------------------------------------------------------
# rules.adducts function-level: deterministic, gate-independent. Directly
# pins the kwarg contract -- kwargs OMITTED (pre-P4 call signature)
# reproduce the pre-P4 PIN-scope refusal; kwargs supplied opt into the
# complete-tier per-component namer.
# --------------------------------------------------------------------------
def test_name_component_pin_default_declines_general_only_fragment():
    smi = Chem.CanonSmiles("[SiH3]c1ccccn1")
    assert _name_component(smi, "pin") is None
    assert _name_component(
        smi, "pin", general_fallback=True,
        allow_aromatic_general=True) == "2-silylpyridine"


def test_name_adduct_pin_default_declines_general_only_component():
    mol = Chem.MolFromSmiles(Chem.CanonSmiles("[SiH3]c1ccccn1.O"))
    assert name_adduct(mol, style="pin") is None
    assert name_adduct(
        mol, style="pin", general_fallback=True,
        allow_aromatic_general=True) == "2-silylpyridine—water (1/1)"


def test_name_adduct_fails_closed_on_unnameable_component():
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(
        "[Fe](c1ccccc1)c1ccccc1.O"))
    assert name_adduct(
        mol, style="pin", general_fallback=True,
        allow_aromatic_general=True) is None


def test_name_adduct_declines_charged_fragment_even_under_complete():
    mol = Chem.MolFromSmiles(Chem.CanonSmiles("CC(=O)[O-].[Na+]"))
    assert name_adduct(
        mol, style="pin", general_fallback=True,
        allow_aromatic_general=True) is None


# --------------------------------------------------------------------------
#: every non-failure name in the case tables above OPSIN-round-trips
# to its input structure.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,name", sorted(_RT_NAMES.items()))
def test_emitted_names_roundtrip(smiles, name, opsin_roundtrip):
    back = opsin_roundtrip.get(name)
    assert back, f"OPSIN could not parse {name!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{name!r} round-trips to {back!r}, not {smiles!r}")
