""": fail-closed routing for default-path substituent-vocab misses.

Root cause fixed: ``fused_rings.get_fused_heterocycle_substituents`` silently
``continue``s past any exocyclic branch ``_identify_fused_substituent`` cannot
name (``sub_info is None``), so ``name_fused_heterocycle`` can assemble a name
that DROPS a whole substituent and denotes a DIFFERENT molecule -- a boronate /
methanesulfonyl / silyl / selanyl group on quinoline collapses to bare
``'quinoline'``. On the PIN/default path this group-dropping name is caught
only by the downstream OPSIN gate, which FAILS OPEN when the jar is
absent (the wrong name then ships), and it also BLOCKS the general engine from
being tried (the late-recovery only fires on a clean abstention).

P3 made ``name_fused_heterocycle`` fail closed at the source, first ONLY under
the general-engine tiers (``general_fallback`` set: ``valid`` / ``complete``), so
the PIN default path stayed byte-identical -- which meant the PIN path kept
returning the group-dropping ``'quinoline'``. Since 2026-09-25 (pre-existing-
failures plan, Task 4, TRIAGE row 88: '5-hydroxy-2,3-dihydro-1H-isoindole-1,3-
dione' for 5-hydroxythalidomide) it fails closed at EVERY tier:
"SUBSTITUTIVE NOMENCLATURE" (the Blue Book) cites every substituent as a
prefix or suffix, and a dropped branch is a different molecule. Under
``complete`` the decline re-routes the molecule through ``name_general`` (the
universal never-None substituent recursion, E1 + gated), which either
names it faithfully or abstains -- never the group-dropping name.

Reproduce-first (confirmed 2026-07-20, production OPSIN gate on):
  - DROP cases below: default pin RAW output (gate off) drops the substituent
    (``'quinoline'``); pin production (gate on) abstains; under ``complete`` the
    molecule is either corrected by the general engine (RT-OK) or fails closed
    (``unknown``) -- never the wrong bare-core name.
  - COVERAGE cases: pin abstains, ``complete`` emits an OPSIN-round-tripping name.

NOTE on the harness: ``conftest`` force-disables the production gate for
the whole suite (tests assert raw output). The full-namer cases here re-enable it
via ``production_gate`` (skipped without Java/OPSIN). The gated-routing cases are
deterministic and gate-independent: they set ``general_fallback_ctx`` directly.
"""
import glob
import shutil
import subprocess

import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name
from orthonym.rules.fused_rings import (
    name_fused_heterocycle, _exocyclic_atoms_accounted,
)
from orthonym.metrics.provenance import general_fallback_ctx
from tests.support.jars import jar_or_none


pytestmark = pytest.mark.unit


# DROP cases: name_fused_heterocycle DROPS the exocyclic group -> the bare
# catalog name (a DIFFERENT molecule). (SMILES, wrong-bare-name-on-pin).
DROP_CASES = [
    ("CB(O)Oc1ccc2ccccc2n1", "quinoline"),        # (dihydroxyboranyl)methoxy...
    ("CS(=O)(=O)c1ccc2ccccc2n1", "quinoline"),     # methanesulfonyl
    ("[Se]c1ccc2ccccc2n1", "quinoline"),           # selanyl (dropped at nfh)
    ("[SiH3]c1ccc2ccccc2n1", "quinoline"),         # silyl
]

# GOOD cases: the exocyclic group IS in vocabulary -> name_fused_heterocycle
# names it correctly. The gated guard MUST NOT touch these (accounted == True).
GOOD_CASES = [
    ("Cc1ccc2ccccc2n1", "2-methylquinoline"),
    ("COc1ccc2ccccc2n1", "2-methoxyquinoline"),
    ("O=C(O)c1ccc2ccccc2n1", "quinoline-2-carboxylic acid"),
    ("CN(C)c1ccc2ccccc2n1", "2-(dimethylamino)quinoline"),
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("Cc1nc2ccccc2[nH]1", "2-methyl-1H-benzimidazole"),
]

# Under complete: DROP case -> either fail closed ('unknown') or a general-engine
# name that OPSIN-round-trips. (SMILES, expected complete name or None=abstain).
COMPLETE_EXPECT = {
    "CB(O)Oc1ccc2ccccc2n1": None,                  # general engine also declines
    "CS(=O)(=O)c1ccc2ccccc2n1": None,
    "[Se]c1ccc2ccccc2n1": None,
    "[SiH3]c1ccc2ccccc2n1":
        "4-silyl-5-azabicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene",  # CORRECTED
}

# COVERAGE (C): pin abstains cleanly, complete emits an RT-OK general name.
COVERAGE_CASES = [
    ("[SiH3]c1ccccn1", "2-silylpyridine"),
    ("CCC(C)c1ccc2ccccc2c1",
     "4-(butan-2-yl)bicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene"),
    ("COCCc1ccc2ccccc2c1",
     "4-(2-methoxyethyl)bicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene"),
]

# Names emitted under complete that must OPSIN-round-trip to their input.
_RT_NAMES = {smi: COMPLETE_EXPECT[smi] for smi in COMPLETE_EXPECT
             if COMPLETE_EXPECT[smi]}
_RT_NAMES.update(dict(COVERAGE_CASES))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _find_opsin_jar():
    """The pinned OPSIN jar via orthonym.jars (tests.support.jars), or None."""
    return jar_or_none()


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


def _nfh(smiles):
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    res = name_fused_heterocycle(mol)
    return res[0] if res else None


# --------------------------------------------------------------------------
# PIN PATH: the GOOD cases are byte-identical, and the DROP cases fail closed
# (None) instead of the group-dropping legacy 'quinoline' (OPSIN reads
# 'quinoline' as C9H7N, a different molecule from each input). Deterministic,
# no Java.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,legacy", GOOD_CASES)
def test_pin_path_byte_identical(smiles, legacy):
    tok = general_fallback_ctx.set(False)  # PIN default path
    try:
        assert _nfh(smiles) == legacy, (
            f"PIN path changed for {smiles!r}: {_nfh(smiles)!r} != {legacy!r}")
    finally:
        general_fallback_ctx.reset(tok)


@pytest.mark.parametrize("smiles,wrong", DROP_CASES)
def test_drop_fails_closed_on_pin_path(smiles, wrong):
    tok = general_fallback_ctx.set(False)  # PIN default path
    try:
        assert _nfh(smiles) is None, (
            f"PIN path shipped the group-dropping {_nfh(smiles)!r} for {smiles!r}")
    finally:
        general_fallback_ctx.reset(tok)


# --------------------------------------------------------------------------
# FAIL-CLOSED ROUTING: under general_fallback the DROP cases decline (None) so
# the namer's late-recovery re-routes them; GOOD cases are untouched.
# Deterministic, no Java.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_legacy", DROP_CASES)
def test_drop_fails_closed_under_general_fallback(smiles, _legacy):
    tok = general_fallback_ctx.set(True)
    try:
        assert _nfh(smiles) is None, (
            f"expected fail-closed None for {smiles!r}, got {_nfh(smiles)!r}")
    finally:
        general_fallback_ctx.reset(tok)


@pytest.mark.parametrize("smiles,expected", GOOD_CASES)
def test_good_unchanged_under_general_fallback(smiles, expected):
    tok = general_fallback_ctx.set(True)
    try:
        assert _nfh(smiles) == expected, (
            f"GOOD case regressed under general_fallback: {smiles!r} -> "
            f"{_nfh(smiles)!r} != {expected!r}")
    finally:
        general_fallback_ctx.reset(tok)


@pytest.mark.parametrize("smiles,_legacy", DROP_CASES)
def test_drop_cases_are_unaccounted(smiles, _legacy):
    """The guard predicate: DROP cases leave an exocyclic atom unaccounted."""
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    from orthonym.rules.fused_rings import match_fused_heterocycle_core
    core = match_fused_heterocycle_core(mol)
    assert core is not None
    assert _exocyclic_atoms_accounted(mol, set(core[1])) is False


# --------------------------------------------------------------------------
# complete-tier full namer: a DROP case is NEVER the wrong bare-core name -- it
# is corrected by the general engine (RT-OK) or fails closed ('unknown').
# Needs the production gate (else the suite ships unverified names).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,wrong", DROP_CASES)
def test_complete_never_ships_the_dropped_name(smiles, wrong, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out != wrong, f"{smiles}: complete shipped the group-dropping {out!r}"
    expected = COMPLETE_EXPECT[smiles]
    if expected is None:
        assert is_failure_name(out), (
            f"{smiles}: expected fail-closed, got {out!r}")
    else:
        assert out == expected, f"{smiles}: {out!r} != {expected!r}"


# --------------------------------------------------------------------------
# COVERAGE (C): pin production ABSTAINS; complete emits the expected RT-OK name.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", COVERAGE_CASES)
def test_coverage_pin_abstains(smiles, expected, production_gate):
    pin = Orthonym(style="pin")
    out = pin.name(Chem.CanonSmiles(smiles))
    assert (not out) or is_failure_name(out), (
        f"expected pin abstention for {smiles!r}, got {out!r}")


@pytest.mark.parametrize("smiles,expected", COVERAGE_CASES)
def test_coverage_complete_emits(smiles, expected, production_gate):
    comp = Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True)
    out = comp.name(Chem.CanonSmiles(smiles))
    assert out == expected, f"{smiles}: complete gave {out!r} != {expected!r}"


# --------------------------------------------------------------------------
#: every name complete emits round-trips to the input structure.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,name", sorted(_RT_NAMES.items()))
def test_emitted_names_roundtrip(smiles, name, opsin_roundtrip):
    back = opsin_roundtrip.get(name)
    assert back, f"OPSIN could not parse {name!r}"
    assert Chem.CanonSmiles(back) == Chem.CanonSmiles(smiles), (
        f"{name!r} round-trips to {back!r}, not {smiles!r}")
