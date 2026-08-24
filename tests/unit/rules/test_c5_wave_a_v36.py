"""v36 Milestone-C Wave A -- two small, pre-pinpointed, independent 0-wrong fixes.

FIX 1 (rules/polycyclics.py:101): ``identify_polycyclic`` calls
``mol.GetSubstructMatches(pattern)`` with no ``maxMatches``/``uniquify`` cap, yet only
ever consumes ``matches[0]``. On a highly-symmetric all-carbon giant cage (a fullerene,
e.g. C70 with 37 SSSR rings) this enumerates a combinatorial number of automorphic
matches and spins (measured 63.1s, `
hang.md``). Fix: bound the call (``maxMatches=1, uniquify=True`` -- behaviour-preserving,
only ``matches[0]`` is ever read) AND add an all-carbon giant-ring-count scope guard so a
fullerene declines FAST instead of spinning (fullerenes are explicitly out-of-scope,
project the contributor guide). An in-scope mixed cage (aspidosperma-shaped,
``c1cc2c(c3c1CNC3)O[C@@]1(CCC[C@H]3CCCC[C@@H]31)C2``, VERIFIED 1.9s -> abstain) must keep
naming/abstaining exactly as today, just fast.

FIX 2 (assembly/substituent_prefix_forms.py::get_sulfanyl_prefix): the existing
fail-closed guard only checks bond ORDER (``GetBondTypeAsDouble() >= 2.0``), which misses
the charge-separated sulfoxide ``C[S+]([O-])CC`` -- its S-O bond is order 1.0, so the
function still builds ``ethylsulfanyl`` (silently dropping the [O-], a wrong molecule
today only caught downstream by SELF-01/RT). Fix: fail closed on
``sulfur.GetDegree() != 2`` (divalent-S contract, degree not just bond order). Sibling
functions ``get_sulfinyl_prefix``/``get_sulfonyl_prefix`` are untouched and must stay
byte-identical (canaries below).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# FIX 1 -- fullerene hang bound
# ---------------------------------------------------------------------------

# VERIFIED (fullerene_probe.py, prior session spy): C70, all-carbon, 37 SSSR rings.
FULLERENE_C70 = (
    "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c%12c%13c%14c%15c%16"
    "c%17c%14c%14c%18c%13c%11c1c1c5c%10c5c(c%14c%10c%17c%11c%13c%16c%14c%16c%"
    "15c%12c%12c%16c(c2c7c%126)c2c8c3c(c%13c%142)c2c9c4c5c%10c%112)c%181"
)

# VERIFIED in-scope mixed cage (aspidosperma-shaped) -- terminates ~1.9s -> abstain today.
INSCOPE_MIXED_CAGE = "c1cc2c(c3c1CNC3)O[C@@]1(CCC[C@H]3CCCC[C@@H]31)C2"

# Regression canary: adamantane, a plain von-Baeyer cage that must keep NAMING.
ADAMANTANE = "C1C2CC3CC1CC(C2)C3"


def _run_bounded(smiles: str, inner_alarm_s: int = 20, hard_kill_s: int = 35) -> dict:
    """Name ``smiles`` in a fresh subprocess. The subprocess self-bounds with
    ``signal.alarm(inner_alarm_s)``; the PARENT additionally hard-kills the child at
    ``hard_kill_s`` via ``subprocess.run(timeout=...)`` (an OS-level SIGKILL from
    outside the process), so the wall-clock bound holds even if a busy RDKit C loop
    does not yield to the in-process signal handler. Returns
    ``{"hung": bool, "result": str|None, "stderr": str}``.
    """
    script = textwrap.dedent(f"""
        import signal

        class _T(Exception):
            pass

        def _h(signum, frame):
            raise _T()

        signal.signal(signal.SIGALRM, _h)
        signal.alarm({inner_alarm_s})
        try:
            from orthonym.jvm_budget import jvm_slots
            with jvm_slots(1, purpose="v36-c5-wave-a-fullerene-bound"):
                from orthonym.namer import name_compound
                out = name_compound({smiles!r})
            signal.alarm(0)
            print("RESULT:" + repr(out))
        except _T:
            print("RESULT:TIMEOUT")
        """)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=hard_kill_s,
        )
    except subprocess.TimeoutExpired:
        return {"hung": True, "result": None, "stderr": ""}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return {"hung": False, "result": line[len("RESULT:"):], "stderr": proc.stderr[-2000:]}
    return {"hung": False, "result": None, "stderr": proc.stderr[-2000:] + proc.stdout[-2000:]}


def test_fullerene_abstains_within_bound():
    """RED at HEAD before the fix: the C70 fullerene spins ~63s and exceeds the
    35s hard bound. After the fix it must decline FAST (abstain / 'unknown organic
    compound' / TIMEOUT-never), never a spin past the bound."""
    res = _run_bounded(FULLERENE_C70, inner_alarm_s=20, hard_kill_s=35)
    assert not res["hung"], (
        f"fullerene naming exceeded the {35}s hard bound (the GetSubstructMatches "
        f"hang is not fixed): {res}")
    result = res["result"]
    assert result is not None, f"no result captured: {res}"
    assert result != "RESULT:TIMEOUT" and "TIMEOUT" not in result, (
        f"the in-process alarm fired (>20s) even though the hard kill did not: {res}")
    # An out-of-scope giant cage must abstain, never fabricate a wrong PAH name.
    assert "unknown" in result.lower() or result in ("None", "''"), (
        f"fullerene should abstain (out-of-scope giant cage), got {result!r}")


def test_inscope_mixed_cage_still_resolves_within_bound():
    """Regression: an in-scope cage must keep terminating fast (name or abstain,
    but never hang) -- the fix must not change behaviour for non-degenerate cages."""
    res = _run_bounded(INSCOPE_MIXED_CAGE, inner_alarm_s=20, hard_kill_s=35)
    assert not res["hung"], f"in-scope mixed cage regressed to hanging: {res}"
    assert res["result"] and "TIMEOUT" not in res["result"], (
        f"in-scope mixed cage did not resolve cleanly: {res}")


def test_adamantane_still_names_within_bound():
    """Regression: a plain von-Baeyer cage must still be NAMED (not just abstain)."""
    res = _run_bounded(ADAMANTANE, inner_alarm_s=20, hard_kill_s=35)
    assert not res["hung"], f"adamantane regressed to hanging: {res}"
    assert res["result"] == "'adamantane'", f"adamantane name regressed: {res}"


# ---------------------------------------------------------------------------
# FIX 2 -- sulfanyl guard by S-degree (charge-separated sulfoxide off-spelling)
# ---------------------------------------------------------------------------


def test_charge_separated_sulfoxide_no_longer_ethylsulfanyl():
    """C[S+]([O-])CC (charge-separated ethyl methyl sulfoxide): the S-O bond is
    order 1.0, so the OLD bond-order-only guard missed it and get_sulfanyl_prefix
    built 'ethylsulfanyl', silently dropping the [O-] (a wrong molecule). After the
    fix (fail closed on GetDegree() != 2) it must NOT produce that wrong prefix
    anywhere in the emitted name."""
    import orthonym
    name = orthonym.name_compound("C[S+]([O-])CC", style="pin")
    assert "sulfanyl" not in name.lower(), (
        f"charge-separated sulfoxide still emits a sulfanyl-family name: {name!r}")
    assert "ethylsulfanyl" not in name.lower() and "methylsulfanyl" not in name.lower()


def test_get_sulfanyl_prefix_returns_none_for_charge_separated_sulfoxide():
    """Direct unit check on the predicate itself (root-cause site)."""
    from rdkit import Chem
    from orthonym.assembly.substituent_prefix_forms import get_sulfanyl_prefix

    mol = Chem.MolFromSmiles("C[S+]([O-])CC")
    s_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")
    s_atom = mol.GetAtomWithIdx(s_idx)
    c_neighbors = [n.GetIdx() for n in s_atom.GetNeighbors() if n.GetSymbol() == "C"]
    assert len(c_neighbors) == 2
    result = get_sulfanyl_prefix(mol, (s_idx, c_neighbors[0], c_neighbors[1]), None)
    assert result is None, (
        f"get_sulfanyl_prefix must fail closed (degree != 2) on a charge-separated "
        f"sulfoxide S, got {result!r}")


def test_genuine_thioether_still_names_sulfanyl():
    """Regression: a real divalent thioether (CSCC) must be UNAFFECTED -- the
    degree-2 S is exactly the function's contract."""
    import orthonym
    name = orthonym.name_compound("CSCC", style="pin")
    assert name == "ethyl methyl sulfide" or "sulfanyl" in name.lower(), (
        f"genuine thioether naming regressed: {name!r}")


def test_get_sulfanyl_prefix_still_builds_for_genuine_thioether():
    """Direct unit check: CSCC's S (degree 2, both neighbours carbon) must still
    build a compound prefix (not None)."""
    from rdkit import Chem
    from orthonym.assembly.substituent_prefix_forms import get_sulfanyl_prefix

    mol = Chem.MolFromSmiles("CSCC")
    s_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")
    s_atom = mol.GetAtomWithIdx(s_idx)
    assert s_atom.GetDegree() == 2
    c_neighbors = [n.GetIdx() for n in s_atom.GetNeighbors() if n.GetSymbol() == "C"]
    result = get_sulfanyl_prefix(
        mol, (s_idx, c_neighbors[0], c_neighbors[1]), [c_neighbors[0]])
    assert result is not None and "sulfanyl" in result, (
        f"genuine thioether predicate regressed: {result!r}")


def test_sulfinyl_canary_unchanged():
    """(methanesulfinyl)benzene CS(=O)c1ccccc1 -- sibling get_sulfinyl_prefix path,
    must stay byte-identical (untouched by the get_sulfanyl_prefix fix)."""
    import orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = orthonym.name_compound("CS(=O)c1ccccc1", style="pin")
    assert name == "(methanesulfinyl)benzene", f"sulfinyl canary regressed: {name!r}"
    assert opsin_roundtrip_check("CS(=O)c1ccccc1", name)["passed"]


def test_sulfonyl_canary_unchanged():
    """(methanesulfonyl)benzene CS(=O)(=O)c1ccccc1 -- sibling get_sulfonyl_prefix
    path, must stay byte-identical."""
    import orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = orthonym.name_compound("CS(=O)(=O)c1ccccc1", style="pin")
    assert name == "(methanesulfonyl)benzene", f"sulfonyl canary regressed: {name!r}"
    assert opsin_roundtrip_check("CS(=O)(=O)c1ccccc1", name)["passed"]
