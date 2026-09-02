"""M2.5 — macrocycle compute-bound HANG guard (per-top-level work budgets).

Before this guard, a fully-reduced symmetric metallo-corrin, a large cyclic
glycopeptide (vancomycin) and a fused-thiazole thiopeptide spun compute-bound
for >20 min (external-watchdog SIGKILL) inside the recursive naming machinery:
the von-Baeyer main-ring path search / O(paths^2) pairing loop (corrins) and
the fused-heterocycle core matcher re-invoked thousands of times (peptides).
Two per-top-level budgets (``fragment_naming._PERF_BUDGET`` inner ops +
``_ANALYSIS_CALL_BUDGET`` expensive-analysis calls) now bound that work; on
exhaustion ``PerfBudgetExceeded`` unwinds to the outermost ``name()`` and the
whole molecule abstains CLEANLY (inv 9 — never a partial / atom-dropped name).

Measured sizing (M2.5 Task 2B, ``.superpowers/sdd/M2-STEREO-PLAN/task-m25b-report.md``):

  * hang witnesses:   Ni/Fe-corrin ~7-12M inner ops; cob(III)yrinate 553 analyze
    calls; vancomycin/thiopeptide ~1000+ fused-matcher calls (all growing without
    bound) — every one blows past a budget in bounded, DETERMINISTIC work.
  * nameable ceiling: Mg-chlorophyll (a genuine non-curated name) needs 7.29M
    inner ops / 147 calls; a 200-row drug-like pubchem sweep peaks at 17,988 inner
    ops / 138 calls. Both budgets clear all of these -> 0 nameable molecule abstains.

⚠ The Ni/Fe corrins do LESS total inner work in their first ~14 s than chlorophyll
does in its ENTIRE nameable run (both ~7M inner ops), so NO inner-op threshold that
still names chlorophyll can abstain them in <10 s (inv 16 — target shown
unsatisfiable). The guarantee this test pins is the categorical one: a >20-min
compute-bound HANG becomes a bounded, deterministic abstain-or-name. The wall
bound below (55 s) is generous headroom over the measured ~28 s worst case, NOT a
performance target.
"""
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Generous OS-kill bound: >> the measured ~28 s worst case, << the pre-fix >20-min
# hang. A process that does NOT return inside this window is the hang (test fails).
_WALL_KILL_S = 55

# ── HANG witnesses that must now abstain CLEANLY (was: >20-min SIGKILL) ──────
# InChIKey / ChEBI provenance in scratchpad; SMILES pinned here as the contract.
_NI_CORRIN = ("C[C@@]1(CC(=O)[O-])C2=CC3=[N+]4C(=Cc5c(CC(=O)[O-])c(CCC(=O)[O-])c6"
              "[n]5[Ni-2]45[N]2C(=CC2=[N+]5C(=C6)C(CCC(=O)[O-])=C2CC(=O)[O-])"
              "[C@H]1CCC(=O)[O-])[C@@](C)(CC(=O)[O-])[C@@H]3CCC(=O)[O-]")
_COBYRINATE = ("C/C1=C2/[N]([Co+])[C@H]([C@H](CC(=O)O)[C@@]2(C)CCC(=O)O)[C@]2(C)"
               "N=C(/C(C)=C3\\N=C(/C=C4\\N=C1[C@@H](CCC(=O)O)C4(C)C)"
               "[C@@H](CCC(=O)O)[C@]3(C)CC(=O)O)[C@@H](CCC(=O)O)[C@]2(C)CC(=O)O")
_VANCOMYCIN = ("CN[C@@H](C(=O)N[C@H]1C(=O)N[C@@H](c2ccc(O)cc2)C(=O)N[C@H]2C(=O)"
               "N[C@H]3C(=O)N[C@H](C(=O)N[C@H](C(=O)O)c4cc(O)cc(O)c4-c4cc3ccc4O)"
               "[C@H](O[C@H]3C[C@@H](N)[C@@H](O)[C@H](C)O3)c3ccc(cc3)Oc3cc2cc(c3O"
               "[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)C2O[C@H]2C[C@@H](N)[C@@H](O)"
               "[C@H](C)O2)Oc2ccc(cc2)[C@H]1O)c1ccc(O[C@@H]2O[C@@H](C)[C@H](O)"
               "[C@@H](O)[C@H]2O)cc1")
_THIOPEPTIDE = ("CNC(=O)CC1NC(=O)c2csc(n2)-c2ccc(-c3nc(C(=O)OCC4NC(=O)C5CCCN5C4=O)"
                "cs3)nc2-c2csc(n2)-c2csc(n2)C(C(C)C)NC(=O)CNC(=O)c2csc(n2)"
                "C(C(C)C)NC(=O)c2nc1sc2C")

_HANG_WITNESSES = {
    "Ni-corrin": _NI_CORRIN,
    "cobyrinate": _COBYRINATE,
    "vancomycin": _VANCOMYCIN,
    "thiopeptide": _THIOPEPTIDE,
}

# ── Molecules that must still NAME byte-identically (was: 2 porphyrins that
# always named + Fe-corrin, a hang witness whose exact-InChIKey ChEBI curated
# coordination name in COORDINATION_RETAINED ships via name()'s audited exit
# regardless of the budget -- so the budget makes it TERMINATE, unchanged). ─────
_FE_CORRIN = ("C[C@@]1(CC(=O)O)C2=CC3=[N+]4C(=Cc5c(CCC(=O)O)c(CC(=O)O)c6[n]5"
              "[Fe-2]45[N]2C(=CC2=[N+]5C(=C6)[C@@H](CCC(=O)O)[C@]2(C)CC(=O)O)"
              "[C@H]1CCC(=O)O)C(CCC(=O)O)=C3CC(=O)O")
_PORPHYRIN = ("C=CC1=C(C)c2cc3[nH]c(cc4nc(cc5[nH]c(cc1n2)c(C)c5CCC(=O)O)"
              "C(CCC(=O)OC)=C4C)[C@@]1(C)C3=CC=C(C(=O)OC)[C@H]1C(=O)OC")
_CHLOROPHYLL = ("C=CC1=C(C)C2=Cc3c(C=C)c(C)c4[n]3[Mg-2]35[n]6c(c(C)c7c6=C(C6=[N+]3"
                "C(=C4)[C@@H](C)[C@@H]6CCC(=O)OC/C=C(\\C)CCC[C@H](C)CCC[C@H](C)"
                "CCCC(C)C)[C-](C(=O)OC)C7=O)=CC1=[N+]25")

_NAME_CONTROLS = {
    "Fe-corrin": (
        _FE_CORRIN,
        "[3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-"
        "8,13-dimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-"
        "kappaN(21),kappaN(22),kappaN(23),kappaN(24)]tetrapropanoato(2-)]iron"),
    "porphyrin": (
        _PORPHYRIN,
        "3-({(23S,24R)-14-ethenyl-5-(3-methoxy-3-oxopropyl)-22,23-"
        "bis(methoxycarbonyl)-4,10,15,24-tetramethyl-25,26,27,28-"
        "tetraazahexacyclo[16.6.1.1^3,6.1^8,11.1^13,16.0^19,24]octacosa-"
        "1,3(28),4,6,8,10,12,14,16(26),17,19,21-dodecaen-9-yl})propanoic acid"),
    "chlorophyll": (
        _CHLOROPHYLL,
        "[methyl (3S,4S)-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-"
        "3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-9,14-divinylphorbine-"
        "21-carboxylatato(3-)-kappa(4)N(23),N(24),N(25),N(26)]magnesate(1-)"),
}


def _name_best_effort(smiles: str, timeout_s: int = _WALL_KILL_S) -> str:
    """Name one SMILES at best-effort in a FRESH subprocess under an OS-level
    kill. Returns the emitted line. A ``TimeoutExpired`` (the process did not
    return -> the hang is back) fails the calling test."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "orthonym", smiles, "--emit-tier", "best-effort"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"macrocycle-hang guard FAILED: naming did not return within "
            f"{timeout_s}s (compute-bound hang) for {smiles[:40]}...")
    assert proc.returncode == 0, (
        f"non-zero exit {proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout.strip()


@pytest.mark.parametrize("label,smiles", sorted(_HANG_WITNESSES.items()))
def test_hang_witness_abstains_cleanly(label, smiles):
    """Each compute-bound-hang witness returns in bounded time with a CLEAN
    abstain (the honest '(no name — ...)' sentinel), never a wrong / partial /
    atom-dropped name (inv 9)."""
    out = _name_best_effort(smiles)
    assert out.startswith("(no name"), (
        f"{label}: expected a clean abstain, got {out[:80]!r}")


@pytest.mark.parametrize("label,smiles,expected", sorted(
    (k, v[0], v[1]) for k, v in _NAME_CONTROLS.items()))
def test_name_controls_unchanged(label, smiles, expected):
    """The negative controls (porphyrin, chlorophyll) and the curated
    coordination witness (Fe-corrin) still name byte-identically -- the guard
    catches ONLY the explosion class, never a nameable molecule."""
    out = _name_best_effort(smiles)
    assert out == expected, f"{label}: name changed.\n got: {out!r}\n exp: {expected!r}"


def test_guard_is_deterministic():
    """The budgets are fixed operation/call counts reset at the outermost
    name(), so the abstain decision is deterministic: naming the same explosive
    witness twice yields byte-identical output."""
    a = _name_best_effort(_COBYRINATE)
    b = _name_best_effort(_COBYRINATE)
    assert a == b == "(no name — UNSUPPORTED_ELEMENT)", (a, b)
