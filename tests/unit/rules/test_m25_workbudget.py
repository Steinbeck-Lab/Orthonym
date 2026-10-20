"""M2.5 — macrocycle compute-bound HANG guard (per-top-level work budgets).

Before this guard, a fully-reduced symmetric metallo-corrin, a large cyclic
glycopeptide (vancomycin) and a fused-thiazole thiopeptide spun compute-bound
for >20 min (external-watchdog SIGKILL) inside the recursive naming machinery:
the von-Baeyer main-ring path search / O(paths^2) pairing loop (corrins) and
the fused-heterocycle core matcher re-invoked thousands of times (peptides).
Two per-top-level budgets (``fragment_naming._PERF_BUDGET`` inner ops +
``_ANALYSIS_CALL_BUDGET`` expensive-analysis calls) now bound that work; on
exhaustion ``PerfBudgetExceeded`` unwinds to the outermost ``name`` and the
whole molecule abstains CLEANLY (inv 9 — never a partial / atom-dropped name).

Measured sizing (M2.5 Task 2B, `internal notes`):

  * hang witnesses: Ni/Fe-corrin ~7-12M inner ops; cob(III)yrinate 553 analyze
    calls; vancomycin/thiopeptide ~1000+ fused-matcher calls (all growing without
    bound) — every one blows past a budget in bounded, DETERMINISTIC work.
  * a 200-row drug-like pubchem sweep peaks at 17,988 inner ops / 138 calls, far
    under both budgets -> 0 nameable drug-like molecule abstains.

⚠ Mg-chlorophyll is NOT under the inner-op budget. Task 2B quoted 7.29M inner ops /
147 calls for it; measured 2026-10-20 (UN lane P) it exhausts the 12M budget at HEAD
(12,000,644 units, 1 trip, 16 analysis calls; 78 calls and a trip again at 120M) and
is still named, and with the Case-3 pair loop of the von Baeyer main-ring search
pruned (``polycyclic._consider_multiatom_bridges``) its whole main path is 36.8M units /
79 calls, so it still exhausts 12M and is still named. The figure was stale; what
this file pins for it is the name, not a budget margin.

⚠ The Ni/Fe corrins stop at the same 12M that the chlorophyll main path overruns, so
NO inner-op threshold can both abstain them quickly and let chlorophyll finish on the
main path (inv 16 — target shown unsatisfiable). The guarantee this test pins is the
categorical one: a >20-min compute-bound HANG becomes a bounded, deterministic
abstain-or-name. The wall bound below (55 s) is generous headroom over the measured
~28 s worst case at HEAD, NOT a performance target. The Case-3 prune made each unit a
more expensive one (the cheapest units, one bitmask AND per pair, left the count): a row
that still abstains at 12M now takes 1.4-1.9x as long (measured 2026-10-20, shared box:
Ni-corrin 14-15 s -> 21-25 s; 30 abstaining PubChem rows 689 s -> 977 s in all).
"""
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Generous OS-kill bound: >> the measured ~28 s worst case, << the pre-fix >20-min
# hang. A process that does NOT return inside this window is the hang (test fails).
_WALL_KILL_S = 55

# ── HANG witnesses that must now abstain CLEANLY (was: >20-min SIGKILL) ──────
# InChIKey / ChEBI provenance in a temp dir; SMILES pinned here as the contract.
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
# coordination name in COORDINATION_RETAINED ships via name's audited exit
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
    # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
    # nesting (the Blue Book) counts the stereo and compound-locant
    # parentheses,:7478) and ignores the von Baeyer brackets
    #,:7469), so the substituent is one level deep and takes '['
    # -- not the old '({...})', '(' around a '{' that nothing inside called for.
    # OPSIN RT exact.
    # 2026-09-26 (wp7) change-asserted-value: (the Blue Book) the
    # multiplier 'bis' is ignored, so 'bis(methoxycarbonyl)' keys at 'methoxycarbonyl',
    # before 'methoxyoxopropyl' ('c' < 'o'); the BB's '1-chloro-1,1-bis(4-methylphenyl)-
    # 3,3-bis(trifluoromethyl)-...benzoxabismole (PIN)' (:39307) orders bis prefixes the
    # same way. Was '...-14-ethenyl-5-(3-methoxy-3-oxopropyl)-22,23-bis(methoxycarbonyl)-...'.
    # OPSIN RT exact.
    "porphyrin": (
        _PORPHYRIN,
        "3-[(23S,24R)-14-ethenyl-22,23-bis(methoxycarbonyl)-5-(3-methoxy-3-"
        "oxopropyl)-4,10,15,24-tetramethyl-25,26,27,28-"
        "tetraazahexacyclo[16.6.1.1^3,6.1^8,11.1^13,16.0^19,24]octacosa-"
        "1,3(28),4,6,8,10,12,14,16(26),17,19,21-dodecaen-9-yl]propanoic acid"),
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
            # --on-failure label: since issue #45 the CLI prints an empty name by default;
            # these checks read the '(no name — <code>)' label.
            [sys.executable, "-m", "orthonym", smiles, "--emit-tier", "best-effort",
             "--on-failure", "label"],
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
    abstain (the honest '(no name —...)' sentinel) OR with a name that reads back
    to the input's full InChIKey -- never a wrong / partial / atom-dropped name
    (inv 9).

    The 'must abstain' clause was the guard's first form: the hang was the defect and
    an abstain the bounded outcome. Since the breadth work the thiopeptide and the
    vancomycin witnesses are NAMED at best-effort within the same wall bound (5 s and
    24 s here, bound 55 s), and OPSIN 2.9.0 reads each name back to the input's FULL
    InChIKey (an InChIKey; an InChIKey, stereo layer
    included). A name that round-trips exactly is not what the guard exists to stop;
    an unparseable, partial or atom-dropped one still fails here, as does a return
    past the wall bound (``_name_best_effort``)."""
    from tests.support.rt_assert import name_is_rt_exact
    out = _name_best_effort(smiles)
    assert out.startswith("(no name") or name_is_rt_exact(out, smiles), (
        f"{label}: expected a clean abstain or a name that round-trips exactly, "
        f"got {out[:80]!r}")


@pytest.mark.parametrize("label,smiles,expected", sorted(
    (k, v[0], v[1]) for k, v in _NAME_CONTROLS.items()))
def test_name_controls_unchanged(label, smiles, expected):
    """The negative controls (porphyrin, chlorophyll) and the curated
    coordination witness (Fe-corrin) still name byte-identically -- the guard
    catches ONLY the explosion class, never a nameable molecule."""
    out = _name_best_effort(smiles)
    assert out == expected, f"{label}: name changed.\n got: {out!r}\n exp: {expected!r}"


# ── Rows the pruned Case-3 pair loop names inside the 12M budget (UN lane P). ──
# At HEAD before the prune each abstained: the Case-3 loop of the von Baeyer main-ring
# search charged one unit per path pair (59-93% of the row's units) and exhausted 12M.
# Pinned: the best-effort name, byte-identical to the one the 120M budget gives, which
# OPSIN 2.9.0 reads back to the input's full InChIKey (a fresh call, outside the engine).
# These are best-effort names, not claimed PINs.
_PRUNE_ROWS = {
    "CID163924931": (
        "C1=CC=C(C=C1)C2=C3C4=CC=CC5=C4C(=CC=C5)C3=CC6=C2C=CC7=C6C=CC8=CC9=C(C=C87)C1=CC=CC2=C1C9"
        "=CC1=CC=CC=C21",
        "9-phenylundecacyclo[27.11.1.1^11,15.0^2,28.0^4,26.0^5,23.0^8,22.0^10,20.0^31,36.0^37,41."
        "0^19,42]dotetraconta-1(40),2,4,6,8(22),9,11,13,15,17,19(42),20,23,25,27,29,31,33,35,37(41),"
        "38-henicosaene"),
    "CID163757532": (
        "C1=CC(=CC=C1C2=CC3=C4C=CC5=C6C=C7C8=C9C(=CC=C1C9=C(C=C8)C8=C1C=C(C=C8)C1=CC=C(C=C1)F)"
        "C7=CC6=C1C5=C4C(=C3C=C2)C=C1)F",
        "11,28-bis(4-fluorophenyl)undecacyclo[23.9.2.2^5,8.0^2,21.0^4,19.0^6,18.0^7,15.0^9,14.0^22,"
        "35.0^26,31.0^32,36]octatriaconta-1(34),2,4,6,8,10,12,14,16,18,20,22,24,26(31),27,29,32,35,"
        "37-nonadecaene"),
    "CID162480633": (
        "B12C3=CC(=CC4=C3N(C5=CC6=CC=CC=C6C=C54)C7=CC(=CC(=C71)N8C9=CC1=CC=CC=C1C=C9C1=C8C2=CC(=C1)"
        "C)C1=CC=C(C=C1)C(C)(C)C)C",
        "20-(4-tert-butylphenyl)-4,36-dimethyl-17,23-diaza-1-boraundecacyclo[20.16.1.1^2,6.1^23,34."
        "0^7,16.0^9,14.0^18,39.0^24,33.0^26,31.0^17,41.0^38,40]hentetraconta-2(41),3,5,7,9,11,13,15,"
        "18(39),19,21,24,26,28,30,32,34,36,38(40)-nonadecaene"),
}


@pytest.mark.parametrize("label,smiles,expected", sorted(
    (k, v[0], v[1]) for k, v in _PRUNE_ROWS.items()))
def test_rows_named_inside_the_budget_by_the_pruned_case3_loop(label, smiles, expected):
    """Each row is named inside the default 12M budget, byte-identically to its 120M name,
    and the name reads back to the input's full InChIKey (stereo, charge and isotope layers
    included)."""
    from tests.support.rt_assert import name_is_rt_exact
    out = _name_best_effort(smiles)
    assert out == expected, f"{label}: name changed.\n got: {out!r}\n exp: {expected!r}"
    assert name_is_rt_exact(out, smiles), f"{label}: the name does not read back to the input"


def test_guard_is_deterministic():
    """The budgets are fixed operation/call counts reset at the outermost
    name, so the abstain decision is deterministic: naming the same explosive
    witness twice yields byte-identical output."""
    a = _name_best_effort(_COBYRINATE)
    b = _name_best_effort(_COBYRINATE)
    assert a == b == "(no name — UNSUPPORTED_ELEMENT)", (a, b)
