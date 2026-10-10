"""Fragment naming infrastructure with cycle-detection guard.

Provides thread-safe cycle detection to prevent infinite loops when
fragment naming calls name_compound recursively. Uses a visited-SMILES
set (threading.local pattern) instead of an arbitrary depth counter.

The visited set tracks which SMILES are currently being named up the call
stack. If a SMILES is encountered that's already being processed, cycle
detection returns None to break the recursion. A safety-net maximum
visited set size (20) prevents unbounded recursion from decomposition
chains where every fragment SMILES is unique.

Usage:
    from orthonym.assembly.fragment_naming import name_fragment_recursively

    # Inside a naming function that needs to recursively name a sub-fragment:
    fragment_name = name_fragment_recursively("CCO")
    if fragment_name is None:
        # Cycle detected or naming failed -- use fallback
        ...
"""

import contextlib
import logging
import threading as _threading
from typing import Dict, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

_fragment_guard = _threading.local()

# Preserved for any external code that imports this constant.
# Not enforced internally — cycle detection via visited set is used.
MAX_NAMING_DEPTH = 7

# Safety-net maximum: even without exact cycle, limit recursion depth
# to prevent unbounded decomposition chains where every fragment SMILES
# is different. Generous limit (20 vs old limit of 7) to allow deep
# but finite naming chains.
_MAX_VISITED_SIZE = 50  # a phase: raised from 30 for deeper decomposition; fallback at limit
# NOTE : raising this to 200 was measured INERT for the acyl-CoA giant (its
# pantetheine-thioester substituent fails for a different reason, not this net) and is a
# global change with gate/perf risk, so it is NOT raised. Revisit as a measured breadth lever
# once the per-top-level work budget is proven a sufficient anti-runaway guard corpus-wide.

# giant-molecule hang fix: negative-cache sentinel. Stored in the runtime
# fragment cache to mark a fragment that is internal notes-INDEPENDENTLY unnameable (the
# recursive namer refused it / it produced a refusal sentinel / it raised) so the
# combinatorial partition search never re-descends the same dead fragment (e.g.
# the bare diphosphate ``COP(=O)(O)OP(=O)(O)O`` in an acyl-CoA). NOT used for the
# cycle or depth-net outcomes, which are context-DEPENDENT (they can succeed at a
# shallower depth), so caching those would cost breadth.
_NEG_CACHE = object()


def _get_visited() -> set:
    """Get the current visited-SMILES set (thread-safe).

    Returns an empty set if no naming session is active.
    """
    visited = getattr(_fragment_guard, 'visited', None)
    if visited is None:
        visited = set()
        _fragment_guard.visited = visited
    return visited


# Pre-computed names for common fragments that frequently hit the depth limit.
# Checked BEFORE the cycle guard so these fragments are always nameable,
# regardless of recursion state. Analogous to retained_names.py but for
# fragments produced during recursive decomposition of complex molecules.
#
# Every entry was verified against name_compound at depth 0 (2026-02-25).
# Only fragments with CORRECT verified names are included.
# fix a performance pass (wp7, 2026-09-26): ten entries had drifted to non-PIN names the
# top-level namer no longer gives ('2-hydroxyethanoic acid' made the carboxylate
# namer ship 'sodium 2-hydroxyethanoate' / 'disodium 2-oxidoethanoate' at
# pin_verified; 'cumene', 'glycerol', 'acetone',...). They now hold the PIN, each
# cited at its line; tests/unit/assembly/test_fragment_cache_pin_names.py keeps
# every entry equal to name_compound except the amino acids, which rest on
# user decision D-a (semisystematic amino-acid names are kept at the PIN tier).
FRAGMENT_NAME_CACHE: Dict[str, str] = {
    # --- Simple alkanes ---
    "CC": "ethane",
    "CCC": "propane",
    "CCCC": "butane",
    "CCCCC": "pentane",
    "CCCCCC": "hexane",
    "CCCCCCC": "heptane",
    "CCCCCCCC": "octane",
    "CCCCCCCCC": "nonane",
    "CCCCCCCCCC": "decane",
    "CCCCCCCCCCC": "undecane",
    "CCCCCCCCCCCC": "dodecane",
    # --- Simple alcohols ---
    "CO": "methanol",
    "CCO": "ethanol",
    "CCCO": "propan-1-ol",
    "CCCCO": "butan-1-ol",
    "CCCCCO": "pentan-1-ol",
    "CC(C)O": "propan-2-ol",
    "CC(C)(C)O": "2-methylpropan-2-ol",
    # --- Simple carboxylic acids ---
    "O=CO": "formic acid",
    "CC(=O)O": "acetic acid",
    "CCC(=O)O": "propanoic acid",
    "CCCC(=O)O": "butanoic acid",
    "CCCCC(=O)O": "pentanoic acid",
    "O=C(O)c1ccccc1": "benzoic acid",
    # --- Simple aldehydes ---
    "C=O": "formaldehyde",
    "CC=O": "acetaldehyde",
    "CCC=O": "propanal",
    "CCCC=O": "butanal",
    # --- Simple ketones ---
    "CC(C)=O": "propan-2-one",  # wp7: 'acetone propan-2-one (PIN)' the Blue Book
    "CCC(C)=O": "butan-2-one",
    # --- Simple amines ---
    # methylamine/ethylamine are general-nomenclature (non-PIN) functional-class
    # names; removed so fragments resolve to the substitutive PIN
    # (methanamine/ethanamine) via the systematic path. DD1 Fix 4 / H5.
    "CCCN": "propan-1-amine",
    "CCCCN": "butan-1-amine",
    "NCCCCCN": "pentane-1,5-diamine",
    # --- Common amides / nitriles ---
    "CC(N)=O": "acetamide",
    "CC#N": "acetonitrile",
    # --- Common aromatics ---
    "c1ccccc1": "benzene",
    "Oc1ccccc1": "phenol",
    "Nc1ccccc1": "aniline",
    "O=Cc1ccccc1": "benzaldehyde",
    "CC(=O)c1ccccc1": "1-phenylethan-1-one",  # wp7: '1-phenylethan-1-one (PIN) acetophenone':28369
    "c1ccc(-c2ccccc2)cc1": "1,1'-biphenyl",
    "c1ccc2ccccc2c1": "naphthalene",
    # --- Common heterocycles ---
    "c1ccncc1": "pyridine",
    "c1ccoc1": "furan",
    "c1cc[nH]c1": "1H-pyrrole",  # wp7: '1H-pyrrole (PIN)':24645
    "c1ccsc1": "thiophene",
    # --- Amino acids (retained names, common in peptide fragments) ---
    "NCC(=O)O": "glycine",
    "CC(N)C(=O)O": "alanine",
    "NC(CO)C(=O)O": "serine",
    "NC(CS)C(=O)O": "cysteine",
    "NC(Cc1ccc(O)cc1)C(=O)O": "tyrosine",
    "NC(Cc1c[nH]cn1)C(=O)O": "histidine",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O": "tryptophan",
    "NCCCCC(N)C(=O)O": "lysine",
    "NC(CCC(=O)O)C(=O)O": "glutamic acid",
    "NC(CC(=O)O)C(=O)O": "aspartic acid",
    "NC(=O)CC(N)C(=O)O": "asparagine",
    "CSCCC(N)C(=O)O": "methionine",
    # --- Other common fragments ---
    "CCCCC(CC)CO": "2-ethylhexan-1-ol",
    "ClCCCl": "1,2-dichloroethane",
    "ClC(Cl)Cl": "trichloromethane",  # wp7: 'chloroform' general only, PINs substitutive:25911
    # --- Fatty acids (common in phospholipids/sphingolipids) ---
    "CCCCCC(=O)O": "hexanoic acid",
    "CCCCCCC(=O)O": "heptanoic acid",
    "CCCCCCCC(=O)O": "octanoic acid",
    "CCCCCCCCC(=O)O": "nonanoic acid",
    "CCCCCCCCCC(=O)O": "decanoic acid",
    "CCCCCCCCCCCC(=O)O": "dodecanoic acid",
    "CCCCCCCCCCCCCC(=O)O": "tetradecanoic acid",
    "CCCCCCCCCCCCCCCC(=O)O": "hexadecanoic acid",
    "CCCCCCCCCCCCCCCCCC(=O)O": "octadecanoic acid",
    "CCCCCCCCCCCCCCCCCCCC(=O)O": "icosanoic acid",
    # --- Common biological fragments ---
    "OCC(O)CO": "propane-1,2,3-triol",  # wp7: 'glycerol propane-1,2,3-triol (PIN)':26776
    "OCCO": "ethane-1,2-diol",  # wp7: 'ethylene glycol ethane-1,2-diol (PIN)':26774
    "NCCO": "2-aminoethan-1-ol",  # wp7: '2-aminoethan-1-ol (PIN)':28156
    "O=P(O)(O)O": "phosphoric acid",
    # --- Unsaturated fatty acids (benchmark-driven, verified 2026-03-09) ---
    "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O": "(4Z,7Z,10Z,13Z,16Z,19Z)-docosa-4,7,10,13,16,19-hexaenoic acid",
    "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O": "(5Z,8Z,11Z,14Z,17Z)-icosa-5,8,11,14,17-pentaenoic acid",
    "CCCCCC/C=C\\CCCCCCCC(=O)O": "(9Z)-hexadec-9-enoic acid",
    "CCCC/C=C\\CCCCCCCC(=O)O": "(9Z)-tetradec-9-enoic acid",
    "CCCCC/C=C\\CCCCCCCC(=O)O": "(9Z)-pentadec-9-enoic acid",
    "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)O": "(8Z,11Z,14Z)-icosa-8,11,14-trienoic acid",
    "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)O": "(7Z,10Z,13Z,16Z)-docosa-7,10,13,16-tetraenoic acid",
    "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)O": "(11Z,14Z)-icosa-11,14-dienoic acid",
    # --- Branched small acids ---
    "CC(C)=CC(=O)O": "3-methylbut-2-enoic acid",
    "C/C=C(/C)C(=O)O": "(2Z)-2-methylbut-2-enoic acid",
    # --- Additional saturated fatty acids ---
    "CCCCCCCCCCCCCCC(=O)O": "pentadecanoic acid",
    "CCCCCCCCCCCCCCCCCCC(=O)O": "nonadecanoic acid",
    # --- Unsaturated fatty alcohols ---
    "CC/C=C\\CCO": "(3Z)-hex-3-en-1-ol",
    "CC/C=C/CCCCCO": "(6E)-non-6-en-1-ol",
    # --- Additional alcohols ---
    "CCCCCCCCCCCCO": "dodecan-1-ol",
    "CCCCCCO": "hexan-1-ol",
    # --- Branched fatty acids ---
    "CC(C)CCCCCCCCCCCC(=O)O": "13-methyltetradecanoic acid",
    # --- Branched alkanes (verified 2026-03-28, a phase-03) ---
    "CC(C)C": "2-methylpropane",
    "CCC(C)C": "2-methylbutane",
    "CC(C)(C)C": "2,2-dimethylpropane",
    "CCCC(C)C": "2-methylpentane",
    "CCC(C)CC": "3-methylpentane",
    # --- Cycloalkanes (verified 2026-03-28, a phase-03) ---
    "C1CC1": "cyclopropane",
    "C1CCC1": "cyclobutane",
    "C1CCCC1": "cyclopentane",
    "C1CCCCC1": "cyclohexane",
    "C1CCCCCC1": "cycloheptane",
    # --- Substituted aromatics (verified 2026-03-28, a phase-03) ---
    "Cc1ccccc1": "toluene",
    "CCc1ccccc1": "ethylbenzene",
    "CC(C)c1ccccc1": "(propan-2-yl)benzene",  # wp7: 'cumene... not retained':8093
    "COc1ccccc1": "anisole",
    "Clc1ccccc1": "chlorobenzene",
    "Fc1ccccc1": "fluorobenzene",
    "Brc1ccccc1": "bromobenzene",
    "O=[N+]([O-])c1ccccc1": "nitrobenzene",
    # --- Substituted heterocycles (verified 2026-03-28, a phase-03) ---
    "Cc1ccncc1": "4-methylpyridine",
    "Cc1ccccn1": "2-methylpyridine",
    "Cc1cccnc1": "3-methylpyridine",
    # --- Ethers and sulfides (verified 2026-03-28, a phase-03) ---
    "COC": "methoxymethane",
    "CCOCC": "ethoxyethane",
    # NB: no "CSC" entry. Per (the Blue Book) the PIN for a sulfide is the
    # substitutive "(methylsulfanyl)methane", NOT the functional-class "dimethyl
    # sulfide"; this cache holds PIN names (see the comment at the FRAGMENT_NAME_CACHE
    # lookup site), so the sulfide is emitted by the substitutive path, not here.
    # --- Dicarboxylic acids (verified 2026-03-28, a phase-03) ---
    "O=C(O)CO": "hydroxyacetic acid",  # wp7: 'hydroxyacetic acid (PIN)':29854
    "O=C(O)CC(=O)O": "propanedioic acid",
    "O=C(O)CCC(=O)O": "butanedioic acid",
    "O=C(O)C(=O)O": "oxalic acid",  # wp7: 'oxalic acid (PIN)':29723
    "O=C(O)CCCC(=O)O": "pentanedioic acid",
}


def _session_depth() -> int:
    return getattr(_fragment_guard, 'session_depth', 0)


def start_naming_session():
    """Initialize runtime fragment cache and visited set for a naming call.

    The runtime cache stores (canonical SMILES -> name) pairs discovered
    during a single top-level naming call. The visited set tracks which
    SMILES are currently being named to detect cycles.

    Only the outermost call starts a session; nested calls inherit the parent's
    cache and visited set.

    ⚠ **"Outermost" is an EXPLICIT COUNTER, not `len(visited) == 0`.** Inferring it
    from the visited set was a latent defect with five milestones of exposure: if an
    exception escaped a fragment naming without discarding its SMILES, `visited`
    stayed non-empty for the rest of the thread, so `is_top_level_naming` never
    returned True again and `namer.name` stopped publishing
    ``general_fallback_ctx`` / ``best_effort_ctx`` — **best-effort silently reverted
    to the PIN substituent vocabulary, with no error anywhere.** It surfaced as a
    test that passed alone and failed in a 17-file run under load, i.e. it needs a
    real interruption, which is what made it rare and long-lived.

    The counter cannot get stuck: it is decremented in ``end_naming_session``, which
    every caller invokes from a ``finally``, and it floors at 0.
    """
    depth = _session_depth() + 1
    _fragment_guard.session_depth = depth
    if depth == 1:
        # Only allocate a fresh cache when there is none. ``isolated_naming_session``
        # deliberately keeps the outer molecule's fragment cache alive across its
        # depth reset (giant-molecule hang fix): a fragment's name is a pure
        # function of its canonical SMILES, so reusing a cached name is always
        # correct AND stops a giant molecule from re-naming the same fragment on
        # every nested recovery entry. A GENUINE top-level call arrives with
        # ``cache is None`` (``end_naming_session`` cleared it), so it still starts
        # fresh; an isolation-induced depth-0 -> 1 keeps the live cache.
        if getattr(_fragment_guard, 'cache', None) is None:
            _fragment_guard.cache = {}
        _fragment_guard.visited = set()


def end_naming_session():
    """Close one naming call; the OUTERMOST one clears the session state.

    Nested calls must not clear their parent's cycle-detection state, which is why
    this is depth-guarded at all. But the outermost call now clears
    **unconditionally** rather than asking ``visited`` for permission — see
    ``start_naming_session`` for the defect that asking caused.
    """
    depth = _session_depth() - 1
    if depth <= 0:
        _fragment_guard.session_depth = 0
        _fragment_guard.visited = set()
        # giant-molecule hang fix: the fragment memo cache is owned by the
        # NAME SCOPE (enter/exit_name_scope, keyed to the true outermost name),
        # NOT by the session. Inside an ``isolated_naming_session`` the session
        # depth is reset to 0, so the FIRST nested ``name_compound`` would drive
        # this branch (session_depth 1 -> 0) and, if it cleared the cache, WIPE the
        # whole-molecule memo mid-naming -- the exact cause of the giant re-explore.
        # Only clear the cache when NOT inside a name scope (a direct
        # start/end_naming_session caller, preserving that legacy behaviour).
        if getattr(_fragment_guard, 'name_call_depth', 0) <= 0:
            _fragment_guard.cache = None
    else:
        _fragment_guard.session_depth = depth


# --- Per-top-level-call fragment work budget (giant-molecule hang fix) ---
#
# A hard ceiling on how many recursive fragment-naming ATTEMPTS one top-level
# ``name`` call may make. It guarantees that EVERY molecule terminates (name or
# abstain) instead of hanging when its decomposition re-explores fragments
# combinatorially -- the >60-heavy-atom best-effort HANG (acyl-CoA, peptide-glycan
# bioconjugates). Normal molecules make far fewer attempts than the budget, so
# their behaviour is byte-identical; only a runaway ever hits it and it then
# abstains fast (0-wrong is unaffected -- the molecule was going to abstain
# anyway, it just does so in bounded time).
#
# CRITICAL: the budget is armed at the TRUE outermost ``name`` entry and is NOT
# touched by ``start_naming_session`` or ``isolated_naming_session``. A previous
# attempt tied it to session state and ``isolated_naming_session`` (which resets
# ``session_depth`` to 0) re-initialised it on every nested recovery entry, so it
# never bounded anything. ``name_call_depth`` is a raw ``name`` call-stack
# counter, independent of the session/isolation machinery.
_WORK_BUDGET = 6000

# ── M2.5: per-top-level budgets against the macrocycle compute-bound HANG ────
# ``_WORK_BUDGET`` above bounds the NUMBER of fragment-naming attempts; it never
# sees what makes a fully-reduced symmetric metallo-macrocycle or a large cyclic
# glyco-/thio-peptide spin compute-bound for >20 min. Task 2B profiling (5 pinned
# witnesses vs the nameable porphyrin/chlorophyll controls, M2-5-HANG-LIVESPY /
# -TASK2A-REFUTED) found TWO ORTHOGONAL explosion modes, needing two budgets:
#
# 1. INNER-OP explosion (metallo-corrins): millions of iterations of the
# ``polycyclic`` von-Baeyer main-ring path DFS + the ``_best_disjoint_pair``
# O(paths^2) pairing loop, spread over only a MODEST number of analysis
# CALLS (a call-count budget cannot see it). Bounded by ``_PERF_BUDGET``.
# 2. CALL-COUNT explosion (cob(III)yrinate; vancomycin; the thiopeptide): the
# substituent recursion re-invokes the expensive analyses THOUSANDS of times
# (von-Baeyer ``analyze`` / the fused-heterocycle core matcher), each call
# individually cheap so the inner-op budget barely moves. Bounded by
# ``_ANALYSIS_CALL_BUDGET``.
#
# No size / aromatic / path-count prefilter separates the hangs from the nameable
# controls: the corrins are SMALLER and have FEWER paths per pair than the
# nameable Mg-chlorophyll (measured). The two cumulative budgets do — every
# nameable control (porphyrin analyze=2/match=4; steroid 5/15; chlorophyll
# inner=7.3M, analyze=19, match=128) sits comfortably below both thresholds,
# while every witness blows past one of them.
#
# ⚠ MEASURED LIMIT (inv 16): the Ni/Fe corrins do LESS total inner work in their
# first ~14 s than the nameable chlorophyll does in its ENTIRE run (both ~7M
# inner ops), so NO inner-op threshold that still names chlorophyll can abstain
# them in <10 s — the honest bound for that class is ~20-30 s. That is already a
# categorical fix (a >20-min compute-bound hang -> a bounded, DETERMINISTIC
# abstain); sub-10 s for that class would need a deeper root-cause change (dedup
# the substituent recursion's redundant re-analysis of the giant symmetric ring).
#
# DETERMINISTIC by construction (fixed operation/call counts, reset at the true
# outermost ``name`` — never wall-clock; SIGALRM is swallowed under the live
# OPSIN JVM). 0-wrong is unaffected: an exhausted budget raises
# ``PerfBudgetExceeded`` which unwinds to the outermost ``name`` and converts
# to the SAME clean abstain the engine already emits for an unnameable input
# (inv 9: a clean abstain, never a partial / atom-dropped / wrong name). A
# molecule that never approaches either budget is byte-identical.
import os as _os  # noqa: E402 (used for the budget-tuning env overrides below)

# Constants are env-overridable (tuning / an OFF switch, mirroring
# ORTHONYM_JVM_BUDGET): set to 0 to DISABLE that budget (never raises).
# Measured selectivity (M2.5 Task 2B, 200 drug-like pubchem rows): max inner-op
# 17,988 and max analysis-calls 138 — both hundreds/several-x below these — so
# the guard abstains ZERO nameable molecules; the only nameable molecule near a
# ceiling is Mg-chlorophyll (7.29M inner / 147 calls), which both budgets clear.
_PERF_BUDGET = int(_os.environ.get("ORTHONYM_PERF_OP_BUDGET", 12_000_000))          # inner-op ceiling (chlorophyll 7.29M)
_ANALYSIS_CALL_BUDGET = int(_os.environ.get("ORTHONYM_ANALYSIS_CALL_BUDGET", 500))  # analysis-call ceiling (chlorophyll 147)


class PerfBudgetExceeded(BaseException):
    """Raised when a per-top-level macrocycle-hang budget (``_PERF_BUDGET`` or
    ``_ANALYSIS_CALL_BUDGET``) is exhausted mid-analysis.

    Derives from ``BaseException`` (NOT ``Exception``) deliberately: the hot
    loops live many frames below dozens of broad ``except Exception:`` handlers
    on the recursive naming path, any one of which would otherwise swallow the
    signal and let the hang resume. As a ``BaseException`` it unwinds straight
    to the ``_budget_scope`` wrapper around the true-outermost ``name``, which
    is the ONLY site that catches it and turns it into a clean abstain. It must
    never escape that boundary. (No bare ``except:`` exists in the package, so
    nothing between the hot loop and that boundary intercepts it.)"""


# Measurement hook: when ORTHONYM_PERF_BUDGET_MEASURE is set in the env, the
# spend_* functions count into these process-globals (readable from a monitor
# thread — the per-molecule thread-locals are not) and NEVER raise, so the exact
# per-molecule counts can be sized empirically. Off in production: one boolean
# test per charge, negligible.
_PERF_MEASURE = bool(_os.environ.get("ORTHONYM_PERF_BUDGET_MEASURE"))
_PERF_SPENT_TOTAL = [0]
_ANALYSIS_SPENT_TOTAL = [0]


def spend_perf_work(n: int = 1) -> None:
    """Charge ``n`` INNER-OP units (polycyclic path DFS / pairing loop; fused
    per-candidate substructure attempt) against ``_PERF_BUDGET``.

    Raises ``PerfBudgetExceeded`` when exhausted. A no-op when no budget is armed
    (a direct producer call outside any ``name`` scope keeps its exact prior
    behaviour), and a pure counter when measurement mode is on.
    """
    if _PERF_MEASURE:
        _PERF_SPENT_TOTAL[0] += n
        return
    pb = getattr(_fragment_guard, 'perf_budget', None)
    if pb is None:
        return
    pb -= n
    if pb <= 0:
        _fragment_guard.perf_budget = 0
        _count_hang_budget_trip()
        raise PerfBudgetExceeded()
    _fragment_guard.perf_budget = pb


def spend_analysis_call(n: int = 1) -> None:
    """Charge ``n`` EXPENSIVE-ANALYSIS-CALL units (one von-Baeyer ``analyze`` /
    one fused-heterocycle core-match) against ``_ANALYSIS_CALL_BUDGET``.

    Same contract as ``spend_perf_work`` — raises ``PerfBudgetExceeded`` when
    exhausted, no-op outside a name scope, counter in measurement mode.
    """
    if _PERF_MEASURE:
        _ANALYSIS_SPENT_TOTAL[0] += n
        return
    ab = getattr(_fragment_guard, 'analysis_budget', None)
    if ab is None:
        return
    ab -= n
    if ab <= 0:
        _fragment_guard.analysis_budget = 0
        _count_hang_budget_trip()
        raise PerfBudgetExceeded()
    _fragment_guard.analysis_budget = ab


def disarm_hang_budgets() -> None:
    """Disable both macrocycle-hang budgets for the remainder of this name
    scope. Called at the outermost boundary ONCE the abstain decision is made,
    so the descriptive/coordination-fallback finishing work (which itself
    re-enters the fused matcher and von-Baeyer via ``_classify``) cannot
    re-trigger ``PerfBudgetExceeded`` and escape the boundary."""
    _fragment_guard.perf_budget = None
    _fragment_guard.analysis_budget = None


def rearm_hang_budgets() -> None:
    """Re-arm both macrocycle-hang budgets to a FRESH ceiling.

    Used by the outermost ``PerfBudgetExceeded`` boundary to BOUND a single
    last-resort whole-molecule rescue attempt made AFTER the main-path budget
    was exhausted (the ``_try_perf_budget_t4_rescue`` recovery). A fresh ceiling
    guarantees the rescue itself terminates: if the rescue's own analysis
    re-explodes it re-raises ``PerfBudgetExceeded`` (caught by the rescue -> clean
    abstain) rather than hanging. Mirrors the arming ``enter_name_scope`` does on
    the 0->1 transition; a constant of 0 (env OFF switch) leaves the budget
    unarmed (None) so the corresponding ``spend_*`` stays a permanent no-op."""
    _fragment_guard.perf_budget = _PERF_BUDGET if _PERF_BUDGET > 0 else None
    _fragment_guard.analysis_budget = (
        _ANALYSIS_CALL_BUDGET if _ANALYSIS_CALL_BUDGET > 0 else None)
    # The unproven-split floor is a level of THIS budget (``unproven_floor``): a fresh
    # ceiling has met no unproven split yet.
    _fragment_guard.unproven_floor = None


def enter_name_scope():
    """Arm the per-top-level fragment work budget AND memo cache at the outermost
    ``name``.

    Increments a raw ``name`` call-stack counter; on the 0 -> 1 transition (the
    TRUE outermost call) it (re)initialises the work budget and allocates the
    whole-molecule fragment memo cache. Nested ``name`` calls -- including the
    recursion re-entry through ``name_compound`` and the isolated producer --
    share both. Crucially, ``isolated_naming_session`` and the nested
    ``end_naming_session`` never reset these, so the memo survives the whole
    molecule (this is what stops a giant from re-exploring the same fragment
    thousands of times -- a per-molecule branch-cache model).
    """
    d = getattr(_fragment_guard, 'name_call_depth', 0) + 1
    _fragment_guard.name_call_depth = d
    if d == 1:
        _fragment_guard.work_budget = _WORK_BUDGET
        # M2.5: arm both macrocycle-hang budgets for the whole molecule alongside
        # the fragment budget; disarmed only by the matching outermost exit. A
        # constant of 0 (env OFF switch) leaves the budget unarmed (None) so the
        # corresponding spend_* is a permanent no-op.
        _fragment_guard.perf_budget = _PERF_BUDGET if _PERF_BUDGET > 0 else None
        _fragment_guard.analysis_budget = (
            _ANALYSIS_CALL_BUDGET if _ANALYSIS_CALL_BUDGET > 0 else None)
        _fragment_guard.unproven_floor = None
        # The name scope OWNS the fragment memo cache for the whole molecule.
        # Allocate a fresh one here so a leaked cache from a prior molecule can
        # never carry over (start_naming_session then keeps this live cache).
        _fragment_guard.cache = {}
        # The molecule's naming passes (``count_naming_pass``) start from 0.
        _fragment_guard.naming_passes = 0
        _fragment_guard.naming_pass_cap = None
    return d


class OptionalNamingCapExceeded(BaseException):
    """Raised by ``count_naming_pass`` when an optional re-naming of the molecule
    (``naming_pass_cap``) has run all the naming passes it was allowed. A
    ``BaseException`` like ``PerfBudgetExceeded``, so the ``except Exception``
    fallbacks inside the producers do not absorb it and it unwinds to the frame
    that armed the cap, which keeps the name it already has."""


def count_naming_pass() -> None:
    """Count one naming pass (one ``Orthonym._name_impl`` call) of the current
    outermost ``name`` (``enter_name_scope`` resets the count). Raises
    ``OptionalNamingCapExceeded`` when a cap armed by ``naming_pass_cap`` is
    passed. A no-op count outside any ``name`` scope."""
    if getattr(_fragment_guard, 'name_call_depth', 0) <= 0:
        return
    n = getattr(_fragment_guard, 'naming_passes', 0) + 1
    _fragment_guard.naming_passes = n
    cap = getattr(_fragment_guard, 'naming_pass_cap', None)
    if cap is not None and n > cap:
        raise OptionalNamingCapExceeded()


def naming_passes() -> int:
    """The naming passes (``count_naming_pass``) the current outermost ``name``
    has made so far; 0 outside any ``name`` scope."""
    return getattr(_fragment_guard, 'naming_passes', 0)


@contextlib.contextmanager
def naming_pass_cap(allowed: int):
    """Allow the body at most ``allowed`` more naming passes: the next pass past
    that raises ``OptionalNamingCapExceeded`` (caught by the caller). The count is
    deterministic -- the passes a naming makes depend on the molecule alone (the
    fragment memo starts empty for every outermost ``name``) -- so a capped
    body gives the same outcome in every process. Nested caps keep the tighter
    one; the previous cap is restored on exit."""
    prev = getattr(_fragment_guard, 'naming_pass_cap', None)
    cap = naming_passes() + max(0, int(allowed))
    _fragment_guard.naming_pass_cap = cap if prev is None else min(prev, cap)
    try:
        yield
    finally:
        _fragment_guard.naming_pass_cap = prev


def name_scope_depth() -> int:
    """The raw ``name`` call-stack depth (``enter_name_scope``): 1 inside the
    body of the TRUE outermost ``name``, the only frame that owns the hang
    budgets, 0 outside any ``name``."""
    return getattr(_fragment_guard, 'name_call_depth', 0)


def exit_name_scope():
    """Close one ``name`` scope; the outermost one disarms the budget and frees
    the whole-molecule memo cache."""
    d = getattr(_fragment_guard, 'name_call_depth', 1) - 1
    if d <= 0:
        _fragment_guard.name_call_depth = 0
        _fragment_guard.work_budget = None
        _fragment_guard.perf_budget = None
        _fragment_guard.analysis_budget = None
        _fragment_guard.unproven_floor = None
        _fragment_guard.cache = None
        _fragment_guard.naming_pass_cap = None
    else:
        _fragment_guard.name_call_depth = d


def spend_fragment_work() -> bool:
    """Charge one fragment-naming unit against the top-level budget.

    Returns True if work may proceed, False if the budget is exhausted (the
    caller must abstain). Returns True when no budget is armed -- e.g. a direct
    producer call outside any ``name`` scope -- so non-``name`` entry points
    keep their exact prior behaviour.
    """
    wb = getattr(_fragment_guard, 'work_budget', None)
    if wb is None:
        return True
    if wb <= 0:
        _count_hang_budget_trip()
        return False
    _fragment_guard.work_budget = wb - 1
    return True


def hang_budget_trips() -> int:
    """How many times a hang budget ran out on this thread (``spend_perf_work``,
    ``spend_analysis_call``, ``spend_fragment_work``). A counter that is never reset:
    a caller compares its value before and after a naming call (roadmap N5: the
    mechanical-spelling retry of ``Orthonym.name`` never names a molecule again whose
    first run hit a hang guard)."""
    return getattr(_fragment_guard, 'hang_budget_trips', 0)


def _count_hang_budget_trip() -> None:
    _fragment_guard.hang_budget_trips = hang_budget_trips() + 1


import contextlib as _contextlib

_OWNED_BUDGETS = ('perf_budget', 'analysis_budget', 'work_budget')


@_contextlib.contextmanager
def own_hang_budgets():
    """Run the body with hang budgets of its own: every budget that is armed
    (``perf_budget``, ``analysis_budget``, ``work_budget``) starts the body at its
    full ceiling, and the enclosing values are put back on exit, so the body
    neither spends nor sees the enclosing molecule's budgets. A disarmed budget
    stays disarmed; a no-op outside any ``name`` scope.

    For the components of a adduct: each is a compound of its own
    , the Blue Book, "Names are formed by citing the names of
    individual compounds"), named by a nested ``name`` that otherwise shares the
    whole assembly's budgets, so a mixture of four drug-size macrocycles ran out
    of the 500 analysis calls one compound gets (PubChem 1M: 296, 102, 65 and a
    fourth component that tripped the budget, abstaining the whole drawing). Still
    bounded -- each component by the ceilings of one compound -- and a trip inside
    the body raises ``PerfBudgetExceeded`` to the outermost ``name`` as before."""
    if getattr(_fragment_guard, 'name_call_depth', 0) <= 0:
        yield
        return
    saved = {attr: getattr(_fragment_guard, attr, None) for attr in _OWNED_BUDGETS}
    # The floor of the unproven-split search is a level of the analysis-call budget
    # (``unproven_floor``, engine._note_unproven_split): the body's fresh budget has met
    # no unproven split, and what the body notes is a level of ITS budget, which the
    # enclosing budget must not inherit.
    saved_floor = getattr(_fragment_guard, 'unproven_floor', None)
    fresh = {'perf_budget': _PERF_BUDGET if _PERF_BUDGET > 0 else None,
             'analysis_budget': (_ANALYSIS_CALL_BUDGET
                                 if _ANALYSIS_CALL_BUDGET > 0 else None),
             'work_budget': _WORK_BUDGET}
    for attr, value in saved.items():
        if value is not None:
            setattr(_fragment_guard, attr, fresh[attr])
    _fragment_guard.unproven_floor = None
    try:
        yield
    finally:
        for attr, value in saved.items():
            setattr(_fragment_guard, attr, value)
        _fragment_guard.unproven_floor = saved_floor


@_contextlib.contextmanager
def isolated_naming_session(reset_cache: bool = False):
    """Run a nested naming as if it were a fresh TOP-LEVEL call.

    Saves the current session state (``session_depth`` + cache + visited),
    resets to a clean depth-0 session, and restores it on exit. Used by
    ``namer``'s recovery-lane producer: ``name_t4_complete`` is conceptually a
    fresh whole-molecule naming, but the recovery lane invokes it mid-``name``
    with ``session_depth >= 1``, so its recursion consumes the shared
    ``MAX_NAMING_DEPTH`` budget from an elevated floor and a deep substituent hits
    the cap prematurely -- it then DEGRADES to an abstention where a standalone
    call names the molecule completely (measured: ``CC(=O)NCN(C)N=O`` and the
    in-scope suppressed cohort). Isolating the session gives the full depth-0
    budget, exactly as a direct call gets. Restores on exit so the enclosing
    session continues unperturbed. Never raises out of the restore.

    ``reset_cache`` (CQ5 Task 1, default False = the T4-producer behaviour below):
    ALSO save the whole-molecule fragment memo cache, install a fresh empty one for
    the isolated body, and restore the original on exit. The default keeps the
    cache LIVE (see the giant-hang note below); the opt-in is for the best-effort
    clean fall-through, which simulates a fresh TOP-LEVEL ``name`` and so needs
    the fresh cache a true top-level call gets from ``enter_name_scope``. WHY it
    matters: the memo caches ``(canonical SMILES -> name)``, and the comment below
    calls that "a context-free pure function" -- but a cached entry for a substituent
    the PRIMARY pass could not name (it hit ``MAX_NAMING_DEPTH`` from the elevated
    session floor and was memoized as a ``recursion_depth_fallback`` SKIP) is NOT
    context-free: the skip is a function of the depth budget at cache time, not of
    the SMILES alone. Resetting the session depth alone (below) gives the isolated
    body the full recursion budget, but with the poisoned skip still in the shared
    cache the deep substituent is read back as unnameable and the good name is never
    produced -- exactly the ``COP(=O)(C=C(F)F)C=C(F)F`` drop (RISK 4). A fresh cache
    lets the isolated body re-derive those fragments from the depth-0 budget.
    """
    saved_depth = getattr(_fragment_guard, 'session_depth', 0)
    saved_visited = getattr(_fragment_guard, 'visited', None)
    # giant-molecule hang fix: the fragment CACHE is deliberately NOT reset or
    # restored here (unless ``reset_cache``). It memoizes (canonical SMILES -> name),
    # a context-free pure function, so keeping it live across the isolation boundary
    # is always correct and prevents a >60-heavy-atom molecule from re-naming the
    # same fragment on every nested recovery entry (the best-effort HANG). Only the
    # depth budget (session_depth + visited) is reset -- the sole reason this
    # isolation exists: give the nested producer the full recursion budget a
    # standalone call gets. The inner ``start_naming_session`` keeps, not clobbers,
    # this cache (it only allocates one when ``cache is None``).
    #
    # ``reset_cache`` opts INTO cache isolation too: it saves the live cache and
    # installs a fresh ``{}`` for the isolated body, restoring the original on exit.
    # ``start_naming_session`` sees a non-None cache and keeps this fresh one. This
    # is bounded against the giant-hang by the per-``name`` work budget, which is
    # name-scope-owned and NOT reset here, so a giant's fall-through re-explore
    # shares the already-partly-spent budget and abstains fast rather than hanging.
    saved_cache = getattr(_fragment_guard, 'cache', None)
    _fragment_guard.session_depth = 0
    _fragment_guard.visited = set()
    if reset_cache:
        _fragment_guard.cache = {}
    try:
        yield
    finally:
        _fragment_guard.session_depth = saved_depth
        _fragment_guard.visited = saved_visited
        if reset_cache:
            _fragment_guard.cache = saved_cache


_SPECULATIVE_STATE = ('cache', 'session_depth', 'visited',
                      'work_budget', 'perf_budget', 'analysis_budget', 'unproven_floor')
_UNSET = object()


@_contextlib.contextmanager
def speculative_fragment_naming():
    """Run a nested naming whose ONLY product is its return value.

    For a caller that names a prefix merely to SORT by it (a (g) key built
    before the real assembly runs): every piece of per-molecule fragment state the
    nested call could leave behind is put back on exit -- the fragment memo cache
    (the body works on a copy, so it still reads what is cached but its writes are
    dropped; the cache is not context-free, see ``isolated_naming_session``), the
    depth counters and the three work budgets (they still bound the body, so a
    runaway still stops, but the spend is not charged to the real naming).
    """
    saved = {attr: getattr(_fragment_guard, attr, _UNSET) for attr in _SPECULATIVE_STATE}
    cache = saved['cache']
    if cache is not _UNSET and cache is not None:
        _fragment_guard.cache = dict(cache)
    visited = saved['visited']
    if visited is not _UNSET and visited is not None:
        _fragment_guard.visited = set(visited)
    try:
        yield
    finally:
        for attr, value in saved.items():
            if value is _UNSET:
                if hasattr(_fragment_guard, attr):
                    delattr(_fragment_guard, attr)
            else:
                setattr(_fragment_guard, attr, value)


def get_naming_depth() -> int:
    """Get current recursion depth proxy for fragment naming.

    Returns the size of the visited set, which represents how many
    fragments are currently being named up the call stack.

    Returns:
        Number of fragments currently being named (0 = top-level).
    """
    visited = getattr(_fragment_guard, 'visited', None)
    return len(visited) if visited else 0


def is_top_level_naming() -> bool:
    """Check if we're at the top level (not inside any recursive naming).

    Returns:
        True if no fragments are currently being named.
    """
    return get_naming_depth() == 0


def name_fragment_recursively(smiles: str, style: str = 'pin',
                              **_kwargs) -> Optional[str]:
    """Name a molecular fragment with cycle-detection guard.

    Uses a visited-SMILES set to detect and break circular recursion.
    Before naming a fragment, checks if its canonical SMILES is already
    being processed up the call stack. If yes (cycle detected), returns
    a cached name or None.

    A safety-net maximum visited set size (_MAX_VISITED_SIZE=20) prevents
    unbounded recursion from decomposition chains where every fragment
    SMILES is unique (different capping produces different SMILES).

    SMILES is canonicalized before processing to ensure consistent keys.

    Args:
        smiles: SMILES string of the fragment to name.

    Returns:
        IUPAC name if successful, None if cycle detected or naming fails.
    """
    # Canonicalize early so cache lookup uses consistent keys
    try:
        canonical = Chem.CanonSmiles(smiles)
    except Exception:
        return None
    if canonical is None:
        return None

    # Cache key is style-aware so a systematic name never poisons the pin cache
    # (or vice versa). For the default 'pin' style the key IS ``canonical`` and
    # every cache read/write below is byte-identical to the pre-style behaviour;
    # the static FRAGMENT_NAME_CACHE holds pin names, so a 'systematic' key never
    # hits it and the systematic path falls through to name_compound. (Slice C
    # slice-2: the amido converter needs the systematic acid name for amino-acid
    # rings — 'pyrrolidine-2-carboxylic acid', not the retained 'proline'.)
    cache_key = canonical if style == 'pin' else f"{canonical}\x00{style}"

    # Tier 1: static fragment cache — always available, cycle-independent
    cached = FRAGMENT_NAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Tier 2: runtime dynamic cache — populated during this naming session
    runtime_cache = getattr(_fragment_guard, 'cache', None)
    if runtime_cache is not None:
        dynamic = runtime_cache.get(cache_key)
        if dynamic is not None:
            return dynamic if dynamic is not _NEG_CACHE else None

    # giant-molecule hang fix: charge the per-top-level work budget for every
    # genuine naming ATTEMPT (a cache hit above is free and already returned). This
    # counts the cheap-but-unbounded outcomes too -- the depth-net and cycle
    # fallbacks below are exactly the hot loop a giant molecule spins in, so the
    # spend MUST precede them. When the budget is exhausted, abstain fast instead of
    # hanging. 0-wrong is unaffected: the molecule was going to abstain anyway; it
    # just terminates in bounded time. The catastrophe backstop; memoization
    # (positive + negative below) does the real bounding.
    if not spend_fragment_work():
        logger.warning(
            "WORK BUDGET exhausted: abstaining on fragment smiles=%s (giant-molecule guard)",
            smiles[:60],
        )
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='fragment_work_budget')
        return None

    # Cycle detection: if this SMILES is already being named up the call
    # stack, we have circular recursion. Return None (caller will fallback).
    visited = _get_visited()
    if canonical in visited:
        logger.info(
            "CYCLE detected: smiles=%s already in visited set (size=%d)",
            smiles[:60], len(visited),
        )
        # Task 0.1: fragment could not be named (cycle guard).
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='fragment_cycle')
        return None

    # Safety net: even without exact cycle, limit the recursion depth
    # to prevent unbounded decomposition chains where every fragment
    # SMILES is different (different capping) but naming never terminates.
    if len(visited) >= _MAX_VISITED_SIZE:
        logger.warning(
            "DEPTH safety net: visited set size=%d >= %d, smiles=%s -- trying pipeline fallback",
            len(visited), _MAX_VISITED_SIZE, smiles[:60],
        )
        from ..errors import is_refusal_sentinel
        from ..namer import name_pipeline_only
        # name_pipeline_only has no `style` parameter, so it can only honour the
        # default 'pin' request; for a 'systematic' request skip it (returning a
        # pin fallback would answer the wrong question) and fall through to the
        # abstention below. Cache under the style-aware key. (a review review NIT 9.)
        fallback_name = (name_pipeline_only(canonical) if style == 'pin'
                         else None)
        if fallback_name is not None and not is_refusal_sentinel(fallback_name):
            if runtime_cache is not None:
                runtime_cache[cache_key] = fallback_name
            return fallback_name
        # Task 0.1: fragment could not be named (depth safety net).
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='fragment_depth_limit')
        return None

    # Mark as in-progress, name it, then unmark
    visited.add(canonical)
    try:
        from ..errors import is_refusal_sentinel
        from ..namer import name_compound
        result = name_compound(canonical, style=style)
        # THE chokepoint where a WHOLE-MOLECULE naming becomes a NAME COMPONENT.
        # `name_compound` is always-emit: when it cannot name the input it returns
        # a refusal sentinel STRING ('zinc compound (not supported)',
        # 'unknown organic compound',...), not None. Returning that string to a
        # caller that only asks "is it non-empty?" is how a sentinel got welded
        # into a name -- CCS[Zn]SCC -> 'zinc compound (not supported)ylethane'.
        # Every one of this function's ~40 call sites consumes the result as a
        # component, so the test belongs HERE, once, not at each of them; the
        # shared predicate is the same one the slot filters use.
        if is_refusal_sentinel(result):
            from ..metrics.abstention import AbstentionCode, record_abstention
            record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                              detail='fragment_refusal_sentinel')
            #: this fragment is context-independently unnameable (the namer
            # refused it standalone). Negative-cache it so the partition search
            # never re-descends this dead end (giant-molecule hang fix).
            if runtime_cache is not None:
                runtime_cache[cache_key] = _NEG_CACHE
            return None
        if result:
            # Populate runtime cache with successful result
            if runtime_cache is not None:
                runtime_cache[cache_key] = result
            return result
        # Task 0.1: fragment could not be named (empty result).
        # NOT negative-cached: an empty result can be depth-INDUCED (name_compound's
        # own substituent recursion hit the depth net), so it may succeed at a
        # shallower depth -- caching it would cost breadth. Only the explicit
        # refusal-sentinel case above (a genuine structural refusal) is cached.
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='fragment_failed')
        return None
    except Exception as e:
        logger.debug(
            "Fragment naming exception: smiles=%s error=%s",
            smiles[:60], e,
        )
        # Task 0.1: fragment could not be named (exception).
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='fragment_exception')
        return None
    finally:
        visited.discard(canonical)
