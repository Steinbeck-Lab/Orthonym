"""Fragment naming infrastructure with cycle-detection guard.

Provides thread-safe cycle detection to prevent infinite loops when
fragment naming calls name_compound() recursively. Uses a visited-SMILES
set (threading.local() pattern) instead of an arbitrary depth counter.

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
_MAX_VISITED_SIZE = 50  # Phase 127: raised from 30 for deeper decomposition; fallback at limit
# NOTE (v33): raising this to 200 was measured INERT for the acyl-CoA giant (its
# pantetheine-thioester substituent fails for a different reason, not this net) and is a
# global change with gate/perf risk, so it is NOT raised. Revisit as a measured breadth lever
# once the per-top-level work budget is proven a sufficient anti-runaway guard corpus-wide.

# v33 giant-molecule hang fix: negative-cache sentinel. Stored in the runtime
# fragment cache to mark a fragment that is CONTEXT-INDEPENDENTLY unnameable (the
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
# Every entry was verified against name_compound() at depth 0 (2026-02-25).
# Only fragments with CORRECT verified names are included.
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
    "CC(C)=O": "acetone",
    "CCC(C)=O": "butan-2-one",
    # --- Simple amines ---
    # methylamine/ethylamine are general-nomenclature (non-PIN) functional-class
    # names (P-62.2.1.2); removed so fragments resolve to the substitutive PIN
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
    "CC(=O)c1ccccc1": "acetophenone",
    "c1ccc(-c2ccccc2)cc1": "1,1'-biphenyl",
    "c1ccc2ccccc2c1": "naphthalene",
    # --- Common heterocycles ---
    "c1ccncc1": "pyridine",
    "c1ccoc1": "furan",
    "c1cc[nH]c1": "pyrrole",
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
    "ClC(Cl)Cl": "chloroform",
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
    "OCC(O)CO": "glycerol",
    "OCCO": "ethylene glycol",
    "NCCO": "2-aminoethanol",
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
    # --- Branched alkanes (verified 2026-03-28, Phase 125-03) ---
    "CC(C)C": "2-methylpropane",
    "CCC(C)C": "2-methylbutane",
    "CC(C)(C)C": "2,2-dimethylpropane",
    "CCCC(C)C": "2-methylpentane",
    "CCC(C)CC": "3-methylpentane",
    # --- Cycloalkanes (verified 2026-03-28, Phase 125-03) ---
    "C1CC1": "cyclopropane",
    "C1CCC1": "cyclobutane",
    "C1CCCC1": "cyclopentane",
    "C1CCCCC1": "cyclohexane",
    "C1CCCCCC1": "cycloheptane",
    # --- Substituted aromatics (verified 2026-03-28, Phase 125-03) ---
    "Cc1ccccc1": "toluene",
    "CCc1ccccc1": "ethylbenzene",
    "CC(C)c1ccccc1": "cumene",
    "COc1ccccc1": "anisole",
    "Clc1ccccc1": "chlorobenzene",
    "Fc1ccccc1": "fluorobenzene",
    "Brc1ccccc1": "bromobenzene",
    "O=[N+]([O-])c1ccccc1": "nitrobenzene",
    # --- Substituted heterocycles (verified 2026-03-28, Phase 125-03) ---
    "Cc1ccncc1": "4-methylpyridine",
    "Cc1ccccn1": "2-methylpyridine",
    "Cc1cccnc1": "3-methylpyridine",
    # --- Ethers and sulfides (verified 2026-03-28, Phase 125-03) ---
    "COC": "methoxymethane",
    "CCOCC": "ethoxyethane",
    "CSC": "dimethyl sulfide",
    # --- Dicarboxylic acids (verified 2026-03-28, Phase 125-03) ---
    "O=C(O)CO": "2-hydroxyethanoic acid",
    "O=C(O)CC(=O)O": "propanedioic acid",
    "O=C(O)CCC(=O)O": "butanedioic acid",
    "O=C(O)C(=O)O": "ethanedioic acid",
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
    stayed non-empty for the rest of the thread, so `is_top_level_naming()` never
    returned True again and `namer.name()` stopped publishing
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
        # depth reset (v33 giant-molecule hang fix): a fragment's name is a pure
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
        # v33 giant-molecule hang fix: the fragment memo cache is owned by the
        # NAME SCOPE (enter/exit_name_scope, keyed to the true outermost name()),
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


# --- Per-top-level-call fragment work budget (v33 giant-molecule hang fix) ---
#
# A hard ceiling on how many recursive fragment-naming ATTEMPTS one top-level
# ``name()`` call may make. It guarantees that EVERY molecule terminates (name or
# abstain) instead of hanging when its decomposition re-explores fragments
# combinatorially -- the >60-heavy-atom best-effort HANG (acyl-CoA, peptide-glycan
# bioconjugates). Normal molecules make far fewer attempts than the budget, so
# their behaviour is byte-identical; only a runaway ever hits it and it then
# abstains fast (0-wrong is unaffected -- the molecule was going to abstain
# anyway, it just does so in bounded time).
#
# CRITICAL: the budget is armed at the TRUE outermost ``name()`` entry and is NOT
# touched by ``start_naming_session`` or ``isolated_naming_session``. A previous
# attempt tied it to session state and ``isolated_naming_session`` (which resets
# ``session_depth`` to 0) re-initialised it on every nested recovery entry, so it
# never bounded anything. ``name_call_depth`` is a raw ``name()`` call-stack
# counter, independent of the session/isolation machinery.
_WORK_BUDGET = 6000


def enter_name_scope():
    """Arm the per-top-level fragment work budget AND memo cache at the outermost
    ``name()``.

    Increments a raw ``name()`` call-stack counter; on the 0 -> 1 transition (the
    TRUE outermost call) it (re)initialises the work budget and allocates the
    whole-molecule fragment memo cache. Nested ``name()`` calls -- including the
    recursion re-entry through ``name_compound`` and the isolated T4 producer --
    share both. Crucially, ``isolated_naming_session`` and the nested
    ``end_naming_session`` never reset these, so the memo survives the whole
    molecule (this is what stops a giant from re-exploring the same fragment
    thousands of times -- per-molecule branch-cache model).
    """
    d = getattr(_fragment_guard, 'name_call_depth', 0) + 1
    _fragment_guard.name_call_depth = d
    if d == 1:
        _fragment_guard.work_budget = _WORK_BUDGET
        # The name scope OWNS the fragment memo cache for the whole molecule.
        # Allocate a fresh one here so a leaked cache from a prior molecule can
        # never carry over (start_naming_session then keeps this live cache).
        _fragment_guard.cache = {}


def exit_name_scope():
    """Close one ``name()`` scope; the outermost one disarms the budget and frees
    the whole-molecule memo cache."""
    d = getattr(_fragment_guard, 'name_call_depth', 1) - 1
    if d <= 0:
        _fragment_guard.name_call_depth = 0
        _fragment_guard.work_budget = None
        _fragment_guard.cache = None
    else:
        _fragment_guard.name_call_depth = d


def spend_fragment_work() -> bool:
    """Charge one fragment-naming unit against the top-level budget.

    Returns True if work may proceed, False if the budget is exhausted (the
    caller must abstain). Returns True when no budget is armed -- e.g. a direct
    producer call outside any ``name()`` scope -- so non-``name()`` entry points
    keep their exact prior behaviour.
    """
    wb = getattr(_fragment_guard, 'work_budget', None)
    if wb is None:
        return True
    if wb <= 0:
        return False
    _fragment_guard.work_budget = wb - 1
    return True


import contextlib as _contextlib


@_contextlib.contextmanager
def isolated_naming_session():
    """Run a nested naming as if it were a fresh TOP-LEVEL call.

    Saves the current session state (``session_depth`` + cache + visited),
    resets to a clean depth-0 session, and restores it on exit. Used by
    ``namer``'s recovery-lane T4 producer: ``name_t4_complete`` is conceptually a
    fresh whole-molecule naming, but the recovery lane invokes it mid-``name()``
    with ``session_depth >= 1``, so its recursion consumes the shared
    ``MAX_NAMING_DEPTH`` budget from an elevated floor and a deep substituent hits
    the cap prematurely -- it then DEGRADES to an abstention where a standalone
    call names the molecule completely (measured: ``CC(=O)NCN(C)N=O`` and the
    in-scope suppressed cohort). Isolating the session gives T4 the full depth-0
    budget, exactly as a direct call gets. Restores on exit so the enclosing
    session continues unperturbed. Never raises out of the restore.
    """
    saved_depth = getattr(_fragment_guard, 'session_depth', 0)
    saved_visited = getattr(_fragment_guard, 'visited', None)
    # v33 giant-molecule hang fix: the fragment CACHE is deliberately NOT reset or
    # restored here. It memoizes (canonical SMILES -> name), a context-free pure
    # function, so keeping it live across the isolation boundary is always correct
    # and prevents a >60-heavy-atom molecule from re-naming the same fragment on
    # every nested recovery entry (the best-effort HANG). Only the depth budget
    # (session_depth + visited) is reset -- the sole reason this isolation exists:
    # give the nested T4 producer the full recursion budget a standalone call gets.
    # The inner ``start_naming_session`` keeps, not clobbers, this cache (it only
    # allocates one when ``cache is None``).
    _fragment_guard.session_depth = 0
    _fragment_guard.visited = set()
    try:
        yield
    finally:
        _fragment_guard.session_depth = saved_depth
        _fragment_guard.visited = saved_visited


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
    # hits it and the systematic path falls through to name_compound. (v30 Slice C
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

    # v33 giant-molecule hang fix: charge the per-top-level work budget for every
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
        # v25 P0 Task 0.1: fragment could not be named (cycle guard).
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
        # abstention below. Cache under the style-aware key. (Fable review NIT 9.)
        fallback_name = (name_pipeline_only(canonical) if style == 'pin'
                         else None)
        if fallback_name is not None and not is_refusal_sentinel(fallback_name):
            if runtime_cache is not None:
                runtime_cache[cache_key] = fallback_name
            return fallback_name
        # v25 P0 Task 0.1: fragment could not be named (depth safety net).
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
        # 'unknown organic compound', ...), not None. Returning that string to a
        # caller that only asks "is it non-empty?" is how a sentinel got welded
        # into a name -- CCS[Zn]SCC -> 'zinc compound (not supported)ylethane'.
        # Every one of this function's ~40 call sites consumes the result as a
        # component, so the test belongs HERE, once, not at each of them; the
        # shared predicate is the same one the slot filters use.
        if is_refusal_sentinel(result):
            from ..metrics.abstention import AbstentionCode, record_abstention
            record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                              detail='fragment_refusal_sentinel')
            # v33: this fragment is context-independently unnameable (the namer
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
        # v25 P0 Task 0.1: fragment could not be named (empty result).
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
        # v25 P0 Task 0.1: fragment could not be named (exception).
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='fragment_exception')
        return None
    finally:
        visited.discard(canonical)
