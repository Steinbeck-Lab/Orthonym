"""
Decomposition engine with quality gating and orchestration.

The engine decides WHETHER to decompose a molecule (quality gate),
calls bond_cleavage and fragment_capping to split it, names each
fragment recursively, and assembles the final multi-component name.

Quality gate prevents regressions: molecules that the existing pipeline
names correctly are left alone. Only molecules with poor names (unknown,
suspiciously short, etc.) are decomposed.
"""

import contextvars
import re
from collections import deque
from typing import Dict, List, Optional

from rdkit import Chem

# ---------------------------------------------------------------------------
# Performance guard
# ---------------------------------------------------------------------------

MAX_CLEAVABLE_BONDS = 20  # a phase: raised from 12, with structural complexity check
MAX_BOND_RETRY_ATTEMPTS = 5  # Max bonds to try when multi-bond retry is active
MAX_DECOMP_LEVELS = 3  # a phase: max iterative decomposition levels for mixed bond types


# ---------------------------------------------------------------------------
# Fragment naming with fallback (a phase)
# ---------------------------------------------------------------------------


def _name_fragment_with_fallback(smiles: str):
    """Name a fragment: recursive naming first, pipeline-only fallback, then
    a T4/general-engine best-effort rescue (a phase).

    Per: When name_fragment_recursively returns None (depth/cycle limit hit),
    fall back to name_pipeline_only instead of aborting the entire decomposition.
    name_pipeline_only uses the full IUPAC pipeline without triggering decomposition
    recursion, preserving all substituents.

     a phase (internal notes): when BOTH
    PIN-tier attempts above fail, retry the fragment through a fresh best-effort
    (T4) namer before giving up (`_name_fragment_t4_rescue`). This rung is
    strictly SCOPED to run only after a PIN failure -- a fragment that already
    names at PIN tier returns from one of the first two rungs and the rescue
    never executes, so every currently-correct decomposition name stays
    byte-identical. The rescue exists so the calling multi-fragment assemblers
    (`_try_multi_bond_decompose`, `_try_iterative_mixed_decompose`) can try
    harder before counting a fragment as failed, so their new fail-closed
    guard doesn't abstain on a fragment that best-effort could have named.

    Returns:
        IUPAC name string, or None if all three attempts fail.
    """
    from ..assembly.fragment_naming import name_fragment_recursively
    from ..namer import name_pipeline_only

    # Primary: recursive naming (may trigger sub-decomposition)
    name = name_fragment_recursively(smiles)
    if name and "unknown" not in name.lower():
        return name

    # Fallback: full systematic pipeline without decomposition
    name = name_pipeline_only(smiles)
    if name and "unknown" not in name.lower():
        return name

    # Last resort (a phase): best-effort T4/general-engine rescue --
    # only reached when both PIN-tier attempts above already failed.
    name = _name_fragment_t4_rescue(smiles)
    if name and "unknown" not in name.lower():
        return name

    return None


def _name_fragment_t4_rescue(smiles: str) -> Optional[str]:
    """Best-effort (T4/general-engine) rescue for a fragment PIN could not name.

     a phase (internal notes, secondary
    lever): every fragment named during decomposition previously went through
    ``name_pipeline_only``, whose fresh ``Orthonym`` defaults
    ``general_fallback=False`` (``namer.py:2047``) -- so the T4/general-engine
    best-effort assist never fired for a fragment, even when the enclosing
    molecule was itself being named at a best-effort tier. A fragment
    unnameable at PIN tier but completable by (an unusual branched acyl
    chain, a decorated sugar ring) spuriously inflated ``failed_count`` in the
    multi-fragment assemblers, feeding the partial-ship defect this phase
    fixes.

    Calls ``Orthonym._try_general_engine_recovery`` DIRECTLY (not a fresh
    ``.name``/``_name_impl`` pass) inside ``isolated_naming_session`` --
    the SAME mechanism ``namer.py``'s own recovery lane uses (see
    ``fragment_naming.isolated_naming_session``'s own docstring) to give the
    call the full depth-0 recursion budget AND to satisfy
    ``is_top_level_naming``, which gates that recovery lane -- without the
    isolated session this fragment is nested (non-zero fragment-naming depth),
    so the lane declines unconditionally regardless of these flags.

    Calling ``_try_general_engine_recovery`` directly (per its own docstring:
    "Re-perceives, runs the general engine, and re-applies the SAME E1 +
     gates to the emission. Returns a verified name or None") is
    deliberate rather than re-running the fragment through a fresh
    ``Orthonym.name``: that method is ALSO the entry point for PIN naming
    AND decomposition, so a plain ``.name`` call would redundantly redo the
    PIN attempt this function's callers already made, and -- because
    decomposition is not skipped on that path -- risks re-decomposing the
    fragment into its own sub-fragments, each of which could again fail and
    recurse into this same rescue. Measured: this caused multi-minute
    slowdowns on real multi-residue fragments during phase development.
    ``_try_general_engine_recovery`` is self-contained (own E1 +
    gates, no decomposition reentry), so this rescue is a single bounded
    best-effort attempt, not a recursive one.

    Every emission still passes through the SAME E1 atom-coverage
    certificate and OPSIN round-trip gates as any other name
    (the shared ladder in ``namer.py``) -- this rescue adds no new bypass of
    either, so it can turn a fragment failure into a best-effort success but
    can never ship a name the existing 0-wrong net would otherwise reject.

    Scoping (PIN byte-identity): this is called ONLY as the third/last rung of
    ``_name_fragment_with_fallback``, after both PIN-tier attempts already
    failed. A fragment that names successfully at PIN tier never reaches this
    function, so this rescue can only ever convert a fragment FAILURE into a
    best-effort NAME -- it never replaces or competes with a PIN name.

    Returns:
        IUPAC name string (best-effort tier), or None (clean abstain -- never
        a partial).

    Memoised for the top-level naming call, None included (``assembly.memo``,
    scope-bound; ``ORTHONYM_MEMO=verify`` recomputes and compares). The
    decomposition asks for the same fragment from every cut context, and each ask
    re-ran the whole rescue (perception, classification, engine, round trips): the
    ChEBI lipid II-type glycopeptide made 816 rescues of 81 fragments, and the 735
    repeats took 35 of its 91 s. Measured on those repeats (fresh process): every
    one returned the name of the first ask and spent no unit of any naming budget.
    A hit replays the provenance variables the cold call wrote (its ``source``
    record above all): a caller that restored its provenance between two asks
    (the isotope decorator names its skeleton twice) otherwise shipped the
    general-engine skeleton under the PIN path's label with the memo on and
    systematic_verified with it off ('2-methylpropan-2-yl 2-{6-[2-(4-{2-[(2H5)
    cyclohexa-1,3,5-trien-1-yl]-1-oxo-2-azaethyl}-...)ethanoate'). The key carries the inputs the
    rescue reads besides the fragment: the ambient breadth context, the N-acyl
    float scope, the peptide re-entrancy flag and the locant/isotope scopes.

    A hit charges the perf and analysis units the fresh call charged (not its work
    units). Those two budgets are charged per ring analysis (von Baeyer ``analyze``,
    the fused core matcher, the polycyclic path search, whose own memo replays its
    units on a hit), so an unmemoised repeat charges them again: measured on the
    ChEBI acetylated oligosaccharide (fresh process, no memo), 52 of 53 repeats
    charged exactly the first ask's perf and analysis units (the 53rd is the call
    in which the analysis budget ran out), and the budget ran out at 33.5 s, where
    the whole-molecule rescue of the budget boundary named it; with hits
    charging nothing it ran 409 s to the same name. The work budget is charged only
    on a fragment-cache miss, which a warm repeat does not make: the glycopeptide's
    first asks charged 10 work units and its 735 repeats none, and 0 perf and 0
    analysis units either way, so it is unchanged.
    """
    from ..assembly import fragment_naming as _fn
    from ..assembly.memo import cache_or_compute, side_get, side_put
    from ..assembly.locant_omission import (_ISOTOPE_PARENT_POSITIONAL,
                                            _ISOTOPIC_NAMING_SCOPE,
                                            forced_locant_reason)
    from ..metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx,
                                      full_coverage_ctx, general_fallback_ctx)
    from ..routing import dispatch_table as _dispatch_table
    key = (smiles, general_fallback_ctx.get(), best_effort_ctx.get(),
           allow_aromatic_general_ctx.get(), full_coverage_ctx.get(),
           nacyl_float_refusing(), _dispatch_table._PEPTIDE_SUBST_ACTIVE,
           forced_locant_reason(), _ISOTOPIC_NAMING_SCOPE.get(),
           _ISOTOPE_PARENT_POSITIONAL.get())
    from ..assembly.nested_memo import _apply as _replay_provenance
    from ..metrics import provenance as _pv
    state = {"hit": True, "units": (0, 0), "replay": {}}

    def _budgets():
        g = _fn._fragment_guard
        return (getattr(g, "perf_budget", None), getattr(g, "analysis_budget", None))

    def _fresh():
        state["hit"] = False
        before = _budgets()
        _pv.push_touched_log()
        try:
            result = _name_fragment_t4_rescue_fresh(smiles)
        finally:
            touched = _pv.pop_touched_log()
        after_prov = _pv.get_provenance()
        state["replay"] = {k: after_prov[k] for k in touched if k in after_prov}
        state["units"] = tuple(
            (b - a) if (a is not None and b is not None and b > a) else 0
            for b, a in zip(before, _budgets()))
        return result

    result = cache_or_compute("t4_fragment_rescue", key, _fresh)
    if state["hit"]:
        meta = side_get("t4_fragment_rescue", key)
        if meta:
            units, replay = meta
            # Raises PerfBudgetExceeded exactly when the repeat would have: the
            # repeat charges these units one analysis at a time, and a spend
            # raises once the running total reaches the remaining budget.
            if units[0]:
                _fn.spend_perf_work(units[0])
            if units[1]:
                _fn.spend_analysis_call(units[1])
            if replay:
                _replay_provenance(replay, (0, 0, 0), replay_budgets=False)
    else:
        side_put("t4_fragment_rescue", key, (state["units"], state["replay"]))
    return result


def _name_fragment_t4_rescue_fresh(smiles: str) -> Optional[str]:
    from ..assembly.fragment_naming import isolated_naming_session
    from ..namer import Orthonym

    try:
        with isolated_naming_session():
            namer = Orthonym(
                style="pin",
                general_fallback=True,
                general_fallback_unverified=True,
                allow_aromatic_general=True,
            )
            return namer._try_general_engine_recovery(smiles)
    except Exception:
        return None


def _get_max_decomp_levels(mol) -> int:
    """Return max decomposition levels based on molecule size.

    Per: HA > 50 molecules (phospholipids, polysaccharides) need one
    extra level to fully decompose mixed bond types.
    """
    if mol.GetNumHeavyAtoms() > 50:
        return 4
    return MAX_DECOMP_LEVELS  # Default: 3


# ---------------------------------------------------------------------------
# Bond type classification for tiered coverage thresholds
# ---------------------------------------------------------------------------

# Functional-class bond types produce compact names (e.g., "phenyl palmitate")
# and use a lower chars/HA coverage threshold (0.6).
# Substitutive bond types (sulfonamide, phosphodiester, ether, default) produce
# longer names and require a higher threshold (0.8).
_FUNCTIONAL_CLASS_TYPES = frozenset({"ester", "amide", "glycosidic", "carbamate", "thioester"})


# ---------------------------------------------------------------------------
# Bond-type-specific thresholds for multi-bond decomposition (a phase-04)
# ---------------------------------------------------------------------------

# Glycosidic bonds: threshold 2 (disaccharide + aglycone benefits from multi-bond).
# Ester/amide: threshold 3 (2-bond molecules better handled by single-bond).
# a phase-02 confirmed count>=2 causes regressions on 2-ester phospholipids.
_MULTI_BOND_THRESHOLD = {
    "ester": 2,       # Lowered from 3 per /: enables diester decomposition
    "glycosidic": 2,
    "amide": 3,
}


# ---------------------------------------------------------------------------
# Retained-name whitelist for quality gate
# ---------------------------------------------------------------------------

# Known retained names that correctly identify a core substructure even
# in large molecules (nucleotide cofactors, natural products, etc.).
# These bypass the "no digits and no hyphens" rejection for heavy_atoms > 25.
# Expanded in a phase with fused heterocycle names.
_RETAINED_CORE_NAMES = frozenset({
    'adenine', 'guanine', 'thymine', 'cytosine', 'uracil',
    'xanthine', 'hypoxanthine', 'purine', 'pyrimidine',
    'indole', 'quinoline', 'isoquinoline', 'acridine',
    'phenothiazine', 'xanthene', 'phenoxazine', 'thianthrene',
    '1h-indole',
    # a phase additions: fused heterocycles and polycyclics
    'flavone', 'chromone', 'coumarin', 'pteridine', 'phenazine',
    'carbazole', 'phenanthridine',
    #: the PIN forms carry the [b,d] fusion descriptor; the exact
    # match at:377/:425 needs them so a >25-heavy-atom core still passes the
    # quality gate. Bare forms kept (harmless) for any legacy path.
    'dibenzo[b,d]furan', 'dibenzo[b,d]thiophene',
    'dibenzofuran', 'dibenzothiophene',
    'fluorene', 'fluorenone',
    'anthracene', 'phenanthrene', 'chrysene',
})


# ---------------------------------------------------------------------------
# Ring-system token detection for quality gate
# ---------------------------------------------------------------------------

# Recognized ring-system name tokens. Names containing any of these tokens
# are considered structurally informative even without digits/hyphens.
# Used to bypass the no-digits/no-hyphens rejection in _name_quality_is_acceptable.
_RING_SYSTEM_TOKENS = frozenset({
    'pyridine', 'pyrimidine', 'pyrazine', 'pyridazine',
    'benzene', 'toluene', 'naphthalene', 'anthracene', 'phenanthrene',
    'morpholine', 'piperidine', 'piperazine', 'pyrrolidine',
    'indole', 'quinoline', 'isoquinoline', 'quinoxaline', 'quinazoline',
    'thiophene', 'furan', 'pyrrole', 'oxazole', 'thiazole', 'isoxazole',
    'imidazole', 'triazole', 'tetrazole',
    'carbazole', 'acridine', 'phenothiazine', 'phenoxazine',
    'flavone', 'chromone', 'coumarin', 'xanthene',
    'purine', 'pteridine', 'phenazine',
    'dibenzofuran', 'dibenzothiophene',
})


# ---------------------------------------------------------------------------
# Bond-type token matching for quality gate (a phase -)
# ---------------------------------------------------------------------------

# Token mapping: what name tokens indicate each bond type.
# Amide tokens include acyl prefixes (anoyl, enoyl, oyl)
# since N-acyl naming IS amide naming (IUPAC.
# Moved to module level from _name_quality_is_acceptable for testability.
_BOND_TYPE_TOKENS = {
    "ester": {"ester", "oate", "ate", "oyloxy",
              "acetyloxy", "benzoyloxy", "acetyl",
              "benzoyl"},
    "amide": {"amide", "amino", "amido", "acetamid",
              "formamid", "carbamoyl", "anilino",
              "anoyl", "enoyl", "oyl", "acyl"},
    "glycosidic": {"glycos", "pyranosyl", "furanosyl",
                   "glucos", "galactos", "mannos", "rhamn",
                   "fucos", "sugar", "osyl"},
    "phosphodiester": {"phosph", "nucleotid"},
    "thioester": {"thio"},
    "sulfonamide": {"sulfonamid", "sulfamid"},
    "carbamate": {"carbamat", "urethane"},
    # Ethers are common and don't always produce distinct
    # name tokens (ether O becomes "oxa" or is absorbed
    # into alkoxy prefixes). Don't penalize.
    "ether": set(),
    # Thioethers produce "thio" prefix in substitutive names
    "thioether": {"thio", "sulfanyl"},
    # Secondary amines produce "amino" prefix
    "sec_amine": {"amino"},
}


def _compile_token_patterns(token_dict):
    """Build regex patterns for IUPAC morpheme-aware token matching.

    Per: tokens match at IUPAC nomenclature boundaries (after hyphen,
    after opening paren, at start of name, before closing paren, at end of name).
    Per: short suffix tokens ('ate', 'oyl') match at word/morpheme end only,
    to prevent false positives from common English words.
    Per: known false positive patterns ('polyester', 'polyamide', etc.) are
    excluded via a separate false-positive check BEFORE regex matching.

    Token matching strategy:
    - Short suffix tokens ('ate', 'oyl'): match at morpheme end only
    - Longer IUPAC morphemes (>= 4 chars): substring match is safe because
      these morphemes are specific enough. False positives from polymer names
      are handled by the _FALSE_POSITIVES exclusion list.
    """
    _SUFFIX_ONLY_TOKENS = {"ate", "oyl"}
    _FALSE_POSITIVES = {"polyester", "polyamide", "polyurethane", "polycarbonate"}

    compiled = {}
    for bond_type, tokens in token_dict.items():
        patterns = []
        for tok in tokens:
            escaped = re.escape(tok)
            if tok in _SUFFIX_ONLY_TOKENS:
                # Short suffix: match at morpheme end only to avoid
                # "calculate" matching "ate", "royal" matching "oyl"
                patterns.append(re.compile(
                    rf'{escaped}(?:$|[-)\s,])', re.IGNORECASE
                ))
            else:
                # Longer IUPAC morphemes: substring match is safe.
                # False positives from polymer names handled by exclusion list.
                patterns.append(re.compile(
                    rf'{escaped}', re.IGNORECASE
                ))
        compiled[bond_type] = patterns
    compiled["_false_positives"] = _FALSE_POSITIVES
    return compiled


# Pre-compiled patterns for use by quality gate
_COMPILED_TOKEN_PATTERNS = _compile_token_patterns(_BOND_TYPE_TOKENS)


def _token_matches_name(name_lower, token_patterns, bond_type):
    """Check if any token for this bond type matches in the name.

    Per: first checks for false positive patterns.
    Per: never raises -- returns True on any exception (benefit of doubt).
    """
    try:
        false_pos = token_patterns.get("_false_positives", set())
        for fp in false_pos:
            if fp in name_lower:
                return False

        patterns = token_patterns.get(bond_type, [])
        if not patterns:
            return True  # Empty token set (e.g., ether): benefit of doubt
        return any(p.search(name_lower) for p in patterns)
    except Exception:
        return True  # Guard: never crash the quality gate



def _name_has_ring_system_token(name: str) -> bool:
    """Check if a name contains a recognized ring-system token.

    Used by the quality gate to distinguish legitimate retained names
    (e.g., "phenothiazine" for a 26-atom molecule) from partial names
    that only cover a small fragment.

    Args:
        name: IUPAC name string.

    Returns:
        True if the name contains at least one recognized ring-system token.
    """
    name_lower = name.lower()
    return any(token in name_lower for token in _RING_SYSTEM_TOKENS)


# ---------------------------------------------------------------------------
# The real coverage oracle (Task Z3)
# ---------------------------------------------------------------------------

# (name, canonical SMILES) -> "does this name denote exactly this molecule?"
# The oracle costs an OPSIN parse, so each distinct pair is asked once.
_PROVEN_COMPLETE_CACHE: Dict[tuple, bool] = {}

# Rollback lever. Set to a false value ('0'/'false'/'no'/'off') to restore the
# pre-Z3 behaviour EXACTLY -- sound, because the only thing the oracle is ever
# allowed to do here is turn a character count's REJECT into an ACCEPT.
_COVERAGE_ORACLE_ENV = "ORTHONYM_DECOMP_COVERAGE_ORACLE"


def _name_is_proven_complete(name: str, mol) -> bool:
    """MEASURED: does *name* denote exactly *mol*'s constitution?

    This is the real oracle -- ``validation/atom_coverage.py``, which parses
    the name back with OPSIN and compares INCHIKEY SKELETON BLOCKS -- not a
    character count. It exists because every "coverage" test in this module is
    really a test of how many CHARACTERS the name has, and a character count is
    anti-correlated with coverage: the correct ``cholesterol`` scores 0.393
    chars/HA while ``2-amino-2-(methylamino)acetamide``, a name that INVENTS
    atoms, scores 5.333.

    THREE PROPERTIES THIS FUNCTION IS RELIED ON FOR -- do not break them:

    1. **It is only ever consulted to RESCUE, never to reject.** Every call
       site sits on a path where a character count has *already decided to
       return False*; the oracle can only overturn that to True. So no name
       that is accepted today can be newly rejected, and the worst case of an
       oracle failure is the pre-Z3 behaviour. Measured on 77 molecules
       (48-molecule mixed corpus + the 29-molecule short-name class): the
       character counts produced 7 distinct rejections and the oracle AGREED
       with 6 of them. It is not a rubber stamp.

    2. **It answers CONSTITUTION, not PIN-preference.** A True here means "the
       same atoms, bonded the same way" and nothing more -- OPSIN proves a name
       VALID, never PREFERRED (the contributor guide). ``ethyl stearate`` is proven
       complete by this function and is NOT the PIN (``the Blue Book``
       puts ``(PIN)`` on ``octadecanoic acid``). That is acceptable *here* only
       because this predicate's question is "should we DECOMPOSE this
       molecule?", and decomposition cannot fix a non-preferred name -- it
       re-derives it. Never promote this to a correctness or PIN test.

    3. **It fails CLOSED.** With OPSIN absent, an unparseable name, or any
       exception, ``is_complete`` is False and no rescue happens, so behaviour
       is byte-identical to pre-Z3. Cost is bounded the same way: the oracle is
       reached only on the reject path and memoised per (name, molecule).

    Source of the constitution rule: ``validation/atom_coverage.py`` docstring,
    "``is_complete`` is therefore decided by CONSTITUTION -- the InChIKey
    skeleton block of the input compared with that of the parse-back -- and
    never by a threshold on a count."
    """
    if not name or mol is None:
        return False
    import os
    if os.environ.get(_COVERAGE_ORACLE_ENV, "1").strip().lower() in (
            "0", "false", "no", "off"):
        return False
    try:
        key = (name, Chem.MolToSmiles(mol))
    except Exception:
        return False
    cached = _PROVEN_COMPLETE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from ..validation.atom_coverage import validate_atom_coverage
        result = validate_atom_coverage(mol, name)
        # extra_atoms is redundant with constitution_match (identical skeleton
        # implies identical formula) and is asserted anyway, so that a future
        # change loosening is_complete cannot silently admit a name that
        # FABRICATES atoms -- the failure mode atom_coverage.py was rewritten
        # in 872cb124 to catch.
        proven = bool(result.is_complete and result.extra_atoms == 0)
    except Exception:
        proven = False
    _PROVEN_COMPLETE_CACHE[key] = proven
    return proven


# ---------------------------------------------------------------------------
# Name-size coverage heuristic (a phase-04)
# ---------------------------------------------------------------------------

def _name_covers_molecule(name: str, mol) -> bool:
    """Check if a pipeline name plausibly covers the whole molecule.

    ⚠ NOT a coverage measurement (Task Z2). The "estimated coverage" below is
    ``len(name) / 1.5 / heavy_atoms`` -- a CHARACTER COUNT rescaled by a
    constant. It is anti-correlated with real coverage: the correct retained
    name ``cholesterol`` scores 0.393 on its own 28-atom molecule, while a name
    that INVENTS atoms scores higher the more verbose it is. Real coverage is
    decided by constitution (InChIKey skeleton), never by a ratio --
    ``validation/atom_coverage.py``. That oracle spawns a JVM per call, so it
    cannot be used here: this function is called inside the decomposition
    recursion. Treat the number below as a cheap triage heuristic and nothing
    more; do not report it, or anything derived from it, as coverage.

    Only rejects when ALL conditions are met:
    (a) molecule has > 20 heavy atoms (small/medium always pass)
    (b) not in decomposition context (visited set empty)
    (c) molecule has cleavable bonds (alternative exists)
    (d) name is under 30 characters
    (e) estimated coverage < 0.45

    (a), (d) and (e) are the LIVE constants, re-read from the code 2026-08-02.
    This docstring previously said "> 15", "below 55%" and "< 0.55"; none of
    those values exists in the body, and the 0.55 was copied into the
    outward-facing `docs/ORTHONYM-vs-AUTONOM-COMPARATIVE-ANALYSIS.md` from
    here. Measured: (d) alone bypasses this function on 10 of the 16 molecules
    that reach it, and the rejection at (e) fired 0 times across 40 molecules.

    Args:
        name: The pipeline name.
        mol: RDKit Mol object for the molecule.

    Returns:
        True if name plausibly covers the molecule, False if it
        likely only describes a substructure.
    """
    try:
        heavy_atoms = mol.GetNumHeavyAtoms()

        # Small/medium molecule bypass: always accept for HA <= 20.
        # Calibrated at 20 (raised from 15) to avoid false positives on
        # molecules like heptanamide (HA=18, amide bond) where the pipeline
        # name is correct despite low coverage ratio.
        if heavy_atoms <= 20:
            return True

        # Decomposition context bypass: if visited set is non-empty,
        # we're evaluating a decomposition result -- always accept.
        from ..assembly.fragment_naming import _get_visited
        visited = _get_visited()
        if len(visited) > 0:
            return True

        # No-cleavable-bonds bypass: if no cleavable bonds exist,
        # the pipeline name is the best we can do regardless of coverage.
        from .bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        if not bonds:
            return True

        # Long name bypass: names >= 30 chars describe something substantial
        # even for very large molecules. This prevents false positives on
        # names like "N-2-hydroxydocosanoyltetracosanolate" (36 chars, 60 HA)
        # which describe complex multi-functional compounds.
        if len(name) >= 30:
            return True

        # Coverage estimation: approximate how many heavy atoms the
        # name describes. IUPAC names use ~1.5 chars per HA for
        # retained names and ~2.0 for substitutive. Use conservative
        # 1.5 as divisor to avoid over-rejection.
        estimated_ha = len(name) / 1.5
        coverage = estimated_ha / heavy_atoms

        # If estimated coverage is below threshold, the name is partial.
        # Calibrated at 0.45 (lowered from 0.55 via benchmark-driven tuning:
        # 0.55 -> rejected "2-methylhexadecanoate" (0.52), 0.50 -> rejected
        # "17-phenylheptadecyl acetate" (0.49)). 0.45 correctly rejects
        # "5-chloroquinoline" (0.44) and "(22E)-stigmasta-7,22-diene" (0.42)
        # while accepting legitimate decomposition names.
        if coverage < 0.45:
            # Task Z3: everything above is a CHARACTER COUNT, so it selects
            # WHICH names to measure -- it does not get to be the verdict.
            # The measurement decides.
            if _name_is_proven_complete(name, mol):
                return True
            return False

        return True

    except Exception:
        return True  # Never crash the quality gate


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def _amine_acyl_ambiguous(amine_smiles: str) -> bool:
    """True when a bare ``N-<acyl>-`` float onto this amine fragment is AMBIGUOUS.

     task 24 (acylspy). The float prepends ``N-`` to the amine PARENT name without
    saying WHICH nitrogen carries the acyl. If the amine fragment has >=2 ACYLATABLE
    nitrogens (an N bearing >=1 hydrogen, i.e. one that could accept the acyl), the
    name denotes >=2 distinct molecules -- OPSIN resolves it to one by its own rule, so
    the float ships an ambiguous name (and, with the OPSIN jar absent, ships it
    unverified). Refuse the float in that case. Fail-open (False) on an unparseable
    fragment -- the caller then relies on its existing quality/ gates, exactly as
    before this guard existed.
    """
    try:
        m = Chem.MolFromSmiles(amine_smiles)
        if m is None:
            return False
        # A bare italic `N-` locant binds to whichever nitrogen OPSIN's own rule
        # picks. An `N` is a candidate target if it is H-bearing (can accept the
        # acyl) OR an AROMATIC RING nitrogen: OPSIN will DEAROMATIZE a ring N to
        # host the acyl (measured: `N-formyl-2-(methylamino)-1,3-thiazole-5-
        # carboxylic acid` parsed with the formyl on the thiazole ring N, not the
        # exocyclic amino N). #29-a review-BLOCKER: the H-only count missed the
        # 0-H aromatic ring N, so a heteroaromatic amine parent + an exocyclic
        # amino floated an ambiguous `N-<acyl>` that shipped a WRONG constitution
        # at gate-off. Counting aromatic ring N as a target fails the float closed.
        return sum(1 for a in m.GetAtoms()
                   if a.GetSymbol() == 'N'
                   and (a.GetTotalNumHs() >= 1 or a.GetIsAromatic())) >= 2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Decision A: the N-acyl float onto a nitrogen that is not a suffix nitrogen
# ---------------------------------------------------------------------------

#: Scope of the gate below. ``None`` (the default) = DEMOTE: a float onto a non-suffix
#: nitrogen ships, recorded as a non-PIN fragment. A list = REFUSE: the float is withheld
#: and appended here, so the caller that opened the scope
#: (routing.dispatch_table._handle_peptide) can tell that its systematic attempt failed
#: because of this gate and fall back to the demoted float. The scope changes what the
#: naming path returns, so the nested memo keys carry ``nacyl_float_refusing``.
_NACYL_FLOAT_REFUSALS: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "orthonym_nonsuffix_nacyl_float_refusals", default=None)


def nacyl_float_refusing() -> bool:
    """True inside a refusal scope (``_NACYL_FLOAT_REFUSALS``); a memo-key component."""
    return _NACYL_FLOAT_REFUSALS.get() is not None


#: Endings of an amine-fragment name whose final (suffix or retained-parent) word
#: expresses a nitrogen -- the only names in which a bare 'N' locant designates that
#: nitrogen. A name ending otherwise ('glycinate', '2-aminoethyl sulfate',
#: '...-sulfonic acid') cites its nitrogen as an 'amino' prefix.
_N_SUFFIX_NAME_ENDINGS = ("amine", "aniline", "amide", "imide", "amidine", "imine",
                          "urea", "guanidine")


def _nacyl_float_n_is_suffix_nitrogen(amine_smiles: str,
                                      amine_name: Optional[str] = None) -> bool:
    """True when the nitrogen a bare ``N-<acyl>`` float acylates is a SUFFIX nitrogen of
    the amine parent: an atom of its principal characteristic group (the -amine of
    'cyclohexanamine', the -amide of 'acetamide', the -sulfonamide of
    'benzenesulfonamide'), which the italic locant 'N' designates. When the parent
    name is given it must also end in a nitrogenous word (``_N_SUFFIX_NAME_ENDINGS``):
    the structural principal group does not see a charged or functional-class parent
    ('glycinate', '2-aminoethyl sulfate'), whose nitrogen the name cites as 'amino'.

    False when that nitrogen is cited in the parent name as an 'amino' prefix -- the
    alpha-amino N of 'glycine' or '(2S)-2-aminopropanoic acid', whose acid is the
    principal characteristic group -- or is a ring atom that carries a numerical
    locant ('pyrrolidine-2-carboxylic acid', N = 1). There the float is not the PIN:
     "Substituents of the types -NH-CO-R and -NH-SO2-R" (the Blue Book
    :32991) names -NH-CO-R (1) as an 'amido' prefix and (2) as an 'acylamino' prefix,
    and "Method (1) generates preferred IUPAC names." (:32998); an N-substituted glycine
    is printed on the acetic acid parent, "[(methanesulfinothioyl)amino]acetic acid
    (PIN)" (:33213). User decision A (2026-09-26): retained amino-acid names stay at the
    PIN tier only when the nitrogen is unsubstituted; the N-substituted retained name is
    general nomenclature,:54480;:50943 identifies no PIN there).

    The acylated nitrogen is the fragment's only acylatable one: the float runs only
    after ``_amine_acyl_ambiguous`` found at most one (an H-bearing or aromatic N), and
    the capped attachment N always bears an H. Fail-open (True, the previous behaviour)
    when the fragment does not parse or that nitrogen is not unique.
    """
    try:
        m = Chem.MolFromSmiles(amine_smiles)
        if m is None:
            return True
        if amine_name and not amine_name.rstrip().endswith(_N_SUFFIX_NAME_ENDINGS):
            return False
        cands = [a.GetIdx() for a in m.GetAtoms()
                 if a.GetSymbol() == 'N'
                 and (a.GetTotalNumHs() >= 1 or a.GetIsAromatic())]
        if len(cands) != 1:
            return True
        n_idx = cands[0]
        if m.GetAtomWithIdx(n_idx).IsInRing():
            return False
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        _pg, matches = get_principal_group(m, detect_functional_groups(m))
        return any(n_idx in tuple(match) for match in (matches or ()))
    except Exception:
        return True


def gate_nonsuffix_nacyl_float(amine_smiles: Optional[str],
                               float_name: Optional[str],
                               amine_name: Optional[str] = None) -> Optional[str]:
    """Decision A gate for a bare ``N-<acyl>`` float (the amide branch of
    ``_try_single_bond_decompose`` and ``fragment_assembly._assemble_amide``).

    Returns ``float_name`` unchanged when the acylated N is a suffix nitrogen
    (``_nacyl_float_n_is_suffix_nitrogen``). Otherwise, inside a refusal scope the float
    is withheld (None) so that a systematic name can be built; outside one it ships as
    an honest demotion: validated as before, but recorded as a non-PIN fragment, so
    ``name_tiered`` labels any name that carries it below pin_verified.
    """
    if (not float_name or not amine_smiles
            or _nacyl_float_n_is_suffix_nitrogen(amine_smiles, amine_name)):
        return float_name
    refusals = _NACYL_FLOAT_REFUSALS.get()
    if refusals is not None:
        refusals.append(float_name)
        return None
    from ..metrics.provenance import record_non_pin_fragment
    record_non_pin_fragment(float_name)
    return float_name


def _name_quality_is_acceptable(name: str, mol) -> bool:
    """Check if an existing name is good enough (no decomposition needed).

    Returns True if the name looks acceptable, False if decomposition
    should be attempted.

    Criteria for UNACCEPTABLE names:
    - None, empty, or "unknown"
    - Suspiciously short for a complex molecule (heavy_atoms > 15,
      name shorter than heavy_atoms // 2)
    - Inadequate char/atom ratio for very large molecules (heavy_atoms > 25,
      ratio < 0.45)
    - No digits and no hyphens for a large molecule (heavy_atoms > 20),
      which suggests only a retained name for one fragment was returned
      (unless the name contains a recognized ring-system token)

    Args:
        name: The existing pipeline name (may be None).
        mol: RDKit Mol object for the molecule.

    Returns:
        True if name is acceptable (skip decomposition).
        False if name is poor (try decomposition).
    """
    if not name or name == "unknown":
        return False

    # Whitelist: known retained names that correctly identify a core
    # substructure even in large molecules (e.g., adenine in nucleotide
    # cofactors).
    #
    # Coverage guard (a phase-03): retained names are valid only if they
    # plausibly describe the whole molecule. "adenine" (7 chars) naming a
    # 58-HA molecule (ratio 0.12) indicates a partial match -- OPSIN
    # round-trips to adenine (10 HA), proving the name only covers 17%.
    # For molecules with HA > 20, require >= 0.25 chars/HA. This catches
    # adenine-in-CoA (58 HA, ratio 0.12) while preserving adenine
    # standalone (10 HA), phenothiazine (13 chars/14 HA = 0.93), and
    # flavone (7 chars/22 HA mock = 0.32).
    if name.lower() in _RETAINED_CORE_NAMES:
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms <= 20 or len(name) / heavy_atoms >= 0.25:
            return True
        # Fall through to other checks (may trigger decomposition)

    heavy_atoms = mol.GetNumHeavyAtoms()

    # Suspiciously short name for a complex molecule.
    #
    # ⚠ THIS IS THE DOMINANT GUARD IN THIS FUNCTION (Task Z2). It is a
    # character count -- `len(name) < heavy_atoms // 2` is `chars/HA < ~0.5`
    # written without a division, which is why a grep for `len(name) /` misses
    # it. Because it runs first and is stricter than the chars/HA tests below,
    # it SHADOWS them: it is what rejects `cholesterol` on a 28-heavy-atom
    # steroid (11 < 14), not the 0.45 test that used to sit at the end of this
    # function. Before re-tuning anything below, check whether this line has
    # already returned.
    if heavy_atoms > 15 and len(name) < heavy_atoms // 2:
        # Task Z3: this character count refuses CORRECT names in bulk. Asked
        # directly, it rejects `pyrene` (PIN, 16 HA), `coronene` (PIN, 24 HA),
        # `picene` (PIN, 22 HA) -- Blue Book "Retained names for
        # hydrocarbons used for parent ring components", Table 2.7, each marked
        # (PIN) -- and the systematic `henicosane`, `docosane`, `hexacosane`,
        # `icosanoic acid`. A retained or systematic name for a big skeleton is
        # SHORT by design, which is exactly what this line punishes. So it is
        # demoted to a pre-filter and the measurement decides.
        if _name_is_proven_complete(name, mol):
            return True
        return False

    #: chars/HA check for medium molecules (15-30 HA).
    #
    # ⚠ The example this comment used to give -- "catches names like
    # 'quinoline' (0.39 chars/HA) for a 23 HA molecule" -- is FALSE, measured
    # 2026-08-02: 'quinoline' is in _RETAINED_CORE_NAMES and 0.391 >= 0.25, so
    # the retained-name branch above returns True and this block is never
    # reached for it. What this block does catch is the opposite shape: a name
    # long enough to clear the // 2 test above but still terse, e.g.
    # 'ethyl stearate' (0.636 on 22 HA) -- a CORRECT name, sent to
    # decomposition for being short. This is the one chars/HA test in this
    # function measured to fire on the naming path.
    # Threshold 0.65 (lowered from 0.7 to avoid rejecting "2-methylicosane" at 0.68).
    # Cleavable-bond bypass: only reject if decomposition is actually possible.
    # "hexadecane" (0.625) for 16 HA has no cleavable bonds -> no point rejecting.
    if 15 < heavy_atoms <= 30:
        chars_per_ha = len(name) / heavy_atoms
        if chars_per_ha < 0.65:
            from .bond_cleavage import find_cleavable_bonds
            try:
                bonds = find_cleavable_bonds(mol)
                if bonds:
                    # Task Z3: measure before discarding (see the X guard above).
                    if _name_is_proven_complete(name, mol):
                        return True
                    return False  # Low ratio + cleavable bonds -> try decomposition
            except Exception:
                return False  # On error, still reject

    # REMOVED (Task Z2, 2026-08-02): a `heavy_atoms > 25 and
    # len(name)/heavy_atoms < 0.45 -> return False` test stood here. It was
    # UNREACHABLE, and provably so rather than by sampling: any name short
    # enough to trip it (len < 0.45*HA) is also short enough to trip the
    # `len(name) < heavy_atoms // 2` test above (len < HA//2), which returns
    # first. Checked exhaustively for every HA in 26..4000 -- no value admits
    # the one without the other. The trace agreed: 68 evaluations across 17 of
    # 40 molecules, 0 rejections. Deleting it changes no name.
    # Do not reintroduce a ratio test here; re-derive from the guard above.

    # Large molecule with no digits and no hyphens: likely just a retained
    # name for one fragment (e.g., "benzene" for a 25-atom ester).
    # a phase: expanded _RETAINED_CORE_NAMES handles known ring-system
    # retained names (phenothiazine, carbazole, flavone, etc.) via early
    # whitelist bypass above. For other names, the no-digits/no-hyphens
    # check remains at HA>20 to catch incomplete names.
    #
    # Fragment-aware threshold (a phase-02): when naming a fragment during
    # decomposition (visited set non-empty), use HA>30 to be more lenient --
    # medium-sized fragments with retained names are valid in decomposition
    # context. At top level (visited set empty), keep HA>20 for strictness.
    from ..assembly.fragment_naming import _get_visited
    visited = _get_visited()
    effective_threshold = 30 if len(visited) > 0 else 20
    if heavy_atoms > effective_threshold and name.lower() not in _RETAINED_CORE_NAMES:
        has_digits = any(c.isdigit() for c in name)
        has_hyphens = "-" in name
        if not has_digits and not has_hyphens:
            # Task Z3: "no locants" is a NAME-SHAPE proxy for under-coverage,
            # and it is wrong for a whole legitimate class -- an unbranched
            # parent hydride or acid needs no locants at all. MEASURED: this
            # exact line, and no other, refuses `henicosane` (21 HA),
            # `icosanoic acid` (22 HA), `docosanedioic acid` (26 HA) and
            # `octadecanedioyl dichloride` (22 HA), each of which is the
            # correct systematic name for the molecule it was asked about.
            # The measurement decides, as at the character-count guards above.
            # The guard's own stated target -- `benzene` naming a 25-atom
            # ester -- is untouched, because such a name can never be proven
            # complete.
            if _name_is_proven_complete(name, mol):
                return True
            return False

    # Multi-amide under-naming detection
    # If molecule has multiple DISTINCT amide carbonyls but the name only
    # references one, the name is partial and decomposition should be attempted.
    # We count distinct carbonyl C atoms (acid_atom) rather than raw amide
    # bond count so that ureas (NC(=O)N -- one carbonyl, two C-N bonds)
    # are NOT falsely flagged as multi-amide.
    from .bond_cleavage import find_cleavable_bonds

    try:
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b.get("type") == "amide"]
        # Distinct carbonyl carbons involved in amide bonds
        distinct_amide_carbonyls = len(
            set(b["acid_atom"] for b in amide_bonds)
        )

        if distinct_amide_carbonyls >= 2:
            # Count how many amide-related patterns the name captures.
            # Each named amide bond should produce "amino", "amido",
            # "amide", "acetamid", "formamid", or similar.
            # Multiplier prefixes (di-, tri-, tetra-) before amide tokens
            # indicate multiple groups, so count them accordingly.
            _MULT_MAP = {'di': 2, 'tri': 3, 'tetra': 4, 'penta': 5}
            amide_refs = 0
            # C1 fix: find_cleavable_bonds labels each hydrazide
            # C(=O)-N bond as type 'amide', so a dihydrazide has
            # distinct_amide_carbonyls==2. The name token 'hydrazide' contains
            # no amide/amido token, so without counting it the complete GENERAL
            # name ('butanedihydrazide') was falsely rejected as under-naming.
            # Count the hydrazide suffix tokens too. Longest-first so
            # 'carbohydrazide' is not partial-matched as 'hydrazide'. Additive:
            # this can only RAISE amide_refs (make the gate more lenient), never
            # newly reject a name that passed before.
            for m in re.finditer(
                r'(di|tri|tetra|penta)?'
                r'(carbohydrazide|hydrazide|hydrazid|amino|amido|amide|acetamid|formamid)',
                name, re.IGNORECASE,
            ):
                prefix = m.group(1)
                amide_refs += _MULT_MAP.get(prefix.lower(), 1) if prefix else 1
            # If the name captures fewer amide references than distinct
            # amide carbonyls, the name is partial -- reject it
            if amide_refs < distinct_amide_carbonyls:
                return False

    except Exception:
        pass  # If bond detection fails, don't block on it

    # Multi-ring fragment drop detection: if the molecule has a cleavable bond
    # separating exactly two distinct ring systems, but the name only references
    # one ring system, reject the name as incomplete.
    #
    # Conservative approach: only trigger when ALL conditions are met:
    # 1. Molecule has exactly 2 ring systems separated by cleavable bonds
    # 2. At least one ring system on each side has >= 5 ring atoms
    # 3. The name's char/heavy-atom ratio is < 1.5
    # Exactly 2 ring systems avoids over-triggering on glycosides and
    # natural products with 3+ ring systems where a single name is correct.
    try:
        # Reuse bonds from above if available, otherwise re-detect
        if 'bonds' not in dir():
            bonds = find_cleavable_bonds(mol)

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        if bonds and len(atom_rings) >= 2:
            ring_atom_sets = [set(r) for r in atom_rings]
            # Merge overlapping ring sets (fused rings are one system)
            merged_systems = []
            for rs in ring_atom_sets:
                merged = False
                for ms in merged_systems:
                    if rs & ms:
                        ms.update(rs)
                        merged = True
                        break
                if not merged:
                    merged_systems.append(set(rs))

            if len(merged_systems) == 2:
                # Only trigger for exactly 2 ring systems. Molecules with 3+
                # ring systems (glycosides, polycyclic natural products) often
                # have a single name that correctly covers all rings.
                for bond_info in bonds:
                    bond_idx = bond_info.get("bond_idx")
                    if bond_idx is None:
                        continue
                    # C4: an amine bridge between two ring systems is
                    # named SUBSTITUTIVELY (aniline parent + N-aryl prefix, e.g.
                    # diphenylamine -> 'N-phenylaniline'), NOT by decomposition.
                    # The substitutive name is legitimately compact (ratio ~1.15
                    # for N-phenylaniline / 13 HA), so the char-ratio proxy below
                    # would wrongly reject it and trigger a garbled cleavage
                    # ('phenylphenol'). The amine bridge is not a decomposition
                    # linkage (unlike ester/amide/ether); skip the ring-drop
                    # rejection for it. Scoped strictly to amine bond types.
                    if bond_info.get("type") in ("sec_amine", "tert_amine"):
                        continue
                    bond_obj = mol.GetBondWithIdx(bond_idx)
                    a1 = bond_obj.GetBeginAtomIdx()
                    a2 = bond_obj.GetEndAtomIdx()

                    # BFS from each side of the cleavable bond to find
                    # which ring systems are reachable from each side
                    def _reachable_atoms(start, exclude):
                        visited = set()
                        queue = deque([start])
                        while queue:
                            a = queue.popleft()
                            if a in visited:
                                continue
                            visited.add(a)
                            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                                nidx = nb.GetIdx()
                                if nidx != exclude and nidx not in visited:
                                    queue.append(nidx)
                        return visited

                    side1_atoms = _reachable_atoms(a1, a2)
                    side2_atoms = _reachable_atoms(a2, a1)

                    # Find ring systems on each side
                    side1_ring_systems = [
                        ms for ms in merged_systems
                        if ms & side1_atoms and len(ms) >= 5
                    ]
                    side2_ring_systems = [
                        ms for ms in merged_systems
                        if ms & side2_atoms and len(ms) >= 5
                    ]

                    if side1_ring_systems and side2_ring_systems:
                        # Both sides have substantial ring systems.
                        # A name covering both ring systems needs enough
                        # characters to name each (parent + prefix/locants).
                        # Use a higher ratio threshold (1.5) because naming
                        # two ring systems requires substantially more chars
                        # than naming one ring + simple substituents.
                        #
                        # ⚠ MEASURED INERT (Task Z2, 2026-08-02): reached 3
                        # times on 1 of 40 molecules, rejected 0 times. Unlike
                        # the 0.45 test deleted above, this is NOT provably
                        # unreachable -- it is a character count that happens
                        # never to have fired on the sample -- so it was left
                        # in place rather than removed on a 40-molecule zero.
                        # Removing it admits new behaviour and needs the full
                        # gate. It is also anti-correlated with what it claims
                        # to check: a verbose name naming ONE ring passes,
                        # a terse name naming BOTH fails.
                        ratio = len(name) / heavy_atoms if heavy_atoms else 999
                        if ratio < 1.5:
                            # Task Z3: measure before discarding. This ratio is
                            # anti-correlated with what it claims to check -- a
                            # verbose name covering ONE ring passes it, a terse
                            # name covering BOTH fails -- so it must not be the
                            # verdict on its own.
                            if _name_is_proven_complete(name, mol):
                                return True
                            return False
    except Exception:
        pass  # Guard: never let ring detection crash the quality gate

    # Multi-bond under-coverage detection (a phase-03): if molecule has
    # multiple distinct cleavable bond types but the name references fewer
    # than half, the name likely describes only one fragment of a
    # multi-component molecule.
    # Example: molecule with ester + phosphodiester bonds named "butanedioic
    # acid" only covers the acid fragment, not the ester or phosphodiester
    # linkages.
    #
    # IMPORTANT: This check is gated on coverage ratio < 0.8 to avoid
    # rejecting decomposition results that adequately describe the molecule.
    # Decomposition names like "N-2,3-dihydroxyhexacosanoylaminocyclohexane-
    # tetraol" (ratio 0.86) are complete despite not mentioning every bond
    # type, because they result from cleaving ONE bond and naming each half.
    # Only pipeline names with low coverage (< 0.8) are suspect.
    #
    # ⚠ THE PRE-FILTER IS A CHARACTER COUNT AND IT KEEPS THIS BLOCK SHUT
    # (Task Z2, 2026-08-02). `coverage_ratio` is `len(name)/heavy_atoms`, so
    # what "< 0.8" actually buys is: a name VERBOSE enough is exempt from the
    # bond-token check below -- which is a real structural test. Measured: the
    # ratio is computed 34 times across 13 of 40 molecules and the branch is
    # entered ZERO times. Neither of the two tests that name this block reach
    # it either -- test_quality_gate.py:360 is rejected earlier by the
    # `len(name) < heavy_atoms // 2` guard (its own comment concedes this) and
    #:331 by the chars/HA reject. So this block has no test coverage and
    # no observed execution.
    # NOT removed: a reachable window exists (ratio in [0.65, 0.8) for
    # 15 < HA <= 30), so unlike the 0.45 test deleted above it is not provably
    # dead, and dropping the pre-filter would START rejecting names. Either
    # change needs the full gate.
    try:
        # Reuse bonds from the multi-amide block above if available
        if 'bonds' not in dir():
            bonds = find_cleavable_bonds(mol)
        coverage_ratio = len(name) / heavy_atoms if heavy_atoms else 999
        if bonds and heavy_atoms > 15 and coverage_ratio < 0.8:
            distinct_bond_types = set(b["type"] for b in bonds)
            if len(distinct_bond_types) >= 2:
                # Use module-level boundary-aware token matching (a phase)
                # instead of substring matching to prevent false positives
                # like "polyester" triggering "ester" token.
                name_lower = name.lower()
                represented_types = 0
                for bt in distinct_bond_types:
                    tokens = _BOND_TYPE_TOKENS.get(bt, set())
                    if not tokens:
                        # ether: give benefit of doubt
                        represented_types += 1
                        continue
                    if _token_matches_name(name_lower, _COMPILED_TOKEN_PATTERNS, bt):
                        represented_types += 1
                # If less than half of distinct bond types are represented,
                # the name is partial -- reject it
                if represented_types < len(distinct_bond_types) / 2:
                    return False
    except Exception:
        pass  # Guard: never let bond detection crash the quality gate

    # Detect half-decomposition artifacts: consecutive duplicate words
    # e.g., "palmitate palmitate" indicates same fragment named twice
    words = name.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 3:
            return False

    # Detect garbled decomposition names with duplicated parent-like tokens.
    # E.g., "(ethanamide)-1-(...tetrahydropyranyl)ethanamide" has "ethanamide"
    # twice, indicating the assembly merged fragments incorrectly.
    # Substituent-like tokens (ending in -oxy, -yl, -amino, -ido) can
    # legitimately repeat (di-glucopyranosyloxy, dimethyl, etc.).
    _PARENT_SUFFIXES = ('amide', 'amine', 'anol', 'anone', 'anediol',
                        'anedione', 'anoic', 'anoate')
    _dup_words = re.findall(r'[a-z]{6,}', name.lower())
    if _dup_words:
        from collections import Counter as _DupCtr
        _dup_counts = _DupCtr(_dup_words)
        for _dw, _dcnt in _dup_counts.items():
            if _dcnt >= 2 and any(_dw.endswith(sfx) for sfx in _PARENT_SUFFIXES):
                return False

    # Name-size coverage heuristic (a phase-04): reject names that
    # describe less than ~55% of molecule heavy atoms when cleavable bonds
    # exist and decomposition hasn't been attempted yet.
    if not _name_covers_molecule(name, mol):
        return False

    return True


def _decomposition_is_worse(decomp_name: str, existing_name: str, mol) -> bool:
    """Check if decomposition produced a worse name than the existing pipeline.

    Returns True if existing_name should be preferred over decomp_name.
    This prevents decomposition from replacing an acceptable (if incomplete)
    name with a garbled or malformed one.

    Args:
        decomp_name: The decomposition result name.
        existing_name: The existing pipeline name (from name_pipeline_only).
        mol: RDKit Mol object for the molecule.

    Returns:
        True if existing_name is better (decomposition is worse).
    """
    # Garbled token detection: names containing fragments that indicate
    # malformed assembly (e.g., "anedicarboxamide", "aneyl", "cycloane")
    garbled_patterns = [
        'anedicarboxamide', 'aneyl', 'unknown', 'cycloane',
        'acidyl',       # "phosphonic acidyl" etc. -- malformed decomposition assembly
        'thioateyl',    # malformed thioester assembly
        'sulfonamideyl',  # malformed sulfonamide assembly
        'phosphateyl',  # malformed phosphodiester assembly
    ]
    for pattern in garbled_patterns:
        if pattern in decomp_name.lower() and pattern not in existing_name.lower():
            return True

    # Bracket mismatch: more open than close or vice versa
    for open_ch, close_ch in [('(', ')'), ('[', ']'), ('{', '}')]:
        if decomp_name.count(open_ch) != decomp_name.count(close_ch):
            return True

    # Duplicate parent-like tokens: if the decomposed name has the same
    # parent suffix name appearing more than once (e.g., "ethanamide...ethanamide"),
    # the assembly is likely garbled. Only flag parent-like tokens (amide, amine, etc.),
    # not substituent prefixes that can legitimately repeat.
    _PARENT_SFXS = ('amide', 'amine', 'anol', 'anone', 'anediol',
                    'anedione', 'anoic', 'anoate')
    _dwords = re.findall(r'[a-z]{6,}', decomp_name.lower())
    if _dwords:
        from collections import Counter as _DCtr
        _dcounts = _DCtr(_dwords)
        for _dw, _dcnt in _dcounts.items():
            if _dcnt >= 2 and any(_dw.endswith(sfx) for sfx in _PARENT_SFXS):
                return True

    # Detect duplicated structural fragment names: if a long token (>= 15 chars)
    # appears 2+ times in the decomposition but not in the existing name, the
    # decomposition likely split the molecule into two copies of the same
    # structural fragment (garbled assembly).
    if _dwords:
        _existing_words = set(re.findall(r'[a-z]{6,}', existing_name.lower()))
        for _dw, _dcnt in _dcounts.items():
            if (_dcnt >= 2 and len(_dw) >= 15
                    and _dw not in _existing_words):
                return True

    return False


# ---------------------------------------------------------------------------
# Coverage-based quality gate for decomposition results
# ---------------------------------------------------------------------------

def _coverage_is_adequate(name: str, mol, bond_type: str = "") -> bool:
    """Check if a decomposed name covers enough of the molecule.

    ⚠ THE LENGTH TEST HERE IS A PRE-FILTER, NOT THE VERDICT (Task Z3). A name
    that clears the character threshold is accepted on the threshold alone; a
    name that FAILS it is then MEASURED against the real oracle
    (``_name_is_proven_complete``) and kept if the measurement proves it
    denotes the molecule exactly. The character count can therefore no longer
    discard a name on its own.

    Uses name length as a first-pass proxy for atom coverage: a well-named
    molecule should have roughly 2+ characters per heavy atom (locants,
    prefixes, parent name, substituents).

    Tiered thresholds (a phase):
    - Functional-class types (ester, amide, glycosidic, carbamate, thioester)
      use 0.6 chars/HA -- these produce compact names like "phenyl palmitate".
    - Substitutive types (sulfonamide, phosphodiester, ether, default)
      use 0.8 chars/HA -- these produce longer substitutive names.

    Applied ONLY to decomposition results, NOT to existing pipeline names.

    Args:
        name: The decomposition-produced name (may be None).
        mol: RDKit Mol object for the molecule.
        bond_type: Bond type string from bond info dict (default="" uses 0.8).

    Returns:
        True if coverage is adequate.
        False if the name is too short for the molecule's size.
    """
    if not name:
        return False

    heavy_atoms = mol.GetNumHeavyAtoms()

    # Small molecules (<=10 heavy atoms) always pass -- coverage
    # heuristic is unreliable for tiny molecules.
    if heavy_atoms <= 10:
        return True

    # Tiered threshold: functional-class bond types produce compact names
    # (e.g., "phenyl palmitate" = 0.67 chars/HA) and use a lower threshold.
    # Substitutive bond types produce longer names and need a higher threshold.
    threshold = 0.6 if bond_type in _FUNCTIONAL_CLASS_TYPES else 0.8
    expected_min = int(heavy_atoms * threshold)

    # a phase: Retained-name coverage bonus.
    # Names containing retained-name tokens (adenine, cholesterol, etc.)
    # describe more structure than their character count suggests -- retained
    # names are intentionally shorter than systematic names. Apply a 1.5x
    # multiplier to effective name length for coverage comparison.
    # This bonus applies ONLY at the fragment-level coverage check, NOT
    # to the final assembled name quality check.
    name_lower = name.lower()
    retained_bonus = 1.0
    for rn in _RETAINED_CORE_NAMES:
        if rn in name_lower:
            retained_bonus = 1.5
            break

    effective_length = len(name) * retained_bonus
    if effective_length >= expected_min:
        return True

    # Task Z3: the comparison above is a CHARACTER COUNT written WITHOUT a
    # division -- `len(name) * bonus >= int(heavy_atoms * threshold)` -- which
    # is why neither a `len(name) /` grep nor a `heavy_atoms //` grep finds it,
    # and why the Task Z2 audit of this class missed this function entirely
    # despite its four production call sites. Same treatment as its siblings:
    # the count selects what to measure, the measurement decides.
    return _name_is_proven_complete(name, mol)


# ---------------------------------------------------------------------------
# Bond selection
# ---------------------------------------------------------------------------

# Priority order for bond types (lower = higher priority)
_BOND_TYPE_PRIORITY = {
    "ester": 1,
    "amide": 2,
    "phosphodiester": 3,
    "thioester": 4,
    "glycosidic": 5,
    "sulfonamide": 6,
    "carbamate": 7,
    "ether": 8,
    "thioether": 9,
    "sec_amine": 10,
}


def _validate_assembly_tokens(assembled: str, fragment_names: list) -> bool:
    """Validate assembled name references tokens from ALL input fragment names.

    Per: Each input fragment should contribute at least one morpheme
    (word token) to the assembled result. If a fragment's tokens are entirely
    absent, the assembly lost that fragment.

    Uses stem-level matching: for each fragment token, checks if a stem
    (first 3+ chars) appears in the assembled name. This handles IUPAC
    suffix transformations (e.g., "hexadecanoic" -> "hexadecanoate" shares
    stem "hexadecano", "ethanol" -> "ethyl" shares stem "eth").

    Args:
        assembled: The assembled IUPAC name string.
        fragment_names: List of individual fragment name strings.

    Returns:
        True if all fragments are represented in the assembly.
    """
    assembled_lower = assembled.lower()
    for name in fragment_names:
        # Extract significant tokens (>= 4 chars to skip locants/prefixes)
        tokens = [t for t in re.split(r'[-\s,()]+', name.lower()) if len(t) >= 4]
        if not tokens:
            continue  # Short-token-only fragments pass by default
        # Check if any token or its stem is present in the assembled name.
        # Uses progressively shorter stems to handle IUPAC suffix transformations
        # (e.g., "ethanol" -> "eth" stem in "ethyl", "propanoic" -> "propan" in "propanoate")
        found = False
        for tok in tokens:
            # Full token first
            if tok in assembled_lower:
                found = True
                break
            # Stem match: try stems of decreasing length down to 3 chars
            # (3-char stems like "eth" are safe for IUPAC roots: eth/meth/prop/etc.)
            for stem_len in range(len(tok) - 1, 2, -1):
                stem = tok[:stem_len]
                if stem in assembled_lower:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _name_sugar_fragment(smiles: str) -> Optional[str]:
    """Try to name a fragment as a sugar using the retained names lookup.

    If the fragment's canonical SMILES matches a known sugar, returns
    the glycosyloxy prefix (e.g., "beta-D-glucopyranosyloxy").
    Returns None if not a recognized sugar.

    Args:
        smiles: Canonical SMILES of the sugar fragment.

    Returns:
        Glycosyloxy prefix string, or None if not a known sugar.
    """
    from ..data.sugar_names import lookup_sugar, sugar_to_glycosyloxy_prefix

    sugar_info = lookup_sugar(smiles)
    if sugar_info:
        anomer, config, base_name = sugar_info
        return sugar_to_glycosyloxy_prefix(anomer, config, base_name)
    return None


def _select_best_bond(mol, bonds: List[Dict]) -> Dict:
    """Select the single best bond to cleave.

    Priority:
    1. By bond type: ester > amide > phosphodiester > thioester >
       glycosidic > sulfonamide > carbamate > ether
    2. Among same type: prefer most balanced split (smallest
       abs(frag1_atoms - frag2_atoms))

    Args:
        mol: RDKit Mol object.
        bonds: List of bond info dicts from find_cleavable_bonds.

    Returns:
        The single best bond dict to cleave.
    """
    if len(bonds) == 1:
        return bonds[0]

    def _balance_score(bond_info: Dict) -> int:
        """Estimate how balanced a split would be.

        Uses a BFS from each side of the bond to count atoms
        reachable without crossing the bond.
        """
        bond_idx = bond_info["bond_idx"]
        rdkit_bond = mol.GetBondWithIdx(bond_idx)
        a1 = rdkit_bond.GetBeginAtomIdx()
        a2 = rdkit_bond.GetEndAtomIdx()

        # BFS from a1 without crossing the bond
        visited1 = set()
        queue = deque([a1])
        while queue:
            curr = queue.popleft()
            if curr in visited1:
                continue
            visited1.add(curr)
            atom = mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in visited1:
                    # Don't cross the cleavage bond
                    if (curr == a1 and nidx == a2) or (curr == a2 and nidx == a1):
                        continue
                    queue.append(nidx)

        visited2 = set()
        queue = deque([a2])
        while queue:
            curr = queue.popleft()
            if curr in visited2:
                continue
            visited2.add(curr)
            atom = mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in visited2:
                    if (curr == a1 and nidx == a2) or (curr == a2 and nidx == a1):
                        continue
                    queue.append(nidx)

        return abs(len(visited1) - len(visited2))

    # Sugar-detection bypass: if any glycosidic bond leads to a known sugar,
    # prefer it. Sugar fragments get retained names (beta-D-glucopyranosyloxy)
    # which are better than systematic oxane/tetrahydropyran names.
    # Cap probe at 3 glycosidic bonds to limit performance impact.
    glycosidic_candidates = [b for b in bonds if b.get("type") == "glycosidic"][:3]
    for bond in glycosidic_candidates:
        from .fragment_capping import cleave_and_cap
        probe_frags = cleave_and_cap(mol, [bond], acid_side_oh=True)
        if probe_frags:
            for pf in probe_frags:
                if pf["side"] == "acid" and _name_sugar_fragment(pf["smiles"]):
                    return bond  # Sugar bond takes priority

    # Sort: first by type priority, then by balance (smaller = better)
    best = min(
        bonds,
        key=lambda b: (_BOND_TYPE_PRIORITY.get(b["type"], 99), _balance_score(b)),
    )

    # (Diacylamines): (R-CO)2NH is named as the N-acyl derivative of the
    # SENIOR primary amide -- 'N-acetylbenzamide (PIN)', NOT 'N-benzoylacetamide'. The
    # balance heuristic above has no seniority / awareness and can pick the
    # MORE senior acid as the cleaved (-> N-acyl prefix) side, inverting the parent.
    # When the balance winner is itself one of two amide bonds sharing the same amine
    # nitrogen (a true diacylamide sibling pair), re-pick to cleave the LESS senior
    # acid so the senior acid stays the retained '-amide' parent (benzoic > acetic by
    # ring>chain). Scoped to the balance winner's own sibling group, so it
    # never overrides a higher-priority bond type or a non-sibling global winner; a
    # genuine seniority tie (identical acyls) falls through unchanged, keeping
    # N-formylformamide / N-acetyl-N-cyclopentylacetamide correct.
    if best.get("type") == "amide" and best.get("amine_atom") is not None:
        siblings = [
            b for b in bonds
            if b.get("type") == "amide" and b.get("amine_atom") == best["amine_atom"]
        ]
        if len(siblings) >= 2:
            from .fragment_capping import cleave_and_cap
            from .fragment_ranker import score_fragment_seniority
            scored = []
            for b in siblings:
                acid_smiles = None
                for pf in cleave_and_cap(mol, [b], acid_side_oh=True):
                    if pf.get("side") == "acid":
                        acid_smiles = pf["smiles"]
                        break
                scored.append((score_fragment_seniority(acid_smiles), b))
            scored.sort(key=lambda t: t[0])  # LOWER = MORE senior
            if scored[0][0] != scored[-1][0]:  # strict seniority difference only
                return scored[-1][1]  # cleave the LEAST senior acid's bond

    # (the Blue Book): "Esters that do not qualify for
    # multiplicative names as described above are named as monoesters and other
    # ester components are expressed as prefixes by substitutive nomenclature";
    # 'methyl 2-chloro-5-[3-(ethoxycarbonyl)phenoxy]benzoate (PIN) (not ethyl
    # 3-[4-chloro-3-(methoxycarbonyl)phenoxy]benzoate; the parent structure of the
    # PIN has more substituents)' (:31813), the rule of. Among ester bonds
    # whose acid components sit on ring systems of one skeleton (no choice),
    # the cleaved ester is the one whose ring carries more substituents; the
    # balance order above decides only what this leaves tied.
    if best.get("type") == "ester":
        peers = _ester_peers_by_parent_substituents(mol, bonds, best)
        if peers:
            best = min(peers, key=_balance_score)

    return best


def _ester_acyl_ring(mol, acyl_c: int):
    """(ring-system atoms, skeleton key) of the ring the ester acyl carbon is bonded
    to, or None for an acyl carbon on a chain."""
    ri = mol.GetRingInfo()
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(acyl_c).GetNeighbors() if n.IsInRing()]
    if len(nbrs) != 1:
        return None
    system = set()
    rings = [set(r) for r in ri.AtomRings()]
    frontier = [r for r in rings if nbrs[0] in r]
    while frontier:
        r = frontier.pop()
        if r <= system:
            continue
        system |= r
        frontier.extend(x for x in rings if x & system and not x <= system)
    key = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(system), canonical=True,
                                   isomericSmiles=False)
    return system, key


def _ester_peers_by_parent_substituents(mol, bonds, best):
    """The ester bonds whose acid ring has the most substituents,
    among those on a ring of the same skeleton as ``best``'s;  when ``best``
    already has the most or no such comparison applies."""
    try:
        ref = _ester_acyl_ring(mol, best["acid_atom"])
        if ref is None:
            return []
        counted = []
        for b in bonds:
            if b.get("type") != "ester":
                continue
            ring = _ester_acyl_ring(mol, b["acid_atom"])
            if ring is None or ring[1] != ref[1]:
                continue
            system = ring[0]
            n_subs = sum(1 for a in system for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                         if nb.GetIdx() not in system and nb.GetIdx() != b["acid_atom"]
                         and nb.GetAtomicNum() > 1)
            counted.append((n_subs, b))
        if len(counted) < 2:
            return []
        top = max(n for n, _ in counted)
        best_n = next((n for n, b in counted if b is best), None)
        if best_n is None or best_n == top:
            return []
        return [b for n, b in counted if n == top]
    except Exception:  # noqa: BLE001 - keep the balance choice
        return []


# ---------------------------------------------------------------------------
# Single-bond decomposition helper
# ---------------------------------------------------------------------------

#: (the Blue Book '7 Acids',:18182 '9 Esters',:18184 '11 Amides'):
#: the class a split's functional-class or amide assembly expresses as the principal
#: characteristic group of the whole name, as its most senior member in
#: ``rules.seniority.SENIORITY_ORDER``.
_SPLIT_EXPRESSED_CLASS = {
    "amide": "primary_amide",
    "sulfonamide": "primary_amide",
    "ester": "ester",
    "thioester": "ester",
    "carbamate": "ester",
}


def _molecule_principal_group(mol, exclude_atoms=frozenset()) -> Optional[str]:
    """The principal characteristic group of ``mol``, leaving out acid derivatives whose
    linkage lies in a ring: class 9 "lactones and other cyclic esters are named as
    heterocycles" (the Blue Book), class 11 likewise for cyclic amides (:18184), so a
    lactone or a lactam is never the senior class that rules out a split. Groups holding
    an atom of ``exclude_atoms`` (the carbonyl carbon of the group a split cuts) are left
    out too: the cut group is the split's own, not a class elsewhere in the molecule."""
    if mol is None:
        return None
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import SENIORITY_ORDER, get_principal_group
        lo, hi = SENIORITY_ORDER.index("anhydride"), SENIORITY_ORDER.index("stibinate_ester")
        derivative = set(SENIORITY_ORDER[lo:hi + 1])

        def cyclic(match):
            atoms = list(match)
            return any(mol.GetBondBetweenAtoms(a, b) is not None
                       and mol.GetBondBetweenAtoms(a, b).IsInRing()
                       for i, a in enumerate(atoms) for b in atoms[i + 1:])

        groups = {}
        for name, matches in detect_functional_groups(mol).items():
            matches = [m for m in matches if not (set(m) & set(exclude_atoms))]
            if name in derivative:
                matches = [m for m in matches if not cyclic(m)]
            if matches:
                groups[name] = matches
        pg, _ = get_principal_group(mol, groups)
        return pg
    except Exception:  # noqa: BLE001 -- unknown: the split is not judged
        return None


def _split_parent_is_junior(mol, expressed: Optional[str], exclude_atoms=frozenset()) -> bool:
    """True when the molecule's principal characteristic group is senior to the
    class ``expressed`` that a split would cite as the parent's principal group: the name
    would then carry the senior group as a prefix ('...carboxamide' with a
    'carboxymethyl' prefix), which is not the PIN. The split declines; another split or
    the substitutive pipeline names the molecule."""
    if not expressed:
        return False
    # classes 1-6 (radicals, ions, zwitterions;:18158-:18168) are senior to every
    # suffix class, and the group perception does not rank them: a charged molecule
    # is not judged here. (Judged, quick-wins Q6: a split of a charged molecule can be
    # right -- the amide split of an N-acyl glycinate names the anion parent,
    # '...glycinate', tests/unit/decomposition/test_split_parent_class_p41.py -- so it
    # is not declined; the read-back judges each such name.)
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return False
    pg = _molecule_principal_group(mol, exclude_atoms)
    rank_pg, rank_expressed = _p41_rank(pg), _p41_rank(expressed)
    if rank_pg is None or rank_expressed is None:
        return False
    return rank_pg < rank_expressed


#: subtypes of one class ranked together class 11 "Amides [in the order of the
#: corresponding acids...]",:18184; the alcohols and the amines are one class each)
_P41_SAME_CLASS = (
    ("primary_amide", "secondary_amide", "tertiary_amide"),
    ("primary_sulfonamide", "secondary_sulfonamide", "tertiary_sulfonamide"),
    ("primary_sulfinamide", "secondary_sulfinamide", "tertiary_sulfinamide"),
)


def _p41_rank(name: Optional[str]) -> Optional[int]:
    """The SENIORITY_ORDER position of the class of ``name``: the first position of any
    member of its class (primary / secondary / tertiary alcohol, amine or amide rank
    alike)."""
    from ..rules.seniority import _SENIORITY_PARENT, SENIORITY_ORDER
    if name not in SENIORITY_ORDER:
        return None
    members = [name]
    parent = _SENIORITY_PARENT.get(name)
    if parent is not None:
        members = [m for m, c in _SENIORITY_PARENT.items() if c == parent]
    for group in _P41_SAME_CLASS:
        if name in group:
            members = list(group)
    return min(SENIORITY_ORDER.index(m) for m in members if m in SENIORITY_ORDER)


def _unproven_split_key(mol, bond: Dict, style: str):
    """The key of one split's unproven verdict (``_try_single_bond_decompose``): the atom-ordered
    graph and the bond the split cuts, the style, and everything else the fragment namings read
    (the ambient breadth context, the N-acyl float scope, the peptide re-entrancy flag, the
    locant and isotope scopes, the fragment-session depth)."""
    from ..assembly.fragment_naming import _session_depth
    from ..assembly.locant_omission import (_ISOTOPE_PARENT_POSITIONAL,
                                            _ISOTOPIC_NAMING_SCOPE,
                                            forced_locant_reason)
    from ..assembly.substituent_enumerator import _mol_graph_key
    from ..metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx,
                                      full_coverage_ctx, general_fallback_ctx)
    from ..routing import dispatch_table as _dispatch_table
    return (_mol_graph_key(mol), bond["type"], bond.get("bond_idx"), bond.get("acid_atom"),
            bond.get("alkyl_atom"), bond.get("amine_atom"), bool(bond.get("roles_swapped")),
            style, general_fallback_ctx.get(), best_effort_ctx.get(),
            allow_aromatic_general_ctx.get(), full_coverage_ctx.get(),
            nacyl_float_refusing(), _dispatch_table._PEPTIDE_SUBST_ACTIVE,
            forced_locant_reason(), _ISOTOPIC_NAMING_SCOPE.get(),
            _ISOTOPE_PARENT_POSITIONAL.get(), _session_depth())


#: The share of a molecule's analysis-call budget (``fragment_naming._ANALYSIS_CALL_BUDGET``)
#: that its searches may spend on other bonds, from the start of its first unproven split: one
#: eighth (62 of 500 calls). The first unproven split of the three 70-130-atom molecules
#: of the sample split cost 10, 104 and 104 calls (naming the remainder); the nine
#: diacylglycerophospholipids and sterol / terpene esters whose other ester gives the name cost
#: 0-3 and need under 8 more.
_UNPROVEN_EXPLORATION_SHARE = 8


def _note_unproven_split(budget_before) -> None:
    """Fix, once per naming scope and when the first unproven split is met, the level of the
    analysis-call budget at which the molecule's searches stop opening other bonds
    (``_exploration_exhausted``): the share below the level ``budget_before`` the budget
    stood at when that split was begun, so the split's own cost is part of the share. No level
    without an armed budget or a naming scope."""
    from ..assembly import fragment_naming as _fn
    from ..assembly import memo as _memo
    if _memo.side_get("unproven_exploration", "floor") is not None:
        return
    ceiling = _fn._ANALYSIS_CALL_BUDGET
    floor = (budget_before - max(1, ceiling // _UNPROVEN_EXPLORATION_SHARE)
             if budget_before is not None and ceiling > 0 else False)
    _memo.side_put("unproven_exploration", "floor", floor)


def _exploration_exhausted() -> bool:
    """True when the naming scope has spent the share of its analysis-call budget that the
    searches may spend on other bonds once the molecule has met an unproven split
    (``_note_unproven_split``). A molecule that met none has no share and is never bounded.

    The search that follows an unproven split cleaves every other bond, names each remainder
    (up to 120 atoms) in full, and does so again in every remainder it names: it is the
    far larger search a molecule the general pipeline names in 30 s may never leave (four
    70-130-atom molecules ran out of the budget in it and were abstained, or ran 64 s). It is
    kept where it is cheap (a diacylglycerophosphoglycerol finds the ester name of its other
    ester in fewer than 8 analysis calls) and ends once its share is spent, the first split's own
    cost included: a split whose remainder alone costs the share (104 calls for a 126-atom
    alcohol) opens nothing more, and the molecule is named by the general pipeline with the rest
    of the budget. A count of operations, not a clock."""
    from ..assembly import fragment_naming as _fn
    from ..assembly import memo as _memo
    floor = _memo.side_get("unproven_exploration", "floor")
    if floor is None or floor is False:
        return False
    remaining = getattr(_fn._fragment_guard, "analysis_budget", None)
    return remaining is not None and remaining <= floor


def _try_single_bond_decompose(mol, bond: Dict, style: str = "pin") -> Optional[str]:
    """Attempt decomposition using a single bond.

    Extracted from try_decompose Steps 4-8. Contains the full
    cleave-cap-name-assemble pipeline for one bond. Handles all 8
    bond types: ester, amide, phosphodiester, thioester, glycosidic,
    sulfonamide, carbamate, ether.

    Args:
        mol: RDKit Mol object to decompose.
        bond: Bond info dict from find_cleavable_bonds.
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Assembled multi-component IUPAC name, or None if decomposition
        fails at any step (capping, size guard, fragment naming, assembly).
    """
    from ..assembly import fragment_naming as _fn
    from ..assembly import memo as _memo
    from .fragment_assembly import assemble_fragment_name
    from .fragment_capping import cleave_and_cap

    # The level of the analysis-call budget when this split is begun (``_note_unproven_split``).
    _budget_before = getattr(_fn._fragment_guard, "analysis_budget", None)

    # An unproven verdict is final for the molecule being named: the same split asked again
    # (the decomposition is entered again from the charged-route re-entry and from every
    # fragment naming that reaches the same remainder) names the same fragments, gets the same
    # answer from the assembler, and cannot prove it. The first ask names the remainder in full;
    # a repeat must not name it again (each repeat of a 120-atom remainder charges the analysis
    # budget its first naming charged). Held for the naming scope only, and not read in
    # ``verify`` mode, which recomputes.
    _split_key = _unproven_split_key(mol, bond, style) if _memo._MODE == "on" else None
    if _split_key is not None and _memo.side_get("unproven_split", _split_key):
        _note_unproven_split(_budget_before)
        return None

    # Cleave and cap
    # For ether bonds, the acid side (larger fragment) gets H-cap (no OH),
    # while the alkyl side naturally keeps the ether oxygen as an alcohol.
    acid_oh = bond["type"] != "ether"
    fragments = cleave_and_cap(mol, [bond], acid_side_oh=acid_oh)
    if not fragments or len(fragments) < 2:
        return None

    # Size guard -- each fragment must be strictly smaller than parent
    parent_heavy = mol.GetNumHeavyAtoms()
    for frag in fragments:
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol and frag_mol.GetNumHeavyAtoms() >= parent_heavy:
            return None  # Fragment not smaller -- abort

    # Sort fragments smallest-first (by heavy atom count) so smaller
    # fragments populate the runtime cache before larger ones that may
    # contain similar structural motifs (IUPAC reuse principle).
    def _frag_sort_key(frag):
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        return frag_mol.GetNumHeavyAtoms() if frag_mol else 999
    fragments.sort(key=_frag_sort_key)

    # Name each fragment with fallback (a phase:)
    # a phase /: build a parallel fragment_smiles dict in lockstep with
    # fragment_names, keyed by the SAME frag["side"], so the glycoside assembler
    # can inspect fragment structure (the sugar-skeleton deriver and the
    # aglycone seniority guard both need the SMILES, not just the names). This
    # is purely additive -- non-glycoside callers never read fragment_smiles.
    fragment_names = {}
    fragment_smiles = {}
    for frag in fragments:
        frag_name = None

        # Sugar intercept: for glycosidic bonds, try sugar lookup on acid-side fragment
        if bond["type"] == "glycosidic" and frag["side"] == "acid":
            frag_name = _name_sugar_fragment(frag["smiles"])

        # Fall through to fallback naming if sugar lookup failed or non-sugar fragment
        if not frag_name:
            frag_name = _name_fragment_with_fallback(frag["smiles"])

        if not frag_name or "unknown" in frag_name.lower():
            return None  # Truly unnameable -- abort
        fragment_names[frag["side"]] = frag_name
        fragment_smiles[frag["side"]] = frag["smiles"]

    # --- Seniority-based substitutive assembly for swapped-role bonds ---
    # When _maybe_swap_parent_roles detected that the non-acid fragment
    # is the correct parent (roles_swapped=True), attempt substitutive
    # naming: acid fragment becomes a prefix on the larger parent fragment.
    # The acid_atom/alkyl_atom in the bond dict are NOT swapped -- only the
    # flag is set. We swap the naming roles here at assembly time.
    # The roles-swapped ester / thioester (the alcohol side is the parent, the acid
    # side its acyloxy / acylthio prefix) is not glued here: gluing '(acyloxy)' in
    # front of the parent's name cannot place the prefix's locant, and the parent was
    # named with the esterified O still on it (the '(acetyloxy)-...-6-ol' shape: no
    # locant, the O cited twice) -- names only the read-back stopped (quick-wins Q6).
    # The roles_swapped flag stays informational: the assembly below names the
    # bond with the original atom roles, as it did when the glue failed.

    # Substitutive naming preference: when the acid fragment already has a
    # principal group (sulfonic acid, phosphonic acid), prefer substitutive
    # naming over functional class assembly. This handles cases like sulfa
    # drugs where the acid fragment is already a well-formed parent name.
    #
    # IMPORTANT: Do NOT apply substitutive preference for esters or amides.
    # Esters use functional class naming ("alkyl alkanoate") and amides use
    # amide naming ("N-alkylalkanamide") -- these are the correct IUPAC forms.
    # Substitutive preference is only for bond types where the acid fragment
    # is already a principal-group parent that should receive a prefix.
    _SUBSTITUTIVE_BOND_TYPES = frozenset({
        "sulfonamide", "thioester", "phosphodiester",
    })
    if bond["type"] in _SUBSTITUTIVE_BOND_TYPES:
        acid_name = fragment_names.get("acid", "")
        _PG_INDICATORS = ("-ic acid", "-sulfonic acid", "-phosphonic acid",
                          "-carboxylic acid")
        acid_has_pg = any(acid_name.lower().endswith(ind) for ind in _PG_INDICATORS)

        if acid_has_pg:
            from .fragment_assembly import _alcohol_to_alkyl, _amine_to_prefix, _join_components
            alkyl_name = fragment_names.get("alkyl") or fragment_names.get("amine")
            if alkyl_name:
                sub_prefix = _alcohol_to_alkyl(alkyl_name) or _amine_to_prefix(alkyl_name)
                if sub_prefix:
                    substitutive_name = _join_components(sub_prefix, acid_name)
                    # Validate substitutive attempt: must pass quality gate and
                    # not be worse than functional class assembly
                    if substitutive_name and _name_quality_is_acceptable(substitutive_name, mol):
                        return substitutive_name

    # Amide seniority-based assembly: when amine fragment has higher
    # seniority than acid fragment, use substitutive naming
    # (amine becomes parent, acid becomes acyl prefix).
    # Guard: skip if roles_swapped is already True (detection-time swap
    # already handled the seniority assignment -- swapping again would
    # double-invert). In practice, amide roles_swapped is always False
    # (amides are exempt from detection-time swap), but this guard
    # provides defense-in-depth.
    if bond["type"] == "amide" and not bond.get("roles_swapped"):
        try:
            from .fragment_ranker import acid_is_more_senior
            acid_frag = next((f for f in fragments if f["side"] == "acid"), None)
            amine_frag = next((f for f in fragments if f["side"] == "amine"), None)
            if acid_frag and amine_frag:
                if (not acid_is_more_senior(acid_frag["smiles"], amine_frag["smiles"])
                        and not _split_parent_is_junior(
                            mol, _molecule_principal_group(
                                Chem.MolFromSmiles(amine_frag["smiles"])),
                            {bond.get("acid_atom")})):
                    # Amine is more senior -> substitutive naming
                    from ..assembly.naming_utils import (
                        _wrap_n_substituent,
                        enclose_if_compound,
                    )
                    from .fragment_assembly import _acid_to_acyl, _join_components
                    acid_name = fragment_names.get("acid", "")
                    amine_name = fragment_names.get("amine", "")
                    acyl = _acid_to_acyl(acid_name)
                    # task 24: a bare 'N-<acyl>-' prefix names WHICH nitrogen bears
                    # the acyl only by position in the parent name -- so if the amine
                    # fragment has >=2 ACYLATABLE nitrogens (N with >=1 H) the float is
                    # AMBIGUOUS (denotes >=2 molecules; OPSIN guesses one). Refuse to
                    # float and fall through -> honest abstention / a locant-anchored
                    # inline candidate, never an ambiguous name. Closes the ORIG/MIRROR
                    # ambiguity + the no-jar fail-open hole (acylspy, PIN-tier T1).
                    if acyl and amine_name and not _amine_acyl_ambiguous(
                            amine_frag["smiles"]):
                        # (the Blue Book): a compound acyl prefix takes
                        # parentheses ('N-(4-hydroxyoctanoyl)...', not
                        # 'N-4-hydroxyoctanoyl...'). _wrap_n_substituent only escalates
                        # an existing enclosure, so add the base level first, as
                        # fragment_assembly._assemble_amide does.
                        wrapped_acyl = _wrap_n_substituent(enclose_if_compound(acyl))
                        sub_name = f"N-{_join_components(wrapped_acyl, amine_name)}"
                        if sub_name and _name_quality_is_acceptable(sub_name, mol):
                            # Decision A: refused (inside _handle_peptide's systematic
                            # attempt) or demoted when the acylated N is not a suffix
                            # nitrogen of the amine parent.
                            sub_name = gate_nonsuffix_nacyl_float(
                                amine_frag["smiles"], sub_name, amine_name)
                            if sub_name:
                                return sub_name
        except Exception:
            pass  # Any failure: fall through to normal assembly

    #: a functional-class / amide assembly cites the bond's class as the
    # principal group of the whole name; decline it when the molecule holds a senior
    # class elsewhere (an acid beside an amide or an ester). The amide assembler cites
    # the amide only for a simple amine ('N-methyl...amide'); otherwise it floats the
    # acyl onto the amine fragment, whose own principal group is then the parent's.
    _expressed = _SPLIT_EXPRESSED_CLASS.get(bond["type"])
    if bond["type"] == "amide":
        from .fragment_assembly import _amine_to_prefix
        if not _amine_to_prefix(fragment_names.get("amine") or ""):
            _amine_frag = next((f for f in fragments if f["side"] == "amine"), None)
            _expressed = (_molecule_principal_group(Chem.MolFromSmiles(_amine_frag["smiles"]))
                          if _amine_frag else None)
    if _split_parent_is_junior(mol, _expressed, {bond.get("acid_atom")}):
        return None

    # Assemble (delegate to fragment_assembly module)
    # a phase /: thread fragment_smiles additively; only the glycoside
    # assembler reads it, every other assembler ignores the kwarg.
    # Incr-1a: thread the parent SMILES too -- the glycoside assembler
    # RT-gates the STRUCTURAL aglycone-substituent fallback against it (0-wrong).
    # Purely additive; every other assembler ignores the kwarg.
    _outcome: Dict[str, str] = {}
    _assembled = assemble_fragment_name(
        bond["type"], fragment_names, style=style,
        fragment_smiles=fragment_smiles,
        parent_smiles=Chem.MolToSmiles(mol),
        outcome=_outcome,
    )
    # An assembler that proves its group word returns None when no candidate round-trips, and
    # reports that it built a name it could not prove: the split is UNPROVEN, which is not the
    # same as unnamed. The name is not used. The molecule's searches are bounded from here
    # (``_exploration_exhausted``); a repeat of the split is not named again.
    if not _assembled and _outcome.get("unproven"):
        _note_unproven_split(_budget_before)
        if _split_key is not None:
            _memo.side_put("unproven_split", _split_key, True)
    if bond["type"] == "ester" and _assembled:
        # Branch review fixes, the Blue Book;:18875):
        # when another ester group shares this ester's acid parent the molecule is
        # an ester of ONE polyacid, and 'methyl 4-amino-3-(methoxycarbonyl)pent-3-
        # enoate' (the functional-class ester of one group, the other cited as a
        # prefix) is not its PIN 'dimethyl 2-(1-aminoethylidene)butanedioate'. Name
        # the polyester when its acid component can be named; otherwise keep the
        # mono-ester name, labelled below the PIN.
        try:
            from ..rules.esters import (
                ester_shares_its_acid_with_another_ester,
                name_polyacid_polyester,
            )
            if ester_shares_its_acid_with_another_ester(mol, bond["acid_atom"]):
                _poly = name_polyacid_polyester(mol)
                if _poly:
                    return _poly
                from ..metrics.provenance import record_non_pin_label
                record_non_pin_label(_assembled)
        except Exception:  # noqa: BLE001 -- the label must never break naming
            from ..metrics.provenance import record_non_pin_label
            record_non_pin_label(_assembled)
    if bond["type"] in ("sulfonamide", "amide") and _assembled:
        # User ruling D3 (2026-10-02): a C-substituent of a one-carbon sulfonamide
        # cites '1-' when it could stand on the nitrogen, or an N-substituent on the
        # carbon, the Blue Book; '(1-cyclohexylmethanesulfonamido)acetic
        # acid',:33024). A sulfonamide parent is assembled here from a fragment name
        # (its acid's, or the amine fragment an N-acyl group floats onto), which cannot
        # carry that locant, so the name is labelled below the PIN.
        try:
            if _one_carbon_sulfonamide_cites_parent_locant(
                    mol, _sulfonamide_parent_match(mol, bond, fragment_names)):
                from ..metrics.provenance import record_non_pin_label
                record_non_pin_label(_assembled)
        except Exception:  # noqa: BLE001 -- the label must never break naming
            from ..metrics.provenance import record_non_pin_label
            record_non_pin_label(_assembled)
    return _assembled


_SULFONAMIDE_UNIT = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[NX3]")


def _sulfonamide_parent_match(mol, bond, fragment_names):
    """The (S, O, O, N) atoms of the sulfonamide whose name is the parent of the
    assembled name, else None: the cleaved sulfonamide itself, or -- for an amide
    whose acyl floats onto the amine fragment (no simple amine prefix, the branch
    ``_assemble_amide`` takes) -- the sulfonamide holding the amide nitrogen."""
    if bond["type"] == "sulfonamide":
        return bond.get("match")
    from .fragment_assembly import _amine_to_prefix
    if _amine_to_prefix(fragment_names.get("amine") or ""):
        return None
    n_idx = bond.get("amine_atom")
    for match in mol.GetSubstructMatches(_SULFONAMIDE_UNIT):
        if match[3] == n_idx:
            return match
    return None


def _one_carbon_sulfonamide_cites_parent_locant(mol, match) -> bool:
    """True when the sulfonamide ``match`` (S, O, O, N) sits on a one-carbon acyclic
    parent whose substituent must cite the locant '1' -- the licence of the one-carbon
    parents, ``locant_omission.mononuclear_parent_locant_required``, read on the whole
    molecule. False for any other shape (a longer chain, a ring carbon, no match)."""
    if not match or len(match) < 4:
        return False
    s_atom = mol.GetAtomWithIdx(int(match[0]))
    carbons = [nb for nb in s_atom.GetNeighbors() if nb.GetAtomicNum() == 6]
    if len(carbons) != 1 or carbons[0].IsInRing():
        return False
    c_atom = carbons[0]
    if any(nb.GetAtomicNum() == 6 and not nb.IsInRing() for nb in c_atom.GetNeighbors()):
        return False
    from types import SimpleNamespace

    from ..assembly.locant_omission import mononuclear_parent_locant_required
    return mononuclear_parent_locant_required(SimpleNamespace(
        mol=mol, principal_chain=[c_atom.GetIdx()],
        principal_group_atoms=[tuple(int(i) for i in match[:4])],
        is_cyclic=False, chain_is_parent=True))


# ---------------------------------------------------------------------------
# Multi-bond same-type decomposition helper
# ---------------------------------------------------------------------------

def _try_multi_bond_decompose(
    mol, bonds: List[Dict], style: str = "pin"
) -> Optional[str]:
    """Attempt decomposition using multiple same-type bonds simultaneously.

    Cleaves all provided bonds at once, names each fragment, and assembles
    a multi-component name. Currently supports ester bonds (polyester naming
    per IUPAC. Other bond types fall through to None.

    Args:
        mol: RDKit Mol object to decompose.
        bonds: List of bond info dicts (must be >= 2, all same type).
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Multi-component IUPAC name, or None if decomposition fails.
    """
    # Validate: need >= 2 bonds, all same type
    if not bonds or len(bonds) < 2:
        return None

    bond_type = bonds[0]["type"]
    if not all(b["type"] == bond_type for b in bonds):
        return None

    from .fragment_assembly import (
        _assemble_multi_amide,
        _assemble_multi_ester,
        _assemble_multi_glycoside,
    )
    from .fragment_capping import cleave_and_cap

    # Cleave with acid_side_oh = True for ester/amide/glycosidic, False for ether
    acid_oh = bond_type != "ether"
    fragments = cleave_and_cap(mol, bonds, acid_side_oh=acid_oh)
    if not fragments or len(fragments) < 2:
        return None

    # Size guard: every fragment must be strictly smaller than parent
    parent_heavy = mol.GetNumHeavyAtoms()
    for frag in fragments:
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol and frag_mol.GetNumHeavyAtoms() >= parent_heavy:
            return None

    # Sort fragments smallest-first for cache warming
    def _frag_sort_key(frag):
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        return frag_mol.GetNumHeavyAtoms() if frag_mol else 999
    fragments.sort(key=_frag_sort_key)

    # Name each fragment (sugar intercept for glycosidic)
    #: use list-of-tuples to preserve identical SMILES duplicates
    #: partial assembly -- collect successful fragments, track failures
    import logging
    logger = logging.getLogger(__name__)

    named_fragments = []  # List[Tuple[Dict, str]] -- preserves all including duplicates
    failed_count = 0
    for frag in fragments:
        frag_name = None

        # Sugar intercept for glycosidic bonds
        if bond_type == "glycosidic" and frag["side"] == "acid":
            frag_name = _name_sugar_fragment(frag["smiles"])

        if not frag_name:
            frag_name = _name_fragment_with_fallback(frag["smiles"])

        if not frag_name or "unknown" in frag_name.lower():
            #: track failure but don't abort yet
            failed_count += 1
            logger.debug("Fragment naming failed for %s (side=%s)",
                         frag["smiles"], frag.get("side", "?"))
            continue

        #: Fragment naming size validation -- reject names that cover
        # less than 60% of the fragment's heavy atoms.
        # Bypass for retained core names (adenine, purine, etc.) which correctly
        # identify the key structural feature even when the fragment includes
        # additional atoms (e.g., adenosine-phosphate fragment named "adenine").
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol:
            frag_ha = frag_mol.GetNumHeavyAtoms()
            is_retained_core = frag_name.lower() in _RETAINED_CORE_NAMES
            if (frag_ha > 5
                    and not is_retained_core
                    and not _name_covers_molecule(frag_name, frag_mol)):
                logger.debug("Fragment name '%s' rejected: poor coverage for %s (%d HA)",
                             frag_name, frag["smiles"], frag_ha)
                failed_count += 1
                continue

        named_fragments.append((frag, frag_name))

    # a phase (fail-closed, invariants 1/9 -- never partial-ship): a
    # fragment that STILL could not be named or adequately covered even after
    # `_name_fragment_with_fallback`'s T4/general-engine rescue rung is a
    # genuinely unnameable residue. Shipping a name built from `named_fragments`
    # alone would describe a SMALLER molecule than the input -- exactly the
    # -suppressed-partial shape `phase2-shared-mechanism-trace.md`
    # identified (a complete, well-formed name for the WRONG, smaller
    # molecule). Decline the whole assembly instead of ever silently dropping
    # atoms; the molecule then either abstains honestly or a different
    # decomposition strategy still gets a chance upstream in `try_decompose`.
    if failed_count > 0:
        logger.info(
            "Partial assembly REFUSED: %d/%d fragments named (%d failed) -- "
            "declining rather than shipping a name for a smaller molecule",
            len(named_fragments), len(fragments), failed_count)
        return None

    #: require at least 2 named fragments for assembly
    if len(named_fragments) < 2:
        return None  # Not enough fragments to assemble

    #: the same class check as the single-bond split, for the functional-class
    # polyester (the polyamide assembler names an amine core with N-acyl prefixes)
    if bond_type == "ester" and _split_parent_is_junior(
            mol, "ester", {b.get("acid_atom") for b in bonds}):
        return None

    # Dispatch to bond-type-specific multi-fragment assembler
    if bond_type == "ester":
        result = _assemble_multi_ester(named_fragments, style)
    elif bond_type == "glycosidic":
        result = _assemble_multi_glycoside(named_fragments, style)
    elif bond_type == "amide":
        result = _assemble_multi_amide(named_fragments, style)
    else:
        return None

    if not result:
        return None

    # Quality checks on the assembled result
    if not _coverage_is_adequate(result, mol, bond_type=bond_type):
        return None

    return result


# ---------------------------------------------------------------------------
# Iterative mixed-type decomposition (a phase)
# ---------------------------------------------------------------------------

def _try_iterative_mixed_decompose(
    mol, bonds: List[Dict], style: str = "pin"
) -> Optional[str]:
    """Iteratively decompose large molecules with mixed bond types.

    After initial single-bond cleavage, re-scan each resulting fragment
    for additional cleavable bonds of DIFFERENT types. This handles
    phospholipid-type molecules (phosphodiester + ester bonds) and
    glycoside-ester hybrids.

    Algorithm:
    1. Initial cleavage with _select_best_bond
    2. For each fragment with HA > 30 that still contains cleavable bonds
       of a DIFFERENT type, cleave the best sub-bond
    3. Maximum MAX_DECOMP_LEVELS levels (iterative, not recursive)
    4. Quality gate comparison at each level

    Args:
        mol: RDKit Mol object.
        bonds: List of all cleavable bond dicts.
        style: Naming style.

    Returns:
        Assembled name string, or None if decomposition fails or is worse.
    """
    import logging

    from .bond_cleavage import find_cleavable_bonds
    from .fragment_capping import cleave_and_cap

    logger = logging.getLogger(__name__)

    if not bonds or len(bonds) < 2:
        return None

    # Need at least 2 different bond types for mixed decomposition
    bond_types_present = set(b["type"] for b in bonds)
    if len(bond_types_present) < 2:
        return None

    # Initial cleavage
    best_bond = _select_best_bond(mol, bonds)
    initial_frags = cleave_and_cap(mol, [best_bond], acid_side_oh=True)
    if not initial_frags or len(initial_frags) < 2:
        return None

    used_bond_types = {best_bond["type"]}

    # Track bond type metadata on fragments
    all_fragments = []
    for frag in initial_frags:
        frag['parent_bond_type'] = best_bond['type']
        all_fragments.append(frag)

    parent_heavy = mol.GetNumHeavyAtoms()

    # Iterative decomposition across levels (a phase: conditional levels)
    max_levels = _get_max_decomp_levels(mol)
    for level in range(max_levels - 1):  # Already did level 0
        new_fragments = []
        changed = False
        for frag in all_fragments:
            frag_mol = Chem.MolFromSmiles(frag["smiles"])
            if frag_mol is None:
                new_fragments.append(frag)
                continue
            frag_ha = frag_mol.GetNumHeavyAtoms()
            if frag_ha <= 30:
                new_fragments.append(frag)
                continue
            # Re-scan for cleavable bonds of DIFFERENT types
            sub_bonds = find_cleavable_bonds(frag_mol)
            sub_bonds = [b for b in sub_bonds if b["type"] not in used_bond_types]
            if not sub_bonds:
                new_fragments.append(frag)
                continue
            # Cleave the best sub-bond
            sub_best = _select_best_bond(frag_mol, sub_bonds)
            sub_frags = cleave_and_cap(frag_mol, [sub_best], acid_side_oh=True)
            if sub_frags and len(sub_frags) >= 2:
                # Size guard: sub-fragments must be smaller than original
                all_smaller = all(
                    (Chem.MolFromSmiles(sf["smiles"]) is None or
                     Chem.MolFromSmiles(sf["smiles"]).GetNumHeavyAtoms() < frag_ha)
                    for sf in sub_frags
                )
                if all_smaller:
                    # Tag new fragments with their bond type
                    for sf in sub_frags:
                        sf['parent_bond_type'] = sub_best['type']
                    new_fragments.extend(sub_frags)
                    used_bond_types.add(sub_best["type"])
                    changed = True
                else:
                    new_fragments.append(frag)
            else:
                new_fragments.append(frag)
        all_fragments = new_fragments
        if not changed:
            break

    # Name each fragment (a phase: fallback)
    # /Pitfall 5: use list-of-tuples to preserve identical SMILES duplicates
    #: partial assembly -- collect successful fragments, track failures
    named_fragments = []
    failed_count = 0
    for frag in all_fragments:
        frag_name = _name_sugar_fragment(frag["smiles"])
        if not frag_name:
            frag_name = _name_fragment_with_fallback(frag["smiles"])
        if not frag_name or "unknown" in frag_name.lower():
            #: track failure but don't abort yet
            failed_count += 1
            logger.debug("Mixed-decomp fragment naming failed for %s", frag["smiles"])
            continue

        #: Fragment naming size validation (with retained core name bypass)
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol:
            frag_ha = frag_mol.GetNumHeavyAtoms()
            is_retained_core = frag_name.lower() in _RETAINED_CORE_NAMES
            if (frag_ha > 5
                    and not is_retained_core
                    and not _name_covers_molecule(frag_name, frag_mol)):
                logger.debug("Mixed-decomp fragment name '%s' rejected: poor coverage for %s (%d HA)",
                             frag_name, frag["smiles"], frag_ha)
                failed_count += 1
                continue

        named_fragments.append((frag, frag_name))

    # a phase (fail-closed, invariants 1/9 -- never partial-ship): same
    # discipline as `_try_multi_bond_decompose` above -- a fragment that still
    # could not be named/covered after the rescue rung is genuinely
    # unnameable, and shipping a name for `named_fragments` alone would
    # describe a SMALLER molecule than the input (the exact
    # "Mixed-decomp partial assembly" shape `phase2-shared-mechanism-trace.md`
    # traced live: GPI-mannoside -> 3/4 named -> -suppressed). Decline
    # rather than silently drop atoms.
    if failed_count > 0:
        logger.info(
            "Mixed-decomp partial assembly REFUSED: %d/%d fragments named "
            "(%d failed) -- declining rather than shipping a name for a "
            "smaller molecule",
            len(named_fragments), len(all_fragments), failed_count)
        return None

    #: require at least 2 named fragments for assembly
    if len(named_fragments) < 2:
        return None  # Not enough fragments to assemble

    # Bond-type-aware assembly : route fragment pairs through
    # bond-type-specific assemblers instead of naive space-join
    from .fragment_assembly import _assemble_by_bond_type

    assembled = _assemble_by_bond_type(named_fragments, used_bond_types, style)

    # Task 3 (2026-09-25): no space-join fallback. Both fallbacks that used to
    # sit here blank-joined the fragment NAMES when the bond-type assembler
    # declined or lost a fragment. That string names the fragments, not the
    # connected input -- for the GPI mannoside, 4 names glued into one string
    # that OPSIN reads as C38H75N2O31PS (input C38H71N2O28PS; TRIAGE T3). Only
    # held it back; with that gate off the same class of glue reached
    # the public output (its phosphodiester sibling, TRIAGE " producer
    # sites"). Declining here voids this candidate only: `try_decompose` keeps
    # its other strategies, and the molecule goes on to the general engine,
    # which names the GPI mannoside RT-exact at the best-effort tier.
    if not assembled:
        return None

    # Token validation : reject if assembly lost a fragment
    fragment_names = [name for _, name in named_fragments]
    if not _validate_assembly_tokens(assembled, fragment_names):
        return None  # Assembly lost a fragment -- reject

    # Quality gate: the assembled name must be acceptable
    if not _name_quality_is_acceptable(assembled, mol):
        return None

    # Coverage gate
    if not _coverage_is_adequate(assembled, mol, bond_type=best_bond["type"]):
        return None

    return assembled


# ---------------------------------------------------------------------------
# Main decomposition entry point
# ---------------------------------------------------------------------------

def try_decompose(mol, style: str = "pin") -> Optional[str]:
    """Attempt decomposition of a molecule into named fragments.

    This is the main entry point for the decomposition engine. It:
    1. Finds cleavable bonds (ester, amide, phosphodiester, thioester,
       glycosidic, sulfonamide, carbamate, ether -- 8 types)
    2. Checks if the existing pipeline name is acceptable (quality gate)
    3. Selects the best bond to cleave
    4. Cleaves and caps the fragments
    5. Names each fragment recursively
    6. Assembles the multi-component IUPAC name

    Returns None if:
    - No cleavable bonds exist
    - The existing pipeline name is good enough (quality gate passes)
    - Fragment naming fails
    - Size guard detects non-shrinking fragments

    Args:
        mol: RDKit Mol object to decompose.
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Multi-component IUPAC name string, or None to fall through
        to the existing naming pipeline.
    """
    from .bond_cleavage import find_cleavable_bonds

    # Step 1: Find cleavable bonds
    bonds = find_cleavable_bonds(mol)
    if not bonds:
        return None  # No cleavable bonds, fall through

    # Step 1b: Performance guard -- skip if too many cleavable bonds
    if len(bonds) > MAX_CLEAVABLE_BONDS:
        return None  # Performance guard: too complex for decomposition

    # Step 2: Get existing pipeline name via systematic-only path.
    # name_pipeline_only skips decomposition, so it cannot recurse back
    # into try_decompose. No cache isolation needed.
    # If this SMILES is already in the visited set (being named up the
    # call stack), skip the probe — the caller already determined the
    # assembled name was inadequate, so proceed directly to decomposition.
    from ..assembly.fragment_naming import _get_visited
    from ..namer import name_pipeline_only

    existing_smiles = Chem.MolToSmiles(mol)
    visited = _get_visited()
    if existing_smiles in visited:
        # Already being named up the call stack — cycle detected.
        # Return None to let the caller's assembly pipeline handle naming
        # instead of decomposing (which would produce garbled results).
        return None
    existing_name = name_pipeline_only(existing_smiles, style=style)

    # Step 3: Quality gate -- only decompose if existing name is poor
    quality_ok = existing_name and _name_quality_is_acceptable(existing_name, mol)

    # Step 3b: Glycoside bypass -- if a glycosidic bond leads to a known sugar,
    # decomposition will produce a better name (retained sugar name vs systematic
    # oxane/tetrahydropyran). Only bypass when sugar lookup would succeed.
    glycoside_bypass = False
    if quality_ok:
        glycosidic_bonds = [b for b in bonds if b.get("type") == "glycosidic"]
        if glycosidic_bonds:
            from .fragment_capping import cleave_and_cap as _probe_cleave
            probe_bond = _select_best_bond(mol, glycosidic_bonds)
            probe_frags = _probe_cleave(mol, [probe_bond], acid_side_oh=True)
            if probe_frags:
                for pf in probe_frags:
                    if pf["side"] == "acid" and _name_sugar_fragment(pf["smiles"]):
                        glycoside_bypass = True
                        break

    if quality_ok and not glycoside_bypass:
        # a phase Step 2 (the assembly WEAVER,
        # internal notes): the length/token
        # heuristic `_name_quality_is_acceptable` uses to decide "good
        # enough, skip decomposition" can be FOOLED by a wrong whole-molecule
        # PARENT choice that happens to be long enough to look complete --
        # measured on the phosphatidylcholine-family -suppressed
        # cluster: `name_pipeline_only` picks the cut choline nitrogen as the
        # whole molecule's own '...aminium' PARENT (ignoring the glycerol
        # backbone and both fatty/phospho arms entirely), and
        # "N,N,N-trimethylethanaminium" (28 chars) clears every length check
        # despite covering ~7 of the molecule's 31 heavy atoms. Give the
        # weaver's own atom-complete-or-abstain oracle (Part C,
        # `weave.weave_is_verified` -- a full OPSIN round-trip, not a
        # heuristic) one cheap chance to override the heuristic's accept,
        # but ONLY when (a) >= 2 cleavable bonds exist (the composer's own
        # star-topology precondition; a single-linkage molecule is
        # untouched) and (b) `existing_name` does NOT itself independently
        # round-trip (so an existing CORRECT short/retained name is never
        # second-guessed -- zero regression risk on every molecule this
        # heuristic already gets right).
        if len(bonds) >= 2 and not _weave_result_is_verified(mol, existing_name):
            try:
                from .weave import try_weave, weave_is_verified
                _weave_early = try_weave(mol, style)
                if _weave_early and weave_is_verified(mol, _weave_early):
                    return _weave_early
            except Exception:
                pass
        return None  # Existing name is good enough

    # Step 4: Choose ONE bond to cleave (the most significant one)
    best_bond = _select_best_bond(mol, bonds)

    # Steps 4-8: Single-bond attempt via helper
    single_result = _try_single_bond_decompose(mol, best_bond, style)

    # Coverage gate: reject decomposition results that don't cover enough
    # of the molecule's heavy atoms (applied only to decomposition output).
    if single_result and not _coverage_is_adequate(single_result, mol, bond_type=best_bond["type"]):
        single_result = None  # Coverage inadequate, discard this result

    #: single-bond path returns result directly (backward compat).
    # Quality comparison: if decomposition produced a worse name than the
    # existing pipeline (garbled tokens, bracket mismatches), prefer the
    # existing name to avoid replacing a parseable name with garbage.
    if len(bonds) == 1:
        if single_result and existing_name:
            if _decomposition_is_worse(single_result, existing_name, mol):
                return existing_name
        return single_result

    # If single-bond result is adequate and not worse than existing, return it
    if single_result and _name_quality_is_acceptable(single_result, mol):
        if not (existing_name and _decomposition_is_worse(single_result, existing_name, mol)):
            return single_result

    # MULTI-BOND RETRY : try alternative bonds
    tried_indices = {best_bond["bond_idx"]}
    for bond in bonds:
        if bond["bond_idx"] in tried_indices:
            continue
        if len(tried_indices) >= MAX_BOND_RETRY_ATTEMPTS:
            break
        if _exploration_exhausted():
            break
        tried_indices.add(bond["bond_idx"])
        alt_result = _try_single_bond_decompose(mol, bond, style)
        # Coverage gate on retry results
        if alt_result and not _coverage_is_adequate(alt_result, mol, bond_type=bond["type"]):
            continue
        if alt_result and _name_quality_is_acceptable(alt_result, mol):
            # Also check that the alternative is not worse than existing name
            if existing_name and _decomposition_is_worse(alt_result, existing_name, mol):
                continue
            return alt_result

    # a phase: Iterative mixed-type decomposition.
    # If single-bond retry produced no acceptable result AND the molecule has
    # cleavable bonds of multiple types, try iterative decomposition.
    if not single_result or not _name_quality_is_acceptable(single_result, mol):
        bond_types_present = set(b["type"] for b in bonds)
        if (len(bond_types_present) >= 2 and mol.GetNumHeavyAtoms() > 30
                and not _exploration_exhausted()):
            iterative_result = _try_iterative_mixed_decompose(mol, bonds, style)
            if iterative_result:
                if not (existing_name and _decomposition_is_worse(iterative_result, existing_name, mol)):
                    return iterative_result

    # MULTI-BOND SAME-TYPE cleavage (a phase): if N+ bonds of the same
    # type exist, try cleaving all same-type bonds simultaneously
    # (IUPAC polyesters / triglycerides, glycosides).
    # Bond-type-specific thresholds (a phase-04):
    # - glycosidic: 2 (disaccharide + aglycone)
    # - ester/amide: 3 (2-bond molecules better handled by single-bond)
    # Only attempt when single-bond produced no result at all.
    from collections import Counter as _BondCounter
    bond_type_counts = _BondCounter(b["type"] for b in bonds)
    if not single_result and not _exploration_exhausted():
        for bond_type_key, count in bond_type_counts.most_common():
            threshold = _MULTI_BOND_THRESHOLD.get(bond_type_key, 99)
            if count >= threshold:
                same_type_bonds = [b for b in bonds if b["type"] == bond_type_key]
                multi_result = _try_multi_bond_decompose(mol, same_type_bonds, style)
                if multi_result:
                    if not (existing_name and _decomposition_is_worse(multi_result, existing_name, mol)):
                        return multi_result

    # a phase: Mixed-bond threshold relaxation.
    # When total cleavable bonds >= 3 across all types and no single type
    # reached its threshold above, relax the ester threshold to 2 for
    # mixed-type molecules (e.g., 2 esters + 1 glycosidic).
    # Only for ester type (glycosidic already at 2, amide stays at 3).
    # Guard: only when single-bond produced no acceptable result.
    if not single_result or not _name_quality_is_acceptable(single_result, mol):
        total_cleavable = sum(bond_type_counts.values())
        ester_count = bond_type_counts.get("ester", 0)
        if (total_cleavable >= 3 and ester_count >= 2 and len(bond_type_counts) >= 2
                and not _exploration_exhausted()):
            # Only apply relaxation if standard threshold was NOT reached
            if ester_count < _MULTI_BOND_THRESHOLD.get("ester", 99):
                ester_bonds = [b for b in bonds if b["type"] == "ester"]
                multi_result = _try_multi_bond_decompose(mol, ester_bonds, style)
                if multi_result:
                    if not (existing_name and _decomposition_is_worse(multi_result, existing_name, mol)):
                        return multi_result

    # Ester-specific fallback: when single_result exists but failed quality,
    # multi-bond ester may produce a better name by exposing the clean core
    # (e.g., removing 3 peripheral acetyloxy groups from a tetracyclic ring).
    if single_result and not _name_quality_is_acceptable(single_result, mol):
        ester_count = bond_type_counts.get("ester", 0)
        if ester_count >= _MULTI_BOND_THRESHOLD.get("ester", 99):
            ester_bonds = [b for b in bonds if b["type"] == "ester"]
            multi_result = _try_multi_bond_decompose(mol, ester_bonds, style)
            if multi_result and _name_quality_is_acceptable(multi_result, mol):
                if not (existing_name and _decomposition_is_worse(multi_result, existing_name, mol)):
                    return multi_result

    # a phase Step 2 -- the assembly WEAVER
    # (internal notes): a core-and-arms composer
    # for star-topology multi-linkage molecules (glycerophospholipids and
    # relatives) that none of the flat assemblers above could weave into ONE
    # connected name (each fragment named fine on its own, but the suffix-
    # string role converters have nothing to match on a T4-rescued/seniority-
    # demoted fragment, so the molecule either space-joins -- sees
    # disconnected OPSIN components -- or silently drops the fragment).
    #
    # Gating (PIN byte-identity + 0-wrong, a project rule): reached only here,
    # after every earlier attempt in this function has already had its
    # chance to return. The decision to PREFER the weave candidate over
    # whatever `single_result` already holds is never a heuristic quality
    # check -- it is a full OPSIN round-trip on BOTH candidates. So:
    # - a `single_result` that is ALREADY correct (round-trips) is kept
    # UNCHANGED -- the weaver is not even consulted for a preference,
    # zero regression risk on any molecule this function already names
    # correctly.
    # - only when `single_result` does NOT round-trip (None, or the exact
    # "complete name for a smaller/different molecule" shape the trace
    # traced) does a weave candidate get a chance, and even then ONLY if
    # IT independently round-trips (`weave_is_verified`, Part C -- the
    # atom-complete-or-abstain guard). A weave candidate that fails to
    # verify is discarded; the function falls through unchanged.
    # Symmetric with site 1 (above): protect a round-tripping `existing_name`
    # from being replaced too, not just `single_result` -- otherwise a
    # correct-but-heuristically-"too short" existing_name could be swapped
    # for a DIFFERENT-but-equally-correct weave name (a byte-level change,
    # never a wrong molecule, but still an unverified override).
    if (not (single_result and _weave_result_is_verified(mol, single_result))
            and not (existing_name and _weave_result_is_verified(mol, existing_name))):
        try:
            from .weave import try_weave, weave_is_verified
            weave_result = try_weave(mol, style)
            if weave_result and weave_is_verified(mol, weave_result):
                return weave_result
        except Exception:
            pass  # any failure here is a decline -- fall through unchanged

    # Compare decomposition result against existing pipeline name:
    # if decomposition produced a worse name (garbled, bracket-mismatched,
    # or containing malformed tokens), fall back to existing pipeline name.
    if single_result and existing_name:
        if _decomposition_is_worse(single_result, existing_name, mol):
            return None  # Let existing pipeline name be used

    return single_result  # Best effort fallback


def _weave_result_is_verified(mol, name: str) -> bool:
    """True iff *name* (an EXISTING `single_result`/multi-bond candidate,
    not a weave candidate) independently round-trips to *mol* -- reuses the
    weaver's own Part-C oracle so the "is the existing result already good
    enough" test and the "is the weave candidate good enough" test are the
    SAME predicate, not two different notions of correctness."""
    try:
        from .weave import weave_is_verified
        return weave_is_verified(mol, name)
    except Exception:
        return False
