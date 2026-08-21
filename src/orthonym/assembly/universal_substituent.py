"""Phase B.2 -- the unconditional recursive substitutive namer core.

``name_universal_substitutive(mol)`` is a NEW, self-contained T4/best-effort
producer: it NEVER declines a hard branch.  Where the existing recursive
substituent namer (``assembly/substituent_enumerator.py``) hits an unnameable
fragment and does ``return None`` / emits the ``'substituent'`` sentinel --
which fails the WHOLE enclosing candidate closed -- this module re-enters
ITSELF on the branch subgraph and always bottoms out at a valid (possibly
ugly, non-PIN) systematic token: a simple leaf prefix, a von-Baeyer/'a'-
replacement skeleton, or a deeper recursive call.  There is no depth cap.

Architecture (``name_subgraph``, NOT copied --
see ``):

    name(component) = parent(spine) + Sum(name(branch_i) rendered as -yl)

Termination (the shrinking-atom-set argument, verbatim from that research):
every recursive call is handed a component that is a PROPER SUBSET of its
caller's component -- the caller's spine is non-empty (>=1 atom) and is
removed before any branch is computed, and branch components are pairwise
disjoint (each BFS is bounded by a monotonically GROWING exclude set built
from already-claimed atoms).  So along any recursion path the component size
strictly decreases every call; recursion cannot be infinite.  Neither
reference bounds the WORK this can cost on a pathological input (has no budget at all; mitigation is a depth>=1 local decline, exactly
the failure mode this module must NOT reproduce), so this module adds an
explicit in-algorithm atom/node WORK BUDGET (``_Budget``, charged by
component size at every call) that fails CLOSED -- returns ``None`` -- when
exceeded.  This is a counter, not a signal/timeout: it can never hang.

Coverage-by-construction (the 0-wrong carry): every ``_ComponentResult``
carries ``covers`` -- the exact set of heavy-atom indices it accounts for --
and the top-level entry point asserts the union of the flat binding list
equals the FULL heavy-atom set of the input before returning anything.  Any
gap voids the candidate (``None``), never a partial name.

Scope of THIS module (be honest about what is built vs. deferred, per the
task brief): parent selection covers acyclic chains (own longest-path/tree
walk), single rings (own minimal cyclo-'a'-replacement numbering) and
polycyclic ring systems (reusing ``rules.vonbaeyer_universal.
analyze_cage_universal`` + ``rules.polycyclic._build_parent_with_unsaturation``
for the parent TEXT only -- NOT ``general_engine``'s substituent-recursion
tail, which is exactly the declining machinery this module replaces).
Isotopes, radicals, wildcard atoms and multi-fragment inputs are OUT OF
SCOPE and void the whole call (``None``).  Indicated hydrogen is also OUT OF
SCOPE for this task (deferred -- no witness in the B2b brief requires it) and
voids nothing on its own: this module simply does not special-case it, which
is a pre-existing gap (unrelated to charge), not something this task
introduces or worsens.

Task B2b (this round) LIFTS the blanket per-atom-charge void and replaces it
with real charge PERCEPTION, reusing ``assembly.general_engine``'s existing
charge-suffix primitives (``_charge_suffix_text`` / ``_zwitterion_suffix_plan``
/ ``_elide_before_ionic_suffix``) so charge is spelled the SAME way as the
rest of the codebase, never reinvented. This is backstopped by a general
RAW-formal-charge void guard (fix round 1 below) so that a charge the reused
primitives cannot spell always VOIDS the whole call rather than being spelled
wrong. Scope, precisely (fail-closed on everything else -- an unspellable
charge VOIDS the whole call, never mis-names):

* A GENUINE ionic centre (``perception.ions.get_ion_sites``, which already
  excludes P-59 INTERNAL charges -- nitro, N-oxide, azide, diazo -- these are
  not "ionic centers" needing a suffix) is only ever resolved when it sits on
  the SPINE atoms of *some* level of the recursion (top-level parent or a
  substituent branch's own local parent) -- never as a bare prefix. Every
  leaf-shortcut (``_leaf_shortcut``) BAILS on any component containing a
  genuinely-charged atom (other than the dedicated nitro shape below), so a
  charged atom always falls through to the generic spine machinery, which
  forces it onto SOME level's spine (the branch-discovery/forced-root
  argument: a component shrinks every recursive call, and a size-1 component
  always makes its own sole atom the spine).
* Net-charged, single-sign species (a plain cation or anion, e.g. a
  quaternary ammonium or a skeletal carbanion): ``general_engine.
  _charge_suffix_text`` on whichever spine holds the site(s) -- SAME scope as
  general_engine's own P5 layer, so e.g. an FG anion (carboxylate, alkoxide,
  phenolate, sulfonate, ...) or an FG cation (diazonium, acylium) is OUT OF
  SCOPE and VOIDS (general_engine's own charged-species PIN path owns those,
  not this general/best-effort producer).
* Net-0 zwitterions with BOTH ionic centres SKELETAL to the SAME spine
  (P-74.1.1, ``general_engine._zwitterion_suffix_plan(..., allow_fg_anion=
  False)``): resolved the same way. The FG-anchored "-olate" branch
  (P-74.1.2, an anion hanging off a characteristic group not itself on the
  spine) is explicitly DISABLED (``allow_fg_anion=False``) -- this producer
  has no functional-group-suffix layer to hold an atom out of substituent
  discovery for it, so it is deferred, not built. A carboxylate-anchored
  zwitterion (amino-acid style, e.g. glycine) therefore VOIDS -- carboxylate
  is out of scope for BOTH the skeletal and the -olate branch of the reused
  primitive (it is owned by the v33 charged-lever's dedicated ester-anion-word
  mechanism, a different, specialized producer this module does not invoke).
* Nitro (``C-N(+)(=O)[O-]``, P-59 internal charge) is spelled directly as a
  dedicated, charge-and-bond-order-VALIDATED leaf shortcut (``_nitro_shortcut``)
  exactly like ``carboxy``/``cyano`` -- never via the genuine-ion machinery
  above. Its nitrogen is also excluded from ordinary chain-spine CONTINUATION
  (``_is_nitro_root`` in ``_tree_neighbors``) so it can never be mis-threaded
  into a replacement-nomenclature chain as if it were an ordinary
  (uncharged, standard-valence) heteroatom -- that would silently drop the
  charge information a plain "aza" locant carries no trace of.
* Any OTHER shape a genuine ionic centre can take (multiply-charged, mixed
  cation+anion under a nonzero net charge, more than one cation/anion pair,
  an ionic centre split across two different spine levels, ...) is out of
  scope for the reused primitives and VOIDS the whole call -- a final
  top-level assertion (mirroring the atom-coverage assertion) checks that
  EVERY genuine ionic-centre atom index was actually resolved by some level's
  suffix, not merely trusted to be, before returning a name.

Fix round 1 (task-review + FABLE adversarial, both on ``dc96929f`` -- see
``.superpowers/sdd/2026-08-21-no-abstain-universal-namer/
task-B2b-fixround1-findings.md``): the four bullets above governed only the
GENUINE ionic centres ``get_ion_sites`` reports. But ``get_ion_sites``
STRIPS P-59 INTERNAL and P-74.2.1 semipolar charges (N-oxide, azide, diazo,
nitrone, nitrile oxide, nitronate, aci-nitro, nitrate ester, thionitro, and
the charge-drawn S/P-oxides), so those charged atoms were invisible to BOTH
``_resolve_spine_charge`` and the charge-coverage assertion -- while the
spine builders thread atoms by ELEMENT SYMBOL ONLY, with no formal-charge
check, and happily absorbed such an atom into the skeleton AS IF NEUTRAL,
yielding a coverage-complete name of a DIFFERENT molecule (all measured
mis-naming on ``dc96929f``; nitro was the sole exception, kept out of the
chain via ``_is_nitro_root``, but its O atoms could still be shredded as a
BFS seed / branch root, so polynitro mis-named too). The fix is ONE general
RAW-formal-charge void guard at the top-level entry point: VOID the whole
result whenever any atom carries a nonzero raw formal charge that was NOT
consumed by (a) the genuine-ion suffix machinery
(``cation_sites | anion_sites``, asserted resolved) or (b) a nitro group
actually rendered by ``_nitro_shortcut`` (tracked as
``_ComponentResult.nitro_atoms``). ``get_ion_sites`` lists every
nonzero-charge atom then removes the internal set, so
``cation_sites | anion_sites`` is EXACTLY the non-internal raw-charged atoms
-- the guard therefore fires precisely on the internal charges, closing the
whole class (N-oxide/azide/diazo/nitrone/... and polynitro shredding) in one
set-membership check WITHOUT building any of their nomenclature (they
correctly VOID = abstain; the floor degrades, never mis-names).

This module is a PURE PRODUCER: it never calls ``verify_or_none`` and never
ships a name; the caller (a later wiring task) is responsible for gating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from rdkit import Chem

from ..perception.stereo import assign_stereochemistry
from ..perception.rings import get_ring_systems
from ..perception.ions import get_ion_sites
from .naming_utils import (
    SIMPLE_MULTIPLIERS,
    alpha_sort_key,
    format_substituent_prefix,
)
from ..rules.skeletal_replacement import REPLACEMENT_TERMS, _A_CITATION_ORDER
from ..rules.vonbaeyer_universal import analyze_cage_universal
from ..rules.polycyclic import _build_parent_with_unsaturation
# Task B2b: reuse general_engine's own charge-suffix primitives so this
# module spells charge the SAME way as the rest of the codebase (never
# reinvented) -- see the module docstring's charge-scope paragraph.
from .general_engine import (
    _charge_suffix_text,
    _zwitterion_suffix_plan,
    _elide_before_ionic_suffix,
)


# ---------------------------------------------------------------------------
# Work budget -- mandatory in-algorithm backstop (NOT a signal/timeout).
# ---------------------------------------------------------------------------

#: Generous for any real organic molecule (dev500's largest is well under a
#: few hundred heavy atoms with modest branch nesting); a pathological giant
#: with deep alternating branch nesting re-charges the same atoms once per
#: ancestor call, so this trips in bounded, FAST (no I/O, no subprocess)
#: Python work long before it would hang. This is the CALLER-TUNABLE
#: cumulative recursive-work budget -- it is NOT, by itself, a safety
#: ceiling on raw atom count; see ``_MAX_ATOMS_FOR_PERCEPTION`` below.
DEFAULT_ATOM_WORK_BUDGET = 20_000

#: Fix round 1 follow-up (found while VERIFYING finding 2's fix -- the broad
#: except alone was not enough): a raw linear chain segfaults inside
#: ``assign_stereochemistry`` (CIP; the vendored ``centres`` bridge, gated
#: by ``ORTHONYM_USE_CENTRES_CIP``, default on) somewhere between 18,000 and
#: 19,999 heavy atoms on a SINGLE call. A process crash is NOT a Python
#: exception -- it cannot be caught by ANY try/except, including the broad
#: one this fix round added. Worse, measured: this is NOT purely a function
#: of single-call size -- calling this module TWICE in one process on
#: moderately large chains (3,000, then another 3,000; separately, 2,500
#: then another 2,500) segfaults on the SECOND call even though the FIRST
#: succeeds cleanly, consistent with state accumulating in the CIP bridge
#: across calls (a realistic deployment shape: one process names many
#: molecules in a batch/eval loop). Stress-tested instead of merely spot-
#: checked: 40 repeated calls at 2,000 atoms each all survived; repeated
#: calls at 2,500 crashed on the second. A caller may legitimately want a
#: large ``atom_work_budget`` (for generous cumulative RECURSIVE-work
#: allowance across many small branches -- entirely orthogonal to raw input
#: size), so this ceiling is a SEPARATE, non-overridable hard cap on the raw
#: heavy-atom count this module will even attempt to run RDKit perception
#: (kekulize/CIP/ring-systems/canonical-ranking) on, enforced regardless of
#: whatever ``atom_work_budget`` value is passed in. Deliberately
#: conservative, not tuned to the exact boundary, which is expected to
#: depend on process memory/stack state and so should not be treated as a
#: fixed constant.
_MAX_ATOMS_FOR_PERCEPTION = 2_000


class _BudgetExceeded(Exception):
    """Raised internally when the work budget is exhausted; converted to a
    clean ``None`` at the top-level entry point. Never escapes this module."""


@dataclass
class _Budget:
    remaining: int

    def charge(self, n: int) -> None:
        self.remaining -= n
        if self.remaining < 0:
            raise _BudgetExceeded()


@dataclass
class _Ctx:
    mol: "Chem.Mol"
    ring_systems: List[FrozenSet[int]]
    ring_system_of: Dict[int, int]
    budget: _Budget
    canon_rank: Tuple[int, ...]  # atom idx -> canonical rank (numbering-invariant)
    # Task B2b: GENUINE ionic centres only (get_ion_sites already excludes
    # P-59 internal charges -- nitro, N-oxide, azide, diazo). Computed ONCE
    # per top-level call and consulted by every recursive ``_name_component``
    # call to decide whether ITS OWN spine needs a charge suffix.
    cation_sites: List[Dict[str, object]]
    anion_sites: List[Dict[str, object]]


@dataclass(frozen=True)
class UniversalResult:
    """Public result: a name string + the atom->token coverage binding."""

    name: str
    bindings: Tuple[Tuple[str, FrozenSet[int]], ...]
    covers: FrozenSet[int]


@dataclass
class _ComponentResult:
    """Internal: one recursively-named component (parent OR substituent)."""

    name: str                                    # assembled name, NOT yet -yl
    bindings: List[Tuple[str, FrozenSet[int]]]    # flat, covers `covers`
    covers: FrozenSet[int]
    attach_locant: Optional[int]                  # this component's own spine
    #                                              locant of its attach atom,
    #                                              or None if not applicable
    #                                              (top-level call).
    is_prefix_ready: bool = False                 # True for a leaf shortcut
    #                                              (e.g. "carboxy", "oxo"):
    #                                              already a complete
    #                                              substituent prefix --
    #                                              _render_as_substituent
    #                                              must NOT mechanically
    #                                              append -yl/-ylidene to it.
    charged: FrozenSet[int] = frozenset()         # Task B2b: genuine
    #                                              ionic-centre atom indices
    #                                              (see _Ctx.cation_sites /
    #                                              .anion_sites) that were
    #                                              successfully expressed by
    #                                              a charge suffix SOMEWHERE
    #                                              within this component
    #                                              (this level's own spine,
    #                                              unioned with every child
    #                                              branch's own `charged`).
    #                                              The top-level entry point
    #                                              asserts this equals the
    #                                              FULL genuine-ion-site set
    #                                              before shipping a name --
    #                                              symmetric to the atom-
    #                                              coverage assertion, but
    #                                              for CHARGE correctness.
    nitro_atoms: FrozenSet[int] = frozenset()     # Task B2b fix round 1:
    #                                              atom indices (N + both its
    #                                              O) that were RENDERED as a
    #                                              nitro group via
    #                                              ``_nitro_shortcut``
    #                                              SOMEWHERE within this
    #                                              component (this level's own
    #                                              leaf, unioned with every
    #                                              child branch's own
    #                                              `nitro_atoms`). Threaded --
    #                                              not re-derived by string-
    #                                              matching the "nitro" token
    #                                              at the top level -- because
    #                                              a nitro nested inside a
    #                                              rendered ``-yl`` substituent
    #                                              is folded into that
    #                                              substituent's token and
    #                                              would be invisible to such a
    #                                              scan. Consumed by the
    #                                              top-level RAW-formal-charge
    #                                              void guard: nitro's P-59
    #                                              INTERNAL charges are the one
    #                                              internal-charge shape this
    #                                              module can actually spell,
    #                                              so they are the sole
    #                                              exception (b) to that guard.


# ===========================================================================
# Public entry point
# ===========================================================================

def name_universal_substitutive(
    mol, atom_work_budget: int = DEFAULT_ATOM_WORK_BUDGET,
) -> Optional[UniversalResult]:
    """Name *mol* unconditionally, or return ``None`` (void -- never partial).

    ``None`` happens for exactly six reasons, all fail-closed:
      1. a scope guard (isotopes / radicals / wildcard atoms / multi-fragment
         input) -- explicitly out of scope for this task, never silently
         mis-named;
      2. the work budget trips on a pathological input;
      3. the atom-coverage assertion finds a gap (should not happen given
         the construction, but is asserted rather than trusted);
      4. a genuine ionic centre (see module docstring's charge-scope
         paragraph, Task B2b) that cannot be faithfully expressed by the
         reused ``general_engine`` charge-suffix primitives on any level's
         spine -- an FG anion/cation, a mixed-sign net-charged species, a
         multiply-charged centre, or an ionic centre split across two
         different spine levels are all out of scope and void rather than
         ship a name that omits the charge (the top-level charge-coverage
         assertion, symmetric to the atom-coverage one, catches this even if
         a lower-level construction bug would otherwise have missed it);
      5. a P-59 INTERNAL / P-74.2.1 semipolar charge (N-oxide, azide, diazo,
         nitrone, nitrile oxide, nitronate, aci-nitro, nitrate ester,
         thionitro, charge-drawn S/P-oxide, or a shredded polynitro) --
         ``get_ion_sites`` strips these, so they never reach reason 4's
         machinery; the top-level RAW-formal-charge void guard (fix round 1;
         see module docstring's fix-round-1 paragraph) voids any atom whose
         nonzero raw formal charge was NOT consumed by the genuine-ion suffix
         machinery or a rendered ``_nitro_shortcut``, so an internal charge
         this module cannot spell VOIDS rather than being absorbed into the
         skeleton as if neutral (a coverage-complete name of a DIFFERENT
         molecule -- the whole defect this fix round closes);
      6. ANY other unexpected exception (fix round 1, finding 2). This is a
         pure ``Optional``-contracted producer: it must never raise. Measured
         holes this closes: a spine longer than 9,999 atoms reaches
         ``data/chain_names.py``'s "chain length outside supported range
         (1-9999)" ``ValueError`` via ``_build_parent_with_unsaturation``
         (now moot in practice below ``_MAX_ATOMS_FOR_PERCEPTION``, but kept
         as defense in depth rather than relying on that cap alone), and a
         caller who passes a huge ``atom_work_budget`` re-enables an
         unbounded Python call-stack depth (``RecursionError``) that the
         normal (default-budget) recursion depth (~200, arithmetically safe
         against Python's ~1000 default limit) would never reach. Caught
         here rather than narrowly patched at each raise site, because a
         pure producer's contract is "never raise", not "never raise for the
         causes I've already found". NOTE: this broad catch cannot help
         against a process-level crash (segfault) -- see
         ``_MAX_ATOMS_FOR_PERCEPTION`` for that measured, separate hole.
    """
    if mol is None:
        return None
    try:
        return _name_universal_substitutive_unsafe(mol, atom_work_budget)
    except Exception:
        return None


def _name_universal_substitutive_unsafe(
    mol, atom_work_budget: int,
) -> Optional[UniversalResult]:
    """The real body -- may raise; ``name_universal_substitutive`` is the
    only caller and converts every exception to ``None``."""
    # Cheap size guard BEFORE any perception work. This is not redundant with
    # the per-call recursive charging below: kekulization, CIP assignment and
    # ring-system perception all run UNCONDITIONALLY on the whole molecule,
    # before the first recursive call/charge ever happens, and are NOT
    # guaranteed safe at arbitrary size -- measured: a 25,000-atom linear
    # chain segfaults inside that pipeline, and (fix round 1 follow-up) so
    # does a ~20,000-atom one, specifically inside CIP assignment. A process
    # crash is NOT a Python exception, so no try/except (including the broad
    # one added this round) can catch it -- the cap below is therefore
    # ``min(atom_work_budget, _MAX_ATOMS_FOR_PERCEPTION)``, never just
    # ``atom_work_budget`` alone, so a caller cannot re-enable the crash by
    # passing a larger budget (that budget still legitimately raises the
    # RECURSIVE-work ceiling; it cannot raise the raw-size safety ceiling).
    built = _build_ctx(mol, atom_work_budget)
    if built is None:
        return None
    ctx, heavy = built

    comp = _name_component(ctx, heavy, attach_hint=None, is_top=True)
    if comp is None:
        return None

    # Charge-coverage assertion (Task B2b), symmetric to the atom-coverage
    # one below: every GENUINE ionic-centre atom must have been resolved by
    # a charge suffix SOMEWHERE in the recursion. By construction this
    # always holds (every genuine ion atom ends up on some level's own
    # spine -- see the module docstring's inductive argument -- and that
    # level's own resolution either succeeds or voids the whole call
    # already), but it is ASSERTED rather than trusted, exactly like the
    # atom-coverage check a few lines down.
    all_charge_ids = frozenset(s['atom_idx'] for s in ctx.cation_sites) | \
        frozenset(s['atom_idx'] for s in ctx.anion_sites)
    if comp.charged != all_charge_ids:
        return None  # void: a genuine ionic centre was not expressed

    # Task B2b fix round 1: the RAW-formal-charge void guard -- the one
    # general check that closes the whole internal-charge mis-naming class.
    # ``get_ion_sites`` (which ``cation_sites``/``anion_sites`` derive from)
    # STRIPS P-59 INTERNAL charges (N-oxide, azide, diazo, nitrone, nitrile
    # oxide, nitronate, aci-nitro, nitrate ester, thionitro, and the P-74.2.1
    # semipolar / charge-drawn S/P-oxides), so those charged atoms are
    # invisible to BOTH ``_resolve_spine_charge`` AND the charge-coverage
    # assertion just above -- while the spine builders thread atoms by ELEMENT
    # SYMBOL ONLY, with no formal-charge check, and would happily absorb such
    # an atom into the skeleton AS IF NEUTRAL, yielding a coverage-complete
    # name of a DIFFERENT molecule. ``all_charge_ids`` is EXACTLY the set of
    # raw-charged atoms that are NOT internal (get_ion_sites lists every
    # nonzero-charge atom, then removes the internal set), so the internal
    # atoms are precisely ``{raw-charged} - all_charge_ids``. VOID whenever
    # any atom carries a nonzero raw formal charge that was NOT consumed by
    # (a) the genuine-ion suffix machinery (``all_charge_ids``, asserted
    # resolved above) or (b) a nitro group actually rendered by
    # ``_nitro_shortcut`` (``comp.nitro_atoms`` -- the one internal-charge
    # shape this module can spell). This also closes polynitro shredding: a
    # torn-apart nitro is never rendered as a nitro leaf, so its O(-) raw
    # charge is unconsumed and trips the guard, voiding the whole molecule
    # (correct -- degrade to abstain, never mis-name).
    charge_ok = all_charge_ids | comp.nitro_atoms
    for a in ctx.mol.GetAtoms():
        if a.GetFormalCharge() != 0 and a.GetIdx() not in charge_ok:
            return None  # unspellable internal charge -> void, never mis-name

    # Phase E: certify the atom->token partition through the SAME shared E1
    # core that certifies coverage elsewhere (invariant 12: extend, don't
    # duplicate), REPLACING the hand-rolled two-sided ``covers != heavy or
    # total_bound != len(heavy)`` self-check that lived here. That self-check
    # re-derived E1's P1 atom-partition -- GAP via ``covers != heavy``,
    # DOUBLE-COUNT via ``total_bound`` (fix round 1, finding 5) -- but LACKED
    # E1's token-in-name and F-E1 chemistry-soundness checks, which this now
    # gains for free. ``UniversalResult.bindings`` is already E1's
    # ``(token, atom_ids)`` pairs shape, so no adapter is needed.
    # ``allow_charged=True`` because this producer owns the charge axis by a
    # STRONGER, earlier guard: the per-atom raw-formal-charge void guard above
    # (task B2b) already voided any charge it could not spell -- E1's G1 is a
    # blanket net==0 check that would over-void a validly-spelled cation/anion
    # this producer legitimately emits (function-level import: keep the module
    # graph acyclic, mirroring how t4_coverage/substituent_enumerator import
    # this module).
    from ..validation.e1_certificate import _verify_partition
    verdict = _verify_partition(ctx.mol, comp.name, comp.bindings,
                                allow_charged=True)
    if not verdict.ok:
        return None  # void: E1 rejected the partition (gap / double-count /
        #              phantom / token-not-in-name / all-carbon-token-on-hetero)
        #              -- never ship an atom-incomplete or mis-tokenised name

    covers = frozenset(a for _tok, ids in comp.bindings for a in ids)
    return UniversalResult(name=comp.name, bindings=tuple(comp.bindings), covers=covers)


def _build_ctx(
    mol, atom_work_budget: int,
) -> Optional[Tuple["_Ctx", FrozenSet[int]]]:
    """Shared whole-molecule perception + ``_Ctx`` setup for BOTH public
    entry points (``name_universal_substitutive`` on the whole graph and
    ``name_universal_substituent_prefix`` on a rooted branch subgraph).

    Returns ``(ctx, heavy)`` or ``None`` -- the SAME fail-closed scope guards
    the whole-molecule entry point has always applied (size ceiling,
    multi-fragment, isotope/radical/wildcard, kekulize) run here unchanged, so
    the branch entry point inherits them identically. Extracted verbatim from
    ``_name_universal_substitutive_unsafe`` (behaviour-preserving refactor);
    the top-level charge/atom-coverage ASSERTIONS stay in that function because
    they are meaningful only for a whole-molecule call (they assert over the
    FULL heavy set / FULL ion-site set), while the branch entry point applies
    the branch-restricted analogues itself.
    """
    size_cap = min(atom_work_budget, _MAX_ATOMS_FOR_PERCEPTION)
    if mol.GetNumHeavyAtoms() > size_cap:
        return None
    work = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
    except Exception:
        return None

    if len(Chem.GetMolFrags(work)) != 1:
        return None  # multi-fragment: out of scope for this producer
    # Task B2b: the blanket "ANY nonzero per-atom formal charge -> void"
    # guard that used to live here is GONE -- charge is now perceived and
    # spelled properly (see the module docstring's charge-scope paragraph
    # and ``_resolve_spine_charge`` below), reusing ``general_engine``'s own
    # charge-suffix primitives so a genuine ionic centre gets a real
    # ``-ium``/``-ylium``/``-ide``/``-uide`` suffix (or the P-74.1.1
    # cumulative zwitterion form) instead of being refused outright. What
    # remains refused is exactly what falls outside that reused primitive's
    # OWN scope (FG anions/cations, mixed-sign net charge, ...) -- see reason
    # 4 in this function's docstring -- never a silent, charge-blind name.
    if any(a.GetIsotope() for a in work.GetAtoms()):
        return None
    if any(a.GetNumRadicalElectrons() for a in work.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 0 for a in work.GetAtoms()):
        return None  # wildcard atom: unverifiable, never claim to name it

    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        return None  # fail closed rather than guess a bond order

    assign_stereochemistry(work)  # canonical CIP path (never raw rdCIPLabeler)

    heavy = frozenset(a.GetIdx() for a in work.GetAtoms() if a.GetAtomicNum() > 1)
    if not heavy:
        return None

    ring_systems = [frozenset(s) for s in get_ring_systems(work, include_spiro=True)]
    ring_system_of: Dict[int, int] = {}
    for i, atoms in enumerate(ring_systems):
        for a in atoms:
            ring_system_of[a] = i

    # Fix round 1, finding 4: EVERY tie-break in this module keys on
    # ``canon_rank`` (``Chem.CanonicalRankAtoms``, invariant to input atom
    # numbering), never a raw RDKit atom index -- a raw-index tie-break made
    # the SAME molecule from differently-numbered SMILES emit DIFFERENT (both
    # individually RT-valid) names, e.g. chlorocyclohexane as
    # ``5-chlorocyclohexane`` from one input numbering and
    # ``1-chlorocyclohexane`` from another. Computed ONCE per top-level call.
    canon_rank = tuple(Chem.CanonicalRankAtoms(work, breakTies=True))

    # Task B2b: GENUINE ionic centres only -- ``get_ion_sites`` already
    # excludes P-59 INTERNAL charges (nitro, N-oxide, azide, diazo), which
    # are spelled directly via ``_nitro_shortcut`` instead (see module
    # docstring). Computed ONCE here; every recursive ``_name_component``
    # call consults it to decide whether ITS OWN spine needs a suffix.
    ion_sites = get_ion_sites(work)
    cation_sites = list(ion_sites.get('cations') or [])
    anion_sites = list(ion_sites.get('anions') or [])

    ctx = _Ctx(mol=work, ring_systems=ring_systems, ring_system_of=ring_system_of,
               budget=_Budget(atom_work_budget), canon_rank=canon_rank,
               cation_sites=cation_sites, anion_sites=anion_sites)
    return ctx, heavy


def name_universal_substituent_prefix(
    mol, frag_atoms, attach_idx: int, bond_order: int = 1,
    atom_work_budget: int = DEFAULT_ATOM_WORK_BUDGET,
) -> Optional[str]:
    """Name a BRANCH subgraph as a ``-yl``/``-ylidene``/``-ylidyne`` substituent
    prefix over the WHOLE-molecule perception context, or ``None`` (void).

    Task B3 (the branch-fallback wiring): where the existing recursive
    substituent namer (``substituent_enumerator.name_substituent``) declines a
    hard branch, this re-enters the SAME unconditional recursive machinery on
    just that branch and renders a covering ``-yl`` prefix, cited at the
    branch's own attachment locant. ``attach_idx`` is the atom WITHIN
    ``frag_atoms`` that bonds to the parent; ``bond_order`` is that attachment
    bond's order (1/2/3 -> yl/ylidene/ylidyne).

    Fail-closed, exactly like ``name_universal_substitutive``: returns ``None``
    (never a partial or a mis-cover) whenever the whole-molecule scope guards
    trip, the branch is not a subset of the heavy-atom set, the attach atom is
    not in the branch, the recursive namer voids, or the branch-restricted
    coverage / charge assertions (the analogues of the whole-molecule ones,
    scoped to ``frag_atoms``) fail. The caller gates the returned string
    through the existing E1 + SELF-01 round-trip net (0-wrong preserved); this
    is a PURE producer and never ships.
    """
    if mol is None:
        return None
    try:
        return _name_universal_substituent_prefix_unsafe(
            mol, frag_atoms, attach_idx, bond_order, atom_work_budget)
    except Exception:
        return None


def _name_universal_substituent_prefix_unsafe(
    mol, frag_atoms, attach_idx: int, bond_order: int, atom_work_budget: int,
) -> Optional[str]:
    built = _build_ctx(mol, atom_work_budget)
    if built is None:
        return None
    ctx, heavy = built

    frag = frozenset(int(a) for a in frag_atoms)
    if not frag or not frag.issubset(heavy):
        return None  # a branch atom is a hydrogen / out of range -> void
    if attach_idx not in frag:
        return None  # contract: attach_idx is the branch-side attachment atom

    comp = _name_component(ctx, frag, attach_hint=attach_idx, is_top=False)
    if comp is None:
        return None

    # Phase E: certify the branch's atom->token partition through the SAME
    # shared E1 core (invariant 12), scoped to ``frag`` (the branch's own
    # heavy-atom set) via ``atoms=frag`` -- the analogue of the whole-molecule
    # certification in ``_name_universal_substitutive_unsafe``, REPLACING the
    # hand-rolled two-sided ``covers != frag or total_bound != len(frag)``
    # self-check. Checked against ``comp.name`` (the branch's assembled name
    # BEFORE the ``-yl`` rewrite ``_render_as_substituent`` applies below),
    # which is exactly what the branch's internal tokens compose. Gains
    # token-in-name + F-E1 over the old union/count self-check.
    # ``allow_charged=True`` for the same reason as the whole-molecule call --
    # the branch-restricted raw-charge guard just below owns the charge axis.
    from ..validation.e1_certificate import _verify_partition
    branch_verdict = _verify_partition(ctx.mol, comp.name, comp.bindings,
                                       allow_charged=True, atoms=frag)
    if not branch_verdict.ok:
        return None  # void: never a partial / re-fragmenting / mis-tokenised branch

    # Branch-restricted charge assertions (the analogues of the two whole-
    # molecule ones): every GENUINE ionic centre inside the branch must have
    # been expressed by a suffix, and no raw internal charge inside the branch
    # may be left unspelled (absorbed into the skeleton as if neutral).
    frag_charge_ids = frozenset(
        s['atom_idx'] for s in (ctx.cation_sites + ctx.anion_sites)
        if s['atom_idx'] in frag)
    if comp.charged != frag_charge_ids:
        return None
    charge_ok = comp.charged | comp.nitro_atoms
    for idx in frag:
        if ctx.mol.GetAtomWithIdx(idx).GetFormalCharge() != 0 and idx not in charge_ok:
            return None

    return _render_as_substituent(comp, bond_order)


# ===========================================================================
# The recursive core
# ===========================================================================

def _name_component(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int], is_top: bool,
) -> Optional[_ComponentResult]:
    """Name one component -- the whole molecule (``is_top``) or one branch.

    Termination: ``component`` shrinks strictly on every recursive call (see
    module docstring); the work budget is charged here, once per call, by
    component size, and fails closed on exhaustion.
    """
    ctx.budget.charge(len(component))
    if not component:
        return None

    mol = ctx.mol

    # ---- direct functional-group / leaf shortcuts (branches only) --------
    # Mirrors the reference architecture's "heteroatom-start shortcut" +
    # "direct-functional-group shortcut": a handful of common small groups
    # get their standard prefix directly rather than an ugly generic
    # skeletal-replacement rendering of a 1-3 atom "chain".
    if not is_top and attach_hint is not None:
        shortcut = _leaf_shortcut(mol, component, attach_hint)
        if shortcut is not None:
            token, atoms = shortcut
            # Task B2b fix round 1: record a rendered nitro group's atoms so
            # the top-level raw-charge guard knows its P-59 internal charges
            # ARE accounted for (exception (b)). ``_nitro_shortcut`` is the
            # sole producer of the "nitro" token, so this single-site check
            # is exact -- no other leaf carries a formal charge.
            nitro_atoms = atoms if token == "nitro" else frozenset()
            return _ComponentResult(
                name=token, bindings=[(token, atoms)], covers=atoms,
                attach_locant=None, is_prefix_ready=True,
                nitro_atoms=nitro_atoms,
            )

    # ---- parent (spine) selection -----------------------------------------
    ring_here = _ring_system_for_component(ctx, component, attach_hint, is_top)
    if ring_here is not None:
        spine_result = _name_ring_spine(ctx, component, ring_here, attach_hint)
    else:
        spine_result = _name_chain_spine(ctx, component, attach_hint)
    if spine_result is None:
        return None
    spine_atoms, spine_core, spine_atom_to_locant, attach_locant = spine_result

    # ---- charge (Task B2b): does THIS level's own spine carry a genuine
    # ionic centre? Checked BEFORE branch discovery so an unresolvable
    # charge voids immediately without wasted recursive work -- see the
    # module docstring's charge-scope paragraph and ``_resolve_spine_charge``.
    needs_charge, charge_suffix_text, charge_ids = _resolve_spine_charge(
        ctx, spine_atom_to_locant)
    if needs_charge and charge_suffix_text is None:
        return None  # genuine ionic centre on this spine, not expressible

    # ---- branches: every off-spine atom, named by RE-ENTERING this SAME
    # function on its own (strictly smaller) subgraph. NO depth cap. -------
    # Phase E (fix round 1): store the FULL, unstemmed ``spine_core`` token
    # (``hexane``/``cyclohexane``/``2-azapropane``...). E1's F-E1 all-carbon
    # classifier and P1 partition both need the true token -- F-E1's grammar
    # only recognises the FULL suffix form (``hexane``->all-carbon,
    # ``hexan``->unclassified), so a pre-stemmed token would silently disable
    # F-E1 for exactly the alkane/cycloalkane class it exists to guard. The
    # parent core's trailing 'e' IS legitimately elided when a charge/ionic
    # suffix is appended below (``_elide_before_ionic_suffix``;
    # P-16.7.1(a)/P-74.1.1), so ``2-azapropane`` is not a literal substring of
    # ``...2-azapropan-2-ium`` -- but that elision is tolerated on the
    # TOKEN-IN-NAME axis ALONE, inside ``_verify_partition`` (accepts the stem
    # ``token[:-1]``), NOT by pre-stemming the stored token here.
    bindings: List[Tuple[str, FrozenSet[int]]] = [
        (spine_core, frozenset(spine_atoms)),
    ]
    charged_accum: set = set(charge_ids)
    nitro_accum: set = set()  # Task B2b fix round 1: rendered-nitro atoms
    prefix_entries: Dict[str, List[int]] = {}
    for s_atom, root, order, branch_atoms in _discover_branches(
        mol, spine_atoms, component,
    ):
        sub = _name_component(ctx, branch_atoms, attach_hint=root, is_top=False)
        if sub is None:
            return None  # never ship a partial name: whole call voids
        rendered = _render_as_substituent(sub, order)
        loc = spine_atom_to_locant[s_atom]
        prefix_entries.setdefault(rendered, []).append(loc)
        bindings.append((rendered, sub.covers))
        charged_accum |= sub.charged
        nitro_accum |= sub.nitro_atoms

    prefix_parts = []
    for name, locs in prefix_entries.items():
        locs = sorted(locs)
        prefix_parts.append((alpha_sort_key(name),
                              format_substituent_prefix(name, locs, len(locs))))
    prefix_parts.sort(key=lambda t: t[0])
    joined = "-".join(p for _k, p in prefix_parts)

    if joined:
        # P-16.5 hyphen glue: a joining '-' is needed only before a
        # locant-initial core (mirrors general_engine._emit_ring_from_analysis).
        glue = "-" if spine_core[:1].isdigit() else ""
        full_name = joined + glue + spine_core
    else:
        full_name = spine_core

    if needs_charge:
        # Appended to the FULLY assembled name (prefixes + parent core), not
        # just ``spine_core`` -- mirrors general_engine's own ordering
        # (``_append_charge_suffix`` is called on the substituent-decorated
        # name), so P-16.7.1(a)/P-74.1.1 elision targets the parent
        # hydride's own trailing 'e', never truncates a substituent prefix.
        full_name = _elide_before_ionic_suffix(full_name, charge_suffix_text)

    covers = frozenset(a for _t, ids in bindings for a in ids)
    return _ComponentResult(
        name=full_name, bindings=bindings, covers=covers,
        attach_locant=attach_locant, charged=frozenset(charged_accum),
        nitro_atoms=frozenset(nitro_accum),
    )


def _resolve_spine_charge(
    ctx: _Ctx, spine_atom_to_locant: Dict[int, int],
) -> Tuple[bool, Optional[str], FrozenSet[int]]:
    """Does THIS level's own spine carry a genuine ionic centre, and if so,
    what suffix text expresses it (reusing ``general_engine``'s primitives)?

    Returns ``(needs_charge, suffix_text, charged_atom_ids)``:
      * ``(False, None, frozenset())`` -- no genuine ionic centre touches
        this spine; nothing to do (the overwhelmingly common, uncharged case).
      * ``(True, None, frozenset())`` -- a genuine ionic centre touches this
        spine but is NOT expressible by the reused primitives (an FG anion
        like carboxylate/alkoxide, an FG cation like diazonium/acylium, a
        mixed-sign net-charged species, ...) -- the caller MUST void.
      * ``(True, "<suffix>", ids)`` -- resolved; the caller appends
        ``<suffix>`` (via ``_elide_before_ionic_suffix``) and records ``ids``
        as consumed (the top-level charge-coverage assertion).

    Two mutually-exclusive reused shapes, tried in turn (a molecule can only
    ever match ONE, since they gate on opposite global site shapes):
      * ``general_engine._charge_suffix_text`` -- a net-charged, SINGLE-SIGN
        species (a plain cation or anion; internally ``bool(anions) ==
        bool(cations)`` must be False, i.e. exactly one sign present at all,
        globally). Scope is general_engine's OWN P5 scope -- 'ylium'/
        'aminium'/'onium'/'quaternary' cations, 'carbanion'/
        'heteroatom_hydride_anion'/'uide_anion' anions; anything else (FG
        anions/cations) returns None from that function, which this treats
        as "not expressible", never as "no charge here".
      * ``general_engine._zwitterion_suffix_plan(..., allow_fg_anion=False)``
        -- a net-0 zwitterion with BOTH ionic centres skeletal to THIS SAME
        spine (P-74.1.1). ``allow_fg_anion=False`` disables the P-74.1.2
        "-olate" FG-anchored branch: this producer has no functional-group-
        suffix layer to hold an atom out of substituent discovery for it, so
        that branch is deferred, not built (a carboxylate-anchored
        zwitterion, e.g. an amino acid, is therefore NOT expressible here and
        voids -- see module docstring).
    """
    spine_atoms = set(spine_atom_to_locant)
    touches = (
        any(s['atom_idx'] in spine_atoms for s in ctx.cation_sites)
        or any(s['atom_idx'] in spine_atoms for s in ctx.anion_sites)
    )
    if not touches:
        return False, None, frozenset()

    single_sign = _charge_suffix_text(ctx.mol, spine_atom_to_locant)
    if single_sign is not None:
        text, charged_atom_ids = single_sign
        return True, text, frozenset(charged_atom_ids)

    zwit = _zwitterion_suffix_plan(ctx.mol, spine_atom_to_locant, allow_fg_anion=False)
    if zwit is not None:
        held, text = zwit
        if not held:  # allow_fg_anion=False -> always empty when it succeeds
            resolved = frozenset(
                s['atom_idx'] for s in (ctx.cation_sites + ctx.anion_sites)
                if s['atom_idx'] in spine_atoms
            )
            return True, text, resolved

    return True, None, frozenset()  # touches, but not expressible -> void


def _render_as_substituent(sub: _ComponentResult, bond_order: int) -> str:
    """Rewrite a component's parent name as a ``-yl``/``-ylidene``/``-ylidyne``
    substituent prefix, cited at ITS OWN attachment locant.

    Mechanical rule (uniform for chain, ring and polycyclic spines alike --
    always VALID, not always the shortest/PIN-preferred spelling): drop the
    trailing parent-hydride ``e`` and append ``-{locant}-yl`` (or
    ``-ylidene``/``-ylidyne`` for a double/triple exocyclic attachment).

    A ``_leaf_shortcut`` result (``is_prefix_ready``, e.g. ``carboxy``,
    ``hydroxy``, ``oxo``) is ALREADY a complete substituent prefix -- its
    bond order to the parent is already baked into which token was chosen
    (``hydroxy`` for a single-bonded leaf O, ``oxo`` for a double-bonded
    one) -- so it is returned unchanged rather than mechanically suffixed.
    """
    if sub.is_prefix_ready:
        return sub.name
    name = sub.name
    stem = name[:-1] if name.endswith("e") else name
    suffix = {1: "yl", 2: "ylidene", 3: "ylidyne"}.get(bond_order, "yl")
    loc = sub.attach_locant if sub.attach_locant is not None else 1
    return f"{stem}-{loc}-{suffix}"


# ===========================================================================
# Branch discovery (the termination primitive)
# ===========================================================================

def _discover_branches(
    mol, spine_atoms: FrozenSet[int], component: FrozenSet[int],
) -> List[Tuple[int, int, int, FrozenSet[int]]]:
    """Every off-spine branch of *spine_atoms* within *component*.

    Returns ``(spine_atom, branch_root, bond_order, branch_component)``
    tuples. ``branch_component`` is computed by BFS bounded by a
    monotonically GROWING exclude set (starts at ``spine_atoms``, gains each
    branch as it is claimed) -- the shrinking-atom-set termination argument.
    """
    growing_exclude = set(spine_atoms)
    roots: List[Tuple[int, int, int]] = []
    seen_roots = set()
    for s in sorted(spine_atoms):
        for nb in mol.GetAtomWithIdx(s).GetNeighbors():
            j = nb.GetIdx()
            if nb.GetAtomicNum() <= 1:
                continue
            if j in growing_exclude or j not in component or j in seen_roots:
                continue
            seen_roots.add(j)
            bond = mol.GetBondBetweenAtoms(s, j)
            order = round(bond.GetBondTypeAsDouble())
            roots.append((s, j, order))

    out: List[Tuple[int, int, int, FrozenSet[int]]] = []
    for s, j, order in roots:
        if j in growing_exclude:
            continue  # claimed by an earlier branch's BFS already
        branch_component = _bfs_component(mol, j, frozenset(growing_exclude))
        growing_exclude |= branch_component
        out.append((s, j, order, branch_component))
    return out


def _bfs_component(mol, start: int, exclude: FrozenSet[int]) -> FrozenSet[int]:
    """Heavy-atom BFS from *start*, never crossing *exclude*."""
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if nb.GetAtomicNum() <= 1 or j in exclude or j in seen:
                continue
            seen.add(j)
            stack.append(j)
    return frozenset(seen)


def _tree_neighbors(ctx: _Ctx, atom: int, component: FrozenSet[int]) -> List[int]:
    """Neighbors of *atom* usable for CHAIN-spine walking: heavy, in
    *component*, NOT a ring atom (rings are branch ports, not chain
    continuations -- see module docstring on chain/ring composition), and
    NOT the nitrogen of a nitro group (Task B2b, ``_is_nitro_root``): nitro
    is always resolved as a branch via the dedicated ``_nitro_shortcut``
    leaf, never mechanically threaded into a replacement-nomenclature chain
    as if its nitrogen were an ordinary standard-valence heteroatom -- doing
    so would silently drop the +1/-1 charge information a plain locanted
    "aza" carries no trace of (see module docstring's charge-scope
    paragraph)."""
    out = []
    for nb in ctx.mol.GetAtomWithIdx(atom).GetNeighbors():
        j = nb.GetIdx()
        if nb.GetAtomicNum() <= 1 or j not in component:
            continue
        if ctx.ring_system_of.get(j) is not None:
            continue
        if _is_nitro_root(ctx.mol, j):
            continue
        out.append(j)
    return out


# ===========================================================================
# Ring-system selection
# ===========================================================================

def _ring_system_for_component(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int], is_top: bool,
) -> Optional[FrozenSet[int]]:
    """Which (if any) ring system in *component* is this call's parent spine.

    Top level: ring beats chain whenever ANY ring atoms are present
    (conventional seniority simplification) -- the LARGEST ring system in
    the component, ties broken by lowest minimum atom index.
    Branch: a ring is the spine ONLY when the branch's own attachment atom
    IS a ring atom (P-29.2: substituent numbering starts at the free
    valence) -- otherwise the branch is a chain and any ring further inside
    is itself a deeper branch, discovered recursively.
    """
    candidates = []
    if is_top:
        seen = set()
        for a in component:
            idx = ctx.ring_system_of.get(a)
            if idx is not None and idx not in seen:
                seen.add(idx)
                sys_atoms = ctx.ring_systems[idx]
                if sys_atoms <= component:
                    candidates.append(sys_atoms)
        if not candidates:
            return None
        candidates.sort(key=lambda s: (-len(s), min(ctx.canon_rank[a] for a in s)))
        return candidates[0]
    else:
        if attach_hint is None:
            return None
        idx = ctx.ring_system_of.get(attach_hint)
        if idx is None:
            return None
        sys_atoms = ctx.ring_systems[idx]
        if not (sys_atoms <= component):
            return None
        return sys_atoms


# ===========================================================================
# Ring spine construction (monocyclic own-built + polycyclic reuse)
# ===========================================================================

def _name_ring_spine(
    ctx: _Ctx, component: FrozenSet[int], ring_atoms: FrozenSet[int],
    attach_hint: Optional[int],
) -> Optional[Tuple[FrozenSet[int], str, Dict[int, int], Optional[int]]]:
    mol = ctx.mol
    ri = mol.GetRingInfo()
    sssr_here = [set(r) for r in ri.AtomRings() if set(r) <= ring_atoms]
    if len(sssr_here) >= 2:
        cage = analyze_cage_universal(mol, cage_atoms=set(ring_atoms))
        if cage is None:
            return None
        parent_block = _build_parent_with_unsaturation(
            cage.total_atoms, cage.unsaturation, fg_suffix=None)
        core = cage.hetero_prefix + cage.descriptor + parent_block
        atom_to_locant = dict(cage.atom_to_locant)
        spine = frozenset(cage.cage_atoms)
        attach_locant = atom_to_locant.get(attach_hint) if attach_hint is not None else None
        return spine, core, atom_to_locant, attach_locant

    # Monocyclic: build our own minimal cyclo/'a'-replacement name.
    if len(sssr_here) != 1:
        return None  # defensive: a ring system with 0 SSSR rings inside it
    ring_tuple = next((r for r in ri.AtomRings() if set(r) == ring_atoms), None)
    if ring_tuple is None:
        return None
    n = len(ring_tuple)
    base = list(ring_tuple)

    def candidates_for(start_first: Optional[int]):
        cands = []
        starts = range(n) if start_first is None else [
            i for i in range(n) if base[i] == start_first
        ]
        for start in starts:
            rotated = base[start:] + base[:start]
            cands.append(rotated)
            cands.append([rotated[0]] + list(reversed(rotated[1:])))
        return cands

    if attach_hint is not None:
        cands = candidates_for(attach_hint)
    else:
        cands = candidates_for(None)
    if not cands:
        return None

    # Fix round 1, finding 3: scoring every candidate is O(len(cands) * n) --
    # for the free-numbering (``attach_hint is None``) case ``len(cands) ==
    # 2n``, so this is O(n^2) work that happened AFTER the one up-front
    # ``ctx.budget.charge(len(component))`` at call entry, i.e. budget-blind.
    # Measured: 1.4s @ n=501 -> 87.5s @ n=4001 (clean quadratic), ~550s
    # projected at n=9999 -- a LEGAL input that would otherwise return a name
    # but stall for minutes first. Charge the actual O(len(cands)*n) work
    # before doing it, so a large monocycle trips the SAME budget a large
    # chain or a deep branch structure would, instead of being exempt from it.
    ctx.budget.charge(len(cands) * n)

    def score(order):
        hetero = sorted(i + 1 for i, a in enumerate(order)
                        if mol.GetAtomWithIdx(a).GetAtomicNum() != 6)
        unsat = []
        for i in range(n):
            a, b = order[i], order[(i + 1) % n]
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None:
                continue
            bt = round(bond.GetBondTypeAsDouble())
            if bt >= 2:
                unsat.append(i + 1)
        # Fix round 1, finding 4: a substituent-locant tier (mirroring
        # ``_chain_score``) -- without it, a plain (or symmetrically
        # decorated) carbocycle ties on (hetero, unsat) for EVERY rotation,
        # and ``min()`` silently falls back to iteration order, which tracks
        # the RDKit ring-tuple order, which tracks INPUT ATOM NUMBERING (e.g.
        # chlorocyclohexane emitted "5-chlorocyclohexane" from one input
        # numbering and "1-chlorocyclohexane" from another -- both RT-valid,
        # neither deterministic, and not even lowest-locant).
        branch_locants = []
        for i, a in enumerate(order):
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if (nb.GetAtomicNum() > 1 and j in component
                        and j not in ring_atoms):
                    branch_locants.append(i + 1)
        branch_locants.sort()
        # Final tie-break: canonical rank of the candidate's OWN atom
        # sequence, invariant to input atom numbering -- resolves any
        # remaining tie (a genuinely symmetric ring) the SAME way regardless
        # of how the input SMILES happened to number its atoms.
        canon_tie = tuple(ctx.canon_rank[a] for a in order)
        return (hetero, unsat, branch_locants, canon_tie)

    best = min(cands, key=score)
    atom_to_locant = {a: i + 1 for i, a in enumerate(best)}

    double_bonds, triple_bonds = [], []
    for i in range(n):
        a, b = best[i], best[(i + 1) % n]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue
        bt = round(bond.GetBondTypeAsDouble())
        if bt == 2:
            double_bonds.append(i + 1)
        elif bt == 3:
            triple_bonds.append(i + 1)
    unsaturation = {"double_bonds": double_bonds, "triple_bonds": triple_bonds}

    hetero_prefix = _build_hetero_prefix(mol, best, atom_to_locant)
    parent_block = _build_parent_with_unsaturation(n, unsaturation, fg_suffix=None)
    core = hetero_prefix + "cyclo" + parent_block

    attach_locant = atom_to_locant.get(attach_hint) if attach_hint is not None else None
    return frozenset(ring_atoms), core, atom_to_locant, attach_locant


# ===========================================================================
# Chain spine construction (acyclic; own tree-diameter / rooted walk)
# ===========================================================================

def _name_chain_spine(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int],
) -> Optional[Tuple[FrozenSet[int], str, Dict[int, int], Optional[int]]]:
    mol = ctx.mol

    if attach_hint is not None:
        path = _farthest_path_from(ctx, attach_hint, component)
    else:
        # Two-BFS tree-diameter technique: BFS from an arbitrary atom finds
        # one end (u) of a longest path; BFS from u finds the other end and,
        # via parent pointers, the path itself. The seed atom is chosen by
        # CANONICAL rank (fix round 1, finding 4), not raw atom index, so the
        # SAME molecule from a differently-numbered SMILES starts from the
        # same graph-invariant seed.
        seed = min(component, key=lambda a: ctx.canon_rank[a])
        u = _farthest_from(ctx, seed, component)
        path = _farthest_path_from(ctx, u, component)
        # Free choice of numbering direction: lowest locants to heteroatoms,
        # then unsaturation, then (P-31 simplified) substituent attachment
        # points, then a canonical-rank tie-break -- an all-carbon saturated
        # chain ties on the first two, so without the third tier a branch
        # could be numbered from the wrong end (e.g. 2-methylbutane emitted
        # as "3-methylbutane"), and without the final tie-break a fully
        # symmetric tie could still resolve by raw atom index (numbering-
        # dependent, non-deterministic across equivalent SMILES).
        rev = list(reversed(path))
        if _chain_score(ctx, rev, component) < _chain_score(ctx, path, component):
            path = rev

    if not path:
        return None
    n = len(path)
    atom_to_locant = {a: i + 1 for i, a in enumerate(path)}

    double_bonds, triple_bonds = [], []
    for i in range(n - 1):
        bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
        if bond is None:
            continue
        bt = round(bond.GetBondTypeAsDouble())
        if bt == 2:
            double_bonds.append(i + 1)
        elif bt == 3:
            triple_bonds.append(i + 1)
    unsaturation = {"double_bonds": double_bonds, "triple_bonds": triple_bonds}

    hetero_prefix = _build_hetero_prefix(mol, path, atom_to_locant)
    parent_block = _build_parent_with_unsaturation(n, unsaturation, fg_suffix=None)
    core = hetero_prefix + parent_block

    attach_locant = atom_to_locant.get(attach_hint) if attach_hint is not None else None
    return frozenset(path), core, atom_to_locant, attach_locant


def _chain_score(ctx: _Ctx, order: List[int], component: Optional[FrozenSet[int]] = None):
    mol = ctx.mol
    hetero = [i + 1 for i, a in enumerate(order)
              if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    unsat = []
    for i in range(len(order) - 1):
        bond = mol.GetBondBetweenAtoms(order[i], order[i + 1])
        if bond is not None and round(bond.GetBondTypeAsDouble()) >= 2:
            unsat.append(i + 1)
    branch_locants: List[int] = []
    if component is not None:
        spine_set = set(order)
        for i, a in enumerate(order):
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if nb.GetAtomicNum() > 1 and j in component and j not in spine_set:
                    branch_locants.append(i + 1)
        branch_locants.sort()
    # Fix round 1, finding 4: final canonical-rank tie-break -- resolves a
    # fully symmetric remaining tie the SAME way regardless of input atom
    # numbering (never falls back to raw atom index / iteration order).
    canon_tie = tuple(ctx.canon_rank[a] for a in order)
    return (hetero, unsat, branch_locants, canon_tie)


def _farthest_from(ctx: _Ctx, start: int, component: FrozenSet[int]) -> int:
    """BFS from *start* over chain-tree neighbors; returns the farthest atom
    reached (ties broken by CANONICAL rank -- fix round 1, finding 4 -- never
    raw atom index, so the result is invariant to input atom numbering)."""
    dist = {start: 0}
    order_seen = [start]
    stack = [start]
    while stack:
        cur = stack.pop(0)
        for nb in _tree_neighbors(ctx, cur, component):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                order_seen.append(nb)
                stack.append(nb)
    maxd = max(dist.values())
    best = min((a for a in order_seen if dist[a] == maxd), key=lambda a: ctx.canon_rank[a])
    return best


def _farthest_path_from(ctx: _Ctx, start: int, component: FrozenSet[int]) -> List[int]:
    """The longest chain-tree path STARTING at *start* (forced root -- this
    is how a branch's attachment atom always becomes its own locant 1)."""
    parent: Dict[int, Optional[int]] = {start: None}
    dist = {start: 0}
    order_seen = [start]
    stack = [start]
    while stack:
        cur = stack.pop(0)
        for nb in _tree_neighbors(ctx, cur, component):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                order_seen.append(nb)
                stack.append(nb)
    maxd = max(dist.values())
    far = min((a for a in order_seen if dist[a] == maxd), key=lambda a: ctx.canon_rank[a])
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ===========================================================================
# Skeletal ('a') replacement prefix (P-15.4.3.1 citation order)
# ===========================================================================

def _build_hetero_prefix(mol, spine_order: List[int], atom_to_locant: Dict[int, int]) -> str:
    by_element: Dict[str, List[int]] = {}
    for a in spine_order:
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym in REPLACEMENT_TERMS:
            by_element.setdefault(sym, []).append(atom_to_locant[a])
    if not by_element:
        return ""
    elements = sorted(by_element, key=lambda e: _A_CITATION_INDEX_SAFE(e))
    parts = []
    for el in elements:
        locs = sorted(by_element[el])
        term = REPLACEMENT_TERMS[el]
        count = len(locs)
        mult = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        locant_str = ",".join(str(x) for x in locs)
        parts.append(f"{locant_str}-{mult}{term}")
    return "-".join(parts)


def _A_CITATION_INDEX_SAFE(el: str) -> int:
    try:
        return _A_CITATION_ORDER.index(el)
    except ValueError:
        return len(_A_CITATION_ORDER)


# ===========================================================================
# Direct functional-group / leaf shortcuts (branches only)
# ===========================================================================

_LEAF_SINGLE = {
    "F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo",
    "O": "hydroxy", "N": "amino", "S": "sulfanyl",
    "Se": "selanyl", "Te": "tellanyl",
    "P": "phosphanyl", "As": "arsanyl", "Sb": "stibanyl", "Bi": "bismuthanyl",
    "Si": "silyl", "Ge": "germyl", "Sn": "stannyl", "Pb": "plumbyl", "B": "boranyl",
}
_LEAF_DOUBLE = {
    "O": "oxo", "S": "sulfanylidene", "Se": "selanylidene", "Te": "tellanylidene",
    "N": "imino",
}


def _leaf_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """A handful of common small groups named directly rather than via the
    generic chain/replacement machinery. Returns ``(token, atom_ids)`` or
    ``None`` (fall through to generic construction)."""
    # Task B2b: nitro is checked FIRST, before the generic charge guard just
    # below -- its charges are P-59 INTERNAL (excluded from
    # ``perception.ions.get_ion_sites``'s genuine-ion-site perception), so
    # this is the standard neutral-molecule nitro group's own Lewis
    # structure, not a "genuine ionic centre" the charge-suffix machinery
    # needs to see.
    nitro = _nitro_shortcut(mol, component, attach_hint)
    if nitro is not None:
        return nitro

    # Task B2b: a GENUINELY charged atom anywhere in this branch (nitro,
    # just checked, is the only exception) must NEVER take one of the
    # NEUTRAL shortcuts below -- none of them checks formal charge, so a
    # lone O- would otherwise be mis-named "hydroxy" (a different, neutral
    # constitution; wrong molecular formula and net charge). Bail to generic
    # spine construction instead, where a charged atom is always forced onto
    # SOME level's own spine and handled by ``_resolve_spine_charge``.
    if any(mol.GetAtomWithIdx(a).GetFormalCharge() != 0 for a in component):
        return None

    atom = mol.GetAtomWithIdx(attach_hint)
    others = [n.GetIdx() for n in atom.GetNeighbors()
              if n.GetIdx() in component and n.GetAtomicNum() > 1]

    # Single heavy atom leaf: F/Cl/Br/I/-OH/=O/-NH2/=NH/-SH/=S/... .
    # Disambiguate single vs. double attachment from the ACTUAL bond order to
    # the (excluded) outside parent atom -- never guessed from H count, which
    # would misclassify monovalent halogens (0 implicit H, always a single
    # bond).
    if len(component) == 1 and not others:
        outside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in component]
        sym = atom.GetSymbol()
        if len(outside) == 1:
            bond = mol.GetBondBetweenAtoms(attach_hint, outside[0])
            order = round(bond.GetBondTypeAsDouble()) if bond is not None else 1
            if order == 1 and sym in _LEAF_SINGLE:
                return _LEAF_SINGLE[sym], frozenset(component)
            if order == 2 and sym in _LEAF_DOUBLE:
                return _LEAF_DOUBLE[sym], frozenset(component)
        return None

    # cyano: C bonded only to a terminal triple-bonded N.
    if len(component) == 2 and attach_hint in component:
        other = next((a for a in component if a != attach_hint), None)
        if (other is not None and atom.GetSymbol() == "C"
                and mol.GetAtomWithIdx(other).GetSymbol() == "N"):
            bond = mol.GetBondBetweenAtoms(attach_hint, other)
            if bond is not None and round(bond.GetBondTypeAsDouble()) == 3:
                other_others = [n.GetIdx() for n in
                               mol.GetAtomWithIdx(other).GetNeighbors()
                               if n.GetIdx() in component]
                if other_others == [attach_hint]:
                    return "cyano", frozenset(component)

    # carboxy: C(=O)(OH) attached via the carbon, exactly 3 atoms.
    if len(component) == 3 and atom.GetSymbol() == "C" and len(others) == 2:
        o_atoms = [o for o in others if mol.GetAtomWithIdx(o).GetSymbol() == "O"]
        if len(o_atoms) == 2:
            kinds = []
            ok = True
            for o in o_atoms:
                o_atom = mol.GetAtomWithIdx(o)
                o_nbrs = [n.GetIdx() for n in o_atom.GetNeighbors() if n.GetIdx() in component]
                if o_nbrs != [attach_hint]:
                    ok = False
                    break
                bond = mol.GetBondBetweenAtoms(attach_hint, o)
                bt = round(bond.GetBondTypeAsDouble())
                kinds.append((bt, o_atom.GetTotalNumHs()))
            if ok and sorted(kinds) == sorted([(2, 0), (1, 1)]):
                return "carboxy", frozenset(component)

    # A genuine N(OH)2 branch (real neutral atoms, no nitro shape -- see
    # ``_nitro_shortcut`` above) falls through to the generic
    # skeletal-replacement chain construction (ugly, but correct).
    return None


def _nitro_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """``-N(+)(=O)[O-]`` attached via N, exactly 3 atoms -- the standard
    neutral-molecule nitro group.

    Task B2b (this replaces the shortcut fix round 1 DELETED rather than
    fixed-in-place, per that round's own note -- see the comment this
    function's call site now precedes): fix round 1's nitro shortcut checked
    NO bond order or charge at all (keyed only on "N bonded to two terminal
    O's"), so it fired on N(OH)2 too and named it "nitro" -- a DIFFERENT,
    wrong-constitution molecule (measured: ``CCCC(CCC)N(O)O`` ->
    "4-nitroheptane", C7H15NO2, not the actual C7H17NO2). This version checks
    EVERY bond order and EVERY formal charge explicitly: the nitrogen must
    carry formal charge +1 and exactly two O neighbours, one DOUBLE-bonded
    and neutral, the other SINGLE-bonded and -1 -- the real nitro Lewis
    structure, and nothing else can match it.

    Nitro's charges are P-59 INTERNAL (excluded from
    ``perception.ions.get_ion_sites``), so this leaf is spelled directly,
    never via the genuine-ion charge-suffix machinery in
    ``_resolve_spine_charge`` -- and ``_is_nitro_root``/``_tree_neighbors``
    keeps this nitrogen out of ordinary chain-spine continuation so it is
    always discovered as a branch (where THIS function can see it), never
    mis-threaded into a replacement-nomenclature chain."""
    if len(component) != 3:
        return None
    atom = mol.GetAtomWithIdx(attach_hint)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 1:
        return None
    in_component = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in component]
    if len(in_component) != 2:
        return None
    outside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in component]
    if len(outside) != 1:
        return None  # nitro's N must have exactly one non-oxygen substituent
    kinds = []
    for o_idx in in_component:
        o_atom = mol.GetAtomWithIdx(o_idx)
        if o_atom.GetSymbol() != "O":
            return None
        o_nbrs = [n.GetIdx() for n in o_atom.GetNeighbors()]
        if o_nbrs != [attach_hint]:
            return None  # each O must bond ONLY to this N (terminal)
        bond = mol.GetBondBetweenAtoms(attach_hint, o_idx)
        bt = round(bond.GetBondTypeAsDouble())
        kinds.append((bt, o_atom.GetFormalCharge(), o_atom.GetTotalNumHs()))
    if sorted(kinds) == sorted([(2, 0, 0), (1, -1, 0)]):
        return "nitro", frozenset(component)
    return None


def _is_nitro_root(mol, j: int) -> bool:
    """True if atom *j* is a nitro group's nitrogen (see ``_nitro_shortcut``
    for the exact shape) -- used by ``_tree_neighbors`` to keep nitro OUT of
    ordinary chain-spine continuation, so it is always resolved as a branch
    via the charge-validated leaf shortcut, never mis-threaded into a
    replacement-nomenclature chain as if it were an uncharged, standard-
    valence heteroatom (which would silently drop its charge)."""
    atom = mol.GetAtomWithIdx(j)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 1 or atom.GetDegree() != 3:
        return False
    o_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
    if len(o_neighbors) != 2:
        return False
    kinds = []
    for o_atom in o_neighbors:
        if o_atom.GetDegree() != 1:
            return False
        bond = mol.GetBondBetweenAtoms(j, o_atom.GetIdx())
        bt = round(bond.GetBondTypeAsDouble())
        kinds.append((bt, o_atom.GetFormalCharge(), o_atom.GetTotalNumHs()))
    return sorted(kinds) == sorted([(2, 0, 0), (1, -1, 0)])
