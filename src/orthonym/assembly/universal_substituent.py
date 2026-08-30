"""Phase B.2 -- the unconditional recursive substitutive namer core.

``name_universal_substitutive(mol)`` is a NEW, self-contained T4/best-effort
producer: it NEVER declines a hard branch.  Where the existing recursive
substituent namer (``assembly/substituent_enumerator.py``) hits an unnameable
fragment and does ``return None`` / emits the ``'substituent'`` sentinel --
which fails the WHOLE enclosing candidate closed -- this module re-enters
ITSELF on the branch subgraph and always bottoms out at a valid (possibly
ugly, non-PIN) systematic token: a simple leaf prefix, a von-Baeyer/'a'-
replacement skeleton, or a deeper recursive call.  There is no depth cap.

Architecture (independent implementation, no depth cap --
a bounded recursive substituent namer):

    name(component) = parent(spine) + Sum(name(branch_i) rendered as -yl)

Termination (the shrinking-atom-set argument, verbatim from that research):
every recursive call is handed a component that is a PROPER SUBSET of its
caller's component -- the caller's spine is non-empty (>=1 atom) and is
removed before any branch is computed, and branch components are pairwise
disjoint (each BFS is bounded by a monotonically GROWING exclude set built
from already-claimed atoms).  So along any recursion path the component size
strictly decreases every call; recursion cannot be infinite.  Neither
recursion here could otherwise cost unbounded WORK on a pathological input (a
depth>=1 local decline is not enough on its own, exactly
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
Isotopes, wildcard atoms and multi-fragment inputs are OUT OF SCOPE and void
the whole call (``None``).  Radicals are OUT OF SCOPE for MORE than one
free-valence centre (P-71.2.3 multi-site / diradicals / radical ions); a
SINGLE monovalent/divalent/trivalent centre is IN SCOPE for ``_build_ctx``
(v36 A2, C2) so ``name_universal_substituent_prefix`` can cite it as a
``-yl``/``-ylidene``/``-ylidyne`` substituent prefix -- see that function's
docstring and ``t4_coverage._best_effort_candidate``'s radical branch, the
only wired caller. ``name_universal_substitutive`` (the whole-molecule entry
point) still has NO suffix logic for that centre and will name the
constitution as if it were fully saturated -- harmless (the caller's OPSIN
``-r`` gate rejects the mismatch) but never a rescue on its own. Indicated
hydrogen is also OUT OF SCOPE for this task (deferred -- no witness in the
B2b brief requires it) and voids nothing on its own: this module simply does
not special-case it, which is a pre-existing gap (unrelated to charge), not
something this task introduces or worsens.

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
(``cation_sites | anion_sites``, asserted resolved) or (b) a P-59/P-74.2.1
internal centre actually rendered by a dedicated internal-charge leaf
(nitro/azido/diazo/..., tracked as
``_ComponentResult.internal_atoms``). ``get_ion_sites`` lists every
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
from ..rules.vonbaeyer_universal import (
    analyze_cage_universal, analyze_spiro_universal,
)
from ..rules.polycyclic import _build_parent_with_unsaturation
# Task B2b: reuse general_engine's own charge-suffix primitives so this
# module spells charge the SAME way as the rest of the codebase (never
# reinvented) -- see the module docstring's charge-scope paragraph.
from .general_engine import (
    _charge_suffix_text,
    _zwitterion_suffix_plan,
    _elide_before_ionic_suffix,
    _stereo_prefix,
    _MULT_SIMPLE,
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
    # Task C: when True the spiro ring-leaf FORCES the von-Baeyer form of a fused
    # spiro component (skipping the systematic fusion namer), so the floor can
    # RETRY a molecule whose systematic fused-spiro name failed the offer RT gate
    # (a wrong/unparseable fusion descriptor) with the always-faithful kekulized
    # von-Baeyer polyene instead. Default off; set only on the retry path.
    force_vonbaeyer_spiro: bool = False


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
    internal_atoms: FrozenSet[int] = frozenset()  # Task B2b fix round 1 +
    #                                              M2: atom indices whose P-59
    #                                              INTERNAL / P-74.2.1 semipolar
    #                                              formal charge was RENDERED by
    #                                              a dedicated internal-charge
    #                                              leaf (``_nitro_shortcut`` ->
    #                                              nitro; M2 ``_azide_shortcut``
    #                                              -> azido; ``_diazo_shortcut``
    #                                              -> diazo; the ``oxido``/-ium
    #                                              N-/P-/S-oxide pair) SOMEWHERE
    #                                              within this component (this
    #                                              level's own leaf, unioned with
    #                                              every child branch's own
    #                                              `internal_atoms`). Threaded --
    #                                              not re-derived by string-
    #                                              matching a token at the top
    #                                              level -- because such a centre
    #                                              nested inside a rendered
    #                                              ``-yl`` substituent is folded
    #                                              into that substituent's token
    #                                              and would be invisible to such
    #                                              a scan. Consumed by the
    #                                              top-level RAW-formal-charge
    #                                              void guard: these P-59/P-74.2.1
    #                                              internal charges are the
    #                                              internal-charge shapes this
    #                                              module can actually spell,
    #                                              so they are exception (b) to
    #                                              that guard (get_ion_sites
    #                                              STRIPS them, so they never
    #                                              appear in ``all_charge_ids``).
    spine_atom_to_locant: Optional[Dict[int, int]] = None  # WS-STEREO: the
    #                                              atom->locant map of THIS
    #                                              level's OWN spine (chain,
    #                                              monocycle or polycycle),
    #                                              exactly the shape
    #                                              ``general_engine._stereo_prefix``
    #                                              consumes -- ``None`` for a
    #                                              leaf-shortcut result (no
    #                                              spine of its own). Only the
    #                                              TOP-level caller reads this
    #                                              (mirroring general_engine's
    #                                              own four call sites, which
    #                                              are each a single PARENT-
    #                                              scope stereo block, never a
    #                                              merge across independently-
    #                                              numbered branch spines).


# ===========================================================================
# Public entry point
# ===========================================================================

def name_universal_substitutive(
    mol, atom_work_budget: int = DEFAULT_ATOM_WORK_BUDGET,
    force_vonbaeyer_spiro: bool = False,
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
        return _name_universal_substitutive_unsafe(
            mol, atom_work_budget, force_vonbaeyer_spiro=force_vonbaeyer_spiro)
    except Exception:
        return None


def _name_universal_substitutive_unsafe(
    mol, atom_work_budget: int, force_vonbaeyer_spiro: bool = False,
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
    built = _build_ctx(mol, atom_work_budget,
                       force_vonbaeyer_spiro=force_vonbaeyer_spiro)
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
    # resolved above) or (b) a P-59 internal / P-74.2.1 semipolar centre
    # actually rendered by a dedicated internal-charge leaf (nitro / azido /
    # diazo / the ``oxido``+``-ium`` N-/P-/S-oxide pair -- ``comp.internal_atoms``,
    # the internal-charge shapes this module can spell). This also closes
    # polynitro shredding: a torn-apart nitro is never rendered as a nitro leaf,
    # so its O(-) raw charge is unconsumed and trips the guard, voiding the whole
    # molecule (correct -- degrade to abstain, never mis-name).
    charge_ok = all_charge_ids | comp.internal_atoms
    for a in ctx.mol.GetAtoms():
        if a.GetFormalCharge() != 0 and a.GetIdx() not in charge_ok:
            return None  # unspellable internal charge -> void, never mis-name

    # Phase E: certify the atom->token partition through the SAME shared E1
    # core that certifies coverage elsewhere (invariant 12: extend, don't
    # duplicate), REPLACING the hand-rolled two-sided ``covers != heavy or
    # total_bound != len(heavy)`` self-check that lived here. That self-check
    # re-derived E1's P1 atom-partition -- GAP via ``covers != heavy``,
    # DOUBLE-COUNT via ``total_bound`` (fix round 1, finding 5) -- but LACKED
    # E1's token-in-name and element_soundness chemistry-soundness checks, which this now
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

    # v35 Track A -- STEREO: ``comp`` above is constitution-complete but its
    # ``name`` is stereo-blind (it was built with ``emit_branch_stereo`` off so
    # the E1/charge partition asserts run on the canonical constitution, never
    # a stereo-decorated token). ``_resolve_floor_stereo`` now runs the
    # cascade full -> top-only -> plain, shipping the first candidate that
    # FULL-InChIKey verifies. ``comp.bindings``/``covers`` are the plain
    # partition regardless of which name wins (stereo is a pure name-string
    # decoration, so the atom->token partition is identical).
    name = _resolve_floor_stereo(ctx, mol, heavy, comp, atom_work_budget)
    return UniversalResult(name=name, bindings=tuple(comp.bindings), covers=covers)


def _resolve_floor_stereo(ctx: "_Ctx", mol, heavy: FrozenSet[int],
                          comp: "_ComponentResult", atom_work_budget: int) -> str:
    """v35 Track A: pick the most stereo-complete floor name that FULL-InChIKey
    verifies, degrading gracefully. Returns the name string to ship.

    Cascade (WS-STEREO extended from top-spine-only to branch-recursive):

    1. **full** -- rebuild the component tree with ``emit_branch_stereo=True``
       so every recursion level writes its own branch's stereodescriptors
       (``(2R)-``/``(1E)-``) with that branch's OWN local locants (the model
       use), then prepend the top spine's own stereo
       block. Captures a stereocentre no matter how deep in a branch it sits.
    2. **top-only** -- prepend just the top spine's stereo block to the plain
       ``comp.name`` (the previous WS-STEREO behaviour): recovers the parent
       stereo when a branch centre is CIP-ambiguous and voids the full
       candidate.
    3. **plain** -- ``comp.name`` unchanged (today's block-1 fallback).

    0-WRONG-SAFE by construction: a candidate ships ONLY if it round-trips to
    the ORIGINAL input's FULL InChIKey (constitution AND stereo AND charge) via
    ``validation.reconstruct.verify_or_none``. A wrong/uncertain descriptor
    fails that gate and the cascade falls through -- it can only ever IMPROVE a
    name (block-1 -> full), never make one wrong. Up to two OPSIN verifies (the
    user-chosen cascade cost); ``plain`` needs none.
    """
    from ..validation.reconstruct import verify_or_none

    plain = comp.name
    # Achiral fast path: no defined stereocentre or stereo bond means every
    # cascade rung would equal ``plain``, so skip the second tree build AND the
    # OPSIN verifies entirely -- keeping the achiral cost identical to pre-v35
    # (``assign_stereochemistry`` already ran in ``_build_ctx``, so the tags are
    # set). This is the common case and must not pay the branch-stereo cost.
    m = ctx.mol
    has_stereo = any(
        a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms()
    ) or any(
        b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds()
    )
    if not has_stereo:
        return plain

    top_block = ""
    if comp.spine_atom_to_locant:
        top_block = _stereo_prefix(ctx.mol, comp.spine_atom_to_locant)

    # rung 1: full branch-recursive stereo (rebuild the tree, stereo on).
    # The plain build above already DEPLETED ``ctx.budget``; give the rebuild a
    # FRESH budget of the SAME allowance so a large-but-nameable molecule still
    # gets its stereo captured, and GUARD the whole build so a genuinely
    # budget-exhausting giant (``_BudgetExceeded``) -- or any other rebuild
    # failure -- falls through to rung2/plain instead of escaping to the
    # top-level ``except: return None`` and abstaining the molecule outright.
    # A stereo rebuild must never turn a valid plain name into silence (the
    # module's never-abstain, always-degrade contract).
    candidates: List[str] = []
    saved_budget = ctx.budget
    try:
        ctx.budget = _Budget(atom_work_budget)
        comp_full = _name_component(ctx, heavy, attach_hint=None, is_top=True,
                                   emit_branch_stereo=True)
    except Exception:
        comp_full = None
    finally:
        ctx.budget = saved_budget
    if comp_full is not None:
        full_name = (top_block + comp_full.name) if top_block else comp_full.name
        if full_name != plain:
            candidates.append(full_name)
    # rung 2: top-spine-only stereo (the previous WS-STEREO candidate)
    if top_block:
        top_name = top_block + plain
        if top_name != plain and top_name not in candidates:
            candidates.append(top_name)

    can = Chem.MolToSmiles(mol)
    for candidate in candidates:
        try:
            verified = verify_or_none(candidate, can)
        except Exception:
            verified = None  # fail-closed: never ship an unverified guess
        if verified is not None:
            return verified
    return plain


def _build_ctx(
    mol, atom_work_budget: int, force_vonbaeyer_spiro: bool = False,
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
    # v36 A2 (C2): the blanket "ANY radical electron -> void" guard used to
    # live here unconditionally, so a radical the dedicated
    # radicals.py/route_charged path DECLINES (e.g. it names a chain-hydride
    # -yl/-ylidene/-ylidyne primitive that itself falls through to a broken
    # textual fallback -- measured: '[CH2]CO' -> the unparseable 'ethanolyl')
    # never even got a chance at this floor. This ctx is now buildable for a
    # SINGLE P-71 monovalent/divalent/trivalent free-valence centre -- the
    # one shape ``name_universal_substituent_prefix``'s ``bond_order``
    # parameter (below) knows how to cite as a ``-yl``/``-ylidene``/
    # ``-ylidyne`` suffix at its own attachment locant. A SECOND radical
    # centre (P-71.2.3 multi-site, diradicals, radical ions, ...) stays out
    # of scope -- this floor's ``_render_as_substituent`` only ever cites ONE
    # attachment/bond-order pair, so a second centre would need a citation
    # this primitive cannot express; fail closed rather than silently drop
    # it. (``name_universal_substitutive``, the WHOLE-molecule entry point,
    # has NO such suffix logic at all and would name the radical as if it
    # were the fully saturated neutral species -- measured: '1-oxapropane'
    # for '[CH2]CO', which the caller's own OPSIN -r gate then rejects, so
    # letting THAT entry point build a ctx here costs nothing: it can only
    # ever fail the downstream verify, never ship wrong. The actual rescue is
    # wired at the caller, ``t4_coverage._best_effort_candidate``, which
    # routes a single free-valence centre through the substituent-prefix
    # entry point instead.)
    radical_atoms = [a for a in work.GetAtoms() if a.GetNumRadicalElectrons()]
    if len(radical_atoms) > 1:
        return None
    if radical_atoms and radical_atoms[0].GetNumRadicalElectrons() not in (1, 2, 3):
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
               cation_sites=cation_sites, anion_sites=anion_sites,
               force_vonbaeyer_spiro=force_vonbaeyer_spiro)
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
    # token-in-name + element_soundness over the old union/count self-check.
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
    charge_ok = comp.charged | comp.internal_atoms
    for idx in frag:
        if ctx.mol.GetAtomWithIdx(idx).GetFormalCharge() != 0 and idx not in charge_ok:
            return None

    return _render_as_substituent(comp, bond_order)


# ===========================================================================
# The recursive core
# ===========================================================================

def _name_component(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int], is_top: bool,
    emit_branch_stereo: bool = False,
) -> Optional[_ComponentResult]:
    """Name one component -- the whole molecule (``is_top``) or one branch.

    Termination: ``component`` shrinks strictly on every recursive call (see
    module docstring); the work budget is charged here, once per call, by
    component size, and fails closed on exhaustion.

    v35 Track A -- ``emit_branch_stereo``: when True, EVERY recursion level
    prepends its OWN spine's stereodescriptor block (``(2R)-``/``(1E)-`` ...)
    to each branch it renders as a substituent, so a stereocentre buried in a
    branch is expressed with the branch's own internal locants (the SAME
    numbering ``_render_as_substituent`` cites). Default False preserves the
    stereo-blind constitution name used for the E1/charge partition
    assertions; the caller rebuilds with True only to form a with-stereo
    candidate that is then FULL-InChIKey verified (see
    ``_resolve_floor_stereo``). The flag is threaded unchanged into the
    recursive call so it reaches every depth.
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
            # Task B2b fix round 1 + M2: record an internal-charge leaf's atoms
            # so the top-level raw-charge guard knows their P-59/P-74.2.1 internal
            # charges ARE accounted for (exception (b)). Each internal-charge leaf
            # token (nitro / azido / diazo) is produced by exactly one shortcut
            # and covers exactly the internal-charged atoms of that group, so this
            # token-set membership check is exact -- no OTHER leaf carries a
            # formal charge (the neutral leaves bail on any charged atom, and the
            # charged-terminal leaf uses its own ``charged`` field).
            internal_atoms = atoms if token in _INTERNAL_CHARGE_LEAF_TOKENS else frozenset()
            return _ComponentResult(
                name=token, bindings=[(token, atoms)], covers=atoms,
                attach_locant=None, is_prefix_ready=True,
                internal_atoms=internal_atoms,
            )

        # WS7 (v34 composed-charge): a lone charged TERMINAL atom -- the bare
        # ``[O-]``/``[S-]`` an FG anion (carboxylate/sulfonate/phosphonate/
        # alkoxide/thiolate) decomposes to once its ``=O`` is threaded into the
        # skeleton, or a terminal ``-NH3(+)`` -- is rendered with its CHARGED
        # substituent prefix (``oxido``/``sulfido``/``azaniumyl``) and its atom
        # recorded in ``charged`` so the top-level charge-coverage assertion
        # sees it accounted for (symmetric to the ``internal_atoms`` thread above).
        # This is the coverage-floor's own FG-anion/terminal-cation handler; a
        # SKELETAL (non-terminal) charge is still resolved by
        # ``_resolve_spine_charge`` below.
        charged_leaf = _charged_leaf_shortcut(mol, component, attach_hint)
        if charged_leaf is not None:
            token, atoms, charged_atoms = charged_leaf
            return _ComponentResult(
                name=token, bindings=[(token, atoms)], covers=atoms,
                attach_locant=None, is_prefix_ready=True,
                charged=charged_atoms,
            )

        # WS-NOABSTAIN class 3: a phosphinate P(=O)([O-])(H) unit (P-V, one
        # explicit P-H) -- see ``_phosphinate_oxide_leaf_shortcut``'s
        # docstring for why this MUST be a dedicated leaf rather than the
        # generic chain/hetero-prefix machinery: OPSIN does not validate the
        # equivalent raw replacement-chain spelling even with a correct
        # lambda-convention citation.
        phosphinate_leaf = _phosphinate_oxide_leaf_shortcut(mol, component, attach_hint)
        if phosphinate_leaf is not None:
            token, atoms, charged_atoms = phosphinate_leaf
            return _ComponentResult(
                name=token, bindings=[(token, atoms)], covers=atoms,
                attach_locant=None, is_prefix_ready=True,
                charged=charged_atoms,
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

    # ---- M2 inc4: internal semipolar-oxide cation(s) on THIS spine (N-oxide /
    # phosphine oxide / sulfoxide) get a paired ``-ium`` suffix; their ``[O-]``
    # renders as an ``oxido`` branch leaf. These are P-59/P-74.2.1 INTERNAL
    # charges get_ion_sites strips, so they are disjoint from the genuine-ion
    # charge above -- a spine that carries BOTH a genuine skeletal charge and an
    # internal oxide is a compound zwitterion out of this floor's scope (void).
    oxide_suffix_text, oxide_ids = _resolve_internal_oxide_suffix(
        ctx, spine_atom_to_locant)
    if oxide_suffix_text is not None and needs_charge:
        return None  # genuine charge + internal oxide on one spine: out of scope

    # ---- branches: every off-spine atom, named by RE-ENTERING this SAME
    # function on its own (strictly smaller) subgraph. NO depth cap. -------
    # Phase E (fix round 1): store the FULL, unstemmed ``spine_core`` token
    # (``hexane``/``cyclohexane``/``2-azapropane``...). E1's element_soundness
    # all-carbon classifier and P1 partition both need the true token --
    # element_soundness's grammar only recognises the FULL suffix form
    # (``hexane``->all-carbon, ``hexan``->unclassified), so a pre-stemmed
    # token would silently disable element_soundness for exactly the
    # alkane/cycloalkane class it exists to guard. The
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
    internal_accum: set = set(oxide_ids)  # Task B2b fix round 1 + M2: rendered
    #                              internal-charge atoms (nitro/azido/diazo/...)
    #                              plus M2 inc4 semipolar-oxide cations whose
    #                              ``-ium`` this spine appends (their ``[O-]``
    #                              adds itself as an ``oxido`` leaf below).
    prefix_entries: Dict[str, List[int]] = {}
    for s_atom, root, order, branch_atoms in _discover_branches(
        mol, spine_atoms, component,
    ):
        sub = _name_component(ctx, branch_atoms, attach_hint=root, is_top=False,
                              emit_branch_stereo=emit_branch_stereo)
        if sub is None:
            return None  # never ship a partial name: whole call voids
        rendered = _render_as_substituent(sub, order)
        # v35 Track A: prepend THIS branch's own spine stereo. ``sub.spine_
        # atom_to_locant`` numbers the branch's own atoms (None for a leaf
        # shortcut -> no block); deeper sub-branch stereo is already baked into
        # ``sub.name`` by the recursion above. ``alpha_sort_key`` strips the
        # descriptor (resolved P-14.5 noise-strip), so ordering is unaffected,
        # and ``format_substituent_prefix`` supplies the enclosing marks the
        # now-complex prefix needs (``4-[(3R)-3-hydroxy...yl]``).
        if emit_branch_stereo and sub.spine_atom_to_locant:
            branch_stereo = _stereo_prefix(ctx.mol, sub.spine_atom_to_locant)
            if branch_stereo:
                rendered = branch_stereo + rendered
        loc = spine_atom_to_locant[s_atom]
        prefix_entries.setdefault(rendered, []).append(loc)
        bindings.append((rendered, sub.covers))
        charged_accum |= sub.charged
        internal_accum |= sub.internal_atoms

    prefix_parts = []
    for name, locs in prefix_entries.items():
        # Prime-aware sort: a mixed-spiro-fused leaf hands us display-string
        # locants (``"5'"``) whose natural order is (prime_rank, number), not
        # lexicographic; ordinary int locants sort identically under this key.
        locs = sorted(locs, key=_locant_sort_key)
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

    # A genuine skeletal charge and an internal semipolar-oxide ``-ium`` are
    # mutually exclusive on one spine (the compound case voided above), so at most
    # one suffix applies. Appended to the FULLY assembled name (prefixes + parent
    # core), not just ``spine_core`` -- mirrors general_engine's own ordering
    # (``_append_charge_suffix`` is called on the substituent-decorated name), so
    # P-16.7.1(a)/P-74.1.1 elision targets the parent hydride's own trailing 'e',
    # never truncates a substituent prefix.
    ionic_suffix = charge_suffix_text if needs_charge else oxide_suffix_text
    if ionic_suffix is not None:
        full_name = _elide_before_ionic_suffix(full_name, ionic_suffix)

    covers = frozenset(a for _t, ids in bindings for a in ids)
    return _ComponentResult(
        name=full_name, bindings=bindings, covers=covers,
        attach_locant=attach_locant, charged=frozenset(charged_accum),
        internal_atoms=frozenset(internal_accum),
        spine_atom_to_locant=dict(spine_atom_to_locant),
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

    # WS7 (v34): ``parent_only=True`` -- this producer resolves charge PER SPINE
    # and expresses off-spine charges of the OTHER sign elsewhere (a terminal
    # ``[O-]``/``[NH3+]`` via ``_charged_leaf_shortcut``), so a SKELETAL cation
    # (or anion) on this spine gets its ``-ium``/``-ide`` suffix even in a
    # zwitterion whose opposite centre is a leaf on another branch. The top-level
    # charge-coverage assertion still proves every genuine ion atom was expressed
    # somewhere, so no charge is dropped.
    single_sign = _charge_suffix_text(ctx.mol, spine_atom_to_locant,
                                      parent_only=True)
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
    continuations -- see module docstring on chain/ring composition), an
    element ON the skeletal ALLOW-LIST (M1: ``_is_skeletal_spine_element`` --
    carbon or a ``REPLACEMENT_TERMS``-spellable heteroatom; the primary gate,
    generalizing the terminal-halogen exclusion), and
    NOT the nitrogen of a nitro group (Task B2b, ``_is_nitro_root``): nitro
    is always resolved as a branch via the dedicated ``_nitro_shortcut``
    leaf, never mechanically threaded into a replacement-nomenclature chain
    as if its nitrogen were an ordinary standard-valence heteroatom -- doing
    so would silently drop the +1/-1 charge information a plain locanted
    "aza" carries no trace of (see module docstring's charge-scope
    paragraph).

    WS7 (v34 composed-charge): likewise NOT a singly-charged TERMINAL atom (a
    degree-1 ``[O-]``/``[S-]``/``[NH3+]`` etc.). Such an atom is always resolved
    as a branch via ``_charged_leaf_shortcut`` (``oxido``/``sulfido``/
    ``azaniumyl``), which carries its charge -- threading it into the chain
    skeleton as a plain ``oxa``/``aza`` atom would silently drop that charge,
    exactly like the nitro case. A degree-1 atom is always a chain ENDPOINT, so
    excluding it only shortens the spine by that terminus; it never severs the
    backbone (an internal, non-terminal charge stays threadable and is handled
    by ``_resolve_spine_charge``).

    WS-NOABSTAIN class 3: likewise NOT a phosphinate P (``-PH(=O)[O-]``,
    ``_is_phosphinate_oxide_root``) -- see ``_phosphinate_oxide_leaf_shortcut``
    for why this group must be resolved as a branch (the ``phosphanyl``
    substituent form) rather than woven into the replacement-nomenclature
    chain as a skeletal ``phospha`` atom (measured: even WITH the correct
    P-15.4.1 lambda citation, OPSIN does not validate that shape)."""
    out = []
    for nb in ctx.mol.GetAtomWithIdx(atom).GetNeighbors():
        j = nb.GetIdx()
        if nb.GetAtomicNum() <= 1 or j not in component:
            continue
        if ctx.ring_system_of.get(j) is not None:
            continue
        if not _is_skeletal_spine_element(ctx.mol, j):
            continue  # ALLOW-LIST (M1): only carbon + a REPLACEMENT_TERMS-
            #            spellable heteroatom MAY thread the chain spine. Every
            #            other element (halogen, metal, noble gas, Al/Ga/In/Tl,
            #            an exotic-valence centre, ...) is COUNTED in the chain
            #            length but SILENTLY SKIPPED by _build_hetero_prefix
            #            (phantom carbon + dropped atom -> wrong constitution:
            #            CHF2 -> "1-fluoroethan-1-yl", commit d3590326 for the
            #            halogen instance). Sent instead to _discover_branches ->
            #            _leaf_shortcut (a monovalent halogen renders fluoro/
            #            chloro/bromo/iodo; an unspellable element fails closed).
            #            Generalizes the terminal-halogen exclusion to the whole
            #            non-skeletal class -- see _SKELETAL_SPINE_ELEMENTS.
        if _is_nitro_root(ctx.mol, j):
            continue
        if _is_azide_root(ctx.mol, j):
            continue  # M2: azide N_alpha is always the ``azido`` branch leaf,
            #            never threaded into the chain (its +1/-1 internal
            #            charge carries no trace as a plain ``aza`` -- nitro
            #            case, generalised). Ring path already branches it.
        if _is_diazo_root(ctx.mol, j):
            continue  # M2: diazo N_beta likewise -- always the ``diazo`` leaf.
        if _is_isocyano_root(ctx.mol, j):
            continue  # M2 inc3: the isocyanide N (R-[N+]#[C-]) is always the
            #            ``isocyano`` branch leaf, never threaded into the chain
            #            (its N+ would drop the internal charge as a plain aza).
        if _is_nitrooxy_root(ctx.mol, j):
            continue  # M2 inc2: the ester O of a nitrate ester -O-NO2 is always
            #            the anchor of the ``nitrooxy`` branch leaf (which carries
            #            the whole -O-N(+)(=O)[O-] group + its internal charges),
            #            never threaded into the chain as a plain skeletal ``oxa``
            #            (which would strand the nitro N+/O- as an unconsumed
            #            internal charge -> void -- the nitro case, generalised).
        if _is_phosphinate_oxide_root(ctx.mol, j):
            continue
        if _charged_leaf_shortcut(ctx.mol, frozenset({j}), atom) is not None:
            continue  # terminal charged atom a charged-leaf owns -> branch, not
            #            spine (a carbanion / heteroatom-hydride anion this leaf
            #            does NOT own stays threadable -> skeletal -ide/-uide)
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

def _locant_sort_key(loc) -> Tuple[int, int]:
    """Total order over a locant that may be a plain int, a primed display
    string (``"5'"``), or a ``rules.spiro._Locant`` tuple ``(5, "'")``.

    Sort is (prime_rank, number): all unprimed locants precede all
    single-primed, which precede double-primed, and within a rank by number.
    This is the ordering P-31.1.4 / OPSIN expect for a spiro locant set, and
    it lets the SAME ``sorted(...)`` in ``_name_component`` handle both the
    ordinary int-locant spines and the mixed-spiro-fused string-locant leaf.
    """
    if isinstance(loc, tuple):
        base, prime = loc[0], loc[1]
        return (len(prime), int(base))
    if isinstance(loc, str):
        i = 0
        while i < len(loc) and loc[i].isdigit():
            i += 1
        num = int(loc[:i]) if i else 0
        return (len(loc) - i, num)   # trailing primes = the suffix length
    return (0, int(loc))


def _locant_display(loc) -> str:
    """Render a locant (int / primed string / ``_Locant`` tuple) as the token
    that goes into the name: ``5`` -> ``"5"``, ``(5, "'")`` -> ``"5'"``,
    ``"5'"`` -> ``"5'"``."""
    if isinstance(loc, tuple):
        from ..rules.spiro import _prime_token
        return _prime_token(loc[0], loc[1])
    return str(loc)


def _mixed_spiro_fused_leaf(
    ctx: _Ctx, ring_atoms: FrozenSet[int], attach_hint: Optional[int],
) -> Optional[Tuple[FrozenSet[int], str, Dict[int, object], Optional[int]]]:
    """Best-effort FLOOR wiring of ``rules.spiro.name_mixed_spiro_fused`` as a
    ring-leaf parent (Task C, the reverted-L2 fall-through, re-anchored).

    A spiro atom that joins a FUSED ring component (indane / chromene /
    indoline / cyclopenta[b]pyridine ...) to a second ring is a
    ``mixed-spiro-fused`` system: it is neither a von-Baeyer cage
    (``analyze_cage_universal`` voids) nor a plain von-Baeyer spiro
    (``analyze_spiro_universal`` would kekulise the aromatic component into a
    polyene that OPSIN reads as a different constitution).
    ``name_mixed_spiro_fused`` builds the P-24.5.1 separable name
    (``spiro[<fused-comp>-x,y'-<comp2>]``) and returns a combined
    atom->locant map whose SECOND-cited component carries primed locants as a
    ``_Locant`` tuple ``(n, "'")``.

    We render each such tuple to its display string (``"5'"``) HERE, via
    ``_prime_token`` -- the L2 defect was leaking the raw ``(1, "'")`` tuple
    into ``format_substituent_prefix``. Downstream sorting is prime-aware
    (``_locant_sort_key``).

    Fail-closed: returns None (fall through to the von-Baeyer spiro floor)
    unless the named spiro CORE is exactly this component's ring system, so a
    molecule with a second, unrelated ring system is never mis-attributed.
    Every emission is still offer-RT-gated by the caller (0-wrong).
    """
    from ..rules.spiro import (
        name_mixed_spiro_fused, _name_general_monospiro_fused,
        _name_linear_polyspiro_fused,
    )
    fvb = ctx.force_vonbaeyer_spiro
    # ``restrict_atoms`` = THIS ring system: a molecule can hold several DISJOINT
    # spiro cores (two chain-bridged spiro-hydantoins, a spiro core plus a
    # spiro-substituent), and each ring system is named independently by the
    # per-ring-system recursion. Without this the namers saw the WHOLE molecule's
    # spiro atoms (>1) and voided every disjoint core as "multi-spiro".
    restrict = set(ring_atoms)
    try:
        # allow_vonbaeyer_component: the FLOOR additionally degrades a
        # spiro-of-bicyclic (the (c)-aliphatic bucket) to the P-24.5.1 separable
        # ``spiro[bicyclo[...]-x,y'-<comp2>]`` covering name. PIN path is
        # untouched (name_mixed_spiro_fused default keeps the flag off).
        # ``force_vonbaeyer`` (retry path): name a fused spiro component as its
        # faithful von-Baeyer polyene instead of the systematic fusion name, so
        # a molecule whose systematic name failed the offer RT gate (wrong /
        # unparseable fusion descriptor) still ships a 0-wrong covering name.
        res = name_mixed_spiro_fused(
            ctx.mol, allow_vonbaeyer_component=True,
            force_vonbaeyer_component=fvb, restrict_atoms=restrict)
        if res is None:
            # both-sides-fused fallback: a monospiro whose BOTH sides are
            # fused/bridged systems (name_mixed_spiro_fused requires one side to
            # be a single ring). Names each side independently and joins the
            # P-24.5.1 separable form. Floor-only; offer-RT-gated.
            res = _name_general_monospiro_fused(
                ctx.mol, allow_vonbaeyer=True, force_vonbaeyer=fvb,
                restrict_atoms=restrict)
        if res is None:
            # linear polyspiro fallback: dispiro/trispiro chain of fused/ring
            # components (the polyspiro (c)-bucket). Floor-only; offer-RT-gated.
            res = _name_linear_polyspiro_fused(
                ctx.mol, allow_vonbaeyer=True, force_vonbaeyer=fvb,
                restrict_atoms=restrict)
    except Exception:
        return None
    if res is None:
        return None
    name, core_ring_atoms, combined_locants, _subs = res
    # The spiro CORE named must be exactly the ring system we were asked to
    # spine (spiro-grouped ring system == fused component + side ring). If the
    # molecule carries another spiro/fused system, decline and let the caller's
    # von-Baeyer floor (or a deeper branch) handle this one.
    if set(core_ring_atoms) != set(ring_atoms):
        return None
    # Render the combined map to display-string locants (primes resolved).
    atom_to_locant: Dict[int, object] = {
        a: _locant_display(loc) for a, loc in combined_locants.items()
    }
    attach_locant = (
        atom_to_locant.get(attach_hint) if attach_hint is not None else None
    )
    return frozenset(core_ring_atoms), name, atom_to_locant, attach_locant


def _name_ring_spine(
    ctx: _Ctx, component: FrozenSet[int], ring_atoms: FrozenSet[int],
    attach_hint: Optional[int],
) -> Optional[Tuple[FrozenSet[int], str, Dict[int, int], Optional[int]]]:
    mol = ctx.mol
    ri = mol.GetRingInfo()
    sssr_here = [set(r) for r in ri.AtomRings() if set(r) <= ring_atoms]
    if len(sssr_here) >= 2:
        # WS-NOABSTAIN: this module is the unconditional best-effort FLOOR
        # (reachable only on the `general_fallback` path -- t4_coverage.py's
        # comment at its call site: "PIN is untouched"), so a fused/mancude
        # aromatic ring system (a charged flavonoid/isoflavone phenolate, a
        # protonated purine, or any plain aromatic bicycle no other producer
        # hosted) must degrade to the kekulized von-Baeyer polyene form
        # (`allow_mancude=True`, the SAME opt-in `_universal_cage_substituent_
        # name` already uses in `ring_substituents.py`) rather than refuse --
        # refusing here means the whole enclosing candidate silently abstains
        # instead of shipping an uglier, non-PIN, but round-trip-verified
        # name. Never reached at PIN tier, so PIN's own (correct) refusal for
        # a mancude cage is untouched.
        #
        # Task C: a spiro atom joining a FUSED ring component to a second ring
        # (indane / chromene / indoline / cyclopenta[b]pyridine spiro-...) is a
        # mixed-spiro-fused system that neither ``analyze_cage_universal`` nor
        # ``analyze_spiro_universal`` names correctly -- the former voids, the
        # latter kekulises the aromatic component into a polyene of a DIFFERENT
        # constitution. Offer the P-24.5.1 separable name FIRST; it is
        # offer-RT-gated by the caller, so a wrong-constitution fusion name is
        # voided rather than shipped. Only when it declines do we fall through
        # to the von-Baeyer spiro floor below (the (c)-bucket covering name).
        mixed = _mixed_spiro_fused_leaf(ctx, ring_atoms, attach_hint)
        if mixed is not None:
            return mixed
        cage = analyze_cage_universal(mol, cage_atoms=set(ring_atoms),
                                      allow_mancude=True)
        if cage is None:
            # Spiro ring systems are not von-Baeyer cages, so
            # ``analyze_cage_universal`` voids on them -- yet a spiro ring is a
            # perfectly nameable branch parent (``analyze_spiro_universal``
            # already produces its descriptor, hetero prefix and numbering for
            # ~74% of the spiro abstention residual). Without this fall-through
            # the unconditional core VOIDED on every spiro-ring branch (measured
            # 2026-08-30: spiro[2.3]hexane -> None), silently abstaining the
            # whole enclosing best-effort candidate. ``analyze_spiro_universal``
            # returns the SAME field shape as the cage (descriptor / total_atoms
            # / unsaturation / hetero_prefix / cage_atoms / atom_to_locant), so
            # the shared emission tail below is unchanged. ``free_valence_atoms``
            # biases the spiro numbering to give the attachment the lowest
            # locant (P-29.3, substituent use). best-effort FLOOR only (never
            # reached at PIN tier), so PIN's own spiro producers are untouched.
            fv = [attach_hint] if attach_hint is not None else None
            cage = analyze_spiro_universal(
                mol, cage_atoms=set(ring_atoms), allow_mancude=True,
                free_valence_atoms=fv)
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
    # ALLOW-LIST (M1), ring-builder side: this monocyclic namer spells its ring
    # heteroatoms through the SAME ``_build_hetero_prefix`` the chain spine uses,
    # which SILENTLY SKIPS any element not in REPLACEMENT_TERMS -> a non-skeletal
    # ring atom (a halogen ring / λ-halane such as iodinane ``I1CCCCC1``, a
    # metallacycle, ...) would be rendered as a phantom CARBOcycle, a wrong
    # constitution. Fail CLOSED (void -> abstain) instead, symmetric to the
    # chain walker/builder guards -- such a ring is unnameable by this floor
    # anyway (no 'ioda'/metal morpheme), so voiding only converts a
    # phantom-carbon name into a clean abstention, never loses a correct name.
    if any(not _is_skeletal_spine_element(mol, a) for a in ring_tuple):
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
        # Seed the diameter from the LARGEST spine-walkable piece, not from a
        # global canonical-rank minimum. The walkable graph is a forest once the
        # off-spine leaf roots (nitro/azide/diazo/isocyanide/nitrooxy/phosphinate)
        # are held off-spine, and an internal-charge leaf can STRAND a skeletal
        # carbon in its own singleton piece -- e.g. the carbon of an isocyanide
        # ``R-[N+]#[C-]`` whose only neighbour is the off-spine isocyanide N.
        # Seeding there would pick that carbon as a degenerate 1-atom "methane"
        # parent and thread the REAL chain (R) in as a charge-dropping azachain
        # branch (measured: butyl isocyanide -> ``1-(1-azapentan-1-ylidyne)methane``,
        # which voids). Choosing the senior (largest) piece makes R the parent and
        # the small group falls out as its proper branch leaf. No-op when the
        # walkable graph is connected -- the overwhelmingly common case.
        eligible = component - _offspine_leaf_atoms(mol, component)
        if not eligible:
            eligible = component  # degenerate: whole graph is a leaf group
        pieces = _walkable_pieces(ctx, eligible, component)
        main = max(pieces,
                   key=lambda p: (len(p), -min(ctx.canon_rank[a] for a in p)))
        seed = min(main, key=lambda a: ctx.canon_rank[a])
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
    # ALLOW-LIST (M1), builder side: a chain spine may be built ONLY from
    # skeletal elements (carbon + a REPLACEMENT_TERMS-spellable heteroatom).
    # ``_tree_neighbors`` already keeps a non-skeletal atom from being THREADED
    # into a multi-atom spine, but a non-skeletal atom can still arrive here as
    # the FORCED ROOT of a size-1 (or root-only) branch component -- e.g. a
    # ``C=I`` iodine ylidene, whose double-bonded I matches no ``_LEAF_DOUBLE``
    # leaf and so falls through to here. Building a 1-atom "chain" from it would
    # SILENTLY DROP the atom in ``_build_hetero_prefix`` (I not in
    # REPLACEMENT_TERMS) and render a PHANTOM ``methane`` -- a wrong
    # constitution. Fail CLOSED instead (void -> abstain), so the whole
    # non-skeletal class fails closed uniformly rather than a leaf-shortcut
    # subset failing closed and the ylidene/exotic-valence tail leaking a
    # phantom carbon. A legitimate chain path is all-skeletal, so this never
    # fires on a real name.
    if any(not _is_skeletal_spine_element(mol, a) for a in path):
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


def _is_offspine_root(mol, j: int) -> bool:
    """True if atom *j* is the anchor of an internal-charge / semipolar leaf
    group that is ALWAYS resolved as an off-spine branch (nitro / azide / diazo /
    isocyanide / nitrate-ester / phosphinate). Such an atom is never a chain-spine
    member, so ``_walkable_pieces`` treats it as a piece BOUNDARY -- it does not
    expand a spine-walk through it (mirroring ``_tree_neighbors`` keeping these
    off-spine, but applied to the SOURCE so a piece cannot leak across one)."""
    return (_is_nitro_root(mol, j) or _is_azide_root(mol, j)
            or _is_diazo_root(mol, j) or _is_isocyano_root(mol, j)
            or _is_nitrooxy_root(mol, j) or _is_phosphinate_oxide_root(mol, j))


def _offspine_leaf_atoms(mol, component: FrozenSet[int]) -> FrozenSet[int]:
    """Every atom in *component* that belongs to an internal-charge / semipolar
    leaf group (nitro / azide / diazo / isocyanide / nitrate-ester / phosphinate)
    -- i.e. that will be claimed WHOLE by a branch leaf and must therefore never
    be a chain-spine member OR a spine seed. The isocyanide is the sharp case: its
    terminal CARBON is a skeletal atom stranded off the excluded isocyanide N, so
    without this exclusion it could seed a degenerate 1-atom parent (see
    ``_name_chain_spine``); the nitrate ester's own N/O likewise form a spurious
    2-atom piece larger than a one-carbon parent."""
    leaf: set = set()
    for j in component:
        atoms = None
        if _is_azide_root(mol, j):
            atoms = _azide_nitrogens(mol, j)
        elif _is_diazo_root(mol, j):
            atoms = _diazo_nitrogens(mol, j)
        elif _is_isocyano_root(mol, j):
            atoms = _isocyanide_atoms(mol, j)
        elif _is_nitrooxy_root(mol, j):
            atoms = _nitrooxy_atoms(mol, j)
        elif _is_nitro_root(mol, j):
            atoms = frozenset({j} | {n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()
                                     if n.GetSymbol() == "O"})
        elif _is_phosphinate_oxide_root(mol, j):
            atoms = frozenset({j} | {n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()
                                     if n.GetSymbol() == "O"})
        if atoms:
            leaf |= set(atoms)
    return frozenset(leaf & component)


def _walkable_pieces(ctx: _Ctx, eligible: FrozenSet[int],
                     component: FrozenSet[int]) -> List[set]:
    """Partition the spine-eligible atoms *eligible* into spine-walkable connected
    pieces. Two atoms are in the same piece iff a chain-spine walk
    (``_tree_neighbors``, over the full *component*) connects them through
    eligible atoms only, without crossing an off-spine leaf root
    (``_is_offspine_root``). Deterministic (canonical-rank visit order). Returns
    one piece for a connected walkable graph (the common case), so the caller's
    seed choice is unchanged there."""
    seen: set = set()
    pieces: List[set] = []
    for a in sorted(eligible, key=lambda x: ctx.canon_rank[x]):
        if a in seen:
            continue
        piece: set = set()
        stack = [a]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            piece.add(x)
            if _is_offspine_root(ctx.mol, x):
                continue  # boundary: never continue a spine THROUGH this atom
            for nb in _tree_neighbors(ctx, x, component):
                if nb in eligible and nb not in seen:
                    stack.append(nb)
        pieces.append(piece)
    return pieces


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
    # NOTE (WS-NOABSTAIN class 3): a lambda-convention citation was tried
    # here first (a skeletal P/S atom whose actual bonding number exceeds
    # its standard one, e.g. a phosphinate's P or a sulfonic-acid-in-chain
    # S). REVERTED: measured regressions on multiple ALREADY-CORRECT,
    # OPSIN-verified existing names that carry such an atom WITHOUT a lambda
    # citation (e.g. "...2-oxo-1,3-dioxa-2-thiahepta-1,5-diene" for a
    # sulfonic-acid chain, "...3-phosphaoctan-7-ium-1-yl..." for an
    # ammonium-ester phosphate chain) -- OPSIN already round-trips those
    # correctly WITHOUT lambda via its own valence-filling on the SUFFIX/
    # charge-suffix-adjacent atom, so adding one only changed working
    # spelling for no gain. The actual class-3 fix is
    # ``_phosphinate_oxide_leaf_shortcut`` + ``_is_phosphinate_oxide_root``,
    # which divert the ONE shape that genuinely needed different handling
    # (a phosphinate P) OUT of this chain-threading path entirely before it
    # ever reaches here -- so this function stays exactly as it always was.
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


# M2: leaf tokens whose atoms carry a P-59 internal / P-74.2.1 semipolar formal
# charge that the leaf itself spells (so ``get_ion_sites`` strips those atoms and
# they never reach the genuine-ion charge machinery). ``_name_component`` records
# such a leaf's atoms in ``_ComponentResult.internal_atoms`` so the top-level and
# prefix-path raw-formal-charge void guards know the charge IS accounted for. Each
# token is produced by exactly one shortcut and covers exactly that group's
# internally-charged atoms -- see the membership check in ``_name_component``.
_INTERNAL_CHARGE_LEAF_TOKENS = frozenset(
    {"nitro", "azido", "diazo", "nitrooxy", "isocyano", "oxido"})


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

    # M2: the other P-59 internal centres this module can spell as a TERMINAL
    # substituent leaf -- azide (``azido``) and diazo (``diazo``). Like nitro,
    # their charges are internal (get_ion_sites strips them), so they are
    # checked HERE, before the generic charge guard below, and their atoms are
    # tracked as ``internal_atoms`` (via ``_INTERNAL_CHARGE_LEAF_TOKENS``).
    azide = _azide_shortcut(mol, component, attach_hint)
    if azide is not None:
        return azide
    diazo = _diazo_shortcut(mol, component, attach_hint)
    if diazo is not None:
        return diazo
    # M2 inc2: the nitrate ester ``-O-[N+](=O)[O-]`` (P-67.1.4.3.1 preselected
    # ``nitrooxy``; its N+/O- are P-59 internal charges get_ion_sites strips).
    # Attached via the ester O, exactly the four ester-O/N/=O/O- atoms.
    nitrooxy = _nitrooxy_shortcut(mol, component, attach_hint)
    if nitrooxy is not None:
        return nitrooxy
    # M2 inc3: the isocyanide ``R-[N+]#[C-]`` (P-66.5.3 ``isocyano``; R-N=C has
    # no uncharged depiction, so its C-/N+ are P-59 internal -- reclassified as
    # such in ``perception.ions`` so ``get_ion_sites`` strips them here).
    isocyano = _isocyano_shortcut(mol, component, attach_hint)
    if isocyano is not None:
        return isocyano
    # M2 inc4: the anionic oxygen of a P-74.2.1 semipolar oxide (aromatic /
    # amine N-oxide, phosphine oxide, sulfoxide) -- a lone terminal ``[O-]``
    # single-bonded to an ``N+``/``P+``/``S+`` -- rendered as the ``oxido``
    # prefix (its INTERNAL charge, paired with the cation's ``-ium`` suffix the
    # spine appends; NOT the genuine-anion ``oxido`` of _charged_leaf_shortcut,
    # which routes a charge-carrying alkoxide/carboxylate via a different path).
    semipolar_oxido = _semipolar_oxido_shortcut(mol, component, attach_hint)
    if semipolar_oxido is not None:
        return semipolar_oxido

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


# WS7 (v34 composed-charge): charged terminal-atom substituent prefixes
# (P-72 anionic / P-73 cationic substituent prefixes). In THIS module's
# skeletal-replacement construction the carbonyl/sulfonyl/phosphoryl ``=O`` of a
# carboxylate/sulfonate/phosphonate is THREADED INTO the parent skeleton (as an
# ``oxa``/etc. atom + ``-ene``), which leaves the anionic ``[O-]`` as a bare,
# single-bonded TERMINAL branch -- structurally identical to an alkoxide/
# phenolate O(-). So every FG anion this module meets decomposes to a lone
# charged terminal atom, and the ONE root-cause fix for the whole class is to
# render that atom with its CHARGED substituent prefix (``oxido`` for ``-O(-)``,
# ``sulfido`` for ``-S(-)``, ``azaniumyl`` for a terminal ``-NH3(+)``) rather
# than bailing to the neutral ``hydroxy``/``amino`` leaf (which drops the charge
# -> a different, neutral molecule -- the exact hazard the ``_leaf_shortcut``
# charge-guard exists to prevent) or voiding. Each prefix was verified to
# OPSIN-round-trip WITH its charge (all confirmed against OPSIN 2.9.0:
# ``2-oxido-1-oxabut-1-ene`` -> propanoate anion, ``1-sulfidoethane`` ->
# ethanethiolate anion, ``1-selenidoethane`` -> ethaneselenolate anion,
# ``2-azaniumyl-1-oxidoethane`` -> the 2-aminoethanolate zwitterion); the
# emission is still gated by the caller's SELF-01 round-trip net, so an
# unverifiable candidate abstains rather than shipping wrong. (NB the Se prefix
# is ``selenido``, NOT ``selanido`` -- OPSIN rejects the latter.)
_ANION_LEAF_SINGLE = {"O": "oxido", "S": "sulfido", "Se": "selenido"}


def _charged_leaf_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """A lone, singly-charged TERMINAL atom named directly as its charged
    substituent prefix. Returns ``(token, atom_ids, charged_atom_ids)`` or
    ``None`` (fall through to generic construction).

    Scope (tight, all validated by the caller's round-trip net):
      * exactly one atom, formal charge +/-1, a SINGLE bond to its one outside
        (parent) neighbour;
      * ANION (-1): a chalcogen ``O``/``S``/``Se`` with no attached H
        -> ``oxido``/``sulfido``/``selenido`` (all OPSIN-RT-verified);
      * CATION (+1): a protonated primary ammonium -- a terminal ``-NH3(+)``,
        i.e. a degree-1 N(+) carrying EXACTLY 3 H -> ``azaniumyl``. The strict
        3-H count is deliberate: it excludes a 2-H terminal N(+) (an
        aminylium/nitrenium ``R-NH2(+)``, a DIFFERENT species ``azaniumyl``
        would mis-spell); such a centre falls through and voids/gate-rejects.

    Anything else (a doubly-charged atom, a =/# multiple-bond attachment, a
    carbanion, a bare halide, a substituted onium that is not a bare terminal
    atom) returns ``None`` and is handled by the generic spine/charge machinery
    or voids there -- never mis-spelled here."""
    if len(component) != 1:
        return None
    idx = next(iter(component))
    atom = mol.GetAtomWithIdx(idx)
    charge = atom.GetFormalCharge()
    if abs(charge) != 1:
        return None
    outside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in component]
    if len(outside) != 1:
        return None  # not a terminal atom (0 or >=2 heavy neighbours)
    bond = mol.GetBondBetweenAtoms(idx, outside[0])
    if bond is None or round(bond.GetBondTypeAsDouble()) != 1:
        return None  # only a single-bond attachment carries these prefixes
    sym = atom.GetSymbol()
    if charge == -1:
        if atom.GetTotalNumHs() != 0:
            return None
        token = _ANION_LEAF_SINGLE.get(sym)
        if token is None:
            return None
        return token, frozenset(component), frozenset(component)
    # charge == +1: a terminal protonated PRIMARY ammonium (-NH3+, exactly 3 H)
    # -> azaniumyl. A 2-H terminal N+ (aminylium/nitrenium) is a different
    # species and must NOT take this token; it falls through and voids.
    if sym == "N" and atom.GetTotalNumHs() == 3:
        return "azaniumyl", frozenset(component), frozenset(component)
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


# --- skeleton-membership ALLOW-LIST (M1: the composition-correctness lever) --
#
# WHICH elements MAY thread the chain spine, as a closed ALLOW-LIST rather than
# a growing list of per-element/per-shape EXCLUSIONS (halogen, ...). Derived
# DIRECTLY from ``REPLACEMENT_TERMS`` -- the same table ``_build_hetero_prefix``
# uses to SPELL a skeletal heteroatom -- plus carbon. This tie is the whole
# point and the 0-wrong-critical invariant:
#
#   * Every element ``_build_hetero_prefix`` can spell IS in the allow-list, so
#     the allow-list can NEVER drop a legitimate skeletal heteroatom (which
#     would regress a correct name) -- membership and spellability are the same
#     set BY CONSTRUCTION, they cannot drift apart.
#   * Every element it CANNOT spell (a halogen, a metal, a noble gas, Al/Ga/In/
#     Tl, At, an exotic-valence centre, ...) is OUT. Such an atom threaded into
#     the spine was COUNTED in the chain length (a phantom carbon) yet SILENTLY
#     SKIPPED by ``_build_hetero_prefix`` (``if sym in REPLACEMENT_TERMS``) --
#     i.e. dropped, yielding a name of a DIFFERENT constitution. The terminal
#     halogen was one instance (CHF2 -> ``1-fluoroethan-1-yl``, commit
#     d3590326); this allow-list closes the WHOLE class in one check.
#
# Excluded atoms become off-spine substituent branches (``_discover_branches``
# -> ``_leaf_shortcut``): a monovalent halogen renders as ``fluoro``/``chloro``/
# ``bromo``/``iodo`` (P-29.3 / P-35.1); an element with no leaf/branch spelling
# fails CLOSED (the whole call voids -> abstain), never a phantom-carbon name.
# ``SKELETON_ATOMS = {"C", *REPLACEMENT}`` and # carbon-only spine -- SAME pattern, Orthonym's own (organic-only) replacement
# table (organometallic 'a'-replacement is out of scope, P-69). Blue Book
# P-15.4 / P-21 ('a'-replacement skeletal-element set).
_SKELETAL_SPINE_ELEMENTS = frozenset({"C"}) | frozenset(REPLACEMENT_TERMS)


def _is_skeletal_spine_element(mol, j: int) -> bool:
    """True if atom *j*'s element MAY be a chain-spine ('a'-replacement)
    skeletal atom -- carbon or a heteroatom ``REPLACEMENT_TERMS`` (and hence
    ``_build_hetero_prefix``) can actually name. Used by ``_tree_neighbors`` as
    the primary spine-membership gate: an element NOT in this set is kept OUT of
    chain-spine continuation and resolved as an off-spine branch instead, so it
    can never be absorbed into the parent skeleton as a phantom carbon (the
    silent-drop mechanism that mis-rendered every non-skeletal terminal atom;
    see ``_SKELETAL_SPINE_ELEMENTS``)."""
    return mol.GetAtomWithIdx(j).GetSymbol() in _SKELETAL_SPINE_ELEMENTS


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


# ===========================================================================
# M2: organic azide  -N=[N+]=[N-]  (<-> resonance twin  -[N-]-[N+]#N)
# ===========================================================================

def _azide_nitrogens(mol, alpha_idx: int):
    """If *alpha_idx* is the ATTACHMENT nitrogen (N_alpha) of a terminal organic
    azide ``R-N=[N+]=[N-]`` -- or its charge-separated resonance twin
    ``R-[N-]-[N+]#N``, which RDKit does not normalise -- return the frozenset of
    its three nitrogen indices ``{alpha, beta, gamma}``, else ``None``.

    The centre is defined by CONNECTIVITY + CHARGE, not one literal bond-order
    pattern (so both drawings match): N_alpha has exactly one N neighbour
    (N_beta, the central nitrogen, always ``+1`` in an azide); N_beta has exactly
    two N neighbours (alpha, gamma) and no other heavy neighbour and no H; the
    terminal N_gamma is heavy-degree 1 (only bonded to beta) and H-free; and the
    end nitrogens {alpha, gamma} carry ``{0, -1}`` so the group is net-neutral.
    Nitro/nitrogen-hydride/azide-anion shapes fail one of these exactly."""
    a = mol.GetAtomWithIdx(alpha_idx)
    if a.GetSymbol() != "N" or a.GetTotalNumHs() != 0:
        return None
    a_n = [n for n in a.GetNeighbors() if n.GetSymbol() == "N"]
    if len(a_n) != 1:
        return None
    beta = a_n[0]
    if beta.GetFormalCharge() != 1 or beta.GetTotalNumHs() != 0:
        return None
    b_n = [n for n in beta.GetNeighbors() if n.GetSymbol() == "N"]
    b_heavy = [n for n in beta.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(b_n) != 2 or len(b_heavy) != 2:
        return None  # central N must bond exactly alpha + gamma, nothing else
    gamma = next((n for n in b_n if n.GetIdx() != alpha_idx), None)
    if gamma is None:
        return None
    g_heavy = [n for n in gamma.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(g_heavy) != 1 or g_heavy[0].GetIdx() != beta.GetIdx():
        return None  # gamma must be the true terminus
    if gamma.GetTotalNumHs() != 0:
        return None
    if sorted([a.GetFormalCharge(), gamma.GetFormalCharge()]) != [-1, 0]:
        return None
    return frozenset({alpha_idx, beta.GetIdx(), gamma.GetIdx()})


def _is_azide_root(mol, j: int) -> bool:
    """True if atom *j* is an azide's ATTACHMENT nitrogen (see
    ``_azide_nitrogens``) -- used by ``_tree_neighbors`` to keep the azide out
    of ordinary chain-spine continuation, so its three nitrogens are always
    resolved as ONE branch by ``_azide_shortcut`` (the ``azido`` leaf) rather
    than threaded into a replacement-nomenclature chain as plain ``aza`` atoms
    (which would silently drop the +1/-1 internal charge -- the nitro case,
    generalised)."""
    return _azide_nitrogens(mol, j) is not None


def _azide_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """``-N=[N+]=[N-]`` (or its resonance twin) attached via N_alpha, exactly
    the three azide nitrogens -- rendered as the ``azido`` substituent prefix
    (P-61.5 / P-59 internal charge; OPSIN-RT-verified: ``azidobenzene`` and
    ``1-azidohexane`` both parse back to the input constitution). Returns
    ``("azido", atom_ids)`` or ``None`` (fall through)."""
    if len(component) != 3:
        return None
    az = _azide_nitrogens(mol, attach_hint)
    if az is None or az != component:
        return None
    # attach_hint must be the real attachment: one heavy neighbour OUTSIDE the
    # azide (the parent R). This also rejects a bare HN3-style fragment.
    outside = [n for n in mol.GetAtomWithIdx(attach_hint).GetNeighbors()
               if n.GetAtomicNum() > 1 and n.GetIdx() not in component]
    if len(outside) != 1:
        return None
    return "azido", frozenset(component)


# ===========================================================================
# M2: diazo  >C=[N+]=[N-]  (<-> resonance twin  >C(-)-[N+]#N)
# ===========================================================================

def _diazo_nitrogens(mol, beta_idx: int):
    """If *beta_idx* is the ATTACHMENT nitrogen (N_beta) of a terminal diazo
    group ``>C=[N+]=[N-]`` -- or its resonance twin ``>C(-)-[N+]#N`` -- return
    the frozenset ``{beta, gamma}`` of its two nitrogens, else ``None``.

    N_beta is the central ``+1`` nitrogen doubly/singly bonded to the parent
    carbon and to the terminal N_gamma; both nitrogens are heavy-degree bounded
    to the diazo skeleton (beta: one C + gamma; gamma: only beta) and H-free,
    and the group is net-neutral (gamma is ``-1``, or ``0`` in the twin where
    beta stays ``+1`` and the carbon carries the ``-1`` -- handled by requiring
    beta ``+1`` and beta+gamma summing with the parent to zero locally)."""
    beta = mol.GetAtomWithIdx(beta_idx)
    if beta.GetSymbol() != "N" or beta.GetFormalCharge() != 1 or beta.GetTotalNumHs() != 0:
        return None
    heavy = [n for n in beta.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(heavy) != 2:
        return None
    n_nbr = [n for n in heavy if n.GetSymbol() == "N"]
    c_nbr = [n for n in heavy if n.GetSymbol() == "C"]
    if len(n_nbr) != 1 or len(c_nbr) != 1:
        return None
    gamma = n_nbr[0]
    g_heavy = [n for n in gamma.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(g_heavy) != 1 or g_heavy[0].GetIdx() != beta_idx or gamma.GetTotalNumHs() != 0:
        return None
    # net-neutral over the two-N terminus: gamma is the -1 end (canonical
    # drawing) or 0 (twin, where the parent carbon holds the -1). Either way the
    # two nitrogens carry a total of 0 or -1; the parent-carbon charge closes it.
    if gamma.GetFormalCharge() not in (-1, 0):
        return None
    return frozenset({beta_idx, gamma.GetIdx()})


def _is_diazo_root(mol, j: int) -> bool:
    """True if atom *j* is a diazo group's attachment nitrogen (see
    ``_diazo_nitrogens``) -- used by ``_tree_neighbors`` to keep it off the
    chain spine (its ``+1`` N would otherwise thread in as a plain ``aza`` atom,
    dropping the internal charge), so the two diazo nitrogens are always
    resolved as one ``diazo`` branch leaf."""
    return _diazo_nitrogens(mol, j) is not None


def _diazo_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """``=[N+]=[N-]`` (or its resonance twin) attached via N_beta to a parent
    carbon -- rendered as the ``diazo`` substituent prefix (P-66.4.1.2.1; OPSIN-
    RT-verified: ``ethyl 2-diazoacetate`` and ``(diazomethyl)benzene`` both parse
    back to the input). Returns ``("diazo", atom_ids)`` or ``None``.

    ``diazo`` is prefix-ready (``is_prefix_ready``) and already encodes its
    double-bond attachment, so ``_render_as_substituent`` returns it unchanged
    regardless of the discovered bond order (mirrors ``oxo``/``nitro``)."""
    if len(component) != 2:
        return None
    dz = _diazo_nitrogens(mol, attach_hint)
    if dz is None or dz != component:
        return None
    # the parent carbon must be OUTSIDE this two-atom branch
    outside = [n for n in mol.GetAtomWithIdx(attach_hint).GetNeighbors()
               if n.GetAtomicNum() > 1 and n.GetIdx() not in component]
    if len(outside) != 1 or outside[0].GetSymbol() != "C":
        return None
    return "diazo", frozenset(component)


# ===========================================================================
# M2 inc2: nitrate ester  -O-[N+](=O)[O-]  (rendered ``nitrooxy``, P-67.1.4.3.1)
# ===========================================================================

def _nitrooxy_atoms(mol, ester_o_idx: int):
    """If *ester_o_idx* is the ester oxygen of a nitrate ester
    ``R-O-[N+](=O)[O-]`` (the ester of nitric acid), return the frozenset of its
    four atoms ``{ester_O, N, =O, O-}``, else ``None``.

    The centre is defined by CONNECTIVITY + CHARGE (mirroring ``_nitro_shortcut``
    but anchored at the ESTER oxygen, one atom further out): the ester O is a
    neutral, H-free, degree-2 oxygen bonded to exactly one nitrogen (the nitrate
    N) and one other heavy atom (the parent R); that nitrogen is a genuine nitro
    nitrogen -- ``+1``, degree 3, its other two neighbours both terminal oxygens,
    one ``=O`` neutral and one single-bonded ``[O-]``. A plain nitro (no ester O),
    a nitrite ester ``-O-N=O``, or an orthonitrate fail one of these exactly."""
    o = mol.GetAtomWithIdx(ester_o_idx)
    if o.GetSymbol() != "O" or o.GetFormalCharge() != 0 or o.GetTotalNumHs() != 0:
        return None
    heavy = [n for n in o.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(heavy) != 2:
        return None
    n_nbrs = [n for n in heavy if n.GetSymbol() == "N"]
    r_nbrs = [n for n in heavy if n.GetSymbol() != "N"]
    if len(n_nbrs) != 1 or len(r_nbrs) != 1:
        return None
    n = n_nbrs[0]
    # the ester O -> N bond must be a single bond (an ester linkage)
    bond_on = mol.GetBondBetweenAtoms(ester_o_idx, n.GetIdx())
    if bond_on is None or round(bond_on.GetBondTypeAsDouble()) != 1:
        return None
    if n.GetFormalCharge() != 1 or n.GetTotalNumHs() != 0 or n.GetDegree() != 3:
        return None
    other_o = [nb for nb in n.GetNeighbors() if nb.GetIdx() != ester_o_idx]
    if len(other_o) != 2:
        return None
    kinds = []
    for oa in other_o:
        if oa.GetSymbol() != "O" or oa.GetDegree() != 1:
            return None  # each terminal O must bond ONLY to this N
        bond = mol.GetBondBetweenAtoms(n.GetIdx(), oa.GetIdx())
        bt = round(bond.GetBondTypeAsDouble())
        kinds.append((bt, oa.GetFormalCharge(), oa.GetTotalNumHs()))
    if sorted(kinds) != sorted([(2, 0, 0), (1, -1, 0)]):
        return None
    return frozenset({ester_o_idx, n.GetIdx()} | {oa.GetIdx() for oa in other_o})


def _is_nitrooxy_root(mol, j: int) -> bool:
    """True if atom *j* is the ester oxygen of a nitrate ester (see
    ``_nitrooxy_atoms``) -- used by ``_tree_neighbors`` to keep it OUT of chain-
    spine continuation, so the whole ``-O-NO2`` group is always resolved as ONE
    ``nitrooxy`` branch leaf (carrying the nitro N+/O- internal charge), never
    threaded into the skeleton as a plain ``oxa`` that strands the charge."""
    return _nitrooxy_atoms(mol, j) is not None


def _nitrooxy_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """``-O-[N+](=O)[O-]`` attached via the ester O, exactly the four nitrate-
    ester atoms -- rendered as the ``nitrooxy`` substituent prefix (P-67.1.4.3.1
    preselected; OPSIN-RT-verified: ``nitrooxymethane`` and ``1-(nitrooxy)ethane``
    both parse back to the input constitution). Returns ``("nitrooxy", atom_ids)``
    or ``None`` (fall through). The N+/O- are P-59 internal charges (tracked as
    ``internal_atoms`` via ``_INTERNAL_CHARGE_LEAF_TOKENS``)."""
    if len(component) != 4:
        return None
    nit = _nitrooxy_atoms(mol, attach_hint)
    if nit is None or nit != component:
        return None
    # attach_hint (the ester O) must have exactly one heavy neighbour OUTSIDE the
    # group -- the real parent R (also rejects a bare, unattached nitrate ion).
    outside = [n for n in mol.GetAtomWithIdx(attach_hint).GetNeighbors()
               if n.GetAtomicNum() > 1 and n.GetIdx() not in component]
    if len(outside) != 1:
        return None
    return "nitrooxy", frozenset(component)


# ===========================================================================
# M2 inc3: isocyanide  R-[N+]#[C-]  (rendered ``isocyano``, P-66.5.3)
# ===========================================================================

def _isocyanide_atoms(mol, n_idx: int):
    """If *n_idx* is the ATTACHMENT nitrogen of an isocyanide ``R-[N+]#[C-]``,
    return the frozenset ``{N, C}`` of its two atoms, else ``None``.

    Defined by CONNECTIVITY + CHARGE (mirroring the azide/diazo leaves): the N is
    ``+1``, H-free, bonded to exactly one carbon that is the terminal isocyanide
    carbon (``-1``, heavy-degree 1, triple-bonded to the N) plus exactly one
    other heavy atom (the parent R). A nitrilium ``R-C#[N+]-R`` (its C is not
    ``-1``) or a cyanide ``[C-]#N`` (its N is not ``+1``) fail this exactly."""
    n = mol.GetAtomWithIdx(n_idx)
    if n.GetSymbol() != "N" or n.GetFormalCharge() != 1 or n.GetTotalNumHs() != 0:
        return None
    heavy = [nb for nb in n.GetNeighbors() if nb.GetAtomicNum() > 1]
    if len(heavy) != 2:
        return None
    c_term = [nb for nb in heavy
              if nb.GetSymbol() == "C" and nb.GetFormalCharge() == -1]
    if len(c_term) != 1:
        return None
    c = c_term[0]
    c_heavy = [nb for nb in c.GetNeighbors() if nb.GetAtomicNum() > 1]
    if len(c_heavy) != 1 or c_heavy[0].GetIdx() != n_idx or c.GetTotalNumHs() != 0:
        return None  # the isocyanide carbon must be the true terminus
    bond = mol.GetBondBetweenAtoms(n_idx, c.GetIdx())
    if bond is None or round(bond.GetBondTypeAsDouble()) != 3:
        return None
    return frozenset({n_idx, c.GetIdx()})


def _is_isocyano_root(mol, j: int) -> bool:
    """True if atom *j* is the attachment nitrogen of an isocyanide (see
    ``_isocyanide_atoms``) -- used by ``_tree_neighbors`` to keep it OUT of
    chain-spine continuation, so the two isocyanide atoms are always resolved as
    ONE ``isocyano`` branch leaf (carrying their P-59 internal charge), never
    threaded into the skeleton as a plain ``aza`` that drops the charge."""
    return _isocyanide_atoms(mol, j) is not None


def _isocyano_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """``R-[N+]#[C-]`` attached via N, exactly the two isocyanide atoms --
    rendered as the ``isocyano`` substituent prefix (P-66.5.3; OPSIN-RT-verified:
    ``isocyanobenzene`` and ``isocyanomethane`` both parse back to the input).
    Returns ``("isocyano", atom_ids)`` or ``None`` (fall through). The C-/N+ are
    P-59 internal charges (get_ion_sites strips them; tracked as
    ``internal_atoms`` via ``_INTERNAL_CHARGE_LEAF_TOKENS``)."""
    if len(component) != 2:
        return None
    iso = _isocyanide_atoms(mol, attach_hint)
    if iso is None or iso != component:
        return None
    # attach_hint (the N) must have exactly one heavy neighbour OUTSIDE the group
    # -- the real parent R (also rejects a bare, unattached isocyanide fragment).
    outside = [nb for nb in mol.GetAtomWithIdx(attach_hint).GetNeighbors()
               if nb.GetAtomicNum() > 1 and nb.GetIdx() not in component]
    if len(outside) != 1:
        return None
    return "isocyano", frozenset(component)


# ===========================================================================
# M2 inc4: semipolar oxide  X(+)-[O-]  (X in N/P/S) -- ``oxido`` + ``-ium``
#   aromatic / amine N-oxide, phosphine oxide, sulfoxide (P-74.2.1)
# ===========================================================================

def _semipolar_oxide_partner(mol, x_idx: int) -> Optional[int]:
    """If atom *x_idx* is the CATION of a P-74.2.1 semipolar oxide -- an
    ``N``/``P``/``S`` carrying formal charge ``+1`` and a TERMINAL anionic oxygen
    ``[O-]`` (heavy-degree 1, H-free, single-bonded, ``-1``) -- return that
    oxygen's index, else ``None``.

    This is the zwitterionic drawing of an N-oxide (aromatic or amine), a
    phosphine oxide, or a sulfoxide (the anion element the Blue Book uses to draw
    the class -- see ``perception.ions._semipolar_chalcogenide_atoms``). A nitro
    or nitrate-ester nitrogen is EXCLUDED (it carries a ``=O`` as well and is
    rendered as its own whole leaf); requiring EXACTLY one oxygen neighbour and no
    double-bonded O on the cation keeps those out."""
    x = mol.GetAtomWithIdx(x_idx)
    if x.GetSymbol() not in ("N", "P", "S") or x.GetFormalCharge() != 1:
        return None
    if _is_nitro_root(mol, x_idx) or _is_nitrooxy_root(mol, x_idx):
        return None
    o_partner = None
    for nb in x.GetNeighbors():
        if nb.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(x_idx, nb.GetIdx())
        bt = round(bond.GetBondTypeAsDouble()) if bond is not None else 0
        if (nb.GetSymbol() == "O" and nb.GetFormalCharge() == -1
                and nb.GetDegree() == 1 and nb.GetTotalNumHs() == 0 and bt == 1):
            if o_partner is not None:
                return None  # two anionic oxides -> not a plain oxide (out of scope)
            o_partner = nb.GetIdx()
        elif nb.GetSymbol() == "O" and bt >= 2:
            return None  # a =O on the cation -> nitro/nitrate/etc., not this class
    return o_partner


def _semipolar_oxido_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """A lone terminal ``[O-]`` that is the anionic oxygen of a semipolar oxide
    (its sole heavy neighbour is an ``N+``/``P+``/``S+`` cation, see
    ``_semipolar_oxide_partner``) -- rendered as the ``oxido`` substituent prefix.
    Returns ``("oxido", {O})`` or ``None``. The paired ``-ium`` on the cation is
    appended by the spine (``_resolve_internal_oxide_suffix``); both atoms are
    tracked ``internal_atoms`` (``oxido`` in ``_INTERNAL_CHARGE_LEAF_TOKENS``)."""
    if len(component) != 1:
        return None
    o = mol.GetAtomWithIdx(attach_hint)
    if (o.GetSymbol() != "O" or o.GetFormalCharge() != -1
            or o.GetTotalNumHs() != 0 or o.GetDegree() != 1):
        return None
    heavy = [n for n in o.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(heavy) != 1:
        return None
    if _semipolar_oxide_partner(mol, heavy[0].GetIdx()) != attach_hint:
        return None
    return "oxido", frozenset(component)


def _resolve_internal_oxide_suffix(
    ctx: _Ctx, spine_atom_to_locant: Dict[int, int],
) -> Tuple[Optional[str], FrozenSet[int]]:
    """Does THIS spine carry any P-74.2.1 semipolar-oxide cation (an ``N+``/
    ``P+``/``S+`` whose terminal ``[O-]`` partner renders as ``oxido``)? If so,
    return the ``-<locants>-[di/tri...]ium`` suffix text and the cation atom ids
    (recorded ``internal_atoms``, NOT ``charged`` -- these are internal charges
    get_ion_sites strips, so they are invisible to ``_resolve_spine_charge``).
    Returns ``(None, frozenset())`` when the spine carries none.

    The paired ``oxido`` anions are rendered as ordinary branch leaves
    (``_semipolar_oxido_shortcut``); this only adds the cation's ``-ium``, so the
    assembled name is ``<n>-oxido...-<n>-ium`` (OPSIN-RT-verified: the kekulized
    ``1-oxido-1-azacyclohexa-1,3,5-trien-1-ium`` is pyridine N-oxide)."""
    locants: List[int] = []
    ids: set = set()
    for atom_idx, loc in spine_atom_to_locant.items():
        if _semipolar_oxide_partner(ctx.mol, atom_idx) is not None:
            locants.append(loc)
            ids.add(atom_idx)
    if not locants:
        return None, frozenset()
    locants.sort()
    n = len(locants)
    if n == 1:
        mult = ""
    else:
        mult = _MULT_SIMPLE.get(n)
        if mult is None:
            return None, frozenset()  # multiplicity beyond the simple table
    text = "-" + ",".join(map(str, locants)) + "-" + mult + "ium"
    return text, frozenset(ids)


def _phosphinate_oxide_leaf_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """``-P(=O)([O-])(H)`` (a phosphinate: P-V, one explicit P-H), attached
    via P to exactly one outside atom by a single bond -- 3 atoms, mirroring
    ``_nitro_shortcut``'s shape. Named directly as the SUBSTITUTIVE
    ``(oxido(oxo)phosphanyl)`` prefix (P-68 mononuclear-hydride substituent
    group ``phosphanyl`` = ``-PH2``, with ``oxo``/``oxido`` replacing two of
    its three H's, one remaining) rather than through the generic chain /
    ``_build_hetero_prefix`` replacement-nomenclature machinery.

    WS-NOABSTAIN class 3 root cause (measured, NOT the brief's original
    "N-simultaneous-leaves composition" hypothesis): OPSIN does not validate
    the equivalent raw skeletal-replacement spelling for this shape even WITH
    a correct P-15.4.1 lambda citation -- ``2-oxido-1-oxa-2λ5-phosphaprop-
    1-ene`` (the chain builder's own output for ``C[PH](=O)[O-]`` once
    lambda is wired in) still round-trips to the neutral, P-H-less
    ``O=P(=O)C``. The substitutive ``phosphanyl`` spelling, by contrast, IS
    OPSIN-verified: ``(oxido(oxo)phosphanyl)ethane`` -> ``[O-]P(=O)CC``,
    which RDKit fills to the SAME 1-implicit-H, valence-5 phosphorus as the
    input (the identical "explicit bond order already 4 of an allowed-
    valence-5 element -> 1 implicit H" inference ``methylphosphinate`` /
    ``CP([O-])=O`` already relies on). Returns
    ``(token, atom_ids, charged_atom_ids)`` or ``None``.

    Scope (tight, all validated by the caller's round-trip net): P formal
    charge 0, degree 3 (2 O's in ``component`` + 1 outside), the outside bond
    a SINGLE bond (substituent ``-yl`` attachment), one O double-bonded /
    neutral / 0-H (the ``oxo``) and one O single-bonded / -1 / 0-H (the
    ``oxido``, each terminal -- bonded ONLY to this P), and P itself carrying
    EXACTLY 1 H (the phosphinate's defining P-H -- a phosphonate-style P with
    a SECOND non-H substituent instead is a different species this shortcut
    must not mis-spell, and is not reachable here anyway since that shape has
    2 outside neighbours, not 1)."""
    if len(component) != 3:
        return None
    atom = mol.GetAtomWithIdx(attach_hint)
    if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0:
        return None
    if atom.GetTotalNumHs() != 1:
        return None
    in_component = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in component]
    if len(in_component) != 2:
        return None
    outside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in component]
    if len(outside) != 1:
        return None  # substituent attachment: exactly one outside neighbour
    bond_out = mol.GetBondBetweenAtoms(attach_hint, outside[0])
    if bond_out is None or round(bond_out.GetBondTypeAsDouble()) != 1:
        return None  # a "-yl" substituent attaches by a single bond
    oxido_idx = None
    for o_idx in in_component:
        o_atom = mol.GetAtomWithIdx(o_idx)
        if o_atom.GetSymbol() != "O":
            return None
        if [n.GetIdx() for n in o_atom.GetNeighbors()] != [attach_hint]:
            return None  # each O must bond ONLY to this P (terminal)
        bond = mol.GetBondBetweenAtoms(attach_hint, o_idx)
        bt = round(bond.GetBondTypeAsDouble())
        if bt == 2 and o_atom.GetFormalCharge() == 0 and o_atom.GetTotalNumHs() == 0:
            continue  # the neutral "oxo" oxygen
        if bt == 1 and o_atom.GetFormalCharge() == -1 and o_atom.GetTotalNumHs() == 0:
            oxido_idx = o_idx
            continue  # the anionic "oxido" oxygen
        return None
    if oxido_idx is None:
        return None
    return ("oxido(oxo)phosphanyl", frozenset(component), frozenset({oxido_idx}))


def _is_phosphinate_oxide_root(mol, j: int) -> bool:
    """True if atom *j* is a phosphinate P (see
    ``_phosphinate_oxide_leaf_shortcut`` for the exact shape) -- used by
    ``_tree_neighbors`` to keep it OUT of ordinary chain-spine continuation,
    mirroring ``_is_nitro_root``."""
    atom = mol.GetAtomWithIdx(j)
    if (atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0
            or atom.GetDegree() != 3 or atom.GetTotalNumHs() != 1):
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
