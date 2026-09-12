"""a phase — Continuous Triviality Controller (P2; AUTONOM-derived tree visitor).

Implements /02/03 via a pure functional transform ``NameTreeNode -> NameTreeNode'``:
  - Depth-first leaves-first traversal (Pitfall 4 avoidance: rewrite children BEFORE the
    parent's swap-decision so the parent sees the post-swap children).
  - Per-Type runtime dispatch (;..3): Type 1 unconditional, Type 2a
    principal-group-bound, Type 2b SMARTS closed-list, Type 2c per-entry-override-or-Type-3,
    Type 3 bare-only + locant_context.
  - Multiplier-feedback after every swap (;): re-derive the di<->bis multiplier
    on the swapped node via ``get_multiplier_prefix`` (which consults ``is_complex_substituent``)
    — never a static per-entry flag.
  - Re-alphabetization of the node whose prefixes changed, applied when that node is
    re-emitted : ``_alphabetize_prefixes`` (NOT a sibling-reorder at the swapped node's
    own level — the leaves-first walk re-alphabetizes a parent when the parent is visited).
  - runtime OPSIN-RT cache (;; memoized per ``(post_swap_subtree_str,
    pre_swap_canon)`` in a plain dict). On RT mismatch: silently keep systematic form +
    emit ``ControllerEvent(kind='swap_reject_rt_unsafe')``.

CRITICAL serializer invariant (#2 fix, reviews iter 1): the Pass-2 serializer short-circuits
on a non-None ``fragment_legacy`` (name_tree_to_string.py:96) and IGNORES ``node.prefixes``. So
ANY rewrite that changes ``parent_stem`` (the swap) OR changes a child
(``new_prefixes != node.prefixes``) MUST set ``fragment_legacy=None``, or the serializer
renders the stale legacy string and discards the rewrite. The helper
``_replace_preserving_or_resetting_legacy`` encodes the conditional: reset when a child
changed, preserve when nothing changed.

SMILES recovery is a 3-path cascade (#3 fix, reviews iter 1): an earlier hint-based
reverse-mapping path was REMOVED — the per-atom locant hint is NOT a field of the frozen
``NameTreeNode`` (name_tree.py:90-103); it lives on ``NamingResult`` (name_tree.py:128) and is
never forwarded onto nodes, so the lookup always returned None and that path was dead code.
Recovery therefore starts at token-match (labelled Path B for cascade continuity), then
reverse-OPSIN (Path C), then bail (Path D).

Recovery reach is bounded (#4, reviews iter 1): ``match_token_atoms_in_mol`` cannot map a
specific IR node to a specific physical duplicate fragment; for duplicate substituents it picks
a deterministic-but-arbitrary match. The recovered fragment SMILES is correct for true
duplicates (same canonical SMILES), but recovery can MISS for nested substituents — so
``controller_fired_count`` may trail ``controller_reach_count`` even for seed members. Plan-04
reports BOTH counts (internal notes) and never claims reach the recovery cannot deliver.

Thread-safety (#6 REJECTED, reviews iter 1): the OPSIN-RT cache is a plain per-instance dict.
The benchmark (benchmark_multi_corpus.py) is single-threaded over rows (max_workers=1 timeout
wrapper at:191; serial rows at:446; no --threads arg) and each OpsinOracle belongs to one
Orthonym instance — no lock needed.

Module boundary preserved per: this module is the ONLY new code in the assembly package;
name_tree.py, name_tree_to_string.py, naming_utils.py stay UNTOUCHED.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rdkit import Chem

from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

from .name_tree import NameTreeNode, _alphabetize_prefixes, is_coarse_node
from .naming_utils import (
    COMPLEX_MULTIPLIERS,
    SIMPLE_MULTIPLIERS,
    get_multiplier_prefix,
    is_complex_substituent,
)
from ..perception.smarts_cache import compiled as _compiled_smarts

# Lazy import of SEED_TABLE / SubstitutionType inside function bodies (Pattern S3)
# to avoid circular-import risk if data/ initialization is not finished.

logger = logging.getLogger(__name__)

# Multiplier -> occurrence count, for di<->bis re-derivation after a swap.
# (code review 2026-05-30): derived by INVERTING the canonical naming_utils maps
# (single source of truth) instead of a hand-maintained local copy. The previous literal
# table stopped at deca/decakis (10) and would silently diverge if naming_utils gained
# undeca-/dodeca-/... — the inversion tracks SIMPLE_MULTIPLIERS / COMPLEX_MULTIPLIERS exactly
# (both currently extend to icosa/icosakis = 20).
_MULT_COUNT: Dict[str, int] = {
    **{prefix: count for count, prefix in SIMPLE_MULTIPLIERS.items()},
    **{prefix: count for count, prefix in COMPLEX_MULTIPLIERS.items()},
}


@dataclass(frozen=True)
class ControllerEvent:
    """a phase diagnostic event (RESEARCH section 4.5)."""

    kind: str  # "swap_emit" | "swap_reject_type_check" | "swap_reject_rt_unsafe" |
    # "passthrough_coarse" | "passthrough_not_in_seed" | "recovery_miss"
    canonical_smiles: Optional[str]
    seed_entry_name: Optional[str]
    substitution_type: Optional[str]
    rt_oracle_result: Optional[bool]
    p_section_cite: Optional[str]


class OpsinOracle:
    """ runtime OPSIN-RT cache (+ RESEARCH R-02 + Pitfall 5).

    Plain per-instance dict cache keyed by ``(post_swap_subtree_str, pre_swap_canon)``.
    #6 REJECTED (reviews iter 1): no lock needed — the benchmark is single-threaded over rows.
    """

    def __init__(self, opsin_jar: Optional[str] = None):
        self._jar = opsin_jar
        self._cache: Dict[Tuple[str, str], bool] = {}
        self._name_cache: Dict[str, Optional[str]] = {}
        # validity-gate parse-outcome cache (/, code review
        # 2026-06-02). Only DEFINITIVE outcomes ('parsed'/'rejected') are stored
        # here — a transient 'unavailable' is never cached, so a one-off timeout
        # cannot poison later lookups of the same name.
        self._parse_status_cache: Dict[str, str] = {}
        self._events: list = []  # ControllerEvent diagnostic log

    def rt_safe(self, pre_swap_canon: str, post_swap_subtree_str: str) -> bool:
        """internal notes T2: parse ``post_swap_subtree_str`` via OPSIN, canonicalize, compare to
        ``pre_swap_canon``. Cached per ``(post_swap_subtree_str, pre_swap_canon)``."""
        key = (post_swap_subtree_str, pre_swap_canon)
        if key in self._cache:
            return self._cache[key]
        if self._jar is None:
            # (code review 2026-05-30): FAIL CLOSED. With no OPSIN jar the round-trip
            # CANNOT be verified, so the swap must be REJECTED (keep the systematic form) — a
            # safety gate that fails open is worse than no gate. The earlier " already verified
            # RT safety at seed load" justification does NOT hold: validates the BARE
            # ``retained_pin_name``, whereas ``post_swap_subtree_str`` is the FULL post-swap
            # subtree (parent + substituents/locants), a different string. is "RT-safety
            # by construction": never emit a retained name we could not round-trip. Stage A stays
            # byte-identical regardless (the flag is OFF, so the controller is never invoked).
            self._cache[key] = False
            return False
        # PERF: prefer the ONE in-process JVM (jvm_bridge, JPype) over a ~130 ms
        # process launch. It hands back exactly the bytes the `java -jar` call
        # below would have written to stdout, so the handling is shared verbatim;
        # served=False falls through to the subprocess unchanged.
        _text = None
        _served = False
        try:
            from ..jvm_bridge import opsin_stdout
            _text, _served = opsin_stdout(post_swap_subtree_str,
                                          allow_radicals=True, jar_path=self._jar)
        except ImportError:  # pragma: no cover - jvm_bridge always present
            _served = False
        try:
            if not _served:
                result = subprocess.run(
                    # -r (--allowRadicals): add consistently with _invoke_opsin so the
                    # oracle accepts radical names; proven strictly additive over 11,668
                    # names (66 gains / 0 changes / 0 regressions; internal notes §-RADICAL).
                    ["java", *JVM_HYGIENE_FLAGS, "-jar", self._jar, "-r", "-osmi"],
                    input=post_swap_subtree_str + "\n",
                    capture_output=True, text=True, timeout=10,
                )
                _text = result.stdout
            opsin_smiles = (_text or "").strip()
            if not opsin_smiles:
                self._cache[key] = False
                return False
            actual_canon = Chem.CanonSmiles(opsin_smiles)
            ok = (actual_canon == pre_swap_canon)
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            logger.debug("OpsinOracle T2 RT check failed for %r: %s", post_swap_subtree_str, exc)
            ok = False
        self._cache[key] = ok
        return ok

    def _invoke_opsin(self, name: str) -> Tuple[Optional[str], bool]:
        """Run OPSIN ``-osmi`` on a single name. Single source of truth for the
        subprocess call shared by ``name_to_smiles`` and ``parse_status``.

        Returns ``(raw_smiles_or_None, ran)``:
          * ``ran=True, raw=<smiles>`` — OPSIN parsed the name, emitted SMILES;
          * ``ran=True, raw=None`` — OPSIN ran but emitted nothing (it
            DEFINITIVELY rejected the name as unparseable);
          * ``ran=False, raw=None`` — the parse could NOT be performed
            (subprocess timeout / ``OSError`` such as a fork failure under memory
            pressure). This is a transient/environmental failure that callers
            MUST NOT treat as a rejection .
        """
        # PERF: try the persistent-OPSIN process first (one long-lived JVM
        # instead of a fresh ~1.7s boot per distinct name — the dominant gate
        # cost). It preserves this method's exact contract: (smiles, True) parse,
        # (None, True) clean rejection, (None, False) transient/unavailable. On
        # (None, False) — server absent/dead/timeout/protocol-guard — we FALL
        # THROUGH to the one-shot subprocess below, so correctness never depends
        # on the optimization; it only removes JVM-startup latency.
        # PERF tier 1: the ONE in-process JVM (jvm_bridge, JPype) -- a JNI call
        # instead of a process launch. It returns exactly the bytes
        # `java -jar... -r -osmi` writes to stdout, so the (raw, ran) mapping
        # here is identical to the one-shot subprocess path's below. When it
        # cannot serve the call it reports served=False and we drop to the
        # persistent server, then to the one-shot subprocess -- the tiers below
        # are untouched and remain the correctness path.
        if self._jar:
            try:
                from ..jvm_bridge import opsin_stdout
                _text, _served = opsin_stdout(name, allow_radicals=True,
                                              jar_path=self._jar)
            except ImportError:  # pragma: no cover - jvm_bridge always present
                _text, _served = None, False
            if _served:
                _smi = (_text or "").strip()
                return (_smi or None), True

        try:
            from ..validation.opsin_server import get_persistent_opsin
            _srv = get_persistent_opsin(self._jar, ("-r", "-osmi"))
        except Exception:  # pragma: no cover - import guard
            _srv = None
        if _srv is not None:
            raw, ran = _srv.invoke(name)
            if ran:
                return (raw or None), True
            # ran is False: transient/unavailable -> fall through to the
            # definitive one-shot subprocess (never cached by callers).

        try:
            result = subprocess.run(
                # -r (--allowRadicals): the single source-of-truth subprocess call
                # used by name_to_smiles AND parse_status. Adding -r lets the
                # validity gate (namer.py, via parse_status) ACCEPT radical names like
                # 'pentan-3-yl' instead of suppressing them to 'unknown organic
                # compound'. Proven strictly additive over 11,668 names (66 empty->
                # parsed gains / 0 non-additive changes / 0 regressions;
                # internal notes §-RADICAL(a)) — byte-identical on all non-radical
                # names. NOT a blanket gate bypass: OPSIN itself still validates the
                # name (the heptanolate-class suppression hole stays closed).
                ["java", *JVM_HYGIENE_FLAGS, "-jar", self._jar, "-r", "-osmi"],
                input=name + "\n",
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("OpsinOracle OPSIN invocation unavailable for %r: %s", name, exc)
            return None, False
        # A NON-ZERO exit means OPSIN itself errored (e.g. a JVM OOM / crash
        # under full-corpus parallel load), NOT a definitive rejection. OPSIN
        # exits 0 for BOTH a successful parse (SMILES on stdout) AND a clean
        # rejection (empty stdout + an "unparsable" note on stderr) — so a
        # non-zero exit is always a transient/environmental failure. Treat it as
        # "unavailable" (ran=False) so the validity gate fails OPEN and
        # never suppresses a valid name on a transient OPSIN crash. This
        # completes the hardening, which previously caught only
        # TimeoutExpired/OSError and let a non-zero exit collapse to "rejected"
        # — the cause of valid heavy aminium names being suppressed to
        # "unknown organic compound" under load (-01 close finding).
        if result.returncode != 0:
            logger.debug(
                "OpsinOracle OPSIN non-zero exit (%s) for %r — treating as "
                "unavailable (transient), not a rejection",
                result.returncode, name,
            )
            return None, False
        smi = result.stdout.strip()
        return (smi or None), True

    def name_to_smiles(self, name: str) -> Optional[str]:
        """Path C reverse-OPSIN helper. Returns canonical SMILES or None on failure.

        A DEFINITIVE result (a real SMILES, OPSIN's rejection, or a
        canonicalisation failure on OPSIN's output) is cached; a transient
        invocation failure (timeout / ``OSError``) is NOT cached (, code
        review 2026-06-02), so a one-off failure never poisons later lookups of
        the same name.
        """
        if self._jar is None or not name:
            return None
        if name in self._name_cache:
            return self._name_cache[name]
        raw, ran = self._invoke_opsin(name)
        if not ran:
            return None  # transient — do NOT cache
        try:
            # OPSIN can emit a SMILES that RDKit cannot parse (e.g. an impossible
            # valence from a lambda-convention candidate such as
            # '1,2lambda6,3-dioxathiolane'): MolFromSmiles then returns None, so
            # guard BEFORE canonicalising — Chem.CanonSmiles(None-parse) calls
            # MolToSmiles(None) which raises Boost.Python.ArgumentError, NOT
            # ValueError, and would escape this handler. Any failure to obtain a
            # clean canonical form means "OPSIN produced no valid structure" ->
            # None (definitive, cached) -> the validity gate suppresses the
            # candidate -> fail-closed. Behaviour-preserving for parseable output
            # (MolToSmiles default isomericSmiles=True == CanonSmiles useChiral=1).
            parsed = Chem.MolFromSmiles(raw) if raw else None
            canon = Chem.MolToSmiles(parsed) if parsed is not None else None
        except Exception as exc:  # noqa: BLE001 - any RDKit failure on OPSIN output => no structure
            logger.debug("OpsinOracle.name_to_smiles canonicalisation failed for %r: %s", name, exc)
            canon = None
        self._name_cache[name] = canon
        return canon

    def parse_status(self, name: str) -> str:
        """Three-valued OPSIN parse outcome for the validity gate .

        Returns one of:
          * ``"parsed"`` — OPSIN accepted the name (emitted SMILES);
          * ``"rejected"`` — OPSIN ran and definitively rejected it (no SMILES);
          * ``"unavailable"`` — the parse could not be performed (no JAR / timeout
            / ``OSError``).

        Callers MUST fail-OPEN on ``"unavailable"`` — a missing JAR or a transient
        subprocess failure must NEVER suppress a name. This is the distinction the
        old ``name_to_smiles is not None`` check collapsed, which let a timeout
        turn a valid, round-trip-passing name into a descriptive fallback.

        Only definitive outcomes are cached, so a transient failure never poisons
        a later lookup of the same name .
        """
        if not name:
            return "rejected"
        if self._jar is None:
            return "unavailable"
        cached = self._parse_status_cache.get(name)
        if cached is not None:
            return cached
        raw, ran = self._invoke_opsin(name)
        if not ran:
            return "unavailable"  # transient — NOT cached
        status = "parsed" if raw else "rejected"
        self._parse_status_cache[name] = status
        return status


def _recompute_multiplicative_prefix(old_mult: Optional[str], new_name: str) -> Optional[str]:
    """ : re-derive the di<->bis multiplier for a repeated substituent whose name
    changed, using ``get_multiplier_prefix`` — never a static per-entry flag. Preserves the
    occurrence count encoded by the old multiplier; returns None when the node carried no
    multiplier.

    ⚠ **Docstring corrected 2026-07-30.** It said ``get_multiplier_prefix`` "consults
    ``is_complex_substituent`` internally". It does **not** — measured with a trace,
    ``is_complex_substituent`` records **zero** calls from it; the predicate actually
    consulted is ``is_substituted_substituent`` (plus
    ``CATENATION_AMBIGUOUS_PREFIXES``). The two genuinely disagree, so the wrong name
    was not a harmless synonym:

        1,2-xylene complex=True substituted=False -> 'di'
        bromomethyl complex=True substituted=True -> 'bis'
        propan-2-yl complex=True substituted=False -> 'di'

    and ``substituted`` is the one that matches the Blue Book — ``:25811``
    ``1,2-bis(bromomethyl)benzene (PIN)`` against ``:25719``'s ``di(propan-2-yl)``.
    A stale docstring naming the wrong predicate is how commit `` came to
    re-point this at ``is_complex_substituent`` and regress it in both directions.

    This matters because the serializer renders ``node.multiplicative_prefix`` verbatim
    (name_tree_to_string.py:164-165) — it does NOT recompute from sibling counts. So a swap that
    flips a substituent simple<->complex (e.g. a hyphenated retained form) MUST re-derive the
    multiplier here or the di/bis prefix would be dropped or stale.
    """
    if not old_mult:
        return None
    count = _MULT_COUNT.get(old_mult.strip().lower())
    if count is None:
        return None
    return get_multiplier_prefix(count, new_name)


def _replace_preserving_or_resetting_legacy(
    node: NameTreeNode, new_prefixes: Tuple[NameTreeNode, ...],
) -> NameTreeNode:
    """#2 FIX (reviews iter 1): no-swap branches (seed-miss / recovery-fail / type-refuse /
    RT-unsafe) return the node with re-alphabetized children. BUT if a CHILD actually changed
    (``new_prefixes != node.prefixes``), the serializer's ``fragment_legacy`` short-circuit
    (name_tree_to_string.py:96) would discard the child rewrite — so we MUST reset
    ``fragment_legacy=None`` in that case. If nothing changed, preserve ``fragment_legacy``
    (original NameTreeNode contract)."""
    alphabetized = _alphabetize_prefixes(new_prefixes)
    if alphabetized != node.prefixes:
        # A child changed (or re-sorted) => force the serializer onto explicit fields.
        return dataclasses.replace(node, prefixes=alphabetized, fragment_legacy=None)
    # Nothing changed => preserve fragment_legacy.
    return dataclasses.replace(node, prefixes=alphabetized)


def apply_triviality_controller(
    tree: NameTreeNode,
    mol: "Chem.Mol",
    principal_group: Optional[str],
    opsin_oracle: Optional[OpsinOracle] = None,
    *,
    enabled: bool = False,
) -> NameTreeNode:
    """a phase entry point (internal notes +).

    - When ``enabled=False``: returns the input tree UNCHANGED (Stage A default-OFF invariant per).
    - When ``enabled=True`` + ``is_coarse_node(tree)``: returns the input tree UNCHANGED (coarse passthrough).
    - Otherwise: depth-first leaves-first walk; per-Type dispatch; rewrite-on-match + RT-gate.
    """
    if not enabled:
        return tree
    if is_coarse_node(tree):
        return tree
    return _walk(tree, mol, principal_group, opsin_oracle)


def _walk(
    node: NameTreeNode,
    mol: "Chem.Mol",
    principal_group: Optional[str],
    opsin_oracle: Optional[OpsinOracle],
) -> NameTreeNode:
    """Depth-first LEAVES-FIRST traversal (Pitfall 4 mitigation per RESEARCH section 4.1).

    Children are rewritten FIRST so the parent's swap-decision sees the post-swap children.
    """
    if is_coarse_node(node):
        return node  # coarse passthrough

    # Rewrite children first (leaves-first).
    new_prefixes = tuple(
        _walk(p, mol, principal_group, opsin_oracle) for p in node.prefixes
    )

    # Lazy import of seed table (Pattern S3).
    from ..data.triviality_controller_seed import SEED_TABLE

    # Recover canonical SMILES of this node's substructure (RESEARCH section 4.2 3-path cascade).
    canonical = _recover_canonical_smiles(node, mol, opsin_oracle)
    if canonical is None:
        # Can't recover SMILES => can't look up => no swap of THIS node; preserve children rewrite
        # + re-alphabetize. #2 fix: reset fragment_legacy IFF a child changed.
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    seed_entry = SEED_TABLE.get(canonical)
    if seed_entry is None:
        # Not in seed table => keep systematic (preserve children rewrite + re-alphabetize per).
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    # Per-Type runtime dispatch (internal notes; RESEARCH section 2.1).
    if not _type_check(node, mol, seed_entry, principal_group, opsin_oracle):
        # Type check refused: keep systematic. #2 fix: reset fragment_legacy IFF a child changed.
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    # Build the candidate rewrite (this DOES change parent_stem => fragment_legacy=None per #2/Pitfall 1).
    rewritten = _build_rewrite(node, seed_entry, new_prefixes)

    # runtime RT-safety check (internal notes + "RT-safety by construction").
    # (code review 2026-05-30): a swap is emitted ONLY when the oracle verifies that the
    # post-swap subtree round-trips. No oracle => the round-trip CANNOT be verified => FAIL CLOSED
    # (keep the systematic form). Previously a None oracle fell straight through to
    # ``return rewritten``, emitting the swap UNVERIFIED — the same fail-open class as the
    # jar-None path in ``OpsinOracle.rt_safe``. In production the candidate_pool / _fallback
    # wiring always passes a non-None OpsinOracle when the flag is ON; the unit tests that
    # exercise swap LOGIC do so via ``_build_rewrite`` / ``_type_*_check`` directly (no oracle).
    if opsin_oracle is None:
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)
    from .name_tree_to_string import name_tree_to_string
    post_swap_str = name_tree_to_string(rewritten, style="pin")
    if not opsin_oracle.rt_safe(canonical, post_swap_str):
        # RT-unsafe => silently reject swap (the systematic form is kept). internal notes + Pitfall 6.
        # #2 fix: reset fragment_legacy IFF a child changed (the swap of THIS node is rejected,
        # but a child below may have changed).
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    return rewritten


def _build_rewrite(
    node: NameTreeNode,
    seed_entry,
    new_prefixes: Tuple[NameTreeNode, ...],
) -> NameTreeNode:
    """Rewrite a node to its retained-PIN form (internal notes + RESEARCH section 4.1).

    CRITICAL INVARIANT per #2 + Pitfall 1: this function DOES change ``parent_stem``, so it MUST
    set ``fragment_legacy=None`` to force the serializer to use explicit fields rather than the
    stale legacy string at name_tree_to_string.py:96.

     : the di<->bis multiplier is re-derived via ``_recompute_multiplicative_prefix``
    (which calls ``get_multiplier_prefix``), and the parenthesization hint is re-derived via
    ``is_complex_substituent`` — both single-source-of-truth predicates, never a static
    flag.

     + (code review 2026-05-30): the seed ``retained_pin_name`` comes in two shapes,
    and the rewrite MUST reset the fields the retained string already subsumes or the serializer
    DOUBLE-renders them (name_tree_to_string.py:185-187 re-prepends locants;:192 re-appends
    suffix):

      * Type 1 — a BARE parent hydride ("benzene", "furan", "1H-pyrrole"). The
        principal-characteristic-group suffix is NOT part of the retained name, so ``suffix`` and
        its ``locants`` are PRESERVED (e.g. a 2-ol on naphthalene survives the swap). Only
        ``indicated_h`` and ring ``unsaturation_locants`` are subsumed by the parent-hydride name
        ("1H-pyrrole" already carries its own 1H; "furan" its own aromaticity) — reset them so we
        never emit "1H-1H-pyrrole".
      * Type 2a / 2b / 2c / 3 — a COMPLETE name that already embeds the principal group and/or
        the intrinsic locants: "phenol"/"aniline" (the -ol/-amine), "acetic acid"/"benzoic acid"
        (the acid word), "1,2-xylene" (the locant cluster), "anisole"/"toluene". Keeping
        ``node.suffix`` here produced "acetic acidoic acid" ; keeping ``node.locants``
        produced "1,2-1,2-xylene" . Both are reset; only the substituent ``prefixes``
        survive (Type 2a/2b permit senior-group-bound / compulsory-prefix substitution; Type 2c
        and Type 3 require ``node.prefixes`` empty via their type checks, so nothing is carried).

    ``substitution_type`` is the single source of truth for the complete-vs-bare split — there is
    no hand-maintained per-entry flag to drift. ``seed_entry.locant_context`` is consumed ONLY by
    ``_type_3_check`` as a MATCH criterion now, never re-emitted as output locants.
    """
    from ..data.triviality_controller_seed import SubstitutionType

    new_name = seed_entry.retained_pin_name
    is_complete = seed_entry.substitution_type != SubstitutionType.TYPE_1
    if is_complete:
        new_suffix = None
        new_locants: Tuple[int, ...] = ()
    else:
        new_suffix = node.suffix
        new_locants = node.locants if node.suffix else ()
    return dataclasses.replace(
        node,
        parent_stem=new_name,
        suffix=new_suffix,
        locants=new_locants,
        indicated_h=(),                 # subsumed by the retained parent name (e.g. "1H-pyrrole")
        unsaturation_locants=((), ()),  # subsumed by the retained stem (e.g. "furan", "indene")
        prefixes=_alphabetize_prefixes(new_prefixes),  # always-re-run
        multiplicative_prefix=_recompute_multiplicative_prefix(
            node.multiplicative_prefix, new_name),  #
        parenthesization_hint=is_complex_substituent(new_name),  #
        iupac_section_cite=seed_entry.iupac_p_section,
        fragment_legacy=None,  # CRITICAL — force serializer to use explicit fields (#2 / Pitfall 1)
    )


def _type_check(
    node: NameTreeNode,
    mol: "Chem.Mol",
    seed_entry,
    principal_group: Optional[str],
    opsin_oracle: Optional[OpsinOracle],  # WARNING #7 fix: propagated for Type 2b prefix recovery
) -> bool:
    """Per-Type runtime check (internal notes + RESEARCH section 2.1)."""
    from ..data.triviality_controller_seed import SubstitutionType

    st = seed_entry.substitution_type
    if st == SubstitutionType.TYPE_1:
        return _type_1_check(node, seed_entry)
    if st == SubstitutionType.TYPE_2A:
        return _type_2a_check(node, principal_group, seed_entry)
    if st == SubstitutionType.TYPE_2B:
        return _type_2b_check(node, mol, seed_entry, opsin_oracle)  # WARNING #7 — oracle propagated
    if st == SubstitutionType.TYPE_2C:
        return _type_2c_check(node, mol, seed_entry)
    if st == SubstitutionType.TYPE_3:
        return _type_3_check(node, seed_entry)
    raise ValueError(f"unknown substitution_type {st!r}")


def _type_1_check(node: NameTreeNode, seed_entry) -> bool:
    """Type 1: unconditional swap on canonical-SMILES match (RESEARCH section 2.1)."""
    return True


def _type_2a_check(node: NameTreeNode, principal_group: Optional[str], seed_entry) -> bool:
    """Type 2a: principal-group-bound (RESEARCH section 2.1 + Pitfall 2).

    NOTE (#8, reviews iter 1): ``principal_group`` is the MOLECULE-LEVEL PCG
    (features.principal_group), not a fragment-level group. So a phenol/aniline fragment occurring
    as a NON-principal substituent on a higher-seniority parent (e.g. an ester) will NOT swap (the
    required PG won't match). This is the correct conservative PIN behavior per internal notes; it
    bounds Type 2a coverage. A Plan-03 fixture pins the intended stay-systematic behavior.
    """
    required = seed_entry.principal_group_required
    if not required:
        return False
    if principal_group is None:
        return False
    return _principal_groups_match(principal_group, required)


def _type_2b_check(
    node: NameTreeNode,
    mol: "Chem.Mol",
    seed_entry,
    opsin_oracle: Optional[OpsinOracle],  # WARNING #7 fix: oracle propagated for prefix SMILES recovery
) -> bool:
    """Type 2b: closed-substituent SMARTS allow-list (RESEARCH section 2.1).

    WARNING #7 FIX: ``opsin_oracle`` is propagated into prefix SMILES recovery (formerly hard-coded
    to None, which disabled Path C and caused Type 2b to silently fail on any prefix where Path B
    token-match had no in-mol atoms). The 3-path cascade is used per prefix.
    """
    smarts_list = seed_entry.compulsory_prefix_smarts
    if smarts_list is None:
        return False
    if not node.prefixes:
        return True  # bare => Type 2b reduces to Type 1 unconditional
    for prefix_node in node.prefixes:
        prefix_canon = _recover_canonical_smiles(prefix_node, mol, opsin_oracle)
        if prefix_canon is None:
            return False
        prefix_mol = Chem.MolFromSmiles(prefix_canon)
        if prefix_mol is None:
            return False
        matched = False
        for smarts in smarts_list:
            pattern = _compiled_smarts(smarts)
            if pattern is not None and prefix_mol.HasSubstructMatch(pattern):
                matched = True
                break
        if not matched:
            return False
    return True


def _type_2c_check(node: NameTreeNode, mol: "Chem.Mol", seed_entry) -> bool:
    """Type 2c: per-name-specific (internal notes default-to-Type-3)."""
    if seed_entry.locus_override_rule_id is None:
        return _type_3_check(node, seed_entry)
    return False


def _type_3_check(node: NameTreeNode, seed_entry) -> bool:
    """Type 3: bare-only + locant_context match (RESEARCH section 2.1 + Pitfall 3)."""
    if node.prefixes:
        return False
    if seed_entry.locant_context is not None:
        if set(node.locants) != set(seed_entry.locant_context):
            return False
    return True


def _principal_groups_match(inferred: str, required: str) -> bool:
    """Normalize PG strings for comparison (case-insensitive, strip whitespace)."""
    return inferred.strip().lower() == required.strip().lower()


def _recover_canonical_smiles(
    node: NameTreeNode,
    mol: "Chem.Mol",
    opsin_oracle: Optional[OpsinOracle],
) -> Optional[str]:
    """Recover the canonical SMILES of the substructure this node represents.

    3-PATH cascade per RESEARCH section 4.2 + #3 fix (the earlier hint-based reverse-mapping path
    was REMOVED — the per-atom locant hint is NOT a field of the frozen NameTreeNode; it lives on
    NamingResult and is never forwarded to nodes):
      Path B — rules.parent_correctness.match_token_atoms_in_mol(node.parent_stem, mol)
      Path C — opsin_oracle.name_to_smiles(node.parent_stem) reverse-OPSIN
      Path D — bail (return None; controller keeps systematic form)

    #4 reach-limit: Path B picks a deterministic-but-arbitrary match for duplicate fragments; it
    cannot map a specific IR node to a specific physical duplicate. Recovery can MISS for nested
    substituents, so controller_fired_count may trail controller_reach_count.
    """
    # Path B: token-match the parent_stem against atoms in the mol.
    try:
        from ..rules.parent_correctness import match_token_atoms_in_mol
        token = (node.parent_stem or "").strip()
        if token and mol is not None:
            atoms = match_token_atoms_in_mol(token, mol)
            if atoms:
                frag = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
                if frag:
                    return Chem.CanonSmiles(frag)
    except Exception as exc:
        logger.debug("Path B SMILES recovery failed for %r: %s", node.parent_stem, exc)

    # Path C: reverse-OPSIN the parent_stem (REQUIRES opsin_oracle propagation per WARNING #7).
    if opsin_oracle is not None:
        try:
            canon = opsin_oracle.name_to_smiles(node.parent_stem or "")
            if canon:
                return canon
        except Exception as exc:
            logger.debug("Path C SMILES recovery failed for %r: %s", node.parent_stem, exc)

    # Path D: bail.
    return None
