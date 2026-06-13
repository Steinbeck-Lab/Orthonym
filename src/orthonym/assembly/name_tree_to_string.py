"""Phase 160 Pass-2 serializer (DECOMP-02 + CONTEXT D-04).

Single source of truth for ``NameTreeNode -> str`` going forward. Plan-02
substrate ships this module ALONGSIDE the legacy ``_assemble_fragments``
path at composer.py:7591 — the legacy path STAYS until commit 03-10 thinning
per CONTEXT D-24 incremental-migration discipline. Plan-02/03 first-wave
handlers return ``NamingResult(name=<existing string>, tree=None, ...)``;
the legacy path is what actually produces those names. The Pass-2
serializer here is exercised by ``--dump-tree`` integration tests in Plan-04
and by v19+ handlers that populate trees explicitly.

Contract per CONTEXT D-04 + 160-AUDIT-DECOMP.md § 5.4:

* If ``node.fragment_legacy is not None``: delegate to the legacy assembly
  path by wrapping the fragment list as
  ``_assemble_fragments([node.fragment_legacy], style)``. This is the
  first-wave compatibility mode that lets Phase 160 ship byte-identical
  even before any handler emits a real tree.
* If ``node.fragment_legacy is None`` AND ``node.parent_stem`` is populated:
  assemble the name from the explicit fields in the order::

      stereo + prefixes + indicated_h + parent_stem + locants
             + unsaturation_locants + suffix

  applying multiplicative-prefix and parenthesization rules per
  P-14.5 + P-14.2.2.
* If ``node.fragment_legacy is None`` AND ``node.parent_stem == ""``: raise
  an explicit ``NameTreeSerializerError`` — this is a malformed tree per
  DECOMP-02 honest-fail-on-data.

Byte-identical contract (DECOMP-03): for every input ``node`` whose
``fragment_legacy`` round-trips through the legacy path, the output of
``name_tree_to_string(node, style)`` MUST equal the corresponding
``_assemble_fragments(fragments, style)`` output.

Anti-pattern hygiene:
- AP-160-05 / AP-160-23: no postprocessor band-aid on inner-dispatch
  output; the explicit-field branch below assembles deterministically
  from the IR fields.
- AP-160-27: NameTreeNode field shape is locked — adding new fields to
  the serializer here without a corresponding NameTreeNode field
  addition is a Rule 4 architectural decision.

References:
- 160-AUDIT-DECOMP.md § 5.4 — Pass-2 serializer contract.
- 160-CONTEXT.md D-04 / Question 3 — explicit two-pass IR + serializer.
- 160-PATTERNS.md § 2 — analog: composer.py:7591-7748 ``_assemble_fragments``.
- 160-RESEARCH.md § "Name-Tree IR Semantic Contract" — collision detection
  per IUPAC P-14.7; suffix-prefix locant priority; unsaturation infix.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .name_tree import NameTreeNode, _alphabetize_prefixes


class NameTreeSerializerError(ValueError):
    """Raised when a malformed NameTreeNode is passed to name_tree_to_string.

    Per DECOMP-02 honest-fail-on-data (CONTEXT D-27): a node with
    ``parent_stem == ""`` and ``fragment_legacy is None`` cannot serialize
    deterministically. The fix is upstream in the handler that produced
    the malformed node, not a band-aid string default.
    """


# Phase 179 (WSA-03): the set of class_ids whose ``NameTreeNode`` carries a BARE
# parent-hydride stem ("but", "meth", "cyclodec") and is therefore assembled by
# the full hydride grammar (-ane/-ene/-yne + suffix infix). Every OTHER node
# reaching ``_assemble_explicit_fields`` carries a COMPLETE name in ``parent_stem``
# (a retained name like "acetic acid"/"phenol", an organometallic like
# "tetramethylstannane", or a coarse fallback) and is assembled by the
# pass-through branch (no hydride ending appended). This is ALSO the production
# flip set: Plan 02 couples the carrier-drop + seam routing to this SAME frozenset
# (single source of truth — prevents "flip does nothing"). Defined here so the
# serializer's two-branch dispatch and the Plan-02 flip share one definition.
SERIALIZER_PRODUCTION_CLASSES = frozenset({"general_acyclic"})


def name_tree_to_string(node: NameTreeNode, style: str = "pin") -> str:
    """Phase 160 D-04 + IUPAC P-14.5 Pass-2 serializer.

    Composition order::

        [stereo] + [prefixes (alpha-sorted; recursively serialized)] +
        [parent_stem] + [indicated_h] + [unsaturation_infix] + [suffix]

    Args:
        node: The NameTreeNode to serialize.
        style: Naming style ("pin", "general", "cas"); forwarded to the
            legacy assembler when ``fragment_legacy`` is set.

    Returns:
        IUPAC name string per CONTEXT D-04 byte-identical contract.

    Raises:
        NameTreeSerializerError: if ``node.parent_stem == ""`` AND
            ``node.fragment_legacy is None`` (malformed tree per
            DECOMP-02 honest-fail-on-data).
    """
    # First-wave compatibility mode (per 160-AUDIT-DECOMP.md § 5.4 first
    # bullet): if the legacy NameFragment is present, route through the
    # legacy assembly path to preserve byte-identical output even before
    # any handler emits a real tree. This is the SOLE wave-1 production
    # path; the explicit-field branch below is exercised by Plan-04
    # --dump-tree integration tests + by v19+ handlers that populate
    # trees.
    if node.fragment_legacy is not None:
        # Phase 165 (SCORE-01) final-string carrier: a ``str`` fragment_legacy
        # holds the pre-assembled final name and is returned verbatim. A single
        # synthetic NameFragment cannot reproduce a multi-fragment concatenation
        # (the parent path appends 'ane' via _build_hydrocarbon_name), so the
        # dedicated string short-circuit is the byte-identical carrier for the
        # fragments_to_tree deriver + caller-side concatenations (e.g. amide
        # N-prefix). RESEARCH 165 lines 459-462.
        if isinstance(node.fragment_legacy, str):
            # IN-4 note: an empty-string ``fragment_legacy=""`` is a VALID
            # (empty) carrier and returns ``""`` verbatim here — deliberately
            # distinct from ``fragment_legacy is None`` (the latter falls through
            # to the explicit-field / honest-fail path below). This str branch is
            # reached only when ``fragment_legacy is not None`` (outer guard), so
            # ``""`` short-circuits BEFORE the ``parent_stem`` honest-fail at the
            # bottom. The SC-3 boundary only synthesizes a node when ``name`` is
            # truthy (namer.py ``if name:``), so an empty name never reaches here
            # in production; this is documented, not a behavioral guard.
            return node.fragment_legacy
        # Lazy import to avoid composer.py -> name_tree_to_string -> composer.py
        # cycle at module-import time (PATTERNS § Lazy Import).
        from .composer import _assemble_fragments
        return _assemble_fragments([node.fragment_legacy], style)

    # Honest-fail-on-data per DECOMP-02 (CONTEXT D-27): a node without
    # both fragment_legacy AND parent_stem cannot serialize. The fix is
    # upstream in the handler.
    if not node.parent_stem:
        raise NameTreeSerializerError(
            f"NameTreeNode with empty parent_stem and no fragment_legacy "
            f"cannot serialize. node={node!r}. Per DECOMP-02 honest-fail-"
            f"on-data: fix the upstream handler that produced this node."
        )

    # Explicit-field assembly (Plan-04+ + v19 path). The body below mirrors
    # the IUPAC P-14.5 composition order; collision detection per
    # IUPAC P-14.7 (suffix has priority over substituent locants) is
    # implemented per RESEARCH § "Name-Tree IR Semantic Contract".
    return _assemble_explicit_fields(node, style)


def _prefix_node_text(sub: NameTreeNode, style: str) -> str:
    """Return the substituent text for a prefix sub-node.

    Production ``general_acyclic`` prefix nodes (built by ``fragments_to_tree``)
    carry the FULL substituent string in ``parent_stem`` with
    ``fragment_legacy=None`` — used DIRECTLY here (NOT run through the hydride
    builder, which would wrongly append 'ane' to a substituent like 'methyl').
    A ``str`` ``fragment_legacy`` carrier short-circuits (other trees). A nested
    complex substituent (one carrying its own prefixes/suffix) recurses through
    the full serializer.
    """
    if isinstance(sub.fragment_legacy, str):
        return sub.fragment_legacy
    if sub.prefixes or sub.suffix:
        return name_tree_to_string(sub, style)
    return sub.parent_stem


def _assemble_explicit_fields(node: NameTreeNode, style: str) -> str:
    """Assemble a name from explicit ``NameTreeNode`` fields, byte-identically to
    the legacy ``_assemble_fragments`` (CONTEXT D-02 / D-03, Phase 179 WSA-03).

    Mirrors ``handlers/_handler_shared.py:_assemble_fragments`` term-for-term,
    reusing the SAME shared grammar from ``composition_primitives`` (one
    composition logic, not two — the no-band-aid mandate). Composition order
    (IUPAC P-14.5)::

        stereo + prefixes(alphabetized, enclosed, joined) + parent
               + unsaturation-infix + suffix(with locants/multiplier)

    The 8 verified gaps (179-RESEARCH) are each closed by a CALL to a shared
    primitive, not a local reimplementation: #1 P-14.7 collision, #2 prefix
    already-has-locant guard, #3 P-16.5.1.3.1 mononuclear enclosing, #4 inter-
    prefix hyphenation, #5 full suffix grammar (multiplier + P-16.7.1 elision),
    #6 unsaturation infix, #7 prefix->parent hyphenation, #8 stereo prepend.
    """
    import re

    from .composition_primitives import (
        _build_unsaturation_infix,
        _build_hydrocarbon_name,
        _join_prefixes,
        _join_prefix_to_name,
        _estimate_parent_size_from_name,
        apply_mononuclear_enclosing,
        resolve_suffix_prefix_collision,
        is_ring_parent_name,
        _MONONUCLEAR_STEMS,
    )
    from .naming_utils import format_suffix_with_locants, get_multiplier_prefix

    stem = node.parent_stem

    # A node uses the BARE-HYDRIDE-STEM grammar (-ane/-ene/-yne + suffix infix)
    # only when its class is in SERIALIZER_PRODUCTION_CLASSES (general_acyclic).
    # Every other node carries a COMPLETE name in parent_stem (retained name,
    # organometallic, coarse fallback) and must NOT get a hydride ending
    # appended (otherwise "acetic acid" -> "acetic acidane").
    hydride_parent = node.class_id in SERIALIZER_PRODUCTION_CLASSES

    # Unsaturation: node.unsaturation_locants = (double_locants, triple_locants).
    double_locants = (
        list(node.unsaturation_locants[0])
        if node.unsaturation_locants and node.unsaturation_locants[0] else []
    )
    triple_locants = (
        list(node.unsaturation_locants[1])
        if node.unsaturation_locants and len(node.unsaturation_locants) > 1
        and node.unsaturation_locants[1] else []
    )

    # D-09 (DERIVE, no field-add): a mononuclear parent is a single-heavy-atom
    # hydride stem with no unsaturation (only "meth" reachable in general_acyclic).
    is_mononuclear = (
        hydride_parent
        and stem in _MONONUCLEAR_STEMS
        and not double_locants
        and not triple_locants
    )

    # --- prefixes -> (text, locants) pairs (mirror _assemble_fragments:1019-1035) ---
    prefix_pairs: List[Tuple[str, tuple]] = []
    for sub in _alphabetize_prefixes(node.prefixes):
        text = _prefix_node_text(sub, style)
        if sub.parenthesization_hint:
            text = f"({text})"
        if sub.multiplicative_prefix:
            text = f"{sub.multiplicative_prefix}{text}"
        prefix_pairs.append((text, tuple(sub.locants)))

    # gap #1: P-14.7 suffix<->prefix locant collision (hydride ring parents only;
    # the general_acyclic-chain path is a no-op early return inside the resolver).
    if (hydride_parent and node.suffix and node.locants and prefix_pairs
            and is_ring_parent_name(stem)):
        prefix_pairs = resolve_suffix_prefix_collision(
            list(node.locants), prefix_pairs,
            is_ring=True, parent_size=_estimate_parent_size_from_name(stem),
        )

    # gap #2: already-has-locant guard before prepending a prefix locant
    # (a prefix whose text already starts with a digit is left untouched).
    prefix_texts: List[str] = []
    for text, locants in prefix_pairs:
        already_has_locant = bool(re.match(r'^\d', text))
        if locants and not already_has_locant:
            loc_str = ",".join(str(l) for l in locants)
            prefix_texts.append(f"{loc_str}-{text}")
        else:
            prefix_texts.append(text)

    # gap #3: P-16.5.1.3.1 mononuclear enclosing marks (+ multiplier carve-out).
    prefix_texts = apply_mononuclear_enclosing(prefix_texts, is_mononuclear)

    # gap #4: inter-prefix hyphenation.
    prefix_str = _join_prefixes(prefix_texts)

    # --- parent + suffix ---
    if hydride_parent:
        # BARE hydride stem -> full grammar (mirror _assemble_fragments:1092-1131).
        if node.suffix:
            suffix_text = node.suffix
            suffix_locants = list(node.locants) if node.locants else []
            # gap #5: on a mononuclear parent a single suffix locant ("1") is
            # meaningless and is omitted (methanol, not methan-1-ol).
            if is_mononuclear and len(suffix_locants) == 1:
                suffix_locants = []
            # Suffix-group multiplicity: a TERMINAL multi-group suffix (dioic
            # acid, dial, dinitrile) carries no locants, so len() under-counts.
            # The count is preserved on node.multiplicative_prefix (set by
            # fragments_to_tree from the suffix fragment's count); otherwise
            # derive it from the locant count. (NameTreeNode has no `count`
            # field — D-09 reuse of an existing field, no schema add.)
            if node.multiplicative_prefix:
                multiplier = node.multiplicative_prefix
            else:
                count = len(suffix_locants)
                multiplier = get_multiplier_prefix(count, suffix_text) if count > 1 else ""
            # gap #6: unsaturation infix (en/yn + euphonic-a) feeds the grammar.
            unsaturation_infix = _build_unsaturation_infix(double_locants, triple_locants)
            # gap #5: full suffix grammar (multiplier + P-16.7.1 vowel elision).
            name = format_suffix_with_locants(
                stem, unsaturation_infix, suffix_text, suffix_locants, multiplier,
            )
        else:
            # gap #6: hydrocarbon branch + WSD-06 ring-bond-locant omission
            # (a SUBSTITUTED cycloalkene keeps its ene-locant; unsubstituted omits).
            _ring_bond_omittable = (not prefix_str) if stem.startswith("cyclo") else None
            name = _build_hydrocarbon_name(
                stem, double_locants, triple_locants,
                ring_bond_locant_omittable=_ring_bond_omittable,
            )
    else:
        # COMPLETE name (retained / organometallic / coarse): parent_stem is
        # already a finished hydride/parent word — pass it through verbatim, NO
        # -ane/-ene/-yne ending. A preserved Type-1 principal-group suffix (e.g.
        # naphthalen-2-ol) is attached with locants but no unsaturation infix
        # (the retained stem subsumes its unsaturation).
        if node.suffix:
            suffix_locants = list(node.locants) if node.locants else []
            if node.multiplicative_prefix:
                multiplier = node.multiplicative_prefix
            else:
                count = len(suffix_locants)
                multiplier = get_multiplier_prefix(count, node.suffix) if count > 1 else ""
            name = format_suffix_with_locants(
                stem, "", node.suffix, suffix_locants, multiplier,
            )
        else:
            name = stem

    # Indicated hydrogen (serializer-only; the legacy fragment assembler has no
    # such field). Empty for general_acyclic and reset retained names; prepended
    # otherwise per P-25.7.
    if node.indicated_h:
        ih_str = ",".join(f"{i}H" for i in node.indicated_h)
        name = f"{ih_str}-{name}"

    # gap #7: prefix -> parent hyphenation (hyphen between a letter/) and a digit).
    if prefix_str:
        name = _join_prefix_to_name(prefix_str, name)

    # gap #8: stereo prepend with NO extra hyphen — the descriptor already
    # carries its own trailing hyphen (e.g. "(2S)-").
    if node.stereo:
        name = f"{node.stereo}{name}"

    return name


def _format_locant_set(locants: Tuple[int, ...]) -> str:
    """Format a sorted tuple of locants as a comma-separated string.

    Example::

        >>> _format_locant_set((1, 2, 3))
        '1,2,3'
        >>> _format_locant_set(())
        ''
    """
    if not locants:
        return ""
    return ",".join(str(l) for l in sorted(locants))


# Phase 179 (WSA-03): the placeholder ``_apply_unsaturation_infix``,
# ``_unsaturation_multiplier``, and ``_format_suffix`` stubs were DELETED.
# Their (divergent, never-production-exercised) logic is superseded by the
# shared ``composition_primitives`` grammar (``_build_unsaturation_infix`` /
# ``_build_hydrocarbon_name`` / ``format_suffix_with_locants``) that
# ``_assemble_explicit_fields`` now calls — one composition logic, byte-
# identical to the legacy assembler (CONTEXT D-02 / D-03, fix-methodology.md:
# no second grammar, no postprocessor).


__all__ = [
    "name_tree_to_string",
    "NameTreeSerializerError",
]
