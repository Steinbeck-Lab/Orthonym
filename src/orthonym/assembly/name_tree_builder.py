"""a phase (SCORE-01/02) — pure ``fragments_to_tree`` deriver.

The single genuinely-new artifact of Plan 165-01. A side-effect-free transform
``List[NameFragment] -> NameTreeNode`` that mirrors the ``_assemble_fragments``
classification loop (``handlers/_handler_shared.py:779-787``) and parent
unsaturation-locant unpacking (``:877-880``), producing the structured Name-Tree
IR consumed by SCORE-02 / a phase per-substring scoring.

 DUAL-CARRY: every derived node ALSO carries ``fragment_legacy`` — the
pre-assembled final name string — so ``name_tree_to_string`` round-trips
byte-identically . A single synthetic ``NameFragment`` CANNOT reproduce a
multi-fragment concatenation (the parent path appends ``"ane"`` via
``_build_hydrocarbon_name``), so the byte-identical carrier is the final string
itself, returned verbatim by the ``name_tree_to_string`` str short-circuit
(RESEARCH lines 459-462: "a dedicated short-circuit that stores the final
string"). The structured fields and the string view both derive from the SAME
fragment list, so they cannot disagree.

Purity (-15 /): NO mutation of ``fragments``, its elements, or module
globals. ``NameFragment`` instances are read, never rewritten.
"""
from __future__ import annotations

from typing import List, Optional

from .name_tree import NameTreeNode, _alphabetize_prefixes, _normalize_locants


def _synthesize_root_fragment(
    fragments: List["object"], is_mononuclear_parent: bool = False
) -> str:
    """Pre-assemble the final name string for the byte-identical ``fragment_legacy``.

    ``name_tree_to_string`` short-circuits on a ``str`` ``fragment_legacy`` by
    returning it verbatim. ``_assemble_fragments`` does not use its ``style``
    argument today, so the carried string is style-independent and
    byte-identical to the production path. ``_assemble_fragments`` does not
    mutate its input (verified), so this stays pure.

    ``is_mononuclear_parent`` is threaded so the carried string stays
    byte-identical to the production ``assembled`` for mononuclear
    parents (otherwise the structured carrier and the production name would
    disagree on ``bromo(chloro)methanol`` vs ``bromochloromethan-1-ol``).
    """
    # Lazy import to avoid the composer.py -> name_tree_to_string -> composer.py
    # import cycle (PATTERNS § Lazy Import; mirrors name_tree.py:165-168).
    from .composer import _assemble_fragments

    return _assemble_fragments(
        list(fragments), "pin", is_mononuclear_parent=is_mononuclear_parent
    )


def fragments_to_tree(
    fragments: List["object"],
    *,
    class_id: str = "",
    section_cite: Optional[str] = None,
    is_mononuclear_parent: bool = False,
) -> NameTreeNode:
    """Derive a structured ``NameTreeNode`` from a ``List[NameFragment]`` (pure).

    Mirrors ``_assemble_fragments`` term-for-term:
    - classification loop (``_handler_shared.py:779-787``)
    - parent ``locants`` is a ``(double_bond_locants, triple_bond_locants)``
      tuple, unpacked into ``unsaturation_locants`` (``:877-880``)
    - prefixes alphabetized via ``_alphabetize_prefixes`` (same ``alpha_sort_key``
      as the legacy sort at ``:854``); locants canonicalized via
      ``_normalize_locants``.
    """
    # Lazy import (cycle avoidance): the multiplicative-prefix helper.
    from .composer import get_multiplier

    stereo: Optional[str] = None
    parent_frag = None
    suffix_frag = None
    prefix_frags = []
    for frag in fragments:
        if frag.fragment_type == "stereo":
            stereo = frag.text or None
        elif frag.fragment_type == "parent":
            parent_frag = frag
        elif frag.fragment_type == "suffix":
            suffix_frag = frag
        elif frag.fragment_type == "prefix":
            prefix_frags.append(frag)

    # parent.locants holds (double_locants, triple_locants) per
    # _assemble_fragments:877-880 — NOT a flat locant list.
    unsat: tuple = ((), ())
    if parent_frag and parent_frag.locants:
        double = tuple(parent_frag.locants[0]) if parent_frag.locants[0] else ()
        triple = (
            tuple(parent_frag.locants[1])
            if len(parent_frag.locants) > 1 and parent_frag.locants[1]
            else ()
        )
        unsat = (double, triple)

    prefixes = _alphabetize_prefixes(
        tuple(
            NameTreeNode(
                parent_stem=p.text,
                # a phase (-03): preserve prefix locants VERBATIM (order +
                # repeats), matching the legacy ``",".join(f.locants)`` at
                # _handler_shared.py:1032. ``_normalize_locants`` (sorted+set-
                # deduped) silently dropped the repeated locants a polysubstituted
                # substituent needs — e.g. ``1,1,1,2,2,...-tridecafluoro`` collapsed
                # to ``1,2,...``. Upstream already emits them ascending, so the
                # well-formed single-occurrence cases are unchanged (byte-identical).
                locants=tuple(p.locants),
                multiplicative_prefix=(None if p.count <= 1 else get_multiplier(p.count)),
            )
            for p in prefix_frags
        )
    )

    # a phase (-03): preserve the suffix-group multiplicity so the
    # name-tree serializer (_assemble_explicit_fields) can reproduce a TERMINAL
    # multi-group suffix (dioic acid / dial / dinitrile) whose locants are
    # omitted. The legacy assembler computes the multiplier from
    # ``max(len(suffix_locants), suffix_frag.count)`` at assembly time; that
    # count is otherwise lost when the locked 12-field schema (-27) drops
    # it. It is carried on the root's otherwise-unused ``multiplicative_prefix``
    # — a coherent semantic (the multiplier of THIS node's head term, here the
    # principal characteristic group). Empty/1-count suffixes carry None
    # (byte-identical to today). Does NOT affect ``fragment_legacy`` (the
    # carrier) nor the per-substring scorer (which reads parent/locant only).
    suffix_multiplier: Optional[str] = None
    if suffix_frag is not None:
        _suffix_count = max(
            len(suffix_frag.locants or ()), getattr(suffix_frag, "count", 1)
        )
        if _suffix_count > 1:
            from .naming_utils import get_suffix_multiplier_prefix
            suffix_multiplier = get_suffix_multiplier_prefix(_suffix_count, suffix_frag.text)

    # a phase (-03) production flip: for a class in
    # SERIALIZER_PRODUCTION_CLASSES the str carrier is DROPPED (fragment_legacy
    # =None) so name_tree_to_string runs the explicit-field path (the structured
    # fields above are byte-identical-complete per Plan 01) instead of returning
    # the legacy string verbatim. This is HALF of the single-source-of-truth
    # coupling (the other half is the composer seam routing); both gate on the
    # SAME frozenset so the flip cannot be a silent no-op (Pitfall 2). Carrier
    # classes keep the byte-identical str carrier (unchanged).
    from .name_tree_to_string import SERIALIZER_PRODUCTION_CLASSES
    fragment_legacy = (
        None
        if class_id in SERIALIZER_PRODUCTION_CLASSES
        else _synthesize_root_fragment(
            fragments, is_mononuclear_parent=is_mononuclear_parent
        )
    )

    return NameTreeNode(
        parent_stem=parent_frag.text if parent_frag else "",
        suffix=suffix_frag.text if suffix_frag else None,
        locants=_normalize_locants(suffix_frag.locants) if suffix_frag else (),
        prefixes=prefixes,
        stereo=stereo,
        unsaturation_locants=unsat,
        class_id=class_id,
        iupac_section_cite=section_cite,
        multiplicative_prefix=suffix_multiplier,
        fragment_legacy=fragment_legacy,
    )


__all__ = ["fragments_to_tree"]
