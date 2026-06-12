"""Phase 165 (SCORE-01/02) — pure ``fragments_to_tree`` deriver.

The single genuinely-new artifact of Plan 165-01. A side-effect-free transform
``List[NameFragment] -> NameTreeNode`` that mirrors the ``_assemble_fragments``
classification loop (``handlers/_handler_shared.py:779-787``) and parent
unsaturation-locant unpacking (``:877-880``), producing the structured Name-Tree
IR consumed by SCORE-02 / Phase 166 per-substring scoring.

D-02 DUAL-CARRY: every derived node ALSO carries ``fragment_legacy`` — the
pre-assembled final name string — so ``name_tree_to_string`` round-trips
byte-identically (SC-1). A single synthetic ``NameFragment`` CANNOT reproduce a
multi-fragment concatenation (the parent path appends ``"ane"`` via
``_build_hydrocarbon_name``), so the byte-identical carrier is the final string
itself, returned verbatim by the ``name_tree_to_string`` str short-circuit
(RESEARCH lines 459-462: "a dedicated short-circuit that stores the final
string"). The structured fields and the string view both derive from the SAME
fragment list, so they cannot disagree.

Purity (AP-160-15 / D-25): NO mutation of ``fragments``, its elements, or module
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
    byte-identical to the production ``assembled`` for P-16.5.1.3.1 mononuclear
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
                locants=_normalize_locants(p.locants),
                multiplicative_prefix=(None if p.count <= 1 else get_multiplier(p.count)),
            )
            for p in prefix_frags
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
        # D-02 byte-identical carrier (verbatim via name_tree_to_string str path).
        fragment_legacy=_synthesize_root_fragment(
            fragments, is_mononuclear_parent=is_mononuclear_parent
        ),
    )


__all__ = ["fragments_to_tree"]
