"""a phase name-tree IR (DECOMP-02 + internal notes).

Authoritative ordered-n-ary tree intermediate representation between
handler computation (Pass-1) and final string emission (Pass-2). Per
internal notes: 12-field frozen dataclass; per internal notes: handlers
return ``NamingResult(name, tree, atom_to_locant_hint)`` and may emit
``tree=None`` during the incremental migration (DECOMP-03 byte-identical
lock binds the ``name`` field only).

Architecture (160-internal notes):
- ``NameTreeNode`` — 12-field frozen dataclass; immutable;
  serializable for ``--dump-tree`` . Tree shape is ordered n-ary
  (``prefixes: Tuple[NameTreeNode,...]`` is variadic per IUPAC
  "complex prefixes are themselves names").
- ``NamingResult`` — NamedTuple returned by every handler.
- Field-coverage map: internal notes-DECOMP.md

The 12-field schema is LOCKED at audit time per internal notes / -27:
adding or removing fields mid-Phase-160 is a Rule 4 architectural
decision, not a silent edit.

Anti-pattern hygiene:
- -15 / -26: predicate purity invariant; NameTreeNode itself
  has no side effects by construction (frozen value type).
- -27: NameTreeNode field additions/removals mid-Phase-160 banned;
  the 12-field schema is the locked spec per internal notes.
- Predicate purity inheritance from a phase: NO mutable defaults;
  every field defaults to ``None`` / ```` / ``False``.
- IUPAC P-section cites for each field-slot live in
  internal notes-DECOMP.md (audit-as-locked-spec per a phase).

References:
- internal notes-DECOMP.md — NameTreeNode Field-Coverage Map.
- 160-internal notes (12-field schema) + (NamingResult shape).
- internal notes (analog: routing/dispatch_table.py:130-148).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple


@dataclass(frozen=True)
class NameTreeNode:
    """One part of a name, and the parts inside it.

    A name is written in the order stereodescriptors, prefixes (in
    alphanumerical order), parent, indicated hydrogen, unsaturation and suffix
    (IUPAC. A node holds those pieces for one parent; each prefix is a
    node of its own, so a substituent with its own substituents is a subtree. The
    node cannot be changed after it is made.

    Attributes
    ----------
    parent_stem: str
        The parent, for example ``'cyclohex'``.
    locants: tuple of int
        Locants of the suffix.
    suffix: str or None
        The suffix, for example ``'ol'``.
    prefixes: tuple of NameTreeNode
        The substituent prefixes, each a node.
    stereo: str or None
        The stereodescriptor part, for example ``'(2R)'``.
    indicated_h: tuple of int
        Locants of indicated hydrogen.
    unsaturation_locants: tuple of (tuple of int, tuple of int)
        Locants of double and of triple bonds.
    class_id: str
        The compound class that built the node.
    multiplicative_prefix: str or None
        A multiplying prefix such as ``'di'``.
    parenthesization_hint: bool
        True when the prefix must be written in enclosing marks.
    iupac_section_cite: str or None
        The section of the recommendations the node follows, for example
        ``''``.
    fragment_legacy: object or None
        An older representation of the same part, kept for the engine's own use.

    Examples
    --------
    >>> from orthonym import name_with_tree
    >>> tree = name_with_tree("OC1CCCCC1").tree
    >>> tree.parent_stem, tree.suffix
    ('cyclohex', 'ol')
    """
    parent_stem: str
    locants: Tuple[int, ...] = ()
    suffix: Optional[str] = None
    prefixes: Tuple["NameTreeNode", ...] = ()
    stereo: Optional[str] = None
    indicated_h: Tuple[int, ...] = ()
    unsaturation_locants: Tuple[Tuple[int, ...], Tuple[int, ...]] = ((), ())
    class_id: str = ""
    multiplicative_prefix: Optional[str] = None
    parenthesization_hint: bool = False
    iupac_section_cite: Optional[str] = None
    fragment_legacy: Optional[object] = None  # composer.NameFragment


class NamingResult(NamedTuple):
    """A name together with the tree of its parts.

    Returned by:func:`orthonym.name_with_tree` and
    :meth:`orthonym.Orthonym.name_with_tree`. It is a named tuple of three
    fields.

    Attributes
    ----------
    name: str
        The name, the same string:func:`orthonym.name_compound` returns.
    tree: NameTreeNode or None
        The parts of the name.
    atom_to_locant_hint: dict of int to int or None
        Atom index to locant, where the part of the engine that built the name
        recorded it.

    Examples
    --------
    >>> from orthonym import name_with_tree
    >>> name, tree, hint = name_with_tree("OC1CCCCC1")
    >>> name
    'cyclohexanol'
    """
    name: str
    tree: Optional["NameTreeNode"] = None
    atom_to_locant_hint: Optional[Dict[int, int]] = None


def _normalize_locants(locants: Tuple[int, ...]) -> Tuple[int, ...]:
    """Sort ascending + dedupe + freeze to a tuple.

    Used by handlers to guarantee invariant ``locants`` is canonical
    (sorted, deduplicated) for stable serialization by ``name_tree_to_string``.

    Example:
        >>> _normalize_locants((3, 1, 1, 2))
        (1, 2, 3)
        >>> _normalize_locants()
        
    """
    return tuple(sorted(set(locants)))


def _alphabetize_prefixes(
    prefixes: Tuple[NameTreeNode, ...],
) -> Tuple[NameTreeNode, ...]:
    """Sort prefixes by IUPAC alpha_sort_key on each subtree's parent_stem.

    Mirrors composer.py:7685 sort discipline (the same ``alpha_sort_key``
    helper consumed by the legacy ``_assemble_fragments`` path). This means
    the alphabetization rules from IUPAC (which ignore multiplicative
    prefixes di-/tri-/tetra-/penta- but include iso-/neo-/sec-/tert-/cyclo-)
    are applied consistently between Pass-1 IR construction and Pass-2
    serialization.

    Example::

        >>> a = NameTreeNode(parent_stem="methyl")
        >>> b = NameTreeNode(parent_stem="ethyl")
        >>> _alphabetize_prefixes((a, b))
        (NameTreeNode(parent_stem='ethyl',...), NameTreeNode(parent_stem='methyl',...))
    """
    # Lazy import to avoid an import cycle if naming_utils imports back from
    # name_tree (unlikely today, but mirrors the lazy-import pattern in
    # composer.py:146 + handlers/* per PATTERNS § Lazy Import).
    from .naming_utils import alpha_sort_key, cip_descriptor_rank_key
    # (the Blue Book): prefixes that differ only in configuration
    # tie on alpha_sort_key and are then ordered by their descriptors, R before S.
    return tuple(sorted(prefixes, key=lambda p: (alpha_sort_key(p.parent_stem),
                                                 cip_descriptor_rank_key(p.parent_stem))))


def is_coarse_node(node: NameTreeNode) -> bool:
    """a phase SCORE-02 : single source of truth for the coarse/structured
    classification used by BOTH the coarse-bucket metric script and the handler
    contract test.

    Provenance-based: a node is "coarse" when it is a flat single-node tree that
    carries ONLY the final string in ``fragment_legacy`` with no recoverable
    structural decomposition — i.e. empty ``prefixes``, no ``suffix``,
    ``fragment_legacy`` set, AND ``parent_stem == fragment_legacy`` (both written
    to the same final string by a coarse construction). A "structured" node has a
    bare stem in ``parent_stem`` (e.g. "but") distinct from the full assembled
    name in ``fragment_legacy`` ("butane"), or carries real prefix/suffix parts,
    or has ``fragment_legacy is None`` (the explicit-field reference handler).

     rationale: the metric script and the contract test previously defined
    this twice with a DIFFERENT last clause (``parent_stem == fragment_legacy``
    in the script vs ``parent_stem == name`` in the test). Those are not
    equivalent in general, so the public-facing "structured %" headline could
    drift from what the test counts. The provenance form below is the robust one
    (per the script's own docstring): it compares against the node's own
    ``fragment_legacy`` rather than the post-processed final ``name``, so it is
    invariant under downstream name rewriting.
    """
    return (
        node.fragment_legacy is not None
        and not node.prefixes
        and node.suffix is None
        and node.parent_stem == node.fragment_legacy
    )


__all__ = [
    "NameTreeNode",
    "NamingResult",
    "_normalize_locants",
    "_alphabetize_prefixes",
    "is_coarse_node",
]
