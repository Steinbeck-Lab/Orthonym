""" name-tree IR (+ CONTEXT).

Authoritative ordered-n-ary tree intermediate representation between
handler computation (Pass-1) and final string emission (Pass-2). Per
CONTEXT: 12-field frozen dataclass; per CONTEXT: handlers
return ``NamingResult(name, tree, atom_to_locant_hint)`` and may emit
``tree=None`` during the incremental migration (byte-identical
lock binds the ``name`` field only).

Architecture (160-CONTEXT.md):
- ``NameTreeNode`` — 12-field frozen dataclass; immutable;
  serializable for ``--dump-tree``. Tree shape is ordered n-ary
  (``prefixes: Tuple[NameTreeNode,...]`` is variadic per IUPAC P-23
  "complex prefixes are themselves names").
- ``NamingResult`` — NamedTuple returned by every handler.
- Field-coverage map: 160-AUDIT-DECOMP.md § 5.

The 12-field schema is LOCKED at audit time per CONTEXT /:
adding or removing fields mid-Phase-160 is a Rule 4 architectural
decision, not a silent edit.

Anti-pattern hygiene:
- /: predicate purity invariant; NameTreeNode itself
  has no side effects by construction (frozen value type).
-: NameTreeNode field additions/removals mid-Phase-160 banned;
  the 12-field schema is the locked spec per CONTEXT.
- Predicate purity inheritance: NO mutable defaults;
  every field defaults to ``None`` / ``()`` / ``False``.
- IUPAC P-section cites for each field-slot live in
  160-AUDIT-DECOMP.md § 5 (audit-as-locked-spec).

References:
- 160-AUDIT-DECOMP.md § 5 — NameTreeNode Field-Coverage Map.
- 160-CONTEXT.md (12-field schema) + (NamingResult shape).
- 160-PATTERNS.md § 1 (analog: routing/dispatch_table.py:130-148).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple


@dataclass(frozen=True)
class NameTreeNode:
    """: 12-field frozen ordered-n-ary IR node.

    Composition order per IUPAC P-14.5::

        [stereo] + [prefixes (alphabetized)] + [parent_stem] + [indicated_h]
                 + [unsaturation_infix] + [suffix]

    ``fragment_legacy`` carries the legacy ``NameFragment`` leaf rep
    (composer.py:537) for incremental migration; handlers MAY return
    ``tree=None`` per (byte-identical lock binds the
    ``name`` field only).

    Per CONTEXT: 12 fields exactly; frozen; immutable. Attempting
    ``node.parent_stem = 'X'`` raises ``FrozenInstanceError``.

    Field-coverage map (per 160-AUDIT-DECOMP.md § 5):

    +-------------------------------+--------------------------+---------------+
    | field_name | iupac_p_section_cite | required? |
    +===============================+==========================+===============+
    | parent_stem | P-14.5 / P-23 / P-44 | REQUIRED |
    +-------------------------------+--------------------------+---------------+
    | locants | P-14.5 / P-14.7 | optional |
    +-------------------------------+--------------------------+---------------+
    | suffix | P-65 / P-14.5 | optional |
    +-------------------------------+--------------------------+---------------+
    | prefixes | P-23 / P-14.2.2 | optional |
    +-------------------------------+--------------------------+---------------+
    | stereo | P-91 / P-14.5 | optional |
    +-------------------------------+--------------------------+---------------+
    | indicated_h | P-25.7 / P-31.1.4.3 | optional |
    +-------------------------------+--------------------------+---------------+
    | unsaturation_locants | P-14.5 / P-31.1 | optional |
    +-------------------------------+--------------------------+---------------+
    | class_id | (audit metadata) | optional |
    +-------------------------------+--------------------------+---------------+
    | multiplicative_prefix | P-14.2.2 / P-14.5.1 | optional |
    +-------------------------------+--------------------------+---------------+
    | parenthesization_hint | P-51.3.5 / P-14.5.1 | optional |
    +-------------------------------+--------------------------+---------------+
    | iupac_section_cite | (audit metadata) | optional |
    +-------------------------------+--------------------------+---------------+
    | fragment_legacy | (migration handle) | optional |
    +-------------------------------+--------------------------+---------------+
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
    """: handler return shape.

    Per: handlers return ``NamingResult(name, tree, atom_to_locant_hint)``;
    ``tree`` is Optional during incremental migration; ``atom_to_locant_hint``
    is the locant map the post-handler stereo injector consumes
    (composer.py:807, 1681-1683 today).

    Fields:
        name: REQUIRED — byte-identical contract per. This is the
            single source of truth at first wave; handlers SHOULD route this
            through the existing composer.py pool.add() / _inject_stereo_if_missing
            pipeline for byte-identical preservation.
        tree: Optional first-wave; populated incrementally per v19+ phases.
            ``None`` is the DEFAULT for /03 ship (per 160-AUDIT-DECOMP.md
            § 5.2). When non-None, the tree's serialization via
            ``name_tree_to_string()`` MUST equal the ``name`` field byte-for-byte.
        atom_to_locant_hint: Forwarded to the post-handler stereo injector
            (composer.py:_inject_stereo_if_missing). For ring handlers this
            is typically ``features.heterocycle_atom_to_locant`` or
            ``features.benzene_atom_to_locant``.
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
        >>> _normalize_locants(())
        ()
    """
    return tuple(sorted(set(locants)))


def _alphabetize_prefixes(
    prefixes: Tuple[NameTreeNode, ...],
) -> Tuple[NameTreeNode, ...]:
    """Sort prefixes by IUPAC P-13 alpha_sort_key on each subtree's parent_stem.

    Mirrors composer.py:7685 sort discipline (the same ``alpha_sort_key``
    helper consumed by the legacy ``_assemble_fragments`` path). This means
    the alphabetization rules from IUPAC P-13 (which ignore multiplicative
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
    from .naming_utils import alpha_sort_key
    return tuple(sorted(prefixes, key=lambda p: alpha_sort_key(p.parent_stem)))


def is_coarse_node(node: NameTreeNode) -> bool:
    """ (WR-1): single source of truth for the coarse/structured
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

    WR-4 rationale: the metric script and the contract test previously defined
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
