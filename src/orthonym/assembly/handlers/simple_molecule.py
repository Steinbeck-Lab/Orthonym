"""Phase 160 simple_molecule handler — LIFT (single-atom / very simple).

Verbatim move of composer.py:1388-1392 dispatch logic. The underlying
``_name_simple_molecule`` body STAYS in composer.py during Plan-02 and
moves to this module in Plan-03 commit 03-10 (composer.py thinning) per
CONTEXT incremental migration.

IUPAC cite: P-14 (simple molecules; noble gases / single-atom symbols).

Byte-identical contract: the predicate matches molecules with neither a
principal chain nor a ring system (atoms / noble gases / very simple
molecules); no other handler in the cascade applies in that case, so
moving this to inner-dispatch (which fires before the inline cascade)
does NOT change behavior — the molecule routes identically.

References:
- composer.py:1388-1392 (inline simple-molecule branch).
- composer.py:_name_simple_molecule body at ~ line 2545 (STAYS until 03-10).
- 160-AUDIT-DECOMP.md § 1 row 'simple_molecule' + § 2.34 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_simple_molecule(features: Any) -> bool:
    """Mirrors composer.py:1390 (``not principal_chain and not ring_systems``).

    Atoms / noble gases / very simple molecules where neither a chain nor
    a ring was perceived.
    """
    if getattr(features, 'principal_chain', None):
        return False
    if getattr(features, 'ring_systems', None):
        return False
    return True


def name_simple_molecule(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return simple-molecule handler.

    Plan-10 DECOMP-02 OBSERVABLE CLOSURE: this is the FIRST tree-emitting
    handler in Orthonym. The NameTreeNode contains only ``parent_stem``
    (the noble-gas / atom name) because simple molecules have no locants,
    no prefixes, no suffix, no stereo. ``name_tree_to_string(tree)`` is
    byte-identical to ``result.name``.
    """
    from ..composer import _name_simple_molecule

    name = _name_simple_molecule(features)
    if not name:
        return None
    tree = NameTreeNode(
        parent_stem=name,
        class_id='simple_molecule',
        iupac_section_cite='P-14',
    )
    return NamingResult(name=name, tree=tree, atom_to_locant_hint=None)


__all__ = ["name_simple_molecule", "_is_simple_molecule"]
