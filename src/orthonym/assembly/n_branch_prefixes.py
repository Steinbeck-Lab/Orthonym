"""The last-resort prefix of an N-attached branch whose ring the direct paths could not name.

Moved out of ``composer._name_n_attached_substituent_fallback`` (a function of the size baseline)
so that the branch that is a sulfonamide can be named the way the book names it.

* ``-NH-SO2-R``: "Substituents of the types -NH-CO-R and -NH-SO2-R" (the Blue Book):
  'Method (1) generates preferred IUPAC names': '3-(methanesulfonamido)propanoic acid (PIN)':33019,
  '(1-cyclohexylmethanesulfonamido)acetic acid':33024; the form '[(cyclohexylmethanesulfonyl)amino]'
  is method (2), a general-nomenclature name. The shared helper builds the method (1) prefix and fails
  closed on a branch it cannot spell, so it is tried first.
* any other branch (audit 2026-09-03, Bug B): name the CARBON side that hangs off the nitrogen and wrap
  it, '(cyclohexylamino)' / '((prefix)amino)'. The previous version put the nitrogen itself into a SMILES
  fragment, named it as an amine ('4-methylcyclohexan-1-amine', 'N-methylaniline') and wrapped THAT in
  'amino' -- one nitrogen twice; every candidate was a different molecule. Only a single carbon branch on
  the nitrogen is handled: with two (N-methyl-N-aryl, N-alkyl-N-cycloalkyl) the prefix needs the N-locant
  grammar of the later paths, so this declines and lets them name it.
"""
from __future__ import annotations

from typing import Any, List, Optional


def n_branch_last_resort_prefix(mol: Any, atom: Any, attach_atom: int, sub_set: set,
                                sub_atoms: List[int], chain_set: set) -> Optional[str]:
    """'(...amino)' / '(...sulfonamido)' for the N-attached branch ``sub_atoms``, or None."""
    if len(sub_atoms) > 25:
        return None
    from ..errors import is_refusal_sentinel
    from .composer import _has_ring_atoms, _name_reflects_ring
    from .naming_utils import apply_enclosing_marks, needs_brackets
    from .substituent_naming import name_substituent_fragment, sulfonamido_prefix_from_n_branch

    sulfonamido = sulfonamido_prefix_from_n_branch(mol, attach_atom, sub_set, chain_set)
    if sulfonamido:
        return f"({sulfonamido})"
    n_branches = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() in sub_set]
    carbon_side = [idx for idx in sub_atoms if idx != attach_atom]
    if len(n_branches) != 1 or not carbon_side:
        return None
    try:
        prefix = name_substituent_fragment(
            mol, carbon_side, n_branches[0], list(chain_set) + [attach_atom])
    except Exception:  # noqa: BLE001 - a declining namer is a decline
        return None
    if not prefix or is_refusal_sentinel(prefix):
        return None
    if _has_ring_atoms(mol, carbon_side) and not _name_reflects_ring(prefix):
        return None
    if needs_brackets(prefix):
        # (the Blue Book): '[(4-fluorophenyl)methyl]amino', not '((4-fluorophenyl)methyl)amino';
        # the outer mark of the returned '(...amino)' is re-chosen by format_substituent_prefix (TRIAGE g3 C10c).
        prefix = apply_enclosing_marks(prefix, -1)
    return f"({prefix}amino)"


def hetero_root_prefix(mol: Any, sub_atoms: List[int], attach_idx: int) -> Optional[str]:
    """The prefix of a branch rooted at O, S or N for the functional-group namer of the wider tiers: an
    ``-NH-SO2-R`` branch is the method (1) '{R}sulfonamido' prefix, the Blue Book), as at the PIN
    tier ('[(1E)-2-phenylethene-1-sulfonamido]acetate', not '{[...sulfonyl]amino}'); any other branch is named by
    ``substituent_naming._located_fg_hetero_root``."""
    from .n_acyl_amido import sulfonamido_prefix
    from .substituent_naming import _located_fg_hetero_root
    atom = mol.GetAtomWithIdx(attach_idx)
    if atom.GetSymbol() == 'N':
        parent = {nb.GetIdx() for nb in atom.GetNeighbors()} - set(sub_atoms)
        sulfonamido = sulfonamido_prefix(mol, attach_idx, sub_atoms, parent)
        if sulfonamido:
            return sulfonamido
    return _located_fg_hetero_root(mol, sub_atoms, attach_idx)
