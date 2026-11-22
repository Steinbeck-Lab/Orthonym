"""The 'carbamimidoyl' prefix of an amidine group -C(=NH)-N(R)R' in a branch, with its N-substituents.

;:17832 "H2N-C(=NH)-NH carbamimidoylamino (preferred prefix)": 'carbamimidoyl', substituted on
the amino nitrogen as 'N-methylcarbamimidoyl', 'N,N-dimethylcarbamimidoyl'; the book's
'4-[methyl(N-methyl-N-phenylcarbamimidoyl)amino]benzoic acid (PIN)' (:34286) carries two substituents there. Moved
out of ``substituent_naming._name_substituent_fragment_uncached`` (a function of the size baseline), which named none or
one; with two, the branch fell to the general '[(dimethylamino)(imino)methyl]' form.
"""
from __future__ import annotations

from typing import List, Optional


def carbamimidoyl_on_n(mol, attach_idx: int, im_n: int, am_n, sub_set: set, parent_set: set) -> Optional[str]:
    """'carbamimidoyl', 'N-{R}carbamimidoyl' or 'N-{R}-N-{R'}carbamimidoyl' for the branch ``sub_set`` whose
    attachment carbon ``attach_idx`` carries the =NH nitrogen ``im_n`` and the amino nitrogen ``am_n`` (an Atom); None
    when a part of the branch is not accounted for (no silent drop) or a substituent has no prefix name."""
    from ..rules.benzene import _build_amidine_n_prefix
    from .naming_utils import enclose_if_compound
    from .substituent_naming import name_substituent_fragment
    seeds = [y.GetIdx() for y in am_n.GetNeighbors() if y.GetIdx() != attach_idx]
    if not seeds:
        return 'carbamimidoyl'
    if len(seeds) > 2 or (len(seeds) == 2 and any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in seeds)):
        return None     # an N-amino, N-hydroxy or N-sulfanyl second substituent is no plain N,N-disubstituted group
    block = {im_n, am_n.GetIdx(), attach_idx}
    covered = set(block)
    names: List[str] = []
    for seed in seeds:
        atoms: List[int] = []
        stack = [seed]
        while stack:
            cur = stack.pop()
            if cur in covered:
                continue
            covered.add(cur)
            atoms.append(cur)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                         if nb.GetIdx() in sub_set and nb.GetIdx() not in covered)
        if not atoms:
            return None
        name = name_substituent_fragment(mol, atoms, seed, list(parent_set | {attach_idx, am_n.GetIdx()}))
        if not name or ' ' in name:
            return None
        names.append(enclose_if_compound(name))     # 'N-(2-methoxyethyl)-N-methyl',:7232)
    if covered != set(sub_set):
        return None
    prefix = _build_amidine_n_prefix([('N', name) for name in names])
    return f"{prefix}carbamimidoyl" if prefix else None
