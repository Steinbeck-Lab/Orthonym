"""Pure-hydrocarbyl substituent-name guard (shared, fail-closed).

Single source of truth for the question "is this substituent a simple *unbranched*
alkyl or a phenyl/naphthyl aryl, and if so what is its prefix name?". Used by the
parent-hydride namers — mononuclear hydrides (P-68 arsane/stibane), chalcogen
chains (P-21.2.2 trisulfane) and polyazanes (P-68.3 hydrazine/diazene).

It wraps ``rules.phosphorus._characterize_substituent``, which follows ONLY the
carbon skeleton and therefore (a) silently DROPS a hanging heteroatom
(2-hydroxyethyl -> "ethyl"), (b) cannot tell propan-1-yl from propan-2-yl
(isopropyl -> "propyl"), and (c) miscounts a ring or benzyl as a linear alkyl
(cyclohexyl -> "hexyl", benzyl -> "heptyl"). The purity walk here rejects every
one of those so only the cases _characterize_substituent names CORRECTLY pass;
everything else returns None and the caller fail-closes.

NOT a SMARTS broadening (feedback_smarts_and_seniority): a graph walk whose every
guard narrows.
"""
from typing import Optional

from rdkit import Chem

from .phosphorus import _characterize_substituent


def pure_organyl_prefix_name(mol, start_idx: int, exclude_idx: int) -> Optional[str]:
    """Return the substituent prefix name (``methyl`` / ``ethyl`` / ``phenyl`` …)
    for a substituent rooted at ``start_idx`` and attached to ``exclude_idx``, iff
    it is a pure-hydrocarbon unbranched alkyl (attached at a chain terminus) or a
    phenyl/naphthyl aryl, else None.

    Pure: no mol mutation.
    """
    subtree = []
    seen = {exclude_idx}
    stack = [start_idx]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'C':          # only a carbon skeleton is allowed
            return None
        subtree.append(i)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j == exclude_idx or nbr.GetSymbol() == 'H':
                continue
            stack.append(j)

    if not subtree:
        return None

    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in subtree):
        subtree_set = set(subtree)

        def _sub_carbons(idx):
            return [n for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if n.GetIdx() in subtree_set]

        # The attachment atom must be a chain TERMINUS (<=1 in-subtree carbon):
        # _characterize_substituent emits the n-alkyl name, so an INTERNAL
        # attachment (isopropyl / sec-butyl) would be mislabelled.
        if len(_sub_carbons(start_idx)) > 1:
            return None
        for i in subtree:
            atom = mol.GetAtomWithIdx(i)
            # a ring in a non-fully-aromatic subtree (cyclohexyl, benzyl) is
            # miscounted as a linear alkyl -> fail-closed.
            if atom.IsInRing():
                return None
            if len(_sub_carbons(i)) > 2:     # branch point -> fail-closed
                return None
            for bond in atom.GetBonds():
                if (bond.GetBondType() != Chem.BondType.SINGLE
                        and not bond.GetIsAromatic()):
                    return None              # vinyl / alkynyl -> mislabel risk

    result = _characterize_substituent(mol, start_idx, {exclude_idx})
    return result[1] if result is not None else None


__all__ = ["pure_organyl_prefix_name"]
