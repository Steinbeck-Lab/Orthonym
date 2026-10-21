"""The N-substituents of an amide whose amine fragment is a secondary amine.

The decomposition cuts an amide at its C-N bond. A secondary amine fragment is named
``N-ethylethanamine`` or ``N-methylcyclohexanamine``: one substituent cited with the locant N
and one substituent that is the parent amine's own group. Both stay on the nitrogen of the
amide, so each is cited with the locant N Substitution of primary amides,
, the Blue Book: R' and R'' of R-CO-NR'R'' "are cited as prefixes preceded
by the locant N"; examples ``N,N-diethylfuran-2-carboxamide (PIN)``:32797 and
``N-methyl-N-phenylbenzamide (PIN)``:32871), identical ones multiplied and different ones in
alphanumerical order. The amine of an N-alkoxy amide is named the same way
(``N-methoxymethanamine (PIN)``,,:38380), and its alkoxy substituent is cited
as a prefix on the amide nitrogen like any other: "N-SUBSTITUTION OF AMINES AND IMINES BY
HETEROATOMS" (:26588), "Substitution of amines is permitted by -OR... groups" (:26596).
"""

import re
from typing import Callable, List, Optional, Tuple

from rdkit import Chem

from ..assembly.naming_utils import (
    _RETAINED_CONTRACTED_ALKOXY,
    _is_fully_enclosed,
    _wrap_n_substituent,
    enclose_if_compound,
    matching_close,
    multiplied_component,
    prefix_citation_sort_key,
)

# The retained contracted R-O- prefixes are simple prefixes cited bare, so an N-substituent
# 'methoxy' stands in front of its parent amine unenclosed: 'N-methoxymethanamine (PIN)'
# "N,O-Disubstitution of hydroxylamine", the Blue Book, example:38380).
# "Retained names" (:27665) lists methoxy, ethoxy, propoxy, butoxy and phenoxy
# (the vocabulary of naming_utils) and 'tert-butoxy' (:27679).
_BARE_ALKOXY = "|".join(sorted(_RETAINED_CONTRACTED_ALKOXY | {"tert-butoxy"}, key=len, reverse=True))

# the first substituent word of an unenclosed name: 'ethyl' of 'ethylethanamine',
# 'tert-butyl' of 'tert-butyl-2-methylpropan-2-amine', 'methoxy' of 'methoxymethanamine'
_BARE_SUBSTITUENT = re.compile(
    rf"(?:(?:tert|sec)-)?[a-z]+?yl(?=[-a-z])|(?:{_BARE_ALKOXY})(?=[-a-z])")


def split_secondary_amine_name(amine_name: str) -> Optional[Tuple[str, str]]:
    """``'N-ethylethanamine'`` -> ``('ethyl', 'ethanamine')``; the N-substituent keeps its
    enclosing marks (``'N-(2-hydroxyethyl)ethanamine'`` -> ``('(2-hydroxyethyl)', 'ethanamine')``).

    ``None`` unless the name is one N-substituent in front of a parent amine."""
    if not amine_name.startswith("N-") or len(amine_name) < 4:
        return None
    rest = amine_name[2:]
    if rest[0] in "([{":
        close = matching_close(rest, 0)
        if close is None or close + 1 >= len(rest):
            return None
        return rest[: close + 1], rest[close + 1:].lstrip("-")
    match = _BARE_SUBSTITUENT.match(rest)
    if match is None:
        return None
    return match.group(0), rest[match.end():].lstrip("-")


def _enclosed(substituent: str) -> str:
    return _wrap_n_substituent(enclose_if_compound(substituent))


def _groups_on_nitrogen(amine_smiles: Optional[str]) -> Optional[Tuple[str, str]]:
    """The canonical SMILES of the two groups on the nitrogen of an acyclic secondary amine
    (stereo kept, the attachment marked by a dummy atom), or None for any other amine."""
    mol = Chem.MolFromSmiles(amine_smiles) if amine_smiles else None
    if mol is None:
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetDegree() == 2
                 and a.GetTotalNumHs() == 1 and not a.IsInRing()]
    if len(nitrogens) != 1:
        return None
    n = nitrogens[0]
    bonds = [mol.GetBondBetweenAtoms(n.GetIdx(), x.GetIdx()).GetIdx() for x in n.GetNeighbors()]
    cut = Chem.FragmentOnBonds(mol, bonds, addDummies=True, dummyLabels=[(0, 0)] * len(bonds))
    groups = [Chem.MolFragmentToSmiles(cut, atomsToUse=list(atoms))
              for atoms in Chem.GetMolFrags(cut) if n.GetIdx() not in atoms]
    return (groups[0], groups[1]) if len(groups) == 2 else None


def amide_n_substituent_block(
    amine_name: str, amine_to_prefix: Callable[[str], Optional[str]],
    amine_smiles: Optional[str] = None,
) -> Optional[Tuple[str, List[str]]]:
    """The cited N-substituents of the amide made from ``amine_name`` and the substituent
    words in citation order: ``('N,N-diethyl', ['ethyl', 'ethyl'])``,
    ``('N-cyclohexyl-N-methyl', ['cyclohexyl', 'methyl'])``. ``None`` when the amine name is
    not one N-substituent on a parent amine whose own group has a prefix form.

    Two groups are identical when the structure says so (``'N-tert-butyl-2-methylpropan-2-amine'``
    names one group twice, as 'tert-butyl' and as '2-methylpropan-2-yl'); the name of the
    N-substituent is then cited for both."""
    split = split_secondary_amine_name(amine_name)
    if split is None:
        return None
    n_substituent, parent_amine = split
    parent_prefix = amine_to_prefix(parent_amine)
    if not parent_prefix or parent_prefix.startswith(("N-", "N,")):
        return None
    groups = _groups_on_nitrogen(amine_smiles)
    if groups is not None and groups[0] == groups[1]:
        parent_prefix = n_substituent
    pair = [_enclosed(n_substituent), _enclosed(parent_prefix)]
    if pair[0] == pair[1]:
        bare = pair[0][1:-1] if _is_fully_enclosed(pair[0]) else pair[0]
        return f"N,N-{multiplied_component(2, bare, pair[0])}", pair
    first, second = sorted(pair, key=prefix_citation_sort_key)
    return f"N-{first}-N-{second}", [first, second]
