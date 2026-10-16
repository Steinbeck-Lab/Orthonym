"""The branch an amine nitrogen carries on a chain carbon, whole.

 "Selection of the principal chain or senior ring system in secondary and tertiary amines"
(the Blue Book): "When a choice for parent hydride is possible between a ring and a chain, the
ring is preferred." An amine whose nitrogen is bonded to a ring and to a chain is named on the ring,
the chain a prefix of the nitrogen ('N-[1-(phenylsulfanyl)butan-2-yl]aniline (PIN)'): the chain is a
substituent whatever it holds, so it is named whole by the substituent namer, rings it carries
included (their own seniority is no competition for the parent: the amine is not on them).
The producers of the ring parents first read a branch that is a plain alkyl; this is the other case.

The group is perceived with ONE bearing carbon of its nitrogen (``seniority._normalize_pcg_match``: the
most heavy neighbours, then the lowest atom index), chosen before any parent is. ``reanchor_bridging_amine``
moves that carbon into the parent the selection chose, so the producers of that parent find the group on it
whatever the order of the input ('CC(C)NC1CCCCC1' and 'C1CCCCC1NC(C)C' are one molecule).
"""
from typing import List, Optional, Set, Tuple

from rdkit import Chem


#: the groups whose heteroatom can be bonded to the carbons of several parents (an amine of one nitrogen)
_AMINE_GROUPS = frozenset({"secondary_amine", "tertiary_amine"})


#: the groups, key atom first, that a branch must not hold if its nitrogen is to be named as the
#: suffix of an amine parent: every class of Table 4.1 that is senior to the amine or the amine itself
#:, the Blue Book): carbonyl and thiocarbonyl groups (acids, esters, amides, aldehydes,
#: ketones), nitriles, hydroxy and sulfanyl groups (alcohols, phenols, acid hydroxy), peroxides, and
#: every acyclic nitrogen but a nitro or nitroso group
_SENIOR_OR_AMINE = tuple(Chem.MolFromSmarts(s) for s in (
    "[#6X3]=[O,S]", "[CX2]#[NX1]", "[OX2H1]", "[SX2H1]", "[OX2][OX2]",
    "[#7;!R;!$([N+](=O)[O-]);!$(N=O)]"))


def holds_amine_or_senior_group(mol, atoms) -> bool:
    """True when ``atoms`` (the branch on a nitrogen) holds an amine or a group of a class senior to it.

     (the Blue Book): "The senior parent structure has the maximum number of
    substituents corresponding to the principal characteristic group (suffix)... in accord with the
    seniority of classes ", (:26375): in a complex polyamine the parent is chosen by
    . A branch that holds one more amine ('N-(2-aminoethyl)aniline') makes the chain the parent
    ('N1-phenylethane-1,2-diamine'); one that holds a senior class ('2-anilinoethan-1-ol',
    '2-anilinoethane-1-peroxol') makes that class the principal group. The nitrogen is then no suffix of
    the ring parent, and no 'N-substituted aniline' is built from it."""
    held = set(atoms)
    return any(m[0] in held for pat in _SENIOR_OR_AMINE for m in mol.GetSubstructMatches(pat))


def amine_promotion(mol, branch_atoms, branch_name) -> dict:
    """The payload with which a nitrogen on a ring promotes to the amine suffix of that ring ('N-<branch>aniline'):
    empty when the branch holds an amine or a group senior to it (:func:`holds_amine_or_senior_group`)."""
    if holds_amine_or_senior_group(mol, branch_atoms):
        return {}
    return {'amine_candidate': {'suffix_name': 'amine', 'n_substituents': [branch_name]}}


def reanchor_bridging_amine(mol, principal_group, principal_group_atoms, parent_atoms):
    """``principal_group_atoms`` with the bearing carbon of a secondary or tertiary amine moved into
    ``parent_atoms``, when the nitrogen is bonded to a carbon of the parent, the Blue Book,
    the nitrogen belongs to every parent it is bonded to). A match whose carbon is in the parent already,
    and any other group, stay as they are."""
    if principal_group not in _AMINE_GROUPS or not principal_group_atoms:
        return principal_group_atoms
    parent = set(parent_atoms or ())
    out = []
    for match in principal_group_atoms:
        if len(match) >= 2 and match[1] not in parent and mol.GetAtomWithIdx(match[0]).GetSymbol() == 'N':
            held = sorted(nb.GetIdx() for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors()
                          if nb.GetSymbol() == 'C' and nb.GetIdx() in parent)
            if held:
                match = (match[0], held[0]) + tuple(match[2:])
        out.append(match)
    return out


def is_saturated_chain_carbon(atom) -> bool:
    """True for a carbon outside every ring whose bonds are all single: the carbon of an alkyl
    chain, not the carbon of an acyl, imidoyl or aryl group (an amide nitrogen is another class)."""
    return (atom.GetSymbol() == 'C' and not atom.IsInRing()
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()))


def acyclic_branch_on_nitrogen(mol, carbon, blocked: Set[int]) -> Tuple[Optional[List[int]], int]:
    """``(atoms, 0)`` of the branch that starts at ``carbon``, a saturated chain carbon bonded to the
    nitrogen, without the ``blocked`` atoms (the parent ring and the nitrogen); ``(None, 0)`` when
    ``carbon`` is no saturated chain carbon, the branch is bonded to a blocked atom other than the
    nitrogen's own bond, or it holds an amine or a group senior to it (``holds_amine_or_senior_group``).
    The zero is the count of carbons of a plain alkyl, which this branch is not: a caller that names it
    by the carbon count then finds no name and declines."""
    if not is_saturated_chain_carbon(carbon):
        return None, 0
    seen: Set[int] = set()
    todo = [carbon.GetIdx()]
    while todo:
        idx = todo.pop()
        if idx in seen:
            continue
        seen.add(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in blocked:
                if idx != carbon.GetIdx() or mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol() != 'N':
                    return None, 0
                continue
            todo.append(nb.GetIdx())
    if holds_amine_or_senior_group(mol, seen):
        return None, 0
    return sorted(seen), 0
