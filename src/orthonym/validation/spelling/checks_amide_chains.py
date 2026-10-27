"""Spelling check: the parent chain of two amides on different chains (registered with
:func:`orthonym.validation.pin_spelling.register`).

A molecule with two acyclic carboxamide groups of different kinds on two chains (a primary and an N-substituted
one) is named by the producers on the chain of the amide that comes first in the order of the kinds (the primary
amide), whichever chain makes the senior parent. (the Blue Book), "The preferred IUPAC name is
based on the senior parent structure that has the maximum number of substituents cited as prefixes", and
(:21698), "... that has the lower locant or set of locants for substituents cited as prefixes": the prefixes of a
chain with an amide include the substituents of the amide nitrogen, at the locant 'N', which is lower than any
number: '3-[5-(3-amino-2-methyl-3-oxopropyl)-2-methylphenyl]-N-methylpropanamide (PIN) [not 2-methyl-3-{4-methyl-3-[3-
(methylamino)-3-oxopropyl]phenyl}propanamide; the locant set 'N,3' in the PIN is lower than '2,3']' (:21730). Before
the prefixes, (:20916) "(b) has the greater number of skeletal atoms" decides by the length of the chain.

The structure decides, from the two chains alone: each chain is the longest path of carbon atoms from the carbonyl
carbon outside rings, its prefixes are the other neighbours of its atoms (the oxygen and the nitrogen of the amide are
its suffix) and the substituents of the amide nitrogen; the chain with the greater length, then the greater number of
multiple bonds,:21033, "The senior ring, ring system or principal chain has the greater number of
multiple bonds";,:21016: it decides when to do not, so before, then the greater
number of prefixes, then the lower set of locants ('N' first) is the parent. The name is a failure when the other
chain wins (the check abstains on every shape it does not model). No naming code, no OPSIN.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_AMIDE = Chem.MolFromSmarts("[CX3;!R](=[OX1])[NX3;!R;!$(N-[!#6;!#1])]")
#: the order of the kinds in which the producers take the principal group (primary before secondary before tertiary)
_KIND_ORDER = {0: 0, 1: 1, 2: 2}


def _longest_paths(mol, start: int):
    """Every longest path of carbon atoms outside rings that starts at ``start``, as atom lists."""
    best, found = 0, []

    def walk(path):
        nonlocal best, found
        extended = False
        for nb in mol.GetAtomWithIdx(path[-1]).GetNeighbors():
            j = nb.GetIdx()
            if j not in path and nb.GetSymbol() == 'C' and not nb.IsInRing():
                extended = True
                walk(path + [j])
        if not extended:
            if len(path) > best:
                best, found = len(path), [path]
            elif len(path) == best:
                found.append(path)

    walk([start])
    return found


def _n_substituents_reach(mol, own, other) -> bool:
    """True when a substituent of the nitrogen of the amide ``own`` leads to the carbonyl carbon of ``other``."""
    carbonyl, nitrogen = own[1], own[2]
    seen, stack = {carbonyl, nitrogen}, [nb.GetIdx() for nb in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                                          if nb.GetIdx() != carbonyl]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        if i == other[1]:
            return True
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in seen)
    return False


def _chain_key(mol, carbonyl: int, nitrogen: int, oxygen: int):
    """``(-length, -multiple bonds, -prefixes, locants)`` of the best chain from ``carbonyl`` (smaller is the senior
    parent), or None."""
    keys = []
    n_subs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
              if nb.GetIdx() != carbonyl and nb.GetAtomicNum() > 1]
    for path in _longest_paths(mol, carbonyl):
        on_path = set(path)
        locants = [0] * len(n_subs)                 # 'N' is lower than any numeral
        for pos, a in enumerate(path, start=1):
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() in on_path or nb.GetAtomicNum() <= 1 or nb.GetIdx() in (nitrogen, oxygen):
                    continue
                locants.append(pos)
        multiple = sum(1 for i, j in zip(path, path[1:]) if mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() >= 2)
        keys.append((-len(path), -multiple, -len(locants), tuple(sorted(locants))))
    return min(keys) if keys else None


@register('P-45.2')
def amide_chain_parent_check(mol, name):
    """Two acyclic carboxamide groups of different kinds on two chains, the chain of the first kind not the senior
    parent by (length), (number of prefixes) and (their locants, 'N' lowest)."""
    if not (name or '').endswith('amide'):
        return None                                  # the amide is not the parent: an ester, an acid... is senior
    amides = []
    for m in mol.GetSubstructMatches(_AMIDE):
        carbonyl, oxygen, nitrogen = m
        c = mol.GetAtomWithIdx(carbonyl)
        if (any(nb.IsInRing() for nb in c.GetNeighbors()) or c.GetTotalNumHs() != 0
                or not any(nb.GetSymbol() == 'C' for nb in c.GetNeighbors())):
            return None                              # a ring-attached amide, a formamide: not modelled
        kind = 2 - mol.GetAtomWithIdx(nitrogen).GetTotalNumHs() if mol.GetAtomWithIdx(nitrogen).GetTotalNumHs() <= 2 else 0
        amides.append((kind, carbonyl, nitrogen, oxygen))
    if len(amides) != 2 or amides[0][0] == amides[1][0]:
        return None
    # the two chains must be apart: no carbon path joins the two carbonyl carbons outside rings, and neither amide
    # nitrogen carries the other group (a peptide-like chain: the amide nitrogen is a substituent of the other chain)
    first, second = sorted(amides)
    if any(second[1] in path for path in _longest_paths(mol, first[1])):
        return None
    if any(_n_substituents_reach(mol, own, other) for own, other in ((first, second), (second, first))):
        return None
    k_first, k_second = _chain_key(mol, first[1], first[2], first[3]), _chain_key(mol, second[1], second[2], second[3])
    if k_first is None or k_second is None or k_second >= k_first:
        return None
    return SpellingFailure('P-45.2', "the parent chain carries the amide of the first kind, but the chain of the "
                                     "other amide has the lower locants for the prefixes ('N,3' before '2,3')")
