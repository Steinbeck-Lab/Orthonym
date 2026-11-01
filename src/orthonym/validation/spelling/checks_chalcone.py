"""Spelling check: a chalcone is named 'chalcone' (registered with
:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book, under KETONES, Retained names): "The name 'chalcone' is the only
retained name as a preferred IUPAC name and is limited to ring substitution only by characteristic groups lower than
'ketone'. Chalcone refers only to the trans- or (E)- stereoisomer.", 'chalcone (PIN) (2E)-1,3-diphenylprop-2-en-1-one'
(:28299), '2',4'-dihydroxy-3,3'-dimethoxychalcone (PIN) (2E)-1-(2,4-dihydroxy-3-methoxyphenyl)-3-(3-methoxyphenyl)
prop-2-en-1-one' (:28303); a group senior to the ketone keeps the systematic name: '4-[(1E)-3-(2,4-dihydroxyphenyl)-3-
oxoprop-1-en-1-yl]benzamide (PIN) (not 2',4'-dihydroxychalcone-4-carboxamide)' (:28305).

The structure decides whether the name is the chalcone's (E-1,3-diaryl-prop-2-en-1-one, two benzene rings joined to the
chain, each ring alone in its system, no group of the ketone's class or a senior one anywhere: nothing outside the chain
and the rings has a double or triple bond to a heteroatom, but the nitro group); the name side is whether it cites the
systematic 'prop-2-en-1-one' parent. No naming code, no OPSIN.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_CHALCONE_CORE = Chem.MolFromSmarts("[c;r6][CX3](=[OX1])[CX3H1]=[CX3H1][c;r6]")
_NITRO = Chem.MolFromSmarts("[N+](=O)[O-]")


def _benzene_ring_alone(mol, atom_idx: int):
    """The six atoms of the benzene ring that holds ``atom_idx`` when that ring is a system of its own, else None."""
    info = mol.GetRingInfo()
    rings = [r for r in info.AtomRings() if atom_idx in r]
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = rings[0]
    if any(info.NumAtomRings(a) != 1 or mol.GetAtomWithIdx(a).GetSymbol() != 'C'
           or not mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
        return None
    return ring


def chalcone_skeleton(mol):
    """``(ring_a_c, c1, o1, c2, c3, ring_b_c, ring_a, ring_b)`` of an (E)-1,3-diarylprop-2-en-1-one of two benzene
    rings, each alone in its ring system, with no group of the ketone's class or a senior one outside the chain
    and the rings (nothing outside them has a double or triple bond to a heteroatom, but the nitro group); else
    None. ``ring_a`` holds the ring of the benzoyl half, ``ring_b`` that of the styryl half. The producer of the
    substituted name (``rules.chalcones``) reads the same structure."""
    cores = mol.GetSubstructMatches(_CHALCONE_CORE)
    if len(cores) != 1:
        return None
    ring_a_c, c1, o1, c2, c3, ring_b_c = cores[0]
    bond = mol.GetBondBetweenAtoms(c2, c3)
    if bond.GetStereo() not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
        return None
    ring_a, ring_b = _benzene_ring_alone(mol, ring_a_c), _benzene_ring_alone(mol, ring_b_c)
    if ring_a is None or ring_b is None or set(ring_a) & set(ring_b):
        return None
    core = {c1, o1, c2, c3} | set(ring_a) | set(ring_b)
    nitro = {i for m in mol.GetSubstructMatches(_NITRO) for i in m}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if {i, j} <= core or {i, j} <= nitro or b.GetBondTypeAsDouble() == 1.0 or b.GetIsAromatic():
            continue
        a, z = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        if a.GetSymbol() != 'C' or z.GetSymbol() != 'C':
            return None      # a double or triple bond to a heteroatom: a group of the ketone's class or a senior one
    return (ring_a_c, c1, o1, c2, c3, ring_b_c, tuple(ring_a), tuple(ring_b))


def _is_chalcone(mol) -> bool:
    return chalcone_skeleton(mol) is not None


@register('P-64.2.1.1')
def chalcone_name_check(mol, name):
    """An (E)-1,3-diarylprop-2-en-1-one of two benzene rings (ring substituents of classes lower than the ketone) named
    on the systematic parent 'prop-2-en-1-one' instead of 'chalcone'."""
    if 'prop-2-en-1-one' not in (name or '') or 'chalcone' in (name or ''):
        return None
    if _is_chalcone(mol):
        return SpellingFailure('P-64.2.1.1', "the (E)-1,3-diarylprop-2-en-1-one of two benzene rings is named "
                                             "'chalcone', with its ring substituents as prefixes")
    return None
