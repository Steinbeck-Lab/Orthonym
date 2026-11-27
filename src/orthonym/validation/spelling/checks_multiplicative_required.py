"""Spelling check: two identical parent structures joined by one symmetric link atom, named substitutively
(registered with:func:`orthonym.validation.pin_spelling.register`).

 "Preferred IUPAC multiplicative names" (the Blue Book): "Multiplicative nomenclature is preferred to
substitutive nomenclature for generating preferred IUPAC names to express multiple occurrences of identical parent
structures, other than alkanes when (1)... (3)"; '2,2'-sulfanediyldiacetic acid (PIN, multiplicative name)' (:23192),
'1,1'-sulfinyldibenzene (PIN)' (:28110), '1,1'-methylenedibenzene (PIN)' (:19360). So '2-(thiophene-2-sulfonyl)thiophene'
(PIN '2,2'-sulfonyldithiophene') or '3-(2,3-dihydroxypropoxy)propane-1,2-diol' (PIN '3,3'-oxydi(propane-1,2-diol)') is
correct but no PIN.

Read only for a narrow shape: the link is one acyclic atom -- O, S, S(=O), S(=O)2, Se or Te, or a CH2 between two ring
atoms -- whose two halves are identical (isomeric SMILES, stereo included) and each half is a ring-bearing group or a chain
that holds the principal characteristic group,:6411, "Acyclic hydrocarbons... are not allowed" as units:
'ethoxyethane'). Not read: an N, C=O, C=S or other centre (a senior class or parent hydride in the link: 'N-phenylaniline',
:26233; 'bis(phenyldiazenyl)methanone'), a chain half with another chain hetero atom (replacement names), an ion, a
radical, a second fragment, a name with a space, or a name that already has primed locants or 'di'/'bis' units.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_PRINCIPAL = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S])[OX2H1,SX2H1]', '[CX3](=[O,S])[#7]', '[CX2]#[NX1]', '[CX3H1](=O)', '[#6][CX3](=O)[#6]',
    '[OX2H1][#6]', '[SX2H1][#6]', '[NX3;H2,H1;!$(N[#6]=[O,S,N])][#6]', '[SX4](=O)(=O)[OX2H1,#7]',
    '[#6][NX2]=[CX2]=[OX1]', '[#6][NX2]=[CX2]=[SX1]', '[CX3](=O)[NX3][NX3]'))
#: a name already multiplicative (primed locants, 'di(' / 'bis[' units, a 'diyl' linker) or a phane name
_NOT_JUDGED_NAME = re.compile(r"'|(?:di|bis|tri|tris|tetrakis)[(\[]|diyl|phane|(?:di|bis)(?:benzene|pyridine|thiophene|furan)\b"
                              r"|(?:oxy|sulfanyl|sulfinyl|sulfonyl|methylene|selanyl|tellanyl|selenonyl|seleninyl)(?:di|bis)")


def _link_atoms(mol):
    """Acyclic atoms that may be a multiplicative link: O, S (any oxidation), Se, Te with two heavy neighbours, or a CH2
    between two ring atoms."""
    for a in mol.GetAtoms():
        if a.IsInRing() or a.GetFormalCharge():
            continue
        heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if any(n.GetAtomicNum() not in (6, 8) for n in heavy):
            continue                    # a link between Si, Ge, B,... atoms belongs to a parent hydride ('disiloxane')
        if a.GetAtomicNum() in (8, 34, 52) and a.GetDegree() == 2 and all(n.GetAtomicNum() == 6 for n in heavy):
            yield a, heavy
        elif a.GetAtomicNum() == 16:
            oxo = [n for n in heavy if n.GetAtomicNum() == 8 and n.GetDegree() == 1]
            rest = [n for n in heavy if n not in oxo]
            if len(rest) == 2 and all(n.GetAtomicNum() == 6 for n in rest) and len(oxo) <= 2 and all(mol.GetBondBetweenAtoms(a.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2
                                                       for o in oxo):
                yield a, rest
        elif a.GetAtomicNum() == 6 and a.GetDegree() == 2 and a.GetTotalNumHs() == 2:
            if all(n.IsInRing() for n in heavy):
                yield a, heavy


def _halves(mol, link, ends):
    """The two halves cut at the link, as (isomeric SMILES with a dummy at the cut, atom set), or None."""
    bonds = [mol.GetBondBetweenAtoms(link.GetIdx(), e.GetIdx()).GetIdx() for e in ends]
    cut = Chem.FragmentOnBonds(mol, bonds, dummyLabels=[(1, 1)] * len(bonds))
    frags = Chem.GetMolFrags(cut, asMols=False, sanitizeFrags=False)
    out = []
    for e in ends:
        atoms = next(f for f in frags if e.GetIdx() in f)
        if link.GetIdx() in atoms:
            return None
        out.append((Chem.MolFragmentToSmiles(cut, list(atoms), isomericSmiles=True, canonical=True),
                    {i for i in atoms if i < mol.GetNumAtoms()}))
    return out


@register('P-51.3.1')
def multiplicative_required_check(mol, name):
    """Two identical parent structures on one symmetric link atom are named substitutively."""
    if not name or ' ' in name.strip() or _NOT_JUDGED_NAME.search(name):
        return None
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    principal = {m[0] for p in _PRINCIPAL for m in mol.GetSubstructMatches(p)}
    in_principal = {i for p in _PRINCIPAL for m in mol.GetSubstructMatches(p) for i in m}
    # chain hetero atoms (any acyclic N, O, S, P,... joining two heavy atoms, an NH of a chain included, outside the
    # acid, amide and other suffix groups): replacement ('a') names may be the PIN for such chains
    hetero = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6) and not a.IsInRing()
              and len([n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]) > 1
              and (a.GetIdx() not in in_principal or a.GetAtomicNum() == 7 and not any(
                  n.GetAtomicNum() == 6 and any(b.GetBondTypeAsDouble() == 2 for b in n.GetBonds()) for n in a.GetNeighbors()))}
    for link, ends in _link_atoms(mol):
        halves = _halves(mol, link, ends)
        if not halves or halves[0][0] != halves[1][0]:
            continue
        atoms = halves[0][1]
        if link.GetIdx() in principal or (principal and not principal <= (halves[0][1] | halves[1][1])):
            continue
        has_ring = any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms)
        if not has_ring and not (principal & atoms):
            continue                    # an alkane unit
        if not has_ring and (hetero & (halves[0][1] | halves[1][1])):
            continue                    # a chain with another hetero atom: a replacement name may be the parent
        return SpellingFailure('P-51.3.1', "two identical parent structures on a symmetric link take a multiplicative "
                                           "name ('2,2'-sulfonyldithiophene', '3,3'-oxydi(propane-1,2-diol)'); a "
                                           "substitutive name is no PIN")
    return None
