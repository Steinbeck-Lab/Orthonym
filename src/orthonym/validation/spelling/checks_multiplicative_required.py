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
    '[CX3](=[O,S])[OX2H1,SX2H1]', '[CX3](=[O,S])[#7]', '[CX2]#[NX1]', '[CX3H1](=O)', '[CX3](=O)([#6])[#6]',
    '[OX2H1][#6]', '[SX2H1][#6]', '[NX3;H2,H1;!$(N[#6]=[O,S,N])][#6]', '[SX4](=O)(=O)[OX2H1,#7]',
    '[NX2](=[CX2]=[OX1])[#6]', '[NX2](=[CX2]=[SX1])[#6]', '[CX3](=O)[NX3][NX3]'))
#: a name already multiplicative (primed locants, 'di(' / 'bis[' units, a 'diyl' linker) or a phane name
_NOT_JUDGED_NAME = re.compile(r"'|(?:di|bis|tri|tris|tetrakis)[(\[]|diyl|phane|(?:di|bis)(?:benzene|pyridine|thiophene|furan)\b"
                              r"|(?:oxy|sulfanyl|sulfinyl|sulfonyl|methylene|selanyl|tellanyl|selenonyl|seleninyl)(?:di|bis)"
                              r"|[)\]](?:di|bis|tri|tris)[a-z]")        # '(1,3-phenylene)dimethanamine'


def _link_atoms(mol):
    """Acyclic atoms that may be a multiplicative link: O, S (any oxidation), Se, Te with two heavy neighbours, or a CH2
    between two ring atoms."""
    for a in mol.GetAtoms():
        if a.IsInRing() or a.GetFormalCharge():
            continue
        heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if any(n.GetAtomicNum() not in (6, 8) for n in heavy):
            continue                    # a link between Si, Ge, B,... atoms belongs to a parent hydride ('disiloxane')
        if a.GetAtomicNum() in (8, 34, 52) and len(heavy) == 2 and all(n.GetAtomicNum() == 6 for n in heavy):
            yield a, heavy
        elif a.GetAtomicNum() == 16:
            oxo = [n for n in heavy if n.GetAtomicNum() == 8 and n.GetDegree() == 1]
            rest = [n for n in heavy if n not in oxo]
            if len(rest) == 2 and all(n.GetAtomicNum() == 6 for n in rest) and len(oxo) <= 2 and all(mol.GetBondBetweenAtoms(a.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2
                                                       for o in oxo):
                yield a, rest
        elif a.GetAtomicNum() == 6 and len(heavy) == 2 and a.GetTotalNumHs() == 2:
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
    if not name or ' ' in re.sub(r' acid$', '', name.strip()) or _NOT_JUDGED_NAME.search(name):
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
        if principal & atoms and not _carried(mol, atoms, principal):
            continue                    # 'dicarbonic acid': the groups have no parent carbon in the half
        return _FAILURE
    if principal and _twins_on_a_centre(mol, principal, hetero):
        return _FAILURE
    return None


_FAILURE = SpellingFailure('P-51.3.1', "two identical parent structures on a symmetric link take a multiplicative name "
                                       "('2,2'-sulfonyldithiophene', 'N,N'-(1,4-phenylene)dibenzamide'); a substitutive "
                                       "name is no PIN")
#: elements a multiplicative centre may hold (a Si, Ge, B, P... centre is a parent hydride: 'disiloxane')
_CENTRE_ELEMENTS = (6, 7, 8, 9, 16, 17, 34, 35, 52, 53)
#: an atom that would carry a suffix: a carbon doubly bonded to O, S, Se or N (ring ketones, lactones and lactams
#: included), an OH, SH or acyclic NH / NH2
_SUFFIXABLE = Chem.MolFromSmarts('[$([#6]=[O,S,Se,N]),$([OX2H1]),$([SX2H1]),$([NX3;H1,H2;!R])]')
#: (pattern, index) of the carbon that a suffix or prefix group owns ('carboxylic acid', 'carbonitrile', 'isocyanato')
_GROUP_CARBON = tuple((Chem.MolFromSmarts(s), i) for s, i in (
    ('[CX3](=[O,S])[OX2H1,SX2H1]', 0), ('[CX3](=[O,S])[#7]', 0), ('[CX2]#[NX1]', 0), ('[CX3H1](=O)', 0),
    ('[NX2]=[CX2]=[O,S]', 1)))


def _group_carbon(atom) -> bool:
    """A carbon a suffix group owns: C=O, C=S or C=N with a hetero atom on it (acid, amide, imidic acid, ester), an
    aldehyde CH=O, a nitrile carbon, or the carbon of an isocyanate. A ketone C=O between two carbons is skeletal."""
    if atom.GetAtomicNum() != 6:
        return False
    multiple = [b for b in atom.GetBonds() if b.GetBondTypeAsDouble() >= 2
                and b.GetOtherAtom(atom).GetAtomicNum() in (7, 8, 16, 34)]
    if not multiple:
        return False
    if any(b.GetBondTypeAsDouble() == 3 for b in multiple) or atom.GetTotalNumHs() == 1:
        return True
    heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    return any(n.GetAtomicNum() != 6 and mol_bond_single(atom, n) for n in heavy)


def mol_bond_single(a, b) -> bool:
    bond = a.GetOwningMol().GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _carried(mol, half, principal) -> bool:
    """Every principal group of the half sits on a skeletal carbon of the half (its parent): a -COOH, -CONHR or -CN
    whose carbon hangs on the centre is the centre's suffix ('N1,N4-dimethylbenzene-1,4-dicarboxamide', 'dicarbonic
    acid'), not a parent structure of its own."""
    for p in principal & half:
        atom = mol.GetAtomWithIdx(p)
        owners = [atom] if _group_carbon(atom) else [n for n in atom.GetNeighbors() if _group_carbon(n)]
        if owners:
            carriers = {n.GetIdx() for o in owners for n in o.GetNeighbors()
                        if n.GetAtomicNum() == 6 and not _group_carbon(n)}
        else:
            carriers = {n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 6}
        if not carriers & half:
            return False
    return True


def _amide_n(atom) -> bool:
    """A nitrogen on a C=X carbon (an amide, urea or amidine nitrogen)."""
    return any(n.GetAtomicNum() == 6 and any(b.GetBondTypeAsDouble() == 2 for b in n.GetBonds())
               for n in atom.GetNeighbors())


def _sides(mol):
    """For every acyclic single bond, both sides: (isomeric SMILES with a dummy, atom set, side atom, other atom)."""
    out = []
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondTypeAsDouble() != 1:
            continue
        cut = Chem.FragmentOnBonds(mol, [b.GetIdx()], dummyLabels=[(1, 1)])
        frags = Chem.GetMolFrags(cut, asMols=False, sanitizeFrags=False)
        for side, other in ((b.GetEndAtom(), b.GetBeginAtom()), (b.GetBeginAtom(), b.GetEndAtom())):
            atoms = next(f for f in frags if side.GetIdx() in f)
            real = frozenset(i for i in atoms if i < mol.GetNumAtoms())
            out.append((Chem.MolFragmentToSmiles(cut, list(atoms), isomericSmiles=True, canonical=True), real, side,
                        other))
    return out


def _twins_on_a_centre(mol, principal, hetero) -> bool:
    """Two identical halves, each holding principal groups, on a centre of two or more atoms (a ring such as phenylene,
    or a multi-atom link) that holds none: (:23178); (:5186) "Any simple polyvalent substituent
    group... may be used as a multiplicative substituent group"; '3,3'-(1,4-phenylene)di(propan-1-ol) (PIN)' (:25171).
    A cut between two acyclic carbons would split one chain (the chain is the parent) and is not read."""
    # a half is a parent structure only with a carbon of its own: a lone -COOH, -CHO, -CN, -CONH2 or -NCO is the
    # suffix (or prefix) group of the centre ('benzene-1,4-dicarboxylic acid', 'pyridine-2,6-dicarbonitrile')
    group_c = {m[i] for p, i in _GROUP_CARBON for m in mol.GetSubstructMatches(p)}
    suffixable = {m[0] for m in mol.GetSubstructMatches(_SUFFIXABLE)}
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        rest = [x for x in systems if not (x & merged)]
        for x in systems:
            if x & merged:
                merged |= x
        systems = rest + [merged]
    sides = [x for x in _sides(mol) if x[1] & principal and len(x[1]) * 2 < mol.GetNumAtoms()
             and any(mol.GetAtomWithIdx(k).GetAtomicNum() == 6 and k not in group_c for k in x[1])]
    for i, (smi1, a1, h1, c1) in enumerate(sides):
        for smi2, a2, h2, c2 in sides[i + 1:]:
            if smi1 != smi2 or a1 & a2 or not principal <= (a1 | a2):
                continue
            centre = set(range(mol.GetNumAtoms())) - a1 - a2
            if len(centre) < 2 or c1.GetIdx() not in centre or c2.GetIdx() not in centre:
                continue
            if any(not c.IsInRing() and c.GetAtomicNum() == 6 and not h.IsInRing() and h.GetAtomicNum() == 6
                   for c, h in ((c1, h1), (c2, h2))):
                continue                # the cut would split one carbon chain
            if any(h.GetAtomicNum() not in (6, 7) or (h.GetAtomicNum() == 7 and not h.IsInRing() and not _amide_n(h))
                   for h in (h1, h2)):
                continue                # a half attached by an amine N, an O,... is not read
            if any(a.GetAtomicNum() not in _CENTRE_ELEMENTS or (a.GetAtomicNum() == 7 and not a.IsInRing())
                   or k in suffixable for k, a in ((k, mol.GetAtomWithIdx(k)) for k in centre)):
                continue                # the centre would carry a suffix of its own (a ring C=O, an OH,...)
            has_ring = any(mol.GetAtomWithIdx(k).IsInRing() for k in a1)
            if not has_ring and (hetero & (a1 | a2) or len(hetero) >= 4):
                continue                # replacement ('a') names may be the parent
            if not _a_multiplicative_group(mol, c1.GetIdx(), c2.GetIdx(), centre, rank, systems):
                continue
            if not _carried(mol, set(a1), principal):
                continue
            return True
    return False


def _a_multiplicative_group(mol, c1, c2, centre, rank, systems) -> bool:
    """ (1)-(2) and (:5238): the link is one simple group -- the path between the two
    attachment atoms runs inside one ring system or along acyclic carbons only (substituted or not, symmetrical or not,
    ,:5208) -- or a concatenated group whose arms are identical about a central group (the two attachment
    atoms are equivalent). '-CH2-C6H4-O-' is neither."""
    if c1 == c2:
        return True                     # one atom carries both halves: a simple group ('ethane-1,1-diyl')
    path = Chem.GetShortestPath(mol, c1, c2)
    if not set(path) <= centre:
        return False
    if any(set(path) <= ring_system for ring_system in systems):
        return True
    if all(not mol.GetAtomWithIdx(k).IsInRing() and mol.GetAtomWithIdx(k).GetAtomicNum() == 6 for k in path):
        return True
    return rank[c1] == rank[c2]
