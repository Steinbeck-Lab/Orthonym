"""Spelling check /: a phosphonic or arsonic acid on a hydrocarbon skeleton is the parent
of the name, and two of them are multiplied (registered with:func:`orthonym.validation.pin_spelling.register`).

 "Names of mononuclear noncarbon oxoacids" (the Blue Book) names an organyl-substituted acid on the
acid ('phenylphosphonic acid (PIN)'; ':35472 (naphthalene-2,6-diyl)bis(phosphonous acid) (PIN)');:35702 extends
the phosphonic names to arsenic. The acid is a functional parent with substitutable hydrogen atoms, an identical
parent structure of multiplicative nomenclature (d),:5166: "functional parent compounds having
substitutable hydrogen atoms, for example, acetic acid or phosphonic acid"). "Preferred IUPAC
multiplicative names" (:23178): "Multiplicative nomenclature is preferred to substitutive nomenclature for
generating preferred IUPAC names to express multiple occurrences of identical parent structures" when the linking
bonds are identical and the parents identically substituted; (:6295) allows an unsymmetrical central
group ('bicyclo[2.2.1]heptane-2,7-diyl'). So for two such acids the PIN is '(<linker>)bis(phosphonic acid)'; a
name that writes either acid as the prefix 'phosphono' ('2,7-diphosphonobicyclo[2.2.1]heptane',
'(6-phosphonodecahydronaphthalen-2-yl)phosphonic acid') is a valid name and not the PIN.

Scope, where the acid is certainly the principal characteristic group and the parents are identical: every heavy
atom other than the acid groups is a neutral carbon, every P (As) atom is the centre of a neutral -P(=O)(OH)2
(-As(=O)(OH)2) on a carbon, and only one of the two elements occurs. One acid: the name must carry the acid word.
Two acids on one multiplying group (``_one_linking_group``): the name must carry 'bis(<acid>)'. Three or more acids: only the acid word is checked decides
between multiplied and substituted forms there). Outside the scope the rule is not about the molecule and the check
returns None (not ABSTAIN, which would keep every other name from a PIN certificate). No naming code, no OPSIN.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import PASS, SpellingFailure, register

#: element -> (acid word, SMARTS of the neutral acid group on a carbon)
_ACIDS = {
    'P': ('phosphonic acid', Chem.MolFromSmarts('[#6][PX4;+0](=[OX1])([OX2H1])[OX2H1]')),
    'As': ('arsonic acid', Chem.MolFromSmarts('[#6][AsX4;+0](=[OX1])([OX2H1])[OX2H1]')),
}


def _acid_in_scope(mol):
    """(element, number of acid groups) when ``mol`` is a hydrocarbon skeleton bearing only acid groups of one
    element; None otherwise."""
    present = [e for e in _ACIDS if any(a.GetSymbol() == e for a in mol.GetAtoms())]
    if len(present) != 1:
        return None
    elem = present[0]
    matches = mol.GetSubstructMatches(_ACIDS[elem][1])
    if not matches or sum(1 for a in mol.GetAtoms() if a.GetSymbol() == elem) != len(matches):
        return None
    acid_atoms = {i for m in matches for i in m[1:]}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 or atom.GetIdx() in acid_atoms:
            continue
        if atom.GetSymbol() != 'C' or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
    return elem, matches


def _connected_through(mol, a, b, keep) -> bool:
    """Whether atoms ``a`` and ``b`` are joined by a path of atoms that all satisfy ``keep``."""
    seen, todo = {a}, [a]
    while todo:
        at = mol.GetAtomWithIdx(todo.pop())
        for nb in at.GetNeighbors():
            i = nb.GetIdx()
            if i == b:
                return True
            if i not in seen and keep(nb):
                seen.add(i)
                todo.append(i)
    return False


def _one_linking_group(mol, matches) -> bool:
    """Whether the two acids hang on one multiplying group: both acid carbons in one ring system or ring assembly
    (a path of ring atoms: 'bicyclo[2.2.1]heptane-2,7-diyl', '[1,1'-biphenyl]-4,4'-diyl'), both on one acyclic chain
    (a path of chain atoms: 'ethane-1,2-diyl'), or the two acids symmetry-equivalent (a symmetric concatenated
    group). A ring and a chain together are no single group, and an unsymmetric concatenation breaks
    "Concatenated multiplicative groups" (the Blue Book: "the sequence of atoms and bonds, starting at the central
    substituent group, are identical in each of the branches"): '[(4-phosphonophenyl)methyl]phosphonic acid' stays
    substitutive."""
    (c1, p1), (c2, p2) = [(m[0], m[1]) for m in matches]
    a1, a2 = mol.GetAtomWithIdx(c1), mol.GetAtomWithIdx(c2)
    if c1 == c2:
        return True
    if a1.IsInRing() and a2.IsInRing() and _connected_through(mol, c1, c2, lambda at: at.IsInRing()):
        return True
    if not a1.IsInRing() and not a2.IsInRing() and _connected_through(
            mol, c1, c2, lambda at: not at.IsInRing() and at.GetSymbol() == 'C'):
        return True
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    return ranks[p1] == ranks[p2]


@register("P-51.3.1")
def phosphonic_acid_parent(mol, name):
    """FAIL when the acid of a molecule in scope is not the parent of ``name``, or two acids are not multiplied."""
    if mol is None or not name:
        return None
    scope = _acid_in_scope(mol)
    if scope is None:
        return None
    elem, matches = scope
    word = _ACIDS[elem][0]
    if word not in name:
        return SpellingFailure("P-67.1.1.1", f"the {word} is written only as a prefix")
    if len(matches) == 2 and f"bis({word})" not in name and _one_linking_group(mol, matches):
        return SpellingFailure("P-51.3.1", f"two {word} parents are not multiplied ('bis({word})')")
    return PASS
