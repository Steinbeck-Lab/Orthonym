"""Spelling check /: the cyanide of an oxoacid written as a nitrile (registered with
:func:`orthonym.validation.pin_spelling.register`).

 "Seniority order for classes", class 10 (the Blue Book): "Acid halides and pseudohalides [first in
the order of the corresponding acid... then in the following order of pseudohalogen groups (-N3 > -CN >..."
comes before class 14 "Nitriles" (:18187). The cyanide of an acid R-E(=O)x-OH is therefore named by functional
class, the name of the acyl group and the class word 'cyanide' "Acyl pseudohalides from suffix
acids",:31460; "Cyanides corresponding to the sulfur, selenium, and tellurium acids",:34807,
'methanesulfonyl cyanide (PIN)':34811, 'benzeneseleninyl cyanide (PIN)':34813; "Acid halides and
pseudohalides",:35706, for the acids of the other central atoms). A name that writes the same C#N as the
nitrile of a substituted methane ('(benzenesulfinyl)methanenitrile', 'nitrosomethanenitrile') is a valid name
and not the PIN.

The structure side is a nitrile carbon whose other neighbour is a heteroatom E (S, Se, Te, N, P, As, Sb) with a
terminal double-bonded oxygen, the central atom of an oxoacid group; the name side is the word 'nitrile'. The
check speaks only when EVERY C#N of the molecule is such a cyanide, so a name whose 'nitrile' belongs to another
group is never lowered by it. The cyanides of the boron acids and silicic acid are named on the parent hydrides
(:35706: 'boranecarbonitrile', 'silanecarbonitrile'); an E without an oxo group is no acid centre here. No naming
code, no OPSIN.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_CENTRAL_ATOMS = ('S', 'Se', 'Te', 'N', 'P', 'As', 'Sb')


def _nitrile_partner(atom):
    """The heavy neighbour of a nitrile carbon other than its nitrogen, or None when ``atom`` is no nitrile
    carbon of two neighbours."""
    if atom.GetSymbol() != 'C' or atom.GetDegree() != 2:
        return None
    partner = None
    nitrile = False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondType() == Chem.BondType.TRIPLE and other.GetSymbol() == 'N':
            nitrile = True
        else:
            partner = other
    return partner if nitrile else None


def _is_oxoacid_centre(atom) -> bool:
    """A heteroatom E bearing a terminal double-bonded oxygen: R-E(=O)-, -N=O, -NO2. Not a ring atom: the
    -carbonitrile of a ring is the PIN, the Blue Book, "attached to a ring or ring
    system"; '1-oxo-1lambda5-phospholane-1-carbonitrile'), and a cyclic ester of an acid is named as a
    heterocycle, (:32177: "Other cyclic esters... are named as heterocycles")."""
    if atom is None or atom.GetSymbol() not in _CENTRAL_ATOMS or atom.IsInRing():
        return False
    return any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() == 'O'
               and b.GetOtherAtom(atom).GetDegree() == 1 for b in atom.GetBonds())


@register('P-66.5.1.3.2')
def acid_cyanide_as_nitrile_check(mol, name):
    """The name writes a 'nitrile' where every C#N of the molecule is the cyanide of an oxoacid."""
    if 'nitrile' not in name:
        return None
    partners = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        if any(b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(atom).GetSymbol() == 'N'
               for b in atom.GetBonds()):
            partner = _nitrile_partner(atom)
            if not _is_oxoacid_centre(partner):
                return None
            partners.append(partner)
    if not partners:
        return None
    element = partners[0].GetSymbol()
    rule = 'P-66.5.1.3.2' if element in ('S', 'Se', 'Te') else 'P-67.1.2.5.1'
    return SpellingFailure(
        rule, f"the C#N on the {element} of an oxoacid group is written as a nitrile; the cyanide of an acid "
              f"is senior to a nitrile (P-41 classes 10 and 14) and is named by functional class")


_METHANE_NITRILE_WORDS = ('methanenitrile', 'formonitrile')


@register('P-66.5.1.1.3')
def nitrile_on_heteroatom_check(mol, name):
    """ "Substitutive and functional class names for nitriles" (the Blue Book): "The suffix
    'carbonitrile' is always used to name nitriles having the -CN group attached to a ring or ring system or to
    an acyclic heteroatom" ('silanecarbonitrile (PIN)',:34722; 'boranecarbonitrile (PIN)',:37393). A name that
    writes the C#N of such a group as 'methanenitrile' with the heteroatom group as a prefix
    ('(dimethylboranyl)methanenitrile', '[dimethyl(phenyl)silyl]methanenitrile') is a valid name, not the PIN.

    Structure side: a nitrile carbon whose other neighbour is a heavy atom that is not carbon; name side: the
    word 'methanenitrile' (or 'formonitrile'). A cyano group of a prefix ('cyano') writes neither word."""
    if not any(word in name for word in _METHANE_NITRILE_WORDS):
        return None
    for atom in mol.GetAtoms():
        partner = _nitrile_partner(atom)
        if partner is not None and partner.GetAtomicNum() > 1 and partner.GetSymbol() != 'C':
            return SpellingFailure(
                'P-66.5.1.1.3',
                f"the C#N on the {partner.GetSymbol()} of an acyclic heteroatom is written as a methanenitrile "
                f"with a prefix; the suffix 'carbonitrile' is always used there")
    return None
