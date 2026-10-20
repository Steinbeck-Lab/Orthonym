"""When an isotopic descriptor may cite a bare element letter.

An italic letter locant ('O', 'N') names a heteroatom that integer locants do not number: the
hydroxyl oxygen of a carboxylic acid, the nitrogen of an amine or an amide. The Blue Book prints it
bare where the parent holds one such atom, '(N-2H1)acetamide (PIN)' NAMES,,
the Blue Book) and '(O-2H, 18O)acetic acid (PIN)',:44294, "the '18O' is an isotopic
descriptor and the 'O' is a locant",:44296); its comparative table (:44506) prints a bare letter
beside numerals, '(O,2,2,2-2H4)ethan-1-ol', for a parent with one oxygen.

 LOCANTS, Omission of locants, (:44180): "In preferred IUPAC names, locants
are omitted if no locants are necessary in unmodified names. However, if isotopic modification
requires a locant to specify its position, then all locants must be specified and none are omitted."
 (:44202): "Locants are not omitted when there is a possibility of isomers".
 Letter and/or numeral locants (:44208): "When locants are needed for defining the structure of
the parent structure or of a unit of structure as defined by its appropriate enclosing marks, then all
locants must be cited for the parent structure or that structural unit."

So a bare letter is cited only where it leaves no possibility of isomers: every atom of that element
the letter could denote is the same position, one symmetry class of the structure. The class is judged
on the structure without the labels the letter itself denotes (the isotopes of those atoms and of their
hydrogens) and with every other label kept: the two hydroxyl oxygens of butanedioic acid are one class
('(O-2H)butanedioic acid'), and so are they when only they carry labels; a 13C in the chain tells them
apart, and then the letter is cited with the number of the atom it is borne by, '(2-13C,O1-18O)
pentanedioic acid' ('O5' is the other isomer). OPSIN 2.9.0 reads a bare letter as the first such atom,
so the read-back cannot decide this; the rule does. One predicate for the decorator (``rules.isotopes``)
and the scope composer (``rules.isotope_scopes``).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from rdkit import Chem


def letter_ranks(original: Chem.Mol, element: str) -> Dict[int, int]:
    """Symmetry class of every atom of ``original`` (index of ``original`` -> class) in the structure
    without hydrogen atoms, without the isotopes of the atoms ``element``'s letter can denote and of their
    hydrogens, and with every other isotope kept: a labelled hydroxyl oxygen, whose hydrogen is an
    explicit atom of ``original``, ranks as the unlabelled hydroxyl oxygen it is, and a 13C elsewhere
    still tells the two ends of a diacid apart."""
    denoted = set(denotable_atoms(original, element))
    work = Chem.Mol(original)
    for atom in work.GetAtoms():
        atom.SetIntProp("_orig", atom.GetIdx())
        on_denoted = atom.GetAtomicNum() == 1 and any(n.GetIdx() in denoted for n in atom.GetNeighbors())
        if atom.GetIdx() in denoted or on_denoted:
            atom.SetIsotope(0)
    try:
        work = Chem.RemoveHs(work)
        ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False))
    except Exception:  # noqa: BLE001 - a molecule RDKit cannot rank has no classes to compare
        return {}
    return {a.GetIntProp("_orig"): ranks[a.GetIdx()] for a in work.GetAtoms()}


def denotable_atoms(original: Chem.Mol, element: str) -> List[int]:
    """The atoms of ``original`` an italic ``element`` letter can denote: for 'O' the hydroxyl oxygens
    (single-bonded, one carbon neighbour; the carbonyl oxygen is the other oxygen of the acid), for 'N'
    the nitrogens that are not ring members, not aromatic and bear a hydrogen (an amine or an amide)."""
    out: List[int] = []
    for atom in original.GetAtoms():
        if atom.GetSymbol() != element:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if element == "O":
            if len(heavy) != 1 or heavy[0].GetSymbol() != "C":
                continue
            bond = original.GetBondBetweenAtoms(atom.GetIdx(), heavy[0].GetIdx())
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
        elif element == "N":
            if atom.IsInRing() or atom.GetIsAromatic() or atom.GetTotalNumHs(includeNeighbors=True) < 1:
                continue
        out.append(atom.GetIdx())
    return out


def distinct_classes(original: Chem.Mol, element: str, atoms: Optional[Iterable[int]] = None) -> int:
    """The number of symmetry classes among ``atoms`` (default: every atom ``element``'s letter can
    denote; ``letter_ranks``), the quantity: more than one means citing the letter alone leaves
    a possibility of isomers. ``atoms`` index ``original``."""
    ranks = letter_ranks(original, element)
    chosen = denotable_atoms(original, element) if atoms is None else list(atoms)
    return len({ranks[a] for a in chosen if a in ranks})


def bare_letter_allowed(original: Chem.Mol, element: str, atoms: Optional[Iterable[int]] = None) -> bool:
    """Whether ``element``'s italic letter may be cited without its number: the atoms it could denote (the
    default, the whole structure; ``atoms``, those of one unit) are one symmetry class."""
    return distinct_classes(original, element, atoms) <= 1
