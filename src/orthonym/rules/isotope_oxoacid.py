"""A nuclide of the central atom of an oxoacid suffix stands before the suffix.

 (heading Names, ``the Blue Book Blue Book``): in a name of two or more words "the isotopic
descriptor is placed before the appropriate word or part of the word that includes the nuclide(s)". The printed names of
a nuclide in a suffix put it before the suffix, not before the whole name: 'benzene(13C)carboxylic acid (PIN)' and
'benzene(13C)carbonitrile (PIN)' (:44254-:44255, under in " Location of nuclides on positions not
normally denoted by locants",:44221), 'cyclopentan-1-(18O)ol (PIN)' (:43744). The S of 'sulfonic acid' is such a nuclide:
'methane(34S)sulfonic acid', not '(34S)methanesulfonic acid'. The suffixes whose central atom is the labelled atom are
'sulfonic acid' and 'sulfinic acid', the seleninic, selenonic, tellurinic and telluronic acids, and every derivative of each
that keeps the stem ('sulfonamide', 'sulfonate', 'sulfonohydrazide', 'sulfonimidic acid'). OPSIN 2.9.0 reads the descriptor at
that slot ('methane(34S)sulfonic acid', 'methane(34S)sulfonamide', 'methane(34S)sulfonohydrazide', 'methane(34S)sulfonimidic
acid', 'methyl methane(34S)sulfonate'); it does not read it before the acyl ending 'sulfonyl' or 'sulfinyl', so a name with a
nuclide of that atom anywhere else than before the stem is built with the descriptor at the front and is below the PIN, as is
a multiplied suffix, which cites the descriptor inside the multiplied part ('cyclohexane-1,1-di[(14C)carboxylic acid] (PIN)',
,:43774). The atom is an S, Se or Te outside a ring that is double-bonded to an O or an N: a ring atom of that
kind is a heteroatom of the ring name ('1,1-dioxo(34S)thiolane'), not the centre of a suffix.

``rules.isotopes._find_best_placement`` ranks the slot first (``rules.isotopes._placement_quality``) and withdraws the PIN
label from a name that puts the descriptor elsewhere (:func:`off_slot_below_pin`).
"""
import re

from rdkit import Chem

_CENTRE_STEM = r"(?:sulfon|sulfin|selenon|selenin|telluron|tellurin)"

#: the word of a name that holds the central atom of an oxoacid group: the stem of the acid with whatever continues it
#: ('sulfonic acid', 'sulfonamide', 'sulfonohydrazide', 'sulfonimidic acid', 'sulfonate', 'sulfonyl'). The slot is the
#: place before the stem; the OPSIN round trip of the caller says which of the places it reads.
_SUFFIX_RE = re.compile(_CENTRE_STEM + r"(?=[a-z])")

_WORD_RE = _SUFFIX_RE

#: the elements that are the central atom of an oxoacid suffix
ELEMENTS = frozenset({"S", "Se", "Te"})

#: a multiplying prefix that ends the text before a slot: a multiplied suffix has no single slot for one nuclide
_MULTIPLIER_AT_END_RE = re.compile(r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)$")


def labels_an_element_of_the_centre(keys) -> bool:
    """True when one of the nuclide keys ``((mass, element), count, max_per_position)`` is an S, Se or Te."""
    return any(el in ELEMENTS for (_mass, el), _count, _maxpos in keys)


def is_centre_slot(skel: str, off: int) -> bool:
    """True when ``off`` is the slot directly before an oxoacid suffix that holds its central atom
    ('methane|sulfonic acid', 'benzene-1-|sulfonamide'). A multiplied suffix has no such slot."""
    if off <= 0 or not (skel[off - 1] == "-" or skel[off - 1].isalpha()):
        return False
    return _SUFFIX_RE.match(skel, off) is not None and _MULTIPLIER_AT_END_RE.search(skel[:off]) is None


def labels_a_centre(original: Chem.Mol) -> bool:
    """Does ``original`` carry a nuclide on the central atom of an oxoacid group: an S, Se or Te outside a ring,
    double-bonded to an O or an N (the atom of a 'sulfonic acid', a 'sulfinamide', a 'sulfonyl')?"""
    for atom in original.GetAtoms():
        if atom.GetIsotope() and atom.GetSymbol() in ELEMENTS and not atom.IsInRing():
            for bond in atom.GetBonds():
                if (bond.GetBondType() == Chem.BondType.DOUBLE
                        and bond.GetOtherAtom(atom).GetSymbol() in ("O", "N")):
                    return True
    return False


def off_slot_below_pin(skel: str, off: int, desc, keys, original: Chem.Mol) -> bool:
    """True when the descriptor ``desc`` of a nuclide of an oxoacid centre is cited at ``off`` of ``skel`` and ``off`` is
    not the slot before the suffix that holds the atom ('(34S)methanesulfonyl chloride': the PIN 'methane(34S)sulfonyl
    chloride' is not read by OPSIN 2.9.0, so it is not built). The name is then below the PIN."""
    return bool(desc and labels_an_element_of_the_centre(keys) and _WORD_RE.search(skel)
                and not is_centre_slot(skel, off) and labels_a_centre(original))
