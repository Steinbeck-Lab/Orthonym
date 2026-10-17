"""When the default tier may cite a hydrogen nuclide without its count and without a locant.

 (the Blue Book,:43720): the descriptor stands "before the part of the compound that is
isotopically substituted"; "When polysubstitution at a single position is possible, the number of atoms
substituted is always specified as a right subscript to the atomic symbol(s), even in case of
monosubstitution." (heading Omission of locants,:44178; the rule:44180): "locants are
omitted if no locants are necessary in unmodified names"; (:44202): "Locants are not omitted
when there is a possibility of isomers"; Letter and/or numeral locants (:44208): "When locants are
needed... then all locants must be cited" (:44210). Printed: '(2-2H1)acetic acid (PIN)' and
'(O-2H)acetic acid (PIN)' (:43804,:43806), '1,1,1-trifluoro(2-2H1)ethane (PIN)' (:44155, under
Numbering in relation to the unmodified compound), 'ethan(2H)ol (PIN) (as in ethanol)' (:44184),
'(1-14C)pentan(3H)oic acid (PIN)' (:43812), 'cyclohexane(2H)carboxylic acid (PIN)' (:43816),
'(2H3)acetonitrile (PIN) (as in trichloroacetonitrile)' (:44182).

A descriptor that cites one hydrogen, count-less and locant-free, is the Blue Book spelling in two places.

* A slot of a group that holds one hydrogen: the nuclide stands directly before the suffix or prefix that
  names the group ('ethan(2H)ol', 'pentan(3H)oic acid', 'cyclohexane(2H)carboxylic acid',
  'methane(2H)sulfonic acid', '2-(2H)hydroxypropanoic acid'). The position is the group, so there is
  neither a count to show nor an isomer to tell apart.
* A unit that holds exactly one hydrogen whose unmodified name cites no locant ('trichloro(2H)methane',
  '(2H)formonitrile'): the nuclide has no other place, and no locant is necessary in the unmodified name.
  The unit is the whole compound for a descriptor outside enclosing marks and the group between the
  marks for a descriptor inside them.

Everywhere else the count is shown or a locant is cited. A front descriptor over a compound with several
hydrogens shows its count ('(2H)phosphoric acid' has three equivalent OH). It also cites a locant when the
hydrogens are not all alike ('(2H)formic acid' has the isomer '(O-2H)formic acid') or when the unmodified
name cites locants ('1-chloro-1,2,2,2-tetrafluoroethane': '1-chloro-1,2,2,2-tetrafluoro(1-2H)ethane').

OPSIN 2.9.0 rejects the count-less spelling, so it is only ever read through its twin
(``validation.twin_spellings``, rule ``isotope_count_one``).:func:`twin_licensed` keeps that reading from
licensing a count-less spelling where the Blue Book shows the count, and:func:`needs_locant` sends the
placement search on to the locanted spellings where it cites a locant. Both apply at the default tier only:
a wider tier shows the count and keeps the names it has.
"""
import re
from typing import Optional

from rdkit import Chem

#: the part of a name that a one-hydrogen group stands in: the suffix of an alcohol, an acid or an
#: aldehyde, or the prefix of a hydroxy or sulfanyl group; a multiplied suffix ('diol', 'dioic acid') is
#: none of them
_GROUP_SLOT = re.compile(r"(?:(?:ol|thiol|selenol|tellurol|oic acid|carboxylic acid|sulfonic acid|"
                         r"sulfinic acid|carbaldehyde)(?![a-z])|hydroxy|sulfanyl|selanyl|tellanyl)")
_INDICATED_HYDROGEN = re.compile(r"\d+[a-z]?'*H-")


def _outside_marks(name: str, pairs) -> str:
    """``name`` without the text between enclosing marks."""
    hidden = set()
    for o, c in pairs:
        hidden.update(range(o, c + 1))
    return "".join(ch for i, ch in enumerate(name) if i not in hidden)


def _hydrogen_unit(mol: Chem.Mol, carrier: int) -> set:
    """The atoms of the named unit that holds the hydrogen carrier ``carrier``: the run of its element that
    stays in one substituent or parent, and the heteroatoms bonded to it."""
    from . import isotopes as iso
    unit = iso._element_scope(mol, carrier, mol.GetAtomWithIdx(carrier).GetSymbol())
    return unit | {nb.GetIdx() for i in unit for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                   if nb.GetAtomicNum() > 1 and nb.GetSymbol() != "C"}


def licensed(skel: str, off: int, original: Chem.Mol) -> bool:
    """Whether a count-less, locant-free hydrogen nuclide may stand at ``off`` of the unmodified name
    ``skel`` of the structure ``original`` (see the module text)."""
    from . import isotopes as iso
    if _GROUP_SLOT.match(skel, off):
        return True
    pairs = iso._enclosing_pairs(skel)
    hydrogens = Chem.AddHs(original)
    if not any(o < off <= c for o, c in pairs):
        # the whole compound: one hydrogen, and no locant in its unmodified name
        cited = _INDICATED_HYDROGEN.sub("", _outside_marks(skel, pairs))
        return not re.search(r"\d", cited) and sum(a.GetAtomicNum() == 1 for a in hydrogens.GetAtoms()) == 1
    if iso._enclosing_unit_cites_a_locant(skel, off):
        return False
    labelled = [a for a in hydrogens.GetAtoms() if a.GetAtomicNum() == 1 and a.GetIsotope()]
    if len(labelled) != 1:
        return False
    unit = _hydrogen_unit(hydrogens, iso._isotope_carrier(labelled[0]))
    return sum(nb.GetAtomicNum() == 1 for i in unit for nb in hydrogens.GetAtomWithIdx(i).GetNeighbors()) == 1


def needs_locant(original: Chem.Mol, skel: str, off: int) -> bool:
    """True at the default tier when the one labelled hydrogen of ``original`` is cited without count at
    ``off`` of ``skel`` (it sits on a carrier that bears no other hydrogen) and the Blue Book then cites a
    locant: the hydrogens of its unit are not all alike, so there is a possibility of isomers,
    or the unmodified name cites locants. A slot of a one-hydrogen group needs none. The
    placement search then goes on to the locanted spellings."""
    from ..validation.twin_spellings import pin_spellings_on
    if not pin_spellings_on():
        return False
    from . import isotopes as iso
    labelled = [a for a in original.GetAtoms() if a.GetAtomicNum() == 1 and a.GetIsotope()]
    if len(labelled) != 1 or iso._max_atoms_at_position(labelled[0]) != 1 or _GROUP_SLOT.match(skel, off):
        return False
    pairs = iso._enclosing_pairs(skel)
    parent = Chem.Mol(original)
    for atom in parent.GetAtoms():
        atom.SetIsotope(0)
    parent = Chem.AddHs(parent)
    ranks = list(Chem.CanonicalRankAtoms(parent, breakTies=False))
    if not any(o < off <= c for o, c in pairs):
        cited = bool(re.search(r"\d", _INDICATED_HYDROGEN.sub("", _outside_marks(skel, pairs))))
        unit = {a.GetIdx() for a in parent.GetAtoms()}
    else:
        cited = iso._enclosing_unit_cites_a_locant(skel, off)
        unit = _hydrogen_unit(parent, iso._isotope_carrier(parent.GetAtomWithIdx(labelled[0].GetIdx())))
    hosts = {i for i in unit if parent.GetAtomWithIdx(i).GetAtomicNum() > 1
             and any(nb.GetAtomicNum() == 1 for nb in parent.GetAtomWithIdx(i).GetNeighbors())}
    return cited or len({ranks[i] for i in hosts}) > 1


def twin_licensed(candidate: str, original: Optional[Chem.Mol]) -> bool:
    """Whether every count-less hydrogen item of the descriptors of ``candidate`` stands where the Blue Book
    cites it so; the isotope oracle reads a candidate through its twin only then."""
    from ..validation.twin_spellings import bare_hydrogen_groups, descriptor_spans
    bare = bare_hydrogen_groups(candidate)
    if not bare:
        return True
    if original is None:
        return False
    spans = descriptor_spans(candidate)
    skel = candidate
    for start, end in reversed(spans):
        skel = skel[:start] + skel[end:]
    for start, _end in bare:
        if not licensed(skel, start - sum(e - s for s, e in spans if e <= start), original):
            return False
    return True
