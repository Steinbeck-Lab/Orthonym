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

A count-only descriptor over a retained name whose characteristic group is IMPLIED (``implied_group_hydrogens``).
 (:43800) prints '(2-2H1)acetic acid (PIN)' (:43804) next to '(O-2H)acetic acid (PIN)' (:43806), and
 (:43824) '(N-2H1)acetamide (PIN)' (:43828): the O-H of 'acetic acid' and the N-H of 'acetamide' are
positions of the unit the front descriptor stands before, because the name has no suffix of its own for them
('ethan(2H)ol',:44184, has one). A descriptor that cites only a count, '(2H3)acetic acid', then has the isomer
'(2-2H1,O-2H)acetic acid',:44202), so its locants are cited, '(2,2,2-2H3)acetic acid', and with them
the locants of the unmodified name,:44180). The retained parents 'acetic acid', 'acetonitrile' and
'acetaldehyde' omit the locants of their substituents,:3031); while the label sits on that parent
the omission is withdrawn (``retained_c2_unit_carries_isotope``, ``locant_omission.isotope_retained_c2_unit_scope``):
'2-hydroxy(2,2-2H2)acetic acid'.

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


#: the word of a name that names the group of an acid or an amide; a hydrogen of such a group is in the unit of
#: the name that holds the word ('acetic acid', 'acetamide', 'acetamido'), not in the unit of a name that
#: names the group by a prefix ('1-(nitrosoamino)-1-oxoethane')
_ACID_WORD = re.compile(r"\bacid\b")
_AMIDE_WORD = re.compile(r"amid|hydrazid")


def _prefix_head(skel: str, off: int) -> Optional[str]:
    """The substituent prefix that the word at ``off`` of ``skel`` begins with and runs into the next part of the
    name ('acetamido' of 'acetamidoacetic acid', 'acetyl' of 'acetylacetamide', 'methyl' of 'methylamino'): a
    prefix that the shared prefix vocabulary recognises, or an alkyl or acyl word ending in 'yl'. None when the
    word is not a prefix run into a longer one: the unit is the rest of the name then."""
    from ..assembly.naming_utils import _is_recognised_prefix_component
    run = re.match(r"[a-z]+", skel[off:])
    if not run:
        return None
    word = run.group(0)
    for k in range(len(word) - 1, 2, -1):
        if _is_recognised_prefix_component(word[:k]):
            return word[:k]
    acyl = re.match(r"[a-z]*?(?:yl|oyl)(?=[a-z])", word)
    return acyl.group(0) if acyl else None


def implied_group_hydrogens(mol: Chem.Mol, scope: set, carriers: set, skel: str, off: int) -> set:
    """The heteroatoms of the unit named at ``off`` of ``skel`` whose hydrogen the unit holds without a slot of
    its own: the O-H of 'acetic acid', the N-H of 'acetamide' and 'acetamido'. ``mol`` is the unlabelled
    structure with explicit hydrogens, ``scope`` the carbon run of the labelled ``carriers``
    (``isotopes._element_scope``).

     (the Blue Book) prints '(2-2H1)acetic acid (PIN)' (:43804) and '(O-2H)acetic acid (PIN)'
    (:43806); (:43824) '(N-2H1)acetamide (PIN)' (:43828): the front descriptor has the O-H and the N-H
    of the retained name in its scope, so a count-only '(2H3)acetic acid' has an isomer and (:44202,
    "Locants are not omitted when there is a possibility of isomers") cites the locants. A suffix written out
    carries its own slot, 'ethan(2H)ol' (:44184), 'pentan(3H)oic acid' (:43812), and a prefix is a unit of its
    own ('N-(2H3)acetylacetamide' labels the acetyl,:43730 '(2H3)methoxybenzene'; the unit is then the prefix
    word, ``_prefix_head``): those hydrogens are not in the scope, and neither is a heteroatom group of the
    labelled carbon itself (a 'hydroxy' prefix or an '-ol' suffix on it).

    The implied group is the characteristic group of an acid or an amide: a non-ring heteroatom with a hydrogen,
    single-bonded to a carbon of the scope that bears a double bond to a heteroatom, and named by the unit (the
    word 'acid' for an oxygen, 'amid-' or 'hydrazid-' for a nitrogen). An acid group stays implied unless the
    unit spells it with a suffix slot (``_GROUP_SLOT``, 'propanoic acid'); the group of a chalcogen heavier than
    oxygen has the slot of its systematic suffix ('ethanethioic S-acid') and is not read."""
    head = _prefix_head(skel, off)
    unit = head if head is not None else skel[off:]
    found = set()
    for a in scope - carriers:
        atom = mol.GetAtomWithIdx(a)
        if not any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() not in (1, 6)
                   for b in atom.GetBonds()):
            continue
        for x in atom.GetNeighbors():
            if (x.IsInRing() or mol.GetBondBetweenAtoms(a, x.GetIdx()).GetBondType() != Chem.BondType.SINGLE
                    or not any(nb.GetAtomicNum() == 1 for nb in x.GetNeighbors())):
                continue
            if x.GetSymbol() == "N":
                implied = _AMIDE_WORD.search(unit)
            elif x.GetSymbol() == "O":
                implied = _ACID_WORD.search(unit) and not _GROUP_SLOT.search(unit)
            else:
                implied = False
            if implied:
                found.add(x.GetIdx())
    return found


def retained_c2_unit_carries_isotope(mol: Optional[Chem.Mol]) -> bool:
    """True when a labelled atom sits on a two-carbon chain unit that a retained parent names:
    'acetic acid' (and 'acetate'), 'acetonitrile', 'acetaldehyde',,
    . The unit is the carbon C-1 of a carboxyl, ester, nitrile or aldehyde group, the
    acyclic carbon C-2 bonded to it, and the heteroatoms of the group; an atom of it is labelled when it
    is, or when a hydrogen on it is, the Blue Book: the locants of the parent are cited
    when the parent holds the label, not when a substituent does). A label in a substituent unit with
    marks of its own, an amide unit ('acetamido') or a ring is not read; this is a structural proxy for
    the parent the isotope module does not name (a longer chain reads too, and its producers do not use
    the retained parents), and every emission stays OPSIN-round-trip-gated. A stereogenic C-2 is not
    read: the producers cite its descriptor bare ('(R)-chloro(hydroxy)acetic acid',,:44643) on the
    locant-free spelling, which does not fit one that cites its locants, so those names keep the omitted
    form. Gates ``isotope_retained_c2_unit_scope``."""
    if mol is None:
        return False

    def labelled(atom: Chem.Atom) -> bool:
        return bool(atom.GetIsotope()) or any(
            nb.GetAtomicNum() == 1 and nb.GetIsotope() for nb in atom.GetNeighbors())

    for c1 in mol.GetAtoms():
        if c1.GetSymbol() != "C" or c1.IsInRing():
            continue
        bonds = [(nb, mol.GetBondBetweenAtoms(c1.GetIdx(), nb.GetIdx()).GetBondType())
                 for nb in c1.GetNeighbors() if nb.GetAtomicNum() not in (1, 6)]
        nitrile = any(t == Chem.BondType.TRIPLE and nb.GetSymbol() == "N" for nb, t in bonds)
        oxo = any(t == Chem.BondType.DOUBLE and nb.GetSymbol() == "O" for nb, t in bonds)
        oxy = any(t == Chem.BondType.SINGLE and nb.GetSymbol() == "O" for nb, t in bonds)
        if not (nitrile or (oxo and (oxy or c1.GetTotalNumHs(includeNeighbors=True) > 0))):
            continue
        for c2 in c1.GetNeighbors():
            if (c2.GetSymbol() == "C" and not c2.IsInRing()
                    and c2.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                    and any(labelled(a) for a in [c1, c2] + [nb for nb, _t in bonds])):
                return True
    return False


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
