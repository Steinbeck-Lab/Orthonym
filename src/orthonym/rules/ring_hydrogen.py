"""Indicated and added hydrogen of a ring system whose atoms carry a double bond to an atom
outside the ring (a ketone, thione or imine suffix, or an oxo, imino or ylidene prefix).

One rule for every producer that writes a fused or Hantzsch-Widman parent with such a group.
The hydrogen of the name is that of the MANCUDE PARENT of the ring skeleton, with the groups
accommodated afterwards:

- (the Blue Book) "After the maximum number of noncumulative double bonds has
  been assigned to the ring structure, any ring atom with a bonding number of three or higher
  connected to adjacent ring atoms by single bonds only, and carrying one or more hydrogen
  atoms, is designated by indicated hydrogen"; (:14569) the same for fused systems.
- (:3721) "in a preferred IUPAC name a locant and the symbol 'H' must be cited";
   (:14607) "In preferred IUPAC names, all indicated hydrogen atoms must be cited".
- (:24766) "Indicated hydrogen is cited at any position of a ring system in order to
  accommodate principal characteristic groups... provided that there are an equal or greater
  number of indicated hydrogen atoms available"; (:24768) the indicated hydrogen
  atoms "are placed at peripheral atoms that will accommodate these principal characteristic
  groups" ('7H-1-benzopyran-7-one (PIN)':24776, '2H,7H-pyrano[2,3-b]pyran-2,7-dione (PIN)'
  :24782); (:24794) the others go "to the lowest nonfusion peripheral atom";
   (:24806) when there are fewer.
- (:3725), (:24689) 'added indicated hydrogen' for the groups no indicated
  hydrogen accommodates, "cited in parentheses after the locant of the structural feature",
  "preferred over the use of nondetachable hydro prefixes"; (:24721) none for a
  pair of groups that "simply removes a double bond".
- (:28410) "Ketones derived from mancude parent hydrides having indicated hydrogen
  atoms are named by direct substitution of a >CH2 group... When no indicated hydrogen is
  present, the methodology of 'added indicated hydrogen' is applied": '4H-pyran-4-one (PIN)
  pyran-4-one' (:28414), '1H-inden-1-one (PIN) inden-1-one' (:28416), 'naphthalen-1(2H)-one
  (PIN)' (:28418).
- Note (:24691): indicated hydrogen placed after the group is introduced (so that no
  added hydrogen is needed) "is not recommended for use in constructing IUPAC names".

The placement itself is the bridged fused builder's (``bridged_fused_pin.hydro``: ``hydro_state``
reads the mancude parent as a maximum matching over the atoms that keep a free valence after
their ring bonds -- so a neutral ring nitrogen with three ring bonds, which can carry neither
kind of hydrogen, is never a position -- and ``accommodate`` applies -.4 and
/.3). A ring system without bridges is the same computation with no bridge atoms.
"""
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class RingHydrogen:
    """Where the name of one ring system puts its hydrogen, as ring atom indices."""
    indicated: FrozenSet[int]     # cited in front of the parent: '2H,5H-'
    added: FrozenSet[int]         # cited after the suffix locants: '-4(3H)-one'
    hydro: FrozenSet[int]         # 'hydro' prefixes, cited before the indicated hydrogen
    groups: FrozenSet[int]        # the suffix atoms that lack hydrogen in the mancude parent
    state: Any = None             # bridged_fused_pin.hydro.HydroState (for the hydro text)


def exocyclic_double_atoms(mol, ring_atoms: Iterable[int]) -> FrozenSet[int]:
    """Ring atoms with a double bond to an atom outside ``ring_atoms`` (=O, =S, =NH, =CH2...),
    read from a Kekule form (an aromatic-flagged C=O ring carbon counts)."""
    ring = set(ring_atoms)
    out = set()
    for a in ring:
        for b in mol.GetAtomWithIdx(a).GetBonds():
            if (b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtomIdx(a) not in ring):
                out.add(a)
    return frozenset(out)


def _kekule(mol):
    km = Chem.Mol(mol)
    try:
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return None
    return km


def ring_hydrogen(mol, ring_atoms: Iterable[int], suffix_atoms: Iterable[int],
                  atom_to_locant: Dict[int, Any]) -> Optional[RingHydrogen]:
    """The indicated, added and hydro positions of ring system ``ring_atoms`` under the
    numbering ``atom_to_locant``, with the =X groups of ``suffix_atoms`` expressed as suffixes
    (the other =X atoms of the system are prefixes, (:24864) "After the introduction
    of indicated and 'added indicated hydrogen' atoms, all substituent groups not expressed as
    suffixes are cited as prefixes", so they are ordinary saturated positions here). None when the rule
    cannot be applied (a charged or radical ring atom, a parent with more than three indicated
    hydrogen atoms, a structure that is no hydro derivative of the mancude parent)."""
    from .bridged_fused_pin import hydro
    from .bridged_fused_pin.selection import Split
    ring = frozenset(ring_atoms)
    if not ring or any(a not in atom_to_locant for a in ring):
        return None
    km = _kekule(mol)
    if km is None:
        return None
    unsat, n_double = set(), 0
    for b in km.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring and j in ring and b.GetBondType() == Chem.BondType.DOUBLE:
            unsat.update((i, j))
            n_double += 1
    state = hydro.hydro_state(km, Split(ring, (), (), None, n_double, frozenset(unsat)))
    if state is None:
        return None
    groups = frozenset(a for a in suffix_atoms if a in ring
                       and hydro.lacks_hydrogen(state, a, 2))
    got = hydro.accommodate(state, atom_to_locant, groups)
    if got is None:
        return None
    ih, added, hydro_atoms = got
    return RingHydrogen(frozenset(ih), frozenset(added), frozenset(hydro_atoms), groups, state)


def indicated_text(rh: RingHydrogen, atom_to_locant: Dict[int, Any]) -> str:
    """'' or '6H' or '2H,5H'."""
    from .bridged_fused_pin import hydro
    return hydro.ih_text(rh.indicated, atom_to_locant)


def added_text(rh: RingHydrogen, atom_to_locant: Dict[int, Any]) -> str:
    """'' or '3H' or '1H,5H' (the text inside the parentheses after the suffix locants)."""
    from .bridged_fused_pin import hydro
    return hydro.ih_text(rh.added, atom_to_locant)


def hydro_text(rh: RingHydrogen, atom_to_locant: Dict[int, Any]) -> Optional[str]:
    """'' or '7,8-dihydro' (None: a count with no multiplying prefix). (:16880):
    "If indicated hydrogen atoms are present in a name, the 'hydro' prefixes precede them"."""
    from .bridged_fused_pin import hydro
    return hydro.hydro_text(rh.state, rh.hydro, atom_to_locant, rh.indicated, rh.groups)


def parent_prefix(rh: RingHydrogen, atom_to_locant: Dict[int, Any]) -> Optional[str]:
    """The text written in front of the parent stem: '7,8-dihydro-2H,5H-', '6H-',
    '3,4-dihydro', '' (None when the hydro count has no multiplying prefix). A hyphen ends
    the text when the stem must follow a locant ('2H-' + 'pyran'); a bare hydro prefix joins
    a stem that starts with a letter directly ('3,4-dihydro' + 'quinolin...'), the caller
    adds the hyphen before a stem that starts with a locant or a bracket."""
    ht = hydro_text(rh, atom_to_locant)
    if ht is None:
        return None
    it = indicated_text(rh, atom_to_locant)
    if it:
        return f"{ht}-{it}-" if ht else f"{it}-"
    return ht


def numbering_key(rh: RingHydrogen, atom_to_locant: Dict[int, Any],
                  suffix_atoms: Iterable[int]) -> Tuple:
    """ (:25036-25044) (c) indicated hydrogen, (d) principal group as
    suffix, (e) added indicated hydrogen, (f) hydro prefixes: the part of the lowest-locants
    comparison this rule decides (the producer appends its prefix locants, (g))."""
    from .bridged_fused_pin.numbering import locant_tuple
    return (locant_tuple(atom_to_locant[a] for a in rh.indicated),
            locant_tuple(atom_to_locant[a] for a in suffix_atoms),
            locant_tuple(atom_to_locant[a] for a in rh.added),
            locant_tuple(atom_to_locant[a] for a in rh.hydro))
