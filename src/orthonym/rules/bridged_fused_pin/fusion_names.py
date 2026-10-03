"""Fusion names of the carbocyclic parents the tables do not hold: two or three
ortho-fused rings whose name has exactly two components, numbered by OPSIN's reading of it.

 (the Blue Book): "'ortho-Fused' or 'ortho- and peri-fused' polycyclic ring
systems with the maximum number of noncumulative double bonds (mancude) that have no
accepted retained or systematic name described in sections and are named by
prefixing to the name of a component ring or ring system (the parent component)
designations of the other component(s) (attached components)." This module builds that
name for one class and declines everything else:

- two rings of sizes m < n, n >= 7: '<prefix>[n]annulene'. (:11968):
  "Monocyclic parent components are named as [n]annulenes where n represent the number of
  carbon atoms. The series starts at n = 7"; (c) (:12234) "A component
  containing the larger ring at the first point of difference when comparing rings in
  order of decreasing size" makes the larger ring the parent; (:13970) "the
  omission is also recommended when the fused ring system is made of two monocyclic
  hydrocarbons" ('1H-cyclopenta[8]annulene (PIN)':13974, 'benzo[8]annulene (PIN)':19564);
- three rings in a row X-Y-Z where X-Y or Y-Z is a two-ring hydrocarbon component
  (pentalene, indene, azulene, naphthalene, heptalene, octalene: and the
   polyalenes): (b) (:12163) "a component containing the greater number
  of rings", then (c), choose the parent; the third ring is the attached component, cited
  with the letter of the parent side it is fused to: (:11911) "each peripheral
  side of the parent component (including sides whose locants are distinguished by
  letters, for example, 2a,3a) using the italic letters a, b, c, etc., beginning with a for
  the side numbered '1,2'... To the letter as early in the alphabet as possible that
  denotes the side where the fusion occurs are prefixed, if necessary, the numbers"; the
  numbers are omitted, (:13970) "numerical and/or letter locants are omitted in
  fused ring systems with only first-order monocyclic attached hydrocarbon components,
  'benzo', and those described in " ('benzo[a]tetracene (PIN)':13972;
  '4,7-methanocyclopenta[a]indene (PIN)':19829).

Prefixes, (:12000): "Monocyclic hydrocarbon prefixes for attached components
other than 'benzo' are formed by dropping 'ne' from the name of the appropriate saturated
monocyclic hydrocarbon.... There is no upper ring-size limit to this criterion." The final
'o'/'a' is kept Note:11909). (:23710): "Fusion nomenclature gives
preferred IUPAC names only to compounds having at least two rings of at least five or more
members."

Declined: heteroatoms, three components, four or more rings, peri-fusion (an atom in three
rings), two equal rings (the polyalenes are table names), a skeleton the tables name, and
any name OPSIN 2.9.0 cannot read back to the same skeleton."""
import re
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

#: /: two-ring hydrocarbon components by ring sizes
_TWO_RING_COMPONENTS = {(5, 5): "pentalene", (6, 5): "indene", (7, 5): "azulene",
                        (6, 6): "naphthalene", (7, 7): "heptalene", (8, 8): "octalene"}
#: (the Blue Book, Table 1.4) numerical terms: the saturated monocycle of n
#: atoms is cyclo<term>ne (cyclononane), its prefix cyclo<term>a
_NUMERAL = {3: "propa", 4: "buta", 5: "penta", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
            11: "undeca", 12: "dodeca", 13: "trideca", 14: "tetradeca", 15: "pentadeca",
            16: "hexadeca", 17: "heptadeca", 18: "octadeca", 19: "nonadeca", 20: "icosa"}
_AV = re.compile(r"^(\S+)\s+\|\$_AV:(.*)\$\|\s*$")


def _prefix(size: int) -> Optional[str]:
    """ (:12000): the attached-component prefix of a monocyclic hydrocarbon."""
    if size == 6:
        return "benzo"
    numeral = _NUMERAL.get(size)
    return f"cyclo{numeral}" if numeral else None


def _rings(key_mol) -> Optional[List[frozenset]]:
    """The rings of an ortho-fused carbocycle without peri-fusion, or None."""
    if any(a.GetAtomicNum() != 6 for a in key_mol.GetAtoms()):
        return None
    ri = key_mol.GetRingInfo()
    if not ri.NumRings():
        Chem.FastFindRings(key_mol)
    rings = [frozenset(r) for r in Chem.GetSymmSSSR(key_mol)]
    if len(rings) != key_mol.GetNumBonds() - key_mol.GetNumAtoms() + 1:
        return None
    if set().union(*rings) != set(range(key_mol.GetNumAtoms())):
        return None
    for a in range(key_mol.GetNumAtoms()):
        if sum(1 for r in rings if a in r) > 2:
            return None                       # peri-fusion
    for x, y in combinations(rings, 2):
        if len(x & y) not in (0, 2):
            return None
    return rings


def _component_numberings(key_mol, atoms: frozenset) -> List[Dict[int, object]]:
    """Every numbering of the two-ring component on ``atoms`` (its table numbering composed
    with each automorphism), as {key_mol atom: locant}."""
    from .parents import fused_parent
    got = fused_parent(key_mol, set(atoms))
    return [dict(n) for n in got.numberings] if got is not None else []


def _side_letter(numbering: Dict[int, object], bond: Tuple[int, int], key_mol) -> Optional[str]:
    """ (:11911): the letter of the peripheral side holding ``bond`` when the
    sides are lettered 'a' (1-2), 'b' (2-3),... around the component in its numbering
    order, the sides of fusion atoms included."""
    from .numbering import loc_key
    order = sorted(numbering, key=lambda a: loc_key(numbering[a]))
    sides = list(zip(order, order[1:] + order[:1]))
    for i, (u, v) in enumerate(sides):
        if key_mol.GetBondBetweenAtoms(u, v) is None:
            return None                       # not one peripheral cycle
        if {u, v} == set(bond):
            return chr(ord("a") + i) if i < 26 else None
    return None


def two_component_name(key_mol) -> Optional[str]:
    """The fusion name (no indicated hydrogen) of the carbocyclic skeleton ``key_mol`` (an
    element-aware all-single-bond graph, ``parents._key_of``), or None outside the class of
    the module docstring. A skeleton the parent tables name has a retained, systematic or
    Blue Book name, which (:11903) puts before any fusion name: declined."""
    from ...data import get_retained_name
    from .parents import _IH, _mancude, _table_index, is_pin_parent_name
    if _table_index().get(Chem.MolToSmiles(key_mol)):
        return None
    std = _mancude(key_mol)
    retained = get_retained_name(Chem.MolToSmiles(std)) if std is not None else None
    if retained and is_pin_parent_name(_IH.sub("", retained)):
        return None
    return _rule_name(key_mol)


def _rule_name(key_mol) -> Optional[str]:
    """The name the rules of the module docstring give the skeleton, before the tables are
    asked (the tests rebuild the book's own names of the class with it)."""
    rings = _rings(key_mol)
    if rings is None or not 2 <= len(rings) <= 3:
        return None
    # (:23710) "Fusion nomenclature gives preferred IUPAC names only to
    # compounds having at least two rings of at least five or more members"
    if sum(1 for r in rings if len(r) >= 5) < 2:
        return None
    if len(rings) == 2:
        m, n = sorted(len(r) for r in rings)
        if m == n or n < 7 or (m, n) == (5, 7):
            return None                       # polyalenes, azulene: table names
        prefix = _prefix(m)
        return f"{prefix}[{n}]annulene" if prefix else None
    # three rings: the middle one shares a bond with each of the others
    middle = [r for r in rings if sum(1 for o in rings if o is not r and len(o & r) == 2) == 2]
    if len(middle) != 1:
        return None
    y = middle[0]
    x, z = [r for r in rings if r is not y]
    if x & z:
        return None
    best = None
    for comp, other in ((x | y, z), (y | z, x)):
        sizes = tuple(sorted((len(r) for r in rings if r <= comp), reverse=True))
        name = _TWO_RING_COMPONENTS.get(sizes)
        if name is None:
            continue
        rank = sizes                          # (b) two rings each, then (c)
        if best is None or rank > best[0]:
            best = (rank, [(name, comp, other)])
        elif rank == best[0]:
            best[1].append((name, comp, other))
    if best is None or len({c[0] for c in best[1]}) != 1:
        return None
    letters = []
    for name, comp, other in best[1]:
        prefix = _prefix(len(other))
        if prefix is None:
            return None
        shared = tuple(sorted(comp & other))
        if len(shared) != 2:
            return None
        numberings = _component_numberings(key_mol, comp)
        if not numberings:
            return None
        for numbering in numberings:
            letter = _side_letter(numbering, shared, key_mol)
            if letter is None:
                return None
            letters.append((letter, prefix, name))
    if not letters:
        return None
    letter, prefix, name = min(letters)
    return f"{prefix}[{letter}]{name}"


#: the forms of a name ``opsin_structure`` offers OPSIN, in order: OPSIN supplies the
#: indicated hydrogen of most bare mancude names itself and reads the '1H-' form of the
#: others (every name the rules give for two rings of up to 20 members and three rings of
#: 3 to 8 members, 195 names: 191 read bare, 4 only as '1H-', e.g. 'cyclopenta[a]azulene')
_IH_FORMS = ("", "1H-", "2H-")


@lru_cache(maxsize=512)
def opsin_structure(name: str, n_atoms: int) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """(SMILES, $_AV locants) of OPSIN 2.9.0's reading of ``name`` (a mancude system with an
    odd number of atoms needs an indicated hydrogen): the first of the forms ``_IH_FORMS``
    that OPSIN reads and RDKit parses decides, because an indicated hydrogen moves no
    skeleton atom, so if its atom count is not ``n_atoms`` no other form's is. None when
    that count differs or no form is read. A call OPSIN could not answer raises
    ``validation.opsin_roundtrip.OpsinUnavailable``, which the cache does not keep."""
    from ...validation.opsin_roundtrip import extended_smiles_or_unavailable
    for ih in _IH_FORMS:
        line = extended_smiles_or_unavailable(f"{ih}{name}")
        m = _AV.match((line or "").strip())
        if not m:
            continue
        mol = Chem.MolFromSmiles(m.group(1))
        if mol is None:
            continue
        locants = tuple(m.group(2).split(";"))
        if mol.GetNumAtoms() == len(locants) == n_atoms:
            return m.group(1), locants
        return None
    return None
