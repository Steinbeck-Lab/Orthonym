"""Assemblies of identical parent radicals, the candidate of the radical floor for centres in separate parts of a molecule.

``assembly/radical_floor.py`` names several monovalent radical centres as the free valences of ONE parent (a ring
system or an acyclic chain that holds them all,, ``universal_substituent.name_universal_polyradical``). A
molecule whose centres sit in different parts of the structure has no such parent. When those parts are identical
and joined by one linker that the floor can name as a polyvalent group, the book names the molecule as an
assembly of identical units, "ASSEMBLIES OF PARENT RADICALS" (the Blue Book): "Polyradicals with
radical centers identically derived from the same parent hydride or the same characteristic group (except for
polyacyl or polyamide radicals described in and, respectively) but located in different parts of
the structure are named, if possible, according to the principles for nomenclature of assemblies of identical
units linked by multivalent substituents (see." Its examples are '(naphthalene-2,6-diyl)bis(disulfanyl)
(PIN)', '(2,4-dimethylpentane-2,4-diyl)bis(oxyl) (PIN)' and '(cyclobutane-1,3-diyl)bis(peroxyl) (PIN)' (with
'bis(oxidanyl)' and 'bis(dioxidanyl)' as the second spelling).

The name built here is ``(<linker>-diyl)bis(<unit>)``. The molecule is cut at bridges (bonds that no ring holds) into
k identical units, each holding the radical centres that no bridge separates, and the one linker that holds none
(:func:`unit_splits`). The linker is named by the floor as the polyvalent group whose free valences are the atoms that
the units were bonded to (``universal_substituent.name_universal_polyradical``), so it must be one ring system or one
acyclic chain that holds every one of those atoms. A single ring that holds them is spelled by its book name, numbered
with the free valences as a set (``_BOOK_RING_FREE_VALENCES``, ``rules.monocycle_forms``: 'benzene-1,2-diyl',
'4H-imidazole-2,5-diyl', 'piperidine-4,1-diyl', / /, never by skeletal replacement
('cyclohexa-1,3,5-triene', '1-azacyclohexane'); when the units are bonded through one-atom divalent groups the host is named
with its valences on the atoms of the groups and those cited after it ('ethane-1,2-diylbis(oxy)',, which is tried
before the linker as one chain with its heteroatoms. The unit is named as a parent radical, in either of two forms:

* the radical of the part with the bond to the linker closed by a hydrogen: the radical of the parent hydride of its atom with
  the groups on it as prefixes ('oxyl', 'azanyl', 'dimethylsilyl', 'methylboranyl'; the parent hydride radical comes from the
  element, ``rules.radicals._parent_hydride_radical_name``);
* when the centre is a bare atom bonded to a ring or chain that carries the bond to the linker, that group as a
  divalent substituent of the radical of the atom: '(2,2,6,6-tetramethylpiperidine-4,1-diyl)oxyl', the first locant
  being the atom bonded to the linker, a divalent group cited before the group it is bonded to).

The names are offers of the radical offer (``assembly/radical_offer.py``): inside
``read_back_policy.radical_last_resort`` only, and each ships only when OPSIN ``-r`` reads it back to the input's full
InChIKey and radical graph, so a unit that the name does not place (a site the name leaves open) never ships. The row is
``systematic_verified``, never a PIN: the book gives the PIN form for the examples above, and this namer does not check
a molecule against that form.
"""
import contextlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

from rdkit import Chem

from ..data.multipliers import multiplier_table
from .naming_utils import apply_enclosing_marks

#: the multiplying prefixes of a compound for the number of units
_MULTIPLIERS = multiplier_table(8, kind="complex")
#: the largest molecule (atoms) that the split is tried on: the search is quadratic in the atoms
_MAX_ATOMS = 400
#: the extents of a unit tried, smallest first (a unit may take in the atoms between its centres and the linker)
_MAX_EXTENTS = 6
#: the skeletal atoms that a chain linker may be threaded through (the floor's own spine elements)
_NOT_SKELETAL = frozenset({1, 9, 17, 35, 53})

#: the radicals of a homonuclear two-atom parent hydride (HOO., HSS.): 'Chalcogen analogues' (the Blue Book) and the
#: examples of ('(cyclobutane-1,3-diyl)bis(peroxyl)', '(naphthalene-2,6-diyl)bis(disulfanyl)'), by atomic number
_TWO_ATOM_RADICALS = {8: ("peroxyl", "dioxidanyl"), 16: ("disulfanyl",), 34: ("diselanyl",)}
#: the radicals of the parent hydride of one atom that take no hydrogen of their own: 'oxyl' and 'oxidanyl',, the
#: others from ``rules.radicals._parent_hydride_radical_name`` ('azanyl', 'boranyl', 'silyl',...)
_OXYL = ("oxyl", "oxidanyl")


def _one_atom_radical_terms(symbol: str) -> Tuple[str, ...]:
    """The names of the radical of the parent hydride of the element ``symbol`` (a neutral atom that holds one radical
    electron and the hydrogens it has), or ."""
    from ..rules.radicals import _parent_hydride_radical_name
    if symbol == "O":
        return _OXYL
    named = _parent_hydride_radical_name(symbol, 1)
    return (named,) if named and symbol != "C" else ()


def _bare_parent_radicals() -> frozenset:
    """The unit names that are a parent hydride radical and nothing else: they take 'bis' and never 'di' ('dioxyl' reads as
    the O-O of a peroxide, 'disilyl' as Si-Si; the book writes 'bis(oxyl)', 'bis(disulfanyl)',, the Blue Book)."""
    from ..rules.radicals import _ELIDE_E_HYDRIDE_RADICAL, _GROUP14_HYDRIDE_RADICAL
    elements = set(_ELIDE_E_HYDRIDE_RADICAL) | set(_GROUP14_HYDRIDE_RADICAL) | {"O"}
    terms = {t for e in elements for t in _one_atom_radical_terms(e)}
    return frozenset(terms | {t for names in _TWO_ATOM_RADICALS.values() for t in names})


@dataclass(frozen=True)
class RadicalUnit:
    """One part of the molecule that holds radical centres: its atoms, its centres, ``root`` (the atom of the part
    that is bonded to the linker) and ``anchor`` (the atom of the linker that ``root`` is bonded to)."""

    atoms: FrozenSet[int]
    centres: FrozenSet[int]
    root: int
    anchor: int


@dataclass(frozen=True)
class UnitSplit:
    """The molecule as identical radical units and the linker (every other atom)."""

    units: Tuple[RadicalUnit, ...]
    linker: FrozenSet[int]


# ---------------------------------------------------------------------------------------------------------------
# the split of the molecule
# ---------------------------------------------------------------------------------------------------------------

def _blocks(mol) -> List[int]:
    """The block of every atom: the atoms joined by ring bonds are one block, every other atom its own."""
    block = list(range(mol.GetNumAtoms()))

    def find(i):
        while block[i] != i:
            block[i] = block[block[i]]
            i = block[i]
        return i
    for bond in mol.GetBonds():
        if bond.IsInRing():
            block[find(bond.GetBeginAtomIdx())] = find(bond.GetEndAtomIdx())
    return [find(i) for i in range(mol.GetNumAtoms())]


def _side(mol, start: int, blocked: int) -> FrozenSet[int]:
    """The atoms reached from ``start`` without passing ``blocked`` (a bridge: no atom is reached from both sides)."""
    seen, stack = {start}, [start]
    while stack:
        for nbr in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nbr.GetIdx()
            if j != blocked and j not in seen:
                seen.add(j)
                stack.append(j)
    return frozenset(seen)


def _centre_groups(mol) -> List[FrozenSet[int]]:
    """The radical centres grouped by block: centres that no bridge separates belong to one unit."""
    block = _blocks(mol)
    groups: Dict[int, set] = {}
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            groups.setdefault(block[atom.GetIdx()], set()).add(atom.GetIdx())
    return [frozenset(g) for g in groups.values()]


def _nested_units(mol, group: FrozenSet[int], others: FrozenSet[int]) -> List[RadicalUnit]:
    """The parts of ``mol`` that one bridge cuts off and that hold ``group`` and none of the centres ``others``, the
    smallest first (each holds the one before it)."""
    found = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for root, anchor in ((a, b), (b, a)):
            side = _side(mol, root, anchor)
            if group <= side and not (others & side):
                found.append(RadicalUnit(side, group, root, anchor))
    return sorted(found, key=lambda u: len(u.atoms))


def _part_mol(mol, atoms: FrozenSet[int], root: int, dummy: bool = False):
    """The part ``atoms`` of ``mol`` as a molecule: the bond to the rest closed by a hydrogen on ``root`` (the radical
    of the part), or by a dummy atom when ``dummy`` (the rooted structure that the units are compared by). None when it
    does not sanitise."""
    rw, new = _frozen_copy(mol, atoms)
    if dummy:
        rw.AddBond(new[root], rw.AddAtom(Chem.Atom(0)), Chem.BondType.SINGLE)
    else:
        target = rw.GetAtomWithIdx(new[root])
        target.SetNumExplicitHs(target.GetNumExplicitHs() + 1)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        if not dummy:
            Chem.RemoveStereochemistry(out)     # the stereo of the input is composed on the name by the offer
        return out
    except Exception:
        return None


def _unit_key(mol, unit: RadicalUnit) -> str:
    """The canonical form of the unit as a rooted structure: its atoms with a dummy atom on the bond to the linker."""
    rooted = _part_mol(mol, unit.atoms, unit.root, dummy=True)
    return Chem.MolToSmiles(rooted) if rooted is not None else ""


def unit_splits(mol) -> Iterator[UnitSplit]:
    """The ways to cut ``mol`` into two or more identical radical units and the one linker between them, the smallest
    units first.

    Every radical centre is monovalent and no atom carries a charge or an isotope. A unit is a part that one bridge
    cuts off, with the centres that no bridge separates (a ring system with several centres is one unit); the parts
    are the same rooted structure, pairwise disjoint, and bonded to different atoms of the linker."""
    if mol is None or mol.GetNumAtoms() > _MAX_ATOMS or len(Chem.GetMolFrags(mol)) != 1:
        return
    centres = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if (len(centres) < 2 or any(a.GetNumRadicalElectrons() != 1 for a in centres)
            or any(a.GetFormalCharge() or a.GetIsotope() for a in mol.GetAtoms())):
        return
    groups = _centre_groups(mol)
    if len(groups) not in _MULTIPLIERS:
        return
    everything = frozenset(a.GetIdx() for a in centres)
    nested = [_nested_units(mol, g, everything - g) for g in groups]
    if any(not n for n in nested):
        return
    for first in nested[0][:_MAX_EXTENTS]:
        key = _unit_key(mol, first)
        chosen, taken = [first], set(first.atoms)
        for others in nested[1:]:
            match = next((u for u in others if not (taken & u.atoms) and _unit_key(mol, u) == key), None)
            if match is None:
                break
            chosen.append(match)
            taken |= match.atoms
        else:
            linker = frozenset(range(mol.GetNumAtoms())) - taken
            anchors = [u.anchor for u in chosen]
            if linker and len(set(anchors)) == len(anchors) and all(a in linker for a in anchors):
                yield UnitSplit(tuple(sorted(chosen, key=lambda u: min(u.atoms))), linker)


def split_identical_units(mol) -> Optional[UnitSplit]:
    """The first (smallest-unit) split of ``mol``, or None."""
    return next(unit_splits(mol), None)


def _holds(mol, atoms: List[int]) -> bool:
    """True when one ring system, or one acyclic chain of skeletal atoms, holds every atom of ``atoms`` (the parents
    that ``universal_substituent._poly_radical_host`` carries free valences on): a cheap check that spares the floor the
    naming of a linker that no parent holds."""
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for held in [r for r in rings if r & merged]:
            rings.remove(held)
            merged |= held
        rings.append(merged)
    if any(set(atoms) <= r for r in rings):
        return True
    if any(mol.GetAtomWithIdx(a).IsInRing() for a in atoms):
        return False
    for a in atoms:
        for b in atoms:
            if a != b and any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetAtomicNum() in _NOT_SKELETAL
                              for i in Chem.GetShortestPath(mol, a, b)):
                return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# the names of the linker and of the unit
# ---------------------------------------------------------------------------------------------------------------

def _frozen_copy(mol, keep: FrozenSet[int]):
    """A copy of ``mol`` with only the atoms ``keep``, each holding the hydrogens it had in ``mol`` (an atom that lost
    a neighbour keeps its count: the missing valence is a free valence, never a fill with hydrogen). Returns
    ``(RWMol, {old index: new index})``."""
    rw = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        target = rw.GetAtomWithIdx(atom.GetIdx())
        target.SetNumExplicitHs(atom.GetTotalNumHs())
        target.SetNoImplicit(True)
    for idx in sorted((i for i in range(mol.GetNumAtoms()) if i not in keep), reverse=True):
        rw.RemoveAtom(idx)
    return rw, {old: new for new, old in enumerate(sorted(keep))}


def _with_free_valences(mol, atoms: FrozenSet[int], sites: List[int]):
    """The part ``atoms`` of ``mol`` with a radical centre on each atom of ``sites`` (the atoms that were bonded to
    the rest of the molecule), and the map of the indices, or None."""
    rw, new = _frozen_copy(mol, atoms)
    for site in sites:
        rw.GetAtomWithIdx(new[site]).SetNumRadicalElectrons(1)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        Chem.RemoveStereochemistry(out)         # the stereo of the input is composed on the name by the offer
        return out, new
    except Exception:
        return None, new


def _name_polyvalent(part, first: Optional[int], book_forms: bool, acyloxy_leaf: bool) -> Optional[str]:
    """The name of the group ``part``, whose free valences are its radical centres, as the floor names a polyradical parent
    with the book's spelling of a ring that holds the valences ('benzene-1,2-diyl', 'piperidine-4,1-diyl', 'thiophene-2,3,5-triyl':
    ``universal_substituent._BOOK_RING_FREE_VALENCES``) and the valence of the atom ``first`` cited first when one is given."""
    from .universal_substituent import _BOOK_RING_FREE_VALENCES, _FIRST_VALENCE, name_universal_polyradical
    ring_token = _BOOK_RING_FREE_VALENCES.set(True)
    first_token = _FIRST_VALENCE.set(first)
    try:
        named = name_universal_polyradical(part, book_forms=book_forms, acyloxy_leaf=acyloxy_leaf)
    finally:
        _FIRST_VALENCE.reset(first_token)
        _BOOK_RING_FREE_VALENCES.reset(ring_token)
    return named.name if named is not None and named.name else None


def _divalent_name(mol, atoms: FrozenSet[int], first: int, second: int, book_forms: bool, acyloxy_leaf: bool):
    """The name of the part ``atoms`` as a divalent group whose free valences are on ``first`` and ``second``, the
    locant of ``first`` cited first, or None. One ring system or one acyclic chain must hold both."""
    if first == second or not _holds(mol, [first, second]):
        return None
    part, new = _with_free_valences(mol, atoms, [first, second])
    return None if part is None else _name_polyvalent(part, new[first], book_forms, acyloxy_leaf)


def _parent_radical_names(mol, atoms: FrozenSet[int], root: int, centre: int, acyloxy_leaf: bool) -> List[str]:
    """The names of the parent radical of the part ``atoms`` (its centre ``centre``, the bond to the rest of the molecule
    closed by a hydrogen on ``root``):

    * a heteroatom that is the root and the centre is the radical of its own parent hydride, every group on it a prefix:
      'oxyl', 'azanyl', 'dimethylsilyl', 'methylboranyl', the Blue Book; ``rules.radicals``
      ``_parent_hydride_radical_name``, the prefixes named by the floor);
    * the radical of a homonuclear two-atom hydride, the centre bonded to the root: 'peroxyl', 'disulfanyl';
    * else the floor's name of the radical when it is the radical of a parent hydride ('benzyl', 'cyclohexyl')."""
    from .radical_ion import one_atom_prefix_block
    from .universal_substituent import name_universal_substituent_prefix
    capped = _part_mol(mol, atoms, root)
    if capped is None:
        return []
    new = {old: i for i, old in enumerate(sorted(atoms))}
    atom = capped.GetAtomWithIdx(new[centre])
    names: List[str] = []
    if centre == root and not atom.GetFormalCharge() and not atom.IsInRing():
        block = one_atom_prefix_block(capped, new[centre], acyloxy_leaf)
        if block is not None:
            names += [block + term for term in _one_atom_radical_terms(atom.GetSymbol())]
    elif (len(atoms) == 2 and not atom.GetFormalCharge()
          and mol.GetAtomWithIdx(root).GetAtomicNum() == atom.GetAtomicNum()):
        names += list(_TWO_ATOM_RADICALS.get(atom.GetAtomicNum(), ()))
    heavy = [a.GetIdx() for a in capped.GetAtoms() if a.GetAtomicNum() > 1]
    named = name_universal_substituent_prefix(
        capped, heavy, new[centre], bond_order=1, cap_attachment=True, acyloxy_leaf=acyloxy_leaf)
    if named and named.endswith("yl") and named not in names:
        names.append(named)
    return names


def _unit_forms(mol, split: UnitSplit, book_forms: bool, acyloxy_leaf: bool) -> Iterator[str]:
    """The names of the unit as a parent radical, one at a time. Units of one centre only."""
    unit = split.units[0]
    if len(unit.centres) != 1:
        return
    (centre,) = unit.centres
    yield from _parent_radical_names(mol, unit.atoms, unit.root, centre, acyloxy_leaf)
    atom = mol.GetAtomWithIdx(centre)
    inside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in unit.atoms]
    if len(unit.atoms) < 3 or len(inside) != 1 or unit.root == centre or inside[0] == unit.root:
        return
    # a bare centre bonded to a group that carries the bond to the linker: '(group-diyl)radical'
    group = _divalent_name(mol, unit.atoms - {centre}, unit.root, inside[0], book_forms, acyloxy_leaf)
    if group is not None:
        for parent in _parent_radical_names(mol, frozenset({centre}), centre, centre, acyloxy_leaf):
            yield f"({group}){parent}"


def _spacer_name(atom) -> Optional[str]:
    """The name of the bare atom ``atom`` as a divalent group that joins a linker's host to a unit
    'benzene-1,4-diylbis(oxy)'): the chalcogens by the components of ``rules.multiplicative_general._CHALCOGEN_COMPONENT``
    ('oxy', 'sulfanediyl', 'selanediyl'), NH and CH2 by the single-atom bridges of ``rules.multiplicative``
    ('azanediyl', 'methylene'); None for any other atom."""
    from ..rules.multiplicative import _SINGLE_ATOM_BRIDGES
    from ..rules.multiplicative_general import _CHALCOGEN_COMPONENT
    number, hydrogens = atom.GetAtomicNum(), atom.GetTotalNumHs()
    if number in (8, 16, 34, 52):
        return _CHALCOGEN_COMPONENT.get((number, 1)) if not hydrogens else None
    return _SINGLE_ATOM_BRIDGES.get({(7, 1): "NH", (6, 2): "CH2"}.get((number, hydrogens), ""))


def _spacers(mol, split: UnitSplit) -> Optional[Tuple[str, List[int]]]:
    """The name of the one-atom divalent group that every unit is bonded to the linker through, and the atoms of the
    linker that carry it, or None when the anchors are not such atoms: a bare atom of the table that is bonded to one
    other atom of the linker (the atom of the host)."""
    host, name = [], None
    for unit in split.units:
        atom = mol.GetAtomWithIdx(unit.anchor)
        inside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in split.linker]
        spacer = _spacer_name(atom)
        if (spacer is None or atom.IsInRing() or atom.GetFormalCharge() or len(inside) != 1
                or atom.GetDegree() != 2 or (name is not None and spacer != name)):
            return None
        name = spacer
        host.append(inside[0])
    anchors = {u.anchor for u in split.units}
    return (name, host) if name and len(set(host)) == len(host) and not anchors & set(host) else None


def _assembly_linker_name(mol, atoms: FrozenSet[int], sites: List[int]) -> Optional[str]:
    """The name of the linker ``atoms`` when it is one ring assembly and nothing else (identical ring systems joined
    directly by single bonds, with one free valence on each ring system at ``sites``: '[1,1'-biphenyl]-3,3'-diyl'
    (``rules.ring_assemblies.name_ring_assembly_polyvalent_prefix``, numbered by ``### ****``, the Blue Book),
    or None. The parent that ``_holds`` asks for is one ring system; an assembly is several."""
    from ..rules.multiplicative_general import _ring_systems
    from ..rules.ring_assemblies import detect_ring_assembly, name_ring_assembly_polyvalent_prefix
    systems = sorted((rs for rs in _ring_systems(mol) if rs <= atoms), key=min)
    if len(systems) < 2 or frozenset().union(*systems) != atoms:
        return None
    info = detect_ring_assembly(mol, [set(rs) for rs in systems])
    if info is None or info.get('hydro') or len(info['ring_systems']) != len(systems):
        return None
    return name_ring_assembly_polyvalent_prefix(mol, info, sites)


def _linker_name(mol, split: UnitSplit, book_forms: bool, acyloxy_leaf: bool) -> Iterator[str]:
    """The names of the linker, the book's spelling of its parent first: when the units are bonded through one-atom divalent
    groups, the host named with the free valences on the atoms of the groups, the groups cited after it
    ('cyclohexane-1,4-diylbis(methylene)',, as the multiplicative names of the book cite them); then the linker as the
    one polyvalent group whose free valences are the atoms the units were bonded to."""
    anchors = [u.anchor for u in split.units]
    plans = []
    spacer = _spacers(mol, split)
    if spacer is not None:
        spacer_name, host_sites = spacer
        plans.append((split.linker - set(anchors), host_sites, f"{_MULTIPLIERS[len(split.units)]}({spacer_name})"))
    plans.append((split.linker, anchors, ""))
    for atoms, sites, tail in plans:
        if not _holds(mol, sites):
            assembly = _assembly_linker_name(mol, atoms, sites)
            if assembly:
                yield assembly + tail
            continue
        linker, _new = _with_free_valences(mol, atoms, sites)
        named = None if linker is None else _name_polyvalent(linker, None, book_forms, acyloxy_leaf)
        if named:
            yield named + tail


def _multiplied_units(unit: str, count: int) -> List[str]:
    """The unit multiplied ``count`` times: in the form of ('dimethyl', 'bis(disulfanyl)', 'bis[...]') and with
    'bis' for the units that 'di' leaves open ('dioxyl' reads as the O-O of a peroxide; the book writes 'bis(oxyl)',
    , the Blue Book)."""
    from .naming_utils import enclose_if_compound, multiplied_component
    bis = f"{_MULTIPLIERS[count]}{apply_enclosing_marks(unit, -1)}"
    if unit in _bare_parent_radicals():
        return [bis]
    book = multiplied_component(count, unit, enclose_if_compound(unit))
    return [book] if book == bis else [book, bis]


def radical_assembly_names(mol, book_forms: bool = True, acyloxy_leaf: bool = True,
                           mechanical_units: bool = False) -> Iterator[str]:
    """The names of ``mol``: ``(<linker>-diyl)bis(<unit>)`` for each split of the molecule into identical units
    and each spelling of the unit, or nothing. ``book_forms`` is the spelling of the linker and ``mechanical_units`` the
    mechanical spelling of the groups of the unit (the retry of the floor, ``book_prefixes.mechanical_forms``): the linker keeps
    the book's name of its ring when only the spelling of a group of the unit stops OPSIN from placing the radical."""
    from .book_prefixes import mechanical_forms
    for split in unit_splits(mol):
        if len(split.units) not in _MULTIPLIERS:
            continue
        with contextlib.nullcontext() if book_forms else mechanical_forms():
            linkers = list(_linker_name(mol, split, book_forms, acyloxy_leaf))
        if not linkers:
            continue
        with mechanical_forms() if mechanical_units or not book_forms else contextlib.nullcontext():
            units = list(_unit_forms(mol, split, book_forms and not mechanical_units, acyloxy_leaf))
        for linker in linkers:
            for unit in units:
                for multiplied in _multiplied_units(unit, len(split.units)):
                    yield f"{apply_enclosing_marks(linker, -1)}{multiplied}"
