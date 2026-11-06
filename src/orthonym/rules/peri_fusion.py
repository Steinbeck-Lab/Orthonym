"""Fusion names of ortho- and peri-fused ring systems of two first-order components (issue 216).

A ring system with an interior atom (an atom in three rings) that no other fusion producer names, for example
`[1]benzofuro[4,3,2-bcd][1]benzofuran`, is named from two components: a parent component and one attached component
that share three or more atoms (peri-fusion) or two (ortho-fusion). The name is built from the rules, and OPSIN only
checks that it describes the input:

* "LIMITATIONS OF FUSION NOMENCLATURE: THREE COMPONENTS ORTHO- AND PERI-FUSED TOGETHER" (the Blue Book):
  "It is not possible by these principles to name a system in which a third component is ortho- and peri-fused to two
  components that are themselves ortho- or ortho- and peri-fused together. It is important to recall that benzo
  heterocycles are considered as one component, thus permitting the construction of names that would not otherwise be
  possible." The components are therefore the catalogue's ring systems (retained names, benzo names, two-component
  names), the heteromonocycles and benzene.
* "Seniority criteria for selecting the parent component" (:12139): the parent is the senior component of
  all the components of the system (``fused_ring_selection._rank``). When the senior component leaves more than one
  attached component, the system is not named here.
* "General principles" (:11911): "the locants of the attached component... are as low as possible
  consistent with the numbering of the component, and their order conforms to the direction of lettering of the
  parent component"; (:11938) "All letters referring to the parent component are cited, but only locants of the
  nonfused atoms of the attached component are indicated", and (:13996) "The numerical locants of
  peripheral fusion carbon atoms of a component are omitted with an ortho- and ortho- and peri-fused system".
* (b)/(c): letters as low as possible, then the attached locants as low as possible.

The hydrogen and substituents of a substituted member are added on the numbering of the system; a name
that would cite an interior locant, '3a^1') is never shipped.
"""
import itertools
import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

from rdkit import Chem

from .fusion_descriptors import (
    fusion_attached_prefix,
    fusion_component_name,
    strip_indicated_hydrogen,
)

_Locant = Union[int, str]


def _ring_sets(mol) -> List[FrozenSet[int]]:
    return [frozenset(r) for r in mol.GetRingInfo().AtomRings()]


def _locant_key(loc):
    """Order of locants, 4 < 4a < 5 (``bridged_fused_pin.numbering.loc_key``; imported late: that package imports
    the fused-ring module that imports this one)."""
    from .bridged_fused_pin.numbering import loc_key
    return loc_key(loc)


def _saturated_carbons(mol, atoms) -> int:
    """The carbons of ``atoms`` that are neither aromatic nor carry a double bond (a ring heteroatom may have three
    single bonds in a mancude system)."""
    return sum(1 for idx in atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and not (
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        or any(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetAtomWithIdx(idx).GetBonds())))


def _in_scope(mol) -> Optional[Tuple[FrozenSet[int], FrozenSet[FrozenSet[int]]]]:
    """``(atoms, rings)`` of the one fused ring system of ``mol`` that has an interior atom (an atom in three rings),
    when that system is neutral, mancude and has a ring heteroatom, else None. Other atoms (substituents, other ring
    systems) may be present."""
    rings = _ring_sets(mol)
    systems: List[set] = []
    for ring in rings:
        joined = [sys_ for sys_ in systems if any(len(ring & r) >= 2 for r in sys_)]
        merged = {ring}.union(*joined) if joined else {ring}
        systems = [sys_ for sys_ in systems if sys_ not in joined] + [merged]
    peri = []
    for sys_ in systems:
        atoms = frozenset().union(*sys_)
        if any(sum(1 for r in sys_ if a in r) >= 3 for a in atoms):
            peri.append((atoms, frozenset(sys_)))
    if len(peri) != 1:
        return None
    atoms, sys_rings = peri[0]
    hetero = False
    for idx in atoms:
        a = mol.GetAtomWithIdx(idx)
        if a.GetFormalCharge() or a.GetNumRadicalElectrons():
            return None
        hetero |= a.GetAtomicNum() != 6
    return (atoms, sys_rings) if hetero and _saturated_carbons(mol, atoms) == 0 else None


def _skeleton_query(m: Chem.Mol) -> Chem.Mol:
    """A catalogue entry as a query: element and ring membership kept, every bond generic."""
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    q = Chem.AdjustQueryProperties(m, params)
    for atom in q.GetAtoms():
        atom.SetIsAromatic(False)
    return q


#: a benzo name: '1-benzofuran', '2H-1-benzopyran', '1,3-benzodioxole'
_BENZO_NAME = re.compile(r'^(?:\d+[a-z]?H-)?(?:\d+(?:,\d+)*-)?benzo[a-z]')


class _Component:
    __slots__ = ('name', 'atoms', 'rings', 'numberings', 'benzo', 'benzo_name')

    def __init__(self, name, atoms, rings, numberings, benzo=False):
        self.name, self.atoms, self.numberings, self.benzo = name, atoms, numberings, benzo
        self.rings = frozenset(r for r in rings if r <= atoms)
        self.benzo_name = bool(_BENZO_NAME.match(name))


def _pin_component_name(name: str) -> bool:
    """A retained or benzo name the book uses as a PIN parent,; the list of
    ``bridged_fused_pin.parents``), indicated hydrogen aside."""
    from .bridged_fused_pin.parents import is_pin_parent_name
    return is_pin_parent_name(strip_indicated_hydrogen(name))


def _catalogue_entries():
    """``(skeleton SMILES, entry)`` of every ring system with a numbering: the heterocycle catalogue and the
    retained hydrocarbons (naphthalene, anthracene,...; their numbering keyed by the atoms of their SMILES)."""
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    from ..data.polycyclic_data import POLYCYCLIC_DATA
    yield from FUSED_HETEROCYCLE_DATA.items()
    for name, entry in POLYCYCLIC_DATA.items():
        if entry.get('canonical_smiles') and entry.get('iupac_numbering'):
            yield entry['canonical_smiles'], {'name': name, 'iupac_locants': entry['iupac_numbering']}


@lru_cache(maxsize=1)
def _eligible_entries() -> Tuple[Tuple[str, Chem.Mol, int, Dict], ...]:
    """``(name, skeleton query, ring count, numbering)`` of every catalogue entry that is a fusion component: a PIN
    parent name (not a fusion name, not 'beta-carboline') of a mancude system (no more saturated carbons than the
    name has indicated hydrogen: not 'quinolizidine'). Independent of the molecule, so built once."""
    out = []
    for key, entry in _catalogue_entries():
        name = entry['name']
        if entry.get('pin') is False or not entry.get('iupac_locants') or '[' in name or not _pin_component_name(name):
            continue
        key_mol = Chem.MolFromSmiles(key)
        indicated = len(re.findall(r'\d+[a-z]?H', name.split('-')[0])) if re.match(r'^\d', name) else 0
        if key_mol is None or _saturated_carbons(key_mol, range(key_mol.GetNumAtoms())) > indicated:
            continue
        out.append((name, _skeleton_query(key_mol), len(Chem.GetSymmSSSR(key_mol)), entry['iupac_locants']))
    return tuple(out)


def _catalogue_components(mol, rings) -> List[_Component]:
    out = []
    for name, q, n_rings, loc in _eligible_entries():
        if q.GetNumAtoms() >= mol.GetNumAtoms():
            continue
        by_atoms: Dict[FrozenSet[int], List[Dict[int, _Locant]]] = {}
        for match in mol.GetSubstructMatches(q, uniquify=False, maxMatches=1000):
            atoms = frozenset(match)
            inside = [r for r in rings if r <= atoms]
            if len(inside) != n_rings or frozenset().union(*inside) != atoms:
                continue                   # the match must be whole rings of the input
            by_atoms.setdefault(atoms, []).append({match[i]: loc[i] for i in range(len(match))})
        for atoms, numberings in by_atoms.items():
            out.append(_Component(name, atoms, rings, numberings))
    return out


def _monocycle_components(mol, rings) -> List[_Component]:
    from .fusion_descriptors import _het_tied_ring_orderings, _identify_ring_name
    out = []
    for ring in rings:
        ring_list = list(ring)
        if all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in ring_list):
            if len(ring_list) != 6:
                continue                   # only benzene among the carbocycles (as 'benzo')
            name, benzo = 'benzene', True
            orders = _cyclic_orders(mol, ring_list)
        else:
            name, benzo = _identify_ring_name(mol, ring_list), False
            orders = _het_tied_ring_orderings(mol, ring_list)
            if not name and orders:
                # (:11982): a heteromonocycle without a retained name is a component by its
                # Hantzsch-Widman name ('[1,2]oxathiole')
                from .heterocycles import build_hw_name
                het = [(i + 1, mol.GetAtomWithIdx(a).GetSymbol()) for i, a in enumerate(orders[0])
                       if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
                name = build_hw_name(het, len(ring_list), False, True, describe_structure=True) or ''
            if not name:
                continue
        if not orders:
            continue
        out.append(_Component(name, ring, rings, [{a: i + 1 for i, a in enumerate(o)} for o in orders], benzo))
    return out


def _cyclic_orders(mol, ring_list) -> List[List[int]]:
    """Every numbering walk of a benzene ring (12: each start, both directions)."""
    ring = set(ring_list)
    nbrs = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in ring] for a in ring_list}
    if any(len(v) != 2 for v in nbrs.values()):
        return []
    orders = []
    for start in ring_list:
        for first in nbrs[start]:
            order, prev, cur = [start], start, first
            while cur != start:
                order.append(cur)
                prev, cur = cur, next(n for n in nbrs[cur] if n != prev)
            orders.append(order)
    return orders


def _periphery(mol, comp_atoms, numbering) -> Optional[List[int]]:
    """The atoms of a component's periphery in the order of its lettering: from locant 1 towards locant 2,
    :11911: "the peripheral sides... are lettered a, b, c, etc., beginning with 'a' for the side '1,2'")."""
    from .fusion_numbering import _peripheral_cycle
    cycle = _peripheral_cycle(mol, set(comp_atoms), allow_interior=True)
    one = [a for a, loc in numbering.items() if str(loc) == '1']
    two = [a for a, loc in numbering.items() if str(loc) == '2']
    if cycle is None or len(one) != 1 or len(two) != 1 or one[0] not in cycle:
        return None
    i, n = cycle.index(one[0]), len(cycle)
    if cycle[(i + 1) % n] != two[0]:
        if cycle[(i - 1) % n] != two[0]:
            return None
        cycle, i = cycle[::-1], n - 1 - i
    return cycle[i:] + cycle[:i]


def _descriptor(mol, parent, parent_numbering, attached_numbering, shared, benzo):
    """``(sort key, descriptor)`` for one numbering of each component, or None when the shared atoms are not one
    run of the parent's periphery."""
    periphery = _periphery(mol, parent.atoms, parent_numbering)
    if periphery is None or not set(shared) <= set(periphery):
        return None
    n = len(periphery)
    for start in range(n):
        seq = [periphery[(start + k) % n] for k in range(len(shared))]
        if set(seq) == set(shared):
            break
    else:
        return None
    sides = [(start + k) % n for k in range(len(seq) - 1)]
    if max(sides) >= 26:
        return None
    letters = ''.join('abcdefghijklmnopqrstuvwxyz'[i] for i in sides)
    locs = [] if benzo else [attached_numbering[a] for a in seq]
    if len(seq) >= 3:
        locs = [loc for loc in locs if str(loc).isdigit()]   #: no fusion-carbon locants
    if any(not re.fullmatch(r"\d+[a-z]?", str(loc)) for loc in locs):
        return None
    keys = [_locant_key(loc) for loc in locs]
    from ..data.locant_text import render_locants
    desc = f"[{letters}]" if benzo else f"[{render_locants(locs)}-{letters}]"
    return (tuple(sorted(letters)), letters, tuple(sorted(keys)), tuple(keys)), desc


def _covers(components, rings):
    """Pairs of components that split the rings of the system between them: components share fusion atoms, never
    a ring."""
    for p, q in itertools.permutations(components, 2):
        if p.rings & q.rings or p.rings | q.rings != rings:
            continue
        shared = p.atoms & q.atoms
        if len(shared) < 2:
            continue
        yield p, q, shared


def _parent_key(mol, comp):
    """: criteria (a)-(g) of ``_rank``; then (h) lower locants for the heteroatoms and (i) for them in the
    order of seniority, read from the component's own numbering."""
    from .fused_ring_selection import _HETEROATOM_SENIORITY, _rank
    r = _rank(mol, set(comp.atoms))
    het = [a for a in comp.atoms if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    h = min(tuple(sorted(_locant_key(n[a]) for a in het)) for n in comp.numberings)
    order = sorted(het, key=lambda a: -_HETEROATOM_SENIORITY.get(mol.GetAtomWithIdx(a).GetSymbol(), 0))
    i = min(tuple(_locant_key(n[a]) for a in order) for n in comp.numberings)
    return (r.senior_het_neg, r.ring_count_neg, r.ring_sizes_neg, r.het_count_neg, r.het_variety_neg,
            r.alt_het_tuple, r.orient_stub, h, i)


def senior_component_ring_count(mol, rings) -> int:
    """The number of rings of the senior component of the fused system made of ``rings``: a producer
    that builds its name on a smaller parent ('naphtho[1,2-b]quinoline' where acridine is a component) has not built
    the PIN."""
    rings = frozenset(frozenset(r) for r in rings)
    components = [c for c in _catalogue_components(mol, rings) + _monocycle_components(mol, rings) if not c.benzo]
    if not components:
        return 0
    return len(min(components, key=lambda c: _parent_key(mol, c)).rings)


def _named_from(mol, components, rings, keys) -> Optional[str]:
    """The name on the senior parent among ``components`` (``keys``: the key of each, by id)."""
    parents = [c for c in components if not c.benzo]
    if not parents:
        return None
    best = min(keys[id(c)] for c in parents)
    options = []
    for parent, attached, shared in _covers(components, rings):
        if parent.benzo or keys[id(parent)] != best:
            continue
        prefix = 'benzo' if attached.benzo else fusion_attached_prefix(attached.name)
        if not prefix:
            continue
        for pn in parent.numberings:
            for an in attached.numberings:
                built = _descriptor(mol, parent, pn, an, shared, attached.benzo)
                if built is not None:
                    options.append((built[0], f"{prefix}{built[1]}{fusion_component_name(parent.name)}"))
    if not options:
        return None
    low = min(key for key, _name in options)
    names = {name for key, name in options if key == low}
    return names.pop() if len(names) == 1 else None    # a tie on every criterion here: not decided


def peri_fusion_name(mol) -> Optional[str]:
    """The fusion name of a molecule on a peri-fused ring system of two first-order components, checked by OPSIN,
    else None. The ring system is named by fusion nomenclature; its hydrogen, lambda atoms and substituents are added
    on the numbering of (``compute_fused_numbering(allow_peri=True)``); a name that would cite an interior
    locant is not built."""
    scope = _in_scope(mol)
    if scope is None:
        return None
    atoms, rings = scope
    if _ester_outside(mol, atoms):
        return None                        # an ester is senior to the ring's own suffix: not this parent
    components = _catalogue_components(mol, rings) + _monocycle_components(mol, rings)
    keys = {id(c): _parent_key(mol, c) for c in components if not c.benzo}
    # (:11815): a benzo name is formed only from an isolated benzo component, so the retained names and
    # the monocycles are tried first and the benzo names only when they leave more than two components
    plain = [c for c in components if not c.benzo_name]
    parent = _named_from(mol, plain, rings, keys) or _named_from(mol, components, rings, keys)
    if not parent:
        return None
    from .fused_rings import _mancude_indicated_h_count
    if (mol.GetNumHeavyAtoms() == len(atoms) and _mancude_indicated_h_count(mol, set(atoms)) == 0
            and _reads_back(mol, parent)):
        # the bare mancude parent that takes no indicated hydrogen; a parent that takes it is named below with its
        # hydrogen cited, since OPSIN reads a bare name with its own default hydrogen
        return parent
    name = _decorated(mol, set(atoms), parent)
    if not name or '^' in name:
        return None                        # interior locants are not read by OPSIN 2.9.0
    return name if _reads_back(mol, name) else None


#: an ester whose acyl carbon is outside the ring system: '-oate' outranks the ring ketone or imide
_ESTER = Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')


def _ester_outside(mol, atoms) -> bool:
    return any(m[0] not in atoms for m in mol.GetSubstructMatches(_ESTER))


def _decorated(mol, atoms, parent) -> Optional[str]:
    from .fused_name_decoration import decorate_fused_name
    from .fused_rings import (
        _exocyclic_atoms_accounted,
        _fused_locant_to_output,
        _select_substituent_numbering,
    )
    from .fusion_numbering import compute_fused_numbering
    substituted = mol.GetNumHeavyAtoms() != len(atoms)
    if substituted:
        if not _exocyclic_atoms_accounted(mol, atoms):
            return None
        selected = _select_substituent_numbering(mol, atoms, allow_peri=True)
        if selected is None:
            return None
        atom_to_locant, substituents = selected
    else:
        numbering = compute_fused_numbering(mol, atoms, allow_peri=True)
        if not numbering:
            return None
        atom_to_locant, substituents = {a: _fused_locant_to_output(v) for a, v in numbering.items()}, None
    return decorate_fused_name(mol, atoms, parent, atom_to_locant, substituents, substituted)


def _reads_back(mol, name: str) -> bool:
    from ..validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse(name)
    parsed = Chem.MolFromSmiles(smiles) if smiles else None
    if parsed is None:
        return False
    return Chem.MolToInchiKey(parsed) == Chem.MolToInchiKey(mol)
