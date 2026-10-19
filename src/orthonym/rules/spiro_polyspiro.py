""" unbranched polyspiro ring systems with different components, one of them polycyclic.

The chain builder behind ``spiro.name_unbranched_polyspiro_different``: the components are named one
by one (``spiro._name_spiro_component``), cited in alphanumerical order and joined by the spiro
locants. It lives in its own module because ``spiro.py`` is over the size budget.

Governing rules (the Blue Book Blue Book):
* 'Unbranched polyspiro ring systems with different ring components one being a polycyclic
  ring system' (:10705): "Unbranched polyspiro ring systems with at least two different ring
  components and at least one of which is a polycyclic ring are named by placing the ring component
  names in order of their occurrence in the structure beginning with the terminal ring component
  lower in alphabetical order and enclosing within square brackets."
* (:10719): "If there is a choice of locants, the lowest set of locants for all spiro atoms
  is selected when compared in increasing numerical order and, if there is still a choice, in order
  of their citation in the name".
* NUMBERING (:3219), (f) "detachable alphabetized prefixes, all considered together in a
  series of increasing numerical order" (:3301): after the spiro atoms, a substituent takes the
  lowest locant its ring allows, whatever the atom order of the input.

A SUBSTITUTED system (``allow_substituents``): the bare ring system is named as for the unsubstituted
one and its atom -> locant map is returned; the caller attaches the prefixes to that map. A ring that
touches no spiro atom hangs off the chain by an exocyclic bond and is a substituent, not a component.
The map is the one the descriptor cites: a fused component is re-anchored to its descriptor locant
(``spiro._reanchor_locmap_to_canonical_spiro``), a monocycle component takes the substituent
tie-break of its own namer, and two terminals with one name are ordered by
``spiro._terminal_orientation_key``. Only the best-effort tier asks for it: the PIN path keeps the
unsubstituted-only reading (the composer's ``is_unbranched_polyspiro_different``).
"""
from typing import Dict, Optional, Set

from rdkit import Chem

from ..perception.molcache import atoms_of
from ..perception.rings import get_spiro_atoms
from . import spiro as _sp


def ring_blocks(mol):
    """The ring components of ``mol`` read from its ring BONDS, not from the smallest set of smallest
    rings: the biconnected components of the graph of ring atoms and ring bonds, as
    ``[(atoms, spiro_atoms, ring_count)]``. A spiro atom is an atom shared by two components. RDKit
    drops one of the equal faces of a cage (adamantane keeps 3 of its 4 six-membered rings, which
    ones depends on the atom order), so an atom of a cage that is a spiro atom can sit in 2 or in 3 of
    the rings it lists; the components do not change with the choice."""
    adj: Dict[int, Set[int]] = {}
    for b in mol.GetBonds():
        if b.IsInRing():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj.setdefault(i, set()).add(j)
            adj.setdefault(j, set()).add(i)
    if len(adj) > 500:  # the walk below recurses once per ring atom
        return []
    order: Dict[int, int] = {}
    low: Dict[int, int] = {}
    edges: list = []
    blocks: list = []

    def visit(v: int, parent: Optional[int]) -> None:
        order[v] = low[v] = len(order)
        for w in sorted(adj[v]):
            if w == parent:
                continue
            if w not in order:
                edges.append((v, w))
                visit(w, v)
                low[v] = min(low[v], low[w])
                if low[w] >= order[v]:
                    block = []
                    while True:
                        e = edges.pop()
                        block.append(e)
                        if e == (v, w):
                            break
                    blocks.append(block)
            elif order[w] < order[v]:
                edges.append((v, w))
                low[v] = min(low[v], order[w])

    for v in sorted(adj):
        if v not in order:
            visit(v, None)
    atom_sets = [{a for e in blk for a in e} for blk in blocks]
    count: Dict[int, int] = {}
    for atoms in atom_sets:
        for a in atoms:
            count[a] = count.get(a, 0) + 1
    return [(atoms, {a for a in atoms if count[a] > 1}, len(blk) - len(atoms) + 1)
            for atoms, blk in zip(atom_sets, blocks)]


def name_polyspiro_different_core(mol, *, allow_substituents: bool = False,
                                  restrict_atoms: Optional[Set[int]] = None):
    """``(name, all_ring_atoms, combined_locants)`` of the linear terminal-middle-terminal chain, or
    None. ``combined_locants`` maps each ring atom of the chain to its locant (an int for the first
    cited component, ``(int, "'")`` and ``(int, "''")`` for the next two). Fails closed unless the
    system is a 3-component, 2-spiro-atom chain with a polycyclic component and not all components
    identical (that is dispiroter). ``restrict_atoms``: only the ring components inside this atom set
    count (a molecule can hold several ring systems; the floor names them one at a time)."""
    ri = mol.GetRingInfo()
    if allow_substituents:
        # the components come from the ring bonds; a ring with no spiro atom hangs off the chain by an
        # exocyclic bond and is a substituent ring, not a component
        blocks = [b for b in ring_blocks(mol) if restrict_atoms is None or b[0] <= restrict_atoms]
        if sum(len(b[1]) for b in blocks) != 4 or any(len(b[1]) > 2 for b in blocks):
            return None  # two spiro atoms, each shared by two components
        comps = [(atoms, spiros, n) for atoms, spiros, n in blocks if spiros]
    else:  # the PIN path: the ring perception as before, unsubstituted only
        if len(get_spiro_atoms(mol)) != 2:
            return None
        ring_atoms = {a for r in ri.AtomRings() for a in r}
        if any(a.GetAtomicNum() > 1 and a.GetIdx() not in ring_atoms for a in atoms_of(mol)):
            return None
        ring_sets = [set(r) for r in ri.AtomRings()]
        comps = [(atoms, spiros, sum(1 for r in ring_sets if r <= atoms))
                 for atoms, spiros in _sp._fused_ring_components(mol)]
    if len(comps) != 3:
        return None
    all_ring_atoms: Set[int] = set()
    for atoms, _s, _n in comps:
        all_ring_atoms |= atoms
    # scope: >=1 component must be polycyclic (>=2 fused rings); a pure spiro of monocycles
    # uses the von Baeyer descriptor form (dispiro[a.b.c.d]) via name_spiro_system.
    if not any(n >= 2 for _a, _s, n in comps):
        return None
    middle = None
    terminals = []
    for atoms, spiros, _n in comps:
        if len(spiros) == 2:
            if middle is not None:
                return None  # >1 two-spiro component -> branched
            middle = (atoms, sorted(spiros))
        elif len(spiros) == 1:
            terminals.append((atoms, next(iter(spiros))))
        else:
            return None
    if middle is None or len(terminals) != 2:
        return None

    # Name each terminal + spiro locant.
    term_info = []
    for atoms, sa in terminals:
        named = _sp._name_spiro_component(mol, atoms, sa, substituent_tiebreak=True)
        if named is None:
            return None
        ext = _sp._extract_subfragment(mol, atoms)
        if ext is None:
            return None
        loc = _sp._canonical_spiro_locant(ext, named[1], sa)
        if loc is None:
            return None
        locmap = named[1]
        if allow_substituents and 'cyclo[' not in named[0]:
            # the map must agree with the descriptor locant, and a symmetric component must hand
            # its lowest locants to the substituents whatever the atom order (spiro-hoist)
            locmap = _sp._reanchor_locmap_to_canonical_spiro(mol, ext, locmap, sa, loc)
        term_info.append({'atoms': atoms, 'spiro': sa, 'name': named[0], 'loc': loc, 'map': locmap})

    # alphanumerical: the first-cited (unprimed) terminal is the lower key; two terminals
    # with one name are ordered by spiro locant, substituent locants and canonical rank.
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    key = [(_sp._component_alpha_key(t['name']),
            _sp._terminal_orientation_key(mol, t['atoms'], t['map'], t['spiro'], ranks))
           for t in term_info]
    first, last = (term_info[0], term_info[1]) if key[0] <= key[1] else (term_info[1], term_info[0])

    # NOT all-identical (else dispiroter owns it).
    if first['name'] == last['name']:
        mid_named = _sp._name_spiro_component(mol, middle[0], middle[1][0])
        if mid_named is not None and mid_named[0] == first['name']:
            return None

    # Middle component numbered so the FIRST (unprimed) terminal's junction spiro atom is locant 1;
    # the other spiro atom takes its walk position.
    mid_named = _sp._name_spiro_component(
        mol, middle[0], first['spiro'], substituent_tiebreak=True,
        other_spiro_atoms=(last['spiro'],) if allow_substituents else None)
    if mid_named is None:
        return None
    mid_name, mid_map = mid_named
    mid_loc_first = mid_map.get(first['spiro'])
    mid_loc_last = mid_map.get(last['spiro'])
    if mid_loc_first is None or mid_loc_last is None:
        return None

    #: the combined indicated-H + lambda front-of-name prefix (empty for a plain,
    # standard-valence system); the descriptor then cites the components without their leading
    # indicated-H.
    front_prefix, _stripped = _sp._build_lambda_ih_front_prefix(mol, [
        {'id': 'first', 'name': first['name'], 'prime': '',
         'spiro_atoms': [first['spiro']], 'map': first['map']},
        {'id': 'mid', 'name': mid_name, 'prime': "'",
         'spiro_atoms': [first['spiro'], last['spiro']], 'map': mid_map},
        {'id': 'last', 'name': last['name'], 'prime': "''",
         'spiro_atoms': [last['spiro']], 'map': last['map']},
    ])
    if front_prefix:
        first_name = _sp._extract_leading_indicated_h(first['name'])[1]
        mid_name_s = _sp._extract_leading_indicated_h(mid_name)[1]
        last_name = _sp._extract_leading_indicated_h(last['name'])[1]
    else:
        first_name = _sp._strip_consumed_indicated_h(first['name'], first['loc'])
        mid_name_s = _sp._strip_consumed_indicated_h(mid_name, mid_loc_first)
        last_name = _sp._strip_consumed_indicated_h(last['name'], last['loc'])
    name = (f"{front_prefix}dispiro[{first_name}-{first['loc']},{mid_loc_first}'-"
            f"{mid_name_s}-{mid_loc_last}',{last['loc']}''-{last_name}]")

    combined: Dict[int, _sp._Locant] = {}
    for a, loc in first['map'].items():
        combined[a] = loc
    for a, loc in mid_map.items():
        if a not in combined:
            combined[a] = (loc, "'")
    for a, loc in last['map'].items():
        if a not in combined:
            combined[a] = (loc, "''")
    if not set(combined) >= all_ring_atoms:
        return None
    return name, all_ring_atoms, combined
