"""The structural half of the ``name_substituent`` memo key, kept for the naming scope (lane/l17 item 13).

``substituent_enumerator._substituent_memo_key`` runs for every ``name_substituent`` call, hit or miss (15,177 calls
on 200 PIN-tier rows, 135,429 on the 15 slowest; 6.7% of the PIN-tier time of the first and 12% of the second), and
half of its calls repeat an earlier call on the same molecule object and the same fragment. Most of its time was
RDKit: ``MolFragmentToSmiles`` of the fragment, the walks over the bonds of the fragment atoms and the parse of
the output order.

What the key holds that is a function of the molecule's graph, the fragment and the attachment atom alone is kept
under ``molcache.cached_by_key`` for the scope: the rooted fragment SMILES, its output order (and the two properties
RDKit leaves on the molecule), the external bond orders and the free valence. The entry is stamped with what an
in-place edit of the molecule could change and the graph does not hold: the graph key (``memo_keys.mol_graph_key``,
which the key holds anyway), and the ``_CIPCode`` of the fragment atoms and bonds (the canonical ranking of the
fragment reads it). A hit puts the two output-order properties back as the ``MolFragmentToSmiles`` call it replaces
would have left them. ``ORTHONYM_MOLCACHE=verify`` recomputes every hit and raises on a difference; ``off`` and a
call outside a naming scope (or on an ``RWMol``) run the unrestricted code, which is the key as it was.

The key is the same tuple as before, component for component; the frozen reference of
``tests/unit/assembly/test_substituent_memo_key_equal.py`` compares it.
"""
import re

from rdkit import Chem

from ..perception.molcache import atoms_of, bonds_of, cached_by_key, caches
from .memo_keys import mol_graph_key as _mol_graph_key

_ORDER_INT_RE = re.compile(r"\d+")   # RDKit's '[0,1,2,]' output-order property

_NS_TOPOLOGY = "fragment_key.topology"
_NS_PARTS = "fragment_key.parts"


def _cip(item):
    return item.GetProp('_CIPCode') if item.HasProp('_CIPCode') else None


def _topology(mol, graph_key):
    """Per atom ``(bond index, other atom index, bond order as a float)`` of each of its bonds, the two ends of every bond,
    the number of fragments of the molecule, and the graph key the entry was made for, for the scope (stamped with the
    graph key). The key it hands back is the entry's own object: every key of the molecule's fragments holds that one
    ``bytes`` (a key per distinct fragment held a copy of its own)."""
    def fresh():
        atoms = atoms_of(mol)
        adjacency = tuple(
            tuple((b.GetIdx(), b.GetOtherAtomIdx(a.GetIdx()), b.GetBondTypeAsDouble()) for b in a.GetBonds())
            for a in atoms)
        ends = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds_of(mol))
        return adjacency, ends, len(Chem.GetMolFrags(mol)), graph_key
    return cached_by_key(mol, _NS_TOPOLOGY, graph_key, fresh)


def _fragment_parts(mol, adjacency, frag_set, frag_bonds, attach_idx, rooted):
    """What ``MolFragmentToSmiles`` and the walks over the fragment atoms' bonds give: the SMILES, the two output-order
    properties it leaves on the molecule (a string, or None when RDKit left none), the atom output order and its rank
    map, the external bond orders in that order and the free valence at the attachment atom."""
    smi = Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(frag_set), bondsToUse=frag_bonds, rootedAtAtom=rooted,
        isomericSmiles=True, canonical=True)
    if attach_idx < 0 or attach_idx >= len(adjacency):       # GetAtomWithIdx refuses what a tuple would take
        raise IndexError(attach_idx)
    ext_free_valence = sum(int(bt) for _b, other, bt in adjacency[attach_idx] if other not in frag_set)
    atom_order = mol.GetProp('_smilesAtomOutputOrder') if mol.HasProp('_smilesAtomOutputOrder') else None
    bond_order = mol.GetProp('_smilesBondOutputOrder') if mol.HasProp('_smilesBondOutputOrder') else None
    if atom_order is None:
        return smi, None, bond_order, None, None, None, ext_free_valence
    order = tuple(int(x) for x in _ORDER_INT_RE.findall(atom_order))
    ext_bond_orders = tuple(
        tuple(sorted(bt for _b, other, bt in adjacency[i] if other not in frag_set)) for i in order)
    return (smi, atom_order, bond_order, order, {idx: pos for pos, idx in enumerate(order)}, ext_bond_orders,
            ext_free_valence)


def structural_parts(mol, frag_atoms, attach_idx):
    """``(smi, attach_in_frag, ext_free_valence, atom_cips, bond_cips, ext_bond_orders, graph_key)`` of the key, or None
    when RDKit left no atom output order (the key then skips the cache), for a molecule that is cached in the scope
    (``molcache.caches``)."""
    adjacency, ends, n_frags, graph_key = _topology(mol, _mol_graph_key(mol))
    frag_set = set(frag_atoms)
    attach_in_frag = attach_idx in frag_set
    # a mixture named as one graph: RDKit refuses a fragment rooted at an atom of a molecule with more than one
    # fragment, so it gets the fragment SMILES unrooted and the attachment atom's place in the canonical atom order
    multi_fragment = attach_in_frag and n_frags > 1
    rooted = attach_idx if attach_in_frag and not multi_fragment else -1
    frag_bonds = sorted({b for i in frag_set for b, other, _bt in adjacency[i] if other in frag_set})
    atoms, bonds = atoms_of(mol), bonds_of(mol)
    atom_cip = {i: _cip(atoms[i]) for i in frag_set}
    bond_cip = {b: _cip(bonds[b]) for b in frag_bonds}
    stamp = (frozenset(frag_set), attach_idx, rooted, graph_key,
             tuple(atom_cip[i] for i in sorted(frag_set)), tuple(bond_cip[b] for b in frag_bonds))
    parts = cached_by_key(
        mol, _NS_PARTS, stamp,
        lambda: _fragment_parts(mol, adjacency, frag_set, frag_bonds, attach_idx, rooted))
    smi, atom_order, bond_order, order, rank, ext_bond_orders, ext_free_valence = parts
    # the two properties RDKit leaves on the molecule, as the call that a hit replaces would leave them
    if atom_order is not None:
        mol.SetProp('_smilesAtomOutputOrder', atom_order, computed=True)
    if bond_order is not None:
        mol.SetProp('_smilesBondOutputOrder', bond_order, computed=True)
    if atom_order is None:
        return None
    atom_cips = tuple(atom_cip[i] if i in atom_cip else _cip(atoms[i]) for i in order)
    if multi_fragment:
        smi = ("multi-fragment", smi, rank.get(attach_idx, -1))
    items = [((rank.get(ends[b][0], -1), rank.get(ends[b][1], -1)), bond_cip[b]) for b in frag_bonds]
    items.sort(key=lambda t: t[0])
    return (smi, attach_in_frag, ext_free_valence, atom_cips, tuple(c for _, c in items), ext_bond_orders,
            graph_key)


def structural_parts_unrestricted(mol, frag_atoms, attach_idx):
    """:func:`structural_parts` with no cache: the key's structural half as it was computed before the cache."""
    attach_in_frag = attach_idx in frag_atoms
    multi_fragment = attach_in_frag and len(Chem.GetMolFrags(mol)) > 1
    frag_set = set(frag_atoms)
    _atoms = atoms_of(mol)
    # Own bonds go to RDKit as ``bondsToUse`` (left to find them, a rooted call walks every bond: 1.7 vs 0.6 ms at
    # 404 atoms, same string); the bond CIP stamp below takes them in bond-index order. With them RDKit skips its
    # single-fragment check of a rooted call, so ``multi_fragment`` must keep a mixture unrooted (-1).
    _frag_bonds = {}
    for _i in frag_set:
        for _b in _atoms[_i].GetBonds():
            if _b.GetOtherAtomIdx(_i) in frag_set:
                _frag_bonds[_b.GetIdx()] = _b
    smi = Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(frag_atoms), bondsToUse=sorted(_frag_bonds),
        rootedAtAtom=(attach_idx if attach_in_frag and not multi_fragment else -1),
        isomericSmiles=True, canonical=True)
    _a = mol.GetAtomWithIdx(attach_idx)
    ext_free_valence = sum(
        int(b.GetBondTypeAsDouble()) for b in _a.GetBonds()
        if b.GetOtherAtom(_a).GetIdx() not in frag_set)
    if not mol.HasProp('_smilesAtomOutputOrder'):
        return None
    _order = [int(x) for x in _ORDER_INT_RE.findall(mol.GetProp('_smilesAtomOutputOrder'))]
    atom_cips = tuple((_atoms[i].GetProp('_CIPCode') if _atoms[i].HasProp('_CIPCode') else None) for i in _order)
    _rank = {idx: pos for pos, idx in enumerate(_order)}
    if multi_fragment:
        smi = ("multi-fragment", smi, _rank.get(attach_idx, -1))
    _bond_items = [((_rank.get(_b.GetBeginAtomIdx(), -1), _rank.get(_b.GetEndAtomIdx(), -1)),
                    _b.GetProp('_CIPCode') if _b.HasProp('_CIPCode') else None)
                   for _b in map(_frag_bonds.get, sorted(_frag_bonds))]
    _bond_items.sort(key=lambda t: t[0])
    bond_cips = tuple(c for _, c in _bond_items)
    # every fragment atom's external-bond-order multiset, in the canonical fragment-SMILES output order
    ext_bond_orders = tuple(
        tuple(sorted(
            b.GetBondTypeAsDouble()
            for b in _atoms[i].GetBonds()
            if b.GetOtherAtomIdx(i) not in frag_set))
        for i in _order)
    return (smi, attach_in_frag, ext_free_valence, atom_cips, bond_cips, ext_bond_orders, _mol_graph_key(mol))


def key_parts(mol, frag_atoms, attach_idx):
    """The structural half of the ``name_substituent`` memo key (see:func:`structural_parts`): from the scope's cache
    where the molecule is cached there, else computed as it always was."""
    if caches(mol):
        return structural_parts(mol, frag_atoms, attach_idx)
    return structural_parts_unrestricted(mol, frag_atoms, attach_idx)
