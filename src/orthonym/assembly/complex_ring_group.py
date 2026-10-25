"""A spiro ring system that stands beside other ring systems in the molecule.

The complex-ring namers (``rules.spiro``) read the whole molecule as one ring system: a ring
substituent next to the spiro system adds a ring to the count of rings, a ring-ring bond to the
ring graph, and none of them recognises the system any more (``phenyl`` on 9,9'-spirobi[fluorene]
was named as a fluorene with two substituents and then declined). and treat the
spiro system as ONE ring system, and a ring substituent of it as a substituent: " The
senior ring or ring system has the greater number of rings" (the Blue Book), the system
ranked by " Polycyclic systems... (a) spiro ring system" (:19536).

``senior_spiro_core`` finds the spiro system that is strictly senior to every other ring system of
the molecule, ``rules.ring_selection.ring_system_score`` of each ring-bond-connected group
of ring atoms) and cuts it out of the molecule with its ring atoms only. The complex-ring namers
then name that stand-alone ring system as they always did, and the result is carried back to the
atoms of the molecule: the substituents, the ring ones included, are named and cited on that
numbering by the caller (``composer._enrich_complex_ring_with_subs``).

Where the numbering of the system has symmetry, the numbering that cites the lowest locants to
the substituents all together is taken "Lowest set of locants", the Blue Book;
 (f),:3301), and a tie between numberings that place different substituents differently is
left to the criteria below it (g) alphanumerical order of the names), which this module
does not decide: it declines, so the molecule keeps the route it had.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

from rdkit import Chem


class SpiroCore(NamedTuple):
    """The ring-only molecule of a spiro system and the molecule atom of each of its atoms."""
    mol: Chem.Mol
    to_full: Tuple[int, ...]
    atoms: FrozenSet[int]


def ring_groups(mol) -> List[FrozenSet[int]]:
    """The ring-bond-connected groups of ring atoms: fused, bridged and spiro systems each one group.
    A single bond between two ring atoms of different systems (a ring substituent, a link of a ring
    assembly) is no ring bond and joins nothing."""
    adjacency: Dict[int, Set[int]] = {}
    for bond in mol.GetBonds():
        if bond.IsInRing():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
    seen: Set[int] = set()
    groups: List[FrozenSet[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        group = {start}
        stack = [start]
        seen.add(start)
        while stack:
            for nbr in adjacency[stack.pop()]:
                if nbr not in seen:
                    seen.add(nbr)
                    group.add(nbr)
                    stack.append(nbr)
        groups.append(frozenset(group))
    return groups


def _has_spiro_atom(mol, group: FrozenSet[int]) -> bool:
    from ..perception.rings import get_spiro_atoms
    return bool(get_spiro_atoms(mol) & group)


def _ring_only_molecule(mol, group: FrozenSet[int]):
    rw = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - set(group), reverse=True):
        rw.RemoveAtom(idx)
    core = rw.GetMol()
    try:
        Chem.SanitizeMol(core)
    except Exception:  # noqa: BLE001 -- a group that cannot stand alone is not cut out
        return None
    return core


def _principal_group_declines(mol, features, best, groups) -> bool:
    """True when a principal characteristic group is on a ring system of the molecule other than as the
    acyclic substituent of ``best``'s ring atoms only: on another ring system (principal
    characteristic group) outranks the ring seniority of and the parent is that ring system; on
    ``best`` itself the group is the suffix of the parent, which this module does not build (the
    writers of the whole molecule and the general engine do, ``tier_a_ring._spiro_principal_suffix_
    preference``), so such a molecule keeps the names they give."""
    pg_atoms = getattr(features, 'principal_group_atoms', None)
    if not pg_atoms:
        return False
    from ..rules.parent_selection import is_principal_group_on_ring
    pg = getattr(features, 'principal_group', None)
    return any(is_principal_group_on_ring(mol, set(g), pg_atoms, pg) for g in groups)


def _ring_on_chain_branch(mol, group: FrozenSet[int]) -> bool:
    """True when a branch of ``group`` that holds a ring is rooted at a non-ring atom (a ring on a
    chain, '(sulfolanyl)carbamoyl'): the complex-ring writers of the whole molecule and the general
    engine keep those, this module names the ring substituents bonded to the system."""
    for items in _attachments(mol, group).values():
        for root, branch, _sig in items:
            if not mol.GetAtomWithIdx(root).IsInRing() and any(
                    mol.GetAtomWithIdx(a).IsInRing() for a in branch):
                return True
    return False


def _group_signature(mol, group: FrozenSet[int]) -> str:
    """The canonical SMILES of the ring group, hydrogens and charges left out and the bond types kept: two
    groups of one skeleton (benzimidazole with a nitrogen bonded onward and with an N-H) give one string,
    benzene and cyclohexane two."""
    flat = Chem.RWMol(mol)
    for atom in flat.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)
    return Chem.MolFragmentToSmiles(flat, atomsToUse=sorted(group), isomericSmiles=False)


def assembly_units(mol, groups: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
    """The ring groups, with the identical groups that are bonded to each other directly merged into one
    unit: a ring assembly, '[1,1'-biphenyl]', '2,2'-binaphthalene') stands in the seniority of ring
    systems as one system whose rings and skeletal atoms are the sums of its components
    "Seniority criteria for ring assemblies are based on the appropriate criteria for rings and ring
    systems given in ", the Blue Book)."""
    owner = {a: i for i, g in enumerate(groups) for a in g}
    signature = [_group_signature(mol, g) for g in groups]
    root = list(range(len(groups)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for bond in mol.GetBonds():
        i, j = owner.get(bond.GetBeginAtomIdx()), owner.get(bond.GetEndAtomIdx())
        if i is not None and j is not None and i != j and not bond.IsInRing() and signature[i] == signature[j]:
            root[find(i)] = find(j)
    units: Dict[int, Set[int]] = {}
    for i, g in enumerate(groups):
        units.setdefault(find(i), set()).update(g)
    return [frozenset(u) for u in units.values()]


def senior_spiro_core(mol, features=None) -> Optional[SpiroCore]:
    """The spiro system of ``mol`` that is the parent: strictly senior to every other ring
    system of it, a ring assembly counting as one system (``assembly_units``), as a ring-only molecule;
    None when the molecule has one ring system, when the senior one is not a spiro system, when two
    systems tie, when a principal characteristic group is on a ring system
    (``_principal_group_declines``) and when a ring hangs on a chain of the system."""
    from ..perception.rings import get_spiro_atoms
    if not get_spiro_atoms(mol):
        return None
    groups = ring_groups(mol)
    if len(groups) < 2:
        return None
    from ..rules.ring_selection import ring_system_score
    units = assembly_units(mol, groups)
    scored = sorted(((ring_system_score(mol, set(u)), min(u)) for u in units))
    if len(scored) < 2 or scored[0][0] == scored[1][0]:
        return None
    best = next(u for u in units if min(u) == scored[0][1])
    if (best not in groups or not _has_spiro_atom(mol, best)
            or _principal_group_declines(mol, features, best, groups)
            or _ring_on_chain_branch(mol, best)):
        return None
    core = _ring_only_molecule(mol, best)
    if core is None:
        return None
    return SpiroCore(core, tuple(sorted(best)), best)


def _branch_signature(mol, branch: Set[int], root: int) -> str:
    """The canonical SMILES of the branch, its atom ``root`` (the one bonded to the ring) marked: two
    branches of one structure give one string whatever the order of the atoms in the molecule."""
    rw = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - branch, reverse=True):
        rw.RemoveAtom(idx)
    kept = sorted(branch)
    rw.GetAtomWithIdx(kept.index(root)).SetAtomMapNum(1)
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:  # noqa: BLE001 -- the unsanitised graph still tells the branches apart
        pass
    return Chem.MolToSmiles(sub, isomericSmiles=False)


def _attachments(mol, group: FrozenSet[int]) -> Dict[int, List[Tuple[int, Set[int], str]]]:
    """For each ring atom of ``group`` with exocyclic heavy neighbours: ``(neighbour, branch, signature)``
    per neighbour, the branch being the atoms that hang on it (cut at the ring)."""
    out: Dict[int, List[Tuple[int, Set[int], str]]] = {}
    for a in sorted(group):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() in group or nb.GetAtomicNum() <= 1:
                continue
            branch = {nb.GetIdx()}
            stack = [nb.GetIdx()]
            while stack:
                for n2 in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                    if n2.GetIdx() not in group and n2.GetIdx() not in branch:
                        branch.add(n2.GetIdx())
                        stack.append(n2.GetIdx())
            out.setdefault(a, []).append((nb.GetIdx(), branch, _branch_signature(mol, branch, nb.GetIdx())))
    return out


def _first_cited_signature(mol, sigs: Dict[str, Tuple[int, Set[int]]]) -> Optional[str]:
    """The signature of the substituent cited first in the name (alphanumerical order,,
    named as the sort keys of the numbering tie-breaks are (``name_substituent_for_ordering``: the
    same name at every tier, and no trace left in the naming memos); None when one cannot be named."""
    from .naming_utils import alpha_sort_key
    from .substituent_enumerator import name_substituent_for_ordering
    names = {}
    for sig, (root, branch) in sigs.items():
        name = name_substituent_for_ordering(mol, sorted(branch), root)
        if not name:
            return None
        names[sig] = alpha_sort_key(name)
    return min(names, key=names.get)


def lowest_locant_numbering(mol, core: SpiroCore, atom_to_locant: Dict[int, object]
                            ) -> Optional[Dict[int, object]]:
    """``atom_to_locant`` (keys: atoms of the molecule) renumbered by the symmetry of the ring
    system: the lowest locants of the substituents all together (f)), then the lowest locant
    for the substituent cited first (g)); None when numberings that no criterion here
    separates place different substituents differently."""
    from .composer import _primed_locant_sort_key
    full_attach = _attachments(mol, core.atoms)
    to_core = {full: i for i, full in enumerate(core.to_full)}
    attach_core = {to_core[a]: [sig for _, _, sig in items] for a, items in full_attach.items()}
    locants = {to_core[a]: loc for a, loc in atom_to_locant.items() if a in to_core}
    if len(locants) != len(core.to_full):
        return None
    matches = core.mol.GetSubstructMatches(core.mol, uniquify=False, maxMatches=10000,
                                           useChirality=False)
    if not matches:
        return None
    labelled = [{i: locants[m[i]] for i in range(len(m))} for m in matches]

    def key(labels, only=None):
        return sorted(_primed_locant_sort_key(labels[a]) for a, sigs in attach_core.items()
                      for sig in sigs if only is None or sig == only)

    def placement(labels):
        return frozenset((labels[a], tuple(sorted(sigs))) for a, sigs in attach_core.items())

    best_key = min(key(labels) for labels in labelled)
    best = [labels for labels in labelled if key(labels) == best_key]
    if len({placement(b) for b in best}) > 1:
        sigs = {sig: (root, branch) for items in full_attach.values() for root, branch, sig in items}
        first = _first_cited_signature(mol, sigs)
        if first is None:
            return None
        g_key = min(key(labels, first) for labels in best)
        best = [labels for labels in best if key(labels, first) == g_key]
        if len({placement(b) for b in best}) > 1:
            return None
    return {core.to_full[i]: loc for i, loc in best[0].items()}


def is_complex_ring_system_by_group(mol, features=None) -> bool:
    """``composer._is_complex_ring_system`` of the senior spiro system when that is the parent beside
    other ring systems (see the module docstring), of the molecule otherwise."""
    from .composer import _is_complex_ring_system
    core = senior_spiro_core(mol, features)
    if core is not None:
        # asked of the system alone: the classification of the whole molecule takes a ring
        # substituent for a part of a cage and spends the hang budget of the first run on it
        return _is_complex_ring_system(core.mol)
    return _is_complex_ring_system(mol)


def assemble_complex_ring_by_group(mol, features):
    """``composer._assemble_complex_ring_name`` of the senior spiro system, named as a stand-alone
    ring system with its atoms carried back to the atoms of ``mol``, where that system is the parent
    beside other ring systems (see the module docstring); of the molecule otherwise, and where the
    system is not named. The substituents are left to the caller (``substituents_included`` False)."""
    from .composer import ComplexRingResult, _assemble_complex_ring_name
    core = senior_spiro_core(mol, features)
    if core is not None:
        from ..namer import compute_features
        result = _assemble_complex_ring_name(core.mol, compute_features(core.mol))
        if result is not None and result.atom_to_locant:
            mapped = {core.to_full[i]: loc for i, loc in result.atom_to_locant.items()}
            numbering = lowest_locant_numbering(mol, core, mapped)
            if numbering is not None:
                return ComplexRingResult(result.name, {core.to_full[i] for i in result.ring_atoms},
                                         numbering, False)
    return _assemble_complex_ring_name(mol, features)
