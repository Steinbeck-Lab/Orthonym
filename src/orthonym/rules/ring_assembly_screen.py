"""Does this molecule join two identical ring systems by a bond?,; fail closed).

``### **** DEFINITIONS`` (the Blue Book),:15542: "Two or more cyclic systems
(single rings or fused systems, alicyclic von Baeyer systems, spiro systems, phane systems,
fullerenes) that are directly joined to each other by single or double bonds are called 'ring
assemblies' when the number of such direct ring junctions is one less than the number of cyclic
systems involved.";:15550: "Ring assemblies are composed of identical cyclic systems (rings or
ring systems); assemblies of nonidentical cyclic systems (rings or ring systems) are not called
ring assemblies for the purposes of organic nomenclature". ``### **** Ring assemblies
with a single bond junction`` (:15560),:15569: "The name biphenyl is retained as
1,1'-biphenyl."

Identity is that of the ring skeleton: ``## **** UNSATURATION IN RING ASSEMBLIES COMPOSED
OF MONOCYCLIC MANCUDE AND SATURATED RINGS`` (:24151),:24153: "When assemblies of otherwise
identical rings contain both mancude and saturated rings, the use of hydro prefixes is
preferred, except in the case of a two ring assembly consisting of one benzene ring and a
cyclohexane ring.";:24159 '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)' (not
'2-(piperidin-2-yl)pyridine');:24157 'cyclohexylbenzene (PIN)'. ``### **** Ring
assemblies composed of monocyclic components`` (:17077),:17079: "In biphenyl and polyphenyl
assemblies, one benzene ring must remain in the assembly... Furthermore, when a modified ring
assembly of two rings consists of a benzene ring and a cyclohexane ring substitutive
nomenclature is preferred (see.";:17083 '2,3-dihydro-1,1'-biphenyl (PIN)' (not
'(cyclohexa-1,3-dien-1-yl)benzene'). So a pyridine bonded to a piperidine, or a benzene bonded
to a cyclohexene, is a ring assembly named with hydro prefixes; only the two-ring assembly of
one benzene ring and one cyclohexane ring is named substitutively. The junction may be a ring
nitrogen (``****``,:15593; '1,1'-bipyrrole (PIN)':15603, '2H-1,2'-bipyridine (PIN)'
:15634), so a pyridine bonded to the nitrogen of a piperidine is one too
('3,4,5,6-tetrahydro-2H-1,2'-bipyridine').

A ring assembly is a parent of its own (``(g) ring assembly``,:19542) and has more
rings than either of its components ("The senior ring or ring system has the greater number of
rings",:19461), so a principal characteristic group on one component is the suffix
of the assembly: '(1P)-2',5'-dimethoxy-6-nitro[1,1'-biphenyl]-2-carboxylic acid (PIN)'
,:49805), and the book does not take the assembly apart into a phenyl prefix on a
benzene ring ('([1,1'-biphenyl]-4-yl)oxy (preferred prefix) (not 4-phenylphenoxy)',
:24607).
A substitutive name such as '4-phenyl-N-(pyridin-2-yl)benzamide' for
N-(pyridin-2-yl)[1,1'-biphenyl]-4-carboxamide is therefore not the PIN.

The producers of the drug lane do not decide between an assembly parent and a component parent.
They consult this screen, so that no name they build with one component as the parent is
certified as the PIN. True means: the molecule holds two ring systems with the same skeleton
joined by a bond, other than the one two-ring benzene + cyclohexane assembly of, or the
screen could not tell (fail closed).
"""
from __future__ import annotations

from typing import Dict, List, Set

from rdkit import Chem

_MEMO_MOL = None
_MEMO: Dict[str, bool] = {}


def _ring_systems(mol) -> List[Set[int]]:
    """Rings that share an atom (fused, bridged, spiro) merged into one ring system."""
    systems: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        r = set(ring)
        for s in [s for s in systems if s & r]:
            r |= s
            systems.remove(s)
        systems.append(r)
    return systems


def ring_system_key(mol, atoms) -> str:
    """The ring system's skeleton: its elements and connectivity, every bond single and no
    aromatic flag (hydrogens, charges, isotopes and decorations ignored), as a canonical SMILES.
    The mancude and the saturated forms of one ring system share it (pyridine and piperidine,
    benzene and cyclohexene): the "otherwise identical rings" of (:24153)."""
    rw = Chem.RWMol()
    idx = {}
    for a in sorted(atoms):
        new = Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum())
        new.SetNoImplicit(True)
        idx[a] = rw.AddAtom(new)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            rw.AddBond(idx[i], idx[j], Chem.BondType.SINGLE)
    return Chem.MolToSmiles(rw.GetMol(), canonical=True)


def _bonds_within(mol, atoms) -> List:
    return [b for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]


def _lone_six_carbon_ring(mol, atoms) -> bool:
    """A single six-membered ring of carbon atoms (no fused, bridged or spiro partner)."""
    return (len(atoms) == 6 and len(_bonds_within(mol, atoms)) == 6
            and all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in atoms))


def _is_benzene_ring(mol, atoms) -> bool:
    return (_lone_six_carbon_ring(mol, atoms)
            and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms))


def _is_cyclohexane_ring(mol, atoms) -> bool:
    return (_lone_six_carbon_ring(mol, atoms)
            and not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms)
            and all(b.GetBondType() == Chem.BondType.SINGLE
                    for b in _bonds_within(mol, atoms)))


def _benzene_cyclohexane_pair(mol, systems, members) -> bool:
    """The exception (:24153): a two-ring assembly of one benzene ring and one
    cyclohexane ring ('cyclohexylbenzene (PIN)',:24157)."""
    if len(members) != 2:
        return False
    a, b = (systems[k] for k in members)
    return ((_is_benzene_ring(mol, a) and _is_cyclohexane_ring(mol, b))
            or (_is_benzene_ring(mol, b) and _is_cyclohexane_ring(mol, a)))


def _assemblies(mol):
    """``(systems, owner, groups)``: the ring systems, the system index of every ring atom,
    and the groups of two or more systems with the same skeleton joined by a bond (the one
    two-ring benzene + cyclohexane assembly of left out), each a list of system
    indices."""
    systems = _ring_systems(mol)
    if len(systems) < 2:
        return systems, {}, []
    owner = {a: k for k, s in enumerate(systems) for a in s}
    keys: Dict[int, str] = {}
    parent = list(range(len(systems)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for b in mol.GetBonds():
        ki, kj = owner.get(b.GetBeginAtomIdx()), owner.get(b.GetEndAtomIdx())
        if ki is None or kj is None or ki == kj:
            continue
        for k in (ki, kj):
            if k not in keys:
                keys[k] = ring_system_key(mol, systems[k])
        if keys[ki] == keys[kj]:
            parent[find(ki)] = find(kj)
    assemblies: Dict[int, List[int]] = {}
    for k in range(len(systems)):
        assemblies.setdefault(find(k), []).append(k)
    groups = [members for members in assemblies.values()
              if len(members) > 1 and not _benzene_cyclohexane_pair(mol, systems, members)]
    return systems, owner, groups


def _compute(mol) -> bool:
    return bool(_assemblies(mol)[2])


def ring_is_in_biphenyl_assembly(mol, ring_atoms) -> bool:
    """True when the ring system that holds ``ring_atoms`` is joined, through a group of ring
    systems with the same skeleton (a ring assembly,:15542; the two-ring benzene +
    cyclohexane assembly of left out), to ANOTHER benzene ring: a biphenyl or polyphenyl
    assembly. On error True (fail closed).

    This is the question a writer of a SINGLE-ring parent (benzene, ``name_substituted_
    benzene``) has to ask about its own parent ring: where the parent ring is one member of such
    an assembly, the assembly, which has more rings, is the senior parent:19461;
    ``### **** Retained names``:27665,:27722-27724 '4-methoxy-1,1'-biphenyl (PIN)...
    [not 1-methoxy-4-phenylbenzene; the biphenyl ring system is senior to a single benzene
    ring]'), and the single-ring name is not the PIN. An assembly elsewhere in the molecule, that
    does not hold the parent ring, does not change the parent. An assembly of a benzene ring with
    a partly saturated ring (hydro prefixes,:17077) is not this question and is
    left to the writers that decide it. Not memoised: the answer depends on the ring."""
    try:
        systems, owner, groups = _assemblies(mol)
        mine = {owner[a] for a in ring_atoms if a in owner}
        return any(
            (mine & set(members))
            and any(k not in mine and _is_benzene_ring(mol, systems[k]) for k in members)
            for members in groups)
    except Exception:  # noqa: BLE001 - unknown: fail closed
        return True


def joins_identical_ring_systems(mol) -> bool:
    """True when two ring systems of ``mol`` with the same skeleton are joined directly by a
    bond (a ring assembly,; mancude and saturated members alike,, unless the
    assembly is the two-ring benzene + cyclohexane one that names substitutively, or on
    error (fail closed). Memoised per molecule object."""
    global _MEMO_MOL, _MEMO
    if _MEMO_MOL is not mol:
        _MEMO_MOL, _MEMO = mol, {}
    if "v" not in _MEMO:
        try:
            _MEMO["v"] = _compute(mol)
        except Exception:  # noqa: BLE001 - unknown: fail closed
            _MEMO["v"] = True
    return _MEMO["v"]
