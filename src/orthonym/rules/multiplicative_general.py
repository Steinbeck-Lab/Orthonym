"""General multiplicative detector,.

One graph-level detector for "identical parent structures linked by a di- or
polyvalent group" that the narrow shape handlers of ``rules/multiplicative.py``
do not claim. It runs after them (``name_multiplicative``).

Blue Book (``the Blue Book``):

* "Preferred IUPAC multiplicative names" (:23178): "Multiplicative
  nomenclature is preferred to substitutive nomenclature for generating preferred
  IUPAC names to express multiple occurrences of identical parent structures,
  other than alkanes when (1) the linking bonds (single or multiple) between the
  central substituent group of the multiplicative group and all subsequent
  structural units are identical and (2) the multiplicative groups, other than the
  central multiplicative group, are symmetrically substituted; and (3) the locants
  of all substituent groups on the identical parent structures, including suffix
  groups, are identical" (:23180-23185).
* (:23246): "A maximum number of identical parent structures must be
  expressed by the multiplicative name."
* (:23242): phane names when four or more rings, two of them terminal,
  form a linear system of at least seven nodes (2),:23829);
  skeletal replacement when its conditions are met,:23348: "four or more
  heterounits... in an acyclic chain").
* (:23309) /: the seniority of classes decides between a
  parent structure and a component of the multiplicative group: every principal
  characteristic group lies in a parent unit here.
* (:5238): a concatenated group cites "first... the central
  multiplicative substituent group, followed by a multiplicative prefix... and
  then, in order, and in the direction toward the identical parent structures,
  the names of successive di- or polyvalent substituent groups"; Note (:5263) "not
  prefixed by separate multiplicative prefixes".
* (:5285): "the sequence of atoms and bonds, starting at the
  central substituent group, are identical in each of the branches".
* (:5297): "lowest locants to the atoms that are at the end of the
  component nearest to the multiplied parent structure, except where the
  component has a fixed numbering. The locants attached to the multiplied parent
  structure are cited last. When there is no choice the locants are cited in
  increasing numerical order."
* (:5776): "The numbering of the identical parent structural unit is
  retained and, when there is a choice, the locants of the point of substitution
  by the linking multiplicative substituent groups on the identical parent
  structure are as low as possible"; "(the locant 1 is omitted when alone in the
  name of a mononuclear parent hydride)".
* (:6176): substituted identical units "are treated as a compound or
  complex group, enclosed in parentheses, square brackets, or braces... and
  designated by the appropriate numerical prefix 'bis', 'tris', 'tetrakis'".
* (:6295): "Unsymmetrical central multiplicative substituent groups are
  allowed if they are formed from a multivalent substituent group to which
  subsequent groups are attached by identical bonds".
* (:18917): "N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In >
  Tl > O > S > Se > Te > C... applied to select the senior atom in parents and to
  choose between rings and chains"; rings are then chosen by.
* (:31794): esters of a multiplied acid component cite "all
  organyl components... in front of the name of the multiplied acid component";
  'dimethyl 3,3'-oxydibenzoate (PIN)', 'ethyl methyl 3,3'-oxydibenzoate (PIN)'
  (:31801,:31803).

Scope (everything else declines, so the narrow handlers and the substitutive path
keep the molecule): neutral (nitro groups aside), no isotopes, no E/Z, monocyclic
ring systems only, saturated or mancude rings that need no indicated hydrogen,
unbranched chain parents, at most one principal characteristic group per unit, a
linker made of rings, unbranched carbon chains, chalcogen atoms or chains, -NH-,
-N< and =N-N=, with no substituents on the linker and no stereocentre in it.
"""
from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

_ALLOWED_Z = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 33, 34, 35, 50, 51,
                        52, 53, 82, 83})
_HALO = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
# /: low locants to ring heteroatoms as a set, then in the
# order O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B.
_HW_ORDER = (8, 16, 34, 52, 7, 15, 33, 51, 83, 14, 32, 50, 82, 5)
_RING_ELEMENTS = frozenset({6, 7, 8, 16})
# mononuclear parent hydrides with their standard bonding number
_HYDRIDE_NAMES = {14: ("silane", 4), 32: ("germane", 4), 50: ("stannane", 4),
                  82: ("plumbane", 4), 15: ("phosphane", 3), 33: ("arsane", 3),
                  51: ("stibane", 3), 83: ("bismuthane", 3), 5: ("borane", 3),
                  16: ("sulfane", 2), 34: ("selane", 2), 52: ("tellane", 2)}
_CHALCOGEN_COMPONENT = {(8, 1): "oxy", (8, 2): "peroxy", (16, 1): "sulfanediyl",
                        (16, 2): "disulfanediyl", (16, 3): "trisulfanediyl",
                        (34, 1): "selanediyl", (34, 2): "diselanediyl",
                        (52, 1): "tellanediyl"}
_VALENCE_WORD = {2: "diyl", 3: "triyl", 4: "tetrayl"}
# principal characteristic groups this detector expresses as a suffix
_RING_SUFFIX = {"acid": "carboxylic acid", "nitrile": "carbonitrile", "ol": "ol",
                "thiol": "thiol", "amine": "amine", "aldehyde": "carbaldehyde",
                "ester": "carboxylate"}
_RETAINED_BENZENE = {"acid": "benzoic acid", "nitrile": "benzonitrile",
                     "ol": "phenol", "amine": "aniline", "aldehyde": "benzaldehyde",
                     "ester": "benzoate"}
_PG_KIND = {"carboxylic_acid": "acid", "nitrile": "nitrile", "thiol": "thiol",
            "aldehyde": "aldehyde", "ester": "ester",
            "primary_alcohol": "ol", "secondary_alcohol": "ol",
            "tertiary_alcohol": "ol", "phenol": "ol", "alcohol": "ol",
            "primary_amine": "amine", "secondary_amine": "amine",
            "tertiary_amine": "amine", "aromatic_amine": "amine"}
# hydride classes read as "no principal characteristic group"
_HYDRIDE_PG = frozenset({"primary_phosphine", "secondary_phosphine",
                         "tertiary_phosphine"})
_MAX_HEAVY = 160
_MAX_SUBSETS = 64


# ---------------------------------------------------------------------------
# small graph helpers
# ---------------------------------------------------------------------------

def _ring_systems(mol) -> List[FrozenSet[int]]:
    systems: List[Set[int]] = []
    for r in mol.GetRingInfo().AtomRings():
        r = set(r)
        merged = [s for s in systems if s & r]
        for s in merged:
            systems.remove(s)
            r |= s
        systems.append(r)
    return [frozenset(s) for s in systems]


def _n_rings_in(mol, atoms) -> int:
    return sum(1 for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms)


def _cycle_order(mol, ring: FrozenSet[int]) -> Optional[List[int]]:
    start = min(ring)
    order = [start]
    prev, cur = None, start
    while True:
        nxt = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                     if n.GetIdx() in ring and n.GetIdx() != prev)
        if prev is None:
            if len(nxt) != 2:
                return None
            nxt = nxt[:1]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        if cur in order:
            return None
        order.append(cur)
    return order if len(order) == len(ring) else None


def _cycle_numberings(order: Sequence[int]) -> List[Dict[int, int]]:
    n = len(order)
    out = []
    for s in range(n):
        for d in (1, -1):
            out.append({order[(s + d * k) % n]: k + 1 for k in range(n)})
    return out


def _hetero_valid(mol, ring, nums: List[Dict[int, int]]) -> List[Dict[int, int]]:
    """Ring numberings that give the heteroatoms their lowest locants (as a set,
    then element by element in Hantzsch-Widman order)."""
    het = [a for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    if not het:
        return nums

    def key(num):
        return (sorted(num[a] for a in het),
                tuple(tuple(sorted(num[a] for a in het
                                   if mol.GetAtomWithIdx(a).GetAtomicNum() == z))
                      for z in _HW_ORDER))
    best = min(key(n) for n in nums)
    return [n for n in nums if key(n) == best]


def _bond_order(mol, a: int, b: int) -> Optional[float]:
    bd = mol.GetBondBetweenAtoms(a, b)
    return bd.GetBondTypeAsDouble() if bd is not None else None


def _component_of(mol, seed: int, allowed: Set[int]) -> Set[int]:
    seen = {seed}
    stack = [seed]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j in allowed and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def _components(mol, atoms: Set[int]) -> List[Set[int]]:
    left = set(atoms)
    out = []
    while left:
        seed = min(left)
        comp = _component_of(mol, seed, left)
        left -= comp
        out.append(comp)
    return out


def _path_order(mol, atoms: Set[int]) -> Optional[List[int]]:
    """The atoms as one unbranched path (end to end), or None."""
    if len(atoms) == 1:
        return [next(iter(atoms))]
    deg = {a: sum(1 for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in atoms)
           for a in atoms}
    ends = sorted(a for a, d in deg.items() if d == 1)
    if len(ends) != 2 or any(d > 2 for d in deg.values()):
        return None
    order = [ends[0]]
    prev = None
    while len(order) < len(atoms):
        cur = order[-1]
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetIdx() in atoms and n.GetIdx() != prev]
        if len(nxt) != 1:
            return None
        prev = cur
        order.append(nxt[0])
    return order


def _canonical_fragment(mol, atoms, isomeric: bool = True) -> str:
    """A canonical SMILES of the fragment itself: MolFragmentToSmiles orders the
    atoms by their ranks in the whole molecule, so two symmetry-equivalent
    fragments can come out as different strings ('[SiH3][SiH2]' and
    '[SiH2][SiH3]'); the string is parsed again and written canonically."""
    smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True,
                                   isomericSmiles=isomeric)
    frag = Chem.MolFromSmiles(smi, sanitize=False)
    if frag is None:
        return ""
    try:
        frag.UpdatePropertyCache(strict=False)
    except Exception:  # noqa: BLE001
        return ""
    return Chem.MolToSmiles(frag, canonical=True, isomericSmiles=isomeric)


def _fragment_key(mol, atoms: Set[int], dummies: Sequence[int] = ()) -> str:
    """Canonical isomeric SMILES of ``atoms`` with each atom of ``dummies`` written
    as a dummy atom (the linker atom next to a unit)."""
    rw = Chem.RWMol(mol)
    for d in dummies:
        at = rw.GetAtomWithIdx(d)
        at.SetAtomicNum(0)
        at.SetFormalCharge(0)
        at.SetIsotope(0)
        at.SetNoImplicit(True)
        at.SetNumExplicitHs(0)
        at.SetIsAromatic(False)
        at.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    inside = set(atoms)
    for d in dummies:
        for bd in rw.GetAtomWithIdx(d).GetBonds():
            if bd.GetOtherAtomIdx(d) in inside and bd.GetIsAromatic():
                return ""
    rw.UpdatePropertyCache(strict=False)
    return _canonical_fragment(rw, set(atoms) | set(dummies))


def _is_nitro_n(atom) -> bool:
    if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
        return False
    os_ = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1]
    return len(os_) == 2 and atom.GetDegree() == 3


def _scope_ok(mol) -> bool:
    if mol.GetNumHeavyAtoms() > _MAX_HEAVY or mol.GetNumHeavyAtoms() < 3:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for at in mol.GetAtoms():
        if at.GetAtomicNum() not in _ALLOWED_Z or at.GetIsotope() or at.GetNumRadicalElectrons():
            return False
        q = at.GetFormalCharge()
        if q:
            if q == 1 and _is_nitro_n(at):
                continue
            if q == -1 and at.GetAtomicNum() == 8 and at.GetDegree() == 1 and \
                    _is_nitro_n(at.GetNeighbors()[0]):
                continue
            return False
    for bd in mol.GetBonds():
        if bd.GetStereo() != Chem.BondStereo.STEREONONE:
            return False
    return True


# ---------------------------------------------------------------------------
# parent units (kernels)
# ---------------------------------------------------------------------------

class _Kernel:
    __slots__ = ("kind", "skel", "order", "pcg", "anchor", "pg", "organyl",
                 "organyl_o", "elem", "ring_name", "key")

    def __init__(self, kind, skel, order, pcg=frozenset(), anchor=None, pg=None,
                 organyl=frozenset(), organyl_o=None, elem=None):
        self.kind = kind            # 'ring' | 'chain' | 'hydride'
        self.skel = frozenset(skel)  # skeletal atoms of the parent hydride
        self.order = list(order)    # cyclic order / path order
        self.pcg = frozenset(pcg)   # atoms of the suffix group outside the skeleton
        self.anchor = anchor        # skeletal atom bearing (or being) the suffix
        self.pg = pg                # 'acid', 'nitrile', 'ol',... or None
        self.organyl = frozenset(organyl)  # ester organyl atoms (not in the unit)
        self.organyl_o = organyl_o  # the ester oxygen bonded to the organyl
        self.elem = elem            # hydride element
        self.ring_name = None
        self.key = None

    @property
    def core(self) -> FrozenSet[int]:
        return self.skel | self.pcg


def _ring_kernel_ok(mol, ring: FrozenSet[int]) -> bool:
    """A monocycle that is saturated, or mancude without indicated hydrogen."""
    if _n_rings_in(mol, ring) != 1:
        return False
    atoms = [mol.GetAtomWithIdx(a) for a in ring]
    if any(a.GetAtomicNum() not in _RING_ELEMENTS for a in atoms):
        return False
    arom = [a.GetIsAromatic() for a in atoms]
    bonds = [mol.GetBondBetweenAtoms(a, b) for a in ring for b in ring
             if a < b and mol.GetBondBetweenAtoms(a, b) is not None]
    if all(arom):
        n = len(ring)
        het = [a for a in atoms if a.GetAtomicNum() != 6]
        if any(a.GetTotalNumHs() for a in het):
            return False
        if n == 6:
            return all(a.GetAtomicNum() in (6, 7) for a in atoms)
        if n == 5:
            return sum(1 for a in het if a.GetAtomicNum() in (8, 16)) == 1 and \
                all(a.GetAtomicNum() in (6, 7, 8, 16) for a in atoms)
        return False
    if any(arom):
        return False
    return all(b.GetBondType() == Chem.BondType.SINGLE for b in bonds)


def _ring_parent_name(mol, ring: FrozenSet[int]) -> Optional[str]:
    from ..assembly.fragment_naming import name_fragment_recursively
    from ..data.retained_names import get_retained_name
    from ..errors import is_failure_name
    from ..perception.molcache import canon_smiles
    rw = Chem.RWMol(mol)
    for a in sorted(set(range(mol.GetNumAtoms())) - set(ring), reverse=True):
        rw.RemoveAtom(a)
    for at in rw.GetAtoms():
        at.SetNoImplicit(False)
        at.SetNumExplicitHs(0)
        at.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    try:
        Chem.SanitizeMol(rw)
    except Exception:  # noqa: BLE001
        return None
    smi = Chem.MolToSmiles(rw)
    if Chem.MolFromSmiles(smi) is None:
        return None
    c = canon_smiles(smi)
    name = get_retained_name(c) or name_fragment_recursively(c)
    if not name or is_failure_name(name) or any(ch in name for ch in "()[]{} "):
        return None
    return name


def _pcg_kernels(mol, pg: str, matches, rings: List[FrozenSet[int]]):
    """One kernel per principal-characteristic-group instance, or None."""
    kind = _PG_KIND.get(pg)
    if kind is None:
        return None
    ring_of = {}
    for rs in rings:
        for a in rs:
            ring_of[a] = rs
    kernels = []
    for m in matches:
        atoms = [mol.GetAtomWithIdx(i) for i in m]
        if kind in ("acid", "nitrile", "aldehyde", "ester"):
            cs = [a for a in atoms if a.GetAtomicNum() == 6]
            if kind == "ester":
                # (acyl C, =O, -O-, organyl C)
                acyl = None
                for a in cs:
                    if any(mol.GetBondBetweenAtoms(a.GetIdx(), o.GetIdx()) is not None
                           and _bond_order(mol, a.GetIdx(), o.GetIdx()) == 2.0
                           for o in atoms if o.GetAtomicNum() == 8):
                        acyl = a
                if acyl is None:
                    return None
                os_ = [o for o in atoms if o.GetAtomicNum() == 8]
                o_dbl = [o for o in os_ if _bond_order(mol, acyl.GetIdx(), o.GetIdx()) == 2.0]
                o_sgl = [o for o in os_ if _bond_order(mol, acyl.GetIdx(), o.GetIdx()) == 1.0]
                if len(o_dbl) != 1 or len(o_sgl) != 1:
                    return None
                c = acyl
                group = {o_dbl[0].GetIdx(), o_sgl[0].GetIdx()}
                eo = o_sgl[0]
                if eo.IsInRing() or eo.GetDegree() != 2:
                    return None
                r_atom = [n for n in eo.GetNeighbors() if n.GetIdx() != acyl.GetIdx()][0]
                if r_atom.GetAtomicNum() != 6:
                    return None
                organyl = _component_of(mol, r_atom.GetIdx(),
                                        set(range(mol.GetNumAtoms())) - {eo.GetIdx()})
                if acyl.GetIdx() in organyl:
                    return None
                organyl_o = eo.GetIdx()
            else:
                if len(cs) != 1:
                    return None
                c = cs[0]
                group = {a.GetIdx() for a in atoms if a.GetIdx() != c.GetIdx()}
                organyl, organyl_o = frozenset(), None
            if c.IsInRing():
                return None
            others = [n for n in c.GetNeighbors() if n.GetIdx() not in group]
            if len(others) != 1 or others[0].GetAtomicNum() != 6:
                return None
            nb = others[0]
            if nb.GetIdx() in ring_of:
                rs = ring_of[nb.GetIdx()]
                kernels.append(_Kernel("ring", rs, [], pcg=group | {c.GetIdx()},
                                       anchor=nb.GetIdx(), pg=kind, organyl=organyl,
                                       organyl_o=organyl_o))
            else:
                kernels.append(_Kernel("chain", {c.GetIdx()}, [], pcg=group,
                                       anchor=c.GetIdx(), pg=kind, organyl=organyl,
                                       organyl_o=organyl_o))
        else:
            het = [a for a in atoms if a.GetAtomicNum() != 6]
            cs = [a for a in atoms if a.GetAtomicNum() == 6]
            if len(het) != 1 or len(cs) != 1:
                return None
            h, c = het[0], cs[0]
            if h.GetDegree() != 1 or _bond_order(mol, h.GetIdx(), c.GetIdx()) != 1.0:
                return None
            if kind == "amine" and h.GetTotalNumHs() != 2:
                return None
            if c.GetIdx() in ring_of:
                kernels.append(_Kernel("ring", ring_of[c.GetIdx()], [], pcg={h.GetIdx()},
                                       anchor=c.GetIdx(), pg=kind))
            else:
                kernels.append(_Kernel("chain", {c.GetIdx()}, [], pcg={h.GetIdx()},
                                       anchor=c.GetIdx(), pg=kind))
    # chain kernels: the unbranched acyclic carbon chain through the anchor
    acyclic_c = {a.GetIdx() for a in mol.GetAtoms()
                 if a.GetAtomicNum() == 6 and not a.IsInRing()}
    pcg_atoms = set()
    for k in kernels:
        pcg_atoms |= k.pcg
    for k in kernels:
        if k.kind == "ring":
            if not _ring_kernel_ok(mol, k.skel):
                return None
            k.order = _cycle_order(mol, k.skel)
            if k.order is None:
                return None
            continue
        chain = _component_of(mol, k.anchor, acyclic_c - pcg_atoms | {k.anchor})
        order = _path_order(mol, chain)
        if order is None:
            return None
        if k.pg in ("acid", "nitrile", "aldehyde", "ester") and k.anchor not in (order[0], order[-1]):
            return None
        for a, b in zip(order, order[1:]):
            if _bond_order(mol, a, b) != 1.0:
                return None
        k.skel = frozenset(chain)
        k.order = order
    # every kernel holds one group, and kernels do not overlap
    seen: Set[int] = set()
    for k in kernels:
        if k.core & seen:
            return None
        seen |= k.core
    return kernels


def _hydride_kernels(mol, rings) -> Optional[List[_Kernel]]:
    ring_atoms = set().union(*rings) if rings else set()
    out = []
    done: Set[int] = set()
    for at in mol.GetAtoms():
        i = at.GetIdx()
        z = at.GetAtomicNum()
        if i in done or i in ring_atoms or z not in _HYDRIDE_NAMES:
            continue
        same = _component_of(mol, i, {a.GetIdx() for a in mol.GetAtoms()
                                      if a.GetAtomicNum() == z and a.GetIdx() not in ring_atoms})
        done |= same
        if z in (16, 34, 52) and len(same) < 3:
            continue    # -S-, -SS- are not parent hydrides,:27874)
        order = _path_order(mol, same)
        if order is None:
            return None
        std = _HYDRIDE_NAMES[z][1]
        for a in same:
            atom = mol.GetAtomWithIdx(a)
            if atom.GetTotalValence() != std:
                return None
            if any(bd.GetBondTypeAsDouble() != 1.0 for bd in atom.GetBonds()):
                return None
        out.append(_Kernel("hydride", same, order, elem=z))
    return out


# ---------------------------------------------------------------------------
# substituent prefixes on a unit
# ---------------------------------------------------------------------------

def _decoration_prefix(mol, comp: Set[int], attach: int, parent: Sequence[int]) -> Optional[str]:
    from ..assembly.substituent_naming import name_substituent_fragment
    from ..errors import is_failure_name
    host = [n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetIdx() in set(parent)]
    if len(host) != 1 or _bond_order(mol, attach, host[0]) != 1.0:
        return None
    at = mol.GetAtomWithIdx(attach)
    if len(comp) == 1:
        z = at.GetAtomicNum()
        if z in _HALO:
            return _HALO[z]
        if z == 8 and at.GetTotalNumHs() == 1:
            return "hydroxy"
        if z == 16 and at.GetTotalNumHs() == 1:
            return "sulfanyl"
        if z == 7 and at.GetTotalNumHs() == 2:
            return "amino"
    if len(comp) == 3 and _is_nitro_n(at):
        return "nitro"
    if any(mol.GetAtomWithIdx(a).GetFormalCharge() for a in comp):
        return None
    try:
        name = name_substituent_fragment(mol, sorted(comp), attach, list(parent))
    except Exception:  # noqa: BLE001 - an unnamed branch declines the detector
        return None
    if not name or is_failure_name(name):
        return None
    return name


# ---------------------------------------------------------------------------
# naming one unit
# ---------------------------------------------------------------------------

def _elide(parent: str, suffix: str) -> str:
    if parent.endswith("e") and suffix[:1] in "aiouy":
        return parent[:-1]
    return parent


def _numberings(mol, k: _Kernel) -> List[Dict[int, int]]:
    if k.kind == "ring":
        return _hetero_valid(mol, k.skel, _cycle_numberings(k.order))
    order = k.order
    if len(order) == 1:
        return [{order[0]: 1}]
    fwd = {a: i + 1 for i, a in enumerate(order)}
    rev = {a: len(order) - i for i, a in enumerate(order)}
    return [fwd, rev]


def _unit_parent(k: _Kernel, num: Dict[int, int]) -> Optional[str]:
    """The parent hydride with its suffix, numbered by ``num``."""
    if k.kind == "ring":
        p = k.ring_name
        if k.pg is None:
            return p
        if p == "benzene" and k.pg in _RETAINED_BENZENE:
            return _RETAINED_BENZENE[k.pg] if num[k.anchor] == 1 else None
        suffix = _RING_SUFFIX[k.pg]
        return f"{_elide(p, suffix)}-{num[k.anchor]}-{suffix}"
    if k.kind == "hydride":
        name, _ = _HYDRIDE_NAMES[k.elem]
        n = len(k.order)
        if n == 1:
            return name
        from ..assembly.naming_utils import simple_multiplier_word
        word = simple_multiplier_word(n)
        return f"{word}{name}" if word else None
    from ..data.chain_names import get_chain_prefix
    n = len(k.order)
    stem = get_chain_prefix(n)
    if k.pg == "acid":
        return None if n == 1 else ("acetic acid" if n == 2 else f"{stem}anoic acid")
    if k.pg == "ester":
        return None if n == 1 else ("acetate" if n == 2 else f"{stem}anoate")
    if k.pg == "nitrile":
        return None if n == 1 else ("acetonitrile" if n == 2 else f"{stem}anenitrile")
    if k.pg == "aldehyde":
        return None if n == 1 else ("acetaldehyde" if n == 2 else f"{stem}anal")
    suffix = {"ol": "ol", "thiol": "thiol", "amine": "amine"}.get(k.pg)
    if suffix is None:
        return None
    if n == 1:
        return f"{_elide('methane', suffix)}{suffix}"
    return f"{_elide(stem + 'ane', suffix)}-{num[k.anchor]}-{suffix}"


def _render_prefixes(entries: List[Tuple[int, str]], mononuclear: bool) -> Optional[str]:
    from ..assembly.composition_primitives import _join_prefixes
    from ..assembly.naming_utils import (
        enclose_if_compound,
        format_substituent_prefix,
        multiplied_component,
        prefix_citation_sort_key,
    )
    groups: Dict[str, List[int]] = {}
    for loc, name in entries:
        groups.setdefault(name, []).append(loc)
    order = sorted(groups, key=prefix_citation_sort_key)
    if mononuclear:
        if len(order) > 1:
            return None
        name = order[0]
        count = len(groups[name])
        if count == 1:
            return enclose_if_compound(name) if enclose_if_compound(name) != name else name
        return multiplied_component(count, name, name)
    return _join_prefixes([format_substituent_prefix(nm, sorted(groups[nm]), len(groups[nm]))
                           for nm in order])


def _name_unit(mol, k: _Kernel, attach: int, decos: List[Tuple[Set[int], int, int]]):
    """(unit name, attachment locant or None, chosen numbering) or None.

    ``decos``: (atoms, decoration atom bonded to the skeleton, skeleton atom)."""
    from ..assembly.composition_primitives import _join_prefix_to_name
    from ..assembly.naming_utils import prefix_citation_sort_key
    entries: List[Tuple[int, str]] = []    # (skeleton atom, prefix)
    for comp, a, host in decos:
        if host not in k.skel:
            return None
        p = _decoration_prefix(mol, comp, a, sorted(k.skel))
        if p is None:
            return None
        entries.append((host, p))
    nums = _numberings(mol, k)
    if not nums:
        return None

    def key(num):
        k1 = [num[k.anchor]] if k.pg is not None else []
        k2 = num[attach]
        k3 = sorted(num[h] for h, _ in entries)
        by: Dict[str, List[int]] = {}
        for h, p in entries:
            by.setdefault(p, []).append(num[h])
        k4 = [sorted(by[p]) for p in sorted(by, key=prefix_citation_sort_key)]
        return (k1, k2, k3, k4)
    best_key = min(key(n) for n in nums)
    best = [n for n in nums if key(n) == best_key]
    mononuclear = len(k.order) == 1 and k.kind in ("hydride", "chain")
    # (the Blue Book): "All locants are omitted in compounds or
    # substituent groups in which all substitutable positions are completely
    # substituted... in the same way" ('1,1'-methylenebis(pentafluorodisilane)
    # (PIN)',:6190); (:3031): "All locants are omitted for parent
    # compounds when all substitutable hydrogen atoms have the same locant"
    # ('difluoroacetic acid (PIN)'). The substitutable positions are the skeleton
    # atoms that carry hydrogen in the unit parent (with its suffix); the
    # attachment is one of them.
    n_deco = Counter(h for h, _ in entries)
    subst = [a for a in k.skel
             if mol.GetAtomWithIdx(a).GetTotalNumHs() + n_deco[a] + (a == attach) > 0]
    one_position = bool(entries) and len(subst) == 1
    # (a single prefix is not a complete substitution: '1,1'-(ethane-1,2-diyl)bis-
    # (3-methyltrisulfane) (PIN)',:23385)
    decorated_all = (len(entries) >= 2 and len({p for _, p in entries}) == 1
                     and all(mol.GetAtomWithIdx(a).GetTotalNumHs() == 0 for a in subst))
    names = set()
    chosen = None
    for num in best:
        parent = _unit_parent(k, num)
        if parent is None:
            continue
        if mononuclear or decorated_all or one_position:
            pre = _render_prefixes([(1, p) for _, p in entries], True) if entries else ""
        else:
            pre = _render_prefixes([(num[h], p) for h, p in entries], False) if entries else ""
        if pre is None:
            return None
        names.add(_join_prefix_to_name(pre, parent) if pre else parent)
        chosen = num
    if len(names) != 1 or chosen is None:
        return None
    name = names.pop()
    loc = None if mononuclear else chosen[attach]
    if k.kind == "ring" and k.ring_name == "benzene" and k.pg in _RETAINED_BENZENE:
        loc = chosen[attach]
    return name, loc, chosen, bool(entries)


# ---------------------------------------------------------------------------
# the linker
# ---------------------------------------------------------------------------

class _Comp:
    __slots__ = ("kind", "atoms", "order", "name", "ring_name")

    def __init__(self, kind, atoms, order=None, name=None):
        self.kind = kind      # 'ring' | 'chain' | 'chalc' | 'NH' | 'N3' | 'NN'
        self.atoms = frozenset(atoms)
        self.order = order
        self.name = name
        self.ring_name = None


def _linker_components(mol, link: Set[int], rings) -> Optional[List[_Comp]]:
    comps: List[_Comp] = []
    left = set(link)
    for rs in rings:
        if rs & left:
            if not rs <= left or not _ring_kernel_ok(mol, rs):
                return None
            c = _Comp("ring", rs, _cycle_order(mol, rs))
            if c.order is None:
                return None
            c.ring_name = _ring_parent_name(mol, rs)
            if c.ring_name is None:
                return None
            comps.append(c)
            left -= rs
    carbons = {a for a in left if mol.GetAtomWithIdx(a).GetAtomicNum() == 6}
    for comp in _components(mol, carbons):
        order = _path_order(mol, comp)
        if order is None:
            return None
        comps.append(_Comp("chain", comp, order))
    left -= carbons
    for z in (8, 16, 34, 52):
        same = {a for a in left if mol.GetAtomWithIdx(a).GetAtomicNum() == z}
        for comp in _components(mol, same):
            order = _path_order(mol, comp)
            name = _CHALCOGEN_COMPONENT.get((z, len(comp)))
            if order is None or name is None:
                return None
            for a in comp:
                atom = mol.GetAtomWithIdx(a)
                if atom.GetDegree() != 2 or atom.GetTotalNumHs():
                    return None
            comps.append(_Comp("chalc", comp, order, name))
        left -= same
    ns = {a for a in left if mol.GetAtomWithIdx(a).GetAtomicNum() == 7}
    for comp in _components(mol, ns):
        if len(comp) == 1:
            a = next(iter(comp))
            atom = mol.GetAtomWithIdx(a)
            if any(bd.GetBondTypeAsDouble() != 1.0 for bd in atom.GetBonds()):
                return None
            if atom.GetDegree() == 2 and atom.GetTotalNumHs() == 1:
                comps.append(_Comp("NH", comp, [a], "azanediyl"))
            elif atom.GetDegree() == 3 and atom.GetTotalNumHs() == 0:
                comps.append(_Comp("N3", comp, [a], "nitrilo"))
            else:
                return None
        elif len(comp) == 2:
            a, b = sorted(comp)
            if _bond_order(mol, a, b) != 1.0:
                return None
            for x in (a, b):
                atom = mol.GetAtomWithIdx(x)
                if atom.GetDegree() != 2 or atom.GetTotalNumHs():
                    return None
                out = [bd for bd in atom.GetBonds() if bd.GetOtherAtomIdx(x) not in comp]
                if len(out) != 1 or out[0].GetBondTypeAsDouble() != 2.0:
                    return None
            comps.append(_Comp("NN", comp, [a, b], "hydrazinediylidene"))
        else:
            return None
    left -= ns
    if left:
        return None
    return comps


def _chain_unsat(mol, order) -> List[Tuple[int, int, float]]:
    """(index, index + 1, bond order) of each multiple bond along the chain."""
    out = []
    for i, (a, b) in enumerate(zip(order, order[1:])):
        o = _bond_order(mol, a, b)
        if o != 1.0:
            out.append((i, i + 1, o))
    return out


def _chain_name(mol, comp: _Comp, fv: Dict[int, int], toward_parent: Optional[Set[int]]):
    """Name a carbon-chain component. ``fv``: atom -> number of free valences;
    ``toward_parent``: atoms whose free valence points to the parent units (a
    branch component), or None for the central component."""
    from ..data.chain_names import get_chain_prefix
    order = comp.order
    n = len(order)
    total = sum(fv.values())
    if n == 1:
        a = order[0]
        h = mol.GetAtomWithIdx(a).GetTotalNumHs()
        word = {2: "methylene", 3: "methanetriyl", 4: "methanetetrayl"}.get(total)
        if word is None or h + total != 4:
            return None
        return word, False
    # one double or triple bond at most ('prop-1-ene-1,3-diyl (preferred prefix)',
    #, the Blue Book); only in the central component
    unsat = _chain_unsat(mol, order)
    if any(o not in (2.0, 3.0) for _, _, o in unsat) or len(unsat) > 1:
        return None
    if toward_parent is not None and unsat:
        return None
    cands = []
    for direction in (order, list(reversed(order))):
        num = {a: i + 1 for i, a in enumerate(direction)}
        fvl = sorted(num[a] for a, c in fv.items() for _ in range(c))
        if toward_parent is not None:
            par = sorted(num[a] for a in toward_parent for _ in range(fv.get(a, 0)))
            key = (par, fvl)
        else:
            mult = sorted(min(num[order[i]], num[order[j]]) for i, j, _ in unsat)
            dbl = sorted(min(num[order[i]], num[order[j]]) for i, j, o in unsat if o == 2.0)
            key = (fvl, mult, dbl)
        cands.append((key, num))
    best = min(c[0] for c in cands)
    names = set()
    for key, num in cands:
        if key != best:
            continue
        if toward_parent is not None:
            outward = sorted(num[a] for a, c in fv.items() if a not in toward_parent for _ in range(c))
            inward = sorted(num[a] for a in toward_parent for _ in range(fv.get(a, 0)))
            locs = outward + inward
        else:
            locs = sorted(num[a] for a, c in fv.items() for _ in range(c))
        stem = get_chain_prefix(n)
        dbl = sorted(min(num[order[i]], num[order[j]]) for i, j, o in unsat if o == 2.0)
        tpl = sorted(min(num[order[i]], num[order[j]]) for i, j, o in unsat if o == 3.0)
        if not unsat:
            hyd = f"{stem}ane"
        elif n == 2:
            hyd = "ethene" if dbl else "ethyne"
        elif dbl:
            hyd = f"{stem}-{dbl[0]}-ene"
        else:
            hyd = f"{stem}-{tpl[0]}-yne"
        word = _VALENCE_WORD.get(total)
        if word is None:
            return None
        names.add(f"{hyd}-{','.join(map(str, locs))}-{word}")
    if len(names) != 1:
        return None
    return names.pop(), True


def _ring_comp_name(mol, comp: _Comp, fv: Dict[int, int], toward_parent: Optional[Set[int]]):
    nums = _hetero_valid(mol, comp.atoms, _cycle_numberings(comp.order))
    fixed = any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in comp.atoms)
    if any(c != 1 for c in fv.values()):
        return None
    cands = []
    for num in nums:
        allv = sorted(num[a] for a in fv)
        if toward_parent is None:
            key = (allv,)
        else:
            par = sorted(num[a] for a in toward_parent)
            key = (allv, par) if fixed else (par, allv)
        cands.append((key, num))
    best = min(c[0] for c in cands)
    names = set()
    for key, num in cands:
        if key != best:
            continue
        if toward_parent is None:
            locs = sorted(num[a] for a in fv)
        else:
            locs = sorted(num[a] for a in fv if a not in toward_parent) + \
                sorted(num[a] for a in toward_parent)
        word = _VALENCE_WORD.get(len(locs))
        if word is None:
            return None
        if comp.ring_name == "benzene" and len(locs) == 2:
            names.add(f"{locs[0]},{locs[1]}-phenylene")
        else:
            names.add(f"{comp.ring_name}-{','.join(map(str, locs))}-{word}")
    if len(names) != 1:
        return None
    return names.pop(), True


def _join_components(parts: List[str]) -> str:
    out = parts[0]
    for p in parts[1:]:
        if out[-1].isalpha() and p[0].isdigit():
            out += "-"
        out += p
    return out


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------

def _senior_ring_units(mol, rings, cands) -> Optional[List[_Kernel]]:
    from .ring_selection import ring_system_score
    scores = {rs: ring_system_score(mol, set(rs)) for rs in rings}
    best = min(scores.values())
    return [k for k in cands if scores[k.skel] == best]


def _units_and_linker(mol, units: List[_Kernel], rings, same_key_kernels):
    """Decompose for the unit set ``S``: (link atoms, per-unit decorations,
    per-unit (attach, link atom, bond order)) or None."""
    n_atoms = mol.GetNumAtoms()
    core = set()
    for k in units:
        core |= k.core | k.organyl
    rest = set(range(n_atoms)) - core
    owner = {}
    for i, k in enumerate(units):
        for a in k.core:
            owner[a] = i
    organyl_owner = {}
    for i, k in enumerate(units):
        for a in k.organyl:
            organyl_owner[a] = i
    decos: List[List[Tuple[Set[int], int, int]]] = [[] for _ in units]
    links = []
    for comp in _components(mol, rest):
        touch = {}
        for a in comp:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in organyl_owner:
                    return None
                if j in owner:
                    touch.setdefault(owner[j], []).append((j, a))
        if not touch:
            return None
        if len(touch) == 1:
            (i, pairs), = touch.items()
            if len(pairs) != 1:
                return None
            host, a = pairs[0]
            decos[i].append((comp, a, host))
        else:
            links.append((comp, touch))
    if len(links) != 1:
        return None
    link, touch = links[0]
    if set(touch) != set(range(len(units))):
        return None
    attach = []
    for i, k in enumerate(units):
        pairs = touch[i]
        if len(pairs) != 1:
            return None
        host, a = pairs[0]
        if host not in k.skel:
            return None
        attach.append((host, a, _bond_order(mol, host, a)))
    if len({o for _, _, o in attach}) != 1:
        return None
    #: another parent of the same skeleton stays in the linker
    for k in same_key_kernels:
        if k in units:
            continue
        if not k.core <= link:
            return None
    return link, decos, attach


def _linker_tree(mol, link: Set[int], units, attach, rings):
    comps = _linker_components(mol, link, rings)
    if comps is None:
        return None
    where = {}
    for ci, c in enumerate(comps):
        for a in c.atoms:
            where[a] = ("c", ci)
    for ui, (host, a, _o) in enumerate(attach):
        for x in units[ui].core:
            where[x] = ("u", ui)
    # edges between nodes
    edges: Dict[Tuple, List[Tuple[int, int]]] = {}
    for bd in mol.GetBonds():
        a, b = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        na, nb = where.get(a), where.get(b)
        if na is None or nb is None or na == nb:
            continue
        if na[0] == "u" and nb[0] == "u":
            return None
        key = tuple(sorted((na, nb)))
        edges.setdefault(key, []).append((a, b))
    if any(len(v) != 1 for v in edges.values()):
        return None
    nodes = [("c", i) for i in range(len(comps))] + [("u", i) for i in range(len(units))]
    if len(edges) != len(nodes) - 1:
        return None
    adj: Dict[Tuple, List[Tuple]] = {n: [] for n in nodes}
    for (x, y) in edges:
        adj[x].append(y)
        adj[y].append(x)
    # connected?
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        cur = stack.pop()
        for nx in adj[cur]:
            if nx not in seen:
                seen.add(nx)
                stack.append(nx)
    if len(seen) != len(nodes):
        return None
    if any(len(adj[("u", i)]) != 1 for i in range(len(units))):
        return None
    return comps, edges, adj, where


def _subtree_atoms(adj, comps, units, start, parent) -> Set[int]:
    out: Set[int] = set()
    stack = [(start, parent)]
    while stack:
        cur, par = stack.pop()
        if cur[0] == "c":
            out |= comps[cur[1]].atoms
        else:
            out |= units[cur[1]].core
        for nx in adj[cur]:
            if nx != par:
                stack.append((nx, cur))
    return out


def _edge_atoms(edges, x, y) -> Tuple[int, int]:
    """(atom in x, atom in y) of the bond joining nodes x and y."""
    key = tuple(sorted((x, y)))
    a, b = edges[key][0]
    return (a, b) if key[0] == x else (b, a)


def _side_atom(where, edges, x, y) -> int:
    a, b = edges[tuple(sorted((x, y)))][0]
    return a if where[a] == x else b


def _identical_children(mol, adj, comps, units, edges, where, node, parent) -> bool:
    kids = [n for n in adj[node] if n != parent]
    keys = set()
    for kid in kids:
        atoms = _subtree_atoms(adj, comps, units, kid, node)
        dummy = _side_atom(where, edges, node, kid)
        keys.add(_fragment_key(mol, atoms, [dummy]))
    return len(kids) >= 2 and len(keys) == 1 and "" not in keys


def _fixed_name(c: _Comp) -> Optional[str]:
    return c.name if c.kind in ("chalc", "NH", "N3", "NN") else None


def _name_component(mol, comps, adj, edges, where, node, parent):
    c = comps[node[1]]
    fixed = _fixed_name(c)
    if fixed is not None:
        return fixed, False
    fv: Dict[int, int] = {}
    toward = None if parent is None else set()
    for nx in adj[node]:
        a = _side_atom(where, edges, node, nx)
        if mol.GetBondBetweenAtoms(*_edge_atoms(edges, node, nx)).GetBondTypeAsDouble() != 1.0:
            return None
        fv[a] = fv.get(a, 0) + 1
        if parent is not None and nx != parent:
            toward.add(a)
    if c.kind == "chain":
        return _chain_name(mol, c, fv, toward)
    if c.kind == "ring":
        return _ring_comp_name(mol, c, fv, toward)
    return None


def _multiplying_group(mol, units, comps, edges, adj, where):
    roots = [("c", i) for i in range(len(comps))
             if _identical_children(mol, adj, comps, units, edges, where, ("c", i), None)]
    if len(roots) != 1:
        return None
    root = roots[0]
    kids = [n for n in adj[root]]
    rn = _name_component(mol, comps, adj, edges, where, root, None)
    if rn is None:
        return None
    root_name, _ = rn
    if all(k[0] == "u" for k in kids):
        return root_name, len(kids)
    if any(k[0] == "u" for k in kids):
        return None
    # walk one branch: components in order, the last one carries the units
    branch: List[str] = []
    branch_locants: List[bool] = []
    cur, par = kids[0], root
    units_per_branch = None
    while True:
        nxt = [n for n in adj[cur] if n != par]
        nm = _name_component(mol, comps, adj, edges, where, cur, par)
        if nm is None:
            return None
        branch.append(nm[0])
        branch_locants.append(nm[1])
        if all(n[0] == "u" for n in nxt):
            if len(nxt) > 1 and not _identical_children(mol, adj, comps, units, edges, where, cur, par):
                return None
            units_per_branch = len(nxt)
            break
        if len(nxt) != 1 or nxt[0][0] != "c":
            return None
        if comps[cur[1]].kind == "NN" or comps[nxt[0][1]].kind == "NN":
            return None
        par, cur = cur, nxt[0]
    if any(comps[n[1]].kind == "NN" for n in kids):
        return None
    from ..assembly.naming_utils import (
        COMPLEX_MULTIPLIERS,
        SIMPLE_MULTIPLIERS,
        apply_enclosing_marks,
    )
    k = len(kids)
    if len(branch) == 1:
        comp = comps[kids[0][1]]
        if branch_locants[0]:
            mult = f"{SIMPLE_MULTIPLIERS[k]}({branch[0]})"
        elif comp.kind == "N3":
            mult = f"{SIMPLE_MULTIPLIERS[k]}{branch[0]}"
        else:
            mult = f"{COMPLEX_MULTIPLIERS[k]}({branch[0]})"
    else:
        mult = COMPLEX_MULTIPLIERS[k] + apply_enclosing_marks(_join_components(branch), -1)
    return root_name + mult, k * units_per_branch


# (the Blue Book): the order whose later element terminates an
# a(ba)n chain of alternating heteroatoms
_ABA_ORDER = (8, 16, 34, 52, 7, 15, 33, 51, 83, 14, 32, 50, 82, 5)


def _max_heterounits_on_a_chain(mol, excluded: Set[int]) -> int:
    """The largest number of heterounits on one unbranched acyclic chain that
    holds a carbon atom. A run of identical heteroatoms counts once
    ('-SS-', a trisulfane parent); halogens and the ``excluded`` atoms (principal
    characteristic groups) are not chain atoms."""
    acyc = [a.GetIdx() for a in mol.GetAtoms()
            if not a.IsInRing() and a.GetAtomicNum() not in (1, 9, 17, 35, 53)
            and a.GetIdx() not in excluded]
    aset = set(acyc)
    nbrs = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in aset]
            for a in acyc}
    z = {a: mol.GetAtomWithIdx(a).GetAtomicNum() for a in acyc}
    best = 0
    if len(acyc) > 400:
        return 99
    for start in acyc:
        # depth-first over simple paths of the acyclic forest from ``start``
        stack = [(start, -1, (1 if z[start] != 6 else 0), z[start] == 6)]
        while stack:
            cur, prev, units, has_c = stack.pop()
            if has_c and units > best:
                best = units
            for nx in nbrs[cur]:
                if nx == prev:
                    continue
                add = 0
                if z[nx] != 6 and z[nx] != z[cur]:
                    add = 1
                stack.append((nx, cur, units + add, has_c or z[nx] == 6))
    return best


def _aba_chain_in(mol, comps) -> bool:
    """An a(ba)n chain of alternating heteroatoms ending in two atoms of the
    later element is a parent hydride of its own: 'dithioxane'
    (HS-O-SH), 'diselenoxane'; '-O-S-O-' is not one ('sulfanediylbis(oxy)')."""
    het = {a for c in comps if c.kind != "ring" for a in c.atoms
           if mol.GetAtomWithIdx(a).GetAtomicNum() != 6}
    rank = {zz: i for i, zz in enumerate(_ABA_ORDER)}
    for mid in het:
        zm = mol.GetAtomWithIdx(mid).GetAtomicNum()
        ends = [n for n in mol.GetAtomWithIdx(mid).GetNeighbors() if n.GetIdx() in het]
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                ze = ends[i].GetAtomicNum()
                if ze == ends[j].GetAtomicNum() and ze != zm and \
                        rank.get(ze, -1) > rank.get(zm, 99):
                    return True
    return False


def _phane_or_replacement(mol, rings, link, comps, pcg_atoms=frozenset()) -> bool:
    """True when a phane name (2)) or a skeletal replacement name
     is the PIN instead."""
    if _max_heterounits_on_a_chain(mol, set(pcg_atoms)) >= 4:
        return True
    if _aba_chain_in(mol, comps):
        return True
    # linear phane: four or more ring systems joined in a line
    if len(rings) >= 4:
        ring_of = {}
        for i, rs in enumerate(rings):
            for a in rs:
                ring_of[a] = i
        nbrs: Dict[int, Set[int]] = {i: set() for i in range(len(rings))}
        nonring = set(range(mol.GetNumAtoms())) - set(ring_of)
        for i, rs in enumerate(rings):
            for a in rs:
                for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = nb.GetIdx()
                    if j in ring_of and ring_of[j] != i:
                        nbrs[i].add(ring_of[j])
                    elif j in nonring:
                        reach = _component_of(mol, j, nonring)
                        for x in reach:
                            for nb2 in mol.GetAtomWithIdx(x).GetNeighbors():
                                y = nb2.GetIdx()
                                if y in ring_of and ring_of[y] != i:
                                    nbrs[i].add(ring_of[y])
        if all(len(v) <= 2 for v in nbrs.values()):
            return True
    return False


def _unit_stereo(mol, units, numberings, primes_order, unit_atoms) -> Optional[str]:
    """'(2R,2'R)' for stereocentres of the unit skeletons; '' when none; None when
    the linker holds one or the units disagree."""
    has_tag = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
    if not has_tag:
        return ""
    from ..perception.stereo import get_stereocenters
    centers = {c["idx"]: c["cip"] for c in get_stereocenters(Chem.Mol(mol))}
    in_units = set().union(*unit_atoms)
    labels = []
    for ui in primes_order:
        k = units[ui]
        num = numberings[ui]
        for a in k.skel:
            if a in centers:
                lab = centers[a]
                if lab not in ("R", "S"):
                    return None
                labels.append((num[a], primes_order.index(ui), lab))
    # a centre outside the units is handled by no one here
    for a in centers:
        if a not in in_units:
            return None
    if not labels:
        return ""
    per_unit: Dict[int, List[Tuple[int, str]]] = {}
    for loc, p, lab in labels:
        per_unit.setdefault(p, []).append((loc, lab))
    sets = {tuple(sorted(v)) for v in per_unit.values()}
    if len(sets) != 1 or len(per_unit) != len(units):
        return None
    labels.sort(key=lambda t: (t[0], t[1]))
    return "(" + ",".join(f"{loc}{chr(39) * p}{lab}" for loc, p, lab in labels) + ")-"


def _unit_token(unit_name: str, count: int, substituted: bool, kind: str, n_chain: int) -> str:
    from ..assembly.naming_utils import (
        COMPLEX_MULTIPLIERS,
        SIMPLE_MULTIPLIERS,
        apply_enclosing_marks,
        opens_with_replacement_prefix,
    )
    if substituted or kind == "hydride" or unit_name[0].isdigit() or \
            opens_with_replacement_prefix(unit_name):
        return COMPLEX_MULTIPLIERS[count] + apply_enclosing_marks(unit_name, -1)
    if any(ch.isdigit() for ch in unit_name) or (kind == "chain" and n_chain >= 5):
        return f"{SIMPLE_MULTIPLIERS[count]}({unit_name})"
    return SIMPLE_MULTIPLIERS[count] + unit_name


def _organyl_words(mol, units) -> Optional[str]:
    from ..assembly.naming_utils import (
        enclose_if_compound,
        multiplied_component,
        prefix_citation_sort_key,
    )
    names = []
    for k in units:
        r_atoms = k.organyl
        r_attach = [a for a in r_atoms
                    if mol.GetBondBetweenAtoms(a, k.organyl_o) is not None]
        if len(r_attach) != 1:
            return None
        p = _decoration_prefix(mol, set(r_atoms), r_attach[0], [k.organyl_o])
        if p is None:
            return None
        names.append(p)
    cnt = Counter(names)
    words = []
    for nm in sorted(cnt, key=prefix_citation_sort_key):
        if cnt[nm] == 1:
            words.append(enclose_if_compound(nm) if " " in nm else nm)
        else:
            words.append(multiplied_component(cnt[nm], nm, nm))
    return " ".join(words)


def _try_unit_set(mol, units: List[_Kernel], rings, same_key) -> Optional[str]:
    dec = _units_and_linker(mol, units, rings, same_key)
    if dec is None:
        return None
    link, decos, attach = dec
    # identical decorated units, each with its linker atom as a dummy
    keys = []
    orders = []
    for i, k in enumerate(units):
        atoms = set(k.core)
        for comp, _a, _h in decos[i]:
            atoms |= comp
        dummy = attach[i][1]
        keys.append(_fragment_key(mol, atoms, [dummy]))
        orders.append((atoms, dummy))
    if len(set(keys)) != 1 or "" in keys:
        return None
    #: alpha-amino acid units are named by the amino acid rules
    for i, k in enumerate(units):
        if k.kind == "chain" and k.pg in ("acid", "ester") and len(k.order) >= 2:
            alpha = k.order[1] if k.order[0] == k.anchor else k.order[-2]
            if any(host == alpha and mol.GetAtomWithIdx(a).GetAtomicNum() == 7
                   for _comp, a, host in decos[i]):
                return None
    tree = _linker_tree(mol, link, units, attach, rings)
    if tree is None:
        return None
    comps, edges, adj, where = tree
    # (the Blue Book): a chain of alternating heteroatoms ending in
    # two identical atoms is a parent hydride of its own ('disiloxane (preselected
    # name)':7631, 'disilathiane':8020); a heteroatom linker bonded directly to
    # two hydride units makes such a chain.
    if units[0].kind == "hydride":
        unit_of = {a: i for i, k in enumerate(units) for a in k.skel}
        for c in comps:
            if c.kind in ("chalc", "NH", "N3", "NN"):
                touched = {unit_of[n.GetIdx()] for a in c.atoms
                           for n in mol.GetAtomWithIdx(a).GetNeighbors()
                           if n.GetIdx() in unit_of}
                if len(touched) >= 2:
                    return None
    pcg_atoms = set()
    for k in units:
        pcg_atoms |= k.pcg
    if _phane_or_replacement(mol, rings, link, comps, pcg_atoms):
        return None
    mg = _multiplying_group(mol, units, comps, edges, adj, where)
    if mg is None:
        return None
    group, n_units = mg
    if n_units != len(units):
        return None
    # name every unit; the names and attachment locants must coincide
    names = set()
    numberings = []
    loc = None
    substituted = False
    for i, k in enumerate(units):
        nu = _name_unit(mol, k, attach[i][0], decos[i])
        if nu is None:
            return None
        nm, lc, num, subd = nu
        names.add((nm, lc))
        numberings.append(num)
        loc, substituted = lc, subd
    if len(names) != 1:
        return None
    unit_name, _ = names.pop()
    unit_atoms = [atoms | units[i].organyl for i, (atoms, _d) in enumerate(orders)]
    stereo = _unit_stereo(mol, units, numberings, list(range(len(units))), unit_atoms)
    if stereo is None:
        return None
    count = len(units)
    from ..assembly.naming_utils import apply_enclosing_marks
    if "(" in group or "[" in group or "{" in group:
        gtok = apply_enclosing_marks(group, -1)
    elif any(ch.isdigit() for ch in group):
        gtok = f"({group})"
    else:
        gtok = group
    loc_block = "" if loc is None else ",".join(f"{loc}{chr(39) * p}" for p in range(count)) + "-"
    utok = _unit_token(unit_name, count, substituted, units[0].kind, len(units[0].order))
    body = f"{loc_block}{gtok}{utok}"
    if units[0].pg == "ester":
        if stereo:
            return None
        words = _organyl_words(mol, units)
        if words is None:
            return None
        return f"{words} {body}"
    return f"{stereo}{body}"


def _kernel_key(mol, k: _Kernel) -> str:
    atoms = set(k.core)
    return _canonical_fragment(mol, atoms, isomeric=False) + f"|{k.kind}|{k.pg}"


def name_general_multiplicative(mol) -> Optional[str]:
    """A multiplicative PIN for ``mol``, or None.

    Only for a whole molecule: a fragment named inside another name (a substituent
    group whose capped fragment is named and turned into a prefix, a decomposition
    fragment) keeps the substitutive name its enclosing name needs; a multiplicative
    name cannot be written as a prefix ('methylenebis(silane)' for the
    '(disilylmethyl)' group of '4-(disilylmethyl)hexan-2-one')."""
    from ..assembly.fragment_naming import is_top_level_naming
    if not is_top_level_naming():
        return None
    try:
        return _name_general_multiplicative(mol)
    except Exception as exc:  # noqa: BLE001 - the detector never breaks naming
        logger.debug("general multiplicative detector error: %s", exc)
        return None


def _same_skeleton_rings_bonded(mol, rings) -> bool:
    """Two ring systems of one skeleton joined by a bond form a ring assembly
    , a parent of its own."""
    ring_of = {}
    for i, rs in enumerate(rings):
        for a in rs:
            ring_of[a] = i
    keys = [_canonical_fragment(mol, rs, isomeric=False) for rs in rings]
    for bd in mol.GetBonds():
        a, b = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if a in ring_of and b in ring_of and ring_of[a] != ring_of[b] \
                and keys[ring_of[a]] == keys[ring_of[b]]:
            return True
    return False


def _name_general_multiplicative(mol) -> Optional[str]:
    if mol is None or not _scope_ok(mol):
        return None
    rings = _ring_systems(mol)
    if _same_skeleton_rings_bonded(mol, rings):
        return None
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import get_principal_group
    pg, matches = get_principal_group(mol, detect_functional_groups(mol))
    if pg is not None and pg not in _HYDRIDE_PG:
        kernels = _pcg_kernels(mol, pg, matches, rings)
        if not kernels or len(kernels) < 2:
            return None
        groups: Dict[str, List[_Kernel]] = {}
        for k in kernels:
            if k.kind == "ring":
                k.ring_name = _ring_parent_name(mol, k.skel)
                if k.ring_name is None:
                    return None
            k.key = _kernel_key(mol, k)
            groups.setdefault(k.key, []).append(k)
        if len(groups) != 1:
            return None
        return _try_unit_set(mol, kernels, rings, kernels)
    # no principal characteristic group: the senior parent hydride
    from ..assembly.candidate_pool import _P_44_1_2_SENTINEL_RANK, P_44_1_2_ELEMENT_RANK
    for at in mol.GetAtoms():
        if at.GetAtomicNum() == 7 and not at.IsInRing() and not _is_nitro_n(at):
            return None
    hyd = _hydride_kernels(mol, rings)
    if hyd is None:
        return None
    ring_atoms = set().union(*rings) if rings else set()

    def rank(atoms):
        return min(P_44_1_2_ELEMENT_RANK.get(mol.GetAtomWithIdx(a).GetSymbol(),
                                             _P_44_1_2_SENTINEL_RANK) for a in atoms)
    top_ring = rank(ring_atoms) if ring_atoms else _P_44_1_2_SENTINEL_RANK
    top_hyd = min((rank(k.skel) for k in hyd), default=_P_44_1_2_SENTINEL_RANK)
    # any other acyclic heteroatom that could head a parent hydride declines
    for at in mol.GetAtoms():
        z = at.GetAtomicNum()
        if at.IsInRing() or z in (1, 6, 7, 8, 16, 34, 52, 9, 17, 35, 53):
            continue
        if not any(at.GetIdx() in k.skel for k in hyd):
            return None
    if top_hyd < top_ring:
        cands = [k for k in hyd if rank(k.skel) == top_hyd]
    else:
        if not rings:
            return None
        cands = []
        for rs in rings:
            if not _ring_kernel_ok(mol, rs):
                continue
            k = _Kernel("ring", rs, _cycle_order(mol, rs))
            if k.order is None:
                continue
            cands.append(k)
        if any(not _ring_kernel_ok(mol, rs) for rs in rings):
            return None
        cands = _senior_ring_units(mol, rings, cands)
        if cands is None:
            return None
        for k in cands:
            k.ring_name = _ring_parent_name(mol, k.skel)
            if k.ring_name is None:
                return None
    for k in cands:
        k.key = _kernel_key(mol, k)
    groups: Dict[str, List[_Kernel]] = {}
    for k in cands:
        groups.setdefault(k.key, []).append(k)
    found = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        for size in range(len(members), 1, -1):
            tried = 0
            hits = []
            for units in combinations(members, size):
                tried += 1
                if tried > _MAX_SUBSETS:
                    return None
                nm = _try_unit_set(mol, list(units), rings, members)
                if nm is not None:
                    hits.append(nm)
            if hits:
                if len(set(hits)) != 1:
                    return None
                found.append(hits[0])
                break
    if len(set(found)) != 1:
        return None
    return found[0]
