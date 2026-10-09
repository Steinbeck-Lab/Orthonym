"""Can a linear phane name be the PIN of this molecule? (2); fail closed).

``### **** Preferred IUPAC names in phane nomenclature`` (the Blue Book),
 (2) (:23829): "linear phanes consist of four or more rings or ring systems, two of
which must be terminal, and together with acyclic atoms or chains must consist of at least
seven nodes (components)"; (:23901): "Phane nomenclature is used to generate
preferred IUPAC names for ring assemblies and linear acyclic/cyclic compounds that include a
minimum of seven nodes including at least four rings or ring systems, two of which must be
terminal, even though the compounds could also be named by substitutive or multiplicative
nomenclature."

The phane parent still has to win first: "These criteria must always be applied before
those applicable to rings and ring systems (see and to chains (see ",
:18873), and a linear phane is one of the ring-system types ("(f) linear phane system",
:19541). A principal characteristic group whose carbonyl carbon and heteroatom are both skeletal
nodes (an acyclic -CO-NH-, -CO-O-) or whose nitrogen is a node (an -NH- between two rings) is cited
by a phane name as 'oxo' + 'aza'/'oxa' or as 'aza', never as the suffix: the phane parent cites
fewer suffixes than the substitutive parent and is not the PIN,:18875). Nilotinib,
whose every qualifying path runs through its amide carbon and nitrogen, keeps the benzamide PIN.
A ketone or pseudoketone carbon on the skeleton is different: the phane name cites it as '-one'
("an acyclic compound having a carbonyl group linked to a heteroatom belonging to a ring or ring
system",:1878;,:33127 "An N-acyl group attached to a nitrogen atom of a heterocyclic
system is now preferably named as a pseudoketone"), so such a group never takes the phane out of
the running (venadaparib, olaparib).

This screen answers True (a phane name may be the PIN: a substitutive name is then a general
name, never a certified PIN) unless it can show that no qualifying path exists, or that every
qualifying path takes in a non-ring atom of every principal-group instance and cites none of
them as '-one'. Simplified skeleton,:14857): each ring or ring system (fused, bridged,
spiro) is one node, rings joined by a bond stay separate nodes,:14911), every
acyclic heavy atom is one node; the condensed graph is a forest, so every path between two ring
nodes is an unbranched skeleton with two terminal rings. Whether each ring system is an allowed
amplificant is not checked: that only makes the screen answer True more often.

The label check of the program (rule id in ``validation/pin_spelling``) is the
general guard over every name; this screen keeps the substitutive producers of the drug lane
from building a name the book would replace by a phane name.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from rdkit import Chem

_MEMO_MOL = None
_MEMO: Dict[str, bool] = {}

#: an acyclic neighbour of these elements makes a skeletal carbonyl an amide, ester or acid
#: halide carbon, cited by a phane name as 'oxo' + a replacement prefix (:1878 "other than esters
#: or acid anhydrides... except for nitrogen and halogens")
_ABSORBING = frozenset({7, 8, 16, 34, 52, 9, 17, 35, 53})


def _ring_systems(mol) -> List[Set[int]]:
    systems: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        r = set(ring)
        merged = [s for s in systems if s & r]
        for s in merged:
            r |= s
            systems.remove(s)
        systems.append(r)
    return systems


def _qualifying_paths(mol) -> List[List[Tuple[str, int]]]:
    """Every path between two ring nodes with >= 4 ring nodes and >= 7 nodes."""
    systems = _ring_systems(mol)
    owner = {a: ("R", k) for k, s in enumerate(systems) for a in s}

    def node(i):
        return owner.get(i, ("A", i))
    adj: Dict[Tuple[str, int], Set[Tuple[str, int]]] = {}
    for b in mol.GetBonds():
        u, v = node(b.GetBeginAtomIdx()), node(b.GetEndAtomIdx())
        if u != v:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    rnodes = [("R", k) for k in range(len(systems))]
    if len(rnodes) < 4:
        return []
    paths = []
    for s in rnodes:
        prev = {s: None}
        order = [s]
        for x in order:
            for y in sorted(adj.get(x, ())):
                if y not in prev:
                    prev[y] = x
                    order.append(y)
        for t in rnodes:
            if t <= s or t not in prev:
                continue
            p, x = [], t
            while x is not None:
                p.append(x)
                x = prev[x]
            if sum(1 for q in p if q[0] == "R") >= 4 and len(p) >= 7:
                paths.append(p)
    return paths


def _cited_as_one(mol, idx: int) -> bool:
    """A skeletal carbon with a terminal =O, =S, =Se or =Te that a phane name cites as '-one'
    ('-thione',...): a ketone, or a pseudoketone whose heteroatom is a ring atom (:1878,
     :33127). A carbonyl with an acyclic N, O, S, Se, Te or halogen neighbour is an amide,
    ester or acid halide carbon and is not cited as '-one'."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return False
    oxo = {b.GetOtherAtomIdx(idx) for b in atom.GetBonds()
           if b.GetBondType() == Chem.BondType.DOUBLE
           and b.GetOtherAtom(atom).GetAtomicNum() in (8, 16, 34, 52)
           and b.GetOtherAtom(atom).GetDegree() == 1}
    if not oxo:
        return False
    return not any(n.GetIdx() not in oxo and n.GetAtomicNum() in _ABSORBING and not n.IsInRing()
                   for n in atom.GetNeighbors())


def _compute(mol) -> bool:
    paths = _qualifying_paths(mol)
    if not paths:
        return False
    try:
        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        pg, matches = get_principal_group(mol, detect_functional_groups(mol))
    except Exception:  # noqa: BLE001 - unknown: fail closed
        return True
    if not pg or not matches:
        return True  # no principal group: does not decide; the phane may win
    ring_info = mol.GetRingInfo()
    instances = []
    for m in matches:
        acyclic = {a for a in m if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
                   and ring_info.NumAtomRings(a) == 0}
        if not acyclic:
            return True  # a ring-borne group (lactam, ring ketone): the phane may cite it
        instances.append(acyclic)
    for p in paths:
        nodes = {q[1] for q in p if q[0] == "A"}
        for inst in instances:
            on_path = inst & nodes
            if not on_path:
                return True  # this path leaves the instance free to be cited as a suffix
            if any(_cited_as_one(mol, a) for a in on_path):
                return True  # a ketone or pseudoketone carbon on the skeleton: cited '-one'
    return False


def phane_may_be_pin(mol) -> bool:
    """True when a linear phane name may be the PIN (2)); fails closed (True) on
    error. Memoised per molecule object."""
    global _MEMO_MOL, _MEMO
    if _MEMO_MOL is not mol:
        _MEMO_MOL, _MEMO = mol, {}
    if "v" not in _MEMO:
        try:
            _MEMO["v"] = _compute(mol)
        except Exception:  # noqa: BLE001
            _MEMO["v"] = True
    return _MEMO["v"]
