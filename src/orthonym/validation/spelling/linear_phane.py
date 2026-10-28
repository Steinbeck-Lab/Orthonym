""" (2): structures whose preferred IUPAC name is a linear phane name.

The reader behind the spelling check ```` (:mod:`.checks_phane`). Read from the
structure alone (RDKit); no engine naming code and no OPSIN, so that the check is independent
of the names it reads. Rules (the Blue Book, each line read with
``sed -n``):

* "Preferred IUPAC names in phane nomenclature" (:23824); (:23826) "For the
  purpose of selecting preferred IUPAC names cyclic and acyclic phane systems are defined as
  follows:" (2) (:23829) "linear phanes consist of four or more rings or ring systems, two of
  which must be terminal, and together with acyclic atoms or chains must consist of at least
  seven nodes (components)."
* (:23901) "Phane nomenclature is used to generate preferred IUPAC names for ring
  assemblies and linear acyclic/cyclic compounds that include a minimum of seven nodes including
  at least four rings or ring systems, two of which must be terminal, even though the compounds
  could also be named by substitutive or multiplicative nomenclature."
* (:24088) "Phane names are preferred IUPAC names rather than ring assembly names when
  seven or more rings or ring systems are present."
* Nodes,:14857): "a numerical term 'di', 'tri', 'tetra', etc. indicating the number of
  nodes (including those designating superatoms), and the term phane. No prefix is used to name
  linear phanes." Every acyclic skeletal atom of the chain is a node (:23959 four benzenes and
  three O: a heptaphane;:26409 "four benzene rings and a total of seven phane nodes").
* Amplificants,:14904): mancude rings and ring systems, bridged fused systems,
  saturated monocycles, von Baeyer systems and spiro alkanes; not (a)(2),:14910)
  "spiro ring systems with at least one fused ring system or polycycloalkane ring system". A ring
  system is one node; the members of a ring assembly are separate nodes (:23917, a heptaphane of
  seven rings joined by bonds).
* Chain atoms: carbon or an element of the skeletal replacement list,:15246).
* The principal characteristic group,:18875) "The senior parent structure has the
  maximum number of substituents corresponding to the principal characteristic group (suffix)",
  applied before the ring criteria (:18873) and between phane parents (:18915: "there are two of
  the principal characteristic group in the PIN and only one in the other names"). The phane must
  cite at least as many of them as any other parent: each ring system, each ring assembly of
  identical ring systems,:15542,:15550) and each acyclic carbon chain of the component
  counts the groups whose suffix can sit on it -- a link that is a chain node of the phane among
  them (:26404 'N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)' cites the link N as a
  suffix N) -- and a group no ring or chain can carry is its own functional parent (urea,
  carbamic or carbonic acid). A group with a chain node of the phane in its key (its carbonyl or
  central atom, or a heteroatom of the group) cannot be a suffix of the phane: an acyclic carbonyl
  on a skeletal nitrogen is no pseudoketone (:1878 "except for nitrogen"), so the phane would cite
  an amide as 'oxo' + 'aza', or as an acyl prefix on an 'aza' node (reading R2 of the lane plan,
  controller decision 1). A ketone or pseudoketone carbon on the skeleton is cited
  ('-one',:1878 "a carbonyl group linked to a heteroatom belonging to a ring"); a carbonyl with
  two heteroatom neighbours (urea, carbamate, carbonate) is an amide or an ester of carbonic
  acid ('piperidine-1-carboxamide (PIN)',:32685; (b),:28263), never a pseudoketone
  unless the carbonyl is a ring atom or every heteroatom neighbour is a ring atom: between two
  ring nitrogens no acyclic nitrogen is left for an amide,:29312, a carbonyl joined to
  two heteroatoms none of which is an acyclic N;,:29370;,:33125;:29544
  'di(1H-imidazol-1-yl)methanethione (PIN)'), so the carbonyl is a skeletal '-one'. A tie goes
  to the phane;:26409 two amines in both names,:31927 one ester in both). When
  another parent cites more, the phane is no parent, but a qualifying chain that stays whole
  outside that parent is one of its substituents, which the PIN may cite as a linear phane
  prefix::19325 'trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane
  (PIN) (Si is senior to O)', the one printed PIN with such a prefix; (:16148)
  "Substituent prefixes derived from phane systems". No printed row decides it for a parent that
  wins by, so the name is not proven and the verdict stands (fail closed; controller
  ruling of 2026-10-05 on the lane's final review, finding F2). Each
  class holds its chalcogen and peroxy analogues (Table 4.1 7a:18172, class 15:18188, class 17
  :18190;:29504), and only the most senior suffix of Table 4.4 (:18597) is the
  principal characteristic group:29561 "C=O > C=S > C=Se > C=Te";
  :33386 carboxylic amides before urea). A multiplied parent is no rival: for more than two
  identical parent structures (:23242) forms the PIN "using phane nomenclature when
  four or more rings are present, two being terminal, in a system containing a minimum of seven
  nodes".
* The one Blue Book PIN row that meets the text of (2) without a phane name is all saturated
  (:3547, four cyclohexanes on a 9-node chain, a substitutive PIN), while every linear phane PIN
  the book prints has a ring system that is not saturated; so at least one ring system of the
  chain must hold an aromatic atom or a ring double bond (controller decision 2).
* An uncompensated charge or a radical classes 1-6) is not modelled; the structure test
  alone decides for such a component (fail closed: the substitutive name is not certified).
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

#: (:15246) the elements a phane skeleton names by 'a' prefixes, plus carbon
_CHAIN_ELEMENTS = frozenset({
    "C", "O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb",
    "B", "Al", "Ga", "In", "Tl"})

#: the numerical term of a phane skeleton of seven or more nodes,:14857;,
#: anchored at the end of the word before 'phane'; the leftmost match is the whole term
_NODE_TERM = re.compile(
    r"(?:(?:hen|do|un|tri|tetra|penta|hexa|hepta|octa|nona)?"
    r"(?:deca|icosa|cosa|(?:tria|tetra|penta|hexa|hepta|octa|nona)conta)"
    r"|hepta|octa|nona|hecta)$")
_PHANE_WORD = re.compile(r"([a-z]+)phan(?:e|-)")


def is_linear_phane_name(name: str) -> bool:
    """True when ``name`` is built on a linear phane parent hydride: an amplification prefix
    ending in 'a' (``benzena``,:14894) directly before the node count and 'phane'
    (``...tetrabenzenaheptaphane``, ``...heptaphan-7^4-yl``). A cyclophane (``...cyclo``
    before the count) and words such as 'phosphane' are not."""
    for match in _PHANE_WORD.finditer(name or ""):
        word = match.group(1)
        term = _NODE_TERM.search(word)
        if term is None:
            continue
        head = word[:term.start()]
        if head.endswith("a") and not head.endswith(("cyclo", "spiro")):
            return True
    return False


@dataclass(frozen=True)
class LinearPhaneVerdict:
    """The chain that makes a phane name the PIN. ``principal_class`` is the compound's senior
    suffix class (None: no suffix class; 'ion': an uncompensated charge or radical, not
    modelled); ``expressed`` of its ``occurrences`` are cited by the phane parent.
    ``as_prefix``: another parent cites more of them than the phane, or a parent with a
    senior atom outranks it, ``criterion``), and the chain (``ring_systems``, ``nodes``)
    stays whole outside that parent, so the PIN may cite it as a linear phane substituent prefix
    (:19325;,:16148). ``criterion`` is '' when the two other parents are identical and hold every
    principal group: the PIN is then a multiplicative name that cites the phane as its central group
    ('3,3'-[...phane-1^4,7^4-diyl]dipropanoic acid'), not a name with the phane in a prefix."""

    ring_systems: int
    nodes: int
    principal_class: Optional[str]
    expressed: int
    occurrences: int
    as_prefix: bool = False
    criterion: str = "P-44.1.1"          # the criterion that makes the phane no parent: for a senior atom


@dataclass(frozen=True)
class _RingSystem:
    atoms: FrozenSet[int]
    amplificant: bool       # (a)(2): not a spiro system with a fused component
    unsaturated: bool       # an aromatic atom or a ring double bond (the:3547 boundary)


def _groups(rings: Sequence[FrozenSet[int]], min_shared: int) -> List[List[int]]:
    """Indices of ``rings`` joined (transitively) when two share ``min_shared`` atoms or more."""
    parent = list(range(len(rings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= min_shared:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(rings)):
        groups[find(i)].append(i)
    return sorted(groups.values())


def _ring_systems(mol: Chem.Mol) -> List[_RingSystem]:
    """Ring systems: rings sharing an atom. Within one, the blocks are the rings sharing two
    atoms or more (fused, bridged, von Baeyer parts); two or more blocks make a spiro system,
    and a spiro system with a block of more than one ring is no amplificant."""
    rings = [frozenset(r) for r in mol.GetRingInfo().AtomRings()]
    out = []
    for idxs in _groups(rings, 1):
        members = [rings[i] for i in idxs]
        blocks = _groups(members, 2)
        atoms = frozenset().union(*members)
        spiro_with_polycycle = len(blocks) > 1 and any(len(b) > 1 for b in blocks)
        unsaturated = any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms) or any(
            b.IsInRing() and b.GetBondType() != Chem.BondType.SINGLE
            and b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms
            for b in mol.GetBonds())
        out.append(_RingSystem(atoms, not spiro_with_polycycle, unsaturated))
    return out


Node = Tuple[str, int]  # ('R', ring-system index) or ('A', atom index)


def _chains(mol: Chem.Mol, systems: Sequence[_RingSystem]) -> List[Tuple[Node, ...]]:
    """Every unbranched path of the simplified skeleton between two ring systems that meets
     (2): at least four ring systems and seven nodes, every ring system an allowed
    amplificant, every chain atom an 'a' element, at least one ring system not saturated. The
    simplified graph is a forest (two ring systems joined twice would be one ring system), so
    the path between two ring systems is unique."""
    node_of: Dict[int, Node] = {}
    for k, rs in enumerate(systems):
        for a in rs.atoms:
            node_of[a] = ("R", k)
    for atom in mol.GetAtoms():
        node_of.setdefault(atom.GetIdx(), ("A", atom.GetIdx()))
    adj: Dict[Node, Set[Node]] = defaultdict(set)
    for b in mol.GetBonds():
        u, v = node_of[b.GetBeginAtomIdx()], node_of[b.GetEndAtomIdx()]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    def allowed(node: Node) -> bool:
        if node[0] == "R":
            return systems[node[1]].amplificant
        return mol.GetAtomWithIdx(node[1]).GetSymbol() in _CHAIN_ELEMENTS

    out: List[Tuple[Node, ...]] = []
    ends = [("R", k) for k in range(len(systems))]
    for s in ends:
        prev: Dict[Node, Optional[Node]] = {s: None}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)
        for t in ends:
            if t <= s or t not in prev:
                continue
            path: List[Node] = []
            x: Optional[Node] = t
            while x is not None:
                path.append(x)
                x = prev[x]
            n_rings = sum(1 for p in path if p[0] == "R")
            if n_rings < 4 or len(path) < 7 or not all(allowed(p) for p in path):
                continue
            if not any(systems[p[1]].unsaturated for p in path if p[0] == "R"):
                continue
            out.append(tuple(reversed(path)))
    return out


# ---------------------------------------------------------------- principal characteristic group
# (:18162) classes 7-20, each with its chalcogen and peroxy analogues: (seniority,
# class, SMARTS, key atoms, centre, where the suffix sits, two heteroatom neighbours, analogue
# atoms). The list is the claim order (the senior reading of an atom claims it first); the
# seniority is (Table 4.1 class, Table 4.4 entry) plus the analogue order, so that only the most
# senior suffix is the principal characteristic group:18875 "the principal
# characteristic group (suffix)"):
# * Table 4.4 (:18597) "Complete list of suffixes and functional replacement analogues... in
# decreasing order of seniority": 1 carboxylic acids (:18599; carboperoxoic acids and their
# chalcogen analogues:18603, then "Carboxylic acids modified by replacement with S, Se, and/or
# Te":18615), 2 carboximidic (:18625), 3 carbohydrazonic (:18657), 4 sulfonic (:18685),
# 9 sulfinic (:18730), 12-15 selenonic, seleninic, telluronic, tellurinic (:18754-:18758),
# 16 carboxamides and their chalcogen analogues (:18759,:18761), 17 carboximidamides (:18764),
# 19 sulfonamides (:18769), 24 sulfinamides (:18782), 31 carbohydrazides (:18791),
# 34 sulfonohydrazides (:18799), 46 nitriles (:18822), 47 aldehydes (:18823; -CHS
# 'carbothialdehyde':18825), 48 ketones (:18831; '>(C)=S thione':18833), 49 hydroxy compounds
# (:18836; '-SH thiol':18838), 50 hydroperoxides (:18841; -OSH, -SOH:18843-:18844),
# 51 amines, 52 imines.
# * Table 4.1 7a (:18172) "Chalcogen analogues follow each of the corresponding oxygen acids and, in
# each case, the chalcogen analogue with the greater number of the preferred chalcogen atom
# (O > S > Se > Te), considered first in -OOH groups, then in -OH groups as necessary"; 7c
# (:18176) the phosphorus acids after them; class 15 (:18188) "Aldehydes and chalcogen
# analogues"; class 17 (:18190) "Hydroxy compounds and chalcogen analogues"; (:29561)
# "The order of seniority of ketonic suffixes is C=O > C=S > C=Se > C=Te."
# * Amides (class 11,:18184) and hydrazides (class 12,:18185) "in the order of the corresponding
# acids"; (:33386) "Thus, in substitutive nomenclature, amides from carboxylic
# acids, including formamide, are senior to urea" (urea, guanidine: amides of carbonic acid,
# Table 4.1 7b:18174; phosphorus amides 7c; sulfamide, of sulfuric acid, 7d); esters
# (class 9) and anhydrides (class 8) are one class each.
# 'neighbour': the suffix sits on a skeletal neighbour of the centre ('-carboxylic acid',
# '-amine', '-ol'); 'self': the centre itself is the skeletal atom ('-one', '-imine'). A carbonyl
# (or C=N, SO2) with two heteroatom neighbours keeps both in its key (carbonate and carbamate:
# esters, class 9; urea "an amide of carbonic acid",:33320, guanidine, sulfamide:
# amides, class 11), so that the reading does not depend on which neighbour a match lists first;
# when all of them are ring atoms the group is read as on a ring heteroatom (:29312,:33125).
# Analogue atoms: the match indices of the acidic -X(-X)H part and of the =X atoms (None: the
# class is read as one, esters and anhydrides); 'peroxy' marks an -X-XH acid.
_X = "O,S,Se,Te"
#: the N' of a hydrazide (Table 4.1 class 12,:18185; Table 4.4 -CO-NHNH2:18791): not the
#: nitrogen of a nitroso or nitro group (a double bond to O;:35880 "amides and hydrazides of
#: nitric and nitrous acids are now systematically based on nitric or nitrous amide", so an
#: N-nitroso or N-nitro carboxamide is a class-11 amide with that group as a prefix), not a
#: charged nitrogen, not a ring nitrogen (aromatic or not: a ring atom cannot be part of the
#: suffix group, so the amide is N-(azol-1-yl) or N-(piperidin-1-yl); a reading, no printed
#: row has such a compound), and not the start of a longer nitrogen chain:24906
#: 'N-(triazan-1-yl)benzamide (PIN)',:24928 'N,N'-(hydrazine-1,2-diyl)dibenzamide (PIN)').
#: An N'-hydroxy or N'-alkoxy group keeps the hydrazide, as an N-hydroxy group keeps the amide
#: (:17572 'N-hydroxypropanamide (PIN)').
_N_PRIME = "[#7;+0;!R;!$([#7]=[#8]);!$([#7](~[#7])~[#7])]"
_GROUPS = [
    # class 7 acids; a peroxy acid has no -XH on its centre, so the two readings never share one
    ((7, 1), "acid", f"[CX3](=[{_X}])[{_X};X2][{_X};X2H1]",
     (0,), 0, "neighbour", False, ((2, 3), (1,), "peroxy")),
    ((7, 1), "acid", f"[CX3](=[{_X}])[{_X};X2H1]",
     (0,), 0, "neighbour", False, ((2,), (1,), "acid")),
    ((7, 3), "acid", f"[CX3;!R](=[NX2][NX3])[{_X};X2H1]",
     (0,), 0, "neighbour", False, ((3,), (), "acid")),
    ((7, 2), "acid", f"[CX3;!R](=[NX2])[{_X};X2H1]",
     (0,), 0, "neighbour", False, ((2,), (), "acid")),
] + [
    row
    for element, onic, inic in (("S", 4, 9), ("Se", 12, 13), ("Te", 14, 15))
    for row in (
        ((7, onic), "acid", f"[{element}X4](=[{_X}])(=[{_X}])[{_X};X2][{_X};X2H1]",
         (0,), 0, "neighbour", False, ((3, 4), (1, 2), "peroxy")),
        ((7, onic), "acid", f"[{element}X4](=[{_X}])(=[{_X}])[{_X};X2H1]",
         (0,), 0, "neighbour", False, ((3,), (1, 2), "acid")),
        ((7, inic), "acid", f"[{element}X3](=[{_X}])[{_X};X2][{_X};X2H1]",
         (0,), 0, "neighbour", False, ((2, 3), (1,), "peroxy")),
        ((7, inic), "acid", f"[{element}X3](=[{_X}])[{_X};X2H1]",
         (0,), 0, "neighbour", False, ((2,), (1,), "acid")),
    )
] + [
    ((7, 16), "acid", f"[PX4](=[{_X}])[{_X};X2H1]",
     (0,), 0, "neighbour", False, ((2,), (1,), "acid")),
    ((8, 0), "anhydride", f"[CX3](=[{_X}])[{_X};X2][CX3](=[{_X}])",
     (0, 2, 3), 0, "neighbour", False, None),
    ((9, 0), "ester", f"[{_X};X2][CX3](=[{_X}])[{_X};X2]", (1, 0, 3), 1, "neighbour", True, None),
    ((9, 0), "ester", f"[NX3][CX3](=[{_X}])[{_X};X2][#6]", (1, 0, 3), 1, "neighbour", True, None),
    # an ester of a carboxylic or peroxy acid, its chalcogen analogues, a formate (H on the C)
    ((9, 0), "ester", f"[CX3;$(C-[#6,#7]),H1](=[{_X}])[{_X};X2][{_X};X2][#6]",
     (0, 2, 3), 0, "neighbour", False, None),
    ((9, 0), "ester", f"[CX3;$(C-[#6,#7]),H1](=[{_X}])[{_X};X2][#6]",
     (0, 2), 0, "neighbour", False, None),
    ((10, 1), "acid halide", f"[CX3](=[{_X}])[F,Cl,Br,I]",
     (0,), 0, "neighbour", False, ((), (1,), None)),
    # a urea, guanidine or sulfamide on a ring N is that ring's '-carboxamide', '-carboximidamide',
    # '-sulfonamide' ('piperidine-1-carboxamide (PIN)',:32685); on acyclic N only it is its own
    # functional parent (_FUNCTIONAL_PARENT_AMIDE)
    ((11, 16), "amide", f"[NX3][CX3](=[{_X}])[NX3]",
     (1, 0, 3), 1, "neighbour", True, ((), (2,), None)),
    ((11, 17), "amide", "[NX3][CX3;!R](=[NX2])[NX3]", (1, 0, 3), 1, "neighbour", True, None),
    ((11, 19), "amide", "[NX3][SX4](=O)(=O)[NX3]", (1, 0, 4), 1, "neighbour", True, None),
    # hydrazides (an N'-ylidene one, an acylhydrazone, too) before the amide rows, which would
    # read their C(=X)N; the N' (_N_PRIME) is an acyclic, uncharged amino nitrogen
    ((12, 31), "hydrazide", f"[#6][CX3](=[{_X}])[NX3]{_N_PRIME}",
     (1, 3, 4), 1, "neighbour", False, ((), (2,), None)),
    ((12, 34), "hydrazide", f"[#6][SX4](=O)(=O)[NX3]{_N_PRIME}",
     (1, 4, 5), 1, "neighbour", False, None),
    ((11, 16), "amide", f"[CX3](=[{_X}])[NX3]", (0, 2), 0, "neighbour", False, ((), (1,), None)),
    ((11, 19), "amide", f"[SX4](=[{_X}])(=[{_X}])[NX3]",
     (0, 3), 0, "neighbour", False, ((), (1, 2), None)),
    ((11, 24), "amide", f"[SX3](=[{_X}])[NX3]", (0, 2), 0, "neighbour", False, ((), (1,), None)),
    ((11, 17), "amide", "[CX3;!R](=[NX2])[NX3]", (0, 2), 0, "neighbour", False, None),
    ((11, 32), "amide", f"[PX4](=[{_X}])[NX3]", (0, 2), 0, "neighbour", False, ((), (1,), None)),
    ((14, 46), "nitrile", "[CX2]#[NX1]", (0,), 0, "neighbour", False, None),
    ((15, 47), "aldehyde", f"[CX3H1](=[{_X}])[#6]", (0,), 0, "neighbour", False, ((), (1,), None)),
    ((16, 48), "ketone", f"[#6X3]=[{_X};X1]", (0,), 0, "self", False, ((), (1,), None)),
    ((17, 49), "hydroxy", f"[{_X};X2H1][#6]", (0,), 0, "neighbour", False, ((0,), (), None)),
    ((18, 50), "hydroperoxide", "[#6][OX2][O,S;X2H1]",
     (1, 2), 1, "neighbour", False, ((1, 2), (), None)),
    ((18, 50), "hydroperoxide", "[#6][SX2][OX2H1]",
     (1, 2), 1, "neighbour", False, ((1, 2), (), None)),
    ((19, 51), "amine", "[NX3;!a;!R;!$(N-[#6,#16,#15]=[O,S]);!$(N-[#6;!R]=N);!$(N-[N,O]);!$(N-[#6]#N);+0]([#6])",
     (0,), 0, "neighbour", False, None),
    ((20, 52), "imine", "[CX3]=[NX2;!R;!$(N-O);!$(N-N)]", (0, 1), 0, "self", False, None),
]
_COMPILED = [(r, c, Chem.MolFromSmarts(s), k, ce, w, x2, an)
             for r, c, s, k, ce, w, x2, an in _GROUPS]

#: a urea or guanidine (amides of carbonic acid, Table 4.1 7b:18174) and a sulfamide (of sulfuric
#: acid, 7d:18178) on acyclic nitrogens only, amides "in the order of the corresponding acids"
#: (class 11,:18184): a urea or guanidine (11, 31) after every suffix amide of Table 4.4 (the
#: last, 30, at:18790) and before the phosphorus amides (7c:18176, (11, 32)); a sulfamide
#: (11, 33) after those. (:33386) "amides from carboxylic acids, including
#: formamide, are senior to urea"
_FUNCTIONAL_PARENT_AMIDE = {"C": (11, 31), "S": (11, 33)}

#: O > S > Se > Te (Table 4.1 7a,:18172)
_CHALCOGEN_RANK = {"O": 0, "S": 1, "Se": 2, "Te": 3}

#: seniority, class, key, centre, where, the atoms its suffix can sit on
_Occurrence = Tuple[Tuple[int, ...], str, Tuple[int, ...], int, str, Tuple[int, ...]]


def _analogue(symbols_xh: Sequence[str], symbols_dx: Sequence[str],
              kind: Optional[str]) -> Tuple[int, ...]:
    """The order of a chalcogen or peroxy analogue within its suffix (Table 4.1 7a,:18172; Table
    4.4,:18597): an unmodified oxygen acid, then its peroxy acids, then its chalcogen analogues;
    among analogues the greater number of O, then S, then Se, "considered first in -OOH groups,
    then in -OH groups"; for a single =X or -XH group simply O > S > Se > Te:29561).
    The last two places order the two spellings of one mixed -X-X'H pair: in a peroxy acid the
    atom of the -XH end first, O before S (Table 4.3 criterion (c),:18493 "oxygen atoms, then S,
    Se, and Te atoms, in -(O)OH and -OH groups": '-CO-SOH':18508 before '-CO-OSH':18509); in a
    hydroperoxide the atom bound to the parent first, as Table 4.4 lists them (-OSH:18843 before
    -SOH:18844)."""
    every = list(symbols_xh) + list(symbols_dx)
    if kind == "peroxy":
        tier = 1
    elif kind == "acid" and any(s != "O" for s in every):
        tier = 2
    else:
        tier = 0
    pair = list(symbols_xh)[::-1] if kind == "peroxy" else list(symbols_xh)
    pair = [_CHALCOGEN_RANK.get(s, 4) for s in pair][:2]
    pair += [0] * (2 - len(pair))
    return (tier, -list(symbols_xh).count("O"),
            -every.count("O"), -every.count("S"), -every.count("Se"), *pair)


def _attach(mol: Chem.Mol, key: Tuple[int, ...], sources: Sequence[int],
            two_hetero: bool) -> Tuple[int, ...]:
    """The atoms a suffix of the group can sit on: the carbon or ring neighbours of its centre
    (both carbonyl carbons of an anhydride) outside the key ('benzoic acid', 'aniline'); for a
    group with two heteroatom neighbours also a ring nitrogen of the key ('piperidine-1-
    carboxamide (PIN)',:32685)."""
    out: Set[int] = set()
    for source in sources:
        for n in mol.GetAtomWithIdx(source).GetNeighbors():
            if n.GetIdx() in key:
                if two_hetero and n.GetSymbol() == "N" and n.IsInRing():
                    out.add(n.GetIdx())
            elif n.GetSymbol() == "C" or n.IsInRing():
                out.add(n.GetIdx())
    return tuple(sorted(out))


def _occurrences(mol: Chem.Mol) -> List[_Occurrence]:
    """The suffix-class groups of ``mol``, each carbonyl or central atom counted once, senior
    pattern first. Within one pattern a nitrogen may serve two groups (an acyclic diacylamine is
    two amides), so that the reading does not depend on the order of the atoms; a nitrogen taken by
    one pattern is not read again by a later one. A cyclic anhydride, lactone, lactam
    or a carbonyl on a ring heteroatom is a ketone (Table 4.1 classes 8, 9, 11 send cyclic ones to
    class 16,:18181-:18189;:1878 pseudoketone); a carbonyl with two heteroatom neighbours is one
    when it is itself a ring atom (a cyclic urea, carbamate or carbonate) or when both neighbours
    are ring atoms (a urea or thiourea between two ring nitrogens::29312, no acyclic N;
    :33125 hidden amides), never when one of them is acyclic ('piperidine-1-carboxamide (PIN)',
    :32685). A C=NH that takes the same branch (an amidine on a ring nitrogen, a guanidine between
    two) is an imine, '-imine' on its carbon: Table 4.1 class 16 (:18189) is a carbonyl in every
    alternative, class 20 (:18193) "Imines, R=NH or R=N-R'". A sulfonyl on a ring nitrogen, also
    between two, has no suffix class. An amidine is acyclic: the exocyclic NH of a ring C=N (a
    2-amino-oxazoline) is an amine on a heterocycle (classes 19, 21). A thione, selone or tellone
    carbon read this way is a class-16 '-thione', '-selone', '-tellone',:29504), junior
    to '-one' (:29561)."""
    ring_info = mol.GetRingInfo()
    claimed: Set[int] = set()
    out: List[_Occurrence] = []

    def symbol(i: int) -> str:
        return mol.GetAtomWithIdx(i).GetSymbol()

    for rank, cls, patt, keys, centre_i, where, two_hetero, analogue in _COMPILED:
        seen: Set[FrozenSet[int]] = set()
        nitrogens: Set[int] = set()
        for match in mol.GetSubstructMatches(patt):
            key = tuple(sorted(match[i] for i in keys))
            if frozenset(key) in seen or any(a in claimed for a in key):
                continue
            seen.add(frozenset(key))
            for a in key:
                (nitrogens if symbol(a) == "N" else claimed).add(a)
            centre = match[centre_i]
            # a cyclic group, or one bound to ring heteroatoms only: '-one' (class 16), '-imine'
            # for a C=N carbon (class 20,:18193), or no suffix class for a sulfonyl
            ring_bound = False
            if two_hetero:
                ring_bound = bool(ring_info.NumAtomRings(centre)) or all(
                    ring_info.NumAtomRings(a) for a in key if a != centre)
            elif cls in ("anhydride", "ester", "amide", "hydrazide"):
                ring_bound = bool(ring_info.NumAtomRings(match[keys[1]]))
            if ring_bound:
                if symbol(centre) != "C":
                    continue
                # one '-one' per carbonyl carbon of the group (an anhydride has two); a carbon
                # double-bonded to N is an imine (Table 4.1 class 20,:18193)
                for a in key:
                    atom = mol.GetAtomWithIdx(a)
                    if atom.GetSymbol() != "C":
                        continue
                    double = [b.GetOtherAtom(atom).GetSymbol() for b in atom.GetBonds()
                              if b.GetBondType() == Chem.BondType.DOUBLE]
                    if "N" in double:
                        out.append(((20, 52) + _analogue((), (), None), "imine",
                                    (a,), a, "self", (a,)))
                    else:
                        out.append(((16, 48) + _analogue((), double, None), "ketone",
                                    (a,), a, "self", (a,)))
                continue
            if where == "self":
                attach: Tuple[int, ...] = (centre,)
            else:
                sources = [a for a in key if symbol(a) == "C"] if cls == "anhydride" else [centre]
                attach = _attach(mol, key, sources, two_hetero)
            base = rank
            if cls == "amide" and two_hetero and not attach:
                base = _FUNCTIONAL_PARENT_AMIDE[symbol(centre)]
            xh, dx, kind = analogue if analogue is not None else ((), (), None)
            seniority = base + _analogue([symbol(match[i]) for i in xh],
                                         [symbol(match[i]) for i in dx], kind)
            out.append((seniority, cls, key, centre, where, attach))
        claimed |= nitrogens
    return out


def _charged(mol: Chem.Mol, atoms: FrozenSet[int]) -> bool:
    """An uncompensated charge (no neighbour of the opposite sign: not a nitro group, an
    N-oxide, an azide) or a radical among ``atoms``."""
    for i in atoms:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetNumRadicalElectrons():
            return True
        q = atom.GetFormalCharge()
        if q and not any(n.GetFormalCharge() * q < 0 for n in atom.GetNeighbors()):
            return True
    return False


def _expressed(mol: Chem.Mol, chain: Tuple[Node, ...], systems: Sequence[_RingSystem],
               occurrences: Sequence[_Occurrence]) -> int:
    """How many of ``occurrences`` the phane on ``chain`` cites as its suffix: a 'self' group whose
    centre is a skeletal atom ('-one' on a ring or chain atom) and a 'neighbour' group bonded to
    the skeleton. A group with a chain node in its key (besides a 'self' centre) is part of the
    skeleton and is cited as 'oxo' + 'aza' (or 'aza'), not as the suffix."""
    chain_atoms = {p[1] for p in chain if p[0] == "A"}
    skeleton = set(chain_atoms)
    for p in chain:
        if p[0] == "R":
            skeleton |= systems[p[1]].atoms
    count = 0
    for rank, cls, key, centre, where, attach in occurrences:
        if any(a in chain_atoms for a in key if not (where == "self" and a == centre)):
            continue  # absorbed: it would be cited as 'a' + 'oxo', not as the suffix
        count += any(a in skeleton for a in attach)
    return count


def ring_system_hydride(mol: Chem.Mol,
                        atoms: FrozenSet[int]) -> Optional[Tuple[Chem.Mol, Dict[int, int]]]:
    """The ring system ``atoms`` with every bond to an outside atom replaced by hydrogen, and the map
    from an atom index in ``mol`` to its index in the hydride; None when that hydride cannot be
    sanitized. Isotopes are dropped 'identical cyclic systems')."""
    rw = Chem.RWMol(mol)
    for idx in atoms:
        atom = rw.GetAtomWithIdx(idx)
        extra = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds()
                    if b.GetOtherAtomIdx(idx) not in atoms)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + extra)
        atom.SetNoImplicit(True)
        atom.SetIsotope(0)
    for idx in sorted(set(range(mol.GetNumAtoms())) - atoms, reverse=True):
        rw.RemoveAtom(idx)
    try:
        frag = rw.GetMol()
        Chem.SanitizeMol(frag)
    except Exception:  # noqa: BLE001 -- RDKit raises several types for an impossible hydride
        return None
    return frag, {atom: i for i, atom in enumerate(sorted(atoms))}


def _identical_parent(mol: Chem.Mol, atoms: FrozenSet[int]) -> Optional[str]:
    """The ring system ``atoms`` as its hydride, a canonical SMILES 'identical cyclic
    systems'), or None when that hydride cannot be written (the system then joins no ring
    assembly)."""
    got = ring_system_hydride(mol, atoms)
    return None if got is None else Chem.MolToSmiles(got[0], isomericSmiles=False)


def _assemblies(mol: Chem.Mol, systems: Sequence[_RingSystem],
                atoms: FrozenSet[int]) -> List[FrozenSet[int]]:
    """The atoms of each ring assembly of the component,:15542: cyclic systems "directly
    joined to each other by single or double bonds";:15550: "Ring assemblies are composed of
    identical cyclic systems"): identical ring systems joined directly, two or more."""
    system_of = {a: k for k, rs in enumerate(systems) if rs.atoms <= atoms for a in rs.atoms}
    links: Dict[int, Set[int]] = defaultdict(set)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in system_of and j in system_of and system_of[i] != system_of[j]:
            links[system_of[i]].add(system_of[j])
            links[system_of[j]].add(system_of[i])
    parent = {k: _identical_parent(mol, systems[k].atoms) for k in links}
    out: List[FrozenSet[int]] = []
    done: Set[int] = set()
    for start in sorted(links):
        if start in done or parent[start] is None:
            continue
        group, queue = {start}, deque([start])
        while queue:
            for nxt in links[queue.popleft()]:
                if nxt not in group and parent[nxt] == parent[start]:
                    group.add(nxt)
                    queue.append(nxt)
        done |= group
        if len(group) > 1:
            out.append(frozenset().union(*(systems[k].atoms for k in group)))
    return out


def _acyclic_chains(mol: Chem.Mol, atoms: FrozenSet[int],
                    occurrences: Sequence[_Occurrence]) -> List[FrozenSet[int]]:
    """For each acyclic carbon part of the component (a tree), the unbranched chain through it
    that carries the most of ``occurrences`` (a chain parent of substitutive nomenclature)."""
    carbons = {a for a in atoms if mol.GetAtomWithIdx(a).GetSymbol() == "C"
               and not mol.GetAtomWithIdx(a).IsInRing()}
    marked = {a for o in occurrences for a in o[5] if a in carbons}
    adj = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in carbons]
           for a in carbons}
    out: List[FrozenSet[int]] = []
    for s in sorted(marked):
        prev: Dict[int, Optional[int]] = {s: None}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)
        for t in sorted(marked):
            if t < s or t not in prev:
                continue
            path: Set[int] = set()
            x: Optional[int] = t
            while x is not None:
                path.add(x)
                x = prev[x]
            out.append(frozenset(path))
    return out


def _rival_parents(mol: Chem.Mol, atoms: FrozenSet[int], systems: Sequence[_RingSystem],
                   principal: Sequence[_Occurrence]) -> Tuple[int, List[FrozenSet[int]]]:
    """ for the other parents of the component: the most ``principal`` groups one ring
    system, one ring assembly or one acyclic carbon chain can cite as suffixes, counting a group
    the phane absorbs ('pyrimidine-2,4-diamine': the link N is a suffix N,:26404); at least 1
    (a group no ring or chain can carry is cited by its own functional parent: urea, carbamic or
    carbonic acid). With it, the atoms of every parent that cites that many leaves
    them tied; the criteria after it are not read here), each with the key atoms of the groups
    it cites; when no ring or chain cites any, each group is its own functional parent."""
    parts = [rs.atoms for rs in systems if rs.atoms <= atoms]
    parts += _assemblies(mol, systems, atoms)
    parts += _acyclic_chains(mol, atoms, principal)
    counts = [sum(any(a in part for a in o[5]) for o in principal) for part in parts]
    best = max([0] + counts)
    if best == 0:
        return 1, [frozenset(o[2]) for o in principal]
    winners = [part.union(*(o[2] for o in principal if any(a in part for a in o[5])))
               for part, n in zip(parts, counts) if n == best]
    return best, winners


def _node_atoms(chain: Tuple[Node, ...], systems: Sequence[_RingSystem]) -> Set[int]:
    """The atoms of the nodes of ``chain``: its ring systems and its chain atoms."""
    out: Set[int] = set()
    for p in chain:
        out |= systems[p[1]].atoms if p[0] == "R" else {p[1]}
    return out


#: (:18917): "N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te > C", the
#: senior atom first
_SENIOR_ATOM_ORDER = ("N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl",
                      "O", "S", "Se", "Te", "C")
#: the elements of the order that head a parent hydride of their own (the book's rows:19325 silane,:19327
#: phosphane); a single N is decided by, where the amines, amides and nitriles that the reader models
#: are principal characteristic groups, a nitro group heads no parent hydride (ASSUMED; the sample's outside
#: nitrogen atoms are a nitro group, an azo group and an isocyanide), an acyclic N-N unit heads hydrazine or diazene
_PARENT_HYDRIDE_HEADS = frozenset({"P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl"})


def _senior_atoms_outside(mol: Chem.Mol, atoms: FrozenSet[int], systems: Sequence[_RingSystem],
                          chain: Tuple[Node, ...]) -> FrozenSet[int]:
    """ and (:18919) "A single senior atom is sufficient to give seniority to the parent
    hydride": the acyclic atoms of the component outside the chain whose element heads a parent hydride and
    ranks above every element of the phane (its chain atoms and the atoms of its ring systems; the
    skeletal heteroatoms of a phane are 'a' replacement, to which the criterion is not applied,:18917).
    A ring outside the chain is not counted: the criterion "is not used to choose between rings"."""
    inside = _node_atoms(chain, systems)
    rank = {e: i for i, e in enumerate(_SENIOR_ATOM_ORDER)}
    best = min(rank.get(mol.GetAtomWithIdx(a).GetSymbol(), len(rank)) for a in inside)

    def heads_parent(atom: Chem.Atom) -> bool:
        if atom.IsInRing():
            return False
        if atom.GetSymbol() in _PARENT_HYDRIDE_HEADS:
            return rank[atom.GetSymbol()] < best
        # an acyclic N-N unit heads hydrazine or diazene: (:18950) '1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine (PIN) (N > Si > O)'
        return (atom.GetSymbol() == "N" and rank["N"] < best
                and any(n.GetSymbol() == "N" and not n.IsInRing() for n in atom.GetNeighbors()))
    return frozenset(a for a in atoms - inside if heads_parent(mol.GetAtomWithIdx(a)))


def _cited_by_senior_parent(mol: Chem.Mol, principal: Sequence[_Occurrence], senior: FrozenSet[int]) -> int:
    """How many of the ``principal`` groups sit on a senior atom (a group bonded to it is cited by the parent
    hydride it heads)."""
    return sum(any(n.GetIdx() in senior for n in mol.GetAtomWithIdx(o[3]).GetNeighbors()) for o in principal)


def _unit_key(mol: Chem.Mol, unit: Set[int], link: Tuple[int, int]) -> str:
    """The unit as a canonical SMILES with a dummy atom where it is attached to the phane."""
    rw = Chem.RWMol(mol)
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(link[0], dummy, Chem.BondType.SINGLE)
    for index in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in unit and a.GetIdx() != dummy), reverse=True):
        rw.RemoveAtom(index)
    return Chem.MolToSmiles(rw.GetMol())


def _multiplicative(mol: Chem.Mol, skeleton: Set[int], principal: Sequence[_Occurrence]) -> bool:
    """ (:23178) "Multiplicative nomenclature is preferred to substitutive nomenclature for generating preferred
    IUPAC names to express multiple occurrences of identical parent structures... when (1) the linking bonds between the
    central substituent group of the multiplicative group and all subsequent structural units are identical...; and (3) the
    locants of all substituent groups on the identical parent structures, including suffix groups, are identical": the
    structure outside the phane ``skeleton`` is two units, each attached by one single bond, identical (the same SMILES
    with the attachment marked) and holding every one of the ``principal`` groups. Two units: with more identical parents
    the book sends the choice to phane nomenclature,:23242), which is not decided here."""
    centres = {o[3] for o in principal}
    units: List[Set[int]] = []
    seen: Set[int] = set(skeleton)
    for start in range(mol.GetNumAtoms()):
        if start in seen:
            continue
        unit, stack = {start}, [start]
        while stack:
            for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                if n.GetIdx() not in seen and n.GetIdx() not in unit:
                    unit.add(n.GetIdx())
                    stack.append(n.GetIdx())
        seen |= unit
        if unit & centres:
            units.append(unit)
    if len(units) != 2 or sum(len(u & centres) for u in units) != len(centres):
        return False
    keys = []
    for unit in units:
        links = [(a, n.GetIdx()) for a in unit for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in skeleton]
        if len(links) != 1 or mol.GetBondBetweenAtoms(*links[0]).GetBondType() != Chem.BondType.SINGLE:
            return False
        keys.append(_unit_key(mol, unit, links[0]))
    return keys[0] == keys[1]


def _component_result(mol: Chem.Mol, atoms: FrozenSet[int], systems: Sequence[_RingSystem],
                      chains: Sequence[Tuple[Node, ...]], occurrences: Sequence[_Occurrence]
                      ) -> Optional[Tuple[LinearPhaneVerdict, List[Tuple[Node, ...]]]]:
    """The verdict of one component and the chains that make its phane the parent (the longest of the usable ones)."""
    sizes = [(sum(p[0] == "R" for p in c), len(c)) for c in chains]
    if _charged(mol, atoms):
        best = max(sizes)
        return LinearPhaneVerdict(*best, "ion", 0, 0), [c for c, z in zip(chains, sizes) if z == best]
    mine = [o for o in occurrences if o[3] in atoms]
    if not mine:
        # has nothing to count; decides between the phane and a parent with a senior atom
        usable = [(c, z) for c, z in zip(chains, sizes) if not _senior_atoms_outside(mol, atoms, systems, c)]
        if not usable:
            return LinearPhaneVerdict(*max(sizes), None, 0, 0, as_prefix=True, criterion="P-44.1.2"), []
        best = max(z for _, z in usable)
        return LinearPhaneVerdict(*best, None, 0, 0), [c for c, z in usable if z == best]
    senior = min(o[0] for o in mine)
    principal = [o for o in mine if o[0] == senior]
    triples = [(_expressed(mol, c, systems, principal),) + z for c, z in zip(chains, sizes)]
    rivals, winners = _rival_parents(mol, atoms, systems, principal)
    # counts the groups of the parents; one that is a parent hydride with a senior atom and cites at
    # least as many as the phane takes the structure from it decides a tie, a lead)
    usable = []
    for c, z, t in zip(chains, sizes, triples):
        outside = _senior_atoms_outside(mol, atoms, systems, c)
        if outside and t[0] >= rivals and _cited_by_senior_parent(mol, principal, outside) >= t[0]:
            continue
        usable.append((c, z, t))
    if not usable:
        best = max(triples)
        return (LinearPhaneVerdict(best[1], best[2], principal[0][1], best[0], len(principal),
                                   as_prefix=True, criterion="P-44.1.2"), [])
    expressed, n_rings, n_nodes = max(t for _, _, t in usable)
    best_chains = [c for c, _, t in usable if t == (expressed, n_rings, n_nodes)]
    if expressed >= rivals:
        return LinearPhaneVerdict(n_rings, n_nodes, principal[0][1], expressed, len(principal)), best_chains
    #: another parent cites more of the principal characteristic group, so the phane is
    # no parent. A qualifying chain that stays whole outside one of the winning parents (any of
    # them: the criteria that choose between tied parents are not read here) is one substituent
    # of that parent, which the PIN may cite as a linear phane prefix (:19325;,:16148):
    # not proven either way, so the verdict stands (fail closed). A winning parent that splits
    # every qualifying chain (nilotinib's benzamide, the pyrimidine-2,4-diamine of decision 1)
    # leaves no linear phane to cite.
    whole = [size for parent in winners for c, size in zip(chains, sizes)
             if parent.isdisjoint(_node_atoms(c, systems))]
    if not whole:
        return None
    criterion = "P-51.3.1" if any(_multiplicative(mol, _node_atoms(c, systems), principal) for c in chains) else "P-44.1.1"
    return (LinearPhaneVerdict(*max(whole), principal[0][1], expressed, len(principal), as_prefix=True,
                               criterion=criterion), [])


def _component_verdict(mol: Chem.Mol, atoms: FrozenSet[int], systems: Sequence[_RingSystem],
                       chains: Sequence[Tuple[Node, ...]],
                       occurrences: Sequence[_Occurrence]) -> Optional[LinearPhaneVerdict]:
    got = _component_result(mol, atoms, systems, chains, occurrences)
    return None if got is None else got[0]


def principal_groups(mol: Chem.Mol) -> Tuple[Optional[str], Tuple[_Occurrence, ...]]:
    """The senior suffix class of ``mol`` and the occurrences of that class: the principal
    characteristic groups), read as ``_component_verdict`` reads them; ``(None, )`` when ``mol`` has no
    group of a suffix class. One occurrence is ``(seniority, class, key, centre, where, attach)``."""
    occurrences = _occurrences(mol)
    if not occurrences:
        return None, ()
    senior = min(o[0] for o in occurrences)
    principal = tuple(o for o in occurrences if o[0] == senior)
    return principal[0][1], principal


def _results(mol: Chem.Mol):
    """(verdict, chains of the phane) of each component of ``mol`` whose chains qualify, and the ring systems."""
    if mol is None:
        return [], []
    try:
        n_rings = mol.GetRingInfo().NumRings()
    except RuntimeError:  # an unsanitized molecule: find its rings first
        Chem.FastFindRings(mol)
        n_rings = mol.GetRingInfo().NumRings()
    if n_rings < 4:
        return [], []
    systems = _ring_systems(mol)
    if len(systems) < 4:
        return [], systems
    chains = _chains(mol, systems)
    if not chains:
        return [], systems
    occurrences = _occurrences(mol)
    results = []
    for component in Chem.GetMolFrags(mol):
        atoms = frozenset(component)
        mine = [c for c in chains if systems[c[0][1]].atoms <= atoms]
        if mine:
            got = _component_result(mol, atoms, systems, mine, occurrences)
            if got is not None:
                results.append(got)
    return results, systems


def _winner(results):
    return max(results, key=lambda r: (not r[0].as_prefix, r[0].expressed, r[0].ring_systems, r[0].nodes,
                                       r[0].principal_class or "", r[0].occurrences))


def linear_phane_pin_expected(mol: Chem.Mol) -> Optional[LinearPhaneVerdict]:
    """The verdict for the connected component of ``mol`` (a sanitized molecule) whose PIN is
    a linear phane name, or may cite one as a substituent prefix (``as_prefix``) -- each
    component read alone, the largest verdict wins, a phane parent before a prefix -- or None
    when no component's PIN is or may cite one."""
    results, _ = _results(mol)
    return _winner(results)[0] if results else None


def linear_phane_chains(mol: Chem.Mol) -> List[Tuple[Node, ...]]:
    """The chains that make the phane the parent of the winning component: every chain that ties on what the
    verdict counts (principal groups, rings, nodes), in no particular order; empty when the verdict is None
    or only a prefix. A ring-system node is ``("R", k)``, an acyclic atom ``("A", i)``; ``k`` indexes
    :func:`ring_system_atoms`."""
    results, _ = _results(mol)
    if not results:
        return []
    return list(_winner(results)[1])


def linear_phane_chain(mol: Chem.Mol) -> Optional[Tuple[Node, ...]]:
    """One of:func:`linear_phane_chains` (the first by node ids), or None."""
    chains = linear_phane_chains(mol)
    return min(chains) if chains else None


def linear_phane_candidate_chains(mol: Chem.Mol) -> List[Tuple[Node, ...]]:
    """The chains of the longest kind that meet (2) in ``mol`` (most ring systems, then most nodes), whichever
    parent the PIN has: the chains a substituent group derived from a linear phane is built on,:16148).
    Empty when the structure has none."""
    if mol is None:
        return []
    try:
        n_rings = mol.GetRingInfo().NumRings()
    except RuntimeError:
        Chem.FastFindRings(mol)
        n_rings = mol.GetRingInfo().NumRings()
    if n_rings < 4:
        return []
    systems = _ring_systems(mol)
    if len(systems) < 4:
        return []
    chains = _chains(mol, systems)
    if not chains:
        return []
    sizes = [(sum(p[0] == "R" for p in c), len(c)) for c in chains]
    best = max(sizes)
    return [c for c, z in zip(chains, sizes) if z == best]


_SYSTEMS_MOL: Optional[Chem.Mol] = None
_SYSTEMS: List[_RingSystem] = []


def ring_system_atoms(mol: Chem.Mol, k: int) -> FrozenSet[int]:
    """The atoms of the ``k``-th ring system of the reader (the ``("R", k)`` nodes of a chain). The ring systems
    of the last molecule are kept: the skeleton of each of many tied chains asks for them again."""
    global _SYSTEMS_MOL, _SYSTEMS
    if _SYSTEMS_MOL is not mol:
        _SYSTEMS_MOL, _SYSTEMS = mol, _ring_systems(mol)
    return _SYSTEMS[k].atoms
