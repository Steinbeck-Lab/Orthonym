"""A di- or polyanhydride name read back through its acids.

OPSIN 2.9.0 has no grammar for the class words of a poly-anhydride: it rejects ``diacetic butanedioic
dianhydride`` ("Unexpected number of words in anhydride") and the multiplied chalcogen word of
``diacetic butanedioic bis(thioanhydride)``. Until this module such a name shipped with no read-back at all
(the ``dianhydride`` and ``chalcogen_dianhydride`` grammar-gap carve-outs), and 116 of the 208 names the
default tier gave for the PubChem rows that reach the carve-out were the name of another molecule (an acid
residue named as a chain: a hydroxy group, a double bond, a ring or a stereo centre of it lost).

 Di- and polyanhydrides (the Blue Book): "Di- and polyanhydrides have two or more
-CO-O-CO- or related groups, such as -SO2-O-SO2-, respectively. They are named using the class name
'dianhydride', 'trianhydride', etc., preceded by the names of the acid groups cited as separate words."
 Mixed anhydrides (:32253) cites an acid "without the class name 'acid'", so each word of the name is
an acid name without its 'acid', and OPSIN reads ``<word> acid``. (:32380) cites the acid groups
"in their order of occurrence in the structure... The numerical prefix 'di-' is used", 'diacetic butanedioic
dianhydride (PIN)' (:32386); (:32443) gives the chalcogen analogues 'bis(thioanhydride)'
('diacetic butanedioic bis(thioanhydride) (PIN)',:32446); (:36925) 'tetraacetic hypodiphosphoric
tetraanhydride (PIN)' (:36933) names the anhydrides of a polynuclear non-carbon oxoacid with the same words.

The name is read back as follows, with nothing taken from the producer of the name.

1. Each word is split into a multiplier and an acid name in every way the multiplier spellings allow
   (``bis(X)``, ``di(X)``, ``diX`` and the word as it stands), and OPSIN reads ``<acid> acid`` to a structure
   (and reads it once: no ``APPEARS_AMBIGUOUS`` warning, Citation of locants,:2869).
2. The input is split at each of its anhydride links, an oxygen (or chalcogen) atom between an acyl carbon and
   a second acyl or oxoacid centre, into the acids it would give by hydrolysis; the acids of the reading and
   these must be the same set (full InChIKeys, so a lost stereo centre or double bond shows here), and the
   number of links and the bridge element must be the ones the class word gives.
3. The poly-anhydride is built from the acids of the reading, every way the acids can be joined into a tree
   with one link between two acid groups (an OH of each, lost) and a bridge atom of the class word, and each
   of these structures must have the FULL InChIKey of the input. A name that several structures answer to (an
   unsymmetrical diacid between two different acids, with no locants to say which way round it lies) is no
   proof of the input: it is not verified.
4. The descriptors of each acid, labelled as a free acid, must be those the same atoms have in the whole structure
   (``perception.stereo.assign_stereochemistry``;, the Blue Book): a word that states the free-acid
   configuration of an atom the link makes another configuration is the name of the same structure with a wrong
   descriptor, and it is not verified.

A name ships when the check holds; the check proves the structure and never the spelling (the order of the
words, the multiplier spelling and the PIN status belong to the producer and to the label rules).
"""
import itertools
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from .opsin_ambiguity import ambiguity_messages

#: the rule id of the check's outcome (``acid_split_verified:anhydride_acids``)
ANHYDRIDE_ACIDS_RULE = "anhydride_acids"

#: the most acids of one poly-anhydride (five links) and the most structures built for one name: a larger one is
#: not checked (the name is not verified, never guessed)
_MAX_ACIDS = 6
_MAX_BUILDS = 400

_NUMERIC = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}
_COMPLEX = {"bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5, "hexakis": 6}
_BRIDGE_Z = {"": 8, "thio": 16, "seleno": 34, "telluro": 52}
_CLASS_NUMERIC = re.compile(r"(?P<n>di|tri|tetra|penta|hexa)anhydride")
_CLASS_CHALCOGEN = re.compile(r"(?P<n>bis|tris|tetrakis|pentakis|hexakis)\((?P<e>thio|seleno|telluro)anhydride\)")
_MULTIPLIED = re.compile(r"(?P<m>bis|tris|tetrakis|pentakis|hexakis|di|tri|tetra|penta|hexa)(?P<rest>.+)")

#: an acid group: the OH of a carboxyl group or of an oxoacid centre (S, P, As, Se, Te) that holds a double-bonded O
_ACID_GROUP = Chem.MolFromSmarts("[OX2H1]-[#6,#15,#16,#33,#34,#52;$([*]=[OX1])]")
#: an anhydride link: a bridge atom between an acyl carbon and a second acyl carbon or oxoacid centre
_LINK = Chem.MolFromSmarts(
    "[#6;$([#6]=[OX1])]-[OX2,SX2,SeX2,TeX2]-[#6,#15,#16,#33,#34,#52;$([*]=[OX1])]")


def _key(mol) -> Optional[str]:
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:
        return None


def _class_of(word: str) -> Optional[Tuple[int, int]]:
    """``(links, bridge atomic number)`` of a class word, else None."""
    found = _CLASS_NUMERIC.fullmatch(word)
    if found is not None:
        return _NUMERIC[found["n"]], _BRIDGE_Z[""]
    found = _CLASS_CHALCOGEN.fullmatch(word)
    if found is not None:
        return _COMPLEX[found["n"]], _BRIDGE_Z[found["e"]]
    return None


def is_poly_anhydride_name(name: str) -> bool:
    """Whether ``name`` has the words of a di- or poly-anhydride name: two or more acid words and a class word."""
    words = (name or "").split(" ")
    return len(words) >= 3 and _class_of(words[-1]) is not None


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += (char in "([{") - (char in ")]}")
        if depth < 0:
            return False
    return depth == 0


def _readings(word: str) -> List[Tuple[int, str]]:
    """``(count, acid name)`` for every way ``word`` can be a multiplied acid name: the word as it stands,
    and after a multiplier with or without the enclosing marks of its acid."""
    out = [(1, word)]
    found = _MULTIPLIED.fullmatch(word)
    if found is not None:
        count = _COMPLEX.get(found["m"]) or _NUMERIC[found["m"]]
        rest = found["rest"]
        out.append((count, rest))
        if rest[0] in "([{" and rest[-1] in ")]}" and _balanced(rest[1:-1]):
            out.append((count, rest[1:-1]))
    return out


def _acid(stem: str, memo: Dict[str, object]):
    """OPSIN's structure of ``<stem> acid`` (read once), or None."""
    if stem not in memo:
        from ..namer import _validity_gate_name_to_smiles
        full = f"{stem} acid"
        smiles = _validity_gate_name_to_smiles(full)
        memo[stem] = (Chem.MolFromSmiles(smiles)
                      if smiles and not ambiguity_messages(full) else None)
    return memo[stem]


def _input_acids(mol, bridge_z: int) -> Optional[Tuple[int, List[str]]]:
    """``(links, sorted InChIKeys of the acids)`` of the input split at its anhydride links, or None when a
    link has a bridge of another element than the class word's."""
    bridges = sorted({match[1] for match in mol.GetSubstructMatches(_LINK)})
    rw = Chem.RWMol(mol)
    for bridge in bridges:
        atom = rw.GetAtomWithIdx(bridge)
        if atom.GetAtomicNum() != bridge_z:
            return None
        first, second = [n.GetIdx() for n in atom.GetNeighbors()]
        rw.RemoveBond(bridge, second)
        atom.SetAtomicNum(8)
        atom.SetNoImplicit(False)
        rw.AddBond(second, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    try:
        pieces = rw.GetMol()
        Chem.SanitizeMol(pieces)
        keys = [_key(piece) for piece in Chem.GetMolFrags(pieces, asMols=True)]
    except Exception:
        return None
    return (len(bridges), sorted(keys)) if None not in keys else None


def _canonical_tree(edges: Sequence[Tuple[int, int]], labels: Sequence[str]) -> str:
    """A canonical string of a labelled tree (the smallest rooted encoding)."""
    adjacent = defaultdict(list)
    for left, right in edges:
        adjacent[left].append(right)
        adjacent[right].append(left)

    def encode(node: int, parent: int) -> str:
        return "(" + labels[node] + "".join(sorted(
            encode(n, node) for n in adjacent[node] if n != parent)) + ")"
    return min(encode(root, -1) for root in range(len(labels)))


def _trees(labels: Sequence[str], capacity: Sequence[int]):
    """Every tree on the acids (one link between two acids, an acid taking no more links than it has acid
    groups), each shape once."""
    count = len(labels)
    seen = set()
    for sequence in itertools.product(range(count), repeat=count - 2):
        degree = [1] * count
        for node in sequence:
            degree[node] += 1
        if any(degree[i] > capacity[i] for i in range(count)):
            continue
        leaves = sorted(i for i in range(count) if degree[i] == 1)
        edges, remaining = [], list(sequence)
        for node in remaining:
            leaf = leaves.pop(0)
            edges.append((leaf, node))
            degree[node] -= 1
            if degree[node] == 1:
                leaves.append(node)
                leaves.sort()
        edges.append((leaves[0], leaves[1]))
        shape = _canonical_tree(edges, labels)
        if shape not in seen:
            seen.add(shape)
            yield edges


def _joined(acids: Sequence, links, bridge_z: int):
    """The structure of ``acids`` joined by ``links``: ``(acid i, group of i, acid j, group of j)`` each, the OH
    of both groups lost and a bridge atom between their centres."""
    combined, offsets = acids[0], [0]
    for acid in acids[1:]:
        offsets.append(combined.GetNumAtoms())
        combined = Chem.CombineMols(combined, acid)
    rw = Chem.RWMol(combined)
    dropped = []
    for i, group_i, j, group_j in links:
        bridge = rw.AddAtom(Chem.Atom(bridge_z))
        rw.AddBond(offsets[i] + group_i[1], bridge, Chem.BondType.SINGLE)
        rw.AddBond(bridge, offsets[j] + group_j[1], Chem.BondType.SINGLE)
        dropped += [offsets[i] + group_i[0], offsets[j] + group_j[0]]
    dropped = sorted(set(dropped))
    for index in reversed(dropped):
        rw.RemoveAtom(index)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol, offsets, dropped


def _has_stereo(mol) -> bool:
    return (any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())
            or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()))


def _labelled(mol):
    """A copy of ``mol`` with the CIP descriptors of the project's labelling (``perception.stereo``) of its own
    structure: the labels a copy carries from a molecule it was cut from are cleared first."""
    from ..perception.stereo import _CIP_ASSIGNED_PROP, assign_stereochemistry
    copy = Chem.Mol(mol)
    copy.ClearProp(_CIP_ASSIGNED_PROP)
    for atom in copy.GetAtoms():
        atom.ClearProp("_CIPCode")
    for bond in copy.GetBonds():
        bond.ClearProp("_CIPCode")
    assign_stereochemistry(copy)
    return copy


def _label(item) -> Optional[str]:
    return item.GetProp("_CIPCode") if item.HasProp("_CIPCode") else None


def _labels_agree(acids: Sequence, built, offsets: Sequence[int], dropped: Sequence[int]) -> bool:
    """Whether every stereo element of every acid has, in the whole structure ``built``, the descriptor it has in
    the free acid. (the Blue Book): "In preferred IUPAC names, stereodescriptors are placed
    immediately at the front of the part of the name to which they relate", and they are those of the structure
    the name describes. The acid words are read as free acids, so a word whose descriptor changes when the
    acid is linked (an alpha carbon with two carboxyl-type groups: S in the free acid, R in the molecule) states
    another configuration than the molecule has, although the structure is the same. The acid atom ``a`` of
    acid ``i`` is atom ``offsets[i] + a`` of the combined acids, less the dropped atoms before it."""
    stereo = [i for i, acid in enumerate(acids) if _has_stereo(acid)]
    if not stereo:
        return True
    whole = _labelled(built)

    def mapped(index: int, i: int) -> int:
        old = offsets[i] + index
        return old - sum(1 for d in dropped if d < old)

    lost = set(dropped)
    for i in stereo:
        free = _labelled(acids[i])
        for atom in free.GetAtoms():
            if offsets[i] + atom.GetIdx() not in lost and _label(atom) != _label(
                    whole.GetAtomWithIdx(mapped(atom.GetIdx(), i))):
                return False
        for bond in free.GetBonds():
            ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if any(offsets[i] + end in lost for end in ends):
                continue
            other = whole.GetBondBetweenAtoms(mapped(ends[0], i), mapped(ends[1], i))
            if other is None or _label(bond) != _label(other):
                return False
    return True


def _every_build_is(acids: Sequence, bridge_z: int, input_key: str, budget: List[int]) -> bool:
    """Whether every way of joining ``acids`` into a tree gives the structure of ``input_key``, and there is
    one; False for a different structure, none, or more joins than the budget."""
    groups = [[(m[0], m[1]) for m in acid.GetSubstructMatches(_ACID_GROUP)] for acid in acids]
    labels = [_key(acid) or "" for acid in acids]
    built = False
    for edges in _trees(labels, [len(g) for g in groups]):
        incident = defaultdict(list)
        for number, (left, right) in enumerate(edges):
            incident[left].append(number)
            incident[right].append(number)
        choices = [list(itertools.permutations(range(len(groups[i])), len(incident[i]))) for i in range(len(acids))]
        for pick in itertools.product(*choices):
            budget[0] -= 1
            if budget[0] < 0:
                return False
            chosen = {}
            for i, taken in enumerate(pick):
                for edge_number, group in zip(incident[i], taken):
                    chosen[(edge_number, i)] = groups[i][group]
            links = [(left, chosen[(number, left)], right, chosen[(number, right)])
                     for number, (left, right) in enumerate(edges)]
            try:
                mol, offsets, dropped = _joined(acids, links, bridge_z)
                if _key(mol) != input_key or (not built and not _labels_agree(acids, mol, offsets, dropped)):
                    return False
            except Exception:
                return False
            built = True
    return built


def anhydride_acids_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """``ANHYDRIDE_ACIDS_RULE`` when the poly-anhydride ``name`` reads back to the FULL InChIKey of ``smiles``
    through its acids (module docstring), else None: no such name, an acid OPSIN does not read or reads as
    ambiguous, a different structure, or a name more structures answer to. Any failure of the read-back is
    no verification."""
    if not is_poly_anhydride_name(name):
        return None
    try:
        return _read_back(name, smiles)
    except Exception:
        return None


def _read_back(name: str, smiles: Optional[str]) -> Optional[str]:
    words = name.split(" ")
    if not smiles or len(words) < 3 or len(words) - 1 > _MAX_ACIDS:
        return None
    target = _class_of(words[-1])
    mol = Chem.MolFromSmiles(smiles)
    input_key = _key(mol) if mol is not None else None
    if target is None or input_key is None:
        return None
    links, bridge_z = target
    if links + 1 > _MAX_ACIDS:
        return None
    split = _input_acids(mol, bridge_z)
    if split is None or split[0] != links:
        return None
    memo: Dict[str, object] = {}
    budget = [_MAX_BUILDS]
    verified = False
    for combination in itertools.product(*[_readings(word) for word in words[:-1]]):
        if sum(count for count, _ in combination) != links + 1:
            continue
        acids = []
        for count, stem in combination:
            acid = _acid(stem, memo)
            if acid is None:
                break
            acids += [acid] * count
        else:
            if sorted(_key(acid) or "" for acid in acids) != split[1]:
                return None
            if not _every_build_is(acids, bridge_z, input_key, budget):
                return None
            verified = True
    return ANHYDRIDE_ACIDS_RULE if verified else None
