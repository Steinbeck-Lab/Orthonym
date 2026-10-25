"""Amplificants of a phane: the ring or ring system that one superatom stands for.

the Blue Book, each line read with ``sed -n``:

* (:14892): the amplification prefix is the name of the allowed ring or ring system with
  the final letter 'e' changed to 'a', or an 'a' added when there is no final 'e'.
* "Allowed parent hydrides" (:14902): "An amplification prefix can be derived from
  monocyclic rings and polycyclic ring systems having the maximum number of noncumulative double
  bonds (mancude), bridged fused ring systems, saturated monocyclic rings, saturated bicycloalkanes
  and polycycloalkanes (von Baeyer hydrocarbons), and spiro alkanes."
* (a) "Disallowed names of parent hydrides" (:14910): spirobi names, "spiro ring systems
  with at least one fused ring system or polycycloalkane ring system", ring assembly names.
* (:14967): "Numbering of an amplificant is determined primarily by the numbering rules
  that apply to the parent name from which the amplification prefix is derived. When there is a
  choice, the general rule of lowest locants is used."
* (:14794): a heterocyclic amplificant with locants takes brackets (``[1,2]oxazola``).
* (:15173): "Indicated hydrogen, when present in amplificants, is placed before the phane parent hydride
  name and preceded by the appropriate composite locant(s)" (``1^4H-1(3,5)-pyrana-...``,:15181). A ring system that is
  a hydro derivative or needs indicated hydrogen is the amplificant of its mancude parent, and the added hydrogen atoms are
  cited as hydro prefixes (``1^1,1^2-dihydro-1(2)-quinolina-...``,:21442; ``decahydro-1(2,7)-naphthalena-...``,:17063),
  directly before the parent hydride name,:16872 and:16874: hydro prefixes are detachable but not alphabetized). (b)
  (:3246) gives indicated hydrogen the lowest locants, before the principal characteristic groups; (e)(i) puts the hydro
  prefixes with the 'ene' and 'yne' endings.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from rdkit import Chem

from ...validation.spelling.cyclophane import _DIVALENT, is_mancude_system, maximum_matching_size
from ...validation.spelling.linear_phane import _CHAIN_ELEMENTS, ring_system_hydride
from ..skeletal_replacement import _A_CITATION_ORDER

#: (:15246) "Lowest locants are assigned to heteroatoms considered in the order of their
#: seniority: O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl"
HETERO_ORDER: Tuple[str, ...] = tuple(e for e in _A_CITATION_ORDER if e in _CHAIN_ELEMENTS and e != "C")

# a name with indicated hydrogen, a hydro prefix, the lambda convention, a spiro or a primed locant is no
# allowed parent of the cases built here (PN4c, PN5)
_NO_AMPLIFICANT_NAME = re.compile(r"\d[A-Za-z]*H-|hydro|λ|lambda|spiro|['′]")
_LEADING_LOCANTS = re.compile(r"^(\d+(?:,\d+)*)-")
_Locant = Tuple[int, str]


@dataclass(frozen=True)
class AmplificantHydride:
    name: str                    # engine PIN of the isolated hydride: "pyridine", "1,3-oxazole", "naphthalene"
    prefix: str                  # amplification prefix: "pyridina", "[1,3]oxazola"
    key: str                     # canonical SMILES of the hydride (the identity of the amplificant type)
    score: tuple                 # rules.ring_selection.ring_system_score of the hydride; lower = senior)
    numberings: Tuple[Mapping[int, _Locant], ...]   # every valid numbering: atom index IN THE INPUT MOLECULE -> (number, letter)
    #: a hydro derivative or a ring that needs indicated hydrogen, read as its mancude parent (``name``, ``key`` and
    #: ``score`` are the parent's): the atoms of the ring system, in the input molecule, that carry an added hydrogen
    #: atom (``saturated``), how many of them the parent cites as indicated hydrogen (``indicated``), the atoms that can
    #: carry a ring double bond (``matchable``) and the ring bonds between them (``bonds``)
    saturated: Tuple[int, ...] = ()
    indicated: int = 0
    matchable: Tuple[int, ...] = ()
    bonds: Tuple[Tuple[int, int], ...] = ()


def amplification_prefix(name: str) -> str:
    """ and: the amplification prefix of the parent hydride ``name``."""
    from ..phane import _amplification_prefix
    return _amplification_prefix(_LEADING_LOCANTS.sub(r"[\1]", name))


def _plain_allowed_atoms(mol: Chem.Mol, atoms: FrozenSet[int]) -> bool:
    """No charge, radical or isotope on the ring system, and no multiple bond to an atom outside it."""
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope():
            return False
        for bond in atom.GetBonds():
            if bond.GetOtherAtomIdx(idx) not in atoms and bond.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True


def _allowed_parent(hydride: Chem.Mol) -> bool:
    """: a mancude ring or ring system, or a saturated monocycle."""
    everything = frozenset(range(hydride.GetNumAtoms()))
    if is_mancude_system(hydride, everything):
        return True
    if hydride.GetRingInfo().NumRings() != 1:
        return False
    return all(b.GetBondType() == Chem.BondType.SINGLE for b in hydride.GetBonds())


def _needs_indicated_hydrogen(hydride: Chem.Mol) -> bool:
    """ (:3557): a mancude hydride that has a saturated ring atom (a pyrrole-type NH, a CH2 of 2H-pyran)
    needs indicated hydrogen. Read from the structure, because the engine's name of a fused hydride of this
    kind can come without it (``pyrido[3,4-b]pyrrolo[2,3-d]pyridine`` for a hydride that OPSIN reads otherwise)."""
    unsaturated = any(b.GetBondType() != Chem.BondType.SINGLE for b in hydride.GetBonds())
    if not unsaturated:
        return False
    for atom in hydride.GetAtoms():
        multiple = atom.GetIsAromatic() or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())
        if atom.GetIsAromatic():
            # a heteroatom with a hydrogen and no valence left for a pi bond (pyrrole N-H); a silabenzene Si-H has one
            if (atom.GetSymbol() != "C" and atom.GetTotalNumHs()
                    and atom.GetDegree() + atom.GetTotalNumHs()
                    >= Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())):
                return True
        elif not multiple and atom.GetTotalNumHs():
            return True
    return False


def _hetero_key(symbols: List[str]) -> tuple:
    """(locants of all heteroatoms, locants per element in the order of of a numbered ring."""
    hetero = tuple(i + 1 for i, s in enumerate(symbols) if s != "C")
    per_element = tuple(tuple(i + 1 for i, s in enumerate(symbols) if s == e) for e in HETERO_ORDER)
    return hetero, per_element


def _monocycle_numberings(hydride: Chem.Mol) -> List[Dict[int, _Locant]]:
    """Every numbering of a monocycle that gives its heteroatoms the lowest locants, then the
    general rule: heteroatoms as a set, then in the order of; a ring without heteroatom keeps
    all 2n of them."""
    ring = list(hydride.GetRingInfo().AtomRings()[0])
    n = len(ring)
    best, found = None, []
    for start in range(n):
        for step in (1, -1):
            order = [ring[(start + step * k) % n] for k in range(n)]
            key = _hetero_key([hydride.GetAtomWithIdx(a).GetSymbol() for a in order])
            if best is None or key < best:
                best, found = key, []
            if key == best:
                found.append({a: (k + 1, "") for k, a in enumerate(order)})
    return found


def _fused_numbering(hydride: Chem.Mol) -> Optional[Dict[int, _Locant]]:
    from ..fusion_numbering import compute_fused_numbering
    got = compute_fused_numbering(hydride, set(range(hydride.GetNumAtoms())))
    if got is None:
        return None
    return {a: ((loc, "") if isinstance(loc, int) else (loc[0], loc[1])) for a, loc in got.items()}


def _numberings(hydride: Chem.Mol) -> Optional[List[Dict[int, _Locant]]]:
    """Every numbering of the hydride, in hydride atom indices: a monocycle as above, a cata-fused system by
    the fusion numbering composed with every automorphism of the hydride."""
    if hydride.GetRingInfo().NumRings() == 1:
        return _monocycle_numberings(hydride)
    base = _fused_numbering(hydride)
    if base is None:
        return None
    seen = []
    for match in hydride.GetSubstructMatches(hydride, uniquify=False, useChirality=False, maxMatches=100000):
        numbering = {match[i]: base[i] for i in range(hydride.GetNumAtoms())}
        if numbering not in seen:
            seen.append(numbering)
    return seen


_INDICATED_HYDROGEN = re.compile(r"\d+[a-z]*H(?:,\d+[a-z]*H)*-")
_HYDRO_COUNT = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6, "hepta": 7, "octa": 8, "nona": 9, "deca": 10,
                "undeca": 11, "dodeca": 12}
#: 'decahydro-' of the hydride of a total hydrogenation has no locants in the engine's name
_HYDRO_PREFIX = re.compile(r"(?:(?:\d+[a-z]*,)*\d+[a-z]*-)?(di|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca)hydro")


def _matching(graph: Dict[int, List[int]], free: List[int]) -> Optional[Dict[int, int]]:
    """A perfect matching of the atoms ``free`` along the bonds of ``graph`` (atom -> neighbours), or None."""
    free_set = set(free)
    match: Dict[int, int] = {}

    def extend(todo: List[int]) -> bool:
        if not todo:
            return True
        a = todo[0]
        for b in graph[a]:
            if b in free_set and b not in match and a not in match and b != a:
                match[a], match[b] = b, a
                if extend([x for x in todo[1:] if x not in match]):
                    return True
                del match[a], match[b]
        return False
    return match if extend(sorted(free)) else None


def _skeleton(hydride: Chem.Mol) -> Chem.Mol:
    """The heavy atoms of the ring system with every ring bond single: what the numbering of the ring depends on."""
    rw = Chem.RWMol()
    for atom in hydride.GetAtoms():
        rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in hydride.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return rw.GetMol()


def _skeleton_numberings(parent: Chem.Mol) -> Optional[List[Dict[int, _Locant]]]:
    """Every numbering of the skeleton of the mancude ``parent`` (the numbering of a hydro derivative is that of its
    parent,: a monocycle as ``_numberings``, a cata-fused system by the fusion numbering composed with every
    automorphism of the skeleton."""
    if parent.GetRingInfo().NumRings() == 1:
        return _monocycle_numberings(parent)
    base = _fused_numbering(parent)
    if base is None:
        return None
    skeleton = _skeleton(parent)
    seen: List[Dict[int, _Locant]] = []
    for match in skeleton.GetSubstructMatches(skeleton, uniquify=False, useChirality=False, maxMatches=100000):
        numbering = {match[i]: base[i] for i in range(parent.GetNumAtoms())}
        if numbering not in seen:
            seen.append(numbering)
    return seen


def _hydro_form(hydride: Chem.Mol, atom_map: Mapping[int, int]) -> Optional[AmplificantHydride]:
    """The ring system as a hydro derivative of its mancude parent, with indicated hydrogen where the parent needs it
    ,:15173; (b):3246 and (e)(i):3289): None when the ring system has no mancude parent, the engine's names of the
    hydride and of the parent do not tell the same hydrogen atoms, or it is a ring the engine names without hydro
    prefixes (the 'ene' of a cycloalkene)."""
    from ...assembly.fragment_naming import name_fragment_recursively
    from ...errors import is_failure_name
    from ..ring_selection import ring_system_score
    kekule = Chem.Mol(hydride)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        return None
    atoms = range(kekule.GetNumAtoms())
    matchable = [a for a in atoms
                 if not (kekule.GetAtomWithIdx(a).GetSymbol() in _DIVALENT
                         and kekule.GetAtomWithIdx(a).GetTotalValence() == 2)]
    graph = {a: [n.GetIdx() for n in kekule.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in matchable]
             for a in matchable}
    doubled = {i for b in kekule.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
               for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
    saturated = [a for a in matchable if a not in doubled]
    if not saturated or not 5 <= len(matchable) <= 30:
        return None
    k = len(matchable) - 2 * maximum_matching_size(graph, matchable)
    if k > len(saturated) or (len(saturated) - k) % 2:
        return None
    first = None
    for indicated in itertools.combinations(saturated, k):
        got = _matching(graph, [a for a in matchable if a not in indicated])
        if got is not None:
            first = (indicated, got)
            break
    if first is None:
        return None
    indicated, got = first
    parent_rw = Chem.RWMol(_skeleton(hydride))
    for a, b in got.items():
        if a < b:
            parent_rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
    try:
        parent = parent_rw.GetMol()
        Chem.SanitizeMol(parent)
    except Exception:  # noqa: BLE001
        return None
    parent_smiles = Chem.MolToSmiles(parent, isomericSmiles=False)
    hydride_name = name_fragment_recursively(Chem.MolToSmiles(hydride, isomericSmiles=False), style="pin")
    parent_name = name_fragment_recursively(parent_smiles, style="pin")
    if not hydride_name or not parent_name or is_failure_name(hydride_name) or is_failure_name(parent_name):
        return None
    marked = _INDICATED_HYDROGEN.findall(hydride_name)
    hydro = _HYDRO_PREFIX.search(hydride_name)
    named_indicated = sum(len(m.rstrip("-").split(",")) for m in marked)
    named_hydro = _HYDRO_COUNT[hydro.group(1)] if hydro else 0
    if (named_indicated != k or named_hydro + named_indicated != len(saturated)
            or not (marked or hydro)):
        return None             # the engine does not name this hydride as a hydro derivative of this parent
    bare = _INDICATED_HYDROGEN.sub("", parent_name)
    if len(_INDICATED_HYDROGEN.findall(parent_name)) != k or _NO_AMPLIFICANT_NAME.search(bare):
        return None
    numberings = _skeleton_numberings(parent)
    if not numberings:
        return None
    back = {i: a for a, i in atom_map.items()}
    return AmplificantHydride(
        name=bare, prefix=amplification_prefix(bare), key=parent_smiles,
        score=ring_system_score(parent, set(range(parent.GetNumAtoms()))),
        numberings=tuple({back[i]: loc for i, loc in numbering.items()} for numbering in numberings),
        saturated=tuple(back[a] for a in saturated), indicated=k,
        matchable=tuple(back[a] for a in matchable),
        bonds=tuple(sorted((back[b.GetBeginAtomIdx()], back[b.GetEndAtomIdx()]) for b in kekule.GetBonds()
                           if b.GetBeginAtomIdx() in graph and b.GetEndAtomIdx() in graph)))


def hydro_split(hydride: AmplificantHydride, numbering: Mapping[int, _Locant]
                ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The atoms of the amplificant that its name cites as indicated hydrogen and as hydro prefixes under
    ``numbering`` (b): the indicated hydrogen atoms have the lowest locants that leave a mancude parent; the
    others are the added hydrogen atoms of the hydro prefixes). Both ordered by locant; empty for a mancude or a
    saturated ring."""
    if not hydride.saturated:
        return (), ()
    graph: Dict[int, List[int]] = {a: [] for a in hydride.matchable}
    for a, b in hydride.bonds:
        graph[a].append(b)
        graph[b].append(a)
    ordered = sorted(hydride.saturated, key=lambda a: numbering[a])
    for indicated in itertools.combinations(ordered, hydride.indicated):
        if _matching(graph, [a for a in hydride.matchable if a not in indicated]) is not None:
            rest = [a for a in ordered if a not in indicated]
            return indicated, tuple(rest)
    return (), tuple(ordered)


def indicated_numberings(hydride: AmplificantHydride) -> Tuple[Mapping[int, _Locant], ...]:
    """The numberings of the amplificant that its parent name allows,:14967: "Numbering of an amplificant is
    determined primarily by the numbering rules that apply to the parent name from which the amplification prefix is
    derived. When there is a choice, the general rule of lowest locants is used"): of the valid numberings those that
    give the indicated hydrogen the lowest locants (b),:3246), because the indicated hydrogen belongs to the parent
    name and the attachment locants are compared among the numberings that remain. The book's `:19325` is
    ``1^2H-1(6)-pyrana-...``, not ``1^6H-1(2)-pyrana-...``. All valid numberings for a ring with no indicated hydrogen."""
    cached = hydride.__dict__.get("_indicated_numberings")
    if cached is None:
        if not hydride.indicated:
            cached = tuple(hydride.numberings)
        else:
            keyed = [(tuple(numbering[a] for a in hydro_split(hydride, numbering)[0]), numbering)
                     for numbering in hydride.numberings]
            best = min(key for key, _ in keyed)
            cached = tuple(numbering for key, numbering in keyed if key == best)
        object.__setattr__(hydride, "_indicated_numberings", cached)
    return cached


def amplificant_hydride(mol: Chem.Mol, atoms: FrozenSet[int]) -> Optional[AmplificantHydride]:
    """The amplificant hydride of the ring system ``atoms`` of ``mol``, or None when it is no parent the
    plain case builds (a charge, a radical, an isotope, a multiple bond to the outside, indicated hydrogen,
    hydro prefixes, a ring system that is neither mancude nor a saturated monocycle, and a system with no
    fusion numbering)."""
    from ...assembly.fragment_naming import name_fragment_recursively
    from ...errors import is_failure_name
    from ..ring_selection import ring_system_score
    if not _plain_allowed_atoms(mol, atoms):
        return None
    got = ring_system_hydride(mol, atoms)
    if got is None:
        return None
    hydride, atom_map = got
    if not _allowed_parent(hydride) or _needs_indicated_hydrogen(hydride):
        return _hydro_form(hydride, atom_map)
    smiles = Chem.MolToSmiles(hydride, isomericSmiles=False)
    name = name_fragment_recursively(smiles, style="pin")
    if not name or is_failure_name(name) or _NO_AMPLIFICANT_NAME.search(name):
        return None
    numberings = _numberings(hydride)
    if not numberings:
        return None
    back = {i: a for a, i in atom_map.items()}
    return AmplificantHydride(
        name=name, prefix=amplification_prefix(name), key=smiles,
        score=ring_system_score(hydride, set(range(hydride.GetNumAtoms()))),
        numberings=tuple({back[i]: loc for i, loc in numbering.items()} for numbering in numberings))
