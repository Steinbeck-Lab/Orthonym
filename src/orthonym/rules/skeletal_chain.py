"""The skeletal replacement ('a') PIN of an acyclic chain: without a principal characteristic group, or with -ol, -thiol
or -amine groups on the chain carbons.

``## **** Skeletal replacement ('a') nomenclature in acyclic chains`` (the Blue Book),
(:23348): "Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative names must be used to
generate preferred IUPAC names for acyclic structures when four or more heterounits are present in a unbranched
chain containing at least one carbon atom and when none of the heteroatoms constitute all or part of the principal
characteristic group of the compound." (:23352): the heteroatoms make a new parent hydride "whose
numbering is fixed"; "Suffixes, endings, and prefixes are added in accordance with this fixed numbering."

The builder reads the structure with its own vocabulary, closed on purpose (anything outside it returns None, so the
legacy path in ``skeletal_replacement`` and then the substitutive namers see the molecule unchanged):

* the molecule: one acyclic fragment of at most 120 heavy atoms, neutral, without isotopes, radicals or stereo,
  atoms C, F, Cl, Br, I and the replacement elements E = O, S, Se, Te, N, P, Si, Ge, B, at least four of E (not
  counting an OH, SH or NH2 on a carbon, which is a group, not a chain atom), multiple bonds only between carbons;
* the chain: a path of neutral carbons and E atoms of standard valence, ending in C or in P, Si, Ge, B,
  :6465;,:23430), with no two adjacent heteroatoms (one heteroatom is then one heterounit) and at least
  four of them; every other part of the molecule hangs on it as a halogen on a carbon, an unbranched alkyl,
  propan-2-yl,:15935), tert-butyl,:24412), or as an OH, SH or NH2 on a carbon;
* the principal group is the most senior of the OH, SH and NH2 groups,:18162, classes 17 and 19;
  ,:18875): every group of that kind is the suffix (``-ol``, ``-thiol``, ``-amine``,,:23405: the
  same number of groups as in the substitutive name) and a junior group is a prefix (``sulfanyl``, ``amino``). A
  chain with a suffix group has no multiple bond;
* a chain nitrogen is the principal amine, not a skeletal atom, when no OH or SH group is on the chain, unless the
  amine groups are exactly two NH2, one on each end atom of the chain, and every chain nitrogen has two neighbours
   Polyamines,:26324, ``3,6,9,12-tetraazatetradecane-1,14-diamine (PIN)``,:26400); any other such chain
  is left to the substitutive namers.

The principal chain is chosen by ``### **** SENIORITY ORDER FOR PARENT STRUCTURES``,:18875, the
greater number of principal groups, which comes before the criteria of chains) and then ``## **** SENIORITY OF
ACYCLIC CHAINS (THE PRINCIPAL CHAIN)`` (:20916): the greater number of heteroatoms, then of skeletal atoms, then of the
most senior heteroatom in the order O > S > Se > Te > N > P > Si > Ge > B:20930,:20938,
:21002). Chains that tie are all named; they must give one name, or the molecule is declined (the criteria of
are not built). A chain that this module does not name (a part hangs on it outside the vocabulary, or the N policy
refuses it) still takes part: when it ranks as high as the best chain that can be named, the molecule is declined,
because a shorter chain is not the principal one.

The numbering follows ``### **** NUMBERING`` (:3219) and Numbering of heteroacyclic parent
hydrides, (:6476), "a lower set of locants to heteroatoms considered together as a set without regard
to kind, and then, if there is a choice, to heteroatoms cited first in the seniority order": the lowest heteroatom
locants, then those of O, S, Se, Te, N, P, Si, Ge, B in turn, then the suffix groups (c),,:23352:
"Suffixes, endings, and prefixes are added in accordance with this fixed numbering"), then the multiple bonds
,:6502: all together, then the double bonds), then all prefixes together, then the prefix cited first in
alphanumerical order. The name is ``<prefixes>-<'a' prefixes><parent>``, the prefixes in alphanumerical order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..assembly.naming_utils import alpha_sort_key, format_substituent_prefix
from ..data.chain_names import get_chain_prefix
from .lambda_convention import nonstandard_bonding_number
from .skeletal_replacement import _A_CITATION_INDEX, REPLACEMENT_TERMS, _build_replacement_name

#: the cap on the molecule the builder handles (heavy atoms)
_MAX_HEAVY_ATOMS = 120
#: heterounits of the chain,:23348)
_MIN_HETEROATOMS = 4
#: the replacement elements of the chain, and the order of their seniority,:21002;
_E = frozenset(REPLACEMENT_TERMS) & frozenset({"O", "S", "Se", "Te", "N", "P", "Si", "Ge", "B"})
_E_ORDER = tuple(sorted(_E, key=_A_CITATION_INDEX.__getitem__))
#: a heteroatom that may end a chain,:6465)
_END_HETERO = frozenset({"P", "Si", "Ge", "B"})
_HALOGEN_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
#: the groups OH, SH, NH2 on a chain carbon by their atom: (suffix, prefix); the order of the kinds is their
#: seniority,:18162, classes 17 and 19)
_SUFFIX = {"O": ("ol", "hydroxy"), "S": ("thiol", "sulfanyl"), "N": ("amine", "amino")}
_ALLOWED_ATOMS = frozenset({"C"}) | frozenset(_HALOGEN_PREFIX) | _E

_SINGLE = Chem.BondType.SINGLE
_DOUBLE = Chem.BondType.DOUBLE
_TRIPLE = Chem.BondType.TRIPLE


@dataclass(frozen=True)
class _Pendant:
    """A part of the molecule off the chain, attached by one single bond to the chain atom ``attach`` (atom index).
    ``kind`` is ``"sub"`` for a prefix (``text`` is its name) or ``"leaf"`` for an OH, SH or NH2 on the chain
    (``text`` is ``"O"``, ``"S"`` or ``"N"``)."""
    attach: int
    kind: str
    text: str


@dataclass(frozen=True)
class _Candidate:
    """A chain of the molecule (atom indices, one orientation) and what hangs on it. ``known`` is False when a part
    that hangs on it is outside the vocabulary: the chain then has no pendants listed and is never named, but it still
    competes for the principal chain."""
    chain: Tuple[int, ...]
    pendants: Tuple[_Pendant, ...]
    known: bool = True


@dataclass(frozen=True)
class _Numbered:
    """One orientation of a candidate: heteroatoms ``(locant, element)``, the locants of the double and of the triple
    bonds, the suffix (``"ol"``, ``"thiol"``, ``"amine"`` or ``""``) with the locants of its groups, and the prefixes
    ``(locant, name)``, all on the numbering of the chain as oriented."""
    hetero: Tuple[Tuple[int, str], ...]
    enes: Tuple[int, ...]
    ynes: Tuple[int, ...]
    suffix: str
    suffix_locants: Tuple[int, ...]
    prefixes: Tuple[Tuple[int, str], ...]


# --- the molecule ----------------------------------------------------------------------------------------------

def _carbon_bond(bond) -> bool:
    return bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "C"


def _leaf(atom) -> bool:
    """An O, S or N atom whose only neighbour is a carbon: the OH, SH or NH2 group of that carbon, not a chain atom."""
    return atom.GetSymbol() in _SUFFIX and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetSymbol() == "C"


def _chain_hetero_count(mol: Chem.Mol) -> int:
    """The atoms of E that the chain must hold: all of them but the OH, SH and NH2 groups."""
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() in _E and not _leaf(a))


def _in_scope(mol: Optional[Chem.Mol]) -> bool:
    """The gates before any enumeration: one acyclic neutral fragment of at most 120 heavy atoms, no isotope, radical
    or stereo, atoms from the closed set, multiple bonds between carbons, at least four atoms of E besides the groups
    OH, SH and NH2."""
    if mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
        return False
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        return False
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() not in _ALLOWED_ATOMS or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetIsotope() or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            return False
    for bond in mol.GetBonds():
        kind = bond.GetBondType()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE or kind not in (_SINGLE, _DOUBLE, _TRIPLE):
            return False
        if kind != _SINGLE and not _carbon_bond(bond):
            return False
    return _chain_hetero_count(mol) >= _MIN_HETEROATOMS


# --- the chains ------------------------------------------------------------------------------------------------

def _chain_atoms_ok(mol: Chem.Mol, chain: List[int]) -> bool:
    """ /: carbons and E atoms of standard valence, no two heteroatoms adjacent, the ends C or
    P, Si, Ge, B (whether a nitrogen may stay in the chain is decided by ``_suffix_ok``)."""
    previous_hetero = False
    for place, idx in enumerate(chain):
        symbol = mol.GetAtomWithIdx(idx).GetSymbol()
        hetero = symbol != "C"
        if hetero and (symbol not in _E or nonstandard_bonding_number(mol, idx) is not None):
            return False
        if hetero and (previous_hetero or (place in (0, len(chain) - 1) and symbol not in _END_HETERO)):
            return False
        previous_hetero = hetero
    return True


def _alkyl_name(mol: Chem.Mol, root: int, part: List[int]) -> Optional[str]:
    """The prefix name of a saturated all-carbon part hanging by ``root``: unbranched ('methyl', 'ethyl',...),
    'propan-2-yl',:15935) or 'tert-butyl',:24412); None for any other part."""
    members = set(part)
    for idx in part:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C" or any(b.GetBondType() != _SINGLE for b in atom.GetBonds()):
            return None
    degree = {idx: sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in members) for idx in part}
    if degree[root] <= 1 and max(degree.values()) <= 2:
        return get_chain_prefix(len(part)) + "yl"
    if len(part) == 3 and degree[root] == 2:
        return "propan-2-yl"
    if len(part) == 4 and degree[root] == 3:
        return "tert-butyl"
    return None


def _pendant(mol: Chem.Mol, attach: int, root: int, chain: set) -> Optional[_Pendant]:
    """The part of the molecule that hangs from the chain atom ``attach`` by the bond to ``root``, or None when it is
    not in the vocabulary (a halogen, an OH, SH or NH2 on a carbon, an alkyl of the kinds above)."""
    if mol.GetBondBetweenAtoms(attach, root).GetBondType() != _SINGLE:
        return None
    part, seen = [root], {root}
    for idx in part:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in seen and nb.GetIdx() not in chain:
                seen.add(nb.GetIdx())
                part.append(nb.GetIdx())
    symbol = mol.GetAtomWithIdx(root).GetSymbol()
    if len(part) == 1 and (symbol in _HALOGEN_PREFIX or symbol in _SUFFIX):
        if mol.GetAtomWithIdx(attach).GetSymbol() != "C":
            return None
        return (_Pendant(attach, "sub", _HALOGEN_PREFIX[symbol]) if symbol in _HALOGEN_PREFIX
                else _Pendant(attach, "leaf", symbol))
    name = _alkyl_name(mol, root, part)
    return None if name is None else _Pendant(attach, "sub", name)


def _principal_kind(pendants) -> Optional[str]:
    """The kind of the principal group (S2): the most senior of the leaf groups on the chain, "O", "S" or "N", or
    None when the chain carries none,:18162: hydroxy compounds, chalcogen analogues, then amines)."""
    kinds = {p.text for p in pendants if p.kind == "leaf"}
    return next((kind for kind in _SUFFIX if kind in kinds), None)


def _suffix_ok(mol: Chem.Mol, cand: _Candidate) -> bool:
    """S2 and the unsaturation rule. Without a principal group a chain nitrogen is the principal amine, so the chain
    is refused. With -ol or -thiol a chain nitrogen is a skeletal atom. With -amine it is one only in the polyamine
    of (:26324, ':26400'): two NH2, one on each end of the chain, and every chain nitrogen of degree two;
    -amine allows one NH2 per carbon and at most two. A suffix group and a multiple bond are not combined."""
    chain, pendants = cand.chain, cand.pendants
    kind = _principal_kind(pendants)
    nitrogens = [i for i in chain if mol.GetAtomWithIdx(i).GetSymbol() == "N"]
    if kind is None:
        return not nitrogens
    if any(bond.GetBondType() != _SINGLE for bond in mol.GetBonds()):
        return False
    if kind != "N":
        return True
    hosts = sorted(p.attach for p in pendants if p.kind == "leaf" and p.text == "N")
    if len(set(hosts)) != len(hosts) or len(hosts) > 2:
        return False
    return not nitrogens or (hosts == sorted((chain[0], chain[-1]))
                             and all(mol.GetAtomWithIdx(i).GetDegree() == 2 for i in nitrogens))


def _candidate(mol: Chem.Mol, chain: List[int]) -> Optional[_Candidate]:
    """The chain with its pendants, or None when its atoms are not a chain (``_chain_atoms_ok``). A pendant outside
    the vocabulary gives a chain that is not ``known``. The suffix and N policy of ``_suffix_ok`` is a separate step, so
    that ``name_a_chain`` sees the chains it refuses."""
    if not _chain_atoms_ok(mol, chain):
        return None
    members = set(chain)
    pendants: List[_Pendant] = []
    for idx in chain:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in members:
                continue
            pendant = _pendant(mol, idx, nb.GetIdx(), members)
            if pendant is None:
                return _Candidate(tuple(chain), (), known=False)
            pendants.append(pendant)
    return _Candidate(tuple(chain), tuple(pendants))


def _path(parent: Dict[int, int], start: int, stop: int) -> List[int]:
    """The path from ``start`` to ``stop`` in the tree whose BFS parents from ``start`` are ``parent``."""
    path = [stop]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]


def _chains(mol: Optional[Chem.Mol]) -> List[_Candidate]:
    """All the chains of the molecule (S1), the ones whose pendants are outside the vocabulary included (they are not
    ``known``): one BFS per end atom gives the one path to each other end atom of the tree (O(n^2) paths). A pendant
    carries no heteroatom (an OH, SH or NH2 is a group, not an atom of the chain), so a chain must hold every atom of E
    but those groups."""
    if not _in_scope(mol):
        return []
    n_hetero = _chain_hetero_count(mol)
    ends = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" or a.GetSymbol() in _END_HETERO]
    found: List[_Candidate] = []
    for k, start in enumerate(ends):
        parent: Dict[int, int] = {}
        order = [start]
        for idx in order:
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetIdx() != start and nb.GetIdx() not in parent:
                    parent[nb.GetIdx()] = idx
                    order.append(nb.GetIdx())
        for stop in ends[k + 1:]:
            chain = _path(parent, start, stop)
            if sum(1 for i in chain if mol.GetAtomWithIdx(i).GetSymbol() in _E) != n_hetero:
                continue
            cand = _candidate(mol, chain)
            if cand is not None:
                found.append(cand)
    return found


def _candidates(mol: Optional[Chem.Mol]) -> List[_Candidate]:
    """The chains of ``_chains`` that are ``known`` and also pass the suffix and N policy (S2)."""
    return [cand for cand in _chains(mol) if cand.known and _suffix_ok(mol, cand)]


def _molecule_principal_kind(mol: Chem.Mol) -> Optional[str]:
    """The kind of the most senior OH, SH or NH2 group of the whole molecule,:18162), or None."""
    kinds = {a.GetSymbol() for a in mol.GetAtoms() if _leaf(a)}
    return next((kind for kind in _SUFFIX if kind in kinds), None)


def _rank(mol: Chem.Mol, cand: _Candidate) -> tuple:
    """Larger is better. (:18875), which comes before: the number of principal groups on the carbons of
    the chain. Then (:20916): the number of heteroatoms (a,:20930), of skeletal atoms (b,:20938), then the
    number of each heteroatom in the order O, S, Se, Te, N, P, Si, Ge, B (c,:21002)."""
    members = set(cand.chain)
    kind = _molecule_principal_kind(mol)
    groups = sum(1 for idx in cand.chain for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
                 if nb.GetIdx() not in members and nb.GetSymbol() == kind and _leaf(nb))
    hetero = [s for s in (mol.GetAtomWithIdx(i).GetSymbol() for i in cand.chain) if s in _E]
    return (groups, len(hetero), len(cand.chain)) + tuple(hetero.count(el) for el in _E_ORDER)


# --- the numbering and the text ----------------------------------------------------------------------------------

def _cite_key(name: str) -> tuple:
    """The order of citation of prefixes: alphanumerical, the name as the last resort."""
    return (alpha_sort_key(name), name)


def _number(mol: Chem.Mol, cand: _Candidate, chain: Tuple[int, ...]) -> _Numbered:
    """``cand`` numbered along ``chain`` (its atoms in one orientation): locants of the heteroatoms, the double and the
    triple bonds (the locant of a bond is that of its lower-numbered atom), the groups of the principal kind (the
    suffix) and the prefixes (the other leaf groups among them)."""
    locant = {idx: place + 1 for place, idx in enumerate(chain)}
    hetero = tuple((locant[i], mol.GetAtomWithIdx(i).GetSymbol()) for i in chain
                   if mol.GetAtomWithIdx(i).GetSymbol() in _E)
    kinds = [mol.GetBondBetweenAtoms(a, b).GetBondType() for a, b in zip(chain, chain[1:])]
    enes = tuple(place + 1 for place, kind in enumerate(kinds) if kind == _DOUBLE)
    ynes = tuple(place + 1 for place, kind in enumerate(kinds) if kind == _TRIPLE)
    kind = _principal_kind(cand.pendants)
    suffix = tuple(sorted(locant[p.attach] for p in cand.pendants if p.kind == "leaf" and p.text == kind))
    prefixes = tuple(sorted((locant[p.attach], p.text if p.kind == "sub" else _SUFFIX[p.text][1])
                            for p in cand.pendants if p.kind == "sub" or p.text != kind))
    return _Numbered(hetero, enes, ynes, _SUFFIX[kind][0] if kind else "", suffix, prefixes)


def _key(num: _Numbered) -> tuple:
    """The numbering criteria of (:3219) and (:6476), in order, the lowest first: the heteroatom
    locants as a set, those of each element in the order of seniority, the suffix groups (c)), the multiple
    bonds together, the double bonds, all the prefixes together, the prefix cited first (g))."""
    first = min((name for _, name in num.prefixes), key=_cite_key, default=None)
    return (
        tuple(loc for loc, _ in num.hetero),
        tuple(tuple(loc for loc, el in num.hetero if el == element) for element in _E_ORDER),
        num.suffix_locants,
        tuple(sorted(num.enes + num.ynes)),
        num.enes,
        tuple(loc for loc, _ in num.prefixes),
        tuple(loc for loc, name in num.prefixes if name == first),
    )


def _text(length: int, num: _Numbered) -> str:
    """``<prefixes>-<'a' prefixes><parent>``,:23352), the prefixes in alphanumerical order; the parent
    carries the suffix groups (``tetradecane-1,14-diol``,,:23405)."""
    by_name: Dict[str, List[int]] = {}
    for loc, name in num.prefixes:
        by_name.setdefault(name, []).append(loc)
    parent = _build_replacement_name(
        length, list(num.hetero), (num.suffix, num.suffix_locants[0]) if num.suffix else None,
        ene_locants=list(num.enes), yne_locants=list(num.ynes), suffix_locants=list(num.suffix_locants) or None)
    cited = [format_substituent_prefix(name, sorted(locs), len(locs)) for name, locs in
             sorted(by_name.items(), key=lambda item: _cite_key(item[0]))]
    return "-".join(cited + [parent])


def _numbered_name(mol: Chem.Mol, cand: _Candidate) -> Optional[str]:
    """The name of ``cand`` on the preferred of its two orientations; a tie of the whole cascade that gives two
    different names declines (None)."""
    fits = [_number(mol, cand, chain) for chain in (cand.chain, cand.chain[::-1])]
    best = min(_key(num) for num in fits)
    names = {_text(len(cand.chain), num) for num in fits if _key(num) == best}
    return names.pop() if len(names) == 1 else None


def name_a_chain(mol: Optional[Chem.Mol]) -> Optional[str]:
    """The 'a' PIN of an acyclic chain,:23348) whose principal group, if any, is -ol, -thiol or -amine,
    or None when the molecule is outside the vocabulary of this module or the chains that tie give two names."""
    candidates: List[_Candidate] = []
    unnamed: List[_Candidate] = []
    for cand in _chains(mol):
        (candidates if cand.known and _suffix_ok(mol, cand) else unnamed).append(cand)
    if not candidates:
        return None
    ranks = [_rank(mol, cand) for cand in candidates]
    top = max(ranks)
    # and would take, or tie with, a chain that this module does not name (a pendant outside the
    # vocabulary, or a nitrogen that ``_suffix_ok`` refuses): no other chain is then the principal one
    if any(_rank(mol, cand) >= top for cand in unnamed):
        return None
    names = {_numbered_name(mol, cand) for cand, rank in zip(candidates, ranks) if rank == top}
    return names.pop() if len(names) == 1 and None not in names else None
