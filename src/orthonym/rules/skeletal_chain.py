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
  is left to the substitutive namers;
* the one nitrogen of a molecule without OH, SH or NH2 groups, bonded by single bonds to carbons only, whose other
  neighbours carry alkyl parts of the kinds above, is the principal amine NHR or NR2 on the chain carbon: the suffix
  ``-amine`` with ``N``-prefixes, never a skeletal atom: none of the chain heteroatoms is part of the
  principal characteristic group; ``4-ethoxy-N,N,N-trimethyl-3,5,8-trioxa-4-phosphadecan-10-aminium (PIN)``,
  :23379). An alkyl name used both on the chain and on the nitrogen declines (the order of the mixed locants is not
  built).

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
import re
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
    (``text`` is ``"O"``, ``"S"`` or ``"N"``); ``n_prefixes`` are the alkyl names on the nitrogen of an NHR or NR2."""
    attach: int
    kind: str
    text: str
    n_prefixes: Tuple[str, ...] = ()


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
    n_prefixes: Tuple[str, ...] = ()


# --- the molecule ----------------------------------------------------------------------------------------------

def _carbon_bond(bond) -> bool:
    return bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "C"


def _leaf(atom) -> bool:
    """An O, S or N atom whose only neighbour is a carbon: the OH, SH or NH2 group of that carbon, not a chain atom."""
    return atom.GetSymbol() in _SUFFIX and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetSymbol() == "C"


def _amine_nitrogen(mol: Chem.Mol) -> Optional[int]:
    """The index of the principal amine NHR or NR2 (see the module text), or None: the one nitrogen of a molecule
    without OH, SH or NH2 groups, of two or three neighbours, all carbons bonded by single bonds. Whether its other
    neighbours carry alkyl parts is read by ``_pendant``."""
    if any(_leaf(a) for a in mol.GetAtoms()):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    if len(nitrogens) != 1:
        return None
    atom = nitrogens[0]
    if atom.GetDegree() not in (2, 3) or any(n.GetSymbol() != "C" for n in atom.GetNeighbors()):
        return None
    return atom.GetIdx() if all(b.GetBondType() == _SINGLE for b in atom.GetBonds()) else None


def _chain_hetero_count(mol: Chem.Mol) -> int:
    """The atoms of E that the chain must hold: all of them but the OH, SH and NH2 groups and the principal amine
    NHR or NR2."""
    amine = _amine_nitrogen(mol)
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() in _E and not _leaf(a) and a.GetIdx() != amine)


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
        if atom.GetSymbol() != "C" or atom.IsInRing() or any(b.GetBondType() != _SINGLE for b in atom.GetBonds()):
            return None                         # a ring atom: cyclopropyl is no propan-2-yl
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
    if len(part) > 1 and root == _amine_nitrogen(mol):
        if mol.GetAtomWithIdx(attach).GetSymbol() != "C":
            return None
        names = []
        for nb in mol.GetAtomWithIdx(root).GetNeighbors():
            if nb.GetIdx() == attach:
                continue
            branch, seen = [nb.GetIdx()], {nb.GetIdx(), root}
            for idx in branch:
                for far in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if far.GetIdx() not in seen:
                        seen.add(far.GetIdx())
                        branch.append(far.GetIdx())
            name = _alkyl_name(mol, nb.GetIdx(), branch)
            if name is None:
                return None
            names.append(name)
        return _Pendant(attach, "leaf", "N", tuple(sorted(names)))
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
    if _amine_nitrogen(mol) is not None:
        kinds.add("N")
    return next((kind for kind in _SUFFIX if kind in kinds), None)


def _rank(mol: Chem.Mol, cand: _Candidate) -> tuple:
    """Larger is better. (:18875), which comes before: the number of principal groups on the carbons of
    the chain. Then (:20916): the number of heteroatoms (a,:20930), of skeletal atoms (b,:20938), then the
    number of each heteroatom in the order O, S, Se, Te, N, P, Si, Ge, B (c,:21002)."""
    members = set(cand.chain)
    kind = _molecule_principal_kind(mol)
    amine = _amine_nitrogen(mol)
    groups = sum(1 for idx in cand.chain for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
                 if nb.GetIdx() not in members and nb.GetSymbol() == kind and (_leaf(nb) or nb.GetIdx() == amine))
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
    n_prefixes = tuple(name for p in cand.pendants for name in p.n_prefixes)
    return _Numbered(hetero, enes, ynes, _SUFFIX[kind][0] if kind else "", suffix, prefixes, n_prefixes)


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


def _text(length: int, num: _Numbered) -> Optional[str]:
    """``<prefixes>-<'a' prefixes><parent>``,:23352), the prefixes in alphanumerical order, the
    ``N``-prefixes of the amine among them (``N,N-dimethyl``); the parent carries the suffix groups
    (``tetradecane-1,14-diol``,,:23405). None when an alkyl name is both on the chain and on N."""
    by_name: Dict[str, list] = {}
    for loc, name in num.prefixes:
        by_name.setdefault(name, []).append(loc)
    if any(name in by_name for name in num.n_prefixes):
        return None
    n_by_name: Dict[str, list] = {}
    for name in num.n_prefixes:
        n_by_name.setdefault(name, []).append("N")
    by_name.update(n_by_name)
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
    if note_onium_a_chain(mol):
        return None
    acyl = name_acyl_a_chain(mol)
    if acyl is not None:
        return acyl
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


# --- cations ------------------------------------------------------------------------------------------

#: an onium centre that may be a skeletal atom of an 'a' chain: (element, number of carbon neighbours) -> the word of
#: the mononuclear parent cation a substitutive name spells it with
_ONIUM_WORD = {("S", 3): "sulfanium", ("Se", 3): "selanium", ("Te", 3): "tellanium", ("P", 4): "phosphanium",
               ("As", 4): "arsanium"}
_ONIUM_ALLOWED = _ALLOWED_ATOMS | {"As"}


def _onium_word(atom: Chem.Atom) -> Optional[str]:
    if atom.GetFormalCharge() != 1 or atom.GetTotalNumHs() or any(nb.GetSymbol() != "C" for nb in atom.GetNeighbors()):
        return None
    return _ONIUM_WORD.get((atom.GetSymbol(), atom.GetDegree()))


def note_onium_a_chain(mol: Optional[Chem.Mol]) -> bool:
    """True, after recording the substitutive names of the molecule below the PIN, when the molecule is a cation whose
    PIN is a skeletal replacement ('a') name this module does not build.

    ``### **** SKELETAL REPLACEMENT ('a') NOMENCLATURE FOR CATIONS`` (the Blue Book): method (1), "name the
    compound using neutral skeletal replacement ('a') prefixes and then describe the cationic centers by the appropriate
    suffix 'ium'": ``3,12-dimethyl-6,9-dioxa-3,12-dithiatetradecane-3,12-diium (PIN)`` (:41785), not the substitutive
    ``1,2-bis{2-[ethyl(methyl)sulfaniumyl]ethoxy}ethane``. The chain holds four heterounits,:23348) when an
    onium centre of sulfur, selenium, tellurium, phosphorus or arsenic with carbon neighbours only counts as one. An
    aminium nitrogen is the principal characteristic group and no skeletal atom (:41775, '2-ethoxy-N-{2-[(2-ethoxyethyl)
    (methyl)sulfaniumyl]ethyl}-N,N-dimethylethanaminium (PIN)', "does not have four heterounits"), and so is the amine
    nitrogen of the neutral compound the cation is named from: a molecule with a nitrogen atom is left alone. The scope is the whole molecule named at the top level: one acyclic fragment, the
    atoms of this module, every other atom neutral, single bonds at every heteroatom; the chain is a path between two
    end atoms (C, or P, Si, Ge, B,:6465) with no two adjacent heteroatoms, an onium centre on it and four
    heteroatoms in all."""
    from ..assembly.fragment_naming import is_top_level_naming
    if mol is None or mol.GetNumHeavyAtoms() > 60 or not is_top_level_naming():
        return False
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        return False
    words = set()
    for atom in mol.GetAtoms():
        word = _onium_word(atom)
        if word:
            words.add(word)
        elif (atom.GetFormalCharge() or atom.GetSymbol() not in _ONIUM_ALLOWED or atom.GetNumRadicalElectrons()
              or atom.GetIsotope() or atom.GetSymbol() == "N"):
            return False
    if not words or any(b.GetBondType() != _SINGLE and not _carbon_bond(b) for b in mol.GetBonds()):
        return False
    # a chain may end at any carbon (or end heteroatom): '-CH2-OH' ends at the carbon, the OH is a substituent
    ends = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in {"C"} | _END_HETERO]
    for i, start in enumerate(ends):
        for end in ends[i + 1:]:
            path = Chem.GetShortestPath(mol, start, end)
            hetero = [k for k, idx in enumerate(path) if mol.GetAtomWithIdx(idx).GetSymbol() != "C"]
            if (len(hetero) >= _MIN_HETEROATOMS and all(b - a > 1 for a, b in zip(hetero, hetero[1:]))
                    and any(_onium_word(mol.GetAtomWithIdx(path[k])) for k in hetero)):
                from ..metrics.provenance import record_non_pin_label
                for word in sorted(words):
                    record_non_pin_label(word, rule="P-73.4", detail="the PIN of the cation is an 'a' name")
                return True
    return False


# --- acids and esters on a chain of skeletal amide nitrogens ---------------------------------------
#
# (the Blue Book): "The same number of characteristic groups that would be expressed as suffixes in
# substitutive names must be present in skeletal replacement ('a') names": 'methyl 7,14,21,28-tetraacetamido-2,9,16,23-
# tetraoxo-3,10,17,24-tetraazatriacontan-30-oate (PIN; an ester is senior to an amide or a ketone)' (:23428). With a
# carboxylic acid or its ester as the principal group, an amide nitrogen of the chain is a skeletal atom and its C=O an
# 'oxo' prefix; the acid carbons end the chain and carry the suffix on the fixed numbering,:6496;
#:29860 "locants are used when hydrocarbon chains are modified by skeletal replacement"). Vocabulary of this
# first step: one acyclic neutral fragment without stereo; on the chain carbons =O ('oxo'), OH, SH, NH2, halogens, the
# alkyl parts of ``_alkyl_name`` and an unbranched acylamino group ('acetamido'); one or two acid groups, all free or
# all esters of one alkyl. Anything else declines (None).

_LEAF_PREFIX = {"O": "hydroxy", "S": "sulfanyl", "N": "amino"}


def _part_from(mol: Chem.Mol, root: int, blocked: set) -> List[int]:
    part, seen = [root], {root} | blocked
    for idx in part:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in seen:
                seen.add(nb.GetIdx())
                part.append(nb.GetIdx())
    return part


def _acid_ends(mol: Chem.Mol) -> Optional[Dict[int, Tuple[str, frozenset]]]:
    """``{acid carbon: (ester alkyl name or '', the atoms of the group besides the carbon)}`` for the C(=O)OH and
    C(=O)OR groups whose carbon has one other heavy neighbour, a carbon; None when another acyl-oxygen group is present
    (a carbamate, a carbonate, an anhydride) or an ester part is not in the vocabulary."""
    ends: Dict[int, Tuple[str, frozenset]] = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        idx = atom.GetIdx()
        dbl = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"
               and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == _DOUBLE]
        sgl = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"
               and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == _SINGLE]
        if not (dbl and sgl):
            continue
        others = [n for n in atom.GetNeighbors() if n.GetSymbol() != "O"]
        if len(dbl) != 1 or len(sgl) != 1 or len(others) != 1 or others[0].GetSymbol() != "C":
            return None
        oxygen = sgl[0]
        if oxygen.GetDegree() == 1:
            ends[idx] = ("", frozenset({dbl[0].GetIdx(), oxygen.GetIdx()}))
            continue
        root = next(n.GetIdx() for n in oxygen.GetNeighbors() if n.GetIdx() != idx)
        part = _part_from(mol, root, {oxygen.GetIdx()})
        name = _alkyl_name(mol, root, part)
        if name is None:
            return None
        ends[idx] = (name, frozenset({dbl[0].GetIdx(), oxygen.GetIdx(), *part}))
    return ends


def _amide_n(mol: Chem.Mol, idx: int) -> bool:
    """An amide nitrogen: single bonds to carbons only, one of them a C=O carbon that is no acid carbon."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or any(b.GetBondType() != _SINGLE for b in atom.GetBonds()):
        return False
    if any(n.GetSymbol() != "C" for n in atom.GetNeighbors()):
        return False
    if any(b.GetBondType() == _DOUBLE and b.GetOtherAtom(n).GetSymbol() == "N" for n in atom.GetNeighbors()
           for b in n.GetBonds()):
        return False                            # an acylguanidine or acylamidine N: no amide N (the reader agrees)
    return sum(1 for n in atom.GetNeighbors() if any(
        b.GetBondType() == _DOUBLE and b.GetOtherAtom(n).GetSymbol() == "O" for b in n.GetBonds())) == 1


def _oxo_like(atom) -> bool:
    """A double bond to N, O, S or Se, or a triple bond to N (an aromatic bond is neither: the indole of Trp)."""
    return any((b.GetBondType() == _DOUBLE and b.GetOtherAtom(atom).GetSymbol() in ("N", "O", "S", "Se"))
               or (b.GetBondType() == _TRIPLE and b.GetOtherAtom(atom).GetSymbol() == "N") for b in atom.GetBonds())


def _guanidine_carbon(mol: Chem.Mol, idx: int) -> bool:
    """A neutral acyclic carbon with three nitrogen neighbours and one C=N bond: the carbon of a guanidine."""
    atom = mol.GetAtomWithIdx(idx)
    return (atom.GetSymbol() == "C" and atom.GetDegree() == 3 and not atom.IsInRing() and not atom.GetFormalCharge()
            and all(n.GetSymbol() == "N" for n in atom.GetNeighbors())
            and any(b.GetBondType() == _DOUBLE for b in atom.GetBonds()))


def _guanidine_n(mol: Chem.Mol, idx: int) -> bool:
    """The N of an Arg side chain between its carbon and the guanidine carbon. It may stand in a chain:
    (the Blue Book) '9-imino-2,8,10,16-tetraazaheptadecanediimidamide (PIN)' puts the nitrogens beside a C=NH
    carbon in the chain. A chain through it ends at the guanidine carbon with 'amino' and 'imino' prefixes; drawn as
    -N=C(NH2)2 the N=C bond would stand in the chain and need an 'ene' term, which this writer does not build, so that
    chain declines when it ranks first,:20916) and never wins unseen."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetDegree() != 2 or atom.IsInRing() or atom.GetFormalCharge():
        return False
    return any(_guanidine_carbon(mol, nb.GetIdx()) for nb in atom.GetNeighbors())


def _skeletal(mol: Chem.Mol, idx: int) -> bool:
    """A heteroatom that may stand in the chain: an amide nitrogen, or a saturated atom of which neither it nor a
    neighbour carries a double bond to N, O, S or Se (the reading of ``checks_skeletal_chain``): no nitrogen of a
    guanidine or amidine, whose group is a prefix."""
    if _amide_n(mol, idx) or _guanidine_n(mol, idx):
        return True
    atom = mol.GetAtomWithIdx(idx)
    return (all(b.GetBondType() == _SINGLE for b in atom.GetBonds())
            and not _oxo_like(atom) and not any(_oxo_like(n) for n in atom.GetNeighbors()))


def _acylamino_name(mol: Chem.Mol, root: int, part: List[int]) -> Optional[str]:
    """'formamido', 'acetamido', 'propanamido',... for -NH-CO-R with R an unbranched saturated alkyl
    method (1), the Blue Book "Method (1) generates preferred IUPAC names"); None otherwise. ``part`` is the whole
    branch from the nitrogen ``root``: the nitrogen, the C=O carbon, its oxygen and the alkyl carbons only."""
    n = mol.GetAtomWithIdx(root)
    if not _amide_n(mol, root) or n.GetTotalNumHs() != 1 or n.GetDegree() != 2:
        return None
    carbonyl = next((nb for nb in n.GetNeighbors() if nb.GetIdx() in part), None)
    if carbonyl is None or carbonyl.GetSymbol() != "C":
        return None
    oxo = [b.GetOtherAtom(carbonyl) for b in carbonyl.GetBonds()
           if b.GetBondType() == _DOUBLE and b.GetOtherAtom(carbonyl).GetSymbol() == "O"]
    rest = [nb for nb in carbonyl.GetNeighbors() if nb.GetIdx() not in (root, oxo[0].GetIdx() if oxo else -1)]
    if len(oxo) != 1 or oxo[0].GetDegree() != 1 or len(rest) > 1 or (rest and rest[0].GetSymbol() != "C"):
        return None
    if not rest:
        return "formamido" if len(part) == 3 else None
    alkyl = _part_from(mol, rest[0].GetIdx(), {carbonyl.GetIdx()})
    if len(alkyl) + 3 != len(part) or _alkyl_name(mol, rest[0].GetIdx(), alkyl) != get_chain_prefix(len(alkyl)) + "yl":
        return None
    n_carbons = len(alkyl) + 1
    return "acetamido" if n_carbons == 2 else get_chain_prefix(n_carbons) + "anamido"


def _acyl_pendant(mol: Chem.Mol, attach: int, root: int, chain: set, acid: set) -> Optional[str]:
    """The prefix name of what hangs from the chain atom ``attach`` at ``root`` in acid mode, or None."""
    bond = mol.GetBondBetweenAtoms(attach, root).GetBondType()
    atom = mol.GetAtomWithIdx(root)
    on_carbon = mol.GetAtomWithIdx(attach).GetSymbol() == "C"
    if root in acid:
        return None
    if bond == _DOUBLE:
        if on_carbon and atom.GetSymbol() == "N" and atom.GetDegree() == 1 and _guanidine_carbon(mol, attach):
            # the guanidine carbon that ends a chain through an Arg side chain: its =NH is 'imino',
            # the Blue Book, '9-imino-2,8,10,16-tetraazaheptadecanediimidamide (PIN)')
            return "imino"
        return "oxo" if on_carbon and atom.GetSymbol() == "O" and atom.GetDegree() == 1 else None
    if bond != _SINGLE:
        return None
    part = _part_from(mol, root, chain)
    if on_carbon and _guanidine_carbon(mol, attach) and len(part) != 1:
        return None                             # a substituted N on the guanidine carbon: not read
    if len(part) == 1 and on_carbon and atom.GetSymbol() != "C":
        symbol = atom.GetSymbol()
        return _HALOGEN_PREFIX.get(symbol) or _LEAF_PREFIX.get(symbol)
    if on_carbon and atom.GetSymbol() == "N":
        named = _acylamino_name(mol, root, part)
        if named:
            return named
        if _amide_n(mol, root) and not any(i in acid for i in part):
            # a substituted acylamino group ('2-aminoacetamido', the N-terminal residue beside a chain through an Arg
            # side chain): the complete name of the amide with its final 'e' changed to 'o' method (1),
            # the Blue Book), read back by OPSIN; None when its part holds a second amide nitrogen
            from ..assembly.n_acyl_amido import decorated_n_acyl_amido_prefix
            named = decorated_n_acyl_amido_prefix(mol, part)
            if named:
                return named
    named = _alkyl_name(mol, root, part)
    if named or not on_carbon:
        return named
    return _shared_prefix_name(mol, attach, root, part, acid)


def _shared_prefix_name(mol: Chem.Mol, attach: int, root: int, part: List[int], acid: set) -> Optional[str]:
    """The prefix of any other part on a chain carbon ('(4-hydroxyphenyl)methyl', 'hydroxymethyl', '4-aminobutyl')
    from the shared substituent namer, when the part holds no acid group or amide (whose principal-group status the
    chain is built on); None otherwise. The ring of such a part is no chain atom (the chain uses acyclic atoms)."""
    # a terminal NH2 of an amide (the carbamoyl end of an Asn or Gln side chain, '2-amino-2-oxoethyl') is no
    # skeletal amide nitrogen: it ends no chain, the chain ends are carbons)
    # an acylamino group on the chain carbon ('2-aminoacetamido', method (1), the Blue Book): its
    # nitrogen is the attachment atom and the one amide nitrogen of the part, so no chain of the structure runs on
    # through a second amide inside it
    inner = [i for i in part if _amide_n(mol, i) and mol.GetAtomWithIdx(i).GetDegree() > 1]
    if any(i in acid for i in part) or (inner and inner != [root]):
        return None
    if any(b.GetBondType() == _DOUBLE and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "N"}
           for i in part for b in mol.GetAtomWithIdx(i).GetBonds()):
        return _guanidine_alkyl(mol, root, part)     # a C=N group: only the Arg side chain is read
    from ..assembly.substituent_naming import name_substituent_fragment
    named = name_substituent_fragment(mol, sorted(part), root, [attach])
    return named if named and "substituent" not in named else None


def _guanidine_alkyl(mol: Chem.Mol, root: int, part: List[int]) -> Optional[str]:
    """'3-(carbamimidoylamino)propyl' or '3-[(diaminomethylidene)amino]propyl' for an unbranched saturated carbon
    chain from ``root`` whose last carbon carries the guanidine of an arginine side chain; None for any other part."""
    carbons, prev, here = [], None, root
    while True:
        atom = mol.GetAtomWithIdx(here)
        if atom.GetSymbol() != "C" or atom.IsInRing() or any(b.GetBondType() != _SINGLE for b in atom.GetBonds()):
            return None
        carbons.append(here)
        nxt = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != prev and n.GetIdx() in part]
        if len(nxt) != 1:
            return None
        if mol.GetAtomWithIdx(nxt[0]).GetSymbol() == "N":
            break
        prev, here = here, nxt[0]
    nh = mol.GetAtomWithIdx(nxt[0])
    if nh.GetDegree() != 2 or nh.GetFormalCharge():
        return None
    centre = next(n for n in nh.GetNeighbors() if n.GetIdx() != here)
    if centre.GetSymbol() != "C" or centre.GetDegree() != 3 or centre.GetIdx() not in part or centre.GetFormalCharge():
        return None
    # either tautomer of the guanidine: -NH-C(=NH)-NH2 or -N=C(NH2)2 (one structure for the InChI, mobile H)
    kinds = sorted((mol.GetBondBetweenAtoms(centre.GetIdx(), n.GetIdx()).GetBondType() == _DOUBLE,
                    n.GetTotalNumHs()) for n in centre.GetNeighbors()
                   if n.GetSymbol() == "N" and (n.GetDegree() == 1 or n.GetIdx() == nh.GetIdx())
                   and not n.GetFormalCharge())
    if len(part) != len(carbons) + 4:
        return None
    # (the Blue Book): 'carbamimidoylamino (preferred prefix)' for H2N-C(=NH)-NH- and
    # '(diaminomethylidene)amino (preferred prefix)' for (H2N)2C=N- ('4-[(diaminomethylidene)amino]butanoic acid
    # (PIN)',:34282): the tautomer as drawn
    if kinds == [(False, 1), (False, 2), (True, 1)]:
        group = "(carbamimidoylamino)"
    elif kinds == [(False, 2), (False, 2), (True, 0)]:
        group = "[(diaminomethylidene)amino]"
    else:
        return None
    n = len(carbons)
    return f"{n}-{group}{get_chain_prefix(n)}yl" if n > 1 else f"{group}methyl"


def _acyl_in_scope(mol: Optional[Chem.Mol]) -> bool:
    if mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() not in _ALLOWED_ATOMS or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetIsotope()):
            return False
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return False                        # E/Z, in a ring prefix too, is not cited by this writer
        if bond.IsInRing():
            continue                            # a ring is a prefix, named by the shared substituent namer
        kind = bond.GetBondType()
        if kind not in (_SINGLE, _DOUBLE, _TRIPLE):
            return False
        # a C=O is a suffix or an 'oxo' group; a C=N stands in a prefix (the guanidine of an Arg side chain,
        # '3-(carbamimidoylamino)propyl'): its atoms are no chain atoms (``_skeletal``) and a chain carbon that carries
        # one declines in ``_acyl_pendant``; a multiple bond between carbons needs ene/yne terms,
        #:6502), which this step does not build
        if kind != _SINGLE and not (
                kind == _DOUBLE and {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} in ({"C", "O"},
                                                                                                        {"C", "N"})):
            return False
    return True


def name_acyl_a_chain(mol: Optional[Chem.Mol]) -> Optional[str]:
    """The 'a' PIN of an acid or ester whose principal chain holds four or more skeletal amide nitrogens and other
    heteroatoms (see the section text), or None (also when a step fails: the other namers then see the molecule)."""
    try:
        return _name_acyl_a_chain(mol)
    except Exception:  # noqa: BLE001
        return None


def _stereo_labels(mol: Chem.Mol) -> Optional[Dict[int, str]]:
    """``{atom: 'R' or 'S'}`` of the tetrahedral stereocentres (CIP labels through ``perception.stereo``, the
    labeller the engine uses), or None when a centre has no R/S label (an r/s pseudoasymmetric centre)."""
    from ..perception.stereo import assign_stereochemistry
    centres = [a for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    if not centres:
        return {}
    assign_stereochemistry(mol)
    labels = {}
    for atom in centres:
        code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        if code not in ("R", "S"):
            return None
        labels[atom.GetIdx()] = code
    return labels


def _name_acyl_a_chain(mol: Optional[Chem.Mol]) -> Optional[str]:
    if not _acyl_in_scope(mol):
        return None
    ends = _acid_ends(mol)
    if not ends or len(ends) > 2:
        return None
    esters = {name for name, _ in ends.values()}
    if len(esters) != 1:
        return None                         # an acid beside an ester, or two different esters: not built here
    ester = esters.pop()
    acid_atoms = set().union(*(atoms for _, atoms in ends.values()))
    if not any(_amide_n(mol, a.GetIdx()) for a in mol.GetAtoms()):
        return None
    stereo = _stereo_labels(mol)
    if stereo is None:
        return None
    allowed = {a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in acid_atoms and not a.IsInRing()
               and (a.GetSymbol() == "C" or (a.GetSymbol() in _E and _skeletal(mol, a.GetIdx())))}
    chain_ends = [i for i in allowed if mol.GetAtomWithIdx(i).GetSymbol() == "C"]
    scored = []
    for k, start in enumerate(chain_ends):
        parent: Dict[int, int] = {}
        order = [start]
        for idx in order:
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = nb.GetIdx()
                if j in allowed and j != start and j not in parent:
                    parent[j] = idx
                    order.append(j)
        for stop in chain_ends[k + 1:]:
            if stop not in parent:
                continue
            chain = _path(parent, start, stop)
            if not _chain_atoms_ok(mol, chain):
                continue
            members = set(chain)
            pendants, known, covered = [], True, set(members)
            for a, b in zip(chain, chain[1:]):
                kind = mol.GetBondBetweenAtoms(a, b).GetBondType()
                # a double bond in the chain is cited by an 'ene' ending (e)); this writer reads one kind, the
                # N=C bond of a guanidine drawn -N=C(NH2)2 at the end of a chain through an Arg side chain
                if kind != _SINGLE and not (kind == _DOUBLE and (_guanidine_carbon(mol, a) or _guanidine_carbon(mol, b))):
                    known = False
            for idx in chain:
                if idx in ends:
                    continue
                for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nb.GetIdx() in members:
                        continue
                    name = _acyl_pendant(mol, idx, nb.GetIdx(), members, acid_atoms)
                    if name is not None:
                        name = _pendant_with_stereo(mol, nb.GetIdx(), members, name, stereo, covered)
                    if name is None:
                        known = False
                        break
                    pendants.append((idx, name))
                if not known:
                    break
            if any(i not in covered for i in stereo):
                known = False                   # a stereocentre in an ester part or a prefix not read here
            hetero = [mol.GetAtomWithIdx(i).GetSymbol() for i in chain if mol.GetAtomWithIdx(i).GetSymbol() in _E]
            groups = sum(1 for i in chain if i in ends)
            rank = (groups, len(hetero), len(chain)) + tuple(hetero.count(el) for el in _E_ORDER)
            scored.append((rank, known, tuple(chain), tuple(pendants)))
    if not scored:
        return None
    top = max(rank for rank, *_ in scored)
    best = [s for s in scored if s[0] == top]
    # then: the chain holds every acid group and four heterounits; a tie with a chain outside the
    # vocabulary declines,:23348)
    if top[0] != len(ends) or top[1] < _MIN_HETEROATOMS or not all(known for _, known, _, _ in best):
        return None
    names = {_acyl_numbered_name(mol, chain, pendants, ends, ester, {i: stereo[i] for i in chain if i in stereo})
             for _, _, chain, pendants in best}
    return names.pop() if len(names) == 1 and None not in names else None


_FREE_VALENCE = re.compile(r"-(\d+)-yl$")
_CITED = re.compile(r"^\(((?:\d+[RS],)*\d+[RS])\)-")


def _pendant_with_stereo(mol, root, members, name, stereo, covered) -> Optional[str]:
    """``name`` with the CIP descriptor of its stereocentre when the prefix holds one (an Ile or Thr side chain:
    '(2S)-butan-2-yl', '(1R)-1-hydroxyethyl'), else ``name``; None when the prefix holds a centre this step does not
    place: more than one, one that is not the attachment atom, or one in a ring. The descriptor is the label of the
    centre in the whole molecule, cited with the locant of the prefix, the Blue Book;
    descriptors of a substituent group precede its name inside its enclosing marks). Adds the centre to ``covered``."""
    part = _part_from(mol, root, members)
    centres = [i for i in part if i in stereo]
    if not centres:
        return name
    cited = _CITED.match(name)
    if cited:
        # the shared substituent namer cites the descriptors itself ('(2S)-butan-2-yl'): one for each centre
        if len(cited.group(1).split(",")) != len(centres):
            return None
        covered.update(centres)
        return name
    if centres != [root] or any(mol.GetAtomWithIdx(i).IsInRing() for i in part):
        return None
    found = _FREE_VALENCE.search(name)
    if found:
        locant = found.group(1)
    elif name.endswith("yl") and re.match(r"(\d+,)*1(,\d+)*-", name):
        locant = "1"                            # '1-hydroxyethyl': the free valence of an 'alkyl' name is at 1
    else:
        return None
    covered.add(root)
    return f"({locant}{stereo[root]})-{name}"


def _acyl_numbered_name(mol, chain, pendants, ends, ester, stereo=None) -> Optional[str]:
    """The name of the chain on its preferred orientation heteroatoms first,:6476; then the suffix,
    ; then the prefixes, (f), (g)); a tie that gives two names declines."""
    stereo = stereo or {}
    if any(idx not in chain for idx in stereo):
        return None                             # a stereocentre off the chain (in a prefix) is not cited here
    fits = []
    for oriented in (chain, chain[::-1]):
        locant = {idx: place + 1 for place, idx in enumerate(oriented)}
        hetero = tuple((locant[i], mol.GetAtomWithIdx(i).GetSymbol()) for i in oriented
                       if mol.GetAtomWithIdx(i).GetSymbol() in _E)
        suffix = tuple(sorted(locant[i] for i in oriented if i in ends))
        # the 'ene' of a double bond in the chain: its lower locant, after the heteroatoms and the suffix and before the
        # prefixes 'NUMBERING', criterion (e)(i), the Blue Book: 'low locants are given to... 'ene' and
        # 'yne' endings')
        enes = tuple(sorted(min(locant[a], locant[b]) for a, b in zip(oriented, oriented[1:])
                            if mol.GetBondBetweenAtoms(a, b).GetBondType() == _DOUBLE))
        prefixes = tuple(sorted((locant[idx], name) for idx, name in pendants))
        first = min((name for _, name in prefixes), key=_cite_key, default=None)
        key = (tuple(loc for loc, _ in hetero),
               tuple(tuple(loc for loc, el in hetero if el == element) for element in _E_ORDER),
               suffix, enes, tuple(loc for loc, _ in prefixes), tuple(loc for loc, name in prefixes if name == first))
        descriptors = tuple(sorted((locant[idx], label) for idx, label in stereo.items()))
        # (j) (the Blue Book): last, "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r"
        key = key + (tuple(loc for loc, label in descriptors if label == "R"),)
        fits.append((key, hetero, suffix, prefixes, descriptors, enes))
    best = min(fit[0] for fit in fits)
    names = {_acyl_text(len(chain), hetero, suffix, prefixes, ester, descriptors, enes)
             for key, hetero, suffix, prefixes, descriptors, enes in fits if key == best}
    return names.pop() if len(names) == 1 else None


def _acyl_cited(name: str, locs: List[int]) -> str:
    """A prefix with its locants and multiplier. Three or more acylamino groups take the basic multiplier, as the book
    prints them ('7,14,21,28-tetraacetamido',:23428); two keep 'bis' (``format_substituent_prefix``), since
    'diacetamido' names (CH3-CO)2N-:33107, '... (not diacetamido)')."""
    if name.endswith("amido") and len(locs) >= 3:
        from ..assembly.naming_utils import simple_multiplier_word
        from ..data.locant_text import render_locants
        return f"{render_locants(locs)}-{simple_multiplier_word(len(locs))}{name}"
    return format_substituent_prefix(name, locs, len(locs))


def _acyl_text(length, hetero, suffix, prefixes, ester, descriptors=(), enes=()) -> str:
    by_name: Dict[str, list] = {}
    for loc, name in prefixes:
        by_name.setdefault(name, []).append(loc)
    word = "dioic acid" if len(suffix) == 2 else "oic acid"
    parent = _build_replacement_name(length, list(hetero), (word, list(suffix) if len(suffix) == 2 else suffix[0]),
                                     ene_locants=list(enes) or None)
    cited = [_acyl_cited(name, sorted(locs)) for name, locs in sorted(by_name.items(), key=lambda item: _cite_key(item[0]))]
    text = "-".join(cited + [parent])
    if descriptors:
        # (the Blue Book): the CIP descriptors with their locants in front of the complete name (of the
        # anion part of an ester: 'methyl (2S)-...-oate')
        from ..data.locant_text import render_locants
        text = f"({render_locants(f'{loc}{label}' for loc, label in descriptors)})-{text}"
    if not ester:
        return text
    # the ester: the alkyl word of the alcohol part, then the anion of the acid,:31663)
    anion = text[:-len(word)] + ("dioate" if len(suffix) == 2 else "oate")
    if len(suffix) == 2:
        # 'di-tert-butyl' (:25717), 'di(propan-2-yl)' (:36921), 'dimethyl': the shared multiplier of a prefix name
        ester = format_substituent_prefix(ester, [], 2)
    return f"{ester} {anion}"
