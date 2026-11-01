"""A (C60-Ih)[5,6]fullerene name read back by building the structure it states, apart from the producer.

OPSIN 2.9.0 (and OPSIN's main branch) reads no fullerene name: neither '(C60-Ih)[5,6]fullerene' nor any
derivative parses, so a fullerene name has no read-back by OPSIN and used to be declined. This module
reads a name of the class the producer builds (``rules.bridged_fused_pin.fullerene``: a C60 skeleton with
at most one addend on a [6,6] bond) as the structure it states and compares that structure with the input
by the FULL standard InChIKey and by the fixed-hydrogen layer, which tells the tautomers of a ring N-H
apart. It shares no code with the producer: the name is parsed here, the skeleton is the fact table of
``data.fullerene_c60`` (verified by its own tests), the attached monocycles are this module's own table
(``_COMPONENTS``, checked against OPSIN's reading of the ring names by the tests), the suffix groups are
this module's own table, and what is compared is the key of the built molecule against the key of the input.
OPSIN reads only the substituent names (``<prefix>silane``), the ester alkyl words and the outer name of a
fullerene that is a substituent of a larger structure.

* Systematic names (the Blue Book): '(C60-Ih)[5,6]fullerene (PIN)'; NUMBERING OF
  FULLERENES (:15336), the locants of ``data.fullerene_c60``.
* Fullerenes and modified fullerenes ortho fused to organic rings or ring systems (:15464-15480):
  "the pair of atoms shared by the fullerene or modified fullerene and the organic ring or ring system is
  regarded as part of both components... each component retains its own bonding pattern and numbering.
  Because of the nature of bonding in fullerenes, the fusion bond is always a single bond and the fusion
  atoms cannot accept an 'exo' double bond. After fusion, nonfullerene components other than alicyclic
  bi- and polycyclic ring systems have the maximum number of noncumulative double bonds and indicated
  hydrogen is cited as needed.... The fullerene or modified fullerene locants are always unprimed, and
  primes are added to the fused organic ring... The fusion is described by citing the primed locants of the
  organic ring or ring system component and the unprimed locants of the fullerene... in that order,
  enclosed in brackets and separated by a colon. Locants for monocyclic hydrocarbons are omitted in
  preferred IUPAC names." ('3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene', '[1,3]dioxolo[4',5':1,9]
  (C60-Ih)[5,6]fullerene'.) A monocyclic hydrocarbon's atoms take the locants 1' and 2' at the fusion bond:
  which cage atom is 1' is not stated, and it does not matter here: the two cage atoms of the fusion bond are
  equivalent in the cage with no other addend (a symmetry of the cage swaps them), so both readings give one
  molecule.
*, and (hydro prefixes, indicated hydrogen, added hydrogen): the atoms
  named by 'hydro', by indicated hydrogen ('3'H') and by added hydrogen ('1(9H)') are saturated; the other
  atoms that can carry a ring double bond have the maximum number of noncumulative double bonds, a perfect
  matching, or the name is not read. (:26870) '(C60-Ih)[5,6]fulleren-1(9H)-ol', (:25834)
  '1-(trifluoromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene'.
* (c) (:7619): the final 'e' of 'fullerene' is elided before a suffix that begins with a vowel.

The check proves the structure and never the spelling; a name this module cannot read (another component, an
unlocanted prefix, a suffix outside the table, a structure of another class) is not read back.
"""
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..data.fullerene_c60 import C60_CORE_TOKEN, C60_NEIGHBOURS

#: standard bonding numbers, the Blue Book) of the ring atoms this reader builds
_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "Si": 4, "B": 3}

#: the attached monocycles: attached-prefix text -> the elements at the locants 1', 2',... of the component's
#: own numbering, the Blue Book: the locant 1 on the senior heteroatom, then the lowest
#: locants to the heteroatoms), each read off the Hantzsch-Widman name ('[1,3]dioxole' O at 1 and 3, '1,2-oxazole'
#: O1 N2, 'pyridazine' = 1,2-diazine N1 N2). ``tests/unit/validation/test_fullerene_read_back.py`` checks every
#: row against OPSIN's own locants of the ring name.
_COMPONENTS: Dict[str, str] = {
    "oxireno": "OCC", "thiireno": "SCC", "azirino": "NCC", "silireno": "SiCC",
    "furo": "OCCCC", "thieno": "SCCCC", "pyrrolo": "NCCCC", "silolo": "SiCCCC",
    "[1,3]dioxolo": "OCOCC", "[1,3]disilolo": "SiCSiCC", "[1,2,3]trioxolo": "OOOCC",
    "[1,3,2]dioxaborolo": "OBOCC", "[1,2]oxazolo": "ONCCC", "[1,3]oxazolo": "OCNCC",
    "[1,2]thiazolo": "SNCCC", "[1,3]thiazolo": "SCNCC", "imidazo": "NCNCC", "pyrazolo": "NNCCC",
    "[1,2,3]triazolo": "NNNCC", "[1,2,4]triazolo": "NNCNC", "[1,3]dithiolo": "SCSCC",
    "pyrido": "NCCCCC", "pyridazino": "NNCCCC", "pyrazino": "NCCNCC", "pyrimido": "NCNCCC",
    "[1,4]dioxino": "OCCOCC", "[1,2]oxazino": "ONCCCC", "[1,3]oxazino": "OCNCCC",
    "cyclopropa": "CCC", "cyclobuta": "CCCC", "cyclopenta": "CCCCC", "benzo": "CCCCCC",
    "cyclohepta": "CCCCCCC",
    "azeto": "NCCC", "oxeto": "OCCC", "thieto": "SCCC", "pyrano": "OCCCCC", "thiopyrano": "SCCCCC",
    "azepino": "NCCCCCC", "oxepino": "OCCCCCC", "thiepino": "SCCCCCC", "tetrazolo": "NNNNC",
    "[1,3]oxathiolo": "OCSCC", "[1,2]oxathiolo": "OSCCC", "[1,2]dithiolo": "SSCCC", "[1,3]dioxino": "OCOCCC",
    "[1,2,4]oxadiazolo": "ONCNC", "[1,3,4]oxadiazolo": "OCNNC", "[1,2,5]oxadiazolo": "ONCCN",
    "[1,2,3]oxadiazolo": "ONNCC", "[1,2,4]thiadiazolo": "SNCNC", "[1,3,4]thiadiazolo": "SCNNC",
}
_COMPONENT_NAMES = sorted(_COMPONENTS, key=len, reverse=True)

#: (Table 1.4) multiplying prefixes of simple and of compound prefixes
_MULTIPLIER = {2: ("di", "bis"), 3: ("tri", "tris"), 4: ("tetra", "tetrakis"), 5: ("penta", "pentakis"),
               6: ("hexa", "hexakis")}
_HYDRO_COUNT = {"di": 2, "tetra": 4, "hexa": 6, "octa": 8, "deca": 10, "dodeca": 12}

#: the suffix groups, each as the atoms bonded to the ring atom (a SMILES with the attaching atom first);
#: 'one' is the ring atom's own =O (two hydrogen atoms replaced) and 'yl' the free valence of a substituent unit
_SUFFIX = {"ol": "O", "thiol": "S", "amine": "N", "carboxylic acid": "C(=O)O", "carbonitrile": "C#N",
           "carbaldehyde": "C=O", "carboxamide": "C(N)=O", "carboxylate": "C(=O)O", "one": "=O", "yl": "",
           "carbonyl": "C=O"}
#: the suffixes that are the free valence of a substituent unit ('yl' on the ring atom, 'carbonyl' on the carbon atom
#: of the group)
_FREE_VALENCE = ("yl", "carbonyl")
#: prefixes that are a double bond to the ring atom, which saturates it
_DOUBLE_BOND_PREFIX = {"oxo": "O", "sulfanylidene": "S"}
#: (the Blue Book): the final 'e' of the stem is elided before a letter 'a', 'e', 'i', 'o', 'u' or 'y'
_ELIDING = "aeiouy"

_LOC = r"\d+'*"
_LOCS = rf"{_LOC}(?:,{_LOC})*"
#: locants of the prefixes: ring locants, or 'N' for the nitrogen atom of an amide or amine suffix...
#: the locant 'N' of a substituent on the nitrogen atom of the suffix group)
_PREFIX_LOCS = rf"(?:{_LOC}|N)(?:,(?:{_LOC}|N))*"
_IH_RE = re.compile(rf"^(?:{_LOC}H)(?:,{_LOC}H)*$")
_LOCS_RE = re.compile(rf"^{_PREFIX_LOCS}$")
_DESCRIPTOR_RE = re.compile(r"\[(?:(?P<att>\d+',\d+'):)?1,9\]$")
_SUFFIX_RE = re.compile(
    rf"^(?P<e>e)?-(?P<locs>{_LOCS})(?:\((?P<added>{_LOC}H(?:,{_LOC}H)*)\))?-(?P<mult>(?:di|tri|tetra)?)"
    rf"(?P<suffix>{'|'.join(sorted(_SUFFIX, key=len, reverse=True))})$")


class _Unreadable(Exception):
    """The name is outside the class this reader certifies."""


def fullerene_name_shape(name: Optional[str]) -> bool:
    """Whether ``name`` holds the (C60-Ih)[5,6]fullerene core token: the shape the validity gate sends here."""
    return bool(name) and C60_CORE_TOKEN in name


# --------------------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------------------

def _depth_scan(name: str, start: int, step: int) -> int:
    """The index where the unit that holds the character at ``start - step`` ends, scanning in direction
    ``step``: the first unmatched enclosing mark, a space at depth 0, or the end of the name."""
    depth = 0
    i = start
    openers, closers = ("([{", ")]}") if step > 0 else (")]}", "([{")
    while 0 <= i < len(name):
        ch = name[i]
        if ch in openers:
            depth += 1
        elif ch in closers:
            if depth == 0:
                return i
            depth -= 1
        elif ch == " " and depth == 0:
            return i
        i += step
    return i


def _split_top(text: str, sep: str = "-") -> List[str]:
    """``text`` split at ``sep`` outside every enclosing mark."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _enclosed_once(text: str) -> bool:
    """True when ``text`` is one enclosing pair: '(trifluoromethyl)'."""
    if not text or text[0] not in "([{" or text[-1] not in ")]}":
        return False
    depth = 0
    for i, ch in enumerate(text):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth == 0 and i < len(text) - 1:
            return False
    return depth == 0


# --------------------------------------------------------------------------------------
# OPSIN: substituent names, ester words, the outer name of a substituent unit
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _opsin(name: str) -> Optional[str]:
    from .opsin_roundtrip import opsin_parse
    return opsin_parse(name)


def _marker_for(*texts: str) -> Tuple[str, int]:
    """(parent word of the marker atom, its atomic number): silicon, or germanium when a text holds silicon."""
    low = " ".join(texts).lower()
    has_si, has_ge = "sil" in low, "germ" in low
    if has_si and has_ge:
        raise _Unreadable("both markers present")
    return ("germane", 32) if has_si else ("silane", 14)


def _refuse_marked(mol: Chem.Mol, what: str) -> None:
    """Refuse a part OPSIN reads with a charge, a radical, an isotope label or a stereo mark. The skeleton is rebuilt
    from element symbols and bond orders only, so such a mark would be dropped and a name that states another species
    ('...pyridin-1-ium', '(2R)-...', '(2H3)methane') would read as the input; the producer names only neutral,
    unlabelled, stereo-free inputs (``rules.bridged_fused_pin.fullerene``), so no name of its own carries one."""
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
           or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        raise _Unreadable(f"{what}: a charged, labelled or stereo atom")
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE or b.GetBondDir() != Chem.BondDir.NONE
           for b in mol.GetBonds()):
        raise _Unreadable(f"{what}: a stereo bond")


def _graft_source(word: str, marker: Tuple[str, int]) -> Tuple[Chem.Mol, int]:
    """(fragment, attaching atom) of the substituent ``word`` read by OPSIN as ``(word)silane``: the marker atom is
    removed and its one neighbour is the attaching atom of the returned fragment."""
    text = word if _enclosed_once(word) else f"({word})"
    smiles = _opsin(f"{text}{marker[0]}")
    if not smiles:
        raise _Unreadable(f"OPSIN does not read {word!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _Unreadable("unreadable fragment")
    marks = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == marker[1]]
    if len(marks) != 1 or mol.GetAtomWithIdx(marks[0]).GetDegree() != 1:
        raise _Unreadable(f"{word!r} has no single attaching atom")
    _refuse_marked(mol, word)
    attach = mol.GetAtomWithIdx(marks[0]).GetNeighbors()[0].GetIdx()
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(marks[0])
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    Chem.Kekulize(out, clearAromaticFlags=True)
    return out, attach - (1 if attach > marks[0] else 0)


# --------------------------------------------------------------------------------------
# the unit: parse
# --------------------------------------------------------------------------------------

class _Unit:
    """The parsed unit: prefixes (locant, name), hydro / indicated / added hydrogen locants, the component,
    the descriptor's attached locants and the suffix."""

    def __init__(self):
        self.prefixes: List[Tuple[str, str]] = []      # (locant, name), once per locant
        self.saturated: List[str] = []                  # hydro, indicated hydrogen, added hydrogen locants
        self.component: Optional[str] = None
        self.attached: Optional[Tuple[str, str]] = None  # the attached locants cited, in the order of 1, 9
        self.suffix: Optional[str] = None
        self.suffix_locants: List[str] = []
        self.free_valence = False
        self.added = False                              # the suffix cites added hydrogen


def _parse_prefixes(text: str, unit: _Unit) -> None:
    """The detachable prefixes, hydro prefixes and indicated hydrogen in front of the fusion prefix."""
    text = text.strip("-")
    if not text:
        return
    tokens = _split_top(text)
    pending: Optional[str] = None                       # a locant list waiting for its name
    name_tokens: List[str] = []

    def flush():
        nonlocal pending, name_tokens
        if pending is None:
            if name_tokens:
                raise _Unreadable("a prefix without locants")
            return
        if not name_tokens:
            raise _Unreadable("locants without a prefix")
        name = "-".join(name_tokens)
        locs = pending.split(",")
        hydro = re.match(r"^([a-z]+?)hydro$", name)
        if hydro:
            if hydro.group(1) not in _HYDRO_COUNT or _HYDRO_COUNT[hydro.group(1)] != len(locs) or "N" in locs:
                raise _Unreadable("hydro prefix count")
            unit.saturated.extend(locs)
        else:
            if len(locs) > 1:
                words = _MULTIPLIER.get(len(locs))
                stripped = None
                for w in words or ():
                    if name.startswith(w):
                        stripped = name[len(w):]
                        break
                if stripped is None:
                    raise _Unreadable("multiplier does not match the locants")
                name = stripped
            for loc in locs:
                unit.prefixes.append((loc, name))
        pending, name_tokens = None, []

    for token in tokens:
        if _IH_RE.match(token):
            flush()
            unit.saturated.extend(x[:-1] for x in token.split(","))
        elif _LOCS_RE.match(token):
            flush()
            pending = token
        else:
            name_tokens.append(token)
    flush()


def _split_head(head: str) -> Tuple[str, Optional[str], Optional[Tuple[str, str]]]:
    """(prefix text, component prefix or None, attached locants or None) of the text in front of the core."""
    found = _DESCRIPTOR_RE.search(head)
    if found is None:
        return head, None, None
    before = head[:found.start()]
    for comp in _COMPONENT_NAMES:
        if before.endswith(comp):
            attached = tuple(found.group("att").split(",")) if found.group("att") else None
            n_het = any(ch != "C" for ch in re.findall(r"[A-Z][a-z]?", _COMPONENTS[comp]))
            if n_het != (attached is not None):
                raise _Unreadable("attached locants cited for a hydrocarbon or missing for a heterocycle")
            return before[:len(before) - len(comp)], comp, attached
    raise _Unreadable("unknown attached component")


def _parse_unit(text: str) -> _Unit:
    unit = _Unit()
    at = text.find(C60_CORE_TOKEN)
    head, tail = text[:at], text[at + len(C60_CORE_TOKEN):]
    prefix_text, comp, attached = _split_head(head)
    unit.component, unit.attached = comp, attached
    _parse_prefixes(prefix_text, unit)
    if tail == "e":
        return unit
    found = _SUFFIX_RE.match(tail)
    if found is None:
        raise _Unreadable("ending outside the table")
    suffix = found.group("suffix")
    locs = found.group("locs").split(",")
    words = {"": 1, "di": 2, "tri": 3, "tetra": 4}
    if len(locs) != words[found.group("mult")]:
        raise _Unreadable("suffix count does not match its locants")
    begins_vowel = (found.group("mult") or suffix)[0] in _ELIDING
    if bool(found.group("e")) == begins_vowel:
        raise _Unreadable("the final e of the stem is misspelled")           # (c)
    if "N" in locs:
        raise _Unreadable("suffix locant")
    if suffix in _FREE_VALENCE:
        unit.free_valence = True
        if len(locs) != 1:
            raise _Unreadable("one free valence")
    unit.suffix, unit.suffix_locants = suffix, locs
    if found.group("added"):
        unit.added = True
        unit.saturated.extend(x[:-1] for x in found.group("added").split(","))
    return unit


# --------------------------------------------------------------------------------------
# the unit: build
# --------------------------------------------------------------------------------------

def _max_matching(n: int, adj: List[List[int]]) -> List[int]:
    """A maximum matching of a general graph (Edmonds' blossom algorithm): ``match[v]`` is the partner or -1."""
    match = [-1] * n
    parent = [-1] * n
    base = list(range(n))
    used = [False] * n
    blossom = [False] * n

    def lca(a, b):
        seen = [False] * n
        while True:
            a = base[a]
            seen[a] = True
            if match[a] == -1:
                break
            a = parent[match[a]]
        while True:
            b = base[b]
            if seen[b]:
                return b
            b = parent[match[b]]

    def mark(v, b, child):
        while base[v] != b:
            blossom[base[v]] = blossom[base[match[v]]] = True
            parent[v] = child
            child = match[v]
            v = parent[match[v]]

    def find(root):
        for i in range(n):
            used[i], parent[i], base[i] = False, -1, i
        used[root] = True
        queue = [root]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for to in adj[v]:
                if base[v] == base[to] or match[v] == to:
                    continue
                if to == root or (match[to] != -1 and parent[match[to]] != -1):
                    cur = lca(v, to)
                    for i in range(n):
                        blossom[i] = False
                    mark(v, cur, to)
                    mark(to, cur, v)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = cur
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if match[to] == -1:
                        return to
                    used[match[to]] = True
                    queue.append(match[to])
        return -1

    for root in range(n):
        if match[root] == -1:
            v = find(root)
            while v != -1:
                pv = parent[v]
                ppv = match[pv]
                match[v], match[pv] = pv, v
                v = ppv
    return match


class _Graph:
    """The atoms and bonds of the unit being built; ``loc`` maps a locant ('9', "3'") to an atom."""

    def __init__(self):
        self.symbols: List[str] = []
        self.bonds: Dict[Tuple[int, int], int] = {}     # (low, high) -> order
        self.system: Set[int] = set()                    # the cage and attached ring atoms
        self.loc: Dict[str, int] = {}

    def add(self, symbol: str) -> int:
        self.symbols.append(symbol)
        return len(self.symbols) - 1

    def bond(self, a: int, b: int, order: int = 1) -> None:
        self.bonds[(min(a, b), max(a, b))] = order

    def graft(self, frag: Chem.Mol, attach: int, onto: int) -> int:
        """Add the molecule ``frag`` and bond its atom ``attach`` to atom ``onto``; the index of the first new atom."""
        base = len(self.symbols)
        for atom in frag.GetAtoms():
            self.add(atom.GetSymbol())
        for b in frag.GetBonds():
            self.bond(base + b.GetBeginAtomIdx(), base + b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()))
        self.bond(base + attach, onto)
        return base


def _build_skeleton(unit: _Unit) -> _Graph:
    g = _Graph()
    for loc in range(1, 61):
        g.loc[str(loc)] = g.add("C")
    for a, ns in C60_NEIGHBOURS.items():
        for b in ns:
            if a < b:
                g.bond(g.loc[str(a)], g.loc[str(b)])
    g.system = set(range(60))
    if unit.component is None:
        return g
    elements = re.findall(r"[A-Z][a-z]?", _COMPONENTS[unit.component])
    n = len(elements)
    if unit.attached is None:
        # a monocyclic hydrocarbon: fusion atoms 1', 2' (the cage atoms 1 and 9 are interchangeable here, see the
        # module docstring), the other atoms numbered on from 2' round the ring
        first, second = 1, 2
    else:
        first, second = (int(x.rstrip("'")) for x in unit.attached)
        if any(x.count("'") != 1 for x in unit.attached) or {first, second} - set(range(1, n + 1)):
            raise _Unreadable("attached locants outside the ring")
        if not ((second - first) % n in (1, n - 1)):
            raise _Unreadable("the fusion atoms are not adjacent in the attached ring")
        if elements[first - 1] != "C" or elements[second - 1] != "C":
            raise _Unreadable("a fusion atom of the attached ring is no carbon atom")
    ring: Dict[int, int] = {}
    for pos in range(1, n + 1):
        if pos == first:
            ring[pos] = g.loc["1"]
        elif pos == second:
            ring[pos] = g.loc["9"]
        else:
            ring[pos] = g.add(elements[pos - 1])
            g.loc[f"{pos}'"] = ring[pos]
    for pos in range(1, n + 1):
        nxt = pos % n + 1
        if {pos, nxt} == {first, second}:
            continue                                   # the fusion bond is the cage bond 1-9
        g.bond(ring[pos], ring[nxt])
    g.system |= set(ring.values())
    return g


def _atom_at(g: _Graph, loc: str) -> int:
    if loc not in g.loc:
        raise _Unreadable(f"no atom with locant {loc}")
    return g.loc[loc]


def _finish(unit: _Unit, g: _Graph, outer: Optional[Tuple[Chem.Mol, int]], marker,
            ester: Optional[Tuple[Chem.Mol, int]] = None) -> Chem.Mol:
    """Suffix groups, substituents, the saturated atoms, the maximum matching of the rest, then the molecule.

    The atoms of the suffix groups that have no hydrogen to give while they carry a double bond (the cage atoms: three
    ring bonds and one free valence) are saturated by the group itself: (the Blue Book) "'Added
    indicated hydrogen' atoms are not cited when the accommodation of a pair of principal characteristic groups or
    free valences simply removes a double bond (directly or after rearrangement of double bonds) from the parent ring
    structure" ('anthracene-9,10-dione', 'pyrazine-2,3-dione'); a single such group is not read without its added
    hydrogen, because the atoms left over then do not pair."""
    saturated: Set[int] = set()
    for loc in unit.saturated:
        a = _atom_at(g, loc)
        if a not in g.system or a in saturated:
            raise _Unreadable("hydrogen cited twice or outside the parent")
        saturated.add(a)
    sigma = {a: 0 for a in g.system}
    for (a, b) in g.bonds:
        if a in g.system and b in g.system:
            sigma[a] += 1
            sigma[b] += 1

    def room(a):
        return _VALENCE.get(g.symbols[a], 0) - sigma[a]

    free_atom = None
    if unit.suffix:
        for loc in unit.suffix_locants:
            atom = _atom_at(g, loc)
            if unit.suffix == "one":
                g.bond(g.add("O"), atom, 2)
                saturated.add(atom)
                continue
            if unit.suffix == "yl":
                free_atom = atom
            elif unit.suffix == "carboxylate":
                if ester is None:
                    raise _Unreadable("a carboxylate without its ester word")
                base = g.graft(Chem.MolFromSmiles("C(=O)O"), 0, atom)
                g.graft(ester[0], ester[1], base + 2)
            else:
                base = g.graft(Chem.MolFromSmiles(_SUFFIX[unit.suffix]), 0, atom)
                if unit.suffix == "carbonyl":
                    free_atom = base                     # the free valence is the carbon atom of the group
                elif unit.suffix in ("amine", "carboxamide") and len(unit.suffix_locants) == 1:
                    g.loc["N"] = base + (0 if unit.suffix == "amine" else 1)     # the locant N of the prefixes
            if room(atom) == 1 or unit.added:
                saturated.add(atom)
    if ester is not None and unit.suffix != "carboxylate":
        raise _Unreadable("an ester word for a unit that is no carboxylate")
    for loc, name in unit.prefixes:
        atom = _atom_at(g, loc)
        if name in _DOUBLE_BOND_PREFIX:
            if loc == "N":
                raise _Unreadable("a double bond to the nitrogen locant")
            g.bond(g.add(_DOUBLE_BOND_PREFIX[name]), atom, 2)
            saturated.add(atom)
            continue
        frag, attach = _graft_source(name, marker)
        g.graft(frag, attach, atom)
    if free_atom is not None:
        if outer is None:
            raise _Unreadable("a substituent unit with no outer name")
        g.graft(outer[0], outer[1], free_atom)
    elif outer is not None:
        raise _Unreadable("an outer name for a unit with no free valence")
    # the maximum number of noncumulative double bonds among the atoms that can carry one in the parent: the atoms
    # of the cage and the attached ring with a free valence after their ring bonds
    eligible = [a for a in sorted(g.system) if room(a) >= 1]
    if saturated - set(eligible):
        raise _Unreadable("hydrogen cited at an atom that cannot carry a double bond")
    free = [a for a in eligible if a not in saturated]
    if len(free) % 2:
        raise _Unreadable("an odd number of atoms for the double bonds")
    index = {a: i for i, a in enumerate(free)}
    adj: List[List[int]] = [[] for _ in free]
    for (a, b) in g.bonds:
        if a in index and b in index:
            adj[index[a]].append(index[b])
            adj[index[b]].append(index[a])
    match = _max_matching(len(free), adj)
    if any(m == -1 for m in match):
        raise _Unreadable("the atoms left do not take a double bond each: hydrogen is missing from the name")
    for i, m in enumerate(match):
        if i < m:
            g.bond(free[i], free[m], 2)
    rw = Chem.RWMol()
    for symbol in g.symbols:
        rw.AddAtom(Chem.Atom(symbol))
    for (a, b), order in g.bonds.items():
        rw.AddBond(a, b, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[order])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _outer_molecule(outer_name: str, marker) -> Tuple[Chem.Mol, int]:
    """(molecule, atom next to the marker) of the outer name read by OPSIN, the marker atom removed."""
    smiles = _opsin(outer_name)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise _Unreadable("OPSIN does not read the outer name")
    marks = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == marker[1]]
    if len(marks) != 1 or mol.GetAtomWithIdx(marks[0]).GetDegree() != 1:
        raise _Unreadable("the marker is not one atom with one neighbour")
    _refuse_marked(mol, "the outer name")
    neighbour = mol.GetAtomWithIdx(marks[0]).GetNeighbors()[0].GetIdx()
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(marks[0])
    rest = rw.GetMol()
    Chem.SanitizeMol(rest)
    Chem.Kekulize(rest, clearAromaticFlags=True)
    return rest, neighbour - (1 if neighbour > marks[0] else 0)


def _ester_word(before: str, unit: _Unit, marker) -> Tuple[Chem.Mol, int]:
    """(alkyl fragment, attaching atom) of the one ester word in front of a carboxylate unit; with several carboxylate
    groups the word is multiplied ('dimethyl...-dicarboxylate', 'bis(2-ethylhexyl)...')."""
    words = before.split()
    if unit.suffix != "carboxylate" or not before.endswith(" ") or len(words) != 1:
        raise _Unreadable("words round a unit that is no ester")
    word, count = words[0], len(unit.suffix_locants)
    if count > 1:
        stripped = next((word[len(w):] for w in _MULTIPLIER.get(count, ()) if word.startswith(w)), None)
        if stripped is None:
            raise _Unreadable("the ester word is not multiplied like the groups")
        word = stripped
    return _graft_source(word, marker)


def _read(name: str) -> Chem.Mol:
    at = name.find(C60_CORE_TOKEN)
    if at < 0 or name.find(C60_CORE_TOKEN, at + 1) >= 0:
        raise _Unreadable("not one core token")
    start = _depth_scan(name, at - 1, -1)
    u0 = start + 1 if 0 <= start < len(name) else 0     # the scan stops at a mark or a space, or runs off the front
    end = _depth_scan(name, at + len(C60_CORE_TOKEN), 1)
    if name[:end].endswith("carboxylic") and name[end:].startswith(" acid"):
        end += len(" acid")                              # the one suffix that holds a space
    unit = _parse_unit(name[u0:end])
    before, after = name[:u0], name[end:]
    marker = _marker_for(name)
    g = _build_skeleton(unit)
    outer = ester = None
    if unit.free_valence:
        # the unit is a substituent: it is the group inside the innermost enclosing marks, or a word of its own (the
        # alcohol part of an ester, 'X-yl acetate'); the outer name is the name with that group replaced by 'silyl'
        # ('germyl'), and the one atom next to the marker is bonded to the free valence
        word = "silyl" if marker[1] == 14 else "germyl"
        if before.endswith(("(", "[", "{")) and after[:1] in (")", "]", "}"):
            outer_name = before[:-1] + "(" + word + ")" + after[1:]      # the marks stay, so no neighbour prefix merges
        elif (before == "" or before.endswith(" ")) and after.startswith(" "):
            outer_name = before + word + after
        else:
            raise _Unreadable("a substituent unit outside enclosing marks")
        outer = _outer_molecule(outer_name, marker)
    elif before or after:
        if after:
            raise _Unreadable("words round a unit that is no ester")
        ester = _ester_word(before, unit, marker)
    return _finish(unit, g, outer, marker, ester)


# --------------------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------------------

def _keys(mol: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
    try:
        return Chem.MolToInchiKey(mol) or None, Chem.MolToInchi(mol, options="-FixedH") or None
    except Exception:
        return None, None


def fullerene_read_back(name: Optional[str], smiles: Optional[str]) -> bool:
    """True when the structure ``name`` states has the FULL standard InChIKey of ``smiles`` and its fixed-hydrogen
    InChI. False for a name this module cannot read, and for any error: the check fails closed."""
    if not name or not smiles or not fullerene_name_shape(name):
        return False
    try:
        built = _read(name)
        reference = Chem.MolFromSmiles(smiles)
        if built is None or reference is None:
            return False
        got, wanted = _keys(built), _keys(reference)
        return got[0] is not None and got[0] == wanted[0] and got[1] is not None and got[1] == wanted[1]
    except _Unreadable:
        return False
    except Exception:  # noqa: BLE001 - the certificate never raises into the naming path
        return False


def fullerene_gate_ships(name: Optional[str], smiles: Optional[str]) -> bool:
    """The validity gate's question for a fullerene name OPSIN cannot read: True when the name has the core token and
    the structure it states is the input's (``fullerene_read_back``). A best-effort request only: the producer names a
    fullerene in no other, so a name that holds the token anywhere else is not shipped."""
    if not fullerene_name_shape(name):
        return False
    from ..metrics.provenance import best_effort_request_ctx, record_non_pin_fragment, record_source
    if best_effort_request_ctx.get() is not True or not fullerene_read_back(name, smiles):
        return False
    record_non_pin_fragment(C60_CORE_TOKEN)      # never a PIN: the label of the name is the best-effort one
    # no OPSIN reads the name, whichever producer built it (a decomposition route claims the OPSIN reading of its
    # parts): the record of the name is the unverified one of every carve-out
    record_source("pin_path", opsin="unverified")
    return True
