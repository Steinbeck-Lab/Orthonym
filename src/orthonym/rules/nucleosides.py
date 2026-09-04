"""Nucleoside / nucleotide decoration engine (Blue Book P-105.2 / P-106).

. Bare nucleosides (adenosine, guanosine, inosine, xanthosine,
cytidine, thymidine, uridine + the 2'-deoxy series) are named by exact-SMILES
retained-name lookup (``data/retained_names.py``).  This module handles the
**decorated** species that those exact keys cannot reach:

* nucleoside 5'-phosphates / di- / tri-phosphates — ``adenosine 5'-(tetrahydrogen
  triphosphate)`` (ATP), ``adenosine 5'-(trihydrogen diphosphate)`` (ADP),
  ``2'-deoxyadenosine 5'-(dihydrogen phosphate)`` (dAMP);  (P-106.1/.2)
* O-acyl esters on the sugar — ``adenosine 2',3',5'-triacetate`` (P-105.2.1).

The engine is **strip-and-recognise**: it perceives the nucleoside *core* (a
furanose N-glycosidically bonded to a nucleobase), strips the recognised sugar
decorations back to the free hydroxyls, canonicalises the bare nucleoside and
looks it up in the retained-name catalog.  Anything it cannot account for —
a modified base, a phosphate at a non-5' ring position it cannot number, a
cyclic phosphate, a mixed decoration set it has no grammar for — makes it
**return ``None`` (fail-closed)**, so the molecule falls through to the general
pipeline (and, ultimately, the self-consistency gate).

Validation is OPSIN round-trip: every name this module emits parses back to the
input skeleton (verified in ``tests/unit/rules/test_phase14_nucleosides.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import RWMol

# Phosphate H-count -> word (P-106.1/.2).  The "<n>hydrogen" multiplier counts
# the free acidic -OH groups on the linear terminal polyphosphate ester:
#   mono:  -O-P(=O)(OH)2                          -> 2 OH -> dihydrogen phosphate
#   di:    -O-P(=O)(OH)-O-P(=O)(OH)2              -> 3 OH -> trihydrogen diphosphate
#   tri:   -O-P(=O)(OH)-O-P(=O)(OH)-O-P(=O)(OH)2  -> 4 OH -> tetrahydrogen triphosphate
_HYDROGEN_MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa", 7: "hepta"}
_PHOSPHATE_STEM = {1: "phosphate", 2: "diphosphate", 3: "triphosphate", 4: "tetraphosphate"}

# O-acyl ester: acyl carbon count -> functional-class ester word.  Straight-chain only.
_ACYL_NAME = {1: "formate", 2: "acetate", 3: "propanoate", 4: "butanoate"}
_PRIME_MULT = {2: "di", 3: "tri", 4: "tetra"}

# ---------------------------------------------------------------------------
# Base-ring numbering + tautomer-robust base classification (P-105.2.1 ring
# substitution).  The base TYPE is read off the EXOCYCLIC heteroatom markers
# (N=amino/imino, O=oxo/hydroxy) at each ring locant — this is robust to the
# amino<->imino / keto<->enol tautomer (and the m7G charge) that ring-N
# alkylation forces.  Carbon groups on a ring atom (or on an exocyclic amino N)
# are SUBSTITUENTS, cited by base locant.
# ---------------------------------------------------------------------------
_PURINE_SMARTS = Chem.MolFromSmarts("[#7:1]1~[#6:2]~[#7:3]~[#6:4]2~[#7:9]~[#6:8]~[#7:7]~[#6:5]2~[#6:6]~1")
_PYRIMIDINE_SMARTS = Chem.MolFromSmarts("[#7:1]1~[#6:2]~[#7:3]~[#6:4]~[#6:5]~[#6:6]~1")
_ALKYL_NAME = {1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl"}
_BASE_SIG = {                                     # {(ring-locant, exo-element)} -> base type
    frozenset([(6, "N")]): "adenine",
    frozenset([(2, "N"), (6, "O")]): "guanine",
    frozenset([(6, "O")]): "hypoxanthine",
    frozenset([(2, "O"), (6, "O")]): "xanthine",
    frozenset([(2, "O"), (4, "O")]): "uracil",
    frozenset([(2, "O"), (4, "N")]): "cytosine",
}
_NUCLEOSIDE_STEM = {                              # (base, sugar) -> bare nucleoside name
    ("adenine", "ribo"): "adenosine", ("adenine", "deoxy"): "2'-deoxyadenosine",
    ("guanine", "ribo"): "guanosine", ("guanine", "deoxy"): "2'-deoxyguanosine",
    ("hypoxanthine", "ribo"): "inosine", ("hypoxanthine", "deoxy"): "2'-deoxyinosine",
    ("xanthine", "ribo"): "xanthosine",
    ("uracil", "ribo"): "uridine", ("uracil", "deoxy"): "2'-deoxyuridine",
    ("cytosine", "ribo"): "cytidine", ("cytosine", "deoxy"): "2'-deoxycytidine",
}


def _base_numbering(mol: Chem.Mol, base_n: int) -> Optional[Dict[int, int]]:
    """{atom idx -> IUPAC base locant}, anchored so the glycosidic N is 9 (purine)
    or 1 (pyrimidine).  None if the base is neither purine nor pyrimidine."""
    # uniquify=False: the pyrimidine ring is symmetric between N1 and N3 (both flank
    # C2), so the default uniquify drops one of the two anchorings non-deterministically
    # by atom order.  Enumerate all and pick the one anchoring the glycosidic N at its
    # canonical locant (9 purine / 1 pyrimidine) — deterministic.
    for patt, glyc in ((_PURINE_SMARTS, 9), (_PYRIMIDINE_SMARTS, 1)):
        for match in mol.GetSubstructMatches(patt, uniquify=False):
            loc = {m: patt.GetAtomWithIdx(q).GetAtomMapNum() for q, m in enumerate(match)}
            if loc.get(base_n) == glyc:
                return loc
    return None


def _frag_from(mol: Chem.Mol, start: int, blocked: set) -> set:
    seen, stack = set(), [start]
    while stack:
        a = stack.pop()
        if a in seen or a in blocked:
            continue
        seen.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in blocked:
                stack.append(nb.GetIdx())
    return seen


def _classify_base(mol: Chem.Mol, core: _Core):
    """Return (bare_nucleoside_name, [(locant_str, alkyl_word)], strip_atoms) or None.

    Reads the base TYPE from exocyclic N/O markers (tautomer/charge robust) and the
    sugar type (ribo vs 2'-deoxy) from C2'.  Carbon groups on a ring atom or on an
    exocyclic amino N become substituents (locant = ``L`` or ``N<L>``).  Fail-closed
    on any non-alkyl base group it cannot name."""
    num = _base_numbering(mol, core.base_n)
    if num is None:
        return None
    base_ring = set(num)
    sugar_side = _frag_from(mol, core.anomeric, {core.base_n})
    markers: Dict[int, set] = {}
    subs: List[Tuple[str, str]] = []
    strip: set = set()
    exo_het: Dict[int, str] = {}   # exocyclic N/O idx -> "N<loc>"/"O<loc>"
    for aidx, loc in num.items():
        for nb in mol.GetAtomWithIdx(aidx).GetNeighbors():
            b = nb.GetIdx()
            if b in base_ring or b in sugar_side:
                continue
            sym = nb.GetSymbol()
            if sym in ("N", "O"):
                markers.setdefault(loc, set()).add(sym)
                exo_het[b] = f"{sym}{loc}"
            elif sym == "C":
                frag = _frag_from(mol, b, {aidx})
                if frag & base_ring or frag & sugar_side or \
                        not all(mol.GetAtomWithIdx(x).GetSymbol() == "C" for x in frag):
                    return None
                word = _ALKYL_NAME.get(len(frag))
                if word is None:
                    return None
                subs.append((str(loc), word))
                strip |= frag
            else:
                return None  # exocyclic halogen/other on the base -> out of scope (fail-closed)
    # substituents on an exocyclic amino N (e.g. N6-methyl on the 6-amino N)
    for nidx, nloc in exo_het.items():
        for nb in mol.GetAtomWithIdx(nidx).GetNeighbors():
            b = nb.GetIdx()
            if b in base_ring or b in sugar_side or nb.GetSymbol() != "C":
                continue
            frag = _frag_from(mol, b, {nidx})
            if frag & base_ring or frag & sugar_side or \
                    not all(mol.GetAtomWithIdx(x).GetSymbol() == "C" for x in frag):
                return None
            word = _ALKYL_NAME.get(len(frag))
            if word is None:
                return None
            subs.append((nloc, word))
            strip |= frag
    base = _BASE_SIG.get(frozenset((loc, e) for loc, es in markers.items() for e in es))
    if base is None:
        return None
    c2 = core.idx_for("2'")
    c2_has_o = any(nb.GetSymbol() == "O" and nb.GetIdx() not in core.ring_atoms
                   for nb in mol.GetAtomWithIdx(c2).GetNeighbors())
    bare = _NUCLEOSIDE_STEM.get((base, "ribo" if c2_has_o else "deoxy"))
    if bare is None:
        return None
    return bare, subs, strip


def _assemble_base_substituents(bare: str, subs: List[Tuple[str, str]]) -> Optional[str]:
    """Assemble ``<locant>-<alkyl><nucleoside>`` (P-105.2.1).  Single substituent, or
    several identical-word substituents with a multiplier; mixed words -> None."""
    if not subs:
        return None
    words = {w for _, w in subs}
    if len(words) == 1:
        word = next(iter(words))
        locs = sorted(subs, key=lambda t: (len(t[0]), t[0]))
        loc_str = ",".join(l for l, _ in locs)
        if len(subs) == 1:
            return f"{loc_str}-{word}{bare}"
        mult = _PRIME_MULT.get(len(subs))
        return f"{loc_str}-{mult}{word}{bare}" if mult else None
    return None  # mixed substituent words -> alphanumeric ordering not built yet


def _canon(smi: str) -> Optional[str]:
    try:
        return Chem.CanonSmiles(smi)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core perception
# ---------------------------------------------------------------------------
class _Core:
    """A perceived nucleoside core: the furanose ring + its 1'..5' numbering."""

    __slots__ = ("ring_o", "anomeric", "base_n", "ring_atoms", "prime", "_idx_for")

    def __init__(self):
        self.ring_o: int = -1
        self.anomeric: int = -1            # C1'
        self.base_n: int = -1              # glycosidic N
        self.ring_atoms: set = set()
        self.prime: Dict[int, str] = {}    # sugar-carbon atom idx -> "1'".."5'"
        self._idx_for: Dict[str, int] = {}

    def idx_for(self, prime: str) -> int:
        return self._idx_for[prime]


def _find_core(mol: Chem.Mol) -> Optional[_Core]:
    """Perceive the (2-deoxy)ribofuranose core + assign 1'..5' numbering.

    A furanose candidate is a 5-membered non-aromatic ring with exactly one ring
    O and four ring C, where one ring C (the anomeric C1') is bonded to a ring
    nitrogen of an N-heterocycle (the base) and another ring C (C4') bears an
    exocyclic carbon (C5').  Returns ``None`` (fail-closed) on any ambiguity.
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) != 5:
            continue
        ring_set = set(ring)
        o_atoms = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
        c_atoms = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "C"]
        if len(o_atoms) != 1 or len(c_atoms) != 4:
            continue
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        ring_o = o_atoms[0]
        o_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(ring_o).GetNeighbors()
                       if n.GetIdx() in ring_set]
        if len(o_neighbors) != 2:
            continue
        anomeric = base_n = c4 = -1
        for c in o_neighbors:
            glyc_n = None
            for nb in mol.GetAtomWithIdx(c).GetNeighbors():
                if nb.GetSymbol() == "N" and nb.GetIdx() not in ring_set and nb.IsInRing():
                    glyc_n = nb.GetIdx()
            if glyc_n is not None:
                anomeric, base_n = c, glyc_n
            else:
                c4 = c
        if anomeric < 0 or c4 < 0:
            continue
        c5 = -1
        for nb in mol.GetAtomWithIdx(c4).GetNeighbors():
            if nb.GetSymbol() == "C" and nb.GetIdx() not in ring_set:
                c5 = -2 if c5 >= 0 else nb.GetIdx()
        if c5 < 0:
            continue
        ring_carbon_set = set(c_atoms)
        c2 = -1
        for nb in mol.GetAtomWithIdx(anomeric).GetNeighbors():
            if nb.GetIdx() in ring_carbon_set:
                c2 = nb.GetIdx()
        c3_candidates = ring_carbon_set - {anomeric, c4, c2}
        if c2 < 0 or len(c3_candidates) != 1:
            continue
        c3 = next(iter(c3_candidates))
        if mol.GetBondBetweenAtoms(c3, c4) is None:
            continue

        core = _Core()
        core.ring_o = ring_o
        core.anomeric = anomeric
        core.base_n = base_n
        core.ring_atoms = ring_set
        core.prime = {anomeric: "1'", c2: "2'", c3: "3'", c4: "4'", c5: "5'"}
        core._idx_for = {v: k for k, v in core.prime.items()}
        return core
    return None


# ---------------------------------------------------------------------------
# Decoration perception
# ---------------------------------------------------------------------------
def _phosphate_chain(mol: Chem.Mol, ester_o: int) -> Optional[Tuple[set, int, int]]:
    """Walk a linear terminal polyphosphate off ``ester_o`` (bridging sugar-O).
    Returns (atoms_to_delete, n_OH, n_P) or None (not a clean linear chain)."""
    p_first = [n.GetIdx() for n in mol.GetAtomWithIdx(ester_o).GetNeighbors()
               if n.GetSymbol() == "P"]
    if len(p_first) != 1:
        return None
    chain_p: List[int] = []
    cur = p_first[0]
    prev_o = ester_o
    n_oh = 0
    while True:
        patom = mol.GetAtomWithIdx(cur)
        if patom.GetSymbol() != "P":
            return None
        chain_p.append(cur)
        dbl_o = 0
        next_p = None
        next_bridge_o = None
        for nb in patom.GetNeighbors():
            if nb.GetSymbol() == "P":
                continue
            if nb.GetSymbol() != "O":
                return None  # P bonded to C/S/... -> out of scope here
            bond = mol.GetBondBetweenAtoms(cur, nb.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                dbl_o += 1
                continue
            if nb.GetIdx() == prev_o:
                continue
            p_on_o = [x for x in nb.GetNeighbors() if x.GetSymbol() == "P" and x.GetIdx() != cur]
            other_heavy = [x for x in nb.GetNeighbors()
                           if x.GetSymbol() not in ("H",) and x.GetIdx() != cur]
            if p_on_o:
                if next_p is not None:
                    return None  # branched chain
                next_p = p_on_o[0].GetIdx()
                next_bridge_o = nb.GetIdx()
            elif not other_heavy:
                n_oh += 1  # terminal OH / O- (both are phosphate acid OH slots for the word)
            else:
                return None  # O bonded to another non-P heavy atom -> not a clean chain
        if dbl_o != 1:
            return None
        if next_p is None:
            break
        prev_o = next_bridge_o
        cur = next_p
        if len(chain_p) > 7:
            return None

    to_del = set(chain_p)
    for p in chain_p:
        for nb in mol.GetAtomWithIdx(p).GetNeighbors():
            if nb.GetSymbol() == "O" and nb.GetIdx() != ester_o:
                heavy_non_p = [x for x in nb.GetNeighbors()
                               if x.GetSymbol() not in ("P",) and x.GetIdx() not in to_del]
                if not heavy_non_p:
                    to_del.add(nb.GetIdx())
    return to_del, n_oh, len(chain_p)


def _acyl(mol: Chem.Mol, ester_o: int, carbonyl_c: int) -> Optional[Tuple[str, set]]:
    """Recognise -O-C(=O)-R; return (ester_word, atoms_to_delete) or None."""
    c = mol.GetAtomWithIdx(carbonyl_c)
    dbl_o = [n for n in c.GetNeighbors()
             if n.GetSymbol() == "O"
             and mol.GetBondBetweenAtoms(carbonyl_c, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
    if len(dbl_o) != 1:
        return None
    acyl_carbons = [carbonyl_c]
    visited = {carbonyl_c, ester_o, dbl_o[0].GetIdx()}
    cur = carbonyl_c
    while True:
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetSymbol() == "C" and n.GetIdx() not in visited]
        if not nxt:
            break
        if len(nxt) > 1:
            return None  # branched acyl
        cur = nxt[0]
        if any(n.GetSymbol() not in ("C", "H") for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetIdx() not in visited):
            return None  # heteroatom on the acyl chain
        visited.add(cur)
        acyl_carbons.append(cur)
        if len(acyl_carbons) > 6:
            return None
    name = _ACYL_NAME.get(len(acyl_carbons))
    if name is None:
        return None
    return name, set(acyl_carbons) | {dbl_o[0].GetIdx()}


def _classify(mol: Chem.Mol, sugar_c: int, core: _Core):
    """Classify the exocyclic O on a sugar carbon.
    Returns ("oh"/"none", None) | ("phosphate", (to_del,n_oh,n_p)) | ("acyl", (word,to_del)) | None."""
    o_idx = None
    for nb in mol.GetAtomWithIdx(sugar_c).GetNeighbors():
        if nb.GetSymbol() == "O" and nb.GetIdx() not in core.ring_atoms:
            if o_idx is not None:
                return None
            o_idx = nb.GetIdx()
    if o_idx is None:
        return ("none", None)
    heavy = [n for n in mol.GetAtomWithIdx(o_idx).GetNeighbors()
             if n.GetSymbol() != "H" and n.GetIdx() != sugar_c]
    if not heavy:
        return ("oh", None)
    if len(heavy) != 1:
        return None
    other = heavy[0]
    if other.GetSymbol() == "P":
        chain = _phosphate_chain(mol, o_idx)
        return ("phosphate", chain) if chain else None
    if other.GetSymbol() == "C":
        acyl = _acyl(mol, o_idx, other.GetIdx())
        return ("acyl", acyl) if acyl else None
    return None


def _strip(mol: Chem.Mol, del_atoms: set) -> Optional[Chem.Mol]:
    rw = RWMol(mol)
    for idx in sorted(del_atoms, reverse=True):
        rw.RemoveAtom(idx)
    try:
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        return m2
    except Exception:
        return None


def _phosphate_word(n_p: int, n_oh: int) -> Optional[str]:
    stem = _PHOSPHATE_STEM.get(n_p)
    mult = _HYDROGEN_MULT.get(n_oh)
    if stem is None or mult is None:
        return None
    return f"{mult}hydrogen {stem}"


def _lookup_bare(bare: Chem.Mol) -> Optional[str]:
    from orthonym.data import get_retained_name
    smi = _canon(Chem.MolToSmiles(bare))
    return get_retained_name(smi) if smi else None


def _name_base_substituted(mol: Chem.Mol, core: _Core) -> Optional[str]:
    """Name a base-substituted nucleoside (P-105.2.1) e.g. 1-methyladenosine,
    N6-methyladenosine, 5-methyluridine.  None if the base is unsubstituted (->
    retained-name path) or carries a group the classifier cannot name."""
    res = _classify_base(mol, core)
    if res is None:
        return None
    bare, subs, _strip_atoms = res
    if not subs:
        return None  # bare nucleoside -> retained-name path
    return _assemble_base_substituents(bare, subs)


def name_nucleoside(mol: Chem.Mol) -> Optional[str]:
    """Name a decorated nucleoside / nucleotide, or return ``None`` (fail-closed)."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    core = _find_core(mol)
    if core is None:
        return None

    decorations: Dict[str, tuple] = {}   # prime -> ("phosphate", (to_del,n_oh,n_p)) | ("acyl", word)
    del_atoms: set = set()
    for prime in ("2'", "3'", "5'"):
        sugar_c = core.idx_for(prime)
        cls = _classify(mol, sugar_c, core)
        if cls is None:
            return None  # unrecognised exocyclic group -> fail-closed
        kind, payload = cls
        if kind in ("oh", "none"):
            continue
        if kind == "phosphate":
            to_del, n_oh, n_p = payload
            decorations[prime] = ("phosphate", (n_oh, n_p))
            del_atoms |= to_del
        else:  # acyl
            word, to_del = payload
            decorations[prime] = ("acyl", word)
            del_atoms |= to_del

    if not decorations:
        # no sugar decoration -> maybe a base-substituted nucleoside (P-105.2.1)
        return _name_base_substituted(mol, core)
    kinds = {d[0] for d in decorations.values()}
    if len(kinds) > 1:
        return None  # mixed phosphate+acyl -> no combined grammar

    bare = _strip(mol, del_atoms)
    if bare is None:
        return None
    bare_name = _lookup_bare(bare)
    if bare_name is None:
        return None

    if kinds == {"phosphate"}:
        if len(decorations) != 1 or "5'" not in decorations:
            return None  # only a single 5'-phosphate ester supported
        _kind, (n_oh, n_p) = decorations["5'"]
        word = _phosphate_word(n_p, n_oh)
        return f"{bare_name} 5'-({word})" if word else None

    # all-acyl
    words = {d[1] for d in decorations.values()}
    if len(words) != 1:
        return None  # mixed acyls -> no combined grammar
    acyl = next(iter(words))
    primes = sorted(decorations.keys(), key=lambda p: int(p.rstrip("'")))
    locants = ",".join(primes)
    n = len(primes)
    if n == 1:
        return f"{bare_name} {locants}-{acyl}"
    mult = _PRIME_MULT.get(n)
    return f"{bare_name} {locants}-{mult}{acyl}" if mult else None
