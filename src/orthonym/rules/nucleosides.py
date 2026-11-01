"""Nucleoside / nucleotide decoration engine (Blue Book /.

 a phase. Bare nucleosides (adenosine, guanosine, inosine, xanthosine,
cytidine, thymidine, uridine + the 2'-deoxy series) are named by exact-SMILES
retained-name lookup (``data/retained_names.py``). This module handles the
**decorated** species that those exact keys cannot reach:

* nucleoside 5'-phosphates / di- / tri-phosphates — ``adenosine 5'-(tetrahydrogen
  triphosphate)`` (ATP), ``adenosine 5'-(trihydrogen diphosphate)`` (ADP),
  ``2'-deoxyadenosine 5'-(dihydrogen phosphate)`` (dAMP); /.2)
* O-acyl esters on the sugar — ``adenosine 2',3',5'-triacetate``.

The engine is **strip-and-recognise**: it perceives the nucleoside *core* (a
furanose N-glycosidically bonded to a nucleobase), strips the recognised sugar
decorations back to the free hydroxyls, canonicalises the bare nucleoside and
looks it up in the retained-name catalog. Anything it cannot account for —
a modified base, a phosphate at a non-5' ring position it cannot number, a
cyclic phosphate, a mixed decoration set it has no grammar for — makes it
**return ``None`` (fail-closed)**, so the molecule falls through to the general
pipeline (and, ultimately, the self-consistency gate).

Validation is OPSIN round-trip: every name this module emits parses back to the
input skeleton (verified in ``tests/unit/rules/test_phase14_nucleosides.py``).
"""

from __future__ import annotations

import contextvars
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import RWMol

from ..data.multipliers import multiplier_table

# Phosphate H-count -> word /.2). The "<n>hydrogen" multiplier counts
# the free acidic -OH groups on the linear terminal polyphosphate ester:
# mono: -O-P(=O)(OH)2 -> 2 OH -> dihydrogen phosphate
# di: -O-P(=O)(OH)-O-P(=O)(OH)2 -> 3 OH -> trihydrogen diphosphate
# tri: -O-P(=O)(OH)-O-P(=O)(OH)-O-P(=O)(OH)2 -> 4 OH -> tetrahydrogen triphosphate
_HYDROGEN_MULT = multiplier_table(7, one="")
_PHOSPHATE_STEM = {1: "phosphate", 2: "diphosphate", 3: "triphosphate", 4: "tetraphosphate"}

# O-acyl ester: acyl carbon count -> functional-class ester word. Straight-chain only.
_ACYL_NAME = {1: "formate", 2: "acetate", 3: "propanoate", 4: "butanoate"}
_PRIME_MULT = multiplier_table(4)

# ---------------------------------------------------------------------------
# Base-ring numbering + tautomer-robust base classification ring
# substitution). The base TYPE is read off the EXOCYCLIC heteroatom markers
# (N=amino/imino, O=oxo/hydroxy) at each ring locant — this is robust to the
# amino<->imino / keto<->enol tautomer (and the m7G charge) that ring-N
# alkylation forces. Carbon groups on a ring atom (or on an exocyclic amino N)
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
# (base, sugar) -> retained nucleoside stem. The 2'-deoxy series uses the same stem: '2'-deoxy'
# is a prefix cited in alphanumerical order with the base prefixes, not part of the stem
# "Nucleosides having retained names can be fully substituted...", the Blue Book;
# example `2'-deoxy-1-methylguanosine`,:54967).
_NUCLEOSIDE_STEM = {
    ("adenine", "ribo"): "adenosine", ("adenine", "deoxy"): "adenosine",
    ("guanine", "ribo"): "guanosine", ("guanine", "deoxy"): "guanosine",
    ("hypoxanthine", "ribo"): "inosine", ("hypoxanthine", "deoxy"): "inosine",
    ("xanthine", "ribo"): "xanthosine",
    ("uracil", "ribo"): "uridine", ("uracil", "deoxy"): "uridine",
    ("cytosine", "ribo"): "cytidine", ("cytosine", "deoxy"): "cytidine",
}

# The configuration every retained nucleoside name states (beta-D-ribofuranosyl,: the CIP
# labels of the sugar carbons, which the base substituents cannot change (C1' ranks O4' > N > C2'
# whatever the base carries; C2', C3' and C4' are decided inside the sugar).
_SUGAR_CIP = {"ribo": {"1'": "R", "2'": "R", "3'": "S", "4'": "R"},
              "deoxy": {"1'": "R", "3'": "S", "4'": "R"}}


def _base_numbering(mol: Chem.Mol, base_n: int) -> Optional[Dict[int, int]]:
    """{atom idx -> IUPAC base locant}, anchored so the glycosidic N is 9 (purine)
    or 1 (pyrimidine). None if the base is neither purine nor pyrimidine."""
    # uniquify=False: the pyrimidine ring is symmetric between N1 and N3 (both flank
    # C2), so the default uniquify drops one of the two anchorings non-deterministically
    # by atom order. Enumerate all and pick the one anchoring the glycosidic N at its
    # canonical locant (9 purine / 1 pyrimidine) — deterministic.
    for patt, glyc in ((_PURINE_SMARTS, 9), (_PYRIMIDINE_SMARTS, 1)):
        for match in mol.GetSubstructMatches(patt, uniquify=False):
            loc = {m: patt.GetAtomWithIdx(q).GetAtomMapNum() for q, m in enumerate(match)}
            if loc.get(base_n) == glyc:
                return loc
    return None


def _alkyl_word(mol: Chem.Mol, frag: set, attach: int) -> Optional[str]:
    """The straight-chain alkyl prefix of ``frag`` (bonded through ``attach``), or None.

    ``_ALKYL_NAME`` holds unbranched saturated names only, so the fragment must be an
    unbranched, acyclic chain of saturated, uncharged carbons that starts at ``attach``:
    a branched (isopropyl), unsaturated (ethynyl) or cyclic group is not what the word says."""
    word = _ALKYL_NAME.get(len(frag))
    if word is None:
        return None
    for x in frag:
        atom = mol.GetAtomWithIdx(x)
        if atom.GetSymbol() != "C" or atom.IsInRing() or atom.GetFormalCharge() \
                or atom.GetNumRadicalElectrons():
            return None
        inner = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() in frag]
        if len(inner) > 2 or (x == attach and len(frag) > 1 and len(inner) != 1):
            return None
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
            return None
    return word


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
    """Return (stem, sugar, [(locant_str, alkyl_word)], strip_atoms) or None; ``sugar`` is
    'ribo' or 'deoxy' and the stem carries no '2'-deoxy' (``_NUCLEOSIDE_STEM``).

    Reads the base TYPE from exocyclic N/O markers (tautomer/charge robust) and the
    sugar type (ribo vs 2'-deoxy) from C2'. Carbon groups on a ring atom or on an
    exocyclic amino N become substituents (locant = ``L`` or ``N<L>``). Fail-closed
    on any non-alkyl base group it cannot name."""
    num = _base_numbering(mol, core.base_n)
    if num is None:
        return None
    base_ring = set(num)
    # the SMARTS bonds are '~', so a hydrogenated base matches too; a base name states the mancude
    # ring, so every ring carbon must be aromatic or carry a double bond (ring or exocyclic C=O/C=N)
    for aidx in base_ring:
        atom = mol.GetAtomWithIdx(aidx)
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and not any(
                b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
            return None
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
                if aidx == core.base_n:
                    return None  # a group on the glycosidic N (a 9-ium) is not a base prefix
                frag = _frag_from(mol, b, {aidx})
                if frag & base_ring or frag & sugar_side:
                    return None
                word = _alkyl_word(mol, frag, b)
                if word is None:
                    return None
                subs.append((str(loc), word))
                strip |= frag
            else:
                return None  # exocyclic halogen/other on the base -> out of scope (fail-closed)
    # substituents on an exocyclic amino N (e.g. N6-methyl on the 6-amino N); any other group on
    # an exocyclic N or O (N-hydroxy, a nitro or nitroso N, a second ring) is not expressed: refuse
    for nidx, nloc in exo_het.items():
        het = mol.GetAtomWithIdx(nidx)
        if het.GetFormalCharge() or het.GetNumRadicalElectrons():
            return None
        for nb in het.GetNeighbors():
            b = nb.GetIdx()
            if b in base_ring:
                continue
            if b in sugar_side or nb.GetSymbol() != "C":
                return None
            frag = _frag_from(mol, b, {nidx})
            if frag & base_ring or frag & sugar_side:
                return None
            word = _alkyl_word(mol, frag, b)
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
    sugar = "ribo" if c2_has_o else "deoxy"
    stem = _NUCLEOSIDE_STEM.get((base, sugar))
    if stem is None:
        return None
    return stem, sugar, subs, strip


def _assemble_base_substituents(stem: str, sugar: str,
                                subs: List[Tuple[str, str]]) -> Optional[str]:
    """Assemble the base prefixes and, in the 2'-deoxy series, the ``2'-deoxy`` prefix in
    alphanumerical order in front of the retained stem, the Blue Book:
    `2'-deoxy-1-methylguanosine`,:54967; order. The base prefixes are one word,
    single or multiplied; mixed words -> None."""
    if not subs:
        return None
    words = {w for _, w in subs}
    if len(words) != 1:
        return None  # mixed substituent words -> alphanumeric ordering not built yet
    word = next(iter(words))
    locs = sorted(subs, key=lambda t: (len(t[0]), t[0]))
    loc_str = ",".join(l for l, _ in locs)
    mult = "" if len(subs) == 1 else _PRIME_MULT.get(len(subs))
    if mult is None:
        return None
    prefixes = [(word, f"{loc_str}-{mult}{word}")]       # alphabetised without the multiplier
    if sugar == "deoxy":
        prefixes.append(("deoxy", "2'-deoxy"))
    return "-".join(text for _, text in sorted(prefixes)) + stem


def _sugar_configuration_ok(mol: Chem.Mol, core: _Core, sugar: str) -> bool:
    """Whether every sugar stereocentre is defined with the configuration a retained
    nucleoside name states (``_SUGAR_CIP``). An undefined or other centre is not what the
    name says, so the name is not built."""
    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)
    return _sugar_labels_ok(mol, core, sugar)


def _sugar_labels_ok(mol: Chem.Mol, core: _Core, sugar: str) -> bool:
    """:func:`_sugar_configuration_ok` for a molecule whose CIP labels are assigned already (a chain of units
    assigns them once, ``rules/oligonucleotides.py``)."""
    for prime, label in _SUGAR_CIP[sugar].items():
        atom = mol.GetAtomWithIdx(core.idx_for(prime))
        if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != label:
            return False
    return True


def _sugar_is_plain(mol: Chem.Mol, core: _Core) -> bool:
    """Whether the furanose carries nothing but what the names of this module express: C1'
    the base, C2' at most one O, C3' and C5' one O each, C4' only C5', every bond to these
    groups single and no charge on a sugar atom. A 3'-azido, 3'-deoxy, 5'-deoxy, 2'-fluoro,
    4'-substituted or keto sugar is a different compound and is refused here."""
    allowed = {"1'": (0, 0), "2'": (0, 1), "3'": (1, 1), "4'": (0, 0), "5'": (1, 1)}
    c5 = core.idx_for("5'")
    for prime, (lo, hi) in allowed.items():
        idx = core.idx_for(prime)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return False
        n_o = 0
        for bond in atom.GetBonds():
            nb = bond.GetOtherAtom(atom)
            j = nb.GetIdx()
            if j in core.ring_atoms or (prime == "1'" and j == core.base_n) \
                    or (prime == "4'" and j == c5) or (prime == "5'" and j == core.idx_for("4'")):
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return False
                continue
            if nb.GetSymbol() != "O" or bond.GetBondType() != Chem.BondType.SINGLE:
                return False
            n_o += 1
        if not lo <= n_o <= hi:
            return False
    return True


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
    exocyclic carbon (C5'). Returns ``None`` (fail-closed) on any ambiguity.
    """
    return next(_iter_cores(mol), None)


def _iter_cores(mol: Chem.Mol):
    """Every furanose core of ``mol`` (:func:`_find_core` takes the first; an oligonucleotide has one per unit)."""
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
        yield core


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
    """Name a base-substituted nucleoside e.g. 1-methyladenosine,
    N6-methyladenosine, 5-methyluridine. None if the base is unsubstituted (->
    retained-name path) or carries a group the classifier cannot name."""
    res = _classify_base(mol, core)
    if res is None:
        return None
    stem, sugar, subs, _strip_atoms = res
    if not subs:
        return None  # bare nucleoside -> retained-name path
    # the stem is not looked up by structure here, so the sugar configuration it states is checked
    if not _sugar_configuration_ok(mol, core, sugar):
        return None
    return _assemble_base_substituents(stem, sugar, subs)


#: The / retained names of the retained-name table (``data/retained_names.py``): the names the
#: systematic offer below stands before.
RETAINED_NUCLEOSIDE_NAMES = frozenset({
    "adenosine", "guanosine", "inosine", "cytidine", "uridine", "thymidine",
    "2'-deoxyadenosine", "2'-deoxyguanosine", "2'-deoxycytidine", "2'-deoxyuridine",
    "2'-adenylic acid", "3'-adenylic acid", "5'-adenylic acid", "5'-inosinic acid",
})

#: True inside the naming of:func:`systematic_nucleoside_name`: no nucleoside or nucleotide name is offered
_NUCLEOSIDE_NAMES_OFF: contextvars.ContextVar = contextvars.ContextVar(
    "orthonym_nucleoside_names_off", default=False)


def nucleoside_names_off() -> bool:
    """Whether the running naming leaves the nucleoside and nucleotide names out (:func:`systematic_nucleoside_name`)."""
    return _NUCLEOSIDE_NAMES_OFF.get()


def systematic_nucleoside_name(mol) -> Optional[str]:
    """The systematic substitutive name of the molecule being named, offered before its nucleoside or nucleotide name.

     INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in
    this Chapter", so 'adenosine' and '5'-adenylic acid':54943,:55003) are no PINs; the same
    paragraph: "When the full structure is known, a 'systematic name' may be generated in accordance with Rules
    described in Chapters through ". As for the amino acids (``amino_acids.systematic_amino_acid_name``) the
    molecule is named once more with the nucleoside names left out and taken only as a PIN with its stereodescriptors
    read back (``rules/systematic_offer.py``). Else None, and the caller keeps its nucleoside name, labelled below the
    PIN. Only for the caller's whole molecule."""
    from .systematic_offer import systematic_pin_offer
    return systematic_pin_offer(mol, _NUCLEOSIDE_NAMES_OFF, "systematic_nucleoside_name")


def name_nucleoside(mol: Chem.Mol) -> Optional[str]:
    """Name a decorated nucleoside / nucleotide, or return ``None`` (fail-closed)."""
    if mol is None or nucleoside_names_off() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    core = _find_core(mol)
    if core is None or not _sugar_is_plain(mol, core):
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
        # no sugar decoration -> maybe a base-substituted nucleoside
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
