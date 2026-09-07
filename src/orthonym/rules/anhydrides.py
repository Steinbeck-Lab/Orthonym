"""
Anhydride naming using IUPAC functional class nomenclature.

Anhydrides use "functional class" naming:
    Symmetric:  "{acid name} anhydride"
    Mixed:      "{acid1} {acid2} anhydride" (alphabetical order)
    Cyclic:     "{diacid name} anhydride"

Examples:
    CC(=O)OC(=O)C       -> ethanoic anhydride (symmetric)
    CC(=O)OC(=O)CC      -> ethanoic propanoic anhydride (mixed, alphabetical)
    CCC(=O)OC(=O)CCC    -> butanoic anhydride (symmetric)
    O=C1CCC(=O)O1       -> butanedioic anhydride (cyclic, from succinic acid)
    O=C1CCCC(=O)O1      -> pentanedioic anhydride (cyclic, from glutaric acid)

The acid name derives from the acyl fragment:
    ethanoyl + ethanoyl -> ethanoic acid -> "ethanoic anhydride"

Reference: IUPAC 2013 Blue Book, P-65.3.1 (Acid anhydrides)
"""

import re
from collections import deque
from typing import Optional

from rdkit import Chem

from ..data.chain_names import get_chain_prefix

# Anhydride core SMARTS: C(=O)-O-C(=O)
ANHYDRIDE_SMARTS = "[CX3](=O)[OX2][CX3](=O)"

# W3-P06 bridge-variant cores (re-detected here from the mol; perception folds
# these into the 'anhydride' bucket so this handler is reached).
_SULFONIC_ANHYDRIDE_SMARTS = "[SX4](=O)(=O)[OX2][SX4](=O)(=O)"       # P-65.7.1
_CHALCOGEN_ANHYDRIDE_SMARTS = "[CX3](=O)[SX2,SeX2,TeX2][CX3](=O)"    # P-65.7.3
_PEROXY_ANHYDRIDE_SMARTS = "[CX3](=O)[OX2][OX2][CX3](=O)"           # P-65.7.4

# Two-word functional-class halide words, alphabetical order bromide<chloride<
# fluoride<iodide (P-65.5.3.2 / P-65.5.1).
_HALIDE_WORD = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}

# W8-P4 (P-65.7.2 mixed organic/inorganic anhydride; P-65.2.2 cyanic acid):
# acyl-O-C#N / acyl-S-C#N core -> the retained inorganic-acid word paired with
# the acyl side's acid name (BB 30979 'NC-OH cyanic acid (PIN)'; BB 30984
# 'NC-SH thiocyanic acid (PIN)'). Scoped ONLY to cyanic/thiocyanic this cycle
# (P-65.7.2 also lists carbonic/boric/phosphinous mixed anhydrides -- those
# need their own acid-parent naming and are out of scope here).
_MIXED_INORGANIC_ANHYDRIDE_SMARTS = {
    "cyanic": "[CX3](=O)[OX2][CX2]#[NX1]",
    "thiocyanic": "[CX3](=O)[SX2][CX2]#[NX1]",
}


def name_anhydride(features) -> Optional[str]:
    """
    Name an anhydride compound using functional class nomenclature.

    Args:
        features: MolecularFeatures with principal_group == 'anhydride'.

    Returns:
        IUPAC name string, or None if not an anhydride.
    """
    mol = features.mol
    pg = features.principal_group

    if pg != "anhydride":
        return None

    # W3-P06: bridge-variant anhydrides (perceived + folded into 'anhydride').
    # Each re-detects its own bridge on the mol; the base carbon/O core below
    # never matches them, so ordering these first is safe (they return only on a
    # genuine bridge match).
    _sn = _name_sulfonic_anhydride(mol)          # P-65.7.1  R-SO2-O-SO2-R'
    if _sn:
        return _sn
    _cn = _name_chalcogen_anhydride(mol)         # P-65.7.3  R-CO-S/Se/Te-CO-R'
    if _cn:
        return _cn
    _tn = _name_thioacyl_anhydride(mol)          # P-65.7.6.4.2/.3  R-C(=X)-Y-C(=X)-R'
    if _tn:
        return _tn
    _pn = _name_peroxy_anhydride(mol)            # P-65.7.4  R-CO-OO-CO-R'
    if _pn:
        return _pn
    _mi = _name_mixed_inorganic_anhydride(mol)   # P-65.7.2  R-CO-O/S-C#N (cyanic/thiocyanic)
    if _mi:
        return _mi

    # Detect anhydride core: C(=O)-O-C(=O)
    pattern = Chem.MolFromSmarts(ANHYDRIDE_SMARTS)
    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        return None

    # W3-P06 Task 8 (P-65.7.6.1): di- and polyanhydrides — TWO or more -CO-O-CO-
    # linkages. Must be handled before the single-match path (which takes
    # matches[0] and whose acyl-BFS would cross the 2nd linkage -> wrong name).
    if len(matches) >= 2:
        _da = _name_dianhydride(mol, matches)
        if _da:
            return _da

    # Take first match
    match = matches[0]
    # SMARTS [CX3](=O)[OX2][CX3](=O) matches: (C1, O1_carbonyl, O_bridge, C2, O2_carbonyl)
    # But RDKit returns atom indices in SMARTS order.
    # Let's identify the atoms from the SMARTS pattern:
    # Position 0: first carbonyl C
    # Position 1: first carbonyl O (=O)
    # Position 2: bridge O
    # Position 3: second carbonyl C
    # Position 4: second carbonyl O (=O)

    # Actually, for the SMARTS [CX3](=O)[OX2][CX3](=O), the match tuple is:
    # (C1, O1, O_bridge, C2, O2) where O1 and O2 are carbonyl oxygens
    c1_idx = match[0]
    c2_idx = match[3] if len(match) >= 5 else match[2]

    # Re-parse more carefully using atom environment
    anhydride_info = _parse_anhydride_core(mol, matches[0])
    if anhydride_info is None:
        return None

    c1_idx = anhydride_info['c1']
    c2_idx = anhydride_info['c2']
    bridge_o = anhydride_info['bridge_o']

    # Check if cyclic anhydride (both carbonyl carbons in same ring)
    ring_info = mol.GetRingInfo()
    is_cyclic = False
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if c1_idx in ring_set and c2_idx in ring_set and bridge_o in ring_set:
            is_cyclic = True
            # Count total carbons in ring (excluding bridge oxygen)
            ring_carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'C']
            total_chain_length = len(ring_carbons)
            break

    if is_cyclic:
        # PIN (P-65.7.7.1 method 1): the heterocyclic-pseudoketone dione
        # (oxolane-2,5-dione, furan-2,5-dione, 2-benzofuran-1,3-dione, ...).
        # Produced from WITHIN this priority-1200 handler so the lactone@1300
        # handler (which also matches a cyclic anhydride) cannot grab + mis-name it.
        dione = _name_cyclic_anhydride_dione(mol, c1_idx, c2_idx, bridge_o, ring_set)
        if dione:
            return dione
        # Fallback: general-nomenclature '{diacid} anhydride' (non-PIN, NOT the
        # wrong molecule for a simple all-carbon ring) for anything the dione namer
        # declines.
        return _name_cyclic_anhydride(total_chain_length)

    # W3-P06 Task 3 (P-65.5.3.2): the diacyl halide of dicarbonic acid,
    # X-CO-O-CO-X. BOTH carbonyl carbons are 'carbono' units (bridge O + one =O +
    # exactly one halide, NO carbon neighbour) -> 'dicarbonic <halide word(s)>'
    # (dicarbonic dichloride; mixed -> alphabetical 'dicarbonic bromide chloride',
    # BB 31492/31496). Checked BEFORE _name_acyl_acid (which would mis-read each
    # Cl-CO- side as '1-chloromethanoic acid'). Fails closed (falls through) if
    # either side has a carbon neighbour (a real acid anhydride) or lacks a halide.
    _dc = _name_dicarbonic_dihalide(mol, c1_idx, c2_idx, bridge_o)
    if _dc:
        return _dc

    # Acyclic anhydride: name each acyl fragment as its corresponding acid.
    # Uses _name_acyl_acid() which calls _integrate_universal_prefixes()
    # per D-01 to discover branch substituents on acyl chains.
    acid1_name, sub1 = _name_acyl_acid(mol, c1_idx, bridge_o, anhydride_info['o1'])
    acid2_name, sub2 = _name_acyl_acid(mol, c2_idx, bridge_o, anhydride_info['o2'])

    if acid1_name is None or acid2_name is None:
        return None

    if acid1_name == acid2_name:
        # Symmetric anhydride. P-65.7.8.1: a SYMMETRICALLY SUBSTITUTED
        # monocarboxylic-acid anhydride is named 'bis({acid}) anhydride'
        # (bis(6-aminohexanoic) anhydride, bis(chloroacetic) anhydride). An
        # unsubstituted symmetric acid keeps '{acid} anhydride' (acetic /
        # propanoic / benzoic).
        if sub1:
            return f"bis({acid1_name}) anhydride"
        return f"{acid1_name} anhydride"
    else:
        # Mixed/asymmetric anhydride: alphabetical order (P-65.7.2 / P-65.7.8.2).
        acids = sorted([acid1_name, acid2_name])
        return f"{acids[0]} {acids[1]} anhydride"


def _bfs_side_atoms(mol, start: int, blocked: set) -> set:
    """All heavy-atom indices reachable from ``start`` without crossing any atom
    in ``blocked`` (the bridge atom(s) / the far side)."""
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            nidx = nb.GetIdx()
            if nidx in seen or nidx in blocked:
                continue
            seen.add(nidx)
            queue.append(nidx)
    return seen


def _name_sulfonic_anhydride(mol) -> Optional[str]:
    """P-65.7.1: R-SO2-O-SO2-R' -> '<sulfonic acid stem(s)> anhydride'.

    Each sulfonyl side is capped with -OH into its free sulfonic acid, named via
    the recursive fragment namer, and the ' acid' word stripped. Symmetric ->
    '{acid} anhydride' (benzenesulfonic anhydride); mixed -> alphabetical two
    words; symmetric-substituted (a heteroatom substituent on R) -> bis(...)
    (P-65.7.8.1). Returns None (fall through) when no S-O-S core is present."""
    pat = Chem.MolFromSmarts(_SULFONIC_ANHYDRIDE_SMARTS)
    if pat is None:
        return None
    matches = mol.GetSubstructMatches(pat, uniquify=True)
    if not matches:
        return None
    m = matches[0]                       # (S1, =O, =O, O_bridge, S2, =O, =O)
    s1, bridge_o, s2 = m[0], m[3], m[4]
    # W8-P4 (atom-drop safety floor): a CYCLIC S-O-S core (both sulfonyl S in
    # the same ring, reachable from each other without crossing the bridge)
    # is NOT this acyclic bis-acid shape -- the BFS-per-side logic below would
    # silently walk into the SAME ring from both ends and mis-name the whole
    # fused system. No cyclic sulfonic-anhydride evidence this cycle -> decline
    # cleanly (fail-closed) rather than risk a wrong name.
    if s2 in _bfs_side_atoms(mol, s1, {bridge_o}):
        return None

    from ..assembly.fragment_naming import name_fragment_recursively
    from .esters import _extract_fragment_smiles

    def _side_acid(s_idx):
        keep = _bfs_side_atoms(mol, s_idx, {bridge_o})
        # 'substituted' = any heteroatom (non C/H/S/O-of-sulfonyl) hanging off R.
        substituted = any(
            mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6, 8, 16)
            for a in keep
        )
        smi = _extract_fragment_smiles(mol, set(keep), s_idx, cap_element=8)
        if not smi:
            return None, False
        nm = name_fragment_recursively(smi)
        if not nm or nm == "unknown" or not nm.endswith(" acid"):
            return None, False
        return nm[:-5].strip(), substituted

    a1, sub1 = _side_acid(s1)
    a2, sub2 = _side_acid(s2)
    if a1 is None or a2 is None:
        return None
    if a1 == a2:
        if sub1:
            return f"bis({a1}) anhydride"
        return f"{a1} anhydride"
    acids = sorted([a1, a2])
    return f"{acids[0]} {acids[1]} anhydride"


_CHALCOGEN_CLASS_TERM = {"S": "thioanhydride", "Se": "selenoanhydride",
                         "Te": "telluroanhydride"}

# P2D (P-65.7.6.4.2): the anhydride CLASS WORD is fixed by the BRIDGE
# chalcogen (O -> anhydride, S -> thioanhydride, Se -> selenoanhydride,
# Te -> telluroanhydride). O just reuses _CHALCOGEN_CLASS_TERM's siblings.
_BRIDGE_CLASS_WORD = {"O": "anhydride", "S": "thioanhydride",
                      "Se": "selenoanhydride", "Te": "telluroanhydride"}

# P2D (P-65.7.6.4.3): the ACYL chalcogen fixes each acid component's affix
# (=O -> plain '...oic'/retained; =S -> '...thioic'; =Se -> '...selenoic').
_ACYL_AFFIX = {"S": "thioic", "Se": "selenoic"}

# P2D: generalized acyl-anhydride motif R-C(=X1)-Y-C(=X2)-R', X in {O,S,Se},
# bridge Y in {O,S,Se,Te}. Broad on BOTH the acyl chalcogen and the bridge; the
# handler declines the all-=O cases (owned by the base / chalcogen paths).
_THIOACYL_ANHYDRIDE_SMARTS = (
    "[CX3](=[OX1,SX1,SeX1])[OX2,SX2,SeX2,TeX2][CX3](=[OX1,SX1,SeX1])")


_NUMERIC_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}

# W8-P4 (P-65.7.6.4.1): 'bis'/'tris'/... wrap the chalcogen-anhydride class
# word for a chalcogen di-/poly-anhydride (BB 32446 'bis(thioanhydride)'),
# unlike the plain numeric-prefixed 'dianhydride'/'trianhydride' used for the
# O-bridge case.
_BIS_MULT = {2: "bis", 3: "tris", 4: "tetrakis"}


def _diacid_stem(mol, frag_atoms, carbonyls) -> Optional[str]:
    """Chain diacid stem (e.g. 'butanedioic') for a LINEAR, unsubstituted diacid
    residue whose two carbonyls sit at both chain ends. Fail-closed (None) for
    branched / substituted / non-terminal-carbonyl residues."""
    frag = set(frag_atoms)
    carbons = [a for a in frag if mol.GetAtomWithIdx(a).GetSymbol() == "C"]
    # Linear: no chain carbon has >2 in-fragment carbon neighbours.
    for a in carbons:
        cn = sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() in frag and nb.GetSymbol() == "C")
        if cn > 2:
            return None
    # Both carbonyls terminal (exactly one in-fragment carbon neighbour).
    for c in carbonyls:
        cn = sum(1 for nb in mol.GetAtomWithIdx(c).GetNeighbors()
                 if nb.GetIdx() in frag and nb.GetSymbol() == "C")
        if cn != 1:
            return None
    # Unsubstituted: every non-carbon fragment atom is a carbonyl =O.
    if any(mol.GetAtomWithIdx(a).GetSymbol() != "O" for a in frag
           if mol.GetAtomWithIdx(a).GetSymbol() != "C"):
        return None
    return get_chain_prefix(len(carbons)) + "anedioic"


def _name_residue_acid(mol, frag_atoms, carbonyls, carbonyl_meta):
    """Name one acid residue of a di/polyanhydride: a monoacid (1 carbonyl) via
    _name_acyl_acid, a diacid (2 carbonyls) via _diacid_stem."""
    if len(carbonyls) == 1:
        c = carbonyls[0]
        bo, co = carbonyl_meta[c]
        nm, _sub = _name_acyl_acid(mol, c, bo, co)
        return nm
    if len(carbonyls) == 2:
        return _diacid_stem(mol, frag_atoms, carbonyls)
    return None


def _name_dianhydride(mol, matches) -> Optional[str]:
    """P-65.7.6.1: di-/polyanhydride (>=2 -CO-O-CO- linkages) ->
    '<acid groups in occurrence order> {di|tri|...}anhydride'.

    Acid residues are cited in their order of occurrence in the structure,
    beginning with the end acid group lower in alphabetical order; identical
    residues collapse with a numeric multiplier ('diacetic butanedioic
    dianhydride', BB 32386). Handles the LINEAR di/poly-anhydride topology
    (mono-...-diacid-...-mono); fail-closed (None) for branched/complex
    topologies, whereupon name_anhydride falls through to the single-linkage
    path."""
    infos = []
    bridge_os = set()
    for m in matches:
        info = _parse_anhydride_core(mol, m)
        if info is None or info['o1'] is None or info['o2'] is None:
            return None
        infos.append(info)
        bridge_os.add(info['bridge_o'])

    # carbonyl carbon -> (its bridge O, its =O) for monoacid residue naming.
    carbonyl_meta = {}
    for info in infos:
        carbonyl_meta[info['c1']] = (info['bridge_o'], info['o1'])
        carbonyl_meta[info['c2']] = (info['bridge_o'], info['o2'])

    # Residue fragment for each carbonyl = atoms reachable without crossing ANY
    # bridge O. Carbonyls in the same fragment (a central diacid) share it.
    frag_of = {c: frozenset(_bfs_side_atoms(mol, c, bridge_os))
               for c in carbonyl_meta}
    uniq = {}
    for c, f in frag_of.items():
        uniq.setdefault(f, []).append(c)

    res_name = {}
    for f, cs in uniq.items():
        nm = _name_residue_acid(mol, f, cs, carbonyl_meta)
        if nm is None:
            return None
        res_name[f] = nm

    # Residue adjacency graph (one edge per anhydride linkage).
    adj = {f: [] for f in uniq}
    for info in infos:
        fa, fb = frag_of[info['c1']], frag_of[info['c2']]
        adj[fa].append(fb)
        adj[fb].append(fa)
    terminals = [f for f in uniq if len(adj[f]) == 1]
    if len(terminals) != 2:
        return None  # not a simple linear di/poly-anhydride chain

    # Walk from the terminal whose acid name is alphabetically lower.
    start = min(terminals, key=lambda f: res_name[f])
    order, visited, prev, cur = [], set(), None, start
    while cur is not None and cur not in visited:
        visited.add(cur)
        order.append(res_name[cur])
        nxts = [x for x in adj[cur] if x != prev and x not in visited]
        prev, cur = cur, (nxts[0] if nxts else None)
    if len(visited) != len(uniq):
        return None  # topology is not a simple path

    # Collapse identical residues (first-occurrence order) with a NUMERIC prefix
    # (P-65.7.6.1: "The numerical prefix 'di-' is used").
    seq, counts = [], {}
    for nm in order:
        if nm not in counts:
            seq.append(nm)
            counts[nm] = 0
        counts[nm] += 1
    parts = [(_NUMERIC_MULT.get(counts[nm], "") if counts[nm] > 1 else "") + nm
             for nm in seq]
    class_term = _NUMERIC_MULT.get(len(matches), "") + "anhydride"
    return " ".join(parts) + " " + class_term


def _hydrazido_class_word(mol, nb, ctr_idx: int) -> Optional[str]:
    """P-66.3.5.2 class word for a hydrazido leaving group (ctr-NH-NH2) on a
    carbonic carbon -> 'hydrazide', else None. Kept LOCAL to the di-/polycarbonic
    anhydride builder so the shared ``_halide_pseudohalide_class_word`` (also used
    by the phosphorus FRN namer, where a -NH-NH2 is classified differently) is
    unchanged. BB:34111 'dicarbonic dihydrazide (PIN)'."""
    if nb.GetSymbol() != "N" or nb.GetTotalNumHs() != 1 or nb.GetFormalCharge() != 0:
        return None
    ctr_bond = mol.GetBondBetweenAtoms(ctr_idx, nb.GetIdx())
    if ctr_bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    others = [x for x in nb.GetNeighbors() if x.GetIdx() != ctr_idx]
    if len(others) != 1:
        return None
    o = others[0]
    if (o.GetSymbol() == "N" and o.GetDegree() == 1 and o.GetTotalNumHs() == 2
            and o.GetFormalCharge() == 0
            and mol.GetBondBetweenAtoms(nb.GetIdx(), o.GetIdx()).GetBondType()
            == Chem.BondType.SINGLE):
        return "hydrazide"
    return None


def _name_dicarbonic_dihalide(mol, c1: int, c2: int, bridge_o: int) -> Optional[str]:
    """P-65.5.3.2 / P-66.3.5.2: X-CO-O-CO-Y (X, Y halide, pseudohalide, or
    hydrazido) -> 'dicarbonic <class word(s)>'.

    BOTH carbonyl carbons must be 'carbono' units: bonded to the bridge O (single
    bond) + exactly one =O + exactly one halide/pseudohalide/hydrazido class group,
    with NO carbon neighbour. Class words (halide F/Cl/Br/I, or pseudohalide
    isocyanate / isothiocyanate / cyanide / azide, or hydrazido -> 'hydrazide') are
    cited in ALPHABETICAL order; identical ones collapse with a numeric multiplier
    ('dicarbonic dichloride', 'dicarbonic diisocyanate', 'dicarbonic dihydrazide',
    BB 31492/31500/34111); different ones -> two words ('dicarbonic bromide
    chloride'). Returns None (fail-closed) if either side is a real acyl (has a
    carbon neighbour) or lacks a recognised halide/pseudohalide/hydrazido."""
    from ..assembly.naming_utils import get_multiplier_prefix
    from .inorganic_acids import _halide_pseudohalide_class_word
    words = []
    for c in (c1, c2):
        atom = mol.GetAtomWithIdx(c)
        has_bridge = n_double_o = n_carbon = 0
        class_word = None
        n_class = 0
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(c, nb.GetIdx())
            bt = bond.GetBondTypeAsDouble()
            sym = nb.GetSymbol()
            if nb.GetIdx() == bridge_o and bt == 1.0:
                has_bridge += 1
            elif sym == 'O' and bt == 2.0:
                n_double_o += 1
            elif sym == 'C':
                n_carbon += 1
            else:
                cw = _halide_pseudohalide_class_word(mol, nb, c)
                if cw is None:
                    cw = _hydrazido_class_word(mol, nb, c)
                if cw is None:
                    return None
                class_word = cw
                n_class += 1
        if not (has_bridge == 1 and n_double_o == 1 and n_class == 1
                and n_carbon == 0):
            return None
        words.append(class_word)
    words = sorted(words)                              # alphabetical
    if words[0] == words[1]:
        mult = get_multiplier_prefix(2, words[0])
        return f"dicarbonic {mult}{words[0]}"
    return f"dicarbonic {words[0]} {words[1]}"


def _name_mixed_inorganic_anhydride(mol) -> Optional[str]:
    """P-65.7.2: R-CO-O-C#N / R-CO-S-C#N -> '<acid> {cyanic|thiocyanic}
    anhydride' (acids cited alphabetically; BB 30979 'CH3-CO-O-CN acetic
    cyanic anhydride (PIN)'). Scoped ONLY to cyanic (acyl-O-C#N) and
    thiocyanic (acyl-S-C#N) per P-65.2.2 (BB 30979/30984) -- other P-65.7.2
    mixed anhydrides (carbonic/boric/phosphinous/...) are out of scope this
    cycle. Returns None (fall through) when no acyl-[O,S]-C#N core is present,
    or when the acyl side cannot be named."""
    for inorganic_word, smarts in _MIXED_INORGANIC_ANHYDRIDE_SMARTS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat, uniquify=True)
        if not matches:
            continue
        m = matches[0]                       # (c1, =O, bridge[O|S], C#N carbon, N)
        c1, o1, bridge = m[0], m[1], m[2]
        acid1, _sub1 = _name_acyl_acid(mol, c1, bridge, o1)
        if acid1 is None:
            return None
        acids = sorted([acid1, inorganic_word])
        return f"{acids[0]} {acids[1]} anhydride"
    return None


def _name_chalcogen_anhydride(mol) -> Optional[str]:
    """P-65.7.3: R-CO-X-CO-R' (X = S/Se/Te) -> '<acid stem(s)> {class}anhydride'.

    Each acyl side is named as its corresponding acid (benzoic, acetic, ...); the
    class term is thio/seleno/telluroanhydride per the bridge element. Symmetric
    -> '{acid} thioanhydride' (benzoic thioanhydride); mixed -> alphabetical two
    words. Returns None when no -CO-X-CO- core is present."""
    pat = Chem.MolFromSmarts(_CHALCOGEN_ANHYDRIDE_SMARTS)
    if pat is None:
        return None
    matches = mol.GetSubstructMatches(pat, uniquify=True)
    if not matches:
        return None
    # W8-P4 (P-65.7.6.4.1): TWO or more -CO-X-CO- linkages is the chalcogen
    # di-/poly-anhydride ('diacetic butanedioic bis(thioanhydride)', BB 32446),
    # not the single acyclic bis-acid shape below -- must be handled BEFORE
    # taking matches[0] (whose BFS would otherwise cross the 2nd linkage and
    # mis-name the whole chain, exactly the O-bridge dianhydride precedent).
    if len(matches) >= 2:
        _cda = _name_chalcogen_dianhydride(mol, matches)
        if _cda:
            return _cda
        return None   # >=2 linkages the dianhydride walk can't resolve -> abstain
    m = matches[0]                       # (c1, =O, X, c2, =O)
    c1, o1, x, c2, o2 = m[0], m[1], m[2], m[3], m[4]
    class_term = _CHALCOGEN_CLASS_TERM.get(mol.GetAtomWithIdx(x).GetSymbol())
    if class_term is None:
        return None
    # W8-P4 (P-65.7.7.3 / atom-drop safety floor): a CYCLIC -CO-X-CO- core
    # (c1 reaches c2 through the ring WITHOUT crossing the bridge X) is a
    # chalcogen analogue of a CYCLIC anhydride (e.g. the thio analogue of
    # phthalic anhydride), not the acyclic bis-acid shape below. The BFS-per-
    # side acid namer would otherwise walk the WHOLE fused ring from BOTH
    # carbonyls and mis-collapse it to a fictitious symmetric acid (verified
    # gate-off leak: 'O=C1SC(=O)c2ccccc12' -> wrongly 'benzoic thioanhydride',
    # dropping the ring fusion). Route to the dedicated cyclic-dione namer
    # (P-65.7.7.3 method 1, the PIN) instead; fail-closed if it declines.
    if c2 in _bfs_side_atoms(mol, c1, {x}):
        return _name_cyclic_chalcogen_dione(mol, c1, c2, x)
    acid1, sub1 = _name_acyl_acid(mol, c1, x, o1)
    acid2, sub2 = _name_acyl_acid(mol, c2, x, o2)
    if acid1 is None or acid2 is None:
        return None
    if acid1 == acid2:
        # Symmetric-substituted -> bis(...) (P-65.7.8.1 extends to chalcogen anhydrides).
        if sub1:
            return f"bis({acid1}) {class_term}"
        return f"{acid1} {class_term}"
    acids = sorted([acid1, acid2])
    return f"{acids[0]} {acids[1]} {class_term}"


def _oxo_acid_name(smi: str) -> Optional[str]:
    """Name the plain =O carboxylic acid of a capped free thio/seleno acid
    (R-C(=X)-OH, X = S/Se): swap the terminal =S/=Se for =O and route through the
    recursive fragment namer. Used only for the RING acyl fallback (the recursive
    namer names 'ethanethioic O-acid' for chains but declines a ring thioacid, so
    we build 'benzoic acid' -> 'benzenecarbothioic' from the =O form). Returns the
    full '... acid' string or None."""
    from ..assembly.fragment_naming import name_fragment_recursively
    fm = Chem.MolFromSmiles(smi)
    if fm is None:
        return None
    rw = Chem.RWMol(fm)
    changed = False
    for atom in rw.GetAtoms():
        if atom.GetSymbol() in ("S", "Se") and atom.GetDegree() == 1:
            b = atom.GetBonds()[0]
            if b.GetBondTypeAsDouble() == 2.0:
                atom.SetAtomicNum(8)
                changed = True
    if not changed:
        return None
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return name_fragment_recursively(Chem.MolToSmiles(rw))


def _name_acyl_acid_component(mol, acyl_c: int, acyl_x: int, bridge: int):
    """Name ONE acid component of a generalized acyl anhydride, returning
    ``(name_without_acid_word, is_substituted)``.

    The component's affix is fixed by the ACYL chalcogen ``acyl_x`` (=O -> the
    retained/plain '...oic' acid via :func:`_name_acyl_acid`; =S -> '...thioic';
    =Se -> '...selenoic'; P-65.7.6.4.3). The name is INDEPENDENT of the bridge
    (the bridge picks the class word). Fail-closed (``(None, False)``) if it
    cannot build the component."""
    x_sym = mol.GetAtomWithIdx(acyl_x).GetSymbol()
    if x_sym == "O":
        # Plain =O acid: the existing retained-aware namer (acetic/propanoic/benzoic).
        return _name_acyl_acid(mol, acyl_c, bridge, acyl_x)
    affix = _ACYL_AFFIX.get(x_sym)
    if affix is None:
        return None, False
    # Cap the acyl side into its free acid R-C(=X)-OH and name it. The recursive
    # namer yields e.g. 'ethanethioic O-acid'/'propaneselenoic O-acid' for chains;
    # strip the trailing acid word.
    from ..assembly.fragment_naming import name_fragment_recursively
    from .esters import _extract_fragment_smiles
    keep = _bfs_side_atoms(mol, acyl_c, {bridge})
    # 'substituted' (P-65.7.8.1 bis(...) wrap): any heteroatom on R other than the
    # acyl chalcogen itself (C/H/O/S/Se are the acid's own atoms).
    substituted = any(
        mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6, 8, 16, 34)
        for a in keep if a != acyl_x
    )
    smi = _extract_fragment_smiles(mol, set(keep), acyl_c, cap_element=8)
    if not smi:
        return None, False
    nm = name_fragment_recursively(smi)
    if nm:
        for suf in (" O-acid", " S-acid", " Se-acid", " acid"):
            if nm.endswith(suf):
                return nm[:-len(suf)].strip(), substituted
    # Ring acyl fallback: the recursive namer declines a ring thioacid, so build
    # the =O acid name ('benzoic'/'cyclohexanecarboxylic') and swap to the
    # 'carbothioic'/'carboselenoic' form.
    ox = _oxo_acid_name(smi)
    if ox and ox.endswith(" acid"):
        ox = ox[:-len(" acid")].strip()
        if ox == "benzoic":
            return f"benzenecarbo{affix}", substituted
        if ox.endswith("carboxylic"):
            return ox[:-len("carboxylic")] + "carbo" + affix, substituted
    return None, False


def _name_thioacyl_anhydride(mol) -> Optional[str]:
    """P-65.7.6.4.2/.3: acyclic thio-/seleno-ACYL anhydride R-C(=X1)-Y-C(=X2)-R'
    where at least one acyl chalcogen X is S/Se (a '...thioic'/'...selenoic'
    component). Each component acid is named with its acyl-chalcogen affix
    (:func:`_name_acyl_acid_component`), the two are cited in ALPHABETICAL order
    (P-65.7.2), and the class word is fixed by the BRIDGE chalcogen Y
    (O -> anhydride, S -> thioanhydride, Se -> selenoanhydride, Te ->
    telluroanhydride; P-65.7.6.4.2).

    Examples: CCC(=S)OC(=S)CC -> 'propanethioic anhydride'; CC(=S)OC(=O)CC ->
    'ethanethioic propanoic anhydride'; CC(=O)[Se]C(=S)CC -> 'acetic
    propanethioic selenoanhydride'.

    Declines (returns None -> the caller falls through):
      * both acyls =O (owned by the base O-bridge / chalcogen-bridge paths);
      * >1 anhydride linkage (di-/polyanhydride, out of scope this cycle);
      * a CYCLIC -C(=X)-Y-C(=X)- (a chalcogen analogue of a cyclic anhydride ->
        Phase 9 ring-fusion/dione nomenclature);
      * any component it cannot build (fail-closed, never a wrong name)."""
    pat = Chem.MolFromSmarts(_THIOACYL_ANHYDRIDE_SMARTS)
    if pat is None:
        return None
    matches = mol.GetSubstructMatches(pat, uniquify=True)
    if not matches or len(matches) != 1:
        return None  # 0 or >1 linkage -> not the single-linkage acyclic case
    c1, x1, bridge, c2, x2 = matches[0]
    # Both acyls =O -> the base/chalcogen paths own it; decline.
    if (mol.GetAtomWithIdx(x1).GetSymbol() == "O"
            and mol.GetAtomWithIdx(x2).GetSymbol() == "O"):
        return None
    # Cyclic (both carbonyls in one ring, reachable without crossing the bridge)
    # -> Phase 9 cyclic-anhydride nomenclature; decline (fail-closed).
    if c2 in _bfs_side_atoms(mol, c1, {bridge}):
        return None
    class_word = _BRIDGE_CLASS_WORD.get(mol.GetAtomWithIdx(bridge).GetSymbol())
    if class_word is None:
        return None
    acid1, sub1 = _name_acyl_acid_component(mol, c1, x1, bridge)
    acid2, sub2 = _name_acyl_acid_component(mol, c2, x2, bridge)
    if acid1 is None or acid2 is None:
        return None
    if acid1 == acid2:
        # Symmetric; a symmetrically SUBSTITUTED acid -> 'bis(...)' (P-65.7.8.1).
        if sub1:
            return f"bis({acid1}) {class_word}"
        return f"{acid1} {class_word}"
    from ..assembly.naming_utils import alpha_sort_key
    acids = sorted([acid1, acid2], key=alpha_sort_key)
    return f"{acids[0]} {acids[1]} {class_word}"


def _name_chalcogen_dianhydride(mol, matches) -> Optional[str]:
    """P-65.7.6.4.1: chalcogen analogue of a di-/poly-anhydride, ALL linkages
    the SAME chalcogen -> '<residues in occurrence order> {mult}({chalcogen}
    anhydride)' (BB 32446: 'CH3-CO-S-CO-CH2-CH2-CO-S-CO-CH3 diacetic
    butanedioic bis(thioanhydride) (PIN)'). Mirrors :func:`_name_dianhydride`'s
    residue walk (occurrence order starting from the alphabetically-lower
    terminal; identical residues numeric-collapsed) but the class term is
    'bis-'/'tris-'/... WRAPPING the singular chalcogen-anhydride word, per
    P-65.7.6.4.1 -- NOT the plain numeric-prefixed 'dianhydride'/'trianhydride'
    used for the O-bridge case. Fail-closed (None) for mixed chalcogens
    (P-65.7.6.4.2 seniority tie-break -- out of scope this cycle), non-linear
    topologies, or any residue the walk can't name."""
    infos = []
    bridge_atoms = set()
    bridge_elems = set()
    for m in matches:
        c1, o1, bridge, c2, o2 = m[0], m[1], m[2], m[3], m[4]
        bridge_elems.add(mol.GetAtomWithIdx(bridge).GetSymbol())
        infos.append({'c1': c1, 'c2': c2, 'bridge_o': bridge, 'o1': o1, 'o2': o2})
        bridge_atoms.add(bridge)
    if len(bridge_elems) != 1:
        return None   # mixed chalcogens -> P-65.7.6.4.2 tie-break, deferred
    class_word = _CHALCOGEN_CLASS_TERM.get(next(iter(bridge_elems)))
    if class_word is None:
        return None

    carbonyl_meta = {}
    for info in infos:
        carbonyl_meta[info['c1']] = (info['bridge_o'], info['o1'])
        carbonyl_meta[info['c2']] = (info['bridge_o'], info['o2'])

    frag_of = {c: frozenset(_bfs_side_atoms(mol, c, bridge_atoms))
               for c in carbonyl_meta}
    uniq = {}
    for c, f in frag_of.items():
        uniq.setdefault(f, []).append(c)

    res_name = {}
    for f, cs in uniq.items():
        nm = _name_residue_acid(mol, f, cs, carbonyl_meta)
        if nm is None:
            return None
        res_name[f] = nm

    adj = {f: [] for f in uniq}
    for info in infos:
        fa, fb = frag_of[info['c1']], frag_of[info['c2']]
        adj[fa].append(fb)
        adj[fb].append(fa)
    terminals = [f for f in uniq if len(adj[f]) == 1]
    if len(terminals) != 2:
        return None  # not a simple linear chain

    start = min(terminals, key=lambda f: res_name[f])
    order, visited, prev, cur = [], set(), None, start
    while cur is not None and cur not in visited:
        visited.add(cur)
        order.append(res_name[cur])
        nxts = [x for x in adj[cur] if x != prev and x not in visited]
        prev, cur = cur, (nxts[0] if nxts else None)
    if len(visited) != len(uniq):
        return None

    seq, counts = [], {}
    for nm in order:
        if nm not in counts:
            seq.append(nm)
            counts[nm] = 0
        counts[nm] += 1
    parts = [(_NUMERIC_MULT.get(counts[nm], "") if counts[nm] > 1 else "") + nm
             for nm in seq]

    bis_word = _BIS_MULT.get(len(matches))
    if bis_word is None:
        return None
    class_term = f"{bis_word}({class_word})"
    return " ".join(parts) + " " + class_term


def _name_cyclic_chalcogen_dione(mol, c1: int, c2: int, bridge_x: int) -> Optional[str]:
    """P-65.7.7.3 method 1 (the PIN): the chalcogen analogue of a cyclic
    anhydride, named as a heterocyclic pseudoketone (dione) with S/Se/Te
    replacing the ring oxygen -- e.g. the thio analogue of phthalic anhydride,
    'O=C1SC(=O)c2ccccc12' -> '2-benzothiophene-1,3-dione' (BB 32546:
    'hexahydro-2-benzothiophene-1,3-dione (PIN)' is the SATURATED benzo form;
    dropping 'hexahydro' gives the mancude/aromatic form here).

    Exact-canonical-SMILES lookup only (mirrors the O-bridge dione's retained-
    table branch in :func:`_name_cyclic_anhydride_dione`) -- no general
    saturated-chalcogen-oxa-dione engine is built this cycle (P4-8 design-
    contract item; the O-only :func:`_name_saturated_oxa_dione` stems are not
    generalised to S/Se/Te). Fail-closed (None) for anything not in the table,
    so the caller (:func:`_name_chalcogen_anhydride`) abstains rather than
    mis-name a cyclic structure it cannot fully resolve.
    """
    from ..data.retained_names import get_retained_name
    rn = get_retained_name(Chem.MolToSmiles(mol))
    if rn and rn.endswith("dione"):
        return rn
    return None


def _name_peroxy_anhydride(mol) -> Optional[str]:
    """P-65.7.4: R-CO-OO-CO-R' -> '<acid stem(s)> peroxyanhydride'.

    Each acyl side is named as its corresponding acid; the class term 'acid' is
    replaced by 'peroxyanhydride'. Symmetric -> '{acid} peroxyanhydride' (acetic
    peroxyanhydride); mixed -> alphabetical two words. Returns None when no
    -CO-OO-CO- core is present."""
    pat = Chem.MolFromSmarts(_PEROXY_ANHYDRIDE_SMARTS)
    if pat is None:
        return None
    matches = mol.GetSubstructMatches(pat, uniquify=True)
    if not matches:
        return None
    m = matches[0]
    carbonyls = [a for a in m if mol.GetAtomWithIdx(a).GetSymbol() == "C"]
    # The two bridge O's = the peroxy O-O pair (each has an O neighbour).
    bridge_pair = [a for a in m
                   if mol.GetAtomWithIdx(a).GetSymbol() == "O"
                   and any(mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == "O"
                           for n in mol.GetAtomWithIdx(a).GetNeighbors())]
    if len(carbonyls) != 2 or len(bridge_pair) != 2:
        return None
    # W8-P4 (atom-drop safety floor): a CYCLIC -CO-O-O-CO- core -- no evidence
    # this cycle, decline cleanly rather than risk the same ring-collapse
    # mis-naming found in the chalcogen-bridge sibling above.
    if carbonyls[1] in _bfs_side_atoms(mol, carbonyls[0], set(bridge_pair)):
        return None
    acids = []
    for c in carbonyls:
        atom = mol.GetAtomWithIdx(c)
        near_o = next((n.GetIdx() for n in atom.GetNeighbors()
                       if n.GetIdx() in bridge_pair), None)
        carbonyl_o = next((n.GetIdx() for n in atom.GetNeighbors()
                           if n.GetSymbol() == "O"
                           and mol.GetBondBetweenAtoms(c, n.GetIdx()).GetBondTypeAsDouble() == 2.0),
                          None)
        if near_o is None or carbonyl_o is None:
            return None
        nm, sub = _name_acyl_acid(mol, c, near_o, carbonyl_o)
        if nm is None:
            return None
        acids.append((nm, sub))
    if acids[0][0] == acids[1][0]:
        if acids[0][1]:
            return f"bis({acids[0][0]}) peroxyanhydride"
        return f"{acids[0][0]} peroxyanhydride"
    names = sorted(a[0] for a in acids)
    return f"{names[0]} {names[1]} peroxyanhydride"


def _parse_anhydride_core(mol, match: tuple) -> Optional[dict]:
    """Parse the anhydride core atoms from a SMARTS match.

    The SMARTS [CX3](=O)[OX2][CX3](=O) can return atoms in different orders
    depending on the molecule. This function identifies the key atoms reliably.

    Args:
        mol: RDKit Mol object.
        match: Tuple of atom indices from SMARTS match.

    Returns:
        Dict with keys 'c1', 'c2', 'bridge_o', 'o1', 'o2', or None.
    """
    # From the SMARTS pattern, we know:
    # - There are exactly 2 carbonyl carbons (CX3)
    # - There is exactly 1 bridge oxygen (OX2, single-bonded)
    # - There are 2 carbonyl oxygens (=O)

    match_set = set(match)
    carbons = []
    oxygens = []

    for idx in match:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            carbons.append(idx)
        elif atom.GetSymbol() == 'O':
            oxygens.append(idx)

    if len(carbons) != 2:
        return None

    # Identify bridge oxygen: bonded to both carbonyl carbons
    bridge_o = None
    carbonyl_os = []
    for oidx in oxygens:
        o_atom = mol.GetAtomWithIdx(oidx)
        bonded_carbons = set()
        for nbr in o_atom.GetNeighbors():
            if nbr.GetIdx() in set(carbons):
                bonded_carbons.add(nbr.GetIdx())
        if len(bonded_carbons) == 2:
            bridge_o = oidx
        else:
            carbonyl_os.append(oidx)

    if bridge_o is None:
        # Fallback: find the oxygen that is single-bonded to both C
        for oidx in oxygens:
            o_atom = mol.GetAtomWithIdx(oidx)
            c_neighbors = [n.GetIdx() for n in o_atom.GetNeighbors()
                           if n.GetIdx() in set(carbons)]
            if len(c_neighbors) == 2:
                bridge_o = oidx
                break

    if bridge_o is None:
        return None

    return {
        'c1': carbons[0],
        'c2': carbons[1],
        'bridge_o': bridge_o,
        'o1': carbonyl_os[0] if len(carbonyl_os) > 0 else None,
        'o2': carbonyl_os[1] if len(carbonyl_os) > 1 else None,
    }


def _count_acyl_fragment_carbons(mol, carbonyl_c: int, bridge_o: int) -> int:
    """Count carbons in an acyl fragment (one side of the anhydride).

    BFS from carbonyl carbon, following only carbon atoms, avoiding the bridge oxygen.

    Args:
        mol: RDKit Mol object.
        carbonyl_c: Atom index of the carbonyl carbon.
        bridge_o: Atom index of the bridge oxygen (to exclude).

    Returns:
        Number of carbons in the acyl fragment (including carbonyl carbon).
    """
    visited = {carbonyl_c}
    queue = deque([carbonyl_c])
    carbon_count = 1

    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in visited:
                continue
            if nidx == bridge_o:
                continue
            # Only follow carbon atoms (skip oxygens, halogens etc.)
            if neighbor.GetSymbol() == 'C':
                visited.add(nidx)
                queue.append(nidx)
                carbon_count += 1

    return carbon_count


def _collect_acyl_fragment_atoms(mol, carbonyl_c: int, bridge_o: int) -> set:
    """Collect ALL heavy atom indices in an acyl fragment via BFS.

    Unlike _count_acyl_fragment_carbons which only follows carbons,
    this follows all bonds except through the bridge oxygen.

    Args:
        mol: RDKit Mol object.
        carbonyl_c: Atom index of the carbonyl carbon.
        bridge_o: Atom index of the bridge oxygen (to exclude).

    Returns:
        Set of atom indices in the acyl fragment.
    """
    visited = {carbonyl_c}
    queue = deque([carbonyl_c])
    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in visited or nidx == bridge_o:
                continue
            visited.add(nidx)
            queue.append(nidx)
    return visited


def _find_longest_chain(mol, start: int, frag_atoms: set) -> list:
    """Find the longest carbon chain starting from ``start`` within frag_atoms.

    Uses DFS to find the longest simple path of carbon atoms.

    Args:
        mol: RDKit Mol object.
        start: Atom index to start from (carbonyl carbon).
        frag_atoms: Set of atom indices in the fragment.

    Returns:
        Ordered list of atom indices [start, ..., end].
    """
    carbon_set = {i for i in frag_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C'}

    best_path = [start]

    def dfs(current, visited, path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in visited or nidx not in carbon_set:
                continue
            visited.add(nidx)
            path.append(nidx)
            dfs(nidx, visited, path)
            path.pop()
            visited.discard(nidx)

    dfs(start, {start}, [start])
    return best_path


def _name_acyl_acid(mol, carbonyl_c: int, bridge_o: int, carbonyl_o: int):
    """Name an acyl fragment as its corresponding acid, with substituent prefixes.

    Uses _integrate_universal_prefixes() (per D-01) to discover branch
    substituents on the acyl chain. For simple linear chains, falls back
    to the fast _build_acid_name() path.

    Args:
        mol: RDKit Mol object.
        carbonyl_c: Index of the carbonyl carbon C(=O).
        bridge_o: Index of the bridge oxygen (to exclude from fragment).
        carbonyl_o: Index of the carbonyl oxygen (=O).

    Returns:
        W3-P06: a ``(name, is_substituted)`` tuple — the acid name WITHOUT the
        " acid" word (e.g. "2-methylpropanoic"), plus a flag that is True iff the
        acid carries a substituent prefix (a non-empty ``prefix_str``). The flag
        drives the P-65.7.8.1 symmetric-substituted 'bis(...)' wrap WITHOUT
        string-sniffing the name. Returns ``(None, False)`` on failure.
    """
    frag_atoms = _collect_acyl_fragment_atoms(mol, carbonyl_c, bridge_o)

    # Identify carbon atoms in the fragment
    chain_carbons = [i for i in frag_atoms
                     if mol.GetAtomWithIdx(i).GetSymbol() == 'C']

    # Quick path: check if fragment has any branching or heteroatom substituents.
    # If it's a simple linear chain of only carbons + the carbonyl O, use fast path.
    non_c_non_carbonyl_o = [i for i in frag_atoms
                            if i != carbonyl_o
                            and mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    is_branched = False
    for i in chain_carbons:
        atom = mol.GetAtomWithIdx(i)
        c_nbrs_in_frag = sum(1 for n in atom.GetNeighbors()
                             if n.GetIdx() in frag_atoms
                             and n.GetSymbol() == 'C')
        if c_nbrs_in_frag > 2:
            is_branched = True
            break

    # Ring acyl (benzoyl -> "benzoic", cyclohexanecarbonyl ->
    # "cyclohexanecarboxylic") or a simple acyclic chain (1C -> "formic",
    # 2C -> "acetic", fatty/unsaturated, else systematic) is named by the
    # canonical retained-aware acid namer.  Per IUPAC P-65.1.1.1 these
    # retained acid names ARE the PINs cited in the anhydride functional-class
    # name (acetic anhydride, benzoic anhydride), so the systematic-only
    # _build_acid_name path produced the wrong stem (ethanoic; heptanoic for a
    # linearised benzene ring).  Branched or heteroatom-substituted *acyclic*
    # acids still need the substituent-prefix path below.
    from .esters import acid_is_ring_acid, get_acid_fragment_name
    frag_list = list(frag_atoms)
    if acid_is_ring_acid(mol, frag_list) or (
        not is_branched and not non_c_non_carbonyl_o
    ):
        return get_acid_fragment_name(mol, frag_list), False

    # Branched or has heteroatom substituents: use _integrate_universal_prefixes()
    # per D-01 to discover and name substituents on the acyl chain.
    principal_chain = _find_longest_chain(mol, carbonyl_c, frag_atoms)
    chain_set = set(principal_chain)

    # Build atom_to_locant: carbonyl C = position 1 (IUPAC carboxylic acid numbering)
    atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(principal_chain)}

    # Exclude atoms: (1) carbonyl oxygen (=O) is part of the acid suffix,
    # not a substituent; (2) all atoms outside this fragment (the other acyl
    # side + bridge oxygen) must be excluded so the discovery engine only
    # sees substituents within THIS acyl fragment.
    all_atom_idxs = set(range(mol.GetNumAtoms()))
    exclude = (all_atom_idxs - frag_atoms) | {carbonyl_o}

    from ..assembly.composer import _integrate_universal_prefixes
    prefix_str = _integrate_universal_prefixes(
        mol, chain_set,
        parent_type="chain",
        principal_chain=principal_chain,
        atom_to_locant=atom_to_locant,
        exclude_atoms=exclude,
    )

    # W3-P06 Task 5 (P-65.1.1.1): a substituted 2-carbon acid uses the RETAINED
    # 'acetic' stem with UNLOCANTED substituent prefixes (chloroacetic,
    # dichloroacetic) — acetic has a single substitutable carbon, so no locant is
    # cited (BB 30346 'difluoroacetic acid (not 2,2-difluoroacetic acid)'; BB
    # 32571 'bis(chloroacetic) anhydride'). Guard: only for SIMPLE substituents
    # (no '(' complex substituent, whose own internal locants must not be
    # stripped) — a complex-substituted C2 acid keeps the systematic 'ethanoic'.
    if prefix_str and len(principal_chain) == 2 and "(" not in prefix_str:
        unlocanted = re.sub(r"\d+(?:,\d+)*-", "", prefix_str)
        return f"{unlocanted}acetic", True
    base_acid = _build_acid_name(len(principal_chain))
    if prefix_str:
        return f"{prefix_str}{base_acid}", True
    return base_acid, False


def _build_acid_name(chain_length: int) -> str:
    """Build the acid name from chain length.

    Args:
        chain_length: Number of carbons including the carbonyl carbon.

    Returns:
        Acid name string (e.g., 'ethanoic', 'propanoic', 'butanoic').
    """
    prefix = get_chain_prefix(chain_length)
    return f"{prefix}anoic"


def _name_cyclic_anhydride(num_carbons: int) -> str:
    """Name a cyclic anhydride from a dicarboxylic acid.

    P-65.7.7.1 METHOD (2) ('{diacid} anhydride') — NOT the PIN (method 1 = the
    heterocyclic-pseudoketone dione). Retained only as the last-resort general-
    nomenclature fallback for a cyclic anhydride the dione namer declines.

    Args:
        num_carbons: Number of carbon atoms in the ring (both carbonyl carbons + chain).

    Returns:
        IUPAC name string (e.g., 'butanedioic anhydride', 'pentanedioic anhydride').
    """
    prefix = get_chain_prefix(num_carbons)
    return f"{prefix}anedioic anhydride"


# Saturated monocyclic oxa-heterocycle stems (the anhydride bridge O is the ring
# heteroatom at locant 1). D-FOLLOWON item 6 (P-65.7.7.1 method 1).
_SATURATED_OXA_STEMS = {
    3: "oxirane", 4: "oxetane", 5: "oxolane", 6: "oxane",
    7: "oxepane", 8: "oxocane", 9: "oxonane", 10: "oxecane",
}


def _name_cyclic_anhydride_dione(mol, c1: int, c2: int, bridge_o: int, ring) -> Optional[str]:
    """Heterocyclic-pseudoketone (dione) PIN for a cyclic anhydride (P-65.7.7.1
    method 1, the PREFERRED form): succinic -> oxolane-2,5-dione, glutaric ->
    oxane-2,6-dione, maleic -> furan-2,5-dione, phthalic -> 2-benzofuran-1,3-dione,
    methylsuccinic -> 3-methyloxolane-2,5-dione. Returns None (fail-closed) for
    anything it cannot name, so name_anhydride falls back to the general
    '{diacid} anhydride'."""
    from ..data.retained_names import get_retained_name

    # (a) Mancude / fused anhydrides whose dione PIN is a retained-table entry
    # (maleic -> furan-2,5-dione, phthalic -> 2-benzofuran-1,3-dione). The catalog
    # value IS the PIN dione (retargeted), so emit it directly. (The cyclic-oxo
    # engine produces the same strings; using the table avoids its retained
    # self-gate and keeps this handler the priority-1200 gatekeeper.)
    rn = get_retained_name(Chem.MolToSmiles(mol))
    if rn and rn.endswith("dione"):
        return rn

    # (b) Saturated monocyclic oxa-heterocycle dione (succinic/glutaric/...).
    return _name_saturated_oxa_dione(mol, c1, c2, bridge_o, ring)


def _name_saturated_oxa_dione(mol, c1: int, c2: int, bridge_o: int, ring) -> Optional[str]:
    """Name a FULLY-SATURATED monocyclic oxa-heterocycle dione (the anhydride O at
    locant 1; the two ring carbonyls flank it at 2 and n). Substituents become
    prefixes with lowest locants. Fail-closed for fused / mancude / non-oxa /
    >oxecane rings."""
    ring = list(ring)
    n = len(ring)
    stem = _SATURATED_OXA_STEMS.get(n)
    if stem is None:
        return None
    ring_set = set(ring)
    # Monocyclic, all atoms in exactly this ring, ring = {one O (bridge) + carbons}.
    ri = mol.GetRingInfo()
    for a in ring:
        if ri.NumAtomRings(a) != 1:
            return None  # fused / bridged -> not this path
        at = mol.GetAtomWithIdx(a)
        if a == bridge_o:
            if at.GetSymbol() != "O":
                return None
        elif at.GetSymbol() != "C":
            return None
    # No in-ring unsaturation (mancude/aromatic handled by the retained-dione branch).
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            if b.GetIsAromatic() or b.GetBondType() == Chem.BondType.DOUBLE:
                return None

    def ring_nbrs(a):
        return [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() in ring_set]

    o_nbrs = ring_nbrs(bridge_o)
    if len(o_nbrs) != 2 or set(o_nbrs) != {c1, c2}:
        return None  # carbonyls must flank the bridge O (true for an anhydride)

    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix
    from ..assembly.substituent_enumerator import name_substituent

    best = None
    for start in o_nbrs:
        loc = {bridge_o: 1}
        prev, cur, k = bridge_o, start, 2
        while cur is not None and cur not in loc:
            loc[cur] = k
            k += 1
            nxt = [x for x in ring_nbrs(cur) if x != prev and x not in loc]
            prev, cur = cur, (nxt[0] if nxt else None)
        if len(loc) != n:
            continue
        subs = []  # (locant, name)
        ok = True
        for a in ring:
            if a in (bridge_o, c1, c2):
                continue
            attach = next((nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                           if nb.GetIdx() not in ring_set), None)
            if attach is None:
                continue
            frag, seen, stack = [], set(ring_set), [attach]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                frag.append(x)
                stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors()
                             if nb.GetIdx() not in seen)
            nm = name_substituent(mol, frag, attach)
            if not nm:
                ok = False
                break
            subs.append((loc[a], nm))
        if not ok:
            continue
        key = sorted(l for l, _ in subs)
        if best is None or key < best[0]:
            best = (key, loc, subs)
    if best is None:
        return None
    _, loc, subs = best
    cl = sorted([loc[c1], loc[c2]])

    # Assemble the substituent prefix (group by name, multiply, alphabetize).
    groups = {}
    for locant, nm in subs:
        groups.setdefault(nm, []).append(locant)
    parts = []
    for nm in sorted(groups, key=alpha_sort_key):
        locs = sorted(groups[nm])
        mult = get_multiplier_prefix(len(locs), nm) if len(locs) > 1 else ""
        parts.append(f"{','.join(str(x) for x in locs)}-{mult}{nm}")
    # The substituent prefix attaches DIRECTLY to the parent stem (no separating
    # hyphen): '3-methyloxolane-2,5-dione', not '3-methyl-oxolane-2,5-dione'.
    prefix = "-".join(parts)
    return f"{prefix}{stem}-{cl[0]},{cl[1]}-dione"


def get_anhydride_consumed_atoms(functional_groups: dict) -> set:
    """Get the set of atom indices consumed by anhydride groups.

    The bridge oxygen and ester-like bonds in anhydrides should be filtered
    from ester FG detection to prevent double-counting.

    Args:
        functional_groups: Dict from detect_functional_groups().

    Returns:
        Set of all atom indices that are part of anhydride groups.
    """
    consumed = set()
    for match in functional_groups.get("anhydride", []):
        consumed.update(match)
    return consumed
