"""Best-effort T4 producer for the acyclic polyol / polyether / polyester class.

The composer names a pure polyol (`1,5,6-tris(2,3-dihydroxypropoxy)hexane-2,3,4-triol`)
cleanly because its only characteristic group is the alcohol. When the SAME polyol
carries one or more esters, the ester
is senior (P-41), so parent selection commits to the ester and `name_ester`
LINEARISES the 23-carbon polyol OR side into a wrong 'tricosyl 2-methylpropanoate'
(SELF-01-suppressed), and the multiplicative-ester PIN neither reference builds.

This producer supplies the T4 degrade P-65.6.3.2 method (2) licenses: DEMOTE every
ester to an ``(Racyloxy)`` prefix, choose the polyol carbon chain as the parent, and
cite the free hydroxyls as the ``-ol`` suffix -- a valid, round-trippable
(non-PIN) systematic name:

    1,5,6-tris[2-hydroxy-3-(2-methylprop-2-enoyloxy)propoxy]hexane-2,3,4-triol

It runs ONLY on the best-effort T4 lane (after the PIN/ester path has abstained),
each emission E1-covered by construction (every atom is bound to the parent, a
free-OH suffix, or a named arm) and SELF-01 round-trip verified downstream, so the
0-wrong net is unchanged.

Scope (fail closed -> None outside it): a SINGLE acyclic component whose only
heteroatom is oxygen, whose carbon skeleton has ONE parent chain covering every
"core" carbon (a carbon bonded to >=1 core carbon), where every oxygen is exactly
one of -- a free hydroxyl on a core carbon (the suffix), a bridging ether to a
name_substituent-nameable arm (an alkoxy prefix), or an acyloxy ester
(-O-C(=O)-R, the acyloxy prefix) -- and at least one ester is present (so a plain
polyol, already named by the composer, is never intercepted).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem


def name_acyclic_polyol_polyester(mol) -> Optional[str]:
    """Return the T4 polyol-polyester name for ``mol`` or ``None`` (fail closed)."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    ri = mol.GetRingInfo()
    if ri.NumRings() > 0:
        return None  # acyclic only
    # Single component, atoms limited to C/H/O.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for a in mol.GetAtoms():
        if a.GetSymbol() not in ('C', 'O', 'H') or a.GetFormalCharge() != 0:
            return None

    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    if not carbons:
        return None

    # --- Classify every oxygen. ---
    # free_oh[core_c] -> a -OH on that core carbon
    # ether_arm[(core_c, o)] -> a bridging ether O to an arm (alkoxy prefix)
    # acyloxy[(core_c, o, cc)] -> an -O-C(=O)- ester (acyloxy prefix)
    # Any oxygen that does not fit exactly one role -> decline.
    # --- Ester carbonyl carbons (-C(=O)-O-): excluded from the parent carbon
    # skeleton so an arm's ester O separates the arm from the core cleanly. ---
    def _is_ester_carbonyl(a) -> bool:
        if a.GetSymbol() != 'C':
            return False
        has_dbl_o = any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(a).GetSymbol() == 'O'
            and b.GetOtherAtom(a).GetDegree() == 1
            for b in a.GetBonds())
        has_single_o = any(
            b.GetBondType() == Chem.BondType.SINGLE
            and b.GetOtherAtom(a).GetSymbol() == 'O'
            for b in a.GetBonds())
        return has_dbl_o and has_single_o
    acyl_carbons: Set[int] = {a.GetIdx() for a in mol.GetAtoms()
                              if _is_ester_carbonyl(a)}

    # Build the carbon-only adjacency (C-C single bonds), excluding acyl carbons.
    adj: Dict[int, List[int]] = {c: [] for c in carbons if c not in acyl_carbons}
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    # Connected carbon components; the parent core is the LARGEST.
    seen: Set[int] = set()
    components: List[Set[int]] = []
    for c in adj:
        if c in seen:
            continue
        comp: Set[int] = set()
        stack = [c]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(n for n in adj[x] if n not in seen)
        components.append(comp)
    if not components:
        return None
    core = max(components, key=len)
    if len(core) < 2:
        return None  # a 1-carbon "polyol" is not this class

    # --- Longest simple path in the core = the parent chain. ---
    parent = _longest_path(adj, core)
    if parent is None or len(parent) != len(core):
        return None  # branched core beyond a single chain -> decline (v1)

    # --- Classify EACH CORE CARBON's own substituents. Arm-internal oxygens
    # (an -OH or ester INSIDE an arm) are named by name_substituent on the
    # arm, never classified here -- only the O directly bonded to a core
    # carbon is a core role (suffix -ol / alkoxy prefix / acyloxy prefix). ---
    core_set = set(core)
    prefix_on: Dict[int, List[str]] = {}
    suffix_carbons: Set[int] = set()
    have_ester = False

    for c in core:
        ca = mol.GetAtomWithIdx(c)
        for n in ca.GetNeighbors():
            if n.GetAtomicNum() <= 1:
                continue
            ni = n.GetIdx()
            if ni in core_set:
                continue  # a chain C-C bond
            if n.GetSymbol() != 'O':
                return None  # a non-O heavy substituent on the core -> out of scope
            o = n
            # a free hydroxyl -> the -ol suffix
            if o.GetTotalNumHs() >= 1 and o.GetDegree() == 1:
                if any(b.GetBondType() != Chem.BondType.SINGLE for b in o.GetBonds()):
                    return None
                suffix_carbons.add(c)
                continue
            # an -O- with exactly one further heavy neighbour (ether or acyloxy)
            if (o.GetTotalNumHs() == 0 and o.GetDegree() == 2
                    and all(b.GetBondType() == Chem.BondType.SINGLE
                            for b in o.GetBonds())):
                others = [x.GetIdx() for x in o.GetNeighbors() if x.GetIdx() != c]
                if len(others) != 1:
                    return None
                other = others[0]
                if mol.GetAtomWithIdx(other).GetSymbol() != 'C':
                    return None
                if other in acyl_carbons:
                    name = _acyloxy_name(mol, other, ni)
                    if name is None:
                        return None
                    have_ester = True
                    prefix_on.setdefault(c, []).append(name)
                    continue
                name = _name_o_arm(mol, ni, other)
                if name is None:
                    return None
                prefix_on.setdefault(c, []).append(name)
                continue
            return None  # any other O shape directly on the core -> decline

    # The intercept only fires when an ester is PRESENT anywhere in the molecule
    # (so a plain polyol/polyether, which the composer already names as a PIN, is
    # never taken over). The ester may sit directly on the core (an acyloxy
    # prefix above) OR inside an arm (named within name_substituent) -- either
    # way it is what blocks the composer's alcohol-parent path (ester seniority).
    if not have_ester:
        _ester_smarts = Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')
        if _ester_smarts is not None and mol.HasSubstructMatch(_ester_smarts):
            have_ester = True
    if not have_ester:
        return None  # a plain polyol/ether is the composer's job, not this degrade
    if not suffix_carbons:
        return None  # no free -OH -> not an "-ol" parent (out of v1 scope)

    # --- Number the parent for lowest locants to the -ol suffix set. ---
    numbering = _number_parent(parent, suffix_carbons, prefix_on)
    if numbering is None:
        return None

    return _assemble(mol, parent, numbering, suffix_carbons, prefix_on)


def _longest_path(adj: Dict[int, List[int]], nodes: Set[int]) -> Optional[List[int]]:
    """Longest simple path within ``nodes`` (a small acyclic carbon tree)."""
    # Endpoints have degree <= 1 within the component (a chain). For a pure chain
    # the two ends are the degree-1 nodes; DFS from one end.
    sub_adj = {n: [m for m in adj[n] if m in nodes] for n in nodes}
    ends = [n for n in nodes if len(sub_adj[n]) <= 1]
    if not ends:
        return None  # a cycle slipped through (shouldn't: acyclic guard)
    best: List[int] = []
    for start in ends:
        # DFS longest simple path from start
        stack = [(start, [start], {start})]
        while stack:
            node, path, visited = stack.pop()
            extended = False
            for nb in sub_adj[node]:
                if nb not in visited:
                    extended = True
                    stack.append((nb, path + [nb], visited | {nb}))
            if not extended and len(path) > len(best):
                best = path
    return best or None


def _name_o_arm(mol, o_idx: int, arm_seed: int) -> Optional[str]:
    """Name a bridging ether -O-<arm> as an alkoxy prefix via the substituent
    cascade (`name_substituent`), enclosing the compound token."""
    from ..assembly.substituent_enumerator import name_substituent
    # fragment = O + everything on the arm side (reachable from arm_seed w/o the host)
    frag: Set[int] = {o_idx}
    stack = [arm_seed]
    seen: Set[int] = {o_idx}
    # exclude the host by only walking from arm_seed and not re-crossing o_idx
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        frag.add(x)
        for n in mol.GetAtomWithIdx(x).GetNeighbors():
            if n.GetIdx() != o_idx and n.GetIdx() not in seen:
                stack.append(n.GetIdx())
    word = name_substituent(mol, sorted(frag), o_idx)
    if not word or word == 'substituent' or ' ' in word:
        return None
    from ..assembly.naming_utils import enclose_if_compound
    return enclose_if_compound(word)


def _acyloxy_name(mol, carbonyl_c: int, ester_o: int) -> Optional[str]:
    """Name a -O-C(=O)-R ester as the '(Racyloxy)' prefix (reuses the shared
    acid engine); the whole acyl side must be plain (one =O, <=1 all-carbon R)."""
    cc = mol.GetAtomWithIdx(carbonyl_c)
    oxo = 0
    r_side = 0
    for b in cc.GetBonds():
        o = b.GetOtherAtom(cc)
        if o.GetIdx() == ester_o:
            continue
        if b.GetBondType() == Chem.BondType.DOUBLE and o.GetSymbol() == 'O' and o.GetDegree() == 1:
            oxo += 1
        elif b.GetBondType() == Chem.BondType.SINGLE and o.GetSymbol() == 'C':
            r_side += 1
        elif o.GetSymbol() == 'H':
            continue
        else:
            return None
    if oxo != 1 or r_side > 1:
        return None
    from .lipids import _acyloxy_for_site
    from ..assembly.naming_utils import enclose_if_compound
    ax = _acyloxy_for_site(mol, ('acyl', carbonyl_c, ester_o))
    if not ax or ' ' in ax:
        return None
    return enclose_if_compound(ax)


def _number_parent(parent: List[int], suffix_carbons: Set[int],
                   prefix_on: Dict[int, List[str]]) -> Optional[Dict[int, int]]:
    """Return atom->locant for the numbering that gives the -ol suffix set the
    lowest locants (P-31.1.4), tie-broken by the prefix set."""
    def score(order):
        pos = {a: i + 1 for i, a in enumerate(order)}
        suf = sorted(pos[a] for a in suffix_carbons)
        pre = sorted(pos[a] for a in prefix_on)
        return (suf, pre)
    fwd = parent
    rev = list(reversed(parent))
    best = min((fwd, rev), key=score)
    return {a: i + 1 for i, a in enumerate(best)}


def _assemble(mol, parent, numbering, suffix_carbons, prefix_on) -> Optional[str]:
    from ..data.chain_names import get_chain_prefix
    from ..assembly.naming_utils import alpha_sort_key
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS, COMPLEX_MULTIPLIERS

    n = len(parent)
    stem = get_chain_prefix(n)
    if not stem:
        return None

    # --- prefixes: group identical tokens, cite with locants, alpha order. ---
    groups: Dict[str, List[int]] = {}
    for c, toks in prefix_on.items():
        for t in toks:
            groups.setdefault(t, []).append(numbering[c])
    parts: List[str] = []
    for tok in sorted(groups, key=alpha_sort_key):
        locs = sorted(groups[tok])
        # a compound (enclosed) token multiplies bis/tris; a bare one di/tri
        compound = tok.startswith('(') or tok.startswith('[')
        table = COMPLEX_MULTIPLIERS if compound else SIMPLE_MULTIPLIERS
        mult = '' if len(locs) == 1 else table.get(len(locs))
        if mult is None:
            return None
        loc_str = ','.join(str(x) for x in locs)
        parts.append(f"{loc_str}-{mult}{tok}")
    prefix_str = '-'.join(parts)

    # --- -ol suffix with lowest locants. ---
    ol_locs = sorted(numbering[c] for c in suffix_carbons)
    ol_mult = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}.get(
        len(ol_locs))
    if ol_mult is None:
        return None
    loc_str = ','.join(str(x) for x in ol_locs)
    suffix = f"-{loc_str}-{ol_mult}ol"

    name = f"{prefix_str}{stem}ane{suffix}" if prefix_str else f"{stem}ane{suffix}"
    # tidy a stray leading hyphen / double hyphen
    name = name.replace('--', '-').lstrip('-')
    return name
