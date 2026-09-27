"""
 a phase Step 2 -- the assembly WEAVER (internal notes).

Step 1  made every fragment of a multi-linkage molecule name
successfully, but the whole molecule still abstained because the flat
weaver (`fragment_assembly.py`'s `_assemble_by_bond_type` / the
`_assemble_multi_*` star assemblers) can only combine fragments whose
STANDALONE names happen to carry a matching suffix (an acid, an alcohol, an
amine,...). A T4-rescued or seniority-demoted fragment often carries none
(a ring parent with every group demoted to a prefix), so the converter
returns ``None`` and the fragment is either silently dropped or space-joined
-- OPSIN then sees disconnected components and rejects the name.

This module is the fix the trace recommends: a **core-and-arms composer**
that never converts a standalone fragment NAME -- it builds every arm's
prefix directly from the ORIGINAL (uncapped) molecule, anchored at the real
attachment atom, via `assembly.substituent_enumerator.name_substituent` (the
same structure-based primitive `composer.py` already uses for every prefix
in a normal, non-decomposed name) or the proven acid-fragment-reuse acyloxy
builder `rules.lipids._acyloxy_for_site` already uses for glycerides.

Scope (v1, FAILS CLOSED outside it -- never guesses, never ships a partial):

- A single connected component, wholly ACYCLIC (no ring anywhere in the
  molecule). A ring hub (e.g. a GPI-anchor's mannose core) is out of scope
  for this version and declines honestly -- see the trace's (ii) bucket.
- A carbon-only "core" chain, selected as the carbon-only connected
  component (after excluding ester-carbonyl carbons, so an acyl group never
  fuses onto the backbone graph) with the most external heavy-atom
  attachment points (>= 2) -- the structural HUB of the star. This is
  chosen by attachment COUNT, not atom count: a 3-carbon glycerol backbone
  is the hub even though every one of its fatty-acid arms is far larger.
- Every attachment directly on a core atom must be exactly one of:
  a free hydroxyl (the ``-ol`` suffix), an ether/alkoxy arm, an acyloxy
  (ester) arm, or a phosphoryloxy (neutral, mono-protonated phosphodiester)
  arm whose own far ("head") side is nameable either directly via
  `name_substituent` (an ordinary alkoxy substituent) or via the narrow
  charged-onium-arm builder below (Part B -- e.g. choline). Any other shape
  on a core atom (a non-oxygen substituent, a charged/anionic phosphate,
  more than one free valence per position,...) declines the WHOLE
  molecule -- never a partial name.

Part B (narrow, PC-family): a cut quaternary-ammonium/onium arm reached via
a SIMPLE, unbranched, all-single-bond carbon chain (e.g. choline's
``-CH2CH2-N+(CH3)3``) is named structurally -- `cation_to_prefix` for the
onium's own substituents, wrapped in a chain-length + locant string (e.g.
``2-(trimethylazaniumyl)ethyl``) -- never a hardcoded per-head-group string.

Part C (atom-coverage / 0-wrong guard): this module returns a CANDIDATE
only. The caller (`decomposition/engine.py`) MUST verify the candidate
covers every heavy atom of the input before shipping it (`weave_is_verified`
below performs the check via a full OPSIN round-trip + heavy-atom-count
comparison); a candidate that fails is discarded, never shipped.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

from rdkit import Chem

from ..assembly.naming_utils import (
    COMPLEX_MULTIPLIERS,
    SIMPLE_MULTIPLIERS,
    alpha_sort_key,
    enclose_if_compound,
)
from ..assembly.substituent_enumerator import name_substituent
from ..data.chain_names import get_chain_prefix

# Onium elements whose closed-shell cation (no free H) can terminate a
# Part-B charged arm (mirrors `rules/ions.py`'s `_YLIDE_ONIUM_ELEMENTS`,
# narrowed here to the elements a simple alkyl-chain-linked head group
# plausibly uses).
_ONIUM_ELEMENTS = frozenset({'N', 'P', 'S', 'O', 'As', 'Sb', 'Se', 'Te'})

_HYDRO_MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}


def try_weave(mol, style: str = "pin") -> Optional[str]:
    """Return a core-and-arms substitutive name for *mol*, or ``None``.

    Never raises -- any internal failure is a decline (returns ``None``) so
    the caller falls through to the existing flat weaver / an honest
    abstention. This function does NOT verify the result; the caller must
    run it through `weave_is_verified` before shipping it (Part C).
    """
    try:
        return _try_weave_impl(mol, style)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core selection
# ---------------------------------------------------------------------------

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


def _select_core_candidates(mol) -> List[Set[int]]:
    """Carbon-only connected components with >= 2 external heavy-atom
    attachments (candidate structural hubs of the star), ordered by
    preference: MOST attachments first, then fewest heavy atoms.

    A single molecule can have more than one such component -- e.g. a
    choline arm's own ``-CH2-CH2-`` linker (2 attachments: the ester O it
    enters from, the onium N it exits to) is itself a 2-attachment
    "component" exactly like a plain ethane-1,2-diyl backbone. The caller
    tries each candidate in turn and keeps the first that fully classifies
    and assembles, so a wrong first guess costs a cheap retry, never a wrong
    emission (Part C also gates the final result)."""
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    if not carbons:
        return []
    acyl_carbons = {a.GetIdx() for a in mol.GetAtoms() if _is_ester_carbonyl(a)}

    # NOTE: every carbon-carbon bond counts here, regardless of bond order --
    # an internal double bond (an arm's own C=C, e.g. an acrylate or an
    # unsaturated fatty-acyl chain) must NOT split that arm into separate
    # "components", or an isolated alkene carbon is miscounted as its own
    # 2-attachment false hub (measured: the acrylate's own CH=CH2 carbon).
    # The core-must-be-saturated restriction is enforced separately, only
    # for bonds WITHIN the chosen core (see `_try_weave_impl`).
    adj: Dict[int, List[int]] = {c: [] for c in carbons if c not in acyl_carbons}
    for b in mol.GetBonds():
        if b.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                    Chem.BondType.AROMATIC):
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

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
        return []

    def _attachments(comp: Set[int]) -> int:
        total = 0
        for c in comp:
            for n in mol.GetAtomWithIdx(c).GetNeighbors():
                if n.GetAtomicNum() > 1 and n.GetIdx() not in comp:
                    total += 1
        return total

    hubs = [comp for comp in components if _attachments(comp) >= 2]
    hubs.sort(key=lambda s: (-_attachments(s), len(s), min(s)))
    return hubs


def _longest_path(adj: Dict[int, List[int]], nodes: Set[int]) -> Optional[List[int]]:
    """Longest simple path within *nodes* (the core must be a plain chain,
    not a branched or cyclic carbon skeleton -- v1 scope)."""
    sub_adj = {n: [m for m in adj[n] if m in nodes] for n in nodes}
    ends = [n for n in nodes if len(sub_adj[n]) <= 1]
    if not ends:
        return None
    best: List[int] = []
    for start in ends:
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _arm_atoms(mol, core_set: Set[int], start: int) -> Set[int]:
    """Heavy atoms reachable from *start* without crossing into *core_set*."""
    seen: Set[int] = set()
    dq = deque([start])
    while dq:
        x = dq.popleft()
        if x in seen or x in core_set:
            continue
        seen.add(x)
        for n in mol.GetAtomWithIdx(x).GetNeighbors():
            ni = n.GetIdx()
            if ni not in seen and ni not in core_set:
                dq.append(ni)
    return seen


# ---------------------------------------------------------------------------
# Part B: charged (onium) arm -- structural, not a per-head-group string.
# ---------------------------------------------------------------------------

def _charged_arm_alkyl_prefix(mol, chain_atoms: List[int], onium_idx: int) -> Optional[str]:
    """'{locant}-({cation}yl){stem}yl' for a chain of *chain_atoms* (ordered
    attach->far) terminating in a quaternary onium bonded to the LAST chain
    atom. E.g. choline -CH2CH2N+(CH3)3 -> '2-(trimethylazaniumyl)ethyl'."""
    if not chain_atoms:
        return None
    from ..assembly.substituent_naming import cation_to_prefix
    try:
        cat_prefix = cation_to_prefix(mol, onium_idx, chain_atoms[-1])
    except Exception:
        return None
    if not cat_prefix:
        return None
    n = len(chain_atoms)
    stem = get_chain_prefix(n)
    if not stem:
        return None
    # nesting order when the prefix carries its own marks (the Blue Book)
    from ..assembly.naming_utils import apply_enclosing_marks
    enclosed = apply_enclosing_marks(cat_prefix, -1)
    if n == 1:
        return f"{enclosed}{stem}yl"
    return f"{n}-{enclosed}{stem}yl"


def _try_charged_arm(mol, frag_atoms: Set[int], core_atom: int) -> Optional[str]:
    """Detect + name a cut arm that is a SIMPLE unbranched all-single-bond
    carbon chain from the core attachment point to exactly one quaternary
    onium cation (no free H, net +1). Returns the ALKYL substituent prefix,
    or None outside this narrow shape (any branching, ring, second
    heteroatom, or non-quaternary terminus fails closed)."""
    entry = None
    for n in mol.GetAtomWithIdx(core_atom).GetNeighbors():
        if n.GetIdx() in frag_atoms:
            entry = n.GetIdx()
            break
    if entry is None:
        return None

    chain: List[int] = []
    prev = core_atom
    cur = entry
    onium_idx = None
    visited_local: Set[int] = set()
    while True:
        if cur in visited_local:
            return None  # cycle -- not a simple chain
        visited_local.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        if atom.IsInRing():
            return None
        if atom.GetSymbol() == 'C':
            if atom.GetFormalCharge() != 0:
                return None
            chain.append(cur)
        elif (atom.GetSymbol() in _ONIUM_ELEMENTS and atom.GetFormalCharge() == 1
              and atom.GetTotalNumHs() == 0):
            onium_idx = cur
            break
        else:
            return None  # any other heteroatom/charge shape -- out of scope

        nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() != prev]
        if len(nbrs) != 1:
            return None  # branching along the chain -- out of scope
        bond = mol.GetBondBetweenAtoms(cur, nbrs[0])
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        prev, cur = cur, nbrs[0]

    if onium_idx is None or not chain:
        return None
    # The onium's own substituents (other than the chain) plus the chain
    # itself must exactly fill frag_atoms -- otherwise there is leftover
    # structure this narrow builder does not account for (fail closed).
    onium_others = {n.GetIdx() for n in mol.GetAtomWithIdx(onium_idx).GetNeighbors()
                    if n.GetIdx() != chain[-1]}
    accounted = set(chain) | {onium_idx} | onium_others
    for oi in onium_others:
        accounted |= _arm_atoms(mol, set(chain) | {onium_idx}, oi)
    if accounted != frag_atoms:
        return None
    return _charged_arm_alkyl_prefix(mol, chain, onium_idx)


def _alkyl_to_alkoxy(name: str) -> Optional[str]:
    """'...yl' -> '...oxy' (Part B's alkyl prefixes always end in 'yl' by
    construction, so this simple suffix swap is safe -- mirrors the same
    technique `fragment_assembly._alcohol_to_alkoxy` uses)."""
    if not name or not name.endswith('yl'):
        return None
    return name[:-2] + 'oxy'


# ---------------------------------------------------------------------------
# Arm prefix builders (ether, ester, phosphodiester)
# ---------------------------------------------------------------------------

def _ether_alkoxy_name(mol, core_atom: int, core_set: Set[int], o_idx: int) -> Optional[str]:
    frag = _arm_atoms(mol, core_set, o_idx)
    name = name_substituent(mol, sorted(frag), core_atom)
    if not name or name == 'substituent' or ' ' in name or 'unknown' in name.lower():
        return None
    return enclose_if_compound(name)


def _acyloxy_name(mol, carbonyl_c: int, ester_o: int) -> Optional[str]:
    from .. import rules  # noqa: F401 -- ensure package import order is safe
    from ..rules.lipids import _acyloxy_for_site
    ax = _acyloxy_for_site(mol, ('acyl', carbonyl_c, ester_o))
    if not ax or ' ' in ax:
        return None
    return enclose_if_compound(ax)


def _phosphoryloxy_name(mol, core_set: Set[int], p_idx: int, near_o_idx: int) -> Optional[str]:
    """'[(<head>)hydroxyphosphoryl]oxy' for a NEUTRAL, mono-protonated
    phosphodiester bridging the core (via near_o_idx-P) to a head group on
    P's other ester oxygen. Fails closed outside the neutral P(=O)(OH)(O-)(O-)
    tetrahedral shape (anionic/zwitterionic phosphates are out of v1 scope)."""
    p = mol.GetAtomWithIdx(p_idx)
    if p.GetSymbol() != 'P' or p.GetFormalCharge() != 0:
        return None
    dbl_o: List = []
    single_o: List = []
    for b in p.GetBonds():
        other = b.GetOtherAtom(p)
        if other.GetSymbol() != 'O':
            return None
        if b.GetBondType() == Chem.BondType.DOUBLE:
            dbl_o.append(other)
        elif b.GetBondType() == Chem.BondType.SINGLE:
            single_o.append(other)
        else:
            return None
    if len(dbl_o) != 1 or len(single_o) != 3:
        return None  # not a neutral tetrahedral phosphate diester

    others = [o for o in single_o if o.GetIdx() != near_o_idx]
    if len(others) != 2:
        return None
    free_oh = [o for o in others
               if o.GetDegree() == 1 and o.GetTotalNumHs() >= 1
               and o.GetFormalCharge() == 0]
    head_cands = [o for o in others
                  if o.GetIdx() not in {x.GetIdx() for x in free_oh}]
    if len(free_oh) != 1 or len(head_cands) != 1:
        return None
    head_o = head_cands[0]
    if head_o.GetFormalCharge() != 0 or head_o.GetDegree() != 2:
        return None
    head_c = next((n for n in head_o.GetNeighbors() if n.GetIdx() != p_idx), None)
    if head_c is None or head_c.GetSymbol() != 'C':
        return None

    p_neighborhood = core_set | {p_idx, near_o_idx, head_o.GetIdx(),
                                  dbl_o[0].GetIdx(), free_oh[0].GetIdx()}
    head_atoms = _arm_atoms(mol, p_neighborhood, head_c.GetIdx())
    frag = {head_o.GetIdx()} | head_atoms

    head_name = name_substituent(mol, sorted(frag), p_idx)
    if (not head_name or head_name == 'substituent'
            or 'unknown' in head_name.lower() or ' ' in head_name):
        # Part B: the head may be a simple charged (onium) arm that the
        # generic cascade cannot root at a charged atom.
        charged_alkyl = _try_charged_arm(mol, head_atoms, head_o.GetIdx())
        head_name = _alkyl_to_alkoxy(charged_alkyl) if charged_alkyl else None

    if not head_name or ' ' in head_name:
        return None
    # '[(2-aminoethoxy)hydroxyphosphoryl]oxy' (the Blue Book) -- the head is
    # enclosed in front of the 'hydroxyphosphoryl' unit and the marks ESCALATE
    # past any inside it,:7446): '[2-(methylamino)ethoxy]', never
    # '(2-(methylamino)ethoxy)' (TRIAGE row 65).
    from ..assembly.naming_utils import _is_fully_enclosed, apply_enclosing_marks
    head = head_name if _is_fully_enclosed(head_name) else apply_enclosing_marks(head_name, -1)
    return apply_enclosing_marks(f"{head}hydroxyphosphoryl", -1) + "oxy"


# ---------------------------------------------------------------------------
# Attachment classification
# ---------------------------------------------------------------------------

def _classify_attachment(mol, core_atom: int, core_set: Set[int], first_o,
                          acyl_carbons: Set[int]):
    """Returns ``('suffix', None)``, ``('prefix', rendered_str)`` or ``None``
    (decline -- an unrecognized shape directly on the core)."""
    if first_o.GetSymbol() != 'O' or first_o.GetFormalCharge() != 0:
        return None
    fi = first_o.GetIdx()

    # Free hydroxyl -> the -ol suffix.
    if first_o.GetTotalNumHs() >= 1 and first_o.GetDegree() == 1:
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in first_o.GetBonds()):
            return None
        return ('suffix', None)

    if first_o.GetDegree() != 2 or first_o.GetTotalNumHs() != 0:
        return None
    if not all(b.GetBondType() == Chem.BondType.SINGLE for b in first_o.GetBonds()):
        return None

    other = next((x for x in first_o.GetNeighbors() if x.GetIdx() != core_atom), None)
    if other is None:
        return None

    if other.GetSymbol() == 'C':
        if other.GetIdx() in acyl_carbons:
            name = _acyloxy_name(mol, other.GetIdx(), fi)
        else:
            name = _ether_alkoxy_name(mol, core_atom, core_set, fi)
        if name is None:
            return None
        return ('prefix', name)

    if other.GetSymbol() == 'P':
        name = _phosphoryloxy_name(mol, core_set, other.GetIdx(), fi)
        if name is None:
            return None
        return ('prefix', name)

    return None  # any other heavy neighbour shape -- out of v1 scope


# ---------------------------------------------------------------------------
# Numbering + assembly
# ---------------------------------------------------------------------------

def _number_parent(parent: List[int], suffix_carbons: Set[int],
                    prefix_on: Dict[int, List[str]]) -> Optional[Dict[int, int]]:
    def score(order):
        pos = {a: i + 1 for i, a in enumerate(order)}
        suf = sorted(pos[a] for a in suffix_carbons)
        pre = sorted(pos[a] for a in prefix_on)
        # (g) (the Blue Book): then "lowest locants for the
        # substituent cited first as a prefix in the name" (and so on down the
        # citation order) -- without it a tie kept the input's chain direction.
        alpha = sorted((alpha_sort_key(t), pos[a])
                       for a, toks in prefix_on.items() for t in toks)
        return (suf, pre, alpha)
    fwd = parent
    rev = list(reversed(parent))
    best = min((fwd, rev), key=score)
    return {a: i + 1 for i, a in enumerate(best)}


def _assemble(mol, parent: List[int], numbering: Dict[int, int],
              suffix_carbons: Set[int], prefix_on: Dict[int, List[str]]) -> Optional[str]:
    n = len(parent)
    stem = get_chain_prefix(n)
    if not stem:
        return None

    groups: Dict[str, List[int]] = {}
    for c, toks in prefix_on.items():
        for t in toks:
            groups.setdefault(t, []).append(numbering[c])
    parts: List[str] = []
    from ..assembly.naming_utils import _is_fully_enclosed, apply_enclosing_marks
    for tok in sorted(groups, key=alpha_sort_key):
        locs = sorted(groups[tok])
        compound = tok[:1] in '([{'
        # A composed prefix whose marks do not cover it whole ('[(...)
        # hydroxyphosphoryl]oxy') is enclosed as one unit when cited, the marks
        # escalated past those inside /, the Blue Book):
        # '3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}' (:55180), never the bare
        # '3-[(...)hydroxyphosphoryl]oxy-2-...'.
        if compound and not _is_fully_enclosed(tok):
            tok_cited = apply_enclosing_marks(tok, -1)
        else:
            tok_cited = tok
        table = COMPLEX_MULTIPLIERS if compound else SIMPLE_MULTIPLIERS
        mult = '' if len(locs) == 1 else table.get(len(locs))
        if mult is None:
            return None
        loc_str = ','.join(str(x) for x in locs)
        parts.append(f"{loc_str}-{mult}{tok_cited}")
    prefix_str = '-'.join(parts)

    if suffix_carbons:
        ol_locs = sorted(numbering[c] for c in suffix_carbons)
        ol_mult = _HYDRO_MULT.get(len(ol_locs))
        if ol_mult is None:
            return None
        loc_str = ','.join(str(x) for x in ol_locs)
        suffix = f"-{loc_str}-{ol_mult}ol"
    else:
        suffix = ""

    # (a) (the Blue Book): the parent's final 'e' is elided before a suffix
    # beginning with a vowel -- 'propan-1-ol', but 'propane-1,2-diol' (the
    # multiplier keeps it). The weave used to write 'propane-1-ol'/'propane-2-ol'.
    base = f"{stem}an" if suffix and not ol_mult else f"{stem}ane"
    name = f"{prefix_str}{base}{suffix}" if prefix_str else f"{base}{suffix}"
    name = name.replace('--', '-').lstrip('-')

    # Core stereocentre(s): the acyl/ether/phospho arms' OWN internal
    # stereo is already embedded in their returned strings (name_substituent /
    # _acyloxy_for_site both handle it); only the CORE's own locants need a
    # descriptor prepended here.
    try:
        from ..rules.stereochemistry import (
            collect_stereodescriptors,
            format_stereodescriptor_string,
        )
        descriptors = collect_stereodescriptors(mol, numbering, include_near_parent_ez=False)
        if descriptors:
            name = format_stereodescriptor_string(descriptors) + name
    except Exception:
        pass

    return name


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _try_weave_impl(mol, style: str) -> Optional[str]:
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None  # v1: acyclic core only (a ring hub, e.g. GPI, declines)
    if len(Chem.GetMolFrags(mol)) != 1:
        return None  # single connected component only

    candidates = _select_core_candidates(mol)
    if not candidates:
        return None
    acyl_carbons = {a.GetIdx() for a in mol.GetAtoms() if _is_ester_carbonyl(a)}

    for core_set in candidates:
        name = _build_from_core(mol, core_set, acyl_carbons)
        if name:
            return name
    return None


def _build_from_core(mol, core_set: Set[int], acyl_carbons: Set[int]) -> Optional[str]:
    """Attempt the full classify + number + assemble pipeline treating
    *core_set* as the parent chain. Returns None (decline) on ANY shape this
    composer does not recognize, so the caller can try the next candidate."""
    adj: Dict[int, List[int]] = {c: [] for c in core_set}
    for c in core_set:
        for n in mol.GetAtomWithIdx(c).GetNeighbors():
            ni = n.GetIdx()
            if ni in core_set:
                bond = mol.GetBondBetweenAtoms(c, ni)
                if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                    return None  # unsaturated core -- out of v1 scope
                adj[c].append(ni)
    core_order = _longest_path(adj, core_set)
    if core_order is None or len(core_order) != len(core_set):
        return None  # branched core -- out of v1 scope

    prefix_on: Dict[int, List[str]] = {}
    suffix_carbons: Set[int] = set()
    acyloxy_arm = False
    phosphoryloxy_arm = False

    for c in core_order:
        ca = mol.GetAtomWithIdx(c)
        for n in ca.GetNeighbors():
            ni = n.GetIdx()
            if n.GetAtomicNum() <= 1 or ni in core_set:
                continue
            verdict = _classify_attachment(mol, c, core_set, n, acyl_carbons)
            if verdict is None:
                return None  # unrecognized shape directly on the core
            kind, payload = verdict
            if kind == 'suffix':
                suffix_carbons.add(c)
            else:
                prefix_on.setdefault(c, []).append(payload)
                far = [o for o in n.GetNeighbors() if o.GetIdx() != c]
                if any(o.GetIdx() in acyl_carbons for o in far):
                    acyloxy_arm = True
                if any(o.GetSymbol() == 'P' for o in far):
                    phosphoryloxy_arm = True

    if not prefix_on:
        return None  # no linkage arm -- not this composer's job (a bare
                     # polyol is already named correctly by the ordinary
                     # composer without decomposition)

    numbering = _number_parent(core_order, suffix_carbons, prefix_on)
    if numbering is None:
        return None

    name = _assemble(mol, core_order, numbering, suffix_carbons, prefix_on)
    if name and acyloxy_arm and not phosphoryloxy_arm:
        # An ester of the core polyol cited as an 'acyloxy' prefix on the bare
        # hydride (or beside the junior '-ol' suffix) is not the PIN.
        # (the Blue Book) ranks esters (class 9,:18182) above alcohols
        # (17,:18190) and ethers; (:31698) cites an ester as a
        # prefix only "when [...] another group is present that has priority for
        # citation as the principal group or when all ester groups cannot be
        # described by the methods prescribed for naming esters", and a polyol
        # ester has such a method: (:31836) "Method (1) generates
        # preferred IUPAC names" ('propane-1,2,3-triyl 1,2-diacetate
        # 3-propanoate (PIN)',:31840). The name is valid and round-trip verified,
        # so it still ships; the name-scoped record keeps it off pin_verified
        # (the twin, polyfunctional._record_el02_ester_prefix_name).
        # A phosphoryloxy arm (a phosphoric acid diester with its own acidic
        # OH) is a different seniority question and is left as it was.
        # TRIAGE g3 C10b.
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return name


# ---------------------------------------------------------------------------
# Part C: atom-coverage / 0-wrong verification (the caller MUST use this)
# ---------------------------------------------------------------------------

def weave_is_verified(mol, name: str) -> bool:
    """True iff *name* denotes EXACTLY *mol* (full OPSIN round-trip, same
    heavy-atom count AND matching InChI) -- the atom-complete-or-abstain
    guard. Fails CLOSED (False) on any error/unavailable-OPSIN, so an
    unverifiable weave candidate is never shipped."""
    if not name:
        return False
    try:
        from ..validation.opsin_roundtrip import opsin_parse
        opsin_smiles = opsin_parse(name)
        if not opsin_smiles:
            return False
        parsed = Chem.MolFromSmiles(opsin_smiles)
        if parsed is None:
            return False
        if parsed.GetNumHeavyAtoms() != mol.GetNumHeavyAtoms():
            return False
        from rdkit.Chem.inchi import MolToInchi
        inchi_in = MolToInchi(mol)
        inchi_out = MolToInchi(parsed)
        return bool(inchi_in) and inchi_in == inchi_out
    except Exception:
        return False
