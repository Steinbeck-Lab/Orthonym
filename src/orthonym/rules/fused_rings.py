"""
Fused ring system detection and naming.

Handles:
- Classification of fused ring systems (ortho-fused, ortho-peri-fused, bridged-fused)
- Naming of fused heterocycles with retained names priority
- Substituent detection and locant assignment for fused systems
- N-substitution handling for fused heterocycles

IUPAC 2013 Rules for fused systems:
- Ortho-fused: rings share exactly one bond (2 atoms)
- Ortho-peri-fused: at least one ring shares atoms with 3+ other rings
- ALWAYS check retained names FIRST before systematic naming
- Tautomer locants (1H-, 2H-, 9H-) must be preserved in names
- N-substitution uses N-locant format (N-methyl, not 1-methyl)

Reference: IUPAC 2013 Blue Book, Section (Fused Ring Systems)
"""

import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..assembly.naming_utils import (
    alpha_sort_key,
    get_alkyl_name,
)
from ..data import get_retained_name as _get_global_retained_name
from ..data.fused_heterocycles import (
    get_fused_heterocycle_name,
    match_fused_heterocycle_core,
)
from ..data.xanthine_derivatives import (
    get_xanthine_name,
)
from ..perception.rings import (
    get_ring_systems,
    is_aromatic_ring,
    is_heterocyclic,
)
from ..perception.stereo import assign_stereochemistry
from .stereochemistry import (
    collect_ring_junction_stereo,
    determine_simple_cis_trans,
    format_ring_junction_stereo,
    get_bridgehead_atoms,
    get_junction_locants_for_fused_system,
)
from ..perception.smarts_cache import compiled as _compiled_smarts

# Simple multiplicative prefixes for substituent naming
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def _compute_general_indicated_h(mol, ring_atom_set: Set[int],
                                  atom_to_locant: Dict[int, Any]) -> List:
    """Compute indicated hydrogen for non-retained fused systems.

    Per IUPAC: In a ring system with maximum non-cumulative double bonds,
    indicated hydrogen marks positions where an 'extra' hydrogen is present
    (the atom is saturated in the actual molecule but would be unsaturated
    in the ideal parent).

    Algorithm:
    1. For each ring atom, compute expected H count in maximally unsaturated parent
    2. Compare with actual H count in molecule
    3. If actual > expected at a tautomeric position, record as indicated H

    Args:
        mol: RDKit Mol object.
        ring_atom_set: Set of atom indices in the fused ring system.
        atom_to_locant: Mapping from atom index to IUPAC locant.

    Returns:
        List of locants for indicated hydrogen positions, sorted.
    """
    import logging
    logger = logging.getLogger(__name__)

    indicated = []
    for idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol not in ('N', 'C', 'O', 'S'):
            continue

        # Only consider atoms that are NOT aromatic in the molecule
        # but are ring atoms (potential tautomeric sites)
        if atom.GetIsAromatic():
            continue

        actual_hs = atom.GetTotalNumHs()

        # Expected Hs in maximally unsaturated parent:
        # - Aromatic C with 2 ring bonds: 1H (like benzene C)
        # - Aromatic C with 3+ ring bonds (fusion junction): 0H
        # - Aromatic N in 6-membered ring (pyridine-type): 0H
        # - Aromatic N in 5-membered ring (pyrrole-type): 1H (indicated H)
        # - O, S in ring: 0H expected
        ring_bond_count = sum(
            1 for bond in atom.GetBonds()
            if bond.GetOtherAtom(atom).GetIdx() in ring_atom_set
        )
        expected = 0
        if symbol == 'C':
            expected = 1 if ring_bond_count <= 2 else 0
        elif symbol == 'N':
            expected = 0  # In maximally unsaturated parent, N donates lone pair

        if actual_hs > expected:
            locant = atom_to_locant.get(idx)
            if locant is not None:
                indicated.append(locant)
            else:
                logger.debug(
                    "Indicated H atom %d (%s) has no locant mapping",
                    idx, symbol,
                )

    # Sort: numeric locants first, then string locants
    def _sort_key(loc):
        if isinstance(loc, int):
            return (0, loc, '')
        return (1, 0, str(loc))

    return sorted(indicated, key=_sort_key)


def _exocyclic_atoms_accounted(mol, core_atoms: Set[int]) -> bool:
    """Source-level completeness check for algorithmic-path substituent naming.

    ``get_fused_heterocycle_substituents`` silently ``continue``s past any
    exocyclic branch it cannot identify (``sub_info is None``), so a name
    assembled from its output can be MISSING a substituent — i.e. denote a
    DIFFERENT molecule. The catalog path leans on the downstream
    round-trip gate to catch that; the fix methodology requires failing closed
    at the source. This verifies that every exocyclic heavy atom is consumed by
    exactly the set of substituents ``get_fused_heterocycle_substituents`` will
    discover (it re-runs the same ``_identify_fused_substituent`` traversal, which
    is read-only/idempotent). Returns False on any unidentifiable branch or any
    heavy atom left unaccounted, so the caller returns None.
    """
    exocyclic = {
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetIdx() not in core_atoms and a.GetAtomicNum() > 1
    }
    if not exocyclic:
        return True
    accounted: Set[int] = set()
    for core_atom_idx in core_atoms:
        core_atom = mol.GetAtomWithIdx(core_atom_idx)
        for neighbor in core_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in core_atoms:
                continue
            sub_info = _identify_fused_substituent(mol, nbr_idx, core_atoms)
            if sub_info is None:
                return False  # unnameable branch -> fail closed
            accounted.update(
                a for a in sub_info.get('atoms', [])
                if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
            )
    return accounted == exocyclic


def _mancude_indicated_h_atoms(mol, ring_atom_set: Set[int]) -> Optional[List[int]]:
    """Ring atoms that carry the indicated hydrogen of a mancude fused system,
    read from the structure; ``None`` when this cannot be established.

     (the Blue Book): "noncumulative double bonds are
    introduced into the completed fused system. Hydrogen atoms not attached to
    atoms connected by double bonds are denoted as indicated hydrogen atom(s)."
     (:14607): "In preferred IUPAC names, all indicated hydrogen
    atoms must be cited when the names are constructed in accordance with the
    principles of fusion nomenclature",:3557). RDKit perceives a
    pyrrole-type N-H of such a system as AROMATIC, which the legacy counter
    (:func:`_compute_general_indicated_h`) skips, so 'thieno[2,3-c]pyrrole'
    shipped without its '5H-'.

    The candidate set F is every C/N ring atom with NO double bond in the
    Kekule structure (an N-H, an N-substituted pyrrole-type N, a saturated
    C). F is the indicated-hydrogen set only when it is the unmatched set of
    a MAXIMUM matching of the mancude parent: the input's own
    Kekule structure is a perfect matching of the rest, and no smaller
    unmatched set may exist -- otherwise the extra saturation is hydro
    ; pyrrolo[3,2-b]pyrrole (PIN):14571 has no indicated
    hydrogen, so its N,N'-dihydro form is a hydro name) and this returns
    ``None``. Also ``None`` for charged or other-element ring atoms and for
    an exocyclic double bond on a ring atom (the 'added indicated hydrogen'
    form,:3725, is not built here), so those keep the existing
    behaviour.
    """
    ring = set(ring_atom_set)
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
        if atom.GetSymbol() not in ('C', 'N', 'O', 'S', 'Se', 'Te'):
            return None
    try:
        km = Chem.Mol(mol)
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return None
    eligible: List[int] = []
    flagged: List[int] = []
    for idx in sorted(ring):
        atom = km.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        has_double = False
        for bond in atom.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                if bond.GetOtherAtomIdx(idx) not in ring:
                    return None  # exocyclic =X: added hydrogen, not built here
                if bond.GetBondType() != Chem.BondType.DOUBLE:
                    return None
                has_double = True
        ring_degree = sum(1 for nb in atom.GetNeighbors() if nb.GetIdx() in ring)
        if sym == 'C' or (sym == 'N' and ring_degree == 2):
            eligible.append(idx)
            if not has_double:
                flagged.append(idx)
        elif sym == 'N' and has_double:
            return None  # a fusion N with a double bond is charged or odd
    if not flagged:
        return None
    adj = {
        a: [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
            if nb.GetIdx() in eligible]
        for a in eligible
    }
    minimum_sets = _saturation_indicated_h_sets(eligible, adj)
    if frozenset(flagged) not in minimum_sets:
        return None
    return flagged


def _mancude_indicated_h_locants(mol, ring_atom_set: Set[int],
                                 atom_to_locant: Dict[int, Any],
                                 has_substituents: bool) -> List:
    """Locants of:func:`_mancude_indicated_h_atoms`, or ```` (no change).

    The locants come from the peripheral numbering
    (``compute_fused_numbering``), whose cascade already puts an N-H at the
    lowest locant the ring criteria allow,:24641). A bare system
    reads them from that engine directly -- its legacy descriptor-order map
    (``atom_to_locant`` here) does not reproduce the fused numbering (it gives
    the N of thieno[2,3-c]pyrrole locant 1, the engine and OPSIN give 5). A
    substituted system already numbers from the engine, but its
    tie-break may pick a ring automorphism; the locants are accepted only
    when every automorphism gives the indicated hydrogen the same locants, so
    the tie-break cannot trade an indicated-hydrogen locant (b),
    :3246, is cited before the suffix and prefix criteria) for a substituent
    one. Only integer locants are cited; anything
    else keeps the existing name (````).
    """
    atoms = _mancude_indicated_h_atoms(mol, ring_atom_set)
    if not atoms:
        return []
    from .fusion_numbering import compute_fused_numbering
    canonical = compute_fused_numbering(mol, ring_atom_set)
    if not canonical:
        return []
    try:
        engine_map = {a: _fused_locant_to_output(canonical[a]) for a in atoms}
    except KeyError:
        return []
    if has_substituents:
        chosen = [atom_to_locant.get(a) for a in atoms]
        if sorted(map(str, chosen)) != sorted(map(str, engine_map.values())):
            return []
        for perm in _ring_system_automorphisms(mol, ring_atom_set):
            try:
                img = sorted(str(_fused_locant_to_output(canonical[perm[a]]))
                             for a in atoms)
            except KeyError:
                return []
            if img != sorted(map(str, chosen)):
                return []
        locs = chosen
    else:
        locs = [engine_map[a] for a in atoms]
    if not all(isinstance(loc, int) for loc in locs):
        return []
    return sorted(locs)


def _fused_locant_to_output(loc):
    """compute_fused_numbering _Locant -> the int/str form the substituent
    machinery consumes: ``int`` stays int; ``(3, 'a')`` -> ``'3a'``."""
    if isinstance(loc, tuple):
        return f"{loc[0]}{loc[1]}"
    return loc


def _fused_locant_num(loc):
    """Numeric comparison key for a locant (int, ``'3a'`` str, or ``(3,'a')``
    tuple). Fusion-letter locants sort just after their integer."""
    if isinstance(loc, tuple):
        return (loc[0], 1)
    if isinstance(loc, str) and loc and loc[-1].isalpha():
        return (int(loc[:-1]), 1)
    return (int(loc), 0)


def _fused_substituent_citation_key(substituents: Dict):
    """ low-locant tie-break key ``(suffix_locants, prefix_locants)``.

    Suffix-forming groups (the ``-one``/``-amine``/``-carboxylic acid`` family)
    are assigned low locants BEFORE detachable prefixes, line 25354);
    among prefixes, low locants together (line 25503). N-substituents carry no
    ring locant and are excluded from the tie-break.
    """
    suffix = []
    suffix += list(substituents.get('oxo_substituents', []))
    suffix += list(substituents.get('amino_substituents', []))
    for locs in substituents.get('suffix_groups', {}).values():
        suffix += list(locs)
    prefix = []
    for locs in substituents.get('c_substituents', {}).values():
        prefix += list(locs)
    for other in substituents.get('other', []):
        if 'locant' in other:
            prefix.append(other['locant'])
    return (
        tuple(sorted(_fused_locant_num(l) for l in suffix)),
        tuple(sorted(_fused_locant_num(l) for l in prefix)),
    )


def _ring_system_automorphisms(
    mol, ring_atoms: Set[int], ignore_bond_order: bool = False,
) -> List[Dict[int, int]]:
    """Automorphisms of the BARE ring skeleton (substituents stripped), each a
    ``{orig_atom -> orig_atom}`` permutation.

    Once the ring criteria fix the numbering, the only residual
    freedom is the ring system's own symmetry (e.g. furo[2,3-b]furan is
    symmetric: positions 2 and 5 are equivalent). (substituents) /
     (hydro) break that tie by lowest locants, so the caller
    enumerates these permutations.

    ``ignore_bond_order=True`` matches on element + topology only (all bonds
    flattened to single), i.e. the symmetry of the MANCUDE parent — needed for
    the hydro tie-break, where the saturated and unsaturated rings of a
    symmetric parent (2,3- vs 5,6-dihydrofuro[3,2-b]furan) must be treated as
    interchangeable so the PIN takes the lower hydro locants.

    Fail-safe: returns just the identity when the skeleton cannot be isolated
    or sanitised — then no tie-break is applied and the canonical numbering
    stands (still round-trips via).
    """
    ring_atoms = set(ring_atoms)
    identity = {a: a for a in ring_atoms}
    bond_idx = [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms
    ]
    amap: Dict[int, int] = {}
    try:
        sub = Chem.PathToSubmol(mol, bond_idx, atomMap=amap)
        if ignore_bond_order:
            sub = Chem.RWMol(sub)
            for b in sub.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
            for a in sub.GetAtoms():
                a.SetIsAromatic(False)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                a.SetFormalCharge(0)
            sub = sub.GetMol()
            Chem.SanitizeMol(
                sub,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            )
        else:
            Chem.SanitizeMol(sub)
    except Exception:
        return [identity]
    inv = {v: k for k, v in amap.items()}  # submol idx -> original idx
    matches = sub.GetSubstructMatches(sub, uniquify=False, maxMatches=64)
    if not matches:
        return [identity]
    perms: List[Dict[int, int]] = []
    for mt in matches:
        try:
            perms.append({inv[i]: inv[img] for i, img in enumerate(mt)})
        except KeyError:
            continue
    return perms or [identity]


def _select_substituent_numbering(mol, ring_atom_set: Set[int]):
    """Choose the peripheral numbering for a SUBSTITUTED 2-component ortho-fused
    mancude ring system and discover its substituents against it.

    Returns ``(atom_to_locant, substituents)`` or ``None`` (fail closed).

    The ring numbering comes from the deterministic engine
    (``compute_fused_numbering``); the legacy descriptor-order map does not
    reproduce OPSIN's canonical numbering for substituent placement. The
     substituent tie-break is applied over the ring-system
    automorphisms, then a SMILES-order-independent canonical-rank signature
    guarantees a single deterministic representative.
    """
    from .fusion_numbering import compute_fused_numbering
    canonical = compute_fused_numbering(mol, ring_atom_set)
    if not canonical:
        return None  # engine declined -> cannot trust locants -> fail closed
    perms = _ring_system_automorphisms(mol, ring_atom_set)
    canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    atoms_by_rank = sorted(ring_atom_set, key=lambda a: canon_rank[a])

    best = None  # (citation_key, signature, cand_map, substituents)
    for perm in perms:
        try:
            cand = {
                a: _fused_locant_to_output(canonical[perm[a]])
                for a in ring_atom_set
            }
        except KeyError:
            continue
        subs = get_fused_heterocycle_substituents(mol, cand)
        key = _fused_substituent_citation_key(subs)
        sig = tuple(_fused_locant_num(cand[a]) for a in atoms_by_rank)
        if best is None or (key, sig) < (best[0], best[1]):
            best = (key, sig, cand, subs)
    if best is None:
        return None
    return best[2], best[3]


def _ring_double_bond(atom, ring_set) -> bool:
    """True iff ``atom`` has a ring double bond to another ring atom."""
    return any(
        b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetOtherAtom(atom).GetIdx() in ring_set
        for b in atom.GetBonds()
    )


def _perfect_matching(nodes: frozenset, adj) -> bool:
    """True iff the induced subgraph on ``nodes`` has a perfect matching.
    Recursive; ``nodes`` is the tiny saturated-carbon set of a 2-ring system."""
    if not nodes:
        return True
    v = min(nodes)
    for w in adj[v]:
        if w in nodes and _perfect_matching(nodes - {v, w}, adj):
            return True
    return False


def _saturation_indicated_h_sets(sat_atoms, adj):
    """: put the MAX number of noncumulative double bonds into the
    saturated region (a maximum matching of the saturated-carbon subgraph). The
    atoms left UNMATCHED are indicated hydrogen; each matched pair is one unit of
    hydro. Returns every minimum-size unmatched-atom set (frozensets) so the
    caller can pick the one giving the lowest indicated-H locants;
    ```` if the subgraph admits no matching at the required parity."""
    from itertools import combinations
    n = len(sat_atoms)
    base = frozenset(sat_atoms)
    for u in range(n % 2, n + 1, 2):          # u has the parity of n
        good = [frozenset(S) for S in combinations(sat_atoms, u)
                if _perfect_matching(base - set(S), adj)]
        if good:
            return good                        # first non-empty = min unmatched = max matching
    return []


def _perfect_matching_pairs(nodes: frozenset, adj) -> Optional[List[Tuple[int, int]]]:
    """One perfect matching of the induced subgraph on ``nodes`` as a list of
    pairs (deterministic: lowest atom first), or ``None`` when there is none."""
    if not nodes:
        return []
    v = min(nodes)
    for w in sorted(adj[v]):
        if w in nodes:
            rest = _perfect_matching_pairs(nodes - {v, w}, adj)
            if rest is not None:
                return [(v, w)] + rest
    return None


def _catalogued_mancude_parent(mol, ring_atom_set: Set[int], hydro_atoms,
                               adj) -> Optional[Tuple[str, Dict[int, Any]]]:
    """ (the Blue Book): the hydro prefixes of a partially
    saturated fused system express the hydrogenation of its MANCUDE parent ring
    system, so the parent is named as the mancude system itself is named.

    Builds the mancude twin of the ring system -- the input's ring skeleton with
    one double bond put back on each pair of a perfect matching of the hydro atoms
    (every other ring atom, its hydrogen and its charge kept) -- and names it with
    the fused ring catalogue (retained names,, and the benzo names of
    : quinoline, 1H-indole, 2-benzofuran, 1,8-naphthyridine,...).

    Returns ``(parent name, {input atom -> locant})`` or ``None`` (no perfect
    matching, a ring atom with an exocyclic double bond, the twin does not
    sanitize, or the catalogue has no entry for it)."""
    pairs = _perfect_matching_pairs(frozenset(hydro_atoms), adj)
    if not pairs:
        return None
    ring_atom_set = set(ring_atom_set)
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        bonds = [b.GetIdx() for b in kek.GetBonds()
                 if b.GetBeginAtomIdx() in ring_atom_set
                 and b.GetEndAtomIdx() in ring_atom_set]
        amap: Dict[int, int] = {}
        twin = Chem.RWMol(Chem.PathToSubmol(kek, bonds, atomMap=amap))
        if set(amap) != ring_atom_set:
            return None
        for orig, idx in amap.items():
            src = kek.GetAtomWithIdx(orig)
            exo = 0
            for b in src.GetBonds():
                if b.GetOtherAtomIdx(orig) in ring_atom_set:
                    continue
                if b.GetBondType() != Chem.BondType.SINGLE:
                    return None            # exocyclic =X: added-hydrogen territory
                exo += 1
            at = twin.GetAtomWithIdx(idx)
            at.SetNoImplicit(True)
            at.SetNumExplicitHs(int(src.GetTotalNumHs()) + exo)
        for a, b in pairs:
            twin.GetBondBetweenAtoms(amap[a], amap[b]).SetBondType(Chem.BondType.DOUBLE)
            for x in (a, b):
                at = twin.GetAtomWithIdx(amap[x])
                if at.GetNumExplicitHs() < 1:
                    return None
                at.SetNumExplicitHs(at.GetNumExplicitHs() - 1)
        twin_mol = twin.GetMol()
        Chem.SanitizeMol(twin_mol)
    except Exception:
        return None
    core = match_fused_heterocycle_core(twin_mol)
    if core is None:
        return None
    name, twin_locants, _key = core
    inv = {idx: orig for orig, idx in amap.items()}
    numbering = {inv[i]: loc for i, loc in twin_locants.items() if i in inv}
    if set(numbering) != ring_atom_set:
        return None
    return name, numbering


_LEADING_IH = re.compile(r"^((?:\d+[a-z]?H)(?:,\d+[a-z]?H)*)-")


def _leading_indicated_h(name: str) -> Set[str]:
    """The indicated-hydrogen locants a catalogue name starts with
    ('1H-benzimidazole' -> {'1'}; 'quinoline' -> set)."""
    m = _LEADING_IH.match(name)
    if not m:
        return set()
    return {part[:-1] for part in m.group(1).split(',')}


def _unsaturated_part_indicated_h(mol, ring_atom_set: Set[int], sat: Set[int]):
    """The ring atoms outside the saturated region that hold indicated hydrogen of
    the mancude parent: no double bond in a Kekule structure of the input (a
    pyrrole-type N, with H or a substituent). A divalent ring chalcogen (the O of
    furan) is no indicated-hydrogen position. ``None`` when the input cannot be
    kekulized (fail closed)."""
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    out = set()
    for a in ring_atom_set:
        if a in sat:
            continue
        at = kek.GetAtomWithIdx(a)
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in at.GetBonds()):
            continue
        ring_deg = sum(1 for b in at.GetBonds() if b.GetOtherAtomIdx(a) in ring_atom_set)
        if at.GetSymbol() in ('O', 'S', 'Se', 'Te') and ring_deg == 2:
            continue
        if at.GetSymbol() == 'N' and ring_deg == 3:
            continue           # a fusion N (indolizine) holds no hydrogen
        out.add(a)
    return frozenset(out)


def _mancude_indicated_h_count(mol, ring_atom_set: Set[int]) -> int:
    """How many indicated hydrogen atoms the mancude form of this ring skeleton
    takes: the ring atoms that can carry a ring double bond (C, and N, P, As, B with
    two ring bonds), minus twice a maximum matching of them. Pyrrole 1, furan 0,
    cyclopenta[c]pyridine 1, indolizine 0.

    A ring atom with a nonstandard bonding number, two ring bonds and no exocyclic
    multiple bond can carry a ring double bond too: '1H-1λ4-thiophene (PIN)' and
    '3H-1λ4-thiophene (PIN)', the Blue Book,:9167,:9171) and
    '2H-5λ4-dibenzo[b,d]thiophene (PIN)',:14531,:14555) each take one
    indicated hydrogen."""
    from .lambda_convention import nonstandard_bonding_number
    ring_atom_set = set(ring_atom_set)

    def _eligible(idx):
        at = mol.GetAtomWithIdx(idx)
        deg = sum(1 for b in at.GetBonds() if b.GetOtherAtomIdx(idx) in ring_atom_set)
        sym = at.GetSymbol()
        if sym == 'C':
            return deg <= 3
        if sym in ('N', 'P', 'As', 'B') and deg == 2:
            return True
        return (deg == 2 and nonstandard_bonding_number(mol, idx) is not None
                and not any(b.GetBondTypeAsDouble() >= 2.0
                            and b.GetOtherAtomIdx(idx) not in ring_atom_set
                            for b in at.GetBonds()))

    nodes = frozenset(a for a in ring_atom_set if _eligible(a))
    adj = {a: {n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
               if n.GetIdx() in nodes} for a in nodes}
    memo: Dict[frozenset, int] = {}

    def _max_matching(rest: frozenset) -> int:
        if len(rest) < 2:
            return 0
        if rest in memo:
            return memo[rest]
        v = min(rest)
        best = _max_matching(rest - {v})          # v unmatched
        for w in adj[v]:
            if w in rest:
                best = max(best, 1 + _max_matching(rest - {v, w}))
        memo[rest] = best
        return best

    return len(nodes) - 2 * _max_matching(nodes)


_AZOLE_COMPONENT_RE = re.compile(r"(?:isoxazol|isothiazol|isoselenazol|oxazol|thiazol|selenazol)(?:o\[|e\b)")


def _unbracketed_azole_component(name: str) -> bool:
    """True when a fusion name spells an oxazole, thiazole or selenazole component
    without its Hantzsch-Widman locants: (the Blue Book) "The names
    'isothiazole', 'isoxazole', 'thiazole', and 'oxazole', although permitted in general
    nomenclature, are not recommended for the names of components in preferred IUPAC
    fusion names. The Hantzsch-Widman names 1,2-thiazole, 1,2-oxazole, 1,3- thiazole,
    and 1,3-oxazole, respectively, must be used; the locants are enclosed in square
    brackets in the completed fusion name" ('[1,3]selenazolo[5,4-d][1,3]thiazole (PIN)',
    :12311). A component starts the name or follows a hyphen or a fusion bracket; one
    inside a benzo name ('1,3-benzothiazole') follows a letter and is not a component
    of this kind, and one after its locant bracket ('[1,3]thiazolo') is the PIN form."""
    for m in _AZOLE_COMPONENT_RE.finditer(name or ""):
        before = name[:m.start()]
        if before and before[-1].isalpha():
            continue                  # inside a benzo name or another word
        if re.search(r"\[\d+,\d+\]$", before):
            continue                  # '[1,3]thiazolo', '[1,2]oxazole'
        return True
    return False


def _record_non_pin_spelling(fragment: str) -> None:
    """Record a name part this module built that is valid but never a PIN
    (name-scoped: only a shipped name that carries it is labelled below PIN)."""
    try:
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(fragment)
    except Exception:  # a label record must never break naming
        pass


def _first_cited_prefix_key(subs) -> tuple:
    """ (g) (the Blue Book): the locants of the detachable prefixes taken
    in alphanumerical citation order, so the prefix cited first gets the lowest."""
    if not subs:
        return ()
    by_name: Dict[str, List[Any]] = defaultdict(list)
    for name, locs in (subs.get('c_substituents') or {}).items():
        by_name[name].extend(locs)
    for other in subs.get('other') or ():
        if 'locant' in other:
            by_name[str(other.get('name', ''))].append(other['locant'])
    return tuple(tuple(sorted(_fused_locant_num(l) for l in by_name[n]))
                 for n in sorted(by_name, key=lambda n: n.lower()))


def _substituent_placement(subs) -> tuple:
    """A comparable record of which substituent sits on which locant (None -> )."""
    if not subs:
        return ()
    out = []
    for kind in ('c_substituents', 'suffix_groups'):
        for name, locs in sorted((subs.get(kind) or {}).items()):
            out.append((kind, name, tuple(sorted(str(l) for l in locs))))
    for kind in ('oxo_substituents', 'amino_substituents'):
        out.append((kind, tuple(sorted(str(l) for l in (subs.get(kind) or ())))))
    for other in subs.get('other') or ():
        out.append(('other', tuple(sorted((str(k), str(v)) for k, v in other.items()))))
    for kind in sorted(k for k in subs if k not in (
            'c_substituents', 'suffix_groups', 'oxo_substituents',
            'amino_substituents', 'other')):
        out.append((kind, repr(subs[kind])))
    return tuple(sorted(out, key=repr))


def _try_partial_saturation_name(mol, ring_atom_set: Set[int], mancude_parent: str):
    """ a phase (full): name a partially-saturated 2-component ortho-fused
    mancude system as ``<hydro>-<indicatedH>-<mancude parent>`` /
     /, bare or prefix-substituted.

    Returns the name or ``None`` (fail closed). On ``None`` the caller keeps its
    legacy indicated-H behaviour, so this can only ADD coverage or UPGRADE a
    round-trip-valid non-PIN name to the PIN — never regress.

    Scope (everything else fails closed):
      * saturated positions are sp3 ring CARBONS; a benign divalent ring
        chalcogen (O/S/Se/Te, 2 ring bonds, 0 H, neutral, no ring double bond)
        is an inherent ring atom, not a hydro position (unlocks the saturated
        S/Se ring). A genuinely saturated sp3 NITROGEN or a charged/hypervalent
        ring atom interacts with pyrrole-type indicated hydrogen -> deferred.
      * The maximum number of noncumulative double bonds is placed into the
        saturated region (a maximum matching of the saturated-carbon subgraph,
        ; the UNMATCHED carbons are indicated hydrogen, each matched
        pair is one unit of hydro. This covers even counts (0 indicated H,
        BYTE-IDENTICAL to the former slice) and odd counts (indicated-H + hydro
        mix, e.g. ``5,6-dihydro-4H-...``).
      * Prefix-only substituents are placed via the Phase-1 machinery; a
        SUFFIX-forming group (oxo/amino/acid/...) needs added indicated
        hydrogen -> deferred.
      * ``compute_fused_numbering`` yields a determinate numbering.

    Locants follow: lowest to indicated H (b), then hydro/'ene' (e), then
    detachable prefixes (f); ties broken over the mancude-skeleton automorphisms
     /. A produced valid-but-non-PIN numbering still
    round-trips and the gate keeps accuracy intact.
    """
    if not mancude_parent:
        return None

    # --- 1. Saturated-position set + heteroatom validation (case b enabler) ---
    sat = []
    for a in ring_atom_set:
        at = mol.GetAtomWithIdx(a)
        if at.GetIsAromatic():
            continue
        # A saturated ring position carries NO remaining unsaturation at the atom
        # note (a), the Blue Book -- "positions that are saturated,
        # i.e., where there are two ring bonds and sufficient exo bonds to satisfy
        # the bonding number of the atom"). Test that STRUCTURALLY, by
        # participation in any double bond, NOT by the RDKit hybridization proxy:
        # a conjugated pyridine-type ring -NH- that becomes saturated is marked
        # SP2 by RDKit (its lone pair delocalises into the adjacent aromatic
        # ring), so `hybridization != SP3` wrongly dropped that saturated N from
        # `sat`, emitting one hydro/indicated-H short (6,7-dihydro-5H- instead of
        # 4,5,6,7-tetrahydro-) -> OPSIN reconstructed a different, more-aromatic
        # skeleton and the gate abstained. The double-bond test admits the
        # conjugated NH while still skipping a genuine residual ring C=C / C=N;
        # and, exactly as the old proxy did, it skips an exocyclic-double-bond
        # position (oxo / =NR / ylidene), which is added-hydrogen
        # territory the substituent branch below fails closed on. (An sp3 atom
        # has no double bond, so this changes nothing for the byte-identical
        # saturated-carbon path.)
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in at.GetBonds()):
            continue                            # remaining unsaturation -> not a hydro pos
        sym = at.GetSymbol()
        if sym == 'C':
            sat.append(a)
            continue
        ring_deg = sum(1 for b in at.GetBonds()
                       if b.GetOtherAtom(at).GetIdx() in ring_atom_set)
        if (sym in ('O', 'S', 'Se', 'Te') and ring_deg == 2
                and at.GetTotalNumHs() == 0 and at.GetFormalCharge() == 0
                and not _ring_double_bond(at, ring_atom_set)):
            continue                            # benign divalent chalcogen -> not a hydro pos
        # Task C: a SATURATED sp3 ring NITROGEN in the added-hydro region is a
        # HYDRO position, not a defer. In a partially-saturated mancude system
        # (e.g. 4,5,6,7-tetrahydro-imidazo[5,4-c]pyridine) the pyridine-type
        # =N- of the mancude parent becomes -NH- under the ring saturation, so
        # it belongs in the max-double-bond partition exactly like a saturated
        # carbon. The pyrrole-type NH the old comment worried about is AROMATIC
        # (excluded by the `GetIsAromatic` check above), so a NON-aromatic N
        # reaching here was pyridine-type. Admit a neutral N as a hydro candidate;
        # a charged/hypervalent N still defers. A post-check below fails closed if
        # any N would be cited as INDICATED hydrogen (the genuinely ambiguous
        # case), so only the unambiguous all-N-hydro partition proceeds.
        if sym == 'N' and at.GetFormalCharge() == 0:
            sat.append(a)
            continue
        return None                             # charged / hypervalent -> defer
    if not sat:
        return None                             # fully mancude -> legacy/aromatic path

    # --- 2. Max-double-bond partition: indicated H (unmatched) vs hydro (matched) ---
    adj = {a: set() for a in sat}
    for a in sat:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() in adj:
                adj[a].add(nb.GetIdx())
    ih_sets = _saturation_indicated_h_sets(sat, adj)
    if not ih_sets:
        return None

    # (the Blue Book): "In preferred IUPAC names, all indicated
    # hydrogen atoms must be cited". The unsaturated part can hold the mancude
    # parent's indicated hydrogen itself (the pyrrole-type N-H of
    # '4,5,6,7-tetrahydro-1H-imidazo[4,5-c]pyridine'); it is cited with the name.
    arom_ih = _unsaturated_part_indicated_h(mol, ring_atom_set, set(sat))
    if arom_ih is None:
        return None

    # (:17026): the hydro prefixes go on the name of the MANCUDE parent.
    # For each way of placing the indicated hydrogen in the saturated region
    # (``ih_sets``), the mancude parent is the twin with one double bond restored on
    # each hydro pair; its catalogue name ('quinoline', '1H-indole',
    # '1H-benzimidazole') replaces a fusion name built from the saturated ring
    # ('cyclohexa[b]pyridine',: the six-membered carbocyclic attached
    # component is 'benzo'; 'benzo[d]imidazole',, with the catalogue's
    # numbering and the indicated hydrogen its name carries. A twin the descriptor
    # already names keeps the descriptor and its numbering.
    from .fusion_numbering import compute_fused_numbering
    options = []   # (parent name, numbering, the ih sets it may take, catalogue?)
    for S in ih_sets:
        _parent = _catalogued_mancude_parent(mol, ring_atom_set, set(sat) - set(S), adj)
        if _parent is not None and _parent[0] != mancude_parent:
            options.append((_parent[0], _parent[1], [S], True))
    if not options:
        canonical = compute_fused_numbering(mol, ring_atom_set)
        if not canonical:
            return None
        options = [(mancude_parent, canonical, ih_sets, False)]

    # --- 3. Substituents (case c). A suffix on a saturated position needs
    # added hydrogen -> out of scope; a suffix on an atom of the unsaturated
    # part takes the ring as it is examples: hydro prefixes with
    # the suffix of the mancude parent). ---
    has_sub = any(at.GetIdx() not in ring_atom_set and at.GetAtomicNum() != 1
                  for at in mol.GetAtoms())
    if has_sub and not _exocyclic_atoms_accounted(mol, ring_atom_set):
        return None

    # --- 4. Enumerate mancude-symmetry numberings; pick lowest per ---
    perms = _ring_system_automorphisms(mol, ring_atom_set, ignore_bond_order=True)
    canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    atoms_by_rank = sorted(ring_atom_set, key=lambda a: canon_rank[a])

    best = None  # (key, sig, cand, ih_set, substituents, parent name, catalogue?)
    for parent_name, canonical, sets, from_catalogue in options:
        named_ih = _leading_indicated_h(parent_name) if from_catalogue else None
        for perm in perms:
            try:
                cand = {a: _fused_locant_to_output(canonical[perm[a]])
                        for a in ring_atom_set}
            except KeyError:
                continue
            # sat carbons must map to plain-integer (non-fusion) locants
            if any(_fused_locant_num(cand[a])[1] != 0 for a in sat):
                continue
            # choose the ih-set giving lowest (ih_locants, hydro_locants) here
            best_ih = None
            for S in sets:
                ih_atoms = set(S) | arom_ih
                if from_catalogue and named_ih != {str(cand[a]) for a in ih_atoms}:
                    continue   # the catalogue name's indicated hydrogen is not this one
                ihl = tuple(sorted(_fused_locant_num(cand[a]) for a in ih_atoms))
                hyl = tuple(sorted(_fused_locant_num(cand[a]) for a in sat if a not in S))
                if best_ih is None or (ihl, hyl) < (best_ih[0], best_ih[1]):
                    best_ih = (ihl, hyl, S)
            if best_ih is None:
                continue
            if has_sub:
                subs = get_fused_heterocycle_substituents(mol, cand)
                sat_locs = {str(cand[a]) for a in sat}
                suffix_locs = {str(loc) for locs in subs['suffix_groups'].values()
                               for loc in locs}
                suffix_locs |= {str(loc) for loc in subs['amino_substituents']}
                if subs['oxo_substituents'] or suffix_locs & sat_locs:
                    return None                 # territory -> fail closed
                suffix_key, prefix_key = _fused_substituent_citation_key(subs)
            else:
                subs, suffix_key, prefix_key = None, (), ()
            # (the Blue Book): (b) indicated hydrogen (:3246), (c) suffixes
            # (:3256), (e) hydro prefixes (:3288-:3289), (f) detachable prefixes (:3301),
            # in this order ('5,6,7,8-tetrahydroquinoxaline-2-carboxylic acid', not -3-).
            # then (g) (:3307): lowest locants for the prefix cited first in the name.
            key = (best_ih[0], suffix_key, best_ih[1], prefix_key,
                   _first_cited_prefix_key(subs))
            sig = tuple(_fused_locant_num(cand[a]) for a in atoms_by_rank)
            placement = _substituent_placement(subs)
            if best is None or key < best[0]:
                best = (key, sig, cand, best_ih[2], subs, parent_name, from_catalogue)
                tied_placements = {placement}
            elif key == best[0]:
                tied_placements.add(placement)
                if sig < best[1]:
                    best = (key, sig, cand, best_ih[2], subs, parent_name, from_catalogue)
    if best is None:
        return None
    # A tie that (b)-(g) leave between numberings that place the substituents
    # differently fails closed; the best-effort tiers keep their own name for the
    # molecule.
    if len(tied_placements) > 1:
        return None

    # --- 5. Assemble '<hydro>-<indicatedH>-<mancude parent>' ---
    _key, _sig, cand, ih_set, subs, mancude_parent, from_catalogue = best
    # Task C fail-closed: a saturated N admitted above may only be a HYDRO atom,
    # unless the catalogue name of the mancude twin carries that indicated hydrogen
    # itself (1H-benzimidazole). Otherwise, if the chosen partition would cite a
    # nitrogen as INDICATED hydrogen, the pyrrole-vs-pyridine ambiguity the old
    # defer guarded against is live for THIS system -- defer to the legacy path
    # rather than risk a wrong constitution. (Carbon indicated-H, the common ``5H``
    # case, is unaffected.)
    if not from_catalogue and any(
            mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in ih_set):
        return None
    # An indicated hydrogen of the unsaturated part and one of the saturated region
    # together are more than the descriptor-named parent takes; the catalogue checks
    # its own name above.
    if not from_catalogue and ih_set and arom_ih:
        return None
    hydro_atoms = [a for a in sat if a not in ih_set]
    if not hydro_atoms:
        return None                             # pure indicated H (no hydro) -> legacy path
    mult = {2: 'di', 4: 'tetra', 6: 'hexa', 8: 'octa', 10: 'deca'}.get(len(hydro_atoms))
    if mult is None:
        return None                             # >10 hydro -> defer
    hydro_locs = sorted((cand[a] for a in hydro_atoms), key=_fused_locant_num)
    hydro_prefix = f"{','.join(str(l) for l in hydro_locs)}-{mult}hydro"
    cited_ih = set() if from_catalogue else (set(ih_set) | arom_ih)
    if cited_ih:
        ih_prefix = ','.join(
            f"{cand[a]}H"
            for a in sorted(cited_ih, key=lambda a: _fused_locant_num(cand[a]))
        )
        parent = f"{hydro_prefix}-{ih_prefix}-{mancude_parent}"
    elif mancude_parent[0].isalpha():
        parent = f"{hydro_prefix}{mancude_parent}"     # BYTE-IDENTICAL to the slice
    else:
        parent = f"{hydro_prefix}-{mancude_parent}"

    if has_sub:
        return _assemble_fused_heterocycle_name(mol, parent, subs, cand)
    return parent


def _try_algorithmic_fusion_name(mol) -> Optional[str]:
    """
    Attempt systematic fusion naming for 2-component ortho-fused systems.

    Called as fallback when dictionary lookup fails. Uses IUPAC
    to rules for component identification, descriptor generation,
    and name assembly.

    Only handles 2-component ortho-fused systems (SSSR has exactly 2 rings
    sharing exactly 2 atoms). 3+ component systems deferred .

    Args:
        mol: RDKit Mol object

    Returns:
        Systematic fusion name, or None if cannot be generated
    """
    import logging
    logger = logging.getLogger(__name__)

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Gate: must have at least 2 rings in SSSR (broadening — was != 2)
    if len(atom_rings) < 2:
        return None

    # a phase: route base decision through select_base_component
    # for any N>=2. The naming engine remains 2-ring-only.
    #
    # Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    # Source: 149-internal notes.
    from .fused_ring_selection import _enumerate_components, select_base_component
    components = _enumerate_components(mol)
    base_atoms, _others = select_base_component(mol, components)

    if len(atom_rings) != 2:
        # 3+ component case (trade-off): emit decision-only.
        # base_atoms is consumed by namer.Branch 6.5 (Task 02-03) via
        # features.parent_selection_result; full systematic-name assembly
        # deferred to 149.x or a phase.
        logger.debug(
            "FR-2.3 base decision for 3+ component system %s: %s",
            Chem.MolToSmiles(mol), base_atoms,
        )
        return None  # Decision-only; full name assembly deferred

    ring1, ring2 = atom_rings[0], atom_rings[1]
    shared = get_shared_atoms(mol, ring1, ring2)

    # Gate: exactly 2 shared atoms = ortho-fused
    if len(shared) != 2:
        return None

    # Gate: at least one ring must be heterocyclic
    # Purely carbocyclic fused systems (naphthalene, tetrahydronaphthalene)
    # are named via the carbocyclic pathway, not heterocycle fusion naming
    ring1_hetero = is_heterocyclic(mol, list(ring1))
    ring2_hetero = is_heterocyclic(mol, list(ring2))
    if not ring1_hetero and not ring2_hetero:
        return None

    # Gate: at least one ring must have aromatic atoms
    # Non-aromatic fused heterocycles (e.g., pyrazolidine bridged systems)
    # need dihydro/tetrahydro prefixes which the simple algorithmic path
    # does not handle. Only generate names for aromatic fused systems.
    ring1_aromatic = is_aromatic_ring(mol, list(ring1))
    ring2_aromatic = is_aromatic_ring(mol, list(ring2))
    if not ring1_aromatic and not ring2_aromatic:
        return None

    # a phase: a SUBSTITUTED 2-component ortho-fused mancude system
    # is named by discovering substituents against the algorithmic locant map
    # (the same machinery cataloged cores use), NOT refused. Record whether any
    # exocyclic heavy atom is present; the substituted branch runs below, after
    # the bare parent name + locant map + indicated-H/lambda are assembled.
    ring_atom_set = set(ring1) | set(ring2)
    has_substituents = any(
        atom.GetIdx() not in ring_atom_set and atom.GetAtomicNum() != 1
        for atom in mol.GetAtoms()
    )

    from .fusion_descriptors import (
        generate_systematic_name_for_fused_pair,
        identify_parent_and_child,
    )

    # Try to generate systematic name
    name = generate_systematic_name_for_fused_pair(
        mol, list(ring1), list(ring2), shared
    )
    # Suite fix j6 (TRIAGE g3 C08): the descriptor's OPSIN check parses the
    # BARE name, which OPSIN reads with its own default indicated hydrogen; for
    # an N-H system whose indicated hydrogen OPSIN puts elsewhere (6H-pyrrolo
    # [3,4-d]pyrimidine, 6H-pyrrolo[3,4-c]pyridazine) every candidate failed
    # and the system abstained. When the input is the mancude parent plus
    # indicated hydrogen (_mancude_indicated_h_atoms establishes the set;
    #, the Blue Book), retry matching the heteroatom skeleton
    # only, and ship the result only if that indicated hydrogen is cited below
    # (``skeleton_matched``); a hydro input (both N-H of pyrrolo[3,4-b]pyrrole)
    # keeps declining.
    skeleton_matched = False
    if not name and _mancude_indicated_h_atoms(mol, ring_atom_set):
        name = generate_systematic_name_for_fused_pair(
            mol, list(ring1), list(ring2), shared,
            allow_skeleton_match=True)
        skeleton_matched = bool(name)

    if not name:
        return None

    # (the Blue Book): this namer spells an azole component without
    # its bracketed Hantzsch-Widman locants ('isoxazolo[4,5-c]pyridine',
    # 'cyclohepta[d]thiazole'): a valid name, never the PIN. Every name built on it
    # (hydro prefixes, substituents) carries it and is labelled below the PIN.
    if _unbracketed_azole_component(name):
        _record_non_pin_spelling(name)

    # a phase: a partially-saturated fused pair is named as
    # '<locants>-<multiplier>hydro-<mancude parent>'. `name` here is the mancude
    # descriptor (generate_systematic_name_for_fused_pair is aromaticity-
    # agnostic). Try the dihydro path first; on None fall through to the legacy
    # indicated-H behaviour so this can never regress a currently-valid name.
    hydro_name = _try_partial_saturation_name(mol, ring_atom_set, name)
    if hydro_name:
        logger.debug(
            "Partial-saturation fusion name for %s: %s",
            Chem.MolToSmiles(mol), hydro_name,
        )
        return hydro_name

    # (the Blue Book): a benzene ring fused to a heteromonocycle is
    # named as a benzoheterocycle with its heteroatom locants. The catalogue match
    # misses such a system when a ring atom has a nonstandard bonding number ([SH2]),
    # so the descriptor name ('benzo[b]thiophene', never the PIN) was built above;
    # the catalogue benzo name of the same skeleton is the parent instead
    # ('1H-1λ4-1-benzothiophene'). The fusion numbering, so every locant added
    # below, is the same for both names.
    from .fusion_descriptors import catalogue_benzo_name
    _benzo = catalogue_benzo_name(mol, ring_atom_set)
    if _benzo is not None:
        name = _benzo

    # Compute indicated hydrogen for the algorithmic fused system
    parent_name, child_name, parent_ring, child_ring = identify_parent_and_child(
        mol, set(ring1), set(ring2)
    )

    # Build the atom->locant map that drives indicated-H, lambda, and (
    # a phase) substituent placement.
    # - BARE systems keep the legacy descriptor-order map -> byte-identical
    # output, zero regression risk.
    # - SUBSTITUTED systems use the deterministic peripheral-numbering
    # engine (compute_fused_numbering) + the lowest-substituent-
    # locant tie-break; the legacy map does NOT reproduce OPSIN's canonical
    # numbering (it placed substituents on the wrong ring position). Fail
    # closed if the engine or the completeness check declines.
    substituents = None
    if has_substituents:
        if not _exocyclic_atoms_accounted(mol, ring_atom_set):
            return None
        selected = _select_substituent_numbering(mol, ring_atom_set)
        if selected is None:
            return None
        atom_to_locant, substituents = selected
    else:
        atom_to_locant = _build_algorithmic_locant_map(
            mol, parent_ring, child_ring, shared, parent_name, child_name
        )

    indicated_h = _compute_general_indicated_h(mol, ring_atom_set, atom_to_locant)
    if not indicated_h:
        indicated_h = _mancude_indicated_h_locants(
            mol, ring_atom_set, atom_to_locant, has_substituents)
    if skeleton_matched:
        # The descriptor was accepted on the skeleton alone: ship only with the
        # established indicated hydrogen cited,:14607), never bare.
        _ih = _mancude_indicated_h_locants(
            mol, ring_atom_set, atom_to_locant, has_substituents)
        if not _ih or list(indicated_h) != list(_ih):
            return None

    #: lambda (nonstandard bonding number) tokens follow the
    # atom's fused-system locant and are cited at the beginning of the name,
    # after any indicated hydrogen (monocyclic precedent 1H-1lambda4-thiophene,
    #. Fail-closed: a lambda atom without a determinate NUMERIC
    # fused locant (or on skeletal carbon) invalidates the whole name — the
    # lambda-less name would denote a DIFFERENT molecule.
    from .lambda_convention import LAMBDA, nonstandard_bonding_number
    lam_entries = []
    for idx in sorted(ring_atom_set):
        lam = nonstandard_bonding_number(mol, idx)
        if lam is None:
            continue
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C':
            return None
        loc = atom_to_locant.get(idx)
        if loc is None or not str(loc).isdigit():
            return None  # fusion-junction letter locants (4a) out of scope
        lam_entries.append((int(loc), lam))
    lam_prefix = ''
    if lam_entries:
        lam_entries.sort()
        lam_prefix = ','.join(
            f"{loc}{LAMBDA}{lam}" for loc, lam in lam_entries
        ) + '-'

    if indicated_h:
        h_parts = ','.join(str(loc) + 'H' for loc in indicated_h)
        name = f"{h_parts}-{lam_prefix}{name}"
        # (the Blue Book): indicated hydrogen tells apart the
        # isomers of the MANCUDE parent; (:17026): the degree of
        # hydrogenation is given by 'hydro' prefixes ('6,7-dihydro-5H-benzo[7]annulene
        # (PIN)',:17032). More indicated hydrogen than the mancude parent takes
        # ('5H,6H,7H,8H,9H-cyclohepta[b]pyridine', '1H,2H,3H-benzo[d]imidazole') is
        # a saturated region spelled as indicated hydrogen: valid, never the PIN.
        if len(indicated_h) > _mancude_indicated_h_count(mol, ring_atom_set):
            _record_non_pin_spelling(name[:-1] if name.endswith('e') else name)
    elif lam_prefix:
        name = f"{lam_prefix}{name}"

    # a phase: attach the substituents discovered above against the
    # canonical numbering, reusing the exact assembler the catalog path uses.
    # The bare-parent `name` (with any indicated-H/lambda prefix already
    # assembled) is the parent the assembler decorates.
    if has_substituents:
        name = _assemble_fused_heterocycle_name(
            mol, name, substituents, atom_to_locant
        )

    logger.debug(
        "Algorithmic fusion name for %s: %s",
        Chem.MolToSmiles(mol), name,
    )

    return name


# ============================================================================
# Polycomponent ortho-fusion constructor — Phase G1b (DD7)
# ============================================================================
#
# Builds systematic fusion PINs for the cata-fused single-base monocyclic-
# component "star" sub-class: a single most-senior base ring /.3)
# with >=2 attached monocyclic components, each ortho-fused to the base by exactly
# one bond, no interior (peri) atom, unsubstituted, no carbocyclic attached
# component. This replaces the `len(atom_rings) != 2 -> return None` decision-only
# gate for the structures it can name CORRECTLY (e.g. difuro[3,2-b:2',3'-e]pyridine,
# furo[3,2-b]thieno[2,3-e]pyridine); everything outside the sub-class fails closed
# (returns None -> existing path / G0 fail-closed veto). It does NOT handle
# multiparent bases ("difuran"), ortho-peri (interior atom), second-order attached
# components, bridged-AND-fused, or carbocyclic children (those are deferred, A10).
#
# Source: IUPAC 2013 Blue Book (descriptor construction),
# (base component), https://iupac.qmul.ac.uk/fusedring/FR23.html

# Carbocyclic attached components (benzo/cyclopenta...) are refused: nearly all
# 3-ring benzo-fused systems are RETAINED (acridine/carbazole/dibenzofuran) and
# named upstream; emitting a systematic dibenzo[...] here would be a non-PIN.
_PCF_CARBOCYCLIC = {
    'benzene', 'cyclopentadiene', 'cyclopentene',
    'cycloheptadiene', 'cycloheptene', 'cyclohexene',
    'cyclooctatetraene', 'cyclooctene',
}
_PCF_MULTIPLIERS = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}
# Heteroatom seniority for the base's isolated IUPAC numbering (senior -> low
# locant): O > S > Se > Te > N > P > As > B (Hantzsch-Widman element order).
_PCF_HET_NUM_SENIORITY = {
    'O': 0, 'S': 1, 'Se': 2, 'Te': 3, 'N': 4, 'P': 5, 'As': 6, 'Sb': 7, 'B': 8,
}


def _pcf_ring_cycle(mol, ring_atoms: List[int]) -> Optional[List[int]]:
    """Return the ring atoms in cyclic adjacency order, or None if not a simple
    monocycle (a fused/interior atom has !=2 in-ring neighbours)."""
    ring_set = set(ring_atoms)
    adj = {}
    for idx in ring_atoms:
        nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None
        adj[idx] = nbrs
    start = ring_atoms[0]
    order = [start]
    prev, curr = start, adj[start][0]
    while curr != start:
        order.append(curr)
        nxt = [n for n in adj[curr] if n != prev]
        if not nxt:
            return None
        prev, curr = curr, nxt[0]
    return order if len(order) == len(ring_atoms) else None


def _pcf_base_numberings(mol, base_ring: List[int]) -> List[List[int]]:
    """Candidate isolated IUPAC numberings of the base monocycle: all
    rotations + reflections that achieve the most-senior-heteroatom locant key
    (heteroatoms at lowest locants, senior element first). For pyridine/furan
    this is the 2 directions with the heteroatom at position 1; for benzene all
    12 orderings. The descriptor-minimising one is chosen by the caller."""
    cyc = _pcf_ring_cycle(mol, base_ring)
    if cyc is None:
        return []
    n = len(cyc)
    orderings = []
    for start in range(n):
        for direction in (1, -1):
            orderings.append([cyc[(start + direction * i) % n] for i in range(n)])

    def het_key(order):
        items = []
        for i, a in enumerate(order):
            sym = mol.GetAtomWithIdx(a).GetSymbol()
            if sym != 'C':
                items.append((i + 1, _PCF_HET_NUM_SENIORITY.get(sym, 9)))
        return tuple(sorted(items))

    best = min((het_key(o) for o in orderings), default=())
    uniq, seen = [], set()
    for o in orderings:
        if het_key(o) == best and tuple(o) not in seen:
            seen.add(tuple(o))
            uniq.append(o)
    return uniq


def _pcf_edge_descriptor(border: List[int], child_order: List[int],
                         shared: Set[int]) -> Optional[Tuple[str, Tuple[int, int]]]:
    """For a fixed base numbering `border`, return (edge_letter, (child_lo,
    child_hi)) for the shared ortho edge. Child locants are cited in the
    base-lettering direction (child locant of the lower-position base atom
    first). Mirrors fusion_descriptors.generate_fusion_descriptor lower/higher
    + wraparound logic."""
    n = len(border)
    a, b = list(shared)
    try:
        pa, pb = border.index(a), border.index(b)
    except ValueError:
        return None
    if pa < pb:
        if pb - pa == 1:
            lower, higher, edge = a, b, pa
        elif pb - pa == n - 1:
            lower, higher, edge = b, a, pb
        else:
            return None
    else:
        if pa - pb == 1:
            lower, higher, edge = b, a, pb
        elif pa - pb == n - 1:
            lower, higher, edge = a, b, pa
        else:
            return None
    if edge >= 26:
        return None
    letter = chr(ord('a') + edge)
    try:
        clo = child_order.index(lower) + 1
        chi = child_order.index(higher) + 1
    except ValueError:
        return None
    return letter, (clo, chi)


def _pcf_assemble(per, base_name: str) -> str:
    """Assemble the fusion name from per-attachment (component_name, letter,
    locs). Identical components are multiplied (difuro) and their descriptors
    combined in one bracket (colon-separated, primed on repeats); distinct
    components are cited as separate prefixes in alphanumerical order."""
    from .fusion_descriptors import get_fusion_prefix
    groups = defaultdict(list)
    for (nm, letter, locs) in per:
        groups[nm].append((letter, locs))

    tokens = []  # (sort_key, token)
    for nm, items in groups.items():
        items.sort(key=lambda x: x[0])  # by edge letter
        prefix = get_fusion_prefix(nm)
        mult = _PCF_MULTIPLIERS.get(len(items), '')
        any_locs = any(locs is not None for (_, locs) in items)
        parts = []
        for k, (letter, locs) in enumerate(items):
            prime = "'" * k
            if locs is None:
                parts.append(letter)
            else:
                lo, hi = locs
                parts.append(f"{lo}{prime},{hi}{prime}-{letter}")
        sep = ":" if any_locs else ","
        tokens.append((prefix, f"{mult}{prefix}[{sep.join(parts)}]"))

    tokens.sort(key=lambda t: t[0])  # alphanumerical citation order
    return "".join(t[1] for t in tokens) + base_name


def _pcf_name_with_base(mol, comps: List[List[int]], names: List[str],
                        base_idx: int) -> Optional[str]:
    """Try to name the system with comps[base_idx] as the single base ring.
    Returns the fusion name, or None if the star topology is not satisfied."""
    from .fusion_descriptors import (
        _get_iupac_ring_order_for_fusion,
        _cooptimal_child_orderings,
    )

    base_ring = comps[base_idx]
    base_name = names[base_idx]
    base_set = set(base_ring)

    attachments = []  # (comp_idx, name, shared_set)
    for i, c in enumerate(comps):
        if i == base_idx:
            continue
        shared = base_set & set(c)
        if len(shared) != 2:
            return None  # not first-order star (peri / multiparent / disjoint)
        s = list(shared)
        if mol.GetBondBetweenAtoms(s[0], s[1]) is None:
            return None  # shared atoms not a bond -> not an ortho edge
        if names[i] in _PCF_CARBOCYCLIC:
            return None  # carbocyclic child -> defer (retained dibenzo systems)
        attachments.append((i, names[i], shared))

    if not attachments:
        return None

    borders = _pcf_base_numberings(mol, base_ring)
    if not borders:
        return None

    best = None  # (key, name)
    for border in borders:
        per = []
        ok = True
        for (ci, nm, shared) in attachments:
            # Enumerate the co-optimal child numberings for this attachment and,
            # for the fixed base `border`, cite the lower descriptor pair. A
            # SYMMETRIC attached component (heteroatom equidistant from the fusion
            # bond, e.g. the second furan of difuro[3,2-b:3',4'-e]pyridine) ties
            # on heteroatom + fusion-bond locants; (d)
            # (the Blue Book) breaks that tie by the lower cited pair
            # ((3,4) < (4,3) -> 3',4'-e, not 4',3'-e), the direction of parent
            # lettering:11911). An asymmetric child yields a single
            # ordering, so the citation is the lettering-forced pair (may descend)
            # and MATCH rows such as difuro[3,2-b:2',3'-e]pyridine are unchanged.
            child_orders = _cooptimal_child_orderings(mol, comps[ci], shared)
            if not child_orders:
                child_orders = [_get_iupac_ring_order_for_fusion(
                    mol, comps[ci], shared, is_child=True)]
            res = None
            for child_order in child_orders:
                r = _pcf_edge_descriptor(border, child_order, shared)
                if r is None:
                    continue
                if res is None or r[1] < res[1]:
                    res = r
            if res is None:
                ok = False
                break
            letter, locs = res
            per.append((nm, letter, locs))
        if not ok:
            continue
        # Lowest letter set; tiebreak by letters in alphanumerical citation
        # order (so the first-cited component gets the lowest letter), then by
        # child locants in citation order. /.
        from .fusion_descriptors import get_fusion_prefix
        by_cite = sorted(per, key=lambda p: (get_fusion_prefix(p[0]), p[1]))
        key = (tuple(sorted(p[1] for p in per)),
               tuple(p[1] for p in by_cite),
               tuple(p[2] for p in by_cite))
        name = _pcf_assemble(per, base_name)
        if best is None or key < best[0]:
            best = (key, name)

    return best[1] if best else None


def _try_polycomponent_fusion_name(mol) -> Optional[str]:
    """Systematic fusion name for a cata-fused single-base monocyclic-component
    star system restricted sub-class). Returns the PIN, or None when
    the molecule is outside the handled sub-class (fail closed; the caller then
    keeps its existing behaviour). MUST be invoked AFTER retained-name checks —
    it does not recognise retained systems and would emit a systematic name for
    them (acridine etc.); carbocyclic children are additionally refused.

    Source: IUPAC 2013 Blue Book,;.3.
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) < 3:
        return None  # >=3 components only; 2-component handled by existing path

    ring_atom_set = set()
    for r in atom_rings:
        ring_atom_set.update(r)
    # a phase: a SUBSTITUTED star system is decorated below against
    # the deterministic final-system numbering (was: hard-refused).
    # Record whether any exocyclic heavy atom is present; the bare descriptor
    # name is computed first (the `_pcf_*` path is substituent-agnostic — it
    # walks ring cycles only), then substituents are attached.
    has_substituents = any(
        atom.GetIdx() not in ring_atom_set and atom.GetAtomicNum() != 1
        for atom in mol.GetAtoms()
    )

    # Cata-fused only: no atom shared by >=3 rings (no interior/peri atom).
    membership = defaultdict(int)
    for r in atom_rings:
        for idx in r:
            membership[idx] += 1
    if any(c >= 3 for c in membership.values()):
        return None

    # Fully aromatic + neutral only. The descriptor carries no hydro /
    # indicated-H / charge, so a saturated, partially-saturated, or charged
    # cata-fused system would be MIS-NAMED — e.g. a perhydro difuropyridine
    # would emit the aromatic 'difuro[3,2-b:2',3'-e]pyridine', silently dropping
    # the saturation (`_identify_ring_name` is aromaticity-agnostic: it returns
    # 'furan' for a saturated O-5-ring). The 2-component algorithmic path has the
    # same aromaticity requirement. Fail closed otherwise.
    for idx in ring_atom_set:
        a = mol.GetAtomWithIdx(idx)
        if (not a.GetIsAromatic() or a.GetFormalCharge() != 0
                or a.GetNumRadicalElectrons() != 0):
            return None

    # Each component must be a recognised monocycle.
    comps = [list(r) for r in atom_rings]
    from .fusion_descriptors import _identify_ring_name
    names = [_identify_ring_name(mol, c) for c in comps]
    if any(not nm for nm in names):
        return None

    #.3(a): base is among the most-senior components. Try EACH most-senior
    # component as base; only the one(s) satisfying the single-base star topology
    # yield a name. Return iff the produced names are unique (deterministic; the
    # central ring of a linear-symmetric system is the only valid base, and
    # symmetric duplicates collapse to one name). Multiparent (two senior rings,
    # both terminal -> no valid base) collapses to no name -> None.
    from .fused_ring_selection import _rank
    ranks = [_rank(mol, set(c)) for c in comps]
    min_rank = min(ranks)
    base_candidates = [i for i, r in enumerate(ranks) if r == min_rank]

    produced = set()
    for bidx in base_candidates:
        nm = _pcf_name_with_base(mol, comps, names, bidx)
        if nm:
            produced.add(nm)
    if len(produced) != 1:
        return None
    bare_name = next(iter(produced))
    if not has_substituents:
        return bare_name

    # a phase: decorate the bare star name with substituents against
    # the canonical numbering, reusing the Phase-1 machinery. Fail
    # closed at source: every exocyclic heavy atom must be an identifiable
    # substituent, and the numbering engine must produce a determinate map,
    # else return None (additionally suppresses any non-round-tripping
    # emission).
    if not _exocyclic_atoms_accounted(mol, ring_atom_set):
        return None
    selected = _select_substituent_numbering(mol, ring_atom_set)
    if selected is None:
        return None
    atom_to_locant, substituents = selected
    return _assemble_fused_heterocycle_name(
        mol, bare_name, substituents, atom_to_locant
    )


def _name_base_subcore(mol, pair_atoms: Set[int]) -> Optional[str]:
    """Name the 2-ring sub-core spanned by ``pair_atoms`` (a fused ring PAIR) via
    the existing 2-component fused namer. Returns the base component name (e.g.
    'quinoline', 'pyrido[2,3-d]pyrimidine') or None. Used by the bounded
    3-component generate-and-test to pick a max-ring retained base."""
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in pair_atoms and b.GetEndAtomIdx() in pair_atoms]
    if not bonds:
        return None
    try:
        sub = Chem.PathToSubmol(mol, bonds)
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    if sub.GetRingInfo().NumRings() != 2:
        return None
    res = name_fused_heterocycle(sub)
    if res is None:
        return None
    return res[0]


def _name_ortho_fused_generate_and_test(mol) -> Optional[str]:
    """Bounded 3-component ortho/ortho-peri-fused MANCUDE fusion namer,
    0-wrong by construction (every candidate is OPSIN-round-trip-gated to the FULL
    InChI before it can be returned).

    Fusion nomenclature is not derived de-novo here; instead — in the manner of an
    offline OPSIN-validated template index, and following
    the contributor guide a project rule (OFFER many, keep the one that round-trips) — a bounded
    candidate set is generated from the actual ring components (a max-ring retained
    BASE named from a fused sub-core per, plus the remaining monocycle as a
    fusion PREFIX with enumerated attachment locants/letters) and each is OPSIN-
    parsed; only a candidate whose parse InChI equals the input's is kept, and the
    deterministic canonical (shortest, then lexicographically least) is returned.
    Fail-CLOSED (None) on anything outside the handled sub-class, so it never emits
    an unverified name; the caller keeps its existing behaviour.

    Scope (this session, -C1C2C6 Build 2): exactly 3 SSSR rings, the whole
    molecule is that one cata-fused (no atom in >=3 rings) neutral fully-aromatic
    ring system with >=1 ring heteroatom, no exocyclic heavy atoms. Higher
    component counts, peri-fused interior atoms, substituted cores and carbocyclic
    parents are a NAMED follow-on (see test_c_waveC_v36.py Build-2 xfail)."""
    import itertools
    import re
    from collections import Counter as _Counter

    from rdkit.Chem.inchi import MolToInchi

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) != 3:
        return None
    ring_sets = [set(r) for r in atom_rings]
    ring_atom_set: Set[int] = set().union(*ring_sets)
    # whole molecule must BE this ring system (bare core, no exocyclic heavy atoms)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1 and a.GetIdx() not in ring_atom_set:
            return None
    # neutral, fully aromatic, no radicals; and >=1 ring heteroatom (carbo-PAHs
    # are retained-catalog territory and reach here only if unnamed -> leave them)
    has_hetero = False
    for idx in ring_atom_set:
        a = mol.GetAtomWithIdx(idx)
        if (not a.GetIsAromatic() or a.GetFormalCharge() != 0
                or a.GetNumRadicalElectrons() != 0):
            return None
        if a.GetAtomicNum() != 6:
            has_hetero = True
    if not has_hetero:
        return None
    # cata-fused only (no interior/peri atom in >=3 rings)
    membership = _Counter(idx for r in atom_rings for idx in r)
    if any(c >= 3 for c in membership.values()):
        return None

    target_inchi = MolToInchi(mol)
    if not target_inchi:
        return None

    from ..validation.opsin_roundtrip import opsin_parse
    from .fusion_descriptors import _identify_ring_name, get_fusion_prefix

    # (the Blue Book): "Locants that describe structural
    # features of components, such as positions of heteroatoms, are kept with
    # the name of the component and are enclosed within square brackets" --
    # '[1]benzopyrano[2,3-c]pyrrole (PIN)' (:12157). And (:11982):
    # the 1,2-/1,3-azoles are components only by their Hantzsch-Widman names.
    _HW_COMPONENT = {'oxazole': '[1,3]oxazole', 'isoxazole': '[1,2]oxazole',
                     'thiazole': '[1,3]thiazole', 'isothiazole': '[1,2]thiazole'}

    def _component(name: str) -> str:
        # a component's own indicated hydrogen is not cited: the fused system
        # gets its own, added below
        core = re.sub(r'^\d+[a-z]?H-', '', name)
        m = re.match(r'^(\d+(?:,\d+)*)-(.+)$', core)
        return f"[{m.group(1)}]{m.group(2)}" if m else _HW_COMPONENT.get(core, core)

    def _attached_prefix(name: str) -> Optional[str]:
        core = re.sub(r'^\d+[a-z]?H-', '', name)
        m = re.match(r'^(\d+(?:,\d+)*)-(.+)$', core)
        if m:
            inner = get_fusion_prefix(m.group(2))
            return f"[{m.group(1)}]{inner}" if inner else None
        if core in _HW_COMPONENT:
            inner = get_fusion_prefix(_HW_COMPONENT[core].split(']', 1)[1])
            return _HW_COMPONENT[core].split(']', 1)[0] + ']' + inner if inner else None
        return get_fusion_prefix(core) or None

    # "Seniority criteria for selecting the parent component"
    # (the Blue Book): "If there is a choice for selecting the parent
    # component, the following criteria are considered, in order, until a
    # decision can be made":
    # (a) (:12139) "a component containing at least one of the heteroatoms
    # occurring earlier in the following order: N > F > Cl > Br > I > O > S
    # > Se > Te > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In >
    # Tl" ('[1]benzopyrano[2,3-c]pyrrole (PIN)',:12157);
    # (b) (:12163) "a component containing the greater number of rings"
    # ('6H-pyrazino[2,3-b]carbazole (PIN)');
    # (c) (:12234) "A component containing the larger ring at the first point
    # of difference when comparing rings in order of decreasing size"
    # ('2H-furo[3,2-b]pyran (PIN) [pyran (6 ring) preferred to furan (5
    # ring)]',:12246);
    # (d) (:12260) "A component containing the greater number of heteroatoms
    # of any kind" ('5H-pyrido[2,3-d][1,2]oxazine (PIN)');
    # (e) (:12282) "A component containing the greater variety of heteroatoms";
    # (f) (:12298) "A component containing the greater number of heteroatoms
    # most senior when considered in the order: F > Cl > Br > I > O > S >
    # Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In
    # > Tl" ('[1,3]selenazolo[5,4-d][1,3]thiazole (PIN) (S,N senior to
    # Se,N)').
    # (h) (:12336) "A component with the lower locants for heteroatoms"
    # ('pyrazino[2,3-d]pyridazine (PIN) (locants '1,2' of pyridazine
    # preferred to locants '1,4' of pyrazine)');
    # (i) (:12341) the lower locants for the heteroatoms "considered in the
    # order: F > Cl > Br > I > O > S > Se > Te > N > P >...".
    # (g) (rings in a horizontal row) never separates the one- and two-ring
    # components met here; (j) (fusion-carbon locants) is not computed: when two
    # DIFFERENT components still tie, the parent is undecided and nothing is
    # returned (never a string order).
    # Only (a) and (b) were applied before, so a tie on them fell to the letter
    # and then the shorter string: 'pyrazino[2,3-g]quinoline' (quinoxaline has
    # two heteroatoms, (d): 'pyrido[2,3-g]quinoxaline') and '3H-pyrido[3,2-e]
    # indole' (quinoline 6,6 is senior to indole 6,5, (c): '3H-pyrrolo[3,2-f]
    # quinoline') shipped as pin_verified.
    _HETERO_ORDER = ('N', 'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'P', 'As',
                     'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl')
    _HETERO_ORDER_F = ('F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As',
                       'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In',
                       'Tl')

    def _parent_key(atoms: Set[int], rings: List[Set[int]]):
        syms = [mol.GetAtomWithIdx(a).GetSymbol() for a in atoms]
        het = [s for s in syms if s != 'C']
        ranks = [_HETERO_ORDER.index(s) for s in het if s in _HETERO_ORDER]
        counts = _Counter(het)
        return (
            min(ranks) if ranks else len(_HETERO_ORDER),              # (a)
            -len(rings),                                              # (b)
            tuple(-len(r) for r in sorted(rings, key=len, reverse=True)),  # (c)
            -len(het),                                                # (d)
            -len(counts),                                             # (e)
            tuple(-counts.get(e, 0) for e in _HETERO_ORDER_F),        # (f)
        )

    def _mono_het_locants(ring: List[int]):
        # (h)/(i) key of a monocycle: its heteroatom locants in its own
        # numbering (heteroatoms as low as possible, then by the (i) order)
        from .heterocycles import _macrocycle_ordered_ring
        order = _macrocycle_ordered_ring(mol, set(ring))
        if not order:
            return None
        n = len(order)
        best = None
        for start in range(n):
            for step in (1, -1):
                seq = [order[(start + step * k) % n] for k in range(n)]
                het = [(i + 1, mol.GetAtomWithIdx(a).GetSymbol())
                       for i, a in enumerate(seq)
                       if mol.GetAtomWithIdx(a).GetSymbol() != 'C']
                h = tuple(sorted(l for l, _ in het))
                i_key = tuple(l for l, e in sorted(
                    het, key=lambda t: (_HETERO_ORDER_F.index(t[1])
                                        if t[1] in _HETERO_ORDER_F else 99, t[0])))
                if best is None or (h, i_key) < best:
                    best = (h, i_key)
        return best

    def _pair_het_locants(name: str):
        # (h)/(i) key of a named two-ring component, from its catalog numbering
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        for cs, entry in FUSED_HETEROCYCLE_DATA.items():
            if entry.get('name') != name or not entry.get('iupac_locants'):
                continue
            cm = Chem.MolFromSmiles(cs)
            if cm is None:
                return None
            het = []
            for a, l in entry['iupac_locants'].items():
                if a >= cm.GetNumAtoms() or not isinstance(l, int):
                    continue
                at = cm.GetAtomWithIdx(a)
                if at.IsInRing() and at.GetSymbol() != 'C':
                    het.append((l, at.GetSymbol()))
            h = tuple(sorted(l for l, _ in het))
            i_key = tuple(l for l, e in sorted(
                het, key=lambda t: (_HETERO_ORDER_F.index(t[1])
                                    if t[1] in _HETERO_ORDER_F else 99, t[0])))
            return (h, i_key)
        return None

    # row (2) (the Blue Book-11523): "phenanthroline
    # (1,7-isomer shown; the PIN is 1,7-phenanthroline; other isomers are: 1,8-;
    # 1,9-; 1,10-; 2,7-; 2,8-; 2,9-; 3,7-; 3,8-; 4,7-)". Every angular 6-6-6
    # system with one pyridine-type N in each terminal ring is a phenanthroline:
    # its PIN is the retained name, never a pyrido-quinoline fusion name. Only
    # 1,10-phenanthroline is in the catalog, so the others are not named here.
    if (all(len(r) == 6 for r in atom_rings)
            and sum(1 for a in ring_atom_set
                    if mol.GetAtomWithIdx(a).GetSymbol() != 'C') == 2
            and all(mol.GetAtomWithIdx(a).GetSymbol() in ('C', 'N')
                    for a in ring_atom_set)):
        central = [r for r in ring_sets
                   if sum(1 for o in ring_sets if o is not r and len(o & r) >= 2) == 2]
        if len(central) == 1:
            outer = [r for r in ring_sets if r is not central[0]]
            fus = [central[0] & o for o in outer]
            # angular (phenanthrene-like): the two fusion bonds of the central
            # ring are joined by one bond; linear (anthracene-like): they are not
            angular = any(mol.GetBondBetweenAtoms(x, y) is not None
                          for x in fus[0] for y in fus[1])
            n_outer = [sum(1 for a in o - central[0]
                           if mol.GetAtomWithIdx(a).GetSymbol() == 'N')
                       for o in outer]
            if angular and n_outer == [1, 1]:
                return None

    # (:11885) the parent component is a ring or ring system with a
    # retained or systematic (monocycle, Hantzsch-Widman, benzo) name; a name
    # that itself carries a fusion descriptor ('thieno[3,2-b]thiophene',
    # 'furo[3,2-b]furan') is a fused system of two components, not one
    # component. As a parent it violated (a)/(b) ('pyrido[3,2-d]thieno[3,2-b]
    # thiophene' for a pyridine-containing system whose PIN is
    # "thieno[2',3':4,5]thieno[2,3-b]pyridine" with primed higher-order locants,
    #, which this namer does not build); as an attached component
    # it needs those primes too. Such a pair is skipped in both orientations.
    _FUSION_DESCRIPTOR = re.compile(r"\[[^\]]*[a-z][^\]]*\]")

    def _is_single_component(name: str) -> bool:
        return not _FUSION_DESCRIPTOR.search(name)

    candidates: Dict[str, tuple] = {}
    tried = 0
    _CAP = 1200  # hard bound on OPSIN probes (safety; the class is small)

    # A ring NH needs the fused system's indicated hydrogen,
    # '9H-pyrido[2,3-b]indole'); a name without it only round-trips because the
    # parser supplies the hydrogen itself. Offer each position and keep the one
    # that round-trips (the contributor guide a project rule), lowest locant first.
    needs_ih = any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6
                   and mol.GetAtomWithIdx(a).GetTotalNumHs() > 0
                   for a in ring_atom_set)
    n_ring_atoms = len(ring_atom_set)

    target_smiles = Chem.MolToSmiles(mol)

    def _skeleton(inchi: str):
        # formula + connection layers: the constitution without its hydrogen
        # layer, so a name the parser completes with a CH2 instead of the NH
        # ('pyrrolo[3,2-g]quinoline') still counts as the right ring system
        parts = inchi.split('/')
        return tuple(parts[1:3])

    target_skeleton = _skeleton(target_inchi)

    def _rt(nm: str, mode: str = 'full') -> bool:
        nonlocal tried
        tried += 1
        s = opsin_parse(nm)
        if not s:
            return False
        om = Chem.MolFromSmiles(s)
        if om is None:
            return False
        if mode == 'tautomer':
            # the standard InChI merges mobile-H tautomers ('1H-' and '9H-
            # pyrido[2,3-b]indole' share it), so the hydrogen position is
            # checked on the canonical SMILES
            return Chem.MolToSmiles(om) == target_smiles
        got = MolToInchi(om)
        if not got:
            return False
        if mode == 'skeleton':
            return _skeleton(got) == target_skeleton
        return got == target_inchi

    def _probe(prefix: str, pairs, letters: str, base: str, parent: str) -> None:
        for pair in pairs:
            for L in letters:
                if tried >= _CAP:
                    return
                # (:11911): "To the letter... are prefixed, if
                # necessary, the numbers of the positions of attachment of the
                # other component" -- never for benzo ('benzo[g]isoquinoline',
                #:11909), whose attachment positions are all alike
                if pair is None:
                    nm = f"{prefix}[{L}]{base}"
                    x_y = ()
                else:
                    nm = f"{prefix}[{pair[0]},{pair[1]}-{L}]{base}"
                    x_y = tuple(pair)
                if not needs_ih:
                    if not _rt(nm):
                        continue
                    # (:11911): the letter "as early in the alphabet
                    # as possible", then the attachment locants "as low as is
                    # consistent with the numbering of the compound"
                    candidates[nm] = (parent, L, x_y)
                    continue
                # With a ring NH only the skeleton is checked first: the parser
                # may complete the hydrogen-free name with a CH2, whose standard
                # InChI differs, and the indicated-hydrogen name then never got
                # offered ('1H-pyrrolo[3,2-g]quinoline').
                if not _rt(nm, 'skeleton'):
                    continue
                for loc in range(1, n_ring_atoms + 1):
                    if tried >= _CAP:
                        return
                    ih = f"{loc}H-{nm}"
                    if _rt(ih, 'tautomer'):
                        candidates[ih] = (parent, L, x_y, loc)
                        break

    # "Heteromonocyclic rings fused to a benzene ring" (:13435):
    # a benzene + heteromonocycle pair is one component unit (a
    # 'benzoheterocycle') only "in which the benzene ring is not part of a system
    # having a retained name such as quinoline or naphthalene", and "this
    # approach is not used if it disrupts a multiparent system",
    # 'benzo[1,2-b:4,5-c']difuran (PIN) (not furo[3,4-f][1]benzofuran');
    # "Retained names are senior to names of benzoheterocycles" ('6H-dibenzo
    # [b,d]pyran (PIN) (not 6H-benzo[c][1]benzopyran'). So '3H-benzo[1,2-e]
    # benzimidazole' (the benzene is part of naphthalene) was not a PIN.
    _BENZO_NAME = re.compile(r'^(?:\d+[a-z]?H-)?(?:\d+(?:,\d+)*-)?benz')

    def _is_benzene(ring: Set[int]) -> bool:
        return len(ring) == 6 and all(
            mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in ring)

    # Every component a parent can be: the three monocycles and each ortho-
    # fused pair that is ONE component (retained / benzo name). The winner must
    # have the senior one as its parent.
    components: Dict[str, tuple] = {}
    het_locant_keys: Dict[str, Optional[tuple]] = {}
    for r in atom_rings:
        rname = _identify_ring_name(mol, list(r))
        if not rname:
            return None
        components[_component(rname)] = _parent_key(set(r), [set(r)])
        het_locant_keys[_component(rname)] = _mono_het_locants(list(r))

    for i, j in itertools.combinations(range(3), 2):
        if len(ring_sets[i] & ring_sets[j]) < 2:
            continue  # rings i,j are not ortho-fused to each other
        pair_atoms = ring_sets[i] | ring_sets[j]
        k = ({0, 1, 2} - {i, j}).pop()
        attached_ring = list(atom_rings[k])
        if len(set(attached_ring) & pair_atoms) < 2:
            continue  # the third ring is not fused to this pair -> wrong base pair
        pair_name = _name_base_subcore(mol, pair_atoms)
        if (pair_name is None and _is_benzene(ring_sets[i])
                and _is_benzene(ring_sets[j])):
            # the heterocycle namer does not name carbocycles; two benzene rings
            # are the retained component naphthalene, Table 2.7,
            # the Blue Book)
            pair_name = 'naphthalene'
        if not pair_name or not _is_single_component(pair_name):
            continue
        if _BENZO_NAME.match(pair_name):
            benzene = next((r for r in (i, j) if _is_benzene(ring_sets[r])), None)
            if benzene is not None and len(ring_sets[benzene] & ring_sets[k]) >= 2:
                # the benzene ring is fused to the third ring too
                hetero = ({i, j} - {benzene}).pop()
                other = _name_base_subcore(mol, ring_sets[benzene] | ring_sets[k])
                if not other or not _BENZO_NAME.match(other):
                    continue  # part of a retained-name system (naphthalene, quinoline)
                if (_identify_ring_name(mol, list(atom_rings[k]))
                        == _identify_ring_name(mol, list(atom_rings[hetero]))):
                    return None  # a multiparent name ('benzo[...]difuran'), not built
        pair_component = _component(pair_name)
        pair_key = _parent_key(pair_atoms, [ring_sets[i], ring_sets[j]])
        if components.get(pair_component, pair_key) != pair_key:
            return None  # one name, two different components: do not guess
        components[pair_component] = pair_key
        het_locant_keys[pair_component] = _pair_het_locants(pair_name)
        mono_name = _identify_ring_name(mol, attached_ring)
        if not mono_name:
            continue
        rs = len(attached_ring)
        mono_pairs = []
        for a in range(1, rs + 1):
            b = a + 1 if a < rs else 1
            mono_pairs.append((a, b))
            mono_pairs.append((b, a))
        if mono_name == 'benzene':
            mono_pairs = [None]
        # (1) the fused pair is the parent, the monocycle the attached prefix
        #: '[1,3]oxazolo', its heteroatom locants in brackets)
        prefix = _attached_prefix(mono_name)
        if prefix:
            _probe(prefix, mono_pairs, 'abcdefghij', pair_component,
                   pair_component)
        # (2) the monocycle is the parent, the fused pair the attached prefix
        pair_prefix = _attached_prefix(pair_name)
        if pair_prefix:
            n_pair = len(pair_atoms)
            pair_pairs = []
            for a in range(1, n_pair + 1):
                for b in (a - 1, a + 1):
                    if 1 <= b <= n_pair:
                        pair_pairs.append((a, b))
            _probe(pair_prefix, pair_pairs, 'abcdefghij'[:rs],
                   _component(mono_name), _component(mono_name))
    if not candidates:
        return None
    best_key = min(components.values())
    senior = [c for c, key in components.items() if key == best_key]
    if len(senior) > 1:
        # (h), (i): heteroatom locants of each tied component
        hk = {c: het_locant_keys.get(c) for c in senior}
        if any(v is None for v in hk.values()):
            return None
        best_h = min(hk.values())
        senior = [c for c in senior if hk[c] == best_h]
    if len(senior) != 1:
        return None  # (j) would decide: not computed here
    kept = {nm: key for nm, key in candidates.items() if key[0] == senior[0]}
    if not kept:
        return None  # the senior component's name was never generated
    return min(kept, key=lambda n: (kept[n][1:], len(n), n))


# Naphthalene has exactly two symmetry-distinct peripheral fusion bonds — the
# alpha,beta (1,2) and beta,beta (2,3) edges (all other peripheral bonds are
# equivalent to these under D2h). Cited both orientations; OPSIN + the InChI
# gate keeps whichever (if any) reproduces the input constitution, and the
# deterministic canonical pick lowest-locants -> lexicographic)
# selects 2,3 over 3,2 for the linear (PIN) descriptor.
_NAPHTHO_FUSION_PAIRS = ((1, 2), (2, 1), (2, 3), (3, 2))


def _ring_adjacency_ortho(atom_rings: List[Set[int]]) -> Dict[int, Set[int]]:
    """Undirected ring-adjacency graph: rings sharing >=2 atoms (an ortho fusion
    bond) are adjacent. Used to enumerate CONNECTED sub-core partitions."""
    import itertools as _it
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(atom_rings))}
    for i, j in _it.combinations(range(len(atom_rings)), 2):
        if len(atom_rings[i] & atom_rings[j]) >= 2:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _rings_connected(subset: Set[int], adj: Dict[int, Set[int]]) -> bool:
    """Is the ring-index ``subset`` connected in the ring-adjacency graph?"""
    if not subset:
        return False
    it = iter(subset)
    seen = {next(it)}
    stack = list(seen)
    while stack:
        c = stack.pop()
        for nb in adj[c]:
            if nb in subset and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(subset)


def _name_naphtho_fused_generate_and_test(mol) -> Optional[str]:
    """CT.4 : general N-component ortho-fused MANCUDE PIN construction for
    the deferred CARBOCYCLIC-CHILD topology that ``_try_polycomponent_fusion_name``
    (:907) explicitly defers (A10): a senior heterocyclic 2-ring BASE
    (quinoxaline / quinoline / quinazoline /...) ortho-fused to a NAPHTHALENE
    2-ring carbocyclic PREFIX -> e.g. ``naphtho[2,3-g]quinoxaline``.

    This UPGRADES the RT-true von-Baeyer degradation (a valid name) to the
    fusion PIN; a system it cannot assemble correctly returns None and the caller
    keeps its existing behaviour (the von-Baeyer no-abstain fallback) -- never
    wrong, never abstain.

    0-wrong by construction: exactly as ``_name_ortho_fused_generate_and_test``,
    a bounded candidate set is generated from the actual ring components (a
    nameable heterocyclic 2-ring BASE named from a fused sub-core per,
    plus the naphthalene 2-ring carbocyclic PREFIX with its two symmetry-distinct
    fusion bonds and enumerated base letters) and each is OPSIN-parsed; only a
    candidate whose parse InChI equals the input's is kept, and the deterministic
    canonical (shortest, then lexicographically least) is returned.

    Scope (CT.4): exactly 4 SSSR rings; the whole molecule is that one neutral
    fully-aromatic cata-fused (no atom in >=3 rings) ring system with >=1 ring
    heteroatom and no exocyclic heavy atoms; it partitions into a nameable
    heterocyclic 2-ring base + a naphthalene 2-ring carbocyclic attached
    component. Higher component counts, 3-ring bases + monocycle prefixes,
    ortho-peri interior atoms, non-naphthalene carbocyclic prefixes
    (indeno / azuleno / anthra / phenanthro), multiparent bases and substituted
    cores are a NAMED follow-on (see test_ct4_ncomponent_fusion.py).

    Source: IUPAC 2013 Blue Book,,;.3.
    """
    import itertools
    from collections import Counter as _Counter

    from rdkit.Chem.inchi import MolToInchi

    ri = mol.GetRingInfo()
    atom_rings_t = ri.AtomRings()
    if len(atom_rings_t) != 4:
        return None
    ring_sets: List[Set[int]] = [set(r) for r in atom_rings_t]
    ring_atom_set: Set[int] = set().union(*ring_sets)

    # Whole molecule must BE this ring system (bare core, no exocyclic heavy).
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1 and a.GetIdx() not in ring_atom_set:
            return None
    # Neutral, fully aromatic, no radicals; >=1 ring heteroatom (all-carbon PAHs
    # -- naphthacene/chrysene -- are retained/von-Baeyer territory, not this).
    has_hetero = False
    for idx in ring_atom_set:
        a = mol.GetAtomWithIdx(idx)
        if (not a.GetIsAromatic() or a.GetFormalCharge() != 0
                or a.GetNumRadicalElectrons() != 0):
            return None
        if a.GetAtomicNum() != 6:
            has_hetero = True
    if not has_hetero:
        return None
    # Cata-fused only (no interior/peri atom shared by >=3 rings).
    membership = _Counter(idx for r in atom_rings_t for idx in r)
    if any(c >= 3 for c in membership.values()):
        return None

    target_inchi = MolToInchi(mol)
    if not target_inchi:
        return None

    from ..validation.opsin_roundtrip import opsin_parse

    def _is_naphthalene(atoms: Set[int]) -> bool:
        """The sub-core spanned by ``atoms`` is a naphthalene: exactly two
        six-membered rings, every atom an aromatic carbon."""
        rs = [r for r in ring_sets if r <= atoms]
        if len(rs) != 2 or any(len(r) != 6 for r in rs):
            return False
        return all(mol.GetAtomWithIdx(x).GetSymbol() == 'C'
                   and mol.GetAtomWithIdx(x).GetIsAromatic() for x in atoms)

    adj = _ring_adjacency_ortho(ring_sets)
    candidates: Set[str] = set()
    tried = 0
    _CAP = 400  # bounded; the class is tiny (<=2 partitions x 4 pairs x 12 letters)
    for base_idx in itertools.combinations(range(4), 2):
        base = set(base_idx)
        attached = set(range(4)) - base
        if not (_rings_connected(base, adj) and _rings_connected(attached, adj)):
            continue
        base_atoms = set().union(*[ring_sets[i] for i in base])
        att_atoms = set().union(*[ring_sets[i] for i in attached])
        # Base must be the senior heterocyclic component; the all-
        # carbon naphthalene is the attached PREFIX, never the base.
        base_name = _name_base_subcore(mol, base_atoms)
        if not base_name:
            continue
        if not _is_naphthalene(att_atoms):
            continue
        for (x, y) in _NAPHTHO_FUSION_PAIRS:
            for L in 'abcdefghijkl':
                if tried >= _CAP:
                    break
                nm = f"naphtho[{x},{y}-{L}]{base_name}"
                tried += 1
                s = opsin_parse(nm)
                if not s:
                    continue
                om = Chem.MolFromSmiles(s)
                if om is not None and MolToInchi(om) == target_inchi:
                    candidates.add(nm)
    if not candidates:
        return None
    return sorted(candidates, key=lambda n: (len(n), n))[0]


def _core_covers_ring_system(mol, atom_mapping) -> bool:
    """Does the matched fused-ring core cover every atom of the fused ring
    system(s) it sits in? Mirrors composer._fused_core_covers_ring_system (kept
    local to avoid the composer->fused_rings circular import). int keys in
    atom_mapping are the named core atoms; a pendant ring joined by a single
    (non-ring) bond is a SEPARATE ring system so a legit cyclic substituent does
    not trip this. Empty mapping (exact-match path) -> True (no-op)."""
    core_atoms = {k for k in (atom_mapping or {}) if isinstance(k, int)}
    if not core_atoms:
        return True
    for rs in get_ring_systems(mol):
        rs = set(rs)
        if (core_atoms & rs) and not rs.issubset(core_atoms):
            return False
    return True


def _build_algorithmic_locant_map(
    mol,
    parent_ring: List[int],
    child_ring: List[int],
    shared: Set[int],
    parent_name: str,
    child_name: str,
) -> Dict[int, Any]:
    """
    Build atom-to-IUPAC-locant mapping for a 2-component fused system.

    For 2-component ortho-fused, peripheral numbering starts at the parent
    ring and continues through the child ring. Fusion junction atoms get
    'Na' suffix locants (e.g., 4a, 8a).

    This mapping is used by _compute_general_indicated_h to assign
    locant labels to atoms with indicated hydrogen.

    Args:
        mol: RDKit Mol object
        parent_ring: List of atom indices in the parent ring
        child_ring: List of atom indices in the child ring
        shared: Set of atom indices shared between the two rings
        parent_name: Name of the parent ring component
        child_name: Name of the child ring component

    Returns:
        Dict mapping atom index to IUPAC locant (int or str like '4a')
    """
    from .fusion_descriptors import _get_iupac_ring_order_for_fusion

    parent_iupac = _get_iupac_ring_order_for_fusion(
        mol, parent_ring, shared, is_child=False
    )
    child_iupac = _get_iupac_ring_order_for_fusion(
        mol, child_ring, shared, is_child=True
    )

    atom_to_locant: Dict[int, Any] = {}
    locant = 1

    # Walk the periphery: parent non-shared atoms first, then junction,
    # then child non-shared atoms, then second junction atom
    # For a simple 2-ring ortho-fused system, the periphery is:
    # parent non-shared atoms -> first shared atom (Xa) ->
    # child non-shared atoms -> second shared atom (Ya)

    # Number parent ring atoms (non-shared get integer locants)
    parent_shared_positions = []
    for i, atom_idx in enumerate(parent_iupac):
        if atom_idx not in shared:
            atom_to_locant[atom_idx] = locant
            locant += 1
        else:
            # Record position for 'a' suffix locant
            parent_shared_positions.append((i, atom_idx, locant - 1))

    # Assign 'a' suffix locants to shared atoms in parent order
    for _, atom_idx, prev_loc in parent_shared_positions:
        atom_to_locant[atom_idx] = f"{prev_loc}a"

    # Number child ring atoms (non-shared continue integer sequence)
    for atom_idx in child_iupac:
        if atom_idx not in shared and atom_idx not in atom_to_locant:
            atom_to_locant[atom_idx] = locant
            locant += 1

    return atom_to_locant


def get_shared_atoms(mol, ring1: Tuple[int, ...], ring2: Tuple[int, ...]) -> Set[int]:
    """
    Get atoms shared between two rings.

    Args:
        mol: RDKit Mol object
        ring1: Tuple of atom indices in first ring
        ring2: Tuple of atom indices in second ring

    Returns:
        Set of atom indices shared by both rings

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1') # indole
        >>> ri = mol.GetRingInfo
        >>> rings = ri.AtomRings
        >>> shared = get_shared_atoms(mol, rings[0], rings[1])
        >>> len(shared) # 2 atoms shared in ortho-fused system
        2
    """
    set1 = set(ring1)
    set2 = set(ring2)
    return set1 & set2


def classify_fused_system(mol) -> str:
    """
    Classify a fused ring system by its fusion type.

    Classification:
    - 'ortho-fused': All ring pairs share exactly 2 atoms (one edge)
    - 'ortho-peri-fused': At least one ring shares atoms with 3+ other rings
    - 'bridged-fused': Bridges exist across fused system (like norbornane)
    - 'not-fused': Rings share 0-1 atoms (isolated or spiro)

    Args:
        mol: RDKit Mol object

    Returns:
        Classification string

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1') # indole
        >>> classify_fused_system(mol)
        'ortho-fused'
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34') # pyrene
        >>> classify_fused_system(mol)
        'ortho-peri-fused'
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return 'not-fused'

    # Track how many rings each ring shares atoms with
    # and how many atoms each pair of rings shares
    ring_connection_count = defaultdict(int)  # ring_idx -> count of connected rings

    # Track atoms shared between each pair of rings
    fused_pairs = []  # List of (ring_i, ring_j, shared_count)

    for i, ring1 in enumerate(atom_rings):
        for j, ring2 in enumerate(atom_rings):
            if i >= j:
                continue

            shared = get_shared_atoms(mol, ring1, ring2)
            shared_count = len(shared)

            if shared_count >= 2:
                # These rings are fused (share at least one edge)
                fused_pairs.append((i, j, shared_count))
                ring_connection_count[i] += 1
                ring_connection_count[j] += 1
            elif shared_count == 1:
                # Spiro connection - only one shared atom
                pass  # Don't count as fused

    if not fused_pairs:
        return 'not-fused'

    # Check for ortho-peri-fused: any ring connected to 3+ other rings
    for ring_idx, count in ring_connection_count.items():
        if count >= 3:
            return 'ortho-peri-fused'

    # Check for bridged-fused: any pair shares more than 2 atoms
    # (indicating a bridge across the ring system)
    for i, j, shared_count in fused_pairs:
        if shared_count > 2:
            # More than 2 shared atoms suggests bridging
            return 'bridged-fused'

    # All fused pairs share exactly 2 atoms - ortho-fused
    return 'ortho-fused'


def is_fused_bicyclic(mol) -> bool:
    """
    Check if molecule is a fused bicyclic system (exactly 2 rings sharing one edge).

    This distinguishes fused bicyclics from bridged bicyclics (like norbornane)
    which have bridgehead atoms shared by more than 2 rings conceptually.

    Args:
        mol: RDKit Mol object

    Returns:
        True if exactly 2 rings sharing exactly 2 atoms

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1') # indole
        >>> is_fused_bicyclic(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12') # carbazole (tricyclic)
        >>> is_fused_bicyclic(mol)
        False
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) != 2:
        return False

    ring1, ring2 = atom_rings[0], atom_rings[1]
    shared = get_shared_atoms(mol, ring1, ring2)

    # Exactly 2 shared atoms = one shared edge = ortho-fused bicyclic
    return len(shared) == 2


def name_fused_heterocycle(mol):
    """
    Generate IUPAC name for a fused heterocycle.

    Naming priority:
    1. Check xanthine derivatives FIRST (caffeine, theophylline, etc.)
       - These have specific N-position numbering (1,3,7-trimethyl format)
    2. Check retained names (indole, quinoline, carbazole, etc.)
    3. Add tautomer locant if present (1H-indole)
    4. For substituted: find substituents and add prefixes
    5. Handle N-substitution specially (N-methyl, not 1-methyl)

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included)
        where substituents_included is True (fused heterocycle handler
        already discovers substituents via get_fused_heterocycle_substituents),
        or None if not a recognized fused heterocycle.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1') # indole
        >>> result = name_fused_heterocycle(mol)
        >>> result[0]
        '1H-indole'
    """
    if mol is None:
        return None

    # Collect all ring atoms from SSSR for structured return
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)

    # /: a fused system containing a nonstandard-bonding-
    # number atom may ONLY be named by the algorithmic path (which emits the
    # mandatory lambda tokens). Catalog and core-substructure matches are
    # standard-valence structures — matching one here would name a different
    # molecule. Fail closed if the algorithmic path declines.
    from .lambda_convention import LAMBDA as _LAMBDA
    from .lambda_convention import nonstandard_bonding_number as _nsbn
    if any(_nsbn(mol, i) is not None for i in ring_atoms):
        _lam_name = _try_algorithmic_fusion_name(mol)
        if _lam_name and _LAMBDA in _lam_name:
            return (_lam_name, ring_atoms, {}, True)
        return None

    # Substituted purine-2,6-dione class (caffeine, theophylline, theobromine,
    # paraxanthine, N-/8-substituted xanthines): systematic PIN on the retained
    # purine parent with fixed numbering. Runs BEFORE the xanthine lookup so any
    # alkyl/halogen-substituted member is named structurally (fixed-parent form
    # 3,7-dihydro-1H-purine-2,6-dione) rather than via brittle canonical-SMILES
    # keys; declines (fail-closed) for the bare parent, which keeps its retained
    # name below.
    from .purine_oxo import name_purine_26_dione
    purine_dione_name = name_purine_26_dione(mol)
    if purine_dione_name:
        return (purine_dione_name, ring_atoms, {}, True)

    # Check xanthine derivatives (caffeine, theophylline, etc.) — retained
    # 'xanthine'/'hypoxanthine' parents and any member the class engine declines.
    xanthine_name = get_xanthine_name(mol)
    if xanthine_name:
        return (xanthine_name, ring_atoms, {}, True)

    # Substituted purine (adenine/hypoxanthine/purine skeleton carrying a ring
    # substituent): systematic PIN on the fixed purine parent with graph-derived
    # indicated H. Runs BEFORE the retained/core-match path so a substituted
    # adenine is NOT captured by the 10-atom retained `adenine` core (which
    # absorbs the C6 amino and mis-places the ring-N substituent). Declines the
    # bare base, which keeps its retained name below.
    from .purine import name_substituted_purine
    subst_purine_name = name_substituted_purine(mol)
    if subst_purine_name:
        return (subst_purine_name, ring_atoms, {}, True)

    # Substituted mono-6-oxo purine (hypoxanthine/guanine family): systematic
    # PIN on the fixed purine parent with a constructive `-6-one` template and
    # graph-derived added-indicated-H. Runs AFTER the 2,6-dione and xanthine
    # handlers above (which keep first refusal on any 2,6-dione/trione) and
    # near `name_substituted_purine` (non-oxo purines), since the two engines
    # handle disjoint (oxo vs non-oxo) cases. Declines the bare base, which
    # keeps its retained name below.
    from .purine import name_oxo_purine
    oxo_purine_name = name_oxo_purine(mol)
    if oxo_purine_name:
        return (oxo_purine_name, ring_atoms, {}, True)

    # First try exact match for unsubstituted fused heterocycle
    result = get_fused_heterocycle_name(mol)
    if result:
        name, tautomer_locant = result
        return (name, ring_atoms, {}, True)  # Name already includes tautomer locant if present

    # Try substructure matching for substituted fused heterocycles
    core_result = match_fused_heterocycle_core(mol)
    if core_result is None:
        # Check global retained names as fallback (catches nucleobases and
        # other retained heterocycles that may not have exact SMILES keys in
        # FUSED_HETEROCYCLE_DATA due to tautomerism). Only for multi-ring
        # heterocyclic systems (has heteroatom in ring) to avoid catching
        # monocyclics (pyridine) or carbocyclics (naphthalene).
        has_ring_heteroatom = any(
            mol.GetAtomWithIdx(idx).GetSymbol() != 'C'
            for idx in ring_atoms
        )
        if ri.NumRings() >= 2 and has_ring_heteroatom:
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            global_retained = _get_global_retained_name(canonical_smi)
            if global_retained:
                return (global_retained, ring_atoms, {}, True)
        # Try algorithmic systematic fusion naming (IUPAC to
        # for 2-component ortho-fused systems not in the dictionary
        algorithmic_name = _try_algorithmic_fusion_name(mol)
        if algorithmic_name:
            return (algorithmic_name, ring_atoms, {}, True)
        # b: polycomponent ortho-fusion constructor for 3+-
        # component cata-fused monocyclic-component systems with no catalog core.
        poly = _try_polycomponent_fusion_name(mol)
        if poly:
            return (poly, ring_atoms, {}, True)
        # -C1C2C6 Build 2: bounded 3-component ortho/ortho-peri-fused
        # mancude namer for a novel system whose senior BASE is a 2-ring retained
        # component (e.g. pyrimido[4,5-b]quinoline) -- the case _try_polycomponent_
        # fusion_name cannot reach (it only builds star-of-monocycles bases).
        # 0-wrong by construction: OPSIN-RT-gated candidate generation.
        gen = _name_ortho_fused_generate_and_test(mol)
        if gen:
            return (gen, ring_atoms, {}, True)
        # CT.4: general N-component fusion -- the deferred
        # carbocyclic-child topology (naphtho[2,3-g]quinoxaline): a senior
        # heterocyclic 2-ring base ortho-fused to a naphthalene carbocyclic
        # prefix. UPGRADES the RT-true von-Baeyer degradation to the fusion PIN.
        # 0-wrong by construction (OPSIN-RT-gated); None -> caller keeps the
        # von-Baeyer no-abstain fallback.
        naphtho = _name_naphtho_fused_generate_and_test(mol)
        if naphtho:
            return (naphtho, ring_atoms, {}, True)
        # Wave-2 P5 fused (Task 7) FAIL-CLOSED follow-up: interior-heteroatom
        # ortho-/peri-fused systems /.2.1/.2.2/.2.3/.3.3.2) require
        # a superscript interior locant (e.g. 3a1 / 2a1H). OPSIN 2.9 CANNOT
        # parse any name carrying that token (verified: '2a1H-cyclopenta[cd]-
        # pyrene', 'pyracylene' -> blank), so there is NO verifiable oracle for
        # the interior-locant PIN. We deliberately decline here (return None)
        # rather than emit an unverifiable interior-superscript fusion name;
        # the production gate additionally suppresses any von-Baeyer
        # fallback that does not round-trip. Build the interior-atom-numbering
        # engine when a parseable oracle (newer OPSIN / hand-checked internal
        # oracle) exists. See tests/unit/rules/test_wave2_p5_fused.py
        #::TestP25InteriorAtomNumberingFailClosed.
        return None

    core_name, atom_mapping, _core_smiles = core_result

    # b: a 2-component catalog core (e.g. furo[3,2-b]pyridine)
    # can match as a SUBSTRUCTURE of a larger polycomponent fused system (e.g.
    # difuropyridine); its leftover fused ring would be mis-named as a phantom
    # acyclic substituent (the G0 '7-ethoxyfuro[3,2-b]pyridine' defect). When the
    # matched core does NOT cover the whole fused ring system, try the
    # polycomponent constructor first; None -> the downstream G0 coverage veto
    # still fails closed on the phantom name.
    if not _core_covers_ring_system(mol, atom_mapping):
        poly = _try_polycomponent_fusion_name(mol)
        if poly:
            return (poly, ring_atoms, {}, True)
        # a phase note: when poly is None here the matched catalog core is
        # fused to leftover ring atoms it cannot name. The fall-through below
        # generates a phantom substituent name (the '7-ethoxyfuro[3,2-b]pyridine'
        # ring-as-acyclic class defect), which is suppressed downstream by the G0
        # coverage veto + the self-consistency gate (verified: such systems emit
        # 'unknown' in production). An explicit `return None` here was evaluated
        # and REJECTED — it routes the molecule to the monocyclic benzene handler,
        # which drops the heterocyclic rings and emits the WORSE 'benzene' in the
        # gate-off diagnostic layer (production is 'unknown' either way). The
        # phantom-then-veto path keeps the molecule in the fused-naming lane and
        # yields a cleaner 'unknown'. Truly fail-closing at the source needs the
        # benzene/monocyclic handlers to decline fused-to-heteroaromatic systems
        # (a dispatch-level change, a phase polycomponent-fusion scope).

    # NOTE: General indicated hydrogen (_compute_general_indicated_h) is
    # implemented but NOT wired here. Dictionary-matched fused systems handle
    # indicated H via tautomer_locant in fused_heterocycles.py entries.
    # The general algorithm is reserved for future systematic fusion naming
    # of non-retained fused systems (deferred from a phase).

    # Find substituents on the core
    substituents = get_fused_heterocycle_substituents(mol, atom_mapping)

    # (fail-closed routing): get_fused_heterocycle_substituents silently
    # skips any exocyclic branch _identify_fused_substituent cannot name
    # (``sub_info is None -> continue``), so the assembled name can DROP a whole
    # substituent and denote a DIFFERENT molecule (a boronate / methanesulfonyl /
    # silyl group on quinoline collapses to bare 'quinoline'). On the PIN/default
    # path that group-dropping name is caught only by the downstream OPSIN
    # gate, which FAILS OPEN when the jar is absent -> the wrong name ships. Under
    # the general-engine tiers (valid / complete; general_fallback set) fail closed
    # HERE: decline the catalog match so the namer's late-recovery re-routes the
    # molecule through name_general (the universal never-None substituent
    # recursion, +E1 gated) instead of shipping — or being blocked by — a
    # group-dropping catalog name. ``_exocyclic_atoms_accounted`` re-runs the same
    # read-only traversal and returns False on any unnameable branch or unaccounted
    # heavy atom. It used to be gated on ``general_fallback_ctx`` (PIN path
    # byte-identical).
    #
    # 2026-09-25 (pre-existing-failures plan, Task 4, TRIAGE row 88): the PIN
    # tier now fails closed here too. "SUBSTITUTIVE NOMENCLATURE"
    # (the Blue Book): the parent's substitutable hydrogen atoms "are
    # substituted by nomenclaturally significant structural fragments represented
    # either by prefixes and/or suffixes" -- a branch with no prefix is simply not
    # in the name. With the gate off the PIN tier shipped
    # '5-hydroxy-2,3-dihydro-1H-isoindole-1,3-dione' (C8H5NO3) for
    # 5-hydroxythalidomide (C13H10N2O5): the 2-(2,6-dioxopiperidin-3-yl) branch,
    # which only the best-effort substituent namer can build, was dropped. With
    # the gate on the same candidate was voided, so no shipped PIN name changes;
    # a declined catalog match lets the next producer try instead.
    if not _exocyclic_atoms_accounted(mol, set(atom_mapping)):
        return None

    if not substituents:
        return (core_name, ring_atoms, atom_mapping, True)

    # (c): number the core so the suffix gets the lowest locants.
    core_name, atom_mapping, substituents = _renumber_core_for_suffix(
        mol, core_name, _core_smiles, atom_mapping, substituents)

    # Build the substituted name (None: the principal characteristic group could
    # not be expressed correctly on this core -- decline)
    name = _assemble_fused_heterocycle_name(mol, core_name, substituents, atom_mapping)
    if name is None:
        return None
    # A core drawn in the bond-shift alternation that numbers its substituents
    # away from the lowest locants ('5-methylheptalene' for the drawing whose
    # partner is '1-methylheptalene') has a Delta-descriptor PIN,
    # the Blue Book-14601) that is not written here: the name is labelled
    # below the PIN (name-scoped non-PIN record), the name itself is unchanged.
    from ..data.fused_heterocycles import numbering_held_by_drawn_alternation
    if numbering_held_by_drawn_alternation(mol, _core_smiles, atom_mapping):
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return (name, ring_atoms, atom_mapping, True)


def _exocyclic_amine_n_substituents(mol, n_idx: int, core_atoms):
    """``(names, atoms)`` for an EXOCYCLIC amine nitrogen, or ``None`` to decline.

    ``names`` are the groups on the nitrogen; ``atoms`` are every atom they
    cover, so a caller that must account for the whole fragment can do so
    without re-walking the graph.

    : an amine is a characteristic group and, when it is the senior one, must
    be expressed as the ``-amine`` SUFFIX -- ``N-methylquinolin-2-amine``, never
    the parent hydride plus an ``(methylamino)`` prefix. The collector this serves
    routed only the BARE ``amino`` to the suffix, so every N-substituted amine on
    a fused parent was demoted to a prefix and shipped a non-PIN
    (``2-(methylamino)quinoline``).

    Decided from the GRAPH, never by parsing the prefix name -- the same lesson
    this module records at the anilino site, where naming from a carbon COUNT
    turned a quinolinyl-amine into ``heptylamino``.

    Declines (``None``) unless the nitrogen is a plain amine N:
      * neutral, unradicalised, three-valent, all single bonds -- so an amide,
        imine, nitro, N-oxide or nitrile N is never mistaken for one. Those are
        senior to or different from an amine and belong to other producers.
      * not in ANY ring, so a ring nitrogen keeps the existing is_nitrogen path.
      * bonded to the core atom plus one or two CARBON branches, none of which is
        an acyl carbon (that would make it an amide,, senior to the amine).
      * every branch nameable by the shared substituent namer. A branch we cannot
        name is refused outright rather than approximated.
    """
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetSymbol() != 'N':
        return None
    if n_atom.GetFormalCharge() != 0 or n_atom.GetNumRadicalElectrons() != 0:
        return None
    if n_atom.IsInRing() or n_atom.GetIsAromatic():
        return None
    if n_atom.GetTotalValence() != 3:
        return None
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in n_atom.GetBonds()):
        return None

    branches = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                if nb.GetIdx() not in core_atoms]
    if not branches or len(branches) > 2:
        return None

    from ..assembly.substituent_enumerator import name_substituent

    names, all_atoms = [], []
    for b_idx in branches:
        b_atom = mol.GetAtomWithIdx(b_idx)
        if b_atom.GetSymbol() != 'C':
            return None
        # An acyl carbon makes this an AMIDE, not an amine.
        for nb in b_atom.GetNeighbors():
            if nb.GetSymbol() in ('O', 'S', 'Se', 'Te') and \
                    mol.GetBondBetweenAtoms(b_idx, nb.GetIdx()).GetBondType() == \
                    Chem.BondType.DOUBLE:
                return None
        frag, seen, stack = [], {n_idx} | set(core_atoms), [b_idx]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            frag.append(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors()
                         if nb.GetIdx() not in seen)
        try:
            nm = name_substituent(mol, frag, b_idx)
        except Exception:                                     # noqa: BLE001
            nm = None
        if not nm:
            return None
        names.append(nm)
        all_atoms.extend(frag)
    return names, all_atoms


def get_fused_heterocycle_substituents(
    mol,
    core_match: Dict[int, Any]
) -> Dict:
    """
    Find substituents on a fused heterocycle core.

    Identifies atoms not in the core match as potential substituents,
    maps them to IUPAC locants using core numbering, and tracks
    N-substitution separately.

    Args:
        mol: RDKit Mol object
        core_match: Dict mapping mol atom indices to IUPAC locants (int or str like '3a')

    Returns:
        Dict with:
        - 'c_substituents': Dict[str, List[int]] - C-substituent name -> locants
        - 'n_substituents': Dict[str, int] - N-substituent name -> count
        - 'other': List[Dict] - Other substituents (halogens, etc.)

    Examples:
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1') # 5-methylindole
        >>> core_match = match_fused_heterocycle_core(mol)[1]
        >>> subs = get_fused_heterocycle_substituents(mol, core_match)
        >>> 'methyl' in subs['c_substituents']
        True
    """
    core_atoms = set(core_match.keys())

    result = {
        'c_substituents': defaultdict(list),  # name -> list of locants
        'n_substituents': defaultdict(int),   # name -> count
        'oxo_substituents': [],   # list of locants for =O (suffix: -one)
        'amino_substituents': [], # list of locants for -NH2 (suffix: -amine)
        'suffix_groups': defaultdict(list),   # suffix_name -> list of locants (carboxylic acid, etc.)
        'other': [],  # For halogens, etc.
    }

    # Find which core atoms have substituents
    for core_atom_idx, locant in core_match.items():
        core_atom = mol.GetAtomWithIdx(core_atom_idx)
        is_nitrogen = core_atom.GetSymbol() == 'N'

        for neighbor in core_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the core
            if nbr_idx in core_atoms:
                continue

            # Identify the substituent
            sub_info = _identify_fused_substituent(mol, nbr_idx, core_atoms)
            if sub_info is None:
                continue

            sub_name = sub_info.get('name')
            sub_type = sub_info.get('type', 'alkyl')

            if sub_type == 'alkyl':
                if is_nitrogen:
                    #: a RING nitrogen has a NUMERIC locant, and that
                    # locant is the PIN citation -- `1-methyl-1H-indole`, on the
                    # model of `(1H-indol-1-yl)acetic acid (PIN)`
                    # (the Blue Book), where the ring N is cited as `1-yl`
                    # and never `N-yl`. The italic-`N` form is the fallback for
                    # when no ring numbering is available, which is never the
                    # case here: `locant` comes straight from `core_match`.
                    # This site emitted `N-methyl-1H-indole`; the monocyclic
                    # sibling (`heterocycles.py::_format_n_substituent`) was
                    # migrated to numeric locants and this one was missed.
                    #
                    # A ring-N substituent is rendered exactly like any other
                    # ring substituent once it has a locant, so it joins
                    # `c_substituents` rather than growing a parallel branch --
                    # the italic-N path below now serves ONLY the exocyclic
                    # amine nitrogen, which genuinely has no numeric locant.
                    result['c_substituents'][sub_name].append(locant)
                else:
                    # C-substitution
                    result['c_substituents'][sub_name].append(locant)
            elif sub_type == 'oxo':
                # Oxo group (=O) - use suffix form (-one)
                result['oxo_substituents'].append(locant)
            elif sub_type == 'functional' and sub_name == 'amino':
                # Amino group (-NH2) - use suffix form (-amine)
                result['amino_substituents'].append(locant)
            elif sub_type == 'functional' and (
                    _amine := _exocyclic_amine_n_substituents(
                        mol, nbr_idx, core_atoms)):
                #: an N-SUBSTITUTED amine is still an amine, so it takes the
                # `-amine` suffix with its N-substituents cited as italic-N
                # prefixes -- `N-methylquinolin-2-amine`. Only the bare `amino`
                # reached the suffix before this branch, so every N-substituted
                # amine on a fused parent fell through to the generic `functional`
                # case below and shipped the parent hydride plus a
                # `(methylamino)`/`anilino` prefix: a name with NO suffix for the
                # senior characteristic group. `4-methoxy-N-phenylaniline (PIN)`
                # (the Blue Book) is the shape required.
                #
                # `anilino` is a genuine preferred PREFIX (:6371
                # `4-[(4-hydroxyanilino)methyl]phenol (PIN)`), but only on a parent
                # whose own characteristic group outranks the amine -- there, a
                # phenol. With no senior group present the amine cannot be demoted.
                result['amino_substituents'].append(locant)
                for _nm in _amine[0]:
                    result['n_substituents'][_nm] += 1
            elif sub_type == 'functional':
                # Other functional groups (nitro, hydroxy, methylamino, etc.)
                # These are prefix substituents on the ring
                result['c_substituents'][sub_name].append(locant)
            elif sub_type == 'suffix':
                # Suffix-forming groups directly on ring (carboxylic acid, carbaldehyde, etc.)
                suffix_name = sub_info.get('suffix_name', sub_name)
                result['suffix_groups'][suffix_name].append(locant)
            elif sub_type == 'functionalized':
                # Functionalized chain substituents (cyanomethyl, etc.)
                result['c_substituents'][sub_name].append(locant)
            else:
                # Halogen or other
                sub_info['locant'] = locant
                sub_info['is_on_nitrogen'] = is_nitrogen
                result['other'].append(sub_info)

    # Sort C-substituent locants (handle mixed int/str like 5, '3a', '7a')
    def _locant_sort_key(loc):
        """Sort key for IUPAC locants - handles int (5) and str ('3a')."""
        if isinstance(loc, str):
            # Parse '3a' -> (3, 'a'), '7a' -> (7, 'a')
            if loc and loc[-1].isalpha():
                return (int(loc[:-1]), loc[-1])
            return (int(loc), '')
        return (loc, '')

    for name in result['c_substituents']:
        result['c_substituents'][name].sort(key=_locant_sort_key)

    # Sort oxo and amino locants
    result['oxo_substituents'].sort(key=_locant_sort_key)
    result['amino_substituents'].sort(key=_locant_sort_key)

    # Sort suffix group locants
    for name in result['suffix_groups']:
        result['suffix_groups'][name].sort(key=_locant_sort_key)

    return dict(result)


def _identify_fused_substituent(
    mol,
    start_idx: int,
    excluded: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Identify a substituent attached to a fused ring core.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        excluded: Set of atom indices to exclude (core atoms)

    Returns:
        Dict with 'name', 'type', 'atoms', or None if unrecognized
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens
    halogen_names = {
        'F': 'fluoro',
        'Cl': 'chloro',
        'Br': 'bromo',
        'I': 'iodo',
    }
    if symbol in halogen_names:
        return {
            'name': halogen_names[symbol],
            'type': 'halogen',
            'atoms': [start_idx]
        }

    # Carbon-based (alkyl) groups
    if symbol == 'C':
        # Try simple alkyl first
        alkyl = _identify_alkyl_substituent(mol, start_idx, excluded)
        if alkyl:
            return alkyl

        # Fallback: try functionalized chain (cyanomethyl, carboxymethyl, etc.)
        # This handles chains with heteroatoms that _identify_alkyl_substituent rejects
        func_chain = _identify_functionalized_substituent(mol, start_idx, excluded)
        if func_chain:
            return func_chain

        # General fallback: collect all substituent atoms and delegate to
        # universal naming (a phase gap closure; task 9 extends
        # it to RING-SYSTEM substituents — the old blanket ring guards
        # silently DROPPED the fragment, naming 2-(pyridin-2-yl)quinoline as
        # bare 'quinoline').
        # Guards kept (retargeted precisely):
        # 1. A ring that STRADDLES the core boundary means the core mapping
        # is incomplete (coumarin 3-ring case) -> not a substituent.
        # 2. Only common organic elements, no exotic (As, Se, etc.)
        # 3. Modest size (<=25 atoms)
        _ring_info = mol.GetRingInfo()
        _COMMON_ORGANIC = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
        sub_atoms = _bfs_collect_all(mol, start_idx, excluded)
        if not sub_atoms or len(sub_atoms) > 25:
            return None
        _sub_set = set(sub_atoms)
        # Guard 1: any ring spanning fragment AND core = incomplete core mapping
        for _r in _ring_info.AtomRings():
            _r_set = set(_r)
            if (_r_set & _sub_set) and (_r_set & excluded):
                return None
        if not all(
            mol.GetAtomWithIdx(idx).GetSymbol() in _COMMON_ORGANIC
            for idx in sub_atoms
        ):
            return None

        if any(_ring_info.NumAtomRings(a) > 0 for a in sub_atoms):
            # Ring-containing substituent (pyridinyl, naphthalenyl, indolyl,
            # naphthalenylmethyl,...) -> the single delegate.
            #
            # core-namer: thread the best-effort tier's ``allow_mancude``
            # into the delegate exactly as the sibling enumerator already does at
            # ``substituent_naming.py:6657-6667``. A decorated / fused ring-bearing
            # compound substituent (the pantoprazole-class half) is named only by
            # the delegate's mancude / recursive-decoration branches, which are
            # gated on ``allow_mancude=True``; the fused-heterocycle enumerator was
            # the one caller that never passed it, so such a substituent silently
            # dropped (``continue`` at ``get_fused_heterocycle_substituents:1552``).
            # ``allow_enumerator_fallback`` stays at its name-producing default
            # (True). PIN / default tier is byte-identical: ``best_effort_ctx`` is
            # False there, so ``allow_mancude`` is False, the prior call exactly.
            from .ring_substituents import name_ring_system_substituent
            _rbs_mancude = False
            try:
                from ..metrics.provenance import best_effort_ctx as _rbe_ctx
                _rbs_mancude = bool(_rbe_ctx.get())
            except Exception:
                _rbs_mancude = False
            fallback_name = name_ring_system_substituent(
                mol, sub_atoms, start_idx,
                allow_enumerator_fallback=True,
                allow_mancude=_rbs_mancude,
            )
        else:
            from ..assembly.substituent_enumerator import name_substituent
            fallback_name = name_substituent(mol, sub_atoms, start_idx)
            if fallback_name == "substituent" or (
                    fallback_name and ' ' in fallback_name):
                fallback_name = None
        # Reject generic descriptive fallbacks -- those indicate unnameable
        if fallback_name:
            return {
                'name': fallback_name,
                'type': 'functionalized',
                'atoms': list(sub_atoms),
            }
        return None

    # Oxygen groups (oxo C=O, hydroxyl -OH)
    if symbol == 'O':
        h_count = start_atom.GetTotalNumHs()
        neighbors_outside_core = [n for n in start_atom.GetNeighbors() if n.GetIdx() not in excluded]

        # Check for oxo group (=O double-bonded to ring carbon)
        if h_count == 0 and len(neighbors_outside_core) == 0:
            # Oxo group: O with no H, double-bonded to core carbon
            for bond in start_atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(start_idx)
                if other_idx in excluded:  # Bond to core atom
                    other_atom = mol.GetAtomWithIdx(other_idx)
                    if other_atom.GetSymbol() == 'C' and bond.GetBondTypeAsDouble() == 2.0:
                        return {
                            'name': 'oxo',
                            'type': 'oxo',  # Special type for suffix handling
                            'atoms': [start_idx],
                            'bond_type': 'double'
                        }

        # Hydroxyl group (-OH)
        if h_count == 1 and len(neighbors_outside_core) == 0:
            return {
                'name': 'hydroxy',
                'type': 'functional',
                'atoms': [start_idx]
            }

        # Alkoxy group (-OR): O with no H and one carbon neighbor outside core
        if h_count == 0 and len(neighbors_outside_core) == 1:
            nbr = neighbors_outside_core[0]
            if nbr.GetSymbol() == 'C':
                # Check bond to core is single (not oxo)
                is_single = True
                for bond in start_atom.GetBonds():
                    other_idx = bond.GetOtherAtomIdx(start_idx)
                    if other_idx in excluded and bond.GetBondTypeAsDouble() == 2.0:
                        is_single = False
                        break
                if is_single:
                    alkyl_atoms = _bfs_alkyl_from(mol, nbr.GetIdx(), excluded | {start_idx})
                    if alkyl_atoms is not None:
                        carbon_count = sum(1 for idx in alkyl_atoms
                                           if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')
                        # Alkoxy uses alkane stem + oxy: methoxy, ethoxy, propoxy
                        _ALKOXY = {
                            1: 'methoxy', 2: 'ethoxy', 3: 'propoxy',
                            4: 'butoxy', 5: 'pentyloxy', 6: 'hexyloxy',
                        }
                        alkoxy_name = _ALKOXY.get(carbon_count)
                        if alkoxy_name is None:
                            from ..assembly.substituent_naming import name_substituent_fragment
                            rec_name = name_substituent_fragment(
                                mol, alkyl_atoms, nbr.GetIdx(), list(excluded | {start_idx})
                            )
                            if rec_name:
                                if rec_name.endswith('yl'):
                                    # composed_alkoxy_prefix: a ring/locant-bearing
                                    # '-yl' keeps its marks ('naphthalen-1-yl' ->
                                    # '(naphthalen-1-yl)oxy', NOT 'naphthalen-1-oxy')
                                    # (F-spell-oxy).
                                    from ..assembly.substituent_enumerator import (
                                        alkoxy_prefix_from_substituent,
                                    )
                                    alkoxy_name = alkoxy_prefix_from_substituent(rec_name)
                                else:
                                    alkoxy_name = f'{rec_name}oxy'
                            if alkoxy_name is None:
                                try:
                                    alkoxy_name = f'{get_alkyl_name(carbon_count)}oxy'
                                except (ValueError, KeyError):
                                    alkoxy_name = None
                        if alkoxy_name:
                            return {
                                'name': alkoxy_name,
                                'type': 'functional',
                                'atoms': [start_idx] + alkyl_atoms
                            }

        # General -O-R delegate. Everything above needs the R group to be a plain
        # ALKYL: `_bfs_alkyl_from` returns None as soon as R contains a ring or a
        # heteroatom, and the branch then fell out of this `if symbol == 'O'`
        # block and returned None -- which SILENTLY DROPPED the whole substituent.
        # A glycosyloxy is exactly that shape (the R is a ring-and-oxygen-rich
        # sugar), so a flavonoid glycoside was named as the bare aglycone with the
        # sugar and its oxygen missing. Delegating to the substituent cascade is
        # what the CARBON branch above already does for a ring-bearing R; doing
        # the same here lets any nameable O-linked substituent be cited, including
        # the glycosyloxy prefix built by Tier 1.75.
        # Placed AFTER the alkoxy logic, so every -OR the tables already handle
        # keeps its existing name. A cascade decline still returns None (the
        # pre-existing drop), which the downstream coverage/ gates catch.
        if start_atom.GetTotalNumHs() == 0 and not start_atom.GetIsAromatic():
            _o_nbrs = [n for n in start_atom.GetNeighbors()
                       if n.GetIdx() not in excluded]
            _bonds_to_core_single = all(
                bond.GetBondTypeAsDouble() != 2.0
                for bond in start_atom.GetBonds()
                if bond.GetOtherAtomIdx(start_idx) in excluded
            )
            if len(_o_nbrs) == 1 and _bonds_to_core_single:
                _frag = _bfs_collect_all(mol, start_idx, excluded)
                if _frag and len(_frag) <= 25:
                    from ..assembly.substituent_enumerator import name_substituent
                    _oname = name_substituent(mol, sorted(_frag), start_idx)
                    if (_oname and _oname != 'substituent'
                            and ' ' not in _oname):
                        return {
                            'name': _oname,
                            'type': 'functional',
                            'atoms': sorted(_frag),
                        }

    # Nitrogen groups (amino, nitro, N-alkyl amino, etc.)
    if symbol == 'N':
        h_count = start_atom.GetTotalNumHs()
        neighbors = [n for n in start_atom.GetNeighbors() if n.GetIdx() not in excluded]

        # Nitro group: N+ with 2 oxygen neighbors
        if start_atom.GetFormalCharge() == 1:
            o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
            if o_count == 2:
                atoms = [start_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
                return {
                    'name': 'nitro',
                    'type': 'functional',
                    'atoms': atoms
                }

        # Simple amino (-NH2)
        if h_count == 2 and len(neighbors) == 0:
            return {
                'name': 'amino',
                'type': 'functional',
                'atoms': [start_idx]
            }

        # N-monoalkyl amino (-NHR) -> (alkylamino) prefix
        if h_count == 1 and len(neighbors) == 1:
            nbr = neighbors[0]
            if nbr.GetSymbol() == 'C':
                # BFS to find alkyl group size
                alkyl_atoms = _bfs_alkyl_from(mol, nbr.GetIdx(), excluded | {start_idx})
                if alkyl_atoms is not None:
                    carbon_count = sum(1 for idx in alkyl_atoms
                                       if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')
                    from ..assembly.substituent_naming import name_substituent_fragment
                    alkyl_name = name_substituent_fragment(
                        mol, alkyl_atoms, nbr.GetIdx(), list(excluded | {start_idx})
                    )
                    # Track WHERE the name came from. `get_alkyl_name` names a
                    # fragment by its CARBON COUNT, which is only meaningful for a
                    # genuine acyclic alkyl -- see the ring guard below.
                    _named_by_carbon_count = alkyl_name is None
                    if alkyl_name is None:
                        try:
                            alkyl_name = get_alkyl_name(carbon_count)
                        except (ValueError, KeyError):
                            alkyl_name = None
                    # (the Blue Book): 'anilino' is the retained PREFERRED
                    # PREFIX for C6H5-NH- with full substitution allowed, cited bare
                    # when it carries no locant of its own (the Blue Book) and enclosed
                    # when it does (the Blue Book). The legacy else-branch emitted
                    # '{ring}amino' UNENCLOSED ('4-methylphenylamino'), which is
                    # both the general-nomenclature column (the Blue Book) and malformed.
                    #
                    # Derive from the GRAPH first, not from `alkyl_name`. This site
                    # reaches `_bfs_alkyl_from`, which rejects only HETEROATOMS --
                    # an all-carbon AROMATIC ring passes it, and when
                    # `name_substituent_fragment` then declined, the
                    # `get_alkyl_name(carbon_count)` fallback below named the ring by
                    # its carbon COUNT: a quinoline bearing -NH-(4-methylphenyl)
                    # emitted 'heptylamino' (7 ring+methyl carbons) and
                    # -NH-(4-ethylphenyl) emitted 'octylamino'. Those name a
                    # DIFFERENT molecule. Pre-existing (the same else-branch shipped
                    # before P4-a) and not reached end-to-end today -- the whole-
                    # molecule pipeline abstains on these -- but it is a fabrication
                    # inside this class, so the ring is named as a ring here.
                    from .ring_substituents import (
                        anilino_preferred_prefix,
                        anilino_prefix_from_n_branch,
                    )
                    _anilino = anilino_prefix_from_n_branch(
                        mol, start_idx, [start_idx] + alkyl_atoms)
                    if _anilino is None and alkyl_name:
                        _anilino = anilino_preferred_prefix(alkyl_name)
                    if _anilino is not None:
                        return {
                            'name': _anilino,
                            'type': 'functional',
                            'atoms': [start_idx] + alkyl_atoms
                        }
                    # Refuse ONLY the carbon-count naming of a RING. The guard is
                    # deliberately narrow: an earlier, broader version keyed on
                    # "any ring atom" also refused the FUSED N-aryl case, whose real
                    # output is `naphthalen-2-ylamino` -- structurally CORRECT (only
                    # under-enclosed vs the `(naphthalen-2-yl)amino`), not a
                    # fabrication. Refusing that traded a usable name for an
                    # abstention, which is the standing invariant-11 trap ("removing
                    # a wrong output can unmask a worse one") pointing the other way.
                    # A mutation test caught it. So the condition is exactly the
                    # defect: a ring named by its carbon count.
                    if _named_by_carbon_count and any(
                            mol.GetAtomWithIdx(i).IsInRing() for i in alkyl_atoms):
                        return None
                    if alkyl_name:
                        return {
                            'name': f'{alkyl_name}amino',
                            'type': 'functional',
                            'atoms': [start_idx] + alkyl_atoms
                        }

        # N,N-dialkyl amino (-NR2) -> (dialkylamino) prefix
        if h_count == 0 and len(neighbors) == 2:
            c_neighbors = [n for n in neighbors if n.GetSymbol() == 'C']
            if len(c_neighbors) == 2:
                alkyl_names = []
                all_sub_atoms = [start_idx]
                for cn in c_neighbors:
                    alkyl_atoms = _bfs_alkyl_from(mol, cn.GetIdx(), excluded | {start_idx})
                    if alkyl_atoms is None:
                        break
                    carbon_count = sum(1 for idx in alkyl_atoms
                                       if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')
                    from ..assembly.substituent_naming import name_substituent_fragment
                    aname = name_substituent_fragment(
                        mol, alkyl_atoms, cn.GetIdx(), list(excluded | {start_idx})
                    )
                    if aname is None:
                        try:
                            aname = get_alkyl_name(carbon_count)
                        except (ValueError, KeyError):
                            break
                    alkyl_names.append(aname)
                    all_sub_atoms.extend(alkyl_atoms)
                else:
                    # Both alkyl groups identified
                    alkyl_names.sort()
                    if alkyl_names[0] == alkyl_names[1]:
                        from ..assembly.naming_utils import get_multiplier_prefix as _get_mp
                        mp = _get_mp(2, alkyl_names[0])
                        prefix_name = f'{mp}{alkyl_names[0]}amino'
                    else:
                        prefix_name = f'{alkyl_names[0]}({alkyl_names[1]}amino)'
                    return {
                        'name': prefix_name,
                        'type': 'functional',
                        'atoms': all_sub_atoms
                    }

    # Sulfur groups
    if symbol == 'S':
        h_count = start_atom.GetTotalNumHs()
        neighbors = [n for n in start_atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 1 and len(neighbors) == 0:
            return {
                'name': 'sulfanyl',
                'type': 'functional',
                'atoms': [start_idx]
            }
        # Alkylthio group (-SR)
        if h_count == 0 and len(neighbors) == 1:
            nbr = neighbors[0]
            if nbr.GetSymbol() == 'C':
                alkyl_atoms = _bfs_alkyl_from(mol, nbr.GetIdx(), excluded | {start_idx})
                if alkyl_atoms is not None:
                    carbon_count = sum(1 for idx in alkyl_atoms
                                       if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')
                    _ALKYLTHIO = {
                        1: 'methylsulfanyl', 2: 'ethylsulfanyl', 3: 'propylsulfanyl',
                    }
                    thio_name = _ALKYLTHIO.get(carbon_count)
                    if thio_name is None:
                        from ..assembly.substituent_naming import name_substituent_fragment
                        rec_name = name_substituent_fragment(
                            mol, alkyl_atoms, nbr.GetIdx(), list(excluded | {start_idx})
                        )
                        if rec_name:
                            thio_name = f'{rec_name}sulfanyl'
                        if thio_name is None:
                            try:
                                thio_name = f'{get_alkyl_name(carbon_count)}sulfanyl'
                            except (ValueError, KeyError):
                                thio_name = None
                    if thio_name:
                        return {
                            'name': thio_name,
                            'type': 'functional',
                            'atoms': [start_idx] + alkyl_atoms
                        }

    # core-namer (general ring-bearing compound substituent, best-effort
    # tier only). Every element-specific branch above returns early for the
    # shapes it recognises; a ring-bearing compound substituent rooted at a
    # HETEROATOM linker reaches here unrecognised and used to silently drop
    # (``return None`` -> caller ``continue``). The pantoprazole C2 half is
    # exactly this shape: it is rooted at the sulfinyl S (a heteroatom), so it
    # NEVER reaches the carbon branch's ring delegate above -- the S branch has
    # no ring-bearing delegate and returned None. Generalise the carbon branch's
    # move: delegate ANY ring-bearing fragment, whatever its root element, to the
    # single delegate ``name_ring_system_substituent``. At the best-effort
    # tier its name is used as is; at the PIN tier only through
    # ``promote_at_pin_tier`` (below). Both name-producing flags are passed (the delegate's
    # own guards + the downstream / RT gate keep it 0-wrong: a fragment
    # it cannot number returns None, a mis-numbering abstains -- never a wrong
    # molecule). The same guards the carbon branch uses apply: modest size, only
    # common-organic elements, and no ring straddling the parent core (which
    # would mean an incomplete core mapping, not a substituent).
    _be_ring_sub = False
    try:
        from ..metrics.provenance import best_effort_ctx as _rbe_ctx2
        _be_ring_sub = bool(_rbe_ctx2.get())
    except Exception:
        _be_ring_sub = False
    # Breadth Job 1 (M01): the same delegate runs at the PIN tier too, but there a
    # name is kept only when every token is PIN vocabulary (`promote_at_pin_tier`),
    # e.g. the pantoprazole-type '[(pyridin-2-yl)methyl]sulfinyl' on a benzimidazole.
    _ri = mol.GetRingInfo()
    _COMMON_ORGANIC2 = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
    _frag2 = _bfs_collect_all(mol, start_idx, excluded)
    if _frag2 and len(_frag2) <= 25:
        _fset2 = set(_frag2)
        _straddle = any(
            (set(_r) & _fset2) and (set(_r) & excluded)
            for _r in _ri.AtomRings()
        )
        _ring_bearing = any(_ri.NumAtomRings(a) > 0 for a in _frag2)
        _all_common = all(
            mol.GetAtomWithIdx(idx).GetSymbol() in _COMMON_ORGANIC2
            for idx in _frag2
        )
        if _ring_bearing and not _straddle and _all_common:
            from .ring_substituents import name_ring_system_substituent

            def _be_tail():
                return name_ring_system_substituent(
                    mol, sorted(_frag2), start_idx,
                    allow_enumerator_fallback=True,
                    allow_mancude=True,
                )
            if _be_ring_sub:
                _tail_name = _be_tail()
            else:
                from .pin_vocabulary import promote_at_pin_tier
                _tail_name = promote_at_pin_tier(_be_tail)
            if (_tail_name and _tail_name != 'substituent'
                    and ' ' not in _tail_name):
                return {
                    'name': _tail_name,
                    'type': 'functionalized',
                    'atoms': sorted(_frag2),
                }

    return None


def _bfs_collect_all(mol, start_idx: int, excluded: Set[int]) -> Set[int]:
    """
    Collect all atoms reachable from start_idx, excluding atoms in the excluded set.

    Unlike _bfs_alkyl_from, this collects ALL atom types (not just carbon),
    making it suitable for general-purpose substituent fragment collection.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index
        excluded: Set of atom indices to exclude (e.g., ring core atoms)

    Returns:
        Set of reachable atom indices (always includes start_idx if not in excluded)
    """
    visited = set()
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        if idx in visited or idx in excluded:
            continue
        visited.add(idx)
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx not in visited and nidx not in excluded:
                queue.append(nidx)
    return visited


def _bfs_alkyl_from(mol, start_idx: int, excluded: Set[int]) -> Optional[List[int]]:
    """
    BFS to collect a pure alkyl substituent (only C and H atoms).

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index
        excluded: Set of atom indices to exclude

    Returns:
        List of atom indices if pure alkyl, None if contains heteroatoms
    """
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() not in ('C', 'H'):
            return None  # Not pure alkyl

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return all_atoms if all_atoms else None


def _identify_alkyl_substituent(
    mol,
    start_idx: int,
    excluded: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Identify an alkyl substituent using BFS.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index
        excluded: Set of atom indices to exclude

    Returns:
        Dict with 'name', 'type', 'atoms', or None
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []
    carbon_count = 0

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if pure alkyl (only C and H)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            return None  # Contains heteroatom, not simple alkyl

    if carbon_count == 0:
        return None

    # Get alkyl name -- try name_substituent_fragment (handles retained + branched)
    from ..assembly.substituent_naming import name_substituent_fragment
    recursive_name = name_substituent_fragment(
        mol, all_atoms, start_idx, list(excluded)
    )
    if recursive_name:
        return {
            'name': recursive_name,
            'type': 'alkyl',
            'atoms': all_atoms
        }

    try:
        alkyl_name = get_alkyl_name(carbon_count)
        return {
            'name': alkyl_name,
            'type': 'alkyl',
            'atoms': all_atoms
        }
    except ValueError:
        return None


def _chain_is_unbranched_alkanoyl(mol, chain_atoms, acyl_c: int) -> bool:
    """True when ``chain_atoms`` is exactly CH3-(CH2)n-C(=O)- with ``acyl_c`` the
    carbonyl carbon: one oxygen, double-bonded to ``acyl_c``; every other atom an
    acyclic sp3 carbon; an unbranched, singly bonded chain. The only shape an
    'acetyl'/'propanoyl' carbon count describes."""
    chain = set(chain_atoms)
    oxygens = 0
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        if atom.IsInRing():
            return False
        if atom.GetAtomicNum() == 8:
            bond = mol.GetBondBetweenAtoms(idx, acyl_c)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                return False
            oxygens += 1
            continue
        if atom.GetAtomicNum() != 6:
            return False
        chain_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in chain]
        carbon_nbrs = [n for n in chain_nbrs
                       if mol.GetAtomWithIdx(n).GetAtomicNum() == 6]
        if len(carbon_nbrs) > (1 if idx == acyl_c else 2):
            return False
        for n in carbon_nbrs:
            if mol.GetBondBetweenAtoms(idx, n).GetBondType() != Chem.BondType.SINGLE:
                return False
    return oxygens == 1


def _identify_functionalized_substituent(
    mol,
    start_idx: int,
    excluded: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Identify functionalized chain substituents like -CH2-C#N (cyanomethyl).

    Unlike _identify_alkyl_substituent which rejects heteroatoms,
    this function recognizes common functional groups at chain termini.

    Handles:
    - Nitrile (C#N): cyanomethyl, 2-cyanoethyl, etc.
    - Carboxylic acid (COOH): carboxymethyl, 2-carboxyethyl, etc.
    - Aldehyde (CHO): formylmethyl, etc.

    Args:
        mol: RDKit Mol object
        start_idx: Index of first atom (attached to ring)
        excluded: Ring atom indices to exclude

    Returns:
        Dict with 'name', 'atoms', 'functional_group', 'type' or None

    IUPAC Reference: (acetic acid derivatives as substituents),
                      (naming fused ring substituents)
    """
    # BFS to collect substituent atoms
    chain_atoms = []
    visited = {start_idx}
    queue = deque([start_idx])

    while queue:
        idx = queue.popleft()
        chain_atoms.append(idx)
        atom = mol.GetAtomWithIdx(idx)

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    if not chain_atoms:
        return None

    # Count carbons in chain
    carbon_count = sum(1 for idx in chain_atoms
                       if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')

    # A -C(=O)NH2 or -C#N bonded directly to the ring is a SUFFIX group, like
    # the -COOH / -CHO below: (the Blue Book) "The suffix
    # 'carboxamide' is always used to name amides with the -CO-NH2 group
    # attached to a ring, ring system, or to a heteroacyclic parent", and
    # (:34720) "The suffix 'carbonitrile' is always used to name
    # nitriles having the -CN group attached to a ring or ring system". Both
    # used to reach the general substituent namer
    # and ship as the 'carbamoyl' / 'cyano' PREFIX with no suffix at all:
    # '3-cyanoquinoline' (PIN 'quinoline-3-carbonitrile'), '5-carbamoyl-1H-
    # indole' (PIN '1H-indole-5-carboxamide'), at pin_verified.
    # (:18162) ranks amides (11) and nitriles (14) above ketones (16), so they
    # also take the suffix over a ring -one.
    if carbon_count == 1 and len(chain_atoms) in (2, 3):
        c_atom = mol.GetAtomWithIdx(start_idx)
        others = [mol.GetAtomWithIdx(i) for i in chain_atoms if i != start_idx]
        if (c_atom.GetSymbol() == 'C' and c_atom.GetFormalCharge() == 0
                and all(o.GetFormalCharge() == 0 and o.GetDegree() == 1
                        for o in others)):
            bonds = {o.GetSymbol(): mol.GetBondBetweenAtoms(start_idx, o.GetIdx())
                     for o in others}
            if (len(others) == 1 and 'N' in bonds
                    and bonds['N'].GetBondType() == Chem.BondType.TRIPLE):
                # 'name' is the PREFIX form: ring_substituents cites it as a
                # decoration of a ring substituent ('3-cyanopyridin-2-yl');
                # 'suffix_name' is what the parent ring's suffix uses
                return {
                    'name': 'cyano',
                    'atoms': chain_atoms,
                    'functional_group': 'nitrile',
                    'type': 'suffix',
                    'suffix_name': 'carbonitrile',
                }
            if (len(others) == 2 and set(bonds) == {'N', 'O'}
                    and bonds['O'].GetBondType() == Chem.BondType.DOUBLE
                    and bonds['N'].GetBondType() == Chem.BondType.SINGLE
                    and next(o for o in others if o.GetSymbol() == 'N')
                    .GetTotalNumHs() == 2):
                return {
                    'name': 'carbamoyl',
                    'atoms': chain_atoms,
                    'functional_group': 'amide',
                    'type': 'suffix',
                    'suffix_name': 'carboxamide',
                }

    # Check for nitrile terminus (N with triple bond to C)
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in chain_atoms:
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.TRIPLE:
                        # Found nitrile: -C#N
                        # carbon_count includes the nitrile carbon
                        if carbon_count == 2:
                            return {
                                'name': 'cyanomethyl',
                                'atoms': chain_atoms,
                                'functional_group': 'nitrile',
                                'type': 'functionalized'
                            }
                        elif carbon_count == 3:
                            return {
                                'name': '2-cyanoethyl',
                                'atoms': chain_atoms,
                                'functional_group': 'nitrile',
                                'type': 'functionalized'
                            }
                        elif carbon_count == 4:
                            return {
                                'name': '3-cyanopropyl',
                                'atoms': chain_atoms,
                                'functional_group': 'nitrile',
                                'type': 'functionalized'
                            }
                        # For longer chains, use generic pattern
                        elif carbon_count > 4:
                            return {
                                'name': f'{carbon_count - 1}-cyano{get_alkyl_name(carbon_count - 1)}',
                                'atoms': chain_atoms,
                                'functional_group': 'nitrile',
                                'type': 'functionalized'
                            }

    # Check for carboxylic acid terminus: C(=O)[OH] or C(=O)[O-]
    acid_pattern = _compiled_smarts('[CX3](=O)[OX2H1,OX1-]')
    if acid_pattern:
        matches = mol.GetSubstructMatches(acid_pattern)
        for match in matches:
            carboxyl_carbon = match[0]
            if carboxyl_carbon in chain_atoms:
                # Carbon count includes the carboxyl carbon
                if carbon_count == 1:
                    # COOH directly on ring → suffix-type "carboxylic acid"
                    return {
                        'name': 'carboxylic acid',
                        'atoms': chain_atoms,
                        'functional_group': 'carboxylic_acid',
                        'type': 'suffix',
                        'suffix_name': 'carboxylic acid',
                    }
                elif carbon_count == 2:
                    return {
                        'name': 'carboxymethyl',
                        'atoms': chain_atoms,
                        'functional_group': 'carboxylic_acid',
                        'type': 'functionalized'
                    }
                elif carbon_count == 3:
                    return {
                        'name': '2-carboxyethyl',
                        'atoms': chain_atoms,
                        'functional_group': 'carboxylic_acid',
                        'type': 'functionalized'
                    }
                elif carbon_count == 4:
                    return {
                        'name': '3-carboxypropyl',
                        'atoms': chain_atoms,
                        'functional_group': 'carboxylic_acid',
                        'type': 'functionalized'
                    }
                elif carbon_count > 4:
                    return {
                        'name': f'{carbon_count - 1}-carboxy{get_alkyl_name(carbon_count - 1)}',
                        'atoms': chain_atoms,
                        'functional_group': 'carboxylic_acid',
                        'type': 'functionalized'
                    }

    # Check for aldehyde terminus: [CH]=O
    aldehyde_pattern = _compiled_smarts('[CX3H1](=O)')
    if aldehyde_pattern:
        matches = mol.GetSubstructMatches(aldehyde_pattern)
        for match in matches:
            aldehyde_carbon = match[0]
            if aldehyde_carbon in chain_atoms:
                if carbon_count == 1:
                    # CHO directly on ring → suffix-type "carbaldehyde"
                    return {
                        'name': 'carbaldehyde',
                        'atoms': chain_atoms,
                        'functional_group': 'aldehyde',
                        'type': 'suffix',
                        'suffix_name': 'carbaldehyde',
                    }
                elif carbon_count == 2:
                    return {
                        'name': 'acetyl',
                        'atoms': chain_atoms,
                        'functional_group': 'aldehyde',
                        'type': 'functionalized'
                    }

    # Check for hydroxyl terminus: -CH2-OH or -CH(-OH)-
    # Handles hydroxymethyl (-CH2OH), 2-hydroxyethyl (-CH2CH2OH), etc.
    hydroxyl_pattern = _compiled_smarts('[OX2H1]')
    if hydroxyl_pattern:
        matches = mol.GetSubstructMatches(hydroxyl_pattern)
        for match in matches:
            o_idx = match[0]
            if o_idx in chain_atoms:
                if carbon_count == 1:
                    return {
                        'name': 'hydroxymethyl',
                        'atoms': chain_atoms,
                        'functional_group': 'alcohol',
                        'type': 'functionalized'
                    }
                elif carbon_count == 2:
                    return {
                        'name': '2-hydroxyethyl',
                        'atoms': chain_atoms,
                        'functional_group': 'alcohol',
                        'type': 'functionalized'
                    }
                elif carbon_count == 3:
                    return {
                        'name': '3-hydroxypropyl',
                        'atoms': chain_atoms,
                        'functional_group': 'alcohol',
                        'type': 'functionalized'
                    }

    # Check for amino terminus: -CH2-NH2, -CH2-CH2-NH2, etc.
    # Handles aminomethyl (-CH2NH2), 2-aminoethyl (-CH2CH2NH2), etc.
    # Only match when the chain between ring and NH2 is purely carbon
    # (avoids misidentifying complex chains through ribose/phosphate as "aminoalkyl")
    amino_pattern = _compiled_smarts('[NX3H2;!$([NX3H2][CX3]=O)]')
    if amino_pattern:
        matches = mol.GetSubstructMatches(amino_pattern)
        for match in matches:
            n_idx = match[0]
            if n_idx in chain_atoms:
                # Verify chain is purely carbon + the terminal N (no intermediate heteroatoms)
                non_cn_count = sum(1 for idx in chain_atoms
                                   if mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'N'))
                if non_cn_count > 0:
                    continue  # Complex chain with O, S, P etc. — skip
                if carbon_count == 1:
                    return {
                        'name': 'aminomethyl',
                        'atoms': chain_atoms,
                        'functional_group': 'amine',
                        'type': 'functionalized'
                    }
                elif carbon_count == 2:
                    return {
                        'name': '2-aminoethyl',
                        'atoms': chain_atoms,
                        'functional_group': 'amine',
                        'type': 'functionalized'
                    }
                elif carbon_count == 3:
                    return {
                        'name': '3-aminopropyl',
                        'atoms': chain_atoms,
                        'functional_group': 'amine',
                        'type': 'functionalized'
                    }
                elif carbon_count > 3:
                    try:
                        return {
                            'name': f'{carbon_count}-amino{get_alkyl_name(carbon_count)}',
                            'atoms': chain_atoms,
                            'functional_group': 'amine',
                            'type': 'functionalized'
                        }
                    except (ValueError, KeyError):
                        pass

    # Check for acetyl/acyl terminus: -C(=O)-R on ring
    # Handles acetyl (-C(=O)-CH3), etc.
    acyl_pattern = _compiled_smarts('[CX3](=O)[#6]')
    if acyl_pattern:
        matches = mol.GetSubstructMatches(acyl_pattern)
        for match in matches:
            acyl_c = match[0]
            if acyl_c in chain_atoms and acyl_c == start_idx:
                # Acyl group directly on ring. 2026-09-25 (pre-existing-failures
                # plan, Task 4 continuation): the SMARTS' [#6] also matches the
                # RING carbon, so an ester -C(=O)-O-CH3 (two carbons) was named
                # 'acetyl' -- '3-acetyl-1H-indole' for methyl
                # 1H-indole-3-carboxylate, a different molecule. 'acetyl' is
                # CH3-CO- (the Blue Book, "acyl groups are formed by
                # subtracting all -OH groups from oxoacids for example 'acetyl',
                # CH3-CO-"), so the branch now requires exactly that shape.
                if not _chain_is_unbranched_alkanoyl(mol, chain_atoms, acyl_c):
                    continue
                if carbon_count == 2:
                    return {
                        'name': 'acetyl',
                        'atoms': chain_atoms,
                        'functional_group': 'ketone',
                        'type': 'functionalized'
                    }
                elif carbon_count == 3:
                    return {
                        'name': 'propanoyl',
                        'atoms': chain_atoms,
                        'functional_group': 'ketone',
                        'type': 'functionalized'
                    }

    # An ester -C(=O)-O-R directly on the ring is NOT recognised here.
    # 2026-09-25 (pre-existing-failures plan, Task 4 continuation): this branch
    # returned the suffix 'carboxylic acid' with an 'ester_alkyl' word that no
    # assembler reads, so methyl 1H-indole-3-carboxylate came out as
    # '1H-indole-3-carboxylic acid', a different molecule. Declining here hands
    # the group to the general substituent namer ('methoxycarbonyl', the prefix
    # form of,:31696) or, when that declines, to the decomposition
    # route, which builds the functional-class PIN form ('ethyl
    # 1H-indole-3-carboxylate';,:31663, "All preferred IUPAC names
    # for esters are named by functional class nomenclature").
    return None


# (the Blue Book): class 17 "Hydroxy compounds and chalcogen
# analogues (includes alcohols and phenols)". These are the alcohol-class subtypes
# in seniority.SENIORITY_ORDER; a molecule whose principal characteristic group is
# one of them has NO group senior to its hydroxy.
_ALCOHOL_CLASS_PCG = frozenset({
    "primary_alcohol", "secondary_alcohol", "tertiary_alcohol",
    "phenol", "enol", "alcohol",
})


def _has_group_senior_to_hydroxy(mol) -> bool:
    """True when the molecule carries a characteristic group SENIOR to hydroxy.

    /: only the senior-most characteristic group is the principal one
    (the suffix). ``get_principal_group`` returns exactly that group over the whole
    molecule using the shared ``SENIORITY_ORDER`` table, so if it returns anything
    other than an alcohol-class group that OUT-ranks hydroxy (class 17 -- amides 11,
    nitriles 14, aldehydes 15, ketones 16, acids/esters/acyl-halides higher), the
    hydroxy cannot be promoted to the ``-ol`` suffix. A group junior to hydroxy
    (amine, thiol,...) or no group returns False -- hydroxy stays eligible.
    """
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import get_principal_group, compare_seniority

    pcg_name, _ = get_principal_group(mol, detect_functional_groups(mol))
    if pcg_name is None or pcg_name in _ALCOHOL_CLASS_PCG:
        return False
    # compare_seniority(pcg, 'alcohol') < 0 <=> pcg ranks above the alcohol tier.
    return compare_seniority(pcg_name, "alcohol") < 0


# Priority of the detachable ring suffixes in _assemble_fused_heterocycle_name:
# (the Blue Book) "7 Acids" (:18172) > "11 Amides" (:18184)
# > "14 Nitriles" (:18187) > "15 Aldehydes" (:18188). (The list had the aldehyde
# second; that never mattered while amides and nitriles were never suffixes.)
_FUSED_SUFFIX_PRIORITY = ['carboxylic acid', 'carboxamide', 'carbonitrile', 'carbaldehyde']


def _fused_principal_locants(mol, substituents: Dict) -> List:
    """Locants of the group _assemble_fused_heterocycle_name cites as the SUFFIX.

    Mirrors that assembler's own choice, in its order: a detachable suffix group
    (by _FUSED_SUFFIX_PRIORITY), else amino / oxo (_build_fused_suffix), else a
    hydroxy that becomes '-ol'. Used only to number the core so the suffix gets
    the lowest locants (c)).
    """
    suffix_groups = substituents.get('suffix_groups') or {}
    if suffix_groups:
        for suf in _FUSED_SUFFIX_PRIORITY:
            if suf in suffix_groups:
                return list(suffix_groups[suf])
        return list(next(iter(suffix_groups.values())))
    amino = substituents.get('amino_substituents') or []
    oxo = substituents.get('oxo_substituents') or []
    c_subs = substituents.get('c_substituents') or {}
    hydroxy_principal = ('hydroxy' in c_subs and not oxo
                         and not _has_group_senior_to_hydroxy(mol))
    if amino and not hydroxy_principal:
        return list(amino)
    if oxo:
        return list(oxo)
    if hydroxy_principal:
        return list(c_subs['hydroxy'])
    return []


def _renumber_core_for_suffix(mol, core_name, core_smiles, atom_mapping, substituents):
    """ "NUMBERING" (the Blue Book): "(c) principal characteristic
    groups and free valences (suffixes)" (:3256) take low locants BEFORE the
    detachable prefixes "(f)" (:3301). The catalog matcher chose the core's
    automorphism by the prefix-and-suffix locant set alone, so CC1NC(=O)c2ccccc21
    got '1-methyl-2,3-dihydro-1H-isoindol-3-one' (both sets {1,3}; methyl < oxo
    alphabetically) where the PIN is '3-methyl-2,3-dihydro-1H-isoindol-1-one',
    cf. '3-imino-2,3-dihydro-1H-isoindol-1-one (PIN)' (:29609).

    Returns ``(core_name, atom_mapping, substituents)`` for the numbering whose
    suffix locants are lowest; among those the matcher's own criteria decide
    (the current numbering is kept whenever it is already among them)."""
    from ..data.fused_heterocycles import (
        _coerce_locant_for_compare, core_numberings, select_lowest_locant_match)
    from .locants import compare_locant_sets

    def _key(subs):
        locs = [_coerce_locant_for_compare(x) for x in _fused_principal_locants(mol, subs)]
        if any(x is None for x in locs):
            return None
        return sorted(locs, key=lambda v: (v, '') if isinstance(v, int) else v)

    current_key = _key(substituents)
    if not current_key:
        return core_name, atom_mapping, substituents
    options = []
    for match, mapping, name in core_numberings(mol, core_smiles, atom_mapping):
        subs = get_fused_heterocycle_substituents(mol, mapping)
        key = _key(subs)
        if key is None or len(key) != len(current_key):
            continue
        options.append((key, match, mapping, name, subs))
    # (b) precedes (c) (the Blue Book,:3256; '2H-pyran-6-carboxylic
    # acid (PIN)'): the indicated hydrogen the input carries takes the lowest
    # locant first, so only the numberings that give it that locant compete on
    # the suffix -- '1H-benzimidazol-6-ol', not '3H-benzimidazol-5-ol'. Applied
    # when every numbering places exactly one indicated hydrogen and they differ
    # (the matcher's own tier, ``_select_lowest_locant_match``).
    from ..data.fused_heterocycles import _input_indicated_h_locants
    _cur_ih = _input_indicated_h_locants(mol, atom_mapping)
    _ihs = [_input_indicated_h_locants(mol, opt[2]) for opt in options]
    if (_cur_ih is not None and len(_cur_ih) == 1
            and all(ih is not None and len(ih) == 1 for ih in _ihs)
            and len({tuple(ih) for ih in _ihs} | {tuple(_cur_ih)}) > 1):
        _low = min([_cur_ih] + _ihs)
        if _cur_ih != _low:
            # the matcher already prefers the lowest indicated hydrogen, so this
            # is reached only by a numbering it could not see; keep it as it was
            return core_name, atom_mapping, substituents
        options = [opt for opt, ih in zip(options, _ihs) if ih == _low]
    if len(options) < 2:
        return core_name, atom_mapping, substituents
    best_key = options[0][0]
    for opt in options[1:]:
        if compare_locant_sets(opt[0], best_key) < 0:
            best_key = opt[0]
    if compare_locant_sets(best_key, current_key) >= 0:
        return core_name, atom_mapping, substituents
    tied = [opt for opt in options if compare_locant_sets(opt[0], best_key) == 0]
    chosen = select_lowest_locant_match(mol, [opt[1] for opt in tied], core_smiles)
    for _key_, match, mapping, name, subs in tied:
        if match == list(chosen):
            return name, mapping, subs
    return core_name, atom_mapping, substituents


# The class each fused suffix expresses, as a seniority.SENIORITY_ORDER entry
#, the Blue Book): 7 acids, 11 amides, 14 nitriles,
# 15 aldehydes, 16 ketones/pseudoketones, 17 hydroxy compounds, 19 amines. For
# '-ol' / '-amine' the FIRST entry of the class is used, so any group of the
# same class never counts as senior.
_SUFFIX_EXPRESSED_CLASS = {
    'carboxylic acid': 'carboxylic_acid',
    'carboxamide': 'primary_amide',
    'carbonitrile': 'nitrile',
    'carbaldehyde': 'aldehyde',
    'one': 'ketone',
    'ol': 'primary_alcohol',
    'amine': 'hydroxylamine',
}


def _senior_group_outside_ring(mol, ring_atoms: Set[int]) -> Optional[str]:
    """The most senior suffix-capable characteristic group of the molecule that
    is not part of the ring system itself (seniority.SENIORITY_ORDER name), or
    None. A lactone / lactam / cyclic imide whose characteristic atoms (its
    heteroatoms and its C=O carbon) are all ring atoms or ring C=O oxygens is a
    pseudoketone named with '-one' class 16,:18189), so it is not counted
    even though the perception module labels it 'ester' / 'imide' /
    'tertiary_amide' (phthalide, N-methylphthalimide, a 1-methylquinoline-2,4-
    dione, whose amide match also lists the N-methyl carbon)."""
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import SENIORITY_ORDER, _PREFIX_ONLY_PRINCIPAL
    inside = set(ring_atoms)
    for a in ring_atoms:
        at = mol.GetAtomWithIdx(a)
        for b in at.GetBonds():
            o = b.GetOtherAtom(at)
            if (b.GetBondType() == Chem.BondType.DOUBLE and o.GetSymbol() == 'O'
                    and o.GetDegree() == 1):
                inside.add(o.GetIdx())

    def _plain_carbon(idx: int) -> bool:
        # a carbon of an R group: no double / triple bond to a heteroatom
        at = mol.GetAtomWithIdx(idx)
        return at.GetSymbol() == 'C' and not any(
            b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            and b.GetOtherAtom(at).GetSymbol() != 'C' for b in at.GetBonds())

    fgs = detect_functional_groups(mol)
    for fg in SENIORITY_ORDER:
        if fg in _PREFIX_ONLY_PRINCIPAL:
            continue
        for inst in fgs.get(fg) or ():
            atoms = set(inst) if isinstance(inst, (tuple, list, set, frozenset)) else {inst}
            if any(a not in inside and not _plain_carbon(a) for a in atoms):
                return fg
    return None


def _name_cites_senior_group_as_prefix(mol, ring_atoms: Set[int], expressed: Optional[str]) -> bool:
    """True when a characteristic group SENIOR to the one the name expresses as
    its suffix (``expressed``: a key of _SUFFIX_EXPRESSED_CLASS, or None for a
    suffix-free name) is present, so the name cites the principal
    characteristic group as a prefix,. Such a name can round-
    trip, but it is not a PIN."""
    from .seniority import compare_seniority
    senior = _senior_group_outside_ring(mol, ring_atoms)
    if senior is None:
        return False
    if expressed is None:
        return True
    return compare_seniority(senior, _SUFFIX_EXPRESSED_CLASS[expressed]) < 0


def _assemble_oxo_prefix_name(mol, ring_atoms: Set[int]) -> Optional[str]:
    """A ring C=O cited as the 'oxo' PREFIX because a senior group takes the
    suffix: '4-oxo-4H-1-benzopyran-2-carboxylic acid', '1-ethyl-4-oxo-1,4-
    dihydroquinoline-3-carboxylic acid', '9-oxo-9H-xanthene-2-carboxylic acid'.

     (the Blue Book): acids (7a,:18172), amides (11),
    nitriles (14) and aldehydes (15) are senior to ketones and pseudoketones
    (16,:18189), so the ring C=O cannot keep its '-one'. (:24862):
    "After the introduction of indicated and 'added indicated hydrogen' atoms,
    all substituent groups not expressed as suffixes are cited as prefixes";
    the parent hydride (indicated hydrogen + hydro prefixes,:24639,
     comes from partial_saturation.oxo_prefix_parent_numberings.
    (PIN) pattern: '9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid'
    (:29471), '5,8-dioxo-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid'
    (:24890), '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium' (:41447).

    The assembler glued the two suffixes instead ('benzo[b]pyran-4-one2-
    carboxylic acid' at pin_verified, parsed only leniently; the NH quinolone
    'quinolin-4-one3-carboxylic acid' parsed to a different tautomer), or put a
    senior suffix on a ketone core name ('xanthone-2-carboxylic acid',
    unparseable; '2H-1-benzopyran-2-one-3-carboxylic acid').

    Numbering, (:3219): (b) indicated hydrogen, (c) the suffix, (e)
    hydro prefixes, (f) all detachable prefixes together, (g) the prefix cited
    first. Returns None (fail closed) when the parent does not resolve, the
    suffix sits on a fusion atom (it would need added indicated hydrogen), an
    exocyclic amine carries N-substituents, or two numberings still tie with
    different names.
    """
    from .partial_saturation import oxo_prefix_parent_numberings, _locant_key

    ring_atoms = set(ring_atoms)
    if not _exocyclic_atoms_accounted(mol, ring_atoms):
        return None  # a branch no prefix can name would be dropped
    options = []
    for parent, loc, ih_atoms, hydro in oxo_prefix_parent_numberings(mol, ring_atoms):
        subs = get_fused_heterocycle_substituents(mol, loc)
        suffix_groups = subs.get('suffix_groups') or {}
        if not suffix_groups or not subs.get('oxo_substituents'):
            return None  # no senior suffix / no ring C=O: not this builder's case
        chosen = next((s for s in _FUSED_SUFFIX_PRIORITY if s in suffix_groups),
                      next(iter(suffix_groups)))
        suf_locs = list(suffix_groups[chosen])
        if any(not isinstance(l, int) for l in suf_locs):
            continue
        amino = list(subs.get('amino_substituents') or [])
        if amino and subs.get('n_substituents'):
            return None  # an N-substituted amine prefix is not built here
        c_subs = {k: list(v) for k, v in (subs.get('c_substituents') or {}).items()}
        c_subs['oxo'] = list(subs['oxo_substituents'])
        if amino:
            c_subs.setdefault('amino', []).extend(amino)
        new_subs = dict(subs)
        new_subs['c_substituents'] = c_subs
        new_subs['oxo_substituents'] = []
        new_subs['amino_substituents'] = []
        prefixes = []  # (alpha key, locants) for (f)/(g)
        for nm, locs in c_subs.items():
            prefixes.append((alpha_sort_key(nm), sorted(locs, key=_locant_key)))
        for other in subs.get('other') or []:
            prefixes.append((alpha_sort_key(other['name']), [other['locant']]))
        for suf, locs in suffix_groups.items():
            if suf != chosen:
                prefixes.append((alpha_sort_key(suf), sorted(locs, key=_locant_key)))
        all_prefix_locs = sorted((_locant_key(l) for _, ls in prefixes for l in ls))
        citation = tuple(tuple(_locant_key(l) for l in ls)
                         for _, ls in sorted(prefixes, key=lambda p: p[0]))
        key = (sorted(_locant_key(loc[a]) for a in ih_atoms),
               sorted(_locant_key(l) for l in suf_locs),
               sorted(_locant_key(loc[a]) for a in hydro),
               all_prefix_locs,
               citation)
        options.append((key, parent, loc, new_subs))
    if not options:
        return None
    best_key = min(o[0] for o in options)
    names = {_assemble_fused_heterocycle_name(mol, parent, new_subs, loc)
             for key, parent, loc, new_subs in options if key == best_key}
    if len(names) != 1 or None in names:
        return None
    return names.pop()


def _assemble_fused_heterocycle_name(
    mol,
    core_name: str,
    substituents: Dict,
    atom_mapping: Dict[int, int]
) -> str:
    """
    Assemble the complete name for a substituted fused heterocycle.

    Handles suffix-forming groups (oxo -> -one, amino -> -amine) as well as
    prefix substituents. For IUPAC 2013 PIN style, functional groups like
    amino and oxo use suffix naming when they are the principal characteristic
    group.

    Args:
        mol: RDKit Mol object
        core_name: Base name (e.g., '1H-indole', '9H-purine')
        substituents: Dict from get_fused_heterocycle_substituents
        atom_mapping: Dict mapping mol atom indices to IUPAC locants

    Returns:
        Complete IUPAC name string

    Examples:
        Adenine: '7H-purin-6-amine'
        Hypoxanthine: '7H-purin-6-one'
        Xanthine: '7H-purine-2,6-dione'
    """
    # Collect stereodescriptors using the fused ring locant mapping
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    assign_stereochemistry(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_mapping)
    stereo_prefix = format_stereodescriptor_string(stereo_descriptors) if stereo_descriptors else ""

    # (the Blue Book) routing of the principal characteristic
    # group before any suffix is written:
    # * a senior suffix group (acid / amide / nitrile / aldehyde) with a ring
    # C=O -- an oxo substituent, or the =O of a ketone catalog core such as
    # 'xanthone' / '2H-1-benzopyran-2-one' -- makes the C=O an 'oxo' prefix on
    # the indicated-hydrogen/hydro parent (_assemble_oxo_prefix_name). The
    # glued '-one' + '-carboxylic acid' ('benzo[b]pyran-4-one2-carboxylic
    # acid') is never written again; None when that parent cannot be built.
    # * an amine with a ring C=O: ketones (16) are senior to amines (19), but
    # _build_fused_suffix gave the amine the suffix and dropped the oxo (an atom
    # drop). Fail closed.
    # * an amine beside a senior suffix group or a principal hydroxy is an
    # 'amino' prefix: acids (7) and hydroxy compounds (17) are senior to amines
    # (19) -- '2-aminoquinoline-3-carboxylic acid', not the unparseable
    # 'quinolin-2-amine3-carboxylic acid'.
    ring_atoms = {a for a in atom_mapping if mol.GetAtomWithIdx(a).IsInRing()}
    suffix_groups0 = substituents.get('suffix_groups') or {}
    oxo0 = substituents.get('oxo_substituents') or []
    amino0 = substituents.get('amino_substituents') or []
    ketone_core = any(not mol.GetAtomWithIdx(a).IsInRing()
                      and mol.GetAtomWithIdx(a).GetSymbol() == 'O'
                      for a in atom_mapping)
    if suffix_groups0 and (oxo0 or ketone_core):
        return _assemble_oxo_prefix_name(mol, ring_atoms)
    if amino0 and (oxo0 or ketone_core):
        return None
    if amino0:
        c_subs0 = substituents.get('c_substituents') or {}
        hydroxy_principal = ('hydroxy' in c_subs0
                             and not _has_group_senior_to_hydroxy(mol))
        if suffix_groups0 or hydroxy_principal:
            if substituents.get('n_substituents'):
                return None  # the N-substituted amine would need a compound prefix
            substituents = dict(substituents)
            c_new = {k: list(v) for k, v in c_subs0.items()}
            c_new.setdefault('amino', []).extend(amino0)
            substituents['c_substituents'] = c_new
            substituents['amino_substituents'] = []

    # /: a hydroxy is the PRINCIPAL characteristic group -- and so the
    # `-ol` SUFFIX, not a `hydroxy-` prefix -- when no senior suffixable group is
    # present on the fused parent. Table 4.1 (the Blue Book "Table 4.1 General
    # compound classes listed in decreasing order of seniority") ranks, in
    # decreasing seniority, 11 amides (:18184), 14 nitriles (:18187), 15 aldehydes
    # (:18188), 16 ketones (:18189), then 17 hydroxy compounds (:18190). So when
    # hydroxy is present with none of {suffix_groups, oxo, amino} it is the senior
    # group and must hold the suffix: `quinolin-8-ol`, not `8-hydroxyquinoline`
    # (both are the same molecule; the PIN is the suffix form, the Blue Book).
    # `_build_fused_suffix` only ever handled oxo/amino, so this class shipped no
    # suffix at all. (Combined hydroxy+amino, where the alcohol should demote the
    # amine, is left to the existing amino-suffix path and not addressed here.)
    #
    # The `not oxo`/`not amino`/`not suffix_groups` guard is NOT sufficient: a
    # carbamoyl / cyano / acyl group DIRECTLY on the ring arrives as a
    # `c_substituents` PREFIX string (the fused collector routes it through the
    # general substituent-namer, never into `suffix_groups`), so those gates do not
    # see it and the promotion over-fired -- `5-carbamoylquinolin-8-ol` wrongly
    # asserted hydroxy as the principal group over a senior carboxamide class
    # 11 > 17). Consult the shared seniority table over the WHOLE molecule: promote
    # only when the molecule's principal characteristic group is itself an
    # alcohol-class group (i.e. no group senior to hydroxy is present anywhere). The
    # PIN then names the carboxamide/nitrile as the suffix on its own parent, or the
    # name degrades to the all-prefix form -- never a wrong principal group.
    hydroxy_ol_suffix = ""
    c_subs = substituents.get('c_substituents', {})
    if ('hydroxy' in c_subs
            and not substituents.get('suffix_groups')
            and not substituents.get('oxo_substituents')
            and not substituents.get('amino_substituents')
            and not _has_group_senior_to_hydroxy(mol)):
        hydroxy_locants = list(c_subs['hydroxy'])
        # hydroxy becomes the suffix, so drop it from the prefix set
        substituents = dict(substituents)
        substituents['c_substituents'] = {
            k: v for k, v in c_subs.items() if k != 'hydroxy'
        }
        hydroxy_ol_suffix = _format_suffix('ol', hydroxy_locants)

    prefix_parts = []

    # Each source's groups, in the order they were cited: the N-substituents of
    # the amine suffix ('N' per substituent), the ring C-substituents and the
    # other (halogen etc.) ring substituents.
    other_groups = defaultdict(list)
    for sub in substituents['other']:
        other_groups[sub['name']].append(sub['locant'])
    _sources = []
    for name, count in substituents['n_substituents'].items():
        _sources.append(('n', name, ["N"] * count))
    for name, locants in substituents['c_substituents'].items():
        _sources.append(('c', name, list(locants)))
    for name, locants in other_groups.items():
        # Ensure all locants are strings for consistent sorting
        # (some may be int, others str like '3a')
        locants = [str(l) for l in locants]
        locants.sort(key=lambda x: (len(x), x))
        _sources.append(('o', name, locants))

    # (b) (the Blue Book) / (a) (:7104): identical prefixes
    # are one multiplied group whatever atom carries them, with the locants in the
    # order (:3195, italic letters before numerals) -- '*N*,4-dimethyl-*N*-
    # (3-methylphenyl)benzamide (PIN)' (:32879), '*N*,1,4-triphenyl-1*H*-1,2,4-
    # triazol-4-ium-3-aminide (PIN)' (:42460). The N-substituent of the amine
    # suffix and a ring substituent of the same name were two prefixes ('3-chloro-
    # N-methyl-2-methyl-N-phenyl-2H-indazol-6-amine'); they are now one group
    # ('3-chloro-N,2-dimethyl-N-phenyl-2H-indazol-6-amine'). A name cited by one
    # source only keeps its formatting byte for byte.
    from ..assembly.composition_primitives import (
        combine_identical_prefix_groups,
        identical_prefixes_grouped,
    )
    _kinds = defaultdict(set)
    for kind, name, _locs in _sources:
        _kinds[name].add(kind)
    _single = {name: (kind, locs) for kind, name, locs in _sources
               if len(_kinds[name]) == 1}
    if not identical_prefixes_grouped():
        # (``identical_prefixes_cited_apart``): each source's groups on
        # their own.
        for kind, name, locs in _sources:
            if kind == 'n':
                prefix_parts.append((_format_n_prefix(name, len(locs)), name))
            else:
                prefix_parts.append((_format_c_prefix(name, locs, len(locs)), name))
    for name, locants in (combine_identical_prefix_groups(
            (name, locs) for _kind, name, locs in _sources)
            if identical_prefixes_grouped() else ()):
        if name in _single:
            kind, locs = _single[name]
            if kind == 'n':
                prefix = _format_n_prefix(name, len(locs))
            else:
                prefix = _format_c_prefix(name, locs, len(locs))
        else:
            prefix = _format_c_prefix(name, [str(l) for l in locants], len(locants))
        prefix_parts.append((prefix, name))

    # Sort alphabetically by base name
    prefix_parts.sort(key=lambda x: alpha_sort_key(x[1]))

    # Join prefixes
    prefix_str = _join_fused_prefixes([p[0] for p in prefix_parts])

    # Handle suffix-forming groups
    # Priority: carboxylic acid > carbaldehyde > amine > one (IUPAC seniority)
    suffix_groups = substituents.get('suffix_groups', {})

    if suffix_groups:
        # Detachable suffixes (carboxylic acid, carbaldehyde, etc.)
        # These append to the ring name: quinoline-2-carboxylic acid
        _SUFFIX_PRIORITY = _FUSED_SUFFIX_PRIORITY
        _SUFFIX_TO_PREFIX = {
            'carboxylic acid': 'carboxy',
            'carbaldehyde': 'formyl',
            'carboxamide': 'carbamoyl',
            'carbonitrile': 'cyano',
        }

        chosen_suffix = None
        chosen_locants = []
        for suf in _SUFFIX_PRIORITY:
            if suf in suffix_groups:
                chosen_suffix = suf
                chosen_locants = suffix_groups[suf]
                break
        if not chosen_suffix:
            chosen_suffix = next(iter(suffix_groups))
            chosen_locants = suffix_groups[chosen_suffix]

        from ..assembly.naming_utils import _join_multiplied_suffix
        count = len(chosen_locants)
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count)) if count > 1 else ""
        locant_str = ",".join(str(loc) for loc in chosen_locants)
        #: elide multiplier-final 'a' before '-ol' (tetra+ol -> tetrol).
        suffix_part = f"-{locant_str}-{_join_multiplied_suffix(multiplier, chosen_suffix)}"

        # Remaining suffix groups become prefixes
        for suf_name, suf_locants in suffix_groups.items():
            if suf_name == chosen_suffix:
                continue
            prefix_name = _SUFFIX_TO_PREFIX.get(suf_name, suf_name)
            loc_str = ",".join(str(loc) for loc in sorted(suf_locants))
            n = len(suf_locants)
            if n == 1:
                extra_prefix = f"{loc_str}-{prefix_name}-"
            else:
                mult = SIMPLE_MULTIPLIERS.get(n, str(n))
                extra_prefix = f"{loc_str}-{mult}{prefix_name}-"
            prefix_parts.append((extra_prefix, prefix_name))

        # Re-sort and re-join prefixes
        prefix_parts.sort(key=lambda x: alpha_sort_key(x[1]))
        prefix_str = _join_fused_prefixes([p[0] for p in prefix_parts])

        # oxo and amino never reach here (routed above): the suffix group is the
        # only suffix.
        name = _join_prefix_to_parent(prefix_str, core_name) + suffix_part
        _record_if_senior_group_prefixed(mol, ring_atoms, chosen_suffix, name)
        return f"{stereo_prefix}{name}" if stereo_prefix else name

    # Handle suffix-forming groups (oxo and amino only, no detachable suffixes).
    # When hydroxy was promoted above (the gated senior-alcohol case) there is by
    # construction no oxo/amino, so `_build_fused_suffix` is empty and the `-ol`
    # suffix is used instead.
    suffix_str = _build_fused_suffix(substituents, core_name) or hydroxy_ol_suffix
    if substituents.get('amino_substituents'):
        expressed = 'amine'
    elif substituents.get('oxo_substituents') or ketone_core:
        expressed = 'one'
    elif hydroxy_ol_suffix:
        expressed = 'ol'
    else:
        expressed = None

    if suffix_str:
        #: a ketone on a saturated position of a hydro catalog core
        # takes the core's indicated hydrogen (TRIAGE j12 finding 5).
        _non_pin_ih = False
        if expressed == 'one' and not substituents.get('amino_substituents'):
            _oxo = list(substituents.get('oxo_substituents') or [])
            _moved = _indicated_h_at_suffix(mol, core_name, atom_mapping, _oxo)
            if _moved is not None:
                if _moved:
                    core_name = _moved
                else:
                    _non_pin_ih = True
        # Apply suffix to core name with vowel elision
        modified_core = _apply_suffix_to_core(core_name, suffix_str)
        name = _join_prefix_to_parent(prefix_str, modified_core)
        _record_if_senior_group_prefixed(mol, ring_atoms, expressed, name)
        if _non_pin_ih:
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(f"{stereo_prefix}{name}" if stereo_prefix else name)
        return f"{stereo_prefix}{name}" if stereo_prefix else name

    name = _join_prefix_to_parent(prefix_str, core_name)
    _record_if_senior_group_prefixed(mol, ring_atoms, expressed, name)
    return f"{stereo_prefix}{name}" if stereo_prefix else name


_HYDRO_CORE_RE = None
_HYDRO_MULT = {2: 'di', 4: 'tetra', 6: 'hexa', 8: 'octa', 10: 'deca'}


def _indicated_h_at_suffix(mol, core_name, atom_mapping, oxo_locants):
    """The hydro catalog core respelled so that its indicated hydrogen sits at the
    ketone, or None when there is nothing to move; '' when it should move but the
    tautomer cannot be built (the caller then labels the name non-PIN).

     (the Blue Book-24768): "the indicated hydrogen atoms are
    placed at peripheral atoms that will accommodate these principal
    characteristic groups... Locants for hydro prefixes are those of the
    saturated positions" ('7H-1-benzopyran-7-one (PIN)',:24776; '3-imino-2,3-
    dihydro-1H-isoindol-1-one (PIN)',:29609). The catalog core '2,3-dihydro-
    1H-indole' with a ketone at C-2 is therefore '1,3-dihydro-2H-indol-2-one',
    not '2,3-dihydro-1H-indol-2-one'.

    Scope: ONE ketone, a core spelled '<locants>-<n>hydro-<h>H-<stem>' (the stem may
    open with its own heteroatom locants, '1-benzopyran': branch review fixes, the
    dev2000 flavanone '...-3,4-dihydro-2H-1-benzopyran-4-one' for '...-2,3-dihydro-
    4H-1-benzopyran-4-one' passed unmoved), the ketone at a saturated position other
    than h. The saturated set is read off
    the molecule (core ring C/N in no ring double bond, the ketone carbon
    included) and must be the core's hydro + indicated-hydrogen set. The ketone
    position p must admit the mancude tautomer 'pH-<stem>': a Kekule arrangement
    over every other core C and every core N with two ring bonds (chalcogens and
    three-connected N carry no double bond)."""
    import re as _re
    global _HYDRO_CORE_RE
    if len(oxo_locants) != 1:
        return None
    if _HYDRO_CORE_RE is None:
        _HYDRO_CORE_RE = _re.compile(
            r'^(?P<hl>\d+[a-z]?(?:,\d+[a-z]?)*)-(?P<mult>di|tetra|hexa|octa|deca)'
            r'hydro-(?P<ih>\d+[a-z]?)H-(?P<stem>(?:\d+(?:,\d+)*-)?[A-Za-z].*)$')
    m = _HYDRO_CORE_RE.match(core_name or '')
    if not m:
        return None
    p = str(oxo_locants[0])
    ih = m.group('ih')
    hydro = m.group('hl').split(',')
    if p == ih or p not in hydro:
        return None
    loc_of = {a: str(l) for a, l in atom_mapping.items()}
    core = {a for a in atom_mapping if mol.GetAtomWithIdx(a).IsInRing()}
    atom_of = {loc_of[a]: a for a in core}
    if p not in atom_of or ih not in atom_of:
        return ''
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return ''
    in_db = set()
    for bond in kek.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in core and j in core:
                in_db.update((i, j))
    sat = {loc_of[a] for a in core
           if a not in in_db and mol.GetAtomWithIdx(a).GetSymbol() in ('C', 'N')}
    if sat != set(hydro) | {ih}:
        return ''
    nodes = set()
    for a in core:
        if a == atom_of[p]:
            continue
        at = mol.GetAtomWithIdx(a)
        sym = at.GetSymbol()
        ring_deg = sum(1 for nb in at.GetNeighbors() if nb.GetIdx() in core)
        if sym == 'C' or (sym == 'N' and ring_deg == 2):
            nodes.add(a)
        elif sym in ('O', 'S', 'Se', 'Te') or (sym == 'N' and ring_deg == 3):
            continue
        else:
            return ''
    from .partial_saturation import _has_perfect_matching
    if not _has_perfect_matching(mol, nodes):
        return ''
    new_hydro = sorted((sat - {p}), key=lambda l: (int(_re.match(r'\d+', l).group()),
                                                    l))
    mult = _HYDRO_MULT.get(len(new_hydro))
    if not mult:
        return ''
    return f"{','.join(new_hydro)}-{mult}hydro-{p}H-{m.group('stem')}"


def _record_if_senior_group_prefixed(mol, ring_atoms, expressed, name) -> None:
    """Mark ``name`` as not a PIN when a characteristic group senior to the one
    it expresses as the suffix is cited as a prefix or sits in a substituent
    , the Blue Book;: '2-(methylcarbamoyl)-...-
    4-one' (amide 11 > ketone 16), '3-(carboxymethyl)quinoline'. The name is
    kept (it still has to round-trip), but NAME-SCOPED: only a shipped name that
    contains it is labelled below the PIN tier (provenance.record_non_pin_fragment),
    so a speculative call never demotes another producer's name."""
    if not name or not _name_cites_senior_group_as_prefix(mol, ring_atoms, expressed):
        return
    from ..metrics.provenance import record_non_pin_fragment
    record_non_pin_fragment(name)


def _join_prefix_to_parent(prefix_str: str, parent_name: str) -> str:
    """
    Join substituent prefix string to parent name with correct hyphenation.

    IUPAC 2013 rules:
    - Hyphen between prefix and parent when parent starts with a digit:
      "7-nitro-1H-indazole" (1H starts with digit)
    - No hyphen when parent starts with a letter:
      "2-methylquinoline" (quinoline starts with letter)
    - Prefix already ends with hyphen from _join_fused_prefixes

    Args:
        prefix_str: Prefix string (e.g., "2-methyl-", "7-nitro-")
        parent_name: Parent ring name (e.g., "quinoline", "1H-indazole")

    Returns:
        Combined name with correct hyphenation
    """
    if not prefix_str:
        return parent_name

    # (d) (the Blue Book): an italic designator of the parent
    # ('s-indacene') is separated from the prefix's Roman letters by a hyphen,
    # '3-methyl-s-indacene' (``begins_with_italic_designator``).
    from ..assembly.naming_utils import begins_with_italic_designator
    if begins_with_italic_designator(parent_name):
        return prefix_str.rstrip('-') + '-' + parent_name

    # If parent starts with a letter, remove trailing hyphen from prefix
    # e.g., "2-methyl-" + "quinoline" -> "2-methylquinoline"
    if parent_name and parent_name[0].isalpha():
        return prefix_str.rstrip('-') + parent_name

    # If parent starts with a digit, keep the hyphen
    # e.g., "7-nitro-" + "1H-indazole" -> "7-nitro-1H-indazole"
    return prefix_str + parent_name


def _build_fused_suffix(substituents: Dict, core_name: str) -> str:
    """
    Build suffix string for oxo (-one) and amino (-amine) groups.

    Used only when the amine or the ring C=O is the principal characteristic
    group (no senior suffix group, no principal hydroxy; routed by
    _assemble_fused_heterocycle_name).

    Args:
        substituents: Dict with 'oxo_substituents' and 'amino_substituents'
        core_name: Base name for context

    Returns:
        Suffix string like '-6-amine' or '-2,6-dione' or '' if no suffix groups
    """
    amino_locants = substituents.get('amino_substituents', [])
    oxo_locants = substituents.get('oxo_substituents', [])

    # (the Blue Book): ketones (16) are senior to amines
    # (19). _assemble_fused_heterocycle_name never passes both (it fails closed
    # on that combination); the amine suffix used to win here and the oxo was
    # silently dropped.
    if amino_locants and oxo_locants:
        return ''
    if amino_locants:
        return _format_suffix('amine', amino_locants)
    if oxo_locants:
        return _format_suffix('one', oxo_locants)
    return ''


def _format_suffix(suffix_base: str, locants: List) -> str:
    """
    Format a suffix with locants and multiplier.

    Args:
        suffix_base: Base suffix ('amine', 'one')
        locants: List of locants

    Returns:
        Formatted suffix like '-6-amine' or '-2,6-dione'
    """
    count = len(locants)
    locant_str = ','.join(str(loc) for loc in locants)

    if count == 1:
        return f"-{locant_str}-{suffix_base}"
    else:
        from ..assembly.naming_utils import _join_multiplied_suffix
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        #: elide multiplier-final 'a' before '-ol' (tetra+ol -> tetrol).
        return f"-{locant_str}-{_join_multiplied_suffix(multiplier, suffix_base)}"


def _apply_suffix_to_core(core_name: str, suffix: str) -> str:
    """
    Apply suffix to core name with vowel elision.

    IUPAC rules require vowel elision when suffix begins with vowel
    and parent name ends in 'e' (or other vowel).

    Examples:
        purine + -amine -> purin-6-amine (elide final 'e')
        indole + -one -> indol-3-one (elide final 'e')

    Args:
        core_name: Base name like '9H-purine' or '1H-indole'
        suffix: Suffix like '-6-amine' or '-3-one'

    Returns:
        Modified name with suffix applied
    """
    # Check if suffix starts with vowel (after the hyphen-locant-hyphen)
    # Suffix is like '-6-amine', the actual suffix starts after the locant part
    suffix_starts_with_vowel = False
    for part in suffix.split('-'):
        if part and part[0] in 'aeiou':
            suffix_starts_with_vowel = True
            break

    # Check if core ends in 'e' (common case for elision)
    if core_name.endswith('e') and suffix_starts_with_vowel:
        # Elide the final 'e'
        return core_name[:-1] + suffix

    return core_name + suffix


def _format_n_prefix(name: str, count: int) -> str:
    """Format an N-substituent prefix (N-methyl, N,N-dimethyl)."""
    from ..assembly.naming_utils import _wrap_n_substituent
    wrapped = _wrap_n_substituent(name)
    if count == 1:
        return f"N-{wrapped}-"
    else:
        n_locants = ",".join(["N"] * count)
        from ..assembly.naming_utils import multiplied_component
        return f"{n_locants}-{multiplied_component(count, name, wrapped)}-"


def _format_c_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a C-substituent prefix with numeric locants.

    Per IUPAC, compound substituent names containing locants
    are enclosed in parentheses to prevent ambiguity.
    """
    locant_str = ",".join(str(loc) for loc in locants)
    # Wrap compound names (those containing digits or hyphens) in parentheses
    # to prevent locant ambiguity (e.g., 8-(2-methylpropyl) not 8-2-methylpropyl)
    display_name = name
    from ..assembly.naming_utils import (
        _is_fully_enclosed,
        enclose_if_compound,
        is_complex_substituent,
    )
    if name.startswith('('):
        # A name that opens with '(' was formerly left bare unconditionally --
        # correct for a single fully-bracketed token ('(2-methylpropyl)'), WRONG
        # for a compound whose leading '(' is a stereo / indicated-H block, not a
        # whole-substituent enclosure: '(2R,3R,4S,5R)-5-...oxolan-2-yl' is NOT
        # fully enclosed (its first ')' closes mid-string) and needs an OUTER
        # mark -> '[(2R,3R,4S,5R)-5-...oxolan-2-yl]'. Only such
        # not-fully-enclosed names change; a fully-bracketed token stays bare, so
        # every currently-parseable name is byte-identical.
        if not _is_fully_enclosed(name):
            display_name = enclose_if_compound(name)
    elif is_complex_substituent(name):
        # Suite fix j6: the mark escalates past the name's own marks,
        # the Blue Book) -- '3-[2-(2,4-dimethylphenyl)-2-oxoethyl]', not
        # '3-(2-(...)-2-oxoethyl)'; a name without marks still gets '(...)'.
        from ..assembly.naming_utils import _is_fully_enclosed, apply_enclosing_marks
        display_name = (name if _is_fully_enclosed(name)
                        else apply_enclosing_marks(name, -1))
    if count == 1:
        return f"{locant_str}-{display_name}-"
    else:
        # The shared primitive: 'di-tert-butyl' (d),
        # the Blue Book), 'bis' for a substituted prefix (a),
        #:7104: '2,6-bis(4-chlorophenyl)-9H-carbazole'), 'di(dodecyl)'
        # (c)). A basic-multiplier table here never wrote 'bis'.
        from ..assembly.naming_utils import multiplied_component
        return f"{locant_str}-{multiplied_component(count, name, display_name)}-"


def _join_fused_prefixes(prefixes: List[str]) -> str:
    """Join fused ring substituent prefixes."""
    if not prefixes:
        return ""

    # Each prefix already ends with '-', just concatenate
    result = ""
    for prefix in prefixes:
        # Remove trailing hyphen if present, we'll manage hyphens ourselves
        p = prefix.rstrip('-')
        if result:
            # Check if hyphen needed between prefixes
            last_char = result[-1]
            first_char = p[0]
            # (b) (the Blue Book, "Hyphens are used in substitutive
            # names:... after parentheses, if the closing parenthesis is followed
            # by a locant"): the rule is about the closing ENCLOSING MARK, and a
            # prefix escalated to  or { },:7446) closes with ']' or
            # '}'. Testing only ')' glued the next locant on:
            # '7-[(1R,...)-...-9-yl]3-(...)-1H-indole'.
            closes_enclosure = last_char in ')]}'
            if (last_char.isalpha() or closes_enclosure) and first_char.isdigit():
                result += "-"
            elif (last_char.isalpha() or closes_enclosure) and first_char == 'N':
                result += "-"
        result += p

    # Add final hyphen before parent name
    return result + "-"


def name_ortho_fused_bicyclic(mol):
    """
    Generate name for an ortho-fused bicyclic system without a retained name.

    This is a fallback for carbocyclic ortho-fused systems not covered by
    polycyclic_data. Most common fused systems should have retained names.

    For systematic naming, uses fusion descriptors like benzo[x]parent.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included),
        or None if not an ortho-fused bicyclic.

    Examples:
        >>> # For systems without retained names, would generate systematic names
        >>> # Most common ones (naphthalene, indole) have retained names
    """
    if not is_fused_bicyclic(mol):
        return None

    # First check if it's a known fused heterocycle
    # name_fused_heterocycle now returns a tuple or None -- pass through directly
    heterocycle_result = name_fused_heterocycle(mol)
    if heterocycle_result:
        return heterocycle_result

    # For fully saturated carbocyclic ortho-fused systems (e.g., decalin),
    # generate {saturation_prefix}{aromatic_parent} naming
    saturated_name = _name_saturated_fused_carbocyclic(mol)
    if saturated_name:
        # Wrap bare string into structured tuple
        # Saturated fused carbocyclics do NOT discover substituents
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        return (saturated_name, ring_atoms, {}, False)

    # For other carbocyclic ortho-fused systems, check polycyclic data
    # (naphthalene, etc.) - this is handled by polycyclics module
    # Return None to indicate this module doesn't handle it
    return None


# =============================================================================
# Saturated Fused Carbocyclic Naming
# =============================================================================

# Map (smaller_ring, larger_ring) -> aromatic parent name
_FUSED_CARBOCYCLIC_PARENTS = {
    (5, 5): 'pentalene',
    (5, 6): 'indene',
    (5, 7): 'azulene',
    (6, 6): 'naphthalene',
    (6, 7): 'heptalene',
}

# Intrinsic indicated-hydrogen form for the fused carbocyclic parents above
# that are not fully mancude (the Blue Book-11319: indene's PIN is
# "1H-indene"). pentalene/naphthalene/azulene/heptalene are fully conjugated
# mancude ring systems and carry no indicated hydrogen.
_FUSED_CARBOCYCLIC_INTRINSIC_IH = {
    'indene': '1H-indene',
}

# Numeric prefix for hydrogen count
_SATURATION_PREFIXES = {
    2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta',
    6: 'hexa', 7: 'hepta', 8: 'octa', 9: 'nona',
    10: 'deca', 11: 'undeca', 12: 'dodeca',
}

# Hydro count for each aromatic parent when fully saturated
_PARENT_HYDRO_COUNTS = {
    'pentalene': 8,
    'indene': 8,
    'azulene': 10,
    'naphthalene': 10,
    'heptalene': 10,
}


def _name_saturated_fused_carbocyclic(mol) -> Optional[str]:
    """
    Name a fully saturated fused carbocyclic system.

    For systems like decalin (fully saturated naphthalene), generates
    names like "decahydronaphthalene", "octahydropentalene", etc.

    Only handles FULLY saturated (no aromatic atoms), carbocyclic-only systems.

    Args:
        mol: RDKit Mol object

    Returns:
        Name like "decahydronaphthalene" or None if not applicable

    IUPAC Reference: (Saturation prefixes)
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) != 2:
        return None

    # Must not contain aromatic atoms (partially aromatic handled elsewhere)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            return None

    # Must be carbocyclic only (no heteroatoms in ring)
    ring_atoms = set()
    for ring in atom_rings:
        ring_atoms.update(ring)

    for idx in ring_atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() != 'C':
            return None

    # V-5 / V2 theme: if the system retains an isolated (non-aromatic)
    # ring C=C, it is only PARTIALLY saturated — emitting the fully-saturated
    # (decahydro) name here drops the double bond and names a different molecule.
    # Delegate to the partial-saturation namer (correct hydro count + 4a/8a-aware
    # lowest locant set), which fails closed (None) for systems it cannot number.
    has_ring_double = any(
        b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
        and b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms
        for b in mol.GetBonds()
    )
    if has_ring_double:
        from .partial_saturation import name_hydrogenated_fused_carbocycle
        return name_hydrogenated_fused_carbocycle(mol)

    # The name built below is the BARE parent hydride and carries no locants,
    # so it can only ever describe the ring system itself -- the caller says as
    # much ("saturated fused carbocyclics do NOT discover substituents") and
    # returns substituents_included=False. Nothing verified that premise,
    # though, so any exocyclic heavy atom was simply left out of the name:
    # 'CC1CCC2CCCCC2C1' and 'OC1CCC2CCCCC2C1' both came out as the bare
    # 'decahydronaphthalene'. Naming 10 of 11 atoms is a wrong STRUCTURE, and
    # for the exocyclic '=CH2' of 'C=C1CCC2CCCCC2C1' it is the same
    # free-valence loss the sibling ring detectors carried -- reached by
    # dropping the atom instead of by mis-spelling it.
    #
    # Fail closed: this producer declines, another candidate may still name the
    # molecule, and no name that omits an atom is emitted. (Expressing the
    # substituent instead needs fusion-parent NUMBERING, which this fallback
    # does not do and must not fake.)
    if any(atom.GetIdx() not in ring_atoms for atom in mol.GetAtoms()):
        return None

    # Get ring sizes (sorted smallest first for lookup)
    size1 = len(atom_rings[0])
    size2 = len(atom_rings[1])
    ring_key = (min(size1, size2), max(size1, size2))

    parent = _FUSED_CARBOCYCLIC_PARENTS.get(ring_key)
    if parent is None:
        return None

    hydro_count = _PARENT_HYDRO_COUNTS.get(parent)
    if hydro_count is None:
        return None

    prefix = _SATURATION_PREFIXES.get(hydro_count)
    if prefix is None:
        return None

    # / the Blue Book-11319: indene's PIN is '1H-indene' --
    # the mancude parent is NOT fully conjugated (unlike naphthalene/
    # pentalene/azulene/heptalene here, all of which need no indicated
    # hydrogen), so it carries an intrinsic indicated-hydrogen locant that
    # MUST be cited once the parent is modified (the Blue Book-14613:
    # omission is permitted only for the bare, unsubstituted "indene", e.g.
    # "1H-indene-3-carboxylic acid" once substituted) -- a hydro-prefixed
    # name is such a modification, so "octahydro-1H-indene", not the bare
    # "octahydroindene".
    _intrinsic_ih = _FUSED_CARBOCYCLIC_INTRINSIC_IH.get(parent)
    if _intrinsic_ih:
        base = f"{prefix}hydro-{_intrinsic_ih}"
    else:
        base = f"{prefix}hydro{parent}"

    #: ring-junction (bridgehead) stereodescriptors. The junction
    # carbons of a symmetric saturated fused parent hydride are PSEUDOASYMMETRIC
    # (BB 49620/49630: cis-decahydronaphthalene = (4as,8as); trans = (4ar,8ar)),
    # so RDKit/centres assign lowercase r/s. This is the production caller that
    # wires the (formerly dead) ring-junction stereo path. Emit ONLY when it is
    # accuracy- AND determinism-safe (see _saturated_fused_junction_prefix);
    # otherwise the name ships without stereo (missing beats wrong, policy).
    return f"{_saturated_fused_junction_prefix(mol)}{base}"


# (j) preference rank for the junction-locant tie-break in
# _saturated_fused_junction_prefix: "the lower locant is assigned to CIP
# stereodescriptors... R... and r (pseudoasymmetry) that are preferred
# to... S... and s, respectively" (the Blue Book). Lower rank wins
# the lower locant.
_JUNCTION_CIP_RANK = {'R': 0, 'r': 1, 'S': 2, 's': 3}


def _saturated_fused_junction_prefix(mol) -> str:
    """Accuracy- and determinism-safe ring-junction stereo prefix.

    Returns a ``(4ar,8ar)-`` / ``(4as,8as)-`` / ``(3aR,7aS)-`` style prefix
    for a saturated fused bicyclic parent hydride, or ``''`` when emitting
    one would risk a wrong or non-deterministic name. All of the following
    must hold, else ``''``:

    * There are exactly two bridgehead (ring-junction) atoms, both carry a
      ``_CIPCode``, and they are the molecule's ONLY stereo-labelled atoms with
      no E/Z bond stereo. This guarantees we are not silently dropping any
      substituent stereocentre (a substituted decahydronaphthalene would have
      extra centres and is fail-closed here).
    * The junction-locant pair is one this module numbers correctly
      (naphthalene 4a/8a, pentalene 3a/6a, indene 3a/7a, azulene 3a/8a,
      heptalene 4a/9a — see get_junction_locants_for_fused_system).

    Which physical bridgehead maps to the LOWER locant (4a/3a) vs the higher
    one (8a/7a) is a genuine numbering CHOICE, not an arbitrary atom-index
    order: the bare bicyclic skeleton (ignoring stereo) of every ortho-fused
    parent supported here has a mirror automorphism swapping the two
    bridgeheads (each ring's non-fusion atoms are symmetric about the
    fusion-bond axis), so relabelling which bridgehead is "first" always
    describes the identical constitution. Two cases:

    * Symmetric-ring parents (5,5 / 6,6 -- pentalene/naphthalene): the two
      bridgeheads sit in a truly equivalent chemical environment (a REAL
      molecular symmetry of that specific stereoisomer, not just a skeletal
      one), so they always get IDENTICAL descriptors when chiral (both r/r,
      s/s, R/R, or S/S) -- the choice of which gets the lower locant is moot.
    * Asymmetric-ring parents (5,6 indene / 5,7 azulene / 6,7 heptalene): the
      two bridgeheads are chemically distinct and CAN carry genuinely
      different descriptors (e.g. hydrindane's real R/S, not pseudoasymmetric
      r/s). BB (j) (the Blue Book) resolves the choice: the
      lower locant goes to the PREFERRED descriptor (R over S, r over s).

    A mixed family (one bridgehead R/S-type, the other r/s-type) is not an
    expected/verified shape for this parent set and fails closed rather than
    guessing an ordering.
    """
    assign_stereochemistry(mol)

    bridgeheads = get_bridgehead_atoms(mol)
    if len(bridgeheads) != 2:
        return ""

    # Guard: junctions must be the ONLY stereo-labelled units (no dropped stereo).
    labelled_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.HasProp('_CIPCode')]
    if set(labelled_atoms) != set(bridgeheads):
        return ""
    if any(b.HasProp('_CIPCode') for b in mol.GetBonds()):
        return ""

    ring_sizes = get_fused_ring_sizes(mol)
    if ring_sizes == (0, 0):
        return ""

    cip_a = mol.GetAtomWithIdx(bridgeheads[0]).GetProp('_CIPCode')
    cip_b = mol.GetAtomWithIdx(bridgeheads[1]).GetProp('_CIPCode')

    if cip_a == cip_b:
        # Symmetric case (decalin/pentalene-like): either physical ordering
        # yields the identical descriptor pair.
        ordered_bridgeheads = bridgeheads
    elif {cip_a, cip_b} in ({'R', 'S'}, {'r', 's'}):
        # (j): the preferred descriptor gets the lower locant.
        if _JUNCTION_CIP_RANK[cip_a] < _JUNCTION_CIP_RANK[cip_b]:
            ordered_bridgeheads = [bridgeheads[0], bridgeheads[1]]
        else:
            ordered_bridgeheads = [bridgeheads[1], bridgeheads[0]]
    else:
        # Mixed descriptor family at chemically-equivalent bridgeheads is not
        # a verified case -- fail closed rather than guess.
        return ""

    junction_locants = get_junction_locants_for_fused_system(
        mol, ordered_bridgeheads, ring_sizes[0], ring_sizes[1]
    )
    descriptors = collect_ring_junction_stereo(mol, ordered_bridgeheads, junction_locants)
    if len(descriptors) != 2:
        return ""

    return format_ring_junction_stereo(descriptors)


def is_fused_aromatic_system(mol) -> bool:
    """
    Check if molecule contains a fused aromatic ring system.

    A fused aromatic system has 2+ aromatic rings sharing edges.

    Args:
        mol: RDKit Mol object

    Returns:
        True if fused aromatic system detected

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1') # naphthalene
        >>> is_fused_aromatic_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccccc1') # benzene
        >>> is_fused_aromatic_system(mol)
        False
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Find aromatic rings
    aromatic_rings = [ring for ring in atom_rings if is_aromatic_ring(mol, ring)]

    if len(aromatic_rings) < 2:
        return False

    # Check if any pair of aromatic rings are fused (share 2+ atoms)
    for i, ring1 in enumerate(aromatic_rings):
        for j, ring2 in enumerate(aromatic_rings):
            if i >= j:
                continue
            shared = get_shared_atoms(mol, ring1, ring2)
            if len(shared) >= 2:
                return True

    return False


def is_fused_heterocyclic_system(mol) -> bool:
    """
    Check if molecule contains a fused heterocyclic ring system.

    A fused heterocyclic system has at least one heterocyclic ring
    fused with another ring.

    Args:
        mol: RDKit Mol object

    Returns:
        True if fused heterocyclic system detected

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1') # indole
        >>> is_fused_heterocyclic_system(mol)
        True
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return False

    # Find heterocyclic rings
    heterocyclic_rings = []
    for ring in atom_rings:
        if is_heterocyclic(mol, ring):
            heterocyclic_rings.append(ring)

    if not heterocyclic_rings:
        return False

    # Check if any heterocyclic ring is fused with another ring
    for hetero_ring in heterocyclic_rings:
        for other_ring in atom_rings:
            if other_ring == hetero_ring:
                continue
            shared = get_shared_atoms(mol, hetero_ring, other_ring)
            if len(shared) >= 2:
                return True

    return False


# =============================================================================
# Ring Junction Stereochemistry Functions for Fused Systems
# =============================================================================

def get_fused_ring_sizes(mol) -> Tuple[int, int]:
    """
    Get the sizes of the two rings in a fused bicyclic system.

    Args:
        mol: RDKit Mol object with exactly 2 fused rings

    Returns:
        Tuple of (ring1_size, ring2_size), sorted largest first

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1') # decalin
        >>> get_fused_ring_sizes(mol)
        (6, 6)
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) != 2:
        return (0, 0)

    size1 = len(atom_rings[0])
    size2 = len(atom_rings[1])

    return (max(size1, size2), min(size1, size2))


def name_saturated_fused_bicyclic(mol, parent_name: str = "decahydronaphthalene") -> Optional[str]:
    """
    Generate IUPAC name for a saturated fused bicyclic with ring junction stereochemistry.

    For saturated fused systems like decalin (decahydronaphthalene), the
    stereochemistry at ring junction atoms (bridgeheads) must be specified.

    IUPAC format: (4aR,8aS)-decahydronaphthalene
    Alternative: cis-decalin or trans-decalin (for common cases)

    Args:
        mol: RDKit Mol object
        parent_name: The parent name for the saturated system

    Returns:
        IUPAC name with stereodescriptor prefix, or None if cannot determine

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1') # cis-decalin
        >>> name_saturated_fused_bicyclic(mol)
        '(4as,8as)-decahydronaphthalene'
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1') # trans-decalin
        >>> name_saturated_fused_bicyclic(mol)
        '(4ar,8ar)-decahydronaphthalene'
    """
    if not is_fused_bicyclic(mol):
        return None

    # Assign CIP labels (idempotent guard)
    assign_stereochemistry(mol)

    # Find bridgehead atoms
    bridgeheads = get_bridgehead_atoms(mol)

    if len(bridgeheads) != 2:
        # No stereochemistry to report
        return parent_name

    # Get ring sizes to determine proper locant scheme
    ring_sizes = get_fused_ring_sizes(mol)

    # Get junction locants (4a, 8a for 6,6-fused)
    junction_locants = get_junction_locants_for_fused_system(
        mol, bridgeheads, ring_sizes[0], ring_sizes[1]
    )

    # Collect stereodescriptors for junction atoms
    stereo_descriptors = collect_ring_junction_stereo(mol, bridgeheads, junction_locants)

    if not stereo_descriptors:
        return parent_name

    # Format the stereodescriptor prefix
    stereo_prefix = format_ring_junction_stereo(stereo_descriptors)

    return f"{stereo_prefix}{parent_name}"


def get_ring_junction_stereo_prefix(mol) -> str:
    """
    Get the stereochemistry prefix for ring junction atoms in a fused system.

    This is a utility function that can be used by other naming functions
    to add ring junction stereochemistry to a name.

    Args:
        mol: RDKit Mol object

    Returns:
        Stereodescriptor prefix like "(4aS,8aS)-" or "" if no stereo

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        >>> get_ring_junction_stereo_prefix(mol)
        '(4as,8as)-'
    """
    # Assign CIP labels (idempotent guard)
    assign_stereochemistry(mol)

    # Find bridgehead atoms
    bridgeheads = get_bridgehead_atoms(mol)

    if not bridgeheads:
        return ""

    # Get ring sizes
    ring_sizes = get_fused_ring_sizes(mol)
    if ring_sizes == (0, 0):
        return ""

    # Get junction locants
    junction_locants = get_junction_locants_for_fused_system(
        mol, bridgeheads, ring_sizes[0], ring_sizes[1]
    )

    # Collect stereodescriptors
    stereo_descriptors = collect_ring_junction_stereo(mol, bridgeheads, junction_locants)

    if not stereo_descriptors:
        return ""

    return format_ring_junction_stereo(stereo_descriptors)


def get_simple_cis_trans_prefix(mol) -> str:
    """
    Get simple cis/trans prefix for bicyclic ring junction.

    For simple bicyclic systems, returns "cis-" or "trans-" instead of
    the full (4aR,8aS)- notation. This is a common simplification.

    Args:
        mol: RDKit Mol object

    Returns:
        "cis-" or "trans-" or "" if cannot determine

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        >>> get_simple_cis_trans_prefix(mol)
        'cis-'
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        >>> get_simple_cis_trans_prefix(mol)
        'trans-'
    """
    bridgeheads = get_bridgehead_atoms(mol)

    if len(bridgeheads) != 2:
        return ""

    result = determine_simple_cis_trans(mol, bridgeheads)

    if result:
        return f"{result}-"
    return ""
