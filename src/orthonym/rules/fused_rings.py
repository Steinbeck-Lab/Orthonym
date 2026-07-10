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

Reference: IUPAC 2013 Blue Book, Section P-25 (Fused Ring Systems)
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque

from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from ..data.fused_heterocycles import (
    get_fused_heterocycle_name,
    match_fused_heterocycle_core,
    FUSED_HETEROCYCLE_DATA,
)
from ..data.xanthine_derivatives import (
    identify_xanthine,
    get_xanthine_name,
    is_xanthine_derivative,
)
from ..data import get_retained_name as _get_global_retained_name
from ..perception.rings import (
    get_ring_info,
    get_ring_systems,
    is_aromatic_ring,
    is_heterocyclic,
)
from ..assembly.naming_utils import (
    get_alkyl_name,
    alpha_sort_key,
    get_multiplier_prefix,
)
from .stereochemistry import (
    get_bridgehead_atoms,
    collect_ring_junction_stereo,
    format_ring_junction_stereo,
    determine_simple_cis_trans,
    get_junction_locants_for_fused_system,
)


# Simple multiplicative prefixes for substituent naming
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def _compute_general_indicated_h(mol, ring_atom_set: Set[int],
                                  atom_to_locant: Dict[int, Any]) -> List:
    """Compute indicated hydrogen for non-retained fused systems.

    Per IUPAC P-14.7: In a ring system with maximum non-cumulative double bonds,
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


def _try_algorithmic_fusion_name(mol) -> Optional[str]:
    """
    Attempt systematic fusion naming for 2-component ortho-fused systems.

    Called as fallback when dictionary lookup fails. Uses IUPAC P-25.1
    to P-25.3 rules for component identification, descriptor generation,
    and name assembly.

    Only handles 2-component ortho-fused systems (SSSR has exactly 2 rings
    sharing exactly 2 atoms). 3+ component systems deferred (DEFR-07).

    Args:
        mol: RDKit Mol object

    Returns:
        Systematic fusion name, or None if cannot be generated
    """
    import logging
    logger = logging.getLogger(__name__)

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Gate: must have at least 2 rings in SSSR (D-08 broadening — was != 2)
    if len(atom_rings) < 2:
        return None

    # Phase 149 D-08: route base decision through select_base_component
    # for any N>=2. The naming engine remains 2-ring-only.
    #
    # Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    # Source: 149-CONTEXT.md D-08.
    from .fused_ring_selection import select_base_component, _enumerate_components
    components = _enumerate_components(mol)
    base_atoms, _others = select_base_component(mol, components)

    if len(atom_rings) != 2:
        # 3+ component case (D-08 trade-off): emit decision-only.
        # base_atoms is consumed by namer.Branch 6.5 (Task 02-03) via
        # features.parent_selection_result; full systematic-name assembly
        # deferred to 149.x or Phase 155.
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

    # Gate: molecule should be an unsubstituted fused ring system
    # (all heavy atoms are ring atoms + no exocyclic substituents beyond H)
    # Substituted fused systems need prefix/suffix handling that the
    # algorithmic path does not yet support. Substituted molecules should
    # fall through to the existing naming pipeline.
    ring_atom_set = set(ring1) | set(ring2)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in ring_atom_set and atom.GetAtomicNum() != 1:
            return None  # Has substituents

    from .fusion_descriptors import (
        generate_systematic_name_for_fused_pair,
        identify_parent_and_child,
        _get_iupac_ring_order_for_fusion,
    )

    # Try to generate systematic name
    name = generate_systematic_name_for_fused_pair(
        mol, list(ring1), list(ring2), shared
    )

    if not name:
        return None

    # Compute indicated hydrogen for the algorithmic fused system
    parent_name, child_name, parent_ring, child_ring = identify_parent_and_child(
        mol, set(ring1), set(ring2)
    )

    # Build atom-to-locant mapping for indicated H computation
    atom_to_locant = _build_algorithmic_locant_map(
        mol, parent_ring, child_ring, shared, parent_name, child_name
    )

    ring_atom_set = set(ring1) | set(ring2)
    indicated_h = _compute_general_indicated_h(mol, ring_atom_set, atom_to_locant)

    # P-25.3.2.5.2: lambda (nonstandard bonding number) tokens follow the
    # atom's fused-system locant and are cited at the beginning of the name,
    # after any indicated hydrogen (monocyclic precedent 1H-1lambda4-thiophene,
    # P-22.2.7.1). Fail-closed: a lambda atom without a determinate NUMERIC
    # fused locant (or on skeletal carbon) invalidates the whole name — the
    # lambda-less name would denote a DIFFERENT molecule.
    from .lambda_convention import nonstandard_bonding_number
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
            f"{loc}lambda{lam}" for loc, lam in lam_entries
        ) + '-'

    if indicated_h:
        h_parts = ','.join(str(loc) + 'H' for loc in indicated_h)
        name = f"{h_parts}-{lam_prefix}{name}"
    elif lam_prefix:
        name = f"{lam_prefix}{name}"

    logger.debug(
        "Algorithmic fusion name for %s: %s",
        Chem.MolToSmiles(mol), name,
    )

    return name


# ============================================================================
# Polycomponent ortho-fusion constructor (P-25.3.4) — v22 Phase G1b (DD7 COV-01)
# ============================================================================
#
# Builds systematic fusion PINs for the cata-fused single-base monocyclic-
# component "star" sub-class: a single most-senior base ring (P-25.3.2 / FR-2.3)
# with >=2 attached monocyclic components, each ortho-fused to the base by exactly
# one bond, no interior (peri) atom, unsubstituted, no carbocyclic attached
# component. This replaces the `len(atom_rings) != 2 -> return None` decision-only
# gate for the structures it can name CORRECTLY (e.g. difuro[3,2-b:2',3'-e]pyridine,
# furo[3,2-b]thieno[2,3-e]pyridine); everything outside the sub-class fails closed
# (returns None -> existing path / G0 fail-closed veto). It does NOT handle
# multiparent bases ("difuran"), ortho-peri (interior atom), second-order attached
# components, bridged-AND-fused, or carbocyclic children (those are deferred, A10).
#
# Source: IUPAC 2013 Blue Book P-25.3.1.3 (descriptor construction),
#         P-25.3.2 (base component), https://iupac.qmul.ac.uk/fusedring/FR23.html

# Carbocyclic attached components (benzo/cyclopenta...) are refused: nearly all
# 3-ring benzo-fused systems are RETAINED (acridine/carbazole/dibenzofuran) and
# named upstream; emitting a systematic dibenzo[...] here would be a non-PIN.
_PCF_CARBOCYCLIC = {
    'benzene', 'cyclopentadiene', 'cyclopentene',
    'cycloheptadiene', 'cycloheptene', 'cyclohexene',
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
    from .fusion_descriptors import _get_iupac_ring_order_for_fusion

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
            child_order = _get_iupac_ring_order_for_fusion(
                mol, comps[ci], shared, is_child=True)
            res = _pcf_edge_descriptor(border, child_order, shared)
            if res is None:
                ok = False
                break
            letter, locs = res
            per.append((nm, letter, locs))
        if not ok:
            continue
        # Lowest letter set; tiebreak by letters in alphanumerical citation
        # order (so the first-cited component gets the lowest letter), then by
        # child locants in citation order. P-25.3.3 / P-25.3.1.3.
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
    star system (P-25.3.4 restricted sub-class). Returns the PIN, or None when
    the molecule is outside the handled sub-class (fail closed; the caller then
    keeps its existing behaviour). MUST be invoked AFTER retained-name checks —
    it does not recognise retained systems and would emit a systematic name for
    them (acridine etc.); carbocyclic children are additionally refused.

    Source: IUPAC 2013 Blue Book P-25.3.1.3, P-25.3.2; FR-2.3.
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) < 3:
        return None  # >=3 components only; 2-component handled by existing path

    # Unsubstituted: every heavy atom is a ring atom (substituted/funct. systems
    # need final-system peripheral numbering -> deferred, fail closed).
    ring_atom_set = set()
    for r in atom_rings:
        ring_atom_set.update(r)
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in ring_atom_set and atom.GetAtomicNum() != 1:
            return None

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

    # FR-2.3(a): base is among the most-senior components. Try EACH most-senior
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
    if len(produced) == 1:
        return next(iter(produced))
    return None


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

    This mapping is used by _compute_general_indicated_h() to assign
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
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> ri = mol.GetRingInfo()
        >>> rings = ri.AtomRings()
        >>> shared = get_shared_atoms(mol, rings[0], rings[1])
        >>> len(shared)  # 2 atoms shared in ortho-fused system
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
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> classify_fused_system(mol)
        'ortho-fused'
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')  # pyrene
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
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> is_fused_bicyclic(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')  # carbazole (tricyclic)
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
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
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

    # P-25.3.2.5.2 / P-25.6: a fused system containing a nonstandard-bonding-
    # number atom may ONLY be named by the algorithmic path (which emits the
    # mandatory lambda tokens). Catalog and core-substructure matches are
    # standard-valence structures — matching one here would name a different
    # molecule. Fail closed if the algorithmic path declines.
    from .lambda_convention import nonstandard_bonding_number as _nsbn
    if any(_nsbn(mol, i) is not None for i in ring_atoms):
        _lam_name = _try_algorithmic_fusion_name(mol)
        if _lam_name and 'lambda' in _lam_name:
            return (_lam_name, ring_atoms, {}, True)
        return None

    # Check xanthine derivatives FIRST (caffeine, theophylline, etc.)
    # These have specific N-position numbering that differs from standard
    # heterocycle patterns (uses numeric locants like 1,3,7-trimethyl)
    xanthine_name = get_xanthine_name(mol)
    if xanthine_name:
        return (xanthine_name, ring_atoms, {}, True)

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
        # Try algorithmic systematic fusion naming (IUPAC P-25.1 to P-25.3)
        # for 2-component ortho-fused systems not in the dictionary
        algorithmic_name = _try_algorithmic_fusion_name(mol)
        if algorithmic_name:
            return (algorithmic_name, ring_atoms, {}, True)
        # v22 G1b (P-25.3.4): polycomponent ortho-fusion constructor for 3+-
        # component cata-fused monocyclic-component systems with no catalog core.
        poly = _try_polycomponent_fusion_name(mol)
        if poly:
            return (poly, ring_atoms, {}, True)
        # Wave-2 P5 fused (Task 7) FAIL-CLOSED follow-up: interior-heteroatom
        # ortho-/peri-fused systems (P-25.3.3.2/.2.1/.2.2/.2.3/.3.3.2) require
        # a superscript interior locant (e.g. 3a1 / 2a1H). OPSIN 2.9 CANNOT
        # parse any name carrying that token (verified: '2a1H-cyclopenta[cd]-
        # pyrene', 'pyracylene' -> blank), so there is NO verifiable oracle for
        # the interior-locant PIN. We deliberately decline here (return None)
        # rather than emit an unverifiable interior-superscript fusion name;
        # the SELF-01 production gate additionally suppresses any von-Baeyer
        # fallback that does not round-trip. Build the interior-atom-numbering
        # engine when a parseable oracle (newer OPSIN / hand-checked internal
        # oracle) exists. See tests/unit/rules/test_wave2_p5_fused.py
        # ::TestP25InteriorAtomNumberingFailClosed.
        return None

    core_name, atom_mapping, _core_smiles = core_result

    # v22 G1b (P-25.3.4): a 2-component catalog core (e.g. furo[3,2-b]pyridine)
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
        # v23 Phase 5 note: when poly is None here the matched catalog core is
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
        # (a dispatch-level change, Phase 13 polycomponent-fusion scope).

    # NOTE: General indicated hydrogen (_compute_general_indicated_h) is
    # implemented but NOT wired here. Dictionary-matched fused systems handle
    # indicated H via tautomer_locant in fused_heterocycles.py entries.
    # The general algorithm is reserved for future systematic fusion naming
    # of non-retained fused systems (deferred from Phase 107).

    # Find substituents on the core
    substituents = get_fused_heterocycle_substituents(mol, atom_mapping)

    if not substituents:
        return (core_name, ring_atoms, atom_mapping, True)

    # Build the substituted name
    name = _assemble_fused_heterocycle_name(mol, core_name, substituents, atom_mapping)
    return (name, ring_atoms, atom_mapping, True)


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
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
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
                    # N-substitution
                    result['n_substituents'][sub_name] += 1
                else:
                    # C-substitution
                    result['c_substituents'][sub_name].append(locant)
            elif sub_type == 'oxo':
                # Oxo group (=O) - use suffix form (-one)
                result['oxo_substituents'].append(locant)
            elif sub_type == 'functional' and sub_name == 'amino':
                # Amino group (-NH2) - use suffix form (-amine)
                result['amino_substituents'].append(locant)
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
        # universal naming (Phase 85 USUB-05 gap closure; WS-A task 9 extends
        # it to RING-SYSTEM substituents — the old blanket ring guards
        # silently DROPPED the fragment, naming 2-(pyridin-2-yl)quinoline as
        # bare 'quinoline').
        # Guards kept (retargeted precisely):
        #   1. A ring that STRADDLES the core boundary means the core mapping
        #      is incomplete (coumarin 3-ring case) -> not a substituent.
        #   2. Only common organic elements, no exotic (As, Se, etc.)
        #   3. Modest size (<=25 atoms)
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
            # naphthalenylmethyl, ...) -> the single WS-A delegate.
            from .ring_substituents import name_ring_system_substituent
            fallback_name = name_ring_system_substituent(
                mol, sub_atoms, start_idx
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
                                    alkoxy_name = rec_name[:-2] + 'oxy'
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
                    if alkyl_name is None:
                        try:
                            alkyl_name = get_alkyl_name(carbon_count)
                        except (ValueError, KeyError):
                            alkyl_name = None
                    if alkyl_name:
                        # OPSIN treats "anilino" as a simple substituent
                        if alkyl_name == 'phenyl':
                            return {
                                'name': 'anilino',
                                'type': 'functional',
                                'atoms': [start_idx] + alkyl_atoms
                            }
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

    IUPAC Reference: P-64.4 (acetic acid derivatives as substituents),
                     P-25.3 (naming fused ring substituents)
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
    acid_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
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
    aldehyde_pattern = Chem.MolFromSmarts('[CX3H1](=O)')
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
    hydroxyl_pattern = Chem.MolFromSmarts('[OX2H1]')
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
    amino_pattern = Chem.MolFromSmarts('[NX3H2;!$([NX3H2][CX3]=O)]')
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
    acyl_pattern = Chem.MolFromSmarts('[CX3](=O)[#6]')
    if acyl_pattern:
        matches = mol.GetSubstructMatches(acyl_pattern)
        for match in matches:
            acyl_c = match[0]
            if acyl_c in chain_atoms and acyl_c == start_idx:
                # Acyl group directly on ring
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

    # Check for ester group: -C(=O)-O-R (suffix: carboxylate/carboxylic acid)
    ester_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')
    if ester_pattern:
        matches = mol.GetSubstructMatches(ester_pattern)
        for match in matches:
            ester_c = match[0]
            if ester_c in chain_atoms and ester_c == start_idx:
                # Ester directly on ring: e.g., -C(=O)OCH3 → "methoxycarbonyl" or
                # treated as suffix "carboxylate" / "carboxylic acid" depending on context
                o_idx = match[2]
                alkyl_atom = match[3]
                # Check alkyl part
                alkyl_atoms = _bfs_alkyl_from(mol, alkyl_atom, excluded | set(chain_atoms) - {alkyl_atom})
                if alkyl_atoms is not None:
                    alkyl_c_count = sum(1 for idx in alkyl_atoms
                                        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')
                    _ALKYL = {1: 'methyl', 2: 'ethyl', 3: 'propyl'}
                    alkyl_name = _ALKYL.get(alkyl_c_count)
                    if alkyl_name:
                        # Return as suffix: "methyl ester" -> suffix_name used in assembly
                        return {
                            'name': 'carboxylic acid',
                            'atoms': chain_atoms,
                            'functional_group': 'ester',
                            'type': 'suffix',
                            'suffix_name': 'carboxylic acid',
                            'ester_alkyl': alkyl_name,
                        }

    return None


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

    prefix_parts = []

    # Handle N-substituents
    for name, count in substituents['n_substituents'].items():
        prefix = _format_n_prefix(name, count)
        prefix_parts.append((prefix, name))

    # Handle C-substituents
    for name, locants in substituents['c_substituents'].items():
        count = len(locants)
        prefix = _format_c_prefix(name, locants, count)
        prefix_parts.append((prefix, name))

    # Handle other substituents (halogens, etc.)
    other_groups = defaultdict(list)
    for sub in substituents['other']:
        name = sub['name']
        locant = sub['locant']
        other_groups[name].append(locant)

    for name, locants in other_groups.items():
        # Ensure all locants are strings for consistent sorting
        # (some may be int, others str like '3a')
        locants = [str(l) for l in locants]
        locants.sort(key=lambda x: (len(x), x))
        count = len(locants)
        prefix = _format_c_prefix(name, locants, count)
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
        _SUFFIX_PRIORITY = ['carboxylic acid', 'carbaldehyde', 'carboxamide', 'carbonitrile']
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
        # P-63.1.2: elide multiplier-final 'a' before '-ol' (tetra+ol -> tetrol).
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

        # Also handle oxo/amino as additional suffixes or prefixes
        oxo_amino_suffix = _build_fused_suffix(substituents, core_name)
        if oxo_amino_suffix:
            modified_core = _apply_suffix_to_core(core_name, oxo_amino_suffix)
            name = _join_prefix_to_parent(prefix_str, modified_core) + suffix_part.lstrip('-')
            return f"{stereo_prefix}{name}" if stereo_prefix else name
        name = _join_prefix_to_parent(prefix_str, core_name) + suffix_part
        return f"{stereo_prefix}{name}" if stereo_prefix else name

    # Handle suffix-forming groups (oxo and amino only, no detachable suffixes)
    suffix_str = _build_fused_suffix(substituents, core_name)

    if suffix_str:
        # Apply suffix to core name with vowel elision
        modified_core = _apply_suffix_to_core(core_name, suffix_str)
        name = _join_prefix_to_parent(prefix_str, modified_core)
        return f"{stereo_prefix}{name}" if stereo_prefix else name

    name = _join_prefix_to_parent(prefix_str, core_name)
    return f"{stereo_prefix}{name}" if stereo_prefix else name


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

    For IUPAC 2013, when amino or oxo are present, they use suffix naming.
    Amino takes precedence (higher seniority) over oxo for suffix position.

    Args:
        substituents: Dict with 'oxo_substituents' and 'amino_substituents'
        core_name: Base name for context

    Returns:
        Suffix string like '-6-amine' or '-2,6-dione' or '' if no suffix groups
    """
    suffix_parts = []

    # Amino has higher seniority than oxo in IUPAC
    # When both present, amino is suffix, oxo becomes prefix
    # For simplicity, handle amino as suffix first

    amino_locants = substituents.get('amino_substituents', [])
    oxo_locants = substituents.get('oxo_substituents', [])

    if amino_locants:
        # Amino suffix: -amine, -diamine, etc.
        suffix_parts.append(_format_suffix('amine', amino_locants))

    if oxo_locants:
        # If amino is present, oxo should be prefix (handled elsewhere)
        # If no amino, oxo is suffix: -one, -dione, etc.
        if not amino_locants:
            suffix_parts.append(_format_suffix('one', oxo_locants))
        # Note: When amino is suffix, oxo should be "oxo" prefix
        # This would require more complex logic; for now, just use suffix

    return ''.join(suffix_parts)


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
        # P-63.1.2: elide multiplier-final 'a' before '-ol' (tetra+ol -> tetrol).
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
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{n_locants}-{multiplier}{wrapped}-"


def _format_c_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a C-substituent prefix with numeric locants.

    Per IUPAC P-14.5.2, compound substituent names containing locants
    are enclosed in parentheses to prevent ambiguity.
    """
    locant_str = ",".join(str(loc) for loc in locants)
    # Wrap compound names (those containing digits or hyphens) in parentheses
    # to prevent locant ambiguity (e.g., 8-(2-methylpropyl) not 8-2-methylpropyl)
    display_name = name
    if not name.startswith('('):
        from ..assembly.naming_utils import is_complex_substituent
        if is_complex_substituent(name):
            display_name = f'({name})'
    if count == 1:
        return f"{locant_str}-{display_name}-"
    else:
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{locant_str}-{multiplier}{display_name}-"


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
            if (last_char.isalpha() or last_char == ')') and first_char.isdigit():
                result += "-"
            elif (last_char.isalpha() or last_char == ')') and first_char == 'N':
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

    IUPAC Reference: P-31.1.1 (Saturation prefixes)
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

    # V-5 / V2 theme (P-31.1.4): if the system retains an isolated (non-aromatic)
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

    return f"{prefix}hydro{parent}"


def is_fused_aromatic_system(mol) -> bool:
    """
    Check if molecule contains a fused aromatic ring system.

    A fused aromatic system has 2+ aromatic rings sharing edges.

    Args:
        mol: RDKit Mol object

    Returns:
        True if fused aromatic system detected

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> is_fused_aromatic_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
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
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
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
        >>> mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin
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
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')  # cis-decalin
        >>> name_saturated_fused_bicyclic(mol)
        '(4as,8as)-decahydronaphthalene'
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')  # trans-decalin
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
