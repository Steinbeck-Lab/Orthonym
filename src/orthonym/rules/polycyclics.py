"""
Polycyclic aromatic hydrocarbon (PAH) naming rules.

Handles identification and naming of common polycyclic aromatics:
- Bicyclic: naphthalene
- Tricyclic: anthracene, phenanthrene, fluorene, acenaphthene, acenaphthylene
- Tetracyclic: pyrene, chrysene, tetracene, triphenylene, benz[a]anthracene, benzo[c]phenanthrene
- Pentacyclic: pentacene, perylene, benzo[a]pyrene
- Hexacyclic+: coronene

Also coordinates with fused_rings module for fused heterocyclic systems.

IUPAC 2013 Rules (Blue Book Section P-25):
- PAH numbering is FIXED by IUPAC standard
- Use retained names as parent
- Substituents are named with their IUPAC locant position
- Numbering is NOT reoriented based on substituents (unlike benzene)

Key difference from benzene:
- Benzene substituent numbering uses lowest locants
- PAH numbering is fixed to the standard IUPAC orientation

PAH Classification:
- Ortho-fused: linear PAHs like naphthalene, anthracene, tetracene, pentacene
- Peri-condensed: PAHs with interior atoms like pyrene, perylene, coronene

Integration with fused_rings.py:
- This module handles carbocyclic PAHs (naphthalene, anthracene, etc.)
- fused_rings.py handles fused heterocycles (indole, quinoline, etc.)
- This module can delegate to fused_rings for heterocyclic detection
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, deque
from rdkit import Chem

from .locants import compare_locant_sets as _compare_locant_sets  # IM-02
from ..data.polycyclic_data import (
    POLYCYCLIC_DATA,
    get_polycyclic_by_smiles,
    is_polycyclic_aromatic,
    match_polycyclic_core,
    get_pah_core_atoms,
)
from ..assembly.naming_utils import (
    format_substituent_prefix,
    alpha_sort_key,
    get_multiplier_prefix,
)
from ..perception.rings import (
    get_ring_info,
    get_ring_systems,
    is_aromatic_ring,
    is_heterocyclic,
)


def identify_polycyclic(mol) -> Optional[str]:
    """
    Identify if a molecule is a (possibly substituted) polycyclic aromatic.

    This function checks if the molecule's core structure matches a known PAH.
    It uses substructure matching to find PAH cores in substituted molecules.

    Args:
        mol: RDKit Mol object

    Returns:
        PAH name if identified, None otherwise

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> identify_polycyclic(mol)
        'naphthalene'
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> identify_polycyclic(mol)
        'naphthalene'
    """
    # First, try exact canonical match (unsubstituted PAH)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    if is_polycyclic_aromatic(canonical_smiles):
        result = get_polycyclic_by_smiles(canonical_smiles)
        if result:
            return result['name']

    # Try substructure matching for substituted PAHs
    # Check from largest to smallest to find the best (largest) match
    pah_by_size = sorted(
        POLYCYCLIC_DATA.items(),
        key=lambda x: x[1]['num_atoms'],
        reverse=True
    )

    for pah_name, pah_data in pah_by_size:
        smarts = pah_data['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # Verify that all atoms in the match are aromatic or expected sp3
            # (for fluorene, acenaphthene which have methylene bridges)
            match_atoms = set(matches[0])

            # Count how many atoms in the molecule are in the core
            # vs how many are substituents
            total_atoms = mol.GetNumAtoms()
            core_atoms = len(match_atoms)

            # If the core matches and we have substituents, this is the PAH
            if core_atoms == pah_data['num_atoms']:
                # Coverage veto (Wave-2 completion; replaces the loose 0.6
                # majority ratio): the fused ring COMPONENT containing the
                # matched core must BE exactly the core. Any extra RING atom
                # fused/bridged onto the core (the 1,4-epoxynaphthalene O, a
                # spiro ring, a larger uncataloged polycycle) means the parent
                # is NOT this PAH -- naming it core+substituents DROPS the
                # extra ring atoms (a wrong name the RT gate had to suppress).
                # Ring systems connected only by single bonds
                # (1-phenylnaphthalene) are separate components and pass.
                # IUPAC P-44.2.1(e): greater number of skeletal ring atoms.
                components = []
                for ring in mol.GetRingInfo().AtomRings():
                    rs = set(ring)
                    merged = [c for c in components if c & rs]
                    for c in merged:
                        components.remove(c)
                    components.append(rs.union(*merged) if merged else rs)
                covered = set()
                for c in components:
                    if c & match_atoms:
                        covered |= c
                if covered != match_atoms:
                    continue  # Skip, try smaller PAH or return None
                return pah_name

    return None


def get_polycyclic_core_atoms(mol, pah_name: str) -> Optional[Set[int]]:
    """
    Get the atom indices that form the PAH core.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Set of atom indices forming the PAH core, or None if no match
    """
    if pah_name not in POLYCYCLIC_DATA:
        return None

    pah_data = POLYCYCLIC_DATA[pah_name]
    smarts = pah_data['smarts']
    pattern = Chem.MolFromSmarts(smarts)

    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return set(matches[0])

    return None


def get_polycyclic_substituents(mol, pah_name: str) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to a polycyclic aromatic core.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Dict mapping IUPAC locant (1-indexed) to list of substituent info dicts.
        Each dict has keys: 'name' (str), 'atoms' (list of atom indices)

    Note:
        Position mapping for naphthalene is based on IUPAC numbering:
        - Atoms are numbered 1-8 around the periphery
        - Fusion carbons (4a, 8a) are not substituent positions
    """
    core_atoms = get_polycyclic_core_atoms(mol, pah_name)
    if core_atoms is None:
        return {}

    pah_data = POLYCYCLIC_DATA.get(pah_name, {})
    allowed_positions = pah_data.get('substituent_positions', [])

    # Get the SMARTS match for atom ordering
    smarts = pah_data.get('smarts', '')
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return {}

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return {}

    match_atoms = list(matches[0])

    # Identify the substituent attachment atoms (core atoms bearing a non-core
    # neighbour) so the numbering can be resolved by the automorphism that gives
    # THEM the lowest locants (P-14.3.5). The subset whose substituent is a
    # principal-characteristic-group suffix (carboxylic acid/aldehyde/amide/
    # nitrile) is tracked separately so it can claim the lowest locant FIRST
    # (P-14.4(c)) — the WR-01 PCG-anchor.
    attach_atoms = set()
    suffix_atoms = set()
    for idx in core_atoms:
        is_attach = False
        is_pcg = False
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in core_atoms:
                continue
            is_attach = True
            sub = _identify_pah_substituent(mol, nb.GetIdx(), core_atoms)
            if sub and sub.get('is_suffix'):
                is_pcg = True
        if is_attach:
            attach_atoms.add(idx)
        if is_pcg:
            suffix_atoms.add(idx)

    # v22 Phase E1 / DD4 (H1): use the authoritative stored ``iupac_numbering``
    # with automorphism-minimization (covers naphthalene/anthracene/phenanthrene/
    # pyrene/azulene). The naphthalene-only alpha/beta heuristic
    # (``_map_pah_atoms_to_iupac``) is kept ONLY as the fallback for cataloged
    # PAHs that still lack a populated ``iupac_numbering`` map (fluorene,
    # acenaphthene, ...). A regression test asserts the populated PAHs never
    # reach the heuristic.
    atom_to_position = get_polycyclic_iupac_locants(
        mol, pah_name, substituent_atoms=attach_atoms, pcg_atoms=suffix_atoms,
    )
    if atom_to_position is None:
        atom_to_position = _map_pah_atoms_to_iupac(mol, pah_name, match_atoms)

    substituents: Dict[int, List[Dict]] = defaultdict(list)

    for atom_idx in core_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the core
            if nbr_idx in core_atoms:
                continue

            # Identify the substituent
            sub_info = _identify_pah_substituent(mol, nbr_idx, core_atoms)
            if sub_info:
                # Get the IUPAC position for this attachment point
                position = atom_to_position.get(atom_idx)

                # Skip fusion carbons (non-integer positions like 4.5, 8.5)
                if position is None or not isinstance(position, int):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "PAH substituent at atom %d on %s mapped to position=%s (skipped)",
                        atom_idx, pah_name, position
                    )
                    continue

                # Only include if it's an allowed substituent position
                if position in allowed_positions or not allowed_positions:
                    substituents[position].append(sub_info)

    return dict(substituents)


def _map_pah_atoms_to_iupac(mol, pah_name: str, match_atoms: List[int]) -> Dict[int, int]:
    r"""
    Map PAH atom indices to IUPAC position numbers.

    For naphthalene, IUPAC numbering is:
        8  1
       /  \ /
      7    2
      |    |
      6    3
       \  / \
        5  4

    Positions 1, 4, 5, 8 are "alpha" (adjacent to fusion carbons 4a/8a).
    Positions 2, 3, 6, 7 are "beta" (NOT adjacent to fusion carbons).

    Key insight: The alpha/beta classification of an atom is a STRUCTURAL
    property that doesn't change with numbering. A methyl at an alpha position
    can only have locants 1, 4, 5, or 8. A methyl at a beta position can
    only have locants 2, 3, 6, or 7.

    The algorithm:
    1. Identify fusion atoms and classify peripheral atoms as alpha/beta
    2. Build the peripheral ring order by traversing
    3. For naphthalene, assign positions 1,2,3,4 to one ring and 5,6,7,8 to other
    4. Apply lowest-locant rule: positions 1,2 should have substituents
       before positions 8,7 (or 4,5 before 3,6)

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH
        match_atoms: Atom indices from SMARTS match

    Returns:
        Dict mapping atom index to IUPAC position (int for peripheral atoms)
    """
    from collections import defaultdict

    core_set = set(match_atoms)

    # Get ring info to identify fusion atoms
    ri = mol.GetRingInfo()

    # Find atoms shared by multiple rings (fusion atoms)
    atom_ring_count = defaultdict(int)
    for ring in ri.AtomRings():
        for atom_idx in ring:
            if atom_idx in core_set:
                atom_ring_count[atom_idx] += 1

    fusion_atoms = {idx for idx, count in atom_ring_count.items() if count > 1}
    peripheral_atoms = [idx for idx in match_atoms if idx not in fusion_atoms]

    if not peripheral_atoms:
        return {}

    # Build adjacency for atoms in the PAH core
    adjacency = defaultdict(set)
    for atom_idx in match_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in core_set:
                adjacency[atom_idx].add(nbr_idx)

    # Classify peripheral atoms as alpha (adjacent to fusion) or beta
    alpha_atoms = set()
    beta_atoms = set()
    for atom_idx in peripheral_atoms:
        if adjacency[atom_idx] & fusion_atoms:
            alpha_atoms.add(atom_idx)
        else:
            beta_atoms.add(atom_idx)

    # Find substituent atoms (atoms with non-core neighbors)
    substituted_atoms = set()
    for atom_idx in peripheral_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in core_set:
                substituted_atoms.add(atom_idx)
                break

    # For naphthalene, we need to assign:
    # - Alpha atoms get positions 1, 4, 5, 8 (corners)
    # - Beta atoms get positions 2, 3, 6, 7 (sides)
    # The LOWEST locant for an alpha substituent is 1
    # The LOWEST locant for a beta substituent is 2

    # Build the peripheral ring order
    peripheral_order = _build_peripheral_order(
        peripheral_atoms[0], peripheral_atoms, fusion_atoms, adjacency
    )

    if not peripheral_order or len(peripheral_order) != len(peripheral_atoms):
        # Fallback
        return {atom_idx: i + 1 for i, atom_idx in enumerate(peripheral_atoms)}

    # For naphthalene (8 peripheral atoms), the pattern alternates:
    # Position 1 (alpha) - Position 2 (beta) - Position 3 (beta) - Position 4 (alpha) -
    # Position 5 (alpha) - Position 6 (beta) - Position 7 (beta) - Position 8 (alpha)
    #
    # So in the traversal order, if we start from an alpha atom:
    # alpha - beta - beta - alpha - alpha - beta - beta - alpha
    #   1      2      3      4       5      6      7       8

    # Find the best starting point to minimize locants
    n = len(peripheral_order)
    best_mapping = None
    best_locants = None

    # For naphthalene, valid starting points must be alpha atoms (positions 1, 4, 5, 8)
    # and the traversal direction must give alpha-beta-beta-alpha pattern
    for start_idx in range(n):
        start_atom = peripheral_order[start_idx]
        if start_atom not in alpha_atoms:
            continue  # Position 1 must be alpha

        for direction in [1, -1]:
            # Check if this orientation gives valid alpha/beta pattern
            valid = True
            mapping = {}
            expected_alpha = [0, 3, 4, 7]  # Positions 1,4,5,8 (0-indexed: 0,3,4,7)

            for i in range(n):
                actual_idx = (start_idx + i * direction) % n
                atom_idx = peripheral_order[actual_idx]
                mapping[atom_idx] = i + 1

                # Check alpha/beta consistency
                is_alpha = atom_idx in alpha_atoms
                should_be_alpha = i in expected_alpha
                if is_alpha != should_be_alpha:
                    valid = False
                    break

            if not valid:
                continue

            # Get locants for substituted positions
            locants = sorted([mapping[pos] for pos in substituted_atoms if pos in mapping])

            # Compare with best using first-point-of-difference
            if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
                best_locants = locants
                best_mapping = mapping

    return best_mapping if best_mapping else {atom_idx: i + 1 for i, atom_idx in enumerate(peripheral_atoms)}


def _engine_canonical_numbering(canonical_mol) -> Dict[int, Any]:
    """Deterministic IUPAC numbering of a bare PAH skeleton via the S1 engine.

    Used by ``get_polycyclic_iupac_locants`` to supply a numbering for cataloged
    PAH entries whose ``iupac_numbering`` was never tabulated (tetracene,
    chrysene, triphenylene, benz[a]anthracene, ...).  Returns the numbering keyed
    by the canonical-mol atom index, in the SAME storage form as the tabulated
    maps (plain ``int`` for peripheral atoms, ``'4a'`` strings for fusion
    carbons) so the existing automorphism-minimization downstream is reused
    unchanged.

    Returns ``{}`` when the fusion-numbering engine declines (peri-fused,
    helicene, non-all-six, or heterocyclic systems) — the caller then falls back
    to its existing path, so this is strictly fail-closed (never a regression).

    Source: IUPAC 2013 P-25.3.2.3.3 / P-25.3.3.1 (rules/fusion_numbering.py).
    """
    from .fusion_numbering import compute_fused_numbering

    ring_atoms = {a.GetIdx() for a in canonical_mol.GetAtoms() if a.IsInRing()}
    engine = compute_fused_numbering(canonical_mol, ring_atoms)
    if engine is None:
        return {}
    out: Dict[int, Any] = {}
    for idx, locant in engine.items():
        out[idx] = f"{locant[0]}{locant[1]}" if isinstance(locant, tuple) else locant
    return out


def get_polycyclic_iupac_locants(
    mol,
    pah_name: str,
    substituent_atoms: Optional[Set[int]] = None,
    pcg_atoms: Optional[Set[int]] = None,
) -> Optional[Dict[int, Any]]:
    """Phase 147: return authoritative IUPAC locants for a cataloged PAH.

    Reads ``POLYCYCLIC_DATA[pah_name]['iupac_numbering']`` (canonical-SMILES
    keyed) and translates canonical-atom indices to mol-atom indices via
    RDKit substructure match. String fusion locants ('4a', '10b', ...)
    are converted to ``(int, str)`` tuples at this boundary per Phase 147
    Decision D-01 (compare_locant_sets tuple-aware after Plan 01).

    Returns a dict covering ALL ring atoms of the PAH, mixing plain int
    locants (peripheral) with ``(int, str)`` tuple locants (fusion atoms).
    Suitable for ``ring_info["iupac_locants"]`` in the cascade step 6 gate.

    The 4 populated PAHs in v17 are: naphthalene (10 keys, 2 fusion tuples),
    anthracene (14 keys, 4 fusion tuples), phenanthrene (14 keys, 4 fusion
    tuples), pyrene (16 keys, 6 fusion tuples). Other entries with empty
    ``iupac_numbering`` (fluorene, acenaphthene, ...) return None — Phase 151
    audit candidates.

    Args:
        mol: RDKit Mol object.
        pah_name: Canonical PAH name (key in POLYCYCLIC_DATA).

    Returns:
        ``Dict[int, int | (int, str)]`` when ``pah_name`` is a populated PAH.
        ``None`` when ``pah_name`` is not in POLYCYCLIC_DATA, its
        ``iupac_numbering`` is empty, or the substructure match fails.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25 (PAH numbering
        is FIXED, not reoriented per substituents — match data directly).
    Source: Phase 147 CONTEXT D-01 (tuple encoding); CD-03 (function form).
    """
    import re

    entry = POLYCYCLIC_DATA.get(pah_name)
    if entry is None:
        return None

    canonical_smiles = entry.get('canonical_smiles')
    if not canonical_smiles:
        return None
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None:
        return None

    canonical_numbering = entry.get('iupac_numbering') or {}
    if not canonical_numbering:
        # v23 13B(a) S1: a real PAH entry whose IUPAC numbering was never
        # tabulated (tetracene, chrysene, triphenylene, benz[a]anthracene,
        # picene, pentacene, ...).  Derive it deterministically from the bare
        # skeleton with the fusion-numbering engine; the SAME automorphism-
        # minimization below then assigns the lowest substituent locants.
        # Fail-closed: the engine returns {} for peri-fused / helicene /
        # non-all-six systems, so we return None and the caller keeps its
        # existing fallback (no regression).
        canonical_numbering = _engine_canonical_numbering(canonical_mol)
        if not canonical_numbering:
            return None

    # Anchored regex: parse 'NN' or 'NNa' / 'NNb' style locants.
    # Two-digit base critical for pyrene's '10b' (must yield (10, 'b'),
    # NOT (1, '0b')).
    locant_re = re.compile(r'^(\d+)([a-z]*)$')

    def _build_map(match) -> Dict[int, Any]:
        result: Dict[int, Any] = {}
        for canonical_idx, locant in canonical_numbering.items():
            if canonical_idx >= len(match):
                continue
            mol_idx = match[canonical_idx]
            if isinstance(locant, int):
                result[mol_idx] = locant
            elif isinstance(locant, str):
                m = locant_re.match(locant)
                if m is None:
                    # Malformed locant — skip defensively (should never fire
                    # for the populated entries).
                    continue
                base = int(m.group(1))
                suffix = m.group(2)
                result[mol_idx] = (base, suffix) if suffix else base
            # Any other type silently skipped (defensive).
        return result

    # v22 Phase E1 / DD4 (H1): when the caller passes the substituent-bearing
    # core atoms, enumerate ALL automorphic substructure matches of the fixed
    # canonical numbering and choose the one giving the substituents the lowest
    # locant set (P-25.3.3.1.2(a) + P-14.3.5). This replaces the
    # naphthalene-only alpha/beta heuristic for cataloged PAHs and is what makes
    # 2-methylanthracene / anthracen-2-amine / phenanthren-3-amine come out as
    # the PIN instead of an input-order-dependent first match. When
    # ``substituent_atoms`` is None the first match is used (byte-identical to
    # the pre-E1 unsubstituted/parent-selection path).
    if substituent_atoms:
        from .locants import compare_numbering

        matches = mol.GetSubstructMatches(canonical_mol, uniquify=False)
        sub_set = set(substituent_atoms)
        # P-14.4(c) (WR-01 fix): the principal-characteristic-group atoms (the
        # suffix-bearing carbons) get the lowest locant BEFORE detachable prefix
        # substituents — via compare_numbering's 'pcg' tier, ahead of the
        # kind-agnostic substituent set. When pcg_atoms is None/empty the tier is
        # inert and this reduces to the prior substituent-set minimization
        # (byte-identical for prefix-only and single-suffix PAHs).
        pcg_set = set(pcg_atoms or ())
        n_canonical = canonical_mol.GetNumAtoms()
        best_map: Optional[Dict[int, Any]] = None
        best_locs: Optional[List[Any]] = None
        best_pcg: Optional[List[Any]] = None
        for match in matches:
            if len(match) != n_canonical:
                continue
            mp = _build_map(match)
            locs = [mp[a] for a in sub_set if a in mp]
            pcg_locs = [mp[a] for a in pcg_set if a in mp]
            if best_map is None or compare_numbering(
                    {'pcg': pcg_locs, 'substituents': locs},
                    {'pcg': best_pcg, 'substituents': best_locs}) < 0:
                best_map, best_locs, best_pcg = mp, locs, pcg_locs
        return best_map

    match = mol.GetSubstructMatch(canonical_mol)
    if not match or len(match) != canonical_mol.GetNumAtoms():
        return None
    return _build_map(match)


def _build_peripheral_order(
    start: int,
    peripheral_atoms: List[int],
    fusion_atoms: Set[int],
    adjacency: Dict[int, Set[int]]
) -> List[int]:
    """
    Build an ordered list of peripheral atoms by traversing around the ring system.

    For naphthalene, this produces the 8 peripheral atoms in order around the
    outer edge, skipping the fusion atoms.
    """
    peripheral_set = set(peripheral_atoms)
    order = []
    visited = set()
    current = start

    while len(order) < len(peripheral_atoms):
        if current in visited:
            break

        visited.add(current)
        order.append(current)

        # Find next unvisited peripheral neighbor
        found_next = False
        for nbr in adjacency[current]:
            if nbr in peripheral_set and nbr not in visited:
                current = nbr
                found_next = True
                break

        if not found_next:
            # Need to go through a fusion atom
            for nbr in adjacency[current]:
                if nbr in fusion_atoms:
                    for next_nbr in adjacency[nbr]:
                        if next_nbr in peripheral_set and next_nbr not in visited:
                            current = next_nbr
                            found_next = True
                            break
                    if found_next:
                        break

        if not found_next:
            break

    return order


def _identify_pah_substituent(mol, start_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a substituent attached to the PAH core.

    Similar to benzene substituent identification but excludes core atoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom attached to the core
        core_atoms: Set of atom indices forming the PAH core

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens - single atom substituents
    halogen_names = {
        'F': 'fluoro',
        'Cl': 'chloro',
        'Br': 'bromo',
        'I': 'iodo',
    }
    if symbol in halogen_names:
        return {
            'name': halogen_names[symbol],
            'atoms': [start_idx]
        }

    # WS-A task 9: ring-system substituents (phenyl, pyridin-2-yl,
    # naphthalen-2-yl, ...) must NEVER reach the chain identifiers below —
    # the alkyl namer counted an all-C/H RING as a chain (phenyl -> 'hexyl',
    # a DIFFERENT molecule) and silently dropped heteroaryl fragments.
    # Collect the fragment; if it contains ring atoms, delegate to the
    # single ring-substituent chokepoint.
    _visited = {start_idx}
    _queue = deque([start_idx])
    _frag_atoms = []
    while _queue:
        _cur = _queue.popleft()
        _frag_atoms.append(_cur)
        for _nbr in mol.GetAtomWithIdx(_cur).GetNeighbors():
            _ni = _nbr.GetIdx()
            if _ni not in _visited and _ni not in core_atoms:
                _visited.add(_ni)
                _queue.append(_ni)
    _ring_info = mol.GetRingInfo()
    if any(_ring_info.NumAtomRings(a) > 0 for a in _frag_atoms):
        _frag_set = set(_frag_atoms)
        # A ring straddling the core boundary = incomplete core mapping ->
        # this is a mis-mapped core atom, not a substituent.
        for _r in _ring_info.AtomRings():
            _r_set = set(_r)
            if (_r_set & _frag_set) and (_r_set & core_atoms):
                return None
        from .ring_substituents import name_ring_system_substituent
        _ring_sub_name = name_ring_system_substituent(mol, _frag_atoms, start_idx)
        if _ring_sub_name:
            return {'name': _ring_sub_name, 'atoms': _frag_atoms}
        return None

    # Carbon-based groups (alkyl or functionalized chain)
    if symbol == 'C':
        alkyl = _identify_pah_alkyl_group(mol, start_idx, core_atoms)
        if alkyl:
            return alkyl
        # Try functionalized chain (hydroxymethyl, carboxymethyl, etc.)
        return _identify_pah_functionalized_chain(mol, start_idx, core_atoms)

    # Nitrogen groups
    if symbol == 'N':
        return _identify_pah_nitrogen_group(mol, start_idx, core_atoms)

    # Oxygen groups
    if symbol == 'O':
        return _identify_pah_oxygen_group(mol, start_idx, core_atoms)

    return None


def _identify_pah_alkyl_group(mol, start_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify alkyl substituent on PAH.

    Uses BFS to find all connected carbons and determines name from carbon count.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = deque([start_idx])
    carbon_count = 0
    all_atoms = []

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in core_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if this is a pure alkyl (only carbons and hydrogens)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            # Contains heteroatom - not a simple alkyl
            return None

    # Get alkyl name from carbon count
    alkyl_names = {
        1: 'methyl',
        2: 'ethyl',
        3: 'propyl',
        4: 'butyl',
        5: 'pentyl',
        6: 'hexyl',
        7: 'heptyl',
        8: 'octyl',
        9: 'nonyl',
        10: 'decyl',
    }

    if carbon_count in alkyl_names:
        return {
            'name': alkyl_names[carbon_count],
            'atoms': all_atoms
        }

    return None


def _identify_pah_nitrogen_group(mol, n_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """Identify nitrogen-based substituent on PAH."""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Count neighbors (excluding core)
    neighbors = [n for n in n_atom.GetNeighbors() if n.GetIdx() not in core_atoms]

    # Check for nitro group: N with 2 oxygens, positive charge
    if n_atom.GetFormalCharge() == 1:
        o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        if o_count == 2:
            atoms = [n_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
            return {'name': 'nitro', 'atoms': atoms}

    # Simple amino (-NH2)
    h_count = n_atom.GetTotalNumHs()
    if h_count == 2 and len(neighbors) == 0:
        return {'name': 'amino', 'atoms': [n_idx]}

    return None


def _identify_pah_oxygen_group(mol, o_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """Identify oxygen-based substituent on PAH.

    Handles:
    - Hydroxy (-OH)
    - Methoxy (-OCH3)
    - Ethoxy (-OCH2CH3) and higher alkoxy
    - Acetyloxy (-OC(=O)CH3) and similar acyloxy groups
    """
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Count neighbors (excluding core)
    neighbors = [n for n in o_atom.GetNeighbors() if n.GetIdx() not in core_atoms]

    # Simple hydroxy (-OH)
    h_count = o_atom.GetTotalNumHs()
    if h_count == 1 and len(neighbors) == 0:
        return {'name': 'hydroxy', 'atoms': [o_idx]}

    # Alkoxy groups (-OR where R is alkyl)
    if h_count == 0 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        c_atom = neighbors[0]
        c_idx = c_atom.GetIdx()

        # BFS to find alkyl chain after oxygen
        visited = {c_idx}
        queue = deque([c_idx])
        alkyl_atoms = []
        carbon_count = 0
        is_pure_alkyl = True

        while queue:
            current_idx = queue.popleft()
            current_atom = mol.GetAtomWithIdx(current_idx)
            alkyl_atoms.append(current_idx)

            if current_atom.GetSymbol() == 'C':
                carbon_count += 1
            elif current_atom.GetSymbol() not in ('C', 'H'):
                is_pure_alkyl = False
                break

            for nbr in current_atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited and nbr_idx not in core_atoms and nbr_idx != o_idx:
                    visited.add(nbr_idx)
                    queue.append(nbr_idx)

        if is_pure_alkyl and carbon_count > 0:
            alkoxy_names = {
                1: 'methoxy',
                2: 'ethoxy',
                3: 'propoxy',
                4: 'butoxy',
                5: 'pentyloxy',
                6: 'hexyloxy',
            }
            alkoxy_name = alkoxy_names.get(carbon_count)
            if alkoxy_name:
                return {'name': alkoxy_name, 'atoms': [o_idx] + alkyl_atoms}

    return None


def _identify_pah_functionalized_chain(
    mol, start_idx: int, core_atoms: Set[int]
) -> Optional[Dict]:
    """
    Identify functionalized chain substituent on PAH.

    Handles carbon chains with terminal functional groups that are not
    simple alkyl groups. Called when _identify_pah_alkyl_group returns None
    due to heteroatom presence.

    Recognizes functional groups that should be expressed as suffixes:
    - -COOH → 'carboxylic acid' (suffix)
    - -CHO → 'carbaldehyde' (suffix)
    - -CONH2 → 'carboxamide' (suffix)
    - -CN → 'carbonitrile' (suffix)
    And prefix-only substituents:
    - -CH2OH → 'hydroxymethyl'
    """
    # BFS to collect chain atoms
    visited = {start_idx}
    queue = deque([start_idx])
    chain_atoms = []
    carbon_count = 0

    while queue:
        idx = queue.popleft()
        chain_atoms.append(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            carbon_count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in core_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    if not chain_atoms or carbon_count == 0:
        return None

    chain_set = set(chain_atoms)

    # Check for carboxylic acid (-COOH): C(=O)(OH) where C is start_idx
    acid_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if acid_pattern:
        matches = mol.GetSubstructMatches(acid_pattern)
        for match in matches:
            if match[0] == start_idx:
                return {'name': 'carboxylic acid', 'atoms': chain_atoms,
                        'is_suffix': True, 'suffix_type': 'carboxylic_acid'}

    # Check for aldehyde (-CHO): C(=O)H where C is start_idx
    # CX3H1 because C is bonded to ring, O (double), and H
    ald_pattern = Chem.MolFromSmarts('[CX3H1](=O)')
    if ald_pattern:
        matches = mol.GetSubstructMatches(ald_pattern)
        for match in matches:
            if match[0] == start_idx:
                return {'name': 'carbaldehyde', 'atoms': chain_atoms,
                        'is_suffix': True, 'suffix_type': 'aldehyde'}

    # Check for amide (-CONH2): C(=O)(NH2)
    amide_pattern = Chem.MolFromSmarts('[CX3](=O)[NX3H2]')
    if amide_pattern:
        matches = mol.GetSubstructMatches(amide_pattern)
        for match in matches:
            if match[0] == start_idx:
                return {'name': 'carboxamide', 'atoms': chain_atoms,
                        'is_suffix': True, 'suffix_type': 'primary_amide'}

    # Check for nitrile (-C≡N)
    nitrile_pattern = Chem.MolFromSmarts('[CX2]#[NX1]')
    if nitrile_pattern:
        matches = mol.GetSubstructMatches(nitrile_pattern)
        for match in matches:
            if match[0] == start_idx:
                return {'name': 'carbonitrile', 'atoms': chain_atoms,
                        'is_suffix': True, 'suffix_type': 'nitrile'}

    # Check for hydroxyl terminus (-CH2OH, -CH2CH2OH) - must NOT have C=O
    # This distinguishes -CH2OH (hydroxymethyl) from -COOH (carboxylic acid)
    carbonyl_on_start = False
    start_atom = mol.GetAtomWithIdx(start_idx)
    for nbr in start_atom.GetNeighbors():
        if nbr.GetSymbol() == 'O' and nbr.GetIdx() not in core_atoms:
            bond = mol.GetBondBetweenAtoms(start_idx, nbr.GetIdx())
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                carbonyl_on_start = True
                break

    if not carbonyl_on_start:
        hydroxyl_pattern = Chem.MolFromSmarts('[OX2H1]')
        if hydroxyl_pattern:
            matches = mol.GetSubstructMatches(hydroxyl_pattern)
            for match in matches:
                if match[0] in chain_set:
                    hydroxyl_names = {
                        1: 'hydroxymethyl',
                        2: '2-hydroxyethyl',
                        3: '3-hydroxypropyl',
                    }
                    name = hydroxyl_names.get(carbon_count)
                    if name:
                        return {'name': name, 'atoms': chain_atoms}

    return None


def name_substituted_polycyclic(
    mol,
    pah_name: str,
    substituents: Dict[int, List[Dict]]
) -> str:
    """
    Generate IUPAC name for a substituted polycyclic aromatic.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH parent (e.g., 'naphthalene')
        substituents: Dict from get_polycyclic_substituents

    Returns:
        IUPAC name string (e.g., '2-methylnaphthalene')

    IUPAC Rules:
    - Locants are FIXED to IUPAC standard numbering
    - Alphabetize substituent prefixes
    - Use multiplicative prefixes (di-, tri-) for repeated substituents
    - Format: locants-substituent-parent
    - Functional groups as suffixes: parent-locant-suffix (e.g., naphthalene-2-carboxylic acid)
    """
    if not substituents:
        return pah_name

    # Collect stereodescriptors for chiral substituents on PAH
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    assign_stereochemistry(mol)

    # Build atom_to_locant from the PAH numbering (used for stereodescriptor
    # locants). E1/DD4: prefer the authoritative stored numbering with
    # automorphism-min over substituent-bearing atoms (consistent with
    # get_polycyclic_substituents); fall back to the SMARTS + naphthalene
    # heuristic only when the PAH has no populated iupac_numbering.
    pah_data = POLYCYCLIC_DATA.get(pah_name, {})
    smarts = pah_data.get('smarts', '')
    pattern = Chem.MolFromSmarts(smarts) if smarts else None
    atom_to_locant = {}
    if pattern:
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            match_atoms = list(matches[0])
            core_set = set(match_atoms)
            # Mirror get_polycyclic_substituents EXACTLY (incl. the WR-01 PCG
            # subset) so the stereodescriptor numbering cannot desync from the
            # substituent numbering (WR-03).
            attach_atoms = set()
            suffix_atoms = set()
            for idx in core_set:
                is_attach = False
                is_pcg = False
                for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nb.GetIdx() in core_set:
                        continue
                    is_attach = True
                    sub = _identify_pah_substituent(mol, nb.GetIdx(), core_set)
                    if sub and sub.get('is_suffix'):
                        is_pcg = True
                if is_attach:
                    attach_atoms.add(idx)
                if is_pcg:
                    suffix_atoms.add(idx)
            atom_to_locant = get_polycyclic_iupac_locants(
                mol, pah_name, substituent_atoms=attach_atoms, pcg_atoms=suffix_atoms,
            )
            if atom_to_locant is None:
                atom_to_locant = _map_pah_atoms_to_iupac(mol, pah_name, match_atoms)

    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant) if atom_to_locant else []
    stereo_prefix = format_stereodescriptor_string(stereo_descriptors) if stereo_descriptors else ""

    # Separate suffix-type FGs from prefix-type substituents
    suffix_groups: Dict[str, List[int]] = defaultdict(list)  # suffix_name -> [locants]
    prefix_substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for position, sub_list in substituents.items():
        for sub_info in sub_list:
            if sub_info.get('is_suffix'):
                suffix_groups[sub_info['name']].append(position)
            else:
                prefix_substituent_groups[sub_info['name']].append(position)

    # Sort locants within each group
    for name in prefix_substituent_groups:
        prefix_substituent_groups[name].sort()
    for name in suffix_groups:
        suffix_groups[name].sort()

    # Build prefix strings, sorted alphabetically by substituent name
    prefixes = []
    for name in sorted(prefix_substituent_groups.keys(), key=alpha_sort_key):
        locants = prefix_substituent_groups[name]
        count = len(locants)
        prefix_str = format_substituent_prefix(name, locants, count)
        prefixes.append(prefix_str)

    # Join prefixes with proper hyphenation
    prefix_part = _join_pah_prefixes(prefixes)

    # Handle suffix-type functional groups
    if suffix_groups:
        # Pick the highest-priority suffix (carboxylic acid > aldehyde > amide, etc.)
        _SUFFIX_PRIORITY = [
            'carboxylic acid', 'carboxamide', 'carbonitrile', 'carbaldehyde',
        ]
        chosen_suffix = None
        chosen_locants = []
        for suf in _SUFFIX_PRIORITY:
            if suf in suffix_groups:
                chosen_suffix = suf
                chosen_locants = suffix_groups[suf]
                break
        if not chosen_suffix:
            # Fallback: pick first
            chosen_suffix = next(iter(suffix_groups))
            chosen_locants = suffix_groups[chosen_suffix]

        # Build suffix with locant(s)
        from ..assembly.naming_utils import get_multiplier_prefix
        count = len(chosen_locants)
        multiplier = get_multiplier_prefix(count, chosen_suffix) if count > 1 else ""
        locant_str = ",".join(str(loc) for loc in chosen_locants)

        # Assemble: prefix-part + parent-locant-suffix
        # e.g., "naphthalene-2-carboxylic acid", "3-chloronaphthalene-2-carbaldehyde"
        suffix_part = f"-{locant_str}-{multiplier}{chosen_suffix}"

        # Any remaining suffix groups become prefixes (carboxy, formyl, etc.)
        from ..rules.seniority import PREFIX_FORMS
        _SUFFIX_TO_PREFIX = {
            'carboxylic acid': 'carboxy',
            'carbaldehyde': 'formyl',
            'carboxamide': 'carbamoyl',
            'carbonitrile': 'cyano',
        }
        for suf_name, suf_locants in suffix_groups.items():
            if suf_name == chosen_suffix:
                continue
            prefix_name = _SUFFIX_TO_PREFIX.get(suf_name, suf_name)
            prefix_str = format_substituent_prefix(prefix_name, sorted(suf_locants), len(suf_locants))
            prefixes.append(prefix_str)
            prefix_part = _join_pah_prefixes(sorted(prefixes, key=lambda s: alpha_sort_key(s.lstrip('0123456789,-'))))

        # P-31.1.4.3.4 (Wave-2 completion): the indicated hydrogen attaches
        # to the parent stem, after any substituent prefixes -- 9H-fluorene-
        # 9-carboxylic acid / 9-methyl-9H-fluorene. The bare-parent early
        # return keeps the retained short form ('fluorene').
        _ih = POLYCYCLIC_DATA.get(pah_name, {}).get('indicated_h')
        _stem = f"{_ih}-{pah_name}" if _ih else pah_name
        _sep = '-' if (_ih and prefix_part and not prefix_part.endswith('-')) else ''
        name = f"{prefix_part}{_sep}{_stem}{suffix_part}"
        return f"{stereo_prefix}{name}" if stereo_prefix else name

    # Build final name (prefix-only, no suffix FGs)
    _ih = POLYCYCLIC_DATA.get(pah_name, {}).get('indicated_h')
    _stem = f"{_ih}-{pah_name}" if _ih else pah_name
    _sep = '-' if (_ih and prefix_part and not prefix_part.endswith('-')) else ''
    name = f"{prefix_part}{_sep}{_stem}"
    return f"{stereo_prefix}{name}" if stereo_prefix else name


def _join_pah_prefixes(prefixes: List[str]) -> str:
    """
    Join PAH substituent prefixes with proper hyphenation.

    Args:
        prefixes: List of formatted prefix strings

    Returns:
        Joined prefix string
    """
    if not prefixes:
        return ""

    if len(prefixes) == 1:
        return prefixes[0]

    result = prefixes[0]
    for i in range(1, len(prefixes)):
        current = prefixes[i]

        # Check if we need a hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]

            if last_char.isalpha() and first_char.isdigit():
                result += "-"

        result += current

    return result


def get_pah_substituent_locants(
    mol,
    core_match: Dict[int, int]
) -> Dict[int, str]:
    """
    Find atoms not in the PAH core and map them to IUPAC locants.

    For substituted PAHs, identifies substituent atoms and maps them
    to their attachment point locants.

    Args:
        mol: RDKit Mol object
        core_match: Dict mapping atom index to IUPAC locant (from match_polycyclic_core)

    Returns:
        Dict mapping IUPAC locant -> substituent name

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> from src.orthonym.data.polycyclic_data import match_polycyclic_core
        >>> _, core_match = match_polycyclic_core(mol)
        >>> get_pah_substituent_locants(mol, core_match)
        {2: 'methyl'}
    """
    core_atoms = set(core_match.keys())
    result = {}

    for atom_idx in core_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        locant = core_match[atom_idx]

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in core_atoms:
                # This is a substituent
                sub_info = _identify_pah_substituent(mol, nbr_idx, core_atoms)
                if sub_info:
                    result[locant] = sub_info['name']

    return result


def is_peri_condensed(mol) -> bool:
    """
    Detect peri-condensed PAH systems.

    Peri-condensed PAHs have interior atoms (not on the periphery) that are
    shared by more than two rings. Examples include pyrene, perylene, and coronene.

    In full IUPAC numbering, these interior atoms may have letter suffixes
    (like 4a, 8a in naphthalene for fusion atoms, but more complex in peri systems).

    Args:
        mol: RDKit Mol object

    Returns:
        True if the molecule is a peri-condensed PAH

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> is_peri_condensed(mol)
        False
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')  # pyrene
        >>> is_peri_condensed(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61')  # coronene
        >>> is_peri_condensed(mol)
        True
    """
    # Peri-condensed PAHs have atoms shared by MORE than 2 rings
    # Ortho-fused PAHs only have atoms shared by exactly 2 rings

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 3:
        return False  # Need at least 3 rings for peri-condensation

    # Count how many rings each atom belongs to
    atom_ring_count = defaultdict(int)
    for ring in atom_rings:
        for atom_idx in ring:
            atom_ring_count[atom_idx] += 1

    # Check for atoms in 3 or more rings (peri-condensation)
    for count in atom_ring_count.values():
        if count >= 3:
            return True

    return False


def get_pah_type(mol) -> str:
    """
    Classify PAH as ortho-fused, peri-condensed, or not a PAH.

    Args:
        mol: RDKit Mol object

    Returns:
        'peri-condensed': PAH with interior atoms shared by 3+ rings
        'ortho-fused': Linear PAH with only edge fusion
        'not-pah': Not a polycyclic aromatic hydrocarbon

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> get_pah_type(mol)
        'ortho-fused'
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')  # pyrene
        >>> get_pah_type(mol)
        'peri-condensed'
    """
    pah_name = identify_polycyclic(mol)
    if not pah_name:
        return 'not-pah'

    if is_peri_condensed(mol):
        return 'peri-condensed'

    return 'ortho-fused'


def name_polycyclic(mol) -> Optional[str]:
    """
    Generate IUPAC name for a polycyclic aromatic hydrocarbon.

    Main entry point for PAH naming. Handles:
    1. Fully aromatic PAHs (naphthalene, anthracene, etc.)
    2. Substituted PAHs (2-methylnaphthalene)
    3. Partially saturated PAHs (tetrahydronaphthalene)

    The routing order is:
    1. Check for partially saturated carbocycles (tetrahydronaphthalene, etc.)
    2. Check for fully aromatic PAHs
    3. Return None if not a recognized PAH

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string if PAH identified, None otherwise

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        >>> name_polycyclic(mol)
        'naphthalene'
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')
        >>> name_polycyclic(mol)
        '2-methylnaphthalene'
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')  # pyrene
        >>> name_polycyclic(mol)
        'pyrene'
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')  # tetrahydronaphthalene
        >>> name_polycyclic(mol)
        '1,2,3,4-tetrahydronaphthalene'
    """
    # Check for partially saturated carbocycles FIRST
    partial_sat_name = name_partially_saturated_carbocycle(mol)
    if partial_sat_name:
        return partial_sat_name

    # Then check fully aromatic PAHs
    pah_name = identify_polycyclic(mol)
    if not pah_name:
        return None

    substituents = get_polycyclic_substituents(mol, pah_name)
    return name_substituted_polycyclic(mol, pah_name, substituents)


def name_partially_saturated_carbocycle(mol) -> Optional[str]:
    """
    Generate IUPAC name for a partially saturated carbocyclic fused system.

    Handles compounds like tetrahydronaphthalene, dihydroanthracene, etc.
    These are PAH systems with some ring atoms saturated (sp3).

    IUPAC 2013 format: [locants]-[prefix][parent]
    Example: 1,2,3,4-tetrahydronaphthalene

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string if partially saturated carbocycle detected,
        None otherwise.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        >>> name_partially_saturated_carbocycle(mol)
        '1,2,3,4-tetrahydronaphthalene'
    """
    from .partial_saturation import (
        detect_carbocyclic_partial_saturation,
        format_saturation_prefix,
        get_saturation_locants,
    )

    if mol is None:
        return None

    # Get all ring atoms
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    if len(ring_atoms) < 9:  # Need at least 9 atoms for fused bicyclic
        return None

    # Detect partial saturation
    sat_info = detect_carbocyclic_partial_saturation(mol, ring_atoms)
    if sat_info is None:
        return None

    # Build the name
    return _assemble_partially_saturated_carbocycle_name(mol, sat_info)


def _assemble_partially_saturated_carbocycle_name(
    mol,
    saturation_info: Dict[str, Any]
) -> str:
    """
    Assemble IUPAC name for partially saturated carbocyclic system.

    Format: [stereo][locants]-[prefix][parent]
    Example: (1R)-1,2,3,4-tetrahydronaphthalene

    Args:
        mol: RDKit Mol object
        saturation_info: Dict from detect_carbocyclic_partial_saturation

    Returns:
        Complete IUPAC name
    """
    from .partial_saturation import format_saturation_prefix, get_saturation_locants
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from ..perception.stereo import assign_stereochemistry

    parent_name = saturation_info['parent_name']
    prefix = saturation_info['prefix']
    saturated_indices = saturation_info['saturated_indices']
    atom_to_locant = saturation_info.get('atom_to_locant', {})
    is_perhydro = saturation_info['is_perhydro']

    # Collect stereodescriptors using the ring system's locant mapping (idempotent guard)
    assign_stereochemistry(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant) if atom_to_locant else []
    stereo_prefix = format_stereodescriptor_string(stereo_descriptors) if stereo_descriptors else ""

    # Get locants for saturated positions
    if is_perhydro:
        # Perhydro doesn't need locants
        formatted_prefix = 'perhydro'
    else:
        locants = get_saturation_locants(mol, saturated_indices, atom_to_locant)
        formatted_prefix = format_saturation_prefix(prefix, locants)

    # Assemble final name
    name = f"{formatted_prefix}{parent_name}"
    return f"{stereo_prefix}{name}" if stereo_prefix else name


# ============================================================================
# Fused Aromatic System Integration
# ============================================================================


def is_fused_aromatic_system(mol) -> bool:
    """
    Check if molecule contains a fused aromatic ring system.

    A fused aromatic system has 2+ aromatic rings sharing edges.
    This includes both carbocyclic PAHs and fused heterocycles.

    Args:
        mol: RDKit Mol object

    Returns:
        True if fused aromatic system detected

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> is_fused_aromatic_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
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
            shared = set(ring1) & set(ring2)
            if len(shared) >= 2:
                return True

    return False


def get_fused_aromatic_core(mol) -> Optional[str]:
    """
    Identify if molecule contains a known fused aromatic core.

    Checks both carbocyclic PAHs (naphthalene, anthracene) and
    fused heterocycles (indole, quinoline).

    Args:
        mol: RDKit Mol object

    Returns:
        Core name if found (e.g., 'naphthalene', '1H-indole'), None otherwise

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        >>> get_fused_aromatic_core(mol)
        'naphthalene'
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        >>> get_fused_aromatic_core(mol)
        '1H-indole'
    """
    # Import here to avoid circular dependency
    from .fused_rings import name_fused_heterocycle

    # Check fused heterocycles FIRST (they take priority)
    heterocycle_result = name_fused_heterocycle(mol)
    if heterocycle_result:
        # name_fused_heterocycle returns a tuple (name, ring_atoms, atom_to_locant, subs_included)
        return heterocycle_result[0] if isinstance(heterocycle_result, tuple) else heterocycle_result

    # Check carbocyclic PAHs
    pah_name = identify_polycyclic(mol)
    if pah_name:
        return pah_name

    return None


def name_substituted_fused_aromatic(
    mol,
    core_name: str,
    atom_map: Optional[Dict[int, int]] = None
) -> str:
    """
    Generate name for a substituted fused aromatic system.

    Handles substituents on both PAHs and fused heterocycles with
    consistent locant assignment using IUPAC numbering.

    Args:
        mol: RDKit Mol object
        core_name: Base core name (e.g., 'naphthalene', '1H-indole')
        atom_map: Optional pre-computed atom index to locant mapping

    Returns:
        Complete IUPAC name with substituent prefixes

    Examples:
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> name_substituted_fused_aromatic(mol, 'naphthalene')
        '2-methylnaphthalene'
    """
    # Import here to avoid circular dependency
    from .fused_rings import name_fused_heterocycle, match_fused_heterocycle_core
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

    # Check if this is a fused heterocycle
    # Look for tautomer locant patterns (1H-, 2H-, 9H-) in name
    is_heterocycle = any(
        core_name == data['name']
        for data in FUSED_HETEROCYCLE_DATA.values()
    )

    if is_heterocycle:
        # Delegate to fused_rings module for heterocycle naming
        # name_fused_heterocycle returns a tuple (name, ring_atoms, atom_to_locant, subs_included)
        fused_result = name_fused_heterocycle(mol)
        if fused_result:
            return fused_result[0] if isinstance(fused_result, tuple) else fused_result
        return core_name

    # Handle carbocyclic PAH
    pah_name = identify_polycyclic(mol)
    if pah_name:
        substituents = get_polycyclic_substituents(mol, pah_name)
        return name_substituted_polycyclic(mol, pah_name, substituents)

    return core_name


def identify_fused_system(mol) -> Optional[Dict[str, Any]]:
    """
    Identify and classify a fused ring system.

    Central coordinator for fused system identification. Returns
    information about the type of fused system and its naming.

    Args:
        mol: RDKit Mol object

    Returns:
        Dict with:
        - 'type': 'carbocyclic' or 'heterocyclic'
        - 'core_name': Base name of the fused system
        - 'is_substituted': Whether molecule has substituents on core
        - 'full_name': Complete IUPAC name
        Or None if not a recognized fused system

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        >>> info = identify_fused_system(mol)
        >>> info['type']
        'carbocyclic'
        >>> info['core_name']
        'naphthalene'
    """
    # Import here to avoid circular dependency
    from .fused_rings import (
        name_fused_heterocycle,
        classify_fused_system,
        is_fused_heterocyclic_system,
    )

    if not is_fused_aromatic_system(mol):
        return None

    # Determine fusion type
    fusion_type = classify_fused_system(mol)
    if fusion_type == 'not-fused':
        return None

    # Check for fused heterocycle first
    if is_fused_heterocyclic_system(mol):
        heterocycle_result = name_fused_heterocycle(mol)
        if heterocycle_result:
            # name_fused_heterocycle returns a tuple (name, ring_atoms, atom_to_locant, subs_included)
            heterocycle_name = heterocycle_result[0] if isinstance(heterocycle_result, tuple) else heterocycle_result
            # Check if substituted
            canonical = Chem.MolToSmiles(mol, canonical=True)
            from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
            is_substituted = canonical not in FUSED_HETEROCYCLE_DATA

            return {
                'type': 'heterocyclic',
                'core_name': heterocycle_name.split('-')[-1] if '-' in heterocycle_name else heterocycle_name,
                'is_substituted': is_substituted,
                'full_name': heterocycle_name,
                'fusion_type': fusion_type,
            }

    # Check for carbocyclic PAH
    pah_name = identify_polycyclic(mol)
    if pah_name:
        substituents = get_polycyclic_substituents(mol, pah_name)
        is_substituted = bool(substituents)
        full_name = name_substituted_polycyclic(mol, pah_name, substituents)

        return {
            'type': 'carbocyclic',
            'core_name': pah_name,
            'is_substituted': is_substituted,
            'full_name': full_name,
            'fusion_type': fusion_type,
        }

    return None
