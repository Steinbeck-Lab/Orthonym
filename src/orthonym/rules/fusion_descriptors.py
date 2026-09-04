"""
Fusion descriptor generation for systematic fusion names.

Generates IUPAC fusion descriptors for naming fused ring systems when no
retained name exists. Used for systematic names like benzo[a]anthracene,
naphtho[2,1-b]furan, etc.

IUPAC 2013 Fusion Descriptor Rules:
- Letter locants (a, b, c...) designate edges of the PARENT ring
- Edge 'a' is between atoms 1-2, edge 'b' is between 2-3, etc.
- Numerical locants indicate which atoms of the CHILD ring are fused
- Format: child[child_locants-letter]parent (e.g., benzo[a]anthracene)
- IUPAC 2013: 'o' in fusion prefix is NOT elided before vowels

Reference: IUPAC 2013 Blue Book, Section P-25 (Fused Ring Systems)
"""

import re
from typing import Dict, List, Optional, Set, Tuple


from ..data.fusion_components import (
    MONOCYCLIC_COMPONENTS,
    get_component_by_pattern,
    get_component_seniority,
)
from ..perception.rings import (
    is_aromatic_ring,
    is_heterocyclic,
)

# Letters for edge designation (a=first edge, b=second, etc.)
EDGE_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


# Mapping from ring names to their fusion prefix forms
# Fusion prefixes follow specific rules:
# - Drop final 'ene' -> 'o' (benzene -> benzo)
# - Drop final 'an' or 'ane' -> 'o' (furan -> furo)
# - Drop final 'ole' -> 'olo' (pyrrole -> pyrrolo)
# - Drop final 'ine' -> 'ino' (pyridine -> pyridino)
# etc.
FUSION_PREFIXES: Dict[str, str] = {
    # Aromatic carbocycles
    'benzene': 'benzo',
    'naphthalene': 'naphtho',
    'anthracene': 'anthra',
    'phenanthrene': 'phenanthro',
    'pyrene': 'pyreno',
    'fluorene': 'fluoreno',
    'cyclopentadiene': 'cyclopenta',
    'cyclopentene': 'cyclopenta',
    'cycloheptadiene': 'cyclohepta',
    'cycloheptene': 'cyclohepta',
    'cyclohexene': 'cyclohexa',

    # 5-membered heterocycles
    'furan': 'furo',
    'thiophene': 'thieno',
    'pyrrole': 'pyrrolo',
    'imidazole': 'imidazo',
    'pyrazole': 'pyrazolo',
    'oxazole': 'oxazolo',
    'isoxazole': 'isoxazolo',
    'thiazole': 'thiazolo',
    'isothiazole': 'isothiazolo',
    'selenazole': 'selenazolo',
    'isoselenazole': 'isoselenazolo',
    'selenophene': 'selenopheno',
    'triazole': 'triazolo',
    'tetrazole': 'tetrazolo',

    # 6-membered heterocycles
    'pyridine': 'pyrido',
    'pyrimidine': 'pyrimido',
    'pyrazine': 'pyrazino',
    'pyridazine': 'pyridazino',
    'triazine': 'triazino',
    # 6-membered O/S/Se/Te heterocycles. P-25.3.2.4 (BlueBookV2.md:11905): the
    # attached-component prefix ADDS 'o' when there is no final 'e' -- "pyrano
    # from pyran". :12030 gives "selenopyrano (from selenopyran, PIN)". These
    # were NOT in this table, so get_fusion_prefix's old general rule truncated
    # '-an' -> 'pyro'/'thiopyro' (OPSIN-unparseable). Cross-checked against
    # OPSIN's own fusionComponents token list (pyrano/thiopyrano/selenopyrano/
    # telluropyrano).
    'pyran': 'pyrano',
    'thiopyran': 'thiopyrano',
    'selenopyran': 'selenopyrano',
    'telluropyran': 'telluropyrano',

    # Fused heterocycles (for multi-fused systems)
    '1H-indole': 'indolo',
    'indole': 'indolo',
    'quinoline': 'quinolino',
    'isoquinoline': 'isoquinolino',
    '1H-benzimidazole': 'benzimidazo',
    'benzimidazole': 'benzimidazo',
    'benzofuran': 'benzofuro',
    '1-benzofuran': 'benzofuro',
    'benzothiophene': 'benzothieno',
    '1-benzothiophene': 'benzothieno',
    '9H-purine': 'purino',
    'purine': 'purino',
}


def get_fusion_edge(parent_ring: List[int], atom1: int, atom2: int) -> int:
    """
    Find which edge (bond) in the parent ring is shared.

    An edge is defined by two adjacent atoms in the ring. Edge numbering
    follows ring atom order: edge 0 is between atoms at positions 0-1,
    edge 1 is between positions 1-2, etc.

    Args:
        parent_ring: List of atom indices in the parent ring (ordered)
        atom1: First shared atom index
        atom2: Second shared atom index

    Returns:
        0-indexed edge number, or -1 if atoms are not adjacent in ring

    Examples:
        >>> get_fusion_edge([0, 1, 2, 3, 4, 5], 0, 1)
        0
        >>> get_fusion_edge([0, 1, 2, 3, 4, 5], 1, 2)
        1
        >>> get_fusion_edge([0, 1, 2, 3, 4, 5], 5, 0)  # wraparound
        5
    """
    ring_size = len(parent_ring)

    # Find positions of both atoms in the ring
    try:
        pos1 = parent_ring.index(atom1)
        pos2 = parent_ring.index(atom2)
    except ValueError:
        return -1  # Atoms not in ring

    # Check if atoms are adjacent (including wraparound)
    diff = abs(pos1 - pos2)

    if diff == 1:
        # Adjacent in sequence
        return min(pos1, pos2)
    elif diff == ring_size - 1:
        # Adjacent via wraparound (last-first)
        return ring_size - 1
    else:
        return -1  # Not adjacent


def get_fusion_letter(parent_ring: List[int], shared_atoms: Tuple[int, int]) -> str:
    """
    Get the fusion letter locant for the parent ring edge.

    Converts edge index to letter: edge 0 -> 'a', edge 1 -> 'b', etc.

    Args:
        parent_ring: List of atom indices in the parent ring
        shared_atoms: Tuple of (atom1, atom2) shared between rings

    Returns:
        Fusion letter ('a', 'b', etc.) or empty string if edge not found

    Examples:
        >>> get_fusion_letter([0, 1, 2, 3, 4, 5], (0, 1))
        'a'
        >>> get_fusion_letter([0, 1, 2, 3, 4, 5], (3, 4))
        'd'
    """
    edge_idx = get_fusion_edge(parent_ring, shared_atoms[0], shared_atoms[1])

    if edge_idx < 0 or edge_idx >= len(EDGE_LETTERS):
        return ''

    return EDGE_LETTERS[edge_idx]


def get_child_locants(
    child_ring: List[int],
    shared_atoms: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Find positions of shared atoms in child ring numbering.

    Returns positions as 1-indexed locants (IUPAC convention).

    Args:
        child_ring: List of atom indices in the child ring
        shared_atoms: Tuple of (atom1, atom2) shared between rings

    Returns:
        Tuple of (lower_locant, higher_locant), 1-indexed
        Returns (0, 0) if atoms not found

    Examples:
        >>> get_child_locants([0, 1, 2, 3, 4], (3, 4))
        (4, 5)
        >>> get_child_locants([6, 7, 8, 9, 10], (6, 7))
        (1, 2)
    """
    try:
        # Find 0-indexed positions
        pos1 = child_ring.index(shared_atoms[0])
        pos2 = child_ring.index(shared_atoms[1])
    except ValueError:
        return (0, 0)

    # Convert to 1-indexed locants
    loc1 = pos1 + 1
    loc2 = pos2 + 1

    # Return in sorted order (lower first)
    if loc1 <= loc2:
        return (loc1, loc2)
    else:
        return (loc2, loc1)


def generate_fusion_descriptor(
    parent_ring: List[int],
    child_ring: List[int],
    shared_atoms: Set[int],
    child_is_benzene: bool = False,
) -> str:
    """
    Generate the fusion descriptor [num,num-letter] format.

    Child locants are ordered according to IUPAC convention: the child
    position bonded to the lower-numbered parent edge atom is listed
    first, then the child position bonded to the higher-numbered parent
    edge atom. This can produce ascending [2,3-b] or descending [3,2-b]
    depending on the relative orientation of parent and child numbering.

    For benzene as child (all positions equivalent), child locants are
    omitted and only the edge letter is used: [b] instead of [1,2-b].

    Args:
        parent_ring: List of atom indices in the parent ring (IUPAC order)
        child_ring: List of atom indices in the child ring (IUPAC order)
        shared_atoms: Set of atom indices shared between rings
        child_is_benzene: If True, omit child locants (all equivalent)

    Returns:
        Fusion descriptor string like "[3,2-b]" or "[b]" for benzene child
        Returns empty string if descriptor cannot be generated

    Examples:
        >>> generate_fusion_descriptor([0,1,2,3,4,5], [6,7,8,9,10], {0,1})
        '[1,2-a]'
    """
    if len(shared_atoms) != 2:
        return ''

    shared_list = list(shared_atoms)
    atom_a, atom_b = shared_list[0], shared_list[1]

    # Find parent positions of shared atoms
    try:
        parent_pos_a = parent_ring.index(atom_a)
        parent_pos_b = parent_ring.index(atom_b)
    except ValueError:
        return ''

    # Determine which shared atom is at the lower parent position
    if parent_pos_a < parent_pos_b:
        # Check adjacency: normal or wraparound
        ring_size = len(parent_ring)
        if parent_pos_b - parent_pos_a == 1:
            parent_lower_atom = atom_a  # at lower pos
            parent_higher_atom = atom_b  # at higher pos
            edge_idx = parent_pos_a
        elif parent_pos_b - parent_pos_a == ring_size - 1:
            # Wraparound: pos_a is near start, pos_b is near end
            parent_lower_atom = atom_b  # at higher pos = lower edge (wraparound)
            parent_higher_atom = atom_a  # at lower pos = higher edge
            edge_idx = parent_pos_b
        else:
            return ''  # Not adjacent
    else:
        ring_size = len(parent_ring)
        if parent_pos_a - parent_pos_b == 1:
            parent_lower_atom = atom_b
            parent_higher_atom = atom_a
            edge_idx = parent_pos_b
        elif parent_pos_a - parent_pos_b == ring_size - 1:
            parent_lower_atom = atom_a
            parent_higher_atom = atom_b
            edge_idx = parent_pos_a
        else:
            return ''

    # Get edge letter
    if edge_idx >= len(EDGE_LETTERS):
        return ''
    fusion_letter = EDGE_LETTERS[edge_idx]

    # For benzene child, omit child locants
    if child_is_benzene:
        return f"[{fusion_letter}]"

    # Get child positions of shared atoms
    try:
        child_pos_lower = child_ring.index(parent_lower_atom) + 1  # 1-indexed
        child_pos_higher = child_ring.index(parent_higher_atom) + 1
    except ValueError:
        return ''

    # IUPAC convention: first child locant corresponds to parent lower edge,
    # second child locant corresponds to parent higher edge
    return f"[{child_pos_lower},{child_pos_higher}-{fusion_letter}]"


def get_fusion_prefix(ring_name: str) -> str:
    """
    Convert ring name to its fusion prefix form.

    Uses lookup table for known rings, then applies general rules
    for unknown rings.

    Args:
        ring_name: The name of the ring (e.g., 'benzene', 'furan')

    Returns:
        Fusion prefix form (e.g., 'benzo', 'furo')

    Examples:
        >>> get_fusion_prefix('benzene')
        'benzo'
        >>> get_fusion_prefix('naphthalene')
        'naphtho'
        >>> get_fusion_prefix('furan')
        'furo'
        >>> get_fusion_prefix('pyrrole')
        'pyrrolo'
    """
    # Normalise a leading indicated-hydrogen descriptor (e.g. '2H-pyran'): the
    # attached-component prefix drops it (the fused system's own indicated H is
    # computed separately). This closes the '2H-pyran' -> '2h-pyro' casing +
    # truncation defect if such a name ever reaches here.
    core = re.sub(r'^\d+[Hh]-', '', ring_name)

    # 1) Curated contracted retained forms (P-25.3.2.2): benzo, furo, thieno,
    #    pyrido, pyrano, ...
    if core in FUSION_PREFIXES:
        return FUSION_PREFIXES[core]

    # 2) The authoritative monocyclic registry is the source of truth for the
    #    attached-component prefix (pyran -> pyrano). This function historically
    #    kept its OWN lookup table that lacked 'pyran', so the general rule below
    #    truncated it to the OPSIN-unparseable 'pyro'. Consult the registry the
    #    rest of the fusion machinery already trusts (get_component_prefix).
    if core in MONOCYCLIC_COMPONENTS:
        return MONOCYCLIC_COMPONENTS[core]['prefix']

    # 3) General rules for names not covered above. P-25.3.2.4
    #    (BlueBookV2.md:11905): "The names of attached components are formed by
    #    replacing the last letter 'e' by 'o' ... (or by ADDING the letter 'o'
    #    when no final letter 'e' is present, i.e., pyrano from pyran)."
    name = core.lower()
    if name.endswith('ene'):
        return name[:-3] + 'o'
    if name.endswith('ole'):
        return name[:-1] + 'o'
    if name.endswith('ine'):
        return name[:-1] + 'o'
    if name.endswith('ane'):
        return name[:-3] + 'o'
    # NOTE: the historical '-an -> -o' truncation was DELETED here -- it produced
    # the OPSIN-unparseable 'pyro' from 'pyran', violating P-25.3.2.4 ("pyrano
    # from pyran", no final 'e' -> ADD 'o'). 'furan'/'pyran' are contracted /
    # regular forms now resolved by the table + registry above, so a bare '-an'
    # name correctly falls through to the "add 'o'" default below.
    if name.endswith('e'):
        return name[:-1] + 'o'
    return name + 'o'


def build_systematic_fusion_name(
    parent_name: str,
    child_name: str,
    descriptor: str
) -> str:
    """
    Assemble the systematic fusion name from components.

    Format: {fusion_prefix}{descriptor}{parent}

    IUPAC 2013 Note: The 'o' in fusion prefixes (benzo, naphtho) is
    NOT elided before vowels (unlike some older conventions).

    Args:
        parent_name: Name of the parent ring system
        child_name: Name of the child (fused) ring
        descriptor: Fusion descriptor (e.g., '[a]', '[2,1-b]')

    Returns:
        Complete systematic fusion name

    Examples:
        >>> build_systematic_fusion_name('anthracene', 'benzene', '[a]')
        'benzo[a]anthracene'
        >>> build_systematic_fusion_name('furan', 'naphthalene', '[2,1-b]')
        'naphtho[2,1-b]furan'
    """
    # Get fusion prefix for the child ring
    prefix = get_fusion_prefix(child_name)

    # Assemble: prefix + descriptor + parent
    # Note: IUPAC 2013 does NOT elide 'o' before vowels
    return f"{prefix}{descriptor}{parent_name}"


def identify_parent_and_child(
    mol,
    ring_a: Set[int],
    ring_b: Set[int]
) -> Tuple[str, str, List[int], List[int]]:
    """
    Determine which ring is parent and which is child for fusion naming.

    Parent selection follows IUPAC 2013 P-25.2.1 seniority:
    1. Heterocyclic ring is senior to carbocyclic (regardless of size)
    2. Among heterocyclic: nitrogen-containing > oxygen > sulfur
    3. Among same heteroatom type: larger ring > smaller ring
    4. Among same size/heteroatom: more heteroatoms > fewer

    The MORE SENIOR ring is the parent (base component, appears last in name).
    The LESS SENIOR ring is the child (becomes the fusion prefix).

    Args:
        mol: RDKit Mol object
        ring_a: Set of atom indices in first ring
        ring_b: Set of atom indices in second ring

    Returns:
        Tuple of (parent_name, child_name, parent_ring_list, child_ring_list)
        Returns ('', '', [], []) if rings cannot be identified
    """
    # Phase 149: route base-component decision through FR-2.3 cascade.
    # Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    # Source: 149-CONTEXT.md,.
    from .fused_ring_selection import select_base_component

    # Convert sets to lists for ordered operations
    ring_a_list = list(ring_a)
    ring_b_list = list(ring_b)

    # Identify both rings first
    name_a = _identify_ring_name(mol, ring_a_list)
    name_b = _identify_ring_name(mol, ring_b_list)

    if not name_a and not name_b:
        return ('', '', [], [])

    # Phase 149 primary: FR-2.3 cascade.
    # The returned base_atoms determines parent/child ordering.
    base_atoms, _others = select_base_component(mol, [set(ring_a), set(ring_b)])
    if base_atoms == set(ring_a):
        return (name_a, name_b, ring_a_list, ring_b_list)
    elif base_atoms == set(ring_b):
        return (name_b, name_a, ring_b_list, ring_a_list)

    # Total tie under FR-2.3 (a)-(f) — fall back to numeric seniority
    # (preserves get_component_seniority as last-resort tiebreaker;
    # reuse-not-rebuild discipline).
    seniority_a = get_component_seniority(name_a) if name_a else 999
    seniority_b = get_component_seniority(name_b) if name_b else 999
    if seniority_a <= seniority_b:
        return (name_a, name_b, ring_a_list, ring_b_list)
    return (name_b, name_a, ring_b_list, ring_a_list)


def _hetero_gap(mol, ring_atoms: List[int], hetero_indices: List[int]) -> int:
    """
    Compute the shortest gap between two heteroatoms walking around a ring.

    The gap is the number of non-heteroatom atoms on the shorter path
    between the two heteroatoms around the ring. This distinguishes
    positional isomers:
    - gap=0: adjacent (pyrazole/1,2-diazole, pyridazine/1,2-diazine)
    - gap=1: separated by 1 C (imidazole/1,3-diazole, pyrimidine/1,3-diazine)
    - gap=2: separated by 2 C (pyrazine/1,4-diazine)

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices defining the ring
        hetero_indices: List of exactly 2 atom indices (heteroatoms in the ring)

    Returns:
        Shortest gap count (0 = adjacent, 1 = one C between, etc.)
    """
    if len(hetero_indices) != 2:
        return -1

    ring_size = len(ring_atoms)
    h0, h1 = hetero_indices

    # Find positions of the heteroatoms in the ring
    try:
        pos0 = ring_atoms.index(h0)
        pos1 = ring_atoms.index(h1)
    except ValueError:
        return -1

    # Walk from pos0 to pos1 in both directions, count non-hetero atoms
    # Direction 1: pos0 -> pos0+1 -> ... -> pos1
    gap_cw = 0
    i = (pos0 + 1) % ring_size
    while i != pos1:
        gap_cw += 1
        i = (i + 1) % ring_size

    # Direction 2: pos0 -> pos0-1 -> ... -> pos1
    gap_ccw = 0
    i = (pos0 - 1) % ring_size
    while i != pos1:
        gap_ccw += 1
        i = (i - 1) % ring_size

    return min(gap_cw, gap_ccw)


def _identify_ring_name(mol, ring_atoms: List[int]) -> str:
    """
    Identify the IUPAC name of a monocyclic ring based on its structure.

    Distinguishes positional isomers by examining the relative positions
    of heteroatoms around the ring:
    - 5-membered 2N: imidazole (1,3) vs pyrazole (1,2)
    - 5-membered N+O: oxazole (1,3) vs isoxazole (1,2)
    - 5-membered N+S: thiazole (1,3) vs isothiazole (1,2)
    - 6-membered 2N: pyrimidine (1,3) vs pyridazine (1,2) vs pyrazine (1,4)

    Uses the MONOCYCLIC_COMPONENTS registry via get_component_by_pattern().

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices in the ring

    Returns:
        Ring name (e.g., 'pyrimidine', 'imidazole') or empty string if unknown
    """
    ring_size = len(ring_atoms)
    is_hetero = is_heterocyclic(mol, ring_atoms)
    is_aromatic = is_aromatic_ring(mol, ring_atoms)

    # Collect heteroatom info
    hetero_symbols = []
    hetero_indices = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            hetero_symbols.append(symbol)
            hetero_indices.append(idx)

    # Carbocyclic rings
    if not is_hetero:
        if ring_size == 6 and is_aromatic:
            return 'benzene'
        if ring_size == 5:
            return 'cyclopentadiene' if is_aromatic else 'cyclopentene'
        if ring_size == 6:
            return 'benzene' if is_aromatic else 'cyclohexene'
        if ring_size == 7:
            return 'cycloheptadiene' if is_aromatic else 'cycloheptene'
        return ''

    # Heterocyclic rings: use registry with gap disambiguation
    sorted_symbols = sorted(hetero_symbols)

    # Compute gap for 2-heteroatom rings
    gap = None
    if len(hetero_indices) == 2:
        gap = _hetero_gap(mol, ring_atoms, hetero_indices)

    # Try registry lookup
    result = get_component_by_pattern(ring_size, sorted_symbols, gap)
    if result:
        return result

    # Fallback for single-heteroatom rings not in registry
    if len(hetero_symbols) == 1:
        sym = hetero_symbols[0]
        if ring_size == 5:
            if sym == 'O':
                return 'furan'
            elif sym == 'S':
                return 'thiophene'
            elif sym == 'N':
                return 'pyrrole'
        elif ring_size == 6:
            if sym == 'N':
                return 'pyridine'
            elif sym == 'O':
                return 'pyran'

    return ''


def _get_iupac_ring_order(mol, ring_atoms: List[int]) -> List[int]:
    """
    Reorder ring atoms to match IUPAC numbering convention.

    For heterocyclic rings:
    - IUPAC position 1 is the highest-priority heteroatom
      (O > S > N per Hantzsch-Widman convention)
    - Number in the direction that gives lowest locants to remaining heteroatoms
    - Walk the ring using molecular adjacency (bonds)

    For carbocyclic rings:
    - Return atoms in current order (numbering determined by fusion context)

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices in the ring

    Returns:
        List of atom indices reordered to match IUPAC numbering
    """
    ring_set = set(ring_atoms)

    # Collect heteroatoms with their priority
    # Priority: O (highest) > S > N (among common heteroatoms)
    # Hantzsch-Widman O > S > Se > Te > N (P-25.3.3; Se/Te inserted for the
    # Wave-2 selenazolo class -- same order as _PCF_HET_NUM_SENIORITY).
    HETERO_PRIORITY = {'O': 0, 'S': 1, 'Se': 2, 'Te': 3, 'N': 4}
    hetero_info = []  # (priority, atom_idx)
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym in HETERO_PRIORITY:
            hetero_info.append((HETERO_PRIORITY[sym], idx))

    if not hetero_info:
        # Carbocyclic: return as-is
        return list(ring_atoms)

    # Start atom: highest-priority heteroatom (lowest priority value)
    hetero_info.sort()
    start_atom = hetero_info[0][1]

    # Build adjacency within ring
    ring_neighbors = {}
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        nbrs = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set and nbr_idx != idx:
                nbrs.append(nbr_idx)
        ring_neighbors[idx] = nbrs

    # Walk ring in both directions from start_atom
    def walk_ring(start, first_step):
        """Walk the ring from start via first_step, return ordered atom list."""
        result = [start]
        prev = start
        curr = first_step
        while curr != start:
            result.append(curr)
            nbrs = ring_neighbors[curr]
            # Go to the neighbor that isn't where we came from
            next_atoms = [n for n in nbrs if n != prev]
            if not next_atoms:
                break
            prev = curr
            curr = next_atoms[0]
        return result

    # Get neighbors of start atom in ring
    start_nbrs = ring_neighbors.get(start_atom, [])
    if len(start_nbrs) < 2:
        return list(ring_atoms)

    # Walk both directions
    order_a = walk_ring(start_atom, start_nbrs[0])
    order_b = walk_ring(start_atom, start_nbrs[1])

    # Pick direction giving lowest locants to remaining heteroatoms
    remaining_hetero_indices = [idx for _, idx in hetero_info[1:]]

    if not remaining_hetero_indices:
        # Only one heteroatom, either direction is fine
        return order_a

    def hetero_locants(order):
        """Get IUPAC positions (1-indexed) of remaining heteroatoms."""
        locs = []
        for h_idx in remaining_hetero_indices:
            try:
                pos = order.index(h_idx) + 1  # 1-indexed
                locs.append(pos)
            except ValueError:
                locs.append(999)
        return sorted(locs)

    locs_a = hetero_locants(order_a)
    locs_b = hetero_locants(order_b)

    # First-point-of-difference comparison
    for la, lb in zip(locs_a, locs_b):
        if la < lb:
            return order_a
        elif lb < la:
            return order_b

    # Tie: use direction A by default
    return order_a


def identify_fusion_edges(
    mol,
    parent_atoms: List[int],
    child_atoms: List[int]
) -> List[Tuple[int, int]]:
    """
    Find which edges of parent ring are involved in fusion with child ring.

    Identifies all edges (bonds) shared between the parent and child rings.
    An edge is defined by two adjacent atoms that appear in both rings.

    Args:
        mol: RDKit Mol object
        parent_atoms: List of atom indices in the parent ring (ordered)
        child_atoms: List of atom indices in the child ring

    Returns:
        List of (edge_start, edge_end) atom index pairs for shared edges
        Empty list if no shared edges found

    Examples:
        >>> # For benzene fused to anthracene at edge 'a' (atoms 0-1)
        >>> identify_fusion_edges(mol, [0,1,2,3,4,5,6,7,8,9], [0,1,10,11,12,13])
        [(0, 1)]
    """
    shared_edges = []
    child_set = set(child_atoms)
    parent_size = len(parent_atoms)

    # Check each edge of the parent ring
    for i in range(parent_size):
        atom1 = parent_atoms[i]
        atom2 = parent_atoms[(i + 1) % parent_size]

        # If both atoms are in the child ring, this is a shared edge
        if atom1 in child_set and atom2 in child_set:
            # Verify they are bonded
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                shared_edges.append((atom1, atom2))

    return shared_edges


def edge_position_to_letter(parent_ring: List[int], edge: Tuple[int, int]) -> str:
    """
    Convert an edge position in a parent ring to its IUPAC letter designator.

    Edge 'a' is between atoms at positions 0-1 (IUPAC atoms 1-2),
    edge 'b' is between positions 1-2 (IUPAC atoms 2-3), etc.

    Args:
        parent_ring: List of atom indices in the parent ring (ordered)
        edge: Tuple of (atom1, atom2) defining the edge

    Returns:
        Letter designator ('a', 'b', 'c', ...) or empty string if not found

    Examples:
        >>> edge_position_to_letter([0,1,2,3,4,5], (0, 1))
        'a'
        >>> edge_position_to_letter([0,1,2,3,4,5], (2, 3))
        'c'
    """
    edge_idx = get_fusion_edge(parent_ring, edge[0], edge[1])
    if edge_idx < 0 or edge_idx >= len(EDGE_LETTERS):
        return ''
    return EDGE_LETTERS[edge_idx]


def handle_duplicate_edge_fusion(edge_letters: List[str]) -> List[str]:
    """
    Apply primed notation when same edge letter appears multiple times.

    IUPAC uses primed notation (a', a'', b', etc.) when the same edge
    letter is used for multiple fusion points. Letters are sorted
    alphabetically with unprimed before primed variants.

    Args:
        edge_letters: List of edge letters (may have duplicates)

    Returns:
        List of letters with primes applied where needed, sorted canonically
        Order: a, b, c, ... a', b', ... a'', b'', ...

    Examples:
        >>> handle_duplicate_edge_fusion(['a', 'c'])
        ['a', 'c']
        >>> handle_duplicate_edge_fusion(['a', 'a'])
        ['a', "a'"]
        >>> handle_duplicate_edge_fusion(['a', 'b', 'a'])
        ['a', 'b', "a'"]
        >>> handle_duplicate_edge_fusion(['a', 'a', 'a'])
        ['a', "a'", "a''"]
    """
    if not edge_letters:
        return []

    # Count occurrences of each letter
    from collections import Counter
    letter_counts = Counter(edge_letters)

    # Track how many times we've used each letter
    letter_usage = {letter: 0 for letter in letter_counts}

    result = []
    for letter in edge_letters:
        usage = letter_usage[letter]
        if usage == 0:
            result.append(letter)
        else:
            # Add primes for subsequent uses
            primes = "'" * usage
            result.append(f"{letter}{primes}")
        letter_usage[letter] += 1

    # Sort canonically: by base letter, then by number of primes
    def sort_key(s):
        base = s.rstrip("'")
        prime_count = len(s) - len(base)
        return (base, prime_count)

    return sorted(result, key=sort_key)


def format_complex_fusion(
    child_locants: Optional[Tuple[int, int]],
    parent_letter: Optional[str],
    multi_component: Optional[List[str]] = None,
    multi_edge: Optional[List[Tuple[Tuple[int, int], str]]] = None
) -> str:
    """
    Format fusion descriptor for various complexity levels.

    Handles three types of fusion descriptors:
    1. Standard: [2,3-b] - single fusion with child locants and parent letter
    2. Multi-component: [a,c] - multiple same-type rings fused to parent
    3. Multi-edge: [1,2-a:4,5-b'] - complex fusions with multiple edges

    Args:
        child_locants: Tuple of (loc1, loc2) child ring locants, or None
        parent_letter: Parent edge letter ('a', 'b', etc.), or None
        multi_component: List of edge letters for multi-component fusion
        multi_edge: List of ((loc1, loc2), letter) for multi-edge fusion

    Returns:
        Formatted fusion descriptor string

    Examples:
        >>> format_complex_fusion((2, 3), 'b')
        '[2,3-b]'
        >>> format_complex_fusion(None, None, multi_component=['a', 'c'])
        '[a,c]'
        >>> format_complex_fusion(None, None, multi_edge=[((1, 2), 'a'), ((4, 5), "b'")])
        "[1,2-a:4,5-b']"
    """
    # Multi-component fusion: [a,c] format for dibenzo, dinaphtho, etc.
    if multi_component is not None:
        # Apply primed notation if needed and sort
        processed = handle_duplicate_edge_fusion(multi_component)
        return f"[{','.join(processed)}]"

    # Multi-edge fusion: [1,2-a:4,5-b'] format
    if multi_edge is not None:
        parts = []
        for (loc1, loc2), letter in multi_edge:
            parts.append(f"{loc1},{loc2}-{letter}")
        return f"[{':'.join(parts)}]"

    # Standard fusion: [2,3-b] format
    if child_locants is not None and parent_letter is not None:
        loc1, loc2 = child_locants
        return f"[{loc1},{loc2}-{parent_letter}]"

    return ''


def generate_multi_fusion_descriptor(
    parent_ring_name: str,
    fused_components: List[Tuple[str, List[int], Set[int]]]
) -> str:
    """
    Generate fusion descriptor for multi-component fusions.

    For compounds like dibenzo[a,c]anthracene where multiple rings of
    the same type are fused to a parent ring.

    Args:
        parent_ring_name: Name of the parent ring (e.g., 'anthracene')
        fused_components: List of (child_name, parent_ring_atoms, shared_atoms) tuples
            where each tuple describes one fused component

    Returns:
        Multi-component descriptor like '[a,c]' for dibenzo
        Returns empty string if descriptor cannot be generated

    Examples:
        >>> # dibenzo[a,c]anthracene: two benzene rings at edges a and c
        >>> generate_multi_fusion_descriptor('anthracene', [
        ...     ('benzene', [0,1,2,3,4,5,6,7,8,9,10,11,12,13], {0, 1}),
        ...     ('benzene', [0,1,2,3,4,5,6,7,8,9,10,11,12,13], {4, 5})
        ... ])
        '[a,c]'
    """
    if not fused_components:
        return ''

    edge_letters = []

    for child_name, parent_ring, shared_atoms in fused_components:
        if len(shared_atoms) != 2:
            continue

        # Get the edge letter for this fusion
        atoms_tuple = tuple(sorted(shared_atoms))
        letter = get_fusion_letter(parent_ring, atoms_tuple)
        if letter:
            edge_letters.append(letter)

    if not edge_letters:
        return ''

    return format_complex_fusion(None, None, multi_component=edge_letters)


def build_multi_component_name(
    parent_name: str,
    child_name: str,
    count: int,
    descriptor: str
) -> str:
    """
    Build name for multi-component fusion (dibenzo, dinaphtho, etc.).

    Args:
        parent_name: Name of parent ring (e.g., 'anthracene')
        child_name: Name of fused ring type (e.g., 'benzene')
        count: Number of fused rings of this type (2 for di-, 3 for tri-)
        descriptor: Fusion descriptor (e.g., '[a,c]')

    Returns:
        Complete multi-component name like 'dibenzo[a,c]anthracene'

    Examples:
        >>> build_multi_component_name('anthracene', 'benzene', 2, '[a,c]')
        'dibenzo[a,c]anthracene'
        >>> build_multi_component_name('anthracene', 'naphthalene', 2, '[a,h]')
        'dinaphtho[a,h]anthracene'
    """
    # Multiplicative prefixes for ring count
    MULTIPLIERS = {
        2: 'di',
        3: 'tri',
        4: 'tetra',
        5: 'penta',
        6: 'hexa',
    }

    prefix = get_fusion_prefix(child_name)
    multiplier = MULTIPLIERS.get(count, '')

    return f"{multiplier}{prefix}{descriptor}{parent_name}"


def _get_iupac_ring_order_for_fusion(
    mol,
    ring_atoms: List[int],
    shared_atoms: Set[int],
    is_child: bool = False,
) -> List[int]:
    """
    Get IUPAC-numbered ring order suitable for fusion descriptor generation.

    For heterocyclic rings, IUPAC numbering starts at the highest-priority
    heteroatom. The direction is chosen to give lowest locants to:
    1. Remaining heteroatoms (standard IUPAC rule)
    2. If single heteroatom (no remaining heteroatoms to tiebreak):
       for child rings, pick direction giving lowest fusion bond locants;
       for parent rings, pick direction giving lowest fusion edge letter.

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices in the ring
        shared_atoms: Set of atom indices shared with the other ring
        is_child: True if this ring is the child (attached component)

    Returns:
        List of atom indices in IUPAC numbering order
    """
    ring_set = set(ring_atoms)

    # Collect heteroatoms with priority
    # Hantzsch-Widman O > S > Se > Te > N (P-25.3.3; Se/Te inserted for the
    # Wave-2 selenazolo class -- same order as _PCF_HET_NUM_SENIORITY).
    HETERO_PRIORITY = {'O': 0, 'S': 1, 'Se': 2, 'Te': 3, 'N': 4}
    hetero_info = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym in HETERO_PRIORITY:
            hetero_info.append((HETERO_PRIORITY[sym], idx))

    if not hetero_info:
        # Carbocyclic ring: for fusion, pick direction giving lowest
        # locants to shared atoms
        return _orient_carbocyclic_for_fusion(mol, ring_atoms, shared_atoms)

    hetero_info.sort()
    start_atom = hetero_info[0][1]

    # Build adjacency within ring
    ring_neighbors = {}
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        nbrs = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set and nbr_idx != idx:
                nbrs.append(nbr_idx)
        ring_neighbors[idx] = nbrs

    def walk_ring(start, first_step):
        result = [start]
        prev = start
        curr = first_step
        while curr != start:
            result.append(curr)
            nbrs = ring_neighbors[curr]
            next_atoms = [n for n in nbrs if n != prev]
            if not next_atoms:
                break
            prev = curr
            curr = next_atoms[0]
        return result

    start_nbrs = ring_neighbors.get(start_atom, [])
    if len(start_nbrs) < 2:
        return list(ring_atoms)

    order_a = walk_ring(start_atom, start_nbrs[0])
    order_b = walk_ring(start_atom, start_nbrs[1])

    # First tiebreaker: remaining heteroatom locants
    remaining_hetero_indices = [idx for _, idx in hetero_info[1:]]

    if remaining_hetero_indices:
        def hetero_locants(order):
            locs = []
            for h_idx in remaining_hetero_indices:
                try:
                    pos = order.index(h_idx) + 1
                    locs.append(pos)
                except ValueError:
                    locs.append(999)
            return sorted(locs)

        locs_a = hetero_locants(order_a)
        locs_b = hetero_locants(order_b)

        for la, lb in zip(locs_a, locs_b):
            if la < lb:
                return order_a
            elif lb < la:
                return order_b

    # No remaining heteroatoms or tied: use fusion bond locants as tiebreaker
    # Pick direction giving lowest locants to shared atoms
    def shared_locants(order):
        locs = []
        for s_idx in shared_atoms:
            try:
                pos = order.index(s_idx) + 1
                locs.append(pos)
            except ValueError:
                locs.append(999)
        return sorted(locs)

    slocs_a = shared_locants(order_a)
    slocs_b = shared_locants(order_b)

    for sa, sb in zip(slocs_a, slocs_b):
        if sa < sb:
            return order_a
        elif sb < sa:
            return order_b

    return order_a


def _orient_carbocyclic_for_fusion(
    mol,
    ring_atoms: List[int],
    shared_atoms: Set[int],
) -> List[int]:
    """
    Orient a carbocyclic ring for fusion descriptor generation.

    Since all atoms are equivalent in a symmetric carbocyclic ring,
    choose numbering that starts adjacent to a shared atom and gives
    lowest locants to the shared atoms.

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices
        shared_atoms: Set of shared atom indices

    Returns:
        Reordered list of atom indices
    """
    ring_set = set(ring_atoms)

    # Build adjacency within ring
    ring_neighbors = {}
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        nbrs = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set and nbr_idx != idx:
                nbrs.append(nbr_idx)
        ring_neighbors[idx] = nbrs

    def walk_ring(start, first_step):
        result = [start]
        prev = start
        curr = first_step
        while curr != start:
            result.append(curr)
            nbrs = ring_neighbors[curr]
            next_atoms = [n for n in nbrs if n != prev]
            if not next_atoms:
                break
            prev = curr
            curr = next_atoms[0]
        return result

    # Try every atom as starting point and both directions
    best_order = None
    best_shared_locs = None

    for start_atom in ring_atoms:
        start_nbrs = ring_neighbors.get(start_atom, [])
        if len(start_nbrs) < 2:
            continue
        for first_nbr in start_nbrs:
            order = walk_ring(start_atom, first_nbr)
            locs = sorted(order.index(s) + 1 for s in shared_atoms if s in order)
            if best_shared_locs is None or locs < best_shared_locs:
                best_shared_locs = locs
                best_order = order

    return best_order if best_order else list(ring_atoms)


def generate_systematic_name_for_fused_pair(
    mol,
    ring1: List[int],
    ring2: List[int],
    shared_atoms: Set[int]
) -> Optional[str]:
    """
    Generate systematic fusion name for a pair of fused rings.

    This is the main entry point for generating fusion names when
    no retained name exists. Uses IUPAC-numbered ring ordering for
    correct edge letters and child locants.

    Args:
        mol: RDKit Mol object
        ring1: List of atom indices in first ring
        ring2: List of atom indices in second ring
        shared_atoms: Set of atom indices shared between rings

    Returns:
        Systematic fusion name, or None if name cannot be generated

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')  # anthracene
        >>> ri = mol.GetRingInfo()
        >>> # Would generate 'benzo[a]naphthalene' for benzene fused to naphthalene
    """
    if len(shared_atoms) != 2:
        return None

    # Identify parent and child
    parent_name, child_name, parent_ring, child_ring = identify_parent_and_child(
        mol, set(ring1), set(ring2)
    )

    if not parent_name or not child_name:
        return None

    # Get IUPAC-ordered rings for correct descriptor generation
    parent_iupac = _get_iupac_ring_order_for_fusion(
        mol, parent_ring, shared_atoms, is_child=False
    )
    child_iupac = _get_iupac_ring_order_for_fusion(
        mol, child_ring, shared_atoms, is_child=True
    )

    # Generate fusion descriptor using IUPAC-ordered rings
    # For carbocyclic child (all positions equivalent), omit child locants
    # benzene -> [b], cyclopentadiene -> [b], cycloheptadiene -> [b]
    CARBOCYCLIC_CHILDREN = {'benzene', 'cyclopentadiene', 'cyclopentene',
                            'cycloheptadiene', 'cycloheptene', 'cyclohexene'}
    child_is_carbo = child_name in CARBOCYCLIC_CHILDREN
    descriptor = generate_fusion_descriptor(
        parent_iupac, child_iupac, shared_atoms, child_is_benzene=child_is_carbo
    )
    if not descriptor:
        return None

    # Build the systematic name
    return build_systematic_fusion_name(parent_name, child_name, descriptor)


def generate_multi_fusion_name(
    mol,
    parent_ring: List[int],
    parent_name: str,
    fused_rings: List[Tuple[List[int], Set[int]]]
) -> Optional[str]:
    """
    Generate systematic name for multiple rings fused to a parent.

    Handles complex fusion scenarios like dibenzo[a,c]anthracene where
    multiple rings of the same type are fused at different edges.

    Args:
        mol: RDKit Mol object
        parent_ring: List of atom indices in the parent ring
        parent_name: Name of the parent ring (e.g., 'anthracene')
        fused_rings: List of (child_ring_atoms, shared_atoms) for each fusion

    Returns:
        Complete systematic fusion name, or None if cannot be generated

    Examples:
        >>> # For dibenzo[a,c]anthracene
        >>> generate_multi_fusion_name(mol, anthracene_atoms, 'anthracene',
        ...     [(benzene1_atoms, {0,1}), (benzene2_atoms, {4,5})])
        'dibenzo[a,c]anthracene'
    """
    if not fused_rings:
        return None

    # Group fused rings by type
    from collections import defaultdict
    rings_by_type: Dict[str, List[Tuple[List[int], Set[int]]]] = defaultdict(list)

    for child_ring, shared in fused_rings:
        child_name = _identify_ring_name(mol, child_ring)
        if child_name:
            rings_by_type[child_name].append((child_ring, shared))

    # For now, handle single child type (e.g., all benzene)
    if len(rings_by_type) == 1:
        child_name = list(rings_by_type.keys())[0]
        fusions = rings_by_type[child_name]

        if len(fusions) == 1:
            # Single fusion - use standard naming
            child_ring, shared = fusions[0]
            return generate_systematic_name_for_fused_pair(
                mol, parent_ring, child_ring, shared
            )
        else:
            # Multi-component fusion
            components = []
            for child_ring, shared in fusions:
                components.append((child_name, parent_ring, shared))

            descriptor = generate_multi_fusion_descriptor(parent_name, components)
            if descriptor:
                return build_multi_component_name(
                    parent_name, child_name, len(fusions), descriptor
                )

    return None
