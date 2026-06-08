"""
Ring-as-substituent naming according to IUPAC 2013 (Blue Book).

Implements IUPAC P-61.5: Standard substituent names for rings when they
become substituents on a chain parent structure.

Examples:
- benzene -> phenyl (4-phenylbutanoic acid)
- cyclohexane -> cyclohexyl (4-cyclohexylbutanoic acid)
- naphthalene -> naphthyl (position-specific: 1-naphthyl, 2-naphthyl)
- pyridine -> pyridyl (position-specific: 2-pyridyl, 3-pyridyl, 4-pyridyl)
"""

from typing import Dict, List, Optional, Set, Tuple

# Chain length prefixes - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix


# IUPAC P-61.5: Standard substituent names for rings
# Maps ring system name to substituent prefix name
RING_SUBSTITUENT_NAMES: Dict[str, str] = {
    # Carbocyclic aromatic
    'benzene': 'phenyl',
    'naphthalene': 'naphthyl',  # Position-specific: 1-naphthyl, 2-naphthyl
    'anthracene': 'anthryl',
    'phenanthrene': 'phenanthryl',
    'pyrene': 'pyrenyl',
    'fluorene': 'fluorenyl',
    'acenaphthylene': 'acenaphthylenyl',
    'chrysene': 'chrysenyl',
    'fluoranthene': 'fluoranthenyl',

    # Carbocyclic saturated (cycloalkanes)
    'cyclopropane': 'cyclopropyl',
    'cyclobutane': 'cyclobutyl',
    'cyclopentane': 'cyclopentyl',
    'cyclohexane': 'cyclohexyl',
    'cycloheptane': 'cycloheptyl',
    'cyclooctane': 'cyclooctyl',

    # Heterocyclic aromatic
    'pyridine': 'pyridyl',  # Position-specific: 2-pyridyl, 3-pyridyl, 4-pyridyl
    'furan': 'furyl',
    'thiophene': 'thienyl',
    'pyrrole': 'pyrrolyl',
    'imidazole': 'imidazolyl',
    'pyrimidine': 'pyrimidinyl',
    'pyrazine': 'pyrazinyl',
    'pyridazine': 'pyridazinyl',

    # Heterocyclic saturated (6-membered)
    'oxane': 'oxanyl',
    'piperidine': 'piperidinyl',
    'morpholine': 'morpholinyl',
    'piperazine': 'piperazinyl',

    # Heterocyclic saturated (5-membered)
    'oxolane': 'oxolanyl',
    'pyrrolidine': 'pyrrolidinyl',

    # Heterocyclic saturated (4-membered)
    'oxetane': 'oxetanyl',
    'azetidine': 'azetidinyl',

    # Heterocyclic saturated (3-membered)
    'oxirane': 'oxiranyl',
    'aziridine': 'aziridinyl',

    # Fused heterocyclic (Phase 139 ARCH-02)
    'indole': 'indolyl',
    'quinoline': 'quinolinyl',
    'isoquinoline': 'isoquinolinyl',
    'benzofuran': 'benzofuranyl',
    'benzothiophene': 'benzothienyl',
    'benzimidazole': 'benzimidazolyl',
    'purine': 'purinyl',
    'carbazole': 'carbazolyl',
}

# Rings that need position-specific names based on attachment point
# Maps ring name -> {ring_position: substituent_name}
POSITION_SPECIFIC_RINGS: Dict[str, Dict[int, str]] = {
    'naphthalene': {
        1: '1-naphthyl',
        2: '2-naphthyl',
    },
    'pyridine': {
        2: '2-pyridyl',
        3: '3-pyridyl',
        4: '4-pyridyl',
    },
    'quinoline': {
        2: '2-quinolinyl',
        3: '3-quinolinyl',
        4: '4-quinolinyl',
        5: '5-quinolinyl',
        6: '6-quinolinyl',
        7: '7-quinolinyl',
        8: '8-quinolinyl',
    },
    'isoquinoline': {
        1: '1-isoquinolinyl',
        3: '3-isoquinolinyl',
        4: '4-isoquinolinyl',
        5: '5-isoquinolinyl',
        6: '6-isoquinolinyl',
        7: '7-isoquinolinyl',
        8: '8-isoquinolinyl',
    },
}


def identify_ring_system(mol, ring_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Identify ring system name from molecular structure.

    Examines the ring atoms to determine what type of ring system it is
    based on size, aromaticity, and heteroatom composition.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        Ring system name string (e.g., 'benzene', 'cyclohexane', 'pyridine'),
        or None if the ring cannot be identified
    """
    ring_size = len(ring_atoms)
    ring_set = set(ring_atoms)

    # Check aromaticity of all ring atoms
    is_aromatic = all(
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring_atoms
    )

    # Get heteroatom list (symbols of non-carbon atoms)
    heteroatoms = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append(symbol)

    # Sort for consistent comparison
    heteroatoms.sort()

    # === 6-membered rings ===
    if ring_size == 6:
        if is_aromatic:
            if not heteroatoms:
                return 'benzene'
            elif heteroatoms == ['N']:
                return 'pyridine'
            elif heteroatoms == ['N', 'N']:
                # Distinguish pyridazine (1,2), pyrimidine (1,3), pyrazine (1,4)
                # by checking the topological relationship between N atoms
                n_indices = [idx for idx in ring_atoms
                             if mol.GetAtomWithIdx(idx).GetSymbol() == 'N']
                if len(n_indices) == 2:
                    n1, n2 = n_indices
                    # Check if N atoms are directly bonded (pyridazine = 1,2-diazine)
                    n1_nbrs = {nbr.GetIdx() for nbr in mol.GetAtomWithIdx(n1).GetNeighbors()}
                    if n2 in n1_nbrs:
                        return 'pyridazine'
                    # Check shortest path between N atoms in the ring
                    # Pyrimidine (1,3): 1 C between N's on shorter side
                    # Pyrazine (1,4): 2 C between N's on both sides (symmetric)
                    ring_list = list(ring_atoms)
                    pos1 = ring_list.index(n1)
                    pos2 = ring_list.index(n2)
                    sep = abs(pos1 - pos2)
                    min_sep = min(sep, ring_size - sep)
                    if min_sep == 3:  # para = pyrazine
                        return 'pyrazine'
                    else:  # min_sep == 2 = meta = pyrimidine
                        return 'pyrimidine'
                return 'pyrimidine'  # fallback
        else:
            # Saturated 6-membered
            if not heteroatoms:
                return 'cyclohexane'
            elif heteroatoms == ['O']:
                return 'oxane'
            elif heteroatoms == ['N']:
                return 'piperidine'
            elif heteroatoms == ['N', 'O']:
                return 'morpholine'
            elif heteroatoms == ['N', 'N']:
                return 'piperazine'

    # === 5-membered rings ===
    elif ring_size == 5:
        if is_aromatic:
            if not heteroatoms:
                # Aromatic 5-membered all carbon = cyclopentadienyl anion
                # Usually not encountered, but handle gracefully
                return 'cyclopentadiene'
            elif heteroatoms == ['O']:
                return 'furan'
            elif heteroatoms == ['S']:
                return 'thiophene'
            elif heteroatoms == ['N']:
                return 'pyrrole'
            elif heteroatoms == ['N', 'N']:
                return 'imidazole'
        else:
            # Saturated 5-membered
            if not heteroatoms:
                return 'cyclopentane'
            elif heteroatoms == ['O']:
                return 'oxolane'
            elif heteroatoms == ['N']:
                return 'pyrrolidine'

    # === 3-membered rings ===
    elif ring_size == 3:
        if not heteroatoms:
            return 'cyclopropane'
        elif heteroatoms == ['O']:
            return 'oxirane'
        elif heteroatoms == ['N']:
            return 'aziridine'

    # === 4-membered rings ===
    elif ring_size == 4:
        if not heteroatoms:
            return 'cyclobutane'
        elif heteroatoms == ['O']:
            return 'oxetane'
        elif heteroatoms == ['N']:
            return 'azetidine'

    # === 7-membered rings ===
    elif ring_size == 7:
        if not heteroatoms:
            if is_aromatic:
                return 'cycloheptatriene'
            else:
                return 'cycloheptane'

    # === 8-membered and larger ===
    elif ring_size == 8 and not heteroatoms:
        return 'cyclooctane'

    # Fallback: generic cycloalkane for all-carbon saturated rings
    if not heteroatoms and not is_aromatic:
        prefix = _get_chain_prefix(ring_size)
        return f'cyclo{prefix}ane'

    return None


# IUPAC P-31.1.4.3.4 / P-29.3.5: PIN substituent stems for monocyclic
# heteroarenes (ring name with the trailing 'e' removed). Only these rings are
# claimed by pin_heteroaryl_substituent_name(); anything else guards to None.
_PIN_HETEROARYL_STEMS: Dict[str, str] = {
    'pyridine': 'pyridin',
    'pyrimidine': 'pyrimidin',
    'pyrazine': 'pyrazin',
    'pyridazine': 'pyridazin',
    'furan': 'furan',
    'thiophene': 'thiophen',
    'pyrrole': 'pyrrol',
    'imidazole': 'imidazol',
}

# Element seniority for assigning low locants to heteroatoms (IUPAC Table 28:
# F > Cl > Br > I > O > S > Se > Te > N > P > ...). Lower value = senior.
_HETEROATOM_SENIORITY: Dict[str, int] = {
    'F': 0, 'Cl': 1, 'Br': 2, 'I': 3,
    'O': 4, 'S': 5, 'Se': 6, 'Te': 7,
    'N': 8, 'P': 9, 'As': 10, 'Sb': 11, 'Bi': 12,
    'Si': 13, 'Ge': 14, 'Sn': 15, 'Pb': 16, 'B': 17,
}


def pin_heteroaryl_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
) -> Optional[str]:
    """PIN substituent name for a monocyclic heteroaromatic ring substituent.

    Computes free-valence-aware numbering per IUPAC 2013 P-31.1.4.3.4: when a
    ring is detached as a substituent it is renumbered so low locants go, in
    order of decreasing priority, to (1) the heteroatoms as a set, (2) the
    heteroatoms in element-seniority order, (3) the indicated hydrogen, and
    (4) the free valence (point of attachment). The result is
    ``{nH-}{stem}-{locant}-yl``.

    Example: ``c1cnc[nH]1`` attached at the carbon next to the NH numbers as
    ``1H-imidazol-5-yl`` — the indicated H at locant 1 outranks the lower
    free-valence locant that the alternative ``3H-imidazol-4-yl`` numbering
    would give.

    Deliberately *guarded*: returns ``None`` (so the caller keeps its existing
    locant-less form, guaranteeing zero regression) whenever the PIN locant is
    not provably correct —
      * the ring is not one of the supported monocyclic heteroarenes;
      * it is a pyrazole (which identify_ring_system() reports as 'imidazole');
      * the ring carries any substituent other than the single attachment;
      * the ring is not a simple aromatic monocycle;
      * ``attachment_atom`` is not a ring atom.

    Args:
        mol: RDKit Mol of the whole molecule.
        ring_atoms: Atom indices forming the candidate ring.
        attachment_atom: Ring atom index bearing the free valence (bonded to
            the parent structure).

    Returns:
        The PIN substituent name, or None when not provably PIN-correct.
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)
    ring_set = set(ring_list)

    # --- Guard: attachment must be a ring atom -------------------------------
    if attachment_atom not in ring_set:
        return None

    # --- Guard: supported, correctly-identified heteroarene ------------------
    ring_name = identify_ring_system(mol, tuple(ring_list))
    stem = _PIN_HETEROARYL_STEMS.get(ring_name)
    if stem is None:
        return None

    # --- Guard: aromatic simple monocycle ------------------------------------
    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list):
        return None
    ring_adj: Dict[int, List[int]] = {}
    for idx in ring_list:
        nbrs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nbr.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None  # fusion atom / spiro / not a simple ring
        ring_adj[idx] = nbrs

    het_atoms = [i for i in ring_list
                 if mol.GetAtomWithIdx(i).GetSymbol() != 'C']

    # --- Guard: imidazole vs pyrazole ----------------------------------------
    # identify_ring_system() reports BOTH 5-membered N,N arenes as 'imidazole'.
    # Genuine imidazole has its two ring N's NON-adjacent (1,3); pyrazole has
    # them adjacent (1,2). Refuse the adjacent-N case so we never emit an
    # imidazol-*-yl name for a pyrazole.
    if ring_name == 'imidazole':
        n_idx = [i for i in het_atoms
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idx) != 2 or mol.GetBondBetweenAtoms(n_idx[0], n_idx[1]) is not None:
            return None

    # --- Guard: no ring substituent other than the single attachment ---------
    for idx in ring_list:
        heavy_exo = sum(
            1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        )
        if idx == attachment_atom:
            if heavy_exo < 1:
                return None  # no parent bond at the claimed attachment
        elif heavy_exo:
            return None  # an extra ring substituent we do not number

    # --- Indicated-hydrogen atom (the ring NH, if any) -----------------------
    indicated_h_atoms = [
        i for i in het_atoms
        if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1
    ]
    if len(indicated_h_atoms) > 1:
        return None  # ambiguous indicated H; do not guess
    indicated_h_atom = indicated_h_atoms[0] if indicated_h_atoms else None

    # --- Build the cyclic atom order -----------------------------------------
    order = [ring_list[0], ring_adj[ring_list[0]][0]]
    while len(order) < n:
        prev, curr = order[-2], order[-1]
        nxt = [x for x in ring_adj[curr] if x != prev]
        if not nxt:
            return None
        order.append(nxt[0])
    if len(order) != n:
        return None

    # --- Enumerate all 2n numberings; take the lexicographic-min key ---------
    best_key = None
    best = None  # (free_valence_locant, indicated_h_locant)
    for start in range(n):
        for direction in (1, -1):
            atom_to_pos = {
                order[(start + direction * p) % n]: p + 1
                for p in range(n)
            }
            het_locs = tuple(sorted(atom_to_pos[i] for i in het_atoms))
            seniority_locs = tuple(
                atom_to_pos[i] for i in sorted(
                    het_atoms,
                    key=lambda a: (
                        _HETEROATOM_SENIORITY.get(
                            mol.GetAtomWithIdx(a).GetSymbol(), 99),
                        atom_to_pos[a],
                    ),
                )
            )
            ih_loc = (atom_to_pos[indicated_h_atom]
                      if indicated_h_atom is not None else 0)
            fv_loc = atom_to_pos[attachment_atom]

            key = (het_locs, seniority_locs, ih_loc, fv_loc)
            if best_key is None or key < best_key:
                best_key = key
                best = (fv_loc, ih_loc)

    if best is None:
        return None
    attach_locant, ih_locant = best
    prefix = f"{ih_locant}H-" if indicated_h_atom is not None else ""
    return f"{prefix}{stem}-{attach_locant}-yl"


def get_ring_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_point: Optional[int] = None
) -> str:
    """
    Get the substituent name for a ring when it becomes a substituent on a chain.

    This is the main entry point for ring-as-substituent naming.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_point: Optional ring atom index where the ring attaches to chain.
                         Used for position-specific names (e.g., 2-pyridyl vs 4-pyridyl).

    Returns:
        Substituent name string (e.g., 'phenyl', 'cyclohexyl', '2-pyridyl')
    """
    # Identify the ring system
    ring_name = identify_ring_system(mol, ring_atoms)

    if ring_name is None:
        # Multi-ring system (fused polycyclic): check retained names
        # identify_ring_system() only handles single rings (up to 8 atoms).
        # For multi-ring systems (naphthalene=10, anthracene=14, etc.),
        # extract the fragment SMILES and look up retained names.
        from rdkit import Chem
        frag_smi = Chem.MolFragmentToSmiles(mol, list(ring_atoms), canonical=True)
        if frag_smi:
            from ..data import get_retained_name
            retained = get_retained_name(frag_smi)
            if retained:
                # Convert retained name to substituent form:
                # naphthalene -> naphthalen-{locant}-yl
                # anthracene -> anthracen-{locant}-yl
                # phenanthrene -> phenanthren-{locant}-yl
                # General rule: drop trailing 'e', add '-yl'
                stem = retained.rstrip('e') if retained.endswith('e') else retained
                # Determine attachment locant for position-specific naming
                if attachment_point is not None:
                    attach_locant = _get_polycyclic_attachment_locant(
                        mol, ring_atoms, attachment_point
                    )
                    if attach_locant is not None:
                        return f'{stem}-{attach_locant}-yl'
                return f'{stem}-yl'

        # Unknown ring - generate generic cycloXyl name
        ring_size = len(ring_atoms)
        prefix = _get_chain_prefix(ring_size)
        return f'cyclo{prefix}yl'

    # IUPAC P-31.1.4.3.4: a monocyclic heteroaryl substituent takes
    # free-valence numbering (pyridin-3-yl, 1H-imidazol-5-yl, furan-2-yl, ...),
    # which supersedes both the legacy POSITION_SPECIFIC forms (3-pyridyl) and
    # the locant-less dictionary forms (imidazolyl). Guarded — returns None and
    # falls through to the legacy forms below whenever the locant is not
    # provably PIN-correct (benzene, saturated rings, substituted rings, etc.).
    if attachment_point is not None:
        pin_name = pin_heteroaryl_substituent_name(
            mol, ring_atoms, attachment_point
        )
        if pin_name is not None:
            return pin_name

    # Check for position-specific name
    if ring_name in POSITION_SPECIFIC_RINGS and attachment_point is not None:
        # Determine the position in the ring
        ring_position = _get_ring_position_for_attachment(
            mol, ring_atoms, attachment_point, ring_name
        )
        if ring_position in POSITION_SPECIFIC_RINGS[ring_name]:
            return POSITION_SPECIFIC_RINGS[ring_name][ring_position]

    # Look up standard substituent name
    if ring_name in RING_SUBSTITUENT_NAMES:
        return RING_SUBSTITUENT_NAMES[ring_name]

    # Fallback: convert ring name to substituent form
    # Generally: remove 'e' and add 'yl' (benzene -> benzyl, but benzene -> phenyl is special)
    if ring_name.endswith('ane'):
        return ring_name[:-1] + 'yl'  # cyclohexane -> cyclohexanyl (but we have cyclohexyl in dict)
    elif ring_name.endswith('ene'):
        return ring_name[:-1] + 'yl'  # cyclohexene -> cyclohexenyl
    elif ring_name.endswith('ine'):
        return ring_name[:-1] + 'yl'  # pyridine -> pyridinyl

    return ring_name + 'yl'


def _get_polycyclic_attachment_locant(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int
) -> Optional[int]:
    """
    Determine the IUPAC locant for an attachment point on a polycyclic ring system.

    Uses the retained name's canonical IUPAC numbering. For polycyclic aromatics,
    we use the IUPAC numbering by checking the fused_heterocycles data or by
    using the RDKit canonical atom ordering as a proxy.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring system
        attachment_atom: The ring atom that connects to the chain

    Returns:
        IUPAC locant (1-indexed), or None if cannot determine
    """
    from rdkit import Chem

    # Try to get IUPAC numbering from fused heterocycle data
    frag_smi = Chem.MolFragmentToSmiles(mol, list(ring_atoms), canonical=True)
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        if frag_smi in FUSED_HETEROCYCLE_DATA:
            entry = FUSED_HETEROCYCLE_DATA[frag_smi]
            iupac_locants = entry.get('iupac_locants', {})
            if iupac_locants and attachment_atom in iupac_locants:
                return iupac_locants[attachment_atom]
    except ImportError:
        pass

    # For monocyclic heterocycles with retained names (e.g. 1,3-dioxolane),
    # use proper IUPAC numbering: start from highest-priority heteroatom,
    # go in direction giving lowest locant set for remaining heteroatoms.
    # For all-carbon rings or polycyclic aromatics without explicit data,
    # fall back to sorted atom order.
    ring_set = set(ring_atoms)

    # Collect heteroatoms in the ring
    het_indices = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            het_indices.append(idx)

    if het_indices and len(ring_atoms) <= 8:
        # Monocyclic heterocycle: build a ring traversal path and number
        # according to IUPAC Hantzsch-Widman rules (heteroatom gets pos 1,
        # direction chosen to give lowest locant set for other heteroatoms).

        # Build adjacency within ring
        ring_adj: dict = {idx: [] for idx in ring_atoms}
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in ring_set:
                    ring_adj[idx].append(nidx)

        # Heteroatom priority: O > S > N (Hantzsch-Widman)
        _het_priority = {'O': 0, 'S': 1, 'Se': 2, 'N': 3}
        het_indices_sorted = sorted(
            het_indices,
            key=lambda i: _het_priority.get(mol.GetAtomWithIdx(i).GetSymbol(), 99)
        )
        start_atom = het_indices_sorted[0]  # Highest priority heteroatom = position 1

        # Try both directions around the ring from start_atom
        def _traverse_ring(start, first_next, adj, ring_size):
            """Walk around ring from start via first_next, return ordered path."""
            path = [start, first_next]
            while len(path) < ring_size:
                prev = path[-2]
                curr = path[-1]
                nexts = [n for n in adj[curr] if n != prev and n in ring_set]
                if not nexts:
                    break
                path.append(nexts[0])
            return path

        neighbors_of_start = ring_adj[start_atom]
        best_path = None
        best_het_locants = None

        for next_atom in neighbors_of_start:
            path = _traverse_ring(start_atom, next_atom, ring_adj, len(ring_atoms))
            if len(path) != len(ring_atoms):
                continue
            # Compute heteroatom locant set (excluding position 1 which is always a het)
            het_locs = tuple(sorted(
                path.index(hi) + 1 for hi in het_indices if hi != start_atom
            ))
            if best_het_locants is None or het_locs < best_het_locants:
                best_het_locants = het_locs
                best_path = path

        if best_path and attachment_atom in best_path:
            return best_path.index(attachment_atom) + 1  # 1-indexed

    # Fallback for all-carbon rings or polycyclic systems:
    # use the canonical atom order within the ring system.
    ring_list = sorted(ring_atoms)
    if attachment_atom in ring_list:
        return ring_list.index(attachment_atom) + 1  # 1-indexed

    return None


def _get_ring_position_for_attachment(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
    ring_name: str
) -> Optional[int]:
    """
    Determine the ring position number for an attachment point.

    For position-specific substituents like pyridine (2-pyridyl, 3-pyridyl, 4-pyridyl),
    we need to determine which position the attachment is at based on IUPAC numbering.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_atom: The ring atom that connects to the chain
        ring_name: Name of the ring system

    Returns:
        Ring position number (1-indexed), or None if cannot determine
    """
    # For pyridine: N is at position 1, so we number relative to N
    # Try both ring directions and pick the one giving the lowest locant
    # at the attachment point (IUPAC lowest locant rule).
    if ring_name == 'pyridine':
        # Find the nitrogen
        n_idx = None
        for idx in ring_atoms:
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'N':
                n_idx = idx
                break

        if n_idx is None:
            return None

        # Try both ring traversal directions from N
        ring_set = set(ring_atoms)
        n_atom = mol.GetAtomWithIdx(n_idx)
        ring_neighbors = [
            nbr.GetIdx() for nbr in n_atom.GetNeighbors()
            if nbr.GetIdx() in ring_set
        ]

        best_pos = None
        for start_nbr in ring_neighbors:
            path = _build_ring_path_from_start_via(
                mol, ring_atoms, n_idx, start_nbr
            )
            if attachment_atom in path:
                pos = path.index(attachment_atom) + 1  # 1-indexed
                if best_pos is None or pos < best_pos:
                    best_pos = pos

        return best_pos

    # For naphthalene: determine alpha (1,4,5,8) vs beta (2,3,6,7) position
    # Alpha positions are adjacent to the fusion bond; beta are farther
    if ring_name == 'naphthalene':
        ring_set = set(ring_atoms)
        # Find fusion atoms: ring atoms bonded to 3 other ring atoms
        # (shared atoms between the two 6-membered rings in naphthalene)
        fusion_atoms = set()
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            ring_nbr_count = sum(
                1 for nbr in atom.GetNeighbors() if nbr.GetIdx() in ring_set
            )
            if ring_nbr_count == 3:
                fusion_atoms.add(idx)
        if len(fusion_atoms) == 2 and attachment_atom in ring_set:
            # Alpha positions: atoms adjacent to a fusion atom (but not fusion atoms themselves)
            is_alpha = any(
                mol.GetBondBetweenAtoms(attachment_atom, fa) is not None
                for fa in fusion_atoms
            )
            if attachment_atom in fusion_atoms:
                return None  # Fusion atom itself: use generic naphthyl
            return 1 if is_alpha else 2  # 1-naphthyl (alpha) or 2-naphthyl (beta)
        return None

    return None


def _build_ring_path_from_start(
    mol,
    ring_atoms: Tuple[int, ...],
    start_idx: int
) -> List[int]:
    """
    Build an ordered path around the ring starting from a given atom.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        start_idx: Starting atom index

    Returns:
        List of atom indices in order around the ring
    """
    ring_set = set(ring_atoms)
    path = [start_idx]
    visited = {start_idx}

    current = start_idx
    while len(path) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)

        # Find next ring neighbor not yet visited
        next_idx = None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                next_idx = nbr_idx
                break

        if next_idx is None:
            break

        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx

    return path


def _build_ring_path_from_start_via(
    mol,
    ring_atoms: Tuple[int, ...],
    start_idx: int,
    first_neighbor: int
) -> List[int]:
    """Build an ordered path around the ring starting from start_idx going
    through first_neighbor.

    This allows choosing the direction of traversal around the ring, which
    is needed for IUPAC lowest-locant rule: try both directions and pick
    the one giving lower locants.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        start_idx: Starting atom index (position 1)
        first_neighbor: The neighbor of start_idx to visit first (direction)

    Returns:
        List of atom indices in order around the ring
    """
    ring_set = set(ring_atoms)
    path = [start_idx, first_neighbor]
    visited = {start_idx, first_neighbor}

    current = first_neighbor
    while len(path) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)
        next_idx = None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                next_idx = nbr_idx
                break
        if next_idx is None:
            break
        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx

    return path


def get_ring_attachment_locant(
    mol,
    ring_atoms: Tuple[int, ...],
    chain_atoms: List[int],
    atom_to_locant: Dict[int, int]
) -> int:
    """
    Find which chain position the ring is attached to.

    The ring connects to the chain via a bond between a ring atom and a chain atom.
    This function finds that chain atom and returns its locant.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        chain_atoms: List of atom indices in the principal chain
        atom_to_locant: Mapping from chain atom index to locant (1-indexed)

    Returns:
        Locant (1-indexed position) where the ring attaches to the chain

    Raises:
        ValueError: If no connection found between ring and chain
    """
    ring_set = set(ring_atoms)
    chain_set = set(chain_atoms)

    # Find the bond connecting ring to chain
    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Check if neighbor is on the chain
            if nbr_idx in chain_set:
                # Found the connection
                if nbr_idx in atom_to_locant:
                    return atom_to_locant[nbr_idx]

    # If no direct connection found, raise error
    raise ValueError(
        "Could not find connection between ring and chain. "
        f"Ring atoms: {ring_atoms}, Chain atoms: {chain_atoms}"
    )


def get_ring_attachment_atom(
    mol,
    ring_atoms: Tuple[int, ...],
    chain_atoms: List[int]
) -> Optional[int]:
    """
    Find the ring atom that attaches to the chain.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        chain_atoms: List of atom indices in the principal chain

    Returns:
        Ring atom index that connects to chain, or None if not found
    """
    ring_set = set(ring_atoms)
    chain_set = set(chain_atoms)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            if nbr_idx in chain_set:
                return ring_idx

    return None
