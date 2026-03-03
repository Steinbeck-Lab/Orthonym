"""
Monocyclic lactone naming using IUPAC heterocyclic replacement nomenclature.

Lactones are cyclic esters. Monocyclic lactones are named as heterocyclic
ketones: the ring oxygen gives the heterocyclic parent name, and the
carbonyl is expressed as a -one suffix at position 2.

Naming algorithm:
1. Detect lactone: ring contains -C(=O)-O- where both C and ester O are
   in the same ring, and the exocyclic O is a double-bonded carbonyl.
2. Determine ring size.
3. For ring size 3-10: Get the Hantzsch-Widman heterocyclic parent name:
   - 3: oxirane, 4: oxetane, 5: oxolane, 6: oxane, 7: oxepane, etc.
4. For ring size 11+: Use replacement nomenclature with chain prefix:
   - 11: oxacycloundecan, 13: oxacyclotridecan, 15: oxacyclopentadecan
5. Apply vowel elision and append '-2-one'.

Reference: IUPAC 2013 Blue Book, P-25.5.2 (Lactones), P-31.1.3 (Replacement)

Examples:
    O=C1CCO1      (beta-propiolactone)    -> oxetan-2-one
    O=C1CCCO1     (gamma-butyrolactone)   -> oxolan-2-one
    O=C1CCCCO1    (delta-valerolactone)   -> oxan-2-one
    O=C1CCCCCO1   (epsilon-caprolactone)  -> oxepan-2-one
    O=C1CCCCCCCCCO1 (10-membered lactone) -> oxacycloundecan-2-one
"""

from collections import deque
from typing import Dict, List, Optional

from rdkit import Chem

from ..rules.heterocycles import build_hw_name


# ---------------------------------------------------------------------------
# Lactone detection
# ---------------------------------------------------------------------------

def is_monocyclic_lactone(mol) -> Optional[Dict]:
    """
    Detect whether a molecule is (or contains) a monocyclic lactone.

    A monocyclic lactone has:
    - A ring containing an ester motif: -C(=O)-O- where both the carbonyl
      carbon and the ester oxygen are in the same ring.
    - The carbonyl oxygen is exocyclic (double-bonded to the carbonyl C).
    - The ring is not fused with another ring (monocyclic only).

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        Dict with detection info if lactone found:
            ring_atoms: tuple of atom indices in the lactone ring
            carbonyl_idx: atom index of the carbonyl carbon (C=O)
            ester_O_idx: atom index of the ring (ester) oxygen
            carbonyl_O_idx: atom index of the exocyclic carbonyl oxygen
            ring_size: number of atoms in the ring
        None if not a monocyclic lactone.
    """
    if mol is None:
        return None

    # SMARTS: carbonyl carbon with double-bonded O and single-bonded O
    # [CX3](=O)[OX2] matches the ester/acid core
    # match[0] = carbonyl carbon
    # match[1] = carbonyl oxygen (=O, exocyclic)
    # match[2] = ester oxygen (-O-, must be in ring)
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        return None

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return None

    for match in matches:
        carbonyl_c = match[0]
        carbonyl_o = match[1]
        ester_o = match[2]

        # Both carbonyl C and ester O must be in the SAME ring
        for ring in atom_rings:
            ring_set = set(ring)
            if carbonyl_c in ring_set and ester_o in ring_set:
                # Exocyclic carbonyl O must NOT be in the ring
                if carbonyl_o in ring_set:
                    continue

                # Check monocyclic: no ring atom should appear in another ring
                is_monocyclic = True
                for other_ring in atom_rings:
                    if set(other_ring) == ring_set:
                        continue
                    if ring_set & set(other_ring):
                        is_monocyclic = False
                        break

                if not is_monocyclic:
                    continue

                return {
                    "ring_atoms": ring,
                    "carbonyl_idx": carbonyl_c,
                    "ester_O_idx": ester_o,
                    "carbonyl_O_idx": carbonyl_o,
                    "ring_size": len(ring),
                }

    return None


# ---------------------------------------------------------------------------
# Lactone ring naming
# ---------------------------------------------------------------------------

# Ring sizes supported via Hantzsch-Widman naming
_HW_RING_SIZES = frozenset(range(3, 11))

# Maximum ring size for macrolide lactone naming
_MAX_MACROLIDE_SIZE = 50


def name_lactone_ring(ring_size: int) -> Optional[str]:
    """
    Get the IUPAC name for a monocyclic lactone of a given ring size.

    For ring sizes 3-10: uses Hantzsch-Widman heterocyclic parent naming.
    For ring sizes 11+: uses replacement nomenclature (oxacyclo{prefix}an-2-one).

    Args:
        ring_size: Number of atoms in the lactone ring (3-50 supported).

    Returns:
        IUPAC name string (e.g., 'oxolan-2-one'), or None if ring size
        is not supported.

    Examples:
        >>> name_lactone_ring(4)
        'oxetan-2-one'
        >>> name_lactone_ring(5)
        'oxolan-2-one'
        >>> name_lactone_ring(6)
        'oxan-2-one'
        >>> name_lactone_ring(7)
        'oxepan-2-one'
        >>> name_lactone_ring(11)
        'oxacycloundecan-2-one'
        >>> name_lactone_ring(13)
        'oxacyclotridecan-2-one'
    """
    if ring_size < 3:
        return None

    # Hantzsch-Widman naming for ring sizes 3-10
    if ring_size in _HW_RING_SIZES:
        parent_name = build_hw_name(
            heteroatoms=[(1, "O")],
            ring_size=ring_size,
            is_saturated=True,
            is_aromatic=False,
        )

        if not parent_name:
            return None

        # Apply vowel elision: remove terminal 'e' before '-one'
        if parent_name.endswith("e"):
            stem = parent_name[:-1]
        else:
            stem = parent_name

        return f"{stem}-2-one"

    # Macrolide naming for ring sizes 11+
    # Uses replacement nomenclature: oxacyclo{chain_prefix}an-2-one
    if ring_size > _MAX_MACROLIDE_SIZE:
        return None

    from ..data.chain_names import get_chain_prefix

    try:
        # The chain prefix corresponds to the total ring size
        # (since one O replaces one C, the ring name uses the total count)
        chain_prefix = get_chain_prefix(ring_size)
    except ValueError:
        return None

    # Build: oxacyclo + {prefix} + an-2-one
    # The chain prefix already provides the stem (e.g., "undec" for 11)
    # Combine: "oxacyclo" + prefix + "an-2-one"
    return f"oxacyclo{chain_prefix}an-2-one"


# ---------------------------------------------------------------------------
# Full lactone naming
# ---------------------------------------------------------------------------

def name_monocyclic_lactone(mol) -> Optional[str]:
    """
    Generate the IUPAC name for a monocyclic lactone, including substituents.

    Detects whether the molecule is a monocyclic lactone and returns
    its name using heterocyclic replacement nomenclature with -one suffix.
    Substituents on the ring are included as prefixes with locants.

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        IUPAC name string (e.g., 'oxolan-2-one', '3-aminooxolan-2-one'),
        or None if the molecule is not a monocyclic lactone.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('O=C1CCCO1')
        >>> name_monocyclic_lactone(mol)
        'oxolan-2-one'
        >>> mol = Chem.MolFromSmiles('NC1CCOC1=O')
        >>> name_monocyclic_lactone(mol)
        '3-aminooxolan-2-one'
    """
    info = is_monocyclic_lactone(mol)
    if info is None:
        return None

    parent_name = name_lactone_ring(info["ring_size"])
    if parent_name is None:
        return None

    # Detect substituents on the lactone ring
    ring_atoms = info["ring_atoms"]
    ester_o_idx = info["ester_O_idx"]
    carbonyl_idx = info["carbonyl_idx"]
    carbonyl_o_idx = info["carbonyl_O_idx"]

    # Build IUPAC locant mapping for lactone ring
    # Numbering: O atom = 1, carbonyl C = 2, then continue around ring
    ring_list = list(ring_atoms)
    ring_set = set(ring_list)

    # Find oxygen position in ring and reorder so O is first
    try:
        o_pos = ring_list.index(ester_o_idx)
    except ValueError:
        return parent_name

    # Reorder ring starting from O, going toward carbonyl C
    ordered = ring_list[o_pos:] + ring_list[:o_pos]

    # Check direction: next atom should be carbonyl C
    if len(ordered) > 1 and ordered[1] != carbonyl_idx:
        # Reverse direction (keep O first)
        ordered = [ordered[0]] + ordered[1:][::-1]

    # Build atom-to-locant mapping (1-indexed)
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(ordered)}

    # Collect stereodescriptors using lactone ring locant mapping
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from rdkit.Chem import rdCIPLabeler

    rdCIPLabeler.AssignCIPLabels(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    # Discover exocyclic substituents via universal pipeline (Phase 86).
    # Parent atoms = ring atoms; exclude = carbonyl O (=O of the lactone).
    # The _integrate_universal_prefixes helper adds exclude_atoms to the
    # effective parent set so they are never discovered as substituents.
    from ..assembly.composer import _integrate_universal_prefixes
    prefix_str = _integrate_universal_prefixes(
        mol, ring_set,
        parent_type="ring",
        oriented_ring=ordered,
        atom_to_locant=atom_to_locant,
        exclude_atoms={carbonyl_o_idx},
    )

    if not prefix_str:
        # No substituents but may have stereo
        if stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            return f"{stereo_prefix}{parent_name}"
        return parent_name

    name = f"{prefix_str}{parent_name}"

    # Prepend stereo prefix if descriptors exist
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        name = f"{stereo_prefix}{name}"

    return name


def _detect_lactone_substituents(mol, ordered_ring, atom_to_locant, excluded):
    """
    Detect substituents on a lactone ring.

    Args:
        mol: RDKit Mol object
        ordered_ring: List of atom indices in IUPAC numbering order
        atom_to_locant: Dict mapping atom index to IUPAC locant
        excluded: Set of atom indices to exclude (ring + carbonyl O)

    Returns:
        List of (substituent_name, locant) tuples
    """
    substituents = []

    for ring_atom_idx in ordered_ring:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        locant = atom_to_locant[ring_atom_idx]

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in excluded:
                continue

            sub_name = _identify_lactone_substituent(mol, nbr_idx, excluded)
            if sub_name:
                substituents.append((sub_name, locant))

    return substituents


def _identify_lactone_substituent(mol, start_idx, excluded):
    """
    Identify a substituent on a lactone ring by its starting atom.

    Returns:
        Substituent prefix name (e.g., 'amino', 'methyl', 'hydroxy') or None
    """
    atom = mol.GetAtomWithIdx(start_idx)
    symbol = atom.GetSymbol()

    # Halogens
    halogen_map = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    if symbol in halogen_map:
        return halogen_map[symbol]

    # Nitrogen: amino (-NH2), nitro, etc.
    if symbol == 'N':
        h_count = atom.GetTotalNumHs()
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 2 and len(neighbors) == 0:
            return 'amino'
        if atom.GetFormalCharge() == 1:
            o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
            if o_count == 2:
                return 'nitro'

    # Oxygen: hydroxy
    if symbol == 'O':
        h_count = atom.GetTotalNumHs()
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 1 and len(neighbors) == 0:
            return 'hydroxy'

    # Carbon: alkyl groups
    if symbol == 'C':
        # BFS for pure alkyl
        visited = {start_idx}
        queue = deque([start_idx])
        all_atoms = []
        carbon_count = 0
        is_pure_alkyl = True

        while queue:
            idx = queue.popleft()
            a = mol.GetAtomWithIdx(idx)
            all_atoms.append(idx)
            if a.GetSymbol() == 'C':
                carbon_count += 1
            elif a.GetSymbol() != 'H':
                is_pure_alkyl = False

            for nbr in a.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited and nbr_idx not in excluded:
                    visited.add(nbr_idx)
                    queue.append(nbr_idx)

        if is_pure_alkyl and carbon_count > 0:
            from ..assembly.naming_utils import get_alkyl_name
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass

    return None
