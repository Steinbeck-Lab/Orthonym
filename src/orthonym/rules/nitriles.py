"""
Nitrile naming rules for IUPAC nomenclature.

Nitrile naming follows these patterns:
- Chain-terminal nitriles: stem + 'nitrile' (acetonitrile, propanenitrile)
- Ring-attached nitriles: parent + 'carbonitrile' (cyclohexanecarbonitrile)

The nitrile carbon (C of -C#N) IS counted in the chain length.

Based on IUPAC 2013 Blue Book.
"""

from typing import List, Optional


from ..data.chain_names import get_chain_prefix as _get_chain_prefix


def is_ring_attached_nitrile(mol, nitrile_atoms: tuple) -> bool:
    """
    Check if a nitrile group is directly attached to a ring.

    A ring-attached nitrile is one where the nitrile carbon (C of -C#N)
    is directly bonded to a ring carbon. These use the -carbonitrile suffix.

    Args:
        mol: RDKit Mol object
        nitrile_atoms: Atom indices from nitrile SMARTS match [CX2]#[NX1]
                       where index 0 is the carbon

    Returns:
        True if the nitrile carbon is attached to a ring atom
    """
    if not nitrile_atoms or len(nitrile_atoms) < 2:
        return False

    # SMARTS "[CX2]#[NX1]" - match[0] is the carbon, match[1] is nitrogen
    nitrile_carbon_idx = nitrile_atoms[0]
    nitrile_carbon = mol.GetAtomWithIdx(nitrile_carbon_idx)

    # Check if any neighbor of the nitrile carbon is in a ring
    for neighbor in nitrile_carbon.GetNeighbors():
        # Skip the nitrogen of the nitrile group itself
        if neighbor.GetIdx() == nitrile_atoms[1]:
            continue
        # Check if this neighbor is part of a ring
        if neighbor.IsInRing():
            return True

    return False


def get_nitrile_parent_chain(
    mol,
    nitrile_atoms: tuple,
    principal_chain: Optional[List[int]] = None
) -> List[int]:
    """
    Get the parent chain for a chain-terminal nitrile.

    For chain-terminal nitriles, the chain includes the nitrile carbon.
    The nitrile carbon is ALWAYS position 1 (terminal).

    Args:
        mol: RDKit Mol object
        nitrile_atoms: Atom indices from nitrile SMARTS match
        principal_chain: Optional pre-computed principal chain

    Returns:
        List of atom indices representing the parent chain
        The nitrile carbon will be at index 0 (position 1 in naming)
    """
    if principal_chain:
        return principal_chain

    # Simple chain building from nitrile carbon
    nitrile_carbon_idx = nitrile_atoms[0]

    # BFS to find longest chain from nitrile carbon
    chain = [nitrile_carbon_idx]
    visited = {nitrile_carbon_idx, nitrile_atoms[1]}  # Exclude the nitrogen

    current = mol.GetAtomWithIdx(nitrile_carbon_idx)

    while True:
        next_carbon = None
        for neighbor in current.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and neighbor.GetSymbol() == 'C':
                visited.add(nbr_idx)
                next_carbon = neighbor
                break

        if next_carbon is None:
            break

        chain.append(next_carbon.GetIdx())
        current = next_carbon

    return chain


def get_ring_parent_name(mol, ring_atoms: tuple) -> str:
    """
    Get the parent name for a ring attached to a nitrile.

    Args:
        mol: RDKit Mol object
        ring_atoms: Atom indices of the ring

    Returns:
        Ring parent name (e.g., "cyclohexane", "cyclopentane")
    """
    ring_size = len(ring_atoms)

    stem = _get_chain_prefix(ring_size)

    return f"cyclo{stem}ane"


def name_nitrile(
    mol,
    nitrile_atoms: tuple,
    parent_name: Optional[str] = None,
    chain_length: Optional[int] = None,
    is_ring: Optional[bool] = None
) -> str:
    """
    Generate IUPAC name for a nitrile compound.

    Args:
        mol: RDKit Mol object
        nitrile_atoms: Atom indices from nitrile SMARTS match
        parent_name: Optional ring parent name (e.g., "cyclohexane")
        chain_length: Optional chain length (if known)
        is_ring: Optional flag if nitrile is ring-attached

    Returns:
        IUPAC name for the nitrile (e.g., "propanenitrile", "cyclohexanecarbonitrile")
    """
    # Determine if ring-attached
    ring_attached = is_ring if is_ring is not None else is_ring_attached_nitrile(mol, nitrile_atoms)

    if ring_attached:
        # Ring-attached: parent + carbonitrile
        if parent_name:
            # Remove trailing 'e' if present (cyclohexane -> cyclohexan)
            base = parent_name.rstrip('e') if parent_name.endswith('e') else parent_name
            return f"{base}ecarbonitrile"
        else:
            # Try to determine ring from neighbor
            nitrile_carbon_idx = nitrile_atoms[0]
            nitrile_carbon = mol.GetAtomWithIdx(nitrile_carbon_idx)

            for neighbor in nitrile_carbon.GetNeighbors():
                if neighbor.IsInRing():
                    # Find the ring containing this atom
                    ring_info = mol.GetRingInfo()
                    for ring in ring_info.AtomRings():
                        if neighbor.GetIdx() in ring:
                            ring_name = get_ring_parent_name(mol, ring)
                            return f"{ring_name}carbonitrile"

            return "carbonitrile"  # Fallback
    else:
        # Chain-terminal: determine chain length
        if chain_length is None:
            chain = get_nitrile_parent_chain(mol, nitrile_atoms)
            chain_length = len(chain)

        stem = _get_chain_prefix(chain_length)

        # stem + ane -> stem + anenitrile (e.g., propane -> propanenitrile)
        return f"{stem}anenitrile"


def get_nitrile_locant(
    nitrile_atoms: tuple,
    atom_to_locant: dict
) -> Optional[int]:
    """
    Get the locant for a nitrile group.

    For chain-terminal nitriles, the locant is always 1 (implicit).
    For nitriles as prefixes (cyano-), we need the chain position.

    Args:
        nitrile_atoms: Atom indices from nitrile SMARTS match
        atom_to_locant: Mapping from atom index to chain locant

    Returns:
        Locant position or None if not on chain
    """
    nitrile_carbon_idx = nitrile_atoms[0]

    # Check if nitrile carbon is on the chain
    if nitrile_carbon_idx in atom_to_locant:
        return atom_to_locant[nitrile_carbon_idx]

    return None
