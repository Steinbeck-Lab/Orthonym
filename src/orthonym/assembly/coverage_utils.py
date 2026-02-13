"""Lightweight coverage estimation functions for naming-time gating.

These functions estimate what fraction of a molecule's heavy atoms the
generated parent name accounts for, WITHOUT requiring OPSIN or any
external process.  All computations are O(1) using data already available
in the naming pipeline (atom indices, name strings, heavy-atom counts).

Two estimators are provided:

1. ``estimate_parent_coverage`` -- uses the actual set of parent atom
   indices.  Exact when available.

2. ``estimate_name_coverage_heuristic`` -- uses name length vs heavy atom
   count.  Useful when atom indices are not tracked explicitly.

Usage example::

    from orthonym.assembly.coverage_utils import estimate_parent_coverage

    coverage = estimate_parent_coverage(mol, parent_atom_indices)
    if coverage < 0.60 and mol.GetNumHeavyAtoms() > 10:
        return None  # fall through to next handler
"""


def estimate_parent_coverage(mol, parent_atom_indices: set) -> float:
    """Fraction of molecule's heavy atoms in the named parent structure.

    Args:
        mol: RDKit Mol object.
        parent_atom_indices: Set of atom indices in the named parent.

    Returns:
        Coverage ratio 0.0-1.0.
    """
    total_heavy = mol.GetNumHeavyAtoms()
    if total_heavy == 0:
        return 1.0
    return len(parent_atom_indices) / total_heavy


def estimate_name_coverage_heuristic(name: str, mol) -> float:
    """Heuristic coverage when atom indices unavailable.

    Uses observation: adequate IUPAC names have >= 0.5 chars per heavy atom.
    Returns approximate coverage 0.0-1.0.

    Args:
        name: The generated IUPAC name string.
        mol: RDKit Mol object.

    Returns:
        Approximate coverage fraction (0.0-1.0), capped at 1.0.
    """
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 1.0
    name_ratio = len(name) / heavy
    return min(name_ratio / 1.5, 1.0)
