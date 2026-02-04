"""
Natural product scaffold detection via RDKit substructure matching.

Uses flexible query patterns (bond-generic) so that unsaturated derivatives
(e.g. cholesterol with C=C in ring A) still match their saturated parent
scaffold (cholestane). Stereochemistry is also relaxed in patterns to allow
matching molecules with undefined or different stereocenters.

Provides:
- detect_natural_product(): Main detection entry point
- get_non_scaffold_atoms(): Identify atoms not part of matched scaffold
- get_scaffold_substituents(): Identify substituents attached to scaffold
- is_steroid(): Quick steroid check
- is_alkaloid(): Quick alkaloid check
"""

import logging
from typing import Dict, List, Optional, Set

from rdkit import Chem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level cache for flexible query patterns
# ---------------------------------------------------------------------------
_FLEXIBLE_PATTERNS: Dict[str, Chem.Mol] = {}
_PATTERNS_INITIALIZED = False


def _init_flexible_patterns() -> None:
    """Build bond-generic, stereo-free query molecules for scaffold matching.

    Each scaffold SMILES is converted to a query where:
    - Bond types are generic (single matches double, etc.)
    - Stereochemistry is removed
    This allows derivatives with unsaturation or different stereo to still
    match their parent scaffold.
    """
    global _PATTERNS_INITIALIZED
    if _PATTERNS_INITIALIZED:
        return

    try:
        from orthonym.data.natural_products import (
            NATURAL_PRODUCT_SCAFFOLDS,
            get_scaffold_patterns,
        )
    except ImportError:
        logger.warning("Natural product data module not available")
        _PATTERNS_INITIALIZED = True
        return

    raw_patterns = get_scaffold_patterns()
    for smiles, mol in raw_patterns.items():
        # Create a copy to avoid modifying the data module's patterns
        mol_copy = Chem.RWMol(mol)
        Chem.RemoveStereochemistry(mol_copy)
        params = Chem.AdjustQueryParameters()
        params.makeBondsGeneric = True
        params.adjustDegree = False
        params.adjustRingCount = False
        query = Chem.AdjustQueryProperties(mol_copy, params)
        if query is not None:
            _FLEXIBLE_PATTERNS[smiles] = query

    _PATTERNS_INITIALIZED = True


def _get_flexible_patterns() -> Dict[str, Chem.Mol]:
    """Return flexible query patterns, initializing on first call."""
    if not _PATTERNS_INITIALIZED:
        _init_flexible_patterns()
    return _FLEXIBLE_PATTERNS


# ---------------------------------------------------------------------------
# BFS helper for substituent collection
# ---------------------------------------------------------------------------

def _bfs_substituent(mol, start_idx: int, exclude_set: set) -> list:
    """BFS from start_idx collecting connected non-scaffold atoms.

    Args:
        mol: RDKit Mol object.
        start_idx: Starting atom index (first atom of substituent).
        exclude_set: Set of atom indices to exclude (scaffold atoms).

    Returns:
        List of atom indices in the substituent fragment.
    """
    visited = {start_idx}
    queue = [start_idx]
    result = [start_idx]
    while queue:
        current = queue.pop(0)
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude_set:
                visited.add(nbr_idx)
                queue.append(nbr_idx)
                result.append(nbr_idx)
    return result


# ---------------------------------------------------------------------------
# Main detection functions
# ---------------------------------------------------------------------------

def detect_natural_product(mol) -> Optional[Dict]:
    """Detect natural product scaffold in a molecule via substructure matching.

    Uses flexible (bond-generic, stereo-free) query patterns so that both
    saturated parent scaffolds and unsaturated derivatives are correctly
    identified. When multiple scaffolds match, the LARGEST match (most atoms)
    wins, giving the most specific scaffold identification.

    Args:
        mol: RDKit Mol object. Returns None if mol is None.

    Returns:
        Dict with keys:
            - scaffold_name: str (e.g. "androstane")
            - scaffold_stem: str (e.g. "androst")
            - scaffold_class: str (e.g. "steroid")
            - scaffold_smiles: str (canonical SMILES of matched scaffold)
            - matched_atoms: tuple of int (atom indices in scaffold)
            - non_scaffold_atoms: set of int (atom indices NOT in scaffold)
        Returns None if no scaffold matches or if inputs are invalid.
    """
    if mol is None:
        return None

    try:
        from orthonym.data.natural_products import NATURAL_PRODUCT_SCAFFOLDS
    except ImportError:
        logger.warning("Natural product data module not available")
        return None

    patterns = _get_flexible_patterns()
    if not patterns:
        logger.warning("No scaffold patterns compiled")
        return None

    # Collect all matches, track the largest
    best_match = None
    best_match_size = 0
    best_smiles = None

    for smiles, query_mol in patterns.items():
        if mol.HasSubstructMatch(query_mol):
            match = mol.GetSubstructMatch(query_mol)
            if len(match) > best_match_size:
                best_match = match
                best_match_size = len(match)
                best_smiles = smiles

    if best_match is None:
        return None

    scaffold_info = NATURAL_PRODUCT_SCAFFOLDS[best_smiles]
    non_scaffold = get_non_scaffold_atoms(mol, best_match)

    return {
        "scaffold_name": scaffold_info["name"],
        "scaffold_stem": scaffold_info["stem"],
        "scaffold_class": scaffold_info["class"],
        "scaffold_smiles": best_smiles,
        "matched_atoms": best_match,
        "non_scaffold_atoms": non_scaffold,
    }


def get_non_scaffold_atoms(mol, matched_atoms: tuple) -> set:
    """Return atom indices NOT part of the matched scaffold.

    Args:
        mol: RDKit Mol object.
        matched_atoms: Tuple of atom indices that are part of the scaffold.

    Returns:
        Set of atom indices not in the scaffold.
    """
    all_atoms = set(range(mol.GetNumAtoms()))
    return all_atoms - set(matched_atoms)


def get_scaffold_substituents(mol, matched_atoms: tuple) -> List[Dict]:
    """Identify substituents attached to the scaffold.

    For each scaffold atom, checks its neighbors. If a neighbor is not part
    of the scaffold, a BFS is performed to collect the full substituent group.

    Args:
        mol: RDKit Mol object.
        matched_atoms: Tuple of atom indices that are part of the scaffold.

    Returns:
        List of dicts, each with:
            - attachment_atom: int (scaffold atom idx where substituent attaches)
            - substituent_atoms: list of int (all atoms in the substituent)
            - first_atom: int (first atom of substituent, bonded to scaffold)
    """
    scaffold_set = set(matched_atoms)
    visited_substituent_atoms: Set[int] = set()
    substituents = []

    for scaffold_idx in matched_atoms:
        atom = mol.GetAtomWithIdx(scaffold_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in scaffold_set and nbr_idx not in visited_substituent_atoms:
                # Found a new substituent starting at nbr_idx
                sub_atoms = _bfs_substituent(mol, nbr_idx, scaffold_set)
                visited_substituent_atoms.update(sub_atoms)
                substituents.append({
                    "attachment_atom": scaffold_idx,
                    "substituent_atoms": sub_atoms,
                    "first_atom": nbr_idx,
                })

    return substituents


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def is_steroid(mol) -> bool:
    """Quick check: does molecule contain a steroid scaffold?

    Args:
        mol: RDKit Mol object. Returns False if mol is None.

    Returns:
        True if molecule contains a steroid scaffold.
    """
    if mol is None:
        return False
    result = detect_natural_product(mol)
    if result is None:
        return False
    return result["scaffold_class"] == "steroid"


def is_alkaloid(mol) -> bool:
    """Quick check: does molecule contain an alkaloid scaffold?

    Args:
        mol: RDKit Mol object. Returns False if mol is None.

    Returns:
        True if molecule contains an alkaloid scaffold.
    """
    if mol is None:
        return False
    result = detect_natural_product(mol)
    if result is None:
        return False
    return result["scaffold_class"] == "alkaloid"
