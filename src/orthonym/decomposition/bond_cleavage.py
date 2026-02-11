"""
Bond cleavage detection for decomposition engine.

Identifies ester, amide, and glycosidic bonds suitable for cleavage,
with guards to exclude cyclic variants (lactones, lactams) and
overlapping patterns (carbamates, ureas).
"""

from typing import Dict, List


def find_cleavable_bonds(mol) -> List[Dict]:
    """Find cleavable bonds in a molecule.

    Detects ester C-O, amide C-N, and glycosidic C-O-C bonds,
    excluding lactones, lactams, carbamates, and ureas.

    Args:
        mol: RDKit Mol object

    Returns:
        List of dicts with keys: bond_idx, type, match, acid_atom,
        alkyl_atom (for esters) or amine_atom (for amides).
    """
    raise NotImplementedError("bond_cleavage.find_cleavable_bonds not yet implemented")
