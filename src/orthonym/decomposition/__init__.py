"""
Decomposition engine for multi-fragment molecule naming.

Detects cleavable functional bonds (ester, amide, glycosidic, carbamate),
fragments molecules at those bonds, H/OH-caps the fragments,
names each fragment recursively, and assembles multi-component IUPAC names.

Public API:
    try_decompose(mol, style="pin") -> Optional[str]
        Main entry point. Returns assembled name or None.
    find_cleavable_bonds(mol) -> List[Dict]
        Detect cleavable bonds in a molecule.
    cleave_and_cap(mol, bond_infos, acid_side_oh=True) -> List[Dict]
        Cleave molecule at bonds and return capped fragments.
"""

from .bond_cleavage import find_cleavable_bonds
from .fragment_capping import cleave_and_cap
from .engine import try_decompose

__all__ = ["try_decompose", "find_cleavable_bonds", "cleave_and_cap"]
