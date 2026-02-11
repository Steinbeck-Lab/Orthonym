"""
Decomposition engine for multi-fragment molecule naming.

Detects cleavable functional bonds (ester, amide, glycosidic),
fragments molecules at those bonds, H/OH-caps the fragments,
and produces valid SMILES for recursive naming.
"""

from .bond_cleavage import find_cleavable_bonds
from .fragment_capping import cleave_and_cap

__all__ = ["find_cleavable_bonds", "cleave_and_cap"]
