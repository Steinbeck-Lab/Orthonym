"""
Fragment capping for decomposition engine.

After bond cleavage, replaces dummy atoms with H or OH to produce
valid molecule fragments for recursive naming.
"""

from typing import Dict, List


def cleave_and_cap(mol, bond_infos: List[Dict], acid_side_oh: bool = True) -> List[Dict]:
    """Cleave molecule at specified bonds and return H/OH-capped fragments.

    Uses RDKit FragmentOnBonds to cleave, then replaces dummy atoms:
    - Acid-side fragments: cap with OH (produces carboxylic acid) if acid_side_oh=True
    - Alkyl/amine-side fragments: cap with H

    Args:
        mol: RDKit Mol object
        bond_infos: List of bond info dicts from find_cleavable_bonds()
        acid_side_oh: If True, cap acid-side fragments with OH

    Returns:
        List of dicts with keys: smiles, side, original_atoms
    """
    raise NotImplementedError("fragment_capping.cleave_and_cap not yet implemented")
