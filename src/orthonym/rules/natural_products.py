"""Natural product naming rules.

Provides the naming function that bridges perception (scaffold detection)
and data (derivative lookup) to return trivial/retained names for natural
product molecules.

Algorithm:
1. Exact derivative lookup (O(1) dict lookup by canonical SMILES)
2. Scaffold substructure match via perception module
3. Return scaffold name for bare scaffolds, or as fallback for substituted ones
4. Return None for non-natural-product molecules

Future: name_natural_product_with_substituents() will add substituent prefixes
to scaffold names (e.g. "3beta-hydroxycholestane").
"""

from typing import Dict, Optional

from rdkit import Chem

from ..data.natural_products import NATURAL_PRODUCT_DERIVATIVES, get_natural_product_name
from ..perception.natural_products import detect_natural_product


def name_natural_product(mol) -> Optional[str]:
    """Name a molecule using natural product recognition.

    Tries exact derivative lookup first, then scaffold substructure matching.
    Returns None for molecules that are not recognized natural products.

    Args:
        mol: RDKit Mol object.

    Returns:
        Trivial/retained name if the molecule is a recognized natural product,
        otherwise None.
    """
    if mol is None:
        return None

    # Step 1: Get canonical SMILES for exact lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Step 2: Exact derivative match (cholesterol, morphine, etc.)
    exact_name = get_natural_product_name(canonical)
    if exact_name is not None:
        return exact_name

    # Step 3: Scaffold substructure match
    scaffold_info = detect_natural_product(mol)
    if scaffold_info is None:
        return None

    # Step 4: Check if molecule is just the bare scaffold (no substituents)
    non_scaffold = scaffold_info["non_scaffold_atoms"]
    if not non_scaffold:
        # Exact scaffold match (no extra atoms) -- return parent name
        return scaffold_info["scaffold_name"]

    # Step 5: Check if non-scaffold atoms are only hydrogens
    # (explicit H atoms that are not part of the scaffold pattern)
    all_h = all(
        mol.GetAtomWithIdx(idx).GetAtomicNum() == 1
        for idx in non_scaffold
    )
    if all_h:
        return scaffold_info["scaffold_name"]

    # Step 6: Scaffold with substituents -- return scaffold name as fallback
    # Future: call name_natural_product_with_substituents() for full naming
    return name_natural_product_with_substituents(mol, scaffold_info)


def name_natural_product_with_substituents(mol, scaffold_info: Dict) -> str:
    """Name a natural product with substituents on the scaffold.

    Currently returns just the scaffold name. Future phases will add
    substituent prefixes (e.g. "3beta-hydroxycholestane").

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product() with keys:
            - scaffold_name: str
            - scaffold_stem: str
            - scaffold_class: str
            - matched_atoms: tuple
            - non_scaffold_atoms: set

    Returns:
        Name string for the natural product.
    """
    return scaffold_info["scaffold_name"]
