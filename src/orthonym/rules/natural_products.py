"""Natural product naming rules.

Provides the naming function that bridges perception (scaffold detection)
and data (derivative lookup) to return trivial/retained names for natural
product molecules.

Algorithm:
1. Exact derivative lookup (O(1) dict lookup by canonical SMILES)
2. Scaffold substructure match via perception module
3. Return scaffold name for bare scaffolds
4. For decorated scaffolds (steroid class): enumerate -OH, =O, C=C
   and assemble a systematic name using the scaffold stem
5. Return None for non-natural-product molecules

Decoration enumeration (steroids):
- Hydroxyl groups  -> prefix "hydroxy" with locant
- Ketone groups    -> suffix "-one" with locant
- Double bonds     -> suffix "-ene" with locant
- Triple bonds     -> suffix "-yne" with locant
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..data.natural_products import (
    NATURAL_PRODUCT_DERIVATIVES,
    get_natural_product_name,
    get_steroid_numbering,
)
from ..perception.natural_products import (
    detect_natural_product,
    get_scaffold_substituents,
)


def name_natural_product(mol) -> Optional[str]:
    """Name a molecule using natural product recognition.

    Tries exact derivative lookup first, then scaffold substructure matching.
    For steroids with decorations, enumerates functional groups.
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
        # Bare scaffold -- but check for internal unsaturation
        numbering = _build_target_to_iupac(scaffold_info)
        if numbering:
            unsaturation = _find_scaffold_unsaturation(
                mol, set(scaffold_info["matched_atoms"]), numbering
            )
            if unsaturation["ene"] or unsaturation["yne"]:
                return _assemble_np_name(
                    scaffold_info["scaffold_stem"],
                    scaffold_info["scaffold_name"],
                    hydroxyls=[],
                    ketones=[],
                    unsaturation=unsaturation,
                )
        return scaffold_info["scaffold_name"]

    # Step 5: Check if non-scaffold atoms are only hydrogens
    all_h = all(
        mol.GetAtomWithIdx(idx).GetAtomicNum() == 1
        for idx in non_scaffold
    )
    if all_h:
        return scaffold_info["scaffold_name"]

    # Step 6: Scaffold with substituents -- enumerate decorations
    return name_natural_product_with_substituents(mol, scaffold_info)


def name_natural_product_with_substituents(mol, scaffold_info: Dict) -> str:
    """Name a natural product with decorations on the scaffold.

    For steroids: enumerates hydroxyl, ketone, and unsaturation decorations,
    assembles a systematic name like "3-hydroxycholest-4-en-17-one".

    For non-steroid scaffolds or when numbering is unavailable, falls back
    to returning just the scaffold name.

    Args:
        mol: RDKit Mol object.
        scaffold_info: Dict from detect_natural_product() with keys:
            - scaffold_name: str
            - scaffold_stem: str
            - scaffold_class: str
            - scaffold_smiles: str
            - matched_atoms: tuple
            - non_scaffold_atoms: set

    Returns:
        Name string for the natural product.
    """
    scaffold_class = scaffold_info["scaffold_class"]
    scaffold_stem = scaffold_info["scaffold_stem"]
    scaffold_name = scaffold_info["scaffold_name"]

    # Only enumerate decorations for steroids with IUPAC numbering maps
    if scaffold_class != "steroid":
        return scaffold_name

    # Build target atom index -> IUPAC locant mapping
    numbering = _build_target_to_iupac(scaffold_info)
    if numbering is None:
        return scaffold_name

    matched_set = set(scaffold_info["matched_atoms"])

    # 1. Find hydroxyl groups (exocyclic -OH on scaffold atoms)
    hydroxyls = _find_hydroxyls(mol, scaffold_info, numbering)

    # 2. Find ketone groups (exocyclic =O on scaffold atoms)
    ketones = _find_ketones(mol, scaffold_info, numbering)

    # 3. Find unsaturation within the scaffold (C=C, C#C)
    unsaturation = _find_scaffold_unsaturation(mol, matched_set, numbering)

    # If no decorations found, return bare scaffold name
    if not hydroxyls and not ketones and not unsaturation["ene"] and not unsaturation["yne"]:
        return scaffold_name

    # 4. Assemble the decorated name
    return _assemble_np_name(scaffold_stem, scaffold_name, hydroxyls, ketones, unsaturation)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_target_to_iupac(scaffold_info: Dict) -> Optional[Dict[int, int]]:
    """Build mapping from target molecule atom indices to IUPAC locants.

    Uses the STEROID_NUMBERING_MAPS from the data module.

    Args:
        scaffold_info: Dict from detect_natural_product().

    Returns:
        Dict mapping target atom index to IUPAC locant, or None if
        no numbering map is available for this scaffold.
    """
    scaffold_smiles = scaffold_info["scaffold_smiles"]
    numbering_map = get_steroid_numbering(scaffold_smiles)
    if numbering_map is None:
        return None

    matched_atoms = scaffold_info["matched_atoms"]
    target_to_iupac = {}
    for query_pos, target_idx in enumerate(matched_atoms):
        if query_pos in numbering_map:
            target_to_iupac[target_idx] = numbering_map[query_pos]

    return target_to_iupac


def _find_hydroxyls(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[int]:
    """Find hydroxyl groups (-OH) attached to scaffold atoms.

    Returns a sorted list of IUPAC locants where -OH groups are found.
    """
    hydroxyls = []
    matched_set = set(scaffold_info["matched_atoms"])

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        # Single-atom substituent that is oxygen with 1 H -> hydroxyl
        if len(sub["substituent_atoms"]) == 1:
            first_idx = sub["first_atom"]
            atom = mol.GetAtomWithIdx(first_idx)
            if atom.GetAtomicNum() == 8:  # Oxygen
                bond = mol.GetBondBetweenAtoms(attach, first_idx)
                if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                    # Check it has at least 1 H (not ether, not ester)
                    if atom.GetTotalNumHs() >= 1:
                        hydroxyls.append(numbering[attach])

    return sorted(hydroxyls)


def _find_ketones(
    mol, scaffold_info: Dict, numbering: Dict[int, int]
) -> List[int]:
    """Find ketone groups (=O) on scaffold atoms.

    Returns a sorted list of IUPAC locants where C=O groups are found.
    """
    ketones = []
    matched_set = set(scaffold_info["matched_atoms"])

    subs = get_scaffold_substituents(mol, scaffold_info["matched_atoms"])
    for sub in subs:
        attach = sub["attachment_atom"]
        if attach not in numbering:
            continue

        # Single-atom substituent that is doubly-bonded oxygen -> ketone
        if len(sub["substituent_atoms"]) == 1:
            first_idx = sub["first_atom"]
            atom = mol.GetAtomWithIdx(first_idx)
            if atom.GetAtomicNum() == 8:  # Oxygen
                bond = mol.GetBondBetweenAtoms(attach, first_idx)
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    ketones.append(numbering[attach])

    return sorted(ketones)


def _find_scaffold_unsaturation(
    mol, matched_set: set, numbering: Dict[int, int]
) -> Dict[str, List[int]]:
    """Find double and triple bonds within the scaffold.

    Returns a dict with 'ene' and 'yne' keys, each a sorted list of
    IUPAC locants (lower locant of each unsaturated bond).
    """
    ene_locants = []
    yne_locants = []

    for bond in mol.GetBonds():
        b_idx = bond.GetBeginAtomIdx()
        e_idx = bond.GetEndAtomIdx()

        # Both atoms must be in the scaffold
        if b_idx not in matched_set or e_idx not in matched_set:
            continue

        # Both must have IUPAC locants
        if b_idx not in numbering or e_idx not in numbering:
            continue

        # Only carbon-carbon bonds
        b_atom = mol.GetAtomWithIdx(b_idx)
        e_atom = mol.GetAtomWithIdx(e_idx)
        if b_atom.GetAtomicNum() != 6 or e_atom.GetAtomicNum() != 6:
            continue

        b_loc = numbering[b_idx]
        e_loc = numbering[e_idx]
        lower_loc = min(b_loc, e_loc)

        if bond.GetBondType() == Chem.BondType.DOUBLE:
            ene_locants.append(lower_loc)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locants.append(lower_loc)

    return {"ene": sorted(ene_locants), "yne": sorted(yne_locants)}


def _assemble_np_name(
    stem: str,
    scaffold_name: str,
    hydroxyls: List[int],
    ketones: List[int],
    unsaturation: Dict[str, List[int]],
) -> str:
    """Assemble a decorated natural product name.

    Format: {prefix}{stem}{unsaturation_suffix}{ketone_suffix}
    Example: "3-hydroxycholest-4-en-17-one"

    Args:
        stem: Scaffold stem (e.g., "cholest", "androst").
        scaffold_name: Full scaffold name for fallback.
        hydroxyls: Sorted IUPAC locants of -OH groups.
        ketones: Sorted IUPAC locants of =O groups.
        unsaturation: Dict with 'ene' and 'yne' locant lists.

    Returns:
        Assembled IUPAC-style natural product name.
    """
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    # --- Build prefix (hydroxy groups) ---
    prefix = ""
    if hydroxyls:
        locant_str = ",".join(str(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        prefix = f"{locant_str}-{multiplier}hydroxy"

    # --- Build unsaturation suffix ---
    ene_locs = unsaturation.get("ene", [])
    yne_locs = unsaturation.get("yne", [])

    unsat_suffix = ""
    if ene_locs or yne_locs:
        parts = []
        if ene_locs:
            ene_locant_str = ",".join(str(loc) for loc in ene_locs)
            ene_count = len(ene_locs)
            if ene_count == 1:
                parts.append(f"-{ene_locant_str}-en")
            else:
                ene_multi = SIMPLE_MULTIPLIERS.get(ene_count, str(ene_count))
                parts.append(f"-{ene_locant_str}-{ene_multi}en")
        if yne_locs:
            yne_locant_str = ",".join(str(loc) for loc in yne_locs)
            yne_count = len(yne_locs)
            if yne_count == 1:
                parts.append(f"-{yne_locant_str}-yn")
            else:
                yne_multi = SIMPLE_MULTIPLIERS.get(yne_count, str(yne_count))
                parts.append(f"-{yne_locant_str}-{yne_multi}yn")
        unsat_suffix = "".join(parts)
    else:
        # Fully saturated -> "an" infix
        unsat_suffix = "an"

    # --- Build ketone suffix ---
    ketone_suffix = ""
    if ketones:
        locant_str = ",".join(str(loc) for loc in ketones)
        count = len(ketones)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        ketone_suffix = f"-{locant_str}-{multiplier}one"

    # --- Combine suffix for -ol (hydroxyl as suffix) when no ketone ---
    # IUPAC convention: if hydroxyl is the only principal group, use -ol suffix
    # But if ketone is present, hydroxyl becomes a prefix
    # For NP names: hydroxyl is always prefix, ketone always suffix
    # If neither ketone nor hydroxyl, and no unsaturation -> bare scaffold name
    if not prefix and not ketone_suffix and not ene_locs and not yne_locs:
        return scaffold_name

    # If only hydroxyls and no ketone -> use -ol suffix instead of prefix
    if hydroxyls and not ketones:
        locant_str = ",".join(str(loc) for loc in hydroxyls)
        count = len(hydroxyls)
        multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        ol_suffix = f"-{locant_str}-{multiplier}ol"
        # stem + unsaturation + -ol
        # e.g., "cholest-5-en-3-ol" for cholesterol-type
        return f"{stem}{unsat_suffix}{ol_suffix}"

    # General case: prefix + stem + unsaturation + ketone
    # e.g., "17-hydroxyandr-4-en-3-one" for testosterone-type
    if unsat_suffix == "an":
        # Saturated: prefix + stem + "an" + ketone
        # e.g., "androstane-3,17-dione" -> but with 'e' before suffix
        if ketone_suffix:
            return f"{prefix}{stem}{unsat_suffix}e{ketone_suffix}"
        else:
            return f"{prefix}{stem}{unsat_suffix}e"
    else:
        # Unsaturated: prefix + stem + unsaturation + ketone
        return f"{prefix}{stem}{unsat_suffix}{ketone_suffix}"
