"""
Name assembly - combining fragments into complete IUPAC names.

Assembly order:
1. Stereodescriptors (R/S, E/Z) at start in parentheses
2. Locanted prefixes (substituents, alphabetized)
3. Parent name (with unsaturation modifiers)
4. Locanted suffix (principal group)
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from ..rules.seniority import get_suffix, get_prefix
from ..rules.locants import get_functional_group_locants, get_bond_locants
from .naming_utils import (
    format_suffix_with_locants,
    get_multiplier_prefix,
    format_substituent_prefix,
    alpha_sort_key,
    get_alkyl_name,
)

# Ion/radical naming imports - deferred to avoid circular imports
# These are imported inside functions that need them

# Complex ring system imports
from ..rules.bicyclo import is_bicyclo_system, name_bicyclo_system
from ..rules.spiro import is_spiro_system, name_spiro_system
from ..rules.fused_rings import classify_fused_system, name_fused_heterocycle, name_ortho_fused_bicyclic

# Partial saturation imports for fused heterocycles
from ..rules.partial_saturation import (
    detect_partial_saturation,
    format_saturation_prefix,
    analyze_saturation_for_naming,
)
from ..data.partial_saturation_refs import get_aromatic_reference, get_reference_smiles


def get_saturation_prefix_for_fused_ring(
    mol,
    aromatic_parent_name: Optional[str] = None,
    atom_to_locant: Optional[Dict[int, Any]] = None
) -> Optional[str]:
    """
    Generate saturation prefix for a fused ring system.

    This is a helper function for the composer that handles the complete
    workflow of detecting partial saturation and formatting the prefix.

    IUPAC 2013 ordering for partially saturated fused heterocycles:
    [substituents]-[saturation prefix]-[indicated H]-[parent]
    Example: 5-methyl-2,3-dihydro-1H-indole

    Args:
        mol: RDKit Mol object
        aromatic_parent_name: Name of the aromatic parent (e.g., 'quinoline')
            If not provided, attempts to auto-detect from molecular structure.
        atom_to_locant: Optional mapping from atom index to IUPAC locant.
            If provided, generates locants in the prefix.

    Returns:
        Formatted saturation prefix string (e.g., '2,3-dihydro', '1,2,3,4-tetrahydro'),
        or None if no saturation detected or parent not found.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        >>> get_saturation_prefix_for_fused_ring(mol, 'quinoline')
        '1,2,3,4-tetrahydro'  # if atom_to_locant provided
    """
    if mol is None:
        return None

    # Try to find aromatic parent if not provided
    aromatic_smiles = None
    if aromatic_parent_name:
        aromatic_smiles = get_reference_smiles(aromatic_parent_name)
    else:
        # Auto-detect aromatic parent
        ref_result = get_aromatic_reference(mol)
        if ref_result:
            _, aromatic_smiles = ref_result

    if not aromatic_smiles:
        return None

    # Use analyze_saturation_for_naming if we have locant mapping
    if atom_to_locant:
        return analyze_saturation_for_naming(mol, aromatic_smiles, atom_to_locant)

    # Otherwise, just detect saturation and return prefix without locants
    result = detect_partial_saturation(mol, aromatic_smiles)
    if result is None:
        return None

    return result['prefix']


def assemble_fused_ring_with_saturation(
    core_name: str,
    saturation_prefix: Optional[str],
    substituent_prefixes: Optional[str] = None,
    indicated_h: Optional[str] = None
) -> str:
    """
    Assemble a fused ring name with saturation prefix in correct IUPAC order.

    IUPAC 2013 ordering rule for partially saturated fused heterocycles:
    [substituents]-[saturation prefix]-[indicated H]-[parent]

    Args:
        core_name: Parent name (e.g., 'indole', 'quinoline')
        saturation_prefix: Saturation prefix (e.g., '2,3-dihydro', 'tetrahydro')
        substituent_prefixes: Optional substituent prefixes (e.g., '5-methyl')
        indicated_h: Optional indicated hydrogen (e.g., '1H')

    Returns:
        Assembled IUPAC name

    Examples:
        >>> assemble_fused_ring_with_saturation('indole', '2,3-dihydro', None, '1H')
        '2,3-dihydro-1H-indole'
        >>> assemble_fused_ring_with_saturation('indole', '2,3-dihydro', '5-methyl', '1H')
        '5-methyl-2,3-dihydro-1H-indole'
    """
    parts = []

    # Add substituent prefixes (alphabetized, with locants)
    if substituent_prefixes:
        parts.append(substituent_prefixes)

    # Add saturation prefix (comes after substituents, before indicated H)
    if saturation_prefix:
        parts.append(saturation_prefix)

    # Add indicated hydrogen (e.g., 1H, 9H)
    if indicated_h:
        parts.append(indicated_h)

    # Add parent name
    parts.append(core_name)

    # Join with hyphens
    return '-'.join(parts)


# Chain length prefixes (IUPAC Blue Book)
CHAIN_PREFIXES = {
    1: "meth", 2: "eth", 3: "prop", 4: "but", 5: "pent",
    6: "hex", 7: "hept", 8: "oct", 9: "non", 10: "dec",
    11: "undec", 12: "dodec", 13: "tridec", 14: "tetradec",
    15: "pentadec", 16: "hexadec", 17: "heptadec", 18: "octadec",
    19: "nonadec", 20: "icos", 21: "henicos", 22: "docos",
    23: "tricos", 24: "tetracos", 25: "pentacos",
    30: "triacont", 40: "tetracont", 50: "pentacont",
    60: "hexacont", 70: "heptacont", 80: "octacont",
    90: "nonacont", 100: "hect",
}

# Simple multiplicative prefixes (for simple substituent names)
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
    11: "undeca", 12: "dodeca",
}

# Complex multiplicative prefixes (for substituents with locants/hyphens)
COMPLEX_MULTIPLIERS = {
    2: "bis", 3: "tris", 4: "tetrakis", 5: "pentakis",
    6: "hexakis", 7: "heptakis", 8: "octakis", 9: "nonakis", 10: "decakis",
}

# Prefixes to IGNORE for alphabetization
IGNORE_FOR_ALPHA = {
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca",
    "undeca", "dodeca",
    "bis", "tris", "tetrakis", "pentakis", "hexakis", "heptakis",
    "octakis", "nonakis", "decakis",
}

# Terminal functional groups that NEVER include locants in the name
# These are always at position 1 by definition (chain numbered from terminal group)
TERMINAL_GROUPS = {
    "carboxylic_acid",  # Always at chain end (locant 1)
    "aldehyde",         # Always at chain end (locant 1)
    "nitrile",          # Always at chain end (locant 1)
    "primary_amide",    # Always at chain end (locant 1)
    "secondary_amide",  # Always at chain end (locant 1)
    "tertiary_amide",   # Always at chain end (locant 1)
}


@dataclass
class NameFragment:
    """A fragment of an IUPAC name."""
    text: str
    locants: tuple = ()
    priority: int = 0
    fragment_type: str = "prefix"  # prefix, parent, suffix, stereo


def assemble_name(features: Any, style: str = "pin") -> str:
    """
    Assemble complete IUPAC name from molecular features.

    Args:
        features: MolecularFeatures object with extracted features
        style: Naming style ("pin", "general", "cas")

    Returns:
        Complete IUPAC name string
    """
    # Check for ionic/radical species first - route to specialized assembly
    species_type = getattr(features, 'species_type', 'neutral')
    if species_type in ('salt', 'ion', 'zwitterion', 'radical'):
        return assemble_ion_name(features, features.mol, style)

    # Handle polyfunctional compounds (multiple distinct functional groups)
    if getattr(features, 'is_polyfunctional', False):
        from ..rules.polyfunctional import name_polyfunctional
        poly_name = name_polyfunctional(features)
        if poly_name:
            return poly_name
        # If name_polyfunctional returns None, fall through to normal handling

    # Handle esters (two-component naming: "alkyl alkanoate")
    if features.principal_group == "ester":
        ester_match = getattr(features, 'ester_match', None)
        if ester_match:
            from ..rules.esters import name_ester
            ester_name = name_ester(features.mol, ester_match)
            if ester_name:
                return ester_name
        # If name_ester returns None (lactone or complex), fall through

    # Handle sulfur functional class compounds (sulfide, sulfoxide, sulfone)
    # These use functional class naming, not suffix-based
    if features.principal_group in ('sulfoxide', 'sulfone'):
        from ..rules.sulfur import name_sulfoxide, name_sulfone
        if features.principal_group == 'sulfoxide':
            matches = features.functional_groups.get('sulfoxide', [])
            if matches:
                name = name_sulfoxide(features.mol, matches[0])
                if name:
                    return name
        elif features.principal_group == 'sulfone':
            matches = features.functional_groups.get('sulfone', [])
            if matches:
                name = name_sulfone(features.mol, matches[0])
                if name:
                    return name

    # Handle thioethers (sulfides) - also use functional class
    if features.principal_group == 'thioether':
        from ..rules.sulfur import name_sulfide
        # Find sulfur atom index
        matches = features.functional_groups.get('thioether', [])
        if matches:
            # SMARTS match gives (S, C, C) - sulfur is first
            sulfur_idx = matches[0][0]
            name = name_sulfide(features.mol, sulfur_idx)
            if name:
                return name

    # Handle phosphorus functional class compounds (phosphine oxide, phosphates, phosphines)
    # These use functional class or substitutive naming, not suffix-based
    if features.principal_group == 'phosphine_oxide':
        from ..rules.phosphorus import name_phosphine_oxide
        matches = features.functional_groups.get('phosphine_oxide', [])
        if matches:
            name = name_phosphine_oxide(features.mol, matches[0])
            if name:
                return name

    # Handle phosphate esters
    if features.principal_group in ('phosphate_triester', 'phosphate_diester', 'phosphate_monoester'):
        from ..rules.phosphorus import name_phosphate_ester
        fg_key = features.principal_group
        matches = features.functional_groups.get(fg_key, [])
        if matches:
            # Find phosphorus atom index from SMARTS match
            for idx in matches[0]:
                atom = features.mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'P':
                    name = name_phosphate_ester(features.mol, idx)
                    if name:
                        return name
                    break

    # Handle phosphines (tertiary, secondary, primary)
    if features.principal_group in ('tertiary_phosphine', 'secondary_phosphine', 'primary_phosphine'):
        from ..rules.phosphorus import name_phosphine
        fg_key = features.principal_group
        matches = features.functional_groups.get(fg_key, [])
        if matches:
            # Find phosphorus atom index
            for idx in matches[0]:
                atom = features.mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'P':
                    name = name_phosphine(features.mol, idx)
                    if name:
                        return name
                    break

    # Handle phosphinic acid (suffix naming, but needs special assembly)
    if features.principal_group == 'phosphinic_acid':
        from ..rules.phosphorus import name_phosphinic_acid
        matches = features.functional_groups.get('phosphinic_acid', [])
        if matches:
            name = name_phosphinic_acid(features.mol, matches[0])
            if name:
                return name

    # Handle complex ring systems FIRST (bicyclo, spiro, fused heterocycles)
    # These take precedence over simple heterocyclic/benzene classification
    # because fused heterocycles (indole, purine) contain benzene/heterocycle parts
    # that would otherwise trigger early exit to wrong naming path
    if features.is_cyclic and _is_complex_ring_system(features.mol):
        complex_name = _assemble_complex_ring_name(features.mol, features)
        if complex_name:
            return complex_name
        # If complex ring naming fails, fall through to simpler handling

    # Handle polycyclic aromatics (naphthalene, anthracene, etc.)
    # Check before benzene since substituted PAHs have benzene substructures
    polycyclic_name = getattr(features, 'polycyclic_name', None)
    if polycyclic_name:
        return _assemble_polycyclic_name(features, style)

    # Handle partially saturated carbocycles (tetrahydronaphthalene, etc.)
    # Check BEFORE benzene since they contain benzene substructure
    if features.is_cyclic:
        partial_sat_name = _try_partially_saturated_carbocycle(features.mol)
        if partial_sat_name:
            return partial_sat_name

    # Handle simple heterocyclic compounds (pyridine, morpholine, etc.)
    # Only reached if not a complex fused system
    ring_type = getattr(features, 'ring_type', None)
    if ring_type == 'heterocyclic':
        return _assemble_heterocycle_name(features, style)

    # Handle benzene derivatives
    # Only reached if not a fused system containing benzene
    if getattr(features, 'is_benzene', False):
        return _assemble_benzene_name(features, style)

    # Handle ring-attached nitriles (cyclohexanecarbonitrile, etc.)
    if features.principal_group == 'nitrile' and features.is_cyclic:
        return _assemble_ring_nitrile_name(features, style)

    # Handle amides (including N-substituted amides)
    if features.principal_group in ('primary_amide', 'secondary_amide', 'tertiary_amide'):
        return _assemble_amide_name(features, style)

    # Handle simple cases
    if not features.principal_chain and not features.ring_systems:
        # Single atom or very simple molecule
        return _name_simple_molecule(features)

    fragments = []

    # Generate parent name (chain or ring)
    if features.principal_chain:
        parent = _generate_chain_parent(features)
    elif features.ring_systems:
        parent = _generate_ring_parent(features)
    else:
        parent = NameFragment(text="", fragment_type="parent")

    fragments.append(parent)

    # Generate suffix for principal group
    if features.principal_group:
        suffix = _generate_suffix(features)
        if suffix:
            fragments.append(suffix)

    # Generate prefixes for substituents and non-principal groups
    prefixes = _generate_prefixes(features)
    fragments.extend(prefixes)

    # Generate stereodescriptors (for R/S stereocenters and E/Z double bonds)
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)

    # Assemble in correct order
    return _assemble_fragments(fragments, style)


def _name_simple_molecule(features: Any) -> str:
    """Name very simple molecules (single atom, etc.)."""
    mol = features.mol

    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        symbol = atom.GetSymbol()
        # Single atom molecules
        if symbol == 'C':
            return "methane"  # CH4
        # Add more as needed

    return "unknown"


def _assemble_benzene_name(features: Any, style: str) -> str:
    """
    Assemble name for benzene derivatives.

    Benzene naming is handled specially because:
    1. Substituents are named relative to the ring, not a chain
    2. Ring orientation determines locants (not chain direction)
    3. The parent is always "benzene"

    Args:
        features: MolecularFeatures with is_benzene=True
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the benzene derivative
    """
    from ..rules.benzene import orient_benzene, name_substituted_benzene

    mol = features.mol
    ring_atoms = features.benzene_ring
    substituents = features.benzene_substituents

    # If no substituents, return "benzene" (should be caught by retained names,
    # but handle here as fallback)
    if not substituents:
        return "benzene"

    # Orient the ring for lowest locants
    oriented_ring = orient_benzene(mol, ring_atoms, substituents)

    # Generate systematic name
    return name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents)


def _assemble_polycyclic_name(features: Any, style: str) -> str:
    """
    Assemble name for polycyclic aromatic hydrocarbons.

    Polycyclic naming is handled specially because:
    1. Substituent numbering is FIXED by IUPAC (not lowest locants)
    2. Parent is a retained name (naphthalene, anthracene, etc.)
    3. Multiple fused rings have standard numbering

    Args:
        features: MolecularFeatures with polycyclic_name set
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the polycyclic aromatic
    """
    from ..rules.polycyclics import name_substituted_polycyclic

    pah_name = features.polycyclic_name
    substituents = features.polycyclic_substituents

    # If no substituents, return the PAH name
    # (This should be caught by retained names, but handle here as fallback)
    if not substituents:
        return pah_name

    # Generate systematic name with substituents
    return name_substituted_polycyclic(features.mol, pah_name, substituents)


def _try_partially_saturated_carbocycle(mol) -> Optional[str]:
    """
    Try to name a molecule as a partially saturated carbocycle.

    Checks if the molecule is a partially saturated PAH (like tetrahydronaphthalene)
    and returns the IUPAC name if so.

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name if partially saturated carbocycle detected, None otherwise

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        >>> _try_partially_saturated_carbocycle(mol)
        '1,2,3,4-tetrahydronaphthalene'
    """
    from ..rules.polycyclics import name_partially_saturated_carbocycle

    return name_partially_saturated_carbocycle(mol)


def _is_complex_ring_system(mol) -> bool:
    """
    Check if a molecule contains a complex ring system (bicyclo, spiro, or fused).

    Complex ring systems require special naming rules beyond simple cycloalkanes.
    This function is used for early routing decision in assemble_name().

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule contains bicyclo, spiro, or non-trivial fused ring system

    Note:
        Simple monocyclic rings return False.
        Aromatic fused systems (naphthalene, etc.) are handled by polycyclics module.
    """
    # Check bicyclo first (bridged bicyclic)
    if is_bicyclo_system(mol):
        return True

    # Check spiro (two rings sharing one atom)
    if is_spiro_system(mol):
        return True

    # Check fused rings (ortho-fused or ortho-peri-fused)
    fused_type = classify_fused_system(mol)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return True

    return False


def _classify_complex_ring(mol) -> str:
    """
    Classify a complex ring system by type.

    Args:
        mol: RDKit Mol object

    Returns:
        Classification string:
        - 'bicyclo': Bridged bicyclic system
        - 'spiro': Spiro system (rings share one atom)
        - 'ortho-fused': Ortho-fused system (rings share one edge)
        - 'ortho-peri-fused': Complex fused system
        - 'simple': Not a complex ring system
    """
    # Check in order of priority
    if is_bicyclo_system(mol):
        return 'bicyclo'

    if is_spiro_system(mol):
        return 'spiro'

    fused_type = classify_fused_system(mol)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return fused_type

    return 'simple'


def _assemble_complex_ring_name(mol, features) -> Optional[str]:
    """
    Assemble IUPAC name for a complex ring system.

    Routes to appropriate naming function based on ring classification:
    - Bicyclo: bicyclo[x.y.z]alkane format (e.g., bicyclo[2.2.1]heptane)
    - Spiro: spiro[a.b]alkane format (e.g., spiro[4.5]decane)
    - Fused: retained names or systematic fusion descriptors

    Args:
        mol: RDKit Mol object
        features: MolecularFeatures object (for substituents, stereo, etc.)

    Returns:
        Complete IUPAC name, or None if naming fails

    Note:
        Supports complete bicyclo naming with substituents, unsaturation, and stereo.
    """
    import logging

    ring_type = _classify_complex_ring(mol)

    try:
        if ring_type == 'bicyclo':
            # Complete bicyclo naming with substituents, unsaturation, stereo
            name = _assemble_complete_bicyclo_name(mol, features)
            if name:
                return name
            logging.warning(f"Bicyclo naming failed for molecule")
            return None

        elif ring_type == 'spiro':
            # Spiro naming (spiro[4.5]decane, etc.)
            name = name_spiro_system(mol)
            if name:
                return name
            logging.warning(f"Spiro naming failed for molecule")
            return None

        elif ring_type in ('ortho-fused', 'ortho-peri-fused'):
            # Fused ring naming - try heterocycle first, then carbocyclic
            # Heterocycles have retained names like indole, quinoline
            name = name_fused_heterocycle(mol)
            if name:
                return name

            # Try ortho-fused bicyclic (carbocyclic fallback)
            name = name_ortho_fused_bicyclic(mol)
            if name:
                return name

            logging.warning(f"Fused ring naming failed for {ring_type} system")
            return None

        else:
            # Not a complex ring - shouldn't reach here
            return None

    except Exception as e:
        # Graceful degradation - log and return None
        logging.warning(f"Complex ring naming error: {e}")
        return None


def _assemble_complete_bicyclo_name(mol, features) -> Optional[str]:
    """
    Assemble complete IUPAC name for a bicyclo system with substituents, unsaturation, and stereo.

    IUPAC name order for bicyclo compounds:
    (stereo)-locants-prefixes-bicyclo[x.y.z]parent-locants-unsaturation

    Example: (1S,6R)-3,7,7-trimethylbicyclo[4.1.0]hept-3-ene

    Args:
        mol: RDKit Mol object
        features: MolecularFeatures object

    Returns:
        Complete IUPAC name, or None if naming fails
    """
    from ..rules.bicyclo import get_complete_bicyclo_data, name_bicyclo_system
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from rdkit.Chem import rdCIPLabeler

    # Get complete bicyclo data
    bicyclo_data = get_complete_bicyclo_data(mol)
    if not bicyclo_data:
        # Fall back to simple naming
        return name_bicyclo_system(mol)

    # If there's a retained name and no substituents or unsaturation, use it
    retained_name = bicyclo_data.get('retained_name')
    substituents = bicyclo_data.get('substituents', {})
    unsaturation = bicyclo_data.get('unsaturation', {})
    double_bonds = unsaturation.get('double_bonds', [])
    triple_bonds = unsaturation.get('triple_bonds', [])

    # Check if there are any modifiers (substituents or unsaturation)
    has_substituents = bool(substituents)
    has_unsaturation = bool(double_bonds) or bool(triple_bonds)

    # For unsubstituted, saturated compounds with retained names, use the retained name
    if retained_name and not has_substituents and not has_unsaturation:
        return retained_name

    # Get key data
    descriptor = bicyclo_data.get('descriptor')
    atom_to_locant = bicyclo_data.get('atom_to_locant', {})
    carbon_count = bicyclo_data.get('carbon_count', 0)

    # Get parent stem
    parent_stem = _get_bicyclo_parent_stem(carbon_count)

    # Build unsaturation suffix
    unsat_suffix = _build_bicyclo_unsaturation_suffix(
        unsaturation, atom_to_locant, parent_stem
    )

    # Build substituent prefix
    sub_prefix = _build_bicyclo_substituent_prefix(mol, substituents, atom_to_locant)

    # Collect stereodescriptors
    stereo_prefix = ""
    rdCIPLabeler.AssignCIPLabels(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)

    # Assemble the name
    # Format: (stereo)-substituent-prefix-descriptor-parent-unsaturation
    parts = []

    if stereo_prefix:
        parts.append(stereo_prefix)

    if sub_prefix:
        parts.append(sub_prefix)

    # Build the main name: bicyclo[x.y.z]parent-unsat
    main_name = f"{descriptor}{unsat_suffix}"
    parts.append(main_name)

    # Join parts
    name = "".join(parts)

    return name


def _get_bicyclo_parent_stem(carbon_count: int) -> str:
    """Get the parent stem for a bicyclo system based on carbon count."""
    stems = {
        4: "but", 5: "pent", 6: "hex", 7: "hept", 8: "oct",
        9: "non", 10: "dec", 11: "undec", 12: "dodec",
    }
    return stems.get(carbon_count, f"C{carbon_count}")


def _build_bicyclo_unsaturation_suffix(
    unsaturation: Dict,
    atom_to_locant: Dict[int, int],
    parent_stem: str
) -> str:
    """
    Build parent name with unsaturation suffix for bicyclo.

    Examples:
        'hept' + no unsaturation -> 'heptane'
        'hept' + double bond at 3 -> 'hept-3-ene'
        'hept' + double bonds at 2,4 -> 'hepta-2,4-diene'

    Args:
        unsaturation: Dict with 'double_bonds' and 'triple_bonds' lists
        atom_to_locant: Mapping from atom index to IUPAC locant
        parent_stem: Parent stem (e.g., 'hept')

    Returns:
        Parent name with unsaturation suffix (e.g., 'heptane', 'hept-3-ene')
    """
    double_bonds = unsaturation.get('double_bonds', [])
    triple_bonds = unsaturation.get('triple_bonds', [])

    # Convert atom indices to locants and get the lower locant for each bond
    double_locants = []
    for bond in double_bonds:
        idx1, idx2 = bond
        loc1 = atom_to_locant.get(idx1, 0)
        loc2 = atom_to_locant.get(idx2, 0)
        double_locants.append(min(loc1, loc2))

    triple_locants = []
    for bond in triple_bonds:
        idx1, idx2 = bond
        loc1 = atom_to_locant.get(idx1, 0)
        loc2 = atom_to_locant.get(idx2, 0)
        triple_locants.append(min(loc1, loc2))

    # Sort locants
    double_locants.sort()
    triple_locants.sort()

    num_double = len(double_locants)
    num_triple = len(triple_locants)

    # Saturated case
    if num_double == 0 and num_triple == 0:
        return f"{parent_stem}ane"

    parts = [parent_stem]

    # Build unsaturation part
    if num_double > 0:
        double_str = ",".join(str(loc) for loc in double_locants)
        if num_double == 1:
            parts.append(f"-{double_str}-en")
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            parts.append(f"a-{double_str}-{multiplier}en")

    if num_triple > 0:
        triple_str = ",".join(str(loc) for loc in triple_locants)
        if num_triple == 1:
            parts.append(f"-{triple_str}-yn")
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            if num_double == 0:
                parts.append(f"a-{triple_str}-{multiplier}yn")
            else:
                parts.append(f"-{triple_str}-{multiplier}yn")

    # Add final 'e'
    return "".join(parts) + "e"


def _build_bicyclo_substituent_prefix(
    mol,
    substituents: Dict[int, List[Dict]],
    atom_to_locant: Dict[int, int]
) -> str:
    """
    Build substituent prefix string for bicyclo naming.

    Groups substituents by name, adds locants and multipliers.

    Args:
        mol: RDKit Mol object
        substituents: Dict from get_bicyclo_substituents
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Formatted prefix string (e.g., '3,7,7-trimethyl-')
    """
    if not substituents:
        return ""

    # Group substituents by name
    sub_groups: Dict[str, List[int]] = defaultdict(list)

    for ring_atom_idx, sub_list in substituents.items():
        locant = atom_to_locant.get(ring_atom_idx)
        if locant is None:
            continue

        for sub_info in sub_list:
            carbon_count = sub_info.get('carbon_count', 0)
            if carbon_count == 0:
                continue

            # Get alkyl name
            alkyl_name = get_alkyl_name(carbon_count)
            sub_groups[alkyl_name].append(locant)

    if not sub_groups:
        return ""

    # Build prefix fragments
    prefixes = []
    for name, locants in sub_groups.items():
        count = len(locants)
        sorted_locants = sorted(locants)

        formatted = format_substituent_prefix(name, sorted_locants, count)
        prefixes.append((name, formatted))

    # Sort alphabetically by base name (ignoring multipliers)
    prefixes.sort(key=lambda x: alpha_sort_key(x[1]))

    # Join with hyphens
    prefix_texts = [p[1] for p in prefixes]
    result = _join_prefixes(prefix_texts)

    return result


def _assemble_heterocycle_name(features: Any, style: str) -> str:
    """
    Assemble name for heterocyclic compounds.

    For unsubstituted heterocycles, returns the parent name directly
    (either retained name or HW systematic name).

    For substituted heterocycles, adds prefixes with locants:
    - N-substitution uses N-locant format (N-methyl, N,N-dimethyl)
    - C-substitution uses numeric locants (2-methyl, 3-ethyl)

    Args:
        features: MolecularFeatures with ring_type='heterocyclic'
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the heterocycle
    """
    from ..rules.heterocycles import name_heterocycle, name_substituted_heterocycle

    # Get the heterocycle parent name
    parent_name = name_heterocycle(features.mol, features.principal_ring)

    # Check for substituents
    substituents = getattr(features, 'heterocycle_substituents', None)
    atom_to_locant = getattr(features, 'heterocycle_atom_to_locant', None)

    if substituents and atom_to_locant:
        # Generate substituted name with N-locants and C-locants
        return name_substituted_heterocycle(
            features.mol,
            features.principal_ring,
            parent_name,
            substituents,
            atom_to_locant
        )

    # No substituents - return parent name directly
    return parent_name


def _assemble_amide_name(features: Any, style: str) -> str:
    """
    Assemble name for amide compounds with N-substitution handling.

    Handles:
    - Primary amides: acetamide, propanamide
    - Secondary amides: N-methylacetamide
    - Tertiary amides: N,N-dimethylformamide
    - Ring-attached amides: cyclohexanecarboxamide

    Args:
        features: MolecularFeatures with amide principal group
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the amide
    """
    from ..rules.amides import name_amide

    # Get amide atoms
    amide_atoms = None
    if features.principal_group_atoms:
        amide_atoms = features.principal_group_atoms[0]

    if amide_atoms:
        return name_amide(features.mol, amide_atoms)

    # Fallback
    return "amide"


def _assemble_ring_nitrile_name(features: Any, style: str) -> str:
    """
    Assemble name for ring-attached nitriles (use -carbonitrile suffix).

    Ring-attached nitriles use the carbonitrile suffix, e.g.:
    - cyclohexanecarbonitrile
    - cyclopentanecarbonitrile

    Args:
        features: MolecularFeatures with principal_group='nitrile' and is_cyclic=True
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the ring nitrile
    """
    from ..rules.nitriles import name_nitrile

    principal_ring = getattr(features, 'principal_ring', None)
    if not principal_ring:
        return "carbonitrile"

    ring_size = len(principal_ring)

    if ring_size in CHAIN_PREFIXES:
        stem = CHAIN_PREFIXES[ring_size]
    else:
        stem = f"{ring_size}C"

    parent_name = f"cyclo{stem}ane"

    # Get nitrile atoms
    nitrile_atoms = None
    if features.principal_group_atoms:
        nitrile_atoms = features.principal_group_atoms[0]

    if nitrile_atoms:
        return name_nitrile(features.mol, nitrile_atoms, parent_name=parent_name, is_ring=True)

    # Fallback: just append carbonitrile
    return f"{parent_name}carbonitrile"


def _generate_chain_parent(features: Any) -> NameFragment:
    """Generate parent name for acyclic chains.

    Returns a NameFragment where:
    - text: stem + unsaturation_base (e.g., "but", "prop")
    - locants: tuple of (double_bond_locants, triple_bond_locants)
    """
    chain_length = len(features.principal_chain)

    # Get chain prefix (stem)
    if chain_length in CHAIN_PREFIXES:
        stem = CHAIN_PREFIXES[chain_length]
    else:
        stem = _build_long_chain_prefix(chain_length)

    # Get bond locants if we have atom_to_locant mapping
    double_locants = []
    triple_locants = []
    if features.atom_to_locant:
        double_locants = get_bond_locants(
            features.principal_chain,
            features.double_bonds,
            features.atom_to_locant
        )
        triple_locants = get_bond_locants(
            features.principal_chain,
            features.triple_bonds,
            features.atom_to_locant
        )

    # Store stem as text, bond locants as locants tuple
    # We'll use a nested tuple: ((double_locants), (triple_locants))
    return NameFragment(
        text=stem,
        locants=(tuple(double_locants), tuple(triple_locants)),
        fragment_type="parent"
    )


def _generate_ring_parent(features: Any) -> NameFragment:
    """
    Generate parent name for cyclic compounds.

    For cycloalkanes: "cyclo" + chain prefix + "an" (e.g., "cyclohexan")
    The final 'e' is added during assembly if no suffix follows.

    For cycloalkenes: "cyclo" + chain prefix + double bond info
    - Mono-cycloalkenes: cyclo + stem + "en" (cyclohexene) - no locant
    - Cycloalkadienes: cyclo + stem + "a" + locants + "dien" (cyclohexa-1,3-diene)

    For other ring types, returns placeholder for now (to be implemented
    in subsequent plans).

    Args:
        features: MolecularFeatures object with ring_type and principal_ring

    Returns:
        NameFragment with parent text and bond locants
    """
    ring_type = getattr(features, 'ring_type', None)
    principal_ring = getattr(features, 'principal_ring', None)

    if not principal_ring:
        # Fallback: no ring identified
        return NameFragment(text="cyclo", fragment_type="parent")

    ring_size = len(principal_ring)

    if ring_size in CHAIN_PREFIXES:
        stem = CHAIN_PREFIXES[ring_size]
    else:
        stem = _build_long_chain_prefix(ring_size)

    if ring_type == 'cycloalkane':
        # Cycloalkane naming: cyclo + stem + an (e.g., cyclohexan)
        # Return stem - the "ane" will be added in assembly
        return NameFragment(
            text=f"cyclo{stem}",
            locants=((), ()),  # No bond locants for saturated rings
            fragment_type="parent"
        )

    elif ring_type == 'cycloalkene':
        # Get double bond locants from features
        ring_double_bond_locants = getattr(features, 'ring_double_bond_locants', [])

        return NameFragment(
            text=f"cyclo{stem}",
            locants=(tuple(ring_double_bond_locants), ()),  # (double_bond_locants, triple_bond_locants)
            fragment_type="parent"
        )

    elif ring_type == 'aromatic':
        # TODO: Implement aromatic naming (Plan 02-04)
        return NameFragment(text="cyclo", fragment_type="parent")

    elif ring_type == 'heterocyclic':
        # TODO: Implement heterocyclic naming (Phase 3)
        return NameFragment(text="cyclo", fragment_type="parent")

    # Unknown ring type, return placeholder
    return NameFragment(text="cyclo", fragment_type="parent")


def _get_unsaturation_suffix(features: Any) -> str:
    """
    Get the unsaturation suffix (an, en, yn, etc.).
    
    Returns suffix without the 'e' ending (that's added later based on what follows).
    """
    num_double = len(features.double_bonds)
    num_triple = len(features.triple_bonds)
    
    if num_double == 0 and num_triple == 0:
        return "an"  # Saturated
    
    parts = []
    
    # Double bonds
    if num_double == 1:
        parts.append("en")
    elif num_double > 1:
        multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
        parts.append(f"{multiplier}en")  # dien, trien, etc.
    
    # Triple bonds
    if num_triple == 1:
        parts.append("yn")
    elif num_triple > 1:
        multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
        parts.append(f"{multiplier}yn")
    
    if not parts:
        return "an"
    
    # For mixed unsaturation, use "a" connector
    if num_double > 0 and num_triple > 0:
        return "a" + "".join(parts)  # e.g., "adienyn"
    
    return "".join(parts)


def _generate_suffix(features: Any) -> Optional[NameFragment]:
    """Generate suffix fragment for principal group.

    Handles locant assignment for the principal functional group:
    - Terminal groups (carboxylic acid, aldehyde): locant always 1, omitted from name
    - Non-terminal groups (alcohol, ketone): locant MUST be included in PIN style
    - Multiple instances: include multiplier (di, tri) and all locants
    """
    fg_name = features.principal_group
    if not fg_name:
        return None

    is_ring = features.is_cyclic
    suffix_text = get_suffix(fg_name, is_ring=is_ring)

    if not suffix_text:
        return None

    # Get locants for functional group positions on the chain
    locants = ()
    if features.principal_chain and features.atom_to_locant and features.principal_group_atoms:
        fg_locants = get_functional_group_locants(
            features.principal_chain,
            features.principal_group_atoms,
            features.atom_to_locant,
            mol=features.mol
        )

        # Terminal groups: locant is implicitly 1, do NOT include in name
        if fg_name in TERMINAL_GROUPS:
            # For terminal groups, we don't include the locant
            locants = ()
        else:
            # For non-terminal groups (alcohol, ketone), include locants
            locants = tuple(fg_locants)

    return NameFragment(
        text=suffix_text,
        locants=locants,
        fragment_type="suffix"
    )


def _generate_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for alkyl substituents and non-principal groups.

    For alkane/cycloalkane naming, this extracts alkyl substituents from
    features.substituents (for chains) or features.ring_substituents (for rings),
    groups them by name (methyl, ethyl, etc.), and formats with locants and
    multiplicative prefixes.

    When chain_is_parent=True (ring-chain compounds where chain won parent selection),
    rings become substituents and are named as prefixes (phenyl, cyclohexyl, etc.).
    """
    prefixes = []

    # --- Handle ring-as-substituent prefixes when chain is parent ---
    if getattr(features, 'chain_is_parent', False):
        ring_sub_prefixes = _generate_ring_substituent_prefixes(features)
        prefixes.extend(ring_sub_prefixes)

    # --- Handle alkyl substituents from features.substituents (chains) ---
    if features.substituents and features.mol:
        alkyl_prefixes = _generate_alkyl_prefixes(features)
        prefixes.extend(alkyl_prefixes)

    # --- Handle ring substituents from features.ring_substituents ---
    ring_substituents = getattr(features, 'ring_substituents', None)
    oriented_ring = getattr(features, 'oriented_ring', None)
    if ring_substituents and features.mol and oriented_ring:
        ring_prefixes = _generate_ring_alkyl_prefixes(features)
        prefixes.extend(ring_prefixes)

    # --- Handle non-principal functional groups as prefixes ---
    for fg_name, matches in features.functional_groups.items():
        if fg_name == features.principal_group:
            continue

        prefix_text = get_prefix(fg_name)
        if prefix_text and matches:
            count = len(matches)

            if count > 1:
                # Add multiplier
                multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
                prefix_text = f"{multiplier}{prefix_text}"

            prefixes.append(NameFragment(
                text=prefix_text,
                locants=(),  # TODO: Add locants
                fragment_type="prefix"
            ))

    return prefixes


def _generate_ring_substituent_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for rings that are substituents on a chain parent.

    When parent selection determines chain is parent (chain_is_parent=True),
    rings become substituents and need to be named as prefixes (phenyl, cyclohexyl, etc.).

    This implements IUPAC P-61.5 ring-as-substituent naming:
    - benzene -> phenyl
    - cyclohexane -> cyclohexyl
    - pyridine -> pyridyl
    - etc.

    Args:
        features: MolecularFeatures with ring_substituents_as_groups populated

    Returns:
        List of NameFragment objects for ring substituent prefixes
    """
    from ..rules.ring_substituents import get_ring_substituent_name, get_ring_attachment_locant

    prefixes = []
    ring_groups = getattr(features, 'ring_substituents_as_groups', [])

    if not ring_groups:
        return prefixes

    # Group ring substituents by name for multiplier handling
    ring_sub_groups: Dict[str, List[int]] = defaultdict(list)

    for ring_atoms in ring_groups:
        # Get substituent name (phenyl, cyclohexyl, etc.)
        sub_name = get_ring_substituent_name(features.mol, ring_atoms)

        # Find which chain position the ring attaches to
        try:
            locant = get_ring_attachment_locant(
                features.mol,
                ring_atoms,
                features.principal_chain,
                features.atom_to_locant
            )
            ring_sub_groups[sub_name].append(locant)
        except ValueError:
            # If we can't find the attachment, skip this ring
            continue

    # Build prefix fragments
    for name, locants in ring_sub_groups.items():
        count = len(locants)
        sorted_locants = sorted(locants)

        formatted = format_substituent_prefix(name, sorted_locants, count)

        prefixes.append(NameFragment(
            text=formatted,
            locants=tuple(sorted_locants),
            fragment_type="prefix"
        ))

    # Sort by IUPAC alphabetization rules (ignoring di-, tri-, etc.)
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    return prefixes


def _generate_alkyl_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for alkyl substituents and alkoxy groups.

    This function:
    1. Iterates over features.substituents (keyed by 1-indexed position/locant)
    2. Counts carbon atoms in each substituent to determine name
    3. Detects ether groups (O bonded to 2 carbons) and names as alkoxy
    4. Groups identical substituents with their locants
    5. Formats each group with locants, multipliers
    6. Sorts alphabetically by base substituent name

    Note: When chain_is_parent=True, ring substituents are handled separately
    by _generate_ring_substituent_prefixes() and should be skipped here.

    Returns:
        List of NameFragment objects for alkyl prefixes, sorted alphabetically
    """
    mol = features.mol

    # Collect ring atoms that should be skipped (handled by ring substituent prefixes)
    ring_atoms_to_skip: set = set()
    if getattr(features, 'chain_is_parent', False):
        ring_groups = getattr(features, 'ring_substituents_as_groups', [])
        for ring_atoms in ring_groups:
            ring_atoms_to_skip.update(ring_atoms)

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for position, sub_list in features.substituents.items():
        # position is already a 1-indexed locant (from get_substituents)
        for sub_atoms in sub_list:
            # Skip substituents that are ring atoms (handled separately)
            if ring_atoms_to_skip and set(sub_atoms) & ring_atoms_to_skip:
                continue

            # Count only carbon atoms in the substituent
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Skip non-alkyl substituents (no carbons = functional group like -OH)
            if carbon_count == 0:
                continue

            # Check if substituent has any heteroatoms (non-C, non-H)
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            if has_heteroatom:
                # Check for ether pattern: O bonded to 2 carbons
                alkoxy_name = _check_for_alkoxy(mol, sub_atoms, features.principal_chain)
                if alkoxy_name:
                    substituent_groups[alkoxy_name].append(position)
                # Skip other complex substituents for now
                continue

            # Get alkyl name from carbon count
            try:
                alkyl_name = get_alkyl_name(carbon_count)
                substituent_groups[alkyl_name].append(position)
            except ValueError:
                # Carbon count > 10, skip for now (complex substituent)
                continue

    # Count total number of substituents for locant omission decision
    total_substituents = sum(len(locs) for locs in substituent_groups.values())

    # Check if this is a simple hydrocarbon (no functional group suffix)
    # In that case, monosubstituted at position 1 omits the locant
    is_simple_hydrocarbon = features.principal_group is None

    # Build prefix fragments
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)
        sorted_locants = sorted(locants)

        # Omit locant for monosubstituted hydrocarbons at position 1
        if is_simple_hydrocarbon and total_substituents == 1 and sorted_locants == [1]:
            # Just the name, no locant: "methoxy" not "1-methoxy"
            formatted = name
        else:
            formatted = format_substituent_prefix(name, sorted_locants, count)

        prefixes.append(NameFragment(
            text=formatted,
            locants=tuple(sorted_locants),
            fragment_type="prefix"
        ))

    # Sort by IUPAC alphabetization rules (ignoring di-, tri-, etc.)
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    return prefixes


def _check_for_alkoxy(mol, sub_atoms: List[int], principal_chain: List[int]) -> Optional[str]:
    """
    Check if a substituent is an alkoxy group (ether attached to chain).

    An alkoxy group is: -O-alkyl where the O is bonded to the chain carbon
    and to an alkyl group.

    Args:
        mol: RDKit Mol object
        sub_atoms: Atom indices in the substituent
        principal_chain: Atom indices of the principal chain

    Returns:
        Alkoxy name (e.g., "methoxy", "ethoxy") or None if not an alkoxy
    """
    chain_set = set(principal_chain)

    # Find oxygen atom in the substituent
    oxygen_idx = None
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'O':
            # Check if it's bonded to exactly 2 atoms (typical ether O)
            if atom.GetDegree() == 2:
                oxygen_idx = idx
                break

    if oxygen_idx is None:
        return None

    oxygen = mol.GetAtomWithIdx(oxygen_idx)

    # Find the alkyl carbon attached to oxygen (not on principal chain)
    alkyl_start = None
    for neighbor in oxygen.GetNeighbors():
        nbr_idx = neighbor.GetIdx()
        if nbr_idx not in chain_set and neighbor.GetSymbol() == 'C':
            alkyl_start = nbr_idx
            break

    if alkyl_start is None:
        return None

    # Count carbons in the alkyl part (excluding the oxygen)
    carbon_count = _count_alkyl_carbons(mol, alkyl_start, {oxygen_idx})

    # Get alkoxy name
    ALKOXY_NAMES = {
        1: "methoxy",
        2: "ethoxy",
        3: "propoxy",
        4: "butoxy",
        5: "pentyloxy",
        6: "hexyloxy",
        7: "heptyloxy",
        8: "octyloxy",
        9: "nonyloxy",
        10: "decyloxy",
    }

    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]
    elif carbon_count > 10:
        return f"{carbon_count}Coxy"  # Fallback for large groups
    return None


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in an alkyl group via BFS."""
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def _generate_ring_alkyl_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for alkyl substituents on rings.

    This function:
    1. Iterates over features.ring_substituents (keyed by ring atom index)
    2. Uses features.oriented_ring to determine locants
    3. Counts carbon atoms in each substituent to determine name
    4. Groups identical substituents with their locants
    5. Formats each group with locants, multipliers
    6. Sorts alphabetically by base substituent name

    IUPAC rules for cycloalkane substituent locants:
    - Monosubstituted cycloalkanes: locant 1 is implicit and omitted
      (methylcyclohexane, not 1-methylcyclohexane)
    - Polysubstituted cycloalkanes: all locants included
      (1,2-dimethylcyclohexane)

    Returns:
        List of NameFragment objects for alkyl prefixes, sorted alphabetically
    """
    mol = features.mol
    ring_substituents = features.ring_substituents
    oriented_ring = features.oriented_ring

    # Build atom-to-locant mapping from oriented ring
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(oriented_ring)}

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for ring_atom_idx, sub_list in ring_substituents.items():
        # Get locant for this ring position
        locant = atom_to_locant.get(ring_atom_idx)
        if locant is None:
            continue

        for sub_atoms in sub_list:
            # Count only carbon atoms in the substituent
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Skip non-alkyl substituents (no carbons = functional group like -OH)
            if carbon_count == 0:
                continue

            # Check if substituent has any heteroatoms (non-C, non-H)
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            # For Phase 2, skip mixed substituents (carbon + heteroatom)
            # These are complex substituents handled in later phases
            if has_heteroatom:
                continue

            # Get alkyl name from carbon count
            try:
                alkyl_name = get_alkyl_name(carbon_count)
                substituent_groups[alkyl_name].append(locant)
            except ValueError:
                # Carbon count > 10, skip for now (complex substituent)
                continue

    # Count total number of substituents to determine if monosubstituted
    total_substituents = sum(len(locs) for locs in substituent_groups.values())
    is_monosubstituted = total_substituents == 1

    # Build prefix fragments
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)

        if is_monosubstituted:
            # Monosubstituted: omit locant (it's always 1)
            # Just the substituent name: "methyl" not "1-methyl"
            formatted = name
        else:
            # Polysubstituted: include locants
            formatted = format_substituent_prefix(name, sorted(locants), count)

        prefixes.append(NameFragment(
            text=formatted,
            locants=tuple(sorted(locants)),
            fragment_type="prefix"
        ))

    # Sort by IUPAC alphabetization rules (ignoring di-, tri-, etc.)
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    return prefixes


def _generate_stereodescriptors(features: Any) -> Optional[NameFragment]:
    """
    Generate stereodescriptor prefix using correct IUPAC locants.

    Uses the stereochemistry rules module to collect R/S and E/Z descriptors
    based on the atom_to_locant mapping (from chain/ring orientation).

    For different compound types:
    - Acyclic: uses features.atom_to_locant (from principal chain orientation)
    - Heterocycles: uses features.heterocycle_atom_to_locant
    - Cycloalkanes/cycloalkenes: builds from features.oriented_ring

    Args:
        features: MolecularFeatures object with stereocenters and/or double_bond_stereo

    Returns:
        NameFragment with stereodescriptor prefix like "(2R)-" or "(2E,3R)-",
        or None if no stereodescriptors.
    """
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Need either stereocenters or double_bond_stereo
    if not features.stereocenters and not getattr(features, 'double_bond_stereo', None):
        return None

    mol = features.mol

    # Determine which atom_to_locant mapping to use
    # For acyclic: features.atom_to_locant (from chain orientation)
    # For rings: features.heterocycle_atom_to_locant or build from oriented_ring
    atom_to_locant = features.atom_to_locant

    # For heterocycles, use ring-specific mapping
    if getattr(features, 'heterocycle_atom_to_locant', None):
        atom_to_locant = features.heterocycle_atom_to_locant
    elif getattr(features, 'oriented_ring', None):
        # Build mapping from oriented_ring for cycloalkanes/cycloalkenes
        # oriented_ring is a list of atom indices in ring order starting at position 1
        # This creates ring_atom_to_locant: {atom_idx: ring_locant} where locants are 1-indexed
        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(features.oriented_ring)}

    if not atom_to_locant:
        return None

    # Collect stereodescriptors with proper locants
    descriptors = collect_stereodescriptors(mol, atom_to_locant)

    if not descriptors:
        return None

    # Format as "(2R,3S)-" etc
    text = format_stereodescriptor_string(descriptors)

    return NameFragment(text=text, fragment_type="stereo")


def _assemble_fragments(fragments: List[NameFragment], style: str) -> str:
    """
    Assemble fragments into final name string.

    Order: stereo + prefixes (alphabetized) + parent + suffix

    For functional group compounds, uses PIN-style infix locants:
    - propan-1-ol (not propanol or 1-propanol)
    - butan-2-one (not butanone or 2-butanone)

    For unsaturated hydrocarbons, includes bond locants:
    - but-1-ene (not butene or 1-butene)
    - pent-1-en-4-yne (enyne)
    """
    stereo = ""
    prefixes = []
    parent_frag = None
    suffix_frag = None

    for frag in fragments:
        if frag.fragment_type == "stereo":
            stereo = frag.text
        elif frag.fragment_type == "prefix":
            prefixes.append(frag)
        elif frag.fragment_type == "parent":
            parent_frag = frag
        elif frag.fragment_type == "suffix":
            suffix_frag = frag

    # Alkyl prefixes are already sorted by _generate_alkyl_prefixes.
    # For non-alkyl prefixes added later, sort all together.
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    # Concatenate prefixes with proper hyphenation
    # IUPAC rule: hyphens separate locants from names, and are needed
    # between prefixes when one ends with a letter and the next starts with a digit
    prefix_str = _join_prefixes([f.text for f in prefixes])

    # Extract parent info: stem is in text, bond locants in locants
    stem = parent_frag.text if parent_frag else ""
    double_locants = []
    triple_locants = []
    if parent_frag and parent_frag.locants:
        # locants is a tuple of (double_bond_locants, triple_bond_locants)
        double_locants = list(parent_frag.locants[0]) if parent_frag.locants[0] else []
        triple_locants = list(parent_frag.locants[1]) if len(parent_frag.locants) > 1 and parent_frag.locants[1] else []

    # Handle suffix attachment using PIN-style formatting
    if suffix_frag and suffix_frag.text:
        suffix_text = suffix_frag.text
        suffix_locants = list(suffix_frag.locants) if suffix_frag.locants else []

        # Determine multiplier for multiple functional groups
        count = len(suffix_locants)
        multiplier = get_multiplier_prefix(count, suffix_text) if count > 1 else ""

        # Build unsaturation infix with locants for compounds with functional groups
        unsaturation_infix = _build_unsaturation_infix(double_locants, triple_locants)

        name = format_suffix_with_locants(
            stem,
            unsaturation_infix,
            suffix_text,
            suffix_locants,
            multiplier
        )
    else:
        # No suffix = hydrocarbon, build name with unsaturation
        name = _build_hydrocarbon_name(stem, double_locants, triple_locants)

    # Add prefixes
    if prefix_str:
        name = f"{prefix_str}{name}"

    # Add stereodescriptors at the very start
    if stereo:
        name = f"{stereo}{name}"

    return name


def _build_unsaturation_infix(
    double_locants: List[int],
    triple_locants: List[int]
) -> str:
    """
    Build the unsaturation infix (e.g., 'an', '-1-en', '-1-en-4-yn').

    This is used when there's a functional group suffix. The unsaturation
    part is inserted between the stem and the suffix locants.

    Args:
        double_locants: Sorted list of locants for double bonds.
        triple_locants: Sorted list of locants for triple bonds.

    Returns:
        Unsaturation infix string (may be empty for saturated).

    Examples:
        >>> _build_unsaturation_infix([], [])
        'an'
        >>> _build_unsaturation_infix([1], [])
        '-1-en'
        >>> _build_unsaturation_infix([1], [4])
        '-1-en-4-yn'
    """
    num_double = len(double_locants)
    num_triple = len(triple_locants)

    if num_double == 0 and num_triple == 0:
        return "an"  # Saturated

    parts = []

    # Double bonds
    if num_double > 0:
        double_str = ",".join(str(loc) for loc in double_locants)
        if num_double == 1:
            parts.append(f"-{double_str}-en")
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            # For multiple double bonds, add 'a' connector: butadiene not butdiene
            parts.append(f"a-{double_str}-{multiplier}en")

    # Triple bonds
    if num_triple > 0:
        triple_str = ",".join(str(loc) for loc in triple_locants)
        if num_triple == 1:
            parts.append(f"-{triple_str}-yn")
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            parts.append(f"-{triple_str}-{multiplier}yn")

    # Special case: if only double bonds and parts starts with 'a-' for diene,
    # we need to handle this differently for saturated stem
    if parts:
        result = "".join(parts)
        # Clean up any double hyphens
        while "--" in result:
            result = result.replace("--", "-")
        return result

    return "an"


def _build_hydrocarbon_name(
    stem: str,
    double_locants: List[int],
    triple_locants: List[int]
) -> str:
    """
    Build a complete hydrocarbon name (no functional group suffix).

    This handles:
    - Saturated: stem + 'ane' (butane, cyclohexane)
    - Alkenes: stem + '-locant-ene' (but-1-ene) or stem + 'ene' (ethene, cyclohexene)
    - Alkynes: stem + '-locant-yne' (but-1-yne) or stem + 'yne' (ethyne)
    - Enynes: stem + '-double-en-triple-yne' (pent-1-en-4-yne)
    - Cycloalkenes: stem + 'ene' for mono (cyclohexene), stem + 'a-locants-diene' for di

    For 2-carbon compounds (ethene, ethyne), locants are omitted since
    there's only one possible position for the multiple bond.

    For mono-cycloalkenes (single double bond in ring), locants are omitted
    per IUPAC convention (cyclohexene, not cyclohex-1-ene).

    Args:
        stem: Chain prefix (e.g., 'but', 'pent', 'cyclohex').
        double_locants: Sorted list of locants for double bonds.
        triple_locants: Sorted list of locants for triple bonds.

    Returns:
        Complete hydrocarbon name.

    Examples:
        >>> _build_hydrocarbon_name("but", [], [])
        'butane'
        >>> _build_hydrocarbon_name("eth", [1], [])
        'ethene'
        >>> _build_hydrocarbon_name("but", [1], [])
        'but-1-ene'
        >>> _build_hydrocarbon_name("but", [2], [])
        'but-2-ene'
        >>> _build_hydrocarbon_name("pent", [1], [4])
        'pent-1-en-4-yne'
        >>> _build_hydrocarbon_name("cyclohex", [], [])
        'cyclohexane'
        >>> _build_hydrocarbon_name("cyclohex", [1], [])
        'cyclohexene'
        >>> _build_hydrocarbon_name("cyclohex", [1, 3], [])
        'cyclohexa-1,3-diene'
    """
    num_double = len(double_locants)
    num_triple = len(triple_locants)

    # Check if this is a cyclic compound
    is_cyclic = stem.startswith("cyclo")

    if num_double == 0 and num_triple == 0:
        # Saturated hydrocarbon
        return f"{stem}ane"

    # Check if this is a 2-carbon chain (ethene/ethyne) where locants are omitted
    # The stem 'eth' indicates 2 carbons
    is_two_carbon = stem == "eth"

    # For mono-cycloalkenes, locant is omitted (cyclohexene, not cyclohex-1-ene)
    # This is IUPAC convention for simple mono-unsaturated cyclic compounds
    omit_mono_cycloalkene_locant = is_cyclic and num_double == 1 and num_triple == 0

    parts = [stem]

    # Double bonds
    if num_double > 0:
        if num_double == 1:
            if is_two_carbon and num_triple == 0:
                # ethene - no locant needed
                parts.append("en")
            elif omit_mono_cycloalkene_locant:
                # cyclohexene - no locant needed for mono-cycloalkene
                parts.append("en")
            else:
                # Single double bond with locant: but-1-ene
                double_str = ",".join(str(loc) for loc in double_locants)
                parts.append(f"-{double_str}-en")
        else:
            # Multiple double bonds: buta-1,3-diene, cyclohexa-1,3-diene
            # Add 'a' before locants for pronunciation
            double_str = ",".join(str(loc) for loc in double_locants)
            multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            parts.append(f"a-{double_str}-{multiplier}en")

    # Triple bonds
    if num_triple > 0:
        if num_triple == 1:
            if is_two_carbon and num_double == 0:
                # ethyne - no locant needed
                parts.append("yn")
            else:
                # Single triple bond with locant
                triple_str = ",".join(str(loc) for loc in triple_locants)
                parts.append(f"-{triple_str}-yn")
        else:
            # Multiple triple bonds
            triple_str = ",".join(str(loc) for loc in triple_locants)
            multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            # If there were also double bonds, we already have 'a', otherwise add it
            if num_double == 0:
                parts.append(f"a-{triple_str}-{multiplier}yn")
            else:
                parts.append(f"-{triple_str}-{multiplier}yn")

    # Add final 'e'
    result = "".join(parts) + "e"

    # Clean up: handle edge cases
    # 1. Double hyphens shouldn't happen but clean up just in case
    while "--" in result:
        result = result.replace("--", "-")

    return result


def _split_parent_stem(parent: str) -> tuple:
    """
    Split parent string into stem and unsaturation components.

    The parent string from _generate_chain_parent is stem + unsaturation,
    e.g., "propan" = "prop" + "an", "buten" = "but" + "en".

    Returns:
        Tuple of (stem, unsaturation)
    """
    # Unsaturation suffixes in order of longest first to avoid partial matches
    unsaturation_patterns = [
        "adienyn", "adien", "adiyn", "aenyn",  # Complex unsaturation
        "dien", "diyn", "enyn",  # Double unsaturation
        "en", "yn", "an",  # Simple unsaturation
    ]

    for pattern in unsaturation_patterns:
        if parent.endswith(pattern):
            stem = parent[:-len(pattern)]
            return (stem, pattern)

    # No recognized unsaturation - return as-is with empty unsaturation
    return (parent, "")


# Note: alpha_sort_key is imported from naming_utils for IUPAC-compliant sorting.


def _join_prefixes(prefix_texts: List[str]) -> str:
    """
    Join multiple prefix strings with proper IUPAC hyphenation.

    When concatenating prefixes like "3-ethyl" and "4-methyl", the result
    should be "3-ethyl-4-methyl" (with hyphen between letter and digit).

    Args:
        prefix_texts: List of prefix strings (e.g., ["3-ethyl", "4-methyl"])

    Returns:
        Concatenated string with proper hyphenation

    Examples:
        >>> _join_prefixes(["3-ethyl", "4-methyl"])
        '3-ethyl-4-methyl'
        >>> _join_prefixes(["2,2-dimethyl"])
        '2,2-dimethyl'
        >>> _join_prefixes([])
        ''
    """
    if not prefix_texts:
        return ""

    if len(prefix_texts) == 1:
        return prefix_texts[0]

    # Join with hyphens where needed
    result = prefix_texts[0]
    for i in range(1, len(prefix_texts)):
        current = prefix_texts[i]
        # If result ends with letter and current starts with digit, add hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]
            if last_char.isalpha() and first_char.isdigit():
                result += "-"
        result += current

    return result


def _build_long_chain_prefix(length: int) -> str:
    """
    Build prefix for chains longer than those in CHAIN_PREFIXES.
    
    Uses IUPAC multiplicative system for very long chains.
    """
    # For now, just return the numerical form
    # TODO: Implement proper long-chain naming (triacontane, etc.)
    return f"{length}C"


def format_locants(locants: tuple) -> str:
    """
    Format a tuple of locants for name insertion.

    Example: (2, 3) -> "2,3-"
    """
    if not locants:
        return ""
    return ",".join(str(l) for l in sorted(locants)) + "-"


def assemble_ion_name(features: Any, mol, style: str = 'pin') -> str:
    """
    Assemble name for ionic or radical species.

    Routes to appropriate naming function based on species_type.
    This function is the entry point for the composer to handle
    non-neutral molecules.

    Args:
        features: MolecularFeatures with species_type populated
        mol: RDKit Mol object
        style: 'pin' for preferred names

    Returns:
        IUPAC name for the ion/radical

    Example:
        >>> # For a salt:
        >>> assemble_ion_name(features, mol)
        'sodium acetate'
        >>> # For a radical:
        >>> assemble_ion_name(features, mol)
        'methyl'
    """
    # Import naming functions here to avoid circular imports
    from ..rules.ions import name_anion, name_cation
    from ..rules.salts import name_salt, name_zwitterion
    from ..rules.radicals import name_radical

    species_type = features.species_type

    if species_type == 'salt':
        return name_salt(mol, style)
    elif species_type == 'radical':
        return name_radical(mol, style)
    elif species_type == 'zwitterion':
        return name_zwitterion(mol, style)
    elif species_type == 'ion':
        # Single ion - determine if cation or anion
        ion_sites = getattr(features, 'ion_sites', {})
        if ion_sites.get('cations') and not ion_sites.get('anions'):
            return name_cation(mol, style)
        elif ion_sites.get('anions') and not ion_sites.get('cations'):
            return name_anion(mol, style)

    # Fallback - should not reach here for valid ionic species
    raise ValueError(f"Cannot assemble name for species_type: {species_type}")


def get_multiplier(count: int, is_complex: bool = False) -> str:
    """
    Get the appropriate multiplier prefix for a count.
    
    Args:
        count: Number of occurrences
        is_complex: True if the substituent name is complex (has locants/hyphens)
        
    Returns:
        Multiplier string (e.g., "di", "tris")
    """
    if count <= 1:
        return ""
    
    multipliers = COMPLEX_MULTIPLIERS if is_complex else SIMPLE_MULTIPLIERS
    return multipliers.get(count, str(count))
