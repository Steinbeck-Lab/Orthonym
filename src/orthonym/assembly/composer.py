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
    SIMPLE_MULTIPLIERS,
    COMPLEX_MULTIPLIERS,
)
from ..data.chain_names import get_chain_prefix

# Ion/radical naming imports - deferred to avoid circular imports
# These are imported inside functions that need them

# Complex ring system imports
from ..rules.bicyclo import is_bicyclo_system, name_bicyclo_system
from ..rules.spiro import is_spiro_system, name_spiro_system
from ..rules.fused_rings import classify_fused_system, name_fused_heterocycle, name_ortho_fused_bicyclic

# Polycyclic system imports (Phase 16)
from ..rules.polycyclic import name_polycyclic_complete, is_polycyclic_system
from ..rules.bridged_fused import detect_bridged_fused, name_bridged_fused_system

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


# Prefixes to IGNORE for alphabetization
IGNORE_FOR_ALPHA = set(SIMPLE_MULTIPLIERS.values()) | set(COMPLEX_MULTIPLIERS.values())

# Terminal functional groups that NEVER include locants in the name
# These are always at position 1 by definition (chain numbered from terminal group)
TERMINAL_GROUPS = {
    "carboxylic_acid",  # Always at chain end (locant 1)
    "aldehyde",         # Always at chain end (locant 1)
    "nitrile",          # Always at chain end (locant 1)
    "primary_amide",    # Always at chain end (locant 1)
    "secondary_amide",  # Always at chain end (locant 1)
    "tertiary_amide",   # Always at chain end (locant 1)
    "acid_chloride",    # Always at chain end (locant 1)
    "acid_bromide",     # Always at chain end (locant 1)
    "acid_fluoride",    # Always at chain end (locant 1)
}


@dataclass
class NameFragment:
    """A fragment of an IUPAC name."""
    text: str
    locants: tuple = ()
    priority: int = 0
    fragment_type: str = "prefix"  # prefix, parent, suffix, stereo
    count: int = 1  # Number of instances (for multiplier when locants are omitted)


def _get_parent_atom_count(features) -> int:
    """Get the atom count of the parent structure for locant validation.

    Returns the number of atoms in the principal chain or oriented ring,
    used by locant validation to filter out-of-range locants.

    Args:
        features: MolecularFeatures object.

    Returns:
        Number of atoms in the parent, or 100 as safe fallback.
    """
    principal_chain = getattr(features, 'principal_chain', None)
    if principal_chain:
        return len(principal_chain)
    oriented_ring = getattr(features, 'oriented_ring', None)
    if oriented_ring:
        return len(oriented_ring)
    oriented_het = getattr(features, 'oriented_heterocycle', None)
    if oriented_het:
        return len(oriented_het)
    principal_ring = getattr(features, 'principal_ring', None)
    if principal_ring:
        return len(principal_ring)
    return 100  # Safe fallback: don't filter anything


def _try_ion_aspect_composition(features, style='pin'):
    """Try to name a single-component ion by composing parent + ion modification.

    For ions that don't have a retained name, this function:
    1. Resolves the parent structure using resolvers.py
    2. Gets the neutral name by temporarily setting species_type to 'neutral'
    3. Applies ion suffix modification (e.g., -ol -> -olate, -ane -> -ide)

    This preserves the parent ring/chain identity in the ion name rather than
    using the simpler systematic fallback (e.g., "propanolate" not "propoxide").

    IMPORTANT: This function is conservative -- if composition doesn't produce
    a clean suffix swap, it returns None and falls back to the existing ion
    naming path (name_anion/name_cation) which handles all edge cases.

    Args:
        features: MolecularFeatures with species_type == 'ion'.
        style: Naming style.

    Returns:
        Composed ion name string, or None if composition fails (triggers fallback).
    """
    from .resolvers import resolve_parent, resolve_suffix, apply_ion_suffix_modification

    mol = getattr(features, 'mol', None)
    if mol is None or mol.GetNumAtoms() < 2:
        return None

    has_carbon = any(a.GetSymbol() == 'C' for a in mol.GetAtoms())
    if not has_carbon:
        return None

    parent_info = resolve_parent(features, mol)
    if not parent_info or parent_info.parent_type == "unknown":
        return None

    # Only attempt aspect composition for chain-based parents with a recognized
    # principal group that has a known ion suffix transformation.
    # Ring-based ions (phenolate, indolate) are more reliably handled by the
    # existing name_anion/name_cation pathway which neutralizes and renames.
    fg_name = getattr(features, 'principal_group', None)
    if parent_info.parent_type != 'chain':
        return None  # Let fallback handle ring-based ions

    # Require a recognized principal group for aspect composition.
    # Without one, FG detection failed (charged atoms don't match neutral SMARTS),
    # and the existing name_anion/name_cation pathway handles this correctly
    # by neutralizing the molecule before naming.
    if not fg_name:
        return None

    # Resolve suffix to check if ion modification would apply
    suffix_info = resolve_suffix(features, parent_info)
    modified_suffix = apply_ion_suffix_modification(suffix_info, features)

    # Only proceed if we have a clear suffix transformation
    if not modified_suffix or modified_suffix.text == suffix_info.text:
        return None  # No transformation available, fall back

    # Get the neutral name using recursion guard
    original_species_type = features.species_type
    features.species_type = 'neutral'
    try:
        neutral_name = assemble_name(features, style, _composing_ion=True)
    finally:
        features.species_type = original_species_type

    if not neutral_name or neutral_name == 'unknown':
        return None

    # Try to swap the suffix in the neutral name
    if suffix_info.text and neutral_name.endswith(suffix_info.text.lstrip('-')):
        old_suffix = suffix_info.text.lstrip('-')
        new_suffix = modified_suffix.text.lstrip('-')
        return neutral_name[:-len(old_suffix)] + new_suffix
    elif not suffix_info.text:
        # No neutral suffix (bare hydrocarbon) -- append ion suffix
        new_suffix = modified_suffix.text.lstrip('-')
        if neutral_name.endswith('e'):
            return neutral_name[:-1] + new_suffix
        return neutral_name + new_suffix

    # Suffix swap didn't match, fall back
    return None


def assemble_name(features: Any, style: str = "pin", _composing_ion: bool = False) -> str:
    """
    Assemble complete IUPAC name from molecular features.

    Args:
        features: MolecularFeatures object with extracted features
        style: Naming style ("pin", "general", "cas")
        _composing_ion: Internal recursion guard. When True, ion aspect
            composition is skipped to prevent infinite loops. Do not
            set manually -- it is used by _try_ion_aspect_composition().

    Returns:
        Complete IUPAC name string
    """
    # Check for ionic/radical species first - route to specialized assembly
    species_type = getattr(features, 'species_type', 'neutral')

    # Salt, zwitterion: functional class naming (truly different naming structure)
    if species_type in ('salt', 'zwitterion'):
        return assemble_ion_name(features, features.mol, style)

    # Radical: also uses specialized naming
    if species_type == 'radical':
        return assemble_ion_name(features, features.mol, style)

    # Single-component ion: try aspect-based composition first
    if species_type == 'ion' and not _composing_ion:
        composed_name = _try_ion_aspect_composition(features, style)
        if composed_name:
            return composed_name
        # Fallback to existing ion naming if composition fails
        return assemble_ion_name(features, features.mol, style)

    # Handle oximes - functional class naming: "propan-2-one oxime"
    if features.principal_group == 'oxime':
        oxime_name = _name_oxime_or_hydrazone(features, 'oxime')
        if oxime_name:
            return oxime_name

    # Handle hydrazones - functional class naming: "propan-2-one hydrazone"
    if features.principal_group == 'hydrazone':
        hydrazone_name = _name_oxime_or_hydrazone(features, 'hydrazone')
        if hydrazone_name:
            return hydrazone_name

    # Handle N-oxides - functional class naming: "pyridine 1-oxide"
    # Must detect early because N-oxides have internal charges that could
    # confuse other routing (they are classified as 'neutral' by ions.py)
    n_oxide_name = _try_name_n_oxide(features)
    if n_oxide_name:
        return n_oxide_name

    # Handle isocyanates - functional class: "methyl isocyanate"
    # Only when isocyanate is the sole FG (principal_group is None because
    # isocyanate is not in SENIORITY_ORDER). When another FG is principal,
    # isocyanate becomes prefix "isocyanato" via the polyfunctional handler.
    if (features.functional_groups.get('isocyanate')
            and features.principal_group is None):
        iso_name = _name_isocyanate(features)
        if iso_name:
            return iso_name

    # Handle isothiocyanates - functional class: "methyl isothiocyanate"
    # Same gating as isocyanate above.
    if (features.functional_groups.get('isothiocyanate')
            and features.principal_group is None):
        isothio_name = _name_isothiocyanate(features)
        if isothio_name:
            return isothio_name

    # Handle carbamates - functional class: "ethyl carbamate"
    # Must detect BEFORE generic ester to prevent N loss.
    # Only when carbamate is the primary FG (no higher-seniority principal group).
    if (features.functional_groups.get('carbamate')
            and features.principal_group is None):
        carb_name = _name_carbamate(features)
        if carb_name:
            return carb_name

    # Handle urea compounds - retained name with N-substitution
    # "urea", "N-methylurea", "N,N-dimethylurea", "N,N'-dimethylurea"
    # Must detect BEFORE polyfunctional handler to prevent garbled output
    if (features.functional_groups.get('urea')
            and features.principal_group is None):
        urea_name = _try_name_urea(features)
        if urea_name:
            return urea_name

    # Handle guanidine compounds - retained name with N-substitution
    # "guanidine", "N-methylguanidine", "N,N-dimethylguanidine"
    if (features.functional_groups.get('guanidine')
            and features.principal_group is None):
        guanidine_name = _try_name_guanidine(features)
        if guanidine_name:
            return guanidine_name

    # Handle acid halides BEFORE polyfunctional and ester handlers
    # Acid halides use functional class naming: "ethanoyl chloride" (two-word)
    # Must come before polyfunctional because acid_chloride + chloro triggers polyfunctional
    if features.principal_group in ('acid_chloride', 'acid_bromide', 'acid_fluoride'):
        from ..rules.acid_halides import name_acid_halide
        halide_name = name_acid_halide(features)
        if halide_name:
            return halide_name

    # Handle anhydrides BEFORE lactone, polyfunctional, and ester handlers
    # Anhydrides use functional class naming: "ethanoic anhydride" (two-word)
    # Must come before lactone because cyclic anhydrides (O=C1CCC(=O)O1) would
    # otherwise be misidentified as lactones
    if features.principal_group == 'anhydride':
        from ..rules.anhydrides import name_anhydride
        anhydride_name = name_anhydride(features)
        if anhydride_name:
            return anhydride_name

    # Handle monocyclic lactones BEFORE polyfunctional, esters, and heterocycles
    # Lactones are cyclic esters that should be named as heterocyclic ketones
    # (e.g., oxolan-2-one, not tetrahydrofuran or "alkyl alkanoate")
    # Must come before polyfunctional because lactones trigger polyfunctional detection
    from ..rules.lactones import is_monocyclic_lactone, name_monocyclic_lactone
    lactone_info = is_monocyclic_lactone(features.mol)
    if lactone_info:
        lactone_name = name_monocyclic_lactone(features.mol)
        if lactone_name:
            return lactone_name

    # Handle monocyclic lactams BEFORE polyfunctional and amide handlers
    # Lactams are cyclic amides named as heterocyclic ketones (parallel to lactones)
    # e.g., azetidin-2-one, pyrrolidin-2-one, piperidin-2-one
    from ..rules.lactams import is_monocyclic_lactam, name_monocyclic_lactam
    lactam_info = is_monocyclic_lactam(features.mol)
    if lactam_info:
        lactam_name = name_monocyclic_lactam(features.mol)
        if lactam_name:
            return lactam_name

    # Handle ring-attached esters BEFORE polyfunctional handler
    # Ring-attached esters (e.g., cyclohexyl acetate, phenyl acetate) should
    # use acyloxy prefix naming on ring parent, not polyfunctional naming
    # BUT: skip for polycyclic/complex ring systems -- those need the complex
    # ring naming path which handles substituents (including esters) properly
    if features.principal_group == "ester":
        from ..rules.esters import detect_exocyclic_esters
        exocyclic = detect_exocyclic_esters(features.mol)
        if exocyclic and not _is_complex_ring_system(features.mol):
            ring_ester_name = _assemble_ring_with_ester_prefixes(features, exocyclic)
            if ring_ester_name:
                return ring_ester_name

    # Handle polyfunctional compounds (multiple distinct functional groups)
    if getattr(features, 'is_polyfunctional', False):
        from ..rules.polyfunctional import name_polyfunctional
        poly_name = name_polyfunctional(features)
        if poly_name:
            return poly_name
        # If name_polyfunctional returns None, fall through to normal handling

    # Handle esters (two-component naming: "alkyl alkanoate")
    # Only reached for acyclic esters (ring-attached esters handled above)
    if features.principal_group == "ester":
        ester_match = getattr(features, 'ester_match', None)
        if ester_match:
            from ..rules.esters import name_ester
            # Simple acyclic ester: use "alkyl alkanoate" naming
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
    # Skip cyclic thioethers (1,3-dithiane, thiane, etc.) - they are named as heterocycles
    if features.principal_group == 'thioether':
        ring_type = getattr(features, 'ring_type', None)
        if ring_type != 'heterocyclic':
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

    # Handle boronic acids - functional class: "methylboronic acid", "phenylboronic acid"
    if features.principal_group == 'boronic_acid':
        boronic_name = _name_boronic_acid(features)
        if boronic_name:
            return boronic_name

    # Handle ring assemblies (biphenyl, bipyridine) BEFORE complex ring systems
    # Ring assemblies are separate identical ring systems connected by single bonds
    assembly_info = getattr(features, 'ring_assembly_info', None)
    if assembly_info:
        from ..rules.ring_assemblies import name_ring_assembly
        assembly_name = name_ring_assembly(features.mol, assembly_info, features)
        if assembly_name:
            return assembly_name

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
        # Safety net: check if this heterocycle is actually a lactone
        # (catches cases where lactone detection in ester routing was bypassed)
        from ..rules.lactones import is_monocyclic_lactone, name_monocyclic_lactone
        lactone_info = is_monocyclic_lactone(features.mol)
        if lactone_info:
            lactone_name = name_monocyclic_lactone(features.mol)
            if lactone_name:
                return lactone_name
        return _assemble_heterocycle_name(features, style)

    # Handle benzene derivatives
    # Only reached if not a fused system containing benzene
    if getattr(features, 'is_benzene', False):
        return _assemble_benzene_name(features, style)

    # Handle ring-attached nitriles (cyclohexanecarbonitrile, etc.)
    if features.principal_group == 'nitrile' and features.is_cyclic:
        return _assemble_ring_nitrile_name(features, style)

    # Handle amides (including N-substituted amides)
    # Multi-amide compounds (diamide, triamide) fall through to normal suffix path
    # so that the multiplier prefix (di-, tri-) is correctly applied.
    if features.principal_group in ('primary_amide', 'secondary_amide', 'tertiary_amide'):
        pg_count = len(features.principal_group_atoms) if features.principal_group_atoms else 1
        if pg_count == 1:
            return _assemble_amide_name(features, style)

    # Handle secondary/tertiary amines: add N-alkyl prefixes
    if features.principal_group in ('secondary_amine', 'tertiary_amine'):
        amine_name = _assemble_amine_name(features, style)
        if amine_name:
            return amine_name

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
    suffix = None
    if features.principal_group:
        suffix = _generate_suffix(features)
        if suffix:
            # Validate suffix locants against parent capacity
            parent_size = _get_parent_atom_count(features)
            from ..rules.locant_validation import validate_suffix_locants
            validated_locants, validated_count = validate_suffix_locants(
                list(suffix.locants), parent_size, suffix.count
            )
            if validated_locants != list(suffix.locants) or validated_count != suffix.count:
                suffix = NameFragment(
                    text=suffix.text,
                    locants=tuple(validated_locants),
                    fragment_type="suffix",
                    count=validated_count,
                )
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


def _name_oxime_or_hydrazone(features: Any, fg_type: str) -> Optional[str]:
    """Name oximes and hydrazones using functional class naming.

    Converts the =N-OH (oxime) or =N-NH2 (hydrazone) back to =O (parent
    carbonyl), names the carbonyl via name_compound(), and appends the
    functional class suffix (" oxime" or " hydrazone").

    Args:
        features: MolecularFeatures object with principal_group == fg_type
        fg_type: Either 'oxime' or 'hydrazone'

    Returns:
        Functional class name like "propan-2-one oxime", or None on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol

    mol = features.mol
    matches = features.functional_groups.get(fg_type, [])
    if not matches:
        return None

    # Take first match: SMARTS "[CX3]=[NX2][OX2H]" for oxime gives (C, N, O)
    # SMARTS "[CX3]=[NX2][NX3]" for hydrazone gives (C, N, N)
    match = matches[0]
    c_idx = match[0]   # The carbon (C=N)
    n_idx = match[1]   # The nitrogen (=N-)
    tail_idx = match[2]  # OH (oxime) or NH2 (hydrazone)

    # Build modified molecule: replace =N-X with =O
    rw = RWMol(mol)

    # We need to be careful about atom indices shifting after removal.
    # Strategy: remove the tail atom and the N atom, then add =O to the carbon.

    # First, find the bond from N to C
    bond_cn = rw.GetBondBetweenAtoms(c_idx, n_idx)
    if bond_cn is None:
        return None

    # Remove all bonds from N
    # Get neighbors of N (beyond C and tail)
    n_atom = rw.GetAtomWithIdx(n_idx)
    n_neighbors = [nbr.GetIdx() for nbr in n_atom.GetNeighbors()]

    # For hydrazone, tail N might have H atoms only (implicit), so just
    # removing the N and tail atoms plus adding O should work.
    # For oxime, tail is O-H.

    # Simpler approach: edit SMILES string
    # Convert to SMILES, substitute the FG pattern
    try:
        # Strategy: Use RWMol to replace atoms
        # 1. Remove bond N-tail
        rw.RemoveBond(n_idx, tail_idx)
        # 2. Remove bond C-N (the double bond)
        rw.RemoveBond(c_idx, n_idx)
        # 3. Add oxygen atom
        o_idx = rw.AddAtom(Chem.Atom(8))  # oxygen
        # 4. Add C=O bond
        rw.AddBond(c_idx, o_idx, Chem.BondType.DOUBLE)

        # Now remove orphaned atoms (N and tail) - must remove higher index first
        atoms_to_remove = sorted([n_idx, tail_idx], reverse=True)

        # But if hydrazone tail N has hydrogens attached as explicit atoms, remove those too
        # Actually, for simple cases implicit H should be fine. Let's just remove the two atoms.
        for aidx in atoms_to_remove:
            rw.RemoveAtom(aidx)

        # Sanitize
        modified_mol = rw.GetMol()
        Chem.SanitizeMol(modified_mol)
        modified_smiles = Chem.MolToSmiles(modified_mol, canonical=True)
    except Exception:
        return None

    # Name the parent carbonyl by recursion
    from ..namer import name_compound
    try:
        parent_name = name_compound(modified_smiles)
    except Exception:
        return None

    if not parent_name:
        return None

    return f"{parent_name} {fg_type}"


# ============================================================================
# N-oxide naming (functional class: "pyridine 1-oxide", "trimethylamine N-oxide")
# ============================================================================

# Recursion guard for N-oxide naming (prevents infinite loop when naming base compound)
import threading as _threading
_n_oxide_guard = _threading.local()


def _try_name_n_oxide(features: Any) -> Optional[str]:
    """Try to name molecule as an N-oxide using functional class naming.

    Aromatic N-oxides: "pyridine 1-oxide", "4-methylpyridine 1-oxide"
    Aliphatic N-oxides: "trimethylamine N-oxide"

    The approach:
    1. Detect N-oxide pattern (aromatic [n+][O-] or aliphatic [NX4+]([C])([C])([C])[O-])
    2. Create a modified molecule with O- removed and N+ neutralized
    3. Name the base compound recursively via name_compound()
    4. Append oxide suffix

    Returns:
        Functional class name, or None if not an N-oxide.
    """
    # Recursion guard: if we're already naming a base compound, skip
    if getattr(_n_oxide_guard, 'active', False):
        return None

    from rdkit import Chem
    from rdkit.Chem import RWMol

    mol = features.mol

    # Check for aromatic N-oxide: [n+][O-]
    aromatic_pat = Chem.MolFromSmarts('[n+][O-]')
    aliphatic_pat = Chem.MolFromSmarts('[NX4+]([#6])([#6])([#6])[O-]')

    aromatic_matches = mol.GetSubstructMatches(aromatic_pat) if aromatic_pat else ()
    aliphatic_matches = mol.GetSubstructMatches(aliphatic_pat) if aliphatic_pat else ()

    if not aromatic_matches and not aliphatic_matches:
        return None

    if aromatic_matches:
        return _name_aromatic_n_oxide(mol, aromatic_matches)
    else:
        return _name_aliphatic_n_oxide(mol, aliphatic_matches)


def _name_aromatic_n_oxide(mol, matches) -> Optional[str]:
    """Name aromatic N-oxide: e.g., 'pyridine 1-oxide'.

    Strategy: remove O- atom, neutralize N+, name the base heterocycle,
    then append '{locant}-oxide'.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol

    # Use the first N-oxide match
    n_idx, o_idx = matches[0]  # [n+] index, [O-] index

    # Build modified molecule: remove O-, neutralize N+
    rw = RWMol(mol)

    # Set N formal charge to 0
    rw.GetAtomWithIdx(n_idx).SetFormalCharge(0)

    # Remove O- atom (remove bond first, then atom)
    rw.RemoveBond(n_idx, o_idx)

    # Need to remove the O atom. But removing an atom shifts indices of atoms
    # with higher indices. Since we only remove one atom, just remove it.
    rw.RemoveAtom(o_idx)

    try:
        modified_mol = rw.GetMol()
        Chem.SanitizeMol(modified_mol)
        modified_smiles = Chem.MolToSmiles(modified_mol, canonical=True)
    except Exception:
        return None

    # Name the base compound recursively
    from ..namer import name_compound
    _n_oxide_guard.active = True
    try:
        base_name = name_compound(modified_smiles)
    except Exception:
        return None
    finally:
        _n_oxide_guard.active = False

    if not base_name:
        return None

    # For heterocycles, the N is typically at position 1
    # IUPAC format: "pyridine 1-oxide"
    return f"{base_name} 1-oxide"


def _name_aliphatic_n_oxide(mol, matches) -> Optional[str]:
    """Name aliphatic N-oxide: e.g., 'trimethylamine N-oxide'.

    Strategy: remove O- atom, change N from +1 to 0 charge,
    name the neutral amine, then append ' N-oxide'.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol

    # Match is (N, C, C, C, O) from SMARTS [NX4+]([#6])([#6])([#6])[O-]
    n_idx = matches[0][0]
    o_idx = matches[0][4]  # The O- atom

    # Build modified molecule
    rw = RWMol(mol)

    # Neutralize N
    rw.GetAtomWithIdx(n_idx).SetFormalCharge(0)

    # Remove O- bond and atom
    rw.RemoveBond(n_idx, o_idx)
    rw.RemoveAtom(o_idx)

    try:
        modified_mol = rw.GetMol()
        Chem.SanitizeMol(modified_mol)
        modified_smiles = Chem.MolToSmiles(modified_mol, canonical=True)
    except Exception:
        return None

    # Name the base amine recursively
    from ..namer import name_compound
    _n_oxide_guard.active = True
    try:
        base_name = name_compound(modified_smiles)
    except Exception:
        return None
    finally:
        _n_oxide_guard.active = False

    if not base_name:
        return None

    return f"{base_name} N-oxide"


# ============================================================================
# Isocyanate / Isothiocyanate naming (functional class)
# ============================================================================

def _name_isocyanate(features: Any) -> Optional[str]:
    """Name isocyanate as 'R isocyanate' (functional class naming).

    Pattern: R-N=C=O
    SMARTS match gives (R_carbon, N, C, O)

    Returns:
        Functional class name like 'methyl isocyanate', or None.
    """
    return _name_iso_x_cyanate(features, 'isocyanate', 'isocyanate')


def _name_isothiocyanate(features: Any) -> Optional[str]:
    """Name isothiocyanate as 'R isothiocyanate' (functional class naming).

    Pattern: R-N=C=S
    SMARTS match gives (R_carbon, N, C, S)

    Returns:
        Functional class name like 'methyl isothiocyanate', or None.
    """
    return _name_iso_x_cyanate(features, 'isothiocyanate', 'isothiocyanate')


def _name_iso_x_cyanate(features: Any, fg_key: str, suffix_word: str) -> Optional[str]:
    """Common implementation for isocyanate and isothiocyanate naming.

    SMARTS: [#6][NX2]=[CX2]=[OX1] or [#6][NX2]=[CX2]=[SX1]
    Match tuple: (R_carbon, N, C, O/S)

    Strategy: find the R group attached to N (the first atom in SMARTS match),
    name it, return 'R isocyanate' or 'R isothiocyanate'.
    """
    from rdkit import Chem

    mol = features.mol
    matches = features.functional_groups.get(fg_key, [])
    if not matches:
        return None

    # SMARTS gives (R_atom, N, C=cumulated, O/S)
    match = matches[0]
    r_atom_idx = match[0]  # The atom bonded to N (the R group start)
    n_idx = match[1]       # Nitrogen

    # Name the R group
    r_name = _name_r_group(mol, r_atom_idx, exclude_atoms={n_idx, match[2], match[3]})
    if not r_name:
        return None

    return f"{r_name} {suffix_word}"


def _name_r_group(mol, start_idx: int, exclude_atoms: set) -> Optional[str]:
    """Name an R group (substituent fragment) starting from start_idx.

    For simple alkyl chains: methyl, ethyl, propyl, butyl, ...
    For branched alkyls: isopropyl, tert-butyl, sec-butyl, isobutyl
    For phenyl: phenyl
    For benzyl: benzyl (if CH2-phenyl)

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index of the R group
        exclude_atoms: Atom indices to exclude (the functional group itself)

    Returns:
        R group name, or None if unable to name.
    """
    from rdkit import Chem
    from collections import deque

    # BFS to find all atoms in the R fragment
    visited = set()
    queue = deque([start_idx])
    frag_atoms = []

    while queue:
        idx = queue.popleft()
        if idx in visited or idx in exclude_atoms:
            continue
        visited.add(idx)
        frag_atoms.append(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in visited and nidx not in exclude_atoms:
                queue.append(nidx)

    if not frag_atoms:
        return None

    # Check for aromatic ring (phenyl)
    ring_info = mol.GetRingInfo()
    frag_set = set(frag_atoms)
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(frag_set) and len(ring) == 6:
            if all(mol.GetAtomWithIdx(r).GetIsAromatic() and
                   mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
                # Has a benzene ring
                non_ring_carbons = sum(
                    1 for i in frag_atoms
                    if i not in ring_set and mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                )
                if non_ring_carbons == 0:
                    return "phenyl"
                elif non_ring_carbons == 1:
                    return "benzyl"

    # Count carbons and check branching for alkyl name
    carbon_atoms = [i for i in frag_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
    carbon_count = len(carbon_atoms)

    if carbon_count == 0:
        return None

    # Check for branched alkyl patterns using the attachment point
    # The start_idx is the atom directly connected to the FG
    start_atom = mol.GetAtomWithIdx(start_idx)
    if start_atom.GetSymbol() == 'C':
        # Count carbon neighbors within the fragment
        c_neighbors_in_frag = [
            nbr.GetIdx() for nbr in start_atom.GetNeighbors()
            if nbr.GetIdx() in frag_set and nbr.GetSymbol() == 'C'
        ]

        if carbon_count == 3:
            # 3 carbons: could be isopropyl (CH(CH3)2) or propyl (CH2CH2CH3)
            if len(c_neighbors_in_frag) == 2:
                # Branched at attachment point: isopropyl
                return "isopropyl"

        elif carbon_count == 4:
            if len(c_neighbors_in_frag) == 3:
                # 3 branches at attachment point: tert-butyl
                return "tert-butyl"
            elif len(c_neighbors_in_frag) == 2:
                # Check for isobutyl (CH2-CH(CH3)2) vs sec-butyl (CH(CH3)(CH2CH3))
                # sec-butyl: attachment carbon has 1 methyl + 1 ethyl branch
                branch_sizes = []
                for nb_idx in c_neighbors_in_frag:
                    # Count carbons in this sub-branch
                    sub_visited = {start_idx}
                    sub_queue = deque([nb_idx])
                    sub_count = 0
                    while sub_queue:
                        a = sub_queue.popleft()
                        if a in sub_visited or a not in frag_set:
                            continue
                        sub_visited.add(a)
                        if mol.GetAtomWithIdx(a).GetSymbol() == 'C':
                            sub_count += 1
                        for nn in mol.GetAtomWithIdx(a).GetNeighbors():
                            if nn.GetIdx() not in sub_visited and nn.GetIdx() in frag_set:
                                sub_queue.append(nn.GetIdx())
                    branch_sizes.append(sub_count)

                branch_sizes.sort()
                if branch_sizes == [1, 2]:
                    return "sec-butyl"
            elif len(c_neighbors_in_frag) == 1:
                # Linear attachment but could be isobutyl
                # Check if there's branching further down
                next_c = c_neighbors_in_frag[0]
                next_atom = mol.GetAtomWithIdx(next_c)
                next_c_nbrs = [
                    nbr.GetIdx() for nbr in next_atom.GetNeighbors()
                    if nbr.GetIdx() in frag_set and nbr.GetIdx() != start_idx
                    and nbr.GetSymbol() == 'C'
                ]
                if len(next_c_nbrs) == 2:
                    return "isobutyl"

    # Default: linear alkyl name
    try:
        return get_alkyl_name(carbon_count)
    except (ValueError, KeyError):
        return None


# ============================================================================
# Boronic acid naming (functional class: "methylboronic acid")
# ============================================================================

def _name_boronic_acid(features: Any) -> Optional[str]:
    """Name boronic acid as 'Rboronic acid' (functional class naming).

    Pattern: R-B(OH)2
    SMARTS match: [#6][BX3]([OX2H])([OX2H]) gives (C, B, O, O)

    Simple cases: "methylboronic acid", "phenylboronic acid"
    Complex R: "(4-methylphenyl)boronic acid"

    Returns:
        Functional class name, or None.
    """
    from rdkit import Chem

    mol = features.mol
    matches = features.functional_groups.get('boronic_acid', [])
    if not matches:
        return None

    match = matches[0]
    # SMARTS: [#6][BX3]([OX2H])([OX2H])
    # match[0] = C attached to B, match[1] = B, match[2] = O, match[3] = O
    r_atom_idx = match[0]
    b_idx = match[1]
    o1_idx = match[2]
    o2_idx = match[3]

    # Name the R group
    r_name = _name_r_group(mol, r_atom_idx, exclude_atoms={b_idx, o1_idx, o2_idx})
    if not r_name:
        return None

    # Check if R name needs parentheses (contains locants/hyphens/spaces)
    # Simple names like "methyl", "phenyl" don't need parens
    # Complex names like "4-methylphenyl" do
    needs_parens = any(c in r_name for c in '-,') and r_name not in (
        'tert-butyl', 'sec-butyl'
    )

    if needs_parens:
        return f"({r_name})boronic acid"
    return f"{r_name}boronic acid"


# ============================================================================
# Carbamate naming (functional class: "ethyl carbamate")
# ============================================================================

def _name_carbamate(features: Any) -> Optional[str]:
    """Name carbamate as 'alkyl [N-substituted]carbamate' (functional class naming).

    Pattern: N-C(=O)-O-R
    SMARTS match: [NX3][CX3](=O)[OX2][#6] gives (N, C, O=, O-R, R)
    But we need the atoms at specific positions.

    Unsubstituted: "ethyl carbamate" (NH2 on N)
    N-monosubstituted: "ethyl N-methylcarbamate"
    N,N-disubstituted: "methyl N,N-dimethylcarbamate"

    Returns:
        Functional class name, or None.
    """
    from rdkit import Chem
    from collections import Counter

    mol = features.mol
    matches = features.functional_groups.get('carbamate', [])
    if not matches:
        return None

    # SMARTS: [NX3][CX3](=O)[OX2][#6]
    # Match gives (N, C_carbonyl, O_carbonyl, O_ether, R_atom)
    # But note: SMARTS [NX3][CX3](=O)[OX2][#6] has 5 atoms in pattern...
    # Actually, the SMARTS has mapped atoms: N(0), C(1), O=(implicit in =O), O(2 in [OX2]), C(3 in [#6])
    # Let's verify the actual match structure
    match = matches[0]

    # The SMARTS "[NX3][CX3](=O)[OX2][#6]" matches:
    # atom 0: N (the nitrogen)
    # atom 1: C (the carbonyl carbon)
    # atom 2: O (the carbonyl oxygen, from =O)
    # atom 3: O (the ether oxygen, from [OX2])
    # atom 4: C/# (the R group first atom, from [#6])
    # Wait, this depends on the SMARTS encoding. Let me check the actual match length.

    if len(match) < 4:
        return None

    n_idx = match[0]   # Nitrogen
    c_idx = match[1]   # Carbonyl carbon

    # Find the ether oxygen bonded to C (not the =O)
    c_atom = mol.GetAtomWithIdx(c_idx)
    o_ether_idx = None
    r_start_idx = None

    for nbr in c_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == n_idx:
            continue
        if nbr.GetSymbol() == 'O':
            bond = mol.GetBondBetweenAtoms(c_idx, nidx)
            if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                o_ether_idx = nidx
                # Find R attached to ether O
                for o_nbr in nbr.GetNeighbors():
                    if o_nbr.GetIdx() != c_idx:
                        r_start_idx = o_nbr.GetIdx()
                break

    if o_ether_idx is None or r_start_idx is None:
        return None

    # Exclude set: all carbamate core atoms
    carbamate_core = {n_idx, c_idx, o_ether_idx}
    # Also find and exclude the =O
    for nbr in c_atom.GetNeighbors():
        if nbr.GetSymbol() == 'O' and nbr.GetIdx() != o_ether_idx:
            carbamate_core.add(nbr.GetIdx())

    # Name the R group (on ether oxygen)
    r_name = _name_r_group(mol, r_start_idx, exclude_atoms=carbamate_core)
    if not r_name:
        return None

    # Check N-substitution
    n_atom = mol.GetAtomWithIdx(n_idx)
    n_subs = []
    for nbr in n_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx:
            continue
        if nbr.GetSymbol() == 'H':
            continue
        # Only count carbon-based substituents (not H)
        if nbr.GetAtomicNum() > 1:
            sub_name = _name_r_group(mol, nidx, exclude_atoms=carbamate_core)
            if sub_name:
                n_subs.append(sub_name)

    if not n_subs:
        # Unsubstituted: "ethyl carbamate"
        return f"{r_name} carbamate"

    # Build N-substitution prefix
    sub_counts = Counter(n_subs)
    n_prefix_parts = []
    for name in sorted(sub_counts.keys()):
        count = sub_counts[name]
        if count == 1:
            n_prefix_parts.append(f"N-{name}")
        else:
            mult = get_multiplier_prefix(count, name)
            n_prefix_parts.append(f"N,N-{mult}{name}")

    n_prefix = ",".join(n_prefix_parts)
    return f"{r_name} {n_prefix}carbamate"


def _try_name_urea(features: Any) -> Optional[str]:
    """Name urea derivatives as retained name with N-substitution.

    Pattern: N1-C(=O)-N2
    SMARTS match: [NX3][CX3](=O)[NX3] gives (N1, C, O, N2)

    Unsubstituted: "urea"
    N-monosubstituted: "N-methylurea"
    N,N-disubstituted (same N): "N,N-dimethylurea"
    N,N'-disubstituted (different N): "N,N'-dimethylurea"

    Returns:
        Retained name with N-substitution prefix, or None.
    """
    from rdkit import Chem
    from collections import Counter

    mol = features.mol
    matches = features.functional_groups.get('urea', [])
    if not matches:
        return None

    match = matches[0]
    if len(match) < 4:
        return None

    n1_idx = match[0]   # First nitrogen
    c_idx = match[1]    # Carbonyl carbon
    # match[2] = carbonyl oxygen
    n2_idx = match[3]   # Second nitrogen

    # Core atoms to exclude from R group naming
    c_atom = mol.GetAtomWithIdx(c_idx)
    urea_core = {n1_idx, c_idx, n2_idx}
    # Add carbonyl oxygen
    for nbr in c_atom.GetNeighbors():
        if nbr.GetSymbol() == 'O':
            urea_core.add(nbr.GetIdx())

    # Collect substituents on N1
    n1_subs = []
    n1_atom = mol.GetAtomWithIdx(n1_idx)
    for nbr in n1_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx or nbr.GetAtomicNum() <= 1:
            continue
        sub_name = _name_r_group(mol, nidx, exclude_atoms=urea_core)
        if sub_name:
            n1_subs.append(sub_name)

    # Collect substituents on N2
    n2_subs = []
    n2_atom = mol.GetAtomWithIdx(n2_idx)
    for nbr in n2_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx or nbr.GetAtomicNum() <= 1:
            continue
        sub_name = _name_r_group(mol, nidx, exclude_atoms=urea_core)
        if sub_name:
            n2_subs.append(sub_name)

    # No substituents: plain "urea"
    if not n1_subs and not n2_subs:
        return "urea"

    # Build N-substitution prefix with N/N' locants
    # When only one N is substituted, it always gets unprimed "N"
    # When both Ns are substituted, they get "N" and "N'"
    # Ensure the more-substituted (or alphabetically first) N gets unprimed "N"
    if n1_subs and not n2_subs:
        # Only N1 substituted -> assign N1 as "N"
        first_subs, second_subs = n1_subs, n2_subs
    elif n2_subs and not n1_subs:
        # Only N2 substituted -> assign N2 as "N"
        first_subs, second_subs = n2_subs, n1_subs
    else:
        # Both substituted -> N1 as "N", N2 as "N'"
        first_subs, second_subs = n1_subs, n2_subs

    tagged_subs = []  # list of (locant, sub_name) pairs
    for s in first_subs:
        tagged_subs.append(("N", s))
    for s in second_subs:
        tagged_subs.append(("N'", s))

    return _build_n_substituted_name(tagged_subs, "urea")


def _try_name_guanidine(features: Any) -> Optional[str]:
    """Name guanidine derivatives as retained name with N-substitution.

    Pattern: N1-C(=N3)-N2  (three nitrogen atoms)
    SMARTS match: [NX3][CX3](=[NX2])[NX3] gives (N_single1, C, N_double, N_single2)

    Unsubstituted: "guanidine"
    N-monosubstituted: "N-methylguanidine"
    N,N-disubstituted (same N): "N,N-dimethylguanidine"

    The =NH nitrogen gets unprimed N locant.
    The two -NH2 nitrogens get N' and N'' locants.

    Returns:
        Retained name with N-substitution prefix, or None.
    """
    from rdkit import Chem
    from collections import Counter

    mol = features.mol
    matches = features.functional_groups.get('guanidine', [])
    if not matches:
        return None

    match = matches[0]
    if len(match) < 4:
        return None

    # SMARTS: [NX3][CX3](=[NX2])[NX3]
    # match[0] = N_single1 (single-bonded nitrogen)
    # match[1] = C_central
    # match[2] = N_double (double-bonded =NH nitrogen)
    # match[3] = N_single2 (single-bonded nitrogen)
    n_single1_idx = match[0]
    c_idx = match[1]
    n_double_idx = match[2]
    n_single2_idx = match[3]

    # Core atoms: all three nitrogens and the central carbon
    guanidine_core = {n_single1_idx, c_idx, n_double_idx, n_single2_idx}

    # Collect substituents on the =NH nitrogen (N_double -> "N" locant)
    n_double_subs = []
    n_double_atom = mol.GetAtomWithIdx(n_double_idx)
    for nbr in n_double_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx or nbr.GetAtomicNum() <= 1:
            continue
        sub_name = _name_r_group(mol, nidx, exclude_atoms=guanidine_core)
        if sub_name:
            n_double_subs.append(sub_name)

    # Collect substituents on N_single1 (-> "N'" locant)
    n_single1_subs = []
    n_single1_atom = mol.GetAtomWithIdx(n_single1_idx)
    for nbr in n_single1_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx or nbr.GetAtomicNum() <= 1:
            continue
        sub_name = _name_r_group(mol, nidx, exclude_atoms=guanidine_core)
        if sub_name:
            n_single1_subs.append(sub_name)

    # Collect substituents on N_single2 (-> "N''" locant)
    n_single2_subs = []
    n_single2_atom = mol.GetAtomWithIdx(n_single2_idx)
    for nbr in n_single2_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx or nbr.GetAtomicNum() <= 1:
            continue
        sub_name = _name_r_group(mol, nidx, exclude_atoms=guanidine_core)
        if sub_name:
            n_single2_subs.append(sub_name)

    # No substituents: plain "guanidine"
    if not n_double_subs and not n_single1_subs and not n_single2_subs:
        return "guanidine"

    # Build N-substitution prefix
    # Collect all substituted nitrogens with their substituents
    # For guanidine: use N, N', N'' to distinguish the three nitrogens
    # But for mono-substitution, just use "N" (no primes needed)
    substituted_nitrogens = []
    if n_single1_subs:
        substituted_nitrogens.append(n_single1_subs)
    if n_single2_subs:
        substituted_nitrogens.append(n_single2_subs)
    if n_double_subs:
        substituted_nitrogens.append(n_double_subs)

    if len(substituted_nitrogens) == 1:
        # Only one nitrogen is substituted -> all get unprimed "N"
        tagged_subs = [("N", s) for s in substituted_nitrogens[0]]
    else:
        # Multiple nitrogens substituted -> assign N, N', N''
        locant_labels = ["N", "N'", "N''"]
        tagged_subs = []
        for i, subs in enumerate(substituted_nitrogens):
            label = locant_labels[i] if i < len(locant_labels) else f"N{''.join(['`'] * i)}"
            for s in subs:
                tagged_subs.append((label, s))

    return _build_n_substituted_name(tagged_subs, "guanidine")


def _build_n_substituted_name(tagged_subs: list, base_name: str) -> str:
    """Build a name like 'N-methyl{base}' or 'N,N'-dimethyl{base}' from tagged substituents.

    Args:
        tagged_subs: list of (locant, sub_name) pairs, e.g. [("N", "methyl"), ("N'", "ethyl")]
        base_name: The retained name, e.g. "urea" or "guanidine"

    Returns:
        Complete name with N-substitution prefix.
    """
    from collections import Counter

    if not tagged_subs:
        return base_name

    # Group by substituent name to apply multipliers
    # e.g., [("N", "methyl"), ("N'", "methyl")] -> "N,N'-dimethylurea"
    # e.g., [("N", "methyl"), ("N'", "ethyl")] -> "N-ethyl-N'-methylurea" (alphabetical)

    # Build mapping: sub_name -> list of locants
    sub_locants = {}
    for locant, name in tagged_subs:
        if name not in sub_locants:
            sub_locants[name] = []
        sub_locants[name].append(locant)

    # Build prefix parts, sorted alphabetically by substituent name
    prefix_parts = []
    for name in sorted(sub_locants.keys()):
        locants = sub_locants[name]
        count = len(locants)
        locant_str = ",".join(locants)
        if count == 1:
            prefix_parts.append(f"{locant_str}-{name}")
        else:
            mult = get_multiplier_prefix(count, name)
            prefix_parts.append(f"{locant_str}-{mult}{name}")

    prefix = "-".join(prefix_parts)
    return f"{prefix}{base_name}"


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
    Check if a molecule contains a complex ring system (bicyclo, spiro, fused, or polycyclic).

    Complex ring systems require special naming rules beyond simple cycloalkanes.
    This function is used for early routing decision in assemble_name().

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule contains bicyclo, spiro, fused, bridged-fused, or polycyclic-bridged system

    Note:
        Simple monocyclic rings return False.
        This function just detects if ANY complex ring system exists;
        the actual classification is done in _classify_complex_ring.
    """
    # Check bridged-fused (fused core + additional bridges)
    if detect_bridged_fused(mol):
        return True

    # Check bicyclo (bridged bicyclic with 2 rings)
    if is_bicyclo_system(mol):
        return True

    # Check polycyclic-bridged (tricyclo+ pure bridged systems)
    if is_polycyclic_system(mol):
        return True

    # Check fused rings (ortho-fused or ortho-peri-fused)
    fused_type = classify_fused_system(mol)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return True

    # Check spiro (two rings sharing one atom)
    if is_spiro_system(mol):
        return True

    return False


def _classify_complex_ring(mol) -> str:
    """
    Classify a complex ring system by type.

    Args:
        mol: RDKit Mol object

    Returns:
        Classification string:
        - 'bridged-fused': Mixed fused + bridged system (e.g., 1,4-methanonaphthalene)
        - 'bicyclo': Bridged bicyclic system (e.g., norbornane)
        - 'polycyclic-bridged': Higher polycyclic bridged system (tricyclo+, e.g., adamantane)
        - 'spiro': Spiro system (rings share one atom)
        - 'ortho-fused': Ortho-fused system (rings share one edge)
        - 'ortho-peri-fused': Complex fused system (e.g., perylene, coronene)
        - 'simple': Not a complex ring system

    Note:
        Priority order is CRITICAL:
        1. bridged-fused FIRST (should not fall through to bicyclo/polycyclic)
        2. bicyclo (2-ring bridged)
        3. polycyclic-bridged (tricyclo+ pure bridged - adamantane, cubane)
           BEFORE fused because classify_fused_system incorrectly flags
           bridged systems as "ortho-peri-fused"
        4. fused (ortho-fused/ortho-peri-fused for PAHs)
        5. spiro

        The key insight: is_polycyclic_system() correctly identifies TRUE bridged
        polycyclic systems (adamantane) vs fused systems (perylene), while
        classify_fused_system() incorrectly flags adamantane as ortho-peri-fused.
        So we check is_polycyclic_system BEFORE classify_fused_system.
    """
    # Check bridged-fused FIRST (fused core + bridges)
    # Must come before bicyclo/polycyclic to avoid misclassification
    if detect_bridged_fused(mol):
        return 'bridged-fused'

    # Check bicyclo (2-ring bridged)
    if is_bicyclo_system(mol):
        return 'bicyclo'

    # Check polycyclic-bridged (tricyclo+ bridged systems) BEFORE fused
    # This is critical: is_polycyclic_system correctly distinguishes
    # TRUE bridged systems (adamantane) from fused systems (perylene)
    # using the _is_purely_fused() check internally
    if is_polycyclic_system(mol):
        return 'polycyclic-bridged'

    # Check fused (ortho-fused or ortho-peri-fused)
    # Only reaches here if NOT a bridged polycyclic
    fused_type = classify_fused_system(mol)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return fused_type

    # Check spiro
    if is_spiro_system(mol):
        return 'spiro'

    return 'simple'


def _assemble_complex_ring_name(mol, features) -> Optional[str]:
    """
    Assemble IUPAC name for a complex ring system.

    Routes to appropriate naming function based on ring classification:
    - Bridged-fused: FR-8 nomenclature (e.g., 1,4-methanonaphthalene)
    - Bicyclo: bicyclo[x.y.z]alkane format (e.g., bicyclo[2.2.1]heptane)
    - Polycyclic-bridged: von Baeyer format (e.g., tricyclo[3.3.1.1(3,7)]decane)
    - Spiro: spiro[a.b]alkane format (e.g., spiro[4.5]decane)
    - Fused: retained names or systematic fusion descriptors

    Args:
        mol: RDKit Mol object
        features: MolecularFeatures object (for substituents, stereo, etc.)

    Returns:
        Complete IUPAC name, or None if naming fails

    Note:
        Supports complete naming with substituents, unsaturation, and stereo for all types.
    """
    import logging

    ring_type = _classify_complex_ring(mol)

    try:
        if ring_type == 'bridged-fused':
            # FR-8 nomenclature for bridged fused systems
            name = name_bridged_fused_system(mol)
            if name:
                return name
            logging.warning("Bridged-fused naming failed for molecule")
            return None

        elif ring_type == 'bicyclo':
            # Complete bicyclo naming with substituents, unsaturation, stereo
            name = _assemble_complete_bicyclo_name(mol, features)
            if name:
                return name
            logging.warning("Bicyclo naming failed for molecule")
            return None

        elif ring_type == 'polycyclic-bridged':
            # Von Baeyer naming for tricyclo+ systems (e.g., adamantane)
            name = name_polycyclic_complete(mol, features)
            if name:
                return name
            logging.warning("Polycyclic-bridged naming failed for molecule")
            return None

        elif ring_type == 'spiro':
            # Spiro naming (spiro[4.5]decane, etc.)
            name = name_spiro_system(mol)
            if name:
                return name
            logging.warning("Spiro naming failed for molecule")
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


def _assemble_ring_with_ester_prefixes(features, exocyclic_esters) -> Optional[str]:
    """
    Assemble name for a ring parent with exocyclic ester substituents as acyloxy prefixes.

    When a molecule has a ring (cyclohexane, benzene, etc.) with ester groups
    attached to it, the ring is the parent and esters become acyloxy prefixes.

    Examples:
        CC(=O)OC1CCCCC1 -> acetyloxycyclohexane
        CC(=O)Oc1ccccc1 -> acetyloxybenzene (phenyl acetate)

    Args:
        features: MolecularFeatures object
        exocyclic_esters: List of dicts from detect_exocyclic_esters()

    Returns:
        IUPAC name, or None if naming fails
    """
    mol = features.mol
    if not exocyclic_esters:
        return None

    # Determine ring parent name by finding the ring containing the attachment atom
    ring_parent = None
    attach_atom = exocyclic_esters[0]['ring_attach_atom_idx']

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    for ring in atom_rings:
        if attach_atom not in ring:
            continue

        ring_size = len(ring)
        ring_set = set(ring)

        # Check if it's an aromatic 6-membered all-carbon ring (benzene)
        is_aromatic_6 = (
            ring_size == 6 and
            all(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring) and
            all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        )
        if is_aromatic_6:
            ring_parent = "benzene"
            break

        # Check for heterocyclic ring
        has_heteroatom = any(
            mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
            for idx in ring
        )
        if has_heteroatom:
            from ..rules.heterocycles import name_heterocycle
            ring_parent = name_heterocycle(mol, tuple(ring))
            break

        # Carbocyclic ring
        all_carbon = all(
            mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            for idx in ring
        )
        if all_carbon:
            try:
                stem = get_chain_prefix(ring_size)
                if stem:
                    ring_parent = f"cyclo{stem}ane"
            except (ValueError, KeyError):
                pass  # ring_parent stays None
        break

    if not ring_parent:
        return None

    # Build acyloxy prefix(es)
    # Group by prefix name for multipliers
    from collections import defaultdict as _dd
    prefix_groups: Dict[str, List[int]] = _dd(list)

    for ester_info in exocyclic_esters:
        prefix = ester_info['acyloxy_prefix']
        prefix_groups[prefix].append(ester_info['ring_attach_atom_idx'])

    # Determine total number of ester substituents on the ring
    total_esters = sum(len(v) for v in prefix_groups.values())

    # Compute ring locants for attachment atoms when multiple esters present
    ring_atom_to_locant: Dict[int, int] = {}
    if total_esters > 1:
        # Find the ring containing the attachment atoms and compute numbering
        ring_atoms = None
        for r in ring_info.AtomRings():
            if attach_atom in r:
                ring_atoms = r
                break
        if ring_atoms:
            # Collect all attachment atom indices
            all_attach = []
            for ester_info in exocyclic_esters:
                all_attach.append(ester_info['ring_attach_atom_idx'])
            # Use _number_ring_from_attachment for IUPAC-optimal numbering
            # Try all attachment atoms as position 1, pick lowest locant set
            best_mapping = None
            best_locants = None
            for start_idx in all_attach:
                candidates = _number_ring_from_attachment(mol, ring_atoms, start_idx)
                if not candidates:
                    continue
                for numbering in candidates:
                    # numbering is {locant: atom_idx}, invert it
                    inv = {v: k for k, v in numbering.items()}
                    locant_set = sorted(inv.get(a, 999) for a in all_attach)
                    if best_locants is None or locant_set < best_locants:
                        best_locants = locant_set
                        best_mapping = inv
            if best_mapping:
                ring_atom_to_locant = best_mapping

    # Build prefix parts with locants
    prefix_parts = []
    for prefix_name, attach_atoms_list in sorted(prefix_groups.items()):
        count = len(attach_atoms_list)
        if total_esters == 1:
            # Single ester on ring: no locant needed
            prefix_parts.append(prefix_name)
        elif count == 1:
            # One instance of this prefix but multiple esters total: need locant
            locant = ring_atom_to_locant.get(attach_atoms_list[0], 1)
            prefix_parts.append(f"{locant}-({prefix_name})")
        else:
            # Multiple instances of same prefix: locants + bis/tris multiplier
            locants = sorted(ring_atom_to_locant.get(a, 1) for a in attach_atoms_list)
            locant_str = ",".join(str(loc) for loc in locants)
            # Acyloxy names are complex (contain "yloxy"), use bis/tris
            multiplier = get_multiplier_prefix(count, prefix_name)
            prefix_parts.append(f"{locant_str}-{multiplier}({prefix_name})")

    # Sort alphabetically
    prefix_parts.sort(key=lambda x: alpha_sort_key(x))

    # Join prefixes with hyphens
    prefix_str = "-".join(prefix_parts)

    # Ensure hyphen before ring parent when prefix ends with letter/paren
    if prefix_str and ring_parent:
        if prefix_str[-1] == ')' or prefix_str[-1].isalpha():
            return f"{prefix_str}{ring_parent}"
        else:
            return f"{prefix_str}{ring_parent}"
    return f"{prefix_str}{ring_parent}"


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
    if carbon_count in stems:
        return stems[carbon_count]
    from ..data.chain_names import get_chain_prefix
    return get_chain_prefix(carbon_count)


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
                # Name non-carbon substituents directly (halogens, hydroxy, amino)
                first_atom = sub_info.get('first_atom')
                if first_atom is not None:
                    atom = mol.GetAtomWithIdx(first_atom)
                    symbol = atom.GetSymbol()
                    halogen_names = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
                    if symbol in halogen_names:
                        sub_groups[halogen_names[symbol]].append(locant)
                    elif symbol == 'O' and atom.GetTotalNumHs() >= 1:
                        sub_groups['hydroxy'].append(locant)
                    elif symbol == 'N' and atom.GetTotalNumHs() >= 2:
                        sub_groups['amino'].append(locant)
                continue  # Still skip alkyl naming path

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
        base_name = name_amide(features.mol, amide_atoms)
        if base_name:
            # Add non-principal group prefixes (halogens, hydroxy, etc.)
            prefixes = _generate_prefixes(features)
            if prefixes:
                prefix_parts = []
                for p in sorted(prefixes, key=lambda x: alpha_sort_key(x.text)):
                    if p.locants:
                        loc_str = ",".join(str(l) for l in p.locants)
                        prefix_parts.append(f"{loc_str}-{p.text}")
                    else:
                        prefix_parts.append(p.text)
                if prefix_parts:
                    prefix_str = "-".join(prefix_parts)
                    return f"{prefix_str}{base_name}"
            return base_name

    # Fallback
    return "amide"


def _assemble_amine_name(features: Any, style: str) -> Optional[str]:
    """
    Assemble name for secondary/tertiary amines with N-alkyl prefixes.

    For secondary amines: N-alkyl + parent amine (N-ethylethanamine)
    For tertiary amines: N,N-dialkyl + parent amine (N,N-dimethylethanamine)

    Returns None if unable to assemble (falls through to general naming).
    """
    mol = features.mol
    pg_atoms = features.principal_group_atoms
    if not pg_atoms:
        return None

    match = pg_atoms[0]
    # Find nitrogen atom in the match
    n_idx = None
    for idx in match:
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'N':
            n_idx = idx
            break
    if n_idx is None:
        return None

    nitrogen = mol.GetAtomWithIdx(n_idx)
    chain_set = set(features.principal_chain) if features.principal_chain else set()

    # Find N-substituents: carbon neighbors of N that are NOT on the principal chain
    from collections import deque
    n_subs = []
    for nbr in nitrogen.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in chain_set:
            continue
        if nbr.GetSymbol() == 'H':
            continue
        # BFS to get the substituent fragment
        visited = set()
        queue = deque([nbr_idx])
        frag = []
        while queue:
            a = queue.popleft()
            if a in visited or a == n_idx:
                continue
            visited.add(a)
            frag.append(a)
            for nn in mol.GetAtomWithIdx(a).GetNeighbors():
                if nn.GetIdx() not in visited and nn.GetIdx() != n_idx:
                    queue.append(nn.GetIdx())
        if frag:
            cc = sum(1 for i in frag if mol.GetAtomWithIdx(i).GetSymbol() == 'C')
            # Check for aromatic rings
            has_phenyl = False
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                if all(r in set(frag) for r in ring) and len(ring) == 6:
                    if all(mol.GetAtomWithIdx(r).GetIsAromatic() and
                           mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
                        non_ring = cc - 6
                        if non_ring == 0:
                            n_subs.append("phenyl")
                            has_phenyl = True
                            break
            if not has_phenyl and cc > 0:
                try:
                    n_subs.append(get_alkyl_name(cc))
                except (ValueError, KeyError):
                    pass

    if not n_subs:
        return None  # No N-substituents found, use general path

    # Build N-prefix (same logic as amide N-substitution)
    from collections import Counter
    sub_counts = Counter(n_subs)
    n_prefix_parts = []
    for name in sorted(sub_counts.keys()):
        count = sub_counts[name]
        if count == 1:
            n_prefix_parts.append(f"N-{name}")
        else:
            mult = SIMPLE_MULTIPLIERS.get(count, str(count))
            n_prefix_parts.append(f"N,{'N,' * (count - 1)}{mult}{name}")

    n_prefix = "-".join(n_prefix_parts)

    # Get base amine name from the general assembly
    # Build name using standard fragments
    fragments = []
    if features.principal_chain:
        parent = _generate_chain_parent(features)
    elif features.ring_systems:
        parent = _generate_ring_parent(features)
    else:
        return None

    fragments.append(parent)

    if features.principal_group:
        suffix = _generate_suffix(features)
        if suffix:
            fragments.append(suffix)

    # Generate non-N prefixes (halogens, hydroxy, etc.)
    other_prefixes = _generate_prefixes(features)
    fragments.extend(other_prefixes)

    # Generate stereodescriptors
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)

    base_name = _assemble_fragments(fragments, style)

    # Prepend N-prefix
    if base_name:
        return f"{n_prefix}{base_name}"

    return None


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

    try:
        stem = get_chain_prefix(ring_size)
    except (ValueError, KeyError):
        return "carbonitrile"  # Fallback for invalid ring size

    if not stem:
        return "carbonitrile"  # Guard against empty stem producing 'cycloane'

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
    stem = get_chain_prefix(chain_length)

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

    # Guard: ring_size must produce a valid stem; otherwise return None-safe placeholder
    try:
        stem = get_chain_prefix(ring_size)
    except (ValueError, KeyError):
        # Invalid ring size (0, negative, etc.) -- return safe placeholder
        return NameFragment(text="cyclo", fragment_type="parent")

    # Guard: stem must be non-empty to avoid generating 'cycloane'
    if not stem:
        return NameFragment(text="cyclo", fragment_type="parent")

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
    fg_count = len(features.principal_group_atoms) if features.principal_group_atoms else 1

    # Validate suffix count against parent capacity:
    # The number of suffix groups cannot exceed the number of atoms in the parent
    # structure (chain length or ring size). E.g., ethane (2C) cannot have tetraol.
    if features.principal_chain:
        max_capacity = len(features.principal_chain)
    elif getattr(features, 'oriented_ring', None):
        max_capacity = len(features.oriented_ring)
    else:
        max_capacity = fg_count  # no constraint if we can't determine parent size

    if fg_count > max_capacity:
        fg_count = max_capacity

    if features.principal_chain and features.atom_to_locant and features.principal_group_atoms:
        fg_locants = get_functional_group_locants(
            features.principal_chain,
            features.principal_group_atoms,
            features.atom_to_locant,
            mol=features.mol
        )

        # Deduplicate locants (overlapping SMARTS can produce duplicates)
        fg_locants = sorted(set(fg_locants))

        # Further validate: count should match unique locants when locants exist
        if fg_locants:
            fg_count = len(fg_locants)

        # Terminal groups: locant is implicitly 1, do NOT include in name
        if fg_name in TERMINAL_GROUPS:
            # For terminal groups, we don't include the locant
            locants = ()
        else:
            # For non-terminal groups (alcohol, ketone), include locants
            locants = tuple(fg_locants)

    elif (not features.principal_chain
          and getattr(features, 'oriented_ring', None)
          and features.principal_group_atoms
          and fg_name not in TERMINAL_GROUPS):
        # Ring compounds: build idx_to_locant from oriented_ring and compute
        # suffix locants.  This mirrors the logic in _get_fg_locants() but
        # uses get_functional_group_locants for consistency.
        oriented_ring = features.oriented_ring
        ring_idx_to_locant = {
            atom_idx: pos + 1
            for pos, atom_idx in enumerate(oriented_ring)
        }
        # Map FG atoms to ring locants
        fg_locants = []
        mol = features.mol
        for match in features.principal_group_atoms:
            found = False
            match_set = set(match)
            ring_set_local = set(ring_idx_to_locant.keys())
            # First pass: prefer a ring C bonded to a non-ring atom in
            # the same FG match (the characteristic heteroatom, e.g.,
            # the C in C=O for ketone, C in C-OH for alcohol).
            for atom_idx in match:
                if atom_idx in ring_idx_to_locant:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        # Check if bonded to a non-ring FG atom
                        has_fg_hetero = any(
                            nbr.GetIdx() in match_set and nbr.GetIdx() not in ring_set_local
                            for nbr in atom.GetNeighbors()
                        )
                        if has_fg_hetero:
                            fg_locants.append(ring_idx_to_locant[atom_idx])
                            found = True
                            break
            if not found:
                # Second pass: any ring C in the match
                for atom_idx in match:
                    if atom_idx in ring_idx_to_locant:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.GetSymbol() == 'C':
                            fg_locants.append(ring_idx_to_locant[atom_idx])
                            found = True
                            break
            if not found:
                # Fallback: any atom in match on the ring
                for atom_idx in match:
                    if atom_idx in ring_idx_to_locant:
                        fg_locants.append(ring_idx_to_locant[atom_idx])
                        found = True
                        break
            if not found:
                # Try neighbors
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in ring_idx_to_locant:
                            fg_locants.append(ring_idx_to_locant[nbr.GetIdx()])
                            found = True
                            break
                    if found:
                        break

        fg_locants = sorted(set(fg_locants))
        if fg_locants:
            fg_count = len(fg_locants)
            locants = tuple(fg_locants)

    # Final safety: reconcile multiplier count with actual locants
    if locants:
        from ..rules.locant_validation import reconcile_multiplier_count
        fg_count = reconcile_multiplier_count(fg_count, list(locants))

    return NameFragment(
        text=suffix_text,
        locants=locants,
        fragment_type="suffix",
        count=fg_count,
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
    # When chain_is_parent, skip FGs on ring atoms (already in ring substituent name)
    ring_atom_set = set()
    if getattr(features, 'chain_is_parent', False):
        for rg in getattr(features, 'ring_substituents_as_groups', []):
            ring_atom_set.update(rg)

    for fg_name, matches in features.functional_groups.items():
        if fg_name == features.principal_group:
            continue

        # Filter out FGs on ring atoms when chain is parent
        if ring_atom_set:
            filtered = []
            for match in matches:
                # Skip FG if its carbon anchor is on a ring substituent
                # Check: is any carbon in the match a ring atom?
                on_ring = any(
                    a in ring_atom_set for a in match
                    if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                )
                if not on_ring:
                    filtered.append(match)
            matches = filtered

        prefix_text = get_prefix(fg_name)
        if prefix_text and matches:
            count = len(matches)

            # Compute locants by mapping FG anchor atoms to chain positions
            fg_locants = _get_fg_locants(features, fg_name, matches)

            # Reconcile multiplier count with actual locants found
            if fg_locants:
                from ..rules.locant_validation import reconcile_multiplier_count
                count = reconcile_multiplier_count(count, fg_locants)

            if count > 1:
                # Add multiplier
                multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
                prefix_text = f"{multiplier}{prefix_text}"

            # Omit locants when they are trivially unambiguous:
            # - Chain length 1 (methane derivatives): only position 1 exists
            # - Single FG at position 1 on a chain with no principal group
            chain = getattr(features, 'principal_chain', [])
            chain_len = len(chain)
            omit_locants = False
            if chain_len == 1:
                omit_locants = True
            elif (chain_len > 0 and features.principal_group is None
                  and count == 1 and fg_locants == [1]):
                omit_locants = True

            prefixes.append(NameFragment(
                text=prefix_text,
                locants=tuple(sorted(fg_locants)) if (fg_locants and not omit_locants) else (),
                fragment_type="prefix"
            ))

    # --- Merge duplicate prefix names ---
    # If the same base prefix name appears multiple times (from different sources),
    # merge them into a single entry with combined count and appropriate multiplier.
    # This prevents stacking like "dihydroxyhydroxy" -> should be "trihydroxy".
    prefixes = _merge_duplicate_prefixes(prefixes)

    return prefixes


def _get_fg_locants(features, fg_name: str, matches: list) -> list:
    """Map FG match atom indices to IUPAC chain or ring locants.

    For each FG match, finds the anchor atom (the carbon the FG is attached to,
    or the heteroatom itself for groups like fluoro/hydroxy) and maps it to
    the corresponding position in the principal chain or ring.

    Returns:
        List of integer locants (1-indexed), one per match.
    """
    chain = getattr(features, 'principal_chain', [])
    oriented_ring = getattr(features, 'oriented_ring', None)
    mol = features.mol

    if not chain and not oriented_ring:
        return []

    # Build atom_idx -> locant map
    idx_to_locant = {}
    if chain:
        for pos, atom_idx in enumerate(chain):
            idx_to_locant[atom_idx] = pos + 1  # 1-indexed
    elif oriented_ring:
        for pos, atom_idx in enumerate(oriented_ring):
            idx_to_locant[atom_idx] = pos + 1

    locants = []
    for match in matches:
        # Find the anchor atom in the chain/ring
        # For halogens, amino, hydroxy: match is (heteroatom, carbon_anchor) or similar
        # Try each atom in the match to find one in the chain
        found = False
        for atom_idx in match:
            if atom_idx in idx_to_locant:
                locants.append(idx_to_locant[atom_idx])
                found = True
                break
        if not found:
            # Try neighbors of match atoms
            for atom_idx in match:
                atom = mol.GetAtomWithIdx(atom_idx)
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in idx_to_locant:
                        locants.append(idx_to_locant[nbr.GetIdx()])
                        found = True
                        break
                if found:
                    break

    return locants


def _merge_duplicate_prefixes(prefixes: List[NameFragment]) -> List[NameFragment]:
    """
    Merge duplicate prefix names into single entries with combined locants and multipliers.

    This prevents stacking like 'dihydroxyhydroxy' (from two sources each detecting hydroxy)
    by combining them into 'trihydroxy' with merged locants.

    The merge works by:
    1. Extracting the base name from each prefix (stripping locants and multipliers)
    2. Grouping by base name
    3. Combining locants and recalculating multiplier

    Args:
        prefixes: List of NameFragment prefix objects

    Returns:
        Deduplicated list of NameFragment prefix objects
    """
    import re

    if len(prefixes) <= 1:
        return prefixes

    # Extract base name from formatted prefix text
    # Examples: "2-hydroxy" -> "hydroxy", "3,4-dihydroxy" -> "hydroxy",
    #           "methyl" -> "methyl", "2,2-dimethyl" -> "methyl"
    def _extract_base_name(text: str) -> str:
        """Strip locants and multipliers to get the base substituent name."""
        # Strip leading locants (digits, commas, hyphens at start)
        stripped = re.sub(r'^[\d,]+-', '', text)
        # Strip multiplicative prefix
        for mult in sorted(SIMPLE_MULTIPLIERS.values(), key=len, reverse=True):
            if stripped.startswith(mult):
                remainder = stripped[len(mult):]
                if remainder:
                    return remainder
        for mult in sorted(COMPLEX_MULTIPLIERS.values(), key=len, reverse=True):
            if stripped.startswith(mult):
                remainder = stripped[len(mult):]
                # Complex multipliers have parenthesized names: tetrakis(methyl) -> methyl
                if remainder.startswith('(') and ')' in remainder:
                    return remainder[1:remainder.index(')')]
                if remainder:
                    return remainder
        return stripped

    # Group by base name
    groups: Dict[str, List[NameFragment]] = defaultdict(list)
    for prefix in prefixes:
        base = _extract_base_name(prefix.text)
        groups[base].append(prefix)

    # Rebuild prefixes, merging duplicates
    merged = []
    for base_name, group in groups.items():
        if len(group) == 1:
            # No duplicates, keep as-is
            merged.append(group[0])
            continue

        # Merge: combine all locants and recalculate count
        all_locants = []
        for frag in group:
            all_locants.extend(frag.locants)
        all_locants = tuple(sorted(set(all_locants)))

        # Calculate total count from all sources
        # Count = number of locants if we have them, otherwise sum of individual counts
        total_count = len(all_locants) if all_locants else sum(
            # Estimate count from multiplier in text
            _count_from_prefix(frag.text, base_name) for frag in group
        )

        if total_count <= 0:
            total_count = len(group)  # Fallback: one per fragment

        # Rebuild the formatted prefix
        if total_count > 1:
            multiplier = SIMPLE_MULTIPLIERS.get(total_count, str(total_count))
            if all_locants:
                locant_str = ",".join(str(l) for l in all_locants)
                text = f"{locant_str}-{multiplier}{base_name}"
            else:
                text = f"{multiplier}{base_name}"
        else:
            if all_locants:
                locant_str = ",".join(str(l) for l in all_locants)
                text = f"{locant_str}-{base_name}"
            else:
                text = base_name

        merged.append(NameFragment(
            text=text,
            locants=all_locants,
            fragment_type="prefix"
        ))

    return merged


def _count_from_prefix(text: str, base_name: str) -> int:
    """Estimate the count from a formatted prefix string."""
    import re
    # Strip locants
    stripped = re.sub(r'^[\d,]+-', '', text)
    # Check for multiplier before base name
    for count, mult in sorted(SIMPLE_MULTIPLIERS.items(), key=lambda x: len(x[1]), reverse=True):
        if stripped.startswith(mult) and stripped[len(mult):] == base_name:
            return count
    return 1


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

    chain_set = set(features.principal_chain)

    for ring_atoms in ring_groups:
        ring_atom_set = set(ring_atoms)
        # Get base substituent name (phenyl, cyclohexyl, etc.)
        base_name = get_ring_substituent_name(features.mol, ring_atoms)

        # Find which chain position the ring attaches to
        try:
            locant = get_ring_attachment_locant(
                features.mol,
                ring_atoms,
                features.principal_chain,
                features.atom_to_locant
            )
        except ValueError:
            continue

        # Detect substituents on the ring itself
        sub_name = _build_substituted_ring_name(
            features.mol, ring_atoms, chain_set, base_name
        )

        ring_sub_groups[sub_name].append(locant)

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


def _build_substituted_ring_name(
    mol, ring_atoms, chain_set: set, base_name: str
) -> str:
    """
    Build ring substituent name including substituents on the ring.

    For example, a benzene ring with two OH groups becomes
    "(3,4-dihydroxyphenyl)" instead of just "phenyl".

    Only handles common cases: halogens, hydroxy, methoxy, amino, alkyl
    on benzene-like rings. Falls back to bare base_name otherwise.
    """
    from rdkit import Chem

    ring_atom_set = set(ring_atoms)

    # Find attachment point (ring atom bonded to chain atom)
    attachment_idx = None
    for ra in ring_atoms:
        atom = mol.GetAtomWithIdx(ra)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in chain_set:
                attachment_idx = ra
                break
        if attachment_idx is not None:
            break

    if attachment_idx is None:
        return base_name

    # Only handle 6-membered carbocyclic rings (benzene-like) for now
    if len(ring_atoms) != 6:
        return base_name
    # Check it's all carbons
    if not all(mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in ring_atoms):
        return base_name

    # Try both numbering directions and pick lowest locant set
    numberings = _number_ring_from_attachment(mol, ring_atoms, attachment_idx)
    if numberings is None:
        return base_name

    # If single numbering returned (dict), wrap in list
    if isinstance(numberings, dict):
        numberings = [numberings]

    best_sub_groups = None
    best_locant_set = None

    for ring_order in numberings:
        if len(ring_order) != 6:
            continue
        sub_groups = _detect_ring_substituents(
            mol, ring_order, ring_atom_set, chain_set
        )
        if not sub_groups:
            continue
        # Collect all locants for comparison
        all_locs = sorted(loc for locs in sub_groups.values() for loc in locs)
        if best_locant_set is None or all_locs < best_locant_set:
            best_locant_set = all_locs
            best_sub_groups = sub_groups

    if not best_sub_groups:
        # No substituents on ring, or couldn't detect - check if any exist
        # Try with first numbering just to detect
        for ring_order in (numberings if isinstance(numberings, list) else [numberings]):
            subs = _detect_ring_substituents(mol, ring_order, ring_atom_set, chain_set)
            if not subs:
                return base_name
        return base_name

    sub_groups = best_sub_groups

    # Build prefix string for ring substituents
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locs = sorted(sub_groups[name])
        count = len(locs)
        loc_str = ','.join(str(l) for l in locs)
        if count == 1:
            prefix_parts.append(f'{loc_str}-{name}')
        else:
            mult = SIMPLE_MULTIPLIERS.get(count, str(count))
            prefix_parts.append(f'{loc_str}-{mult}{name}')

    ring_prefix = '-'.join(prefix_parts)
    # Wrap in parentheses: (3,4-dihydroxyphenyl)
    return f'({ring_prefix}{base_name})'


def _number_ring_from_attachment(mol, ring_atoms, attachment_idx):
    """Number ring atoms starting from attachment point.

    Returns list of both CW and CCW numberings for locant comparison.
    """
    ring_set = set(ring_atoms)

    # Get the two neighbors of attachment in the ring
    attachment_atom = mol.GetAtomWithIdx(attachment_idx)
    ring_nbrs = [n.GetIdx() for n in attachment_atom.GetNeighbors()
                 if n.GetIdx() in ring_set]

    if len(ring_nbrs) < 2:
        return None

    candidates = []
    for start_nbr in ring_nbrs:
        numbering = {1: attachment_idx}
        visited = {attachment_idx}
        current = start_nbr
        pos = 2

        while pos <= len(ring_atoms):
            numbering[pos] = current
            visited.add(current)
            pos += 1
            atom = mol.GetAtomWithIdx(current)
            next_atom = None
            for nbr in atom.GetNeighbors():
                ni = nbr.GetIdx()
                if ni in ring_set and ni not in visited:
                    next_atom = ni
                    break
            if next_atom is None:
                break
            current = next_atom

        if len(numbering) == len(ring_atoms):
            candidates.append(numbering)

    return candidates if candidates else None


def _detect_ring_substituents(mol, ring_order, ring_atom_set, chain_set):
    """Detect substituents on ring atoms and return {prefix_name: [locants]}."""
    _HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    _ALKOXY = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy'}

    sub_groups = defaultdict(list)
    for ring_pos, atom_idx in ring_order.items():
        if ring_pos == 1:
            continue  # Skip attachment point
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_atom_set or ni in chain_set:
                continue
            sym = nbr.GetSymbol()
            if sym in _HALOGEN_PREFIX:
                sub_groups[_HALOGEN_PREFIX[sym]].append(ring_pos)
            elif sym == 'O':
                h_count = nbr.GetTotalNumHs()
                o_nbrs = [n for n in nbr.GetNeighbors() if n.GetIdx() != atom_idx]
                if h_count == 1 and len(o_nbrs) == 0:
                    sub_groups['hydroxy'].append(ring_pos)
                elif h_count == 0 and len(o_nbrs) == 1 and o_nbrs[0].GetSymbol() == 'C':
                    c_start = o_nbrs[0].GetIdx()
                    c_start_atom = mol.GetAtomWithIdx(c_start)

                    # Aryloxy: O -> aromatic C -> "phenoxy"
                    if c_start_atom.GetIsAromatic():
                        sub_groups['phenoxy'].append(ring_pos)
                    else:
                        # Check for benzyloxy: O -> CH2 -> aromatic
                        benz_nbrs = [n for n in c_start_atom.GetNeighbors()
                                     if n.GetIdx() != ni and n.GetIsAromatic()]
                        non_h_non_arom = [n for n in c_start_atom.GetNeighbors()
                                          if n.GetIdx() != ni
                                          and not n.GetIsAromatic()
                                          and n.GetSymbol() != 'H']
                        if benz_nbrs and not non_h_non_arom and c_start_atom.GetTotalNumHs() >= 1:
                            sub_groups['benzyloxy'].append(ring_pos)
                        else:
                            # Original pure alkyl path
                            c_count = _count_pure_alkyl(mol, c_start, ring_atom_set | {ni})
                            if c_count and c_count in _ALKOXY:
                                sub_groups[_ALKOXY[c_count]].append(ring_pos)
                            elif c_count:
                                from ..data.chain_names import get_alkyl_name as _gal
                                try:
                                    sub_groups[f'{_gal(c_count)}oxy'].append(ring_pos)
                                except (ValueError, KeyError):
                                    pass
            elif sym == 'N':
                h_count = nbr.GetTotalNumHs()
                n_nbrs = [n for n in nbr.GetNeighbors() if n.GetIdx() != atom_idx]
                if h_count == 2 and len(n_nbrs) == 0:
                    sub_groups['amino'].append(ring_pos)
            elif sym == 'C':
                bond = mol.GetBondBetweenAtoms(atom_idx, ni)
                if bond and bond.GetBondTypeAsDouble() == 1.0:
                    c_count = _count_pure_alkyl(mol, ni, ring_atom_set)
                    if c_count:
                        from ..data.chain_names import get_alkyl_name as _gal
                        try:
                            sub_groups[_gal(c_count)].append(ring_pos)
                        except (ValueError, KeyError):
                            pass

    return dict(sub_groups) if sub_groups else None


def _count_pure_alkyl(mol, start_idx, excluded):
    """Count carbons in a pure alkyl chain from start_idx."""
    visited = {start_idx}
    queue = [start_idx]
    carbon_count = 0

    while queue:
        idx = queue.pop(0)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None  # Not pure alkyl
        carbon_count += 1
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni not in visited and ni not in excluded:
                visited.add(ni)
                queue.append(ni)

    return carbon_count if carbon_count > 0 else None


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

    # Collect atoms that belong to principal group (amide/amine N-substituents
    # are handled by specialized naming functions, not here)
    pg_atom_set = set()
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            pg_atom_set.update(match)

    for position, sub_list in features.substituents.items():
        # position is already a 1-indexed locant (from get_substituents)
        for sub_atoms in sub_list:
            # Skip substituents that are ring atoms (handled separately)
            if ring_atoms_to_skip and set(sub_atoms) & ring_atoms_to_skip:
                continue

            # Skip substituents containing the principal group nitrogen
            # (amide N-substituents are handled by _assemble_amide_name,
            #  amine N-substituents by _assemble_amine_name)
            if features.principal_group in (
                'primary_amide', 'secondary_amide', 'tertiary_amide',
                'secondary_amine', 'tertiary_amine',
            ):
                if set(sub_atoms) & pg_atom_set:
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
                    continue

                # Check for acylamino pattern: -NH-C(=O)-R
                acylamino_name = _check_for_acylamino(mol, sub_atoms, features.principal_chain)
                if acylamino_name:
                    substituent_groups[acylamino_name].append(position)
                    continue

                # Check for acyloxy pattern: -O-C(=O)-R
                acyloxy_name = _check_for_acyloxy(mol, sub_atoms, features.principal_chain)
                if acyloxy_name:
                    substituent_groups[acyloxy_name].append(position)
                    continue

                # General fallback for other heteroatom substituents
                hetero_name = _name_heteroatom_substituent(mol, sub_atoms, features.principal_chain)
                if hetero_name:
                    substituent_groups[hetero_name].append(position)
                    continue

                # Truly unnameable substituent - skip
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
            # Use empty locants so _assemble_fragments won't re-add "1-"
            emit_locants = ()
        else:
            formatted = format_substituent_prefix(name, sorted_locants, count)
            emit_locants = tuple(sorted_locants)

        prefixes.append(NameFragment(
            text=formatted,
            locants=emit_locants,
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

    # Check if the ether fragment is aromatic (phenoxy, naphthyloxy, benzyloxy)
    alkyl_atom = mol.GetAtomWithIdx(alkyl_start)

    # Case A: O -> aromatic C in 6-membered all-carbon ring -> "phenoxy"
    if alkyl_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if alkyl_start in ring and len(ring) == 6:
                if all(mol.GetAtomWithIdx(r).GetIsAromatic()
                       and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                       for r in ring):
                    return "phenoxy"
        # Fallback for other aromatic ethers (e.g., naphthyloxy)
        return "phenoxy"

    # Case B: O -> CH2 -> aromatic ring -> "benzyloxy"
    if (not alkyl_atom.GetIsAromatic()
            and alkyl_atom.GetSymbol() == 'C'
            and alkyl_atom.GetTotalNumHs() >= 1):
        arom_nbrs = [n for n in alkyl_atom.GetNeighbors()
                     if n.GetIdx() != oxygen_idx and n.GetIsAromatic()]
        non_h_non_arom = [n for n in alkyl_atom.GetNeighbors()
                          if n.GetIdx() != oxygen_idx
                          and not n.GetIsAromatic()
                          and n.GetSymbol() != 'H']
        if arom_nbrs and not non_h_non_arom:
            return "benzyloxy"

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
        return get_chain_prefix(carbon_count) + "yloxy"
    return None


def _check_for_acylamino(mol, sub_atoms: List[int], principal_chain: List[int]) -> Optional[str]:
    """
    Check if a substituent is an acylamino group: -NH-C(=O)-R or -N(R)-C(=O)-R.

    Pattern: nitrogen bonded to chain, also bonded to a carbonyl carbon C(=O),
    which in turn is bonded to an alkyl chain R.

    Returns name like "(hexacosenoylamino)" for -NH-C(=O)-C25H51.
    Enclosing parens included for IUPAC compound substituent formatting.
    OPSIN requires: 2-(pentanoylamino)pentanedioic acid.
    """
    chain_set = set(principal_chain)
    sub_set = set(sub_atoms)

    # Find nitrogen atoms in the substituent
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'N':
            continue

        # Check if N is bonded to a chain atom
        bonded_to_chain = any(
            nbr.GetIdx() in chain_set for nbr in atom.GetNeighbors()
        )
        if not bonded_to_chain:
            continue

        # Look for carbonyl carbon bonded to this N (C(=O) in sub)
        carbonyl_c = None
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in sub_set or nbr.GetSymbol() != 'C':
                continue
            # Check if this C has a double-bonded O
            for nbr2 in nbr.GetNeighbors():
                if nbr2.GetSymbol() == 'O' and nbr2.GetIdx() in sub_set:
                    bond = mol.GetBondBetweenAtoms(nbr_idx, nbr2.GetIdx())
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        carbonyl_c = nbr_idx
                        break
            if carbonyl_c is not None:
                break

        if carbonyl_c is None:
            # No carbonyl - could be a simple alkylamino
            # Check for ring in substituent first
            ring_info = mol.GetRingInfo()
            sub_has_ring = any(
                ring_info.NumAtomRings(a) > 0 for a in sub_atoms
                if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
            )
            if sub_has_ring:
                # Check for phenyl
                for ring in ring_info.AtomRings():
                    if all(r in sub_set for r in ring) and len(ring) == 6:
                        all_arom = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                        all_c = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                        if all_arom and all_c:
                            return "(phenylamino)"
                continue  # Skip non-phenyl ring substituents

            # Count carbons attached to N via C-C bonds only
            n_alkyl_carbons = 0
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in chain_set:
                    continue
                if nbr.GetSymbol() == 'C':
                    n_alkyl_carbons += _count_carbon_chain(
                        mol, nbr.GetIdx(), chain_set | {idx}
                    )
            if n_alkyl_carbons > 0:
                try:
                    alkyl = get_alkyl_name(n_alkyl_carbons)
                    return f"({alkyl}amino)"
                except (ValueError, KeyError):
                    pass
            continue

        # Found carbonyl: count carbons in the acyl chain (including carbonyl C)
        carbonyl_o = None
        for nbr in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
            if nbr.GetSymbol() == 'O' and nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(carbonyl_c, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() == 2.0:
                    carbonyl_o = nbr.GetIdx()
                    break

        # Count carbons from carbonyl C through C-C bonds only
        # (don't traverse through N to reach other peptide fragments)
        exclude = chain_set | {idx}  # exclude chain and the N
        if carbonyl_o is not None:
            exclude.add(carbonyl_o)  # exclude the C=O oxygen
        acyl_carbons = _count_carbon_chain(mol, carbonyl_c, exclude)
        if acyl_carbons == 0:
            acyl_carbons = 1  # at minimum the carbonyl C

        # Build acylamino name: (prefixanoylamino) with enclosing parens
        # OPSIN requires: 2-(pentanoylamino)pentanedioic acid
        try:
            acyl_prefix = get_chain_prefix(acyl_carbons)
            return f"({acyl_prefix}anoylamino)"
        except (ValueError, KeyError):
            pass

    return None


def _check_for_acyloxy(mol, sub_atoms: List[int], principal_chain: List[int]) -> Optional[str]:
    """
    Check if a substituent is an acyloxy group: -O-C(=O)-R.

    Pattern: oxygen bonded to chain, also bonded to a carbonyl carbon C(=O),
    which is bonded to an alkyl chain R.

    Returns name like "(ethanoyloxy)" for -O-C(=O)-CH3.
    Enclosing parens included for IUPAC compound substituent formatting.
    OPSIN requires: 2-(acetyloxy)benzoic acid.
    """
    chain_set = set(principal_chain)
    sub_set = set(sub_atoms)

    # Find oxygen atom bonded to chain
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'O' or atom.GetDegree() != 2:
            continue

        bonded_to_chain = any(
            nbr.GetIdx() in chain_set for nbr in atom.GetNeighbors()
        )
        if not bonded_to_chain:
            continue

        # Look for carbonyl carbon bonded to this O
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in sub_set or nbr.GetSymbol() != 'C':
                continue
            # Check if this C has another double-bonded O
            has_carbonyl = False
            carbonyl_o = None
            for nbr2 in nbr.GetNeighbors():
                if nbr2.GetIdx() == idx:
                    continue
                if nbr2.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(nbr_idx, nbr2.GetIdx())
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        has_carbonyl = True
                        carbonyl_o = nbr2.GetIdx()
                        break

            if has_carbonyl:
                # Count carbons in acyl chain via C-C bonds only
                exclude = chain_set | {idx}
                if carbonyl_o is not None:
                    exclude.add(carbonyl_o)
                acyl_carbons = _count_carbon_chain(mol, nbr_idx, exclude)
                if acyl_carbons == 0:
                    acyl_carbons = 1

                # Build acyloxy name: (prefixanoyloxy) with enclosing parens
                # OPSIN requires: 2-(acetyloxy)benzoic acid
                try:
                    acyl_prefix = get_chain_prefix(acyl_carbons)
                    return f"({acyl_prefix}anoyloxy)"
                except (ValueError, KeyError):
                    pass

    return None


def _name_heteroatom_substituent(mol, sub_atoms: List[int], principal_chain: List[int]) -> Optional[str]:
    """
    Fallback naming for heteroatom-containing substituents that aren't
    alkoxy, acylamino, or acyloxy.

    Handles: simple amino alkyl chains, hydroxyalkyl, etc.
    """
    chain_set = set(principal_chain)
    sub_set = set(sub_atoms)

    # Identify the atom bonded to the chain (the attachment point)
    attach_atom = None
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in chain_set:
                attach_atom = idx
                break
        if attach_atom is not None:
            break

    if attach_atom is None:
        return None

    atom = mol.GetAtomWithIdx(attach_atom)
    symbol = atom.GetSymbol()

    # If attachment is through N (amino substituent)
    if symbol == 'N':
        # Check if any atoms reachable from N (excluding chain) are in a ring
        ring_info = mol.GetRingInfo()
        sub_has_ring = any(
            ring_info.NumAtomRings(idx) > 0
            for idx in sub_atoms
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
        )
        if sub_has_ring:
            # Substituent has a ring - check for phenyl/benzene
            for ring in ring_info.AtomRings():
                if all(r in sub_set for r in ring) and len(ring) == 6:
                    all_arom = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                    all_c = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                    if all_arom and all_c:
                        return "(phenylamino)"
            # Non-phenyl ring: skip (complex, would need recursive naming)
            return None

        # Count carbons reachable from N via C-C bonds only
        carbon_count = 0
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in chain_set:
                continue
            if nbr.GetSymbol() == 'C':
                carbon_count += _count_carbon_chain(mol, nbr.GetIdx(), chain_set | {attach_atom})

        if carbon_count == 0:
            return "amino"
        try:
            alkyl = get_alkyl_name(carbon_count)
            return f"({alkyl}amino)"
        except (ValueError, KeyError):
            return "amino"

    # If attachment is through O but not alkoxy or acyloxy
    # (could be plain hydroxyl on carbon substituent)
    if symbol == 'O':
        return None  # handled by FG prefixes

    # If attachment is through C with heteroatoms deeper in the chain
    if symbol == 'C':
        # Check if substituent contains a ring - skip complex ring naming
        ring_info = mol.GetRingInfo()
        sub_has_ring = any(
            ring_info.NumAtomRings(idx) > 0 for idx in sub_atoms
        )
        if sub_has_ring:
            return None  # Complex ring substituent - needs specialized naming

        # Count carbons via C-C bonds only (don't traverse through heteroatoms)
        total_carbons = _count_carbon_chain(mol, attach_atom, chain_set)
        if total_carbons > 0:
            # Check what heteroatoms are present
            heteroatoms = set()
            for i in sub_atoms:
                sym = mol.GetAtomWithIdx(i).GetSymbol()
                if sym not in ('C', 'H'):
                    heteroatoms.add(sym)

            # For C-chain with OH: name as hydroxyalkyl
            if heteroatoms == {'O'}:
                # Check if the O is -OH (not C=O or ether)
                for i in sub_atoms:
                    a = mol.GetAtomWithIdx(i)
                    if a.GetSymbol() == 'O' and a.GetDegree() == 1:
                        try:
                            alkyl = get_alkyl_name(total_carbons)
                            return f"(hydroxy{alkyl})"
                        except (ValueError, KeyError):
                            pass

            # For C-chain with NH2: name as (aminoalkyl)
            # e.g., -CH2CH2NH2 -> (2-aminoethyl)
            if 'N' in heteroatoms:
                # Check for terminal primary amine (-NH2) on the chain
                for i in sub_atoms:
                    a = mol.GetAtomWithIdx(i)
                    if (a.GetSymbol() == 'N' and a.GetDegree() == 1
                            and a.GetTotalNumHs() == 2):
                        try:
                            alkyl = get_alkyl_name(total_carbons)
                            # Locant for amino on sub-chain: N is at terminal
                            # position = total_carbons (farthest from attachment)
                            return f"({total_carbons}-amino{alkyl})"
                        except (ValueError, KeyError):
                            pass

            # For simple case: just name as alkyl (ignoring heteroatoms)
            # This is imperfect but better than dropping entirely
            try:
                return get_alkyl_name(total_carbons)
            except (ValueError, KeyError):
                pass

    return None


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in an alkyl group via BFS (traverses all atom types)."""
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


def _count_carbon_chain(mol, start_idx: int, exclude: set) -> int:
    """Count carbons reachable only through C-C bonds (no heteroatom traversal).

    Unlike _count_alkyl_carbons, this stops at heteroatoms. Used for naming
    acyl chains where we don't want to cross amide/ester bonds.
    """
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
        if atom.GetSymbol() != 'C':
            continue  # Don't count or traverse through non-carbon atoms

        count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                # Only traverse to other carbon atoms
                if neighbor.GetSymbol() == 'C':
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

    # Count ALL ring substituents (alkyl + FG), not just the ones we named
    # This prevents monosubstituted=True when there's 1 alkyl + 1 halogen
    total_all_substituents = sum(
        len(sub_list) for sub_list in ring_substituents.values()
    )
    is_monosubstituted = total_all_substituents == 1

    # Build prefix fragments
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)

        if is_monosubstituted:
            # Monosubstituted: omit locant (it's always 1)
            # Just the substituent name: "methyl" not "1-methyl"
            formatted = name
            # Use empty locants so _assemble_fragments won't re-add "1-"
            emit_locants = ()
        else:
            # Polysubstituted: include locants
            formatted = format_substituent_prefix(name, sorted(locants), count)
            emit_locants = tuple(sorted(locants))

        prefixes.append(NameFragment(
            text=formatted,
            locants=emit_locants,
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


def _estimate_parent_size_from_name(parent_name: str) -> int:
    """Estimate the number of atoms in the parent from its name.

    Used by collision detection to determine if a locant exceeds the parent
    structure capacity.  Returns a conservative estimate; unknown parents
    default to 100 (effectively disabling capacity validation).

    Args:
        parent_name: The parent fragment text (e.g., 'cyclohex', 'benz', 'prop').

    Returns:
        Estimated atom count in the parent structure.
    """
    lower = parent_name.lower()

    # Common ring systems with fixed sizes
    _RING_SIZES = {
        'benzene': 6, 'phenyl': 6, 'benz': 6,
        'cycloprop': 3, 'cyclobut': 4, 'cyclopent': 5,
        'cyclohex': 6, 'cyclohept': 7, 'cycloocta': 8,
        'cyclonon': 9, 'cyclodec': 10,
        'naphthal': 10, 'naphthyl': 10,
        'indol': 9, 'indene': 9,
        'quinol': 10, 'isoquinol': 10,
        'pyrid': 6, 'pyrimid': 6, 'pyrazin': 6,
        'pyrrol': 5, 'furan': 5, 'thiophen': 5,
        'imidazol': 5, 'pyrazol': 5, 'oxazol': 5,
        'thiazol': 5, 'triazin': 6,
    }
    for key, size in _RING_SIZES.items():
        if key in lower:
            return size

    # Try chain prefix matching using FIRST_20 from chain_names
    from ..data.chain_names import FIRST_20
    # Check longest prefixes first to avoid partial matches (e.g., "eth" in "meth")
    for length in sorted(FIRST_20.keys(), reverse=True):
        prefix = FIRST_20[length]
        if lower.startswith(prefix) or lower == prefix:
            return length

    # Safe fallback: return large value to disable capacity validation
    # for unknown parent structures.
    import logging
    logging.getLogger(__name__).debug(
        "Unknown parent '%s' -- skipping locant capacity validation (fallback=100)",
        parent_name,
    )
    return 100


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

    # ----------------------------------------------------------------
    # Detect suffix-prefix locant collisions on ring systems.
    # A collision occurs when a suffix locant (e.g., ketone at position 3)
    # and a prefix locant (e.g., methyl at position 3) share the same
    # numeric value.  OPSIN interprets this as both groups on the same
    # carbon, producing an unphysical valency.
    # Resolution: remove the colliding prefix locant (suffix has priority
    # per IUPAC P-14.7).  Only applies to ring parents.
    # ----------------------------------------------------------------
    if suffix_frag and suffix_frag.locants and prefixes:
        # Determine if parent is a ring (collision only matters for rings)
        parent_text = parent_frag.text if parent_frag else ""
        is_ring_parent = any(
            kw in parent_text.lower()
            for kw in ('cyclo', 'benz', 'pyrid', 'pyrrol', 'furan',
                       'thiophen', 'imidazol', 'naphthal', 'indol',
                       'quinol', 'pyrimid', 'pyrazin', 'oxazol',
                       'thiazol', 'triazol', 'morpholin', 'piperidin',
                       'pyrrolidin', 'aziridin', 'oxiran', 'thiiran',
                       'oxetan', 'azetidin', 'thietan')
        )
        if is_ring_parent:
            from ..rules.locant_validation import detect_locant_collisions

            suffix_locants_list = list(suffix_frag.locants)
            prefix_locant_groups = [
                list(p.locants) for p in prefixes if p.locants
            ]
            parent_size = _estimate_parent_size_from_name(parent_text)

            if suffix_locants_list and prefix_locant_groups:
                collisions = detect_locant_collisions(
                    suffix_locants_list,
                    prefix_locant_groups,
                    parent_type="ring",
                    parent_size=parent_size,
                )
                if collisions:
                    import logging
                    _log = logging.getLogger(__name__)
                    collision_set = set(loc for _, loc in collisions)
                    adjusted = []
                    for pf in prefixes:
                        if pf.locants:
                            new_locants = tuple(
                                l for l in pf.locants if l not in collision_set
                            )
                            if new_locants != pf.locants:
                                _log.debug(
                                    "Collision resolved: removed prefix locant(s) %s "
                                    "for '%s' (suffix has priority per IUPAC P-14.7)",
                                    set(pf.locants) - set(new_locants),
                                    pf.text,
                                )
                                pf = NameFragment(
                                    text=pf.text,
                                    locants=new_locants,
                                    fragment_type=pf.fragment_type,
                                    count=len(new_locants) if new_locants else pf.count,
                                )
                        adjusted.append(pf)
                    prefixes = adjusted

    # Alkyl prefixes are already sorted by _generate_alkyl_prefixes.
    # For non-alkyl prefixes added later, sort all together.
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    # Build prefix strings with locants
    # IUPAC rule: hyphens separate locants from names, and are needed
    # between prefixes when one ends with a letter and the next starts with a digit
    # NOTE: Some prefixes already have locants baked in (ring substituent prefixes
    # like "4-phenyl"). Only add locants to those that don't already have them.
    import re
    prefix_texts = []
    for f in prefixes:
        text = f.text
        already_has_locant = bool(re.match(r'^\d', text))
        if f.locants and not already_has_locant:
            loc_str = ",".join(str(l) for l in f.locants)
            prefix_texts.append(f"{loc_str}-{text}")
        else:
            prefix_texts.append(text)
    prefix_str = _join_prefixes(prefix_texts)

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
        # Use suffix_frag.count (set by _generate_suffix) which includes terminal
        # groups like diacids where locants are omitted but multiplier is needed
        count = max(len(suffix_locants), getattr(suffix_frag, 'count', 1))
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

    # Add prefixes with proper hyphenation at boundary
    if prefix_str:
        name = _join_prefix_to_name(prefix_str, name)

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


def _join_prefix_to_name(prefix_str: str, name: str) -> str:
    """
    Join a prefix string to a parent/suffix name with proper IUPAC hyphenation.

    Ensures a hyphen is inserted when:
    - The prefix ends with a letter and the name starts with a digit

    This prevents broken names like 'pentabutyl1,4,7,10,13-pentaaza'
    by inserting a hyphen: 'pentabutyl-1,4,7,10,13-pentaaza'.

    Args:
        prefix_str: The assembled prefix string (e.g., '3-ethyl-4-methyl')
        name: The parent+suffix name (e.g., 'propan-1-ol')

    Returns:
        Properly hyphenated combined name
    """
    if not prefix_str or not name:
        return prefix_str + name

    if prefix_str[-1].isalpha() and name[0].isdigit():
        return f"{prefix_str}-{name}"

    return f"{prefix_str}{name}"


def _build_long_chain_prefix(length: int) -> str:
    """
    Build prefix for long chains.

    Delegates to centralized chain_names module.
    Kept for backward compatibility.
    """
    return get_chain_prefix(length)


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
        IUPAC name for the ion/radical, or empty string on failure

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

    try:
        if species_type == 'salt':
            return name_salt(mol, style)
        elif species_type == 'radical':
            return name_radical(mol, style)
        elif species_type == 'zwitterion':
            result = name_zwitterion(mol, style)
            # Guard: never return the literal 'zwitterion'
            if result and result != 'zwitterion':
                return result
            return ''
        elif species_type == 'ion':
            # Single ion - determine if cation or anion
            ion_sites = getattr(features, 'ion_sites', {})
            if ion_sites.get('cations') and not ion_sites.get('anions'):
                return name_cation(mol, style)
            elif ion_sites.get('anions') and not ion_sites.get('cations'):
                return name_anion(mol, style)
    except RecursionError:
        # Safety net: if recursion still occurs, return empty string
        return ''

    # Fallback for unrecognized species
    return ''


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
