"""
Resolver functions for parent structure classification and suffix resolution.

These are standalone functions that classify a molecule's parent type and
resolve its suffix information, without generating the final name string.
They serve as the foundation for aspect-based composition in the composer.

Functions:
    resolve_parent: Classify the parent structure type and extract metadata
    resolve_suffix: Determine suffix text, locants, and multiplier
    apply_ion_suffix_modification: Transform suffix for ionic species

Design principles:
    - Resolvers are CLASSIFIERS, not name generators
    - parent_label is a tag for handler selection, NOT the final IUPAC name
    - No circular imports: function-level imports for ring detection
    - Compatible with existing MolecularFeatures dataclass
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..rules.locants import get_functional_group_locants
from ..rules.seniority import get_suffix
from .naming_utils import TERMINAL_FG_TYPES, should_omit_locant_one

# Polycyclic aromatic names that have fixed IUPAC numbering
_NAMED_PAH_SYSTEMS = {
    "naphthalene", "anthracene", "phenanthrene", "pyrene",
    "fluorene", "acenaphthylene", "fluoranthene", "chrysene",
    "triphenylene", "coronene", "perylene", "azulene",
    "acenaphthene",
}


@dataclass
class ParentInfo:
    """Describes the resolved parent structure for naming.

    NOTE: This is a classification result, not a name generator.
    parent_label is a classification tag (e.g., "cyclohexane", "indole"),
    NOT the final generated name string. It identifies the parent type
    for downstream handlers that will produce the actual name.

    Attributes:
        parent_label: Classification tag (e.g., "cyclohexane", "indole", "propane").
                      Used to select the naming handler. NOT the final IUPAC name.
        parent_type:  One of "chain", "ring", "benzene", "fused_heterocycle",
                      "polycyclic", "complex_ring", "polycyclic_aromatic".
        atom_count:   Number of atoms in parent (chain length or ring size).
        atom_to_locant: Atom index -> IUPAC locant mapping.
        is_named_ring: True for benzene, naphthalene, indole (fixed numbering).
    """
    parent_label: str
    parent_type: str
    atom_count: int
    atom_to_locant: Dict[int, Any] = field(default_factory=dict)
    is_named_ring: bool = False


@dataclass
class SuffixInfo:
    """Describes the resolved suffix for the principal functional group.

    Attributes:
        text:        Suffix text (e.g., "ol", "one", "oic acid", "").
        locants:     Locant positions (e.g., [1,3] for diol, [2] for ketone).
        count:       Number of suffix instances (e.g., 2 for diol).
        is_terminal: True if suffix group is always terminal (e.g., -oic acid, -al).
    """
    text: str = ""
    locants: List[int] = field(default_factory=list)
    count: int = 0
    is_terminal: bool = False


def resolve_parent(features: Any, mol: Any) -> ParentInfo:
    """Classify the parent structure and extract naming metadata.

    Inspects MolecularFeatures to determine the parent type. This is a
    CLASSIFICATION function -- it does NOT generate name strings. The
    parent_label is a tag used to select the appropriate naming handler.

    Priority order (matches assemble_name routing in composer.py):
        1. polycyclic_name set -> polycyclic_aromatic
        2. is_cyclic and complex ring system -> complex_ring
        3. is_benzene -> benzene
        4. ring_type startswith "heterocyclic" -> ring (heterocyclic_aromatic/heterocyclic_saturated)
        5. ring_type in (cycloalkane, cycloalkene, aromatic) -> ring
        6. principal_chain -> chain
        7. Fallback -> chain (methane)

    Args:
        features: MolecularFeatures object with perceived/classified data.
        mol: RDKit Mol object.

    Returns:
        ParentInfo with classification tag, type, atom count, and locant map.
    """
    # --- 1. Polycyclic aromatic (naphthalene, anthracene, etc.) ---
    polycyclic_name = getattr(features, 'polycyclic_name', None)
    if polycyclic_name:
        atom_count = _get_polycyclic_atom_count(mol)
        atom_to_locant = _extract_atom_to_locant(features)
        return ParentInfo(
            parent_label=polycyclic_name,
            parent_type="polycyclic_aromatic",
            atom_count=atom_count,
            atom_to_locant=atom_to_locant,
            is_named_ring=True,
        )

    # --- 2. Complex ring system (bicyclo, spiro, fused, polycyclic-bridged) ---
    is_cyclic = getattr(features, 'is_cyclic', False)
    chain_is_parent = getattr(features, 'chain_is_parent', False)
    if is_cyclic and not chain_is_parent and _check_complex_ring_system(mol):
        atom_count = _get_complex_ring_atom_count(mol)
        atom_to_locant = _extract_atom_to_locant(features)

        # Determine if this is a named fused heterocycle
        is_named = _is_named_fused_heterocycle(mol)
        label = _get_complex_ring_label(mol, features, is_named)
        parent_type = "fused_heterocycle" if is_named else "complex_ring"

        return ParentInfo(
            parent_label=label,
            parent_type=parent_type,
            atom_count=atom_count,
            atom_to_locant=atom_to_locant,
            is_named_ring=is_named,
        )

    # --- 3. Benzene ---
    is_benzene = getattr(features, 'is_benzene', False)
    if is_benzene:
        atom_to_locant = _extract_atom_to_locant(features)
        return ParentInfo(
            parent_label="benzene",
            parent_type="benzene",
            atom_count=6,
            atom_to_locant=atom_to_locant,
            is_named_ring=True,
        )

    # --- 4. Heterocyclic ring ---
    ring_type = getattr(features, 'ring_type', None)
    if ring_type and ring_type.startswith('heterocyclic'):
        atom_count = _get_ring_atom_count(features, mol)
        atom_to_locant = _extract_heterocycle_locant_map(features)
        label = _get_heterocycle_label(features)
        return ParentInfo(
            parent_label=label,
            parent_type="ring",
            atom_count=atom_count,
            atom_to_locant=atom_to_locant,
            is_named_ring=False,  # Simple heterocycles don't have fixed numbering
        )

    # --- 5. Cycloalkane / cycloalkene ring ---
    if is_cyclic and not chain_is_parent and ring_type in ('cycloalkane', 'cycloalkene', 'aromatic'):
        atom_count = _get_ring_atom_count(features, mol)
        atom_to_locant = _extract_ring_locant_map(features)
        # Build a label like "cyclohexane" from ring size + type
        from ..data.chain_names import get_chain_prefix
        try:
            stem = get_chain_prefix(atom_count)
        except (ValueError, KeyError):
            stem = "cyclo"
        label = f"cyclo{stem}"
        return ParentInfo(
            parent_label=label,
            parent_type="ring",
            atom_count=atom_count,
            atom_to_locant=atom_to_locant,
            is_named_ring=False,
        )

    # --- 6. Chain parent ---
    principal_chain = getattr(features, 'principal_chain', None)
    if principal_chain and len(principal_chain) > 0:
        atom_count = len(principal_chain)
        atom_to_locant = getattr(features, 'atom_to_locant', {}) or {}
        from ..data.chain_names import get_chain_prefix
        try:
            label = get_chain_prefix(atom_count)
        except (ValueError, KeyError):
            label = "meth"
        return ParentInfo(
            parent_label=label,
            parent_type="chain",
            atom_count=atom_count,
            atom_to_locant=dict(atom_to_locant),
            is_named_ring=False,
        )

    # --- 7. Fallback: single atom / methane ---
    return ParentInfo(
        parent_label="meth",
        parent_type="chain",
        atom_count=1,
        atom_to_locant={},
        is_named_ring=False,
    )


def resolve_suffix(features: Any, parent_info: ParentInfo) -> SuffixInfo:
    """Determine suffix information for the principal functional group.

    Extracts suffix text, locants, count, and terminal status from the
    features and the resolved parent info. Uses parent_info.atom_count
    for capacity validation (suffix count cannot exceed parent size).

    Args:
        features: MolecularFeatures object.
        parent_info: ParentInfo from resolve_parent().

    Returns:
        SuffixInfo with text, locants, count, and is_terminal flag.
    """
    fg_name = getattr(features, 'principal_group', None)
    if not fg_name:
        return SuffixInfo()

    # Correct is_ring: chain suffix for chain parents, ring suffix for ring parents.
    if getattr(features, 'chain_is_parent', False):
        is_ring = False
    elif getattr(features, 'principal_chain', None):
        is_ring = False
    else:
        is_ring = getattr(features, 'is_cyclic', False)
    suffix_text = get_suffix(fg_name, is_ring=is_ring)

    if not suffix_text:
        return SuffixInfo()

    # Determine count from principal group atoms
    pg_atoms = getattr(features, 'principal_group_atoms', None)
    fg_count = len(pg_atoms) if pg_atoms else 1

    # Capacity validation: count cannot exceed parent atom count
    max_capacity = parent_info.atom_count
    if fg_count > max_capacity:
        fg_count = max_capacity

    # Resolve locants
    locants = []
    principal_chain = getattr(features, 'principal_chain', None)
    atom_to_locant = getattr(features, 'atom_to_locant', None)

    if principal_chain and atom_to_locant and pg_atoms:
        fg_locants = get_functional_group_locants(
            principal_chain,
            pg_atoms,
            atom_to_locant,
            mol=getattr(features, 'mol', None),
        )
        # Deduplicate (overlapping SMARTS can produce duplicates)
        fg_locants = sorted(set(fg_locants))

        # Update count to match unique locants
        if fg_locants:
            fg_count = len(fg_locants)

        # Terminal groups: locant implicitly 1, omit from name
        if not should_omit_locant_one(context="suffix", fg_type=fg_name):
            locants = fg_locants

    is_terminal = fg_name in TERMINAL_FG_TYPES

    return SuffixInfo(
        text=suffix_text,
        locants=locants,
        count=fg_count,
        is_terminal=is_terminal,
    )


# Ion suffix modification lookup tables
_ANION_SUFFIX_MAP = {
    "ol": "olate",
    "oic acid": "oate",
    "amine": "aminide",
    "thiol": "thiolate",
    "sulfonic acid": "sulfonate",
    "sulfinic acid": "sulfinate",
    "phosphonic acid": "phosphonate",
    "phosphinic acid": "phosphinate",
    # IN-01 (code review 2026-06-02): RESERVED / unreachable-by-design. A correct
    # chemical mapping, but O-P phosphate ESTERS are deliberately excluded from
    # the oxoacid-anion routing upstream (classify_anion sends them to 'alkoxide',
    # not 'phosphonate'/'phosphate'), so no neutral name ending in "phosphoric
    # acid" currently reaches _ionize_acid_name. Kept for completeness/future use.
    "phosphoric acid": "phosphate",  # SUB-01/D-02
    # P-72.2.2.2.1.1 (169.6-02): "the 'ic acid' or 'ous acid' ending ... by
    # 'ate' or 'ite', respectively" — the -ous-acid anion takes -ite, parallel
    # to the -ic-acid -> -ate transforms above.
    "ous acid": "ite",
    #: "nitric acid": "nitrate" DEFERRED — Plan-01 reach = 2/7,500; the
    # internal-charge-filter precision change (protecting every nitro) is not
    # justified at that frequency.
    "carboxylic acid": "carboxylate",  # P-72.2.2.2.1.1 (CORRECT — stays)
    # P-72.2.2.2.4 (169.6-02 FIX): amide/carboxamide/carbonitrile anionic
    # centers are named on the corresponding ANIONIC PARENT HYDRIDE form
    # (e.g. CH3-CO-NH(-) -> acetylazanide), NOT a suffix -ate/-ate/-ate.
    # "Suffixes such as 'amidide' and 'carboxamidide' are not recommended."
    # The four non-IUPAC entries amide->amidate, carboxamide->carboxamidate,
    # carbonitrile->carbonitrilate were REMOVED here (audit §4.1); the seam now
    # returns '' for them so route_charged falls through to the parent-hydride
    # path (Plan 03) instead of emitting a wrong -ate name.
}

_CATION_SUFFIX_MAP = {
    "amine": "aminium",  # P-73.1.2.1 WAY1 / Table 7.4 (protonated amine, the PIN)
    # Table 7.4 (P-73.1.2.1 / 169.6-02 ADD): cationic characteristic-group
    # suffixes formed by adding 'ium' to the neutral nitrogen-bearing suffix
    # ("the largest neutral parent possible is used", P-73.1.2).
    "amide": "amidium",
    "carboxamide": "carboxamidium",
    "imide": "imidium",
    "carboximide": "carboximidium",
    "nitrile": "nitrilium",
    "carbonitrile": "carbonitrilium",
    "imine": "iminium",
    # P-73.1.2.1 (169.6-02 FIX): "ol" -> "olium" REMOVED. A protonated alcohol
    # is named on the oxidanium parent cation (e.g. ethylideneoxidanium), NOT a
    # bogus -olium suffix; -ol carries no nitrogen so WAY-2 (substitute a
    # cationic parent hydride) governs, not a suffix swap. Removing it makes the
    # seam return '' for an -ol stem so the cation path does not mis-fire.
}


def apply_ion_suffix_modification(suffix_info: SuffixInfo, features: Any) -> SuffixInfo:
    """Apply ion suffix modification to transform neutral suffix to ionic form.

    For single-component ions (not salts/zwitterions), modifies the suffix:
    - Anion: "-ol" -> "-olate", "-oic acid" -> "-oate", bare parent -> "-ide"
    - Cation: "-amine" -> "-aminium", bare parent -> "-ium"

    Salts, zwitterions, and neutral species are returned unchanged.

    Args:
        suffix_info: SuffixInfo from resolve_suffix().
        features: MolecularFeatures with species_type and total_charge.

    Returns:
        Modified SuffixInfo for ionic species, or unchanged for neutral/salt/zwitterion.
    """
    species_type = getattr(features, 'species_type', 'neutral')

    # Only modify for single-component ions
    if species_type not in ('ion',):
        return suffix_info

    total_charge = getattr(features, 'total_charge', 0)

    if total_charge < 0:
        # Anion modification
        return _apply_anion_modification(suffix_info)
    elif total_charge > 0:
        # Cation modification
        return _apply_cation_modification(suffix_info)

    # Neutral charge but species_type == 'ion' (edge case): return unchanged
    return suffix_info


def _apply_anion_modification(suffix_info: SuffixInfo) -> SuffixInfo:
    """Transform suffix for anionic species."""
    text = suffix_info.text

    if not text:
        # No suffix -> "-ide" (carbanide, etc.)
        return SuffixInfo(
            text="ide",
            locants=list(suffix_info.locants),
            count=suffix_info.count,
            is_terminal=suffix_info.is_terminal,
        )

    # Try direct lookup
    for neutral, anionic in _ANION_SUFFIX_MAP.items():
        if text == neutral:
            return SuffixInfo(
                text=anionic,
                locants=list(suffix_info.locants),
                count=suffix_info.count,
                is_terminal=suffix_info.is_terminal,
            )

    # Fallback: no matching transformation, return unchanged
    return suffix_info


def _apply_cation_modification(suffix_info: SuffixInfo) -> SuffixInfo:
    """Transform suffix for cationic species."""
    text = suffix_info.text

    if not text:
        # No suffix -> "-ium" (e.g., methanium)
        return SuffixInfo(
            text="ium",
            locants=list(suffix_info.locants),
            count=suffix_info.count,
            is_terminal=suffix_info.is_terminal,
        )

    # Try direct lookup
    for neutral, cationic in _CATION_SUFFIX_MAP.items():
        if text == neutral:
            return SuffixInfo(
                text=cationic,
                locants=list(suffix_info.locants),
                count=suffix_info.count,
                is_terminal=suffix_info.is_terminal,
            )

    # IN-02 (code review 2026-06-02): no canonical cation transform for this
    # suffix -> return it UNCHANGED (the safe default). The previous comment
    # claimed an "-ium" was appended ("one" -> "onium"), which the code never
    # did; corrected to describe the actual behavior.
    return suffix_info


# =============================================================================
# Private helper functions
# =============================================================================


def _extract_atom_to_locant(features: Any) -> Dict[int, Any]:
    """Extract the best available atom-to-locant mapping from features."""
    # Priority: heterocycle > oriented ring > chain > empty
    het_map = getattr(features, 'heterocycle_atom_to_locant', None)
    if het_map:
        return dict(het_map)

    chain_map = getattr(features, 'atom_to_locant', None)
    if chain_map:
        return dict(chain_map)

    # Build from oriented_ring if available
    oriented = getattr(features, 'oriented_ring', None)
    if oriented:
        return {atom_idx: i + 1 for i, atom_idx in enumerate(oriented)}

    return {}


def _extract_heterocycle_locant_map(features: Any) -> Dict[int, Any]:
    """Extract locant map for heterocyclic rings."""
    het_map = getattr(features, 'heterocycle_atom_to_locant', None)
    if het_map:
        return dict(het_map)

    oriented = getattr(features, 'oriented_heterocycle', None)
    if oriented:
        return {atom_idx: i + 1 for i, atom_idx in enumerate(oriented)}

    return _extract_ring_locant_map(features)


def _extract_ring_locant_map(features: Any) -> Dict[int, Any]:
    """Extract locant map for simple rings (cycloalkane, cycloalkene)."""
    oriented = getattr(features, 'oriented_ring', None)
    if oriented:
        return {atom_idx: i + 1 for i, atom_idx in enumerate(oriented)}

    chain_map = getattr(features, 'atom_to_locant', None)
    if chain_map:
        return dict(chain_map)

    return {}


def _get_ring_atom_count(features: Any, mol: Any) -> int:
    """Get atom count for a monocyclic ring."""
    oriented = getattr(features, 'oriented_ring', None)
    if oriented:
        return len(oriented)

    oriented_het = getattr(features, 'oriented_heterocycle', None)
    if oriented_het:
        return len(oriented_het)

    principal_ring = getattr(features, 'principal_ring', None)
    if principal_ring:
        return len(principal_ring)

    # Fallback: largest ring from RDKit
    if mol is not None:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        if atom_rings:
            return max(len(r) for r in atom_rings)

    return 0


def _get_polycyclic_atom_count(mol: Any) -> int:
    """Heavy-atom count for a polycyclic aromatic parent (capacity check input).

    IN-03 (code review 2026-06-02): returns the molecule's heavy-atom count. This
    is exact for an unsubstituted PAH (the parent IS the whole molecule) but
    OVER-counts when the PAH carries substituents/side-chains (it counts those
    too). The previously-declared ``polycyclic_name`` parameter was never used and
    has been dropped. Computing the true named-ring-system atom count is a
    follow-on; the over-count only widens the downstream ``max_capacity`` check,
    so it cannot reject a valid suffix.
    """
    if mol is not None:
        return mol.GetNumHeavyAtoms()
    return 0


def _get_complex_ring_atom_count(mol: Any) -> int:
    """Get total ring system atom count for complex rings."""
    if mol is None:
        return 0

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        return 0

    # Union of all ring atoms
    all_ring_atoms = set()
    for ring in atom_rings:
        all_ring_atoms.update(ring)
    return len(all_ring_atoms)


def _check_complex_ring_system(mol: Any) -> bool:
    """Check if molecule has a complex ring system (bicyclo, spiro, fused, etc.).

    Uses function-level imports to avoid circular dependency.
    """
    from ..rules.bicyclo import is_bicyclo_system
    from ..rules.bridged_fused import detect_bridged_fused
    from ..rules.fused_rings import classify_fused_system
    from ..rules.polycyclic import is_polycyclic_system
    from ..rules.spiro import is_spiro_system

    if detect_bridged_fused(mol):
        return True
    if is_bicyclo_system(mol):
        return True
    if is_polycyclic_system(mol):
        return True
    fused_type = classify_fused_system(mol)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return True
    if is_spiro_system(mol):
        return True
    return False


def _is_named_fused_heterocycle(mol: Any) -> bool:
    """Check if molecule matches a named fused heterocycle (indole, quinoline, etc.)."""
    from ..data.fused_heterocycles import match_fused_heterocycle_core
    core_match = match_fused_heterocycle_core(mol)
    return core_match is not None


def _get_complex_ring_label(mol: Any, features: Any, is_fused_het: bool) -> str:
    """Get classification label for complex ring system."""
    if is_fused_het:
        from ..data.fused_heterocycles import match_fused_heterocycle_core
        core_match = match_fused_heterocycle_core(mol)
        if core_match is not None:
            # Returns (core_name, atom_mapping, core_smiles) tuple
            return core_match[0]  # core_name, e.g., "1H-indole"
        return 'fused_heterocycle'

    # Classify the ring system type
    from ..rules.bicyclo import is_bicyclo_system
    from ..rules.bridged_fused import detect_bridged_fused
    from ..rules.polycyclic import is_polycyclic_system
    from ..rules.spiro import is_spiro_system

    if detect_bridged_fused(mol):
        return "bridged-fused"
    if is_bicyclo_system(mol):
        return "bicyclo"
    if is_polycyclic_system(mol):
        return "polycyclic-bridged"
    if is_spiro_system(mol):
        return "spiro"

    return "complex_ring"


def _get_heterocycle_label(features: Any) -> str:
    """Get classification label for a heterocyclic ring."""
    het_info = getattr(features, 'heterocycle_info', None)
    if het_info and isinstance(het_info, dict):
        # Use the IUPAC name if available (e.g., "pyridine", "morpholine")
        name = het_info.get('name') or het_info.get('hw_name')
        if name:
            return name

    return "heterocycle"
