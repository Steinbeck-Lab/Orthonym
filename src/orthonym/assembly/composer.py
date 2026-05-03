"""
Name assembly - combining fragments into complete IUPAC names.

Assembly order:
1. Stereodescriptors (R/S, E/Z) at start in parentheses
2. Locanted prefixes (substituents, alphabetized)
3. Parent name (with unsaturation modifiers)
4. Locanted suffix (principal group)
"""

import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, namedtuple

logger = logging.getLogger(__name__)

# Structured result from complex ring sub-handlers.
# Returned by _assemble_complex_ring_name() to provide ring atom and
# numbering information for the universal substituent pipeline (Phase 119).
ComplexRingResult = namedtuple(
    'ComplexRingResult',
    ['name', 'ring_atoms', 'atom_to_locant', 'substituents_included']
)


@dataclass
class HandlerResult:
    """Observational coverage metric for naming handlers (Phase 139 ARCH-06).

    Tracks what fraction of the molecule's heavy atoms are accounted for
    in the generated name. Low coverage indicates potential substituent drops.
    This is OBSERVATIONAL ONLY -- does not affect naming behavior.
    """
    name: str
    handler_id: str
    parent_atoms: Set[int] = field(default_factory=set)
    accounted_atoms: Set[int] = field(default_factory=set)
    total_heavy_atoms: int = 0

    @property
    def coverage(self) -> float:
        """Return fraction of heavy atoms accounted for (0.0 to 1.0)."""
        return len(self.accounted_atoms) / self.total_heavy_atoms if self.total_heavy_atoms > 0 else 0.0


from ..rules.seniority import get_suffix, get_prefix
from ..rules.locants import get_functional_group_locants, get_bond_locants
from .naming_utils import (
    format_suffix_with_locants,
    get_multiplier_prefix,
    format_substituent_prefix,
    alpha_sort_key,
    get_alkyl_name,
    is_complex_substituent,
    should_omit_locant_one,
    _wrap_n_substituent,
    SIMPLE_MULTIPLIERS,
    COMPLEX_MULTIPLIERS,
    TERMINAL_FG_TYPES,
    BRANCH_HANDLED_FGS,
)
from ..data.chain_names import get_chain_prefix
from .substituent_naming import name_substituent_fragment, _is_linear_alkyl
from .substituent_enumerator import (
    extract_ring_substituents,
    classify_and_name_fragment,
    collect_substituent_atom_set,
)
from .coverage_scoring import (
    compute_confidence,
    select_best_candidate,
    store_confidence,
    clear_confidence,
    log_confidence,
    CandidateName,
)
from .candidate_pool import get_current_pool, clear_pool, push_pool, pop_pool

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

# _coverage_gate_threshold() removed in Phase 81: replaced by
# graduated confidence scoring in coverage_scoring.py


# ============================================================================
# Universal Prefix Integration Helper (Phase 86)
# ============================================================================


def _integrate_universal_prefixes(
    mol,
    parent_atoms,
    parent_type="auto",
    oriented_ring=None,
    principal_chain=None,
    atom_to_locant=None,
    exclude_atoms=None,
):
    """Discover and format all substituents on a parent structure.

    Uses the universal pipeline (Phase 84-85) to enumerate ALL
    non-parent atoms and name them as IUPAC prefixes. Any handler
    can call this to get a correctly formatted, alphabetically sorted
    prefix string ready to prepend to its core name.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.
        parent_type: ``"ring"``, ``"chain"``, or ``"auto"`` (auto-detects).
        oriented_ring: Ring atom indices in IUPAC order (for ring parents).
        principal_chain: Chain atom indices in order (for chain parents).
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.
        exclude_atoms: Atoms already accounted for (e.g., carbonyl O of
            lactone, halogen of acid halide) -- added to parent set so
            they are not discovered as substituents.

    Returns:
        Prefix string (e.g., ``"3-methyl-"`` or ``"2-chloro-3-methyl-"``)
        ready to prepend to the handler's core name. Returns empty string
        ``""`` if no substituents are found.

    References:
        IUPAC 2013 P-31.1 (detachable prefixes)
        Phase 84: ``discover_substituents()``
        Phase 85: ``name_substituent()``
    """
    from .substituent_enumerator import discover_substituents, name_substituent

    parent_set = set(parent_atoms)

    # Merge exclude_atoms into the effective parent set so the discovery
    # engine treats them as "accounted for" (not substituents).
    effective_parent = set(parent_set)
    if exclude_atoms:
        effective_parent |= set(exclude_atoms)

    try:
        subs = discover_substituents(
            mol, effective_parent,
            parent_type=parent_type,
            oriented_ring=oriented_ring,
            principal_chain=principal_chain,
            atom_to_locant=atom_to_locant,
        )
    except (AssertionError, Exception) as exc:
        logger.debug("_integrate_universal_prefixes discovery failed: %s", exc)
        return ""

    if not subs:
        return ""

    # Name each substituent and group by name for multiplier handling
    prefix_groups = defaultdict(list)
    for sub_info in subs:
        attach_idx = _find_attach_idx_in_frag(mol, sub_info, effective_parent)
        prefix_name = name_substituent(mol, sub_info.frag_atoms, attach_idx)
        if prefix_name and prefix_name != "substituent":
            prefix_groups[prefix_name].append(sub_info.locant)

    if not prefix_groups:
        return ""

    # Format with locants, multipliers, and alphabetical sorting
    return _format_prefix_groups(prefix_groups)


# ============================================================================
# Handler Enrichment Helper (Phase 139 ARCH-03/04)
# ============================================================================


def _enrich_handler_name(features, base_name, handler_id="unknown"):
    """Enrich a handler's base name with non-principal substituents.

    Standard enrichment wrapper for Tier B handlers (Phase 139 ARCH-03/04).
    Discovers substituents not already accounted for in the handler's base name,
    names them, and prepends as alphabetized prefixes.

    Args:
        features: MolecularFeatures object.
        base_name: The handler's base name (e.g., "carbamic acid").
        handler_id: Handler identifier for logging.

    Returns:
        Enriched name with prefixes, or base_name if no enrichment needed.
    """
    # Determine parent atoms based on what the handler named
    parent_atoms = None
    atom_to_locant = None

    if getattr(features, 'chain_is_parent', False) and features.principal_chain:
        parent_atoms = set(features.principal_chain)
        atom_to_locant = features.atom_to_locant
    elif getattr(features, 'oriented_ring', None):
        parent_atoms = set(features.oriented_ring)
        atom_to_locant = (
            getattr(features, 'heterocycle_atom_to_locant', None)
            or features.atom_to_locant
        )
    elif getattr(features, 'principal_ring', None):
        parent_atoms = set(features.principal_ring)
        atom_to_locant = features.atom_to_locant

    if not parent_atoms:
        return base_name

    # Exclude FG atoms that are already represented in the handler name
    exclude_atoms = set()
    if features.principal_group and features.principal_group in features.functional_groups:
        for match in features.functional_groups[features.principal_group]:
            exclude_atoms.update(match)
    # Also exclude FG atoms for non-principal-group FGs that are already
    # in the handler's detection key (e.g., isocyanate, urea, guanidine,
    # carbamate). These handlers fire via functional_groups.get(fg) checks.
    for _fg_key in ('isocyanate', 'isothiocyanate', 'carbamic_acid',
                     'carbamate', 'urea', 'guanidine', 'boronic_acid',
                     'oxime', 'hydrazone', 'sulfoxide', 'sulfone', 'thioether'):
        if _fg_key in features.functional_groups and _fg_key != getattr(features, 'principal_group', None):
            for match in features.functional_groups[_fg_key]:
                exclude_atoms.update(match)
    # Also exclude N-substituent atoms already named by the handler
    for n_sub in getattr(features, 'n_substituents', []):
        if isinstance(n_sub, dict) and 'atoms' in n_sub:
            exclude_atoms.update(n_sub['atoms'])

    prefix_str = _integrate_universal_prefixes(
        features.mol, parent_atoms,
        parent_type="chain" if getattr(features, 'chain_is_parent', False) else "ring",
        oriented_ring=getattr(features, 'oriented_ring', None),
        principal_chain=(
            features.principal_chain
            if getattr(features, 'chain_is_parent', False)
            else None
        ),
        atom_to_locant=atom_to_locant,
        exclude_atoms=exclude_atoms,
    )

    enriched_name = f"{prefix_str}{base_name}" if prefix_str else base_name

    # Observational coverage logging (ARCH-06)
    if logger.isEnabledFor(logging.DEBUG):
        total_ha = features.mol.GetNumHeavyAtoms()
        accounted = set(parent_atoms) | exclude_atoms
        hr = HandlerResult(
            name=enriched_name,
            handler_id=handler_id,
            parent_atoms=set(parent_atoms),
            accounted_atoms=accounted,
            total_heavy_atoms=total_ha,
        )
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=%.2f accounted=%d/%d name=%s",
            hr.handler_id, hr.coverage, len(hr.accounted_atoms),
            hr.total_heavy_atoms, hr.name[:60],
        )

    return enriched_name


def _confidence_gate(name: str, handler_id: str, features) -> bool:
    """Check if a Tier B handler name meets the confidence threshold.

    Computes confidence score for the handler's output and returns True
    if the name is acceptable (confidence >= CONFIDENCE_GATE_THRESHOLD),
    False if the handler should fall through to the next handler.

    This gates Tier B handlers to prevent low-quality names from being
    returned when the handler only covers a small fraction of the molecule.

    Args:
        name: The handler's output name (after enrichment).
        handler_id: Handler identifier (e.g., 'boronic_acid', 'urea').
        features: MolecularFeatures object.

    Returns:
        True if name should be accepted, False if handler should fall through.
    """
    from .coverage_scoring import compute_confidence, CONFIDENCE_GATE_THRESHOLD

    cand = compute_confidence(name, handler_id, features)
    if cand.confidence < CONFIDENCE_GATE_THRESHOLD:
        logger.debug(
            "CONFIDENCE_GATE: %s rejected (%.2f < %.2f) name=%s",
            handler_id, cand.confidence, CONFIDENCE_GATE_THRESHOLD, name[:60],
        )
        return False
    return True


def _find_attach_idx_in_frag(mol, sub_info, parent_atoms):
    """Find the attachment atom index within a substituent fragment.

    The attachment atom is the atom in ``sub_info.frag_atoms`` that is
    bonded to an atom in ``parent_atoms``.

    Args:
        mol: RDKit Mol object.
        sub_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices (including excluded atoms).

    Returns:
        Atom index of the attachment atom, or the first atom in frag_atoms
        as a fallback.
    """
    for idx in sub_info.frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in parent_atoms:
                return idx
    # Fallback: use attach_mol_idx from sub_info if it's in frag_atoms
    if sub_info.attach_mol_idx in sub_info.frag_atoms:
        return sub_info.attach_mol_idx
    # Last resort: first atom in frag
    if sub_info.frag_atoms:
        return next(iter(sub_info.frag_atoms))
    return 0


def _format_prefix_groups(prefix_groups):
    """Format named substituent groups into an IUPAC prefix string.

    Groups substituents by name, applies multiplicative prefixes
    (di-, tri-, bis-, tris-), sorts alphabetically per IUPAC rules,
    and joins with hyphens.

    Args:
        prefix_groups: Dict mapping prefix name to list of locants.
            Example: ``{"methyl": [2], "chloro": [3, 5]}``

    Returns:
        Formatted prefix string (e.g., ``"3,5-dichloro-2-methyl"``).
        Trailing hyphen is NOT included.
    """
    parts = []
    for name in sorted(prefix_groups.keys(), key=alpha_sort_key):
        locants = sorted(prefix_groups[name])
        count = len(locants)
        prefix_str = format_substituent_prefix(name, locants, count)
        parts.append(prefix_str)

    if not parts:
        return ""

    return "-".join(parts)


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
# Canonical source: TERMINAL_FG_TYPES in naming_utils.py
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
    "thioic_S_acid",    # Always at chain end (locant 1)
    "thioic_O_acid",    # Always at chain end (locant 1)
    "dithioic_acid",    # Always at chain end (locant 1)
    "carbamic_acid",    # Retained name, terminal (locant 1)
}

# Token list for DROP-04 validation: ring+heteroatom branch names must contain
# a recognized ring system identifier to avoid passing linearized-ring names.
_RING_NAME_TOKENS = (
    # Monocyclic
    'cyclo', 'phenyl', 'pyri', 'piper', 'morphol',
    'furan', 'thio', 'indol', 'pyrrol', 'imidaz',
    'oxan', 'oxol', 'azetidin', 'aziridin',
    # Fused heterocyclic
    'quinolin', 'isoquinolin', 'benzofur', 'benzothio',
    'benzimidaz', 'chromen', 'chromane', 'chroman', 'xanthen',
    'carbazol', 'acridin', 'phenazin', 'phenoxazin',
    'phenothiazin', 'thianthr', 'purin', 'indazol',
    'benzotriazol', 'benzoxazol', 'benzisoxazol',
    'benzothiazol', 'coumarin', 'naphthyridin',
    'pteridin', 'indolizin', 'isoindol', 'naphth',
    # Phase 79-02: additional fused het tokens
    'benzisothiaz', 'benzothiadiaz', 'benzoxadiaz',
    'cinnol', 'isochroman', 'phthalaz', 'quinaz', 'quinox',
)


def _has_ring_atoms(mol, frag_atoms):
    """Check if any atom in frag_atoms belongs to a ring in mol.

    Used to replace the pure string-based _RING_NAME_TOKENS check with a
    structural validation: only reject a substituent name as "linearized ring"
    if the fragment actually contains ring atoms (Phase 85 USUB-06).
    """
    ring_info = mol.GetRingInfo()
    return any(ring_info.NumAtomRings(idx) > 0 for idx in frag_atoms)


def _name_reflects_ring(name):
    """Check if a generated name contains any ring system identifier.

    Uses the existing _RING_NAME_TOKENS list for substring matching.
    """
    name_lower = name.lower()
    return any(tok in name_lower for tok in _RING_NAME_TOKENS)


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
    """Assemble complete IUPAC name from molecular features.

    PUBLIC API + RECURSION-SAFE WRAPPER (Phase 145.1 drift fix, 2026-04-23).

    Each invocation gets its own CandidatePool scope per IUPAC P-44.0
    "selection of a preferred parent structure is based on the seniority
    of classes" — applied per-molecule, single-pass. The wrapper pushes a
    fresh pool on the per-thread stack before delegating to the body, and
    pops it in a finally clause so every exit path (normal return, early
    return, exception) restores the previous pool for the calling frame.

    This fixes the byte-identical drift discovered in Plan 04 where
    recursive name_compound() calls (N-oxide handler, fragment naming,
    substituent enumeration, decomposition fallback) shared a single
    thread-local pool with the outer molecule, causing inner candidates
    to pollute pool[0] and beat the outer molecule's correct candidate
    under selection_mode='first_applicable'. See:
    .

    Args:
        features: MolecularFeatures object with extracted features
        style: Naming style ("pin", "general", "cas")
        _composing_ion: Internal recursion guard. When True, ion aspect
            composition is skipped to prevent infinite loops. Do not
            set manually -- it is used by _try_ion_aspect_composition().

    Returns:
        Complete IUPAC name string
    """
    push_pool()
    try:
        return _assemble_name_impl(features, style, _composing_ion)
    finally:
        pop_pool()


def _assemble_name_impl(features: Any, style: str = "pin", _composing_ion: bool = False) -> str:
    """Implementation body of assemble_name(). Do NOT call directly — call
    assemble_name() instead so the per-call pool scope is set up correctly.

    Split out from assemble_name() in 2026-04-23 to add the push_pool /
    pop_pool wrapper without indenting the original 1000+ line body.
    Behavior is unchanged from the pre-fix assemble_name() body.
    """
    # Clear confidence store at start of each naming call
    clear_confidence()

    # Phase 145.1: reset thread-local pool for this naming call (D-09 lifecycle)
    # MUST appear next to clear_confidence() to guarantee per-call state
    # isolation between consecutive orthonym.name() calls (T-145.1-02
    # mitigation, verified by tests/integration/test_pool_state_isolation.py).
    clear_pool()

    # INST: Assembly dispatch trace
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "ASSEMBLY_DISPATCH: smiles=%s species=%s cyclic=%s pg=%s",
            getattr(features, 'canonical_smiles', ''),
            getattr(features, 'species_type', 'neutral'),
            getattr(features, 'is_cyclic', False),
            getattr(features, 'principal_group', None),
        )

    # Check for ionic/radical species first - route to specialized assembly
    species_type = getattr(features, 'species_type', 'neutral')

    # =====================================================================
    # ION / SALT / RADICAL ROUTING — Phase 145.1 PRE-POOL INTENTIONAL BYPASS
    # =====================================================================
    # The four return statements at lines 694, 699, 708, 710 (assemble_ion_name
    # and composed_name) execute BEFORE pool.add() is called for this molecule.
    # They are EXCLUDED from pool dispatch by design per RESEARCH §9.2 + ISS-001
    # enumeration:
    #
    # - salt/zwitterion (L694), radical (L699), ion fallback (L710): route to
    #   assemble_ion_name() which uses IUPAC P-73 functional class naming with
    #   completely different semantics from the cascading direct-return /
    #   Tier B / Tier A handlers. They have NO entry in HANDLER_POLICIES and
    #   never participate in candidate competition.
    # - composed_name (L708): early product of _try_ion_aspect_composition()
    #   for ion aspects (multi-component salts). Same exclusion as above.
    #
    # Phase 146 does NOT change this exclusion (ions stay outside the pool;
    # the pool is for neutral-molecule competition only).
    # =====================================================================

    # ASML-10 by-design: Salt/zwitterion handlers use functional class naming
    # (IUPAC P-73). Ions have no detachable prefixes -- the ion composition
    # IS the name. No substituent discovery needed.
    # Stereo: handled by assemble_ion_name() (ions/salts rarely have stereo in benchmark)
    if species_type in ('salt', 'zwitterion'):
        return assemble_ion_name(features, features.mol, style)

    # ASML-10 by-design: Radical handler uses specialized naming (IUPAC P-68).
    # Stereo: handled by assemble_ion_name()
    if species_type == 'radical':
        return assemble_ion_name(features, features.mol, style)

    # ASML-10 by-design: Single-component ion uses aspect composition or
    # assemble_ion_name(). Ion naming follows IUPAC P-73 functional class
    # naming; no detachable prefixes needed.
    # Stereo: handled by assemble_ion_name() / aspect composition
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
            oxime_name = _enrich_handler_name(features, oxime_name, "oxime")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            # Pool's gate_threshold=0.40 (HANDLER_POLICIES['oxime']) reproduces
            # the deleted _confidence_gate() check bit-for-bit.
            pool = get_current_pool()
            cand = pool.add(oxime_name, "oxime", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # Handle hydrazones - functional class naming: "propan-2-one hydrazone"
    if features.principal_group == 'hydrazone':
        hydrazone_name = _name_oxime_or_hydrazone(features, 'hydrazone')
        if hydrazone_name:
            hydrazone_name = _enrich_handler_name(features, hydrazone_name, "hydrazone")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(hydrazone_name, "hydrazone", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 verified: N-oxide handler creates modified molecule and names
    # recursively via name_compound(). The recursive call handles substituents
    # through whatever handler matches the base compound. Verified: "4-methylpyridine
    # 1-oxide" correctly includes methyl via recursive path. No enrichment needed.
    # Must detect early because N-oxides have internal charges that could
    # confuse other routing (they are classified as 'neutral' by ions.py)
    n_oxide_name = _try_name_n_oxide(features)
    if n_oxide_name:
        # N-oxide names the heterocycle with N-oxide. Use heterocycle locant map if available.
        _noxide_locant_map = getattr(features, 'heterocycle_atom_to_locant', None) or features.atom_to_locant
        # Phase 145.1: route through pool.add() — direct_return handler.
        pool = get_current_pool()
        pool.add(n_oxide_name, "n_oxide", features)
        return _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=_noxide_locant_map)

    # ASML-10 complete: Isocyanate handler calls _name_r_group() which uses
    # the Phase 125 non_ring_heavy fix and name_substituent() fallback for
    # substituted aromatic R-groups. Substituted phenyl correctly named.
    # Only when isocyanate is the sole FG (principal_group is None because
    # isocyanate is not in SENIORITY_ORDER). When another FG is principal,
    # isocyanate becomes prefix "isocyanato" via the polyfunctional handler.
    if (features.functional_groups.get('isocyanate')
            and features.principal_group is None):
        iso_name = _name_isocyanate(features)
        if iso_name:
            iso_name = _enrich_handler_name(features, iso_name, "isocyanate")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(iso_name, "isocyanate", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 complete: Isothiocyanate handler uses same _name_r_group() path
    # as isocyanate -- Phase 125 fix applies. Same gating as isocyanate above.
    if (features.functional_groups.get('isothiocyanate')
            and features.principal_group is None):
        isothio_name = _name_isothiocyanate(features)
        if isothio_name:
            isothio_name = _enrich_handler_name(features, isothio_name, "isothiocyanate")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(isothio_name, "isothiocyanate", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 complete: Carbamic acid handler calls _name_r_group() which uses
    # the Phase 125 fix for substituted aromatic R-groups.
    # Retained name with N-substitution (IUPAC P-65.2.3)
    # N-C(=O)-OH -> "carbamic acid", "N-methylcarbamic acid", etc.
    if features.principal_group == 'carbamic_acid':
        carbamic_name = _name_carbamic_acid(features)
        if carbamic_name:
            carbamic_name = _enrich_handler_name(features, carbamic_name, "carbamic_acid")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(carbamic_name, "carbamic_acid", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 complete: Carbamate handler calls _name_r_group() (Phase 125 fix)
    # for both N- and O-substituent naming.
    # Must detect BEFORE generic ester to prevent N loss.
    # Only when carbamate is the primary FG (no higher-seniority principal group).
    if (features.functional_groups.get('carbamate')
            and features.principal_group is None):
        carb_name = _name_carbamate(features)
        if carb_name:
            carb_name = _enrich_handler_name(features, carb_name, "carbamate")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(carb_name, "carbamate", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 complete: Urea handler calls _name_r_group() (Phase 125 fix)
    # for N-substituent naming. Retained name with N-substitution.
    # "urea", "N-methylurea", "N,N-dimethylurea", "N,N'-dimethylurea"
    # Must detect BEFORE polyfunctional handler to prevent garbled output
    if (features.functional_groups.get('urea')
            and features.principal_group is None):
        urea_name = _try_name_urea(features)
        if urea_name:
            urea_name = _enrich_handler_name(features, urea_name, "urea")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(urea_name, "urea", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 complete: Guanidine handler calls _name_r_group() (Phase 125 fix)
    # for N-substituent naming. Retained name with N-substitution.
    # "guanidine", "N-methylguanidine", "N,N-dimethylguanidine"
    if (features.functional_groups.get('guanidine')
            and features.principal_group is None):
        guanidine_name = _try_name_guanidine(features)
        if guanidine_name:
            guanidine_name = _enrich_handler_name(features, guanidine_name, "guanidine")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(guanidine_name, "guanidine", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 complete: Acid halide handler (acid_halides.py) uses its own
    # chain/ring parent naming with suffix. Substituents handled via normal
    # prefix generation in the dedicated rule module.
    # Acid halides use functional class naming: "ethanoyl chloride" (two-word)
    # Must come before polyfunctional because acid_chloride + chloro triggers polyfunctional
    if features.principal_group in ('acid_chloride', 'acid_bromide', 'acid_fluoride'):
        from ..rules.acid_halides import name_acid_halide
        halide_name = name_acid_halide(features)
        if halide_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "acid_halide", _ha, halide_name[:60],
                )
            # Phase 145.1: route through pool.add() — direct_return handler.
            pool = get_current_pool()
            pool.add(halide_name, "acid_halide", features)
            return _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)

    # ASML-10 complete: Anhydride handler (anhydrides.py) uses its own
    # component naming. Substituents handled within the module.
    # Anhydrides use functional class naming: "ethanoic anhydride" (two-word)
    # Must come before lactone because cyclic anhydrides (O=C1CCC(=O)O1) would
    # otherwise be misidentified as lactones
    if features.principal_group == 'anhydride':
        from ..rules.anhydrides import name_anhydride
        anhydride_name = name_anhydride(features)
        if anhydride_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "anhydride", _ha, anhydride_name[:60],
                )
            # Phase 145.1: route through pool.add() — direct_return handler.
            pool = get_current_pool()
            pool.add(anhydride_name, "anhydride", features)
            return _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)

    # ASML-10 complete: Lactone handler calls _integrate_universal_prefixes()
    # for exocyclic substituent discovery (Phase 86 wired, lactones.py).
    # Lactones are cyclic esters named as heterocyclic ketones.
    # Must come before polyfunctional because lactones trigger polyfunctional detection
    # Coverage guard: if molecule is much larger than the lactone ring, the bare
    # lactone name is incomplete and we should fall through to a handler that
    # can include substituents (e.g., benzene handler, polyfunctional handler).
    from ..rules.lactones import is_monocyclic_lactone, name_monocyclic_lactone
    lactone_info = is_monocyclic_lactone(features.mol)
    if lactone_info:
        total_heavy = features.mol.GetNumHeavyAtoms()
        ring_size = lactone_info.get('ring_size', 0)
        # Only use bare lactone naming when molecule is not much larger than ring
        # Ring atoms + carbonyl O + up to 8 exocyclic heavy atoms = ring_size + 8
        # For macrocycles (ring_size > 8), the ring IS the parent — skip guard
        if ring_size > 8 or total_heavy <= ring_size + 8:
            lactone_name = name_monocyclic_lactone(features.mol)
            if lactone_name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "lactone", _ha, lactone_name[:60],
                    )
                # Lactone handler already includes stereo with correct ring locants.
                # Pass None -- regex guard detects existing stereo prefix and returns early.
                # Phase 145.1: route through pool.add() — direct_return handler.
                pool = get_current_pool()
                pool.add(lactone_name, "lactone", features)
                return _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)

    # ASML-10 complete: Lactam handler calls _integrate_universal_prefixes()
    # for exocyclic substituent discovery (parallel to lactones).
    # Lactams are cyclic amides named as heterocyclic ketones.
    # e.g., azetidin-2-one, pyrrolidin-2-one, piperidin-2-one
    # Same coverage guard as lactones above.
    from ..rules.lactams import is_monocyclic_lactam, name_monocyclic_lactam
    lactam_info = is_monocyclic_lactam(features.mol)
    if lactam_info:
        total_heavy = features.mol.GetNumHeavyAtoms()
        ring_size = lactam_info.get('ring_size', 0)
        # For macrocycles (ring_size > 8), the ring IS the parent — skip guard
        if ring_size > 8 or total_heavy <= ring_size + 8:
            lactam_name = name_monocyclic_lactam(features.mol)
            if lactam_name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "lactam", _ha, lactam_name[:60],
                    )
                # Lactam handler already includes stereo with correct ring locants.
                # Pass None -- regex guard detects existing stereo prefix and returns early.
                # Phase 145.1: route through pool.add() — direct_return handler.
                pool = get_current_pool()
                pool.add(lactam_name, "lactam", features)
                return _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)

    # ASML-10 complete: Ring-attached ester handler uses
    # _assemble_ring_with_ester_prefixes() which generates acyloxy prefixes
    # on the ring parent. Ring substituents handled via ring naming path.
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
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "ring_ester", _ha, ring_ester_name[:60],
                    )
                # Phase 145.1: route through pool.add() — direct_return handler.
                pool = get_current_pool()
                pool.add(ring_ester_name, "ring_ester", features)
                return _inject_stereo_if_missing(features, pool.best().name)

    # ASML-10 complete: Polyfunctional handler (polyfunctional.py) uses
    # _integrate_universal_prefixes() for substituent discovery on both
    # chain and ring parents. Stereo: handled by name_polyfunctional() internally.
    if getattr(features, 'is_polyfunctional', False):
        from ..rules.polyfunctional import name_polyfunctional
        poly_name = name_polyfunctional(features)
        if poly_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "polyfunctional", _ha, poly_name[:60],
                )
            # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
            pool = get_current_pool()
            pool.add(poly_name, "polyfunctional", features)
            return pool.best().name
        # If name_polyfunctional returns None, fall through to normal handling
        # (by-design: specialized handlers produce correct names via fallthrough)
        logger.debug(
            "DROP-22 substituent_skip: reason=polyfunctional_returned_none",
        )

    # Handle multi-ester compounds (dicarboxylic acid diesters)
    # Must come BEFORE single-ester naming so 2-ester diacids are caught
    if features.principal_group == "ester":
        all_esters = getattr(features, 'all_ester_matches', None)
        if all_esters and len(all_esters) >= 2 and not getattr(features, 'is_polyfunctional', False):
            from ..rules.esters import classify_multi_ester, name_dicarboxylic_diester, name_polyol_polyester, name_independent_esters
            # Stereo: handled by ester naming functions (acid-fragment stereo internally)
            ester_type = classify_multi_ester(features.mol, all_esters)
            if ester_type == "dicarboxylic_diester":
                diester_name = name_dicarboxylic_diester(features.mol, all_esters)
                if diester_name:
                    if logger.isEnabledFor(logging.DEBUG):
                        _ha = features.mol.GetNumHeavyAtoms()
                        logger.debug(
                            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                            "multi_ester", _ha, diester_name[:60],
                        )
                    # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
                    pool = get_current_pool()
                    pool.add(diester_name, "multi_ester", features)
                    return pool.best().name
            elif ester_type == "polyol_polyester":
                polyol_name = name_polyol_polyester(features.mol, all_esters)
                if polyol_name:
                    if logger.isEnabledFor(logging.DEBUG):
                        _ha = features.mol.GetNumHeavyAtoms()
                        logger.debug(
                            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                            "multi_ester", _ha, polyol_name[:60],
                        )
                    # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
                    pool = get_current_pool()
                    pool.add(polyol_name, "multi_ester", features)
                    return pool.best().name
            elif ester_type == "independent":
                indep_name = name_independent_esters(features.mol, all_esters)
                if indep_name:
                    if logger.isEnabledFor(logging.DEBUG):
                        _ha = features.mol.GetNumHeavyAtoms()
                        logger.debug(
                            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                            "multi_ester", _ha, indep_name[:60],
                        )
                    # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
                    pool = get_current_pool()
                    pool.add(indep_name, "multi_ester", features)
                    return pool.best().name
                # If returns None, fall through to single ester or decomposition

    # ASML-10 complete: Ester handler (esters.py) names acid and alkyl
    # components independently. Acid-fragment stereo handled internally.
    # Only reached for acyclic esters (ring-attached esters handled above)
    if features.principal_group == "ester":
        ester_match = getattr(features, 'ester_match', None)
        if ester_match:
            from ..rules.esters import name_ester
            # Simple acyclic ester: use "alkyl alkanoate" naming
            # Stereo: handled by name_ester() (collects acid-fragment stereo internally)
            ester_name = name_ester(features.mol, ester_match)
            if ester_name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "ester", _ha, ester_name[:60],
                    )
                # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
                pool = get_current_pool()
                pool.add(ester_name, "ester", features)
                return pool.best().name
        # If name_ester returns None (lactone or complex), fall through

    # ASML-10 self-gating: Sulfoxide/sulfone handlers use _count_alkyl_carbons()
    # which returns None for any non-simple-alkyl R-group. The handler returns
    # None, and the molecule falls through to polyfunctional or chain/ring parent
    # path where the universal pipeline operates. No substituents silently dropped.
    if features.principal_group in ('sulfoxide', 'sulfone'):
        from ..rules.sulfur import name_sulfoxide, name_sulfone
        if features.principal_group == 'sulfoxide':
            matches = features.functional_groups.get('sulfoxide', [])
            if matches:
                name = name_sulfoxide(features.mol, matches[0])
                if name:
                    name = _enrich_handler_name(features, name, "sulfoxide")
                    # Phase 145.1: route through pool.add() — returns None on gate-fail.
                    pool = get_current_pool()
                    cand = pool.add(name, "sulfoxide", features)
                    if cand is not None:
                        return _inject_stereo_if_missing(features, cand.name)
                    # Low confidence: pool.add returned None, fall through
        elif features.principal_group == 'sulfone':
            matches = features.functional_groups.get('sulfone', [])
            if matches:
                name = name_sulfone(features.mol, matches[0])
                if name:
                    name = _enrich_handler_name(features, name, "sulfone")
                    # Phase 145.1: route through pool.add() — returns None on gate-fail.
                    pool = get_current_pool()
                    cand = pool.add(name, "sulfone", features)
                    if cand is not None:
                        return _inject_stereo_if_missing(features, cand.name)
                    # Low confidence: pool.add returned None, fall through

    # ASML-10 self-gating: Thioether handler uses name_sulfide() which returns
    # None for complex R-groups. Falls through to universal pipeline. No silent drop.
    # Skip cyclic thioethers (1,3-dithiane, thiane, etc.) - they are named as heterocycles
    if features.principal_group == 'thioether':
        ring_type = getattr(features, 'ring_type', None)
        if not (ring_type and ring_type.startswith('heterocyclic')):
            # Guard: skip known fused heterocycles (phenothiazine, thianthrene)
            # These contain S atoms that match thioether SMARTS but should use
            # their retained fused heterocycle names, not functional class sulfide naming.
            from ..data.fused_heterocycles import match_fused_heterocycle_core
            if match_fused_heterocycle_core(features.mol) is None:
                from ..rules.sulfur import name_sulfide
                # Find sulfur atom index
                matches = features.functional_groups.get('thioether', [])
                if matches:
                    # SMARTS match gives (S, C, C) - sulfur is first
                    sulfur_idx = matches[0][0]
                    name = name_sulfide(features.mol, sulfur_idx)
                    if name:
                        name = _enrich_handler_name(features, name, "thioether")
                        # Phase 145.1: route through pool.add() — returns None on gate-fail.
                        pool = get_current_pool()
                        cand = pool.add(name, "thioether", features)
                        if cand is not None:
                            return _inject_stereo_if_missing(features, cand.name)
                        # Low confidence: pool.add returned None, fall through

    # ASML-10 self-gating: Phosphorus handlers use _characterize_substituent()
    # which returns None for non-phenyl/non-simple-alkyl R-groups. Complex
    # molecules fall through to universal pipeline. No substituents silently dropped.
    # Tier C (Phase 139): Self-gating design confirmed complete -- when handlers
    # return a name, molecule is simple enough that no additional enrichment needed.
    # When molecule is complex, handlers return None and fall through to universal pipeline.
    if features.principal_group == 'phosphine_oxide':
        from ..rules.phosphorus import name_phosphine_oxide
        matches = features.functional_groups.get('phosphine_oxide', [])
        if matches:
            name = name_phosphine_oxide(features.mol, matches[0])
            if name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "phosphine_oxide", _ha, name[:60],
                    )
                # Phase 145.1: route through pool.add() — direct_return handler.
                pool = get_current_pool()
                pool.add(name, "phosphine_oxide", features)
                return _inject_stereo_if_missing(features, pool.best().name)

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
                        if logger.isEnabledFor(logging.DEBUG):
                            _ha = features.mol.GetNumHeavyAtoms()
                            logger.debug(
                                "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                                "phosphate_ester", _ha, name[:60],
                            )
                        # Phase 145.1: route through pool.add() — direct_return handler.
                        pool = get_current_pool()
                        pool.add(name, "phosphate_ester", features)
                        return _inject_stereo_if_missing(features, pool.best().name)
                    break

    # Handle phosphines (tertiary, secondary, primary)
    if features.principal_group in ('tertiary_phosphine', 'secondary_phosphine', 'primary_phosphine'):
        from ..rules.phosphorus import name_phosphine
        fg_key = features.principal_group
        matches = features.functional_groups.get(fg_key, [])

        # When molecule is a benzene derivative with P on the ring alongside
        # other substituents (or multiple P atoms), prefer benzene-as-parent naming.
        # The P will be expressed as a phosphanyl prefix on benzene.
        _skip_for_benzene = False
        if getattr(features, 'is_benzene', False) and matches:
            if len(matches) > 1:
                # Multiple phosphine groups -> ring should be parent
                _skip_for_benzene = True
            else:
                # Single phosphine -- check if any benzene ring atom has
                # non-P, non-ring substituent neighbors
                _bz_ring = getattr(features, 'benzene_ring', None)
                if _bz_ring:
                    _bz_set = set(_bz_ring)
                    for _ra in _bz_ring:
                        _ra_atom = features.mol.GetAtomWithIdx(_ra)
                        for _nbr in _ra_atom.GetNeighbors():
                            if _nbr.GetIdx() not in _bz_set and _nbr.GetSymbol() != 'P':
                                _skip_for_benzene = True
                                break
                        if _skip_for_benzene:
                            break

        if not _skip_for_benzene and matches:
            # Find phosphorus atom index
            for idx in matches[0]:
                atom = features.mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'P':
                    name = name_phosphine(features.mol, idx)
                    if name:
                        if logger.isEnabledFor(logging.DEBUG):
                            _ha = features.mol.GetNumHeavyAtoms()
                            logger.debug(
                                "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                                "phosphine", _ha, name[:60],
                            )
                        # Phase 145.1: route through pool.add() — direct_return handler.
                        pool = get_current_pool()
                        pool.add(name, "phosphine", features)
                        return _inject_stereo_if_missing(features, pool.best().name)
                    break

    # Handle phosphinic acid (suffix naming, but needs special assembly)
    if features.principal_group == 'phosphinic_acid':
        from ..rules.phosphorus import name_phosphinic_acid
        matches = features.functional_groups.get('phosphinic_acid', [])
        if matches:
            name = name_phosphinic_acid(features.mol, matches[0])
            if name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "phosphinic_acid", _ha, name[:60],
                    )
                # Phase 145.1: route through pool.add() — direct_return handler.
                pool = get_current_pool()
                pool.add(name, "phosphinic_acid", features)
                return _inject_stereo_if_missing(features, pool.best().name)

    # ASML-10 complete: Boronic acid handler calls _name_r_group() (Phase 125
    # fix) for R-group naming. Substituted aromatic R-groups correctly named.
    if features.principal_group == 'boronic_acid':
        boronic_name = _name_boronic_acid(features)
        if boronic_name:
            boronic_name = _enrich_handler_name(features, boronic_name, "boronic_acid")
            # Phase 145.1: route through pool.add() — returns None on gate-fail.
            pool = get_current_pool()
            cand = pool.add(boronic_name, "boronic_acid", features)
            if cand is not None:
                return _inject_stereo_if_missing(features, cand.name)
            # Low confidence: pool.add returned None, fall through to next handler

    # ASML-10 verified: Ring assembly has own _get_substituent_info() at
    # ring_assemblies.py which discovers substituents via BFS + _name_substituent().
    # Verified: ring assembly substituent discovery covers halogens, hydroxy, amino,
    # and alkyl groups. No enrichment needed.
    # Ring assemblies are separate identical ring systems connected by single bonds
    assembly_info = getattr(features, 'ring_assembly_info', None)
    if assembly_info and not getattr(features, 'chain_is_parent', False):
        from ..rules.ring_assemblies import name_ring_assembly
        assembly_name = name_ring_assembly(features.mol, assembly_info, features)
        if assembly_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "ring_assembly", _ha, assembly_name[:60],
                )
            # Phase 145.1: route through pool.add() — direct_return handler.
            pool = get_current_pool()
            pool.add(assembly_name, "ring_assembly", features)
            return _inject_stereo_if_missing(features, pool.best().name)

    # =========================================================================
    # TIER A RING COMPETITION — Phase 145.1 routes through CandidatePool
    # =========================================================================
    #
    # PHASE 146 PRESERVE — DO NOT remove this comment block when Phase 146
    # deletes the cascade gates. The Tier A subset slicing pattern below
    # (`_tier_a_pool.all_candidates()[_tier_a_pool_count_before_complex:]`)
    # MUST survive Phase 146's gate deletion. Why:
    #
    # In selection_mode='first_applicable' (145.1) the pool's ordering reflects
    # the dispatch order. The pool already contains candidates from earlier
    # Tier B / direct-return handlers (those handlers add to pool BEFORE
    # returning their direct result). When Tier A reaches its final dispatch,
    # we MUST select the best candidate from JUST the Tier A subset — not
    # from the whole pool — otherwise an earlier Tier B candidate (e.g. a
    # gate-rejected oxime that didn't return) would pollute Tier A selection.
    #
    # In Phase 146's selection_mode='score_based', this slicing pattern is
    # ALSO necessary: Tier A handlers compete via select_best_candidate on
    # the Tier A subset; non-Tier-A candidates (chain, direct-return) compete
    # via the FINAL pool.best() call after fall-through. The two competitions
    # are SEPARATE by design — Tier A picks the best ring; chain decides
    # later whether to override. Mixing them here would cause chain to win
    # over benzene on small rings, breaking IUPAC P-44.1 cascade order.
    #
    # PHASE 146 CHANGES: the cascade_ratio_min=0.40 gate is removed, and the
    # /1.5 ratio exit gate is replaced by the D-10-preserved literal 0.20
    # fallback (see _MIN_RATIO_FALLBACK below), but the SLICING PATTERN
    # remains. Reviewer of Phase 146's diff: verify the
    # `[_tier_a_pool_count_before_complex:]` slice is preserved.
    # =========================================================================

    # Tier A ring competition — routes through CandidatePool.
    # Phase 146 SC-4: the complex-ring cascade ratio gate (0.40 threshold)
    # is REMOVED (zero IUPAC justification per Blue Book grep — see
    # 146-CONTEXT.md D-09). The cascade now competes via Tier-1
    # lexicographic filters in pool._best_two_tier (V18 mode). In V17
    # first_applicable mode, the `_complex_ring_accepted = True`
    # short-circuit still fires whenever a complex_cand is produced —
    # the cascade priority ordering is preserved without the ratio
    # threshold.
    # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    _complex_ring_accepted = False
    # _gate_candidates was the in-line prototype of CandidatePool. The real
    # pool from get_current_pool() is now the source of truth. We keep a
    # local reference for readability and so the pool.add() calls below have
    # an obvious target (the pool is also used by Tier B handlers above).
    _tier_a_pool = get_current_pool()  # same thread-local pool used by Tier B
    _tier_a_pool_count_before_complex = len(_tier_a_pool.all_candidates())

    # Handle complex ring systems FIRST (bicyclo, spiro, fused heterocycles)
    # These take precedence over simple heterocyclic/benzene classification
    # because fused heterocycles (indole, purine) contain benzene/heterocycle parts
    # that would otherwise trigger early exit to wrong naming path
    if features.is_cyclic and not getattr(features, 'chain_is_parent', False) and _is_complex_ring_system(features.mol):
        complex_result = _assemble_complex_ring_name(features.mol, features)
        if complex_result:
            complex_name = complex_result.name

            # RING-03: Universal pipeline discovers substituents for handlers
            # that don't do their own substituent discovery (spiro, bridged-fused,
            # saturated fused carbocyclic). Handlers that already include subs
            # (fused heterocycles, bicyclo, polycyclic) set substituents_included=True
            # and are skipped to avoid double-counting.
            if (not complex_result.substituents_included
                    and complex_result.atom_to_locant):
                complex_name = _enrich_complex_ring_with_subs(
                    features.mol, complex_name,
                    complex_result.ring_atoms,
                    complex_result.atom_to_locant,
                )

            # Pool.add() is byte-identical to the previous _gate_candidates.append:
            # it calls compute_confidence(complex_name, 'complex_ring', features)
            # internally and stores the candidate. Tier A handlers have
            # direct_return=False (pool decides via best()), so .add() does NOT
            # short-circuit pool.best() to this candidate.
            _complex_cand = _tier_a_pool.add(complex_name, 'complex_ring', features)
            # Phase 146 SC-4: the top-level 0.40 ratio gate is removed from
            # the module-level constants. In V18 score_based mode, the
            # cascade competes via pool._best_two_tier so the cascade-ratio
            # short-circuit is not needed. In V17 first_applicable mode, we
            # preserve the original cascade-priority behavior via an inline
            # 0.40 threshold so byte-identical output on the 7,500-row
            # baseline is maintained (glycoside / polyfunctional regressions
            # observed when the short-circuit fires unconditionally).
            # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
            if _complex_cand is not None:
                if _tier_a_pool.selection_mode == 'score_based':
                    # V18: two-tier selector decides at pool.best(); no gate.
                    _complex_ring_accepted = True
                elif _complex_cand.factors.get('ratio', 0) >= 0.40:
                    # V17 soak: preserve cascade-priority short-circuit.
                    _complex_ring_accepted = True
        # If complex ring naming fails, fall through to simpler handling

    # Handle polycyclic aromatics (naphthalene, anthracene, etc.)
    # Check before benzene since substituted PAHs have benzene substructures
    # NOT gated -- direct return (no coverage quality ambiguity)
    # Skip if complex_ring already identified the system with adequate
    # confidence (it provides a more complete VB/fused name for systems
    # that also have a simpler polycyclic_name or partial_sat match).
    if not _complex_ring_accepted and not getattr(features, 'chain_is_parent', False):
        polycyclic_name = getattr(features, 'polycyclic_name', None)
        if polycyclic_name:
            # Tier C (Phase 139): Polycyclic handler has its own complete substituent
            # handling via features.polycyclic_substituents + name_substituted_polycyclic().
            # Includes suffix groups and prefix groups with proper PAH numbering.
            # No enrichment needed -- verified complete by design.
            # Stereo: handled by name_substituted_polycyclic() internally
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "polycyclic", _ha, (polycyclic_name or "")[:60],
                )
            # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
            poly_assembled = _assemble_polycyclic_name(features, style)
            pool = get_current_pool()
            pool.add(poly_assembled, "polycyclic", features)
            return pool.best().name

        # Handle partially saturated carbocycles (tetrahydronaphthalene, etc.)
        # Check BEFORE benzene since they contain benzene substructure
        # NOT gated -- direct return
        # Stereo: handled by _assemble_partially_saturated_carbocycle_name() internally
        # Phase 139 ARCH-03: enrichment for non-parent substituents
        if features.is_cyclic and not getattr(features, 'chain_is_parent', False):
            partial_sat_name = _try_partially_saturated_carbocycle(features.mol)
            if partial_sat_name:
                partial_sat_name = _enrich_handler_name(features, partial_sat_name, "partial_sat")
                # Phase 145.1: route through pool.add() — Tier B gate-fall-through.
                # NOTE: partial_sat returns cand.name DIRECTLY (no _inject_stereo wrapper).
                pool = get_current_pool()
                cand = pool.add(partial_sat_name, "partial_sat", features)
                if cand is not None:
                    return cand.name
                # Low confidence: pool.add returned None, fall through to next handler

    # Only collect heterocycle/benzene candidates if complex_ring didn't
    # produce a high-confidence result. This preserves the handler priority
    # cascade: fused/complex ring names should not compete with simpler
    # monocyclic names when the fused name is adequate.
    if not _complex_ring_accepted and not getattr(features, 'chain_is_parent', False):
        # Handle simple heterocyclic compounds (pyridine, morpholine, etc.)
        # Only reached if not a complex fused system and ring is actual parent
        # Tier C (Phase 139): Heterocycle handler has internal substituent handling
        # via features.heterocycle_substituents + name_substituted_heterocycle().
        # Verified complete by design -- no enrichment needed.
        ring_type = getattr(features, 'ring_type', None)
        if ring_type and ring_type.startswith('heterocyclic'):
            # Safety net: check if this heterocycle is actually a lactone
            # (catches cases where lactone detection in ester routing was bypassed)
            from ..rules.lactones import is_monocyclic_lactone, name_monocyclic_lactone
            lactone_info = is_monocyclic_lactone(features.mol)
            if lactone_info:
                lactone_name = name_monocyclic_lactone(features.mol)
                if lactone_name:
                    # Phase 145.1 ISS-001: Tier A safety net (DIFFERENT from primary
                    # lactone at L878). Route through pool.add() — direct_return.
                    # Same handler_id "lactone" — same handler, different code path.
                    pool = get_current_pool()
                    pool.add(lactone_name, "lactone", features)
                    return pool.best().name
            # Collect heterocycle candidate via pool.add()
            hetero_name = _assemble_heterocycle_name(features, style)
            if hetero_name:
                _tier_a_pool.add(hetero_name, 'heterocycle', features)
            # If heterocycle naming fails, fall through

        # Handle benzene derivatives
        # Only reached if not a fused system containing benzene
        # Tier C (Phase 139): Benzene handler has internal substituent handling
        # via features.benzene_substituents + name_substituted_benzene().
        # Verified complete by design -- no enrichment needed.
        if getattr(features, 'is_benzene', False):
            benzene_name = _assemble_benzene_name(features, style)
            if benzene_name:
                _tier_a_pool.add(benzene_name, 'benzene', features)
            # If benzene naming fails, fall through

    # Phase 146 SC-5: wire chain as a first-class pool candidate (V18 only).
    # In V17 first_applicable mode, the existing chain path still runs as
    # fallback at the tail of _assemble_name_impl via features.principal_chain
    # -> _generate_chain_parent(). The V18 two-tier cascade instead competes
    # chain against ring candidates in pool._best_two_tier (P-44.1.1 PCG-count
    # filter picks chain over ring when chain has an acid/aldehyde/etc.).
    # Defensive: compute_chain_candidate returns None whenever the chain
    # pipeline is not importable or the molecule lacks a principal_chain,
    # so this block is a no-op in those cases.
    # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1
    if _tier_a_pool.selection_mode == 'score_based':
        from .candidate_pool import compute_chain_candidate
        _chain_cand = compute_chain_candidate(features, style=style)
        if _chain_cand is not None:
            _tier_a_pool.add(
                _chain_cand.name, 'chain', features,
                parent_atom_indices=_chain_cand.parent_atom_indices,
            )

    # --- Select best Tier A candidate if any were collected ---
    # PHASE 146 PRESERVE: the slicing pattern below MUST survive Phase 146's
    # gate deletion (see comment block at top of this section).
    _tier_a_candidates_added = _tier_a_pool.all_candidates()[_tier_a_pool_count_before_complex:]
    if _tier_a_candidates_added:
        # In first_applicable mode, pool.best() returns the FIRST added candidate
        # in the pool overall. That may be a Tier B / direct-return candidate from
        # earlier handlers if any fired. Those handlers all return immediately
        # on success, so reaching this point means none of them fired -- the
        # _tier_a_candidates_added list is the only set of candidates in the pool
        # for the molecule's ring system. Use select_best_candidate over THAT
        # subset to preserve current "Tier A wins on confidence within Tier A"
        # behavior.
        best = select_best_candidate(_tier_a_candidates_added)
        log_confidence(best)
        # Only store confidence at top-level depth (not during recursive
        # fragment naming) to prevent overwriting top-level metadata.
        # Risk 3 (PATTERNS): preserve store_confidence so name_with_confidence()
        # continues returning factor data via retrieve_confidence().
        from .fragment_naming import is_top_level_naming
        if is_top_level_naming():
            store_confidence(best)
        # Accept the best candidate if its name-length ratio is adequate.
        # The ratio factor measures len(name)/heavy_atoms, matching the
        # old binary gate's acceptance criterion. When the ratio is too low,
        # the handler only captured a small substructure and chain naming
        # may produce a more complete name.
        # Phase 146 SC-4: the top-level 0.30 ratio-accept gate has been
        # REMOVED. The internal /1.5 rescue fallback below is PRESERVED
        # per D-10 — it is a low-heavy-atom rescue, NOT a weighted-sum
        # gate. Inlined as the literal 0.20 (= 0.30 / 1.5) for clarity,
        # binding it to a private constant so static introspection (grep)
        # confirms the preservation invariant lives in source.
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
        _MIN_RATIO_FALLBACK = 0.20  # PRESERVED per D-10 (low-HA rescue threshold)
        total_heavy = features.mol.GetNumHeavyAtoms()
        # D-10 PRESERVED: low-heavy-atom rescue fallback (NOT a weighted-sum gate).
        # Small molecules (<=15 heavy atoms) always return the best Tier A
        # candidate; larger molecules require a minimum ratio so chain naming
        # can take over when a small ring fragment is dominating a large molecule.
        if total_heavy <= 15 or best.factors.get('ratio', 0) >= _MIN_RATIO_FALLBACK:
            # Stereo: handled by individual handlers (complex_ring, heterocycle, benzene)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    best.handler, total_heavy, best.name[:60],
                )
            # Phase 152 D-04: predicate-first handler-level stereo injection
            # for benzene + heterocycle Tier-A candidates.  Per D-09, no atom-
            # index fallback.  Per D-11, restrict to top-level naming so
            # substituent decomposition fragments (e.g. thiazolyl substituent
            # on a macrocyclic lactone) do NOT inject near-parent E/Z that
            # belongs to the parent's frame, not the fragment's frame.
            #
            # Atom-coverage guard (auto-deviation Rule 1, 2026-05-03): only
            # inject when the chosen candidate's name covers ~all heavy
            # atoms.  Low-coverage candidates are PARTIAL fragment names
            # (e.g. the benzene handler returning "3-methoxyphenol" for a
            # complex polyketide); injecting stereo for atoms outside the
            # named fragment would attach the wrong locants (near-parent
            # E/Z leak from neighbouring chains).  Per D-09 / D-21, prefer
            # a missing stereo block over a wrong one.
            #
            # Cycloalkane / cycloalkene wiring lands in commit 5 at the
            # chain-fragment fallback (composer.py:1756).
            candidate_name = best.name
            if best.handler in ('benzene', 'heterocycle'):
                from .fragment_naming import is_top_level_naming
                _coverage = best.factors.get('atom_coverage', 0.0)
                if is_top_level_naming() and _coverage >= 0.99:
                    from ..rules.stereochemistry import (
                        needs_stereo_injection, inject_stereo_from_locant_map,
                    )
                    if needs_stereo_injection(features.mol, candidate_name):
                        atom_to_locant = (
                            getattr(features, 'benzene_atom_to_locant', None)
                            if best.handler == 'benzene'
                            else getattr(features, 'heterocycle_atom_to_locant', None)
                        )
                        candidate_name = inject_stereo_from_locant_map(
                            candidate_name, features.mol, atom_to_locant,
                        )
            return candidate_name
        # Low ratio: fall through but store metadata for debugging
        logger.debug(
            "Coverage gate: best candidate ratio too low, falling through "
            "to chain naming. best_handler=%s best_confidence=%.4f ratio=%.4f",
            best.handler, best.confidence, best.factors.get('ratio', 0),
        )
        # BYTE-IDENTICAL FIX: Tier A candidates were rejected by the ratio gate.
        # Truncate them from the pool so subsequent direct-return handlers
        # (ring_nitrile, amide, amine) and the chain fallback don't see them
        # via pool.best() (which returns _candidates[0] in first_applicable mode).
        # Without this truncation, the chain fallback at the end of assemble_name
        # would return the rejected Tier A candidate's name (e.g. "1H-indole" for
        # a 35-atom indole peptide) instead of the chain assembly. DELETE IN
        # PHASE 146 when score_based mode is enabled — score-based selection
        # naturally lets chain win over a low-ratio ring candidate.
        del _tier_a_pool._candidates[_tier_a_pool_count_before_complex:]
    # else: no ring candidates -- fall through to chain/simple naming

    # ASML-10 complete: Ring nitrile handler uses _assemble_ring_nitrile_name()
    # which includes ring substituent prefixes in the name.
    if features.principal_group == 'nitrile' and features.is_cyclic and not getattr(features, 'chain_is_parent', False):
        _rn_name = _assemble_ring_nitrile_name(features, style)
        if logger.isEnabledFor(logging.DEBUG):
            _ha = features.mol.GetNumHeavyAtoms()
            logger.debug(
                "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                "ring_nitrile", _ha, _rn_name[:60],
            )
        # Phase 145.1: route through pool.add() — direct_return handler.
        pool = get_current_pool()
        pool.add(_rn_name, "ring_nitrile", features)
        return _inject_stereo_if_missing(features, pool.best().name)

    # ASML-10 complete: Amide handler uses _assemble_amide_name() which includes
    # N-substituent prefixes and chain/ring substituent discovery.
    # Multi-amide compounds (diamide, triamide) fall through to normal suffix path
    # so that the multiplier prefix (di-, tri-) is correctly applied.
    # Stereo: handled within _assemble_amide_name() -- unsaturated path calls
    # _generate_stereodescriptors(); ring/saturated paths inject stereo at return.
    if features.principal_group in ('primary_amide', 'secondary_amide', 'tertiary_amide'):
        pg_count = len(features.principal_group_atoms) if features.principal_group_atoms else 1
        if pg_count == 1:
            _amide_name = _assemble_amide_name(features, style)
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "amide", _ha, (_amide_name or "")[:60],
                )
            # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
            pool = get_current_pool()
            pool.add(_amide_name, "amide", features)
            return pool.best().name

    # ASML-10 complete: Amine handler uses _assemble_amine_name() which adds
    # N-alkyl prefixes and generates chain/ring substituent prefixes.
    # Stereo: handled by _assemble_amine_name() (calls _generate_stereodescriptors)
    if features.principal_group in ('secondary_amine', 'tertiary_amine'):
        amine_name = _assemble_amine_name(features, style)
        if amine_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "amine", _ha, amine_name[:60],
                )
            # Phase 145.1 ISS-001: route through pool.add() — direct_return handler.
            pool = get_current_pool()
            pool.add(amine_name, "amine", features)
            return pool.best().name

    # Handle simple cases
    # Stereo: not applicable (single atom / very simple molecules have no stereocenters)
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
    assembled = _assemble_fragments(fragments, style)

    # Observational coverage logging for fallback chain/ring path (ARCH-06)
    if logger.isEnabledFor(logging.DEBUG):
        _fb_total_ha = features.mol.GetNumHeavyAtoms()
        _fb_parent = set(features.principal_chain or []) | set(getattr(features, 'principal_ring', None) or [])
        _fb_accounted = set(_fb_parent)
        # Include FG atoms
        for _fb_fg_matches in getattr(features, 'functional_groups', {}).values():
            for _fb_m in _fb_fg_matches:
                _fb_accounted.update(_fb_m)
        _fb_hr = HandlerResult(
            name=assembled,
            handler_id="fallback_chain_ring",
            parent_atoms=_fb_parent,
            accounted_atoms=_fb_accounted,
            total_heavy_atoms=_fb_total_ha,
        )
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=%.2f accounted=%d/%d name=%s",
            _fb_hr.handler_id, _fb_hr.coverage, len(_fb_hr.accounted_atoms),
            _fb_hr.total_heavy_atoms, _fb_hr.name[:60],
        )

    # ASSEMBLY_AUDIT: detect FGs present in molecule but missing from final name.
    # Guarded by logger level check so there is no performance impact in production.
    if logger.isEnabledFor(logging.DEBUG):
        from ..rules.seniority import PREFIX_FORMS
        detected_fgs = set()
        fg_dict = getattr(features, 'functional_groups', {})
        pg = getattr(features, 'principal_group', None)
        for fg_name_audit, fg_matches in fg_dict.items():
            if fg_name_audit in ('alkene', 'alkyne'):
                continue
            if fg_name_audit == pg:
                continue  # principal group is the suffix, not a prefix
            if fg_matches:
                detected_fgs.add(fg_name_audit)
        missing_fgs = set()
        for fg_audit in detected_fgs:
            prefix = PREFIX_FORMS.get(fg_audit)
            if prefix is None:
                continue  # functional-class-only, no prefix form expected
            if prefix and prefix in assembled:
                continue
            missing_fgs.add(fg_audit)
        if missing_fgs:
            logger.debug(
                "ASSEMBLY_AUDIT: missing_fg=%s in name=%s smiles=%s",
                missing_fgs, assembled, getattr(features, 'canonical_smiles', '?'),
            )

    # Phase 145.1: route chain-naming through pool.
    # In first_applicable mode, pool.best() returns the FIRST added
    # candidate. If a higher-priority handler already added one above,
    # pool.best() is that one (chain naming computed but not returned).
    # If no other handler fired (this is the only candidate), pool.best()
    # is the chain candidate. D-02: chain has priority=fallback in 145.1
    # (preserves byte-identical); Phase 146 raises priority for competition.
    pool = get_current_pool()
    pool.add(assembled, "chain", features)
    return pool.best().name


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

    # Capture C=N E/Z stereo BEFORE modifying the molecule
    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)

    cn_stereo_tag = None
    cn_bond = mol.GetBondBetweenAtoms(c_idx, n_idx)
    if cn_bond is not None:
        stereo = cn_bond.GetStereo()
        if stereo != Chem.BondStereo.STEREONONE:
            # Use _CIPCode property (set by AssignCIPLabels) for E/Z
            if cn_bond.HasProp('_CIPCode'):
                cn_stereo_tag = cn_bond.GetProp('_CIPCode')  # 'E' or 'Z'

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
    from .fragment_naming import name_fragment_recursively
    try:
        parent_name = name_fragment_recursively(modified_smiles)
    except Exception:
        return None

    if not parent_name:
        return None

    # Merge C=N stereo descriptor into the parent name if captured
    if cn_stereo_tag and parent_name:
        import re
        # Find the ketone locant from the parent name (e.g., "3-one" -> locant "3")
        one_match = re.search(r'(\d+)-on', parent_name)
        if one_match:
            cn_locant = one_match.group(1)
            cn_desc = f"{cn_locant}{cn_stereo_tag}"

            # If parent already has a stereo prefix like (6E), merge
            stereo_match = re.match(r'\(([^)]+)\)-(.*)', parent_name)
            if stereo_match:
                existing_stereo = stereo_match.group(1)
                rest = stereo_match.group(2)
                all_descs = existing_stereo.split(',')
                all_descs.append(cn_desc)
                # Sort by leading locant number
                all_descs.sort(
                    key=lambda d: int(re.match(r'(\d+)', d).group(1))
                    if re.match(r'(\d+)', d) else 999
                )
                parent_name = f"({','.join(all_descs)})-{rest}"
            else:
                # No existing stereo prefix -- add one
                parent_name = f"({cn_desc})-{parent_name}"

    return f"{parent_name} {fg_type}"


# ============================================================================
# N-oxide naming (functional class: "pyridine 1-oxide", "trimethylamine N-oxide")
# ============================================================================

# N-oxide naming uses is_top_level_naming() from fragment_naming to prevent
# infinite recursion. N-oxide functional class naming only runs at top level;
# during fragment naming, N-oxides are named as prefixes instead (correct IUPAC).


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
    # Only apply N-oxide functional class naming at top level.
    # During fragment naming, N-oxides are named as prefixes instead.
    from .fragment_naming import is_top_level_naming
    if not is_top_level_naming():
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
    from .fragment_naming import name_fragment_recursively
    try:
        base_name = name_fragment_recursively(modified_smiles)
    except Exception:
        return None

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
    from .fragment_naming import name_fragment_recursively
    try:
        base_name = name_fragment_recursively(modified_smiles)
    except Exception:
        return None

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
                # Count ALL non-ring heavy atoms (not just carbons).
                # Non-carbon substituents (Cl, OH, NH2, F, Br, NO2) on
                # the ring were invisible to the old carbon-only check,
                # causing "phenyl" to be returned for e.g. 4-chlorophenyl.
                # Phase 125 fix: count any atom with atomic number > 1
                # (i.e., exclude only hydrogens).
                non_ring_heavy = sum(
                    1 for i in frag_atoms
                    if i not in ring_set
                    and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                )
                if non_ring_heavy == 0:
                    return "phenyl"
                elif non_ring_heavy == 1:
                    # Only return "benzyl" if the single non-ring heavy
                    # atom is a carbon that is the attachment point
                    # (CH2-phenyl pattern, i.e. start_idx is outside the ring).
                    # If the non-ring atom is a heteroatom (e.g., Cl on ring)
                    # or a carbon substituent on the ring (e.g., methyl in
                    # 4-methylphenyl), fall through to name_substituent().
                    non_ring_atoms = [
                        i for i in frag_atoms
                        if i not in ring_set
                        and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                    ]
                    if (len(non_ring_atoms) == 1
                            and mol.GetAtomWithIdx(non_ring_atoms[0]).GetSymbol() == 'C'
                            and non_ring_atoms[0] == start_idx):
                        return "benzyl"
                # Fragment has ring substituents or complex structure:
                # jump directly to name_substituent() universal fallback,
                # bypassing the alkyl chain code which would miscount
                # aromatic ring carbons as a linear chain (e.g. "hexyl").
                try:
                    from .substituent_enumerator import name_substituent as _ns_r
                    prefix = _ns_r(mol, set(frag_atoms), start_idx)
                    if prefix and prefix != "substituent":
                        return prefix
                except Exception:
                    pass
                return None

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

    # Phase 139 gap closure: detect chain-FG oxygen/sulfur in fragment
    # BEFORE simple alkyl path.  Oxygen and sulfur on chains indicate
    # functional groups (=O -> "oxo", -OH -> "hydroxy", =S -> "thioxo")
    # that get_alkyl_name() would silently ignore.
    # Scope limited to O/S only:
    #   - Nitrogen is excluded because N atoms in fragments are typically
    #     part of functional class patterns (urea, guanidine, amide) that
    #     are handled by dedicated naming paths, not chain FG prefixes.
    #   - Ring heteroatoms are excluded (structural ring members).
    #   - Exocyclic heteroatoms bonded ONLY to ring atoms (C=O on a ring
    #     carbon in fused ureas/lactams) are excluded (ring decorations).
    # A non-ring O/S qualifies as a chain FG indicator only when at least
    # one of its fragment-neighbors is also NOT in a ring (chain context).
    def _is_chain_fg_heteroatom(idx):
        atom = mol.GetAtomWithIdx(idx)
        # Only oxygen (8) and sulfur (16) -- not nitrogen or others
        if atom.GetAtomicNum() not in (8, 16):
            return False
        if atom.IsInRing():
            return False
        # Non-ring O/S: check if any of its fragment-neighbors is also
        # non-ring (chain context) vs all ring (exocyclic decoration)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in frag_set and not nbr.IsInRing():
                return True
        return False

    heteroatom_in_frag = any(_is_chain_fg_heteroatom(i) for i in frag_atoms)
    if heteroatom_in_frag:
        try:
            from .substituent_enumerator import name_substituent as _ns_r
            prefix = _ns_r(mol, set(frag_atoms), start_idx)
            if prefix and prefix != "substituent":
                return prefix
        except Exception:
            pass
        # Fall through to simple alkyl if name_substituent failed

    # Default: linear alkyl name
    try:
        return get_alkyl_name(carbon_count)
    except (ValueError, KeyError):
        pass

    # Phase 86: Universal pipeline fallback for complex R groups that
    # cannot be named by the simple alkyl/phenyl/benzyl classification above.
    # This lets functional class handlers (isocyanate, boronic acid, urea, etc.)
    # correctly name molecules with complex R-groups (functionalized chains,
    # substituted rings, heteroatom-containing fragments).
    try:
        from .substituent_enumerator import name_substituent as _ns_r
        prefix = _ns_r(mol, set(frag_atoms), start_idx)
        if prefix and prefix != "substituent":
            return prefix
    except Exception:
        pass

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
# Carbamic acid naming (IUPAC P-65.2.3: retained name with N-substitution)
# ============================================================================

def _name_carbamic_acid(features: Any) -> Optional[str]:
    """Name carbamic acid as '[N-substituted]carbamic acid' (retained name).

    Pattern: N-C(=O)-OH (free acid, not ester)
    SMARTS match: [NX3][CX3](=O)[OX2H1] gives (N, C, O=, OH)

    Unsubstituted: "carbamic acid" (H2N-COOH)
    N-monosubstituted: "N-methylcarbamic acid" (CH3-NH-COOH)
    N,N-disubstituted: "N,N-dimethylcarbamic acid" ((CH3)2N-COOH)
    Mixed: "N-ethyl-N-methylcarbamic acid"

    Returns:
        Retained name with N-substitution prefix, or None.
    """
    from collections import Counter

    mol = features.mol
    matches = features.functional_groups.get('carbamic_acid', [])
    if not matches:
        return None

    match = matches[0]
    # SMARTS: [NX3][CX3](=O)[OX2H1]
    # match[0] = N (nitrogen)
    # match[1] = C (carbonyl carbon)
    # match[2] = O (carbonyl oxygen, =O)
    # match[3] = O (hydroxyl oxygen, -OH)
    if len(match) < 4:
        return None

    n_idx = match[0]   # Nitrogen
    c_idx = match[1]   # Carbonyl carbon

    # Core atoms to exclude from R group naming
    carbamic_core = set(match)  # N, C, O=, OH

    # Check N-substituents
    n_atom = mol.GetAtomWithIdx(n_idx)
    n_subs = []
    for nbr in n_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == c_idx:
            continue
        if nbr.GetAtomicNum() <= 1:
            continue
        # Carbon-based substituent on nitrogen
        sub_name = _name_r_group(mol, nidx, exclude_atoms=carbamic_core)
        if sub_name:
            n_subs.append(sub_name)

    if not n_subs:
        # Unsubstituted: "carbamic acid"
        return "carbamic acid"

    # Build N-substitution prefix using _build_n_substituted_name
    tagged_subs = [("N", s) for s in n_subs]
    return _build_n_substituted_name(tagged_subs, "carbamic acid")


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
            n_prefix_parts.append(f"N-{_wrap_n_substituent(name)}")
        else:
            mult = get_multiplier_prefix(count, name)
            n_prefix_parts.append(f"N,N-{mult}{_wrap_n_substituent(name)}")

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

    # INST-01: Atom coverage audit for benzene naming path
    if logger.isEnabledFor(logging.DEBUG):
        total_heavy = features.mol.GetNumHeavyAtoms()
        parent_count = len(ring_atoms) if ring_atoms else 6
        named_count = len(substituents) if substituents else 0
        coverage = (parent_count + named_count) / max(total_heavy, 1)
        logger.debug(
            "ATOM_COVERAGE: smiles=%s total_heavy=%d parent=%d named_subs=%d coverage=%.2f",
            features.canonical_smiles, total_heavy, parent_count, named_count, coverage,
        )
        named_atom_count = 0
        if substituents:
            for sub_atoms_list in substituents.values() if isinstance(substituents, dict) else [substituents]:
                if isinstance(sub_atoms_list, (list, tuple)):
                    named_atom_count += len(sub_atoms_list)
                else:
                    named_atom_count += 1
        logger.debug(
            "ATOM_COVERAGE_DETAIL: smiles=%s named_atom_count=%d",
            features.canonical_smiles, named_atom_count,
        )

    # Orient the ring for lowest locants
    oriented_ring = orient_benzene(mol, ring_atoms, substituents)

    # Phase 152 D-06: mirror features.heterocycle_atom_to_locant convention so
    # the Tier-A return injector at composer.py:1574 can consume an authoritative
    # benzene locant map.  Single source of truth via _ring_atom_to_locant_from_oriented
    # (D-08).
    from ..rules.stereochemistry import _ring_atom_to_locant_from_oriented
    features.benzene_atom_to_locant = _ring_atom_to_locant_from_oriented(oriented_ring)

    # Generate systematic name
    detected_fgs = getattr(features, 'functional_groups', None) or {}
    return name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents, detected_fgs)


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

    # Phase 151-02 D-09: mixed-spiro-fused recognized as complex BEFORE
    # the pure-spiro and polycyclic-bridged checks (mirrors the dispatch
    # order in _classify_complex_ring).
    from ..rules.spiro import is_mixed_spiro_fused as _is_mixed_spiro_fused
    if _is_mixed_spiro_fused(mol):
        return True

    # Check spiro BEFORE polycyclic-bridged (P-24.2).
    # Dispiro compounds have 3+ SSSR rings, causing is_polycyclic_system()
    # to return True. Spiro check must precede polycyclic to avoid misrouting.
    if is_spiro_system(mol):
        return True

    # Check polycyclic-bridged (tricyclo+ pure bridged systems)
    if is_polycyclic_system(mol):
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

    # Phase 151-02 D-09: skip the fused-heterocycle catalog block when
    # the input is mixed-spiro/fused. Without this guard the catalog
    # match (matching the fused PART of the molecule, e.g., indoline)
    # returns 'ortho-fused' for the WHOLE molecule and the
    # mixed-spiro-fused branch never fires.
    from ..rules.spiro import is_mixed_spiro_fused as _phase151_is_mixed_spiro_fused
    _phase151_input_is_mixed = _phase151_is_mixed_spiro_fused(mol)

    # Check known fused heterocycles BEFORE polycyclic-bridged check.
    # Prevents tricyclic fused heterocycles (xanthene, phenothiazine,
    # phenoxazine, thianthrene) from being misclassified as VB polycyclics.
    # These have non-aromatic ring atoms (O, S, N in central ring) which
    # causes is_polycyclic_system() to return True, but they are ortho-fused
    # systems with IUPAC retained names.
    # Guard: only divert if the molecule's ring system IS the matched core
    # (ring_atoms <= parent_atoms + 2), not a larger polycyclic containing it.
    # For molecules with multiple DISCONNECTED ring systems where the core
    # is the largest ring system (nucleotide cofactors: adenine + ribose),
    # use only the core's ring system atom count.
    from ..data.fused_heterocycles import match_fused_heterocycle_core, FUSED_HETEROCYCLE_DATA
    core_match = None if _phase151_input_is_mixed else match_fused_heterocycle_core(mol)
    if core_match is not None:
        core_smiles = core_match[2]
        core_data = FUSED_HETEROCYCLE_DATA.get(core_smiles, {})
        parent_atoms = core_data.get('parent_atoms', 0)
        ri = mol.GetRingInfo()
        # Count all ring atoms as the default
        all_ring_atoms = set()
        for r in ri.AtomRings():
            all_ring_atoms.update(r)
        ring_atom_count = len(all_ring_atoms)
        # For molecules with multiple disconnected ring systems, check if
        # the core's ring system is the largest. If so, use only that
        # system's atom count (handles nucleotide cofactors where adenine
        # is the main ring system with smaller ribose/biotin rings).
        from ..perception.rings import get_ring_systems
        ring_systems = get_ring_systems(mol)
        if len(ring_systems) >= 2:
            core_atom_indices = {k for k in core_match[1].keys() if isinstance(k, int)}
            core_rs_size = 0
            max_other_rs_size = 0
            for rs in ring_systems:
                if core_atom_indices & rs:
                    core_rs_size = len(rs)
                else:
                    max_other_rs_size = max(max_other_rs_size, len(rs))
            # Only use core ring system size when the core IS the largest
            # ring system AND no other system is comparably large (>= 60%
            # of core). This prevents molecules with two equal-sized ring
            # systems (e.g., indole + benzamide) from being named as just
            # the core heterocycle.
            if (core_rs_size > max_other_rs_size
                    and max_other_rs_size < core_rs_size * 0.6):
                ring_atom_count = core_rs_size
        if ring_atom_count <= parent_atoms + 2:
            fused_type = classify_fused_system(mol)
            if fused_type in ('ortho-fused', 'ortho-peri-fused'):
                return fused_type

    # Phase 151-02 D-09: mixed-spiro-fused MUST come BEFORE both pure-spiro
    # AND polycyclic-bridged. Mixed inputs have ≥1 spiro atom AND ≥1 fused
    # junction (n_rings > n_spiro + 1) — they would otherwise route to
    # polycyclic-bridged (and be misnamed as pure VB systems) or to the
    # ortho-fused branch (and lose the spiro junction).
    from ..rules.spiro import is_mixed_spiro_fused
    if is_mixed_spiro_fused(mol):
        return 'mixed-spiro-fused'

    # Check spiro BEFORE polycyclic-bridged (P-24.2).
    # Dispiro compounds have 3+ SSSR rings, causing is_polycyclic_system()
    # to return True. The refined is_spiro_system() checks n_rings == n_spiro + 1,
    # so complex polycyclic molecules with incidental spiro atoms are excluded.
    if is_spiro_system(mol):
        return 'spiro'

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

    return 'simple'


def _assemble_complex_ring_name(mol, features):
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
        ComplexRingResult namedtuple with (name, ring_atoms, atom_to_locant,
        substituents_included), or None if naming fails.

    Note:
        Supports complete naming with substituents, unsaturation, and stereo for all types.
    """
    import logging

    ring_type = _classify_complex_ring(mol)

    try:
        if ring_type == 'bridged-fused':
            # FR-8 nomenclature for bridged fused systems
            result = name_bridged_fused_system(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Bridged-fused naming failed for molecule")
            return None

        elif ring_type == 'bicyclo':
            # Complete bicyclo naming with substituents, unsaturation, stereo
            result = _assemble_complete_bicyclo_name(mol, features)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Bicyclo naming failed for molecule")
            return None

        elif ring_type == 'polycyclic-bridged':
            # Von Baeyer naming for tricyclo+ systems (e.g., adamantane)
            result = name_polycyclic_complete(mol, features)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Polycyclic-bridged naming failed for molecule")
            return None

        elif ring_type == 'spiro':
            # Spiro naming (spiro[4.5]decane, etc.)
            result = name_spiro_system(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Spiro naming failed for molecule")
            return None

        elif ring_type == 'mixed-spiro-fused':
            # Phase 151 D-09 + D-13: HERITAGE §4 separable-parts naming for
            # mixed spiro/fused systems (e.g., spiro[indoline-3,1'-cyclohexane]).
            # The classifier tags this at line ~3043; the dispatcher must
            # invoke name_mixed_spiro_fused or the entire HERITAGE §4 path is
            # dead code from name_compound (151-04 BLK-01 closure).
            from ..rules.spiro import name_mixed_spiro_fused
            result = name_mixed_spiro_fused(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Mixed-spiro-fused naming failed for molecule")
            return None

        elif ring_type in ('ortho-fused', 'ortho-peri-fused'):
            # Fused ring naming - try heterocycle first, then carbocyclic
            # Heterocycles have retained names like indole, quinoline
            result = name_fused_heterocycle(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)

            # Try ortho-fused bicyclic (carbocyclic fallback)
            result = name_ortho_fused_bicyclic(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)

            logging.warning(f"Fused ring naming failed for {ring_type} system")
            return None

        else:
            # Not a complex ring - shouldn't reach here
            return None

    except Exception as e:
        # Graceful degradation - log and return None
        logging.warning(f"Complex ring naming error: {e}")
        return None


def _enrich_complex_ring_with_subs(mol, ring_name, ring_atoms, atom_to_locant):
    """Discover and prepend substituent prefixes to a complex ring parent name.

    Called when a complex ring sub-handler sets substituents_included=False,
    meaning the handler returned a bare parent name without substituent prefixes.
    Uses the universal pipeline (discover_substituents + name_substituent) to
    find all non-ring atoms bonded to ring atoms, name them, and prepend as
    alphabetized IUPAC prefixes.

    Args:
        mol: RDKit Mol object
        ring_name: The bare parent ring name (e.g., "spiro[4.5]decane")
        ring_atoms: Set of all atom indices in the ring system
        atom_to_locant: Dict mapping mol atom indices to IUPAC locant values

    Returns:
        Name with substituent prefixes prepended, or original ring_name if
        no substituents found or on error.
    """
    from .substituent_enumerator import discover_substituents, name_substituent
    from .naming_utils import get_multiplier_prefix, alpha_sort_key
    from collections import defaultdict

    if not atom_to_locant:
        return ring_name

    # Build oriented_ring from atom_to_locant: atoms sorted by locant value
    # This gives extract_ring_substituents the IUPAC numbering order
    try:
        oriented_ring = tuple(
            a for a, _ in sorted(atom_to_locant.items(), key=lambda x: (
                # Handle mixed int/str locants: ints sort before strings
                (0, x[1]) if isinstance(x[1], int) else (1, x[1])
            ))
        )
    except (TypeError, ValueError):
        return ring_name

    # Discover substituents using universal pipeline
    try:
        subs = discover_substituents(
            mol, ring_atoms, "ring", oriented_ring=oriented_ring
        )
    except Exception as e:
        logger.debug("_enrich_complex_ring_with_subs: discover failed: %s", e)
        return ring_name

    if not subs:
        return ring_name

    # Group substituents by prefix name, collecting locants
    prefix_groups = defaultdict(list)  # name -> [locant1, locant2, ...]
    for sub_info in subs:
        frag_ha = len(sub_info.frag_atoms)

        a_idx = _find_attach_idx_in_frag(mol, sub_info, ring_atoms)
        prefix_name = name_substituent(mol, sub_info.frag_atoms, a_idx)

        if not prefix_name or prefix_name == "substituent":
            continue
        # Quality filter: reject garbled names with spaces
        if ' ' in prefix_name:
            log_level = logging.WARNING if frag_ha > 20 else logging.DEBUG
            logger.log(
                log_level,
                "_enrich_complex_ring_with_subs: reject garbled (HA=%d): %s",
                frag_ha, prefix_name,
            )
            continue

        # Use locant from sub_info (assigned by discover_substituents using
        # oriented_ring ordering). Convert to string for formatting.
        locant = sub_info.locant
        if locant is not None:
            prefix_groups[prefix_name].append(locant)
        else:
            # Fallback: use atom_to_locant directly
            attach_idx = sub_info.attach_mol_idx
            if attach_idx in atom_to_locant:
                prefix_groups[prefix_name].append(atom_to_locant[attach_idx])

    if not prefix_groups:
        return ring_name

    # Build prefix parts with locants and multiplier prefixes
    prefix_parts = []
    for name, locants in prefix_groups.items():
        # Sort locants numerically (int) then lexically (str)
        locants.sort(key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
        count = len(locants)
        if count == 1:
            prefix_parts.append(f"{locants[0]}-{name}")
        else:
            locant_str = ",".join(str(loc) for loc in locants)
            multiplier = get_multiplier_prefix(count, name)
            prefix_parts.append(f"{locant_str}-{multiplier}{name}")

    # Alphabetize prefixes per IUPAC P-14.4
    prefix_parts.sort(key=lambda x: alpha_sort_key(x))

    # Join prefix parts and prepend to ring name
    prefix_str = "-".join(prefix_parts)
    return f"{prefix_str}{ring_name}" if prefix_str else ring_name


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
    ester_prefix_groups: Dict[str, List[int]] = _dd(list)

    for ester_info in exocyclic_esters:
        prefix = ester_info['acyloxy_prefix']
        ester_prefix_groups[prefix].append(ester_info['ring_attach_atom_idx'])

    # Determine total number of ester substituents on the ring
    total_esters = sum(len(v) for v in ester_prefix_groups.values())

    # IUPAC P-31.1.2: all substituents on the parent ring receive coordinated
    # locants from a single unified numbering system.  Always find the ring
    # atoms so that ester prefixes and non-ester substituents share one system.
    ring_atoms_tuple = None
    ring_set_for_parent: set = set()
    for r in ring_info.AtomRings():
        if attach_atom in r:
            ring_atoms_tuple = r
            ring_set_for_parent = set(r)
            break

    # Collect all ester-related atoms (acid fragment + ester O) to exclude
    # from universal prefix discovery so they are not re-discovered as
    # generic substituents.
    from ..rules.esters import parse_ester_fragments
    ester_exclude_atoms: set = set()
    for ester_info in exocyclic_esters:
        match = ester_info['ester_match']
        acid_atoms, _alkyl = parse_ester_fragments(mol, match)
        ester_exclude_atoms.update(acid_atoms)
        # Also exclude the ester oxygen itself (match[2])
        if len(match) > 2:
            ester_exclude_atoms.add(match[2])

    # Discover non-ester substituents via the universal pipeline.
    non_ester_prefix_groups: Dict[str, List[int]] = _dd(list)
    if ring_atoms_tuple is not None:
        try:
            from .substituent_enumerator import discover_substituents, name_substituent
            effective_parent = ring_set_for_parent | ester_exclude_atoms

            # Compute oriented_ring from ring numbering so that
            # extract_ring_substituents can assign locants properly.
            oriented = None
            numbering_candidates = _number_ring_from_attachment(
                mol, ring_atoms_tuple, attach_atom
            )
            if numbering_candidates:
                numbering = numbering_candidates[0]  # {locant: atom_idx}
                oriented = tuple(
                    numbering[loc] for loc in sorted(numbering.keys())
                )

            subs = discover_substituents(
                mol, effective_parent,
                parent_type="ring",
                oriented_ring=oriented,
            )
            if subs:
                for sub_info in subs:
                    # Size guard: only accept small substituent fragments
                    # that the naming pipeline handles reliably. Complex
                    # multi-atom fragments (nucleobases, long chains with
                    # functional groups) tend to produce garbled names.
                    frag_ha = len(sub_info.frag_atoms)
                    if frag_ha > 6:
                        logger.debug(
                            "Skipping large non-ester sub (HA=%d): %s",
                            frag_ha, sub_info.frag_atoms,
                        )
                        continue
                    a_idx = _find_attach_idx_in_frag(
                        mol, sub_info, effective_parent
                    )
                    prefix_name = name_substituent(
                        mol, sub_info.frag_atoms, a_idx
                    )
                    if not prefix_name or prefix_name == "substituent":
                        continue
                    # Quality filter: reject garbled names (spaces, digits
                    # directly following letters without hyphens).
                    if ' ' in prefix_name:
                        logger.debug(
                            "Rejecting garbled non-ester sub name: %s",
                            prefix_name,
                        )
                        continue
                    non_ester_prefix_groups[prefix_name].append(
                        sub_info.attach_mol_idx
                    )
        except Exception as exc:
            logger.debug(
                "Ring ester non-ester sub discovery failed: %s", exc
            )

    # Merge ester and non-ester prefix groups
    all_prefix_groups: Dict[str, List[int]] = _dd(list)
    for name, atoms in ester_prefix_groups.items():
        all_prefix_groups[name].extend(atoms)
    for name, atoms in non_ester_prefix_groups.items():
        all_prefix_groups[name].extend(atoms)

    total_substituents = sum(len(v) for v in all_prefix_groups.values())

    # Compute ring numbering when locants are needed (2+ total substituents)
    ring_atom_to_locant: Dict[int, int] = {}
    if total_substituents > 1 and ring_atoms_tuple is not None:
        # Collect all ring attachment atom indices (ester + non-ester)
        all_attach = []
        for atoms_list in all_prefix_groups.values():
            for a in atoms_list:
                if a in ring_set_for_parent:
                    all_attach.append(a)
        # Use _number_ring_from_attachment for IUPAC-optimal numbering.
        # Try all attachment atoms as position 1, pick lowest locant set.
        best_mapping = None
        best_locants = None
        for start_idx in all_attach:
            candidates = _number_ring_from_attachment(
                mol, ring_atoms_tuple, start_idx
            )
            if not candidates:
                continue
            for numbering in candidates:
                # numbering is {locant: atom_idx}, invert to {atom_idx: locant}
                inv = {v: k for k, v in numbering.items()}
                locant_set = sorted(inv.get(a, 999) for a in all_attach)
                if best_locants is None or locant_set < best_locants:
                    best_locants = locant_set
                    best_mapping = inv
        if best_mapping:
            ring_atom_to_locant = best_mapping

    # Build prefix parts with locants
    prefix_parts = []
    for prefix_name, attach_atoms_list in sorted(all_prefix_groups.items()):
        count = len(attach_atoms_list)
        is_acyloxy = prefix_name in ester_prefix_groups
        if total_substituents == 1:
            # Single substituent on ring: no locant needed
            prefix_parts.append(prefix_name)
        elif count == 1:
            # One instance of this prefix: need locant
            locant = ring_atom_to_locant.get(attach_atoms_list[0], 1)
            if is_acyloxy:
                prefix_parts.append(f"{locant}-({prefix_name})")
            else:
                prefix_parts.append(f"{locant}-{prefix_name}")
        else:
            # Multiple instances of same prefix: locants + multiplier
            locants = sorted(
                ring_atom_to_locant.get(a, 1) for a in attach_atoms_list
            )
            locant_str = ",".join(str(loc) for loc in locants)
            multiplier = get_multiplier_prefix(count, prefix_name)
            if is_acyloxy:
                prefix_parts.append(
                    f"{locant_str}-{multiplier}({prefix_name})"
                )
            else:
                prefix_parts.append(
                    f"{locant_str}-{multiplier}{prefix_name}"
                )

    # Sort alphabetically per IUPAC P-14.4
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


def _assemble_complete_bicyclo_name(mol, features):
    """
    Assemble complete IUPAC name for a bicyclo system with substituents, unsaturation, and stereo.

    IUPAC name order for bicyclo compounds:
    (stereo)-locants-prefixes-bicyclo[x.y.z]parent-locants-unsaturation

    Example: (1S,6R)-3,7,7-trimethylbicyclo[4.1.0]hept-3-ene

    Args:
        mol: RDKit Mol object
        features: MolecularFeatures object

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included)
        where substituents_included is True (bicyclo handler discovers
        substituents via get_complete_bicyclo_data), or None if naming fails.
    """
    from ..rules.bicyclo import get_complete_bicyclo_data, name_bicyclo_system
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from rdkit.Chem import rdCIPLabeler

    # Collect all ring atoms from SSSR for structured return
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)

    # Get complete bicyclo data
    bicyclo_data = get_complete_bicyclo_data(mol)
    if not bicyclo_data:
        # Fall back to simple naming
        fallback_name = name_bicyclo_system(mol)
        if fallback_name:
            return (fallback_name, ring_atoms, {}, True)
        return None

    # If there's a retained name and no substituents or unsaturation, use it
    retained_name = bicyclo_data.get('retained_name')
    substituents = bicyclo_data.get('substituents', {})
    unsaturation = bicyclo_data.get('unsaturation', {})
    double_bonds = unsaturation.get('double_bonds', [])
    triple_bonds = unsaturation.get('triple_bonds', [])

    # Check if there are any modifiers (substituents or unsaturation)
    has_substituents = bool(substituents)
    has_unsaturation = bool(double_bonds) or bool(triple_bonds)

    atom_to_locant = bicyclo_data.get('atom_to_locant', {})

    # For unsubstituted, saturated compounds with retained names, use the retained name
    if retained_name and not has_substituents and not has_unsaturation:
        return (retained_name, ring_atoms, atom_to_locant, True)

    # Get key data
    descriptor = bicyclo_data.get('descriptor')
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

    # Get heteroatom replacement prefix (oxa, aza, thia) if present
    heteroatom_prefix = bicyclo_data.get('heteroatom_prefix', '')

    # Assemble the name
    # Format: (stereo)-substituent-prefix-heteroatom-prefix-descriptor-parent-unsaturation
    parts = []

    if stereo_prefix:
        parts.append(stereo_prefix)

    if sub_prefix:
        # Ensure substituent prefix ends with hyphen before heteroatom or descriptor
        if not sub_prefix.endswith('-'):
            sub_prefix += '-'
        parts.append(sub_prefix)

    if heteroatom_prefix:
        parts.append(heteroatom_prefix)

    # Build the main name: bicyclo[x.y.z]parent-unsat
    main_name = f"{descriptor}{unsat_suffix}"
    parts.append(main_name)

    # Join parts
    name = "".join(parts)

    return (name, ring_atoms, atom_to_locant, True)


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

    # Convert atom indices to locants.
    # IUPAC P-31.1.4.1: use compound locant "N(M)" when the two atoms of
    # a double bond do not have locants differing by one.
    def _to_locant_str(idx1, idx2):
        loc1 = atom_to_locant.get(idx1, 0)
        loc2 = atom_to_locant.get(idx2, 0)
        lower, higher = min(loc1, loc2), max(loc1, loc2)
        if higher - lower == 1:
            return str(lower)
        return f"{lower}({higher})"

    double_locants = [_to_locant_str(b[0], b[1]) for b in double_bonds]
    triple_locants = [_to_locant_str(b[0], b[1]) for b in triple_bonds]

    # Sort by primary (lower) locant number
    def _locant_sort_key(s):
        return int(s.split('(')[0])

    double_locants.sort(key=_locant_sort_key)
    triple_locants.sort(key=_locant_sort_key)

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

    # INST-01: Atom coverage audit for heterocycle naming path
    if logger.isEnabledFor(logging.DEBUG):
        total_heavy = features.mol.GetNumHeavyAtoms()
        parent_count = len(features.principal_ring) if features.principal_ring else 0
        named_count = len(substituents) if substituents else 0
        coverage = (parent_count + named_count) / max(total_heavy, 1)
        logger.debug(
            "ATOM_COVERAGE: smiles=%s total_heavy=%d parent=%d named_subs=%d coverage=%.2f",
            features.canonical_smiles, total_heavy, parent_count, named_count, coverage,
        )
        named_atom_count = 0
        if substituents:
            for sub_entry in substituents.values() if isinstance(substituents, dict) else substituents:
                if isinstance(sub_entry, (list, tuple)):
                    named_atom_count += len(sub_entry)
                else:
                    named_atom_count += 1
        logger.debug(
            "ATOM_COVERAGE_DETAIL: smiles=%s named_atom_count=%d",
            features.canonical_smiles, named_atom_count,
        )

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

    For chain amides, uses the general assembly fragment pipeline
    (_generate_chain_parent, _generate_suffix, _generate_stereodescriptors)
    to correctly produce unsaturated names with E/Z stereodescriptors.

    For ring-attached amides, delegates to rules.amides.name_amide()
    which handles the -carboxamide suffix convention.

    Handles:
    - Primary amides: acetamide, propanamide, (9Z)-octadec-9-enamide
    - Secondary amides: N-methylacetamide, N-methyl(9Z)-octadec-9-enamide
    - Tertiary amides: N,N-dimethylformamide
    - Ring-attached amides: cyclohexanecarboxamide

    Args:
        features: MolecularFeatures with amide principal group
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the amide
    """
    from ..rules.amides import (
        name_amide, is_ring_attached_amide, get_amide_type,
        get_n_substituents, format_n_substitution,
    )

    mol = features.mol
    amide_atoms = None
    if features.principal_group_atoms:
        amide_atoms = features.principal_group_atoms[0]

    if not amide_atoms:
        return "amide"

    # Ring-attached amides use -carboxamide suffix via name_amide()
    # (ring systems rarely have E/Z issues; existing path is correct)
    if is_ring_attached_amide(mol, amide_atoms):
        base_name = name_amide(mol, amide_atoms)
        if base_name:
            # Add non-principal group prefixes (halogens, hydroxy, etc.)
            # Note: NameFragment.text already includes locants -- use directly.
            prefixes = _generate_prefixes(features)
            if prefixes:
                prefix_parts = []
                for p in sorted(prefixes, key=lambda x: alpha_sort_key(x.text)):
                    prefix_parts.append(p.text)
                if prefix_parts:
                    prefix_str = "-".join(prefix_parts)
                    # Insert hyphen before N-locant prefix (base_name may
                    # start with "N-" or "N,N-" from name_amide())
                    sep = "-" if base_name[:1] == "N" else ""
                    return _inject_stereo_if_missing(features, f"{prefix_str}{sep}{base_name}")
            return _inject_stereo_if_missing(features, base_name)
        return "amide"

    # Chain amides: use general assembly fragments when the chain has unsaturation
    # or E/Z stereo that name_amide() cannot handle. For saturated chains, use
    # name_amide() which handles retained names (formamide, acetamide) and correctly
    # identifies the acyl chain length independent of N-substituent chains.
    has_chain_unsaturation = bool(
        getattr(features, 'double_bonds', None) or
        getattr(features, 'triple_bonds', None)
    )
    if not has_chain_unsaturation:
        base_name = name_amide(mol, amide_atoms)
        if base_name:
            # Add non-principal group prefixes (halogens, hydroxy, etc.)
            # Note: _generate_prefixes returns NameFragments whose .text
            # already includes locants (e.g., "2-methyl"), so use .text
            # directly -- do NOT re-prepend locants from .locants field.
            prefixes = _generate_prefixes(features)
            if prefixes:
                prefix_parts = []
                for p in sorted(prefixes, key=lambda x: alpha_sort_key(x.text)):
                    prefix_parts.append(p.text)
                if prefix_parts:
                    prefix_str = "-".join(prefix_parts)
                    # Insert hyphen before N-locant prefix (base_name may
                    # start with "N-" or "N,N-" from name_amide())
                    sep = "-" if base_name[:1] == "N" else ""
                    return _inject_stereo_if_missing(features, f"{prefix_str}{sep}{base_name}")
            return _inject_stereo_if_missing(features, base_name)

    # Unsaturated chain amides: use general assembly fragments for correct stereo + unsaturation
    fragments = []

    # Generate parent name (chain stem with unsaturation markers)
    if features.principal_chain:
        parent = _generate_chain_parent(features)
    else:
        # Unexpected for chain amide, but fallback gracefully
        base_name = name_amide(mol, amide_atoms)
        return base_name if base_name else "amide"

    fragments.append(parent)

    # Generate suffix for amide group
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

    # Generate non-principal group prefixes (halogens, hydroxy, etc.)
    prefixes = _generate_prefixes(features)
    fragments.extend(prefixes)

    # Generate stereodescriptors (R/S stereocenters and E/Z double bonds)
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)

    # Assemble base name with correct IUPAC ordering
    base_name = _assemble_fragments(fragments, style)

    # Handle N-substitution for secondary/tertiary amides
    amide_type = get_amide_type(mol, amide_atoms)
    if amide_type in ("secondary", "tertiary"):
        n_subs = get_n_substituents(mol, amide_atoms)
        n_prefix = format_n_substitution(n_subs)
        if n_prefix:
            return f"{n_prefix}{base_name}"

    return base_name


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
            ring_info = mol.GetRingInfo()
            frag_set_fused = set(frag)

            # --- Phase 79-02: Fused het detection for amine N-subs ---
            # Try fused het identification FIRST (before phenyl check).
            # Fused hets like quinoline contain a benzene sub-ring that would
            # falsely match the all-C aromatic phenyl check.
            has_fused_het_sub = False
            if cc > 0:
                from ..data.fused_heterocycles import (
                    match_fused_heterocycle_core as _match_fh_amine,
                    get_fused_heterocycle_prefix as _get_fh_prefix_amine,
                )
                fused_r_amine = _match_fh_amine(mol)
                if fused_r_amine is not None:
                    fh_name, fh_mapping, fh_core_smiles = fused_r_amine
                    fh_core_atoms = set(fh_mapping.keys())
                    if fh_core_atoms & frag_set_fused:
                        # Find the ring atom directly bonded to N
                        fh_attach = None
                        for ra in fh_core_atoms & frag_set_fused:
                            atom_ra = mol.GetAtomWithIdx(ra)
                            for nbr in atom_ra.GetNeighbors():
                                if nbr.GetIdx() == n_idx:
                                    fh_attach = ra
                                    break
                            if fh_attach is not None:
                                break
                        if fh_attach is not None:
                            fh_prefix = _get_fh_prefix_amine(fh_core_smiles, fh_attach, fh_mapping)
                            if fh_prefix is not None:
                                non_core_c_amine = sum(
                                    1 for idx in frag_set_fused
                                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                                    and idx not in fh_core_atoms
                                )
                                if non_core_c_amine == 0:
                                    n_subs.append(fh_prefix)
                                    has_fused_het_sub = True
            # --- End Phase 79-02 fused het detection for amine N-subs ---

            # Check for aromatic rings (phenyl)
            has_phenyl = False
            if not has_fused_het_sub:
                for ring in ring_info.AtomRings():
                    if all(r in frag_set_fused for r in ring) and len(ring) == 6:
                        if all(mol.GetAtomWithIdx(r).GetIsAromatic() and
                               mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
                            # Verify this is an isolated benzene (not part of fused system)
                            ring_set_chk = set(ring)
                            is_fused_ring = False
                            for other_ring in ring_info.AtomRings():
                                if set(other_ring) != ring_set_chk and set(other_ring) & ring_set_chk:
                                    is_fused_ring = True
                                    break
                            if not is_fused_ring:
                                non_ring = cc - 6
                                if non_ring == 0:
                                    n_subs.append("phenyl")
                                    has_phenyl = True
                                    break

            # --- Phase 79-01: Direct ring identification for amine N-subs (DROP-25 fix) ---
            # Before branched/linear alkyl naming, check if the fragment IS a ring.
            # This handles non-phenyl ring substituents on amines (piperidinyl, cyclohexyl, etc.)
            has_ring_sub = False
            if not has_phenyl and not has_fused_het_sub and cc > 0:
                for ring in ring_info.AtomRings():
                    ring_set_inner = set(ring)
                    if ring_set_inner.issubset(frag_set_fused):
                        # Check if ring accounts for all C atoms (pure ring sub)
                        non_ring_c = sum(
                            1 for idx in frag
                            if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                            and idx not in ring_set_inner
                        )
                        if non_ring_c == 0:
                            from ..rules.ring_substituents import get_ring_substituent_name as _get_rsn
                            ring_prefix = _get_rsn(mol, tuple(ring))
                            if ring_prefix:
                                n_subs.append(ring_prefix)
                                has_ring_sub = True
                                break
            # --- End Phase 79-01 direct ring identification ---

            if not has_phenyl and not has_ring_sub and not has_fused_het_sub and cc > 0:
                # Check if substituent is branched (any carbon with 3+ heavy
                # atom neighbors = branching point). Branched groups need
                # recursive naming for correct IUPAC 2013 names
                # (e.g., propan-2-yl not propyl for isopropyl).
                is_branched = False
                frag_set = set(frag) | {n_idx}  # include parent N
                for fi in frag:
                    fatom = mol.GetAtomWithIdx(fi)
                    if fatom.GetSymbol() == 'C':
                        # Count ALL heavy neighbors (including back to N)
                        heavy_nbrs = sum(
                            1 for nn in fatom.GetNeighbors()
                            if nn.GetSymbol() != 'H'
                        )
                        if heavy_nbrs >= 3:
                            is_branched = True
                            break

                if is_branched:
                    # Branched: use recursive naming for correct IUPAC name
                    sub_name = name_substituent_fragment(
                        mol, frag, frag[0], list(chain_set) + [n_idx]
                    )
                    if sub_name:
                        n_subs.append(sub_name)
                    else:
                        # Fallback to simple count
                        try:
                            n_subs.append(get_alkyl_name(cc))
                        except (ValueError, KeyError):
                            logger.debug(
                                "DROP-25 substituent_skip: reason=amine_nsub_still_unnameable carbon_count=%d",
                                cc,
                            )
                else:
                    try:
                        n_subs.append(get_alkyl_name(cc))
                    except (ValueError, KeyError):
                        # Try recursive naming for complex N-substituents
                        sub_name = name_substituent_fragment(
                            mol, frag, frag[0], list(chain_set)
                        )
                        if sub_name:
                            n_subs.append(sub_name)
                        else:
                            logger.debug(
                                "DROP-25 substituent_skip: reason=amine_nsub_still_unnameable carbon_count=%d",
                                cc,
                            )

    if not n_subs:
        return None  # No N-substituents found, use general path

    # Build N-prefix (same logic as amide N-substitution)
    from collections import Counter
    sub_counts = Counter(n_subs)
    n_prefix_parts = []
    for name in sorted(sub_counts.keys()):
        count = sub_counts[name]
        if count == 1:
            n_prefix_parts.append(f"N-{_wrap_n_substituent(name)}")
        else:
            mult = SIMPLE_MULTIPLIERS.get(count, str(count))
            n_locants = ",".join(["N"] * count)
            n_prefix_parts.append(f"{n_locants}-{mult}{_wrap_n_substituent(name)}")

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
    - 1-methylcyclohexanecarbonitrile

    Phase 86-03: Retrofitted to discover ring substituents via universal
    pipeline (_integrate_universal_prefixes). Previously dropped all
    substituents on the ring.

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

    # Get nitrile atoms to exclude from substituent discovery
    nitrile_atoms = None
    exclude_atoms = set()
    if features.principal_group_atoms:
        nitrile_atoms = features.principal_group_atoms[0]
        if nitrile_atoms:
            exclude_atoms = set(nitrile_atoms)

    # Build the base nitrile name (ring + carbonitrile)
    if nitrile_atoms:
        base_name = name_nitrile(features.mol, nitrile_atoms, parent_name=parent_name, is_ring=True)
    else:
        base_name = f"{parent_name}carbonitrile"

    # Phase 86-03: Discover ring substituents via universal pipeline
    # The nitrile C#N atoms are excluded so they are not named as substituents
    parent_atoms = set(principal_ring)
    oriented_ring = list(principal_ring)

    prefix_str = _integrate_universal_prefixes(
        features.mol, parent_atoms,
        parent_type="ring",
        oriented_ring=oriented_ring,
        exclude_atoms=exclude_atoms,
    )

    if prefix_str:
        return f"{prefix_str}{base_name}"

    return base_name


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
        # Fallback: no ring identified -- return empty parent to avoid
        # generating garbled 'cycloane' (cyclo + ane with no stem)
        return NameFragment(text="", fragment_type="parent")

    ring_size = len(principal_ring)

    # Guard: ring_size must produce a valid stem; otherwise return empty parent
    try:
        stem = get_chain_prefix(ring_size)
    except (ValueError, KeyError):
        # Invalid ring size (0, negative, etc.) -- return empty parent
        return NameFragment(text="", fragment_type="parent")

    # Guard: stem must be non-empty to avoid generating 'cycloane'
    if not stem:
        return NameFragment(text="", fragment_type="parent")

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
        # Aromatic ring type not handled by cyclic naming -- return empty
        # parent to signal that this ring needs a specialized handler
        # (benzene retained name, fused ring dictionary, etc.)
        return NameFragment(text="", fragment_type="parent")

    elif ring_type and ring_type.startswith('heterocyclic'):
        # Heterocyclic ring type not handled by cyclic naming -- return
        # empty parent to signal need for specialized handler
        return NameFragment(text="", fragment_type="parent")

    # Unknown ring type -- return empty parent rather than garbled 'cyclo'
    return NameFragment(text="", fragment_type="parent")


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

    # Correct is_ring: use chain suffix when chain is parent or principal chain exists,
    # ring suffix only when ring is parent and PG is on a ring carbon.
    if getattr(features, 'chain_is_parent', False):
        is_ring = False
    elif features.principal_chain:
        is_ring = False
    else:
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
        if should_omit_locant_one(context="suffix", fg_type=fg_name):
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
    # Skip when chain_is_parent: ring is a substituent of the chain, not the parent.
    # Ring substituent data was populated for ring-as-substituent naming but should
    # not be used for prefix generation on the chain parent (Phase 139 ARCH-01).
    ring_substituents = getattr(features, 'ring_substituents', None)
    oriented_ring = getattr(features, 'oriented_ring', None)
    # DROP-07 fix: track FG atoms handled by ring alkyl prefixes to prevent
    # double-emission in the global FG loop below
    handled_ring_fg_atoms = frozenset()
    if ring_substituents and features.mol and oriented_ring and not getattr(features, 'chain_is_parent', False):
        ring_prefixes, handled_ring_fg_atoms = _generate_ring_alkyl_prefixes(features)
        prefixes.extend(ring_prefixes)

    # --- Handle non-principal functional groups as prefixes ---
    # When chain_is_parent, skip FGs on ring atoms (already in ring substituent name).
    # Also include non-chain atoms reachable from ring atoms: inner substituents on
    # fused heterocycles (CF3, NO2, CN, etc.) are named as part of the ring compound
    # prefix and must NOT leak as FG prefixes on the parent chain.
    ring_atom_set = set()
    if getattr(features, 'chain_is_parent', False):
        chain_set_for_expand = set(features.principal_chain)
        for rg in getattr(features, 'ring_substituents_as_groups', []):
            ring_atom_set.update(rg)
        # BFS expand: include all atoms reachable from ring atoms that are not
        # on the principal chain (captures inner substituents like CF3, NO2, CN)
        expand_queue = list(ring_atom_set)
        while expand_queue:
            aidx = expand_queue.pop()
            for nbr in features.mol.GetAtomWithIdx(aidx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_atom_set and nidx not in chain_set_for_expand:
                    ring_atom_set.add(nidx)
                    expand_queue.append(nidx)

    # BUG-B: Collect all substituent branch atom indices.
    # FG matches located entirely on a branch are handled by substituent naming,
    # so we skip them here to avoid double-counting (e.g., standalone "hydroxy"
    # when the branch is already named "(hydroxymethyl)").
    branch_atoms = set()
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                branch_atoms.update(sub_atoms)
    # Also collect ring substituent branch atoms -- the enumerator now names
    # compound substituents (trifluoromethyl, etc.) on rings, so their FG atoms
    # should not be double-counted as standalone FG prefixes.
    ring_substituents = getattr(features, 'ring_substituents', None)
    if ring_substituents:
        for _ring_idx, sub_list in ring_substituents.items():
            for sub_atoms in sub_list:
                branch_atoms.update(sub_atoms)

    for fg_name, matches in features.functional_groups.items():
        if fg_name == features.principal_group:
            continue

        # Unsaturation indicators are NOT functional groups (IUPAC P-31.1).
        # They are handled as -ene/-yne infixes by _build_unsaturation_infix(),
        # not as prefixes.  Skip to avoid false DROP-16 noise.
        if fg_name in ('alkene', 'alkyne'):
            continue

        # Filter out FGs on ring atoms when chain is parent
        if ring_atom_set:
            filtered = []
            for match in matches:
                # Skip FG if ANY atom in the match is part of the ring
                # substituent fragment (includes inner substituents like CF3)
                on_ring = any(a in ring_atom_set for a in match)
                if not on_ring:
                    filtered.append(match)
            matches = filtered

        # DROP-07 fix: skip FG matches already handled as ring substituents
        # by _generate_ring_alkyl_prefixes() to prevent double-emission.
        # FG SMARTS matches include anchor atoms (e.g., C-F match = (C_idx, F_idx)),
        # so check if ANY atom in the match is in the handled set.
        if handled_ring_fg_atoms:
            matches = [m for m in matches if not any(a in handled_ring_fg_atoms for a in m)]

        # BUG-B: Skip simple FG matches on small substituent branches (<=3 carbons)
        # that get named as compound substituents (hydroxymethyl, aminomethyl, etc.)
        # Uses shared BRANCH_HANDLED_FGS from naming_utils (unified in Phase 113).
        if branch_atoms and fg_name in BRANCH_HANDLED_FGS:
            filtered_branch = []
            for match in matches:
                if not all(a in branch_atoms for a in match):
                    filtered_branch.append(match)
                    continue
                # Check if FG is on a small branch with carbons
                # Search both chain substituents and ring substituents
                on_small_branch = False
                # Chain substituents
                if features.substituents:
                    for _pos, sub_list in features.substituents.items():
                        for sub_atoms in sub_list:
                            sub_set = set(sub_atoms)
                            if all(a in sub_set for a in match):
                                c_count = sum(1 for a in sub_atoms
                                              if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                                if 1 <= c_count <= 3:
                                    on_small_branch = True
                                    break
                        if on_small_branch:
                            break
                # Ring substituents (for enumerator-handled compound substituents)
                if not on_small_branch and ring_substituents:
                    for _ring_idx, sub_list in ring_substituents.items():
                        for sub_atoms in sub_list:
                            sub_set = set(sub_atoms)
                            if all(a in sub_set for a in match):
                                c_count = sum(1 for a in sub_atoms
                                              if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                                if 1 <= c_count <= 3:
                                    on_small_branch = True
                                    break
                        if on_small_branch:
                            break
                if not on_small_branch:
                    filtered_branch.append(match)
            original_count = len(matches)
            matches = filtered_branch
            if not matches and original_count > 0:
                # DROP-17: BUG-B removed all matches. Per-branch containment
                # check ensures each match is genuinely on a small branch whose
                # compound name (e.g., "chloromethyl") already includes the FG.
                # Do NOT blindly restore -- that would cause double-emission.
                # Only log for diagnostic purposes.
                logger.debug(
                    "DROP-17 all_filtered: fg_name=%s original_count=%d (branch naming handles these)",
                    fg_name, original_count,
                )

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

            # Omit locants when they are trivially unambiguous (centralized)
            chain = getattr(features, 'principal_chain', [])
            chain_len = len(chain)
            omit_locants = should_omit_locant_one(
                context="prefix",
                chain_length=chain_len,
                is_monosubstituted=(count == 1 and fg_locants == [1]
                                    and features.principal_group is None),
            )

            prefixes.append(NameFragment(
                text=prefix_text,
                locants=tuple(sorted(fg_locants)) if (fg_locants and not omit_locants) else (),
                fragment_type="prefix"
            ))
        elif not prefix_text and matches:
            # By-design: FGs using functional class naming (ether, sulfoxide, etc.)
            # don't have prefix forms — handled by specialized naming paths
            logger.debug(
                "DROP-16 substituent_skip: reason=no_fg_prefix_form fg_name=%s match_count=%d",
                fg_name, len(matches),
            )

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


def _detect_fused_het_inner_subs(
    mol, core_atom_set, atom_mapping, ring_atom_set, chain_set,
    attach_ring_idx
) -> str:
    """Detect substituents on a fused heterocycle core and return formatted prefix string.

    Finds atoms bonded to core atoms that are NOT part of the core and NOT in
    the chain. Maps their positions to IUPAC locants via atom_mapping.

    Returns:
        Formatted inner substituent prefix string (e.g., "5-methyl-") or ""
        if no inner substituents found.
    """
    _HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    _ALKOXY_DIRECT = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy'}

    from ..assembly.naming_utils import get_alkyl_name, get_multiplier_prefix as _gmp

    sub_groups: Dict[str, List] = defaultdict(list)

    for core_idx in core_atom_set:
        if core_idx == attach_ring_idx:
            continue  # Skip attachment point
        iupac_locant = atom_mapping.get(core_idx)
        if iupac_locant is None:
            continue
        atom = mol.GetAtomWithIdx(core_idx)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in core_atom_set or ni in chain_set or ni in ring_atom_set:
                continue
            sym = nbr.GetSymbol()
            if sym in _HALOGEN_PREFIX:
                sub_groups[_HALOGEN_PREFIX[sym]].append(iupac_locant)
            elif sym == 'C' and not nbr.GetIsAromatic():
                # Non-core carbon neighbor -- classify by substructure

                cn_nbrs = [n for n in nbr.GetNeighbors()
                           if n.GetIdx() != core_idx
                           and n.GetIdx() not in core_atom_set]

                # (a) Cyano detection: C with triple bond to N, no H on C
                if (len(cn_nbrs) == 1 and cn_nbrs[0].GetSymbol() == 'N'
                        and nbr.GetTotalNumHs() == 0):
                    bond_cn = mol.GetBondBetweenAtoms(ni, cn_nbrs[0].GetIdx())
                    if bond_cn and bond_cn.GetBondTypeAsDouble() == 3.0:
                        sub_groups['cyano'].append(iupac_locant)
                        continue

                # (b) Haloalkyl detection: C bonded only to halogens (no H, no non-hal)
                hal_map = _HALOGEN_PREFIX
                hal_nbrs = [n for n in cn_nbrs if n.GetSymbol() in hal_map]
                non_hal = [n for n in cn_nbrs
                           if n.GetSymbol() not in hal_map
                           and n.GetSymbol() != 'H']
                if hal_nbrs and not non_hal and nbr.GetTotalNumHs() == 0:
                    hal_counts: Dict[str, int] = defaultdict(int)
                    for h in hal_nbrs:
                        hal_counts[h.GetSymbol()] += 1
                    parts = []
                    for h_sym in sorted(hal_counts.keys(),
                                        key=lambda s: hal_map[s]):
                        cnt = hal_counts[h_sym]
                        mp = _gmp(cnt, hal_map[h_sym]) if cnt > 1 else ''
                        parts.append(f'{mp}{hal_map[h_sym]}')
                    haloalkyl_name = '(' + ''.join(parts) + 'methyl)'
                    sub_groups[haloalkyl_name].append(iupac_locant)
                    continue

                # (c) General alkyl using _count_pure_alkyl()
                c_count = _count_pure_alkyl(
                    mol, ni, core_atom_set | chain_set | ring_atom_set
                )
                if c_count:
                    try:
                        sub_groups[get_alkyl_name(c_count)].append(iupac_locant)
                    except (ValueError, KeyError):
                        pass  # Skip unrecognizable

            elif sym == 'O':
                h_count = nbr.GetTotalNumHs()
                o_nbrs = [n for n in nbr.GetNeighbors()
                          if n.GetIdx() != core_idx
                          and n.GetIdx() not in core_atom_set]
                if h_count == 1 and len(o_nbrs) == 0:
                    sub_groups['hydroxy'].append(iupac_locant)
                elif (h_count == 0 and len(o_nbrs) == 1
                      and o_nbrs[0].GetSymbol() == 'C'):
                    # General alkoxy: count carbons in the -O-C... chain
                    c_start_idx = o_nbrs[0].GetIdx()
                    c_count = _count_pure_alkyl(
                        mol, c_start_idx,
                        core_atom_set | chain_set | ring_atom_set | {ni}
                    )
                    if c_count:
                        if c_count in _ALKOXY_DIRECT:
                            sub_groups[_ALKOXY_DIRECT[c_count]].append(
                                iupac_locant)
                        else:
                            from ..data.chain_names import get_chain_prefix
                            sub_groups[
                                f'{get_chain_prefix(c_count)}oxy'
                            ].append(iupac_locant)

            elif sym == 'N':
                # Nitrogen neighbor -- check nitro vs amino
                n_nbrs = [n for n in nbr.GetNeighbors()
                          if n.GetIdx() != core_idx
                          and n.GetIdx() not in core_atom_set]
                if (nbr.GetFormalCharge() == 1
                        and len(n_nbrs) == 2
                        and all(n.GetSymbol() == 'O' for n in n_nbrs)):
                    sub_groups['nitro'].append(iupac_locant)
                elif nbr.GetTotalNumHs() == 2 and len(n_nbrs) == 0:
                    sub_groups['amino'].append(iupac_locant)

    if not sub_groups:
        return ""

    # Build prefix string sorted by IUPAC alphabetization
    from ..assembly.naming_utils import get_multiplier_prefix, is_complex_substituent
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locs = sorted(sub_groups[name], key=lambda x: (int(x) if str(x).isdigit() else 999, str(x)))
        count = len(locs)
        loc_str = ','.join(str(l) for l in locs)
        if count == 1:
            prefix_parts.append(f'{loc_str}-{name}')
        else:
            mult = get_multiplier_prefix(count, name)
            prefix_parts.append(f'{loc_str}-{mult}{name}')

    return '-'.join(prefix_parts) + '-' if prefix_parts else ""


def _merge_connected_ring_groups(
    mol, ring_groups: list
) -> tuple:
    """Merge ring substituent tuples that are connected by single bonds.

    When parent selection identifies ring substituents, each SSSR ring
    appears as a separate tuple in ring_substituents_as_groups. For
    multi-ring fragments like biphenyl, two separate 6-membered ring
    tuples need to be merged into a single multi-ring fragment before
    naming.

    Uses union-find to group ring tuples connected by direct single
    bonds between their atoms.

    Args:
        mol: RDKit Mol object
        ring_groups: List of tuples of atom indices (from ring_substituents_as_groups)

    Returns:
        Tuple of (multi_ring_fragments, single_ring_groups) where:
        - multi_ring_fragments: list of lists of ring tuples (each with 2+ rings)
        - single_ring_groups: list of ring tuples not part of any multi-ring fragment
    """
    from rdkit.Chem import rdchem

    n = len(ring_groups)
    if n < 2:
        return [], list(ring_groups)

    # Build sets for quick membership check
    ring_sets = [set(rg) for rg in ring_groups]

    # Union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Acceptable inter-ring bond types (biphenyl bond may be typed as
    # SINGLE or AROMATIC depending on kekulization)
    acceptable_types = {
        rdchem.BondType.SINGLE,
        rdchem.BondType.AROMATIC,
    }

    # Check all pairs of ring groups for connecting single bonds
    for i in range(n):
        for j in range(i + 1, n):
            # Look for a bond between an atom in ring_groups[i] and ring_groups[j]
            found = False
            for ai in ring_sets[i]:
                atom = mol.GetAtomWithIdx(ai)
                for nbr in atom.GetNeighbors():
                    nj = nbr.GetIdx()
                    if nj in ring_sets[j]:
                        # Check bond type
                        bond = mol.GetBondBetweenAtoms(ai, nj)
                        if bond and bond.GetBondType() in acceptable_types:
                            union(i, j)
                            found = True
                            break
                if found:
                    break

    # Group by root
    groups_by_root: dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_by_root:
            groups_by_root[root] = []
        groups_by_root[root].append(i)

    multi_ring_fragments = []
    single_ring_groups = []

    for root, members in groups_by_root.items():
        if len(members) >= 2:
            multi_ring_fragments.append([ring_groups[m] for m in members])
        else:
            single_ring_groups.append(ring_groups[members[0]])

    return multi_ring_fragments, single_ring_groups


def _generate_ring_substituent_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for rings that are substituents on a chain parent.

    When parent selection determines chain is parent (chain_is_parent=True),
    rings become substituents and need to be named as prefixes (phenyl, cyclohexyl, etc.).

    This implements IUPAC P-61.5 ring-as-substituent naming:
    - benzene -> phenyl
    - cyclohexane -> cyclohexyl
    - pyridine -> pyridyl
    - Fused heterocycles -> stem-locant-yl (quinolin-2-yl, 1H-indol-3-yl, etc.)

    Args:
        features: MolecularFeatures with ring_substituents_as_groups populated

    Returns:
        List of NameFragment objects for ring substituent prefixes
    """
    from ..rules.ring_substituents import get_ring_substituent_name, get_ring_attachment_locant
    from ..data.fused_heterocycles import match_fused_heterocycle_core, get_fused_heterocycle_prefix, get_substituted_fused_het_prefix

    prefixes = []
    ring_groups = getattr(features, 'ring_substituents_as_groups', [])

    if not ring_groups:
        return prefixes

    # Group ring substituents by name for multiplier handling
    ring_sub_groups: Dict[str, List[int]] = defaultdict(list)

    chain_set = set(features.principal_chain)

    # --- Phase 82: Multi-ring fragment detection pass ---
    # Before processing individual rings, detect multi-ring fragments
    # (biphenyl, terphenyl, phenyl-pyridyl pairs, etc.) where ring groups
    # are connected by single bonds. These must be named as ring assembly
    # prefixes (P-28.3) or compound substituent prefixes (P-31) instead
    # of as independent single rings.
    from ..rules.ring_assemblies import (
        detect_ring_assembly,
        name_ring_assembly_prefix,
        name_mixed_ring_prefix,
        _find_inter_system_bonds,
    )
    from ..rules.ring_substituents import get_ring_attachment_locant as _get_ring_attach_loc

    multi_ring_fragments, single_ring_groups = _merge_connected_ring_groups(
        features.mol, ring_groups
    )

    consumed_ring_sets = set()  # Track ring tuples consumed by multi-ring naming

    for fragment in multi_ring_fragments:
        # fragment is a list of ring tuples (each from ring_substituents_as_groups)
        ring_systems = [set(rg) for rg in fragment]

        # Find which atom in the fragment connects to the chain
        attach_atom_idx = None
        attach_chain_locant = None
        all_frag_atoms = set()
        for rg in fragment:
            all_frag_atoms.update(set(rg))

        # Build a combined tuple of all fragment atoms for locant lookup
        combined_tuple = tuple(sorted(all_frag_atoms))

        for ra in all_frag_atoms:
            atom = features.mol.GetAtomWithIdx(ra)
            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx in chain_set and nbr_idx in features.atom_to_locant:
                    attach_atom_idx = ra
                    attach_chain_locant = features.atom_to_locant[nbr_idx]
                    break
            if attach_atom_idx is not None:
                break

        if attach_atom_idx is None or attach_chain_locant is None:
            # Could not find chain attachment -- fall back to single-ring processing
            single_ring_groups.extend(fragment)
            continue

        # Try ring assembly detection (identical rings)
        assembly_info = detect_ring_assembly(features.mol, ring_systems)
        if assembly_info is not None:
            prefix = name_ring_assembly_prefix(
                features.mol, assembly_info, attach_atom_idx
            )
            if prefix is not None:
                # Ring assembly prefixes contain square brackets and locants,
                # so they need parentheses wrapping per IUPAC P-16.3.3:
                # 4-([1,1'-biphenyl]-4-yl)butanoic acid
                ring_sub_groups[f'({prefix})'].append(attach_chain_locant)
                # Mark all rings in this fragment as consumed
                for rg in fragment:
                    consumed_ring_sets.add(tuple(sorted(rg)))
                continue

        # Not an assembly (non-identical rings) -- try compound prefix
        inter_bonds = _find_inter_system_bonds(features.mol, ring_systems)
        prefix = name_mixed_ring_prefix(
            features.mol, ring_systems, inter_bonds, attach_atom_idx
        )
        if prefix is not None:
            ring_sub_groups[prefix].append(attach_chain_locant)
            for rg in fragment:
                consumed_ring_sets.add(tuple(sorted(rg)))
            continue

        # Multi-ring naming failed -- fall back to single-ring processing
        single_ring_groups.extend(fragment)

    # --- End Phase 82 multi-ring detection pass ---

    for ring_atoms in single_ring_groups:
        from ..perception.rings import get_containing_ring_system
        ring_atom_set = set(get_containing_ring_system(features.mol, ring_atoms))

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

        # --- Phase 78: Check for fused heterocycle substituent ---
        # Try to match the ring fragment against known fused heterocycle cores.
        # If matched, use static O(1) prefix lookup instead of monocyclic naming.
        fused_result = match_fused_heterocycle_core(features.mol)
        if fused_result is not None:
            het_name, atom_mapping, core_smiles = fused_result
            # Check that this fused het core overlaps with the ring atoms
            core_atom_set = set(atom_mapping.keys())
            if core_atom_set & ring_atom_set:
                # Find the ring atom that attaches to the chain
                attach_ring_idx = None
                for ra in ring_atoms:
                    atom = features.mol.GetAtomWithIdx(ra)
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in chain_set:
                            attach_ring_idx = ra
                            break
                    if attach_ring_idx is not None:
                        break

                if attach_ring_idx is not None:
                    # Detect inner substituents on the fused het core
                    inner_prefix = _detect_fused_het_inner_subs(
                        features.mol, core_atom_set, atom_mapping,
                        ring_atom_set, chain_set, attach_ring_idx
                    )
                    if inner_prefix:
                        # Substituted fused het: compound prefix
                        cpx = get_substituted_fused_het_prefix(
                            core_smiles, attach_ring_idx,
                            atom_mapping, inner_prefix
                        )
                        if cpx is not None:
                            ring_sub_groups[cpx].append(locant)
                            continue
                    # Unsubstituted fused het: simple prefix
                    prefix = get_fused_heterocycle_prefix(
                        core_smiles, attach_ring_idx, atom_mapping
                    )
                    if prefix is not None:
                        # Fused het prefixes always need parentheses
                        # (they contain locants and hyphens per IUPAC P-16.3.3)
                        ring_sub_groups[f'({prefix})'].append(locant)
                        continue
        # --- End Phase 78 fused het detection ---

        # Find ring attachment atom for position-specific naming
        _ring_attach_atom = None
        for ra in ring_atoms:
            atom = features.mol.GetAtomWithIdx(ra)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in chain_set:
                    _ring_attach_atom = ra
                    break
            if _ring_attach_atom is not None:
                break
        # Also check expanded ring system atoms
        if _ring_attach_atom is None:
            for ra in ring_atom_set:
                atom = features.mol.GetAtomWithIdx(ra)
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in chain_set:
                        _ring_attach_atom = ra
                        break
                if _ring_attach_atom is not None:
                    break

        # Get base substituent name (phenyl, cyclohexyl, etc.)
        # For multi-ring systems, try the full fused system first for retained name
        # lookup (naphthalene, anthracene), but fall back to the SSSR ring if no
        # retained name is found (to avoid renaming a 5-membered pyrrole ring in
        # a porphyrin as "cyclononacosyl").
        if len(ring_atom_set) > len(ring_atoms):
            _full_ring = tuple(sorted(ring_atom_set))
            _full_name = get_ring_substituent_name(features.mol, _full_ring, _ring_attach_atom)
            # Accept the full-system name only if it produced a retained name
            # (not a generic cyclo-name)
            if not _full_name.startswith('cyclo') or _full_name in ('cyclohexyl', 'cyclopentyl', 'cyclopropyl', 'cyclobutyl', 'cycloheptyl', 'cyclooctyl'):
                base_name = _full_name
            else:
                # Full system produced generic cyclo-name; use original SSSR ring
                base_name = get_ring_substituent_name(features.mol, ring_atoms, _ring_attach_atom)
        else:
            base_name = get_ring_substituent_name(features.mol, ring_atoms, _ring_attach_atom)

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
    from ..perception.rings import get_containing_ring_system

    # Use the complete ring system as BFS boundary (IUPAC P-25.3)
    ring_atom_set = set(get_containing_ring_system(mol, ring_atoms))

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
    from ..assembly.naming_utils import get_multiplier_prefix, is_complex_substituent
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locs = sorted(sub_groups[name])
        count = len(locs)
        loc_str = ','.join(str(l) for l in locs)
        if count == 1:
            if is_complex_substituent(name):
                prefix_parts.append(f'{loc_str}-({name})')
            else:
                prefix_parts.append(f'{loc_str}-{name}')
        else:
            mult = get_multiplier_prefix(count, name)
            if is_complex_substituent(name):
                prefix_parts.append(f'{loc_str}-{mult}({name})')
            else:
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
                elif h_count == 0 and len(o_nbrs) == 1 and o_nbrs[0].GetSymbol() == 'S':
                    # O -> S: check for sulfooxy (-O-S(=O)(=O)-OH)
                    s_idx = o_nbrs[0].GetIdx()
                    s_atom = mol.GetAtomWithIdx(s_idx)
                    # Count =O bonds and -OH bonds on sulfur
                    s_dbl_o = 0
                    s_oh = 0
                    for s_nbr in s_atom.GetNeighbors():
                        if s_nbr.GetIdx() == ni:
                            continue  # skip the O that links to ring
                        if s_nbr.GetSymbol() == 'O':
                            bond = mol.GetBondBetweenAtoms(s_idx, s_nbr.GetIdx())
                            if bond and bond.GetBondTypeAsDouble() == 2.0:
                                s_dbl_o += 1
                            elif s_nbr.GetTotalNumHs() == 1:
                                s_oh += 1
                    if s_dbl_o == 2 and s_oh >= 1:
                        sub_groups['sulfooxy'].append(ring_pos)
                    elif s_dbl_o == 2 and s_oh == 0:
                        # -O-S(=O)(=O)- without OH: sulfonyloxy
                        sub_groups['sulfonyloxy'].append(ring_pos)
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
            elif sym == 'P':
                # Phosphanyl prefix: detect P with aryl/alkyl substituents
                # Exclude ring atoms so the attachment carbon isn't counted
                from ..rules.phosphorus import get_phosphanyl_prefix
                prefix = get_phosphanyl_prefix(mol, ni, exclude_atoms=ring_atom_set)
                if prefix:
                    sub_groups[prefix].append(ring_pos)
            elif sym == 'C':
                bond = mol.GetBondBetweenAtoms(atom_idx, ni)
                if bond and bond.GetBondTypeAsDouble() == 1.0:
                    alkyl_atoms = _collect_pure_alkyl_atoms(mol, ni, ring_atom_set)
                    if alkyl_atoms:
                        c_count = len(alkyl_atoms)
                        sub_name = None
                        # Try recursive naming (handles retained names + branched)
                        if c_count > 0:
                            from .substituent_naming import name_substituent_fragment
                            sub_name = name_substituent_fragment(
                                mol, alkyl_atoms, ni, list(ring_atom_set)
                            )
                        if sub_name is None:
                            from ..data.chain_names import get_alkyl_name as _gal
                            try:
                                sub_name = _gal(c_count)
                            except (ValueError, KeyError):
                                pass
                        if sub_name:
                            sub_groups[sub_name].append(ring_pos)

    return dict(sub_groups) if sub_groups else None


def _count_pure_alkyl(mol, start_idx, excluded):
    """Count carbons in a pure alkyl chain from start_idx."""
    visited = {start_idx}
    queue = deque([start_idx])
    carbon_count = 0

    while queue:
        idx = queue.popleft()
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


def _collect_pure_alkyl_atoms(mol, start_idx, excluded):
    """Collect atom indices of a pure alkyl chain from start_idx.

    Returns list of carbon atom indices, or None if non-carbon encountered.
    """
    visited = {start_idx}
    queue = deque([start_idx])
    atoms = []

    while queue:
        idx = queue.popleft()
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None  # Not pure alkyl
        atoms.append(idx)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni not in visited and ni not in excluded:
                visited.add(ni)
                queue.append(ni)

    return atoms if atoms else None


def _generate_alkyl_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for all substituents on chain parents.

    Uses the universal pipeline (extract_chain_substituents +
    classify_and_name_fragment) to enumerate and name every non-hydrogen
    substituent on the principal chain. Mirrors the proven
    _generate_ring_alkyl_prefixes() pattern.

    IUPAC rules:
    - P-31.1.1: Simple substituents (unbranched, no FG) -> di-/tri- multipliers
    - P-31.1.2: Compound substituents (branched or substituted) -> bis-/tris-
    - P-14.5: Alphabetical order ignoring multiplicative prefixes
    - P-16.3.2: Multiplicative prefix selection based on complexity

    Returns:
        List of NameFragment objects for chain substituent prefixes,
        sorted alphabetically.
    """
    mol = features.mol

    # Collect ring atoms that should be skipped (handled by ring substituent prefixes)
    ring_atoms_to_skip: set = set()
    if getattr(features, 'chain_is_parent', False):
        ring_groups = getattr(features, 'ring_substituents_as_groups', [])
        for ring_atoms in ring_groups:
            ring_atoms_to_skip.update(ring_atoms)

    # Collect atoms belonging to the principal group (handled as suffix)
    pg_atom_set = set()
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            pg_atom_set.update(match)

    # Use universal pipeline to discover all chain substituents
    from .substituent_enumerator import extract_chain_substituents
    sub_infos = extract_chain_substituents(
        mol, features.principal_chain or [], features.substituents or {}
    )

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)
    chain_set = set(features.principal_chain) if features.principal_chain else set()

    for sub_info in sub_infos:
        # Guard 1: Skip substituents overlapping with ring atoms already
        # handled by _generate_ring_substituent_prefixes.
        # BUT if the attachment point is a non-ring heteroatom linker
        # (e.g., N in N-quinolinylamino), let it through.
        if ring_atoms_to_skip and sub_info.frag_atoms & ring_atoms_to_skip:
            _attach_is_hetero_linker = False
            for _idx in sub_info.frag_atoms:
                _atom = mol.GetAtomWithIdx(_idx)
                if any(nbr.GetIdx() in chain_set for nbr in _atom.GetNeighbors()):
                    if _atom.GetSymbol() not in ('C', 'H') and _idx not in ring_atoms_to_skip:
                        _attach_is_hetero_linker = True
                    break
            if not _attach_is_hetero_linker:
                logger.debug(
                    "DROP-02 substituent_skip: reason=ring_overlap locant=%d",
                    sub_info.locant,
                )
                continue

        # Guard 2: Skip substituents overlapping with the principal group
        # (amide/amine N-substituents handled by specialized assemblers)
        if features.principal_group in (
            'primary_amide', 'secondary_amide', 'tertiary_amide',
            'secondary_amine', 'tertiary_amine',
        ):
            if sub_info.frag_atoms & pg_atom_set:
                logger.debug(
                    "DROP-03 substituent_skip: reason=pg_branch_overlap locant=%d pg=%s",
                    sub_info.locant, features.principal_group,
                )
                continue

        # Guard 3: Skip pure FG-only substituents (no carbon atoms)
        # These are handled by the FG prefix loop in _generate_prefixes()
        has_carbon = any(
            mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            for idx in sub_info.frag_atoms
        )
        if not has_carbon:
            logger.debug(
                "DROP-01 substituent_skip: reason=fg_only locant=%d (by-design: FG prefix loop handles these)",
                sub_info.locant,
            )
            continue

        # Guard 3b: Skip substituent branches entirely covered by a single
        # non-principal FG match ONLY for FG types whose prefix form includes
        # the carbon (carbamoyl, carboxy, carbonochloridoyl, etc.).
        # These branches ARE the FG and should be emitted as FG prefixes
        # by _generate_prefixes(), NOT as alkyl compound substituents.
        # IUPAC P-66.1(c): non-principal amide = carbamoyl prefix.
        # NOTE: Only applies to specific terminal-C FG types. Other FGs
        # (amine, ketone, secondary_amide, etc.) must NOT trigger this guard.
        _GUARD3B_FG_TYPES = {
            'primary_amide', 'carboxylic_acid',
            'acid_chloride', 'acid_bromide', 'acid_fluoride',
        }
        _skip_as_fg_branch = False
        for _fg_name, _fg_matches in features.functional_groups.items():
            if _fg_name == features.principal_group:
                continue
            if _fg_name not in _GUARD3B_FG_TYPES:
                continue
            for _fg_match in _fg_matches:
                fg_match_set = set(_fg_match)
                if fg_match_set and sub_info.frag_atoms.issubset(fg_match_set):
                    _skip_as_fg_branch = True
                    break
            if _skip_as_fg_branch:
                break
        if _skip_as_fg_branch:
            continue

        # IUPAC P-31.1.3.1 / P-29.1(b): Detect exocyclic double bond attachment.
        # If the bond from the chain atom to the substituent is DOUBLE,
        # the substituent uses -ylidene suffix instead of -yl.
        is_exocyclic_double = False
        attach_idx = sub_info.attach_mol_idx
        for frag_idx in sub_info.frag_atoms:
            bond = mol.GetBondBetweenAtoms(attach_idx, frag_idx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                is_exocyclic_double = True
                break

        # Name via the universal classify-and-name pipeline
        name = classify_and_name_fragment(mol, sub_info, chain_set, features)
        if name is None:
            # Fallback: try simple carbon-count alkyl naming
            carbon_count = sum(
                1 for idx in sub_info.frag_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )
            if carbon_count > 0:
                try:
                    name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    pass
            if name is None:
                logger.warning(
                    "DROP-09 substituent_skip: reason=universal_pipeline_unnameable locant=%d",
                    sub_info.locant,
                )
                continue

        # IUPAC P-31.1.3.1: Convert -yl to -ylidene for exocyclic double bonds.
        # =CH2 -> methylidene, =CHCH3 -> ethylidene, =C(CH3)2 -> propan-2-ylidene
        if is_exocyclic_double and name:
            name = _convert_yl_to_ylidene(name)

        substituent_groups[name].append(sub_info.locant)

    # INST-01: Atom coverage audit
    if logger.isEnabledFor(logging.DEBUG):
        total_heavy = features.mol.GetNumHeavyAtoms()
        parent_count = len(features.principal_chain or [])
        named_count = sum(len(locs) for locs in substituent_groups.values()) if substituent_groups else 0
        coverage = (parent_count + named_count) / max(total_heavy, 1)
        logger.debug(
            "ATOM_COVERAGE: smiles=%s total_heavy=%d parent=%d named_subs=%d coverage=%.2f",
            features.canonical_smiles, total_heavy, parent_count, named_count, coverage,
        )
        # ATOM_COVERAGE_DETAIL: actual heavy atoms in named substituent fragments
        named_atom_count = sum(
            len(si.frag_atoms) for si in sub_infos
            if si.frag_atoms and not (pg_atom_set and si.frag_atoms & pg_atom_set)
        )
        logger.debug(
            "ATOM_COVERAGE_DETAIL: smiles=%s named_atom_count=%d",
            features.canonical_smiles, named_atom_count,
        )

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

        # Omit locant for monosubstituted hydrocarbons at position 1.
        # The is_monosubstituted guard encodes the full original condition so
        # that multi-substituent methane derivatives still go through
        # format_substituent_prefix for bracket wrapping.
        if should_omit_locant_one(
            context="prefix",
            chain_length=len(getattr(features, 'principal_chain', [])),
            is_monosubstituted=(is_simple_hydrocarbon and total_substituents == 1
                                and sorted_locants == [1]),
        ) and total_substituents == 1:
            formatted = name
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
                # --- Phase 79-02: Try fused het detection before anilino ---
                # Fused hets like quinoline contain a benzene sub-ring that
                # would falsely match the all-C aromatic phenyl check.
                from ..data.fused_heterocycles import (
                    match_fused_heterocycle_core as _match_fh_acylamino,
                    get_fused_heterocycle_prefix as _get_fh_prefix_acylamino,
                )
                fused_r_acylamino = _match_fh_acylamino(mol)
                if fused_r_acylamino is not None:
                    fh_name_a, fh_mapping_a, fh_core_a = fused_r_acylamino
                    fh_atoms_a = set(fh_mapping_a.keys())
                    if fh_atoms_a & sub_set:
                        # Find ring atom bonded to N
                        fh_attach_a = None
                        for ra in fh_atoms_a & sub_set:
                            atom_ra = mol.GetAtomWithIdx(ra)
                            for nbr_fh in atom_ra.GetNeighbors():
                                if nbr_fh.GetIdx() == idx:  # idx is the N atom
                                    fh_attach_a = ra
                                    break
                            if fh_attach_a is not None:
                                break
                        if fh_attach_a is not None:
                            fh_prefix_a = _get_fh_prefix_acylamino(fh_core_a, fh_attach_a, fh_mapping_a)
                            if fh_prefix_a is not None:
                                non_core_c_a = sum(
                                    1 for i in sub_set
                                    if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                                    and i not in fh_atoms_a
                                )
                                if non_core_c_a == 0:
                                    return f"(({fh_prefix_a})amino)"
                # --- End Phase 79-02 fused het in acylamino ---
                # Check for isolated phenyl (not part of a fused system)
                for ring in ring_info.AtomRings():
                    if all(r in sub_set for r in ring) and len(ring) == 6:
                        all_arom = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                        all_c = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                        if all_arom and all_c:
                            # Verify this is an isolated benzene (not part of fused system)
                            ring_set_chk_a = set(ring)
                            is_fused_a = False
                            for other_ring in ring_info.AtomRings():
                                if set(other_ring) != ring_set_chk_a and set(other_ring) & ring_set_chk_a:
                                    is_fused_a = True
                                    break
                            if not is_fused_a:
                                return "anilino"
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

        # Found carbonyl: identify the C=O oxygen
        carbonyl_o = None
        for nbr in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
            if nbr.GetSymbol() == 'O' and nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(carbonyl_c, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() == 2.0:
                    carbonyl_o = nbr.GetIdx()
                    break

        # PEP-04 fix: Check if carbonyl_c is directly bonded to a ring
        # carbon. This detects aromatic acyl groups (benzoyl, naphthoyl)
        # and cycloalkane-carbonyl groups (cyclopentanecarbonyl) where
        # _count_carbon_chain() would incorrectly linearize ring C-C bonds.
        _ri = mol.GetRingInfo()
        _has_ring_neighbor = False
        for _cn in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
            if (_cn.GetSymbol() == 'C' and _cn.GetIdx() in sub_set
                    and _cn.GetIdx() != idx
                    and _ri.NumAtomRings(_cn.GetIdx()) > 0):
                _has_ring_neighbor = True
                break

        if _has_ring_neighbor:
            # Ring directly on carbonyl: extract acyl fragment as acid
            # SMILES, name it, convert to acyl prefix.
            from collections import deque as _dq
            _exc = chain_set | {idx}
            if carbonyl_o is not None:
                _exc.add(carbonyl_o)
            _ccs = set()
            _qq = _dq([carbonyl_c])
            while _qq:
                _aa = _qq.popleft()
                if _aa in _ccs or _aa in _exc:
                    continue
                _at = mol.GetAtomWithIdx(_aa)
                if _at.GetSymbol() != 'C':
                    continue
                _ccs.add(_aa)
                for _nb in _at.GetNeighbors():
                    _ni = _nb.GetIdx()
                    if (_ni not in _ccs and _ni not in _exc
                            and _nb.GetSymbol() == 'C'):
                        _qq.append(_ni)
            _af = _ccs.copy()
            if carbonyl_o is not None:
                _af.add(carbonyl_o)
            try:
                from rdkit import Chem as _Ch
                _rw = _Ch.RWMol(mol)
                _oh = _rw.AddAtom(_Ch.Atom(8))
                _rw.AddBond(carbonyl_c, _oh, _Ch.BondType.SINGLE)
                _hh = _rw.AddAtom(_Ch.Atom(1))
                _rw.AddBond(_oh, _hh, _Ch.BondType.SINGLE)
                _fa = sorted(_af | {_oh, _hh})
                _fs = _Ch.MolFragmentToSmiles(_rw, _fa, canonical=True)
                if _fs:
                    from .fragment_naming import name_fragment_recursively
                    _an = name_fragment_recursively(_fs)
                    if _an:
                        from ..decomposition.fragment_assembly import (
                            _acid_to_acyl,
                        )
                        _ac = _acid_to_acyl(_an)
                        if _ac:
                            return f"({_ac}amino)"
            except Exception:
                pass  # Fall through to linear chain logic

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

    Returns bare name like "acetyloxy" for -O-C(=O)-CH3.
    Callers handle parenthesization via is_complex_substituent() and
    format_substituent_prefix(). Per IUPAC P-16.3.3, acyloxy groups are
    compound prefixes requiring complex multipliers: bis(acetyloxy).
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

                # Build acyloxy name: bare prefixanoyloxy (no parens)
                # Callers use format_substituent_prefix() for IUPAC parens
                # Check trivial acid name first, then fall back to systematic
                from ..data.chain_names import get_acid_stem
                from ..rules.esters import get_acyloxy_prefix as _get_acyloxy
                try:
                    acid_stem = get_acid_stem(acyl_carbons)
                    acyloxy = _get_acyloxy(acid_stem)
                    return acyloxy
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
            # --- Phase 79-02: Fused heterocycle detection for N-branch ---
            # Try fused heterocycle identification FIRST (O(1) static lookup,
            # Phase 78 infrastructure). This must come before the anilino check
            # because fused hets like quinoline contain a benzene ring that
            # would falsely match the all-C aromatic 6-membered ring test.
            from ..data.fused_heterocycles import (
                match_fused_heterocycle_core as _match_fused_het,
                get_fused_heterocycle_prefix as _get_fused_het_prefix,
            )
            fused_result = _match_fused_het(mol)
            if fused_result is not None:
                het_name, atom_mapping, core_smiles = fused_result
                core_atom_set = set(atom_mapping.keys())
                # Check that fused het core overlaps with substituent atoms
                if core_atom_set & sub_set:
                    # Find the ring atom that attaches to N (or nearest to N)
                    attach_ring_idx = None
                    for ra in core_atom_set & sub_set:
                        atom_ra = mol.GetAtomWithIdx(ra)
                        for nbr in atom_ra.GetNeighbors():
                            if nbr.GetIdx() == attach_atom:  # attach_atom is the N
                                attach_ring_idx = ra
                                break
                        if attach_ring_idx is not None:
                            break
                    if attach_ring_idx is None:
                        # Fused het may not be directly bonded to N -- check through intermediate atoms
                        for ra in core_atom_set & sub_set:
                            atom_ra = mol.GetAtomWithIdx(ra)
                            for nbr in atom_ra.GetNeighbors():
                                if nbr.GetIdx() in sub_set and nbr.GetIdx() not in core_atom_set:
                                    attach_ring_idx = ra
                                    break
                            if attach_ring_idx is not None:
                                break
                    if attach_ring_idx is not None:
                        prefix = _get_fused_het_prefix(core_smiles, attach_ring_idx, atom_mapping)
                        if prefix is not None:
                            # Check if ring accounts for all sub atoms (pure fused het, no linker)
                            non_core_c = sum(
                                1 for idx in sub_set
                                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                                and idx not in core_atom_set
                            )
                            if non_core_c == 0:
                                # Pure fused het on N: e.g., (quinolin-8-ylamino)
                                return f"(({prefix})amino)"
                            # Fused het with linker carbons: fall through to recursive naming
            # --- End Phase 79-02 fused het detection for N-branch ---

            # Substituent has a ring - check for phenyl/benzene (anilino)
            # Only match if the ring is NOT part of a fused system (to avoid
            # falsely matching benzene ring of quinoline etc.)
            for ring in ring_info.AtomRings():
                if all(r in sub_set for r in ring) and len(ring) == 6:
                    all_arom = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                    all_c = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                    if all_arom and all_c:
                        # Verify this is an isolated benzene ring (not part of fused system)
                        ring_set_check = set(ring)
                        is_fused = False
                        for other_ring in ring_info.AtomRings():
                            if set(other_ring) != ring_set_check and set(other_ring) & ring_set_check:
                                is_fused = True
                                break
                        if not is_fused:
                            return "anilino"

            # --- Phase 79-01: Direct ring identification for N-branch (DROP-18 fix) ---
            # Try O(1) ring identification before recursive naming fallback.
            # Only applies when the substituent IS a pure ring (all C atoms are ring atoms).
            from ..rules.ring_substituents import get_ring_substituent_name as _get_ring_sub_name
            from ..rules.ring_substituents import identify_ring_system as _identify_ring
            for ring in ring_info.AtomRings():
                ring_set_inner = set(ring)
                if ring_set_inner.issubset(sub_set):
                    # Check if ring accounts for all C atoms in substituent
                    # (pure ring vs ring+chain like cyclohexylmethyl)
                    non_ring_c = sum(
                        1 for idx in sub_atoms
                        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                        and idx not in ring_set_inner
                    )
                    if non_ring_c == 0:
                        ring_prefix = _get_ring_sub_name(mol, tuple(ring))
                        if ring_prefix:
                            return f"({ring_prefix}amino)"
            # --- End Phase 79-01 direct ring identification ---

            # Non-phenyl ring: try recursive naming (piperidinyl, cyclohexyl, etc.)
            # Guard: only for moderately-sized substituents (<=25 atoms).
            if len(sub_atoms) <= 25:
                from .substituent_naming import name_substituent_fragment
                from .naming_utils import needs_brackets
                sub_name = name_substituent_fragment(
                    mol, list(sub_set), attach_atom, list(chain_set)
                )
                # Validate: reject only if fragment has ring atoms but name is acyclic
                # (Phase 85 structural validation replaces pure string-based check)
                if sub_name and _has_ring_atoms(mol, sub_atoms) and not _name_reflects_ring(sub_name):
                    # Likely linearized a ring -- reject
                    sub_name = None
                if sub_name:
                    if needs_brackets(sub_name):
                        sub_name = f"({sub_name})"
                    return sub_name
            # Fallback: recursive naming via name_fragment_recursively()
            # for ring-containing N-branch fragments that failed direct naming.
            if len(sub_atoms) <= 25:
                try:
                    frag_smiles = Chem.MolFragmentToSmiles(mol, list(sub_set))
                    if frag_smiles:
                        from .fragment_naming import name_fragment_recursively
                        from .substituent_naming import parent_to_prefix
                        frag_name = name_fragment_recursively(frag_smiles)
                        if frag_name:
                            carbon_count = sum(
                                1 for idx in sub_atoms
                                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                            )
                            prefix = parent_to_prefix(frag_name, chain_length=carbon_count)
                            if prefix:
                                from .naming_utils import needs_brackets
                                if needs_brackets(prefix):
                                    prefix = f"({prefix})"
                                return f"({prefix}amino)"
                except Exception:
                    pass
            logger.debug(
                "DROP-18 substituent_skip: reason=n_branch_nonphenyl_ring_still_unnameable",
            )
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
            # --- Phase 79-02: Fused heterocycle detection for C-branch ---
            # Try fused het identification first (O(1) static lookup).
            from ..data.fused_heterocycles import (
                match_fused_heterocycle_core as _match_fused_het_c,
                get_fused_heterocycle_prefix as _get_fused_het_prefix_c,
            )
            fused_result_c = _match_fused_het_c(mol)
            if fused_result_c is not None:
                het_name_c, atom_mapping_c, core_smiles_c = fused_result_c
                core_atom_set_c = set(atom_mapping_c.keys())
                if core_atom_set_c & sub_set:
                    # Find ring atom attached to the C-branch attachment point
                    attach_ring_idx_c = None
                    for ra in core_atom_set_c & sub_set:
                        atom_ra = mol.GetAtomWithIdx(ra)
                        for nbr in atom_ra.GetNeighbors():
                            if nbr.GetIdx() == attach_atom:
                                attach_ring_idx_c = ra
                                break
                            if nbr.GetIdx() in chain_set:
                                attach_ring_idx_c = ra
                                break
                        if attach_ring_idx_c is not None:
                            break
                    if attach_ring_idx_c is None:
                        # Check through intermediate atoms
                        for ra in core_atom_set_c & sub_set:
                            atom_ra = mol.GetAtomWithIdx(ra)
                            for nbr in atom_ra.GetNeighbors():
                                if nbr.GetIdx() in sub_set and nbr.GetIdx() not in core_atom_set_c:
                                    attach_ring_idx_c = ra
                                    break
                            if attach_ring_idx_c is not None:
                                break
                    if attach_ring_idx_c is not None:
                        prefix_c = _get_fused_het_prefix_c(core_smiles_c, attach_ring_idx_c, atom_mapping_c)
                        if prefix_c is not None:
                            non_core_c_count = sum(
                                1 for idx in sub_set
                                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                                and idx not in core_atom_set_c
                            )
                            if non_core_c_count == 0:
                                # Pure fused het on C-branch
                                from .naming_utils import needs_brackets
                                wrapped = f"({prefix_c})"
                                return wrapped
                            # Fused het with linker: fall through to recursive naming
            # --- End Phase 79-02 fused het detection for C-branch ---

            # --- Phase 79-01: Direct ring identification for C-branch (DROP-19 fix) ---
            # Try O(1) ring identification before recursive naming fallback.
            from ..rules.ring_substituents import get_ring_substituent_name as _get_ring_sub_name_c
            for ring in ring_info.AtomRings():
                ring_set_inner = set(ring)
                if ring_set_inner.issubset(sub_set):
                    # Check if ring accounts for all C atoms in substituent
                    non_ring_c = sum(
                        1 for idx in sub_atoms
                        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                        and idx not in ring_set_inner
                    )
                    if non_ring_c == 0:
                        ring_prefix = _get_ring_sub_name_c(mol, tuple(ring))
                        if ring_prefix:
                            from .naming_utils import needs_brackets
                            if needs_brackets(ring_prefix):
                                ring_prefix = f"({ring_prefix})"
                            return ring_prefix
            # --- End Phase 79-01 direct ring identification ---

            # Ring-containing C-branch: try recursive naming
            # Guard: only for moderately-sized substituents (<=25 atoms).
            if len(sub_atoms) <= 25:
                from .substituent_naming import name_substituent_fragment
                from .naming_utils import needs_brackets
                sub_name = name_substituent_fragment(
                    mol, list(sub_set), attach_atom, list(chain_set)
                )
                # Validate: reject only if fragment has ring atoms but name is acyclic
                # (Phase 85 structural validation replaces pure string-based check)
                if sub_name and _has_ring_atoms(mol, sub_atoms) and not _name_reflects_ring(sub_name):
                    sub_name = None
                if sub_name:
                    if needs_brackets(sub_name):
                        sub_name = f"({sub_name})"
                    return sub_name
            # Fallback: recursive naming via name_fragment_recursively()
            # for ring-containing C-branch fragments that failed direct naming.
            if len(sub_atoms) <= 25:
                try:
                    frag_smiles = Chem.MolFragmentToSmiles(mol, list(sub_set))
                    if frag_smiles:
                        from .fragment_naming import name_fragment_recursively
                        from .substituent_naming import parent_to_prefix
                        frag_name = name_fragment_recursively(frag_smiles)
                        if frag_name:
                            carbon_count = sum(
                                1 for idx in sub_atoms
                                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                            )
                            prefix = parent_to_prefix(frag_name, chain_length=carbon_count)
                            if prefix:
                                from .naming_utils import needs_brackets
                                if needs_brackets(prefix):
                                    prefix = f"({prefix})"
                                return prefix
                except Exception:
                    pass
            logger.debug(
                "DROP-19 substituent_skip: reason=c_branch_ring_sub_still_unnameable",
            )
            return None

        # Count carbons via C-C bonds only (don't traverse through heteroatoms)
        total_carbons = _count_carbon_chain(mol, attach_atom, chain_set)
        if total_carbons > 0:
            # Check what heteroatoms are present
            heteroatoms = set()
            halogen_counts = {'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
            other_hetero = 0
            for i in sub_atoms:
                sym = mol.GetAtomWithIdx(i).GetSymbol()
                if sym not in ('C', 'H'):
                    heteroatoms.add(sym)
                    if sym in halogen_counts:
                        halogen_counts[sym] += 1
                    else:
                        other_hetero += 1

            # BUG-A: Haloalkyl naming (check FIRST, before hydroxy/amino)
            # Pure haloalkyl: only C and halogens, no other heteroatoms
            if other_hetero == 0 and any(halogen_counts.values()):
                halogen_prefix_map = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
                if total_carbons == 1:
                    # Single-carbon haloalkyl: trifluoromethyl, dichloromethyl, etc.
                    halogen_parts = []
                    for hal in ['Br', 'Cl', 'F', 'I']:  # alphabetical by prefix
                        cnt = halogen_counts[hal]
                        if cnt > 0:
                            mp = get_multiplier_prefix(cnt, halogen_prefix_map[hal]) if cnt > 1 else ''
                            halogen_parts.append(f'{mp}{halogen_prefix_map[hal]}')
                    return '(' + ''.join(halogen_parts) + 'methyl)'
                elif total_carbons <= 3:
                    # Multi-carbon haloalkyl (2-3 carbons): (2-fluoroethyl), etc.
                    # Find halogen positions via BFS distance from attachment point
                    from collections import deque
                    distances = {}
                    bfs_queue = deque([(attach_atom, 0)])
                    bfs_visited = set()
                    while bfs_queue:
                        curr, dist = bfs_queue.popleft()
                        if curr in bfs_visited or curr in chain_set:
                            continue
                        bfs_visited.add(curr)
                        curr_atom = mol.GetAtomWithIdx(curr)
                        curr_sym = curr_atom.GetSymbol()
                        if curr_sym in halogen_counts and halogen_counts[curr_sym] > 0:
                            distances[curr] = dist
                        for nbr in curr_atom.GetNeighbors():
                            if nbr.GetIdx() not in bfs_visited and nbr.GetIdx() not in chain_set:
                                bfs_queue.append((nbr.GetIdx(), dist + 1))

                    try:
                        alkyl = get_alkyl_name(total_carbons)
                    except (ValueError, KeyError):
                        alkyl = None

                    if alkyl:
                        # Group halogens by position on the sub-chain
                        # For single halogen type at single position: (2-fluoroethyl)
                        # For halogens all on terminal C: (2,2,2-trifluoroethyl)
                        hal_positions = {}  # {halogen_symbol: [sub_locants]}
                        for h_idx, dist in distances.items():
                            h_sym = mol.GetAtomWithIdx(h_idx).GetSymbol()
                            sub_locant = dist  # distance from attachment = sub-chain locant
                            if h_sym not in hal_positions:
                                hal_positions[h_sym] = []
                            hal_positions[h_sym].append(sub_locant)

                        halogen_parts = []
                        for hal in ['Br', 'Cl', 'F', 'I']:
                            if hal not in hal_positions:
                                continue
                            locs = sorted(hal_positions[hal])
                            cnt = len(locs)
                            mp = get_multiplier_prefix(cnt, halogen_prefix_map[hal]) if cnt > 1 else ''
                            loc_str = ','.join(str(l) for l in locs)
                            halogen_parts.append(f'{loc_str}-{mp}{halogen_prefix_map[hal]}')

                        return '(' + ''.join(halogen_parts) + alkyl + ')'

            # BUG-D: For C-chain with OH: name as hydroxyalkyl
            # Locant computed via BFS distance from attachment point
            if heteroatoms == {'O'}:
                # Check if the O is -OH (not C=O or ether)
                for i in sub_atoms:
                    a = mol.GetAtomWithIdx(i)
                    if a.GetSymbol() == 'O' and a.GetDegree() == 1:
                        try:
                            alkyl = get_alkyl_name(total_carbons)
                            if total_carbons == 1:
                                return f"(hydroxy{alkyl})"
                            else:
                                # Find the carbon bearing OH via BFS
                                oh_carbon = None
                                for nb in a.GetNeighbors():
                                    if nb.GetSymbol() == 'C':
                                        oh_carbon = nb.GetIdx()
                                        break
                                # Compute BFS distance from attachment to OH carbon
                                oh_locant = total_carbons  # default: terminal
                                if oh_carbon is not None:
                                    from collections import deque as _deque
                                    _bfs_q = _deque([(attach_atom, 1)])
                                    _bfs_v = set()
                                    while _bfs_q:
                                        _ci, _d = _bfs_q.popleft()
                                        if _ci in _bfs_v or _ci in chain_set:
                                            continue
                                        _bfs_v.add(_ci)
                                        if _ci == oh_carbon:
                                            oh_locant = _d
                                            break
                                        _ca = mol.GetAtomWithIdx(_ci)
                                        if _ca.GetSymbol() == 'C':
                                            for _nb in _ca.GetNeighbors():
                                                _ni = _nb.GetIdx()
                                                if _ni not in _bfs_v and _ni not in chain_set:
                                                    _bfs_q.append((_ni, _d + 1))
                                return f"({oh_locant}-hydroxy{alkyl})"
                        except (ValueError, KeyError):
                            pass

            # BUG-C: For C-chain with NH2: name as (aminoalkyl)
            # e.g., -CH2NH2 -> (aminomethyl), -CH2CH2NH2 -> (2-aminoethyl)
            # Locant computed via BFS distance from attachment point
            if 'N' in heteroatoms:
                # Check for terminal primary amine (-NH2) on the chain
                for i in sub_atoms:
                    a = mol.GetAtomWithIdx(i)
                    if (a.GetSymbol() == 'N' and a.GetDegree() == 1
                            and a.GetTotalNumHs() == 2):
                        try:
                            alkyl = get_alkyl_name(total_carbons)
                            if total_carbons == 1:
                                return f"(amino{alkyl})"
                            else:
                                # Find the carbon bearing NH2 via BFS
                                nh2_carbon = None
                                for nb in a.GetNeighbors():
                                    if nb.GetSymbol() == 'C':
                                        nh2_carbon = nb.GetIdx()
                                        break
                                nh2_locant = total_carbons  # default: terminal
                                if nh2_carbon is not None:
                                    from collections import deque as _deque
                                    _bfs_q = _deque([(attach_atom, 1)])
                                    _bfs_v = set()
                                    while _bfs_q:
                                        _ci, _d = _bfs_q.popleft()
                                        if _ci in _bfs_v or _ci in chain_set:
                                            continue
                                        _bfs_v.add(_ci)
                                        if _ci == nh2_carbon:
                                            nh2_locant = _d
                                            break
                                        _ca = mol.GetAtomWithIdx(_ci)
                                        if _ca.GetSymbol() == 'C':
                                            for _nb in _ca.GetNeighbors():
                                                _ni = _nb.GetIdx()
                                                if _ni not in _bfs_v and _ni not in chain_set:
                                                    _bfs_q.append((_ni, _d + 1))
                                return f"({nh2_locant}-amino{alkyl})"
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


def _convert_yl_to_ylidene(name: str) -> str:
    """Convert an alkyl substituent name to its ylidene form.

    IUPAC P-31.1.3.1: Substituents attached to a parent by a double bond
    use the suffix -ylidene instead of -yl.

    Examples:
        methyl   -> methylidene
        ethyl    -> ethylidene
        propyl   -> propylidene
        isopropyl -> isopropylidene
        phenyl   -> phenylid (not applicable for ring exocyclic, but handled)
        vinyl    -> vinylidene

    If the name doesn't end in -yl, returns the name with 'idene' appended
    (handles edge cases like retained names).
    """
    if name.endswith('yl'):
        # methyl -> methylidene, ethyl -> ethylidene
        return name + 'idene'
    return name


def _detect_fg_only_prefix(mol, frag_atoms, attach_mol_idx):
    """Detect the IUPAC prefix for an FG-only ring substituent fragment.

    DROP-07 fix: instead of deferring FG-only ring substituents to the global
    FG prefix loop (which lacks ring locant context), detect them here so
    _generate_ring_alkyl_prefixes() can emit them with correct oriented_ring
    locants and proper monosubstituted locant-1 elision.

    Args:
        mol: RDKit Mol object
        frag_atoms: frozenset of atom indices in the fragment (no carbons)
        attach_mol_idx: Ring atom index this fragment is bonded to

    Returns:
        str: IUPAC prefix name (e.g., "fluoro", "hydroxy", "oxo", "amino"),
             or None if unrecognizable (let global FG loop handle).
    """
    if not frag_atoms:
        return None

    # Single-atom fragments (most common: halogens, single O/N/S)
    if len(frag_atoms) == 1:
        idx = next(iter(frag_atoms))
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()

        halogen_map = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
        if sym in halogen_map:
            return halogen_map[sym]

        # Oxygen: check bond type to distinguish hydroxy vs oxo
        if sym == 'O':
            bond = mol.GetBondBetweenAtoms(idx, attach_mol_idx)
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                return 'oxo'
            return 'hydroxy'

        if sym == 'N':
            return 'amino'
        if sym == 'S':
            return 'sulfanyl'

    # Multi-atom FG-only fragments (e.g., -NO2 = nitro, -N3 = azido)
    # These are complex; let the global FG loop handle them with its
    # SMARTS-based detection for correct classification.
    return None


def _generate_ring_alkyl_prefixes(features: Any) -> tuple:
    """
    Generate prefix fragments for ALL substituents on rings (alkyl, heteroatom, compound).

    Uses the unified substituent enumerator (ReplaceCore-based) to extract and
    name every non-hydrogen substituent on the ring parent. FG-only substituents
    (halogens, -OH, -NH2, =O as non-principal) are detected via _detect_fg_only_prefix()
    and emitted with correct oriented_ring locants (DROP-07 fix).

    IUPAC rules for cycloalkane substituent locants:
    - Monosubstituted cycloalkanes: locant 1 is implicit and omitted
      (methylcyclohexane, not 1-methylcyclohexane)
    - Polysubstituted cycloalkanes: all locants included
      (1,2-dimethylcyclohexane)

    Returns:
        Tuple of (List[NameFragment], frozenset): prefix fragments sorted
        alphabetically, and the set of atom indices for FG-only substituents
        that were handled here (to prevent double-emission in the global FG loop).
    """
    from rdkit import Chem

    mol = features.mol
    ring_substituents = features.ring_substituents
    oriented_ring = features.oriented_ring

    # Use the unified enumerator to extract all substituent fragments
    ring_atoms_tuple = tuple(oriented_ring)
    sub_infos = extract_ring_substituents(mol, ring_atoms_tuple, oriented_ring)

    # Collect principal group atoms to skip substituents overlapping with them
    pg_atom_set = set()
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            pg_atom_set.update(match)

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)
    ring_set = set(oriented_ring)

    # DROP-07 fix: track FG-only substituents handled here to prevent
    # double-emission in the global FG prefix loop of _generate_prefixes()
    handled_ring_fg_atoms = set()

    for sub_info in sub_infos:
        # Skip substituents whose atoms overlap with the principal group
        # (e.g., =O when ketone is the principal group -- handled as suffix)
        if pg_atom_set and sub_info.frag_atoms & pg_atom_set:
            continue

        # Check if this is a pure FG-only substituent (no carbon atoms)
        # DROP-07 fix: detect FG type and emit prefix with correct ring locant
        # instead of deferring to the global FG loop (which lacks ring context)
        has_carbon = any(
            mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            for idx in sub_info.frag_atoms
        )
        if not has_carbon:
            fg_prefix = _detect_fg_only_prefix(
                mol, sub_info.frag_atoms, sub_info.attach_mol_idx
            )
            if fg_prefix:
                substituent_groups[fg_prefix].append(sub_info.locant)
                handled_ring_fg_atoms.update(sub_info.frag_atoms)
                logger.debug(
                    "DROP-07 fixed: fg_only_ring_sub prefix=%s locant=%d",
                    fg_prefix, sub_info.locant,
                )
            else:
                # Multi-atom FG-only (e.g., -NO2): let global FG loop handle
                logger.debug(
                    "DROP-07 substituent_defer: reason=complex_fg_only locant=%d",
                    sub_info.locant,
                )
            continue

        # IUPAC P-31.1.3.1: Detect exocyclic double bond attachment.
        # If the bond from the ring atom to the substituent is DOUBLE,
        # the substituent uses -ylidene suffix instead of -yl.
        is_exocyclic_double = False
        attach_idx = sub_info.attach_mol_idx
        for frag_idx in sub_info.frag_atoms:
            bond = mol.GetBondBetweenAtoms(attach_idx, frag_idx)
            if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                is_exocyclic_double = True
                break

        # Classify and name via the unified pipeline
        name = classify_and_name_fragment(mol, sub_info, ring_set, features)
        if name is None:
            # Fallback: try simple carbon-count alkyl naming
            carbon_count = sum(
                1 for idx in sub_info.frag_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )
            if carbon_count > 0:
                from .naming_utils import get_alkyl_name
                try:
                    name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    pass
            if name is None:
                logger.warning(
                    "DROP-08 substituent_skip: reason=unnameable_ring_fragment locant=%d",
                    sub_info.locant,
                )
                continue

        # IUPAC P-31.1.3.1: Convert -yl to -ylidene for exocyclic double bonds.
        # =CH2 -> methylidene, =CHCH3 -> ethylidene, =C(CH3)2 -> propan-2-ylidene
        if is_exocyclic_double and name:
            name = _convert_yl_to_ylidene(name)

        substituent_groups[name].append(sub_info.locant)

    # INST-01: Atom coverage audit for ring parent naming path
    if logger.isEnabledFor(logging.DEBUG):
        total_heavy = features.mol.GetNumHeavyAtoms()
        parent_count = len(oriented_ring)
        named_count = sum(len(locs) for locs in substituent_groups.values()) if substituent_groups else 0
        coverage = (parent_count + named_count) / max(total_heavy, 1)
        logger.debug(
            "ATOM_COVERAGE: smiles=%s total_heavy=%d parent=%d named_subs=%d coverage=%.2f",
            features.canonical_smiles, total_heavy, parent_count, named_count, coverage,
        )
        # ATOM_COVERAGE_DETAIL: actual heavy atoms in named substituent fragments
        named_atom_count = 0
        for sub_info in sub_infos:
            if sub_info.frag_atoms and not (pg_atom_set and sub_info.frag_atoms & pg_atom_set):
                named_atom_count += len(sub_info.frag_atoms)
        logger.debug(
            "ATOM_COVERAGE_DETAIL: smiles=%s named_atom_count=%d",
            features.canonical_smiles, named_atom_count,
        )

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

        # Determine if this ring is heterocyclic (for locant-1 elision decision)
        _ring_is_heterocyclic = any(
            mol.GetAtomWithIdx(a).GetSymbol() != 'C'
            for a in oriented_ring
        )
        _omit_locant = should_omit_locant_one(
            context="prefix",
            is_ring=True,
            is_heterocyclic=_ring_is_heterocyclic,
            is_monosubstituted=is_monosubstituted,
        )

        if _omit_locant:
            # Monosubstituted carbocyclic ring: omit locant (it's always 1)
            # But complex substituents still need enclosing marks per
            # IUPAC P-14.5.2 (e.g., "(2-methylbut-2-en-1-yl)benzene")
            if is_complex_substituent(name) and not name.startswith('('):
                formatted = f"({name})"
            else:
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

    return prefixes, frozenset(handled_ring_fg_atoms)


def _generate_stereodescriptors(features: Any, atom_to_locant_override: Optional[Dict[int, int]] = None) -> Optional[NameFragment]:
    """
    Generate stereodescriptor prefix using correct IUPAC locants.

    Uses the stereochemistry rules module to collect R/S and E/Z descriptors
    based on the atom_to_locant mapping (from chain/ring orientation).

    For different compound types:
    - Acyclic: uses features.atom_to_locant (from principal chain orientation)
    - Heterocycles: uses features.heterocycle_atom_to_locant
    - Cycloalkanes/cycloalkenes: builds from features.oriented_ring

    When atom_to_locant_override is provided, it takes priority over all
    features.* fields. This allows early-return handlers to pass the correct
    locant map for the structure they are naming (which may differ from
    features.principal_ring).

    Args:
        features: MolecularFeatures object with stereocenters and/or double_bond_stereo
        atom_to_locant_override: Optional explicit locant map. When provided,
            bypasses the features.* priority chain entirely.

    Returns:
        NameFragment with stereodescriptor prefix like "(2R)-" or "(2E,3R)-",
        or None if no stereodescriptors.
    """
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Need either stereocenters or double_bond_stereo
    if not features.stereocenters and not getattr(features, 'double_bond_stereo', None):
        return None

    mol = features.mol

    # When an explicit override is provided, use it directly instead of
    # reading from features.* fields. This allows early-return handlers
    # to pass the correct locant map for the structure they are naming
    # (which may differ from features.principal_ring).
    if atom_to_locant_override:
        atom_to_locant = atom_to_locant_override
    else:
        # Existing priority chain (unchanged):
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

    # Collect stereodescriptors with proper locants.
    # Enable near-parent E/Z detection for top-level naming only:
    # substituent E/Z bonds one hop from the parent ring/chain should
    # be included in the stereo block per IUPAC P-93.5.2.
    descriptors = collect_stereodescriptors(
        mol, atom_to_locant, include_near_parent_ez=True
    )

    if not descriptors:
        return None

    # Format as "(2R,3S)-" etc
    text = format_stereodescriptor_string(descriptors)

    return NameFragment(text=text, fragment_type="stereo")


def _inject_stereo_if_missing(features: Any, name: str, atom_to_locant: Optional[Dict[int, int]] = None) -> str:
    """Prepend stereodescriptor prefix to a name if stereocenters exist but aren't represented.

    Used after early-return handlers that bypass _generate_stereodescriptors().
    Per IUPAC P-91, stereodescriptors are detachable prefixes placed before the name.

    Args:
        features: MolecularFeatures with stereocenters and/or double_bond_stereo
        name: The generated name from a handler (may or may not have stereo already)
        atom_to_locant: Optional explicit locant map to forward to
            _generate_stereodescriptors. When provided, bypasses the features.*
            priority chain. Pass None to use default features.* resolution.

    Returns:
        Name with stereo prefix prepended if needed, or original name if:
        - No stereocenters/double bond stereo in features
        - Name already has a stereo prefix
        - No atom_to_locant mapping available for locant resolution
    """
    import re

    if not features.stereocenters and not getattr(features, 'double_bond_stereo', None):
        return name
    if not name or name == 'unknown':
        return name

    # Check if name already has a stereo prefix: starts with (R)-, (2R)-, (E)-, (4E)-,
    # (2R,3S)-, etc.  Allow zero or more digits before each stereo letter; require
    # closing ")- " to avoid false matches with parenthesized substituent names
    # like "(oxan-2-yl)oxy".  The pattern matches the full stereo descriptor block:
    # one or more comma-separated [digits][stereo-letter] groups inside parentheses.
    if re.match(r'\(\d*[RSrsEZez](,\d*[RSrsEZez])*\)-', name):
        return name

    # Generate stereo prefix using the centralized function
    stereo_frag = _generate_stereodescriptors(features, atom_to_locant_override=atom_to_locant)
    if stereo_frag and stereo_frag.text:
        return f"{stereo_frag.text}{name}"
    return name


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

    The infix is assembled structurally so that hyphens are inserted only
    where needed -- no post-hoc band-aid cleanup required.

    IUPAC P-31.1.3.4: when total unsaturation locant count >= 2, prefix
    with 'a' for euphony (e.g., 'a-1,3-dien' not '-1,3-dien').

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
        >>> _build_unsaturation_infix([1, 3], [])
        'a-1,3-dien'
        >>> _build_unsaturation_infix([], [1, 3])
        'a-1,3-diyn'
        >>> _build_unsaturation_infix([1, 3], [5])
        'a-1,3-dien-5-yn'
    """
    num_double = len(double_locants)
    num_triple = len(triple_locants)

    if num_double == 0 and num_triple == 0:
        return "an"  # Saturated

    # Determine if the 'a' euphonic connector is needed (IUPAC P-31.1.3.4):
    # used when total unsaturation locants >= 2 (diene, diyne, enyne etc.)
    total_locants = num_double + num_triple
    needs_a = (total_locants >= 2) and (num_double > 1 or num_triple > 1)

    # Build segments: each is a tuple (locant_str, multiplier, bond_type)
    segments = []

    if num_double > 0:
        loc_str = ",".join(str(loc) for loc in double_locants)
        multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double)) if num_double > 1 else ""
        segments.append((loc_str, multiplier, "en"))

    if num_triple > 0:
        loc_str = ",".join(str(loc) for loc in triple_locants)
        multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple)) if num_triple > 1 else ""
        segments.append((loc_str, multiplier, "yn"))

    # Assemble: "a" (if needed) then each segment as "-locants-[mult]bond"
    # joined with hyphens between them.
    result_parts = []
    if needs_a:
        result_parts.append("a")

    for loc_str, mult, bond in segments:
        result_parts.append(loc_str)
        result_parts.append(f"{mult}{bond}")

    # Join all parts with hyphens; prepend leading hyphen if no 'a' connector
    result = "-".join(result_parts)
    if not needs_a:
        result = "-" + result

    return result


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

    The name is assembled structurally so that hyphens are inserted only
    where needed -- no post-hoc band-aid cleanup required.

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

    # Centralized bond locant elision: ethene/ethyne (2-carbon) and mono-cycloalkenes
    _chain_len = 2 if stem == "eth" else 0
    omit_bond_locant = should_omit_locant_one(
        context="bond",
        chain_length=_chain_len,
        is_ring=is_cyclic,
        is_monosubstituted=(num_double == 1 and num_triple == 0),
    )

    # Determine if the 'a' euphonic connector is needed (IUPAC P-31.1.3.4):
    # used when multiple bonds have multiplied locants (diene, diyne, etc.)
    needs_a = (num_double > 1) or (num_triple > 1 and num_double == 0)

    # Build segments as structured data, then join cleanly
    # Each segment: (locant_str_or_None, multiplier, bond_suffix)
    segments = []

    if num_double > 0:
        if num_double == 1:
            if omit_bond_locant:
                segments.append((None, "", "en"))
            else:
                loc_str = ",".join(str(loc) for loc in double_locants)
                segments.append((loc_str, "", "en"))
        else:
            loc_str = ",".join(str(loc) for loc in double_locants)
            mult = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            segments.append((loc_str, mult, "en"))

    if num_triple > 0:
        if num_triple == 1:
            if omit_bond_locant and num_double == 0:
                segments.append((None, "", "yn"))
            else:
                loc_str = ",".join(str(loc) for loc in triple_locants)
                segments.append((loc_str, "", "yn"))
        else:
            loc_str = ",".join(str(loc) for loc in triple_locants)
            mult = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            segments.append((loc_str, mult, "yn"))

    # Assemble: stem [+ "a" if needed] [+ segments joined with hyphens] + "e"
    result = stem
    if needs_a:
        result += "a"

    for loc_str, mult, bond in segments:
        if loc_str is not None:
            result += f"-{loc_str}-{mult}{bond}"
        else:
            # No locant: directly append bond suffix (ethene, cyclohexene)
            result += f"{mult}{bond}"

    # Final terminal 'e' for the hydrocarbon ending
    result += "e"

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
            if (last_char.isalpha() or last_char in (')', ']')) and first_char.isdigit():
                result += "-"
            elif (last_char.isalpha() or last_char in (')', ']')) and first_char == 'N':
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

    if (prefix_str[-1].isalpha() or prefix_str[-1] in (')', ']')) and name[0].isdigit():
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


def _warn_if_bare_oxy(name: str) -> bool:
    """Defensive check: detect bare 'oxy' prefix in a generated name.

    Returns True if the name contains a standalone 'oxy' token that is
    NOT part of a qualified compound word (methoxy, ethoxy, oxybis, etc.).
    This is a safety net -- the primary fix is at the generation point.

    Args:
        name: Generated IUPAC name string.

    Returns:
        True if bare 'oxy' detected, False otherwise.
    """
    import re
    if not name:
        return False
    # Split on hyphens, spaces, commas, parentheses
    tokens = re.split(r'[-\s,()]', name)
    for tok in tokens:
        if tok == 'oxy':
            logger.warning(
                "BARE_OXY_DETECTED: name=%r contains standalone 'oxy' prefix",
                name,
            )
            return True
    return False
