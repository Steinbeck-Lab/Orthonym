"""
Name assembly - combining fragments into complete IUPAC names.

Assembly order:
1. Stereodescriptors (R/S, E/Z) at start in parentheses
2. Locanted prefixes (substituents, alphabetized)
3. Parent name (with unsaturation modifiers)
4. Locanted suffix (principal group)
"""

import logging
from typing import Optional, List, Dict, Any, Set, Tuple
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
    _join_multiplied_suffix,
    SIMPLE_MULTIPLIERS,
    COMPLEX_MULTIPLIERS,
    TERMINAL_FG_TYPES,
    BRANCH_HANDLED_FGS,
)
# Phase 179 (D-03): the name-composition grammar primitives were lifted VERBATIM
# into the leaf module composition_primitives.py (single source of truth, shared
# with the name-tree serializer). Re-export them here under their original names
# so existing composer call sites resolve unchanged.
from .composition_primitives import (
    _estimate_parent_size_from_name,
    _build_unsaturation_infix,
    _build_hydrocarbon_name,
    _join_prefixes,
    _join_prefix_to_name,
)
from ..data.chain_names import get_chain_prefix
from .substituent_naming import name_substituent_fragment, _is_linear_alkyl, _name_aryl_methyl_ether
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

# Phase 160.2 Plan-02-01: helpers lifted to handlers/_handler_shared.py per
# CONTEXT D-02 (two-step helpers-first lift). composer.py keeps these
# local-name shim re-exports to preserve in-file call sites byte-identical
# per CONTEXT D-13 forbidden-boundary preservation. The handler file uses
# lazy imports inside its function bodies for any composer.py-private
# symbols (NameFragment, _generate_alkyl_prefixes, etc.) so this module-load-
# time import does NOT create an import cycle.
from .handlers._handler_shared import (  # noqa: F401  # Local-name shim per CONTEXT D-13
    _generate_chain_parent,        # was def at composer.py:3785-3818
    _generate_ring_parent,         # was def at composer.py:3821-3893
    _generate_suffix,              # was def at composer.py:3934-4076
    _generate_prefixes,            # was def at composer.py:4079-4282
    _generate_stereodescriptors,   # was def at composer.py:6634-6704
    _assemble_fragments,           # was def at composer.py:6799-6954
)

# Ion/radical naming imports - deferred to avoid circular imports
# These are imported inside functions that need them

# Complex ring system imports
from ..rules.bicyclo import is_bicyclo_system, name_bicyclo_system
from ..rules.spiro import is_spiro_system, name_spiro_system
from ..rules.fused_rings import classify_fused_system, name_fused_heterocycle, name_ortho_fused_bicyclic

# Polycyclic system imports (Phase 16)
from ..rules.polycyclic import (
    name_polycyclic_complete,
    is_polycyclic_system,
    _detect_ring_functional_groups,
)
from ..rules.bridged_fused import detect_bridged_fused, name_bridged_fused_system
from ..errors import OrthonymLimitError  # G0 fail-closed refusal (DD7 S1)

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

    # WSD-02 (RING-04): a partially-saturated fused carbocycle's parent name
    # (e.g. `1,2,3,4-tetrahydronaphthalene`) covers the ENTIRE fused ring system.
    # oriented_ring / principal_ring is only ONE SSSR ring, so leaving it as the
    # parent makes substituent discovery mis-trace the REST of the fused system
    # (tetralin's aromatic half) as a phantom alkyl (`4-butyl-`). Use the full
    # ring-atom union as the parent so only TRUE exocyclic substituents enrich.
    # Hard invariant (D-04): a fused ring atom never becomes a chain substituent.
    if handler_id == "partial_sat" and getattr(features, 'mol', None) is not None:
        ring_union = set()
        for _r in features.mol.GetRingInfo().AtomRings():
            ring_union.update(_r)
        if ring_union:
            parent_atoms = ring_union

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

    enriched_name = _join_prefix_to_name(prefix_str, base_name)  # L2 (P-16.3.3): hyphen before digit-leading parent

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
    # Phase 163 Tier FRN-A: chalcogen acids (parallel to thioic_*_acid) — additive per CONTEXT D-08
    "selenoic_Se_acid",    # Always at chain end (locant 1)
    "selenoic_O_acid",     # Always at chain end (locant 1)
    "diselenoic_acid",     # Always at chain end (locant 1)
    "telluroic_Te_acid",   # Always at chain end (locant 1)
    "telluroic_O_acid",    # Always at chain end (locant 1)
    "ditelluroic_acid",    # Always at chain end (locant 1)
    # Phase 163 Tier FRN-B: chalcogen amides (parallel to primary/secondary/tertiary_amide)
    "thioamide",           # Always at chain end (locant 1)
    "selenoamide",         # Always at chain end (locant 1)
    "telluroamide",        # Always at chain end (locant 1)
    # Phase 163 Tier FRN-C: chalcogen aldehydes (parallel to aldehyde; "-thial"/"-selenal"/"-tellural")
    "thioaldehyde",        # Always at chain end (locant 1)
    "selenoaldehyde",      # Always at chain end (locant 1)
    "telluroaldehyde",     # Always at chain end (locant 1)
    "carbamic_acid",    # Retained name, terminal (locant 1)
    # D-FOLLOWON item 8 (P-66.4.1): amidine/imidamide characteristic C is terminal.
    "amidine",          # Always at chain end (locant 1)
    # R8a (P-66.3.1.1): hydrazide characteristic C is always chain-terminal.
    "hydrazide",        # Always at chain end (locant 1)
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
    push_pool(features)  # WR-06: bind Phase 168 controller flag/oracle at pool construction
    try:
        return _assemble_name_impl(features, style, _composing_ion)
    finally:
        pop_pool()


def _serializer_flip_or_name(result: Any, style: str) -> str:
    """Phase 179 (WSA-03) production flip seam.

    For a ``NamingResult`` whose tree class is in
    ``SERIALIZER_PRODUCTION_CLASSES`` (general_acyclic), return the name-tree
    serializer's explicit-field rendering — BUT only when it is byte-identical
    to the handler's legacy ``name`` (the staleness guard from
    ``namer.py:1183-1188``). On any mismatch, serializer exception, missing
    tree, or non-flipped class, return the legacy ``name`` unchanged.

    Production output is therefore byte-identical BY CONSTRUCTION; the flip only
    changes WHICH path produced an identical string for the flipped classes (the
    name-tree serializer becomes the production composition site). The
    byte-identical guard also catches degenerate empty-name rows: a divergent
    serialization is dropped in favour of the legacy ``name``.
    """
    name = result.name
    tree = getattr(result, 'tree', None)
    if tree is None:
        return name
    from .name_tree_to_string import (
        SERIALIZER_PRODUCTION_CLASSES,
        name_tree_to_string,
    )
    if getattr(tree, 'class_id', '') not in SERIALIZER_PRODUCTION_CLASSES:
        return name
    try:
        serialized = name_tree_to_string(tree, style=style)
    except Exception as exc:  # fail SAFE, never SILENT (mirrors the WR-05 pattern)
        logger.warning(
            "Phase 179 serializer flip failed on %s: %s; falling back to legacy name",
            getattr(tree, 'class_id', '?'), exc,
        )
        return name
    if serialized == name:
        return serialized
    logger.debug(
        "Phase 179 serializer flip diverged on %s (serialized=%r != name=%r); keeping legacy",
        getattr(tree, 'class_id', '?'), serialized, name,
    )
    return name


def _emit_ion_with_tree(ion_name: str) -> str:
    """Phase 165 SC-3 / Pitfall 4: the ion/salt/radical pre-pool bypass returns
    before dispatch_inner and never writes the name_with_tree capture slot. Write
    a coarse NameTreeNode (str fragment_legacy -> verbatim round-trip) so
    ``name_with_tree`` / ``--dump-tree`` work for ANY SMILES including ions. The
    production string path is unchanged (returns ion_name as before). ion_dispatch
    stays EXCLUDED from the per-handler contract suite (it never runs via
    dispatch_inner). Returns ion_name unchanged.
    """
    if ion_name:
        from ..namer import _name_with_tree_capture
        from .name_tree import NameTreeNode, NamingResult
        _slot = _name_with_tree_capture.get()
        if _slot is not None:
            _slot["naming"] = NamingResult(
                name=ion_name,
                tree=NameTreeNode(
                    parent_stem=ion_name, class_id="ion_path",
                    iupac_section_cite="P-73", fragment_legacy=ion_name,
                ),
                atom_to_locant_hint=None,
            )
    return ion_name


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
    # WR-06: bind the Phase 168 controller flag/oracle off `features` at construction (this
    # clear_pool replaces the push_pool() pool from the wrapper, so it owns the active pool).
    clear_pool(features)

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
        return _emit_ion_with_tree(assemble_ion_name(features, features.mol, style))

    # ASML-10 by-design: Radical handler uses specialized naming (IUPAC P-68).
    # Stereo: handled by assemble_ion_name()
    if species_type == 'radical':
        return _emit_ion_with_tree(assemble_ion_name(features, features.mol, style))

    # ASML-10 by-design: Single-component ion uses aspect composition or
    # assemble_ion_name(). Ion naming follows IUPAC P-73 functional class
    # naming; no detachable prefixes needed.
    # Stereo: handled by assemble_ion_name() / aspect composition
    if species_type == 'ion' and not _composing_ion:
        composed_name = _try_ion_aspect_composition(features, style)
        if composed_name:
            return _emit_ion_with_tree(composed_name)
        # Fallback to existing ion naming if composition fails
        return _emit_ion_with_tree(assemble_ion_name(features, features.mol, style))

    # =========================================================================
    # DD1 Fix 3 (B3): PSEUDOKETONE for acyl-on-ring-N "hidden amides"
    # (P-66.1.3 / P-64.3.2 / P-66.1.4.3). An acyl group on a ring-system N is a
    # ketone (the C=O carbon is the principal group, the ring is an N-yl
    # substituent), NOT an amide. The legacy amide path drops the C=O entirely
    # (is_ring_attached_amide is False, no -carboxamide ring fire) -> bare
    # 'piperidine'. Detect on the original graph and emit the pseudoketone.
    # Returns None for ordinary acyclic amides, lactams, or decorated acyl chains.
    # =========================================================================
    from ..rules.pseudoketones import name_pseudoketone as _name_pseudoketone
    _pk_name = _name_pseudoketone(features, style)
    if _pk_name:
        return _emit_ion_with_tree(_pk_name)

    # =========================================================================
    # DD1 Fix 2 (B2): ADDED-CARBON MULTI-SUFFIX PARENT (P-65.1.1.1 /
    # P-66.1.1.1.1.2 / P-66.5.1.1.2). When >=3 carboxylic-acid / carboxamide /
    # carbonitrile groups sit on a clean acyclic carbon skeleton they cannot all
    # be chain-terminal suffixes, so the PIN is the parent hydride (carbonyl/
    # nitrile carbons removed) + a multiplied 'carbo*' suffix. This MUST run
    # before the legacy parent-selection/clamp path, which walks through the
    # carbonyl/nitrile carbons and drops the third group. The namer returns None
    # for any structure outside the clean-acyclic-parent class (decorated parent,
    # ring-attached, n<3) so control falls through unchanged.
    # =========================================================================
    from ..rules.added_carbon_parent import (
        requires_added_carbon_suffix as _req_added_carbon,
        name_added_carbon_parent as _name_added_carbon,
    )
    if _req_added_carbon(
        features.mol,
        getattr(features, "principal_group", None),
        getattr(features, "principal_group_atoms", None),
    ):
        _ac_name = _name_added_carbon(features, style)
        if _ac_name:
            return _emit_ion_with_tree(_ac_name)

    # =========================================================================
    # PHASE 160 INNER DISPATCH (DECOMP-01 + CONTEXT D-08 + D-10)
    # =========================================================================
    # The inner-dispatch table at orthonym.assembly.inner_dispatch is
    # populated by per-handler atomic commits 02-01..02-29 (Plan-02) +
    # 03-01..03-09 (Plan-03). Each registered handler's predicate is
    # checked in priority order (first-match-wins); on match, the handler
    # produces a NamingResult containing the FINAL name string (per
    # CONTEXT D-05 + DECOMP-03 byte-identical contract).
    #
    # CRITICAL byte-identical preservation rule (CONTEXT D-13 layering):
    # the handler is responsible for ANY post-naming processing required
    # to match the legacy inline-branch behavior. Handlers that originally
    # called ``_inject_stereo_if_missing(...)`` in their inline branch MUST
    # call it themselves and place the result in ``NamingResult.name``.
    # Handlers that originally returned ``pool.best().name`` DIRECTLY
    # (e.g., polyfunctional, multi_ester, ester at composer.py:863, 943,
    # 980, 1043, 1070, 1096) MUST NOT inject stereo — those handlers
    # internally produce stereo-included names per their rule modules.
    # This caller treats ``NamingResult.name`` as the FINAL string —
    # no post-processing.
    #
    # Plan-02 wave: dispatch_inner returns None on no-match because the
    # catch-all general_acyclic handler is registered in Plan-03 commit
    # 03-09. When dispatch_inner returns None, control falls through to
    # the inline mid-tier + root branches still present at composer.py:
    # 1501-1903 (per CONTEXT D-01 + D-08 Plan-02 boundary).
    #
    # Byte-identical lock per CONTEXT D-21 (DECOMP-03): the per-handler
    # atomic commit canary delta (
    # --mode delta`) gates every commit at zero diff vs the frozen baseline
    # `tests/canary/canary_pre_decomp_160.csv`.
    # =========================================================================
    # Phase 160.1 D-18 / ADR-19-04: dispatch_inner now invokes handlers
    # internally and retries the next-priority entry on gate-fail (handler
    # returning None). The caller collapses to a single return; the
    # matched-but-None fall-through branch DELETED per D-18. If no
    # handler succeeds at this commit, dispatch_inner returns None and
    # control falls through to the inline cascade below — Plan-03-01..03-04
    # remove those inline cascades, and general_acyclic@99999 (Plan-03-03)
    # becomes the catch-all that guarantees a non-None result.
    # D1 (P-66.4.1.2 / P-66.4.1.3.1): acyclic N-/N'-substituted amidine. Must run
    # BEFORE dispatch_inner because the general_acyclic catch-all otherwise
    # mis-expresses the amino nitrogen as a '(methylamino)' chain prefix
    # (OPSIN-unparseable -> SELF-01 -> unknown). _assemble_amidine_name is
    # fail-closed: it returns None for unsubstituted amidines (general path names
    # those correctly), for ring-attached amidines (benzene/cyclohexane ring
    # namers own those), and for anything it cannot name cleanly, so this branch
    # NEVER produces a wrong name — it only upgrades unknown -> correct.
    if (features.principal_group == 'amidine'
            and features.principal_chain
            and not getattr(features, 'is_cyclic', False)):
        _pg_count = len(features.principal_group_atoms) if features.principal_group_atoms else 1
        # AM-5: also accept a conjoined diamidine (two matches sharing one N) —
        # the shared-N amidine is named as an N-imidoyl substituent on the other.
        if _pg_count == 1 or (
            _pg_count == 2
            and _amidines_share_one_n(features.mol, features.principal_group_atoms)
        ):
            _amidine_name = _assemble_amidine_name(features, style)
            if _amidine_name:
                pool = get_current_pool()
                pool.add(_amidine_name, "amidine", features)
                return pool.best().name

    # W2E-D4 (P-66.4.1.4.2 / P-16.9.1): N-substituted GEMINAL ring
    # dicarboximidamide -> per-group primed + superscript italic-N locants
    # (N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide). Runs BEFORE
    # dispatch_inner because the general_acyclic catch-all names the bare
    # 'cyclohexane-1,1-dicarboximidamide' base and DROPS the N-substituents
    # (a different molecule; SELF-01 then -> unknown). The assembler is
    # fail-closed: it returns None for anything outside the two-geminal-amidine
    # ring class (unsubstituted, mixed amide/imidamide, non-geminal, aryl,
    # non-nameable N-sub, >2 groups), so this branch only upgrades unknown ->
    # correct and NEVER emits a wrong name.
    if (features.principal_group == 'amidine'
            and getattr(features, 'is_cyclic', False)
            and len(features.principal_group_atoms or []) == 2):
        _gem_name = _assemble_geminal_dicarboximidamide_name(features, style)
        if _gem_name:
            pool = get_current_pool()
            pool.add(_gem_name, "amidine", features)
            return pool.best().name

    # Wave2 T3d (P-66.3): acyclic N/N'-substituted (thio)hydrazide, or a
    # hydrazinecarboxylic acid. Runs BEFORE dispatch_inner for the same reason
    # as the amidine branch (the general path mis-expresses the -NH-NH2 as a
    # 'hydrazinyl'/'hydrazinecarbonyl' chain prefix). Fail-closed: returns None
    # for the unsubstituted hydrazide (general path already correct) and for
    # anything it cannot name cleanly. The hydrazinecarboxylic case perceives as
    # carboxylic_acid (senior) + hydrazide co-located on the same C.
    _hz_trigger = (
        features.principal_group in ('hydrazide', 'thiohydrazide')
        or (features.principal_group == 'carboxylic_acid'
            and 'hydrazide' in getattr(features, 'functional_groups', {}))
    )
    if _hz_trigger:
        # Ring parents route through the ring assembler (recursion on the
        # des-N-substituted base: N'-methylbenzohydrazide,
        # N-methylcyclohexanecarbohydrazide); acyclic parents keep the T3d
        # chain assembler. Both are fail-closed (None -> general path).
        if getattr(features, 'is_cyclic', False):
            _hz_name = _assemble_ring_hydrazide_name(features, style)
        else:
            _hz_name = _assemble_hydrazide_name(features, style)
        if _hz_name:
            pool = get_current_pool()
            pool.add(_hz_name, "hydrazide", features)
            return pool.best().name

    from .inner_dispatch import dispatch_inner
    _inner_result = dispatch_inner(features, mol=features.mol, style=style)
    if _inner_result is not None:
        # CR-04 part B + W7: capture the inner-dispatch NamingResult
        # for name_with_tree() consumers via contextvars.ContextVar
        # (PEP 567). The slot is installed by Orthonym.name_with_tree()
        # before invoking self.name(); when absent (default=None), this
        # is a no-op for the regular Orthonym.name() path. ContextVar
        # is thread-local AND asyncio-task-local — safe under
        # concurrent invocation from multiple threads / tasks.
        from ..namer import _name_with_tree_capture
        _slot = _name_with_tree_capture.get()
        if _slot is not None:
            _slot["naming"] = _inner_result.result
        # Per CONTEXT D-13 + the comment above: the handler's
        # NamingResult.name is the FINAL byte-identical string. No
        # post-processing here. Handlers that need _inject_stereo_if_missing
        # call it themselves (oxime, hydrazone, n_oxide, isocyanate,
        # isothiocyanate, carbamic_acid, carbamate, urea, guanidine,
        # boronic_acid, acid_halide, anhydride, lactone, lactam, sulfoxide,
        # sulfone, thioether, phosphine_oxide, phosphate_ester, phosphine,
        # phosphinic_acid, ring_assembly, polycyclic, partial_sat);
        # handlers that do NOT inject stereo are polyfunctional,
        # multi_ester, ester, simple_molecule (final string is direct
        # from pool.best().name in their inline branches).
        # Phase 179 (WSA-03): route flipped classes through the name-tree
        # serializer (byte-identical fallback to .name otherwise).
        return _serializer_flip_or_name(_inner_result.result, style)

    # [Phase 160 + 160.1 + 160.2: 32 handler classes dispatched via inner_dispatch
    #  (oxime, hydrazone, n_oxide, isocyanate, isothiocyanate, carbamic_acid,
    #  carbamate, urea, guanidine, boronic_acid, acid_halide, anhydride, lactone,
    #  lactam, ring_ester, sulfoxide, sulfone, thioether, phosphine_oxide,
    #  phosphate_ester, phosphine, phosphinic_acid, ring_assembly, polycyclic,
    #  partial_sat, simple_molecule, ring_nitrile, amide, amine, ester_family,
    #  tier_a_ring, ion_dispatch) + general_acyclic@99999 catch-all per
    #  Plan-02-03. Inline amide+amine branches below are CASE B carryover per
    #  AP-160.2-06 — kept because handler-side _is_amide/_is_amine predicates
    #  are stricter than the inline guards (polyfunctional + Tier-A mutexes).
    #  Full closure → Phase 160.3 CR-X carryover.]


    # ASML-10 complete: Amide handler uses _assemble_amide_name() which includes
    # N-substituent prefixes and chain/ring substituent discovery.
    # Multi-amide compounds (diamide, triamide) fall through to normal suffix path
    # so that the multiplier prefix (di-, tri-) is correctly applied.
    # Stereo: handled within _assemble_amide_name() -- unsaturated path calls
    # _generate_stereodescriptors(); ring/saturated paths inject stereo at return.
    if features.principal_group in (
        'primary_amide', 'secondary_amide', 'tertiary_amide',
        # Phase 163 chalcogen amides — single-permissive thio/seleno/telluro PG
        # per AUDIT § 2.2 (no 3-way primary/secondary/tertiary split). The same
        # _assemble_amide_name pipeline handles N-substitution rendering; the
        # suffix_form lookup happens inside via _CHALCOGEN_AMIDE_SUFFIX_FORMS.
        'thioamide', 'selenoamide', 'telluroamide',
    ):
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
            _best = pool.best()
            # PHASE 168 STAGE B CUTOVER (CONTEXT D-08): flag ON -> render from the rewritten tree;
            # flag OFF -> fall through to pool.best().name (byte-identical Stage A).
            if (getattr(pool, '_enable_triviality_controller', False) and _best is not None
                    and getattr(_best, 'tree_rewritten', None) is not None):
                try:
                    from .name_tree_to_string import name_tree_to_string
                    _rw = name_tree_to_string(_best.tree_rewritten, style=style)
                    if _rw:
                        return _rw
                except Exception as exc:
                    # WR-05 (code review 2026-05-30): fail SAFE (fall back to the systematic
                    # name) but never SILENT — a bare except-pass hid genuine controller/
                    # serializer logic errors and made a broken controller look like a clean
                    # no-op. Correctness against malformed-but-valid strings comes from the
                    # RT-gate (CR-03) + _build_rewrite (CR-01/CR-02), not from swallowing here.
                    logger.warning(
                        "Phase 168 Stage-B re-render failed on amide surface: %s; "
                        "falling back to systematic name", exc,
                    )
            return pool.best().name

    # ASML-10 complete: Amine handler uses _assemble_amine_name() which adds
    # N-alkyl prefixes and generates chain/ring substituent prefixes.
    # Stereo: handled by _assemble_amine_name() (calls _generate_stereodescriptors)
    # Wave-2 completion C2: 'primary_amine' joins the gate — the RC-4 amine
    # union can leave a mixed diamine classified primary; _assemble_amine_name
    # is a verified no-op (None -> general fallback) for amines without
    # N-substituents, so pure-primary molecules are untouched.
    if features.principal_group in ('primary_amine', 'secondary_amine',
                                    'tertiary_amine'):
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
            _best = pool.best()
            # PHASE 168 STAGE B CUTOVER (CONTEXT D-08): flag ON -> render from the rewritten tree;
            # flag OFF -> fall through to pool.best().name (byte-identical Stage A).
            if (getattr(pool, '_enable_triviality_controller', False) and _best is not None
                    and getattr(_best, 'tree_rewritten', None) is not None):
                try:
                    from .name_tree_to_string import name_tree_to_string
                    _rw = name_tree_to_string(_best.tree_rewritten, style=style)
                    if _rw:
                        return _rw
                except Exception as exc:
                    # WR-05 (code review 2026-05-30): fail SAFE but never SILENT (see amide).
                    logger.warning(
                        "Phase 168 Stage-B re-render failed on amine surface: %s; "
                        "falling back to systematic name", exc,
                    )
            return pool.best().name

    # Phase 160.2 Plan-02-04: chain-fallback section (composer.py:935-1082 in
    # pre-amendment line numbers) DELETED — extracted to handlers/general_acyclic.py
    # per CONTEXT D-02 + D-04 + AP-160.2-06 CASE B carryover. The cycloalkane
    # stereo backstop (composer.py:1057-1097) DELETED too — general_acyclic's
    # body emits stereo natively via _generate_stereodescriptors per the
    # verbatim lift; the backstop is dead code post-extraction (empirical
    # verification: 199 of 200 sampled canary rows route via dispatch_inner;
    # 1 routes via inline amide; 0 reach the legacy chain-fallback). When
    # _is_general_acyclic returns False (single-amide / amine cases per
    # Plan-02-03 AP-160.2-06 refinement) AND inline amide/amine branches above
    # DIDN'T fire (e.g., _assemble_amine_name returns falsy), invoke
    # general_acyclic directly as the safety net.
    from .handlers.general_acyclic import name_general_acyclic
    _fallback = name_general_acyclic(features, mol=features.mol, style=style)
    if _fallback is not None:
        # PHASE 168 STAGE B CUTOVER (CONTEXT D-08): _fallback bypasses pool.add (no POST-HOC attach),
        # so apply the controller directly to _fallback.tree when the flag is ON; flag-OFF falls
        # through to _fallback.name (byte-identical Stage A).
        if (getattr(features, '_enable_triviality_controller', False)
                and getattr(_fallback, 'tree', None) is not None):
            try:
                from .retained_substitution import apply_triviality_controller
                from .name_tree_to_string import name_tree_to_string
                _rw_tree = apply_triviality_controller(
                    _fallback.tree, features.mol,
                    getattr(features, 'principal_group', None),
                    opsin_oracle=getattr(features, '_triv_oracle', None), enabled=True,
                )
                _rw = name_tree_to_string(_rw_tree, style=style)
                if _rw:
                    return _rw
            except Exception as exc:
                # WR-05 (code review 2026-05-30): fail SAFE but never SILENT (see amide).
                logger.warning(
                    "Phase 168 Stage-B re-render failed on _fallback surface: %s; "
                    "falling back to systematic name", exc,
                )
        # Phase 179 (WSA-03): the general_acyclic safety-net seam — route flipped
        # classes through the serializer (byte-identical fallback otherwise). The
        # triviality-controller branch above runs first when its flag is ON.
        return _serializer_flip_or_name(_fallback, style)
    # Truly empty result — degenerate molecule. Return empty string preserving
    # pre-amendment behavior (chain-fallback section returned pool.best().name
    # which was '' when nothing fired).
    return ""


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

    # Merge C=N stereo descriptor into the parent name if captured.
    # Phase 177 STEREO-04 (D-11): derive the C=N carbon's locant from the parent's
    # atom_to_locant (P-68.3.1.1), NOT from a ketone-only `(\d+)-on` regex.  The old
    # regex never matched aldehyde (`-al`) parents, so `C/C=N/O` shipped
    # `acetaldehyde oxime` with the E/Z descriptor dropped.  c_idx (the C=N carbon)
    # and cn_stereo_tag (the E/Z label) are captured above and reused unchanged.
    if cn_stereo_tag and parent_name:
        import re
        atl = getattr(features, 'atom_to_locant', None) or {}
        cn_locant = atl.get(c_idx)
        # Mononuclear / terminal C=N at locant 1 takes the BARE descriptor (no
        # locant) per P-68.3.1.1 — matches OPSIN-verified `(E)-acetaldehyde oxime`.
        # If the locant cannot be resolved we likewise omit it (D-09: a missing
        # locant beats a wrong one) rather than fabricating the old ketone parse.
        if cn_locant == 1 or cn_locant is None:
            cn_desc = cn_stereo_tag
        else:
            cn_desc = f"{cn_locant}{cn_stereo_tag}"

        # If parent already has a stereo prefix like (6E), merge into the block.
        stereo_match = re.match(r'\(([^)]+)\)-(.*)', parent_name)
        if stereo_match:
            existing_stereo = stereo_match.group(1)
            rest = stereo_match.group(2)
            all_descs = existing_stereo.split(',')
            all_descs.append(cn_desc)
            # Sort by leading locant number (bare descriptors sort last).
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
    # W4-I4 (P-74.2.2.1.9 / P-62.5): imine (aldo-/keto-nitrone) N-oxide, an sp2 N+
    # double-bonded to C and single-bonded to a terminal O-. Named by functional
    # class as '<imine> N-oxide' (CH3)2C=N+(CH3)-O- -> N-methylpropan-2-imine N-oxide.
    imine_pat = Chem.MolFromSmarts('[NX2,NX3;+](=[#6])[OX1-]')

    aromatic_matches = mol.GetSubstructMatches(aromatic_pat) if aromatic_pat else ()
    aliphatic_matches = mol.GetSubstructMatches(aliphatic_pat) if aliphatic_pat else ()
    imine_matches = mol.GetSubstructMatches(imine_pat) if imine_pat else ()

    if not aromatic_matches and not aliphatic_matches and not imine_matches:
        return None

    if aromatic_matches:
        return _name_aromatic_n_oxide(mol, aromatic_matches)
    elif aliphatic_matches:
        return _name_aliphatic_n_oxide(mol, aliphatic_matches)
    else:
        return _name_imine_n_oxide(mol, imine_matches)


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


def _name_imine_n_oxide(mol, matches) -> Optional[str]:
    """W4-I4 (P-74.2.2.1.9 / P-62.5): name an imine (nitrone) N-oxide as
    '<imine> N-oxide' (functional class, the PIN). Strategy: neutralize the N+ and
    delete the O-, name the base neutral imine recursively, then append ' N-oxide'::

        (CH3)2C=N+(CH3)-O-  ->  N-methylpropan-2-imine N-oxide  (BB 26626)
        CH2=N+(Cl)-O-       ->  N-chloromethanimine N-oxide     (BB 23126)

    Fail-closed (returns None) on any decline; the caller falls through."""
    from rdkit import Chem
    from rdkit.Chem import RWMol

    # SMARTS [NX2,NX3;+](=[#6])[OX1-]: (N+ index, C index, O- index).
    n_idx = matches[0][0]
    o_idx = matches[0][2]

    rw = RWMol(mol)
    rw.GetAtomWithIdx(n_idx).SetFormalCharge(0)
    rw.RemoveBond(n_idx, o_idx)
    rw.RemoveAtom(o_idx)

    try:
        modified_mol = rw.GetMol()
        Chem.SanitizeMol(modified_mol)
        modified_smiles = Chem.MolToSmiles(modified_mol, canonical=True)
    except Exception:
        return None

    from .fragment_naming import name_fragment_recursively
    try:
        base_name = name_fragment_recursively(modified_smiles)
    except Exception:
        return None

    if not base_name or 'unknown' in base_name.lower():
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
    For branched alkyls: the located PIN (propan-2-yl, butan-2-yl,
    2-methylpropyl) via the substituent chokepoint; tert-butyl is the only
    retained branched prefix emitted directly (P-29.6.1). F-T9 / DD6 RET-02.
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
        # Wave-2 completion C2 (P-61.5.2): a carbon-free -N=O branch on a
        # functional-parent N is the 'nitroso' prefix (N-methyl-N-nitrosourea
        # BB verbatim). Returning None here made _try_name_urea silently DROP
        # the branch (structure loss the RT gate then suppressed).
        _heavy_frag = [i for i in frag_atoms
                       if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if len(_heavy_frag) == 2:
            _n = mol.GetAtomWithIdx(start_idx)
            _other = mol.GetAtomWithIdx(
                next(i for i in _heavy_frag if i != start_idx))
            _bond = mol.GetBondBetweenAtoms(start_idx, _other.GetIdx())
            if (_n.GetSymbol() == 'N' and _n.GetFormalCharge() == 0
                    and _other.GetSymbol() == 'O'
                    and _other.GetDegree() == 1
                    and _other.GetFormalCharge() == 0
                    and _bond is not None
                    and _bond.GetBondType() == Chem.BondType.DOUBLE):
                return "nitroso"
        return None

    # RETAINED PREFERRED branched alkyl: only tert-butyl (P-29.6.1) is emitted
    # directly. F-T9 / DD6 RET-02: isopropyl (P-29.6.2.2) and sec-butyl / isobutyl
    # (P-29.6.3) are NO LONGER hardcoded here — like the parallel
    # _check_retained_substituent, this functional-class R-group namer must
    # headline the located/systematic PIN. Every other branched / secondary
    # pure-carbon alkyl is routed to the substituent chokepoint below (which goes
    # through _located_acyclic_alkyl_name: propan-2-yl, butan-2-yl, 2-methylpropyl).
    start_atom = mol.GetAtomWithIdx(start_idx)
    if start_atom.GetSymbol() == 'C':
        c_neighbors_in_frag = [
            nbr.GetIdx() for nbr in start_atom.GetNeighbors()
            if nbr.GetIdx() in frag_set and nbr.GetSymbol() == 'C'
        ]
        if carbon_count == 4 and len(c_neighbors_in_frag) == 3:
            # tert-butyl: C(CH3)3 -- 3 carbon branches at the attachment carbon.
            return "tert-butyl"

    # Phase 139 gap closure: detect chain-FG oxygen/sulfur in fragment
    # BEFORE simple alkyl path.  Oxygen and sulfur on chains indicate
    # functional groups (=O -> "oxo", -OH -> "hydroxy", =S -> "thioxo")
    # that get_alkyl_name() would silently ignore.
    # Phase 163 (FRN) extension: also recognize selenium (34) and tellurium (52)
    # parallel to S — =Se -> "selenoxo", =Te -> "telluroxo" per AUDIT-FRN § 5.4
    # (PREFIX_FORMS entries for selenoaldehyde / telluroaldehyde / selenoketone /
    # telluroketone shipped in commit 163-03-01).
    # Scope limited to O/S/Se/Te:
    #   - Nitrogen is excluded because N atoms in fragments are typically
    #     part of functional class patterns (urea, guanidine, amide) that
    #     are handled by dedicated naming paths, not chain FG prefixes.
    #     (=NH iminoester naming is delegated to handlers/imidate.py per
    #     CONTEXT D-03.)
    #   - Ring heteroatoms are excluded (structural ring members).
    #   - Exocyclic heteroatoms bonded ONLY to ring atoms (C=O on a ring
    #     carbon in fused ureas/lactams) are excluded (ring decorations).
    # A non-ring O/S/Se/Te qualifies as a chain FG indicator only when at
    # least one of its fragment-neighbors is also NOT in a ring (chain context).
    def _is_chain_fg_heteroatom(idx):
        atom = mol.GetAtomWithIdx(idx)
        # Oxygen (8), sulfur (16), selenium (34, Phase 163), tellurium (52, Phase 163)
        if atom.GetAtomicNum() not in (8, 16, 34, 52):
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

    # F-T9 / DD6 RET-02: a BRANCHED or internally-attached PURE-CARBON alkyl
    # R-group must go through the substituent chokepoint for its located PIN
    # (propan-2-yl, butan-2-yl, 2-methylpropyl) BEFORE the linear default — the
    # get_alkyl_name() fallback below would otherwise collapse it to the linear
    # name (propyl/butyl), the exact defect RET-02 removes. A simple TERMINAL
    # LINEAR alkyl keeps the fast get_alkyl_name() path (byte-identical), and the
    # chokepoint declining still falls through to it as a last resort.
    if not heteroatom_in_frag:
        try:
            from .substituent_naming import _is_linear_alkyl, _attach_is_chain_terminus
            _terminal_linear = (
                _is_linear_alkyl(mol, frag_atoms)
                and _attach_is_chain_terminus(mol, frag_atoms, start_idx)
            )
        except Exception:
            _terminal_linear = True  # conservative: keep the legacy linear path
        if not _terminal_linear:
            try:
                from .substituent_enumerator import name_substituent as _ns_r
                prefix = _ns_r(mol, set(frag_atoms), start_idx)
                if prefix and prefix != "substituent":
                    return prefix
            except Exception:
                pass

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
    #
    # v29: the carve-out used to be a hand-written `r_name not in ('tert-butyl',
    # 'sec-butyl')` exception list — the only OTHER site in the tree that spelled
    # the rule out itself, and therefore the one guaranteed to drift from the
    # primitive. It was enumerated by NAME, so it silently failed for any other
    # member of the class ('tert-pentyl', a capitalized spelling) and equally
    # silently suppressed the marks on 'tert-butylsulfanyl', which is compound
    # under P-16.3.3 and must keep them. The primitive decides on the REMAINDER,
    # so it gets both right.
    from .naming_utils import italicized_prefix_is_bare
    needs_parens = (any(c in r_name for c in '-,')
                    and not italicized_prefix_is_bare(r_name))

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


def _try_name_semicarbazone(features: Any) -> Optional[str]:
    """P-15.2.2 (BB 5074ff, W2E-P1FG Task 12): R2C=N-NH-C(=O)-NH2 ->
    '2-({R2C}ylidene)hydrazine-1-carboxamide' (PIN; acetone semicarbazone
    -> 2-(propan-2-ylidene)hydrazine-1-carboxamide).

    Fail-closed: exactly one motif; the terminal carboxamide N and the
    bridge N must be UNSUBSTITUTED (the SMARTS [NX3H2]/[NX3H1] enforces
    this — N-methyl variants need N-locant machinery not built here); the
    ylidene fragment must name via the substituent pipeline as an '-ylidene'."""
    from rdkit import Chem
    mol = features.mol
    patt = Chem.MolFromSmarts("[CX3](=[NX2][NX3H1][CX3](=[OX1])[NX3H2])")
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 1:
        return None
    c, n2, n1, cc, o, n_am = matches[0]
    core = {n2, n1, cc, o, n_am}
    # BFS the ylidene fragment from the sp2 C, never crossing into the core.
    frag = {c}
    _stack = [c]
    while _stack:
        _i = _stack.pop()
        for _nb in mol.GetAtomWithIdx(_i).GetNeighbors():
            _j = _nb.GetIdx()
            if _j not in frag and _j not in core:
                frag.add(_j)
                _stack.append(_j)
    if frag | core != {a.GetIdx() for a in mol.GetAtoms()}:
        return None
    # The ylidene C's only bond leaving the fragment must be the =N2 (double).
    c_atom = mol.GetAtomWithIdx(c)
    _ext = [nb.GetIdx() for nb in c_atom.GetNeighbors()
            if nb.GetIdx() not in frag]
    if _ext != [n2]:
        return None
    # The substituent pipeline emits the P-29.2 morphology for the attachment
    # bond it is given, and this attachment IS the C=N double bond, so the
    # '-ylidene' comes back already formed. It used to be spelled here by
    # appending 'idene' to a '-yl' token; that duplicated the morphology
    # decision in a second place and only worked because the pipeline was
    # returning the WRONG (single-valence) token for a double bond.
    ylidene = _double_bonded_carbon_prefix(mol, frag, c)
    if ylidene is None:
        return None
    return f"2-({ylidene})hydrazine-1-carboxamide"


def _double_bonded_carbon_prefix(mol, frag, c) -> Optional[str]:
    """The P-29.2 ``-ylidene`` prefix for fragment ``frag`` double-bonded to
    the rest of the molecule at its carbon ``c``, or ``None``.

    Shared by the semicarbazone and hydrazone namers, which differ only in the
    tail they cite it on. The morphology is verified rather than assumed: the
    substituent pipeline is the single producer, and the returned token is
    accepted only if its own text spells the two free valences the C=N bond
    actually has.
    """
    from .substituent_enumerator import name_substituent
    from ..validation.name_morphemes import free_valence_morphology
    token = name_substituent(mol, frag, c)
    morphology = free_valence_morphology(token)
    if morphology.confident and morphology.free_valences == 2:
        return token
    return None


def _try_name_hydrazone_substitutive(features: Any) -> Optional[str]:
    """P-68.3.1.2.2 (BB 38562): R2C=N-NH2 -> the SUBSTITUTIVE PIN, an 'ylidene'
    derivative of the parent hydride hydrazine (H2N-NH2) — 'propylidenehydrazine'
    (PIN), NOT the functional-class 'propanal hydrazone'. Method (1) is the PIN.

    Mirrors ``_try_name_semicarbazone`` minus the carboxamide tail: match the
    bare hydrazone motif R2C=N-NH2, BFS the ylidene fragment from the sp2 C
    (never crossing the =N), name it via the substituent pipeline -- which
    emits the '-ylidene' directly, off the C=N bond order -- and cite it on
    'hydrazine'. A LONE ylidene on the symmetric
    hydrazine takes no position locant (P-14.3.4.2: unambiguous); a complex
    ylidene name is enclosed in parentheses.

    Fail-closed (returns None -> the functional-class fallback): exactly one
    motif; the terminal N MUST be an unsubstituted -NH2 (the [NX3H2] guard) and
    the imino N must bear only the =C and the -NH2 (no N-substituent — that is
    the 1,1-/1,2-disubstituted-hydrazine class, N-locant machinery not built
    here); the ylidene fragment must name via the substituent pipeline as an
    '-ylidene'; and it must account for every non-core atom."""
    from rdkit import Chem
    mol = features.mol
    patt = Chem.MolFromSmarts("[CX3]=[NX2][NX3H2]")
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 1:
        return None
    c, n2, n1 = matches[0]
    # The imino N (n2) must carry ONLY the =C and the -NH2 (degree 2, no extra
    # substituent -> not a 1,2-disubstituted hydrazine).
    if mol.GetAtomWithIdx(n2).GetDegree() != 2:
        return None
    core = {n2, n1}
    # BFS the ylidene fragment from the sp2 C, never crossing into the core.
    frag = {c}
    _stack = [c]
    while _stack:
        _i = _stack.pop()
        for _nb in mol.GetAtomWithIdx(_i).GetNeighbors():
            _j = _nb.GetIdx()
            if _j not in frag and _j not in core:
                frag.add(_j)
                _stack.append(_j)
    if frag | core != {a.GetIdx() for a in mol.GetAtoms()}:
        return None
    # The ylidene C's only bond leaving the fragment must be the =N2 (double).
    c_atom = mol.GetAtomWithIdx(c)
    _ext = [nb.GetIdx() for nb in c_atom.GetNeighbors()
            if nb.GetIdx() not in frag]
    if _ext != [n2]:
        return None
    from .naming_utils import is_complex_substituent
    ylidene = _double_bonded_carbon_prefix(mol, frag, c)
    if ylidene is None:
        return None
    if is_complex_substituent(ylidene):
        ylidene = f"({ylidene})"
    return f"{ylidene}hydrazine"


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

    # v23 Phase 7 (7d): halogen-only N-substituted urea. _name_r_group below
    # returns None for a lone halogen, so a halogenated urea would otherwise DROP
    # the halogens and mis-name as "urea" (structure loss — gate-suppressed). Cite
    # the halogens with the Blue Book locant-omission rule (P-66.1.6.1.1 /
    # P-14.3.4.2): omit locants only when UNAMBIGUOUS — a single substituent
    # ("fluorourea") or all four positions identically substituted
    # ("tetrafluorourea", BB P-66.1.6.1). Ambiguous partial patterns (N,N'- vs
    # N,N-difluoro, mixed halogens) fall through to the carbon path (declined ->
    # gate-suppressed; their N,N' locants need the carbon-handler rework, deferred).
    _HALO_UREA = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}

    def _n_halogen_subs(n_idx):
        """Halogen symbols on this urea N, or None if it bears any non-halogen
        (carbon) heavy substituent."""
        halos = []
        for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            if nbr.GetIdx() == c_idx or nbr.GetAtomicNum() <= 1:
                continue
            sym = nbr.GetSymbol()
            if sym not in _HALO_UREA or nbr.GetDegree() != 1:
                return None
            halos.append(sym)
        return halos

    _h1 = _n_halogen_subs(n1_idx)
    _h2 = _n_halogen_subs(n2_idx)
    if _h1 is not None and _h2 is not None and (_h1 or _h2):
        _all = _h1 + _h2
        if len(set(_all)) == 1:                       # one halogen species only
            _hp = _HALO_UREA[_all[0]]
            if len(_all) == 1:                        # mono -> no locant
                return f"{_hp}urea"
            if len(_h1) == 2 and len(_h2) == 2:       # all four positions -> no locant
                return f"tetra{_hp}urea"
        # else: ambiguous partial / mixed-halogen pattern -> fall through.

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
        # Both Ns substituted: choose which N is UNPRIMED ("N") by IUPAC
        # P-66.1.6.1.1 + P-14.3.5 (lowest set of locants), then alphanumerical-
        # first (P-14.5.2). The N bearing MORE substituents takes the unprimed
        # 'N' ({N,N,N'} < {N,N',N'}); on an equal count, the N whose substituent
        # set is alphanumerically first takes 'N'. The single sort key
        # (-len, sorted(subs)) captures both (more subs -> smaller; ties broken
        # by the alphabetically-first substituent set). This ALSO removes the
        # SMILES-atom-order dependence (the SMARTS [N]C(=O)[N] match order was
        # the old, nondeterministic basis) -- a determinism fix as well as a PIN fix.
        key1 = (-len(n1_subs), sorted(n1_subs))
        key2 = (-len(n2_subs), sorted(n2_subs))
        if key1 <= key2:
            first_subs, second_subs = n1_subs, n2_subs
        else:
            first_subs, second_subs = n2_subs, n1_subs

    tagged_subs = []  # list of (locant, sub_name) pairs
    for s in first_subs:
        tagged_subs.append(("N", s))
    for s in second_subs:
        tagged_subs.append(("N'", s))

    return _build_n_substituted_name(tagged_subs, "urea")


def _try_name_cyanamide(features: Any) -> Optional[str]:
    """AM-1 (P-66.1.6.2): name a cyanamide (H2N-C#N) and its N-substituted
    derivatives with 'cyanamide' as the retained parent — no N-locants
    ('(propan-2-yl)cyanamide', 'diethylcyanamide', BB 33531/33535/33537).

    Modelled on _try_name_urea but single-nitrogen. Fail-closed: returns None
    when any N-substituent branch is un-nameable, so a structure-dropping name
    is never emitted (SELF-01 keeps it honest).
    """
    from collections import Counter
    from .naming_utils import needs_brackets

    mol = features.mol
    matches = (getattr(features, 'functional_groups', {}) or {}).get(
        'cyanamide', [])
    if not matches:
        return None
    match = matches[0]
    if len(match) < 3:
        return None
    # SMARTS [NX3;!R][CX2]#[NX1] -> (amine_N, C, terminal_N)
    n_idx, c_idx, term_n_idx = match[0], match[1], match[2]
    core = {n_idx, c_idx, term_n_idx}

    def _wrap(nm):
        # A compound substituent directly abutting 'cyanamide' takes enclosing
        # marks (P-16.3.3): '(propan-2-yl)cyanamide'; simple 'methyl' stays bare.
        return f"({nm})" if needs_brackets(nm) else nm

    sub_names = []
    for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        nb = nbr.GetIdx()
        if nb in core or nbr.GetAtomicNum() <= 1:
            continue
        nm = _name_r_group(mol, nb, exclude_atoms=core)
        if not nm:
            return None  # un-nameable branch -> fail closed
        sub_names.append(nm)

    if not sub_names:
        return "cyanamide"
    if len(sub_names) == 1:
        return f"{_wrap(sub_names[0])}cyanamide"
    _counts = Counter(sub_names)
    if len(_counts) == 1:
        nm = next(iter(_counts))
        return f"{get_multiplier_prefix(len(sub_names), nm)}{_wrap(nm) if needs_brackets(nm) else nm}cyanamide"
    # 2+ distinct substituents -> alphanumeric order, wrap compounds
    _parts = []
    for nm in sorted(_counts, key=alpha_sort_key):
        _c = _counts[nm]
        piece = _wrap(nm)
        if _c > 1:
            piece = f"{get_multiplier_prefix(_c, nm)}{piece}"
        _parts.append(piece)
    return f"{''.join(_parts)}cyanamide"


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
    from .naming_utils import needs_brackets, apply_enclosing_marks
    prefix_parts = []
    for name in sorted(sub_locants.keys()):
        locants = sub_locants[name]
        count = len(locants)
        locant_str = ",".join(locants)
        # v26 BP-2 RC-5 (P-16.3.3; BB 33336 'N-[1-cyano-3-(methylsulfanyl)propyl]
        # -N'-methylurea (PIN)'): a COMPOUND N-substituent (its own locants /
        # hyphens) is enclosed so a numeral never abuts the next N-locant; simple
        # names (methyl, phenyl) are byte-identical (needs_brackets False). Sort
        # + multiplier stay on the RAW name -> ordering unchanged.
        enc = apply_enclosing_marks(name, -1) if needs_brackets(name) else name
        if count == 1:
            prefix_parts.append(f"{locant_str}-{enc}")
        else:
            mult = get_multiplier_prefix(count, name)
            prefix_parts.append(f"{locant_str}-{mult}{enc}")

    prefix = "-".join(prefix_parts)
    return _join_prefix_to_name(prefix, base_name)  # L2 (P-16.3.3)


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


def _complex_ring_parent_atom_indices(
    complex_result: Any,
    mol: Optional[Any] = None,
) -> Optional[Set[int]]:
    """Compute the set of heavy-atom indices the complex_ring name accounts for.

    Phase 153 D-05 mirror of Phase 152 BL-01 (_ring_handler_parent_atom_indices,
    below). For complex_ring, the atom set is::

        ring_atoms ∪ {substituent atoms reachable from ring_atoms in `mol`}

    The substituent walk MUST be included because complex_ring substituent
    discovery happens AT THE TIER-A return site via _enrich_complex_ring_with_subs
    (composer.py:1393-1397) using discover_substituents(); without including
    those off-ring atoms in the gate's coverage denominator, every
    substituted complex_ring (e.g. natural-product spiro/bicyclo) has
    coverage < 0.99 and the >= 0.99 gate at composer.py:1623 rejects
    injection. Mirrors the benzene/heterocycle BL-01 pattern at
    _ring_handler_parent_atom_indices.

    `mol` is OPTIONAL for backwards compatibility with code that passes
    only the complex_result; when None, the bare ring_atoms set is
    returned (used by old callers; the gate consumer at
    composer.py:1404 always passes mol).

    Returns None when ring_atoms is empty/missing -- caller treats as
    "no real coverage measurement available" and skips injection per D-09
    (missing > wrong).
    """
    if not complex_result or not getattr(complex_result, 'ring_atoms', None):
        return None
    ring_set: Set[int] = {int(i) for i in complex_result.ring_atoms}
    if mol is None:
        return ring_set
    # Walk all heavy atoms reachable from any ring atom (BFS over non-ring
    # atoms; stops at the next ring atom). This covers exocyclic substituent
    # trees of arbitrary depth -- matching what discover_substituents finds
    # from oriented_ring.
    accounted: Set[int] = set(ring_set)
    visited: Set[int] = set(ring_set)
    queue = list(ring_set)
    while queue:
        idx = queue.pop()
        try:
            atom = mol.GetAtomWithIdx(idx)
        except Exception:
            continue
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in visited:
                continue
            if nbr_idx in ring_set:
                continue
            visited.add(nbr_idx)
            accounted.add(nbr_idx)
            queue.append(nbr_idx)
    return accounted


def _ring_is_whole_molecule_for_complex(
    complex_result: Any, mol: Any,
) -> bool:
    """Mirror of the cycloalkane/cycloalkene BL-02 flag at composer.py:1851-1856.

    Returns True iff the complex_ring's atom set IS exactly the molecule's
    heavy atoms. When True, exocyclic E/Z attribution to ring locants is
    safe (because there are no exocyclic atoms). When False, the chain
    pipeline owns exocyclic E/Z and we must pass include_near_parent_ez=False.

    Phase 153 D-06 -- per D-09 (missing > wrong) when in doubt, suppress.
    The new wiring at composer.py:1614+ DOES NOT inherit the
    composer.py:7305 hardcoded `include_near_parent_ez=True` anti-pattern
    (D-21 -- that path is composer.py-decomposition territory / IM-22 / v19).
    """
    if not complex_result or not getattr(complex_result, 'ring_atoms', None):
        return False
    return len(set(complex_result.ring_atoms)) == mol.GetNumHeavyAtoms()


def _ring_handler_parent_atom_indices(features: Any, handler: str) -> Optional[Set[int]]:
    """Compute the set of heavy-atom indices accounted for by the chosen
    ring-handler name (BL-01 fix, 2026-05-03; see 152-VERIFICATION.md).

    Returns None when the data is not yet populated (caller treats None as
    "no real coverage measurement available" and skips injection per D-09:
    "better a missing stereo block than a wrong one").

    handler must be 'benzene' or 'heterocycle'. The atom set is::

        ring_atoms ∪ {atom_idx for sub_list in substituents.values()
                      for atom_idx in sub_list}

    The substituents dict on features stores per-ring-position lists of
    substituent info dicts (Dict[int, List[Dict]] for benzene/heterocycle;
    the inner Dict carries 'atoms' / 'atom_indices' / 'substituent_atoms'
    field naming inherited from get_benzene_substituents /
    get_heterocycle_substituents). Defensively handles list-of-int,
    list-of-dict-with-'atoms', and list-of-tuple shapes so that future
    substituent-shape evolution (T-152-02-01) cannot silently break the gate.
    """
    if handler == 'benzene':
        ring_atoms = getattr(features, 'benzene_ring', None)
        substituents = getattr(features, 'benzene_substituents', None) or {}
    elif handler == 'heterocycle':
        ring_atoms = (
            getattr(features, 'heterocycle_ring', None)
            or getattr(features, 'principal_ring', None)
        )
        substituents = getattr(features, 'heterocycle_substituents', None) or {}
    else:
        return None
    if not ring_atoms:
        return None
    accounted: Set[int] = set(int(i) for i in ring_atoms)
    # substituents is a Dict[ring_pos, List[...]]. Each inner item is either a
    # dict with 'atoms' / 'atom_indices', a list of int, or a single int.
    if isinstance(substituents, dict):
        sub_iter = substituents.values()
    else:
        sub_iter = [substituents]
    for sub_entry in sub_iter:
        if not sub_entry:
            continue
        for sub in sub_entry:
            if isinstance(sub, dict):
                atoms = (
                    sub.get('atoms')
                    or sub.get('atom_indices')
                    or sub.get('substituent_atoms')
                    or []
                )
                accounted.update(int(a) for a in atoms)
            elif isinstance(sub, (list, tuple, set)):
                accounted.update(int(a) for a in sub)
            elif isinstance(sub, int):
                accounted.add(sub)
    return accounted


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

    # P-45.5.1 / P-45.6.3 (diaryl-linked-by-heteroatom, W2F-P8): when the
    # molecule has 2+ benzene rings and NO senior suffix PCG competing for the
    # parent — a diaryl ether (principal_group None) or a secondary diaryl amine
    # (principal_group 'secondary_amine', named as the retained aniline via
    # substituent promotion, P-62.2.1.1) — the parent ring must be chosen by the
    # alphanumerical order of the complete candidate names (with the R<S
    # tie-break), which is DETERMINISTIC. The candidate pool otherwise competes a
    # per-benzene-ring parent pass under 'first_applicable' (first-added wins), so
    # the choice flipped with the SMILES atom order. Delegate to the single
    # preferred-parent authority so every pass yields the same, correct name.
    _pg = getattr(features, 'principal_group', None)
    if _pg is None or _pg == 'secondary_amine':
        from ..rules.benzene import is_benzene_ring, _preferred_benzene_parent_ring
        _bz_rings = [r for r in mol.GetRingInfo().AtomRings()
                     if is_benzene_ring(mol, r)]
        if len(_bz_rings) > 1:
            _pref = _preferred_benzene_parent_ring(mol)
            if _pref is not None:
                return _pref[1]

    ring_atoms = features.benzene_ring
    substituents = features.benzene_substituents

    # If no substituents, return "benzene" (should be caught by retained names,
    # but handle here as fallback). P-31.2.4.1 (Wave-2 completion): a bare
    # ring whose carbons are H-deficient is benzyne — emit the
    # didehydrobenzene parent instead of the structure-dropping 'benzene'.
    if not substituents:
        from ..rules.benzene import didehydro_benzene_name
        _ddh = didehydro_benzene_name(mol, ring_atoms)
        if _ddh is not None:
            return _ddh
        # CONSERVATION (jar-independent, Wave-2 completion B4): bare benzene
        # has exactly 6 heavy atoms; MORE heavy atoms with zero perceived
        # substituents means dropped atoms (a fused system whose fused namer
        # failed and fell through here) -- decline instead of emitting the
        # partial 'benzene'.
        if mol.GetNumHeavyAtoms() != 6:
            return None
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

    # Orient the ring for lowest locants.
    # W3-P04 (P-14.4(c)): anchor the principal characteristic group (ring atoms
    # bearing a suffix-type FG) to the lowest locant BEFORE detachable
    # substituents. Without this the numberer minimized the COMBINED substituent
    # set, giving e.g. '1-methylbenzene-2,4-disulfonic acid' instead of the PIN
    # '4-methylbenzene-1,3-disulfonic acid'. Mirrors the shipped E1/DD4 anchor in
    # namer.py Branch 3 (so the locant HINT and emitted NAME agree). Phenols
    # (hydroxy = prefix) leave the set empty -> no-op (anchored downstream).
    _pcg_positions = set()
    if isinstance(substituents, dict):
        _pcg_positions = {
            atom_idx for atom_idx, subs in substituents.items()
            if isinstance(subs, list) and any(
                isinstance(s, dict) and s.get("is_suffix") for s in subs
            )
        }
    oriented_ring = orient_benzene(
        mol, ring_atoms, substituents,
        principal_group_positions=_pcg_positions or None,
    )

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
    return name_substituted_polycyclic(
        features.mol, pah_name, substituents,
        principal_group=getattr(features, 'principal_group', None),
    )


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

    # P-24.3.1 spirobi (two identical polycyclic components at one spiro atom):
    # n_rings > n_spiro+1 so is_spiro_system rejects it; both sides polycyclic
    # so is_mixed_spiro_fused (one side = single ring) declines too.
    from ..rules.spiro import is_spirobi as _is_spirobi
    if _is_spirobi(mol):
        return True

    # P-24.4.1 dispiroter (three identical polycyclic components, two spiro
    # atoms): n_rings > n_spiro+1 so is_spiro_system rejects; two spiro atoms so
    # is_spirobi declines.
    from ..rules.spiro import is_dispiroter as _is_dispiroter
    if _is_dispiroter(mol):
        return True

    # P-24.6 unbranched polyspiro, different components (>=1 polycyclic).
    from ..rules.spiro import is_unbranched_polyspiro_different as _is_polyspiro_diff
    if _is_polyspiro_diff(mol):
        return True

    # P-24.7 branched polyspiro (central >=3-junction). Recognised as complex so
    # the dispatcher can FAIL CLOSED (refuse) rather than leaking a partial name.
    from ..rules.spiro import is_branched_polyspiro as _is_branched_polyspiro
    if _is_branched_polyspiro(mol):
        return True

    # P-24.8.3 λ spiroter (three identical components + λ spiro atom in 3 rings).
    from ..rules.spiro import is_spiroter as _is_spiroter
    if _is_spiroter(mol):
        return True

    # P-24.8.4.2 three DISTINCT named components + one λ spiro atom in 3 rings.
    from ..rules.spiro import is_spiro_named_components as _is_spiro_named_components
    if _is_spiro_named_components(mol):
        return True

    # P-24.8.1.3 λ spiro atom in >=3 rings — recognised so the dispatcher can
    # FAIL CLOSED rather than leaking a partial monocyclic name.
    from ..rules.spiro import is_lambda_multiring_spiro as _is_lambda_multiring
    if _is_lambda_multiring(mol):
        return True

    # Check spiro BEFORE polycyclic-bridged (P-24.2).
    # Dispiro compounds have 3+ SSSR rings, causing is_polycyclic_system()
    # to return True. Spiro check must precede polycyclic to avoid misrouting.
    if is_spiro_system(mol):
        return True

    # P-24.5 spiro-of-von-Baeyer (Phase 13B(c)): monospiro with >=1 cage
    # component (mirrors the _classify_complex_ring dispatch order).
    from ..rules.spiro import is_spiro_vonbaeyer
    if is_spiro_vonbaeyer(mol):
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
    # P-24.6 unbranched polyspiro, DIFFERENT components (>=1 polycyclic).
    # Computed FIRST: a fluorene/cyclohexane/indene polyspiro chain trips
    # detect_bridged_fused (a fused core + a spiro-linked ring reads as a bridge),
    # so this positive gate must guard the bridged-fused branch below.
    from ..rules.spiro import is_unbranched_polyspiro_different as _is_polyspiro_diff_classify
    _input_is_polyspiro_diff = _is_polyspiro_diff_classify(mol)
    if _input_is_polyspiro_diff:
        return 'polyspiro-different'

    # P-24.7 BRANCHED polyspiro (central component with >=3 spiro junctions).
    # The full component-name build is a follow-on; recognise the class here and
    # route to a FAIL-CLOSED tag so it refuses (UNSUPPORTED_RING_SYSTEM) rather
    # than leaking a structure-dropping partial name from the ortho-fused path.
    from ..rules.spiro import is_branched_polyspiro as _is_branched_polyspiro_classify
    if _is_branched_polyspiro_classify(mol):
        return 'polyspiro-branched-different'

    # P-24.8.3 λ spiroter (three identical polycyclic components + one λ spiro
    # atom in 3 rings). Must precede the lambda-multiring FAIL-CLOSED guard.
    from ..rules.spiro import is_spiroter as _is_spiroter_classify
    if _is_spiroter_classify(mol):
        return 'spiroter'

    # P-24.8.4.2 three DISTINCT named components + one λ spiro atom in 3 rings
    # (BB:11250). Must precede the lambda-multiring FAIL-CLOSED guard, which
    # would otherwise catch this λ-atom-in-3-rings shape and refuse.
    from ..rules.spiro import is_spiro_named_components as _is_spiro_named_classify
    if _is_spiro_named_classify(mol):
        return 'spiro-named-components'

    # P-24.8.1.3 λ spiro atom in >=3 monocyclic rings: detection-only, FAIL
    # CLOSED (the λ von-Baeyer build is a follow-on). Refuse rather than leak a
    # partial monocyclic name.
    from ..rules.spiro import is_lambda_multiring_spiro as _is_lambda_multiring_classify
    if _is_lambda_multiring_classify(mol):
        return 'lambda-multiring-spiro'

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
    # P-24.3.1 spirobi: same hazard — the catalog matches ONE polycyclic
    # component (e.g. indene) as a sub-core and returns 'ortho-fused' for the
    # whole two-component spiro system, so the spirobi branch never fires.
    from ..rules.spiro import is_spirobi as _is_spirobi_classify
    _input_is_spirobi = _is_spirobi_classify(mol)
    # P-24.4.1 dispiroter (three identical polycyclic components, two spiro
    # atoms): SAME catalog-skip hazard as spirobi — the catalog matches one
    # component. Computed once, reused at the dispatch branch below.
    from ..rules.spiro import is_dispiroter as _is_dispiroter_classify
    _input_is_dispiroter = _is_dispiroter_classify(mol)
    # _input_is_polyspiro_diff (P-24.6) was computed + returned at the top of
    # this function (it must guard the bridged-fused branch); reused here only
    # for the catalog-skip guard.
    # P-24.5 spiro-of-von-Baeyer / spiro-PAH (Phase 13B): SAME hazard. When a
    # spiro component is a fused HETEROCYCLE (e.g. spiro[fluorene-9,9'-xanthene])
    # the catalog matches the xanthene sub-core; worse, get_ring_systems splits
    # the by-atom spiro junction into two systems and the shared spiro atom makes
    # the core atoms intersect BOTH, tricking the dominant-system heuristic into
    # returning 'ortho-fused' for the whole molecule -> the spiro-vonbaeyer
    # branch never fires. Computed once here, reused at the dispatch branch below.
    from ..rules.spiro import is_spiro_vonbaeyer as _is_spiro_vb_classify
    _input_is_spiro_vb = _is_spiro_vb_classify(mol)

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
    core_match = None if (_phase151_input_is_mixed or _input_is_spirobi or _input_is_dispiroter or _input_is_polyspiro_diff or _input_is_spiro_vb) else match_fused_heterocycle_core(mol)
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

    # P-24.3.1 spirobi (two identical polycyclic components at one spiro atom).
    # Disjoint from mixed-spiro-fused (both sides polycyclic, not a single side
    # ring) and from pure spiro (n_rings > n_spiro + 1). Reuse the value already
    # computed above for the catalog-skip (avoid a second full partition).
    if _input_is_spirobi:
        return 'spirobi'

    # P-24.4.1 dispiroter (three identical polycyclic components, two spiro
    # atoms). n_rings > n_spiro + 1 so is_spiro_system declines; two spiro atoms
    # so is_spirobi declines. Must precede pure-spiro and polycyclic-bridged.
    if _input_is_dispiroter:
        return 'dispiroter'

    # P-24.6 polyspiro-different already handled at the top of this function
    # (returned before detect_bridged_fused).

    # Check spiro BEFORE polycyclic-bridged (P-24.2).
    # Dispiro compounds have 3+ SSSR rings, causing is_polycyclic_system()
    # to return True. The refined is_spiro_system() checks n_rings == n_spiro + 1,
    # so complex polycyclic molecules with incidental spiro atoms are excluded.
    if is_spiro_system(mol):
        return 'spiro'

    # P-24.5 spiro-of-von-Baeyer / spiro-PAH (Phase 13B(c)/(b)): a monospiro
    # system with >=1 von Baeyer cage OR carbo-PAH (fluorene) component (e.g.
    # spiro[bicyclo[2.2.1]heptane-2,1'-cyclohexane], 2,2'-spirobi[bicyclo[2.2.1]
    # heptane], 9,9'-spirobi[fluorene], spiro[fluorene-9,9'-xanthene]). The
    # cage-bridge / C9 spiro atom sits in >2 SSSR rings (or the components only
    # share the spiro atom), so is_spiro_system / is_spirobi / is_mixed_spiro_
    # fused all decline (they key off get_spiro_atoms) and the system would
    # otherwise mis-route to polycyclic-bridged / ortho-fused -> 'unknown'.
    # Must precede the polycyclic-bridged check. Reuses the flag computed for
    # the catalog-skip above. Fail-closed.
    if _input_is_spiro_vb:
        return 'spiro-vonbaeyer'

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


def _fused_core_covers_ring_system(mol, atom_to_locant) -> bool:
    """G0 fail-closed safety (DD7 S1): does the matched fused-ring core cover
    every atom of the fused ring system(s) it sits in?

    ``name_fused_heterocycle`` can match a 2-component catalog core (e.g.
    ``furo[3,2-b]pyridine``) as a SUBSTRUCTURE of a larger polycomponent fused
    system (e.g. difuropyridine). The leftover FUSED ring atoms are then handed
    to substituent discovery and mis-named as an acyclic prefix (the phantom
    ``7-ethoxy``). Returning False here lets the caller refuse rather than emit
    that structurally-wrong name.

    ``atom_to_locant`` int keys are the named core atoms. A pendant ring joined
    by a single (non-ring) bond is a SEPARATE fused ring system, so a legitimate
    cyclic substituent (e.g. 2-phenylquinoline's phenyl) does NOT trip this:
    only fused ring atoms left OUT of the matched core do. When ``atom_to_locant``
    is empty (the exact-match catalog path, which fully names the system) the
    check is a no-op (no core atoms -> nothing uncovered)."""
    # IN-03 contract: keys for NAMED ring atoms in atom_to_locant are always int
    # (RDKit atom indices); name_fused_heterocycle never emits non-int atom keys.
    # The isinstance filter is defensive — if that contract ever breaks, the
    # coverage check would under-count and miss a refusal, so keep it int-keyed.
    core_atoms = {k for k in (atom_to_locant or {}) if isinstance(k, int)}
    if not core_atoms:
        return True
    from ..perception.rings import get_ring_systems
    for rs in get_ring_systems(mol):
        rs = set(rs)
        if (core_atoms & rs) and not rs.issubset(core_atoms):
            return False
    return True


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
        # Wave-2 P5 fused (Task 8): an exact-match cataloged carbocyclic PAH
        # parent (POLYCYCLIC_DATA) must be nameable regardless of how
        # _classify_complex_ring labels it. A 3-component all-carbon fusion
        # parent that includes a non-aromatic cyclobuta/cyclopenta ring (P-25.5.2
        # cyclobuta[1,7]indeno[5,6-b]naphthalene) is misclassified 'polycyclic-
        # bridged' and routed to von Baeyer, which declines -> the cataloged
        # fusion PIN was unreachable. Check the exact PAH catalog FIRST, guarded
        # to a bare (unsubstituted) parent with a full authoritative locant map;
        # fail-closed (fall through) if the map is incomplete. Substituted PAHs
        # keep their existing dedicated route (this only fires when every heavy
        # atom is core).
        from ..rules.polycyclics import (
            identify_polycyclic,
            get_polycyclic_iupac_locants,
            get_polycyclic_core_atoms,
        )
        _pah_name = identify_polycyclic(mol)
        if _pah_name is not None:
            _core = get_polycyclic_core_atoms(mol, _pah_name)
            if _core is not None and len(_core) == mol.GetNumAtoms():
                _pah_locants = get_polycyclic_iupac_locants(mol, _pah_name)
                if (_pah_locants is not None
                        and len(_pah_locants) == mol.GetNumAtoms()):
                    return ComplexRingResult(
                        _pah_name, tuple(_core), _pah_locants, True)

        # Wave-2 P5 fused (Task 8): likewise an exact-match cataloged fused
        # HETEROCYCLE (FUSED_HETEROCYCLE_DATA) parent must be nameable even when
        # _classify_complex_ring misroutes it (P-25.5.1.2
        # 2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene — skeletal-'a' + methano
        # bridge — classifies 'polycyclic-bridged' and would decline). The
        # exact-SMILES lookup inside name_fused_heterocycle only returns a bare
        # closed-structure catalog name; it is fail-closed for anything not in
        # the table (returns None -> normal dispatch continues).
        _fh = name_fused_heterocycle(mol)
        if _fh is not None:
            _fname, _fring_atoms, _fmap, _fsubs = _fh
            if _fused_core_covers_ring_system(mol, _fmap):
                return ComplexRingResult(_fname, _fring_atoms, _fmap, _fsubs)

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

        elif ring_type == 'spirobi':
            # P-24.3.1: two identical polycyclic components at one spiro atom
            # (e.g. 1,1'-spirobi[indene]).
            from ..rules.spiro import name_spirobi
            result = name_spirobi(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Spirobi naming failed for molecule")
            return None

        elif ring_type == 'spiroter':
            # P-24.8.3: three identical polycyclic components at one λ spiro atom
            # (e.g. 2lambda6,2',2''-spiroter[[1,3,2]benzodioxathiole]).
            from ..rules.spiro import name_spiroter
            result = name_spiroter(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Spiroter naming failed for molecule")
            return None

        elif ring_type == 'dispiroter':
            # P-24.4.1: three identical polycyclic components at two spiro atoms
            # (e.g. 3,3':6',6''-dispiroter[bicyclo[3.1.0]hexane]).
            from ..rules.spiro import name_dispiroter
            result = name_dispiroter(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Dispiroter naming failed for molecule")
            return None

        elif ring_type == 'polyspiro-different':
            # P-24.6: unbranched polyspiro, different components (>=1 polycyclic)
            # (e.g. dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]).
            from ..rules.spiro import name_unbranched_polyspiro_different
            result = name_unbranched_polyspiro_different(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Unbranched-polyspiro-different naming failed for molecule")
            return None

        elif ring_type == 'polyspiro-branched-different':
            # P-24.7: branched polyspiro (central >=3-junction component). The
            # component-name build is a documented follow-on -> FAIL CLOSED
            # (refuse) so no structure-dropping partial name leaks.
            from ..rules.spiro import name_branched_polyspiro
            result = name_branched_polyspiro(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Branched-polyspiro-different unsupported (fail-closed)")
            return None

        elif ring_type == 'spiro-named-components':
            # P-24.8.4.2: three distinct named components at one λ spiro atom in
            # 3 rings (e.g. 2lambda6-spiro[[1,3,2]benzodioxathiole-2,2'-
            # ([1,2,3]benzoxadithiole)-2,5''-dibenzo[b,d]thiophene]).
            from ..rules.spiro import name_spiro_named_components
            result = name_spiro_named_components(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Spiro-named-components naming failed for molecule")
            return None

        elif ring_type == 'lambda-multiring-spiro':
            # P-24.8.1.3: λ spiro atom in >=3 rings -> FAIL CLOSED (follow-on).
            from ..rules.spiro import name_lambda_multiring_spiro
            result = name_lambda_multiring_spiro(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Lambda-multiring-spiro unsupported (fail-closed)")
            return None

        elif ring_type == 'spiro-vonbaeyer':
            # P-24.5 (Phase 13B(c)): monospiro with >=1 von Baeyer cage
            # component, named by the component-name method
            # (spiro[bicyclo[2.2.1]heptane-2,1'-cyclohexane]) or the spirobi
            # form for two identical cages (2,2'-spirobi[bicyclo[2.2.1]heptane]).
            from ..rules.spiro import name_spiro_vonbaeyer
            result = name_spiro_vonbaeyer(mol)
            if result:
                name, ring_atoms, atom_to_locant, subs_included = result
                return ComplexRingResult(name, ring_atoms, atom_to_locant, subs_included)
            logging.warning("Spiro-von-Baeyer naming failed for molecule")
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
                # G0 fail-closed safety (DD7 S1): when the matched fused-ring
                # core covers only PART of its fused ring system (a 3+-component
                # system matched a 2-component sub-core, P-25.3.4), the leftover
                # FUSED ring atoms get mis-named as an acyclic substituent (e.g.
                # difuropyridine -> '7-ethoxyfuro[3,2-b]pyridine', the 2nd furan
                # read as a phantom ethoxy). That is a structurally-WRONG name;
                # refuse instead (polycomponent fusion is a Phase-G1 build).
                if not _fused_core_covers_ring_system(mol, atom_to_locant):
                    # IN-01: no smiles arg — Orthonym.name back-fills the input SMILES.
                    from ..errors import unsupported_ring_system
                    raise unsupported_ring_system()
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

    except OrthonymLimitError:
        # G0 fail-closed (DD7 S1): a deliberate refusal must NOT be swallowed by
        # the graceful-degradation except below (which would let the molecule
        # cascade to a fragment namer and emit a sub-ring name). Propagate it to
        # Orthonym.name, which converts it to 'unknown organic compound'
        # (default) or re-raises it (raise_on_limit=True).
        raise
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
    return _join_prefix_to_name(prefix_str, ring_name)  # L2 (P-16.3.3)


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

    # P-16.3.3 (L2, Phase 173.5): a hyphen separates the prefix block from a
    # ring parent that starts with a locant (e.g. 2,2,7,7-tetramethyl-1,6-
    # dioxaspiro[4.4]nonane). The prior if/else here was dead scaffolding (both
    # branches concatenated WITHOUT a hyphen). Route through the one correct
    # join helper — byte-identical for letter-leading parents, correct for
    # digit-leading (spiro / Hantzsch-Widman / von Baeyer) parents.
    return _join_prefix_to_name(prefix_str, ring_parent)


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

    # WS-6 / BBR-RCON (DEF-7): identify the ring atoms bearing the principal
    # characteristic group so von Baeyer numbering gives the suffix the lowest
    # locant (P-14.4(c)). Same extraction as the cyclo/heterocycle paths: a ring C
    # whose FG-match heteroatom is exocyclic (the ketone carbonyl C, a C-OH, ...).
    suffix_ring_atoms = set()
    _pg = getattr(features, 'principal_group', None)
    _fgs = getattr(features, 'functional_groups', None) or {}
    if _pg and _pg in _fgs:
        for match in _fgs[_pg]:
            match_set = set(match)
            for aidx in match:
                a = mol.GetAtomWithIdx(aidx)
                if aidx in ring_atoms:
                    # ring-member FG carbon (e.g. the ketone carbonyl C): its
                    # exocyclic non-C heteroatom is in the match.
                    if a.GetSymbol() != 'C':
                        continue
                    for nbr in a.GetNeighbors():
                        nidx = nbr.GetIdx()
                        if (nidx not in ring_atoms and nidx in match_set
                                and nbr.GetSymbol() != 'C'):
                            suffix_ring_atoms.add(aidx)
                            break
                else:
                    # WSD-01: EXOCYCLIC FG carbon/heteroatom (the COOH/CHO/CN
                    # carbon, or an amine N) -> bias numbering to the RING carbon
                    # it attaches to so the suffix gets the lowest locant (P-14.4 c).
                    for nbr in a.GetNeighbors():
                        nidx = nbr.GetIdx()
                        if nidx in ring_atoms and mol.GetAtomWithIdx(nidx).GetSymbol() == 'C':
                            suffix_ring_atoms.add(nidx)
                            break

    # Get complete bicyclo data
    bicyclo_data = get_complete_bicyclo_data(mol, suffix_ring_atoms=suffix_ring_atoms or None)
    if not bicyclo_data:
        # Fall back to simple naming
        fallback_name = name_bicyclo_system(mol)
        if fallback_name:
            return (fallback_name, ring_atoms, {}, True)
        return None

    # G0 fail-closed safety (DD7 S1): bicyclo[...] nomenclature, like von Baeyer,
    # cannot represent aromaticity. Check the EXACT bicyclo cage atoms (WR-01:
    # bicyclo_data['ring_atoms'], NOT the molecule's raw ring-atom union — a
    # pendant aromatic ring is not part of the cage). If the cage carries an
    # aromatic atom, naming it here would silently de-aromatise it into a WRONG
    # saturated cage; refuse (raise the named limit, caught at Orthonym.name).
    from ..rules.polycyclic import vonbaeyer_cage_has_aromaticity
    if vonbaeyer_cage_has_aromaticity(mol, bicyclo_data.get('ring_atoms', ())):
        # IN-01: no smiles arg — Orthonym.name back-fills the original input SMILES.
        from ..errors import unsupported_ring_system
        raise unsupported_ring_system()

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

    # WSD-01: detect ring functional groups via the capable polycyclic logic
    # (ketone/alcohol/aldehyde/carboxylic_acid/amine/nitrile) so a COOH/CHO/CN/
    # amine on a 2-ring von Baeyer system emits the correct SUFFIX, not a phantom
    # 'methyl' (the carbon-only counter bug). get_bicyclo_numbering (Phase 170,
    # P-14.4) stays the numbering source.
    _bicyclo_ring_atoms = set(atom_to_locant.keys()) or ring_atoms
    fg_info = _detect_ring_functional_groups(mol, _bicyclo_ring_atoms, atom_to_locant)
    fg_atoms = fg_info.get('fg_atoms', set())
    # Drop substituents whose atoms ARE FG atoms (the COOH/CHO/CN carbon, amine N)
    # so the carbon-only counter no longer collapses a 1-C FG to 'methyl'.
    if fg_atoms:
        substituents = {
            ra: [s for s in subs if not (set(s.get('atoms', [])) & fg_atoms)]
            for ra, subs in substituents.items()
        }
        substituents = {ra: subs for ra, subs in substituents.items() if subs}

    # Get key data
    descriptor = bicyclo_data.get('descriptor')
    carbon_count = bicyclo_data.get('carbon_count', 0)

    # Get parent stem
    parent_stem = _get_bicyclo_parent_stem(carbon_count)

    # Build unsaturation suffix
    unsat_suffix = _build_bicyclo_unsaturation_suffix(
        unsaturation, atom_to_locant, parent_stem
    )

    # Build substituent prefix (including any NON-principal FG prefixes, e.g. a
    # hydroxy on a bicyclo-carboxylic-acid, so a demoted FG is not dropped).
    sub_prefix = _build_bicyclo_substituent_prefix(
        mol, substituents, atom_to_locant, fg_prefixes=fg_info.get('prefixes')
    )
    if sub_prefix is None:
        # P-29.2 fail-closed: a substituent whose free valence this parent
        # cannot express. Declining lets another candidate producer try, and
        # SELF-01 never sees a name that describes the wrong molecule.
        logging.debug("Bicyclo naming declined: unnameable free valence")
        return None

    # WSD-01: principal characteristic group suffix for ANY ring FG
    # (-one/-ol/-amine inline; -carboxylic acid/-carbaldehyde/-carbonitrile
    # appended), built from the capable _detect_ring_functional_groups result.
    principal_suffix = _build_bicyclo_principal_suffix(fg_info.get('suffix'))

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
        # SUB-02 (B-fix): a hyphen separates the substituent prefix from the
        # NEXT part ONLY when that part starts with a locant digit (e.g.
        # 'octyl-9-oxa'); a letter-initial next part glues directly
        # ('trimethyl' + 'bicyclo[...]' -> 'trimethylbicyclo[...]', which OPSIN
        # requires). The previously-forced unconditional hyphen produced the
        # OPSIN-rejected 'trimethyl-bicyclo[...]'.
        sub_prefix = sub_prefix.rstrip('-')
        next_part = heteroatom_prefix if heteroatom_prefix else descriptor
        if next_part and next_part[0].isdigit():
            sub_prefix += '-'
        parts.append(sub_prefix)

    if heteroatom_prefix:
        parts.append(heteroatom_prefix)

    # Build the main name: bicyclo[x.y.z]parent-unsat(-L-suffix)
    main_name = f"{descriptor}{unsat_suffix}"
    suffix_str, elide_terminal_e = principal_suffix
    if suffix_str:
        # IUPAC P-16.3.3 / P-31.1.4.3.4 vowel elision: drop the parent's trailing
        # 'e' ONLY before a vowel-initial suffix. '-one' elides
        # ('...heptan-6-one', and '...hept-5-en-6-one' for an unsaturated parent);
        # the consonant-initial multiplied '-dione'/'-trione' keeps the 'e'
        # ('...heptane-2,6-dione'). IN-06: the old unconditional endswith('e')
        # strip mis-elided the multiplied form and any '-diene'/'-triene' parent.
        if elide_terminal_e and main_name.endswith('e'):
            main_name = main_name[:-1]
        main_name = f"{main_name}{suffix_str}"
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
    atom_to_locant: Dict[int, int],
    fg_prefixes: Optional[List[Dict]] = None,
) -> Optional[str]:
    """
    Build substituent prefix string for bicyclo naming.

    Groups substituents by name, adds locants and multipliers.

    Returns ``None`` -- distinct from ``""`` -- when a substituent carries the
    P-29.2 ``unnameable`` verdict from ``get_bicyclo_substituents``: its free
    valence is double or triple and no correct prefix exists for it. The caller
    must decline the whole bicyclo parent. Emitting the ``-yl`` form would name
    a different molecule and returning ``""`` would drop the atom, which names
    a different molecule too.

    Args:
        mol: RDKit Mol object
        substituents: Dict from get_bicyclo_substituents
        atom_to_locant: Mapping from atom index to IUPAC locant
        fg_prefixes: Optional non-principal functional-group prefixes
            (``[{'name','locant'}, ...]`` from _detect_ring_functional_groups) so a
            demoted ring FG (e.g. hydroxy on a bicyclo-carboxylic-acid) is emitted
            as a prefix, not dropped (WSD-01).

    Returns:
        Formatted prefix string (e.g., '3,7,7-trimethyl-')
    """
    if not substituents and not fg_prefixes:
        return ""

    # Group substituents by name
    sub_groups: Dict[str, List[int]] = defaultdict(list)

    for ring_atom_idx, sub_list in substituents.items():
        locant = atom_to_locant.get(ring_atom_idx)
        if locant is None:
            continue

        for sub_info in sub_list:
            # P-29.2 verdict recorded by get_bicyclo_substituents, which is the
            # only place that saw the attachment BOND. carbon_count below is a
            # count of atoms and can never recover it.
            if sub_info.get('unnameable'):
                return None
            prefix_name = sub_info.get('prefix_name')
            if prefix_name:
                sub_groups[prefix_name].append(locant)
                continue

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

    # WSD-01: fold in non-principal ring-FG prefixes (hydroxy/oxo/amino/...) so a
    # demoted functional group on a bicyclic parent is named, not dropped.
    if fg_prefixes:
        for fg in fg_prefixes:
            name = fg.get('name')
            locant = fg.get('locant')
            if name and locant is not None:
                sub_groups[name].append(locant)

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


def _build_bicyclo_principal_suffix(
    suffix_info: Optional[Dict]
) -> Tuple[str, bool]:
    """WSD-01 (Phase 175): format the principal-characteristic-group suffix for a
    von Baeyer (bicyclo) parent from a ``_detect_ring_functional_groups`` suffix
    dict ``{'suffix','locants','type'}``. Supersedes the old ketone-only builder
    so COOH / CHO / CN / amine / OH on a 2-ring system are all expressed as the
    correct suffix instead of collapsing to a phantom ``methyl``.

    Returns ``(suffix, elide_terminal_e)``:
      * inline groups (``one`` / ``ol`` / ``amine``) replace the parent's trailing
        ``e`` (``heptan-2-one``, ``heptan-2-amine``); the consonant-initial
        multiplied form (``dione`` / ``diol`` / ``diamine``) keeps the ``e``
        (``heptane-2,6-dione``) per IUPAC P-16.3.3.
      * appended groups (``carboxylic acid`` / ``carbaldehyde`` / ``carbonitrile``)
        attach to the full parent with a hyphen and never elide
        (``heptane-2-carboxylic acid``, ``heptane-2,3-dicarboxylic acid``).

    Seniority + multi-FG selection (the senior group becomes the suffix, the rest
    prefixes) is done upstream in ``_detect_ring_functional_groups`` per P-41;
    this is a pure formatter.
    """
    if not suffix_info:
        return "", False
    suffix_text = suffix_info.get('suffix')
    if not suffix_text:
        return "", False
    locants = sorted(suffix_info.get('locants', []))
    suffix_type = suffix_info.get('type', 'inline')
    count = len(locants)
    # Principal-group suffixes always take the SIMPLE multiplier (di/tri), never
    # bis/tris (those are for complex substituents).
    multiplier = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
    locant_str = ",".join(str(loc) for loc in locants)

    # P-63.1.2: elide the multiplier-final 'a' before '-ol' (tetra+ol -> tetrol);
    # _join_multiplied_suffix is a no-op for any other suffix ('carb...'/one/amine).
    full_suffix = _join_multiplied_suffix(multiplier, suffix_text)

    if suffix_type == 'appended':
        # consonant-initial 'carb...' -> never elide the parent 'e'.
        if locants:
            return f"-{locant_str}-{full_suffix}", False
        return f"-{full_suffix}", False

    # inline (one/ol/amine): elide the parent 'e' only before the bare,
    # vowel-initial suffix; the multiplied di*/tri* form is consonant-initial.
    if locants:
        suffix_str = f"-{locant_str}-{full_suffix}"
    else:
        suffix_str = f"-{full_suffix}"
    return suffix_str, (not multiplier)


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
    if parent_name is None:
        # A ring heteroatom has no replacement prefix in the governing table, so
        # the parent cannot be spelled. Refuse before decorating it with
        # substituents (which would concatenate onto a missing parent).
        return None

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
        # Generate substituted name with N-locants and C-locants.
        # WS-4 / BBR-RSFX: pass the principal group so a senior FG on the ring is
        # emitted as a suffix (-one/-ol/-amine), not a detachable prefix.
        return name_substituted_heterocycle(
            features.mol,
            features.principal_ring,
            parent_name,
            substituents,
            atom_to_locant,
            principal_group=getattr(features, 'principal_group', None)
        )

    # No substituents - return parent name directly
    return parent_name


def _amide_acyl_parent_locants(mol, amide_atoms) -> Optional[Dict[int, int]]:
    """Phase 177 WSB-02 (D-09, STEREO-03 hoisting fix): build an atom_to_locant
    restricted to the amide ACYL parent (the carbonyl carbon + its acyl carbon
    chain), EXCLUDING the nitrogen and the N-substituent.

    The general chain perception may pick the longer N-substituent chain as
    `features.atom_to_locant` (e.g. for `CC(=O)N[C@@H](C)CC` it maps the butyl
    chain), so `_inject_stereo_if_missing` would HOIST the N-substituent's
    stereocenter to the parent front as a spurious `(2S)-`. Restricting the
    parent collection to the acyl side keeps the substituent's descriptor in its
    own bracket (the substituent-naming path emits it) and prevents the hoist.

    Returns {acyl_atom_idx: 1-indexed locant} with the carbonyl carbon at locant
    1 (acid/amide numbering, P-66.1), or None when the acyl parent cannot be
    resolved (caller then falls back to the default features.* map).
    """
    if not amide_atoms:
        return None
    # Identify the carbonyl carbon: the C in amide_atoms bonded to BOTH a
    # chalcogen (=O/=S/=Se/=Te) and the amide nitrogen.
    carbonyl_c = None
    nitrogen = None
    amide_set = set(amide_atoms)
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        nbr_syms = {n.GetSymbol() for n in atom.GetNeighbors()}
        if 'N' in nbr_syms and (nbr_syms & {'O', 'S', 'Se', 'Te'}):
            carbonyl_c = idx
            for n in atom.GetNeighbors():
                if n.GetSymbol() == 'N':
                    nitrogen = n.GetIdx()
            break
    if carbonyl_c is None or nitrogen is None:
        return None
    # BFS the acyl carbon chain from the carbonyl carbon, never crossing the
    # nitrogen and never leaving carbon (the acyl parent is a carbon chain;
    # heteroatom branches are handled as substituents elsewhere).
    acyl_atoms = []
    seen = {nitrogen}
    stack = [carbonyl_c]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        if atom.GetSymbol() != 'C':
            continue
        acyl_atoms.append(cur)
        for n in atom.GetNeighbors():
            if n.GetIdx() not in seen and n.GetSymbol() == 'C':
                stack.append(n.GetIdx())
    if not acyl_atoms:
        return None
    # Number from the carbonyl carbon (locant 1) outward along the chain. For a
    # straight acyl chain this is the standard acid/amide numbering; the
    # ordering is a simple BFS distance which is sufficient here because the
    # restricted map's ONLY consumer is the parent stereo collection (whether a
    # PARENT carbon is stereogenic), not name construction.
    from collections import deque
    dist = {carbonyl_c: 1}
    q = deque([carbonyl_c])
    acyl_set = set(acyl_atoms)
    while q:
        cur = q.popleft()
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = n.GetIdx()
            if ni in acyl_set and ni not in dist:
                dist[ni] = dist[cur] + 1
                q.append(ni)
    return dist


def _find_amidine_carbon(mol, pg_atoms, chain=None):
    """D1: locate the amidine central carbon from the principal_group_atoms.

    The classifier may re-order the raw SMARTS match tuple, so identify the
    amidine C structurally: the sp2 carbon in the pg_atoms set that has exactly
    one DOUBLE-bonded N and at least one SINGLE-bonded N neighbour, and no other
    heteroatom double bond (excludes guanidine's 3-N carbon defensively — a
    guanidine C has two single-bonded N besides the =N, so it is still an
    amidine-shaped C, but guanidine never reaches this path because perception
    suppresses the amidine read on it).

    AM-5: a conjoined diamidine (CC(=N)NC(C)=N) has TWO amidine carbons; ``chain``
    (the principal chain) disambiguates the parent — a chain member is preferred,
    ties broken by lowest atom index so selection is deterministic (a bare set
    iteration order is not).
    """
    from rdkit import Chem
    candidate_atoms = set()
    for grp in pg_atoms:
        candidate_atoms.update(grp)
    _chain_set = set(chain or [])
    _candidates = []
    for idx in candidate_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        double_n = 0
        single_n = 0
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() != 'N':
                continue
            bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                double_n += 1
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                single_n += 1
        if double_n == 1 and single_n >= 1:
            _candidates.append(idx)
    if not _candidates:
        return None
    _candidates.sort(key=lambda i: (0 if i in _chain_set else 1, i))
    return _candidates[0]


def _amidines_share_one_n(mol, pg_atoms) -> bool:
    """AM-5 (P-66.4.1.6): True for exactly two amidine matches that intersect on
    a single shared N atom (a conjoined diamidine, R-C(=NH)-NH-C(=NH)-R'). Two
    amidines at opposite chain ends (NC(=N)CCC(=N)N -> butanediimidamide) share
    no atom and stay on the general path."""
    if not pg_atoms or len(pg_atoms) != 2:
        return False
    _a, _b = set(pg_atoms[0]), set(pg_atoms[1])
    _shared = _a & _b
    if len(_shared) != 1:
        return False
    return mol.GetAtomWithIdx(next(iter(_shared))).GetSymbol() == 'N'


def _hydrazide_n_sub_names(mol, n_atom_idx, exclude):
    """Wave2 T3d/ring-hydrazide shared collector: the C-substituent names on a
    hydrazide nitrogen, plus the atom indices they claim.

    Returns ``(names, atoms)``; ``(None, None)`` fail-closed when any
    substituent on the nitrogen is not a nameable pure-carbon fragment
    (N-N-N / N-O / unnameable branches). An unsubstituted nitrogen returns
    ``([], [])``.
    """
    from ..rules.benzene import _collect_pure_alkyl
    from .substituent_naming import name_substituent_fragment
    names, atoms = [], []
    for nb in mol.GetAtomWithIdx(n_atom_idx).GetNeighbors():
        ni = nb.GetIdx()
        if ni in exclude:
            continue
        if nb.GetSymbol() != 'C':
            return None, None  # N-N-N / N-O etc. out of scope
        alk, cc = _collect_pure_alkyl(mol, ni, set(exclude) | {n_atom_idx})
        if not alk or cc == 0:
            return None, None
        nm = name_substituent_fragment(mol, alk, ni, list(set(exclude) | {n_atom_idx}))
        if not nm:
            return None, None
        names.append(nm)
        atoms.extend(alk)
    return names, atoms


def _assemble_hydrazide_name(features: Any, style: str):
    """Wave2 T3d (P-66.3.3 / P-66.3.4 / P-66.3.5.1): assemble an acyclic
    N/N'-substituted (thio)hydrazide, or a hydrazinecarboxylic acid.

    Three shapes, all fail-closed (return None -> general path -> SELF-01):
      * acyl hydrazide  R-C(=O)-NH-NH2  -> {stem}hydrazide with N/N' locants
        (N = the nitrogen bonded to the C=O; N' = the terminal nitrogen).
        'CNNC(C)=O' -> N'-methylacetohydrazide; 'CN(N)C(C)=O' -> N-methyl...
      * thiohydrazide  R-C(=S)-NH-NH2  -> {stem}thiohydrazide (same N/N' model).
      * hydrazinecarboxylic acid  H2N-NH-C(=O)-OH  -> 'hydrazinecarboxylic acid'
        (the C bears =O + OH + the hydrazine N; no R carbon).
    """
    from ..rules.benzene import _collect_pure_alkyl
    mol = features.mol
    pg = features.principal_group
    pg_atoms = features.principal_group_atoms
    if not pg_atoms:
        return None

    # Locate the characteristic carbon: the C double-bonded to O (hydrazide,
    # hydrazinecarboxylic) or S (thiohydrazide) that also bears the -N-N-.
    match = pg_atoms[0]
    c_idx = None
    for a in match:
        at = mol.GetAtomWithIdx(a)
        if at.GetSymbol() != 'C':
            continue
        has_dbl_chalc = any(
            nb.GetSymbol() in ('O', 'S')
            and mol.GetBondBetweenAtoms(a, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nb in at.GetNeighbors())
        has_nn = any(nb.GetSymbol() == 'N' for nb in at.GetNeighbors())
        if has_dbl_chalc and has_nn:
            c_idx = a
            break
    if c_idx is None:
        return None
    c_atom = mol.GetAtomWithIdx(c_idx)

    # Classify the C: =O + -OH + N-N (no R carbon) = hydrazinecarboxylic acid;
    # =O/=S + N-N + one R carbon = (thio)hydrazide.
    dbl_o = dbl_s = oh_o = 0
    r_carbon = None
    n_inner = None
    for nb in c_atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
        sym = nb.GetSymbol()
        if sym == 'O' and b.GetBondTypeAsDouble() == 2.0:
            dbl_o += 1
        elif sym == 'S' and b.GetBondTypeAsDouble() == 2.0:
            dbl_s += 1
        elif sym == 'O' and b.GetBondTypeAsDouble() == 1.0 and nb.GetTotalNumHs() >= 1:
            oh_o += 1
        elif sym == 'C':
            r_carbon = nb.GetIdx()
        elif sym == 'N':
            n_inner = nb.GetIdx()
    if n_inner is None:
        return None
    n_inner_atom = mol.GetAtomWithIdx(n_inner)
    # the terminal N: the N bonded to n_inner that is not the C
    n_term = None
    for nb in n_inner_atom.GetNeighbors():
        if nb.GetSymbol() == 'N' and nb.GetIdx() != c_idx:
            n_term = nb.GetIdx()
    if n_term is None:
        return None

    # Collect C-substituents on each hydrazide N (fail-closed on non-C / unnameable).
    core = {c_idx, n_inner, n_term}
    if r_carbon is not None:
        # R chain atoms belong to the parent, not an N-substituent.
        pass

    # --- hydrazinecarboxylic acid (no R carbon; C = OH-bearing acid) ---
    if r_carbon is None and dbl_o == 1 and oh_o == 1 and dbl_s == 0:
        # N = the N bonded to the C (inner); N' = the terminal N (P-66.3.5.1).
        inner_subs, _ = _hydrazide_n_sub_names(mol, n_inner, core)
        term_subs, _ = _hydrazide_n_sub_names(mol, n_term, core)
        if inner_subs is None or term_subs is None:
            return None
        prefix = _format_hydrazide_nn_prefix(inner_subs, term_subs)
        if prefix is None:
            return None
        return f"{prefix}hydrazinecarboxylic acid"

    # --- (thio)hydrazide: needs an R carbon (the acyl/thioacyl parent) ---
    if r_carbon is None:
        return None
    if dbl_o == 1 and dbl_s == 0:
        suffix_word = 'hydrazide'
    elif dbl_s == 1 and dbl_o == 0:
        suffix_word = 'thiohydrazide'
    else:
        return None

    inner_subs, _ = _hydrazide_n_sub_names(mol, n_inner, core)
    term_subs, _ = _hydrazide_n_sub_names(mol, n_term, core)
    if inner_subs is None or term_subs is None:
        return None
    if not inner_subs and not term_subs:
        return None  # unsubstituted -> general path already correct

    # Build the {stem}(thio)hydrazide base via the SAME machinery the general
    # path uses for the bare hydrazide (stem + 'ane' + suffix w/ elision), so it
    # is 'ethanehydrazide' not the raw-stem 'ethhydrazide'. Mirror _assemble_
    # amidine_name. thiohydrazide's SUFFIX_FORMS already gives the right word.
    if not features.principal_chain:
        return None
    parent = _generate_chain_parent(features)
    suffix = _generate_suffix(features)
    if parent is None or suffix is None:
        return None
    base = _assemble_fragments([parent, suffix], style)
    if not base or suffix_word not in base:
        return None

    # Guard: the chain must carry nothing but the hydrazide C — no dropped subs.
    _grp = set(core)
    if r_carbon is not None:
        _grp.add(r_carbon)
    # the carbonyl / thiocarbonyl chalcogen is part of the FG, not a dropped sub
    for nb in c_atom.GetNeighbors():
        if nb.GetSymbol() in ('O', 'S') and mol.GetBondBetweenAtoms(
                c_idx, nb.GetIdx()).GetBondTypeAsDouble() == 2.0:
            _grp.add(nb.GetIdx())
    for nsub_n in (n_inner, n_term):
        for nb in mol.GetAtomWithIdx(nsub_n).GetNeighbors():
            if nb.GetSymbol() == 'C' and nb.GetIdx() != c_idx:
                alk, _ = _collect_pure_alkyl(mol, nb.GetIdx(), _grp | {nsub_n})
                if alk:
                    _grp.update(alk)
    chain_set = set(features.principal_chain)
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                if not set(sub_atoms).issubset(_grp | chain_set):
                    return None

    prefix = _format_hydrazide_nn_prefix(inner_subs, term_subs)
    if prefix is None:
        return None
    return f"{prefix}{base}"


def _format_hydrazide_nn_prefix(inner_subs, term_subs):
    """Wave2 T3d: build the italic-N / N' substituent prefix for a hydrazide.
    ``inner_subs`` sit on N (the nitrogen bonded to the C=O/C=S / acid C);
    ``term_subs`` on N' (the terminal nitrogen). P-66.3.3 / P-16.3.3:
    identical substituents share one multiplier ACROSS the two nitrogens
    ("N,N'-dimethyl...", BB 'N1,N'4-dimethylnaphthalene-1,4-dicarbohydrazide'
    style), distinct substituents are cited alphabetically
    ("N'-acetyl-N'-ethyl-N-methylbutanehydrazide", P-66.3.3.2)."""
    from .naming_utils import _wrap_n_substituent, get_multiplier_prefix, alpha_sort_key
    by_name: Dict[str, List[str]] = {}
    for locant, subs in (("N", inner_subs), ("N'", term_subs)):
        for nm in subs or []:
            by_name.setdefault(nm, []).append(locant)
    if not by_name:
        return ""
    parts = []  # (sort_key, rendered)
    for nm, locs in by_name.items():
        locs.sort(key=lambda l: (l.count("'"), l))  # N before N'
        wrapped = _wrap_n_substituent(nm)
        loc_str = ",".join(locs)
        if len(locs) == 1:
            parts.append((alpha_sort_key(nm), f"{loc_str}-{wrapped}"))
        else:
            mp = get_multiplier_prefix(len(locs), nm)
            parts.append((alpha_sort_key(nm), f"{loc_str}-{mp}{wrapped}"))
    parts.sort(key=lambda p: p[0])
    # No trailing hyphen: the last N-substituent prepends directly to the base
    # ('N-ethyl-N'-methylacetohydrazide', 'N'-methylethanehydrazide').
    return "-".join(p[1] for p in parts)


def _assemble_ring_hydrazide_name(features: Any, style: str):
    """Wave2 ring-hydrazide (P-66.3.1.1 / P-66.3.1.2.1 / P-66.3.3): assemble a
    ring-parent N/N'-substituted (thio)hydrazide — N'-methylbenzohydrazide,
    N-methylcyclohexanecarbohydrazide, N'-methylpyridine-4-carbohydrazide.

    Builds the base name by deleting the N-substituents (RWMol surgery) and
    re-entering the full namer — the substitutive-oxime re-entry pattern — then
    prepends the italic-N/N' prefix (N = the nitrogen bonded to the acyl C,
    N' = the terminal nitrogen). Recursion terminates because the
    des-N-substituted molecule has no N-substituents, for which this assembler
    returns None (the retained/ring paths already name it correctly).

    FAIL-CLOSED scope (returns None -> ring paths emit the bare suffix and the
    validity gate keeps the result honest):
      * exactly one (thio)hydrazide match; no stereocenters / double-bond
        stereo (the shared fragment namers do not emit descriptors);
      * the acyl C attaches to a ring CARBON (N-acyl ring hydrazides such as
        N-substituted piperidine-1-carbohydrazide are deferred);
      * every heavy atom is claimed by ring system + hydrazide core + N-subs.
        A ring-substituted AND N-substituted combination needs the combined
        alphanumeric ordering of italic-N + numeric locants (P-14.5.2), which
        the prefix-prepend model cannot express — deferred, stays unknown;
      * the recursive base name is a bare (thio)hydrazide ring parent
        ('benzohydrazide' retained per P-66.3.1.2.1, or
        '{ring}(-<loc>-)carbo(thio)hydrazide'), matching the suffix family of
        the perceived group.
    """
    import re
    from rdkit import Chem
    mol = features.mol
    pg_atoms = features.principal_group_atoms
    if not pg_atoms or len(pg_atoms) != 1:
        return None
    if getattr(features, 'stereocenters', None) or \
            getattr(features, 'double_bond_stereo', None):
        return None

    # Locate the characteristic C (double-bonded chalcogen + N-N), then
    # classify its neighbors. Mirrors _assemble_hydrazide_name's locator.
    match = pg_atoms[0]
    c_idx = None
    for a in match:
        at = mol.GetAtomWithIdx(a)
        if at.GetSymbol() != 'C':
            continue
        has_dbl_chalc = any(
            nb.GetSymbol() in ('O', 'S')
            and mol.GetBondBetweenAtoms(a, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nb in at.GetNeighbors())
        has_nn = any(nb.GetSymbol() == 'N' for nb in at.GetNeighbors())
        if has_dbl_chalc and has_nn:
            c_idx = a
            break
    if c_idx is None:
        return None
    c_atom = mol.GetAtomWithIdx(c_idx)

    chalc = None
    suffix_word = None
    r_attach = None
    n_inner = None
    for nb in c_atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
        sym = nb.GetSymbol()
        if sym in ('O', 'S') and b.GetBondTypeAsDouble() == 2.0:
            if chalc is not None:
                return None
            chalc = nb.GetIdx()
            suffix_word = 'hydrazide' if sym == 'O' else 'thiohydrazide'
        elif sym == 'C' and b.GetBondTypeAsDouble() == 1.0:
            if r_attach is not None:
                return None
            r_attach = nb.GetIdx()
        elif sym == 'N' and b.GetBondTypeAsDouble() == 1.0:
            # the inner hydrazide N must itself bear the terminal N
            if n_inner is not None:
                return None
            if not any(x.GetSymbol() == 'N' for x in nb.GetNeighbors()):
                return None
            n_inner = nb.GetIdx()
        else:
            return None
    if chalc is None or n_inner is None or r_attach is None:
        return None
    # ring-C attachment only (the acyclic case belongs to the chain assembler)
    if mol.GetRingInfo().NumAtomRings(r_attach) == 0:
        return None

    n_term = None
    for nb in mol.GetAtomWithIdx(n_inner).GetNeighbors():
        if nb.GetSymbol() == 'N' and nb.GetIdx() != c_idx:
            if n_term is not None:
                return None
            n_term = nb.GetIdx()
    if n_term is None:
        return None

    core = {c_idx, n_inner, n_term}
    inner_subs, inner_atoms = _hydrazide_n_sub_names(mol, n_inner, core)
    term_subs, term_atoms = _hydrazide_n_sub_names(mol, n_term, core)
    if inner_subs is None or term_subs is None:
        return None
    if not inner_subs and not term_subs:
        return None  # bare ring hydrazide -> retained/ring paths already correct

    # Coverage guard: ring-system atoms ONLY (no decorations) + hydrazide core
    # + N-substituent atoms must claim every heavy atom, so a ring substituent
    # (or any second FG) leaves an unclaimed atom and we decline.
    ri = mol.GetRingInfo()
    ring_sys = set()
    stack = [r_attach]
    while stack:
        i = stack.pop()
        if i in ring_sys:
            continue
        ring_sys.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            ni = nb.GetIdx()
            if ni not in ring_sys and ri.NumAtomRings(ni) > 0:
                stack.append(ni)
    claimed = (core | {chalc} | ring_sys
               | set(inner_atoms) | set(term_atoms))
    if any(a.GetIdx() not in claimed for a in mol.GetAtoms()):
        return None

    # Base = the des-N-substituted molecule, named through the full pipeline
    # (retained lookup gives 'benzohydrazide'; ring suffix paths give
    # '{ring}carbo(thio)hydrazide'). Deterministic: canonical SMILES in,
    # descending-index atom removal.
    nsub_atoms = sorted(set(inner_atoms) | set(term_atoms), reverse=True)
    try:
        em = Chem.RWMol(mol)
        for a in nsub_atoms:
            em.RemoveAtom(a)
        base_mol = em.GetMol()
        Chem.SanitizeMol(base_mol)
        base_smiles = Chem.MolToSmiles(base_mol)
    except Exception:
        return None
    from ..namer import name_compound as _name_compound
    try:
        base = _name_compound(base_smiles, style=style)
    except Exception:
        return None
    if not base:
        return None

    # Validate the base shape: a bare ring (thio)hydrazide parent of the SAME
    # suffix family — no substituent prefixes, brackets or multiplied suffixes.
    if base == 'benzohydrazide':
        base_is_thio = False
    else:
        m = re.match(
            r"^(?:\d+(?:,\d+)*-)?(?:\d+H-)?[a-z]+(?:-\d+-)?carbo(thio)?hydrazide$",
            base)
        if not m:
            return None
        base_is_thio = bool(m.group(1))
    if base_is_thio != (suffix_word == 'thiohydrazide'):
        return None

    prefix = _format_hydrazide_nn_prefix(inner_subs, term_subs)
    if not prefix:
        return None
    joiner = "-" if base[:1].isdigit() else ""
    return f"{prefix}{joiner}{base}"


def _assemble_amidine_name(features: Any, style: str):
    """D1 (P-66.4.1 / P-66.4.1.2 / P-66.4.1.3.1): assemble an acyclic amidine
    name with N/N'-substituent locants (e.g. N-methylethanimidamide,
    N,N'-dimethylethanimidamide).

    Mirrors the unsaturated-amide pattern: build the bare imidamide base from the
    chain parent + '-imidamide' suffix, then prepend the italic-N/N' prefix.

    Fail-closed: returns None (caller falls through to the general path, whose
    output SELF-01 then keeps honest) whenever
      * the amidine carbon or its N/N'-substituents cannot be resolved cleanly,
      * a substituent nitrogen carries a non-nameable fragment, or
      * the chain carries an extra substituent beyond the amidine group itself
        (so a wrong 'dropped-substituent' name is never emitted).
    """
    from ..rules.benzene import _detect_amidine_n_substituents, _build_amidine_n_prefix

    mol = features.mol
    pg_atoms = features.principal_group_atoms
    if not pg_atoms:
        return None

    c_idx = _find_amidine_carbon(mol, pg_atoms, features.principal_chain)
    if c_idx is None:
        return None

    # The chain (parent) atoms carry the amidine; exclude them so the detector
    # walks only the true N-substituents.
    parent_atoms = set(features.principal_chain or [])
    parent_atoms.add(c_idx)
    n_subs = _detect_amidine_n_substituents(mol, c_idx, parent_atoms)
    if n_subs is None:
        return None
    if not n_subs:
        # Unsubstituted amidine -> the existing general path already produces the
        # correct bare '-imidamide' (e.g. ethanimidamide); nothing to add here.
        return None

    # Collect the amidine group's atoms (C + both N + all N-substituent carbons)
    # so we can verify the chain has NO extra (genuine) substituent. If it does,
    # fall through rather than risk dropping it.
    c_atom = mol.GetAtomWithIdx(c_idx)
    amidine_n_idxs = [nbr.GetIdx() for nbr in c_atom.GetNeighbors()
                      if nbr.GetSymbol() == 'N']
    amidine_group_atoms = set(amidine_n_idxs) | {c_idx}
    from ..rules.benzene import _collect_pure_alkyl
    for n_idx in amidine_n_idxs:
        for sub in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            if sub.GetIdx() == c_idx or sub.GetIdx() in parent_atoms:
                continue
            # Wave2 T3d (amidoxime): the N'-O(H)/O-alkyl is part of the amidine
            # group (named as N'-hydroxy/N'-alkyloxy by _detect_amidine_n_
            # substituents), so it must be counted here or the extra-substituent
            # guard below wrongly treats it as a dropped chain substituent.
            if sub.GetSymbol() == 'O':
                amidine_group_atoms.add(sub.GetIdx())
                for o_nbr in sub.GetNeighbors():
                    if o_nbr.GetIdx() == n_idx:
                        continue
                    o_alkyl, _ = _collect_pure_alkyl(
                        mol, o_nbr.GetIdx(),
                        set(parent_atoms) | {c_idx, n_idx, sub.GetIdx()},
                    )
                    if o_alkyl:
                        amidine_group_atoms.update(o_alkyl)
                continue
            # AM-5 (P-66.4.1.6): an imidoyl-carbon N-substituent R-C(=NH)- (named
            # '{stem}animidoyl' by the detector) — its imino =N and alkyl carbons
            # belong to the amidine group, not a dropped chain substituent.
            if sub.GetSymbol() == 'C':
                _sub_imino = [
                    x for x in sub.GetNeighbors()
                    if x.GetSymbol() == 'N'
                    and mol.GetBondBetweenAtoms(
                        sub.GetIdx(), x.GetIdx()).GetBondTypeAsDouble() == 2.0
                ]
                if (len(_sub_imino) == 1
                        and not [y for y in _sub_imino[0].GetNeighbors()
                                 if y.GetIdx() != sub.GetIdx()]):
                    amidine_group_atoms.add(sub.GetIdx())
                    amidine_group_atoms.add(_sub_imino[0].GetIdx())
                    _im_alk, _ = _collect_pure_alkyl(
                        mol, sub.GetIdx(),
                        set(parent_atoms) | {c_idx, n_idx, _sub_imino[0].GetIdx()},
                    )
                    if _im_alk:
                        amidine_group_atoms.update(_im_alk)
                    continue
            alkyl_atoms, _ = _collect_pure_alkyl(
                mol, sub.GetIdx(), set(parent_atoms) | {c_idx, n_idx}
            )
            if alkyl_atoms:
                amidine_group_atoms.update(alkyl_atoms)

    # Any chain substituent branch not fully inside the amidine group is a
    # genuine extra substituent -> fail closed.
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                if not set(sub_atoms).issubset(amidine_group_atoms):
                    return None

    # Build the bare imidamide base from parent + suffix.
    parent = _generate_chain_parent(features)
    suffix = _generate_suffix(features)
    if parent is None or suffix is None:
        return None
    base_name = _assemble_fragments([parent, suffix], style)
    if not base_name or 'imidamide' not in base_name:
        return None

    n_prefix = _build_amidine_n_prefix(n_subs)
    if not n_prefix:
        return None
    return f"{n_prefix}{base_name}"


def _assemble_geminal_dicarboximidamide_name(features: Any, style: str):
    """W2E-D4 (P-66.4.1.4.2 / P-16.9.1): assemble an N-substituted GEMINAL
    ring dicarboximidamide with per-group primed + superscript N-locants.

    A geminal ring dicarboximidamide is a saturated (or unsaturated) ring
    carbon bearing TWO amidine groups -C(=NH)-NH2, named
    ``{ring}-{loc},{loc}-dicarboximidamide`` (P-66.4.1). When one or both of
    the four amidine nitrogens carries a substituent, IUPAC cites the
    substituent with an *italic-N* locant that pairs a **prime count**
    (which of the two amidine groups it belongs to) with a **superscript
    numeral** (the ring locant of the geminal carbon), per P-66.4.1.4.2 and
    P-16.9.1. For two geminal groups both at ring locant ``L`` the four N
    tokens are:

        N{L}    (0 primes)  -> group-1 amino   (single-bonded  -NH-)
        N'{L}   (1 prime)   -> group-1 imino    (double-bonded  =N-)
        N''{L}  (2 primes)  -> group-2 amino
        N'''{L} (3 primes)  -> group-2 imino

    matching OPSIN's ``N1 / N'1 / N''1 / N'''1`` locant grammar (verified
    2026-07-10). The ASCII rendering keeps the superscript numeral inline
    (``N''1``), so ``CCNC(=N)C1(C(=N)N(C)C)CCCCC1`` ->
    ``N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide``.

    Group -> prime assignment follows the lowest-locant rule (P-14.3.2 /
    P-31.1.4.3.4): the assignment whose sorted N-locant multiset is lowest at
    the first point of difference wins (so the more-substituted group takes
    the lower prime -> more low locants). Substituent CITATION order is
    alphanumerical (P-14.5.2).

    Fail-closed: returns ``None`` (caller falls through -> general path ->
    SELF-01 keeps it honest, never a wrong name) unless ALL of these hold:
      * principal group is 'amidine' with exactly TWO matches,
      * both amidine carbons attach to the SAME single ring atom (geminal),
      * the ring parent + '-dicarboximidamide' base assembles cleanly,
      * every N-substituent is a nameable pure fragment,
      * at least one nitrogen actually carries a substituent (the
        unsubstituted case is already named by the general path).
    """
    from rdkit import Chem
    from ..rules.benzene import _detect_amidine_n_substituents
    from ..assembly.naming_utils import (
        _wrap_n_substituent,
        get_multiplier_prefix,
    )

    mol = features.mol
    pg_atoms = features.principal_group_atoms
    if features.principal_group != 'amidine' or not pg_atoms:
        return None
    if len(pg_atoms) != 2:
        return None
    if not getattr(features, 'is_cyclic', False):
        return None

    # Locate the two amidine carbons structurally (a pg_atoms tuple may be
    # re-ordered by the classifier). Each must be a C with exactly one =N and
    # one -NH- and no extra heteroatom bonds.
    amidine_carbons = []
    for grp in pg_atoms:
        c_idx = None
        for idx in grp:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != 'C':
                continue
            dbl = sgl = 0
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() != 'N':
                    continue
                b = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if b.GetBondType() == Chem.BondType.DOUBLE:
                    dbl += 1
                elif b.GetBondType() == Chem.BondType.SINGLE:
                    sgl += 1
            if dbl == 1 and sgl == 1:
                c_idx = idx
                break
        if c_idx is None:
            return None
        amidine_carbons.append(c_idx)
    if len(set(amidine_carbons)) != 2:
        return None

    # Geminal check: both amidine carbons must attach to ONE common ring atom.
    ring_info = mol.GetRingInfo()
    common = None
    for a in mol.GetAtomWithIdx(amidine_carbons[0]).GetNeighbors():
        if a.GetIdx() in [n.GetIdx() for n in
                          mol.GetAtomWithIdx(amidine_carbons[1]).GetNeighbors()]:
            common = a.GetIdx()
            break
    if common is None or ring_info.NumAtomRings(common) == 0:
        return None
    # The amidine carbons themselves must be acyclic (ring-attached, not in-ring)
    # and must not carry any heavy neighbour besides the ring atom + their two N.
    ring_atom_set = set()
    for ring in ring_info.AtomRings():
        if common in ring:
            ring_atom_set.update(ring)
    for c_idx in amidine_carbons:
        if ring_info.NumAtomRings(c_idx) != 0:
            return None
        heavy = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors()]
        n_heavy = [i for i in heavy
                   if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        non_n = [i for i in heavy if i not in n_heavy]
        if len(n_heavy) != 2 or non_n != [common]:
            return None

    # Perceive N-substituents per amidine group. parent_atoms excludes the ring
    # + both amidine carbons so the detector walks only true N-substituents.
    base_parent = set(ring_atom_set) | set(amidine_carbons)
    groups_subs = []   # list per amidine carbon of [(local_nlocant, name), ...]
    any_sub = False
    for c_idx in amidine_carbons:
        subs = _detect_amidine_n_substituents(mol, c_idx, base_parent)
        if subs is None:
            return None
        if subs:
            any_sub = True
        groups_subs.append(subs)
    if not any_sub:
        return None  # unsubstituted -> general path already correct

    # Build the bare dicarboximidamide base via the general fragment pipeline
    # (same path the unsubstituted case uses); it embeds the ring locants.
    parent = _generate_ring_parent(features)
    suffix = _generate_suffix(features)
    if parent is None or suffix is None or not parent.text:
        return None
    base_name = _assemble_fragments([parent, suffix], style)
    if not base_name or 'dicarboximidamide' not in base_name:
        return None

    # The geminal carbon's ring locant (all suffix instances share it here).
    suffix_locants = [loc for loc in (suffix.locants or ()) if loc]
    if not suffix_locants:
        return None
    ring_loc = suffix_locants[0]
    # Guard: this builder only handles the truly geminal case where BOTH
    # carboximidamide suffixes sit on the same ring locant.
    if any(loc != ring_loc for loc in suffix_locants):
        return None

    # Build, for each group->index permutation, the composite N-locant tokens
    # and pick the assignment with the lowest sorted locant multiset (lowest
    # locants rule, P-14.3.2). Prime count = 2*group_index (amino) or
    # 2*group_index + 1 (imino); the local nlocant 'N' -> amino, "N'" -> imino.
    import itertools

    def _tokens_for(assignment):
        """assignment: tuple mapping amidine-carbon position -> group index.
        Returns list of (prime_count, token, sub_name)."""
        out = []
        for carbon_pos, grp_index in enumerate(assignment):
            for local_nloc, sub_name in groups_subs[carbon_pos]:
                base_primes = 2 * grp_index + (1 if local_nloc == "N'" else 0)
                token = "N" + ("'" * base_primes) + str(ring_loc)
                out.append((base_primes, token, sub_name))
        return out

    best_tokens = None
    best_key = None
    for assignment in itertools.permutations(range(len(amidine_carbons))):
        toks = _tokens_for(assignment)
        # locant multiset sorted by (prime_count) at first point of difference
        key = tuple(sorted(t[0] for t in toks))
        if best_key is None or key < best_key:
            best_key = key
            best_tokens = toks

    if not best_tokens:
        return None

    # Render: collapse identical substituent names sharing a multiplier
    # (N1,N1-dimethyl); cite distinct substituents alphanumerically (P-14.5.2).
    from collections import defaultdict
    by_name = defaultdict(list)
    for prime_count, token, sub_name in best_tokens:
        by_name[sub_name].append((prime_count, token))

    segments = []  # (alpha_key, rendered)
    for sub_name, toks in by_name.items():
        toks_sorted = sorted(toks, key=lambda t: t[0])
        loc_str = ",".join(tok for _p, tok in toks_sorted)
        count = len(toks_sorted)
        if count > 1:
            mp = get_multiplier_prefix(count, sub_name)
            rendered = f"{loc_str}-{mp}{_wrap_n_substituent(sub_name)}"
        else:
            rendered = f"{loc_str}-{_wrap_n_substituent(sub_name)}"
        segments.append((alpha_sort_key(sub_name), rendered))

    segments.sort(key=lambda s: s[0])
    n_prefix = "-".join(rendered for _k, rendered in segments)
    if not n_prefix:
        return None
    return f"{n_prefix}{base_name}"


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
    # WR-2: reset the transient side-channel BEFORE any early return. Only the
    # unsaturated-chain branch (below) sets ``features._amide_tree``; the
    # saturated and ring-attached branches return without setting it. name_amide
    # (handlers/amide.py) reads ``getattr(features, "_amide_tree", None)`` and
    # falls back to a coarse node only when None. Resetting at function entry
    # guarantees the read can never observe a STALE structured tree from a prior
    # molecule if a MolecularFeatures instance is ever reused. This touches only
    # the post-hoc tree path, never the returned name string (byte-identical).
    features._amide_tree = None
    from ..rules.amides import (
        name_amide, is_ring_attached_amide, get_amide_type,
        get_n_substituents, format_n_substitution,
    )

    # Phase 163 chalcogen amides funnel into this pipeline; map the
    # principal_group to its IUPAC PIN suffix form. Regular amides use the
    # default "amide" suffix.
    _CHALCOGEN_AMIDE_SUFFIX_FORMS = {
        'thioamide': 'thioamide',
        'selenoamide': 'selenoamide',
        'telluroamide': 'telluroamide',
    }
    amide_suffix_form = _CHALCOGEN_AMIDE_SUFFIX_FORMS.get(
        features.principal_group, 'amide',
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
        base_name = name_amide(mol, amide_atoms, suffix_form=amide_suffix_form)
        if base_name:
            # Add non-principal group prefixes (halogens, hydroxy, etc.)
            # Note: NameFragment.text already includes locants -- use directly.
            prefixes = _generate_prefixes(features)
            if prefixes:
                prefix_parts = []
                for p in sorted(prefixes, key=lambda x: alpha_sort_key(x.text)):
                    # P-14.3.4 (Phase 171 BBR-ASM, DEF-4): render the prefix LOCANT.
                    # _generate_prefixes is inconsistent — alkyl prefixes embed the
                    # locant in .text ('3-methyl') while FG prefixes keep it separate
                    # ('chloro', locants=(2,)). The old .text-only append silently
                    # dropped the FG locant (chloroacetamide -> PIN 2-chloroacetamide;
                    # 3-chloropropanamide). Prepend .locants only when text lacks one.
                    if p.locants and not str(p.text)[:1].isdigit():
                        _loc = ",".join(str(_l) for _l in p.locants)
                        prefix_parts.append(f"{_loc}-{p.text}")
                    else:
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
        base_name = name_amide(mol, amide_atoms, suffix_form=amide_suffix_form)
        if base_name:
            # STEREO-03 (Phase 177 WSB-02 D-09): restrict the parent-level stereo
            # collection to the ACYL parent so an N-substituent's stereocenter is
            # NOT hoisted to the parent front (the substituent emits its own
            # descriptor in its bracket). Falls back to features.* when the acyl
            # parent cannot be resolved.
            _acyl_locants = _amide_acyl_parent_locants(mol, amide_atoms)

            # Wave2 T5b: when the acyl carbon is OFF the features principal
            # chain (the N-side chain won parent selection), the features-
            # level prefixes carry locants of the WRONG parent — appending
            # them double-expresses substituents already named inside the
            # N-substituent bracket ('2-bromo-4-chloro-N-(2-bromo-4-chloro-
            # pentan-3-yl)acetamide', a different molecule). base_name (acyl
            # parent + N + located N-substituents) must then account for
            # EVERY heavy atom; if it does, emit it bare; if not, decline
            # (fail-closed) rather than attach mis-locanted prefixes.
            _acyl_c = next(
                (i for i in amide_atoms
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                 and any(
                     nb.GetSymbol() in ('O', 'S', 'Se', 'Te')
                     and mol.GetBondBetweenAtoms(i, nb.GetIdx())
                            .GetBondTypeAsDouble() == 2.0
                     for nb in mol.GetAtomWithIdx(i).GetNeighbors())),
                None,
            )
            _pchain = set(features.principal_chain or [])
            if _acyl_c is not None and _pchain and _acyl_c not in _pchain:
                _n_idx = next(
                    (i for i in amide_atoms
                     if mol.GetAtomWithIdx(i).GetSymbol() == 'N'), None)
                _covered = set(amide_atoms)
                _stack = [_acyl_c]
                while _stack:
                    _a = _stack.pop()
                    if _a in _covered and _a != _acyl_c:
                        continue
                    _covered.add(_a)
                    for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                        _ni = _nb.GetIdx()
                        if (_nb.GetAtomicNum() > 1 and _ni != _n_idx
                                and _ni not in _covered):
                            _stack.append(_ni)
                for _s in get_n_substituents(mol, amide_atoms):
                    _covered |= set(_s.get('atoms') or [])
                _heavy = {
                    a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
                }
                if _covered == _heavy:
                    # AM-2 ROOT-1 (P-66.1.7, plan P1AM Task 10): atom-coverage is
                    # NOT name-coverage — name_amide never enumerates acyl-CHAIN
                    # substituents, so any heavy atom hanging off the acyl parent
                    # (beyond the amide N + its substituents + the acyl =O/chain)
                    # is silently dropped. Enumerate them here and prepend
                    # located prefixes; decline on any un-nameable branch.
                    from ..perception.functional_groups import (
                        detect_functional_groups as _detect_fgs_am2,
                    )
                    from ..rules.polyfunctional import (
                        get_fg_prefix_form as _get_fg_prefix_am2,
                        format_fg_prefix as _fmt_fg_prefix_am2,
                    )
                    _acyl_atom_set = set(_acyl_locants.keys())
                    _named_atoms = set(amide_atoms) | _acyl_atom_set
                    for _s in get_n_substituents(mol, amide_atoms):
                        _named_atoms |= set(_s.get('atoms') or [])
                    # simple FG prefixes whose CENTER lies on an acyl-chain atom
                    # (amino/hydroxy/halo/... on the acyl parent, e.g. the 2-amino
                    # of 2-amino-N-...-acetamide). Center = the atom bonded to a
                    # chain carbon; render via the shared FG-prefix machinery.
                    _acyl_prefixes = []  # (locant:int, alpha_name:str, rendered)
                    _acyl_ok = True
                    _fg_by_center = {}
                    _detected = _detect_fgs_am2(mol)
                    _acyl_principal_chain = sorted(_acyl_atom_set)
                    for _ci in list(_acyl_locants.keys()):
                        _cl = _acyl_locants[_ci]
                        for _nb in mol.GetAtomWithIdx(_ci).GetNeighbors():
                            _bi = _nb.GetIdx()
                            if _bi in _named_atoms or _nb.GetAtomicNum() <= 1:
                                continue
                            # (a) simple FG-prefix branch (amino/hydroxy/halo/...)
                            _pfx_name = None
                            if _nb.GetSymbol() in ('N', 'O', 'S', 'F', 'Cl',
                                                   'Br', 'I') and _nb.GetDegree() == 1:
                                for _fgn, _matches in _detected.items():
                                    for _m in _matches:
                                        if _m and _m[0] == _bi:
                                            _pf = _get_fg_prefix_am2(
                                                _fgn, mol, _m,
                                                _acyl_principal_chain)
                                            if _pf:
                                                _pfx_name = _pf
                                                break
                                    if _pfx_name:
                                        break
                                # fallback: a lone terminal N/O/halogen prefix
                                if _pfx_name is None:
                                    _pfx_name = {
                                        'N': 'amino', 'O': 'hydroxy',
                                        'F': 'fluoro', 'Cl': 'chloro',
                                        'Br': 'bromo', 'I': 'iodo',
                                        'S': 'sulfanyl',
                                    }.get(_nb.GetSymbol())
                                if _pfx_name is None:
                                    _acyl_ok = False
                                    break
                                _acyl_prefixes.append(
                                    (_cl, _pfx_name,
                                     _fmt_fg_prefix_am2(_pfx_name, [_cl], 1)))
                                _named_atoms.add(_bi)
                            else:
                                # (b) carbon (or complex) branch -> _name_r_group
                                _rn = _name_r_group(
                                    mol, _bi, exclude_atoms=_acyl_atom_set)
                                if not _rn:
                                    _acyl_ok = False
                                    break
                                _acyl_prefixes.append(
                                    (_cl, _rn,
                                     _fmt_fg_prefix_am2(_rn, [_cl], 1)))
                                # BFS the branch atoms into _named_atoms
                                _bstack = [_bi]
                                while _bstack:
                                    _ba = _bstack.pop()
                                    if _ba in _named_atoms:
                                        continue
                                    _named_atoms.add(_ba)
                                    for _bnb in mol.GetAtomWithIdx(_ba).GetNeighbors():
                                        if (_bnb.GetIdx() not in _named_atoms
                                                and _bnb.GetIdx() not in _acyl_atom_set
                                                and _bnb.GetAtomicNum() > 1):
                                            _bstack.append(_bnb.GetIdx())
                        if not _acyl_ok:
                            break
                    # every heavy atom must now be accounted for
                    if not _acyl_ok or _named_atoms != _heavy:
                        return "amide"  # fail-closed fallthrough, as before
                    if _acyl_prefixes:
                        _acyl_prefixes.sort(
                            key=lambda t: (alpha_sort_key(t[1]), t[0]))
                        _pfx = "-".join(r for _l, _n, r in _acyl_prefixes)
                        _sep = "-" if base_name[:1] == "N" else ""
                        base_name = f"{_pfx}{_sep}{base_name}"
                    return _inject_stereo_if_missing(
                        features, base_name, atom_to_locant=_acyl_locants,
                    )
                return "amide"
            # Add non-principal group prefixes (halogens, hydroxy, etc.)
            # Note: _generate_prefixes returns NameFragments whose .text
            # already includes locants (e.g., "2-methyl"), so use .text
            # directly -- do NOT re-prepend locants from .locants field.
            prefixes = _generate_prefixes(features)
            if prefixes:
                prefix_parts = []
                for p in sorted(prefixes, key=lambda x: alpha_sort_key(x.text)):
                    # P-14.3.4 (Phase 171 BBR-ASM, DEF-4): render the prefix LOCANT.
                    # _generate_prefixes is inconsistent — alkyl prefixes embed the
                    # locant in .text ('3-methyl') while FG prefixes keep it separate
                    # ('chloro', locants=(2,)). The old .text-only append silently
                    # dropped the FG locant (chloroacetamide -> PIN 2-chloroacetamide;
                    # 3-chloropropanamide). Prepend .locants only when text lacks one.
                    if p.locants and not str(p.text)[:1].isdigit():
                        _loc = ",".join(str(_l) for _l in p.locants)
                        prefix_parts.append(f"{_loc}-{p.text}")
                    else:
                        prefix_parts.append(p.text)
                if prefix_parts:
                    prefix_str = "-".join(prefix_parts)
                    # Insert hyphen before N-locant prefix (base_name may
                    # start with "N-" or "N,N-" from name_amide())
                    sep = "-" if base_name[:1] == "N" else ""
                    return _inject_stereo_if_missing(
                        features, f"{prefix_str}{sep}{base_name}",
                        atom_to_locant=_acyl_locants,
                    )
            return _inject_stereo_if_missing(
                features, base_name, atom_to_locant=_acyl_locants,
            )

    # Unsaturated chain amides: use general assembly fragments for correct stereo + unsaturation
    fragments = []

    # Generate parent name (chain stem with unsaturation markers)
    if features.principal_chain:
        parent = _generate_chain_parent(features)
    else:
        # Unexpected for chain amide, but fallback gracefully
        base_name = name_amide(mol, amide_atoms, suffix_form=amide_suffix_form)
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
    final_name = base_name
    if amide_type in ("secondary", "tertiary"):
        n_subs = get_n_substituents(mol, amide_atoms)
        n_prefix = format_n_substitution(n_subs)
        if n_prefix:
            final_name = f"{n_prefix}{base_name}"

    # Phase 165 SCORE-01 (Pitfall 2): stash a STRUCTURED tree from the chain
    # fragment list, with fragment_legacy overridden to the FINAL (post-N-prefix)
    # string so name_tree_to_string round-trips byte-identically for BOTH
    # unsubstituted and N-substituted unsaturated-chain amides (the N-prefix is
    # carried in fragment_legacy, not dropped). The amide handler reads
    # features._amide_tree and attaches it via the pool tree= carry.
    import dataclasses as _dc
    from .name_tree_builder import fragments_to_tree
    features._amide_tree = _dc.replace(
        fragments_to_tree(fragments, class_id="amide", section_cite="P-66.1"),
        fragment_legacy=final_name,
    )
    return final_name


def _walk_amine_n_substituents(
    mol: Any, n_idx: int, chain_set: set, stop_set: Optional[set] = None,
    hetero_aware: bool = False,
) -> list:
    """Walk the N-substituent fragments off one amine nitrogen and return
    their prefix names (e.g. ['methyl'], ['ethyl'], ['phenyl'], ['methyl','methyl']).

    Extracted verbatim from the original single-N body of _assemble_amine_name
    so both the single-N and the multi-N (di/poly-amine) paths share one
    implementation. `chain_set` gates only the ENTRY neighbour (a carbon on the
    principal chain/ring is not the start of an N-substituent), exactly as the
    original code did — traversal then follows freely (a benzyl walks through
    the ring even when the ring is the parent). `stop_set` (multi-N path only,
    D2 spec risk 3) HARD-BLOCKS traversal at the OTHER principal amine
    nitrogens so one N's branch is never re-attributed to another; the chain
    carbons between amines are already excluded by chain_set at entry.
    """
    from collections import deque
    if stop_set is None:
        stop_set = set()
    nitrogen = mol.GetAtomWithIdx(n_idx)
    n_subs: list = []
    # Find N-substituents: carbon neighbors of N that are NOT on the principal chain
    for nbr in nitrogen.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in chain_set:
            continue
        if nbr_idx in stop_set:
            continue
        if nbr.GetSymbol() == 'H':
            continue
        # BFS to get the substituent fragment
        visited = set()
        queue = deque([nbr_idx])
        frag = []
        while queue:
            a = queue.popleft()
            if a in visited or a == n_idx or a in stop_set:
                continue
            visited.add(a)
            frag.append(a)
            for nn in mol.GetAtomWithIdx(a).GetNeighbors():
                nn_i = nn.GetIdx()
                if nn_i not in visited and nn_i != n_idx and nn_i not in stop_set:
                    queue.append(nn_i)
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

            # P-62.2.4.1.3 (plan P1AM Task 9): in the COMPLEX-polyamine path a
            # demoted amine N lives INSIDE this branch (-CH2-NH2 = aminomethyl,
            # -CH2CH2-NH2 = 2-aminoethyl). get_alkyl_name counts only carbons
            # and would DROP the amino N (structure-wrong). When hetero_aware is
            # set and the fragment carries a non-carbon heavy atom, name it with
            # the hetero-aware recursive namer instead. Fail-closed on decline.
            _hetero_named = False
            if (hetero_aware and not has_phenyl and not has_ring_sub
                    and not has_fused_het_sub and cc > 0
                    and any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)
                            for i in frag)):
                sub_name = name_substituent_fragment(
                    mol, frag, frag[0], list(chain_set) + [n_idx]
                )
                if sub_name:
                    n_subs.append(sub_name)
                    _hetero_named = True
                else:
                    # a hetero branch we cannot name -> fail closed (drop the
                    # whole molecule rather than emit a name missing the N).
                    n_subs.append(None)
                    _hetero_named = True

            if (not _hetero_named and not has_phenyl and not has_ring_sub
                    and not has_fused_het_sub and cc > 0):
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
    return n_subs


def _assemble_imine_name(features: Any, style: str) -> Optional[str]:
    """Wave2 T2a (P-62.3.1.1): N-substituted acyclic imine R-CH=N-R'.

    BB VERBATIM: 'N-methylethanimine (PIN) [not N-ethylidenemethanamine; nor
    N-ethylidene(methyl)amine]'. The imine nitrogen carries exactly one
    substituent (its other neighbour is the double-bonded parent carbon),
    cited as an italic N- prefix on the parent imine name — the single-N
    shape of _assemble_amine_name with the imine suffix.

    FAIL-CLOSED: returns None (generic path unchanged) for anything but a
    single acyclic imine whose lone N-substituent names cleanly — bare
    imines, di-imines, ring imines and unnameable substituents keep today's
    behavior.
    """
    mol = features.mol
    pg_atoms = features.principal_group_atoms
    if not pg_atoms or len(pg_atoms) != 1:
        return None
    match = pg_atoms[0]
    n_idx = None
    c_idx = None
    for idx in match:
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if sym == 'N':
            n_idx = idx
        elif sym == 'C':
            c_idx = idx
    if n_idx is None or c_idx is None:
        return None
    nitrogen = mol.GetAtomWithIdx(n_idx)
    if nitrogen.IsInRing():
        return None
    n_sub_roots = [nb.GetIdx() for nb in nitrogen.GetNeighbors()
                   if nb.GetIdx() != c_idx]
    if len(n_sub_roots) != 1:
        return None  # bare =NH imine -> generic path unchanged
    chain_set = set(features.principal_chain) if features.principal_chain else set()
    if not chain_set or c_idx not in chain_set or n_idx in chain_set:
        return None
    n_subs = _walk_amine_n_substituents(mol, n_idx, chain_set)
    if len(n_subs) != 1:
        return None  # unnameable N-substituent -> fail closed downstream

    n_prefix = f"N-{_wrap_n_substituent(n_subs[0])}"

    # Base = parent + imine suffix (+ stereo); C-substituent prefixes render
    # separately and merge with the N-block (mirrors _assemble_amine_name).
    fragments = []
    if not features.principal_chain:
        return None
    parent = _generate_chain_parent(features)
    if not parent:
        return None
    fragments.append(parent)
    suffix = _generate_suffix(features)
    if not suffix:
        return None
    fragments.append(suffix)
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)
    base_name = _assemble_fragments(fragments, style)
    if not base_name:
        return None

    other_prefixes = _generate_prefixes(features)
    c_prefix_parts = []
    for p in sorted(other_prefixes, key=lambda x: alpha_sort_key(x.text)):
        if p.locants and not str(p.text)[:1].isdigit():
            _loc = ",".join(str(_l) for _l in p.locants)
            c_prefix_parts.append(f"{_loc}-{p.text}")
        else:
            c_prefix_parts.append(p.text)
    if c_prefix_parts:
        return f"{'-'.join(c_prefix_parts)}-{n_prefix}{base_name}"
    return f"{n_prefix}{base_name}"


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

    # WS-A task 9 (P-66.6.1): a RING nitrogen is a skeletal heteroatom of a
    # ring parent hydride, never an amine. Walking its "substituents" from
    # here CUT the ring open ('N,N-dibutyl...' from morpholine). The amine
    # SMARTS now excludes ring N (perception keystone); this emitter-level
    # gate is defense-in-depth for any legacy caller.
    if nitrogen.IsInRing():
        return None

    chain_set = set(features.principal_chain) if features.principal_chain else set()
    # WS-A task 9: when a RING is the parent, its atoms must not be
    # re-enumerated as N-substituents (the parent cyclohexane was emitted
    # AGAIN as 'N-cyclohexyl').
    if getattr(features, 'principal_ring', None):
        chain_set |= set(features.principal_ring)

    # D2 (P-62.2.2 / P-16.3.3): collect ALL principal amine nitrogens, not just
    # pg_atoms[0]. When >=2 non-ring principal N are present, this is an acyclic
    # di/poly-amine; a single-N walk would drop every substituent except the
    # first nitrogen's (a WRONG structure SELF-01 then suppresses to 'unknown').
    # Dispatch to the multi-N handler, which cites N-substituents with PRIMED
    # italic-N locants (N, N', N''). One-nitrogen amines fall through unchanged.
    principal_n = []
    for m in pg_atoms:
        for idx in m:
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() == 'N' and not a.IsInRing():
                if idx not in principal_n:
                    principal_n.append(idx)
                break
    if len(principal_n) >= 2:
        return _assemble_polyamine_name(features, style, principal_n, chain_set)

    # Find N-substituents off this single nitrogen (byte-identical to the
    # original single-N walk; the multi-N branch below reuses the same helper).
    n_subs = _walk_amine_n_substituents(mol, n_idx, chain_set)
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
    # Build name using standard fragments. Phase 184 WS-E.1: the parent + suffix
    # (+ stereo) form the BASE; the C-substituent (hydroxy/chloro/...) prefixes are
    # rendered SEPARATELY and merged with the N-substituent prefix block below so
    # the detachable prefixes alphabetize together (P-14.5.2). The previous
    # `f"{n_prefix}{base_name}"` blindly PREPENDED the N-prefix to a base_name that
    # already carried the C-prefixes, producing malformed concatenations like
    # `N,N-dimethyl2-chloroethan-1-amine` (no hyphen, wrong order). Mirror the amide
    # handler (composer.py:3700-3726), which alphabetizes the C-prefixes and joins
    # them before the N-locant block with a hyphen separator.
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

    # Generate stereodescriptors (part of the BASE, not a detachable prefix)
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)

    base_name = _assemble_fragments(fragments, style)
    if not base_name:
        return None

    # Generate non-N (C-substituent) prefixes (halogens, hydroxy, etc.) as their
    # own alphabetized block, mirroring the amide handler's prefix rendering.
    other_prefixes = _generate_prefixes(features)
    c_prefix_parts = []
    for p in sorted(other_prefixes, key=lambda x: alpha_sort_key(x.text)):
        # P-14.3.4 (Phase 171 BBR-ASM, DEF-4): _generate_prefixes is inconsistent —
        # alkyl prefixes embed the locant in .text ('3-methyl') while FG prefixes
        # keep it separate ('hydroxy', locants=(2,)). Prepend .locants only when the
        # text lacks a leading digit (same shape as amide handler composer.py:3713).
        if p.locants and not str(p.text)[:1].isdigit():
            _loc = ",".join(str(_l) for _l in p.locants)
            c_prefix_parts.append(f"{_loc}-{p.text}")
        else:
            c_prefix_parts.append(p.text)

    # Merge: C-substituent prefixes (alphabetized) + N-locant prefix block + base.
    # Insert a hyphen before the N-locant block (it starts with 'N'/'N,N-').
    if c_prefix_parts:
        c_prefix_str = "-".join(c_prefix_parts)
        return f"{c_prefix_str}-{n_prefix}{base_name}"
    return f"{n_prefix}{base_name}"


def _assemble_polyamine_name(
    features: Any, style: str, principal_n: list, chain_set: set
) -> Optional[str]:
    """Assemble an acyclic di/poly-amine (>=2 principal amine N) with PRIMED
    italic-N locant prefixes (N, N', N''), the acyclic analog of
    _name_substituted_benzenediamine (benzene.py) (D2, P-62.2.2 / P-16.3.3).

    Each principal nitrogen is walked for its N-substituents (reusing
    _walk_amine_n_substituents), excluding the principal chain, principal ring,
    AND every OTHER principal amine nitrogen and their chain carbons, so a chain
    carbon between two amines is never named as an N-substituent and one N's
    branch is never re-attributed to another nitrogen (spec risk 3).

    Numbering (P-31.1.4): nitrogens are ordered by their amine-carbon chain
    locant ascending; the lowest-locant SUBSTITUTED nitrogen is cited with a
    bare 'N', each subsequent substituted nitrogen with 'N'', 'N''', ...
    Identical substituents across nitrogens are grouped into one multiplied
    prefix with the combined locant set (e.g. N,N,N',N'-tetramethyl). N- and
    C-substituent prefixes alphabetize together (P-14.5.2).
    """
    mol = features.mol
    atom_to_locant = getattr(features, 'atom_to_locant', {}) or {}

    def amine_carbon_locant(n_idx: int) -> Any:
        """Lowest chain-carbon locant among this N's chain neighbours."""
        locs = []
        for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            li = atom_to_locant.get(nbr.GetIdx())
            if li is not None:
                locs.append(li)
        return min(locs) if locs else float('inf')

    principal_n_set = set(principal_n)
    # Order the nitrogens by their amine-carbon locant (P-31.1.4). Tie-break on
    # the atom index for determinism.
    ordered_n = sorted(principal_n, key=lambda ni: (amine_carbon_locant(ni), ni))

    # P-62.2.4.1.3 (BB 26375, plan P1AM Task 9): in a COMPLEX polyamine the
    # parent is the senior DIAMINE on the principal chain (P-44.3 longest
    # chain / P-45.2.1 most substituted); principal nitrogens with NO chain-
    # carbon neighbour are NOT suffix nitrogens — they belong inside an
    # N-substituent branch of a chain-attached nitrogen (aminomethyl,
    # 2-aminoethyl). Demote them: they leave principal_n so the branch walk
    # names them as part of the branch (hetero_aware).
    chain_n = [ni for ni in ordered_n
               if amine_carbon_locant(ni) != float('inf')]
    demoted_n = [ni for ni in ordered_n if ni not in chain_n]
    if demoted_n:
        if len(chain_n) < 2:
            return None  # not a diamine parent — fail closed
        ordered_n = chain_n
        principal_n_set = set(chain_n)

    # Walk each nitrogen's substituents, excluding the parent chain/ring AND all
    # OTHER principal nitrogens (and their chain carbons) so branches are never
    # cross-attributed (spec risk 3). In the complex-polyamine case the demoted
    # nitrogens are NO LONGER in principal_n_set, so the walk follows the branch
    # THROUGH them (with hetero_aware naming -> aminomethyl / 2-aminoethyl).
    from collections import Counter
    per_n_subs = []  # list of (n_idx, Counter(sub_name -> count))
    for n_idx in ordered_n:
        # chain_set gates the entry neighbour (parent chain/ring + inter-amine
        # chain carbons are all in the principal chain -> never an N-substituent).
        # stop_set hard-blocks traversal at every OTHER principal nitrogen so a
        # branch is never re-attributed across nitrogens (spec risk 3).
        stop_set = principal_n_set - {n_idx}
        subs = _walk_amine_n_substituents(
            mol, n_idx, chain_set, stop_set, hetero_aware=bool(demoted_n))
        if any(s is None for s in subs):
            return None  # un-nameable hetero branch -> fail closed
        per_n_subs.append((n_idx, Counter(subs)))

    # P-31.1.4.3.4 / P-45.2.1 (plan P1AM Task 9): choose the chain numbering
    # that gives the LOWEST locants to the set of cited N-substituents. The
    # symmetric ethane-1,2-diamine parent admits two carbon-locant assignments
    # (a swap of {1,2}); pick the direction whose sorted (locant, alpha-name)
    # substituent set wins first-point-of-difference. Scoped to the complex-
    # polyamine (demoted) case; the primed 2-N path keeps atom_to_locant as-is.
    carbon_locant_of = amine_carbon_locant
    if demoted_n and len(chain_n) == 2:
        _base_locs = sorted(amine_carbon_locant(ni) for ni in chain_n)
        _n_names = {ni: sorted(cnt.elements())
                    for ni, cnt in per_n_subs}

        def _locset_for(assign):
            # assign: dict n_idx -> chain locant; produce the sorted list of
            # (locant, alpha_name) pairs over every cited N-substituent.
            pairs = []
            for ni in chain_n:
                for nm in _n_names.get(ni, []):
                    pairs.append((assign[ni], alpha_sort_key(nm)))
            return sorted(pairs)

        _asc = {chain_n[i]: _base_locs[i] for i in range(2)}
        _desc = {chain_n[i]: _base_locs[1 - i] for i in range(2)}
        _chosen = _asc if _locset_for(_asc) <= _locset_for(_desc) else _desc
        carbon_locant_of = lambda ni, _c=_chosen, _f=amine_carbon_locant: (
            _c.get(ni, _f(ni)))

    # Assign primed italic-N tags: only SUBSTITUTED nitrogens receive a tag, in
    # amine-carbon-locant order (lowest -> bare 'N', then N', N'', ...).
    # Re-order the substituted nitrogens by the CHOSEN carbon locant so the
    # lowest-locant substituted N is cited first.
    if demoted_n:
        per_n_subs = sorted(per_n_subs, key=lambda t: (carbon_locant_of(t[0]), t[0]))
    n_tag_by_idx = {}
    tag_count = 0
    for n_idx, counter in per_n_subs:
        if not counter:
            continue
        if demoted_n:
            # P-62.2.4.1.3 examples: numeric superscript tags (flattened),
            # e.g. N1-(aminomethyl)..., N1,N2,N2-trimethyl... — load-bearing
            # for OPSIN disambiguation when a third amine lives in a branch.
            n_tag_by_idx[n_idx] = f"N{carbon_locant_of(n_idx)}"
        else:
            n_tag_by_idx[n_idx] = "N" + ("'" * tag_count)
        tag_count += 1

    if not n_tag_by_idx:
        return None  # no N-substituents on any nitrogen -> primary-diamine path

    # Group identical substituents across nitrogens into one multiplied prefix
    # with the combined locant set (P-16.3.3: N,N,N',N'-tetramethyl). Preserve
    # the tag order (N < N' < N'') within each locant string.
    sub_tags = {}  # sub_name -> ordered list of italic-N tags (one per occurrence)
    for n_idx, counter in per_n_subs:
        tag = n_tag_by_idx.get(n_idx)
        if tag is None:
            continue
        for name in sorted(counter.keys()):
            sub_tags.setdefault(name, []).extend([tag] * counter[name])

    def tag_sort_key(t: str):
        return (t.count("'"), int(t[1:]) if t[1:].isdigit() else 0, t)

    n_prefix_entries = []  # (alpha_key, rendered)
    for name, tags in sub_tags.items():
        tags_sorted = sorted(tags, key=tag_sort_key)
        count = len(tags_sorted)
        loc_str = ",".join(tags_sorted)
        # P-16.3.3 (plan P1AM Task 9): a compound N-substituent that carries
        # its own locant/hyphen ('2-aminoethyl') or the simple 'aminomethyl'
        # is enclosed in parentheses so it alphabetises on and cites as its
        # complete name. Simple alkyls ('methyl') stay bare.
        _wrapped = _wrap_n_substituent(name)
        if demoted_n and _wrapped == name and (
                any(ch.isdigit() for ch in name) or 'amino' in name):
            _wrapped = f"({name})"
        if count == 1:
            rendered = f"{loc_str}-{_wrapped}"
        else:
            mult = get_multiplier_prefix(count, name)
            rendered = f"{loc_str}-{mult}{_wrapped}"
        n_prefix_entries.append((alpha_sort_key(name), rendered))

    # Base name: parent + '-diamine' suffix (+ stereo). Built from the SUFFIX
    # nitrogens by _generate_suffix/_generate_chain_parent. In the complex-
    # polyamine case the demoted nitrogens must NOT inflate the multiplier
    # (P-62.2.4.1.3: the parent is a DIAMINE), so filter principal_group_atoms
    # to the chain nitrogens for the duration of suffix generation (a scoped
    # override, mirroring _assemble_amide_name's _acyl_locants pattern — no
    # persistent mutation of shared state).
    _pga_backup = features.principal_group_atoms
    if demoted_n:
        _chain_n_set = set(chain_n)
        features.principal_group_atoms = [
            _m for _m in (_pga_backup or [])
            if any(a in _chain_n_set for a in _m)]
    try:
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

        if features.stereocenters or getattr(features, 'double_bond_stereo', None):
            stereo = _generate_stereodescriptors(features)
            if stereo:
                fragments.append(stereo)

        base_name = _assemble_fragments(fragments, style)
    finally:
        features.principal_group_atoms = _pga_backup
    if not base_name:
        return None

    # C-substituent prefixes (halogens, hydroxy, chain-alkyls, ...) as their own
    # entries, mirroring the single-N path; alphabetized WITH the N-prefixes.
    # Wave-2 completion C2: perceived chain-substituent branches that CONTAIN a
    # principal amine N (the '(methylamino)' record for a suffix N's branch)
    # are already fully expressed by the -diamine suffix + the N-prefixes above
    # — filter them out or the branch is double-counted (the SELF-01-suppressed
    # 'N-methyl-3-(methylamino)propane-1,3-diamine' defect).
    _subs_backup = features.substituents
    if _subs_backup:
        _filtered = {}
        for _loc, _branches in _subs_backup.items():
            _kept = [b for b in _branches
                     if not (set(b) & principal_n_set)]
            if _kept:
                _filtered[_loc] = _kept
        features.substituents = _filtered
    try:
        _gen_prefixes = _generate_prefixes(features)
    finally:
        features.substituents = _subs_backup

    c_prefix_entries = []  # (alpha_key, rendered)
    for p in _gen_prefixes:
        if p.locants and not str(p.text)[:1].isdigit():
            _loc = ",".join(str(_l) for _l in p.locants)
            rendered = f"{_loc}-{p.text}"
        else:
            rendered = p.text
        c_prefix_entries.append((alpha_sort_key(p.text), rendered))

    all_entries = n_prefix_entries + c_prefix_entries
    all_entries.sort(key=lambda e: e[0])
    prefix_part = "-".join(rendered for _k, rendered in all_entries)
    return f"{prefix_part}{base_name}"


def _assemble_aromatic_benzonitrile(mol, principal_ring, nitrile_atoms):
    """v28 Cluster D (P-66.5.4.2): name an aromatic benzene ring bearing a nitrile
    as a substituted benzonitrile ('benzonitrile', '4-methylbenzonitrile',
    '4-(methoxycarbonyl)benzonitrile'), delegating to the benzene assembler.

    Returns the name, or None (fail closed) if the nitrile ring-carbon cannot be
    located or ANY ring substituent is outside the supported table — the caller
    then falls back to the saturated cyclo path (which itself fails closed).
    """
    from ..rules.benzene import _name_substituted_benzonitrile
    from ..rules.ring_substituents import _ring_atom_simple_substituents
    from .naming_utils import apply_enclosing_marks

    ring_set = set(principal_ring)
    nitrile_set = set(nitrile_atoms)
    # The nitrile carbon (the C of C#N) is the group atom bonded to a ring atom;
    # the ring atom it attaches to is position 1 of the benzonitrile.
    nitrile_rc = None
    for na in nitrile_atoms:
        for nb in mol.GetAtomWithIdx(na).GetNeighbors():
            if nb.GetIdx() in ring_set:
                nitrile_rc = nb.GetIdx()
                break
        if nitrile_rc is not None:
            break
    if nitrile_rc is None:
        return None

    oriented_ring = list(principal_ring)
    atom_to_locant = {atom: i + 1 for i, atom in enumerate(oriented_ring)}

    def _is_compound_prefix(nm: str) -> bool:
        # A compound substituent prefix (formed by substitution, e.g. the
        # alkoxycarbonyl family) is enclosed even as a bare word (P-16.3.3);
        # simple table prefixes (methyl/chloro/methoxy/nitro/...) are not.
        # The hyphen in the character class made this a copy of the compound
        # predicate; the P-16.3.4 carve-out is the shared primitive.
        from .naming_utils import italicized_prefix_is_bare
        if italicized_prefix_is_bare(nm):
            return False
        return nm.endswith('oxycarbonyl') or any(c in nm for c in '-()[]0123456789')

    substituent_groups: Dict[str, List[int]] = defaultdict(list)
    for ra in principal_ring:
        if ra == nitrile_rc:
            continue  # the nitrile itself is the parent group, not a prefix
        found = _ring_atom_simple_substituents(mol, ra, ring_set, nitrile_set)
        if found is None:
            return None  # unsupported exocyclic group -> fail closed
        prefixes, _covered = found
        for p in prefixes:
            rendered = apply_enclosing_marks(p, -1) if _is_compound_prefix(p) else p
            substituent_groups[rendered].append(atom_to_locant[ra])

    if not substituent_groups:
        return "benzonitrile"
    return _name_substituted_benzonitrile(
        dict(substituent_groups), atom_to_locant[nitrile_rc],
        atom_to_locant, oriented_ring,
    )


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

    # Get nitrile atoms to exclude from substituent discovery
    nitrile_atoms = None
    exclude_atoms = set()
    if features.principal_group_atoms:
        nitrile_atoms = features.principal_group_atoms[0]
        if nitrile_atoms:
            exclude_atoms = set(nitrile_atoms)

    # v28 Cluster D (P-66.5.4.2): an AROMATIC benzene principal ring bearing a
    # (forced) nitrile is a BENZONITRILE, not a saturated cyclohexanecarbonitrile.
    # This path is reached via the polyfunctional forced-nitrile route (e.g. the
    # nitrile_oxide handler re-entry on '4-(methoxycarbonyl)benzonitrile oxide',
    # where a co-present ester makes the molecule polyfunctional and bypasses the
    # benzene handler). Delegate to the benzene substituted-nitrile assembler so
    # the parent, the nitrile=1 renumbering, and any compound prefix
    # (methoxycarbonyl, enclosed per P-16.3.3) are correct. Fail closed to the
    # saturated cyclo path when any ring substituent is not fully nameable.
    if (ring_size == 6 and nitrile_atoms
            and all(features.mol.GetAtomWithIdx(a).GetIsAromatic()
                    and features.mol.GetAtomWithIdx(a).GetAtomicNum() == 6
                    for a in principal_ring)):
        _benzo = _assemble_aromatic_benzonitrile(
            features.mol, principal_ring, nitrile_atoms
        )
        if _benzo is not None:
            return _benzo

    try:
        stem = get_chain_prefix(ring_size)
    except (ValueError, KeyError):
        return "carbonitrile"  # Fallback for invalid ring size

    if not stem:
        return "carbonitrile"  # Guard against empty stem producing 'cycloane'

    parent_name = f"cyclo{stem}ane"

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
        return _join_prefix_to_name(prefix_str, base_name)  # L2 (P-16.3.3)

    return base_name


# Phase 160.2 Plan-02-01: _generate_chain_parent + _generate_ring_parent lifted to
# handlers/_handler_shared.py per CONTEXT D-02 + D-03. Module-top shim re-imports
# (see top of file) preserve all in-file call sites byte-identical per CONTEXT
# D-13 forbidden-boundary preservation.


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


# Phase 160.2 Plan-02-01: _generate_suffix lifted to handlers/_handler_shared.py per CONTEXT D-02 + D-03.


# Phase 160.2 Plan-02-01: _generate_prefixes lifted to handlers/_handler_shared.py per CONTEXT D-02 + D-03.


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


def _ring_sub_group_stereo_veto(features: Any, frag_atom_sets: List[frozenset]) -> bool:
    """W8-P6 Cluster E (P-93.6 Ex6) fail-closed veto for a multiplicative
    ring-substituent collapse (bis/tris/di/tri).

    ``_generate_ring_substituent_prefixes`` groups ring substituents by their
    GENERATED NAME STRING and merges same-name groups into one multiplicative
    prefix. That merge silently asserts the merged occurrences are the
    IDENTICAL substituent -- but per-substituent ring stereo is dropped when
    the name string is generated (``get_ring_substituent_name`` /
    ``_build_substituted_ring_name`` do not encode CIP descriptors into the
    returned text), so two ring copies that are genuinely
    stereo-DIFFERENT collapse to the same string and merge wrongly.

    P-93.6 Ex6 (Blue Book) explicitly FORBIDS the multiplicative form when
    the ligands differ by a stereodescriptor -- the correct PIN then needs a
    per-substituent bracket-internal stereo block
    (``(1r,4S)``/``(1s,4S)``-style) Orthonym does not yet construct. Rather
    than emit a wrong (falsely-symmetric) stereoisomer name, decline the
    collapse so the caller fails closed.

    This is deliberately NARROW (Wave-4 W4-S2 "P-93.6 flagship" finding, not
    a general per-substituent CIP engine): it fires ONLY when
      (a) at least one of the merged ring fragments carries an explicit
          tetrahedral chiral tag (the substituent itself is stereogenic --
          an achiral ring like plain cyclohexyl/phenyl never trips this), AND
      (b) somewhere OUTSIDE all the merged fragments' own atoms, some atom
          had an explicit chiral tag (``@``/``@@``) IN THE ORIGINAL INPUT
          SMILES that RDKit's default sanitize (``Chem.MolFromSmiles``,
          which runs ``AssignStereochemistry(cleanIt=True)``) subsequently
          ERASED -- the "CIP ceiling" signature. This is exactly the
          flagship's signature: the central propan-2-ol carbon is written
          ``[C@@H]`` in the input SMILES (BB wants it ``(2R)``), yet RDKit's
          perception decides its two (ring-bearing) branches are
          constitutionally-and-configurationally equivalent and strips the
          tag entirely -- a known CIP-perception limit (~62% ceiling; W4-S2:
          "RDKit gives no _CIPCode on the central propan-2-ol carbon...
          DISAGREEING with the BB PIN"), not a fixable local bug. Detecting
          "present pre-sanitize, absent post-sanitize" is a cheap, reliable,
          LOCAL proxy for that global CIP-ceiling condition -- it needs no
          recursive CIP computation, only a second parse of the same input
          string with the stereo-cleanup step skipped.

    Genuinely achiral / verifiably-identical multiplicative names (no chiral
    tag anywhere in the merged fragments, e.g. 1,3-dicyclohexylpropan-2-ol,
    1,3-dimethylbenzene) never enter condition (a) and are unaffected. A
    chiral merged substituent whose surrounding molecule has no erased
    stereo tag (condition (b) false) also proceeds unchanged.
    """
    if not frag_atom_sets or len(frag_atom_sets) < 2:
        return False

    from rdkit import Chem as _Chem

    mol = features.mol

    any_chiral_fragment = any(
        any(
            mol.GetAtomWithIdx(a).GetChiralTag() != _Chem.ChiralType.CHI_UNSPECIFIED
            for a in atoms
        )
        for atoms in frag_atom_sets
    )
    if not any_chiral_fragment:
        return False

    excluded_atoms: set = set()
    for atoms in frag_atom_sets:
        excluded_atoms |= set(atoms)

    # Re-parse the ORIGINAL input string with sanitize=False (preserves the
    # literal @/@@ parity from the SMILES) then a generic SanitizeMol (which,
    # unlike Chem.MolFromSmiles's default pipeline, does NOT run the
    # stereo-perception cleanIt step that erases non-canonically-stereogenic
    # tags). features.mol was built via ONE Chem.MolFromSmiles(features.smiles)
    # call (namer.py:_perceive), so atom indices line up exactly -- both mols
    # are parsed from the identical string, and SMILES atom numbering is
    # string-order-determined independent of the sanitize flag.
    try:
        raw_mol = _Chem.MolFromSmiles(features.smiles, sanitize=False)
        if raw_mol is None or raw_mol.GetNumAtoms() != mol.GetNumAtoms():
            return False
        _Chem.SanitizeMol(raw_mol)
    except Exception:
        return False

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in excluded_atoms:
            continue
        if atom.GetChiralTag() != _Chem.ChiralType.CHI_UNSPECIFIED:
            continue  # already resolved/preserved post-sanitize -- not erased
        raw_atom = raw_mol.GetAtomWithIdx(idx)
        if raw_atom.GetChiralTag() != _Chem.ChiralType.CHI_UNSPECIFIED:
            return True  # explicit in the input, erased by perception

    return False


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
    # W8-P6 Cluster E (P-93.6 Ex6 fail-closed guard): parallel tracking of the
    # ATOMS behind each occurrence pushed into ring_sub_groups[name], so a
    # count>1 (multiplicative bis/tris) collapse can be vetoed when the
    # collapse would silently discard a genuine stereo difference between
    # ring copies. Populated only by the single_ring_groups loop below (the
    # multi-ring-fragment / fused-het insertion points leave this empty for
    # their keys, so the guard is a no-op there — unchanged, narrowly scoped
    # behaviour per the Wave-4 W4-S2 "P-93.6 flagship" CIP-ceiling finding:
    # `C[C@H]1CC[C@@H](C[C@@H](O)C[C@@H]2CC[C@H](C)CC2)CC1` is CIP-engine-
    # BLOCKED (RDKit assigns no _CIPCode to the central propan-2-ol carbon,
    # yet both rings show the SAME 's' label — the multiplicative collapse
    # asserts the two 4-methylcyclohexyl rings are IDENTICAL when the BB PIN
    # (P-93.6 Ex1) requires DIFFERENT bracket-internal descriptors
    # ((1r,4S) vs (1s,4S)) — a wrong-stereoisomer name Orthonym cannot yet
    # construct correctly. See _ring_sub_group_stereo_veto below.
    ring_sub_group_atoms: Dict[str, List[frozenset]] = defaultdict(list)

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

        # BP-3 cluster R integration (P-29 / P-31.1.4.3.4 / P-59.2.3): a
        # MONOCYCLIC ring substituent carrying its OWN decorations is named as
        # ONE bracketed ring-yl prefix via the substituent chokepoint
        # (name_ring_system_substituent, which numbers the free valence and
        # places decorations), NOT as a bare ring name — the bare path has no
        # home for a monocyclic ring's decorations (they are neither ring nor
        # chain atoms, so a decorated heteroaryl substituent, e.g. a
        # dimethylpyrazolyl on a diaryl methanone, could not be fully named) and
        # additionally misidentifies pyrazole as imidazole. Fires only for a
        # decorated monocycle where the chokepoint returns a complete numbered
        # name; bare rings (no decorations) and chokepoint declines fall through
        # to the existing paths unchanged -> additive, fail-closed.
        if _ring_attach_atom is not None and len(ring_atom_set) == len(ring_atoms):
            _rc_frag = set(ring_atoms)
            _rc_stack = list(ring_atoms)
            while _rc_stack:
                _rc_cur = _rc_stack.pop()
                for _rc_nb in features.mol.GetAtomWithIdx(_rc_cur).GetNeighbors():
                    _rc_ni = _rc_nb.GetIdx()
                    if (_rc_ni in _rc_frag or _rc_ni in chain_set
                            or _rc_nb.GetAtomicNum() <= 1):
                        continue
                    _rc_frag.add(_rc_ni)
                    _rc_stack.append(_rc_ni)
            if _rc_frag != set(ring_atoms):  # the ring carries decorations
                from ..rules.ring_substituents import (
                    name_ring_system_substituent as _rc_nrss,
                )
                _rc_name = _rc_nrss(
                    features.mol, sorted(_rc_frag), _ring_attach_atom
                )
                if (_rc_name and ' ' not in _rc_name
                        and any(_ch.isdigit() for _ch in _rc_name)):
                    # P-16.3.3 nested enclosure: a decorated ring name that
                    # itself contains parentheses (e.g. '2-(methoxycarbonyl)-
                    # cyclohexyl') must be enclosed in the NEXT bracket level
                    # ([...]), not another pair of parens. apply_enclosing_marks
                    # (-1) auto-detects the depth from the name's own brackets,
                    # so a plain decorated name ('2-methylcyclohexyl') still
                    # gets (...) — byte-identical for the undecorated-paren case.
                    from ..assembly.naming_utils import apply_enclosing_marks as _aem
                    ring_sub_groups[_aem(_rc_name, -1)].append(locant)
                    continue

        # Get base substituent name (phenyl, cyclohexyl, etc.)
        # For multi-ring systems, try the full fused system first for retained name
        # lookup (naphthalene, anthracene), but fall back to the SSSR ring if no
        # retained name is found (to avoid renaming a 5-membered pyrrole ring in
        # a porphyrin as "cyclononacosyl").
        if len(ring_atom_set) > len(ring_atoms):
            _full_ring = tuple(sorted(ring_atom_set))
            _full_name = get_ring_substituent_name(features.mol, _full_ring, _ring_attach_atom)
            # Accept the full-system name only if it produced a retained / routed
            # name (not a generic cyclo-name). Phase 4 SUBST-01: get_ring_substituent_name
            # may now return None (fail-closed for an unnameable polycyclic) — fall
            # back to the SSSR ring in that case too.
            if _full_name and (not _full_name.startswith('cyclo') or _full_name in ('cyclohexyl', 'cyclopentyl', 'cyclopropyl', 'cyclobutyl', 'cycloheptyl', 'cyclooctyl')):
                base_name = _full_name
            else:
                # Full system produced generic cyclo-name / None; use SSSR ring
                base_name = get_ring_substituent_name(features.mol, ring_atoms, _ring_attach_atom)
        else:
            base_name = get_ring_substituent_name(features.mol, ring_atoms, _ring_attach_atom)
        # Phase 4 SUBST-01: a None base_name (unnameable ring) must not reach the
        # string assembly below — emit the honest fallback marker so the molecule
        # surfaces as unknown rather than crashing / dropping the ring.
        if base_name is None:
            # v25 P0 Task 0.1: an unnameable ring BRANCH is the root cause of
            # the eventual abstention (the E2 recursive-namer census bucket).
            from ..metrics.abstention import AbstentionCode, record_abstention
            record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                              detail='ring_substituent_unnameable')
            base_name = 'unknown'

        # Detect substituents on the ring itself
        sub_name = _build_substituted_ring_name(
            features.mol, ring_atoms, chain_set, base_name
        )
        # Wave2 T3a conservation: None = the ring carries branches this
        # machinery cannot express — emit the honest fallback marker so the
        # molecule surfaces as unknown rather than dropping the branch atoms
        # (same convention as the unnameable-ring marker above).
        if sub_name is None:
            # v25 P0 Task 0.1: decorated-ring branch the machinery cannot
            # express — same census bucket as the unnameable-ring marker.
            from ..metrics.abstention import AbstentionCode, record_abstention
            record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                              detail='ring_substituent_branches_unnameable')
            sub_name = 'unknown'

        ring_sub_groups[sub_name].append(locant)
        ring_sub_group_atoms[sub_name].append(frozenset(ring_atom_set))

    # Build prefix fragments
    # Wave2 T3a (P-14.3.4 Rule 1): a mononuclear (1-atom) chain parent has a
    # single trivially-1 position, so ring-substituent locants are never cited
    # (phenylmethanol / diphenylmethanone, NOT 1-phenylmethanol). Route the
    # decision through the shared chokepoint rather than an inline test.
    _omit_ring_sub_locants = should_omit_locant_one(
        context="prefix",
        chain_length=len(features.principal_chain or ()),
    )
    for name, locants in ring_sub_groups.items():
        count = len(locants)
        sorted_locants = sorted(locants)

        # W8-P6 Cluster E (P-93.6 Ex6): about to fold count>=2 occurrences of
        # this ring-substituent name into ONE multiplicative (bis/tris/di/tri)
        # prefix — i.e., assert they are the SAME substituent. Veto that
        # assertion when it cannot be verified (CIP-ceiling case) rather than
        # risk emitting a wrong (more-symmetric) stereoisomer.
        if count > 1 and _ring_sub_group_stereo_veto(
            features, ring_sub_group_atoms.get(name, [])
        ):
            from ..errors import unsupported_ring_system
            raise unsupported_ring_system()

        if _omit_ring_sub_locants:
            formatted = format_substituent_prefix(name, [], count)
        else:
            formatted = format_substituent_prefix(name, sorted_locants, count)

        prefixes.append(NameFragment(
            text=formatted,
            locants=() if _omit_ring_sub_locants else tuple(sorted_locants),
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

    Only handles common cases on benzene-like rings, plus compound branches
    via the recursive fragment namer (hydroxymethyl). Wave2 T3a
    constitution-conservation guard: when the ring DOES carry exocyclic
    branches but this function cannot express them (wrong ring class,
    numbering failure, unnameable branch), it returns ``None`` — the caller
    must fail closed. Returning the bare ``base_name`` in that situation
    silently dropped the branch atoms (a different molecule; only the
    OPSIN-dependent SELF-01 oracle caught it). ``base_name`` is still
    returned unchanged when the ring genuinely has no branches.
    """
    from rdkit import Chem
    from ..perception.rings import get_containing_ring_system

    # Use the complete ring system as BFS boundary (IUPAC P-25.3)
    ring_atom_set = set(get_containing_ring_system(mol, ring_atoms))

    # Wave2 T3a: does the ring system carry ANY exocyclic non-chain branch?
    # Decides whether an early bare-base_name return is honest (no branches)
    # or a constitution drop (branches present -> None, fail closed).
    _has_exo_branches = any(
        nbr.GetIdx() not in ring_atom_set and nbr.GetIdx() not in chain_set
        for ra in ring_atom_set
        for nbr in mol.GetAtomWithIdx(ra).GetNeighbors()
    )

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
        return None if _has_exo_branches else base_name

    # Carbocyclic monocyclic rings only. 6-membered rings keep their existing
    # behaviour (alkyl/halo/hydroxy already enumerated). Other sizes (3-8) are
    # admitted ONLY when the ring carries a previously-dropped FG prefix
    # (oxo/cyano) — so the widening cannot silently activate alkyl/halo
    # emission across the whole small-ring population, which would be an
    # unbudgeted canary change (V21 WS-D.1 skeptic guard).
    # Wave2 T3a: a branch-free ring needs no decoration anywhere below; a
    # branch-carrying ring that cannot reach (or survive) the decoration
    # machinery must fail closed rather than emit the bare base_name.
    if not _has_exo_branches:
        return base_name

    if not all(mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in ring_atoms):
        return None
    ring_size = len(ring_atoms)
    if ring_size != 6:
        if not (3 <= ring_size <= 8):
            return None
        # Attachment-relative single-ring numbering is only provably correct
        # when the numbered ring IS the complete ring system (a true
        # monocycle; D-09 missing > wrong).
        if set(ring_atoms) != ring_atom_set:
            return None

    # Try both numbering directions and pick lowest locant set
    numberings = _number_ring_from_attachment(mol, ring_atoms, attachment_idx)
    if numberings is None:
        return None

    # If single numbering returned (dict), wrap in list
    if isinstance(numberings, dict):
        numberings = [numberings]

    best_sub_groups = None
    best_locant_set = None

    for ring_order in numberings:
        if len(ring_order) != len(ring_atoms):
            continue
        sub_groups, _complete = _detect_ring_substituents(
            mol, ring_order, ring_atom_set, chain_set
        )
        if not _complete:
            # An exocyclic branch could not be named: emitting any name here
            # would drop its atoms. Fail closed (Wave2 T3a conservation).
            return None
        if not sub_groups:
            continue
        # Collect all locants for comparison
        all_locs = sorted(loc for locs in sub_groups.values() for loc in locs)
        if best_locant_set is None or all_locs < best_locant_set:
            best_locant_set = all_locs
            best_sub_groups = sub_groups

    if not best_sub_groups:
        # Branches exist (checked above) but no numbering produced a complete
        # substituent map -> fail closed rather than drop them.
        return None

    sub_groups = best_sub_groups

    # Build prefix string for ring substituents
    from ..assembly.naming_utils import (
        get_multiplier_prefix, is_complex_substituent, needs_brackets,
    )
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locs = sorted(sub_groups[name])
        count = len(locs)
        loc_str = ','.join(str(l) for l in locs)
        # Wave2 T3a: compound FG-on-alkyl prefixes (hydroxymethyl) take
        # enclosing marks with a simple multiplier — the Phase 11
        # '1,3,5-tri(hydroxymethyl)benzene' convention (needs_brackets;
        # is_complex_substituent / di-vs-bis deliberately unchanged, see the
        # needs-parens-consolidation tripwire).
        _enclose = is_complex_substituent(name) or needs_brackets(name)
        if count == 1:
            if _enclose and not name.startswith('('):
                prefix_parts.append(f'{loc_str}-({name})')
            else:
                prefix_parts.append(f'{loc_str}-{name}')
        else:
            mult = get_multiplier_prefix(count, name)
            if _enclose and not name.startswith('('):
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
    """Detect substituents on ring atoms.

    Returns ``(sub_groups, complete)``:
      * ``sub_groups``: ``{prefix_name: [locants]}`` or ``None`` when the ring
        carries no substituents.
      * ``complete``: False when ANY exocyclic branch could not be named —
        the caller must fail closed instead of emitting a name that silently
        drops atoms (Wave2 T3a constitution-conservation guard; the old
        contract skipped unrecognised branches, so a -CH2OH arm vanished from
        the name and only the OPSIN-dependent SELF-01 oracle caught it).
    """
    _HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    _ALKOXY = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy'}

    from ..rules.ring_substituents import ring_atom_fg_prefixes

    # FG prefixes (oxo/cyano) use the attachment-relative single-ring
    # numbering below, which is only provably correct when the numbered ring
    # IS the complete ring system (a true monocycle). For a ring inside a
    # fused system the emitted locant would be wrong (D-09: missing > wrong),
    # so FG emission is guarded off there and the legacy form is kept.
    _fg_ring_is_monocyclic = set(ring_order.values()) == set(ring_atom_set)

    complete = True
    sub_groups = defaultdict(list)
    for ring_pos, atom_idx in ring_order.items():
        # Characteristic groups carried by the ring atom itself (oxo, cyano)
        # via the shared primitive — applies even to the attachment atom
        # (e.g. 1-cyanocyclohexyl), unlike the neighbour-substituent scan
        # below which skips the chain attachment point.
        fg_claimed_atoms = set()
        if _fg_ring_is_monocyclic:
            _fgs, fg_claimed_atoms = ring_atom_fg_prefixes(
                mol, atom_idx, ring_atom_set, return_atoms=True
            )
            for fg in _fgs:
                sub_groups[fg].append(ring_pos)
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_atom_set or ni in chain_set:
                continue
            if ni in fg_claimed_atoms:
                continue  # already expressed by an atom-level FG prefix
            # Wave2 T3a: track whether this branch got named. The attachment
            # position (ring_pos 1) previously skipped the whole scan — its
            # extra substituents silently vanished; now they are named (with
            # locant 1) or flagged incomplete like every other position.
            _before = sum(len(v) for v in sub_groups.values())
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
                        # HYG-04 (Phase 167): shared aryl-count helper (benzyloxy / diphenylmethoxy).
                        _aryl_ether = _name_aryl_methyl_ether(mol, c_start, ni)
                        if _aryl_ether is not None:
                            sub_groups[_aryl_ether].append(ring_pos)
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
                    else:
                        # Wave2 T3a: hetero-containing branch (-CH2OH, -CF3,
                        # ...) — name it exactly via the recursive fragment
                        # namer over the FULL branch atom set. A branch that
                        # reconnects to the chain (bridge topology) or that
                        # the namer declines stays unnamed and flags
                        # incomplete below (fail closed; never drop atoms).
                        _branch, _touches_chain = _collect_branch_atoms(
                            mol, ni, ring_atom_set, chain_set
                        )
                        if _branch and not _touches_chain:
                            from .substituent_naming import name_substituent_fragment
                            _bname = name_substituent_fragment(
                                mol, _branch, ni, list(ring_atom_set)
                            )
                            if _bname:
                                sub_groups[_bname].append(ring_pos)

            # Wave2 T3a conservation check: this exocyclic branch produced no
            # prefix through any recognizer above -> the assembled ring name
            # would silently drop its atoms. Flag the detection incomplete so
            # the caller fails closed (P-29.4.2 safety; missing > wrong).
            if sum(len(v) for v in sub_groups.values()) == _before:
                complete = False

    return (dict(sub_groups) if sub_groups else None), complete


def _collect_branch_atoms(mol, start_idx, ring_atom_set, chain_set):
    """BFS a substituent branch from ``start_idx``, never entering the ring.

    Returns ``(atoms, touches_chain)`` — ``touches_chain`` is True when the
    branch reaches a principal-chain atom (bridge topology, not a simple
    substituent; Wave2 T3a conservation guard support).
    """
    visited = {start_idx}
    queue = deque([start_idx])
    atoms = []
    touches_chain = False
    while queue:
        idx = queue.popleft()
        if idx in chain_set:
            touches_chain = True
            continue
        atoms.append(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in visited and nidx not in ring_atom_set:
                visited.add(nidx)
                queue.append(nidx)
    return atoms, touches_chain


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
        # (amide/amine N-substituents handled by specialized assemblers).
        # Phase 163: chalcogen amides use single-permissive [NX3] SMARTS so the
        # N-substituent atoms are NOT in pg_atom_set directly. _assemble_amide_name
        # renders the N-substitution via name_amide()'s get_n_substituents() path;
        # this guard skips them here to prevent double-counting as chain prefixes.
        if features.principal_group in (
            'primary_amide', 'secondary_amide', 'tertiary_amide',
            'secondary_amine', 'tertiary_amine',
            'thioamide', 'selenoamide', 'telluroamide',
            # Wave2 T2a (P-62.3.1.1): the N-substituted imine's N-branch is
            # rendered as an italic-N prefix by _assemble_imine_name; without
            # this skip the walker re-names it as a phantom '(methylamino)'
            # co-substituent ('1-(methylamino)ethanimine' for CC=NC).
            'imine',
        ):
            # For chalcogen amides extend pg_atom_set with all atoms reachable from
            # the principal-group N (excluding the carbonyl C) so the N-bonded
            # carbons (methyl / ethyl / etc.) overlap correctly.
            effective_pg = pg_atom_set
            if features.principal_group in (
                'thioamide', 'selenoamide', 'telluroamide',
            ):
                effective_pg = set(pg_atom_set)
                for match in features.principal_group_atoms or ():
                    # Match layout: (C_carbonyl, =chalcogen, N) per Phase 163 SMARTS
                    if len(match) >= 3:
                        n_idx = match[2]
                        carbonyl_c_idx = match[0]
                        n_atom = mol.GetAtomWithIdx(n_idx)
                        for nbr in n_atom.GetNeighbors():
                            if nbr.GetIdx() == carbonyl_c_idx:
                                continue
                            # BFS the N-substituent fragment so multi-atom
                            # N-substituents (e.g., N-ethyl, N-benzyl) are all covered.
                            stack = [nbr.GetIdx()]
                            seen = {n_idx, carbonyl_c_idx}
                            while stack:
                                cur = stack.pop()
                                if cur in seen:
                                    continue
                                seen.add(cur)
                                effective_pg.add(cur)
                                for nn in mol.GetAtomWithIdx(cur).GetNeighbors():
                                    if nn.GetIdx() not in seen:
                                        stack.append(nn.GetIdx())
            if sub_info.frag_atoms & effective_pg:
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
        # D-FOLLOWON item 9 (P-68.2.2): a CARBON-FREE silyl/germyl chain substituent
        # (-SiH3, -GeH3, -Si(OH)3) would be dropped by Guard 3 (has_carbon False),
        # and the FG-prefix loop cannot emit silyl/germyl (they are substituent
        # prefixes, not functional groups) -> the whole group was lost (the acid
        # collapsed to its de-silylated form, SELF-01-suppressed to unknown). Name it
        # via the shared group-14 namer (the SAME one the benzene RING path uses,
        # Phase 8c) and append it as a prefix. Gated on NOT has_carbon so the
        # carbon-bearing-silyl path (trimethylsilyl etc.) stays byte-identical via
        # classify_and_name_fragment below; fail-closed (None) for any non-Si/Ge,
        # charged, ring, or multivalent centre -> falls through unchanged.
        if not has_carbon:
            _g14_attach = next(
                (fi for fi in sub_info.frag_atoms
                 if any(nb.GetIdx() in chain_set
                        for nb in mol.GetAtomWithIdx(fi).GetNeighbors())),
                None,
            )
            if (_g14_attach is not None
                    and mol.GetAtomWithIdx(_g14_attach).GetSymbol() in ('Si', 'Ge')):
                from ..assembly.substituent_naming import _name_group14_substituent
                _g14 = _name_group14_substituent(
                    mol, list(sub_info.frag_atoms), _g14_attach)
                if _g14:
                    substituent_groups[_g14].append(sub_info.locant)
                    continue
        # W2F-P7 (P-68.3 / P-45.3.1): a CARBON-FREE phosphanyl chain substituent
        # would likewise be lost by Guard 3 (has_carbon False). This is the path a
        # λ5-hydride -PH4 takes: -PH4 is NOT detected as a functional group, so the
        # molecule is NOT polyfunctional (OCC[PH4] = mono primary_alcohol) and the
        # polyfunctional carbon-free namer never runs. The FG-prefix loop cannot
        # emit 'phosphanyl' / 'lambda5-phosphanyl' (substituent prefixes, not FGs),
        # so the whole P was dropped ('ethan-1-ol', SELF-01-suppressed to unknown).
        # Named via the shared phosphorus namer (same one wired into the
        # polyfunctional path); the all-H/organyl-only guard EXCLUDES a
        # phosphoryl/phosphonic P=O. Mirrors the Si/Ge D-FOLLOWON item-9 block.
        if not has_carbon:
            _p_attach = next(
                (fi for fi in sub_info.frag_atoms
                 if any(nb.GetIdx() in chain_set
                        for nb in mol.GetAtomWithIdx(fi).GetNeighbors())),
                None,
            )
            if (_p_attach is not None
                    and mol.GetAtomWithIdx(_p_attach).GetSymbol() == 'P'):
                from ..rules.phosphorus import name_phosphanyl_substituent
                _ph = name_phosphanyl_substituent(
                    mol, list(sub_info.frag_atoms), _p_attach)
                if _ph:
                    substituent_groups[_ph].append(sub_info.locant)
                    continue
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
            # Wave2 T2a (claimed-atom mask, general-acyclic path): the same
            # skip Wave2 T2b added to polyfunctional _POLY_GUARD_FG_TYPES.
            # These FG prefixes fully name their branch (isocyanato/
            # isothiocyanato P-35.2.1, isocyano P-66.5.3, guanidino
            # P-66.4.1.2.2); without the skip this walker re-reads the same
            # atoms as a phantom co-substituent ('formamido-1-isocyanato-
            # ethane' for O=C=NCC — a different molecule).
            'isocyanate', 'isothiocyanate', 'isocyanide', 'guanidine',
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
            # Fallback: simple carbon-count alkyl naming -- but ONLY when that name
            # is honest for this fragment. Counting carbons and spelling an alkyl
            # DISCARDS every other atom, so this line renamed the unnameable
            # -S-Zn-S-CH2CH3 branch of CCS[Zn]SCC 'ethyl' and shipped
            # 'ethylethane', a different molecule. It was masked while the branch
            # namer still leaked the refusal sentinel through as a string
            # ('zinc compound (not supported)ylethane'): visibly broken, so nobody
            # reached this line. Shared guard, not a fourth private copy.
            from .substituent_naming import fragment_is_linear_terminal_alkyl
            _frag_attach = next(
                (i for i in sub_info.frag_atoms
                 if mol.GetBondBetweenAtoms(attach_idx, i) is not None), None)
            carbon_count = len(sub_info.frag_atoms)
            if (carbon_count > 0 and _frag_attach is not None
                    and fragment_is_linear_terminal_alkyl(
                        mol, sub_info.frag_atoms, _frag_attach)):
                try:
                    name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    pass
            if name is None:
                # Fail closed rather than SKIP, when the branch we cannot name
                # carries an element outside the organic set: dropping it emits a
                # name for a different molecule, and unlike the sentinel-bearing
                # string it replaced, 'ethane' for CCS[Zn]SCC is plausible enough
                # that no failure predicate can flag it. Same mid-assembly refusal
                # channel as unsupported_ring_system(), caught once in name().
                from ..errors import _ORGANIC_ELEMENTS, unsupported_element_branch
                for _idx in sub_info.frag_atoms:
                    _sym = mol.GetAtomWithIdx(_idx).GetSymbol()
                    if _sym not in _ORGANIC_ELEMENTS:
                        raise unsupported_element_branch(_sym)
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

        # Omit locant for monosubstituted hydrocarbons at position 1, OR for ANY
        # substituent on a methane parent (chain_length == 1): methane has a single
        # skeletal position, so every substituent is trivially at locant 1 and the
        # locant is omitted even when MULTIPLE distinct substituents are present
        # (DD5 SEN-02: methoxy(methylsulfanyl)methane, NOT 1-methoxy-1-(methyl-
        # sulfanyl)methane; bis(methylsulfanyl)methane). Bracket wrapping for
        # complex substituents is preserved in the omit branch below.
        _prefix_chain_len = len(getattr(features, 'principal_chain', []))
        if should_omit_locant_one(
            context="prefix",
            chain_length=_prefix_chain_len,
            is_monosubstituted=(is_simple_hydrocarbon and total_substituents == 1
                                and sorted_locants == [1]),
        ) and (total_substituents == 1 or _prefix_chain_len == 1):
            # A complex/compound substituent still needs enclosing marks even when
            # its locant is elided (P-16.3.3): '(methylperoxy)ethane',
            # '(methyldisulfanyl)methane'. Mirrors the ring-prefix path. The bare
            # name passes through apply_enclosing_marks (correct ()->[]->{} nesting
            # + leading-stereo escalation; avoids the double-enclose hazard).
            from ..assembly.naming_utils import (
                apply_enclosing_marks, _has_stereo_prefix, get_multiplier_prefix,
            )
            # Preserve the multiplier when the (locant-elided) substituent occurs
            # more than once — e.g. two methylsulfanyl groups on a methane parent
            # -> bis(methylsulfanyl)methane (NOT a single dropped group). The locant
            # is omitted (methane, position 1) but the COUNT must still be cited.
            _mult = get_multiplier_prefix(count, name) if count > 1 else ""
            if is_complex_substituent(name) or _has_stereo_prefix(name):
                formatted = _mult + apply_enclosing_marks(name, depth=-1)
            else:
                formatted = _mult + name
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

    # Case B: O -> CH(aryl)n -> benzyloxy (1 aryl) / diphenylmethoxy (2 phenyl).
    # HYG-04 (Phase 167): single shared aryl-count helper (was inline benzyloxy here).
    _aryl_ether = _name_aryl_methyl_ether(mol, alkyl_start, oxygen_idx)
    if _aryl_ether is not None:
        return _aryl_ether

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


def _amino_branch_atoms(mol, branch_start: int, n_idx: int, chain_set: set) -> list:
    """Collect ONE N-branch's heavy atoms by BFS from ``branch_start``,
    never crossing the amino N (``n_idx``) or the principal chain (w2f p1,
    P-35.4.1 chain-site support)."""
    from collections import deque
    atoms = []
    seen = {n_idx} | set(chain_set)
    queue = deque([branch_start])
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        atoms.append(cur)
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nbr.GetIdx() not in seen and nbr.GetAtomicNum() > 1:
                queue.append(nbr.GetIdx())
    return atoms


def _name_decorated_amino_branch(
    mol, branch_start: int, n_idx: int, chain_set: set
) -> Optional[str]:
    """P-35.4.1 (BB 18112 '(chloromethyl)amino (preferred prefix)'): name ONE
    decorated N-branch of a chain-parent amino substituent via the centralized
    substituent namer, or None — the caller MUST then fail closed. NEVER
    carbon-count a decorated branch: '(methylamino)' for -NH-CH2Cl names a
    DIFFERENT molecule (the w2f p1 root cause).

    Fail-closed guards (mirrors the proven ring-parent path,
    rules/benzene.py:1430-1475):
      * ring-bearing branch -> None (parent-hydride seniority competition
        belongs to parent selection; the sub_has_ring branches at
        composer.py own ring N-branches);
      * producer None or space-bearing output -> None (garbled producer,
        e.g. 'methylboronic acidyl' for -CH2-B(OH)2 — the benzene.py:1448
        space-guard rationale).

    Returns the RAW branch name WITHOUT enclosing marks ('chloromethyl',
    '1-chloroethyl', '2-hydroxyethyl'; locants anchored at the free valence);
    the call sites own P-16.5.1.1/P-16.5.1.3.1 parenthesization via
    _assemble_decorated_amino_prefix.
    """
    branch_atoms = _amino_branch_atoms(mol, branch_start, n_idx, chain_set)
    if not branch_atoms:
        return None
    ring_info = mol.GetRingInfo()
    if any(ring_info.NumAtomRings(a) > 0 for a in branch_atoms):
        return None
    from .substituent_naming import name_substituent_fragment
    name = name_substituent_fragment(
        mol, branch_atoms, branch_start, sorted(chain_set) + [n_idx]
    )
    if not name or ' ' in name:
        return None
    return name


def _assemble_decorated_amino_prefix(branch_entries) -> Optional[str]:
    """Assemble the P-35.4.1 compound amino prefix from named N-branches.

    ``branch_entries``: list of ``(raw_branch_name, decorated: bool)``. At
    least one entry is decorated — the pure-alkyl fast path never calls this
    (byte-stability contract for '(methylamino)'/'(dimethylamino)').

    Grammar (Blue-Book-cited in  §1.C/§1.E):
      * decorated branch names are ALWAYS parenthesized (P-16.5.1.1: parens
        around ALL compound prefixes): '(chloromethyl)';
      * k identical decorated branches take bis/tris OUTSIDE the parens
        (P-16.5.1.10; BB 40703 'bis(chloromethyl)aminoxyl (PIN)');
      * distinct branches cite in alphanumerical order via alpha_sort_key
        (letters-only: '2-hydroxyethyl' sorts at 'h' — NEVER raw sorted());
      * a simple-alkyl branch is bare when FIRST-cited, parenthesized after
        (P-16.5.1.3.1; BB 26308 '3-[methyl(phenyl)amino]phenol');
      * the whole prefix takes the outer mark ESCALATED via
        apply_enclosing_marks(-1) (P-16.5.2.4): '[(chloromethyl)amino]',
        cited verbatim by the composer prefix-joiner like the legacy
        '(methylamino)' convention.
    """
    from collections import Counter
    from .naming_utils import (
        COMPLEX_MULTIPLIERS,
        SIMPLE_MULTIPLIERS,
        alpha_sort_key,
        apply_enclosing_marks,
    )
    counts = Counter(branch_entries)
    cited = []
    ordered = sorted(counts.items(), key=lambda kv: alpha_sort_key(kv[0][0]))
    for i, ((bname, decorated), k) in enumerate(ordered):
        if decorated:
            if k == 1:
                cited.append(f"({bname})")
            else:
                mult = COMPLEX_MULTIPLIERS.get(k)
                if mult is None:
                    return None
                cited.append(f"{mult}({bname})")
        else:
            if k == 1:
                cited.append(bname if i == 0 else f"({bname})")
            else:
                mult = SIMPLE_MULTIPLIERS.get(k)
                if mult is None:
                    return None
                cited.append(f"{mult}{bname}" if i == 0 else f"{mult}({bname})")
    return apply_enclosing_marks("".join(cited) + "amino", -1)


def _check_for_acylamino(mol, sub_atoms: List[int], principal_chain: List[int]) -> Optional[str]:
    """
    Check if a substituent is an acylamino group: -NH-C(=O)-R or -N(R)-C(=O)-R.

    Pattern: nitrogen bonded to chain, also bonded to a carbonyl carbon C(=O),
    which in turn is bonded to an alkyl chain R.

    P-66.1.1.4.3 method (1) generates the PIN: the amido-family prefix
    (formamido/acetamido/{stem}anamido, benzamido for ring acyls), returned
    BARE — amido prefixes are simple and take no enclosing marks
    (4-formamidobenzoic acid). Locant-bearing forms self-wrap:
    '(4-methylbenzamido)'. Branches the strict amido builder cannot fully
    describe keep the legacy method-(2) '({acyl}amino)' fallback.
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
            # AM-6 (P-66.1.1.4.3 method (1), BB 33019): an R-SO2-NH- branch on
            # a chain is the '(...sulfonamido)' prefix (sulfonamide final 'e' ->
            # 'o'), NOT an 'amino' split of the N. Only fires when the N attaches
            # to the chain via a sulfonyl S (SX4, two =O) and carries no other
            # heavy substituent (heavy degree 2: chain + S). N-alkyl sulfonamido
            # (BB 33029 N-cyclopropyl) fails closed to None (unknown).
            if atom.GetDegree() == 2:
                _sulfonyl = None
                for _sn in atom.GetNeighbors():
                    if _sn.GetSymbol() != 'S' or _sn.GetIdx() not in sub_set:
                        continue
                    _dbl_o = [
                        _o.GetIdx() for _o in _sn.GetNeighbors()
                        if _o.GetSymbol() == 'O' and _o.GetIdx() in sub_set
                        and mol.GetBondBetweenAtoms(
                            _sn.GetIdx(), _o.GetIdx()
                        ).GetBondTypeAsDouble() == 2.0
                    ]
                    _rest = [
                        _x.GetIdx() for _x in _sn.GetNeighbors()
                        if _x.GetIdx() != idx and _x.GetIdx() not in _dbl_o
                    ]
                    if len(_dbl_o) == 2 and len(_rest) == 1:
                        _sulfonyl = (_sn.GetIdx(), _rest[0])
                        break
                if _sulfonyl is not None:
                    _s_idx, _acyl = _sulfonyl
                    _n = _pure_linear_alkyl_len(mol, _acyl, {_s_idx})
                    if _n is not None and _n >= 1:
                        return f"({get_chain_prefix(_n)}anesulfonamido)"
                    # isolated benzene wholly in sub -> (benzenesulfonamido)
                    _ra = mol.GetAtomWithIdx(_acyl)
                    if _ra.GetIsAromatic() and _ra.GetSymbol() == 'C':
                        _ri_s = mol.GetRingInfo()
                        for _ring in _ri_s.AtomRings():
                            if (_acyl in _ring and len(_ring) == 6
                                    and all(r in sub_set for r in _ring)
                                    and all(
                                        mol.GetAtomWithIdx(r).GetIsAromatic()
                                        and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                                        for r in _ring)):
                                _rset = set(_ring)
                                if not any(set(o) != _rset and set(o) & _rset
                                           for o in _ri_s.AtomRings()):
                                    return "(benzenesulfonamido)"
                    # sulfonyl-N present but the acyl is un-nameable -> fail closed
                    return None
            # PF-2 (P-66.4.2.3.5, BB 34537): an -N(H)-C(=N-NH2)-R amidrazone
            # branch is the '(...hydrazonamido)' prefix (analogous to acetamido
            # but with =N-NH2 in place of =O). Detect an in-branch hydrazono C
            # neighbour of the N and delegate to the shared builder BEFORE the
            # alkylamino fallback (which would drop the =N-NH2 as '(ethylamino)').
            for _cn in atom.GetNeighbors():
                if _cn.GetSymbol() != 'C' or _cn.GetIdx() not in sub_set:
                    continue
                _is_hydrazono = any(
                    _x.GetSymbol() == 'N'
                    and mol.GetBondBetweenAtoms(
                        _cn.GetIdx(), _x.GetIdx()).GetBondTypeAsDouble() == 2.0
                    and any(_y.GetSymbol() == 'N' and _y.GetDegree() == 1
                            for _y in _x.GetNeighbors()
                            if _y.GetIdx() != _cn.GetIdx())
                    for _x in _cn.GetNeighbors()
                )
                if _is_hydrazono:
                    from .substituent_naming import (
                        hydrazonoyl_amido_prefix_from_branch,
                    )
                    _hp = hydrazonoyl_amido_prefix_from_branch(
                        mol, idx, _cn.GetIdx(), sub_atoms
                    )
                    if _hp:
                        return f"({_hp})"
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

            # Count carbons PER N-branch via C-C bonds only (Phase 167
            # HYG-04 site#2 multiplicity fix preserved). w2f p1 (P-35.4.1):
            # a branch carrying ANY non-C heavy atom is DECORATED and must
            # be named by the centralized producer — carbon-counting it
            # drops the decoration ('(methylamino)' for -NH-CH2Cl = a
            # DIFFERENT molecule). Pure-carbon branches keep the legacy
            # _count_carbon_chain + get_alkyl_name path BYTE-IDENTICAL.
            _branch_entries = []
            _branch_impure = False
            _decorated_failed = False
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in chain_set:
                    continue
                if nbr.GetSymbol() == 'C':
                    _batoms = _amino_branch_atoms(
                        mol, nbr.GetIdx(), idx, chain_set)
                    if any(mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)
                           for a in _batoms):
                        _bname = _name_decorated_amino_branch(
                            mol, nbr.GetIdx(), idx, chain_set)
                        if _bname is None:
                            _decorated_failed = True
                        else:
                            _branch_entries.append((_bname, True))
                        continue
                    _bc = _count_carbon_chain(mol, nbr.GetIdx(), chain_set | {idx})
                    if _bc <= 0:
                        continue
                    try:
                        _branch_entries.append((get_alkyl_name(_bc), False))
                    except (ValueError, KeyError):
                        _branch_impure = True
            if _decorated_failed:
                # P-35.4.1 fail-closed: never the truncated alkyl name, never
                # a branch-dropping 'amino'. (The chain-bonded N is unique per
                # substituent, so no later iteration can re-emit this amino.)
                return None
            if any(_d for _n, _d in _branch_entries):
                if _branch_impure:
                    return None
                return _assemble_decorated_amino_prefix(_branch_entries)
            _branch_alkyls = [_n for _n, _d in _branch_entries]
            if _branch_alkyls and not _branch_impure:
                if len(_branch_alkyls) == 1:
                    return f"({_branch_alkyls[0]}amino)"
                from collections import Counter as _Counter

                _counts = _Counter(_branch_alkyls)
                _parts = []
                for _nm in sorted(_counts.keys()):
                    _c = _counts[_nm]
                    _parts.append(
                        _nm if _c == 1
                        else f"{SIMPLE_MULTIPLIERS.get(_c, str(_c))}{_nm}"
                    )
                return f"({''.join(_parts)}amino)"
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
                        # Wave2 T1c (P-66.1.1.4.3): method (1) amido prefix
                        # is the PIN — benzamido, (4-methylbenzamido),
                        # (naphthalene-1-carboxamido). Only when the walked
                        # acyl fragment covers the whole substituent (the
                        # carbon BFS above drops heteroatom decorations, and
                        # a name must never claim atoms it dropped).
                        if set(sub_atoms) == _af | {idx}:
                            from .substituent_naming import (
                                acid_name_to_amido_prefix,
                            )
                            _amido = acid_name_to_amido_prefix(_an)
                            if _amido:
                                if (any(ch.isdigit() for ch in _amido)
                                        or '-' in _amido):
                                    return f"({_amido})"
                                return _amido
                        from ..decomposition.fragment_assembly import (
                            _acid_to_acyl,
                        )
                        _ac = _acid_to_acyl(_an)
                        if _ac:
                            return f"({_ac}amino)"
            except Exception:
                pass  # Fall through to linear chain logic

        # Wave2 T1c (P-66.1.1.4.3): method (1) — the amido-family prefix
        # (formamido/acetamido/{stem}anamido) is the PIN; the acylamino form
        # below is method (2), not preferred. The strict builder emits only
        # when the branch is exactly -NH-CO-(unbranched saturated carbon
        # chain) with every substituent atom covered, so it can never claim
        # atoms it dropped. Impure branches keep the legacy fallback (which
        # the self-consistency gate already blocks; the Tier-2 claimed-atom
        # mask owns that class).
        from .substituent_naming import linear_acyl_amido_prefix
        _amido_nm = linear_acyl_amido_prefix(mol, carbonyl_c, idx, sub_atoms)
        if _amido_nm:
            return _amido_nm

        # Count carbons from carbonyl C through C-C bonds only
        # (don't traverse through N to reach other peptide fragments)
        exclude = chain_set | {idx}  # exclude chain and the N
        if carbonyl_o is not None:
            exclude.add(carbonyl_o)  # exclude the C=O oxygen
        acyl_carbons = _count_carbon_chain(mol, carbonyl_c, exclude)
        if acyl_carbons == 0:
            acyl_carbons = 1  # at minimum the carbonyl C

        # Legacy fallback (method (2), non-preferred): (prefixanoylamino)
        # with enclosing parens — kept only for branches the strict amido
        # builder cannot fully describe.
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
                # Phase 160.2 Plan-04-01 CR-01 guard per IUPAC P-66.6.4:
                # If the carbonyl C has an N neighbor inside the fragment,
                # this is a CARBAMATE (-OC(=O)NHR / -OC(=O)NH2), NOT a simple
                # acyloxy. Per IUPAC P-66.6.4 Branch B the prefix is
                # ``carbamoyloxy`` (or N-substituted variant), and the naming
                # belongs to the Tier-0.5 prefix-form hook downstream, not to
                # the acyloxy generator. Skipping here ensures the next
                # fallback (_name_heteroatom_substituent → Tier-0.5
                # _check_substituent_prefix_form) sees the fragment and
                # routes it to ``carbamoyloxy``. Without this guard, the
                # acyloxy generator drops the N atom and produces
                # ``methanoyloxy`` for the canonical CR-01 fixture
                # (``O=C(N)OCCCC(=O)O``), which OPSIN round-trips to a
                # different molecule (missing N+H per RESEARCH §2).
                # See 160.2-AUDIT-DECOMP-CLOSURE.md §5 + 160.1-REVIEW.md CR-01.
                has_nitrogen_on_carbonyl = any(
                    nbr2.GetSymbol() == 'N' and nbr2.GetIdx() in sub_set
                    for nbr2 in nbr.GetNeighbors()
                )
                if has_nitrogen_on_carbonyl:
                    return None

                # R5 convergence: name the acyl fragment with the ring-aware
                # acid namer instead of _count_carbon_chain, which linearises a
                # ring acyl (benzoyl -> phantom "heptanoyl", the heptanoyloxy
                # bug).  Collect the whole acyl fragment R-C(=O)- by BFS from the
                # carbonyl C, excluding the ester oxygen (the bond to the parent
                # chain); get_acid_fragment_name then returns "benzoic" for a
                # benzoyl ring acyl (same primitive esters.name_ester_as_prefix
                # uses).  Returns the BARE prefix (callers add parens).
                acyl_atoms = set()
                _stack = [nbr_idx]
                _seen = {idx}
                while _stack:
                    _a = _stack.pop()
                    if _a in _seen:
                        continue
                    _seen.add(_a)
                    acyl_atoms.add(_a)
                    for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                        if _nb.GetIdx() not in _seen:
                            _stack.append(_nb.GetIdx())
                from ..rules.esters import (
                    get_acid_fragment_name,
                    get_acyloxy_prefix as _get_acyloxy,
                )
                try:
                    acid_name = get_acid_fragment_name(mol, list(acyl_atoms))
                    if acid_name:
                        return _get_acyloxy(acid_name)
                except (ValueError, KeyError):
                    pass

    return None


def _name_heteroatom_substituent(mol, sub_atoms: List[int], principal_chain: List[int]) -> Optional[str]:
    """Orchestrator for heteroatom-substituent fallback naming (Phase 160.2 D-05).

    Runs Tier-0.5 (P-65/P-66) prefix-form check (Phase 160.1 D-04), identifies the
    attachment atom, then dispatches by attach-symbol + structural class to one of
    three in-file helpers (CONTEXT D-13). Verbatim mechanical lift per Phase 145.1
    D-09; each helper re-derives local scope as the only edit. Returns None if no
    symbol branch applies (preserves pre-Plan-03 contract).
    """
    chain_set = set(principal_chain)
    sub_set = set(sub_atoms)

    # ---- Tier 0.5 (Phase 160.1 D-04): IUPAC P-65 / P-66 prefix-form check ----
    try:
        from .substituent_prefix_forms import _check_substituent_prefix_form
        # Identify attach atom (matches the logic below; if undetermined,
        # any atom of the fragment is acceptable for the SMARTS subset check).
        _attach = None
        for _idx in sub_atoms:
            for _nbr in mol.GetAtomWithIdx(_idx).GetNeighbors():
                if _nbr.GetIdx() in chain_set:
                    _attach = _idx
                    break
            if _attach is not None:
                break
        if _attach is None and sub_atoms:
            _attach = sub_atoms[0]
        prefix_form = _check_substituent_prefix_form(mol, sub_set, _attach)
        if prefix_form is not None:
            return prefix_form
    except Exception:
        pass

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

    # Dispatch by attachment-atom symbol + structural class (Phase 160.2 D-05)
    if symbol == 'N':
        return _name_n_attached_substituent_fallback(
            mol, sub_atoms, sub_set, chain_set, attach_atom
        )
    if symbol == 'O':
        return None  # handled by FG prefixes (kept inline per CONTEXT D-05)
    if symbol == 'C':
        ring_info = mol.GetRingInfo()
        sub_has_ring = any(
            ring_info.NumAtomRings(idx) > 0 for idx in sub_atoms
        )
        if sub_has_ring:
            return _name_c_attached_ring_substituent_fallback(
                mol, sub_atoms, sub_set, chain_set, attach_atom
            )
        return _name_c_attached_chain_substituent_fallback(
            mol, sub_atoms, sub_set, chain_set, attach_atom
        )
    # BBR-PERC (169.7): Se/Te-attached substituent → (alkyl)selanyl/tellanyl (P-63.6).
    # Without this branch a -Se-R / -Te-R substituent falls through to the Tier-4
    # recursive namer, producing an OPSIN-unparseable "methaneselenolyl" form that the
    # validity gate then suppresses to "unknown". Reuses the chalcogen-agnostic prefix
    # builder (the chalcogen is the attach atom; it walks the chalcogen's C neighbours).
    if symbol in ('Se', 'Te'):
        from .substituent_prefix_forms import get_sulfanyl_prefix
        suffix = 'selanyl' if symbol == 'Se' else 'tellanyl'
        nbr_idxs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbr_idxs) >= 2:
            return get_sulfanyl_prefix(
                mol, (attach_atom, nbr_idxs[0], nbr_idxs[1]),
                principal_chain, suffix=suffix,
            )

    return None


def _name_n_attached_substituent_fallback(
    mol, sub_atoms: List[int], sub_set: set, chain_set: set, attach_atom: int
) -> Optional[str]:
    """N-attached heteroatom substituent fallback (DECOMP-05 closure helper #1).

    Per Phase 160.2 CONTEXT D-05 + Phase 145.1 D-09 mechanical-lift: VERBATIM lift
    of the prior `_name_heteroatom_substituent` N-branch body (composer.py:5253-5407
    in pre-Plan-03 layout). Covers Phase 79-02 fused het detection, aniline matching,
    Phase 79-01 direct ring identification, recursive ring sub, alkyl chain count +
    (alkylamino) prefix routing. IUPAC P-66.6 (amines) + P-29 substituent grammar.

    Per AP-160.2-07: cross-branch shared helpers (`_count_carbon_chain`,
    `_has_ring_atoms`, `_name_reflects_ring`, `get_alkyl_name`, etc.) STAY in
    composer.py at their current line ranges; this helper references them via
    Python module-level scope (no imports needed within composer.py).
    """
    # Re-derive locals that lived in orchestrator scope:
    atom = mol.GetAtomWithIdx(attach_atom)
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

    # Count carbons PER N-branch via C-C bonds only (Phase 167 HYG-04
    # multiplicity fix preserved). w2f p1 (P-35.4.1): a branch carrying ANY
    # non-C heavy atom is DECORATED -> centralized producer or fail-closed
    # None (never the branch-dropping 'amino', never the carbon-count name).
    # Pure-carbon branches keep the legacy path BYTE-IDENTICAL.
    branch_entries = []
    branch_impure = False
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() in chain_set:
            continue
        if nbr.GetSymbol() == 'C':
            _batoms = _amino_branch_atoms(
                mol, nbr.GetIdx(), attach_atom, chain_set)
            if any(mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)
                   for a in _batoms):
                _bname = _name_decorated_amino_branch(
                    mol, nbr.GetIdx(), attach_atom, chain_set)
                if _bname is None:
                    return None  # P-35.4.1 fail-closed
                branch_entries.append((_bname, True))
                continue
            bc = _count_carbon_chain(mol, nbr.GetIdx(), chain_set | {attach_atom})
            if bc <= 0:
                continue
            try:
                branch_entries.append((get_alkyl_name(bc), False))
            except (ValueError, KeyError):
                branch_impure = True

    if any(_d for _n, _d in branch_entries):
        if branch_impure:
            return None  # decorated + un-nameable sibling -> fail closed
        return _assemble_decorated_amino_prefix(branch_entries)
    branch_alkyls = [_n for _n, _d in branch_entries]
    if not branch_alkyls or branch_impure:
        return "amino"
    if len(branch_alkyls) == 1:
        # Single N-alkyl: exact legacy form (byte-identical).
        return f"({branch_alkyls[0]}amino)"
    # N,N-dialkyl(+): multiplicity prefix, e.g. -N(CH3)2 -> (dimethylamino).
    from collections import Counter as _Counter

    _counts = _Counter(branch_alkyls)
    _parts = []
    for _nm in sorted(_counts.keys()):
        _c = _counts[_nm]
        _parts.append(_nm if _c == 1 else f"{SIMPLE_MULTIPLIERS.get(_c, str(_c))}{_nm}")
    return f"({''.join(_parts)}amino)"


def _name_c_attached_ring_substituent_fallback(
    mol, sub_atoms: List[int], sub_set: set, chain_set: set, attach_atom: int
) -> Optional[str]:
    """C-attached ring-containing substituent fallback (DECOMP-05 closure helper #2).

    Per Phase 160.2 CONTEXT D-05 + Phase 145.1 D-09 mechanical-lift: VERBATIM lift
    of the prior `_name_heteroatom_substituent` C-branch ring sub-body
    (composer.py:5424-5534 in pre-Plan-03 layout). Covers Phase 79-02 fused het
    detection (C-branch), Phase 79-01 direct-ring identification, recursive ring
    sub, fragment-recursive fallback. IUPAC P-23 + P-29 substituent grammar.

    Per AP-160.2-07: cross-branch shared helpers (`_has_ring_atoms`,
    `_name_reflects_ring`, `name_substituent_fragment`, `name_fragment_recursively`,
    `parent_to_prefix`, etc.) STAY in composer.py at their current line ranges;
    this helper references them via Python module-level scope.
    """
    # Re-derive ring_info that lived in orchestrator scope:
    ring_info = mol.GetRingInfo()

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


def _name_c_attached_chain_substituent_fallback(
    mol, sub_atoms: List[int], sub_set: set, chain_set: set, attach_atom: int
) -> Optional[str]:
    """C-attached chain-only substituent fallback (DECOMP-05 closure helper #3).

    Per Phase 160.2 CONTEXT D-05 + Phase 145.1 D-09 mechanical-lift: VERBATIM lift
    of the prior `_name_heteroatom_substituent` C-branch chain sub-body
    (composer.py:5540-5704 in pre-Plan-03 layout). Covers BUG-A haloalkyl
    (fluoroalkyl / chloroalkyl / bromoalkyl / iodoalkyl rendering with BFS-locant
    positioning), BUG-D hydroxyalkyl, BUG-C aminoalkyl, plain alkyl fallback.
    IUPAC P-29 + P-65 + P-66 substituent grammar.

    Per AP-160.2-07: cross-branch shared helpers (`_count_carbon_chain`,
    `get_alkyl_name`, `get_multiplier_prefix`, etc.) STAY in composer.py at
    their current line ranges; this helper references them via Python
    module-level scope.
    """
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

        # W2F-P3 (defect b-het, P-45.5 / P-29.2): do NOT fall back to a carbon-count
        # alkyl name here — it DROPS heteroatom FGs + the branch shape
        # (-CH(NO2)CH(NO2)CH3 -> 'propyl'), shadowing the faithful Tier-4 enumerator
        # (classify_and_name_fragment names it '1,2-dinitropropyl', verified). Return
        # None so the sole runtime caller (polyfunctional.py:2100 via
        # _name_heteroatom_substituent) reaches the enumerator fallback. Side benefit:
        # >3C haloalkyl branches (past the total_carbons<=3 gate) also reach it.

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


def _pure_linear_alkyl_len(mol, start_idx: int, exclude: set) -> Optional[int]:
    """Length of a pure, unbranched, acyclic, saturated all-carbon chain.

    Returns the carbon count only when EVERY atom reachable from ``start_idx``
    (via single C-C bonds, ignoring ``exclude``) is a non-ring carbon whose
    only heavy neighbours are other chain carbons or an atom in ``exclude``
    (the attachment point), and the chain is unbranched (no carbon has >2
    in-fragment carbon neighbours). Returns ``None`` for branched, cyclic,
    unsaturated, or hetero-decorated fragments so callers fail closed. Used by
    the AM-6 (P-66.1.1.4.3) sulfonamido recognizer.
    """
    from collections import deque

    ri = mol.GetRingInfo()
    frag = set()
    queue = deque([start_idx])
    while queue:
        a = queue.popleft()
        if a in frag or a in exclude:
            continue
        at = mol.GetAtomWithIdx(a)
        if at.GetSymbol() != 'C' or ri.NumAtomRings(a) > 0:
            return None
        frag.add(a)
        for nb in at.GetNeighbors():
            ni = nb.GetIdx()
            if ni in exclude:
                continue
            bond = mol.GetBondBetweenAtoms(a, ni)
            if nb.GetSymbol() == 'C':
                if bond.GetBondTypeAsDouble() != 1.0:
                    return None  # unsaturated -> different stem
                queue.append(ni)
            else:
                return None  # hetero decoration -> impure
    if not frag:
        return None
    for a in frag:
        deg = sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                  if nb.GetIdx() in frag)
        if deg > 2:
            return None  # branched
    return len(frag)


def _convert_yl_to_ylidene(name: str) -> str:
    """Give a ring substituent's name the P-29.2 double-valence morphology.

    IUPAC P-29.2 / P-31.1.3.1: a substituent joined to its parent by a double
    bond ends in ``-ylidene``, not ``-yl``.

    IDEMPOTENT, and that is the point. The substituent pipeline now reads the
    attachment bond order itself and hands back ``methylidene`` already formed,
    so a token that already spells two free valences is returned untouched --
    only a token that confidently spells ONE is given the ending it is missing.
    Anything the morphology oracle cannot read (``oxo``, a ``-diyl``) is left
    alone rather than rewritten on a guess.

    The rewrite survives for the ``get_alkyl_name(carbon_count)`` fallback at
    the two call sites, which counts carbons and can only produce the ``-yl``
    form. Retiring it belongs with retiring that fallback.
    """
    from ..validation.name_morphemes import free_valence_morphology
    estimate = free_valence_morphology(name)
    if estimate.confident and estimate.free_valences == 1:
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
            # P-62.3.1.2 (BB 26540): distinguish exocyclic =NH (imino
            # prefix, in the presence of a senior C=O suffix) from -NH2
            # (amino). Mirror the oxygen branch's bond-order check. An
            # N-substituted =N-R (degree 2) is NOT bare 'imino' and would
            # drop the R substituent -> fail closed (None).
            bond = mol.GetBondBetweenAtoms(idx, attach_mol_idx)
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                if atom.GetFormalCharge() == 0 and atom.GetDegree() == 1:
                    return 'imino'
                return None
            return 'amino'
        if sym == 'S':
            return 'sulfanyl'

    # Multi-atom FG-only fragments (e.g., -NO2 = nitro, -N3 = azido)
    # These are complex; let the global FG loop handle them with its
    # SMARTS-based detection for correct classification.
    return None


def ring_anchored_pg_atoms(mol, pg_matches, ring_atom_set: set) -> set:
    """Atoms of principal-group matches ANCHORED to the parent ring.

    A match is ring-anchored when it contains a ring atom (inline suffixes:
    ring C=O ketone, ring C-OH alcohol) or when one of its carbons is bonded
    directly to a ring atom (appended suffixes: -carbaldehyde, -carboxylic
    acid, -carbonitrile, whose match atoms are all exocyclic).

    P-44.1.2.2 co-delivery (v21 WS-A.1 S4): only ring-anchored matches are
    expressed as the ring suffix. A PG match wholly inside a demoted chain
    substituent (e.g., the terminal CHO of a 7-oxoheptyl chain) must stay
    with the substituent and be named there (oxo/cyano/... prefix).
    """
    anchored: set = set()
    for match in (pg_matches or []):
        match_set = set(match)
        if match_set & ring_atom_set:
            anchored.update(match_set)
            continue
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != 'C':
                continue
            if any(nbr.GetIdx() in ring_atom_set for nbr in atom.GetNeighbors()):
                anchored.update(match_set)
                break
    return anchored


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

    # Collect principal group atoms to skip substituents overlapping with them.
    # v21 WS-A.1 S4: only RING-ANCHORED matches count — they are the suffix.
    # A PG match wholly inside a chain substituent (terminal CHO of an
    # oxoalkyl chain) must NOT suppress that substituent (it previously
    # dropped the whole chain: 'cyclopentanedicarbaldehyde' bug).
    ring_set = set(oriented_ring)
    pg_atom_set = ring_anchored_pg_atoms(
        mol, features.principal_group_atoms, ring_set
    )

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

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

        # Wave2 T2a (claimed-atom mask, ring-parent path): mirror of guard 3b
        # in _generate_alkyl_prefixes / _POLY_GUARD_FG_TYPES in polyfunctional.
        # A branch fully covered by one of these FG matches is emitted by the
        # global FG-prefix loop (isocyanato/isothiocyanato/isocyano/guanidino);
        # re-walking it here mis-reads the heterocumulene ('formamido-1-
        # isocyanatocyclohexane', 'isothiocyanic acidyl' garbage).
        _RING_GUARD_FG_TYPES = (
            'isocyanate', 'isothiocyanate', 'isocyanide', 'guanidine',
        )
        _skip_ring_fg_branch = False
        for _fg_name in _RING_GUARD_FG_TYPES:
            if _fg_name == features.principal_group:
                continue
            for _fg_match in features.functional_groups.get(_fg_name, []):
                if _fg_match and sub_info.frag_atoms.issubset(set(_fg_match)):
                    _skip_ring_fg_branch = True
                    break
            if _skip_ring_fg_branch:
                break
        if _skip_ring_fg_branch:
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
    # WSD-06 (NUM-01): the substituent-locant is omitted only when the ring is
    # genuinely SYMMETRIC (one substitutable position), not merely monosubstituted.
    # The old count proxy dropped '3-' from a substituted cycloalkene
    # (`bromocyclohexene` for `3-bromocyclohex-1-ene`). Compute the real
    # topological-symmetry predicate on the PARENT HYDRIDE (ring + unsaturation,
    # substituents stripped) and AND it with the count.
    from ..assembly.naming_utils import is_only_one_substitutable_position
    _parent_hydride = None
    try:
        _frag = Chem.MolFragmentToSmiles(mol, atomsToUse=list(oriented_ring))
        _parent_hydride = Chem.MolFromSmiles(_frag)
    except Exception:
        _parent_hydride = None
    _ring_symmetric = (
        is_only_one_substitutable_position(_parent_hydride)
        if _parent_hydride is not None else (total_all_substituents == 1)
    )
    is_monosubstituted = (total_all_substituents == 1) and _ring_symmetric

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
            # Monosubstituted carbocyclic ring: omit locant (it's always 1).
            # SUB-05/D-17: complex OR stereo-prefixed substituents need
            # enclosing marks (P-16.3.3). The old guard skipped enclosing for a
            # leading-stereo name ((R)-3-methylpentyl starts with '(') -> the
            # broken (R)-3-methylpentylbenzene. Route through apply_enclosing_marks
            # passing the BARE name (it does correct ()->[]->{} nesting +
            # leading-stereo escalation; passing the bare name avoids the
            # double-enclose hazard).
            from ..assembly.naming_utils import apply_enclosing_marks, _has_stereo_prefix
            if is_complex_substituent(name) or _has_stereo_prefix(name):
                formatted = apply_enclosing_marks(name, depth=-1)
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


# Phase 160.2 Plan-02-01: _generate_stereodescriptors lifted to handlers/_handler_shared.py per CONTEXT D-02 + D-03.


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


# Phase 179 (D-03): _estimate_parent_size_from_name, _build_unsaturation_infix,
# and _build_hydrocarbon_name were lifted VERBATIM to composition_primitives.py
# (the single shared composition-grammar source). They are re-exported via the
# top-of-module `from .composition_primitives import (...)` so all composer call
# sites resolve unchanged. _assemble_fragments itself lives in
# handlers/_handler_shared.py per CONTEXT D-02 (Phase 160.2).


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


# Phase 179 (D-03): _join_prefixes and _join_prefix_to_name were lifted VERBATIM
# to composition_primitives.py and are re-exported via the top-of-module import.


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
