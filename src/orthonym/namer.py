"""
Main IUPAC nomenclature generator.

Architecture:
    SMILES → Perception → Classification → Assembly → IUPAC Name

This is the inverse of OPSIN's pipeline:
    OPSIN:     Name → Tokenize → Parse → Build Structure
    Orthonym: Structure → Perceive → Classify → Assemble Name
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

logger = logging.getLogger(__name__)

from .perception.ions import detect_species_type, get_ion_sites, get_radical_sites
from .perception.functional_groups import detect_functional_groups
from .perception.chains import find_principal_chain
from .perception.rings import get_ring_systems, get_ring_info, is_aromatic_ring, classify_ring, get_complete_ring_atom_set
from .perception.stereo import get_stereocenters, get_double_bond_stereo, _CIP_ASSIGNED_PROP, assign_stereochemistry
from .rules.seniority import get_principal_group
from .rules.locants import orient_chain, build_atom_to_locant
from .rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
from .assembly.composer import assemble_name
from .data import ALL_RETAINED_NAMES as RETAINED_NAMES


# ---------------------------------------------------------------------------
# Universal stereo backstop (Phase 140, STER-16)
# ---------------------------------------------------------------------------


def _final_stereo_check(mol, name: str, handler: str = 'unknown') -> str:
    """Universal stereo backstop: detect missing stereodescriptors in name.

    Runs AFTER all handler-specific stereo injection. Only activates when
    a handler missed stereo. Logs WARNING to flag handler gaps for future fixes.

    This is an architectural safety net per Phase 140 D-02.  It does NOT
    inject stereo with raw atom-index locants because those don't correspond
    to IUPAC numbering for the named parent structure -- injecting them would
    produce incorrect names.  Handler-specific injection via
    _inject_stereo_if_missing() remains the primary stereo injection mechanism.

    Args:
        mol: RDKit Mol object (with stereo info from original SMILES)
        name: Generated IUPAC name (may or may not contain stereo)
        handler: Name of the handler that produced this name (for attribution)

    Returns:
        Original name unchanged.  Logs WARNING if stereo gap detected.
    """
    if mol is None or not name or name == 'unknown':
        return name

    # Already has stereo prefix? Use same regex as _inject_stereo_if_missing
    if re.match(r'\(\d*[RSrsEZez](,\d*[RSrsEZez])*\)-', name):
        return name

    # Check if stereo descriptors appear ANYWHERE in the name (e.g., already
    # embedded by a handler within the name body).  Pattern: digit(s) followed
    # by R/S/E/Z inside parentheses -- matches (2R), (3S,5R), (E), etc.
    if re.search(r'\(\d*[RSEZrsez](,\d*[RSEZrsez])*\)', name):
        return name

    # Names using traditional carbohydrate/amino acid stereo notation
    # (alpha/beta, D-/L-) already convey stereochemistry -- don't flag.
    if re.search(r'(alpha|beta|alfa)-[DL]-', name, re.IGNORECASE):
        return name

    # Does molecule have stereo?
    assign_stereochemistry(mol)
    has_atom_stereo = any(a.HasProp('_CIPCode') for a in mol.GetAtoms())
    has_bond_stereo = any(b.HasProp('_CIPCode') for b in mol.GetBonds())
    if not has_atom_stereo and not has_bond_stereo:
        return name

    # Log the handler gap for future fixes.
    n_atom_stereo = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    n_bond_stereo = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
    logger.warning(
        "Stereo backstop: '%s' (handler: %s) has %d R/S + %d E/Z but name lacks "
        "descriptors. Fix handler to include stereo natively.",
        name[:50], handler, n_atom_stereo, n_bond_stereo
    )
    return name


# ---------------------------------------------------------------------------
# Compound class pre-routing (Phase 141, CLASS-06)
# ---------------------------------------------------------------------------

# Pre-compiled SMARTS for sugar ring detection
_PYRANOSE_SMARTS = None
_FURANOSE_SMARTS = None
_RING_OH_SMARTS = None


def _get_sugar_smarts():
    """Lazy-initialize SMARTS patterns for sugar ring detection."""
    global _PYRANOSE_SMARTS, _FURANOSE_SMARTS, _RING_OH_SMARTS
    if _PYRANOSE_SMARTS is None:
        # 6-membered ring with 1 O and 5 C (pyranose)
        _PYRANOSE_SMARTS = Chem.MolFromSmarts("[OX2;r6]1[CX4][CX4][CX4][CX4][CX4]1")
        # 5-membered ring with 1 O and 4 C (furanose)
        _FURANOSE_SMARTS = Chem.MolFromSmarts("[OX2;r5]1[CX4][CX4][CX4][CX4]1")
        # OH group on a ring carbon
        _RING_OH_SMARTS = Chem.MolFromSmarts("[C;r]([OX2H])")
    return _PYRANOSE_SMARTS, _FURANOSE_SMARTS, _RING_OH_SMARTS


def _has_sugar_ring_pattern(mol) -> bool:
    """Detect pyranose/furanose ring with minimum 2 OH groups.

    Uses SMARTS matching to identify sugar-like rings:
    - Pyranose: 6-membered ring with 1 O and 5 C
    - Furanose: 5-membered ring with 1 O and 4 C
    - Requires at least 2 hydroxyl groups on ring carbons
      (2 not 3, to handle deoxy sugars)

    Args:
        mol: RDKit Mol object.

    Returns:
        True if molecule has a sugar ring pattern.
    """
    if mol is None:
        return False

    pyranose, furanose, ring_oh = _get_sugar_smarts()

    has_sugar_ring = False
    if pyranose is not None and mol.HasSubstructMatch(pyranose):
        has_sugar_ring = True
    elif furanose is not None and mol.HasSubstructMatch(furanose):
        has_sugar_ring = True

    if not has_sugar_ring:
        return False

    # Count OH groups on ring carbons
    if ring_oh is not None:
        oh_matches = mol.GetSubstructMatches(ring_oh)
        if len(oh_matches) >= 2:
            return True

    return False


def classify_compound_class(mol, canonical_smiles: str) -> Optional[str]:
    """Classify molecule into compound class for pre-routing.

    Returns class label or None for general routing.
    Classification order per D-01:
        steroid -> alkaloid -> terpene -> peptide -> amino_acid -> carbohydrate -> general

    Args:
        mol: RDKit Mol object.
        canonical_smiles: Canonical SMILES string.

    Returns:
        One of 'steroid', 'alkaloid', 'terpene', 'carbohydrate', or None.
    """
    if mol is None:
        return None

    # Sugar detection: check sugar lookup table first
    from .data.sugar_names import lookup_sugar
    if lookup_sugar(canonical_smiles) is not None:
        return "carbohydrate"

    # Pyranose/furanose SMARTS for sugars not in lookup
    if _has_sugar_ring_pattern(mol):
        return "carbohydrate"

    # NP detection handles steroid/alkaloid/terpene via scaffold matching
    from .perception.natural_products import detect_natural_product
    np_info = detect_natural_product(mol)
    if np_info is not None:
        return np_info.get("scaffold_class")  # "steroid", "alkaloid", "terpene"

    # Check exact derivative lookup -- some derivatives (e.g. alpha-pinene)
    # don't match a scaffold via substructure but ARE in the derivatives dict.
    # Infer class from the derivative name patterns.
    from .data.natural_products import get_natural_product_name
    deriv_name = get_natural_product_name(canonical_smiles)
    if deriv_name is not None:
        return _infer_class_from_derivative_name(deriv_name)

    return None  # General routing


# Terpene-related name patterns for class inference from derivative names
_TERPENE_KEYWORDS = frozenset([
    "pinene", "pinane", "bornane", "camphor", "limonene",
    "terpineol", "terpinene", "carotene", "lycopene", "menthane",
    "thujane", "pinanol", "borneol", "fenchone", "prostane",
])

# Steroid-related name patterns
_STEROID_KEYWORDS = frozenset([
    "cholesterol", "testosterone", "progesterone", "estradiol",
    "androstane", "pregnane", "cholestane", "estrane", "gonane",
    "campestanol", "ergostane", "stigmastane",
    "androstenedione", "androstanediol", "androstenediol",
    "androstenol", "androstenone", "estratetraenol",
    "androstadienone", "cardenolide", "cardanolide",
    "bufanolide", "bufadienolide",
])

# Alkaloid-related name patterns
_ALKALOID_KEYWORDS = frozenset([
    "morphine", "codeine", "diamorphine", "hydromorphone",
    "hydrocodone", "oxycodone", "lysergic", "lysergamide",
    "lysergol", "tropane", "morphinan", "aporphine",
    "dihydromorphine", "dihydrocodeine", "codeinone",
    "morphinone",
])

# Flavonoid / other
_FLAVONOID_KEYWORDS = frozenset([
    "flavone", "flavanone", "isoflavone", "chromanone", "chromone",
])


def _infer_class_from_derivative_name(name: str) -> Optional[str]:
    """Infer compound class from a derivative's trivial name.

    Args:
        name: Trivial/retained name of the derivative.

    Returns:
        Class label or None if class cannot be inferred.
    """
    name_lower = name.lower().replace("-", "")
    # Check terpene keywords
    for kw in _TERPENE_KEYWORDS:
        if kw in name_lower:
            return "terpene"
    # Check steroid keywords
    for kw in _STEROID_KEYWORDS:
        if kw in name_lower:
            return "steroid"
    # Check alkaloid keywords
    for kw in _ALKALOID_KEYWORDS:
        if kw in name_lower:
            return "alkaloid"
    # Beta-lactam and flavonoid are not routed to specific class handlers
    return None


@dataclass
class MolecularFeatures:
    """Container for perceived molecular features."""

    mol: Any  # RDKit Mol object
    smiles: str = ""
    canonical_smiles: str = ""

    # Functional group information
    functional_groups: Dict[str, List[tuple]] = field(default_factory=dict)
    principal_group: Optional[str] = None
    principal_group_atoms: List[tuple] = field(default_factory=list)

    # Chain information
    principal_chain: List[int] = field(default_factory=list)
    atom_to_locant: Dict[int, int] = field(default_factory=dict)
    substituents: Dict[int, List[List[int]]] = field(default_factory=dict)

    # Ring information
    ring_systems: List[set] = field(default_factory=list)
    all_ring_atoms: frozenset = field(default_factory=frozenset)  # All atoms in all ring systems (fused/bridged/spiro merged)
    is_cyclic: bool = False
    is_aromatic: bool = False
    ring_type: Optional[str] = None  # 'cycloalkane', 'cycloalkene', 'aromatic', 'heterocyclic_aromatic', 'heterocyclic_saturated'
    principal_ring: Optional[tuple] = None  # Atom indices of the principal ring
    senior_ring_system: Optional[tuple] = None  # P-44.2 most senior ring system (atom indices)
    oriented_ring: Optional[List[int]] = None  # Ring atoms reordered for naming
    ring_substituents: Dict[int, List[List[int]]] = field(default_factory=dict)  # Substituents on ring
    ring_double_bonds: List[tuple] = field(default_factory=list)  # Double bonds in ring
    ring_double_bond_locants: List[int] = field(default_factory=list)  # Locants for ring double bonds

    # Ring assembly information (biphenyl, bipyridine, etc.)
    ring_assembly_info: Optional[Dict] = None

    # Benzene-specific information
    is_benzene: bool = False  # True if principal ring is benzene
    benzene_ring: Optional[tuple] = None  # Atom indices of the benzene ring
    benzene_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_benzene_substituents

    # Polycyclic aromatic information
    polycyclic_name: Optional[str] = None  # Name of PAH parent (naphthalene, etc.)
    polycyclic_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_polycyclic_substituents

    # Heterocycle-specific information
    heterocycle_info: Optional[Dict] = None  # From classify_heterocycle
    oriented_heterocycle: Optional[List[int]] = None  # Ring atoms in IUPAC numbering order
    heterocycle_atom_to_locant: Optional[Dict[int, int]] = None  # Atom idx -> locant mapping
    heterocycle_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_heterocycle_substituents

    # Stereochemistry
    stereocenters: List[dict] = field(default_factory=list)
    double_bond_stereo: List[dict] = field(default_factory=list)

    # Multiple bonds
    double_bonds: List[tuple] = field(default_factory=list)
    triple_bonds: List[tuple] = field(default_factory=list)

    # Polyfunctional compound information
    is_polyfunctional: bool = False  # True if molecule has 2+ distinct functional groups
    non_principal_groups: Dict[str, List[tuple]] = field(default_factory=dict)  # FGs other than principal

    # Ester-specific information
    ester_match: Optional[tuple] = None  # SMARTS match for principal ester group

    # Amide-specific information
    amide_type: Optional[str] = None  # "primary", "secondary", or "tertiary"
    n_substituents: List[Dict] = field(default_factory=list)  # N-substituents from get_n_substituents()

    # Ring-as-substituent information (when chain is parent per IUPAC P-44.1)
    ring_substituents_as_groups: List[tuple] = field(default_factory=list)  # Rings that become substituents
    chain_is_parent: bool = False  # True when parent selection chose chain over ring

    # Ion/radical species information
    species_type: str = 'neutral'  # 'neutral', 'ion', 'zwitterion', 'salt', 'radical'
    ion_sites: Dict[str, List[Dict]] = field(default_factory=dict)  # From get_ion_sites()
    radical_sites: List[Dict] = field(default_factory=list)  # From get_radical_sites()
    total_charge: int = 0  # Net formal charge of the molecule

    # Phase 148 D-09 / V18 Appendix A.5: parent-selection result for downstream
    # coverage_scoring readers. Populated by _classify when select_parent()
    # runs for a cyclic+chain molecule (chain_len >= 2). Phase 149 / IM-11
    # will recalibrate FACTOR_WEIGHTS_V18['parent_correctness'] against the
    # discrimination this slot provides.
    # Source: V18_MILESTONE_PLAN Appendix A.5.
    parent_selection_result: Optional[Any] = None


def compute_features(mol, smiles: Optional[str] = None) -> MolecularFeatures:
    """Phase 147 BL-1: thin module-level wrapper around Orthonym()._perceive.

    Provides a public-API perception entry for tests and downstream
    callers. Mirrors what name_compound() does internally before naming.

    Args:
        mol: RDKit Mol object.
        smiles: Optional input SMILES; if None, derived via Chem.MolToSmiles(mol).

    Returns:
        MolecularFeatures populated by Orthonym()._perceive.

    Source: Phase 147 Plan 02 BL-1 fix (public-API perception entry).
    """
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    canonical_smiles = Chem.CanonSmiles(smiles)
    return Orthonym()._perceive(mol, smiles, canonical_smiles)


def _collect_ring_substituent_positions(features, ring_atoms):
    """Set of ring atom indices that bear an off-ring (non-H) substituent.

    Used by Phase 147 dispatch helper branch 4 (simple heterocycle) to feed
    ``orient_heterocycle_with_substituents``.
    """
    ring_set = set(ring_atoms)
    positions = set()
    for atom_idx in ring_atoms:
        atom = features.mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_set and nbr.GetSymbol() != 'H':
                positions.add(atom_idx)
                break
    return positions


def _build_ring_info_for_parent_selection(features):
    """Phase 147 D-03: dispatch on ring type and produce authoritative IUPAC locants.

    7-branch cascade (order per 147-CONTEXT.md D-03 with W-1 fix):
      1. Fused-heterocycle (match_fused_heterocycle_core) — preserve D-09
         byte-identical path; runs first.
      2. Polycyclic aromatic (identify_polycyclic + get_polycyclic_iupac_locants)
         — runs for ANY ring system with a PAH match, NOT gated on fused_type
         (W-1 fix: pyrene is ortho-peri-fused but must still flow here).
      3. Benzene-only single ring (orient_benzene with canonical
         get_benzene_substituents helper — BL-2 fix).
      4. Simple heterocycle single ring (orient_heterocycle_with_substituents).
      5. Spiro detection -> {"iupac_locants": None} stub (Phase 151 fills).
      6. Bridged/VB detection -> {"iupac_locants": None} stub (Phase 151 fills).
      7. Else -> None (carbocyclic monocycle / hydrocarbon polycycle:
         sorted fallback in _build_ring_pos preserves back-compat per D-09).

    Returns None for acyclic molecules.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-03, D-04, D-06, D-09; Plan 02 W-1, W-2, BL-2.
    """
    if not features.is_cyclic:
        return None

    # Match namer.py top-level relative-import style (W-2 fix).
    from .rules.fused_rings import classify_fused_system
    from .data.fused_heterocycles import match_fused_heterocycle_core
    from .rules.polycyclics import (
        identify_polycyclic,
        get_polycyclic_iupac_locants,
    )
    from .rules.benzene import orient_benzene, get_benzene_substituents
    from .rules.heterocycles import orient_heterocycle_with_substituents
    from .perception.rings import get_spiro_atoms
    from .rules.bridged_fused import is_bridged_fused

    mol = features.mol

    # Branch 1: fused-heterocycle (preserve D-09 byte-identical path for
    # ACTUAL heterocycles — indole/quinoline/etc.). Skip when the matched
    # core has no heteroatoms (e.g., pyrene also lives in
    # FUSED_HETEROCYCLES registry incidentally; PAHs must flow to branch 2
    # so their tuple-locant numbering is used per W-1 fix).
    fused_type = classify_fused_system(mol)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        het_match = match_fused_heterocycle_core(mol)
        if het_match is not None:
            _, atom_mapping, _ = het_match
            core_atom_indices = list(atom_mapping.keys())
            has_heteroatom_in_core = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in core_atom_indices
            )
            if has_heteroatom_in_core:
                return {"iupac_locants": atom_mapping}

    # Branch 2: PAH (W-1 fix — runs for ANY ring system with a PAH match,
    # not gated on the fused-only block above). Pyrene/anthracene/phenanthrene
    # are ortho-peri-fused; naphthalene is ortho-fused. All four flow through
    # here when not a fused-heterocycle.
    pah_name = identify_polycyclic(mol)
    if pah_name is not None:
        pah_locants = get_polycyclic_iupac_locants(mol, pah_name)
        if pah_locants is not None:
            return {"iupac_locants": pah_locants}

    ring_systems = features.ring_systems
    if len(ring_systems) == 1:
        ring_atoms = tuple(ring_systems[0])
        ring_mol_atoms = [mol.GetAtomWithIdx(i) for i in ring_atoms]

        # Branch 3: benzene-only (single 6-aromatic-C ring).
        # BL-2 fix: use the canonical get_benzene_substituents helper
        # (returns Dict[int, List[Dict]] with real substituent entries),
        # NOT a {idx: []} placeholder — the latter would yield arbitrary
        # orientation because orient_benzene checks ``if atom_idx in
        # substituents:`` and every dict key matches an empty-list value.
        if (len(ring_atoms) == 6
                and all(a.GetIsAromatic() and a.GetSymbol() == 'C'
                        for a in ring_mol_atoms)):
            try:
                substituents = get_benzene_substituents(mol, ring_atoms)
                oriented = orient_benzene(mol, ring_atoms, substituents)
                atom_to_locant = {
                    atom_idx: i + 1
                    for i, atom_idx in enumerate(oriented)
                }
                return {"iupac_locants": atom_to_locant}
            except Exception:
                pass  # Defensive: fall through on handler edge case.

        # Branch 4: simple heterocycle (single ring with >=1 heteroatom).
        has_heteroatom = any(
            a.GetSymbol() not in ('C', 'H') for a in ring_mol_atoms
        )
        if has_heteroatom:
            substituent_positions = _collect_ring_substituent_positions(
                features, ring_atoms,
            )
            try:
                _, atom_to_locant = orient_heterocycle_with_substituents(
                    mol, ring_atoms, substituent_positions,
                )
                return {"iupac_locants": atom_to_locant}
            except Exception:
                pass

    # Branch 5: spiro stub (Phase 151-02 fills).
    if bool(get_spiro_atoms(mol)):
        return {"iupac_locants": None}

    # Branch 6 (Phase 151-01 D-21): Von Baeyer ≥4-ring authoritative locants.
    # Routes tetracyclic / pentacyclic / higher non-cataloged bridged systems
    # to the new polycyclic_von_baeyer module. Anti-canary lock D-04 inside
    # is_higher_polycyclo guarantees bicyclo / tricyclo / aromatic / steroid
    # / mixed cases skip this branch and fall through to is_bridged_fused
    # below or downstream branches.
    #
    # Source: 151-CONTEXT.md D-04 / D-06 / D-21; 151-AUDIT-A.md verdict
    # THIN_WRAPPER; Phase 147 cascade-step-6 gate (candidate_pool.py:634).
    from .rules.polycyclic_von_baeyer import (
        get_higher_polycyclo_iupac_locants,
        is_higher_polycyclo,
    )
    if is_higher_polycyclo(mol):
        vbl = get_higher_polycyclo_iupac_locants(mol)
        if vbl is not None:
            return {"iupac_locants": vbl}

    # Branch 6 (existing): residual bridged / Von Baeyer stub for cases the
    # new module did NOT handle (e.g., bicyclic / tricyclic / aromatic
    # bridged systems caught by is_bridged_fused). Phase 151-02 / 03 wire
    # spiro and ring-assembly suppliers here.
    if is_bridged_fused(mol):
        return {"iupac_locants": None}

    # Branch 6.5 (Phase 149 D-09): non-cataloged fused systems.
    # Cataloged compounds reach Branches 1 (fused-heterocycle catalog) and
    # 2 (PAH) first. If they didn't, but the system is ortho-fused or
    # ortho-peri-fused with EXACTLY 2 SSSR components, route base-component
    # selection through FR-2.3.
    #
    # CRITICAL: emits `base_component_atoms` (NEW key), NOT `iupac_locants`.
    # Cascade step 6 in candidate_pool._has_iupac_locants checks
    # specifically for `iupac_locants` — Branch 6.5's new key is invisible
    # to that gate, so cascade step 6 stays GATED for non-cataloged fused
    # systems (Phase 147 D-06 + Phase 149 SC-7 lock).
    #
    # SCOPE LIMIT (Phase 149 Plan 02 triage): restricted to 2-component
    # fused systems where FR-2.3 base selection is reliable. 3+ component
    # systems (steroids, complex polycycles) fall through to Branch 7 to
    # avoid propagating partial base-atoms that disrupt downstream parent
    # selection for systems whose IUPAC name requires the full ring system
    # as parent. 3+ component systematic-name assembly is deferred to
    # Phase 149.x or Phase 155 per D-08 trade-off.
    #
    # Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    # Source: 149-CONTEXT.md D-09; SC-2; SC-7; D-08 trade-off.
    # Source: 147-CONTEXT.md D-06 (cascade step 6 gate carry-forward).
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        # Branch 1 (catalog) and Branch 2 (PAH) already missed (control
        # flow reached this point — Branch 1 returns earlier on
        # heterocycle match; Branch 2 returns earlier on pah_locants).
        from .rules.fused_ring_selection import (
            select_base_component,
            _enumerate_components,
        )
        components = _enumerate_components(mol)
        # Scope guard: 2-component fused systems only (Plan 02 triage).
        if len(components) == 2:
            try:
                base_atoms, _ = select_base_component(mol, components)
                return {"base_component_atoms": frozenset(base_atoms)}
            except (ValueError, Exception):
                pass  # Fall through to Branch 7

    # Branch 7: else -> sorted fallback in _build_ring_pos.
    return None


# Confidence threshold below which the quality gate rejects a name as
# truncated/incomplete and falls back to decomposition naming.
# Calibrated against 500-compound ChEBI benchmark: 0.30 catches
# catastrophically incomplete names without false positives on correct names.
_TRUNCATION_CONFIDENCE_THRESHOLD = 0.30


class Orthonym:
    """
    IUPAC nomenclature generator.
    
    Implements the structure-to-name workflow:
    1. Perception: Extract molecular features using RDKit
    2. Classification: Apply IUPAC seniority rules
    3. Assembly: Build name from fragments
    
    Example:
        >>> namer = Orthonym()
        >>> namer.name("CCO")
        'ethanol'
        >>> namer.name("CC(=O)O")
        'acetic acid'
    """
    
    def __init__(self, style: str = "pin"):
        """
        Initialize namer.
        
        Args:
            style: Naming style
                - "pin": Preferred IUPAC Names (IUPAC 2013)
                - "general": General IUPAC (more flexible)
                - "cas": CAS-style naming
        """
        self.style = style
    
    def name(self, smiles: str) -> str:
        """
        Generate IUPAC name from SMILES.

        Args:
            smiles: SMILES string representing the molecule

        Returns:
            IUPAC systematic name

        Raises:
            ValueError: If SMILES is invalid
        """
        # Start runtime fragment cache session (only at top-level depth)
        from .assembly.fragment_naming import start_naming_session, end_naming_session, is_top_level_naming
        start_naming_session()
        try:
            result = self._name_impl(smiles)
            # Universal stereo backstop (Phase 140, STER-16)
            # Only apply at top level -- decomposition fragments handle stereo
            # through their own naming paths.
            if is_top_level_naming():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    from .assembly.coverage_scoring import retrieve_confidence
                    handler = retrieve_confidence().get('handler', 'unknown')
                    result = _final_stereo_check(mol, result, handler=handler)
            return result
        finally:
            end_naming_session()

    def name_with_confidence(self, smiles: str) -> dict:
        """Generate IUPAC name with confidence metadata.

        Returns:
            dict with keys:
              - 'name' (str): The IUPAC name
              - 'confidence' (float): 0.0-1.0 aggregate confidence score
              - 'factors' (dict): Individual factor scores
                  {'ratio': float, 'atom_coverage': float,
                   'fg_recognition': float, 'substituent_completeness': float}
              - 'handler' (str): Which handler produced the name

        Raises:
            ValueError: If SMILES is invalid
        """
        from .assembly.fragment_naming import start_naming_session, end_naming_session, is_top_level_naming
        from .assembly.coverage_scoring import retrieve_confidence, clear_confidence
        start_naming_session()
        clear_confidence()
        try:
            name = self._name_impl(smiles)
            # Universal stereo backstop (Phase 140, STER-16)
            if is_top_level_naming():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    handler = retrieve_confidence().get('handler', 'unknown')
                    name = _final_stereo_check(mol, name, handler=handler)
            metadata = retrieve_confidence()
            # If no candidate was scored (early return path), build minimal metadata
            if not metadata['name']:
                metadata = {
                    'name': name,
                    'confidence': 1.0,  # Early return paths are high confidence
                    'factors': {'ratio': 1.0, 'atom_coverage': 1.0,
                                'fg_recognition': 1.0,
                                'substituent_completeness': 1.0},
                    'handler': 'direct',
                }
            else:
                # Ensure name matches (the stored candidate should match
                # what was returned)
                metadata['name'] = name
            return metadata
        finally:
            end_naming_session()
            clear_confidence()

    def _name_impl(self, smiles: str, _skip_decomposition: bool = False) -> str:
        """Internal naming implementation (wrapped by session management)."""
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get canonical SMILES for consistent processing
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # EARLY SPECIES DETECTION - before retained names check
        # This routes ionic/radical species to specialized naming paths
        species_type = detect_species_type(mol)

        # Route to specialized naming for ionic/radical species
        if species_type == 'salt':
            from .rules.salts import name_salt
            return name_salt(mol, style=self.style)
        elif species_type == 'radical':
            from .rules.radicals import name_radical
            return name_radical(mol, style=self.style)
        elif species_type == 'zwitterion':
            from .rules.salts import name_zwitterion
            return name_zwitterion(mol, style=self.style)
        elif species_type == 'ion':
            # Single-component ion: check retained names first, then fall through
            from .rules.ions import name_anion, name_cation

            sites = get_ion_sites(mol)

            # Check retained ion names first (acetate, benzoate, etc.)
            if sites['anions'] and not sites['cations']:
                result = name_anion(mol, style=self.style, retained_only=True)
                if result:
                    return result
            elif sites['cations'] and not sites['anions']:
                result = name_cation(mol, style=self.style, retained_only=True)
                if result:
                    return result

            # No retained ion name found.
            # For SINGLE anions in small/medium molecules (HA <= 25), use the
            # dedicated ion naming pipeline (handles aromatic carboxylates like
            # benzoate/naphthoate correctly). For larger molecules, fall through
            # to the neutral naming pipeline which produces more complete names.
            if (len(sites['anions']) == 1 and not sites['cations']
                    and mol.GetNumHeavyAtoms() <= 25):
                # Try the existing ion pipeline first
                ion_result = name_anion(mol, style=self.style)
                if ion_result:
                    return ion_result
                # Ion pipeline returned empty -- try neutralize-then-name
                try:
                    from rdkit.Chem import RWMol
                    from .rules.ions import classify_anion, _acid_name_to_carboxylate
                    from .perception.ions import _get_internal_charge_atoms
                    anion_type = classify_anion(mol, sites['anions'][0])
                    if anion_type == 'carboxylate':
                        internal_charge_atoms = _get_internal_charge_atoms(mol)
                        rwmol = RWMol(mol)
                        for atom in rwmol.GetAtoms():
                            if (atom.GetSymbol() == 'O'
                                    and atom.GetFormalCharge() == -1
                                    and atom.GetIdx() not in internal_charge_atoms):
                                atom.SetFormalCharge(0)
                                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                        neutral_mol = rwmol.GetMol()
                        neutral_smi = Chem.MolToSmiles(neutral_mol, canonical=True)
                        neutral_namer = Orthonym(style=self.style)
                        neutral_name = neutral_namer.name(neutral_smi)
                        if neutral_name:
                            anion_name = _acid_name_to_carboxylate(
                                neutral_name, 1
                            )
                            if anion_name:
                                return anion_name
                except Exception:
                    pass  # Fall through to normal pipeline

            # For POLY-anionic species (2+ anionic sites, e.g., dicarboxylate),
            # try neutralize-then-name: convert [O-] -> OH so carboxylic acid
            # SMARTS can match, then name the neutral form. This prevents
            # empty-string results for polycarboxylate anions where the FG
            # detection only recognizes protonated acids.
            if len(sites['anions']) >= 2 and not sites['cations']:
                try:
                    from rdkit.Chem import RWMol
                    from .rules.ions import classify_anion, _acid_name_to_carboxylate
                    from .perception.ions import _get_internal_charge_atoms
                    internal_charge_atoms = _get_internal_charge_atoms(mol)
                    rwmol = RWMol(mol)
                    neutralized = False
                    for atom in rwmol.GetAtoms():
                        if (atom.GetSymbol() == 'O'
                                and atom.GetFormalCharge() == -1
                                and atom.GetIdx() not in internal_charge_atoms):
                            atom.SetFormalCharge(0)
                            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                            neutralized = True
                    if neutralized:
                        neutral_mol = rwmol.GetMol()
                        neutral_smi = Chem.MolToSmiles(neutral_mol, canonical=True)
                        neutral_namer = Orthonym(style=self.style)
                        neutral_name = neutral_namer.name(neutral_smi)
                        if neutral_name:
                            # IUPAC P-72.2.1: Convert acid suffix to carboxylate
                            # for deprotonated carboxylate sites
                            carboxylate_count = sum(
                                1 for a in sites['anions']
                                if classify_anion(mol, a) == 'carboxylate'
                            )
                            if carboxylate_count > 0:
                                anion_name = _acid_name_to_carboxylate(
                                    neutral_name, carboxylate_count
                                )
                                if anion_name:
                                    return anion_name
                            return neutral_name
                except Exception:
                    pass  # Fall through to normal pipeline

        # Continue with normal neutral molecule naming

        # NEUTRAL DOT-DISCONNECTED SMILES: cocrystals, solvates, neutral mixtures
        # These have '.' in SMILES but no formal charges (species_type == 'neutral').
        # Salts/ions/zwitterions are already routed above this point.
        # Split into components, name each independently, join with space.
        # Only activates when there are 2+ multi-atom (nameable) fragments.
        # Single-atom fragments (Cl, Br, I, O) are not true mixture components
        # and their presence means the original pipeline should handle the molecule.
        if '.' in canonical_smiles and species_type == 'neutral':
            frags = canonical_smiles.split('.')
            if len(frags) >= 2:
                # Parse all fragments and filter out single-atom fragments
                frag_mols = []
                for frag_smi in frags:
                    frag_mol = Chem.MolFromSmiles(frag_smi)
                    if frag_mol:
                        ha = frag_mol.GetNumHeavyAtoms()
                        frag_mols.append((frag_smi, ha))

                # Only split when 2+ multi-atom fragments exist.
                # Single-atom dots (e.g., .Cl.Cl) are non-molecular entities
                # that the normal pipeline handles better as part of the whole mol.
                multi_atom_frags = [(s, ha) for s, ha in frag_mols if ha >= 2]
                if len(multi_atom_frags) >= 2:
                    # Sort by descending heavy atom count for consistent output
                    frag_mols.sort(key=lambda x: -x[1])
                    component_names = []
                    for frag_smi, _ in frag_mols:
                        try:
                            frag_namer = Orthonym(style=self.style)
                            frag_name = frag_namer.name(frag_smi)
                            if frag_name and frag_name != "unknown":
                                component_names.append(frag_name)
                        except Exception:
                            pass  # Skip unnamed fragments

                    if component_names:
                        return ' '.join(component_names)
                # If fewer than 2 multi-atom fragments, or no names produced,
                # fall through to normal pipeline

        # CHARGE NEUTRALIZATION for large zwitterions reclassified as 'neutral'
        # by the HA>20 + quaternary-N guard in detect_species_type().
        # These molecules still carry formal charges (P-O-, N+) that prevent
        # FG detection SMARTS from matching (e.g., [OX2H] for COOH).
        # Neutralize O- to OH; leave quaternary N+ (no H) charged to
        # preserve valid valence. ONLY applies to molecules that have true
        # zwitterion character -- NOT to molecules with internal charges
        # (nitro [N+](=O)[O-], azide, N-oxide) which are normal functional groups.
        if Chem.GetFormalCharge(mol) == 0:
            from .perception.ions import _has_true_zwitterion_character
            if _has_true_zwitterion_character(mol):
                try:
                    from rdkit.Chem import RWMol
                    rwmol = RWMol(mol)
                    for atom in rwmol.GetAtoms():
                        charge = atom.GetFormalCharge()
                        if charge < 0:
                            # O- -> OH (add H for each negative charge)
                            atom.SetFormalCharge(0)
                            cur_h = atom.GetNumExplicitHs()
                            atom.SetNumExplicitHs(cur_h + abs(charge))
                        elif charge > 0:
                            total_h = atom.GetTotalNumHs()
                            if total_h >= charge:
                                # Protonated amine: remove H to neutralize
                                cur_h = atom.GetNumExplicitHs()
                                atom.SetFormalCharge(0)
                                atom.SetNumExplicitHs(max(0, cur_h - charge))
                            # else: quaternary N+ (no H) -- leave charged
                            # to preserve valid valence
                    Chem.SanitizeMol(rwmol)
                    mol = rwmol.GetMol()
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    pass  # If neutralization fails, continue with original mol

        # MULTIPLICATIVE NOMENCLATURE (IUPAC P-51.3)
        # Detect symmetric molecules with bridge atoms linking identical parent
        # structures (e.g., 4,4'-methylenedianiline). Must come before NP detection
        # and retained names to catch these before half the molecule is dropped.
        from .rules.multiplicative import name_multiplicative
        mult_name = name_multiplicative(mol)
        if mult_name is not None:
            return mult_name

        # COMPOUND CLASS PRE-ROUTING (Phase 141)
        # Classify into compound class for routing. Sugar detection handles
        # carbohydrates that were previously missing from the main cascade.
        compound_class = classify_compound_class(mol, canonical_smiles)

        # Direct sugar routing: carbohydrates found via sugar lookup
        if compound_class == "carbohydrate":
            from .data.sugar_names import lookup_sugar
            sugar_info = lookup_sugar(canonical_smiles)
            if sugar_info is not None:
                anomer, config, base_name = sugar_info
                parts = []
                if anomer:
                    parts.append(anomer)
                if config:
                    parts.append(config)
                parts.append(base_name)
                return "-".join(parts)
            # If SMARTS matched but no lookup hit, fall through to systematic

        # NATURAL PRODUCT DETECTION
        # Check before retained names because NP detection uses substructure matching
        # while retained names use exact SMILES matching.
        # Even with style="systematic", NP names are returned (IUPAC 2013 has no
        # systematic PIN for natural products - see P-10).
        from .rules.natural_products import name_natural_product
        np_name = name_natural_product(mol)
        if np_name is not None:
            return np_name

        # PEPTIDE DETECTION
        # Check before retained names and amino acids to prevent peptides from
        # being flattened to single amino acid names. Must come after NP detection.
        from .rules.amino_acids import is_peptide as _is_peptide
        if _is_peptide(mol):
            from .rules.peptides import name_peptide
            peptide_name = name_peptide(mol)
            if peptide_name:
                return peptide_name

        # Check retained names first (benzene, methanol, etc.) unless systematic style requested
        if self.style != "systematic":
            if canonical_smiles in RETAINED_NAMES:
                return RETAINED_NAMES[canonical_smiles]

            # Check for amino acids (standard amino acids use trivial names)
            from .rules.amino_acids import name_amino_acid
            aa_name = name_amino_acid(mol, canonical_smiles)
            if aa_name:
                return aa_name

        # Skeletal replacement nomenclature (IUPAC P-15.4)
        # Detect chains where heteroatoms are embedded in the backbone (C-O-C, C-N-C, C-S-C)
        # Must come BEFORE perception to override substitutive naming (ether/amine prefix)
        from .rules.skeletal_replacement import try_skeletal_replacement_name
        skel_name = try_skeletal_replacement_name(mol)
        if skel_name:
            return skel_name

        # DECOMPOSITION ENGINE (v4.0 Phase 39)
        # For complex molecules with ester/amide/glycosidic bonds, try cleaving
        # at functional bonds and naming fragments individually.
        if not _skip_decomposition:
            from .decomposition import try_decompose
            decomposed_name = try_decompose(mol, style=self.style)
            if decomposed_name is not None:
                return decomposed_name

        # Perceive molecular features
        features = self._perceive(mol, smiles, canonical_smiles)

        # Classify: determine naming strategy and principal group
        self._classify(features)

        # Assemble: build final name from fragments
        assembled = assemble_name(features, style=self.style)

        # Final quality gate: if composer produced a garbled name
        # (e.g., "cycloanedicarboxamide" for a multi-amide molecule),
        # fall back to decomposition directly. Does NOT recurse into
        # name_fragment_recursively to avoid circular feedback loops.
        if not _skip_decomposition and assembled and mol.GetNumHeavyAtoms() > 15:
            _GARBLED_TOKENS = ('cycloane', 'anedicarboxamide', 'aneyl')
            assembled_lower = assembled.lower()
            is_garbled = any(tok in assembled_lower for tok in _GARBLED_TOKENS)

            # Also detect stub-only names: when the parent text is empty,
            # the assembly may produce just a bare suffix like "ane" or "ol".
            # A name shorter than 6 chars for a 15+ atom molecule is garbled.
            if not is_garbled and len(assembled) < 6:
                is_garbled = True

            if is_garbled:
                from .decomposition import try_decompose
                decomp_name = try_decompose(mol, style=self.style)
                if decomp_name and decomp_name != assembled:
                    return decomp_name
                # Decomposition also failed: return the SMILES-based
                # canonical SMILES as an honest fallback rather than a
                # garbled pseudo-IUPAC name that could mislead.
                return canonical_smiles

            # Confidence-based rejection: if the coverage scoring system
            # indicates the assembled name is catastrophically incomplete
            # (confidence < threshold), attempt decomposition fallback.
            # This is NOT a postprocessor -- the confidence score reflects
            # genuine structural coverage analysis computed during naming.
            from .assembly.coverage_scoring import retrieve_confidence
            conf_data = retrieve_confidence()
            conf_score = conf_data.get('confidence', None)
            conf_handler = conf_data.get('handler', 'unknown')
            # Only gate on confidence when it was actually computed during
            # assemble_name() -- handler='unknown' means no scoring happened
            # (e.g., decomposition path, retained names, etc.).
            if (conf_score is not None
                    and conf_handler != 'unknown'
                    and conf_score < _TRUNCATION_CONFIDENCE_THRESHOLD):
                from .decomposition import try_decompose
                decomp_name = try_decompose(mol, style=self.style)
                if decomp_name and decomp_name != assembled:
                    logger.info(
                        "Quality gate: rejecting low-confidence name "
                        "(%.4f < %.2f), using decomposition fallback",
                        conf_score, _TRUNCATION_CONFIDENCE_THRESHOLD,
                    )
                    return decomp_name

            # D-01: atom_coverage secondary gate -- catches quality-gate false
            # positives where retained ring names inflate character-based
            # confidence but atom coverage reveals only partial molecule
            # description.
            if (conf_score is not None
                    and conf_handler != 'unknown'
                    and conf_handler != 'retained_name'):
                atom_cov = conf_data.get('factors', {}).get('atom_coverage', 1.0)
                if atom_cov < 0.55:
                    # Verify molecule has cleavable bonds before triggering
                    from .decomposition.bond_cleavage import find_cleavable_bonds
                    if find_cleavable_bonds(mol):
                        from .decomposition import try_decompose
                        decomp_name = try_decompose(mol, style=self.style)
                        if decomp_name and decomp_name != assembled:
                            logger.info(
                                "Atom coverage gate: rejecting low-coverage "
                                "name (atom_cov=%.4f < 0.55), using "
                                "decomposition fallback",
                                atom_cov,
                            )
                            return decomp_name

        return assembled
    
    def _perceive(self, mol, smiles: str, canonical_smiles: str) -> MolecularFeatures:
        """
        Extract molecular features using RDKit.

        This is the perception layer - converts structure to features.
        """
        features = MolecularFeatures(
            mol=mol,
            smiles=smiles,
            canonical_smiles=canonical_smiles
        )

        # Add species type detection for ionic/radical compounds
        features.species_type = detect_species_type(mol)
        features.total_charge = Chem.GetFormalCharge(mol)

        if features.species_type in ('ion', 'zwitterion', 'salt'):
            features.ion_sites = get_ion_sites(mol)
        if features.species_type == 'radical':
            features.radical_sites = get_radical_sites(mol)

        # Detect functional groups using SMARTS patterns
        features.functional_groups = detect_functional_groups(mol)

        # Filter consumed atoms to prevent double-counting
        # (e.g., acid halide Cl should not also appear as "chloro" prefix)
        features.functional_groups = _filter_consumed_fg_atoms(features.functional_groups)

        # Detect ring systems
        features.ring_systems = get_ring_systems(mol)
        features.all_ring_atoms = get_complete_ring_atom_set(mol)
        features.is_cyclic = len(features.ring_systems) > 0

        # Check aromaticity
        for ring_system in features.ring_systems:
            if is_aromatic_ring(mol, ring_system):
                features.is_aromatic = True
                break

        # Detect multiple bonds
        features.double_bonds = self._find_double_bonds(mol)
        features.triple_bonds = self._find_triple_bonds(mol)

        # Assign CIP stereochemistry labels BEFORE extracting stereo info
        # rdCIPLabeler sets _CIPCode on atoms (R/S) and bonds (E/Z)
        rdCIPLabeler.AssignCIPLabels(mol)
        mol.SetProp(_CIP_ASSIGNED_PROP, '1')  # Mark as done for idempotent guard

        # Extract stereochemistry (now depends on _CIPCode being set)
        features.stereocenters = get_stereocenters(mol)
        features.double_bond_stereo = get_double_bond_stereo(mol)

        return features
    
    def _classify(self, features: MolecularFeatures) -> None:
        """
        Apply IUPAC classification rules.

        Determines:
        - Principal characteristic group (highest seniority)
        - Principal chain/ring (following IUPAC 2013 criteria)
        - Substituent positions and identities
        """
        # Determine principal functional group
        pg_name, pg_atoms = get_principal_group(
            features.mol,
            features.functional_groups
        )
        features.principal_group = pg_name
        features.principal_group_atoms = pg_atoms

        # Store ester match(es) if principal group is ester
        if pg_name == "ester" and pg_atoms:
            features.ester_match = pg_atoms[0]  # First ester match (backward compat)
            features.all_ester_matches = pg_atoms  # All ester matches

        # Detect polyfunctional compounds (multiple distinct FGs)
        from .rules.polyfunctional import detect_polyfunctional, get_non_principal_groups
        features.is_polyfunctional = detect_polyfunctional(
            features.mol, features.functional_groups
        )
        if features.is_polyfunctional:
            features.non_principal_groups = get_non_principal_groups(
                features.functional_groups, features.principal_group
            )

        # Parent selection for ALL cyclic molecules (Phase 148: no fused-heterocycle bypass).
        # P-44.1 cascade runs whenever a meaningful chain exists; cascade itself
        # enforces P-31.1.3.4 NP override (parent_selection.py:688-700) and
        # P-52.2.8 ring-on-tie tiebreaker (parent_selection.py:862-869).
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html  P-44.1
        # Source: https://iupac.qmul.ac.uk/BlueBook/P5.html  P-52.2.8
        # Source: HERITAGE 1990 §4 (full seniority cascade on ALL structures).
        if features.is_cyclic:
            from .rules.parent_selection import select_parent

            # Get ring atoms to exclude when finding chain
            all_ring_atoms = set()
            for ring in features.ring_systems:
                all_ring_atoms.update(ring)

            # Find potential principal chain (excluding ring atoms)
            # IMPORTANT: namer.py does the chain finding, then passes result to select_parent()
            potential_chain = find_principal_chain(
                features.mol,
                features.functional_groups,
                features.principal_group,
                exclude_atoms=all_ring_atoms
            )

            # Only do parent selection if we found a meaningful chain (>= 2 carbons)
            if potential_chain and len(potential_chain) >= 2:
                # Phase 147 D-03: delegate ring-type dispatch to helper.
                # Replaces the prior inline fused-hetero-only block; new
                # helper covers fused-hetero / PAH / benzene / simple-
                # hetero / spiro-stub / VB-stub / else->None per D-03.
                _ring_info = _build_ring_info_for_parent_selection(features)
                # Phase 147: stash on features so downstream pool.add()
                # call sites can read it without a signature change at
                # every composer.py call site (transient runtime
                # attribute; not a MolecularFeatures dataclass field
                # per D-08; safe because the dataclass is not frozen).
                features._ring_info = _ring_info

                # Pass pre-computed chain to select_parent
                selection = select_parent(
                    mol=features.mol,
                    ring_systems=features.ring_systems,
                    principal_chain=potential_chain,
                    principal_group=features.principal_group,
                    principal_group_atoms=features.principal_group_atoms,
                    ring_info=_ring_info
                )
                features.parent_selection_result = selection  # Phase 148 D-02 (V18 Appendix A.5)

                if selection.parent_type == 'chain':
                    features.chain_is_parent = True
                    # is_cyclic stays True -- ring data needed for ring-as-substituent naming (Phase 139 ARCH-01)
                    features.principal_chain = selection.parent_atoms
                    features.ring_substituents_as_groups = selection.substituent_rings

        # For cyclic molecules, identify principal ring and its type
        if features.is_cyclic:
            # Check for ring assemblies FIRST (identical disconnected ring systems)
            # Must come before fused/polycyclic classification because ring assemblies
            # have 2+ separate ring systems that would otherwise be misrouted
            if len(features.ring_systems) >= 2:
                from .rules.ring_assemblies import detect_ring_assembly
                assembly_info = detect_ring_assembly(features.mol, features.ring_systems)
                if assembly_info:
                    features.ring_assembly_info = assembly_info
                    if not features.chain_is_parent:
                        return  # Skip other ring classification for assemblies

            # Check for polycyclic aromatics FIRST (naphthalene, anthracene, etc.)
            # These take precedence over single-ring classification
            from .rules.polycyclics import identify_polycyclic, get_polycyclic_substituents
            pah_name = identify_polycyclic(features.mol)
            if pah_name:
                features.polycyclic_name = pah_name
                features.polycyclic_substituents = get_polycyclic_substituents(
                    features.mol, pah_name
                )
                # Set ring type for consistency
                features.ring_type = 'aromatic'
                if not features.chain_is_parent:
                    return  # Skip other ring classification for PAHs

            ring_info = get_ring_info(features.mol)
            atom_rings = ring_info['atom_rings']

            if atom_rings:
                # Select the most senior ring system per IUPAC P-44.2
                # and store it for downstream use (e.g., ring-vs-chain
                # comparison in the composer). The monocyclic dispatch
                # path below uses atom_rings[0] which preserves SSSR
                # cyclic traversal order needed by orientation functions.
                # Complex multi-ring (fused/bridged) systems are handled
                # by _classify_complex_ring() in the composer.
                from .rules.ring_selection import select_principal_ring_system
                principal = select_principal_ring_system(
                    features.mol, features.ring_systems
                )
                features.senior_ring_system = principal if principal else atom_rings[0]

                # For multi-ring-system molecules, use a SSSR ring from
                # the senior system as principal_ring when the senior
                # system is strictly LARGER than the default ring's system.
                # This ensures fused/bridged senior systems (imidazopyridine,
                # xanthene) take precedence over small monocyclic rings
                # (benzene) per IUPAC P-44.2.
                #
                # Guards (all must pass to switch):
                # (a) Senior system must be significantly larger (>= 3
                #     atoms) than the default — marginal differences
                #     (1-2 atoms) cause churn without improving names.
                # (b) Principal FG must NOT be attached to the default
                #     ring system — per P-44.1, the parent must contain
                #     the principal characteristic group.
                if principal and len(features.ring_systems) >= 2:
                    senior_set = set(principal)
                    # Find the ring system that contains the default ring
                    default_system = set(atom_rings[0])
                    default_system_size = len(atom_rings[0])
                    for rs in features.ring_systems:
                        if set(atom_rings[0]).issubset(rs):
                            default_system = rs
                            default_system_size = len(rs)
                            break

                    # Guard (a): senior system must be >= 3 atoms larger
                    size_diff = len(senior_set) - default_system_size
                    size_ok = size_diff >= 3

                    # Guard (b): PG must not be on the default system
                    # (P-44.1: parent must contain the principal group)
                    from .rules.parent_selection import is_principal_group_on_ring
                    pg_on_default = False
                    if features.principal_group_atoms:
                        pg_on_default = is_principal_group_on_ring(
                            features.mol, default_system,
                            features.principal_group_atoms,
                            features.principal_group,
                        )

                    if size_ok and not pg_on_default:
                        best_ring = atom_rings[0]
                        best_overlap = 0
                        for ring in atom_rings:
                            overlap = len(set(ring) & senior_set)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_ring = ring
                        features.principal_ring = best_ring
                    else:
                        features.principal_ring = atom_rings[0]
                else:
                    features.principal_ring = atom_rings[0]
                features.ring_type = classify_ring(features.mol, features.principal_ring)

                # Check if this is a benzene ring
                from .rules.benzene import is_benzene_ring, get_benzene_substituents
                if is_benzene_ring(features.mol, features.principal_ring):
                    features.is_benzene = True
                    features.benzene_ring = features.principal_ring
                    features.benzene_substituents = get_benzene_substituents(
                        features.mol, features.principal_ring
                    )
                elif features.ring_type and features.ring_type.startswith('heterocyclic'):
                    # Heterocyclic ring: classify, detect substituents, and orient
                    from .rules.heterocycles import (
                        classify_heterocycle,
                        orient_heterocycle_with_substituents,
                        get_heterocycle_substituents,
                    )

                    features.heterocycle_info = classify_heterocycle(
                        features.mol, features.principal_ring
                    )

                    # Detect substituent positions for proper orientation
                    ring_set = set(features.principal_ring)
                    sub_positions = set()
                    for idx in features.principal_ring:
                        atom = features.mol.GetAtomWithIdx(idx)
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetIdx() not in ring_set:
                                sub_positions.add(idx)
                                break

                    # Orient considering substituents for lowest locants
                    oriented, atom_to_locant = orient_heterocycle_with_substituents(
                        features.mol, features.principal_ring, sub_positions
                    )
                    features.oriented_heterocycle = oriented
                    features.heterocycle_atom_to_locant = atom_to_locant

                    # Get substituent details for naming
                    features.heterocycle_substituents = get_heterocycle_substituents(
                        features.mol,
                        features.principal_ring,
                        oriented,
                        atom_to_locant
                    )
                else:
                    # Non-benzene, non-heterocyclic ring: detect substituents and orient
                    from .rules.cycloalkanes import (
                        get_ring_substituents, get_ring_double_bonds,
                        orient_cycloalkane, orient_cycloalkene
                    )

                    # Get ring substituents
                    features.ring_substituents = get_ring_substituents(
                        features.mol, features.principal_ring
                    )

                    # Get ring double bonds
                    features.ring_double_bonds = get_ring_double_bonds(
                        features.mol, features.principal_ring
                    )

                    # Orient the ring based on type
                    if features.ring_type == 'cycloalkane':
                        features.oriented_ring = orient_cycloalkane(
                            features.mol,
                            features.principal_ring,
                            features.ring_substituents
                        )
                    elif features.ring_type == 'cycloalkene':
                        # Determine principal group atoms on the ring
                        # IUPAC P-31.1.3.4: principal group gets lowest locant
                        # We identify ring C atoms that directly bear the principal
                        # group's characteristic heteroatom (e.g., C=O for ketone,
                        # C-OH for alcohol). The characteristic heteroatom must be
                        # bonded DIRECTLY to a ring carbon (not via an exocyclic C).
                        # Exocyclic groups (aldehyde -CHO, -COOH) use -carbaldehyde/
                        # -carboxylic acid suffixes and don't override ring numbering.
                        pg_ring_atoms = set()
                        if features.principal_group and features.principal_group in features.functional_groups:
                            ring_set = set(features.principal_ring)
                            for match in features.functional_groups[features.principal_group]:
                                match_set = set(match)
                                for atom_idx in match:
                                    if atom_idx in ring_set:
                                        atom = features.mol.GetAtomWithIdx(atom_idx)
                                        if atom.GetSymbol() != 'C':
                                            continue
                                        # Check if this ring C is bonded to a non-ring
                                        # HETEROATOM that is in the FG match
                                        for nbr in atom.GetNeighbors():
                                            nbr_idx = nbr.GetIdx()
                                            if (nbr_idx not in ring_set
                                                    and nbr_idx in match_set
                                                    and nbr.GetSymbol() != 'C'):
                                                pg_ring_atoms.add(atom_idx)
                                                break

                        features.oriented_ring = orient_cycloalkene(
                            features.mol,
                            features.principal_ring,
                            features.ring_double_bonds,
                            features.ring_substituents,
                            principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None
                        )
                        # Calculate ring double bond locants
                        if features.oriented_ring and features.ring_double_bonds:
                            oriented = features.oriented_ring
                            locants = []
                            for a1, a2 in features.ring_double_bonds:
                                pos1 = oriented.index(a1)
                                pos2 = oriented.index(a2)
                                # Lower position is the locant
                                locants.append(min(pos1, pos2) + 1)
                            features.ring_double_bond_locants = sorted(locants)

        # Find principal chain (for acyclic molecules or chain-is-parent cyclic molecules)
        # Skip if chain was already set by parent selection (chain_is_parent = True)
        if not features.is_cyclic or features.chain_is_parent:
            if not features.chain_is_parent:
                # Normal acyclic molecule - find principal chain
                # Guard: exclude ring atoms even in "acyclic" path (defensive)
                _ring_exclude = set()
                ri = features.mol.GetRingInfo()
                for ring in ri.AtomRings():
                    _ring_exclude.update(ring)
                features.principal_chain = find_principal_chain(
                    features.mol,
                    features.functional_groups,
                    features.principal_group,
                    exclude_atoms=_ring_exclude if _ring_exclude else None
                )

            if features.principal_chain:
                # Collect principal group atom indices for orientation
                pg_atom_set: set = set()
                if features.principal_group and features.principal_group in features.functional_groups:
                    for match in features.functional_groups[features.principal_group]:
                        pg_atom_set.update(match)

                # Get initial substituents for orientation criterion (d)
                # This is needed BEFORE orientation to apply lowest-locant rule
                initial_subs = self._find_substituents_by_atom(
                    features.mol,
                    features.principal_chain
                )

                # Orient chain using IUPAC 2013 criteria
                features.principal_chain = orient_chain(
                    chain=features.principal_chain,
                    mol=features.mol,
                    principal_group_atoms=pg_atom_set,
                    double_bonds=features.double_bonds,
                    triple_bonds=features.triple_bonds,
                    substituent_positions=initial_subs,
                )

                # Build atom-to-locant mapping from oriented chain
                features.atom_to_locant = build_atom_to_locant(features.principal_chain)

                # Get substituents with final locant-based positions
                features.substituents = self._find_substituents(
                    features.mol,
                    features.principal_chain
                )

        # INST-05: Naming decision trace
        if logger.isEnabledFor(logging.DEBUG):
            parent_type = 'chain' if getattr(features, 'chain_is_parent', False) else 'ring'
            parent_atoms = (features.principal_chain if parent_type == 'chain'
                            else features.principal_ring or [])
            sub_count = (sum(len(v) for v in features.substituents.values())
                         if features.substituents else 0)
            logger.debug(
                "NAMING_DECISION: smiles=%s parent_type=%s parent_size=%d "
                "principal_group=%s fg_count=%d sub_count=%d is_cyclic=%s",
                features.canonical_smiles,
                parent_type,
                len(parent_atoms) if parent_atoms else 0,
                features.principal_group,
                len(features.functional_groups),
                sub_count,
                features.is_cyclic,
            )

    def _find_double_bonds(self, mol) -> List[tuple]:
        """Find all C=C double bonds."""
        double_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                begin = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                # Only C=C double bonds (not C=O, etc.)
                if begin.GetSymbol() == 'C' and end.GetSymbol() == 'C':
                    double_bonds.append((
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()
                    ))
        return double_bonds
    
    def _find_triple_bonds(self, mol) -> List[tuple]:
        """Find all C≡C triple bonds."""
        triple_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                begin = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                # Only C≡C triple bonds (not C≡N)
                if begin.GetSymbol() == 'C' and end.GetSymbol() == 'C':
                    triple_bonds.append((
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()
                    ))
        return triple_bonds
    
    def _find_substituents(self, mol, chain: List[int]) -> Dict[int, List[List[int]]]:
        """Find substituents attached to the principal chain (keyed by position)."""
        from .perception.chains import get_substituents
        return get_substituents(mol, chain)

    def _find_substituents_by_atom(self, mol, chain: List[int]) -> Dict[int, List[List[int]]]:
        """
        Find substituents attached to the principal chain, keyed by atom index.

        This is used for orient_chain() which expects substituent_positions
        keyed by atom indices on the chain, not by position numbers.

        Args:
            mol: RDKit Mol object
            chain: List of atom indices in the principal chain

        Returns:
            Dict mapping chain atom index to list of substituent atom lists
        """
        from .perception.chains import _bfs_substituent

        chain_set = set(chain)
        substituents = {}

        for chain_idx in chain:
            chain_atom = mol.GetAtomWithIdx(chain_idx)
            position_subs = []

            for neighbor in chain_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()

                # Skip atoms that are part of the main chain
                if nbr_idx in chain_set:
                    continue

                # BFS to find full substituent
                sub_atoms = _bfs_substituent(mol, nbr_idx, chain_set)
                position_subs.append(sub_atoms)

            if position_subs:
                substituents[chain_idx] = position_subs

        return substituents


def _filter_consumed_fg_atoms(functional_groups: dict) -> dict:
    """Filter out functional group matches whose atoms are consumed by higher-priority groups.

    This prevents double-counting. For example, the Cl in an acid chloride
    (-C(=O)Cl) matches both the acid_chloride SMARTS and the chloro SMARTS.
    Without filtering, the Cl would appear as both "oyl chloride" (suffix) and
    "chloro" (prefix), producing incorrect names like "1-chloroethanoyl chloride"
    instead of "acetyl chloride".

    Rules:
        1. Acid halides consume halogens: remove chloro/bromo/fluoro matches
           where the halogen atom is part of an acid halide group.
        2. Anhydrides consume esters: remove ester matches where atoms overlap
           with an anhydride group.

    Args:
        functional_groups: Dict from detect_functional_groups().

    Returns:
        Filtered copy of functional_groups with consumed matches removed.
    """
    from .rules.acid_halides import get_acid_halide_consumed_atoms
    from .rules.anhydrides import get_anhydride_consumed_atoms

    fg = dict(functional_groups)  # shallow copy

    # --- Rule 1: Acid halides consume halogens ---
    halide_consumed = get_acid_halide_consumed_atoms(fg)
    if halide_consumed:
        for halogen_key in ("chloro", "bromo", "fluoro"):
            if halogen_key in fg:
                filtered = []
                for match in fg[halogen_key]:
                    # match = (halogen_idx, C_idx) from SMARTS [X][#6]
                    halogen_atom = match[0]
                    if halogen_atom not in halide_consumed:
                        filtered.append(match)
                if filtered:
                    fg[halogen_key] = filtered
                else:
                    del fg[halogen_key]

    # --- Rule 2: Anhydrides consume esters ---
    anhydride_consumed = get_anhydride_consumed_atoms(fg)
    if anhydride_consumed:
        if "ester" in fg:
            filtered = []
            for match in fg["ester"]:
                match_set = set(match)
                # If ANY atom in the ester match overlaps with anhydride, remove it
                if not match_set & anhydride_consumed:
                    filtered.append(match)
            if filtered:
                fg["ester"] = filtered
            else:
                del fg["ester"]

    return fg



def name_compound(smiles: str, style: str = "pin",
                   include_confidence: bool = False):
    """
    Convenience function to generate IUPAC name from SMILES.

    Args:
        smiles: SMILES string
        style: Naming style
            - "pin": Preferred IUPAC Names (default, uses retained names when available)
            - "systematic": Always generate systematic name (bypass retained names)
            - "general": General IUPAC (more flexible)
            - "cas": CAS-style naming
        include_confidence: If True, return dict with confidence metadata
            instead of plain str

    Returns:
        str: IUPAC systematic name (default)
        dict: {'name': str, 'confidence': float, 'factors': dict, 'handler': str}
              when include_confidence=True

    Example:
        >>> name_compound("CCO")
        'ethanol'
        >>> name_compound("CC(=O)O")
        'acetic acid'
        >>> name_compound("c1ccccc1")
        'benzene'
        >>> name_compound("C=CCO")
        'allyl alcohol'
        >>> name_compound("C=CCO", style="systematic")
        'prop-2-en-1-ol'
        >>> name_compound("CCO", include_confidence=True)
        {'name': 'ethanol', 'confidence': 1.0, ...}
    """
    namer = Orthonym(style=style)

    if include_confidence:
        try:
            return namer.name_with_confidence(smiles)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("name_with_confidence failed: %s", e)
            return {
                'name': _descriptive_fallback(smiles),
                'confidence': 0.0,
                'factors': {},
                'handler': 'fallback',
            }

    try:
        result = namer.name(smiles)
        if result:
            # Check if result contains 'unknown' as a component (partial failure)
            if 'unknown' in result.lower():
                return _descriptive_fallback(smiles)
            return result
        # If name() returned empty/None, generate descriptive fallback
        return _descriptive_fallback(smiles)
    except ValueError:
        raise  # Re-raise ValueError (invalid SMILES) for caller to handle
    except (TypeError, KeyError, IndexError, AttributeError) as e:
        # Graceful fallback for unexpected errors in the naming pipeline.
        # Log the error type for debugging but return a fallback name rather
        # than crashing or returning None.
        logger.debug(
            "Naming error for %s: %s: %s", smiles, type(e).__name__, e
        )
        return _descriptive_fallback(smiles)


def name_pipeline_only(smiles: str, style: str = "pin"):
    """Name a molecule using only the systematic pipeline, skipping decomposition.

    This provides the non-decomposition name without consuming any depth budget
    on decomposition probes. Used by the decomposition engine to compare its
    result against what the systematic pipeline would produce.

    Returns:
        IUPAC name string, or None if naming fails.
    """
    try:
        namer = Orthonym(style=style)
        return namer._name_impl(smiles, _skip_decomposition=True)
    except Exception:
        return None


# Metals and inorganic elements (not C, H, N, O, S, P, Se, halogens)
_ORGANIC_ELEMENTS = {
    'C', 'H', 'N', 'O', 'S', 'P', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'Si',
}

# Metal element -> name mapping for descriptive messages
_METAL_NAMES = {
    'Li': 'lithium', 'Na': 'sodium', 'K': 'potassium', 'Rb': 'rubidium',
    'Cs': 'cesium', 'Be': 'beryllium', 'Mg': 'magnesium', 'Ca': 'calcium',
    'Sr': 'strontium', 'Ba': 'barium', 'Al': 'aluminium', 'Ga': 'gallium',
    'In': 'indium', 'Tl': 'thallium', 'Sn': 'tin', 'Pb': 'lead',
    'Bi': 'bismuth', 'Ti': 'titanium', 'V': 'vanadium', 'Cr': 'chromium',
    'Mn': 'manganese', 'Fe': 'iron', 'Co': 'cobalt', 'Ni': 'nickel',
    'Cu': 'copper', 'Zn': 'zinc', 'Zr': 'zirconium', 'Mo': 'molybdenum',
    'Ru': 'ruthenium', 'Rh': 'rhodium', 'Pd': 'palladium', 'Ag': 'silver',
    'Cd': 'cadmium', 'W': 'tungsten', 'Re': 'rhenium', 'Os': 'osmium',
    'Ir': 'iridium', 'Pt': 'platinum', 'Au': 'gold', 'Hg': 'mercury',
    'Sb': 'antimony', 'Te': 'tellurium', 'Yb': 'ytterbium', 'La': 'lanthanum',
    'Ce': 'cerium', 'Nd': 'neodymium', 'Sm': 'samarium', 'Eu': 'europium',
    'Gd': 'gadolinium', 'Tb': 'terbium', 'Dy': 'dysprosium', 'Ho': 'holmium',
    'Er': 'erbium', 'Tm': 'thulium', 'Lu': 'lutetium', 'Sc': 'scandium',
    'Y': 'yttrium',
}


def _descriptive_fallback(smiles: str) -> str:
    """Generate a descriptive fallback message instead of bare 'unknown'.

    For inorganic/metallic compounds, returns a descriptive message like
    'gold compound (not supported)'. For organic molecules that failed
    naming, returns 'unknown organic compound'.

    Args:
        smiles: The SMILES string that could not be named.

    Returns:
        A descriptive string (never bare 'unknown').
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return "unknown"

        # Check for wildcard atoms (*)
        has_wildcard = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        if has_wildcard:
            return "compound with wildcard atoms (not supported)"

        # Check for non-organic elements.
        # IMPORTANT: iterate atoms in atom-index order to ensure deterministic
        # output across runs. A previous implementation iterated over a set()
        # of element symbols, which under Python's randomized hash returned a
        # different metal name on every run for multi-metal compounds — that
        # broke byte-identical reproducibility. Atom-index order is stable
        # (defined by canonical SMILES) and semantically intuitive: the
        # fallback names the molecule after the first metal encountered in
        # the structure, mirroring how a chemist reading the formula would.
        non_organic_in_order: list = []
        seen: set = set()
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in _ORGANIC_ELEMENTS or sym in seen:
                continue
            seen.add(sym)
            non_organic_in_order.append(sym)

        if non_organic_in_order:
            # Find the first metal in atom-index order
            metal_name = None
            for elem in non_organic_in_order:
                if elem in _METAL_NAMES:
                    metal_name = _METAL_NAMES[elem]
                    break

            if metal_name:
                return f"{metal_name} compound (not supported)"
            else:
                # Non-organic but unknown element
                return "inorganic compound (not supported)"

        # Organic compound that failed naming
        return "unknown organic compound"
    except Exception:
        return "unknown"
