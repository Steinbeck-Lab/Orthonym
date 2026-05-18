"""
Main IUPAC nomenclature generator.

Architecture:
    SMILES → Perception → Classification → Assembly → IUPAC Name

This is the inverse of OPSIN's pipeline:
    OPSIN:     Name → Tokenize → Parse → Build Structure
    Orthonym: Structure → Perceive → Classify → Assemble Name
"""

import contextvars
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

logger = logging.getLogger(__name__)

# CR-04 part B + W7: per-call NamingResult capture slot for name_with_tree.
# ContextVar provides thread-local AND asyncio-task-local isolation per PEP 567;
# safer than a module-global dict against concurrent Orthonym().name_with_tree()
# calls. The slot is installed by Orthonym.name_with_tree() before invoking
# self.name(); composer._assemble_name_impl writes the inner-dispatch
# NamingResult into the slot when present (default=None makes the regular
# Orthonym.name() path a no-op).
_name_with_tree_capture: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = (
    contextvars.ContextVar("name_with_tree_capture", default=None)
)

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

# Phase 158 NEW: routing substrate. `StoutClass` + `ClassDispatchResult`
# are imported eagerly at module-load time so `_name_impl`'s
# cascade-continuation + GENERAL fallback can reference them by name. The
# `routing` sub-package is callable-only (no callbacks back into namer)
# so this is NOT a circular import — the predicate factories + handler
# shims inside `routing/dispatch_table.py` use lazy imports for
# `orthonym.rules.*` and `orthonym.namer.Orthonym` per
# PATTERNS § 3 + namer.py:853 lazy-import precedent.
from .routing.dispatch_table import StoutClass, ClassDispatchResult


# ---------------------------------------------------------------------------
# Universal stereo backstop (Phase 140, STER-16)
# ---------------------------------------------------------------------------


def _final_stereo_check(mol, name: str, handler: str = 'unknown') -> str:
    """Universal stereo backstop: detect missing stereodescriptors in name.

    Runs AFTER all handler-specific stereo injection. Only activates when
    a handler missed stereo. Logs WARNING to flag handler gaps for future fixes.

    This is an architectural safety net per Phase 140 D-02 + Phase 152 D-04.
    Detection is delegated to rules.stereochemistry.needs_stereo_injection
    so that the new handler-level injector and this backstop share one
    predicate. Behavior is byte-identical to pre-152: backstop continues to
    fire WARNING for any handler that did NOT inject stereo (e.g.
    complex_ring, polycyclic, retained-name, decomposition fragments).

    It does NOT inject stereo with raw atom-index locants (Phase 140 D-02 +
    Phase 152 D-09) because those don't correspond to IUPAC numbering for
    the named parent structure -- injecting them would produce incorrect
    names. Handler-specific injection (Phase 152: benzene / heterocycle /
    cycloalkane / cycloalkene via inject_stereo_from_locant_map; legacy:
    _inject_stereo_if_missing for 33+ direct-return sites) remains the
    primary stereo injection mechanism.

    Args:
        mol: RDKit Mol object (with stereo info from original SMILES)
        name: Generated IUPAC name (may or may not contain stereo)
        handler: Name of the handler that produced this name (for attribution)

    Returns:
        Original name unchanged.  Logs WARNING if stereo gap detected.
    """
    from .rules.stereochemistry import needs_stereo_injection

    if not needs_stereo_injection(mol, name):
        return name

    # Predicate said True -> stereo was needed but not injected.  Log gap.
    n_atom_stereo = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    n_bond_stereo = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
    logger.warning(
        "Stereo backstop: '%s' (handler: %s) has %d R/S + %d E/Z but name lacks "
        "descriptors. Fix handler to include stereo natively.",
        name[:50], handler, n_atom_stereo, n_bond_stereo
    )
    return name


# ---------------------------------------------------------------------------
# Universal OPSIN-grammar backstop (Phase 156)
# ---------------------------------------------------------------------------


def _final_grammar_check(name: str, smiles: Optional[str], handler: str,
                         grammar, stats: Dict[str, int]) -> str:
    """Universal OPSIN-grammar backstop: validate + (round-trip-gated) repair.

    Mirrors the `_final_stereo_check` shape (Phase 152 D-04 pattern).
    Single chokepoint per CONTEXT.md D-13 — never per-handler-exit
    (AP-6). On unrepairable failure, log WARNING and return ORIGINAL
    name (D-11, D-15). Never silently mutate.

    Args:
        name: The assembled IUPAC name (post stereo backstop).
        smiles: Original SMILES the name was generated from. Forwarded
            to `OpsinGrammar.suggest_fix` for the round-trip gate per
            CONTEXT.md D-09 / D-10. May be None on cold paths; the
            grammar layer documents the degraded-path contract.
        handler: Handler attribution string (per Phase 152 D-04
            pattern); used in WARNING logs only.
        grammar: An `OpsinGrammar` instance (or None when the
            `_disable_grammar_validation` kwarg was passed at
            construction time per D-14).
        stats: The per-instance counter dict (D-17 / AP-19) shared by
            reference with the grammar instance. Mutated in-place.

    Returns:
        Either the validated name (happy path), the round-trip-gated
        repair (on validate-fail + successful repair), or the
        original name (on validate-fail + no repair). Never raises.
    """
    if grammar is None or not name:
        return name

    if grammar.validate(name):
        stats["validate_passed"] = stats.get("validate_passed", 0) + 1
        return name

    # validate() rejected — try suggest_fix (D-09 round-trip-gated).
    # D-10 LOCKED signature: name FIRST, source_smiles SECOND.
    repaired, repair_class = grammar.suggest_fix(name, source_smiles=smiles)

    if repaired is not None and repaired != name:
        # The grammar layer already incremented the matching
        # `repair_succeeded_<class>` bucket internally per D-17.
        logger.warning(
            "OPSIN grammar repair: handler=%s class=%s original=%r repaired=%r",
            handler, repair_class, name[:50], repaired[:50],
        )
        return repaired

    # validate() rejected and no class produced a round-trip-passing
    # candidate. Log unrepairable WARNING and fall back to ORIGINAL
    # name per D-11 + D-15 (never silently mutate).
    logger.warning(
        "OPSIN grammar validation failed: handler=%s name=%r",
        handler, name[:50],
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

    # Phase 151-02 D-09: skip Branch 1 (fused-heterocycle catalog) when
    # the input is mixed-spiro/fused — Branch 5b owns that dispatch.
    # Without this guard, the catalog returns a PARTIAL locant map
    # (only the fused component's atoms; missing the spiro side ring),
    # which violates the cascade-step-6 coverage invariant downstream.
    from .rules.spiro import is_mixed_spiro_fused as _phase151_is_mixed_spiro_fused
    _is_mixed_spiro_fused_input = _phase151_is_mixed_spiro_fused(mol)

    # Branch 1: fused-heterocycle (preserve D-09 byte-identical path for
    # ACTUAL heterocycles — indole/quinoline/etc.). Skip when the matched
    # core has no heteroatoms (e.g., pyrene also lives in
    # FUSED_HETEROCYCLES registry incidentally; PAHs must flow to branch 2
    # so their tuple-locant numbering is used per W-1 fix).
    fused_type = classify_fused_system(mol)
    if not _is_mixed_spiro_fused_input and fused_type in ('ortho-fused', 'ortho-peri-fused'):
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

    # Branch 5 (Phase 151-02 D-21): pure spiro + mixed spiro/fused
    # cascade-step-6 suppliers.
    #
    # Pure spiro: is_spiro_system True iff n_rings == n_spiro + 1 (D-09 lock).
    # Wraps existing get_spiro_numbering / _get_polyspiro_numbering with
    # the cascade-step-6 coverage invariant (Pitfall 7).
    #
    # Mixed spiro/fused: is_mixed_spiro_fused True iff ≥1 spiro atom AND
    # n_rings > n_spiro + 1 AND detect_natural_product is None (Pitfall 3
    # false-positive guard) AND the topology is HERITAGE §4 separable AND
    # the fused part has a catalog name (D-22(b) canary-stability guard).
    #
    # If neither supplier returns full coverage (None), fall through to
    # Branch 6 (Phase 151-01 VB) and onwards.
    #
    # Source: 151-CONTEXT.md D-09 / D-13 / D-21; 151-AUDIT-B.md verdict
    # MIXED_SPIRO_FUSED_MISSING + Q-05 NESTED_FORM_PARSEABLE.
    from .rules.spiro import (
        is_spiro_system,
        is_mixed_spiro_fused,
        get_spiro_iupac_locants,
        get_mixed_spiro_fused_iupac_locants,
    )
    if is_spiro_system(mol):
        sl = get_spiro_iupac_locants(mol)
        if sl is not None:
            return {"iupac_locants": sl}
        # Else: spiro system but supplier declined coverage — fall through
        # to the existing get_spiro_atoms-True stub return so other
        # downstream branches don't attempt to take over.
        return {"iupac_locants": None}
    if is_mixed_spiro_fused(mol):
        msfl = get_mixed_spiro_fused_iupac_locants(mol)
        if msfl is not None:
            return {"iupac_locants": msfl}
        # Mixed-spiro/fused but supplier declined — emit None so the
        # cascade-step-6 gate falls through to the sorted-int proxy
        # rather than mis-routing to Branch 6 (VB).
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

    # Branch 7 (Phase 151-03 D-21): ring assembly size 3+ supplier.
    # Wires get_ring_assembly_iupac_locants per 151-AUDIT-C.md verdict
    # SUPPLIER_MISSING + 151-PATTERNS.md Pattern S-3. Path-topology check
    # added to detect_ring_assembly per D-15 rejects branched arrangements
    # (1,3,5-triphenylbenzene) so Branch 7 only fires on linear chains.
    # 2-system bi- assemblies are NOT routed here (covered by existing
    # naming pipeline + the >=3 gate keeps blast-radius minimal).
    #
    # Source: 151-CONTEXT.md D-15 / D-18 / D-19 / D-21; 151-AUDIT-C.md
    # composite verdict; Pitfall 7 (full coverage or None).
    if hasattr(features, 'ring_systems') and len(features.ring_systems) >= 3:
        from .rules.ring_assemblies import (
            detect_ring_assembly as _phase151_detect_ring_assembly,
            get_ring_assembly_iupac_locants as _phase151_get_ra_locants,
        )
        info = _phase151_detect_ring_assembly(mol, features.ring_systems)
        if info is not None and info.get("count", 0) >= 3:
            ral = _phase151_get_ra_locants(mol)
            if ral is not None:
                return {"iupac_locants": ral}

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
    
    def __init__(self, style: str = "pin", *,
                 _disable_grammar_validation: bool = False,
                 allow_ml_fallback: bool = False,
                 opsin_parse_required: bool = True):
        """
        Initialize namer.

        Args:
            style: Naming style
                - "pin": Preferred IUPAC Names (IUPAC 2013)
                - "general": General IUPAC (more flexible)
                - "cas": CAS-style naming
            _disable_grammar_validation: Phase 156 escape hatch (D-14).
                When True, the OPSIN grammar pre-validation layer is
                disabled (`self._grammar` is None). ON by default in
                production; OFF only for unit tests inspecting raw
                handler output.
            allow_ml_fallback: Phase 162 MLF-01 opt-in flag. Default
                False (rule-based pipeline only). When True, the
                _name_impl wrapper (Plan-03 T01) consults
                is_name_quality_inadequate() and may attach the STOUT
                ML model when the rule-based pipeline emits a degraded
                name. Requires the [ml] optional extra.
            opsin_parse_required: Phase 162 D-08 quality-gate flag.
                Default True (OPSIN-parse criterion ON for production
                correctness). When False, the expensive P5 OPSIN
                subprocess is bypassed (raw-attach-rate measurement
                mode for the MLF-04 dual-config benchmark).
        """
        self.style = style
        # Phase 162 ML Fallback Gate per MLF-01 + D-08 (kwargs stored;
        # the wrapper at _name_impl():1248 lands in Plan-03 T01).
        self._allow_ml_fallback: bool = allow_ml_fallback
        self._opsin_parse_required: bool = opsin_parse_required
        # Per-instance state: populated by Plan-03 T01 wrapper when
        # MLFallbackInvoker fires. None at construction; reset to None
        # at the start of every name_with_confidence call.
        self._last_ml_result: Optional[Any] = None
        # Phase 156 D-17 + AP-19: per-instance counter dict, NEVER
        # module-global. Pre-seed all seven buckets so callers see a
        # complete histogram even before any name() invocation.
        from .validation.opsin_grammar import OpsinGrammar
        self._grammar_stats: Dict[str, int] = {
            k: 0 for k in OpsinGrammar.STAT_KEYS
        }
        if _disable_grammar_validation:
            self._grammar = None
        else:
            # Share the dict by reference per D-17: the grammar
            # instance increments the same dict the Orthonym instance
            # exposes via `get_validation_stats()`.
            self._grammar = OpsinGrammar(stats=self._grammar_stats)

        # Phase 158 D-14 NEW: instantiate CFR router (default-ON; no opt-out
        # flag per CONTEXT D-14 + AP-2 + AP-19). Per audit § 3.6 RL-4
        # fresh-instance invariant: each Orthonym() carries its OWN router
        # with empty dispatch_stats; recursive `Orthonym(...)` calls produce
        # fresh routers, and the byte-identical contract is on NAME OUTPUT
        # only, NOT on per-call dispatch_stats.
        from .routing import ClassFirstRouter
        self._cfr_router = ClassFirstRouter()

        # Phase 158 RL-7 option (b) NEW: skip-decomposition flag threading.
        # `_name_impl(_skip_decomposition=True)` (called by
        # `name_pipeline_only()`) sets this on entry; the
        # DECOMPOSITION_PRE_GENERAL predicate factory reads from kwargs
        # passed by `dispatch()`.
        self._skip_decomposition: bool = False

    def get_dispatch_stats(self) -> Dict[Any, int]:
        """Phase 158 D-16: per-instance CFR dispatch histogram.

        Returns a defensive copy of the (StoutClass -> int) histogram
        recorded by the CFR router on every dispatch. Per CONTEXT D-16 +
        AP-6 the counter lives on the Orthonym instance via the CFR
        router; reset on demand via `reset_dispatch_stats()`.

        The return type is `Dict[Any, int]` (not `Dict[StoutClass, int]`)
        to avoid eager `from .routing import StoutClass` at module-load
        time, which would create a circular import. Callers that need
        the StoutClass type can import it directly from `orthonym.routing`.
        """
        return self._cfr_router.get_dispatch_stats()

    def reset_dispatch_stats(self) -> None:
        """Phase 158 D-16 + Phase 160 D-18: explicit reset for batch-run boundaries.

        Resets BOTH the Phase 158 outer-CFR (per-instance) counter AND the
        Phase 160 inner-dispatch (module-level) counter. The inner-dispatch
        counter is module-level today (see assembly/inner_dispatch.py
        :_INNER_DISPATCH_STATS) which means it is shared across Orthonym
        instances — calling ``reset_dispatch_stats()`` on one instance
        resets the shared inner counter visible to all instances.
        """
        self._cfr_router.reset_dispatch_stats()
        # Phase 160 D-18: also reset the inner-dispatch counter.
        from .assembly.inner_dispatch import reset_inner_dispatch_stats
        reset_inner_dispatch_stats()

    def get_inner_dispatch_stats(self) -> Dict[str, int]:
        """Phase 160 D-18: inner-dispatch per-handler-id counters.

        Returns a defensive copy of the (handler_id -> int) histogram
        recorded by ``assembly/inner_dispatch.dispatch_inner`` on every
        match. Inner-dispatch is the second stage of the Phase 158 +
        Phase 160 dispatch pipeline: outer CFR routes to a StoutClass;
        for the GENERAL class, inner-dispatch then routes to one of the
        30 handlers in ``INNER_DISPATCH_TABLE``.

        Per CONTEXT D-18: companion to ``get_dispatch_stats()``; CLI
        ``--dispatch-stats`` flag prints both together. Per AP-160-13 the
        underlying counter is module-level (shared across instances)
        because inner-dispatch is a pure-function call site — adding
        per-instance threading would require touching every handler entry
        point. The counter is reset via ``reset_dispatch_stats()`` (which
        clears BOTH outer + inner counters).

        Returns:
            Dict mapping handler_id (str) to dispatch count (int). Empty
            dict if no inner-dispatch calls have happened yet.
        """
        from .assembly.inner_dispatch import get_inner_dispatch_stats as _stats
        return _stats()

    def name_with_tree(self, smiles: str):
        """Phase 160 DECOMP-02 public API: return NamingResult(name, tree, hint).

        Phase 160 ships the NameTreeNode IR substrate alongside the legacy
        ``assemble_name`` path; first-wave handlers (Plans 02-03 ship)
        return ``NamingResult(name=<final string>, tree=None, ...)`` per
        CONTEXT D-05 incremental migration. The ``tree`` field is None for
        the 30 currently-extracted handlers; tree population is a v19+
        follow-up phase. The ``name`` field is byte-identical to
        ``Orthonym.name(smiles)``.

        For tree-emitting handlers (v19+1 onwards), this method returns a
        NamingResult whose ``tree`` is a NameTreeNode and where
        ``name_tree_to_string(tree)`` round-trips to ``name`` byte-for-byte.

        Args:
            smiles: SMILES string to convert to IUPAC name.

        Returns:
            NamingResult NamedTuple with fields:
              - name: str (byte-identical to Orthonym.name(smiles))
              - tree: Optional[NameTreeNode] (None for first-wave handlers)
              - atom_to_locant_hint: Optional[Dict[int, int]]

        Raises:
            ValueError: If SMILES is invalid.
        """
        # Lazy import to avoid composer.py -> name_tree -> namer.py cycle
        # at module-load time.
        from .assembly.name_tree import NamingResult
        # CR-04 part B + W7: install a per-call capture slot via
        # contextvars.ContextVar (PEP 567) so composer._assemble_name_impl
        # can write the inner-dispatch NamingResult into it without
        # changing the public assemble_name return type. ContextVar is
        # thread-local AND asyncio-task-local — safe under concurrent
        # invocation from multiple threads / tasks.
        token = _name_with_tree_capture.set({"naming": None})
        try:
            name = self.name(smiles)
            slot = _name_with_tree_capture.get()
            captured = slot["naming"] if slot else None
            tree = captured.tree if captured is not None else None
            hint = captured.atom_to_locant_hint if captured is not None else None
        finally:
            _name_with_tree_capture.reset(token)
        return NamingResult(name=name, tree=tree, atom_to_locant_hint=hint)

    def get_validation_stats(self) -> Dict[str, int]:
        """Return a defensive copy of the per-instance grammar counters.

        Phase 156 D-17 telemetry accessor. Buckets are pre-seeded in
        `__init__`; counters are mutated in-place by the underlying
        `OpsinGrammar` instance via the shared-by-reference dict.
        """
        return dict(self._grammar_stats)

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
                    # Universal OPSIN-grammar backstop (Phase 156, D-13).
                    # Order: stereo-backstop -> grammar-backstop. Stereo
                    # may have repositioned descriptors that grammar
                    # then re-validates.
                    result = _final_grammar_check(
                        result, smiles, handler,
                        self._grammar, self._grammar_stats,
                    )
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
                    # Universal OPSIN-grammar backstop (Phase 156, D-13).
                    name = _final_grammar_check(
                        name, smiles, handler,
                        self._grammar, self._grammar_stats,
                    )
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
        """Internal naming implementation (wrapped by session management).

        Phase 158 substrate: the v18 implicit cascade at lines 853-1115 is
        REPLACED by a single ``self._cfr_router.dispatch(...)`` call. Per
        audit § 3.5 RL-3 option (c) the zwitterion-character in-place
        mutation (v18 lines 1000-1034) lives INLINE here BEFORE the dispatch
        call — preserves D-26 predicate-purity invariant cleanly. Per
        Task 158-02-01 design choice (a) the GENERAL handler shim returns
        None to signal that ``_name_impl`` runs the legacy ``_perceive ->
        _classify -> assemble_name`` pipeline INLINE — keeps ``routing/``
        decoupled from ``composer.py`` (D-19 boundary). Quality-gate
        post-checks at v18 lines 1129-1207 are PRESERVED VERBATIM after
        the dispatch call.
        """
        # Phase 158 RL-7 option (b): thread `_skip_decomposition` through
        # the instance attribute so the DECOMPOSITION_PRE_GENERAL predicate
        # factory + handler shim can read it from kwargs forwarded by
        # `dispatch()`.
        self._skip_decomposition = _skip_decomposition

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get canonical SMILES for consistent processing
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # ============================================================
        # Pre-dispatch: zwitterion-character in-place mutation (RL-3 (c))
        # ============================================================
        # Phase 158 audit § 3.5 RL-3 option (c) lock: the v18 cascade at
        # namer.py:1000-1034 MUTATED `mol` and `canonical_smiles` in-place
        # for downstream cascade consumption. CFR's predicate-handler model
        # does NOT support "mutate state, continue cascade" patterns
        # natively. Per the audit's UNANIMOUS decision, the mutation lives
        # INLINE here BEFORE CFR.dispatch — preserves D-26 hard invariant
        # (every entry's `side_effect_inventory == ()`) cleanly.
        #
        # CRITICAL byte-identical gate: in the v18 cascade the mutation at
        # line 1000-1034 was only REACHED when ALL of the prior branches
        # (salt/radical/zwitterion/ion at 852-956 + multi-component-neutral
        # at 967-998) had already declined to handle the molecule. Those
        # branches return early on success, so the mutation effectively
        # only ran on `species_type == 'neutral'` molecules that were not
        # multi-component cocrystals. To preserve byte-identical behavior
        # we MUST gate the inline pre-dispatch mutation on
        # `species_type == 'neutral'`; otherwise salts like `[Ag+].[Cl-]`
        # (which DO satisfy `_has_true_zwitterion_character` because they
        # have both + and - atoms) get mutated before SALT routing fires
        # and salt detection fails downstream.
        # [Rule 1 - Bug] Found during Task 158-02-07 canary investigation.
        #
        # CHARGE NEUTRALIZATION for large zwitterions reclassified as
        # 'neutral' by the HA>20 + quaternary-N guard in
        # `detect_species_type()`. These molecules still carry formal
        # charges (P-O-, N+) that prevent FG detection SMARTS from matching
        # (e.g., [OX2H] for COOH). Neutralize O- to OH; leave quaternary
        # N+ (no H) charged to preserve valid valence. ONLY applies to
        # molecules that have true zwitterion character -- NOT to molecules
        # with internal charges (nitro [N+](=O)[O-], azide, N-oxide) which
        # are normal functional groups.
        _zwitter_species_type = detect_species_type(mol)
        if _zwitter_species_type == 'neutral' and Chem.GetFormalCharge(mol) == 0:
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

        # ============================================================
        # CFR dispatch — replaces v18 cascade lines 853-1115
        # ============================================================
        # Phase 158 D-13 single chokepoint integration: CFR routes input
        # mol; backstops (Phase 152 + 156) wrap output name at
        # `Orthonym.name()` line 654. The CFR dispatcher walks
        # DISPATCH_TABLE in priority order; the first predicate that
        # matches wins; the result's handler is invoked here.
        #
        # The audit § 1 DISPATCH_TABLE has 17 outer-cascade rows + GENERAL
        # catch-all = 18 entries. Several handler shims (ANION_SMALL,
        # POLY_ANION, MULTI_COMPONENT_NEUTRAL, DECOMPOSITION_PRE_GENERAL)
        # may return None to signal "no match — continue cascade", in
        # which case we re-dispatch with the matched class excluded so the
        # next priority entry runs. This mirrors the v18 cascade's
        # fall-through behavior (e.g., namer.py:913-914 + 955-956).
        #
        # `_skip_decomposition` is threaded via kwargs per RL-7 option (b);
        # `style` is threaded via kwargs so the RETAINED_NAME and
        # AMINO_ACID predicates can read it without binding `self`.
        result = self._cfr_router.dispatch(
            mol, smiles, canonical_smiles,
            _skip_decomposition=self._skip_decomposition,
            _style=self.style,
        )
        name = result.handler(
            mol, smiles, canonical_smiles,
            features=None,
            style=self.style,
            _skip_decomposition=self._skip_decomposition,
        )

        # Cascade-continuation when a handler returns None (audit § 1 row
        # notes for ANION_SMALL / POLY_ANION / MULTI_COMPONENT_NEUTRAL /
        # DECOMPOSITION_PRE_GENERAL). The v18 cascade falls through to the
        # next sibling; mirror that here by re-running dispatch with
        # progressively higher priority floors.
        if name is None and result.class_id != StoutClass.GENERAL:
            from .routing.dispatch_table import DISPATCH_TABLE
            current_priority = DISPATCH_TABLE[result.class_id].priority
            for entry in sorted(DISPATCH_TABLE.values(), key=lambda e: e.priority):
                if entry.priority <= current_priority:
                    continue
                # Predicate signature: (mol, smiles, canonical_smiles, features, **kwargs)
                try:
                    matched = entry.predicate(
                        mol, smiles, canonical_smiles, None,
                        _skip_decomposition=self._skip_decomposition,
                        _style=self.style,
                    )
                except TypeError:
                    matched = entry.predicate(mol, smiles, canonical_smiles, None)
                if not matched:
                    continue
                self._cfr_router._dispatch_stats[entry.class_id] += 1
                name = entry.handler(
                    mol, smiles, canonical_smiles,
                    features=None,
                    style=self.style,
                    _skip_decomposition=self._skip_decomposition,
                )
                result = ClassDispatchResult(
                    class_id=entry.class_id,
                    handler=entry.handler,
                    audit_record={},
                    tier=entry.tier,
                )
                if name is not None or entry.class_id == StoutClass.GENERAL:
                    break

        # ============================================================
        # GENERAL pipeline fallback — Task 158-02-01 design choice (a)
        # ============================================================
        # The GENERAL handler shim returns None to signal that
        # `_name_impl` runs the legacy `_perceive -> _classify ->
        # assemble_name` pipeline inline. This avoids coupling
        # `routing/` to `composer.py` (D-19 boundary). The check below
        # mirrors the v18 cascade's GENERAL pipeline at namer.py:1124-1131.
        if name is None and result.class_id == StoutClass.GENERAL:
            features = self._perceive(mol, smiles, canonical_smiles)
            self._classify(features)
            assembled = assemble_name(features, style=self.style)
            name = assembled

        # ============================================================
        # Post-dispatch quality gates — preserved VERBATIM from v18
        # lines 1129-1207. CFR is the dispatch substrate; these gates
        # run on the GENERAL pipeline's assembled name AFTER dispatch,
        # NOT on outputs from class-specific handlers.
        #
        # [Rule 1 - Bug] Found during Task 158-02 verification: gating
        # on ANY non-None name (the original implementation) caused
        # PEPTIDE / SALT / etc. handler outputs to be re-checked
        # against the GENERAL pipeline's confidence store, which is
        # populated by `assemble_name()` and may still hold stale data
        # from the PREVIOUS call (the `_confidence_store` is thread-
        # local and persists across `Orthonym.name()` invocations; see
        # `assembly/coverage_scoring.py:_confidence_store`). Pre-CFR
        # the gate only ran on `assembled = assemble_name(...)` output
        # at the very end of the cascade — handler early-returns
        # (e.g. peptide / salt / amino_acid) bypassed it entirely.
        # Restrict the gate to GENERAL-pipeline outputs to preserve
        # CFR-04 byte-identical canary. Honest-fail-on-data per
        # CONTEXT D-29: the fix is upstream (gate scope), NOT a
        # threshold relaxation or postprocessor band-aid.
        # ============================================================
        if (not _skip_decomposition
                and name
                and result.class_id == StoutClass.GENERAL
                and mol.GetNumHeavyAtoms() > 15):
            assembled = name  # local alias for v18-byte-identical body
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

        return name
    
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



def name_with_tree(smiles: str, style: str = "pin"):
    """Module-level convenience wrapper for Orthonym(style).name_with_tree(smiles).

    CR-04 part A (BLOCKER): downstream consumers expect
    ``from orthonym import name_with_tree`` to work analogously to
    ``name_compound``.

    Args:
        smiles: SMILES string to name.
        style: 'pin' (preferred) | 'systematic' | 'cas'. Default 'pin'.

    Returns:
        NamingResult(name, tree, atom_to_locant_hint).
    """
    return Orthonym(style=style).name_with_tree(smiles)


def name_compound(smiles: str, style: str = "pin",
                   include_confidence: bool = False,
                   *,
                   allow_ml_fallback: bool = False,
                   opsin_parse_required: bool = True):
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
        allow_ml_fallback: Phase 162 MLF-01 opt-in flag (default False).
            See Orthonym.__init__ docstring.
        opsin_parse_required: Phase 162 D-08 quality-gate flag (default True).
            See Orthonym.__init__ docstring.

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
    namer = Orthonym(
        style=style,
        allow_ml_fallback=allow_ml_fallback,
        opsin_parse_required=opsin_parse_required,
    )

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
