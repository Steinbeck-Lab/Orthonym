"""
Stereochemistry perception and CIP assignment.

CRITICAL: Always use rdCIPLabeler.AssignCIPLabels(), not the legacy
Chem.AssignStereochemistry() which fails on complex molecules.
"""

import logging
import os
from typing import List, Dict, Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

logger = logging.getLogger(__name__)

_CIP_ASSIGNED_PROP = '_Orthonym_CIPAssigned'

# WSB-03: centres CIP engine. Read once at import time
# (same idiom as namer.ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER).
#
# STER-02 (Phase H, 2026-06-21): the default is now ON. The vendored `centres`
# scores 279/290 on the Hanson 2018 CIP Validation Suite vs
# rdCIPLabeler's 235/290 -- a net +44 CORRECT labels with **0 per-compound
# regressions vs RDKit** (centres remains a strict superset of RDKit on the
# suite; the gain is exotic CIP rule cases RDKit mis-ranks). When centres is
# unavailable (jar or Java absent) the code falls through to rdCIPLabeler
# UNCHANGED -- a missing JVM never hard-fails a name. Set
# ORTHONYM_USE_CENTRES_CIP=0/off to force the legacy RDKit-only path.
#
# CIP-UPDATE: engine refreshed 1.2.1 -> 1.5 (SiMolecule/centres
# develop @ d4b3cf0). All R/S/E/Z labels are byte-identical to 1.2.1; the only
# delta is 2 exotic CYCLIC-CUMULENE axial M/P labels (suite 281->279), which
# Orthonym does NOT consume (allene/axial CIP is computed independently in
# detect_axial_chirality + _manual_allene_cip), so naming is unaffected.
_USE_CENTRES_CIP = os.environ.get(
    "ORTHONYM_USE_CENTRES_CIP", "on"
).strip().lower() in ("1", "true", "yes", "on")


def _fill_missing_bond_cip_from_rdkit(mol) -> None:
    """Complementary rdCIPLabeler pass: FILL double-bond ``_CIPCode`` that the
    primary labeller (centres) left empty, without touching any label it set.

    The vendored ``centres`` engine is the CIP source-of-truth for atoms (279/290
    vs rdCIPLabeler's 235/290 on the Hanson 2018 suite) but does NOT emit an E/Z
    label for an *exocyclic* double bond to an aromatic-flagged ring atom -- the
    o-/p-quinoid and fulvenoid (``-ylidene``) systems -- so those bonds reached
    ``collect_stereodescriptors`` with no ``_CIPCode`` and their stereo was
    silently dropped (M3 finding, 2026-08-30). rdCIPLabeler labels them correctly.

    This runs rdCIPLabeler on a COPY and copies over ``_CIPCode`` ONLY for a
    DOUBLE bond that (a) mol currently has no E/Z code for and (b) the copy codes
    ``E``/``Z``. It never overwrites a code centres already set, so centres keeps
    ownership of every atom and every bond it did label. Additive-only and
    fail-open: any error leaves the primary labels unchanged.
    """
    try:
        needs = [b.GetIdx() for b in mol.GetBonds()
                 if b.GetBondType() == Chem.BondType.DOUBLE
                 and not (b.HasProp('_CIPCode') and b.GetProp('_CIPCode') in ('E', 'Z'))]
        if not needs:
            return
        probe = Chem.Mol(mol)  # preserves atom/bond indices and bond stereo
        rdCIPLabeler.AssignCIPLabels(probe)
        for idx in needs:
            pb = probe.GetBondWithIdx(idx)
            if pb.HasProp('_CIPCode') and pb.GetProp('_CIPCode') in ('E', 'Z'):
                mol.GetBondWithIdx(idx).SetProp('_CIPCode', pb.GetProp('_CIPCode'))
    except Exception:
        # Fail-open on ANYTHING: this fill is a purely-additive enhancement, so a
        # failure must leave the primary (centres) labels intact and MUST NOT
        # propagate. A narrower clause let an unlisted exception escape to
        # assign_stereochemistry's outer ``except Exception``, which re-ran the
        # RDKit-only labeller on a mol centres had ALREADY labelled -- silently
        # downgrading it from the 279/290 engine to 235/290 (review #3).
        return


def assign_stereochemistry(mol) -> None:
    """
    Assign CIP stereochemistry labels to a molecule (idempotent guard).

    Uses a private marker property to track whether rdCIPLabeler has
    already been called on this mol object. This is more reliable than
    checking for _CIPCode because RDKit's MolFromSmiles() automatically
    sets atom _CIPCode from @/@@ notation, but does NOT set bond _CIPCode
    for E/Z -- so an atom-based check would short-circuit and skip the
    bond labels.

    The authoritative call site in namer.py:_perceive() sets the marker
    after calling rdCIPLabeler. Handler modules call this function for
    safety (e.g., natural_products runs BEFORE _perceive()).

    Args:
        mol: RDKit Mol object (modified in place)
    """
    if mol.HasProp(_CIP_ASSIGNED_PROP):
        return

    # WSB-03: when the centres gate is ON AND the engine is available,
    # centres is the CIP source-of-truth (it sets the same _CIPCode props the
    # downstream consumers read). If the gate is OFF (default) or centres is
    # unavailable (jar/Java absent), this branch is skipped and the path below
    # is byte-identical to HEAD -- a missing JVM never hard-fails a name.
    if _USE_CENTRES_CIP:
        try:
            from .centres_bridge import centres_label_mol
            if centres_label_mol(mol):
                # centres owns atoms + the bonds it labelled; fill only the
                # exocyclic-ylidene double bonds it leaves unlabelled (M3).
                _fill_missing_bond_cip_from_rdkit(mol)
                mol.SetProp(_CIP_ASSIGNED_PROP, '1')
                return
        except Exception as exc:  # pragma: no cover - defensive; fall back to RDKit
            logger.warning(
                "centres CIP path errored (%s); falling back to rdCIPLabeler", exc,
            )

    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except (ValueError, RuntimeError, KeyError) as exc:
        mol_info = Chem.MolToSmiles(mol) if mol.GetNumAtoms() > 0 else "empty"
        logger.warning(
            "CIP assignment failed for %s, falling back to legacy: %s",
            mol_info, exc,
        )
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    mol.SetProp(_CIP_ASSIGNED_PROP, '1')


def get_stereocenters(mol) -> List[Dict]:
    """
    Get all stereocenters with their CIP labels.

    Args:
        mol: RDKit Mol object (stereochemistry should be assigned first)

    Returns:
        List of dicts with keys:
        - idx: atom index
        - cip: 'R' or 'S'
        - symbol: atom element symbol
        - neighbors: list of neighbor atom indices
    """
    # Ensure stereochemistry is assigned
    assign_stereochemistry(mol)
    
    centers = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            cip_code = atom.GetProp('_CIPCode')
            # Preserve CIP code as-is: uppercase R/S for normal stereocenters,
            # lowercase r/s for pseudoasymmetric centers per IUPAC P-92.1.4.2.
            centers.append({
                'idx': atom.GetIdx(),
                'cip': cip_code,
                'symbol': atom.GetSymbol(),
                'neighbors': [n.GetIdx() for n in atom.GetNeighbors()],
            })
    
    return centers


def get_double_bond_stereo(mol) -> List[Dict]:
    """
    Get E/Z configuration of double bonds.

    Uses the _CIPCode property set by rdCIPLabeler as the sole source
    of E/Z labels. BondStereo fallback was removed-03.

    Args:
        mol: RDKit Mol object (stereochemistry should be assigned via
             rdCIPLabeler.AssignCIPLabels BEFORE calling this function)

    Returns:
        List of dicts with keys:
        - idx: bond index
        - stereo: 'E' or 'Z'
        - atoms: (begin_atom_idx, end_atom_idx)
    """
    stereo_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if bond.HasProp('_CIPCode'):
                cip_code = bond.GetProp('_CIPCode')
                if cip_code in ('E', 'Z'):
                    stereo_bonds.append({
                        'idx': bond.GetIdx(),
                        'stereo': cip_code,
                        'atoms': (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    })

    return stereo_bonds


def has_stereochemistry(mol) -> bool:
    """
    Check if molecule has any defined stereochemistry.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule has stereocenters or double bond stereo
    """
    assign_stereochemistry(mol)
    
    # Check for atom stereocenters
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            return True
    
    # Check for defined double bond stereochemistry via _CIPCode
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if bond.HasProp('_CIPCode') and bond.GetProp('_CIPCode') in ('E', 'Z'):
                return True
    
    return False


def get_stereodescriptor_string(
    mol,
    locant_map: Optional[Dict[int, int]] = None
) -> str:
    """
    Generate stereodescriptor string for name prefix.

    DEPRECATED: Prefer collect_stereodescriptors() + format_stereodescriptor_string()
    from orthonym.rules.stereochemistry directly.

    Delegates to the production pipeline. When locant_map is None, uses an identity
    mapping (atom_idx -> idx+1) for backward compatibility with test call sites.

    Args:
        mol: RDKit Mol object
        locant_map: Optional mapping from atom index to locant number.
                    If None, uses identity mapping (idx+1).

    Returns:
        Stereodescriptor string like "(2R,3S)-" or empty string if no stereochemistry.
    """
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)

    if locant_map is None:
        # Identity mapping for backward compat (test call sites)
        locant_map = {atom.GetIdx(): atom.GetIdx() + 1 for atom in mol.GetAtoms()}

    descriptors = collect_stereodescriptors(mol, locant_map)
    if not descriptors:
        return ""

    return format_stereodescriptor_string(descriptors)


def count_stereocenters(mol) -> int:
    """
    Count the number of stereocenters in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of defined stereocenters
    """
    return len(get_stereocenters(mol))


def count_double_bond_stereo(mol) -> int:
    """
    Count the number of double bonds with defined E/Z stereochemistry.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of E/Z defined double bonds
    """
    return len(get_double_bond_stereo(mol))


def is_chiral(mol) -> bool:
    """
    Check if molecule has any chiral centers.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule has at least one defined stereocenter
    """
    return count_stereocenters(mol) > 0


def get_undefined_stereocenters(mol) -> List[int]:
    """
    Find potential stereocenters without defined stereochemistry.

    Args:
        mol: RDKit Mol object

    Returns:
        List of atom indices that are potential stereocenters
        but don't have defined R/S configuration
    """
    # Get atoms flagged as potential stereocenters
    Chem.AssignStereochemistry(mol, cleanIt=False, force=False, flagPossibleStereoCenters=True)

    undefined = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            if not atom.HasProp('_CIPCode'):
                undefined.append(atom.GetIdx())

    return undefined


def input_stereo_undefined(mol, atom_indices: Optional[Iterable[int]] = None) -> bool:
    """True iff ``mol`` leaves a REAL stereo feature UNDEFINED: a tetrahedral
    stereocentre flagged possible but carrying no CIP configuration, or a
    stereogenic double bond left undirected. When ``atom_indices`` is given, the
    tetrahedral check is restricted to those atoms (bond check is unrestricted).

    The shared stereo-honesty predicate for: a config-implying
    retained name (steroid ``cholest-``/``androst-``, amino acid
    ``S-methylcysteine``) asserts a specific configuration, so it must not be
    emitted when this returns True -- that would fabricate stereo the input
    never defined (P-103.1.3.1 / P-92). Fail-CLOSED: any failure returns True,
    so an uncomputable case never lets a fabrication through.
    """
    if mol is None:
        return True
    try:
        undef = get_undefined_stereocenters(mol)   # tetrahedral, existing helper
        if atom_indices is not None:
            allowed = set(atom_indices)
            undef = [i for i in undef if i in allowed]
        if undef:
            return True
        # Stereogenic-but-undirected double bonds (belt-level; scoped producers
        # rarely hit this, but the contract covers it).
        probe = Chem.Mol(mol)
        Chem.FindPotentialStereoBonds(probe)
        for bond in probe.GetBonds():
            if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                return True
        return False
    except Exception:
        return True


# =============================================================================
# Axial Chirality Detection (IUPAC P-93.5)
# =============================================================================

#: The axial descriptor in GENERAL nomenclature, keyed by the PIN (helicity)
#: descriptor. P-91.2.1.1 "Cahn-Ingold-Prelog (CIP) stereodescriptors"
#: (BlueBookV2.md:44582) lists under *"The following stereodescriptors are used
# : as preferred stereodescriptors"* clause (c) (44588) *"'M' and 'P', to specify
#: the absolute configuration of an axial or planar entity using the helicity
#: rule"*; 'Ra'/'Sa' appear only under *"The following stereodescriptors are
# : recommended for general nomenclature"* (44594). The two describe the same
# : sense: P-92.1.2.2 "The helicity rule: stereodescriptors 'M' and 'P'" (44812)
#: -- *"the chirality is described by the symbols 'M' if the path is
#: anticlockwise; the symbol is 'P' if the path is clockwise"* -- is the same
#: clockwise/anticlockwise test the Ra/Sa elongated-tetrahedron model applies,
#: so Ra == P and Sa == M.
AXIAL_GENERAL_FORM = {"P": "Ra", "M": "Sa"}


def detect_axial_chirality(mol, style: str = "pin") -> List[Dict]:
    """
    Detect allene and atropisomer axial chirality in a molecule.

    Identifies two types of axial chirality encoded in the input:
    1. Atropisomers: bonds with STEREOATROPCW or STEREOATROPCCW stereo
    2. Allenes: atoms with CHI_ALLENE chiral tag on central C of C=C=C

    Only detects chirality that is explicitly encoded in the molecular
    representation. Does NOT attempt to infer chirality where the input
    is silent.

    Descriptor (P-91.2.1.1,:44582 -- see ``AXIAL_GENERAL_FORM``): the helicity
    letters 'M'/'P' are the PREFERRED (PIN) stereodescriptors for an axial
    entity, so they are what this returns by default. 'Ra'/'Sa' are recommended
    for GENERAL nomenclature only and are produced with ``style="general"``.

    Args:
        mol: RDKit Mol object
        style: 'pin' (default) -> 'M'/'P'; 'general' -> 'Sa'/'Ra'.

    Returns:
        List of dicts with keys:
        - type: 'allene' or 'atropisomer'
        - idx: atom index (allene) or bond index (atropisomer)
        - cip: 'M'/'P' ('Sa'/'Ra' when style='general'), or None if undetermined
        - locant_atom: atom index to use for IUPAC locant mapping
    """
    def _styled(code):
        if code is None:
            return None
        return AXIAL_GENERAL_FORM.get(code, code) if style == "general" else code

    # Ensure CIP labels are assigned (needed for atropisomer P/M)
    assign_stereochemistry(mol)

    results = []

    # 1. Atropisomers: check bonds for STEREOATROPCW/STEREOATROPCCW
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOATROPCW,
                      Chem.BondStereo.STEREOATROPCCW):
            # RDKit already reports the helicity letter for an atropisomeric
            # bond, which IS the PIN descriptor (P-91.2.1.1(c),:44588) -- it is
            # passed through rather than re-lettered to Ra/Sa.
            cip = None
            if bond.HasProp('_CIPCode'):
                rdkit_cip = bond.GetProp('_CIPCode')
                if rdkit_cip in ('P', 'M'):
                    cip = _styled(rdkit_cip)
            results.append({
                'type': 'atropisomer',
                'idx': bond.GetIdx(),
                'cip': cip,
                'locant_atom': bond.GetBeginAtomIdx(),
            })

    # 2. Allenes: find atoms with CHI_ALLENE chiral tag
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_ALLENE:
            # rdCIPLabeler does not assign CIP to allene atoms in RDKit 2025.09
            # Use manual CIP determination
            cip = _styled(_manual_allene_cip(mol, atom.GetIdx()))
            results.append({
                'type': 'allene',
                'idx': atom.GetIdx(),
                'cip': cip,
                'locant_atom': atom.GetIdx(),
            })

    return results


def _cip_priority_key(mol_h, atom_idx: int, exclude_idx: int) -> tuple:
    """CIP priority key for allene terminal substituent.

    Uses IUPAC P-92.1.3 sequence rules (simplified):
    (1) Higher atomic number > lower
    (2) Sum of neighbor atomic numbers (excluding connection to terminal C)

    This handles the vast majority of allene cases without full CIP tree
    traversal.

    Args:
        mol_h: RDKit Mol with explicit H.
        atom_idx: Index of the substituent atom.
        exclude_idx: Index of the terminal carbon (excluded from neighbor sum).

    Returns:
        Tuple (atomic_number, neighbor_atomic_number_sum) for comparison.
    """
    atom = mol_h.GetAtomWithIdx(atom_idx)
    primary = atom.GetAtomicNum()
    secondary = sum(
        mol_h.GetAtomWithIdx(n.GetIdx()).GetAtomicNum()
        for n in atom.GetNeighbors()
        if n.GetIdx() != exclude_idx
    )
    return (primary, secondary)


def _manual_allene_cip(mol, central_idx: int) -> Optional[str]:
    """
    Determine the axial helicity descriptor M/P for an allene.

    For an allene C1=C=C2, view along the C=C=C axis. The allene is treated
    as an elongated tetrahedron with 4 substituents (2 on each terminal carbon).
    P-92.1.2.2 "The helicity rule: stereodescriptors 'M' and 'P'"
    (BlueBookV2.md:44812): *"Looking along the chirality axis the ligands are
    arranged in pairs. When proceeding from the nearer ligand having priority in
    the pair to the further away atom or group having priority in the pair, the
    chirality is described by the symbols 'M' if the path is anticlockwise; the
    symbol is 'P' if the path is clockwise."* So clockwise -> P, anticlockwise
    -> M; these are the PIN descriptors (P-91.2.1.1(c),:44588). The general
    forms Ra/Sa are derived by the caller via ``AXIAL_GENERAL_FORM``.

    Uses true CIP priority based on atomic number (primary) and neighbor
    atomic number sums (secondary), per IUPAC P-92.1.3.

    Args:
        mol: RDKit Mol object
        central_idx: Atom index of the central allene carbon

    Returns:
        'P', 'M', or None (if achiral -- fewer than 4 distinct groups)
    """
    central_atom = mol.GetAtomWithIdx(central_idx)

    # Get the two double-bond neighbors of the central allene C
    terminal_atoms = []
    for bond in central_atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            other_idx = bond.GetOtherAtomIdx(central_idx)
            terminal_atoms.append(other_idx)

    if len(terminal_atoms) != 2:
        return None

    # Add explicit Hs for priority analysis
    mol_h = Chem.AddHs(mol)

    term_a_idx, term_b_idx = terminal_atoms

    # Get substituents on each terminal atom (excluding the allene central C)
    def _get_terminal_substituents(atom_idx):
        """Get substituent atom indices and their CIP priority keys for a terminal atom."""
        atom = mol_h.GetAtomWithIdx(atom_idx)
        subs = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx != central_idx:
                priority = _cip_priority_key(mol_h, nbr_idx, atom_idx)
                subs.append((nbr_idx, priority))
        # Sort by priority descending (higher priority first)
        subs.sort(key=lambda x: x[1], reverse=True)
        return subs

    subs_a = _get_terminal_substituents(term_a_idx)
    subs_b = _get_terminal_substituents(term_b_idx)

    # Check achirality: if both substituents on a terminal have identical
    # CIP priority keys, the allene is achiral at that terminal.
    # Compares atomic number + neighbor atomic number sums -- not just
    # atomic number equality.
    def _terminal_is_achiral(subs_list):
        """Check if a terminal's substituents are chemically equivalent.

        Compares CIP priority keys (atomic number + neighbor atomic number
        sums) -- not just atomic number. Two carbons with different
        substituent trees (e.g., methyl vs ethyl) are NOT equivalent.
        """
        if len(subs_list) < 2:
            return False
        # Compare CIP priority keys -- if equal, substituents are equivalent
        # at the first two levels of the CIP tree
        return subs_list[0][1] == subs_list[1][1]

    if _terminal_is_achiral(subs_a) or _terminal_is_achiral(subs_b):
        return None  # Achiral allene

    # Now determine chirality using the elongated tetrahedron model.
    # The CHI_ALLENE tag encodes the enantiomer via the neighbor ordering
    # in the atom's neighbor list.

    # Priority ordering: for each terminal, get the high-priority sub
    # atom index and CIP priority key
    high_a = subs_a[0][0] if subs_a else None
    high_b = subs_b[0][0] if subs_b else None

    if high_a is None or high_b is None:
        return None

    priority_high_a = subs_a[0][1] if subs_a else None
    priority_high_b = subs_b[0][1] if subs_b else None

    if priority_high_a is None or priority_high_b is None:
        return None

    # Get the neighbor order as stored in the molecule for the central atom
    central_atom_h = mol_h.GetAtomWithIdx(central_idx)
    nbr_list = [n.GetIdx() for n in central_atom_h.GetNeighbors()]

    # The central allene C has exactly 2 neighbors (the two terminal Cs)
    if len(nbr_list) != 2:
        return None

    # The terminal atom that appears first in the central atom's neighbor
    # list defines the "near" end for chirality determination.
    first_terminal = nbr_list[0]

    if first_terminal == term_a_idx:
        near_high_priority = priority_high_a
        far_high_priority = priority_high_b
    else:
        near_high_priority = priority_high_b
        far_high_priority = priority_high_a

    # Determine sense based on the elongated tetrahedron model:
    # View along the allene axis from the near terminal to the far terminal.
    # P-92.1.2.2 (44812): clockwise -> 'P', anticlockwise -> 'M'.
    if near_high_priority > far_high_priority:
        return 'P'
    elif near_high_priority < far_high_priority:
        return 'M'
    else:
        return None  # Identical priorities -- achiral
