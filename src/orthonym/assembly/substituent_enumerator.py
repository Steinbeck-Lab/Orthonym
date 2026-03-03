"""Unified substituent enumeration module with ReplaceCore-based extraction.

Provides a single enumeration path for all substituents on both ring and chain
parent structures. Replaces the previously fragmented three-path system that
caused silent drops, double-counting, and wrong locants.

Architecture:
  - Ring parents: ReplaceCore(mol, core_from_ring_atoms) -> fragment mols
  - Chain parents: Branch-point enumeration from features.substituents dict
  - All fragments: classify -> name via existing naming infrastructure

Public API:
  - discover_substituents(mol, parent_atoms, parent_type, ...) [Phase 84]
  - extract_ring_substituents(mol, ring_atoms, oriented_ring)
  - extract_chain_substituents(mol, principal_chain, substituents_dict)
  - classify_and_name_fragment(mol, frag_info, parent_atoms, features=None)
  - collect_substituent_atom_set(substituent_infos)

References:
    IUPAC 2013 P-31.1 (detachable prefixes)
    IUPAC 2013 P-44 (parent selection determines what's a substituent)
"""

import logging
from collections import deque, namedtuple
from typing import List, Optional, Set, Dict

from rdkit import Chem
from rdkit.Chem import RWMol

from .naming_utils import get_alkyl_name
from .substituent_naming import name_substituent_fragment
from ..rules.seniority import get_prefix

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

SubstituentInfo = namedtuple(
    'SubstituentInfo',
    ['frag_mol', 'locant', 'attach_mol_idx', 'frag_atoms']
)
"""
Represents a single substituent on a parent structure.

Fields:
    frag_mol: RDKit Mol of the isolated fragment (with dummy atom at attachment),
              or None for chain-parent substituents where fragment is inline.
    locant: IUPAC locant (1-indexed integer) on the parent.
    attach_mol_idx: Original mol atom index of the attachment point on parent.
    frag_atoms: Set of original mol atom indices belonging to this substituent.
"""


# ============================================================================
# Universal Substituent Discovery (Phase 84)
# ============================================================================


def discover_substituents(
    mol,
    parent_atoms,
    parent_type="auto",
    oriented_ring=None,
    principal_chain=None,
    atom_to_locant=None,
):
    """Discover ALL substituents on a parent structure.

    Universal entry point that replaces six parallel substituent discovery
    systems. Every non-parent, non-hydrogen atom in mol is assigned to
    exactly one SubstituentInfo. No silent drops, no size limits.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.
        parent_type: ``"ring"``, ``"chain"``, or ``"auto"`` (auto-detects).
        oriented_ring: Ring atom indices in IUPAC order (required for ring parents).
        principal_chain: Chain atom indices in order (required for chain parents).
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.

    Returns:
        List[SubstituentInfo] with one entry per substituent fragment.
    """
    parent_set = set(parent_atoms)

    if parent_type == "auto":
        parent_type = _detect_parent_type(mol, parent_set)

    if parent_type == "ring":
        results = extract_ring_substituents(
            mol, tuple(parent_set), oriented_ring
        )
    else:
        results = _discover_chain_substituents(
            mol, parent_set, principal_chain, atom_to_locant
        )

    _verify_completeness(mol, parent_set, results)

    return results


def _detect_parent_type(mol, parent_atoms):
    """Auto-detect whether parent_atoms represent a ring or chain parent.

    Checks if any complete ring in the molecule is a subset of parent_atoms.
    If so, returns ``"ring"``; otherwise ``"chain"``.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.

    Returns:
        ``"ring"`` or ``"chain"``.
    """
    parent_set = set(parent_atoms)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if set(ring).issubset(parent_set):
            return "ring"
    return "chain"


def _discover_chain_substituents(mol, parent_atoms, principal_chain,
                                  atom_to_locant=None):
    """BFS-based substituent discovery for chain parents.

    For each atom on the principal chain, finds non-parent neighbors and
    BFS-collects complete substituent fragments. Walks through ALL atom
    types (no carbon-only restriction). No size limit.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices in the parent chain.
        principal_chain: List of atom indices in chain order.
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.

    Returns:
        List[SubstituentInfo] namedtuples.
    """
    results = []
    parent_set = set(parent_atoms)
    assigned = set()

    if principal_chain is None:
        principal_chain = sorted(parent_set)

    for chain_pos, chain_atom_idx in enumerate(principal_chain):
        chain_atom = mol.GetAtomWithIdx(chain_atom_idx)
        for nbr in chain_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in parent_set or nbr_idx in assigned:
                continue
            if nbr.GetAtomicNum() == 1:
                continue

            frag_atoms = _bfs_collect_fragment(
                mol, nbr_idx, parent_set, assigned
            )
            if not frag_atoms:
                continue
            assigned.update(frag_atoms)

            locant = chain_pos + 1
            if atom_to_locant and chain_atom_idx in atom_to_locant:
                locant = atom_to_locant[chain_atom_idx]

            results.append(SubstituentInfo(
                frag_mol=None,
                locant=locant,
                attach_mol_idx=chain_atom_idx,
                frag_atoms=frozenset(frag_atoms),
            ))

    return results


def _bfs_collect_fragment(mol, start_idx, parent_set, already_assigned):
    """BFS from start_idx, collecting all non-parent heavy atoms.

    CRITICAL: Does NOT stop at heteroatoms. Collects O, N, S, P and all
    atoms reachable through them. This ensures FG-containing substituents
    are discovered as compound fragments (USUB-04). No size limit.

    Args:
        mol: RDKit Mol object.
        start_idx: Atom index to start BFS from.
        parent_set: Set of parent atom indices (BFS boundary).
        already_assigned: Set of atom indices already claimed by another
            substituent.

    Returns:
        Set of atom indices in the collected fragment.
    """
    visited = set()
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        if idx in visited or idx in parent_set or idx in already_assigned:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:
            continue
        visited.add(idx)
        for nbr in atom.GetNeighbors():
            queue.append(nbr.GetIdx())
    return visited


def _verify_completeness(mol, parent_atoms, substituents):
    """Assert that all non-parent heavy atoms are accounted for.

    Every non-parent, non-hydrogen atom must be assigned to exactly one
    SubstituentInfo. Reports double-assigned and missed atoms via assert.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of parent atom indices.
        substituents: List of SubstituentInfo namedtuples.

    Raises:
        AssertionError: If atoms are double-assigned or missed.
    """
    parent_set = set(parent_atoms)
    all_sub_atoms = set()
    for sub in substituents:
        overlap = all_sub_atoms & set(sub.frag_atoms)
        assert not overlap, (
            f"Double-assigned atoms: {overlap}"
        )
        all_sub_atoms.update(sub.frag_atoms)

    expected = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in parent_set:
            expected.add(atom.GetIdx())

    missed = expected - all_sub_atoms
    assert not missed, f"Unassigned atoms: {missed}"
    extra = all_sub_atoms - expected
    assert not extra, f"Extra atoms not in molecule: {extra}"


# ============================================================================
# Universal Substituent Naming (Phase 85)
# ============================================================================


def name_substituent(mol, frag_atoms, attach_idx):
    """Name any substituent fragment. Never returns None.

    Five-tier naming cascade:
      1. Retained substituent names (isopropyl, phenyl, etc.) -- IUPAC preferred
      2. Static fragment cache (FRAGMENT_NAME_CACHE) -- O(1) lookup
      3. Linear alkyl fast path (chain_prefixes table)
      4. Recursive compound naming (name_substituent_fragment)
      5. Descriptive fallback (guaranteed non-None)

    Args:
        mol: RDKit Mol of the full molecule.
        frag_atoms: Set/list of atom indices belonging to the substituent.
        attach_idx: Atom index WITHIN frag_atoms that bonds to the parent.

    Returns:
        str: IUPAC prefix name (always non-None, always non-empty).

    References:
        IUPAC 2013 P-31.1 (detachable prefixes)
        Phase 85 design: five-tier cascade with guaranteed fallback
    """
    from .fragment_naming import FRAGMENT_NAME_CACHE
    from .substituent_naming import (
        parent_to_prefix,
        _check_retained_substituent,
        _is_linear_alkyl,
    )

    frag_atoms_set = set(frag_atoms)

    # Edge case: empty fragment
    if not frag_atoms_set:
        return "substituent"

    # ---- Tier 1: Retained substituent names ----
    # Checked first per IUPAC: retained names (phenyl, isopropyl, etc.)
    # are the preferred forms and must take priority over cache-derived
    # parent-to-prefix conversions (e.g., "phenyl" not "benzenyl").
    try:
        retained = _check_retained_substituent(
            mol, list(frag_atoms_set), attach_idx
        )
        if retained:
            return retained
    except Exception:
        pass

    # ---- Tier 2: Static fragment cache (O(1)) ----
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, list(frag_atoms_set))
        if frag_smiles:
            canonical = Chem.CanonSmiles(frag_smiles)
            if canonical:
                cached = FRAGMENT_NAME_CACHE.get(canonical)
                if cached:
                    # Convert parent name to prefix form
                    carbon_count = sum(
                        1 for i in frag_atoms_set
                        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                    )
                    prefix = parent_to_prefix(cached, chain_length=carbon_count)
                    if prefix:
                        return prefix
    except Exception:
        pass  # Cache miss is fine, continue to next tier

    # ---- Tier 3: Linear alkyl fast path ----
    try:
        if _is_linear_alkyl(mol, list(frag_atoms_set)):
            carbon_count = sum(
                1 for i in frag_atoms_set
                if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            )
            if carbon_count > 0:
                return get_alkyl_name(carbon_count)
    except Exception:
        pass

    # ---- Tier 4: Recursive compound naming ----
    try:
        result = name_substituent_fragment(
            mol, list(frag_atoms_set), attach_idx, []
        )
        if result and "unknown" not in result.lower():
            return result
    except Exception:
        pass

    # ---- Tier 5: Descriptive fallback (guaranteed non-None) ----
    return _descriptive_fallback(mol, frag_atoms_set, attach_idx)


def _descriptive_fallback(mol, frag_atoms, attach_idx):
    """Produce a compositional description for unnameable fragments.

    Analyzes fragment atoms directly to build a best-effort prefix name.
    For simple fragments (1-3 atoms), produces specific names like
    "hydroxy", "amino", "methyl". For complex unnameable fragments,
    returns "substituent" as absolute last resort.

    Args:
        mol: RDKit Mol object.
        frag_atoms: Set of atom indices in the fragment.
        attach_idx: Attachment atom index.

    Returns:
        str: Always non-None, always non-empty.
    """
    if not frag_atoms:
        return "substituent"

    # Analyze fragment composition
    carbons = 0
    heteroatoms = {}
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        anum = atom.GetAtomicNum()
        if anum == 1:
            continue
        if anum == 6:
            carbons += 1
        else:
            sym = atom.GetSymbol()
            heteroatoms[sym] = heteroatoms.get(sym, 0) + 1

    # Single-atom fragments
    if len(frag_atoms) == 1:
        idx = next(iter(frag_atoms))
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        total_hs = atom.GetTotalNumHs()

        # Halogens
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]
        # Oxygen
        if sym == 'O':
            return 'hydroxy' if total_hs >= 1 else 'oxo'
        # Nitrogen
        if sym == 'N':
            if total_hs >= 2:
                return 'amino'
            elif total_hs == 1:
                return 'imino'
            else:
                return 'azanyl'
        # Sulfur
        if sym == 'S':
            return 'sulfanyl' if total_hs >= 1 else 'sulfanylidene'
        # Single carbon
        if sym == 'C':
            return 'methyl'

    # Carbon-only fragments: use alkyl names
    if carbons > 0 and not heteroatoms:
        try:
            return get_alkyl_name(carbons)
        except (ValueError, KeyError):
            pass

    # Multi-atom heteroatom-only fragments
    if carbons == 0 and heteroatoms:
        symbols = sorted(heteroatoms.keys())
        # -NO2 (nitro)
        if symbols == ['N', 'O'] and heteroatoms.get('N', 0) == 1 and heteroatoms.get('O', 0) == 2:
            return 'nitro'
        # -N3 (azido)
        if symbols == ['N'] and heteroatoms.get('N', 0) == 3:
            return 'azido'
        # Single heteroatom type
        if len(symbols) == 1:
            sym = symbols[0]
            if sym == 'O':
                return 'hydroxy'
            if sym == 'N':
                return 'amino'
            if sym == 'S':
                return 'sulfanyl'
            if sym in _HALOGEN_MAP:
                return _HALOGEN_MAP[sym]

    # Absolute last resort
    return "substituent"


# ============================================================================
# Halogen Name Map (for fg_only classification)
# ============================================================================

_HALOGEN_MAP = {
    'F': 'fluoro',
    'Cl': 'chloro',
    'Br': 'bromo',
    'I': 'iodo',
}


# ============================================================================
# Ring Substituent Extraction (ReplaceCore-based)
# ============================================================================


def extract_ring_substituents(mol, ring_atoms, oriented_ring):
    """Extract all substituent fragments from a ring parent using ReplaceCore.

    Builds a core mol from ring_atoms, calls ReplaceCore to extract all
    non-ring fragments as separate mol objects with isotope-labeled dummy
    atoms indicating attachment points.

    Args:
        mol: RDKit Mol object.
        ring_atoms: Tuple or list of ring atom indices (from principal_ring).
        oriented_ring: List of ring atom indices in IUPAC numbering order.

    Returns:
        List of SubstituentInfo namedtuples, one per substituent fragment.
        Empty list if ring has no substituents.
    """
    ring_set = set(ring_atoms)

    # Build core mol from ring atoms
    core = RWMol()
    idx_map = {}  # original mol idx -> core mol idx
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = core.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        # Preserve aromaticity for correct matching
        core.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
        idx_map[atom_idx] = new_idx

    # Copy bonds between ring atoms
    added_bonds = set()
    for atom_idx in ring_atoms:
        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in ring_set and end in ring_set:
                bond_key = (min(begin, end), max(begin, end))
                if bond_key not in added_bonds:
                    core.AddBond(
                        idx_map[begin], idx_map[end], bond.GetBondType()
                    )
                    added_bonds.add(bond_key)

    core_mol = core.GetMol()

    # Match tuple maps core atom index -> original mol atom index
    match = tuple(ring_atoms)

    # ReplaceCore: removes core, returns fragments with isotope-labeled dummies
    frags = Chem.ReplaceCore(mol, core_mol, match, labelByIndex=True)
    if frags is None:
        return []

    # Split into individual fragment mols
    subfgs = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
    if not subfgs:
        return []

    # Pre-compute per-branch atom sets for geminal substituent disambiguation
    # Key: ring_atom_idx -> list of frozensets (one per separate branch from that atom)
    branch_map = _compute_branch_map(mol, ring_set)

    # Track which branches have been claimed (for geminal disambiguation)
    claimed_branches = set()  # set of (attach_idx, branch_id) tuples

    results = []
    for frag in subfgs:
        # Find the dummy atom(s) to determine attachment point
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atom
                # CRITICAL: isotope 0 is valid (maps to match position 0)
                core_pos = atom.GetIsotope()
                if core_pos < len(match):
                    mol_atom_idx = match[core_pos]

                    # Map to IUPAC locant via oriented_ring
                    locant = _get_locant_from_oriented_ring(
                        mol_atom_idx, oriented_ring
                    )

                    # Collect original mol atom indices for this fragment
                    # Use per-branch disambiguation for geminal substituents
                    frag_atoms = _collect_frag_atoms_for_fragment(
                        mol, ring_set, mol_atom_idx, frag,
                        branch_map, claimed_branches
                    )

                    if locant is not None:
                        results.append(SubstituentInfo(
                            frag_mol=frag,
                            locant=locant,
                            attach_mol_idx=mol_atom_idx,
                            frag_atoms=frag_atoms,
                        ))
                break  # only process first dummy atom per fragment

    return results


def _get_locant_from_oriented_ring(mol_atom_idx, oriented_ring):
    """Map a mol atom index to its IUPAC locant via oriented_ring.

    Args:
        mol_atom_idx: Atom index in the original mol.
        oriented_ring: List of atom indices in IUPAC numbering order.

    Returns:
        1-indexed IUPAC locant, or None if not found.
    """
    for pos, ring_atom in enumerate(oriented_ring):
        if ring_atom == mol_atom_idx:
            return pos + 1
    return None


def _compute_branch_map(mol, ring_set):
    """Compute per-branch atom sets for each ring atom.

    For geminal substituents (two substituents on the same ring atom), each
    separate branch needs its own atom set. This function BFS-es from each
    individual non-ring neighbor of each ring atom to produce separate sets.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.

    Returns:
        Dict mapping ring_atom_idx -> list of frozensets (one per branch).
    """
    branch_map = {}
    for ring_idx in ring_set:
        branches = []
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in ring_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set:
                continue
            # BFS from this specific neighbor to collect its branch
            visited = set()
            stack = [nbr_idx]
            while stack:
                idx = stack.pop()
                if idx in visited or idx in ring_set:
                    continue
                visited.add(idx)
                for nbr2 in mol.GetAtomWithIdx(idx).GetNeighbors():
                    nbr2_idx = nbr2.GetIdx()
                    if nbr2_idx not in visited and nbr2_idx not in ring_set:
                        stack.append(nbr2_idx)
            if visited:
                branches.append(frozenset(visited))
        if branches:
            branch_map[ring_idx] = branches
    return branch_map


def _collect_frag_atoms_for_fragment(mol, ring_set, attach_ring_idx, frag_mol,
                                      branch_map, claimed_branches):
    """Collect original mol atom indices for a specific fragment.

    Handles geminal substituents by matching frag_mol composition against
    individual branches from branch_map, and tracking which branches have
    already been claimed.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.
        attach_ring_idx: Ring atom index the fragment is attached to.
        frag_mol: RDKit Mol of the isolated fragment.
        branch_map: Dict from _compute_branch_map.
        claimed_branches: Mutable set of (attach_idx, branch_idx) tuples already used.

    Returns:
        Frozenset of original mol atom indices.
    """
    branches = branch_map.get(attach_ring_idx, [])

    if len(branches) <= 1:
        # Single substituent at this position -- use all non-ring neighbors
        return branches[0] if branches else frozenset()

    # Multiple branches (geminal): match fragment composition to find the right one
    # Count non-dummy heavy atoms in frag_mol
    frag_heavy = _count_frag_heavy_atoms(frag_mol)

    for branch_idx, branch_atoms in enumerate(branches):
        key = (attach_ring_idx, branch_idx)
        if key in claimed_branches:
            continue

        # Count heavy atoms in this branch from original mol
        branch_heavy = {}
        for idx in branch_atoms:
            atom = mol.GetAtomWithIdx(idx)
            anum = atom.GetAtomicNum()
            if anum != 1:  # skip hydrogen
                sym = atom.GetSymbol()
                branch_heavy[sym] = branch_heavy.get(sym, 0) + 1

        if branch_heavy == frag_heavy:
            claimed_branches.add(key)
            return branch_atoms

    # Fallback: if no exact match, claim the first unclaimed branch
    for branch_idx, branch_atoms in enumerate(branches):
        key = (attach_ring_idx, branch_idx)
        if key not in claimed_branches:
            claimed_branches.add(key)
            return branch_atoms

    # Last resort: return union of all branches (shouldn't happen)
    all_atoms = set()
    for b in branches:
        all_atoms.update(b)
    return frozenset(all_atoms)


def _count_frag_heavy_atoms(frag_mol):
    """Count non-dummy, non-H atoms in a fragment mol by element symbol.

    Returns:
        Dict of {symbol: count} for heavy atoms in the fragment.
    """
    counts = {}
    if frag_mol is None:
        return counts
    for atom in frag_mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum == 0 or anum == 1:
            continue
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _collect_frag_original_atoms(mol, ring_set, attach_ring_idx):
    """BFS from a ring atom to collect all non-ring substituent atoms.

    Note: This collects ALL substituent atoms from the ring atom. For geminal
    substituents, use _collect_frag_atoms_for_fragment instead.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.
        attach_ring_idx: Ring atom index that the substituent is attached to.

    Returns:
        Frozenset of original mol atom indices belonging to the substituent.
    """
    visited = set()
    stack = []

    # Start from neighbors of the ring atom that are NOT in the ring
    ring_atom = mol.GetAtomWithIdx(attach_ring_idx)
    for nbr in ring_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx not in ring_set:
            stack.append(nbr_idx)

    while stack:
        idx = stack.pop()
        if idx in visited or idx in ring_set:
            continue
        visited.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_set:
                stack.append(nbr_idx)

    return frozenset(visited)


# ============================================================================
# Chain Substituent Extraction
# ============================================================================


def extract_chain_substituents(mol, principal_chain, substituents_dict):
    """Convert a substituents dict to a list of SubstituentInfo namedtuples.

    Takes the existing features.substituents dict (position -> list of
    atom index lists) and wraps each entry as a SubstituentInfo for
    unified downstream processing.

    Args:
        mol: RDKit Mol object.
        principal_chain: List of atom indices in the principal chain.
        substituents_dict: Dict mapping chain position (1-indexed) to list
            of substituent atom index lists.

    Returns:
        List of SubstituentInfo namedtuples.
    """
    if not substituents_dict:
        return []

    chain_set = set(principal_chain)
    results = []

    for position, sub_atom_lists in substituents_dict.items():
        for sub_atoms in sub_atom_lists:
            if not sub_atoms:
                continue

            # The first atom in sub_atoms is the one bonded to the chain
            attach_idx = sub_atoms[0] if isinstance(sub_atoms, (list, tuple)) else sub_atoms

            # Compute the actual atom index on the chain that this attaches to
            # position is 1-indexed into principal_chain
            if isinstance(position, int) and 1 <= position <= len(principal_chain):
                chain_atom_idx = principal_chain[position - 1]
            else:
                chain_atom_idx = attach_idx

            frag_atoms = frozenset(sub_atoms) if isinstance(sub_atoms, (list, tuple)) else frozenset([sub_atoms])

            results.append(SubstituentInfo(
                frag_mol=None,  # Chain substituents use atom indices, not frag mols
                locant=position,
                attach_mol_idx=chain_atom_idx,
                frag_atoms=frag_atoms,
            ))

    return results


# ============================================================================
# Fragment Classification and Naming
# ============================================================================


def classify_and_name_fragment(mol, frag_info, parent_atoms, features=None):
    """Classify a substituent fragment and produce its IUPAC prefix name.

    Routes each fragment to the appropriate naming function based on its
    composition:
      - fg_only: No carbon atoms (halogens, -OH, -NH2, -NO2, etc.)
      - pure_alkyl: Only carbon atoms (methyl, ethyl, etc.)
      - compound: Carbon + heteroatoms (trifluoromethyl, hydroxymethyl, etc.)

    Args:
        mol: RDKit Mol object of the full molecule.
        frag_info: SubstituentInfo namedtuple for this substituent.
        parent_atoms: Set of atom indices in the parent structure.
        features: Optional features object (for additional context).

    Returns:
        IUPAC prefix name string (e.g., "methyl", "hydroxy", "trifluoromethyl"),
        or None if naming fails (with WARNING logged).
    """
    frag_mol = frag_info.frag_mol
    frag_atoms = frag_info.frag_atoms

    # Classify the fragment by composition
    category = _classify_fragment(mol, frag_mol, frag_atoms)

    if category == 'fg_only':
        return _name_fg_only(mol, frag_mol, frag_atoms)
    elif category == 'pure_alkyl':
        return _name_pure_alkyl(mol, frag_info, parent_atoms)
    elif category == 'compound':
        return _name_compound_substituent(mol, frag_info, parent_atoms)
    else:
        # Unknown -- log warning, never silently drop
        _frag_smiles = _get_frag_smiles(mol, frag_atoms)
        logger.warning(
            "Unrecognized substituent fragment at locant %s: %s",
            frag_info.locant, _frag_smiles
        )
        return None


def _classify_fragment(mol, frag_mol, frag_atoms):
    """Classify a fragment as fg_only, pure_alkyl, compound, or unknown.

    Uses the fragment mol if available (ring parent), otherwise uses
    original mol atom indices.

    Args:
        mol: RDKit Mol of the full molecule.
        frag_mol: RDKit Mol of the isolated fragment (may be None).
        frag_atoms: Set/frozenset of original mol atom indices.

    Returns:
        String category: 'fg_only', 'pure_alkyl', 'compound', or 'unknown'.
    """
    carbons = 0
    heteroatoms = 0

    if frag_mol is not None:
        # Use fragment mol (from ReplaceCore)
        for atom in frag_mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum == 0:
                continue  # skip dummy atom
            if anum == 6:
                carbons += 1
            elif anum != 1:
                heteroatoms += 1
    else:
        # Use original mol indices
        for idx in frag_atoms:
            atom = mol.GetAtomWithIdx(idx)
            anum = atom.GetAtomicNum()
            if anum == 6:
                carbons += 1
            elif anum != 1:
                heteroatoms += 1

    if carbons == 0 and heteroatoms > 0:
        return 'fg_only'
    elif carbons > 0 and heteroatoms == 0:
        return 'pure_alkyl'
    elif carbons > 0 and heteroatoms > 0:
        return 'compound'
    else:
        return 'unknown'


def _name_fg_only(mol, frag_mol, frag_atoms):
    """Name a fragment that is a pure functional group (no carbon).

    Checks halogens by symbol, then common FG patterns.

    Args:
        mol: RDKit Mol of full molecule.
        frag_mol: RDKit Mol of isolated fragment (may be None).
        frag_atoms: Set of original mol atom indices.

    Returns:
        IUPAC prefix name string, or None if unrecognized.
    """
    # Collect non-dummy, non-hydrogen atoms
    if frag_mol is not None:
        atoms_info = []
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue  # skip dummy
            atoms_info.append({
                'symbol': atom.GetSymbol(),
                'atomic_num': atom.GetAtomicNum(),
                'total_hs': atom.GetTotalNumHs(),
                'num_bonds': atom.GetDegree(),
            })
    else:
        atoms_info = []
        for idx in frag_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:
                continue
            atoms_info.append({
                'symbol': atom.GetSymbol(),
                'atomic_num': atom.GetAtomicNum(),
                'total_hs': atom.GetTotalNumHs(),
                'num_bonds': atom.GetDegree(),
            })

    if len(atoms_info) == 0:
        return None

    # Single atom cases
    if len(atoms_info) == 1:
        ai = atoms_info[0]
        sym = ai['symbol']

        # Halogens
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]

        # -OH (oxygen with 1 H)
        if sym == 'O' and ai['total_hs'] >= 1:
            return 'hydroxy'

        # =O (oxo -- oxygen with no H, double bonded)
        if sym == 'O' and ai['total_hs'] == 0:
            return 'oxo'

        # -NH2 (nitrogen with 2 H)
        if sym == 'N' and ai['total_hs'] >= 2:
            return 'amino'

        # =NH (imino -- nitrogen with 1 H)
        if sym == 'N' and ai['total_hs'] == 1:
            return 'imino'

        # -SH (sulfanyl)
        if sym == 'S' and ai['total_hs'] >= 1:
            return 'sulfanyl'

        # =S (sulfanylidene)
        if sym == 'S' and ai['total_hs'] == 0:
            return 'sulfanylidene'

    # Multi-atom FG-only patterns
    if len(atoms_info) >= 2:
        symbols = sorted(ai['symbol'] for ai in atoms_info)

        # -NO2 (nitro): N + 2O
        if symbols == ['N', 'O', 'O']:
            return 'nitro'

        # -N3 (azido): 3 N atoms
        if symbols == ['N', 'N', 'N']:
            return 'azido'

    # Try seniority.get_prefix as fallback
    # For FG-only fragments on ring parents, try SMARTS matching
    if frag_mol is not None:
        from ..perception.functional_groups import FUNCTIONAL_GROUP_SMARTS
        for fg_name, smarts_str in FUNCTIONAL_GROUP_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts_str)
            if pattern and frag_mol.HasSubstructMatch(pattern):
                prefix = get_prefix(fg_name)
                if prefix:
                    return prefix

    # Last resort for single-atom halogens on original mol
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]
        if sym == 'O' and atom.GetTotalNumHs() >= 1:
            return 'hydroxy'
        if sym == 'O' and atom.GetTotalNumHs() == 0:
            return 'oxo'
        if sym == 'N' and atom.GetTotalNumHs() >= 2:
            return 'amino'
        if sym == 'S' and atom.GetTotalNumHs() >= 1:
            return 'sulfanyl'

    logger.warning(
        "Could not name fg_only fragment with atoms: %s",
        [ai['symbol'] for ai in atoms_info]
    )
    return None


def _name_pure_alkyl(mol, frag_info, parent_atoms):
    """Name a pure alkyl substituent (carbon-only).

    Uses name_substituent_fragment for retained names (isopropyl, etc.)
    and recursive naming, with get_alkyl_name as fallback.

    Args:
        mol: RDKit Mol of full molecule.
        frag_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices.

    Returns:
        Alkyl prefix name string, or None.
    """
    frag_atoms = list(frag_info.frag_atoms)
    if not frag_atoms:
        return None

    # Find the attachment atom within the fragment
    attach_idx = _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms)
    if attach_idx is None and frag_atoms:
        attach_idx = frag_atoms[0]

    # Delegate to existing naming infrastructure
    parent_list = list(parent_atoms) if parent_atoms else []
    name = name_substituent_fragment(mol, frag_atoms, attach_idx, parent_list)
    if name:
        return name

    # Fallback: count carbons
    carbon_count = sum(
        1 for idx in frag_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )
    if carbon_count > 0:
        try:
            return get_alkyl_name(carbon_count)
        except (ValueError, KeyError):
            pass

    return None


def _name_compound_substituent(mol, frag_info, parent_atoms):
    """Name a compound substituent (carbon + heteroatoms).

    Handles haloalkyl (trifluoromethyl), hydroxyalkyl, aminoalkyl,
    alkoxy, sulfanylalkyl, and other compound types.

    Args:
        mol: RDKit Mol of full molecule.
        frag_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices.

    Returns:
        Compound prefix name string, or None.
    """
    frag_atoms = list(frag_info.frag_atoms)
    if not frag_atoms:
        return None

    # Find the attachment atom
    attach_idx = _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms)
    if attach_idx is None and frag_atoms:
        attach_idx = frag_atoms[0]

    # Special case: S-attached branches (thioether substituents)
    # -S-R -> "alkylsulfanyl" (e.g., methylsulfanyl, ethylsulfanyl)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'S' and attach_atom.GetTotalNumHs() == 0:
            name = _name_sulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Delegate to existing naming infrastructure
    parent_list = list(parent_atoms) if parent_atoms else []
    name = name_substituent_fragment(mol, frag_atoms, attach_idx, parent_list)
    if name:
        # Reject garbled names from S/P-attached branches where
        # name_substituent_fragment doesn't handle the root heteroatom
        if attach_idx is not None:
            attach_sym = mol.GetAtomWithIdx(attach_idx).GetSymbol()
            if attach_sym in ('S', 'P') and 'thiyl' in name:
                pass  # fall through to warning
            else:
                return name

    # If naming infrastructure couldn't handle it, log warning
    frag_smiles = _get_frag_smiles(mol, frag_atoms)
    logger.warning(
        "Could not name compound substituent at locant %s: %s",
        frag_info.locant, frag_smiles
    )
    return None


def _name_sulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name a sulfanyl-attached branch: -S-R -> alkylsulfanyl.

    For simple -S-alkyl branches, produces IUPAC substitutive prefix names:
    -S-CH3 -> methylsulfanyl
    -S-C2H5 -> ethylsulfanyl

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the S attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Sulfanyl prefix name, or None if not a simple case.
    """
    frag_set = set(frag_atoms)
    s_atom = mol.GetAtomWithIdx(attach_idx)

    # Collect carbon atoms bonded to S (excluding parent)
    alkyl_atoms = []
    for nbr in s_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set and nbr.GetSymbol() == 'C':
            # BFS from this C to collect all connected carbons in fragment
            visited = set()
            stack = [nbr_idx]
            while stack:
                idx = stack.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    visited.add(idx)
                    for n in atom.GetNeighbors():
                        if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                            stack.append(n.GetIdx())
            alkyl_atoms.extend(visited)

    if not alkyl_atoms:
        return None

    # Check for non-C non-H atoms in the alkyl portion
    has_hetero_in_alkyl = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in alkyl_atoms
    )
    if has_hetero_in_alkyl:
        return None  # Complex case, defer

    carbon_count = len(alkyl_atoms)
    if carbon_count == 0:
        return None

    try:
        alkyl_name = get_alkyl_name(carbon_count)
        return f"{alkyl_name}sulfanyl"
    except (ValueError, KeyError):
        return None


def _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms):
    """Find the fragment atom that is bonded to the parent structure.

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        parent_atoms: Set of atom indices in the parent.

    Returns:
        Atom index of the attachment atom, or None.
    """
    frag_set = set(frag_atoms)
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in parent_atoms:
                return idx
    return None


def _get_frag_smiles(mol, frag_atoms):
    """Get SMILES for a fragment defined by atom indices.

    Args:
        mol: RDKit Mol.
        frag_atoms: Collection of atom indices.

    Returns:
        SMILES string, or "unknown" if extraction fails.
    """
    try:
        atoms = list(frag_atoms)
        if atoms:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms)
            return smi if smi else "unknown"
    except Exception:
        pass
    return "unknown"


# ============================================================================
# Atom Set Collection (for deduplication)
# ============================================================================


def collect_substituent_atom_set(substituent_infos):
    """Collect the union of all substituent atom indices.

    Used by callers to prevent double-counting: FGs whose atoms are
    entirely within a named branch should be skipped in standalone
    FG prefix generation.

    Args:
        substituent_infos: List of SubstituentInfo namedtuples.

    Returns:
        Frozenset of all original mol atom indices covered by substituents.
    """
    all_atoms = set()
    for info in substituent_infos:
        if info.frag_atoms:
            all_atoms.update(info.frag_atoms)
    return frozenset(all_atoms)
