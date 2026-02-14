"""Centralized substituent fragment naming module.

Provides name_substituent_fragment() -- the single entry point for naming
any substituent (linear, branched, functionalized, or ring-containing) from
its atom indices within a parent molecule.

Architecture:
  1. Fast path: linear alkyl substituents use get_alkyl_name() directly.
  2. Retained names: isopropyl, tert-butyl, sec-butyl, isobutyl, phenyl.
  3. Recursive path: extract fragment SMILES, name via name_fragment_recursively(),
     convert parent name to prefix form via parent_to_prefix().

The function returns RAW prefix names WITHOUT enclosing marks (parentheses/brackets).
The caller (format_substituent_prefix in naming_utils.py) handles wrapping based on
is_complex_substituent() and multiplier logic.

References:
    IUPAC 2013 P-31.1.3 (substituent prefix naming)
    IUPAC 2013 P-14.5.2 (compound substituent enclosing marks)
"""

import re
import logging
from collections import deque
from typing import List, Optional, Set

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from .naming_utils import get_alkyl_name
from .fragment_naming import name_fragment_recursively

logger = logging.getLogger(__name__)


# ============================================================================
# Linear Alkyl Detection (Fast Path Guard)
# ============================================================================


def _is_linear_alkyl(mol, sub_atoms: List[int]) -> bool:
    """Check if a substituent is a straight-chain pure alkyl group.

    Returns True if ALL atoms in sub_atoms are carbon AND no carbon has
    more than 2 carbon neighbors within sub_atoms (i.e., no branching).

    This is the fast-path guard: if True, use get_alkyl_name(carbon_count)
    directly, avoiding unnecessary recursion.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.

    Returns:
        True if the substituent is a linear (unbranched) pure-carbon chain.
    """
    if not sub_atoms:
        return False

    sub_set = set(sub_atoms)

    # Reject if any atom is in a ring (cyclic substituents are not linear alkyl)
    ring_info = mol.GetRingInfo()
    for idx in sub_atoms:
        if ring_info.NumAtomRings(idx) > 0:
            return False

    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Any non-carbon atom means it's not pure alkyl
        if atom.GetSymbol() != 'C':
            return False

        # Count carbon neighbors within the substituent
        c_nbrs_in_sub = sum(
            1 for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in sub_set and nbr.GetSymbol() == 'C'
        )

        # A linear chain carbon has at most 2 C neighbors within the fragment
        # (or 1 at the terminal). More than 2 means branching.
        if c_nbrs_in_sub > 2:
            return False

    return True


# ============================================================================
# Fragment SMILES Extraction
# ============================================================================


def _extract_fragment_smiles(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_chain: Set[int],
) -> Optional[str]:
    """Extract a valid SMILES for a substituent fragment.

    Uses RDKit's MolFragmentToSmiles to extract the substituent as a
    standalone molecule. The attachment point atom gets its valence
    satisfied implicitly (MolFragmentToSmiles caps the cut bond with H).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: Index of the first atom of the substituent (bonded to parent).
        parent_chain: Set of atom indices in the parent chain.

    Returns:
        SMILES string of the fragment, or None if extraction fails.
    """
    if not sub_atoms:
        return None

    try:
        frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sub_atoms)
        if not frag_smi:
            return None

        # Validate: the fragment SMILES must parse back to a valid molecule
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            return None

        return frag_smi
    except Exception:
        return None


# ============================================================================
# Parent-to-Prefix Conversion (IUPAC P-31.1.3)
# ============================================================================


def parent_to_prefix(parent_name: str, chain_length: int, attach_locant: int = 1) -> str:
    """Convert a parent compound name to substituent prefix form.

    Per IUPAC P-31.1.3, the parent compound name is transformed into
    a substituent prefix by:
    1. Removing the suffix (e.g., -oic acid, -ol, -one)
    2. Converting the suffix to its prefix form (e.g., -ol -> hydroxy)
    3. Adding the prefix at the correct locant
    4. Appending -yl at the free-valence position

    Args:
        parent_name: Parent compound IUPAC name (e.g., "propan-2-ol").
        chain_length: Number of carbons in the substituent chain.
        attach_locant: Locant of the free-valence carbon (default 1).

    Returns:
        Prefix-form name (e.g., "2-hydroxypropyl"). Returns raw name
        WITHOUT enclosing marks.
    """
    if not parent_name:
        return ""

    name = parent_name.strip()

    # ---- Carboxylic acids: -oic acid / -anoic acid ----
    # e.g., "butanoic acid" -> "3-carboxypropyl"
    m_oic = re.match(r'^(.+?)(?:an)?oic acid$', name)
    if m_oic:
        stem = m_oic.group(1)
        # Carboxy goes on the terminal carbon (chain_length for original chain)
        # The stem is shortened by one carbon (the COOH carbon becomes "carboxy")
        carboxy_locant = chain_length
        # Build: {locant}-carboxy{shortened_stem}yl
        # For butanoic acid (4C): carboxy at C4, stem = prop (3C), result = 3-carboxypropyl
        # Actually: butanoic acid -> carboxy replaces the acid, chain becomes 3C (propyl)
        # The parent chain of the substituent WITHOUT the carboxylic acid is chain_length - 1
        shortened_length = chain_length - 1
        if shortened_length >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened_length)
            # carboxy locant is the shortened chain length (farthest from attachment)
            return f"{shortened_length}-carboxy{short_stem}yl"
        else:
            return "carboxy"

    # ---- Locanted alcohol: -N-ol ----
    # Matches saturated (-an-N-ol), unsaturated (-en-N-ol, -yn-N-ol),
    # and bare (-N-ol) patterns.
    # e.g., "propan-2-ol" -> "2-hydroxypropyl"
    # e.g., "3-methylbut-2-en-1-ol" -> "1-hydroxy-3-methylbut-2-en-1-yl"
    m_ol = re.search(r'-(\d+)-ol$', name)
    if m_ol:
        locant = m_ol.group(1)
        # Get the stem (everything before "-N-ol")
        stem = name[:m_ol.start()]
        # Convert saturated suffix to yl: -an -> -yl (propan -> propyl)
        # Keep unsaturation: -en stays as -en, -yn stays as -yn
        if stem.endswith('an'):
            stem = stem[:-2]  # propan -> prop
        # Insert hyphen between hydroxy and stem when stem starts with a
        # digit (e.g., "3-methylbut-2-en") to avoid "hydroxy3-methylbut"
        sep = "-" if stem and stem[0].isdigit() else ""
        return f"{locant}-hydroxy{sep}{stem}yl"

    # ---- Unlocanted alcohol: ends in -ol (e.g., "ethanol", "methanol") ----
    if name.endswith('ol') and not name.endswith('diol'):
        # Strip -ol, check for -an prefix
        base = name[:-2]  # remove "ol"
        if base.endswith('an'):
            stem = base[:-2]  # remove "an"
        elif base.endswith('a'):
            # e.g., "methan" case (methanol -> methan -> meth)
            stem = base[:-1]
        else:
            stem = base
        # Alcohol: add hydroxy prefix
        return f"hydroxy{stem}yl"

    # ---- Locanted ketone: -an-N-one ----
    # e.g., "butan-2-one" -> "2-oxobutyl"
    m_one = re.search(r'an?-(\d+)-one$', name)
    if m_one:
        locant = m_one.group(1)
        stem_end = m_one.start()
        stem = name[:stem_end].rstrip('-')
        return f"{locant}-oxo{stem}yl"

    # ---- Unlocanted ketone: ends in -one ----
    if name.endswith('one') and not name.endswith('none'):
        base = name[:-3]  # remove "one"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return f"oxo{stem}yl"

    # ---- Locanted amine: -an-N-amine ----
    # e.g., "propan-1-amine" -> "1-aminopropyl"
    m_amine = re.search(r'an?-(\d+)-amine$', name)
    if m_amine:
        locant = m_amine.group(1)
        stem_end = m_amine.start()
        stem = name[:stem_end].rstrip('-')
        return f"{locant}-amino{stem}yl"

    # ---- Unlocanted amine: ends in -amine / -anamine ----
    if name.endswith('amine'):
        base = name[:-5]  # remove "amine"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return f"amino{stem}yl"

    # ---- Aldehyde: ends in -al or -anal ----
    # e.g., "propanal" -> "1-oxopropyl"
    if name.endswith('al') and not name.endswith('nal') or name.endswith('anal'):
        if name.endswith('anal'):
            stem = name[:-4]  # remove "anal"
        elif name.endswith('al'):
            base = name[:-2]  # remove "al"
            if base.endswith('an'):
                stem = base[:-2]
            elif base.endswith('a'):
                stem = base[:-1]
            else:
                stem = base
        # Aldehyde: oxo at C-1
        return f"1-oxo{stem}yl"

    # ---- Alkane: -ane or -e ending ----
    # e.g., "propane" -> "propyl", "2-methylpropane" -> "2-methylpropyl"
    if name.endswith('ane'):
        stem = name[:-3]  # remove "ane"
        return f"{stem}yl"

    # ---- Fallback: strip terminal -e if present, add -yl ----
    if name.endswith('e'):
        return name[:-1] + "yl"

    # If name already ends in -yl, return as-is
    if name.endswith('yl'):
        return name

    return name + "yl"


# ============================================================================
# Retained Substituent Names
# ============================================================================


def _check_retained_substituent(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
) -> Optional[str]:
    """Check for common retained substituent names.

    Detects isopropyl, tert-butyl, sec-butyl, isobutyl, neopentyl,
    and phenyl by analyzing the branching pattern at the attachment point.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.
        attach_idx: First atom of the substituent (bonded to parent chain).

    Returns:
        Retained name string, or None if no retained name applies.
    """
    frag_set = set(sub_atoms)

    # --- Phenyl detection ---
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(frag_set) and len(ring) == 6:
            if all(mol.GetAtomWithIdx(r).GetIsAromatic() and
                   mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
                # Has a benzene ring
                non_ring = [i for i in sub_atoms if i not in ring_set]
                non_ring_carbons = sum(
                    1 for i in non_ring
                    if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                )
                if non_ring_carbons == 0:
                    return "phenyl"
                elif non_ring_carbons == 1:
                    return "benzyl"

    # --- Alkyl branching detection ---
    carbon_atoms = [i for i in sub_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
    carbon_count = len(carbon_atoms)

    if carbon_count == 0:
        return None

    attach_atom = mol.GetAtomWithIdx(attach_idx)
    if attach_atom.GetSymbol() != 'C':
        return None

    # Count carbon neighbors of the attachment atom within the fragment
    c_neighbors_in_frag = [
        nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
        if nbr.GetIdx() in frag_set and nbr.GetSymbol() == 'C'
    ]

    if carbon_count == 3:
        # isopropyl: CH(CH3)2 -- 2 branches at attachment point
        if len(c_neighbors_in_frag) == 2:
            return "isopropyl"

    elif carbon_count == 4:
        if len(c_neighbors_in_frag) == 3:
            # tert-butyl: C(CH3)3 -- 3 branches at attachment
            return "tert-butyl"
        elif len(c_neighbors_in_frag) == 2:
            # Could be sec-butyl: CH(CH3)(CH2CH3)
            # Check branch sizes
            branch_sizes = []
            for nb_idx in c_neighbors_in_frag:
                sub_visited = {attach_idx}
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
            # Check for isobutyl: CH2CH(CH3)2
            next_c = c_neighbors_in_frag[0]
            next_atom = mol.GetAtomWithIdx(next_c)
            next_c_nbrs = [
                nbr.GetIdx() for nbr in next_atom.GetNeighbors()
                if nbr.GetIdx() in frag_set and nbr.GetIdx() != attach_idx
                and nbr.GetSymbol() == 'C'
            ]
            if len(next_c_nbrs) == 2:
                return "isobutyl"

    elif carbon_count == 5:
        # neopentyl: CH2C(CH3)3
        if len(c_neighbors_in_frag) == 1:
            next_c = c_neighbors_in_frag[0]
            next_atom = mol.GetAtomWithIdx(next_c)
            next_c_nbrs = [
                nbr.GetIdx() for nbr in next_atom.GetNeighbors()
                if nbr.GetIdx() in frag_set and nbr.GetIdx() != attach_idx
                and nbr.GetSymbol() == 'C'
            ]
            if len(next_c_nbrs) == 3:
                return "neopentyl"

    return None


# ============================================================================
# Substituent CIP Stereo Descriptor
# ============================================================================


def _add_substituent_stereo(mol, sub_atoms, name):
    """Add CIP stereodescriptors to a substituent name if stereocenters exist.

    When a substituent contains one or more stereocenters with defined CIP
    labels (R/S), the descriptor is prepended: e.g. "sec-butyl" becomes
    "(R)-sec-butyl", "1-methylpropyl" becomes "(1R)-1-methylpropyl".

    For a single stereocenter the format is "(R)-name" or "(S)-name".
    For multiple stereocenters the format uses locants: "(1R,2S)-name".

    This is a systemic fix: any substituent on any parent (chain, ring,
    heterocycle) that has a stereocenter gets the descriptor.

    Args:
        mol: RDKit Mol object (CIP labels must already be assigned).
        sub_atoms: Atom indices of the substituent fragment.
        name: The substituent name without stereo (e.g., "sec-butyl").

    Returns:
        Name with stereo prefix if stereocenters found, otherwise unchanged.
    """
    if not sub_atoms or not name:
        return name

    # Ensure CIP labels are assigned
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        return name

    # Collect CIP-labeled atoms within the substituent
    stereo_atoms = []
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('_CIPCode'):
            cip = atom.GetProp('_CIPCode')
            # Enforce uppercase for OPSIN compatibility
            if cip in ('r', 's'):
                cip = cip.upper()
            stereo_atoms.append((idx, cip))

    if not stereo_atoms:
        return name

    if len(stereo_atoms) == 1:
        # Single stereocenter: just (R) or (S), no locant needed
        _, cip = stereo_atoms[0]
        return f"({cip})-{name}"

    # Multiple stereocenters: need locants within the substituent.
    # Use a simplified numbering: sort by atom index and assign 1-based.
    sorted_atoms = sorted(sub_atoms)
    idx_to_internal = {idx: pos + 1 for pos, idx in enumerate(sorted_atoms)}

    descs = []
    for atom_idx, cip in stereo_atoms:
        locant = idx_to_internal.get(atom_idx, 0)
        descs.append((locant, cip))
    descs.sort(key=lambda x: x[0])

    desc_str = ",".join(f"{loc}{cip}" for loc, cip in descs)
    return f"({desc_str})-{name}"


# ============================================================================
# Main Entry Point
# ============================================================================


def name_substituent_fragment(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_chain: list,
) -> Optional[str]:
    """Name a substituent fragment, routing to the appropriate naming path.

    This is the centralized entry point for all substituent naming.
    It detects the complexity of the substituent and routes accordingly:
    1. Retained names: isopropyl, tert-butyl, sec-butyl, isobutyl, phenyl.
    2. Linear alkyl (fast path): get_alkyl_name() directly.
    3. Recursive naming: extract SMILES, name recursively, convert to prefix.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: First atom of the substituent (bonded to parent chain).
        parent_chain: List of atom indices in the parent chain.

    Returns:
        Raw prefix name (e.g., "methyl", "isopropyl", "2-methylpropyl")
        WITHOUT enclosing marks. Returns None if naming fails.
    """
    if not sub_atoms:
        return None

    parent_set = set(parent_chain) if parent_chain else set()

    # Step 1: Check retained substituent names FIRST (isopropyl, tert-butyl,
    # sec-butyl, isobutyl, neopentyl, phenyl, benzyl). These depend on
    # the attachment point context and must be checked before the linear
    # alkyl fast path, which cannot distinguish e.g. propyl from isopropyl.
    retained = _check_retained_substituent(mol, sub_atoms, attach_idx)
    if retained:
        return _add_substituent_stereo(mol, sub_atoms, retained)

    # Step 2: Fast path -- linear alkyl (no branching, no heteroatoms)
    if _is_linear_alkyl(mol, sub_atoms):
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                return None
        return None

    # Step 3: Extract fragment SMILES and name recursively
    frag_smiles = _extract_fragment_smiles(mol, sub_atoms, attach_idx, parent_set)
    if frag_smiles is None:
        # Fallback: try simple carbon count for pure-carbon substituents
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass
        return None

    # Step 4: Recursive naming via name_fragment_recursively()
    parent_name = name_fragment_recursively(frag_smiles)
    if parent_name is None:
        # Recursion depth limit or naming failure: fallback to carbon count
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass
        return None

    # Step 5: Convert parent name to prefix form
    carbon_count = sum(
        1 for i in sub_atoms
        if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
    )
    prefix_name = parent_to_prefix(parent_name, chain_length=carbon_count)

    return prefix_name


# ============================================================================
# Convenience: check if substituent needs recursive naming
# ============================================================================


def needs_recursive_naming(mol, sub_atoms: List[int]) -> bool:
    """Check whether a substituent requires recursive naming.

    Returns True if the substituent is branched or contains heteroatoms
    (i.e., not a simple linear alkyl chain).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.

    Returns:
        True if recursive naming may be needed, False for simple linear alkyls.
    """
    return not _is_linear_alkyl(mol, sub_atoms)
