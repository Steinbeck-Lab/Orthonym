"""
Monocyclic lactam naming using IUPAC heterocyclic replacement nomenclature.

Lactams are cyclic amides. Monocyclic lactams are named as heterocyclic
ketones: the ring nitrogen gives the heterocyclic parent name (aza-prefix
via Hantzsch-Widman), and the carbonyl is expressed as a -one suffix.

Naming algorithm:
1. Detect lactam: ring contains -N-C(=O)- where both N and C are in the
   same ring, and the exocyclic O is a double-bonded carbonyl.
2. Distinguish lactam from imide: N bonded to exactly 1 ring C=O (lactam)
   vs 2 ring C=O (imide like succinimide).
3. Determine ring size.
4. For ring size 3-10: Get HW heterocyclic parent name (azetidine, pyrrolidine,
   piperidine, azepane, etc.) from build_hw_name().
5. Apply vowel elision if needed and append '-2-one'.

Reference: IUPAC 2013 Blue Book, P-25.5.3 (Lactams), P-31.1.3 (Replacement)

Examples:
    C1CC(=O)N1     (beta-lactam)       -> azetidin-2-one
    C1CCC(=O)N1    (gamma-lactam)      -> pyrrolidin-2-one
    C1CCCC(=O)N1   (delta-lactam)      -> piperidin-2-one
    C1CCCCC(=O)N1  (epsilon-lactam)    -> azepan-2-one
"""

from collections import deque
from typing import Dict, List, Optional

from rdkit import Chem

from ..rules.heterocycles import build_hw_name


# ---------------------------------------------------------------------------
# Lactam detection
# ---------------------------------------------------------------------------

# SMARTS: trivalent nitrogen in ring bonded to trigonal carbon in ring with
# double-bonded exocyclic oxygen
LACTAM_SMARTS = Chem.MolFromSmarts("[NX3;R]-[CX3;R](=O)")

# Retained parent names for common saturated monocyclic N-heterocycles.
# These override HW systematic names (azolidine -> pyrrolidine, azinane -> piperidine).
_PARENT_NAMES = {
    3: "aziridine",
    4: "azetidine",
    5: "pyrrolidine",
    6: "piperidine",
    7: "azepane",
    8: "azocane",
    9: "azonane",
    10: "azecane",
}


def is_monocyclic_lactam(mol) -> Optional[Dict]:
    """
    Detect whether a molecule is (or contains) a monocyclic lactam.

    A monocyclic lactam has:
    - A ring containing an amide motif: -N-C(=O)- where both the amide
      nitrogen and the carbonyl carbon are in the same ring.
    - The carbonyl oxygen is exocyclic (double-bonded to the carbonyl C).
    - The nitrogen is bonded to exactly ONE ring C=O (not an imide).
    - The ring is not fused with another ring (monocyclic only).

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        Dict with detection info if lactam found:
            ring_atoms: tuple of atom indices in the lactam ring
            nitrogen_idx: atom index of the ring nitrogen
            carbonyl_idx: atom index of the carbonyl carbon (C=O)
            carbonyl_o_idx: atom index of the exocyclic carbonyl oxygen
            ring_size: number of atoms in the ring
        None if not a monocyclic lactam.
    """
    if mol is None:
        return None

    matches = mol.GetSubstructMatches(LACTAM_SMARTS)
    if not matches:
        return None

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return None

    for match in matches:
        n_idx = match[0]      # Ring nitrogen
        co_idx = match[1]     # Ring carbonyl carbon
        # match[2] is the exocyclic =O (from the SMARTS)
        o_idx = match[2]

        # Imide disambiguation: check how many ring C=O groups this N is bonded to.
        # If N is bonded to 2+ ring carbons that each have a C=O, it's an imide.
        n_atom = mol.GetAtomWithIdx(n_idx)
        ring_co_count = 0
        for neighbor in n_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx == n_idx:
                continue
            # Check if this neighbor is a ring carbon with an exocyclic C=O
            if neighbor.GetSymbol() == "C" and neighbor.IsInRing():
                for bond in neighbor.GetBonds():
                    other_idx = bond.GetOtherAtomIdx(nbr_idx)
                    other_atom = mol.GetAtomWithIdx(other_idx)
                    if (
                        other_atom.GetSymbol() == "O"
                        and bond.GetBondType() == Chem.BondType.DOUBLE
                        and not other_atom.IsInRing()
                    ):
                        ring_co_count += 1
                        break

        if ring_co_count >= 2:
            # This nitrogen has 2+ ring C=O neighbors -> imide, not lactam
            continue

        # Find the ring containing both N and C
        for ring in atom_rings:
            ring_set = set(ring)
            if n_idx in ring_set and co_idx in ring_set:
                # Exocyclic carbonyl O must NOT be in the ring
                if o_idx in ring_set:
                    continue

                # Monocyclic guard: no ring atom should be shared with another ring
                is_monocyclic = True
                for other_ring in atom_rings:
                    if set(other_ring) == ring_set:
                        continue
                    if ring_set & set(other_ring):
                        is_monocyclic = False
                        break

                if not is_monocyclic:
                    continue

                return {
                    "ring_atoms": ring,
                    "nitrogen_idx": n_idx,
                    "carbonyl_idx": co_idx,
                    "carbonyl_o_idx": o_idx,
                    "ring_size": len(ring),
                }

    return None


# ---------------------------------------------------------------------------
# Lactam ring naming
# ---------------------------------------------------------------------------

# Ring sizes supported via Hantzsch-Widman naming
_HW_RING_SIZES = frozenset(range(3, 11))

# Maximum ring size for macrolactam naming
_MAX_MACROLACTAM_SIZE = 50


def name_lactam_ring(ring_size: int) -> Optional[str]:
    """
    Get the IUPAC name for a monocyclic lactam of a given ring size.

    For ring sizes 3-10: uses retained parent names (pyrrolidine, piperidine)
    where available, falling back to Hantzsch-Widman systematic naming.
    Applies vowel elision and appends '-2-one'.

    Args:
        ring_size: Number of atoms in the lactam ring (3-50 supported).

    Returns:
        IUPAC name string (e.g., 'pyrrolidin-2-one'), or None if ring size
        is not supported.

    Examples:
        >>> name_lactam_ring(4)
        'azetidin-2-one'
        >>> name_lactam_ring(5)
        'pyrrolidin-2-one'
        >>> name_lactam_ring(6)
        'piperidin-2-one'
        >>> name_lactam_ring(7)
        'azepan-2-one'
    """
    if ring_size < 3:
        return None

    # Use retained/common parent names for ring sizes 3-10
    if ring_size in _HW_RING_SIZES:
        parent_name = _PARENT_NAMES.get(ring_size)
        if not parent_name:
            # Fallback to HW systematic for uncommon sizes
            parent_name = build_hw_name(
                heteroatoms=[(1, "N")],
                ring_size=ring_size,
                is_saturated=True,
                is_aromatic=False,
            )
        if not parent_name:
            return None

        # Apply vowel elision: remove terminal 'e' before '-one'
        if parent_name.endswith("e"):
            stem = parent_name[:-1]
        else:
            stem = parent_name

        return f"{stem}-2-one"

    # Large ring lactams (>10): replacement nomenclature
    if ring_size > _MAX_MACROLACTAM_SIZE:
        return None

    from ..data.chain_names import get_chain_prefix

    try:
        chain_prefix = get_chain_prefix(ring_size)
    except ValueError:
        return None

    # Build: azacyclo + {prefix} + an-2-one
    return f"azacyclo{chain_prefix}an-2-one"


# ---------------------------------------------------------------------------
# Full lactam naming
# ---------------------------------------------------------------------------

def name_monocyclic_lactam(mol) -> Optional[str]:
    """
    Generate the IUPAC name for a monocyclic lactam, including substituents.

    Detects whether the molecule is a monocyclic lactam and returns
    its name using heterocyclic replacement nomenclature with -one suffix.
    Substituents on the ring are included as prefixes with locants.

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        IUPAC name string (e.g., 'pyrrolidin-2-one', '3-methylazetidin-2-one'),
        or None if the molecule is not a monocyclic lactam.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('C1CCC(=O)N1')
        >>> name_monocyclic_lactam(mol)
        'pyrrolidin-2-one'
    """
    info = is_monocyclic_lactam(mol)
    if info is None:
        return None

    parent_name = name_lactam_ring(info["ring_size"])
    if parent_name is None:
        return None

    # Detect substituents on the lactam ring
    ring_atoms = info["ring_atoms"]
    nitrogen_idx = info["nitrogen_idx"]
    carbonyl_idx = info["carbonyl_idx"]
    carbonyl_o_idx = info["carbonyl_o_idx"]

    # Build IUPAC locant mapping for lactam ring
    # Numbering: N atom = 1, carbonyl C = 2, then continue around ring
    ring_list = list(ring_atoms)
    ring_set = set(ring_list)

    # Find nitrogen position in ring and reorder so N is first
    try:
        n_pos = ring_list.index(nitrogen_idx)
    except ValueError:
        return parent_name

    # Reorder ring starting from N, going toward carbonyl C
    ordered = ring_list[n_pos:] + ring_list[:n_pos]

    # Check direction: next atom should be carbonyl C
    if len(ordered) > 1 and ordered[1] != carbonyl_idx:
        # Reverse direction (keep N first)
        ordered = [ordered[0]] + ordered[1:][::-1]

    # Build atom-to-locant mapping (1-indexed)
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(ordered)}

    # Collect stereodescriptors using lactam ring locant mapping
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from rdkit.Chem import rdCIPLabeler

    rdCIPLabeler.AssignCIPLabels(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    # Find exocyclic substituents (excluding carbonyl O which is the =O)
    excluded = ring_set | {carbonyl_o_idx}
    substituents = _detect_lactam_substituents(mol, ordered, atom_to_locant, excluded)

    if not substituents:
        # No substituents but may have stereo
        if stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            return f"{stereo_prefix}{parent_name}"
        return parent_name

    # Build prefix string
    from collections import defaultdict
    groups = defaultdict(list)
    for sub_name, locant, is_on_nitrogen in substituents:
        groups[(sub_name, is_on_nitrogen)].append(locant)

    # Sort locants within each group
    for key in groups:
        groups[key].sort()

    # Build prefix parts, sorted alphabetically
    from ..assembly.naming_utils import alpha_sort_key, format_substituent_prefix
    prefix_parts = []
    for (name, is_on_nitrogen), locants in sorted(
        groups.items(), key=lambda x: alpha_sort_key(x[0][0])
    ):
        count = len(locants)
        if is_on_nitrogen:
            # N-substitution: N-methyl, N,N-dimethyl
            prefix_str = _format_n_prefix(name, count)
        else:
            prefix_str = format_substituent_prefix(name, locants, count)
        prefix_parts.append(prefix_str)

    if not prefix_parts:
        return parent_name

    # Join prefix parts
    prefix = "-".join(prefix_parts)
    name = f"{prefix}{parent_name}"

    # Prepend stereo prefix if descriptors exist
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        name = f"{stereo_prefix}{name}"

    return name


def _format_n_prefix(name: str, count: int) -> str:
    """Format an N-substituent prefix for a lactam."""
    from ..assembly.naming_utils import get_multiplier_prefix

    if count == 1:
        return f"N-{name}"
    n_locants = ",".join(["N"] * count)
    multiplier = get_multiplier_prefix(count, name)
    return f"{n_locants}-{multiplier}{name}"


def _detect_lactam_substituents(mol, ordered_ring, atom_to_locant, excluded):
    """
    Detect substituents on a lactam ring.

    Args:
        mol: RDKit Mol object
        ordered_ring: List of atom indices in IUPAC numbering order
        atom_to_locant: Dict mapping atom index to IUPAC locant
        excluded: Set of atom indices to exclude (ring + carbonyl O)

    Returns:
        List of (substituent_name, locant, is_on_nitrogen) tuples
    """
    substituents = []
    ring_set = set(ordered_ring)

    for ring_atom_idx in ordered_ring:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        locant = atom_to_locant[ring_atom_idx]
        is_nitrogen = ring_atom.GetSymbol() == "N"

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in excluded:
                continue

            sub_name = _identify_lactam_substituent(mol, nbr_idx, excluded)
            if sub_name:
                substituents.append((sub_name, locant, is_nitrogen))

    return substituents


def _identify_lactam_substituent(mol, start_idx, excluded):
    """
    Identify a substituent on a lactam ring by its starting atom.

    Returns:
        Substituent prefix name (e.g., 'amino', 'methyl', 'hydroxy') or None
    """
    atom = mol.GetAtomWithIdx(start_idx)
    symbol = atom.GetSymbol()

    # Halogens
    halogen_map = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
    if symbol in halogen_map:
        return halogen_map[symbol]

    # Nitrogen: amino (-NH2)
    if symbol == "N":
        h_count = atom.GetTotalNumHs()
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 2 and len(neighbors) == 0:
            return "amino"

    # Oxygen: hydroxy (-OH)
    if symbol == "O":
        h_count = atom.GetTotalNumHs()
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 1 and len(neighbors) == 0:
            return "hydroxy"

    # Carbon: alkyl groups
    if symbol == "C":
        # BFS for pure alkyl
        visited = {start_idx}
        queue = deque([start_idx])
        carbon_count = 0
        is_pure_alkyl = True

        while queue:
            idx = queue.popleft()
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() == "C":
                carbon_count += 1
            elif a.GetSymbol() != "H":
                is_pure_alkyl = False

            for nbr in a.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited and nbr_idx not in excluded:
                    visited.add(nbr_idx)
                    queue.append(nbr_idx)

        if is_pure_alkyl and carbon_count > 0:
            from ..assembly.naming_utils import get_alkyl_name

            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass

    return None
