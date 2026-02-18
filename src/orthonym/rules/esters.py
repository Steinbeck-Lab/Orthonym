"""
Ester naming following IUPAC two-component format.

Esters are named as "alkyl alkanoate" where:
- alkyl: derived from the alcohol portion (attached to ester oxygen)
- alkanoate: derived from the acid portion (contains C=O)

Example: CH3-COO-CH2-CH3 -> "ethyl acetate"
         (acetate from acetic acid, ethyl from ethanol)

SMARTS: "[CX3](=O)[OX2][#6]"
        - Position 0: carbonyl carbon (acid side)
        - Position 1: carbonyl oxygen (=O)
        - Position 2: ester oxygen (-O-)
        - Position 3: first alkyl carbon (alcohol side)
"""

from typing import Tuple, List, Optional
from rdkit import Chem

from ..data.trivial_acids import get_acylate_name
from ..assembly.naming_utils import get_alkyl_name
from ..data.chain_names import get_chain_prefix, get_alkyl_name as _chain_alkyl_name, get_acid_stem


def parse_ester_fragments(mol, ester_match: tuple) -> Tuple[List[int], List[int]]:
    """
    Split ester into acid and alkyl fragments.

    Args:
        mol: RDKit Mol object
        ester_match: Tuple of atom indices from ester SMARTS match
                     (carbonyl_c, carbonyl_o, ester_o, alkyl_c)

    Returns:
        Tuple of (acid_atoms, alkyl_atoms) where each is a list of atom indices

    The acid fragment includes the carbonyl carbon and everything attached
    to it except via the ester oxygen.
    The alkyl fragment includes everything attached to the ester oxygen
    except the carbonyl carbon.
    """
    # SMARTS "[CX3](=O)[OX2][#6]" gives us:
    # match[0] = carbonyl carbon
    # match[1] = carbonyl oxygen (=O)
    # match[2] = ester oxygen
    # match[3] = first alkyl carbon (may not be present depending on SMARTS)

    carbonyl_c = ester_match[0]
    ester_o = ester_match[2] if len(ester_match) > 2 else None

    if ester_o is None:
        # Handle different SMARTS match structures
        # Look for the ester oxygen by finding O neighbor of carbonyl C that isn't =O
        carbonyl_atom = mol.GetAtomWithIdx(carbonyl_c)
        for neighbor in carbonyl_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(carbonyl_c, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    ester_o = neighbor.GetIdx()
                    break

    if ester_o is None:
        return [], []

    # BFS to find acid fragment (from carbonyl_c, excluding ester_o)
    acid_atoms = _bfs_fragment(mol, carbonyl_c, exclude_atom=ester_o)

    # BFS to find alkyl fragment (from ester_o, excluding carbonyl_c)
    # Start from neighbors of ester_o that aren't carbonyl_c
    ester_o_atom = mol.GetAtomWithIdx(ester_o)
    alkyl_start = None
    for neighbor in ester_o_atom.GetNeighbors():
        if neighbor.GetIdx() != carbonyl_c:
            alkyl_start = neighbor.GetIdx()
            break

    if alkyl_start is None:
        return acid_atoms, []

    alkyl_atoms = _bfs_fragment(mol, alkyl_start, exclude_atom=ester_o)

    return acid_atoms, alkyl_atoms


def _bfs_fragment(mol, start_atom: int, exclude_atom: int) -> List[int]:
    """BFS to collect all atoms in a fragment, excluding one connection."""
    visited = {start_atom}
    queue = [start_atom]

    while queue:
        current = queue.pop(0)
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx != exclude_atom:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return list(visited)


def acid_fragment_has_ring(mol, acid_atoms: List[int]) -> bool:
    """
    Check if the acid fragment of an ester contains ring atoms.

    When the acid portion contains a ring (e.g., benzoate, indole-carboxylate),
    simple carbon-counting is wrong -- the ring atoms must not be linearized.

    Args:
        mol: RDKit Mol object
        acid_atoms: Atom indices of the acid fragment

    Returns:
        True if any acid atom is in a ring
    """
    for idx in acid_atoms:
        if mol.GetAtomWithIdx(idx).IsInRing():
            return True
    return False


def get_ring_acid_name(mol, acid_atoms: List[int]) -> Optional[str]:
    """
    Name the acid portion of an ester when it contains a ring.

    Handles cases like:
    - Benzene + COOH -> "benzoic" (trivial)
    - Ring + COOH -> "[ring-name]carboxylic" (systematic)

    Args:
        mol: RDKit Mol object
        acid_atoms: Atom indices of the acid fragment

    Returns:
        Acid name stem (e.g., "benzoic"), or None if cannot determine
    """
    acid_set = set(acid_atoms)
    ring_info = mol.GetRingInfo()

    # Find ring atoms within the acid fragment
    ring_atoms_in_acid = set()
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(acid_set):
            ring_atoms_in_acid.update(ring_set)

    if not ring_atoms_in_acid:
        return None

    # Count how many complete rings are in the acid fragment
    rings_in_acid = []
    for ring in ring_info.AtomRings():
        if set(ring).issubset(acid_set):
            rings_in_acid.append(ring)

    n_rings = len(rings_in_acid)

    # Single ring: try simple naming (benzene -> benzoic, cycloalkane -> carboxylic)
    if n_rings == 1:
        ring = rings_in_acid[0]
        if len(ring) == 6:
            all_carbon = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring
            )
            all_aromatic = all(
                mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring
            )
            if all_carbon and all_aromatic:
                return "benzoic"

        # Check for heterocyclic single ring
        has_heteroatom = any(
            mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
            for idx in ring
        )
        if has_heteroatom:
            from ..rules.heterocycles import classify_heterocycle
            het_info = classify_heterocycle(mol, tuple(ring))
            if het_info and het_info.get('name'):
                return f"{het_info['name']}-carboxylic"

        # Simple carbocyclic ring + COOH
        all_carbon = all(
            mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring
        )
        if all_carbon:
            ring_size = len(ring)
            stem = get_chain_prefix(ring_size)
            if stem:
                return f"cyclo{stem}anecarboxylic"

    # For fused/complex ring systems with COOH, try to get the ring system name
    # and append "-carboxylic"
    # Build a sub-molecule from just the ring atoms to identify
    from ..data.retained_names import RETAINED_NAMES
    from ..rules.polycyclics import identify_polycyclic

    # Try identifying as a polycyclic aromatic
    # Create sub-mol from acid fragment
    try:
        atom_map = {}
        rw = Chem.RWMol()
        for i, idx in enumerate(sorted(acid_set)):
            atom = mol.GetAtomWithIdx(idx)
            new_idx = rw.AddAtom(atom)
            atom_map[idx] = new_idx

        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in atom_map and a2 in atom_map:
                rw.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())

        sub_mol = rw.GetMol()

        # Check retained names for the ring system
        sub_smiles = Chem.MolToSmiles(sub_mol, canonical=True)
        if sub_smiles in RETAINED_NAMES:
            ring_name = RETAINED_NAMES[sub_smiles]
            # If it ends in "-carboxylic acid", extract the stem
            if "carboxylic acid" in ring_name:
                return ring_name.replace(" acid", "").strip()

        # Try polycyclic identification on the sub-mol
        pah_name = identify_polycyclic(sub_mol)
        if pah_name:
            return f"{pah_name}carboxylic"

        # For heterocyclic rings, try heterocycle naming
        from ..rules.heterocycles import classify_heterocycle
        for ring in ring_info.AtomRings():
            ring_set = set(ring)
            if ring_set.issubset(acid_set):
                has_heteroatom = any(
                    mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                    for idx in ring
                )
                if has_heteroatom:
                    het_info = classify_heterocycle(mol, tuple(ring))
                    if het_info and het_info.get('name'):
                        return f"{het_info['name']}-carboxylic"
    except Exception:
        pass

    # Fallback: cannot name ring acid
    return None


def get_acid_fragment_name(mol, acid_atoms: List[int]) -> str:
    """
    Get the acid name from the acid fragment.

    For simple chain acids, determines chain length and returns systematic name.
    For recognized trivial acids, returns trivial name.

    Args:
        mol: RDKit Mol object
        acid_atoms: Atom indices of the acid fragment

    Returns:
        Acid stem name (e.g., "acetic", "propanoic", "benzoic")
    """
    # Check if acid fragment contains ring atoms
    if acid_fragment_has_ring(mol, acid_atoms):
        ring_name = get_ring_acid_name(mol, acid_atoms)
        if ring_name:
            return ring_name
        # If ring naming fails, fall through to carbon-count (imperfect but safe)

    # Count carbons in acid fragment
    carbon_count = sum(
        1 for idx in acid_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Check for common trivial acids by structure
    # 1 carbon = formic (methanoic)
    # 2 carbons = acetic (ethanoic)
    if carbon_count == 1:
        return "formic"  # Trivial name preferred
    elif carbon_count == 2:
        return "acetic"  # Trivial name preferred

    # Detect C=C double bonds and rings in the acid fragment
    acid_atoms_set = set(acid_atoms)
    has_ring = any(mol.GetAtomWithIdx(idx).IsInRing() for idx in acid_atoms)
    cc_double_bond_count = 0
    if not has_ring:
        seen_bonds = set()
        for idx in acid_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != 'C':
                continue
            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)
                if (nbr.GetIdx() in acid_atoms_set
                        and nbr.GetSymbol() == 'C'
                        and bond.GetBondTypeAsDouble() == 2.0
                        and bond.GetIdx() not in seen_bonds):
                    seen_bonds.add(bond.GetIdx())
                    cc_double_bond_count += 1
    has_cc_double_bond = cc_double_bond_count > 0

    # Fatty acid trivial names by (carbon_count, double_bond_count)
    # Saturated (0 double bonds)
    FATTY_ACID_TRIVIAL_BY_STRUCTURE = {
        (12, 0): "lauric",
        (14, 0): "myristic",
        (16, 0): "palmitic",
        (18, 0): "stearic",
        (18, 1): "oleic",           # C18:1
        (18, 2): "linoleic",        # C18:2
        (18, 3): "linolenic",       # C18:3
        (20, 0): "arachidic",
        (20, 4): "arachidonic",     # C20:4
    }

    if not has_ring:
        key = (carbon_count, cc_double_bond_count)
        if key in FATTY_ACID_TRIVIAL_BY_STRUCTURE:
            return FATTY_ACID_TRIVIAL_BY_STRUCTURE[key]

    # For unsaturated acids not matching a known trivial name, extract the
    # fragment and name it via the naming pipeline to get the full systematic
    # name including double bond positions (e.g., "octadeca-9,12-dienoic").
    # This prevents unsaturated acids from being named with saturated stems.
    if has_cc_double_bond and not has_ring:
        acid_name = _name_acid_fragment_with_unsaturation(mol, acid_atoms)
        if acid_name:
            return acid_name

    # Systematic for others - use centralized chain naming
    return get_acid_stem(carbon_count)


def _extract_fragment_smiles(mol, keep_atoms: set, cap_atom_idx: int,
                             cap_element: int = 8) -> Optional[str]:
    """Extract a fragment from *mol* preserving stereochemistry.

    Works by copying the whole molecule, removing all bonds that cross
    the fragment boundary, extracting the desired fragment, and capping
    the attachment point.  Because we copy the molecule rather than
    building from scratch, all E/Z and R/S stereochemistry is preserved.

    Args:
        mol: Source RDKit Mol.
        keep_atoms: Set of atom indices to keep in the fragment.
        cap_atom_idx: Atom index in *keep_atoms* to cap with a new atom.
        cap_element: Atomic number for the capping atom (8 = oxygen).

    Returns:
        SMILES of the capped fragment, or None on failure.
    """
    rw = Chem.RWMol(mol)

    # Collect bonds that cross the fragment boundary
    bonds_to_remove = []
    for bond in rw.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (a in keep_atoms) != (b in keep_atoms):
            bonds_to_remove.append((a, b))

    # Remove crossing bonds
    for a, b in bonds_to_remove:
        rw.RemoveBond(a, b)

    # Add capping atom (e.g. -OH on carbonyl to make acid)
    cap_idx = rw.AddAtom(Chem.Atom(cap_element))
    rw.AddBond(cap_atom_idx, cap_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rw)
        # Get fragments with atom-index mapping so we can identify which
        # fragment contains the cap_atom_idx
        frag_atom_lists = Chem.GetMolFrags(rw.GetMol())
        frags = Chem.GetMolFrags(
            rw.GetMol(), asMols=True, sanitizeFrags=True
        )
        # Find the fragment whose atom mapping includes cap_atom_idx
        for frag_atoms, frag_mol in zip(frag_atom_lists, frags):
            if cap_atom_idx in frag_atoms:
                return Chem.MolToSmiles(frag_mol)
    except Exception:
        pass

    return None


def _name_acid_fragment_with_unsaturation(mol, acid_atoms: List[int]) -> Optional[str]:
    """Extract an unsaturated acid fragment and name it via the pipeline.

    Uses RWMol bond-removal to extract the fragment, preserving E/Z
    stereochemistry.  Caps the carbonyl carbon with -OH to form a
    carboxylic acid, then names it to get the full systematic name
    including double bond positions (e.g., "(4E)-octa-4,7-dienoic").

    Args:
        mol: RDKit Mol object (the full molecule).
        acid_atoms: Atom indices of the acid fragment (R-C(=O) without the
                    ester oxygen).

    Returns:
        Acid stem name with unsaturation info (e.g., "(4E)-octa-4,7-dienoic"),
        or None if extraction/naming fails.
    """
    acid_atoms_set = set(acid_atoms)

    # Find the carbonyl carbon: the C with a double-bond to O
    carbonyl_c = None
    for idx in acid_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if (nbr.GetSymbol() == 'O'
                    and bond.GetBondTypeAsDouble() == 2.0
                    and nbr.GetIdx() in acid_atoms_set):
                carbonyl_c = idx
                break
        if carbonyl_c is not None:
            break

    if carbonyl_c is None:
        return None

    frag_smi = _extract_fragment_smiles(mol, acid_atoms_set, carbonyl_c)
    if not frag_smi:
        return None

    # Name via the pipeline
    from ..assembly.fragment_naming import name_fragment_recursively
    full_name = name_fragment_recursively(frag_smi)
    if not full_name or full_name == "unknown":
        return None

    # Extract the acid stem: strip " acid" suffix and any leading stereo descriptors
    # e.g., "(9Z,12Z)-octadeca-9,12-dienoic acid" -> "octadeca-9,12-dienoic"
    # We keep the descriptors since get_acyloxy_prefix handles the -oic -> -oyloxy conversion
    stem = full_name.strip()
    if stem.endswith(" acid"):
        stem = stem[:-5].strip()

    # Validate: stem should end in "-oic" or "-ic" for acid conversion to work
    if stem.endswith("oic") or stem.endswith("ic"):
        return stem

    return None


def _name_alkyl_fragment_with_unsaturation(
    mol, alkyl_atoms: List[int], alkyl_set: set
) -> Optional[str]:
    """Extract an unsaturated alkyl fragment and name it via the pipeline.

    Uses RWMol bond-removal to extract the fragment, preserving E/Z
    stereochemistry.  Caps the attachment carbon with -OH to form an
    alcohol, names it, then converts the alcohol name to the alkyl form.

    Args:
        mol: RDKit Mol object (the full molecule).
        alkyl_atoms: Atom indices of the alkyl fragment.
        alkyl_set: Set of alkyl atom indices (for fast membership check).

    Returns:
        Alkyl name with unsaturation (e.g., "(10Z)-pentadec-10-en-1-yl"),
        or None if extraction/naming fails.
    """
    # Find the attachment carbon: the C bonded to the ester oxygen (not in alkyl set)
    attach_c = None
    for idx in alkyl_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'O' and nbr.GetIdx() not in alkyl_set:
                attach_c = idx
                break
        if attach_c is not None:
            break

    if attach_c is None:
        return None

    frag_smi = _extract_fragment_smiles(mol, alkyl_set, attach_c)
    if not frag_smi:
        return None

    # Name via the pipeline
    from ..assembly.fragment_naming import name_fragment_recursively
    full_name = name_fragment_recursively(frag_smi)
    if not full_name or full_name == "unknown":
        return None

    # Convert alcohol name to alkyl form:
    # "pentadec-10-en-1-ol" -> "pentadec-10-en-1-yl"
    # "(10Z)-pentadec-10-en-1-ol" -> "(10Z)-pentadec-10-en-1-yl"
    name = full_name.strip()

    # Strip trailing "-ol" and replace with "-yl"
    # Handle patterns like: "propan-1-ol" -> "propyl", "pentadec-10-en-1-ol" -> "pentadec-10-en-1-yl"
    if name.endswith("-ol"):
        # e.g., "pentadec-10-en-1-ol" -> "pentadec-10-en-1-yl"
        return name[:-2] + "yl"
    elif name.endswith("ol"):
        # e.g., "methanol" -> "methyl" (via -an-ol -> -yl)
        base = name[:-2]
        if base.endswith("an"):
            return base[:-2] + "yl"
        return base + "yl"

    return None


def get_alkyl_fragment_name(mol, alkyl_atoms: List[int]) -> str:
    """
    Get the alkyl name from the alkyl fragment.

    For simple chains, returns methyl, ethyl, propyl, etc.
    When the alkyl fragment contains a ring, tries to name it properly
    (e.g., "phenyl" for benzene, "cyclohexyl" for cyclohexane).

    Args:
        mol: RDKit Mol object
        alkyl_atoms: Atom indices of the alkyl fragment

    Returns:
        Alkyl name (e.g., "methyl", "ethyl", "phenyl")
    """
    alkyl_set = set(alkyl_atoms)

    # Check if alkyl fragment contains a ring
    has_ring = any(mol.GetAtomWithIdx(idx).IsInRing() for idx in alkyl_atoms)

    if has_ring:
        # Try to name the ring-containing alkyl fragment
        # Common case: phenyl (benzene ring)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            ring_in_fragment = all(r in alkyl_set for r in ring)
            if not ring_in_fragment:
                continue
            if len(ring) == 6:
                all_aromatic = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                all_carbon = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                if all_aromatic and all_carbon:
                    # Count chain carbons outside the ring
                    chain_carbons = sum(
                        1 for idx in alkyl_atoms
                        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and idx not in set(ring)
                    )
                    if chain_carbons == 0:
                        return "phenyl"
                    # e.g., benzyl = phenyl + CH2
                    if chain_carbons == 1:
                        return "benzyl"
                    if chain_carbons == 2:
                        return "2-phenylethyl"
                    # For longer chains: name as n-phenylalkyl
                    try:
                        chain_name = _chain_alkyl_name(chain_carbons)
                        return f"{chain_carbons}-phenyl{chain_name}"
                    except ValueError:
                        pass

        # Fallback: try naming the full fragment as a simple ring substituent
        # Count only non-ring carbons if ring naming is too complex
        pass

    # Simple alkyl chain (no ring)
    carbon_atoms = [
        idx for idx in alkyl_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    ]
    carbon_count = len(carbon_atoms)

    if carbon_count == 0:
        return ""

    # Detect C=C double bonds in the alkyl fragment
    has_cc_double_bond = False
    seen_bonds = set()
    for idx in alkyl_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if (nbr.GetIdx() in alkyl_set
                    and nbr.GetSymbol() == 'C'
                    and bond.GetBondTypeAsDouble() == 2.0
                    and bond.GetIdx() not in seen_bonds):
                seen_bonds.add(bond.GetIdx())
                has_cc_double_bond = True

    # For unsaturated alkyl chains, extract the fragment and name it via the
    # pipeline to get the full name including double bond positions, then
    # convert the alcohol name to the alkyl form.
    if has_cc_double_bond and not has_ring:
        alkyl_name = _name_alkyl_fragment_with_unsaturation(mol, alkyl_atoms, alkyl_set)
        if alkyl_name:
            return alkyl_name

    # Detect branching: find the attachment point (C bonded to ester O)
    # and check if it's a branching carbon
    if carbon_count >= 3:
        for idx in carbon_atoms:
            atom = mol.GetAtomWithIdx(idx)
            # Find carbon bonded to ester oxygen (not in alkyl set)
            bonded_to_o = any(
                nbr.GetSymbol() == 'O' and nbr.GetIdx() not in alkyl_set
                for nbr in atom.GetNeighbors()
            )
            if bonded_to_o:
                # Count carbon neighbors (branches)
                c_nbrs = sum(
                    1 for nbr in atom.GetNeighbors()
                    if nbr.GetSymbol() == 'C' and nbr.GetIdx() in alkyl_set
                )
                if c_nbrs >= 2:
                    # Branched at attachment point
                    if carbon_count == 3 and c_nbrs == 2:
                        return "propan-2-yl"
                    elif carbon_count == 4 and c_nbrs == 3:
                        return "2-methylpropan-2-yl"
                    elif carbon_count == 4 and c_nbrs == 2:
                        return "butan-2-yl"
                break

    try:
        return _chain_alkyl_name(carbon_count)
    except ValueError:
        # Fallback for edge cases
        return f"{carbon_count}C-yl"


def is_lactone(mol, ester_match: tuple) -> bool:
    """
    Check if the ester is a lactone (cyclic ester).

    Lactones have the carbonyl carbon and an alkyl carbon in the same ring.
    """
    carbonyl_c = ester_match[0]

    # Get ester oxygen and find alkyl carbon
    ester_o = ester_match[2] if len(ester_match) > 2 else None
    if ester_o is None:
        # Find ester oxygen manually
        carbonyl_atom = mol.GetAtomWithIdx(carbonyl_c)
        for neighbor in carbonyl_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(carbonyl_c, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    ester_o = neighbor.GetIdx()
                    break

    if ester_o is None:
        return False

    ester_o_atom = mol.GetAtomWithIdx(ester_o)
    alkyl_c = None
    for neighbor in ester_o_atom.GetNeighbors():
        if neighbor.GetIdx() != carbonyl_c:
            alkyl_c = neighbor.GetIdx()
            break

    if alkyl_c is None:
        return False

    # Check if carbonyl_c and alkyl_c are in the same ring
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if carbonyl_c in ring and alkyl_c in ring:
            return True

    return False


def name_ester(mol, ester_match: tuple) -> Optional[str]:
    """
    Generate IUPAC name for an ester.

    Format: "alkyl alkanoate" (e.g., "methyl acetate", "ethyl propanoate")

    For ring-containing acid portions (e.g., methyl benzoate), the ring acid
    naming is used to produce correct names like "methyl benzoate" instead of
    incorrectly linearizing ring atoms ("methyl heptanoate").

    Args:
        mol: RDKit Mol object
        ester_match: Tuple from ester SMARTS match

    Returns:
        Ester name string, or None if cannot be named (e.g., lactone, complex ring)
    """
    # Check for lactone (cyclic ester) - handle differently
    if is_lactone(mol, ester_match):
        # Lactones need special handling - defer to later implementation
        return None  # Signal to use different naming path

    # Parse into fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_match)

    if not acid_atoms or not alkyl_atoms:
        return None  # Cannot determine fragments

    # Guard: if acid portion contains a ring but ring naming fails,
    # return None to let complex ring naming handle it
    if acid_fragment_has_ring(mol, acid_atoms):
        ring_acid_name = get_ring_acid_name(mol, acid_atoms)
        if ring_acid_name is None:
            # Complex ring acid (fused heterocycle etc.) - defer to complex naming
            return None

    # Get acid name and convert to acylate
    acid_name = get_acid_fragment_name(mol, acid_atoms)
    acylate_name = get_acylate_name(acid_name)

    # Get alkyl name
    alkyl_name = get_alkyl_fragment_name(mol, alkyl_atoms)

    if not alkyl_name:
        return None

    # Combine: "alkyl acylate"
    return f"{alkyl_name} {acylate_name}"


def find_ester_match(mol) -> Optional[tuple]:
    """
    Find the first ester match in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of atom indices for the ester, or None if no ester found
    """
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return matches[0]
    return None


# ============================================================================
# Acyloxy Prefix Naming (IUPAC P-65.6.3.2.2)
# ============================================================================
#
# When an ester is named as a substituent prefix (e.g., on a ring parent),
# the R-CO-O- portion is named as an "acyloxy" group:
#   acid name (drop "-ic") + "-yloxy"
#
# Examples:
#   formic    -> formyloxy
#   acetic    -> acetyloxy
#   benzoic   -> benzoyloxy
#   propanoic -> propanoyloxy

# Explicit mapping for trivial acid names to acyloxy prefixes.
# These are common acids whose acyloxy forms have established trivial stems.
TRIVIAL_ACID_TO_ACYLOXY = {
    "formic": "formyloxy",
    "acetic": "acetyloxy",
    "propionic": "propionyloxy",
    "butyric": "butyryloxy",
    "valeric": "valeryloxy",
    "caproic": "caproyloxy",
    "benzoic": "benzoyloxy",
    "oxalic": "oxalyloxy",
    "lactic": "lactyloxy",
    # Fatty acids (common long-chain acids with retained names)
    "lauric": "lauroyloxy",          # C12:0
    "myristic": "myristoyloxy",      # C14:0
    "palmitic": "palmitoyloxy",      # C16:0
    "stearic": "stearoyloxy",        # C18:0
    "oleic": "oleoyloxy",            # C18:1
    "linoleic": "linoleoyloxy",      # C18:2
    "linolenic": "linolenoyloxy",    # C18:3
    "arachidic": "arachidoyloxy",    # C20:0
    "arachidonic": "arachidonoyloxy", # C20:4
    # Systematic names for saturated fatty acids (ensures decomposition path
    # also uses trivial acyloxy forms)
    "dodecanoic": "lauroyloxy",      # C12:0 systematic
    "tetradecanoic": "myristoyloxy", # C14:0 systematic
    "hexadecanoic": "palmitoyloxy",  # C16:0 systematic
    "octadecanoic": "stearoyloxy",   # C18:0 systematic
    "icosanoic": "arachidoyloxy",    # C20:0 systematic
}


def get_acyloxy_prefix(acid_name: str) -> str:
    """
    Convert an acid name to its acyloxy prefix form (IUPAC P-65.6.3.2.2).

    Used when an ester group is named as a substituent prefix rather than
    the principal characteristic group. The R-CO-O- portion becomes an
    "acyloxy" prefix.

    Conversion rule: drop "-ic" (or "-ic acid"), add "-yloxy".

    Args:
        acid_name: The acid name without "acid" suffix
                   (e.g., "acetic", "propanoic", "benzoic")

    Returns:
        The acyloxy prefix (e.g., "acetyloxy", "propanoyloxy", "benzoyloxy")

    Examples:
        >>> get_acyloxy_prefix("acetic")
        'acetyloxy'
        >>> get_acyloxy_prefix("propanoic")
        'propanoyloxy'
        >>> get_acyloxy_prefix("benzoic")
        'benzoyloxy'
        >>> get_acyloxy_prefix("formic")
        'formyloxy'
    """
    name = acid_name.lower().strip()

    # Strip trailing " acid" if present
    if name.endswith(" acid"):
        name = name[:-5].strip()

    # Check trivial acid lookup first
    if name in TRIVIAL_ACID_TO_ACYLOXY:
        return TRIVIAL_ACID_TO_ACYLOXY[name]

    # Handle -carboxylic acids (ring acids like cyclopentanecarboxylic, cyclohexanecarboxylic)
    # carboxylic -> carbonyloxy (not carboxylyloxy from the generic -ic rule)
    if name.endswith("carboxylic"):
        return name[:-len("carboxylic")] + "carbonyloxy"

    # Systematic conversion: drop "-ic", add "-yloxy"
    # Works for both "-oic" (propanoic -> propanoyloxy) and "-ic" (generic)
    if name.endswith("ic"):
        return name[:-2] + "yloxy"

    # Fallback: just append "yloxy"
    return name + "yloxy"


def name_ester_as_prefix(mol, ester_match: tuple) -> Optional[str]:
    """
    Generate an acyloxy prefix name for an ester group.

    Used when the ester is a substituent (not the principal characteristic
    group). Extracts the acid fragment, determines its name, and converts
    to the acyloxy prefix form.

    Args:
        mol: RDKit Mol object
        ester_match: Tuple of atom indices from ester SMARTS match

    Returns:
        Acyloxy prefix string (e.g., "acetyloxy", "propanoyloxy"),
        or None if the ester is a lactone or cannot be named.

    Examples:
        For methyl acetate (COC(C)=O), returns "acetyloxy"
        For methyl propanoate (COC(=O)CC), returns "propanoyloxy"
    """
    # Lactones cannot be named as acyloxy prefixes
    if is_lactone(mol, ester_match):
        return None

    # Parse into acid and alkyl fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_match)

    if not acid_atoms:
        return None

    # Get the acid name from the fragment
    acid_name = get_acid_fragment_name(mol, acid_atoms)

    # Convert to acyloxy prefix
    return get_acyloxy_prefix(acid_name)


def detect_exocyclic_esters(mol) -> List[dict]:
    """
    Detect ester groups where the ester oxygen is bonded to a ring atom.

    These are "exocyclic" esters: the ring is the parent structure and
    the ester should be named as an acyloxy prefix on the ring.

    For example, cyclohexyl acetate (CC(=O)OC1CCCCC1) has an ester oxygen
    bonded to a cyclohexane ring carbon. The ester should be named as
    "acetyloxy" prefix on cyclohexane.

    Excludes lactones (cyclic esters where both the carbonyl C and alkyl C
    are in the same ring).

    Args:
        mol: RDKit Mol object

    Returns:
        List of dicts, each containing:
            - ester_match: tuple of atom indices from SMARTS match
            - ring_attach_atom_idx: index of the ring atom bonded to ester O
            - acyloxy_prefix: the acyloxy prefix string (e.g., "acetyloxy")

    Examples:
        For cyclohexyl acetate: returns [{"ester_match": (...),
            "ring_attach_atom_idx": 3, "acyloxy_prefix": "acetyloxy"}]
        For ethyl acetate: returns [] (no ring attachment)
    """
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    matches = mol.GetSubstructMatches(pattern)

    results = []
    for match in matches:
        # Skip lactones
        if is_lactone(mol, match):
            continue

        # match[2] is the ester oxygen, match[3] is the alkyl carbon
        # Check if the alkyl carbon (position 3) is in a ring
        alkyl_c_idx = match[3]
        alkyl_c_atom = mol.GetAtomWithIdx(alkyl_c_idx)

        if not alkyl_c_atom.IsInRing():
            continue

        # This is an exocyclic ester on a ring parent
        # Get the acyloxy prefix
        acyloxy = name_ester_as_prefix(mol, match)
        if acyloxy is None:
            continue

        results.append({
            "ester_match": match,
            "ring_attach_atom_idx": alkyl_c_idx,
            "acyloxy_prefix": acyloxy,
        })

    return results


# ============================================================================
# Multi-ester Detection and Naming (IUPAC P-65.6.3.3)
# ============================================================================
#
# Dicarboxylic acid diesters: two ester groups sharing a diacid backbone.
# Named as "[multiplier]alkyl [parent]anedioate".
#   e.g., COC(=O)CC(=O)OC -> "dimethyl propanedioate"
#
# Trivial diacid names used where available:
#   oxalic -> oxalate, malonic -> malonate, succinic -> succinate, etc.
# Otherwise systematic: get_chain_prefix(N) + "anedioate"


# Trivial diacid name -> trivial diacid ester suffix (dioate form)
TRIVIAL_DIACID_TO_DIOATE = {
    "oxalic": "oxalate",
    "malonic": "malonate",
    "succinic": "succinate",
    "glutaric": "glutarate",
    "adipic": "adipate",
    "phthalic": "phthalate",
    "fumaric": "fumarate",
    "maleic": "maleate",
}

# Backbone carbon count -> trivial diacid acid name
# Only use trivial names when IUPAC prefers them.
# Oxalic (2C) is the standard case; longer chains use systematic names
# per IUPAC 2013 preference for "propanedioate" over "malonate", etc.
BACKBONE_LENGTH_TO_TRIVIAL = {
    2: "oxalic",
}


def classify_multi_ester(mol, ester_matches: list) -> str:
    """
    Classify a multi-ester compound by its structural pattern.

    Args:
        mol: RDKit Mol object
        ester_matches: List of ester SMARTS match tuples
                       (carbonyl_c, carbonyl_o, ester_o, alkyl_c)

    Returns:
        Classification string:
        - "single": only one ester match
        - "dicarboxylic_diester": two esters sharing a diacid backbone
        - "polyol_polyester": multiple esters on a polyol (handled in plan 41-02)
        - "independent": multiple esters with no shared backbone
    """
    if len(ester_matches) < 2:
        return "single"

    if len(ester_matches) == 2:
        # Check if the two carbonyl carbons are connected via C-C bonds
        # (i.e., they share a dicarboxylic acid backbone)
        c1 = ester_matches[0][0]  # carbonyl carbon of first ester
        c2 = ester_matches[1][0]  # carbonyl carbon of second ester

        # Collect ester oxygen indices to exclude from BFS
        ester_oxygens = set()
        for match in ester_matches:
            ester_oxygens.add(match[2])  # ester oxygen (-O-)

        # BFS from c1 to c2, only following C-C bonds, excluding ester oxygens
        if _carbons_connected(mol, c1, c2, ester_oxygens):
            return "dicarboxylic_diester"

    # Polyol polyester detection: multiple ester oxygens connect to a shared
    # alcohol backbone (e.g., triacetin = glycerol triacetate).
    # For each ester match, the alkyl_c (match[3]) is a backbone carbon.
    # If all backbone carbons are connected via C-C bonds -> polyol polyester.
    backbone_carbons = []
    ester_oxygens = set()
    carbonyl_carbons = set()
    for match in ester_matches:
        ester_oxygens.add(match[2])  # ester oxygen (-O-)
        carbonyl_carbons.add(match[0])  # carbonyl carbon
        backbone_carbons.append(match[3])  # alkyl carbon (backbone side)

    # All backbone carbons must be connected via carbon-only paths
    # excluding ester oxygens and carbonyl carbons
    exclude = ester_oxygens | carbonyl_carbons
    if len(backbone_carbons) >= 2:
        all_connected = True
        for i in range(1, len(backbone_carbons)):
            if not _carbons_connected(mol, backbone_carbons[0], backbone_carbons[i], exclude):
                all_connected = False
                break
        if all_connected:
            return "polyol_polyester"

    return "independent"


def _carbons_connected(mol, start: int, target: int, exclude_atoms: set) -> bool:
    """
    Check if two atoms are connected via a path of carbon atoms only.

    BFS from start to target, only traversing C-C single/double bonds,
    excluding specified atoms (ester oxygens).

    Args:
        mol: RDKit Mol object
        start: Starting atom index
        target: Target atom index
        exclude_atoms: Set of atom indices to exclude from traversal

    Returns:
        True if start and target are connected via carbon-only path
    """
    visited = {start}
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current == target:
            return True

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in visited or nbr_idx in exclude_atoms:
                continue
            # Only follow carbon atoms
            if neighbor.GetSymbol() != 'C':
                continue
            visited.add(nbr_idx)
            queue.append(nbr_idx)

    return False


def _find_backbone_carbons(mol, c1: int, c2: int, ester_oxygens: set) -> Optional[List[int]]:
    """
    Find the shortest carbon-only path between two carbonyl carbons.

    Uses BFS to find the backbone of a dicarboxylic acid diester.
    The backbone includes both carbonyl carbons.

    Args:
        mol: RDKit Mol object
        c1: First carbonyl carbon index
        c2: Second carbonyl carbon index
        ester_oxygens: Set of ester oxygen indices to exclude

    Returns:
        List of atom indices forming the backbone path (c1 to c2),
        or None if no carbon-only path exists.
    """
    # BFS for shortest path
    visited = {c1}
    queue = [(c1, [c1])]

    while queue:
        current, path = queue.pop(0)
        if current == c2:
            return path

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in visited or nbr_idx in ester_oxygens:
                continue
            if neighbor.GetSymbol() != 'C':
                continue
            visited.add(nbr_idx)
            queue.append((nbr_idx, path + [nbr_idx]))

    return None


def name_dicarboxylic_diester(mol, ester_matches: list) -> Optional[str]:
    """
    Name a dicarboxylic acid diester compound.

    Produces names in the format "[multiplier]alkyl [parent]anedioate".
    If the two alkyl groups differ, they are listed in alphabetical order.

    Args:
        mol: RDKit Mol object
        ester_matches: List of exactly 2 ester SMARTS match tuples

    Returns:
        Name string (e.g., "dimethyl propanedioate"), or None on failure.
    """
    if len(ester_matches) != 2:
        return None

    # Collect ester oxygens
    ester_oxygens = set()
    for match in ester_matches:
        ester_oxygens.add(match[2])

    # Get alkyl names for each ester
    alkyl_names = []
    for match in ester_matches:
        _acid_atoms, alkyl_atoms = parse_ester_fragments(mol, match)
        if not alkyl_atoms:
            return None
        alkyl_name = get_alkyl_fragment_name(mol, alkyl_atoms)
        if not alkyl_name:
            return None
        alkyl_names.append(alkyl_name)

    # Find backbone between the two carbonyl carbons
    c1 = ester_matches[0][0]
    c2 = ester_matches[1][0]
    backbone = _find_backbone_carbons(mol, c1, c2, ester_oxygens)
    if backbone is None:
        return None

    backbone_length = len(backbone)

    # Get the dioate name
    dioate_name = _get_dioate_name(backbone_length)

    # Assemble the name
    if alkyl_names[0] == alkyl_names[1]:
        # Same alkyl groups: use multiplier
        return f"di{alkyl_names[0]} {dioate_name}"
    else:
        # Different alkyl groups: alphabetical order
        sorted_names = sorted(alkyl_names)
        return f"{sorted_names[0]} {sorted_names[1]} {dioate_name}"


def _get_dioate_name(backbone_length: int) -> str:
    """
    Get the dioate suffix name for a dicarboxylic acid ester.

    Checks trivial diacid names first, then falls back to systematic.

    Args:
        backbone_length: Number of carbons in the diacid backbone
                         (including both carbonyl carbons)

    Returns:
        Dioate name (e.g., "oxalate", "propanedioate", "hexanedioate")
    """
    # Check for trivial diacid name
    trivial_acid = BACKBONE_LENGTH_TO_TRIVIAL.get(backbone_length)
    if trivial_acid and trivial_acid in TRIVIAL_DIACID_TO_DIOATE:
        return TRIVIAL_DIACID_TO_DIOATE[trivial_acid]

    # Systematic: get_chain_prefix(N) + "anedioate"
    prefix = get_chain_prefix(backbone_length)
    return f"{prefix}anedioate"


# ============================================================================
# Polyol Polyester Naming (IUPAC P-65.6.3.2.2)
# ============================================================================
#
# Fully-esterified polyols (triacetin, triglycerides) are named using
# acyloxy prefixes on the polyol backbone:
#   "locants-multiplier(acyloxy)parent"
#
# Examples:
#   triacetin -> "1,2,3-tri(acetyloxy)propane"
#   mixed triester -> "1,3-di(acetyloxy)-2-(propanoyloxy)propane"


def name_polyol_polyester(mol, ester_matches: list) -> Optional[str]:
    """
    Name a fully-esterified polyol compound using acyloxy prefixes.

    Identifies the polyol backbone, determines acyloxy prefixes for each
    ester group, and assembles the name with locants and multipliers.

    Args:
        mol: RDKit Mol object
        ester_matches: List of ester SMARTS match tuples
                       (carbonyl_c, carbonyl_o, ester_o, alkyl_c)

    Returns:
        Name string (e.g., "1,2,3-tri(acetyloxy)propane"), or None on failure.
    """
    from ..assembly.naming_utils import get_multiplier_prefix

    if len(ester_matches) < 2:
        return None

    # Collect structural data from each ester match
    ester_oxygens = set()
    carbonyl_carbons = set()
    for match in ester_matches:
        ester_oxygens.add(match[2])
        carbonyl_carbons.add(match[0])

    # Find backbone: BFS from backbone carbons through C-C bonds only,
    # excluding ester oxygens and carbonyl carbons
    exclude = ester_oxygens | carbonyl_carbons
    backbone_start_atoms = [match[3] for match in ester_matches]  # alkyl_c per ester

    # BFS to find full backbone
    backbone = _find_polyol_backbone(mol, backbone_start_atoms, exclude)
    if backbone is None or len(backbone) < 2:
        return None

    # Order backbone as a linear chain (find two endpoints with degree 1 within backbone)
    ordered = _order_backbone_chain(mol, backbone, exclude)
    if ordered is None:
        return None

    backbone_length = len(ordered)
    parent_prefix = get_chain_prefix(backbone_length)
    if not parent_prefix:
        return None
    parent_name = parent_prefix + "ane"

    # Map backbone atom index -> 1-based locant
    locant_map = {atom_idx: i + 1 for i, atom_idx in enumerate(ordered)}

    # For each ester, get the acyloxy prefix and the attachment locant
    ester_info = []  # list of (locant, acyloxy_prefix)
    for match in ester_matches:
        carbonyl_c = match[0]
        ester_o = match[2]
        alkyl_c = match[3]  # backbone attachment point

        # Get acid fragment for this ester
        acid_atoms = _bfs_fragment(mol, carbonyl_c, exclude_atom=ester_o)
        acid_name = get_acid_fragment_name(mol, acid_atoms)
        acyloxy = get_acyloxy_prefix(acid_name)

        # Get locant for attachment
        locant = locant_map.get(alkyl_c)
        if locant is None:
            return None
        ester_info.append((locant, acyloxy))

    # Group by acyloxy prefix
    from collections import defaultdict
    groups = defaultdict(list)
    for locant, acyloxy in ester_info:
        groups[acyloxy].append(locant)

    # Sort groups alphabetically by acyloxy prefix name
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])

    # Build prefix parts
    parts = []
    for acyloxy, locants in sorted_groups:
        locants.sort()
        locant_str = ",".join(str(loc) for loc in locants)
        count = len(locants)
        multiplier = get_multiplier_prefix(count, acyloxy)
        # Acyloxy prefixes always get parentheses when used with multipliers
        # or when they contain internal structure
        if count > 1:
            parts.append(f"{locant_str}-{multiplier}({acyloxy})")
        else:
            parts.append(f"{locant_str}-({acyloxy})")

    # Check if we can simplify by trying lowest locants numbering
    # Try reverse numbering and pick whichever gives lower first-point-of-difference
    reverse_locant_map = {atom_idx: backbone_length - i
                          for i, atom_idx in enumerate(ordered)}
    reverse_info = []
    for match in ester_matches:
        alkyl_c = match[3]
        locant = reverse_locant_map.get(alkyl_c)
        if locant is None:
            break
        # Reuse same acyloxy prefix
        acid_atoms = _bfs_fragment(mol, match[0], exclude_atom=match[2])
        acid_name = get_acid_fragment_name(mol, acid_atoms)
        acyloxy = get_acyloxy_prefix(acid_name)
        reverse_info.append((locant, acyloxy))

    if len(reverse_info) == len(ester_info):
        # Compare locant sets for first-point-of-difference
        forward_locants = sorted(loc for loc, _ in ester_info)
        reverse_locants = sorted(loc for loc, _ in reverse_info)
        if reverse_locants < forward_locants:
            # Use reverse numbering
            groups_rev = defaultdict(list)
            for locant, acyloxy in reverse_info:
                groups_rev[acyloxy].append(locant)

            sorted_groups_rev = sorted(groups_rev.items(), key=lambda x: x[0])
            parts = []
            for acyloxy, locants in sorted_groups_rev:
                locants.sort()
                locant_str = ",".join(str(loc) for loc in locants)
                count = len(locants)
                multiplier = get_multiplier_prefix(count, acyloxy)
                if count > 1:
                    parts.append(f"{locant_str}-{multiplier}({acyloxy})")
                else:
                    parts.append(f"{locant_str}-({acyloxy})")

    # Assemble: join parts with hyphens, append parent name
    prefix_str = "-".join(parts)
    return f"{prefix_str}{parent_name}"


def _find_polyol_backbone(mol, start_atoms: list, exclude: set) -> Optional[set]:
    """
    BFS from backbone start atoms through C-C bonds to find the full backbone.

    Args:
        mol: RDKit Mol object
        start_atoms: List of atom indices that are known backbone carbons
        exclude: Set of atom indices to exclude (ester oxygens, carbonyl carbons)

    Returns:
        Set of backbone atom indices, or None if not all start atoms connected.
    """
    if not start_atoms:
        return None

    visited = set()
    queue = list(start_atoms)
    visited.update(start_atoms)

    while queue:
        current = queue.pop(0)
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in visited or nbr_idx in exclude:
                continue
            if neighbor.GetSymbol() != 'C':
                continue
            visited.add(nbr_idx)
            queue.append(nbr_idx)

    # Verify all start atoms are in the connected set
    for sa in start_atoms:
        if sa not in visited:
            return None

    return visited


def _order_backbone_chain(mol, backbone: set, exclude: set) -> Optional[list]:
    """
    Order backbone atoms as a linear chain from one end to the other.

    Finds endpoint atoms (those with only 1 backbone neighbor) and
    traverses from one endpoint to the other.

    Args:
        mol: RDKit Mol object
        backbone: Set of backbone atom indices
        exclude: Set of atom indices to exclude from neighbor counting

    Returns:
        Ordered list of backbone atom indices, or None if not a linear chain.
    """
    # Build adjacency within backbone
    adj = {idx: [] for idx in backbone}
    for idx in backbone:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in backbone:
                adj[idx].append(nbr_idx)

    # Find endpoints (degree 1 in backbone)
    endpoints = [idx for idx in backbone if len(adj[idx]) == 1]
    if len(endpoints) < 2:
        # Not a linear chain (could be cyclic backbone or single atom)
        if len(backbone) == 1:
            return list(backbone)
        return None

    # Traverse from first endpoint
    ordered = []
    visited = set()
    current = endpoints[0]
    while current is not None:
        ordered.append(current)
        visited.add(current)
        next_atom = None
        for nbr in adj[current]:
            if nbr not in visited:
                next_atom = nbr
                break
        current = next_atom

    if len(ordered) != len(backbone):
        return None  # Not a simple linear chain

    return ordered
