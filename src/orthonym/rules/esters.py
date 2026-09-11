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

import logging
from collections import deque
from typing import List, Optional, Tuple

from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name
from ..data.chain_names import get_acid_stem, get_chain_prefix
from ..data.chain_names import get_alkyl_name as _chain_alkyl_name
from ..data.trivial_acids import get_acylate_name

logger = logging.getLogger(__name__)


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
    queue = deque([start_atom])

    while queue:
        current = queue.popleft()
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


def acid_is_ring_acid(mol, acid_atoms: List[int]) -> bool:
    """True iff the acid's carbonyl carbon is DIRECTLY bonded to a ring atom.

     task 9: only then do the ring-acid forms apply (benzoic,
    cyclohexanecarboxylic —. An acid fragment that merely CONTAINS
    a ring further down the chain (cyclohexyl-CH2CH2CH2-COO-) is a CHAIN
    acid with a ring substituent ('4-cyclohexylbutanoate'); naming it
    'cyclohexanecarboxylate' described a different molecule.
    """
    acid_set = set(acid_atoms)
    ring_info = mol.GetRingInfo()
    for idx in acid_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        is_carbonyl = any(
            b.GetOtherAtom(atom).GetSymbol() == 'O'
            and b.GetBondTypeAsDouble() == 2.0
            and b.GetOtherAtom(atom).GetIdx() in acid_set
            for b in atom.GetBonds()
        )
        if not is_carbonyl:
            continue
        if ring_info.NumAtomRings(idx) > 0:
            return True  # lactone-like / carbonyl in ring
        return any(
            nbr.GetIdx() in acid_set and nbr.IsInRing()
            for nbr in atom.GetNeighbors()
        )
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
    from ..data import ALL_RETAINED_NAMES as RETAINED_NAMES
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
    # Check if acid fragment IS a ring acid (carbonyl C bonded to the ring —
    # task 9: a ring merely elsewhere in the fragment is a chain acid
    # with a ring substituent, never 'Xcarboxylic').
    if acid_is_ring_acid(mol, acid_atoms):
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

    # Fatty acid trivial names by (carbon_count, double_bond_count).
    #
    # "Retained names as preferred IUPAC names" (the Blue Book)
    # -- "Only the following five carboxylic acids retained names and are also
    # preferred IUPAC names": formic, oxalic, acetic, benzoic, oxamic. No fatty
    # acid is among them.
    #
    # "Systematic names" (heading:29858) states the disposal rule at
    #:29860 -- "Except for formic acid, acetic acid, oxalic acid (see
    #, and oxamic acid (see, systematically formed names
    # are preferred IUPAC names; the names given in are retained
    # names for use in general nomenclature."
    #
    # The Blue Book prints the (PIN) marker on the SYSTEMATIC name in every
    # fatty row it lists --:29787 "palmitic acid hexadecanoic acid (PIN)",
    #:29791 "stearic acid octadecanoic acid (PIN)".
    #
    # This stem feeds the ester acyl word, and (:31659) -- "All
    # preferred IUPAC names for esters are named by functional class
    # nomenclature" -- takes that word from the PIN acid, e.g. "ethyl acetate
    # (PIN)" (acetic IS retained) but "ethyl methyl butanedioate (PIN)" (NOT
    # succinate). So the saturated straight-chain rows C12/C14/C16/C18/C20 were
    # REMOVED (Task J2): they made the ester path emit 'ethyl palmitate' while the
    # acid path for the same chain already emitted the PIN 'hexadecanoic acid'.
    # Falling through to get_acid_stem below yields the PIN stem.
    #
    # Task J3 removed the four remaining (unsaturated) rows -- (18,1) 'oleic',
    # (18,2) 'linoleic', (18,3) 'linolenic', (20,4) 'arachidonic' -- so the whole
    # table is gone. This stem ALSO feeds the acyl PREFIX via
    # get_acyloxy_prefix, and "Esters cited as prefixes" (:31696)
    # settles that position directly. Its worked examples straddle the boundary:
    #:31711 3-(benzoyloxy)propanoic acid (PIN)
    # -- benzoic IS retained as a PIN, so 'benzoyloxy' is
    # preferred and MUST survive;
    #:31723 3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)
    # 3-(nicotinoyloxy)propanoic acid
    # -- nicotinic acid is retained for GENERAL nomenclature only
    # list, 'nicotinic acid pyridine-3-carboxylic acid
    # (PIN)' at:29773), and the trivial-derived acyloxy prefix is
    # printed as the NON-preferred alternative.
    # 'oleic acid' sits in that same general-only list (:29785, 'oleic acid
    # (9Z)-octadec-9-enoic acid (PIN)'), so 'oleoyloxy' is non-PIN for the same
    # reason 'nicotinoyloxy' is. Appendix 2 confirms per row -- its legend
    # (:55416) reads "The symbol * designates the preferred prefix", and it
    # prints '(9Z)-octadec-9-enoyl* = oleoyl' (:56443), 'hexadecanoyl* =
    # palmitoyl' (:56482), 'octadecanoyl* = stearoyl' (:56441).
    #
    # J2 deferred these four rows fearing that an unsaturated acid falling
    # through would be caught by the SATURATED get_acid_stem and become a WRONG
    # MOLECULE. Measured: it is not. The unsaturation branch immediately below
    # runs FIRST and returns the full systematic stem with double-bond locants and
    # E/Z descriptors -- '(9Z)-octadec-9-enoic', '(9Z,12Z)-octadeca-9,12-dienoic',
    # '(9Z,12Z,15Z)-octadeca-9,12,15-trienoic',
    # '(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoic' -- which are verbatim the Blue
    # Book PINs. get_acid_stem is reached for an unsaturated acid only if that
    # branch returns falsy, which was already true for every unsaturated chain
    # outside the four deleted rows, so the rows were never the guard.
    # See internal notes

    # For unsaturated acids, extract the
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
    the attachment point. Because we copy the molecule rather than
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
    stereochemistry. Caps the carbonyl carbon with -OH to form a
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
    stereochemistry. Caps the attachment carbon with -OH to form an
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


def _ester_attachment_atom(mol, alkyl_set: set) -> Optional[int]:
    """The alkyl-fragment atom bonded to the ester chalcogen.

    `parse_ester_fragments` cuts at the ester O (or S/Se/Te), so the fragment
    has exactly one atom with a neighbour outside it. If that is not true the
    caller was handed something other than a clean alcohol component — return
    None and let the legacy path decide rather than guessing an attachment.
    """
    external = [
        idx for idx in sorted(alkyl_set)
        if any(nbr.GetIdx() not in alkyl_set
               for nbr in mol.GetAtomWithIdx(idx).GetNeighbors())
    ]
    return external[0] if len(external) == 1 else None


def _alkyl_name_via_substituent_primitive(mol, alkyl_set: set) -> Optional[str]:
    """Name the alcohol component with the centralized substituent primitive.

    Returns the raw organyl word (`2-methylpropyl`, `cyclohexyl`,
    `(2S)-butan-2-yl`) or None when the primitive declines — including when it
    declines a ring-bearing fragment, which it does deliberately rather than
    anchoring a free valence it cannot place (see the ring_fragment_declined_by_ring_engine guard in
    assembly/substituent_naming.py).
    """
    attach = _ester_attachment_atom(mol, alkyl_set)
    if attach is None:
        return None
    try:
        from ..assembly.substituent_naming import name_substituent_fragment
        parent = sorted(set(range(mol.GetNumAtoms())) - alkyl_set)
        word = name_substituent_fragment(
            mol, sorted(alkyl_set), attach, parent
        ) or None
        if word:
            return word
    except Exception:
        # The primitive is a large recursive surface; a failure here must
        # degrade to the legacy word, never break ester naming outright.
        word = None

    # breadth: name_substituent_fragment DECLINES a ring-bearing alcohol
    # component (its ring_fragment_declined_by_ring_engine guard). The legacy count path below then LINEARISES
    # the ring -- `CC(=O)O[C@H]1CCCCC[C@@H]1O` came out `(1S,2S)-heptyl acetate`
    # (cycloheptane counted as 7 chain carbons, the -OH silently dropped): a
    # WRONG molecule that suppresses into a silent abstention. The
    # recursive substituent namer (C4 keystone) names a decorated ring correctly
    # (`(1S,2S)-2-hydroxycycloheptyl`). Fail-closed: this only runs after the old
    # primitive already declined, so it can convert a currently-abstaining ester
    # or stay declined -- it can never regress a name the old path emitted.
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in alkyl_set):
        try:
            from ..assembly.substituent_enumerator import name_substituent
            ring_word = name_substituent(mol, sorted(alkyl_set), attach)
            if ring_word and ring_word != "substituent":
                return ring_word
        except Exception:
            return None
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

    # task 9: the ring-branch shortcuts below ('benzyl',
    # 'n-phenylalkyl') and the carbon-count tail name SATURATED carriers
    # only — an unsaturated non-ring carbon makes every count-based form
    # describe a DIFFERENT molecule ((2E)-prop-2-enyl emitted as 'propyl',
    # canary rt75_0430 RT True->False once the ester path was unlocked).
    # Refuse and let name_ester defer to the general pipeline.
    _has_nonring_unsat = False
    for idx in alkyl_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.IsInRing():
            continue
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if (nbr.GetIdx() in alkyl_set
                    and not bond.GetIsAromatic()
                    and bond.GetBondTypeAsDouble() != 1.0):
                _has_nonring_unsat = True
                break
        if _has_nonring_unsat:
            break
    if has_ring and _has_nonring_unsat:
        return ""

    # --- a phase: delegate to the centralized substituent primitive -----
    # Everything below this point derives the organyl word from a CARBON COUNT
    # plus a branch check that only looks at the attachment carbon. Branching
    # anywhere else is silently lost, so CC(=O)OCC(C)C (isobutyl) came out
    # 'butyl' and CC(=O)OC1CCCCC1 came out 'hexyl' — different molecules, saved
    # only by suppressing them into an abstention.
    #
    # name_substituent_fragment is the project's documented "centralized entry
    # point for all substituent naming" and handles branching, rings,
    # unsaturation, attachment position and stereo. Measured over 20 acetate
    # esters: 12 words identical, 8 different, and in all 8 the count-based
    # word was the WRONG MOLECULE while the primitive's was right.
    #
    # The legacy path stays as the fallback for fragments the primitive
    # declines, so nothing it already named correctly can regress.
    _primitive = _alkyl_name_via_substituent_primitive(mol, alkyl_set)
    if _primitive:
        return _primitive

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


def _find_acid_principal_chain(mol, acid_atoms: List[int]) -> Optional[List[int]]:
    """Find the principal (longest) carbon chain in the acid fragment.

    Starts from the carbonyl carbon and finds the longest path through
    carbon atoms in the acid fragment. This correctly identifies the acid
    chain length even when the acid fragment is branched.

    Args:
        mol: RDKit Mol object.
        acid_atoms: Atom indices of the acid fragment.

    Returns:
        List of atom indices forming the principal chain (carbonyl C first),
        or None if carbonyl carbon cannot be found.
    """
    acid_set = set(acid_atoms)

    # Find the carbonyl carbon (C with =O bond in acid set)
    carbonyl_c = None
    carbonyl_o = None
    for idx in acid_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if (nbr.GetSymbol() == 'O'
                    and bond.GetBondTypeAsDouble() == 2.0
                    and nbr.GetIdx() in acid_set):
                carbonyl_c = idx
                carbonyl_o = nbr.GetIdx()
                break
        if carbonyl_c is not None:
            break

    if carbonyl_c is None:
        return None

    # BFS/DFS to find the longest carbon chain from carbonyl_c.
    # task 9: a parent CHAIN never runs through ring atoms — rings
    # attach as substituents (the walk previously absorbed a cyclohexyl
    # into the "chain", mis-counting the acid length).
    acid_carbons = {idx for idx in acid_atoms
                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                    and not mol.GetAtomWithIdx(idx).IsInRing()}
    if carbonyl_c not in acid_carbons:
        return None

    best_path = [carbonyl_c]
    queue = deque([(carbonyl_c, [carbonyl_c])])
    while queue:
        curr, path = queue.popleft()
        atom = mol.GetAtomWithIdx(curr)
        extended = False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if (nidx in acid_carbons
                    and nidx not in set(path)
                    and nbr.GetSymbol() == 'C'):
                queue.append((nidx, path + [nidx]))
                extended = True
        if not extended and len(path) > len(best_path):
            best_path = path

    return best_path


def _name_amino_acid_ester(
    mol, acid_atoms: List[int], alkyl_atoms: List[int], ester_match: tuple,
) -> Optional[str]:
    """ (BB 54595-54608): name an amino-acid ester with the retained
    '-ate' stem, e.g. 'methyl L-alaninate' (BB 54601).

    Scope (accuracy-safe subset): the single-alpha-stereocentre monocarboxylic
    standard amino acids + glycine (``AMINO_ACID_ATE_STEMS``). Returns None
    (falls through to the pre-existing systematic ester name -- no
    regression) for:
      - diacid AAs (aspartic/glutamic -- need positional ester locants like
        '1-methyl L-aspartate', BB 54606),
      - 2-stereocentre AAs (threonine/isoleucine -- allo descriptor
        entanglement),
      - anything whose acid fragment does not reconstruct to a bare retained
        amino acid.

    Descriptor policy: the ester stem uses an EXPLICIT L-/D- descriptor (BB
    54601 shows 'methyl L-alaninate', unlike the bare AA which suppresses
    implicit L). Glycine (achiral) gets no descriptor.
    """
    carbonyl_c = ester_match[0]

    # Reconstruct the acid fragment as a standalone carboxylic acid molecule
    # (restore -OH on the carbonyl carbon) using the existing RWMol fragment
    # helper -- preserves stereo, no hand-rolled SMILES surgery.
    frag_smi = _extract_fragment_smiles(mol, set(acid_atoms), carbonyl_c, cap_element=8)
    if not frag_smi:
        return None

    acid_mol = Chem.MolFromSmiles(frag_smi)
    if acid_mol is None:
        return None

    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(acid_mol)

    # Stereo-free canonical match against the retained amino-acid tables.
    try:
        nostereo_smi = Chem.MolToSmiles(acid_mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None

    from ..data.amino_acids import get_amino_acid_ate_stem, get_amino_acid_name
    aa_name = get_amino_acid_name(nostereo_smi, mol=acid_mol, with_descriptor=False)
    if aa_name is None:
        return None  # acid fragment is not a bare retained amino acid

    ate_stem = get_amino_acid_ate_stem(aa_name)
    if ate_stem is None:
        return None  # out of scope (diacid / 2-stereocentre / non-standard AA)

    # a phase (stereo honesty, mirrors L3-2e + the amino_acids.py
    # standalone-AA fix): `_get_stereo_prefix` falls back to "" both for TRUE
    # achirality (glycine) and for a genuine alpha-carbon stereocentre the
    # INPUT never defines (no wedge/parity). Emitting the bare '-ate' stem in
    # that case (e.g. 'methyl alaninate') is NOT the achiral-glycine case this
    # docstring's "" carve-out describes -- it silently asserts the implicit-L
    # convention on a structure that does not define it (`## ****
    # The stereodescriptors 'D' and 'L'`, the Blue Book: "The
    # stereodescriptor 'xi'... indicates unknown configuration"). DECLINE
    # (return None) so the caller falls through to the ordinary systematic
    # ester path -- no regression, see `name_ester`'s docstring at:1024-1026.
    from .peptides import _alpha_stereo_undefined, _get_stereo_prefix
    if _alpha_stereo_undefined(acid_mol, aa_name):
        return None
    # Explicit alpha-carbon descriptor. Reuses the peptide stereo-prefix logic
    # (already returns "L-"/"D-"/"" -- "" only for achiral glycine) -- no new
    # CIP code.
    descriptor = _get_stereo_prefix(acid_mol, aa_name)

    r_prime = get_alkyl_fragment_name(mol, alkyl_atoms)
    if not r_prime:
        return None

    return f"{r_prime} {descriptor}{ate_stem}"


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

    # W8 P3, BB 54595-54608): amino-acid esters use the
    # retained '-ate' stem (e.g. 'methyl L-alaninate') for the in-scope
    # single-alpha-stereocentre monocarboxylic standard AAs + glycine. Tried
    # BEFORE the normal alkanoate/ring-acid path; declines (returns None,
    # falls through with no regression) for anything out of scope (diacid
    # AAs, 2-stereocentre AAs, non-standard AAs, non-AA esters).
    aa_ester_name = _name_amino_acid_ester(mol, acid_atoms, alkyl_atoms, ester_match)
    if aa_ester_name is not None:
        return aa_ester_name

    # / (BB 18114, W2E-P1FC Task 7): a mono-ester of carbonic
    # acid whose OTHER acid function is an acyl HALIDE, X-C(=O)-O-R, is named as
    # the functional-class '<R> carbono<halide>idate' (benzyl carbonochloridate).
    # This is perceived as ester+acid_halide on the SAME carbonyl; the acid-side
    # carbonyl bears a halide and NO carbon. Delegate to the carbono-halidate
    # namer BEFORE the ordinary alkanoate path (which would mis-name the -C(=O)Cl
    # as a '1-chloro-1-oxo' substituent). Fail-closed: only fires on that exact
    # skeleton, else falls through to the normal ester logic.
    _acyl_c = ester_match[0]
    _c_at = mol.GetAtomWithIdx(_acyl_c)
    if _c_at.GetSymbol() == 'C':
        _hal = next((nb for nb in _c_at.GetNeighbors()
                     if nb.GetSymbol() in ('Cl', 'Br', 'F', 'I')), None)
        _has_c = any(nb.GetSymbol() == 'C' for nb in _c_at.GetNeighbors())
        if _hal is not None and not _has_c:
            from .acid_halides import (
                HALIDE_WORDS,
                name_carbonic_monoester_acyl_halide,
            )
            _hal_pg = {'Cl': 'acid_chloride', 'Br': 'acid_bromide',
                       'F': 'acid_fluoride', 'I': 'acid_iodide'}[_hal.GetSymbol()]
            _o_dbl = next((nb.GetIdx() for nb in _c_at.GetNeighbors()
                           if nb.GetSymbol() == 'O'
                           and mol.GetBondBetweenAtoms(
                               _acyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0),
                          None)
            if _o_dbl is not None:
                _cc = name_carbonic_monoester_acyl_halide(
                    mol, (_acyl_c, _o_dbl, _hal.GetIdx()),
                    HALIDE_WORDS[_hal_pg],
                )
                if _cc is not None:
                    return _cc

    # Guard: a true RING ACID (carbonyl bonded to the ring) whose ring
    # naming fails defers to complex naming. task 9: an acid that
    # merely CONTAINS a ring down-chain is a chain acid with a ring
    # substituent — it takes the chain path below.
    #
    # a phase (C): anchored on the KNOWN carbonyl carbon
    # (``ester_match[0]``) via ``_carbonyl_is_ring_bonded``, never on a
    # BFS-scanned ``acid_atoms`` set -- see that helper's docstring for why
    # ``acid_is_ring_acid`` can wander into a second ester's atoms and
    # return an order-dependent answer for the SAME molecule.
    if _carbonyl_is_ring_bonded(mol, ester_match[0]):
        ring_acid_name = get_ring_acid_name(mol, acid_atoms)
        if ring_acid_name is None:
            # Complex ring acid (fused heterocycle etc.) - defer to complex naming
            return None

    # Ensure CIP labels are assigned (idempotent guard)
    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)

    # a phase: the acid-side word (chain-vs-ring selection, acid-side
    # substituent discovery, and R/S stereo citation) is built by the shared
    # primitive so `name_polyfunctional_diester_free_hydroxy` (the
    # diacylglycerol-shape multi-ester path) builds it identically for
    # whichever ester it selects as the principal/senior acid.
    acylate_name = _build_ester_acid_word(mol, ester_match, acid_atoms, alkyl_atoms)
    if acylate_name is None:
        return None

    # Get alkyl name
    alkyl_name = get_alkyl_fragment_name(mol, alkyl_atoms)

    if not alkyl_name:
        return None

    #: Collect alkyl-side (alcohol fragment) stereo descriptors.
    # Skip if the alkyl name already contains a stereo prefix.
    import re as _re

    from .stereochemistry import format_stereodescriptor_string as _fmt_stereo
    if not _re.match(r'^\(\d*[RSrsEZez](,\d*[RSrsEZez])*\)', alkyl_name):
        alkyl_stereo = _collect_alkyl_fragment_stereo(mol, alkyl_atoms, ester_match)
        alkyl_stereo = [(loc, cip) for loc, cip in alkyl_stereo if cip in ('R', 'S')]
        if alkyl_stereo:
            alkyl_stereo_prefix = _fmt_stereo(alkyl_stereo)
            alkyl_name = f"{alkyl_stereo_prefix}{alkyl_name}"

    # Combine: "alkyl acylate"
    return f"{alkyl_name} {acylate_name}"


def _build_ester_acid_word(
    mol, ester_match: tuple, acid_atoms: List[int], alkyl_atoms: List[int],
) -> Optional[str]:
    """Build the acid-side word ('<stereo><prefixes><acid>oate') for ONE
    ester match: the ester's suffix half).

     a phase: extracted verbatim from ``name_ester`` (which used to build
    this inline) so it is a SINGLE source of truth, shared by ``name_ester``
    itself and by ``name_polyfunctional_diester_free_hydroxy`` (the
    diacylglycerol-shape path, which builds the acid word for whichever of
    its two esters is selected as the senior/principal acid, then names the
    OTHER ester as an acyloxy prefix on the alkyl side instead of calling
    this function a second time).

    Returns None (fail-closed) when the acid side cannot be named — a
    complex ring acid whose ring-naming fails, or an acid-side substituent
    the universal prefix pipeline cannot discover.
    """
    # Ensure CIP labels are assigned (idempotent guard)
    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)

    # --- a phase-02: Acid-side substituent discovery via universal pipeline ---
    # Find the principal chain in the acid fragment to correctly identify
    # chain length (excluding branch carbons) and discover substituents.
    acid_set = set(acid_atoms)
    # task 9: chain-vs-ring acid naming is decided by the CARBONYL
    # bond, not mere ring presence (see acid_is_ring_acid). a phase (C):
    # anchored on the KNOWN carbonyl carbon (``ester_match[0]``), never a
    # BFS-scanned ``acid_atoms`` set -- see ``_carbonyl_is_ring_bonded``.
    acid_has_ring = _carbonyl_is_ring_bonded(mol, ester_match[0])
    acid_principal_chain = None
    acid_prefix_str = ""
    ring_locant_for_suffix = None

    if not acid_has_ring:
        # --- Chain acid path (existing logic, unchanged) ---
        acid_principal_chain = _find_acid_principal_chain(mol, acid_atoms)

        if acid_principal_chain and len(acid_principal_chain) >= 2:
            # Check if acid fragment has branches (substituents on the acid chain).
            # Branches can be carbon-based (alkyl) OR heteroatom-based (halogens,
            # hydroxy, etc.). Check for ANY non-chain, non-carbonyl-O atom.
            chain_set = set(acid_principal_chain)
            carbonyl_o_set = set()
            for idx in acid_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'O':
                    for bond in atom.GetBonds():
                        if bond.GetBondTypeAsDouble() == 2.0:
                            carbonyl_o_set.add(idx)
                            break
            non_chain_atoms = acid_set - chain_set - carbonyl_o_set
            has_acid_branches = bool(non_chain_atoms)

            if has_acid_branches:
                # Use principal chain length for the acid name (not total carbon count)
                chain_length = len(acid_principal_chain)
                acid_name = get_acid_stem(chain_length)

                # Exclude atoms: carbonyl O(s), ester O, alkyl fragment
                ester_o = ester_match[2] if len(ester_match) > 2 else None
                exclude = set(carbonyl_o_set)
                if ester_o is not None:
                    exclude.add(ester_o)
                exclude.update(set(alkyl_atoms))

                # Use universal pipeline for acid-side substituents
                try:
                    from ..assembly.composer import _integrate_universal_prefixes
                    acid_prefix_str = _integrate_universal_prefixes(
                        mol, chain_set,
                        parent_type="chain",
                        principal_chain=acid_principal_chain,
                        exclude_atoms=exclude,
                    )
                except Exception as exc:
                    logger.debug("Ester acid-side prefix discovery failed: %s", exc)
                    acid_prefix_str = ""
            else:
                # No branches -- use standard acid naming
                acid_name = get_acid_fragment_name(mol, acid_atoms)
        else:
            # Short chain -- use standard acid naming
            acid_name = get_acid_fragment_name(mol, acid_atoms)
    else:
        # --- /: Ring acid path (NEW) ---
        # `name_ester`'s own early guard already verified
        # get_ring_acid_name(mol, acid_atoms) is non-None before calling this
        # helper (it returns None outright otherwise); recomputed here so
        # this function is self-contained for any other caller.
        ring_acid_name = get_ring_acid_name(mol, acid_atoms)
        if ring_acid_name is None:
            return None
        acid_name = ring_acid_name

        # Discover substituents on the ring atoms using universal pipeline
        ring_info = mol.GetRingInfo()
        ring_atoms_in_acid = []
        for ring in ring_info.AtomRings():
            ring_set_local = set(ring)
            if ring_set_local.issubset(acid_set):
                ring_atoms_in_acid = list(ring)
                break  # Use first complete ring in acid fragment

        if ring_atoms_in_acid:
            # Orient ring: position 1 = ring atom bonded to carbonyl C
            carbonyl_c = ester_match[0]  # carbonyl carbon index
            start_atom = None
            for idx in ring_atoms_in_acid:
                for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nbr.GetIdx() == carbonyl_c:
                        start_atom = idx
                        break
                if start_atom is not None:
                    break

            if start_atom is not None:
                # a phase (C): try BOTH ring-walk directions and pick
                # the one giving lowest locants to the ring's OTHER
                # substituents (f)/(g)), reusing the SAME
                # pg-aware orientation primitive the free-acid path already
                # gets right (`orient_cycloalkane` /
                # `_orient_cycloalkane_with_pg` in `cycloalkanes.py`). That
                # primitive is order-invariant -- it exhaustively tries all
                # 2n rotations, never just the ONE direction
                # `RingInfo.AtomRings` happened to return. The OLD code
                # here only rotated `ring_atoms_in_acid` to start at
                # `start_atom` in whatever direction AtomRings gave it --
                # never tried the reverse -- which was both a lowest-locant
                # violation ('methyl 5-methylcyclopentanecarboxylate'
                # instead of '2-methyl...') AND a determinism defect: the
                # SAME molecule, spelled with a different SMILES atom
                # order, can flip which direction AtomRings returns.
                from .cycloalkanes import get_ring_substituents, orient_cycloalkane
                _sub_positions = get_ring_substituents(mol, tuple(ring_atoms_in_acid))
                oriented_ring = orient_cycloalkane(
                    mol, tuple(ring_atoms_in_acid), _sub_positions,
                    principal_group_atoms={start_atom},
                )
                ring_locant_for_suffix = oriented_ring.index(start_atom) + 1
            else:
                oriented_ring = ring_atoms_in_acid

            # Build exclude set: carbonyl C, carbonyl O(s), ester O, alkyl atoms
            ester_o = ester_match[2] if len(ester_match) > 2 else None
            carbonyl_o_set = set()
            for idx in acid_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'O':
                    for bond in atom.GetBonds():
                        nbr_atom = bond.GetOtherAtom(atom)
                        # Only match true carbonyl O (O=C), not nitro O (O=N)
                        if (bond.GetBondTypeAsDouble() == 2.0
                                and nbr_atom.GetSymbol() == 'C'):
                            carbonyl_o_set.add(idx)
                            break
            exclude = set(alkyl_atoms) | carbonyl_o_set | {carbonyl_c}
            if ester_o is not None:
                exclude.add(ester_o)

            # a phase Wave 2 (#6): a SECOND, non-principal ester whose
            # alcohol-side atom sits directly ON this ring (Ar-O-C(=O)R,
            # e.g. an aryl acetate substituent on the same ring that also
            # carries THIS ester's acid) is invisible to the exclude set
            # above -- its atoms belong to a wholly separate ester bond,
            # not to this ester's own acid/alkyl fragments. Left in, the
            # generic substituent walker below mis-reads the whole
            # '-O-C(=O)-CH3' branch (measured: 'hydroxyethyl' for an
            # acetoxy branch -- a WRONG MOLECULE, not just an ugly name).
            # Name any such exocyclic ester with the SAME helper
            # `detect_exocyclic_esters` uses elsewhere, and exclude its
            # atoms from the generic walk so it is not named twice.
            extra_ester_groups: dict = {}
            for exo in detect_exocyclic_esters(mol):
                exo_match = exo['ester_match']
                if exo_match[0] == carbonyl_c:
                    continue  # this ester itself
                exo_ring_atom = exo['ring_attach_atom_idx']
                if exo_ring_atom not in ring_atoms_in_acid or exo_ring_atom not in oriented_ring:
                    continue  # different ring, or numbering unavailable -- leave to the generic walker
                exo_locant = oriented_ring.index(exo_ring_atom) + 1
                extra_ester_groups.setdefault(exo['acyloxy_prefix'], []).append(exo_locant)
                exo_acid_atoms, _ = parse_ester_fragments(mol, exo_match)
                exclude.update(exo_acid_atoms)
                if len(exo_match) > 2:
                    exclude.add(exo_match[2])

            try:
                from ..assembly.composer import _format_prefix_groups, _integrate_universal_prefixes
                acid_prefix_str = _integrate_universal_prefixes(
                    mol, set(ring_atoms_in_acid),
                    parent_type="ring",
                    oriented_ring=oriented_ring,
                    exclude_atoms=exclude,
                )
            except Exception as exc:
                logger.debug("Ester ring-acid prefix discovery failed: %s", exc)
                acid_prefix_str = ""

            if extra_ester_groups:
                extra_str = _format_prefix_groups(dict(extra_ester_groups))
                if acid_prefix_str:
                    from ..assembly.naming_utils import alpha_sort_key as _alpha_sort_key
                    parts = sorted(
                        [p for p in (extra_str, acid_prefix_str) if p],
                        key=_alpha_sort_key,
                    )
                    acid_prefix_str = "-".join(parts)
                else:
                    acid_prefix_str = extra_str

    acylate_name = get_acylate_name(acid_name)

    # a phase (C), nit-1: "Citation of locants" is
    # deny-by-default -- once the ring bears a substituent prefix (an
    # essential locant), the ring's OWN suffix-attachment locant must also
    # be cited ('methyl 2-methylcyclohexane-1-carboxylate', never
    # '...cyclohexanecarboxylate'). An unsubstituted ring acid keeps the
    # licensed omission -- 'methyl cyclohexanecarboxylate' is
    # unaffected. Mirrors the Wave-2 independent-ester path
    # (`_name_ring_principal_independent_esters`), which already does this.
    if acid_has_ring and acid_prefix_str and ring_locant_for_suffix is not None:
        acylate_name = _insert_ring_ester_locant(acylate_name, ring_locant_for_suffix)

    # Prepend acid-side substituent prefixes to the acylate name.
    # The prefix string from _format_prefix_groups ends without a trailing
    # hyphen; IUPAC joins prefix directly to parent (e.g., "3-methylbutanoate").
    if acid_prefix_str:
        acylate_name = f"{acid_prefix_str}{acylate_name}"

    # Collect R/S stereodescriptors for atoms in the acid fragment.
    # Skip if the acid name already contains stereo (e.g., from the unsaturation
    # path which produces names like "(4E)-octa-4,7-dienoic").
    # Use specific stereo-prefix regex to avoid false matches with parenthesized
    # substituent names like "(oxan-2-yl)oxy" (IUPAC.
    import re

    from .stereochemistry import format_stereodescriptor_string
    acid_stereo = []
    if not re.match(r'^\(\d*[RSrsEZez](,\d*[RSrsEZez])*\)', acylate_name):
        acid_stereo = _collect_ester_fragment_stereo(mol, acid_atoms, ester_match)
        # Only keep R/S descriptors (E/Z is handled by the unsaturation path)
        acid_stereo = [(loc, cip) for loc, cip in acid_stereo if cip in ('R', 'S')]

    # Prepend R/S stereo prefix to the acylate word
    if acid_stereo:
        stereo_prefix = format_stereodescriptor_string(acid_stereo)
        acylate_name = f"{stereo_prefix}{acylate_name}"

    return acylate_name


def _collect_ester_fragment_stereo(mol, acid_atoms: List[int],
                                   ester_match: tuple) -> list:
    """Collect R/S stereodescriptors for the acid fragment of an ester.

    Builds an atom_to_locant mapping for the acid chain (carbonyl C = 1,
    then chain outward) and collects R/S descriptors for stereocenters
    within the acid fragment.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned).
        acid_atoms: Atom indices of the acid fragment.
        ester_match: Ester SMARTS match tuple.

    Returns:
        List of (locant, cip_code) tuples for acid fragment stereocenters.
    """
    from .stereochemistry import collect_stereodescriptors

    acid_set = set(acid_atoms)

    # Find the carbonyl carbon (C with =O in acid fragment)
    carbonyl_c = None
    carbonyl_o_idx = None
    for idx in acid_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if (nbr.GetSymbol() == 'O'
                    and bond.GetBondTypeAsDouble() == 2.0
                    and nbr.GetIdx() in acid_set):
                carbonyl_c = idx
                carbonyl_o_idx = nbr.GetIdx()
                break
        if carbonyl_c is not None:
            break

    if carbonyl_c is None:
        return []

    # For a CHAIN acid, a ring inside the acid fragment is a ring SUBSTITUENT whose
    # stereo is cited in the substituent's OWN descriptor block -- numbering
    # into it here double-cites those stereocenters at spurious parent locants
    # ('ethyl (5R,5S,6S)-2-[(3S,4S,5R)-...cyclohexylidene]acetate'). Exclude ring
    # atoms so the acid parent cites only its chain stereo. A RING acid (the ring IS
    # the parent) still cites its ring stereo.
    #
    # a phase (C): anchored on the KNOWN carbonyl carbon
    # (``ester_match[0]``), never a BFS-scanned ``acid_atoms`` set -- see
    # ``_carbonyl_is_ring_bonded``.
    _chain_acid = not _carbonyl_is_ring_bonded(mol, ester_match[0])

    # BFS from carbonyl C through the acid fragment to build chain ordering
    # (carbonyl C = locant 1, next C = locant 2, etc.)
    visited = {carbonyl_c}
    queue = deque([(carbonyl_c, 1)])
    atom_to_locant = {carbonyl_c: 1}

    while queue:
        current, locant = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in visited or nbr_idx not in acid_set:
                continue
            if nbr_idx == carbonyl_o_idx:
                # Skip the carbonyl oxygen (=O), it's not numbered
                continue
            if nbr.GetSymbol() == 'O' and nbr.GetTotalNumHs() >= 1:
                # Skip -OH of the acid/fragment (not numbered)
                continue
            if _chain_acid and nbr.IsInRing():
                # ring substituent on a chain acid -> its stereo is the
                # substituent's, not the parent's; do not number/descend into it.
                continue
            visited.add(nbr_idx)
            next_locant = locant + 1 if nbr.GetSymbol() == 'C' else locant
            atom_to_locant[nbr_idx] = next_locant
            queue.append((nbr_idx, next_locant))

    return collect_stereodescriptors(mol, atom_to_locant)


def _collect_alkyl_fragment_stereo(mol, alkyl_atoms: List[int],
                                   ester_match: tuple) -> list:
    """Collect R/S stereodescriptors for the alkyl fragment of an ester.

    Builds an atom_to_locant mapping for the alkyl chain using IUPAC
    numbering: finds the longest carbon chain, numbers from the terminal
    that gives the lowest locant to the attachment point (O-bonded C).

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned).
        alkyl_atoms: Atom indices of the alkyl fragment.
        ester_match: Ester SMARTS match tuple.

    Returns:
        List of (locant, cip_code) tuples for alkyl fragment stereocenters.
    """
    from .stereochemistry import collect_stereodescriptors

    alkyl_set = set(alkyl_atoms)

    # Find the oxygen-bonded carbon in the alkyl fragment (attachment point).
    anchor_c = None
    for idx in alkyl_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'O' and nbr.GetIdx() not in alkyl_set:
                anchor_c = idx
                break
        if anchor_c is not None:
            break

    if anchor_c is None:
        return []

    # Collect carbon atoms in the alkyl fragment
    carbon_atoms = [
        idx for idx in alkyl_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    ]

    if len(carbon_atoms) <= 1:
        return []

    # Build adjacency for carbons within alkyl fragment
    adj: dict = {c: [] for c in carbon_atoms}
    for c in carbon_atoms:
        atom = mol.GetAtomWithIdx(c)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in adj:
                adj[c].append(nbr.GetIdx())

    # Find the longest chain through the alkyl fragment using DFS from
    # each terminal carbon. The longest chain determines IUPAC numbering.
    terminals = [c for c in carbon_atoms if len(adj[c]) <= 1]
    if not terminals:
        terminals = carbon_atoms  # fallback: all are potential starts

    best_chain: list = []
    for start in terminals:
        # DFS to find longest path from start
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            extended = False
            for nbr in adj[node]:
                if nbr not in path:
                    stack.append((nbr, path + [nbr]))
                    extended = True
            if not extended and len(path) > len(best_chain):
                best_chain = path

    if not best_chain:
        return []

    # Number the chain: choose direction that gives the anchor_c the
    # lowest locant (IUPAC rule: lowest locant for attachment point).
    if anchor_c in best_chain:
        idx_fwd = best_chain.index(anchor_c) + 1  # 1-based locant forward
        idx_rev = len(best_chain) - best_chain.index(anchor_c)  # 1-based reversed
        if idx_rev < idx_fwd:
            best_chain = list(reversed(best_chain))
    # else: anchor not on longest chain (branch); keep forward order

    # Build atom_to_locant from the numbered chain
    atom_to_locant = {}
    for i, atom_idx in enumerate(best_chain):
        atom_to_locant[atom_idx] = i + 1  # 1-based

    return collect_stereodescriptors(mol, atom_to_locant)


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


def name_noncarbon_ester(mol, match: tuple) -> Optional[str]:
    """Name an ester whose acid OR alcohol component is not the ordinary
    carbon-on-oxygen carboxylic ester / /:

      * pseudoester R-CO-O-Z (Z a Group-13/14/15 organyl) -> 'Zyl acylate'
                      (CH3-CO-O-Si(CH3)3 -> 'trimethylsilyl acetate')
      * sulfonic ester R-SO2-O-R' -> 'R'yl R-sulfonate'
                      (CH3-SO2-O-CH3 -> 'methyl methanesulfonate')
      * sulfinic ester R-S(=O)-O-R' -> 'R'yl R-sulfinate'

    Single mechanism (root-cause, not a per-class string surgery): the acid
    center is ``match[0]`` (a carbonyl C or an S). Locate the ester oxygen and
    the O-side organyl structurally, sever the ester bond to recover the NEUTRAL
    free acid, name it through the general pipeline (``name_compound`` — handles
    retained/systematic, ring, unsaturated, substituted acids), convert the
    '-ic acid' ending to the '-ate' anion stem (``_acid_name_to_ate``), and name
    the O-side organyl via ``name_substituent`` (yields 'methyl' / 'trimethylsilyl'
    / 'ethyl'). Compose '<organyl> <acid>ate'. Fail-closed (return None) on any
    unnameable component so the caller cascade-continues.

    ``match[0]`` = acid center; the ester oxygen is the single-bonded O on the
    acid center whose OTHER heavy neighbour is the organyl.
    """
    acid_center = match[0]
    center_atom = mol.GetAtomWithIdx(acid_center)

    ester_o = None
    organyl = None
    for nb in center_atom.GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue
        bond = mol.GetBondBetweenAtoms(acid_center, nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            continue
        others = [x for x in nb.GetNeighbors()
                  if x.GetIdx() != acid_center and x.GetAtomicNum() > 1]
        if len(others) == 1:
            ester_o = nb.GetIdx()
            organyl = others[0].GetIdx()
            break
    if ester_o is None or organyl is None:
        return None

    # --- Build the neutral free acid by severing the ester O -> organyl bond. ---
    try:
        rw = Chem.RWMol(mol)
        rw.RemoveBond(ester_o, organyl)
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
    except (ValueError, RuntimeError):
        return None
    acid_frag = next((f for f in Chem.GetMolFrags(m2) if acid_center in f), None)
    if acid_frag is None:
        return None
    # Cyclic ester (sultone/sultine/lactone-like): severing the ester bond leaves
    # the organyl STILL bonded to the acid center via the ring — the two-component
    # 'organyl acylate' form does not apply. Fail-closed.
    if organyl in acid_frag:
        return None
    try:
        acid_smi = Chem.MolFragmentToSmiles(
            m2, atomsToUse=list(acid_frag), canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None

    from orthonym import name_compound  # lazy: re-entrant on the free-acid fragment
    try:
        acid_name = name_compound(acid_smi)
    except Exception:
        return None
    if (not acid_name
            or "unknown" in acid_name.lower()
            or "not supported" in acid_name.lower()):
        return None
    anion = _acid_name_to_ate(acid_name)
    if anion is None:
        return None

    # --- Name the O-side organyl as a substituent group. ---
    from ..assembly.substituent_enumerator import name_substituent
    organyl_atoms = _bfs_fragment(mol, organyl, exclude_atom=ester_o)
    if not organyl_atoms:
        return None
    try:
        organyl_name = name_substituent(mol, organyl_atoms, organyl)
    except Exception:
        return None
    if not organyl_name:
        return None

    return f"{organyl_name} {anion}"


# ============================================================================
# Acyloxy Prefix Naming (IUPAC
# ============================================================================
#
# When an ester is named as a substituent prefix (e.g., on a ring parent),
# the R-CO-O- portion is named as an "acyloxy" group:
# acid name (drop "-ic") + "-yloxy"
#
# "Esters cited as prefixes" (the Blue Book) is the governing
# rule -- "an ester group is indicated by prefixes as 'acyloxy' for the group
# R-CO-O-" -- and it is also the rule cited by every acyloxy row of Appendix 2.
# (The section heading previously named here does not exist in the
# book; grep returns the.2.3 heading only.)
#
# THE RULE, in one line: the acyloxy prefix is the PIN ACYL GROUP name + 'oxy'.
# It is NOT "the trivial form if one exists". 's own examples give
# both sides of the boundary:
#:31711 3-(benzoyloxy)propanoic acid (PIN)
# benzoic acid IS retained as a preferred IUPAC name
#:29715, "Only the following five carboxylic acids retained names and
# are also preferred IUPAC names"), so 'benzoyloxy' is PREFERRED.
#:31723 3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)
# 3-(nicotinoyloxy)propanoic acid
# nicotinic acid is retained for GENERAL nomenclature only
# heading:29745; its row at:29773 reads "nicotinic acid
# pyridine-3-carboxylic acid (PIN)"), so the trivial-derived acyloxy
# prefix is the NON-preferred alternative.
# Also from the same rule: "The systematic name 'acetyloxy' is preferred to the
# contracted name 'acetoxy'."
#
# Per-row PIN status. The preferred ACYL prefixes are enumerated by
# (:30432) "Acyl groups derived from carboxylic acids having retained names that
# are preferred IUPAC names... i.e., carboacyl groups" and listed under
# (:30438): acetyl (:30442), formyl (:30444), benzoyl (:30446),
# oxalyl (:30450), oxalo (:30454). Appendix 2 confirms each row independently --
# its legend (:55416) reads "The symbol * designates the preferred prefix":
#
# PIN, must survive: formyloxy* (:56044), acetyloxy* (:55432 "acetoxy =
# acetyloxy*"), benzoyloxy* (:55589,:56544)
# NON-PIN: propionyloxy -- Appendix 2 prints "propanoyloxy* =
# propionyloxy" (:56658,:56680), so propanoyloxy is
# preferred; propionic acid is general-only (:29789)
# NON-PIN: palmitic/stearic/oleic -- general-only
#:29745; rows:29787,:29791,:29785). Appendix 2:
# "hexadecanoyl* = palmitoyl" (:56482,:56511),
# "octadecanoyl* = stearoyl" (:56441,:56491,:56760),
# "(9Z)-octadec-9-enoyl* = oleoyl" (:56443,:56452,:56489)
# NOT IN THE BOOK AT ALL (0 hits, controls prove the grep finds known
# positives): lauric, myristic, arachidic, arachidonic, linoleic, linolenic,
# valeric, caproic -- and 'butyryloxy', 'valeryloxy',
# 'caproyloxy', 'oxalyloxy', 'lactyloxy',
# 'palmitoyloxy', 'lauroyloxy', 'stearoyloxy',
# 'oleoyloxy', 'arachidoyloxy' are each 0 hits, while
# 'hexadecanoyloxy' (:31846,:55170,:55199) and
# 'octadecanoyloxy' (:55162) do appear.
#
# ⚠ WHY THE NON-PIN ROWS BELOW ARE STILL HERE. This function is a SPELLING
# CONVERTER: it turns a stem that a caller ALREADY CHOSE into its 'oxy' form. It
# is not the PIN decision point -- that is get_acid_fragment_name, which is
# where Tasks J2 and J3 made the fix. Measured (fresh process per molecule, trace
# on this function): after J3 no namer path supplies ANY of the trivial keys
# below except formic/acetic/benzoic, which are the three PIN rows. Deleting the
# non-PIN rows would therefore change no emitted name, and would make this
# function strictly WORSE for a direct caller, because the generic '-ic' ->
# '-yloxy' rule below FABRICATES rather than failing closed:
# 'palmitic' -> 'palmityloxy', 'oleic' -> 'oleyloxy'. That is the contributor guide #9
# ("removing a wrong output can unmask a worse generator") at function level.
#
# ⚠ DO NOT ADD A ROW KEYED ON A SYSTEMATIC STEM. Five such rows
# ('dodecanoic'/'tetradecanoic'/'hexadecanoic'/'octadecanoic'/'icosanoic' ->
# 'lauroyloxy'/'myristoyloxy'/'palmitoyloxy'/'stearoyloxy'/'arachidoyloxy') were
# removed in Task J3. They were the LIVE defect: they took a stem that was
# already the PIN and converted it BACK to the non-PIN word, which is how
# '(palmitoyloxy)acetic acid' survived the Task J2 fix to the ester word. A
# systematic key here cannot ever be correct -- by construction its input is
# already preferred. See internal notes
TRIVIAL_ACID_TO_ACYLOXY = {
    # --- preferred prefixes (PIN); these MUST survive ---
    "formic": "formyloxy",           # formyloxy* Appendix 2:56044
    "acetic": "acetyloxy",           # acetyloxy* Appendix 2:55432
    "benzoic": "benzoyloxy",         # benzoyloxy* Appendix 2:55589,:56544
    # --- general-nomenclature spellings only; NON-PIN, and unreachable from the
    # --- namer after J2/J3. Retained solely so a direct caller gets the
    # --- documented general form instead of a fabrication (see note above).
    "propionic": "propionyloxy",     # non-PIN: propanoyloxy*:56658
    "butyric": "butyryloxy",
    "valeric": "valeryloxy",
    "caproic": "caproyloxy",
    "oxalic": "oxalyloxy",
    "lactic": "lactyloxy",
    "lauric": "lauroyloxy",          # C12:0 non-PIN: dodecanoyl*:55953
    "myristic": "myristoyloxy",      # C14:0 non-PIN: tetradecanoyl
    "palmitic": "palmitoyloxy",      # C16:0 non-PIN: hexadecanoyl*:56482
    "stearic": "stearoyloxy",        # C18:0 non-PIN: octadecanoyl*:56441
    "oleic": "oleoyloxy",            # C18:1 non-PIN: (9Z)-octadec-9-enoyl*:56443
    "linoleic": "linoleoyloxy",      # C18:2 non-PIN
    "linolenic": "linolenoyloxy",    # C18:3 non-PIN
    "arachidic": "arachidoyloxy",    # C20:0 non-PIN: icosanoyl
    "arachidonic": "arachidonoyloxy", # C20:4 non-PIN
}


def get_acyloxy_prefix(acid_name: str) -> str:
    """
    Convert an acid name to its acyloxy prefix form (IUPAC.

    Used when an ester group is named as a substituent prefix rather than
    the principal characteristic group. The R-CO-O- portion becomes an
    "acyloxy" prefix.

    Conversion rule: drop "-ic" (or "-ic acid"), add "-yloxy".

    ⚠ This is a SPELLING converter, not the PIN decision point. It converts the
    stem its caller already chose; whether that stem is preferred is decided by
    get_acid_fragment_name. See the note on TRIVIAL_ACID_TO_ACYLOXY above --
    the table deliberately retains non-PIN general-nomenclature spellings, and
    they are unreachable from the namer.

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
    name = acid_name.strip()

    # Strip trailing " acid" if present (case-insensitive check)
    if name.lower().endswith(" acid"):
        name = name[:-5].strip()

    # Check trivial acid lookup first (use lowercase key for lookup)
    name_lower = name.lower()
    if name_lower in TRIVIAL_ACID_TO_ACYLOXY:
        return TRIVIAL_ACID_TO_ACYLOXY[name_lower]

    # Handle -carboxylic acids (ring acids like cyclopentanecarboxylic, cyclohexanecarboxylic)
    # carboxylic -> carbonyloxy (not carboxylyloxy from the generic -ic rule)
    if name_lower.endswith("carboxylic"):
        return name[:-len("carboxylic")] + "carbonyloxy"

    # Systematic conversion: drop "-ic", add "-yloxy"
    # Works for both "-oic" (propanoic -> propanoyloxy) and "-ic" (generic)
    # Preserves original case (important for stereodescriptors like (11Z,14Z)-)
    if name_lower.endswith("ic"):
        return name[:-2] + "yloxy"

    # Fallback: just append "yloxy"
    return name + "yloxy"


def name_ester_as_prefix(mol, ester_match: tuple) -> Optional[str]:
    """
    Generate an acyloxy prefix name for an ester group.

    Used when the ester is a substituent (not the principal characteristic
    group). Extracts the acid fragment, determines its name, and converts
    to the acyloxy prefix form.

    For branched acid fragments, the principal chain is used for the acid
    stem and branch substituents are included as prefixes in the acyloxy
    name (IUPAC.

    Args:
        mol: RDKit Mol object
        ester_match: Tuple of atom indices from ester SMARTS match

    Returns:
        Acyloxy prefix string (e.g., "acetyloxy", "propanoyloxy",
        "(2-methylpropanoyl)oxy" for branched acids),
        or None if the ester is a lactone or cannot be named.

    Examples:
        For methyl acetate (COC(C)=O), returns "acetyloxy"
        For methyl propanoate (COC(=O)CC), returns "propanoyloxy"
        For isobutyrate ester (OC(=O)C(C)C), returns "(2-methylpropanoyl)oxy"
    """
    # Lactones cannot be named as acyloxy prefixes
    if is_lactone(mol, ester_match):
        return None

    # Parse into acid and alkyl fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_match)

    if not acid_atoms:
        return None

    # Try standard acid naming first -- this handles trivial/retained acid
    # names (acetic, linoleic, palmitic, etc.) and unsaturated acid naming.
    # Only fall through to branched-chain analysis when the standard path
    # would produce an incorrect stem from total carbon count.
    acid_name = get_acid_fragment_name(mol, acid_atoms)

    # IUPAC: For branched acid fragments where total carbon
    # count differs from principal chain length, use the principal chain
    # and include branch substituent prefixes. This catches cases like
    # isobutyric acid (CC(C)C=O: 4 total C, but principal chain = 3C).
    # Skip if the acid fragment has a ring or if a trivial name was found.
    acid_has_ring = acid_fragment_has_ring(mol, acid_atoms)

    if not acid_has_ring:
        acid_set_early = set(acid_atoms)
        acid_principal_chain = _find_acid_principal_chain(mol, acid_atoms)
        if acid_principal_chain and len(acid_principal_chain) >= 2:
            total_carbons = sum(
                1 for idx in acid_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )
            chain_carbons = len(acid_principal_chain)

            # Only use branched naming when the chain is shorter than total
            # carbons, indicating true carbon branching (not just heteroatom
            # substituents like -OH, -OOH on a linear chain).
            has_carbon_branch = chain_carbons < total_carbons

            # 0-WRONG (verified atom-order hazard): this branched carbon-COUNT
            # path selects a principal chain and names every off-chain carbon as
            # a saturated alkyl branch -- it is BLIND to a C=C/C#C that connects
            # the branch. For methacrylic acid (C=C(C)C(=O)O) it named the
            # =CH2 branch '2-methyl' (SATURATED, = a DIFFERENT molecule,
            # isobutyryloxy) for one atom ordering while the standard path below
            # correctly gives '2-methylprop-2-enoyloxy'. Same bug shape as this
            # file's `_has_nonring_unsat` guard (:681). Defer ANY unsaturated acid
            # fragment to the standard get_acid_fragment_name path, which handles
            # unsaturation (ene/yne locants) correctly regardless of atom order.
            _acid_unsaturated = any(
                b.GetBondTypeAsDouble() in (2.0, 3.0)
                and b.GetBeginAtomIdx() in acid_set_early
                and b.GetEndAtomIdx() in acid_set_early
                and b.GetBeginAtom().GetSymbol() == 'C'
                and b.GetEndAtom().GetSymbol() == 'C'
                for b in mol.GetBonds())
            if _acid_unsaturated:
                has_carbon_branch = False

            if has_carbon_branch:
                chain_set = set(acid_principal_chain)
                acid_set = set(acid_atoms)
                chain_length = len(acid_principal_chain)
                acid_stem = get_acid_stem(chain_length)

                # Identify carbonyl O atoms
                carbonyl_o_set = set()
                for idx in acid_atoms:
                    atom = mol.GetAtomWithIdx(idx)
                    if atom.GetSymbol() == 'O':
                        for bond in atom.GetBonds():
                            if bond.GetBondTypeAsDouble() == 2.0:
                                carbonyl_o_set.add(idx)
                                break

                # Build acid-side substituent prefixes using universal pipeline
                ester_o = ester_match[2] if len(ester_match) > 2 else None
                exclude = set(carbonyl_o_set)
                if ester_o is not None:
                    exclude.add(ester_o)
                exclude.update(set(alkyl_atoms))

                acid_prefix_str = ""
                try:
                    from ..assembly.composer import _integrate_universal_prefixes
                    acid_prefix_str = _integrate_universal_prefixes(
                        mol, chain_set,
                        parent_type="chain",
                        principal_chain=acid_principal_chain,
                        exclude_atoms=exclude,
                    )
                except Exception as exc:
                    logger.debug("Ester acyloxy prefix acid-side discovery failed: %s", exc)

                # Build the full acyloxy prefix with branches
                # e.g., "2-methylpropanoic" -> "(2-methylpropanoyl)oxy"
                full_acid_name = f"{acid_prefix_str}{acid_stem}" if acid_prefix_str else acid_stem
                acyloxy = get_acyloxy_prefix(full_acid_name)

                # If branched, wrap in parentheses: "(2-methylpropanoyl)oxy"
                if acid_prefix_str and acyloxy:
                    # Rewrite: strip "yloxy" suffix, wrap acyl in parens, add "oxy"
                    if acyloxy.endswith("yloxy"):
                        acyl_part = acyloxy[:-3]  # "...yl"
                        acyloxy = f"({acyl_part})oxy"
                return acyloxy

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
        For ethyl acetate: returns  (no ring attachment)
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
# Multi-ester Detection and Naming (IUPAC
# ============================================================================
#
# Dicarboxylic acid diesters: two ester groups sharing a diacid backbone.
# Named as "[multiplier]alkyl [parent]anedioate".
# e.g., COC(=O)CC(=O)OC -> "dimethyl propanedioate"
#
# Trivial diacid names used where available:
# oxalic -> oxalate, malonic -> malonate, succinic -> succinate, etc.
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


def _ring_walk_locants(mol, ring_set: set, start: int, second: int) -> Optional[list]:
    """Walk a simple monocyclic ring atom-by-atom from ``start`` through
    ``second``, returning the full bond-adjacency atom order.

    Returns None if the walk cannot complete (defensive; should not happen
    for a genuine SSSR ring atom set).
    """
    order = [start, second]
    prev, current = start, second
    while len(order) < len(ring_set):
        atom = mol.GetAtomWithIdx(current)
        nxt = None
        for nbr in atom.GetNeighbors():
            nid = nbr.GetIdx()
            if nid in ring_set and nid != prev:
                nxt = nid
                break
        if nxt is None:
            return None
        order.append(nxt)
        prev, current = current, nxt
    return order


def _insert_ring_ester_locant(acylate_name: str, locant: int = 1) -> str:
    """Insert an explicit ring-attachment locant before a SYSTEMATIC
    ring-acid ESTER suffix -- 'cyclohexanecarboxylate' ->
    'cyclohexane-1-carboxylate'.

     "Citation of locants" is deny-by-default: once ANY locant in
    a name's scope is essential (a substituent prefix on the same ring
    takes one), every locant in that scope must be cited, including the
    parent's own suffix attachment. Retained trivial acyl stems
    ('benzoate', 'acetate',...) do not match this systematic
    '-carboxylate' suffix and are returned unchanged -- a retained
    parent's single attachment point needs no numeral licence).
    """
    suffix = "carboxylate"
    if acylate_name.endswith(suffix):
        stem = acylate_name[: -len(suffix)]
        if stem.endswith('-') or (stem and stem[-1].isdigit()):
            return acylate_name  # already locanted
        return f"{stem}-{locant}-{suffix}"
    return acylate_name


def _carbonyl_is_ring_bonded(mol, carbonyl_c: int) -> bool:
    """True iff the KNOWN carbonyl carbon ``carbonyl_c`` (identified
    directly from its ester SMARTS match, never inferred from a scanned
    atom set) is itself in a ring (lactone-like) or directly bonded to a
    ring atom (benzoic-/cyclohexanecarboxylic-shape).

     a phase Wave 2 (#6, review fix): ``acid_is_ring_acid(mol,
    acid_atoms)`` decides this by scanning a whole ``acid_atoms`` set for
    the FIRST carbon that looks like a carbonyl and returning based on
    THAT one. For a multi-ester molecule, ``parse_ester_fragments``'s BFS
    excludes only THIS ester's own ``ester_o``, so a ring principal's
    ``acid_atoms`` can be contaminated with a SECOND ester's atoms
    (including its carbonyl carbon) reachable around the ring. Which
    carbon the scan hits first then depends on set/atom-index order --
    the SAME molecule, spelled with a different SMILES atom order, could
    take a different branch. Anchoring directly on the specific,
    already-known ``carbonyl_c`` is invariant to all of that.
    """
    ring_info = mol.GetRingInfo()
    if ring_info.NumAtomRings(carbonyl_c) > 0:
        return True  # lactone-like / carbonyl itself in a ring
    atom = mol.GetAtomWithIdx(carbonyl_c)
    return any(nbr.IsInRing() for nbr in atom.GetNeighbors())


def _name_ring_principal_independent_esters(
    mol, principal: dict, non_principal: list,
    alkyl_name: str,
) -> Optional[str]:
    """Rebuild an independent-ester name when the PRINCIPAL ester's acid is
    a ring acid (e.g. cyclohexanecarboxylic, benzoic) and one or more
    NON-PRINCIPAL esters sit on that same ring -- either directly
    (Ar-O-C(=O)R) or through a single pendant atom (Ar-CH2-O-C(=O)R).

     a phase Wave 2 (#6): the OLD code named the non-principal acid
    ALONE (dropping the alcohol-side linking atom entirely -- the
    methylene of a '-CH2-O-C(=O)R' arm) and space-joined it in front of
    the whole name with no locant -- a wrong-molecule / OPSIN-unparseable
    candidate. This rebuilds each non-principal ester as a COMPLETE,
    LOCANTED substituent prefix (methylene included) sited on the ring,
    reusing the same ``name_substituent_fragment`` compound-substituent
    namer the ring-as-parent composer already uses for this exact shape
    (``rules/polyfunctional.py``'s ring-as-parent path, which names the
    same '-CH2-O-C(=O)CH3' arm as ``'(acetyloxy)methyl'`` when it reaches
    it directly).

    Fails closed (returns None) on any topology this narrow
    reconstruction cannot positively confirm -- a fused/complex ring
    acid, a non-principal ester whose alcohol-side atom is not either ON
    the ring or a single plain atom bonded to it, or a substituent
    fragment ``name_substituent_fragment`` cannot express. NEVER falls
    back to the old space-joined form for a ring principal -- that form
    is proven wrong (dropped atom, no locant), never an acceptable
    "uglier but honest" degrade.

     a phase Wave 2 (#6, review fix): every structural decision here
    is anchored on the PRINCIPAL ester's own known match atoms
    (``principal['match']``), NEVER on ``principal['acid_atoms']`` --
    that set can be contaminated with a second ester's atoms (see
    ``_carbonyl_is_ring_bonded``'s docstring), and scanning it for "the"
    ring or "the" acid name is exactly the order-dependent bug that let
    the SAME molecule, spelled with a different SMILES atom order, fall
    through to the old broken path and fabricate a wrong acid stem
    (measured: 'decanoate' for a cyclohexanecarboxylate principal).
    """
    principal_carbonyl_c = principal['match'][0]
    ring_info = mol.GetRingInfo()

    # Find the ring atom directly bonded to the KNOWN carbonyl carbon --
    # never inferred from principal['acid_atoms'].
    start_atom = None
    for nbr in mol.GetAtomWithIdx(principal_carbonyl_c).GetNeighbors():
        if nbr.IsInRing():
            start_atom = nbr.GetIdx()
            break
    if start_atom is None:
        return None

    # The ring at start_atom must be a single, simple SSSR ring (not
    # shared atom-for-atom with a second ring) for this reconstruction.
    candidate_rings = [r for r in ring_info.AtomRings() if start_atom in r]
    if len(candidate_rings) != 1:
        return None  # fused/spiro/bridged at this atom -- out of scope
    ring_atoms = set(candidate_rings[0])

    # Derive the acid name from the CLEAN ring atom set alone (no
    # dependency on principal['acid_atoms'] at all) -- order-invariant.
    acid_name = get_ring_acid_name(mol, sorted(ring_atoms))
    if not acid_name:
        return None
    acylate_name = get_acylate_name(acid_name)
    if not acylate_name:
        return None

    ring_neighbors = [
        nbr.GetIdx() for nbr in mol.GetAtomWithIdx(start_atom).GetNeighbors()
        if nbr.GetIdx() in ring_atoms
    ]
    if len(ring_neighbors) != 2:
        return None  # not a simple monocyclic ring atom -- decline

    from ..assembly.substituent_naming import name_substituent_fragment

    # Classify each non-principal ester's shape relative to the ring.
    entries = []  # [{'ring_atom': int, 'prefix_name': str},...]
    for ester in non_principal:
        match = ester['match']
        ester_o = match[2]
        attach = match[3]

        if attach in ring_atoms:
            # Direct acyloxy substituent on the ring (Ar-O-C(=O)R).
            non_principal_acid_name = get_acid_fragment_name(mol, ester['acid_atoms'])
            if not non_principal_acid_name:
                return None
            prefix_name = get_acyloxy_prefix(non_principal_acid_name)
            if not prefix_name:
                return None
            entries.append({'ring_atom': attach, 'prefix_name': prefix_name})
            continue

        # Pendant single-atom arm (Ar-CH2-O-C(=O)R). The arm's attach atom
        # must have EXACTLY two heavy neighbors -- the ring atom and the
        # ester oxygen -- so name_substituent_fragment's parent framing
        # (ring = parent_chain) is unambiguous. Anything else (further
        # branching, a second ring, a longer topological path back to the
        # ring) is declined rather than guessed at.
        attach_atom_obj = mol.GetAtomWithIdx(attach)
        heavy_neighbors = [n.GetIdx() for n in attach_atom_obj.GetNeighbors()]
        ring_side = [n for n in heavy_neighbors if n in ring_atoms]
        if (len(heavy_neighbors) != 2 or len(ring_side) != 1
                or ester_o not in heavy_neighbors):
            return None
        parent_ring_atom = ring_side[0]

        sub_atoms = set(ester['acid_atoms']) | {attach, ester_o}
        prefix_name = name_substituent_fragment(
            mol, sorted(sub_atoms), attach, sorted(ring_atoms),
        )
        if not prefix_name:
            return None  # cannot safely express this arm -- decline, never flatten
        entries.append({'ring_atom': parent_ring_atom, 'prefix_name': prefix_name})

    if not entries:
        return None

    # Orient the ring in the direction giving the lowest locant SET to the
    # non-principal substituents lowest locants).
    best_order = None
    best_locants = None
    for second in ring_neighbors:
        order = _ring_walk_locants(mol, ring_atoms, start_atom, second)
        if order is None:
            continue
        atom_to_locant = {a: i + 1 for i, a in enumerate(order)}
        locants = sorted(atom_to_locant[e['ring_atom']] for e in entries)
        if best_locants is None or locants < best_locants:
            best_locants = locants
            best_order = atom_to_locant

    if best_order is None:
        return None

    # Group identical prefixes for multiplier handling and format with locants.
    from collections import defaultdict

    from ..assembly.composer import _format_prefix_groups
    prefix_groups: dict = defaultdict(list)
    for e in entries:
        prefix_groups[e['prefix_name']].append(best_order[e['ring_atom']])

    prefix_str = _format_prefix_groups(dict(prefix_groups))
    if not prefix_str:
        return None

    ring_locant = best_order[start_atom]  # always 1 by construction
    final_acylate = _insert_ring_ester_locant(acylate_name, ring_locant)

    if prefix_str[-1].isalpha() and final_acylate[:1].isdigit():
        ring_part = f"{prefix_str}-{final_acylate}"
    else:
        ring_part = f"{prefix_str}{final_acylate}"

    return f"{alkyl_name} {ring_part}"


def name_independent_esters(mol, ester_matches: list) -> Optional[str]:
    """
    Name a molecule with independent ester groups (IUPAC.

    Independent esters are multiple ester groups that do not share an acid
    backbone (not dicarboxylic diester) or alcohol backbone (not polyol
    polyester). The most senior ester bond becomes the principal suffix
    (-oate) and the remaining ester bonds become acyloxy prefixes.

    Strategy:
      1. Score each ester by acid fragment size (largest acid = principal).
      2. The principal ester uses standard "alkyl [parent]oate" format.
      3. Non-principal esters become acyloxy prefixes on the parent chain.
      4. If the molecule is too complex, return None (decomposition handles it).

    Args:
        mol: RDKit Mol object
        ester_matches: List of ester SMARTS match tuples
                       (carbonyl_c, carbonyl_o, ester_o, alkyl_c)

    Returns:
        Name string, or None if the molecule is too complex for this handler.
    """
    if len(ester_matches) < 2:
        return None

    # Parse all ester fragments and score by acid fragment size
    ester_data = []
    for match in ester_matches:
        acid_atoms, alkyl_atoms = parse_ester_fragments(mol, match)
        if not acid_atoms or not alkyl_atoms:
            return None  # Cannot parse fragments -- too complex
        acid_carbon_count = sum(
            1 for idx in acid_atoms
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
        )
        ester_data.append({
            'match': match,
            'acid_atoms': acid_atoms,
            'alkyl_atoms': alkyl_atoms,
            'acid_carbon_count': acid_carbon_count,
        })

    # Sort by acid carbon count descending -- largest acid = principal ester
    ester_data.sort(key=lambda e: e['acid_carbon_count'], reverse=True)

    # Principal ester: name as "alkyl [acid]oate"
    principal = ester_data[0]

    # a phase Wave 2 (#6): when the PRINCIPAL ester's acid is a RING
    # acid, the OLD space-joined/unlocanted acyloxy-prefix path below is
    # PROVEN WRONG (it drops the alcohol-side linking atom of any
    # non-principal ester whose alkyl carbon is a pendant substituent on
    # that ring, and never cites a locant at all). Rebuild it as a
    # complete, locanted substituent prefix instead; fail closed (None)
    # rather than fall through to the broken join for this shape.
    #
    # (Review fix) This check -- and the WHOLE ring-principal branch it
    # guards -- must be decided from principal['match'][0] (the KNOWN
    # carbonyl carbon), never from principal['acid_atoms']: that set can
    # be contaminated with the OTHER ester's atoms (see
    # `_carbonyl_is_ring_bonded`'s docstring), which previously made
    # `acid_is_ring_acid(mol, principal['acid_atoms'])` -- and the
    # subsequent `get_acid_fragment_name` call -- return DIFFERENT
    # answers for the SAME molecule depending on SMILES atom order.
    if _carbonyl_is_ring_bonded(mol, principal['match'][0]):
        alkyl_name = get_alkyl_fragment_name(mol, principal['alkyl_atoms'])
        if not alkyl_name:
            return None
        return _name_ring_principal_independent_esters(
            mol, principal, ester_data[1:], alkyl_name,
        )

    acid_name = get_acid_fragment_name(mol, principal['acid_atoms'])
    if not acid_name:
        return None

    from ..data.trivial_acids import get_acylate_name
    acylate_name = get_acylate_name(acid_name)
    if not acylate_name:
        return None

    alkyl_name = get_alkyl_fragment_name(mol, principal['alkyl_atoms'])
    if not alkyl_name:
        return None

    # Non-principal esters: convert to acyloxy prefixes
    acyloxy_prefixes = []
    for ester in ester_data[1:]:
        non_principal_acid_name = get_acid_fragment_name(mol, ester['acid_atoms'])
        if not non_principal_acid_name:
            return None  # Cannot name a non-principal acid -- bail out
        acyloxy = get_acyloxy_prefix(non_principal_acid_name)
        acyloxy_prefixes.append(acyloxy)

    # Sort acyloxy prefixes alphabetically per IUPAC
    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix
    if not acyloxy_prefixes:
        return None

    # Group identical acyloxy prefixes with multipliers
    from collections import Counter
    prefix_counts = Counter(acyloxy_prefixes)
    prefix_parts = []
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: alpha_sort_key(x[0])):
        if count == 1:
            prefix_parts.append(f"({prefix})")
        else:
            multiplier = get_multiplier_prefix(count, prefix)
            prefix_parts.append(f"{multiplier}({prefix})")

    prefix_str = "-".join(prefix_parts)

    # Assemble: "[acyloxy prefixes] alkyl [acid]oate"
    # The acyloxy prefixes modify the alkyl part, so format is:
    # "alkyl [prefix][acid]oate" or "[prefix] alkyl [acid]oate"
    # Per IUPAC, acyloxy prefixes on the acid parent chain go in front of the
    # oate name. For truly independent esters this is not standard; return the
    # best-effort name or None for decomposition to handle.
    if prefix_str:
        return f"{prefix_str} {alkyl_name} {acylate_name}"
    else:
        return f"{alkyl_name} {acylate_name}"


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
    queue = deque([start])

    while queue:
        current = queue.popleft()
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
    queue = deque([(c1, [c1])])

    while queue:
        current, path = queue.popleft()
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


# ---------------------------------------------------------------------------
# Ring dicarboxylic-acid diester /
# ---------------------------------------------------------------------------
# When BOTH ester carbonyls are bonded to a ring, the parent is the ring
# di-carboxylic acid (benzene-1,3-dicarboxylic acid), NOT an acyclic -dioate
# chain. The legacy backbone BFS (_find_backbone_carbons) walks THROUGH the
# ring carbons and linearises the ring, so dimethyl isophthalate became
# "dimethyl pentanedioate" — a constitutionally different molecule.

_diacid_namer_singleton = None


def _diacid_namer():
    """Lazily-cached namer for intermediate (di)acid fragments.

    Grammar + OPSIN validity gate are disabled: the di-acid name is an
    INTERMEDIATE that gets transformed into the ester ("...dicarboxylate"),
    not a final emitted name, so it must not be gated (mirrors the
    fragment-intermediate contract at namer.py:1044).
    """
    global _diacid_namer_singleton
    if _diacid_namer_singleton is None:
        from ..namer import Orthonym
        _diacid_namer_singleton = Orthonym(
            _disable_grammar_validation=True,
            _disable_opsin_validity_gate=True,
        )
    return _diacid_namer_singleton


def _build_diacid_from_diester(mol, ester_matches):
    """Return the di-acid Mol obtained by stripping both ester alkyl groups.

    For each ester (carbonyl_c, =O, ester_o, alkyl_c) the ester_o--alkyl_c bond
    is broken and the entire alkyl fragment removed; the ester oxygen keeps its
    bond to the carbonyl carbon and sanitises to -C(=O)OH. Returns None on
    failure.
    """
    rw = Chem.RWMol(mol)
    to_remove = set()
    for m in ester_matches:
        ester_o, alkyl_c = m[2], m[3]
        stack = [alkyl_c]
        seen = {ester_o}
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            to_remove.add(a)
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                if nbr.GetIdx() not in seen:
                    stack.append(nbr.GetIdx())
        bond = rw.GetBondBetweenAtoms(ester_o, alkyl_c)
        if bond is not None:
            rw.RemoveBond(ester_o, alkyl_c)
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    try:
        diacid = rw.GetMol()
        Chem.SanitizeMol(diacid)
    except Exception:
        return None
    return diacid


def _both_ester_carbonyls_on_ring(mol, ester_matches) -> bool:
    """True iff every ester carbonyl carbon is bonded to a ring atom and all
    those ring atoms belong to one connected ring system (so the parent is a
    single ring di/poly-carboxylic acid, not two separate ring acids)."""
    ring_neighbors = []
    for m in ester_matches:
        carbonyl_c = m[0]
        ring_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors()
                     if n.IsInRing()]
        if not ring_nbrs:
            return False
        ring_neighbors.append(ring_nbrs[0])
    # All anchoring ring atoms in the same ring system (share a fused component).
    ri = mol.GetRingInfo()
    components = []
    for ring in ri.AtomRings():
        rset = set(ring)
        merged = False
        for comp in components:
            if comp & rset:
                comp |= rset
                merged = True
        if not merged:
            components.append(set(rset))
    # second pass to merge transitively-fused rings
    changed = True
    while changed:
        changed = False
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                if components[i] & components[j]:
                    components[i] |= components[j]
                    components[j] = set()
                    changed = True
        components = [c for c in components if c]
    for comp in components:
        if all(a in comp for a in ring_neighbors):
            return True
    return False


def _acid_name_to_ate(acid_name: str) -> Optional[str]:
    """Convert an acid name to its ester anion stem:
    '...ic acid' -> '...ate' (dicarboxylic acid -> dicarboxylate; dioic ->
    dioate). Returns None when the input is not an '-ic acid' form."""
    if acid_name and acid_name.endswith("ic acid"):
        return acid_name[: -len("ic acid")] + "ate"
    return None


def _name_ring_dicarboxylic_diester(mol, ester_matches: list) -> Optional[str]:
    """Name a ring di-carboxylic-acid diester as 'dialkyl <ring>-x,y-dicarboxylate'.

    Builds the di-acid, names it via the GENERAL pipeline (bypassing the
    retained-name dispatch so the SYSTEMATIC PIN benzene-1,3-dicarboxylic acid
    is used, not the non-PIN retained 'isophthalic acid'), then converts the
    '-ic acid' suffix to '-ate' and prefixes the alkyl group(s). Fail-closed.
    """
    if len(ester_matches) != 2:
        return None
    if not _both_ester_carbonyls_on_ring(mol, ester_matches):
        return None

    alkyl_names = []
    for m in ester_matches:
        _acid_atoms, alkyl_atoms = parse_ester_fragments(mol, m)
        if not alkyl_atoms:
            return None
        alkyl_name = get_alkyl_fragment_name(mol, alkyl_atoms)
        if not alkyl_name:
            return None
        alkyl_names.append(alkyl_name)

    diacid = _build_diacid_from_diester(mol, ester_matches)
    if diacid is None:
        return None

    from ..assembly.composer import assemble_name
    from ..namer import compute_features
    try:
        diacid_smiles = Chem.MolToSmiles(diacid)
        feats = compute_features(diacid, diacid_smiles)
        _diacid_namer()._classify(feats)
        acid_name = assemble_name(feats, style="pin")
    except Exception:
        return None

    ate = _acid_name_to_ate(acid_name)
    # Guard: only accept a well-formed ring-acid PIN (must carry the
    # 'carboxyl' suffix root); reject garbled/empty stems.
    if ate is None or "carboxyl" not in ate:
        return None

    if alkyl_names[0] == alkyl_names[1]:
        return f"di{alkyl_names[0]} {ate}"
    first, second = sorted(alkyl_names)
    return f"{first} {second} {ate}"


def name_polyfunctional_ester_via_acid(mol, ester_match: tuple) -> Optional[str]:
    """Name a polyfunctional compound whose most-senior group is a SINGLE ester.

    The ester stays the principal group (suffix '-oate'); every junior group
    (acyl halide -> oxo+halo, ketone/aldehyde -> oxo, nitrile -> cyano, -OH ->
    hydroxy,...) is a prefix on the acid-side chain (IUPAC +:
    esters outrank acyl halides/amides/nitriles/aldehydes/ketones/alcohols).

    Strategy (mirrors the diester ring path): build the ACID analog (ester ->
    free -COOH), name it via the GENERAL pipeline (which already emits those
    junior groups as prefixes), then convert '-ic acid' -> '-ate' and prepend
    the alkyl group as a separate word. The junior groups must lie on the ACID
    side; the removed alkyl side must be a plain hydrocarbon. Fail-closed.
    """
    _acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_match)
    if not alkyl_atoms:
        return None

    # W8 P3, BB 54595-54608): amino-acid esters (the alpha-amino
    # group is what routes a molecule like 'methyl alaninate' through this
    # POLYFUNCTIONAL path rather than the plain single-FG name_ester) use the
    # retained '-ate' stem (e.g. 'methyl L-alaninate'). Tried BEFORE the
    # acid-analog/general-pipeline route below; declines (returns None, falls
    # through with no regression) for anything out of scope.
    aa_ester_name = _name_amino_acid_ester(mol, _acid_atoms, alkyl_atoms, ester_match)
    if aa_ester_name is not None:
        return aa_ester_name

    # The acid-analog strategy only reconstructs the ACID side, so the alcohol
    # side must be named as ONE complete substituent word or its atoms would be
    # silently dropped into a different molecule. A plain hydrocarbon alkyl side
    # takes the fast path; an alkyl side that carries its OWN junior functional
    # groups (oxo, hydroxy, halo,...) is legal -- the ester stays
    # senior, the alcohol side's groups are prefixes on that substituent) PROVIDED
    # the recursive substituent namer expresses the whole fragment. That namer is
    # complete-by-construction (it names the entire fragment or declines), so it
    # is the fail-closed gate: `CC(=O)OCC1CCCC1=O` -> `(2-oxocyclopentyl)methyl
    # acetate`, and anything it cannot fully name returns None (no atom drop).
    alkyl_set = set(alkyl_atoms)
    _alkyl_has_fg = any(
        mol.GetAtomWithIdx(a).GetSymbol() not in ('C', 'H') for a in alkyl_set
    )
    if _alkyl_has_fg:
        from ..assembly.substituent_enumerator import name_substituent
        _attach = _ester_attachment_atom(mol, alkyl_set)
        if _attach is None:
            return None
        try:
            alkyl_name = name_substituent(mol, sorted(alkyl_set), _attach)
        except Exception:
            return None
        if not alkyl_name or alkyl_name == "substituent":
            return None
    else:
        alkyl_name = get_alkyl_fragment_name(mol, alkyl_atoms)
        if not alkyl_name:
            return None

    # W8-P6 Cluster D: the alkyl (alcohol-side) component, cited
    # as a separate word, must carry its OWN stereodescriptor immediately
    # before it -- mirrors name_ester's collector (:1094-1103), which
    # this polyfunctional-routed path (taken whenever the acid side carries a
    # junior functional group, e.g. -OH, alongside the ester) never called,
    # silently dropping a chiral alkyl fragment's descriptor.
    import re as _re_pf_alkyl
    if not _re_pf_alkyl.match(r'^\(\d*[RSrsEZez](,\d*[RSrsEZez])*\)', alkyl_name):
        alkyl_stereo = _collect_alkyl_fragment_stereo(mol, alkyl_atoms, ester_match)
        alkyl_stereo = [(loc, cip) for loc, cip in alkyl_stereo if cip in ('R', 'S')]
        if alkyl_stereo:
            from .stereochemistry import format_stereodescriptor_string as _fmt_pf_alkyl
            alkyl_name = f"{_fmt_pf_alkyl(alkyl_stereo)}{alkyl_name}"

    acid_mol = _build_diacid_from_diester(mol, [ester_match])
    if acid_mol is None:
        return None

    from ..assembly.composer import assemble_name
    from ..namer import compute_features
    try:
        acid_smiles = Chem.MolToSmiles(acid_mol)
    except Exception:
        return None

    # (BB 33398): if the acid analog is EXACTLY one of the
    # retained / functional-replacement inorganic-acid parents (carbamimidic
    # acid, carbonimidic acid,...), its '-ic acid' name is the PIN stem — the
    # general chain pipeline (which does not consult the inorganic-acids
    # dispatch row @40) would emit the systematic-but-non-PIN
    # '1-aminomethanimidic acid' instead. Only substitute when the whole acid
    # analog is the exact table SMILES (no substituents to drop).
    from .inorganic_acids import lookup_exact_acid_name
    retained = lookup_exact_acid_name(acid_smiles)
    if retained is not None:
        acid_name = retained
    else:
        try:
            feats = compute_features(acid_mol, acid_smiles)
            _diacid_namer()._classify(feats)
            acid_name = assemble_name(feats, style="pin")
        except Exception:
            return None

    ate = _acid_name_to_ate(acid_name)
    if ate is None:
        return None
    # Reject garbled stems (empty / bare-suffix artefacts).
    if len(ate) < 4 or ate.startswith("ano"):
        return None
    return f"{alkyl_name} {ate}"


def name_polyfunctional_diester_free_hydroxy(
    mol, ester_matches: list, principal_chain: list,
) -> Optional[str]:
    """ (BB verbatim worked example at: '2-(acetyl-
    oxy)-3-(hexadecanoyloxy)propyl (9Z)-octadec-9-enoate') + (greater
    number of skeletal atoms): a partially-esterified acyclic polyol bearing
    exactly TWO different noncyclic ester groups plus >=1 free hydroxyl, all
    on the SAME short saturated carbon backbone (the diacylglycerol shape).

    The senior acid (the ester whose acid-side principal chain has MORE
    carbons -- seniority of chains) stays the functional-class parent
    ('<yl> <acid>oate'); the OTHER ester is demoted to an 'acyloxy' prefix
    and the free hydroxyl(s) to 'hydroxy' prefixes, both cited on the 'yl'
    word, exactly as 's worked examples do it.

     a phase: this closes the gap where ``name_polyfunctional_ester_via_
    acid`` (the single-ester acid-analog strategy) declines outright for
    >1 ester match, and the legacy fallback in
    ``rules/polyfunctional.py::name_polyfunctional`` then demoted BOTH
    esters and promoted the junior hydroxy class to principal -- inverting
     (ester class 9 outranks hydroxy class 17).

    Fail-closed scope (returns None -- caller falls through to the legacy
     acyloxy-all/'-ol' demotion, which is uglier but not wrong -- for
    anything broader):
      - exactly 2 ester matches;
      - `principal_chain` is a plain acyclic, saturated, all-carbon chain;
      - BOTH esters' alkyl (alcohol-side) attachment atoms sit ON that
        chain, and neither acid is a ring acid;
      - every atom of `principal_chain` not consumed by an ester attachment
        is either UNDECORATED or bears exactly one free hydroxyl (any other
        decoration -- halogen, amine, a third ester,... -- declines);
      - the two acids' principal-chain lengths are NOT tied (a tie is
        outside this narrow scope).
    """
    if len(ester_matches) != 2:
        return None
    if not principal_chain or len(principal_chain) < 2:
        return None
    chain_set = set(principal_chain)

    # The backbone this path decorates must be a plain acyclic saturated
    # carbon chain -- it never renders a ring or an unsaturated backbone.
    for a in principal_chain:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != 'C' or atom.IsInRing():
            return None
    for i in range(len(principal_chain) - 1):
        bond = mol.GetBondBetweenAtoms(principal_chain[i], principal_chain[i + 1])
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None

    infos = []
    for m in ester_matches:
        if len(m) < 4:
            return None
        alkyl_c = m[3]
        if alkyl_c not in chain_set:
            return None
        acid_atoms, alkyl_atoms = parse_ester_fragments(mol, m)
        if not acid_atoms or not alkyl_atoms:
            return None
        if acid_is_ring_acid(mol, acid_atoms):
            return None  # bounded to acyclic acids
        acid_chain = _find_acid_principal_chain(mol, acid_atoms)
        if not acid_chain:
            return None
        infos.append({
            'match': m, 'attach': alkyl_c,
            'acid_atoms': acid_atoms, 'alkyl_atoms': alkyl_atoms,
            'acid_len': len(acid_chain),
        })

    if infos[0]['acid_len'] == infos[1]['acid_len']:
        return None  # tied acid length -- outside this narrow scope

    infos.sort(key=lambda x: -x['acid_len'])
    principal, demoted = infos[0], infos[1]

    # The demoted ester's acyloxy prefix -- reuses the SAME single-ester
    # prefix builder the ring-acid path already relies on, so the spelling
    # ('(9Z)-pentadec-9-enoyloxy') is identical to what OST already emits
    # elsewhere for this exact acyl group.
    acyloxy_word = name_ester_as_prefix(mol, demoted['match'])
    if not acyloxy_word:
        return None

    # Any OTHER backbone position must carry a free chain hydroxyl (the
    # only junior decoration this path knows how to place). Anything else
    # (halogen, amine, a third ester,...) declines fail-closed.
    consumed = {principal['attach'], demoted['attach']}
    hydroxy_atoms = []
    for a in principal_chain:
        if a in consumed:
            continue
        atom = mol.GetAtomWithIdx(a)
        oh_nbrs = [nb for nb in atom.GetNeighbors()
                   if nb.GetSymbol() == 'O' and nb.GetIdx() not in chain_set
                   and nb.GetTotalNumHs() >= 1]
        if len(oh_nbrs) == 1:
            hydroxy_atoms.append(a)
        elif oh_nbrs:
            return None  # more than one O-substituent here -- unmodelled

    # The ester oxygen that bridges each attachment atom to its own acid
    # side belongs to NEITHER acid_atoms nor alkyl_atoms (parse_ester_
    # fragments splits AT that atom), so it must be matched by identity,
    # not fragment membership.
    ester_o_for = {
        principal['attach']: principal['match'][2],
        demoted['attach']: demoted['match'][2],
    }
    accounted = consumed | set(hydroxy_atoms)
    for a in principal_chain:
        atom = mol.GetAtomWithIdx(a)
        for nb in atom.GetNeighbors():
            nidx = nb.GetIdx()
            if nidx in chain_set:
                continue
            if nb.GetAtomicNum() <= 1:
                continue  # implicit/explicit H
            if a in ester_o_for and nidx == ester_o_for[a]:
                continue  # this position's own ester oxygen
            if a in accounted and nb.GetSymbol() == 'O' and nb.GetTotalNumHs() >= 1:
                continue  # this position's own free hydroxyl
            return None  # unaccounted decoration -- decline

    # ---- Numbering /: the principal ester's attachment
    # atom gets the LOWEST possible locant on the 'yl' word; on an
    # exact-centre tie, gives the lowest locant to whichever
    # decoration is cited first in alphanumerical order. ----
    chain = list(principal_chain)
    attach = principal['attach']
    fwd_k = chain.index(attach) + 1
    rev_k = len(chain) - chain.index(attach)

    decorations = {demoted['attach']: acyloxy_word}
    for a in hydroxy_atoms:
        decorations[a] = 'hydroxy'

    if rev_k < fwd_k:
        chain = list(reversed(chain))
    elif rev_k == fwd_k:
        from ..assembly.naming_utils import alpha_sort_key as _alpha_sort_key
        chain_len0 = len(chain)
        fwd_pos = {a: chain.index(a) + 1 for a in decorations}
        rev_pos = {a: chain_len0 + 1 - p for a, p in fwd_pos.items()}
        if fwd_pos != rev_pos:
            names_alpha = sorted(decorations, key=lambda a: _alpha_sort_key(decorations[a]))
            cur_seq = tuple(fwd_pos[a] for a in names_alpha)
            flip_seq = tuple(rev_pos[a] for a in names_alpha)
            if flip_seq < cur_seq:
                chain = list(reversed(chain))

    k = chain.index(attach) + 1
    chain_pos = {a: i + 1 for i, a in enumerate(chain)}

    # ---- Build the decorated 'yl' word prefix ordering +
    # enclosing-mark escalation, both via the shared formatter). ----
    from collections import defaultdict as _dd

    from ..assembly.composer import _format_prefix_groups
    branch_groups: dict = _dd(list)
    if hydroxy_atoms:
        branch_groups['hydroxy'] = sorted(chain_pos[a] for a in hydroxy_atoms)
    branch_groups[acyloxy_word].append(chain_pos[demoted['attach']])
    prefix_str = _format_prefix_groups(dict(branch_groups))

    chain_len = len(chain)
    if k == 1:
        try:
            base = get_alkyl_name(chain_len)
        except (ValueError, KeyError):
            return None
        yl_word = f"{prefix_str}{base}"
    else:
        try:
            stem = get_chain_prefix(chain_len)
        except (ValueError, KeyError):
            return None
        yl_word = f"{prefix_str}{stem}an-{k}-yl"

    # Stereo on the yl fragment / mirrors name_ester's
    # collector, renumbered to THIS chain's own locants -- the attachment
    # atom is frequently the stereocentre, e.g. '(2S)-...propan-2-yl').
    from .stereochemistry import (
        collect_stereodescriptors as _collect_stereo,
    )
    from .stereochemistry import (
        format_stereodescriptor_string as _fmt_stereo,
    )
    yl_stereo = [(loc, cip) for loc, cip in _collect_stereo(mol, chain_pos)
                 if cip in ('R', 'S')]
    if yl_stereo:
        yl_word = f"{_fmt_stereo(yl_stereo)}{yl_word}"

    acylate_word = _build_ester_acid_word(
        mol, principal['match'], principal['acid_atoms'], principal['alkyl_atoms'],
    )
    if not acylate_word:
        return None

    return f"{yl_word} {acylate_word}"


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

    # Ring-attached diester: both carbonyls bonded to a ring ->
    # the parent is the ring di-carboxylic acid, not an acyclic -dioate.
    ring_name = _name_ring_dicarboxylic_diester(mol, ester_matches)
    if ring_name is not None:
        return ring_name

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
# Polyol Polyester Naming (IUPAC
# ============================================================================
#
# Fully-esterified polyols (triacetin, triglycerides) are named using
# acyloxy prefixes on the polyol backbone:
# "locants-multiplier(acyloxy)parent"
#
# Examples:
# triacetin -> "1,2,3-tri(acetyloxy)propane"
# mixed triester -> "1,3-di(acetyloxy)-2-(propanoyloxy)propane"


def _try_functional_class_diol_diester(mol, ester_matches: list) -> Optional[str]:
    """ /: a symmetric diol diester (two IDENTICAL acyl
    groups esterifying a divalent acyclic diol) is the functional-class
    multiplicative PIN '<diol-diyl> di<acid>oate' (ethane-1,2-diyl diacetate),
    NOT the substitutive bis(acyloxy) form.

    Fail-closed scope: exactly two esters, identical acyl fragments, a clean
    acyclic diol backbone whose two attachment carbons are the two chain ends,
    carrying no other substituents. Returns the PIN or None (fall through).
    """

    if len(ester_matches) != 2:
        return None

    ester_oxygens = set()
    carbonyl_carbons = set()
    for match in ester_matches:
        ester_oxygens.add(match[2])
        carbonyl_carbons.add(match[0])
    exclude = ester_oxygens | carbonyl_carbons

    # (1) identical acyl groups (as '-oate' anion names)
    anions = []
    for c_c, c_o, e_o, alk_c in ester_matches:
        acid_atoms = _bfs_fragment(mol, c_c, exclude_atom=e_o)
        acid_name = get_acid_fragment_name(mol, acid_atoms)
        if not acid_name:
            return None
        # get_acid_fragment_name returns e.g. 'acetic' (no ' acid' tail);
        # _acid_name_to_ate needs the '...ic acid' form.
        acid_full = acid_name if acid_name.endswith("acid") else acid_name + " acid"
        anion = _acid_name_to_ate(acid_full)
        if anion is None:
            return None
        anions.append(anion)
    if len(set(anions)) != 1:
        return None

    # (2) name the diol residue as a divalent -diyl group with attachment locants.
    backbone_start_atoms = [m[3] for m in ester_matches]
    backbone = _find_polyol_backbone(mol, backbone_start_atoms, exclude)
    if backbone is None or len(backbone) < 2:
        return None
    ordered = _order_backbone_chain(mol, backbone, exclude)
    if ordered is None:
        return None

    # Fail closed if any backbone carbon carries a heavy substituent outside
    # the backbone / the two ester oxygens (keep it a clean unsubstituted diol).
    backbone_set = set(ordered)
    allowed = backbone_set | ester_oxygens
    for idx in ordered:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() <= 1:
                continue
            if nbr.GetIdx() not in allowed:
                return None

    n = len(ordered)
    parent_prefix = get_chain_prefix(n)
    if not parent_prefix:
        return None

    # Attachment locants = positions of the two ester-bearing carbons, numbered
    # for lowest locant set (first-point-of-difference over both directions).
    fwd = {idx: i + 1 for i, idx in enumerate(ordered)}
    rev = {idx: n - i for i, idx in enumerate(ordered)}
    attach_atoms = [m[3] for m in ester_matches]
    fwd_locs = sorted(fwd[a] for a in attach_atoms)
    rev_locs = sorted(rev[a] for a in attach_atoms)
    locs = fwd_locs if fwd_locs <= rev_locs else rev_locs

    loc_str = ",".join(str(x) for x in locs)
    diyl = f"{parent_prefix}ane-{loc_str}-diyl"

    # (3) multiplied acid anion: 'di' + 'acetate' -> 'diacetate',
    # 'di' before a consonant, no elision).
    return f"{diyl} di{anions[0]}"


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

    # Build prefix parts using format_substituent_prefix for proper
    # IUPAC parenthesization (acyloxy = compound prefix = bis/tris)
    from ..assembly.naming_utils import format_substituent_prefix
    parts = []
    for acyloxy, locants in sorted_groups:
        locants.sort()
        count = len(locants)
        parts.append(format_substituent_prefix(acyloxy, locants, count))

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
                count = len(locants)
                parts.append(format_substituent_prefix(acyloxy, locants, count))

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
    queue = deque(start_atoms)
    visited.update(start_atoms)

    while queue:
        current = queue.popleft()
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
