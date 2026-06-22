"""
Polycyclic Aromatic Hydrocarbon (PAH) data for IUPAC naming.

Contains lookup tables for common polycyclic aromatic hydrocarbons including:
- Canonical SMILES for exact matching
- SMARTS patterns for substructure matching
- Number of atoms in the PAH core
- IUPAC standard numbering (atom index to IUPAC locant mapping)
- Allowed substituent positions

IUPAC Naming Rules for PAHs:
- PAH numbering is FIXED by IUPAC, not reoriented based on substituents
- Use retained names (naphthalene, anthracene, etc.) as parent
- Substituents are named with their IUPAC locant position

Reference: IUPAC Blue Book 2013, Section P-25 (Fused and Bridged Fused Ring Systems)

PAH Classification:
- Bicyclic: naphthalene
- Tricyclic: anthracene, phenanthrene, fluorene, acenaphthene, acenaphthylene
- Tetracyclic: pyrene, chrysene, tetracene, triphenylene, benz[a]anthracene, benzo[c]phenanthrene
- Pentacyclic: pentacene, perylene, benzo[a]pyrene
- Hexacyclic+: coronene

Partially saturated PAHs:
- 9,10-dihydroanthracene
- 1,2-dihydronaphthalene
"""

from typing import Dict, Optional, List, Tuple, Any, Set


# Polycyclic aromatic hydrocarbon data
# Key: retained name
# Value: dict with canonical_smiles, smarts, num_atoms, iupac_numbering, substituent_positions
POLYCYCLIC_DATA: Dict[str, Dict[str, Any]] = {
    'naphthalene': {
        'canonical_smiles': 'c1ccc2ccccc2c1',
        'smarts': 'c1ccc2ccccc2c1',  # For substructure matching
        'num_atoms': 10,
        # IUPAC numbering for naphthalene:
        #     8  1
        #    /  \ /
        #   7    2
        #   |    |
        #   6    3
        #    \  / \
        #     5  4
        # Maps canonical atom index -> IUPAC position (1-indexed)
        # Note: This mapping is determined empirically based on RDKit's canonical ordering
        # Empirically derived from RDKit canonical SMILES 'c1ccc2ccccc2c1':
        # Rings: [0,9,8,3,2,1] and [4,5,6,7,8,3]. Fusion: idx 3, idx 8.
        # Peripheral path: 9->0->1->2->[3]->4->5->6->7->[8]
        # IUPAC: 1->2->3->4->[4a]->5->6->7->8->[8a]
        'iupac_numbering': {
            9: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: 5, 5: 6, 6: 7, 7: 8, 8: '8a'
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],  # Allowed positions
        'num_rings': 2,
    },
    'azulene': {
        # Retained fused-ring hydrocarbon (IUPAC 2013 P-25.1.1, Table 28.1):
        # a 5-membered ring ortho-fused to a 7-membered ring, fully mancude
        # (aromatic). Carbocyclic, so it is NOT caught by name_fused_heterocycle,
        # and being aromatic it is NOT caught by _name_saturated_fused_carbocyclic
        # either -- without this entry identify_polycyclic() returns None and the
        # molecule falls through to the acyclic chain catch-all, which emits the
        # malformed empty stem 'ane' (V-1 / theme T10).
        'canonical_smiles': 'c1ccc2cccc-2cc1',
        'smarts': 'c1ccc2cccc-2cc1',  # fusion bond is formally single (c-c)
        'num_atoms': 10,
        # IUPAC numbering for azulene: 1,2,3 on the 5-membered ring; 4,5,6,7,8 on
        # the 7-membered ring; 3a / 8a the two fusion atoms. Maps canonical atom
        # index (of 'c1ccc2cccc-2cc1') -> IUPAC position. Derived from the
        # canonical topology: fusion atoms idx 3 (=3a) and idx 7 (=8a); the
        # 5-ring non-fusion arc 3-[4-5-6]-7 carries C3,C2,C1; the 7-ring
        # non-fusion arc 7-[8-9-0-1-2]-3 carries C8,C7,C6,C5,C4. (Azulene has a
        # mirror plane through C2/C6, so this orientation's locant set is the
        # unique lowest set regardless of the C3a/C8a labelling direction.)
        # NOTE: this dict is consumed by get_polycyclic_iupac_locants (correct
        # for bare azulene). The *substituent*-naming path
        # (rules/polycyclics._map_pah_atoms_to_iupac) is hardcoded to
        # naphthalene's symmetric 6-6 alpha/beta pattern and has no azulene
        # branch, so SUBSTITUTED azulene locants are unproven/likely wrong here
        # -- that is fused-ring numbering, owned by Phase E1/DD4, out of C-T10
        # scope. Bare azulene (the V-1 gold) is correct.
        'iupac_numbering': {
            6: 1, 5: 2, 4: 3, 3: '3a', 2: 4, 1: 5, 0: 6, 9: 7, 8: 8, 7: '8a'
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 2,
    },
    'anthracene': {
        'canonical_smiles': 'c1ccc2cc3ccccc3cc2c1',
        'smarts': 'c1ccc2cc3ccccc3cc2c1',
        'num_atoms': 14,
        # IUPAC numbering for anthracene (linear tricyclic). AUTHORITATIVE:
        # re-derived 2026-06-22 (v23 IH-01) from OPSIN `anthracene -o extendedsmi`
        # ($_AV: locants) mapped onto this canonical SMILES. The prior numbering
        # was INVALID — it placed a *meso* carbon (central-ring atoms 4 & 11) at
        # locant 5 instead of the correct 9/10, so 9-substituted/9,10-dihydro
        # anthracenes were mis-numbered (e.g. 9-methyl -> wrong "5-methyl").
        # Meso (central-ring CH) atoms 4 -> 9, 11 -> 10.
        'iupac_numbering': {
            2: 1, 1: 2, 0: 3, 13: 4, 9: 5, 8: 6, 7: 7, 6: 8, 4: 9, 11: 10,
            3: '9a', 5: '8a', 10: '10a', 12: '4a',
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    'phenanthrene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccccc12',
        'smarts': 'c1ccc2c(c1)ccc1ccccc12',
        'num_atoms': 14,
        # IUPAC numbering for phenanthrene (angular tricyclic)
        # Empirically derived from RDKit canonical SMILES 'c1ccc2c(c1)ccc1ccccc12':
        # Rings: [0,5,4,3,2,1], [6,7,8,13,3,4], [9,10,11,12,13,8]
        # Fusion: idx 3(=4a), 4(=10a), 8(=8a), 13(=4b)
        # Ring A(R0): 5->0->1->2 peripheral, Ring C(R1): 6->7 peripheral,
        # Ring B(R2): 12->11->10->9 peripheral
        # IUPAC: 1->2->3->4->[4a]->[10a]->10->9->[8a]->[4b]->5->6->7->8
        'iupac_numbering': {
            5: 1, 0: 2, 1: 3, 2: 4, 3: '4a', 4: '10a',
            6: 10, 7: 9, 8: '8a', 13: '4b',
            12: 5, 11: 6, 10: 7, 9: 8
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    'pyrene': {
        'canonical_smiles': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'smarts': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'num_atoms': 16,
        # IUPAC numbering for pyrene (peri-condensed tetracyclic). AUTHORITATIVE:
        # re-derived 2026-06-22 (v23 IH-01) from OPSIN `pyrene -o extendedsmi`
        # ($_AV: locants) mapped onto this canonical SMILES. The prior numbering
        # was INVALID (interior/fusion locants '3b'/'10a' instead of the correct
        # peri carbons 10a/10b/10c), so substituted pyrenes were mis-numbered.
        # Interior carbons: 14 -> 10b, 15 -> 10c.
        'iupac_numbering': {
            1: 1, 0: 2, 13: 3, 11: 4, 10: 5, 8: 6, 7: 7, 6: 8, 4: 9, 3: 10,
            12: '3a', 9: '5a', 5: '8a', 2: '10a', 14: '10b', 15: '10c',
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 4,
    },
    'fluorene': {
        'canonical_smiles': 'c1ccc2c(c1)Cc1ccccc1-2',  # Has sp3 carbon (position 9)
        'smarts': 'c1ccc2c(c1)Cc1ccccc1-2',
        'num_atoms': 13,
        # IUPAC numbering for fluorene (with methylene bridge at position 9)
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # 9 is the sp3 carbon
        'num_rings': 3,
    },
    'acenaphthene': {
        'canonical_smiles': 'c1cc2c3c(cccc3c1)CC2',  # Has two sp3 carbons
        'smarts': 'c1cc2c3c(cccc3c1)CC2',
        'num_atoms': 12,
        # IUPAC numbering for acenaphthene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],  # Excluding the ethylene bridge
        'num_rings': 3,
    },
    'acenaphthylene': {
        'canonical_smiles': 'C1=Cc2cccc3cccc1c23',  # Fully unsaturated
        'smarts': 'C1=Cc2cccc3cccc1c23',
        'num_atoms': 12,
        # IUPAC numbering for acenaphthylene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 3,
    },
    'chrysene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1c3ccccc3ccc21',
        'smarts': 'c1ccc2c(c1)ccc1c3ccccc3ccc21',
        'num_atoms': 18,
        # IUPAC numbering for chrysene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    # ==========================================================================
    # Tetracyclic (4 rings) - additional
    # ==========================================================================
    'tetracene': {
        # Also known as naphthacene - linear 4-ring PAH
        'canonical_smiles': 'c1ccc2cc3cc4ccccc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4ccccc4cc3cc2c1',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'triphenylene': {
        # Angular 4-ring PAH (three benzene rings sharing a central ring)
        'canonical_smiles': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'smarts': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'benz[a]anthracene': {
        # Bent 4-ring PAH (benzene fused to anthracene)
        'canonical_smiles': 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
        'smarts': 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'benzo[c]phenanthrene': {
        # Angular 4-ring PAH
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccc3ccccc3c12',
        'smarts': 'c1ccc2c(c1)ccc1ccc3ccccc3c12',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    # ==========================================================================
    # Pentacyclic (5 rings)
    # ==========================================================================
    'pentacene': {
        # Linear 5-ring PAH
        'canonical_smiles': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
        'num_atoms': 22,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'num_rings': 5,
    },
    'perylene': {
        # Peri-condensed PAH (two naphthalene units joined peri)
        'canonical_smiles': 'c1ccc2cccc3cc4c(c1)cc1cccc4c1c23',
        'smarts': 'c1ccc2cccc3cc4c(c1)cc1cccc4c1c23',
        'num_atoms': 20,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 6,  # RDKit counts 6 rings due to perception
    },
    'benzo[a]pyrene': {
        # Important carcinogen - benzene fused to pyrene
        'canonical_smiles': 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
        'smarts': 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
        'num_atoms': 20,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 5,
    },
    # ==========================================================================
    # Hexacyclic+ (6+ rings)
    # ==========================================================================
    'coronene': {
        # 7 rings, hexagonal symmetry
        'canonical_smiles': 'c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61',
        'smarts': 'c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61',
        'num_atoms': 24,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 7,
    },
    # ==========================================================================
    # Partially saturated PAHs
    # ==========================================================================
    '9,10-dihydroanthracene': {
        # Anthracene with positions 9 and 10 saturated (sp3)
        'canonical_smiles': 'c1ccc2c(c1)Cc1ccccc1C2',
        'smarts': 'c1ccc2c(c1)Cc1ccccc1C2',
        'num_atoms': 14,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    '1,2-dihydronaphthalene': {
        # Naphthalene with positions 1 and 2 saturated
        'canonical_smiles': 'C1=Cc2ccccc2CC1',
        'smarts': 'C1=Cc2ccccc2CC1',
        'num_atoms': 10,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 2,
    },
}


# Reverse lookup: canonical SMILES -> PAH name
_SMILES_TO_NAME: Dict[str, str] = {
    data['canonical_smiles']: name
    for name, data in POLYCYCLIC_DATA.items()
}


def get_polycyclic_by_smiles(canonical_smiles: str) -> Optional[Dict[str, Any]]:
    """
    Look up polycyclic data by canonical SMILES.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        Dict with 'name' and all PAH data if found, None otherwise

    Example:
        >>> get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        {'name': 'naphthalene', 'canonical_smiles': 'c1ccc2ccccc2c1', ...}
    """
    name = _SMILES_TO_NAME.get(canonical_smiles)
    if name is None:
        return None

    return {'name': name, **POLYCYCLIC_DATA[name]}


def get_polycyclic_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up polycyclic data by name.

    Args:
        name: PAH name (e.g., 'naphthalene', 'anthracene')

    Returns:
        Dict with PAH data if found, None otherwise
    """
    if name not in POLYCYCLIC_DATA:
        return None

    return {'name': name, **POLYCYCLIC_DATA[name]}


def is_polycyclic_aromatic(canonical_smiles: str) -> bool:
    """
    Check if a canonical SMILES represents a known polycyclic aromatic.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        True if the SMILES matches a known PAH
    """
    return canonical_smiles in _SMILES_TO_NAME


def get_pah_names() -> List[str]:
    """
    Get list of all supported PAH names.

    Returns:
        List of PAH names
    """
    return list(POLYCYCLIC_DATA.keys())


def match_polycyclic_core(mol) -> Optional[Tuple[str, Dict[int, int]]]:
    """
    Match a molecule against known PAH cores using substructure matching.

    For substituted PAHs, finds the largest matching core and returns
    the atom mapping from molecule indices to IUPAC locants.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (pah_name, atom_mapping) where atom_mapping is {mol_idx: iupac_locant}
        Returns None if no PAH core found

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> name, mapping = match_polycyclic_core(mol)
        >>> name
        'naphthalene'
    """
    # Lazy import to avoid circular dependency
    from rdkit import Chem

    # Sort PAHs by size (largest first) to find best match
    pah_by_size = sorted(
        POLYCYCLIC_DATA.items(),
        key=lambda x: x[1]['num_atoms'],
        reverse=True
    )

    for pah_name, pah_data in pah_by_size:
        smarts = pah_data['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # Found a match - build atom mapping
            match_atoms = matches[0]

            # Verify the match contains expected number of atoms
            if len(match_atoms) != pah_data['num_atoms']:
                continue

            # Build atom index to locant mapping
            # The mapping is based on the SMARTS match order
            # For now, use position in match as proxy for locant
            # More sophisticated mapping uses _map_pah_atoms_to_iupac in polycyclics.py
            atom_mapping = {match_atoms[i]: i + 1 for i in range(len(match_atoms))}

            return (pah_name, atom_mapping)

    return None


def get_pah_core_atoms(mol, pah_name: str) -> Optional[Set[int]]:
    """
    Get the atom indices that form a PAH core in a molecule.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Set of atom indices forming the PAH core, or None if no match

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> core = get_pah_core_atoms(mol, 'naphthalene')
        >>> len(core)  # 10 atoms in naphthalene core
        10
    """
    from rdkit import Chem

    if pah_name not in POLYCYCLIC_DATA:
        return None

    pah_data = POLYCYCLIC_DATA[pah_name]
    smarts = pah_data['smarts']
    pattern = Chem.MolFromSmarts(smarts)

    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return set(matches[0])

    return None


def get_pah_substituent_positions(mol, pah_name: str) -> List[int]:
    """
    Get IUPAC locant positions where substituents are attached.

    Args:
        mol: RDKit Mol object with substituted PAH
        pah_name: Name of the PAH core (e.g., 'naphthalene')

    Returns:
        List of IUPAC locants with substituents

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> get_pah_substituent_positions(mol, 'naphthalene')
        [2]  # Methyl at position 2
    """
    from rdkit import Chem

    core_atoms = get_pah_core_atoms(mol, pah_name)
    if not core_atoms:
        return []

    result = match_polycyclic_core(mol)
    if not result or result[0] != pah_name:
        return []

    _, atom_mapping = result

    # Find atoms with non-core neighbors (these have substituents)
    substituted_positions = []
    for atom_idx in core_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in core_atoms:
                # This core atom has a substituent
                if atom_idx in atom_mapping:
                    substituted_positions.append(atom_mapping[atom_idx])
                break

    return sorted(set(substituted_positions))


# =============================================================================
# Complex Fusion Data
# Pre-computed fusion descriptors for known complex polycyclic systems
# =============================================================================

# Complex fusion data for multi-component and advanced fused systems
# Key: canonical name
# Value: dict with smiles, prefix, descriptor, parent, child, child_count
COMPLEX_FUSION_DATA: Dict[str, Dict[str, Any]] = {
    # Dibenzo compounds (two benzene rings fused to parent)
    'dibenzo[a,c]anthracene': {
        'smiles': 'c1ccc2c(c1)cc1ccc3cc4ccccc4cc3c1c2',
        'prefix': 'dibenzo',
        'descriptor': '[a,c]',
        'parent': 'anthracene',
        'child': 'benzene',
        'child_count': 2,
        'num_rings': 5,
    },
    'dibenzo[a,h]anthracene': {
        'smiles': 'c1ccc2c(c1)ccc1cc3ccc4ccccc4c3cc12',
        'prefix': 'dibenzo',
        'descriptor': '[a,h]',
        'parent': 'anthracene',
        'child': 'benzene',
        'child_count': 2,
        'num_rings': 5,
    },
    'dibenzo[a,j]anthracene': {
        'smiles': 'c1ccc2c(c1)c3ccc4ccccc4c3cc2c1ccccc1',
        'prefix': 'dibenzo',
        'descriptor': '[a,j]',
        'parent': 'anthracene',
        'child': 'benzene',
        'child_count': 2,
        'num_rings': 5,
    },

    # Naphtho compounds (naphthalene fused to heterocycle)
    'naphtho[2,3-b]furan': {
        'smiles': 'c1ccc2cc3occc3cc2c1',
        'prefix': 'naphtho',
        'descriptor': '[2,3-b]',
        'parent': 'furan',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },
    'naphtho[1,2-b]furan': {
        'smiles': 'c1ccc2c(c1)cc1ccoc1c2',
        'prefix': 'naphtho',
        'descriptor': '[1,2-b]',
        'parent': 'furan',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },
    'naphtho[2,3-b]thiophene': {
        'smiles': 'c1ccc2cc3sccc3cc2c1',
        'prefix': 'naphtho',
        'descriptor': '[2,3-b]',
        'parent': 'thiophene',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },
    'naphtho[1,2-b]thiophene': {
        'smiles': 'c1ccc2c(c1)cc1ccsc1c2',
        'prefix': 'naphtho',
        'descriptor': '[1,2-b]',
        'parent': 'thiophene',
        'child': 'naphthalene',
        'child_count': 1,
        'num_rings': 3,
    },

    # Pyrido compounds (pyridine fused to pyrimidine/other heterocycles)
    'pyrido[2,3-d]pyrimidine': {
        'smiles': 'c1cnc2nccnc2c1',
        'prefix': 'pyrido',
        'descriptor': '[2,3-d]',
        'parent': 'pyrimidine',
        'child': 'pyridine',
        'child_count': 1,
        'num_rings': 2,
    },
    'pyrido[3,4-d]pyrimidine': {
        'smiles': 'c1cnc2ncncc2c1',
        'prefix': 'pyrido',
        'descriptor': '[3,4-d]',
        'parent': 'pyrimidine',
        'child': 'pyridine',
        'child_count': 1,
        'num_rings': 2,
    },
    'pyrido[4,3-d]pyrimidine': {
        'smiles': 'c1cnc2cncnc2c1',
        'prefix': 'pyrido',
        'descriptor': '[4,3-d]',
        'parent': 'pyrimidine',
        'child': 'pyridine',
        'child_count': 1,
        'num_rings': 2,
    },

    # Furo compounds
    'furo[2,3-b]pyridine': {
        'smiles': 'c1cc2ccoc2nc1',
        'prefix': 'furo',
        'descriptor': '[2,3-b]',
        'parent': 'pyridine',
        'child': 'furan',
        'child_count': 1,
        'num_rings': 2,
    },
    'furo[3,2-b]pyridine': {
        'smiles': 'c1cc2occc2nc1',
        'prefix': 'furo',
        'descriptor': '[3,2-b]',
        'parent': 'pyridine',
        'child': 'furan',
        'child_count': 1,
        'num_rings': 2,
    },

    # Thieno compounds
    'thieno[2,3-b]pyridine': {
        'smiles': 'c1cc2ccsc2nc1',
        'prefix': 'thieno',
        'descriptor': '[2,3-b]',
        'parent': 'pyridine',
        'child': 'thiophene',
        'child_count': 1,
        'num_rings': 2,
    },
    'thieno[3,2-b]pyridine': {
        'smiles': 'c1cc2sccc2nc1',
        'prefix': 'thieno',
        'descriptor': '[3,2-b]',
        'parent': 'pyridine',
        'child': 'thiophene',
        'child_count': 1,
        'num_rings': 2,
    },

    # Imidazo compounds
    'imidazo[1,2-a]pyridine': {
        'smiles': 'c1ccn2ccnc2c1',
        'prefix': 'imidazo',
        'descriptor': '[1,2-a]',
        'parent': 'pyridine',
        'child': 'imidazole',
        'child_count': 1,
        'num_rings': 2,
    },
    'imidazo[4,5-b]pyridine': {
        'smiles': 'c1cc2nc[nH]c2nc1',
        'prefix': 'imidazo',
        'descriptor': '[4,5-b]',
        'parent': 'pyridine',
        'child': 'imidazole',
        'child_count': 1,
        'num_rings': 2,
    },
}

# Reverse lookup: canonical SMILES -> complex fusion name
_COMPLEX_SMILES_TO_NAME: Dict[str, str] = {
    data['smiles']: name
    for name, data in COMPLEX_FUSION_DATA.items()
}


# Edge numbering data for common parent rings
# Maps parent ring name to dict of edge positions (0-indexed) to letters
# Edge 'a' is between IUPAC atoms 1-2, 'b' between 2-3, etc.
PARENT_RING_EDGES: Dict[str, Dict[int, str]] = {
    'naphthalene': {
        # 10 atoms, 10 edges (some are fusion edges, not substituable)
        # Edge labels based on IUPAC numbering
        0: 'a',  # between atoms 1-2
        1: 'b',  # between atoms 2-3
        2: 'c',  # between atoms 3-4 (peri-fusion)
        3: 'd',  # between atoms 4-4a
        4: 'e',  # between atoms 4a-5
        5: 'f',  # between atoms 5-6
        6: 'g',  # between atoms 6-7
        7: 'h',  # between atoms 7-8
        8: 'i',  # between atoms 8-8a
        9: 'j',  # between atoms 8a-1
    },
    'anthracene': {
        # 14 atoms, 14 edges
        0: 'a',   # between atoms 1-2
        1: 'b',   # between atoms 2-3
        2: 'c',   # between atoms 3-4
        3: 'd',   # between atoms 4-4a
        4: 'e',   # between atoms 4a-10
        5: 'f',   # between atoms 10-10a
        6: 'g',   # between atoms 10a-5
        7: 'h',   # between atoms 5-6
        8: 'i',   # between atoms 6-7
        9: 'j',   # between atoms 7-8
        10: 'k',  # between atoms 8-8a
        11: 'l',  # between atoms 8a-9
        12: 'm',  # between atoms 9-9a
        13: 'n',  # between atoms 9a-1
    },
    'phenanthrene': {
        # 14 atoms, 14 edges (angular arrangement)
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'g',
        7: 'h',
        8: 'i',
        9: 'j',
        10: 'k',
        11: 'l',
        12: 'm',
        13: 'n',
    },
    'benzene': {
        # 6 atoms, 6 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
    },
    'furan': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
    'thiophene': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
    'pyrrole': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
    'pyridine': {
        # 6 atoms, 6 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
    },
    'pyrimidine': {
        # 6 atoms, 6 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
    },
    'imidazole': {
        # 5 atoms, 5 edges
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
    },
}


# Extended fusion prefixes (additions to existing FUSION_PREFIXES)
EXTENDED_FUSION_PREFIXES: Dict[str, str] = {
    # Additional prefixes not in main fusion_descriptors.py
    'phenanthro': 'phenanthro',  # explicit form
    'acenaphtho': 'acenaphtho',
    'acenaphtheno': 'acenaphtheno',  # for acenaphthene
    'fluoreno': 'fluoreno',
    'chryseno': 'chryseno',
    'triphenyleno': 'triphenyleno',
    'peryleno': 'peryleno',
    'coroneno': 'coroneno',
    'pyrazolo': 'pyrazolo',
    'isoxazolo': 'isoxazolo',
    'isothiazolo': 'isothiazolo',
    'oxazolo': 'oxazolo',
    'thiazolo': 'thiazolo',
    'triazolo': 'triazolo',
    'tetrazolo': 'tetrazolo',
    'pyrazino': 'pyrazino',
    'pyridazino': 'pyridazino',
    'triazino': 'triazino',
}


def get_complex_fusion_info(smiles: str) -> Optional[Tuple[str, str, str]]:
    """
    Look up pre-computed fusion descriptor for a known complex polycyclic.

    Args:
        smiles: SMILES string of the compound

    Returns:
        Tuple of (prefix, descriptor, parent) if found, None otherwise
        Example: ('dibenzo', '[a,c]', 'anthracene')

    Example:
        >>> get_complex_fusion_info('c1ccc2c(c1)cc1ccc3cc4ccccc4cc3c1c2')
        ('dibenzo', '[a,c]', 'anthracene')
    """
    # Try exact SMILES match first
    name = _COMPLEX_SMILES_TO_NAME.get(smiles)
    if name and name in COMPLEX_FUSION_DATA:
        data = COMPLEX_FUSION_DATA[name]
        return (data['prefix'], data['descriptor'], data['parent'])

    # Canonicalize and try again
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            name = _COMPLEX_SMILES_TO_NAME.get(canonical)
            if name and name in COMPLEX_FUSION_DATA:
                data = COMPLEX_FUSION_DATA[name]
                return (data['prefix'], data['descriptor'], data['parent'])
    except Exception:
        pass

    return None


def get_complex_fusion_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up complex fusion data by name.

    Args:
        name: Full fusion name (e.g., 'dibenzo[a,c]anthracene')

    Returns:
        Dict with fusion data if found, None otherwise
    """
    return COMPLEX_FUSION_DATA.get(name)


def get_edge_letter(ring_name: str, edge_index: int) -> str:
    """
    Get the IUPAC edge letter for a given edge index in a parent ring.

    Args:
        ring_name: Name of the parent ring (e.g., 'anthracene')
        edge_index: 0-indexed edge position

    Returns:
        Edge letter ('a', 'b', etc.) or empty string if not found

    Example:
        >>> get_edge_letter('anthracene', 0)
        'a'
        >>> get_edge_letter('anthracene', 2)
        'c'
    """
    if ring_name not in PARENT_RING_EDGES:
        return ''

    edges = PARENT_RING_EDGES[ring_name]
    return edges.get(edge_index, '')


def get_all_edge_letters(ring_name: str) -> List[str]:
    """
    Get all available edge letters for a parent ring.

    Args:
        ring_name: Name of the parent ring

    Returns:
        List of edge letters in order

    Example:
        >>> get_all_edge_letters('benzene')
        ['a', 'b', 'c', 'd', 'e', 'f']
    """
    if ring_name not in PARENT_RING_EDGES:
        return []

    edges = PARENT_RING_EDGES[ring_name]
    return [edges[i] for i in sorted(edges.keys())]
