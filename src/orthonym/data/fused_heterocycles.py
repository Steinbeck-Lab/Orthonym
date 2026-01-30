"""
Retained names for fused heterocyclic systems.

IUPAC 2013 prefers retained names (indole over benzo[b]pyrrole) for these
common fused heterocycles. ALWAYS check this lookup before applying
systematic fusion naming rules.

Keys are canonical SMILES (verified with RDKit), values contain:
- name: The IUPAC retained name
- tautomer_locant: Position of indicated hydrogen (e.g., 1 for 1H-indole), or None
- ring_system: Classification (benzo-5-membered, benzo-6-membered, tricyclic, etc.)
- parent_atoms: Number of heavy atoms in parent ring system
"""

from typing import Dict, Optional, Tuple, List, Any
from rdkit import Chem


# Fused heterocycle data - canonical SMILES verified with RDKit
# Format: canonical_smiles -> {name, tautomer_locant, ring_system, parent_atoms}
FUSED_HETEROCYCLE_DATA: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # BENZO-FUSED 5-MEMBERED RINGS (N-containing, aromatic)
    # =========================================================================

    # Indole: benzo[b]pyrrole - N at position 1
    'c1ccc2[nH]ccc2c1': {
        'name': '1H-indole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # 1H-Isoindole: benzo[c]pyrrole - N at position 2 (aromatic tautomer)
    'c1ccc2c[nH]cc2c1': {
        'name': '1H-isoindole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # 2H-Isoindole: non-aromatic in 5-ring
    'C1=Nc2ccccc2C1': {
        'name': '2H-isoindole',
        'tautomer_locant': 2,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Indazole: benzo[c]pyrazole - N at positions 1,2
    'c1ccc2[nH]ncc2c1': {
        'name': '1H-indazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzimidazole: benzo[d]imidazole - N at positions 1,3
    'c1ccc2[nH]cnc2c1': {
        'name': '1H-benzimidazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzotriazole: benzo[d][1,2,3]triazole - N at positions 1,2,3
    'c1ccc2[nH]nnc2c1': {
        'name': '1H-benzotriazole',
        'tautomer_locant': 1,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # =========================================================================
    # BENZO-FUSED 5-MEMBERED RINGS (O/S-containing)
    # =========================================================================

    # Benzofuran: benzo[b]furan
    'c1ccc2occc2c1': {
        'name': '1-benzofuran',
        'tautomer_locant': None,  # No tautomeric H
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzothiophene: benzo[b]thiophene
    'c1ccc2sccc2c1': {
        'name': '1-benzothiophene',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzoxazole: benzo[d]oxazole
    'c1ccc2ocnc2c1': {
        'name': '1,3-benzoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzisoxazole: benzo[c]isoxazole (1,2-benzisoxazole)
    'c1ccc2nocc2c1': {
        'name': '1,2-benzisoxazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzothiazole: benzo[d]thiazole
    'c1ccc2scnc2c1': {
        'name': '1,3-benzothiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # Benzisothiazole: benzo[c]isothiazole (1,2-benzisothiazole)
    'c1ccc2nscc2c1': {
        'name': '1,2-benzisothiazole',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-membered',
        'parent_atoms': 9,
    },

    # =========================================================================
    # BENZO-FUSED 6-MEMBERED RINGS
    # =========================================================================

    # Quinoline: benzo[b]pyridine
    'c1ccc2ncccc2c1': {
        'name': 'quinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
    },

    # Isoquinoline: benzo[c]pyridine
    'c1ccc2cnccc2c1': {
        'name': 'isoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
    },

    # Quinazoline: benzo[d]pyrimidine
    'c1ccc2ncncc2c1': {
        'name': 'quinazoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
    },

    # Quinoxaline: benzo[e]pyrazine
    'c1ccc2nccnc2c1': {
        'name': 'quinoxaline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
    },

    # Cinnoline: benzo[c]pyridazine
    'c1ccc2cnncc2c1': {
        'name': 'cinnoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
    },

    # Phthalazine: benzo[d]pyridazine
    'c1ccc2nnccc2c1': {
        'name': 'phthalazine',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-membered',
        'parent_atoms': 10,
    },

    # =========================================================================
    # NAPHTHYRIDINES (pyridopyridines)
    # =========================================================================

    # 1,5-naphthyridine
    'c1cnc2ccncc2c1': {
        'name': '1,5-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
    },

    # 1,7-naphthyridine
    'c1cc2ccncc2cn1': {
        'name': '1,7-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
    },

    # 1,8-naphthyridine
    'c1cnc2nccnc2c1': {
        'name': '1,8-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
    },

    # 2,7-naphthyridine
    'c1cnc2cnccc2c1': {
        'name': '2,7-naphthyridine',
        'tautomer_locant': None,
        'ring_system': 'naphthyridine',
        'parent_atoms': 10,
    },

    # =========================================================================
    # TRICYCLIC SYSTEMS
    # =========================================================================

    # Carbazole: dibenzo[b,d]pyrrole
    'c1ccc2c(c1)[nH]c1ccccc12': {
        'name': '9H-carbazole',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 13,
    },

    # Acridine: dibenzo[b,e]pyridine
    'c1ccc2nc3ccccc3cc2c1': {
        'name': 'acridine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
    },

    # Phenazine: dibenzo[b,e]pyrazine
    'c1ccc2nc3ccccc3nc2c1': {
        'name': 'phenazine',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
    },

    # Phenoxazine: dibenzo[b,e][1,4]oxazine
    'c1ccc2c(c1)Nc1ccccc1O2': {
        'name': '10H-phenoxazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
    },

    # Phenothiazine: dibenzo[b,e][1,4]thiazine
    'c1ccc2c(c1)Nc1ccccc1S2': {
        'name': '10H-phenothiazine',
        'tautomer_locant': 10,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
    },

    # Xanthene: dibenzo[b,e]pyran (9H-xanthene)
    'c1ccc2c(c1)Cc1ccccc1O2': {
        'name': '9H-xanthene',
        'tautomer_locant': 9,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
    },

    # Thianthrene: dibenzo[b,e][1,4]dithiine
    'c1ccc2c(c1)Sc1ccccc1S2': {
        'name': 'thianthrene',
        'tautomer_locant': None,
        'ring_system': 'tricyclic',
        'parent_atoms': 14,
    },

    # =========================================================================
    # N-BRIDGEHEAD SYSTEMS
    # =========================================================================

    # Indolizine: pyrrolo[1,2-a]pyridine
    'c1ccn2cccc2c1': {
        'name': 'indolizine',
        'tautomer_locant': None,
        'ring_system': 'bridgehead',
        'parent_atoms': 9,
    },

    # =========================================================================
    # PURINES AND PTERIDINES (nucleobase-related)
    # =========================================================================

    # Purine: imidazo[4,5-d]pyrimidine
    'c1ncc2nc[nH]c2n1': {
        'name': '9H-purine',
        'tautomer_locant': 9,
        'ring_system': 'purine',
        'parent_atoms': 9,
    },

    # Pteridine: pyrimido[4,5-b]pyrazine
    'c1cnc2ncncc2n1': {
        'name': 'pteridine',
        'tautomer_locant': None,
        'ring_system': 'pteridine',
        'parent_atoms': 10,
    },

    # =========================================================================
    # PARTIALLY SATURATED (dihydro, tetrahydro) VARIANTS
    # =========================================================================

    # Indoline: 2,3-dihydro-1H-indole
    'c1ccc2c(c1)CCN2': {
        'name': 'indoline',
        'tautomer_locant': None,  # Saturated, no tautomeric H
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
    },

    # Isoindoline: 1,3-dihydro-2H-isoindole
    'c1ccc2c(c1)CNC2': {
        'name': 'isoindoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-5-saturated',
        'parent_atoms': 9,
    },

    # 1,2,3,4-Tetrahydroquinoline
    'c1ccc2c(c1)CCCN2': {
        'name': '1,2,3,4-tetrahydroquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
    },

    # 1,2,3,4-Tetrahydroisoquinoline
    'c1ccc2c(c1)CCNC2': {
        'name': '1,2,3,4-tetrahydroisoquinoline',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
    },

    # Chromane: 3,4-dihydro-2H-chromene
    'c1ccc2c(c1)CCCO2': {
        'name': 'chromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
    },

    # Isochromane: 3,4-dihydro-1H-isochromene
    'c1ccc2c(c1)CCOC2': {
        'name': 'isochromane',
        'tautomer_locant': None,
        'ring_system': 'benzo-6-saturated',
        'parent_atoms': 10,
    },
}


# Build SMARTS patterns for substructure matching
# Use the same SMILES but allow variable substituents
def _build_smarts_lookup() -> Dict[str, str]:
    """
    Build SMARTS patterns from canonical SMILES for substructure matching.

    Returns dict mapping canonical SMILES to core name.
    """
    return {smiles: data['name'] for smiles, data in FUSED_HETEROCYCLE_DATA.items()}


# Cache for substructure matching patterns
_SUBSTRUCTURE_PATTERNS: Dict[str, Chem.Mol] = {}


def _get_substructure_patterns() -> Dict[str, Chem.Mol]:
    """
    Get cached substructure patterns.

    Returns dict mapping canonical SMILES to RDKit Mol patterns.
    """
    global _SUBSTRUCTURE_PATTERNS
    if not _SUBSTRUCTURE_PATTERNS:
        for smiles in FUSED_HETEROCYCLE_DATA:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                _SUBSTRUCTURE_PATTERNS[smiles] = mol
    return _SUBSTRUCTURE_PATTERNS


def get_fused_heterocycle_name(mol: Chem.Mol) -> Optional[Tuple[str, Optional[int]]]:
    """
    Get retained name for an exact fused heterocycle match.

    Checks if the molecule is an unsubstituted fused heterocycle with a
    retained name. For substituted molecules, use match_fused_heterocycle_core.

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (name, tautomer_locant) if found, None otherwise.
        tautomer_locant is the position of indicated hydrogen (e.g., 1 for 1H-indole),
        or None if no tautomeric hydrogen.

    Example:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> get_fused_heterocycle_name(mol)
        ('1H-indole', 1)

        >>> mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline
        >>> get_fused_heterocycle_name(mol)
        ('quinoline', None)
    """
    if mol is None:
        return None

    # Canonicalize input
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    # Check exact match
    if canonical_smiles in FUSED_HETEROCYCLE_DATA:
        data = FUSED_HETEROCYCLE_DATA[canonical_smiles]
        return (data['name'], data['tautomer_locant'])

    return None


def match_fused_heterocycle_core(mol: Chem.Mol) -> Optional[Tuple[str, Dict[int, int]]]:
    """
    Match a substituted molecule against fused heterocycle cores.

    Uses substructure matching to identify if a molecule contains a known
    fused heterocycle core. Returns the core name and atom index mapping
    for locant assignment.

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (core_name, atom_mapping) if a core is matched, None otherwise.
        atom_mapping maps mol atom indices to IUPAC locants in the core.

    Example:
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        >>> name, mapping = match_fused_heterocycle_core(mol)
        >>> name
        '1H-indole'
        >>> # mapping: {mol_atom_idx: core_locant}
    """
    if mol is None:
        return None

    patterns = _get_substructure_patterns()
    best_match: Optional[Tuple[str, List[int], int]] = None

    # Find the largest matching core
    for smiles, pattern in patterns.items():
        if mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                match = matches[0]  # Take first match
                data = FUSED_HETEROCYCLE_DATA[smiles]
                core_size = data['parent_atoms']

                # Keep the largest matching core
                if best_match is None or core_size > best_match[2]:
                    best_match = (data['name'], list(match), core_size)

    if best_match is None:
        return None

    name, match_atoms, _ = best_match

    # Build atom index to locant mapping
    # IUPAC locants are 1-indexed, following the match order
    atom_mapping = {atom_idx: locant + 1 for locant, atom_idx in enumerate(match_atoms)}

    return (name, atom_mapping)


def get_fused_heterocycle_info(canonical_smiles: str) -> Optional[Dict[str, Any]]:
    """
    Get full information about a fused heterocycle from canonical SMILES.

    Args:
        canonical_smiles: Canonical SMILES string

    Returns:
        Dict with name, tautomer_locant, ring_system, parent_atoms
        or None if not found.
    """
    return FUSED_HETEROCYCLE_DATA.get(canonical_smiles)


def is_fused_heterocycle(mol: Chem.Mol) -> bool:
    """
    Check if molecule is a known fused heterocycle (exact match only).

    Args:
        mol: RDKit molecule object

    Returns:
        True if molecule is a known fused heterocycle
    """
    if mol is None:
        return False
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles in FUSED_HETEROCYCLE_DATA


def get_ring_system_type(mol: Chem.Mol) -> Optional[str]:
    """
    Get the ring system type classification for a fused heterocycle.

    Args:
        mol: RDKit molecule object

    Returns:
        Ring system type string (e.g., 'benzo-5-membered', 'tricyclic'),
        or None if not a known fused heterocycle
    """
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    data = FUSED_HETEROCYCLE_DATA.get(canonical_smiles)
    if data:
        return data['ring_system']
    return None
