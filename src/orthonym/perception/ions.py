"""
Ion and radical detection for IUPAC nomenclature.

This module provides functions to detect and classify charged and radical species.
Detection happens early in the naming pipeline to route molecules to the correct
naming path (neutral, ion, zwitterion, salt, or radical).

IUPAC 2013 rules:
- Cations: named with suffix -ium or -ylium
- Anions: named with suffix -ide, -ate, or -uide
- Radicals: named with suffix -yl
- Salts: named as "cation anion" (e.g., sodium acetate)
- Zwitterions: named as neutral compounds with +/- indicated
"""

from typing import Any, Dict, List, Optional
from rdkit import Chem


def detect_species_type(mol) -> str:
    """
    Detect the type of charged/radical species.

    Classification priority:
    1. Radical - any atom with unpaired electrons
    2. Salt - multiple fragments with opposite charges
    3. Ion - net non-zero charge (single fragment)
    4. Zwitterion - net zero charge but has both + and - atoms
    5. Neutral - no charges or radicals

    Args:
        mol: RDKit Mol object

    Returns:
        One of: 'radical', 'salt', 'ion', 'zwitterion', 'neutral'

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> detect_species_type(mol)
        'ion'
        >>> mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        >>> detect_species_type(mol)
        'salt'
    """
    if mol is None:
        return 'neutral'

    # Check for radicals first (highest priority)
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return 'radical'

    # Get molecular fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    if len(frags) > 1:
        # Multiple fragments - check if salt (opposite charges)
        has_positive = False
        has_negative = False

        for frag in frags:
            frag_charge = Chem.GetFormalCharge(frag)
            if frag_charge > 0:
                has_positive = True
            elif frag_charge < 0:
                has_negative = True

        if has_positive and has_negative:
            return 'salt'

    # Single fragment or multi-fragment without opposite charges
    net_charge = Chem.GetFormalCharge(mol)

    if net_charge != 0:
        return 'ion'

    # Check for zwitterion (net zero but has both + and - atoms)
    has_positive_atom = False
    has_negative_atom = False

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            has_positive_atom = True
        elif charge < 0:
            has_negative_atom = True

    if has_positive_atom and has_negative_atom:
        return 'zwitterion'

    return 'neutral'


def get_ion_sites(mol) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all charged atom sites in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary with 'cations' and 'anions' lists.
        Each entry contains:
        - atom_idx: int - atom index in molecule
        - charge: int - formal charge (+1, -1, +2, etc.)
        - element: str - element symbol (N, O, C, etc.)
        - hybridization: str - hybridization state (SP3, SP2, etc.)
        - n_hydrogens: int - number of attached hydrogens

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> sites = get_ion_sites(mol)
        >>> sites['cations'][0]['element']
        'N'
        >>> sites['cations'][0]['charge']
        1
    """
    result: Dict[str, List[Dict[str, Any]]] = {
        'cations': [],
        'anions': []
    }

    if mol is None:
        return result

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()

        if charge == 0:
            continue

        site_info = {
            'atom_idx': atom.GetIdx(),
            'charge': charge,
            'element': atom.GetSymbol(),
            'hybridization': str(atom.GetHybridization()),
            'n_hydrogens': atom.GetTotalNumHs()
        }

        if charge > 0:
            result['cations'].append(site_info)
        else:
            result['anions'].append(site_info)

    return result


def get_radical_sites(mol) -> List[Dict[str, Any]]:
    """
    Get all radical centers in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        List of radical site dictionaries, each containing:
        - atom_idx: int - atom index in molecule
        - n_electrons: int - number of unpaired electrons (1, 2, 3)
        - element: str - element symbol
        - radical_type: str - 'monovalent' (1), 'divalent' (2), 'trivalent' (3)
        - hybridization: str - hybridization state

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> sites = get_radical_sites(mol)
        >>> sites[0]['element']
        'C'
        >>> sites[0]['radical_type']
        'monovalent'
    """
    result: List[Dict[str, Any]] = []

    if mol is None:
        return result

    radical_type_map = {
        1: 'monovalent',
        2: 'divalent',
        3: 'trivalent'
    }

    for atom in mol.GetAtoms():
        n_radical = atom.GetNumRadicalElectrons()

        if n_radical == 0:
            continue

        site_info = {
            'atom_idx': atom.GetIdx(),
            'n_electrons': n_radical,
            'element': atom.GetSymbol(),
            'radical_type': radical_type_map.get(n_radical, f'{n_radical}-valent'),
            'hybridization': str(atom.GetHybridization())
        }

        result.append(site_info)

    return result


def parse_salt_fragments(mol) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a salt into its cation and anion fragments.

    For multi-component salts (dot-separated SMILES), this separates
    the positively and negatively charged fragments.

    Args:
        mol: RDKit Mol object (may contain multiple fragments)

    Returns:
        Dictionary with 'cations' and 'anions' lists.
        Each entry contains:
        - mol: RDKit Mol object for the fragment
        - charge: int - net charge of the fragment
        - smiles: str - canonical SMILES of the fragment

    Example:
        >>> mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        >>> frags = parse_salt_fragments(mol)
        >>> len(frags['cations'])
        1
        >>> len(frags['anions'])
        1
        >>> frags['cations'][0]['smiles']
        '[Na+]'
    """
    result: Dict[str, List[Dict[str, Any]]] = {
        'cations': [],
        'anions': []
    }

    if mol is None:
        return result

    # Get molecular fragments as separate molecules
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    for frag in frags:
        charge = Chem.GetFormalCharge(frag)
        smiles = Chem.MolToSmiles(frag)

        frag_info = {
            'mol': frag,
            'charge': charge,
            'smiles': smiles
        }

        if charge > 0:
            result['cations'].append(frag_info)
        elif charge < 0:
            result['anions'].append(frag_info)
        # Neutral fragments are ignored (e.g., water of crystallization)

    return result
