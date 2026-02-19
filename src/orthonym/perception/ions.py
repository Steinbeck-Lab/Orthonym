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

    # Gather charge/radical info in one pass
    net_charge = Chem.GetFormalCharge(mol)
    has_any_charge = False
    has_radical = False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            has_any_charge = True
        if atom.GetNumRadicalElectrons() > 0:
            has_radical = True

    # If molecule has formal charges, prioritize ionic classification
    # over radical detection. RDKit may assign radical electrons to
    # certain charged heteroatoms (e.g., [SeH+] gets 2 radical electrons).
    if has_any_charge:
        # Get molecular fragments for salt detection
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            has_pos = any(Chem.GetFormalCharge(f) > 0 for f in frags)
            has_neg = any(Chem.GetFormalCharge(f) < 0 for f in frags)
            if has_pos and has_neg:
                return 'salt'

        if net_charge != 0:
            species_type = 'ion'

            # Guard: large organic molecules with a single minor charge on a
            # heteroatom (protonated amine, quaternary N, etc.) are better
            # served by the normal perceive/classify/assemble pipeline than
            # by the ion naming path.  Ion naming is designed for small
            # standalone ions (ammonium, acetate, methylium) not for
            # "dodecylamine + H+" or "phenylhexylamine + H+".
            #
            # Keep carboxylates and alkoxides as ions because they have
            # dedicated retained-name tables (acetate, benzoate, etc.).
            heavy_atom_count = mol.GetNumHeavyAtoms()
            charge_sites = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
            total_abs_charge = sum(abs(a.GetFormalCharge()) for a in charge_sites)

            from rdkit.Chem import MolFromSmarts
            carboxylate_pat = MolFromSmarts('[O-]C=O')
            alkoxide_pat = MolFromSmarts('[O-]')
            has_carboxylate = mol.HasSubstructMatch(carboxylate_pat) if carboxylate_pat else False
            has_alkoxide = mol.HasSubstructMatch(alkoxide_pat) if alkoxide_pat else False

            if (heavy_atom_count > 10
                    and total_abs_charge <= 1
                    and len(charge_sites) <= 1
                    and not has_carboxylate
                    and not has_alkoxide):
                species_type = 'neutral'

            return species_type

        # Net charge is 0 but has charges -> possible zwitterion
        # Fall through to zwitterion check below
    elif has_radical:
        # No charges at all, just radical electrons -> radical
        return 'radical'

    # Check for zwitterion (net zero but has both + and - atoms)
    # EXCLUDE functional groups with internal charges (nitro, azide, etc.)
    # These are not true zwitterions in IUPAC nomenclature sense
    if _has_true_zwitterion_character(mol):
        return 'zwitterion'

    return 'neutral'


def _has_true_zwitterion_character(mol) -> bool:
    """
    Determine if a molecule has true zwitterionic character.

    A true zwitterion has separated positive and negative sites that are
    NOT part of a single functional group with internal charge distribution.

    Excludes:
    - Nitro groups: [N+](=O)[O-] - internal charge distribution
    - Azide groups: [N-]=[N+]=[N-] - internal charge distribution
    - N-oxides: [N+][O-] directly bonded - internal charge distribution
    - Sulfonyl groups with charge separation

    True zwitterions:
    - Amino acid zwitterions: [NH3+] ... [COO-] separated by carbon(s)
    - Betaines: [N+](C)(C)(C)....[O-] separated by carbons

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule has true zwitterionic character
    """
    if mol is None:
        return False

    # Collect positive and negative atoms
    positive_atoms = []
    negative_atoms = []

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            positive_atoms.append(atom)
        elif charge < 0:
            negative_atoms.append(atom)

    if not positive_atoms or not negative_atoms:
        return False

    # Check if ALL charge pairs are functional group internal charges
    # If any charge pair is NOT directly bonded, it's a true zwitterion
    for pos_atom in positive_atoms:
        for neg_atom in negative_atoms:
            # Check if directly bonded (functional group internal charge)
            bond = mol.GetBondBetweenAtoms(pos_atom.GetIdx(), neg_atom.GetIdx())
            if bond is None:
                # Not directly bonded - could be true zwitterion
                # Check for nitro group pattern: N+ bonded to O= and O-
                if not _is_nitro_or_similar_group(mol, pos_atom, neg_atom):
                    return True

    return False


def _is_nitro_or_similar_group(mol, pos_atom, neg_atom) -> bool:
    """
    Check if the + and - atoms are part of a nitro or similar functional group.

    Nitro group: [N+](=O)[O-] where N+ is bonded to the O- through a shared C
    or directly through a resonance structure.

    Args:
        mol: RDKit Mol object
        pos_atom: Positively charged atom
        neg_atom: Negatively charged atom

    Returns:
        True if atoms are part of a functional group with internal charges
    """
    pos_element = pos_atom.GetSymbol()
    neg_element = neg_atom.GetSymbol()

    # Nitro group: N+ bonded to O= and O-
    # The N+ neighbors should include both the O- and an O=
    if pos_element == 'N' and neg_element == 'O':
        # Check if this N is bonded to both an O= and an O-
        n_neighbors = list(pos_atom.GetNeighbors())
        oxygen_neighbors = [n for n in n_neighbors if n.GetSymbol() == 'O']

        if len(oxygen_neighbors) >= 2:
            # N bonded to 2+ oxygens - likely nitro or nitroso
            has_double_o = False
            has_negative_o = False

            for o_neighbor in oxygen_neighbors:
                bond = mol.GetBondBetweenAtoms(pos_atom.GetIdx(), o_neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_double_o = True
                if o_neighbor.GetFormalCharge() < 0:
                    has_negative_o = True

            if has_double_o and has_negative_o:
                return True

    # N-oxide: N+ directly bonded to O-
    if pos_element == 'N' and neg_element == 'O':
        bond = mol.GetBondBetweenAtoms(pos_atom.GetIdx(), neg_atom.GetIdx())
        if bond is not None:
            return True  # Directly bonded N+-O- is N-oxide or similar

    # Azide: [N-]=[N+]=[N-] pattern
    if pos_element == 'N' and neg_element == 'N':
        # Check if they're part of an azide chain
        n_neighbors = [n for n in pos_atom.GetNeighbors() if n.GetSymbol() == 'N']
        if len(n_neighbors) >= 2:
            return True  # Likely azide

    return False


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
        Dictionary with 'cations', 'anions', and 'neutrals' lists.
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
        'anions': [],
        'neutrals': []
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
        else:
            result['neutrals'].append(frag_info)

    return result
