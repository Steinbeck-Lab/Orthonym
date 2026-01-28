"""
Chain detection and principal chain selection.

Implements IUPAC 2013 rules for selecting the principal chain.
Key change in IUPAC 2013: Chain length takes priority over unsaturation!
"""

from typing import Dict, List, Optional, Set, Tuple
from rdkit import Chem


def find_all_carbon_chains(mol, min_length: int = 1) -> List[List[int]]:
    """
    Find all carbon chains in a molecule using DFS.
    
    Args:
        mol: RDKit Mol object
        min_length: Minimum chain length to return
        
    Returns:
        List of lists, each inner list contains atom indices of a chain
    """
    chains = []
    
    def dfs(atom_idx: int, visited: Set[int], path: List[int]):
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Only follow carbon atoms (not heteroatoms)
        if atom.GetSymbol() != 'C':
            return
        
        visited.add(atom_idx)
        path.append(atom_idx)
        
        # Record this path if it meets minimum length
        if len(path) >= min_length:
            chains.append(path.copy())
        
        # Explore neighbors
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path)
        
        # Backtrack
        path.pop()
        visited.discard(atom_idx)
    
    # Start DFS from each carbon atom to find all possible chains
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            dfs(atom.GetIdx(), set(), [])
    
    return chains


def find_longest_carbon_chain(mol) -> List[int]:
    """
    Find the longest continuous carbon chain.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        List of atom indices forming the longest chain
    """
    def dfs(atom_idx: int, visited: Set[int], path: List[int], results: List[List[int]]):
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom.GetSymbol() != 'C':
            return
        
        visited.add(atom_idx)
        path.append(atom_idx)
        
        # Update longest if this path is longer
        if len(path) > len(results[0]):
            results[0] = path.copy()
        
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path, results)
        
        path.pop()
        visited.discard(atom_idx)
    
    results = [[]]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            dfs(atom.GetIdx(), set(), [], results)
    
    return results[0]


def find_principal_chain(
    mol,
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str] = None
) -> List[int]:
    """
    Find the principal chain following IUPAC 2013 rules.
    
    Selection criteria (in order of priority):
    1. Contains principal characteristic group
    2. Maximum number of principal groups
    3. Maximum chain length (IUPAC 2013: length BEFORE unsaturation!)
    4. Maximum multiple bonds (double + triple)
    5. Maximum double bonds
    6. Lowest locants for principal groups (first point of difference)
    7. Lowest locants for multiple bonds
    8. Maximum substituents
    9. Lowest locants for substituents
    
    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of principal functional group (or None)
        
    Returns:
        List of atom indices forming the principal chain, in order
    """
    chains = find_all_carbon_chains(mol, min_length=1)
    
    if not chains:
        return []
    
    # Get atoms belonging to principal functional group
    fg_atoms = set()
    if principal_group and principal_group in functional_groups:
        for match in functional_groups[principal_group]:
            fg_atoms.update(match)
    
    def count_bonds_in_chain(chain: List[int]) -> Tuple[int, int]:
        """Count double and triple bonds within the chain."""
        double_bonds = 0
        triple_bonds = 0
        
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond:
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.DOUBLE:
                    double_bonds += 1
                elif bond_type == Chem.BondType.TRIPLE:
                    triple_bonds += 1
        
        return double_bonds, triple_bonds
    
    def chain_score(chain: List[int]) -> tuple:
        """
        Calculate selection score for a chain.
        Returns tuple for comparison (higher = better).
        """
        chain_set = set(chain)
        
        # Criterion 1: Contains principal group
        contains_fg = 1 if (fg_atoms & chain_set) else 0
        
        # Criterion 2: Count of principal groups in chain
        fg_count = len(fg_atoms & chain_set)
        
        # Criterion 3: Chain length (IUPAC 2013 prioritizes length!)
        length = len(chain)
        
        # Criterion 4 & 5: Count multiple bonds
        double_bonds, triple_bonds = count_bonds_in_chain(chain)
        multiple_bonds = double_bonds + triple_bonds
        
        return (contains_fg, fg_count, length, multiple_bonds, double_bonds)
    
    # Find chain with highest score
    best_chain = max(chains, key=chain_score)
    
    # Determine numbering direction (lowest locants for principal group)
    if principal_group and fg_atoms:
        best_chain = _orient_chain_for_lowest_locants(best_chain, fg_atoms)
    
    return best_chain


def _orient_chain_for_lowest_locants(chain: List[int], priority_atoms: Set[int]) -> List[int]:
    """
    Orient chain so priority atoms have lowest locants.
    
    Uses first-point-of-difference rule.
    """
    forward_locants = [
        i + 1 for i, idx in enumerate(chain)
        if idx in priority_atoms
    ]
    reverse_locants = [
        len(chain) - i for i, idx in enumerate(chain)
        if idx in priority_atoms
    ]
    
    # Compare using first-point-of-difference
    if _compare_locants(reverse_locants, forward_locants):
        return list(reversed(chain))
    return chain


def _compare_locants(set_a: List[int], set_b: List[int]) -> bool:
    """
    Compare two locant sets using first-point-of-difference rule.
    
    Returns True if set_a is preferred (lower).
    """
    a_sorted = sorted(set_a)
    b_sorted = sorted(set_b)
    
    for a, b in zip(a_sorted, b_sorted):
        if a < b:
            return True
        if a > b:
            return False
    
    # If all compared elements are equal, shorter set wins
    return len(a_sorted) <= len(b_sorted)


def get_substituents(mol, main_chain: List[int]) -> Dict[int, List[List[int]]]:
    """
    Find substituents attached to the main chain.
    
    Args:
        mol: RDKit Mol object
        main_chain: List of atom indices in main chain (ordered)
        
    Returns:
        Dict mapping chain position (1-indexed) to list of substituent atom lists.
        Each substituent is represented as a list of its atom indices.
    """
    chain_set = set(main_chain)
    substituents = {}
    
    for position, chain_idx in enumerate(main_chain, 1):
        chain_atom = mol.GetAtomWithIdx(chain_idx)
        position_subs = []
        
        for neighbor in chain_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            
            # Skip atoms that are part of the main chain
            if nbr_idx in chain_set:
                continue
            
            # BFS to find full substituent
            sub_atoms = _bfs_substituent(mol, nbr_idx, chain_set)
            position_subs.append(sub_atoms)
        
        if position_subs:
            substituents[position] = position_subs
    
    return substituents


def _bfs_substituent(mol, start_idx: int, exclude_set: Set[int]) -> List[int]:
    """
    Find all atoms in a substituent using BFS.
    
    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        exclude_set: Set of atom indices to exclude (main chain atoms)
        
    Returns:
        List of atom indices in the substituent
    """
    visited = {start_idx}
    queue = [start_idx]
    
    while queue:
        current = queue.pop(0)
        atom = mol.GetAtomWithIdx(current)
        
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude_set:
                visited.add(nbr_idx)
                queue.append(nbr_idx)
    
    return list(visited)


def get_chain_atoms_with_locants(chain: List[int]) -> Dict[int, int]:
    """
    Create mapping from atom index to locant number.
    
    Args:
        chain: Ordered list of atom indices
        
    Returns:
        Dict mapping atom_idx -> locant (1-indexed)
    """
    return {atom_idx: locant for locant, atom_idx in enumerate(chain, 1)}
