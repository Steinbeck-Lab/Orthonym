"""
Main IUPAC nomenclature generator.

Architecture:
    SMILES → Perception → Classification → Assembly → IUPAC Name
    
This is the inverse of OPSIN's pipeline:
    OPSIN:     Name → Tokenize → Parse → Build Structure
    Orthonym: Structure → Perceive → Classify → Assemble Name
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from .perception.functional_groups import detect_functional_groups
from .perception.chains import find_principal_chain
from .perception.rings import get_ring_systems, get_ring_info, is_aromatic_ring, classify_ring
from .perception.stereo import assign_stereochemistry, get_stereocenters, get_double_bond_stereo
from .rules.seniority import get_principal_group
from .rules.locants import orient_chain, build_atom_to_locant
from .assembly.composer import assemble_name
from .data.retained_names import RETAINED_NAMES


@dataclass
class MolecularFeatures:
    """Container for perceived molecular features."""

    mol: Any  # RDKit Mol object
    smiles: str = ""
    canonical_smiles: str = ""

    # Functional group information
    functional_groups: Dict[str, List[tuple]] = field(default_factory=dict)
    principal_group: Optional[str] = None
    principal_group_atoms: List[tuple] = field(default_factory=list)

    # Chain information
    principal_chain: List[int] = field(default_factory=list)
    atom_to_locant: Dict[int, int] = field(default_factory=dict)
    substituents: Dict[int, List[List[int]]] = field(default_factory=dict)

    # Ring information
    ring_systems: List[set] = field(default_factory=list)
    is_cyclic: bool = False
    is_aromatic: bool = False
    ring_type: Optional[str] = None  # 'cycloalkane', 'cycloalkene', 'aromatic', 'heterocyclic'
    principal_ring: Optional[tuple] = None  # Atom indices of the principal ring
    oriented_ring: Optional[List[int]] = None  # Ring atoms reordered for naming
    ring_substituents: Dict[int, List[List[int]]] = field(default_factory=dict)  # Substituents on ring
    ring_double_bonds: List[tuple] = field(default_factory=list)  # Double bonds in ring
    ring_double_bond_locants: List[int] = field(default_factory=list)  # Locants for ring double bonds

    # Benzene-specific information
    is_benzene: bool = False  # True if principal ring is benzene
    benzene_ring: Optional[tuple] = None  # Atom indices of the benzene ring
    benzene_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_benzene_substituents

    # Polycyclic aromatic information
    polycyclic_name: Optional[str] = None  # Name of PAH parent (naphthalene, etc.)
    polycyclic_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_polycyclic_substituents

    # Heterocycle-specific information
    heterocycle_info: Optional[Dict] = None  # From classify_heterocycle
    oriented_heterocycle: Optional[List[int]] = None  # Ring atoms in IUPAC numbering order
    heterocycle_atom_to_locant: Optional[Dict[int, int]] = None  # Atom idx -> locant mapping
    heterocycle_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_heterocycle_substituents

    # Stereochemistry
    stereocenters: List[dict] = field(default_factory=list)
    double_bond_stereo: List[dict] = field(default_factory=list)

    # Multiple bonds
    double_bonds: List[tuple] = field(default_factory=list)
    triple_bonds: List[tuple] = field(default_factory=list)

    # Polyfunctional compound information
    is_polyfunctional: bool = False  # True if molecule has 2+ distinct functional groups
    non_principal_groups: Dict[str, List[tuple]] = field(default_factory=dict)  # FGs other than principal

    # Ester-specific information
    ester_match: Optional[tuple] = None  # SMARTS match for principal ester group

    # Amide-specific information
    amide_type: Optional[str] = None  # "primary", "secondary", or "tertiary"
    n_substituents: List[Dict] = field(default_factory=list)  # N-substituents from get_n_substituents()


class Orthonym:
    """
    IUPAC nomenclature generator.
    
    Implements the structure-to-name workflow:
    1. Perception: Extract molecular features using RDKit
    2. Classification: Apply IUPAC seniority rules
    3. Assembly: Build name from fragments
    
    Example:
        >>> namer = Orthonym()
        >>> namer.name("CCO")
        'ethanol'
        >>> namer.name("CC(=O)O")
        'acetic acid'
    """
    
    def __init__(self, style: str = "pin"):
        """
        Initialize namer.
        
        Args:
            style: Naming style
                - "pin": Preferred IUPAC Names (IUPAC 2013)
                - "general": General IUPAC (more flexible)
                - "cas": CAS-style naming
        """
        self.style = style
    
    def name(self, smiles: str) -> str:
        """
        Generate IUPAC name from SMILES.
        
        Args:
            smiles: SMILES string representing the molecule
            
        Returns:
            IUPAC systematic name
            
        Raises:
            ValueError: If SMILES is invalid
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Get canonical SMILES for consistent processing
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Check retained names first (benzene, methanol, etc.) unless systematic style requested
        if self.style != "systematic":
            if canonical_smiles in RETAINED_NAMES:
                return RETAINED_NAMES[canonical_smiles]

            # Check for amino acids (standard amino acids use trivial names)
            from .rules.amino_acids import name_amino_acid
            aa_name = name_amino_acid(mol, canonical_smiles)
            if aa_name:
                return aa_name

        # Perceive molecular features
        features = self._perceive(mol, smiles, canonical_smiles)
        
        # Classify: determine naming strategy and principal group
        self._classify(features)
        
        # Assemble: build final name from fragments
        return assemble_name(features, style=self.style)
    
    def _perceive(self, mol, smiles: str, canonical_smiles: str) -> MolecularFeatures:
        """
        Extract molecular features using RDKit.
        
        This is the perception layer - converts structure to features.
        """
        features = MolecularFeatures(
            mol=mol,
            smiles=smiles,
            canonical_smiles=canonical_smiles
        )
        
        # Detect functional groups using SMARTS patterns
        features.functional_groups = detect_functional_groups(mol)
        
        # Detect ring systems
        features.ring_systems = get_ring_systems(mol)
        features.is_cyclic = len(features.ring_systems) > 0
        
        # Check aromaticity
        for ring_system in features.ring_systems:
            if is_aromatic_ring(mol, ring_system):
                features.is_aromatic = True
                break
        
        # Detect multiple bonds
        features.double_bonds = self._find_double_bonds(mol)
        features.triple_bonds = self._find_triple_bonds(mol)

        # Assign CIP stereochemistry labels BEFORE extracting stereo info
        # rdCIPLabeler sets _CIPCode on atoms (R/S) and bonds (E/Z)
        rdCIPLabeler.AssignCIPLabels(mol)

        # Extract stereochemistry (now depends on _CIPCode being set)
        assign_stereochemistry(mol)  # Additional cleanup/assignment
        features.stereocenters = get_stereocenters(mol)
        features.double_bond_stereo = get_double_bond_stereo(mol)

        return features
    
    def _classify(self, features: MolecularFeatures) -> None:
        """
        Apply IUPAC classification rules.

        Determines:
        - Principal characteristic group (highest seniority)
        - Principal chain/ring (following IUPAC 2013 criteria)
        - Substituent positions and identities
        """
        # Determine principal functional group
        pg_name, pg_atoms = get_principal_group(
            features.mol,
            features.functional_groups
        )
        features.principal_group = pg_name
        features.principal_group_atoms = pg_atoms

        # Store ester match if principal group is ester
        if pg_name == "ester" and pg_atoms:
            features.ester_match = pg_atoms[0]  # First ester match

        # Detect polyfunctional compounds (multiple distinct FGs)
        from .rules.polyfunctional import detect_polyfunctional, get_non_principal_groups
        features.is_polyfunctional = detect_polyfunctional(
            features.mol, features.functional_groups
        )
        if features.is_polyfunctional:
            features.non_principal_groups = get_non_principal_groups(
                features.functional_groups, features.principal_group
            )

        # For cyclic molecules, identify principal ring and its type
        if features.is_cyclic:
            # Check for polycyclic aromatics FIRST (naphthalene, anthracene, etc.)
            # These take precedence over single-ring classification
            from .rules.polycyclics import identify_polycyclic, get_polycyclic_substituents
            pah_name = identify_polycyclic(features.mol)
            if pah_name:
                features.polycyclic_name = pah_name
                features.polycyclic_substituents = get_polycyclic_substituents(
                    features.mol, pah_name
                )
                # Set ring type for consistency
                features.ring_type = 'aromatic'
                return  # Skip other ring classification for PAHs

            ring_info = get_ring_info(features.mol)
            atom_rings = ring_info['atom_rings']

            if atom_rings:
                # For simple single-ring molecules, the ring IS the parent
                # TODO: For multi-ring systems, apply selection criteria
                features.principal_ring = atom_rings[0]
                features.ring_type = classify_ring(features.mol, features.principal_ring)

                # Check if this is a benzene ring
                from .rules.benzene import is_benzene_ring, get_benzene_substituents
                if is_benzene_ring(features.mol, features.principal_ring):
                    features.is_benzene = True
                    features.benzene_ring = features.principal_ring
                    features.benzene_substituents = get_benzene_substituents(
                        features.mol, features.principal_ring
                    )
                elif features.ring_type == 'heterocyclic':
                    # Heterocyclic ring: classify, detect substituents, and orient
                    from .rules.heterocycles import (
                        classify_heterocycle,
                        orient_heterocycle_with_substituents,
                        get_heterocycle_substituents,
                    )

                    features.heterocycle_info = classify_heterocycle(
                        features.mol, features.principal_ring
                    )

                    # Detect substituent positions for proper orientation
                    ring_set = set(features.principal_ring)
                    sub_positions = set()
                    for idx in features.principal_ring:
                        atom = features.mol.GetAtomWithIdx(idx)
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetIdx() not in ring_set:
                                sub_positions.add(idx)
                                break

                    # Orient considering substituents for lowest locants
                    oriented, atom_to_locant = orient_heterocycle_with_substituents(
                        features.mol, features.principal_ring, sub_positions
                    )
                    features.oriented_heterocycle = oriented
                    features.heterocycle_atom_to_locant = atom_to_locant

                    # Get substituent details for naming
                    features.heterocycle_substituents = get_heterocycle_substituents(
                        features.mol,
                        features.principal_ring,
                        oriented,
                        atom_to_locant
                    )
                else:
                    # Non-benzene, non-heterocyclic ring: detect substituents and orient
                    from .rules.cycloalkanes import (
                        get_ring_substituents, get_ring_double_bonds,
                        orient_cycloalkane, orient_cycloalkene
                    )

                    # Get ring substituents
                    features.ring_substituents = get_ring_substituents(
                        features.mol, features.principal_ring
                    )

                    # Get ring double bonds
                    features.ring_double_bonds = get_ring_double_bonds(
                        features.mol, features.principal_ring
                    )

                    # Orient the ring based on type
                    if features.ring_type == 'cycloalkane':
                        features.oriented_ring = orient_cycloalkane(
                            features.mol,
                            features.principal_ring,
                            features.ring_substituents
                        )
                    elif features.ring_type == 'cycloalkene':
                        features.oriented_ring = orient_cycloalkene(
                            features.mol,
                            features.principal_ring,
                            features.ring_double_bonds,
                            features.ring_substituents
                        )
                        # Calculate ring double bond locants
                        if features.oriented_ring and features.ring_double_bonds:
                            oriented = features.oriented_ring
                            locants = []
                            for a1, a2 in features.ring_double_bonds:
                                pos1 = oriented.index(a1)
                                pos2 = oriented.index(a2)
                                # Lower position is the locant
                                locants.append(min(pos1, pos2) + 1)
                            features.ring_double_bond_locants = sorted(locants)

        # Find principal chain (for acyclic molecules)
        if not features.is_cyclic:
            features.principal_chain = find_principal_chain(
                features.mol,
                features.functional_groups,
                features.principal_group
            )

            if features.principal_chain:
                # Collect principal group atom indices for orientation
                pg_atom_set: set = set()
                if features.principal_group and features.principal_group in features.functional_groups:
                    for match in features.functional_groups[features.principal_group]:
                        pg_atom_set.update(match)

                # Get initial substituents for orientation criterion (d)
                # This is needed BEFORE orientation to apply lowest-locant rule
                initial_subs = self._find_substituents_by_atom(
                    features.mol,
                    features.principal_chain
                )

                # Orient chain using IUPAC 2013 criteria
                features.principal_chain = orient_chain(
                    chain=features.principal_chain,
                    mol=features.mol,
                    principal_group_atoms=pg_atom_set,
                    double_bonds=features.double_bonds,
                    triple_bonds=features.triple_bonds,
                    substituent_positions=initial_subs,
                )

                # Build atom-to-locant mapping from oriented chain
                features.atom_to_locant = build_atom_to_locant(features.principal_chain)

                # Get substituents with final locant-based positions
                features.substituents = self._find_substituents(
                    features.mol,
                    features.principal_chain
                )
    
    def _find_double_bonds(self, mol) -> List[tuple]:
        """Find all C=C double bonds."""
        double_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                begin = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                # Only C=C double bonds (not C=O, etc.)
                if begin.GetSymbol() == 'C' and end.GetSymbol() == 'C':
                    double_bonds.append((
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()
                    ))
        return double_bonds
    
    def _find_triple_bonds(self, mol) -> List[tuple]:
        """Find all C≡C triple bonds."""
        triple_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                begin = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                # Only C≡C triple bonds (not C≡N)
                if begin.GetSymbol() == 'C' and end.GetSymbol() == 'C':
                    triple_bonds.append((
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()
                    ))
        return triple_bonds
    
    def _find_substituents(self, mol, chain: List[int]) -> Dict[int, List[List[int]]]:
        """Find substituents attached to the principal chain (keyed by position)."""
        from .perception.chains import get_substituents
        return get_substituents(mol, chain)

    def _find_substituents_by_atom(self, mol, chain: List[int]) -> Dict[int, List[List[int]]]:
        """
        Find substituents attached to the principal chain, keyed by atom index.

        This is used for orient_chain() which expects substituent_positions
        keyed by atom indices on the chain, not by position numbers.

        Args:
            mol: RDKit Mol object
            chain: List of atom indices in the principal chain

        Returns:
            Dict mapping chain atom index to list of substituent atom lists
        """
        from .perception.chains import _bfs_substituent

        chain_set = set(chain)
        substituents = {}

        for chain_idx in chain:
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
                substituents[chain_idx] = position_subs

        return substituents


def name_compound(smiles: str, style: str = "pin") -> str:
    """
    Convenience function to generate IUPAC name from SMILES.

    Args:
        smiles: SMILES string
        style: Naming style
            - "pin": Preferred IUPAC Names (default, uses retained names when available)
            - "systematic": Always generate systematic name (bypass retained names)
            - "general": General IUPAC (more flexible)
            - "cas": CAS-style naming

    Returns:
        IUPAC systematic name

    Example:
        >>> name_compound("CCO")
        'ethanol'
        >>> name_compound("CC(=O)O")
        'acetic acid'
        >>> name_compound("c1ccccc1")
        'benzene'
        >>> name_compound("C=CCO")
        'allyl alcohol'
        >>> name_compound("C=CCO", style="systematic")
        'prop-2-en-1-ol'
    """
    namer = Orthonym(style=style)
    return namer.name(smiles)
