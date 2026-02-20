"""
Main IUPAC nomenclature generator.

Architecture:
    SMILES → Perception → Classification → Assembly → IUPAC Name

This is the inverse of OPSIN's pipeline:
    OPSIN:     Name → Tokenize → Parse → Build Structure
    Orthonym: Structure → Perceive → Classify → Assemble Name
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

logger = logging.getLogger(__name__)

from .perception.ions import detect_species_type, get_ion_sites, get_radical_sites
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
    senior_ring_system: Optional[tuple] = None  # P-44.2 most senior ring system (atom indices)
    oriented_ring: Optional[List[int]] = None  # Ring atoms reordered for naming
    ring_substituents: Dict[int, List[List[int]]] = field(default_factory=dict)  # Substituents on ring
    ring_double_bonds: List[tuple] = field(default_factory=list)  # Double bonds in ring
    ring_double_bond_locants: List[int] = field(default_factory=list)  # Locants for ring double bonds

    # Ring assembly information (biphenyl, bipyridine, etc.)
    ring_assembly_info: Optional[Dict] = None

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

    # Ring-as-substituent information (when chain is parent per IUPAC P-44.1)
    ring_substituents_as_groups: List[tuple] = field(default_factory=list)  # Rings that become substituents
    chain_is_parent: bool = False  # True when parent selection chose chain over ring

    # Ion/radical species information
    species_type: str = 'neutral'  # 'neutral', 'ion', 'zwitterion', 'salt', 'radical'
    ion_sites: Dict[str, List[Dict]] = field(default_factory=dict)  # From get_ion_sites()
    radical_sites: List[Dict] = field(default_factory=list)  # From get_radical_sites()
    total_charge: int = 0  # Net formal charge of the molecule


def _should_bypass_fused_guard(features, core_match):
    """Check if chain should be parent despite fused heterocycle presence.

    Per IUPAC P-44.1.1: chain wins when it has STRICTLY MORE principal
    characteristic groups than the fused ring system.
    Per IUPAC P-52.2.8: ring wins when PG counts are equal.

    Only applies when the matched fused core is the dominant ring system.
    Molecules with substantial additional ring systems beyond the matched
    core are too complex for simple chain-as-parent bypass.

    Returns True to bypass (chain should be parent), False to keep guard.
    """
    if not features.principal_group or not features.principal_group_atoms:
        return False  # No PG -> ring is parent (P-44.1.2.2)

    core_atoms = set(core_match[1].keys())  # Mol atom indices in fused core

    # Guard 1: if molecule has substantial ring structure beyond the matched
    # core, it's a complex polycyclic — don't bypass the fused guard.
    all_ring_atoms = set()
    for ring in features.mol.GetRingInfo().AtomRings():
        all_ring_atoms.update(ring)
    extra_ring_atoms = len(all_ring_atoms - core_atoms)
    if extra_ring_atoms > 4:
        return False

    # Guard 2: chain must be longer than ring core (P-52.2.8 ring preference
    # for equal size). Non-ring heavy atoms must exceed core atom count.
    non_ring_heavy = features.mol.GetNumHeavyAtoms() - len(all_ring_atoms)
    if non_ring_heavy <= len(core_atoms):
        return False

    # Count PGs on fused ring core (directly on ring or bonded to ring atom)
    ring_pg = 0
    total_pg = 0
    for pg_atoms in features.principal_group_atoms:
        if not pg_atoms:
            continue
        total_pg += 1
        attach = pg_atoms[0]
        if attach in core_atoms:
            ring_pg += 1
            continue
        atom = features.mol.GetAtomWithIdx(attach)
        if any(nbr.GetIdx() in core_atoms for nbr in atom.GetNeighbors()):
            ring_pg += 1

    chain_pg = total_pg - ring_pg
    return chain_pg > ring_pg  # STRICT inequality per P-52.2.8


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

        # EARLY SPECIES DETECTION - before retained names check
        # This routes ionic/radical species to specialized naming paths
        species_type = detect_species_type(mol)

        # Route to specialized naming for ionic/radical species
        if species_type == 'salt':
            from .rules.salts import name_salt
            return name_salt(mol, style=self.style)
        elif species_type == 'radical':
            from .rules.radicals import name_radical
            return name_radical(mol, style=self.style)
        elif species_type == 'zwitterion':
            from .rules.salts import name_zwitterion
            return name_zwitterion(mol, style=self.style)
        elif species_type == 'ion':
            # Single-component ion: check retained names first, then fall through
            from .rules.ions import name_anion, name_cation

            sites = get_ion_sites(mol)

            # Check retained ion names first (acetate, benzoate, etc.)
            if sites['anions'] and not sites['cations']:
                result = name_anion(mol, style=self.style, retained_only=True)
                if result:
                    return result
            elif sites['cations'] and not sites['anions']:
                result = name_cation(mol, style=self.style, retained_only=True)
                if result:
                    return result

            # No retained ion name found.
            # For SINGLE anions in small/medium molecules (HA <= 25), use the
            # dedicated ion naming pipeline (handles aromatic carboxylates like
            # benzoate/naphthoate correctly). For larger molecules, fall through
            # to the neutral naming pipeline which produces more complete names.
            if (len(sites['anions']) == 1 and not sites['cations']
                    and mol.GetNumHeavyAtoms() <= 25):
                # Try the existing ion pipeline first
                ion_result = name_anion(mol, style=self.style)
                if ion_result:
                    return ion_result
                # Ion pipeline returned empty -- try neutralize-then-name
                try:
                    from rdkit.Chem import RWMol
                    from .rules.ions import classify_anion, _acid_name_to_carboxylate
                    from .perception.ions import _get_internal_charge_atoms
                    anion_type = classify_anion(mol, sites['anions'][0])
                    if anion_type == 'carboxylate':
                        internal_charge_atoms = _get_internal_charge_atoms(mol)
                        rwmol = RWMol(mol)
                        for atom in rwmol.GetAtoms():
                            if (atom.GetSymbol() == 'O'
                                    and atom.GetFormalCharge() == -1
                                    and atom.GetIdx() not in internal_charge_atoms):
                                atom.SetFormalCharge(0)
                                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                        neutral_mol = rwmol.GetMol()
                        neutral_smi = Chem.MolToSmiles(neutral_mol, canonical=True)
                        neutral_namer = Orthonym(style=self.style)
                        neutral_name = neutral_namer.name(neutral_smi)
                        if neutral_name:
                            anion_name = _acid_name_to_carboxylate(
                                neutral_name, 1
                            )
                            if anion_name:
                                return anion_name
                except Exception:
                    pass  # Fall through to normal pipeline

            # For POLY-anionic species (2+ anionic sites, e.g., dicarboxylate),
            # try neutralize-then-name: convert [O-] -> OH so carboxylic acid
            # SMARTS can match, then name the neutral form. This prevents
            # empty-string results for polycarboxylate anions where the FG
            # detection only recognizes protonated acids.
            if len(sites['anions']) >= 2 and not sites['cations']:
                try:
                    from rdkit.Chem import RWMol
                    from .rules.ions import classify_anion, _acid_name_to_carboxylate
                    from .perception.ions import _get_internal_charge_atoms
                    internal_charge_atoms = _get_internal_charge_atoms(mol)
                    rwmol = RWMol(mol)
                    neutralized = False
                    for atom in rwmol.GetAtoms():
                        if (atom.GetSymbol() == 'O'
                                and atom.GetFormalCharge() == -1
                                and atom.GetIdx() not in internal_charge_atoms):
                            atom.SetFormalCharge(0)
                            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                            neutralized = True
                    if neutralized:
                        neutral_mol = rwmol.GetMol()
                        neutral_smi = Chem.MolToSmiles(neutral_mol, canonical=True)
                        neutral_namer = Orthonym(style=self.style)
                        neutral_name = neutral_namer.name(neutral_smi)
                        if neutral_name:
                            # IUPAC P-72.2.1: Convert acid suffix to carboxylate
                            # for deprotonated carboxylate sites
                            carboxylate_count = sum(
                                1 for a in sites['anions']
                                if classify_anion(mol, a) == 'carboxylate'
                            )
                            if carboxylate_count > 0:
                                anion_name = _acid_name_to_carboxylate(
                                    neutral_name, carboxylate_count
                                )
                                if anion_name:
                                    return anion_name
                            return neutral_name
                except Exception:
                    pass  # Fall through to normal pipeline

        # Continue with normal neutral molecule naming

        # CHARGE NEUTRALIZATION for large zwitterions reclassified as 'neutral'
        # by the HA>20 + quaternary-N guard in detect_species_type().
        # These molecules still carry formal charges (P-O-, N+) that prevent
        # FG detection SMARTS from matching (e.g., [OX2H] for COOH).
        # Neutralize O- to OH; leave quaternary N+ (no H) charged to
        # preserve valid valence. ONLY applies to molecules that have true
        # zwitterion character -- NOT to molecules with internal charges
        # (nitro [N+](=O)[O-], azide, N-oxide) which are normal functional groups.
        if Chem.GetFormalCharge(mol) == 0:
            from .perception.ions import _has_true_zwitterion_character
            if _has_true_zwitterion_character(mol):
                try:
                    from rdkit.Chem import RWMol
                    rwmol = RWMol(mol)
                    for atom in rwmol.GetAtoms():
                        charge = atom.GetFormalCharge()
                        if charge < 0:
                            # O- -> OH (add H for each negative charge)
                            atom.SetFormalCharge(0)
                            cur_h = atom.GetNumExplicitHs()
                            atom.SetNumExplicitHs(cur_h + abs(charge))
                        elif charge > 0:
                            total_h = atom.GetTotalNumHs()
                            if total_h >= charge:
                                # Protonated amine: remove H to neutralize
                                cur_h = atom.GetNumExplicitHs()
                                atom.SetFormalCharge(0)
                                atom.SetNumExplicitHs(max(0, cur_h - charge))
                            # else: quaternary N+ (no H) -- leave charged
                            # to preserve valid valence
                    Chem.SanitizeMol(rwmol)
                    mol = rwmol.GetMol()
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    pass  # If neutralization fails, continue with original mol

        # MULTIPLICATIVE NOMENCLATURE (IUPAC P-51.3)
        # Detect symmetric molecules with bridge atoms linking identical parent
        # structures (e.g., 4,4'-methylenedianiline). Must come before NP detection
        # and retained names to catch these before half the molecule is dropped.
        from .rules.multiplicative import name_multiplicative
        mult_name = name_multiplicative(mol)
        if mult_name is not None:
            return mult_name

        # NATURAL PRODUCT DETECTION
        # Check before retained names because NP detection uses substructure matching
        # while retained names use exact SMILES matching.
        # Even with style="systematic", NP names are returned (IUPAC 2013 has no
        # systematic PIN for natural products - see P-10).
        from .rules.natural_products import name_natural_product
        np_name = name_natural_product(mol)
        if np_name is not None:
            return np_name

        # PEPTIDE DETECTION
        # Check before retained names and amino acids to prevent peptides from
        # being flattened to single amino acid names. Must come after NP detection.
        from .rules.amino_acids import is_peptide as _is_peptide
        if _is_peptide(mol):
            from .rules.peptides import name_peptide
            peptide_name = name_peptide(mol)
            if peptide_name:
                return peptide_name

        # Check retained names first (benzene, methanol, etc.) unless systematic style requested
        if self.style != "systematic":
            if canonical_smiles in RETAINED_NAMES:
                return RETAINED_NAMES[canonical_smiles]

            # Check for amino acids (standard amino acids use trivial names)
            from .rules.amino_acids import name_amino_acid
            aa_name = name_amino_acid(mol, canonical_smiles)
            if aa_name:
                return aa_name

        # Skeletal replacement nomenclature (IUPAC P-15.4)
        # Detect chains where heteroatoms are embedded in the backbone (C-O-C, C-N-C, C-S-C)
        # Must come BEFORE perception to override substitutive naming (ether/amine prefix)
        from .rules.skeletal_replacement import try_skeletal_replacement_name
        skel_name = try_skeletal_replacement_name(mol)
        if skel_name:
            return skel_name

        # DECOMPOSITION ENGINE (v4.0 Phase 39)
        # For complex molecules with ester/amide/glycosidic bonds that the
        # existing single-pass pipeline cannot name completely, try cleaving
        # at functional bonds and naming fragments individually.
        # Quality gate inside try_decompose() ensures this only activates
        # when the existing pipeline would produce a poor result.
        from .decomposition import try_decompose
        decomposed_name = try_decompose(mol, style=self.style)
        if decomposed_name is not None:
            return decomposed_name

        # Perceive molecular features
        features = self._perceive(mol, smiles, canonical_smiles)

        # Classify: determine naming strategy and principal group
        self._classify(features)

        # Assemble: build final name from fragments
        assembled = assemble_name(features, style=self.style)

        # Final quality gate: if composer produced a garbled name
        # (e.g., "cycloanedicarboxamide" for a multi-amide molecule),
        # fall back to the fragment naming result. Only triggers for
        # names containing known garbled tokens to avoid performance
        # overhead on normal molecules.
        if assembled and mol.GetNumHeavyAtoms() > 15:
            _GARBLED_TOKENS = ('cycloane', 'anedicarboxamide', 'aneyl')
            assembled_lower = assembled.lower()
            if any(tok in assembled_lower for tok in _GARBLED_TOKENS):
                from .assembly.fragment_naming import name_fragment_recursively
                frag_name = name_fragment_recursively(canonical_smiles)
                if frag_name and frag_name != assembled:
                    return frag_name

        return assembled
    
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

        # Add species type detection for ionic/radical compounds
        features.species_type = detect_species_type(mol)
        features.total_charge = Chem.GetFormalCharge(mol)

        if features.species_type in ('ion', 'zwitterion', 'salt'):
            features.ion_sites = get_ion_sites(mol)
        if features.species_type == 'radical':
            features.radical_sites = get_radical_sites(mol)

        # Detect functional groups using SMARTS patterns
        features.functional_groups = detect_functional_groups(mol)

        # Filter consumed atoms to prevent double-counting
        # (e.g., acid halide Cl should not also appear as "chloro" prefix)
        features.functional_groups = _filter_consumed_fg_atoms(features.functional_groups)

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

        # Store ester match(es) if principal group is ester
        if pg_name == "ester" and pg_atoms:
            features.ester_match = pg_atoms[0]  # First ester match (backward compat)
            features.all_ester_matches = pg_atoms  # All ester matches

        # Detect polyfunctional compounds (multiple distinct FGs)
        from .rules.polyfunctional import detect_polyfunctional, get_non_principal_groups
        features.is_polyfunctional = detect_polyfunctional(
            features.mol, features.functional_groups
        )
        if features.is_polyfunctional:
            features.non_principal_groups = get_non_principal_groups(
                features.functional_groups, features.principal_group
            )

        # Parent selection for molecules with ring AND functionalized chain (IUPAC P-44.1)
        # This must happen BEFORE ring classification to potentially redirect to chain naming
        # EXCEPTION: Skip parent selection for known fused heterocycles (indole, quinoline, etc.)
        # UNLESS chain has strictly more principal groups than ring (P-44.1.1 override)
        if features.is_cyclic and features.principal_group:
            from .rules.parent_selection import select_parent
            from .rules.fused_rings import classify_fused_system
            from .data.fused_heterocycles import match_fused_heterocycle_core

            # Check if this is a known fused heterocycle
            fused_type = classify_fused_system(features.mol)
            is_known_fused_heterocycle = False

            if fused_type in ('ortho-fused', 'ortho-peri-fused'):
                core_match = match_fused_heterocycle_core(features.mol)
                if core_match is not None:
                    # P-44.1.1 / P-52.2.8: bypass guard when chain has
                    # strictly more principal groups than the fused ring
                    if _should_bypass_fused_guard(features, core_match):
                        is_known_fused_heterocycle = False
                    else:
                        is_known_fused_heterocycle = True

            # Only do parent selection for non-fused systems or unknown fused systems
            if not is_known_fused_heterocycle:
                # Get ring atoms to exclude when finding chain
                all_ring_atoms = set()
                for ring in features.ring_systems:
                    all_ring_atoms.update(ring)

                # Find potential principal chain (excluding ring atoms)
                # IMPORTANT: namer.py does the chain finding, then passes result to select_parent()
                potential_chain = find_principal_chain(
                    features.mol,
                    features.functional_groups,
                    features.principal_group,
                    exclude_atoms=all_ring_atoms
                )

                # Only do parent selection if we found a meaningful chain (>= 2 carbons)
                if potential_chain and len(potential_chain) >= 2:
                    # Pass pre-computed chain to select_parent
                    selection = select_parent(
                        mol=features.mol,
                        ring_systems=features.ring_systems,
                        principal_chain=potential_chain,
                        principal_group=features.principal_group,
                        principal_group_atoms=features.principal_group_atoms
                    )

                    if selection.parent_type == 'chain':
                        # Chain wins - switch from ring naming to chain naming
                        features.chain_is_parent = True
                        features.is_cyclic = False  # Disable ring naming path
                        features.principal_chain = selection.parent_atoms
                        features.ring_substituents_as_groups = selection.substituent_rings
                        # Continue with chain classification below (is_cyclic is now False)

        # For cyclic molecules, identify principal ring and its type
        if features.is_cyclic:
            # Check for ring assemblies FIRST (identical disconnected ring systems)
            # Must come before fused/polycyclic classification because ring assemblies
            # have 2+ separate ring systems that would otherwise be misrouted
            if len(features.ring_systems) >= 2:
                from .rules.ring_assemblies import detect_ring_assembly
                assembly_info = detect_ring_assembly(features.mol, features.ring_systems)
                if assembly_info:
                    features.ring_assembly_info = assembly_info
                    return  # Skip other ring classification for assemblies

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
                # Select the most senior ring system per IUPAC P-44.2
                # and store it for downstream use (e.g., ring-vs-chain
                # comparison in the composer). The monocyclic dispatch
                # path below uses atom_rings[0] which preserves SSSR
                # cyclic traversal order needed by orientation functions.
                # Complex multi-ring (fused/bridged) systems are handled
                # by _classify_complex_ring() in the composer.
                from .rules.ring_selection import select_principal_ring_system
                principal = select_principal_ring_system(
                    features.mol, features.ring_systems
                )
                features.senior_ring_system = principal if principal else atom_rings[0]

                # For multi-ring-system molecules, use a SSSR ring from
                # the senior system as principal_ring when the senior
                # system is strictly LARGER than the default ring's system.
                # This ensures fused/bridged senior systems (imidazopyridine,
                # xanthene) take precedence over small monocyclic rings
                # (benzene) per IUPAC P-44.2.
                #
                # Guards (all must pass to switch):
                # (a) Senior system must be significantly larger (>= 3
                #     atoms) than the default — marginal differences
                #     (1-2 atoms) cause churn without improving names.
                # (b) Principal FG must NOT be attached to the default
                #     ring system — per P-44.1, the parent must contain
                #     the principal characteristic group.
                if principal and len(features.ring_systems) >= 2:
                    senior_set = set(principal)
                    # Find the ring system that contains the default ring
                    default_system = set(atom_rings[0])
                    default_system_size = len(atom_rings[0])
                    for rs in features.ring_systems:
                        if set(atom_rings[0]).issubset(rs):
                            default_system = rs
                            default_system_size = len(rs)
                            break

                    # Guard (a): senior system must be >= 3 atoms larger
                    size_diff = len(senior_set) - default_system_size
                    size_ok = size_diff >= 3

                    # Guard (b): PG must not be on the default system
                    # (P-44.1: parent must contain the principal group)
                    from .rules.parent_selection import is_principal_group_on_ring
                    pg_on_default = False
                    if features.principal_group_atoms:
                        pg_on_default = is_principal_group_on_ring(
                            features.mol, default_system,
                            features.principal_group_atoms
                        )

                    if size_ok and not pg_on_default:
                        best_ring = atom_rings[0]
                        best_overlap = 0
                        for ring in atom_rings:
                            overlap = len(set(ring) & senior_set)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_ring = ring
                        features.principal_ring = best_ring
                    else:
                        features.principal_ring = atom_rings[0]
                else:
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
                        # Determine principal group atoms on the ring
                        # IUPAC P-31.1.3.4: principal group gets lowest locant
                        # We identify ring C atoms that directly bear the principal
                        # group's characteristic heteroatom (e.g., C=O for ketone,
                        # C-OH for alcohol). The characteristic heteroatom must be
                        # bonded DIRECTLY to a ring carbon (not via an exocyclic C).
                        # Exocyclic groups (aldehyde -CHO, -COOH) use -carbaldehyde/
                        # -carboxylic acid suffixes and don't override ring numbering.
                        pg_ring_atoms = set()
                        if features.principal_group and features.principal_group in features.functional_groups:
                            ring_set = set(features.principal_ring)
                            for match in features.functional_groups[features.principal_group]:
                                match_set = set(match)
                                for atom_idx in match:
                                    if atom_idx in ring_set:
                                        atom = features.mol.GetAtomWithIdx(atom_idx)
                                        if atom.GetSymbol() != 'C':
                                            continue
                                        # Check if this ring C is bonded to a non-ring
                                        # HETEROATOM that is in the FG match
                                        for nbr in atom.GetNeighbors():
                                            nbr_idx = nbr.GetIdx()
                                            if (nbr_idx not in ring_set
                                                    and nbr_idx in match_set
                                                    and nbr.GetSymbol() != 'C'):
                                                pg_ring_atoms.add(atom_idx)
                                                break

                        features.oriented_ring = orient_cycloalkene(
                            features.mol,
                            features.principal_ring,
                            features.ring_double_bonds,
                            features.ring_substituents,
                            principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None
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
        # Skip if chain was already set by parent selection (chain_is_parent = True)
        if not features.is_cyclic:
            if not features.chain_is_parent:
                # Normal acyclic molecule - find principal chain
                # Guard: exclude ring atoms even in "acyclic" path (defensive)
                _ring_exclude = set()
                ri = features.mol.GetRingInfo()
                for ring in ri.AtomRings():
                    _ring_exclude.update(ring)
                features.principal_chain = find_principal_chain(
                    features.mol,
                    features.functional_groups,
                    features.principal_group,
                    exclude_atoms=_ring_exclude if _ring_exclude else None
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

        # INST-05: Naming decision trace
        if logger.isEnabledFor(logging.DEBUG):
            parent_type = 'chain' if getattr(features, 'chain_is_parent', False) else 'ring'
            parent_atoms = (features.principal_chain if parent_type == 'chain'
                            else features.principal_ring or [])
            sub_count = (sum(len(v) for v in features.substituents.values())
                         if features.substituents else 0)
            logger.debug(
                "NAMING_DECISION: smiles=%s parent_type=%s parent_size=%d "
                "principal_group=%s fg_count=%d sub_count=%d is_cyclic=%s",
                features.canonical_smiles,
                parent_type,
                len(parent_atoms) if parent_atoms else 0,
                features.principal_group,
                len(features.functional_groups),
                sub_count,
                features.is_cyclic,
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


def _filter_consumed_fg_atoms(functional_groups: dict) -> dict:
    """Filter out functional group matches whose atoms are consumed by higher-priority groups.

    This prevents double-counting. For example, the Cl in an acid chloride
    (-C(=O)Cl) matches both the acid_chloride SMARTS and the chloro SMARTS.
    Without filtering, the Cl would appear as both "oyl chloride" (suffix) and
    "chloro" (prefix), producing incorrect names like "1-chloroethanoyl chloride"
    instead of "acetyl chloride".

    Rules:
        1. Acid halides consume halogens: remove chloro/bromo/fluoro matches
           where the halogen atom is part of an acid halide group.
        2. Anhydrides consume esters: remove ester matches where atoms overlap
           with an anhydride group.

    Args:
        functional_groups: Dict from detect_functional_groups().

    Returns:
        Filtered copy of functional_groups with consumed matches removed.
    """
    from .rules.acid_halides import get_acid_halide_consumed_atoms
    from .rules.anhydrides import get_anhydride_consumed_atoms

    fg = dict(functional_groups)  # shallow copy

    # --- Rule 1: Acid halides consume halogens ---
    halide_consumed = get_acid_halide_consumed_atoms(fg)
    if halide_consumed:
        for halogen_key in ("chloro", "bromo", "fluoro"):
            if halogen_key in fg:
                filtered = []
                for match in fg[halogen_key]:
                    # match = (halogen_idx, C_idx) from SMARTS [X][#6]
                    halogen_atom = match[0]
                    if halogen_atom not in halide_consumed:
                        filtered.append(match)
                if filtered:
                    fg[halogen_key] = filtered
                else:
                    del fg[halogen_key]

    # --- Rule 2: Anhydrides consume esters ---
    anhydride_consumed = get_anhydride_consumed_atoms(fg)
    if anhydride_consumed:
        if "ester" in fg:
            filtered = []
            for match in fg["ester"]:
                match_set = set(match)
                # If ANY atom in the ester match overlaps with anhydride, remove it
                if not match_set & anhydride_consumed:
                    filtered.append(match)
            if filtered:
                fg["ester"] = filtered
            else:
                del fg["ester"]

    return fg



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
    try:
        result = namer.name(smiles)
        if result:
            # Check if result contains 'unknown' as a component (partial failure)
            if 'unknown' in result.lower():
                return _descriptive_fallback(smiles)
            return result
        # If name() returned empty/None, generate descriptive fallback
        return _descriptive_fallback(smiles)
    except ValueError:
        raise  # Re-raise ValueError (invalid SMILES) for caller to handle
    except (TypeError, KeyError, IndexError, AttributeError) as e:
        # Graceful fallback for unexpected errors in the naming pipeline.
        # Log the error type for debugging but return a fallback name rather
        # than crashing or returning None.
        logger.debug(
            "Naming error for %s: %s: %s", smiles, type(e).__name__, e
        )
        return _descriptive_fallback(smiles)


# Metals and inorganic elements (not C, H, N, O, S, P, Se, halogens)
_ORGANIC_ELEMENTS = {
    'C', 'H', 'N', 'O', 'S', 'P', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'Si',
}

# Metal element -> name mapping for descriptive messages
_METAL_NAMES = {
    'Li': 'lithium', 'Na': 'sodium', 'K': 'potassium', 'Rb': 'rubidium',
    'Cs': 'cesium', 'Be': 'beryllium', 'Mg': 'magnesium', 'Ca': 'calcium',
    'Sr': 'strontium', 'Ba': 'barium', 'Al': 'aluminium', 'Ga': 'gallium',
    'In': 'indium', 'Tl': 'thallium', 'Sn': 'tin', 'Pb': 'lead',
    'Bi': 'bismuth', 'Ti': 'titanium', 'V': 'vanadium', 'Cr': 'chromium',
    'Mn': 'manganese', 'Fe': 'iron', 'Co': 'cobalt', 'Ni': 'nickel',
    'Cu': 'copper', 'Zn': 'zinc', 'Zr': 'zirconium', 'Mo': 'molybdenum',
    'Ru': 'ruthenium', 'Rh': 'rhodium', 'Pd': 'palladium', 'Ag': 'silver',
    'Cd': 'cadmium', 'W': 'tungsten', 'Re': 'rhenium', 'Os': 'osmium',
    'Ir': 'iridium', 'Pt': 'platinum', 'Au': 'gold', 'Hg': 'mercury',
    'Sb': 'antimony', 'Te': 'tellurium', 'Yb': 'ytterbium', 'La': 'lanthanum',
    'Ce': 'cerium', 'Nd': 'neodymium', 'Sm': 'samarium', 'Eu': 'europium',
    'Gd': 'gadolinium', 'Tb': 'terbium', 'Dy': 'dysprosium', 'Ho': 'holmium',
    'Er': 'erbium', 'Tm': 'thulium', 'Lu': 'lutetium', 'Sc': 'scandium',
    'Y': 'yttrium',
}


def _descriptive_fallback(smiles: str) -> str:
    """Generate a descriptive fallback message instead of bare 'unknown'.

    For inorganic/metallic compounds, returns a descriptive message like
    'gold compound (not supported)'. For organic molecules that failed
    naming, returns 'unknown organic compound'.

    Args:
        smiles: The SMILES string that could not be named.

    Returns:
        A descriptive string (never bare 'unknown').
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return "unknown"

        # Check for wildcard atoms (*)
        has_wildcard = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        if has_wildcard:
            return "compound with wildcard atoms (not supported)"

        # Check for non-organic elements
        elements = set(a.GetSymbol() for a in mol.GetAtoms())
        non_organic = elements - _ORGANIC_ELEMENTS

        if non_organic:
            # Find the most prominent metal/inorganic element
            metal_name = None
            for elem in non_organic:
                if elem in _METAL_NAMES:
                    metal_name = _METAL_NAMES[elem]
                    break

            if metal_name:
                return f"{metal_name} compound (not supported)"
            else:
                # Non-organic but unknown element
                return "inorganic compound (not supported)"

        # Organic compound that failed naming
        return "unknown organic compound"
    except Exception:
        return "unknown"
