"""Multiplicative nomenclature for symmetric bridge-linked molecules (IUPAC P-51.3).

Detects molecules with two or more identical parent structures connected by a
polyvalent linking group (bridge) and produces multiplicative names like
"4,4'-methylenedianiline" or "4,4'-oxydibenzoic acid".

Three conditions for multiplicative naming (IUPAC P-51.3):
    1. Two or more identical parent structures (verified by canonical SMILES)
    2. Connected by a polyvalent linking group (the bridge atom(s))
    3. Parent has a principal characteristic group (suffix-forming FG)

Uses the recursion depth guard from fragment_naming.py to prevent infinite
loops when naming fragment halves.
"""

from typing import Optional, List, Tuple, Dict

from rdkit import Chem
from rdkit.Chem import RWMol

from ..assembly.naming_utils import SIMPLE_MULTIPLIERS


# ---------------------------------------------------------------------------
# Saturation/modification prefixes that must not be preceded by "di"
# ---------------------------------------------------------------------------
# When a parent name starts with one of these prefixes, multiplicative naming
# would produce unparseable concatenation (e.g., "ditetrahydropyran").
# In such cases, fall through to substitutive naming instead.

SATURATION_PREFIXES = [
    'tetrahydro', 'dihydro', 'hexahydro', 'perhydro', 'octahydro',
    'decahydro', 'dodecahydro',
]


# ---------------------------------------------------------------------------
# Bridge definitions: atom pattern -> IUPAC bridge name
# ---------------------------------------------------------------------------

# Single-atom bridge names
_SINGLE_ATOM_BRIDGES: Dict[str, str] = {
    "O": "oxy",
    "S": "thio",
    "NH": "imino",
    "CH2": "methylene",
}

# Two-atom bridge patterns
_TWO_ATOM_BRIDGES = [
    # (element1, element2, h_count1, h_count2, bridge_name)
    ("C", "C", 2, 2, "ethylene"),     # CH2-CH2
    ("C", "C", 1, 1, "vinylene"),     # CH=CH (rare)
    # Phase 154.B D-09 audit-driven adds (154-AUDIT-B.md §3 ranks 1-2):
    ("O", "O", 0, 0, "peroxy"),       # O-O; IUPAC P-29; OPSIN multiRadicalSubstituents.xml line 53; 154-AUDIT-B.md §3 #1
    ("S", "S", 0, 0, "disulfanediyl"),# S-S; IUPAC P-29; OPSIN multiRadicalSubstituents.xml; 154-AUDIT-B.md §3 #2
]

# Multi-atom bridge names: (element, H_count, ring_neighbor_count) -> bridge_name
# These handle star-topology bridges where 3+ identical parents radiate from
# a single central atom (IUPAC P-51.3.3).
_MULTI_BRIDGE_NAMES: Dict[Tuple[str, int, int], str] = {
    ("N", 0, 3): "nitrilo",           # N connecting 3 rings (trivalent)
    ("C", 1, 3): "methylidyne",       # CH connecting 3 rings (trivalent)
    ("C", 0, 4): "methanetetrayl",    # C connecting 4 rings (tetravalent)
    # Phase 154.B D-09 audit-driven add (154-AUDIT-B.md §3 rank 3):
    ("P", 0, 3): "phosphinidyne",     # P connecting 3 rings (trivalent); IUPAC P-68; OPSIN multiRadicalSubstituents.xml line 47; 154-AUDIT-B.md §3 #3
}


# Phase 154.B D-10: _RETAINED_PARENT_NAMES (4-entry hardcoded dict) DELETED.
# The registry-query layer below (_resolve_parent_name) replaces it.  The
# global ALL_RETAINED_NAMES registry in data/retained_names.py is the
# single source of truth (Phase 151 D-21 reuse pattern).  See
# 154-AUDIT-B.md §5 for the registry-coverage verification that confirms
# all 4 previously-hardcoded SMILES are present in the registry when
# accessed via Chem.CanonSmiles(...).


def _resolve_parent_name(canon_smiles: str) -> Optional[Tuple[str, int]]:
    """Phase 154.B D-10: registry-query layer for retained parent + locant.

    Replaces the hardcoded _RETAINED_PARENT_NAMES dict (deleted) with a
    query against the global ALL_RETAINED_NAMES registry via
    get_retained_name. Falls back to name_fragment_recursively wrapped
    with _extract_pg_locant_from_fragment for the principal-group locant.

    No SMILES strings hardcoded inline -- single source of truth is the
    global registry per Phase 151 D-21 reuse pattern.

    Args:
        canon_smiles: Canonical SMILES of the parent fragment.

    Returns:
        (parent_name, fg_locant) where fg_locant is the IUPAC position of the
        principal group on the fragment, or None if no parent name resolvable.

    Source: 154-CONTEXT.md D-10; data/retained_names.py:get_retained_name:465.
    """
    from ..data.retained_names import get_retained_name
    from ..assembly.fragment_naming import name_fragment_recursively

    # 1. Global registry first (single source of truth).
    retained = get_retained_name(canon_smiles)
    if retained is not None:
        locant = _extract_pg_locant_from_fragment(canon_smiles)
        return (retained, locant if locant is not None else 1)

    # 2. Algorithmic fallback.
    name = name_fragment_recursively(canon_smiles)
    if name is None:
        return None
    locant = _extract_pg_locant_from_fragment(canon_smiles)
    return (name, locant if locant is not None else 1)


def _extract_pg_locant_from_fragment(canon_smiles: str) -> Optional[int]:
    """Find the principal-group atom in the fragment and return its IUPAC locant.

    For benzene-derived parents (aniline, phenol, benzoic acid), the principal
    group is at locant 1.  For other ring parents, the v18 scope returns 1 as
    the safe default (the caller's `_get_bridge_locant` D-12 cascade computes
    the bridge attachment locant relative to that PG-1 anchor).

    Args:
        canon_smiles: Canonical SMILES of the parent fragment.

    Returns:
        1-indexed IUPAC locant of the principal group, or None if extraction
        fails.

    Source: 154-CONTEXT.md D-10; 154-AUDIT-B.md §5.
    """
    mol = Chem.MolFromSmiles(canon_smiles)
    if mol is None:
        return None
    # For all 4 hardcoded benzene-derived parents (aniline / phenol / benzoic
    # acid x2), the principal group sits at locant 1 by IUPAC convention.
    # This is the safe default for the v18 multiplicative scope (which only
    # consumes benzene-derived retained parents per 154-AUDIT-B.md §5).
    return 1


def _is_pure_single_bond_assembly(mol) -> bool:
    """Phase 154.B D-11: detect single-bond-joined identical rings (ring_assemblies territory).

    True iff every inter-ring-system connection in the molecule is a single
    bond directly between two ring atoms with NO bridge atom.  In that
    case the multiplicative path returns None and the cascade falls
    through to detect_ring_assembly (Phase 151's path).

    The contract is symmetric: this guard is the multiplicative-side
    enforcer; ring_assemblies._find_inter_system_bonds (lines 77-124) is
    the ring-assembly-side enforcer (only counts ring-to-ring single
    bonds; bridge atoms are not in rings, so atom-bridged cases never
    produce inter-system bonds there).

    Cross-handler regression test:
    tests/integration/test_assembly_vs_multiplicative_dispatch.py.

    Source: 154-CONTEXT.md D-11; ring_assemblies.py:_find_inter_system_bonds:77.
    Source: 154-RESEARCH.md §4.5.
    """
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 2:
        return False

    # Build ring_atoms set.
    atom_rings = ring_info.AtomRings()
    ring_atoms = set()
    for r in atom_rings:
        ring_atoms.update(r)

    # Use perception.rings.get_ring_systems to cluster fused rings.
    from ..perception.rings import get_ring_systems
    ring_systems = get_ring_systems(mol)
    # ring_systems is a list of sets of atom indices, one per fused ring system.
    if len(ring_systems) < 2:
        # Only one ring system (e.g., naphthalene) -- multiplicative does not apply.
        return False

    atom_to_system: Dict[int, int] = {}
    for sys_idx, atoms in enumerate(ring_systems):
        for a in atoms:
            atom_to_system[a] = sys_idx

    # Look for inter-ring-system bonds (single bond, both atoms in distinct systems).
    has_single_bond_inter_system = False
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in ring_atoms and a2 in ring_atoms:
            sys1 = atom_to_system.get(a1)
            sys2 = atom_to_system.get(a2)
            if sys1 is not None and sys2 is not None and sys1 != sys2:
                if bond.GetBondTypeAsDouble() == 1.0:
                    has_single_bond_inter_system = True

    # Detect bridge atoms (non-ring atoms touching >= 2 distinct ring systems).
    has_bridge_atom = False
    for atom in mol.GetAtoms():
        if atom.GetIdx() in ring_atoms:
            continue
        touched_systems = set()
        for nbr in atom.GetNeighbors():
            nbr_sys = atom_to_system.get(nbr.GetIdx())
            if nbr_sys is not None:
                touched_systems.add(nbr_sys)
        if len(touched_systems) >= 2:
            has_bridge_atom = True
            break

    # Pure single-bond assembly = inter-system single bond present AND no bridge atom.
    return has_single_bond_inter_system and not has_bridge_atom


def name_multiplicative(mol) -> Optional[str]:
    """Detect and name multiplicative nomenclature cases.

    Args:
        mol: RDKit Mol object.

    Returns:
        Multiplicative IUPAC name if applicable, None otherwise.
    """
    if mol is None:
        return None

    # Quick reject: need at least 2 ring systems
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 2:
        return None

    # Phase 154.B D-11: topology guard for ring-assembly mutual exclusion.
    # If every inter-fragment connection is a single bond between two ring
    # atoms with NO bridge atom, return None and let the cascade fall
    # through to detect_ring_assembly (Phase 151's path).  Cross-handler
    # regression test: tests/integration/test_assembly_vs_multiplicative_dispatch.py.
    #
    # Source: 154-CONTEXT.md D-11; ring_assemblies.py:_find_inter_system_bonds:77.
    if _is_pure_single_bond_assembly(mol):
        return None  # ring_assemblies.py owns this case

    # Collect ring atom indices
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)

    # --- Try single-atom bridges first ---
    result = _try_single_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Try two-atom bridges ---
    result = _try_two_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Try multi-atom bridges (3+ units, star topology) ---
    result = _try_multi_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    return None


def _try_single_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find single-atom bridges between identical ring systems."""
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue  # Bridge atoms are NOT in rings

        neighbors = atom.GetNeighbors()
        heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() > 1]

        # Single-atom bridge: exactly 2 heavy neighbors, both in rings
        if len(heavy_neighbors) != 2:
            continue
        if not all(n.GetIdx() in ring_atoms for n in heavy_neighbors):
            continue

        # Classify the bridge atom
        bridge_type = _classify_single_atom_bridge(atom)
        if bridge_type is None:
            continue

        bridge_name = _SINGLE_ATOM_BRIDGES.get(bridge_type)
        if bridge_name is None:
            continue

        # Split the molecule at the bridge
        nbr_indices = [n.GetIdx() for n in heavy_neighbors]
        fragments = _split_at_bridge(mol, idx, nbr_indices)
        if fragments is None:
            continue

        frag_smiles_a, frag_smiles_b, conn_atom_a, conn_atom_b = fragments

        # Check if fragments are identical
        canon_a = Chem.CanonSmiles(frag_smiles_a)
        canon_b = Chem.CanonSmiles(frag_smiles_b)
        if canon_a != canon_b:
            continue

        # Name the parent structure
        parent_name = _name_parent(canon_a)
        if parent_name is None:
            continue

        # Get locant of bridge attachment point in the parent
        locant = _get_bridge_locant(mol, idx, nbr_indices[0], ring_atoms)

        # Assemble multiplicative name (returns None for prefix-derived parents)
        result = _assemble_multiplicative_name(locant, bridge_name, parent_name)
        if result is not None:
            return result
        # Saturation-prefix parent: skip multiplicative, let caller fall through
        continue

    return None


def _try_two_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find two-atom bridges (e.g., CH2-CH2 ethylene) between identical ring systems."""
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        idx1 = a1.GetIdx()
        idx2 = a2.GetIdx()

        # Both bridge atoms must NOT be in rings
        if idx1 in ring_atoms or idx2 in ring_atoms:
            continue

        # Each bridge atom must have exactly one ring neighbor (besides its bridge partner)
        ring_nbrs_1 = [n for n in a1.GetNeighbors()
                       if n.GetIdx() in ring_atoms and n.GetIdx() != idx2]
        ring_nbrs_2 = [n for n in a2.GetNeighbors()
                       if n.GetIdx() in ring_atoms and n.GetIdx() != idx1]

        # Each atom in the bridge must connect to exactly one ring atom
        if len(ring_nbrs_1) != 1 or len(ring_nbrs_2) != 1:
            continue

        # Each bridge atom must have exactly 2 heavy neighbors total
        # (one ring atom + one bridge partner)
        heavy_nbrs_1 = [n for n in a1.GetNeighbors() if n.GetAtomicNum() > 1]
        heavy_nbrs_2 = [n for n in a2.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs_1) != 2 or len(heavy_nbrs_2) != 2:
            continue

        # Classify the two-atom bridge
        bridge_name = _classify_two_atom_bridge(a1, a2)
        if bridge_name is None:
            continue

        # Split at both bridge atoms
        ring_conn_1 = ring_nbrs_1[0].GetIdx()
        ring_conn_2 = ring_nbrs_2[0].GetIdx()

        fragments = _split_at_two_atom_bridge(
            mol, idx1, idx2, ring_conn_1, ring_conn_2
        )
        if fragments is None:
            continue

        frag_smiles_a, frag_smiles_b = fragments

        # Check if fragments are identical
        canon_a = Chem.CanonSmiles(frag_smiles_a)
        canon_b = Chem.CanonSmiles(frag_smiles_b)
        if canon_a != canon_b:
            continue

        # Name the parent structure
        parent_name = _name_parent(canon_a)
        if parent_name is None:
            continue

        # Get locant
        locant = _get_bridge_locant(mol, idx1, ring_conn_1, ring_atoms)

        # Assemble multiplicative name (returns None for prefix-derived parents)
        result = _assemble_multiplicative_name(locant, bridge_name, parent_name)
        if result is not None:
            return result
        # Saturation-prefix parent: skip multiplicative, let caller fall through
        continue

    return None


def _classify_single_atom_bridge(atom) -> Optional[str]:
    """Classify a single bridge atom into a known type.

    Returns:
        Bridge type key (e.g., 'CH2', 'O', 'NH', 'S') or None.
    """
    sym = atom.GetSymbol()
    total_h = atom.GetTotalNumHs()

    if sym == "C" and total_h == 2:
        return "CH2"
    elif sym == "O" and total_h == 0:
        return "O"
    elif sym == "N" and total_h == 1:
        return "NH"
    elif sym == "S" and total_h == 0:
        return "S"
    return None


def _classify_two_atom_bridge(atom1, atom2) -> Optional[str]:
    """Classify a two-atom bridge into a known type."""
    sym1 = atom1.GetSymbol()
    sym2 = atom2.GetSymbol()
    h1 = atom1.GetTotalNumHs()
    h2 = atom2.GetTotalNumHs()

    for s1, s2, expected_h1, expected_h2, name in _TWO_ATOM_BRIDGES:
        if sym1 == s1 and sym2 == s2 and h1 == expected_h1 and h2 == expected_h2:
            return name
        # Check reverse order
        if sym1 == s2 and sym2 == s1 and h1 == expected_h2 and h2 == expected_h1:
            return name
    return None


def _classify_multi_bridge(atom, ring_nbr_count: int) -> Optional[str]:
    """Classify a multi-valent bridge atom by element + H count + ring neighbor count.

    Args:
        atom: RDKit atom (the bridge candidate).
        ring_nbr_count: Number of ring-atom neighbors.

    Returns:
        Bridge name (e.g., 'nitrilo', 'methylidyne', 'methanetetrayl') or None.
    """
    sym = atom.GetSymbol()
    h = atom.GetTotalNumHs()
    return _MULTI_BRIDGE_NAMES.get((sym, h, ring_nbr_count))


def _all_fragments_are_simple_carbocycles(mol, bridge_idx: int) -> bool:
    """Check whether removing the bridge atom yields only simple carbocyclic
    fragments (rings with no principal characteristic group / no heteroatoms
    in the parent ring).

    Phase 157 cleanup helper: the substitutive-PIN guard in
    `_try_multi_atom_bridges` invokes this function to distinguish:
      - Plain benzene fragments (e.g., (Ph)3P -> triphenylphosphane PIN)
      - Phenol-like fragments with -OH (e.g., (HOPh)3P -> 4,4',4''-
        phosphinidynetriphenol multiplicative PIN per P-14.5)

    A fragment is "simple carbocyclic" if every atom is a ring carbon
    OR an explicit hydrogen — no heteroatoms (O/N/S/etc.) AND no
    extra-ring substituent atoms. Plain benzene `c1ccccc1` qualifies;
    phenol `Oc1ccccc1` does NOT (has the hydroxyl O).

    Args:
        mol: the original RDKit Mol.
        bridge_idx: atom index of the bridge to remove.

    Returns:
        True if every fragment after bridge removal is a simple
        carbocycle with no principal characteristic group; False
        otherwise (in which case multiplicative may be preferred).
    """
    emol = RWMol(Chem.RWMol(mol))
    emol.RemoveAtom(bridge_idx)
    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return False
    frag_mols = Chem.GetMolFrags(emol.GetMol(), asMols=True, sanitizeFrags=True)
    for frag in frag_mols:
        for atom in frag.GetAtoms():
            # Heteroatom in or out of ring -> NOT simple carbocyclic
            if atom.GetAtomicNum() != 6 and atom.GetAtomicNum() != 1:
                return False
            # Extra-ring carbon (e.g., methyl substituent) -> still
            # carbocyclic but has a substituent; out of "simple" scope.
            if not atom.IsInRing() and atom.GetAtomicNum() == 6:
                return False
    return True


def _try_multi_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find star-topology bridges connecting 3+ identical ring systems.

    A multi-atom bridge is a single non-ring atom connected to 3 or more ring
    atoms, where removing the bridge produces 3+ identical fragments.

    Examples:
        N connecting 3 phenol rings -> nitrilotriphenol
        CH connecting 3 phenol rings -> methylidynetriphenol
        C connecting 4 phenol rings -> methanetetrayltetraphenol

    Phase 157 cleanup substitutive-PIN guard:
        For mononuclear parent hydrides (NH3, PH3, AsH3, SiH4, GeH4, SnH4,
        PbH4, BH3, plus the multivalent CH4 case) substituted with 3+
        IDENTICAL SIMPLE-RING groups, IUPAC P-66.6.1.1.3 / P-67.1.1.1 /
        P-68 mandate the SUBSTITUTIVE form (e.g., `triphenylphosphane`,
        `triphenylamine`, `triphenylmethane`) as PIN — not the
        multiplicative form (`1,1',1''-phosphinidynetribenzene` etc.).
        See `cleanup-deferred-items.md` D-157-11 for the full bug
        provenance and IUPAC rule citations.
    """
    # Mononuclear-parent-hydride elements where the SUBSTITUTIVE form is
    # PIN over the multiplicative form ONLY when the substituent rings
    # are SIMPLE (no principal characteristic group). Per IUPAC P-14.5,
    # multiplicative is preferred when the parent ring has a principal
    # characteristic group (e.g., 4,4',4''-nitrilotriphenol uses
    # `triphenol` parent with a `nitrilo` bridge). When the parent ring
    # has NO principal characteristic group (e.g., plain benzene), the
    # substitutive form on the mononuclear parent hydride is PIN per
    # P-66.6.1.1.3 / P-67.1.1.1 / P-66.1.1.1: `triphenylamine`,
    # `triphenylphosphane`, `triphenylmethane`.
    _SUBSTITUTIVE_PIN_CENTERS = frozenset({
        'B', 'C', 'N', 'P', 'As', 'Sb', 'Bi',
        'Si', 'Ge', 'Sn', 'Pb', 'S', 'Se', 'Te',
    })

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue  # Bridge atoms are NOT in rings

        neighbors = atom.GetNeighbors()
        heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() > 1]

        # Count ring neighbors
        ring_neighbors = [n for n in heavy_neighbors if n.GetIdx() in ring_atoms]
        ring_nbr_count = len(ring_neighbors)

        # Must have 3+ ring neighbors for multi-bridge
        if ring_nbr_count < 3:
            continue

        # All heavy neighbors must be ring atoms (no non-ring non-H substituents)
        if len(heavy_neighbors) != ring_nbr_count:
            continue

        # Phase 157 cleanup substitutive-PIN guard: when the bridge atom
        # is a mononuclear-parent-hydride element AND the substituent
        # rings are simple carbocycles with no principal characteristic
        # group, defer to substitutive nomenclature. This honors IUPAC
        # P-66.6.1.1.3 / P-67.1.1.1 / P-66.1.1.1 PIN preference for
        # triphenylamine / triphenylphosphane / triphenylmethane while
        # leaving the multiplicative path active for cases where the
        # parent ring has a principal characteristic group (e.g.,
        # 4,4',4''-nitrilotriphenol per P-14.5).
        if atom.GetSymbol() in _SUBSTITUTIVE_PIN_CENTERS:
            if _all_fragments_are_simple_carbocycles(mol, idx):
                continue

        # Classify the bridge
        bridge_name = _classify_multi_bridge(atom, ring_nbr_count)
        if bridge_name is None:
            continue

        # Split molecule by removing the bridge atom
        emol = RWMol(Chem.RWMol(mol))
        emol.RemoveAtom(idx)

        try:
            Chem.SanitizeMol(emol)
        except Exception:
            continue

        result_mol = emol.GetMol()
        frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)
        unit_count = len(frag_mols)

        if unit_count < 3:
            continue

        # Check all fragments are identical
        canon_smiles_list = [Chem.MolToSmiles(f) for f in frag_mols]
        canon_set = set(Chem.CanonSmiles(s) for s in canon_smiles_list)
        if len(canon_set) != 1:
            continue  # Non-identical fragments

        canon_parent = canon_set.pop()

        # Name the parent structure
        parent_name = _name_parent(canon_parent)
        if parent_name is None:
            continue

        # Get bridge locant using the first ring neighbor
        first_ring_nbr_idx = ring_neighbors[0].GetIdx()
        locant = _get_bridge_locant(mol, idx, first_ring_nbr_idx, ring_atoms)

        # Assemble multiplicative name with unit_count
        result = _assemble_multiplicative_name(
            locant, bridge_name, parent_name, unit_count=unit_count
        )
        if result is not None:
            return result
        continue

    return None


def _split_at_bridge(
    mol, bridge_idx: int, nbr_indices: List[int]
) -> Optional[Tuple[str, str, int, int]]:
    """Split molecule by removing a single bridge atom.

    Returns:
        (frag_smiles_a, frag_smiles_b, conn_atom_in_a, conn_atom_in_b)
        or None if split doesn't produce exactly 2 fragments.
    """
    emol = RWMol(Chem.RWMol(mol))

    # Record neighbors before removal (atom indices will shift!)
    nbr_a, nbr_b = nbr_indices[0], nbr_indices[1]

    # Remove bridge atom
    emol.RemoveAtom(bridge_idx)

    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None

    result_mol = emol.GetMol()
    frag_indices = Chem.GetMolFrags(result_mol)

    if len(frag_indices) != 2:
        return None

    frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)
    if len(frag_mols) != 2:
        return None

    smi_a = Chem.MolToSmiles(frag_mols[0])
    smi_b = Chem.MolToSmiles(frag_mols[1])

    # Adjust neighbor indices for atom removal
    # When atom at bridge_idx is removed, all atoms with idx > bridge_idx shift down by 1
    adj_a = nbr_a if nbr_a < bridge_idx else nbr_a - 1
    adj_b = nbr_b if nbr_b < bridge_idx else nbr_b - 1

    return (smi_a, smi_b, adj_a, adj_b)


def _split_at_two_atom_bridge(
    mol, bridge_idx1: int, bridge_idx2: int,
    ring_conn1: int, ring_conn2: int
) -> Optional[Tuple[str, str]]:
    """Split molecule by removing two bridge atoms.

    Returns:
        (frag_smiles_a, frag_smiles_b) or None.
    """
    emol = RWMol(Chem.RWMol(mol))

    # Remove in reverse index order to avoid index shifting issues
    idx_high = max(bridge_idx1, bridge_idx2)
    idx_low = min(bridge_idx1, bridge_idx2)

    emol.RemoveAtom(idx_high)
    emol.RemoveAtom(idx_low)

    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None

    result_mol = emol.GetMol()
    frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)

    if len(frag_mols) != 2:
        return None

    smi_a = Chem.MolToSmiles(frag_mols[0])
    smi_b = Chem.MolToSmiles(frag_mols[1])

    return (smi_a, smi_b)


def _name_parent(canon_smiles: str) -> Optional[str]:
    """Backwards-compat wrapper around _resolve_parent_name (D-10).

    Returns just the name string (drops the principal-group locant) so
    existing callers in this module see no behavior change.  New callers
    should prefer _resolve_parent_name directly to get both the parent
    name and the principal-group locant.

    Source: 154-CONTEXT.md D-10 (replaces hardcoded _RETAINED_PARENT_NAMES dict).
    """
    result = _resolve_parent_name(canon_smiles)
    return result[0] if result else None


def _get_bridge_locant(
    mol, bridge_idx: int, ring_conn_idx: int, ring_atoms: set
) -> int:
    """Phase 154.B D-12: query Phase 151 cascade for IUPAC ring locants.

    Replaces the legacy "ring shortest-path from principal group"
    heuristic (which defaulted to 4 / para on failure -- the
    architectural debt this edit clears) with a query against the
    existing handler's IUPAC locant map (Phase 151 cascade --
    heterocycle / fused / VB / spiro / ring-assembly locants).

    Falls back to the legacy shortest-path heuristic
    (_shortest_path_heuristic_locant) only when the cascade returns None
    for the target ring -- preserves current behavior on cascade misses.
    The legacy hard-default-4 branches at lines 501 and 511 are GONE per
    154-AUDIT-B.md §7.

    Args:
        mol: RDKit Mol object.
        bridge_idx: index of the bridge atom.
        ring_conn_idx: index of the ring atom connected to the bridge.
        ring_atoms: set of all ring atom indices.

    Returns:
        Locant number (1-indexed).

    Source: 154-CONTEXT.md D-12; rules/parent_selection.py:_build_ring_pos:77;
            namer.py:_build_ring_info_for_parent_selection:372;
            rules/locants.py:compare_locant_sets:96.
    """
    from .parent_selection import _build_ring_pos

    # 1. Find the target ring set containing ring_conn_idx
    ring_info = mol.GetRingInfo()
    target_ring = None
    for ring in ring_info.AtomRings():
        if ring_conn_idx in ring:
            target_ring = set(ring)
            break
    if target_ring is None:
        # No ring contains the connection atom -- fall back to legacy
        # heuristic.  Defensive guard: should never hit because callers
        # assert ring_conn_idx is in ring_atoms.
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    # 2. Build ring_info dict via Phase 151 cascade (fused-hetero, PAH,
    #    benzene, heterocycle, spiro, mixed-spiro/fused, VB, ring-assembly).
    handler_ring_info = None
    try:
        from ..namer import _build_ring_info_for_parent_selection, compute_features

        features = compute_features(mol)
        handler_ring_info = _build_ring_info_for_parent_selection(features)
    except Exception:
        # Cascade unavailable for this molecule shape -- fall back gracefully.
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    # 3. Phase 151 cascade locants (authoritative IUPAC numbering when available)
    #
    # IMPORTANT: only trust the cascade when handler_ring_info contains an
    # `iupac_locants` dict covering EVERY atom of target_ring.  When the
    # cascade has no IUPAC numbering for this ring shape (e.g. functional-
    # group-anchored cyclohexane-1,3-dione where the locants come from the
    # PG positions, not from the ring-system handler), `_build_ring_pos`
    # falls back to atom-sorted positional integers -- which are NOT the
    # IUPAC locants the caller needs.  Detect this case via the
    # `iupac_locants` presence + complete-coverage check and fall through
    # to the heuristic, which IS PG-anchored.  This preserves the v17
    # heuristic correctness on functional-group-anchored rings while
    # adopting the cascade's authoritative numbering on Hantzsch-Widman /
    # fused / PAH / spiro / VB / ring-assembly handler-controlled rings.
    iupac_locants = (
        handler_ring_info.get("iupac_locants") if handler_ring_info else None
    )
    cascade_covers_ring = (
        iupac_locants
        and all(a in iupac_locants for a in target_ring)
    )
    if not cascade_covers_ring:
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    try:
        ring_pos = _build_ring_pos(target_ring, ring_info=handler_ring_info)
    except Exception:
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )
    locant = ring_pos.get(ring_conn_idx) if ring_pos else None

    if locant is None:
        # Cascade returned no locant for this atom -- fall back to heuristic
        # (preserves current behavior on cascade misses; documented as
        # "remaining heuristic share" in 154-VERIFICATION.md per Q-B5).
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    # Tuple-coercion: tuple locants like (4, 'a') reduce to int (Phase 147 D-01)
    if isinstance(locant, tuple):
        return locant[0]
    return locant


def _shortest_path_heuristic_locant(
    mol, bridge_idx: int, ring_conn_idx: int, ring_atoms: set
) -> int:
    """Legacy ring-shortest-path-from-PG heuristic (D-12 fallback path).

    Preserved as the fallback when the Phase 151 cascade returns None for
    the target ring.  The legacy "default to 4" branches at the previous
    lines 501 and 511 are GONE -- this function returns the heuristic
    distance + 1 even on edge cases; if the heuristic itself fails (no
    PG atom found), it returns 1 (top-of-ring) instead of silently
    emitting 4 / para.

    Source: 154-CONTEXT.md D-12 (fallback path); legacy
            multiplicative.py:475-526 pre-Plan-02.
    """
    ring_info = mol.GetRingInfo()
    target_ring = None
    for ring in ring_info.AtomRings():
        if ring_conn_idx in ring:
            target_ring = ring
            break
    if target_ring is None:
        return 1

    pg_atom_idx = _find_principal_group_atom_in_ring(
        mol, target_ring, ring_atoms, bridge_idx
    )
    if pg_atom_idx is None:
        # NOT 4 anymore -- legacy default removed per D-12.  Returning 1
        # makes cascade misses observably wrong rather than silently
        # right-ish (since 1 is rarely the correct bridge locant).
        return 1

    ring_list = list(target_ring)
    try:
        pg_pos = ring_list.index(pg_atom_idx)
        conn_pos = ring_list.index(ring_conn_idx)
    except ValueError:
        return 1

    ring_size = len(ring_list)
    dist = abs(conn_pos - pg_pos)
    dist = min(dist, ring_size - dist)
    return dist + 1  # 1-indexed (PG is at position 1)


def _find_principal_group_atom_in_ring(
    mol, ring: tuple, ring_atoms: set, bridge_idx: int
) -> Optional[int]:
    """Find the ring atom bearing the principal characteristic group.

    The principal group atom is a ring atom that has a non-ring, non-bridge
    neighbor that is part of a functional group (N, O attached to C=O, etc.).

    Returns:
        Atom index of the ring atom bearing the principal group, or None.
    """
    ring_set = set(ring)

    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            # Skip ring atoms and the bridge atom
            if nbr_idx in ring_set or nbr_idx == bridge_idx:
                continue
            if nbr_idx in ring_atoms:
                continue  # Part of another ring

            # This is a non-ring, non-bridge substituent
            # Check if it's a functional group atom
            sym = nbr.GetSymbol()
            if sym in ("N", "O", "S"):
                return idx
            # Check for C=O type groups (carboxylic acid, aldehyde, etc.)
            if sym == "C":
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() != idx:
                        bond = mol.GetBondBetweenAtoms(nbr_idx, nbr2.GetIdx())
                        if (bond and bond.GetBondTypeAsDouble() == 2.0
                                and nbr2.GetSymbol() == "O"):
                            return idx

    return None


def _build_primed_locant_str(locant: int, unit_count: int) -> str:
    """Build a primed locant string for multiplicative naming.

    For each unit i in [0, unit_count), appends i primes to the locant.
    Example: locant=4, unit_count=3 -> "4,4',4''"
    Example: locant=4, unit_count=4 -> "4,4',4'',4'''"

    Args:
        locant: The IUPAC locant number.
        unit_count: Number of identical parent units.

    Returns:
        Comma-separated primed locant string.
    """
    prime = "'"
    parts = []
    for i in range(unit_count):
        parts.append(f"{locant}{prime * i}")
    return ",".join(parts)


def _select_multiplier(parent_name: str, unit_count: int) -> Optional[str]:
    r"""Phase 154.B D-08: select between SIMPLE_MULTIPLIERS (P-14.2.1) and COMPLEX_MULTIPLIERS (P-14.2.2).

    Per IUPAC Blue Book:
      - P-14.2.1 (simple): di / tri / tetra default for clean parent names.
      - P-14.2.2 (group):  bis / tris / tetrakis fire when the parent name
        contains a comma-separated locant pattern that would make di+name
        parse ambiguously (e.g., "1,3-thiazole" -> "bis(1,3-thiazole)"
        instead of the unparseable "di-1,3-thiazole").
      - P-14.2.3 (ring-assembly): bi / ter / quater for identical rings
        joined by single bonds -- Phase 151's territory
        (rules.ring_assemblies.ASSEMBLY_MULTIPLIERS); NOT touched here.

    Heuristic (R-154-NEW-7 refinement): trigger COMPLEX_MULTIPLIERS only
    when parent_name contains a comma-separated locant pattern (\d+,\d).
    Bare digits without commas (e.g., "but-2-ene") use SIMPLE_MULTIPLIERS;
    those don't create di+name parse ambiguity.

    Args:
        parent_name: name of the parent fragment.
        unit_count: number of identical parent units (2, 3, 4, ...).

    Returns:
        The multiplier prefix string (e.g., "di", "bis"), or None if
        unit_count is unsupported by both tables.

    Source: 154-CONTEXT.md D-08; 154-AUDIT-B.md §4;
            IUPAC Blue Book P-14.2.1, P-14.2.2;
            OPSIN multipliers.xml type="basic" / type="group".
    """
    import re
    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS

    # P-14.2.2 trigger: parent name contains a comma-separated locant
    # pattern.  Match at least one ",N" where N is a digit (e.g. "1,3-",
    # "2,4,6-").  Bare digits without commas do NOT trigger.
    if re.search(r"\d+,\d", parent_name):
        return COMPLEX_MULTIPLIERS.get(unit_count)

    # P-14.2.1 default for clean parent names.
    return SIMPLE_MULTIPLIERS.get(unit_count)


def _assemble_multiplicative_name(
    locant: int, bridge_name: str, parent_name: str, unit_count: int = 2
) -> Optional[str]:
    """Assemble the final multiplicative name.

    Format: [locants]-[bridge][multiplier][parent]
    Examples:
        4,4'-methylenedianiline (2 units; P-14.2.1)
        4,4',4''-nitrilotriphenol (3 units; P-14.2.1)
        4,4',4'',4'''-methanetetrayltetraphenol (4 units; P-14.2.1)
        4,4'-bis(1,3-thiazole)... (P-14.2.2 group multiplier)

    For acid names with spaces (like "benzoic acid"), the multiplier prefix
    goes before the base name: 4,4'-oxydibenzoic acid

    If the parent name starts with a saturation/modification prefix
    (e.g., "tetrahydro", "dihydro"), returns None to signal that
    multiplicative naming would produce an unparseable concatenation
    (e.g., "ditetrahydropyran") and the caller should fall through
    to substitutive naming.

    Args:
        locant: IUPAC locant of the bridge attachment point.
        bridge_name: Name of the bridge group (e.g., "methylene", "oxy").
        parent_name: IUPAC name of one parent unit (e.g., "aniline", "benzoic acid").
        unit_count: Number of identical parent units (default 2 for backward compat).

    Returns:
        Complete multiplicative name, or None if the parent name has a
        saturation prefix that would produce unparseable multiplier+prefix
        output OR the multiplier table has no entry for unit_count.
    """
    # Check for saturation/modification prefixes in the parent name
    # that would produce unparseable "di+prefix" concatenation
    parent_lower = parent_name.lower()
    if any(parent_lower.startswith(p) for p in SATURATION_PREFIXES):
        return None

    # Phase 154.B D-08: P-14.2.1 / P-14.2.2 three-way split (P-14.2.3
    # belongs to ring_assemblies.py and is NOT touched here).
    multiplier = _select_multiplier(parent_name, unit_count)
    if not multiplier:
        return None

    # Build primed locant string: e.g., "4,4'" for 2 units, "4,4',4''" for 3
    locant_str = _build_primed_locant_str(locant, unit_count)

    # Handle names with spaces (e.g., "benzoic acid" -> "tribenzoic acid")
    if " " in parent_name:
        parts = parent_name.split(" ", 1)
        base = parts[0]  # "benzoic"
        suffix = parts[1]  # "acid"
        return f"{locant_str}-{bridge_name}{multiplier}{base} {suffix}"
    else:
        # Simple name: "aniline" -> "dianiline"
        return f"{locant_str}-{bridge_name}{multiplier}{parent_name}"
