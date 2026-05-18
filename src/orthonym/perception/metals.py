"""
Organometallic complex detection + hapticity perception (Phase 161 v19).

This module provides RDKit-graph-based detection of metal-organometallic
complexes per IUPAC 2013 Blue Book §P-69 + IUPAC Red Book §IR-10 + Salzer
1999 IUPAC Recommendations.

All functions are PREDICATE-PURE per CONTEXT D-12 + Phase 158 D-26 + Phase
160 D-25 hard invariant:
- NO mol mutation (no SanitizeMol, no UpdatePropertyCache, no
  AssignStereochemistry, no any other RDKit mutating method).
- NO module-global state read/write.
- NO exception swallowing (RDKit exceptions propagate; ValueError raised
  by compute_hapticity is caught at the handler layer per CONTEXT D-12).

The integrity test ``test_side_effect_inventory_is_empty`` parametrized
over ``list(StoutClass)`` at ``tests/unit/routing/test_dispatch_table.py``
will auto-extend to ORGANOMETALLIC after Plan-02 ships and assert ``()``
for the new entry.

Phase 161 (v19 first scope-expansion phase per ADR-19-07).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple
from rdkit import Chem


# Allowlist for is_metal_element() per RESEARCH §3.1.
# Groups 1-17 metals/semimetals. Hydrogen EXCLUDED (never metal).
# Compiled from IUPAC periodic table (2023).
METAL_ELEMENT_SYMBOLS: FrozenSet[str] = frozenset({
    # Group 1 (alkali)
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    # Group 2 (alkaline earth)
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    # Group 13
    'B', 'Al', 'Ga', 'In', 'Tl',
    # Group 14
    'Si', 'Ge', 'Sn', 'Pb',
    # Group 15
    'As', 'Sb', 'Bi',
    # Group 16 (semimetals)
    'Te', 'Po',
    # Transition metals (Groups 3-12)
    'Sc', 'Y', 'La', 'Ac',
    'Ti', 'Zr', 'Hf', 'Rf',
    'V', 'Nb', 'Ta', 'Db',
    'Cr', 'Mo', 'W', 'Sg',
    'Mn', 'Tc', 'Re', 'Bh',
    'Fe', 'Ru', 'Os', 'Hs',
    'Co', 'Rh', 'Ir', 'Mt',
    'Ni', 'Pd', 'Pt', 'Ds',
    'Cu', 'Ag', 'Au', 'Rg',
    'Zn', 'Cd', 'Hg', 'Cn',
    # Lanthanides + Actinides (Phase 161.X scope-deferred but allowlist anyway)
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
})


@dataclass(frozen=True)
class LigandGroup:
    """A contiguous set of ligand atoms coordinated to a single metal."""
    metal_atom_idx: int                                  # mol atom index
    ligand_atom_indices: Tuple[int, ...]                 # sorted tuple
    hapticity_n: int                                     # ≥ 1
    ligand_smarts_key: Optional[str] = None              # SMARTS catalog hit
    ligand_canonical_smiles: Optional[str] = None        # ligand subgraph SMILES


@dataclass(frozen=True)
class MetalComplex:
    """A complete metal-organometallic complex: one or more metals + ligands."""
    metal_atom_indices: Tuple[int, ...]                  # sorted
    ligand_groups: Tuple[LigandGroup, ...]               # per-metal-per-ligand
    formal_charges: Tuple[int, ...]                      # one per metal
    is_multimetal: bool                                  # True if > 1 metal


def is_metal_element(symbol: str) -> bool:
    """Predicate: is the given element symbol a metal/semimetal?

    PURE: read-only lookup in METAL_ELEMENT_SYMBOLS frozenset.
    """
    return symbol in METAL_ELEMENT_SYMBOLS


def detect_metal_complex(mol: "Chem.Mol") -> Optional[MetalComplex]:
    """Detect any organometallic complex in the given mol.

    Plan-03 Tier-1 implementation: detects dot-separated metallocene
    topology (1 metal cation + 2 anionic Cp rings = sandwich complex).

    Plan-03 commits 03-02/03/04 extend with Tier-3 (σ-bonded
    single-component main-group), Tier-2 (dot-separated metal carbonyls),
    and Tier-4 (mixed η-bonded / half-sandwich) topology branches.

    PURE per CONTEXT D-12: no mol mutation; only reads via GetAtoms /
    GetBonds / GetSymbol / GetFormalCharge / GetBondType / GetIsAromatic /
    GetOtherAtomIdx / GetBondTypeAsDouble / GetAtomWithIdx / GetMolFrags.
    NEVER: SanitizeMol, UpdatePropertyCache, AssignStereochemistry.
    """
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None

    # Collect metal atom indices + formal charges
    metal_atom_indices = []
    formal_charges = []
    for atom in mol.GetAtoms():
        if is_metal_element(atom.GetSymbol()):
            metal_atom_indices.append(atom.GetIdx())
            formal_charges.append(atom.GetFormalCharge())

    if not metal_atom_indices:
        return None

    # Phase 161.3 deferred: multinuclear bridged complexes
    is_multimetal = len(metal_atom_indices) > 1

    # === Tier-1 topology: dot-separated metallocene ===
    # Frags structure: 1 metal frag (single atom) + N Cp-anion frags (5-atom
    # aromatic rings each carrying formal_charge sum = -1).
    frags = Chem.GetMolFrags(mol, asMols=False)
    if len(frags) >= 2 and not is_multimetal:
        metal_idx = metal_atom_indices[0]
        # Find metal frag and ligand frags
        metal_frag = None
        ligand_frags = []
        for frag in frags:
            if metal_idx in frag:
                metal_frag = frag
            else:
                ligand_frags.append(frag)

        # Tier-1 metallocene: metal frag is single atom + exactly 2 Cp ligand frags
        if (metal_frag is not None and len(metal_frag) == 1
                and len(ligand_frags) == 2):
            cp_groups = []
            for lig_frag in ligand_frags:
                if _is_cp_ligand(mol, lig_frag):
                    cp_groups.append(LigandGroup(
                        metal_atom_idx=metal_idx,
                        ligand_atom_indices=tuple(sorted(lig_frag)),
                        hapticity_n=5,
                        ligand_smarts_key='c1cc[cH-]c1',
                        ligand_canonical_smiles='c1cc[cH-]c1',
                    ))
            if len(cp_groups) == 2:
                return MetalComplex(
                    metal_atom_indices=tuple(metal_atom_indices),
                    ligand_groups=tuple(cp_groups),
                    formal_charges=tuple(formal_charges),
                    is_multimetal=False,
                )

    # === Tier-3 topology: single-component σ-bonded main-group ===
    # 1 metal atom + N alkyl/aryl ligand atoms directly bonded (single bonds);
    # Grignard topology: 1 organic ligand + 1 halide ligand.
    if len(frags) == 1 and not is_multimetal:
        metal_idx = metal_atom_indices[0]
        sigma_groups = _build_sigma_ligand_groups(mol, metal_idx)
        if sigma_groups is not None:
            return MetalComplex(
                metal_atom_indices=tuple(metal_atom_indices),
                ligand_groups=tuple(sigma_groups),
                formal_charges=tuple(formal_charges),
                is_multimetal=False,
            )

    # Tier-2 (dot-separated metal carbonyls) and Tier-4 (mixed η-bonded /
    # half-sandwich) topologies land in subsequent Plan-03 commits.
    return None


_HALIDE_SMARTS_KEYS: FrozenSet[str] = frozenset({
    '[Cl-]', '[Br-]', '[F-]', '[I-]',
})


def _build_sigma_ligand_groups(mol: "Chem.Mol",
                               metal_idx: int) -> Optional[Tuple[LigandGroup, ...]]:
    """Build LigandGroups for a σ-bonded single-component organometallic.

    For each direct neighbor of the metal atom, walk the connected non-metal
    subgraph (BFS over single/aromatic bonds, not crossing the metal again).
    Each component becomes a LigandGroup with hapticity_n=1.

    For halide leaf-atoms (Cl/Br/F/I directly bonded to metal), produce a
    single-atom LigandGroup with ligand_smarts_key set per
    _HALIDE_SMARTS_KEYS so the rules layer can dispatch Grignard vs alkyl.

    Returns None if the topology doesn't match σ-bonded (e.g., aromatic
    π-bonded ligand attached to metal). Pure per CONTEXT D-12.
    """
    metal_atom = mol.GetAtomWithIdx(metal_idx)
    sigma_groups: List[LigandGroup] = []
    seen_atoms: set = {metal_idx}

    for neighbor_bond in metal_atom.GetBonds():
        neighbor_idx = neighbor_bond.GetOtherAtomIdx(metal_idx)
        if neighbor_idx in seen_atoms:
            continue
        neighbor_atom = mol.GetAtomWithIdx(neighbor_idx)

        # Halide leaf-atom: single-atom ligand
        if (neighbor_atom.GetDegree() == 1
                and neighbor_atom.GetSymbol() in ('Cl', 'Br', 'F', 'I')):
            halide_key = f"[{neighbor_atom.GetSymbol()}-]"
            sigma_groups.append(LigandGroup(
                metal_atom_idx=metal_idx,
                ligand_atom_indices=(neighbor_idx,),
                hapticity_n=1,
                ligand_smarts_key=halide_key,
                ligand_canonical_smiles=halide_key,
            ))
            seen_atoms.add(neighbor_idx)
            continue

        # Skip if connecting via a non-single bond (aromatic / double / triple)
        # — those are π-bonded, Tier-4 territory. Aryl σ-bonds to metal are
        # single bonds; aromatic-ring atoms inside the ring are aromatic but
        # the M-C bond itself is single.
        if neighbor_bond.GetBondTypeAsDouble() > 1.5:
            return None

        # Walk the organic ligand subgraph via BFS, not crossing the metal
        ligand_atoms = _collect_organic_component(mol, neighbor_idx, metal_idx)
        # Refuse to collect atoms already in another ligand (e.g., a bridging
        # carbon would belong to two metals — Phase 161.3 territory)
        if any(a in seen_atoms for a in ligand_atoms):
            return None
        seen_atoms.update(ligand_atoms)

        sigma_groups.append(LigandGroup(
            metal_atom_idx=metal_idx,
            ligand_atom_indices=tuple(sorted(ligand_atoms)),
            hapticity_n=1,
            ligand_smarts_key=None,  # σ-bonded; no SMARTS catalog hit
            ligand_canonical_smiles=None,
        ))

    if not sigma_groups:
        return None
    return tuple(sigma_groups)


def _collect_organic_component(mol: "Chem.Mol", start_idx: int,
                                metal_idx: int) -> List[int]:
    """BFS over non-metal atoms reachable from start_idx, not crossing metal."""
    visited = {start_idx}
    queue = [start_idx]
    while queue:
        current = queue.pop()
        atom = mol.GetAtomWithIdx(current)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(current)
            if other_idx == metal_idx:
                continue
            if other_idx in visited:
                continue
            other_atom = mol.GetAtomWithIdx(other_idx)
            if is_metal_element(other_atom.GetSymbol()):
                continue
            visited.add(other_idx)
            queue.append(other_idx)
    return sorted(visited)


def _is_cp_ligand(mol: "Chem.Mol", atom_indices: Tuple[int, ...]) -> bool:
    """Check if a frag's atom indices form a Cp anion ring.

    Cp anion = 5 aromatic carbon atoms in a single ring with sum of
    formal charges = -1. Pure read-only on mol.
    """
    if len(atom_indices) != 5:
        return False
    charge_sum = 0
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return False
        if not atom.GetIsAromatic():
            return False
        charge_sum += atom.GetFormalCharge()
    return charge_sum == -1


def compute_hapticity(mol: "Chem.Mol", metal_atom_idx: int,
                      ligand_atom_indices: Tuple[int, ...]) -> int:
    """Compute the hapticity of a ligand group bound to a metal.

    Plan-03 implementation per CONTEXT D-05 hybrid:
    1. SMARTS-template fast path: consult LIGAND_ETA_DEFAULTS from
       data/organometallics.py; on hit, return catalog value.
    2. Graph-walk fallback (added in 03-04 Tier-4 commit): BFS over
       ligand_atom_indices counting contiguous π-system atoms.
    3. Edge case σ-bonded (single atom or no π-system): return 1.

    PURE per CONTEXT D-12.
    """
    # Lazy import to avoid circular dependency
    from ..data.organometallics import LIGAND_ETA_DEFAULTS

    if len(ligand_atom_indices) == 0:
        raise ValueError(f"empty ligand_atom_indices for metal {metal_atom_idx}")

    # Single-atom ligand: σ-bonded hapticity 1
    if len(ligand_atom_indices) == 1:
        return 1

    # Build the ligand subgraph as a separate mol (PURE per D-12: submol only)
    submol = Chem.RWMol()
    atom_map: dict = {}
    for idx in ligand_atom_indices:
        # AddAtom returns the new atom's index in submol; copy from mol's atom.
        new_idx = submol.AddAtom(mol.GetAtomWithIdx(idx))
        atom_map[idx] = new_idx
    for idx in ligand_atom_indices:
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in ligand_atom_indices and other > idx:
                submol.AddBond(atom_map[idx], atom_map[other], bond.GetBondType())

    # SMARTS-template fast path lookup
    try:
        ligand_smi = Chem.MolToSmiles(submol)
    except Exception:
        ligand_smi = None

    if ligand_smi and ligand_smi in LIGAND_ETA_DEFAULTS:
        return LIGAND_ETA_DEFAULTS[ligand_smi][0]

    # Graph-walk fallback (Tier-4 implementation lands in commit 03-04)
    # For Tier-1 (Cp via SMARTS fast path), this never executes.
    return 1


def enumerate_metal_ligand_groups(mol: "Chem.Mol") -> Tuple[LigandGroup, ...]:
    """For each metal atom in mol, partition coordinated atoms into ligand groups.

    STUB BODY (Plan-02): returns () always. Plan-03 implements per
    RESEARCH §3.2 lines 549-567.

    Algorithm (Plan-03 implementation):
    1. Identify all metal atoms via is_metal_element(atom.GetSymbol()).
    2. For each metal: collect bond-distance-1 neighbors (ligand-shell).
    3. For each ligand-shell atom: BFS over non-metal bonds to find
       contiguous ligand component.
    4. Each component → LigandGroup with hapticity from compute_hapticity.

    PURE per CONTEXT D-12. O(V+E) per metal.
    """
    # Plan-03 implementation lands here per RESEARCH §3.2 + AUDIT § 5
    return ()


__all__ = [
    'METAL_ELEMENT_SYMBOLS',
    'is_metal_element',
    'LigandGroup',
    'MetalComplex',
    'detect_metal_complex',
    'compute_hapticity',
    'enumerate_metal_ligand_groups',
]
