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
from typing import FrozenSet, Optional, Tuple
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

    STUB BODY (Plan-02): returns None always. Plan-03 implements per
    RESEARCH §3.2 lines 516-525 (per-tier branches in Plan-03 commits
    03-01..03-04).

    PURE per CONTEXT D-12: no mol mutation; only reads via GetAtoms /
    GetBonds / GetSymbol / GetFormalCharge / GetBondType /
    GetIsAromatic / GetOtherAtomIdx / GetBondTypeAsDouble /
    GetAtomWithIdx / GetSymmSSSR (read-only) / MolToSmiles (submol-only).
    NEVER: SanitizeMol, UpdatePropertyCache, AssignStereochemistry,
    RWMol(mol).SetAtomMapNum, Orthonym(...).name(...).
    """
    # Plan-03 implementation lands here per RESEARCH §3.2 + AUDIT § 5
    return None


def compute_hapticity(mol: "Chem.Mol", metal_atom_idx: int,
                      ligand_atom_indices: Tuple[int, ...]) -> int:
    """Compute the hapticity of a ligand group bound to a metal.

    STUB BODY (Plan-02): returns 1 always (σ-bonded default). Plan-03
    implements per RESEARCH §3.2 lines 528-545 + §3.4 graph-walk
    pseudocode (lines 606-645).

    Algorithm (Plan-03 implementation per CONTEXT D-05 hybrid):
    1. SMARTS-template fast path: consult LIGAND_ETA_DEFAULTS from
       data/organometallics.py; on hit, return catalog value.
    2. Graph-walk fallback: BFS over ligand_atom_indices counting
       contiguous π-system atoms (aromatic OR has double/triple bond).
    3. Edge case σ-bonded (no π-system): return 1.
    4. Edge case no contiguous π-system: raise ValueError (caught at
       handler layer per CONTEXT D-12 honest-fail-on-data).

    PURE per CONTEXT D-12.
    """
    # Plan-03 implementation lands here per RESEARCH §3.4 + AUDIT § 5
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
