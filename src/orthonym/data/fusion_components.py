"""
Monocyclic component registry for algorithmic fused ring naming.

Provides a registry of common monocyclic ring components with their:
- Fusion prefix form (e.g., 'benzo', 'furo', 'pyrido')
- Ring size
- Heteroatom types and IUPAC positions
- Seniority for parent/child selection (lower = more senior)
- Aromaticity flag

Used by fusion_descriptors.py to:
1. Identify ring components by their heteroatom pattern
2. Determine parent/child using IUPAC seniority
3. Generate correct fusion prefixes

IUPAC 2013 Seniority (P-25.2.1):
- Nitrogen heterocycle > Oxygen heterocycle > Sulfur heterocycle > Carbocycle
- Among same heteroatom class: larger ring > smaller ring
- Among same size/heteroatom: more heteroatoms > fewer

Reference: IUPAC 2013 Blue Book, Section P-25 (Fused Ring Systems)
"""

from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# MONOCYCLIC COMPONENT REGISTRY
# ============================================================================
#
# Each entry maps a component name to its properties.
# 'hetero_positions' lists the IUPAC ring positions (1-indexed) of heteroatoms.
# 'seniority' is a numeric value -- LOWER is MORE SENIOR (parent preference).
#
# Seniority tiers:
#   40-49: 6-membered N-heterocycles (most senior among heterocycles)
#   50-59: 5-membered N-heterocycles
#   60-69: 5-membered N+O heterocycles
#   70-79: O-heterocycles
#   80-89: S-heterocycles / N+S heterocycles
#   200:   Carbocycles (least senior)
# ============================================================================

MONOCYCLIC_COMPONENTS: Dict[str, Dict[str, Any]] = {
    # ---- Carbocyclic ----
    'cyclopentadiene': {
        'prefix': 'cyclopenta',
        'ring_size': 5,
        'heteroatoms': [],
        'hetero_positions': [],
        'seniority': 200,
        'aromatic': True,
    },
    'benzene': {
        'prefix': 'benzo',
        'ring_size': 6,
        'heteroatoms': [],
        'hetero_positions': [],
        'seniority': 200,
        'aromatic': True,
    },
    'cycloheptadiene': {
        'prefix': 'cyclohepta',
        'ring_size': 7,
        'heteroatoms': [],
        'hetero_positions': [],
        'seniority': 200,
        'aromatic': False,
    },

    # ---- 5-membered 1-heteroatom ----
    'furan': {
        'prefix': 'furo',
        'ring_size': 5,
        'heteroatoms': ['O'],
        'hetero_positions': [1],
        'seniority': 71,
        'aromatic': True,
    },
    'thiophene': {
        'prefix': 'thieno',
        'ring_size': 5,
        'heteroatoms': ['S'],
        'hetero_positions': [1],
        'seniority': 85,
        'aromatic': True,
    },
    'pyrrole': {
        'prefix': 'pyrrolo',
        'ring_size': 5,
        'heteroatoms': ['N'],
        'hetero_positions': [1],
        'seniority': 55,
        'aromatic': True,
    },

    # ---- 5-membered 2-heteroatom (N,N) ----
    'imidazole': {
        'prefix': 'imidazo',
        'ring_size': 5,
        'heteroatoms': ['N', 'N'],
        'hetero_positions': [1, 3],   # 1,3-diazole
        'seniority': 47,
        'aromatic': True,
    },
    'pyrazole': {
        'prefix': 'pyrazolo',
        'ring_size': 5,
        'heteroatoms': ['N', 'N'],
        'hetero_positions': [1, 2],   # 1,2-diazole
        'seniority': 48,
        'aromatic': True,
    },

    # ---- 5-membered 2-heteroatom (N,O) ----
    'oxazole': {
        'prefix': 'oxazolo',
        'ring_size': 5,
        'heteroatoms': ['N', 'O'],
        'hetero_positions': [1, 3],   # O at 1, N at 3
        'seniority': 63,
        'aromatic': True,
    },
    'isoxazole': {
        'prefix': 'isoxazolo',
        'ring_size': 5,
        'heteroatoms': ['N', 'O'],
        'hetero_positions': [1, 2],   # O at 1, N at 2
        'seniority': 64,
        'aromatic': True,
    },

    # ---- 5-membered 2-heteroatom (N,S) ----
    'thiazole': {
        'prefix': 'thiazolo',
        'ring_size': 5,
        'heteroatoms': ['N', 'S'],
        'hetero_positions': [1, 3],   # S at 1, N at 3
        'seniority': 77,
        'aromatic': True,
    },
    'isothiazole': {
        'prefix': 'isothiazolo',
        'ring_size': 5,
        'heteroatoms': ['N', 'S'],
        'hetero_positions': [1, 2],   # S at 1, N at 2
        'seniority': 78,
        'aromatic': True,
    },

    # ---- 6-membered 1-heteroatom ----
    'pyridine': {
        'prefix': 'pyrido',
        'ring_size': 6,
        'heteroatoms': ['N'],
        'hetero_positions': [1],
        'seniority': 49,
        'aromatic': True,
    },
    'pyran': {
        'prefix': 'pyrano',
        'ring_size': 6,
        'heteroatoms': ['O'],
        'hetero_positions': [1],
        'seniority': 72,
        'aromatic': False,
    },

    # ---- 6-membered 2-heteroatom (N,N) ----
    'pyrimidine': {
        'prefix': 'pyrimido',
        'ring_size': 6,
        'heteroatoms': ['N', 'N'],
        'hetero_positions': [1, 3],   # 1,3-diazine
        'seniority': 43,
        'aromatic': True,
    },
    'pyrazine': {
        'prefix': 'pyrazino',
        'ring_size': 6,
        'heteroatoms': ['N', 'N'],
        'hetero_positions': [1, 4],   # 1,4-diazine
        'seniority': 45,
        'aromatic': True,
    },
    'pyridazine': {
        'prefix': 'pyridazino',
        'ring_size': 6,
        'heteroatoms': ['N', 'N'],
        'hetero_positions': [1, 2],   # 1,2-diazine
        'seniority': 44,
        'aromatic': True,
    },
}


def get_component_by_pattern(
    ring_size: int,
    heteroatom_symbols: List[str],
    hetero_gap: Optional[int] = None,
) -> Optional[str]:
    """
    Find a component name matching the given ring structure.

    For rings with 2+ heteroatoms of the same type, the hetero_gap (shortest
    number of non-hetero atoms between heteroatoms walking around the ring)
    is required to distinguish positional isomers:
    - gap=0: adjacent (pyrazole, pyridazine, isoxazole, isothiazole)
    - gap=1: separated by 1 C (imidazole, pyrimidine, oxazole, thiazole)
    - gap=2: separated by 2 C (pyrazine)

    Args:
        ring_size: Number of atoms in the ring
        heteroatom_symbols: List of heteroatom element symbols (sorted)
        hetero_gap: Shortest gap between heteroatoms (for disambiguation)

    Returns:
        Component name (e.g., 'pyrimidine') or None if no match
    """
    sorted_symbols = sorted(heteroatom_symbols)

    for name, props in MONOCYCLIC_COMPONENTS.items():
        if props['ring_size'] != ring_size:
            continue
        if sorted(props['heteroatoms']) != sorted_symbols:
            continue

        # For single-heteroatom or carbocyclic rings, no gap needed
        if len(sorted_symbols) <= 1:
            return name

        # For 2+ heteroatom rings, use gap to distinguish positional isomers
        if hetero_gap is not None:
            # The gap encodes which positional isomer this is
            # hetero_positions tells us expected gap for this component
            positions = props['hetero_positions']
            if len(positions) >= 2:
                expected_gap = positions[1] - positions[0] - 1
                if expected_gap == hetero_gap:
                    return name

    return None


def get_component_seniority(name: str) -> int:
    """
    Get the seniority value for a named component.

    Lower seniority = more senior (preferred as parent in fusion naming).
    Returns 999 for unknown components.

    Args:
        name: Component name (e.g., 'pyrimidine', 'benzene')

    Returns:
        Seniority value (int). Lower = more senior.
    """
    if name in MONOCYCLIC_COMPONENTS:
        return MONOCYCLIC_COMPONENTS[name]['seniority']
    return 999


def get_component_prefix(name: str) -> str:
    """
    Get the fusion prefix form for a named component.

    Returns the prefix used when this ring is the child (attached component)
    in a fusion name. Falls back to the name + 'o' if not in registry.

    Args:
        name: Component name (e.g., 'benzene' -> 'benzo', 'furan' -> 'furo')

    Returns:
        Fusion prefix string
    """
    if name in MONOCYCLIC_COMPONENTS:
        return MONOCYCLIC_COMPONENTS[name]['prefix']
    # Fallback: apply general suffix rules
    if name.endswith('ene'):
        return name[:-3] + 'o'
    if name.endswith('ole'):
        return name[:-1] + 'o'
    if name.endswith('ine'):
        return name[:-1] + 'o'
    if name.endswith('an'):
        return name[:-2] + 'o'
    return name + 'o'


def get_component_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get full property dict for a named component.

    Args:
        name: Component name

    Returns:
        Property dict or None if not found
    """
    return MONOCYCLIC_COMPONENTS.get(name)
