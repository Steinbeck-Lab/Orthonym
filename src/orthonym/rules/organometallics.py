"""
IUPAC P-69 organometallic seniority + naming rules (Phase 161 v19).

This module defines ORGM-specific seniority cascade SEPARATE from the
P-43 carbon-organic seniority in src/orthonym/rules/seniority.py per
CONTEXT D-06 hard invariant (ZERO edits to seniority.py).

The two cascades NEVER share data structures; the only common file is
src/orthonym/data/organometallics.py (NEW; ORGM-exclusive).

Anti-patterns to avoid (PATTERNS lines 256-259):
- D-06 forbids any import path pointing at rules/seniority (relative
  ..rules dot-seniority or absolute orthonym dot rules dot seniority);
  this enforces ZERO cross-contamination with the P-43 carbon-organic
  seniority cascade. The grep gate in Plan-02 task 02-03 asserts neither
  import line exists in this module.
- NEVER add ORGM entries to the existing SENIORITY_ORDER list in
  seniority.py — define a NEW ORGM_LIGAND_ORDER in this file instead.
- NEVER call name_compound(...) recursively from
  assemble_organometallic_name(...) — recursion path is
  OUTER CFR → handler → rules → data; no upward call.

Phase 161 (v19 first scope-expansion phase per ADR-19-07).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# Import from data layer only; D-06 forbids cross-contamination with
# the P-43 seniority cascade (no import of rules dot seniority).
from ..data.organometallics import (
    METAL_NAMES, LIGAND_NAMES, LIGAND_ETA_DEFAULTS, RETAINED_METALLOCENES,
)
from ..assembly.name_tree import NameTreeNode


# Phase 161 Stock-notation Roman numerals (scope -3 .. +8 per CONTEXT D-07)
_ROMAN_NUMERALS: Dict[int, str] = {
    0: '0', 1: 'I', 2: 'II', 3: 'III', 4: 'IV',
    5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII',
    -1: '-I', -2: '-II', -3: '-III',
}


def _to_roman(n: int) -> str:
    """Stock-notation oxidation state in Roman numerals.

    Per CONTEXT D-07: Phase 161 scope is -3..+8. Raises ValueError outside
    that range (caller catches per CONTEXT D-12 honest-fail-on-data).
    """
    if n not in _ROMAN_NUMERALS:
        raise ValueError(
            f"Stock-notation oxidation state {n} out of Phase 161 scope (-3..+8)"
        )
    return _ROMAN_NUMERALS[n]


# Multiplicative prefixes for simple ligand counts per Salzer §5.2.
_SIMPLE_MULTIPLICATIVE_PREFIXES: Dict[int, str] = {
    1: '',
    2: 'di',
    3: 'tri',
    4: 'tetra',
    5: 'penta',
    6: 'hexa',
    7: 'hepta',
    8: 'octa',
}


def _multiplicative_prefix(count: int) -> str:
    """Return the multiplicative prefix for a simple ligand count."""
    if count not in _SIMPLE_MULTIPLICATIVE_PREFIXES:
        raise ValueError(f"ligand count {count} outside simple-prefix range (1..8)")
    return _SIMPLE_MULTIPLICATIVE_PREFIXES[count]


# Halide ligand name swaps for Grignard form (chlorido → chloride, etc.)
_HALIDE_LIGAND_TO_HALIDE_WORD: Dict[str, str] = {
    '[F-]':  'fluoride',
    '[Cl-]': 'chloride',
    '[Br-]': 'bromide',
    '[I-]':  'iodide',
}


def _ligand_name_from_atoms(mol: Any, atom_indices: Tuple[int, ...]) -> Optional[str]:
    """Identify the IUPAC name of an alkyl/aryl ligand from its atom indices.

    Recognised ligands (Phase 161 scope; per LIGAND_NAMES + AUDIT § 2):
    - methyl (1 C, all sp³)
    - ethyl (2 C, all sp³)
    - propyl / n-propyl (3 C, all sp³, linear)
    - butyl / n-butyl (4 C, all sp³, linear)
    - phenyl (6 aromatic C in a ring)
    - vinyl / ethenyl (2 C with C=C double bond)

    Returns None if the ligand doesn't match any known pattern.
    """
    n_atoms = len(atom_indices)
    if n_atoms == 0:
        return None

    atoms = [mol.GetAtomWithIdx(i) for i in atom_indices]
    symbols = [a.GetSymbol() for a in atoms]

    # All carbons?
    if not all(s == 'C' for s in symbols):
        return None

    aromatic_count = sum(1 for a in atoms if a.GetIsAromatic())

    # Phenyl: 6 aromatic carbons in a ring
    if n_atoms == 6 and aromatic_count == 6:
        return 'phenyl'

    # Alkyl groups: all sp³ carbons in a linear chain
    if aromatic_count == 0:
        # Check linearity: degree 1 carbons at the ends, degree 2 in middle
        # (the metal-attached carbon will have degree 1 since metal isn't in
        # atom_indices). For n_atoms == 1, it's methyl.
        if n_atoms == 1:
            return 'methyl'

        # Build internal degree map (degree within the ligand subgraph)
        idx_set = set(atom_indices)
        internal_degrees = {}
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            internal_degrees[idx] = sum(
                1 for b in atom.GetBonds()
                if b.GetOtherAtomIdx(idx) in idx_set
            )
        # Linear chain: exactly 2 atoms with degree 1; rest with degree 2
        deg_counts = list(internal_degrees.values())
        if deg_counts.count(1) == 2 and deg_counts.count(2) == n_atoms - 2:
            return {2: 'ethyl', 3: 'propyl', 4: 'butyl',
                    5: 'pentyl', 6: 'hexyl', 7: 'heptyl',
                    8: 'octyl'}.get(n_atoms)

    return None


# === ORGM_LIGAND_ORDER (Salzer 1999 §5.2 alphabetic with multiplicatives ignored) ===
# This is NOT the carbon-organic seniority of P-43 — that stays UNTOUCHED
# in src/orthonym/rules/seniority.py per CONTEXT D-06.
#
# Used for alphabetizing ligands in coordination names per RESEARCH §1.3.
# Multiplicative prefixes (di-, tri-, tetra-, bis-, tris-, tetrakis-)
# are IGNORED for alphabetization; structural prefixes (cyclo-, η, κ, μ)
# are INCLUDED.
ORGM_LIGAND_ORDER: List[str] = [
    # Alphabetical ligand names per Salzer 1999 §5.2
    'allyl',
    'benzene',
    'bromido',
    '1,3-butadiene',
    'carbonyl',
    'chlorido',
    'cycloheptatrienyl',
    'cyclooctatetraene',
    'cyclopentadienyl',
    'ethene',
    'ethyne',
    'fluorido',
    'hydrido',
    'iodido',
    'methyl',
    'pentamethylcyclopentadienyl',
    'phenyl',
    'tropylium',
]


# === METAL_OXIDATION_STATE_HINTS (per CONTEXT D-07 + AUDIT § 6) ===
# (metal_symbol, ligand_class) → {default_state, stock_required, iupac_cite}
# Stock notation rule per Salzer 1999 §3.2 + Red Book IR-10.
METAL_OXIDATION_STATE_HINTS: Dict[Tuple[str, str], Dict[str, Any]] = {
    # Tier 1 metallocenes (Cp2 ligand class; Stock required in systematic ONLY)
    ('Fe', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Ru', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Os', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Co', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Ni', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Cr', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('V',  'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Mn', 'Cp2'): {'default_state': 2, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Fe', 'Cp2_cation'): {'default_state': 3, 'stock_required_systematic': True,
                            'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    # Tier 2 metal carbonyls (always include Stock per Salzer §6)
    ('Fe', 'CO5'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('Ni', 'CO4'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('Cr', 'CO6'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('Mo', 'CO6'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    ('W',  'CO6'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §6'},
    # Charged carbonyl anions (ALWAYS Stock per Ewens-Bassett or Stock convention)
    ('V',  'CO6_anion'): {'default_state': -1, 'stock_required_systematic': True,
                          'stock_required_pin': True, 'iupac_cite': 'Salzer §6'},
    ('Co', 'CO4_anion'): {'default_state': -1, 'stock_required_systematic': True,
                          'stock_required_pin': True, 'iupac_cite': 'Salzer §6'},
    ('Mn', 'CO5_anion'): {'default_state': -1, 'stock_required_systematic': True,
                          'stock_required_pin': True, 'iupac_cite': 'Salzer §6'},
    # Tier 3 σ-bonded main-group (NEVER Stock per D-07 — single canonical state)
    ('Li', 'alkyl'): {'default_state': 1, 'stock_required_systematic': False,
                      'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Na', 'alkyl'): {'default_state': 1, 'stock_required_systematic': False,
                      'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('K',  'alkyl'): {'default_state': 1, 'stock_required_systematic': False,
                      'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Mg', 'alkyl_halide'): {'default_state': 2, 'stock_required_systematic': False,
                              'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Zn', 'alkyl2'): {'default_state': 2, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Cd', 'alkyl2'): {'default_state': 2, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Hg', 'alkyl2'): {'default_state': 2, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    ('Al', 'alkyl3'): {'default_state': 3, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.3'},
    # Tier 3 hydride-parent main-group (Si/Ge/Sn/Pb; NEVER Stock — implicit +4)
    ('Si', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    ('Ge', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    ('Sn', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    ('Pb', 'alkyl4'): {'default_state': 4, 'stock_required_systematic': False,
                       'stock_required_pin': False, 'iupac_cite': 'P-69.2'},
    # Tier 4 mixed sandwich + arene
    ('Cr', 'bz2'): {'default_state': 0, 'stock_required_systematic': True,
                    'stock_required_pin': False, 'iupac_cite': 'Salzer §5.4'},
    ('Cr', 'bz_CO3'): {'default_state': 0, 'stock_required_systematic': True,
                       'stock_required_pin': False, 'iupac_cite': 'Salzer §5+§6'},
    ('Mn', 'Cp_CO3'): {'default_state': 1, 'stock_required_systematic': True,
                       'stock_required_pin': False, 'iupac_cite': 'Salzer §5+§6'},
}


# === METAL_RANKING_FOR_PARENT_SELECTION (per CONTEXT D-06) ===
# When multiple metals present, which becomes the central-atom parent:
# defaults to highest-oxidation-state metal; ties broken by atomic number.
# Phase 161 D-01 defers polynuclear bridges to Phase 161.3 — this constant
# is present for forward compatibility; multinuclear compounds return None
# from name_organometallic per Risk R-08 mitigation.
METAL_RANKING_FOR_PARENT_SELECTION: Dict[str, int] = {
    'Li': 3, 'Na': 11, 'K': 19, 'Mg': 12,
    'Zn': 30, 'Cd': 48, 'Hg': 80,
    'B': 5, 'Al': 13, 'Ga': 31, 'In': 49, 'Tl': 81,
    'Si': 14, 'Ge': 32, 'Sn': 50, 'Pb': 82,
    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Mo': 42, 'W': 74,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Os': 76, 'Ir': 77, 'Pt': 78,
}


def select_ligand_naming(ligand_group: Any, hapticity: int,
                         style: str = "pin") -> str:
    """Generate the IUPAC name for a single ligand with η-notation.

    STUB BODY (Plan-02): returns empty string. Plan-03 implements per
    RESEARCH §1.3 + AUDIT § 2 SMARTS catalog.

    Algorithm (Plan-03 implementation):
    1. Look up ligand_group.ligand_canonical_smiles in LIGAND_NAMES.
    2. If hapticity > 1 OR ligand in LIGAND_ETA_DEFAULTS: prefix η<N>-.
    3. Apply multiplicative prefix (bis-, tris-, etc.) per outer caller.
    """
    return ""


def assemble_organometallic_name(metal_complex: Any, mol: Any,
                                  style: str = "pin"
                                  ) -> Optional[Tuple[str, str, List[Any]]]:
    """Assemble systematic IUPAC name for a metal complex per Salzer 1999 §5.

    Plan-03 implementation: Tier-1 Cp2 metallocene branch (commit 03-01).
    Subsequent commits 03-02/03/04 add Tier-3/2/4 branches.

    Returns: (full_name, metal_name_part, ligand_tree_nodes) per
    RESEARCH §4.3 lines 799-815. The handler wraps this in NameTreeNode.

    Returns None to signal cascade-continuation when:
    - metal_complex.is_multimetal (Phase 161.3 deferred per Risk R-08)
    - Stock-notation lookup fails for the (metal, ligand_class) tuple
    - The compound's topology doesn't match any tier branch yet
    """
    # Risk R-08: defer multinuclear bridges to Phase 161.3
    if metal_complex.is_multimetal:
        return None

    metal_idx = metal_complex.metal_atom_indices[0]
    metal_symbol = mol.GetAtomWithIdx(metal_idx).GetSymbol()
    metal_charge = metal_complex.formal_charges[0]
    ligand_groups = metal_complex.ligand_groups

    # === TIER-2 dispatch: mononuclear metal carbonyls ===
    if (ligand_groups
            and all(lg.ligand_smarts_key == '[C-]#[O+]' for lg in ligand_groups)):
        n_co = len(ligand_groups)
        co_prefix = _multiplicative_prefix(n_co)

        # Determine ligand_class for Stock lookup (neutral vs anionic)
        if metal_charge == 0:
            ligand_class = f'CO{n_co}'
        else:
            ligand_class = f'CO{n_co}_anion'

        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, ligand_class))
        if hints is None:
            return None

        metal_name_info = METAL_NAMES.get(metal_symbol)
        if metal_name_info is None:
            return None
        metal_name = metal_name_info['direct']

        include_stock = (
            hints['stock_required_systematic'] if style == 'systematic'
            else hints['stock_required_pin']
        )

        # Stock notation per CONTEXT D-07:
        # - Systematic: always Roman (e.g., (0), (-I))
        # - PIN with anionic metals: Ewens-Bassett charge form (e.g., (1-))
        # - PIN with neutral metals: omit
        if include_stock:
            oxidation = metal_charge if metal_charge != 0 else hints['default_state']
            if metal_charge < 0 and style == 'pin':
                stock_str = f"({abs(metal_charge)}-)"
            elif metal_charge > 0 and style == 'pin':
                stock_str = f"({metal_charge}+)"
            else:
                stock_str = f"({_to_roman(oxidation)})"
        else:
            stock_str = ""

        # Salzer §5: ligand-first composition order — carbonyl prefix before metal
        full_name = f"{co_prefix}carbonyl{metal_name}{stock_str}"
        metal_name_part = f"{metal_name}{stock_str}"
        co_node = NameTreeNode(
            parent_stem='carbonyl',
            multiplicative_prefix=co_prefix or None,
            class_id='organometallic_ligand',
        )
        return (full_name, metal_name_part, [co_node])

    # === TIER-1 dispatch: bis(η5-cyclopentadienyl)M sandwich complexes ===
    if (len(ligand_groups) == 2
            and all(lg.ligand_smarts_key == 'c1cc[cH-]c1' for lg in ligand_groups)):
        # Cp2 ligand class; metal in +2 (or +3 for ferrocenium) oxidation state
        ligand_class = 'Cp2_cation' if metal_charge == 3 else 'Cp2'
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, ligand_class))
        if hints is None:
            return None  # cascade to SALT@100

        metal_name_info = METAL_NAMES.get(metal_symbol)
        if metal_name_info is None:
            return None
        metal_name = metal_name_info['direct']

        include_stock = (
            hints['stock_required_systematic'] if style == 'systematic'
            else hints['stock_required_pin']
        )

        # Use empirical formal charge for Stock when nonzero, else hint default
        oxidation_state = metal_charge if metal_charge != 0 else hints['default_state']
        stock_str = f"({_to_roman(oxidation_state)})" if include_stock else ""

        # Salzer 1999 §5.4: bis(η⁵-cyclopentadienyl)<metal>(<Stock>)
        full_name = f"bis(η⁵-cyclopentadienyl){metal_name}{stock_str}"
        metal_name_part = f"{metal_name}{stock_str}"

        ligand_node = NameTreeNode(
            parent_stem='η⁵-cyclopentadienyl',
            multiplicative_prefix='bis',
            class_id='organometallic_ligand',
        )
        return (full_name, metal_name_part, [ligand_node])

    # === TIER-3 dispatch: σ-bonded main-group organometallics ===
    if all(lg.hapticity_n == 1 for lg in ligand_groups):
        metal_name_info = METAL_NAMES.get(metal_symbol)
        if metal_name_info is None:
            return None
        naming_system = metal_name_info['naming_system']

        # Partition into organic ligands and halide ligands
        halide_ligs = [
            lg for lg in ligand_groups
            if lg.ligand_smarts_key in _HALIDE_LIGAND_TO_HALIDE_WORD
        ]
        organic_ligs = [
            lg for lg in ligand_groups
            if lg.ligand_smarts_key not in _HALIDE_LIGAND_TO_HALIDE_WORD
        ]

        # Identify organic ligand names from atom indices
        organic_names: List[Optional[str]] = [
            _ligand_name_from_atoms(mol, lg.ligand_atom_indices)
            for lg in organic_ligs
        ]
        if any(n is None for n in organic_names):
            return None  # cascade to SALT@100

        # === Branch A: hydride-parent system (Group 14: Si/Ge/Sn/Pb) ===
        if naming_system == 'hydride_parent':
            parent_name = metal_name_info['hydride_parent']
            if parent_name is None:
                return None
            if halide_ligs:
                # Halide on Si/Ge/Sn/Pb: out of Phase 161 scope (defer)
                return None
            # Group identical ligands
            grouped = _group_ligand_counts(organic_names)
            sorted_groups = _alphabetize_simple_ligands(grouped)
            ligand_prefix = ''.join(
                _multiplicative_prefix(count) + name
                for count, name in sorted_groups
            )
            full_name = f"{ligand_prefix}{parent_name}"
            ligand_tree_nodes = [
                NameTreeNode(
                    parent_stem=name,
                    multiplicative_prefix=_multiplicative_prefix(count) or None,
                    class_id='organometallic_ligand',
                )
                for count, name in sorted_groups
            ]
            # No Stock notation (hydride-parent system implicit +4)
            return (full_name, parent_name, ligand_tree_nodes)

        # === Branch B: metal-direct system (Groups 1/2/12/13) ===
        if naming_system == 'metal_direct':
            metal_name = metal_name_info['direct']

            # Determine ligand_class for Stock lookup
            if halide_ligs and metal_symbol == 'Mg':
                ligand_class = 'alkyl_halide'
            elif metal_symbol in ('Zn', 'Cd', 'Hg') and len(organic_ligs) == 2:
                ligand_class = 'alkyl2'
            elif metal_symbol == 'Al' and len(organic_ligs) == 3:
                ligand_class = 'alkyl3'
            elif metal_symbol in ('Li', 'Na', 'K') and len(organic_ligs) == 1:
                ligand_class = 'alkyl'
            else:
                # Unknown combination — cascade to SALT@100
                return None

            hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, ligand_class))
            if hints is None:
                return None

            include_stock = (
                hints['stock_required_systematic'] if style == 'systematic'
                else hints['stock_required_pin']
            )

            # Grignard: space-separated "ethylmagnesium bromide" form
            if ligand_class == 'alkyl_halide':
                organic_name = organic_names[0]  # exactly one organic ligand
                halide_word = _HALIDE_LIGAND_TO_HALIDE_WORD[halide_ligs[0].ligand_smarts_key]
                full_name = f"{organic_name}{metal_name} {halide_word}"
                metal_name_part = metal_name
                ligand_tree_nodes = [
                    NameTreeNode(parent_stem=organic_name,
                                 class_id='organometallic_ligand'),
                    NameTreeNode(parent_stem=halide_word,
                                 class_id='organometallic_ligand'),
                ]
                return (full_name, metal_name_part, ligand_tree_nodes)

            # Single-component multi-alkyl (dimethylzinc, trimethylaluminum, etc.)
            grouped = _group_ligand_counts(organic_names)
            sorted_groups = _alphabetize_simple_ligands(grouped)
            ligand_prefix = ''.join(
                _multiplicative_prefix(count) + name
                for count, name in sorted_groups
            )

            if include_stock:
                oxidation = metal_charge if metal_charge != 0 else hints['default_state']
                stock_str = f"({_to_roman(oxidation)})"
            else:
                stock_str = ""

            full_name = f"{ligand_prefix}{metal_name}{stock_str}"
            metal_name_part = f"{metal_name}{stock_str}"
            ligand_tree_nodes = [
                NameTreeNode(
                    parent_stem=name,
                    multiplicative_prefix=_multiplicative_prefix(count) or None,
                    class_id='organometallic_ligand',
                )
                for count, name in sorted_groups
            ]
            return (full_name, metal_name_part, ligand_tree_nodes)

    # Tier-2 / Tier-4 dispatch lands in commits 03-03/04
    return None


def _group_ligand_counts(ligand_names: List[Optional[str]]) -> List[Tuple[int, str]]:
    """Group identical ligand names with their occurrence counts.

    Returns [(count, name), ...]. Names are not yet sorted.
    """
    from collections import Counter
    counter = Counter(n for n in ligand_names if n is not None)
    return [(count, name) for name, count in counter.items()]


def _alphabetize_simple_ligands(
    grouped: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """Sort ligand entries alphabetically per Salzer §5.2.

    Multiplicative prefixes (di-, tri-, tetra-) are IGNORED — alphabetize
    by the ligand name itself (the input here is already stripped).
    """
    return sorted(grouped, key=lambda entry: entry[1])


__all__ = [
    'ORGM_LIGAND_ORDER',
    'METAL_OXIDATION_STATE_HINTS',
    'METAL_RANKING_FOR_PARENT_SELECTION',
    'select_ligand_naming',
    'assemble_organometallic_name',
]
