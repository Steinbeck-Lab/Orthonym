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

    # Ethenyl (CH2=CH-): a clean 2-carbon σ-ligand joined by a C=C double
    # bond. P-31.1.4.3.4 — 'ethenyl' is the PIN substituent prefix ('vinyl'
    # is retained, general-nomenclature only; both OPSIN-RT). This case was
    # listed in the docstring but the body never implemented it, so C=C[M]
    # silently collapsed to 'ethyl' — a structure-loss bug that dropped the
    # double bond (→ a different molecule, SELF-01-suppressed in production).
    # Scope is the clean 2-carbon terminal vinyl ONLY; any larger / branched /
    # internal unsaturated ligand stays None (fail-closed) — not yet
    # confidently nameable.
    if n_atoms == 2 and aromatic_count == 0:
        bond = mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            return 'ethenyl'

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


# v22 G2 COV-02 (P-68.2): a bare -OH/-NH2/-SH bonded DIRECTLY to a Group-14
# parent-hydride centre (Si/Ge/Sn/Pb) is a principal characteristic group, named
# with a substitutive suffix on the hydride stem (silanol/silanamine/silanethiol)
# — NOT a ligand. element-Z + H-count -> the suffix word.
_GROUP14_PRINCIPAL_SUFFIX: Dict[Tuple[int, int], str] = {
    (8, 1): 'ol',      # -OH  (hydroxy)
    (7, 2): 'amine',   # -NH2 (primary amine)
    (16, 1): 'thiol',  # -SH  (sulfanyl)
}

# v22 G2 COV-02: the substitutive -ol/-amine/-thiol suffix mode is Group-14 ONLY
# (Si/Ge/Sn/Pb). Boron is ALSO a 'hydride_parent' naming system but its hydroxy
# acid is named with the boronic/borinic-acid characteristic group (P-68.1/P-68.3,
# e.g. phenylboronic acid), NOT 'phenylboranediol' — so B must be EXCLUDED from
# the principal-group diversion and left to cascade to the boronic-acid handler.
_GROUP14_SUFFIX_ELEMENTS = frozenset({'Si', 'Ge', 'Sn', 'Pb'})


def _principal_suffix_for_ligand(mol: Any, lg: Any) -> Optional[str]:
    """Return the substitutive suffix ('ol'/'amine'/'thiol') if ``lg`` is a bare
    neutral -OH/-NH2/-SH single-atom ligand, else ``None`` (fail-closed).

    Strictly single-atom O/N/S with exactly one heavy neighbour (the metal) and
    the canonical H-count, so ethers (Si-O-C, fragment has 2 atoms), secondary
    amines (Si-NH-C), metal-oxo (=O, 0 H), and siloxides (charged) all decline.
    """
    idxs = lg.ligand_atom_indices
    if len(idxs) != 1:
        return None
    atom = mol.GetAtomWithIdx(idxs[0])
    if atom.GetFormalCharge() != 0:
        return None
    heavy_deg = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1)
    if heavy_deg != 1:
        return None
    return _GROUP14_PRINCIPAL_SUFFIX.get((atom.GetAtomicNum(), atom.GetTotalNumHs()))


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
    # Cycloheptatrienyl-Mn(CO)3 — Mn in +1 oxidation state per Salzer §5+§6
    ('Mn', 'CHT_CO3'): {'default_state': 1, 'stock_required_systematic': True,
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

    # === TIER-1 dispatch: bis(η⁵-Cp-class)M sandwich complexes ===
    # Match 2 identical Cp-class ligand groups (hapticity 5, both same key).
    if (len(ligand_groups) == 2
            and all(lg.hapticity_n == 5 for lg in ligand_groups)
            and ligand_groups[0].ligand_smarts_key == ligand_groups[1].ligand_smarts_key
            and ligand_groups[0].ligand_smarts_key is not None
            and ligand_groups[0].ligand_smarts_key in LIGAND_ETA_DEFAULTS):
        # Lookup the Cp variant name (cyclopentadienyl / methylcyclopentadienyl /
        # pentamethylcyclopentadienyl)
        _hap, cp_name = LIGAND_ETA_DEFAULTS[ligand_groups[0].ligand_smarts_key]

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

        oxidation_state = metal_charge if metal_charge != 0 else hints['default_state']
        stock_str = f"({_to_roman(oxidation_state)})" if include_stock else ""

        # Salzer 1999 §5.4: bis(η⁵-<cp_name>)<metal>(<Stock>)
        full_name = f"bis(η⁵-{cp_name}){metal_name}{stock_str}"
        metal_name_part = f"{metal_name}{stock_str}"

        ligand_node = NameTreeNode(
            parent_stem=f"η⁵-{cp_name}",
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

        # Partition into halide, (Group-14 only) principal-group, and organic
        # ligands. v22 G2 COV-02: a bare -OH/-NH2/-SH directly on a Si/Ge/Sn/Pb
        # centre is a principal characteristic group, not a ligand — divert it so
        # Branch A can emit it as a substitutive suffix (-> trimethylsilanol).
        # The diversion is Group-14-scoped so the metal-direct branch keeps its
        # exact prior behaviour (such ligands stay organic -> None -> cascade).
        is_group14 = metal_symbol in _GROUP14_SUFFIX_ELEMENTS
        halide_ligs: List[Any] = []
        pg_ligs: List[Any] = []
        organic_ligs: List[Any] = []
        for lg in ligand_groups:
            if lg.ligand_smarts_key in _HALIDE_LIGAND_TO_HALIDE_WORD:
                halide_ligs.append(lg)
            elif is_group14 and _principal_suffix_for_ligand(mol, lg) is not None:
                pg_ligs.append(lg)
            else:
                organic_ligs.append(lg)

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
            ligand_tree_nodes = [
                NameTreeNode(
                    parent_stem=name,
                    multiplicative_prefix=_multiplicative_prefix(count) or None,
                    class_id='organometallic_ligand',
                )
                for count, name in sorted_groups
            ]

            if pg_ligs:
                # v22 G2 COV-02 (P-68.2): emit the bare -OH/-NH2/-SH on the
                # Group-14 centre as a substitutive suffix on the hydride stem
                # (trimethylsilanol, dimethylsilanediol, trimethylsilanamine,
                # trimethylsilanethiol). Every principal group must be the same
                # kind — a mixed -OH/-NH2 centre is ambiguous, so fail closed.
                suffix_kinds = {
                    _principal_suffix_for_ligand(mol, lg) for lg in pg_ligs
                }
                if len(suffix_kinds) != 1:
                    return None
                from ..assembly.naming_utils import (
                    apply_vowel_elision, _join_multiplied_suffix,
                )
                suffix = suffix_kinds.pop()
                mult = _multiplicative_prefix(len(pg_ligs))  # '', 'di', 'tri'
                full_suffix = _join_multiplied_suffix(mult, suffix)
                parent_with_suffix = apply_vowel_elision(parent_name, full_suffix)
                full_name = f"{ligand_prefix}{parent_with_suffix}"
                # No Stock notation (hydride-parent system implicit +4)
                return (full_name, parent_name, ligand_tree_nodes)

            full_name = f"{ligand_prefix}{parent_name}"
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

    # === TIER-4 dispatch: mixed π-ligand sandwich / half-sandwich ===
    # Fires AFTER Tier-1/2/3 (which handle specific topologies). Detected
    # when any ligand has hapticity > 1 (η-bonded).
    if any(lg.hapticity_n > 1 for lg in ligand_groups):
        return _assemble_tier4(metal_complex, mol, style)

    # No tier matched — cascade to SALT@100
    return None


def _assemble_tier4(metal_complex: Any, mol: Any,
                     style: str = "pin") -> Optional[Tuple[str, str, List[Any]]]:
    """Tier-4 assembly for mixed η-bonded / half-sandwich complexes.

    Per Salzer 1999 §5 + audit § 1 lock: CO (carbonyl) ligands are emitted
    BEFORE η-bonded ligands in the prefix sequence, even though strict
    Salzer alphabetic order would place benzene (b) before carbonyl (c).
    The audit-locked output convention reflects established IUPAC practice
    for organometallic carbonyl complexes (cf. Wikipedia "Tricarbonyl
    (η⁶-benzene)chromium").
    """
    metal_idx = metal_complex.metal_atom_indices[0]
    metal_symbol = mol.GetAtomWithIdx(metal_idx).GetSymbol()
    metal_charge = metal_complex.formal_charges[0]
    ligand_groups = metal_complex.ligand_groups

    metal_name_info = METAL_NAMES.get(metal_symbol)
    if metal_name_info is None:
        return None
    metal_name = metal_name_info['direct']

    # Partition ligands: CO + η-bonded
    co_ligands = [lg for lg in ligand_groups if lg.ligand_smarts_key == '[C-]#[O+]']
    pi_ligands = [lg for lg in ligand_groups if lg.ligand_smarts_key != '[C-]#[O+]']

    # Resolve each π-ligand to (hapticity, ligand_iupac_name)
    pi_names: List[Tuple[int, str]] = []
    for lg in pi_ligands:
        if lg.ligand_smarts_key is None or lg.ligand_smarts_key not in LIGAND_ETA_DEFAULTS:
            return None
        hap, name = LIGAND_ETA_DEFAULTS[lg.ligand_smarts_key]
        pi_names.append((lg.hapticity_n, name))

    # Style-aware ligand name overrides (PIN uses different names than systematic
    # for some ligands; per IUPAC P-69 + Salzer §5).
    if style == 'systematic':
        pi_names = [(hap, _SYSTEMATIC_LIGAND_NAME_OVERRIDES.get(name, name))
                    for hap, name in pi_names]

    # Group identical π-ligands (count by (hapticity, name) tuple)
    from collections import Counter
    pi_counter = Counter(pi_names)
    # Sort π-ligands alphabetically by name (ignoring multiplicative prefix)
    pi_sorted = sorted(pi_counter.items(), key=lambda x: x[0][1])

    # Determine if CO uses 'bis/tris/tetrakis'-style or simple multiplicative
    # Per Salzer: simple ligands use di/tri/tetra; complex ligands (with
    # parens/locants/η-prefix) use bis/tris/tetrakis.
    pi_prefix_parts: List[str] = []
    pi_tree_nodes: List[Any] = []
    for (hap, name), count in pi_sorted:
        eta = f"η{_superscript_int(hap)}-"
        ligand_str = f"({eta}{name})"
        if count == 1:
            pi_prefix_parts.append(ligand_str)
        else:
            mult = _COMPLEX_MULTIPLICATIVE_PREFIXES.get(count)
            if mult is None:
                return None
            pi_prefix_parts.append(f"{mult}{ligand_str}")
        pi_tree_nodes.append(NameTreeNode(
            parent_stem=f"{eta}{name}",
            multiplicative_prefix=_COMPLEX_MULTIPLICATIVE_PREFIXES.get(count) if count > 1 else None,
            class_id='organometallic_ligand',
            parenthesization_hint=True,
        ))

    # Carbonyl prefix
    co_prefix_str = ""
    co_tree_nodes: List[Any] = []
    if co_ligands:
        n_co = len(co_ligands)
        co_mult = _multiplicative_prefix(n_co)
        co_prefix_str = f"{co_mult}carbonyl"
        co_tree_nodes.append(NameTreeNode(
            parent_stem='carbonyl',
            multiplicative_prefix=co_mult or None,
            class_id='organometallic_ligand',
        ))

    # Stock notation: use metal's empirical formal charge as oxidation state.
    # For neutral metals (charge 0) bound to neutral π-ligands (benzene,
    # butadiene, ethene, COT), the metal is in 0 oxidation state.
    # For neutral metals bound to anionic ligands (Cp anion), the metal's
    # SMILES form would carry the corresponding positive charge already.
    # For cymantrene CpMn(CO)3 specifically: Mn is neutral in SMILES but the
    # complex has Mn(I); audit § 6 lock applies via METAL_OXIDATION_STATE_HINTS.
    # We use the empirical charge if non-zero; otherwise look up hints.
    hints = None
    # Pick a ligand_class hint key based on the ligand mix
    if len(pi_ligands) == 1 and pi_ligands[0].ligand_smarts_key == 'c1ccccc1' and len(co_ligands) == 3:
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'bz_CO3'))
    elif (len(pi_ligands) == 1 and pi_ligands[0].ligand_smarts_key == 'c1cc[cH-]c1'
          and len(co_ligands) == 3):
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'Cp_CO3'))
    elif (len(pi_ligands) == 2 and len(co_ligands) == 0
          and all(lg.ligand_smarts_key == 'c1ccccc1' for lg in pi_ligands)):
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'bz2'))
    elif (len(pi_ligands) == 1 and pi_ligands[0].ligand_smarts_key == 'C1=CC=CC=CC=1'
          and len(co_ligands) == 3):
        hints = METAL_OXIDATION_STATE_HINTS.get((metal_symbol, 'CHT_CO3'))

    if metal_charge != 0:
        effective_oxidation = metal_charge
    elif hints is not None:
        effective_oxidation = hints['default_state']
    else:
        effective_oxidation = 0  # neutral metal default

    # PIN: no Stock for these by default; systematic: include Stock
    include_stock = (style == 'systematic')
    if include_stock:
        try:
            stock_str = f"({_to_roman(effective_oxidation)})"
        except ValueError:
            stock_str = ""
    else:
        stock_str = ""

    # Compose full name: CO-prefix + π-ligands + metal + Stock
    full_name = f"{co_prefix_str}{''.join(pi_prefix_parts)}{metal_name}{stock_str}"
    metal_name_part = f"{metal_name}{stock_str}"
    return (full_name, metal_name_part, co_tree_nodes + pi_tree_nodes)


def _superscript_int(n: int) -> str:
    """Convert an integer to Unicode superscript digits."""
    digits = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '-': '⁻',
    }
    return ''.join(digits[c] for c in str(n))


# Complex-ligand multiplicative prefixes per Salzer §5.2 (used when ligand
# contains parens / locants / η-prefix).
_COMPLEX_MULTIPLICATIVE_PREFIXES: Dict[int, str] = {
    2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis',
    6: 'hexakis', 7: 'heptakis', 8: 'octakis',
}


# Style-aware ligand name overrides for systematic forms.
# PIN uses the strict IUPAC-2013 substitutive name; systematic uses the
# traditional Salzer 1999 / Red Book IR-10 organometallic-specific name.
_SYSTEMATIC_LIGAND_NAME_OVERRIDES: Dict[str, str] = {
    'prop-2-en-1-yl': 'allyl',  # T4-06 systematic: bis(η³-allyl)nickel(0)
}


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
