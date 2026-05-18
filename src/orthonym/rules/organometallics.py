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

    STUB BODY (Plan-02): returns None. Plan-03 implements per-tier
    branches (commits 03-01 Tier-1 → 03-02 Tier-3 → 03-03 Tier-2 → 03-04
    Tier-4) per RESEARCH §7.

    Returns: (full_name, metal_name_part, ligand_tree_nodes) per
    RESEARCH §4.3 lines 799-815, so the handler can build
    NameTreeNode(parent_stem=metal_name_part, prefixes=tuple(ligand_tree_nodes),
    iupac_section_cite='P-69 / IR-10 / Salzer 1999', class_id='organometallic').

    The byte-identical contract per CONTEXT D-11:
    name_tree_to_string(tree) == full_name.

    Returns None to signal cascade-continuation when:
    - metal_complex.is_multimetal (Phase 161.3 deferred per Risk R-08)
    - Hapticity computation raises ValueError (no contiguous π-system)
    - Stock-notation lookup fails for the (metal, ligand_class) tuple
    """
    return None


__all__ = [
    'ORGM_LIGAND_ORDER',
    'METAL_OXIDATION_STATE_HINTS',
    'METAL_RANKING_FOR_PARENT_SELECTION',
    'select_ligand_naming',
    'assemble_organometallic_name',
]
