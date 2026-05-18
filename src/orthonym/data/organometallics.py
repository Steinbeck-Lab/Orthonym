"""
Retained names + lookup tables for IUPAC P-69 organometallic compounds.

This module provides lookup tables for organometallic compounds named per
IUPAC 2013 Blue Book §P-69 + IUPAC Red Book §IR-10 + Salzer 1999 IUPAC
Recommendations (Pure Appl. Chem. 71(8) 1557).

Keys are RDKit canonical SMILES (verified per CONTEXT line 18 + RESEARCH §1.7
empirical measurement). All entries are pure data; no logic in this module.

Phase 161 (v19 first scope-expansion phase per ADR-19-07).

Anti-patterns to avoid (PATTERNS lines 385-388):
- NEVER use raw SMARTS without `[p for p in [...] if p is not None]` filter.
- NEVER canonicalize a key at lookup time without ensuring the dict was built
  with the same Chem.MolToSmiles canonicalization.
- NEVER mutate the dicts at runtime (they are module-level constants).
"""

from typing import Dict, Optional, Tuple
from rdkit import Chem


# === RETAINED METALLOCENES (Phase 161 D-01 Tier-1; AUDIT § 1) ===
# Canonical SMILES → retained PIN. All keys verified RDKit-canonical per
# RESEARCH §2.2 empirical measurement.
RETAINED_METALLOCENES: Dict[str, str] = {
    '[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1': 'ferrocene',          # ORG-T1-01
    '[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1': 'ruthenocene',        # ORG-T1-02
    '[Os+2].c1cc[cH-]c1.c1cc[cH-]c1': 'osmocene',           # ORG-T1-03
    '[Co+2].c1cc[cH-]c1.c1cc[cH-]c1': 'cobaltocene',        # ORG-T1-04
    '[Ni+2].c1cc[cH-]c1.c1cc[cH-]c1': 'nickelocene',        # ORG-T1-05
    '[Cr+2].c1cc[cH-]c1.c1cc[cH-]c1': 'chromocene',         # ORG-T1-06
    '[V+2].c1cc[cH-]c1.c1cc[cH-]c1':  'vanadocene',         # ORG-T1-07
    '[Mn+2].c1cc[cH-]c1.c1cc[cH-]c1': 'manganocene',        # ORG-T1-08
    '[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1': 'ferrocenium',        # ORG-T1-09
    # Tier 4 stretch — decamethylferrocene
    # canonical form per RESEARCH §1.7 ORG-T4-10
    '[Fe+2].CC1=C(C)C(C)=C(C)[C-]1C.CC1=C(C)C(C)=C(C)[C-]1C': 'decamethylferrocene',
}


# === METAL NAMES (per IUPAC P-69 + Salzer 1999; AUDIT § 6) ===
# Element symbol → naming-system dict per Risk R-06 mitigation
# (Group 14: hydride-parent system; Groups 1/2/12/13: metal-direct).
METAL_NAMES: Dict[str, Dict[str, Optional[str]]] = {
    # Group 1 alkali metals (metal-direct; oxidation state always +1)
    'Li': {'direct': 'lithium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Na': {'direct': 'sodium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'K':  {'direct': 'potassium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Rb': {'direct': 'rubidium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cs': {'direct': 'caesium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Group 2 alkaline earths (metal-direct; +2)
    'Be': {'direct': 'beryllium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Mg': {'direct': 'magnesium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ca': {'direct': 'calcium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Group 13 (metal-direct except B which uses hydride-parent 'borane')
    'B':  {'direct': 'boron',     'hydride_parent': 'borane',    'naming_system': 'hydride_parent'},
    'Al': {'direct': 'aluminum',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ga': {'direct': 'gallium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'In': {'direct': 'indium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Tl': {'direct': 'thallium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Group 14 (hydride-parent per IUPAC P-69.2)
    'Si': {'direct': 'silicon',   'hydride_parent': 'silane',    'naming_system': 'hydride_parent'},
    'Ge': {'direct': 'germanium', 'hydride_parent': 'germane',   'naming_system': 'hydride_parent'},
    'Sn': {'direct': 'tin',       'hydride_parent': 'stannane',  'naming_system': 'hydride_parent'},
    'Pb': {'direct': 'lead',      'hydride_parent': 'plumbane',  'naming_system': 'hydride_parent'},
    # Group 12 (metal-direct; +2)
    'Zn': {'direct': 'zinc',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cd': {'direct': 'cadmium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Hg': {'direct': 'mercury',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Transition metals (variable oxidation state; Stock notation in systematic)
    'V':  {'direct': 'vanadium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cr': {'direct': 'chromium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Mn': {'direct': 'manganese',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Fe': {'direct': 'iron',       'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Co': {'direct': 'cobalt',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ni': {'direct': 'nickel',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cu': {'direct': 'copper',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Mo': {'direct': 'molybdenum', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'W':  {'direct': 'tungsten',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ru': {'direct': 'ruthenium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Os': {'direct': 'osmium',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Rh': {'direct': 'rhodium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ir': {'direct': 'iridium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Pd': {'direct': 'palladium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Pt': {'direct': 'platinum',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Re': {'direct': 'rhenium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ti': {'direct': 'titanium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Zr': {'direct': 'zirconium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Hf': {'direct': 'hafnium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
}


# === METAL_HYDRIDE_PARENT_NAMES (per IUPAC P-69.2; AUDIT § 6) ===
# Convenience flat view of METAL_NAMES restricted to hydride-parent system.
# Used by rules.organometallics for Group 14 substitutive naming.
METAL_HYDRIDE_PARENT_NAMES: Dict[str, str] = {
    sym: data['hydride_parent']  # type: ignore[misc]
    for sym, data in METAL_NAMES.items()
    if data.get('hydride_parent') is not None
}


# === LIGAND NAMES (SMARTS-canonical-key → IUPAC ligand name; AUDIT § 2) ===
# Used for naming the prefix part of organometallic names. Per CBC ligand
# classification (LibreTexts 13.02).
LIGAND_NAMES: Dict[str, str] = {
    '[C-]#[O+]':         'carbonyl',
    'c1cc[cH-]c1':       'cyclopentadienyl',
    '[c-]1cccc1':        'cyclopentadienyl',
    'CC1=C(C)C(C)=C(C)[C-]1C': 'pentamethylcyclopentadienyl',
    'c1ccccc1':          'benzene',
    'C=CC=C':            '1,3-butadiene',
    'C=CC':              'prop-2-en-1-yl',
    '[CH2]=CC':          'allyl',
    'C=C':               'ethene',
    'C#C':               'ethyne',
    'C1=CC=CC=CC=C1':    'cycloheptatrienyl',
    '[H-]':              'hydrido',
    '[Cl-]':             'chlorido',
    '[Br-]':             'bromido',
    '[F-]':              'fluorido',
    '[I-]':              'iodido',
    '[CH3]':             'methyl',
    '[CH2][CH3]':        'ethyl',
    '[CH2][CH2][CH2][CH3]': 'butyl',
    'c1ccc[c-]c1':       'phenyl',
}


# === LIGAND ETA DEFAULTS (per RESEARCH §3.3; AUDIT § 2) ===
# SMARTS-canonical-key → (default_hapticity, ligand_name) per Salzer 1999.
LIGAND_ETA_DEFAULTS: Dict[str, Tuple[int, str]] = {
    'c1cc[cH-]c1':       (5, 'cyclopentadienyl'),
    '[c-]1cccc1':        (5, 'cyclopentadienyl'),
    'CC1=C(C)C(C)=C(C)[C-]1C': (5, 'pentamethylcyclopentadienyl'),
    'c1ccccc1':          (6, 'benzene'),
    'C=CC=C':            (4, '1,3-butadiene'),
    'C=CC':              (3, 'prop-2-en-1-yl'),
    '[CH2]=CC':          (3, 'allyl'),
    'C=C':               (2, 'ethene'),
    'C#C':               (2, 'ethyne'),
    'C1=CC=CC=CC=C1':    (7, 'cycloheptatrienyl'),
    '[c+]1cccccc1':      (7, 'tropylium'),
    '[C-]#[O+]':         (1, 'carbonyl'),
    'C#O':               (1, 'carbonyl'),
    '[H-]':              (1, 'hydrido'),
    '[Cl-]':             (1, 'chlorido'),
    '[Br-]':             (1, 'bromido'),
    '[F-]':              (1, 'fluorido'),
    '[I-]':              (1, 'iodido'),
}


def get_retained_metallocene_name(smiles: str) -> Optional[str]:
    """Look up retained metallocene PIN by canonical SMILES.

    Per PATTERNS lines 346-381 + ion_retained_names.py:207-241 analog:
    canonicalizes input via Chem.MolToSmiles before dict lookup.

    Example:
        >>> get_retained_metallocene_name('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        'ferrocene'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    return RETAINED_METALLOCENES.get(canonical)


__all__ = [
    'RETAINED_METALLOCENES',
    'METAL_NAMES',
    'METAL_HYDRIDE_PARENT_NAMES',
    'LIGAND_NAMES',
    'LIGAND_ETA_DEFAULTS',
    'get_retained_metallocene_name',
]
