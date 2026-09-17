"""
Retained names + lookup tables for IUPAC organometallic compounds.

This module provides lookup tables for organometallic compounds named per
IUPAC 2013 Blue Book § + IUPAC Red Book § + Salzer 1999 IUPAC
Recommendations (Pure Appl. Chem. 71(8) 1557).

Keys are RDKit canonical SMILES (verified per internal notes + RESEARCH
empirical measurement). All entries are pure data; no logic in this module.

a phase (first scope-expansion phase per -07).

Anti-patterns to avoid (PATTERNS lines 385-388):
- NEVER use raw SMARTS without `[p for p in [...] if p is not None]` filter.
- NEVER canonicalize a key at lookup time without ensuring the dict was built
  with the same Chem.MolToSmiles canonicalization.
- NEVER mutate the dicts at runtime (they are module-level constants).
"""

from typing import Dict, Optional, Tuple

from rdkit import Chem

# === RETAINED METALLOCENES (a phase Tier-1; AUDIT) ===
# Canonical SMILES → retained PIN. All keys verified RDKit-canonical per
# RESEARCH empirical measurement.
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
    # Tier 4 stretch — decamethylferrocene; key is the RDKit canonical
    # SMILES (Chem.MolToSmiles output of the AUDIT input SMILES),
    # which is what the lookup site (Chem.MolToSmiles(mol)) produces.
    'Cc1c(C)c(C)[c-](C)c1C.Cc1c(C)c(C)[c-](C)c1C.[Fe+2]': 'decamethylferrocene',
}


# === METAL NAMES (per IUPAC + Salzer 1999; AUDIT) ===
# Element symbol → naming-system dict per Risk R-06 mitigation
# (Group 14: hydride-parent system; Groups 1/2/12/13: metal-direct).
METAL_NAMES: Dict[str, Dict[str, Optional[str]]] = {
    # Group 1 alkali metals (metal-direct; oxidation state always +1)
    'Li': {'direct': 'lithium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Na': {'direct': 'sodium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'K':  {'direct': 'potassium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Rb': {'direct': 'rubidium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cs': {'direct': 'caesium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Fr': {'direct': 'francium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Group 2 alkaline earths (metal-direct; +2)
    'Be': {'direct': 'beryllium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Mg': {'direct': 'magnesium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ca': {'direct': 'calcium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Sr': {'direct': 'strontium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ba': {'direct': 'barium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ra': {'direct': 'radium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Group 13 (metal-direct except B which uses hydride-parent 'borane')
    'B':  {'direct': 'boron',     'hydride_parent': 'borane',    'naming_system': 'hydride_parent'},
    'Al': {'direct': 'aluminum',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ga': {'direct': 'gallium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'In': {'direct': 'indium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Tl': {'direct': 'thallium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Group 14 (hydride-parent per IUPAC
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
    # Task M1 (v51): Group 3 + lanthanides + actinides (metal-direct). These are
    # BEST-EFFORT σ-organometallic parents, no PIN derivation); every
    # `methyl<metal>` was verified to OPSIN-`-r` round-trip to `C[<sym>]`, and
    # any emission is round-trip-gated downstream, so a metal whose alkyl does
    # not round-trip simply abstains. Group 3 (Sc/Y/La/Ac):
    'Sc': {'direct': 'scandium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Y':  {'direct': 'yttrium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'La': {'direct': 'lanthanum',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ac': {'direct': 'actinium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Lanthanides (Ce..Lu):
    'Ce': {'direct': 'cerium',       'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Pr': {'direct': 'praseodymium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Nd': {'direct': 'neodymium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Pm': {'direct': 'promethium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Sm': {'direct': 'samarium',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Eu': {'direct': 'europium',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Gd': {'direct': 'gadolinium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Tb': {'direct': 'terbium',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Dy': {'direct': 'dysprosium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Ho': {'direct': 'holmium',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Er': {'direct': 'erbium',       'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Tm': {'direct': 'thulium',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Yb': {'direct': 'ytterbium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Lu': {'direct': 'lutetium',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    # Actinides (Th..Lr):
    'Th': {'direct': 'thorium',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Pa': {'direct': 'protactinium', 'hydride_parent': None, 'naming_system': 'metal_direct'},
    'U':  {'direct': 'uranium',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Np': {'direct': 'neptunium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Pu': {'direct': 'plutonium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Am': {'direct': 'americium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cm': {'direct': 'curium',       'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Bk': {'direct': 'berkelium',    'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Cf': {'direct': 'californium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Es': {'direct': 'einsteinium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Fm': {'direct': 'fermium',      'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Md': {'direct': 'mendelevium',  'hydride_parent': None, 'naming_system': 'metal_direct'},
    'No': {'direct': 'nobelium',     'hydride_parent': None, 'naming_system': 'metal_direct'},
    'Lr': {'direct': 'lawrencium',   'hydride_parent': None, 'naming_system': 'metal_direct'},
}


# === METAL_HYDRIDE_PARENT_NAMES (per IUPAC; AUDIT) ===
# Convenience flat view of METAL_NAMES restricted to hydride-parent system.
# Used by rules.organometallics for Group 14 substitutive naming.
METAL_HYDRIDE_PARENT_NAMES: Dict[str, str] = {
    sym: data['hydride_parent']  # type: ignore[misc]
    for sym, data in METAL_NAMES.items()
    if data.get('hydride_parent') is not None
}


# === METALLACYCLE_A_PREFIX; W8-P9 Task 9.5) ===
# Skeletal-replacement nondetachable 'a'-prefix for a Group 2-12 metal ring
# atom (BB verbatim, P6a.pdf): "selecting a parent hydrocarbon ring
#... and replacing one or more carbon atoms by a metal atom from Groups 2
# through 12 using a nondetachable skeletal replacement ('a') prefix".
# Table scoped to the metals with a worked BB example (platina/irida/titana)
# plus the common analogues by the standard 'a'-suffix pattern (element stem
# + 'a', matching the existing METAL_NAMES 'direct' stems). Deliberately
# excludes Groups 13-16 (Si/Ge/Sn/Pb/B/etc.) — those are -04's existing
# Hantzsch-Widman-only ring territory (silole/borole/stannole), a SEPARATE
# established path this table must not overlap.
METALLACYCLE_A_PREFIX: Dict[str, str] = {
    'Ti': 'titana', 'Zr': 'zirconia', 'Hf': 'hafnia',
    'V': 'vanadia', 'Nb': 'nioba', 'Ta': 'tantala',
    'Cr': 'chroma', 'Mo': 'molybda', 'W': 'tungsta',
    'Mn': 'mangana', 'Re': 'rhena',
    'Fe': 'ferra', 'Ru': 'ruthena', 'Os': 'osma',
    'Co': 'cobalta', 'Rh': 'rhoda', 'Ir': 'irida',
    'Ni': 'nickela', 'Pd': 'pallada', 'Pt': 'platina',
    'Cu': 'cupra', 'Ag': 'argenta', 'Au': 'aura',
    'Zn': 'zinca', 'Cd': 'cadmia', 'Hg': 'mercura',
}


# === LIGAND NAMES (SMARTS-canonical-key → IUPAC ligand name; AUDIT) ===
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
    # 7-atom ring cycloheptatrienyl + 8-atom ring cyclooctatetraene
    # — Plan-02 substrate previously mislabeled the 8-atom ring as
    # cycloheptatrienyl. Amendment per AUDIT ORG-T4-08/09.
    'C1=CC=CC=CC=1':     'cycloheptatrienyl',
    'C1=CC=CC=CC=C1':    'cyclooctatetraene',
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


# === LIGAND ETA DEFAULTS (per RESEARCH; AUDIT) ===
# SMARTS-canonical-key → (default_hapticity, ligand_name) per Salzer 1999.
LIGAND_ETA_DEFAULTS: Dict[str, Tuple[int, str]] = {
    'c1cc[cH-]c1':       (5, 'cyclopentadienyl'),
    '[c-]1cccc1':        (5, 'cyclopentadienyl'),
    # pentamethyl-Cp: key is RDKit canonical SMILES per Plan-03-04 amendment.
    'Cc1c(C)c(C)[c-](C)c1C': (5, 'pentamethylcyclopentadienyl'),
    # mono-methyl-Cp: per AUDIT ORG-T4-25 amendment.
    'C[c-]1cccc1':       (5, 'methylcyclopentadienyl'),
    'c1ccccc1':          (6, 'benzene'),
    'C=CC=C':            (4, '1,3-butadiene'),
    'C=CC':              (3, 'prop-2-en-1-yl'),
    '[CH2]=CC':          (3, 'allyl'),
    'C=C':               (2, 'ethene'),
    'C#C':               (2, 'ethyne'),
    # 7-atom ring cycloheptatrienyl (C1=CC=CC=CC=1; 7 atoms total)
    'C1=CC=CC=CC=1':     (7, 'cycloheptatrienyl'),
    # 8-atom ring cyclooctatetraene (COT; C1=CC=CC=CC=C1; 8 atoms total)
    # — Plan-02 substrate previously mislabeled this as 'cycloheptatrienyl'.
    # Amendment per AUDIT ORG-T4-08/09.
    'C1=CC=CC=CC=C1':    (8, 'cyclooctatetraene'),
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
    'METALLACYCLE_A_PREFIX',
    'get_retained_metallocene_name',
]
