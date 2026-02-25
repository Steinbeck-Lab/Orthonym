"""Natural product scaffold data and derivative lookup tables.

Provides pre-canonicalized SMILES for natural product scaffolds (for
substructure matching) and exact derivative names (for direct lookup).
All SMILES keys are already in RDKit canonical form -- no runtime
canonicalization is needed.

Also provides IUPAC atom numbering maps for steroid scaffolds, enabling
decoration enumeration (hydroxy, ketone, unsaturation) on scaffold atoms.
"""

from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Parent scaffolds for substructure matching
# ---------------------------------------------------------------------------
# Each key is a pre-canonicalized SMILES string.
# Each value is a dict with 'name', 'stem', and 'class' keys.

NATURAL_PRODUCT_SCAFFOLDS = {
    # ---- Steroids (9 scaffolds) ----
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2": {
        "name": "androstane", "stem": "androst", "class": "steroid",
    },
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2": {
        "name": "estrane", "stem": "estr", "class": "steroid",
    },
    "CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        "name": "pregnane", "stem": "pregn", "class": "steroid",
    },
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        "name": "cholestane", "stem": "cholest", "class": "steroid",
    },
    "CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        "name": "cholane", "stem": "chol", "class": "steroid",
    },
    "CC(C)[C@@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        "name": "ergostane", "stem": "ergost", "class": "steroid",
    },
    "CC(C)[C@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        "name": "campestane", "stem": "campest", "class": "steroid",
    },
    "CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C": {
        "name": "stigmastane", "stem": "stigmast", "class": "steroid",
    },
    "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12": {
        "name": "gonane", "stem": "gon", "class": "steroid",
    },

    # ---- Alkaloids (5 scaffolds) ----
    "c1ccc2c(c1)C[C@H]1NCC[C@@]23CCCC[C@@H]13": {
        "name": "morphinan", "stem": "morphin", "class": "alkaloid",
    },
    "CN1[C@@H]2CCC[C@H]1CC2": {
        "name": "tropane", "stem": "trop", "class": "alkaloid",
    },
    "C=C[C@H]1C[N@@]2CC[C@H]1C[C@@H]2Cc1ccnc2ccccc12": {
        "name": "cinchonane", "stem": "cinchon", "class": "alkaloid",
    },
    "CN1CCc2cccc3c2C1Cc1ccccc1-3": {
        "name": "aporphine", "stem": "aporphin", "class": "alkaloid",
    },
    "c1cc2c3c(c[nH]c3c1)C[C@H]1NCCC[C@H]21": {
        "name": "ergoline", "stem": "ergolin", "class": "alkaloid",
    },
}


# ---------------------------------------------------------------------------
# 2. Exact derivatives -- direct SMILES-to-name lookup
# ---------------------------------------------------------------------------
# Key = pre-canonicalized SMILES, value = retained/trivial name.

NATURAL_PRODUCT_DERIVATIVES = {
    # ---- Steroid derivatives ----
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C":
        "cholesterol",

    # ---- Opioid derivatives ----
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5":
        "morphine",
    "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341":
        "codeine",
    "CC(=O)Oc1ccc2c3c1O[C@H]1[C@@H](OC(C)=O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341":
        "diamorphine",
    "COc1ccc2c3c1O[C@H]1C(=O)CC[C@H]4[C@@H](C2)N(C)CC[C@]314":
        "hydrocodone",
    "COc1ccc2c3c1O[C@H]1C(=O)CC[C@@]4(O)[C@@H](C2)N(C)CC[C@]314":
        "oxycodone",
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@H]3[C@H]1C5":
        "hydromorphone",

    # ---- Terpenoid derivatives ----
    "CC12CCC(CC1=O)C2(C)C":
        "camphor",
    "C=C(C)C1CC=C(C)CC1":
        "limonene",
    "CC1=CCC2CC1C2(C)C":
        "alpha-pinene",
    "CC1(C)C2=CCC1CC2":
        "beta-pinene",
    "CC1=CCC(C(C)(C)O)CC1":
        "alpha-terpineol",
    "C=C(C)C1CCC(C)(O)CC1":
        "beta-terpineol",
    "C=CCC(O)CC=C(C)C":
        "gamma-terpineol",

    # ---- Carotenoid derivatives ----
    "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)CCC2(C)C)C(C)(C)CCC1":
        "beta-carotene",
    "CC(C)=CC=CC(C)=CC=C/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)C":
        "lycopene",

    # ---- Beta-lactam scaffolds ----
    "O=C(O)C1CSC2CC(=O)N21":
        "penam",
    "O=C(O)C1CSCC2CC(=O)N21":
        "cepham",
}


# ---------------------------------------------------------------------------
# 3. Helper function for derivative lookup
# ---------------------------------------------------------------------------

def get_natural_product_name(canonical_smiles: str) -> Optional[str]:
    """Look up exact natural product derivative name by canonical SMILES.

    Args:
        canonical_smiles: RDKit canonical SMILES string.

    Returns:
        The retained/trivial name if found, otherwise None.
    """
    return NATURAL_PRODUCT_DERIVATIVES.get(canonical_smiles)


# ---------------------------------------------------------------------------
# 4. Pre-compiled RDKit Mol objects for substructure matching
# ---------------------------------------------------------------------------

_SCAFFOLD_PATTERNS: dict = {}


def _init_patterns() -> None:
    """Compile scaffold SMILES into RDKit Mol objects at import time."""
    from rdkit import Chem

    for smiles in NATURAL_PRODUCT_SCAFFOLDS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            _SCAFFOLD_PATTERNS[smiles] = mol


_init_patterns()


def get_scaffold_patterns() -> dict:
    """Return pre-compiled RDKit Mol objects keyed by canonical SMILES.

    Returns:
        Dict mapping canonical SMILES to compiled RDKit Mol objects.
    """
    return _SCAFFOLD_PATTERNS


# ---------------------------------------------------------------------------
# 5. IUPAC atom numbering maps for steroid scaffolds
# ---------------------------------------------------------------------------
# Each map: query atom index (position in scaffold SMILES) -> IUPAC locant.
# When a molecule is matched against a scaffold, matched_atoms[query_pos] gives
# the target molecule's atom index. Combined with the numbering map, this gives:
#   iupac_locant = NUMBERING_MAP[query_pos]
#   target_atom  = matched_atoms[query_pos]
#
# Numbering follows IUPAC 2013 steroid conventions:
#   Ring A: 1-5,10    Ring B: 5-10    Ring C: 8,9,11-14    Ring D: 13-17
#   C-18: angular methyl on C-13
#   C-19: angular methyl on C-10
#   C-20+: side chain

STEROID_NUMBERING_MAPS: Dict[str, Dict[int, int]] = {
    # Gonane (17 carbons, no angular methyls)
    "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12": {
        0: 3, 1: 2, 2: 1, 3: 10, 4: 5, 5: 4,
        6: 6, 7: 7, 8: 8, 9: 14, 10: 15,
        11: 16, 12: 17, 13: 13, 14: 12, 15: 11, 16: 9,
    },

    # Androstane (19 carbons: gonane + C-18, C-19 angular methyls)
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2": {
        0: 18, 1: 13, 2: 17, 3: 16, 4: 15, 5: 14,
        6: 8, 7: 7, 8: 6, 9: 5,
        10: 4, 11: 3, 12: 2, 13: 1,
        14: 10, 15: 19, 16: 9, 17: 11, 18: 12,
    },

    # Estrane (18 carbons: gonane + C-18, no C-19)
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2": {
        0: 18, 1: 13, 2: 17, 3: 16, 4: 15, 5: 14,
        6: 8, 7: 7, 8: 6, 9: 5,
        10: 4, 11: 3, 12: 2, 13: 1,
        14: 10, 15: 9, 16: 12, 17: 11,
    },

    # Pregnane (21 carbons: androstane + C-20, C-21 side chain)
    "CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        0: 21, 1: 20, 2: 17, 3: 16, 4: 15, 5: 14,
        6: 8, 7: 7, 8: 6, 9: 5,
        10: 4, 11: 3, 12: 2, 13: 1,
        14: 10, 15: 19, 16: 9, 17: 11, 18: 12,
        19: 13, 20: 18,
    },

    # Cholane (24 carbons: C-20 to C-24 side chain)
    "CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        0: 24, 1: 23, 2: 22, 3: 20, 4: 21, 5: 17,
        6: 16, 7: 15, 8: 14, 9: 8,
        10: 7, 11: 6, 12: 5, 13: 4, 14: 3, 15: 2, 16: 1,
        17: 10, 18: 19, 19: 9, 20: 12, 21: 11,
        22: 13, 23: 18,
    },

    # Cholestane (27 carbons: C-20 to C-27 side chain)
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        0: 26, 1: 25, 2: 27, 3: 24, 4: 23, 5: 22,
        6: 20, 7: 21, 8: 17, 9: 16, 10: 15, 11: 14,
        12: 8, 13: 7, 14: 6, 15: 5, 16: 4, 17: 3,
        18: 2, 19: 1, 20: 10, 21: 19, 22: 9, 23: 11,
        24: 12, 25: 13, 26: 18,
    },

    # Ergostane (28 carbons: cholestane + extra C-28 methyl at C-24)
    "CC(C)[C@@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        0: 26, 1: 25, 2: 27, 3: 24, 4: 28, 5: 23, 6: 22,
        7: 20, 8: 21, 9: 17, 10: 16, 11: 15, 12: 14,
        13: 8, 14: 7, 15: 6, 16: 5,
        17: 4, 18: 3, 19: 2, 20: 1,
        21: 10, 22: 19, 23: 9, 24: 12, 25: 11,
        26: 13, 27: 18,
    },

    # Campestane (28 carbons: same topology as ergostane, different stereo at C-24)
    "CC(C)[C@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C": {
        0: 26, 1: 25, 2: 27, 3: 24, 4: 28, 5: 23, 6: 22,
        7: 20, 8: 21, 9: 17, 10: 16, 11: 15, 12: 14,
        13: 8, 14: 7, 15: 6, 16: 5,
        17: 4, 18: 3, 19: 2, 20: 1,
        21: 10, 22: 19, 23: 9, 24: 12, 25: 11,
        26: 13, 27: 18,
    },

    # Stigmastane (29 carbons: cholestane + ethyl at C-24)
    "CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C": {
        0: 29, 1: 28, 2: 24, 3: 23, 4: 22, 5: 20, 6: 21,
        7: 17, 8: 16, 9: 15, 10: 14,
        11: 8, 12: 7, 13: 6, 14: 5,
        15: 4, 16: 3, 17: 2, 18: 1,
        19: 10, 20: 19, 21: 9, 22: 12, 23: 11,
        24: 13, 25: 18,
        26: 25, 27: 26, 28: 27,
    },
}


# ---------------------------------------------------------------------------
# 4. Alkaloid numbering maps  (query atom position → IUPAC locant)
# ---------------------------------------------------------------------------
# Same principle as STEROID_NUMBERING_MAPS: the key is the canonical SMILES
# of the scaffold, and the value maps each query atom index (from
# GetSubstructMatch) to the traditional IUPAC locant number.

ALKALOID_NUMBERING_MAPS: Dict[str, Dict[int, int]] = {
    # Morphinan (16C + 1N = 17 atoms, positions 1-17)
    # Ring A (aromatic): {1, 2, 3, 4, 11, 12}
    # Ring B: {5, 6, 7, 8, 13, 14}
    # Ring C: {9, 10, 11, 12, 13, 14}
    # Ring D (piperidine): {9, 13, 14, 15, 16, 17(N)}
    "c1ccc2c(c1)C[C@H]1NCC[C@@]23CCCC[C@@H]13": {
        0: 2,    # C, Ring A aromatic
        1: 3,    # C, Ring A aromatic (phenolic OH site in morphine)
        2: 4,    # C, Ring A aromatic (epoxy bridge site)
        3: 12,   # C, Ring A/C junction (aromatic)
        4: 11,   # C, Ring A/C junction (aromatic)
        5: 1,    # C, Ring A aromatic
        6: 10,   # C, Ring C (CH2 bridge)
        7: 9,    # C, Ring C/D junction
        8: 17,   # N, Ring D nitrogen
        9: 16,   # C, Ring D
        10: 15,  # C, Ring D
        11: 13,  # C, Ring B/C/D tri-junction
        12: 5,   # C, Ring B (epoxy bridge site)
        13: 6,   # C, Ring B (hydroxyl site in morphine)
        14: 7,   # C, Ring B (unsaturation site)
        15: 8,   # C, Ring B (unsaturation site)
        16: 14,  # C, Ring B/C/D tri-junction
    },
}


def get_steroid_numbering(scaffold_smiles: str) -> Optional[Dict[int, int]]:
    """Get IUPAC numbering map for a steroid scaffold.

    Args:
        scaffold_smiles: Canonical SMILES of the scaffold.

    Returns:
        Dict mapping query atom index to IUPAC locant, or None if not found.
    """
    return STEROID_NUMBERING_MAPS.get(scaffold_smiles)


def get_scaffold_numbering(scaffold_smiles: str) -> Optional[Dict[int, int]]:
    """Get IUPAC numbering map for any scaffold (steroid or alkaloid).

    Checks steroid maps first, then alkaloid maps.

    Args:
        scaffold_smiles: Canonical SMILES of the scaffold.

    Returns:
        Dict mapping query atom index to IUPAC locant, or None if not found.
    """
    result = STEROID_NUMBERING_MAPS.get(scaffold_smiles)
    if result is not None:
        return result
    return ALKALOID_NUMBERING_MAPS.get(scaffold_smiles)
