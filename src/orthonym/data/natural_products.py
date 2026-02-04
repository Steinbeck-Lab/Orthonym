"""Natural product scaffold data and derivative lookup tables.

Provides pre-canonicalized SMILES for natural product scaffolds (for
substructure matching) and exact derivative names (for direct lookup).
All SMILES keys are already in RDKit canonical form -- no runtime
canonicalization is needed.
"""

from typing import Optional


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
