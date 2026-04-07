"""
OPSIN XML Data Imports - consolidated exports.

Auto-generated data modules from OPSIN XML resource files.
See  for the generation pipeline.

Main export: OPSIN_RETAINED_NAMES -- merged SMILES-to-name lookup
from aryl_groups, simple_groups, cyclic_groups, and natural_products.

Individual module exports available for specialized use.
"""

from typing import Dict, List, Optional

# Import all generated module dicts
from .aryl_groups import OPSIN_ARYL_GROUPS
from .simple_groups import OPSIN_SIMPLE_GROUPS
from .carboxylic_acids_opsin import OPSIN_ACID_STEMS
from .substituent_names_opsin import OPSIN_SUBSTITUENT_NAMES
from .amino_acids_opsin import OPSIN_AMINO_ACIDS
from .carbohydrates_opsin import OPSIN_CARBOHYDRATES, OPSIN_CARBOHYDRATE_SUFFIXES
from .natural_products_opsin import OPSIN_NATURAL_PRODUCTS
from .cyclic_groups import OPSIN_CYCLIC_GROUPS
from .suffix_rules import OPSIN_SUFFIX_RULES, OPSIN_SUFFIX_APPLICABILITY
from .word_rules import OPSIN_WORD_RULES
from .functional_terms import OPSIN_FUNCTIONAL_TERMS
from .fusion_components_opsin import OPSIN_FUSION_COMPONENTS


def _build_retained_names() -> Dict[str, str]:
    """Build consolidated SMILES-to-name lookup from multiple OPSIN sources.

    Merges aryl_groups, simple_groups, cyclic_groups, natural_products,
    amino_acids, and carbohydrates into a single dict.

    For entries with composite keys (containing '||' separator for
    structural variants), the base SMILES is extracted.

    Returns:
        Dict mapping canonical SMILES to the first (primary) name variant.
    """
    merged: Dict[str, str] = {}

    # Sources for retained name lookup, in precedence order
    sources = [
        OPSIN_ARYL_GROUPS,
        OPSIN_SIMPLE_GROUPS,
        OPSIN_CYCLIC_GROUPS,
        OPSIN_NATURAL_PRODUCTS,
        OPSIN_AMINO_ACIDS,
        OPSIN_CARBOHYDRATES,
    ]

    for source in sources:
        for key, meta in source.items():
            # Extract base SMILES from composite keys
            smiles = meta.get("smiles", key.split("||")[0] if "||" in key else key)
            names = meta.get("names", [])
            if names and smiles not in merged:
                # Use first name variant as the primary name
                merged[smiles] = names[0]

    return merged


# Consolidated SMILES -> name lookup (D-04)
OPSIN_RETAINED_NAMES: Dict[str, str] = _build_retained_names()


__all__ = [
    "OPSIN_RETAINED_NAMES",
    "OPSIN_ARYL_GROUPS",
    "OPSIN_SIMPLE_GROUPS",
    "OPSIN_ACID_STEMS",
    "OPSIN_SUBSTITUENT_NAMES",
    "OPSIN_AMINO_ACIDS",
    "OPSIN_CARBOHYDRATES",
    "OPSIN_CARBOHYDRATE_SUFFIXES",
    "OPSIN_NATURAL_PRODUCTS",
    "OPSIN_CYCLIC_GROUPS",
    "OPSIN_SUFFIX_RULES",
    "OPSIN_SUFFIX_APPLICABILITY",
    "OPSIN_WORD_RULES",
    "OPSIN_FUNCTIONAL_TERMS",
    "OPSIN_FUSION_COMPONENTS",
]
