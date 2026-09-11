# Derived from OPSIN (Open Parser for Systematic IUPAC Nomenclature),
# MIT License, Copyright (c) Daniel Lowe and contributors.
# Source project: https://github.com/dan2097/opsin — see the repository NOTICE file.
"""
OPSIN XML Data Imports - consolidated exports.

Auto-generated data modules from OPSIN XML resource files.
See scripts/import_opsin_xml.py for the generation pipeline.

Main export: OPSIN_RETAINED_NAMES -- merged SMILES-to-name lookup
from aryl_groups, simple_groups, cyclic_groups, and natural_products.

Individual module exports available for specialized use.
"""

from typing import Dict, List, Optional

from .amino_acids_opsin import OPSIN_AMINO_ACIDS

# Import all generated module dicts
from .aryl_groups import OPSIN_ARYL_GROUPS
from .carbohydrates_opsin import OPSIN_CARBOHYDRATE_SUFFIXES, OPSIN_CARBOHYDRATES
from .carboxylic_acids_opsin import OPSIN_ACID_STEMS
from .cyclic_groups import OPSIN_CYCLIC_GROUPS
from .functional_terms import OPSIN_FUNCTIONAL_TERMS
from .fusion_components_opsin import OPSIN_FUSION_COMPONENTS
from .natural_products_opsin import OPSIN_NATURAL_PRODUCTS
from .simple_groups import OPSIN_SIMPLE_GROUPS
from .substituent_names_opsin import OPSIN_SUBSTITUENT_NAMES
from .suffix_rules import OPSIN_SUFFIX_APPLICABILITY, OPSIN_SUFFIX_RULES
from .word_rules import OPSIN_WORD_RULES


def _select_primary_name(names):
    """Per RESEARCH Pitfall 5 + ADDITION 3: prefer longest space-containing name.

    OPSIN stores names in parser-preference order (shortest/most-common
    first), NOT IUPAC-output order. Token like CC(=O)O>acetic|acetic acid|
    aceticacid has 'acetic' first; we want 'acetic acid' as the IUPAC
    output.

    Source: internal notes Pitfall 5 + ADDITION 3.
    """
    spaced = [n for n in names if " " in n]
    if spaced:
        return max(spaced, key=len)
    return names[0]


def _build_retained_names() -> Dict[str, str]:
    """Build consolidated SMILES-to-name lookup from 4 OPSIN sources.

    a phase: broadened from 2 sources (cyclic + NP) to 4
    (adds aryl + simple). Stem-vs-PIN classification + round-trip gate
    happens DOWNSTREAM in data/__init__.py:_is_promotable (a phase's
    3-signal AND classifier).

    Source: 150-internal notes.
    Source: internal notes section 4.3.
    """
    merged: Dict[str, str] = {}

    # 4-source merge per internal notes (a phase broadens from 2 to 4
    # sources). Cyclic + NP first (higher-quality data); aryl + simple
    # appended.
    sources = [
        OPSIN_CYCLIC_GROUPS,
        OPSIN_NATURAL_PRODUCTS,
        OPSIN_ARYL_GROUPS,           # NEW per
        OPSIN_SIMPLE_GROUPS,         # NEW per
    ]

    for source in sources:
        for key, meta in source.items():
            # RESEARCH Pitfall 3 + R5: skip saltComponent / chalcogenide
            # entries. These are not retained-name PINs in the parent-name
            # sense.
            if meta.get("subType") in ("saltComponent", "chalcogenide"):
                continue
            # a phase REVIEW root-cause fix: skip composite-key
            # entries. The "||" discriminator records OPSIN parser-side
            # transformations (addGroup / addBond / addHeteroAtom) that
            # would need the full OPSIN parser to materialise the actual
            # canonical SMILES of the modified molecule. Without that, the
            # base SMILES extracted from key.split("||")[0] does NOT
            # represent the molecular structure the discriminated name
            # refers to (e.g., 'acenaphthoquinone' is keyed on the
            # acenaphthylene SMILES + '=O locant 1;=O locant 2'; the base
            # SMILES alone is acenaphthylene). Storing the discriminated
            # name against the base SMILES would mis-attribute the name
            # to the wrong structure. Skipping is the conservative
            # data-source-correct choice; 26 such entries (21 aryl + 2
            # simple + 3 cyclic + 0 NP) are excluded. Aryl-side rendering
            # of acenaphthoquinone et al. falls back to systematic naming
            # until a future phase implements OPSIN-style addGroup
            # materialisation. Source: internal notes +
            # the contributor guide fix-methodology.md (root-cause-only fixes).
            if "||" in key:
                continue
            smiles = meta.get("smiles", key)
            names = meta.get("names", [])
            if names and smiles not in merged:
                merged[smiles] = _select_primary_name(names)

    return merged


# Consolidated SMILES -> name lookup
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
