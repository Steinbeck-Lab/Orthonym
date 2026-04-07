"""
Data module - Lookup tables and reference data.

Contains hand-curated data (primary) merged with auto-imported OPSIN data.
Hand-curated entries always take precedence over OPSIN imports on conflict.
"""

import logging
from typing import Optional, Dict

from .retained_names import RETAINED_NAMES as _HAND_CURATED_NAMES
from .retained_names import get_retained_name as _hand_curated_get

logger = logging.getLogger(__name__)

# Try to import OPSIN data (may not exist if import script hasn't run)
try:
    from .opsin_imports import OPSIN_RETAINED_NAMES as _OPSIN_NAMES_RAW
except ImportError:
    _OPSIN_NAMES_RAW = {}
    logger.debug("OPSIN imports not available (run )")


# --- Stem filtering ---
# OPSIN stores many names as stems (e.g., "norbornen" instead of "norbornene",
# "norborn" instead of "norbornane"). These stems are OPSIN parser internals
# meant to have suffixes appended. Using them directly as retained names causes
# wrong output. Filter to only include names that look like complete words.

# Suffixes that indicate a complete chemical name
_COMPLETE_NAME_ENDINGS = (
    # Hydrocarbons
    "ane", "ene", "yne", "ylene", "adiene", "atriene",
    # Heterocycles and rings (complete forms)
    "ole", "ine", "oline", "azine", "azole",
    "idine", "idine",
    "furan", "pyran", "thiophene", "pyrrole", "pyridine",
    "oxetane", "azetidine", "thiirane", "oxirane", "aziridine",
    # Saturated heterocycles (complete -an forms that are NOT stems)
    "acridan", "indan", "thiinan", "dithian",
    # Functional groups
    "ol", "al", "one", "ide", "ate", "ite",
    "amine", "amide", "nitrile", "oxide", "ether",
    # Cations / ions
    "ium", "ion",
    # Acids
    "acid",
    # Sugars
    "ose",
    # Steroids / NPs
    "sterol", "sterone", "stenone", "stadiene",
    # Natural products with complete names
    "curcumin", "capsaicin", "caffeine", "morphine", "codeine",
    "nicotine", "quinine", "strychnine", "colchicine",
    "resveratrol", "melatonin", "serotonin", "dopamine",
    # Generic complete name endings
    "iol", "diol", "triol",
    # Aromatic retained names
    "benzene", "toluene", "xylene", "styrene",
    "naphthalene", "anthracene", "phenanthrene",
    "indole", "purine", "pteridine", "quinoline", "isoquinoline",
    "catechol", "resorcinol", "hydroquinone",
    # Amino acids
    "glycine", "alanine", "valine", "leucine",
    "proline", "serine", "threonine", "cysteine", "tyrosine",
    "histidine", "tryptophan", "phenylalanine",
    # Nucleobases
    "guanine", "adenine", "cytosine", "thymine", "uracil",
    # Steroids
    "cholesterol", "cortisone", "testosterone", "estradiol",
    # Multi-word endings
    "anhydride",
    # Phenols and specific compound class endings
    "phenol",
    # Thiin family (complete names, not stems)
    "thiin", "oxathiin", "dithiin",
    # Specific complete retained names that don't fit patterns
    "urea", "thiourea",
)


def _is_complete_name(name: str) -> bool:
    """Check if an OPSIN name is a complete chemical name (not a stem).

    OPSIN stores many entries as stems (e.g., 'norbornen', 'stilben', 'corrin')
    that need suffix appending by OPSIN's parser. These should not be used
    directly as retained names.

    Returns True if the name appears to be a complete, usable name.
    """
    name_lower = name.lower().rstrip()

    # Names containing spaces are typically complete (e.g., "acetic acid")
    if " " in name_lower:
        return True

    # Names with locant prefixes and hyphens often indicate complete names
    # but stems can also have locants (e.g., "isoflavan-3-en")

    # Check against known complete endings
    return any(name_lower.endswith(ending) for ending in _COMPLETE_NAME_ENDINGS)


# OPSIN names that are acceptable but NOT IUPAC 2013 PINs for simple compounds.
# These would override correct systematic naming. Exclude them so the naming
# algorithm produces the PIN (e.g., "ethene" not "ethylene").
_OPSIN_NON_PIN_EXCLUSIONS = {
    "ethylene", "propylene", "acetylene",  # PINs: ethene, propene, ethyne
    "isobutylene",  # PIN: 2-methylpropene
    "trimethylene",  # not a PIN
    "tetramethylene",  # not a PIN
    "pentamethylene",  # not a PIN
}

# Filter OPSIN names to only complete names (not stems) and exclude non-PINs
_OPSIN_NAMES: Dict[str, str] = {
    smi: name for smi, name in _OPSIN_NAMES_RAW.items()
    if _is_complete_name(name) and name.lower() not in _OPSIN_NON_PIN_EXCLUSIONS
}

_stem_count = len(_OPSIN_NAMES_RAW) - len(_OPSIN_NAMES)
if _stem_count > 0:
    logger.debug(
        "OPSIN imports: %d complete names included, %d stems filtered out",
        len(_OPSIN_NAMES), _stem_count
    )

# Merge: OPSIN first, then hand-curated overwrites (D-11: hand-curated wins)
ALL_RETAINED_NAMES: Dict[str, str] = {**_OPSIN_NAMES, **_HAND_CURATED_NAMES}

# Backward-compatible alias (D-12)
RETAINED_NAMES = ALL_RETAINED_NAMES


def get_retained_name(canonical_smiles: str) -> Optional[str]:
    """Look up retained name from merged dictionary."""
    return ALL_RETAINED_NAMES.get(canonical_smiles)


def has_retained_name(canonical_smiles: str) -> bool:
    """Check if a retained name exists for this SMILES."""
    return canonical_smiles in ALL_RETAINED_NAMES


def register_retained_name(canonical_smiles: str, name: str):
    """Register a retained name at runtime (for testing/dynamic use)."""
    ALL_RETAINED_NAMES[canonical_smiles] = name


# Log merge stats at import time
_conflict_count = sum(1 for k in _OPSIN_NAMES if k in _HAND_CURATED_NAMES)
if _conflict_count > 0:
    logger.info(
        "Retained names merged: %d hand-curated + %d OPSIN imports "
        "(%d conflicts, hand-curated wins)",
        len(_HAND_CURATED_NAMES), len(_OPSIN_NAMES), _conflict_count
    )

__all__ = ["RETAINED_NAMES", "ALL_RETAINED_NAMES", "get_retained_name",
           "has_retained_name", "register_retained_name"]
