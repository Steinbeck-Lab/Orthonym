"""
Acid name to acylate (-ate/-oate) conversion for ester naming.

Maps acid names (both trivial and systematic) to their ester suffix forms.
For systematic names, the conversion is straightforward:
- -oic acid -> -oate (propanoic acid -> propanoate)
- -ic acid -> -ate (acetic acid -> acetate)

For trivial names, we need explicit mappings.
"""

from typing import Optional


# Trivial acid name -> acylate form
# These are common acids with retained trivial names
TRIVIAL_ACID_TO_ACYLATE = {
    "formic": "formate",
    "acetic": "acetate",
    "propionic": "propionate",  # Alternative to propanoic
    "butyric": "butyrate",      # Alternative to butanoic
    "valeric": "valerate",      # Alternative to pentanoic
    "caproic": "caproate",      # Alternative to hexanoic
    "benzoic": "benzoate",
    "oxalic": "oxalate",
    "malonic": "malonate",
    "succinic": "succinate",
    "glutaric": "glutarate",
    "adipic": "adipate",
    "phthalic": "phthalate",
    "lactic": "lactate",
    "citric": "citrate",
    "tartaric": "tartrate",
    "malic": "malate",
    "fumaric": "fumarate",
    "maleic": "maleate",
}

# Chain length to systematic acylate form
# These are the IUPAC systematic names
CHAIN_TO_ACYLATE = {
    1: "methanoate",   # formate is trivial
    2: "ethanoate",    # acetate is trivial
    3: "propanoate",
    4: "butanoate",
    5: "pentanoate",
    6: "hexanoate",
    7: "heptanoate",
    8: "octanoate",
    9: "nonanoate",
    10: "decanoate",
}


def get_acylate_name(acid_name: str) -> str:
    """
    Convert an acid name to its acylate (ester suffix) form.

    Args:
        acid_name: The acid name (e.g., "acetic", "propanoic", "benzoic")

    Returns:
        The acylate form (e.g., "acetate", "propanoate", "benzoate")

    Examples:
        >>> get_acylate_name("acetic")
        'acetate'
        >>> get_acylate_name("propanoic")
        'propanoate'
        >>> get_acylate_name("benzoic")
        'benzoate'
    """
    # Check trivial names first
    if acid_name.lower() in TRIVIAL_ACID_TO_ACYLATE:
        return TRIVIAL_ACID_TO_ACYLATE[acid_name.lower()]

    # Handle systematic names: -oic -> -oate, -ic -> -ate
    if acid_name.endswith("oic"):
        return acid_name[:-2] + "ate"  # propanoic -> propanoate
    if acid_name.endswith("ic"):
        return acid_name[:-2] + "ate"  # generic fallback

    # Default: append -ate (shouldn't normally happen)
    return acid_name + "ate"


def get_systematic_acylate(chain_length: int) -> str:
    """
    Get the systematic IUPAC acylate name for a chain length.

    Args:
        chain_length: Number of carbons in the acid chain (including C=O)

    Returns:
        Systematic acylate name

    Examples:
        >>> get_systematic_acylate(2)
        'ethanoate'
        >>> get_systematic_acylate(3)
        'propanoate'
    """
    if chain_length in CHAIN_TO_ACYLATE:
        return CHAIN_TO_ACYLATE[chain_length]

    # For longer chains, generate programmatically
    from ..assembly.composer import CHAIN_PREFIXES
    if chain_length in CHAIN_PREFIXES:
        return CHAIN_PREFIXES[chain_length] + "anoate"

    return f"{chain_length}Canoate"  # Fallback
