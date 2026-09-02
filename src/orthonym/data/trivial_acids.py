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
    # Fatty acids (common long-chain acids with retained names)
    "lauric": "laurate",          # C12:0
    "myristic": "myristate",      # C14:0
    "palmitic": "palmitate",      # C16:0
    "stearic": "stearate",        # C18:0
    "oleic": "oleate",            # C18:1
    "linoleic": "linoleate",      # C18:2
    "linolenic": "linolenate",    # C18:3
    "arachidic": "arachidate",    # C20:0
    "arachidonic": "arachidonate", # C20:4
    # NOTE: there are deliberately NO rows keyed on the SYSTEMATIC stems
    # dodecanoic / tetradecanoic / hexadecanoic / octadecanoic / icosanoic.
    #
    # Five such rows used to live here, mapping the systematic stem onto the
    # trivial ester word ("hexadecanoic" -> "palmitate") so that "the
    # decomposition engine also uses trivial acylate forms when it encounters
    # systematic acid names". That is backwards for a preferred IUPAC name: it
    # takes a stem that is ALREADY the PIN and converts it into one that is not.
    #
    # P-65.1.2 "Systematic names" (BlueBookV2.md heading:29858, rule:29860) --
    # "Except for formic acid, acetic acid, oxalic acid (see P-65.1.1.1), and
    # oxamic acid (see P-65.1.1.1), systematically formed names are preferred
    # IUPAC names; the names given in P-65.1.1.2 are retained names for use in
    # general nomenclature." The Blue Book prints (PIN) on the systematic side
    # of every fatty row::29787 "palmitic acid hexadecanoic acid (PIN)".
    #
    # Re-adding any of them silently reverts the fix: with the count map in
    # rules/esters.py corrected but these rows present, all five esters STILL
    # emitted the trivial word (measured -- 5/5 unchanged). Both halves are
    # load-bearing. A systematic stem now falls through to the "-oic" -> "-oate"
    # branch below, which is the PIN.
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


def get_systematic_acylate(chain_length: int, unsaturation=None) -> str:
    """
    Get the systematic IUPAC acylate name for a chain length.

    Args:
        chain_length: Number of carbons in the acid chain (including C=O)
        unsaturation: Optional list of ``(double_bond_locant, 'E'|'Z'|'')`` tuples
            (locants from the carbonyl carbon = 1). None → saturated form,
            byte-identical to the prior bare-int behavior.

    Returns:
        Systematic acylate name

    Examples:
        >>> get_systematic_acylate(2)
        'ethanoate'
        >>> get_systematic_acylate(3)
        'propanoate'
        >>> get_systematic_acylate(18, unsaturation=[(9, 'Z')])
        '(9Z)-octadec-9-enoate'
    """
    if not unsaturation:
        if chain_length in CHAIN_TO_ACYLATE:
            return CHAIN_TO_ACYLATE[chain_length]
        # For longer chains, use centralized chain naming
        from .chain_names import get_anoate_name
        return get_anoate_name(chain_length)

    # Unsaturated: systematic -oate stem with leading (E/Z) block (no trivial map).
    from .chain_names import get_enoate_name
    return get_enoate_name(chain_length, unsaturation)
