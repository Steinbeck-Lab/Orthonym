"""
Centralized IUPAC chain naming module.

Provides authoritative chain prefix generation for all carbon chain lengths
from 1 to 9999, following IUPAC 2013 Blue Book nomenclature rules.

IUPAC Long Chain Naming System:
- 1-20: Individual retained prefixes (meth, eth, prop,... icos)
- 21+: Compositional system using units + tens + hundreds + thousands
  - Units (1-9): hen, do, tri, tetra, penta, hexa, hepta, octa, nona
  - Tens (20-90): cos, triacont, tetracont, pentacont, hexacont, heptacont,
                   octacont, nonacont
  - Hundreds (100-900): hect, dict, trict, tetract, pentact, hexact, heptact,
                         octact, nonact
  - Thousands (1000-9000): kili, dili, trili, tetrali, pentali, hexali,
                            heptali, octali, nonali
  - Assembly: units + tens + hundreds + thousands (right-to-left composition)

Special linking rules:
- 21: hen + i + cos = "henicos" (linking 'i' before "cos" after "hen")
- 30+: ones + tens-prefix (e.g., dotriacont, tripentacont)
- 100+: (ones+tens) + "a" + hundreds (linking 'a' before hundreds)
- 1000+: (sub-thousand) + "a" + thousands (linking 'a' before thousands)

References:
    IUPAC 2013 Blue Book, Table A6.1 (Numerical terms used in nomenclature)
    IUPAC 2013 Blue Book, (Thousands digit prefixes)

Examples:
    >>> get_chain_prefix(1)
    'meth'
    >>> get_chain_prefix(21)
    'henicos'
    >>> get_chain_prefix(32)
    'dotriacont'
    >>> get_chain_prefix(100)
    'hect'
    >>> get_chain_prefix(132)
    'dotriacontahect'
    >>> get_chain_prefix(1000)
    'kili'
    >>> get_chain_prefix(1001)
    'henakili'
"""


# ============================================================================
# Direct lookup for standard prefixes (1-20)
# ============================================================================

FIRST_20 = {
    1: "meth", 2: "eth", 3: "prop", 4: "but", 5: "pent",
    6: "hex", 7: "hept", 8: "oct", 9: "non", 10: "dec",
    11: "undec", 12: "dodec", 13: "tridec", 14: "tetradec",
    15: "pentadec", 16: "hexadec", 17: "heptadec", 18: "octadec",
    19: "nonadec", 20: "icos",
}

# ============================================================================
# Compositional components for 21+
# ============================================================================

# Units digit (1-9) used in compositional naming
UNITS = {
    0: "", 1: "hen", 2: "do", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona",
}

# Tens digit (2-9) -> tens prefix
# Note: tens=1 (10-19) is handled by FIRST_20 for n<=20,
# and by "deca" for 110, 210, etc. (in combination with hundreds)
# tens=2 (20-29) uses "cos" (special: "icos" for 21 with linking 'i')
TENS = {
    2: "cos", 3: "triacont", 4: "tetracont", 5: "pentacont",
    6: "hexacont", 7: "heptacont", 8: "octacont", 9: "nonacont",
}

# Hundreds digit (1-9) -> hundreds prefix
HUNDREDS = {
    1: "hect", 2: "dict", 3: "trict", 4: "tetract", 5: "pentact",
    6: "hexact", 7: "heptact", 8: "octact", 9: "nonact",
}

# Thousands digit (1-9) -> thousands prefix
# Source: IUPAC 2013 Blue Book Table 1.4
THOUSANDS = {
    1: "kili", 2: "dili", 3: "trili", 4: "tetrali", 5: "pentali",
    6: "hexali", 7: "heptali", 8: "octali", 9: "nonali",
}


# ============================================================================
# Public API
# ============================================================================

def get_chain_prefix(n: int) -> str:
    """Get IUPAC chain prefix for n carbons.

    Supports chain lengths from 1 to 9999.

    Args:
        n: Number of carbon atoms (1-9999).

    Returns:
        IUPAC chain prefix string (e.g., 'meth', 'eth', 'henicos', 'dotriacont').

    Raises:
        ValueError: If n is outside supported range (1-9999).

    Examples:
        >>> get_chain_prefix(1)
        'meth'
        >>> get_chain_prefix(10)
        'dec'
        >>> get_chain_prefix(20)
        'icos'
        >>> get_chain_prefix(21)
        'henicos'
        >>> get_chain_prefix(27)
        'heptacos'
        >>> get_chain_prefix(30)
        'triacont'
        >>> get_chain_prefix(32)
        'dotriacont'
        >>> get_chain_prefix(41)
        'hentetracont'
        >>> get_chain_prefix(53)
        'tripentacont'
        >>> get_chain_prefix(100)
        'hect'
        >>> get_chain_prefix(132)
        'dotriacontahect'
        >>> get_chain_prefix(1000)
        'kili'
        >>> get_chain_prefix(1001)
        'henakili'
    """
    if n < 1 or n > 9999:
        raise ValueError(
            f"Chain length {n} outside supported range (1-9999)."
        )

    # Direct lookup for 1-20
    if n <= 20:
        return FIRST_20[n]

    # Thousands decomposition
    thousands = n // 1000
    remainder = n % 1000

    if thousands == 0:
        # 21-999: use compositional sub-thousand builder
        return _build_prefix_21_to_999(n)

    if remainder == 0:
        # Pure thousands: 1000, 2000,...
        return THOUSANDS[thousands]

    # Build sub-thousand part using compositional units (not retained names)
    # For 1-9: use UNITS (hen, do, tri,...) - same as hundreds context
    # For 10-20: use compositional sub-hundred system
    # For 21-999: use full compositional builder
    if remainder <= 9:
        sub_thousand = UNITS[remainder]
    elif remainder <= 20:
        tens = remainder // 10
        ones = remainder % 10
        sub_thousand = _build_sub_hundred(ones, tens)
    else:
        sub_thousand = _build_prefix_21_to_999(remainder)

    thousands_prefix = THOUSANDS[thousands]
    # Linking vowel 'a' between sub-thousand and thousands prefix
    if sub_thousand.endswith("a"):
        return sub_thousand + thousands_prefix
    else:
        return sub_thousand + "a" + thousands_prefix


def _build_prefix_21_to_999(n: int) -> str:
    """Build chain prefix for 21-999 using compositional rules.

    Decomposes the number into hundreds, tens, and ones digits,
    then assembles the prefix using IUPAC linking rules.

    Args:
        n: Number in range 21-999.

    Returns:
        Compositional chain prefix string.
    """
    hundreds = n // 100
    tens = (n % 100) // 10
    ones = n % 10

    # Build the sub-hundred part (ones + tens)
    sub_hundred = _build_sub_hundred(ones, tens)

    # Pure hundreds (100, 200, 300,...)
    if not sub_hundred and hundreds > 0:
        return HUNDREDS[hundreds]

    # Add hundreds if present
    if hundreds > 0:
        hundreds_prefix = HUNDREDS[hundreds]
        # Linking vowel 'a' between sub-hundred and hundreds
        if sub_hundred.endswith("a"):
            return sub_hundred + hundreds_prefix
        else:
            return sub_hundred + "a" + hundreds_prefix

    return sub_hundred


def _build_sub_hundred(ones: int, tens: int) -> str:
    """Build the sub-hundred component (ones + tens) of a chain prefix.

    Handles special linking rules:
    - For tens=2 (the "cos" series): "hen" becomes "henicos" (linking 'i')
    - For tens=1 (the "dec" series in hundreds context): uses "dec"
    - For tens=0: just the units prefix

    Args:
        ones: Units digit (0-9).
        tens: Tens digit (0-9).

    Returns:
        Sub-hundred prefix string.
    """
    if tens == 0 and ones == 0:
        return ""

    if tens == 0:
        # Only units, no tens (e.g., 101 = hen + hect)
        return UNITS[ones]

    if tens == 1:
        # Teens in hundreds context (e.g., 110 = dec + a + hect,
        # 111 = hendec + a + hect, 112 = dodec + a + hect)
        if ones == 0:
            return "dec"
        else:
            # Use the standard teen names: undec, dodec, tridec, etc.
            # These are already in FIRST_20 for 11-19
            return FIRST_20[10 + ones]

    # tens >= 2
    tens_prefix = TENS[tens]

    if ones == 0:
        # Pure tens: 20, 30, 40,...
        # 20 is handled by FIRST_20, so this is for 30+, or 20 in hundreds
        return tens_prefix

    unit_prefix = UNITS[ones]

    # Special linking for tens=2 ("cos" series):
    # "hen" + "icos" (linking 'i') for ones=1
    # All others: unit + "cos" directly
    if tens == 2 and ones == 1:
        return "henicos"

    # For tens >= 3, unit + tens directly (e.g., do + triacont = dotriacont)
    # For tens == 2 and ones >= 2, unit + cos directly (e.g., do + cos = docos)
    return unit_prefix + tens_prefix


def get_chain_name(n: int) -> str:
    """Get full IUPAC chain name (alkane) for n carbons.

    Args:
        n: Number of carbon atoms (1-9999).

    Returns:
        Full alkane name (e.g., 'methane', 'ethane', 'henicosane').

    Examples:
        >>> get_chain_name(1)
        'methane'
        >>> get_chain_name(21)
        'henicosane'
        >>> get_chain_name(32)
        'dotriacontane'
    """
    prefix = get_chain_prefix(n)
    return prefix + "ane"


def get_alkyl_name(n: int) -> str:
    """Get IUPAC alkyl substituent name for n carbons.

    Uses the standard '-yl' suffix appended to the chain prefix.

    Args:
        n: Number of carbon atoms (1-9999).

    Returns:
        Alkyl substituent name (e.g., 'methyl', 'ethyl', 'henicosyl').

    Examples:
        >>> get_alkyl_name(1)
        'methyl'
        >>> get_alkyl_name(6)
        'hexyl'
        >>> get_alkyl_name(21)
        'henicosyl'
        >>> get_alkyl_name(41)
        'hentetracontyl'
    """
    prefix = get_chain_prefix(n)
    return prefix + "yl"


def get_acid_name(n: int) -> str:
    """Get IUPAC carboxylic acid name for n carbons (including COOH carbon).

    Args:
        n: Number of carbon atoms (1-9999).

    Returns:
        Acid name (e.g., 'methanoic acid', 'ethanoic acid', 'henicosanoic acid').

    Examples:
        >>> get_acid_name(1)
        'methanoic acid'
        >>> get_acid_name(2)
        'ethanoic acid'
        >>> get_acid_name(28)
        'octacosanoic acid'
    """
    prefix = get_chain_prefix(n)
    return prefix + "anoic acid"


def get_acid_stem(n: int) -> str:
    """Get IUPAC acid stem (prefix + 'anoic') for n carbons.

    Used when just the stem is needed without ' acid'.

    Args:
        n: Number of carbon atoms (1-9999).

    Returns:
        Acid stem (e.g., 'methanoic', 'ethanoic', 'octacosanoic').

    Examples:
        >>> get_acid_stem(2)
        'ethanoic'
        >>> get_acid_stem(28)
        'octacosanoic'
    """
    prefix = get_chain_prefix(n)
    return prefix + "anoic"


def get_anoate_name(n: int) -> str:
    """Get IUPAC ester suffix (anoate form) for n carbons.

    Args:
        n: Number of carbon atoms (1-9999).

    Returns:
        Anoate name (e.g., 'methanoate', 'ethanoate', 'icosanoate').

    Examples:
        >>> get_anoate_name(1)
        'methanoate'
        >>> get_anoate_name(20)
        'icosanoate'
    """
    prefix = get_chain_prefix(n)
    return prefix + "anoate"


_ENE_MULTIPLIER = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}


def get_enoate_name(n: int, unsaturation=None) -> str:
    """Systematic (un)saturated ``-oate`` stem, with leading (E/Z) stereo block.

    Args:
        n: Number of carbons including the carbonyl carbon.
        unsaturation: Optional list of ``(double_bond_locant, 'E'|'Z'|'')`` tuples
            (locants counted from the carbonyl carbon = 1). None/empty → the
            saturated ``-anoate`` form (identical to:func:`get_anoate_name`).

    Returns:
        e.g. ``'octadecanoate'`` (saturated), ``'(9Z)-octadec-9-enoate'`` (1 db),
        ``'(9Z,12Z)-octadeca-9,12-dienoate'`` (2 db). Mirrors the systematic form
        the chain/ester pipeline already emits for unsaturated fatty esters.
    """
    if not unsaturation:
        return get_anoate_name(n)
    prefix = get_chain_prefix(n)
    locs = sorted(unsaturation, key=lambda t: t[0])
    k = len(locs)
    locants = ",".join(str(loc) for loc, _ in locs)
    if k == 1:
        core = f"{prefix}-{locants}-enoate"
    else:
        mult = _ENE_MULTIPLIER.get(k, "")
        core = f"{prefix}a-{locants}-{mult}enoate"
    block = ",".join(f"{loc}{geom}" for loc, geom in locs if geom in ("E", "Z"))
    return f"({block})-{core}" if block else core
