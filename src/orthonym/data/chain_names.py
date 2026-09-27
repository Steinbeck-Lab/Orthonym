"""
Centralized IUPAC chain naming module.

Provides authoritative chain prefix generation for all carbon chain lengths
from 1 to 9999, following IUPAC 2013 Blue Book nomenclature rules.

IUPAC Long Chain Naming System, the Blue Book; numerical terms
, Table 1.4 at:2794):
- 1-20: Individual retained prefixes (meth, eth, prop,... icos)
- 21+: the basic numerical term of the number with its terminal 'a' elided
  before 'ane'. The term is built from the basic terms of Table 1.4 cited in the
  order opposite to the digits (units, tens, hundreds, thousands):
  - Units (1-9), in association: hen, do, tri, tetra, penta, hexa, hepta,
    octa, nona,:2807: '1' is 'hen' and '2' is 'do' in
    association, except 'undeca', 'dicta' and 'dilia')
  - Tens (10-90): deca, icosa, triaconta, tetraconta,... nonaconta
  - Hundreds (100-900): hecta, dicta, tricta, tetracta,... nonacta
  - Thousands (1000-9000): kilia, dilia, trilia, tetralia,... nonalia

Joining,:2811): "The composite terms are formed by direct joining
of the basic terms, without hyphen(s). The letter 'i' in 'icosa' is elided after
a vowel." So there is no linking letter: 101 henhecta (Table 1.4), 111
undecahecta, 363 trihexacontatricta, 486 hexaoctacontatetracta, 1001 henkilia
(Table 1.4); and 'icosa' keeps its 'i' unless a vowel precedes it: 21 henicosa,
22 docosa, 120 icosahecta.

References:
    IUPAC 2013 Blue Book,, Table 1.4 (basic numerical terms)
    IUPAC 2013 Blue Book, (derivation of basic numerical terms)
    IUPAC 2013 Blue Book, (unbranched acyclic hydrocarbons)

Examples:
    >>> get_chain_prefix(1)
    'meth'
    >>> get_chain_prefix(21)
    'henicos'
    >>> get_chain_prefix(32)
    'dotriacont'
    >>> get_chain_prefix(100)
    'hect'
    >>> get_chain_prefix(101)
    'henhect'
    >>> get_chain_prefix(120)
    'icosahect'
    >>> get_chain_prefix(132)
    'dotriacontahect'
    >>> get_chain_prefix(1000)
    'kili'
    >>> get_chain_prefix(1001)
    'henkili'
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
# Basic numerical terms (Table 1.4, the Blue Book), for 21+
# ============================================================================

# Units digit (1-9) in association with other numerical terms
#: 'hen' for 1 and 'do' for 2).
UNIT_TERMS = {
    1: "hen", 2: "do", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona",
}

# Tens digit (1-9). 11 is 'undeca', handled in _sub_hundred_term.
TENS_TERMS = {
    1: "deca", 2: "icosa", 3: "triaconta", 4: "tetraconta", 5: "pentaconta",
    6: "hexaconta", 7: "heptaconta", 8: "octaconta", 9: "nonaconta",
}

# Hundreds digit (1-9).
HUNDREDS_TERMS = {
    1: "hecta", 2: "dicta", 3: "tricta", 4: "tetracta", 5: "pentacta",
    6: "hexacta", 7: "heptacta", 8: "octacta", 9: "nonacta",
}

# Thousands digit (1-9).
THOUSANDS_TERMS = {
    1: "kilia", 2: "dilia", 3: "trilia", 4: "tetralia", 5: "pentalia",
    6: "hexalia", 7: "heptalia", 8: "octalia", 9: "nonalia",
}

_VOWELS = frozenset("aeiou")


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
        >>> get_chain_prefix(102)
        'dohect'
        >>> get_chain_prefix(132)
        'dotriacontahect'
        >>> get_chain_prefix(1000)
        'kili'
        >>> get_chain_prefix(1001)
        'henkili'
    """
    if n < 1 or n > 9999:
        raise ValueError(
            f"Chain length {n} outside supported range (1-9999)."
        )

    # Direct lookup for 1-20
    if n <= 20:
        return FIRST_20[n]

    #: the numerical term with its terminal 'a' elided before 'ane'.
    # For n >= 21 the term always ends in the 'a' of its highest-digit term.
    return numerical_term(n)[:-1]


def numerical_term(n: int) -> str:
    """The basic numerical term for ``n`` (21-9999) in association.

    The basic terms of Table 1.4 are cited in the order opposite to the digits
    (units, tens, hundreds, thousands) and joined directly, without a linking
    letter; the 'i' of 'icosa' is elided after a vowel.

    Examples: 21 'henicosa', 22 'docosa', 101 'henhecta', 111 'undecahecta',
    120 'icosahecta', 363 'trihexacontatricta', 486 'hexaoctacontatetracta',
    1001 'henkilia'.
    """
    if n < 21 or n > 9999:
        raise ValueError(f"numerical_term({n}): supported range is 21-9999.")
    thousands = n // 1000
    hundreds = (n % 1000) // 100
    tens = (n % 100) // 10
    ones = n % 10
    return (_sub_hundred_term(ones, tens)
            + HUNDREDS_TERMS.get(hundreds, "")
            + THOUSANDS_TERMS.get(thousands, ""))


def _sub_hundred_term(ones: int, tens: int) -> str:
    """Units + tens part of a numerical term ('' for 0).

    Direct joining: 'hen' + 'hecta' is 'henhecta', 'do' + 'triaconta'
    'dotriaconta'. Exceptions: 11 is 'undeca', and the 'i' of
    'icosa' is elided after a vowel ('docosa', 'tricosa'; 'henicosa', and a bare
    'icosa' keep it).
    """
    if tens == 1 and ones == 1:
        return "undeca"
    unit = UNIT_TERMS.get(ones, "")
    tens_term = TENS_TERMS.get(tens, "")
    if tens_term.startswith("i") and unit and unit[-1] in _VOWELS:
        tens_term = tens_term[1:]
    return unit + tens_term




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
