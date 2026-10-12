"""The multiplying prefixes: ``di, tri, tetra,...`` and ``bis, tris, tetrakis,...``.

This module is the one source for those words (IUPAC 2013, Table 1.4, the
basic numerical terms, up to ``icosa`` / ``icosakis`` at 20).
``assembly.naming_utils`` re-exports ``SIMPLE_MULTIPLIERS`` and ``COMPLEX_MULTIPLIERS``
(the same dict objects), and a module that needs a table of its own range builds it with
``multiplier_table`` instead of writing the words out again.

It is a leaf: it imports nothing from ``orthonym``. Modules in every package import it
at load time, among them data modules that load before the assembly package, so it must
not depend on them (``naming_utils`` pulls in the whole assembly package).
"""

from typing import Dict, Optional

# Simple multiplicative prefixes (for simple substituent names)
SIMPLE_MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",
    13: "trideca",
    14: "tetradeca",
    15: "pentadeca",
    16: "hexadeca",
    17: "heptadeca",
    18: "octadeca",
    19: "nonadeca",
    20: "icosa",
}

# Complex multiplicative prefixes (for substituents with locants/hyphens)
COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
    7: "heptakis",
    8: "octakis",
    9: "nonakis",
    10: "decakis",
    11: "undecakis",
    12: "dodecakis",
    13: "tridecakis",
    14: "tetradecakis",
    15: "pentadecakis",
    16: "hexadecakis",
    17: "heptadecakis",
    18: "octadecakis",
    19: "nonadecakis",
    20: "icosakis",
}

_TABLES = {"simple": SIMPLE_MULTIPLIERS, "complex": COMPLEX_MULTIPLIERS}
_MAX_N = 20  # the last key of both tables


def multiplier_table(
    max_n: int, one: Optional[str] = None, kind: str = "simple"
) -> Dict[int, str]:
    """A new ``{count: prefix}`` dict for the counts ``2..max_n``, in ascending key order.

    ``kind`` picks the words: ``"simple"`` (``di, tri,...``) or ``"complex"``
    (``bis, tris,...``). ``one`` is the word for a count of 1: ``None`` leaves key 1
    out, a string adds it as the first key (``""`` where one unit takes no multiplier,
    ``"mono"`` where it is spelled out). The values come from ``SIMPLE_MULTIPLIERS`` /
    ``COMPLEX_MULTIPLIERS``, so a site states only the range it needs.

    Every call returns a fresh dict: a site that edits its table cannot change the table
    of another site or the shared ``*_MULTIPLIERS``.

    ``max_n`` outside ``2..20`` raises ``ValueError``. Words above 20 are composed, not
    tabulated (see ``naming_utils.simple_multiplier_word``), so a larger range is a
    naming decision at the caller and never a silent truncation here.
    """
    try:
        source = _TABLES[kind]
    except KeyError:
        raise ValueError(f"kind must be 'simple' or 'complex', not {kind!r}") from None
    if one is not None and not isinstance(one, str):
        raise TypeError(f"one must be None or a str, not {type(one).__name__}")
    if not 2 <= max_n <= _MAX_N:
        raise ValueError(f"max_n must be in 2..{_MAX_N}, not {max_n!r}")
    table = {} if one is None else {1: one}
    for n in range(2, max_n + 1):
        table[n] = source[n]
    return table
