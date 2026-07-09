"""P-45.5 / P-14.5 alphanumerical name comparison + P-14.3.5 locant ordering.

Pure string logic — no RDKit, no imports from handlers (safe to import from
anywhere in assembly/ or rules/ without cycles).

BB P-14.3.5 (BlueBookV2.md:3191-3195): "Primed locants are placed immediately
after the corresponding unprimed locants ...; locants consisting of a number
and a lower-case letter with or without primes as 4a and 4'a (not 4a') are
placed immediately after the corresponding numeric locant and are followed by
locants having superscripts. Italic capital and lower-case letter locants are
lower than Greek letter locants, which, in turn, are lower than numerals."
"""
from __future__ import annotations

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# P-14.3.5 — single-locant total order
# ---------------------------------------------------------------------------

_GREEK = {
    'alpha': 1, 'beta': 2, 'gamma': 3, 'delta': 4, 'epsilon': 5, 'zeta': 6,
    'eta': 7, 'theta': 8, 'omega': 24,
    'α': 1, 'β': 2, 'γ': 3, 'δ': 4, 'ε': 5, 'ζ': 6, 'η': 7, 'θ': 8, 'ω': 24,
}

# ASCII locant token grammar used by Orthonym names:
#   numeral form:  4 | 4a | 2' | 4'a | 3a^1 | 1^2 | 1λ5 | 1lambda5
#   italic form:   N | N' | N2 | N2' | O | S | P
_NUMERAL_TOKEN_RE = re.compile(
    r"^(?P<num>\d+)(?P<primes1>'*)(?P<letter>[a-z]?)(?P<primes2>'*)"
    r"(?:\^(?P<sup>\d+))?(?:(?:λ|lambda)(?P<lam>\d+))?$"
)
_ITALIC_TOKEN_RE = re.compile(
    r"^(?P<sym>[NOSP])(?P<sup>\d*)(?P<primes>'*)$"
)


def locant_sort_key(token: str) -> tuple:
    """Total-order key for one locant token per P-14.3.5.

    Key layout: (class, base, letter, primes, superscript)
      class: 0 = italic Roman letter (N/O/S/P), 1 = Greek, 2 = numeral
      within a numeral: base number, then letter suffix ('' < 'a' < 'b'),
      with primes ranking between the bare number and the letter-suffixed
      forms (4 < 4' < 4a < 4'a), superscripts last (3a < 3a^1).

    The λ bonding-number mark (P-45.3.2 material) does NOT perturb the
    P-14.3.5 order — the base locant decides; the λ value is exposed via
    parse_lambda_locant() for the P-45.3.2 comparator (Task 4).
    """
    tok = token.strip()
    m = _NUMERAL_TOKEN_RE.match(tok)
    if m:
        primes = len(m.group('primes1') or '') + len(m.group('primes2') or '')
        letter = m.group('letter') or ''
        sup = int(m.group('sup')) if m.group('sup') else 0
        # letter rank: '' sorts before any letter; prime beats letter (4' < 4a)
        # encode as (has_letter, primes_when_no_letter, letter, primes_when_letter)
        if letter:
            return (2, int(m.group('num')), 1, letter, primes, sup)
        return (2, int(m.group('num')), 0, '', primes, sup)
    m = _ITALIC_TOKEN_RE.match(tok)
    if m:
        sup = int(m.group('sup')) if m.group('sup') else 0
        return (0, m.group('sym'), 0, '', len(m.group('primes')), sup)
    low = tok.lower().rstrip("'")
    if low in _GREEK:
        primes = len(tok) - len(tok.rstrip("'"))
        return (1, _GREEK[low], 0, '', primes, 0)
    # Fail-closed for the comparator: unknown token sorts after everything
    # known, deterministically by its text (never raises mid-naming).
    return (3, tok, 0, '', 0, 0)


def compare_locant_strings(a: str, b: str) -> int:
    ka, kb = locant_sort_key(a), locant_sort_key(b)
    return -1 if ka < kb else (1 if ka > kb else 0)


def compare_locant_str_sets(set_a: List[str], set_b: List[str]) -> int:
    """First-point-of-difference over string locant sets (P-14.3.5).

    Mirrors rules/locants.py::compare_locant_sets semantics: both sets are
    sorted ascending (by locant_sort_key), compared term by term; on a tied
    prefix the shorter set wins; identical sets return 0.
    """
    a_sorted = sorted(set_a, key=locant_sort_key)
    b_sorted = sorted(set_b, key=locant_sort_key)
    for a, b in zip(a_sorted, b_sorted):
        r = compare_locant_strings(a, b)
        if r != 0:
            return r
    if len(a_sorted) < len(b_sorted):
        return -1
    if len(a_sorted) > len(b_sorted):
        return 1
    return 0
