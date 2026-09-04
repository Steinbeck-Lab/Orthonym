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


# ---------------------------------------------------------------------------
# P-45.5 — alphanumerical order of complete names
# ---------------------------------------------------------------------------

# Stereodescriptor parenthetical: (R)-, (2S)-, (1R,2S)-, (E)-, (2E,4Z)- ...
_STEREO_DESC_RE = re.compile(r"\((?:\d*[a-zA-Z]?'*[RSEZrsez](?:,\d*[a-zA-Z]?'*[RSEZrsez])*)\)-?")
# Leading/embedded indicated hydrogen: 1H-, 9aH-, 2H,3H- (italic H + locant)
_IH_STEM_RE = re.compile(r"\d+[a-z]?'*H[,-]")
# Italic heteroatom locants: N-, N,N'-, N2-, O-, S- (locant position only)
_ITALIC_LOCANT_RE = re.compile(r"(?<![a-zA-Z])[NOSP]\d*'*(?=[,-])")
# Numeral locant tokens in appearance order (incl. primes/superscript/lambda).
# The negative lookbehind stops a BARE lambda-convention marker with no
# preceding locant ("λ5-phosphanyl", P-45.3.1 mononuclear substituent) from
# having its bonding-number digit mis-read as an unrelated standalone locant
# ("5"); a real locant+lambda token ("2λ5", P-21.2.4 skeletal replacement)
# is unaffected since the lookbehind only guards the digit run's OWN start,
# and there the "2" -- not "λ" -- immediately precedes it.
_LOCANT_TOKEN_FINDER = re.compile(
    r"(?<!λ)(?<!lambda)\d+'*[a-z]?'*(?:\^\d+)?(?:(?:λ|lambda)\d+)?")

# naming_utils owns the nesting-relevant bracket grammar — reuse it.
from .naming_utils import _FUSION_BRACKET_RE, _INDICATED_H_RE  # noqa: E402


def _roman_letter_key(name: str) -> str:
    """Tier 1: Roman letters in order of appearance; italic elements removed.

    Removes (per BB P-45.5) stereodescriptors, indicated-hydrogen descriptors,
    fusion/von-Baeyer bracket contents, and italic heteroatom locants, then
    keeps only alphabetic characters, lowercased.
    """
    work = _STEREO_DESC_RE.sub('', name)
    work = _INDICATED_H_RE.sub('', work)
    work = _IH_STEM_RE.sub('', work)
    work = _FUSION_BRACKET_RE.sub('', work)
    work = _ITALIC_LOCANT_RE.sub('', work)
    return ''.join(ch for ch in work if ch.isalpha()).lower()


def _italic_letter_key(name: str) -> str:
    """Tier 2: italic letters in order of appearance (fusion letters, H,
    heteroatom locants) — BB P-45.5 / P-14.5.3."""
    out: List[str] = []
    for m in _FUSION_BRACKET_RE.finditer(name):
        out.extend(ch for ch in m.group(0) if ch.isalpha())
    for m in _IH_STEM_RE.finditer(name):
        out.append('h')
    for m in _ITALIC_LOCANT_RE.finditer(name):
        out.append(m.group(0)[0].lower())
    return ''.join(out)


def _numeral_key(name: str) -> Tuple[tuple, ...]:
    """Tier 3: numerical locants in order of APPEARANCE (not sorted) —
    BB P-45.5 final sentence."""
    return tuple(locant_sort_key(t) for t in _LOCANT_TOKEN_FINDER.findall(name))


def compare_names(a: str, b: str) -> int:
    """P-45.5 alphanumerical comparison of two complete candidate names.

    Returns -1 if a is earlier (preferred as PIN), 1 if b, 0 if equal.
    Tier 1: Roman letters in order of appearance (italics excluded).
    Tier 2: italic letters in order of appearance.
    Tier 3: numerical locants in order of appearance (locant_sort_key each).
    """
    for keyf in (_roman_letter_key, _italic_letter_key, _numeral_key):
        ka, kb = keyf(a), keyf(b)
        if ka < kb:
            return -1
        if ka > kb:
            return 1
    return 0


# ---------------------------------------------------------------------------
# P-45.3.2 — lower locant set for higher-bonding-number (λ) prefixes
# ---------------------------------------------------------------------------

_LAMBDA_SPLIT_RE = re.compile(r"^(?P<base>\d+'*[a-z]?'*)(?:λ|lambda)(?P<bond>\d+)$")


def parse_lambda_locant(token: str):
    """Split '1λ5' / '1lambda5' / "2'λ4" into (base_locant, bonding_number).

    Returns None for tokens without a λ mark. P-14.1.3: the λ symbol is
    'cited in conjunction with an appropriate locant'.
    """
    m = _LAMBDA_SPLIT_RE.match(token.strip())
    if not m:
        return None
    return (m.group('base'), int(m.group('bond')))


def compare_lambda_locant_sets(set_a: List[str], set_b: List[str]) -> int:
    """P-45.3.2 (BlueBookV2.md:22200): 'The preferred IUPAC name has the
    lower locant set for substituent group(s) with the higher bonding
    number(s) cited as prefixes.'

    Only λ-bearing tokens participate; their BASE locants are compared with
    P-14.3.5 first-point-of-difference semantics. Returns 0 when neither
    side cites a λ prefix (tier does not apply).
    """
    a_bases = [p[0] for p in (parse_lambda_locant(t) for t in set_a) if p]
    b_bases = [p[0] for p in (parse_lambda_locant(t) for t in set_b) if p]
    if not a_bases and not b_bases:
        return 0
    if a_bases and not b_bases:
        return -1   # P-45.3.1: more higher-bonding-number prefixes wins
    if b_bases and not a_bases:
        return 1
    return compare_locant_str_sets(a_bases, b_bases)
