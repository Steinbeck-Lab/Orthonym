"""Regression lock: the λ-spelling fix (ASCII 'lambda' -> Greek 'λ') exposed a
latent citation-order bug for substituents whose name differs from a plain
substituent ONLY by a P-45.3.1 lambda-convention marker.

TWO independent root causes, both in code that predates the λ-spelling change
and were masked by it, not caused by it:

1. ``naming_utils._ALPHA_NOISE_HEAD`` (P-14.5 preamble, BlueBookV2.md:3446:
   "Italicized Greek letters ... are not involved in the alphanumerical
   order") matched only a BARE single Greek letter (``[α-ω]``), never a
   Greek letter immediately followed by a bonding-number digit
   (``λ5``, ``λ4``, ``λ6``). ``strip_alphanumerical_noise('λ5-phosphanyl')``
   therefore returned the string UNCHANGED, and the leftover ``λ`` (U+03BB)
   sorts after every ASCII letter -- so ``λ5-phosphanyl`` sorted AFTER
   ``phosphanylmethyl`` instead of before it (W2F-P7-04). The OLD ASCII
   spelling ``lambda5-phosphanyl`` was ALSO never stripped by this regex,
   but happened to sort correctly anyway ('l' < 'p'), which is exactly why
   the λ-spelling change (not this defect) is what turned the gold row red.

2. ``assembly.name_comparison._LOCANT_TOKEN_FINDER`` (P-14.3.5 numeral-locant
   tokenizer, reused by ``prefix_citation_sort_key`` tier 2 / P-14.5.4) is a
   bare ``\\d+...`` pattern with no anchor on what precedes it. Once (1) is
   fixed, ``λ5-phosphanyl`` and ``phosphanyl`` tie at tier 1 (both strip to
   'phosphanyl'), so the comparison falls through to tier 2 -- where the
   tokenizer mis-read the bonding-number digit inside the BARE marker
   ``λ5`` (no locant precedes it) as if it were an ordinary standalone
   locant ``'5'``, manufacturing a false non-empty locant tuple that always
   sorts AFTER the true empty tuple ``phosphanyl`` gets. That flipped
   ``1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol`` to
   ``3-phosphanyl-1-(λ5-phosphanyl)propan-2-ol`` (W2F-P7-05). A genuine
   locant+lambda token (``2λ5``, P-21.2.4 skeletal replacement / P-24.2.4.1
   spiro) is unaffected: there the digit immediately before the λ IS a real
   locant, so the fix's negative lookbehind lets that match through exactly
   as before.

Both are P-14.5/.3.5 rule-derivation bugs in shared infrastructure, not
lambda-spelling bugs -- see ``
and the lambda-fix-report.md append for the fast-gate finding
(W2F-P7-04/W2F-P7-05, 1650/1652) that surfaced them.
"""
import pytest

from orthonym.assembly.naming_utils import strip_alphanumerical_noise
from orthonym.assembly.name_comparison import _LOCANT_TOKEN_FINDER
from orthonym.namer import Orthonym


@pytest.mark.unit
def test_strip_alphanumerical_noise_removes_bare_lambda_marker():
    """P-14.5: the λ-convention descriptor is excluded from alphanumerical
    order exactly like a bare Greek letter, INCLUDING its bonding-number
    digit (the digit is part of the same excluded token, not a locant)."""
    assert strip_alphanumerical_noise('λ5-phosphanyl') == 'phosphanyl'
    assert strip_alphanumerical_noise('λ4-thiaspiro[3.5]nonane') == 'thiaspiro[3.5]nonane'
    assert strip_alphanumerical_noise('λ6-sulfanone') == 'sulfanone'


@pytest.mark.unit
def test_strip_alphanumerical_noise_unaffected_regressions():
    """The new alternative must not disturb the noise-head's existing jobs."""
    assert strip_alphanumerical_noise('(e)-3-phenylprop-2-en-1-yl') == '3-phenylprop-2-en-1-yl'
    assert strip_alphanumerical_noise('beta-d-glucopyranosyloxy') == 'glucopyranosyloxy'
    assert strip_alphanumerical_noise('[4-2h]benzoyl') == 'benzoyl'
    assert strip_alphanumerical_noise('decyl') == 'decyl'


@pytest.mark.unit
def test_locant_token_finder_ignores_bare_lambda_digit():
    """A standalone λN/lambdaN marker with NO preceding locant contributes
    no locant token (P-14.3.5 tier 2 must not manufacture a fake tie-break
    out of the bonding-number digit)."""
    assert _LOCANT_TOKEN_FINDER.findall('λ5-phosphanyl') == []
    assert _LOCANT_TOKEN_FINDER.findall('(λ5-phosphanyl)') == []
    assert _LOCANT_TOKEN_FINDER.findall('lambda5-phosphanyl') == []


@pytest.mark.unit
def test_locant_token_finder_still_reads_genuine_locant_lambda_tokens():
    """A REAL locant immediately followed by its bonding-number marker
    (P-21.2.4 skeletal replacement, P-24.2.4.1 spiro) must still tokenize as
    ONE combined token, unaffected by the bare-marker fix above."""
    assert _LOCANT_TOKEN_FINDER.findall('2λ5,3-oxathiolane') == ['2λ5', '3']
    assert _LOCANT_TOKEN_FINDER.findall('4λ4-thiaspiro') == ['4λ4']
    # plain (non-lambda) locants are untouched
    assert _LOCANT_TOKEN_FINDER.findall('3-methylbutyl') == ['3']


# --- the two fast-gate-caught gold targets (characteristic_groups.json) ---

@pytest.mark.unit
def test_w2f_p7_04_lambda_before_phosphanylmethyl():
    """W2F-P7-04 (BlueBookV2.md:22182, P-45.3.1 evidence PIN): once λ5 is
    excluded from alphanumerical order, 'phosphanyl' is a proper PREFIX of
    'phosphanylmethyl' and is cited first, regardless of which locant is
    numerically lower."""
    assert Orthonym().name("OC(=O)C(CP)C[PH4]") == (
        "3-(λ5-phosphanyl)-2-(phosphanylmethyl)propanoic acid")


@pytest.mark.unit
def test_w2f_p7_05_lambda_before_plain_phosphanyl_on_tie():
    """W2F-P7-05 (BlueBookV2.md:3318,3334, P-14.4(h)): 'phosphanyl' and
    'λ5-phosphanyl' tie completely at P-14.5 tier 1 (both strip to the
    identical word); citation must not fall back to insertion-order luck."""
    assert Orthonym().name("OC(C[PH4])CP") == (
        "1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol")
