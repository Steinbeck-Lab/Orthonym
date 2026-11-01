"""Name with the nucleoside producers themselves, for a test of their own spelling.

K2 2026-11-01: INTRODUCTION (the Blue Book) "Preferred IUPAC names (PINs) are not identified for
the compounds in this Chapter", so the retained-name route and the nucleoside route offer the systematic name of
the whole molecule first (``rules.nucleosides.systematic_nucleoside_name``). A test whose subject is the
nucleoside name ('5-methyluridine', 'adenosine 5'-acetate') switches that offer off with
``nucleoside_producer_only``.
"""
import contextlib

import pytest

import orthonym.rules.nucleosides as _nuc


@contextlib.contextmanager
def producer_only():
    """The nucleoside routes without their systematic offer (the nucleoside name stands)."""
    real = _nuc.systematic_nucleoside_name
    _nuc.systematic_nucleoside_name = lambda mol: None
    try:
        yield
    finally:
        _nuc.systematic_nucleoside_name = real


@pytest.fixture
def nucleoside_producer_only():
    with producer_only():
        yield


#: retained table name -> the systematic PIN offered in its place
SYSTEMATIC_FIRST = {
    'adenosine': '(2R,3R,4S,5R)-2-(6-amino-9H-purin-9-yl)-5-(hydroxymethyl)oxolane-3,4-diol',
    'cytidine': '4-amino-1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidin-2(1H)-one',
    'thymidine': '1-[(2R,4S,5R)-4-hydroxy-5-(hydroxymethyl)oxolan-2-yl]-5-methylpyrimidine-2,4(1H,3H)-dione',
    'uridine': '1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidine-2,4(1H,3H)-dione',
    "2'-deoxyadenosine": '(2R,3S,5R)-5-(6-amino-9H-purin-9-yl)-2-(hydroxymethyl)oxolan-3-ol',
    "2'-deoxycytidine": '4-amino-1-[(2R,4S,5R)-4-hydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidin-2(1H)-one',
    "2'-deoxyuridine": '1-[(2R,4S,5R)-4-hydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidine-2,4(1H,3H)-dione',
}
