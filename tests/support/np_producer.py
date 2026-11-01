"""Name with the natural-product producer itself, for a test of the producer's own spelling.

NPL 2026-11-01: INTRODUCTION (the Blue Book) "Preferred IUPAC names (PINs) are not
identified for the compounds in this Chapter", so the natural-product route offers the systematic name of
the whole molecule first (``rules.natural_products.systematic_name_before_natural_product``). A test
whose subject is the producer's spelling (alpha/beta descriptors, decoration locants, scaffold numbering,
ester spelling) switches that offer off with ``np_producer_only`` so the route names with the producer,
and names at the best-effort tier (``np_namer``), the tier that still emits a non-PIN name.
"""
import contextlib

import pytest

import orthonym.rules.natural_products as _np


@contextlib.contextmanager
def producer_only():
    """The natural-product route without its systematic offer (the producer's name stands)."""
    real = _np.systematic_name_before_natural_product
    _np.systematic_name_before_natural_product = lambda mol: None
    try:
        yield
    finally:
        _np.systematic_name_before_natural_product = real


@pytest.fixture
def np_producer_only():
    with producer_only():
        yield


def np_namer():
    """An engine at the best-effort tier: a natural-product name is emitted, below the PIN."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))
