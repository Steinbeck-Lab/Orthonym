"""A decomposition that declines leaves the provenance of the molecule as it found it (lane TI2).

The rescue of a fragment (``engine._name_fragment_t4_rescue``) records the source of the fragment's name
(``general_engine``). The record belongs to the route that is taken: a decomposition that names nothing is not taken,
and the name of the assembly pipeline that ships next was labelled by the source of a fragment it does not contain.
The stub stands for ``try_decompose``; the molecule is in tests/unit/test_tier_inversions.py.
"""
import pytest

from orthonym.decomposition.engine import _declined_leaves_no_provenance, try_decompose
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_provenance():
    pv.clear_provenance()
    yield
    pv.clear_provenance()


def _records():
    pv.record_source("general_engine", opsin="verified")
    pv.record_non_pin_fragment("2-oxaprop")


def test_a_decline_puts_the_source_back_and_keeps_the_name_scoped_record():
    @_declined_leaves_no_provenance
    def declines(mol, style="pin"):
        _records()
        return None
    assert declines(None) is None
    prov = pv.get_provenance()
    assert prov["source"] is None and prov["opsin"] is None
    # a recorded non-PIN spelling only demotes a name that contains it (metrics.provenance.restore_provenance)
    assert prov["non_pin_fragments"] == ("2-oxaprop",)


def test_a_name_keeps_the_provenance_of_the_route_that_made_it():
    @_declined_leaves_no_provenance
    def names(mol, style="pin"):
        _records()
        return "a name"
    assert names(None) == "a name"
    assert pv.get_provenance()["source"] == "general_engine"


def test_the_decorated_entry_point_is_try_decompose():
    assert try_decompose.__wrapped__.__name__ == "try_decompose"
