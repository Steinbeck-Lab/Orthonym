"""``name_fragment_recursively`` keys its caches by the canonical SMILES, taken through the
process-wide memo of ``Chem.CanonSmiles``: the answer for any spelling, valid or not, is what
it was before."""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.perception import molcache

pytestmark = pytest.mark.unit


def test_any_spelling_of_a_cached_fragment_gets_its_cached_name():
    for spelling in ("OCC", "C(C)O", "CCO"):
        assert fn.name_fragment_recursively(spelling) == "ethanol"


def test_an_unreadable_fragment_gives_none_each_time():
    assert fn.name_fragment_recursively("not a smiles") is None
    assert fn.name_fragment_recursively("not a smiles") is None
    assert fn.name_fragment_recursively(None) is None


def test_canonical_form_is_rdkits():
    for spelling in ("OCC", "C(C)O", "c1ccccc1O", "Oc1ccccc1", "CC(=O)OC"):
        assert molcache.canon_smiles(spelling) == Chem.CanonSmiles(spelling)
