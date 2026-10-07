"""Lane L2 proper fix: rows kept from the (d) test file when that feature was taken
out of the lane (it is rebuilt later with substituent groups keyed by structure).

The tetrazolo rows pin the kept M1 behaviour: (the Blue Book) "Locants that
describe structural features of components, such as positions of heteroatoms, are kept with
the name of the component and are enclosed within square brackets" -- the tetrazolo locant set
stays on PARENT fusion names (``describe_structure=True``).

The multiplier rows pin the basic multiplier for fused prefixes that the lane no longer
decides: retained fused names are simple components (a)), and a fused prefix with
no writer record takes main's 'di'."""
import pytest

from orthonym import Orthonym
from orthonym.assembly.naming_utils import format_substituent_prefix
from tests.support.rt_assert import name_is_rt_exact


def test_an_unrecorded_fused_name_takes_the_basic_multiplier():
    assert format_substituent_prefix("imidazo[1,2-a]pyridin-3-yl", [3, 5], 2) == \
        "3,5-di(imidazo[1,2-a]pyridin-3-yl)"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1cc(-c2c[nH]c3ccccc23)cc(-c2c[nH]c3ccccc23)c1", "3,5-di(1H-indol-3-yl)benzoic acid"),
    ("OC(=O)c1cc(-c2ccc3ccccc3n2)cc(-c2ccc3ccccc3n2)c1", "3,5-di(quinolin-2-yl)benzoic acid"),
])
def test_retained_fused_prefixes_take_the_basic_multiplier(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert (row.get("name"), row.get("tier")) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("c1cc2nnnn2nc1", "[1,2,3,4]tetrazolo[1,5-b]pyridazine"),
    ("c1cnc2nnnn2c1", "[1,2,3,4]tetrazolo[1,5-a]pyrimidine"),
    ("Clc1ccc2nnnn2n1", "6-chloro[1,2,3,4]tetrazolo[1,5-b]pyridazine"),
])
def test_tetrazolo_parent_keeps_the_structural_locant_set(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert (row.get("name"), row.get("tier")) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)
