"""Two identical fused ring systems, one aromatic and one with indicated hydrogen, form one ring assembly.

 (the Blue Book): "The maximum number of noncumulative double bonds is then added taking into account
the junction positions. Any remaining saturated ring positions are designated as indicated hydrogen, placed together
with the appropriate locant(s) at the front of the name of the assembly"; '2'H-1,4'a-binaphthalene (PIN)' (:15630).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def test_the_junction_at_a_fusion_atom_takes_indicated_hydrogen():
    assert_pin_at_both_tiers("C1=CC2=CCC=CC2(c2cccc3ccccc23)C=C1", "2'H-1,4'a-binaphthalene")


@pytest.mark.parametrize("smiles,name", [
    ("c1ccc2c(c1)cccc2-c1cccc2ccccc12", "1,1'-binaphthalene"),          # both aromatic: unchanged
    ("C1=Cc2ccccc2C1C1C=Cc2ccccc21", "1H,1'H-1,1'-biindene"),           # both with indicated hydrogen: unchanged
])
def test_the_other_assemblies_keep_their_names(smiles, name):
    assert_pin_at_both_tiers(smiles, name)
