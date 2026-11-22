"""The group -CO-CO-NH2 on a ring is the 'oxamoyl' prefix.

 (the Blue Book): "the group -CO-CO-NH2 is named using the prefixes 'amino' and 'oxo' as in method
(1) in; by using the name 'oxamoyl' as in method (2)"; (:32928): "method (1) is
preferred for chains and method (2) for rings and ring systems": '2-oxamoylpyridine-3-carboxylic acid (PIN)' (:32967),
'4-amino-3,4-dioxobutanoic acid (PIN)' (:32965). assembly.acyl_prefix.oxamoyl_prefix.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name,tier", [
    ("NC(=O)C(=O)c1ccc(C(=O)O)cc1", "4-oxamoylbenzoic acid", "pin"),
    ("NC(=O)C(=O)c1ncccc1C(=O)O", "2-oxamoylpyridine-3-carboxylic acid", "best-effort"),
    ("NC(=O)C(=O)C1CCC(C(=O)O)CC1", "4-oxamoylcyclohexane-1-carboxylic acid", "best-effort"),
    # on a chain the group is part of the chain (method (1))
    ("NC(=O)C(=O)CC(=O)O", "4-amino-3,4-dioxobutanoic acid", "pin"),
    # the amide is the suffix: no prefix
    ("NC(=O)C(=O)c1ccccc1", "2-oxo-2-phenylacetamide", "pin"),
])
def test_oxamoyl_prefix(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] in ("pin_verified", "pin_unverified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "NC(=O)[13C](=O)c1ccc(C(=O)O)cc1",
    "[15NH2]C(=O)C(=O)c1ccc(C(=O)O)cc1",
])
def test_labelled_group_is_not_oxamoyl(smiles):
    # a nuclide on the group: 'oxamoyl' numbers none of its atoms, the Blue Book)
    row = tier_row(smiles, "best-effort")
    assert "oxamoyl" not in (row["name"] or ""), row

