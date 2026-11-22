"""The group -CO-CO-NH2 on a ring is the 'oxamoyl' prefix.

 (the Blue Book): "the group -CO-CO-NH2 is named using the prefixes 'amino' and 'oxo' as in method
(1) in; by using the name 'oxamoyl' as in method (2)"; (:32928): "method (1) is
preferred for chains and method (2) for rings and ring systems": '2-oxamoylpyridine-3-carboxylic acid (PIN)' (:32967),
'4-amino-3,4-dioxobutanoic acid (PIN)' (:32965). assembly.acyl_prefix.oxamoyl_prefix. 'oxamoyl (preferred prefix) (full
substitution; see ' (:17784): an N-substituted group is '(methyloxamoyl)', '[methyl(phenyl)oxamoyl]'
(the carbamoyl writer's substituents, '[methyl(phenyl)carbamoyl]':32957), not
'[(methylamino)(oxo)acetyl]'.
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



@pytest.mark.parametrize("smiles,name", [
    ("CNC(=O)C(=O)c1ccc(C(=O)O)cc1", "4-(methyloxamoyl)benzoic acid"),
    ("CN(C)C(=O)C(=O)c1ccc(C(=O)O)cc1", "4-(dimethyloxamoyl)benzoic acid"),
    ("CN(c1ccccc1)C(=O)C(=O)c1ccc(C(=O)O)cc1", "4-[methyl(phenyl)oxamoyl]benzoic acid"),
    ("O=C(O)c1ccc(C(=O)C(=O)NCCO)cc1", "4-[(2-hydroxyethyl)oxamoyl]benzoic acid"),
    # a ring-bearing or stereogenic N-substituent is named as an amide N-substituent, with its descriptors
    #, the Blue Book), on every tier
    ("O=C(O)c1ccc(C(=O)C(=O)NC2CCN(C)CC2)cc1", "4-[(1-methylpiperidin-4-yl)oxamoyl]benzoic acid"),
    ("O=C(O)c1ccc(C(=O)C(=O)N[C@@H](C)c2ccccc2)cc1", "4-{[(1S)-1-phenylethyl]oxamoyl}benzoic acid"),
    ("O=C(O)c1ccc(C(=O)C(=O)N[C@H]2CCCC[C@@H]2C)cc1", "4-{[(1S,2S)-2-methylcyclohexyl]oxamoyl}benzoic acid"),
    # a charge or a label on an N-substituent is the substituent's own concerns the oxamoyl atoms)
    ("O=C(O)c1ccc(C(=O)C(=O)Nc2ccc([N+](=O)[O-])cc2)cc1", "4-[(4-nitrophenyl)oxamoyl]benzoic acid"),
    ("O=C(O)c1ccc(C(=O)C(=O)N[13CH3])cc1", "4-[(13C)methyloxamoyl]benzoic acid"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_substituted_oxamoyl(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "O=C(O)c1ccc([13C](=O)C(=O)NC)cc1",
    "O=C(O)c1ccc(C(=O)C(=O)[15NH]C)cc1",
])
def test_label_on_substituted_group_is_not_oxamoyl(smiles):
    # a nuclide on the oxamoyl atoms themselves: 'oxamoyl' numbers none of them, the Blue Book)
    row = tier_row(smiles, "best-effort")
    assert "oxamoyl" not in (row["name"] or ""), row


@pytest.mark.parametrize("smiles,name", [
    # a ring nitrogen or a hydrazine on the carbonyl is no carbamoyl group: no oxamoyl
    ("O=C(O)c1ccc(C(=O)C(=O)N2CCCCC2)cc1", "4-[oxo(piperidin-1-yl)acetyl]benzoic acid"),
    ("O=C(O)c1ccc(C(=O)C(=O)NN)cc1", "4-[hydrazinyl(oxo)acetyl]benzoic acid"),
])
def test_other_ends_are_not_oxamoyl(smiles, name):
    assert tier_row(smiles, "pin")["name"] == name
