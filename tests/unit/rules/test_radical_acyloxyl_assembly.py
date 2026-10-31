"""Several acyloxyl radical centres on one polyacyl group are an assembly of 'oxyl' units.

 "ASSEMBLIES OF PARENT RADICALS" (the Blue Book) names radical centres in different parts of a structure
as an assembly of identical units on a multivalent group ('(2,4-dimethylpentane-2,4-diyl)bis(oxyl) (PIN)',:40749).
The unit of an acyloxyl radical is 'oxyl', as in 'butanoyloxyl (PIN)',:40718); the central group is the
polyacyl group of the polyacid, 'benzene-1,4-dicarbonyl (PIN)',:40618). A substituted central group is a
compound prefix and is enclosed, as '(chloroacetyl)oxyl (PIN)',:40677; the example is:40694, in the section
``### **** Polyamine, polyimine and polyamide radicals`` of the OCR); an unsubstituted chain polyacyl group takes
no enclosing marks: 'butanedioylbis(oxy-2,1-phenylene)' (:31805, ``## **** Polyester names formed by
multiplicative nomenclature``).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,expected", [
    ("[O]C(=O)c1ccc(cc1)C(=O)[O]", "(benzene-1,4-dicarbonyl)bis(oxyl)"),
    ("[O]C(=O)c1ccc(cc1)-c1ccc(cc1)C(=O)[O]", "([1,1'-biphenyl]-4,4'-dicarbonyl)bis(oxyl)"),
    ("[O]C(=O)CCC(=O)[O]", "butanedioylbis(oxyl)"),
    ("[O]C(=O)c1cc(C(=O)[O])cc(C(=O)[O])c1", "(benzene-1,3,5-tricarbonyl)tris(oxyl)"),
    # a substituted central group is enclosed whether or not its name holds a locant ('(chloroacetyl)oxyl',:40694)
    ("[O]C(=O)C(C)(C)C(=O)[O]", "(dimethylpropanedioyl)bis(oxyl)"),
    ("[O]C(=O)C(C)CC(=O)[O]", "(methylbutanedioyl)bis(oxyl)"),
    ("[O]C(=O)C(N)CC(=O)[O]", "(aminobutanedioyl)bis(oxyl)"),
    ("[O]C(=O)C(Cl)CC(=O)[O]", "(chlorobutanedioyl)bis(oxyl)"),
    # an unsubstituted chain polyacyl group stays bare
    ("[O]C(=O)CC(=O)[O]", "propanedioylbis(oxyl)"),
    ("[O]C(=O)CCCC(=O)[O]", "pentanedioylbis(oxyl)"),
    ("[O]C(=O)C(=O)[O]", "oxalylbis(oxyl)"),
])
def test_an_acyloxyl_assembly_is_named_on_its_polyacyl_group(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    ("O=C([O])c1ccccc1", "benzoyloxyl"),
    ("CC(C)([O])CC(C)(C)[O]", "(2,4-dimethylpentane-2,4-diyl)bis(oxyl)"),
    ("[O]c1ccc([O])cc1", "(benzene-1,4-diyl)bis(oxyl)"),
])
def test_the_other_oxyl_names_are_unchanged(smiles, expected):
    assert Orthonym().name_tiered(smiles)["name"] == expected


def test_a_stereo_prefix_keeps_the_assembly_below_the_pin():
    # the stereodescriptor inside the prefix has no derived PIN placement yet: named at a general tier only
    smiles = "C(=C/C(=O)[O])\\C(=O)[O]"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    # the parentheses of a stereodescriptor count in the nesting order, so the enclosure of the central group is the
    # bracket (``### **** Multiple types of enclosing marks``, ****, the Blue Book); the
    # lexical spelling check agrees, and it rejects the double parentheses this test used to expect
    assert row["name"] == "[(2E)-but-2-enedioyl]bis(oxyl)" and not row["is_pin"], row
    from rdkit import Chem
    from orthonym.validation.pin_spelling import check_pin_spelling
    mol = Chem.MolFromSmiles(smiles)
    assert check_pin_spelling(mol, row["name"], strict=True) == []
    assert [f.rule for f in check_pin_spelling(mol, "((2E)-but-2-enedioyl)bis(oxyl)", strict=True)] == ["P-16.5.4"]
