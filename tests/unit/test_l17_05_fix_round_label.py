"""The label of the method (2) polyester names is the rule of a polyol that is one multivalent organyl group (lane/l17 items 2 and 5).

 "Polyesters formed from a single 'alcoholic' component" (the Blue Book): "Esters derived from a single
'polyhydroxylic' component are named by placing the name of the 'polyhydroxylic' component denoted by a multivalent organyl
group (alkyl, aryl, etc.) in front of the name(s) of the acid component"; (:31831) "When anions are
different, two methods are used" -- "Method (1) generates preferred IUPAC names but names formed by using method (2) are
acceptable in general nomenclature" -- is the rule of such a polyol only. "Polyester names formed by using
substitutive nomenclature, and/or multiplicative nomenclature, and functional class nomenclature" (:31856): "Polyesters that
cannot be named by functional class multiplicative nomenclature as described above, are named by using substitutive
nomenclature", the PINs '2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate (PIN)' (:31899) and '1-{2-[2-(acetyloxy)-1-
(formyloxy)ethyl]cyclohexyl}-2-(propanoyloxy)ethyl butanoate (PIN)'.

Item 2 labelled every split polyol ester whose alcohol part carries an ester of another anion as method (2); that is wrong for
a polyol of unequal arms or of a ring and its side chain (item 5 names those substitutively, and the merged PIN tier abstained
on the book's row). The label asks the one reading of the rule of the spelling check of the polyester
(``validation.spelling.checks_polyol_ester.polyol_group_kind``).
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# ---- Part A: the polyol of a method (2) name is one multivalent organyl group ---------------------------------------


def _first_ester(mol):
    match = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))[0]
    return match[3], match[2]


@pytest.mark.parametrize("smiles,one_group", [
    ("CC(=O)OCCOC(=O)CCl", True),                          # ethane-1,2-diyl: a chain
    ("CC(=O)OCC(OC(=O)CCl)COC(C)=O", True),               # propane-1,2,3-triyl
    ("CC(=O)Oc1ccc(OC(=O)C(Cl)Cl)cc1", True),             # 1,4-phenylene 's own example)
    ("CC(=O)OCc1ccccc1COC(=O)CCl", True),                 # 1,2-phenylenebis(methylene): identical arms
    ("CCC(=O)OCCc1ccccc1COC(C)=O", False),                # an ethyl and a methyl arm: no multivalent group
    ("CC(=O)Oc1ccccc1CCOC(=O)CCl", False),                # an ester on a ring and one on its side chain
])
def test_the_polyol_of_a_method_two_name_is_one_multivalent_group(smiles, one_group):
    from orthonym.rules.esters import polyol_is_one_multivalent_group
    mol = Chem.MolFromSmiles(smiles)
    alkyl_c, ester_o = _first_ester(mol)
    assert polyol_is_one_multivalent_group(mol, alkyl_c, ester_o) is one_group


def test_the_book_row_of_unequal_arms_keeps_its_pin_beside_the_item_two_label():
    """ (:31899): the name of the unequal-arms polyol, acetate and propanoate, is the substitutive PIN; the
    label of method (2) (``rules.esters.record_split_polyol_mixed_anion_non_pin``) is the rule of a multivalent group only."""
    row = _row("CCC(=O)OCCc1ccccc1COC(C)=O", "pin")
    assert row["name"] == "2-{2-[(acetyloxy)methyl]phenyl}ethyl propanoate", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row


@pytest.mark.parametrize("smiles,name", [
    ("O=NOCCc1ccccc1COC(C)=O", "{2-[2-(nitrosooxy)ethyl]phenyl}methyl acetate"),
    ("CC(=O)OCCc1ccccc1CO[N+](=O)[O-]", "2-{2-[(nitrooxy)methyl]phenyl}ethyl acetate"),
    ("CC(=O)Oc1ccccc1CCOC(=O)CCl", "2-[2-(acetyloxy)phenyl]ethyl chloroacetate"),
])
def test_an_ester_of_a_polyol_of_no_one_group_is_the_substitutive_pin(smiles, name):
    """Acetic acid ranks above nitrous acid (the anion of the principal group,: 'butanedioic acid is
    preferred to propanedioic acid'), the substituted acid above the unsubstituted one ('phenyl 3-(benzoyloxy)benzoate
    (PIN)'): the name cites the junior ester as an acyloxy prefix, whichever route (the ester route of the pipeline or the
    split of the decomposition) built it."""
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles", [
    "CC(=O)OCCOC(=O)CCl", "CC(=O)OCC(OC(=O)CCl)COC(C)=O", "CC(=O)Oc1ccc(OC(=O)C(Cl)Cl)cc1", "CC(=O)OCOC=O",
    "CC(=O)OCCCOC(=O)CCl",
])
def test_the_method_two_names_of_a_multivalent_polyol_stay_below_the_pin(smiles):
    """The PIN of these is method (1), '1,4-phenylene acetate dichloroacetate' (:31840), which OPSIN 2.9.0 does not read;
    the best-effort tier names them by method (2), below the PIN."""
    row = _row(smiles, "best-effort")
    assert row["name"] and not row["is_pin"] and row["tier"] != "pin_verified", row


