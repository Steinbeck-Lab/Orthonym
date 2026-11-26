"""An ester of a carboxylic acid on a ring nitrogen is named as a traditional ester "Pseudoesters",
, the Blue Book: "When 'E', in R-CO-O-E, is an element from the following list: B, Al, In, Ga, Tl, Si,
Ge, Sn, Pb, N(cyclic), P, As, Sb, Bi, the pseudoester is named as a traditional ester, unless... salts > acids >
anhydrides > esters"; 'borinan-1-yl acetate (PIN)', 'phosphinan-1-yl benzoate (PIN)'). An open-chain N is another class:
'1-[(dimethylamino)oxy]ethan-1-one (PIN)' (:32068). The ring_n_pseudoester handler offers the ester name; the spelling
check (validation/spelling/checks_ring_n_pseudoester.py) lowers the label of an acyloxy-prefix name.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import registered_rules
from orthonym.validation.spelling.checks_ring_n_pseudoester import ring_n_pseudoester_prefix_check
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)ON1C(=O)CCC1=O", "1-(acetyloxy)pyrrolidine-2,5-dione"),
    ("C1CC1C(=O)ON2C=CC=CC2=S", "1-[(cyclopropanecarbonyl)oxy]pyridine-2(1H)-thione"),
    ("NC(=O)ON1CCCCC1", "1-[(piperidin-1-yl)oxy]formamide"),
])
def test_the_acyloxy_prefix_name_is_flagged(smiles, name):
    failure = ring_n_pseudoester_prefix_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-65.6.3.4.1", name


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)ON1C(=O)CCC1=O", "2,5-dioxopyrrolidin-1-yl acetate"),
    ("OC(=O)CCC(=O)ON1C(=O)CCC1=O", "4-[(2,5-dioxopyrrolidin-1-yl)oxy]-4-oxobutanoic acid"),   # an acid outranks it
    ("CC(=O)ON(C)C", "1-[(dimethylamino)oxy]ethan-1-one"),                                  # open-chain N,:32068
    ("CC(=O)OC1CC(C)(C)N(OC(=O)c2ccccc2)C(C)(C)C1",
     "1-(benzoyloxy)-2,2,6,6-tetramethylpiperidin-4-yl acetate"),                            # another ester: not decided
    ("CC(=O)O[n+]1ccccc1", "1-(acetyloxy)pyridin-1-ium"),                                    # a cation outranks it
])
def test_ester_names_and_other_classes_pass(smiles, name):
    assert ring_n_pseudoester_prefix_check(Chem.MolFromSmiles(smiles), name) is None, name


def test_the_check_is_registered():
    assert "P-65.6.3.4.1" in registered_rules()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("CC(=O)ON1C(=O)CCC1=O", "2,5-dioxopyrrolidin-1-yl acetate"),
    ("C1CCC(CC1)C(=O)ON2C(=O)CCC2=O", "2,5-dioxopyrrolidin-1-yl cyclohexanecarboxylate"),
    ("CC(=O)ON1CCCCC1", "piperidin-1-yl acetate"),
    ("O=C(ON1C(=O)c2ccccc2C1=O)C=Cc1ccccc1", "1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl 3-phenylprop-2-enoate"),
    ("NC(=O)ON1CCCCC1", "piperidin-1-yl carbamate"),
    ("CC(=O)OP1CCCCC1", "phosphinan-1-yl acetate"),
])
def test_the_ester_on_a_ring_n_is_the_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["C1CC(=O)N(C1=O)OC(=O)C2=CC(=CC=C2)[Hg]Cl",
                                    "C1[C@@H](CC[C@H](C1)C(=O)ON2C(=O)CCC2=O)CN3C(=O)C=CC3=O"])
def test_molecules_the_ester_builder_cannot_name_keep_their_best_effort_name(smiles):
    breadth = name_breadth(smiles)
    assert breadth.get("name") and breadth.get("tier") not in ("abstain", None), breadth
    assert not name_default(smiles).get("is_pin")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "CC(C)(C)OC(=O)NCC(=O)ON1C(=O)CCC1=O",     # a carbamate ester beside it: which ester names it is not decided here
    "COC(=O)CCC(=O)ON1C(=O)CCC1=O",            # a carboxylic ester beside it
])
def test_a_second_ester_leaves_the_ring_n_ester_name_unbuilt(smiles):
    res = name_default(smiles)
    assert not (res.get("is_pin") and "2,5-dioxopyrrolidin-1-yl" in (res.get("name") or "").split(" ")[0]), res


@pytest.mark.opsin_gate
def test_a_fused_ring_n_keeps_its_fusion_name():
    """The ring producers' fusion name, not a von Baeyer prefix of the ester ('...azatricyclo[6.3.1.0^4,12]...')."""
    res = name_breadth("CC(=O)ON1C(=O)c2cccc3cccc1c23")
    assert res.get("name") and "^" not in res["name"], res

