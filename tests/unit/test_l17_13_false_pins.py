"""The last false PINs of lane/l17: polyesters of ring acids, and the choice of the ester the name is built on.

 "When anions are identical functional class multiplicative nomenclature is used" (the Blue Book):
a diester of one diol and two identical acids is 'ethane-1,2-diyl di(oxolane-2-carboxylate)', not the lateral name
'2-[(oxolane-2-carbonyl)oxy]ethyl oxolane-2-carboxylate'. A ring parent has no principal chain, so the route of the
polyfunctional namer that offers the multiplicative name was never reached for an acid with a ring that is no mancude
carbocycle (a ring ether, a saturated ring with a hydroxy group).

 "All preferred IUPAC names for esters are named by functional class nomenclature" (:31663): a name that
cites the molecule's principal ester as a prefix on a polycyclic parent
('2-{[4-(methoxycarbonyl)phenoxy]carbonyl}naphthalene') is no PIN.

 "Polyester names formed by using substitutive nomenclature, and/or multiplicative nomenclature, and functional
class nomenclature" (:31856): the functional class name is built on the senior ester, and the senior parent hydride
comes before the number of substituents INTRODUCTION,:21551; "The senior ring or ring system has the
greater number of rings",:19461), so a naphthalenecarboxylic acid is senior to a benzoic acid with more substituents.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=C(OCCOC(=O)C1CCCO1)C1CCCO1", "ethane-1,2-diyl di(oxolane-2-carboxylate)"),
    ("O=C(OCCOC(=O)C1CCCCO1)C1CCCCO1", "ethane-1,2-diyl di(oxane-2-carboxylate)"),
    ("O=C(OCCOC(=O)C1CCOCC1)C1CCOCC1", "ethane-1,2-diyl di(oxane-4-carboxylate)"),
    ("O=C(OCCCOC(=O)C1CCCO1)C1CCCO1", "propane-1,3-diyl di(oxolane-2-carboxylate)"),
    ("O=C(OCC(C)OC(=O)C1CCCO1)C1CCCO1", "propane-1,2-diyl di(oxolane-2-carboxylate)"),
    ("O=C(OCCOC(=O)C1CCC(=O)CC1)C1CCC(=O)CC1", "ethane-1,2-diyl bis(4-oxocyclohexane-1-carboxylate)"),
    ("O=C(OCCOC(=O)C1CCC(O)CC1)C1CCC(O)CC1", "ethane-1,2-diyl bis(4-hydroxycyclohexane-1-carboxylate)"),
    ("O=C(OCCOC(=O)c1ccc(OC)cc1)c1ccc(OC)cc1", "ethane-1,2-diyl bis(4-methoxybenzoate)"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_diester_of_a_ring_acid_has_the_functional_class_multiplicative_name(smiles, name, tier):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_the_benzodioxole_diester_is_named_by_the_class_name_and_is_no_pin_by_the_lateral_name():
    """The lateral name that was labelled a PIN is gone: the functional class name of the producer ships at the wider tier
    (a ring system of several rings is named below the PIN there), and the PIN tier declines."""
    smiles = "O=C(OCCOC(=O)c1ccc2OCOc2c1)c1ccc2OCOc2c1"
    assert _row(smiles, "pin")["tier"] != "pin_verified"
    row = _row(smiles, "best-effort")
    assert row["name"] == "ethane-1,2-diyl di(2H-1,3-benzodioxole-5-carboxylate)", row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=C(c1ccc2ccccc2c1)Oc1ccc(C(=O)OC)cc1", "4-(methoxycarbonyl)phenyl naphthalene-2-carboxylate"),
    ("O=C(c1cccc2ccccc12)Oc1ccc(C(=O)OC)cc1", "4-(methoxycarbonyl)phenyl naphthalene-1-carboxylate"),
    ("COC(=O)c1ccc(OC(=O)c2cc3ccccc3c3ccccc23)cc1", "4-(methoxycarbonyl)phenyl phenanthrene-9-carboxylate"),
    ("COC(=O)c1ccc(OC(=O)c2ccc3ccc4cccc5ccc2c3c45)cc1", "4-(methoxycarbonyl)phenyl pyrene-1-carboxylate"),
    ("O=C(c1ccc2ccccc2c1)OCc1ccc(C(=O)OC)cc1", "[4-(methoxycarbonyl)phenyl]methyl naphthalene-2-carboxylate"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_an_ester_cited_as_a_prefix_on_a_polycyclic_parent_is_not_the_name(smiles, name, tier):
    """The ring cascade named these '2-{[4-(methoxycarbonyl)phenoxy]carbonyl}naphthalene' with the PIN label. The probe of
    the decomposition engine now reads that name as no name at the PIN tier too, and the split names the ester by
    functional class: the senior ester is the naphthalenecarboxylic acid, two rings before one,
    , a name OPSIN reads back."""
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_the_probe_of_the_split_reads_the_ester_prefix_name_of_a_polycyclic_parent_as_no_name():
    """The probe name of the decomposition engine is empty where the ring cascade cites the principal ester as a prefix on a
    polycyclic parent, at the PIN tier too; a plain ester keeps its probe name."""
    from orthonym.assembly.ester_prefix_probe import probe_name
    assert probe_name("O=C(c1ccc2ccccc2c1)Oc1ccc(C(=O)OC)cc1", "pin") == ""
    assert probe_name("CCOC(=O)c1ccc2ccccc2c1", "pin") == "ethyl naphthalene-2-carboxylate"


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_wider_tier_keeps_the_pin_of_the_senior_ester(tier):
    """'methyl 4-{[(1-benzofuran-2-carbonyl)oxy]methyl}benzoate' (pin_unverified) was the wider tier's name beside the PIN
    tier's PIN: the producer of independent esters took the acid with the most carbons for the principal one, and that count
    reaches the atoms of the second ester."""
    smiles = "O=C(c1cc2ccccc2o1)OCc1ccc(C(=O)OC)cc1"
    row = _row(smiles, tier)
    assert row["name"] == "[4-(methoxycarbonyl)phenyl]methyl 1-benzofuran-2-carboxylate", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_the_senior_ester_has_the_senior_parent_hydride_before_the_substituents():
    from orthonym.rules.ester_seniority import senior_ester
    mol = Chem.MolFromSmiles("O=C(c1ccc2ccccc2c1)Oc1ccc(C(=O)OC)cc1")
    matches = [tuple(m) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))]
    chosen = senior_ester(mol, matches)
    # the naphthalene-2-carboxylate (carbonyl carbon 1): two rings before one, though the benzoate of the
    # other ester has the same number of substituents beside its ester group
    assert chosen[0] == 1, chosen
