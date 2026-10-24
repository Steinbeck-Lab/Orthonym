"""The von Baeyer cage collector names EVERY non-ring branch or declines the cage (Task 3a).

``polycyclic.get_polycyclic_substituents`` used to skip a non-carbon branch and to name a
heteroatom-bearing carbon branch by its carbon count ('1-bromoadamantane' came out as
'tricyclo[3.3.1.1^3,7]decane', a bromomethyl branch as 'methyl'). The collector now declines such
a cage, and builds the name through the shared substituent cascade (``rules.cage_substituents``).

 (the Blue Book): "Halogen compounds... are named in two ways: (1) by substitutive
nomenclature, using the prefixes 'bromo', 'chloro', 'fluoro', and 'iodo'...". "Nitro and
nitroso compounds" (the Blue Book,:25935): "Compounds containing the -NO2 or -NO group are
named by means of the prefixes 'nitro' and 'nitroso'...".

The adamantane rows are named at the strict path and declined by the default tier, because
adamantane is a retained parent "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES",
the Blue Book: "The retained names adamantane and cubane are used in general nomenclature
and as preferred IUPAC names."): their von Baeyer names are not PINs (Task 12 builds the retained
names). The nortricyclene rows have no retained name and no fusion reading, so the von Baeyer
name with its prefix is their PIN.
"""
import pytest

from tests.support.default_tier import assert_default_tier_declines, strict_path_row
from tests.support.large_polycycles import be, ladder, pin
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate
L = ladder()
ADAMANTANES = ["V01_1bromoadamantane_F6", "V02_1fluoroadamantane_F6", "V03_1nitroadamantane_F6",
               "V04_1bromomethyladamantane_F6", "V05_BB48753_constitution_F6",
               "A07_1bromo3chloroadamantane", "V08_1methylsulfanyladamantane_F6"]
NORTRICYCLENES = [("ClC1C2CC3C1C3C2", "3-chlorotricyclo[2.2.1.0^2,6]heptane"),
                  ("BrC1C2CC3C1C3C2", "3-bromotricyclo[2.2.1.0^2,6]heptane")]


@pytest.mark.unit
@pytest.mark.parametrize("label", ADAMANTANES)
def test_every_branch_is_named_and_the_strict_name_reads_back(label):
    row = strict_path_row(L[label])
    assert row["name"] and row["tier"] != "abstain", row
    assert_full_rt(row["name"], L[label])


@pytest.mark.unit
@pytest.mark.parametrize("label", ADAMANTANES)
def test_default_tier_declines_a_name_of_a_retained_parent(label):
    assert_default_tier_declines(L[label])


@pytest.mark.unit
@pytest.mark.parametrize("label", ADAMANTANES)
def test_best_effort_gives_the_strict_name(label):
    assert be(L[label])["name"] == strict_path_row(L[label])["name"]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,token", [
    (L["V01_1bromoadamantane_F6"], "1-bromo"),                   #:25768
    (L["V02_1fluoroadamantane_F6"], "1-fluoro"),
    (L["V03_1nitroadamantane_F6"], "1-nitro"),                   #:25935
    (L["V04_1bromomethyladamantane_F6"], "1-(bromomethyl)"),
    (L["V08_1methylsulfanyladamantane_F6"], "1-(methylsulfanyl)"),
    ("BrCOC12CC3CC(CC(C3)C1)C2", "1-(bromomethoxy)"),            # the alkoxy-table hole
])
def test_prefix_tokens(smiles, token):
    row = strict_path_row(smiles)
    assert (row["name"] or "").startswith(token), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", NORTRICYCLENES)
def test_a_cage_without_a_retained_name_gets_its_pin(smiles, expected):
    r = pin(smiles)
    assert (r["name"], r["tier"]) == (expected, "pin_verified"), r
    assert_full_rt(expected, smiles)


@pytest.mark.unit
def test_a_branch_the_cascade_cannot_spell_declines_the_cage():
    # an isopropoxy branch: the alkoxy table spells straight chains of one to six carbons only, and
    # a plain alkoxy branch is not the cascade's. Fail closed: no name, never a name that drops it.
    row = strict_path_row("CC(C)OC12CC3CC(CC(C3)C1)C2")
    assert row["tier"] == "abstain", row


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "CN(C)C12CC3CC(CC(C3)C1)C2",          # an amine: the principal group of the molecule
    "CC(=O)OC12CC3CC(CC(C3)C1)C2",        # an ester
    "OOC12CC3CC(CC(C3)C1)C2",             # a hydroperoxide
    "OCC12CC3CC(CC(C3)C1)C2",             # an alcohol: the chain is the parent, (adamantan-1-yl)methanol
    "COC(=O)C12CC3CC(CC(C3)C1)C2",        # a carbon-attached ester
    "O=C(N)C12CC3CC(CC(C3)C1)C2",         # an amide
])
def test_a_branch_that_can_be_the_principal_group_is_not_spelled_as_a_prefix(smiles):
    """A prefix form of a group that can be the suffix ('carbamoyl', '(dimethylamino)') on a cage
    that has no senior group is not the preferred name, the Blue Book, Table 4.1:
    "classes expressed by suffixes (classes 1 through 20)"): the collector declines the cage and the
    producers that choose the principal group name the molecule."""
    from rdkit import Chem

    from orthonym.rules.polycyclic import name_polycyclic_complete
    assert name_polycyclic_complete(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.unit
def test_a_branch_with_a_heteroatom_outside_the_prefix_only_set_is_declined():
    """Not a principal-group case: silicon is none of the groups that are only ever cited as a prefix
    (halogen, nitro, ether oxygen, thioether sulfur; ``rules.cage_substituents``), and no rule here
    says how to spell it, so the collector declines the cage as it did before this task."""
    from rdkit import Chem

    from orthonym.rules.polycyclic import name_polycyclic_complete
    assert name_polycyclic_complete(Chem.MolFromSmiles("C[Si](C)(C)C12CC3CC(CC(C3)C1)C2")) is None


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("BrC1CC2CCC1C2", "2-bromobicyclo[2.2.1]heptane"),           # the bicyclo path was correct
    ("CC12CC3CC(CC(C3)C1)C2", "1-methyltricyclo[3.3.1.1^3,7]decane"),
    ("OC12CC3CC(CC(C3)C1)C2", "tricyclo[3.3.1.1^3,7]decan-1-ol"),
])
def test_controls_unchanged(smiles, expected):
    assert strict_path_row(smiles)["name"] == expected
