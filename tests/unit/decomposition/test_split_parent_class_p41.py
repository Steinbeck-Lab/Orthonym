"""A bond split never names a parent whose principal characteristic group is junior to a
class present elsewhere in the molecule.

 (the Blue Book '7 Acids',:18182 '9 Esters',:18184 '11 Amides'): an
acid is senior to an ester and to an amide. Splitting the amide C-N bond of a molecule that
also carries a carboxylic acid and naming '...carboxamide' with a 'carboxymethyl' prefix
gives a name that reads back to the molecule but is not its PIN (the acid is the parent,
the amide a carbamoyl prefix). The split declines; the best-effort tier keeps the acid
parent name it gives today (RT-exact)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ACID_AND_AMIDE = [
    # an amide split of a molecule with a carboxylic acid elsewhere
    "C1CCC(C1)SC2=C(C=CC(=N2)N3CCC[C@H](C3)CC(=O)O)C(=O)NC4CCCCC4",
    "C1CCC(C1)SC2=C(C=CC(=N2)N3CCC[C@H](C3)CC(=O)O)C(=O)NC4C5CCC6C4C[C@](C5)(CC6)OC(F)F",
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="split-p41"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("smiles", ACID_AND_AMIDE)
def test_the_amide_is_not_the_parent_when_an_acid_is_present(smiles):
    row = _row(smiles, "pin")
    name = row.get("name") or ""
    assert not name.endswith("carboxamide"), (name, row["tier"])
    assert row["tier"] == "abstain", row      # the acid-parent name is not certified a PIN
    be = _row(smiles, "best-effort")
    assert be["tier"] != "abstain" and name_is_rt_exact(be["name"], smiles), be
    assert be["name"].endswith("acetic acid"), be["name"]


def test_an_acyl_floated_onto_a_senior_amine_fragment_is_kept():
    # The amide assembler floats the acyl onto an amine fragment that is not a simple
    # amine; the parent is then that fragment ('...-2-aminoethane-1-sulfonic acid', a
    # sulfonic acid, class 7,:18170), so the split is not junior and stays.
    smiles = ("C[C@H](CCC(=O)NCCS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)"
              "CC[C@]4(C)[C@H]3CC[C@]12C")
    be = _row(smiles, "best-effort")
    assert be["name"].endswith("-2-aminoethane-1-sulfonic acid"), be["name"]
    assert name_is_rt_exact(be["name"], smiles)


@pytest.mark.parametrize("smiles,senior", [
    ("NCC1CCOC1=O", "primary_amine"),     # a lactone is a heterocycle class 9,:18182)
    ("NCC1CCNC1=O", "primary_amine"),     # a lactam likewise (class 11,:18184)
    ("CCOC(=O)CCN", "ester"),             # an acyclic ester
])
def test_cyclic_esters_and_amides_are_not_the_senior_class(smiles, senior):
    from rdkit import Chem
    from orthonym.decomposition.engine import _molecule_principal_group
    assert _molecule_principal_group(Chem.MolFromSmiles(smiles)) == senior


@pytest.mark.parametrize("smiles,ending", [
    # an anion is senior to every suffix class classes 4-6,:18158-:18168): the
    # glycinate parent stays
    ("C[C@H](CCC(=O)NCC(=O)[O-])[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)"
     "[C@H]3CC[C@]12C", "glycinate"),
    # primary and secondary alcohols are one class: the N-acyl lactam name stays
    ("CC[C@H](C)C[C@H](C)C[C@H](CO)C[C@@H](C)/C=C(C)/C=C(C)/C=C/C=C/C(=O)N1CCCC(O)C1=O",
     "-3-hydroxypiperidin-2-one"),
])
def test_ions_and_one_class_subtypes_do_not_rule_out_a_split(smiles, ending):
    be = _row(smiles, "best-effort")
    assert be["name"].endswith(ending), be["name"]
    assert name_is_rt_exact(be["name"], smiles)


def test_the_cut_group_is_not_a_class_elsewhere():
    # the amide a split cuts is the split's own group; only the other groups can rule
    # the split out (the heptanoyl pyrrolizinone keeps its acyl-on-amine split)
    from rdkit import Chem
    from orthonym.decomposition.engine import _molecule_principal_group
    mol = Chem.MolFromSmiles("CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12")
    carbonyl = 6
    assert mol.GetAtomWithIdx(carbonyl).GetSymbol() == "C"
    assert _molecule_principal_group(mol) == "secondary_amide"
    assert _molecule_principal_group(mol, {carbonyl}) != "secondary_amide"
