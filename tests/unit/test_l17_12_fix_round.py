"""Fix round of the merged lane/l17 (item 12): the principal group of an acyclic oxoacid ester beside an amide.

 SENIORITY ORDER FOR CLASSES, Table 4.1 (:18158): "9 Esters (functional class names are given to noncyclic esters;
lactones and other cyclic esters are named as heterocycles; see 16 below)" ranks above "11 Amides"; class 7 acids rank
above both, and a partial ester of an oxoacid ('hydrogen' or 'dihydrogen' in its name,,:35916 'methyl dihydrogen
phosphate (PIN)') is named as that ester. The principal group of the perception did not rank these esters above the amides
('N-[2-(sulfooxy)ethyl]tetradecanamide'), and the amido prefix was kept from them by a SMARTS in ``n_acyl_amido``.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# ---- Part B1: an acyclic ester of a phosphorus or sulfur oxoacid is senior to an amide -------------------------------

# (smiles, the PIN): the ester is the parent, the amide a prefix of its organyl group
_ESTER_OVER_AMIDE = [
    ("CCCCCCCCCCCCCC(=O)NCCOS(=O)(=O)O", "2-tetradecanamidoethyl hydrogen sulfate"),
    ("CC(=O)NCCOP(=O)(O)O", "2-acetamidoethyl dihydrogen phosphate"),                  # the partial esters
    ("CC(=O)NCCOP(=O)(O)OC", "2-acetamidoethyl methyl hydrogen phosphate"),
    ("CC(=O)Nc1ccc(OS(=O)(=O)O)cc1", "4-acetamidophenyl hydrogen sulfate"),
    ("CC(=O)NCCOS(=O)(=O)OC", "2-acetamidoethyl methyl sulfate"),
    ("CC(=O)NCCOP(=O)(OC)OC", "2-acetamidoethyl dimethyl phosphate"),
    ("CCOP(=O)(OCC)CN(C)C(C)=O", "diethyl [(N-methylacetamido)methyl]phosphonate"),   # class 7c, a phosphonate
    ("CC(=O)NCCCP(=O)(OC)OC", "dimethyl (3-acetamidopropyl)phosphonate"),
    ("CC(=O)NCCOP(=O)(OC)c1ccccc1", "2-acetamidoethyl methyl phenylphosphonate"),
]


@pytest.mark.parametrize("smiles,name", _ESTER_OVER_AMIDE)
def test_an_oxoacid_ester_is_the_parent_over_an_amide(smiles, name):
    """ (:18182,:18184): class 9 esters rank above class 11 amides; (:35916) names the esters of
    the oxoacids in functional class nomenclature. The amide is cited as the amido prefix,:32991)."""
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(name, smiles)


def test_the_principal_group_ranks_the_acyclic_oxoacid_esters_above_the_amides():
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group

    def principal(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return get_principal_group(mol, detect_functional_groups(mol))[0]

    assert principal("CC(=O)NCCOS(=O)(=O)O") == "sulfate_monoester"
    assert principal("CC(=O)NCCOP(=O)(OC)OC") == "phosphate_triester"
    assert principal("CC(=O)NCCOP(=O)(O)OC") == "phosphate_diester"
    assert principal("CCOP(=O)(OCC)CN(C)C(C)=O") == "phosphonate_diester"
    assert principal("CC(=O)NCCOP(=O)(O)O") == "phosphate_monoester"
    # a carboxylic ester (class 7a) and a carbamate (class 7b) are senior to the esters of the acids of class 7d
    assert principal("CC(=O)OCCOS(=O)(=O)O") == "ester"
    assert principal("CC(=O)NCCOP(=O)(OC)OC") != "secondary_amide"
    # the cyclic esters are heterocycles (Table 4.1, class 9 note), below the amides
    assert principal("CC(=O)NCCOP1(=O)OCCO1") == "secondary_amide"
    assert principal("CC(=O)NCC1COP(=O)(O)O1") == "secondary_amide"
    assert principal("CC(=O)N(C)CC1CCOS1(=O)=O") == "tertiary_amide"
    assert principal("CC(=O)N(C)CC1CCOC1=O") == "tertiary_amide"
    # no amide: the sultone is the senior group of the molecule, as it was
    assert principal("OCC1CCOS1(=O)=O") == "sulfonic_ester"


def test_an_ester_that_cannot_be_named_in_functional_class_keeps_the_head_group():
    """The class is offered, not returned: with a stereo layer the functional-class namer is not offered
    (``rules.phosphorus._OFFER_STEREO_OWNERS``) and the amide stays the principal group."""
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles("CC(=O)N[C@@H](C)COP(=O)(OC)OC")
    assert get_principal_group(mol, detect_functional_groups(mol))[0] == "secondary_amide"


@pytest.mark.parametrize("smiles", [
    "CC(=O)NCCOP(=O)(O)O", "CC(=O)NCCOS(=O)(=O)[O-]", "CC(=O)NCCOP(=O)([O-])[O-]", "CC(=O)NCCCOS(=O)(=O)[O-]",
])
def test_the_amido_prefix_of_an_oxoacid_ester_anion_is_cited(smiles):
    """The anion is class 4 of Table 4.1 (:18167), above the amides: '2-acetamidoethyl sulfate (PIN)'."""
    from orthonym.assembly.n_acyl_amido import group_is_principal
    mol = Chem.MolFromSmiles(smiles)
    amide = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[NX3]"))
    assert not group_is_principal(mol, set(amide))
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert "acetamido" in row["name"] and row["tier"] == "pin_verified", row


def test_the_cyclic_esters_beside_an_amide_name_the_amide_as_the_parent():
    """A lactone, a sultone and a cyclic phosphate are heterocycles (Table 4.1, class 9 note,:18182) and class 16
    pseudoketones,:29314): the amide (class 11) is the parent, the ring a prefix of it."""
    row = _row("CC(=O)N(C)CC1CCOC1=O", "pin")
    assert row["name"] == "N-methyl-N-[(2-oxooxolan-3-yl)methyl]acetamide", row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], "CC(=O)N(C)CC1CCOC1=O")
    for smiles in ("CC(=O)N(C)CC1CCOS1(=O)=O", "CC(=O)NCC1COP(=O)(O)O1", "CC(=O)NCCOP1(=O)OCCO1"):
        row = _row(smiles, "best-effort")
        assert row["name"] and "acetamide" in row["name"] and row["name"].startswith("N-"), row
        assert row["tier"] != "pin_verified", row
        assert_full_rt(row["name"], smiles)


def test_the_systematic_name_of_a_dipeptide_is_a_pin_of_the_project():
    """ (:50943) identifies no PIN for the semisystematic names of its chapter (``alanylaspartylglutamic acid``);
    the systematic substitutive name of a dipeptide is the expected PIN,:54247), so ``in_peptide`` is no rule
    of about the PIN tier but a scoping of the chain amido pass (its docstring has the measurement)."""
    from orthonym.assembly.n_acyl_amido import in_peptide
    smiles = "CC(=O)NCC(=O)NCC(=O)O"
    assert in_peptide(Chem.MolFromSmiles(smiles))
    row = _row(smiles, "pin")
    assert row["name"] == "(2-acetamidoacetamido)acetic acid", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)
