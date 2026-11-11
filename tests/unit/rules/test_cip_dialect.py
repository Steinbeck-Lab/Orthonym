"""The CIP dialect judge (``validation/cip_dialect.py``; still-unnamed PubChem plan of 2026-11-11, Topic 2b).

 "Mancude rings and ring systems" (the Blue Book): "For mancude heterocycles, each duplicate atom is
given an atomic number that is the mean of what it would have if the double bonds were located at each of the possible
positions." OPSIN 2.9.0 lacks the averaging (dan2097/opsin#349), so it reads a correct name as the other configuration
at a unit that rule decides. The best-effort tier ships such a name when the twin with the letters of those units
swapped reads back to the input, labelled ``systematic_verified``; the default tier declines it.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.validation.cip_dialect import cip_dialect_name, dialect_units

pytestmark = pytest.mark.opsin_gate

# the three examples of dan2097/opsin#349
EX1 = "Fc1cccc(F)c1-c1cc2c(nn1)[C@@]1(c3ccnc(Cl)n3)CC[C@@H]2C1"
EX2 = "CCOC(=O)[C@@H]1C2[C@@H]1[C@@H]1C=C[C@H]2[C@@H](C(=O)O)[C@H]1C(=O)OC"
EX3 = "CN1C(=O)[C@H]2[C@H](CC=C3[C@H]2C[C@@]2(Cl)C(=O)N(CBr)C(=O)[C@@]2(Cl)[C@H]3c2ccc(O)cc2)C1=O"
EX1_FLAT = "8-(2-chloropyrimidin-4-yl)-3-(2,6-difluorophenyl)-5,6,7,8-tetrahydro-5,8-methanocinnoline"
EX1_NAME = "(5R,8S)-8-(2-chloropyrimidin-4-yl)-3-(2,6-difluorophenyl)-5,6,7,8-tetrahydro-5,8-methanocinnoline"


def _record(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assign_stereochemistry(mol)
    return mol


def test_example_1_ships_at_best_effort():
    row = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True).name_tiered(EX1)
    assert (row["name"], row["tier"], row["verified"]) == (EX1_NAME, "systematic_verified", "opsin_cip_dialect")


def test_example_1_declines_at_the_default_tier():
    assert Orthonym().name_tiered(EX1)["tier"] != "pin_verified"


def test_the_judge_on_example_1():
    assert dialect_units(_record(EX1)) == {("atom", 14): "S"}
    assert cip_dialect_name(EX1_FLAT, EX1) == (EX1_NAME, "1_units")


@pytest.mark.parametrize("smiles", [EX2, EX3])
def test_examples_2_and_3_have_no_dialect_unit(smiles):
    """OPSIN's reading of these differs by something other than the averaging rule: D is empty, no name."""
    assert dialect_units(_record(smiles)) == {}


def test_no_dialect_unit_without_a_mancude_ring():
    assert dialect_units(_record("C[C@H](O)[C@@H](N)CC")) == {}


def test_declined_when_rdkit_disagrees_at_a_dialect_unit(monkeypatch):
    """The judge removes OPSIN's cross-check exactly at D; rdCIPLabeler must hold it there."""
    import orthonym.validation.cip_dialect as judge
    monkeypatch.setattr(judge, "_rdkit_labels", lambda mol: {("atom", 14): "R"})
    assert cip_dialect_name(EX1_FLAT, EX1) is None


def test_declined_when_the_twin_does_not_read_back(monkeypatch):
    import orthonym.namer as namer
    monkeypatch.setattr(namer, "_shipped_name_round_trip", lambda name, smiles: "failed")
    assert cip_dialect_name(EX1_FLAT, EX1) is None


def test_a_substituent_descriptor_stands_in_front_of_its_prefix():
    """ NAMING OF STEREOISOMERS (the Blue Book): "When they relate to substituent groups, they are cited
    at the front of the corresponding prefix" (CID 176863122)."""
    row = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True).name_tiered("C1CN([C@@H](C2=C1NC=N2)C3=CC4=CC=CC=C4C=N3)C(=O)C5=C6C=CC(=CN6N=C5)C7=NC=CN=C7")
    assert row["verified"] == "opsin_cip_dialect"
    assert row["name"].startswith("[(2S)-2-(isoquinolin-3-yl)-3,7,9-triazabicyclo[4.3.0]nona-1(6),8-dien-3-yl]")


def test_a_block_that_cites_a_locant_twice_is_refused():
    from orthonym.validation.cip_dialect import stereo_locants_once
    assert not stereo_locants_once("(2S,2S,2S)-3-[(4-{2-[4-(butan-2-yl)phenyl]butan-2-yl}phenyl)amino]propane-1,2-diol")
    assert stereo_locants_once(EX1_NAME)


# CID 169166535 (fusion-carbon centre beside an isoquinolin-3-yl and an imidazole carbon) and CID 10929611 (two units):
# the units the judge finds are the ones where OPSIN's reading differs from the input, and rdCIPLabeler, which the
# judge only consults at D, labels the input with the name's letter there (checked here without the judge)
MANCUDE_CONFLICTS = [
    ("CC(C)(C)C1=NC=C(O1)C(=O)N2CCC3=C([C@@H]2C4=CC5=CC=CC=C5C=N4)N=CN3",
     "(2-tert-butyl-1,3-oxazol-5-yl)[(2R)-2-(isoquinolin-3-yl)-3,7,9-triazabicyclo[4.3.0]nona-1(6),8-dien-3-yl]"
     "methanone", {16: ("R", "S")}),
    ("C1C[C@H]2C3=NC4=CC5=C(C=C4N=C3[C@@H]1C6=NC(=C(N=C26)C#N)C#N)N=C7[C@@H]8CC[C@H](C7=N5)C9=NC(=C(N=C89)C#N)C#N",
     None, {26: ("S", "R"), 29: ("R", "S")}),
]


@pytest.mark.parametrize("smiles, name, differ", MANCUDE_CONFLICTS)
def test_a_mancude_conflict_ships_the_name_of_record(smiles, name, differ):
    from rdkit.Chem import rdCIPLabeler
    from orthonym.namer import _validity_gate_name_to_opsin_smiles
    row = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True).name_tiered(smiles)
    assert row["tier"] == "systematic_verified" and row["gate_outcome"] == f"cip_dialect_verified:{len(differ)}_units"
    assert name is None or row["name"] == name

    def labels(mol):
        rdCIPLabeler.AssignCIPLabels(mol)
        return {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    given = Chem.MolFromSmiles(smiles)
    read = Chem.MolFromSmiles(_validity_gate_name_to_opsin_smiles(row["name"]))
    assert Chem.MolToSmiles(given, isomericSmiles=False) == Chem.MolToSmiles(read, isomericSmiles=False)
    match = read.GetSubstructMatch(given, useChirality=False)
    mine, opsin = labels(Chem.Mol(given)), labels(Chem.Mol(read))
    assert {i: (mine[i], opsin[match[i]]) for i in mine if mine[i] != opsin.get(match[i])} == differ
