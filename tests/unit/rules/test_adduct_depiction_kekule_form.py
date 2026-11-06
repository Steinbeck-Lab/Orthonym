"""An adduct name that OPSIN reads as another Kekule structure of the drawn ring system is the drawn structure (lane l18 item 4).

The component name of a non-aromatic mancude ring system ('cyclopenta[b]phosphol', 'pentalene') is read by OPSIN with
the double bonds placed as it likes, so the canonical SMILES of the read-back differs from the drawn one while both
are one structure. Maximum number of noncumulative double bonds (the Blue Book): "The names of
polycyclic fused ring systems are considered to correspond to the structure with the maximum number of noncumulative
double bonds consistent with the appropriate bonding number of the skeletal atoms": the name fixes the bonding number
of every atom, not the place of the double bonds. The test of ``rules.adducts._parse_reproduces_depiction`` stays
strict for every other difference (position of a substituent, hydrogen count, charge, stereo, and the bonding number
of an atom: the lambda4-sulfur drawing of a thiophene, a metal-metal bond order). Before the change the Zr adducts
below had no name (the earlier von Baeyer component names, with their explicit double-bond locants, had read back
exactly). No PIN: (the Blue Book).
"""
import pytest
from rdkit import Chem

from orthonym.rules.adducts import _same_up_to_kekule_form
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

ROWS = [
    ("CC1=C2C(=CC(=P2)C)C(=C1)C3=CC=CC=N3.[Cl-].[Cl-].[Zr+2]",
     "2-(2,6-dimethylcyclopenta[b]phosphol-4-yl)pyridine—chloride—zirconium(2+) (1/2/1)"),
    ("CSC1=CC2=C(C=C(C2=C1)SC)C3=CC=CO3.CSC1=CC2=C(C=C(C2=C1)SC)C3=CC=CO3.[Cl-].[Cl-].[Zr+2]",
     "2-[3,5-bis(methylsulfanyl)pentalen-1-yl]furan—chloride—zirconium(2+) (2/2/1)"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_zirconium_adduct_of_a_mancude_ligand(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_another_kekule_structure_is_the_same_structure():
    # the ligand of the first row above: OPSIN's reading (left) and the drawing (right)
    assert _same_up_to_kekule_form("CC1=CC2=C(c3cccnc3)C=C(C)C2=P1", _mol("CC1=PC2=C(C)C=C(c3cccnc3)C2=C1"))


@pytest.mark.parametrize("other", [
    "CC1=CC2=C(c3cccnc3)C(C)=CC2=P1",    # the methyl and the pyridyl group on the other ring carbons
    "CC1=CC2=C(c3cccnc3)C=C(C)C2=C1C",   # an added methyl: another hydrogen count
])
def test_another_substituent_position_is_another_structure(other):
    assert not _same_up_to_kekule_form(other, _mol("CC1=PC2=C(C)C=C(c3cccnc3)C2=C1"))


def test_a_tautomer_and_a_charge_are_other_structures():
    assert not _same_up_to_kekule_form("Oc1ccccn1", _mol("O=c1cccc[nH]1"))
    assert not _same_up_to_kekule_form("C[NH3+]", _mol("CN"))


def test_a_stereo_layer_keeps_the_strict_test():
    assert not _same_up_to_kekule_form("C/C=C/C", _mol("C/C=C\\C"))


def test_the_group_13_atom_of_a_mancude_ring_moves_its_double_bond_too():
    # an 'indiginine' ring: OPSIN's reading (left) and the drawing (right) differ in the place of the In=C bond only; a
    # Group 13-16 element has a parent hydride, so its ring bonds are Kekule bonds like those of carbon
    assert _same_up_to_kekule_form("COc1nnc(NC(=O)C2=C(c3cc(Cl)ncc3OC)C=[C](C)[In]=[CH]2)s1.[OH-]",
                                   _mol("CC1=[In]C=C(C(=C1)C2=CC(=NC=C2OC)Cl)C(=O)NC3=NN=C(S3)OC.[OH-]"))


@pytest.mark.parametrize("read_back,drawn", [
    ("c1ccsc1", "C1=CC=S=C1"),                        # thiophene against its lambda4-sulfur form
    ("C1=CC=C[PH]1", "C1=[PH]=CC=C1"),                # 1H-phosphole against its lambda5-phosphorus form
    ("[Fe][Fe]", "[Fe]=[Fe]"),                        # the order of a metal-metal bond
    ("Cl[Mo][Mo]Cl", "Cl[Mo]#[Mo]Cl"),
    ("[Mo]S[Mo]", "[Mo]=S=[Mo]"),                     # a metal-sulfur-metal bridge
    ("C[Sn](C)=[Sn](C)C", "C[Sn](C)[Sn](C)C"),
    ("C=[Ti](Cl)Cl", "[CH2][Ti](Cl)Cl"),
])
def test_another_bonding_number_is_another_structure(read_back, drawn):
    # one standard InChIKey each, and the same graph once the bond orders are erased: only the bonding number of an
    # atom tells them apart (a Kekule structure keeps every bonding number)
    assert not _same_up_to_kekule_form(read_back, _mol(drawn))
    assert not _same_up_to_kekule_form(drawn, _mol(read_back))


def test_a_ring_that_alternates_its_double_bonds_is_one_structure_either_way():
    assert _same_up_to_kekule_form("C1=CC=C2C=CC=C2C=C1", _mol("C1=CC2=CC=CC2=CC=C1"))   # azulene
    assert _same_up_to_kekule_form("C1=CC2=CC=CC2=CC=C1", _mol("C1=CC=C2C=CC=C2C=C1"))


@pytest.mark.parametrize("read_back,drawn", [
    ("c1ccc2ccccc2c1", "C1=CC2=CC=CC=C2C=C1"),             # naphthalene (aromatic)
    ("C1=CC2=CC=CC2=C1", "C1=CC2=CC=CC2=C1"),              # pentalene (ortho-fused, not aromatic)
    ("CC1=CC2=CC=CC2=C1", "CC1=CC2=CC=CC2=C1"),
    ("c1ccncc1", "C1=CC=NC=C1"),                           # pyridine
    ("C[n+]1ccccc1", "C[N+]1=CC=CC=C1"),                   # N-methylpyridinium
    ("CC1=NC=CC=CC=C1", "CC1=CC=CC=CC=N1"),                # a 2-methylazocine: names no double-bond locant
])
def test_an_aromatic_or_ortho_fused_ring_system_is_one_structure_in_any_kekule_form(read_back, drawn):
    assert _same_up_to_kekule_form(read_back, _mol(drawn))


@pytest.mark.parametrize("read_back,drawn", [
    ("CC1=C(C)C=CC=CC=C1", "CC1=CC=CC=CC=C1C"),                        # 1,2- and 1,8-dimethylcycloocta-1,3,5,7-tetraene
    ("CC1=C(C)C(C)=C(C)C=CC=C1", "CC1=CC=CC=C(C)C(C)=C1C"),            # the tetramethyl pair
    ("CC1=C(C)C=C1", "CC1=CC=C1C"),                                    # 1,2- and 1,4-dimethylcyclobutadiene
])
def test_a_single_ring_that_is_not_aromatic_states_its_double_bond_locants(read_back, drawn):
    # one standard InChIKey each; is the rule of fused names, and a monocycle or a von Baeyer skeleton
    # names its double bonds with locants, so a moved double bond is another compound
    assert not _same_up_to_kekule_form(read_back, _mol(drawn))
    assert not _same_up_to_kekule_form(drawn, _mol(read_back))


def test_a_bridged_skeleton_states_its_double_bond_locants():
    # a methyl-1,6-methano[10]annulene and its bond-shift isomer: one standard InChIKey, two von Baeyer names
    assert not _same_up_to_kekule_form("CC1=CC2=CC=CC=C(C=C1)C2", _mol("CC1=CC=C2C=CC=CC(=C1)C2"))
