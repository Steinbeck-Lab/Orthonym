"""The preselected name of a von Baeyer or spiro system of one noncarbon element.

 "Heterocyclic noncarbon von Baeyer and spiro compounds" (the Blue Book), (:23614):
"Preselected names for von Baeyer compounds and for spiro compounds having only monocyclic components and
consisting entirely of heteroatoms of the same kind are names formed by citing the appropriate prefix, such
as 'bicyclo', 'spiro', etc., and descriptor enclosed within square brackets followed by a numerical prefix for
the total number of heteroatoms and the name of the mononuclear parent hydride; alternative names for use in
general nomenclature are formed by skeletal replacement ('a') nomenclature". The book prints
'bicyclo[4.2.1]nonasilane (preselected name)' beside 'nonasilabicyclo[4.2.1]nonane' (:23618, and
HOMOGENEOUS HETEROCYCLIC VON BAEYER PARENT HYDRIDES,:9810) and 'tricyclo[5.3.1.1^2,6]dodecasilane
(preselected name)' beside 'dodecasilatricyclo[5.3.1.1^2,6]dodecane' (:23626,:9814).

OPSIN has no grammar for 'bicyclo[4.2.1]nonasilane'; the PIN tier ships it through its twin, the 'a' name
(``validation.preselected_cage_twin``). The 'a' name is a name for general nomenclature, so it is the name of
the wider tiers, below the PIN, and a cage with a group on it has no preselected name here.
"""
import pytest

from orthonym import Orthonym
from orthonym.rules.preselected_cage import preselected_cage_name
from orthonym.validation.preselected_cage_twin import preselected_cage_twin
from orthonym.validation.twin_spellings import twin_of

# the SMILES are the structures OPSIN 2.9.0 reads from the 'a' names of the book's own examples
SI9_BICYCLO_421 = "[SiH2]1[SiH2][SiH2][SiH]2[SiH2][SiH2][SiH]([SiH2]1)[SiH2]2"
SI12_TRICYCLO = "[SiH2]1[SiH2][SiH]2[SiH2][SiH]([SiH2]1)[SiH]1[SiH2][SiH2][SiH2][SiH]2[SiH2]1"

PREFERRED = [
    # the two examples of the book (:23618,:23626)
    (SI9_BICYCLO_421, "bicyclo[4.2.1]nonasilane"),
    (SI12_TRICYCLO, "tricyclo[5.3.1.1^2,6]dodecasilane"),
    # the same construction for the other elements the book names a mononuclear hydride of: 'germane',
    # 'stannane', 'plumbane', 'arsane', 'stibane', 'bismuthane', Table 2.1)
    ("[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2", "bicyclo[2.2.1]heptasilane"),
    ("[GeH2]1[GeH2][GeH]2[GeH2][GeH2][GeH]1[GeH2]2", "bicyclo[2.2.1]heptagermane"),
    ("[SnH2]1[SnH2][SnH]2[SnH2][SnH2][SnH]1[SnH2]2", "bicyclo[2.2.1]heptastannane"),
    ("[PbH2]1[PbH2][PbH]2[PbH2][PbH2][PbH]1[PbH2]2", "bicyclo[2.2.1]heptaplumbane"),
    ("[AsH]1[AsH][As]2[AsH][AsH][As]1[AsH]2", "bicyclo[2.2.1]heptaarsane"),
    ("[SbH]1[SbH][Sb]2[SbH][SbH][Sb]1[SbH]2", "bicyclo[2.2.1]heptastibane"),
    ("[BiH]1[BiH][Bi]2[BiH][BiH][Bi]1[BiH]2", "bicyclo[2.2.1]heptabismuthane"),
    # a spiro system with only monocyclic components, and a polycycle of more than two rings
    ("[SiH2]1[SiH2][SiH2][Si]2([SiH2]1)[SiH2][SiH2][SiH2][SiH2]2", "spiro[4.4]nonasilane"),
    ("[SiH]12[SiH]3[SiH]4[SiH]1[SiH]1[SiH]2[SiH]3[SiH]41", "pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octasilane"),
    # OPSIN reads these two elements' preselected names itself
    ("P1PP2PPPP(P1)P2", "bicyclo[3.3.1]nonaphosphane"),
    ("N1NN2NNN1NN2", "bicyclo[2.2.2]octaazane"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,preselected", PREFERRED)
def test_pin_tier_gives_the_preselected_name(smiles, preselected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == preselected, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [SI9_BICYCLO_421, SI12_TRICYCLO])
def test_the_book_examples_are_read_back_through_their_twin(smiles):
    """OPSIN 2.9.0 has no grammar for the preselected name: the gate outcome is the twin's."""
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["gate_outcome"] == "twin_verified:preselected_cage", row
    assert row["verified"] == "opsin_twin", row


@pytest.mark.opsin_gate
def test_the_wider_tier_gives_the_a_name_below_the_pin():
    """: the 'a' name is the alternative 'for use in general nomenclature', never the PIN."""
    from tests.support.rt_assert import assert_full_rt, name_best_effort
    be = name_best_effort(SI9_BICYCLO_421)
    assert be["name"] == "1,2,3,4,5,6,7,8,9-nonasilabicyclo[4.2.1]nonane", be
    assert be["tier"] == "systematic_verified" and be["is_pin"] is False, be
    assert_full_rt(be["name"], SI9_BICYCLO_421, what="best-effort: ")


# a group on the cage, a nonstandard bonding number (a lambda5 spiro atom), boron,:7992, 'other than
# boron hydrides') and a stereo-specified cage have no preselected name here
NO_PREFERRED_NAME = [
    ("C[SiH]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2", "methyl on the cage"),
    ("O=C(O)[SiH]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2", "carboxyl on the cage"),
    ("Cl[As]1[AsH][As]2[AsH][AsH][As]1[AsH]2", "chloro on the cage"),
    ("P1P[PH]2(P1)PPP2", "a lambda5 spiro atom"),
    ("B1BB2BBB1B2", "boron"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,why", NO_PREFERRED_NAME, ids=[w for _s, w in NO_PREFERRED_NAME])
def test_a_cage_without_a_preselected_name_is_never_a_pin(smiles, why):
    """The PIN tier gives no name (the 'a' name is not the preselected one) and the wider tier gives it,
    read back, below the PIN. Before: the 'a' name was labelled pin_verified."""
    from tests.support.rt_assert import assert_full_rt, name_best_effort
    pin = Orthonym(style="pin").name_tiered(smiles)
    assert pin["is_pin"] is False and "bicyclo" not in pin["name"] and "spiro" not in pin["name"], pin
    be = name_best_effort(smiles)
    assert be["tier"] != "pin_verified" and be["is_pin"] is False, be
    assert_full_rt(be["name"], smiles, what="best-effort: ")


def test_twin_is_the_a_name_of_the_same_cage():
    """The twin is a pure token rewrite of the name: the 'a' name of the book's own example."""
    assert preselected_cage_twin("bicyclo[4.2.1]nonasilane") == "nonasilabicyclo[4.2.1]nonane"
    assert preselected_cage_twin("tricyclo[5.3.1.1^2,6]dodecasilane") == "dodecasilatricyclo[5.3.1.1^2,6]dodecane"
    assert preselected_cage_twin("spiro[4.4]nonagermane") == "nonagermaspiro[4.4]nonane"
    assert preselected_cage_twin("bicyclo[10.9.1]docosasilane") == "docosasilabicyclo[10.9.1]docosane"
    assert twin_of("bicyclo[4.2.1]nonasilane") == ("nonasilabicyclo[4.2.1]nonane", ("preselected_cage",))


@pytest.mark.parametrize("name", [
    "bicyclo[4.2.1]nonane", "nonasilane", "trisilane", "bicyclo[2.2.1]heptan-2-ol", "2-oxabicyclo[2.2.1]heptane",
    "spiro[4.4]nonane", "bicyclo[4.2.1]nonasilane-1-ol", "hexasilinane"])
def test_twin_leaves_every_other_name_alone(name):
    assert preselected_cage_twin(name) == name
    assert twin_of(name) is None


def test_producer_refuses_a_structure_that_is_not_exactly_the_cage():
    """A mutant that skipped the whole-structure test would name these as the bare cage."""
    from rdkit import Chem
    cage = Chem.MolFromSmiles(SI9_BICYCLO_421)
    assert preselected_cage_name(cage, "bicyclo[4.2.1]") == "bicyclo[4.2.1]nonasilane"
    assert preselected_cage_name(cage, "") is None
    for smiles in ("C[SiH]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2",       # a group on it
                   "[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2.[SiH4]",  # two fragments
                   "[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH]=2",         # a double bond
                   "[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2][SiH2+]2",  # a charge
                   "[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2][29SiH2]2",  # an isotope
                   "B1BB2BBB1B2",                                          # boron
                   "[Te]1[Te][TeH]2[Te][Te][TeH]1[Te]2"):                  # nonstandard bonding number
        assert preselected_cage_name(Chem.MolFromSmiles(smiles), "bicyclo[2.2.1]") is None, smiles
