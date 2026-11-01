"""Lane DT5: the ring-system ladder of the PIN path (fusion -> bridged fusion -> von Baeyer).

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* ``### **** Five-membered ring requirement`` (:23706): "Fusion nomenclature gives preferred IUPAC
  names only to compounds having at least two rings of at least five or more members.... When fusion names
  are not allowed, unsaturated von Baeyer ring system names are preferred IUPAC names (see."
  ('bicyclo[4.2.0]octa-1,3,5,7-tetraene (PIN)', 'bicyclo[4.1.0]hepta-1,3,5-triene (PIN)'.)
* ``### **** Fusion names composed of first- and higher-order attached components`` (:12718): the
  locants of second-order attached components are primed. A three- or four-membered ring is an attached
  component like any other (``cyclopropa[de]anthracene (PIN)``:12525, ``cyclobuta[1,7]indeno[5,6-b]
  naphthalene (PIN)``:14513); only asks for two rings of five or more members in the system.
* ``## **** Preferred IUPAC names in phane nomenclature`` (:23824, example:23883): "the bridged fused
  ring name is preferred to the von Baeyer name".

Each expected name is read back by OPSIN to the full InChIKey of its molecule (``name_is_rt_exact``, an
independent parse). The names of the census rows of lane DT5 are the ones the default tier emits."""
import pytest
from rdkit import Chem

from orthonym.rules import ring_ladder
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# (SMILES, the PIN-tier name). The first two are rows of the DT4 census (rsu91, cause B); the others are
# neighbours of the class: a three-membered ring fused to a macrocycle or to a benzo-cycloalkane.
SMALL_RING_SECOND_ORDER = [
    # census row 49: furan + cyclodecene + oxirane
    ('C/C1=C\\CC[C@]2(C)O[C@H]2C(=O)c2c(C)coc2C1',
     "(1aS,4E,10aR)-1a,5,9-trimethyl-1a,3,6,10a-tetrahydrooxireno[2',3':4,5]cyclodeca[1,2-b]furan-10(2H)-one"),
    # census row 54: pyran + cycloundecene + oxirane
    ('C/C1=C\\[C@H](O)C[C@]2(C)O[C@H]2C[C@@]2(C)OC(=O)C=C(C(C)C)[C@@H]2CC1',
     "(1aS,3R,4E,7aS,11aR,12aS)-3-hydroxy-1a,5,11a-trimethyl-8-(propan-2-yl)-2,3,6,7,7a,11a,12,12a-"
     "octahydrooxireno[2',3':9,10]cycloundeca[1,2-b]pyran-10(1aH)-one"),
    # the parthenolide skeleton, without stereo
    ('CC1=CCCC2(C)OC2C2OC(=O)C(=C)C2CC1',
     "7,10a-dimethyl-4-methylidene-1a,4,4a,5,6,9,10,10a-octahydrooxireno[2',3':9,10]cyclodeca[1,2-b]furan-3(1bH)-one"),
    ('O=C1OC2CC3OC3C2C1', "hexahydro-4H-oxireno[2',3':3,4]cyclopenta[1,2-b]furan-4-one"),
    ('C1CCC2OC2C2CCCCC12', 'decahydro-2H-benzo[3,4]cyclohepta[1,2-b]oxirene'),
]


@pytest.mark.parametrize("smiles,name", SMALL_RING_SECOND_ORDER)
def test_a_three_membered_ring_is_a_second_order_component(smiles, name):
    """Two rings of five or more members are present, so the system has a fusion name, and its
    three-membered ring is the second-order 'oxireno' of."""
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "pin_verified"), b


# A bridged fused system whose fused parent has a three-membered second-order component: the PIN is the bridged fused
# name, the Blue Book: "No mancude ring, so a phane name is not allowed; the bridged fused ring
# name is preferred to the von Baeyer name"), no longer the von Baeyer name the row was declined with.
BRIDGED_WITH_SMALL_RING = [
    ('CC1(C)C(O)C(O)CC2(C)C1CCC13CC(CCC21)C1(C)OC31',
     "4,4,7a,10a-tetramethyltetradecahydro-1b,10-methanonaphtho[2',1':3,4]cyclohepta[1,2-b]oxirene-5,6-diol"),
    # census row 540 of the canary: an epoxy-cycloundecene fused to an isoindolone
    ('CC1=CC2/C=C(\\C)CCC3OC3/C=C/C(=O)C23C(=O)NC(CC(C)C)C3C1C',
     "(7E,12E)-4,5,8-trimethyl-3-(2-methylpropyl)-3,3a,4,6a,9,10,10a,11a-octahydro-1H-oxireno[2',3':7,8]"
     "cycloundeca[1,2-d]isoindole-1,14(2H)-dione"),
]


@pytest.mark.parametrize("smiles,name", BRIDGED_WITH_SMALL_RING)
def test_a_bridged_fused_parent_may_hold_a_three_membered_component(smiles, name):
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d


# (SMILES, the PIN-tier name): fewer than two rings of five or more members, so gives the von Baeyer
# name. The first is census row 16 (cause C: two oxiranes on a ten-membered ring, a lactone).
VON_BAEYER_RUNG = [
    ('C[C@@H]1C[C@@H]2O[C@@H]2[C@H]2O[C@@H]2[C@@H](O)CC(=O)O1',
     '(1S,2S,4S,6R,10S,11R)-10-hydroxy-6-methyl-3,7,12-trioxatricyclo[9.1.0.0^2,4]dodecan-8-one'),
    ('C1CC2OC2C2OC2C1', '3,9-dioxatricyclo[6.1.0.0^2,4]nonane'),
    ('O=C1OCC2C1C1CC21', '7-oxatricyclo[3.3.0.0^2,4]octan-6-one'),
    ('OC1CC2OC2C2CC12', '3-oxatricyclo[5.1.0.0^2,4]octan-6-ol'),
    ('C1CC2C3CC3C2C1', 'tricyclo[3.3.0.0^2,4]octane'),
]


@pytest.mark.parametrize("smiles,name", VON_BAEYER_RUNG)
def test_the_von_baeyer_rung_names_a_system_fusion_nomenclature_may_not_name(smiles, name):
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d


def test_fusion_name_not_allowed_is_the_five_membered_ring_requirement():
    """: fewer than two rings of five or more members. Three- and four-membered rings fused to one
    ring of five or more are the class; two rings of six are not."""
    def not_allowed(smiles):
        return ring_ladder.fusion_name_not_allowed(Chem.MolFromSmiles(smiles))
    assert not_allowed('C1CC2OC2C2OC2C1')              # 3, 3, 7
    assert not_allowed('C1CC2C3CC3C2C1')               # 3, 4, 5: the five-membered ring is the only one
    assert not not_allowed('C1CCC2OC2C2CCCCC12')       # 3, 6, 7: two rings of five or more
    assert not not_allowed('C1CCC2CCCCC2C1')           # naphthalene skeleton
    assert not not_allowed('C1CCCCC1')                 # no ring system of two rings


def test_the_rung_does_not_replace_a_name_the_engine_gives_without_it():
    """The von Baeyer namer is asked only after the fusion namers decline, and a refusal of its own is no
    candidate: benzocyclobutene keeps the von Baeyer name of the general engine at the breadth tier (the rung
    once raised its refusal, and the engine then fell back to a trivial name)."""
    b = name_breadth('c1ccc2c(c1)CC2')
    assert b.get("name") == 'bicyclo[4.2.0]octa-1,3,5-triene', b
    assert name_is_rt_exact(b["name"], 'c1ccc2c(c1)CC2')
