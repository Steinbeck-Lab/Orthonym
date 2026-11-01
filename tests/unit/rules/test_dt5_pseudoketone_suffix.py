"""Lane DT5: the suffix of a fused parent beside a pseudoketone (``bridged_fused_pin.principal_suffix``).

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* ``### ** SENIORITY ORDER FOR CLASSES**`` Table 4.1 (:18189): "16 Ketones..., pseudoketones... See
  Classes 8, 9, 11 and 13, on lactones, lactams, anhydrides, imides", below 14 nitriles and 15 aldehydes;
  ``****`` (:29628) "There is no seniority order difference between ketones and pseudoketones";
  ``## **** Lactones`` (:32112) "A lactone, as a pseudoketone, ranks lower in the seniority of
  classes than an acid or an ester, but higher than an alcohol, amine, or imine" ('8-oxo-7-oxabicyclo[4.2.0]
  octane-4,5-dicarboxylic acid (PIN)':32120: the lactone is the 'oxo' prefix beside an acid).
* ``### **** Amidrazone suffixes`` (:34428): 'carbohydrazonamide' names the group on a cyclic parent
  hydride, so a -C(=N-N)-N- unit with every atom in the ring is skeletal.

Each expected name is read back by OPSIN to the full InChIKey of its molecule (``name_is_rt_exact``, an
independent parse)."""
import pytest

from orthonym.rules.seniority import without_ring_skeleton_amidine
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# (SMILES, the PIN-tier name): a lactone (pseudoketone, class 16) beside an aldehyde (class 15) or a nitrile (14):
# the senior group is the suffix and the lactone is 'oxo'. The first is census row 11 (cause D1).
PSEUDOKETONE_BESIDE_SENIOR_GROUP = [
    ('O=Cc1ccc2n1[C@@H](Cc1ccccc1)C(=O)OC2',
     '(4S)-4-benzyl-3-oxo-3,4-dihydro-1H-pyrrolo[2,1-c][1,4]oxazine-6-carbaldehyde'),
    ('O=Cc1ccc2n1CC(=O)OC2', '3-oxo-3,4-dihydro-1H-pyrrolo[2,1-c][1,4]oxazine-6-carbaldehyde'),
    ('N#Cc1ccc2n1CC(=O)OC2', '3-oxo-3,4-dihydro-1H-pyrrolo[2,1-c][1,4]oxazine-6-carbonitrile'),
    ('O=Cc1ccc2n1CC(=O)NC2', '3-oxo-1,2,3,4-tetrahydropyrrolo[1,2-a]pyrazine-6-carbaldehyde'),
]


@pytest.mark.parametrize("smiles,name", PSEUDOKETONE_BESIDE_SENIOR_GROUP)
def test_a_senior_group_takes_the_suffix_and_the_lactone_is_oxo(smiles, name):
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d


def test_an_ester_senior_to_the_lactone_still_declines_at_the_pin_tier():
    """An acyclic ester is class 9, senior to a pseudoketone: the builder does not name it (the class name
    'acetate' is not a suffix it spells), and the breadth tier keeps a name."""
    smiles = 'CC(=O)Oc1ccc2n1CC(=O)OC2'
    assert name_default(smiles).get("tier") != "pin_verified"
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


def test_a_cyclic_hydrazonamide_unit_is_skeletal():
    """Census row 39 (cause D1): the ring holds N=C-N and N-N whole, a heterocycle's own atoms."""
    smiles = 'CC(=O)C[C@]12NC(=O)N(C)N=C1N(C)C(=O)N(C)C2=O'
    name = ('(4aS)-2,6,8-trimethyl-4a-(2-oxopropyl)-2,4,4a,8-tetrahydropyrimido[5,4-e][1,2,4]triazine-'
            '3,5,7(6H)-trione')
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d


def test_without_ring_skeleton_amidine_sets_aside_each_unit_kind_that_the_ring_holds_whole():
    ring = {1, 2, 3, 4, 5}
    groups = {'hydrazonamide': [(1, 2, 3, 4)], 'amidine': [(1, 2, 9)], 'ketone': [(1, 8, 7)]}
    got = without_ring_skeleton_amidine(groups, ring)
    assert 'hydrazonamide' not in got                  # every atom of the unit is a ring atom
    assert 'amidine' in got                            # one amidine atom (9) lies outside: a real group
    assert 'ketone' in got
    # nothing is set aside when a unit has an atom outside the ring, and the dict is returned as it was
    only_outside = {'hydrazonamide': [(1, 2, 3, 40)]}
    assert without_ring_skeleton_amidine(only_outside, ring) is only_outside


# Census row 81 (cause B: an 11-membered ring fused to a five-membered one): the fusion name was built, and the PIN tier
# declined it for the spelling 'dioxidanyl' of the -OOH, which is no PIN vocabulary. ``## **** Hydroperoxides``
# (the Blue Book): "The prefix 'hydroperoxy' is formed by concatenation to describe the group -OOH as a
# substituent in the presence of a characteristic group having priority for citation as a suffix"; Table 4.1 (:18189)
# class 18 'Hydroperoxides' stands below the ketones (16) and hydroxy compounds (17). A hydroperoxide with no senior
# group is the peroxol of (1), which the builder does not spell: it stays declined.
HYDROPEROXY_PREFIX = [
    ('CC(C)C1=C2CCC(=C)[C@@H](CCC(=C)C(=O)C[C@]2(CC1=O)C)OO',
     '(3aS,9R)-9-hydroperoxy-3a-methyl-6,10-dimethylidene-1-(propan-2-yl)-3a,4,7,8,9,10,11,12-octahydrocyclopenta'
     '[11]annulene-2,5(3H,6H)-dione'),
    ('O=C1CCCC2CCCC(OO)C12', '8-hydroperoxyoctahydronaphthalen-1(2H)-one'),
    ('OOC1CCC2CCCC2C1O', '5-hydroperoxyoctahydro-1H-inden-4-ol'),
    ('OC(=O)C1CCCC2CCC(OO)CC12', '7-hydroperoxydecahydronaphthalene-1-carboxylic acid'),
]


@pytest.mark.parametrize("smiles,name", HYDROPEROXY_PREFIX)
def test_a_hydroperoxy_group_beside_a_senior_group_is_the_prefix_hydroperoxy(smiles, name):
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d


@pytest.mark.parametrize("smiles", ['OOC1CCCC2CCCC12', 'OOC1CCC2CCCC2C1OO'])
def test_a_hydroperoxide_with_no_senior_group_is_not_named_with_the_prefix(smiles):
    """The peroxol is the principal group there (1)): 'hydroperoxy' would be a prefix with no suffix."""
    assert name_default(smiles).get("tier") != "pin_verified"
    b = name_breadth(smiles)
    assert b.get("name") and "hydroperoxy" not in b["name"] or b.get("tier") != "pin_verified", b

