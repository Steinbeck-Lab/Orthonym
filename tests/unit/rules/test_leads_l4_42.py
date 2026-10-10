"""Leads program L4, item 42: phosphoric-acid tri- and diesters rank with the esters.

 "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1: "9 Esters (functional
class names are given to noncyclic esters; lactones and other cyclic esters are named as
heterocycles; see 16 below)" (:18182) ranks above "16 Ketones" (:18189), "17 Hydroxy
compounds" (:18190) and "19 Amines" (:18192); "Esters of mononuclear noncarbon
oxoacids" (:35916): 'trimethyl phosphate (PIN)' (:35936), 'methyl dihydrogen phosphate
(PIN)' (:35940). ASSUMED (not stated): the acidic diester counts as class 9 rather than 7d;
either reading puts it above the amines, hydroxy compounds, ketones, aldehydes and nitriles.

The class is OFFERED, not returned: ``get_principal_group`` ranks the ester only when
``rules.phosphorus.name_phosphate_ester`` can write the whole-molecule name (an ester the
namer declines keeps the head order, so 'propan-2-ol', '-one' and a carboxylic ester parent
are never demoted to prefixes of a suffix-less principal group).
"""
import importlib
import inspect

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# (smiles, the functional-class ester name). Every name reads back to the full InChIKey.
ESTER_ROWS = [
    ("NCCOP(=O)(O)OC", "2-aminoethyl methyl hydrogen phosphate"),
    ("NCCOP(=O)(OC)OC", "2-aminoethyl dimethyl phosphate"),
    ("OCCOP(=O)(OC)OC", "2-hydroxyethyl dimethyl phosphate"),
    ("CC(=O)COP(=O)(OC)OC", "dimethyl 2-oxopropyl phosphate"),
    ("N#CCOP(=O)(OC)OC", "cyanomethyl dimethyl phosphate"),
    ("O=CCOP(=O)(OC)OC", "dimethyl 2-oxoethyl phosphate"),
    ("Nc1ccc(OP(=O)(OCC)OCC)cc1", "4-aminophenyl diethyl phosphate"),
    ("OC(CO)COP(=O)(O)OCCN", "2-aminoethyl 2,3-dihydroxypropyl hydrogen phosphate"),
    # the tier-inversion row: the default tier gave this name, the wider tiers the
    # substitutive amine name
    ("CCCCCCCCCCCCCCCCCCOCC(COP(=O)(O)OCCN)OC",
     "2-aminoethyl 2-methoxy-3-(octadecyloxy)propyl hydrogen phosphate"),
]


def _pg(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return get_principal_group(mol, detect_functional_groups(mol))[0]


@pytest.mark.parametrize("smiles,expected", ESTER_ROWS)
def test_phosphate_ester_is_the_principal_group(smiles, expected):
    assert _pg(smiles) in ("phosphate_triester", "phosphate_diester")


@pytest.mark.parametrize("smiles,expected", ESTER_ROWS)
def test_functional_class_ester_name_at_both_tiers(smiles, expected):
    assert name_is_rt_exact(expected, smiles), expected
    assert_pin_at_both_tiers(smiles, expected)


# The head order stays where the ester is not offered or not outranking.
HEAD_ORDER_ROWS = [
    # the acid-like amide is class 11 and was not measured above the esters (follow-up below)
    ("NC(=O)COP(=O)(OC)OC", "primary_amide"),
    # a carboxylic ester is senior to a phosphate ester (the acids' order, class 9)
    ("COC(=O)CCOP(=O)(OC)OC", "ester"),
    # a phosphorus ring atom is a cyclic ester, named as a heterocycle (Table 4.1 class 9 note)
    ("NC1COP(=O)(O)OC1", "primary_amine"),
    # two phosphate esters: no whole-molecule functional-class name exists, the head stays
    ("NCCOP(=O)(O)OCCOP(=O)(O)OC", "primary_amine"),
    # an ester the namer declines (a charged ester) keeps the head order
    ("NCCOP(=O)(O)OC[C@@H](O)C[N+](C)(C)C", "secondary_alcohol"),
    # plain esters: the phosphate triester is its own head
    ("COP(=O)(OC)OC", "phosphate_triester"),
]


@pytest.mark.parametrize("smiles,head", HEAD_ORDER_ROWS)
def test_head_order_kept_when_the_ester_is_not_offered(smiles, head):
    assert _pg(smiles) == head


def test_fragment_named_inside_another_name_keeps_the_head_order(monkeypatch):
    """A functional-class name cannot take the substituents of the other fragments of a larger
    name, so the ester is ranked for a whole molecule only (the same reasoning as
    ``name_general_multiplicative``)."""
    smiles = "NCCOP(=O)(OC)OC"
    assert _pg(smiles) == "phosphate_triester"
    import orthonym.assembly.fragment_naming as fragment_naming
    monkeypatch.setattr(fragment_naming, "is_top_level_naming", lambda: False)
    assert _pg(smiles) == "primary_amine"


def test_declined_offer_keeps_the_next_suffix():
    """The coumarin lactone: the PIN tier cannot write the ester name (the owner is a ring
    system), so the ketone stays the principal group and the name keeps its '-one' suffix (it
    is not demoted to 'oxo'). That substitutive name is correct and OPSIN reads it back, but
    it names a junior class (Table 4.1: the lactone is class 16, the ester class 9), so it is
    no PIN: the default tier declines it and the valid tier ships it as a systematic name."""
    smiles = "CCOP(=O)(OCC)Oc1ccc2c(C)c(Cl)c(=O)oc2c1"
    name = "3-chloro-7-[(diethoxyphosphoryl)oxy]-4-methyl-2H-1-benzopyran-2-one"
    assert _pg(smiles) == "ketone"
    assert name_is_rt_exact(name, smiles), name
    assert name_default(smiles).get("tier") != "pin_verified"
    v = Orthonym(style="pin", **_emit_tier_flags("valid")).name_tiered(smiles)
    assert (v.get("name"), v.get("tier")) == (name, "systematic_verified"), v


@pytest.mark.xfail(strict=True, reason=(
    "P-41 Table 4.1 puts esters (class 9) above amides (class 11): the expected PIN is the "
    "ester '2-amino-2-oxoethyl dimethyl phosphate'. Ranking the phosphate esters above the "
    "amides was not measured (follow-up of lane L4), so the amide stays the head."))
def test_ester_above_amide_follow_up():
    assert _pg("NC(=O)COP(=O)(OC)OC") in ("phosphate_triester", "phosphate_diester")


# Stereo-bearing owners: the functional-class name carries the descriptors inside the owners, and
# the handler must declare that scope to the stereo injector. The ester rank is not offered for a
# molecule with stereo (``rules.phosphorus._OFFER_STEREO_OWNERS`` is False, held by measurement:
# see the comment there), so these rows abstain at the default tier (strict xfail; ordinary tests
# once the handler declares the scope and the switch is True).
def _stereo_owner_esters_offered() -> bool:
    from orthonym.rules import phosphorus
    handler = importlib.import_module("orthonym.assembly.handlers.phosphate_ester")
    return phosphorus._OFFER_STEREO_OWNERS and "retained_no_locants" in inspect.getsource(handler)


NEEDS_STEREO_OWNER_ESTERS = pytest.mark.xfail(
    not _stereo_owner_esters_offered(), strict=True,
    reason="stereo-bearing phosphate esters are not offered the ester rank yet (item 42c)")

STEREO_ESTER_ROWS = [
    pytest.param("OC[C@H](O)COP(=O)(O)OCCN",
                 "2-aminoethyl (2S)-2,3-dihydroxypropyl hydrogen phosphate",
                 marks=NEEDS_STEREO_OWNER_ESTERS, id="glycerophosphoethanolamine-S"),
    pytest.param("OC[C@@H](O)COP(=O)(O)OCCN",
                 "2-aminoethyl (2R)-2,3-dihydroxypropyl hydrogen phosphate",
                 marks=NEEDS_STEREO_OWNER_ESTERS, id="glycerophosphoethanolamine-R"),
]


@pytest.mark.parametrize("smiles,expected", STEREO_ESTER_ROWS)
def test_stereo_owner_ester_name(smiles, expected):
    assert name_is_rt_exact(expected, smiles), expected
    assert_pin_at_both_tiers(smiles, expected)
