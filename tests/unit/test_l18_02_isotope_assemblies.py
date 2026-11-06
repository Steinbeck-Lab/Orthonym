"""lane/l18 item 2: labelled molecules whose skeleton name the descriptor could not enter (ring assemblies and
saturated fused ring systems), unnamed with OPSIN 2.9.0 and OPSIN main alike.

* A ring assembly parent stands after its prefixes ('4-(trifluoromethoxy)-2,2'-bi-1,3-thiazole').
  (the Blue Book): the descriptor goes "before the part of the compound that is isotopically substituted", here
  the whole assembly name with its locants: '4-(trifluoromethoxy)(5-2H)-2,2'-bi-1,3-thiazole'.
* A fully saturated fused ring system is named by the count form of the hydro prefix,:17026, "Total
  hydrogenation is indicated by the appropriate multiplicative prefixes... but locants are omitted"). OPSIN reads no
  descriptor on that form when two nuclides sit on one atom ('decahydro(1,1-2H2)naphthalene'). (:44180)
  "if isotopic modification requires a locant to specify its position, then all locants must be specified": the
  skeleton is named again with its hydro locants cited, below the PIN.
"""
import pytest

from orthonym import Orthonym
from orthonym.assembly import locant_omission as lo
from orthonym.rules import isotopes as iso
from orthonym.rules.isotope_hydro_locants import _ACTIVE, stereo_scope_reason

pytestmark = pytest.mark.unit


def _best_effort(smiles):
    r = Orthonym(general_fallback=True, general_fallback_unverified=True,
                 allow_aromatic_general=True).name_tiered(smiles)
    return r.get("name"), r.get("tier")


def test_ring_assembly_takes_the_descriptor_before_its_locants():
    r = Orthonym().name_tiered("[2H]C1=C(N=C(S1)C2=NC=CS2)OC(F)(F)F")
    assert r.get("name") == "4-(trifluoromethoxy)(5-2H)-2,2'-bi-1,3-thiazole"
    assert r.get("tier").startswith("pin")      # the unlabelled name is the PIN and the placement is the book's (the OPSIN gate decides verified)


def test_ring_assembly_slot_is_not_offered_for_bicyclo():
    from orthonym.rules.isotopes import _insertion_offsets
    assert len(_insertion_offsets("4-chloro-2,2'-bi-1,3-thiazole")) > len(_insertion_offsets("4-chloro-1,3-thiazole"))
    assert not _RING_RE_MATCHES("2,3-bicyclo")


def _RING_RE_MATCHES(skeleton):
    from orthonym.rules.isotopes import _RING_ASSEMBLY_PARENT_RE
    return _RING_ASSEMBLY_PARENT_RE.search("x-" + skeleton)


CASES = [
    # decalin ring with a CD2 pair on one ring and one on the other
    ("[2H]C1(CC2CC(CCC2C(C1CCCCC)([2H])[2H])C3=CC=C(C=C3)C#N)[2H]",
     "4-[6-pentyl-1,2,3,4,4a,5,6,7,8,8a-decahydro(5,5,7,7-2H4)naphthalen-2-yl]benzonitrile"),
    ("[2H]C1([2H])CCC2CCCCC2C1", "1,2,3,4,4a,5,6,7,8,8a-decahydro(2,2-2H2)naphthalene"),
]


@pytest.mark.parametrize("smiles,expected", CASES)
def test_saturated_fused_skeleton_cites_its_hydro_locants(smiles, expected):
    name, tier = _best_effort(smiles)
    assert name == expected
    assert tier != "pin_verified"       # the book omits the locants of total hydrogenation


def test_unlabelled_saturated_fused_name_keeps_the_count_form():
    name, tier = _best_effort("CC1CCC2CCCCC2C1")
    assert name == "2-methyldecahydronaphthalene"


def test_single_label_keeps_its_name():
    name, _ = _best_effort("C1CCC2CCCCC2C1[2H]")
    assert name == "decahydro(1-2H1)naphthalene"


def test_scope_of_the_stereo_retry_keeps_the_hydro_locants_of_the_try_it_runs_in():
    """ (:44180) "all locants must be specified and none are omitted": the retry that cites the locant of a
    lone stereodescriptor inside the try that cites the hydro locants cites both."""
    assert stereo_scope_reason() == lo.ISOTOPIC_STEREO_LOCANTS
    token = _ACTIVE.set(True)
    try:
        assert stereo_scope_reason() == lo.ISOTOPIC_STEREO_HYDRO_LOCANTS
    finally:
        _ACTIVE.reset(token)
    with lo.isotopic_naming_scope(lo.ISOTOPIC_STEREO_LOCANTS):
        assert lo.scope_cites_stereo_locants() and not lo.scope_cites_hydro_locants()
    with lo.isotopic_naming_scope(lo.ISOTOPIC_HYDRO_LOCANTS):
        assert lo.scope_cites_hydro_locants() and not lo.scope_cites_stereo_locants()
    with lo.isotopic_naming_scope(lo.ISOTOPIC_STEREO_HYDRO_LOCANTS):
        assert lo.scope_cites_hydro_locants() and lo.scope_cites_stereo_locants()


@pytest.mark.parametrize("name,expected", [
    ("4-(trifluoromethoxy)-2,2'-bi-1,3-thiazole", [20]),
    ("2-bromo-6,6'-bis[(propan-2-yl)oxy]-1,1'-biphenyl", [34]),
    ("6,6'-bis(4-methylphenyl)pyridine", []),
    ("2,2'-tert-butyl-1,1'-biphenyl", [15]),
    ("x-2,2'-bicyclo[2.2.1]heptane", []),
])
def test_ring_assembly_parent_slot_is_the_parent_and_not_a_multiplied_prefix(name, expected):
    """ ring assemblies: ``bi``, ``ter``,... multiply a ring; ``bis(`` and ``tert-`` only begin like them."""
    assert [m.start() for m in iso._RING_ASSEMBLY_PARENT_RE.finditer(name)] == expected


@pytest.mark.opsin_gate
def test_stereo_retry_inside_the_hydro_try_keeps_the_hydro_locants():
    """ (:44180): the chain prefix cites the locant of its '(5E)' (the labels sit on its other double bond)
    and the saturated ring system cites its hydro locants, in one name; the retry used to leave the hydro scope, so
    the descriptor moved to the front of the name, where its locant 5 reads as one of the ring system."""
    r = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True).name_tiered(
        "[2H]C=C([2H])C(C/C=C/c1ccccc1)C1CCC2C([2H])([2H])CCCC2C1")
    assert r.get("name") == ("2-[(5E)-6-phenyl(1,2-2H2)hexa-1,5-dien-3-yl]-"
                             "1,2,3,4,4a,5,6,7,8,8a-decahydro(5,5-2H2)naphthalene")
    assert r.get("tier") != "pin_verified"      # the hydro locants of total hydrogenation are the try's
