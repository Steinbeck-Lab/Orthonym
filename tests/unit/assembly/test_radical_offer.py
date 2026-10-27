"""The radical offer is the last resort of the best-effort tier: no name that ships today changes.

A molecule with radical centres that nothing else names is named by the floor with every centre cited in the name
(``assembly/radical_offer.py``). The offer is made only for a molecule that every other path of the tier, the other
offers included, has left without a name (``default_tier_offer.default_tier_row_for``), and what it cites is cited
inside ``read_back_policy.radical_last_resort`` only: the lambda convention for a divalent or trivalent centre
 "The λ-convention", the Blue Book, sentence:40562: "This method is only for general nomenclature"),
the free valences of one parent for several monovalent ones "CHOICE OF PARENT RADICAL",:40788, and
"Multiple radical centers (polyradicals)",:40524), and for a monovalent centre on a ring atom the group of
(:40426), with no bonding number cited for it.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# molecules with radical centres that a producer names today (the radical path of an earlier build named them another way)
NAMED_TODAY = [
    "C[Si](C)COCCOC[Si](C)C",                                           # the decomposition's name
    "[B]c1ccc2c(c1)[B]c1ccccc12",                                       # the general engine's name, a lambda on the ring boron
    "C1CC[B]CC1",                                                       # a PIN of the radical path of the strict tier
    "CC([NH])C(=O)OC",                                                  # the floor of the best-effort tier
]


@pytest.fixture
def offer_calls(monkeypatch):
    from orthonym.assembly import radical_offer
    calls = []
    real = radical_offer._radical_row

    def spy(namer, smiles, mol):
        calls.append(smiles)
        return real(namer, smiles, mol)

    monkeypatch.setattr(radical_offer, "_radical_row", spy)
    return calls


@pytest.mark.parametrize("smiles", NAMED_TODAY)
def test_a_molecule_that_has_a_name_never_reaches_the_radical_offer(smiles, offer_calls):
    res = name_best_effort(smiles)
    assert res["name"], res
    assert offer_calls == []


def test_a_declined_radical_reaches_the_radical_offer_once(offer_calls):
    res = name_best_effort("[B]CC[B]")
    assert res["name"] == "1,2-di(λ1-boranyl)ethane" and res["tier"] == "systematic_verified", res
    assert offer_calls == ["[B]CC[B]"]


def test_the_floor_cites_a_radical_centre_only_in_the_radical_offer():
    """Outside the offer the floor keeps the scope it always had: one radical centre, cited as a '-yl' prefix. Inside
    it the lambda convention and the free valences of one parent are built."""
    from orthonym.assembly.universal_substituent import name_universal_polyradical, name_universal_substitutive
    from orthonym.validation.read_back_policy import radical_last_resort
    pair = Chem.MolFromSmiles("[B]c1ccccc1[B]")
    cyclopentapentayl = Chem.MolFromSmiles("CCCC[C]1[CH][CH][CH][CH]1")
    assert name_universal_substitutive(pair) is None
    assert name_universal_polyradical(cyclopentapentayl) is None
    with radical_last_resort():
        assert name_universal_substitutive(pair).name == "1,2-di(λ1-boranyl)benzene"
        assert name_universal_polyradical(cyclopentapentayl).name == "1-butylcyclopentane-1,2,3,4,5-pentayl"


def test_a_monovalent_ring_centre_cites_no_bonding_number():
    """A ring boron that holds two bonds and one radical electron is a monovalent centre: licenses a lambda for
    a divalent or trivalent one only. The ring names of the floor cite the bonding number of an atom that holds fewer
    bonds than its standard valence, which cited the free valence a second time ('8λ2-boratricyclo...-8-yl', which OPSIN
    reads as another structure); the group is the one, '5H-dibenzo[b,d]borol-5-yl'."""
    smiles = ("[B]C1=C(C=CC=C1C2=CC=CC(=C2)C3=CC=C(C=C3)OC)B(C4=CC=C(C=C4)C5=CC(=CC(=C5)C#N)[N+]#[C-])"
              "C6=C7[B]C8=C(C7=CC=C6)C=C(C=C8)C9=CC=C(C=C9)C=O")
    res = name_best_effort(smiles)
    assert res["name"] and res["name"].endswith("-5H-dibenzo[b,d]borol-5-yl"), res
    assert "λ2" not in res["name"] and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


def test_the_free_valences_of_a_polyradical_cage_cite_no_bonding_number():
    """Two ring borons of a cage hold two bonds and one radical electron each: monovalent centres, the free valences
    of the parent ('-30,31-diyl'). The ring names of the floor cite the bonding number of an atom that holds fewer bonds
    than its standard valence ('30λ2,31λ2-'), which cited each free valence twice and had no name that reads back; the
    free valences of the ring are closed by hydrogens while the ring is named, and the name cites each once."""
    smiles = "[B]1N2CCN3[B]N(C4=CC=CC=C43)C5=CC=CC(=N5)C6(C7=NC(=CC=C7)N1C8=CC=CC=C82)C9=CC=CC=C9C1=CC=CC=C61"
    res = name_best_effort(smiles)
    assert res["name"] and res["name"].endswith("-30,31-diyl") and "λ" not in res["name"], res
    assert name_is_rt_exact(res["name"], smiles), res


def test_a_name_that_cites_a_locant_twice_in_its_descriptors_is_offered_only_as_one_structure():
    """The rule of the stereo-composed offer (``stereo_composed_offer.repeated_locants_are_one_structure``): a repeated
    locant only when all its descriptors are equal and every stereo unit is stated."""
    from orthonym.assembly.radical_offer import _verified_name
    assert _verified_name("[B]CC[B]", "1,2-di(λ1-boranyl)ethane") == "1,2-di(λ1-boranyl)ethane"
    assert _verified_name("[B]CC[B]", "(1S,1R)-1,2-di(λ1-boranyl)ethane") is None


def test_a_name_that_leaves_out_stereo_the_key_cannot_see_carries_the_stereo_unexpressed_flag():
    """Four configured C=C bonds of this input are single bonds in the form of the ylidene that the name states, so the
    full InChIKey has no layer for them: the name reads back to the key and says nothing about them, and the row says
    that it states fewer stereo elements than the input (the documented ``stereo_unexpressed``, as for any producer)."""
    from orthonym import Orthonym
    from orthonym.tiers import emit_tier_flags
    smiles = ("[B][B]B([B])/C(=C\\1/CC(C(N1C)C)C)/C(=C(\\B([B])[B])/C(=C(\\[B][B])/C(=C(\\[B])/C2=[N+](C(C(C2)C)C)C)/C)"
              "/C#C)/C#CC")
    row = Orthonym(style="pin", **emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["name"] and row["name"].endswith("-1,8-diyl"), row
    assert row["stereo_unexpressed"] is True and row["tier"] == "systematic_verified" and not row["is_pin"], row
