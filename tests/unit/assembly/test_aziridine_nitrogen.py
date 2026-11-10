"""An aziridine nitrogen whose tag RDKit drops is cited in the name (program 22, 2026-11-10).

RDKit's parser removes the tag of the nitrogen of an N-acyl aziridine; the standard InChI and CDK keep it.
'Trigonal pyramid' (the Blue Book) treats the centre as tetrahedral with a phantom atom, and the nitrogen of an
aziridine is position 1,:8222). ``assembly/aziridine_nitrogen.py`` adds '1R' or '1S' to the name of
RDKit's reading when the drawing as written gets no name, and labels it systematic_verified.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    # a new group in front of the parent
    ("CC1[N@@](C1)C(=O)OC", "methyl (1R)-2-methylaziridine-1-carboxylate"),
    ("CC1[N@](C1)C(=O)OC", "methyl (1S)-2-methylaziridine-1-carboxylate"),
    # merged into the group the producer wrote
    ("C[C@H]1[N@@](C1)C(=O)OC", "methyl (1R,2R)-2-methylaziridine-1-carboxylate"),
    # inside a substituent: the enclosing marks step up
    ("CC1[N@@](C1)C(=O)c1ccccc1", "[(1R)-2-methylaziridin-1-yl](phenyl)methanone"),
    ("COC(=O)C1[N@](C1)C(=O)N1C[C@H](CC1)C(=O)OCC1=CC=CC=C1",
     "benzyl (3S)-1-[(1S)-2-(methoxycarbonyl)aziridine-1-carbonyl]pyrrolidine-3-carboxylate"),
])
def test_aziridine_nitrogen_configuration_cited(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"], row["source"]) == (name, "systematic_verified", "aziridine_nitrogen")


def test_a_tag_rdkit_keeps_is_named_by_the_producer(best_effort):
    row = best_effort.name_tiered("CC1[N@@](C1)C")
    assert row["name"] == "(1R)-1,2-dimethylaziridine" and row["source"] != "aziridine_nitrogen"


def test_default_tier_declines():
    row = Orthonym().name_tiered("CC1[N@@](C1)C(=O)OC")
    assert row["tier"] == "abstain" and row["source"] != "aziridine_nitrogen"


def test_nitrogen_written_at_a_ring_closure_digit(best_effort):
    """RDKit's tag-keeping reading reads 'C[N@]1' the other way; the name follows CDK and OPSIN (the lone pair an
    implicit hydrogen), and the two spellings of one compound get one name."""
    a = best_effort.name_tiered("CC1C[N@]1C(=O)OC")
    b = best_effort.name_tiered("[N@@]1(C(=O)OC)CC1C")
    assert a["name"] == b["name"] and a["source"] == "aziridine_nitrogen", (a, b)


def test_name_and_name_with_confidence_take_the_fallback(best_effort):
    name = best_effort.name("CC1[N@@](C1)C(=O)OC")
    record = best_effort.name_with_confidence("CC1[N@@](C1)C(=O)OC")
    assert name == record["name"] == "methyl (1R)-2-methylaziridine-1-carboxylate"
    assert record["tier"] == "systematic_verified"


def test_fallback_never_raises_a_tier():
    from orthonym.assembly.decline_fallback import capped
    row = capped({"tier": "systematic_verified", "name": "x"}, {"tier": "best_effort", "spelling_failures": ["s"]})
    assert row["tier"] == "best_effort" and row["spelling_failures"] == ["s"]
