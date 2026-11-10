"""A charge-separated sulfoxide that gets no name as drawn is named through its S=O drawing (program 22, 2026-11-10).

 SULFOXIDES AND SULFONES (the Blue Book): 'R-SO-R′'. The drawing 'C[S@+]([O-])...' and 'C[S@](=O)...'
are one compound. ``assembly/sulfoxide_drawing.py`` names the S=O drawing only when the drawing as written gets no name,
builds it from the standard reading of the lone-pair centre (the lone pair an implicit hydrogen, as CDK and OPSIN read
it; RDKit reads '[O-][S@@+]1CCC...' the other way), and labels the name systematic_verified.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate

_FUSED = "[O-][S@@+]1CCC=2N=C(N=C(C21)NC)N"


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


def test_fused_sulfoxide_named_through_its_s_o_drawing(best_effort):
    row = best_effort.name_tiered(_FUSED)
    # (R) by hand: from O-, the lone pair, C7a and CH2 run clockwise ('@@'); O > C7a > CH2 > lone pair
    assert (row["name"], row["tier"], row["source"]) == (
        "(7R)-3-amino-5-(methylamino)-7-oxo-7-thia-2,4-diazabicyclo[4.3.0]nona-1(6),2,4-triene",
        "systematic_verified", "sulfoxide_drawing")


def test_the_drawing_is_built_from_the_standard_reading():
    from orthonym.assembly.sulfoxide_drawing import s_o_drawing
    from orthonym.perception.lone_pair_reading import standard_centres_labels
    drawing = s_o_drawing(_FUSED)
    assert drawing is not None and "+" not in drawing and "-]" not in drawing
    assert set(standard_centres_labels(drawing).values()) == set(standard_centres_labels(_FUSED).values()) == {"R"}


@pytest.mark.parametrize("smiles", ["C[S@+]([O-])c1ccccc1", "[O-][S+]1CCc2ncncc21"])
def test_a_drawing_named_as_written_keeps_its_name(best_effort, smiles):
    assert best_effort.name_tiered(smiles)["source"] != "sulfoxide_drawing"


def test_default_tier_declines(smiles=_FUSED):
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] == "abstain" and row["source"] != "sulfoxide_drawing"
