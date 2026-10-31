"""DT3 step B: a zwitterion drawing named as the neutral compound is the name of another drawing.

 "Ionic centers in the same parent structure" (the Blue Book,:42419): "zwitterionic compounds having
the ionic centers in the same parent structure are not considered as neutral compounds". The standard InChIKey does
not see where the hydrons sit, so the round trip passes for the neutral name; the label is systematic_verified (the
drawing as written is the structure to name, the contributor guide "Scope"), where it was pin_unverified. Only a name the strict
path does not build is lowered: ``L-alanine`` for the alanine zwitterion is the strict path's own name,
:54518: "to name the amino acid alanine as 2-aminopropanoic acid rather than as 2-azaniumylpropanoate") and keeps
pin_verified.
"""
import pytest

from orthonym import Orthonym, name_compound
from orthonym.cli import _emit_tier_flags
from orthonym.validation.zwitterion_drawing import named_as_neutral

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles, name, expected", [
    ("[NH3+]CC([O-])=O", "glycine", True),
    ("[NH3+][C@@H](CS)C(=O)[O-]", "(2R)-2-amino-3-sulfanylpropanoic acid", True),
    ("NC(=[NH2+])NCC(=O)[O-]", "(carbamimidoylamino)acetic acid", True),
    ("[NH3+]CC([O-])=O", "2-azaniumylacetate", False),                       # the name cites the cationic centre
    ("C[N+](C)(C)CC([O-])=O", "(trimethylazaniumyl)acetate", False),
    ("C[N+]1=CC=CC=C1C([O-])=O", "1-methylpyridin-1-ium-2-carboxylate", False),
    ("C[N+](=O)[O-]", "nitromethane", False),                               # a nitro group is no zwitterion
    ("C[N+]([O-])(C)C", "N,N-dimethylmethanamine N-oxide", False),          # nor an N-oxide
    ("[Na+].[O-]C(=O)C", "sodium acetate", False),                          # a salt: two components
    ("CCO", "ethanol", False),
])
def test_a_zwitterion_drawing_named_without_its_cationic_centre(smiles, name, expected):
    assert named_as_neutral(smiles, name) is expected


def test_no_structure_or_no_name_is_not_a_zwitterion_drawing():
    assert named_as_neutral("", "glycine") is False
    assert named_as_neutral("[NH3+]CC([O-])=O", "") is False
    assert named_as_neutral("not a smiles", "glycine") is False


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name, label", [
    ("[NH3+][C@@H](CS)C(=O)[O-]", "(2R)-2-amino-3-sulfanylpropanoic acid", "systematic_verified"),
    ("NC(=[NH2+])NCC(=O)[O-]", "(carbamimidoylamino)acetic acid", "systematic_verified"),
    ("C[C@H]([NH3+])C([O-])=O", "L-alanine", "systematic_verified"),         # the strict path's own name, no PIN
    ("C[N+]1=CC=CC=C1C([O-])=O", "1-methylpyridin-1-ium-2-carboxylate", "pin_verified"),
])
def test_the_label_of_a_zwitterion_drawing_at_the_best_effort_tier(smiles, name, label):
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, label)


@pytest.mark.opsin_gate
def test_the_pin_tier_is_unchanged():
    #:50943: the retained amino-acid name is no PIN, so the PIN tier declines it
    assert not name_compound("C[C@H]([NH3+])C([O-])=O", emit_tier="pin")
    assert name_compound("C[N+]1=CC=CC=C1C([O-])=O", emit_tier="pin") == "1-methylpyridin-1-ium-2-carboxylate"
