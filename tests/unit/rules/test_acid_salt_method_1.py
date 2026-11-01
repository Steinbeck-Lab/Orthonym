"""The word 'hydrogen' (method (2)) in the name of the salt of a polybasic organic acid is no PIN.

 (the Blue Book): "Acid salts of polybasic organic acids are named in two ways: (1) by substitutive
nomenclature in which the free acid is cited as a prefix to the name of the anion; (2) in the same way as the neutral
salts, the remaining acid hydrogen atoms being indicated by the word 'hydrogen'... Method (1) generates preferred
IUPAC names, except when the structure of the acid salt is unknown": 'potassium 6-carboxyhexanoate (PIN)' (:31602),
not 'potassium hydrogen heptanedioate'. The drawing of a salt gives the site of the hydron (the 2026-09-30 ruling of
the PIN class plan, Task 28). (:31620): the acid salts of organic derivatives of polybasic inorganic
oxoacids, carbonic acid included, are named by method (2) in a PIN: 'sodium hydrogen carbonate (PIN)' (:31623),
'potassium hydrogen methylphosphonate (PIN)'.

Where method (1) is not built (an anion of two carboxylates with a third group, a sulfonic acid), the method (2) name
was labelled pin_verified; it is a valid name below the PIN now.
"""
import pytest

from tests.support.default_tier import assert_default_tier_declines
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

METHOD_2_BELOW_THE_PIN = [
    ("OS(=O)(=O)CCS(=O)(=O)[O-].[Na+]", "sodium hydrogen ethane-1,2-disulfonate"),
    ("[O-]C(=O)CC(C(=O)[O-])C(=O)O.[Na+].[Na+]", "disodium hydrogen ethane-1,1,2-tricarboxylate"),
]


@pytest.mark.parametrize("smiles,expected", METHOD_2_BELOW_THE_PIN)
def test_the_hydrogen_word_of_an_organic_polybasic_acid_is_below_the_pin(smiles, expected):
    assert_default_tier_declines(smiles)
    be = name_best_effort(smiles)
    assert be["name"] == expected and be["tier"] == "systematic_verified" and be["is_pin"] is False, be
    assert_full_rt(be["name"], smiles, what="best-effort: ")


@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)CCCCCC(=O)[O-].[K+]", "potassium 6-carboxyhexanoate"),             # method (1),:31602
    ("OC(=O)CCC(=O)[O-].[NH4+]", "ammonium 3-carboxypropanoate"),              # method (1),:31606
    ("[Na+].OC(=O)[O-]", "sodium hydrogen carbonate"),                          #,:31623
    ("[O-]C(=O)CCC(=O)[O-].[Na+].[Na+]", "disodium butanedioate"),              # no hydrogen
])
def test_method_1_and_the_inorganic_oxoacid_salts_keep_their_label(smiles, pin):
    be = name_best_effort(smiles)
    assert be["name"] == pin and be["tier"] == "pin_verified", be


@pytest.mark.parametrize("smiles,pin", [
    # a hydron written as a fragment of its own has no site: "the structure of the acid salt is unknown", method (2) is
    # the PIN,:31608 'sodium hydrogen 2-(carboxylatomethyl)benzoate (PIN)')
    ("[K+].[H+].[O-]C(=O)CCCCCC(=O)[O-]", "potassium hydrogen heptanedioate"),
    ("[Na+].[H+].[O-]C(=O)Cc1ccccc1C(=O)[O-]", "sodium hydrogen 2-(carboxylatomethyl)benzoate"),
])
def test_an_open_hydron_site_keeps_method_2_as_the_pin(smiles, pin):
    be = name_best_effort(smiles)
    assert be["name"] == pin and be["tier"] == "pin_verified", be
