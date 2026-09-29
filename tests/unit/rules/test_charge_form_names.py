"""P0 charge-form names: a shipped name reproduces the input's charge form.

The class: a cation drawn beside a NEUTRAL inorganic oxoacid, with no anion (a
metal ion or an organic cation with carbonic, sulfuric, phosphoric or perchloric
acid; net charge > 0). name used to deprotonate the acid before naming, so the
salt name ('sodium hydrogen carbonate') was checked against the rewritten, balanced
structure and shipped as verified although OPSIN reads it back as a different
protonation state: [Na+].OC(=O)O has the full an InChIKey,
the name's parse C(O)([O-])=O.[Na+] has an InChIKey.

The drawn species is a mixed adduct of the cation and the acid. "Mixed
organic - inorganic adducts" (the Blue Book): "preferred IUPAC names cannot be
assigned to mixed adducts... names are formed by citing the names of individual
compounds, connected by long (em) dashes (—). The proportions of the components are
indicated after the name by an arabic number(s) separated by a solidus enclosed in
parentheses". So the PIN tier abstains and the best-effort tier gives the adduct
name, which OPSIN reads back to the input's full key.

Members: the three metal witnesses below, and the eight rows of the 1,000,000-row
PubChem set whose names differed from the input only in the protonation flag (the
organic-cation rows here are three of them).
"""
import pytest

from orthonym import name_compound
from orthonym.errors import is_failure_name
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

NAHCO3_IMBALANCED = "[Na+].OC(=O)O"

MEMBERS = [
    NAHCO3_IMBALANCED,
    "[K+].OS(=O)(=O)O",
    "[Na+].[Na+].OP(=O)(O)O",
    "C1CCC2=C(C1)C=NC3=CC=CC=[N+]23.OCl(=O)(=O)=O",
    "CC1=CC(=[N+]2C(=N1)SC(=N2)C)C.OS(=O)(=O)O",
    "C1=CC=C(C=C1)C(=O)C(NC(=O)C2=CC=CC=C2)[P+](C3=CC=CC=C3)(C4=CC=CC=C4)C5=CC=CC=C5"
    ".OCl(=O)(=O)=O",
]


def test_sodium_hydrogen_carbonate_is_not_the_name_of_the_imbalanced_input():
    # 'sodium hydrogen carbonate' is NaHCO3 (net 0); the input is Na+ beside
    # neutral carbonic acid (net +1). The default tier has no PIN for a mixed
    # adduct and abstains; the best-effort tier names the adduct.
    pin = name_compound(NAHCO3_IMBALANCED)
    assert pin != "sodium hydrogen carbonate"
    assert is_failure_name(pin), pin
    be = name_best_effort(NAHCO3_IMBALANCED)["name"]
    assert be != "sodium hydrogen carbonate"
    assert be == "carbonic acid—sodium(1+) (1/1)"


@pytest.mark.parametrize("smiles", MEMBERS)
def test_every_member_ships_only_a_name_of_the_drawn_charge_form(smiles):
    row = name_best_effort(smiles)
    be = row["name"]
    # breadth: the best-effort tier names the drawn species, and OPSIN (a fresh
    # call outside the engine) reads the name back to the input's full InChIKey,
    # protonation flag included
    assert be and row["source"] != "abstain", row
    assert name_is_rt_exact(be, smiles), (smiles, be)
    assert row["verified"] == "opsin"
    # the default tier may abstain, but whatever it ships must round-trip too
    pin = name_compound(smiles)
    assert is_failure_name(pin) or name_is_rt_exact(pin, smiles), (smiles, pin)
