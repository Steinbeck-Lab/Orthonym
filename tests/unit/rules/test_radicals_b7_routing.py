"""A radical that also carries an internal charge pair (nitro, N-oxide) is a radical.

perception.ions.detect_species_type used to class it 'neutral' (charge first), so
closed-shell producers named it. Names must match and pass a strict OPSIN -r round
trip; the closed-shell controls keep their names.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from orthonym.perception.ions import detect_species_type
from tests.support.jars import jar_or_skip
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "[O]c1ccc(cc1)[N+](=O)[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



def _strict_rt(name, smiles):
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles,species", [
    ("[CH2]c1ccc(cc1)[N+](=O)[O-]", "radical"),
    ("[O]c1ccc(cc1)[N+](=O)[O-]", "radical"),
    ("[O-][N+](=O)c1ccccc1", "neutral"),
    ("C[N+](C)(C)[O-]", "neutral"),
])
def test_species(smiles, species):
    assert detect_species_type(Chem.MolFromSmiles(smiles)) == species


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,expected", [
    ("[O]c1ccc(cc1)[N+](=O)[O-]", "pin", "(4-nitrophenyl)oxyl"),
    ("[S]c1ccc(cc1)[N+](=O)[O-]", "pin", "(4-nitrophenyl)sulfanyl"),
    ("O=[NH2]C[CH2]", "pin", "2-(oxo-λ5-azanyl)ethyl"),          # BB
    ("[CH2]c1ccc(cc1)[N+](=O)[O-]", "best-effort", "(4-nitrophenyl)methyl"),
])
def test_named(smiles, tier, expected):
    with jvm_slots(1, purpose="test-radical-b7"):
        row = (_dt_row(smiles) if tier == "pin"
               else Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles))
    assert row["name"] == expected
    assert _strict_rt(expected, smiles)
