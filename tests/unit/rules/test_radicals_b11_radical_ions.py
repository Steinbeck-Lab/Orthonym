"""Radical ions from mononuclear ionic parent hydrides.

the Blue Book-43422: the ionic parent hydride's name + 'yl'/'ylidene', "with
elision of the final letter 'e'". '-ylium' cations are NOT radical ions even
though RDKit counts their empty orbital as two radical electrons; they keep their
names (user decision Q2 = A: a separate build).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
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
    "CC(=O)[NH2+]",
    "C[N+](C)C",
    "C[N-]",
    "C[NH2+]",
    "[NH2+]c1ccccc1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)



def _strict_rt(name, smiles):
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _name(smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    with jvm_slots(1, purpose="test-radical-b11"):
        return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[B-](C)C", "trimethylboranuidyl"),                    # (PIN):43430
    ("COC(=O)[C-]", "(methoxycarbonyl)methanidylidene"),      # (PIN):43439
    ("[CH2+]", "methyliumyl"),                                # (PIN):43443
    ("CCC[OH+]", "propyloxidaniumyl"),                        # (PIN):43515
    ("CC(=O)[N-]", "acetylazanidyl"),                         # (PIN):43517
])
def test_radical_ion(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"]
    assert _strict_rt(expected, smiles)


# 'azanidyl' is the preselected name (:43426): a carbon-free compound has a
# preselected name at most, the Blue Book;:2062).
@pytest.mark.opsin_gate
def test_carbon_free_radical_ion_is_a_preselected_name():
    # Paper conformance (user decision 2026-09-30): the tier label is the one of the
    # naming path, as in the paper's measured run ('sodium chloride' pin_verified);
    # the name is unchanged.
    row = _name("[NH-]")
    assert row["name"] == "azanidyl" and row["tier"] == "pin_verified"
    assert row["is_pin"]
    assert _strict_rt("azanidyl", "[NH-]")


# "Radical ions on ionic suffix groups" (the Blue Book): "When ions
# may be named by using modified suffixes (see and, the
# suffixes denoting radical centers are added to the name of the cationic or anionic
# parent hydride" -- 'benzenaminiumyl (PIN)' (:43501), 'methanaminidyl (PIN)' (:43503),
# 'methanaminiumyl (PIN)' (:17608); an acyl group on N+ gives an amide cation
# (Table 7.4:41421 'amidium'). The round-trip parser reads none of those PINs, so they
# cannot be verified: the verified azanium/azanide name ships, never as pin_verified.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[NH2+]", "methylazaniumyl"),          # PIN methanaminiumyl (:17608)
    ("[NH2+]c1ccccc1", "phenylazaniumyl"),   # PIN benzenaminiumyl (:43501)
    ("C[N+](C)C", "trimethylazaniumyl"),     # PIN N,N-dimethylmethanaminiumyl
    ("CC(=O)[NH2+]", "acetylazaniumyl"),     # amidium-based PIN (Table 7.4)
    ("C[N-]", "methylazanidyl"),             # PIN methanaminidyl (:43503)
])
def test_suffix_parent_radical_ion_is_not_pin_verified(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected
    assert row["tier"] not in ("pin_verified", "abstain") and not row["is_pin"]
    assert _strict_rt(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[O+]", "methoxylium"),          # retained PIN, not 'methyloxidaniumylidene'
    ("[NH2+]", "azanylium"),           # gold row
    ("[O+]c1ccccc1", "phenoxylium"),
    ("[NH3+]C", "methanaminium"),      # closed-shell control
])
def test_ylium_and_closed_shell_cations_unchanged(smiles, expected):
    assert _name(smiles)["name"] == expected
