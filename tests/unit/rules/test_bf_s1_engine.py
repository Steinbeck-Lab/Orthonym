"""Slice S1 through the whole engine: stereodescriptors on bridged locants (Review Focus 4),
the label of names OPSIN cannot certify, and the von Baeyer fallback when a bridged fused
name is rejected."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s1"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", [
    ("O[C@@H]1C[C@@H]2C=C[C@H]1c1ccccc12", "(1R,4S,9R)-1,4-dihydro-1,4-ethanonaphthalen-9-ol"),
    ("CC1=C[C@H]2C[C@@H]1c1ccccc12", "(1S,4R)-2-methyl-1,4-dihydro-1,4-methanonaphthalene"),
    ("C[C@@H]1CC2c3ccccc3C1c1ccccc12", "(11R)-11-methyl-9,10-dihydro-9,10-ethanoanthracene"),
])
def test_bridge_and_bridgehead_descriptors_are_verified_pins(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified")
    assert name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("smiles", [
    "OC(=O)[C@@H]1[C@@H](C(O)=O)[C@H]2c3ccccc3[C@@H]1c1ccccc21",
    "C[C@@H]1C[C@H]2c3ccccc3[C@@H]1c1ccccc12",
])
def test_descriptors_opsin_cannot_place_are_declined_at_the_default_tier(smiles):
    # OPSIN 2.9.0: "Could not find atom that... 9R appeared to be referring to" for the
    # 9,10 bridgeheads of an 11- or 11,12-substituted 9,10-ethanoanthracene. The default
    # tier declines (never pin_unverified); best-effort keeps its result (an RT-exact
    # name, or none: these two have none).
    assert _row(smiles, "pin")["tier"] == "abstain"
    be = _row(smiles, "best-effort")
    assert be["tier"] != "pin_verified"
    if be["tier"] != "abstain":
        assert name_is_rt_exact(be["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "C1CC2CC1C1CCCCC12",                   # von Baeyer name from the von Baeyer path today
    "CC1=CC2CC1c1ccccc12",                 # von Baeyer name from the general engine today
    "CC1(C)C2=CC=CC(C)(C)C23C=CC1C3",      # the a dev split / milestone1500 row
])
def test_a_rejected_bridged_name_leaves_the_von_baeyer_fallback(smiles, monkeypatch):
    # Force a wrong bridged fused name: the OPSIN gate rejects it at the PIN tier, and
    # the best-effort tier still ships an RT-exact name (hard rule 2).
    import orthonym.rules.bridged_fused_pin as pkg
    real = pkg.build

    def wrong(mol):
        res = real(mol)
        return None if res is None else ("1,4-dihydro-1,4-methanonaphthalene",) + tuple(res[1:])
    monkeypatch.setattr(pkg, "build", wrong)
    assert _row(smiles, "pin")["tier"] == "abstain"
    be = _row(smiles, "best-effort")
    assert be["tier"] == "systematic_verified" and "cyclo[" in be["name"], be
    assert name_is_rt_exact(be["name"], smiles)


def test_a_general_bridge_inside_a_larger_name_is_not_a_pin():
    # Review Focus 1 continued: the package also records 'epithio':14097)
    smiles = "CC1=CC2SC1c1ccccc12"
    assert _row(smiles, "pin")["tier"] == "abstain"
    be = _row(smiles, "best-effort")
    assert be["name"] == "2-methyl-1,4-dihydro-1,4-epithionaphthalene"
    assert be["tier"] == "systematic_verified" and be["is_pin"] is False
    assert name_is_rt_exact(be["name"], smiles)
