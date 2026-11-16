"""A mixture with a radical component is named by its components when the OPSIN in use reads adducts with a radical
component (``opsin_capabilities.radical_adduct_enabled``; OPSIN 2.9.0 reads none, so its names stay as before).

 "Organic adducts" (the Blue Book): "Names are formed by citing the names of individual compounds". A
tin or lead radical piece takes its radical name ('tribenzylstannyl', 'plumbylidene') only in the last
producer of a mixture name (``rules.adducts.in_component_adduct_scope``), so a mixture named today through the
lambda-convention name of its piece keeps that name.
"""
import pytest

from orthonym import opsin_capabilities as caps
from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _reads_radical_adducts() -> bool:
    try:
        return caps.radical_adduct_enabled()
    except Exception:
        return False


needs_radical_adducts = pytest.mark.skipif(not _reads_radical_adducts(),
                                           reason="the OPSIN in use reads no adduct with a radical component")

_RADICAL = [
    ("[Br-].[I]", "bromide—iodanyl (1/1)", "opsin"),
    ("CCO.[O]", "ethanol—oxidanylidene (1/1)", "opsin"),
    ("CC(=O)O.C1=CC=C(C=C1)C[Sn](CC2=CC=CC=C2)CC3=CC=CC=C3", "acetic acid—tribenzylstannyl (1/1)", "opsin"),
    ("CCCC=CC(=O)O.[PbH2]", "hex-2-enoic acid—plumbylidene (1/1)", "opsin"),
    ("O.O.[Ni].[SnH2]", "nickel—stannylidene—water (1/1/2)", "opsin_twin"),
]


@needs_radical_adducts
@pytest.mark.parametrize("smiles,expected,verified", _RADICAL)
def test_radical_component_named_when_opsin_reads_it(smiles, expected, verified):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["verified"]) == (expected, "systematic_verified", verified), res
    assert not res["is_pin"]
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles", [s for s, _n, _v in _RADICAL])
def test_no_radical_component_without_the_capability(monkeypatch, smiles):
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: False)
    assert name_best_effort(smiles)["name"] is None


@pytest.mark.parametrize("capability", [True, False])
@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)CC(C)CC(=O)O.CC(C)(C)CC(C)CC(=O)O.CCCCCCCC[Sn]CCCCCCCC",
     "3,5,5-trimethylhexanoic acid—9λ2-stannaheptadecane (2/1)"),
    ("CC(=O)O.CCC[Sn]CCC", "acetic acid—4λ2-stannaheptane (1/1)"),
    ("CCC[Sn]CCC.[Na+].[Cl-]", "4λ2-stannaheptane—chloride—sodium(1+) (1/1/1)"),
])
def test_tin_mixtures_named_today_keep_their_names(monkeypatch, capability, smiles, expected):
    if capability and not _reads_radical_adducts():
        pytest.skip("the OPSIN in use reads no adduct with a radical component")
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: capability)
    res = name_best_effort(smiles)
    assert (res["name"], res["verified"]) == (expected, "opsin"), res
    assert name_is_rt_exact(res["name"], smiles)
