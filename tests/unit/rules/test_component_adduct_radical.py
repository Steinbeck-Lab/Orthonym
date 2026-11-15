"""Adducts with a radical component "Organic adducts", the Blue Book: "the individual components are
cited in the order of seniority of classes (see "; radicals head, so the radical is cited first).

OPSIN reads a leading radical word as a substituent prefix of the rest and reads the same components in the other
order, so the name ships through its component-order twin (``validation.twin_spellings.component_order_twin``) when
the OPSIN in use reads adducts with a radical component (``opsin_capabilities.radical_adduct_enabled``; OPSIN 2.9.0
reads none). The twin proves the structure, never the spelling; the label is ``opsin_twin`` and never a PIN.
"""
import pytest

from orthonym import opsin_capabilities as caps
from orthonym.validation import twin_spellings as T
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


def test_component_order_twin_moves_the_first_component_and_its_proportion():
    assert T.component_order_twin("phenylboranylidene—bromomethane (1/1)") == \
        "bromomethane—phenylboranylidene (1/1)"
    assert T.component_order_twin("phenylboranylidene—ethane (1/2)") == "ethane—phenylboranylidene (2/1)"
    assert T.component_order_twin("a—b—c (1/2/3)") == "b—c—a (2/3/1)"


def test_component_order_twin_needs_an_adduct_name():
    assert T.component_order_twin("ethanol") is None
    assert T.component_order_twin("benzene—water") is None              # no proportions
    assert T.component_order_twin("benzene—water (1/1/1)") is None      # proportions do not match


def test_radical_adduct_probe_reads_with_the_radical_setting():
    name, smiles, radical = caps._PROBES["radical_adduct"]
    assert name == "ethane—boranylidyne (1/1)" and smiles == "[B].CC" and radical is True
    assert all(len(v) == 3 for v in caps._PROBES.values())


def test_capability_set_without_the_entry_reads_as_cannot(monkeypatch):
    monkeypatch.setattr(caps, "opsin_capabilities",
                        lambda: {"phane": True, "halogen_uide": True, "composite_assembly": True})
    assert caps.radical_adduct_enabled() is False


def test_no_twin_without_the_capability(monkeypatch):
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: False)
    assert T.component_order_round_trip("phenylboranylidene—bromomethane (1/1)", "CBr.[B]c1ccccc1") is None


def _main_reads_radical_adducts() -> bool:
    try:
        return caps.radical_adduct_enabled()
    except Exception:
        return False


needs_main = pytest.mark.skipif(not _main_reads_radical_adducts(),
                                reason="the OPSIN in use reads no adduct with a radical component")


@needs_main
@pytest.mark.parametrize("smiles, name", [
    ("CBr.[B]c1ccccc1", "phenylboranylidene—bromomethane (1/1)"),
    ("CC.CC.[B]c1ccccc1", "phenylboranylidene—ethane (1/2)"),
])
def test_radical_mixture_ships_through_its_order_twin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name
    assert r["tier"] == "systematic_verified" and not r["is_pin"]
    assert r["verified"] == "opsin_twin"
    assert r["gate_outcome"] == "twin_verified:component_order"


@needs_main
def test_twin_of_another_structure_is_refused():
    # one hydrogen more on boron: the key differs, the twin proves nothing
    assert T.component_order_round_trip("phenylboranylidene—bromomethane (1/1)", "CBr.[BH]c1ccccc1") is None


@needs_main
def test_two_radical_components_have_no_readable_order():
    twin = T.component_order_twin("phenylboranylidene—dimethylarsanyl (1/1)")
    assert twin == "dimethylarsanyl—phenylboranylidene (1/1)"
    assert T.component_order_round_trip("phenylboranylidene—dimethylarsanyl (1/1)",
                                        "[B]c1ccccc1.C[As]C") is None


@needs_main
def test_adduct_opsin_can_read_itself_takes_no_twin():
    r = name_best_effort("CCO.CC(=O)O")
    assert r["name"] == "acetic acid—ethanol (1/1)" and r["verified"] == "opsin"


def test_component_order_twins_are_every_rotation():
    assert T.component_order_twins("a—b—c (1/2/3)") == ["b—c—a (2/3/1)", "c—a—b (3/1/2)"]
    assert T.component_order_twins("a—b—c—d—e—f—g (1/1/1/1/1/1/1)") == []


@needs_main
def test_a_radical_cited_in_the_middle_reads_back_through_a_rotation():
    r = name_best_effort("C[As].C1=CC=NC=C1.N=O")
    assert r["gate_outcome"] == "twin_verified:component_order" and r["verified"] == "opsin_twin"
    assert r["name"] == "pyridine—methylarsanylidene—2-oxa-1-azaeth-1-ene (1/1/1)"
