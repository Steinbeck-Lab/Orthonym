"""Adducts with a radical component "Organic adducts", the Blue Book: "the individual components are
cited in the order of seniority of classes (see "; radicals head, so the radical is cited first).

OPSIN reads a radical component only as the LAST component and reads any other position as a substituent prefix of
the next one. Where the OPSIN in use reads adducts with a radical component (``opsin_capabilities.radical_adduct_enabled``;
OPSIN 2.9.0 reads none), the name ships in the order OPSIN reads, one component moved to the end
(``rules.adducts.readable_component_order``): an adduct name is no PIN,:4667; "these 'adduct
names' are not PINs",:37556), and a name no reader reads left these rows unnamed. The order twins
(``validation.twin_spellings``) still prove the structure of a name built in the class order.
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
    ("CBr.[B]c1ccccc1", "bromomethane—phenylboranylidene (1/1)"),
    ("CC.CC.[B]c1ccccc1", "ethane—phenylboranylidene (2/1)"),
])
def test_radical_mixture_ships_in_the_order_opsin_reads(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name
    assert r["tier"] == "systematic_verified" and not r["is_pin"]
    assert r["verified"] == "opsin"


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
def test_a_radical_cited_in_the_middle_ships_last():
    r = name_best_effort("C[As].C1=CC=NC=C1.N=O")
    assert r["verified"] == "opsin" and not r["is_pin"]
    assert r["name"] == "pyridine—2-oxa-1-azaeth-1-ene—methylarsanylidene (1/1/1)"


def _stub_reader(monkeypatch, readable):
    """OPSIN stood in by ``readable`` (the names it reads); every read reproduces the drawing."""
    import orthonym.namer as N
    from orthonym.rules import adducts as A
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: True)
    monkeypatch.setattr(N, "_validity_gate_name_to_smiles", lambda name: "C" if name in readable else None)
    monkeypatch.setattr(A, "_parse_reproduces_depiction", lambda name, mol: True)
    return A


def test_readable_order_moves_one_component_to_the_end(monkeypatch):
    A = _stub_reader(monkeypatch, {"a—c—b (1/3/2)"})
    assert A.readable_component_order("a—b—c (1/2/3)", None) == "a—c—b (1/3/2)"


def test_readable_order_keeps_a_name_opsin_reads_or_one_without_a_readable_order(monkeypatch):
    A = _stub_reader(monkeypatch, {"a—b—c (1/2/3)"})
    assert A.readable_component_order("a—b—c (1/2/3)", None) == "a—b—c (1/2/3)"
    A = _stub_reader(monkeypatch, set())
    assert A.readable_component_order("a—b—c (1/2/3)", None) == "a—b—c (1/2/3)"
    assert A.readable_component_order("ethanol", None) == "ethanol"


def test_readable_order_needs_the_radical_adduct_reader(monkeypatch):
    A = _stub_reader(monkeypatch, {"b—a (2/1)"})
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: False)
    assert A.readable_component_order("a—b (1/2)", None) == "a—b (1/2)"
