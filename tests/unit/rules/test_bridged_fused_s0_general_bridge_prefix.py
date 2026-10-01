""" (the Blue Book-14110): 'sulfano', 'disulfano', 'selano', 'tellano' and 'azano'
are the preselected bridge prefixes; 'epithio', 'epidithio', 'episeleno', 'epitelluro' and 'epimino'
are general nomenclature (:14097 "may be used in general nomenclature"; PIN example:28063
'1,4-dihydro-1,4-sulfanonaphthalene (PIN)'). OPSIN 2.9.0 cannot read the preselected forms, so a
name with a general-nomenclature bridge is never labelled a PIN: the default tier declines it and
the best-effort tier ships it as systematic_verified. 'epoxy' is itself preselected (:14099) and
keeps its PIN."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

GENERAL_BRIDGE_ROWS = [
    ("C12SC(C=C1)c1ccccc12", "1,4-dihydro-1,4-epithionaphthalene"),
    ("C12SC(CC1)c1ccccc12", "1,2,3,4-tetrahydro-1,4-epithionaphthalene"),
    ("C12NC(C=C1)c1ccccc12", "1,4-dihydro-1,4-epiminonaphthalene"),
    ("C12NC(CC1)c1ccccc12", "1,2,3,4-tetrahydro-1,4-epiminonaphthalene"),
]
SUBSTITUTED_ROWS = [  # the bridge label follows the name into a larger name
    "CC1=CC2SC1c1ccccc12",
    "OC1=CC2SC1c1ccccc12",
    "OC(=O)CC12SC(C=C1)c1ccccc12",
]
PRESELECTED_CONTROL = ("C12OC(C=C1)c1ccccc12", "1,4-dihydro-1,4-epoxynaphthalene")


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s0"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", GENERAL_BRIDGE_ROWS)
def test_general_bridge_name_is_declined_at_default_and_kept_at_best_effort(smiles, name):
    pin = _row(smiles, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    be = _row(smiles, "best-effort")
    assert be.get("name") == name, be.get("name")
    assert be["tier"] == "systematic_verified" and be["is_pin"] is False, be["tier"]
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", SUBSTITUTED_ROWS)
def test_general_bridge_inside_a_larger_name_is_not_a_pin(smiles):
    be = _row(smiles, "best-effort")
    name = be.get("name") or ""
    assert name_is_rt_exact(name, smiles), name
    assert not ("epithio" in name and be["tier"] == "pin_verified"), (name, be["tier"])
    pin = _row(smiles, "pin")
    assert not ("epithio" in (pin.get("name") or "") and pin["tier"] == "pin_verified"), pin.get("name")


def test_a_recorded_general_bridge_demotes_every_name_that_carries_it():
    """The record the producer leaves is name-scoped: it demotes the bare parent and any larger
    name built on it, and no name without the prefix."""
    from rdkit import Chem

    from orthonym.metrics import provenance as pv
    from orthonym.rules.bridged_fused import GENERAL_ONLY_BRIDGE_PREFIXES, name_bridged_fused_pin

    assert GENERAL_ONLY_BRIDGE_PREFIXES == frozenset(
        {"epithio", "epidithio", "episeleno", "epitelluro", "epimino"})
    pv.clear_provenance()
    res = name_bridged_fused_pin(Chem.MolFromSmiles("C12SC(C=C1)c1ccccc12"))
    assert res is not None and res[0] == "1,4-dihydro-1,4-epithionaphthalene", res
    prov = pv.get_provenance()
    assert pv.name_carries_non_pin_part(prov, res[0])
    assert pv.name_carries_non_pin_part(prov, "2-methyl-1,4-dihydro-1,4-epithionaphthalene")
    assert not pv.name_carries_non_pin_part(prov, "1,4-dihydro-1,4-epoxynaphthalene")
    pv.clear_provenance()
    res = name_bridged_fused_pin(Chem.MolFromSmiles("C12OC(C=C1)c1ccccc12"))
    assert res is not None and res[0] == PRESELECTED_CONTROL[1], res
    assert not pv.name_carries_non_pin_part(pv.get_provenance(), res[0])
    pv.clear_provenance()


@pytest.mark.opsin_gate
def test_preselected_epoxy_bridge_keeps_its_pin():
    smiles, name = PRESELECTED_CONTROL
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row.get("name") == name and row["tier"] == "pin_verified", (tier, row.get("name"), row["tier"])


UNTABULATED_HETERO_BRIDGE_ROWS = [  # -SiH2-, -Se-, -PH-, -Te-, -BH-, -GeH2- across naphthalene 1,4
    "C12[SiH2]C(C=C1)c1ccccc12",
    "C12[Se]C(C=C1)c1ccccc12",
    "C12[PH]C(C=C1)c1ccccc12",
    "C12[Te]C(C=C1)c1ccccc12",
    "C12[BH]C(C=C1)c1ccccc12",
    "C12[GeH2]C(C=C1)c1ccccc12",
]


@pytest.mark.parametrize("smiles", UNTABULATED_HETERO_BRIDGE_ROWS)
def test_a_heteroatom_bridge_without_a_tabulated_prefix_gets_no_bridged_name(smiles):
    """The bridge prefix table names O, S and NH bridges only; any other heteroatom bridge must
    not fall through to the carbon table ('1,4-methano...' would describe a different molecule)."""
    from rdkit import Chem

    from orthonym.rules.bridged_fused import get_bridge_prefix, name_bridged_fused_pin

    mol = Chem.MolFromSmiles(smiles)
    het = next(a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "C")
    assert get_bridge_prefix({"length": 1, "element": het, "heteroatom": het}) == ""
    assert name_bridged_fused_pin(mol) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", UNTABULATED_HETERO_BRIDGE_ROWS[:3])
def test_a_heteroatom_bridge_without_a_tabulated_prefix_keeps_its_best_effort_name(smiles):
    be = _row(smiles, "best-effort")
    assert be["tier"] != "abstain" and name_is_rt_exact(be["name"], smiles), (be.get("name"), be["tier"])
