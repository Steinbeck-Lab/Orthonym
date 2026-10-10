"""Leads L2 (F), item N9b producer: a ring assembly of von Baeyer (bicyclo) components is named as one.

``### **** DEFINITIONS`` (the Blue Book),:15542 lists "alicyclic von Baeyer systems" among the
cyclic systems of a ring assembly; ``****`` (:24072), example:24076 "2,2'-bi(bicyclo[2.2.2]
octane) (PIN)"; ``### ****`` (:15560),:15564 "Parentheses are used to avoid confusion with von
Baeyer names". The component is the von Baeyer parent hydride itself; the junction takes the lowest locant
its own numbering allows (7 for the one-atom bridge of bicyclo[2.2.1]heptane).

``_get_ring_parent_name`` named only fused systems, so ``name_ring_assembly`` returned None for two
bicyclo components and a later substitutive producer's name ('2-(bicyclo[2.2.2]octan-2-yl)bicyclo[2.2.2]
octane') was certified. The check (``test_leads_l2_assembly_label``) withdraws that label; this
producer builds the assembly name. Heteroatomic von Baeyer assemblies ('5,6'-diaza-2,2'-bi(bicyclo
[2.2.2]octane)':15685, which needs the 'a' prefixes of in front), substituted ones and spiro
components are not built (see the lane's open items).
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

#: (id, SMILES, PIN)
ROWS = [
    ("octane-2-2", "C1(C2CC3CCC2CC3)CC2CCC1CC2", "2,2'-bi(bicyclo[2.2.2]octane)"),     #:24076 (PIN)
    ("heptane-7-7", "C1CC2CCC1C2C1C2CCC1CC2", "7,7'-bi(bicyclo[2.2.1]heptane)"),
    ("heptane-2-7", "C1CC2CCC1C2C1CC2CCC1C2", "2,7'-bi(bicyclo[2.2.1]heptane)"),
    ("octane-1-1", "C12(CCC(CC1)CC2)C12CCC(CC1)CC2", "1,1'-bi(bicyclo[2.2.2]octane)"),
    ("octane-1-2", "C12(CCC(CC1)CC2)C1CC2CCC1CC2", "1,2'-bi(bicyclo[2.2.2]octane)"),
]
IDS = [r[0] for r in ROWS]

TER = ("C1CC2CCC1CC2C1CC2CCC1CC2C1CC2CCC1CC2", "2,2':5',2''-terbicyclo[2.2.2]octane")


# ------------------------------------------------------------------ the names read back
@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
def test_the_name_reads_back_to_the_full_inchikey(_id, smiles, pin):
    assert_full_rt(pin, smiles)


def test_the_three_component_name_reads_back_to_the_full_inchikey():
    assert_full_rt(TER[1], TER[0])


@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
def test_the_assembly_checks_pass_the_name(_id, smiles, pin):
    #: the name cites its assembly;: no 'bicyclo' without its descriptor
    assert [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), pin, strict=True)] == []


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_engine_names_the_von_baeyer_assembly(namers, _id, smiles, pin, tier):
    row = namers[tier].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
def test_the_locants_do_not_depend_on_the_atom_order(namers, _id, smiles, pin):
    mol = Chem.MolFromSmiles(smiles)
    orders = {smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(10)}
    assert {namers["pin"].name_tiered(o)["name"] for o in orders} == {pin}


def test_three_components_are_named_without_parentheses_and_are_not_a_pin(namers):
    # (:15651) and:16853 '[1,1':4',1''-terbicyclo[2.2.2]octane]-2,2',2''-triene (PIN)': no
    # parentheses after 'ter'; the primed numbering is not the PIN numbering,:24078)
    smiles, name = TER
    row = namers["best-effort"].name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert "P-52.2.7.2" in [f["rule"] for f in row.get("spelling_failures") or []], row
    mol = Chem.MolFromSmiles(smiles)
    orders = {smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(10)}
    assert {namers["best-effort"].name_tiered(o)["name"] for o in orders} == {name}


def test_a_substituted_assembly_keeps_the_substitutive_name_and_loses_the_pin_label(namers):
    # not built: the von Baeyer locants of a substituent. The name is kept (best effort), the label is
    # withdrawn by the check, and the default tier declines
    smiles = "CC1(C2CC3CCC2CC3)CC2CCC1CC2"
    row = namers["best-effort"].name_tiered(smiles)
    assert (row["tier"], row["is_pin"]) == ("systematic_verified", False), row
    assert "P-28.1" in [f["rule"] for f in row.get("spelling_failures") or []], row
    assert_full_rt(row["name"], smiles)
    assert namers["pin"].name_tiered(smiles)["tier"] == "abstain"


def test_a_heteroatomic_assembly_is_not_named_as_an_assembly(namers):
    # the book's '5,6'-diaza-2,2'-bi(bicyclo[2.2.2]octane)' (:15685) needs the 'a' prefixes of
    smiles = "C12C(C3CC4CCC3NC4)CC(NC1)CC2"
    assert namers["pin"].name_tiered(smiles)["tier"] == "abstain"
