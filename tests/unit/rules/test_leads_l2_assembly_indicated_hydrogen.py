"""Leads L2 (B), item N9a: the indicated hydrogen of a ring assembly is cited in front of the assembly.

``****`` (the Blue Book): "Indicated hydrogen of components in two-component ring
assemblies is named and numbered according to, ignoring the indicated hydrogen atoms of the
component rings or ring systems. The maximum number of noncumulative double bonds is then added taking
into account the junction positions. Any remaining saturated ring positions are designated as
indicated hydrogen, placed together with the appropriate locant(s) at the front of the name of the
assembly.";:15595 "The citation of indicated hydrogen, if needed, at the front of the name of the ring
assembly is a change from its position in previous editions (refs. 1 and 2) where it was kept with the
name of the individual ring, for example, 2,2'-bi-2H-pyran.";:15599 "6H,6'H-2,2'-bipyran (PIN) (not
2,2'-bi-6H-pyran)";:15603 "1,1'-bipyrrole (PIN) (no indicated hydrogen needed) (not
1,1'-bi-1H-pyrrole)". A component name that starts with a locant follows the multiplier after a hyphen:
 (a),:6938 "to separate locants from words or word fragments", printed in the example of
 (:20912) "2,2'-bi-3,1,5-benzoxadiarsepine (PIN)".

The producer kept the component's own descriptor inside the parentheses ('2,4'-bi(4H-1,4-oxazine)') and,
for a prefix, copied the first ring's descriptor onto every ring, even onto a junction atom
('[1H,1'H-1,1'-bipyrrol]-3-yl', which OPSIN rejects). OPSIN 2.9.0 reads the old and the new spellings of
an assembly to the same structure, so only the spelling check ```` tells them apart.
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.validation import pin_spelling
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

RULE = "P-28.2.3"

#: (SMILES, the name the producer gave before, the PIN)
PARENTS = [
    ("C1(=CNC=CO1)N1C=COC=C1", "2,4'-bi(4H-1,4-oxazine)", "4H-2,4'-bi-1,4-oxazine"),
    ("O1C=CP(C=C1)P1C=COC=C1", "4,4'-bi(4H-1,4-oxaphosphinine)", "4,4'-bi-1,4-oxaphosphinine"),
    ("O1C=CN(C=C1)N1C=COC=C1", "4,4'-bi(4H-1,4-oxazine)", "4,4'-bi-1,4-oxazine"),
    ("C1(=CNC=CO1)C1=CNC=CO1", "2,2'-bi(4H-1,4-oxazine)", "4H,4'H-2,2'-bi-1,4-oxazine"),
    ("C1(OC=CO1)C1OC=CO1", "2,2'-bi(2H-1,3-dioxole)", "2H,2'H-2,2'-bi-1,3-dioxole"),
]
PARENT_IDS = ["oxazine-NH-and-N-junction", "oxaphosphinine-P-junction", "oxazine-N-N", "oxazine-2-2",
              "dioxole-2-2"]

#: (SMILES, the PIN of a prefix) -- the junction atom holds no indicated hydrogen
PREFIXES = [
    ("OCCc1ccn(c1)-n1cccc1", "2-([1,1'-bipyrrol]-3-yl)ethan-1-ol"),
    ("OCCC1=COC=CN1N1C=COC=C1", "2-([4,4'-bi-1,4-oxazin]-3-yl)ethan-1-ol"),
    # the free valence takes the lowest locant,:16109 "Low locants are assigned to ring
    # junctions, then to free valences"): the junction N is locant 4 in both walking directions, so
    # the carbon next to the ring oxygen is 2, never 6
    ("OCCC1=CN(C=CO1)N1C=COC=C1", "2-([4,4'-bi-1,4-oxazin]-2-yl)ethan-1-ol"),
]

#: the previous editions' spellings (:15595,:15599 'not 2,2'-bi-6H-pyran')
OLD_SPELLINGS = [
    "2,4'-bi(4H-1,4-oxazine)",
    "4,4'-bi(4H-1,4-oxaphosphinine)",
    "2,2'-bi-6H-pyran",
    "2,2'-bi-2H-pyran",
    "6-(2-hydroxyethyl)-4,4'-bi(4H-1,4-oxazine)",
]


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
            if f.rule == RULE]


# ------------------------------------------------------------------ the check alone
def test_the_check_is_registered_under_the_rule_it_cites():
    assert RULE in pin_spelling.registered_rules()


@pytest.mark.parametrize("smiles,old,pin", PARENTS, ids=PARENT_IDS)
def test_the_previous_editions_spelling_fails(smiles, old, pin):
    assert _rules(smiles, old) == [RULE]
    assert _rules(smiles, pin) == []


@pytest.mark.parametrize("name", [
    "6H,6'H-2,2'-bipyran",                  #:15599 (PIN)
    "1,1'-bipyrrole",                        #:15603 (PIN)
    "1H,1'H-1,1'-biindene",                  # the book's biindene PIN,:15605
    "2,2'-bi-3,1,5-benzoxadiarsepine",       #:20912 (PIN)
    "1,1'-bi(cyclohexane)",
    "[1,1'-biphenyl]-4-yl",
    "2,2'-bipyridine",
])
def test_the_books_assembly_pins_pass(name):
    assert _rules("C", name) == []


@pytest.mark.parametrize("name", OLD_SPELLINGS)
def test_every_old_spelling_is_read_without_a_structure(name):
    assert _rules("C", name) == [RULE]


@pytest.mark.parametrize("smiles,old,pin", PARENTS, ids=PARENT_IDS)
def test_the_pin_reads_back_to_the_full_inchikey(smiles, old, pin):
    # independent of the engine; OPSIN reads the old spelling too, which is why a check is needed
    assert_full_rt(pin, smiles)
    assert_full_rt(old, smiles)


@pytest.mark.parametrize("smiles,pin", PREFIXES)
def test_the_prefix_pin_reads_back_to_the_full_inchikey(smiles, pin):
    assert_full_rt(pin, smiles)


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


def _random_orders(smiles, n):
    mol = Chem.MolFromSmiles(smiles)
    out = {smiles}
    for _ in range(n * 4):
        if len(out) > n:
            break
        out.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
    return sorted(out)


@pytest.mark.parametrize("smiles,old,pin", PARENTS, ids=PARENT_IDS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_engine_names_the_assembly_with_the_indicated_hydrogen_in_front(namers, smiles, old, pin, tier):
    row = namers[tier].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.parametrize("smiles,old,pin", PARENTS, ids=PARENT_IDS)
def test_the_name_does_not_depend_on_the_atom_order(namers, smiles, old, pin):
    names = {namers["pin"].name_tiered(order)["name"] for order in _random_orders(smiles, 6)}
    assert names == {pin}


@pytest.mark.parametrize("smiles,pin", PREFIXES)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_engine_names_the_assembly_prefix_without_an_indicated_hydrogen_on_a_junction(
        namers, smiles, pin, tier):
    # '[1H,1'H-1,1'-bipyrrol]-3-yl' and '[4H,4'H-4,4'-bi-1,4-oxazin]-6-yl' put the descriptor on a
    # junction atom: OPSIN rejects them and the default tier used to abstain
    row = namers[tier].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.parametrize("smiles,pin", PREFIXES)
def test_the_prefix_name_does_not_depend_on_the_atom_order(namers, smiles, pin):
    names = {namers["pin"].name_tiered(order)["name"] for order in _random_orders(smiles, 6)}
    assert names == {pin}


def test_a_fused_component_keeps_its_name_and_loses_the_pin_label(namers):
    # per-ring indicated hydrogen of a FUSED component is not built (a separate written item): the
    # name stays the component's own descriptor inside parentheses, a correct non-PIN spelling that
    # the check labels below the PIN (the row records the rule); no name is lost
    smiles = "C1(=NC2=CC=CC=C2CO1)C1=NC2=CC=CC=C2CO1"
    row = namers["best-effort"].name_tiered(smiles)
    assert row["name"] == "2,2'-bi(4H-3,1-benzoxazine)"
    assert (row["tier"], row["is_pin"]) == ("systematic_verified", False), row
    assert RULE in [f["rule"] for f in row.get("spelling_failures") or []], row
    assert_full_rt(row["name"], smiles)
