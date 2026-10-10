"""Leads L2 (D), item N9c: the spelling residuals of the primed names of three or more components.

The composite-locant PIN producer is NOT built (OPSIN 2.9.0 reads no composite locant, so no independent
check exists; the label half is closed by ``_multiplied_assembly``, which records the non-PIN rule
 for every primed name of three or more components). What is built here is the spelling of those
names at the tiers that keep them:

* ``### **** UNBRANCHED RING ASSEMBLIES OF THREE THROUGH SIX IDENTICAL CYCLIC SYSTEMS``
  (the Blue Book), ``****`` (:15651): "...placing the appropriate numerical Latin-based
  (see prefix, 'ter', 'quater', 'quinque', etc., before the name of the parent hydride
  corresponding to the repeating unit." The book prints no parentheses after 'ter': "1,1':2',1''-
  tercyclopropane" (:15657), "[1,1':4',1''-tercyclohexane]-1',2-diene (PIN)" (:16849) and
  "[1,1':4',1''-terbicyclo[2.2.2]octane]-2,2',2''-triene (PIN)" (:16853); the parentheses of
  (:15564) avoid 'bi' + 'cyclo' = 'bicyclo', which 'ter' + 'cyclo' is not.
* the middle ring takes the lowest junction locant set in the order of citation::15663 "the locant set
  1,1':2',1'':3'',1''' is lower than 1,1':3',1'':2'',1'''" (quatercyclobutane),:15667
  "2,2':6',2'':6'',2'''-quaterpyridine", so '2,2':5',2''-terthiophene', not '2,5':2',2''-terthiophene'.
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules import ring_assemblies as ra
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

RULE = "P-52.2.7.2"

#: (id, SMILES, the best-effort name before, the name now)
ROWS = [
    ("tercyclopropane", "C1CC1C1CC1C1CC1", "1,1':2',1''-ter(cyclopropane)", "1,1':2',1''-tercyclopropane"),
    ("tercyclohexane", "C1CCC(CC1)C1CCC(CC1)C1CCCCC1", "1,1':4',1''-ter(cyclohexane)",
     "1,1':4',1''-tercyclohexane"),
    ("tercyclohexanol", "OC1CCC(CC1)C1CCC(CC1)C1CCCCC1", "[1,1':4',1''-ter(cyclohexan)]-4-ol",
     "[1,1':4',1''-tercyclohexan]-4-ol"),
    ("terthiophene", "s1cccc1-c1ccc(s1)-c1cccs1", "2,5':2',2''-terthiophene", "2,2':5',2''-terthiophene"),
    ("terfuran", "c1cc(oc1)-c1ccc(o1)-c1ccco1", "2,5':2',2''-terfuran", "2,2':5',2''-terfuran"),
    ("quaterfuran", "c1cc(oc1)-c1ccc(o1)-c1ccc(o1)-c1ccco1", "2,5':2',5'':2'',2'''-quaterfuran",
     "2,2':5',2'':5'',2'''-quaterfuran"),
    ("terpyridine", "c1ccc(nc1)-c1cccc(n1)-c1ccccn1", "2,2':6',2''-terpyridine", "2,2':6',2''-terpyridine"),
    ("terphenyl", "c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", "1,1':4',1''-terphenyl", "1,1':4',1''-terphenyl"),
]
IDS = [r[0] for r in ROWS]


def _random_orders(smiles, n):
    mol = Chem.MolFromSmiles(smiles)
    out = {smiles}
    for _ in range(n * 4):
        if len(out) > n:
            break
        out.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
    return sorted(out)


# ------------------------------------------------------------------ the writer
def test_the_parentheses_of_the_component_stay_for_two_rings_only():
    join = ra._multiplied_assembly
    assert join("1,1'", "bi", ra._enclose_component("cyclopropane"), 2) == "1,1'-bi(cyclopropane)"
    assert join("1,1':2',1''", "ter", ra._enclose_component("cyclopropane"), 3) == "1,1':2',1''-tercyclopropane"
    assert join("1,1':4',1'':4'',1'''", "quater", ra._enclose_component("bicyclo[2.2.2]octane"), 4) == \
        "1,1':4',1'':4'',1'''-quaterbicyclo[2.2.2]octane"
    assert join("1,1'", "bi", ra._enclose_component("bicyclo[2.2.2]octane"), 2) == "1,1'-bi(bicyclo[2.2.2]octane)"
    # a component whose own marks do not enclose all of it keeps them
    assert ra._is_enclosed("(cyclohexane)") and not ra._is_enclosed("(a)b(c)") and not ra._is_enclosed("a(b)")


# ------------------------------------------------------------------ the names read back
@pytest.mark.parametrize("_id,smiles,old,new", ROWS, ids=IDS)
def test_the_new_name_reads_back_to_the_full_inchikey(_id, smiles, old, new):
    # independent of the engine; OPSIN 2.9.0 also reads the old spelling, so only the book decides
    assert_full_rt(new, smiles)


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


@pytest.mark.parametrize("_id,smiles,old,new", ROWS, ids=IDS)
def test_best_effort_names_the_assembly_as_the_book_spells_it(namers, _id, smiles, old, new):
    row = namers["best-effort"].name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (new, "systematic_verified", False), row
    assert RULE in [f["rule"] for f in row.get("spelling_failures") or []], row


@pytest.mark.parametrize("_id,smiles,old,new", ROWS, ids=IDS)
def test_the_default_tier_still_declines_a_primed_name_of_three_components(namers, _id, smiles, old, new):
    # the PIN uses composite locants,:24078); the producer is a separate written item
    row = namers["pin"].name_tiered(smiles)
    assert (row["tier"], row["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), row


@pytest.mark.parametrize("smiles", [r[1] for r in ROWS if r[0] in ("terthiophene", "quaterfuran", "terpyridine")])
def test_the_numbering_does_not_depend_on_the_atom_order(namers, smiles):
    assert len({namers["best-effort"].name_tiered(o)["name"] for o in _random_orders(smiles, 6)}) == 1
