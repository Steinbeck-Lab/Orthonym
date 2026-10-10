"""Leads L2 (C), item N9d: the cycloalkane component of an assembly with a suffix is enclosed.

``### **** Ring assemblies with a single bond junction`` (the Blue Book),:15564 "(1) by
placing the prefix 'bi' (see before the name of the corresponding parent hydride enclosed in
parentheses, if necessary. Parentheses are used to avoid confusion with von Baeyer names"; printed:
:49838 "(1S,1's,2S,4S,4'R)-4,4'-dimethyl[1,1'-bi(cyclohexan)]-2-ol (PIN)" and:16122
"[1,1'-bi(cyclohexan)]-4-yl-4'-ylidene (preferred prefix)". The final 'e' of the parent hydride is
elided before a suffix that begins with a vowel (a); the book prints 'cyclohexan' inside the
enclosure before '-2-ol') and stays before a consonant ('cyclohexane-1,4-diol', '[2,2'-binaphthalene]-
1,1'-diol':27224).

The suffix branch of ``name_ring_assembly`` and the mixed builder passed the bare ring name to the
multiplier, so a cycloalkane assembly with a suffix read '[1,1'-bicyclohexane]-4-ol' ('bicyclo' is the von
Baeyer prefix), while the parent branch ('1,1'-bi(cyclohexane)') and the prefix branch already enclosed it.
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

RULE = "P-28.2.1"

#: (id, SMILES, the name the producer gave before, the PIN); 'None' where the name did not change
ROWS = [
    ("ol", "OC1CCC(CC1)C1CCCCC1", "[1,1'-bicyclohexane]-4-ol", "[1,1'-bi(cyclohexan)]-4-ol"),
    ("hexyl-carbonitrile", "CCCCCCC1CCC(CC1)C2CCCC(C2)C#N",
     "4'-hexyl[1,1'-bicyclohexane]-3-carbonitrile", "4'-hexyl[1,1'-bi(cyclohexane)]-3-carbonitrile"),
    ("amine", "NC1CCC(CC1)C1CCCCC1", "[1,1'-bicyclohexane]-4-amine", "[1,1'-bi(cyclohexan)]-4-amine"),
    ("carboxylic-acid", "OC(=O)C1CCC(CC1)C1CCCCC1", "[1,1'-bicyclohexane]-4-carboxylic acid",
     "[1,1'-bi(cyclohexane)]-4-carboxylic acid"),
    ("diol", "OC1CCC(CC1)C1CCC(O)CC1", "[1,1'-bicyclohexane]-4,4'-diol",
     "[1,1'-bi(cyclohexane)]-4,4'-diol"),
    ("chloro-ol", "OC1CCC(CC1)C1CCC(Cl)CC1", "4'-chloro[1,1'-bicyclohexane]-4-ol",
     "4'-chloro[1,1'-bi(cyclohexan)]-4-ol"),
    ("hydroxy-acid", "OC1CCC(CC1)C1CCC(C(=O)O)CC1", "4'-hydroxy[1,1'-bicyclohexane]-4-carboxylic acid",
     "4'-hydroxy[1,1'-bi(cyclohexane)]-4-carboxylic acid"),
    ("cyclopropane-ol", "OC1CC1C1CC1", "[1,1'-bicyclopropane]-2-ol", "[1,1'-bi(cyclopropan)]-2-ol"),
]
IDS = [r[0] for r in ROWS]

#: a mancude component takes the same elision (a)): the book prints '[2,2'-bipyridin]-5-yl'
#: (:16130) for the prefix; before a consonant the 'e' stays
MANCUDE = [
    ("Nc1ccnc(c1)-c1ccccn1", "[2,2'-bipyridin]-4-amine"),
    ("Oc1ccnc(c1)-c1ccccn1", "[2,2'-bipyridin]-4-ol"),
    ("OC(=O)c1ccnc(c1)-c1ccccn1", "[2,2'-bipyridine]-4-carboxylic acid"),
    ("Nc1ccc(cc1)-c1ccccc1", "[1,1'-biphenyl]-4-amine"),
]


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
            if f.rule == RULE]


# ------------------------------------------------------------------ the check alone
@pytest.mark.parametrize("_id,smiles,old,pin", ROWS, ids=IDS)
def test_bi_before_cyclo_without_parentheses_fails(_id, smiles, old, pin):
    assert _rules(smiles, old) == [RULE]
    assert _rules(smiles, pin) == []


@pytest.mark.parametrize("name", [
    "bicyclo[2.2.2]octane",
    "2,2'-bi(bicyclo[2.2.2]octane)",                 #:24076 (PIN)
    "1,1'-bi(cyclohexane)",
    "1,1':4',1''-tercyclohexane",                    # 'ter' + 'cyclo' is no von Baeyer prefix
    "[1,1'-bi(cyclohexan)]-4-yl-4'-ylidene",         #:16122
    "(1S,1's,2S,4S,4'R)-4,4'-dimethyl[1,1'-bi(cyclohexan)]-2-ol",   #:49838 (PIN)
    "7-oxabicyclo[2.2.1]heptane",
])
def test_the_books_names_pass(name):
    assert _rules("C", name) == []


# ------------------------------------------------------------------ the reads back
@pytest.mark.parametrize("_id,smiles,old,pin", ROWS, ids=IDS)
def test_the_pin_reads_back_to_the_full_inchikey(_id, smiles, old, pin):
    assert_full_rt(pin, smiles)


@pytest.mark.parametrize("smiles,pin", MANCUDE)
def test_the_mancude_name_reads_back_to_the_full_inchikey(smiles, pin):
    assert_full_rt(pin, smiles)


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


@pytest.mark.parametrize("_id,smiles,old,pin", ROWS, ids=IDS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_engine_encloses_the_cycloalkane_component(namers, _id, smiles, old, pin, tier):
    row = namers[tier].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.parametrize("smiles,pin", MANCUDE)
def test_the_engine_elides_the_e_of_a_mancude_component_before_a_vowel_suffix(namers, smiles, pin):
    row = namers["pin"].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.parametrize("smiles", [ROWS[0][1], ROWS[1][1]])
def test_the_name_does_not_depend_on_the_atom_order(namers, smiles):
    mol = Chem.MolFromSmiles(smiles)
    orders = {smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(12)}
    assert len({namers["pin"].name_tiered(o)["name"] for o in orders}) == 1
