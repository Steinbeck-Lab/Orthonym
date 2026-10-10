"""Leads L2 (A): a ring assembly that a non-assembly producer takes apart is not a PIN (item 32 part A
and the label half of N9b).

``### **** DEFINITIONS`` (the Blue Book),:15542 "Two or more cyclic systems (single rings
or fused systems, alicyclic von Baeyer systems, spiro systems, phane systems, fullerenes) that are
directly joined to each other by single or double bonds are called 'ring assemblies' when the number
of such direct ring junctions is one less than the number of cyclic systems involved.";
``### **** Preferred IUPAC names and numbering for ring assemblies`` (:24070), ``****``
(:24072) "Preferred IUPAC names for assemblies of two or more identical cyclic systems joined by a
single bond are formed using the names of parent hydrides rather than the names of substituent
groups", example:24076 "2,2'-bi(bicyclo[2.2.2]octane) (PIN)"; ``## **** UNSATURATION IN RING
ASSEMBLIES COMPOSED OF MONOCYCLIC MANCUDE AND SATURATED RINGS`` (:24151),:24153 "the use of hydro
prefixes is preferred, except in the case of a two ring assembly consisting of one benzene ring and a
cyclohexane ring",:24159 "1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN) 2-(piperidin-2-yl)pyridine";
``### **** Ring assemblies composed of monocyclic components`` (:17077),:17083
"2,3-dihydro-1,1'-biphenyl (PIN) (numbering shown) (cyclohexa-1,3-dien-1-yl)benzene";:16884
"1',2',3',4'-tetrahydro-1,2'-binaphthalene (PIN)".

Seven producers build the substitutive names below (ring_assemblies.py cannot take them: its
``detect_ring_assembly`` needs every ring system joined), and all of them pass the one call site of
the spelling checks (``namer.py`` ``check_pin_spelling``), so the check ````
(``validation/spelling/checks_assembly.py``) decides the class. The round trip cannot: OPSIN reads
both spellings of each molecule back to the same full InChIKey, which the last test shows.
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.ring_assembly_screen import _assemblies
from orthonym.validation import pin_spelling
from orthonym.validation.pin_spelling import check_pin_spelling
from orthonym.validation.spelling import checks_assembly
from tests.support.hydro_wiring import WIRED
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

RULE = "P-28.1"

#: (id, SMILES, the name the producers give, the PIN, where the book decides it)
SPLIT = [
    ("bicyclooctane", "C1(C2CC3CCC2CC3)CC2CCC1CC2",
     "2-(bicyclo[2.2.2]octan-2-yl)bicyclo[2.2.2]octane", "2,2'-bi(bicyclo[2.2.2]octane)",
     "P-52.2.7.1 :24076"),
    ("bicycloheptane", "C1CC2CCC1C2C1C2CCC1CC2",
     "7-(bicyclo[2.2.1]heptan-7-yl)bicyclo[2.2.1]heptane", "7,7'-bi(bicyclo[2.2.1]heptane)",
     "P-52.2.7.1 :24076, same class as the printed octane"),
    ("benzyloxy-bipyridine", "c1ccc(COc2ccc(-c3ccccn3)nc2)cc1",
     "5-(benzyloxy)-2-(pyridin-2-yl)pyridine", "5-(benzyloxy)-2,2'-bipyridine",
     "P-52.2.7.1 :24076 '2,2'-bipyridine (PIN)'"),
    ("piperidinyl-pyridine", "c1ccc(C2CCCCN2)nc1",
     "2-(piperidin-2-yl)pyridine", "1,2,3,4,5,6-hexahydro-2,2'-bipyridine", "P-54.3 :24159"),
    ("piperidin-1-yl-pyridine", "c1ccc(N2CCCCC2)nc1",
     "2-(piperidin-1-yl)pyridine", "3,4,5,6-tetrahydro-2H-1,2'-bipyridine",
     "P-54.3 :24153 with P-28.2.3 :15593 (ring-nitrogen junction, '2H-1,2'-bipyridine (PIN)' :15634)"),
    ("cyclohexenylbenzene", "c1ccccc1C1=CCCCC1",
     "(cyclohex-1-en-1-yl)benzene", "2,3,4,5-tetrahydro-1,1'-biphenyl", "P-31.2.3.3.5.1 :17083"),
    ("tetrahydronaphthyl-naphthalene", "c1ccc2c(c1)CCC(c1cccc3ccccc13)C2",
     "1-(1,2,3,4-tetrahydronaphthalen-2-yl)naphthalene", "1',2',3',4'-tetrahydro-1,2'-binaphthalene",
     "P-31.2.2 :16884"),
    ("assembly-in-a-prefix", "CC(=O)OCC(O)C#Cc1ccc(-c2cccs2)s1",
     "2-hydroxy-4-[5-(thiophen-2-yl)thiophen-2-yl]but-3-yn-1-yl acetate",
     "4-([2,2'-bithiophen]-5-yl)-2-hydroxybut-3-yn-1-yl acetate",
     "P-29.3.5 :16118 '[1,1'-biphenyl]-4-yl (preferred prefix)' (the same construction)"),
]
IDS = [row[0] for row in SPLIT]

#: the engine builds the PIN for the two von Baeyer pairs (test_leads_l2_von_baeyer_assembly.py), so only the
#: other rows still reach the engine with a split name; the check is read on all of them above
#: leads R2A: the hydro-modified ring assembly producer (rules/ring_assemblies.py ``detect_hydro_ring_assembly``,
#: called from namer.py) builds the PIN of these four rows, so the engine no longer ships the split name for them
#: (their engine tests are in test_leads_r2a_hydro_ring_assembly.py); the check itself is still read on them above
HYDRO_NAMED = {"piperidinyl-pyridine", "piperidin-1-yl-pyridine", "cyclohexenylbenzene",
               "tetrahydronaphthyl-naphthalene"}
ENGINE_SPLIT = [row for row in SPLIT
                if not row[0].startswith("bicyclo") and not (WIRED and row[0] in HYDRO_NAMED)]
ENGINE_IDS = [row[0] for row in ENGINE_SPLIT]

#: names of the same structures that cite their assembly, or that are no assembly at all
CONTROLS = [
    ("COc1ccc(-c2ccccn2)nc1", "5-methoxy-2,2'-bipyridine"),
    ("OCCc1ccc(-c2ccccn2)nc1", "2-([2,2'-bipyridin]-5-yl)ethan-1-ol"),
    #:24153 "except in the case of a two ring assembly consisting of one benzene ring and a
    # cyclohexane ring" (:24157 'cyclohexylbenzene (PIN)')
    ("C1CCC(CC1)c1ccccc1", "cyclohexylbenzene"),
    # (a): the heterocycle is the senior ring; benzene and pyridine are no assembly
    ("Cc1ccc(cc1Oc1ccccn1)-c1ccccc1", "2-[(4-methyl[1,1'-biphenyl]-3-yl)oxy]pyridine"),
    # a book PIN whose assembly is a ring-nitrogen junction of two different-element ring members
    ("n1ccccc1-c1ccccc1", "2-phenylpyridine"),
    # a multiplied assembly prefix: the check counts texts, never the structure's repeats
    ("c1ccc(cc1)-c1cccc(c1)-c1cncc(c1)-c1cccc(c1)-c1ccccc1", "3,5-di([1,1'-biphenyl]-3-yl)pyridine"),
]


def _failures(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
            if f.rule == RULE]


# ------------------------------------------------------------------ the check alone (no engine)
def test_the_check_is_registered_under_the_rule_it_cites():
    assert "orthonym.validation.spelling.checks_assembly" in pin_spelling.BUILTIN_CHECK_MODULES
    assert RULE in pin_spelling.registered_rules()


@pytest.mark.parametrize("_id,smiles,split,pin,_why", SPLIT, ids=IDS)
def test_the_name_that_takes_the_assembly_apart_fails(_id, smiles, split, pin, _why):
    assert _failures(smiles, split) == [RULE]


@pytest.mark.parametrize("_id,smiles,split,pin,_why", SPLIT, ids=IDS)
def test_the_pin_that_cites_its_assembly_passes(_id, smiles, split, pin, _why):
    assert _failures(smiles, pin) == []


@pytest.mark.parametrize("smiles,name", CONTROLS)
def test_a_name_that_is_not_an_assembly_split_passes(smiles, name):
    assert _failures(smiles, name) == []


@pytest.mark.parametrize("_id,smiles,split,pin,_why", SPLIT, ids=IDS)
def test_the_pin_reads_back_to_the_full_inchikey(_id, smiles, split, pin, _why):
    # independent of the engine: a fresh OPSIN call, full standard InChIKey. OPSIN reads the split
    # name back too (its row below): only the spelling check can tell the two apart.
    assert_full_rt(pin, smiles)
    assert_full_rt(split, smiles)


@pytest.mark.parametrize("name,count", [
    ("2,2'-bipyridine", 1),
    ("[1,1'-biphenyl]-4-yl", 1),
    ("1,1':4',1''-terphenyl", 1),
    ("1,1':2',1''-ter(cyclopropane)", 1),
    ("1^1,2^1:2^2,3^1-tercyclopropane", 1),               # composite locants (:15655), never primes
    ("4H-2,4'-bi-1,4-oxazine", 1),                        # the hyphen form of a locant-initial component
    ("1,1'-bistibinane", 1),                              #:39103: 'bi' + 'stibinane' is no 'bis'
    ("4,4'-bis(4-chlorophenyl)benzene", 0),               # 'bis' before an enclosing mark multiplies
    ("2,2'-di-tert-butylbenzene", 0),                     # 'tert-' is no 'ter'
    ("2,2'-terephthaloyldibenzene", 0),
    ("[1,1'-bi(cyclohexan)]-4-yl-4'-ylidene", 1),         #:16122
    ("1λ4,1'λ4-bithiophene", 1),                          # a lambda convention inside the junction locants
    ("3a,3'a-biindene", 1),                               #:15609 (PIN)
    ("(2-(pyridin-2-yl)pyridine)", 0),
])
def test_the_assembly_texts_are_counted_lexically(name, count):
    assert checks_assembly.count_assembly_texts(name) == count


def test_a_group_of_seven_ring_systems_is_a_phane_and_is_not_decided_here():
    # (:24088): phane names are the PINs there; checks_phane decides, this one abstains
    septi = "c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccccc7)cc6)cc5)cc4)cc3)cc2)cc1"
    assert _failures(septi, "1-phenyl-4-(4-phenylphenyl)benzene") == []


def test_a_linear_phane_name_is_not_decided_here():
    book_23959 = "c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"
    assert _failures(book_23959, "2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane") == []


def test_the_check_reads_the_structure_it_is_given_and_abstains_without_one():
    assert checks_assembly.ring_assembly_named_as_such_check(None, "x") is None


def test_the_structure_reader_agrees_with_the_screen_the_producers_consult():
    # the check keeps its own reader (the checks import no rules code); the producers' screen
    # (rules/ring_assembly_screen) must see the same groups, or the two would drift
    smiles = [row[1] for row in SPLIT] + [c[0] for c in CONTROLS] + [
        "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "C1CC1C1CC1C1CC1", "O=C1C=CC(=O)C=C1", "c1ccc2ccccc2c1-c1cccc2ccccc12",
        "C1CCC(CC1)C1CCCCC1", "CC(=O)Nc1ccc(-c2ccccc2)cc1", "C1CCCCN1C1CCCCN1", "C1CCCCC1C1=CC=CC=C1",
    ]
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        ours = sorted(sorted(sorted(s) for s in g) for g in checks_assembly.ring_assembly_groups(mol))
        systems, _owner, groups = _assemblies(mol)
        theirs = sorted(sorted(sorted(systems[k]) for k in g) for g in groups)
        assert ours == theirs, smi


# ------------------------------------------------------------------ the engine's labels
pytestmark = pytest.mark.opsin_gate           # the labels are the production ones only with the gate on


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


@pytest.mark.parametrize("_id,smiles,split,pin,_why", ENGINE_SPLIT, ids=ENGINE_IDS)
def test_the_default_tier_declines_the_split_name_with_its_reason(namers, _id, smiles, split, pin, _why):
    row = namers["pin"].name_tiered(smiles)
    assert (row["tier"], row["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), row
    assert RULE in [f["rule"] for f in row.get("spelling_failures") or []], row


@pytest.mark.parametrize("_id,smiles,split,pin,_why", ENGINE_SPLIT, ids=ENGINE_IDS)
def test_best_effort_keeps_the_round_tripped_name_as_a_verified_non_pin(namers, _id, smiles, split, pin, _why):
    row = namers["best-effort"].name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (split, "systematic_verified", False), row
    assert RULE in [f["rule"] for f in row.get("spelling_failures") or []], row


@pytest.mark.parametrize("smiles,name", CONTROLS[:4])
def test_the_names_the_check_does_not_lower_keep_pin_verified(namers, smiles, name):
    row = namers["pin"].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert not row.get("spelling_failures"), row


# ------------------------------------------------------------------ the gold row the book contradicts
def test_the_gold_row_cyclohexenylbenzene_asserts_the_books_pin():
    # packs/rings_numbering.json SUBST-01a asserted '(cyclohex-1-en-1-yl)benzene' and cited
    #, an id the book does not have. (:17077),:17079 "In biphenyl and
    # polyphenyl assemblies, one benzene ring must remain in the assembly... Furthermore, when a
    # modified ring assembly of two rings consists of a benzene ring and a cyclohexane ring
    # substitutive nomenclature is preferred (see." and:17083 print
    # '2,3-dihydro-1,1'-biphenyl (PIN)... (cyclohexa-1,3-dien-1-yl)benzene': the PIN of a benzene
    # ring joined to a cyclohexene ring is the hydro assembly name; only the saturated pair is
    # substitutive (:24153,:24157).
    root = Path(__file__).resolve().parents[3] / "benchmarks/pin_oracle"
    rows = json.loads((root / "packs/rings_numbering.json").read_text())["rows"]
    row = next(r for r in rows if r["def_id"] == "SUBST-01a" and r["smiles"] == "c1ccccc1C1=CCCCC1")
    assert row["expected_pin"] == "2,3,4,5-tetrahydro-1,1'-biphenyl"
    assert row["bluebook_ref"] == "P-31.2.3.3.5.1"
    assert _failures(row["smiles"], row["expected_pin"]) == []
    assert _failures(row["smiles"], "(cyclohex-1-en-1-yl)benzene") == [RULE]
    assert_full_rt(row["expected_pin"], row["smiles"])
    floor = json.loads((root / "baseline_passing_rows.json").read_text())["passing_target_rows"]
    # leads R2A built the hydro-assembly producer, so the row is on the floor again (1662)
    assert ["SUBST-01a", "c1ccccc1C1=CCCCC1"] in floor
