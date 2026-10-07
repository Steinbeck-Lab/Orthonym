"""The spelling check: no substitutive, multiplicative or ring assembly name keeps
the pin_verified label where the PIN is a linear phane name.

 (the Blue Book) "Phane nomenclature is used to generate preferred IUPAC names
for ring assemblies and linear acyclic/cyclic compounds that include a minimum of seven nodes
including at least four rings or ring systems, two of which must be terminal, even though the
compounds could also be named by substitutive or multiplicative nomenclature.";:23959
'2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane (PIN, a phane name) 1,1'-oxybis(4-
phenoxybenzene) (a multiplicative name) 1-phenoxy-4-(4-phenoxyphenoxy)benzene (a substitutive
name)'; (:24088) "Phane names are preferred IUPAC names rather than ring assembly
names when seven or more rings or ring systems are present."

The check is registered in the spelling-check registry (``validation/pin_spelling.py``) and runs
at its call site, ``Orthonym._tier_row_and_pin_form``: a failure lowers the label and never the
name; the row records ``spelling_failures``. OPSIN 2.9.0 reads no phane name
(``test_opsin_reads_no_linear_phane_name``), so no phane name can be certified: the default tier
declines the class with the reason, the wider tiers keep the round-tripped name with
the label systematic_verified (a correct name that is not the PIN).
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.validation import pin_spelling
from orthonym.validation.opsin_roundtrip import opsin_parse
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.rt_assert import assert_full_rt

#: the labels are the production ones only with the OPSIN validity gate on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate

RULE = "P-52.2.5.1"

BOOK_23959 = "c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"
SEPTIPHENYL = "c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccccc7)cc6)cc5)cc4)cc3)cc2)cc1"
VENADAPARIB = "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O"
NILOTINIB = "CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1"
#: four ring systems on a 7-node chain, but pyrimidine-2,4-diamine cites two amines (the NH2 and the
#: N4 link) and the phane one: keeps the substitutive PIN
DIAMINE = ("Nc1nc(Nc2ccc(OCc3ccc(-c4ccccc4)cc3)cc2)ccn1",
           "N4-{4-[([1,1'-biphenyl]-4-yl)methoxy]phenyl}pyrimidine-2,4-diamine")


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
            if f.rule == RULE]


def _row_rules(row):
    return [f["rule"] for f in row.get("spelling_failures") or []]


# ------------------------------------------------------------------ the check alone (no engine)
def test_the_check_is_registered_under_its_rule_id():
    # spec section 2: a lane adds its check module to the registry's built-in check modules
    assert "orthonym.validation.spelling.checks_phane" in pin_spelling.BUILTIN_CHECK_MODULES
    assert RULE in pin_spelling.registered_rules()


@pytest.mark.parametrize("name", [
    "1-phenoxy-4-(4-phenoxyphenoxy)benzene",     #:23959 "(a substitutive name)"
    "1,1'-oxybis(4-phenoxybenzene)",             #:23959 "(a multiplicative name)"
])
def test_the_books_other_names_of_a_linear_phane_fail(name):
    assert _rules(BOOK_23959, name) == [RULE]


def test_the_phane_pin_itself_passes():
    assert _rules(BOOK_23959, "2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane") == []


def test_a_ring_assembly_name_of_seven_rings_fails():
    # (:24088): the phane name '1,7(1),2,3,4,5,6(1,4)-heptabenzenaheptaphane'
    name = "1,1':4',1'':4'',1''':4''',1'''':4'''',1''''':4''''',1''''''-septiphenyl"
    assert _rules(SEPTIPHENYL, name) == [RULE]


def test_a_substitutive_name_of_venadaparib_fails():
    # the derived PIN is '3^4-fluoro-7-aza-1(1)-phthalazina-5(1,3)-azetidina-3(1,3)-benzena-
    # 8(1)-cyclopropanaoctaphane-1^4,4(1^3H)-dione'
    name = ("4-[(3-{3-[(cyclopropylamino)methyl]azetidine-1-carbonyl}-4-fluorophenyl)methyl]"
            "phthalazin-1(2H)-one")
    assert _rules(VENADAPARIB, name) == [RULE]


@pytest.mark.parametrize("smiles,name", [
    #:23925 a multiplicative PIN (3 rings, 5 nodes)
    ("c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1", "1,1'-[1,4-phenylenebis(oxy)]dibenzene"),
    # nilotinib: the amide links two chain rings, so the benzamide is the parent (reading R2)
    (NILOTINIB, "4-methyl-N-[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]-3-"
                "{[4-(pyridin-3-yl)pyrimidin-2-yl]amino}benzamide"),
    #:3547 four saturated rings: the book's substitutive PIN
    ("C(OCC1CCCCC1)C1CCC(CC2CCC(CC3CCCCC3)CC2)CC1",
     "1-[(cyclohexylmethoxy)methyl]-4-{[4-(cyclohexylmethyl)cyclohexyl]methyl}cyclohexane"),
    # the substitutive parent cites more of the principal characteristic group
    DIAMINE,
])
def test_names_of_structures_whose_pin_is_no_phane_pass(smiles, name):
    assert _rules(smiles, name) == []


#::23959's heptaphane with one group on a terminal ring
ON_A_CHAIN_RING = "{}c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"


@pytest.mark.parametrize("group,name", [
    ("SC(=O)", "4-[4-(4-phenoxyphenoxy)phenoxy]benzene-1-carbothioic S-acid"),
    ("OOC(=O)", "4-[4-(4-phenoxyphenoxy)phenoxy]benzene-1-carboperoxoic acid"),
])
def test_names_of_an_acid_analogue_on_a_chain_ring_fail(group, name):
    # Table 4.1 7a (:18172): the chalcogen and peroxy analogues of a carboxylic acid are acids,
    # which the phane cites on its chain ring as it cites the carboxylic acid (a tie,
    assert _rules(ON_A_CHAIN_RING.format(group), name) == [RULE]


#: final review a performance pass (finding F2; the pendant chalcogen ketones of F1): another parent cites
#: more of the principal characteristic group than the phane, and the qualifying
#: chain stays whole in its substituent, which the PIN may cite as a linear phane prefix
#: (:19325 'trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane (PIN)';
#:,:16148). No printed row decides it for such a parent: the name is not certified
#: (controller ruling of 2026-10-05: fail closed). Each name was pin_verified before the ruling.
PREFIX_LOWERED = [
    (ON_A_CHAIN_RING.format("OC(=O)C"), "{4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}acetic acid"),
    (ON_A_CHAIN_RING.format("COC(=O)N"), "methyl {4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}carbamate"),
    (ON_A_CHAIN_RING.format("NC(=O)N"), "{4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}urea"),
    (ON_A_CHAIN_RING.format("CC(=O)"), "1-{4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}ethan-1-one"),
    (ON_A_CHAIN_RING.format("CC(=S)"), "1-{4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}ethane-1-thione"),
    (ON_A_CHAIN_RING.format("CC(=[Se])"),
     "1-{4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}ethane-1-selone"),
    ("Nc1nc(Nc2ccc(Oc3ccc(Oc4ccc(Oc5ccccc5)cc4)cc3)cc2)ccn1",
     "N4-{4-[4-(4-phenoxyphenoxy)phenoxy]phenyl}pyrimidine-2,4-diamine"),
    ("OC(=O)CC(C(=O)O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)O)cc4)cc3)cc2)cc1",
     "2-(4-{4-[4-(4-carboxyphenoxy)phenoxy]phenoxy}phenyl)butanedioic acid"),
]
PREFIX_LOWERED_IDS = ["acetic_acid", "carbamate", "urea", "ethanone", "ethanethione",
                      "ethaneselone", "pyrimidine_diamine", "butanedioic_acid"]


@pytest.mark.parametrize("smiles,name", PREFIX_LOWERED, ids=PREFIX_LOWERED_IDS)
def test_names_whose_pin_may_cite_a_linear_phane_prefix_fail(smiles, name):
    assert _rules(smiles, name) == [RULE]


# ------------------------------------------------------------------ the engine's labels
def _row(smiles, tier):
    with jvm_slots(1, purpose="l5-phane"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: Blue Book linear phane PIN structures (and the gate's seven-ring assembly) whose substitutive,
#: multiplicative or ring assembly name the base labelled pin_verified (:20577's from the ring
#: assembly and ether route;:23917's name is 'di', verbatim at that line -- a ring assembly is a
#: SIMPLE component despite its brackets (f),:7104) even at three-or-more rings, and
#: 'bis' there fails the registry's own check;
#: the reason is asserted for every row but the three of RING_ASSEMBLY_ROWS)
LOWERED = [
    (BOOK_23959, "1-phenoxy-4-(4-phenoxyphenoxy)benzene"),                          #:23959
    ("c1ccc(Oc2cccc(Oc3cccc(Oc4ccccc4)c3)c2)cc1",
     "1-phenoxy-3-(3-phenoxyphenoxy)benzene"),                                        #:23282
    ("c1ccc(-c2cccc(-c3cccc(-c4cncc(-c5cccc(-c6cccc(-c7ccccc7)c6)c5)c4)c3)c2)cc1",
     "3,5-di([1,1':3',1''-terphenyl]-3-yl)pyridine"),                                 #:23917
    ("c1ccc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)cc1",
     "4-({4-[(4-benzylphenyl)methyl]phenyl}methyl)pyridine"),                         #:20029
    ("c1ccc(-c2ccc(-c3ccc(Oc4cccc(-c5cccc(-c6ccccn6)n5)n4)cc3)cc2)cc1",
     "2-([2,2'-bipyridin]-6-yl)-6-[([1,1':4',1''-terphenyl]-4-yl)oxy]pyridine"),     #:20577
    (SEPTIPHENYL, "1,1':4',1'':4'',1''':4''',1'''':4'''',1''''':4''''',1''''''-septiphenyl"),
]
LOWERED_IDS = ["23959", "23282", "23917", "20029", "20577", "septiphenyl"]

RING_ASSEMBLY_RULE = "P-52.2.7.2"
#: the rows whose name holds a ring assembly of three or more benzene rings with primed locants
RING_ASSEMBLY_ROWS = {LOWERED[2][0], LOWERED[4][0], SEPTIPHENYL}


def _decline_reason_ok(smiles, row):
    rules = _row_rules(row)
    if smiles in RING_ASSEMBLY_ROWS:
        # (:24078) lowers the assembly before the (:23826) check runs
        return RULE in rules or RING_ASSEMBLY_RULE in rules
    return RULE in rules


@pytest.mark.parametrize("smiles,name", LOWERED, ids=LOWERED_IDS)
def test_the_default_tier_declines_a_linear_phane_structure(smiles, name):
    row = _row(smiles, "pin")
    assert (row["tier"], row["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), row
    assert row["name"] != name
    assert _decline_reason_ok(smiles, row), row         # the decline carries its reason


@pytest.mark.parametrize("smiles,name", LOWERED, ids=LOWERED_IDS)
@pytest.mark.parametrize("tier", ["valid", "best-effort"])
def test_the_wider_tiers_keep_the_name_as_a_verified_non_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert _decline_reason_ok(smiles, row), row


def test_the_seven_ring_assembly_gold_row_is_judged_through_the_default_tier_decline():
    # (:24088): the gate's protect row W2E-P2IP-P2 expects the ring assembly name, a
    # correct name that is not the PIN; the gate passes it only when the default tier declines it
    # with NO_VERIFIED_PIN and best-effort gives the name exactly
    path = Path(__file__).resolve().parents[3] / "benchmarks/pin_oracle/default_tier_non_pin_rows.json"
    rows = json.loads(path.read_text())["rows"]
    listed = {(r["pack"], r["def_id"], r["smiles"], r["expected_pin"]) for r in rows}
    assert ("characteristic_groups", "W2E-P2IP-P2", SEPTIPHENYL, LOWERED[-1][1]) in listed


@pytest.mark.parametrize("smiles,name", LOWERED, ids=LOWERED_IDS)
def test_the_kept_names_read_back_to_the_full_inchikey(smiles, name):
    # the name the wider tiers keep is RT-exact: a fresh OPSIN call reads it back to the input
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,name", [
    #:23925 the multiplicative PIN (3 rings): the check is silent
    ("c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1", "1,1'-[1,4-phenylenebis(oxy)]dibenzene"),
    # a parent (natural-product route): its label is the controller's ruling; the call
    # site does not read the routes although four pyrroles sit on a 7-node chain
    ("C1=CC(=CC2=NC(=CC3=NC(=Cc4ccc[nH]4)C=C3)C=C2)N=C1", "21H-biline"),
    # four ring systems on a 7-node chain whose substitutive parent cites more amines
    # and holds a ring of the chain: what is left (three ring systems) is no linear phane
    DIAMINE,
])
def test_names_the_check_does_not_lower_keep_pin_verified(smiles, name):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert not row.get("spelling_failures"), row


@pytest.mark.parametrize("smiles,name", PREFIX_LOWERED, ids=PREFIX_LOWERED_IDS)
def test_the_default_tier_declines_a_name_whose_pin_may_cite_a_phane_prefix(smiles, name):
    row = _row(smiles, "pin")
    assert (row["tier"], row["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), row
    assert RULE in _row_rules(row), row


@pytest.mark.parametrize("smiles,name", PREFIX_LOWERED, ids=PREFIX_LOWERED_IDS)
def test_best_effort_keeps_the_name_whose_pin_may_cite_a_phane_prefix(smiles, name):
    # the label is lowered, never the name: best-effort keeps it as a verified non-PIN, and a
    # fresh OPSIN call reads it back to the input's full InChIKey
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert RULE in _row_rules(row), row
    assert_full_rt(name, smiles)


# ------------------------------------------------------------------ why no phane name is certified
#: phane PINs the Blue Book prints (line), and the two derived drug PINs of the lane plan
PHANE_NAMES = [
    "2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane",                               #:23959
    "2,4,6-trithia-1,7(1),3,5(1,4)-tetrabenzenaheptaphane",                              #:15196
    "4(3,5)-pyridina-1,7(1),2,3,5,6(1,3)-hexabenzenaheptaphane",                         #:23917
    "1(4)-pyridina-3,5(1,4),7(1)-tribenzenaheptaphane",                                  #:20029
    "2,5,7,10,12,15-hexathia-1,16(3),6,11(3,4)-tetrafuranahexadecaphane",                #:23965
    "1,10(1),3,5,7(1,4)-pentabenzenadecaphane",                                          #:20405
    "1,4(1,4)-dibenzenacyclohexaphane",                                                  #:14951
    "3^4-fluoro-7-aza-1(1)-phthalazina-5(1,3)-azetidina-3(1,3)-benzena-8(1)-"
    "cyclopropanaoctaphane-1^4,4(1^3H)-dione",                                           # venadaparib
    "3^4-fluoro-1(1)-phthalazina-5(1,4)-piperazina-3(1,3)-benzena-7(1)-"
    "cyclopropanaheptaphane-1^4,4,6(1^3H)-trione",                                       # olaparib
]


@pytest.mark.parametrize("name", PHANE_NAMES)
def test_opsin_reads_no_linear_phane_name(name):
    assert opsin_parse(name) is None


def test_opsin_reads_the_books_substitutive_name_of_the_same_compound():
    # known positive for the test above: the same OPSIN call reads:23959's substitutive name
    smiles = opsin_parse("1-phenoxy-4-(4-phenoxyphenoxy)benzene")
    assert smiles is not None
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(BOOK_23959))
