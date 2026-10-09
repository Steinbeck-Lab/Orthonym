"""A name in PIN form that breaks a spelling rule of the PIN is not labelled pin_verified.

The label site is ``Orthonym._tier_row_and_pin_form`` (the one function that derives the
``pin_verified`` label, at every tier and on every public entry point). There the spelling checks
of ``validation/pin_spelling.py`` run on a name about to be labelled pin_verified; a failure lowers
the label and never the molecule: the row records ``spelling_failures``, the label is
``systematic_verified`` (a correct name that is not the PIN), the default tier declines the name
with NO_VERIFIED_PIN and keeps the failures as the reason, and the wider tiers keep the name.
Each name is read back by a fresh OPSIN call to the input's full InChIKey: the round trip proves
the molecule, and passes for every name below although none is the PIN.

* the roadmap N8 (b) rows: (c), (f) (the Blue Book,:3256,:3301), (:21604,
  :21624).
* '2-(pentyloxycarbonyl)benzoic acid': (:7232) with (:27667), the PIN
  spelling is '2-[(pentyloxy)carbonyl]benzoic acid'.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.namer import _default_tier_emits
from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.default_tier import (
    assert_best_effort_gives,
    assert_default_tier_declines,
    default_tier_row,
    strict_path_row,
)

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

LOWERED = [
    ("CCc1ccc(OC2CCCCC2C#N)cc1", "6-(4-ethylphenoxy)cyclohexane-1-carbonitrile", "P-14.4"),
    ("OC(COC1CCCc2ccccc21)CN1CCN(c2ccccc2)CC1",
     "1-(1-phenylpiperazin-4-yl)-3-[(1,2,3,4-tetrahydronaphthalen-1-yl)oxy]propan-2-ol", "P-14.4"),
    ("COC(=O)C(C)Cc1c(C)nn(-c2ccccc2Cl)c1C",
     "methyl 2-{[1-(2-chlorophenyl)-3,5-dimethyl-1H-pyrazol-4-yl]methyl}propanoate", "P-45.2.1"),
    ("CCCCCOC(=O)c1ccccc1C(=O)O", "2-(pentyloxycarbonyl)benzoic acid", "P-16.5.1.1"),
]


#: the row is lowered before the spelling checks run: the chain-parent check of the
#: candidate pool (``perception.chains.chain_parent_prefix_seniority``;:21604, with
#: the ring prefix counted as in example (4):21624) records the name as not the PIN, so no
#: pin_verified label reaches ``check_pin_spelling`` and the decline row carries no spelling
#: failure; both checks fail the name (``test_the_p45_2_1_row_is_lowered_by_both_checks``)
P4521_ROW = LOWERED[2]
LOWERED_BY_SPELLING = [row for row in LOWERED if row is not P4521_ROW]


def _rules(row):
    return [f["rule"] for f in row.get("spelling_failures") or []]


def test_the_p45_2_1_row_is_lowered_by_both_checks(monkeypatch):
    """The PIN is 'methyl 3-[1-(2-chlorophenyl)-3,5-dimethyl-1H-pyrazol-4-yl]-2-methylpropanoate'
    (two prefixes on its chain, one on the name's). The lexical check fails the name on its own;
    with that check switched off the chain-parent check still lowers the label at every tier and
    keeps the name (known positive: before the chain-parent check, the same name was
    pin_verified with the spelling check off)."""
    from tests.support.spelling_checks import disable_spelling_rule
    smiles, name, rule = P4521_ROW
    assert rule in [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name)]
    disable_spelling_rule(monkeypatch, rule)
    assert_default_tier_declines(smiles)
    strict = strict_path_row(smiles)
    assert (strict["name"], strict["tier"], strict["is_pin"]) == (name, "systematic_verified", False), strict
    best = assert_best_effort_gives(smiles, name)
    assert (best["tier"], best["is_pin"]) == ("systematic_verified", False), best


@pytest.mark.parametrize("smiles,name,rule", LOWERED_BY_SPELLING)
def test_a_spelling_failure_lowers_the_label_never_the_name(smiles, name, rule):
    declined = assert_default_tier_declines(smiles)
    assert rule in _rules(declined), declined                  # the decline carries its reason
    strict = strict_path_row(smiles)
    assert (strict["name"], strict["tier"], strict["is_pin"]) == (name, "systematic_verified", False), strict
    assert rule in _rules(strict), strict
    best = assert_best_effort_gives(smiles, name)              # the same name, full-key round trip
    assert (best["tier"], best["is_pin"]) == ("systematic_verified", False), best
    assert rule in _rules(best), best


def test_name_with_confidence_declines_with_the_failures():
    """``name_with_confidence`` applies the default tier's rule to the name it returns: the
    lowered name is declined with NO_VERIFIED_PIN, and the record keeps the spelling failures as
    the reason, as the ``name_tiered`` decline row does. A decline for another reason (betaine:
    the strict path labels it systematic_verified) records none."""
    smiles, _name, rule = LOWERED[3]
    md = Orthonym().name_with_confidence(smiles)
    assert md["limit"]["code"] == "NO_VERIFIED_PIN", md
    assert rule in [f["rule"] for f in md["spelling_failures"]], md
    assert md["spelling_failures"] == default_tier_row(smiles)["spelling_failures"]
    other = Orthonym().name_with_confidence("C[N+](C)(C)CC(=O)[O-]")
    assert other["limit"]["code"] == "NO_VERIFIED_PIN" and other["spelling_failures"] == [], other


Q1_SMILES = "O=c1ccn2ncccc2c1"


def test_the_q1_name_is_pin_verified_only_with_its_indicated_hydrogen():
    """Item Q1. The PIN is '6H-pyrido[1,2-b]pyridazin-6-one', the Blue Book;
    ,:24768); before the indicated-hydrogen producer fix (lane L1a) the strict path
    writes 'pyrido[1,2-b]pyridazin-6-one', which OPSIN reads back to the same full InChIKey. In
    either state of the producer no pin_verified name omits the indicated hydrogen: the omission
    is lowered with, declined at the default tier, and keeps its best-effort name."""
    row = default_tier_row(Q1_SMILES)
    if row["tier"] == "pin_verified":
        assert row["name"] == "6H-pyrido[1,2-b]pyridazin-6-one", row
        assert not row.get("spelling_failures"), row
        return
    declined = assert_default_tier_declines(Q1_SMILES)
    assert "P-14.7.1" in _rules(declined), declined             # declined with its reason
    strict = strict_path_row(Q1_SMILES)
    assert (strict["name"], strict["tier"], strict["is_pin"]) == (
        "pyrido[1,2-b]pyridazin-6-one", "systematic_verified", False), strict
    best = assert_best_effort_gives(Q1_SMILES, "pyrido[1,2-b]pyridazin-6-one")
    assert (best["tier"], best["is_pin"]) == ("systematic_verified", False), best


#: 'hopane' (gate gold P14C-) is a name of the exact-match natural-product list; it keeps the
#: label of its path (user decision 2026-09-30), so the checks do not read it (the indicated-
#: hydrogen count misreads a saturated stereoparent name)
HOPANE = ("CC(C)[C@H]1CC[C@]2(C)[C@H]3CC[C@@H]4[C@@]5(C)CCCC(C)(C)[C@@H]5CC[C@@]4(C)[C@]3(C)"
          "CC[C@@H]12")


def test_a_name_of_the_exact_match_list_is_not_read():
    assert check_pin_spelling(Chem.MolFromSmiles(HOPANE), "hopane", strict=True)   # it would fail
    row = Orthonym().name_tiered(HOPANE)
    assert (row["name"], row["tier"]) == ("hopane", "pin_verified"), row
    assert not row.get("spelling_failures"), row


def test_a_name_of_the_exact_match_list_is_not_read_on_a_route_outside_p10(monkeypatch):
    """'hopane' reaches the label site through the NATURAL_PRODUCT route, so the class
    skip alone keeps its label; with that skip turned off, the list-name clause must still keep
    the checks from reading a name of the exact-match list (user decision 2026-09-30)."""
    import orthonym.namer as namer_mod
    monkeypatch.setattr(namer_mod, "_SPELLING_CHECK_SKIPPED_CLASSES", ())
    row = Orthonym().name_tiered(HOPANE)
    assert (row["name"], row["tier"]) == ("hopane", "pin_verified"), row
    assert not row.get("spelling_failures"), row


def test_a_name_of_a_p10_route_is_not_read():
    # 'aspart-4-ol' (the AMINO_ACID route, nomenclature;:50943 gives no PIN in
    # Chapter: its label is the natural-product decision's, not this check's
    smiles = "N[C@@H](CCO)C(=O)O"
    assert check_pin_spelling(Chem.MolFromSmiles(smiles), "aspart-4-ol", strict=True)
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == ("aspart-4-ol", "pin_verified"), row


def test_a_substitutive_name_of_the_peptide_route_is_read():
    # the PEPTIDE route emits a substitutive name here; (:3477): 'hydroxy' sorts before
    # 'pyrrolidine-2-carboxamido'
    smiles = "CSCCC(NC(=O)C(C)NC(=O)C(NC(=O)C1CCCN1)C(C)O)C(=O)O"
    declined = assert_default_tier_declines(smiles)
    assert "P-14.5" in _rules(declined), declined


def test_the_default_tier_reads_the_failures_before_its_grammar_carve_out():
    # a name of a format OPSIN has no grammar for ships on its construction ('carveout:<class>');
    # a spelling failure still declines it
    row = {"name": "x", "tier": "pin_unverified", "gate_outcome": "carveout:inositol",
           "source": "pin_path", "spelling_failures": []}
    assert _default_tier_emits(row, True, "C")
    row["spelling_failures"] = [{"rule": "P-14.5", "detail": "order"}]
    assert not _default_tier_emits(row, False, "C")
