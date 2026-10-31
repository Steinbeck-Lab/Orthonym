"""Item 12a (labels): a pin_verified label at a wider tier needs the default tier to emit
the same string. The strict twin of ``Orthonym._strict_pin_twin_name`` runs with the default
tier's emission rule: a string the strict path builds but the default tier declines
(NO_VERIFIED_PIN) comes back as the label, never equal to the name, so the label is demoted.
Hard rule of the project: a PIN label only for a certified PIN."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate

BE = lambda: Orthonym(style="pin", **_emit_tier_flags("best-effort"))  # noqa: E731

#: rows the default tier declined (NO_VERIFIED_PIN) whose best-effort name was labelled pin_verified once the
#: 'a' ring (recorded as non-PIN) gave way to the book ring. DT3 step A: the promotion re-run now builds the
#: book ring at the default tier too (the same name, shipped pin_verified at both tiers), so these three no
#: longer decline; the rule they pin is the one above, read on every row below:
#: the label pin_verified at a wider tier iff the default tier emits the same string.
#: The first three rows are emitted by the default tier. The fourth is declined (an alkaloid,: the class
#: guard of the promotion re-run keeps it out). The fifth is emitted by the default tier too, as the PIN
#: '4-([2,2'-bithiophen]-5-yl)-2-hydroxybut-3-yn-1-yl acetate', but its best-effort name is another string (the
#: split ring assembly, systematic_verified), so the default tier does not emit "the same string" and the
#: wider tier must not label it pin_verified: that is why the rule holds on it.
LABEL_ROWS = [
    "COC(=O)CCc1cc(O)c(C)c(=O)o1",
    "COC(=O)CCCc1c(OC)cc(=O)oc1C",
    "COC(=O)/C=C/c1oc(=O)c(CO)c(OC)c1C=O",
    # was the tropane ester 'CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1', which the default tier emits since DT4 F2
    "COc1ccc(-c2cc3cc(CCCO)ccc3o2)c(O)c1",
    "CC(=O)OCC(O)C#Cc1ccc(-c2cccs2)s1",
]


@pytest.mark.parametrize("smiles", LABEL_ROWS)
def test_a_wider_tier_labels_pin_verified_only_the_name_the_default_tier_emits(smiles):
    """Changed asserted value (DT3 step A): the rule was read on three rows the default tier declined
    ('the default tier declines, so best-effort is not pin_verified'); those rows are emitted now, so it is read
    in both directions on rows of both kinds. A tier that labels a name pin_verified when the default tier
    declines it, or declines to label the name the default tier emits, breaks the equivalence."""
    d = Orthonym(style="pin").name_tiered(smiles)
    row = BE().name_tiered(smiles)
    emitted = d["tier"] == "pin_verified" and d["name"] == row["name"]
    assert row["name"]
    assert (row["tier"] == "pin_verified") == emitted, (d, row)
    assert row["is_pin"] is emitted


def test_the_rows_cover_both_sides_of_the_rule():
    kinds = {Orthonym(style="pin").name_tiered(smiles)["tier"] == "pin_verified" for smiles in LABEL_ROWS}
    assert kinds == {True, False}


@pytest.mark.parametrize("smiles", ["COC(=O)CCc1ccccc1", "CC(=O)Nc1ccc(O)cc1"])
def test_a_name_the_default_tier_emits_keeps_its_label_at_best_effort(smiles):
    d = Orthonym(style="pin").name_tiered(smiles)
    b = BE().name_tiered(smiles)
    assert d["tier"] == "pin_verified" and b["tier"] == "pin_verified" and d["name"] == b["name"]
