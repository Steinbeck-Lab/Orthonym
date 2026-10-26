"""PIN class program, Task 20: anhydrides of noncarbon acids,,.

 "Pseudoesters" (the Blue Book): when 'E', in R-CO-O-E, is B, Al, In, Ga, Tl, Si, Ge, Sn,
Pb, N (cyclic), P, As, Sb or Bi, "the pseudoester is named as a traditional ester, unless other names must be
selected in accordance with the seniority order of classes, in decreasing order: salts > acids > anhydrides >
esters. For anhydrides, see." The book's esters stay esters: 'trimethylsilyl acetate (PIN)' (:32053),
'phosphinan-1-yl benzoate (PIN)' (:32064); its anhydride is 'acetic phosphinous anhydride (PIN)' (:32064 ff.).
 "Mixed anhydrides" (:32253): "Mixed anhydrides with carbonic acid, cyanic acid, and inorganic acids are
named as anhydrides", 'benzoic phosphinous anhydride (PIN)' (:32280); "Anhydrides of mononuclear
noncarbon oxoacids" (:35982): 'acetic dimethylarsinic anhydride (PIN)' (:35992), 'acetic dimethylborinic
anhydride (PIN)' (:35990). The two acids are cited in alphabetical order, without 'acid', before 'anhydride'.

Not built here (internal notes): 'dimethylborinic acid' has no name in the pipeline, so
the borinic row stays an ester name; the polybasic acids (boric, phosphoric: the numerical prefix of the
anhydride linkages,:32283) and the choice of the O-anhydride over an S-anhydride,:32451).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("CC(=O)OP", "acetic phosphinous anhydride"),                      #:32064 ff.,
    ("CC(=O)OP(C)C", "acetic dimethylphosphinous anhydride"),
    ("O=C(OP)c1ccccc1", "benzoic phosphinous anhydride"),              #,:32280
    ("CC(=O)O[As](C)(C)=O", "acetic dimethylarsinic anhydride"),       #,:35992
]

# The same class, other members (each read back by assert_pin_at_both_tiers).
PIN_ROWS_MORE = [
    ("CC(=O)OP(C)c1ccccc1", "acetic methyl(phenyl)phosphinous anhydride"),
    ("CC(=O)OP(C)(C)=O", "acetic dimethylphosphinic anhydride"),
    ("CC(=O)O[Sb](C)C", "acetic dimethylstibinous anhydride"),
    # (:32253) "citing in alphabetical order the names of the two acids": an acid name is a whole name,
    # (:3477) "considered to begin with the first letter of its complete name": 'dimethylphosphinous' stands
    # at 'd', before 'formic'
    ("O=COP(C)C", "dimethylphosphinous formic anhydride"),
]

# The Blue Book's esters of pseudoesters, which stay esters, and the groups the rule does not take.
CONTROL_ROWS = [
    ("CC(=O)O[Si](C)(C)C", "trimethylsilyl acetate"),                  #:32053
    ("O=C(OP1CCCCC1)c1ccccc1", "phosphinan-1-yl benzoate"),            #:32064, ring phosphorus
    ("CC(=O)O[P@](C)c1ccccc1", "(R)-methyl(phenyl)phosphanyl acetate"),  # a stereogenic E is not built
]

# A dibasic acyl side with both groups on E is no '<acid> <acid> anhydride' of this rule: the name is the one before
# (a best-effort name that is no PIN; the PIN tier declines it)
UNCHANGED_ROWS = ["O=C(OP(C)C)C(=O)OP(C)C"]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + PIN_ROWS_MORE + CONTROL_ROWS)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", UNCHANGED_ROWS)
def test_a_dibasic_acyl_side_keeps_its_ester_name(smiles):
    from tests.support.pin_tiers import assert_declined_at_default, name_breadth
    assert_declined_at_default(smiles)
    assert "anhydride" not in name_breadth(smiles)["name"]
