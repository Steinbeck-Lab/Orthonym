"""The alkyl group of an ester of 2-methylpropan-2-ol is 'tert-butyl', not '2-methylpropan-2-yl'
(lane W2, item 27).

 "Retained prefixes that are preferred prefixes" (the Blue Book),:16282: "The
retained name 'tert-butyl' has never been recommended for further substitution; this is
maintained in these recommendations."; "PREFIXES DERIVED FROM PARENT HYDRIDES" (:24381),
:24412 "-C(CH3)3 tert-butyl (preferred prefix)"; "Monoesters" (:31743), example
:31751: "tert-butyl octanoate (PIN)". The '2-methylpropan-2-yl (PIN)' of:40453 names the RADICAL
,:40426); (1) "General rules for the selection of preferred names" (:40376):
"the preferred IUPAC name for a radical may not be the same as the preferred prefix".
A SUBSTITUTED group keeps the systematic name, '(1-chloro-2-methylpropan-2-yl)silane (PIN)'
(:16288); the other simple alkyl groups are named by their systematic prefixes, as before:
'propan-2-yl (preferred prefix)',:24402; isopropyl is for general nomenclature only,
 :16334), 'butan-2-yl (preferred prefix) (not sec-butyl)' and '2-methylpropyl (preferred
prefix) (not isobutyl)',:16398;:16412,:16416-16418).
The decomposition engine's ``_alcohol_to_alkyl`` turned the alcohol name '2-methylpropan-2-ol' into
'2-methylpropan-2-yl' (the locant-2 rule for a secondary or tertiary group) and the ester was
labelled pin_verified."""
import pytest

from orthonym.decomposition.fragment_assembly import _alcohol_to_alkyl
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

ALCOHOL_ROWS = [
    ("2-methylpropan-2-ol", "tert-butyl"),                      # the item
    ("1-chloro-2-methylpropan-2-ol", "1-chloro-2-methylpropan-2-yl"),   # substituted: systematic,:16288
    ("2-methylbutan-2-ol", "2-methylbutan-2-yl"),
    ("propan-2-ol", "propan-2-yl"),
    ("butan-2-ol", "butan-2-yl"),
    ("2-methylpropan-1-ol", "2-methylpropyl"),
    ("2,2-dimethylpropan-1-ol", "2,2-dimethylpropyl"),
    ("methanol", "methyl"),
    ("cyclohexanol", "cyclohexyl"),
]


@pytest.mark.parametrize("alcohol,prefix", ALCOHOL_ROWS)
def test_alcohol_to_alkyl(alcohol, prefix):
    assert _alcohol_to_alkyl(alcohol) == prefix


ESTER_ROWS = [
    # the two molecules of the review: main '2-methylpropan-2-yl (3-carbamoyl...)acetate', pin_verified
    ("CC(C)(C)OC(=O)Cn1c(C)c(Cl)c(C(N)=O)c1C",
     "tert-butyl (3-carbamoyl-4-chloro-2,5-dimethyl-1H-pyrrol-1-yl)acetate"),
    ("CC(C)(C)OC(=O)Cn1ccc(C(N)=O)c1", "tert-butyl (3-carbamoyl-1H-pyrrol-1-yl)acetate"),
    # the siblings that were already right, one per alkyl group
    ("CC(C)(C)OC(=O)Cn1ccnc1", "tert-butyl (1H-imidazol-1-yl)acetate"),
    ("CC(C)OC(=O)Cn1ccc(C(N)=O)c1", "propan-2-yl (3-carbamoyl-1H-pyrrol-1-yl)acetate"),
    ("CCC(C)OC(=O)Cn1ccc(C(N)=O)c1", "butan-2-yl (3-carbamoyl-1H-pyrrol-1-yl)acetate"),
    ("CC(C)COC(=O)Cn1ccc(C(N)=O)c1", "2-methylpropyl (3-carbamoyl-1H-pyrrol-1-yl)acetate"),
    ("CC(C)(C)COC(=O)Cn1ccc(C(N)=O)c1", "2,2-dimethylpropyl (3-carbamoyl-1H-pyrrol-1-yl)acetate"),
    # a substituted tert-butyl is a systematic group
    ("ClCC(C)(C)OC(=O)Cn1ccc(C(N)=O)c1",
     "1-chloro-2-methylpropan-2-yl (3-carbamoyl-1H-pyrrol-1-yl)acetate"),
]


@pytest.mark.parametrize("smiles,expected", ESTER_ROWS)
def test_ester_alkyl_group_at_the_default_tier(smiles, expected):
    from tests.support.default_tier import default_tier_row
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


def test_best_effort_names_the_item_rows_and_reads_back():
    for smiles, _ in ESTER_ROWS[:2]:
        row = name_best_effort(smiles)
        assert "2-methylpropan-2-yl" not in row["name"], row
        assert_full_rt(row["name"], smiles)
