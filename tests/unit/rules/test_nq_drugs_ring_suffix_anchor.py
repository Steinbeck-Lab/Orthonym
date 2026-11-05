"""Rows that guard the numbering of rings that carry a principal group and an added-carbon
group, against an anchor that cites the wrong suffix.

 (the Blue Book,:18184) ranks esters above amides, so on a ring that carries an
ester the ester is the suffix ('-carboxylate') and a carboxamide, nitrile or formyl group is
a prefix. The names below are main's; the hydro-heterocycle rows pin that no anchor makes the
amide the cited suffix of a ring whose principal group is an ester.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("COC(=O)C1CCC(C(N)=O)C(C)C1", "methyl 4-carbamoyl-3-methylcyclohexane-1-carboxylate"),
    ("CCOC(=O)C1CC(C(N)=O)CCC1", "ethyl 3-carbamoylcyclohexane-1-carboxylate"),
    ("COC(=O)C1CCC(C#N)C(C)C1", "methyl 4-cyano-3-methylcyclohexane-1-carboxylate"),
])
def test_a_carbocycle_keeps_the_principal_groups_suffix_at_locant_1(smiles, name):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # The two hydro-heterocycle rows that stood here ('COC(=O)C1=CCCN(C)C1C(N)=O' and
    # 'COC(=O)C1=CCCOC1C(N)=O') named an amide or a '(methoxycarbonyl)' prefix because the ring
    # writer dropped its non-cited ring group; they are now the ester names
    # 'methyl 2-carbamoyl-...-3-carboxylate' (PIN,: esters 9 above amides 11), pinned with an
    # OPSIN full-InChIKey read-back in test_leads_l5_item22.py (leads item 22).
    ("O=C1OCC(C(N)=O)C1C", "4-methyl-5-oxooxolane-3-carboxamide"),
])
def test_a_lactone_with_an_amide_names_the_amide_as_the_suffix(smiles, name):
    """change-asserted-value (lane/l17 fix a performance pass; it already failed at c302d83df4, from item 12): the old name was
    '3-methyl-2-oxooxolane-4-carboxamide' (systematic_verified, the PIN tier abstained), and the test named the ring's lactone
    an ester 'whose principal group' beats the amide. A lactone is a cyclic ester named as a heterocycle, a pseudoketone: (the Blue Book) "lactones and other cyclic esters are named as heterocycles; see 16 below", class 16
    (:18189) "See Classes 8, 9, 11 and 13, on lactones, lactams"; (:29314) "Cyclic anhydrides, esters and amides are
    named as pseudoketones"; so the carboxamide (class 11,:18184) is the suffix and the ring C=O the prefix 'oxo',
    and the principal group takes the lowest locant, 3 rule c): '4-methyl-5-oxooxolane-3-carboxamide', not the
    locant 4 of the old name. Independent check: a fresh OPSIN call reads the name to the input's full InChIKey. Mutation:
    with ``seniority._head_principal_group`` not stepping aside for a ring ester the old name returns."""
    from tests.support.rt_assert import assert_full_rt
    assert Orthonym(style="pin").name_tiered(smiles)["name"] == name
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(name, smiles)
