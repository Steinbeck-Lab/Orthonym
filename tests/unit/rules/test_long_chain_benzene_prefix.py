"""A chain of more than ten atoms on a benzene ring is a substituent prefix of the ring.

The benzene substituent collector (``rules.benzene._identify_substituent``) handed a fragment of more than ten atoms
to the substituent namer only for a ring-nitrogen '-yl' group or an 'a' chain, so an unsaturated or substituted chain
of eleven or more atoms ('undec-10-en-1-yl', '7-hydroxydodecyl') had no prefix and the PIN tier declined the molecule.
A ring is the parent and every chain on it a substituent, (1) (the Blue Book) "Within the same class,
a ring or ring system has seniority over a chain", and (:25691) "Names of cyclic hydrocarbons substituted by
chains are formed by substituting chains, saturated or unsaturated, into rings... This rule must be strictly applied
in the context of preferred IUPAC names". ``rules.ring_nitrogen_yl.large_substituent_in_scope`` now admits a group
without a ring atom.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("C=CCCCCCCCCCc1ccccc1", "(undec-10-en-1-yl)benzene"),
    ("CC/C=C/CCCCCCCCc1ccccc1", "[(9E)-dodec-9-en-1-yl]benzene"),
    ("CCCCC/C=C/CCCCCCCCCCc1cc(O)cc(O)c1", "5-[(11E)-heptadec-11-en-1-yl]benzene-1,3-diol"),
    ("CCCCCC(O)CCCCCCc1cc(O)cc(O)c1", "5-(7-hydroxydodecyl)benzene-1,3-diol"),
    ("CCCCCC(O)CCCCCCCCCCc1ccc(O)cc1", "4-(11-hydroxyhexadecyl)phenol"),
    # the chains of ten atoms or fewer and the saturated chains keep their names
    ("CC/C=C/CCCCCCc1cc(O)cc(O)c1", "5-[(7E)-dec-7-en-1-yl]benzene-1,3-diol"),
    ("CCCCCCCCCCCCCCCCCc1cc(O)cc(O)c1", "5-heptadecylbenzene-1,3-diol"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)
