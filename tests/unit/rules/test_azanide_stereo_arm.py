"""A disubstituted azanide whose arm has an E/Z double bond is named, the descriptor composed onto the name.

``charged_router.emit_secondary_amine_azanide`` names the two arms of an N-anion R-[N-]-R' as prefixes and appends
'azanide', the Blue Book, 'acetylazanide (PIN)'). An arm the strict prefix namer declines
(the 1,3-diazapentadienyl of a beta-diketiminate) is named by the universal substituent namer at the wider tiers,
and neither cites the E/Z of its double bond, so the name read back to the right skeleton with the bond
undefined and the molecule abstained. The stereo-omission reclaim (``namer._try_compose_input_stereo``) puts the
descriptor on the name with OPSIN's own locants and returns it only when it reads back to the input's full
InChIKey: '(1Z)-...azanide'.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

ROWS = [
    # an N-aryl-N-vinyl anion with an N-aryl imine on the vinyl group (a beta-diketiminate arm)
    "CC1=C(C(=CC=C1)C)[N-]/C(=C\\C2=CC=CC=C2)/C(=NC3=C(C=CC=C3C)C)[Si](C)(C)C",
    "CC(C)C1=C(C(=CC=C1)C(C)C)[N-]/C(=C\\C(=NC2=C(C=CC=C2C(C)C)C(C)C)C3=CC=CC=C3)/C4=CC=CC=C4",
    # the same anion with its beryllium cation and a second anion
    "[Be+2].CC(C)C1=C(C(=CC=C1)C(C)C)[N-]/C(=C\\C(=NC2=C(C=CC=C2C(C)C)C(C)C)C3=CC=CC=C3)/C4=CC=CC=C4.C(CCI)C[O-]",
]


@pytest.mark.parametrize("smiles", ROWS)
def test_azanide_with_a_stereo_double_bond_in_an_arm(smiles):
    res = name_best_effort(smiles)
    assert res["name"] and "azanide" in res["name"], res
    assert name_is_rt_exact(res["name"], smiles), res
