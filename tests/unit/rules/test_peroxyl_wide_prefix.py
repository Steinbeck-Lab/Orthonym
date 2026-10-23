"""A peroxyl radical R-O-O. whose group R only the wider substituent route names.

 (the Blue Book-40709) names the radical by method (1), R + 'peroxyl' ('hexanoylperoxyl (PIN)'),
and ``_name_peroxyl_radical`` named R with the strict substituent namer alone. A fused or hetero ring system the
strict namer declines (a purine-5-yl) left the radical unnamed. At the wider tiers the group is named by the
same wider substituent route every other radical producer uses (``_n_substituent_prefix``, as for N-oxyl and
methane-centred radicals), the strict radical round trip confirms the name, and it is demoted out of
pin_verified (the suite runs with the validity gate off, so the labels are not asserted here).
"""
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

PURINYL_PEROXYL = "CN1C(N=C2C1(C(=O)NC(=O)N2)O[O])O"


def test_peroxyl_radical_of_a_fused_ring_group_is_named_at_best_effort():
    res = name_best_effort(PURINYL_PEROXYL)
    assert res["name"] == "(8-hydroxy-7-methyl-2,6-dioxo-1,2,3,6,7,8-hexahydro-5H-purin-5-yl)peroxyl", res
    assert name_is_rt_exact(res["name"], PURINYL_PEROXYL)


def test_peroxyl_radical_of_a_plain_group_keeps_its_pin():
    res = name_best_effort("CCO[O]")
    assert res["name"] == "ethylperoxyl", res
