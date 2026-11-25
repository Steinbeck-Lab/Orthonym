"""Tin and lead cations of charge 2+ and 3+ named on the parent hydride of the metal.

 (the Blue Book): "stannane, SnH4, and plumbane, PbH4... by replacing the 'ane' ending... by the
suffix 'ylium'"; the dications and trications take 'bis(ylium)' / 'tris(ylium)' ('ethylazanebis(ylium) (PIN)',:41672).
``data/metal_scope.py`` admits tin and lead to PARENT_HYDRIDE_IONS. OPSIN reads a substituted Sn2+/Pb2+ centre as
divalent, so 'dimethylstannanebis(ylium)' reads as nothing; the producer then offers the name with the standard bonding
number cited, 'dimethyl-λ4-stannanebis(ylium)' "Standard bonding numbers", Table 1.3: 4 for Sn and Pb), which
is redundant, not wrong. Every name is best-effort and recorded as no PIN.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[Sn+2]C", "dimethyl-λ4-stannanebis(ylium)"),
    ("CCCC[Pb+2]CCCC", "dibutyl-λ4-plumbanebis(ylium)"),
    ("CCCO[Sn+3]", "propoxystannanetris(ylium)"),
    ("CC(C)C[Sn+2]CC(C)C.C(CCC(=O)[O-])CC(=O)[O-]", "hexanedioate—bis(2-methylpropyl)-λ4-stannanebis(ylium) (1/1)"),
])
def test_a_tin_or_lead_cation_of_charge_two_or_three(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_the_singly_charged_cation_and_the_silicon_dication_keep_their_names():
    assert name_best_effort("C[Sn+](C)C")["name"] == "trimethylstannylium"
    assert name_best_effort("C[Si+2]C")["name"] == "dimethylsilanebis(ylium)"
