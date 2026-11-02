"""-O-PH(=O)-R is the '[R(oxo)-λ5-phosphanyl]oxy' prefix, below the PIN.

 (the Blue Book) names '=P(O)OH' 'hydroxy(oxo)-λ5-phosphanylidene (preselected prefix)' (:36279);
the acyl-group name 'hydroxyphosphonoyl' (PH(O)<,:36049) is not read by OPSIN. Declining the P-H made a
277-atom phosphate polyether search every other split of its chain until it ran out of time (r0 of the slow rows)."""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("OCCO[PH](=O)O", "2-{[hydroxy(oxo)-λ5-phosphanyl]oxy}ethan-1-ol"),
    ("CCO[PH](=O)OCCO", "2-{[ethoxy(oxo)-λ5-phosphanyl]oxy}ethan-1-ol"),
])
def test_h_phosphonate_oxy_prefix(smiles, name):
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)
