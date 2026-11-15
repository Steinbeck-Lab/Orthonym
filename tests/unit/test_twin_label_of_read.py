"""The label of a name the gate proves through a twin spelling is the twin's (lane D1 of the all-organic program).

``metrics.provenance.verified_field_of_read``: a name read through ``validation.twin_spellings`` carries
``opsin_twin``, never ``opsin`` (CIDs 145774203 and 145774252 carried ``opsin`` with the gate outcome
``twin_verified:halido_o``).
"""
import pytest

from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles", [
    "C1=CC=C(C=C1)N(C=[Ru](Cl)Cl)C(=O)C2=CC=C(C=C2)[N+](=O)[O-]",
    "COC1=CC=C(C=C1)N(C=[Ru](Cl)Cl)C(=O)C2=CC=C(C=C2)[N+](=O)[O-]",
])
def test_a_twin_read_is_labelled_opsin_twin(smiles):
    r = name_best_effort(smiles)
    assert r["gate_outcome"].startswith("twin_verified:") and r["verified"] == "opsin_twin", r
