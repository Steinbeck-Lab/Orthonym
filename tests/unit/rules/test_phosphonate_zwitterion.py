"""A phosphonic monoester zwitterion: R-P(=O)(O-)-O-R' with a quaternary ammonium group in R'.

The anion of the monoester is the ester name without 'hydrogen' ('ethyl methylphosphonate',, the Blue Book:
35922), the cation a prefix inside the ester group,:42462), as for the phosphoric diester twins
('methyl 2-(trimethylazaniumyl)ethyl phosphate'); the PIN tier writes the ammonium prefix as the book's '-aminiumyl'
,:42296).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin,best", [
    ("CP(=O)([O-])OCC[N+](C)(C)C", "2-(N,N-dimethylmethanaminiumyl)ethyl methylphosphonate",
     "2-(trimethylazaniumyl)ethyl methylphosphonate"),
    ("ClCP(=O)([O-])OCC[N+](C)(C)C", "2-(N,N-dimethylmethanaminiumyl)ethyl (chloromethyl)phosphonate",
     "2-(trimethylazaniumyl)ethyl (chloromethyl)phosphonate"),
])
def test_the_phosphonate_zwitterion_names_like_its_phosphate_twin(smiles, pin, best):
    assert (tier_row(smiles, "pin")["name"], tier_row(smiles, "pin")["tier"]) == (pin, "pin_verified")
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (best, "systematic_verified"), row
    assert name_is_rt_exact(best, smiles)
