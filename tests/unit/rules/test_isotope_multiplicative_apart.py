"""Labels that make the identical units of a multiplicative name unlike (program 22, 2026-11-10).

 (the Blue Book): "Isotopically modified atoms or groups that are not identically modified in equivalent
positions are expressed separately." No spelling of '4,4'-(propane-2,2-diyl)diphenol' carries a label on one phenol
only, so the labelled molecule got no name. The isotope decorator's second skeleton (``rules.isotope_alternative``)
now withholds the multiplicative producer, and the next producer names the units apart. (:21477) makes
the unit with the modified atoms the senior parent, which the spliced descriptor does not decide, so the name is
``systematic_verified``, never a PIN.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("OC=1C(=CC(=CC1)C(C)(C)C1=CC=C(C=C1)O)[2H]", "4-[2-(4-hydroxyphenyl)propan-2-yl](2-2H)phenol"),
    ("O[13C]1=CC=C(C=C1)C(C)(C)C1=CC=C(C=C1)O", "4-[2-(4-hydroxyphenyl)propan-2-yl](1-13C)phenol"),
    ("C[Si](C)(C)C=1C(=C(C(=C(C1[2H])[2H])[2H])[2H])[Si](C)(C)C", "1,2-bis(trimethylsilyl)(3,4,5,6-2H4)benzene"),
])
def test_units_of_a_multiplicative_name_made_unlike_are_named_apart(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


def test_identically_labelled_units_keep_the_multiplicative_name(best_effort):
    """The second skeleton runs only when the first gives no name: equal labels on both units stay multiplied."""
    row = best_effort.name_tiered("O(C1=CC=[14CH]C=C1)C1=CC=[14CH]C=C1")
    assert row["name"] == "1,1'-oxydi(4-14C)benzene"
