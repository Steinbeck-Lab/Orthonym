"""A ring substituent group whose substitutable positions all carry the same substituent cites
no locants for them.

 (the Blue Book): "All locants are omitted in compounds or substituent groups in
which all substitutable positions are completely substituted or modified, for example, by hydro,
in the same way"; '1-chloro-2-(pentafluoroethyl)benzene (PIN)' (:3023), 'benzenehexayl
(preferred prefix)' (:3021). The ring prefix producers wrote '2,3,4,5,6-pentafluorophenyl'
(pin_verified). A partial or mixed substitution keeps its locants (:3009). OPSIN 2.9.0 reads every
name back to the input's full InChIKey.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("OCc1c(F)c(F)c(F)c(F)c1F", "(pentafluorophenyl)methanol"),
    ("CC(=O)Oc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "pentachlorophenyl acetate"),
    ("OC(=O)Cc1c(F)c(F)c(F)c(F)c1F", "(pentafluorophenyl)acetic acid"),
    ("OC(=O)COc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "(pentachlorophenoxy)acetic acid"),
    ("OCc1c(C)c(C)c(C)c(C)c1C", "(pentamethylphenyl)methanol"),
    ("OCc1c(F)c(F)nc(F)c1F", "(tetrafluoropyridin-4-yl)methanol"),
    # partial or mixed: the locants stay
    ("Cc1c(F)c(F)c(CO)c(F)c1F", "(2,3,5,6-tetrafluoro-4-methylphenyl)methanol"),
    ("FC(F)(F)C(F)(F)c1ccccc1Cl", "1-chloro-2-(pentafluoroethyl)benzene"),
])
def test_completely_substituted_ring_prefix(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
