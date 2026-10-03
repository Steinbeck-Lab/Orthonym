"""Esters of nitric and nitrous acid outrank a junior principal group.

, Table 4.1 (the Blue Book): esters (class 9) rank above amides, nitriles, aldehydes,
ketones, alcohols, phenols and amines. (:35916),:35918: "Esters of mononuclear noncarbon
acids are named in the same way as esters of organic acids"; the book's own junior-group row is
'CH3-CO-CH2-CH2-O-BrO2 3-oxobutyl bromate (PIN)' (:35974). The polyfunctional producer named these
with the junior group as the parent ('2-(nitrooxy)ethan-1-ol', '4-(nitrooxy)butan-2-one') and the
PIN tier declined. Every name below reads back to the input's full InChIKey with OPSIN 2.9.0.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("OCCO[N+](=O)[O-]", "2-hydroxyethyl nitrate"),
    ("CC(=O)CCO[N+](=O)[O-]", "3-oxobutyl nitrate"),
    ("NCCO[N+](=O)[O-]", "2-aminoethyl nitrate"),
    ("Oc1ccc(O[N+](=O)[O-])cc1", "4-hydroxyphenyl nitrate"),
    ("OCC(O)CO[N+](=O)[O-]", "2,3-dihydroxypropyl nitrate"),
    ("OCCON=O", "2-hydroxyethyl nitrite"),
])
def test_noncarbon_ester_names_the_molecule(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", [
    ("ClCCO[N+](=O)[O-]", "2-chloroethyl nitrate"),          # no principal group: unchanged
    ("CCCCCON=O", "pentyl nitrite"),                         #:35922
    ("OC(=O)CCO[N+](=O)[O-]", "3-(nitrooxy)propanoic acid"),  # an acid outranks the ester
])
def test_controls(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", [
    "N#CCCO[N+](=O)[O-]",          # PIN '2-cyanoethyl nitrate'
    "CC(=O)NCCO[N+](=O)[O-]",      # PIN '2-acetamidoethyl nitrate'
])
def test_an_alcohol_part_the_prefix_pipeline_cannot_name_keeps_its_name(smiles):
    # the strict substituent pipeline returns its refusal sentinel for these alcohol parts;
    # the ester handler declines ('substituent nitrate' is no name) and the molecule keeps
    # the name the other producers give it: best-effort RT-exact, the PIN tier no worse
    # than before (it abstained)
    from tests.support.rt_assert import assert_tier_contract
    assert_tier_contract(smiles)
