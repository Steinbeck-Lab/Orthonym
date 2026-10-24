"""The one-carbon amide is 'formamide' in the fragment assembly, the Blue Book)."""
import pytest

from tests.support.pin_tiers import name_breadth, name_default


def test_the_one_carbon_amide_is_formamide_and_a_substituted_one_is_not():
    """ (the Blue Book): formamide is retained and substitution is on the N; a prefix on the
    carbon takes another parent (``carbonochloridic amide``,:32707, 'not 1-chloroformamide')."""
    from orthonym.assembly.composition_primitives import retained_amide_word
    assert retained_amide_word("meth", "amide", 1, False, []) == "formamide"
    assert retained_amide_word("meth", "amide", 1, False, ["N-hydroxy"]) == "formamide"
    assert retained_amide_word("meth", "amide", 1, False, ["chloro"]) is None
    assert retained_amide_word("meth", "amide", 2, False, []) is None
    assert retained_amide_word("eth", "amide", 1, False, ["2-chloro"]) == "acetamide"
    assert retained_amide_word("eth", "amide", 1, True, []) is None
    assert retained_amide_word("prop", "amide", 1, False, []) is None


@pytest.mark.opsin_gate
def test_the_fragment_assembly_writes_formamide_not_methanamide():
    """'N-hydroxyformamide' is the name of HCO-NH-OH, (:32699): 'formamide (PIN)'; substitution is
    permitted on the -NH2 group."""
    for res in (name_default("ONC=O"), name_breadth("ONC=O")):
        assert (res["name"], res["tier"]) == ("N-hydroxyformamide", "pin_verified"), res
