from orthonym.validation import opsin_ambiguity

from ._helpers import needs_phane_opsin

pytestmark = needs_phane_opsin

LONG_LOCANT = "2,5,7,10,12,15-hexathia-1,16(3),6,11(3,4)-tetrafuranahexadecaphane"       #:23965


def test_the_skeleton_locant_warning_is_not_an_ambiguity():
    # (:15111): a plain number is a skeleton locant, an amplificant locant is a superscript
    from orthonym.jvm_bridge import opsin_warnings
    assert any("could refer to an atom of the simplified skeleton" in m for _, m in opsin_warnings(LONG_LOCANT))
    assert opsin_ambiguity.ambiguity_messages(LONG_LOCANT) == ()


def test_every_other_ambiguity_warning_still_counts():
    # (:2869): 'aziridinyl' names no position of the ring
    assert opsin_ambiguity.ambiguity_messages("methyl aziridinylacetate")
