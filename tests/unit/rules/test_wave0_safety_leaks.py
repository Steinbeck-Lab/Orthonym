"""Wave 0 safety: fail-close verified wrong-structure leaks.

P-32.2.2 — mixed-ring fused substituent (5/6-ring system with partial saturation)
must not emit the wrong-constitution 'inden-3-yl' name.
"""
from orthonym.namer import name_compound


def test_p32_mixed_ring_substituent_not_wrong_constitution():
    out = name_compound("OC(=O)CCC1=CCc2ccccc21", style="pin")
    # Must NOT emit the wrong 5-ring name. Correct PIN or honest unknown — never inden-3-yl.
    assert out != "3-(1H-inden-3-yl)propanoic acid"
    assert ("inden" not in out) or (out == "3-(3,4-dihydronaphthalen-1-yl)propanoic acid")


def test_p32_tetrahydronaphthalenyl_regression():
    """The symmetric fully-saturated 6+6 case must still be named correctly."""
    out = name_compound("OC(=O)CCC1CCCc2ccccc21", style="pin")
    assert out == "3-(1,2,3,4-tetrahydronaphthalen-1-yl)propanoic acid"
