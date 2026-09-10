"""W2F-P7 Task 5 (h)): the nonstandard-valence atom gets the lower locant.

BB (h) (the Blue Book,3334): when a numbering choice remains, the
substituent whose attachment atom is in a NONSTANDARD (λ) valence state is
assigned the lower locant. 'OC(C[PH4])CP' -> the λ5-phosphanyl arm takes C1.
"""

import orthonym


def test_p1444h_lambda_lowest_locant():
    assert orthonym.name_compound("OC(C[PH4])CP", style="pin") == \
        "1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol"
