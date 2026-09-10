"""W2F-P7 Task 3: emit the (λ5-phosphanyl) substituent prefix.

House Greek 'λ5' with NO internal locant on the mononuclear prefix; the
standard 'phosphanyl' path stays byte-identical (regression).
"""

import orthonym


def test_lambda5_phosphanyl_ethanol():
    assert orthonym.name_compound("OCC[PH4]", style="pin") == \
        "2-(λ5-phosphanyl)ethan-1-ol"


def test_lambda5_phosphanyl_propanoic():
    assert orthonym.name_compound("OC(=O)CC[PH4]", style="pin") == \
        "3-(λ5-phosphanyl)propanoic acid"


def test_plain_phosphanyl_regression():
    assert orthonym.name_compound("OC(=O)CCP", style="pin") == \
        "3-phosphanylpropanoic acid"
