"""W2F-P7 Task 4: the evidence PIN, end-to-end.

BB verbatim (the Blue Book): the PIN parent contains the
substituent with the highest bonding number (λ5 > λ3), and citation order is
'phosphanyl' < 'phosphanylmethyl' (shorter is a prefix of the longer) so
'3-(λ5-phosphanyl)' is cited FIRST despite the higher locant.
"""

import orthonym


def test_evidence_pin():
    assert orthonym.name_compound("OC(=O)C(CP)C[PH4]", style="pin") == \
        "3-(λ5-phosphanyl)-2-(phosphanylmethyl)propanoic acid"
