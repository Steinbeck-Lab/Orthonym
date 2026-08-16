"""W2F-P7 Task 4 (P-45.3.1): the evidence PIN, end-to-end.

BB P-45.3.1 verbatim (BlueBookV2.md:22182): the PIN parent contains the
substituent with the highest bonding number (λ5 > λ3), and citation order is
'phosphanyl' < 'phosphanylmethyl' (shorter is a prefix of the longer) so
'3-(λ5-phosphanyl)' is cited FIRST despite the higher locant.
"""

import orthonym


def test_evidence_pin():
    assert orthonym.name_compound("OC(=O)C(CP)C[PH4]", style="pin") == \
        "3-(λ5-phosphanyl)-2-(phosphanylmethyl)propanoic acid"
