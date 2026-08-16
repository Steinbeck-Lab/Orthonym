"""W2F-P7 Task 1 (P-68.3): standard-valence phosphanyl substituent on a chain/acid.

Wiring the existing ``rules/phosphorus.get_phosphanyl_prefix`` into the acyclic
substituent path via ``name_substituent`` Tier 1.93 so a ``-PH2`` on an acid
chain is cited as the ``phosphanyl`` prefix. The phosphoryl/phosphonic ``P=O``
oxoacid path must stay untouched (fail-closed collision guard, P-45.3.1).
"""

import orthonym


class TestPhosphanylSubstituent:
    def test_phosphanyl_on_acid_chain(self):
        # P-68.3 / P-65.1.1: -PH2 on propanoic acid at C3
        assert orthonym.name_compound("OC(=O)CCP", style="pin") == \
            "3-phosphanylpropanoic acid"

    def test_phosphanyl_on_alkane(self):
        # BB P-29.6.2.3: 'propyl' is the retained PIN prefix; phosphane parent
        assert orthonym.name_compound("CCCP", style="pin") == "propylphosphane"

    def test_phosphoric_acid_unchanged(self):
        # regression: P=O oxoacid must NOT be renamed as a phosphanyl
        assert orthonym.name_compound("OP(O)(O)=O", style="pin") == "phosphoric acid"

    def test_phosphoryl_unchanged(self):
        # regression: trimethylphosphine oxide P=O must NOT poach the phosphanyl tier
        assert orthonym.name_compound("CP(C)(C)=O", style="pin") == \
            "trimethyl-λ5-phosphanone"
