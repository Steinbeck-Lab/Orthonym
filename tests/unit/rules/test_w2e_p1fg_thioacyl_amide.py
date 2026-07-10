"""P-66.1.4.2 (BB 33172): "CH3-CS-NH-CS-CH3
N-(ethanethioyl)ethanethioamide (PIN)" — BB verbatim example.

BUILT (W2E-D3): a dedicated fail-closed thioimide subsystem
(``rules/thioimides.py``, dispatch THIOIMIDE@49.7). Reproduce-first at the
CODE level showed the original W2E-P1FG spec named the wrong subsystem: the
O-imide analogue ``CCC(=O)NC=O -> N-propanoylformamide`` is produced by the
DECOMPOSITION engine (it cleaves the C(=O)-N amide bond and reassembles),
NOT by the amide handler / a substituent producer. The decomposition engine
never reaches the thio case because its amide-bond SMARTS ``[CX3](=O)[NX3]``
requires C(=O); the C(=S)-N bond is classified as a secondary amine and the
molecule falls through to GENERAL and fails closed ('unknown organic
compound'). Rather than make the whole decomposition/amide/perception stack
chalcogen-aware (broad ripple), a dedicated graph classifier recognises
exactly the acyclic N-H thioimide R-C(=S)-NH-C(=S)-R' with plain
alkanethioyl branches, names parent = shorter chain's alkanethioamide +
N-(longer chain)alkanethioyl (mirroring the O-imide decomposition
convention), and returns None for everything else (N-substituted / branched
/ aryl / unsaturated / mixed-O forms) so the cascade continues.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestThioacylThioamide:
    def test_n_ethanethioyl_ethanethioamide(self):
        assert name_compound("CC(=S)NC(C)=S", style="pin") \
            == "N-(ethanethioyl)ethanethioamide"

    def test_thioimide_asymmetric_propanethioyl(self):
        # Asymmetric: shorter chain (C2) is the parent thioamide, longer (C3)
        # is the N-(alkanethioyl) substituent (OPSIN round-trips to CCC(=S)NC(C)=S).
        assert name_compound("CCC(=S)NC(C)=S", style="pin") \
            == "N-(propanethioyl)ethanethioamide"

    def test_thioimide_methanethioyl(self):
        # C1 formyl-analogue branch (methanethioyl / methanethioamide).
        assert name_compound("CC(=S)NC=S", style="pin") \
            == "N-(ethanethioyl)methanethioamide"

    def test_n_substituted_thioimide_fails_closed(self):
        # N-substituted thioimides are NOT built (defer). Must never emit the
        # N-H thioimide target as if it were this different molecule.
        n = name_compound("CC(=S)N(C)C(C)=S", style="pin")
        assert n != "N-(ethanethioyl)ethanethioamide"

    def test_thioamide_parent_protect(self):
        assert name_compound("CC(=S)N", style="pin") == "ethanethioamide"

    def test_n_methyl_thioamide_protect(self):
        assert name_compound("CC(=S)NC", style="pin") == "N-methylethanethioamide"

    def test_o_acyl_protect(self):
        assert name_compound("CCC(=O)NC=O", style="pin") == "N-propanoylformamide"
