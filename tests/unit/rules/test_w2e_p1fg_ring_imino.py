""" (BB 26540): "The prefix 'imino' for =NH is used in presence
of characteristic groups having seniority over imines."
Target: quinone-imine O=C1C=CC(=N)C=C1 -> 4-iminocyclohexa-2,5-dien-1-one
(OPSIN-parse verified). N-substituted =N-R fails closed.

HEAD reproduce-first divergence: the plan located this in
partial_saturation.py::name_cyclic_oxo_compound, but that namer returns
None for this molecule. The real path is the general_acyclic /
fallback_chain_ring pool candidate, whose FG-only ring-substituent
detector composer.py::_detect_fg_only_prefix returned 'amino' for ANY
single-atom nitrogen without checking bond order (unlike the parallel
oxygen branch, which distinguishes oxo/hydroxy). The =NH was silently
turned into 'amino' (-NH2), a DIFFERENT molecule, so suppressed it.
Fix: mirror the oxygen branch — double-bonded degree-1 neutral N -> 'imino';
substituted =N-R -> None (fail closed).
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestRingIminoPrefix:
    def test_quinone_imine(self):
        assert name_compound("O=C1C=CC(=N)C=C1", style="pin") \
            == "4-iminocyclohexa-2,5-dien-1-one"

    def test_methyl_dienone_protect(self):
        assert name_compound("O=C1C=CC(C)C=C1", style="pin") \
            == "4-methylcyclohexa-2,5-dien-1-one"

    def test_n_substituted_imine_fails_closed(self):
        # =N-CH3 on the ring is NOT expressible as bare 'imino' —
        # must not emit a structure-dropping name.
        n = name_compound("O=C1C=CC(=NC)C=C1", style="pin")
        assert "iminocyclohexa" not in n
