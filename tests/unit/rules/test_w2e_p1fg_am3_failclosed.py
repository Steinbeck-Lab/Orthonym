""" / geminal-diimidamide contract.

W2E-D4 (2026-07-10): the per-group primed-N superscript locant subsystem
(N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide, BB 34236) is
now BUILT — composer._assemble_geminal_dicarboximidamide_name. Priming
order is load-bearing (BB 34405 N'''1 variant = a DIFFERENT molecule), so
the contract now asserts the EXACT correctly-primed PIN (previously this was
a FAIL_CLOSED_KEEP asserting the subsystem was unbuilt).

The geminal DICARBOXAMIDE NC(=O)C1(C(N)=O)CCCCC1 also names correctly as
'cyclohexane-1,1-dicarboxamide' (the {1,1} locants never collapse).
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestAM3GeminalAmidineFailClosed:
    def test_geminal_diimidamide_named(self):
        # W2E-D4 BUILT: the N-ethyl (imino, group-2 -> N'') / N,N-dimethyl
        # (amino, group-1 -> N) decorations get per-group primed+superscript
        # italic-N locants (lowest-locant group assignment +
        # alphanumerical citation. OPSIN-RT verified.
        n = name_compound("CCNC(=N)C1(C(=N)N(C)C)CCCCC1", style="pin")
        assert n == "N''1-ethyl-N1,N1-dimethylcyclohexane-1,1-dicarboximidamide"

    def test_geminal_dicarboxamide_never_collapses(self):
        # {1,1} must never collapse to {1}: names correctly at HEAD.
        n = name_compound("NC(=O)C1(C(N)=O)CCCCC1", style="pin")
        assert n != "cyclohexanecarboxamide"
        assert n == "cyclohexane-1,1-dicarboxamide"

    def test_mono_imidamide_protect(self):
        assert name_compound("N=C(N)C1CCCCC1", style="pin") \
            == "cyclohexanecarboximidamide"
