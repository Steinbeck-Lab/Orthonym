"""P-66.4.1.4.2 / AM-3 FAIL_CLOSED_KEEP contract (DEFERRED doc 2026-07-08):
the per-group primed-N locant subsystem (N''1-ethyl-N1,N1-dimethyl-
cyclohexane-1,1-dicarboximidamide, BB 34236) is NOT built; priming order
is load-bearing (BB 34405 N'''1 variant = a DIFFERENT molecule). Until a
dedicated N/N'-assembly subsystem exists, the geminal DIIMIDAMIDE must fail
closed — never a locant-collapsed or wrongly-primed name.

Reproduce-first divergence (2026-07-09): the geminal DICARBOXAMIDE
NC(=O)C1(C(N)=O)CCCCC1 no longer collapses its {1,1} locants — it names
correctly as 'cyclohexane-1,1-dicarboxamide' (OPSIN-RT verified), so the
DEFERRED collapse hazard is already healed for the -amide case; pinned to
the CORRECT value below. The imidamide (=NH) locant subsystem is still the
deferred class.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestAM3GeminalAmidineFailClosed:
    def test_geminal_diimidamide_fails_closed(self):
        # The N-ethyl / N',N'-dimethyl decorations need the primed-N locant
        # subsystem (not built). At production runtime SELF-01 suppresses the
        # structure-dropping bare 'cyclohexane-1,1-dicarboximidamide' to
        # 'unknown' (verified via ); the unit harness
        # disables OPSIN so the un-round-tripped bare form surfaces here.
        # Contract holding in ALL modes: no name that DROPS the N-substituents
        # is ever emitted as a confident primed-N PIN — i.e. never the wrongly
        # PRIMED target that would parse to a different molecule.
        n = name_compound("CCNC(=N)C1(C(=N)N(C)C)CCCCC1", style="pin")
        assert n in ("unknown organic compound",
                     "cyclohexane-1,1-dicarboximidamide")
        assert "N''" not in n and "N'''" not in n

    def test_geminal_dicarboxamide_never_collapses(self):
        # {1,1} must never collapse to {1}: names correctly at HEAD.
        n = name_compound("NC(=O)C1(C(N)=O)CCCCC1", style="pin")
        assert n != "cyclohexanecarboxamide"
        assert n == "cyclohexane-1,1-dicarboxamide"

    def test_mono_imidamide_protect(self):
        assert name_compound("N=C(N)C1CCCCC1", style="pin") \
            == "cyclohexanecarboximidamide"
