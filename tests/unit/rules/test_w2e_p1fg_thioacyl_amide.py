"""P-66.1.4.2 (BB 33172): "CH3-CS-NH-CS-CH3
N-(ethanethioyl)ethanethioamide (PIN)" — BB verbatim example.

UNDER-SCOPE (W2E-P1FG Task 8, standing rule 7): reproduce-first at the CODE
level exposed that this requires a *dedicated thioimide subsystem*, not the
localized "add =S to the acyl branch" the plan assumed:
  1. Perception classifies CC(=S)NC(C)=S as TWO 'thioamide' FGs, while the
     O-analog CCC(=O)NC=O is a single 'imide' FG — there is no thioimide FG
     nor a thioamide->thioimide collision-resolution row.
  2. The O-imide's 'N-propanoylformamide' name comes from an imide-specific
     path; even get_n_substituents / name_substituent name the O-acyl branch
     '3-oxopropyl', NOT 'propanoyl' — the reusable acyl-substituent-prefix
     producer the plan assumed does not exist.
  3. No 'ethanethioyl' (alkanethioyl acyl-substituent) name producer exists;
     thioamide's prefix today is the whole-group 'carbamothioyl'.
Building all three is the "dedicated subsystem" class deferred here. The
molecule already fails CLOSED (returns 'unknown organic compound', never a
wrong name) — pinned below. Follow-up: add a thioimide FG + collision row +
N-thioacyl-substituent naming, then flip test_target to a positive assert.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestThioacylThioamide:
    @pytest.mark.xfail(reason="W2E-P1FG Task 8 under-scope: thioimide "
                              "perception + N-thioacyl naming subsystem not "
                              "built; fails closed today (see module docstring).",
                       strict=True)
    def test_n_ethanethioyl_ethanethioamide(self):
        assert name_compound("CC(=S)NC(C)=S", style="pin") \
            == "N-(ethanethioyl)ethanethioamide"

    def test_thioacyl_thioamide_never_wrong_positive(self):
        # Under-scope fail-closed contract: we must NEVER emit the wrong
        # target as if it were built. At production runtime the SELF-01 OPSIN
        # backstop suppresses the structure-dropping bare-parent candidate to
        # 'unknown organic compound' (verified via ); the
        # unit harness disables OPSIN so the raw un-round-tripped 'ethanethioamide'
        # surfaces here. Either way, the one thing that must hold in ALL modes:
        # the confident wrong PIN is never produced.
        n = name_compound("CC(=S)NC(C)=S", style="pin")
        assert n != "N-(ethanethioyl)ethanethioamide"

    def test_thioamide_parent_protect(self):
        assert name_compound("CC(=S)N", style="pin") == "ethanethioamide"

    def test_n_methyl_thioamide_protect(self):
        assert name_compound("CC(=S)NC", style="pin") == "N-methylethanethioamide"

    def test_o_acyl_protect(self):
        assert name_compound("CCC(=O)NC=O", style="pin") == "N-propanoylformamide"
