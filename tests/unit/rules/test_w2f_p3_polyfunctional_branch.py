"""W2F-P3 item 8 (P-45.5-corroborating / P-45.2.1 / P-29.2): branch-resident nitro
naming + fail-closed count guard. Tasks 1 (defect a + b-het) and 2 (defect b-pure-C).

All expected PINs OPSIN-verified (opsin-cli-2.9.0 -> RDKit canonical == input) at
plan-authoring time; see internal notes §A/§D.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound

pytestmark = pytest.mark.unit


@pytest.fixture
def gated(monkeypatch):
    """Re-enable the production OPSIN-validity + SELF-01 gate for the
    end-to-end fail-closed tripwire. The autouse conftest fixture disables the
    gate by default (for speed), so an out-of-envelope molecule surfaces its
    pre-suppression wrong name (a bare un-locanted 'thiocyanato' leak) instead
    of 'unknown'; the fail-closed count guard makes name_polyfunctional return
    None and the production gate turns that into 'unknown organic compound'.
    Mirrors tests/unit/namer/test_self_consistency_gate.py; skips if the OPSIN
    jar is unavailable (portable)."""
    import orthonym.namer as _nm
    if not _nm._validity_gate_jar_present():
        pytest.skip("OPSIN jar unavailable for gate-inclusive fail-closed test")
    monkeypatch.setattr(_nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(_nm, "_SC_MODE", "on", raising=False)
    yield


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


class TestItem8BranchNitro:
    # --- Task 1: defect (a) + (b)-het heals ---
    def test_evidence_dinitropropyl_branch(self):
        # off-chain dinitro branch -> owned by the branch name; F-branch is the
        # P-45.2.1 parent chain (max prefixes). Was raw
        # '6,7-difluoro-5-methyldinitro-4-propylheptanoic acid' -> unknown.
        assert _name("OC(=O)CCC(C(C)C(F)CF)C([N+](=O)[O-])C([N+](=O)[O-])C") == \
            "4-(1,2-dinitropropyl)-6,7-difluoro-5-methylheptanoic acid"

    def test_mixed_on_and_off_chain_nitro(self):
        # one nitro on-chain (C2), one on a -CH2NO2 branch (nitromethyl at C3).
        # Was raw '3-methyl-2-nitropentanoic acid' (off-chain nitro dropped) -> unknown.
        assert _name("OC(=O)C([N+](=O)[O-])C(C[N+](=O)[O-])CC") == \
            "2-nitro-3-(nitromethyl)pentanoic acid"

    # --- protects (must stay byte-identical) ---
    def test_protect_on_chain_dinitro(self):
        assert _name("CC([N+](=O)[O-])C([N+](=O)[O-])CCCC(=O)O") == \
            "5,6-dinitroheptanoic acid"

    def test_protect_difluoropropyl_mirror(self):
        # the on-chain-nitro MIRROR (existing gold W2E-P0CF-01): nitros on-chain,
        # branch is halo -> untouched by edits 1/3a/3b.
        assert _name("OC(=O)CCC(C(F)C(F)C)C([N+](=O)[O-])C([N+](=O)[O-])C") == \
            "4-(1,2-difluoropropyl)-5,6-dinitroheptanoic acid"

    # --- fail-closed tripwire (count guard load-bearing beyond nitro) ---
    @pytest.mark.usefixtures("gated")
    def test_failclosed_thiocyanatomethyl_branch(self):
        # 'thiocyanato' is not in BRANCH_HANDLED_FGS and the branch is not cleanly
        # nameable -> the count guard refuses -> molecule fails closed. Gate-inclusive:
        # the fail-closed end state is produced by the production validity backstop
        # (name_polyfunctional returns None; a downstream handler would otherwise leak
        # a bare un-locanted 'thiocyanato'), so the production gate must be active.
        assert _name("OC(=O)CC(CSC#N)CC([N+](=O)[O-])C") == "unknown organic compound"

class TestItem8PureCBranch:
    # --- Task 2: defect (b)-pure-C heal ---
    def test_pure_c_branched_isopropyl(self):
        # -CH(CH3)2 branch must be 'propan-2-yl', not 'propyl' (carbon-count).
        # Was raw '6-nitro-4-propylheptanoic acid' -> unknown.
        assert _name("OC(=O)CCC(C(C)C)CC([N+](=O)[O-])C") == \
            "6-nitro-4-(propan-2-yl)heptanoic acid"
