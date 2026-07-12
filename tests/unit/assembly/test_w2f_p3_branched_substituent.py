"""W2F-P3 item 15 (P-59.2.1.8 / P-29.2 / P-16.3.3): branched acyclic FG-bearing
substituent on a ring principal-characteristic-group parent. Task 3.

Expected PINs OPSIN-verified (opsin-cli-2.9.0 -> RDKit canonical == input) at
plan-authoring time; see  §A/§D.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.substituent_naming import _name_polyfunctional_acyclic_substituent

pytestmark = pytest.mark.unit


@pytest.fixture
def gated(monkeypatch):
    """Re-enable the production OPSIN-validity + SELF-01 gate for the end-to-end
    fail-closed tripwire. The autouse conftest fixture disables the gate by default,
    so an out-of-envelope molecule surfaces its pre-suppression wrong name (a bare
    'cyano' leak) instead of 'unknown'; the v1 envelope declines and the production
    gate turns that into 'unknown organic compound'. Mirrors
    tests/unit/namer/test_self_consistency_gate.py; skips if the OPSIN jar is
    unavailable (portable)."""
    import orthonym.namer as _nm
    if not _nm._validity_gate_jar_present():
        pytest.skip("OPSIN jar unavailable for gate-inclusive fail-closed test")
    monkeypatch.setattr(_nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(_nm, "_SC_MODE", "on", raising=False)
    yield


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


class TestItem15BranchedSubstituent:
    def test_evidence_dichloro_hexyl(self):
        assert _name("ClC1CC(C(CC1Cl)C(=O)O)CC(CC(C(C)=O)Cl)CO") == \
            "4,5-dichloro-2-[4-chloro-2-(hydroxymethyl)-5-oxohexyl]cyclohexane-1-carboxylic acid"

    def test_minimal_hydroxymethyl_oxo_pentyl(self):
        assert _name("OC(=O)C1CCCCC1CC(CO)CC(C)=O") == \
            "2-[2-(hydroxymethyl)-4-oxopentyl]cyclohexane-1-carboxylic acid"

    # --- protect: linear branch path stays byte-identical ---
    def test_protect_linear_oxobutyl(self):
        assert _name("OC(=O)C1CCCCC1CCC(C)=O") == \
            "2-(3-oxobutyl)cyclohexane-1-carboxylic acid"

    # --- fail-closed: nitrile branch is outside the v1 recognised-FG set ---
    @pytest.mark.usefixtures("gated")
    def test_failclosed_nitrile_branch(self):
        assert _name("OC(=O)C1CCCCC1CC(C#N)CC(C)=O") == "unknown organic compound"

    # --- unit-level: the substituent namer itself ---
    def test_substituent_namer_branched(self):
        m = Chem.MolFromSmiles("OC(=O)C1CCCCC1CC(CO)CC(C)=O")
        ring = set(a for r in m.GetRingInfo().AtomRings() for a in r)
        # branch = the acyclic chain attached at atom 9 (see reproduce-first)
        fa = [9, 10, 11, 12, 13, 14, 15, 16]
        assert _name_polyfunctional_acyclic_substituent(m, fa, 9, ring) == \
            "2-(hydroxymethyl)-4-oxopentyl"

    def test_substituent_namer_linear_unchanged(self):
        m = Chem.MolFromSmiles("OC(=O)C1CCCCC1CCC(C)=O")
        ring = set(a for r in m.GetRingInfo().AtomRings() for a in r)
        fa = [9, 10, 11, 12, 13]
        assert _name_polyfunctional_acyclic_substituent(m, fa, 9, ring) == "3-oxobutyl"


class TestItem15Determinism:
    @pytest.mark.parametrize("_i", range(12))
    def test_evidence_spelling_invariant(self, _i):
        import random
        smi = "ClC1CC(C(CC1Cl)C(=O)O)CC(CC(C(C)=O)Cl)CO"
        m = Chem.MolFromSmiles(smi)
        order = list(range(m.GetNumAtoms()))
        random.Random(_i).shuffle(order)
        rand_smi = Chem.MolToSmiles(Chem.RenumberAtoms(m, order), canonical=False)
        assert _name(rand_smi) == \
            "4,5-dichloro-2-[4-chloro-2-(hydroxymethyl)-5-oxohexyl]cyclohexane-1-carboxylic acid"
