# tests/unit/assembly/test_coverage_audit.py
"""v33 Phase 0 Task L0.1/L0.2: producer-agnostic coverage AUDIT (SHADOW).

Two carrier-specific proofs feed ONE CoverageVerdict:
  * GeneralEngineResult (has .bindings) -> certify_general_result (E1 + spine).
  * bare str (PIN handlers) -> the SELF-01 verdict already computed at
    `_final_opsin_validity_gate` (CARRIED RULING, see task-L0-brief.md), falling
    back to a fresh OPSIN re-anchor (validate_atom_coverage) only when no
    SELF-01 result is available for this name.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.coverage_audit import audit_coverage, CoverageVerdict

pytestmark = pytest.mark.unit


def _general_result(smiles):
    """Build a real GeneralEngineResult exactly as the GENERAL block does
    (mirrors tests/unit/assembly/test_general_engine.py's `_features` helper —
    no free-function `perceive_features` exists; perception is the `_perceive`
    instance method)."""
    from orthonym.namer import Orthonym
    from orthonym.assembly.general_engine import name_general
    nm = Orthonym(_disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    feats = nm._perceive(mol, smiles, canonical)
    nm._classify(feats)
    result = name_general(mol, feats, allow_aromatic_general=True,
                           allow_suffix_free=True)
    return mol, result


class TestGeneralEngineResultPath:
    def test_general_engine_result_complete_is_certified(self):
        mol, result = _general_result("CN(C)CC1CCCCC1O")
        assert result is not None
        v = audit_coverage(mol, result.name, result)
        assert isinstance(v, CoverageVerdict)
        assert v.complete is True
        assert v.method == "e1_spine"


class TestBareStrReanchorPath:
    def test_bare_str_correct_name_is_complete(self):
        mol = Chem.MolFromSmiles("CCO")
        v = audit_coverage(mol, "ethanol", None)
        assert v.complete is True
        assert v.method == "reanchor"

    def test_bare_str_atom_dropping_name_is_incomplete(self):
        # trioxaundecane skeleton deliberately named "ethanol" (wrong/short) ->
        # the OPSIN re-anchor must catch the dropped atoms.
        mol = Chem.MolFromSmiles("CCOCCOCCOCC")
        v = audit_coverage(mol, "ethanol", None)
        assert v.complete is False
        assert v.method == "reanchor"


class TestBareStrSelf01Reuse:
    """CARRIED RULING: when a SELF-01 verdict is supplied, use it and do NOT
    call OPSIN a second time (no `validate_atom_coverage` fallback)."""

    def test_self01_complete_true_is_used_verbatim(self):
        mol = Chem.MolFromSmiles("CCO")
        v = audit_coverage(mol, "ethanol", None, self01_complete=True)
        assert v == CoverageVerdict(True, "self01", "")

    def test_self01_complete_false_is_used_verbatim(self):
        mol = Chem.MolFromSmiles("CCO")
        v = audit_coverage(mol, "ethanol", None, self01_complete=False)
        assert v.complete is False
        assert v.method == "self01"
