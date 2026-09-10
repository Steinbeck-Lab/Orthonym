""" phane-vs-ring-assembly boundary (Wave-2 P2 Task 5).

BB (the Blue Book): "Phane names are preferred IUPAC names
rather than ring assembly names when seven or more rings or ring systems are
present." Orthonym does NOT build the linear-phane amplification engine
, >=7 nodes) in Wave-2 P2 — this guard fails closed so no WRONG phane
name is ever emitted. It also asserts the existing cyclophane API is
untouched.
"""
import pytest
from rdkit import Chem

from orthonym.rules.phane import (
    linear_phane_scope_guard,
    is_seven_plus_ring_assembly,
    is_cyclophane,       # existing API must remain importable/intact
    name_cyclophane,
)

# 7 benzene rings in a linear single-bond assembly (septiphenyl).
SEPTIPHENYL = "c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccccc7)cc6)cc5)cc4)cc3)cc2)cc1"
BIPHENYL = "c1ccc(-c2ccccc2)cc1"


class TestLinearPhaneScopeGuard:
    def test_guard_always_declines(self):
        # Fail-closed contract: the guard NEVER emits a name (engine unbuilt).
        for smi in [SEPTIPHENYL, BIPHENYL, "CCO", "c1ccccc1"]:
            mol = Chem.MolFromSmiles(smi)
            assert linear_phane_scope_guard(mol) is None

    def test_guard_none_input(self):
        assert linear_phane_scope_guard(None) is None


class TestSevenPlusRingAssemblyDetection:
    def test_detects_septiphenyl(self):
        assert is_seven_plus_ring_assembly(Chem.MolFromSmiles(SEPTIPHENYL)) is True

    def test_rejects_biphenyl(self):
        assert is_seven_plus_ring_assembly(Chem.MolFromSmiles(BIPHENYL)) is False

    def test_rejects_none(self):
        assert is_seven_plus_ring_assembly(None) is False


class TestExistingCyclophaneApiIntact:
    def test_cyclophane_helpers_still_callable(self):
        assert callable(is_cyclophane)
        assert callable(name_cyclophane)
        # A plain benzene is not a cyclophane (regression guard on the gate).
        assert is_cyclophane(Chem.MolFromSmiles("c1ccccc1")) is False
