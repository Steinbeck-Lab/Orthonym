"""Phase 155.A D-16 cross-handler dispatch contract.

cyclophane vs ring_assembly vs multiplicative are MUTUALLY EXCLUSIVE by
topology after Plan 155-01 ships:

  - >= 2 ring systems joined by acyclic chain >= 2 atoms -> cyclophane
  - single-bond-joined identical rings                   -> ring_assemblies
  - atom/group-bridged identical units (1-atom bridge)   -> multiplicative

Wave 0 scaffold: smoke test that the cyclophane module is importable; six
parametrize edge cases below are skip-marked until Task 3 fills the
implementation.

Source: 155-CONTEXT.md D-03/D-16; tests/integration/test_assembly_vs_multiplicative_dispatch.py
(Phase 154 D-11 pattern).
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.multiplicative import (
    _is_pure_single_bond_assembly,
    name_multiplicative,
)
from orthonym.rules.phane import is_cyclophane, name_cyclophane
from orthonym.rules.ring_assemblies import detect_ring_assembly


def test_phane_module_importable() -> None:
    """Smoke test: the phane public API is importable for dispatch tests."""
    assert callable(is_cyclophane)
    assert callable(name_cyclophane)


@pytest.mark.skip(reason="Phase 155-01 Task 3 fills this")
@pytest.mark.integration
class TestPhaneVsRingAssemblyDispatch:
    """D-16 mutual-exclusion contract -- no double-fire on 6 D-03 edge cases."""

    @pytest.mark.parametrize(
        "smiles,expected_handler",
        [
            ("c1ccc(-c2ccccc2)cc1", "ring_assembly"),  # biphenyl
            ("C1Cc2ccc(cc2)CCc2ccc1cc2", "cyclophane"),  # [2.2]paracyclophane
            ("Nc1ccc(Cc2ccc(N)cc2)cc1", "multiplicative"),  # 4,4'-methylenedianiline
            # Task 3 will add the remaining D-03 cases: [2.2.2]paracyclophane,
            # cyclophane with -O-CH2- linker, two benzenes via -CH2-CH2- bridge.
        ],
    )
    def test_dispatch_routing(self, smiles, expected_handler):
        raise AssertionError("Task 3 replaces this stub.")
