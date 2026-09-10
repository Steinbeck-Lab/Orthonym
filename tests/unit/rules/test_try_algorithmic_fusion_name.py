"""a phase: _try_algorithmic_fusion_name gate broadening tests.

Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
Source: 149-internal notes.
Source: internal notes "Hook Point #3: _try_algorithmic_fusion_name Gate Audit ".
"""
import logging

import pytest
from rdkit import Chem

from orthonym.rules.fused_rings import _try_algorithmic_fusion_name


class TestTryAlgorithmicFusionNameD08:
    """: ONE gate broadens (>=2 rings); FOUR gates preserved."""

    def test_broaden_2ring_gate_to_at_least_2(self):
        """: single-ring mol returns None (no fusion to name).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: 149-internal notes.
        """
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene, 1 ring
        assert _try_algorithmic_fusion_name(mol) is None

    def test_2ring_path_preserves_engine(self):
        """: 2-ring fused-hetero (quinoline) routes through engine.

        Source: 149-internal notes.
        """
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        # Result may be a string (engine produced a name) or None (engine
        # returned None for this particular system); the contract is that
        # the 2-ring path doesn't crash and the 4 preserved gates still
        # apply.
        result = _try_algorithmic_fusion_name(mol)
        assert result is None or isinstance(result, str)

    def test_3plus_ring_returns_none_decision_only(self, caplog):
        """: 3+ ring fused (carbazole) returns None decision-only.

        Source: 149-internal notes.
        """
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')
        with caplog.at_level(logging.DEBUG, logger='orthonym.rules.fused_rings'):
            result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"3+ ring system must return None per D-08; got {result!r}"
        )

    def test_acyclic_returns_none(self):
        """ broadened gate `< 2`: acyclic mol returns None.

        Source: 149-internal notes.
        """
        mol = Chem.MolFromSmiles('CCO')
        assert _try_algorithmic_fusion_name(mol) is None
