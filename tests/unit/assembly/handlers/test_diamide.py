"""D3 unit tests for the ``chain_diamide`` handler.

Mixed primary + N-substituted acyclic diamides (and symmetric variants)
that previously returned 'unknown organic compound' because the
primary_amide + secondary_amide combination was mis-counted as
polyfunctional (double-counting the terminal secondary amide) or the
N-alkyl carbons were swept into pg_atom_set  and dropped.

Fix: dedicated ``name_chain_diamide`` handler dispatched at inner_dispatch
priority 1490 (just before ester_family), naming both amide ends with
numeric-N locants per BB.

IUPAC cite: (acyclic diamide parent) /
(N{locant} substituent prefixes).
"""
from __future__ import annotations

import inspect

import pytest

from orthonym import Orthonym
from orthonym.assembly.handlers.diamide import (
    _is_chain_diamide, name_chain_diamide_handler,
)
from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE


# (smiles, expected_pin) — all OPSIN-verified in the D3 spec.
ACCEPTANCE = [
    ("O=C(N)CCC(=O)NC", "N1-methylbutanediamide"),
    ("O=C(N)CC(=O)NC", "N1-methylpropanediamide"),
    ("O=C(NC)CCC(=O)NC", "N1,N4-dimethylbutanediamide"),
]

# Guards that MUST stay byte-identical / untouched by the new handler.
GUARDS = [
    ("NC(=O)CCC(=O)N", "butanediamide"),        # symmetric primary (byte-identical)
    ("CC(=O)NC", "N-methylacetamide"),           # mono-amide (1 match -> not intercepted)
    ("NC(=O)C(=O)N", "oxamide"),                 # 2-C diamide (caught upstream)
    ("NC(=O)CCCCC(=O)N", "hexanediamide"),       # long-chain symmetric primary
]


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


class TestAcceptance:
    """Mixed primary + N-substituted diamides — the D3 target class."""

    @pytest.mark.parametrize("smiles,expected", ACCEPTANCE)
    def test_acceptance(self, namer, smiles, expected):
        assert namer.name(smiles) == expected


class TestGuards:
    """Regression guards — must remain unchanged after the fix."""

    @pytest.mark.parametrize("smiles,expected", GUARDS)
    def test_guard(self, namer, smiles, expected):
        assert namer.name(smiles) == expected


class TestAdversarial:
    """Adversarial cases that probe the tight predicate."""

    def test_n1_n3_dimethylpropanediamide(self, namer):
        # 3-C chain, both ends N-methyl secondary amides.
        assert namer.name("O=C(NC)CC(=O)NC") == "N1,N3-dimethylpropanediamide"

    def test_diacid_not_intercepted(self, namer):
        # Diacid must NOT be caught by the diamide handler.
        assert namer.name("OC(=O)CCC(=O)O") == "butanedioic acid"


class TestHandlerShape:
    """Handler / predicate signatures + registration."""

    def test_handler_signature(self):
        sig = inspect.signature(name_chain_diamide_handler)
        assert "features" in sig.parameters

    def test_predicate_signature(self):
        sig = inspect.signature(_is_chain_diamide)
        assert "features" in sig.parameters

    def test_predicate_returns_false_on_empty(self):
        class EmptyFeatures:
            principal_group = None
            is_cyclic = False
            is_polyfunctional = False
            chain_is_parent = False
            principal_chain = None
            atom_to_locant = None
            functional_groups = {}
            mol = None
        assert _is_chain_diamide(EmptyFeatures()) is False

    def test_registered_at_1490(self):
        entry = INNER_DISPATCH_TABLE.get("chain_diamide")
        assert entry is not None
        assert entry.priority == 1490
