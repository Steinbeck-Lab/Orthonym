"""Wave-2 P5 bridged-fused engine (C4) tests.

Every OPSIN-RT-verified row is a passing gold; INTERNAL-ORACLE rows verify the
structure round-trips (OPSIN 2.9 cannot parse the BB PIN); FAIL-CLOSED rows
prove Orthonym declines (never emits a wrong bridged name).

Plan: 
"""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP25UnsaturatedAcyclicBridges:
    """P-25.4.2.1.1 — unsaturated acyclic bridges (all OPSIN-RT-verified)."""

    def test_etheno_naphthalene(self):
        # OPSIN: 1,4-ethenonaphthalene -> C12=CC=C(C3=CC=CC=C13)C=C2 (canonical match)
        assert name_compound("C12=CC=C(C3=CC=CC=C13)C=C2") == "1,4-ethenonaphthalene"

    def test_butadieno_naphthalene(self):
        # OPSIN: 1,4-buta[1,3]dienonaphthalene -> C12=CC=C(C3=CC=CC=C13)C=CC=C2
        assert name_compound("C12=CC=C(C3=CC=CC=C13)C=CC=C2") == "1,4-buta[1,3]dienonaphthalene"

    def test_dibenzobarrelene_protect(self):
        # anchor already-correct — lock against regression
        assert name_compound("C1=CC=CC=2C3C4=CC=CC=C4C(C12)C=C3") == \
            "9,10-dihydro-9,10-ethenoanthracene"
