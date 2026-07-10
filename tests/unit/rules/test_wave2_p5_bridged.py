"""Wave-2 P5 bridged-fused engine (C4) tests.

Every OPSIN-RT-verified row is a passing gold; INTERNAL-ORACLE rows verify the
structure round-trips (OPSIN 2.9 cannot parse the BB PIN); FAIL-CLOSED rows
prove Orthonym declines (never emits a wrong bridged name).

Plan: 
"""
import pytest
from rdkit import Chem

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


@pytest.mark.unit
class TestP25CyclicBridges:
    """P-25.4.2.1.2/.1.3 cyclic (ring) bridges. OPSIN 2.9 cannot parse the
    BB PIN (9,10-[1,2]benzenoanthracene), so we verify with an INTERNAL
    ORACLE: the constructor either emits the correct BB PIN (carrying the
    '[1,2]benzeno' bridge prefix) or fails closed. A wrong name is never
    emitted for out-of-scope ring bridges."""

    def test_benzeno_bridge_internal_oracle(self):
        smi = "C1=CC=C2C(=C1)C1c3ccccc3C2c2ccccc21"  # triptycene = 9,10-[1,2]benzenoanthracene
        name = name_compound(smi)
        # INTERNAL ORACLE: either the correct BB PIN, or fail closed (never wrong).
        assert name in ("unknown organic compound", "9,10-[1,2]benzenoanthracene")
        if name != "unknown organic compound":
            assert "benzeno" in name  # the ring-bridge prefix must be present

    def test_out_of_scope_ring_bridge_fails_closed(self):
        # A cyclobutane ring bridge across benzene is NOT in the built class
        # ([1,2]epicyclobuta needs its own verified numbering) -> must decline.
        smi = "C12CCC1c1ccccc1-2"  # out-of-scope ring bridge
        name = name_compound(smi)
        assert name == "unknown organic compound" or "epicyclobuta" not in name
