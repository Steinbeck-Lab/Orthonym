"""WS-A.1 S4 — P-44.1.2.2 ring-senior-to-chain in the select_parent PG-tie cascade.

When a ring and a chain both bear the principal characteristic group (PG count
tied) and they are the SAME seniority class (same senior element, both
carbon-based skeletons), the ring is senior to the chain regardless of chain
length / hydrogenation (P-44.1.2.2). This must be applied BEFORE the
P-44.1(c) chain-length comparison.

Tight same-class gate (V21-ALGORITHM-FIX-PLAN.md §3 WS-A.1 step 4):
  - fires ONLY when best_ring senior element == candidate_chain senior element
    AND both are carbon-based;
  - the heteroatom-bridged skeletal-chain path (P-44.3, different element:
    O/Si/ester chains) stays reachable — the ring must NOT win there.

Gold targets (Blue Book exact-topology PINs; among-rings gold rows):
  O=CCCCCCC1CCCCC1C=O  -> 2-(6-oxohexyl)cyclohexane-1-carbaldehyde
  O=CCCCCCCC1CCCC1C=O  -> 2-(7-oxoheptyl)cyclopentane-1-carbaldehyde

Protects (the same-class gate must NOT mis-fire ring-wins):
  C1CCCCC1COCCOCCOCCOC  -> 1-cyclohexyl-2,5,8,11-tetraoxadodecane
                           (heteroatom skeletal chain; different element)
  CCCCCCCc1ccccc1       -> heptylbenzene (no PG; benzene ring parent — ring
                           winning here is correct)
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestRingSeniorToChainGold:
    def test_oxohexyl_cyclohexane_carbaldehyde(self):
        assert (
            name_compound("O=CCCCCCC1CCCCC1C=O").strip()
            == "2-(6-oxohexyl)cyclohexane-1-carbaldehyde"
        )

    def test_oxoheptyl_cyclopentane_carbaldehyde(self):
        assert (
            name_compound("O=CCCCCCCC1CCCC1C=O").strip()
            == "2-(7-oxoheptyl)cyclopentane-1-carbaldehyde"
        )


@pytest.mark.unit
class TestRingSeniorToChainProtect:
    @pytest.mark.xfail(
        reason="Pre-existing P-44.3 different-element defect (already red on "
        "HEAD: names '(2,5,8,11-tetraoxadodecyl)cyclohexane'). S4's same-class "
        "gate must not touch this row either way; the oxa-chain-vs-carbocycle "
        "decision is a Task-9/among-rings-reds root-cause pass.",
        strict=True,
    )
    def test_tetraoxadodecane_keeps_chain_parent(self):
        # Heteroatom (oxa) skeletal chain is a DIFFERENT element class than the
        # carbocycle -> P-44.3 chain path stays reachable, ring must NOT win.
        assert (
            name_compound("C1CCCCC1COCCOCCOCCOC").strip()
            == "1-cyclohexyl-2,5,8,11-tetraoxadodecane"
        )

    def test_heptylbenzene_ring_parent(self):
        assert name_compound("CCCCCCCc1ccccc1").strip() == "heptylbenzene"

    def test_propanoic_acid_chain_parent(self):
        # COOH only on the chain (pg_count_on_chain > on_ring) -> resolved at the
        # PG-count branch before the cascade; S4 must leave it chain-parented.
        assert (
            name_compound("OC(=O)CCC1CCCCC1").strip()
            == "3-cyclohexylpropanoic acid"
        )


@pytest.mark.unit
class TestRingSeniorToChainEsterInvestigate:
    """`methyl 4-cyclohexylbutanoate` — ester PG is on the chain, so the
    PG-count branch SHOULD pick the chain. It is currently mis-named
    `methyl cyclohexanecarboxylate` (among-rings gold MISMATCH), which points
    at ester-routing upstream of this cascade, not the S4 reorder. Captured
    here as a target; resolved during S4 implementation if the reorder
    surfaces it, otherwise tracked as a separate routing defect.
    """

    @pytest.mark.xfail(
        reason="Pre-existing ester-routing defect (already red on HEAD: names "
        "'methyl cyclohexanecarboxylate'). The ester PG is on the chain, so "
        "the PG-count branch should resolve this BEFORE the S4 cascade; the "
        "miss is upstream ester routing, tracked separately.",
        strict=True,
    )
    def test_methyl_cyclohexylbutanoate(self):
        assert (
            name_compound("COC(=O)CCCC1CCCCC1").strip()
            == "methyl 4-cyclohexylbutanoate"
        )


@pytest.mark.unit
class TestRingClosureBondLocant:
    """The ring-closure bond (oriented positions 0 and n-1) is locant n,
    never 1. The naive min(pos)+1 let the cycloalkene orientation comparator
    treat a 1,6-double-bond direction as '1-ene' and emit a name describing
    a different structure (canary rt75_0019, RT True->False under S4).
    """

    def test_wrap_bond_locant_is_n(self):
        from orthonym.rules.cycloalkanes import ring_double_bond_locant
        assert ring_double_bond_locant(0, 5, 6) == 6
        assert ring_double_bond_locant(5, 0, 6) == 6
        assert ring_double_bond_locant(0, 1, 6) == 1
        assert ring_double_bond_locant(4, 5, 6) == 5

    def test_tetrahydroxy_cyclohexene_carbaldehyde_numbering(self):
        # CHO anchor at C1, ene 1(2), OHs 3,4,5,6 — the OH-lower direction
        # (2,3,4,5) is only reachable through the wrap-bond mis-locant and
        # would describe an enol that is NOT this molecule.
        assert (
            name_compound("O=CC1=CC(O)C(O)C(O)C1O").strip()
            == "3,4,5,6-tetrahydroxycyclohex-1-ene-1-carbaldehyde"
        )
