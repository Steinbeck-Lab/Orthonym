"""W2E-P1FC Task 11 — P-59.2.1.6 (BB 25207): PG in both chain and ring; the
portion with the GREATER number of the PG is the parent; tie -> ring.
BB Example 2: 4-(2-oxobutyl)cyclopentane-1,2-dione (PIN) — ring has 2 ketones
(dione) vs chain 1 ketone, so ring wins; chain ketone -> 2-oxobutyl prefix.

STATUS (2026-07-09): DEFERRED to strict-xfail. At HEAD the fallback_chain_ring
handler ABSORBS the pendant chain ketone (the 2-oxobutyl) as a THIRD ring oxo
and drops the ethyl, producing 'cyclopentane-1,2,4-trione' (a different
molecule). In production this is caught by SELF-01 (OPSIN re-perception) and the
molecule fails closed (never a wrong name). The correct build — chain-vs-ring
PG-count parent selection (ring 2 diones beats chain 1 ketone) plus expressing
the chain ketone as the '2-oxobutyl' substituent prefix — requires broad
parent-selection changes with determinism risk, so it is a documented follow-up
(). NEVER ship the trione absorption.
The working saturated-dione precedent (4-methylcyclopentane-1,2-dione) is pinned
below so the base capability cannot regress.
"""
import pytest

from orthonym.namer import name_compound

# OPSIN-canonical evidence SMILES for '4-(2-oxobutyl)cyclopentane-1,2-dione'.
OXOBUTYL_CYCLOPENTANEDIONE = "O=C(CC1CC(C(C1)=O)=O)CC"


@pytest.mark.unit
class TestP59216ChainVsRingPG:
    def test_saturated_dione_precedent(self):
        # Working base capability — must not regress.
        assert name_compound("O=C1CC(C)CC1=O") == "4-methylcyclopentane-1,2-dione"
        assert name_compound("O=C1C(=O)CCC1") == "cyclopentane-1,2-dione"

    def test_oxobutyl_cyclopentanedione_target(self):
        assert name_compound(OXOBUTYL_CYCLOPENTANEDIONE) == \
            "4-(2-oxobutyl)cyclopentane-1,2-dione"

    def test_oxobutyl_determinism_random_spellings(self):
        # Determinism guard: the P-59.2.1.6 off-ring-PG demotion must be
        # spelling-independent (parent/suffix split keyed on canonical ring
        # membership, not atom/registration order). Name 8 random SMILES
        # renderings and assert one identical PIN.
        from rdkit import Chem

        mol = Chem.MolFromSmiles(OXOBUTYL_CYCLOPENTANEDIONE)
        names = {
            name_compound(Chem.MolToSmiles(mol, doRandom=True), style="pin")
            for _ in range(8)
        }
        assert names == {"4-(2-oxobutyl)cyclopentane-1,2-dione"}
