"""Wave2 T2c — ring-parent suffix locant from the FG CENTER atom (P-64.7.1).

The polyfunctional ring path derived the suffix locant from the FIRST match
atom present in the locant map. The ketone SMARTS match is
(C_neighbor, C=O, O, C_neighbor), so match[0] read the NEIGHBOR carbon's
locant — emitting the impossible '2-aminocyclohexan-2-one' (amino and oxo
sharing one carbon = a 5-bond carbon; OPSIN-invalid, gate-suppressed to
unknown). Fix: _find_fg_center_atom picks the carbonyl carbon.

Reproduce-first note: the OTHER planned 2c item (single-heteroatom ring
numbering direction) was proven a NON-defect at HEAD — 2-methylthiolane /
3-methyloxolane / 3-aminoazepan-2-one / 1-selenacyclotridecan-3-one all
number correctly; ledger rows 1519/1535 carried evidence SMILES that do not
match their expected names (stale claims, overridden).
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
class TestRingKetoneSuffixLocant:
    """Alpha-substituted cycloalkanones: suffix locant = carbonyl carbon."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NC1CCCCC1=O", "2-aminocyclohexan-1-one"),
        ("OC1CCCCC1=O", "2-hydroxycyclohexan-1-one"),
        ("NC1CCCC1=O", "2-aminocyclopentan-1-one"),
        ("OC1CCCCCC1=O", "2-hydroxycycloheptan-1-one"),
    ])
    def test_alpha_substituted_cycloalkanones(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O=C1CCCCC1", "cyclohexan-1-one"),
        ("OC1CCCCC1", "cyclohexan-1-ol"),
        ("NC1CCCCC1", "cyclohexan-1-amine"),
        ("OC1CCC(N)CC1", "4-aminocyclohexan-1-ol"),
        ("O=C1CCC(C)CC1", "4-methylcyclohexan-1-one"),
    ])
    def test_ring_controls(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestSingleHeteroatomDirectionNonDefect:
    """Reproduce-first: the planned direction fix targets were already
    correct at HEAD — these pins protect that state."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CC1CCCS1", "2-methylthiolane"),
        ("CC1CCOC1", "3-methyloxolane"),
        ("O=C1C(N)CCCCN1", "3-aminoazepan-2-one"),
        ("NC1CCCCC(=O)N1", "7-aminoazepan-2-one"),  # ledger 1535's SMILES:
        # the amino-C is ADJACENT to the ring N, so 7- is the correct PIN
        # (the '3-amino' expectation belonged to a different structure)
        ("O=C1C[Se]CCCCCCCCCC1", "1-selenacyclotridecan-3-one"),
        ("O=C1CCCCCN1", "caprolactam"),
        ("O=C1CCCCN1", "piperidin-2-one"),
        ("C1COCCN1", "morpholine"),
    ])
    def test_direction_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected
