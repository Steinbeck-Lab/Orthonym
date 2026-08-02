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
        # v29 Phase C tranche A: MONOsubstituted homogeneous monocycles omit the
        # locant '1' per P-14.3.4.2(c) (``BlueBookV2.md:2913``), worked verbatim as
        # ``cyclohexanethiol`` at ``:2917``, ``cyclopentanone`` at ``:28394`` and
        # ``'cyclohexanone' (PIN)`` at ``:14916``. Updated from the ``-1-`` forms,
        # which that licence makes non-PINs.
        ("O=C1CCCCC1", "cyclohexanone"),
        ("OC1CCCCC1", "cyclohexanol"),
        ("NC1CCCCC1", "cyclohexanamine"),
        # ★ THE BOUNDARY, and why the rows above are not a blanket strip: add ANY
        # second substituent and the ring is no longer monosubstituted, so
        # P-14.3.3's deny-default restores every locant. These two must NOT change.
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
        # v29 Task A: was "caprolactam" -- a non-PIN trivial name (0 BlueBookV2.md
        # hits) now withdrawn from the PIN path. P-64.3.1 (BB:29314) makes cyclic
        # amides pseudoketones and its own example prints `azepan-2-one (PIN)`
        # (BB:29323). Note the two rows above already assert `...azepan-2-one` as
        # the SUBSTITUTED parent, so this row was internally inconsistent.
        ("O=C1CCCCCN1", "azepan-2-one"),
        ("O=C1CCCCN1", "piperidin-2-one"),
        ("C1COCCN1", "morpholine"),
    ])
    def test_direction_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected
