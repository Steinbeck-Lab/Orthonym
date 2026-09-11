"""W8-P11 NEEDS_EXAMPLE verification sweep.

Pins the ledger rows that were reproduced live at HEAD and confirmed to
already emit the PIN (Clusters 1-3 of
docs/the workflow tooling/plans/2026-07-16-wave8-p11-needs-example-sweep.md).
These are verification/regression-lock tests, not red->green TDD: the
rows were confirmed WORKING before this file was written. See that plan
doc + internal notes for the full row-by-row
disposition of the 128-row P11 scope (most rows are reclassified OPEN and
routed to an owning phase, not gold-worthy).
"""
import pytest

from orthonym.namer import Orthonym

namer = Orthonym()


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("[C@H](F)(Cl)Br", "(S)-bromo(chloro)(fluoro)methane"),  # Sequence Rules -> S
    ("C[C@@H](O)CC", "(2R)-butan-2-ol"),                       # Rule 1a -> 2R
    ("CC=C1CC(=CC)C1", "1,3-diethylidenecyclobutane"),         #: symmetric -> NO E/Z descriptor
])
def test_p92_cip_delegated(smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("O=C(Cl)C1CCCCC1", "cyclohexanecarbonyl chloride"),           #
    ("C1CCCCC1Cc1ccccc1", "(cyclohexylmethyl)benzene"),            # /.10
    ("O=C(C1CCCCC1)c1ccccc1", "cyclohexyl(phenyl)methanone"),      # parens
    ("C(Oc1ccccc1)Oc1ccccc1", "1,1'-[methylenebis(oxy)]dibenzene"),# /.5.1.10
])
def test_p16_enclosing_marks(smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.unit
def test_p23_secondary_bridge_locants():
    #: secondary-bridge superscript locants as an ascending set
    assert namer.name("C1C2CC3CC1C23") == "tricyclo[3.1.1.0^3,7]heptane"


@pytest.mark.unit
def test_p23_5_1_siloxane_pentaoxa_tetrasila():
    #: heterogeneous alternating-heteroatom von Baeyer with Si.
    # Bonus finding: the ledger recorded this as a live Si-drop leak
    # ("3,5,7,9,10-pentaoxa-bicyclo[4.3.1]decane", Si silently dropped),
    # but reproducing at current HEAD shows the Group-14/15 heteroatom
    # prefixes (a phase) already fixed it -- Si is correctly cited
    # via 'sila', gated == raw, OPSIN RT-clean. No longer a live leak.
    smiles = "O1[SiH2]O[SiH]2O[SiH2]O[SiH]1O2"
    expected = "2,4,6,8,9-pentaoxa-1,3,5,7-tetrasilabicyclo[3.3.1]nonane"
    assert namer.name(smiles) == expected
    namer_raw = Orthonym(_disable_opsin_validity_gate=True)
    assert namer_raw.name(smiles) == expected


@pytest.mark.unit
def test_p93_5_7_3_ring_assembly_stereo_no_longer_wrong():
    #: E/Z on unsaturated ring assemblies. Reproducing this row
    # surfaced a live wrong-name leak in ring_assemblies._system_signature
    # (fixed in the same commit as this test): a cyclooctenyl-cyclooctane
    # pair (different saturation) falsely compared identical and was named
    # "1,1'-bi(cyclooctene)" -- a wrong structure (only one ring actually
    # has the double bond). Post-fix, detect_ring_assembly correctly
    # declines the pair and a different handler names the real molecule.
    # This is NOT yet a full ring-assembly-stereo PIN engine
    # (that stays routed to a phase) -- it only confirms the leak is gone.
    smiles = r"C1CCCCC/C=C\1C1CCCCCCC1"
    expected = "(1E)-1-cyclooctylcyclooct-1-ene"
    namer_raw = Orthonym(_disable_opsin_validity_gate=True)
    assert namer.name(smiles) == expected
    assert namer_raw.name(smiles) == expected


@pytest.mark.unit
def test_p93_6_ex6_forbidden_multiplicative_fails_closed():
    # Ex6: multiplicative name forbidden when substituents differ in
    # stereo descriptor. Fixed in commit f5f891f4 (immediately prior to this
    # sweep): _ring_sub_group_stereo_veto declines the merge instead of
    # emitting a wrong stereoisomer. Verification-lock: confirms the veto
    # still fails closed (gated == raw == "unknown organic compound"), not a
    # new build -- the full per-substituent stereo-bracket PIN stays routed
    # to a phase.
    smiles = "C[C@H]1CC[C@@H](C[C@@H](O)C[C@@H]2CC[C@H](C)CC2)CC1"
    namer_raw = Orthonym(_disable_opsin_validity_gate=True)
    assert namer.name(smiles) == "unknown organic compound"
    assert namer_raw.name(smiles) == "unknown organic compound"
