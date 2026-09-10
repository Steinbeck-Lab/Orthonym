"""W8-P6 Cluster E Ex6) fail-closed veto.

A multiplicative ring-substituent collapse (bis/tris/di/tri) silently asserts the
merged occurrences are IDENTICAL. When the merged ring substituents genuinely
differ by a stereodescriptor, that assertion is a WRONG (falsely-symmetric)
stereoisomer -- Ex6 forbids the multiplicative form in that case. Orthonym
cannot yet build the correct per-substituent bracket-internal stereo block
((1r,4S)/(1s,4S)-style), so it must FAIL CLOSED rather than ship the wrong name.

This veto is deliberately narrow (see ``_ring_sub_group_stereo_veto`` in
assembly/composer.py): it fires only when a merged fragment is itself stereogenic
AND the input carried an explicit stereo tag that RDKit's sanitize erased (the
CIP-ceiling signature of the flagship). Genuinely achiral / verifiably-identical
multiplicative names are unaffected.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.mark.unit
def test_cluster_e_stereo_differing_bis_fails_closed():
    # BB Ex1: correct PIN is
    # (2R)-1-[(1r,4S)-4-methylcyclohexyl]-3-[(1s,4S)-4-methylcyclohexyl]propan-2-ol
    # The two 4-methylcyclohexyl substituents DIFFER in stereo, so the previously
    # shipped '(1s,4s)-1,3-bis(4-methylcyclohexyl)propan-2-ol' is a wrong stereoisomer.
    # Until the component-stereo-block engine exists, fail closed (both paths).
    smi = "C[C@H]1CC[C@@H](C[C@@H](O)C[C@@H]2CC[C@H](C)CC2)CC1"
    assert Orthonym().name(smi) == "unknown organic compound"
    assert Orthonym(_disable_opsin_validity_gate=True).name(smi) == "unknown organic compound"


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # Genuinely-identical / achiral multiplicative names MUST still ship (no over-veto).
    ("OC(CC1CCCCC1)CC1CCCCC1", "1,3-dicyclohexylpropan-2-ol"),
    ("Cc1cccc(C)c1", "1,3-dimethylbenzene"),
    ("ClCCCl", "1,2-dichloroethane"),
    ("CC(C)C(C)C", "2,3-dimethylbutane"),
    ("Cc1cc(C)cc(C)c1", "1,3,5-trimethylbenzene"),
    ("OCC(O)CO", "propane-1,2,3-triol"),
    ("BrCc1ccccc1CBr", "1,2-di(bromomethyl)benzene"),
])
def test_cluster_e_veto_does_not_over_veto_correct_multiplicative(smiles, expected):
    assert Orthonym().name(smiles) == expected
