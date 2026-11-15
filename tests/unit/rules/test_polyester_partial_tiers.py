"""A partial ester of a polyol with different acids at the wider tiers (``rules.lipids``, ``opsin_capabilities``).

 "When anions are different, two methods are used:" (the Blue Book): "Method (1) generates preferred
IUPAC names but names formed by using method (2) are acceptable in general nomenclature" ('propane-1,2,3-triyl
1,2-diacetate 3-propanoate (PIN)',:31840). A free hydroxy group is a prefix of the organyl group, as in the
identical-anion name '2-hydroxypropane-1,3-diyl diacetate'. OPSIN 2.9.0 reads no multi-anion name, so the wider tiers
ship the ester-family name it reads; an OPSIN that reads it (dan2097/opsin#354) gets the PIN at every tier.
"""
import pytest

from tests.support.opsin_marks import needs_polyester_opsin, release_polyester_only

pytestmark = pytest.mark.unit

PARTIAL = [
    ("OC(COC(=O)CCC)COC(=O)CC", "2-hydroxypropane-1,3-diyl 1-butanoate 3-propanoate"),
    ("OCC(OC(=O)CCC)COC(=O)CC", "3-hydroxypropane-1,2-diyl 2-butanoate 1-propanoate"),
    ("OC[C@H](OC(=O)CCC)COC(=O)CC", "(2S)-3-hydroxypropane-1,2-diyl 2-butanoate 1-propanoate"),
    ("CCCCCCCCCCCCCCCCCC(=O)O[C@@H](CO)COC(=O)CCCCCCC", "(2S)-3-hydroxypropane-1,2-diyl 2-octadecanoate 1-octanoate"),
]


def _namer(tier):
    from orthonym.cli import _tiered_namer
    return _tiered_namer("pin", tier)


@needs_polyester_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", PARTIAL)
def test_an_opsin_that_reads_the_multi_anion_name_gives_every_tier_the_pin(smiles, name):
    for tier in ("pin", "best-effort"):
        row = _namer(tier).name_tiered(smiles)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), (tier, row)


@needs_polyester_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)OCC(OC(=O)C)COC(=O)CC", "propane-1,2,3-triyl 1,2-diacetate 3-propanoate"),     #:31840
    ("CC(=O)OCC(OC(=O)CCl)COC(C)=O", "propane-1,2,3-triyl 1,3-diacetate 2-chloroacetate"),
])
def test_an_opsin_that_reads_the_multi_anion_name_reads_the_pin_without_the_twin(smiles, name):
    row = _namer("best-effort").name_tiered(smiles)
    assert (row["name"], row["tier"], row["gate_outcome"]) == (name, "pin_verified", "self_consistency_verified"), row


@needs_polyester_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "OC(COC(=O)CCl)COC(C)=O",
    "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)OC(CO)COC(=O)CCCCCCCCCCCCCCCCC",
])
def test_a_multi_anion_name_opsin_main_refuses_leaves_a_verified_name_below_the_pin(smiles):
    # OPSIN main refuses some multi-anion spellings ('... 2-[(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoate]'); the
    # best-effort tier then ships another name that reads back, below the PIN, and never loses the molecule
    from tests.support.rt_assert import assert_full_rt
    row = _namer("best-effort").name_tiered(smiles)
    assert row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@release_polyester_only
@pytest.mark.opsin_gate
def test_opsin_2_9_0_keeps_the_ester_family_name_at_the_wider_tier():
    row = _namer("best-effort").name_tiered(PARTIAL[0][0])
    assert row["name"] == "2-hydroxy-3-(propanoyloxy)propyl butanoate", row
    assert row["tier"] == "pin_unverified", row    # a known false label of OPSIN 2.9.0 (roadmap 24.2 lead (i))
