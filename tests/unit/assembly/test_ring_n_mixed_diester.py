"""A ring-nitrogen carboxylic ester and a carboxylic ester on the same ring are the mixed diester of one diacid
(2026-11-09).

'pyrrolidine-1,2-dicarboxylic acid' is one acid, 'pyrrolidine-1-carboxylic acid (PIN)'), so its two esters
are named on it with organyl locants, the Blue Book, '1,1-diethyl 3-methyl
butane-1,1,3-tricarboxylate (PIN)':31783). The PIN tier declined these Boc amino-acid esters.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC(C)(C)OC(=O)N1CCCC1C(=O)OC", "1-tert-butyl 2-methyl pyrrolidine-1,2-dicarboxylate"),
    ("CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)OC", "1-tert-butyl 2-methyl (2S)-pyrrolidine-1,2-dicarboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)C(=O)OCC", "1-tert-butyl 4-ethyl piperidine-1,4-dicarboxylate"),
    ("COC(=O)N1CCCC1C(=O)OC", "dimethyl pyrrolidine-1,2-dicarboxylate"),
    ("CC(C)(C)OC(=O)N1C[C@H](O)C[C@H]1C(=O)OC", "1-tert-butyl 2-methyl (2S,4R)-4-hydroxypyrrolidine-1,2-dicarboxylate"),
    # an ester of another acid stays a prefix of the ring-nitrogen ester
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)C", "tert-butyl 4-(acetyloxy)piperidine-1-carboxylate"),
])
def test_ring_n_mixed_diester(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", [
    ("CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)Oc1ccc(cc1)[N+](=O)[O-]", "1-tert-butyl 2-(4-nitrophenyl) (2S)-pyrrolidine-1,2-dicarboxylate"),
    ("CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)OCc1ccccn1", "1-tert-butyl 2-[(pyridin-2-yl)methyl] (2S)-pyrrolidine-1,2-dicarboxylate"),
    ("CC(C)OC(=O)N1CCC[C@H]1C(=O)OC", "2-methyl 1-(propan-2-yl) (2S)-pyrrolidine-1,2-dicarboxylate"),
    ("O=C(OCC1c2ccccc2-c2ccccc21)N1CCC[C@H]1C(=O)OC", "1-[(9H-fluoren-9-yl)methyl] 2-methyl (2S)-pyrrolidine-1,2-dicarboxylate"),
])
def test_a_compound_organyl_behind_its_locant_is_enclosed(smiles, name):
    """ (the Blue Book): a compound prefix is enclosed, also behind an organyl locant (review 2026-11-09:
    '2-methyl 1-propan-2-yl...' shipped as a PIN)."""
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", [
    # an aromatic ring nitrogen carries a carboxylic acid too ('1H-indole-1,5-dicarboxylic acid'); the two esters reach
    # each other through the N, not a C-C path (rules.esters.classify_multi_ester)
    ("CC(C)(C)OC(=O)n1ccc2cc(C(=O)OC)ccc21", "1-tert-butyl 5-methyl 1H-indole-1,5-dicarboxylate"),
    ("CC(C)(C)OC(=O)n1cc(C(=O)OC)cn1", "1-tert-butyl 4-methyl 1H-pyrazole-1,4-dicarboxylate"),
    ("CCOC(=O)n1ccc(C(=O)OCC)c1", "diethyl 1H-pyrrole-1,3-dicarboxylate"),
    # two ring-nitrogen esters of one ring
    ("CC(C)(C)OC(=O)N1CCN(CC1)C(=O)OCC", "tert-butyl ethyl piperazine-1,4-dicarboxylate"),
    ("CC(C)(C)OC(=O)N1CCN(C(C)C1)C(=O)OCc1ccccc1", "1-benzyl 4-tert-butyl 2-methylpiperazine-1,4-dicarboxylate"),
])
def test_ring_nitrogen_diesters_of_aromatic_and_two_nitrogen_rings(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
