"""Slice S2b targets: bridged fused PINs on the fused parents the Blue Book prints and the
older tables do not number (Task S2b.3), on the phenanthridine skeleton once its catalogue map
is corrected (S2b.4), and on the two-component carbocyclic parents no table holds (S2b.5).

Every expected name was read back by OPSIN 2.9.0 to the input's full InChIKey before it was
written here (S2b planning notes, section 2). Blue Book rows (the Blue Book)::19829
'4,7-methanocyclopenta[a]indene (PIN)',:23875 'hexadecahydro-1H-8,12-methanobenzo[13]annulene
(PIN)',:19904 '1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine (PIN)',:14454
'6,13-ethano-6,13-methanodibenzo[b,g][1,6]diazecine (PIN)',:14448
'6,14:7,14-dimethanobenzo[7,8]cycloundeca[1,2-b]pyridine (PIN)'. The dev rows (milestone1500
2, dev2000 5) and the class rows (census and the real-data sample, and probes on the new
parents) abstain at the PIN tier at the base."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

BB_TARGETS = [
    pytest.param('C1=CC2=CC3=C4C=CC(=C3C2=C1)C4',
                 '4,7-methanocyclopenta[a]indene'),
    pytest.param('C1CC2CCCC3CCCCC3CCCC(C1)C2',
                 'hexadecahydro-1H-8,12-methanobenzo[13]annulene'),
    pytest.param('C1=CC2=C3N=Cc4cc5ccc6cc5cc4C(=CC3=C1C2)CCCCC6',
                 '1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine'),
    pytest.param('C1=c2ccccc2=NC23C=c4ccccc4=NC1(CC2)C3',
                 '6,13-ethano-6,13-methanodibenzo[b,g][1,6]diazecine'),
    pytest.param('C1=c2ccccc2=CC23C=c4ncccc4=CC(=C1C2)C3',
                 '6,14:7,14-dimethanobenzo[7,8]cycloundeca[1,2-b]pyridine'),
]
DEV_TARGETS = [  # milestone1500 (2) and dev2000 (5): terpenoids on the benzo[a]azulene,
    # cyclohepta[a]naphthalene and cyclobuta[a]heptalene parents
    pytest.param('C=C1C[C@]23C[C@@]1(O)CC[C@H]2[C@]1(C)CC[C@H](O)[C@@](C)(C(=O)O)[C@H]1[C@@H]3C(=O)O',
                 '(1S,2S,4aS,4bS,7S,9aS,10S,10aS)-2,7-dihydroxy-1,4a-dimethyl-8-methylidenedodecahydro-1H-7,9a-methanobenzo[a]azulene-1,10-dicarboxylic acid'),
    pytest.param('C=C1C[C@]23C[C@H]1CC[C@H]2[C@]1(C)CCC[C@@](C)(C(=O)O)[C@H]1[C@@H]3C=O',
                 '(1R,4aS,4bS,7R,9aR,10S,10aS)-10-formyl-1,4a-dimethyl-8-methylidenedodecahydro-1H-7,9a-methanobenzo[a]azulene-1-carboxylic acid'),
    pytest.param('C[C@@]1(C(=O)O)CCC[C@@]2(C)[C@H]1CC[C@H]1C[C@@H]3C[C@@]12CC[C@]3(O)CO',
                 '(4R,4aR,6aS,8R,9R,11aS,11bS)-9-hydroxy-9-(hydroxymethyl)-4,11b-dimethyltetradecahydro-8,11a-methanocyclohepta[a]naphthalene-4-carboxylic acid'),
    pytest.param('C=C1C[C@]23C[C@H]1CC[C@H]2[C@]1(CO)CC[C@H](O)[C@@](C)(C(=O)O)[C@H]1[C@@H]3C(=O)O',
                 '(1S,2S,4aR,4bR,7R,9aR,10S,10aS)-2-hydroxy-4a-(hydroxymethyl)-1-methyl-8-methylidenedodecahydro-1H-7,9a-methanobenzo[a]azulene-1,10-dicarboxylic acid'),
    pytest.param('CC1=C[C@@]23CC[C@@H]4C(C)(C)C[C@H](O)C[C@@]4(C)[C@@H]2CC[C@@H]1C3',
                 '(2S,4aR,6aS,9R,11aS,11bR)-4,4,8,11b-tetramethyl-1,2,3,4,4a,5,6,9,10,11,11a,11b-dodecahydro-6a,9-methanocyclohepta[a]naphthalen-2-ol'),
    pytest.param('CC(=O)OC[C@@]1(O)CC[C@]23C[C@H]1C=C2CC[C@H]1[C@](C)(C(=O)O)[C@H](O)CC[C@@]13C',
                 '(3R,4S,4aR,8S,9R,11aS,11bS)-9-[(acetyloxy)methyl]-3,9-dihydroxy-4,11b-dimethyl-1,2,3,4,4a,5,6,8,9,10,11,11b-dodecahydro-8,11a-methanocyclohepta[a]naphthalene-4-carboxylic acid'),
    pytest.param('CC1=C2[C@H](O)C[C@@]2(C)[C@@H]2C[C@@H]3C[C@H](O)[C@@H](C)[C@@]2(CC1)C3(C)C',
                 '(2R,5aS,6S,7S,9R,10aS,10bS)-3,6,10b,11,11-pentamethyl-1,4,5,6,7,8,9,10,10a,10b-decahydro-2H-5a,9-methanocyclobuta[a]heptalene-2,7-diol'),
]
CLASS_TARGETS = [
    # Blue Book parents the older tables did not number (pentalene:11459, dibenzo[b,d]furan
    #:33050, tetraphene:25762)
    pytest.param('C12CC3CC(C1)CC2C3',
                 'octahydro-2,5-methanopentalene'),
    pytest.param('C12CC3CC(C1)C(C2)C3c1ccccc1',
                 '1-phenyloctahydro-2,5-methanopentalene'),
    pytest.param('C12CCC3C(C1)CCC23',
                 'octahydro-1,4-methanopentalene'),
    pytest.param('C1=CC2CC1c1oc3ccccc3c12',
                 '1,4-dihydro-1,4-methanodibenzo[b,d]furan'),
    pytest.param('C12CCC(C3=CC=C4C=C5C=CC=CC5=CC4=C13)C2',
                 '1,2,3,4-tetrahydro-1,4-methanotetraphene'),
    # phenanthridine:11537, once the catalogue map is 's
    pytest.param('C1=CC=CC2=NC=C3C4CCC(C3=C12)C4',
                 '7,8,9,10-tetrahydro-7,10-methanophenanthridine'),
    pytest.param('COC1=CC=CC(=C1)[C@H]2[C@@H]3[C@H]4CC[C@@H](C4)[C@H]3C5=C(N2)C=CC(=C5)C(=O)O',
                 '(6R,6aR,7S,10S,10aR)-6-(3-methoxyphenyl)-5,6,6a,7,8,9,10,10a-octahydro-7,10-methanophenanthridine-2-carboxylic acid'),
    # two-component carbocyclic fusion names (sample rows)
    pytest.param('C=C1C[C@]23CC[C@H]4C(C)(C)CCC[C@]4(C)[C@H]2CC[C@@H]1C3',
                 '(4aS,6aR,9R,11aS,11bS)-4,4,11b-trimethyl-8-methylidenetetradecahydro-6a,9-methanocyclohepta[a]naphthalene'),
    pytest.param('CC(C)C1=C2[C@H]3C[C@H]4OC(O)(C=C4C=O)[C@]3(C)CC[C@@]2(C)C=C1',
                 '(3aS,5aR,9R,10aR)-6-hydroxy-3a,5a-dimethyl-1-(propan-2-yl)-3a,4,5,5a,6,9,10,10a-octahydro-6,9-epoxycyclohepta[e]indene-8-carbaldehyde'),
    pytest.param('CC1=C2CC[C@@H](C)CC/C=C(/C)[C@@H]3CC[C@](C)(C[C@@H]2O)[C@H]13',
                 '(1R,3aR,4Z,8S,12aR,13S)-1,4,8,12-tetramethyl-1,2,3,3a,6,7,8,9,10,12a-decahydro-1,11-ethanocyclopenta[11]annulen-13-ol'),
    pytest.param('CC(C)[C@H]1CC[C@]([C@H]2[C@@H]1[C@@H]3[C@]4(CCC[C@](O4)(C[C@H]2O3)C)C)(C)O',
                 '(1S,4R,4aR,5R,6R,10S,12R,12aS)-1,6,10-trimethyl-4-(propan-2-yl)tetradecahydro-5,12:6,10-diepoxybenzo[10]annulen-1-ol'),
    pytest.param('C[Si](C)(C)C1=C[C@@H]2C=C[C@H]3[C@H]4C3(C1[C@H]2C4)[Si](C)(C)C',
                 '(1S,4R,4aS,5aS)-1a,2-bis(trimethylsilyl)-1a,1b,4,4a,5,5a-hexahydro-1H-1,4-ethenocyclopropa[a]pentalene'),
    pytest.param('CCOC(=O)C1([C@@H]2[C@H]1[C@H]3C4=CC=CC=C4[C@@H]2O3)CC(=O)C',
                 'ethyl (1aR,2S,7R,7aS)-1-(2-oxopropyl)-1a,2,7,7a-tetrahydro-1H-2,7-epoxycyclopropa[b]naphthalene-1-carboxylate'),
    pytest.param('CC(C)C[C@@H]1CCC2CC[C@H](C[C@H]1C(=O)O)C3=CC(=C(C=C3CC2)OC)OC',
                 '(5R,7R,8S)-2,3-dimethoxy-8-(2-methylpropyl)-6,7,8,9,10,11,12,13-octahydro-5H-5,11-ethanobenzo[11]annulene-7-carboxylic acid'),
    pytest.param('C[C@@H]1CC[C@H]2C[C@H]3[C@]4(C)CC[C@@H]4[C@@](C)(O)CC[C@@]13C2(C)C',
                 '(2aS,3S,5aR,6R,9S,10aS,10bS)-3,6,10b,11,11-pentamethyldodecahydro-2H-5a,9-methanocyclobuta[a]heptalen-3-ol'),
]
ALL_TARGETS = BB_TARGETS + DEV_TARGETS + CLASS_TARGETS


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2b"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ALL_TARGETS)
def test_s2b_target_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
