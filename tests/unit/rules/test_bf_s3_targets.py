"""Slice S3 targets: suffixes that need hydrogen the mancude bridged parent does not have.

Task S3.3: a ketone on a ring atom of the fused parent (indicated hydrogen that holds it,
, or 'added indicated hydrogen',; S3.4: cyclic anhydrides, esters, amides
and imides as pseudoketones; S3.5: a monovalent suffix or free valence on an atom
with no hydrogen in the mancude parent. Blue Book rows (the Blue Book)::24792 '1,2,3,7,8,8a-
hexahydro-4H-3a,7-methanoazulene-4,9-dione (PIN)',:24800 '5,6-dihydro-1H,3H,4H-3a,6a-
methanocyclopenta[c]furan-1,3-dione (PIN)',:32535 'tetrahydro-4,8-ethanopyrano[4,3-c]pyran-
1,3,5,7-tetrone (PIN)'. The dev rows (a dev split 1, milestone1500 2, dev2000 4) and the class
rows (census, the real-data sample, the pubchem10k imides) abstain at the PIN tier at the base
or (S3.5) ship there a name that is not the book's form. Every expected name was read back by
OPSIN 2.9.0 to the input's full InChIKey before it was written here (S3 planning notes,
ledger)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact


BB_TARGETS = [
    pytest.param('O=C1C=CC2CC3CCCC13C2=O',
                 '1,2,3,7,8,8a-hexahydro-4H-3a,7-methanoazulene-4,9-dione'),
    pytest.param('O=C1OC(=O)C23CCCC12C3',
                 '5,6-dihydro-1H,3H,4H-3a,6a-methanocyclopenta[c]furan-1,3-dione'),
    pytest.param('O=C1OC(=O)C2C3CCC1C2C(=O)OC3=O',
                 'tetrahydro-4,8-ethanopyrano[4,3-c]pyran-1,3,5,7-tetrone'),
]
DEV_TARGETS = [
    pytest.param('CC1=CC(=O)[C@@H]2[C@@](O)(CC[C@]34O[C@]23C(=O)c2cccc(O)c2C4=O)C1',
                 '(4aR,6aS,12aR,12bR)-4a,8-dihydroxy-3-methyl-4a,5,6,12b-tetrahydro-6a,12a-epoxytetraphene-1,7,12(4H)-trione'),  # a dev split, milestone1500
    pytest.param('CC1(C)C(=O)CC[C@]2(C)[C@@H]1CC[C@@]13CCC(C[C@H]12)[C@@](O)(CO)C3',
                 '(2R,4aR,4bS,8aS,10aS)-2-hydroxy-2-(hydroxymethyl)-4b,8,8-trimethyldodecahydro-7H-3,10a-ethanophenanthren-7-one'),  # dev2000, census
    pytest.param('C1=CC=C(C=C1)C23C=CC(C4C2C(=O)N(C4=O)C5=CC=C(C=C5)[N+](=O)[O-])C6C3C(=O)N(C6=O)C7=CC=C(C=C7)[N+](=O)[O-]',
                 "2,6-bis(4-nitrophenyl)-4-phenylhexahydro-4,8-ethenobenzo[1,2-c:4,5-c']dipyrrole-1,3,5,7(2H,6H)-tetrone"),  # milestone1500, census
    pytest.param('CC(C)C1=C[C@@]23CC[C@H]4C(C)(C)CCC[C@]4(C(=O)O2)C3=CC1=O',
                 '(4aR,8aR,10aS)-1,1-dimethyl-7-(propan-2-yl)-1,3,4,9,10,10a-hexahydro-2H,6H-8a,4a-(epoxymethano)phenanthrene-6,12-dione'),  # dev2000, census
    pytest.param('CC1C2[C@]3(CC[C@H]1C)CCC1(C)[C@@]2(C=CC2[C@@]4(C)CC[C@H](O)C(C)(C)C4CC[C@]21C)OC3=O',
                 '(2R,4aS,6bR,10S,12aS,14aS)-10-hydroxy-1,2,6a,6b,9,9,12a-heptamethyl-1,3,4,6,6a,6b,7,8,8a,9,10,11,12,12a,12b,14b-hexadecahydro-2H,5H-14a,4a-(epoxymethano)picen-16-one'),  # dev2000, census
    pytest.param('C[C@@H]1CC[C@H](O)[C@H]2O[C@@]3(C)OC(=O)[C@@]12[C@H]3O',
                 '(2S,4aR,5R,8S,8aS,9R)-8,9-dihydroxy-2,5-dimethyl-6,7,8,8a-tetrahydro-2H,4H,5H-2,4a-methano-1,3-benzodioxin-4-one'),  # dev2000, census
]
CLASS_TARGETS = [
    pytest.param('C1CCC(=O)[C@@H]2[C@@H](C1)[C@@H]3C=C[C@@H]2C3=O',
                 '(1S,4S,4aR,9aR)-1,4,4a,6,7,8,9,9a-octahydro-5H-1,4-methanobenzo[7]annulene-5,10-dione'),  # sample
    pytest.param('CC1(C)CC[C@]2(O)[C@]13CCC(=O)[C@](O)(C3)[C@]2(C)O',
                 '(3aS,7S,8R,8aS)-7,8,8a-trihydroxy-3,3,8-trimethyloctahydro-6H-3a,7-methanoazulen-6-one'),  # sample
    pytest.param('COc1cc2c(cc1O)CN1CC[C@@]23C=CC(=O)C[C@H]13',
                 '(4aS,10bR)-8-hydroxy-9-methoxy-4,4a-dihydro-3H,6H-5,10b-ethanophenanthridin-3-one'),  # sample
    pytest.param('CC[C@@H]1CC(=O)C2C1CCC13CCC2(C1)C3',
                 '(1R)-1-ethyloctahydro-3H-4,7:4,7-dimethanocyclopenta[8]annulen-3-one'),  # census
    pytest.param('C[C@@H]1[C@@H](O)[C@]2(C)C(=O)C=CC3=CC(=O)[C@@]1(O)C[C@@]32C',
                 '(1R,7R,8aS,9R,10R)-7,10-dihydroxy-1,8a,9-trimethyl-1,7,8,8a-tetrahydro-1,7-ethanonaphthalene-2,6-dione'),  # sample
    pytest.param('C[C@@H]1[C@@H]2CC[C@@H]3[C@@]4(CCCC([C@H]4CC[C@]3(C2)C1=O)(C)C)C',
                 '(4aR,6aR,8R,9R,11aR,11bR)-4,4,8,11b-tetramethyldodecahydro-6a,9-methanocyclohepta[a]naphthalen-7(1H)-one'),  # sample
    pytest.param('CC1=C[C@@H]2C[C@@H]3[C@H]1[C@H](C2=O)OC3',
                 '(3R,3aR,6S,7aR)-4-methyl-2,3,3a,7a-tetrahydro-3,6-methano-1-benzofuran-7(6H)-one'),  # sample
    pytest.param('C1[C@@H]2C[C@@H]3C[C@H]1CC(=[18O])[C@H]3C2',
                 '(2R,3aR,5S,7aS)-octahydro-7H-2,5-methanoinden-7-(18O)one'),  # sample
    pytest.param('C[C@@H]1CCC23CC[C@H](C2[C@@]1(C(C[C@@](C(=O)[C@@H]3C)(C)SC4=CC=C(C=C4)Cl)OCOC)C)OC',
                 '(1R,4R,6R,9R,10R)-6-[(4-chlorophenyl)sulfanyl]-1-methoxy-8-(methoxymethoxy)-4,6,9,10-tetramethyloctahydro-3a,9-propanocyclopenta[8]annulen-5(4H)-one'),  # sample; '(methoxymethoxy)': Task S3.1b,
    pytest.param('O=C1[C@@H]2[C@H](C(=O)N1c1ccccc1O)[C@@H]1C=C[C@H]2C1',
                 '(3aR,4S,7R,7aS)-2-(2-hydroxyphenyl)-3a,4,7,7a-tetrahydro-1H-4,7-methanoisoindole-1,3(2H)-dione'),  # census
    pytest.param('O=C1C2C3CC(c4ccccc4)C(C3)C2C(=O)N1c1ccc(F)cc1',
                 '2-(4-fluorophenyl)-5-phenylhexahydro-1H-4,7-methanoisoindole-1,3(2H)-dione'),  # census
    pytest.param('O=C1[C@@H]2[C@@H](C(=O)N1c1ccc(Cl)cc1[N+](=O)[O-])[C@H]1C=C[C@H]2CC1',
                 '(3aS,4R,7R,7aS)-2-(4-chloro-2-nitrophenyl)-3a,4,7,7a-tetrahydro-1H-4,7-ethanoisoindole-1,3(2H)-dione'),  # census
    pytest.param('CN1C(=O)C2C3C=CC(C2C1=O)C1C(=O)N(C)C(=O)C31',
                 "2,6-dimethylhexahydro-4,8-ethenobenzo[1,2-c:4,5-c']dipyrrole-1,3,5,7(2H,6H)-tetrone"),  # census
    pytest.param('CC1=CC=C(C=C1)N2C(=O)[C@H]3[C@@H](C2=O)[C@]4(C(=C([C@]3(C4(Cl)Cl)Cl)Cl)Cl)Cl',
                 '(3aR,4R,7R,7aS)-4,5,6,7,8,8-hexachloro-2-(4-methylphenyl)-3a,4,7,7a-tetrahydro-1H-4,7-methanoisoindole-1,3(2H)-dione'),  # sample
    pytest.param('C1[C@H]2[C@@H]3C=C[C@@H]([C@@H]2C(=O)O1)O3',
                 '(3aS,4S,7S,7aR)-3a,4,7,7a-tetrahydro-4,7-epoxy-2-benzofuran-1(3H)-one'),  # sample
    pytest.param('CC=CC=CC(O)=C1C(=O)[C@]2(C)C(=O)[C@@](C)(O)[C@@H]1[C@@H]1[C@H]2CC(=O)N1C',
                 '(3aS,4R,6S,7S,7aS)-6-hydroxy-8-(1-hydroxyhexa-2,4-dien-1-ylidene)-1,4,6-trimethyltetrahydro-1H-4,7-ethanoindole-2,5,9(3H,4H)-trione'),  # sample
    pytest.param('C=C1[C@@H]2C(=O)OC[C@H]3[C@@H]2[C@@H](C(C)(C)O)CC[C@@]13C',
                 '(4R,4aR,5S,8R,8aS)-5-(2-hydroxypropan-2-yl)-8-methyl-9-methylideneoctahydro-3H-4,8-methano-2-benzopyran-3-one'),  # sample
    pytest.param('C[C@]12CC[C@@]3(O1)CCC[C@]34CC[C@H]2C(=O)O4',
                 '(3aS,6R,7R,9aS)-6-methyloctahydro-1H-3a,6-epoxy-9a,7-(epoxymethano)cyclopenta[8]annulen-11-one'),  # sample
    pytest.param('CC(C)C1=C2[C@H]3C[C@H]4O[C@](O)([C@H](O)[C@H]4CO)[C@]3(C)CC[C@@]2(C)[C@H](O)[C@H]1O',
                 '(2S,3S,3aR,5aR,6S,7R,8R,9R,10aR)-8-(hydroxymethyl)-3a,5a-dimethyl-1-(propan-2-yl)-3,3a,4,5,5a,7,8,9,10,10a-decahydro-6,9-epoxycyclohepta[e]indene-2,3,6,7(2H)-tetrol'),  # sample
    pytest.param('NCCCC12CCC(c3ccccc31)c1ccccc12',
                 '3-(9,10-ethanoanthracen-9(10H)-yl)propan-1-amine'),  # sample
    pytest.param('C=C1C[C@@]2(O)O[C@@]3(C[C@H]2C(C)C)[C@@H](C)CC[C@@H]13',
                 '(3S,3aS,5S,6R,8aS)-3-methyl-8-methylidene-5-(propan-2-yl)octahydro-6H-3a,6-epoxyazulen-6-ol'),  # sample
    pytest.param('Cc1cc2c(cc1O)[C@@]1(C)C[C@](O)(C[C@]1(C)CO)O2',
                 '(2R,4S,5S)-4-(hydroxymethyl)-4,5,8-trimethyl-4,5-dihydro-2,5-methano-1-benzoxepine-2,7(3H)-diol'),  # sample
    pytest.param('C=C1C[C@]23C[C@@]1(O)CC[C@H]2[C@]1(C(=O)O)CC[C@H](O)[C@@](C)(C(=O)O)[C@H]1[C@@H]3C(=O)O',
                 '(1S,2S,4aR,4bR,7S,9aS,10S,10aS)-2,7-dihydroxy-1-methyl-8-methylidenedodecahydro-4aH-7,9a-methanobenzo[a]azulene-1,4a,10-tricarboxylic acid'),  # sample
    pytest.param('COC(=O)C12C3CC(C1C4CC2C=C4)C(C3)OC5CCCC5',
                 'methyl 2-(cyclopentyloxy)-1,3,4,5,8,8a-hexahydro-1,4:5,8-dimethanonaphthalene-4a(2H)-carboxylate'),  # sample
]

ALL_TARGETS = BB_TARGETS + DEV_TARGETS + CLASS_TARGETS


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s3"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ALL_TARGETS)
def test_s3_target_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
