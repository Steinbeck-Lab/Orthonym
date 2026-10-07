"""Slice S2c-1 targets, each a strict xfail until the task that makes it live lands.

- Task S2c1.3 (h), the Blue Book;:12403 'pyrazino[2,3-d]pyridazine (PIN)
  (locants '1,2' of pyridazine preferred to locants '1,4' of pyrazine)'): the same molecule in
  every SMILES spelling, today 'pyridazino[4,5-b]pyrazine' for two of the four.
- Task S2c1.4 (the catalogue's PIN spellings): (:11982) "The Hantzsch-Widman
  names 1,2-thiazole, 1,2-oxazole, 1,3- thiazole, and 1,3-oxazole, respectively, must be used;
  the locants are enclosed in square brackets in the completed fusion name";
  (:11815) "for preferred IUPAC names locants must be cited";:11628 "the PIN is
  1H-pyrrolizine"; (:11903) no fusion name for a system with a retained name
  ('3H-pyrrolizine', not '3H-pyrrolo[1,2-a]pyrrole'); (:6936) no hyphen between a
  word and an opening bracket ('6-methyl[1,3]thiazolo[5,4-b]pyridine', as
  'octahydro[1,4]dioxocino[2,3-c][1,6]dioxecine-2,5,9,12-tetrone (PIN)':32191).
- Task S2c1.6 (the two-component heterocyclic fusion names and benzo names of the bridged
  fused parents no table holds, numbered by OPSIN and cross-checked by the grid fusion
  numbering of Task S2c1.2): the real-data sample's sole-blocker rows of the S2c-1 class (the
  S2c-1 measure study), three rows of older tests whose von Baeyer names were the known
  non-PIN class, and four bridged rows on three-ring parents with a five-membered ring.

Every expected name was read back by OPSIN 2.9.0 to the input's full InChIKey before it was
written here (S2c-1 planning notes, the ledger); the old spellings read back the same, so only
these tests, never a read-back, hold the spelling."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact


def _pending(task):
    return pytest.mark.xfail(strict=True, reason=f"built by Task {task}")


BASE_COMPONENT_TARGETS = [
    pytest.param('c1cnc2cnncc2n1',
                 'pyrazino[2,3-d]pyridazine'),
    pytest.param('n1ccnc2cnncc12',
                 'pyrazino[2,3-d]pyridazine'),
    pytest.param('Cc1cnc2cnncc2n1',
                 '2-methylpyrazino[2,3-d]pyridazine'),
]
SPELLING_TARGETS = [
    pytest.param('N1C=NC2=C1C=CC=C2',
                 '1H-1,3-benzimidazole'),
    pytest.param('CC1=NC2=C(N1)C=CC=C2',
                 '2-methyl-1H-1,3-benzimidazole'),
    pytest.param('N1C(NC2=C1C=CC=C2)=S',
                 '1,3-dihydro-2H-1,3-benzimidazole-2-thione'),
    pytest.param('OC(=O)c1ccc2[nH]cnc2c1',
                 '1H-1,3-benzimidazole-5-carboxylic acid'),
    pytest.param('OCc1nc2ccccc2[nH]1',
                 '(1H-1,3-benzimidazol-2-yl)methanol'),
    pytest.param('Cn1cnc2ccccc12',
                 '1-methyl-1H-1,3-benzimidazole'),
    pytest.param('N1N=NC2=C1C=CC=C2',
                 '1H-1,2,3-benzotriazole'),
    pytest.param('Cn1nnc2ccccc12',
                 '1-methyl-1H-1,2,3-benzotriazole'),
    pytest.param('On1nnc2ccccc12',
                 '1H-1,2,3-benzotriazol-1-ol'),
    pytest.param('O1N=CC2=C1C=CC=C2',
                 '1,2-benzoxazole'),
    pytest.param('c1ccc2nocc2c1',
                 '2,1-benzoxazole'),
    pytest.param('Cc1noc2ccccc12',
                 '3-methyl-1,2-benzoxazole'),
    pytest.param('O1C=NC2=NC=CC=C21',
                 '[1,3]oxazolo[4,5-b]pyridine'),
    pytest.param('c1cnc2ocnc2c1',
                 '[1,3]oxazolo[5,4-b]pyridine'),
    pytest.param('c1cnc2scnc2c1',
                 '[1,3]thiazolo[5,4-b]pyridine'),
    pytest.param('c1cnc2ncsc2c1',
                 '[1,3]thiazolo[4,5-b]pyridine'),
    pytest.param('Cc1cnc2scnc2c1',
                 '6-methyl[1,3]thiazolo[5,4-b]pyridine'),
    pytest.param('Cc1nc2ncccc2o1',
                 '2-methyl[1,3]oxazolo[4,5-b]pyridine'),
    pytest.param('Oc1ccccc1-c1nc2cccnc2s1',
                 '2-([1,3]thiazolo[5,4-b]pyridin-2-yl)phenol'),
    pytest.param('c1cnc2oncc2c1',
                 '[1,2]oxazolo[5,4-b]pyridine'),
    pytest.param('Cc1noc2ncccc12',
                 '3-methyl[1,2]oxazolo[5,4-b]pyridine'),
    pytest.param('c1cnc2nocc2c1',
                 '[1,2]oxazolo[3,4-b]pyridine'),
    pytest.param('c1cn2ccsc2n1',
                 'imidazo[2,1-b][1,3]thiazole'),
    pytest.param('C1CSc2nccn21',
                 '2,3-dihydroimidazo[2,1-b][1,3]thiazole'),
    pytest.param('c1ccc(cc1)-c1cn2ccsc2n1',
                 '6-phenylimidazo[2,1-b][1,3]thiazole'),
    pytest.param('OCc1cn2ccsc2n1',
                 '(imidazo[2,1-b][1,3]thiazol-6-yl)methanol'),
    # the 1H tautomer: lane L1a respelled the catalogue entry (Table 2.8,:11628 "the PIN is
    # 1H-pyrrolizine"), so these two rows hold from the start, beside the 3H tautomer below
    pytest.param('C1C=CN2C=CC=C12', '1H-pyrrolizine'),
    pytest.param('CC1=CCC2=CC=CN12', '3-methyl-1H-pyrrolizine'),
    pytest.param('C1Nc2ncccc2O1',
                 '2,3-dihydro[1,3]oxazolo[4,5-b]pyridine'),
    pytest.param('c1cnc2scnc2n1',
                 '[1,3]thiazolo[4,5-b]pyrazine'),
    pytest.param('c1cc2ncoc2cn1',
                 '[1,3]oxazolo[5,4-c]pyridine'),
    pytest.param('c1cc2ncsc2cn1',
                 '[1,3]thiazolo[5,4-c]pyridine'),
    pytest.param('Cc1cc2ncoc2cn1',
                 '6-methyl[1,3]oxazolo[5,4-c]pyridine'),
    pytest.param('Cc1cc2ncsc2cn1',
                 '6-methyl[1,3]thiazolo[5,4-c]pyridine'),
    pytest.param('C1=CCN2C=CC=C12',
                 '3H-pyrrolizine'),
    pytest.param('CC1C=CC2=CC=CN12',
                 '3-methyl-3H-pyrrolizine'),
    pytest.param('Cc1ccc2C=CCn12',
                 '5-methyl-3H-pyrrolizine'),
]
PRODUCER_TARGETS = [
    pytest.param('C=1C2=C(OC1)C=CC=1C3CCC(C12)C3',
                 '6,7,8,9-tetrahydro-6,9-methanonaphtho[2,1-b]furan'),
    pytest.param('CC1=CC2=C(O1)C=CC=1C3CCC(C12)C3',
                 '2-methyl-6,7,8,9-tetrahydro-6,9-methanonaphtho[2,1-b]furan'),
    pytest.param('C1=CNC=2C=CC3=C(C12)C1CCC3C1',
                 '6,7,8,9-tetrahydro-3H-6,9-methanobenzo[e]indole'),
    pytest.param('C=1C2=C(SC1)C=CC=1C3CCC(C12)C3',
                 '6,7,8,9-tetrahydro-6,9-methanonaphtho[2,1-b]thiophene'),
    pytest.param('C1=CC=C(C=C1)C23C4=C5C(=CC=C4)OP(O5)OOC26OP(O6)OO3',
                 '12b-phenyl-12bH-3,4a:7,9-diepoxy[1,2,4,3]trioxaphosphinino[5,6-e][1,3,4,2]benzotrioxaphosphocine'),
    pytest.param('C1C2C3CN(C=N3)C(=C2NN1)C(=O)O',
                 '2,3,3a,4-tetrahydro-1H-4,7-methanopyrazolo[3,4-e][1,3]diazepine-8-carboxylic acid'),
    pytest.param('C1C2C3COC(C1=NC4=CC=CC=C4N2)O3',
                 '5,6,7,8-tetrahydro-3H-3,6-epoxy-2,7-methano-4,1,8-benzoxadiazecine'),
    pytest.param('C1C2CN(C1C3C2ON=C3C4=CC=CC=C4)CC5=CC=CC=C5',
                 '5-benzyl-3-phenyl-3a,4,5,6,7,7a-hexahydro-4,7-methano[1,2]oxazolo[4,5-c]pyridine'),
    pytest.param('C1CC2C3CCCN(C3)C2C1',
                 'octahydro-2H-1,5-methanocyclopenta[b]azepine'),
    pytest.param('C1[C@H]2CNC[C@@H]1C3=C2C=CN=C3',
                 '(5R,9S)-6,7,8,9-tetrahydro-5H-5,9-methanopyrido[3,4-d]azepine'),
    pytest.param('C1[C@H]2[C@@H](CO[C@@H]1C3=C(N2)C=C(C=C3)CCC4=CC=CC=N4)O',
                 '(2S,3S,6S)-9-[2-(pyridin-2-yl)ethyl]-1,3,4,6-tetrahydro-2H-2,6-methano-5,1-benzoxazocin-3-ol'),
    pytest.param('C=CCN1CC[C@@]2(C)c3cc(O)ccc3C[C@@H]1[C@@H]2C',
                 '(2R,6R,11R)-6,11-dimethyl-3-(prop-2-en-1-yl)-1,2,3,4,5,6-hexahydro-2,6-methano-3-benzazocin-8-ol'),
    pytest.param('C=CCN1CC[C@]2(C)c3cc(O)ccc3C[C@H]1[C@H]2C',
                 '(2S,6S,11S)-6,11-dimethyl-3-(prop-2-en-1-yl)-1,2,3,4,5,6-hexahydro-2,6-methano-3-benzazocin-8-ol'),
    pytest.param('C=CC1C2CCCC3C(C2)C1OC3(C(F)(F)F)C(F)(F)F',
                 '9-ethenyl-3,3-bis(trifluoromethyl)octahydro-1H-1,7-methanocyclohepta[c]furan'),
    pytest.param('CC(C)(O)C1=CCC23COC(C2)C(O)(C(=O)O)CCC13',
                 '4-hydroxy-7-(2-hydroxypropan-2-yl)-3,4,5,6,6a,9-hexahydro-1H-3,9a-methanocyclopenta[c]oxocine-4-carboxylic acid'),
    pytest.param('CC(C)(O)[C@@H]1CC[C@]23CO[C@H](C2)[C@@](O)(C(=O)O)CC[C@@H]13',
                 '(3R,4R,6aS,7R,9aS)-4-hydroxy-7-(2-hydroxypropan-2-yl)octahydro-1H-3,9a-methanocyclopenta[c]oxocine-4-carboxylic acid'),
    pytest.param('CC(C)=CCc1ccc(O)c2c1C=C[C@H]1O[C@@H]2O[C@H]1C',
                 '(1S,3S,4R)-3-methyl-7-(3-methylbut-2-en-1-yl)-3,4-dihydro-1H-1,4-epoxy-2-benzoxocin-10-ol'),
    pytest.param('CC1(C=CC2=CC(=C3C4CCC(C3=C2O1)CC4)O)C',
                 '2,2-dimethyl-7,8,9,10-tetrahydro-2H-7,10-ethanonaphtho[1,2-b]pyran-6-ol'),
    pytest.param('CC1(O[C@@H]2[C@H]3[C@H]4C=C[C@@H]([C@H]3[C@H]([C@@H]2O1)O)C4=O)C',
                 '(3aR,3bR,4R,7S,7aS,8R,8aS)-8-hydroxy-2,2-dimethyl-3a,3b,7,7a,8,8a-hexahydro-2H,4H-4,7-methanoindeno[1,2-d][1,3]dioxol-9-one'),
    pytest.param('CC1=CC23C(CC1)(C4(C(CC(C4(O2)CO)O3)O)C)C',
                 '3a-(hydroxymethyl)-6,8a,8b-trimethyl-1,2,3,3a,7,8,8a,8b-octahydro-3,4a-epoxycyclopenta[b][1]benzofuran-1-ol'),
    pytest.param('CC1CCC2C(C)(C)C3CC12CC1OC13C',
                 '3,6,6,7a-tetramethyloctahydro-2H-2a,7-methanoazuleno[5,6-b]oxirene'),
    pytest.param('CC1COCC2=C1C1C(C)(C)[C@H]3CC[C@@]1(C3)C(C)(C)C2',
                 '(6aS,9S)-1,6,6,10,10-pentamethyl-1,2,4,5,6,7,8,9,10,10a-decahydro-6a,9-methanonaphtho[2,1-c]pyran'),
    pytest.param('CCC[C@H]1CCC[C@@H]2N1C[C@H]3C[C@@H]2CN(C3)C4=NC=NC5=C4NC=C5',
                 '(1R,5R,8S,11aS)-8-propyl-3-(5H-pyrrolo[3,2-d]pyrimidin-4-yl)decahydro-2H-1,5-methanopyrido[1,2-a][1,5]diazocine'),
    pytest.param('CO[C@H]1[C@H]2C(O)O[C@@H]1[C@]1(C)CC[C@@]3(C(=O)O)CCC(C(C)C)=C3[C@H]1C[C@H]2OC',
                 '(2R,3S,6R,6aR,8aS,11bR,12S)-4-hydroxy-2,12-dimethoxy-6a-methyl-11-(propan-2-yl)-1,3,4,6,6a,7,8,9,10,11b-decahydro-3,6-methanoindeno[5,4-c]oxocine-8a(2H)-carboxylic acid'),
    pytest.param('COc1cccc2c1[C@H]1Oc3ccc4cc(C)cc(O)c4c3[C@@H]2O1',
                 '(8R,13R)-9-methoxy-3-methyl-8,13-dihydro-8,13-epoxynaphtho[2,1-c][2]benzoxepin-1-ol'),
    pytest.param('C[C@@H]1CC=C2[C@]13C[C@@]4([C@]([C@@]2(COC([C@@H]3O)O4)C)(C)O)O',
                 '(1S,6R,7aS,8R,11R,12R)-1,8,12-trimethyl-1,2,8,9-tetrahydro-4H-1,6:4,7a-dimethanocyclopenta[f][1,3]dioxonine-6,11,12(7H)-triol'),
    pytest.param('C[C@@H]1CC[C@@H]2[C@@]13C[C@@H](C2(C)C)C4(C(C3)O4)C',
                 '(2aR,3R,5aS,7S)-3,6,6,7a-tetramethyloctahydro-2H-2a,7-methanoazuleno[5,6-b]oxirene'),
    pytest.param('C[C@@]12CCC[C@@](C)(O1)c1ccc(C(=O)O)cc1O2',
                 '(2R,6R)-2,6-dimethyl-3,4,5,6-tetrahydro-2H-2,6-epoxy-1-benzoxocine-9-carboxylic acid'),
    pytest.param('C[C@]12CC[C@H](C1(C)C)C3=CN=C4C(=C23)C=CC5=CC=CC=C54',
                 '(7R,10S)-10,13,13-trimethyl-7,8,9,10-tetrahydro-7,10-methanobenzo[c]phenanthridine'),
    pytest.param('C[C@]12[C@H]3C[C@H](C[C@@H]1[C@@H](OC3)C4=CC=C(C=C4)OC)C(O2)(C)C',
                 '(3R,4aR,5R,8S,8aS)-5-(4-methoxyphenyl)-2,2,8a-trimethylhexahydro-2H,5H-3,8-methanopyrano[4,3-b]pyran'),
    pytest.param('NCCC1=C[C@H]2Cc3nc4cc(Cl)ccc4c(N)c3[C@@H](C1)C2',
                 '(7R,11R)-9-(2-aminoethyl)-3-chloro-6,7,10,11-tetrahydro-7,11-methanocycloocta[b]quinolin-12-amine'),
    pytest.param('CC1(C)CC=C[C@]2(C)OO[C@@H]3C[C@@]12CC[C@H]3O',
                 '(3R,4R,6aS,10aS)-7,7,10a-trimethyl-3,4,5,6,8,10a-hexahydro-7H-3,6a-methano-1,2-benzodioxocin-4-ol'),
    pytest.param('COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)C15CC2)[C@@H]3[C@@H]4O',
                 'methyl (1S,7R,12bS,12cR)-1-hydroxy-8-methyl-1,4,5,6,7,12c-hexahydro-3H,8H-2,12b:5a,7a-diethanoazepino[4,3-c]carbazole-7-carboxylate'),
    pytest.param('C1=CC2OOOC1c1ccccc12',
                 '1,5-dihydro-1,5-etheno-2,3,4-benzotrioxepine'),
]
ALL_TARGETS = BASE_COMPONENT_TARGETS + SPELLING_TARGETS + PRODUCER_TARGETS


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2c1"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ALL_TARGETS)
def test_s2c1_target_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
