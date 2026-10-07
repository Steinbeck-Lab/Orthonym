"""The seniority of the fusion components (``fusion_components.rank_key``,
``hetero_fusion._senior``) and the side letters of a parent component, each on the Blue Book's
own example.

 (the Blue Book) "If there is a choice for selecting the parent component, the
following criteria are considered, in order, until a decision can be made": (a):12139, (b)
:12163, (c):12234, (d):12260, (e):12282, (f):12298, (g):12317, (h):12392, (i):12407;
the book's own reasons are cited row by row. (:11911) the side letters "beginning
with a for the side numbered '1,2', b for '2,3'", on the parent's own numbering (acridine and
carbazole keep their traditional numbering,:12493)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import fusion_components as fc
from orthonym.rules.bridged_fused_pin import hetero_fusion, parents


def _key(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


def _senior_names(smiles):
    km = _key(smiles)
    rings = fc.ortho_rings(km)
    comps = hetero_fusion._stage1_components(km, rings)
    got = hetero_fusion._senior(km, rings, list(comps.values()))
    return {c.name for c in got} if got else None


@pytest.mark.parametrize("smiles,senior,why", [
    ("c1cc2ccc3ncccc3cc-2c1", "pyridine", "(a) :12143 pyridine is senior to azulene"),
    ("C1=C2CNC=C2Oc2ccccc21", "pyrrole", "(a) :12157 pyrrole is senior to 1-benzopyran (N > O)"),
    ("C1=CSc2cocc2SC1", "furan", "(a) :12161 furan is senior to dithiepine (O > S)"),
    ("c1ccc2c(c1)[nH]c1cc3nccnc3cc12", "carbazole", "(b) :12232 carbazole (3 rings) over quinoxaline"),
    ("C1=COC2C=COC2=C1", "pyran", "(c) :12246 pyran (6 ring) preferred to furan"),
    ("C1=NOCc2cccnc21", "1,2-oxazine", "(d) :12269 oxazine (2 heteroatoms) preferred to pyridine"),
    ("c1cc2c(o1)OCO2", "1,3-dioxole", "(d) :12278"),
    ("c1poc2c1OCO2", "1,2-oxaphosphole", "(e) :12296 an O and a P atom preferred to two O atoms"),
    ("c1nc2[se]cnc2s1", "1,3-thiazole", "(f) :12311 S,N senior to Se,N"),
    ("c1csc2[se]ccoc=2o1", "1,4-oxathiine", "(f) :12315 O,S senior to O,Se"),
    ("c1ccc2nc3cc4cnc5ccccc5c4cc3cc2c1", "acridine", "(g) :12386 acridine over phenanthridine"),
    ("c1cnc2cnncc2n1", "pyridazine", "(h) :12403 locants '1,2' of pyridazine preferred to '1,4'"),
    ("[nH]1oc2os[nH]c=2s1", "1,2,3-oxathiazole", "(i) :12416 '1,2,3' lower than '1,3,2'"),
])
def test_the_parent_component_is_the_senior_one(smiles, senior, why):
    assert _senior_names(smiles) == {senior}, why


@pytest.mark.parametrize("smiles,name", [
    ("C1=Cc2c(ccc3cc4ccccc4nc23)OC1", "pyrano[2,3-c]acridine"),   #:10818, acridine numbering
    ("c1ccc2c(c1)[nH]c1cc3nccnc3cc12", "pyrazino[2,3-b]carbazole"),  #:12232, carbazole numbering
    ("C1=Cc2cc3cccc-3cn2C1", "cyclopenta[f]indolizine"),           #:49192, fusion N numbered 4
])
def test_the_parent_sides_are_lettered_on_its_own_numbering(smiles, name):
    # (:11911) "a for the side numbered '1,2', b for '2,3'", the periphery walked
    # from locant 1, the traditional numberings of acridine and carbazole kept (:12493)
    assert hetero_fusion.rule_name(_key(smiles)) == name
