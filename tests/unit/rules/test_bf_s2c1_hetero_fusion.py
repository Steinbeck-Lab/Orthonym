"""Two-component fusion names with a heterocyclic component, and benzo names
(``bridged_fused_pin.hetero_fusion``): every such name the Blue Book prints in a PIN-marked
name is the spelling gold, the book's own non-PIN spellings are negative controls, and the
numbering is OPSIN's reading of the produced name.

 (the Blue Book) parent and attached components, (:11907) heteroatom locants of
a component in square brackets, (:11911) the side letters and attached locants, (:11917) no
space or hyphen around the brackets; (:11982) '1,2-oxazole', '1,3-thiazole', never
'isoxazole', 'thiazole'; (:12137-:12418) the parent component; (:11815)
benzo names with locants; (:13437-:13463) the benzo-name unit; (:13970)
locants omitted for 'benzo' and monocyclic hydrocarbon prefixes; (:23710)."""
import re

import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import hetero_fusion, parents


def _skeleton(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


#: every two-component ortho-fused name with a heterocyclic component that the book prints in a
#: PIN-marked name (the S2c-1 rules study, section 3), on OPSIN 2.9.0's structure of the name;
#: the comment is one the Blue Book line that prints it
BOOK_FUSION_NAMES = [
    ('c1cc2cc[se]c2[se]1', 'selenopheno[2,3-b]selenophene'),  #:7549
    ('C1=CC2=C(CO1)OCO2', '[1,3]dioxolo[4,5-c]pyran'),  #:10172
    ('c1ccc2cc3cnccc3cc2c1', 'benzo[g]isoquinoline'),  #:10301
    ('c1ccc2c(c1)ccc1ccncc12', 'benzo[h]isoquinoline'),  #:10301
    ('c1cc2ccsc2o1', 'thieno[2,3-b]furan'),  #:10310
    ('c1cc2sccc2o1', 'thieno[3,2-b]furan'),  #:10310
    ('c1coc2cccc-2c1', 'cyclopenta[b]pyran'),  #:10713
    ('c1occ2c1OCS2', 'furo[3,4-d][1,3]oxathiole'),  #:10816
    ('C1=Cc2c(ccc3cc4ccccc4nc23)OC1', 'pyrano[2,3-c]acridine'),  #:10818
    ('c1cc2c[se]cc2[se]1', 'selenopheno[3,4-b]selenophene'),  #:11932
    ('c1cc2[se]ccc2[se]1', 'selenopheno[3,2-b]selenophene'),  #:11936
    ('c1cc2ccc3ncccc3cc-2c1', 'azuleno[6,5-b]pyridine'),  #:12143
    ('C1=C2CNC=C2Oc2ccccc21', '[1]benzopyrano[2,3-c]pyrrole'),  #:12157
    ('C1=CSc2cocc2SC1', '[1,4]dithiepino[2,3-c]furan'),  #:12161
    ('c1ccc2c(c1)[nH]c1cc3nccnc3cc12', 'pyrazino[2,3-b]carbazole'),  #:12232
    ('C1=COC2C=COC2=C1', 'furo[3,2-b]pyran'),  #:12246
    ('C1=NOCc2cccnc21', 'pyrido[2,3-d][1,2]oxazine'),  #:12269
    ('c1cc2c(o1)OCO2', 'furo[2,3-d][1,3]dioxole'),  #:12278
    ('c1poc2c1OCO2', '[1,3]dioxolo[4,5-d][1,2]oxaphosphole'),  #:12296
    ('c1nc2[se]cnc2s1', '[1,3]selenazolo[5,4-d][1,3]thiazole'),  #:12311
    ('c1csc2[se]ccoc=2o1', '[1,4]oxaselenino[2,3-b][1,4]oxathiine'),  #:12315
    ('c1ccc2nc3cc4cnc5ccccc5c4cc3cc2c1', 'quinolino[4,3-b]acridine'),  #:12386
    ('c1cnc2cnncc2n1', 'pyrazino[2,3-d]pyridazine'),  #:12403
    ('[nH]1oc2os[nH]c=2s1', '[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole'),  #:12416
    ('c1cc2c([nH]1)OCS2', '[1,3]oxathiolo[5,4-b]pyrrole'),  #:12556
    ('c1nc2sccc2[nH]1', 'thieno[2,3-d]imidazole'),  #:12576
    ('c1cnn2ccnc2n1', 'imidazo[1,2-b][1,2,4]triazine'),  #:12590
    ('c1cc2cc3csnc3cc2s1', 'thieno[3,2-f][2,1]benzothiazole'),  #:13443
    ('c1cnc2cc3cc4cnoc4cc3cc2c1', '[1,2]benzoxazolo[6,5-g]quinoline'),  #:13449
    ('C1=Nc2cc3ncccc3cc2SC1', '[1,4]thiazino[2,3-g]quinoline'),  #:13459
    ('c1ccc2cc3ncccc3cc2c1', 'benzo[g]quinoline'),  #:13978
    ('C1=C2N=NN=C2Cc2ccccc21', 'naphtho[2,3-d][1,2,3]triazole'),  #:13980
    ('C1=Cc2cocc2OC1', 'furo[3,4-b]pyran'),  #:14207
    ('C1=Cc2ccoc2CO1', 'furo[2,3-c]pyran'),  #:14211
    ('C1=CC=Nc2cc3nnccc3cc2N=C1', '[1,4]diazocino[2,3-g]cinnoline'),  #:14369
    ('C1=Cc2ccccc2N=Cc2cc3ccccc3cc21', 'naphtho[2,3-c][1]benzazocine'),  #:14377
    ('C1=COC2C=CC=PC2=C1', 'phosphinino[3,2-b]pyran'),  #:14547
    ('C1=CC=C2C=CSC=C2C=C1', 'cyclohepta[c]thiopyran'),  #:14559
    ('C1=CC2=NC=CC2=N1', 'pyrrolo[3,2-b]pyrrole'),  #:14571
    ('C1=c2ncccc2=NCS1', 'pyrido[3,2-d][1,3]thiazine'),  #:14581
    ('C1=COC2=C(C=C1)OCO2', '[1,3]dioxolo[4,5-b]oxepine'),  #:14589
    ('c1cnc2cc[nH]c2c1', 'pyrrolo[3,2-b]pyridine'),  #:14630
    ('c1scc2c1CSC2', 'thieno[3,4-c]thiophene'),  #:14638
    ('c1ccc2cc3cc4cc5c[nH]cc5cc4cc3cc2c1', 'anthra[2,3-f]isoindole'),  #:14655
    ('c1cnc2cc3ccc4ccccc4c3cc2c1', 'naphtho[1,2-g]quinoline'),  #:19763
    ('c1ccc2c(c1)ccc1c3cccnc3ccc21', 'naphtho[2,1-f]quinoline'),  #:19763
    ('c1opc2c1OCO2', '[1,3]dioxolo[4,5-c][1,2]oxaphosphole'),  #:19786
    ('c1ccc2c(c1)ccc1ccc3ncccc3c12', 'naphtho[1,2-f]quinoline'),  #:19792
    ('c1ccc2cc3c(ccc4ncccc43)cc2c1', 'naphtho[2,3-f]quinoline'),  #:19800
    ('c1ccc2cc3c(ccc4cnccc43)cc2c1', 'naphtho[2,3-f]isoquinoline'),  #:19800
    ('c1coc2coccoc-2c1', 'pyrano[3,2-e][1,4]dioxepine'),  #:20912
    ('C1=CC2=C(CO1)COSO2', 'pyrano[4,3-d][1,3,2]dioxathiine'),  #:21239
    ('C1=CC2=C(CO1)COOS2', 'pyrano[4,3-d][1,2,3]dioxathiine'),  #:21239
    ('C1=CC2=C(OC=CC2)OC1', 'pyrano[2,3-b]pyran'),  #:24649
    ('C1=CC2=COCC2=C1', 'cyclopenta[c]furan'),  #:24800
    ('C1=CC=CC=CC=CC=CC=CC=CN2C=CC=CC2=COCC=CC=CC=CC=CC=CC=C1', 'pyrido[2,1-c][1,4]oxaazacyclohentriacontine'),  #:24837
    ('C1=COC=c2occccoc2=COC=C1', '[1,4]dioxocino[2,3-c][1,6]dioxecine'),  #:32191
    ('C1=COC=C2OC=CC=COC=C2OC=C1', '[1,5]dioxonino[3,2-b][1,5]dioxonine'),  #:32195
    ('C1=CC2=COC=CC2=CO1', 'pyrano[4,3-c]pyran'),  #:32535
    ('c1scc2cc3c(cc12)COC3', 'thieno[3,4-f][2]benzofuran'),  #:32555
    ('B1Oc2nc[nH]c2O1', '[1,3,2]dioxaborolo[4,5-d]imidazole'),  #:37248
    ('B1=CC2=CC=CC2=C1', 'cyclopenta[c]borole'),  #:41183
    ('c1ccc2c(c1)[pH]c1ccccc12', 'benzo[b]phosphindole'),  #:41804
    ('C1=CC2=CN=CCN2C=C1', 'pyrido[1,2-a]pyrazine'),  #:41808
    ('C1=CC=C2NC=CC=C2C=C1', 'cyclohepta[b]pyridine'),  #:42203
    ('c1ccc2c3cn4ccccc4cc-3nc2c1', 'indolo[2,3-b]quinolizine'),  #:42431
    ('B1OC2=CC=CCN2O1', '[1,3,4,2]dioxazaborolo[4,5-a]pyridine'),  #:42435
    ('B1c2ccccc2-c2nc3ccccc3n21', '[1,3]benzimidazolo[1,2-b][2,1]benzazaborole'),  #:42437
    ('c1ccc2cc3c(cc2c1)O[SiH2]O3', 'naphtho[2,3-d][1,3,2]dioxasilole'),  #:46235
    ('c1ccc2c(c1)c1c(c3ccccc32)OPO1', 'phenanthro[9,10-d][1,3,2]dioxaphosphole'),  #:46286
    ('C1=COc2ccc3cc[nH]c3c2OC1', '[1,4]dioxepino[2,3-g]indole'),  #:49192
    ('C1=Cc2cc3cccc-3cn2C1', 'cyclopenta[f]indolizine'),  #:49192
]


@pytest.mark.parametrize("smiles,name", BOOK_FUSION_NAMES)
def test_the_book_fusion_names_are_rebuilt_by_the_rules(smiles, name):
    assert hetero_fusion.rule_name(_skeleton(smiles)) == name


#: the benzo names the book prints in PIN-marked names (rules study, section 4; the
#: lambda-convention names aside), on OPSIN 2.9.0's structure
BOOK_BENZO_NAMES = [
    ('C1=Cc2ccccc2C=CO1', '3-benzoxepine'),  #:7643
    ('C1=c2ccccc2=NCO1', '3,1-benzoxazine'),  #:7645
    ('c1ccc2sccc2c1', '1-benzothiophene'),  #:10160
    ('C1=Cc2ccccc2OC1', '1-benzopyran'),  #:10182
    ('c1ccc2c(c1)CSS2', '1,2-benzodithiole'),  #:10312
    ('c1ccc2c(c1)SCS2', '1,3-benzodithiole'),  #:10312
    ('C1=NOCc2ccccc21', '2,3-benzoxazine'),  #:10318
    ('c1ccc2c(c1)OCO2', '1,3-benzodioxole'),  #:10756
    ('C1=COc2ccccc2O1', '1,4-benzodioxine'),  #:10770
    ('c1ccc2occc2c1', '1-benzofuran'),  #:11827
    ('c1ccc2cocc2c1', '2-benzofuran'),  #:11829
    ('C1=CC=COC=CC=Cc2ccccc2C=CC=COC=C1', '5,12-benzodioxacyclooctadecine'),  #:11831
    ('C1=CC=Cc2ccccc2C=CNC=C1', '3-benzazacycloundecine'),  #:11835
    ('C1=COC=c2ccccc2=CSC=CN=C1', '9,2,5-benzoxathiaazacyclododecine'),  #:11839
    ('c1ccc2[nH]cnc2c1', '1,3-benzimidazole'),  #:12600
    ('c1ccc2nscc2c1', '2,1-benzothiazole'),  #:13443
    ('c1ccc2c(c1)N=NCO2', '4,1,2-benzoxadiazine'),  #:14327
    ('C1=Cc2ccccc2CO1', '2-benzopyran'),  #:14357
    ('C1=CC=Nc2ccccc2C=C1', '1-benzazocine'),  #:14377
    ('c1ccc2c(c1)OCS2', '1,3-benzoxathiole'),  #:14575
    ('C1=COC=c2ccccc2=C1', '2-benzoxepine'),  #:14931
    ('C1=Cc2ccccc2SC1', '1-benzothiopyran'),  #:17006
    ('C1=Cc2ccccc2[Se]C1', '1-benzoselenopyran'),  #:17008
    ('C1=Cc2ccccc2[Te]C1', '1-benzotelluropyran'),  #:17010
    ('C1=Cc2ccccc2CS1', '2-benzothiopyran'),  #:17018
    ('C1=Cc2ccccc2C[Se]1', '2-benzoselenopyran'),  #:17020
    ('C1=Cc2ccccc2C[Te]1', '2-benzotelluropyran'),  #:17022
    ('C1=[As]c2ccccc2[As]=CO1', '3,1,5-benzoxadiarsepine'),  #:20912
    ('C1=[SiH]O[SiH2]c2ccccc21', '2,1,3-benzoxadisiline'),  #:21209
    ('C1=[SiH]c2ccccc2[SiH2]O1', '2,1,4-benzoxadisiline'),  #:21209
    ('C1=CNC=c2ccccc2=C1', '2-benzazepine'),  #:24774
    ('c1ccc2c(c1)[O][GeH2][O]2', '1,3,2-benzodioxagermole'),  #:28202
    ('C1=c2ccccc2=C[O][PbH2][O]1', '2,4,3-benzodioxaplumbepine'),  #:31587
    ('c1ccc2coccocc2c1', '2,5-benzodioxocine'),  #:32187
    ('c1ccc2cscc2c1', '2-benzothiophene'),  #:32546
    ('c1ccc2c(c1)[O][BiH][O]2', '1,3,2-benzodioxabismole'),  #:39298
    ('B1C=Cc2ccccc21', '1-benzoborole'),  #:41122
    ('c1ccc2scnc2c1', '1,3-benzothiazole'),  #:43538
    ('C1=CSc2ccccc2N=C1', '1,5-benzothiazepine'),  #:49307
    # printed as bracketed components of fusion and spiro PINs
    ('O1COCC2=C1C=CC=C2', '1,3-benzodioxine'),  #:14462 '[1,3]benzodioxino'
    ('O1SOC2=C1C=CC=C2', '1,3,2-benzodioxathiole'),  #:10913
    ('N1PN=CC2=C1C=CC=C2', '1,3,2-benzodiazaphosphinine'),  #:10917
    ('O1PNC2=C1C=CC=C2', '1,3,2-benzoxazaphosphole'),  #:11230
    ('O1SSC2=C1C=CC=C2', '1,2,3-benzoxadithiole'),  #:11250
    ('B1N=CC2=C1C=CC=C2', '2,1-benzazaborole'),  #:42437
    ('O1POC2=C1C=CC=C2', '1,3,2-benzodioxaphosphole'),  #:46227
    ('S1OCC2=C1C=CC=C2', '2,1-benzoxathiole'),  #:46247
    ('c1ccc2c(c1)cno2', '1,2-benzoxazole'),  #:13449
]


@pytest.mark.parametrize("smiles,name", BOOK_BENZO_NAMES)
def test_the_book_benzo_names_are_rebuilt_by_the_rules(smiles, name):
    # (:11815): heteroatom locants always cited, the 'o' of 'benzo' elided before
    # a vowel, the heterocyclic ring numbered first from an atom next to a fusion atom
    assert hetero_fusion.rule_name(_skeleton(smiles)) == name


@pytest.mark.parametrize("smiles,why", [
    ("C1=CC2=CC=CC3C=CC=C[PH]23C=C1", "5lambda5-phosphinino[2,1-d]phosphinolizine :12476: P in three rings"),
    ("C1=CC2=CC=CC3=NC23C=C1", "naphtho[1,8a-b]azirine :14011: C8a in three rings"),
    ("C1=CC2=CCS[SH]2S1", "7lambda4-[1,2]dithiolo[5,1-e][1,2]dithiole :14545: a lambda4 sulfur"),
])
def test_the_book_names_outside_the_class_bounds_are_declined(smiles, why):
    # an atom common to three rings, and the lambda-convention:12452), are
    # outside the S2c-1 class (slice S2c-3); the producer never guesses them
    assert hetero_fusion.rule_name(_skeleton(smiles)) is None, why


@pytest.mark.parametrize("element", ["Be", "Mg", "Ca", "Sr", "Ba", "Po", "Ra"])
def test_a_ring_atom_outside_the_a_prefix_order_is_declined(element):
    # (:8284) and (a) (:12139) order the ring heteroatoms these names
    # cite, F... Tl; a ring atom of another element is declined, never an exception (RDKit's
    # default valence of these elements equals the ring degree, so the lambda test passes them)
    from orthonym.rules.bridged_fused_pin import fusion_components as fc
    key = _skeleton(f"C1=CC=C2C(=C1)C=C[{element}]2")
    assert hetero_fusion.rule_name(key) is None
    rings = fc.ortho_rings(key)
    ring = next(r for r in rings if len(r) == 5)
    assert fc.mono_numberings(key, ring, hw=True) == []
    comp = fc.Component("hetero", "x", frozenset([rings.index(ring)]), ring, ())
    assert fc.rank_key(key, comp, rings) is None


@pytest.mark.parametrize("smiles,why", [
    ("c1ccc2c(c1)Cc1cccc3c1N2CC3", "pyrrolo[3,2,1-de]acridine :6914: peri"),
    ("C1=Cc2cocc21", "cyclobuta[1,2-c]furan :7186: one ring of five or more members (P-52.2.4.1)"),
    ("C1=Cc2cccc3c2N2C1=CCC=C2C=C3", "quinolizino[3,4,5,6-ija]quinoline :12608: peri"),
    ("c1cc2ccc3ccnc4ccc(c1)c2c34", "naphtho[2,1,8-def]quinoline :14003: peri"),
    ("C1=Cc2coc3cccc1c23", "indeno[7,1-bc]furan :14347: peri"),
    ("c1cc2c3c(cccc3c1)SO2", "naphtho[1,8-cd][1,2]oxathiole :32160: peri"),
    ("c1cc2c3c(cccc3c1)COC2", "benzo[de][2]benzopyran :32498: peri"),
    ("C1=NCN2CCCN3CNC1=C23", "pyrimido[1,2,3-cd]purine :41706: peri"),
    ("c1cc2cc3occc3cc2o1", "benzo[1,2-b:4,5-b']difuran: a multiparent name (P-25.3.7, :13453)"),
    ("c1cc2cc3cocc3cc2o1", "benzo[1,2-b:4,5-c']difuran (PIN) :13453, not furo[3,4-f][1]benzofuran"),
    ("C1=NC=c2cc3c(cc21)=CN=C3", "benzo[1,2-c:4,5-c']dipyrrole :41087, not pyrrolo[3,4-f]isoindole"),
    ("c1ccc2c(c1)oc1ccccc12", "dibenzo[b,d]furan :7475: multiplied 'benzo'"),
    ("c1ccc2c(c1)COc1ccccc1-2", "6H-dibenzo[b,d]pyran :13463, not benzo[c][1]benzopyran"),
    ("C1=Cc2ccccc2Oc2ccccc21", "dibenzo[b,f]oxepine: multiplied 'benzo'"),
    ("c1cnc2cc3ncccc3cc2c1", "two pyridine rings across a benzene ring: the multiparent reading of :41087"),
    ("C1=Cc2c(ccc3oc4ncccc4c23)C1", "three components (P-25.3.4)"),
])
def test_other_slices_are_declined(smiles, why):
    assert hetero_fusion.rule_name(_skeleton(smiles)) is None, why


@pytest.mark.parametrize("smiles,name", [
    ("c1ccc2ncccc2c1", "quinoline"), ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("c1ccc2c(c1)[nH]c1ccccc12", "9H-carbazole"), ("c1ccc2nc3ccccc3cc2c1", "acridine"),
    ("c1ncc2[nH]cnc2n1", "7H-purine"), ("c1cnc2ncncc2n1", "pteridine"),
    ("c1ccc2nc3ccccc3nc2c1", "phenazine"), ("c1ccn2cccc2c1", "indolizine"),
    ("C1=CC2=CC=CN2C1", "1H-pyrrolizine"),
    ("c1cnc2ncccc2c1", "1,8-naphthyridine"), ("c1ccc2c(c1)Oc1ccccc1S2", "phenoxathiine"),
    ("c1ccc2[nH]ncc2c1", "1H-indazole"), ("c1ccc2[pH]ccc2c1", "1H-phosphindole"),
])
def test_skeletons_with_a_one_component_name_get_no_fusion_name(smiles, name):
    # (:11903) fusion names only for systems with no retained or systematic name;
    #:11628 "the PIN is 1H-pyrrolizine" (not pyrrolo[1,2-a]pyrrole)
    key = _skeleton(smiles)
    assert hetero_fusion.hetero_component_name(key) is None, name


#: (:11982), Note (:11909), Table 2.8 (:11656 chromene),
#: (:11815 locants cited;:11829 isobenzofuran, benzo[c]furan), (:12045
#: quino/isoquino general only): spellings the book prints beside the PINs, never produced
_ALWAYS_BAD = ("isoxazol", "isothiazol", "isoselenazol", "isotellurazol", "benzis", "benz[",
               "chromen", "isobenzofuran", "benzo[c]furan", "benzo[c]thiophene", "cyclohexa[",
               "annuleno")
_LOCANT_BRACKET = re.compile(r"\[\d+(?:,\d+)*\]$")


def _non_pin_spellings(name):
    bad = [t for t in _ALWAYS_BAD if t in name]
    bad += re.findall(r"(?<![a-z])(?:iso)?quino\[", name)
    for m in re.finditer(r"(?:ox|thi|selen|tellur)azol", name):
        before = name[:m.start()]
        if _LOCANT_BRACKET.search(before) or (before[-1:].isalpha() and not before.endswith("benz")):
            continue                     # '[1,3]thiazole', or inside a longer stem ('oxathiazole')
        if before.endswith("benz") and re.search(r"(?:\d+(?:,\d+)*-|\[\d+(?:,\d+)*\])benz$", before):
            continue                     # '1,3-benzothiazole', '[1,2]benzoxazolo'
        bad.append(name[m.start():m.end()])
    for m in re.finditer(r"benz(?!o\[)", name):
        before = name[:m.start()]
        if not re.search(r"(?:\d+(?:,\d+)*-|\[\d+(?:,\d+)*\])$", before):
            bad.append("benzo name without locants")
    return bad


@pytest.mark.parametrize("name,bad", [
    ("oxazolo[4,5-b]pyridine", True), ("isoxazolo[4,5-c]pyridine", True),
    ("imidazo[2,1-b]thiazole", True), ("1H-benzimidazole", True), ("1,2-benzisoxazole", True),
    ("benzo[c]thiophene", True), ("benz[g]isoquinoline", True), ("chromeno[2,3-c]pyrrole", True),
    ("[1,3]oxazolo[4,5-b]pyridine", False), ("imidazo[2,1-b][1,3]thiazole", False),
    ("1H-1,3-benzimidazole", False), ("[1,2]benzoxazolo[6,5-g]quinoline", False),
    ("[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole", False), ("benzo[g]quinoline", False),
])
def test_the_spelling_scan_has_teeth(name, bad):
    assert bool(_non_pin_spellings(name)) is bad, _non_pin_spellings(name)


def test_no_produced_name_has_a_non_pin_spelling():
    produced = [hetero_fusion.rule_name(_skeleton(s)) for s, _ in BOOK_FUSION_NAMES + BOOK_BENZO_NAMES]
    extra = ["c1ccc2cscc2c1", "c1ccc2c(c1)cno2", "c1ccc2c(c1)ocn2", "c1cnc2c(c1)ocn2",
             "c1cnc2c(c1)nco2", "c1cn2ccsc2n1", "c1ccc2nc3ccccc3cc2c1"]
    produced += [hetero_fusion.rule_name(_skeleton(s)) for s in extra]
    bad = [(n, _non_pin_spellings(n)) for n in produced if n and _non_pin_spellings(n)]
    assert not bad, bad


@pytest.mark.parametrize("smiles,name", [
    ("c1ccc2cscc2c1", "2-benzothiophene"),            #:32546 prints benzo[c]thiophene too;:11815
    ("c1ccc2c(c1)cno2", "1,2-benzoxazole"),           # not 1,2-benzisoxazole (:11982,:13449)
    ("c1ccc2[nH]cnc2c1", "1,3-benzimidazole"),        # not benzimidazole (:11815,:12600)
    ("c1ccc2[nH]nnc2c1", "1,2,3-benzotriazole"),      # (:11815), no book row
    ("c1cnc2c(c1)ocn2", "[1,3]oxazolo[4,5-b]pyridine"),   # not oxazolo[4,5-b]pyridine
    ("c1cnc2c(c1)nco2", "[1,3]oxazolo[5,4-b]pyridine"),
    ("c1cn2ccsc2n1", "imidazo[2,1-b][1,3]thiazole"),      # not imidazo[2,1-b]thiazole
    ("c1cnc2onc(c2c1)", "[1,2]oxazolo[5,4-b]pyridine"),   # not isoxazolo[5,4-b]pyridine
    ("c1ccc2c(c1)ccc1cccnc12", "benzo[h]quinoline"),      # a fusion name
    ("c1ccc2c(c1)ccc1ncccc12", "benzo[f]quinoline"),
    ("c1cnc2cnncc2n1", "pyrazino[2,3-d]pyridazine"),      #:12403, (h)
    ("c1ccc2c(c1)[nH]c1cnccc12", "pyrido[3,4-b]indole"),  # the tables' '9H-beta-carboline' is no PIN
])
def test_the_spellings_the_engine_got_wrong(smiles, name):
    assert hetero_fusion.rule_name(_skeleton(smiles)) == name


@pytest.mark.parametrize("smiles", [
    "c1cnc2cnncc2n1", "c12cnncc1nccn2", "n1ccnc2cnncc12", "c1nncc2nccnc12",
    "c1ccc2c(c1)ccc1cccnc12", "c1cc2ccsc2o1", "c1ccc2c(c1)[nH]c1cc3nccnc3cc12",
])
def test_the_name_does_not_depend_on_the_atom_order(smiles):
    # the rules read the ring graph only,
    import random
    mol = Chem.MolFromSmiles(smiles)
    names = set()
    for seed in range(6):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        names.add(hetero_fusion.rule_name(_skeleton(shuffled)))
    assert len(names) == 1 and None not in names, names


@pytest.mark.opsin_gate
def test_the_parent_is_numbered_by_opsins_reading_of_the_name():
    # '[1,2]oxazolo[4,5-c]pyridine': O1, N2,... N5 (OPSIN $_AV); the parent source maps the
    # locants onto the skeleton and names its source
    key = _skeleton("c1ncc2c(c1)onc2")
    got = parents._parent_for_key(Chem.MolToSmiles(key))
    assert got is not None and got[0] == "[1,2]oxazolo[4,5-c]pyridine", got
    assert got[2] == "hetero_fusion_name+opsin"
    locs = dict(got[1])
    het = sorted(str(locs[a.GetIdx()]) for a in key.GetAtoms() if a.GetAtomicNum() != 6)
    assert het == ["1", "2", "5"], het


@pytest.mark.opsin_gate
def test_two_indicated_hydrogens_are_searched():
    #:21239 '4H,5H-pyrano[4,3-d][1,2,3]dioxathiine' needs two indicated hydrogens, which the
    # carbocyclic producer's forms ('', '1H-', '2H-') do not offer
    from orthonym.rules.bridged_fused_pin import fusion_names
    key = _skeleton("C1=CC2=C(CO1)COOS2")
    name = "pyrano[4,3-d][1,2,3]dioxathiine"
    assert fusion_names.opsin_structure(name, key.GetNumAtoms()) is None
    struct = hetero_fusion.opsin_structure(name, key.GetNumAtoms())
    assert struct is not None and len(struct[1]) == key.GetNumAtoms()


@pytest.fixture
def _opsin_reads_nothing(monkeypatch):
    # OPSIN unavailable to name construction (or a name it rejects): the in-process helper
    # returns None for every name
    from orthonym.rules.bridged_fused_pin import fusion_names
    from orthonym.validation import opsin_roundtrip
    monkeypatch.setattr(opsin_roundtrip, "extended_smiles_or_unavailable",
                        lambda name, jar_version="2.9.0": None)
    caches = (fusion_names.opsin_structure, hetero_fusion.opsin_structure, parents._parent_for_key)
    for f in caches:
        f.cache_clear()
    yield
    for f in caches:
        f.cache_clear()


def test_without_opsins_reading_the_parent_declines(_opsin_reads_nothing):
    # fail closed: the rules give the name, but without OPSIN's structure there is no numbering,
    # so the parent source returns no parent and the builder names nothing
    key = _skeleton("c1ncc2c(c1)onc2")
    assert hetero_fusion.hetero_component_name(key) == "[1,2]oxazolo[4,5-c]pyridine"
    assert parents._parent_for_key(Chem.MolToSmiles(key)) is None


#: (1) (the Blue Book) read literally on a parent of this producer: the
#: bridged fused reading cuts a carbon bridge out of a ring system whose atoms are all sp2
#: and that a chain-linked ring crosses at nonadjacent positions, so the compound is a
#: cyclophane:23843 "cyclic phane systems > fused ring systems > bridged fused
#: systems") and the builder declines the reading
CYCLOPHANE_SHAPED = [
    ("C1=CC=C2C(=C1)OC3=CC=C(C=C3)OOS2",
     "a para-phenylene linked by -O- and -O-O-S-, cut into an etheno bridge"),
    ("C1C=CC=CN=CC=NC=C2C=CC=CC2=NC3=NC=NC4=C3C=C(O1)C=C4",
     "an etheno bridge cut out of a benzene ring that a heteroatom-linked macrocycle crosses"),
    ("C/C1=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](O)[C@H](C)[C@@H](O)[C@@H](C)"
     "/C=C(\\C)C(=O)c2c(O)c(C)cc3c2C(=O)C=C(NC1=O)C3=O",
     "a quinone ring of an ansa macrocycle cut into a methano bridge"),
]


@pytest.mark.parametrize("smiles,why", CYCLOPHANE_SHAPED)
def test_a_cyclophane_shaped_reading_on_a_produced_parent_is_declined(smiles, why, monkeypatch):
    import orthonym.rules.bridged_fused_pin as bfp
    from orthonym.rules.bridged_fused_pin import selection
    mol = Chem.MolFromSmiles(smiles)
    assert bfp.build(mol) is None, why
    # the decline is the phane check's: without it the produced parent gives a name
    monkeypatch.setattr(selection, "_on_a_produced_parent", lambda mol, split: False)
    assert bfp.build(mol) is not None, why


@pytest.mark.opsin_gate
def test_a_cyclophane_shaped_compound_keeps_its_best_effort_name():
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from orthonym.jvm_budget import jvm_slots
    from tests.support.rt_assert import name_is_rt_exact
    smiles = CYCLOPHANE_SHAPED[0][0]
    with jvm_slots(1, purpose="s2c1-phane"):
        pin = Orthonym().name_tiered(smiles)
        be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert pin["tier"] == "abstain"
    assert "etheno" not in be["name"] and name_is_rt_exact(be["name"], smiles), be["name"]
