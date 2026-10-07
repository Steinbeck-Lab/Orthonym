"""The fusion components of a skeleton (``bridged_fused_pin.fusion_components``): component
names, their own numberings, attached-component prefixes and the seniority, each
checked on the Blue Book's own examples.

 (the Blue Book) Hantzsch-Widman numbering and citation order; Table 2.2
(:8111-:8178) retained names; (:11982) '1,2-oxazole' and '1,3-thiazole' as
components, (:11984) and (:11735) the 'ine' names of larger heteromonocycles;
 (:11905) and (:12024) prefixes, (:12047) the contracted
ones, (:12000) 'cyclo...a'; (:11815) benzo names. The seniority of
 (a)-(j) (:12139-:12418) is tested with the producer (``test_bf_s2c1_hetero_fusion``)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import fusion_components as fc
from orthonym.rules.bridged_fused_pin import parents


def _key(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


def _mono_name(smiles):
    km = _key(smiles)
    ring = fc.ortho_rings(km)[0]
    nums = fc.mono_numberings(km, ring, hw=len(ring) <= 10)
    return {fc.hetero_mono_name(km, ring, n) for n in nums}


@pytest.mark.parametrize("smiles,name", [
    ("c1cocn1", "1,3-oxazole"), ("c1cnoc1", "1,2-oxazole"), ("c1cscn1", "1,3-thiazole"),
    ("c1c[se]cn1", "1,3-selenazole"), ("C1=COCO1", "1,3-dioxole"), ("B1OC=CO1", "1,3,2-dioxaborole"),
    ("O1SNC=C1", "1,2,3-oxathiazole"), ("O1NSC=C1", "1,3,2-oxathiazole"), ("O1C=NN=C1", "1,3,4-oxadiazole"),
    ("C1=COC=C[Se]1", "1,4-oxaselenine"), ("C1=COC=CS1", "1,4-oxathiine"), ("c1ccpcc1", "phosphinine"),
    ("C1=CC=COC=C1", "oxepine"), ("C1=CSC=CSC1", "1,4-dithiepine"), ("C1=CC=NC=CC=N1", "1,5-diazocine"),
    ("C1=COC=CC=COC1", "1,5-dioxonine"), ("C1=COC=CC=COC=C1", "1,6-dioxecine"),
    ("N1C=C1", "azirine"), ("C1=CO1", "oxirene"), ("B1C=CC=C1", "borole"),
    ("c1ccoc1", "furan"), ("c1ccsc1", "thiophene"), ("c1cc[nH]c1", "pyrrole"), ("c1ccncc1", "pyridine"),
    ("C1=CCOC=C1", "pyran"), ("C1=CCSC=C1", "thiopyran"), ("c1cc[se]c1", "selenophene"),
    ("c1c[nH]cn1", "imidazole"), ("c1cn[nH]c1", "pyrazole"), ("c1cncnc1", "pyrimidine"),
    ("c1cnccn1", "pyrazine"), ("c1ccnnc1", "pyridazine"), ("c1cn[nH]n1", "1,2,3-triazole"),
])
def test_heteromonocycle_component_names(smiles, name):
    # Table 2.2 retained names (never 'oxazole', 'isoxazole', 'thiazole', 'isothiazole'::11982),
    # otherwise the Hantzsch-Widman name of the mancude ring
    assert _mono_name(smiles) == {name}


def test_the_large_heteromonocycles_take_the_ine_names():
    #:24837 'pyrido[2,1-c][1,4]oxaazacyclohentriacontine',:12149
    # '[1,4,7,10,13,16]hexaoxacyclohenicosine',:11835 '1H-3-benzazacycloundecine'
    assert _mono_name("O1CCN" + "C" * 27 + "1") == {"1,4-oxaazacyclohentriacontine"}
    assert _mono_name("O1CCOCCOCCOCCOCCOCCCCC1") == {"1,4,7,10,13,16-hexaoxacyclohenicosine"}
    assert _mono_name("N1C=CC=CC=CC=CC=C1") == {"1-azacycloundecine"}


@pytest.mark.parametrize("comp,prefix", [
    (fc.Component("hetero", "1,2-oxazole", frozenset([0]), frozenset(range(5)), ()), "[1,2]oxazolo"),
    (fc.Component("hetero", "1,3-dioxole", frozenset([0]), frozenset(range(5)), ()), "[1,3]dioxolo"),
    (fc.Component("hetero", "furan", frozenset([0]), frozenset(range(5)), ()), "furo"),
    (fc.Component("hetero", "thiophene", frozenset([0]), frozenset(range(5)), ()), "thieno"),
    (fc.Component("hetero", "pyridine", frozenset([0]), frozenset(range(6)), ()), "pyrido"),
    (fc.Component("hetero", "pyrimidine", frozenset([0]), frozenset(range(6)), ()), "pyrimido"),
    (fc.Component("hetero", "imidazole", frozenset([0]), frozenset(range(5)), ()), "imidazo"),
    (fc.Component("hetero", "pyrazole", frozenset([0]), frozenset(range(5)), ()), "pyrazolo"),
    (fc.Component("hetero", "pyran", frozenset([0]), frozenset(range(6)), ()), "pyrano"),
    (fc.Component("hetero", "selenophene", frozenset([0]), frozenset(range(5)), ()), "selenopheno"),
    (fc.Component("hetero", "1,4-oxaselenine", frozenset([0]), frozenset(range(6)), ()), "[1,4]oxaselenino"),
    (fc.Component("retained", "naphthalene", frozenset([0, 1]), frozenset(range(10)), ()), "naphtho"),
    (fc.Component("retained", "anthracene", frozenset([0, 1, 2]), frozenset(range(14)), ()), "anthra"),
    (fc.Component("retained", "phenanthrene", frozenset([0, 1, 2]), frozenset(range(14)), ()), "phenanthro"),
    (fc.Component("retained", "quinoline", frozenset([0, 1]), frozenset(range(10)), ()), "quinolino"),
    (fc.Component("retained", "azulene", frozenset([0, 1]), frozenset(range(10)), ()), "azuleno"),
    (fc.Component("retained", "indole", frozenset([0, 1]), frozenset(range(9)), ()), "indolo"),
    (fc.Component("carbo", "benzene", frozenset([0]), frozenset(range(6)), ()), "benzo"),
    (fc.Component("carbo", "cyclo5", frozenset([0]), frozenset(range(5)), ()), "cyclopenta"),
    (fc.Component("carbo", "cyclo7", frozenset([0]), frozenset(range(7)), ()), "cyclohepta"),
    (fc.Component("benzo", "1-benzopyran", frozenset([0, 1]), frozenset(range(10)), ()), "[1]benzopyrano"),
    (fc.Component("benzo", "1,2-benzoxazole", frozenset([0, 1]), frozenset(range(9)), ()), "[1,2]benzoxazolo"),
    (fc.Component("benzo", "1,3-benzimidazole", frozenset([0, 1]), frozenset(range(9)), ()), "[1,3]benzimidazolo"),
])
def test_attached_component_prefixes(comp, prefix):
    #:12047 contracted prefixes (furo, thieno, pyrido, pyrimido, imidazo, naphtho, anthra,
    # phenanthro, benzo; 'quino' and 'isoquino' are general nomenclature only); otherwise
    # 'e' -> 'o' or '+o' (:11905,:12024: 'pyrazolo', 'selenopyrano'); the book's components
    # '[1]benzopyrano':12157, '[1,2]benzoxazolo':13449, '[1,3]benzimidazolo':42437
    assert fc.prefix_form(comp) == prefix


@pytest.mark.parametrize("name", ["1-benzofuran", "2-benzofuran", "1-benzothiophene"])
def test_benzofuran_and_benzothiophene_prefixes_are_not_certified(name):
    # the book prints no PIN with '[1]benzofuro' or '[1]benzofurano' (its only such spelling,
    #:13461, is a rejected name); the producer declines rather than guess
    comp = fc.Component("benzo", name, frozenset([0, 1]), frozenset(range(9)), ())
    assert fc.prefix_form(comp) is None


def test_hantzsch_widman_numbering_gives_one_to_the_senior_heteroatom():
    # (:8284): '1,3,2-dioxaborole' (O1, B2, O3), '1,3,4-oxadiazole' (O1, N3, N4)
    for smiles, het in (("O1BOC=C1", {"O": [1, 3], "B": [2]}), ("O1C=NN=C1", {"O": [1], "N": [3, 4]})):
        km = _key(smiles)
        ring = fc.ortho_rings(km)[0]
        for num in fc.mono_numberings(km, ring, hw=True):
            got = {}
            for a, loc in num.items():
                sym = km.GetAtomWithIdx(a).GetSymbol()
                if sym != "C":
                    got.setdefault(sym, []).append(loc)
            assert {k: sorted(v) for k, v in got.items()} == het


@pytest.mark.parametrize("smiles,name", [
    ("O1COC2=C1C=CC=C2", "1,3-benzodioxole"),      #:14624 '2H-1,3-benzodioxole'
    ("N1=COCC2=C1C=CC=C2", "3,1-benzoxazine"),     #:11823 '4H-3,1-benzoxazine (PIN)'
    ("C1=NC=CC=CC2=C1C=CC=C2", "2-benzazocine"),
    ("c1ccc2c(c1)cno2", "1,2-benzoxazole"),         #:13449
    ("C1=CC=Nc2ccccc2C=C1", "1-benzazocine"),       #:14377
])
def test_benzo_name_numbering(smiles, name):
    # (:11815) "The locant '1' is always assigned to the atom of the heterocyclic
    # component next to a fusion atom. Heteroatoms are allocated lowest locants as a set
    #... The letter 'o' of the 'benzo' prefix is elided when followed by a vowel"
    km = _key(smiles)
    rings = fc.ortho_rings(km)
    benzene = next(r for r in rings if all(km.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in r))
    hetero = next(r for r in rings if r is not benzene)
    nums = fc.benzo_numberings(km, benzene, hetero)
    assert nums and {fc.benzo_name(km, benzene, hetero, n) for n in nums} == {name}
