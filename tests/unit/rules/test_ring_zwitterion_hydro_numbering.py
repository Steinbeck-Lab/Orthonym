"""A zwitterion whose ring cation and ring carboxylate share a partially saturated (hydro) ring is
named in ONE numbering: the 'ium' at the lowest locant, then the carboxylate at the lowest locant it
can have, then the hydro prefixes and the substituents.

Ectoine (the cyclic amidinium of 2-methyl-1,4,5,6-tetrahydropyrimidine-4-carboxylic acid) and
hydroxyectoine had no name at the default tier. ``emit_zwitterion_ring_carboxylate`` took the stem
from the neutral bare ring ('1,4,5,6-tetrahydropyrimidine', its own numbering) and the '-ium' and
'-carboxylate' locants from the charged ring ('-1-ium-6-carboxylate'): two numberings in one name,
another species once OPSIN read it ('2-methyl1,4,5,6-tetrahydropyrimidin-1-ium-6-carboxylate' is the
protonated amine), and no hyphen after the prefix. rejected it.

 "Zwitterionic compounds with at least one ionic center on a characteristic group"
(the Blue Book),:42447: "Zwitterionic compounds with at least one ionic center on a
characteristic group may be named by adding the appropriate ionic suffix to the name of the ionic
parent hydride. In names, cationic suffixes are cited before anionic suffixes. For assignment of
lower locants, ionic centers on skeletal atoms of the parent hydride are preferred to the locants
for positions of attachment of characteristic groups denoted by ionic suffixes." The example is
'1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate (PIN)' (:42456). Both are suffixes, so
"NUMBERING" (:3219) ranks them by (c) "principal characteristic groups and free valences
(suffixes)" (:3256), after (b) indicated hydrogen (:3246) and before (e) "(i) low locants are given
to hydro/dehydro prefixes" (:3288-:3289) and (f) the detachable alphabetized prefixes (:3301).

The charge forms of an amidinium (the hydron on the nitrogen near or far from the carboxylate) are one
species; each is named and the form with the lower locants stands (``Orthonym._name_with_form_choice``).
The carboxylate locant is a suffix locant of (c): '2,6-dimethyl-...-1-ium-4-carboxylate' stands over
'2,4-dimethyl-...-1-ium-6-carboxylate' although the prefix locants are the same set.

A cation that holds no hydron (``N+(CH3)=C``, the methyl on the nitrogen that takes the 'ium') is the neutral
parent plus a hydron, then substituted, the Blue Book; '1-methyl-4,6-diphenylpyridin-1-ium-2-
carboxylate (PIN)',:42456), so it is named in one numbering too, and of the two charge forms of one species
the one that gives the carboxylate the lower locant stands: when that form is the quaternary one, a producer
that cannot name it leaves the other form, a numbering that is not the PIN. The same holds for an azolium whose
indicated-hydrogen nitrogen carries the substituent ('1,2-dimethyl-1H-imidazol-3-ium-4-carboxylate').

Every name is read back by a FRESH OPSIN call and must give the input's key and fixed-H InChI.
"""
import pytest

from tests.support.jars import jar_or_skip
from tests.support.read_back import same_hydron_sites

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _named(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert same_hydron_sites(name, smiles)


ECTOINE = "(4S)-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"
N_METHYL_AWAY = "1,2-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"
HYDROXYECTOINE = "(4S,5S)-5-hydroxy-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"


@pytest.mark.parametrize("smiles,name", [
    # ectoine, the two charge forms of one species (the hydron beside or away from the carboxylate)
    ("CC1=[NH+][C@H](C(=O)[O-])CCN1", ECTOINE),
    ("CC1=[NH+]CC[C@H](N1)C(=O)[O-]", ECTOINE),
    ("CC1=[NH+][C@@H](C(=O)[O-])CCN1", "(4R)-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+]C(C(=O)[O-])CCN1", "2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("C1=[NH+]C(C(=O)[O-])CCN1", "3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    # hydroxyectoine and its diastereomers: the hydro prefixes stand before the parent, the
    # alphabetized prefixes before them, the descriptors at the front in this numbering
    ("CC1=[NH+][C@H](C(=O)[O-])[C@@H](O)CN1", HYDROXYECTOINE),
    ("CC1=[NH+][C@@H](C(=O)[O-])[C@@H](O)CN1",
     "(4R,5S)-5-hydroxy-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+][C@@H](C(=O)[O-])[C@H](O)CN1",
     "(4R,5R)-5-hydroxy-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+]C(C(=O)[O-])C(O)CN1", "5-hydroxy-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    # the carboxylate is a suffix of (c): it takes the lower locant (4, not 6) before the prefixes (f)
    ("CC1=[NH+]C(C(=O)[O-])CC(C)N1", "2,6-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+]C(CC(C)N1)C(=O)[O-]", "2,6-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+]C(C(=O)[O-])CC(CC)N1", "6-ethyl-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+]C(C(=O)[O-])CC(O)N1", "6-hydroxy-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    # a methyl on one ring nitrogen. Both charge forms are one species and are named alike. The
    # methyl away from the carboxylate (the quaternary form puts the 'ium' on it, the parent plus a
    # hydron, then substituted, as '1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate'): 'ium' 1 in
    # either form, then the carboxylate at 4, not 6:42447, (c):3256)
    ("CC1=[NH+]C(C(=O)[O-])CCN1C", N_METHYL_AWAY),
    ("CC1=[N+](C)CCC(C(=O)[O-])N1", N_METHYL_AWAY),
    ("CC1=[NH+][C@H](C(=O)[O-])CCN1C", "(4S)-" + N_METHYL_AWAY),
    ("CC1=[N+](C)CC[C@@H](C(=O)[O-])N1", "(4S)-" + N_METHYL_AWAY),
    ("CC1=[NH+][C@@H](C(=O)[O-])CCN1C", "(4R)-" + N_METHYL_AWAY),
    # the methyl beside the carboxylate: the unsubstituted nitrogen takes the 'ium' (1) and the
    # substituted one is 3, the carboxylate beside it 4, where the quaternary form would be 6
    ("CC1=[N+](C)C(C(=O)[O-])CCN1", "2,3-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
    ("CC1=[NH+]CCC(C(=O)[O-])N1C", "2,3-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate"),
])
def test_ectoine_class_is_named_in_one_numbering(smiles, name):
    _named(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # other rings and heteroatoms: indicated hydrogen (b) first, then the suffixes (c), then hydro (e)
    ("CC1=[NH+]C(C(=O)[O-])CN1", "2-methyl-4,5-dihydro-1H-imidazol-3-ium-4-carboxylate"),
    ("CC1=[NH+]C(C(=O)[O-])CCCN1", "2-methyl-4,5,6,7-tetrahydro-1H-1,3-diazepin-3-ium-4-carboxylate"),
    ("O=C([O-])C1=[NH+]CCC1", "3,4-dihydro-2H-pyrrol-1-ium-5-carboxylate"),
    ("O=C([O-])C1=[NH+]CCCC1", "3,4,5,6-tetrahydropyridin-1-ium-2-carboxylate"),
    ("O=C([O-])C1=[NH+]CCCCC1", "3,4,5,6-tetrahydro-2H-azepin-1-ium-7-carboxylate"),
    ("CC1=[NH+]C(C(=O)[O-])CS1", "2-methyl-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    # a methyl on the nitrogen beside the carboxylate: the quaternary form is the PIN ('ium' 3, the
    # carboxylate 4), the form with the hydron there would give the carboxylate 5; both drawings
    ("CC1=[N+](C)C(C(=O)[O-])CN1", "2,3-dimethyl-4,5-dihydro-1H-imidazol-3-ium-4-carboxylate"),
    ("CC1=[NH+]CC(C(=O)[O-])N1C", "2,3-dimethyl-4,5-dihydro-1H-imidazol-3-ium-4-carboxylate"),
    # 2-amino-thiazoline-4-carboxylate: the amino group is a prefix, the hyphen follows it
    ("NC1=[NH+]C(C(=O)[O-])CS1", "2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    ("NC1=[NH+][C@H](C(=O)[O-])CS1", "(4R)-2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
    ("NC1=[NH+][C@@H](C(=O)[O-])CS1", "(4S)-2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate"),
])
def test_other_hydro_rings_with_a_ring_carboxylate(smiles, name):
    _named(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # the aromatic ring: the protonated nitrogen is not the indicated hydrogen of the parent. The
    # neutral ring's NH is 1H (b)), the '-ium' is 3 '1H-imidazol-3-ium (PIN)',
    # the Blue Book), and no hyphen was missing after the prefix
    ("Cc1[nH]c(C(=O)[O-])c[nH+]1", "2-methyl-1H-imidazol-3-ium-4-carboxylate"),
    ("Cc1[nH+]c(C(=O)[O-])c[nH]1", "2-methyl-1H-imidazol-3-ium-4-carboxylate"),
    ("Cc1c(C(=O)[O-])[nH]c[nH+]1", "5-methyl-1H-imidazol-3-ium-4-carboxylate"),
    ("c1c(C(=O)[O-])[nH]c[nH+]1", "1H-imidazol-3-ium-4-carboxylate"),
    ("O=C([O-])c1c[nH]n[nH+]1", "1H-1,2,3-triazol-3-ium-4-carboxylate"),
    # the indicated-hydrogen nitrogen carries a substituent (the parent is 1H-imidazole, its H
    # replaced): '1,2-dimethyl-1H-imidazol-3-ium-4-carboxylate', the 'ium' 3 then the carboxylate 4.
    # The other drawing (the charge on the methylated nitrogen) is the same species and the same name
    ("Cn1cc(C(=O)[O-])[nH+]c1C", "1,2-dimethyl-1H-imidazol-3-ium-4-carboxylate"),
    ("Cc1[nH]c(C(=O)[O-])c[n+]1C", "1,2-dimethyl-1H-imidazol-3-ium-4-carboxylate"),
    ("Cn1cc[nH+]c1C(=O)[O-]", "1-methyl-1H-imidazol-3-ium-2-carboxylate"),
])
def test_aromatic_azolium_carboxylate_does_not_read_the_proton_as_indicated_hydrogen(smiles, name):
    _named(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # names that were right do not change: aromatic, saturated, quaternary
    ("O=C([O-])c1ccc[nH+]c1", "pyridin-1-ium-3-carboxylate"),
    ("C[n+]1cccc(C(=O)[O-])c1", "1-methylpyridin-1-ium-3-carboxylate"),
    # the Blue Book's own example of (:42456)
    ("C[n+]1c(C(=O)[O-])cc(-c2ccccc2)cc1-c1ccccc1", "1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate"),
    ("O=C([O-])C1CSCC[NH2+]1", "thiomorpholin-4-ium-3-carboxylate"),
    ("C[N+]1(C)CCCCC1C(=O)[O-]", "1,1-dimethylpiperidin-1-ium-2-carboxylate"),
    ("O=C([O-])C1CSC[NH2+]1", "1,3-thiazolidin-3-ium-4-carboxylate"),
])
def test_names_that_were_right_do_not_change(smiles, name):
    _named(smiles, name)


# ---- the carboxylate locant in the choice among charge forms (no JVM) ----

@pytest.mark.parametrize("lower,higher", [
    # (c) before (f): the carboxylate attachment 4 over 6, the prefix locants being one set
    ("2,6-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate",
     "2,4-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-6-carboxylate"),
    ("(4S)-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate",
     "(6S)-2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-6-carboxylate"),
    ("2-methyl-1H-imidazol-3-ium-4-carboxylate", "2-methyl-1H-imidazol-3-ium-5-carboxylate"),
])
def test_form_choice_ranks_the_carboxylate_locant_as_a_suffix(lower, higher):
    from orthonym.assembly.form_choice import lower_locants
    assert lower_locants(lower, higher) == -1
    assert lower_locants(higher, lower) == 1


def test_the_cationic_suffix_stays_ahead_of_the_anionic_one():
    #: "ionic centers on skeletal atoms of the parent hydride are preferred to the locants for
    # positions of attachment of characteristic groups denoted by ionic suffixes": 'ium' 1 then 'oate'
    from orthonym.assembly.form_choice import lower_locants
    a = "2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-6-carboxylate"
    b = "2-methyl-1,4,5,6-tetrahydropyrimidin-3-ium-4-carboxylate"
    assert lower_locants(a, b) == -1


def test_neutralized_cation_removes_the_hydron_and_keeps_the_atoms():
    from rdkit import Chem

    from orthonym.rules.ring_cation_hydro import neutralized_cation
    mol = Chem.MolFromSmiles("Cc1[nH]c(C(=O)[O-])c[nH+]1")
    idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    neutral = neutralized_cation(mol, idx)
    assert neutral.GetNumAtoms() == mol.GetNumAtoms()
    assert neutral.GetAtomWithIdx(idx).GetFormalCharge() == 0
    assert neutral.GetAtomWithIdx(idx).GetTotalNumHs() == 0
    # a quaternary ring nitrogen has no hydron to remove
    quaternary = Chem.MolFromSmiles("C[N+]1(C)CCCCC1C(=O)[O-]")
    assert neutralized_cation(quaternary, 1) is None


def test_hydron_site_parent_cuts_the_substituent_of_the_quaternary_cation():
    #: the quaternary ring nitrogen is the parent's atom plus a hydron, then substituted
    from rdkit import Chem

    from orthonym.rules.ring_cation_hydro import hydron_site_parent
    mol = Chem.MolFromSmiles("CC1=[N+](C)CCC(C(=O)[O-])N1")
    idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    ring = set(next(r for r in mol.GetRingInfo().AtomRings() if idx in r))
    parent = hydron_site_parent(mol, ring, idx)
    assert parent.GetNumAtoms() == mol.GetNumAtoms()
    atom = parent.GetAtomWithIdx(idx)
    assert atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() == 0 and atom.GetDegree() == 2
    # a cation that holds a hydron is the other helper's, and a second substituent is out of scope
    protonated = Chem.MolFromSmiles("CC1=[NH+]CCC(C(=O)[O-])N1C")
    p_idx = next(a.GetIdx() for a in protonated.GetAtoms() if a.GetFormalCharge() == 1)
    p_ring = set(next(r for r in protonated.GetRingInfo().AtomRings() if p_idx in r))
    assert hydron_site_parent(protonated, p_ring, p_idx) is None
    ammonium = Chem.MolFromSmiles("C[N+]1(C)CCCCC1C(=O)[O-]")
    a_idx = next(a.GetIdx() for a in ammonium.GetAtoms() if a.GetFormalCharge() == 1)
    a_ring = set(next(r for r in ammonium.GetRingInfo().AtomRings() if a_idx in r))
    assert hydron_site_parent(ammonium, a_ring, a_idx) is None
