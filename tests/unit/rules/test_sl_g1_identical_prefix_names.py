"""SL lane G, G1: a substituent cited twice in one scope is one multiplied prefix.

 "MULTIPLICATIVE PREFIXES 'DI', 'TRI', ETC. vs. 'BIS', 'TRIS', ETC." (the Blue Book),
 (:7027): "Multiplicative prefixes... are the principal method for describing a
multiplicity of identical features of a structure in chemical names." The locants of the group
are cited together, the italic N before the numerals: "Lowest set of locants" (:3189),
"Italic capital and lower-case letter locants are lower than Greek letter locants, which, in
turn, are lower than numerals" (:3195). The group takes the place of its members in the order of
 (:3448): "multiplicative prefixes, if necessary, are then inserted and do not alter the
alphabetical order already established". The Blue Book's own names: 'N,4-dimethyl-N-(3-methyl-
phenyl)benzamide (PIN)' (:32879), 'N,N,2-trimethyl-3-{...}propanamide (PIN)' (:21624),
'N,N,N,1-tetramethylquinolin-1-ium-3-aminium (PIN)' (:42213).

Before SL lane G the producers that cite the prefixes of the parent from several sources put the
sources side by side: the chain and the ring of ``polyfunctional`` ('2-(4-methoxyphenyl)-2-methyl-
N-methylpropanamide', '2-hydroxy-3-hydroxy-2-(1-hydroxytridecyl)...cyclopentan-1-one'), the
aniline and the benzene diamine of ``benzene`` ('4-methyl-N-methylaniline'), and the amide of
two fragments in ``decomposition`` ('N-methyl-3,3-dimethyl-2-propanamidobutanamide'). Every name
below reads back to the input's full InChIKey (OPSIN 2.9.0); the old spelling is the mutation
each test rejects (the old spellings are at the bottom: never a pin_verified label).
"""
import pytest

from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers, name_default
from tests.support.rt_assert import assert_full_rt, name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    # the chain parent of polyfunctional (the amide / amine N and the carbons)
    ("CC(C)(C1=CC=C(C=C1)OC)C(=O)NC", "2-(4-methoxyphenyl)-N,2-dimethylpropanamide"),
    ("CC(C)(C)C(C(C1=CC2=CC=CC=C2O1)NC)OC", "1-(1-benzofuran-2-yl)-2-methoxy-N,3,3-trimethylbutan-1-amine"),
    ("CCCCCCCC(C(C)(C)N(C)C)NN", "3-hydrazinyl-N,N,2-trimethyldecan-2-amine"),
    ("CC1=CC=C(C=C1)[C@H]([C@H](C)C(=O)N(C)C)O",
     "(2S,3S)-3-hydroxy-N,N,2-trimethyl-3-(4-methylphenyl)propanamide"),
    ("CC(C)(C)C#C/C=C/CN(C)CC1=CC(=CC=C1)OCC2=CC=CC=C2C(F)(F)F",
     "(2E)-N,6,6-trimethyl-N-[(3-{[2-(trifluoromethyl)phenyl]methoxy}phenyl)methyl]hept-2-en-4-yn-1-amine"),
    # the ring parent of polyfunctional (two sources of 'hydroxy')
    ("CCCCCCCCCCCCC(O)[C@@]1(O)C(=O)CC(OC)[C@H]1O",
     "(2S,3R)-2,3-dihydroxy-2-(1-hydroxytridecyl)-4-methoxycyclopentan-1-one"),
    # the aniline, the anilino prefix and the benzene diamine of benzene
    ("CNc1ccc(C)cc1", "N,4-dimethylaniline"),
    ("CN(C)c1ccc(C)cc1", "N,N,4-trimethylaniline"),
    ("CCN(C)c1ccc(C)cc1", "N-ethyl-N,4-dimethylaniline"),
    ("CCC1=CC(=CC(=C1NCC)CC)C", "N,2,6-triethyl-4-methylaniline"),
    ("CNc1ccc(NC)cc1C", "N1,N4,2-trimethylbenzene-1,4-diamine"),
    ("CNc1cc(C)c(N)cc1", "N1,3-dimethylbenzene-1,4-diamine"),
    # the amide of two fragments (decomposition)
    ("CNC(=O)C(NC(=O)CC)C(C)(C)C", "N,3,3-trimethyl-2-propanamidobutanamide"),
]

# The marimastat row (d1914). G1 spells its N-block correctly, which let the amide split name a molecule
# that the default tier used to decline; but the split chooses the monoamide chain as the parent while the
# molecule holds a butanediamide chain (two amide groups). "Seniority of parent hydrides..." /
# "The senior parent structure has the maximum number of substituents corresponding to the principal
# characteristic group (suffix)... in accord with the seniority of classes and the seniority of
# suffixes " (the Blue Book): the name is the right molecule with the wrong parent, so it must
# never carry pin_verified (the structure-side check ``checks_group_count`` demotes it).
D1914 = "CNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)[C@H](O)C(=O)NO)C(C)(C)C"
D1914_NAME = ("(2S)-2-{(2R)-2-[(1S)-1-hydroxy-2-(hydroxyamino)-2-oxoethyl]-4-methylpentanamido}"
              "-N,3,3-trimethylbutanamide")

# What shipped at pin_verified before the fix: a correct name that is not the PIN.
NOT_PIN_ROWS = [
    ("CC(C)(C1=CC=C(C=C1)OC)C(=O)NC", "2-(4-methoxyphenyl)-2-methyl-N-methylpropanamide"),
    ("CC(C)(C)C(C(C1=CC2=CC=CC=C2O1)NC)OC",
     "1-(1-benzofuran-2-yl)-2-methoxy-3,3-dimethyl-N-methylbutan-1-amine"),
    ("CCCCCCCC(C(C)(C)N(C)C)NN", "3-hydrazinyl-2-methyl-N,N-dimethyldecan-2-amine"),
    ("CCCCCCCCCCCCC(O)[C@@]1(O)C(=O)CC(OC)[C@H]1O",
     "(2S,3R)-2-hydroxy-3-hydroxy-2-(1-hydroxytridecyl)-4-methoxycyclopentan-1-one"),
    ("CNc1ccc(C)cc1", "4-methyl-N-methylaniline"),
    ("CCC1=CC(=CC(=C1NCC)CC)C", "2,6-diethyl-N-ethyl-4-methylaniline"),
    ("N#CCCN(C)c1ccc(C)cc1", "3-(4-methyl-N-methylanilino)propanenitrile"),
    ("CNc1ccc(NC)cc1C", "2-methyl-N1,N4-dimethylbenzene-1,4-diamine"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_the_old_spelling_is_not_pin_labelled(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


def test_the_marimastat_amide_split_is_not_a_pin_label():
    """Right molecule, wrong parent: the default tier declines, the breadth tier names it
    systematic_verified, with the N-block spelled once (G1)."""
    assert name_is_rt_exact(D1914_NAME, D1914)
    assert name_default(D1914).get("tier") != "pin_verified"
    assert_not_pin_labelled(D1914, D1914_NAME)
    res = name_best_effort(D1914)
    assert res.get("name") == D1914_NAME and res.get("tier") != "pin_verified", res


def test_the_ring_parent_names_cited_twice_are_one_group_at_the_breadth_tier():
    """The phosphonooxy sugar ketone: the default tier declines it (another reason), the breadth
    tier names it with one 'tetrahydroxy' group; the old name cited 'hydroxy' in two groups."""
    smiles = "O=C1C[C@@](O)(COP(=O)(O)O)[C@@H](O)[C@H](O)[C@H]1O"
    name = assert_full_rt(name_best_effort(smiles).get("name"), smiles)
    assert name == "(2R,3S,4S,5R)-2,3,4,5-tetrahydroxy-5-[(phosphonooxy)methyl]cyclohexan-1-one", name


@pytest.mark.parametrize("smiles,pin", [
    ("N#CCCN(C)c1ccc(C)cc1", "3-(N,4-dimethylanilino)propanenitrile"),
    ("OC(=O)CCN(C)c1ccc(C)cc1", "3-(N,4-dimethylanilino)propanoic acid"),
])
def test_the_anilino_prefix_cites_the_shared_substituent_once(smiles, pin):
    """The anilino prefix is the aniline joiner's name with the head morpheme changed
    (``anilino_prefix_from_aniline_name``). The breadth tier names these rows with the general
    '[methyl(4-methylphenyl)amino]' spelling on its own (the row's label there is
    systematic_verified, before and after), so the PIN tier is asserted here."""
    assert name_is_rt_exact(pin, smiles)
    res = name_default(smiles)
    assert (res.get("name"), res.get("tier"), res.get("is_pin")) == (pin, "pin_verified", True), res
