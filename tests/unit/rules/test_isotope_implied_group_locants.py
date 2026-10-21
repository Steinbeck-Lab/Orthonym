"""A count-only hydrogen descriptor in front of a retained name whose characteristic group is implied.

The rule
--------
 Omission of locants (the Blue Book). (:44180): "In preferred IUPAC names,
locants are omitted if no locants are necessary in unmodified names. However, if isotopic
modification requires a locant to specify its position, then all locants must be specified and none
are omitted." (:44196): "Locants are omitted in compounds or substituent groups in which
all positions are completely isotopically substituted or modified in the same way." ('(2H3)acetonitrile
(PIN) (as in trichloroacetonitrile)',:44182.) (:44202): "Locants are not omitted when there
is a possibility of isomers".

 (heading NAMES,:43716; the rule:43800): the descriptor of a name of two words stands
"before the appropriate word or part of the word that includes the nuclide(s)"; printed
'(2-2H1)acetic acid (PIN)' (:43804) next to '(O-2H)acetic acid (PIN)' (:43806): the O-H of acetic acid
is a position of the word 'acetic acid', so a descriptor before the word that cites only a count has an
isomer. (:43824): "In a name consisting of one word, the isotopic descriptor is placed before
the name, with an appropriate locant. This method is preferred to that of placing the descriptor before
the implied name of the characteristic group." ('(N-2H1)acetamide (PIN)',:43828.)

So when the characteristic group of a retained name is IMPLIED in the name ('acetic acid', 'acetamide',
'acetamido': no suffix of its own to carry a slot, unlike 'ethan(2H)ol',:44184), its O-H or N-H is in the
scope of the front descriptor. '(2H3)acetic acid' is CD3-COOH and also CD2H-COOD, a possibility of
isomers: the locants are cited, '(2,2,2-2H3)acetic acid', and with them, by, the locants of
the unmodified name ('2-hydroxy(2,2-2H2)acetic acid').

The locants of the parent: (heading Letter and/or numeral locants,:44208) "When locants are
needed for defining the structure of the parent structure or of a unit of structure as defined by its
appropriate enclosing marks, then all locants must be cited for the parent structure or that structural
unit". The retained 'acetic acid', 'acetonitrile' and 'acetaldehyde' omit the locants of their substituents
,:3031, 'difluoroacetic acid (PIN) (not 2,2-difluoroacetic acid)',:3037); a descriptor that
needs a locant withdraws that licence for the parent that holds the label, '2,2-dichloro(2-2H)acetic acid',
and not for a parent whose label sits in a substituent unit of its own,
'[3,4-dihydro(7-2H)-2H-1-benzopyran-2-yl]acetic acid'. A stereogenic C-2 keeps its bare descriptor on the
omitted form ('(R)-chloro(hydroxy)(2-2H)acetic acid'); not read here.

Controls: a unit with no heteroatom hydrogen ('(2H3)acetonitrile', 'methyl (2H3)acetate'), a descriptor
that stands before a prefix and covers that prefix only ('(2H3)methoxybenzene',:43730,
'4-[(2H3)methylamino]benzoic acid'), a group that a written-out suffix names ('(2H3)methanol',
'phenyl(2H2)methanol', '(2H3)methanethiol') and a unit completely labelled keep
their names. Not in this topic: a count-only descriptor on a unit whose unmodified name cites locants
('(2H2)ethan-1-ol'-like names, "as in"), an amine suffix ('(2H3)methanamine').
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

# (smiles, name): each must be a pin_verified name at the default tier and at the best-effort tier,
# read back to the input's full InChIKey by a fresh OPSIN call (assert_pin_at_both_tiers)
RED_ROWS = [
    ("[2H]C([2H])([2H])C(=O)O", "(2,2,2-2H3)acetic acid"),
    ("[2H]C([2H])([2H])C(=O)N", "(2,2,2-2H3)acetamide"),
    ("[3H]C([3H])([3H])C(=O)N", "(2,2,2-3H3)acetamide"),
    ("[2H]C([2H])([2H])C(=O)NCCC1=CNC2=C1C=C(C=C2)O",
     "N-[2-(5-hydroxy-1H-indol-3-yl)ethyl](2,2,2-2H3)acetamide"),
    ("[2H]C([2H])([2H])C(=O)NC(CSCCCO)C(=O)O",
     "2-(2,2,2-2H3)acetamido-3-[(3-hydroxypropyl)sulfanyl]propanoic acid"),
    ("[2H]C([2H])(O)C(=O)O", "2-hydroxy(2,2-2H2)acetic acid"),
    ("[2H]C([2H])(C(=O)O)NC(=O)OCC1C2=CC=CC=C2C3=CC=CC=C13",
     "2-({[(9H-fluoren-9-yl)methoxy]carbonyl}amino)(2,2-2H2)acetic acid"),
]

# the same rule on other implied groups and on the locants of the parent (both tiers, as above)
RELATED_ROWS = [
    ("[2H]C([2H])(Cl)C(=O)O", "2-chloro(2,2-2H2)acetic acid"),
    ("[2H]C([2H])(F)C(=O)O", "2-fluoro(2,2-2H2)acetic acid"),
    ("[2H]C([2H])(c1ccccc1)C(=O)O", "2-phenyl(2,2-2H2)acetic acid"),
    ("[2H]C([2H])(C#N)C(=O)O", "2-cyano(2,2-2H2)acetic acid"),
    ("[2H]C([2H])(N(C)C)C(=O)O", "2-(dimethylamino)(2,2-2H2)acetic acid"),
    ("CC(=O)NC([2H])([2H])C(=O)O", "2-acetamido(2,2-2H2)acetic acid"),
    ("[2H]C([2H])(O)C=O", "2-hydroxy(2,2-2H2)acetaldehyde"),
    ("[2H]C([2H])(c1ccccc1)C(=O)N", "2-phenyl(2,2-2H2)acetamide"),
    ("[2H]C([2H])(F)C(=O)NC", "2-fluoro-N-methyl(2,2-2H2)acetamide"),
    ("[2H]C([2H])([2H])C(=O)NN", "(2,2,2-2H3)acetohydrazide"),
    ("[2H]C([2H])([2H])C(=O)Nc1ccccc1", "N-phenyl(2,2,2-2H3)acetamide"),
    ("[2H]C([2H])([2H])C(=O)NC1CCCCC1", "N-cyclohexyl(2,2,2-2H3)acetamide"),
    ("[2H]C([2H])([2H])C(=O)NO", "N-hydroxy(2,2,2-2H3)acetamide"),
    ("[2H]C([2H])([2H])C(=O)NC(C)C(=O)O", "2-(2,2,2-2H3)acetamidopropanoic acid"),
    # the descriptor already cited its locant (the isomer on the OH); the locants of the parent follow
    ("OC(=O)C([2H])(Cl)Cl", "2,2-dichloro(2-2H)acetic acid"),
    ("Cl[13CH2]C(=O)O", "2-chloro(2-13C)acetic acid"),
]

# descriptors of several nuclide groups, each with its locants ('(1-13C,2,2-2H2)', '(2-13C,O-18O)'): the spelling
# checks read them as isotope descriptors (``validation.spelling.lexer._ISOTOPE``), so the locants cited on the
# parent are not read as a error; (:43834), '(2,4-2H2,15N)pyridine (PIN)' (:44194),
# '(O-2H,18O)acetic acid (PIN)' (:43808); the locants of the parent follow (:44180)
MULTI_GROUP_ROWS = [
    ("[2H]C([2H])(Cl)[13C](=O)O", "2-chloro(1-13C,2,2-2H2)acetic acid"),
    ("[2H]C([2H])(Br)[13C](=O)O", "2-bromo(1-13C,2,2-2H2)acetic acid"),
    ("N#[13C]C([2H])([2H])Cl", "2-chloro(1-13C,2,2-2H2)acetonitrile"),
    ("Cl[13CH2]C(=O)[18OH]", "2-chloro(2-13C,O-18O)acetic acid"),
    ("[2H]C([2H])(Cl)C(=O)[18OH]", "2-chloro(2,2-2H2,O-18O)acetic acid"),
]

# names that stay as they are, both tiers (the names of the tree before this topic)
CONTROLS = [
    ("[2H]C([2H])([2H])Oc1ccccc1", "(2H3)methoxybenzene"),
    ("[2H]C([2H])([2H])C#N", "(2H3)acetonitrile"),
    ("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2H6)benzene"),
    ("ClC(Cl)([2H])[2H]", "dichloro(2H2)methane"),
    ("[2H]C", "(2H1)methane"),
    ("[2H]C([2H])([2H])O", "(2H3)methanol"),
    ("OC([2H])([2H])c1ccccc1", "phenyl(2H2)methanol"),
    ("[2H]C([2H])([2H])C(=O)OC", "methyl (2H3)acetate"),
    ("Nc1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2,3,4,5,6-2H5)aniline"),
    ("Oc1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2,3,4,5,6-2H5)phenol"),
    ("OC(=O)c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2,3,4,5,6-2H5)benzoic acid"),
    ("CC(=O)O[2H]", "(O-2H)acetic acid"),
    ("CC(=O)N[2H]", "(N-2H1)acetamide"),
    ("Oc1ccc(C([2H])([2H])[2H])cc1", "4-(2H3)methylphenol"),
    ("Nc1ccc(C(=O)NC([2H])([2H])[2H])cc1", "4-amino-N-(2H3)methylbenzamide"),
    ("OC(=O)c1ccc(NC([2H])([2H])[2H])cc1", "4-[(2H3)methylamino]benzoic acid"),
    ("[2H]C([2H])([2H])N", "(2H3)methanamine"),
    ("[2H]C([2H])([2H])S", "(2H3)methanethiol"),
    # the hydroxy of the labelled carbon is a prefix before the descriptor, outside its scope, and the
    # nitrile has no hydrogen: nothing the count-only descriptor leaves open
    ("[2H]C([2H])(O)C#N", "hydroxy(2H2)acetonitrile"),
    # no hydrogen on the amide nitrogen, an ester, an acid chloride, a labelled acetyl prefix: no heteroatom
    # hydrogen in the unit the descriptor stands before
    ("[2H]C([2H])([2H])C(=O)N(C)C", "N,N-dimethyl(2H3)acetamide"),
    ("[2H]C([2H])(Cl)C(=O)OC", "methyl chloro(2H2)acetate"),
    ("[2H]C([2H])(Cl)C(=O)Cl", "2-chloro(2H2)acetyl chloride"),
    ("[2H]C([2H])([2H])C(=O)NC(=O)C", "N-(2H3)acetylacetamide"),
    # the label sits in a substituent unit with marks of its own: cites the locants of that unit,
    # the parent keeps its omission
    ("OC(=O)CC1CCc2ccc([2H])cc2O1", "[3,4-dihydro(7-2H)-2H-1-benzopyran-2-yl]acetic acid"),
]


@pytest.mark.parametrize("smiles,name", RED_ROWS)
def test_implied_group_hydrogen_makes_the_descriptor_cite_its_locants(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


@pytest.mark.parametrize("smiles,name", RELATED_ROWS)
def test_related_implied_groups_and_the_locants_of_the_parent(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


@pytest.mark.parametrize("smiles,name", MULTI_GROUP_ROWS)
def test_a_descriptor_of_several_nuclide_groups_is_read_as_a_descriptor(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


@pytest.mark.parametrize("content,expected", [
    ("2,2-2H2", True), ("N-2H1", True), ("O-18O", True), ("13C", True),
    ("1-13C,2,2-2H2", True), ("2,4-2H2,15N", True), ("O-2H,18O", True), ("18O-2H,18O", True),
    ("2-13C,O-18O", True), ("2,2-2H2,O-18O", True), ("N,2,2-2H3", True), ("2-2H1,O-2H", True),
    ("2-13C,O1-18O", True), ("O1,6-2H2", True),
    ("4-chlorophenyl", False), ("2,2-dimethylpropyl", False), ("1-13C,chloro", False), ("2,3", False),
])
def test_the_lexer_reads_repeated_nuclide_groups_only_as_an_isotope_descriptor(content, expected):
    from orthonym.validation.spelling.lexer import _ISOTOPE
    assert bool(_ISOTOPE.match(content)) is expected


@pytest.mark.parametrize("smiles,name", CONTROLS)
def test_controls_keep_their_names(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


# PS lane 1 a performance pass change-asserted-value (user decision 2026-10-20): the two completely labelled glycine
# names left CONTROLS. They keep their NAMES (the strict path builds them, best-effort ships them, OPSIN
# reads both back), but 'glycine' is a retained name of Table 10.4, the Blue Book
# 'glycine / aminoacetic acid', no (PIN) tag; (:1978):2052 "Preferred IUPAC names (PINs) for the
# natural products in Chapter are not identified."), so a name built on it is no PIN: the default
# tier declines it and best-effort labels it systematic_verified.
@pytest.mark.parametrize("smiles,name", [
    ("[2H]C([2H])(N([2H])[2H])C(=O)O[2H]", "(2H5)glycine"),
    ("[2H]C([2H])(N)C(=O)O", "(2,2-2H2)glycine"),
])
def test_labelled_glycine_keeps_its_name_below_pin(smiles, name):
    from tests.support.default_tier import declined_at_default
    declined_at_default(smiles, name)


def test_the_ester_of_a_labelled_acid_keeps_its_omission_where_no_locant_is_needed():
    # no heteroatom hydrogen and a count that fills C-2: (:44196) omits the locants of the
    # descriptor and, by, those of the unmodified name
    assert_pin_at_both_tiers("[2H]C([2H])(Cl)C(=O)OC", "methyl chloro(2H2)acetate")


def test_a_stereogenic_alpha_carbon_keeps_its_omitted_form():
    # the producers cite its descriptor bare on the locant-free spelling,:44643); not changed here
    from tests.support.pin_tiers import name_default
    assert name_default("[2H][C@](Cl)(O)C(=O)O")["name"] == "(R)-chloro(hydroxy)(2-2H)acetic acid"


def _heavy_with_hydrogens(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.AddHs(mol)


@pytest.mark.parametrize("smiles,skel,off,count", [
    ("CC(=O)O", "acetic acid", 0, 1),                    # the O-H of the retained name (:43806)
    ("CC(=O)N", "acetamide", 0, 1),                      # the N-H (:43828)
    ("CC(=O)NCC(=O)O", "acetamidoacetic acid", 0, 1),    # the N-H of the acetamido prefix
    ("CC(=O)NC(=O)C", "N-acetylacetamide", 2, 0),        # a descriptor before 'acetyl' labels the prefix
    ("CC(=O)OC", "methyl acetate", 7, 0),                # no hydrogen on the ester oxygen
    ("CC(=O)N(C)C", "N,N-dimethylacetamide", 10, 0),     # no hydrogen on the nitrogen
    ("CC(=O)O", "ethanoic acid", 0, 0),                  # 'oic acid' is a suffix with its own slot (:43812)
    ("CC(=O)S", "ethanethioic S-acid", 0, 0),            # the slot of the systematic suffix
    ("CO", "methanol", 0, 0),                            # an '-ol' on the labelled carbon itself: its own slot
])
def test_the_hydrogens_a_unit_holds_without_a_slot(smiles, skel, off, count):
    from orthonym.rules.isotope_licence import implied_group_hydrogens
    from orthonym.rules import isotopes as iso
    mol = _heavy_with_hydrogens(smiles)
    carrier = next(a.GetIdx() for a in mol.GetAtoms()
                   if a.GetSymbol() == "C" and any(n.GetAtomicNum() == 1 for n in a.GetNeighbors()))
    scope = iso._element_scope(mol, carrier, "C")
    assert len(implied_group_hydrogens(mol, scope, {carrier}, skel, off)) == count


@pytest.mark.parametrize("smiles,expected", [
    ("[2H]C([2H])(O)C(=O)O", True),              # C-2 of acetic acid
    ("[2H]C([2H])(O)C#N", True),                 # C-2 of acetonitrile
    ("[2H]C([2H])(O)C=O", True),                 # C-2 of acetaldehyde
    ("Cl[13CH2]C(=O)O", True),                   # a heavy nuclide on C-2
    ("ClCC(=O)[18OH]", True),                    # a nuclide on the group of C-1
    ("OC(=O)CC1CCc2ccc([2H])cc2O1", False),      # the label sits in a ring substituent
    ("[2H]C([2H])([2H])C(=O)NCC(=O)O", False),   # the label sits on an amide unit, not on the acid unit
    ("[2H][C@](Cl)(O)C(=O)O", False),            # a stereogenic C-2 keeps its bare descriptor
    ("[2H]C([2H])([2H])C", False),               # no acyl unit
])
def test_a_label_on_a_retained_two_carbon_unit(smiles, expected):
    from rdkit import Chem
    from orthonym.rules.isotope_licence import retained_c2_unit_carries_isotope
    assert retained_c2_unit_carries_isotope(Chem.MolFromSmiles(smiles)) is expected
