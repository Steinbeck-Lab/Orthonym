""": a nuclide on a position a retained name does not number takes the systematic name.

 'Location of nuclides on positions not normally denoted by locants' (the Blue Book),
: "When the nuclide is located at a position in a retained name that is not numbered a systematic
name that identifies separately the relevant atom is used for the IUPAC preferred name." Its examples
(:44253-:44259): 'benzene(13C)carbonitrile (PIN)', 'benzene(13C)carboxylic acid (PIN)'.

The decorator of ``rules.isotopes`` names the isotope-stripped molecule with the retained names first. When
no descriptor placement reads back on that skeleton it names the molecule again inside
``assembly.locant_omission.systematic_names_for_unnumbered_labels`` (``_decorate_with_systematic_unnumbered
_parents``), and the producers of retained names with an unnumbered position spell the systematic name there:
the retained 'benzaldehyde', 'benzoic acid', 'benzonitrile', 'benzamide' and 'benzohydrazide' number no
position of the carbon of their group (``rules.benzene._label_on_unnumbered_group``, ``rules.esters.
get_ring_acid_name``), and 'valinate', 'leucinate' and 'isoleucinate' number one chain of the side chain
(``rules.esters._unnumbered_side_chain_atoms``). Only after every strategy on the retained spelling has
declined, so no name that ships without the scope changes.

The retained prefix 'benzyl' numbers no position of its methylene either ('[phenyl(13C)methyl]hydrazine (PIN)',
:44259): the producers of the prefix (``assembly.book_prefixes.retained_group_prefix``, ``assembly.
substituent_naming._check_retained_substituent`` and ``_name_aryl_methyl_ether``, ``rules.ring_substituents``,
``rules.esters``) spell 'phenylmethyl' ('phenylmethoxy') inside the scope where a label sits on the methylene, so
the assembler sorts, multiplies and encloses it like any other prefix; a benzyl group none of whose methylene
is labelled keeps 'benzyl'.

The other cases here are the small gaps found on the way: the letter locant ``O`` of a hydroxyl on boron or
phosphorus,:44282), the nitrogens of a geminal diamine cited ``N`` and ``N'``,
:7739), the slot before the heteroatom locants of a Hantzsch-Widman substituent ring,:43718: "when
the name, or a part of a name, includes a preceding locant, a hyphen is inserted"), and one nuclide cited
without locants beside another cited with them in one descriptor ('(2-13C,15N)acetamido',,:44190).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules import esters
from orthonym.rules import isotope_scopes as S
from orthonym.rules import isotopes as I
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


_PIN_NAMER = Orthonym(style="pin")


def _assert_row(smiles, expected, tier="systematic_verified"):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == tier, row
    assert name_is_rt_exact(row["name"], smiles), row


def _assert_pin_row(smiles, expected):
    """The default tier emits the name when it is the PIN, spelt as the Blue Book spells a count of one
    ('(2H)', not the '(2H1)' of the wider tiers)."""
    row = _PIN_NAMER.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"], row


# --- the carbon of a retained benzo group has no number (rules/benzene.py) --------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("[2H]C(=O)C1=CC=C(C=C1)SC", "4-(methylsulfanyl)benzene-1-(2H1)carbaldehyde"),
    ("[2H]C(=O)C1=CC(=C(C=C1)C)C", "3,4-dimethylbenzene-1-(2H1)carbaldehyde"),
    ("[2H]C(=O)C1=CC=CC=C1C(=O)C", "2-acetylbenzene-1-(2H1)carbaldehyde"),
    ("[3H]C(=O)C1=CC=CC(=C1C)C", "2,3-dimethylbenzene-1-(3H1)carbaldehyde"),
    ("[2H]C(=O)C1=C(C=C(C=C1F)Br)F", "4-bromo-2,6-difluorobenzene-1-(2H1)carbaldehyde"),
])
def test_a_label_on_the_formyl_of_a_substituted_benzaldehyde(smiles, expected):
    """'benzaldehyde' numbers neither the carbon, the oxygen nor the hydrogen of its formyl: the systematic
    'benzene-1-carbaldehyde' does, and the locant 1 is cited beside the other substituents,
    the Blue Book: deny by default)."""
    _assert_row(smiles, expected)


def test_the_default_tier_emits_the_systematic_name_of_a_labelled_formyl_as_the_pin():
    """ makes the systematic name the PIN ('benzene(13C)carbonitrile (PIN)'); the default tier spells
    the count of one as the book does."""
    _assert_pin_row("[2H]C(=O)C1=CC=C(C=C1)SC", "4-(methylsulfanyl)benzene-1-(2H)carbaldehyde")
    _assert_pin_row("[2H]C(=O)C1=CC=CC=C1", "benzene(2H)carbaldehyde")


def test_an_unsubstituted_benzaldehyde_keeps_the_name_it_had():
    _assert_row("[2H]C(=O)C1=CC=CC=C1", "benzene(2H1)carbaldehyde")


@pytest.mark.parametrize("smiles,expected", [
    # read back through 'benzaldehyde' by OPSIN, as before the scope existed
    ("[13CH](=O)c1ccc(C)cc1", "4-methyl(13C)benzaldehyde"),
    ("CO[13C](=O)c1ccc(Cl)cc1", "methyl 4-chloro(13C)benzoate"),
    ("N#[13C]c1ccc(C)cc1", "4-methyl(13C)benzonitrile"),
])
def test_a_name_that_shipped_with_the_retained_parent_is_not_replaced(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert name_is_rt_exact(row["name"], smiles), row


def test_the_ring_and_the_carboxyl_of_a_benzoic_acid_labelled_together():
    """The ring carbons and the carboxyl carbon are two parts of the name ('benzoic acid' has no place for the
    second); the systematic 'benzene-1-carboxylic acid' has."""
    smiles = "C1=[13C]([13CH]=[13CH][13C](=[13CH]1)Br)[13C](=O)O"
    row = name_best_effort(smiles)
    assert row["name"] and "benzene-1-(13C)carboxylic acid" in row["name"], row
    assert row["tier"] == "systematic_verified", row
    assert name_is_rt_exact(row["name"], smiles), row


def test_the_carboxyl_of_a_benzoate_ester_labelled_with_its_ring():
    """The ester names its acid by the retained stem 'benzoic'; ``get_ring_acid_name`` gives the systematic
    'benzenecarboxylic' where the carboxyl carbon is labelled."""
    smiles = "CCCO[13C](=O)[13C]1=C[13CH]=[13C]([13CH]=[13CH]1)O"
    row = name_best_effort(smiles)
    assert row["name"] and row["name"].startswith("propyl 4-hydroxy(") and "benzene-1-(13C)carboxylate" in row["name"], row
    assert name_is_rt_exact(row["name"], smiles), row


# --- the side chain of a retained amino acid is numbered along one chain (rules/esters.py) ----------------

@pytest.mark.parametrize("smiles,expected", [
    ("[2H]C([2H])([2H])C(C([2H])([2H])[2H])[C@]([2H])(N)C(=O)OC1CCCCC1",
     "cyclohexyl (2S)-2-amino-3-(2H3)methyl(2,4,4,4-2H4)butanoate"),
])
def test_both_methyls_of_a_valinate_labelled(smiles, expected):
    """The retained 'valinate' numbers one methyl of the isopropyl (C-4); the other is a substituent of the
    chain, and a label on both has no place in 'cyclohexyl (2-2H)L-valinate'. The stereodescriptor stands at
    the front of the acid word,:44643)."""
    _assert_row(smiles, expected)


def test_a_valinate_labelled_on_a_numbered_position_keeps_its_retained_name():
    _assert_row("CC(C)[C@]([2H])(N)C(=O)OC1CCCCC1", "cyclohexyl (2-2H)L-valinate")


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)[C@H](N)C(=O)OC", [0, 2]),             # valine: the two methyls of C-3 tie for the chain
    ("CC(C)C[C@H](N)C(=O)OC", [0, 2]),            # leucine: the two methyls of C-4 tie
    ("CC[C@H](C)[C@H](N)C(=O)OC", [3]),           # isoleucine: the methyl of C-3, the ethyl is the chain
    ("C[C@H](N)C(=O)OC", []),                     # alanine: a chain of one
    ("N[C@@H](Cc1ccccc1)C(=O)OC", []),            # phenylalanine: a ring, numbered as the ring
])
def test_the_atoms_a_retained_amino_acid_name_does_not_number(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    match = esters.find_ester_match(mol)
    acid_atoms, _alkyl = esters.parse_ester_fragments(mol, match)
    assert esters._unnumbered_side_chain_atoms(mol, acid_atoms, match[0]) == expected


# --- the hydroxyl of boron and phosphorus takes the letter O (rules/isotopes.py) ---------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("B(O)(O[2H])c1ccccc1", "phenyl(O-2H)boronic acid"),
    ("CP(=O)(O)O[2H]", "methyl(O-2H)phosphonic acid"),
])
def test_the_hydroxyl_hydrogen_of_a_boronic_or_phosphonic_acid(smiles, expected):
    """ (the Blue Book): an italic capital letter is the locant of an atom the numbers do not
    reach, '(O-2H, 18O)acetic acid'; OPSIN reads it on the hydroxyl of boron and of phosphorus."""
    _assert_row(smiles, expected, tier="pin_verified")


# --- the nitrogens of a geminal diamine (rules/isotopes.py) -------------------------------------------------

def test_the_nitrogens_of_methanediamine_are_cited_n_and_n_prime():
    """ under ' PRIMES' (the Blue Book, sentence:7739): the numbers of the parent differentiate
    the nitrogen atoms of di- and polyamines, "except for geminal amines, imines, and amides", whose nitrogens
    are told apart by primes, N and N'."""
    _assert_row("[2H]NCNC", "N-methyl(N'-2H1)methanediamine", tier="pin_verified")


# --- a Hantzsch-Widman substituent ring (rules/isotopes.py) -------------------------------------------------

def test_a_label_on_a_dioxolanyl_prefix_takes_the_slot_before_its_locants():
    _assert_row("COC(=O)C1=CC=C(C=C1)C=C[13CH]2OCCO2",
                "methyl 4-{2-[(2-13C)-1,3-dioxolan-2-yl]ethenyl}benzoate", tier="pin_verified")


# --- one nuclide cited with locants beside another cited without (rules/isotope_scopes.py) ---------------

def test_two_nuclides_of_one_slot_one_with_locants_and_one_without():
    """'(2-13C,15N)acetamido': the one nitrogen of the unit needs no locant, the Blue Book:
    "Locants are omitted when there is only one atom of a given element"), the carbon does."""
    smiles = "CN(C)CC1CCCN1[13CH2]C(=O)[15NH][13CH2][13CH2][13C](=O)[15NH][13CH2][13CH2][13C](=O)O"
    original = Chem.MolFromSmiles(smiles)
    stripped, label_map = I.strip_isotopes(original)
    name = S.compose_scopes(
        "3-[3-(2-{2-[(dimethylamino)methyl]pyrrolidin-1-yl}acetamido)propanamido]propanoic acid",
        original, label_map, stripped.GetNumHeavyAtoms())
    assert name == ("3-[3-(2-{2-[(dimethylamino)methyl]pyrrolidin-1-yl}(2-13C,15N)acetamido)"
                    "(13C3,15N)propanamido](13C3)propanoic acid")
    assert name_is_rt_exact(name, smiles)


# --- the retained prefix 'benzyl' (its producers, inside the scope) -------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # the methylene of the alcohol part of an ester; the ester word is not enclosed, as 'cyclohexyl(2H2)methyl acetate'
    ("[2H]C([2H])(OC(=O)C)c1ccccc1", "phenyl(2H2)methyl acetate"),
    # the ring and the methylene of one benzyl group, labelled apart
    ("[2H]C1=C(C(=C(C=C1)C([2H])([2H])OC(=O)C2=CC=CC=C2)[2H])[2H]", "(2,3,4-2H3)phenyl(2H2)methyl benzoate"),
    # the Blue Book's own example (:44259), and its benzyloxy counterpart
    ("[13CH2](c1ccccc1)NN", "[phenyl(13C)methyl]hydrazine"),
    ("[2H]C([2H])(c1ccccc1)OCC(=O)O", "[phenyl(2H2)methoxy]acetic acid"),
    # the substituent of a nitrogen, with and without the stereocentre the label induces; the prefixes are cited in
    # alphanumerical order, 'phenyl' before 'phenylmethyl'
    ("[2H]C([2H])(NC(C)(C)C)c1ccccc1", "2-methyl-N-[phenyl(2H2)methyl]propan-2-amine"),
    ("[H][C@@]([2H])(C1=CC=CC=C1)N[C@@](C2=CC=CC=C2)(C(F)(F)F)OC",
     "(1R)-2,2,2-trifluoro-1-methoxy-1-phenyl-N-[(1R)-phenyl(2H1)methyl]ethan-1-amine"),
])
def test_the_methylene_of_a_benzyl_group_takes_phenylmethyl_where_it_is_labelled(smiles, expected):
    """ (the Blue Book): the retained 'benzyl' numbers no position of its methylene, so a label there
    takes the systematic prefix ('[phenyl(13C)methyl]hydrazine (PIN)',:44259). Both tiers name it as the PIN."""
    _assert_pin_row(smiles, expected)
    _assert_row(smiles, expected, tier="pin_verified")


@pytest.mark.parametrize("smiles", [
    "O=C(OCc1ccccc1)c1ccccc1",
    "[2H]c1ccccc1C(=O)OCc1ccccc1",
])
def test_a_benzyl_group_without_a_label_on_its_methylene_keeps_its_name(smiles):
    assert "benzyl benzoate" in _PIN_NAMER.name(smiles) or "benzyl (2-2H)benzoate" in _PIN_NAMER.name(smiles)


def test_a_producer_that_hands_on_a_kekule_form_is_still_mapped_onto_the_labelled_molecule():
    """The skeleton a producer names can be drawn with the Kekule alternation of its ring (the fragments of a large
    glycoside are): the label is read on it all the same, or a benzyl group of a 141-atom tetrasaccharide stays
    'benzyl' where its methylene is labelled."""
    from orthonym.assembly import locant_omission as LO
    original = Chem.MolFromSmiles("[2H]C([2H])(OC)c1ccccc1")
    kekule = Chem.MolFromSmiles("COCc1ccccc1")
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    with LO.isotopic_skeleton_scope(original), LO.systematic_names_for_unnumbered_labels():
        assert LO.label_on_unnumbered_position(kekule, [2])
        assert not LO.label_on_unnumbered_position(kekule, [0])


def test_the_benzyloxy_groups_a_label_does_not_reach_keep_their_name():
    """One benzyloxy group of a glycoside carries the label, the other two do not: 'phenylmethoxy' is the prefix of the
    group with a label on its methylene only, the others stay 'benzyloxy'."""
    smiles = ("[H][C@@]([2H])(C1=CC=CC=C1)O[C@@H]2[C@H](O[C@@H]([C@@H]([C@H]2OCC3=CC=CC=C3)OCC4=CC=CC=C4)OC)"
              "CO[Si](C5=CC=CC=C5)(C6=CC=CC=C6)C(C)(C)C")
    row = name_best_effort(smiles)
    assert "bis(benzyloxy)" in row["name"] and "[(1R)-phenyl(2H1)methoxy]" in row["name"], row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles,must_hold", [
    # four benzyloxy groups of one sugar, each with one deuterium on its methylene: multiplied by the assembler
    ("[2H]C(C1=CC=CC=C1)OC[C@@H]2[C@H]([C@@H]([C@H](C(O2)O)OC([2H])C3=CC=CC=C3)OC([2H])C4=CC=CC=C4)"
     "OC([2H])C5=CC=CC=C5",
     ["tris[phenyl(1-2H1)methoxy]", "{[phenyl(1-2H1)methoxy]methyl}"]),
    # the substituents of a heptanone, in the order of the Blue Book: hydroxy before phenyl
    ("[2H]C(C1=CC=CC=C1)C(=O)[C@@](C([2H])C2=CC=CC=C2)([C@](C([2H])C3=CC=CC=C3)([C@@]([C@@H](CO)O)"
     "(C([2H])C4=CC=CC=C4)O)O)O",
     ["3,4,5,6,7-pentahydroxy-1-phenyl-3,4,5-tris[phenyl(1-2H1)methyl]"]),
    # the substituents of a chain are cited in alphanumerical order and not in the order the methylene was met
    ("[2H]C1(C(N(C(C(C1([2H])CC2CC3=CC(=C(C=C3C2=O)OC)OC)([2H])[2H])([2H])[2H])C([2H])([2H])C4=CC=CC=C4)"
     "([2H])[2H])[2H]",
     ["5,6-dimethoxy-2-({1-[phenyl(2H2)methyl](2H9)piperidin-4-yl}methyl)"]),
])
def test_the_methylene_of_a_benzyl_group_labelled_beside_other_labels(smiles, must_hold):
    row = name_best_effort(smiles)
    assert row["name"], row
    assert all(part in row["name"] for part in must_hold), row
    assert row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles,expected", [
    ("[2H]/C(=C(/CC)\\C(=O)O)/C1=CC=CC=C1", "(2E)-2-[phenyl(2H)methylidene]butanoic acid"),
    ("[13CH](=C(/CC)C(=O)O)c1ccccc1", "2-[phenyl(13C)methylidene]butanoic acid"),
    # the parent needs the locant 4, the unit of its own marks does not
    ("CCOC(=O)/C(=[13CH]/C1=CC=CC=C1)/C=[13CH]\\C2=CC=CC=C2",
     "ethyl (2E,3Z)-4-phenyl-2-[phenyl(13C)methylidene](4-13C)but-3-enoate"),
])
def test_a_one_carbon_methylidene_unit_cites_no_locant(smiles, expected):
    """ (the Blue Book): "locants are omitted if no locants are necessary in unmodified names"; the
    methine of a benzylidene is the whole unit of its marks,:44210: "all locants must be cited for the parent
    structure or that structural unit" that needs them), as in '[phenyl(13C)methyl]hydrazine (PIN)' (:44259) and
    '(1-14C)pentan(3H)oic acid (PIN)' (:43812)."""
    _assert_pin_row(smiles, expected)


# --- the lowest locants of the modified atoms (rules/isotope_scopes.py) -----------------------------------------

def test_the_composer_cites_the_lowest_locants_whatever_the_spelling_of_the_input():
    """ NUMBERING criterion (i) (the Blue Book): "the starting point and the direction of numbering of
    the analogous isotopically substituted compound are chosen so as to give lowest locants to the modified atoms or
    groups considered together as a set in increasing numerical order". A phenyl ring is read onto the input in two
    ways; the name cites the lower set, and the same name whatever order the atoms of the input are written in."""
    smiles = "[2H]C1=C(C(=C(C=C1)C([2H])([2H])OC(=O)C2=CC=CC=C2)[2H])[2H]"
    mol = Chem.MolFromSmiles(smiles)
    names = {name_best_effort(Chem.MolToSmiles(mol, doRandom=True))["name"] for _ in range(4)}
    names.add(name_best_effort(smiles)["name"])
    assert names == {"(2,3,4-2H3)phenyl(2H2)methyl benzoate"}, names


@pytest.mark.parametrize("smiles,expected", [
    # the ring of the labelled phenyl is read onto the input in two directions: 2,3,4,5 and not 3,4,5,6
    ("[2H]C1=CC(=C(C(=C1[2H])[2H])[2H])[Si](C2=C(C(=C(C(=C2[2H])[2H])[2H])[2H])[2H])"
     "C3=C(C(=C(C(=C3[2H])[2H])[2H])[2H])[2H]", "(2,3,4,5-2H4)phenyldi(2H5)phenylsilyl"),
    # the two equivalent ends of a ring: {2,2,6} and not {2,6,6}, and the configuration is cited at the atom that has it
    ("[H][C@@]1(CC(=O)CC(C1C2CCC(CC2)CCC)([2H])[2H])[2H]",
     "1-[(6S)-4-oxo(2,2,6-2H3)cyclohexyl]-4-propylcyclohexane"),
])
def test_the_lowest_set_of_locants_is_cited_for_a_symmetric_skeleton(smiles, expected):
    """ NUMBERING (i) (the Blue Book): "lowest locants to the modified atoms or groups considered
    together as a set in increasing numerical order"."""
    _assert_row(smiles, expected)


def test_a_unit_cited_without_locants_is_not_a_lower_numbering():
    """Three pivalates of one sugar carry one deuterium each on a methyl: the descriptor cites the position, '(3-2H1)'
    , the Blue Book, locants are omitted only "if no locants are necessary in unmodified names", and
    '2,2-dimethylpropanoyl' needs them); a reading of the skeleton whose unit holds every position of its nuclide is
    another spelling, not another numbering, and does not replace it."""
    smiles = ("[2H]CC(C)(C)C(=O)OC[C@@H]1[C@H]([C@@H]([C@H](C(O1)N)OC(=O)C(C)(C)C[2H])OC(=O)C(C)(C)C[2H])"
              "OC(=O)C(C)(C)C[2H]")
    row = name_best_effort(smiles)
    assert "{[2,2-dimethyl-1-oxo(3-2H1)propoxy]methyl}" in row["name"] and "(2H1)propoxy" not in row["name"], row
    assert name_is_rt_exact(row["name"], smiles), row


# --- a borono or a phosphono PREFIX is no parent: the letter stays bare (rules/isotope_letters.py) -----------

@pytest.mark.parametrize("smiles,expected", [
    ("[2H]OC(=O)c1ccc(B(O)O)cc1", "4-borono(O-2H)benzoic acid"),
    ("[2H]OC(=O)c1ccc(P(=O)(O)O)cc1", "4-phosphono(O-2H)benzoic acid"),
])
def test_the_hydroxyls_of_a_prefix_do_not_number_the_letter_locant(smiles, expected):
    """ (the Blue Book) writes '(O-2H, 18O)acetic acid': one letter, not 'O1'. The hydroxyls of the
    'borono' and 'phosphono' of a prefix are not oxygens of the parent, so they do not make the bare letter
    ambiguous: both names are what the engine shipped before the letter was offered on boron and phosphorus."""
    _assert_pin_row(smiles, expected)
    _assert_row(smiles, expected, tier="pin_verified")
