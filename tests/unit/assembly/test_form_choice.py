"""Which name of one species' charge forms stands: ``assembly.form_choice`` (pure strings, no JVM).

Resonance fix RH made the best-effort tier name the other drawings of a delocalised cation, and the choice
among the forms' names took the form that comes first by canonical SMILES wherever the names differ in more
than their locant digits. 12 of the 41 renames were worse names than before (audit
`internal notes`, "Rebased end checks", Step 4). The rules, with the Blue
Book heading and the sentence they apply:

* CHOICE OF A PARENT STRUCTURE (the Blue Book), "A parent cationic structure is chosen by
  applying the following criteria in order until a definitive choice is achieved: (a) parent with the maximum
  number of cationic, including cationic suffixes" (:42334).
* NUMBERING (:3219): (c) "principal characteristic groups and free valences (suffixes)", (e)(i) "low
  locants are given to hydro/dehydro prefixes... and 'ene' and 'yne' endings", (f) "detachable alphabetized
  prefixes, all considered together in a series of increasing numerical order" (:3301), (g) "lowest locants
  for the substituent cited first as a prefix in the name" (:3307).
* Bi- and polycyclic von Baeyer parent hydrides, (:16633) "If there is a choice of names
  and numbering, the following criteria are considered in order until a decision is reached: (1) a minimum
  number of compound locants" (:16635), "(2) when comparing double bond locants that also include compound
  locants, any number in parentheses is ignored" (:16655), "(3) if there is still a choice, low locants are
  selected considering all locants (including those in parentheses) as a set" (:16667).
"""
import pytest

from orthonym.assembly.form_choice import (
    beats,
    cationic_parent_centres,
    locant_roles,
    lower_locants,
    non_pin_parts,
    stereo_decides_below_pin,
)

pytestmark = pytest.mark.unit


# ---- (a): the cationic centres of the parent --------------------------------------------

@pytest.mark.parametrize("name, centres", [
    # (a), the example of:42334: the parent with two cationic suffixes, not the piperidinium
    ("2-(piperidin-1-ium-3-yl)propane-1,2-bis(aminium)", 2),
    ("3-(1,2-diaminiumylpropan-2-yl)piperidin-1-ium", 1),
    # and (1): a ring 'ium', 'aminium', 'diium'
    ("1-methylpyridin-1-ium", 1),
    ("N,N,N-trimethylmethanaminium", 1),
    ("tetramethyldiazene-1,2-diium", 2),
    ("1,4-dioxane-1,4-diium", 2),
    ("2-(3-azaniumylcyclopentyl)propanamidium", 1),
    # a cation in a substituent prefix is not the parent's
    ("4-(1-methylpyridin-1-ium-4-yl)aniline", 0),
    ("1-({[4-(1-methylpyridin-1-ium-4-yl)phenyl]amino}diazenyl)benzene", 0),
    # a von Baeyer parent with compound locants: the marks of '1(30)' do not hide the suffix
    ("24-methyl-2,5-diazaheptacyclo[21.6.1.1^2,6]dotriaconta-1(30),6,8(31),9,11-pentaen-15-ium", 1),
    ("23-oxa-4,6,9,14,34-pentaazadecacyclo[22.10.1.0^1,6]pentatriaconta-2,5,24(35),25-tetraene-6,34-diium", 2),
    # a salt: the cation's parent, the anion is a word of its own
    ("1,3-dimethyl-1H-imidazol-3-ium chloride", 1),
    ("sodium acetate", 0),
])
def test_the_cationic_centres_of_the_parent(name, centres):
    assert sum(cationic_parent_centres(name)) == centres


def test_the_components_of_a_salt_are_counted_each_and_a_multiplied_cation_is_one_component():
    # a cation named once ('bis[...]', one component) and written twice (two components) is not compared
    assert cationic_parent_centres("1,3-dimethyl-1H-imidazol-3-ium chloride") == (1, 0)
    assert cationic_parent_centres("1,6-diazabicyclo[5.4.0]undec-6-en-6-ium 3-methyl-2-oxo-2H-1-benzopyran-4-olate") == (1, 0)
    assert cationic_parent_centres("4-methylpyridin-1-ium\u2014bromide (1/1)") == (1, 0)
    multiplied = "bis[1,5,7-triazabicyclo[4.4.0]dec-1(6)-en-1-ium] pentanedioate pentahydrate"
    twice = ("1,5,7-triazabicyclo[4.4.0]dec-1(6)-en-1-ium 1,5,7-triazabicyclo[4.4.0]dec-5-en-5-ium "
             "pentanedioate pentahydrate")
    assert cationic_parent_centres(multiplied) == (0, 0, 0)
    assert cationic_parent_centres(twice) == (1, 1, 0, 0)
    assert not beats(twice, multiplied) and not beats(multiplied, twice)


def test_the_parent_that_holds_the_cation_beats_the_one_that_moves_it_into_a_prefix():
    # (a); the second name has the lower ene locants, and still loses
    parent = "9-[1,4-di(naphthalen-1-yl)-1,2-dihydro-1,3-diazet-2-ylidene]-9-azahexacyclo[15.8.0.0^2,10]pentacosa-1(25),2(10)-dodecaen-9-ium"
    prefix = "9-[1,4-di(naphthalen-1-yl)-1,3-diazet-1-ium-2-yl]-9-azahexacyclo[15.8.0.0^2,10]pentacosa-1(17),2(10)-dodecaene"
    assert cationic_parent_centres(parent) == (1,) and cationic_parent_centres(prefix) == (0,)
    assert beats(parent, prefix) and not beats(prefix, parent)
    assert lower_locants(parent, prefix) == 0           # not one skeleton: the locants do not compare them
    # a label-only part (the 1,2-dihydro prefix) is the last criterion and does not move the cation back
    soft = {"non_pin_labels": ("1,2-dihydro-1,3-diazet-2-ylidene",)}
    assert beats(parent, prefix, soft, {}) and not beats(prefix, parent, {}, soft)


def test_a_construction_the_blue_book_does_not_allow_ranks_before_the_parent_that_holds_the_cation():
    # the parent that holds the cation, spelled with a von Baeyer ring where a fusion name exists
    #, "When fusion names are not allowed, unsaturated von Baeyer ring system names are preferred
    # IUPAC names":23710), against a neutral parent with the fusion name and the cation in a prefix
    von_baeyer = "7-{[5-amino-1H-pyrazol-4-yl]imino}-8,9-diphenyl-1-azabicyclo[4.3.0]nona-1,3,5,8-tetraen-1-ium"
    fusion = "1-{[5-amino-4H-pyrazol-1-ium-4-ylidene]amino}-2,3-diphenylindolizine"
    assert cationic_parent_centres(von_baeyer) == (1,) and cationic_parent_centres(fusion) == (0,)
    assert beats(von_baeyer, fusion)                    # no record on either side: (a)
    record = {"non_pin_fragments": ("azabicyclo[4.3.0]nona-1,3,5,8-tetraen-1-ium",)}
    assert beats(fusion, von_baeyer, {}, record) and not beats(von_baeyer, fusion, record, {})


# ----: the Blue Book's own examples ---------------------------------------------------

@pytest.mark.parametrize("better, worse, why", [
    # (1) a minimum number of compound locants (:16635), the examples of the criterion
    ("bicyclo[4.2.0]oct-6-ene", "bicyclo[4.2.0]oct-1(8)-ene", "(1)"),
    ("bicyclo[6.5.1]tetradec-8-ene", "bicyclo[6.5.1]tetradec-1(13)-ene", "(1)"),
    ("bicyclo[4.1.0]hepta-1,3,5-triene", "bicyclo[4.1.0]hepta-1(6),2,4-triene", "(1)"),
    # (2) the numbers in parentheses are ignored (:16655): '1,11,13' is lower than '4,5,7' and '4,6,8', and
    # (IV) has two compound locants where (I) has one
    ("tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene", "tricyclo[9.3.1.1^4,8]hexadeca-4(16),5,7-triene", "(2)"),
    ("tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene", "tricyclo[9.3.1.1^4,8]hexadeca-4,6,8(16)-triene", "(2)"),
    ("tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene", "tricyclo[9.3.1.1^4,8]hexadeca-1(14),11(15),12-triene", "(1)"),
    ("tricyclo[9.3.1.1^4,8]hexadeca-1(15),4(16),5,7,11,13-hexaene",
     "tricyclo[9.3.1.1^4,8]hexadeca-1(15),4,6,8(16),11,13-hexaene", "(2)"),
    ("tricyclo[9.3.1.1^4,8]hexadeca-1(15),4(16),5,7,11,13-hexaene",
     "tricyclo[9.3.1.1^4,8]hexadeca-1(14),4,6,8(16),11(15),12-hexaene", "(1)"),
    # (3) all the locants, those in parentheses included, as a set (:16667): '3,11,18' is lower than '3,11,19'
    ("tetracyclo[7.7.1.1^3,7.1^11,15]nonadeca-3,11(18)-diene", "tetracyclo[7.7.1.1^3,7.1^11,15]nonadeca-3(19),11-diene", "(3)"),
])
def test_the_von_baeyer_ene_locants_by_p_31_1_4_2(better, worse, why):
    assert lower_locants(better, worse) == -1 and lower_locants(worse, better) == 1, why
    assert beats(better, worse) and not beats(worse, better)


@pytest.mark.parametrize("better, worse, why", [
    # "Bi- and polycyclic von Baeyer structures with both double and triple bonds" (:16675):
    # "(1) lower locants are assigned to multiple bonds as a set" (:16677), the set '2,11,13,18' is lower than
    # '3,5,14,17'
    ("bicyclo[14.3.1]icosa-11,13,18-trien-2-yne", "bicyclo[14.3.1]icosa-3,5,17-trien-14-yne", "(1)"),
    # "(2) lower locants are assigned to double bonds" (:16683)
    ("bicyclo[11.3.1]heptadec-2-en-11-yne", "bicyclo[11.3.1]heptadec-11-en-2-yne", "(2)"),
])
def test_the_ene_and_yne_locants_by_p_31_1_4_3(better, worse, why):
    assert lower_locants(better, worse) == -1 and lower_locants(worse, better) == 1, why
    assert beats(better, worse) and not beats(worse, better)


def test_with_a_triple_bond_the_compound_locants_come_third_not_first():
    # (3) "compound locants are kept to a minimum" ranks after the sets of the multiple bonds and of
    # the double bonds: the name with the lower set stands although it has the compound locant (1)
    # would put the compound locants first)
    one_compound = "bicyclo[8.3.1]tetradeca-1(13),4,6-trien-8-yne"
    none = "bicyclo[8.3.1]tetradeca-2,4,6-trien-10-yne"
    assert lower_locants(one_compound, none) == -1
    # the same lists without a triple bond: the compound locants first
    assert lower_locants("bicyclo[8.3.1]tetradeca-2,4,6-triene", "bicyclo[8.3.1]tetradeca-1(13),4,6-triene") == -1


def test_the_enclosing_marks_that_follow_from_a_compound_locant_do_not_make_two_names_two_skeletons():
    # dropping '(24)' from the ene list changes the nesting, '[...]' against '(...)': one skeleton
    a = "1-methyl-2-[4-methylpentacyclo[18.4.0.0^2,7]tetracosa-1(24),2,4,6-tetraen-3-yl]-1H-imidazol-3-ium"
    b = "1-methyl-2-(4-methylpentacyclo[18.4.0.0^2,7]tetracosa-1,3,5,7-tetraen-3-yl)-1H-imidazol-3-ium"
    assert locant_roles(a)[0] == locant_roles(b)[0]
    assert lower_locants(b, a) == -1                   # no compound locant: (1)


def test_a_locant_that_opens_an_enclosing_mark_is_a_locant():
    # '[3-(4-...' and '[1-(4-...': the first locant after a '[' was not read, so the two were not one skeleton
    a = "6-[1-(4-cyanophenyl)-1H-imidazol-3-ium-3-yl]-4-oxo-4,5-dihydro-1,3,5-triazine"
    b = "4-[3-(4-cyanophenyl)-1H-imidazol-3-ium-1-yl]-6-oxo-1,6-dihydro-1,3,5-triazine"
    assert locant_roles(a)[0] == locant_roles(b)[0]
    assert lower_locants(b, a) == -1                   # free valence 1 against 3, hydro locants 1,6 against 4,5


# ---- (c): the ionic suffix of the parent, then of a prefix -------------------------------

def test_the_free_valence_of_a_prefix_is_a_suffix_that_ranks_before_the_prefix_locants():
    # (c) "principal characteristic groups and free valences (suffixes)" (:3256) before (f): the
    # free valence at 1 stands over the lower set of prefix locants that the other name has (row 137 of the audit)
    a = "[2-ethyl-4-(2-hydroxybenzamido)-1H-pyrazol-2-ium-1-yl](methyl)amino"
    b = "[1-ethyl-4-(2-hydroxybenzamido)-1H-pyrazol-2-ium-2-yl](methyl)amino"
    assert lower_locants(a, b) == -1 and lower_locants(b, a) == 1


def test_a_multiplied_ionic_suffix_is_the_suffix_of_the_name():
    # '-4,34-diium' against '-6,34-diium': (c) decides before the ene locants (e)
    a = "23-oxa-4,6,9,14,34-pentaazadecacyclo[22.10.1.0^1,6]pentatriaconta-2,4,7,9,24(35)-pentaene-4,34-diium"
    b = "23-oxa-4,6,9,14,34-pentaazadecacyclo[22.10.1.0^1,6]pentatriaconta-2,5,7,9,24(35)-pentaene-6,34-diium"
    assert lower_locants(a, b) == -1
    # the suffix outranks the ene set: a lower ene set does not beat a lower 'ium'
    c = "x-azahexacyclo[21.6.1.1^2,6]dotriaconta-1,3,5-trien-16-ium"
    d = "x-azahexacyclo[21.6.1.1^2,6]dotriaconta-1(30),6,8(31)-trien-15-ium"
    assert lower_locants(d, c) == -1


def test_the_ium_of_the_parent_is_read_before_the_ium_of_a_prefix():
    a = "2-(1-methylpyridin-1-ium-4-yl)-3-azabicyclo[4.3.0]nona-1,3,5-trien-4-ium"
    b = "2-(1-methylpyridin-1-ium-4-yl)-3-azabicyclo[4.3.0]nona-1,3,5-trien-7-ium"
    # the ium of the parent, then the ium and the free valence of the prefix, each with its locants
    assert locant_roles(a)[2] == (((1, 4, ''),),)
    assert locant_roles(a)[3] == (((1, 1, ''),), ((1, 4, ''),))
    assert lower_locants(a, b) == -1


# ---- the parts of a name that are not PIN -------------------------------------------------------

def test_the_non_pin_parts_a_name_carries_are_counted_once_each():
    name = "4-({1-[4-(1-methyl-1,4-dihydropyridin-4-ylidene)cyclohexa-2,5-dien-1-ylidene]-1-azamethan-1-ium-1-yl}(oxo)methyl)benzene"
    provenance = {
        "non_pin_fragments": ("1-azameth", "{1-[4-(1-methyl-1,4-dihydropyridin-4-ylidene)cyclohexa-2,5-dien-1-ylidene]-1-aza",
                              "2-azaeth", "a fragment of a name that was discarded"),
        "non_pin_labels": ("1,4-dihydropyridin-4-ylidene",),
        "uncertified_pin_names": (),
    }
    # two constructs of the name ('2-azaeth' and the discarded fragment are not in it) and one label-only part
    assert non_pin_parts(name, provenance) == (2, 1)
    assert non_pin_parts(name, {"non_pin_fragments": ("1-azameth", "benzene")}) == (2, 0)
    assert non_pin_parts(name, {"uncertified_pin_names": ("benzene", "benzene")}) == (0, 1)
    # a fragment that holds another carried fragment is the enclosing substituent: only the innermost counts
    assert non_pin_parts("1-(2-methylpropyl)benzene", {"non_pin_fragments": ("methyl", "2-methylpropyl")}) == (1, 0)
    # a construct is counted as often as the name writes it: the same von Baeyer ring twice is two
    twice = "2-(bicyclo[4.3.0]nona-1(6),2,4,7-tetraen-7-ium-8-yl)-3-(bicyclo[4.3.0]nona-1,3,5,7-tetraen-7-ium-8-yl)benzene"
    assert non_pin_parts(twice, {"non_pin_fragments": ("bicyclo[4.3.0]",)}) == (2, 0)
    assert non_pin_parts(name, None) == (0, 0) and non_pin_parts(None, provenance) == (0, 0)
    assert non_pin_parts(name, {}) == (0, 0)


def test_the_constructs_that_are_not_pin_rank_first_and_the_label_only_parts_last():
    drawn = "1-({[4-(1-methylpyridin-1-ium-4-yl)phenyl]amino}diazenyl)benzene"
    other = "1-({1-[4-(1-methyl-1,4-dihydropyridin-4-ylidene)cyclohexa-2,5-dien-1-ylidene]-1-azamethan-1-ium-1-yl}diazenyl)benzene"
    assert lower_locants(drawn, other) == 0
    assert beats(drawn, other, {}, {"non_pin_fragments": ("1-azameth",)})
    assert not beats(other, drawn, {"non_pin_fragments": ("1-azameth",)}, {})
    # no record on either side, or equal records: a tie, the canonical order stands
    assert not beats(drawn, other, {}, {}) and not beats(other, drawn, {}, {})
    # the records of constructs outrank the lower locants of one skeleton...
    low, high = "4-methyl-1H-imidazol-3-ium", "5-methyl-1H-imidazol-3-ium"
    assert beats(low, high) and not beats(high, low)
    assert beats(high, low, {}, {"non_pin_fragments": ("methyl",)})
    #... and the label-only parts rank after them
    assert beats(low, high, {"non_pin_labels": ("methyl",)}, {}) and not beats(high, low, {}, {"non_pin_labels": ("methyl",)})
    assert beats(low, high, {}, {"non_pin_labels": ("methyl",)})


# ----: the form that keeps more of the input's configured stereo elements ranks first --------

def test_the_form_that_keeps_more_of_the_configuration_of_the_input_ranks_before_everything_else():
    # STERODESCRIPTORS (the Blue Book), "The configuration of an organic compound is systematically
    # indicated by one or more affixes added to a name that does not itself prescribe configuration" (:44570):
    # rh-forms review, row r1863. The drawn form names four E/Z bonds; the form with the shifted charge has the
    # parent that holds the cation (a)) and names none.
    drawn = ("disodium (2Z)-2-{(2E,4Z,6E)-4-chloro-7-[4-chloro-7,7-dimethyl-2-(4-sulfonatobutyl)-2,9-diazabicyclo"
             "[4.3.0]nona-1(6),2,4,8-tetraen-2-ium-8-yl]hepta-2,4,6-trien-1-ylidene}-3,3-dimethyl-5-sulfonato-1-"
             "(4-sulfonatobutyl)-2,3-dihydro-1H-indole")
    shifted = ("disodium 8-{4-chloro-7-[5-chloro-3,3-dimethyl-7-(4-sulfonatobutyl)-3,7-dihydro-2H-pyrrolo[2,3-b]"
               "pyridin-2-ylidene]hepta-1,3,5-trien-1-yl}-9,9-dimethyl-3-sulfonato-7-(4-sulfonatobutyl)-7-azabicyclo"
               "[4.3.0]nona-1(6),2,4,7-tetraen-7-ium")
    assert cationic_parent_centres(drawn) == (0, 0) and cationic_parent_centres(shifted) == (0, 1)
    assert beats(shifted, drawn)                                   # without the counts: (a) decides
    assert not beats(drawn, shifted)
    assert beats(drawn, shifted, None, None, 4, 0) and not beats(shifted, drawn, None, None, 0, 4)
    #... and a record of constructs that are not PIN on the drawn form does not move it back
    record = {"non_pin_fragments": ("2,9-diazabicyclo[4.3.0]",)}
    assert beats(drawn, shifted, record, {}, 4, 0)


def test_the_stereo_count_is_a_tie_when_equal_or_absent_and_the_other_criteria_decide():
    parent = "9-[1,4-di(naphthalen-1-yl)-1,2-dihydro-1,3-diazet-2-ylidene]-9-azahexacyclo[15.8.0.0^2,10]pentacosa-1(25),2(10)-dodecaen-9-ium"
    prefix = "9-[1,4-di(naphthalen-1-yl)-1,3-diazet-1-ium-2-yl]-9-azahexacyclo[15.8.0.0^2,10]pentacosa-1(17),2(10)-dodecaene"
    for mine, theirs in ((2, 2), (0, 0), (None, 3), (3, None), (None, None)):
        assert beats(parent, prefix, None, None, mine, theirs) and not beats(prefix, parent, None, None, theirs, mine)
    # the more stereo wins even where the other criteria prefer the other name, in both directions
    assert beats(prefix, parent, None, None, 2, 1) and not beats(parent, prefix, None, None, 1, 2)
    record = {"non_pin_fragments": ("1,3-diazet-1-ium",)}
    assert beats(prefix, parent, {}, record, 2, 1)                 # more stereo, though it carries a construct


# ---- the names of the audit's 41 renames: the 12 worse ones lose, the better ones win -------------

#: (the better name of one species, the worse; the criterion that decides, as the audit graded it)
RANKED = [
    # row 1 of the audit (2))
    ("1-[4,9,14,19-tetraphenyl-2,21,23,24-tetraazahexacyclo[13.5.1.1^3,20.1^5,8.1^10,13.0^18,21]tetracosa-1,3(22),4,6,8,10(23),11,13,15,17,19-undecaen-22-yl]pyridin-1-ium",
     "1-[4,9,14,19-tetraphenyl-2,21,23,24-tetraazahexacyclo[13.5.1.1^3,20.1^5,8.1^10,13.0^18,21]tetracosa-1,3,5,7,9,11,13(23),14,16,18,20(22)-undecaen-22-yl]pyridin-1-ium"),
    # row 2 of the audit (c))
    ("8-(7-{2-[7-(3-phenylphenyl)-5,7-diazatricyclo[6.4.0.0^2,6]dodeca-1(12),2(6),3,8,10-pentaen-11-yl]phenyl}-5,7-diazatricyclo[6.4.0.0^2,6]dodeca-1(8),2,4,5,9,11-hexaen-5-ium-11-yl)-6,8-diazatricyclo[7.4.0.0^2,7]trideca-1(13),2,4,6,9,11-hexaene",
     "8-(7-{2-[7-(3-phenylphenyl)-5,7-diazatricyclo[6.4.0.0^2,6]dodeca-1(12),2(6),3,8,10-pentaen-11-yl]phenyl}-5,7-diazatricyclo[6.4.0.0^2,6]dodeca-1,3,5,7,9,11-hexaen-7-ium-11-yl)-6,8-diazatricyclo[7.4.0.0^2,7]trideca-1(13),2,4,6,9,11-hexaene"),
    # row 4 of the audit (a))
    ("9-[1,4-di(naphthalen-1-yl)-1,2-dihydro-1,3-diazet-2-ylidene]-9-azahexacyclo[15.8.0.0^2,10.0^3,8.0^11,16.0^20,25]pentacosa-1(25),2(10),3,5,7,11,13,15,17,19,21,23-dodecaen-9-ium",
     "9-[1,4-di(naphthalen-1-yl)-1,3-diazet-1-ium-2-yl]-9-azahexacyclo[15.8.0.0^2,10.0^3,8.0^11,16.0^20,25]pentacosa-1(17),2(10),3,5,7,11,13,15,18,20,22,24-dodecaene"),
    # row 12 of the audit (g))
    ("1-methyl-2-[4-methylpentacyclo[18.4.0.0^2,7.0^8,13.0^14,19]tetracosa-1(24),2,4,6,8,10,12,14,16,18,20,22-dodecaen-3-yl]-3-(2,4,6-triphenylphenyl)-1,3-diazol-3-ium",
     "3-methyl-2-[4-methylpentacyclo[18.4.0.0^2,7.0^8,13.0^14,19]tetracosa-1(24),2,4,6,8,10,12,14,16,18,20,22-dodecaen-3-yl]-1-(2,4,6-triphenylphenyl)-1,3-diazol-3-ium"),
    # row 17 of the audit (a))
    ("4-{[(1E)-3-({(2E)-5-[(1E)-3-tert-butoxy-2-cyano-3-oxoprop-1-en-1-yl]-1-ethyl-3,3-dimethyl-2,3-dihydro-1H-indol-2-ylidene}methyl)-2-oxido-4-oxocyclobut-2-en-1-ylidene]methyl}-5-dodecyl-3,3-dimethyl-5-azatricyclo[7.4.0.0^2,6]trideca-1(13),2(6),4,7,9,11-hexaen-5-ium",
     "2-{[3-({3-[(1E)-3-tert-butoxy-2-cyano-3-oxoprop-1-en-1-yl]-7-ethyl-9,9-dimethyl-7-azabicyclo[4.3.0]nona-1(6),2,4,7-tetraen-7-ium-8-yl}methylidene)-2-oxido-4-oxocyclobut-1-en-1-yl]methylidene}-3-dodecyl-1,1-dimethyl-2,3-dihydro-1H-benzo[e]indole"),
    # row 21 of the audit (a))
    ("3-[(3-{[(2E)-3,3-dimethyl-5-sulfo-1-(4-sulfobutyl)-2,3-dihydro-1H-indol-2-ylidene]methyl}-2-hydroxy-4-oxocyclobut-2-en-1-ylidene)methyl]-2-methyl-8-nitro-2-azabicyclo[4.4.0]deca-1(6),2,4,7,9-pentaen-2-ium",
     "2-[(3-{[9,9-dimethyl-3-sulfo-7-(4-sulfobutyl)-7-azabicyclo[4.3.0]nona-1(6),2,4,7-tetraen-7-ium-8-yl]methylidene}-2-hydroxy-4-oxocyclobut-1-en-1-yl)methylidene]-1-methyl-6-nitro-1,2-dihydroquinoline"),
    # row 26 of the audit (1))
    ("6-(1-methyl-1,2-dihydropyridin-2-ylidene)-16-(1-methylpyridin-1-ium-2-yl)-11-(pentafluorophenyl)-20,21,22,23-tetraazapentacyclo[15.2.1.1^2,5.1^7,10.1^12,15]tricosa-1,3,5(23),7(22),8,10,12,14,16,18-decaen-20-ium",
     "16-(1-methyl-1,2-dihydropyridin-2-ylidene)-6-(1-methylpyridin-1-ium-2-yl)-11-(pentafluorophenyl)-20,21,22,23-tetraazapentacyclo[15.2.1.1^2,5.1^7,10.1^12,15]tricosa-1(19),2(23),3,5,7(22),8,10,12,14,17-decaen-20-ium"),
    # row 29 of the audit (c))
    ("(4R)-24-(amino{[(methylimino)methyl]imino}methyl)-7-(3-{[3,5-bis(difluoromethyl)cyclohexa-2,4-dien-1-yl]oxy}propyl)-11-chloro-27,28-dimethoxy-4,15-dimethyl-32-oxo-19-oxa-2,5,15,16,23-pentaazaheptacyclo[21.6.1.1^2,6.1^5,8.0^13,17.0^26,30.0^12,31]dotriaconta-1(30),6,8(31),9,11,13(17),14,24,26,28-decaen-15-ium",
     "(4R)-24-(amino{[(methylimino)methyl]imino}methyl)-7-(3-{[3,5-bis(difluoromethyl)cyclohexa-2,4-dien-1-yl]oxy}propyl)-11-chloro-27,28-dimethoxy-4,15-dimethyl-32-oxo-19-oxa-2,5,15,16,23-pentaazaheptacyclo[21.6.1.1^2,6.1^5,8.0^13,17.0^26,30.0^12,31]dotriaconta-1(30),6,8(31),9,11,13,16,24,26,28-decaen-16-ium"),
    # row 0 of the audit (c), (e)(i), (f))
    ("1-cyano-4-(3-{4-[3-(4-cyanophenyl)-1H-imidazol-3-ium-1-yl]-6-oxo-1,6-dihydro-1,3,5-triazin-2-yl}-1H-imidazol-3-ium-1-yl)benzene",
     "1-cyano-4-(3-{6-[1-(4-cyanophenyl)-1H-imidazol-3-ium-3-yl]-4-oxo-4,5-dihydro-1,3,5-triazin-2-yl}-1H-imidazol-3-ium-1-yl)benzene"),
    # row 5 of the audit (c))
    ("23-oxa-4,6,9,14,34-pentaazadecacyclo[22.10.1.0^1,6.0^2,22.0^3,19.0^4,12.0^5,10.0^13,18.0^28,35.0^29,34]pentatriaconta-2,4,7,9,11,13,15,17,19,21,24(35),25,27,29,31,33-hexadecaene-4,34-diium",
     "23-oxa-4,6,9,14,34-pentaazadecacyclo[22.10.1.0^1,6.0^2,22.0^3,19.0^4,12.0^5,10.0^13,18.0^28,35.0^29,34]pentatriaconta-2,5,7,9,11,13,15,17,19,21,24(35),25,27,29,31,33-hexadecaene-6,34-diium"),
    # row 6 of the audit (c))
    ("3-[4-(1-aminocyclobutyl)phenyl]-12-phenyl-5-(1,5,7,9-tetraazabicyclo[4.3.0]nona-2,4,6,8-tetraen-8-yl)-3,4,6,10-tetraazatricyclo[7.4.0.0^2,6]trideca-1(9),2,4,7,10,12-hexaen-3-ium",
     "3-[4-(1-aminocyclobutyl)phenyl]-12-phenyl-5-(1,5,7,9-tetraazabicyclo[4.3.0]nona-2,4,6,8-tetraen-8-yl)-3,4,6,10-tetraazatricyclo[7.4.0.0^2,6]trideca-1,4,6,8,10,12-hexaen-6-ium"),
    # row 8 of the audit (c))
    ("3-tert-butyl-7-[4-({4-[(3-carboxyphenyl)amino]-6-hydroxy-1,3,5-triazin-2-yl}amino)phenyl]-4-({2-chloro-4-[di(4-sulfonatobutyl)amino]phenyl}imino)-1,2,6,7-tetraazabicyclo[3.3.0]octa-1(8),2,5-trien-1-ium—sodium(1+) (1/2)",
     "3-tert-butyl-7-[4-({4-[(3-carboxyphenyl)amino]-6-hydroxy-1,3,5-triazin-2-yl}amino)phenyl]-4-({2-chloro-4-[di(4-sulfonatobutyl)amino]phenyl}imino)-1,2,6,7-tetraazabicyclo[3.3.0]octa-2,5,7-trien-7-ium—sodium(1+) (1/2)"),
    # row 10 of the audit (c))
    ("23,25-difluoro-3-methyl-15-(1-methylethyl)-26-(2-methyl-1H-pyrazol-2-ium-1-yl)-21-oxa-3,10,18,24-tetraazaheptacyclo[15.12.0.0^2,10.0^4,9.0^11,16.0^20,29.0^22,27]nonacosa-1(29),2,4,6,8,11,13,15,17,19,22(27),23,25-tridecaen-3-ium",
     "23,25-difluoro-3-methyl-15-(1-methylethyl)-26-(1-methyl-1H-pyrazol-2-ium-2-yl)-21-oxa-3,10,18,24-tetraazaheptacyclo[15.12.0.0^2,10.0^4,9.0^11,16.0^20,29.0^22,27]nonacosa-1,4,6,8,10,12,14,16,18,20(29),22(27),23,25-tridecaen-10-ium"),
    # row 14 of the audit (g))
    ("7-[4-(2-{3,5-bis[2-(1,4-dimethyl-6-phenylpyridin-1-ium-3-yl)phenyl]phenyl}phenyl)phenyl]-9-methyl-2,5,7,9-tetraazabicyclo[4.3.0]nona-1,3,5,7-tetraen-7-ium",
     "9-[4-(2-{3,5-bis[2-(1,4-dimethyl-6-phenylpyridin-1-ium-3-yl)phenyl]phenyl}phenyl)phenyl]-7-methyl-2,5,7,9-tetraazabicyclo[4.3.0]nona-1,3,5,7-tetraen-7-ium"),
    # row 16 of the audit (1))
    ("(1S)-4-(ethoxycarbonyl)-14-ethyl-12-methoxy-9,9,19,19-tetramethyl-5,15-diphenyl-2,22,23,25-tetraazaoctacyclo[11.9.1.1^1,8.1^3,21.0^2,6.0^16,23.0^18,22.0^11,25]pentacosa-3(24),4,6,8(25),11,13,15,17,21-nonaene-22,25-diium",
     "(1S)-4-(ethoxycarbonyl)-14-ethyl-12-methoxy-9,9,19,19-tetramethyl-5,15-diphenyl-2,22,23,25-tetraazaoctacyclo[11.9.1.1^1,8.1^3,21.0^2,6.0^16,23.0^18,22.0^11,25]pentacosa-3,5,7,11(25),12,14,16,18(22),21(24)-nonaene-22,25-diium"),
    # row 19 of the audit (a))
    ("3-({3-[(3-butyl-2,3-dihydro-1,3-benzothiazol-2-ylidene){3-[(9-butyl-7-thia-9-azabicyclo[4.3.0]nona-1,3,5,8-tetraen-9-ium-8-yl)methylidene]-2-oxido-4-oxocyclobut-1-en-1-yl}methyl]-2-hydroxy-4-oxocyclobut-2-en-1-ylidene}methyl)-2-methyl-2-azabicyclo[4.4.0]deca-1(10),2,4,6,8-pentaen-2-ium",
     "2-({3-[(9-butyl-7-thia-9-azabicyclo[4.3.0]nona-1,3,5,8-tetraen-9-ium-8-yl){3-[(9-butyl-7-thia-9-azabicyclo[4.3.0]nona-1,3,5,8-tetraen-9-ium-8-yl)methylidene]-2-oxido-4-oxocyclobut-1-en-1-yl}methylidene]-2-hydroxy-4-oxocyclobut-1-en-1-yl}methylidene)-1-methyl-1,2-dihydroquinoline"),
    # row 22 of the audit (2))
    ("21-(dimethylamino)-5,10,15,23-tetraphenyl-20,22,24,25-tetraazahexacyclo[17.3.1.1^6,9.1^11,14.0^4,22.0^16,20]pentacosa-1(23),2,4(22),5,7,9,11(24),12,14,16,18-undecaen-22-ium",
     "21-(dimethylamino)-5,10,15,23-tetraphenyl-20,22,24,25-tetraazahexacyclo[17.3.1.1^6,9.1^11,14.0^4,22.0^16,20]pentacosa-1(22),2,4,6(25),7,9,11,13,15,17,19(23)-undecaen-22-ium"),
    # row 24 of the audit (e)(i))
    ("17-(1-methyl-1,4-dihydropyridin-4-ylidene)-7-(1-methylpyridin-1-ium-4-yl)-2,12-diphenyl-21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11(23),12,14,16(22),18-undecaen-21-ide",
     "12-(1-methyl-1,4-dihydropyridin-4-ylidene)-2-(1-methylpyridin-1-ium-4-yl)-7,17-diphenyl-21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11(23),13(22),14,16,18-undecaen-21-ide"),
    # row 27 of the audit (g))
    ("13-(2-carboxy-5-{[(2,5-dioxopyrrolidin-1-yl)oxy](oxo)methyl}phenyl)-6,7,7,19,19,20-hexamethyl-9-(sulfomethyl)-17-(sulfonatomethyl)-2-oxa-6,20-diazapentacyclo[12.8.0.0^3,12.0^5,10.0^16,21]docosa-1(22),3,5,8,10,12,14,16(21),17-nonaen-6-ium",
     "13-(2-carboxy-5-{[(2,5-dioxopyrrolidin-1-yl)oxy](oxo)methyl}phenyl)-6,7,7,19,19,20-hexamethyl-17-(sulfomethyl)-9-(sulfonatomethyl)-2-oxa-6,20-diazapentacyclo[12.8.0.0^3,12.0^5,10.0^16,21]docosa-1(22),3,5,8,10,12,14,16(21),17-nonaen-6-ium"),
    # row 30 of the audit (c))
    ("7-methyl-8-{4-[(1Z)-{[(2-{(1E)-[4-(7-methyl-1,7-diazabicyclo[4.3.0]nona-1,3,5,8-tetraen-1-ium-8-yl)phenyl]methylidene}hydrazinyl)(oxo)methyl]hydrazinylidene}methyl]phenyl}-1,7-diazabicyclo[4.3.0]nona-1,3,5,8-tetraen-1-ium",
     "7-methyl-8-{4-[(1Z)-{[(2-{(1E)-[4-(7-methyl-1,7-diazabicyclo[4.3.0]nona-2,4,6,8-tetraen-7-ium-8-yl)phenyl]methylidene}hydrazinyl)(oxo)methyl]hydrazinylidene}methyl]phenyl}-1,7-diazabicyclo[4.3.0]nona-2,4,6,8-tetraen-7-ium"),
    # row 33 of the audit (a))
    ("12-(1-methyl-1,4-dihydropyridin-4-ylidene)-7-(1-methylpyridin-1-ium-4-yl)-2,17-diphenyl-21-oxa-22,23,24-triazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11(23),13(22),14,16,18-undecaen-23-ium",
     "7,12-bis(1-methylpyridin-1-ium-4-yl)-2,17-diphenyl-21-oxa-22,23,24-triazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11,13(22),14,16,18-undecaene"),
]


@pytest.mark.parametrize("better, worse", [(b, w) for b, w in RANKED])
def test_the_audit_names_are_ranked_as_the_audit_graded_them(better, worse):
    assert beats(better, worse), (better, worse)
    assert not beats(worse, better), (better, worse)


# The label ranks the names of the charge forms first; below the PIN a form that has lost the configuration of the
# input is the name of a less specific compound STEREODESCRIPTORS, the Blue Book), so the form that keeps
# more of it stands over a higher label below the PIN (labels: pin_verified 4, pin_unverified 3, systematic_verified 2,
# best_effort 1; a label is ``(rank, 0)``).
@pytest.mark.parametrize("label, incumbent, kept, incumbent_kept, expected", [
    ((2, 0), (3, 0), 1, 0, 1),        # systematic_verified keeping the configuration over pin_unverified without it
    ((3, 0), (2, 0), 0, 1, -1),       # and the reverse
    ((2, 0), (3, 0), 1, 1, 0),        # equal counts: the labels decide
    ((2, 0), (3, 0), None, 0, 0),     # a count that is not known: the labels decide
    ((2, 0), (4, 0), 1, 0, 0),        # the PIN label outranks, as before
    ((4, 0), (3, 0), 1, 0, 0),        # a PIN label keeps its rank
    ((2, 0), (1, 0), 0, 1, 0),        # the last-resort floor (best_effort) is not compared by it, either way
    ((1, 0), (3, 0), 1, 0, 0),
    ((0, 0), (3, 0), 1, 0, 0),        # a form that is not named never wins by it
])
def test_the_stereo_a_form_keeps_ranks_before_a_label_below_the_pin(label, incumbent, kept, incumbent_kept, expected):
    assert stereo_decides_below_pin(label, incumbent, kept, incumbent_kept, 4) == expected

