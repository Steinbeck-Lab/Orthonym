""": one citation of the substituents of a mononuclear hydride, for the parent and for its groups.

 (the Blue Book, under: "For mononuclear parent hydrides with two or more
substituents the first cited substituent never has enclosing marks unless it includes a locant. The second and
further substituents are each enclosed with parentheses even for simple substituents. When the simple
substituent groups are accompanied by multiplicative prefixes such as 'di' and 'tri', the multiplicative
prefixes are not included in the parentheses."

The silyl and germyl prefix groups used to cite their substituents without the marks ('4-(dimethylpropylsilyl)
benzoic acid', '2-(chlorodimethylsilyl)benzoic acid') while the phosphanyl group and every parent hydride
(silane, phosphane, borane) cited them with the marks. All of them now take
``prefix_marks.mononuclear_substituent_block``. The Blue Book spells the group the way it spells the parent:
'4-[ethyl(methyl)phosphanyl]-1H-imidazole (PIN)' (:39228), '3-[amino(methyl)silyl]-3-[(aminomethyl)silyl]
cyclopentan-1-ol (PIN)' (:3545), 'dihydroxy(sulfanyl)silyl (preselected prefix)' (:36393) and, with a multiplied
second group, 'tert-butoxytri(phenyl)-lambda5-phosphanyl (PIN)' (:40442); the parent names it takes the same
block from are 'ethyldi(methyl)phosphane (PIN)' (:7290), 'tert-butyldi(methyl)phosphane (PIN)' (:16286),
'chlorodi(methyl)borane (PIN)' (:25866), 'trichloro(methyl)silane' (:35754) and '(difluoroboranyl)tri(methyl)
silane (PIN)' (:37520).

RULING (coordinator, 2026-10-24; ASSUMED-book-inconsistency): for a multiplied later group of a silyl or germyl
group the book contradicts itself. Three places cite it bare:
  (tert-butylperoxy)dimethylsilyl propanoate (PIN) the Blue Book
  aminodichlorosilyl (preselected prefix) the Blue Book
  N1-{2-[(acetyldimethylsilyl)methoxy]ethyl}-... (an alternative name under, not its PIN) the Blue Book
while the rule text (:7272, maintained with an erratum in the online edition), its parents
('ethyldi(methyl)phosphane (PIN)':7290, 'tert-butyldi(methyl)phosphane (PIN)':16286) and the groups
'tert-butoxytri(phenyl)-lambda5-phosphanyl (PIN)' (:40442), 'methyldi(phenyl)phosphaniumyl' (:42468) enclose it. The
rule text is followed. The one BB row of the three that is a whole molecule, 51.4.1.2, is not a MATCH before or
after this change.
"""
import pytest

from orthonym.assembly.prefix_marks import mononuclear_substituent_block
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.unit


# The block of each row is the part of the book's own name that stands between the (absent) locant and the
# hydride stem; the quote is the name it is taken from.
BLOCK_ROWS = [
    # first group bare, later groups enclosed, multiplier outside the marks
    (["ethyl", "methyl", "methyl"], "ethyldi(methyl)", "ethyldi(methyl)phosphane (PIN), :7290"),
    (["tert-butyl", "methyl", "methyl"], "tert-butyldi(methyl)", "tert-butyldi(methyl)phosphane (PIN), :16286"),
    (["chloro", "methyl", "methyl"], "chlorodi(methyl)", "chlorodi(methyl)borane (PIN), :25866"),
    (["chloro", "chloro", "chloro", "methyl"], "trichloro(methyl)", "trichloro(methyl)silane, :35754"),
    (["ethyl", "methyl"], "ethyl(methyl)", "4-[ethyl(methyl)phosphanyl]-1H-imidazole (PIN), :39228"),
    (["amino", "methyl"], "amino(methyl)", "3-[amino(methyl)silyl]...cyclopentan-1-ol (PIN), :3545"),
    (["hydroxy", "hydroxy", "sulfanyl"], "dihydroxy(sulfanyl)", "dihydroxy(sulfanyl)silyl, :36393"),
    (["butyl", "ethyl", "methyl", "propyl"], "butyl(ethyl)(methyl)(propyl)",
     "butyl(ethyl)(methyl)(propyl)silane (PIN), :7278"),
    (["methyl", "phenyl", "phenyl"], "methyldi(phenyl)", "methyldi(phenyl)phosphaniumyl in a PIN, :42468"),
    # the first group stands bare whatever its multiplier
    (["methyl", "methyl", "methyl"], "trimethyl", "1-(trimethylsilyl)ethan-1-one (PIN), :29380"),
    (["methoxy", "methoxy", "methoxy"], "trimethoxy", "trimethoxysilyl (preferred prefix), :36389"),
    (["methyl", "methyl", "methyl", "methyl"], "tetramethyl", "tetramethylsilane"),
    # alphanumerical order ignores the multiplier and the italic prefix,
    (["methyl", "propyl", "methyl"], "dimethyl(propyl)", "dimethyl before propyl: P-14.5.1, :3448"),
    (["methyl", "tert-butyl"], "tert-butyl(methyl)", "tert-butyl cited at 'butyl': P-14.5.2, :3477"),
    # a compound group takes the marks of its own name wherever it stands
    (["methyl", "2-methylphenyl"], "methyl(2-methylphenyl)", "[amino(methyl)...] with a compound group"),
    (["methyl", "methyl", "2-methylphenyl"], "dimethyl(2-methylphenyl)", "P-14.5.1 order, P-16.5.1.1, :7232"),
    (["iodomethyl", "methyl", "methyl", "methyl"], "(iodomethyl)tri(methyl)",
     "'(difluoroboranyl)tri(methyl)silane (PIN)', :37520: a compound first group in its own marks"),
]


@pytest.mark.parametrize("names,block,quote", BLOCK_ROWS)
def test_block_of_the_book(names, block, quote):
    assert mononuclear_substituent_block(names) == block, quote


def test_block_does_not_depend_on_the_order_the_names_arrive_in():
    """A set of groups has one citation: the walk order of the structure is no part of it (determinism)."""
    names = ["propyl", "methyl", "ethyl", "methyl"]
    expected = mononuclear_substituent_block(names)
    assert expected == "ethyldi(methyl)(propyl)"
    for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
        assert mononuclear_substituent_block([names[i] for i in perm]) == expected


def test_block_bound_is_the_callers():
    five = ["methyl"] * 5
    assert mononuclear_substituent_block(five) == "pentamethyl"
    assert mononuclear_substituent_block(five, max_count=4) is None
    assert mononuclear_substituent_block(["methyl"] * 4, max_count=4) == "tetramethyl"


def test_all_the_writers_are_the_one_block():
    """The parent hydride, the phosphanyl group, the onium group, the silyl group, the pseudoketone hub, the
    lambda-sulfane family, the a(ba)n chains, diazene, nitramide and the ligands of a hydride parent reach the
    block (or its one-group token) through ``prefix_marks``; none keeps a loop of its own (one home per rule).
    Not routed: the composed 'amino' prefix of ``composer`` (its marks and multiplier come from the
    composed-prefix citation; the comment at the site says why) and the locanted hydrazine citation."""
    import inspect

    from orthonym.assembly import substituent_naming
    from orthonym.rules import (catenated_hydrides, ions, mononuclear_hydrides, nitramide, organometallics,
                                phosphorus, polyazane, pseudoketones)

    for fn, home in ((substituent_naming._name_group14_substituent, "mononuclear_group_name"),
                     (substituent_naming._compose_n_substituent_prefix, "mononuclear_substituent_block"),
                     (phosphorus._build_substituent_string, "mononuclear_substituent_block"),
                     (mononuclear_hydrides._build_mixed_substituent_string, "mononuclear_substituent_block"),
                     (mononuclear_hydrides._compose_io_prefixes, "mononuclear_substituent_block"),
                     (catenated_hydrides._try_aba_parent, "mononuclear_substituent_block"),
                     (nitramide._name_single_nitramide, "mononuclear_substituent_block"),
                     (polyazane._format_n2_substituents, "mononuclear_substituent_block"),
                     (organometallics._ligand_token, "mononuclear_group_token"),
                     (pseudoketones.name_acyl_hetero_pseudoketone, "mononuclear_substituent_block"),
                     (ions._assemble_unlocanted_ium, "mononuclear_substituent_block"),
                     (ions.emit_onium_hydride_parent, "mononuclear_substituent_block")):
        src = inspect.getsource(fn)
        assert home in src, fn.__name__
        assert "Counter" not in src, fn.__name__


def test_the_group_token_is_the_decision_of_one_group():
    """First group bare, later groups enclosed with the multiplier outside, a group the caller knows to be
    compound enclosed as the first too, and a group cited already enclosed never doubled."""
    from orthonym.assembly.prefix_marks import mononuclear_group_token as tok

    assert tok(1, "methyl", first=True) == "methyl"
    assert tok(2, "methyl", first=True) == "dimethyl"
    assert tok(1, "methyl", first=False) == "(methyl)"
    assert tok(2, "methyl", first=False) == "di(methyl)"
    assert tok(2, "tert-butyl", first=True) == "di-tert-butyl"
    assert tok(1, "acetyloxy", first=True, enclosed=True) == "(acetyloxy)"
    assert tok(1, "(2-chloroethyl)", first=False) == "(2-chloroethyl)"


def test_a_silyl_group_with_substituents_is_a_substituted_group():
    """ (c) (the Blue Book) "any component which is substituted automatically requires use of the
    multiplicative forms 'bis', 'tris', etc.": the group says so itself, so the enclosed second group
    ('dimethyl(propyl)silyl') that the text of the name does not show as substituted still takes 'bis'."""
    from orthonym.assembly.naming_utils import get_multiplier_prefix, multiplied_component
    from orthonym.assembly.prefix_marks import mononuclear_group_name

    group = mononuclear_group_name(["methyl", "methyl", "propyl"], "silyl")
    assert group == "dimethyl(propyl)silyl"
    assert get_multiplier_prefix(2, group) == "bis"
    assert multiplied_component(2, group, f"[{group}]") == "bis[dimethyl(propyl)silyl]"
    assert mononuclear_group_name([], "silyl") == "silyl"


# The molecule is named at both tiers; the name must be the PIN and must read back to the molecule (OPSIN, a fresh
# call outside the engine: tests.support.rt_assert).
GROUP_ROWS = [
    # the second group of a silyl / germyl is enclosed;:3545 'amino(methyl)silyl')
    ("CCC[Si](C)(C)c1ccc(cc1)C(=O)O", "4-[dimethyl(propyl)silyl]benzoic acid"),
    ("CCC[Ge](C)(C)c1ccc(cc1)C(=O)O", "4-[dimethyl(propyl)germyl]benzoic acid"),
    ("CC[SiH](C)c1ccc(cc1)C(=O)O", "4-[ethyl(methyl)silyl]benzoic acid"),
    ("CCC[SiH](c1ccccc1)c1ccc(cc1)C(=O)O", "4-[phenyl(propyl)silyl]benzoic acid"),
    # a multiplied second group keeps its multiplier outside the marks (:7290,:40442)
    ("CC[Si](C)(C)CCc1ccc(cc1)C(=O)O", "4-{2-[ethyldi(methyl)silyl]ethyl}benzoic acid"),
    ("C[Si](C)(Cl)c1ccccc1C(=O)O", "2-[chlorodi(methyl)silyl]benzoic acid"),
    ("O=C(O)c1ccc(cc1)[Si](C)(C)N", "4-[aminodi(methyl)silyl]benzoic acid"),
    # the book's own group name (:36393)
    ("O=C(O)c1ccc(cc1)[Si](O)(O)S", "4-[dihydroxy(sulfanyl)silyl]benzoic acid"),
    # the group multiplied: a substituted group takes 'bis' (c),:7035), the second group of each is enclosed
    ("CC(C)(C)[Si](C)(C)C1=C(C(C1=O)(Cl)Cl)[Si](C)(C)C(C)(C)C",
     "2,3-bis[tert-butyldi(methyl)silyl]-4,4-dichlorocyclobut-2-en-1-one"),
    ("CCC[Si](C)(C)C1=C(C(C1=O)(Cl)Cl)[Si](C)(C)CCC",
     "4,4-dichloro-2,3-bis[dimethyl(propyl)silyl]cyclobut-2-en-1-one"),
    # a first group stands bare, whatever its multiplier (:29380,:36389)
    ("C[Si](C)(C)c1ccc(cc1)C(=O)O", "4-(trimethylsilyl)benzoic acid"),
    ("O=C(O)c1ccc(cc1)[Si](OC)(OC)OC", "4-(trimethoxysilyl)benzoic acid"),
    # the groups that already followed the rule are unchanged (:39228)
    ("CC[P](C)c1c[nH]cn1", "4-[ethyl(methyl)phosphanyl]-1H-imidazole"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", GROUP_ROWS)
def test_group_is_spelled_as_its_parent_hydride_is(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


PARENT_ROWS = [
    # the parents, whose spelling the groups now follow
    ("CCP(C)C", "ethyldi(methyl)phosphane"),
    ("CC[Si](C)(C)C", "ethyltri(methyl)silane"),
    # puts a compound FIRST group in its own marks and encloses the second: the multiplier of
    # 'methyl' stays outside, as in '(difluoroboranyl)tri(methyl)silane (PIN)' (:37520)
    ("C[Si](C)(C)CI", "(iodomethyl)tri(methyl)silane"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PARENT_ROWS)
def test_parent_hydride_block_is_unchanged(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
