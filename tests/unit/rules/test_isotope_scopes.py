""" /: isotope labels in several scopes of one name.

The decorator of ``rules.isotopes`` places one descriptor per attachment atom, each found alone,
and asks for integer locants only. ``rules.isotope_scopes`` composes the name from what OPSIN
reads: every probe (one label at one slot with one locant token) is bound to the atom it labels,
the labelled atoms of the input are covered by probes, and the cover is spelled with one descriptor
per slot and read back whole.

Cited rules (the Blue Book, with their section headings):

* NAMES,: the descriptor stands "before the part of the compound that is
  isotopically substituted"; several independently scoped descriptors in one name
  ('1-(13C)methyl(2-13C)benzene (PIN)'), one descriptor per unit of a multiplied group
  ('1,2-di[(13C)methyl]benzene (PIN)'); nuclide symbols "at the same place in the name" are cited
  together, alphabetically and then by mass number.
* Different isotopic modifications on otherwise identical substituents: "they are
  cited separately" when they cannot be combined with 'di-', 'bis-'.
*: in a name of several words the descriptor stands before the word or part that holds
  the nuclide.
* ORDER OF NUCLIDE SYMBOLS, /: alphabetical, then increasing mass number.
* Omission of locants, (all locants cited once one is needed) and
  (none for a completely labelled group): '(2H5)phenyl'.
* Multiple types of enclosing marks,: the parentheses of a descriptor count
  in the nesting order.

The expected names are what OPSIN reads back to the input (each is checked again here by an
independent OPSIN call); the PIN is not claimed (the composer's names are labelled below it).
"""
import re

import pytest
from rdkit import Chem

from orthonym.rules import isotope_scopes as S
from orthonym.rules import isotopes as I
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit


def _compose(smiles, skeleton):
    original = Chem.MolFromSmiles(smiles)
    stripped, label_map = I.strip_isotopes(original)
    return S.compose_scopes(skeleton, original, label_map, stripped.GetNumHeavyAtoms())


# (id, smiles, skeleton as the engine names the isotope-stripped molecule, the composed name, rule)
CASES = [
    ("symmetric-collision-13C-and-D",
     "[2H][13C]([13CH3])([13CH3])[13C](=O)[13C](=O)O",
     "3-methyl-2-oxobutanoic acid",
     "3-(13C)methyl-2-oxo(1,2,3,4-13C4,3-2H)butanoic acid",
     "P-82.2.1, P-82.3: one methyl is the substituent, the other the end of the chain; the "
     "carbons of the chain and the hydrogen nuclide of C-3 are one place, alphabetical"),
    ("multiplied-unit-with-another-copy-elsewhere",
     "[2H]CC(=O)OC[C@@H]1[C@H]([C@@H]([C@@H](C(O1)CC(=O)N)OC(=O)C[2H])OC(=O)C[2H])OC(=O)C[2H]",
     "(2R,3R,4R,5R)-3,4,5-tris(acetyloxy)-2-[(acetyloxy)methyl]-6-(2-amino-2-oxoethyl)oxane",
     "(2R,3R,4R,5R)-3,4,5-tris[(2-2H1)acetyloxy]-2-{[(2-2H1)acetyloxy]methyl}"
     "-6-(2-amino-2-oxoethyl)oxane",
     "P-82.2.1, P-16.5.4.1.3: one descriptor inside the multiplied group, one in the fourth acetyl"),
    ("two-multiplied-methyls-beside-other-labels",
     "[2H][C@](COC([2H])([2H])[2H])(C1=CC=C(C=C1)C(F)(F)F)N2CCN(C[C@@H]2C)C3(CCN(CC3)C(=O)"
     "C4=C(N=CN=C4C([2H])([2H])[2H])C([2H])([2H])[2H])C",
     "5-{4-[(3S)-4-{(1R)-2-methoxy-1-[4-(trifluoromethyl)phenyl]ethyl}-3-methylpiperazin-1-yl]"
     "-4-methylpiperidine-1-carbonyl}-4,6-dimethylpyrimidine",
     "5-{4-[(3S)-4-{(1R)-2-(2H3)methoxy-1-[4-(trifluoromethyl)phenyl](1-2H)ethyl}-3-methylpiperazin"
     "-1-yl]-4-methylpiperidine-1-carbonyl}-4,6-di[(2H3)methyl]pyrimidine",
     "P-82.2.1, P-82.6.1.3: 'di[(2H3)methyl]' and '(2H3)methoxy' are completely labelled"),
    ("two-phenylenes-each-with-four-labels",
     "[2H]C1=C(C(=C(C(=C1C2=CC=CC=C2)[2H])[2H])N(C3=CC=CC(=C3)C4=CC=CC5=C4SC6=CC=CC=C56)"
     "C7=C(C(=C(C(=C7[2H])[2H])C8=CC=CC9=CC=CC=C98)[2H])[2H])[2H]",
     "4-(3-{[4-(naphthalen-1-yl)phenyl](4-phenylphenyl)amino}phenyl)dibenzo[b,d]thiophene",
     "4-(3-{[4-(naphthalen-1-yl)(2H4)phenyl][4-phenyl(2H4)phenyl]amino}phenyl)dibenzo[b,d]thiophene",
     "P-82.2.1, P-82.6.1.3: a descriptor before each ring, none cited as 2,2,3,3"),
    ("primed-locants-in-a-ring-assembly-substituent",
     "[2H]C1=C(C(=C(C(=C1[2H])[2H])C2=CC(=CC=C2)[Si](C3=CC=CC=C3)(C4=CC=CC(=C4)C5=C(C(=C(C(=C5"
     "[2H])[2H])[2H])[2H])[2H])C6=CC=CC7=C6C8C9=CC=CC=C9C7C1=C8C(=CC=C1)N1C2=C(C=C(C=C2)[Si]"
     "(C2=CC=CC=C2)(C2=CC=CC=C2)C2=CC=CC=C2)C2=CC=CC=C21)[2H])[2H]",
     "9-{13-[di([1,1'-biphenyl]-3-yl)phenylsilyl]pentacyclo[6.6.6.0^2,7.0^9,14.0^15,20]icosa-"
     "2,4,6,9,11,13,15,17,19-nonaen-3-yl}-3-(triphenylsilyl)-9H-carbazole",
     "9-(13-{di[(2',3',4',5',6'-2H5)[1,1'-biphenyl]-3-yl]phenylsilyl}pentacyclo[6.6.6.0^2,7.0^9,14."
     "0^15,20]icosa-2,4,6,9,11,13,15,17,19-nonaen-3-yl)-3-(triphenylsilyl)-9H-carbazole",
     "P-82.2.1, P-16.5.4.1.3: the terminal ring of the assembly needs primed locants, inside "
     "the multiplied group"),
    ("stem-of-an-assembly-with-primes-and-colons",
     "[2H]C1=C(C(=C(C(=C1[2H])C2=CC=CC(=C2)C3=CC=C(C=C3)C4=CC=CC=C4)[2H])C5=C(C(=C(C(=C5[2H])"
     "[2H])C6=C(C(=C(C(=C6[2H])[2H])[2H])C7=C(C(=C(C(=C7[2H])C8=C(C(=C(C(=C8[2H])[2H])C9=C(C(="
     "C(C(=C9[2H])[2H])C)[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H]",
     "4-methyl-1,1':4',1'':3'',1''':3''',1'''':4'''',1''''':3''''',1'''''':3'''''',"
     "1''''''':4''''''',1''''''''-noviphenyl",
     "4-methyl(2,2',2'',2''',2'''',2''''',3,3',3'''',4'',4''',4''''',5,5',5'',5''',5'''',5''''',"
     "6,6',6'',6''',6'''',6'''''-2H24)-1,1':4',1'':3'',1''':3''',1'''':4'''',1''''':3''''',"
     "1'''''':3'''''',1''''''':4''''''',1''''''''-noviphenyl",
     "P-82.2.1: the descriptor stands before the locant set of the assembly stem, whose locants "
     "carry up to eight primes"),
    ("one-methyl-of-each-multiplied-pair",
     "[2H]C1(CCCCC1)C2=C(C=C(C(=C2)C3=[N+](C=CC(=C3)C([2H])([2H])[2H])C)C)C([2H])([2H])[2H]",
     "2-(5-cyclohexyl-2,4-dimethylphenyl)-1,4-dimethylpyridin-1-ium",
     "2-[5-(1-2H)cyclohexyl-2-methyl-4-(2H3)methylphenyl]-1-methyl-4-(2H3)methylpyridin-1-ium",
     "P-82.2.2.1: the labelled methyl is cited apart from the unlabelled one"),
    ("a-pair-multiplied-again-when-no-copy-is-labelled",
     "[2H]C1(C2=C(C=C(C=C2)C3=C(C=CC4=C3OC5=C4C=CC(=C5C6=[N+](C=C(C(=C6)F)C([2H])([2H])[2H])C)C)F)"
     "C(C1(C)C)([2H])[2H])[2H]",
     "2-[6-(2,2-dimethyl-2,3-dihydro-1H-inden-5-yl)-7-fluoro-3-methyldibenzo[b,d]furan-4-yl]-4-"
     "fluoro-1,5-dimethylpyridin-1-ium",
     "2-{6-[2,2-dimethyl-2,3-dihydro(1,1,3,3-2H4)-1H-inden-5-yl]-7-fluoro-3-methyldibenzo[b,d]furan"
     "-4-yl}-4-fluoro-1-methyl-5-(2H3)methylpyridin-1-ium",
     "P-82.2.2.1: only the pair that holds the labelled copy is cited apart"),
    ("equivalent-carbazoles-labelled-on-both",
     "[2H]C1=C(C(=C2C(=C1[2H])C3=CC=CC=C3N2C4=NC(=NC(=N4)C5=CC=CC(=C5N6C7=CC=CC=C7C8=CC=CC=C86)"
     "C9=CC=CC=C9)N1C2=CC=CC=C2C2=C(C(=C(C(=C21)[2H])[2H])[2H])[2H])[2H])[2H]",
     "9-{6-(9H-carbazol-9-yl)-4-[2-(9H-carbazol-9-yl)-3-phenylphenyl]-1,3,5-triazin-2-yl}-9H-carbazole",
     "9-{6-[(1,2,3,4-2H4)-9H-carbazol-9-yl]-4-[2-(9H-carbazol-9-yl)-3-phenylphenyl]-1,3,5-triazin-2-yl}"
     "(1,2,3,4-2H4)-9H-carbazole",
     "P-82.2.1: two symmetry-equivalent carbazoles, four labels on each; a label bound to an atom "
     "and not to a symmetry class"),
    ("unit-of-a-multiplied-group-closes-before-the-next-descriptor",
     "[2H]C(=O)N(C([2H])([2H])[2H])C([2H])([2H])[2H]",
     "N,N-dimethylformamide",
     "N,N-di[(2H3)methyl](2H1)formamide",
     "P-82.2.1, P-16.5.4.1.3: '(2H3)methyl' stands inside the brackets of 'di[ ]', the descriptor of "
     "the formyl hydrogen after them"),
    ("one-methyl-of-six-cited-apart-without-locants",
     "[13CH3]c1c(C)c(C)c(C)c(C)c1C",
     "hexamethylbenzene",
     "(13C)methylpentamethylbenzene",
     "P-82.2.2.1: the labelled copy is cited apart from the five unlabelled ones, which stay "
     "multiplied (P-14.3.4.5: no locants where every position is substituted)"),
    ("tert-butyl-completely-labelled-beside-an-unlabelled-one",
     "[2H]C([2H])([2H])C(C([2H])([2H])[2H])(C([2H])([2H])[2H])c1ccc(cc1)C(C)(C)C",
     "1,4-di-tert-butylbenzene",
     "1-(2H9)tert-butyl-4-tert-butylbenzene",
     "P-82.6.1.3: the nine hydrogens of one tert-butyl are all labelled, so no locants; "
     "P-82.2.2.1: the two groups are cited apart"),
    ("methyl-of-tert-butyl-in-the-systematic-prefix",
     "[2H]C([2H])([2H])C(C)(C)c1ccccc1",
     "tert-butylbenzene",
     "[2-(2H3)methylpropan-2-yl]benzene",
     "P-82.6.3.2: the retained prefix does not number its methyl groups, the systematic one does"),
    ("labelled-carbons-and-nitrogen-of-a-nitrile-suffix",
     "[13CH3]C([13C]#[15N])C1=CC2=CC=CC=C2N1",
     "2-(1H-indol-2-yl)propanenitrile",
     "2-(1H-indol-2-yl)(1,3-13C2)propane(15N)nitrile",
     "P-82.2.1, P-82.3.1: the nitrogen of the suffix has a descriptor of its own, before the suffix"),
    ("multiplied-tert-butyl-in-marks",
     "c1cc(C(C([2H])([2H])[2H])(C([2H])([2H])[2H])C([2H])([2H])[2H])cc"
     "(C(C([2H])([2H])[2H])(C([2H])([2H])[2H])C([2H])([2H])[2H])c1",
     "1,3-di-tert-butylbenzene",
     "1,3-di[(2H9)tert-butyl]benzene",
     "P-82.2.1, P-16.5.4.1.3: the descriptor of a unit of a multiplied group stands inside its marks, "
     "'1,2-di[(13C)methyl]benzene (PIN)'; the hyphen of 'di-tert-butyl' gives way to the opening mark"),
    ("numbered-oxygen-locant-in-a-diol",
     "[2H]Oc1ccc(O)cc1[2H]",
     "benzene-1,4-diol",
     "(O1,2-2H2)benzene-1,4-diol",
     "P-82.6.1.1, P-82.6.1.4: two oxygens that can hold the label, so the oxygen is cited by its number; "
     "P-14.4 NUMBERING (i) (BlueBookV2.md:3336): the lowest locants to the modified atoms, 2 and not 6"),
    ("numbered-oxygen-locant-in-a-diacid",
     "[2H]OC(=O)CC[13CH2]C(=O)O",
     "pentanedioic acid",
     "(4-13C,O1-2H)pentanedioic acid",
     "P-82.6.1.1, P-82.6.1.4: the nuclides of one place are alphabetical (P-82.3.1); the bare letter "
     "is not cited where the parent holds two such oxygens"),
    ("bare-nitrogen-locant-where-the-parent-holds-one-nitrogen",
     "CC(=O)N([2H])c1ccc(O)cc1",
     "N-(4-hydroxyphenyl)acetamide",
     "N-(4-hydroxyphenyl)(N-2H)acetamide",
     "P-82.2.5 (BlueBookV2.md:43824): '(N-2H1)acetamide (PIN)'; P-82.6.1.2 (:44190): one atom of "
     "the element in the unit"),
    ("ring-nitrogen-takes-its-ring-numeral-beside-a-letter-locant",
     "[2H]n1c([2H])c(CC(=O)N[2H])c2ccccc12",
     "2-(1H-indol-3-yl)acetamide",
     "2-[(1,2-2H2)-1H-indol-3-yl](N-2H1)acetamide",
     "P-82.6.2: the nitrogen of the ring is a skeletal atom numbered 1; the italic letter denotes the "
     "nitrogen of the suffix (rules/isotope_letters.py, denotable_atoms)"),
    ("hydroxyl-deuterium-in-one-of-two-hydroxy-prefixes",
     "[2H]OC[C@@H]([C@H](C)O)N1C(=O)C2=C3C(=CC=C4C3=C(C=C2)C5=C6C4=CC=C7C6=C(C=C5)C(=O)N(C7=O)"
     "C(C)C)C1=O",
     "2-[(2S,3S)-1,3-dihydroxybutan-2-yl]-9-(propan-2-yl)anthra[2,1,9-def:6,5,10-d'e'f']"
     "diisoquinoline-1,3,8,10(2H,9H)-tetrone",
     "2-[(2S,3S)-1-(2H1)hydroxy-3-hydroxybutan-2-yl]-9-(propan-2-yl)anthra[2,1,9-def:6,5,10-d'e'f']"
     "diisoquinoline-1,3,8,10(2H,9H)-tetrone",
     "P-82.2.2.1, P-82.6.1.3: the labelled hydroxy is cited apart; its one hydrogen is all of it, "
     "so no locant (OPSIN reads the count shown, not '(2H)hydroxy')"),
]


@pytest.mark.unit
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,skeleton,expected", [c[1:4] for c in CASES], ids=[c[0] for c in CASES])
def test_composed_name(smiles, skeleton, expected):
    name = _compose(smiles, skeleton)
    assert name == expected
    # read again by a fresh OPSIN call, which shares nothing with the composer's own gate
    assert name_is_rt_exact(name, smiles)


@pytest.mark.unit
def test_no_name_for_an_isotope_induced_stereocentre():
    """The deuterium of CH2D beside a CH makes that CH a stereocentre; the name has no descriptor
    for it, so the composed name reads back without its configuration and is not returned
    is the decorator's, for a single descriptor)."""
    smiles = "[2H]C[C@@H](C)OS(=O)(=O)C1=CC=C(C=C1)Br"
    assert _compose(smiles, "propan-2-yl 4-bromobenzene-1-sulfonate") is None


@pytest.mark.unit
@pytest.mark.parametrize("smiles,skeleton", [
    ("[2H]Oc1ccc(O)cc1[2H]", "benzene-1,4-diol"),
    ("[2H]Oc1ccc(O)c([2H])c1", "benzene-1,4-diol"),
    ("[2H]OC(=O)CC[13CH2]C(=O)O", "pentanedioic acid"),
])
def test_bare_heteroatom_letter_is_never_cited_where_the_parent_holds_two(smiles, skeleton):
    """ (the Blue Book, under ' Omission of locants'): "if isotopic modification
    requires a locant to specify its position, then all locants must be specified and none are
    omitted"; (:44202): "Locants are not omitted when there is a possibility of isomers".
    OPSIN reads a bare 'O' as the first oxygen, so '(O,6-2H2)benzene-1,4-diol' reads back, but its text
    licenses the other oxygen too; the oxygen is cited by its number, 'O1'."""
    name = _compose(smiles, skeleton)
    assert name is not None and name_is_rt_exact(name, smiles)
    assert "(O," not in name and "(O-" not in name and ",O," not in name and ",O-" not in name, name
    assert re.search(r"\(.*\bO\d+[,-]", name), name


@pytest.mark.unit
def test_no_name_when_the_skeleton_is_not_the_molecule():
    """The frame is the parse of the skeleton; a skeleton that reads as another molecule gives none."""
    assert _compose("[2H]C([2H])([2H])CCO", "2-methylpropan-1-ol") is None


# -- no JVM: the pieces of the composer ------------------------------------------------------------


def test_heavy_view_binds_a_labelled_hydrogen_to_its_carbon():
    mol = Chem.MolFromSmiles("[2H]C([2H])([2H])C[13CH3]")
    view = S._heavy_view(mol)
    heavy, labels = view
    assert heavy.GetNumAtoms() == 3
    assert sorted(labels) == [((2, "H"), 0), ((2, "H"), 0), ((2, "H"), 0), ((13, "C"), 2)]


def test_model_records_the_atoms_of_each_position():
    model = S._Model(Chem.MolFromSmiles("[2H]C([2H])([2H])C[13CH3]"))
    assert model.need == {0: (((2, "H"), 3),), 2: (((13, "C"), 1),)}
    assert model.maxpos[(0, (2, "H"))] == 3
    assert model.maxpos[(2, (13, "C"))] == 1


@pytest.mark.parametrize("skeleton,texts", [
    ("2-(5-cyclohexyl-2,4-dimethylphenyl)-1,4-dimethylpyridin-1-ium",
     [("2,4-dimethyl", ("2-methyl-4-methyl",)), ("1,4-dimethyl", ("1-methyl-4-methyl",))]),
    ("3,4,5-tris(acetyloxy)-2-[(acetyloxy)methyl]",
     [("3,4,5-tris(acetyloxy)", ("3-(acetyloxy)-4-(acetyloxy)-5-(acetyloxy)",))]),
    ("N,N-di([1,1'-biphenyl]-4-yl)-5-x",
     [("N,N-di([1,1'-biphenyl]-4-yl)", ("N-([1,1'-biphenyl]-4-yl)-N-([1,1'-biphenyl]-4-yl)",))]),
    # (heading '', the Blue Book): "The second and further substituents are each
    # enclosed with parentheses even for simple substituents"; a copy with a descriptor includes a locant and is
    # enclosed too, so the enclosed spelling is the first and the bare one the second
    ("2-(dimethylamino)ethyl", [("dimethyl", ("(methyl)(methyl)", "methylmethyl"))]),
    ("4,11,18-tri-tert-butyl-22-methyl",
     [("4,11,18-tri-tert-butyl", ("4-tert-butyl-11-tert-butyl-18-tert-butyl",))]),
    ("2,3-dihydro-1H-indene", []),            # hydro is not a prefix of a group
    ("1,4-diazabicyclo[2.2.2]octane", []),    # nor is the 'aza' of a skeletal replacement
])
def test_multiplied_groups_cited_apart(skeleton, texts):
    spans = S._group_spans(skeleton)
    assert [(skeleton[start:end], got) for start, end, got in spans] == texts


def test_one_descriptor_per_slot_in_one_place_cited_together():
    """ /: the nuclides of one place are one descriptor, alphabetical (the carbon
    before the hydrogen), and: every locant is cited."""
    mol = Chem.MolFromSmiles("[2H][13C]([13CH3])([13CH3])[13C](=O)[13C](=O)O")
    model = S._Model(mol)
    # the heavy atoms in the order of the SMILES: C(D), CH3, CH3, C, C; the name numbers the chain
    # C-1 (carboxyl).. C-4, with the carbon that holds the hydrogen nuclide at C-3
    c3, _methyl_a, methyl_b, c2, c1 = sorted(model.need)
    pieces = [("3", (c3,)), ("4", (methyl_b,)), ("2", (c2,)), ("1", (c1,))]
    text = S._descriptor(model, pieces, False, False)
    assert text == "(1,2,3,4-13C4,3-2H)"
    slot = S._Slot(0, 14, None, 0)
    assert slot.edits(text) == [(14, text)]


# -- the decorator ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def best_effort():
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.unit
@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_decorator_reaches_the_composer_and_labels_the_name_below_the_pin(best_effort):
    smiles = "[2H][13C]([13CH3])([13CH3])[13C](=O)[13C](=O)O"
    got = best_effort.name_tiered(smiles)
    assert got["name"] == "3-(13C)methyl-2-oxo(1,2,3,4-13C4,3-2H)butanoic acid"
    assert got["tier"] == "systematic_verified"                              # below the PIN
    from orthonym import Orthonym
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] == "abstain"   # the default tier declines it


@pytest.mark.unit
@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # NUMBERING (i) (the Blue Book): the lowest locants to the modified atoms as a set, whichever way
    # the ring is read onto the input
    ("[2H]Oc1ccc(O)cc1[2H]", "(O1,2-2H2)benzene-1,4-diol"),
    ("[2H]Oc1ccc(O)c([2H])c1", "(O1,3-2H2)benzene-1,4-diol"),
    ("[2H]OC(=O)CC[13CH2]C(=O)O", "(4-13C,O1-2H)pentanedioic acid"),
])
def test_numbered_oxygen_locant_through_the_decorator(best_effort, smiles, expected):
    got = best_effort.name_tiered(smiles)
    assert (got["name"], got["tier"]) == (expected, "systematic_verified")
    assert name_is_rt_exact(expected, smiles)
    # the bare letter reads back too (OPSIN takes the first oxygen) and is a different name for the same
    # structure; it is the form the composer must not cite
    assert not got["name"].startswith("(O,") and "(4-13C,O-" not in got["name"]


@pytest.mark.unit
@pytest.mark.parametrize("flags", [
    {},                                                                  # the default tier
    {"general_fallback": True},                                          # valid
    {"general_fallback": True, "allow_aromatic_general": True},          # complete
])
def test_composer_runs_at_the_best_effort_tier_only(monkeypatch, flags):
    """Its names are below the PIN: every tier but best-effort declines them, so the decorator does not
    ask for one (``general_fallback_unverified`` is the best-effort discriminator)."""
    def boom(*args, **kwargs):
        raise AssertionError("the composer ran below the best-effort tier")
    monkeypatch.setattr(S, "compose_last_resort", boom)
    from orthonym import Orthonym
    eng = Orthonym(style="pin", **flags)
    assert I.decorate_isotopic_name("[2H][13C]([13CH3])([13CH3])[13C](=O)[13C](=O)O", eng.style, eng) is None


@pytest.mark.unit
def test_composer_is_not_asked_when_the_decorator_names_the_molecule(monkeypatch):
    """Every name the decorator builds is the name it built before: the composer is the last resort."""
    def boom(*args, **kwargs):
        raise AssertionError("the composer ran for a molecule the decorator names")
    monkeypatch.setattr(S, "compose_last_resort", boom)
    from orthonym import Orthonym
    eng = Orthonym(style="pin", general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    assert I.decorate_isotopic_name("[13CH3]c1ccccc1[13CH3]", eng.style, eng) is not None
    assert I.decorate_isotopic_name("[2H]C1CCC([2H])CC1", eng.style, eng) == "(1,4-2H2)cyclohexane"


# -- the budget of the skeleton namings ----------------------------------------------------------------


class _Trips:
    """A naming that runs out of the hang budget the first ``trips`` times it is asked."""

    def __init__(self, trips):
        self.trips, self.calls = trips, 0

    def name(self, smiles):
        from orthonym.assembly.fragment_naming import PerfBudgetExceeded
        self.calls += 1
        if self.calls <= self.trips:
            raise PerfBudgetExceeded()
        return "name of " + smiles


def test_skeleton_naming_that_runs_out_is_begun_again_under_a_fresh_budget(monkeypatch):
    rearmed = []
    monkeypatch.setattr("orthonym.assembly.fragment_naming.rearm_hang_budgets", lambda: rearmed.append(1))
    token = I._SKELETON_NAMINGS.set([0])
    try:
        naming = _Trips(1)
        assert I._name_skeleton(naming, "CCO") == "name of CCO"
        assert (naming.calls, len(rearmed)) == (2, 1)
        # a naming that stays within the budget is not touched
        fine = _Trips(0)
        assert I._name_skeleton(fine, "CCO") == "name of CCO"
        assert (fine.calls, len(rearmed)) == (1, 1)
    finally:
        I._SKELETON_NAMINGS.reset(token)


def test_skeleton_namings_are_begun_again_at_most_five_times_a_molecule(monkeypatch):
    from orthonym.assembly.fragment_naming import PerfBudgetExceeded
    monkeypatch.setattr("orthonym.assembly.fragment_naming.rearm_hang_budgets", lambda: None)
    token = I._SKELETON_NAMINGS.set([0])
    try:
        for _ in range(5):
            assert I._name_skeleton(_Trips(1), "CCO") == "name of CCO"
        with pytest.raises(PerfBudgetExceeded):
            I._name_skeleton(_Trips(1), "CCO")
    finally:
        I._SKELETON_NAMINGS.reset(token)


def test_a_naming_outside_a_decorator_call_is_not_begun_again():
    from orthonym.assembly.fragment_naming import PerfBudgetExceeded
    with pytest.raises(PerfBudgetExceeded):
        I._name_skeleton(_Trips(1), "CCO")
