"""The fullerene reader (``validation.fullerene_read_back``): a name is read as the structure it states.

OPSIN reads no fullerene name, so the reader is the certificate of every fullerene name the engine ships, and a
certificate that cannot refuse is worth nothing: every group below pairs names with structures built by hand from
the template (``tests/support/fullerene_adducts.py``), the book's own names first, then mutants of them, each of
which must be refused (a mutation test of the certificate).

* Systematic names (the Blue Book): '(C60-Ih)[5,6]fullerene (PIN)'.
* Fullerenes and modified fullerenes ortho fused to organic rings or ring systems (:15464-15480):
  '3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene (PIN)', '[1,3]dioxolo[4',5':1,9](C60-Ih)[5,6]fullerene (PIN)',
  'tris(oxireno)[2',3':1,9;2'',3'':2,12;2''',3''':7,8](C60-Ih)[5,6]fullerene (PIN)'.
* (:25834) '1-(trifluoromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene'; (:26870) '(C60-Ih)
  [5,6]fulleren-1(9H)-ol'.
"""
import pytest

from orthonym.validation import fullerene_read_back as reader
from orthonym.validation.fullerene_read_back import fullerene_name_shape, fullerene_read_back
from tests.support.fullerene_adducts import C60_SMILES, adduct
from tests.support.fullerene_adducts import adduct
from tests.support.jars import jar_or_skip

pytestmark = pytest.mark.unit

CORE = "(C60-Ih)[5,6]fullerene"


def _holds(name, smiles):
    return fullerene_read_back(name, smiles)


# --- the book's names, each against the structure it draws --------------------------------------------------

BOOK_NAMES = [
    # (heading of the rule, name, fragment on 1,9)
    ("P-27.2.1 Systematic names", "(C60-Ih)[5,6]fullerene", None),
    ("P-27.6.1 cyclopropa", "3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene", "[1*]C[2*]"),
    ("P-27.6.1 oxireno (tris(oxireno) component)", "oxireno[2',3':1,9](C60-Ih)[5,6]fullerene", "[1*]O[2*]"),
    ("P-61.3.1", "1-(trifluoromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene", "[1*]C(F)(F)F.[2*]"),
    ("P-63.1.2", "(C60-Ih)[5,6]fulleren-1(9H)-ol", "[1*]O.[2*]"),
]


@pytest.mark.parametrize("heading,name,fragment", BOOK_NAMES, ids=[b[1] for b in BOOK_NAMES])
def test_the_books_names_read_back_to_their_structures(heading, name, fragment):
    jar_or_skip()
    smiles = C60_SMILES if fragment is None else adduct(fragment)
    assert _holds(name, smiles), heading


def test_dioxolo_needs_the_indicated_hydrogen_of_its_methylene():
    # prints '[1,3]dioxolo[4',5':1,9](C60-Ih)[5,6]fullerene (PIN)' with no indicated hydrogen, although the
    # CH2 between the two oxygen atoms is saturated: "indicated hydrogen is cited as needed". The reader proves the
    # structure of a name that states it; it does not guess a missing '2'H', so the printed form is not read and the
    # engine's own name, '2'H-[1,3]dioxolo[4',5':1,9]...', is
    jar_or_skip()
    smiles = adduct("[1*]OCO[2*]")
    assert _holds("2'H-[1,3]dioxolo[4',5':1,9]" + CORE, smiles)
    assert not _holds("[1,3]dioxolo[4',5':1,9]" + CORE, smiles)


def test_the_two_atoms_of_the_fusion_bond_are_interchangeable():
    # a symmetry of the cage swaps the cage atoms 1 and 9: the name that cites the group on 9 is of the structure with
    # the group on 1 (it is not the name with the lowest locants, which the reader does not judge)
    jar_or_skip()
    assert _holds("9-(trifluoromethyl)-1,9-dihydro" + CORE, adduct("[1*]C(F)(F)F.[2*]"))


def test_a_multiply_fused_name_is_not_read():
    # the three-addend name of is outside the one-addend class: unread, so never shipped
    jar_or_skip()
    name = "tris(oxireno)[2',3':1,9;2'',3'':2,12;2''',3''':7,8]" + CORE
    assert not _holds(name, adduct("[1*]O[2*]"))


def test_a_substituent_unit_is_read_through_the_outer_name():
    # the unit is a substituent in the innermost enclosing marks; the outer name, with the unit replaced by 'silyl',
    # is read by OPSIN and the atom next to the marker takes the free valence
    jar_or_skip()
    smiles = adduct("[1*]C(CCCC(=O)OC)(c1ccccc1)[2*]")
    name = "methyl 4-[3'-phenyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fulleren-3'-yl]butanoate"
    assert _holds(name, smiles)
    assert _holds(name.replace("[3'", "(3'", 1).replace("-yl]", "-yl)"), smiles)       # the enclosing mark is no matter


def test_an_ester_of_a_carboxylic_acid_on_the_ring_is_read():
    jar_or_skip()
    smiles = adduct("[1*]C(C(=O)OCC)(C(=O)OCC)[2*]")
    name = "ethyl 3'-(ethoxycarbonyl)-3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-3'-carboxylate"
    assert _holds(name, smiles)
    assert not _holds(name.replace("ethyl 3'", "methyl 3'"), smiles)                # the ester word
    assert not _holds(name.replace("(ethoxycarbonyl)", "(methoxycarbonyl)"), smiles)  # the ester prefix


EXTENDED = [
    # (what the name has, the name, the fragment)
    ("the alcohol part of an ester, a word of its own", "9-hydroxy(C60-Ih)[5,6]fulleren-1(9H)-yl acetate",
     "[1*]OC(C)=O.[2*]O"),
    ("a unit inside the marks of a prefix that follows other prefixes",
     "ethyl difluoro[(C60-Ih)[5,6]fulleren-1(9H)-yl]acetate", "[1*]C(F)(F)C(=O)OCC.[2*]"),
    ("the acyl group of an ester", "methyl 4-({3'-phenyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-3'-carbonyl}oxy)"
                                   "butanoate", "[1*]C(c1ccccc1)(C(=O)OCCCC(=O)OC)[2*]"),
    ("two substituents on the nitrogen atom of an amide",
     "N,N-diethyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-3'-carboxamide", "[1*]C(C(=O)N(CC)CC)[2*]"),
    ("one substituent on the nitrogen atom of an amide, and one on the ring",
     "N-cyclohexyl-3'-methyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-3'-carboxamide",
     "[1*]C(C)(C(=O)NC1CCCCC1)[2*]"),
    ("an ester word multiplied like its carboxylate groups",
     "dimethyl 3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-3',3'-dicarboxylate", "[1*]C(C(=O)OC)(C(=O)OC)[2*]"),
    ("a double bond to the ring (sulfanylidene)",
     "5'-sulfanylidene-1',5'-dihydro-2'H-pyrrolo[3',4':1,9](C60-Ih)[5,6]fullerene", "[1*]CNC(=S)[2*]"),
]


@pytest.mark.parametrize("what,name,fragment", EXTENDED, ids=[e[0] for e in EXTENDED])
def test_the_forms_the_writer_adds_to_the_unit_are_read(what, name, fragment):
    jar_or_skip()
    assert _holds(name, adduct(fragment)), what


EXTENDED_MUTANTS = [
    ("a substituent on the wrong nitrogen locant", "N-cyclohexyl-3'-methyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-"
                                                   "3'-carboxamide", "[1*]C(C)(C(=O)N(C)C1CCCCC1)[2*]"),
    ("one ester word for two carboxylate groups",
     "methyl 3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene-3',3'-dicarboxylate", "[1*]C(C(=O)OC)(C(=O)OC)[2*]"),
    ("the wrong alcohol part", "9-hydroxy(C60-Ih)[5,6]fulleren-1(9H)-yl propanoate", "[1*]OC(C)=O.[2*]O"),
    ("an oxo for a thioxo", "5'-oxo-1',5'-dihydro-2'H-pyrrolo[3',4':1,9](C60-Ih)[5,6]fullerene", "[1*]CNC(=S)[2*]"),
]


@pytest.mark.parametrize("what,name,fragment", EXTENDED_MUTANTS, ids=[e[0] for e in EXTENDED_MUTANTS])
def test_a_mutant_of_the_extended_forms_is_refused(what, name, fragment):
    jar_or_skip()
    assert not _holds(name, adduct(fragment)), what


# --- mutants: every one names another structure or states too little, and none is read back ------------------

MUTANTS = [
    # (what changed, name, fragment of the structure it is held against)
    ("1,2 for 1,9", "3'H-cyclopropa[1,2](C60-Ih)[5,6]fullerene", "[1*]C[2*]"),
    ("another ring size", "3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene", "[1*]CC[2*]"),
    ("the ring the structure does not have", "oxireno[2',3':1,9](C60-Ih)[5,6]fullerene", "[1*]C[2*]"),
    ("wrong prefix", "1-(trichloromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene", "[1*]C(F)(F)F.[2*]"),
    ("prefix on an atom the hydro prefix does not saturate", "2-(trifluoromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene",
     "[1*]C(F)(F)F.[2*]"),
    ("dropped hydro", "1-(trifluoromethyl)(C60-Ih)[5,6]fullerene", "[1*]C(F)(F)F.[2*]"),
    ("dropped hydro (bare dihydro)", "(C60-Ih)[5,6]fullerene", "[1*].[2*]"),
    ("hydro on the wrong pair", "1,2-dihydro(C60-Ih)[5,6]fullerene", "[1*].[2*]"),
    ("added hydrogen on the wrong atom", "(C60-Ih)[5,6]fulleren-1(2H)-ol", "[1*]O.[2*]"),
    ("a suffix for another group", "(C60-Ih)[5,6]fulleren-1(9H)-thiol", "[1*]O.[2*]"),
    ("a multiplier that does not match", "(C60-Ih)[5,6]fullerene-1-diol", "[1*]O.[2*]O"),
    ("wrong prime on the attached locants", "1'-methyl-1',5'-dihydro-2'H-pyrrolo[2',3':1,9](C60-Ih)[5,6]fullerene",
     "[1*]CN(C)C[2*]"),
    ("methyl on the other atom of the ring", "2'-methyl-1',5'-dihydro-2'H-pyrrolo[3',4':1,9](C60-Ih)[5,6]fullerene",
     "[1*]CN(C)C[2*]"),
    ("indicated hydrogen dropped", "1'-methyl-1',5'-dihydropyrrolo[3',4':1,9](C60-Ih)[5,6]fullerene",
     "[1*]CN(C)C[2*]"),
    ("another core", "3'H-cyclopropa[1,9](C70-D5h(6))[5,6]fullerene", "[1*]C[2*]"),
    ("no 'e' before a consonant suffix", "(C60-Ih)[5,6]fullerene-1(9H)-ol", "[1*]O.[2*]"),
]


@pytest.mark.parametrize("what,name,fragment", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_a_mutant_is_not_read_back(what, name, fragment):
    jar_or_skip()
    assert not _holds(name, adduct(fragment)), what


def test_a_wrong_ester_word_is_refused():
    jar_or_skip()
    smiles = adduct("[1*]C(CCCC(=O)OC)(c1ccccc1)[2*]")
    wrong = "ethyl 4-[3'-phenyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fulleren-3'-yl]butanoate"
    assert not _holds(wrong, smiles)
    assert not _holds("methyl 3-[3'-phenyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fulleren-3'-yl]propanoate", smiles)


def test_a_wrong_substituent_prefix_is_refused():
    jar_or_skip()
    smiles = adduct("[1*]C(CCCC(=O)OC)(c1ccccc1)[2*]")
    assert not _holds("methyl 4-[3'-(4-methylphenyl)-3'H-cyclopropa[1,9](C60-Ih)[5,6]fulleren-3'-yl]butanoate", smiles)
    assert not _holds("methyl 4-[3'-cyclohexyl-3'H-cyclopropa[1,9](C60-Ih)[5,6]fulleren-3'-yl]butanoate", smiles)


def test_the_fixed_hydrogen_layer_tells_the_nh_tautomers_apart():
    # the standard InChI does not see which nitrogen of a triazole carries the hydrogen atom; the reader compares the
    # fixed-hydrogen InChI as well, so the tautomer of the structure is the name's
    jar_or_skip()
    smiles = adduct("[1*]N=NN[2*]")
    assert _holds("1'H-[1,2,3]triazolo[4',5':1,9]" + CORE, smiles)
    assert not _holds("2'H-[1,2,3]triazolo[4',5':1,9]" + CORE, smiles)
    # a structure whose key differs in the standard layer is refused as before
    assert not _holds("1'H-[1,2,3]triazolo[4',5':1,9]" + CORE, C60_SMILES)


def test_names_without_the_core_token_or_with_two_are_not_read():
    jar_or_skip()
    assert not fullerene_name_shape("ethanol") and not _holds("ethanol", adduct("[1*]C[2*]"))
    assert not _holds(None, adduct("[1*]C[2*]")) and not _holds("3'H-cyclopropa[1,9]" + CORE, None)
    assert not _holds(CORE + " " + CORE, C60_SMILES)
    assert not _holds("3'H-cyclopropa[1,9]" + CORE, "")


def test_a_structure_of_another_class_is_never_equal():
    jar_or_skip()
    assert not _holds(CORE, adduct("[1*]C[2*]"))
    assert not _holds("3'H-cyclopropa[1,9]" + CORE, C60_SMILES)
    assert not _holds(CORE, "CCO")


# --- the reader's own tables ---------------------------------------------------------------------------------

#: the ring each attached prefix comes from
_RING_OF = {
    "oxireno": "oxirene", "thiireno": "thiirene", "azirino": "azirine", "silireno": "silirene",
    "furo": "furan", "thieno": "thiophene", "pyrrolo": "pyrrole", "silolo": "silole",
    "[1,3]dioxolo": "1,3-dioxole", "[1,3]disilolo": "1,3-disilole", "[1,2,3]trioxolo": "1,2,3-trioxole",
    "[1,3,2]dioxaborolo": "1,3,2-dioxaborole", "[1,2]oxazolo": "1,2-oxazole", "[1,3]oxazolo": "1,3-oxazole",
    "[1,2]thiazolo": "1,2-thiazole", "[1,3]thiazolo": "1,3-thiazole", "imidazo": "imidazole",
    "pyrazolo": "pyrazole", "[1,2,3]triazolo": "1,2,3-triazole", "[1,2,4]triazolo": "1,2,4-triazole",
    "[1,3]dithiolo": "1,3-dithiole", "pyrido": "pyridine", "pyridazino": "pyridazine", "pyrazino": "pyrazine",
    "pyrimido": "pyrimidine", "[1,4]dioxino": "1,4-dioxine", "[1,2]oxazino": "1,2-oxazine",
    "[1,3]oxazino": "1,3-oxazine", "cyclopropa": "cyclopropane", "cyclobuta": "cyclobutane",
    "cyclopenta": "cyclopentane", "benzo": "benzene", "cyclohepta": "cycloheptane",
    "azeto": "azete", "oxeto": "oxete", "thieto": "thiete", "pyrano": "2H-pyran", "thiopyrano": "2H-thiopyran",
    "azepino": "1H-azepine", "oxepino": "oxepine", "thiepino": "thiepine", "tetrazolo": "1H-tetrazole",
    "[1,3]oxathiolo": "1,3-oxathiole", "[1,2]oxathiolo": "1,2-oxathiole", "[1,2]dithiolo": "1,2-dithiole",
    "[1,3]dioxino": "1,3-dioxine", "[1,2,4]oxadiazolo": "1,2,4-oxadiazole", "[1,3,4]oxadiazolo": "1,3,4-oxadiazole",
    "[1,2,5]oxadiazolo": "1,2,5-oxadiazole", "[1,2,3]oxadiazolo": "1,2,3-oxadiazole",
    "[1,2,4]thiadiazolo": "1,2,4-thiadiazole", "[1,3,4]thiadiazolo": "1,3,4-thiadiazole",
}


def test_every_component_has_a_ring():
    assert set(_RING_OF) == set(reader._COMPONENTS)


@pytest.mark.parametrize("prefix", sorted(_RING_OF))
def test_component_table_agrees_with_opsins_own_locants(prefix):
    # (the Blue Book): the locant 1 on the senior heteroatom, then the lowest heteroatom locants:
    # OPSIN numbers the ring name, and the table lists the elements at its locants 1', 2',...
    jar_or_skip()
    import re

    from orthonym.validation.opsin_roundtrip import _extended_smiles
    ext = _extended_smiles(_RING_OF[prefix])
    assert ext, _RING_OF[prefix]
    smiles, values = re.match(r"^(\S+)\s+\|\$_AV:(.*)\$\|$", ext).groups()
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    locants = values.split(";")
    elements = {int(loc): mol.GetAtomWithIdx(i).GetSymbol() for i, loc in enumerate(locants) if loc.strip()}
    table = re.findall(r"[A-Z][a-z]?", reader._COMPONENTS[prefix])
    assert [elements[i] for i in range(1, len(table) + 1)] == table


def test_the_atom_matcher_finds_a_perfect_matching_of_the_cage():
    # the matching of the double bonds: C60 has a perfect one, and so has the cage with the atoms of a [6,6] bond
    # taken out; a cage with the two ends of a [5,6] bond and one more atom out has none (odd)
    from orthonym.data.fullerene_c60 import C60_NEIGHBOURS
    atoms = sorted(C60_NEIGHBOURS)
    index = {a: i for i, a in enumerate(atoms)}
    adj = [[index[n] for n in C60_NEIGHBOURS[a]] for a in atoms]
    assert all(m != -1 for m in reader._max_matching(60, adj))
    keep = [a for a in atoms if a not in (1, 9)]
    pos = {a: i for i, a in enumerate(keep)}
    adj2 = [[pos[n] for n in C60_NEIGHBOURS[a] if n in pos] for a in keep]
    assert all(m != -1 for m in reader._max_matching(58, adj2))
    keep = [a for a in atoms if a not in (1, 9, 2, 12)]      # 1,9 and 2,12 are [6,6] bonds: still matchable
    pos = {a: i for i, a in enumerate(keep)}
    adj3 = [[pos[n] for n in C60_NEIGHBOURS[a] if n in pos] for a in keep]
    assert all(m != -1 for m in reader._max_matching(len(keep), adj3))


# --- a part OPSIN reads with a charge, a radical, an isotope or a stereo mark is refused ---------------------
# The skeleton is rebuilt from element symbols and bond orders, so such a mark would be dropped and the name of
# another species would read as the (neutral, unlabelled, stereo-free) input. Review 2026-11-01: each of these
# passed before ``_refuse_marked``.

MARKED_PARTS = [
    ("[1*]c1ccncc1.[2*]", "4-[(C60-Ih)[5,6]fulleren-1(9H)-yl]pyridin-1-ium"),
    ("[1*]C.[2*]", "[(C60-Ih)[5,6]fulleren-1(9H)-yl]methyl"),
    ("[1*]C(C)CC.[2*]", "(2R)-2-[(C60-Ih)[5,6]fulleren-1(9H)-yl]butane"),
    ("[1*]C(C)CC.[2*]", "1-[(2R)-butan-2-yl]-1,9-dihydro(C60-Ih)[5,6]fullerene"),
    ("[1*]C.[2*]", "[(C60-Ih)[5,6]fulleren-1(9H)-yl](2H3)methane"),
    ("[1*]C.[2*]", "1-(2H3)methyl-1,9-dihydro(C60-Ih)[5,6]fullerene"),
]


@pytest.mark.parametrize("fragment, name", MARKED_PARTS)
def test_a_marked_part_is_refused(fragment, name):
    jar_or_skip()
    assert not _holds(name, adduct(fragment))


@pytest.mark.parametrize("fragment, name", [
    ("[1*]c1ccncc1.[2*]", "4-[(C60-Ih)[5,6]fulleren-1(9H)-yl]pyridine"),
    ("[1*]C(C)CC.[2*]", "2-[(C60-Ih)[5,6]fulleren-1(9H)-yl]butane"),
])
def test_the_unmarked_part_still_reads(fragment, name):
    jar_or_skip()
    assert _holds(name, adduct(fragment))
