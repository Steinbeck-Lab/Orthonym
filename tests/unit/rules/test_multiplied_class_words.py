"""A multiplied unit that is an acid halide or pseudohalide multiplies its class word too.

Governing rules (``the Blue Book Blue Book``):

* "Acyl halides from suffix acids" (heading:31412, text:31414): the name of the acyl
  group is "followed by the name(s) of the specific class(es) as a separate word(s), in alphabetical
  order, each preceded by a multiplicative prefix, as needed"; examples ``benzene-1,4-dicarbonyl
  dichloride (PIN)`` (:31450) and ``ethane-1,2-disulfonyl dibromide (PIN)`` (:31452). Table 6.3
  (:31416) lists the classes.
* "Acyl pseudohalides from suffix acids" (heading:31458, text:31460): the same for the
  pseudohalides, ``oxalyl diisothiocyanate (PIN)`` (:31468).
* (:30608): the acyl group of an 'oic acid' is named by changing 'oic acid' to 'oyl', so
  the acyl group of the multiplied acid ``4,4'-oxydibenzoic acid`` is ``4,4'-oxydibenzoyl``.
* "Polyester names formed by multiplicative nomenclature" (:31794): the esters of a
  multiplied acid keep the multiplied acid component, ``dimethyl 3,3'-oxydibenzoate (PIN)`` (:31801).
* "Assemblies of identical structural units" (:5799) with ``4,4'-oxydi(benzene-1-sulfonic
  acid) (PIN)`` (:5827) and (e) (:7085): a multiplied functionalized parent hydride with a
  locant is enclosed, so its acyl group is ``di(benzene-1-sulfonyl)``.

The Blue Book gives no example of the multiplied acyl group of an acyl halide; the expected names are
the reading of and for it (ASSUMED, as the ester is,.

Before this class the producer split the unit 'benzoyl chloride' at its space and multiplied the first
word alone: ``4,4'-oxydibenzoyl chloride`` cites one halide for two, and was labelled a PIN.
"""
import re

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _tiered_namer
from orthonym.rules.multiplicative import _assemble_multiplicative_name
from orthonym.rules.multiplied_class_word_units import ACYL_CLASS_WORDS
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# (locant, bridge, unit, unit count, expected multiplicative name)
_ASSEMBLED = [
    (4, "oxy", "benzoyl chloride", 2, "4,4'-oxydibenzoyl dichloride"),
    (4, "methylene", "benzoyl chloride", 2, "4,4'-methylenedibenzoyl dichloride"),
    (4, "oxy", "benzoyl bromide", 2, "4,4'-oxydibenzoyl dibromide"),
    (4, "oxy", "benzoyl fluoride", 2, "4,4'-oxydibenzoyl difluoride"),
    (4, "oxy", "benzoyl cyanide", 2, "4,4'-oxydibenzoyl dicyanide"),
    (4, "oxy", "benzoyl azide", 2, "4,4'-oxydibenzoyl diazide"),
    (4, "oxy", "benzoyl isocyanate", 2, "4,4'-oxydibenzoyl diisocyanate"),
    (4, "oxy", "benzoyl isothiocyanate", 2, "4,4'-oxydibenzoyl diisothiocyanate"),
    (3, "oxy", "benzoyl chloride", 2, "3,3'-oxydibenzoyl dichloride"),
    # the unit of a multiplied suffix acid with a locant is enclosed (:5827), so its acyl group is
    (4, "oxy", "benzenesulfonyl chloride", 2, "4,4'-oxydi(benzene-1-sulfonyl) dichloride"),
    (4, "oxy", "benzenesulfonyl cyanide", 2, "4,4'-oxydi(benzene-1-sulfonyl) dicyanide"),
    (4, "oxy", "benzenesulfinyl chloride", 2, "4,4'-oxydi(benzene-1-sulfinyl) dichloride"),
    (4, "oxy", "cyclohexanecarbonyl chloride", 2, "4,4'-oxydi(cyclohexane-1-carbonyl) dichloride"),
    # three and four units: the class word takes the prefix of the count
    (4, "nitrilo", "benzoyl chloride", 3, "4,4',4''-nitrilotribenzoyl trichloride"),
    (4, "methanetetrayl", "benzoyl chloride", 4, "4,4',4'',4'''-methanetetrayltetrabenzoyl tetrachloride"),
    # a substituted unit takes 'bis' and enclosing marks, as 4,4'-oxybis(2-chlorobenzoic acid)
    (4, "oxy", "2-chlorobenzoyl chloride", 2, "4,4'-oxybis(2-chlorobenzoyl) dichloride"),
]


@pytest.mark.unit
@pytest.mark.parametrize("locant,bridge,unit,count,expected", _ASSEMBLED)
def test_the_class_word_is_multiplied_with_the_acyl_group(locant, bridge, unit, count, expected):
    """ (:31414) / (:31460): acyl group, then the class word 'each preceded by a
    multiplicative prefix'."""
    assert _assemble_multiplicative_name(locant, bridge, unit, count) == expected


@pytest.mark.unit
@pytest.mark.parametrize("unit", [
    "benzene-1,3-dicarbonyl dichloride",    # two acyl groups in one unit: no multiplied name in the book
    "butanedioyl bromide chloride",         # two class words
    "phenylphosphonic dichloride",          # an acid name with the class word, not an acyl group
])
def test_a_unit_with_more_than_one_acyl_group_or_class_word_is_declined(unit):
    """The Blue Book gives no multiplied name for these units, so the producer declines them rather than
    cite fewer class words than the molecule holds."""
    assert _assemble_multiplicative_name(4, "oxy", unit, 2) is None


@pytest.mark.unit
@pytest.mark.parametrize("unit,expected", [
    ("benzoic acid", "4,4'-oxydibenzoic acid"),
    ("benzenesulfonic acid", "4,4'-oxydi(benzene-1-sulfonic acid)"),   # BB:5827 verbatim, (PIN)
    ("benzamide", "4,4'-oxydibenzamide"),
    ("benzonitrile", "4,4'-oxydibenzonitrile"),
    ("benzaldehyde", "4,4'-oxydibenzaldehyde"),
    ("2-chlorobenzoic acid", "4,4'-oxybis(2-chlorobenzoic acid)"),     # BB verbatim, (PIN)
])
def test_units_that_are_not_class_word_units_are_unchanged(unit, expected):
    assert _assemble_multiplicative_name(4, "oxy", unit, 2) == expected


@pytest.mark.unit
def test_the_class_words_are_table_6_3():
    """Table 6.3 'Acyl halide and pseudohalide classes' (:31416)."""
    assert set(ACYL_CLASS_WORDS) == {
        "fluoride", "chloride", "bromide", "iodide", "azide", "cyanide", "isocyanide", "isocyanate",
        "isothiocyanate", "isoselenocyanate", "isotellurocyanate"}


# --- whole molecules, read back by OPSIN 2.9.0 ----------------------------------------------------------

_MOLECULES = [
    ("ClC(=O)c1ccc(Oc2ccc(C(Cl)=O)cc2)cc1", "4,4'-oxydibenzoyl dichloride"),
    ("ClS(=O)(=O)c1ccc(Oc2ccc(S(Cl)(=O)=O)cc2)cc1", "4,4'-oxydi(benzene-1-sulfonyl) dichloride"),
    ("N#CS(=O)(=O)c1ccc(Oc2ccc(S(=O)(=O)C#N)cc2)cc1", "4,4'-oxydi(benzene-1-sulfonyl) dicyanide"),
    ("ClC(=O)c1ccc(Cc2ccc(C(Cl)=O)cc2)cc1", "4,4'-methylenedibenzoyl dichloride"),
    ("N#CS(=O)c1ccc(Oc2ccc(S(=O)C#N)cc2)cc1", "4,4'-oxydi(benzene-1-sulfinyl) dicyanide"),
    ("ClC(=O)c1ccc(N(c2ccc(C(Cl)=O)cc2)c2ccc(C(Cl)=O)cc2)cc1", "4,4',4''-nitrilotribenzoyl trichloride"),
    ("[N-]=[N+]=NC(=O)c1ccc(Oc2ccc(C(=O)N=[N+]=[N-])cc2)cc1", "4,4'-oxydibenzoyl diazide"),
    ("ClC(=O)C1CCC(OC2CCC(C(Cl)=O)CC2)CC1", "4,4'-oxydi(cyclohexane-1-carbonyl) dichloride"),
]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", _MOLECULES)
def test_multiplied_acid_halides_cite_one_class_word_per_group(smiles, expected):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected
    assert row["tier"] == "pin_verified" and row["opsin"] == "verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [s for s, _ in _MOLECULES])
def test_the_class_word_count_is_the_number_of_acyl_groups(smiles):
    """The invariant behind the class: a name that ends in a class word of Table 6.3 carries the
    multiplicative prefix of the number of acyl halide or pseudohalide groups of the molecule."""
    groups = len(Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmarts(
        "[$([CX3](=O)),$([SX4](=O)=O),$([SX3]=O)]-[$([Cl,Br,F,I]),$(N=[N+]=[N-]),$(C#N)]")))
    name = Orthonym().name_tiered(smiles)["name"]
    last = name.rsplit(" ", 1)[-1]
    prefix = re.fullmatch(r"(di|tri|tetra|penta|hexa)?(?:%s)" % "|".join(ACYL_CLASS_WORDS), last)
    assert prefix is not None, name
    assert {None: 1, "di": 2, "tri": 3, "tetra": 4}[prefix.group(1)] == groups, name


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1ccc(Oc2ccc(C(O)=O)cc2)cc1", "4,4'-oxydibenzoic acid"),
    ("OS(=O)(=O)c1ccc(Oc2ccc(S(O)(=O)=O)cc2)cc1", "4,4'-oxydi(benzene-1-sulfonic acid)"),   # BB:5827
    ("COC(=O)c1ccc(Oc2ccc(C(=O)OC)cc2)cc1", "dimethyl 4,4'-oxydibenzoate"),   #:31794
    ("NC(=O)c1ccc(Oc2ccc(C(N)=O)cc2)cc1", "4,4'-oxydibenzamide"),
    # the names of the book itself are not multiplicative units
    ("ClC(=O)c1ccc(C(Cl)=O)cc1", "benzene-1,4-dicarbonyl dichloride"),            #:31450
    ("BrC(=O)CCC(Cl)=O", "butanedioyl bromide chloride"),                          #:31454
])
def test_the_names_of_the_acids_esters_and_the_book_examples_are_unchanged(smiles, expected):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_the_sulfinyl_cyanide_unit_is_multiplied_not_declined():
    """It was declined by a decorator of the producer while the class word was not multiplied; the
    multiplied form is the name of its sulfonyl sibling:34807 with:31460)."""
    from orthonym.rules.multiplicative import name_multiplicative
    smiles = "N#CS(=O)c1ccc(cc1)Oc1ccc(cc1)S(=O)C#N"
    assert name_multiplicative(Chem.MolFromSmiles(smiles)) == "4,4'-oxydi(benzene-1-sulfinyl) dicyanide"


# --- an acyclic arm: the acyl halide of a multiplied acid ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("ClC(=O)COCC(=O)Cl", "2,2'-oxydiacetyl dichloride"),
    ("ClC(=O)CCOCCC(=O)Cl", "3,3'-oxydipropanoyl dichloride"),
    ("BrC(=O)COCC(=O)Br", "2,2'-oxydiacetyl dibromide"),
    ("FC(=O)CCCOCCCC(F)=O", "4,4'-oxydibutanoyl difluoride"),
    ("ClC(=O)CSCC(=O)Cl", "2,2'-sulfanediyldiacetyl dichloride"),
    ("ClC(=O)CNCC(=O)Cl", "2,2'-azanediyldiacetyl dichloride"),
])
def test_the_acyclic_acid_halide_of_a_multiplied_acid_is_multiplied_like_the_ring_one(smiles, expected):
    """ "Acyl halides from suffix acids" (:31412): the acyl group of the acid, then the class word
    "each preceded by a multiplicative prefix, as needed"; the acid is the multiplicative PIN '2,2'-oxydiacetic
    acid',:5848), so its acyl group is '2,2'-oxydiacetyl',:30608), as the ring
    acid '4,4'-oxydibenzoic acid' gives '4,4'-oxydibenzoyl dichloride'. Before, the arm was an acid or an
    alcohol only, and the halide was named by the substitutive '(2-chloro-2-oxoethoxy)acetyl chloride'."""
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "ClC(=O)COCC(=O)Br",        # two halides: the arms are not identical, 'bromide chloride' is not built
    "ClC(=O)COC",               # one acyl halide: the substitutive name
    "ClC(=O)COCC(=O)O",         # an acid and its halide
])
def test_a_unit_that_is_not_two_identical_acyl_halides_keeps_its_name(smiles):
    name = Orthonym().name_tiered(smiles)["name"]
    assert "oxydi" not in name, name
