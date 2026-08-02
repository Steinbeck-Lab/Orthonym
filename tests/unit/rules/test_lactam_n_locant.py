"""A lactam ring nitrogen is numbered, so its substituent takes an ARABIC NUMERAL.

THE RULE. A lactam's preferred IUPAC name is a RING name, so every ring atom --
the nitrogen included -- carries an arabic ring locant, and a substituent on that
nitrogen is cited with that numeral. Italic '*N*' is for a nitrogen that receives
no numeral.

  * P-66.1.5 "Lactams, lactims, sultams, and sultims" / P-66.1.5.1 "Lactams and
    lactims" (`BlueBookV2.md:33222`, `:33224`): "Lactams are named in two ways:
    (1) as heterocyclic pseudoketones; (2) by substituting 'lactam' for the 'ic
    acid' ending ...".  The decisive sentence is the last one (`:33229`):
    "**Method (1) generates preferred IUPAC names.**"  So the PIN parent is a
    heterocycle, carrying ring numbering -- not an amide parent.

  * The same rule's own examples number the lactam nitrogen '1':
    `:33232` `pyrrolidin-2-one (PIN)  butano-4-lactam` and
    `:33236` `1-azacyclotridecan-2-one (PIN)  dodecano-12-lactam`
    -- the locant '1' in `1-aza...` IS the lactam nitrogen.

  * P-66.1.3 "'Hidden' amides" (`:33125`) settles the N/C crux head-on: "An
    *N*-acyl group attached to a nitrogen atom of a heterocyclic system has been
    called a 'hidden amide' ... The traditional way to name such compounds by
    using acyl groups as substituents on the nitrogen atom of the heterocyclic
    system is allowed **but only in general nomenclature**. Such compounds are now
    considered as pseudoketones (see P-64.3) and preferred IUPAC names are
    constructed accordingly."  Example `:33129`
    `1-(piperidin-1-yl)ethan-1-one (PIN)  1-acetylpiperidine` -- the ring nitrogen
    is `piperidin-1-yl`, a NUMERAL, and the italic-N amide reading is explicitly
    demoted to general nomenclature.

  * A ring AMIDE nitrogen is cited with a numeral in a `(PIN)`-tagged name:
    `:40645` `2,5-dioxopyrrolidin-1-yl (PIN)  succinimidyl` (and `:33865` as a
    preferred prefix). Position 1 is the imide nitrogen; the Blue Book's own
    diagram at `:40643` numbers it '1'.

  * P-14.3.3 "Citation of locants" (`:2869`) is deny-by-default, so once the N
    locant is the numeral '1' the whole set is cited: `1,5-dimethyl...`.

  * P-16.3.3 "The basic numerical prefixes 'di', 'tri', 'tetra', etc. are used to
    indicate a multiplicity of:" (`:7038`), clause (b) (`:7067`): multiplicity is
    a property of the substituent NAME, not of which ring atom carries it. So a
    methyl on the ring N and a methyl on a ring C are ONE group of two.

CONTRAST -- these are NOT lactams and italic '*N*' is correct for them, because
their nitrogen is not a ring atom and receives no numeral: `N-methylacetamide`,
`methyl N-methylcarbamate`, `N-cyclohexylthiourea`. P-66.1.2 "Secondary and
tertiary amides" (`:33093`) prints `*N*-acetylbenzamide (PIN)` etc. Those travel a
different producer entirely (measured: `name_monocyclic_lactam` records 0 calls
for all three), so they are covered here only as a boundary statement.
"""

import pytest
from rdkit import Chem

from orthonym.rules.lactams import name_monocyclic_lactam


def _name(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable SMILES {smiles!r}"
    return name_monocyclic_lactam(mol)


# --------------------------------------------------------------------------- #
# 1. The ring nitrogen takes a NUMERAL, never italic 'N'.                      #
#    Ring sizes 4 / 5 / 6 / 7 = beta / gamma / delta / epsilon lactam.         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CN1CCC1=O", "1-methylazetidin-2-one"),        # beta  (4)
        ("CN1CCCC1=O", "1-methylpyrrolidin-2-one"),     # gamma (5)
        ("CN1CCCCC1=O", "1-methylpiperidin-2-one"),     # delta (6)
        ("CN1CCCCCC1=O", "1-methylazepan-2-one"),       # epsilon (7)
    ],
)
def test_ring_nitrogen_locant_is_a_numeral_across_ring_sizes(smiles, expected):
    assert _name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCN1CCCC1=O", "1-ethylpyrrolidin-2-one"),
        ("CCCN1CCCC1=O", "1-propylpyrrolidin-2-one"),
        # P-16.3.5 enclosing marks: a substituent name carrying its own locant is
        # parenthesised once it is cited with a ring locant.
        ("CC(C)N1CCCC1=O", "1-(propan-2-yl)pyrrolidin-2-one"),
        ("c1ccccc1N1CCCC1=O", "1-phenylpyrrolidin-2-one"),
    ],
)
def test_ring_nitrogen_locant_is_a_numeral_across_substituents(smiles, expected):
    assert _name(smiles) == expected


@pytest.mark.parametrize(
    "smiles",
    [
        "CN1CCC1=O", "CN1CCCC1=O", "CN1CCCCC1=O", "CN1CCCCCC1=O",
        "CCN1CCCC1=O", "CC(C)N1CCCC1=O", "c1ccccc1N1CCCC1=O",
        "CN1C(C)CCC1=O", "CN1CCC(C)C1=O", "CN1C(CC)CCC1=O",
    ],
)
def test_no_italic_n_locant_survives_on_any_lactam(smiles):
    """The italic form is what P-66.1.3 demotes to general nomenclature."""
    name = _name(smiles)
    assert not name.startswith("N-"), f"italic N- locant in {name!r}"
    assert "-N-" not in name, f"italic N- locant in {name!r}"
    assert "N," not in name, f"italic N locant set in {name!r}"


# --------------------------------------------------------------------------- #
# 2. Identical substituents on ring N and ring C are ONE multiplied prefix.    #
#    (P-16.3.3 -- only reachable once the N locant is a numeral.)              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CN1C(C)CCC1=O", "1,5-dimethylpyrrolidin-2-one"),
        ("CN1CCC(C)C1=O", "1,3-dimethylpyrrolidin-2-one"),
        ("CN1C(C)CCCC1=O", "1,6-dimethylpiperidin-2-one"),
        ("CN1CCCC(C)C1=O", "1,3-dimethylpiperidin-2-one"),
        ("CN1C(C)CC1=O", "1,4-dimethylazetidin-2-one"),
        ("CN1C(C)CCCCC1=O", "1,7-dimethylazepan-2-one"),
    ],
)
def test_identical_n_and_c_substituents_collapse(smiles, expected):
    assert _name(smiles) == expected


def test_three_identical_substituents_collapse_to_tri():
    assert _name("CN1C(C)C(C)CC1=O") == "1,4,5-trimethylpyrrolidin-2-one"


# --------------------------------------------------------------------------- #
# 3. NON-identical substituents must NOT collapse, and sort alphabetically.    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCN1C(C)CCC1=O", "1-ethyl-5-methylpyrrolidin-2-one"),
        ("CN1C(CC)CCC1=O", "5-ethyl-1-methylpyrrolidin-2-one"),
    ],
)
def test_non_identical_substituents_do_not_collapse(smiles, expected):
    assert _name(smiles) == expected
    assert "dimethyl" not in expected and "diethyl" not in expected


# --------------------------------------------------------------------------- #
# 4. The merge removes an input-order nondeterminism.                          #
#    `CN1C(C)CC1=O` and `CC1CC(=O)N1C` are the SAME molecule (verified:         #
#    both InChIKey IYTPSDMCQSELPF-UHFFFAOYSA-N) and used to yield               #
#    `N-methyl-4-methyl...` vs `4-methyl-N-methyl...`.                          #
# --------------------------------------------------------------------------- #

def test_same_molecule_two_smiles_one_name():
    a, b = _name("CN1C(C)CC1=O"), _name("CC1CC(=O)N1C")
    assert a == b == "1,4-dimethylazetidin-2-one"


# --------------------------------------------------------------------------- #
# 5. CONTROLS -- must stay byte-identical (the contributor guide #9: check what is EMITTED).#
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("O=C1CCN1", "azetidin-2-one"),
        ("O=C1CCCN1", "pyrrolidin-2-one"),
        ("O=C1CCCCN1", "piperidin-2-one"),
        # C-substituted only: never touched the N bucket.
        ("CC1CCC(=O)N1", "5-methylpyrrolidin-2-one"),
    ],
)
def test_controls_unchanged(smiles, expected):
    assert _name(smiles) == expected


@pytest.mark.parametrize(
    "smiles",
    ["CC(=O)NC", "COC(=O)NC", "NC(=S)NC1CCCCC1", "CN1CCCC1", "Cn1ccnc1C"],
)
def test_non_lactams_are_refused_by_this_producer(smiles):
    """Boundary. The acyclic amide / carbamate / thiourea controls keep their
    italic 'N-' because they never reach this producer at all (P-66.1.2)."""
    assert name_monocyclic_lactam(Chem.MolFromSmiles(smiles)) is None
