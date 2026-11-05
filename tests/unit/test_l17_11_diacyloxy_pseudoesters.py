"""A pseudoester of one polyvalent hub group is named as one functional class ester (lane/l17 item 11).

 "Polyesters formed from a single 'alcoholic' component" (the Blue Book), (:31819):
"When anions are identical functional class multiplicative nomenclature is used. Names are formed by citing the
multivalent group, the multiplicative prefix, and the multiplied anionic component name. Multiplicative prefixes
'di', 'tri', etc. are used when anions are unsubstituted; when substituted, prefixes 'bis', 'tris', etc." --
'ethane-1,2-diyl diacetate (PIN)', 'propane-1,3-diyl bis(chloroacetate) (PIN)'. The Group 13 and 14 hydrides give the
same kind of group: the book's own 'dimethyl dibutylstannanediyl dibutanedioate (PIN; see '
"Substitutive nomenclature: suffix mode",:38202) and 'Al(O-CO-[CH2]16-CH3)3 alumanetriyl tri(octadecanoate) (a
pseudoester)' "Compounds with groups of higher seniority",:37526). The groups are named on the parent
hydrides ('silanediyl', 'stannanetetrayl', 'alumanetriyl':,.

Labels. The hydrides of Si, Ge, Sn and Pb are the parents of PINs "PREFERRED IUPAC NAMES",:2056) and the book
prints the PIN of the stannanediyl ester of a dibasic acid (:38202) and of the distannoxanediyl dipropanoate (:38252),
so the ester of one of them with a plain anion (formic acid, an unbranched chain, an aromatic or saturated monocycle)
is the PIN. The compounds of Al, Ga, In and Tl have none PRESELECTED NAMES,:2062: 'the names based on these
parent hydrides currently do not have PIN status', and the book's 'alumanetriyl tri(octadecanoate) (a pseudoester)',
:37526, carries no label), and neither has a name whose anion is substituted, has several rings or begins with a
multiplying prefix: the default tier gives no name for those and the best-effort tier gives the checked name.

The multiplied anion is spelled as to and spell it ('di' for an unsubstituted anion, 'bis' for
a substituted one, enclosed when it cites locants, a descriptor or brackets: 'di[(3E,5E)-hepta-3,5-dienoate]',:47993).
An anion that begins with a multiplying prefix ('dodecanoate') is spelled bare, as the glycerides of are
(:55107, no PIN label), and the name stays below the PIN: (c), (d) enclose it ('tri(octadecanoate)',:37526).

Before the change the pseudoester producer declined an organyl atom that bears several identical groups ("the one-cut
name with the other groups as '(acyloxy)' prefixes is no name of this rule") and the molecule fell to the general
engine's 'lateral' name, '1-{[(acetyloxy)di(butyl)stannyl]oxy}-1-oxoethane'.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


def _tin_diacylate(carbons):
    """Dibutyltin diacylate of the unbranched acid of ``carbons`` carbon atoms."""
    acyl = "C" * (carbons - 1)
    return f"CCCC[Sn](CCCC)(OC(=O){acyl})OC(=O){acyl}"


# (SMILES, best-effort name, whether the PIN tier gives it as the PIN)
_CASES = [
    # Group 14: the group is '<substituents><hydride>diyl', two acyloxy groups on one atom
    ("CCCC[Sn](CCCC)(OC(C)=O)OC(C)=O", "dibutylstannanediyl diacetate", True),
    ("C[Si](C)(OC(C)=O)OC(C)=O", "dimethylsilanediyl diacetate", True),
    ("C[Ge](C)(OC(C)=O)OC(C)=O", "dimethylgermanediyl diacetate", True),
    ("C[Pb](C)(OC(C)=O)OC(C)=O", "dimethylplumbanediyl diacetate", True),
    # a long acyl: 'dibutyltin dilaurate'. The anion begins with a multiplying prefix, which (c), (d) enclose
    # ('di(decanoic acid)', 'tri(octadecanoate)',:37526) and the book prints bare in (:55107): below the PIN
    ("CCCCCCCCCCCC(=O)O[Sn](CCCC)(CCCC)OC(=O)CCCCCCCCCCC", "dibutylstannanediyl didodecanoate", False),
    ("C[Si](C)(OC(=O)CCCCCCCCC)OC(=O)CCCCCCCCC", "dimethylsilanediyl didecanoate", False),
    # the table of multiplying prefixes goes past 'icosa': 'henicosa', 'docosa', 'hentriaconta', 'dotriaconta'
    # begin the anion of an acid of 21 or more carbon atoms as 'dodeca' begins the one of 12, so none of them is certain
    (_tin_diacylate(20), "dibutylstannanediyl diicosanoate", False),
    (_tin_diacylate(21), "dibutylstannanediyl dihenicosanoate", False),
    (_tin_diacylate(22), "dibutylstannanediyl didocosanoate", False),
    (_tin_diacylate(31), "dibutylstannanediyl dihentriacontanoate", False),
    (_tin_diacylate(32), "dibutylstannanediyl didotriacontanoate", False),
    # an anion of a chain of 4 carbon atoms or fewer begins with no multiplying prefix, and the one of 5 does ('penta')
    (_tin_diacylate(4), "dibutylstannanediyl dibutanoate", True),
    (_tin_diacylate(5), "dibutylstannanediyl dipentanoate", False),
    # three and four groups: '-triyl', '-tetrayl'
    ("CCCC[Sn](OC(C)=O)(OC(C)=O)OC(C)=O", "butylstannanetriyl triacetate", True),
    ("CC(=O)O[Sn](OC(C)=O)(OC(C)=O)OC(C)=O", "stannanetetrayl tetraacetate", True),
    ("CC(=O)O[Pb](OC(C)=O)(OC(C)=O)OC(C)=O", "plumbanetetrayl tetraacetate", True),
    # hydrogen and several kinds of organyl on the hub: the block of
    ("C[SiH](OC(C)=O)OC(C)=O", "methylsilanediyl diacetate", True),
    ("c1ccccc1[Si](C)(OC(C)=O)OC(C)=O", "methyl(phenyl)silanediyl diacetate", True),
    # an anion that opens with a stereodescriptor is enclosed, as the book's 'di[(3E,5E)-hepta-3,5-dienoate]' (:47993)
    ("C[Si](C)(OC(=O)/C=C/C)OC(=O)/C=C/C", "dimethylsilanediyl di[(2E)-but-2-enoate]", True),
    # an unsubstituted anion that cites a locant is enclosed (e)); the anion of the formic acid, a benzene
    # ring and a saturated ring is plain
    ("C[Si](C)(OC(=O)C=C)OC(=O)C=C", "dimethylsilanediyl di(prop-2-enoate)", True),
    ("C[Si](C)(OC=O)OC=O", "dimethylsilanediyl diformate", True),
    ("C[Si](C)(OC(=O)c1ccccc1)OC(=O)c1ccccc1", "dimethylsilanediyl dibenzoate", True),
    ("C[Si](C)(OC(=O)C1CCCCC1)OC(=O)C1CCCCC1", "dimethylsilanediyl dicyclohexanecarboxylate", True),
    # a ring acid the acid namer names, whatever its ring: the acid is the acid of the name (the unsubstituted anion
    # takes 'di' and its marks, (e), (f))
    ("C[Si](C)(OC(=O)c1ccncc1)OC(=O)c1ccncc1", "dimethylsilanediyl di(pyridine-4-carboxylate)", True),
    ("C[Si](C)(OC(=O)c1ccc[nH]1)OC(=O)c1ccc[nH]1", "dimethylsilanediyl di(1H-pyrrole-2-carboxylate)", True),
    ("C[Si](C)(OC(=O)[C@@H]1CCCO1)OC(=O)[C@@H]1CCCO1", "dimethylsilanediyl di[(2S)-oxolane-2-carboxylate]", True),
    ("C[Si](C)(OC(=O)c1ccc2ccccc2c1)OC(=O)c1ccc2ccccc2c1", "dimethylsilanediyl di(naphthalene-2-carboxylate)", False),
    ("C[Si](C)(OC(=O)c1ccc(cc1)-c1ccccc1)OC(=O)c1ccc(cc1)-c1ccccc1",
     "dimethylsilanediyl di([1,1'-biphenyl]-4-carboxylate)", False),
    # a substituted anion takes 'bis', 'propane-1,3-diyl bis(chloroacetate)'), with the marks of
    # after a descriptor: 'bis[(2S)-2-chloropropanoate]' (not 'bis((2S)-...)')
    ("C[Si](C)(OC(=O)C(F)(F)F)OC(=O)C(F)(F)F", "dimethylsilanediyl bis(trifluoroacetate)", False),
    ("C[Si](C)(OC(=O)[C@H](C)Cl)OC(=O)[C@H](C)Cl", "dimethylsilanediyl bis[(2S)-2-chloropropanoate]", False),
    ("C[Si](C)(OC(=O)/C(C)=C/C)OC(=O)/C(C)=C/C", "dimethylsilanediyl bis[(2E)-2-methylbut-2-enoate]", False),
    # the carbamic acid is a retained name (a)): 'di', not 'bis'
    ("C[Si](C)(OC(=O)N)OC(=O)N", "dimethylsilanediyl dicarbamate", False),
    # Group 13: preselected parent hydrides, no PIN
    ("CC(=O)O[Al](OC(C)=O)OC(C)=O", "alumanetriyl triacetate", False),
    ("FC(F)(F)C(F)(F)C(=O)O[Al](OC(=O)C(F)(F)C(F)(F)F)OC(=O)C(F)(F)C(F)(F)F",
     "alumanetriyl tris(pentafluoropropanoate)", False),
    ("CC(=O)O[Al](C)OC(C)=O", "methylalumanediyl diacetate", False),
    ("CC(=O)O[Ga](OC(C)=O)OC(C)=O", "gallanetriyl triacetate", False),
]


@pytest.mark.parametrize("smiles,name,pin", _CASES)
def test_the_hub_group_is_named_as_one_ester(smiles, name, pin):
    """The best-effort name is the one of and OPSIN 2.9.0 reads it back to the full InChIKey of the input."""
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name,pin", _CASES)
def test_the_label_of_the_name(smiles, name, pin):
    """The PIN tier gives the name as the PIN where the book prints the PIN of this class (Si, Ge, Sn, Pb, a plain
    anion), and no name where it does not: Al, Ga, In, Tl; a substituted or multi-ring anion)."""
    wide = _row(smiles, "best-effort")
    narrow = _row(smiles, "pin")
    if pin:
        assert wide["is_pin"] and wide["tier"] == "pin_verified", wide
        assert narrow["name"] == name and narrow["is_pin"], narrow
    else:
        assert not wide["is_pin"] and wide["tier"] != "pin_verified", wide
        assert narrow["tier"] == "abstain" and not narrow["is_pin"], narrow


def test_the_name_does_not_depend_on_the_atom_order():
    """The same molecule drawn from either acyloxy group, with the organyls first or last, has one name."""
    drawings = [
        "CCCC[Sn](CCCC)(OC(C)=O)OC(C)=O",
        "CC(=O)O[Sn](CCCC)(CCCC)OC(C)=O",
        "O=C(C)O[Sn](OC(C)=O)(CCCC)CCCC",
        "CCCC[Sn](OC(C)=O)(CCCC)OC(C)=O",
    ]
    names = {_row(s, "best-effort")["name"] for s in drawings}
    assert names == {"dibutylstannanediyl diacetate"}, names


# what the change must leave as it was
def test_one_acyloxy_group_keeps_the_monoester_name():
    """'tributylstannyl acetate' is the name of one group and keeps the PIN label it had."""
    row = _row("CCCC[Sn](CCCC)(CCCC)OC(C)=O", "best-effort")
    assert row["name"] == "tributylstannyl acetate" and row["is_pin"], row


@pytest.mark.parametrize("smiles", [
    "C[Si](C)(OC(C)=O)OC(=O)CC",                       # two different anions: method (1) or (2) of
    "CCCC[Sn](OC(C)=O)(OC(C)=O)OC(=O)CC",
    "C[Sn](C)(OC(=O)CCC(=O)O)OC(=O)CCC(=O)O",          # a free carboxyl on the acid: 'butanedioate' would be another molecule
    "C[Si](Cl)(OC(C)=O)OC(C)=O",                       # a halogen on the hub: no 'chloro' block is built
    "CC(=O)O[Si](C)(C)O[Si](C)(C)OC(C)=O",             # two hubs
    "C=CCN(CCN)CCC[Si](OC(C)=O)(OC(C)=O)OC(C)=O",      # an organyl with a nitrogen: no prefix name for it here
])
def test_a_shape_outside_the_class_is_not_named_as_a_hub_group(smiles):
    """The producer declines (the molecule keeps the name it had) and no name it ships is a wrong molecule."""
    row = _row(smiles, "best-effort")
    assert not (row["name"] or "").startswith(("dimethylsilanediyl", "dimethylstannanediyl", "butylstannanetriyl")), row
    # no multivalent group of the class is cited at all, whatever its hub
    assert not any(word in (row["name"] or "") for word in ("diyl ", "triyl ", "tetrayl ")), row
    if row["name"] and row["source"] != "abstain":
        assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)OCCOC(C)=O", "ethane-1,2-diyl diacetate"),
    ("CC(=O)OCC(OC(C)=O)COC(C)=O", "propane-1,2,3-triyl triacetate"),
])
def test_the_carbon_polyol_esters_keep_their_pins(smiles, name):
    row = _row(smiles, "pin")
    assert row["name"] == name and row["is_pin"], row


def test_the_book_stannanediyl_diester_of_a_dibasic_acid_is_untouched():
    """ (:38202): 'dimethyl dibutylstannanediyl dibutanedioate' is a polyester of two alcohols and two acid
    units; this producer names one acid and one group and leaves it as it was."""
    smiles = "CCCC[Sn](CCCC)(OC(=O)CCC(=O)OC)OC(=O)CCC(=O)OC"
    row = _row(smiles, "best-effort")
    assert row["name"] and "diacetate" not in row["name"], row
    assert_full_rt(row["name"], smiles)


# the labelled forms: the isotope decorator names the label-free skeleton and places the descriptor in it
@pytest.mark.parametrize("smiles,name", [
    # every anion carries the label: one descriptor per anion, enclosed as the book's '1,2-di[(13C)methyl]benzene (PIN)'
    #, "stereochemistry, specific isotopic substitution": the parentheses of a descriptor count)
    ("C[Si](C)(OC(=O)[13CH3])OC(=O)[13CH3]", "dimethylsilanediyl di[(2-13C)acetate]"),
    ("[2H]C([2H])([2H])C(=O)O[Si](C)(C)OC(=O)C([2H])([2H])[2H]", "dimethylsilanediyl di[(2H3)acetate]"),
    ("[2H]C([2H])([2H])C(=O)OCCOC(=O)C([2H])([2H])[2H]", "ethane-1,2-diyl di[(2H3)acetate]"),
    ("CCCC[Sn](CCCC)(OC(=O)[14CH3])OC(=O)[14CH3]", "dibutylstannanediyl di[(2-14C)acetate]"),
])
def test_a_label_on_every_anion_is_cited_once_inside_the_multiplier(smiles, name):
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "C[Si](C)(OC(=O)[13CH3])OC(C)=O",
    "[2H]C([2H])([2H])C(=O)O[Si](C)(C)OC(C)=O",
    "C[Si](C)(OC(C)=O)OC(=O)C([2H])([2H])[2H]",
    "C[Si](C)(O[13C](C)=O)OC(C)=O",
    "CCCC[Sn](CCCC)(OC(=O)C)OC(=O)[14CH3]",
    "[2H]C([2H])([2H])C(=O)OCCOC(C)=O",
    "[13CH3]C(=O)OCCOC(=O)C",
])
def test_a_label_on_one_of_identical_anions_keeps_a_name(smiles):
    """One labelled anion among identical ones makes them unlike: the Blue Book cites the anions apart
    method (1), 'ethane-1,2-diyl 1-acetate 2-propanoate'), a form OPSIN 2.9.0 does not read, so it cannot be checked.
    The decorator names the groups as acyloxy prefixes instead (method (2), "acceptable in general nomenclature"):
    a name, below the PIN, that reads back to the labelled molecule. Best-effort never loses a name."""
    row = _row(smiles, "best-effort")
    assert row["name"] and row["tier"] != "abstain" and not row["is_pin"], row
    assert any(token in row["name"] for token in ("13C", "14C", "2H")), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # (e) 'functionalized parent hydrides with locants' are enclosed, as the salts word them ('zinc
    # di(prop-2-enoate)'); (f) /:47993 'di[(3E,5E)-hepta-3,5-dienoate]'
    ("C=CC(=O)OCCOC(=O)C=C", "ethane-1,2-diyl di(prop-2-enoate)"),
    ("C/C=C/C(=O)OCCOC(=O)/C=C/C", "ethane-1,2-diyl di[(2E)-but-2-enoate]"),
])
def test_the_carbon_polyol_ester_spells_its_multiplied_anion_alike(smiles, name):
    """The carbon polyol esters and the hub groups join the multiplier to the anion by one primitive."""
    row = _row(smiles, "pin")
    assert row["name"] == name and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=C(OCCOC(=O)c1ccncc1)c1ccncc1", "ethane-1,2-diyl di(pyridine-4-carboxylate)"),
    ("O=C(OCCOC(=O)c1cccnc1)c1cccnc1", "ethane-1,2-diyl di(pyridine-3-carboxylate)"),
    ("O=C(OCCOC(=O)c1ccco1)c1ccco1", "ethane-1,2-diyl di(furan-2-carboxylate)"),
    ("O=C(OCCOC(=O)c1ccc[nH]1)c1ccc[nH]1", "ethane-1,2-diyl di(1H-pyrrole-2-carboxylate)"),
    ("O=C(OCC(OC(=O)c1ccncc1)COC(=O)c1ccncc1)c1ccncc1", "propane-1,2,3-triyl tri(pyridine-4-carboxylate)"),
])
def test_the_carbon_polyol_ester_of_a_ring_acid_is_the_functional_class_name(smiles, name):
    """ "When anions are identical functional class multiplicative nomenclature is used" (the Blue Book):
    the diol and the triol of identical heterocyclic acids are named as the benzoates are ('ethane-1,2-diyl dibenzoate'),
    the unsubstituted anion with its locant enclosed (e)). The acyl namer has no word for these acids, and the
    producer used to decline, leaving the substitutive '2-[(pyridine-4-carbonyl)oxy]ethyl pyridine-4-carboxylate'
    under the PIN label: the book's PIN is the multiplicative name, and the acid is read as the hub groups' is."""
    wide = _row(smiles, "best-effort")
    narrow = _row(smiles, "pin")
    assert wide["name"] == name and wide["is_pin"] and wide["tier"] == "pin_verified", wide
    assert narrow["name"] == name and narrow["is_pin"], narrow
    assert_full_rt(wide["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # a ring system of several rings: the spelling is not certain (f)), so the name is below the PIN
    ("O=C(OCCOC(=O)c1ccc2ccccc2c1)c1ccc2ccccc2c1", "ethane-1,2-diyl di(naphthalene-2-carboxylate)"),
    ("O=C(OCCCCOC(=O)c1cc2ccccc2[nH]1)c1cc2ccccc2[nH]1", "butane-1,4-diyl di(1H-indole-2-carboxylate)"),
])
def test_a_ring_system_acid_of_a_carbon_polyol_ester_is_named_below_the_pin(smiles, name):
    wide = _row(smiles, "best-effort")
    narrow = _row(smiles, "pin")
    assert wide["name"] == name and not wide["is_pin"] and wide["tier"] != "pin_verified", wide
    assert narrow["tier"] == "abstain" and not narrow["is_pin"], narrow
    assert_full_rt(wide["name"], smiles)


def test_two_unlike_ring_acids_are_not_one_multiplied_anion():
    """The acids differ, so no multiplied anion is built: the molecule keeps the name of the producers of unlike anions
     method (2)), below the PIN."""
    smiles = "O=C(OCCOC(=O)c1ccncc1)c1ccccc1"
    row = _row(smiles, "best-effort")
    assert row["name"] and "di(" not in row["name"] and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)
