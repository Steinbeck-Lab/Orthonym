"""'rel'/'rac' names for a molecule whose whole configuration is one OR or AND stereo group
(orthonym.assembly.stereo_group_names; program 20, from the SureChEMBL study).

 'Relative configuration', (the Blue Book): "This prefix 'rel' with plain
stereodescriptors is preferred in preferred IUPAC names"; 'Racemates' (:45914): "This prefix 'rac'
with plain stereodescriptors is preferred in preferred IUPAC names... the stereodescriptor... is cited for
the chirality center having the lowest locant". Each expected name below is a (PIN) example of the book.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.stereo_group_names import _forms, _partial_forms, general_form, leading_descriptor

pytestmark = pytest.mark.opsin_gate


def _mirror(smiles):
    """The same string with every tetrahedral centre inverted (the group annotation kept)."""
    body, _, ext = smiles.partition(" |")
    flipped = body.replace("@@", "\0").replace("@", "@@").replace("\0", "@")
    return flipped + (" |" + ext if ext else "")


BOOK = [
    #:45905
    ("OC(=O)[C@H]1CCC[C@@H]1Cl |o1:3,7|", "rel-(1R,2S)-2-chlorocyclopentane-1-carboxylic acid"),
    #:45925
    ("OC(=O)[C@H]1CCC[C@@H]1Cl |&1:3,7|", "rac-(1R,2S)-2-chlorocyclopentane-1-carboxylic acid"),
    #:45931
    ("C[C@H](O)[C@H](C)Br |&1:1,3|", "rac-(2R,3R)-3-bromobutan-2-ol"),
    #:45940
    ("C[C@@H](Br)CC |&1:1|", "rac-(2R)-2-bromobutane"),
    #:48167 (the cis acid),:48183,:48213
    ("O=C(O)[C@@H]1CCC[C@@H]1Cl |o1:3,7|", "rel-(1R,2R)-2-chlorocyclopentane-1-carboxylic acid"),
    ("Br[C@@H]1CC[C@H](Br)[C@H](Br)C1 |o1:1,4,6|", "rel-(1R,2S,4S)-1,2,4-tribromocyclohexane"),
    ("O=C(O)[C@@]1(Cl)CCC[C@@H]1Cl |o1:3,8|", "rel-(1R,2R)-1,2-dichlorocyclopentane-1-carboxylic acid"),
]


@pytest.mark.parametrize("smiles, expected", BOOK)
def test_the_book_examples_from_either_enantiomer(smiles, expected):
    engine = Orthonym()
    for drawn in (smiles, _mirror(smiles)):
        row = engine.name_tiered(drawn)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), (drawn, row)


@pytest.mark.parametrize("smiles", [
    "C[C@H](O)CC |o1:1|",                          # one OR centre: no relative configuration to state
    "C[C@H](O)C[C@@H](C)CC |a:1,&1:4|",            # a group beside an absolute centre
    "C[C@H](O)[C@H](O)C |&1:1|",                   # one of two centres in the group
    "C[C@H](O)[C@H](O)C |o1:1,3|",                 # meso: the group inverts to itself
    "C[C@H](O)CC.C[C@H](O)CC |&1:1,6|",            # two fragments
    "C[C@H](O)C[C@@H](C)O |&1:1,&2:4|",            # two groups
    "C/C=C/[C@H](C)O |o1:4|",                      # a stereo double bond beside the group
    "C[C@H](O)C[S@](=O)c1ccccc1 |&1:1,5|",         # a lone-pair centre in the group
    "C[C@H](O)CC",                                 # no group
    "C[C@H](O)CC |a:1|",                           # an absolute group only
])
def test_outside_the_scope_nothing_is_built(smiles):
    assert _forms(smiles) is None


def test_the_scope_gives_both_enantiomers():
    prefix, drawn, mirror = _forms("C[C@H](O)[C@@H](C)CC |&1:1,3|")
    assert prefix == "rac" and drawn != mirror
    assert _forms("C[C@H](O)[C@@H](C)CC |o1:1,3|")[0] == "rel"
    assert Chem.MolToInchi(Chem.MolFromSmiles(drawn)) != Chem.MolToInchi(Chem.MolFromSmiles(mirror))


@pytest.mark.parametrize("name, code", [
    ("(1R,2S)-2-chlorocyclopentane-1-carboxylic acid", "R"),
    ("(1S,2R)-2-chlorocyclopentane-1-carboxylic acid", "S"),
    ("(2R)-butan-2-ol", "R"),
    ("(R)-(methanesulfinyl)benzene", "R"),
    ("(2E,4S)-hex-2-en-4-ol", "S"),
    ("(4aR,8aS)-decahydronaphthalene", "R"),
    ("(1r,4r)-cyclohexane-1,4-diol", None),
    ("(2RS)-2-bromobutane", None),
    ("(1R*,2S*)-2-chlorocyclopentane-1-carboxylic acid", None),
    ("methyl (2S)-2-aminopropanoate", None),
    ("butan-2-ol", None),
])
def test_leading_descriptor(name, code):
    assert leading_descriptor(name) == code


@pytest.mark.parametrize("name, prefix, expected", [
    ("tert-butyl (2R,5S)-2-(3-aminophenyl)-5-methylpyrrolidine-1-carboxylate", "rel",
     "tert-butyl (2R*,5S*)-2-(3-aminophenyl)-5-methylpyrrolidine-1-carboxylate"),
    ("sodium (2R,3S)-2,3-dihydroxybutanoate", "rac", "sodium (2RS,3SR)-2,3-dihydroxybutanoate"),
    ("N-[(1R,2S)-2-phenylcyclopropyl]acetamide", "rel", "N-[(1R*,2S*)-2-phenylcyclopropyl]acetamide"),
    ("(2R)-2-[(1S)-1-hydroxyethyl]oxirane", "rac", "(2RS)-2-[(1SR)-1-hydroxyethyl]oxirane"),
    ("methyl (2S)-2-amino-3-methylbutanoate", "rac", None),         # the first descriptor must be R
    ("butan-2-ol", "rel", None),                                     # nothing to rewrite
    # review 2026-11-08: a block the locators do not read must not stay plain (absolute) in a racemic name
    ("methyl (1R,2R)-2-{(1R)-1-[(tert-butoxycarbonyl)amino]ethyl}cyclopropane-1-carboxylate", "rac", None),
    ("methyl (1R,2S)-2-[(1'R)-x]", "rel", None),
])
def test_general_form(name, prefix, expected):
    #:45901 ('R*', 'S*'),:45914 ('RS', 'SR'): the general forms
    assert general_form(name, prefix) == expected


def test_the_general_form_is_not_given_at_the_pin_only_tier():
    smiles = "CC(C)[C@@H](C(=O)OC)N |&1:3|"
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"
    wide = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    for drawn in (smiles, _mirror(smiles)):
        row = wide.name_tiered(drawn)
        assert (row["name"], row["tier"]) == ("methyl (2RS)-2-amino-3-methylbutanoate", "systematic_verified")


def test_a_block_left_plain_never_ships():
    # review 2026-11-08: the old rewrite gave 'methyl (1RS,2RS)-2-{(1R)-1-...}' (a mixture of two diastereomers)
    smiles = "COC(=O)[C@@H]1C[C@H]1[C@@H](C)NC(=O)OC(C)(C)C |&1:4,6,7|"
    wide = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    name = wide.name_tiered(smiles)["name"]
    assert "(1R)" not in name and "(1S)" not in name, name


def test_a_non_tetrahedral_centre_is_outside_the_scope():
    assert _forms("C[C@H](O)[C@@H](C)Cl.[Pt@SP1](Cl)(Cl)(N)N |&1:1,3|") is None


# ---- part 2: one group beside absolute centres:45910,:45914) ----------------------------------

def _invert(smiles, atoms):
    body, _, ext = smiles.partition(" |")
    mol = Chem.MolFromSmiles(body)
    for k in atoms:
        atom = mol.GetAtomWithIdx(k)
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                          if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                          else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(mol, canonical=False) + " |" + ext


@pytest.mark.parametrize("smiles, members, expected, tier", [
    # the book's own example (:45912), named as a checked systematic name by the engine
    ("CC[C@H](C)O[C@H]1C[C@@H](Cl)C[C@@H]([N+](=O)[O-])C1 |o1:5,7,10|", [5, 7, 10],
     None, None),
    ("C[C@H](O)C[C@@H](C)[C@H](C)Cl |o1:4,6|", [4, 6], "(2S,4R*,5S*)-5-chloro-4-methylhexan-2-ol", "pin_verified"),
    ("C[C@H](O)C[C@@H](C)[C@H](C)Cl |&1:4,6|", [4, 6], "(2S,4RS,5SR)-5-chloro-4-methylhexan-2-ol",
     "systematic_verified"),
    ("C[C@H](O)C[C@@H](C)CC |&1:4|", [4], "(2S,4RS)-4-methylhexan-2-ol", "systematic_verified"),
])
def test_a_group_beside_absolute_centres(smiles, members, expected, tier):
    wide = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    names = {wide.name_tiered(s)["name"] for s in (smiles, _invert(smiles, members))}
    assert len(names) == 1, names                 # the same name whichever member configuration was drawn
    if expected:
        row = wide.name_tiered(smiles)
        assert (row["name"], row["tier"]) == (expected, tier)
    else:
        assert names.pop().startswith("(1R*,"), names


def test_the_book_example_of_starred_descriptors():
    wide = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    found = set()
    for a in ("@", "@@"):
        for b in ("@", "@@"):
            for c in ("@", "@@"):
                found.add(wide.name_tiered(f"CC[C@H](C)O[C{a}H]1C[C{b}H](Cl)C[C{c}H]([N+](=O)[O-])C1 |o1:5,7,10|")["name"])
    assert "(1R*,3R*,5S*)-1-{[(2S)-butan-2-yl]oxy}-3-chloro-5-nitrocyclohexane" in found
    assert len(found) == 4 and all(n.startswith("(1R*,") for n in found)


def test_partial_scope():
    assert _partial_forms("C[C@H](O)C[C@@H](C)CC |o1:4|") is None          # one OR centre: nothing to state
    assert _partial_forms("C[C@H](O)[C@@H](C)Cl |o1:1,3|") is None         # no absolute centre: part 1
    assert _partial_forms("C[C@H](O)C[C@@H](C)[C@H](C)Cl |&1:4,&2:6|") is None   # two groups
    assert _partial_forms("C[C@H](O)C[C@@H](C)[C@H](C)Cl |o1:4,6|")[0] == "rel"
    # an OR group of the PIN-only call keeps its PIN form; an AND group is not given there
    assert Orthonym().name_tiered("C[C@H](O)C[C@@H](C)[C@H](C)Cl |o1:4,6|")["name"] == \
        "(2S,4R*,5S*)-5-chloro-4-methylhexan-2-ol"
    assert Orthonym().name_tiered("C[C@H](O)C[C@@H](C)[C@H](C)Cl |&1:4,6|")["tier"] == "abstain"


def test_review_cases_of_part_2():
    wide = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    # C-4 is pseudoasymmetric in one form only: a name leaving it out would fit two inputs, so nothing is built
    for smiles in ("C[C@H](Br)[C@H](Cl)[C@H](O)[C@H](Cl)[C@H](Br)C |o1:1,3|",
                   "C[C@H](Br)[C@H](Cl)[C@@H](O)[C@H](Cl)[C@H](Br)C |o1:1,3|"):
        assert "*" not in (Orthonym().name_tiered(smiles)["name"] or "")
        assert "*" not in (wide.name_tiered(smiles)["name"] or "")
    # a symmetric molecule: every redrawing (atom order) of one input gives one name
    import random
    base = Chem.MolFromSmiles("C[C@H](Cl)[C@H](Cl)CC[C@H](Cl)[C@H](Cl)C |o1:1,3|")
    rng = random.Random(7)
    drawings = []
    for _ in range(6):
        order = list(range(base.GetNumAtoms()))
        rng.shuffle(order)
        drawings.append(Chem.MolToCXSmiles(Chem.RenumberAtoms(base, order)))
    names = {wide.name_tiered(s)["name"] for s in drawings}
    assert names == {"(2R*,3S*,6S,7R)-2,3,6,7-tetrachlorooctane"}, names
