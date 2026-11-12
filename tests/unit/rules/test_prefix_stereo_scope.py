"""A descriptor of a substituent stands at the front of its prefix, not at the front of the name.

The stereo-omission reclaim (``namer._try_compose_input_stereo``) composes the input's stereo on a stereo-free name
with OPSIN's labels, which give every substituent atom its own locant: a double bond inside a substituent was cited
at the front of the whole name with the substituent's locant ('(1E)-4-{...[(...-5-ylidene)methyl]phenoxy}butanoic
acid', C-1 of 'methyl'). OPSIN reads such a name back exactly, because a front descriptor that finds no bond in the
parent is tried on every group of the name, so no read-back sees it.

 "Naming of stereoisomers" (the Blue Book,:44643): "When they relate to substituent groups, they are
cited at the front of the corresponding prefix." "Specification by stereodescriptors 'E' and 'Z'"
(the Blue Book), (1)(a): the double bond of an 'ylidene' group is "an integral part of the parent structure;
the stereodescriptor is placed at the front of the substitutive name, preceded by the locant indicating its point of
attachment to the parent structure" -- "Method (a) generates preferred IUPAC names" ('(1E)-1-(butan-2-ylidene)-1H-
indene (PIN)'). The same holds inside a prefix: the descriptor goes to the front of the part that holds the
'ylidene' prefix.
"""
import pytest
from rdkit import Chem

from orthonym.rules.prefix_stereo import _attachment_texts


@pytest.mark.parametrize("name, prefix, expected", [
    # one enclosing mark deep: the host is the whole name, the locant is the one cited before the prefix
    ("1-[1-(propan-2-yl)azepan-4-ylidene]propan-2-one", "[1-(propan", (0, ["1Z"], "4")),
    ("N'-(2-methylpropylidene)hexanehydrazide", "(2-methyl", (0, ["N'Z"], "1")),
    ("2,7-diethenyl-1,5-di(prop-2-en-1-ylidene)-1H-pyrrolizine", "(prop-2-en-1-ylidene)", (0, ["1Z", "5Z"], "1")),
    ("(propylidene)hydrazine", "(propylidene)", (0, ["Z"], "1")),
    # inside a prefix: the host is the front of the prefix that holds the 'ylidene' prefix; no locant -> bare
    ("4-{2-methoxy-4-[(4-oxo-1,3-thiazolidin-5-ylidene)methyl]phenoxy}butanoic acid", "(4-oxo", None),
    # a prefix that is not an 'ylidene' or 'imino' group: nothing
    ("4-[(2-methylphenyl)methyl]phenol", "(2-methylphenyl)", (0, [], "")),
])
def test_attachment_texts(name, prefix, expected):
    place = name.index(prefix) + 1
    if expected is None:
        expected = (name.index("[(") + 1, ["Z"], "5")
    assert _attachment_texts(name, place, "Z") == expected


# (input SMILES, best-effort name): each descriptor at the front of the part that holds its stereogenic unit.
SCOPE_CASES = [
    # the C=C joins C-5 of the pyrrole (a substituent) to the 'prop-2-en-1-ylidene' group: cited at the front of
    # the pyrrolyl prefix with the pyrrole's locant (was '(1E)-[4-methylidene-5-(prop-2-en-1-ylidene)-1H-pyrrol-3-
    # yl]acetic acid', the locant of the ylidene group's C-1)
    ("C=C\\1C(=CN/C1=C/C=C)CC(=O)O",
     "[(5E)-4-methylidene-5-(prop-2-en-1-ylidene)-1H-pyrrol-3-yl]acetic acid"),
    # the C=C joins the thiazolidine to 'methyl': front of the '[(...ylidene)methyl]' prefix, bare (C-1 of methyl
    # is not cited, (was '(1Z)-N-[(4-{[3-(4-chlorophenyl)-...ylidene]methyl}-...)acetyl]pyrrolidine')
    ("ClC1=CC=C(C=C1)N1C(S\\C(\\C1=O)=C/C1=CC(=C(OCC(=O)N2CCCC2)C=C1)OC)=S",
     "N-[(4-{(Z)-[3-(4-chlorophenyl)-4-oxo-2-sulfanylidene-1,3-thiazolidin-5-ylidene]methyl}-2-methoxyphenoxy)"
     "acetyl]pyrrolidine"),
    # the exocyclic C=C at C-3 of the indolyl prefix (was '(3Z)-N-(2,5-dimethylphenyl)-2-[2-oxo-3-(...)...]acetamide')
    ("CC1=C(C=C(C=C1)C)NC(CN1C2=CC=CC=C2/C(/C1=O)=C\\1/SC(N(C1=O)CCC)=S)=O",
     "N-(2,5-dimethylphenyl)-2-[(3Z)-2-oxo-3-(4-oxo-3-propyl-2-sulfanylidene-1,3-thiazolidin-5-ylidene)-2,3-"
     "dihydro-1H-indol-1-yl]acetamide"),
    # a C=N and a centre, both inside substituents (was '(1Z,2R)-2-[2-ethoxy-4-({[3-(oxolan-2-yl)-...')
    ("C(C)OC1=C(OCC(=O)N(C)C)C=CC(=C1)\\C=N/N1C(=NNC1=S)[C@@H]1OCCC1",
     "2-{2-ethoxy-4-[(1Z)-({3-[(2R)-oxolan-2-yl]-5-sulfanylidene-1,5-dihydro-4H-1,2,4-triazol-4-yl}imino)methyl]"
     "phenoxy}-N,N-dimethylacetamide"),
    # every unit belongs to the parent chain (C=N of the two '-imino' groups at C-1 and C-5, C=C at C-3): the front
    # block stays as it was
    ("Cl\\C(\\C(\\C=C/C=N/OC)=C)=N/OCCCN1CCCCC1",
     "(1Z,3Z,5E)-1-chloro-5-(methoxyimino)-2-methylidene-1-{[3-(piperidin-1-yl)propoxy]imino}pent-3-ene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, expected", SCOPE_CASES)
def test_descriptor_at_the_front_of_its_part(smiles, expected):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    from tests.support.pin_tiers import tier_row
    row = tier_row(smiles, "best-effort")
    assert row["name"] == expected
    read = opsin_parse(expected)
    assert read and Chem.MolToInchiKey(Chem.MolFromSmiles(read)) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_word_fronts_are_places_and_count_as_the_front():
    from orthonym.rules.prefix_stereo import _places, _without_word_front_blocks
    # the acid word of an ester is a front of its own: the part the descriptor relates to), tried before 0
    assert _places("methyl {3-[a]b}acetate")[-2:] == [7, 0]
    assert _without_word_front_blocks("methyl (2Z)-but-2-enoate") == "methyl but-2-enoate"
    assert _without_word_front_blocks("2-[(Z)-a]b") == "2-[(Z)-a]b"


@pytest.mark.opsin_gate
def test_a_double_bond_inside_an_ylidene_group_stays_in_that_group():
    # review 2026-11-12: C-2=C-3 of 'prop-2-en-1-ylidene' is not the bond that joins the group to its host (C-1=C-2
    # of the dinitrile carries two CN and is not stereogenic), so (1)(a) does not move it to the front
    from orthonym.namer import _self_consistency_full_key, _try_compose_input_stereo
    smiles = "N#CC(C#N)=C/C=C/c1ccc([C@H](C)Cl)cc1"
    name = _try_compose_input_stereo("{3-[4-(1-chloroethyl)phenyl]prop-2-en-1-ylidene}propanedinitrile", smiles,
                                     _self_consistency_full_key(smiles))
    assert name == "[(2E)-3-{4-[(1S)-1-chloroethyl]phenyl}prop-2-en-1-ylidene]propanedinitrile"
