"""PIN class program, topic HX: hydroximic and thiohydroximic acids with a substituted acyl group.

 "Substitutive nomenclature, suffix mode" (the Blue Book-30096): "Acids in which the
carbonyl oxygen atom of a carboxylic acid group has been replaced by =N-OH are named as *N*-hydroxy
derivatives of imidic acids named as described in. This method is used to generate preferred
IUPAC names." The N-hydroxy is a prefix like any other, so it is cited with the acyl group's own
prefixes in one alphanumerical series "ALPHANUMERICAL ORDER", the Blue Book,:3440;
'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)',:32879, puts the italic N and the numbers in one list),
its italic locant is essential "Citation of locants",:2869) so no licence withholds a
locant beside it, and one that the acyl group also carries is one multiplied prefix (b),:7067).
The chalcogen analogues are named the same way on the imidothioic, imidoselenoic and imidotelluroic acids
,:30208: 'N-hydroxycyclohexanecarboximidoselenoic acid (PIN)',:30273).

The engine used to build only a bare acyl group (an unbranched chain or one bare ring): it put 'N-hydroxy'
in front of the name of the imidic acid, which is right only when that name has no prefix of its own. The
substituted group got the spelling 'N-hydroxy2-methylpropanimidic acid' at the best-effort tier, nothing at
the default tier, or (for the thio, seleno and telluro acids) the name of a thiol, which is not the PIN
(the acid outranks the thiol, Table 4.3,:18537).

A row is the Blue Book's own where the comment quotes its line; the others are what the rules above give
for a compound the book does not print, and each is read back by OPSIN (``assert_pin_at_both_tiers``)
to the input's full InChIKey; the configuration of a C=N, which that key does not see, is read back to
the isomeric SMILES (``_isomeric_read_back``).
"""
import re

import pytest

from orthonym import Orthonym
from orthonym.rules.hydroximic_acids import cn_configuration_verdict, junior_to, suffix_group
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# the Blue Book's own (PIN) rows
BB_ROWS = [
    ("ON=C(C)O", "N-hydroxyethanimidic acid"),                                #:30106
    ("ON=C(O)c1ccccc1", "N-hydroxybenzenecarboximidic acid"),                 #:30108
    ("ON=C(O)CCC", "N-hydroxybutanimidic acid"),                              #:30110
    ("ON=C(O)CC", "N-hydroxypropanimidic acid"),                              #:17576
    ("ON=C([SeH])C1CCCCC1", "N-hydroxycyclohexanecarboximidoselenoic acid"),  #:30273
    # (1),:30131: the group demoted by a senior one on a ring is a compound prefix
    ("OC(=O)c1ccc(C(O)=NO)cc1", "4-(C,N-dihydroxycarbonimidoyl)benzoic acid"),  #:30138
    ("ON=C(O)C1CCCC1C(=O)O",
     "2-(C,N-dihydroxycarbonimidoyl)cyclopentane-1-carboxylic acid"),         #:30134
    # (2),:30132: at the end of a chain it is 'hydroxy' and 'hydroxyimino'
    ("ON=C(O)CCCC(=O)O", "5-hydroxy-5-(hydroxyimino)pentanoic acid"),         #:30140
]

# the rules above composed for a compound the book does not print
CHAIN_ROWS = [
    # 'N-hydroxy' after the chain's own prefix when it sorts after it, before when it does not
    ("CC(C)C(=NO)O", "N-hydroxy-2-methylpropanimidic acid"),
    ("CC(C)C(=NO)S", "N-hydroxy-2-methylpropanimidothioic acid"),
    ("CC(C)C(=NO)[SeH]", "N-hydroxy-2-methylpropanimidoselenoic acid"),
    ("CC(C)C(=NO)[TeH]", "N-hydroxy-2-methylpropanimidotelluroic acid"),
    ("ClCC(=NO)S", "2-chloro-N-hydroxyethanimidothioic acid"),
    ("ON=C(S)CCN", "3-amino-N-hydroxypropanimidothioic acid"),
    ("ON=C(O)C(C)(C)C", "N-hydroxy-2,2-dimethylpropanimidic acid"),
    # a substituent with its own enclosing marks
    ("CSCCCCCCCCC(S)=NO", "N-hydroxy-9-(methylsulfanyl)nonanimidothioic acid"),
    ("ON=C(S)Cc1ccccc1", "N-hydroxy-2-phenylethanimidothioic acid"),
    ("ON=C(S)Cc1ccc(O)cc1", "N-hydroxy-2-(4-hydroxyphenyl)ethanimidothioic acid"),
    # a group junior to the acid, cited as a prefix
    ("ON=C(O)CC(=O)C", "N-hydroxy-3-oxobutanimidic acid"),
    ("ON=C(S)CCC#N", "3-cyano-N-hydroxypropanimidothioic acid"),
    ("CS(=O)CCCC(S)=NO", "N-hydroxy-4-(methanesulfinyl)butanimidothioic acid"),
    # a 'hydroxy' of the chain and the N-hydroxy are one multiplied prefix, the italic N first
    ("ON=C(O)CC(O)C", "N,3-dihydroxybutanimidic acid"),
    ("ON=C(S)C(O)C", "N,2-dihydroxypropanimidothioic acid"),
    ("ON=C(O)CC(O)CO", "N,3,4-trihydroxybutanimidic acid"),
    # the N locant is essential, so the locants beside it are all cited, whatever the
    # hydrogens left on the chain would license without it,
    ("ON=C(O)C(F)(F)C(F)(F)F", "2,2,3,3,3-pentafluoro-N-hydroxypropanimidic acid"),
    ("ON=C(S)CS", "N-hydroxy-2-sulfanylethanimidothioic acid"),
    ("ON=C(S)CSC", "N-hydroxy-2-(methylsulfanyl)ethanimidothioic acid"),
    ("ON=C(O)C=O", "N-hydroxy-2-oxoethanimidic acid"),
    # unsaturation outside C-2: the spelling check read 'but-3-enimidic' as a suffix at C-3
    ("ON=C(O)CC=C", "N-hydroxybut-3-enimidic acid"),
    ("ON=C(O)CCC=C", "N-hydroxypent-4-enimidic acid"),
]

STEREO_ROWS = [
    # the descriptor opens the name,:44639); C=N: the OH on N against the OH or SH on C
    ("O/N=C(/S)Cc1c[nH]c2ccccc12", "(1E)-N-hydroxy-2-(1H-indol-3-yl)ethanimidothioic acid"),
    ("O/N=C(\\O)Cc1c[nH]c2ccccc12", "(1Z)-N-hydroxy-2-(1H-indol-3-yl)ethanimidic acid"),
    ("C/C(O)=N/O", "(1Z)-N-hydroxyethanimidic acid"),
    ("C/C(O)=N\\O", "(1E)-N-hydroxyethanimidic acid"),
    ("O/N=C(/S)CCc1ccccc1", "(1E)-N-hydroxy-3-phenylpropanimidothioic acid"),
    ("C/C=C/C(=NO)O", "(2E)-N-hydroxybut-2-enimidic acid"),
    ("C/C=C\\C(=NO)S", "(2Z)-N-hydroxybut-2-enimidothioic acid"),
    ("ON=C(O)[C@H](C)CC", "(2R)-N-hydroxy-2-methylbutanimidic acid"),
    ("ON=C(S)[C@@H](N)C", "(2S)-2-amino-N-hydroxypropanimidothioic acid"),
    ("O/N=C(/S)[C@H](N)C", "(1E,2R)-2-amino-N-hydroxypropanimidothioic acid"),
]

RING_ROWS = [
    ("ON=C([SeH])C1CCC(C)CC1", "N-hydroxy-4-methylcyclohexane-1-carboximidoselenoic acid"),
    ("ON=C(O)C1CCC(C)CC1", "N-hydroxy-4-methylcyclohexane-1-carboximidic acid"),
    ("ON=C(O)C1CCCCC1Cl", "2-chloro-N-hydroxycyclohexane-1-carboximidic acid"),
    ("ON=C(S)C1(C)CCCCC1", "N-hydroxy-1-methylcyclohexane-1-carboximidothioic acid"),
    ("ON=C(O)C1CCC(O)CC1", "N,4-dihydroxycyclohexane-1-carboximidic acid"),
    # benzene: the suffix keeps its locant beside a ring prefix, as it does for '4-methylbenzene-1-
    # carboximidic acid'; the italic N is cited with the ring's prefixes
    ("ON=C(O)c1ccc(C)cc1", "N-hydroxy-4-methylbenzene-1-carboximidic acid"),
    ("ON=C(S)c1ccc(Cl)cc1", "4-chloro-N-hydroxybenzene-1-carboximidothioic acid"),
    ("ON=C(O)c1ccc(OC)cc1", "N-hydroxy-4-methoxybenzene-1-carboximidic acid"),
    ("Oc1ccccc1C(O)=NO", "N,2-dihydroxybenzene-1-carboximidic acid"),
    ("Nc1ccc(C(O)=NO)cc1", "4-amino-N-hydroxybenzene-1-carboximidic acid"),
    ("ON=C([SeH])c1ccccc1", "N-hydroxybenzenecarboximidoselenoic acid"),
    ("ON=C([TeH])c1ccccc1", "N-hydroxybenzenecarboximidotelluroic acid"),
    # the N-hydroxy imidic acid is senior to the sulfonic acid class 7a,:18172: each acid is
    # 'followed in turn by the corresponding peroxy, imidic, and hydrazonic acids', then the next acid)
    ("ON=C(O)c1ccc(S(=O)(=O)O)cc1", "N-hydroxy-4-sulfobenzene-1-carboximidic acid"),
]


@pytest.mark.parametrize("smiles,pin", BB_ROWS)
def test_blue_book_hydroximic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# the neighbouring N-hydroxy classes, which this change does not touch: the hydroxamic acids are named as
# N-hydroxy amides,:30144), the hydroximic acids of the sulfonic and sulfinic acids as
# N-hydroxy sulfonimidic acids,:31230)
BB_CONTROL_ROWS = [
    ("ONC(C)=O", "N-hydroxyacetamide"),                                       #:30148
    ("ONC(CC)=O", "N-hydroxypropanamide"),                                    #:32825
    ("ONC(=O)C1CCCCC1", "N-hydroxycyclohexanecarboxamide"),                   #:30150
    ("CS(=O)(=O)NO", "N-hydroxymethanesulfonamide"),                          #:38338
    ("CS(=O)(=NO)O", "N-hydroxymethanesulfonimidic acid"),                    #:31234
]


@pytest.mark.parametrize("smiles,pin", BB_CONTROL_ROWS)
def test_the_neighbouring_n_hydroxy_classes_keep_their_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CHAIN_ROWS)
def test_substituted_chain_hydroximic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _isomeric_read_back(name, smiles):
    """True iff OPSIN reads ``name`` to the input's ISOMERIC canonical SMILES.

    The full InChIKey the other read-backs compare does not see the configuration of the C=N of a
    hydroximic acid: the mobile hydrogen of -C(=N-OH)-XH puts the group in one tautomer-neutral
    layer, so 'C/C(O)=N/O' and 'C/C(O)=N\\O' share the key an InChIKey. A descriptor of
    that bond is therefore proved here, against OPSIN's own reading of the name."""
    from tests.support.rt_assert import _canon, _independent_parse
    read = _independent_parse(name, smiles)
    return bool(read) and _canon(read) == _canon(smiles)


@pytest.mark.parametrize("smiles,pin", STEREO_ROWS)
def test_stereo_hydroximic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert _isomeric_read_back(pin, smiles), (smiles, pin)
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res["stereo_unexpressed"] is False, res


def test_the_isomeric_read_back_sees_the_configuration_the_key_does_not():
    """The helper discriminates: the same name with the other descriptor reads back to the same
    InChIKey and to a different isomeric SMILES (a known positive and a known negative)."""
    smiles = "C/C(O)=N/O"
    assert name_is_rt_exact("(1E)-N-hydroxyethanimidic acid", smiles)       # the key is blind to it
    assert name_is_rt_exact("(1Z)-N-hydroxyethanimidic acid", smiles)
    assert _isomeric_read_back("(1Z)-N-hydroxyethanimidic acid", smiles)
    assert not _isomeric_read_back("(1E)-N-hydroxyethanimidic acid", smiles)


# A ring parent has no locant of its own for the carbon of the suffix, so the configuration of the C=N is the
# descriptor at the front of the name. "NAMING OF STEREOISOMERS" (the Blue Book, the paragraph at:44643):
# "In preferred IUPAC names, stereodescriptors are placed immediately at the front of the part of the name to which
# they relate. They are placed at the front of the complete name when related to the parent structure... When they
# relate to substituent groups, they are cited at the front of the corresponding prefix. They are preceded by a
# numerical or letter locant... when such locants are present". "Stereogenic units" (:44718): "In preferred
# IUPAC names, all stereogenic units must be specified, unless an omission is allowed according to ", which
# "Omission of stereodescriptors" (:44637) allows for the fixed double bonds of three- to seven-membered rings, not for
# a C=N outside the ring. The book prints the pattern for a methanimine, '(Z)-N-hydroxy(4-chlorophenyl)(phenyl)
# methanimine (PIN)' (:47666), and for a group of a ring parent, '1-[(Z)-ethylidene]-3-[(E)-ethylidene]cyclopentane'
# (:48292). It prints no imidic acid of a ring parent with a configured C=N (ASSUMED by analogy: the descriptor is cited
# as for these two). Where the numbering leaves a choice between two prefixes of one name (the 3,5 rows) Z takes the
# lower locant: "NUMBERING" (j) (:3346), "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r
#... that are preferred to E, S, P, and s". The C=N is read as the OH of N against the group of C: CIP puts the
# chalcogen of C above the ring carbon.
RING_STEREO_ROWS = [
    ("O/N=C(\\O)c1ccccc1", "(Z)-N-hydroxybenzenecarboximidic acid"),
    ("O/N=C(/O)c1ccccc1", "(E)-N-hydroxybenzenecarboximidic acid"),
    ("O/N=C(\\O)C1CCCCC1", "(Z)-N-hydroxycyclohexanecarboximidic acid"),
    ("O/N=C(/O)C1CCCCC1", "(E)-N-hydroxycyclohexanecarboximidic acid"),
    ("O/N=C(/S)C1CCCCC1", "(E)-N-hydroxycyclohexanecarboximidothioic acid"),
    ("O/N=C(\\[SeH])c1ccccc1", "(Z)-N-hydroxybenzenecarboximidoselenoic acid"),
    ("O/N=C(/O)c1ccc(C)cc1", "(E)-N-hydroxy-4-methylbenzene-1-carboximidic acid"),
    ("O/N=C(\\S)C1CCC(C)CC1", "(Z)-N-hydroxy-4-methylcyclohexane-1-carboximidothioic acid"),
    ("O/N=C(/O)c1ccc(cc1)[C@H](C)Cl", "(E)-4-[(1S)-1-chloroethyl]-N-hydroxybenzene-1-carboximidic acid"),
    # the group demoted by a senior one (1),:30131): the descriptor is that of the prefix
    ("O/N=C(/O)c1ccc(C(=O)O)cc1", "4-[(E)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("O/N=C(\\O)c1ccc(C(=O)O)cc1", "4-[(Z)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("O/N=C(/O)c1ccc(C(=O)O)cc1C", "4-[(E)-C,N-dihydroxycarbonimidoyl]-3-methylbenzoic acid"),
    ("O/N=C(\\O)C1CCC(C(=O)O)CC1", "4-[(Z)-C,N-dihydroxycarbonimidoyl]cyclohexane-1-carboxylic acid"),
    # two groups: each prefix carries its own descriptor; two of one configuration are one multiplied prefix
    # (b),:7067), two of different configurations are two prefixes, cited in alphanumerical order
    # with the descriptors ignored for it,:3436), the lower locant first
    ("OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N\\O",
     "2-[(E)-C,N-dihydroxycarbonimidoyl]-5-[(Z)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N/O", "2,5-bis[(Z)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("OC(=O)c1cc(/C(O)=N\\O)ccc1/C(O)=N\\O", "2,5-bis[(E)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("OC(=O)c1cc(/C(O)=N/O)cc(/C(O)=N\\O)c1",
     "3-[(Z)-C,N-dihydroxycarbonimidoyl]-5-[(E)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("OC(=O)c1cc(/C(O)=N/O)cc(/C(O)=N/O)c1", "3,5-bis[(Z)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("OC(=O)c1ccc(/C(O)=N\\O)cc1/C(O)=N/O",
     "2-[(Z)-C,N-dihydroxycarbonimidoyl]-4-[(E)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
    ("OC(=O)c1ccc(/C(O)=N/O)cc1/C(O)=N/O", "2,4-bis[(Z)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
]


@pytest.mark.parametrize("smiles,pin", RING_STEREO_ROWS)
def test_a_ring_parent_states_the_cn_configuration(smiles, pin):
    """The two readings of the ring row 'O/N=C(\\O)c1ccccc1' and 'O/N=C(/O)c1ccccc1' share the full key (the
    mobile hydrogen of -C(=N-OH)-OH), so the descriptor is proved against OPSIN's isomeric reading of the name."""
    assert_pin_at_both_tiers(smiles, pin)
    assert _isomeric_read_back(pin, smiles), (smiles, pin)
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res["stereo_unexpressed"] is False, res


@pytest.mark.parametrize("smiles", [
    "O/N=C(/O)[C@H]1CCC[C@H](C)C1",      # a ring descriptor of the parent beside the C=N: not built together
    "O/N=C(/O)[C@H]1CC[C@H](C)CC1",
])
def test_a_ring_descriptor_beside_a_configured_cn_is_not_labelled_a_pin(smiles):
    """The name that states the stereocentres and leaves the C=N open would be the constitution's name, not the
    input's: no PIN label at the default tier, and the best-effort name keeps every descriptor (it is read back
    to the isomeric SMILES of the input) below PIN."""
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert b.get("tier") in ("systematic_verified", "best_effort") and b.get("name"), b
    assert _isomeric_read_back(b["name"], smiles), (smiles, b)


@pytest.mark.parametrize("smiles,pin", [
    # the input leaves the C=N open: the name states no descriptor either, and the row is not flagged as unexpressed
    ("ON=C(O)c1ccc(C(=O)O)cc1", "4-(C,N-dihydroxycarbonimidoyl)benzoic acid"),
    ("ON=C(O)c1ccccc1", "N-hydroxybenzenecarboximidic acid"),
    ("ON=C(S)C1CCCCC1", "N-hydroxycyclohexanecarboximidothioic acid"),
])
def test_an_open_cn_is_left_open(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res["stereo_unexpressed"] is False, res


@pytest.mark.parametrize("smiles,pin", RING_ROWS)
def test_ring_hydroximic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_indole_row_is_the_pin_not_a_thiol():
    """The row the milestone census shipped as '(1E)-1-(hydroxyimino)-2-(1H-indol-3-yl)ethane-1-thiol': the
    acid outranks the thiol class 7a before 18,:18162,:18190)."""
    smiles = "O/N=C(/S)Cc1c[nH]c2ccccc12"
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res["name"] == "(1E)-N-hydroxy-2-(1H-indol-3-yl)ethanimidothioic acid", res
        assert res["tier"] == "pin_verified" and not res["name"].endswith("thiol"), res


# The boundary: what is not built keeps a name that reads back and is not labelled a PIN, or none at the
# default tier. None of these may be a name that omits or relocates a group.
BOUNDARY_ROWS = [
    # two hydroximic groups: their N locants carry the parent's numbers (:30112,:30125)
    "ON=C(O)CCC(O)=NO",                 # 'N1,N4-dihydroxybutanediimidic acid (PIN)',:30112
    "ON=C(O)CCCC(O)=NO",
    "ON=C(S)CCCC(S)=NO",
    "ON=C(O)c1ccc(C(O)=NO)cc1",
    # a hydroximic acid and a plain imidic acid on one ring: a dicarboximidic acid with one N-hydroxy
    "ON=C(O)c1ccc(C(=N)O)cc1",
    # the S, Se and Te analogues have no prefix form in the book: nothing to cite them demoted
    "OC(=O)c1ccc(C(S)=NO)cc1",
    # the ring is not benzene or a saturated carbocycle: 'N2-hydroxy-1H-pyrrole-2-carboximidic acid (PIN)'
    # (:30121) numbers the N with its ring position, which is not built
    "ON=C(O)c1ccc[nH]1",
    "ON=C(O)c1cccnc1",
    "ON=C(S)c1ccc2ccccc2c1",
]


@pytest.mark.parametrize("smiles", BOUNDARY_ROWS)
def test_unbuilt_hydroximic_acid_is_not_labelled_a_pin(smiles):
    assert_declined_at_default(smiles)


def test_no_spelling_defect_over_the_panel():
    """Over every row above and the boundary: a name is the structure (OPSIN, full InChIKey) and cites
    'N-hydroxy' as a prefix, with its hyphen ('N-hydroxy2-methyl' is not a name; the order of the prefixes,
    ,:3440, is pinned by the rows themselves)."""
    rows = [s for s, _ in BB_ROWS + CHAIN_ROWS + STEREO_ROWS + RING_ROWS] + BOUNDARY_ROWS
    for smiles in rows:
        for res in (name_default(smiles), name_breadth(smiles)):
            name = res.get("name")
            if not name or res.get("tier") == "abstain":
                continue
            assert name_is_rt_exact(name, smiles), (smiles, res)
            assert not re.search(r"N-hydroxy[0-9(\[]", name), (smiles, name)


@pytest.mark.parametrize("smiles,group", [
    ("ON=C(C)O", "imidic_acid"),
    ("ON=C(C)S", "imidothioic_acid"),
    ("ON=C(C)[SeH]", "imidoselenoic_acid"),
    ("ON=C(C)[TeH]", "imidotelluroic_acid"),
])
def test_a_hydroximic_acid_is_written_with_the_suffix_of_its_imidic_acid(smiles, group):
    """ (:30096): the N-hydroxy derivative of the imidic acid of the same acid chalcogen."""
    from orthonym.perception.functional_groups import detect_functional_groups
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol)["hydroximic_acid"]
    assert suffix_group(mol, "hydroximic_acid", matches) == group
    assert suffix_group(mol, "imidic_acid", matches) == "imidic_acid"      # every other group is its own


# The rank of the group. class 7a (the Blue Book): the acids "each followed in turn by the
# corresponding peroxy, imidic, and hydrazonic acids. Chalcogen analogues follow each of the corresponding oxygen
# acids": the N-hydroxy derivative of an imidic acid has its rank, so it is senior to the hydrazonic acids and to
# the imidothioic acids, and of one rank with the plain imidic acid. (:30129): the prefixes are for
# "another group... that has seniority for citation as principal group": a plain imidic acid is none, so two
# imidic acids are a dicarboximidic acid with an N-hydroxy on one ('N1,N'4-dihydroxybenzene-1,4-dicarboximidic
# acid (PIN)',:30125), which is not built. Before: the group ranked BELOW the hydrazonic acids and was demoted
# to 'C,N-dihydroxycarbonimidoyl' or to 'hydroxy' and 'hydroxyimino' under any acid, one of its own rank included.
RANK_PIN_ROWS = [
    ("ON=C(O)CCC(=NN)O", "4-hydrazinylidene-N,4-dihydroxybutanimidic acid"),
    ("ON=C(O)CCC(=N)S", "N-hydroxy-4-imino-4-sulfanylbutanimidic acid"),
    ("ON=C(S)CCC(=NN)O", "4-hydrazinylidene-N,4-dihydroxybutanimidothioic acid"),
    # junior: the imidic acid of the O acid outranks the S analogue, so the S-hydroximic acid is demoted
    ("ON=C(S)CCC(=N)O", "4-(hydroxyimino)-4-sulfanylbutanimidic acid"),
    ("ON=C(O)CCC(N)=O", "4-amino-N-hydroxy-4-oxobutanimidic acid"),
]
RANK_DECLINED_ROWS = [
    # one rank: ring parents
    "ON=C(O)C1CCC(CC1)C(=N)O",
    "ON=C(O)C1CCCC1C(=N)O",
    # junior to the N-hydroxy imidic acid, whose demoted prefix (the ring) is not built
    "ON=C(O)C1CCC(CC1)C(=NN)O",
    "ON=C(O)C1CCC(CC1)C(=N)S",
    # one rank: chain ends
    "ON=C(O)CCC(=N)O",
    "ON=C(S)CCC(=N)S",
]


@pytest.mark.parametrize("smiles,pin", RANK_PIN_ROWS)
def test_the_group_ranks_as_its_imidic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", RANK_DECLINED_ROWS)
def test_two_acids_of_one_rank_are_not_a_demotion(smiles):
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles,group", [
    ("ON=C(O)CCC(=NN)O", "hydroximic_acid"),     # above the hydrazonic acid
    ("ON=C(S)CCC(=NN)O", "hydroximic_acid"),     # the imidothioic acid is above the hydrazonic acid too
    ("ON=C(O)CCC(=N)S", "hydroximic_acid"),      # above the imidothioic acid
    ("ON=C(O)CCC(=N)O", "imidic_acid"),          # one rank: the plain acid is the principal group
    ("ON=C(S)CCC(=N)O", "imidic_acid"),          # O above S
    ("ON=C(S)CCC(=N)S", "imidothioic_acid"),
    ("ON=C(O)CCC(=O)O", "carboxylic_acid"),
])
def test_the_principal_group_of_a_hydroximic_acid_is_by_its_imidic_rank(smiles, group):
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles(smiles)
    assert get_principal_group(mol, detect_functional_groups(mol))[0] == group


def test_a_hydroximic_acid_is_junior_only_to_a_senior_group():
    from orthonym.perception.functional_groups import detect_functional_groups
    mol = Chem.MolFromSmiles("ON=C(O)CCC(=O)O")
    match = detect_functional_groups(mol)["hydroximic_acid"][0]
    assert junior_to("carboxylic_acid", mol, match)
    assert not junior_to("imidic_acid", mol, match)          # one rank
    assert not junior_to("hydrazonic_acid", mol, match)      # junior to it, not it to that
    assert not junior_to(None, mol, match)


# The configuration of the C=N is in no layer of the full key, so no key comparison sees a descriptor that is
# flipped; the verdict reads the configurations of the group's C=N bonds back from OPSIN's reading of the name.
@pytest.mark.parametrize("smiles,name,verdict", [
    ("C/C(O)=N/O", "(1Z)-N-hydroxyethanimidic acid", "ok"),
    ("C/C(O)=N/O", "(1E)-N-hydroxyethanimidic acid", "mismatch"),      # flipped
    ("C/C(O)=N\\O", "(1E)-N-hydroxyethanimidic acid", "ok"),
    ("CC(O)=NO", "(1E)-N-hydroxyethanimidic acid", "mismatch"),         # states what the input leaves open
    # CHANGED ok -> mismatch. Source: "Stereogenic units" (the Blue Book): "In preferred IUPAC
    # names, all stereogenic units must be specified, unless an omission is allowed according to ", whose
    # omissions (:44637) are the fixed double bonds of three- to seven-membered rings; a C=N outside a ring is none.
    # Independent check: OPSIN's isomeric reading of the name differs from the input
    # (``test_a_name_without_the_descriptor_does_not_read_back_to_a_configured_input``); mutation: the old verdict
    # returned 'ok' for this row and for the swap rows below. Would make it wrong: a Blue Book licence to omit the
    # configuration of a hydroximic C=N (none found: prints only '(Z)-N-hydroxy...methanimine',:47666).
    ("C/C(O)=N/O", "N-hydroxyethanimidic acid", "mismatch"),
    ("CC(O)=NO", "N-hydroxyethanimidic acid", "ok"),
    ("O/N=C(/S)CCc1ccccc1", "(1E)-N-hydroxy-3-phenylpropanimidothioic acid", "ok"),
    ("O/N=C(/S)CCc1ccccc1", "(1Z)-N-hydroxy-3-phenylpropanimidothioic acid", "mismatch"),
])
def test_the_cn_configuration_verdict_reads_the_descriptor_back(smiles, name, verdict):
    from tests.support.rt_assert import _independent_parse
    assert cn_configuration_verdict(smiles, _independent_parse(name, smiles)) == verdict


# The verdict compares bond by bond. A configured C=N of the input "Stereogenic units", the Blue Book:
# "all stereogenic units must be specified") that the reading of the name does not configure is a name for another
# molecule, as is a swap of two groups' configurations (a comparison of the sorted lists passes it).
@pytest.mark.parametrize("smiles,reading,verdict", [
    ("O/N=C(/O)c1ccc(C(=O)O)cc1", "ON=C(O)c1ccc(C(=O)O)cc1", "mismatch"),          # the name states none
    ("O/N=C(/O)c1ccc(C(=O)O)cc1", "O/N=C(/O)c1ccc(C(=O)O)cc1", "ok"),
    ("O/N=C(/O)c1ccc(C(=O)O)cc1", "O/N=C(\\O)c1ccc(C(=O)O)cc1", "mismatch"),        # the other configuration
    ("ON=C(O)c1ccc(C(=O)O)cc1", "ON=C(O)c1ccc(C(=O)O)cc1", "ok"),                  # the input leaves it open
    ("ON=C(O)c1ccc(C(=O)O)cc1", "O/N=C(/O)c1ccc(C(=O)O)cc1", "mismatch"),          # the name states what it leaves open
    # two groups of different places: a swap between them is not the input
    ("OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N\\O", "OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N\\O", "ok"),
    ("OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N\\O", "OC(=O)c1cc(/C(O)=N\\O)ccc1/C(O)=N/O", "mismatch"),
    # two groups of one place (equivalent by the ring's symmetry): the same molecule, either way round
    ("OC(=O)c1cc(/C(O)=N/O)cc(/C(O)=N\\O)c1", "OC(=O)c1cc(/C(O)=N\\O)cc(/C(O)=N/O)c1", "ok"),
    ("OC(=O)c1cc(/C(O)=N/O)cc(/C(O)=N/O)c1", "OC(=O)c1cc(/C(O)=N/O)cc(/C(O)=N\\O)c1", "mismatch"),
    # a group that is not hydroximic is not the verdict's business
    ("C/C=C/C(=O)O", "CC=CC(=O)O", "ok"),
])
def test_the_cn_configuration_verdict_compares_bond_by_bond(smiles, reading, verdict):
    assert cn_configuration_verdict(smiles, reading) == verdict


@pytest.mark.parametrize("smiles,name", [
    ("C/C(O)=N/O", "N-hydroxyethanimidic acid"),
    ("O/N=C(/O)c1ccc(C(=O)O)cc1", "4-(C,N-dihydroxycarbonimidoyl)benzoic acid"),
    ("OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N\\O", "2,5-bis[(Z)-C,N-dihydroxycarbonimidoyl]benzoic acid"),
])
def test_a_name_without_the_descriptor_does_not_read_back_to_a_configured_input(smiles, name):
    """The independent check of the verdict's rows above: the full key reads the name back to the input (it is
    blind to the C=N), OPSIN's isomeric reading of it is not the input's."""
    assert name_is_rt_exact(name, smiles)
    assert not _isomeric_read_back(name, smiles)


@pytest.mark.parametrize("smiles", [
    "O/N=C(/O)c1ccc(C(=O)O)cc1",
    "O/N=C(/O)c1ccccc1",
    "O/N=C(/O)C1CCCCC1",
    "OC(=O)c1cc(/C(O)=N/O)ccc1/C(O)=N\\O",
])
def test_a_name_that_states_no_configuration_of_a_configured_cn_does_not_ship(smiles, monkeypatch):
    """Mutation: the producers drop the descriptor again (no configuration is read off the input, as before
    the descriptor was built). The key is blind to it, so only the verdict stops the name: the default tier declines it, and with
    the verdict out of the way the descriptorless name ships as a PIN (the test can fail)."""
    import orthonym.rules.hydroximic_acids as hx
    monkeypatch.setattr(hx, "cn_labels", lambda mol: {})
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res.get("tier") != "pin_verified", res
    monkeypatch.setattr(hx, "cn_configuration_verdict", lambda *args, **kwargs: "ok")
    # the key-blind double-bond check reads the same C=N (validation/key_blind_double_bonds.py): off too
    import orthonym.validation.key_blind_double_bonds as kb
    monkeypatch.setattr(kb, "key_blind_double_bond_verdict", lambda *args, **kwargs: "n/a")
    res = name_default(smiles)
    assert res.get("tier") == "pin_verified" and "(E)" not in res["name"] and "(Z)" not in res["name"], res


# "General methodology" (c) (the Blue Book): "any component which is substituted automatically requires
# use of the multiplicative forms 'bis', 'tris', etc."; "Numerical terms for compound or complex features"
# (:2826): "Multiplicative prefixes for compound or complex entities, such as substituted substituents... 'bis' and
# 'tris' correspond to 'di' and 'tri'". 'C,N-dihydroxycarbonimidoyl' is a substituted substituent prefix, so it is
# multiplied by 'bis', not 'di'.
@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1cc(C(=N)O)cc(C(=N)O)c1", "3,5-bis(C-hydroxycarbonimidoyl)benzoic acid"),
    ("OC(=O)c1cc(C(O)=NO)cc(C(O)=NO)c1", "3,5-bis(C,N-dihydroxycarbonimidoyl)benzoic acid"),
    ("OC(=O)C1CC(C(=NO)O)CC(C(=NO)O)C1", "3,5-bis(C,N-dihydroxycarbonimidoyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CC(C(=N)O)CC(C(=N)O)C1", "3,5-bis(C-hydroxycarbonimidoyl)cyclohexane-1-carboxylic acid"),
])
def test_a_multiplied_carbonimidoyl_prefix_is_bis(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert "di(" not in pin


# one rank, three groups: the two plain imidic acids are the principal groups and the N-hydroxy is theirs ('N1,N'4-
# dihydroxybenzene-1,4-dicarboximidic acid (PIN)',:30125), not a 'C,N-dihydroxycarbonimidoyl' prefix of a
# dicarboximidic acid: not built, so no PIN label
@pytest.mark.parametrize("smiles", [
    "ON=C(O)c1cc(C(=N)O)cc(C(=N)O)c1",
    "ON=C(O)C1CC(C(=N)O)CC(C(=N)O)C1",
    "ON=C(O)c1cc(C(=N)S)cc(C(=N)S)c1",
])
def test_two_imidic_acids_beside_a_hydroximic_acid_are_not_a_demotion(smiles):
    assert_declined_at_default(smiles)


# the best-effort tier keeps a name for every row the default tier declines: demoted, never deleted
@pytest.mark.parametrize("smiles", RANK_DECLINED_ROWS + [
    "ON=C(O)CCCC(=N)O",
    "ON=C(O)c1ccc(C(=N)O)cc1",
])
def test_a_declined_row_keeps_a_systematic_best_effort_name(smiles):
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert b.get("tier") in ("systematic_verified", "pin_unverified") and name_is_rt_exact(b["name"], smiles), b


# Every reader of the order ranks the group by the imidic acid of its acid chalcogen class 7a,
#:18172: "Chalcogen analogues follow each of the corresponding oxygen acids")
@pytest.mark.parametrize("smiles,rank_name", [
    ("ON=C(C)O", "imidic_acid"),
    ("ON=C(C)S", "imidothioic_acid"),
    ("ON=C(C)[SeH]", "imidoselenoic_acid"),
    ("ON=C(C)[TeH]", "imidotelluroic_acid"),
])
def test_the_one_reader_of_the_order_ranks_a_hydroximic_acid_by_its_chalcogen(smiles, rank_name):
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import SENIORITY_ORDER, seniority_index
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol)["hydroximic_acid"]
    assert seniority_index("hydroximic_acid", mol, matches) == SENIORITY_ORDER.index(rank_name)
    assert seniority_index(rank_name) == SENIORITY_ORDER.index(rank_name)          # every other name is its own
    # without the molecule the group is the O acid's, as it was before
    assert seniority_index("hydroximic_acid") == SENIORITY_ORDER.index("imidic_acid")


def test_the_fragment_and_adduct_orders_rank_an_s_hydroximic_acid_below_an_o_imidic_acid():
    from orthonym.decomposition.fragment_ranker import _score_fragment_seniority_impl
    from orthonym.rules.adducts import component_sort_key
    s_hydroximic, o_imidic = "CC(S)=NO", "CC(=N)O"
    # the S acid ranks as the imidothioic acid, after the O imidic acid
    assert _score_fragment_seniority_impl(o_imidic)[1] < _score_fragment_seniority_impl(s_hydroximic)[1]
    assert component_sort_key(o_imidic)[1] < component_sort_key(s_hydroximic)[1]
    # the O hydroximic acid shares the imidic acid's rank (a place after it in the order, before the thio acid)
    assert component_sort_key("CC(O)=NO")[1] < component_sort_key(s_hydroximic)[1]


def test_a_flipped_cn_descriptor_does_not_ship(monkeypatch):
    """Mutation: the producer's descriptor is flipped; the key is blind to it, the ship-time verdict is not."""
    import dataclasses
    from orthonym.assembly.handlers import _handler_shared as hs
    smiles, right = "C/C(S)=N\\O", "(1E)-N-hydroxyethanimidothioic acid"
    assert name_default(smiles)["name"] == right
    original = hs._generate_stereodescriptors

    def flipped(features, *args, **kwargs):
        fragment = original(features, *args, **kwargs)
        if fragment is not None:
            fragment = dataclasses.replace(
                fragment, text=fragment.text.translate(str.maketrans("EZ", "ZE")))
        return fragment

    monkeypatch.setattr(hs, "_generate_stereodescriptors", flipped)
    from orthonym import Orthonym
    res = Orthonym().name_tiered(smiles)
    assert res.get("name") != "(1Z)-N-hydroxyethanimidothioic acid", res
    # and without the verdict the flipped name ships: the key sees nothing (the test can fail)
    import orthonym.rules.hydroximic_acids as hx
    monkeypatch.setattr(hx, "cn_configuration_verdict", lambda *args, **kwargs: "ok")
    # the key-blind double-bond check reads the same C=N (validation/key_blind_double_bonds.py): off too
    import orthonym.validation.key_blind_double_bonds as kb
    monkeypatch.setattr(kb, "key_blind_double_bond_verdict", lambda *args, **kwargs: "n/a")
    assert Orthonym().name_tiered(smiles).get("name") == "(1Z)-N-hydroxyethanimidothioic acid"


def test_the_dedicated_handler_is_gone():
    """One home for the rule: the fragment assembly and the benzene namer build the group."""
    from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
    assert "hydroximic_acid" not in INNER_DISPATCH_TABLE
    assert Orthonym().name_tiered("ON=C(O)C")["name"] == "N-hydroxyethanimidic acid"
