""" (j) numbering and determinism at the PIN tier (fix a performance pass, wp2).

``### **** NUMBERING`` clause (j) (the Blue Book): "the lower locant is
assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry) that are
preferred to E, S, P, and s, respectively". The numbering keys used to rank the
descriptor CODES only and drop their locants, so a lone descriptor tied across
every numbering and the SMILES atom order decided: '(1r)-', '(3r)-' and
'(5r)-1,3,5-trimethylcyclohexane' for one molecule. Several numbering sites had
no (j) tier (cycloalkene, heteromonocycle, von Baeyer) or no (g) tier ("lowest
locants for the substituent cited first as a prefix",:3307), and the citation
order had no descriptor tier (:22606). The carbocycle (g) tier named
hydrocarbon prefixes only ('3-chloro-1-methylcyclohexane'). Every witness below is named in
six SMILES spellings (the input plus five RDKit random spellings that keep the
stereo); each spelling must give the one expected name.

Blue Book rows the key reproduces: '(2Z,4S,8R,9E)-undeca-2,9-diene-4,8-diol
(PIN)' (:3403, "the choice is between 'E' and 'Z' for position '2'"),
'(1Z,3E)-cyclododeca-1,3-diene (PIN)' (:3366), '1-[(2R)-butan-2-yl]-3-[(2S)-
butan-2-yl]benzene (PIN)' (:3390ff), '1-[(1R)-1-bromoethyl]-1-[(1S)-1-bromo-
ethyl]cyclopentane (PIN)' (:22609), hexachlorocyclohexane isomer 2
'(1R,2R,3r,4S,5S,6s)' (:48131). Isomers 3 and 4 (:48135,:48147) are NOT used:
no single numbering order reproduces both (see the wp2 report).

Names OPSIN 2.9.0 can parse were checked by an independent full-InChIKey round
trip; the lowercase r/s names (OPSIN parses none) by a stereo-stripped round
trip plus a check that each r/s locant lands on an input atom with that code.

 wp7 (verification panel NIT, re-derived): the single-descriptor targets
'(1r)-1,3,5-trimethylcyclohexane', '(1r)-cyclohexane-1,3,5-triol', '(1r)-1,3,5-
trichlorocyclohexane' and '(2r)-2,4,6-trimethyl-1,3,5-trioxane' are NOT all-cis C3v
molecules (the panel's premise, repeated here by an earlier wp7 edit). Each is the
cis,trans ('r,c,t') isomer, which has exactly ONE stereogenic centre: inverting either
of the two other substituted centres gives back the same compound (identical standard
InChI), because the ring can be turned over; only the centre on the mirror plane is
stereogenic, and it is pseudoasymmetric. "In preferred IUPAC names, stereodescriptors,
preceded by a locant, must be cited to specify each stereogenic unit",
the Blue Book), so one unit takes one descriptor, as C-5 of '(2R,3s,4S,6R)-2,6-
dichloro-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptan-4-ol (PIN)' takes none
("'C-5' is a nonstereogenic center",:45465/:45469). The all-cis isomer has three
stereogenic centres and cites all three, like '(2r,4r,6r,8r)-...-tetrabenzena-
cyclooctaphane-... (PIN)' (:49682): '(1s,3s,5s)-1,3,5-trimethylcyclohexane' (ALL_CIS
below). Both premises are asserted structurally (test_stereogenic_unit_count), and each
descriptor set picks out exactly one of the two stereoisomers (RDKit CIP labeller:
r,c,t -> ('r',), all-cis -> ('s','s','s')).
"""
import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import cip_locant_rank_key

pytestmark = pytest.mark.unit


def test_lone_descriptor_takes_the_lower_locant():
    assert cip_locant_rank_key([(1, "r")]) < cip_locant_rank_key([(3, "r")])
    assert cip_locant_rank_key([(2, "S")]) < cip_locant_rank_key([(4, "S")])


def test_first_point_of_difference_bb_undecadienediol():
    # the Blue Book: Z at position 2 decides, not R/S at position 4
    good = [(2, "Z"), (4, "S"), (8, "R"), (9, "E")]
    bad = [(2, "E"), (4, "R"), (8, "S"), (9, "Z")]
    assert cip_locant_rank_key(good) < cip_locant_rank_key(bad)


def test_chiral_before_pseudoasymmetric_on_an_exact_tie():
    # hexachlorocyclohexane isomer 2 (the Blue Book)
    isomer2 = list(zip(range(1, 7), ["R", "R", "r", "S", "S", "s"]))
    other = list(zip(range(1, 7), ["r", "R", "R", "s", "S", "S"]))
    assert cip_locant_rank_key(isomer2) < cip_locant_rank_key(other)


def test_no_descriptor_gives_an_empty_key():
    assert cip_locant_rank_key([]) == ()
    assert cip_locant_rank_key([(1, "?")]) == ()


RINGS = [
    # (1r)-1,3,5-trimethylcyclohexane
    ('C[C@H]1C[C@H](C)C[C@@H](C)C1', '(1r)-1,3,5-trimethylcyclohexane'),
    ('[C@H]1(C[C@@H](C[C@H](C1)C)C)C', '(1r)-1,3,5-trimethylcyclohexane'),
    ('C[C@H]1C[C@@H](C[C@H](C)C1)C', '(1r)-1,3,5-trimethylcyclohexane'),
    ('C1[C@H](C[C@@H](C[C@H]1C)C)C', '(1r)-1,3,5-trimethylcyclohexane'),
    ('[C@H]1(C[C@H](C[C@@H](C)C1)C)C', '(1r)-1,3,5-trimethylcyclohexane'),
    ('C[C@H]1C[C@H](C)C[C@H](C1)C', '(1r)-1,3,5-trimethylcyclohexane'),
    # (1r)-cyclohexane-1,3,5-triol
    ('O[C@H]1C[C@H](O)C[C@@H](O)C1', '(1r)-cyclohexane-1,3,5-triol'),
    ('[C@H]1(O)C[C@@H](C[C@H](C1)O)O', '(1r)-cyclohexane-1,3,5-triol'),
    ('O[C@H]1C[C@@H](C[C@H](O)C1)O', '(1r)-cyclohexane-1,3,5-triol'),
    ('C1[C@H](C[C@@H](C[C@H]1O)O)O', '(1r)-cyclohexane-1,3,5-triol'),
    ('[C@H]1(C[C@H](C[C@@H](O)C1)O)O', '(1r)-cyclohexane-1,3,5-triol'),
    ('O[C@H]1C[C@H](O)C[C@H](C1)O', '(1r)-cyclohexane-1,3,5-triol'),
    # (1r)-1,3,5-trichlorocyclohexane
    ('Cl[C@H]1C[C@H](Cl)C[C@@H](Cl)C1', '(1r)-1,3,5-trichlorocyclohexane'),
    ('[C@H]1(Cl)C[C@@H](C[C@H](C1)Cl)Cl', '(1r)-1,3,5-trichlorocyclohexane'),
    ('Cl[C@H]1C[C@@H](C[C@H](Cl)C1)Cl', '(1r)-1,3,5-trichlorocyclohexane'),
    ('C1[C@H](C[C@@H](C[C@H]1Cl)Cl)Cl', '(1r)-1,3,5-trichlorocyclohexane'),
    ('[C@H]1(C[C@H](C[C@@H](Cl)C1)Cl)Cl', '(1r)-1,3,5-trichlorocyclohexane'),
    ('Cl[C@H]1C[C@H](Cl)C[C@H](C1)Cl', '(1r)-1,3,5-trichlorocyclohexane'),
    # (1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane
    ('Cl[C@H]1[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H]1Cl', '(1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane'),
    ('Cl[C@H]1[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H]([C@H]1Cl)Cl', '(1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane'),
    ('Cl[C@H]1[C@H]([C@H]([C@@H](Cl)[C@H](Cl)[C@H]1Cl)Cl)Cl', '(1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane'),
    ('[C@H]1(Cl)[C@@H](Cl)[C@@H]([C@H]([C@@H](Cl)[C@H]1Cl)Cl)Cl', '(1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane'),
    ('Cl[C@H]1[C@H](Cl)[C@@H](Cl)[C@H]([C@@H](Cl)[C@H]1Cl)Cl', '(1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane'),
    ('[C@H]1(Cl)[C@H]([C@H]([C@@H]([C@H]([C@H]1Cl)Cl)Cl)Cl)Cl', '(1R,2R,3r,4S,5S,6s)-1,2,3,4,5,6-hexachlorocyclohexane'),
]

# wp7: the all-cis isomers of the four (1r)/(2r) rows above. Three stereogenic
# centres, so three descriptors the Blue Book; '(2r,4r,6r,8r)-...
# (PIN)':49682 cites each of its four equivalent centres). Verified outside the
# engine: OPSIN 2.9.0 parses the stereo-stripped name to the input constitution, and
# of the two stereoisomers only the all-cis one carries ('s','s','s') (RDKit CIP
# labeller); OPSIN parses no lowercase r/s name.
ALL_CIS = [
    # (1s,3s,5s)-1,3,5-trimethylcyclohexane
    ('C[C@H]1C[C@@H](C)C[C@@H](C)C1', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    ('[C@H]1(C[C@@H](C[C@@H](C1)C)C)C', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    ('C1[C@H](C[C@H](C[C@H]1C)C)C', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    ('[C@H]1(C[C@@H](C[C@H](C)C1)C)C', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    ('C[C@H]1C[C@H](C[C@H](C1)C)C', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    ('C1[C@H](C)C[C@@H](C[C@@H]1C)C', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    # (1s,3s,5s)-cyclohexane-1,3,5-triol
    ('O[C@H]1C[C@@H](O)C[C@@H](O)C1', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    ('O[C@H]1C[C@H](C[C@@H](O)C1)O', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    ('C1[C@H](C[C@@H](O)C[C@H]1O)O', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    ('[C@H]1(O)C[C@H](C[C@H](C1)O)O', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    ('C1[C@H](C[C@H](C[C@H]1O)O)O', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    ('C1[C@H](O)C[C@@H](C[C@@H]1O)O', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    # (1s,3s,5s)-1,3,5-trichlorocyclohexane
    ('Cl[C@H]1C[C@@H](Cl)C[C@@H](Cl)C1', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    ('C1[C@H](Cl)C[C@@H](C[C@@H]1Cl)Cl', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    ('[C@H]1(C[C@@H](C[C@H](Cl)C1)Cl)Cl', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    ('[C@H]1(C[C@@H](C[C@@H](C1)Cl)Cl)Cl', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    ('C1[C@H](C[C@@H](Cl)C[C@H]1Cl)Cl', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    ('[C@H]1(Cl)C[C@H](C[C@@H](Cl)C1)Cl', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    # (2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane
    ('C[C@H]1O[C@@H](C)O[C@@H](C)O1', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('C[C@H]1O[C@H](O[C@@H](C)O1)C', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('[C@H]1(O[C@@H](O[C@@H](O1)C)C)C', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('O1[C@H](O[C@H](O[C@H]1C)C)C', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('C[C@H]1O[C@H](O[C@H](O1)C)C', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('C[C@H]1O[C@@H](C)O[C@H](O1)C', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
]

CHAINS = [
    # (2S)-pentane-2,4-diol
    ('C(C[C@H](C)O)(C)O', '(2S)-pentane-2,4-diol'),
    ('[C@H](C)(CC(O)C)O', '(2S)-pentane-2,4-diol'),
    ('OC(C[C@H](C)O)C', '(2S)-pentane-2,4-diol'),
    ('CC(O)C[C@H](C)O', '(2S)-pentane-2,4-diol'),
    ('O[C@@H](C)CC(C)O', '(2S)-pentane-2,4-diol'),
    ('C[C@@H](CC(C)O)O', '(2S)-pentane-2,4-diol'),
    # (2S)-2,6-dichloroheptane
    ('CC(Cl)CCC[C@@H](Cl)C', '(2S)-2,6-dichloroheptane'),
    ('C(C(C)Cl)CC[C@H](C)Cl', '(2S)-2,6-dichloroheptane'),
    ('C(C[C@@H](Cl)C)CC(C)Cl', '(2S)-2,6-dichloroheptane'),
    ('C[C@H](Cl)CCCC(C)Cl', '(2S)-2,6-dichloroheptane'),
    ('C(C(C)Cl)CC[C@@H](Cl)C', '(2S)-2,6-dichloroheptane'),
    ('Cl[C@@H](C)CCCC(C)Cl', '(2S)-2,6-dichloroheptane'),
    # (2E)-octa-2,6-diene
    ('CC=CCC/C=C/C', '(2E)-octa-2,6-diene'),
    ('C(C/C=C/C)C=CC', '(2E)-octa-2,6-diene'),
    ('C/C=C/CCC=CC', '(2E)-octa-2,6-diene'),
    ('C(/C)=C\\CCC=CC', '(2E)-octa-2,6-diene'),
    ('C(=C\\C)/CCC=CC', '(2E)-octa-2,6-diene'),
    ('C(CC=CC)/C=C/C', '(2E)-octa-2,6-diene'),
    # (2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid
    ('C(C(CC(O)=O)Cl)(C(O)=O)[C@H](Cl)CC(=O)O', '(2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid'),
    ('O=C(O)C[C@H](C(C(CC(=O)O)Cl)C(=O)O)Cl', '(2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid'),
    ('OC(=O)C[C@H](C(C(O)=O)C(CC(O)=O)Cl)Cl', '(2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid'),
    ('O=C(O)C[C@H](C(C(O)=O)C(CC(O)=O)Cl)Cl', '(2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid'),
    ('C(O)(C[C@@H](Cl)C(C(CC(O)=O)Cl)C(=O)O)=O', '(2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid'),
    ('C([C@@H](Cl)C(C(CC(=O)O)Cl)C(O)=O)C(=O)O', '(2R)-2,4-dichloropentane-1,3,5-tricarboxylic acid'),
]

CYCLOALKENES = [
    # (1Z,3E)-cyclododeca-1,3-diene
    ('C/1=C/C=C/CCCCCCCC1', '(1Z,3E)-cyclododeca-1,3-diene'),
    ('C1/C=C/CCCCCCCC\\C=1', '(1Z,3E)-cyclododeca-1,3-diene'),
    ('C1CCC/C=C/C=C\\CCCC1', '(1Z,3E)-cyclododeca-1,3-diene'),
    ('C1/C=C\\CCCCCCCC/C=1', '(1Z,3E)-cyclododeca-1,3-diene'),
    ('C1CCCCC/C=C/C=C\\CC1', '(1Z,3E)-cyclododeca-1,3-diene'),
    ('C1/C=C/C=C\\CCCCCCC1', '(1Z,3E)-cyclododeca-1,3-diene'),
    # (4R,5S)-4,5-dimethylcyclohex-1-ene
    ('C1C[C@@H]([C@@H](CC=1)C)C', '(4R,5S)-4,5-dimethylcyclohex-1-ene'),
    ('[C@H]1(CC=CC[C@@H]1C)C', '(4R,5S)-4,5-dimethylcyclohex-1-ene'),
    ('C[C@H]1CC=CC[C@H]1C', '(4R,5S)-4,5-dimethylcyclohex-1-ene'),
    ('C[C@H]1[C@H](CC=CC1)C', '(4R,5S)-4,5-dimethylcyclohex-1-ene'),
    ('C1C[C@@H](C)[C@@H](C)CC=1', '(4R,5S)-4,5-dimethylcyclohex-1-ene'),
    ('C1[C@H]([C@H](CC=C1)C)C', '(4R,5S)-4,5-dimethylcyclohex-1-ene'),
]

DESCRIPTORS_IN_PREFIXES = [
    # 1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane
    ('CC[C@@H](C)C1CCCC(C1)[C@@H](C)CC', '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane'),
    ('C[C@H](CC)C1CC(CCC1)[C@@H](C)CC', '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane'),
    ('CC[C@@H](C1CCCC([C@@H](CC)C)C1)C', '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane'),
    ('C([C@@H](C1CC(CCC1)[C@@H](CC)C)C)C', '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane'),
    ('C[C@@H](CC)C1CC([C@@H](CC)C)CCC1', '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane'),
    ('[C@@H](C1CC(CCC1)[C@H](C)CC)(CC)C', '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]cyclohexane'),
    # 1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane
    ('C[C@@H](Cl)C1(CCCC1)[C@H](C)Cl', '1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane'),
    ('C1([C@H](C)Cl)([C@@H](C)Cl)CCCC1', '1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane'),
    ('C1([C@H](C)Cl)(CCCC1)[C@@H](C)Cl', '1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane'),
    ('C1CCC(C1)([C@H](Cl)C)[C@@H](Cl)C', '1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane'),
    ('C[C@H](Cl)C1([C@@H](C)Cl)CCCC1', '1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane'),
    ('[C@H](C1([C@@H](Cl)C)CCCC1)(C)Cl', '1-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclopentane'),
]

# (g) on carbocycles: the legacy position-1 heuristic named hydrocarbon prefixes
# only, so any heteroatom prefix (and '-ylidene') sorted last: '3-chloro-1-methyl-
# cyclohexane' and '4-chloro-1-fluorocyclohexane' shipped pin_verified.
CYCLOALKANE_FIRST_CITED = [
    # 1-chloro-3-methylcyclohexane
    ('ClC1CCCC(C)C1', '1-chloro-3-methylcyclohexane'),
    ('CC1CCCC(Cl)C1', '1-chloro-3-methylcyclohexane'),
    ('C1C(CCCC1Cl)C', '1-chloro-3-methylcyclohexane'),
    ('C1(CCCC(Cl)C1)C', '1-chloro-3-methylcyclohexane'),
    ('C1CCC(Cl)CC1C', '1-chloro-3-methylcyclohexane'),
    ('C1CC(CC(C1)Cl)C', '1-chloro-3-methylcyclohexane'),
    # 1-bromo-4-ethylcyclohexane
    ('BrC1CCC(CC)CC1', '1-bromo-4-ethylcyclohexane'),
    ('C1CC(CCC1CC)Br', '1-bromo-4-ethylcyclohexane'),
    ('C1(CCC(Br)CC1)CC', '1-bromo-4-ethylcyclohexane'),
    ('C1CC(CC)CCC1Br', '1-bromo-4-ethylcyclohexane'),
    ('C1CC(Br)CCC1CC', '1-bromo-4-ethylcyclohexane'),
    ('C1C(CC)CCC(C1)Br', '1-bromo-4-ethylcyclohexane'),
    # 1-chloro-4-fluorocyclohexane
    ('FC1CCC(Cl)CC1', '1-chloro-4-fluorocyclohexane'),
    ('C1CC(Cl)CCC1F', '1-chloro-4-fluorocyclohexane'),
    ('C1CC(CCC1F)Cl', '1-chloro-4-fluorocyclohexane'),
    ('C1C(CCC(F)C1)Cl', '1-chloro-4-fluorocyclohexane'),
    ('ClC1CCC(CC1)F', '1-chloro-4-fluorocyclohexane'),
    ('C1(Cl)CCC(CC1)F', '1-chloro-4-fluorocyclohexane'),
    # 1-bromo-3,5-dimethylcyclohexane
    ('CC1CC(C)CC(Br)C1', '1-bromo-3,5-dimethylcyclohexane'),
    ('C1C(C)CC(CC1Br)C', '1-bromo-3,5-dimethylcyclohexane'),
    ('CC1CC(CC(Br)C1)C', '1-bromo-3,5-dimethylcyclohexane'),
    ('C1C(CC(CC1C)C)Br', '1-bromo-3,5-dimethylcyclohexane'),
    ('C1(CC(CC(Br)C1)C)C', '1-bromo-3,5-dimethylcyclohexane'),
    ('BrC1CC(C)CC(C1)C', '1-bromo-3,5-dimethylcyclohexane'),
    # 1-chloro-2-methylcyclohexane
    ('ClC1CCCCC1C', '1-chloro-2-methylcyclohexane'),
    ('CC1C(CCCC1)Cl', '1-chloro-2-methylcyclohexane'),
    ('CC1CCCCC1Cl', '1-chloro-2-methylcyclohexane'),
    ('C1(C)CCCCC1Cl', '1-chloro-2-methylcyclohexane'),
    ('C1CCCC(C1C)Cl', '1-chloro-2-methylcyclohexane'),
    ('C1CCC(Cl)C(C1)C', '1-chloro-2-methylcyclohexane'),
    # 1-methoxy-3-methylcyclohexane
    ('COC1CCCC(C)C1', '1-methoxy-3-methylcyclohexane'),
    ('C1C(OC)CCCC1C', '1-methoxy-3-methylcyclohexane'),
    ('C1CC(CC(C1)C)OC', '1-methoxy-3-methylcyclohexane'),
    ('C1C(CCCC1OC)C', '1-methoxy-3-methylcyclohexane'),
    ('C1C(CC(CC1)C)OC', '1-methoxy-3-methylcyclohexane'),
    ('CC1CCCC(OC)C1', '1-methoxy-3-methylcyclohexane'),
    # 3-chloro-5-methylcyclohexan-1-ol
    ('OC1CC(Cl)CC(C)C1', '3-chloro-5-methylcyclohexan-1-ol'),
    ('C1(C)CC(CC(C1)O)Cl', '3-chloro-5-methylcyclohexan-1-ol'),
    ('ClC1CC(CC(C)C1)O', '3-chloro-5-methylcyclohexan-1-ol'),
    ('C1C(CC(CC1O)Cl)C', '3-chloro-5-methylcyclohexan-1-ol'),
    ('C1(CC(CC(C)C1)O)Cl', '3-chloro-5-methylcyclohexan-1-ol'),
    ('CC1CC(Cl)CC(C1)O', '3-chloro-5-methylcyclohexan-1-ol'),
    # (1s,4s)-1-chloro-4-methylcyclohexane
    ('C[C@H]1CC[C@@H](Cl)CC1', '(1s,4s)-1-chloro-4-methylcyclohexane'),
    ('C1C[C@H](Cl)CC[C@@H]1C', '(1s,4s)-1-chloro-4-methylcyclohexane'),
    ('C1C[C@H](CC[C@H]1C)Cl', '(1s,4s)-1-chloro-4-methylcyclohexane'),
    ('C1[C@H](CC[C@@H](C)C1)Cl', '(1s,4s)-1-chloro-4-methylcyclohexane'),
    ('Cl[C@H]1CC[C@H](CC1)C', '(1s,4s)-1-chloro-4-methylcyclohexane'),
    ('[C@H]1(Cl)CC[C@H](CC1)C', '(1s,4s)-1-chloro-4-methylcyclohexane'),
    # 1-chloro-4-methylidenecyclohexane
    ('ClC1CCC(=C)CC1', '1-chloro-4-methylidenecyclohexane'),
    ('C1C(CCC(C1)Cl)=C', '1-chloro-4-methylidenecyclohexane'),
    ('C1CC(CCC1Cl)=C', '1-chloro-4-methylidenecyclohexane'),
    ('C1C(CCC(Cl)C1)=C', '1-chloro-4-methylidenecyclohexane'),
    ('C=C1CCC(CC1)Cl', '1-chloro-4-methylidenecyclohexane'),
    ('C1(=C)CCC(CC1)Cl', '1-chloro-4-methylidenecyclohexane'),
    # 3-methyl-6-methylidenecyclohex-1-ene
    ('CC1C=CC(=C)CC1', '3-methyl-6-methylidenecyclohex-1-ene'),
    ('C1C(CCC(C=1)C)=C', '3-methyl-6-methylidenecyclohex-1-ene'),
    ('C1CC(C=CC1C)=C', '3-methyl-6-methylidenecyclohex-1-ene'),
    ('C1C(C=CC(C)C1)=C', '3-methyl-6-methylidenecyclohex-1-ene'),
    ('C=C1CCC(C=C1)C', '3-methyl-6-methylidenecyclohex-1-ene'),
    ('C1(=C)C=CC(CC1)C', '3-methyl-6-methylidenecyclohex-1-ene'),
    # (6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene
    ('C=C1C=C[C@H](C(C)C)CC1', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
    ('C([C@@H]1CCC(=C)C=C1)(C)C', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
    ('C1(=C)C=C[C@H](C(C)C)CC1', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
    ('C1C(CC[C@@H](C(C)C)C=1)=C', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
    ('C1=CC(CC[C@H]1C(C)C)=C', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
    ('[C@H]1(C(C)C)C=CC(CC1)=C', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
]

HETEROCYCLES = [
    # (3R,5S)-3,5-dimethylpiperidine
    ('C[C@@H]1CNC[C@H](C)C1', '(3R,5S)-3,5-dimethylpiperidine'),
    ('N1C[C@H](C)C[C@@H](C1)C', '(3R,5S)-3,5-dimethylpiperidine'),
    ('C1[C@H](CNC[C@H]1C)C', '(3R,5S)-3,5-dimethylpiperidine'),
    ('C[C@H]1CNC[C@@H](C)C1', '(3R,5S)-3,5-dimethylpiperidine'),
    ('[C@@H]1(CNC[C@@H](C)C1)C', '(3R,5S)-3,5-dimethylpiperidine'),
    ('C1NC[C@@H](C)C[C@H]1C', '(3R,5S)-3,5-dimethylpiperidine'),
    # (3R,5S)-3,5-dimethyloxane
    ('C[C@@H]1COC[C@H](C)C1', '(3R,5S)-3,5-dimethyloxane'),
    ('O1C[C@H](C)C[C@@H](C1)C', '(3R,5S)-3,5-dimethyloxane'),
    ('C1[C@H](COC[C@H]1C)C', '(3R,5S)-3,5-dimethyloxane'),
    ('C[C@H]1COC[C@@H](C)C1', '(3R,5S)-3,5-dimethyloxane'),
    ('[C@@H]1(COC[C@@H](C)C1)C', '(3R,5S)-3,5-dimethyloxane'),
    ('C1OC[C@@H](C)C[C@H]1C', '(3R,5S)-3,5-dimethyloxane'),
    # (2r)-2,4,6-trimethyl-1,3,5-trioxane
    ('C[C@H]1O[C@@H](C)O[C@H](C)O1', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('O1[C@H](C)O[C@@H](O[C@H]1C)C', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('C[C@H]1O[C@H](O[C@H](C)O1)C', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('O1[C@H](O[C@@H](O[C@@H]1C)C)C', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('[C@H]1(O[C@@H](O[C@@H](C)O1)C)C', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('C[C@H]1O[C@H](C)O[C@@H](O1)C', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    # 3-chloro-5-methylpiperidine
    ('CC1CNCC(Cl)C1', '3-chloro-5-methylpiperidine'),
    ('ClC1CC(CNC1)C', '3-chloro-5-methylpiperidine'),
    ('C1C(CNCC1C)Cl', '3-chloro-5-methylpiperidine'),
    ('ClC1CNCC(C)C1', '3-chloro-5-methylpiperidine'),
    ('C1(CNCC(C)C1)Cl', '3-chloro-5-methylpiperidine'),
    ('C1NCC(C)CC1Cl', '3-chloro-5-methylpiperidine'),
    # 2-chloro-4,6-dimethylpiperidine
    ('CC1CC(C)NC(Cl)C1', '2-chloro-4,6-dimethylpiperidine'),
    ('C1C(C)CC(NC1Cl)C', '2-chloro-4,6-dimethylpiperidine'),
    ('CC1CC(CC(Cl)N1)C', '2-chloro-4,6-dimethylpiperidine'),
    ('C1C(NC(CC1C)C)Cl', '2-chloro-4,6-dimethylpiperidine'),
    ('C1(CC(CC(Cl)N1)C)C', '2-chloro-4,6-dimethylpiperidine'),
    ('ClC1NC(C)CC(C1)C', '2-chloro-4,6-dimethylpiperidine'),
    # 3-chloro-1,5-dimethylpiperidine
    ('CC1CN(C)CC(Cl)C1', '3-chloro-1,5-dimethylpiperidine'),
    ('C1(Cl)CN(CC(C1)C)C', '3-chloro-1,5-dimethylpiperidine'),
    ('CN1CC(CC(Cl)C1)C', '3-chloro-1,5-dimethylpiperidine'),
    ('C1C(CN(CC1C)C)Cl', '3-chloro-1,5-dimethylpiperidine'),
    ('N1(CC(CC(Cl)C1)C)C', '3-chloro-1,5-dimethylpiperidine'),
    ('ClC1CN(C)CC(C1)C', '3-chloro-1,5-dimethylpiperidine'),
]

VON_BAEYER = [
    # (1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol
    ('O[C@@H]1[C@H]2CC[C@@H](C2)[C@@H]1O', '(1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol'),
    ('C1[C@@H]2CC[C@@H]1[C@@H](O)[C@H]2O', '(1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol'),
    ('C1C[C@H]2C[C@H]1[C@H](O)[C@@H]2O', '(1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol'),
    ('O[C@H]1[C@@H]2C[C@H](CC2)[C@H]1O', '(1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol'),
    ('C1[C@@H]2[C@@H](O)[C@@H](O)[C@H](C2)C1', '(1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol'),
    ('[C@@H]1([C@@H]2C[C@H](CC2)[C@H]1O)O', '(1S,2R,3S,4S)-bicyclo[2.2.1]heptane-2,3-diol'),
    # 2-chloro-3-methylbicyclo[2.2.1]heptane
    ('CC1C(Cl)C2CCC1C2', '2-chloro-3-methylbicyclo[2.2.1]heptane'),
    ('C1CC2CC1C(C)C2Cl', '2-chloro-3-methylbicyclo[2.2.1]heptane'),
    ('C12C(Cl)C(C)C(CC1)C2', '2-chloro-3-methylbicyclo[2.2.1]heptane'),
    ('C1C2CCC1C(C2C)Cl', '2-chloro-3-methylbicyclo[2.2.1]heptane'),
    ('ClC1C2CC(C1C)CC2', '2-chloro-3-methylbicyclo[2.2.1]heptane'),
    ('C12CCC(C(Cl)C1C)C2', '2-chloro-3-methylbicyclo[2.2.1]heptane'),
    # 2-chloro-6-methylbicyclo[2.2.1]heptane
    ('ClC1CC2CC1C(C)C2', '2-chloro-6-methylbicyclo[2.2.1]heptane'),
    ('CC1CC2CC1C(Cl)C2', '2-chloro-6-methylbicyclo[2.2.1]heptane'),
    ('C1C2CC(Cl)C1C(C2)C', '2-chloro-6-methylbicyclo[2.2.1]heptane'),
    ('C1C(C)C2C(CC1C2)Cl', '2-chloro-6-methylbicyclo[2.2.1]heptane'),
    ('C12CC(Cl)C(C(C2)C)C1', '2-chloro-6-methylbicyclo[2.2.1]heptane'),
    ('CC1C2C(Cl)CC(C1)C2', '2-chloro-6-methylbicyclo[2.2.1]heptane'),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", RINGS)
def test_ring_descriptor_takes_the_lower_locant(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ALL_CIS)
def test_all_cis_isomer_cites_every_stereogenic_centre(smiles, expected):
    assert name_compound(smiles) == expected


def _stereogenic_centres(smiles):
    """Centres whose inversion changes the compound (standard InChI), independent of
    the engine: a centre that can be inverted to the SAME compound is not a
    stereogenic unit and takes no descriptor, the Blue Book)."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    mol = Chem.MolFromSmiles(smiles)
    ref = inchi.MolToInchi(mol)
    out = 0
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        flipped = Chem.RWMol(mol)
        flipped.GetAtomWithIdx(atom.GetIdx()).InvertChirality()
        out += inchi.MolToInchi(flipped) != ref
    return out


@pytest.mark.parametrize("smiles,expected", [
    ('C[C@H]1C[C@H](C)C[C@@H](C)C1', '(1r)-1,3,5-trimethylcyclohexane'),
    ('O[C@H]1C[C@H](O)C[C@@H](O)C1', '(1r)-cyclohexane-1,3,5-triol'),
    ('Cl[C@H]1C[C@H](Cl)C[C@@H](Cl)C1', '(1r)-1,3,5-trichlorocyclohexane'),
    ('C[C@H]1O[C@H](C)O[C@@H](C)O1', '(2r)-2,4,6-trimethyl-1,3,5-trioxane'),
    ('C[C@H]1C[C@@H](C)C[C@@H](C)C1', '(1s,3s,5s)-1,3,5-trimethylcyclohexane'),
    ('O[C@H]1C[C@@H](O)C[C@@H](O)C1', '(1s,3s,5s)-cyclohexane-1,3,5-triol'),
    ('Cl[C@H]1C[C@@H](Cl)C[C@@H](Cl)C1', '(1s,3s,5s)-1,3,5-trichlorocyclohexane'),
    ('C[C@H]1O[C@@H](C)O[C@@H](C)O1', '(2s,4s,6s)-2,4,6-trimethyl-1,3,5-trioxane'),
])
def test_stereogenic_unit_count(smiles, expected):
    """One descriptor per stereogenic unit: the r,c,t rows have one, the all-cis
    rows three (wp7; the verification panel read the r,c,t rows as all-cis)."""
    descriptors = expected[1:expected.index(')')].split(',')
    assert _stereogenic_centres(smiles) == len(descriptors), (smiles, expected)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", CHAINS)
def test_chain_descriptor_takes_the_lower_locant(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", CYCLOALKENES)
def test_cycloalkene_descriptor_takes_the_lower_locant(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", DESCRIPTORS_IN_PREFIXES)
def test_descriptor_in_a_prefix_numbers_and_orders(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", CYCLOALKANE_FIRST_CITED)
def test_carbocycle_first_cited_prefix_takes_the_lower_locant(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", HETEROCYCLES)
def test_heteromonocycle_first_cited_prefix_then_descriptor(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", VON_BAEYER)
def test_von_baeyer_first_cited_prefix_then_descriptor(smiles, expected):
    assert name_compound(smiles) == expected
