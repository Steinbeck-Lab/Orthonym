"""Imidic acid esters beyond the =NH oxygen ester: imidothioates, N-substituted imidates, hydrazonates,
carbamimidothioates "Esters of acids modified by functional replacement nomenclature",
, the Blue Book-31999;.

Each was named by the systematic imine/hydrazone form and shipped ``pin_verified``
('1-(methylsulfanyl)ethan-1-imine', '1-ethoxy-N-methylethan-1-imine',
'(1-ethoxypropylidene)hydrazine'), or not named at all (the S-ester of carbamimidic acid, the
imibenconazole row d1454 of dev2000). The names below are the Blue Book's verbatim PINs where the book
prints the molecule, and OPSIN reads each back to the input.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _same(smi, name):
    out = opsin_parse(name)
    m_out = Chem.MolFromSmiles(out) if out else None
    return m_out is not None and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == inchi.MolToInchiKey(m_out)


# (smiles, PIN): the first five are Blue Book examples (:31993-:31999,:33473-:33485)
BOOK = [
    ("CSC(=N)c1ccccc1", "methyl benzenecarboximidothioate"),
    ("CCOC(CC)=NN", "ethyl propanehydrazonate"),
    ("CCSC(=N)N(C)C", "ethyl N,N-dimethylcarbamimidothioate"),
    ("CCSC(N)=NC", "ethyl N'-methylcarbamimidothioate"),
    ("CCSC(=N)NC", "ethyl N-methylcarbamimidothioate"),
]
# the molecules the review found shipping a systematic imine name as pin_verified, and the dev2000 row
BUILT = [
    ("CC(=N)SC", "methyl ethanimidothioate"),
    ("CC(=NC)OCC", "ethyl N-methylethanimidate"),
    ("CC(=Nc1ccccc1)OC", "methyl N-phenylethanimidate"),
    ("Clc1ccc(CSC(Cn2cncn2)=Nc2ccc(Cl)cc2Cl)cc1",
     "(4-chlorophenyl)methyl N-(2,4-dichlorophenyl)-2-(1H-1,2,4-triazol-1-yl)ethanimidothioate"),
]


@pytest.mark.parametrize("smiles,expected", BOOK + BUILT)
def test_imidic_ester_pin(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected, row["name"]
    assert row["tier"] == "pin_verified" and row["producer"] == "inner:imidic_ester", row
    assert _same(smiles, expected), expected


# A ring stem without a strict-path name for its imidic acid (a heteroaromatic or fused ring) takes the name of
# the carboxylic acid of the same ring with '-carboxylic acid' turned into the ester suffix:
# (the Blue Book) "names of esters are all derived from acids modified by functional replacement whose
# substitutive names are systematically formed", 'methyl benzenecarboximidothioate (PIN)' (:31997). These were
# '1-(methylsulfanyl)-1-(pyridin-4-yl)methanimine' and the like, shipped pin_verified; the molecule with a
# stereo element in the organyl group R' cites it there ('(2S)-2-chloropropyl'), as (:44639) "When they
# relate to substituent groups, they are cited at the front of the corresponding prefix" (:44643).
RING_STEMS = [
    ("CSC(=N)c1ccncc1", "methyl pyridine-4-carboximidothioate"),
    ("CSC(=N)c1ccco1", "methyl furan-2-carboximidothioate"),
    ("CSC(=N)c1ccc2ccccc2c1", "methyl naphthalene-2-carboximidothioate"),
    ("CSC(=N)c1c[nH]c2ccccc12", "methyl 1H-indole-3-carboximidothioate"),
    ("CSC(=N)c1cc(Cl)ccn1", "methyl 4-chloropyridine-2-carboximidothioate"),
    ("CCOC(=N)c1ccncc1", "ethyl pyridine-4-carboximidate"),
    ("CCOC(=N)C1CCCCC1", "ethyl cyclohexanecarboximidate"),
    ("CCOC(=NN)c1ccncc1", "ethyl pyridine-4-carbohydrazonate"),
    ("C[C@H](Cl)CSC(=N)C1CCCCC1", "(2S)-2-chloropropyl cyclohexanecarboximidothioate"),
]


@pytest.mark.parametrize("smiles,expected", RING_STEMS)
def test_a_ring_stem_without_an_imidic_acid_name_and_a_stereo_organyl_group(smiles, expected):
    from tests.support.pin_tiers import assert_pin_at_both_tiers
    assert_pin_at_both_tiers(smiles, expected)
    assert Orthonym().name_tiered(smiles)["producer"] == "inner:imidic_ester"


# The ester of an imidic acid is class 9, senior to the amides, nitriles, aldehydes, ketones and alcohols
#,:18162); a name that cites the group as an imine is not the PIN, and a molecule whose ester
# this handler does not build (a group on the N of a heteroaromatic stem, a stereo element in the stem) is no
# longer labelled one.
@pytest.mark.parametrize("smiles", [
    "CSC(=N)CCO", "CSC(=N)CC(=O)N", "CSC(=N)CC#N", "CSC(=N)CC(=O)C", "CSC(=N)CC=O",
    "CSC(=NC)c1ccncc1",
    "C[C@H](Cl)CSC(=N)C1CCC[C@@H](C)C1",
    # class 7a (:18172) puts the carboximidic acids before the sulfonic and the sulfinic acids, and
    # the esters keep the order of their acids (ASSUMED: class 9 states no order): the sulfonate and the sulfinate
    # are junior to the imidic ester
    "CSC(=N)CCS(=O)(=O)OC", "CSC(=N)CCS(=O)OC", "CSC(=N)c1ccc(cc1)S(=O)(=O)OC",
    "CSC(=N)CCP(=O)(OC)OC",      # the phosphonate ester: the same order of the acids, 7a before 7c
])
def test_an_imine_name_of_an_imidic_ester_is_not_a_pin(smiles):
    from tests.support.pin_tiers import name_breadth, name_default
    for row in (name_default(smiles), name_breadth(smiles)):
        assert row["tier"] != "pin_verified" and not row["is_pin"], row


@pytest.mark.parametrize("smiles,expected", [
    ("CSC(=N)CCOP(=O)(OC)OC", "methyl 3-[(dimethoxyphosphoryl)oxy]propanimidothioate"),
    ("CSC(=N)CCOP(=O)(O)OC", "methyl 3-{[hydroxy(methoxy)phosphoryl]oxy}propanimidothioate"),
])
def test_a_phosphate_group_beside_the_imidic_ester_is_a_prefix_at_the_default_tier(smiles, expected):
    # the default tier reads the group as the imine and builds the ester; the best-effort tier reads the phosphate
    # ester as the principal group and named the molecule 'propyl dimethyl phosphate', below the PIN. The class 7c
    # esters of phosphorus (:18172) are junior to the imidic ester, so both give the ester (the second row's best-effort
    # name is a prefix spelling below the PIN,
    from tests.support.pin_tiers import name_breadth, name_default
    from tests.support.rt_assert import name_is_rt_exact
    default, breadth = name_default(smiles), name_breadth(smiles)
    assert default["name"] == breadth["name"] == expected, (default["name"], breadth["name"])
    assert default["tier"] == "pin_verified" and default["producer"] == "inner:imidic_ester", default
    assert name_is_rt_exact(expected, smiles)


def test_the_tier_of_a_fused_ring_stem_does_not_depend_on_the_tier_asked():
    # the best-effort tier named the acid by the general engine and the ester took that engine's label
    from tests.support.pin_tiers import name_breadth, name_default
    smiles = "CSC(=N)c1ccc2ccccc2c1"
    assert name_default(smiles)["tier"] == name_breadth(smiles)["tier"] == "pin_verified"


# an unsaturated chain: the principal chain is the longest "Greater number of skeletal atoms"),
# then the one with the greater number of multiple bonds, then of double bonds;
# its multiple bonds are cited by 'ene' / 'yne' endings, never as an ylidene / ylidyne prefix on a shorter chain
UNSATURATED = [
    ("C=CCC(=NC)OC", "methyl N-methylbut-3-enimidate"),
    ("C#CCC(=NC)OC", "methyl N-methylbut-3-ynimidate"),
    ("C=C(C)CC(=NC)OC", "methyl N,3-dimethylbut-3-enimidate"),
    ("C=CC(C)C(=NC)OC", "methyl N,2-dimethylbut-3-enimidate"),
    ("CC=CCC(=NC)SC", "methyl N-methylpent-3-enimidothioate"),
    ("CCC(C=C)C(=NC)OC", "methyl 2-ethyl-N-methylbut-3-enimidate"),    # equal length: the double bond wins
    ("C=C(CC)CC(=NC)OC", "methyl N-methyl-3-methylidenepentanimidate"),  # the longer chain beats the double bond
    ("CC=CCC(=NN)OC", "methyl pent-3-enehydrazonate"),
]


@pytest.mark.parametrize("smiles,expected", UNSATURATED)
def test_an_unsaturated_chain_is_cited_by_ene_and_yne(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected, row["name"]
    assert row["tier"] == "pin_verified" and row["producer"] == "inner:imidic_ester", row
    assert _same(smiles, expected), expected


@pytest.mark.parametrize("name", ["methyl N-methylbut-3-enimidate", "methyl N-methylbut-3-ynimidate",
                                  "methyl pent-3-enehydrazonate", "methyl N-methylpent-3-enimidothioate"])
def test_the_numbering_check_reads_the_imidic_endings_as_chain_terminal(name):
    # 'but-3-enimidate' carries its suffix at C-1 like 'but-3-enoate'; the check read the 3 as the suffix locant
    from orthonym.validation.spelling.checks_locants import numbering_check
    assert numbering_check(None, name) is None


# "Carbohydroximic acids": 'N-hydroxyethanimidic acid (PIN)'; the ester keeps the N-hydroxy prefix.
# These were 'S-methyl ethanethioate oxime' / 'methyl acetate oxime' (the oxime handler's class name) before.
N_HYDROXY = [
    ("CC(=NO)SC", "methyl N-hydroxyethanimidothioate"),
    ("CC(=NO)OC", "methyl N-hydroxyethanimidate"),
    ("ClCC(=NO)OC", "methyl 2-chloro-N-hydroxyethanimidate"),
    ("c1ccccc1C(=NO)OC", "methyl N-hydroxybenzenecarboximidate"),
]


@pytest.mark.parametrize("smiles,expected", N_HYDROXY)
def test_n_hydroxy_imidic_ester(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected, row["name"]
    assert row["tier"] == "pin_verified" and row["producer"] == "inner:imidic_ester", row
    assert _same(smiles, expected), expected


# An N-alkoxy imidic ester. The ester is class 9 of (the Blue Book), senior to the imines
# (class 20), so the ester names the molecule and the N group is a prefix, as 'N-hydroxy' (: and 'N-sulfanyl'
# are: 'ethyl N-sulfanylbenzenecarboximidothioate (PIN)',:31999). The old path named these
# '1-methoxy-1-(methoxyimino)ethane' (changed from "left to the old path"; OPSIN reads each name below back to the input).
N_ALKOXY = [
    ("CC(=NOC)OC", "methyl N-methoxyethanimidate"),
    ("CC(=NOC)OCC", "ethyl N-methoxyethanimidate"),
    ("CC(=NOCC)OC", "methyl N-ethoxyethanimidate"),
    ("CC(=NOc1ccccc1)OC", "methyl N-phenoxyethanimidate"),
    ("CC(=NOCc1ccccc1)OC", "methyl N-(benzyloxy)ethanimidate"),
    ("COC(=NOC)c1ccccc1", "methyl N-methoxybenzenecarboximidate"),
    ("COC(=NOC)CC(C)C", "methyl N-methoxy-3-methylbutanimidate"),
    ("CC(=NOC)SC", "methyl N-methoxyethanimidothioate"),
    ("C(=NOC)OC", "methyl N-methoxymethanimidate"),
]


@pytest.mark.parametrize("smiles,expected", N_ALKOXY)
def test_n_alkoxy_imidic_ester(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected, row["name"]
    assert row["tier"] == "pin_verified" and row["producer"] == "inner:imidic_ester", row
    assert _same(smiles, expected), expected


@pytest.mark.parametrize("smiles", [
    "C/C(=N\\OC)OC",           # a C=N descriptor would be dropped
    "CC(=NOC(C)=O)OC",         # the O-acyl group is an ester of its own: the pipeline decides
])
def test_n_alkoxy_edges_are_not_claimed(namer, smiles):
    assert namer.name_tiered(smiles)["producer"] != "inner:imidic_ester"


@pytest.mark.parametrize("smiles", [
    "CCC(=NN)SC",              # the S-ester of a hydrazonic acid: the book prints no example, the old path runs
    "ON=C(N)SC",               # N-hydroxy carbamimidothioate: not this handler's stem
    "ON=C(N)OC",               # the O-ester of N-hydroxycarbamimidic acid: the imidate handler's stem
])
def test_n_hydroxy_and_hydrazonothio_edges_are_not_claimed(namer, smiles):
    assert namer.name_tiered(smiles)["producer"] != "inner:imidic_ester"


@pytest.mark.parametrize("smiles,name,producer", [
    ("CC(=N)OC", "methyl ethanimidate", "inner:imidate"),                  #:31993
    ("CCOC(=N)N(C)C", "ethyl N,N-dimethylcarbamimidate", "inner:imidate"),
])
def test_the_imidate_handler_keeps_its_molecules(namer, smiles, name, producer):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["producer"]) == (name, producer)


@pytest.mark.parametrize("smiles", [
    "CC1=NCCO1",                                       # ring imidate (2-oxazoline): not this class
    "C/C(=N\\C)OCC",                                   # a C=N stereo descriptor would be dropped
    "CC(=N)SCC(=O)N",                                  # an amide beside it: the pipeline decides
    "CC(=N)SC.CC(=N)OC",                               # two imidic ester groups
    "C=CCC(=NOS(=O)(=O)O)SC1OC(CO)C(O)C(O)C1O",        # thiohydroximate, not an imidate
    "CCSC(=N)N1CCCC1",                                 # ring amino N: left to the old path
])
def test_not_claimed_outside_the_scope(namer, smiles):
    assert namer.name_tiered(smiles)["producer"] != "inner:imidic_ester"


def test_registered_ahead_of_the_claimants():
    entry = INNER_DISPATCH_TABLE["imidic_ester"]
    assert entry.priority == 90
    assert entry.priority < INNER_DISPATCH_TABLE["oxime"].priority < INNER_DISPATCH_TABLE["hydrazone"].priority


@pytest.mark.parametrize("name", [
    "4-(1-sulfanylideneethyl)benzoic acid",
    "4-chloro-7-[(3-chloro-4-nitroquinolin-7-yl)sulfanyl]-3-nitroquinoline",
    "2-(cyanosulfonyl)cyclohexane-1-carboxylic acid",
    "1-(2-oxophosphepan-3-yl)propan-1-one",
])
def test_the_four_refuted_names_are_not_pin_verified(namer, name):
    # PB trace 4: a re-run / breadth name must never reach pin_verified by a label change
    out = opsin_parse(name)
    if out:
        row = namer.name_tiered(out)
        assert not (row["tier"] == "pin_verified" and row["name"] == name)
