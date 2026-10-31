"""PB lane R: decorated ring-yl prefixes the default (PIN) tier builds in its promotion re-run, certified.

``assembly.strict_ring_prefix`` puts one producer in front of the breadth promotion of the PIN tier's re-run
(``rules.pin_vocabulary.promote_at_pin_tier``): the decoration composer restricted to the classes whose name
is the PIN by construction. A name every piece of which that producer or the strict namer built keeps
``pin_verified`` (``metrics.provenance.record_pin_promotion_rerun`` flags the re-run only when a breadth
piece was kept). The rows are dev2000 witnesses (ids in the comments of the plan, never a holdout split rows); each
expected name is read back by a fresh OPSIN call (``tests.support.rt_assert``) and was derived by hand:

* NUMBERING (the Blue Book): (c) the free valence before (e) the ene locants (:3289) before (f)
  the prefixes as a set (:3301); ``cyclohex-3-en-1-yl (preferred prefix)``:3268.
* Prefix nomenclature (:24862): a ring C=O is the prefix 'oxo' of a hydro form
  ('4-oxo-4,5-dihydro-1H-imidazol-5-yl').
"""
import random

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly import strict_ring_prefix
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]


def _row(smiles):
    return Orthonym(style="pin").name_tiered(smiles)


def _shuffled(smiles, seed):
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


# ---------------------------------------------------------------------------------------------
# T1: a decorated monocycle-yl (carbocycle with ene locants, PIN-stem heterocycle, book form)
# ---------------------------------------------------------------------------------------------

T1_ROWS = [
    ('d574', 'CC/C=C\\C[C@H]1C(=O)C=C[C@@H]1CCCCCCCC(=O)O',
     '8-{(1S,5R)-4-oxo-5-[(2Z)-pent-2-en-1-yl]cyclopent-2-en-1-yl}octanoic acid'),
    ('d452', 'C=C(C)C1CC=C(CO)CC1',
     '[4-(prop-1-en-2-yl)cyclohex-1-en-1-yl]methanol'),
    ('d9', 'O=C(O)C(=O)CC1=CCC(O)C=C1',
     '3-(4-hydroxycyclohexa-1,5-dien-1-yl)-2-oxopropanoic acid'),
    ('d281', 'CC[C@H]1C(=O)C=C[C@@H]1/C=C/[C@H](O)C/C=C\\C/C=C\\C/C=C\\CCC(=O)O',
     '(4Z,7Z,10Z,13R,14E)-15-[(1S,5R)-5-ethyl-4-oxocyclopent-2-en-1-yl]-13-hydroxypentadeca-4,7,10,14-tetraenoic acid'),
    ('d1200', 'CC1=C[C@H](O)[C@H]([C@H](C)C(=O)O)CC1',
     '(2S)-2-[(1S,2R)-2-hydroxy-4-methylcyclohex-3-en-1-yl]propanoic acid'),
    ('d1280', 'CC1=CCC(C(C)CC=O)CC1',
     '3-(4-methylcyclohex-3-en-1-yl)butanal'),
    ('d1127', 'CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(\\C)CO)C(C)(C)CCC1',
     '(2E,4E,6E,8E,10E,12E,14E,16E,18E,20E,22E,24E)-2,6,10,14,19,23-hexamethyl-25-(2,6,6-trimethylcyclohex-1-en-1-yl)pentacosa-2,4,6,8,10,12,14,16,18,20,22,24-dodecaen-1-ol'),
    ('d1176', 'COC1CCC(=O)C(CC2C(C(C)CCC=C(C)C)=CCC2(C)O)=C1O',
     '3-hydroxy-2-{[5-hydroxy-5-methyl-2-(6-methylhept-5-en-2-yl)cyclopent-2-en-1-yl]methyl}-4-methoxycyclohex-2-en-1-one'),
    ('d1758', 'CC(=O)C1(C)CC=C(C)CC1',
     '1-(1,4-dimethylcyclohex-3-en-1-yl)ethan-1-one'),
    ('d1946', 'CC1=CC(=O)CC(C)(C)C1CCC(C)CC(=O)O',
     '3-methyl-5-(2,6,6-trimethyl-4-oxocyclohex-2-en-1-yl)pentanoic acid'),
    ('d860', 'CC(=O)C1CC=C(C2CC=C(C)C2(C)C)CC1',
     '1-[4-(2,2,3-trimethylcyclopent-3-en-1-yl)cyclohex-3-en-1-yl]ethan-1-one'),
    ('d1004', 'C=C1CC[C@H](Cl)C(C)(C)[C@@H]1C[C@@]12OC(C)(C)[C@H](Cl)C=C1C(=O)c1c(O)cc(O)cc1C2=O',
     '(3R,10aR)-3-chloro-10a-{[(1R,3S)-3-chloro-2,2-dimethyl-6-methylidenecyclohexyl]methyl}-6,8-dihydroxy-2,2-dimethyl-3,10a-dihydro-2H-naphtho[2,3-b]pyran-5,10-dione'),
    ('d1259', 'OC[C@H]1CCN1c1nc(-c2cnn(C3CCNCC3)c2)cc(C(F)(F)F)n1',
     '[(2R)-1-{4-[1-(piperidin-4-yl)-1H-pyrazol-4-yl]-6-(trifluoromethyl)pyrimidin-2-yl}azetidin-2-yl]methanol'),
    ('d851', 'CCNC(=O)CCC/C=C\\C[C@@H]1[C@@H](/C=C/C(=O)CCc2ccccc2)[C@H](O)C[C@@H]1O',
     '(5Z)-7-{(1R,2R,3R,5S)-3,5-dihydroxy-2-[(1E)-3-oxo-5-phenylpent-1-en-1-yl]cyclopentyl}-N-ethylhept-5-enamide'),
    ('d1792', 'CCNC(=O)CCC/C=C\\C[C@@H]1[C@@H](CC[C@@H](O)CCc2ccccc2)[C@H](O)C[C@@H]1O',
     '(5Z)-7-{(1R,2R,3R,5S)-3,5-dihydroxy-2-[(3R)-3-hydroxy-5-phenylpentyl]cyclopentyl}-N-ethylhept-5-enamide'),
    ('d1845', 'CC(C)=C[C@@H]1C[C@H](C)[C@@H](COc2ccc3ccc(=O)oc3c2)[C@]1(C)CCC(=O)O',
     '3-[(1R,2R,3S,5S)-1,3-dimethyl-5-(2-methylprop-1-en-1-yl)-2-{[(2-oxo-2H-1-benzopyran-7-yl)oxy]methyl}cyclopentyl]propanoic acid'),
    ('d713', 'O=C(O)CC[C@@H]1NC=NC1=O',
     '3-[(5S)-4-oxo-4,5-dihydro-1H-imidazol-5-yl]propanoic acid'),
    ('d1640', 'CC1=CC(=O)OC1CC(=O)[O-]',
     '(3-methyl-5-oxo-2,5-dihydrofuran-2-yl)acetate'),
]


@pytest.mark.parametrize("rid,smiles,expected", T1_ROWS, ids=[r[0] for r in T1_ROWS])
def test_t1_the_default_tier_ships_the_pin(rid, smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert row["opsin"] == "verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("rid,smiles,expected", T1_ROWS[::3], ids=[r[0] for r in T1_ROWS[::3]])
def test_t1_the_name_does_not_depend_on_the_input_atom_order(rid, smiles, expected):
    for seed in (1, 2, 3):
        assert _row(_shuffled(smiles, seed))["name"] == expected, (seed, _shuffled(smiles, seed))


@pytest.mark.parametrize("rid,smiles,expected", [T1_ROWS[5], T1_ROWS[12], T1_ROWS[16]],
                         ids=["d1280", "d1259", "d713"])
def test_t1_mutation_without_the_producer_the_row_is_declined(monkeypatch, rid, smiles, expected):
    # the assertions above depend on the producer: with it removed the re-run's breadth name is
    # labelled below the PIN again and the default tier declines (NO_VERIFIED_PIN)
    monkeypatch.setattr(strict_ring_prefix, "_KINDS", frozenset())
    row = _row(smiles)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row
    assert not row["is_pin"], row


# ---------------------------------------------------------------------------------------------
# negative controls: they never reach pin_verified (brief trace 4 and the plan)
# ---------------------------------------------------------------------------------------------

NEGATIVE = [
    # (the thioyl acyl prefix row, BB '4-(ethanethioyl)benzoic acid':30246, left this list: the strict
    # path builds its PIN, asserted at both tiers in test_pin_class_t12_acid_suffixes.py)
    # BB '3-chloro-7-[(4-chloro-3-nitroquinolin-7-yl)sulfanyl]-4-nitroquinoline' (:22088)
    ("identical ring halves", "O=[N+]([O-])c1cnc2cc(Sc3ccc4c([N+](=O)[O-])c(Cl)cnc4c3)ccc2c1Cl"),
    # BB '2-sulfurocyanidoylcyclohexane-1-carboxylic acid'
    ("sulfurocyanidoyl", "N#CS(=O)(=O)C1CCCCC1C(=O)O"),
    # BB '3-propanoylphosphepan-2-one': the ring ketone is the parent:19340)
    ("ring-ketone parent", "CCC(=O)C1CCCCPC1=O"),
    # a ring assembly is a parent of its own:15542)
    ("bi-indole", "Oc1cc2[nH]cc(-c3c[nH]c4cc(O)c(O)cc34)c2cc1O"),
    # carbohydrate rings are named by carbohydrate nomenclature:52681)
    ("ribotide", "NC(=O)c1ncn([C@@H]2O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c1N"),
]


@pytest.mark.parametrize("label,smiles", NEGATIVE, ids=[n[0] for n in NEGATIVE])
def test_negative_controls_never_reach_pin_verified(label, smiles):
    row = _row(smiles)
    assert row["tier"] != "pin_verified" and not row["is_pin"], row


def test_a_lactone_beside_a_ring_ketone_ships_only_the_ketone_parent():
    # (the Blue Book) "There is no seniority order difference between ketones and
    # pseudoketones... the seniority order between rings and ring systems" and (d) (:19414)
    # more rings: the benzopyranone is the parent. The oxolan-2-one parent with the benzopyranone as a
    # prefix (the route the re-run leaves) is never the shipped name
    smiles = "O=C1CC(C2CCC(=O)O2)Oc2ccccc21"
    row = _row(smiles)
    assert row["name"] == "2-(5-oxooxolan-2-yl)-2,3-dihydro-4H-1-benzopyran-4-one", row
    assert "oxolan-2-one" not in row["name"], row
    assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------------------------
# T2: (g) "lowest locants for the substituent cited first as a prefix in the name"
# (the Blue Book): the ring-yl is numbered the same way whatever the order of the input atoms
# ---------------------------------------------------------------------------------------------

G_SPELLINGS = [
    "OC(=O)CC1C(Br)=CCC=C1Cl",
    "OC(=O)CC1C(Cl)=CCC=C1Br",
    "ClC1=CCC=C(Br)C1CC(=O)O",
    "BrC1=CCC=C(Cl)C1CC(=O)O",
]


@pytest.mark.parametrize("smiles", G_SPELLINGS)
def test_t2_bromo_is_cited_first_and_takes_locant_two(smiles):
    row = _row(smiles)
    assert row["name"] == "(2-bromo-6-chlorocyclohexa-2,5-dien-1-yl)acetic acid", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_t2_a_tie_that_moves_the_descriptors_is_left_to_the_breadth_path():
    # the descriptors of 2,6-dimethylcyclohexyl would take their locants from the input order
    row = _row("OC(=O)C[C@H]1[C@H](C)CCC[C@@H]1C")
    assert row["tier"] != "pin_verified", row


# ---------------------------------------------------------------------------------------------
# T3: a decorated FUSED ring-yl named by the fusion forms (retained and fusion names with hydro
# prefixes and indicated hydrogen: (b) (e),:24862,:24872
# '5,8-dioxo-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid (PIN)',:29471
# '9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid (PIN)') where a fusion name is allowed
#,:23708: at least two rings of five or more members)
# ---------------------------------------------------------------------------------------------

T3_ROWS = [
    ('d16', 'NCc1c[nH]c(=S)n1[C@H]1CCc2c(F)cc(F)cc2C1',
     '5-(aminomethyl)-1-[(2S)-5,7-difluoro-1,2,3,4-tetrahydronaphthalen-2-yl]-1,3-dihydro-2H-imidazole-2-thione'),
    ('d69', 'Clc1cc(Cl)c2c(c1)CCc1ccccc1C2n1ccnc1',
     '1-(2,4-dichloro-10,11-dihydro-5H-dibenzo[a,d][7]annulen-5-yl)-1H-imidazole'),
    ('d251', 'Cn1cnc2c1c(=O)n(CCCC(=O)O)c(=O)n2C',
     '4-(3,7-dimethyl-2,6-dioxo-2,3,6,7-tetrahydro-1H-purin-1-yl)butanoic acid'),
    ('d261', 'C=CC(C)(C)n1cc(C[C@@H]2NC(=O)[C@H](C[C@@H](C)CO)N(C)C2=O)c2ccccc21',
     '(3S,6S)-6-[(2R)-3-hydroxy-2-methylpropyl]-1-methyl-3-{[1-(2-methylbut-3-en-2-yl)-1H-indol-3-yl]methyl}piperazine-2,5-dione'),
    ('d1400', 'CC(=O)CC[C@@H](C(=O)O)c1c(O)cc2c(c1O)C(=O)c1c(O)cc(O)cc1C2=O',
     '(2R)-5-oxo-2-(1,3,6,8-tetrahydroxy-9,10-dioxo-9,10-dihydroanthracen-2-yl)hexanoic acid'),
    ('d1537', 'COC1=C(N)C(=O)c2nc(-c3nc(C(=O)O)c(C)c(-c4cccc(OC)c4O)c3N)ccc2C1=O',
     '5-amino-6-(7-amino-6-methoxy-5,8-dioxo-5,8-dihydroquinolin-2-yl)-4-(2-hydroxy-3-methoxyphenyl)-3-methylpyridine-2-carboxylic acid'),
    ('d1919', 'CC(=O)OC1CC(C)Cc2ccc(C)c(/C=C/C(O)CC(O)CC(=O)O)c21',
     '(6E)-7-[8-(acetyloxy)-2,6-dimethyl-5,6,7,8-tetrahydronaphthalen-1-yl]-3,5-dihydroxyhept-6-enoic acid'),
    ('d722', 'CN1C(=O)[C@H](Cc2c[nH]c3cccc([N+](=O)[O-])c23)N(C)C(=O)C1Cc1ccccc1',
     '(6S)-3-benzyl-1,4-dimethyl-6-[(4-nitro-1H-indol-3-yl)methyl]piperazine-2,5-dione'),
    ('d1006', 'CC(C)=CCc1c2c(c(O)c3c1OC(c1ccc4c(c1)C=CC(C)(C)O4)CC3=O)C=CC(C)(C)O2',
     "8-(2,2-dimethyl-2H-1-benzopyran-6-yl)-5-hydroxy-2,2-dimethyl-10-(3-methylbut-2-en-1-yl)-7,8-dihydro-2H,6H-benzo[1,2-b:5,4-b']dipyran-6-one"),
    ('d1217', 'CC(C)=CCc1ccc2c(C3=C(O)C(=O)C(c4c[nH]c5cc(CC=C(C)C)ccc45)=C(O)C3=O)c[nH]c2c1',
     '2,5-dihydroxy-3,6-bis[6-(3-methylbut-2-en-1-yl)-1H-indol-3-yl]cyclohexa-2,5-diene-1,4-dione'),
    ('m2-126', 'COc1ccc(C2Oc3cccc(OC(F)F)c3-c3ccc(NC(=O)N(C)C)cc32)cc1OC',
     "N'-[1-(difluoromethoxy)-6-(3,4-dimethoxyphenyl)-6H-dibenzo[b,d]pyran-8-yl]-N,N-dimethylurea"),
    ('d1711', 'OCCN1CCN(CCCN2c3ccccc3Sc3ccc(C(F)(F)F)cc32)CC1',
     '2-(4-{3-[2-(trifluoromethyl)-10H-phenothiazin-10-yl]propyl}piperazin-1-yl)ethan-1-ol'),
]


@pytest.mark.parametrize("rid,smiles,expected", T3_ROWS, ids=[r[0] for r in T3_ROWS])
def test_t3_the_default_tier_ships_the_pin(rid, smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_t3_a_glycoside_is_declined_at_the_default_tier():
    # d200 left T3_ROWS relabel 2026-10-31): the 'α-L-rhamnopyranosyloxy' prefix is a name and
    #:50943 gives no PIN in Chapter, so the default tier declines; the best-effort tier gives the name
    from orthonym.cli import _emit_tier_flags
    smiles = 'C[C@@H]1O[C@@H](Oc2cccc3c2C(=O)c2ccc(C[C@@](C)(O)CC(=O)O)c(O)c2C3=O)[C@H](O)[C@H](O)[C@H]1O'
    name = ('(3R)-3-hydroxy-4-[1-hydroxy-5-(α-L-rhamnopyranosyloxy)-9,10-dioxo-9,10-dihydroanthracen-2-yl]'
            '-3-methylbutanoic acid')
    assert _row(smiles)["limit_code"] == "NO_VERIFIED_PIN"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("rid,smiles,expected", T3_ROWS[::4], ids=[r[0] for r in T3_ROWS[::4]])
def test_t3_the_name_does_not_depend_on_the_input_atom_order(rid, smiles, expected):
    for seed in (1, 2):
        assert _row(_shuffled(smiles, seed))["name"] == expected, (seed, _shuffled(smiles, seed))


def test_t3_a_benzopyranyl_methyl_prefix_of_a_lactone():
    smiles = "COc1ccc(C2=C(Cc3ccc4c(c3)CC(O)C(C)(C)O4)[C@H](OC)OC2=O)cc1"
    row = _row(smiles)
    assert row["name"] == ("(5R)-4-[(3-hydroxy-2,2-dimethyl-3,4-dihydro-2H-1-benzopyran-6-yl)methyl]-5-methoxy-"
                           "3-(4-methoxyphenyl)furan-2(5H)-one"), row
    assert row["tier"] == "pin_verified" and row["is_pin"], row


@pytest.mark.parametrize("smiles", [
    "O=[N+]([O-])c1cnc2cc(Sc3ccc4c([N+](=O)[O-])c(Cl)cnc4c3)ccc2c1Cl",   # identical ring skeletons, no principal group
    "CC(=O)NC1CCS(=O)(=O)C1",                                              # a ring sulfone: lambda convention
    #: the benzimidazole is the senior ring on the amine nitrogen, so it is the parent; the
    # re-run used to cite it as a prefix of a piperidin-4-amine
    "CN1CCC(CC1)N(C)C2=CC3=C(C=C2)N=C(N3)C4=NNC5=C4C=C(C=C5)OCC6=CC=CC=C6",
    # a urea carbon on a ring nitrogen: the ring is the parent of a carboxamide (PIN piperidine-1-carboxamide)
    "O=C(Nc1ccc(N2CCOC2=O)cc1)N1CCc2c(C(C)C)cccc2C1",
])
def test_t3_controls_never_reach_pin_verified(smiles):
    row = _row(smiles)
    assert row["tier"] != "pin_verified" and not row["is_pin"], row


# ---------------------------------------------------------------------------------------------
# T4: a von Baeyer core with no fusion name (the Blue Book "Numbering bicyclic alicyclic
# hydrocarbons":9589: numbering starts at a bridgehead, the longer segment first, the main bridge last;
#:16697: low locants to the heteroatoms of an 'a' name, in accord with the fixed numbering of
# the system, then to unsaturated sites; (c):3256 the free valence next)
# ---------------------------------------------------------------------------------------------

T4_ROWS = [
    ('d190', 'C=C1CC[C@H]2C[C@@H]1[C@@]2(C)CC/C=C(\\C)C(=O)O',
     '(2E)-2-methyl-5-[(1S,5S,6S)-6-methyl-2-methylidenebicyclo[3.1.1]heptan-6-yl]pent-2-enoic acid'),
    ('d124', '[NH3+][C@@H](C[C@@H]1CCC(=O)[C@@H]2O[C@H]12)C(=O)[O-]',
     '(2S)-2-azaniumyl-3-[(1R,2S,6R)-5-oxo-7-oxabicyclo[4.1.0]heptan-2-yl]propanoate'),
    ('d897', '[NH3+][C@@H](C[C@@H]1CC[C@@H](O)[C@@H]2O[C@H]12)C(=O)[O-]',
     '(2S)-2-azaniumyl-3-[(1R,2S,5R,6S)-5-hydroxy-7-oxabicyclo[4.1.0]heptan-2-yl]propanoate'),
    # (:9885) 'adamantane (PIN)'; 'adamantan-2-yl (preferred prefix)' (:16374, 'also 1-isomer')
    ('adamantane', 'C1C2CC3CC1CC(C2)(C3)COC(=O)CCl', '(adamantan-1-yl)methyl chloroacetate'),
    # the Blue Book's own PIN (the Blue Book,: 9-borabicyclo[3.3.1]nonane, boron at the
    # one-atom bridge, on a stannolane
    ('bb68.1.5.2.3', '[CH3][Sn]1([CH3])[CH2]CC(B2C3CCCC2CCC3)[CH2]1',
     '3-(9-borabicyclo[3.3.1]nonan-9-yl)-1,1-dimethylstannolane'),
    # a cage on a bridgehead-free position of a bicyclo[2.1.1]hexane, a norbornane on a methylene
    ('p176969388', 'C1[C@@H](C2CC1(C2)C3=CC=NC=C3)N4C(=O)CC(CC4=O)C5=CC=C(C=C5)C(F)(F)F',
     '1-[(2S)-4-(pyridin-4-yl)bicyclo[2.1.1]hexan-2-yl]-4-[4-(trifluoromethyl)phenyl]piperidine-2,6-dione'),
    ('p107053118', 'CN1C=C(N=N1)CC2(CC3CCC2C3)CNC4CC4',
     'N-({2-[(1-methyl-1H-1,2,3-triazol-4-yl)methyl]bicyclo[2.2.1]heptan-2-yl}methyl)cyclopropanamine'),
]


@pytest.mark.parametrize("rid,smiles,expected", T4_ROWS, ids=[r[0] for r in T4_ROWS])
def test_t4_the_default_tier_ships_the_pin(rid, smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("rid,smiles,expected", T4_ROWS, ids=[r[0] for r in T4_ROWS])
def test_t4_the_name_does_not_depend_on_the_input_atom_order(rid, smiles, expected):
    for seed in (1, 2):
        assert _row(_shuffled(smiles, seed))["name"] == expected, (seed, _shuffled(smiles, seed))


@pytest.mark.parametrize("rid,smiles,expected", [T4_ROWS[0], T4_ROWS[3]], ids=["d190", "adamantane"])
def test_t4_mutation_without_the_cage_branch_the_row_is_declined(monkeypatch, rid, smiles, expected):
    # the assertions above depend on the cage branch of the policy
    monkeypatch.setattr(strict_ring_prefix, "cage_core_ok", lambda mol, core: False)
    row = _row(smiles)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row
    assert not row["is_pin"], row


@pytest.mark.parametrize("label,smiles", [
    # a sugar of unspecified configuration: no cited rule makes the systematic oxane parent the PIN
    ("d418 unspecified sugar", "CC1(C)OC2(C)CCC1CC2OC1OC(CO)C(O)C(O)C1O"),
    # two five-membered rings share a bond: a bridged fused name is preferred,:23831)
    ("tricyclo core", "OC(=O)CC1CC(O)C2C1C1CCC2C1"),
    # a ring double bond: (c) before (e); 'bicyclo[2.2.1]hept-2-en-5-yl' would be wrong
    ("ring double bond", "OC(=O)CC1C(O)C2C=CC1C2"),
    # d1414: two six-membered rings share a bond, so a bridged fused name is preferred to the cage name
    ("d1414 bridged fused", "C=C1CCC2C3C(C(C)(C)O)CCC2(C)C13"),
])
def test_t4_controls_never_reach_pin_verified(label, smiles):
    row = _row(smiles)
    assert row["tier"] != "pin_verified" and not row["is_pin"], row


# ---------------------------------------------------------------------------------------------
# T5: a six-membered azine ring ketone takes added hydrogen and hydro prefixes. the Blue Book
# "'Added indicated hydrogen'" (:24687): "This method is preferred over the use of nondetachable hydro
# prefixes for preferred IUPAC names."; (:24695) 'pyridin-1(2H)-yl (preferred
# prefix)', 'naphthalen-1(2H)-one (PIN)'; (:24766) hydro prefixes take the locants of the
# saturated positions; 1-ethyl-2-oxopyridin-1-ium-1(2H)-yl (PIN) (:43483) against the nondetachable-hydro
# '1-ethyl-2-oxo-1,2-dihydropyridin-1-ium-1-yl'.
# ---------------------------------------------------------------------------------------------

T5_ROWS = [
    ('d73', 'N[C@@H](Cn1ccc(=O)[nH]c1=O)C(=O)O',
     '(2S)-2-amino-3-(2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)propanoic acid'),
    ('p79873996', 'CC(C)(C(=O)N)N1C=C(C(=O)NC1=O)C#N',
     '2-(5-cyano-2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)-2-methylpropanamide'),
    ('uracil-1-yl acetic acid', 'OC(=O)Cn1ccc(=O)[nH]c1=O',
     '(2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)acetic acid'),
    ('thymin-1-yl acetic acid', 'OC(=O)Cn1cc(C)c(=O)[nH]c1=O',
     '(5-methyl-2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)acetic acid'),
]


@pytest.mark.parametrize("rid,smiles,expected", T5_ROWS, ids=[r[0] for r in T5_ROWS])
def test_t5_the_default_tier_ships_the_pin(rid, smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("rid,smiles,expected", T5_ROWS, ids=[r[0] for r in T5_ROWS])
def test_t5_the_name_does_not_depend_on_the_input_atom_order(rid, smiles, expected):
    for seed in (1, 2):
        assert _row(_shuffled(smiles, seed))["name"] == expected, (seed, _shuffled(smiles, seed))


def test_t5_mutation_without_the_azine_kind_the_row_is_declined(monkeypatch):
    monkeypatch.setattr(strict_ring_prefix, "_KINDS",
                        strict_ring_prefix._KINDS - {"azine"})
    row = _row(T5_ROWS[0][1])
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row
    assert not row["is_pin"], row


def test_t5_a_ring_ketone_beside_a_ketone_stays_the_parent():
    # class 16 beside class 16:18189): the ring ketone competes for the parent,
    # so the oxo is not a prefix; the strict first run names the dione and this producer is not asked
    smiles = "CC(=O)Cn1ccc(=O)[nH]c1=O"
    row = _row(smiles)
    assert row["name"] == "1-(2-oxopropyl)pyrimidine-2,4(1H,3H)-dione", row
    assert row["tier"] == "pin_verified", row


# ---------------------------------------------------------------------------------------------
# T6: the strict twin runs the promotion re-run, so a certified name has the same label at every tier
# (metrics.provenance: a re-run name is flagged only when a breadth piece was kept)
# ---------------------------------------------------------------------------------------------

def _best_effort_row(smiles):
    return Orthonym(style="pin", general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True).name_tiered(smiles)


ALL_ROWS = T1_ROWS + T3_ROWS + T4_ROWS + T5_ROWS


@pytest.mark.parametrize("rid,smiles,expected", ALL_ROWS[::2], ids=[r[0] for r in ALL_ROWS[::2]])
def test_t6_the_best_effort_tier_gives_the_same_name_and_label(rid, smiles, expected):
    row = _best_effort_row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row


def test_t6_mutation_a_twin_without_the_rerun_leaves_the_best_effort_label_below_the_pin(monkeypatch):
    from orthonym import namer
    original = namer.Orthonym._strict_pin_twin_name

    def without_rerun(self, smiles):
        token = namer._PIN_PROMOTION_WRAPPED.set(True)
        try:
            return original(self, smiles)
        finally:
            namer._PIN_PROMOTION_WRAPPED.reset(token)
    monkeypatch.setattr(namer.Orthonym, "_strict_pin_twin_name", without_rerun)
    row = _best_effort_row(T5_ROWS[0][1])
    assert row["name"] == T5_ROWS[0][2], row
    assert row["tier"] == "pin_unverified" and not row["is_pin"], row


def test_t6_a_name_with_a_breadth_piece_stays_below_the_pin_at_best_effort():
    # (PB merge) the sulfinylbenzimidazole used before is named by the strict first run since lane P ('methanesulfinyl',; the dihydropyridazinyl prefix below is still a breadth piece
    row = _best_effort_row("CN(CCO)C1=CNNC=C1")
    assert row["tier"] == "pin_unverified" and not row["is_pin"], row


# ---------------------------------------------------------------------------------------------
# fix a performance pass (review of the lane): every name below is the PIN form or declines
# ---------------------------------------------------------------------------------------------

def test_fix1_an_n_yl_takes_added_hydrogen():
    # (the Blue Book) + (:24687): 'hexahydropyrrolo[1,2-a]pyrazin-2(1H)-yl', the form
    # of the bare core, not 'octahydropyrrolo[1,2-a]pyrazin-2-yl'
    smiles = "CC1CN2CCCC2CN1C(C)C(=O)C3=CC=C(S3)Cl"
    row = _row(smiles)
    assert row["name"] == ("1-(5-chlorothiophen-2-yl)-2-(3-methylhexahydropyrrolo[1,2-a]pyrazin-2(1H)-yl)"
                           "propan-1-one"), row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("label,smiles", [
    # '2-methyl-1,2,3,4-tetrahydroquinolin-1-yl' (the retained-hydro catalogue); the PIN prefix is
    # '2-methyl-3,4-dihydroquinolin-1(2H)-yl' (:7467,:29376,:33131)
    ("F3 tetrahydroquinolin-1-yl", "CC1CCc2ccccc2N1c1ccc(C(C)NC)nc1"),
    # (:21698): two pyridines tie on the amine nitrogen; the lower prefix locants decide
    ("F2 tied pyridines", "CN1C(=C(C(=N1)Br)[N+](=O)[O-])N2CCC(CC2)N(C3=CN=C(C=C3)OC(F)F)C4=C(C=CN=C4)OC"),
    # the PIN of an azo compound with no senior group is a diazene (:38829)
    ("F5 azo", "c1ccc(N=Nc2ccc3ccc(N=Nc4ccc5cc6ccccc6cc5c4)cc3c2)cc1"),
])
def test_fix1_the_default_tier_declines(label, smiles):
    row = _row(smiles)
    assert row["tier"] != "pin_verified" and not row["is_pin"], row


def test_fix1_the_sulfolane_row_is_named_by_its_ring_heterone():
    """2026-10-26 (PINB lane), change-asserted-value: row 'F4 sulfolane' was a decline (the ring sulfone written
    '1,1-dioxothiolan' lacks the lambda,,:2756). The ring producer of (the Blue Book: "When
    a group -SO- or -SO2- is part of a ring system, oxygen atom(s) are expressed substitutively by the suffix '-one'
    added to the name of the heterocycle in which the sulfur atoms are designated as lambda4 or lambda6 atoms...
    This method generates preferred IUPAC names") names it with the heterone as the suffix, which stands above the
    alcohol and the amine in (Table 4.1,:18158); the prefix cited first, '(2,3-dimethylthiomorpholin-4-yl)'
    ('d'), takes the lower locant (g),:3307;,:3477 "first letter of its complete name")."""
    smiles = "CC1C(SCCN1C2CS(=O)(=O)CC2O)C"
    row = _row(smiles)
    assert row["name"] == "3-(2,3-dimethylthiomorpholin-4-yl)-4-hydroxy-1λ6-thiolane-1,1-dione", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_fix1_an_azo_compound_with_a_principal_group_keeps_its_ring_parent():
    row = _row("O=S(=O)(O)c1ccc(N=Nc2c(O)ccc3ccccc23)cc1")
    assert row["name"] == "4-[(2-hydroxynaphthalen-1-yl)diazenyl]benzene-1-sulfonic acid", row
    assert row["tier"] == "pin_verified", row


@pytest.mark.parametrize("smiles,expected", [
    # '(propane-1-sulfonyl)benzene (PIN)' (the Blue Book): the locant of a chain of three or more
    # carbons is cited; ethane and methane omit it
    ("CCCS(=O)(=O)N1CCC(CC1)NC(=O)COCC", "2-ethoxy-N-[1-(propane-1-sulfonyl)piperidin-4-yl]acetamide"),
    ("CCCS(=O)(=O)N1CCCCC1", "1-(propane-1-sulfonyl)piperidine"),
    ("CCCCS(=O)(=O)N1CCCCC1", "1-(butane-1-sulfonyl)piperidine"),
    ("CCS(=O)(=O)N1CCCCC1", "1-(ethanesulfonyl)piperidine"),
    ("CC(C)S(=O)(=O)N1CCCCC1", "1-(propane-2-sulfonyl)piperidine"),
    ("CCCS(=O)(=O)c1ccc(C(=O)O)cc1", "4-(propane-1-sulfonyl)benzoic acid"),
])
def test_fix1_the_acyl_prefix_of_a_sulfonic_acid_cites_its_locant(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)
