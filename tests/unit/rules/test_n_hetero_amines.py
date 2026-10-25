""" N-substitution of amines and amides by heteroatom groups (NAZ).

The class: a nitrogen that carries one heteroatom group -- azido, chalcogen (-OR, -SR, -SeR, -SH...),
nitro, nitroso, halogen and its oxides, -OH. Two producers, one per class of the compound:

* ``rules/nitramide.py``: the amide of a acid ('dimethylhypochlorous amide', 'methylnitrous
  amide', 'methyl(nitro)nitramide'), where makes the nitrogen the nitrogen of an amide;
* ``rules/n_hetero_amines.py``: the group as an N-prefix of an amine, carboxamide or sulfonamide
  ('N-azido-N-methylmethanamine', 'N-chloro-N-methylacetamide').

Every expected name below carries the Blue Book heading and line it rests on, or says that it is derived
from a rule (and which) because the book prints no such compound. VERIFIED means a printed name; DERIVED a name
the rule gives; ASSUMED a reading the book does not decide. The names are read back by OPSIN as the
suite has it (2.9.0 in CI) to the input's full InChIKey, independently of the engine's own gate.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

BB = "BlueBookV2.md"

# --------------------------------------------------------------------------------------------------
# Names the book prints (PIN), verbatim: (smiles, name, heading and line)
# --------------------------------------------------------------------------------------------------
PRINTED = [
    ("CCNCl", "ethylhypochlorous amide",
     f'P-62.4 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" {BB}:26604; P-68.5.3 "Amides of halogen acids" :39706'),
    ("CNN=O", "methylnitrous amide", f'P-62.4 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" {BB}:26606'),
    ("CN([N+](=O)[O-])[N+](=O)[O-]", "methyl(nitro)nitramide",
     f'P-62.4 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" {BB}:26608'),
    ("CNCl", "methylhypochlorous amide", f'P-61.3.2.2 {BB}:25886 ("methylhypochlorous amide (PIN; see P-68.5.3)")'),
    ("CCCN(CCC)N=O", "dipropylnitrous amide",
     f'P-67.1.2.6.3 "Amides and hydrazides of nitric and nitrous acids" {BB}:35882'),
    ("CCCCN(CC)N=O", "butyl(ethyl)nitrous amide",
     f'P-67.1.2.6.3 "Amides and hydrazides of nitric and nitrous acids" {BB}:35884'),
    ("CN(CCl)[N+](=O)[O-]", "(chloromethyl)(methyl)nitramide",
     f'P-67.1.2.6.3 "Amides and hydrazides of nitric and nitrous acids" {BB}:35886'),
    ("NN=O", "nitrous amide", f'P-58.3.2 "When one or more atoms of a homogeneous heteroatomic chain ..." {BB}:24922 (preselected name)'),
    ("O=NNc1ccccc1", "phenylnitrous amide", f'P-15.1.8.2.1 "Substitution rules for Type 2a retained names" {BB}:4985'),
    ("CN(C(N)=O)N=O", "N-methyl-N-nitrosourea",
     f'P-61.5.2 "Nitro attached to heteroatoms" {BB}:25965: a senior amide is the parent, the nitroso group a prefix'),
    ("O=C1CCC(=O)N1Br", "1-bromopyrrolidine-2,5-dione",
     f'P-66.2 "IMIDES" {BB}:33847 "1-bromopyrrolidine-2,5-dione (PIN)": a ring nitrogen keeps the heterocycle'),
    ("CCNOC", "N-methoxyethanamine", f'P-62.4 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" {BB}:26600'),
]

# --------------------------------------------------------------------------------------------------
# Names derived from a rule, no printed example (the rule and the reading are named)
# --------------------------------------------------------------------------------------------------
DERIVED = [
    #:26594 "R-NH-Cl, R-NH-NO, and R-NH-NO2 are now named as derivatives of amides" +: both
    # substituents of the amide N are prefixes, cited without locant like 'butyl(ethyl)nitrous amide':35884
    ("CN(C)Cl", "dimethylhypochlorous amide"),
    ("CN(C)N=O", "dimethylnitrous amide"),
    ("c1ccccc1N(C)Cl", "methyl(phenyl)hypochlorous amide"),
    ("CCN(CC)Br", "diethylhypobromous amide"),
    #:26588 "unless a higher class is formed" + "AZIDES":25989 + Table 5.1:24990 'azido'; ASSUMED:
    # the book prints no N-azido amine, the general sentence of gives the reading
    ("CN(C)N=[N+]=[N-]", "N-azido-N-methylmethanamine"),
    ("CNN=[N+]=[N-]", "N-azidomethanamine"),
    ("CCN(CC)N=[N+]=[N-]", "N-azido-N-ethylethanamine"),
    ("CC(C)N(C)N=[N+]=[N-]", "N-azido-N-methylpropan-2-amine"),
    ("c1ccccc1N(C)N=[N+]=[N-]", "N-azido-N-methylaniline"),
    #:26596 "Substitution of amines is permitted by -OR, -SR, -SeR, and -TeR groups" and its example
    # 'N-methoxyethanamine (PIN)':26600; the second N-substituent takes its N locant, order
    ("CN(C)OC", "N-methoxy-N-methylmethanamine"),
    ("CN(C)SC", "N-methyl-N-(methylsulfanyl)methanamine"),
    ("CNSC", "N-(methylsulfanyl)methanamine"),
    ("CN(C)[Se]C", "N-methyl-N-(methylselanyl)methanamine"),
    #:18184 (class 11 amides "in the order of the corresponding acids") and:18453 (the
    # nitrogen and halogen acids are class 7e, after the carboxylic and sulfonic acids of 7a): the carboxamide
    # or sulfonamide is the class, the group a prefix. The book says it for the nitro and nitroso groups,
    # "Nitro attached to heteroatoms" (:25957): "In the presence of a characteristic group having seniority to be
    # named as a suffix or a parent structure, nitro and nitroso groups can be attached to any atom", with
    # 'N-methyl-N-nitrosourea (PIN)' (:25965) for a senior amide as the parent and the group a prefix of its N
    ("CC(=O)N[N+](=O)[O-]", "N-nitroacetamide"),
    ("CC(=O)N(C)[N+](=O)[O-]", "N-methyl-N-nitroacetamide"),
    ("CC(=O)N(C)Cl", "N-chloro-N-methylacetamide"),
    ("CC(=O)NCl", "N-chloroacetamide"),
    ("CC(=O)N(C)N=O", "N-methyl-N-nitrosoacetamide"),
    ("CC(=O)N(C)N=[N+]=[N-]", "N-azido-N-methylacetamide"),
    ("c1ccccc1C(=O)N(C)Cl", "N-chloro-N-methylbenzamide"),
    ("CS(=O)(=O)N(C)Cl", "N-chloro-N-methylmethanesulfonamide"),
    ("CS(=O)(=O)N(C)N=O", "N-methyl-N-nitrosomethanesulfonamide"),
    ("CS(=O)(=O)N(C)[N+](=O)[O-]", "N-methyl-N-nitromethanesulfonamide"),
    #:26590 "-OH groups and chalcogen analogues"; 'N-hydroxypropanamide (PIN)':17572 keeps the amide
    ("CS(=O)(=O)N(C)O", "N-hydroxy-N-methylmethanesulfonamide"),
    ("CC(=O)N(C)O", "N-hydroxy-N-methylacetamide"),
    # the groups of the amide of a acid with a group of a JUNIOR class in the substituent:
    # hydrazides 12, aldehydes 15, ketones 16, hydroxy compounds 17, amines 19; heterocycles; the prefix-only groups)
    ("CN(Cl)CC(=O)NN", "(2-hydrazinyl-2-oxoethyl)(methyl)hypochlorous amide"),
    ("CCN(N=O)CCO", "ethyl(2-hydroxyethyl)nitrous amide"),
    ("CN(N=O)CC(C)=O", "methyl(2-oxopropyl)nitrous amide"),
    ("CN(Cl)c1ccccn1", "methyl(pyridin-2-yl)hypochlorous amide"),
    ("CN(N=O)CCN(C)C", "[2-(dimethylamino)ethyl](methyl)nitrous amide"),
    # classes 22 and below of (phosphorus, silicon, boron, selenium) are junior to the amides: a
    # phosphane or a selenide in the substituent leaves the amide the class
    ("CN(N=O)CCP(C)C", "[2-(dimethylphosphanyl)ethyl](methyl)nitrous amide"),
    ("CN(N=O)CCP", "methyl(2-phosphanylethyl)nitrous amide"),
    ("CN(N=O)CC[Se]C", "methyl[2-(methylselanyl)ethyl]nitrous amide"),
    ("CN(Cl)CCP(C)C", "[2-(dimethylphosphanyl)ethyl](methyl)hypochlorous amide"),
    # two groups of acids on the nitrogen: the more senior acid is the amide (nitric > nitrous, N > F > Cl,
    #:18453 "(a) the central atom first in the list N > F > Cl > Br > I; (b) highest oxidation number"), the
    # other group a prefix: the book's 'methyl(nitro)nitramide':26608 with another group
    ("CN(N=O)[N+](=O)[O-]", "methyl(nitroso)nitramide"),
    ("CN(Cl)N=O", "chloro(methyl)nitrous amide"),
]

# One printed row is not reachable as an input: RDKit does not read a halogen oxide drawn with its
# hypervalent bonds ('CNBr=O'), and the charge separated drawing has another InChIKey than OPSIN's reading, so
# no name of such an input can be verified (the engine abstains). The producer itself is tested on the drawing.
HALOGEN_OXIDES = [
    ("CN[Br+][O-]", "methylbromous amide", f'P-62.4 {BB}:26610 "methylbromous amide (PIN)"'),
    ("CN[Br+](=O)[O-]", "methylbromic amide", f'P-68.5.3 "Amides of halogen acids" {BB}:39704'),
    ("CN[Cl+](=O)(=O)[O-]", "methylperchloric amide", "P-42.5 :18453 perchloric acid, P-68.5.3 :39700"),
    ("CNI=O", "methyliodous amide", "P-42.5 :18453 iodous acid, P-68.5.3 :39700"),
]


def _row(smiles):
    return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.parametrize("smiles,name,cite", PRINTED, ids=[r[1] for r in PRINTED])
def test_printed_name(smiles, name, cite):
    row = _row(smiles)
    assert row["name"] == name, (cite, row)
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,name", DERIVED, ids=[r[1] for r in DERIVED])
def test_derived_name(smiles, name):
    row = _row(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,name,cite", HALOGEN_OXIDES, ids=[r[1] for r in HALOGEN_OXIDES])
def test_halogen_oxide_amide_names(smiles, name, cite):
    from orthonym.rules.nitramide import name_substituted_nitramide

    assert name_substituted_nitramide(Chem.MolFromSmiles(smiles)) == name, cite


def test_best_effort_tier_gives_the_same_names():
    """The PIN is the best-effort name too: nothing in the class ranks below it."""
    from tests.support.rt_assert import name_best_effort

    for smiles, name in [("CCNCl", "ethylhypochlorous amide"), ("CN(C)N=[N+]=[N-]", "N-azido-N-methylmethanamine"),
                         ("CC(=O)N(C)Cl", "N-chloro-N-methylacetamide")]:
        row = name_best_effort(smiles)
        assert row["name"] == name and row["tier"] == "pin_verified", row


# --------------------------------------------------------------------------------------------------
# What the producers leave alone
# --------------------------------------------------------------------------------------------------
SENIOR_TO_THE_AMIDE = [
    # a carboxylic acid, an ester, a carboxamide, a urea, a carbamate, an amidine, a hydroxamic acid, a cation, an
    # anion, a sulfonic acid or amide, an ester of an oxoacid: all senior to the amide of a class 7e acid
    #, the Blue Book-:18192: classes 4-6 ions, 7 acids, 9 esters, 11 amides)
    "CN(Cl)CC(=O)O", "CN(Cl)CC(=O)OC", "CN(Cl)CC(N)=O", "CN(Cl)CNC(N)=O", "CN(Cl)COC(=O)N", "CN(Cl)CC(=N)N",
    "CN(Cl)CC(=O)NO", "CN(Cl)CC(=O)[O-]", "CN(Cl)CC[N+](C)(C)C", "CN(Cl)CS(N)(=O)=O", "CN(Cl)CS(=O)(=O)O",
    "CN(Cl)COS(C)(=O)=O", "CN(Cl)CO[N+](=O)[O-]", "CN(Cl)CC(=S)N", "CN(Cl)CP(=O)(O)O", "CN(Cl)CC(Cl)=O",
    "CN(Cl)CC(=O)n1ccnc1", "CN(Cl)CC(=O)NC(C)=O",
    "CN(N=O)CNC(N)=O", "CN(N=O)COC(=O)N", "CN(N=O)CC(=O)NO", "CN(N=O)CC(=N)N", "CN(N=O)CC[N+](C)(C)C",
    "CCN(N=O)CCNC(=O)NC", "CN(N=O)Cc1ccncc1C(=O)O",
    # the acids and esters of the junior elements are groups of the elements' own senior classes, and the groups the
    # substituent namer cannot name (a selenol) are not taken: a phosphine oxide, a phosphonous or boronic acid,
    # a silanol, a selenoxide, a selenol
    "CN(N=O)CCP(=O)(C)C", "CN(N=O)CCP(O)O", "CN(N=O)CB(O)O", "CN(N=O)C[Si](C)(C)O", "CN(N=O)CC[Se](=O)C",
    "CN(N=O)CC[SeH]",
    # the nitramide is held to the same order: the acetamide, the acid, the amide, the urea are the class
    "CC(=O)N[N+](=O)[O-]", "CC(=O)N(C)[N+](=O)[O-]", "O=[N+]([O-])NCC(=O)O", "CN([N+](=O)[O-])CC(=O)O",
    "CN([N+](=O)[O-])CC(N)=O", "O=C(N)NCN([N+](=O)[O-])C",
]


@pytest.mark.parametrize("smiles", [
    "C1CCCCN1Cl",                      # a ring nitrogen: the heterocycle:18184 "cyclic amides are named as heterocycles")
    "C1CCCCN1N=O",
    "CC(=O)NCl",                       # an acyl on the nitrogen: the carboxamide is the class (nitramide.py declines)
    "CNOC",                            # R-NH-O-R': the hydroxylamine handler's,
] + SENIOR_TO_THE_AMIDE)
def test_amide_producer_declines(smiles):
    from orthonym.rules.nitramide import name_substituted_nitramide

    assert name_substituted_nitramide(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles", [
    "CN(C)Cl",                         # the amide of hypochlorous acid: nitramide.py
    "CN(C)N=O",
    "CN(C)O",                          # N-hydroxy amine: the hydroxylamine class
    "CNO",
    "CNOC",                            # the hydroxylamine handler's
    "C1CCCCN1N=[N+]=[N-]",             # a ring nitrogen
    "CC(Cl)C(=O)N(C)Cl",               # the skeleton carries a prefix of its own
    "CC[C@H](C)N(C)N=[N+]=[N-]",       # a specified configuration
    "CN(CC(=O)O)N=[N+]=[N-]",          # the acid is the class and the group a prefix of 'amino'
    "CC(=O)N(C)N",                     # a nitrogen chain
    "CC(=O)NC(=O)N(C)Cl",              # two acyl groups, an imide-like skeleton pair
])
def test_hetero_amine_producer_declines(smiles):
    from orthonym.rules.n_hetero_amines import name_n_hetero_substituted

    assert name_n_hetero_substituted(Chem.MolFromSmiles(smiles)) is None


def test_ring_nitrogen_names_are_unchanged():
    assert _row("C1CCCCN1Cl")["name"] == "1-chloropiperidine"
    assert _row("C1CCCCN1N=O")["name"] == "1-nitrosopiperidine"


def test_a_fragment_is_not_named_as_an_amine():
    """The recursion that names the prefix '[azido(methyl)amino]' of a substituent asks for the capped
    fragment's name; 'N-azidomethanamine' there has no conversion to the prefix, and the group was lost
    (measured: [azido(methyl)amino]acetic acid became a floor name). The best-effort name is as before."""
    from tests.support.rt_assert import name_best_effort

    smiles = "CN(N=[N+]=[N-])CC(=O)O"
    row = name_best_effort(smiles)
    assert row["name"] == "[azido(methyl)amino]acetic acid", row
    assert_full_rt(row["name"], smiles)


# --------------------------------------------------------------------------------------------------
# Mutations: each guard of the producers is load-bearing (the test fails if it is taken out)
# --------------------------------------------------------------------------------------------------
def test_mutation_senior_class_guard(monkeypatch):
    """Without the check of the amide of a class 7e acid is named over a carboxylic acid, a urea, a
    cation: the names the guard refuses (and, for the nitramide, over an acetamide)."""
    import orthonym.rules.nitramide as nitramide

    for smiles in ("CN(Cl)CC(=O)O", "CN(Cl)CNC(N)=O", "CC(=O)N[N+](=O)[O-]", "O=[N+]([O-])NCC(=O)O"):
        assert nitramide.name_substituted_nitramide(Chem.MolFromSmiles(smiles)) is None, smiles
    with monkeypatch.context() as patch:
        patch.setattr(nitramide, "_branches_are_junior", lambda *_a, **_k: True)
        for smiles, false_name in [
                ("CN(Cl)CC(=O)O", "(carboxymethyl)(methyl)hypochlorous amide"),
                ("CN(Cl)CNC(N)=O", "[(carbamoylamino)methyl](methyl)hypochlorous amide"),
                ("CC(=O)N[N+](=O)[O-]", "acetylnitramide"),
                ("O=[N+]([O-])NCC(=O)O", "(carboxymethyl)nitramide")]:
            assert nitramide.name_substituted_nitramide(Chem.MolFromSmiles(smiles)) == false_name, smiles


def test_mutation_each_layer_of_the_class_check_is_load_bearing(monkeypatch):
    """The check has two layers; each refuses something the other does not: the atoms (a charge, a carbonyl with
    a heteroatom neighbour) and the perceived groups (a urea, a carbamate, an ammonium)."""
    import orthonym.rules.nitramide as nitramide

    cation = Chem.MolFromSmiles("CN(N=O)CC[N+](C)(C)C")
    urea = Chem.MolFromSmiles("CN(Cl)CNC(N)=O")
    for mol in (cation, urea):
        assert nitramide.name_substituted_nitramide(mol) is None
    with monkeypatch.context() as patch:                 # the group layer alone refuses both
        patch.setattr(nitramide, "_atoms_are_junior", lambda *_a, **_k: True)
        for mol in (cation, urea):
            assert nitramide.name_substituted_nitramide(mol) is None
    more = frozenset(nitramide._JUNIOR_GROUPS | {"urea", "ammonium"})
    with monkeypatch.context() as patch:                 # the atom layer alone refuses both
        patch.setattr(nitramide, "_JUNIOR_GROUPS", more)
        for mol in (cation, urea):
            assert nitramide.name_substituted_nitramide(mol) is None
    with monkeypatch.context() as patch:                 # both off: the false name
        patch.setattr(nitramide, "_JUNIOR_GROUPS", more)
        patch.setattr(nitramide, "_atoms_are_junior", lambda *_a, **_k: True)
        assert nitramide.name_substituted_nitramide(urea) == "[(carbamoylamino)methyl](methyl)hypochlorous amide"


def test_mutation_parent_choice(monkeypatch):
    """:18453 chooses the amide of two groups: "(a) the central atom first in the list N > F > Cl > Br > I;
    (b) highest oxidation number for the central atom" ('nitric acid HO-NO2 nitrous acid HO-NO perfluoric acid...
    hypochlorous acid... hypoiodous acid IOH' is the order of the list). Reversing either criterion changes the
    parent, and with it the name."""
    import orthonym.rules.nitramide as nitramide

    chloro_nitroso = Chem.MolFromSmiles("CN(Cl)N=O")
    nitroso_nitro = Chem.MolFromSmiles("CN(N=O)[N+](=O)[O-]")
    assert nitramide.name_substituted_nitramide(chloro_nitroso) == "chloro(methyl)nitrous amide"
    assert nitramide.name_substituted_nitramide(nitroso_nitro) == "methyl(nitroso)nitramide"
    with monkeypatch.context() as patch:                        # (a) reversed: I > Br > Cl > F > N
        patch.setattr(nitramide, "_ELEMENT_RANK", {7: 4, 9: 3, 17: 2, 35: 1, 53: 0})
        assert nitramide.name_substituted_nitramide(chloro_nitroso) == "methyl(nitroso)hypochlorous amide"
    with monkeypatch.context() as patch:                        # (b) reversed: the lowest oxidation number
        patch.setattr(nitramide._AcidGroup, "seniority_key",
                      lambda self: (nitramide._ELEMENT_RANK[self.element], self.n_oxygen))
        assert nitramide.name_substituted_nitramide(nitroso_nitro) == "methyl(nitro)nitrous amide"


def test_mutation_host_name_is_read_not_assumed(monkeypatch):
    """The name of the host must be the name rebuilt from the parts; where it is not, nothing is built."""
    import orthonym.rules.n_hetero_amines as hetero

    assert hetero.name_n_hetero_substituted(Chem.MolFromSmiles("CN(C)N=[N+]=[N-]")) == "N-azido-N-methylmethanamine"
    real = hetero._pin_name
    # the host 'CNC' (N-methylmethanamine) is named otherwise than its parts make it; the primary host stays
    monkeypatch.setattr(hetero, "_pin_name", lambda smi: (real(smi) + "x") if smi == "CNC" else real(smi))
    assert hetero.name_n_hetero_substituted(Chem.MolFromSmiles("CN(C)N=[N+]=[N-]")) is None


def test_mutation_prefix_order(monkeypatch):
    """The group takes its place among the N-prefixes by (alphanumerical order)."""
    import orthonym.assembly.naming_utils as naming_utils
    import orthonym.rules.n_hetero_amines as hetero

    assert hetero.name_n_hetero_substituted(Chem.MolFromSmiles("CC(=O)N(C)Cl")) == "N-chloro-N-methylacetamide"
    real = naming_utils.alpha_sort_key
    monkeypatch.setattr(naming_utils, "alpha_sort_key", lambda name: tuple(-ord(c) for c in real(name)))
    assert hetero.name_n_hetero_substituted(Chem.MolFromSmiles("CC(=O)N(C)Cl")) == "N-methyl-N-chloroacetamide"


def test_mutation_whole_molecule_only(monkeypatch):
    """Named inside a fragment recursion both producers must decline; without that check the substituent route
    loses the group (the glycine row of ``test_a_fragment_is_not_named_as_an_amine``)."""
    import orthonym.assembly.fragment_naming as fragment_naming
    import orthonym.rules.n_hetero_amines as hetero
    import orthonym.rules.nitramide as nitramide

    azide = Chem.MolFromSmiles("CNN=[N+]=[N-]")
    chloro = Chem.MolFromSmiles("CNCl")
    assert hetero.name_n_hetero_substituted(azide) == "N-azidomethanamine"
    assert nitramide.name_substituted_nitramide(chloro) == "methylhypochlorous amide"
    monkeypatch.setattr(fragment_naming, "is_top_level_naming", lambda: False)
    assert hetero.name_n_hetero_substituted(Chem.MolFromSmiles("CNN=[N+]=[N-]")) is None
    assert nitramide.name_substituted_nitramide(Chem.MolFromSmiles("CNCl")) is None


def test_the_n_chloro_amide_best_effort_names_do_not_use_the_class_name_of_a_fragment():
    """The recursion that composes the prefix of an N-chloro-N-acylamino group asked for the name of the capped
    fragment; 'hypochlorous amide' there made the best-effort name another one (N-chloroacetyl...). The names
    stay those of the engine's own composition, and read back exactly."""
    from tests.support.rt_assert import name_best_effort

    for smiles in ("ClCC(=O)N(C)Cl", "CC(C)C(=O)N(C)Cl", "CC(=O)N(Cl)C(C)=O"):
        row = name_best_effort(smiles)
        assert row["name"] and "hypochlorous" not in row["name"], row
        assert_full_rt(row["name"], smiles)


def test_n_hydroxy_cited_among_the_n_prefixes_of_an_n_substituted_amide():
    """The hydroxylamine handler (``assembly/handlers/hydroxylamine.py``) spelled 'N-hydroxy' glued to the N-prefix
    of an N-substituted sulfonamide ('N-hydroxyN-(2-chloroethyl)methanesulfonamide', on the main tree):
    ``_n_hydroxy_among_n_prefixes`` cites it among the N-prefixes in the order of (the Blue Book,
    "simple prefixes are arranged alphabetically"), with the hyphen of the italic locant. The shapes below are
    this handler's, not ``n_hetero_amines``' (the substituent carries a prefix of its own)."""
    for smiles, name in [("CS(=O)(=O)N(CCCl)O", "N-(2-chloroethyl)-N-hydroxymethanesulfonamide"),
                         ("CS(=O)(=O)N(CCF)O", "N-(2-fluoroethyl)-N-hydroxymethanesulfonamide"),
                         ("CS(=O)(=O)N(CC(F)F)O", "N-(2,2-difluoroethyl)-N-hydroxymethanesulfonamide")]:
        row = Orthonym(style="pin").name_tiered(smiles)
        assert (row["name"], row["producer"]) == (name, "inner:hydroxylamine"), row
        assert_full_rt(name, smiles)


def test_mutation_n_hydroxy_order(monkeypatch):
    """Reversing the order of changes where 'N-hydroxy' is cited; without the re-anchored merge the old
    spelling ('N-hydroxyN-(2-chloroethyl)...') came out."""
    import orthonym.assembly.naming_utils as naming_utils

    smiles = "CS(=O)(=O)N(CCCl)O"
    real = naming_utils.alpha_sort_key
    monkeypatch.setattr(naming_utils, "alpha_sort_key", lambda name: tuple(-ord(c) for c in real(name)))
    name = Orthonym(style="pin").name_tiered(smiles)["name"]
    assert name != "N-(2-chloroethyl)-N-hydroxymethanesulfonamide", name


def test_the_silane_and_borane_substituents_keep_their_best_effort_names():
    """A silane or a borane in the substituent is junior to the amide (classes 26 and 30); the engine labels
    these names below the PIN (pin_unverified: the default tier declines them, as it did before the elements
    were admitted), and they read back to the input's full InChIKey."""
    from tests.support.rt_assert import name_best_effort

    for smiles, name in [("CN(N=O)C[Si](C)(C)C", "methyl[(trimethylsilyl)methyl]nitrous amide"),
                         ("CN(N=O)CB(C)C", "[(dimethylboranyl)methyl](methyl)nitrous amide")]:
        row = name_best_effort(smiles)
        assert row["name"] == name, row
        assert_full_rt(name, smiles)


def test_the_memo_is_keyed_on_the_naming_depth(monkeypatch):
    """The answer is kept for the naming call, and it depends on whether this is the name of a whole molecule:
    asked again of the same object inside a fragment recursion it must not be served from the top-level entry
    (it was: the memo keyed on the object and the style only)."""
    import orthonym.assembly.fragment_naming as fragment_naming
    import orthonym.rules.n_hetero_amines as hetero
    from orthonym.assembly.memo import pop_scope, push_scope

    token = push_scope()
    try:
        mol = Chem.MolFromSmiles("CN(C)N=[N+]=[N-]")
        assert hetero.name_n_hetero_substituted(mol) == "N-azido-N-methylmethanamine"
        with monkeypatch.context() as patch:
            patch.setattr(fragment_naming, "is_top_level_naming", lambda: False)
            assert hetero.name_n_hetero_substituted(mol) is None
        assert hetero.name_n_hetero_substituted(mol) == "N-azido-N-methylmethanamine"
    finally:
        pop_scope(token)


def test_the_memo_keeps_the_answer_for_the_call_and_never_holds_an_rwmol(monkeypatch):
    """Inside a naming call the second ask of the same molecule is the kept answer (the dispatch asks twice); an
    RWMol edited in place is asked afresh every time."""
    import orthonym.rules.n_hetero_amines as hetero
    from orthonym.assembly.memo import pop_scope, push_scope

    target = Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)N(C)Cl"))
    calls = []
    real = hetero._name_n_hetero_substituted

    def counted(mol, style):                     # the hosts, named inside, ask too; count the molecule's own
        if Chem.MolToSmiles(mol) == target:
            calls.append(1)
        return real(mol, style)

    monkeypatch.setattr(hetero, "_name_n_hetero_substituted", counted)
    token = push_scope()
    try:
        mol = Chem.MolFromSmiles("CC(=O)N(C)Cl")
        assert hetero.name_n_hetero_substituted(mol) == hetero.name_n_hetero_substituted(mol) == \
            "N-chloro-N-methylacetamide"
        assert len(calls) == 1
        edited = Chem.RWMol(Chem.MolFromSmiles("CC(=O)N(C)Cl"))
        assert hetero.name_n_hetero_substituted(edited) == "N-chloro-N-methylacetamide"
        edited.GetAtomWithIdx(4).SetAtomicNum(35)             # the methyl carbon becomes bromine: CC(=O)N(Cl)Br
        edited.GetAtomWithIdx(4).SetNoImplicit(True)
        assert hetero.name_n_hetero_substituted(edited) is None
    finally:
        pop_scope(token)
    outside = Chem.MolFromSmiles("CC(=O)N(C)Cl")                # no naming call: nothing is kept
    n = len(calls)
    hetero.name_n_hetero_substituted(outside)
    hetero.name_n_hetero_substituted(outside)
    assert len(calls) == n + 2


def test_mutation_a_memo_without_the_depth_serves_the_stale_answer(monkeypatch):
    """The depth in the key is load-bearing: keyed on the style alone the entry of the whole molecule is served
    inside a fragment recursion."""
    import orthonym.assembly.fragment_naming as fragment_naming
    import orthonym.perception.molcache as molcache
    import orthonym.rules.n_hetero_amines as hetero
    from orthonym.assembly.memo import pop_scope, push_scope

    real = molcache.cached_by_key
    monkeypatch.setattr(molcache, "cached_by_key", lambda mol, ns, key, fresh: real(
        mol, ns, key[0] if ns == "n_hetero_amines" else key, fresh))
    token = push_scope()
    try:
        mol = Chem.MolFromSmiles("CN(C)N=[N+]=[N-]")
        assert hetero.name_n_hetero_substituted(mol) == "N-azido-N-methylmethanamine"
        with monkeypatch.context() as patch:
            patch.setattr(fragment_naming, "is_top_level_naming", lambda: False)
            assert hetero.name_n_hetero_substituted(mol) == "N-azido-N-methylmethanamine"      # stale
    finally:
        pop_scope(token)
