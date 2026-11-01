"""A charge suffix cites the locant of its charged atom,,.

 Citation of locants (the Blue Book): "In preferred IUPAC names, if any locants are
essential for defining the structure of the parent structure or of a unit of structure as defined
by its appropriate enclosing marks, then all locants must be cited for the parent structure or
that structural unit."
"""
import re

import pytest
from rdkit import Chem

from orthonym.rules.charged_router import emit_primary_amine_anide
from orthonym.rules.ion_locants import ring_ylium_may_be_locant_free
from orthonym.rules.ions import apply_ion_suffix_to_name
from orthonym.rules.ring_position_kinds import one_kind_of_substitutable_position

from ._helpers import ambiguity_messages, assert_clean, name_row

# the names are the shipped ones: the OPSIN validity gate is on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate

_CATION_LOCANT = re.compile(r"\d[a-z]?'?-(?:ium|ylium|diazonium|diazonio)")


# one input per producer of a locant-free charge suffix
AZOLE_AND_AZINE_CATIONS = [
    "CPC1=NN=N[NH2+]1",                                                   # 1H-tetrazole
    "CC1[NH+]=C(NO1)C",                                                   # dihydro-1,2,4-oxadiazole
    "C1[NH+](N=CN1CC2=C(C=C(C=C2)Cl)Cl)CC3=C(C=C(C=C3)Cl)Cl.[Cl-]",      # dihydro-1,2,4-triazole
    "CC1=C(C=CC(=C1)Cl)CC2=CONC(=[NH+]2)C3=CC(=NN=C3SC4=CC=C(C=C4)N=O)Cl",  # 4H-1,2,4-oxadiazine
]
VON_BAEYER_AND_SPIRO_CATIONS = [
    "C[C@@H]1C2[NH+]1CCN(C2)C(=O)C(C3=CC(=C(C=C3)C)F)Br",                # 1,4-diazabicyclo[4.1.0]heptane
    "CN1C2=CC=CC=C2[C@]3(C1=O)CC[NH2+][C@@H]3C4=CC=CC=C4",               # spiro[indole-3,3'-pyrrolidine]
    "C1CCC2(CC1)[NH2+]C3=C(CCCC3)C(=S)N2",                               # spiro[cyclohexane-quinazoline]
    "CCC(=O)N1C(C2=C(N1)C3CC[NH+]2CC3)C4=CC=C(C=C4)[N+](=O)[O-]",        # tricyclic, one protonated N
    "CC1=CC(=NN1)[AsH]C2=CN(CC3=NC=C[NH+]32)SC4=CC=C(C=C4)C(=O)O",      # three N of the bicycle
]
CARBENIUM_IONS = [
    "C1CC2=CC=CC3=C2C(=CC=C3)[CH+]1",                                     # hydrophenalene
    "C1CCC2C(C1)CCC3C2CCC[CH+]3",                                         # perhydrophenanthrene
    "COC1=CC(=C(C=C1)C2=C[C+]=NN2C3=CC=CC=C3)OC",                         # pyrazole ring carbon
    "C1C=CC2=C([CH+]1)CC3=CC=CC=C32",                                     # dihydrofluorene
    "C1C2=C3C4=C(C=CC=C4C[C+]3C5=CC=CC=C51)C=C2",                         # polycyclic carbenium
]
DIAZONIUM_IONS = [
    "N#[N+]c1cccc2ccccc12",                                               # naphthalene, no locant in 'naphthalene'
    "N#[N+]c1c[nH]c(=O)[nH]c1=O",
    "C1=CN=CC2=C1C(=NN2)[N+]#N",
    "C1=CN=CC2=C1N=C(S2)[N+]#N",
    "N#[N+]c1ccc(C(=O)[O-])cc1",
]
RING_AND_AMINE_ANIONS = [
    "C1[C-](O1)C(F)(F)F",                                                # oxirane
    "C1[C-](O1)C",
    "C1=CC=C2C(C(=O)C=CC2=C1)[NH-]",                                     # amine on a ring that carries a ketone
    "CC1=C(SC(=N1)[NH-])S(=O)(=O)NC(C)(C)CO",                            # amine on a substituted thiazole
]


@pytest.mark.parametrize("smiles", AZOLE_AND_AZINE_CATIONS + VON_BAEYER_AND_SPIRO_CATIONS
                         + CARBENIUM_IONS + DIAZONIUM_IONS + RING_AND_AMINE_ANIONS)
def test_charge_suffix_names_are_clean(smiles):
    # (see above): the suffix of an ion cites the position of its charge
    assert_clean(smiles)


@pytest.mark.parametrize("smiles", AZOLE_AND_AZINE_CATIONS + VON_BAEYER_AND_SPIRO_CATIONS
                         + CARBENIUM_IONS + DIAZONIUM_IONS)
def test_cation_suffix_cites_its_locant(smiles):
    # General rule for systematically naming cationic centers in parent hydrides
    # (the Blue Book): 'adding one or more hydrons to any position of a neutral parent hydride';
    # examples '1-methylpyridin-1-ium (PIN)' and '1H-imidazol-3-ium (PIN)' (:41396).
    # Specific method, general method (:41559): 'removal of one hydride ion... from any
    # position of a parent hydride', 'furan-2-ylium (PIN)' (:41571).
    # Diazonium ions (:41607): '2,4-dioxopentane-3-diazonium (PIN)'; the prefix
    # 'diazonio' cites its locant as every prefix does.
    assert _CATION_LOCANT.search(name_row(smiles)["name"]), name_row(smiles)["name"]


@pytest.mark.parametrize("smiles,name", [
    ("C1[C-](O1)C(F)(F)F", "2-(trifluoromethyl)oxiran-2-ide"),
    ("C1[C-](O1)C", "2-methyloxiran-2-ide"),
])
def test_ring_anion_cites_the_locant_of_each_prefix(smiles, name):
    # Anions derived from parent hydrides and their derivatives (the Blue Book):
    # "locants identify positions of the negative charges"; the suffix cites locant 2, so
    # (:2869, see above) cites every locant of the unit: the oxirane has two kinds of position
    assert name_row(smiles)["name"] == name


@pytest.mark.parametrize("smiles", AZOLE_AND_AZINE_CATIONS + VON_BAEYER_AND_SPIRO_CATIONS
                         + CARBENIUM_IONS + DIAZONIUM_IONS + RING_AND_AMINE_ANIONS)
def test_default_tier_never_ships_an_ambiguous_ion_name(smiles):
    # the default tier ships a preferred name or declines
    row = name_row(smiles, tier="pin")
    if row["tier"] not in (None, "abstain"):
        assert not ambiguity_messages(row["name"]), row


@pytest.mark.parametrize("neutral,cls", [
    ("pyrimidine-2,4(1H,3H)-dione", "diazonium"),
    ("1H-pyrazolo[3,4-c]pyridine", "diazonium"),
    ("2,3-dihydro-1H-phenalene", "ylium"),
    ("1,2-dihydro-9H-fluorene", "ylium"),
])
def test_class_keyed_cation_suffix_declines_a_name_that_cites_locants(neutral, cls):
    # (the Blue Book) and (:41609): the suffix is cited at the
    # position of the cationic centre, which the numbering of the neutral name does not give
    assert apply_ion_suffix_to_name(neutral, 1, cation_class=cls) == ""


@pytest.mark.parametrize("neutral,cls,cation", [
    ("methane", "ylium", "methylium"),
    ("cyclobutane", "ylium", "cyclobutylium"),
    ("benzene", "diazonium", "benzenediazonium"),
    ("methane", "diazonium", "methanediazonium"),
    ("2-methylbutanoic acid", "acylium", "2-methylbutanoylium"),
])
def test_class_keyed_cation_suffix_keeps_the_licensed_forms(neutral, cls, cation):
    # Specific method (the Blue Book): 'methylium (PIN)', 'cyclobutylium (PIN)';
    # (:41613): 'methanediazonium (PIN)'; Acylium cations (:41621),:41623:
    # the 'oic acid' ending is replaced by 'oylium', so the name keeps the locants of the acid
    assert apply_ion_suffix_to_name(neutral, 1, cation_class=cls) == cation


@pytest.mark.parametrize("smiles,name", [
    ("[CH3+]", "methylium"),
    ("C1CC[CH+]C1", "cyclopentylium"),
    ("N#[N+]c1ccccc1", "benzenediazonium"),
    ("C[N+]#N", "methanediazonium"),
    ("CC(=O)C(C(C)=O)[N+]#N", "2,4-dioxopentane-3-diazonium"),
    ("c1c[nH]c[nH+]1", "1H-imidazol-3-ium"),
    ("C[NH3+]", "methanaminium"),
    ("[NH2+]=c1cc[nH]cc1", "pyridin-4(1H)-iminium"),
    ("[NH-]c1ccccc1", "benzenaminide"),
])
def test_licensed_charge_names_stay(smiles, name):
    # (the Blue Book), (:41613,:41615), (:41396),
    # Cationic centers on characteristic groups (:41413): the cation of an amine, imine
    # or nitrile group is named on the neutral parent, whose name cites the locant of the group;
    # Anions derived from amines and imines (:41047), 'benzenaminide (PIN)' (:41057):
    # one kind of position, or a cited locant
    assert name_row(smiles, tier="pin")["name"] == name


@pytest.mark.parametrize("smiles", [
    "CC1=C(SC(=N1)[NH-])S(=O)(=O)NC(C)(C)CO",
    "C1=CC=C2C(C(=O)C=CC2=C1)[NH-]",
    "[NH-]c1ccccc1C",
])
def test_primary_amine_anion_of_a_parent_with_several_positions_has_no_aminide_name(smiles):
    # (the Blue Book): the amine locant is omitted only on a parent hydride with
    # one kind of substitutable hydrogen; a parent that carries other groups needs the locant,
    # which the text transform has not, so no '...aminide' is built from it
    mol = Chem.MolFromSmiles(smiles)
    anion = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == -1)
    assert not emit_primary_amine_anide(mol, anion, "pin").endswith("aminide")


@pytest.mark.parametrize("smiles,name", [
    ("[NH-]c1ccccc1C", "(2-methylphenyl)azanide"),
    ("C1=CC=C2C(C(=O)C=CC2=C1)[NH-]", "(2-oxo-1,2-dihydronaphthalen-1-yl)azanide"),
])
def test_primary_amine_anion_of_a_substituted_parent_is_the_azanide_of_its_group(smiles, name):
    # Anionic centers on other characteristic groups (the Blue Book): "named on
    # the basis of the corresponding anionic parent hydrides" ('acetylazanide (PIN)',:41079); the
    # substituent group cites the locant of its free valence,:15814)
    assert name_row(smiles)["name"] == name


@pytest.mark.parametrize("smiles,one_kind", [
    ("c1ccccc1", True),
    ("C1CCCCC1", True),
    ("CC", True),
    ("C", True),
    ("Cc1ccccc1", False),
    ("CCC", False),
    ("c1ccc2ccccc2c1", False),
    ("C1CO1", True),
    ("FC(F)(F)C1CO1", False),
])
def test_one_kind_of_substitutable_position(smiles, one_kind):
    # (the Blue Book): "The locant is omitted in monosubstituted symmetrical parent
    # hydrides or parent compounds where there is only one kind of substitutable hydrogen."
    assert one_kind_of_substitutable_position(Chem.MolFromSmiles(smiles)) is one_kind


@pytest.mark.parametrize("smiles,licensed", [
    ("C1CC[CH+]C1", True),
    ("C1CC[CH+]CC1", True),
    ("C[C+]1CCC1", False),
    ("C1CC2CCC1[CH+]2", False),
    ("C1=CC=C[CH+]C=C1", False),
])
def test_ring_ylium_may_be_locant_free_only_on_a_bare_cycloalkane(smiles, licensed):
    # Specific method (the Blue Book): hydride removed "from... a saturated
    # monocyclic hydrocarbon" gives 'cyclobutylium (PIN)'; a ring that is not one takes the
    # general method (:41559), which cites the position of the centre. This producer has no
    # locant for a substituent, so it also declines a substituted saturated monocycle
    # ('1-methylcyclobutylium' is the likely specific-method PIN: ASSUMED, follow-up)
    assert ring_ylium_may_be_locant_free(Chem.MolFromSmiles(smiles)) is licensed


@pytest.mark.parametrize("smiles,name", [
    ("C[CH+]C", "propan-2-ylium"),
    ("CC(C)C[CH2+]", "3-methylbutan-1-ylium"),
    ("[CH2+]CC(C)C", "3-methylbutan-1-ylium"),
    ("CC=C[CH2+]", "but-2-en-1-ylium"),
    ("C=CC[CH2+]", "but-3-en-1-ylium"),
    ("C[CH+]C=CC", "pent-3-en-2-ylium"),
    ("OCC[CH2+]", "3-hydroxypropan-1-ylium"),
    ("[CH2+]CCO", "3-hydroxypropan-1-ylium"),
    ("CC1CC[CH+]C1", "3-methylcyclopentan-1-ylium"),
    ("[CH+]1CCCC(C)C1", "3-methylcyclohexan-1-ylium"),
])
def test_ylium_centre_has_the_lowest_locant_whatever_the_order_of_the_atoms(smiles, name):
    # NUMBERING (the Blue Book), criterion (c) "principal characteristic groups and
    # free valences (suffixes)" (:3256): the cumulative suffix 'ylium' Suffixes (2),
    #:4777: "Cumulative suffixes are used to designate radicals, ions, radical ions") is
    # decided before unsaturation and the detachable prefixes. The same molecule has one name
    # whatever the order of the atoms of its SMILES (core principle 4).
    row = name_row(smiles)
    assert row["name"] == name, row
    assert_clean(smiles, must_contain=name)


@pytest.mark.parametrize("smiles,chain,features,expected", [
    ("CC[CH2+]", [0, 1, 2], {0: 1, 1: 2, 2: 3}, {0: 3, 1: 2, 2: 1}),
    ("[CH2+]CC", [0, 1, 2], {0: 1, 1: 2, 2: 3}, {0: 1, 1: 2, 2: 3}),
    ("CCC[CH2+]", [0, 1, 2, 3], {0: 1, 1: 2, 2: 3, 3: 4}, {0: 4, 1: 3, 2: 2, 3: 1}),
    ("[CH2+]CCC", [0, 1, 2, 3], {0: 1, 1: 2, 2: 3, 3: 4}, {0: 1, 1: 2, 2: 3, 3: 4}),
    ("CC[CH+]CC", [0, 1, 2, 3, 4], {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}, {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}),
])
def test_chain_is_numbered_from_the_end_that_gives_the_ionic_centre_the_lowest_locant(
        smiles, chain, features, expected):
    # NUMBERING (the Blue Book), criterion (c) (:3256): see the test above. The
    # direction the features carry does not know the charge; the chain has two directions.
    from orthonym.assembly.general_engine import _orient_chain_for_ionic_centres
    mol = Chem.MolFromSmiles(smiles)
    assert _orient_chain_for_ionic_centres(mol, chain, dict(features)) == expected


@pytest.mark.parametrize("smiles", ["CC[CH2+]", "[CH2+]CC", "CCC[CH2+]", "[CH2+]CCC"])
def test_terminal_ylium_of_an_unbranched_alkane_is_the_specific_method_name(smiles):
    # Specific method (the Blue Book): "Cations formed formally by the removal
    # of a hydride ion, H-, from a terminal atom of a saturated unbranched acyclic hydrocarbon
    #... are named by replacing the 'ane' ending in the name of the parent hydride by the suffix
    # 'ylium'"; 'propylium (PIN)' (:41557). OPSIN reads 'propylium' as ambiguous and gives the
    # right structure; the method names the terminal atom by definition.
    from orthonym.rules.ions import specific_method_cation_name
    from orthonym.validation.opsin_ambiguity import licensed_ambiguity
    mol = Chem.MolFromSmiles(smiles)
    name = specific_method_cation_name(mol)
    assert name in ("propylium", "butylium")
    assert licensed_ambiguity(name, smiles) == "P-73.2.2.1.1"
    for tier in ("pin", "best-effort"):
        row = name_row(smiles, tier=tier)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    "C[CH+]C",            # the carbon of the centre is not at the end of the chain
    "CC[CH+]CC",
    "[CH2+]C(C)C",        # branched
    "C[C+](C)C",
    "[CH2+]C=C",          # unsaturated
    "[CH2+]C1CC1",        # a ring
    "[CH2+]CO",           # a heteroatom
    "[CH2+]CC.[Cl-]",     # a salt
])
def test_specific_method_class_is_a_terminal_cation_of_an_unbranched_alkane(smiles):
    # (the Blue Book, quoted above): the class is the terminal atom of a
    # saturated unbranched acyclic hydrocarbon, and nothing else
    from orthonym.rules.ions import specific_method_cation_name
    assert specific_method_cation_name(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles,name", [
    ("c1ccccc1[S+](c1ccccc1)c1ccccc1[O-]", "2-(diphenylsulfaniumyl)phenolate"),
    ("C1CCC(CC1)S(=O)(=O)c1ccc(cc1)[S+](c1ccccc1)c1ccccc1[O-]",
     "2-{[4-(cyclohexanesulfonyl)phenyl](phenyl)sulfaniumyl}phenolate"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_zwitterion_cites_the_locant_of_the_cation_on_a_ring_parent(smiles, name, tier):
    # Citation of locants (the Blue Book, quoted above): the onium prefix of the
    # phenolate is one of several positions of the ring and is cited with its locant, the
    # anionic carbon being C-1; the anion is the parent of a zwitterion /.
    row = assert_clean(smiles, tier=tier)
    assert row["name"] == name


@pytest.mark.parametrize("smiles,name", [
    ("C1CCCC(CC1)(C(=O)[O-])[P+](c2ccccc2)(c2ccccc2)c2ccccc2",
     "1-(triphenylphosphaniumyl)cycloheptanecarboxylate"),
    ("C[N+](C)(C)C1CCCCC1C(=O)[O-]", "2-(trimethylazaniumyl)cyclohexanecarboxylate"),
])
def test_zwitterion_of_a_ring_carboxylate_cites_the_locant_of_the_cation(smiles, name):
    # (the Blue Book): the locant of the substituent of the ring parent is cited,
    # the carbon of the carboxylate group being C-1. OPSIN reads '(triphenylphosphaniumyl)
    # cycloheptanecarboxylate' as ambiguous and the locanted name once. The ring of a cycloalkane
    # names its own suffix with a locant ('cycloheptane-1-carboxylate'), which the producer of the
    # neutral ring name does not cite, so the name is valid and not preferred; the default tier
    # declines it.
    row = assert_clean(smiles)
    assert row["name"] == name
    assert name_row(smiles, tier="pin")["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles,pin,wider", [
    ("C[N+](C)(C)c1ccccc1C(=O)[O-]",
     "2-(N,N-dimethylmethanaminiumyl)benzoate", "2-(trimethylazaniumyl)benzoate"),
    ("C[N+](C)(C)c1cccc(c1)C(=O)[O-]",
     "3-(N,N-dimethylmethanaminiumyl)benzoate", "3-(trimethylazaniumyl)benzoate"),
])
def test_zwitterion_of_benzoate_is_the_aminiumyl_pin_and_its_twin(smiles, pin, wider):
    # The locant of the cation is cited as in the rows above, the Blue Book). The onium prefix of the
    # PIN is '(N,N-dimethylmethanaminiumyl)' method (1) leads to the PIN,:42299;, example:42473), so
    # the default tier ships the name pin_verified (ruling R5, RULINGS-ANION.md); the wider tiers spell the same atoms
    # '(trimethylazaniumyl)', which OPSIN 2.9.0 reads (validation/twin_spellings.py), without the PIN label.
    row = assert_clean(smiles)
    assert row["name"] == wider
    pin_row = name_row(smiles, tier="pin")
    assert (pin_row["name"], pin_row["tier"]) == (pin, "pin_verified")


def test_zwitterion_ring_locant_is_the_lower_direction_round_the_ring():
    from rdkit.Chem import rdmolops

    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _ring_attachment_locant_on_anion_parent
    for smiles, expected in [("C[N+](C)(C)c1ccccc1[O-]", 2), ("C[N+](C)(C)c1cccc(c1)[O-]", 3),
                             ("C[N+](C)(C)c1ccc(cc1)[O-]", 4), ("C[N+](C)(C)C1CCCC1[O-]", 2),
                             ("C[N+](C)(C)C1CCCCCC1[O-]", 2), ("C[N+](C)(C)C1(CCCCC1)C(=O)[O-]", 1)]:
        mol = Chem.MolFromSmiles(smiles)
        sites = get_ion_sites(mol)
        cation, anion = sites["cations"][0]["atom_idx"], sites["anions"][0]["atom_idx"]
        attach = rdmolops.GetShortestPath(mol, cation, anion)[1]
        assert _ring_attachment_locant_on_anion_parent(mol, anion, cation, attach) == expected, smiles
    # a ring that carries another substituent, a fused ring and a chain between the ring and the
    # cation have no locant here
    for smiles in ["C[N+](C)(C)c1cc(C)ccc1[O-]", "C[N+](C)(C)c1ccc2ccccc2c1[O-]",
                   "C[N+](C)(C)Cc1ccccc1[O-]"]:
        mol = Chem.MolFromSmiles(smiles)
        sites = get_ion_sites(mol)
        cation, anion = sites["cations"][0]["atom_idx"], sites["anions"][0]["atom_idx"]
        attach = rdmolops.GetShortestPath(mol, cation, anion)[1]
        assert _ring_attachment_locant_on_anion_parent(mol, anion, cation, attach) is None, smiles
