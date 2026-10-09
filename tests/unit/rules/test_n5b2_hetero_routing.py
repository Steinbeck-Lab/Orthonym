"""Every tier names a fused ring system with ring heteroatoms by fusion nomenclature (lane N5b step 2).

 (the Blue Book, "Five-membered ring requirement"): a von Baeyer descriptor is
never the preferred name of an ortho- or ortho- and peri-fused system with two rings of five or
more members, with or without ring heteroatoms. The default tier ships the fusion name as a
certified PIN and best-effort gives the same name; both read back to the input's full InChIKey.
The older hetero producers (``name_fused_heterocycle`` and the retained names) still run first,
and so does the natural-product route: every name they gave stays byte-identical."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (smiles, name); the von Baeyer name each had before is in the comment
ROUTED = [
    ("C1CC2CCOC2O1", "hexahydrofuro[2,3-b]furan"),                              # 2,8-dioxabicyclo[3.3.0]octane
    ("OC1CC2CCCCC2O1", "octahydro-1-benzofuran-2-ol"),                          # 7-oxabicyclo[4.3.0]nonan-8-ol
    ("C1CCC2OCCC2C1", "octahydro-1-benzofuran"),
    ("O=C1CN=C(c2ccccc2)c2ccccc2N1", "5-phenyl-1,3-dihydro-2H-1,4-benzodiazepin-2-one"),
    ("O=C1CN=C(c2ccccc2Cl)c2cc([N+](=O)[O-])ccc2N1",
     "5-(2-chlorophenyl)-7-nitro-1,3-dihydro-2H-1,4-benzodiazepin-2-one"),
    ("OC12CCOC1COC2", "tetrahydrofuro[3,4-b]furan-3a(4H)-ol"),                  # 2,7-dioxabicyclo[3.3.0]octan-5-ol
    ("C1COC2OCCC2O1", "hexahydrofuro[2,3-b][1,4]dioxine"),
    ("C1CCC2=C(C1)OC1=C2CCCC1", "1,2,3,4,6,7,8,9-octahydrodibenzo[b,d]furan"),
    ("C1CCc2c(C1)oc1ccccc21", "1,2,3,4-tetrahydrodibenzo[b,d]furan"),
    # a ring system as a '-yl' prefix
    ("OC(=O)CCc1ccc2OCOc2c1", "3-(2H-1,3-benzodioxol-5-yl)propanoic acid"),     # 7,9-dioxabicyclo[4.3.0]nona-1,3,5-trien-4-yl
    ("OC(=O)CC1CCC2OCCC2C1", "(octahydro-1-benzofuran-5-yl)acetic acid"),       # (7-oxabicyclo[4.3.0]nonan-3-yl)acetic acid
]


@pytest.mark.parametrize("smiles,expected", ROUTED)
def test_both_tiers_give_the_fusion_name(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


# names right before this lane (the older hetero producers); they stay byte-identical at both tiers
UNCHANGED = [
    ("c1ccc2ncccc2c1", "quinoline"),
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("c1ccc2c(c1)[nH]c1ccccc12", "9H-carbazole"),
    ("O=c1ccc2ccccc2o1", "2H-1-benzopyran-2-one"),
    ("c1ccc2c(c1)OCO2", "2H-1,3-benzodioxole"),
    ("CN1CCc2ccccc21", "1-methyl-2,3-dihydro-1H-indole"),
    ("O=C1CCOc2ccccc12", "2,3-dihydro-4H-1-benzopyran-4-one"),
    ("OC(=O)CC1CCc2ccccc2O1", "(3,4-dihydro-2H-1-benzopyran-2-yl)acetic acid"),
    ("OC(=O)Cc1ccc2c(c1)OCCO2", "(2,3-dihydro-1,4-benzodioxin-6-yl)acetic acid"),
    ("CC(=O)OC1CCc2ccccc2O1", "3,4-dihydro-2H-1-benzopyran-2-yl acetate"),
    ("O=C1C=CC=C2Oc3ccccc3N=C12", "1H-phenoxazin-1-one"),
    ("OC(=O)CC1Cc2ccccc2O1", "(2,3-dihydro-1-benzofuran-2-yl)acetic acid"),
]


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_names_that_were_right_stay_byte_identical(smiles, expected):
    assert default_tier_row(smiles)["name"] == expected
    assert name_best_effort(smiles)["name"] == expected


# the natural-product route parents,:50943 "Preferred IUPAC names (PINs) are not
# identified for the compounds in this Chapter") keeps these names at the default tier
NATURAL_PRODUCT_PARENTS = [
    ("CC[C@H]1C[C@H]2c3[nH]c4ccccc4c3CCN2C[C@@H]1CC", "corynan"),
    ("c1ccc2c3c([nH]c2c1)[C@@H]1C[C@@H]2CCCC[C@H]2CN1CC3", "yohimban"),
    ("C1CCN2C[C@@H]3C[C@@H](CN4CCCC[C@@H]34)[C@H]2C1", "sparteine"),
    ("CC[C@H]1C[C@@H]2C[C@H]3c4[nH]c5ccccc5c4CCN(C2)[C@@H]13", "ibogamine"),
    ("CC[C@@]12CCCN3CC[C@]4(c5ccccc5N[C@@H]4CC1)[C@H]32", "aspidospermidine"),
    ("c1ccc2c(c1)CC[C@@]13CCCC[C@@]21CCN3", "hasubanan"),
    ("CC[C@]12CCCN3CCc4c(n(c5ccccc45)CC1)[C@@H]32", "vincane"),
    ("CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
     "3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano[1]benzofuro[3,2-e]isoquinoline-7,9-diol"),
]


@pytest.mark.parametrize("smiles,expected", NATURAL_PRODUCT_PARENTS)
def test_natural_product_parents_keep_winning(smiles, expected):
    assert default_tier_row(smiles)["name"] == expected


# retained non-PIN names of the older hetero producer win, as before: the default tier declines
# them (NO_VERIFIED_PIN) and best-effort keeps the name. Listed as a divergence, not changed here.
@pytest.mark.parametrize("smiles,retained", [
    ("OC1CCC2CCCCN2C1", "quinolizidin-3-ol"),
    ("O=C1CCC2CCCN12", "pyrrolizidin-3-one"),
    ("OC(=O)CC1CCC2CCCN2C1", "(indolizidin-6-yl)acetic acid"),
])
def test_a_retained_non_pin_name_still_comes_first(smiles, retained):
    assert name_best_effort(smiles)["name"] == retained
    assert default_tier_row(smiles)["tier"] == "abstain"


# isotopically labelled inputs with a hetero ring system: the label is stripped before naming and
# spliced back in (``rules/isotopes``), and OPSIN 2.9.0 cannot read '(2-2H)' on the nitrogen of an
# indicated-hydrogen fusion name, so the builder leaves such a molecule to the von Baeyer path
@pytest.mark.parametrize("smiles,expected", [
    ("[2H]N1CC2CCCCC2C1", "(8-2H)-8-azabicyclo[4.3.0]nonane"),
    ("[2H][C@@]12[C@@H](O1)CCC3=CC=CC=C23", "(2R,4S)-(2-2H)-3-oxatricyclo[5.4.0.0^2,4]undeca-1(11),7,9-triene"),
    ("OC(=O)CC1CCc2ccc([2H])cc2O1", "[3,4-dihydro(7-2H)-2H-1-benzopyran-2-yl]acetic acid"),
    ("CN1CCc2cc([2H])ccc21", "1-methyl-2,3-dihydro(5-2H)-1H-indole"),
])
def test_isotopic_hetero_inputs_keep_their_name(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles", [
    "O=C1OC2=C(C=CC=C2)C(=O)c2ccccc12",         # three components: no fusion parent
    "CC1=CC(=O)c2c(C)c3[o+]c4ccccc4cc3cc2O1",   # ring cation
])
def test_declined_classes_never_get_a_wrong_name(smiles):
    row = default_tier_row(smiles)
    if row["name"] and not row["name"].startswith("unknown"):
        assert_full_rt(row["name"], smiles)
    be = name_best_effort(smiles)
    if be["name"] and not be["name"].startswith("unknown"):
        assert_full_rt(be["name"], smiles)


def test_a_ring_assembly_of_identical_hetero_ring_systems_is_not_named_by_one_component():
    """ (the Blue Book): identical ring systems joined by a bond are a ring assembly,
    a parent of its own,:19461): '2'H-1,2'-biindole (PIN)'. The fusion name of one
    component with the other as a prefix ('1-(2H-indol-2-yl)-1H-indole') is not the PIN, so the
    builder does not name it and the default tier does not label it pin_verified."""
    from rdkit import Chem
    from orthonym.rules.bridged_fused import joins_identical_hetero_ring_systems
    smiles = "C1=c2ccccc2=NC1n1ccc2ccccc21"
    mol = Chem.MolFromSmiles(smiles)
    from orthonym.rules.bridged_fused_pin import selection
    assert joins_identical_hetero_ring_systems(mol, selection.ring_system(mol))
    assert not joins_identical_hetero_ring_systems(
        Chem.MolFromSmiles("C1CCC2CCCC2C1.C1CCC2CCCC2C1"), range(9))
    assert default_tier_row(smiles)["tier"] != "pin_verified"
    assert default_tier_row("c1ccc2oc(-c3cc4ccccc4o3)cc2c1")["name"] == "2,2'-bi-1-benzofuran"


# The assembly test is scoped to the ring system being named: a phenyl on a benzene ring, or a
# hetero ring elsewhere in the molecule, does not stop the fusion name of a carbocyclic system
#, the Blue Book); the fusion name of the prefix stands. The benzene ring that
# carries the phenyl is a member of a biphenyl assembly, which is the senior parent
# "Retained names",:27722-27724: '[not 1-methoxy-4-phenylbenzene; the biphenyl ring system is
# senior to a single benzene ring]';,:19461), so these benzene-parent names are kept at
# every tier below the PIN and the default tier declines them (lane W2, item 26).
SCOPED = [
    ("OC(=O)c1cc(-c2ccccc2)cc(C2CC3CCCCC3C2)c1N1CCOCC1",
     "2-(morpholin-4-yl)-3-(octahydro-1H-inden-2-yl)-5-phenylbenzoic acid"),
    ("NC(=O)c1cc(-c2ccccc2)cc(C2CC3CCCCC3C2)c1C1CCOC1",
     "3-(octahydro-1H-inden-2-yl)-2-(oxolan-3-yl)-5-phenylbenzamide"),
]


@pytest.mark.parametrize("smiles,expected", SCOPED)
def test_a_carbocyclic_fused_prefix_is_not_stopped_by_a_hetero_ring_elsewhere(smiles, expected):
    from tests.support.default_tier import declined_pin_row
    row = declined_pin_row(smiles)      # default tier declines; best-effort gives the same name, read back
    assert row["name"] == expected and row["tier"] != "pin_verified", row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


def test_the_octahydroindenyl_prefix_survives_at_best_effort_beside_a_hetero_ring():
    smiles = "OC(=O)c1cc(C2CC3CCCCC3C2)ccc1-c1ccc(N2CCOCC2)cc1"
    name = name_best_effort(smiles)["name"]
    assert "octahydro-1H-inden-2-yl" in name, name
    assert_full_rt(name, smiles)


def test_the_assembly_test_is_scoped_to_the_named_ring_system():
    from rdkit import Chem
    from orthonym.rules.bridged_fused import joins_identical_hetero_ring_systems
    mol = Chem.MolFromSmiles("OC(=O)c1cc(-c2ccccc2)cc(C2CC3CCCCC3C2)c1N1CCOCC1")
    indane = {a for a in range(mol.GetNumAtoms())
              if mol.GetAtomWithIdx(a).IsInRing() and not mol.GetAtomWithIdx(a).GetIsAromatic()
              and mol.GetAtomWithIdx(a).GetAtomicNum() == 6}
    assert not joins_identical_hetero_ring_systems(mol, indane)


def test_a_lone_pair_stereocentre_in_the_ring_system_keeps_the_older_producers():
    """A P(III) stereocentre of the ring system: the descriptor the builder takes from the working
    molecule can be the other configuration than the input's, the engine's exit check
    (``_exit_lone_pair_check``) withdraws the whole name and, with the older route pre-empted, the
    row fell to the mechanical spellings ('2-methyl-1-oxapropan-1-yl', 'cyclohexa-1,3,5-trien-1-yl').
    The builder declines and the book-spelled name of the older route stands."""
    from rdkit import Chem
    from orthonym.rules.bridged_fused_pin import build_fused
    smiles = "CC(C)O[P@]1N(c2ccccc2)C[C@@H]2CCCN21"
    assert build_fused(Chem.MolFromSmiles(smiles)) is None
    be = name_best_effort(smiles)
    assert "cyclohexa-1,3,5-trien-1-yl" not in be["name"] and "oxapropan" not in be["name"], be
    assert be["name"] == "(2R,5S)-2-(1-methylethoxy)-3-phenyl-1,3-diaza-2-phosphabicyclo[3.3.0]octane"
    assert be.get("prefix_order_fallback") is False
    assert_full_rt(be["name"], smiles)
