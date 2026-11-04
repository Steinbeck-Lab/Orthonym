"""Two false PINs of the ester decomposition (lane l16, b1).

L1. The group word of the alcohol half of an ester is the substituent group of the alcohol, and a substituent group is
named by its principal chain. "THE PRINCIPAL SUBSTITUENT CHAIN" (the Blue Book), (b): "the greater number
of skeletal atoms, i.e., the longest chain", which comes before unsaturation, 'In acyclic substituents the order of
seniority between unsaturation and length of chain given in earlier recommendations is reversed' (:22639); so the chain
runs THROUGH the free-valence atom when that atom lies inside the carbon chain: '1-hydroxypropan-2-yl (preferred
prefix) [not 1-(hydroxymethyl)ethyl]' (:22744). The writer of the best-effort group counted the chain from the free
valence ('2-hydroxy-1-methyl-4-phenylbut-3-yn-1-yl'), and the PIN tier shipped that word as a PIN; the PIN is
'3-hydroxy-5-phenylpent-4-yn-2-yl acetate'.

L2. A molecule with two esters is named from the senior one. "Polyester names formed by using
substitutive nomenclature, and/or multiplicative nomenclature, and functional class nomenclature" (:31858): "When
required, seniority of rings, ring systems, and chains; number and location of substituents; and alphanumerical order is
applied in forming the alcoholic component of a functional class ester name (see through ", and the acid
component by "Systems composed of rings and chains" (1): "Within the same class, a ring or ring system has
seniority over a chain". The decomposition cleaved the ester bond that halves the molecule most evenly, whatever its
rank, and named the molecule from an acetate of a ring acid's ester.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.decomposition import bond_cleavage, engine
from orthonym.decomposition.tied_splits import acyl_chain_atoms
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_L1 = "CC(=O)OC(C)C(O)C#Cc1ccccc1"
_L1_PIN = "3-hydroxy-5-phenylpent-4-yn-2-yl acetate"
_L1_BIPHENYL = "CC(=O)OC(C)C(O)C#Cc1ccc(-c2ccccc2)cc1"
_L1_BIPHENYL_PIN = "5-([1,1'-biphenyl]-4-yl)-3-hydroxypent-4-yn-2-yl acetate"
#: a saturated twin: the group the PIN tier already names by its principal chain
_L1_SATURATED = "CC(=O)OC(C)C(O)CC"
_L1_SATURATED_PIN = "3-hydroxypentan-2-yl acetate"
_L2 = r"C/C/1=C\CCC(=C)[C@@H]([C@H](CC1)/C(=C\CC(C(C)(C)O)O)/COC(=O)C)C(=O)OC"


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


def _balance_winner(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bonds = bond_cleavage.find_cleavable_bonds(mol)
    best = engine._select_best_bond(mol, bonds)
    return mol, best


# --- L2: the senior ester is the one the molecule is named from -----------------------------------------------------

def test_the_ring_acid_ester_is_cleaved_before_the_more_even_acetate_split():
    """The two ester bonds of L2: the acetate halves the molecule more evenly, the ring acid's ester is senior."""
    mol, best = _balance_winner(_L2)
    acyl = mol.GetAtomWithIdx(best["acid_atom"])
    assert any(n.GetAtomicNum() == 6 and n.IsInRing() for n in acyl.GetNeighbors()), best


def test_a_longer_acid_chain_is_cleaved_before_a_shorter_one():
    """ "The principal chain has the greater number of skeletal atoms": the butanoate is the parent, though the
    phenylacetate halves the molecule evenly."""
    mol, best = _balance_winner("O=C(Cc1ccccc1)OCCOC(=O)CCC")
    assert acyl_chain_atoms(mol, best["acid_atom"]) == 4, best


def test_the_pin_tier_does_not_ship_the_acetate_parent_of_l2():
    """The acetate-parent name was ``pin_verified``: a false PIN (the ring acid ester is senior)."""
    row = _row(_L2, "pin")
    assert not (row["tier"] == "pin_verified" and row["name"].endswith(" acetate")), row


def test_best_effort_names_l2_from_the_ring_acid():
    """The ring acid is the parent, and its substituent group runs through the free valence (hept-2-en-2-yl, seven
    atoms; the group counted from the free valence was 'hex-1-en-1-yl' with an '(acetyloxy)methyl' branch)."""
    row = _row(_L2, "best-effort")
    assert (row["name"], row["tier"]) == (
        "methyl (1R,2S,5E)-2-[(2E)-1-(acetyloxy)-5,6-dihydroxy-6-methylhept-2-en-2-yl]-5-methyl-9-methylidene"
        "cyclonon-5-ene-1-carboxylate", "pin_unverified"), row
    assert_full_rt(row["name"], _L2)


#: a steroid with an enol acetate and a ring acid methyl ester: the ring acid is the parent
_RING_ACID_ESTER = ("COC(=O)[C@@]12C(OC(C)=O)=C(C)C(=O)[C@]1(C)C(C)=CC1[C@]3(C=O)CC[C@H](OC(C)=O)C(C)(C)C3CC[C@@]12C")
_RING_ACID_ESTER_PIN = ("methyl (3S,8S,10S,13R,14R)-3,15-bis(acetyloxy)-10-formyl-4,4,8,12,13,16-hexamethyl-17-oxo-"
                        "1,2,3,4,5,6,7,8,9,10,13,17-dodecahydro-14H-cyclopenta[a]phenanthrene-14-carboxylate")
#: a 2-methylbutanoate of a ring beside a methyl 3,5-dihydroxy-7-(...)heptanoate: the heptanoate chain (seven atoms)
#: is the longer acid chain
_LONG_ACID_CHAIN = ("CCC(C)C(=O)O[C@H]1C[C@@H](C)[C@@H](O)C2=CC(=O)[C@H](C)[C@H](CC[C@H](O)C[C@H](O)CC(=O)OC)[C@H]21")


def test_a_ring_acid_ester_is_the_parent_at_the_pin_tier():
    row = _row(_RING_ACID_ESTER, "pin")
    assert (row["name"], row["tier"]) == (_RING_ACID_ESTER_PIN, "pin_verified"), row
    assert_full_rt(row["name"], _RING_ACID_ESTER)


def test_the_pin_tier_does_not_ship_the_shorter_acid_chain_as_the_parent():
    row = _row(_LONG_ACID_CHAIN, "pin")
    assert not (row["tier"] == "pin_verified" and row["name"].endswith("2-methylbutanoate")), row


# --- L1: the substituent group is named by its principal chain ------------------------------------------------------

@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_group_word_runs_through_the_free_valence(tier):
    row = _row(_L1, tier)
    assert (row["name"], row["tier"]) == (_L1_PIN, "pin_verified"), row
    assert_full_rt(row["name"], _L1)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_stereocentres_are_cited_at_the_locants_of_the_principal_chain(tier):
    smiles = "CC(=O)O[C@H](C)[C@@H](O)C#Cc1ccccc1"
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == ("(2R,3S)-3-hydroxy-5-phenylpent-4-yn-2-yl acetate", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_ring_assembly_in_the_group_keeps_the_principal_chain(tier):
    """The PIN tier names it with the assembly cited, or declines: it never ships the free-valence chain."""
    row = _row(_L1_BIPHENYL, tier)
    assert "pent-4-yn-2-yl" in row["name"] or row["tier"] == "abstain", row
    if row["tier"] == "pin_verified":
        assert row["name"] == _L1_BIPHENYL_PIN, row
    if row["tier"] != "abstain":
        assert_full_rt(row["name"], _L1_BIPHENYL)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_saturated_group_is_unchanged(tier):
    row = _row(_L1_SATURATED, tier)
    assert (row["name"], row["tier"]) == (_L1_SATURATED_PIN, "pin_verified"), row


# --- the deriver of the group word -----------------------------------------------------------------------------------

def _group_word(hydroxyl_first_smiles):
    """The group word of the fragment of ``hydroxyl_first_smiles`` without its atom 0 (the linking oxygen)."""
    from orthonym.assembly.substituent_naming import _located_acyclic_alkyl_name
    mol = Chem.MolFromSmiles(hydroxyl_first_smiles)
    attach = mol.GetAtomWithIdx(0).GetNeighbors()[0].GetIdx()
    found = _located_acyclic_alkyl_name(mol, list(range(1, mol.GetNumAtoms())), attach, allow_functional=True)
    return None if found is None else found[0]


@pytest.mark.parametrize("smiles,word", [
    ("OC(C)C(O)C#Cc1ccccc1", "3-hydroxy-5-phenylpent-4-yn-2-yl"),
    ("OC(C)C(O)C=CC", "3-hydroxyhex-4-en-2-yl"),
])
def test_an_unsaturated_chain_through_the_free_valence_is_named_by_its_principal_chain(smiles, word):
    """ (b) (:22632): the chain runs through the free valence; (d) the multiple bonds are cited in it."""
    assert _group_word(smiles) == word


@pytest.mark.parametrize("smiles", [
    "OCC(O)C=CC",        # the chain starts at the free valence: the writer of the group names it
    "OC(C)C(C=C)C#CC",   # a vinyl branch the principal chain leaves out
])
def test_the_deriver_declines_what_it_does_not_spell(smiles):
    assert _group_word(smiles) is None
