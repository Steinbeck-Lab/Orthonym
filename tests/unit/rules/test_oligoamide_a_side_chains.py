"""Asn, Gln and Arg side chains on the skeletal replacement ('a') PIN of a peptide acid (rules/skeletal_chain).

## **** Skeletal replacement ('a') nomenclature in acyclic chains (the Blue Book), (:23428) gives
the 'a' PIN of a chain of four or more skeletal amide nitrogens under an acid. Its side chains are prefixes:

- "Substituents of the types -CO-NH2 and -CO-CO-NH2" (:32928): "method (1) is preferred for chains",
  the prefixes 'amino' and 'oxo' on the terminal atom of the carbon chain: '5-(2-amino-2-oxoethyl)furan-2-carboxylic
  acid (PIN)' (:32940), not 'carbamoylmethyl'.
- (:34266): 'carbamimidoylamino (preferred prefix)' for H2N-C(=NH)-NH- and '(diaminomethylidene)amino
  (preferred prefix)' for (H2N)2C=N- ('4-[(diaminomethylidene)amino]butanoic acid (PIN)',:34282): the tautomer as drawn.
"""
import pytest
from rdkit import Chem

from orthonym.rules.skeletal_chain import name_acyl_a_chain
from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("NCC(=O)N[C@@H](CC(N)=O)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-(2-amino-2-oxoethyl)-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("NCC(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-(3-amino-3-oxopropyl)-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    # an Arg whose side chain holds fewer chain heteroatoms than the backbone, the Blue Book)
    ("NCC(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-17-amino-11-[3-(carbamimidoylamino)propyl]-4,7,10,13,16-pentaoxo-3,6,9,12,15-pentaazaheptadecan-1-oic acid"),
    ("NCC(=O)NCC(=O)N[C@@H](CCCN=C(N)N)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-17-amino-11-{3-[(diaminomethylidene)amino]propyl}-4,7,10,13,16-pentaoxo-3,6,9,12,15-pentaazaheptadecan-1-oic "
     "acid"),
])
def test_side_chain_pendants_keep_the_a_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # a stereocentre of a side chain (Ile, Thr) takes its descriptor inside the prefix's enclosing marks, labelled in
    # the whole molecule, the Blue Book)
    ("NCC(=O)N[C@@H]([C@@H](C)CC)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-[(2S)-butan-2-yl]-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("NCC(=O)N[C@@H]([C@@H](C)O)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-[(1R)-1-hydroxyethyl]-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CCC(=O)N)C(=O)O)NC(=O)CN",
     "(2S,8S,11S)-14-amino-2-(3-amino-3-oxopropyl)-11-[(2S)-butan-2-yl]-8-methyl-4,7,10,13-tetraoxo-"
     "3,6,9,12-tetraazatetradecan-1-oic acid"),
    # an acyl cap is a chain part too (the acetyl carbons end the chain, more skeletal atoms)
    ("CC(=O)N[C@@H]([C@@H](C)CC)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-11-[(2S)-butan-2-yl]-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
])
def test_side_chain_stereocentres_keep_the_a_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_a_block_on_a_prefix_is_no_a_parent():
    """With an acid present the 'a' block must carry the acid suffix, the Blue Book): an 'a' block on an
    amido prefix of a substitutive acid leaves the chain of the structure unnamed as the 'a' parent."""
    from orthonym.validation.spelling.checks_skeletal_chain import a_chain_name_check, names_an_a_acid_parent
    assert names_an_a_acid_parent("4,7,10,13-tetraoxo-3,6,9,12-tetraazapentadecane-1,15-dioic acid")
    assert names_an_a_acid_parent("3,6,9,12,15,18,21,24,27-nonaoxahentriacontan-1-yl acetate")
    assert names_an_a_acid_parent("2,5,8,11-tetraoxatridecan-13-yl 2-methylprop-2-enoate")
    assert names_an_a_acid_parent("3,6,9,12-tetraoxatetradecane-1,14-diyl diacetate")
    assert names_an_a_acid_parent("5,8,11,14-tetraoxapentadec-2-enoic acid")
    assert not names_an_a_acid_parent("2-[(2,5,8,11-tetraoxatridecan-13-yl)amino]propanoic acid")
    prefix_only = ("(2S)-2-[(5S)-14-amino-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-amido]-3-phenylpropanoic acid")
    assert not names_an_a_acid_parent(prefix_only)
    mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)N[C@@H](Cc1ccccc1)C(=O)O")
    assert a_chain_name_check(mol, prefix_only) is not None


@pytest.mark.parametrize("smiles", [
    "NCC(=O)NC(C(=N)N)C(=O)NCC(=O)NCC(=O)NCC(=O)O",           # an amidine on a chain carbon
    "NCC(=O)NC(CC(=O)NC)C(=O)NCC(=O)NCC(=O)NCC(=O)O",         # a secondary amide in a side chain
    "NCC(=O)NC(CCCNC(=N)NC)C(=O)NCC(=O)NCC(=O)NCC(=O)O",      # a substituted guanidine
    "CC[C@H](C)OC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CN",         # a stereocentre in the ester part
    # an Arg near a chain end: the chain through its side chain to the guanidine carbon (its N beside the C=NH is a
    # chain atom, '9-imino-2,8,10,16-tetraazaheptadecanediimidamide (PIN)', the Blue Book) is as rich in
    # heteroatoms and longer or as long,:20916); this writer does not build it, so it declines
    "CC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
    "NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
    "NCC(=O)N[C@@H](CCCN=C(N)N)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
])
def test_other_c_n_groups_and_amides_in_side_chains_decline(smiles):
    assert name_acyl_a_chain(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles,name", [
    # the alcohol word of an ester is an 'a' parent too, whatever the acid word starts with (review finding)
    ("COCCOCCOCCOCCOC(=O)C(C)=C", "2,5,8,11-tetraoxatridecan-13-yl 2-methylprop-2-enoate"),
    ("COCCOCCOCCOCCOC(=O)CCCl", "2,5,8,11-tetraoxatridecan-13-yl 3-chloropropanoate"),
])
def test_an_a_alcohol_word_keeps_the_ester_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
