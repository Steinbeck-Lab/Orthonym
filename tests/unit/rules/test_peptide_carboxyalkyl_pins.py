"""Peptides named systematically at the PIN tier (``assembly.decorated_n_branch``).

 (the Blue Book): "Preferred IUPAC names (PINs) for the natural products in Chapter are not identified", so
'aspartylleucylalanine' is no PIN; the PIN is the substitutive name, whose amide nitrogens carry 'carboxyalkyl' groups and
further chain carboxamides:32928, "Carbamoyl and amino groups may be substituted in the normal way").
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    # Asp-Leu-Ala: abstained at the PIN tier (the amide N-branch held a carboxy group)
    ("CC(C)C[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](C)C(=O)O",
     "(3S)-3-amino-4-{[(2S)-1-{[(1S)-1-carboxyethyl]amino}-4-methyl-1-oxopentan-2-yl]amino}-4-oxobutanoic acid"),
    # a middle Glu whose N-terminal amide is reversed (-NH-CO-R on the chain) beside the chain amide (polyfunctional_parts)
    ("C[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "(4S)-4-[(2S)-2-aminopropanamido]-5-{[(1S)-1-carboxy-2-phenylethyl]amino}-5-oxopentanoic acid"),
    ("CC(C)[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "(4S)-4-[(2S)-2-amino-3-methylbutanamido]-5-{[(1S)-1-carboxy-2-phenylethyl]amino}-5-oxopentanoic acid"),
    # rows that already had the PIN keep it
    ("N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(=O)O",
     "(3S)-3-amino-4-({(2S)-1-[(carboxymethyl)amino]-3-(1H-indol-3-yl)-1-oxopropan-2-yl}amino)-4-oxobutanoic acid"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_the_decorated_branch_shapes():
    from rdkit import Chem
    from orthonym.assembly.decorated_n_branch import decorated_alkyl_branch
    mol = Chem.MolFromSmiles("NC(CS)C(=O)O")         # cysteine-like: carboxy, sulfanyl
    assert decorated_alkyl_branch(mol, {1, 2, 3, 4, 5, 6}, 1)
    mol = Chem.MolFromSmiles("NC(CC=C)C(=O)O")       # a C=C on the chain is not decorated alkyl
    assert not decorated_alkyl_branch(mol, {1, 2, 3, 4, 5, 6, 7}, 1)
    mol = Chem.MolFromSmiles("NC(CNC(N)=N)C(=O)O")   # a guanidino group is not in the list
    assert not decorated_alkyl_branch(mol, set(range(1, 9)), 1)


@pytest.mark.parametrize("smiles,expected", [
    # an acyl with a carboxy group keeps the amido form of method (1) (the Blue Book)
    ("OC(=O)CC(NC(=O)CCC(=O)O)C(=O)NCC(=O)O", "4-[(carboxymethyl)amino]-3-(3-carboxypropanamido)-4-oxobutanoic acid"),
    ("OC(=O)CCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O",
     "(4S)-5-[(carboxymethyl)amino]-4-(3-carboxypropanamido)-5-oxopentanoic acid"),
])
def test_a_carboxy_acyl_keeps_its_amido_prefix(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_the_branch_check_is_linear_in_the_chain_length():
    # the recursion follows only each branch's own amide nitrogen (was about 4x per two residues: 16 residues 5.8 s)
    import time
    from rdkit import Chem
    from orthonym.assembly.chain_carboxamide import pin_amide_prefix
    mol = Chem.MolFromSmiles("CC(=O)" + "NCC(=O)" * 16 + "NCC#N")
    start = time.perf_counter()
    pin_amide_prefix(mol, list(range(1, mol.GetNumAtoms())), 1)
    assert time.perf_counter() - start < 1.0
