"""The string converter of a capped fragment's amidine name to a prefix (``substituent_naming.parent_to_prefix``).

An N-substituted amidine keeps its N-substituent in the fragment's name ("N'-hydroxymethanimidamide"); the string
forms dropped it ('carbamimidoyl' for -C(=NOH)NH2) or counted its carbon into the chain ('carbamimidoylmethyl' for
-C(=NCH3)NH2): other molecules. names the substituted group from the structure; the converter declines
and a structure-based tier names it.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix


@pytest.mark.parametrize("name, carbons, prefix", [
    ("methanimidamide", 1, "carbamimidoyl"),
    ("ethanimidamide", 2, "carbamimidoylmethyl"),
    ("N'-hydroxymethanimidamide", 1, None),
    ("N'-methylmethanimidamide", 2, None),
    ("propanimidamide", 3, None),
])
def test_amidine_prefix(name, carbons, prefix):
    assert parent_to_prefix(name, chain_length=carbons, attach_locant=ATTACH_LOCANT_UNKNOWN) == prefix


@pytest.mark.parametrize("smiles, prefix", [
    ("ON=C(N)[As]C", "amino(hydroxyimino)methyl"),
    ("CN=C(N)[As]C", "amino(methylimino)methyl"),
    ("NC(=N)[As]C", "carbamimidoyl"),
])
def test_the_ligand_namer_keeps_the_n_substituent(smiles, prefix):
    from orthonym.assembly import substituent_enumerator
    from orthonym.rules import centre_ligands
    from orthonym.rules.ions import _collect_substituent_atoms
    tokens = [ctx.set(True) for ctx in centre_ligands._breadth_contexts()]
    try:
        mol = Chem.MolFromSmiles(smiles)
        centre = next(a for a in mol.GetAtoms() if a.GetSymbol() == "As")
        group = next(n for n in centre.GetNeighbors() if n.GetDegree() > 1)
        frag = sorted(_collect_substituent_atoms(mol, centre.GetIdx(), group.GetIdx(), set()))
        assert substituent_enumerator.name_substituent(mol, frag, group.GetIdx(), allow_mancude=False) == prefix
    finally:
        for ctx, token in zip(centre_ligands._breadth_contexts(), tokens):
            ctx.reset(token)
