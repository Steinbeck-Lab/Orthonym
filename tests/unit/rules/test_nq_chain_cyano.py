"""Roadmap item 12a (task t4): a nitrile carbon is a 'cyano' branch, not a chain member.

The nitrile carbon of R-C#N was walked into the chain spine and its nitrogen left as the off-spine
'1-azamethan-1-ylidyne' (an 'a' chain ending on N,, the Blue Book, section
 General rules). "When a group is present that has priority for citation as the principal
characteristic group..., the –CN group is designated by the preferred prefix 'cyano'"
,:34734). Only a carbon R qualifies: a cyanate or a thiocyanate has its own prefixes
,:30981; not built) and cyanogen is no 'cyano'.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.universal_substituent import name_universal_substitutive


def test_a_nitrile_carbon_is_a_cyano_branch_not_a_chain_member():
    mol = Chem.MolFromSmiles("CC(C#N)CCC(=O)O")
    assert name_universal_substitutive(mol).name == "4-cyano-1-hydroxy-1-oxopentane"
    # the mechanical spelling threads the nitrile into the chain, its nitrogen as the end
    assert name_universal_substitutive(mol, book_forms=False).name == \
        "2-hydroxy-5-methyl-1-oxa-7-azahept-1-en-6-yne"
    mol = Chem.MolFromSmiles("N#CC(OC(=O)C1CC1)c1ccccc1")
    assert name_universal_substitutive(mol).name == \
        "{cyano[(cyclopropanecarbonyl)oxy]methyl}benzene"


@pytest.mark.parametrize("smiles,name", [
    ("N#CS", "(1-azamethan-1-ylidyne)(sulfanyl)methane"),
    ("COC#N", "(1-azamethan-1-ylidyne)(methoxy)methane"),
    ("N#C[S-]", "(1-azamethan-1-ylidyne)(sulfido)methane"),
    ("N#CC#N", "bis(1-azamethan-1-ylidyne)ethane"),
])
def test_a_cyanate_a_thiocyanate_and_cyanogen_keep_their_spelling(smiles, name):
    assert name_universal_substitutive(Chem.MolFromSmiles(smiles)).name == name


def test_the_chain_rung_keeps_the_methyl_spelling():
    """Under ``mechanical_forms(keep_forms={'chain'})`` the one-carbon parent is 'cyanomethyl',
    not 'cyanomethan-1-yl' (a), the Blue Book)."""
    from orthonym.assembly.book_prefixes import mechanical_forms
    mol = Chem.MolFromSmiles("N#CCc1ccccc1")
    with mechanical_forms(keep_forms=frozenset({"chain"})):
        assert "cyanomethyl" in name_universal_substitutive(mol, book_forms=False,
                                                            keep_forms=frozenset({"chain"})).name
