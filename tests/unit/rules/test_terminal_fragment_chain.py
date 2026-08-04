"""The terminal fragment namer: straight saturated chains.

Every expectation here was OPSIN-verified on 2026-08-04 by wrapping the token in
a parent -- `(2-oxabutyl)benzene` -> `C(OCC)c1ccccc1`. A BARE token does not
parse, so never assert against `opsin("2-oxabutyl")`.

The completeness invariant is the point of the module: `result.atoms` must equal
the input fragment. An atom-short name is the defect this phase exists to remove,
so it is asserted on every case rather than spot-checked.
"""
import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import (
    TerminalFragmentName,
    terminal_fragment_name,
)


def _frag(smiles, attach_smarts_idx=0):
    """Whole molecule as one fragment, attached at atom `attach_smarts_idx`."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, set(range(mol.GetNumAtoms())), attach_smarts_idx


@pytest.mark.parametrize("smiles,expected", [
    # plain alkyl -- no replacement prefix
    ("CC",      "ethyl"),
    ("CCCC",    "butyl"),
    ("C",       "methyl"),
    # one heteroatom: locant 1 is the attachment atom, numbering runs away
    ("COCC",    "2-oxabutyl"),
    ("CSCC",    "2-thiabutyl"),
    ("CNCC",    "2-azabutyl"),
    ("CCOCC",   "3-oxapentyl"),
    # two of a kind -> multiplier
    ("COCOC",   "2,4-dioxapentyl"),
    # two kinds -> cited in seniority order, each with its locant
    ("CSCOCC",  "4-oxa-2-thiahexyl"),
    ("COCNC",   "2-oxa-4-azapentyl"),
    # group 14 heteroatoms are in the admitted table
    ("C[SiH2]CC", "2-silabutyl"),
])
def test_straight_saturated_chain(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.basis == "chain"


@pytest.mark.parametrize("smiles", ["CC", "COCC", "CSCOCC", "C[SiH2]CC"])
def test_completeness_invariant_every_atom_accounted(smiles):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.atoms == frozenset(frag), (
        "an atom-short name is the defect this module exists to remove"
    )


def test_attachment_atom_is_locant_one():
    mol, frag, attach = _frag("COCC")
    got = terminal_fragment_name(mol, frag, attach)
    assert got.numbering[attach] == 1


def test_off_table_element_refuses_rather_than_inventing_a_morpheme():
    """Zn has no row in Table 1.5, so no morpheme can be spelled for it.

    `ring_replacement.build_replacement_prefix` reports it in `unexpressed` and
    its contract obliges the caller to refuse -- inventing 'zna' is how
    `3-znaspiro[5.5]undecane` happened.
    """
    mol = Chem.MolFromSmiles("C[Zn]CC")
    assert mol is not None
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is None


def test_branched_and_ring_fragments_are_declined_in_this_task():
    """Honest intermediate state: unsupported shapes return None, never a
    partial name that drops the unsupported part."""
    for smi in ("CC(C)C", "C1CCCCC1"):
        mol = Chem.MolFromSmiles(smi)
        assert terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0) is None


def test_never_raises_on_degenerate_input():
    mol = Chem.MolFromSmiles("CCO")
    assert terminal_fragment_name(mol, set(), 0) is None
    assert terminal_fragment_name(None, {0}, 0) is None
    assert terminal_fragment_name(mol, {0}, 99) is None
