"""The natural-product detection runs once per molecule graph, not once per Mol object.

The PIN tier's promotion re-run (``namer._name_with_pin_promotion``) names the same SMILES a second
time, which parses it into a new Mol; the per-Mol memo missed there and the detection, the most
expensive step for a giant cage, ran twice (the C70 fullerene: 8 of 15 s). A second memo keyed by the
exact graph -- every atom and bond by index -- returns the first result. Two different molecules, or
the same molecule numbered differently, never share an entry (``matched_atoms`` are atom indices).
"""
from rdkit import Chem

import orthonym.perception.natural_products as NP


def _count_impl(monkeypatch):
    calls = []
    orig = NP._detect_natural_product_impl

    def spy(mol):
        calls.append(Chem.MolToSmiles(mol))
        return orig(mol)
    monkeypatch.setattr(NP, "_detect_natural_product_impl", spy)
    monkeypatch.setattr(NP, "_NP_GRAPH_MEMO", type(NP._NP_GRAPH_MEMO)())
    return calls


def test_same_smiles_new_mol_is_detected_once(monkeypatch):
    calls = _count_impl(monkeypatch)
    smi = "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CCC2=O"   # a steroid
    first = NP.detect_natural_product(Chem.MolFromSmiles(smi))
    second = NP.detect_natural_product(Chem.MolFromSmiles(smi))
    assert len(calls) == 1
    assert first == second and first is not None
    assert first is not second and first["non_scaffold_atoms"] is not second["non_scaffold_atoms"]


def test_other_molecules_and_other_numberings_are_detected_apart(monkeypatch):
    calls = _count_impl(monkeypatch)
    NP.detect_natural_product(Chem.MolFromSmiles("CCO"))
    NP.detect_natural_product(Chem.MolFromSmiles("OCC"))       # same molecule, other numbering
    NP.detect_natural_product(Chem.MolFromSmiles("CC=O"))      # same skeleton, other bonds
    assert len(calls) == 3


def test_the_memo_is_bounded(monkeypatch):
    _count_impl(monkeypatch)
    for n in range(1, NP._NP_GRAPH_MEMO_MAX + 20):
        NP.detect_natural_product(Chem.MolFromSmiles("C" * n))
    assert len(NP._NP_GRAPH_MEMO) == NP._NP_GRAPH_MEMO_MAX
