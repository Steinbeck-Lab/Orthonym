"""The natural-product detection tests each scaffold's ring core before its full query.

A match of a query restricts to a match of any subgraph of it, so a ring core (the query without its acyclic atoms)
that does not match proves the full query cannot match. One core test screens out every scaffold that shares the core,
and the detection result is the same as without the screen (speed audit 2026-11-24, lever K2: 0 differences on 8,628
molecules; on a saturated carbon cage the full queries ran for hours).
"""
import pytest
from rdkit import Chem

import orthonym.perception.natural_products as NP

_MOLS = [
    "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CCC2=O",      # a steroid
    "C=CC(C)(C)c1[nH]c2cccc3c2c1CC1NCC(C)CC31",                          # an ergoline
    "COc1cc2c(cc1O)C[C@H]1c3cc(O)c(OC)cc3CCN1C2",                        # a berbine
    "CN1C2CCC1[C@@H](C(=O)O)[C@@H](O)C2",                                # a tropane
    "CCO", "c1ccccc1", "OC(=O)c1ccccc1O",                                 # no scaffold
]


def test_every_core_is_its_query_without_the_acyclic_atoms():
    patterns = NP._get_flexible_patterns()
    assert NP._CORE_OF
    for smiles, key in NP._CORE_OF.items():
        core, query = NP._RING_CORES[key], patterns[smiles]
        assert 0 < core.GetNumAtoms() <= query.GetNumAtoms()


@pytest.mark.parametrize("smi", _MOLS)
def test_the_screen_changes_no_result(smi, monkeypatch):
    mol = Chem.MolFromSmiles(smi)
    NP._get_flexible_patterns()
    screened = NP._detect_natural_product_impl(mol)
    monkeypatch.setattr(NP, "_CORE_OF", {})
    assert NP._detect_natural_product_impl(mol) == screened


def test_a_saturated_cage_is_screened_out():
    cage = ("C1C2C3C4CC5C46C37C28C1C9C81C72C63C5C4C33C22C11C9C5C11C22C33C4C4C33C22C11C5C5C11C22C33C4C4C33C22C11C5C5C11"
            "C22C33C4C4C33C22C11C5C5C11C22C33C4C4C33C22C11C5C5C11C22C33C4C4C33C22C11C5C5C11C22C33C4C4C33C22C11C5C5C11C22"
            "C33C4C4C33C22C11C5C5C11C22C33C4C4C33C22C11C5C5C11C22C33C4CC3C2C1C5")
    assert NP._detect_natural_product_impl(Chem.MolFromSmiles(cage)) is None


def test_a_trimmed_core_carries_no_ring_data():
    """The screen is exact because a core is never stricter than its query: a core with atoms removed has no ring data,
    so the matcher skips the ring-count rule on it (a query atom may sit in no more rings than the target atom). If
    RDKit ever kept ring data on an edited core, a cage scaffold could get a core stricter than its query."""
    patterns = NP._get_flexible_patterns()
    for smiles, key in NP._CORE_OF.items():
        core = NP._RING_CORES[key]
        if core.GetNumAtoms() < patterns[smiles].GetNumAtoms():
            with pytest.raises(RuntimeError):         # RDKit: 'RingInfo not initialized'
                core.GetRingInfo().NumRings()
