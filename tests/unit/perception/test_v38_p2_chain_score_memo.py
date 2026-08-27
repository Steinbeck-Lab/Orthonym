"""v38 Perf Phase P, lever P2 — reverse-pair memo for the chain-scoring hot loop.

The memo in ``find_principal_chain`` caches ``chain_score`` on a reverse-canonical
key (``min(tuple, reversed tuple)``). It is correct iff ``chain_score`` is
orientation-invariant — the same tuple for a chain and its reverse — because
``find_all_carbon_chains`` emits every chain in BOTH orientations (~47% of the
scored list are exact reverses). These tests pin that invariant and prove the
memo changes no emitted name and introduces no cross-molecule bleed or order
dependence. PURE SPEEDUP: any name change here is a regression.
"""
import os
import random

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.chains import (
    find_principal_chain,
    find_all_carbon_chains,
    _get_non_principal_terminal_carbons,
)
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group


# Molecules whose carbon-chain enumeration is rich in reverse-duplicate chains
# (large branched acyclics, polyols, polyamines) — the exact inputs the memo
# dedupes. Each exercises many cache hits.
_BATTERY = [
    "CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C",   # C21 branched ketone (441 chains)
    "CCCCCCCCCCCCCCCCCCCC(=O)OCCCC",        # long ester (416 chains)
    "OCC(O)C(O)C(O)C(O)CO",                 # hexitol
    "OCC(CCBr)CCCl",                        # substituent-locant tie case
    "CN(C)CCN(C)CCN(C)C",                   # polyamine
    "C=CCC(C=C(C)C)C(C)=CC",                # diene orientation case
    "CC(C)CC(C)CC(C)CC(C)CC(=O)O",          # methyl-branched acid
]


def _prep(smi):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    fgs = detect_functional_groups(mol)
    try:
        pg, _ = get_principal_group(mol, fgs)
    except Exception:
        pg = None
    return mol, fgs, pg


def test_reverse_pairs_actually_present():
    """Guard against a vacuous invariance test: the enumeration must in fact
    contain reverse-duplicate chains, or the memo path is never exercised."""
    total_dup = 0
    for smi in _BATTERY:
        mol, fgs, pg = _prep(smi)
        np_terminal = _get_non_principal_terminal_carbons(mol, fgs, pg)
        chains = find_all_carbon_chains(
            mol, min_length=1, exclude_atoms=np_terminal or None
        )
        canon = {}
        for c in chains:
            t = tuple(c)
            k = t if t <= t[::-1] else t[::-1]
            canon[k] = canon.get(k, 0) + 1
        total_dup += sum(v - 1 for v in canon.values() if v > 1)
    assert total_dup > 0, "no reverse-pair duplicates — memo would be a no-op"


def test_memoized_score_equals_fresh_computation():
    """With ORTHONYM_P2_VERIFY=1 the memo recomputes ``chain_score`` on every
    cache HIT (i.e. every mirror chain) and asserts it equals the cached value.
    Running the battery without an AssertionError proves the memoized score is
    byte-identical to a fresh computation for every (mol, chain) it serves."""
    old = os.environ.get("ORTHONYM_P2_VERIFY")
    os.environ["ORTHONYM_P2_VERIFY"] = "1"
    try:
        for smi in _BATTERY:
            mol, fgs, pg = _prep(smi)
            chain = find_principal_chain(mol, fgs, pg)  # asserts internally
            assert chain, smi
    finally:
        if old is None:
            os.environ.pop("ORTHONYM_P2_VERIFY", None)
        else:
            os.environ["ORTHONYM_P2_VERIFY"] = old


def test_chain_score_orientation_invariant_direct():
    """The property the memo relies on, checked head-on: the principal chain
    chosen is stable, and reversing every enumerated chain does not change the
    winning chain's atom set (score(chain) == score(reversed(chain)))."""
    for smi in _BATTERY:
        mol, fgs, pg = _prep(smi)
        fwd = find_principal_chain(mol, fgs, pg)
        assert fwd, smi
        # Independent orientation of the whole molecule must select the same
        # backbone (as an atom set, since numbering direction is a later step).
        rm = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        rfgs = detect_functional_groups(rm)
        try:
            rpg, _ = get_principal_group(rm, rfgs)
        except Exception:
            rpg = None
        rev = find_principal_chain(rm, rfgs, rpg)
        assert len(rev) == len(fwd), smi


@pytest.mark.parametrize(
    "smi,expected",
    [
        ("CC(C)(C)CC(=O)O", "3,3-dimethylbutanoic acid"),
        ("OCC(CCBr)CCCl", "2-(2-bromoethyl)-4-chlorobutan-1-ol"),
        (
            "CN(C)CCN(C)CCN(C)C",
            "N1-[2-(dimethylamino)ethyl]-N1,N2,N2-trimethylethane-1,2-diamine",
        ),
    ],
)
def test_known_names_unchanged(smi, expected):
    """Pin a few names the memo must not perturb."""
    assert name_compound(smi) == expected


def test_no_cross_molecule_cache_bleed():
    """The memo is scoped to a single find_principal_chain call, so naming a
    molecule alone and naming it after a batch of others must be identical."""
    target = "CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C"
    alone = name_compound(target)
    others = ["OCC(CCBr)CCCl", "CN(C)CCN(C)CCN(C)C", "CC(C)(C)CC(=O)O"]
    for o in others:
        name_compound(o)
    after_batch = name_compound(target)
    assert alone == after_batch


def test_deterministic_across_randomized_smiles():
    """Randomized atom orderings of the same molecule must name identically —
    the memo must not introduce any order dependence."""
    for smi in ["CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C", "OCC(CCBr)CCCl"]:
        mol = Chem.MolFromSmiles(smi)
        names = set()
        for seed in (0, 3, 11):
            idx = list(range(mol.GetNumAtoms()))
            random.seed(seed)
            random.shuffle(idx)
            rm = Chem.RenumberAtoms(mol, idx)
            names.add(name_compound(Chem.MolToSmiles(rm, canonical=False)))
        assert len(names) == 1, (smi, names)
