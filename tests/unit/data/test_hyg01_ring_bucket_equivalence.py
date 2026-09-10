""" (a phase) — HARD equivalence gate.

Asserts the new hash-bucketed match_fused_heterocycle_core returns results
byte-identical to the original O(N) full-scan over: every catalog entry, every
catalog entry + a methyl decoration, and a corpus sample of ring-bearing SMILES.
The reference _old_full_scan below is a verbatim copy of the pre-Phase-173 loop.
"""
import csv
import os
import warnings

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    match_fused_heterocycle_core,
    ring_skeleton_key,
    _get_substructure_patterns,
    _build_pattern_index,
    _PATTERN_BUCKETS,
)


def _old_full_scan(mol):
    """Verbatim pre-Phase-173 implementation (reference oracle)."""
    if mol is None:
        return None
    patterns = _get_substructure_patterns()
    best_match = None
    for smiles, pattern in patterns.items():
        if mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                match = matches[0]
                data = FUSED_HETEROCYCLE_DATA[smiles]
                core_size = data['parent_atoms']
                if best_match is None or core_size > best_match[2]:
                    best_match = (data['name'], list(match), core_size, smiles)
    if best_match is None:
        return None
    name, match_atoms, _, core_smiles = best_match
    data = FUSED_HETEROCYCLE_DATA[core_smiles]
    iupac_locants = data.get('iupac_locants')
    if iupac_locants is None:
        return None
    atom_mapping = {}
    for pattern_idx, mol_atom_idx in enumerate(match_atoms):
        iupac_locant = iupac_locants.get(pattern_idx)
        if iupac_locant is not None:
            atom_mapping[mol_atom_idx] = iupac_locant
    return (name, atom_mapping, core_smiles)


def _assert_equiv(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _old_full_scan(mol)
        new = match_fused_heterocycle_core(mol)
    # Normalise dict identity (atom_mapping) for comparison.
    def norm(r):
        if r is None:
            return None
        return (r[0], dict(r[1]), r[2])
    assert norm(new) == norm(old), f"divergence for {smi!r}:\n old={norm(old)}\n new={norm(new)}"


CATALOG_SMILES = list(FUSED_HETEROCYCLE_DATA.keys())


@pytest.mark.parametrize("smi", CATALOG_SMILES)
def test_equiv_catalog_unsubstituted(smi):
    _assert_equiv(smi)


def _methylate(smi):
    """Attach a methyl to the first substitutable ring atom (best-effort)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
            c = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(atom.GetIdx(), c, Chem.BondType.SINGLE)
            try:
                m2 = rw.GetMol()
                Chem.SanitizeMol(m2)
                return Chem.MolToSmiles(m2)
            except Exception:
                return None
    return None


@pytest.mark.parametrize("smi", CATALOG_SMILES)
def test_equiv_catalog_methylated(smi):
    msmi = _methylate(smi)
    if msmi:
        _assert_equiv(msmi)


def _corpus_sample(n=120):
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'benchmarks', 'chebi_5000.csv')
    path = os.path.normpath(path)
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            smi = row[1].strip()
            if not smi or '*' in smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.GetRingInfo().NumRings() > 0:
                out.append(smi)
            if len(out) >= n:
                break
    return out


@pytest.mark.parametrize("smi", _corpus_sample())
def test_equiv_corpus_sample(smi):
    _assert_equiv(smi)


def test_bucket_index_built_and_nonempty():
    _build_pattern_index()
    assert _PATTERN_BUCKETS, "bucket index should be populated"
    total = sum(len(v) for v in _PATTERN_BUCKETS.values())
    # parseable catalog entries (a couple may be unparseable and skipped)
    assert total >= len(FUSED_HETEROCYCLE_DATA) - 5


def test_acyclic_returns_none_without_scan():
    # An acyclic molecule must short-circuit to None (no ring patterns can match).
    assert match_fused_heterocycle_core(Chem.MolFromSmiles('CCCCO')) is None


def test_ring_skeleton_key_basic():
    # Substituted indole's ring skeleton canonicalises to indole's key.
    indole_key = ring_skeleton_key(Chem.MolFromSmiles('c1ccc2[nH]ccc2c1'))
    methylindole_key = ring_skeleton_key(Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1'))
    assert indole_key == methylindole_key
    assert ring_skeleton_key(Chem.MolFromSmiles('CCO')) is None
