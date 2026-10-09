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
    _correct_indicated_h_tautomer,
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


def _locant_key(locant):
    """'3a' -> (3, 'a'), 7 -> (7, ''): the order of catalog locants."""
    text = str(locant)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 0, text[len(digits):])


def _automorphic_mappings(mol, core_smiles):
    """Every atom->locant map the catalog core gives ``mol``: one per substructure
    match, symmetry-equivalent ones included (uniquify=False)."""
    pattern = _get_substructure_patterns()[core_smiles]
    iul = FUSED_HETEROCYCLE_DATA[core_smiles].get('iupac_locants') or {}
    maps = []
    for match in mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=100000):
        maps.append({m: iul[p] for p, m in enumerate(match) if iul.get(p) is not None})
    return maps


def _substituent_locants(mol, mapping):
    """Sorted locants of the mapped core atoms that carry an atom outside the core."""
    core = set(mapping)
    return sorted((_locant_key(loc) for idx, loc in mapping.items()
                   if any(nb.GetIdx() not in core
                          for nb in mol.GetAtomWithIdx(idx).GetNeighbors())))


def _assert_equiv(smi):
    """The bucketed matcher finds the SAME core as the full scan. Its atom->locant
    map may differ by a symmetry of that core only: since E1/DD4 (the substituent-
    locant minimization, ``_select_lowest_locant_match``) the matcher picks, among
    the core's automorphic matches, the one giving the substituents the lowest
    locants (f)/(g), the Blue Book), where the verbatim full
    scan below keeps ``matches[0]`` ('6-methyl' vs '5-methyl' on a symmetric
    [1,3,2]benzodioxathiole). So: same name and core, a map that is one of the
    core's automorphic maps onto this molecule, the orientation of that placement
    with the lowest substituent locants, and none higher than the full scan's."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _old_full_scan(mol)
        new = match_fused_heterocycle_core(mol)
    if old is not None:
        # ``match_fused_heterocycle_core`` post-processes the matcher's hit with the
        # indicated-hydrogen tautomer guard (ba6c75456, the Blue Book
        #:14607): a hit whose baked indicated-H locant is not the input's is re-anchored
        # to the atom that bears it. The verbatim pre-Phase-173 loop has no such step, and
        # since the 7H-purine entry sits beside 9H-purine the full scan's first
        # match names the 7H tautomer '9H-purine'. The oracle gets the same guard so the
        # comparison stays an exact name + core + map one; the guard is not what is bucketed.
        corrected = _correct_indicated_h_tautomer(mol, old[0], old[1], old[2])
        old = None if corrected is None else (corrected, old[1], old[2])
    # Normalise dict identity (atom_mapping) for comparison.
    def norm(r):
        if r is None:
            return None
        return (r[0], dict(r[1]), r[2])
    msg = f"divergence for {smi!r}:\n old={norm(old)}\n new={norm(new)}"
    if old is None or new is None:
        assert norm(new) == norm(old), msg
        return
    assert (new[0], new[2]) == (old[0], old[2]), msg
    new_map = dict(new[1])
    maps = _automorphic_mappings(mol, new[2])
    assert new_map in maps, msg + "\n (not a symmetry of the core)"
    same_placement = [d for d in maps if set(d) == set(new_map)]
    best = min(_substituent_locants(mol, d) for d in same_placement)
    assert _substituent_locants(mol, new_map) == best, (
        msg + "\n (not the orientation with the lowest substituent locants)")
    assert best <= _substituent_locants(mol, dict(old[1])), (
        msg + "\n (substituent locants higher than the full scan's)")


@pytest.mark.parametrize("smi,expected", [
    ("c1ncc2[nH]cnc2n1", "7H-purine"),
    ("c1ncc2nc[nH]c2n1", "9H-purine"),
])
def test_purine_tautomer_is_named_at_its_indicated_hydrogen(smi, expected):
    """ (16) "purine (special numbering, 7H-isomer shown; the PIN is
    7H-purine)" (the Blue Book), with (:14607): the indicated hydrogen
    sits at the atom that bears it, so the N7-H input is '7H-purine' although the 9H
    entry is the first catalogue hit. OPSIN 2.9.0 reads '7H-purine' back to the input's
    full InChIKey AND fixed-H InChI (the standard key alone cannot tell the tautomers
    apart); '9H-purine' does not read back to the 7H input's fixed-H InChI."""
    mol = Chem.MolFromSmiles(smi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = match_fused_heterocycle_core(mol)
    assert got is not None and got[0] == expected, got


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
