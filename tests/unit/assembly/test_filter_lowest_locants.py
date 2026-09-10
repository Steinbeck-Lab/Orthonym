"""a phase Plan 02 Task 4: tests for _filter_lowest_locants + cascade.

Replaces the a phase stub. Cascade order:
  1. / (f) — lowest principal-group locants
  2. (g) — lowest multiple-bond locants
  3. (i) — lowest substituent locants

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html -
Source: a phase internal notes (cascade gate), (stub semantics),
         (reuse parent_selection.py extraction patterns).
"""

import pytest
from rdkit import Chem


def _make_candidate(
    name, parent_atom_indices, ring_info, mol, pg_atoms_list,
    handler='benzene',
):
    """Construct a CandidateName with all the fields _filter_lowest_locants
    expects (POST-HOC pattern matching what pool.add does).
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    cand = CandidateName(name=name, handler=handler)
    cand.parent_atom_indices = set(parent_atom_indices)
    cand.ring_info = ring_info
    cand._mol_ref = mol
    cand._principal_group_atoms = list(pg_atoms_list)
    return cand


def test_pg_locants_decisive():
    """Step 1: lower PG locant wins decisively.

    Two candidates on the same molecule but different orientations: A
    places the PG attachment at locant 1; B at locant 2. A wins.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    """
    from orthonym.assembly.candidate_pool import _filter_lowest_locants
    mol = Chem.MolFromSmiles('Oc1ccccc1')  # phenol
    parent = {1, 2, 3, 4, 5, 6}
    # PG attachment is mol-atom 1 (ipso aromatic C bearing OH at idx 0).
    pg_atoms = [(0,)]
    # Orientation A: ipso=1
    a_locants = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    # Orientation B: ipso=2 (worse — PG would be locant 2)
    b_locants = {1: 2, 2: 1, 3: 6, 4: 5, 5: 4, 6: 3}
    a = _make_candidate('A', parent, {'iupac_locants': a_locants}, mol,
                        pg_atoms)
    b = _make_candidate('B', parent, {'iupac_locants': b_locants}, mol,
                        pg_atoms)
    result = _filter_lowest_locants([a, b])
    assert result == [a]


def test_full_tie_returns_both():
    """All cascade steps tie -> both candidates returned (caller drops
    to Tier-2 weighted-sum).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    """
    from orthonym.assembly.candidate_pool import _filter_lowest_locants
    mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
    parent = {0, 1, 2, 3, 4, 5}
    pg_atoms = []  # no PG
    # Identical locants -> identical PG/MB/sub locants -> all-tie.
    locants = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    a = _make_candidate('A', parent, {'iupac_locants': locants}, mol,
                        pg_atoms)
    b = _make_candidate('B', parent, {'iupac_locants': locants}, mol,
                        pg_atoms)
    result = _filter_lowest_locants([a, b])
    assert len(result) == 2


def test_single_candidate_short_circuit():
    """Single candidate -> returned unchanged (cascade no-op).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    """
    from orthonym.assembly.candidate_pool import _filter_lowest_locants
    from orthonym.assembly.coverage_scoring import CandidateName
    a = CandidateName(name='A', handler='benzene')
    out = _filter_lowest_locants([a])
    assert out == [a]


def test_empty_input_returns_empty():
    """Empty input -> empty output (cascade no-op).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    """
    from orthonym.assembly.candidate_pool import _filter_lowest_locants
    assert _filter_lowest_locants([]) == []


def test_defensive_bailout_on_missing_parent_atom_indices():
    """If any candidate lacks parent_atom_indices, return all candidates
    unchanged (defensive fall-through; Tier-2 takes over).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    Source: a phase conservative gate.
    """
    from orthonym.assembly.candidate_pool import _filter_lowest_locants
    from orthonym.assembly.coverage_scoring import CandidateName
    a = CandidateName(name='A', handler='benzene',
                      ring_info={'iupac_locants': {0: 1}})
    b = CandidateName(name='B', handler='benzene',
                      ring_info={'iupac_locants': {0: 1}})
    # parent_atom_indices is None on both -> bail out.
    out = _filter_lowest_locants([a, b])
    assert out == [a, b]


def test_tuple_locant_fusion_atom_wins():
    """Tuple-locant comparison: PG attached at fusion atom (4, 'a') wins
    over PG at peripheral 5 because (4, '') < (4, 'a') < (5, '').

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html first-point-of-difference
    Source: a phase internal notes tuple encoding (Plan 01 enables comparison).
    """
    from orthonym.assembly.candidate_pool import _filter_lowest_locants
    # Synthetic: just construct two candidates with hand-built ring_info
    # and pg_atoms that resolve to different locants.
    # Use a real mol so _build_ring_pos works (naphthalene).
    mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    parent = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # Candidate A: PG attached at mol idx 3 (which we mark as locant (4, 'a'))
    a_locants = {0: 2, 1: 3, 2: 4, 3: (4, 'a'), 4: 5, 5: 6, 6: 7, 7: 8,
                 8: (8, 'a'), 9: 1}
    # Candidate B: PG attached at mol idx 4 (peripheral locant 5)
    b_locants = dict(a_locants)
    a_pg = [(10, 3)]  # PG at attachment idx 3 (the fusion atom)
    b_pg = [(10, 4)]  # PG at attachment idx 4 (peripheral 5)
    # We need atom 10 to exist on the mol; build a doctored mol with an
    # extra C neighbour for testing purposes.
    mol2 = Chem.RWMol(mol)
    new_idx = mol2.AddAtom(Chem.Atom(6))
    mol2.AddBond(3, new_idx, Chem.BondType.SINGLE)
    new_idx2 = mol2.AddAtom(Chem.Atom(6))
    mol2.AddBond(4, new_idx2, Chem.BondType.SINGLE)
    real_mol = mol2.GetMol()

    a = _make_candidate(
        'A', parent, {'iupac_locants': a_locants}, real_mol,
        [(new_idx,)],
    )
    b = _make_candidate(
        'B', parent, {'iupac_locants': b_locants}, real_mol,
        [(new_idx2,)],
    )
    result = _filter_lowest_locants([a, b])
    # A's PG locant is (4, 'a'); B's is 5 (which becomes (5, '') in
    # homogeneity coercion). (4, 'a') < (5, '') -> A wins.
    assert result == [a]
