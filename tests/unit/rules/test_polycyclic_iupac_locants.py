"""Phase 147 Plan 02 Task 2: tests for get_polycyclic_iupac_locants wrapper.

The wrapper reads ``POLYCYCLIC_DATA[name]['iupac_numbering']`` (string fusion
locants like '4a' / '10b') and returns mol-atom-keyed dicts with peripheral
ints + (int, str) tuple fusion locants per Phase 147 Decision D-01.

Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25 (PAH numbering FIXED)
Source: Phase 147 CONTEXT D-01 (tuple encoding); CD-03 (function form).
"""

import pytest
from rdkit import Chem


def _expected_dict(mol, pah_name):
    """Compute expected dict by hand using the same canonical-match path."""
    import re
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    entry = POLYCYCLIC_DATA[pah_name]
    canonical_mol = Chem.MolFromSmiles(entry['canonical_smiles'])
    match = mol.GetSubstructMatch(canonical_mol)
    locant_re = re.compile(r'^(\d+)([a-z]*)$')
    out = {}
    for canonical_idx, locant in entry['iupac_numbering'].items():
        mol_idx = match[canonical_idx]
        if isinstance(locant, int):
            out[mol_idx] = locant
        else:
            m = locant_re.match(locant)
            base = int(m.group(1))
            suffix = m.group(2)
            out[mol_idx] = (base, suffix) if suffix else base
    return out


def test_naphthalene_complete_coverage():
    """naphthalene: 10 keys (8 ints + 2 fusion tuples (4,'a') and (8,'a')).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01 tuple encoding.
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    result = get_polycyclic_iupac_locants(mol, 'naphthalene')
    assert result is not None
    assert len(result) == 10
    expected = _expected_dict(mol, 'naphthalene')
    assert result == expected
    fusion_tuples = [v for v in result.values() if isinstance(v, tuple)]
    assert len(fusion_tuples) == 2
    bases = sorted(t[0] for t in fusion_tuples)
    assert bases == [4, 8]
    assert all(t[1] == 'a' for t in fusion_tuples)


def test_anthracene_four_fusion_atoms():
    """anthracene: 14 keys (10 ints + 4 fusion tuples).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01.
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
    result = get_polycyclic_iupac_locants(mol, 'anthracene')
    assert result is not None
    assert len(result) == 14
    fusion_tuples = sorted(
        v for v in result.values() if isinstance(v, tuple)
    )
    assert fusion_tuples == [(4, 'a'), (8, 'a'), (9, 'a'), (10, 'a')]


def test_phenanthrene_4b_locant():
    """phenanthrene: 14 keys; fusion set includes (4,'a'), (4,'b'),
    (8,'a'), (10,'a').

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01.
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1ccc2ccc3ccccc3c2c1')
    result = get_polycyclic_iupac_locants(mol, 'phenanthrene')
    assert result is not None
    assert len(result) == 14
    fusion_tuples = sorted(
        v for v in result.values() if isinstance(v, tuple)
    )
    assert fusion_tuples == [(4, 'a'), (4, 'b'), (8, 'a'), (10, 'a')]


def test_pyrene_six_fusion_atoms_and_10b_two_digit():
    """pyrene: 16 keys; fusion set is the six interior/peri carbons
    3a, 5a, 8a, 10a, 10b, 10c (includes (10,'b') two-digit-base regression).

    Critical: '10b' MUST parse to (10, 'b'), NOT (1, '0b').

    v23 IH-01 (2026-06-22): the stored pyrene ``iupac_numbering`` was corrected
    to the OPSIN-authoritative numbering (``pyrene -o extendedsmi``). The prior
    fusion set wrongly contained ``(3,'b')`` and omitted ``(10,'c')`` — pyrene's
    peri carbons are 10a/10b/10c, there is no 3b. This test had pinned the
    invalid numbering.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01; v23 IH-01 numbering correction.
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')
    result = get_polycyclic_iupac_locants(mol, 'pyrene')
    assert result is not None
    assert len(result) == 16
    fusion_tuples = sorted(
        v for v in result.values() if isinstance(v, tuple)
    )
    assert fusion_tuples == [
        (3, 'a'), (5, 'a'), (8, 'a'), (10, 'a'), (10, 'b'), (10, 'c'),
    ]


def test_unknown_name_returns_none():
    """Unknown PAH name returns None — wrapper does not raise.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01.
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    assert get_polycyclic_iupac_locants(mol, 'nonexistent_pah') is None


def test_unpopulated_pah_returns_none():
    """fluorene's iupac_numbering is empty -> returns None (Phase 151 audit).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01; Phase 151 ring-class completion (deferred).
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1-2')
    assert get_polycyclic_iupac_locants(mol, 'fluorene') is None


def test_two_digit_base_locant_parses_correctly():
    """Defensive: pyrene's (10, 'b') must parse from '10b' as
    (10, 'b'), NOT (1, '0b'). Anchored regex ^(\\d+)([a-z]*)$ guards this.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 D-01 (tuple encoding rejects float-style ambiguity).
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')
    result = get_polycyclic_iupac_locants(mol, 'pyrene')
    assert result is not None
    # 10b case: tuple base==10, suffix=='b'
    assert any(v == (10, 'b') for v in result.values() if isinstance(v, tuple))
    # And ensure NO malformed (1, '0b') value snuck in.
    assert not any(
        v == (1, '0b') for v in result.values() if isinstance(v, tuple)
    )
    # 10a case as well.
    assert any(v == (10, 'a') for v in result.values() if isinstance(v, tuple))


def test_non_canonical_smiles_input_aligns_via_substructure_match():
    """Non-canonical input SMILES must still map fusion locants to the
    correct mol atoms via GetSubstructMatch alignment.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: Phase 147 CONTEXT D-01 (mol indices via substructure match).
    """
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    # Use a deliberately non-canonical SMILES (atom order randomised)
    mol = Chem.MolFromSmiles('c1ccc2ccccc2c1', sanitize=True)
    non_canon = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
    mol2 = Chem.MolFromSmiles(non_canon)
    assert mol2 is not None
    result = get_polycyclic_iupac_locants(mol2, 'naphthalene')
    assert result is not None
    assert len(result) == 10
    # Each fusion tuple value must be present and its key must point to a
    # ring atom shared between two rings of mol2.
    ri = mol2.GetRingInfo()
    for atom_idx, locant in result.items():
        if isinstance(locant, tuple):
            shared = sum(
                1 for ring in ri.AtomRings() if atom_idx in ring
            )
            assert shared >= 2, (
                f"fusion atom idx {atom_idx} (locant {locant}) is not in "
                f"multiple rings on the non-canonical input"
            )
