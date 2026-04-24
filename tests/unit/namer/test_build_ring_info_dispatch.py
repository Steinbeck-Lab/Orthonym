"""Phase 147 Plan 02 Task 3: tests for _build_ring_info_for_parent_selection
+ compute_features module-level helper.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
Source: Phase 147 CONTEXT D-03 (7-branch dispatch order),
        D-04 (reuse handler orient functions),
        D-06 (spiro/VB stubs),
        D-09 (preserve fused-hetero byte-identical path).
"""

import pytest
from rdkit import Chem


def test_compute_features_returns_molecularfeatures():
    """compute_features(mol) wraps Orthonym()._perceive (BL-1 fix).

    Source: Phase 147 BL-1.
    """
    from orthonym.namer import compute_features, MolecularFeatures
    mol = Chem.MolFromSmiles('CCO')
    feats = compute_features(mol)
    assert isinstance(feats, MolecularFeatures)
    assert feats.functional_groups is not None


def test_compute_features_accepts_explicit_smiles():
    """compute_features(mol, smiles='CCO') accepts an explicit SMILES.

    Source: Phase 147 BL-1.
    """
    from orthonym.namer import compute_features
    mol = Chem.MolFromSmiles('CCO')
    feats = compute_features(mol, smiles='CCO')
    assert feats.smiles == 'CCO'


def test_acyclic_returns_none():
    """Acyclic molecules: helper returns None (no ring_info needed).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 D-03 branch 7.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    mol = Chem.MolFromSmiles('CCO')
    feats = compute_features(mol)
    assert _build_ring_info_for_parent_selection(feats) is None


def test_fused_heterocycle_branch_byte_identical():
    """Branch 1 (D-09 byte-identical): 2-methylindole returns the same
    atom_mapping as match_fused_heterocycle_core (preserves the existing
    fused-hetero path bit-for-bit).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-09.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    from orthonym.data.fused_heterocycles import match_fused_heterocycle_core
    mol = Chem.MolFromSmiles('Cc1cc2ccccc2[nH]1')
    feats = compute_features(mol)
    het = match_fused_heterocycle_core(mol)
    assert het is not None, "indole core should match"
    _, expected_mapping, _ = het
    actual = _build_ring_info_for_parent_selection(feats)
    assert actual == {"iupac_locants": expected_mapping}


def test_pah_naphthalene_branch():
    """Branch 2: naphthalene returns the wrapper output directly.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-03.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants
    mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    feats = compute_features(mol)
    expected = get_polycyclic_iupac_locants(mol, 'naphthalene')
    actual = _build_ring_info_for_parent_selection(feats)
    assert actual is not None
    assert actual["iupac_locants"] == expected


def test_pah_pyrene_w1_regression():
    """Branch 2 W-1 fix regression: pyrene is ortho-peri-fused but PAH
    detection MUST run for it (not gated on the fused-only block).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-03 W-1 fix.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')
    feats = compute_features(mol)
    actual = _build_ring_info_for_parent_selection(feats)
    assert actual is not None
    assert actual["iupac_locants"] is not None
    locants = actual["iupac_locants"]
    fusion_tuples = [v for v in locants.values() if isinstance(v, tuple)]
    assert len(fusion_tuples) >= 6, (
        f"pyrene should have 6 fusion tuples, got {fusion_tuples!r}"
    )


def test_benzene_only_toluene_bl2_regression():
    """Branch 3 BL-2 fix regression: toluene's methyl-bearing aromatic
    carbon (mol idx 1) gets locant 1 because get_benzene_substituents is
    used (not a {idx: []} placeholder).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-14.4(g)
    Source: Phase 147 CONTEXT D-03 BL-2 fix.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    mol = Chem.MolFromSmiles('Cc1ccccc1')
    feats = compute_features(mol)
    actual = _build_ring_info_for_parent_selection(feats)
    assert actual is not None
    assert actual["iupac_locants"] is not None
    locants = actual["iupac_locants"]
    # Toluene atom 0 = methyl C (sp3, off-ring); atom 1 = ipso aromatic C
    # bearing the methyl. orient_benzene must place that ipso C at locant 1.
    assert locants.get(1) == 1, (
        f"toluene ipso aromatic C (idx 1) should be locant 1; got "
        f"{locants!r}"
    )
    # Six locants total.
    assert len(locants) == 6


def test_simple_heterocycle_pyridine_branch():
    """Branch 4: pyridine returns 6 keys; the N atom gets locant 1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-25.3
    Source: Phase 147 CONTEXT D-03 branch 4.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    mol = Chem.MolFromSmiles('c1ccncc1')
    feats = compute_features(mol)
    actual = _build_ring_info_for_parent_selection(feats)
    assert actual is not None
    assert actual["iupac_locants"] is not None
    locants = actual["iupac_locants"]
    assert len(locants) == 6
    # Find the nitrogen atom index.
    n_idx = next(
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N'
    )
    assert locants[n_idx] == 1, (
        f"pyridine N (idx {n_idx}) should be locant 1; got {locants!r}"
    )


def test_spiro_stub_branch():
    """Branch 5: spiro[4.4]nonane returns ``{'iupac_locants': None}`` stub.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-24.2
    Source: Phase 147 CONTEXT D-06 (Phase 151 fills the real numbering).
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    mol = Chem.MolFromSmiles('C1CCCC12CCCC2')
    feats = compute_features(mol)
    actual = _build_ring_info_for_parent_selection(feats)
    assert actual is not None
    assert actual["iupac_locants"] is None


def test_else_cyclohexane_branch():
    """Branch 7: cyclohexane (carbocyclic monocycle, no FG) returns None
    so _build_ring_pos falls through to sorted fallback.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-03 branch 7.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )
    mol = Chem.MolFromSmiles('C1CCCCC1')
    feats = compute_features(mol)
    actual = _build_ring_info_for_parent_selection(feats)
    # cyclohexane: not benzene (not aromatic), no heteroatom, not spiro,
    # not bridged -> None per branch 7.
    assert actual is None
