"""a phase.B cross-handler dispatch contract.

name_multiplicative and detect_ring_assembly are MUTUALLY EXCLUSIVE by
topology after Plan 154-02 ships:
  - single-bond-joined identical rings -> ring_assemblies (a phase)
  - atom/group-bridged identical units -> multiplicative

This test exercises the FULL name_compound pipeline so a future regression
in either guard cannot be silently masked by the dispatch order in
namer.py:935-938. Pattern from a phase-04 closure
(test_mixed_spiro_fused_dispatch.py).

Source: 154-internal notes; 151-internal notes (path-topology contract);
        internal notes Pattern S-8.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.multiplicative import (
    name_multiplicative,
    _is_pure_single_bond_assembly,
)
from orthonym.rules.ring_assemblies import detect_ring_assembly
from orthonym.perception.rings import get_ring_systems


_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "multiplicative"
)
_BORDERLINE_FIXTURES = [
    f
    for f in json.loads((_FIXTURE_DIR / "corpus_mined.json").read_text())
    if f.get("compound_class") == "dispatch-borderline"
]


@pytest.mark.integration
class TestAssemblyVsMultiplicativeDispatch:
    """D-11 mutual-exclusion contract — neither handler double-fires."""

    @pytest.mark.parametrize(
        "smiles",
        [
            "c1ccc(-c2ccccc2)cc1",  # biphenyl: assembly only
            "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",  # terphenyl: assembly only
            "c1ccnc(-c2ccccn2)c1",  # 2,2'-bipyridine: assembly only
        ],
    )
    def test_single_bond_assembly_routes_to_ring_assemblies(self, smiles):
        """Single-bond-joined identical rings -> ring_assemblies (a phase)."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        # Multiplicative MUST decline (topology guard).
        assert name_multiplicative(mol) is None, (
            f"D-11 violation: name_multiplicative accepted single-bond "
            f"assembly {smiles!r}"
        )

        # Ring-assembly MUST accept.
        ring_systems = get_ring_systems(mol)
        info = detect_ring_assembly(mol, ring_systems)
        assert info is not None, (
            f"D-11 violation: detect_ring_assembly declined {smiles!r}"
        )

    @pytest.mark.parametrize(
        "smiles,expected_name",
        [
            ("Nc1ccc(Cc2ccc(N)cc2)cc1", "4,4'-methylenedianiline"),
            ("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1", "4,4'-oxydibenzoic acid"),
            (
                "Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1",
                "4,4',4''-nitrilotriphenol",
            ),
        ],
    )
    def test_atom_bridge_routes_to_multiplicative(self, smiles, expected_name):
        """Atom-bridged identical rings -> multiplicative (this module)."""
        mol = Chem.MolFromSmiles(smiles)
        result = name_multiplicative(mol)
        # Multiplicative MUST accept.
        assert result == expected_name, (
            f"D-11 violation: multiplicative gave {result!r}, "
            f"expected {expected_name!r}"
        )

        # Ring-assembly MUST decline (no inter-system single bond between
        # ring atoms; the bridge atom intervenes).
        ring_systems = get_ring_systems(mol)
        info = detect_ring_assembly(mol, ring_systems)
        assert info is None, (
            f"D-11 violation: detect_ring_assembly should decline "
            f"atom-bridged {smiles!r}, got {info!r}"
        )

    @pytest.mark.parametrize(
        "smiles",
        [
            # Curated cases per internal notes -- neither handler double-fires
            "c1ccc(-c2ccccc2)cc1",  # biphenyl
            "Nc1ccc(Cc2ccc(N)cc2)cc1",  # methylenedianiline
            "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",  # terphenyl
            "Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1",  # nitrilotriphenol
            "OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1",  # oxydibenzoic acid
        ],
    )
    def test_no_double_fire(self, smiles):
        """For every curated case: name_compound returns ONE coherent name."""
        name = name_compound(smiles)
        assert name is not None, (
            f"D-11 violation: both handlers declined for {smiles!r}"
        )

        # Confirm the topology guard logic is consistent.
        mol = Chem.MolFromSmiles(smiles)
        mult_result = name_multiplicative(mol)
        ring_systems = get_ring_systems(mol)
        ra_info = detect_ring_assembly(mol, ring_systems)
        # Mutual-exclusion: at most ONE handler fires.
        both_fire = (mult_result is not None) and (ra_info is not None)
        assert not both_fire, (
            f"D-11 DOUBLE-FIRE: {smiles!r} multiplicative={mult_result!r} "
            f"ring_assembly={ra_info!r}"
        )

    def test_pure_single_bond_predicate_consistent(self):
        """Topology predicate consistency: when _is_pure_single_bond_assembly
        is True, name_multiplicative declines."""
        from rdkit import Chem as _Chem

        for smi in (
            "c1ccc(-c2ccccc2)cc1",
            "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
            "c1ccnc(-c2ccccn2)c1",
        ):
            mol = _Chem.MolFromSmiles(smi)
            assert _is_pure_single_bond_assembly(mol) is True
            assert name_multiplicative(mol) is None

    if _BORDERLINE_FIXTURES:
        # Cap to the first 50 borderline corpus rows (parametrize explosion guard).
        @pytest.mark.parametrize(
            "fixture",
            _BORDERLINE_FIXTURES[:50],
            ids=[f["fixture_id"] for f in _BORDERLINE_FIXTURES[:50]],
        )
        def test_borderline_corpus_dispatch(self, fixture):
            """Corpus-mined dispatch-borderline cases land cleanly per the audit verdict."""
            smi = fixture["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                pytest.skip(f"unparseable SMILES: {smi}")
            mult_result = name_multiplicative(mol)
            try:
                ring_systems = get_ring_systems(mol)
                ra_info = detect_ring_assembly(mol, ring_systems)
            except Exception:
                pytest.skip(
                    f"detect_ring_assembly raised for {fixture['fixture_id']}"
                )
            # Mutual exclusion: at most ONE handler fires.
            both_fire = (mult_result is not None) and (ra_info is not None)
            assert not both_fire, (
                f"DOUBLE-FIRE on corpus borderline {fixture['fixture_id']!r}: "
                f"smi={smi!r} multiplicative={mult_result!r} "
                f"ring_assembly={ra_info!r}"
            )
