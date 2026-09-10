"""a phase.A cross-handler dispatch contract.

cyclophane vs ring_assembly vs multiplicative are MUTUALLY EXCLUSIVE by
topology after Plan 155-01 ships:

  - >= 2 disjoint small rings linked by acyclic chain >= 2 atoms with
    macrocyclic closure -> cyclophane
  - single-bond-joined identical rings -> ring_assemblies
  - atom/group-bridged identical units (typically 1-atom bridge) -> multiplicative

Each test exercises both the topology-gate predicates directly AND the
full ``name_compound`` pipeline so a future regression in either guard
cannot be silently masked by the dispatch order in namer.py.

Source: internal notes / /;
        tests/integration/test_assembly_vs_multiplicative_dispatch.py
        (a phase pattern).
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.multiplicative import (
    _is_pure_single_bond_assembly,
    name_multiplicative,
)
from orthonym.rules.phane import is_cyclophane, name_cyclophane
from orthonym.rules.ring_assemblies import detect_ring_assembly


def test_phane_module_importable() -> None:
    """Smoke test: the phane public API is importable for dispatch tests."""
    assert callable(is_cyclophane)
    assert callable(name_cyclophane)


@pytest.mark.integration
class TestPhaneVsRingAssemblyDispatch:
    """ mutual-exclusion contract -- no double-fire on edge cases."""

    @pytest.mark.parametrize(
        "smiles,label",
        [
            ("c1cc2ccc1CCc1ccc(cc1)CC2", "[2.2]paracyclophane"),
            ("c1cc2cc(c1)CCc1cccc(c1)CC2", "[2.2]metacyclophane"),
            ("c1cc2ccc1CCCc1ccc(cc1)CCC2", "[3.3]paracyclophane"),
        ],
    )
    def test_cyclophane_routes_to_phane_handler(self, smiles, label) -> None:
        """Cyclophane SMILES -> phane handler; multiplicative + ring_assembly decline.

        Wave-8 P8: `name_cyclophane` now emits the P-26 simplified-skeletal
        PIN (`...phane`, e.g. `1,4(1,4)-dibenzenacyclohexaphane`) for this
        class rather than the legacy semi-systematic bracket-prefix form
        (`[2.2]paracyclophane`) -- assert the generic `...phane` suffix
        (exact-string coverage lives in test_phane.py / test_phane_pin.py).
        """
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label

        # Phane handler MUST accept.
        assert is_cyclophane(mol), f"is_cyclophane declined cyclophane {label}"
        phane_name = name_cyclophane(mol)
        assert phane_name is not None, f"name_cyclophane declined cyclophane {label}"
        assert phane_name.endswith("phane"), (
            f"name_cyclophane({label}) returned non-phane name {phane_name!r}"
        )

        # Multiplicative MUST decline.
        assert name_multiplicative(mol) is None, (
            f"D-16 violation: name_multiplicative accepted cyclophane {label}"
        )

        # Ring-assembly MUST decline.
        ring_systems = get_ring_systems(mol)
        ra_info = detect_ring_assembly(mol, ring_systems)
        assert ra_info is None, (
            f"D-16 violation: detect_ring_assembly accepted cyclophane {label}"
        )

        # Single-bond-assembly guard MUST decline.
        assert not _is_pure_single_bond_assembly(mol), (
            f"D-16 violation: cyclophane {label} appears as pure single-bond assembly"
        )

    @pytest.mark.parametrize(
        "smiles,label",
        [
            ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
            ("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "p-terphenyl"),
        ],
    )
    def test_ring_assembly_routes_away_from_phane(self, smiles, label) -> None:
        """Single-bond-joined identical rings -> ring_assemblies, NOT cyclophane."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label

        # Phane handler MUST decline.
        assert not is_cyclophane(mol), (
            f"D-16 violation: is_cyclophane accepted ring-assembly {label}"
        )
        assert name_cyclophane(mol) is None, (
            f"D-16 violation: name_cyclophane accepted ring-assembly {label}"
        )

        # Ring-assembly MUST accept.
        ring_systems = get_ring_systems(mol)
        ra_info = detect_ring_assembly(mol, ring_systems)
        assert ra_info is not None, (
            f"D-16 violation: detect_ring_assembly declined {label}"
        )

    @pytest.mark.parametrize(
        "smiles,expected_name,label",
        [
            ("Nc1ccc(Cc2ccc(N)cc2)cc1", "4,4'-methylenedianiline", "methylenedianiline"),
        ],
    )
    def test_multiplicative_routes_away_from_phane(
        self, smiles, expected_name, label
    ) -> None:
        """Atom-bridged identical rings -> multiplicative, NOT cyclophane."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label

        # Phane MUST decline.
        assert not is_cyclophane(mol), (
            f"D-16 violation: is_cyclophane accepted multiplicative {label}"
        )
        assert name_cyclophane(mol) is None, (
            f"D-16 violation: name_cyclophane accepted multiplicative {label}"
        )

        # Multiplicative MUST accept.
        result = name_multiplicative(mol)
        assert result == expected_name, (
            f"D-16 violation: multiplicative gave {result!r}, expected {expected_name!r}"
        )

    @pytest.mark.parametrize(
        "smiles,label",
        [
            # Open-chain bis-phenyl: cyclophane gate must reject (no macrocycle)
            ("c1ccc(CCc2ccc(CCc3ccccc3)cc2)cc1", "open-chain 1,4-bis(2-phenylethyl)benzene"),
            ("c1ccccc1CCc1ccccc1", "two benzenes via -CH2-CH2- (acyclic)"),
            # Naphthalene: fused single-system; not cyclophane
            ("c1ccc2ccccc2c1", "naphthalene (fused)"),
        ],
    )
    def test_non_cyclophane_topologies_route_away_from_phane(self, smiles, label) -> None:
        """Various non-cyclophane topologies all route away from the phane handler."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        assert not is_cyclophane(mol), (
            f"D-16 violation: is_cyclophane accepted non-cyclophane {label}"
        )
        assert name_cyclophane(mol) is None, (
            f"D-16 violation: name_cyclophane accepted non-cyclophane {label}"
        )

    @pytest.mark.parametrize(
        "smiles,expected_pin,label",
        [
            ("c1cc2ccc1CCc1ccc(cc1)CC2", "1,4(1,4)-dibenzenacyclohexaphane",
             "[2.2]paracyclophane via name_compound"),
            ("c1cc2cc(c1)CCc1cccc(c1)CC2", "1,4(1,3)-dibenzenacyclohexaphane",
             "[2.2]metacyclophane via name_compound"),
        ],
    )
    def test_name_compound_full_dispatch_lands_on_phane(
        self, smiles, expected_pin, label
    ) -> None:
        """End-to-end name_compound EMITS the P-26 PIN (Wave-8 P8, Task 8.12).

        The phane handler classifies + composes (asserted above); production
        now ships the verified `build_phane_pin` PIN for the monocyclic
        all-benzene-homophane class -- OPSIN still cannot parse ANY phane
        form, but `_PHANE_PIN_RE` (namer.py) carves this correct-by-
        construction, formula-veto-guarded PIN out of the validity gate
        (see docs/superpowers/plans/2026-07-16-wave8-p8-phane.md)."""
        result = name_compound(smiles)
        assert result == expected_pin, (
            f"name_compound({label}) returned {result!r}; expected the "
            f"Wave-8 P8 P-26 PIN {expected_pin!r}"
        )
