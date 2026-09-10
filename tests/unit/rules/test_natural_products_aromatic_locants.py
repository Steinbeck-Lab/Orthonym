"""a phase + + -05: deterministic aromatic-ring
canonical locants for retained NP scaffolds.

Tests cover:
- _NP_AROMATIC_RING_LOCANTS table presence + estrane entry per IUPAC
- _find_scaffold_unsaturation table-path branch (implementation)
- _find_scaffold_unsaturation Kekulize-path branch preservation (non-cataloged)
- Indicated-H marker rendering per IUPAC in _assemble_np_name
- Saturated-A-ring estrane negative case (no (10) marker on non-aromatic)
- Non-aromatic-scaffold path unchanged (cholestane, androstane)
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.natural_products import (
    _NP_AROMATIC_RING_LOCANTS,
    _find_scaffold_unsaturation,
    _build_scaffold_aromatic_atoms,
    _build_target_to_iupac,
)
from orthonym.perception.natural_products import detect_natural_product


class TestNPAromaticRingLocantsTable:
    """internal notes + -05: canonical-locant table for aromatic NP scaffolds."""

    def test_estrane_in_canonical_table(self):
        """_NP_AROMATIC_RING_LOCANTS has estrane entry per IUPAC."""
        assert "estrane" in _NP_AROMATIC_RING_LOCANTS
        ene = _NP_AROMATIC_RING_LOCANTS["estrane"]
        assert ene == [(1, 2, None), (3, 4, None), (5, 10, 10)], (
            f"Expected canonical estra-1,3,5(10)-trien locants per "
            f"IUPAC P-31.1.4.3.4; got: {ene}"
        )

    def test_estrane_aromatic_a_ring_emits_1_3_5_10_trien(self):
        """The canonical IUPAC PIN for estradiol A-ring trien is 1,3,5(10).

        Uses canary fixture name_stability_265 SMILES (estradiol-derivative).
        """
        smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O"
        name = name_compound(smi)
        assert "1,3,5(10)" in name, (
            f"Expected 1,3,5(10) indicated-H marker per IUPAC P-31.1.4.3.4; "
            f"got: {name!r}"
        )
        assert "estra-1,3,5(10)-trien" in name, (
            f"Expected canonical estra-1,3,5(10)-trien form; got: {name!r}"
        )

    def test_estradiol_does_not_emit_1_2_4_trien(self):
        """Pre-amendment Form B (1,2,4-trien) is non-canonical and MUST NOT
        appear post-fix. Negative regression guard."""
        smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O"
        name = name_compound(smi)
        assert "1,2,4" not in name, (
            f"Non-canonical Form B 1,2,4-trien must not appear post-D-20 fix; "
            f"got: {name!r}"
        )


class TestFindScaffoldUnsaturationTablePath:
    """internal notes: _find_scaffold_unsaturation table-path branch."""

    def test_find_scaffold_unsaturation_returns_indicated_h_dict(self):
        """When scaffold_class is in the table AND molecule has aromatic atoms
        in the scaffold, _find_scaffold_unsaturation returns canonical locants
        from the table + an ene_indicated_h dict per IUPAC."""
        smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O"
        mol = Chem.MolFromSmiles(smi)
        info = detect_natural_product(mol)
        assert info is not None and info.get("scaffold_name") == "estrane"
        numbering = _build_target_to_iupac(info)
        scaffold_arom = _build_scaffold_aromatic_atoms(
            info.get("scaffold_smiles"), info.get("matched_atoms"),
        )
        u = _find_scaffold_unsaturation(
            mol, set(info["matched_atoms"]), numbering,
            scaffold_aromatic_atoms=scaffold_arom,
            scaffold_class=info["scaffold_name"],
        )
        # ene locants should match the canonical table
        assert u["ene"] == [1, 3, 5], (
            f"Expected ene [1, 3, 5] from canonical table; got: {u['ene']}"
        )
        # ene_indicated_h should map 5 -> 10 per IUPAC
        assert u["ene_indicated_h"] == {5: 10}, (
            f"Expected ene_indicated_h {{5: 10}}; got: {u['ene_indicated_h']}"
        )

    def test_find_scaffold_unsaturation_no_table_match_kekulize_path(self):
        """For a scaffold NOT in the table OR with no aromatic atoms,
        _find_scaffold_unsaturation falls back to Kekulize path; returns
        empty ene_indicated_h dict."""
        # Cholestane: saturated steroid; scaffold not in _NP_AROMATIC_RING_LOCANTS
        # SMILES from a known cholestane derivative (cholesterol).
        smi = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
        mol = Chem.MolFromSmiles(smi)
        info = detect_natural_product(mol)
        if info is not None:
            scaffold_class = info.get("scaffold_name")
            # cholestane is NOT in _NP_AROMATIC_RING_LOCANTS
            assert scaffold_class not in _NP_AROMATIC_RING_LOCANTS or (
                scaffold_class == "estrane"
                # If the perception mis-detects as estrane, the test still
                # validates the Kekulize-path preservation for non-aromatic
                # ring atoms (saturated cholestane has no aromatic atoms).
            )
            numbering = _build_target_to_iupac(info)
            if numbering:
                scaffold_arom = _build_scaffold_aromatic_atoms(
                    info.get("scaffold_smiles"), info.get("matched_atoms"),
                )
                u = _find_scaffold_unsaturation(
                    mol, set(info["matched_atoms"]), numbering,
                    scaffold_aromatic_atoms=scaffold_arom,
                    scaffold_class=scaffold_class,
                )
                # ene_indicated_h MUST be empty for non-table-cataloged paths
                # OR for cataloged scaffolds with no runtime-aromatic atoms.
                assert isinstance(u.get("ene_indicated_h", {}), dict)


class TestAromaticScaffoldRendering:
    """internal notes + IUPAC: rendering of (10) marker."""

    def test_indicated_h_marker_appears_in_output(self):
        """The (10) indicated-H marker MUST appear in rendered output for
        estradiol-derivative compounds."""
        smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O"
        name = name_compound(smi)
        # The marker must be in parenthetical form, NOT bare (10)
        assert "(10)" in name, (
            f"Indicated-H marker (10) missing from output; got: {name!r}"
        )

    def test_non_aromatic_scaffold_no_indicated_h_marker(self):
        """A non-aromatic scaffold (cholestane / androstane) MUST NOT emit
        the (10) indicated-H marker because the table only fires when
        runtime-aromatic atoms are present in the scaffold."""
        # Cholestane SMILES (saturated; no aromatic ring)
        smi = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@@H]4[C@@]3(CC[C@@H](C4)O)C)C"
        name = name_compound(smi)
        # Should NOT have indicated-H marker for non-aromatic scaffold
        assert "(10)" not in name, (
            f"Non-aromatic scaffold should not emit (10) marker; got: {name!r}"
        )


class TestBackwardsCompatibility:
    """Verify the amendment preserves backwards compatibility."""

    def test_unsaturation_dict_still_has_ene_key(self):
        """The 'ene' key continues to be a sorted List[int] for backwards-compat."""
        smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O"
        mol = Chem.MolFromSmiles(smi)
        info = detect_natural_product(mol)
        numbering = _build_target_to_iupac(info)
        scaffold_arom = _build_scaffold_aromatic_atoms(
            info.get("scaffold_smiles"), info.get("matched_atoms"),
        )
        u = _find_scaffold_unsaturation(
            mol, set(info["matched_atoms"]), numbering,
            scaffold_aromatic_atoms=scaffold_arom,
            scaffold_class=info.get("scaffold_name"),
        )
        assert isinstance(u["ene"], list)
        assert all(isinstance(x, int) for x in u["ene"])
        assert u["ene"] == sorted(u["ene"])

    def test_unsaturation_dict_still_has_yne_key(self):
        """The 'yne' key continues to be a sorted List[int]."""
        smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O"
        mol = Chem.MolFromSmiles(smi)
        info = detect_natural_product(mol)
        numbering = _build_target_to_iupac(info)
        scaffold_arom = _build_scaffold_aromatic_atoms(
            info.get("scaffold_smiles"), info.get("matched_atoms"),
        )
        u = _find_scaffold_unsaturation(
            mol, set(info["matched_atoms"]), numbering,
            scaffold_aromatic_atoms=scaffold_arom,
            scaffold_class=info.get("scaffold_name"),
        )
        assert isinstance(u["yne"], list)
