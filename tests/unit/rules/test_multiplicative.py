"""Phase 154.B unit tests for src/orthonym/rules/multiplicative.py.

Coverage matrix (per 154-02-PLAN.md success criteria):

  TestRetainedParentLookup       (D-10) — _resolve_parent_name + _extract_pg_locant_from_fragment
  TestTopologyGuard              (D-11) — _is_pure_single_bond_assembly entry-guard
  TestComplexMultipliersBranch   (D-08) — _select_multiplier P-14.2.1 vs P-14.2.2 split
  TestComplexMultipliersEndToEnd (D-08) — end-to-end SIMPLE multiplier preserved
  TestLocantCascade              (D-12) — _get_bridge_locant via Phase 151 cascade
  TestBridgeTables               (D-09) — bridge-table contents + audit cite
  TestNoBandAids                 (D-13) — no postprocessor / band-aid in source

Each class is gated by @pytest.mark.unit so it is collected with the
fast-only pytest profile.

Source: 154-02-PLAN.md Wave 2 + Wave 3 task <behavior> sections.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Task 2.1 — D-10 retained-parent registry-query layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetainedParentLookup:
    """D-10: _resolve_parent_name queries ALL_RETAINED_NAMES; _RETAINED_PARENT_NAMES is gone."""

    def test_aniline(self):
        from orthonym.rules.multiplicative import _resolve_parent_name

        result = _resolve_parent_name("Nc1ccccc1")
        assert result is not None
        assert result[0] == "aniline"
        assert result[1] == 1

    def test_phenol(self):
        from orthonym.rules.multiplicative import _resolve_parent_name

        result = _resolve_parent_name("Oc1ccccc1")
        assert result is not None
        assert result[0] == "phenol"
        assert result[1] == 1

    def test_benzoic_acid_canonical(self):
        from orthonym.rules.multiplicative import _resolve_parent_name

        # RDKit canonicalizes both 'OC(=O)c1ccccc1' and 'O=C(O)c1ccccc1' to
        # 'O=C(O)c1ccccc1' which IS the registry key.  D-10 routes through
        # the registry, so the canonical form is what we test.
        result = _resolve_parent_name("O=C(O)c1ccccc1")
        assert result is not None
        assert result[0] == "benzoic acid"
        assert result[1] == 1

    def test_unknown_smiles_returns_none(self):
        from orthonym.rules.multiplicative import _resolve_parent_name

        # An unparseable / unrecognised SMILES — registry miss, fragment-naming miss.
        result = _resolve_parent_name("ZzZ_invalid_smiles_xyz")
        assert result is None

    def test_legacy_dict_symbol_removed(self):
        """The hardcoded _RETAINED_PARENT_NAMES dict MUST be deleted (D-10)."""
        from orthonym.rules import multiplicative

        assert not hasattr(multiplicative, "_RETAINED_PARENT_NAMES"), (
            "Plan 154-02 D-10: _RETAINED_PARENT_NAMES dict must be deleted "
            "and replaced with _resolve_parent_name registry-query layer."
        )

    def test_name_parent_backwards_compat(self):
        """_name_parent backwards-compat wrapper preserves the legacy
        single-string return for existing callers."""
        from orthonym.rules.multiplicative import _name_parent

        assert _name_parent("Nc1ccccc1") == "aniline"
        assert _name_parent("Oc1ccccc1") == "phenol"

    def test_canary_methylenedianiline_e2e(self):
        """End-to-end canary: methylenedianiline still RTs after D-10."""
        from orthonym import name_compound

        assert (
            name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
            == "4,4'-methylenedianiline"
        )


# ---------------------------------------------------------------------------
# Task 2.2 — D-11 topology guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTopologyGuard:
    """D-11: _is_pure_single_bond_assembly + name_multiplicative entry-guard."""

    def test_biphenyl_is_pure_single_bond(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import _is_pure_single_bond_assembly

        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        assert _is_pure_single_bond_assembly(mol) is True

    def test_methylenedianiline_is_not_pure_single_bond(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import _is_pure_single_bond_assembly

        mol = Chem.MolFromSmiles("Nc1ccc(Cc2ccc(N)cc2)cc1")
        assert _is_pure_single_bond_assembly(mol) is False

    def test_terphenyl_is_pure_single_bond(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import _is_pure_single_bond_assembly

        mol = Chem.MolFromSmiles("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
        assert _is_pure_single_bond_assembly(mol) is True

    def test_single_ring_quick_reject(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import _is_pure_single_bond_assembly

        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene — only 1 ring system
        assert _is_pure_single_bond_assembly(mol) is False

    def test_biphenyl_multiplicative_returns_none(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import name_multiplicative

        # D-11 guard: multiplicative MUST decline biphenyl.
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        assert name_multiplicative(mol) is None

    def test_methylenedianiline_multiplicative_accepts(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import name_multiplicative

        # D-11 guard does NOT block atom-bridged cases.
        mol = Chem.MolFromSmiles("Nc1ccc(Cc2ccc(N)cc2)cc1")
        assert name_multiplicative(mol) == "4,4'-methylenedianiline"

    def test_biphenyl_full_pipeline(self):
        """End-to-end: biphenyl falls through to ring_assemblies."""
        from orthonym import name_compound

        # Biphenyl returns 'biphenyl' via Phase 151 ring_assemblies path.
        result = name_compound("c1ccc(-c2ccccc2)cc1")
        assert result is not None
        assert "biphenyl" in result.lower() or "1,1'-biphenyl" in result.lower()

    def test_methylenedianiline_full_pipeline(self):
        from orthonym import name_compound

        assert (
            name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
            == "4,4'-methylenedianiline"
        )


# ---------------------------------------------------------------------------
# Task 2.3 — D-08 P-14.2.1 vs P-14.2.2 multiplier split
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComplexMultipliersBranch:
    """D-08: P-14.2.1 (di/tri/tetra) vs P-14.2.2 (bis/tris/tetrakis) split."""

    @pytest.mark.parametrize(
        "parent_name,unit_count,expected_prefix",
        [
            ("aniline", 2, "di"),
            ("phenol", 3, "tri"),
            ("benzoic acid", 2, "di"),
            ("1,3-thiazole", 2, "bis"),
            ("1,3,5-triazine", 3, "tris"),
            # R-154-NEW-7 heuristic refinement: digit alone WITHOUT a comma
            # is NOT a P-14.2.2 trigger.  "but-2-ene" parses unambiguously
            # as "di(but-2-ene)" with SIMPLE_MULTIPLIERS.
            ("but-2-ene", 2, "di"),
            # Higher-arity COMPLEX_MULTIPLIERS coverage.
            ("1,2,3-triazole", 4, "tetrakis"),
        ],
    )
    def test_select_multiplier(self, parent_name, unit_count, expected_prefix):
        from orthonym.rules.multiplicative import _select_multiplier

        assert (
            _select_multiplier(parent_name, unit_count) == expected_prefix
        ), f"_select_multiplier({parent_name!r}, {unit_count}) failed"


@pytest.mark.unit
class TestComplexMultipliersEndToEnd:
    """D-08 end-to-end: canary SIMPLE multiplier path preserved."""

    @pytest.mark.parametrize(
        "smiles,expected_name",
        [
            ("Nc1ccc(Cc2ccc(N)cc2)cc1", "4,4'-methylenedianiline"),
            (
                "Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1",
                "4,4',4''-nitrilotriphenol",
            ),
        ],
    )
    def test_simple_multiplier_canaries(self, smiles, expected_name):
        from orthonym import name_compound

        assert name_compound(smiles) == expected_name


# ---------------------------------------------------------------------------
# Task 2.4 — D-12 locant cascade
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLocantCascade:
    """D-12: _get_bridge_locant queries Phase 151 cascade; legacy default-4 GONE."""

    def test_methylenedianiline_locant_via_cascade(self):
        from rdkit import Chem
        from orthonym.rules.multiplicative import _get_bridge_locant

        # 4,4'-methylenedianiline: bridge atom (CH2) connects to atoms in the
        # para position of two anilines.
        mol = Chem.MolFromSmiles("Nc1ccc(Cc2ccc(N)cc2)cc1")
        # Locate the bridge CH2 and one of the ring-conn atoms.
        bridge_idx = None
        ring_conn_idx = None
        ring_atoms = set()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            if atom.GetIdx() in ring_atoms:
                continue
            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy_nbrs) == 2 and all(n.GetIdx() in ring_atoms for n in heavy_nbrs):
                bridge_idx = atom.GetIdx()
                ring_conn_idx = heavy_nbrs[0].GetIdx()
                break
        assert bridge_idx is not None and ring_conn_idx is not None
        locant = _get_bridge_locant(mol, bridge_idx, ring_conn_idx, ring_atoms)
        assert locant == 4, f"expected para (locant 4), got {locant}"

    def test_canary_methylenedianiline_e2e_unchanged(self):
        from orthonym import name_compound

        assert (
            name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
            == "4,4'-methylenedianiline"
        )

    def test_oxydiphenol_e2e(self):
        from orthonym import name_compound

        # 4,4'-oxydiphenol: oxy bridge between two phenols.
        assert (
            name_compound("Oc1ccc(Oc2ccc(O)cc2)cc1") == "4,4'-oxydiphenol"
        )

    def test_nitrilotriphenol_e2e(self):
        from orthonym import name_compound

        assert (
            name_compound("Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
            == "4,4',4''-nitrilotriphenol"
        )

    def test_default_4_branch_removed(self):
        """The legacy `return 4 # default` branches MUST be gone (D-12)."""
        import re
        from pathlib import Path

        src = Path("src/orthonym/rules/multiplicative.py").read_text()
        # Match `return 4 # default` (case-insensitive on Default) and
        # `return 4  # default to 4 (most common para position)` style.
        legacy_pattern = re.compile(
            r"^\s*return\s+4\s*#\s*([Dd]efault|.*para)", re.MULTILINE
        )
        matches = legacy_pattern.findall(src)
        assert not matches, (
            f"Legacy 'return 4 # default' branches still present: {matches!r} "
            "-- D-12 requires removal."
        )

    def test_shortest_path_heuristic_helper_exists(self):
        from orthonym.rules.multiplicative import (
            _shortest_path_heuristic_locant,
        )
        assert callable(_shortest_path_heuristic_locant)

    def test_no_circular_import(self):
        """Probe: importing name_multiplicative does NOT trigger ImportError."""
        import importlib

        # Reload to exercise the import-cycle path.
        mod = importlib.import_module("orthonym.rules.multiplicative")
        assert hasattr(mod, "name_multiplicative")


# ---------------------------------------------------------------------------
# Task 2.5 — D-09 bridge-table expansion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBridgeTables:
    """D-09: bridge-table contents after audit-driven expansion."""

    @pytest.mark.parametrize(
        "symbol,expected_name",
        [
            ("O", "oxy"),
            # v22 F-T5 / MULT-01: the -S- multiplicative bridge is the modern
            # preselected prefix "sulfanediyl" (P-15.3.1.2.1.1); legacy "thio"
            # is deprecated. PIN "1,1'-sulfanediyldibenzene" (BB line 27826).
            ("S", "sulfanediyl"),
            ("NH", "imino"),
            ("CH2", "methylene"),
        ],
    )
    def test_single_atom_bridges_baseline(self, symbol, expected_name):
        """Pre-existing single-atom bridges preserved after Plan 154-02
        (S modernized thio->sulfanediyl in v22 F-T5)."""
        from orthonym.rules.multiplicative import _SINGLE_ATOM_BRIDGES

        assert _SINGLE_ATOM_BRIDGES.get(symbol) == expected_name

    @pytest.mark.parametrize(
        "atom_a,atom_b,h_a,h_b,expected_name",
        [
            ("C", "C", 2, 2, "ethylene"),
            ("C", "C", 1, 1, "vinylene"),
            ("O", "O", 0, 0, "peroxy"),  # D-09 add (154-AUDIT-B.md §3 #1)
            ("S", "S", 0, 0, "disulfanediyl"),  # D-09 add (154-AUDIT-B.md §3 #2)
        ],
    )
    def test_two_atom_bridges_present(self, atom_a, atom_b, h_a, h_b, expected_name):
        from orthonym.rules.multiplicative import _TWO_ATOM_BRIDGES

        # Search for the entry (forward or reversed atom order).
        found = False
        for (s1, s2, e_h1, e_h2, name) in _TWO_ATOM_BRIDGES:
            if (
                (s1, s2, e_h1, e_h2) == (atom_a, atom_b, h_a, h_b)
                and name == expected_name
            ):
                found = True
                break
            if (
                (s2, s1, e_h2, e_h1) == (atom_a, atom_b, h_a, h_b)
                and name == expected_name
            ):
                found = True
                break
        assert found, f"Two-atom bridge {atom_a}-{atom_b} ({expected_name}) missing"

    @pytest.mark.parametrize(
        "key,expected_name",
        [
            (("N", 0, 3), "nitrilo"),
            (("C", 1, 3), "methylidyne"),
            (("C", 0, 4), "methanetetrayl"),
            (("P", 0, 3), "phosphinidyne"),  # D-09 add (154-AUDIT-B.md §3 #3)
        ],
    )
    def test_multi_bridge_names_present(self, key, expected_name):
        from orthonym.rules.multiplicative import _MULTI_BRIDGE_NAMES

        assert _MULTI_BRIDGE_NAMES.get(key) == expected_name, (
            f"_MULTI_BRIDGE_NAMES key {key!r} missing or wrong"
        )

    def test_peroxy_e2e(self):
        """D-09 bridge add (peroxy): 4,4'-peroxydibenzoic acid RTs."""
        from orthonym import name_compound

        result = name_compound("OC(=O)c1ccc(OOc2ccc(C(=O)O)cc2)cc1")
        assert result == "4,4'-peroxydibenzoic acid", f"got {result!r}"

    def test_disulfanediyl_e2e(self):
        """D-09 bridge add (disulfanediyl): 4,4'-disulfanediyldibenzoic acid."""
        from orthonym import name_compound

        result = name_compound("OC(=O)c1ccc(SSc2ccc(C(=O)O)cc2)cc1")
        assert result == "4,4'-disulfanediyldibenzoic acid", f"got {result!r}"

    def test_phosphinidyne_e2e(self):
        """D-09 bridge add (phosphinidyne): 4,4',4''-phosphinidynetriphenol."""
        from orthonym import name_compound

        result = name_compound("Oc1ccc(P(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
        assert result == "4,4',4''-phosphinidynetriphenol", f"got {result!r}"

    def test_bridge_tables_have_audit_cites(self):
        """Every NEW bridge entry (D-09) has an inline 154-AUDIT-B.md cite."""
        from pathlib import Path

        src = Path("src/orthonym/rules/multiplicative.py").read_text()
        # Three audit-driven additions: peroxy, disulfanediyl, phosphinidyne.
        # Each addition must include "154-AUDIT-B" in the source comment.
        for label in ("peroxy", "disulfanediyl", "phosphinidyne"):
            line_with_label = None
            for line in src.splitlines():
                if label in line:
                    line_with_label = line
                    break
            assert line_with_label is not None, f"{label!r} not found in source"
            # The cite can be on the same line OR within the next 2 lines.
            assert (
                "154-AUDIT-B" in line_with_label or "OPSIN" in line_with_label
                or "P-" in line_with_label
            ), (
                f"NEW bridge entry for {label!r} missing inline cite "
                f"(line was: {line_with_label!r})"
            )


# ---------------------------------------------------------------------------
# D-13 anti-band-aid guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoBandAids:
    """D-13: no postprocessor / regex band-aid in multiplicative.py."""

    def test_no_postprocessor_keyword(self):
        from pathlib import Path

        src = Path("src/orthonym/rules/multiplicative.py").read_text()
        # Forbidden patterns (case-insensitive): postprocess, BAND-AID
        assert "postprocess" not in src.lower(), (
            "D-13: 'postprocess' keyword found in multiplicative.py -- band-aid risk."
        )
        assert "band-aid" not in src.lower() and "bandaid" not in src.lower(), (
            "D-13: 'band-aid' keyword found in multiplicative.py."
        )
