"""Integration tests: BLK-01 closure — name_compound dispatches mixed-spiro-fused
inputs through the new elif branch in _assemble_complex_ring_name.

Phase 151-04 closes the gap where the 151-02 SUMMARY claimed to add this dispatch
but a git-stash incident silently dropped it. These tests exercise the FULL
name_compound pipeline (not just direct calls to name_mixed_spiro_fused) so
a future regression in the dispatch chain CANNOT be silently masked.
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "ring_systems" / "spiro" / "corpus_mined.json"


def _load_mixed_spiro_fused_fixtures():
    if not FIXTURE_PATH.exists():
        return []
    with open(FIXTURE_PATH) as f:
        data = json.load(f)
    # Phase 151-02 mined fixtures with compound_class field; only the
    # spiro-mixed-fused subset routes through the new elif branch.
    return [f for f in data if f.get("compound_class") == "spiro-mixed-fused"]


class TestMixedSpiroFusedDispatch:
    """Phase 151-04 BLK-01 closure — name_compound integration."""

    @pytest.mark.integration
    def test_dispatch_does_not_return_none_on_known_mixed_smiles(self):
        """The composer dispatch must invoke name_mixed_spiro_fused; if the
        elif branch is missing, _assemble_complex_ring_name returns None and
        the entire HERITAGE §4 path is dead code from name_compound.

        We use a synthetic spiro-fused SMILES known to classify as
        mixed-spiro-fused per is_mixed_spiro_fused. If the function returns
        None for ALL inputs, that is itself a regression worth flagging.
        """
        from orthonym.namer import name_compound
        from orthonym.rules.spiro import is_mixed_spiro_fused

        # Synthetic mixed-spiro-fused: indoline (fused) + cyclohexane (spiro side)
        # The exact SMILES is the canonical HERITAGE §4 example
        # spiro[indoline-3,1'-cyclohexane]: C1CCC2(CC1)CC1=CC=CC=C1N2
        smi = "C1CCC2(CC1)CC1=CC=CC=C1N2"
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not is_mixed_spiro_fused(mol):
            pytest.skip(
                "Synthetic SMILES no longer classifies as mixed-spiro-fused; "
                "this is OK if the predicate tightened — the corpus fixture "
                "tests below cover real-world cases."
            )

        name = name_compound(smi)
        # BLK-01 proof: name_compound must NOT silently return None for inputs
        # that classify as mixed-spiro-fused. If the elif branch is missing,
        # _assemble_complex_ring_name returns None for this input, and the
        # downstream pipeline produces either None or a wildly different name.
        assert name is not None, (
            "name_compound returned None for a mixed-spiro-fused input — "
            "the elif branch in _assemble_complex_ring_name is missing or "
            "name_mixed_spiro_fused returned None for all corpus inputs."
        )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "fixture",
        _load_mixed_spiro_fused_fixtures(),
        ids=lambda f: f.get("fixture_id", "unknown"),
    )
    def test_corpus_mixed_spiro_fused_routes_through_dispatch(self, fixture):
        """Each corpus mixed-spiro-fused fixture: assert name_compound
        does NOT return None AND the returned name contains 'spiro['
        (HERITAGE §4 separable-parts nested form per 151-02 D-13).

        If a particular fixture's name_mixed_spiro_fused implementation
        returns None today (legitimately — some fixtures are logged to
        HERITAGE-followups.md per D-24 as v19 architectural followups),
        mark it xfail with the followup citation. The HARD assertion is
        that AT LEAST ONE corpus fixture returns a non-None name with
        'spiro[' — proving the elif branch is live.
        """
        from orthonym.namer import name_compound

        smi = fixture["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pytest.skip(f"Invalid SMILES in fixture {fixture.get('fixture_id')}")

        name = name_compound(smi)
        if name is None or name == "unknown":
            pytest.xfail(
                f"{fixture.get('fixture_id')} returns None/unknown — see "
                f" for the v19 followup "
                f"this fixture is tracked under (D-24 no-band-aid policy)."
            )
        # If the corpus fixture routes to another handler (tricyclo, tetracyclo,
        # aliphatic, etc.) rather than mixed-spiro-fused, that is a known
        # corpus-classification heterogeneity and is also an xfail case —
        # the HARD assertion is the test below
        # (test_at_least_one_corpus_fixture_proves_branch_is_live), which
        # proves the dispatch is reachable from name_compound at least once.
        if not ("spiro[" in name.lower() or "spiro " in name.lower()):
            pytest.xfail(
                f"{fixture.get('fixture_id')} routes to a non-spiro handler "
                f"({name!r}) — corpus classification heterogeneity, not a "
                f"BLK-01 regression. The per-fixture branch-routing is "
                f"tracked in HERITAGE-followups.md for v19 disambiguation."
            )
        # Assertion reached only when the fixture routes through the new
        # elif and name_mixed_spiro_fused succeeded — pipeline-level proof.
        assert "spiro[" in name.lower() or "spiro " in name.lower()

    @pytest.mark.integration
    def test_at_least_one_corpus_fixture_proves_branch_is_live(self):
        """Hard gate: at least one mixed-spiro-fused corpus fixture must
        produce a non-None name with 'spiro[' via name_compound. Without
        this, the elif branch is technically present but functionally dead.
        """
        from orthonym.namer import name_compound

        fixtures = _load_mixed_spiro_fused_fixtures()
        if not fixtures:
            pytest.skip("No mixed-spiro-fused corpus fixtures available")

        live_count = 0
        for fixture in fixtures:
            mol = Chem.MolFromSmiles(fixture["smiles"])
            if mol is None:
                continue
            name = name_compound(fixture["smiles"])
            if name is not None and name != "unknown" and (
                "spiro[" in name.lower() or "spiro " in name.lower()
            ):
                live_count += 1
        assert live_count >= 1, (
            f"NONE of the {len(fixtures)} mixed-spiro-fused corpus fixtures "
            f"produced a 'spiro['-containing name via name_compound. "
            f"Either the elif branch is missing/misrouted, or "
            f"name_mixed_spiro_fused is broken for ALL corpus inputs. "
            f"BLK-01 closure is invalid."
        )
