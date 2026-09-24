"""Integration test: closure — substituted-terphenyl round-trip via
name_compound + OPSIN with InChI L1 comparison.

a phase-04: the canary at test_canary_name_stability.py:1196
previously stored a wrong-prime form (1,1':4,1''-terphenyl) that contradicted
the unit test at test_ring_assemblies_3plus.py:276 (1,1':4',1''-terphenyl).
A frozen-string canary alone cannot detect prime-dropping bugs because the
canary's expected-string can be silently updated to match a buggy emission.

This round-trip test pipes the SMILES through name_compound, then through
OPSIN to reparse the name, and asserts the InChI L1 (skeleton) layer matches.
If the engine emits 1,1':4,1''-terphenyl (wrong-prime), OPSIN will parse it
to a DIFFERENT topology than the input SMILES and the InChI L1 assertion
will fail — surfacing the regression even if a frozen-string canary masks it.
"""
from pathlib import Path

import pytest
from rdkit import Chem


# The exact substituted-terphenyl SMILES from test_canary_name_stability.py:1191
SUBSTITUTED_TERPHENYL_SMI = "COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)c(O)c1-c1ccc(O)c(O)c1"
UNSUBSTITUTED_TERPHENYL_SMI = "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"


def _opsin_available():
    """Match the helper pattern from tests/integration/test_canary_rt75.py."""
    import shutil
    from tests.support.jars import jar_or_none
    return jar_or_none() is not None and shutil.which("java") is not None


class TestSubstitutedTerphenylRoundTrip:
    """a phase-04 closure — round-trip OPSIN gate against
    prime-dropping regressions in substituted ring-assembly emission."""

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(), reason="OPSIN jar not available")
    def test_substituted_terphenyl_inchi_l1_match(self):
        """Pipe substituted terphenyl SMILES through name_compound +
        OPSIN; assert InChI L1 (skeleton) match. A prime-dropping bug
        (e.g., emitting 1,1':4,1''-terphenyl instead of 1,1':4',1''-terphenyl)
        causes OPSIN to parse a different topology and the InChI L1 layer
        diverges — surfacing the regression."""
        from orthonym.namer import name_compound

        mol = Chem.MolFromSmiles(SUBSTITUTED_TERPHENYL_SMI)
        assert mol is not None, "Canary SMILES no longer parses"

        name = name_compound(SUBSTITUTED_TERPHENYL_SMI)
        assert name is not None and name != "unknown", (
            f"name_compound returned None/unknown on canary input: {name!r}"
        )

        # a phase-04 hard gate: the emitted name MUST contain the
        # primed form. This is a fast-fail before round-trip — if the engine
        # drops the prime, surface it immediately rather than waiting for
        # OPSIN to disagree.
        assert "1,1':4',1''-terphenyl" in name, (
            f"BLK-02 regression: emitted name lacks the primed-4 form "
            f"required by IUPAC P-28.2.1. Got: {name!r}"
        )

        # Round-trip via OPSIN: name -> SMILES -> InChI L1
        from tests.integration.test_canary_rt75 import _opsin_parse_batch

        parsed = _opsin_parse_batch([name]).get(name)
        if parsed is None:
            pytest.fail(
                f"OPSIN cannot parse the emitted name {name!r}. This is a "
                f"BLK-02 regression: a frozen-string canary would have "
                f"masked it, but the round-trip gate catches it."
            )

        inchi_in = Chem.MolToInchi(Chem.MolFromSmiles(SUBSTITUTED_TERPHENYL_SMI))
        inchi_rt = Chem.MolToInchi(Chem.MolFromSmiles(parsed))
        # Layer 1 (skeleton) match — strip /h, /b, /t etc. and compare /c layer
        l1_in = inchi_in.split("/c")[0] if "/c" in inchi_in else inchi_in
        l1_rt = inchi_rt.split("/c")[0] if "/c" in inchi_rt else inchi_rt
        assert l1_in == l1_rt, (
            f"InChI L1 mismatch on round-trip:\n"
            f"  input SMILES: {SUBSTITUTED_TERPHENYL_SMI}\n"
            f"  emitted name: {name!r}\n"
            f"  OPSIN reparsed SMILES: {parsed}\n"
            f"  L1 in:  {l1_in}\n"
            f"  L1 rt:  {l1_rt}"
        )

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(), reason="OPSIN jar not available")
    def test_unsubstituted_terphenyl_round_trip_unchanged(self):
        """Sanity check: the unsubstituted unit-test path (which was
        VERIFIED before closure) must still round-trip after any
        engine changes. This guards Scenario A (engine fix) against
        regressing the unsubstituted path while fixing the substituted
        path."""
        from orthonym.namer import name_compound
        from tests.integration.test_canary_rt75 import _opsin_parse_batch

        name = name_compound(UNSUBSTITUTED_TERPHENYL_SMI)
        assert name == "1,1':4',1''-terphenyl", (
            f"Unsubstituted terphenyl path regressed: got {name!r}, "
            f"expected '1,1':4',1''-terphenyl'"
        )
        parsed = _opsin_parse_batch([name]).get(name)
        assert parsed is not None, f"OPSIN cannot parse {name!r}"
        inchi_in = Chem.MolToInchi(Chem.MolFromSmiles(UNSUBSTITUTED_TERPHENYL_SMI))
        inchi_rt = Chem.MolToInchi(Chem.MolFromSmiles(parsed))
        l1_in = inchi_in.split("/c")[0] if "/c" in inchi_in else inchi_in
        l1_rt = inchi_rt.split("/c")[0] if "/c" in inchi_rt else inchi_rt
        assert l1_in == l1_rt
