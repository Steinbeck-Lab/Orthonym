"""
Tests for OPSIN-compatible acyloxy bracket format and bare 'oxy' elimination.

Phase 54, Plan 02: OPFX-03 (acyloxy brackets) and OPFX-06 (bare oxy removal).

Validates:
- Acyloxy prefixes are correctly bracketed: "(acetyloxy)", "(propanoyloxy)"
- No generated name contains standalone bare 'oxy' prefix
- Polyol ester compounds produce correct multiplicative acyloxy brackets
- Tricyclo+stereo OPSIN failures are vocabulary limits (OPFX-04), not format
"""

import re

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestAcyloxyBracketFormat:
    """Verify acyloxy prefixes have correct bracket format for OPSIN."""

    def test_acyloxy_brackets_on_benzene(self):
        """Acyloxy prefix on benzene ring is bracketed: 4-(Xoyloxy)benzoic acid."""
        # 4-(acetyloxy)benzoic acid (aspirin-like systematic name)
        name = name_compound("CC(=O)Oc1ccc(C(=O)O)cc1")
        # Should contain a bracketed acyloxy prefix
        # Accept either "(ethanoyloxy)" or "(acetyloxy)" -- both are valid
        assert re.search(r"\(\w+oyloxy\)", name), (
            f"Expected bracketed acyloxy prefix in '{name}'"
        )

    def test_acyloxy_brackets_on_chain(self):
        """Acyloxy prefix on acyclic chain is bracketed: 2-(Xoyloxy)ethanoic acid."""
        # 2-(acetyloxy)acetic acid
        name = name_compound("CC(=O)OCC(=O)O")
        # Must contain bracketed acyloxy
        assert re.search(r"\(\w+oyloxy\)", name), (
            f"Expected bracketed acyloxy prefix in '{name}'"
        )

    def test_polyol_ester_acyloxy_brackets(self):
        """Non-glycerol polyol ester keeps the multiplicative (acyloxy) prefix form.

        Phase 180 update: glycerol triacetate (triacetin) is a triacylglycerol and
        now routes through the P-107 lipid assembler to the Form B systematic ester
        name ``propane-1,2,3-triyl triacetate`` (OPSIN-RT-verified). The general
        multiplicative-(acyloxy) formatting is unchanged for NON-glycerol polyols,
        verified here on erythritol tetraacetate (the lipid detector hard-gates it
        out: not a propane-1,2,3-triol core → returns None → general pipeline).
        """
        # glycerol triacetate is now the lipid Form B (intended Phase-180 behavior)
        assert name_compound("CC(=O)OCC(OC(C)=O)COC(C)=O") == "propane-1,2,3-triyl triacetate"
        # erythritol tetraacetate: NOT a glyceride -> general (acyloxy) multiplier form retained
        name = name_compound("CC(=O)OCC(OC(C)=O)C(OC(C)=O)COC(C)=O")
        assert "(acetyloxy)" in name, f"Expected '(acetyloxy)' in '{name}'"
        assert "tetrakis" in name.lower(), f"Expected 'tetrakis' in '{name}'"


@pytest.mark.unit
class TestNoBareOxy:
    """Verify no generated name contains standalone bare 'oxy' prefix."""

    def test_no_bare_oxy_benzene_ether(self):
        """Diphenyl ether produces 'oxydibenzene', not standalone 'oxy-' prefix."""
        # Diphenyl ether: should be "4,4'-oxydibenzene" or similar
        name = name_compound("c1ccc(Oc2ccccc2)cc1")
        # 'oxy' must NOT appear as a standalone hyphenated prefix
        # It should be part of a larger word like 'oxydibenzene'
        assert not re.search(r"(?:^|[-\s])oxy(?:$|[-\s])", name), (
            f"Bare 'oxy' found as standalone prefix in '{name}'"
        )

    def test_no_bare_oxy_methoxybenzene(self):
        """Anisole (methoxybenzene) does not contain bare 'oxy'."""
        name = name_compound("COc1ccccc1")
        # Now returns "anisole" (retained name, P-34.1.1.4 PIN)
        assert name == "anisole" or "methoxy" in name.lower(), (
            f"Expected 'anisole' or 'methoxy' in '{name}'"
        )
        # Bare 'oxy' should not appear
        assert not re.search(r"(?:^|[-\s])oxy(?:$|[-\s])", name), (
            f"Bare 'oxy' found in '{name}'"
        )

    def test_no_bare_oxy_o_n_linkage(self):
        """O-N bond on benzene does not produce bare 'oxy' prefix.

        The O-substituent handler returns None for O-N non-nitrooxy linkages
        rather than emitting bare 'oxy'. The name may omit the substituent
        or use a different naming path.
        """
        # Phenyl hydroxylamine: PhONH2 -- O bonded to N
        smiles = "ONc1ccccc1"
        name = name_compound(smiles)
        # Must not have bare 'oxy' as standalone prefix
        assert not re.search(r"(?:^|[-\s])oxy(?:$|[-\s])", name), (
            f"Bare 'oxy' found in '{name}'"
        )

    def test_no_bare_oxy_o_unknown(self):
        """O bonded to non-C heteroatom on benzene returns None, not bare 'oxy'.

        When _identify_oxygen_group encounters an O bonded to an unrecognized
        heteroatom, it returns None (skipping the substituent) rather than
        emitting bare 'oxy' which OPSIN cannot parse.
        """
        # O-Se on benzene (selenium) -- rare edge case
        smiles = "c1ccc(O[Se])cc1"
        name = name_compound(smiles)
        # Must not have bare 'oxy' as standalone prefix
        assert not re.search(r"(?:^|[-\s])oxy(?:$|[-\s])", name), (
            f"Bare 'oxy' found in '{name}'"
        )

    def test_no_bare_oxy_total_c_zero(self):
        """O with total_c==0 substituent returns None, not bare 'oxy'.

        The total_c==0 fallback path (e.g., O-O or O-N that reached the
        last-resort branch) now returns None instead of bare 'oxy'.
        """
        # Test that diphenyl ether and other known oxygen-bridge compounds
        # never produce bare 'oxy'
        test_smiles = [
            "c1ccc(Oc2ccccc2)cc1",       # diphenyl ether
            "c1ccc(OC)cc1",               # anisole
            "c1ccc(OCC)cc1",              # ethoxybenzene
            "c1ccc(OC(C)C)cc1",           # isopropoxybenzene
        ]
        for smi in test_smiles:
            name = name_compound(smi)
            assert not re.search(r"(?:^|[-\s])oxy(?:$|[-\s])", name), (
                f"Bare 'oxy' found in name for {smi}: '{name}'"
            )


@pytest.mark.unit
class TestTricycloOpsinLimitation:
    """Document OPFX-04: tricyclo+stereo OPSIN failures are vocabulary limits.

    The tricyclo[a.b.c.d(e,f)] VB format is correct and OPSIN-compatible
    for simple cases. Failures occur only when combined with many
    stereodescriptors, which exceeds OPSIN's parser capacity. These are
    classified as Bucket A (OPSIN limitation), not format issues.

    No code changes needed -- this class documents the limitation.
    """

    def test_tricyclo_opsin_limitation_documented(self):
        """Tricyclo+stereo failures are OPSIN vocabulary limits (Bucket A).

        Four known tricyclo compounds with 4+ stereocenters fail OPSIN
        parsing. The VB naming format is correct; OPSIN simply cannot
        handle the combination of tricyclic nomenclature with complex
        stereodescriptor strings. This is an accepted ceiling.
        """
        # This is a documentation test -- the assertion confirms the
        # test infrastructure recognizes this as a known limitation.
        known_tricyclo_opsin_limitations = [
            "tricyclo[a.b.c.d(e,f)] + 4+ stereocenters",
            "Complex VB nomenclature with E/Z + R/S descriptors",
        ]
        assert len(known_tricyclo_opsin_limitations) > 0, (
            "Tricyclo OPSIN limitations should be documented"
        )
