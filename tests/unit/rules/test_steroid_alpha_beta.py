"""Unit tests for steroid ring-face α/β stereoparent emission (Phase 181, WSC-02).

IUPAC P-101.2.6 ring-face descriptors (`3beta`, `5alpha`) replace whole-graph CIP
R/S on steroid scaffolds. Latin spelling is the canonical output form locked for
this phase (OPSIN parses both Greek and Latin; Latin is ASCII-safe and matches the
dominant corpus reference spelling).

WAVE 0 CONTRACT: imports the not-yet-built `collect_steroid_alpha_beta` symbol
INSIDE each test body (NOT at module level) so `pytest --collect-only` succeeds;
RED at run time until Waves 1-2 build + wire the converter.
"""

import re

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# The 9 OPSIN-verified α/β assembly exemplars (181-00 <gold_smiles>)
# ---------------------------------------------------------------------------

class TestAlphaBetaAssembly:
    def test_cholestane_3b_ol(self):
        """Row 1: 5α-cholestan-3β-ol (free-config C-5 on stem + 3β -ol suffix)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"
        assert name_compound(smiles) == "5alpha-cholestan-3beta-ol"

    def test_cholestane_3a_ol(self):
        """Row 2: 5α-cholestan-3α-ol."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"
        assert name_compound(smiles) == "5alpha-cholestan-3alpha-ol"

    def test_5beta_case(self):
        """Row 3: 5β-cholestan-3α-ol (the D-13 5β requirement)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"
        assert name_compound(smiles) == "5beta-cholestan-3alpha-ol"

    def test_androstane_multi(self):
        """Row 4: 3β-hydroxy-5α-androstan-17β-ol (multi-stereo: 3β prefix + 5α stem + 17β suffix)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "O[C@@H]1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O"
        assert name_compound(smiles) == "3beta-hydroxy-5alpha-androstan-17beta-ol"

    def test_delta5_c5_not_cited(self):
        """Row 5: androst-5-en-3β-ol (Δ5 → C-5 sp2 NOT cited; 3β on the -ol suffix).

        A non-retained Δ5 androstene (cholesterol's SMILES is intercepted by the
        retained-name lookup and emits "cholesterol", so it is not a whole-graph-R/S
        defect; this androstenol currently emits (3S,8S,9S,10R,13S,14S)-androst-5-en-3-ol
        — a genuine defect that exercises the Δ5/C-5-sp2-not-cited path).
        """
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CCC2"
        n = name_compound(smiles)
        assert n == "androst-5-en-3beta-ol", n
        # C-5 is sp2 (Δ5) → must NOT carry a greek token (Pitfall 1)
        assert "5alpha" not in n and "5beta" not in n, n

    def test_pregnane_sidechain(self):
        """Row 6: 5α-pregnane-3β,20-diol (side-chain C-20 present; 20 stays plain)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC([C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)O"
        assert name_compound(smiles) == "5alpha-pregnane-3beta,20-diol"

    def test_sidechain_coexistence(self):
        """Row 7: (22R)-cholest-5-ene-3β,20,22-triol — side-chain (22R) leading block + 3β ring inline."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC(C)CC[C@H]([C@@](C)([C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)O)O"
        assert name_compound(smiles) == "(22R)-cholest-5-ene-3beta,20,22-triol"

    def test_stigmastane_24s(self):
        """Row 8: (24S)-stigmast-5-en-3β-ol (stigmastane + side-chain (24S) + Δ5)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC[C@@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)C(C)C"
        assert name_compound(smiles) == "(24S)-stigmast-5-en-3beta-ol"

    def test_ketone_suffix(self):
        """Row 9: 3-oxo-5α-androstan-17β-ol (ketone C not a stereocentre → 3-oxo plain)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "O=C1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O"
        n = name_compound(smiles)
        assert n == "3-oxo-5alpha-androstan-17beta-ol", n
        assert "3alpha-oxo" not in n and "3beta-oxo" not in n, n


# ---------------------------------------------------------------------------
# Negative / protect guards (mirror test_lipids.py None-on-miss class)
# ---------------------------------------------------------------------------

class TestProtect:
    def test_bridgehead_suppression(self):
        """Natural-config bridgeheads C-8/9/10/13/14 are SUPPRESSED, never cited (D-04c)."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        smiles = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"
        n = name_compound(smiles)
        for b in (8, 9, 10, 13, 14):
            assert f"{b}alpha" not in n and f"{b}beta" not in n, f"bridgehead {b} leaked: {n}"

    def test_no_mix_fallback(self):
        """D-08: a steroid name must NEVER mix ring α/β tokens with a whole-graph ring R/S block.

        Either the molecule resolves fully to α/β (no ring R/S block at all) or it falls back to
        the whole-graph R/S leading block (no α/β tokens). A side-chain `(22R)-` single-centre block
        is allowed; a multi-centre ring CIP block `(3R,5S,...)` is the forbidden whole-graph form.
        """
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        for smiles in (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O",  # row 1
            "O[C@@H]1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O",          # row 4
            "O=C1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O",               # row 9
        ):
            n = name_compound(smiles)
            has_ab = bool(re.search(r"\d(alpha|beta)", n))
            has_ring_rs_block = bool(re.search(r"\(\d+[RSrs],\d+[RSrs]", n))  # ≥2-centre leading CIP block
            assert not (has_ab and has_ring_rs_block), f"mixed α/β + ring R/S block: {n}"

    def test_alkaloid_keeps_rs(self):
        """Non-steroid NP (atropine, an alkaloid) keeps whole-graph R/S — α/β branch must not leak."""
        from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta  # noqa: F401
        n = name_compound("CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(=O)C(CO)c1ccccc1")
        assert "alpha" not in n.lower() and "beta" not in n.lower(), n
