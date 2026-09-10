"""Unit tests for the lipid backbone-aware assembler (a phase, -01).

Target name form = Form B (systematic substitutive / functional-class),
empirically OPSIN-round-trip-verified (180-internal notes). Each test asserts
the exact PIN the assembler must produce.

WAVE 0 CONTRACT: these import the not-yet-built `name_lipid` symbol INSIDE each
test body (NOT at module level) so `pytest --collect-only` succeeds; they are
RED at run time until Waves 1-4 build the subsystem.
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Glycerides — ester PCG on the glycerol-derived parent
# ---------------------------------------------------------------------------

class TestGlycerides:
    def test_tag_saturated(self):
        """Tripalmitin → propane-1,2,3-triyl trihexadecanoate."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401 (RED in Wave 0)
        smiles = "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC"
        assert name_compound(smiles) == "propane-1,2,3-triyl trihexadecanoate"

    def test_tag_unsaturated(self):
        """Triolein → propane-1,2,3-triyl tris[(9Z)-octadec-9-enoate]."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = ("CCCCCCCC/C=C\\CCCCCCCC(=O)OCC("
                  "COC(=O)CCCCCCC/C=C\\CCCCCCCC)"
                  "OC(=O)CCCCCCC/C=C\\CCCCCCCC")
        assert name_compound(smiles) == "propane-1,2,3-triyl tris[(9Z)-octadec-9-enoate]"

    def test_mag(self):
        """1-monopalmitin → 2,3-dihydroxypropyl hexadecanoate."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = "CCCCCCCCCCCCCCCC(=O)OCC(O)CO"
        assert name_compound(smiles) == "2,3-dihydroxypropyl hexadecanoate"

    def test_dag(self):
        """1,2-dipalmitoyl-3-OH glycerol → 3-hydroxypropane-1,2-diyl dihexadecanoate."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = "CCCCCCCCCCCCCCCC(=O)OCC(CO)OC(=O)CCCCCCCCCCCCCCC"
        assert name_compound(smiles) == "3-hydroxypropane-1,2-diyl dihexadecanoate"


# ---------------------------------------------------------------------------
# Phospholipids — functional-class phosphate diester
# ---------------------------------------------------------------------------

class TestPhospholipids:
    def test_pc(self):
        """1,2-dipalmitoyl-sn-glycero-3-phosphocholine →
        [(2R)-2,3-bis(hexadecanoyloxy)propyl] 2-(trimethylazaniumyl)ethyl phosphate."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = ("CCCCCCCCCCCCCCCC(=O)OC[C@H](COP([O-])(=O)OCC[N+](C)(C)C)"
                  "OC(=O)CCCCCCCCCCCCCCC")
        assert name_compound(smiles) == (
            "[(2R)-2,3-bis(hexadecanoyloxy)propyl] 2-(trimethylazaniumyl)ethyl phosphate")

    def test_pe(self):
        """1,2-dipalmitoyl-sn-glycero-3-phosphoethanolamine →
        (2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl dihexadecanoate."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCCN)OC(=O)CCCCCCCCCCCCCCC"
        name = name_compound(smiles)
        assert name in {
            "(2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl dihexadecanoate",
            "2-aminoethyl [(2R)-2,3-bis(hexadecanoyloxy)propyl] phosphate",
        }


# ---------------------------------------------------------------------------
# Sphingolipids / ceramides — amide PCG, sphingoid N-substituent
# ---------------------------------------------------------------------------

class TestSphingolipids:
    def test_ceramide(self):
        """N-hexadecanoylsphingosine →
        N-[(2S,3R,4E)-1,3-dihydroxyoctadec-4-en-2-yl]hexadecanamide."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO)NC(=O)CCCCCCCCCCCCCCC"
        assert name_compound(smiles) == (
            "N-[(2S,3R,4E)-1,3-dihydroxyoctadec-4-en-2-yl]hexadecanamide")


# ---------------------------------------------------------------------------
# Glyco-lipids — single sugar via reused Phase-176 machinery
# ---------------------------------------------------------------------------

class TestGlycoLipids:
    def test_glyco_ceramide(self):
        """β-D-galactosylceramide →
        N-[(2S,3R,4E)-1-(β-D-galactopyranosyloxy)-3-hydroxyoctadec-4-en-2-yl]hexadecanamide."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = ("CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H]("
                  "CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)"
                  "NC(=O)CCCCCCCCCCCCCCC")
        assert name_compound(smiles) == (
            "N-[(2S,3R,4E)-1-(β-D-galactopyranosyloxy)-3-hydroxyoctadec-4-en-2-yl]"
            "hexadecanamide")

    def test_glyco_glycerolipid(self):
        """3-O-β-D-galactopyranosyl-1,2-di-O-octadecanoyl-sn-glycerol →
        (2S)-3-(β-D-galactopyranosyloxy)propane-1,2-diyl dioctadecanoate (Blue Book P-107.4.2)."""
        from orthonym.rules.lipids import name_lipid  # noqa: F401
        smiles = ("CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](OC(=O)CCCCCCCCCCCCCCCCC)"
                  "CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O")
        assert name_compound(smiles) == (
            "(2S)-3-(β-D-galactopyranosyloxy)propane-1,2-diyl dioctadecanoate")


# ---------------------------------------------------------------------------
# Hard-gate negatives  — must NOT be claimed by the lipid path
# ---------------------------------------------------------------------------

class TestHardGateNegatives:
    def test_wax_ester_names_normally(self):
        """A wax ester (no glycerol/sphingoid backbone) must name via the general
        ester pipeline, NOT the lipid backbone path (detector returns None)."""
        name = name_compound("CCCCCCCCCCCCCCCC(=O)OCC")
        assert name is not None and name != "unknown organic compound"
        # ethyl hexadecanoate (or hexadecanoate ester) — NOT a lipid-backbone form
        assert "propane-1,2,3-triyl" not in name
