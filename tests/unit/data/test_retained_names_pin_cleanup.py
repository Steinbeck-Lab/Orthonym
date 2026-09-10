"""
Tests for IUPAC 2013 PIN cleanup of retained names (a phase-01).

Verifies:
- Acetone removed from RETAINED_NAMES (PIN is propan-2-one per
- Cresol isomers removed (PINs are 2/3/4-methylphenol)
- Isobutyric/isovaleric/pivalic acid removed (PINs are systematic names)
- Adamantane added to BICYCLO_RETAINED_NAMES
- Cubane added to BICYCLO_RETAINED_NAMES
"""

import glob
import shutil
import subprocess

import pytest
from orthonym.data.retained_names import RETAINED_NAMES
from orthonym.data import ALL_RETAINED_NAMES
from orthonym.data.bicyclo_systems import BICYCLO_RETAINED_NAMES
from orthonym import name_compound


def _find_opsin_jar():
    for pat in (
        "opsin-cli-*-jar-with-dependencies.jar",
        "opsin.jar",
        "opsin/opsin-cli-*-jar-with-dependencies.jar",
        "opsin/opsin-cli/target/opsin-cli-*-jar-with-dependencies.jar",
    ):
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


_OPSIN_JAR = _find_opsin_jar()
_OPSIN_AVAILABLE = bool(_OPSIN_JAR) and shutil.which("java") is not None


def _opsin_smiles(name):
    """Name -> SMILES via OPSIN over STDIN.

    OPSIN 2.9.0 reads the name on stdin (post-`-osmi` argv is treated as a
    file; the conftest `opsin_to_smiles` fixture's argv form returns None on
    this version). stdout carries only the SMILES; the banner goes to stderr.
    """
    if not _OPSIN_AVAILABLE:
        return None
    try:
        result = subprocess.run(
            ["java", "-jar", _OPSIN_JAR, "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    out = result.stdout.strip()
    return out or None


def _inchi(smiles):
    """InChI for a SMILES (None if unparseable). Used for structural round-trip."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol) if mol else None


class TestAcetoneRemoved:
    """Acetone should NOT be a retained name -- PIN is propan-2-one."""

    @pytest.mark.unit
    def test_acetone_not_in_retained_names_keys(self):
        """CC(C)=O should not be a key in RETAINED_NAMES."""
        assert "CC(C)=O" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_acetone_not_in_retained_names_values(self):
        """'acetone' should not appear as a value in RETAINED_NAMES."""
        # acetophenone contains 'acetone' substring, so check exact match
        for val in RETAINED_NAMES.values():
            assert val != "acetone", f"Found 'acetone' as retained name value"

    @pytest.mark.unit
    def test_acetone_smiles_produces_propan_2_one(self):
        """name_compound('CC(C)=O') should produce 'propan-2-one'."""
        result = name_compound("CC(C)=O")
        assert result == "propan-2-one", f"Got '{result}'"


class TestCresolsRemoved:
    """Cresol isomers should NOT be retained names -- PINs are methylphenols."""

    @pytest.mark.unit
    def test_4_cresol_not_in_retained_names(self):
        """Cc1ccc(O)cc1 should not be a key in RETAINED_NAMES."""
        assert "Cc1ccc(O)cc1" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_2_cresol_not_in_retained_names(self):
        """Cc1ccccc1O should not be a key in RETAINED_NAMES."""
        assert "Cc1ccccc1O" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_3_cresol_not_in_retained_names(self):
        """Cc1cccc(O)c1 should not be a key in RETAINED_NAMES."""
        assert "Cc1cccc(O)c1" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_no_cresol_values_in_retained_names(self):
        """No value in RETAINED_NAMES should contain 'cresol'."""
        for val in RETAINED_NAMES.values():
            assert "cresol" not in val.lower(), f"Found cresol in value: {val}"

    @pytest.mark.unit
    def test_4_methylphenol_naming(self):
        """name_compound('Cc1ccc(O)cc1') should produce '4-methylphenol'."""
        result = name_compound("Cc1ccc(O)cc1")
        assert result == "4-methylphenol", f"Got '{result}'"

    @pytest.mark.unit
    def test_2_methylphenol_naming(self):
        """name_compound('Cc1ccccc1O') should produce '2-methylphenol'."""
        result = name_compound("Cc1ccccc1O")
        assert result == "2-methylphenol", f"Got '{result}'"

    @pytest.mark.unit
    def test_3_methylphenol_naming(self):
        """name_compound('Cc1cccc(O)c1') should produce '3-methylphenol'."""
        result = name_compound("Cc1cccc(O)c1")
        assert result == "3-methylphenol", f"Got '{result}'"


class TestBranchedAcidsRemoved:
    """Isobutyric/isovaleric/pivalic should NOT be retained names."""

    @pytest.mark.unit
    def test_isobutyric_not_in_retained_names(self):
        """'isobutyric acid' should not appear as a retained name value."""
        for val in RETAINED_NAMES.values():
            assert val != "isobutyric acid", "Found 'isobutyric acid' as retained name"

    @pytest.mark.unit
    def test_isovaleric_not_in_retained_names(self):
        """'isovaleric acid' should not appear as a retained name value."""
        for val in RETAINED_NAMES.values():
            assert val != "isovaleric acid", "Found 'isovaleric acid' as retained name"

    @pytest.mark.unit
    def test_pivalic_not_in_retained_names(self):
        """'pivalic acid' should not appear as a retained name value."""
        for val in RETAINED_NAMES.values():
            assert val != "pivalic acid", "Found 'pivalic acid' as retained name"


class TestAdamantaneAdded:
    """Adamantane should be in BICYCLO_RETAINED_NAMES."""

    @pytest.mark.unit
    def test_adamantane_in_bicyclo_retained_names(self):
        """Adamantane canonical SMILES should be a key in BICYCLO_RETAINED_NAMES."""
        assert "C1C2CC3CC1CC(C2)C3" in BICYCLO_RETAINED_NAMES

    @pytest.mark.unit
    def test_adamantane_value_correct(self):
        """BICYCLO_RETAINED_NAMES should map adamantane SMILES to 'adamantane'."""
        assert BICYCLO_RETAINED_NAMES.get("C1C2CC3CC1CC(C2)C3") == "adamantane"

    @pytest.mark.unit
    def test_adamantane_naming_e2e(self):
        """name_compound('C1C2CC3CC1CC(C2)C3') should produce 'adamantane'."""
        result = name_compound("C1C2CC3CC1CC(C2)C3")
        assert result == "adamantane", f"Got '{result}'"


class TestCubaneAdded:
    """Cubane should be in BICYCLO_RETAINED_NAMES."""

    @pytest.mark.unit
    def test_cubane_in_bicyclo_retained_names(self):
        """Cubane canonical SMILES should be a key in BICYCLO_RETAINED_NAMES.

         a phase: rekeyed from the stale C12C3C4C1C1C3C2C41 (InChIKey
        BOLISNSTKUABPW, a DIFFERENT (CH)8 cage, NOT cubane) to the true cubane
        canonical (InChIKey TXWRERCHRDBNLG)."""
        assert "C12C3C4C1C1C2C3C41" in BICYCLO_RETAINED_NAMES

    @pytest.mark.unit
    def test_cubane_value_correct(self):
        """BICYCLO_RETAINED_NAMES should map the true cubane SMILES to 'cubane'."""
        assert BICYCLO_RETAINED_NAMES.get("C12C3C4C1C1C2C3C41") == "cubane"


# ============================================================================
# a phase  — RED tests: retained-name PIN corrections.
# These FAIL now and are made green by Plan 167-02 (deny-based exclusion of the
# audited corrections). Membership/value/!=archaic asserts are the RED drivers;
# the structural-RT assert  is a correctness check (green now AND after,
# since OPSIN recognises both the archaic name and the systematic PIN).
# Audit + targets: docs/retained_name_conflicts.md § "a phase".
# ============================================================================


class TestErythreneRemoved:
    """C=CC=C must NOT emit 'erythrene' (archaic/incorrect) — PIN is buta-1,3-diene.

    'erythrene' is an OPSIN-imported synonym leaking into the output path; Plan
    167-02 adds it to _PIN_DENY. Audit: docs/retained_name_conflicts.md § a phase.
    """

    SMILES = "C=CC=C"
    ARCHAIC = "erythrene"

    @pytest.mark.unit
    def test_erythrene_not_in_all_retained_names_keys(self):
        assert self.SMILES not in ALL_RETAINED_NAMES

    @pytest.mark.unit
    def test_erythrene_not_in_all_retained_names_values(self):
        for val in ALL_RETAINED_NAMES.values():
            assert val != self.ARCHAIC, "Found 'erythrene' as a retained-name value"

    @pytest.mark.unit
    def test_erythrene_not_emitted(self):
        assert name_compound(self.SMILES) != self.ARCHAIC

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_emitted_name_roundtrips(self):
        """Whatever is emitted (target: buta-1,3-diene) must round-trip ."""
        if not _OPSIN_AVAILABLE:
            pytest.skip("OPSIN/Java not available")
        emitted = name_compound(self.SMILES)
        rt = _opsin_smiles(emitted)
        assert rt is not None, f"OPSIN could not parse emitted name {emitted!r}"
        assert _inchi(rt) == _inchi(self.SMILES)


class TestTrimethyleneGlycolRemoved:
    """OCCCO must NOT emit 'trimethylene glycol' (deprecated) — PIN is propane-1,3-diol."""

    SMILES = "OCCCO"
    ARCHAIC = "trimethylene glycol"

    @pytest.mark.unit
    def test_not_in_all_retained_names_keys(self):
        assert self.SMILES not in ALL_RETAINED_NAMES

    @pytest.mark.unit
    def test_not_in_all_retained_names_values(self):
        for val in ALL_RETAINED_NAMES.values():
            assert val != self.ARCHAIC, "Found 'trimethylene glycol' as a value"

    @pytest.mark.unit
    def test_not_emitted(self):
        assert name_compound(self.SMILES) != self.ARCHAIC

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_emitted_name_roundtrips(self):
        """Target: propane-1,3-diol (structural round-trip)."""
        if not _OPSIN_AVAILABLE:
            pytest.skip("OPSIN/Java not available")
        emitted = name_compound(self.SMILES)
        rt = _opsin_smiles(emitted)
        assert rt is not None, f"OPSIN could not parse emitted name {emitted!r}"
        assert _inchi(rt) == _inchi(self.SMILES)


class TestAspirinRemoved:
    """CC(=O)Oc1ccccc1C(=O)O must NOT emit 'aspirin' (brand) — PIN is 2-acetyloxybenzoic acid."""

    SMILES = "CC(=O)Oc1ccccc1C(=O)O"
    ARCHAIC = "aspirin"

    @pytest.mark.unit
    def test_not_in_all_retained_names_keys(self):
        assert self.SMILES not in ALL_RETAINED_NAMES

    @pytest.mark.unit
    def test_not_in_all_retained_names_values(self):
        for val in ALL_RETAINED_NAMES.values():
            assert val != self.ARCHAIC, "Found 'aspirin' as a value"

    @pytest.mark.unit
    def test_not_emitted(self):
        assert name_compound(self.SMILES) != self.ARCHAIC

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_emitted_name_roundtrips(self):
        """Target: 2-acetyloxybenzoic acid (structural round-trip)."""
        if not _OPSIN_AVAILABLE:
            pytest.skip("OPSIN/Java not available")
        emitted = name_compound(self.SMILES)
        rt = _opsin_smiles(emitted)
        assert rt is not None, f"OPSIN could not parse emitted name {emitted!r}"
        assert _inchi(rt) == _inchi(self.SMILES)
