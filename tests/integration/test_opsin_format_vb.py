"""
Von Baeyer format OPSIN compatibility verification .

Per Research Finding 1 (supersedes internal notes,,):
All 4 VB secondary bridge locant formats (parenthesized, superscript,
bracket, bare inline) parse successfully in OPSIN 2.8.0 and 2.9.0.
The current Orthonym parenthesized format is already OPSIN-compatible.

This test confirms that empirically -- no changes to
polycyclic.py:_build_descriptor are needed.

The 10 DEEPER VB compounds that fail even without stereo have
computational bugs (wrong bridge lengths, impossible valencies) and
are deferred to a phase/142.
"""

import os
import shutil
import subprocess

import pytest


def _java_available() -> bool:
    """Check if Java runtime is available."""
    if not shutil.which("java"):
        return False
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_jar_path(version: str = "2.9.0") -> str:
    """Get path to OPSIN JAR."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    return os.path.join(
        project_root,
        f"opsin-cli-{version}-jar-with-dependencies.jar",
    )


def _opsin_available(version: str = "2.9.0") -> bool:
    """Check if OPSIN JAR and Java are both available."""
    return _java_available() and os.path.isfile(_opsin_jar_path(version))


def _opsin_parse(name: str, version: str = "2.9.0") -> str:
    """Parse a name with OPSIN CLI and return SMILES or empty string."""
    jar_path = _opsin_jar_path(version)
    if not os.path.isfile(jar_path):
        return ""
    try:
        result = subprocess.run(
            ["java", "-jar", jar_path, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = result.stdout.strip().split("\n")
        out = lines[-1].strip() if lines else ""
        if "could not be interpreted" in out.lower():
            return ""
        if "unsure of the meaning" in out.lower():
            return ""
        return out
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


HAS_OPSIN = _opsin_available()
skip_no_opsin = pytest.mark.skipif(
    not HAS_OPSIN, reason="OPSIN 2.9.0 JAR or Java not available"
)


@pytest.mark.integration
class TestVBFormatOpsinCompatibility:
    """Verify Von Baeyer parenthesized format parses with OPSIN.

    Research Finding 1 confirmed all VB secondary bridge formats work.
    These tests provide ongoing regression protection for.
    """

    @skip_no_opsin
    def test_bicyclo_heptane(self):
        """bicyclo[2.2.1]heptane should parse (basic VB)."""
        smiles = _opsin_parse("bicyclo[2.2.1]heptane")
        assert smiles, "OPSIN failed to parse bicyclo[2.2.1]heptane"

    @skip_no_opsin
    def test_tricyclo_decane_parenthesized(self):
        """tricyclo[3.3.1.1(3,7)]decane should parse (parenthesized secondary bridge).

        This is the critical test: Orthonym uses parenthesized format
        for secondary bridge locants. Research Finding 1 confirmed this
        format is OPSIN-compatible.
        """
        smiles = _opsin_parse("tricyclo[3.3.1.1(3,7)]decane")
        assert smiles, (
            "OPSIN failed to parse tricyclo[3.3.1.1(3,7)]decane "
            "(parenthesized secondary bridge format)"
        )

    @skip_no_opsin
    def test_adamantane_retained_name(self):
        """adamantane (retained name for tricyclo[3.3.1.1(3,7)]decane) should parse."""
        smiles = _opsin_parse("adamantane")
        assert smiles, "OPSIN failed to parse adamantane"

    @skip_no_opsin
    def test_bicyclo_octane(self):
        """bicyclo[2.2.2]octane should parse."""
        smiles = _opsin_parse("bicyclo[2.2.2]octane")
        assert smiles, "OPSIN failed to parse bicyclo[2.2.2]octane"

    @skip_no_opsin
    def test_bicyclo_hexane(self):
        """bicyclo[2.1.1]hexane should parse."""
        smiles = _opsin_parse("bicyclo[2.1.1]hexane")
        assert smiles, "OPSIN failed to parse bicyclo[2.1.1]hexane"

    @skip_no_opsin
    def test_norbornane(self):
        """norbornane (retained name for bicyclo[2.2.1]heptane) should parse."""
        smiles = _opsin_parse("norbornane")
        assert smiles, "OPSIN failed to parse norbornane"

    @skip_no_opsin
    def test_vb_v28_compatibility(self):
        """Parenthesized VB format also works with OPSIN 2.8.0."""
        if not _opsin_available("2.8.0"):
            pytest.skip("OPSIN 2.8.0 JAR not available")
        smiles = _opsin_parse("tricyclo[3.3.1.1(3,7)]decane", version="2.8.0")
        assert smiles, (
            "OPSIN 2.8.0 failed to parse tricyclo[3.3.1.1(3,7)]decane"
        )


# 10 DEEPER VB compounds with computational bugs, deferred to a phase/142.
# These are NOT expected to pass -- they have wrong bridge lengths,
# impossible valencies, or garbled naming. Documented here for traceability.
VB_COMPUTATION_BUGS_DEFERRED = [
    # (description, why it fails)
    ("Compounds with wrong VB bridge lengths", "Phase 139: bridge calculation fix"),
    ("Compounds with impossible valencies in VB", "Phase 139: valency validation"),
    ("Compounds with garbled VB naming", "Phase 142: handler rewrite"),
]
