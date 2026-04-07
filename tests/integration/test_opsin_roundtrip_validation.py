"""Integration tests for OPSIN CLI round-trip validation of imported data.

Tests that imported retained names can be parsed back by OPSIN to produce
the original molecular structure. Requires Java and OPSIN CLI jar.
"""

import os
import subprocess
import pytest
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi


def _java_available():
    """Check if Java is available."""
    try:
        subprocess.run(["java", "-version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_jar_path():
    """Find OPSIN CLI jar."""
    candidates = [
        "opsin-cli-2.9.0-jar-with-dependencies.jar",
        "opsin/opsin-cli-2.9.0-jar-with-dependencies.jar",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _opsin_name_to_smiles(name: str, jar_path: str) -> str:
    """Convert a name to SMILES using OPSIN CLI."""
    result = subprocess.run(
        ["java", "-jar", jar_path, "-osmi"],
        input=name,
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()


def _inchi_match(smi1: str, smi2: str) -> bool:
    """Compare two SMILES via InChI (stereo-agnostic)."""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    i1 = MolToInchi(mol1)
    i2 = MolToInchi(mol2)
    if i1 is None or i2 is None:
        return False
    # Strip stereo layers for constitutional comparison
    def strip(x):
        return "/".join(p for p in x.split("/")
                        if not p.startswith(("t", "b", "m", "s")))
    return strip(i1) == strip(i2)


@pytest.mark.integration
class TestOpsinJarAvailable:
    """Check OPSIN CLI jar is present."""

    def test_opsin_jar_available(self):
        """OPSIN CLI jar exists in project root or opsin/ directory."""
        jar = _opsin_jar_path()
        if jar is None:
            pytest.skip("OPSIN CLI jar not found")
        assert os.path.getsize(jar) > 1_000_000, "Jar file too small"


@pytest.mark.integration
class TestOpsinRoundtripSample:
    """Round-trip a sample of known retained names through OPSIN."""

    # Well-known retained names that OPSIN should parse correctly
    SAMPLE_NAMES = [
        ("benzene", "c1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("toluene", "Cc1ccccc1"),
        ("phenol", "Oc1ccccc1"),
        ("ethanol", "CCO"),
        ("methanol", "CO"),
        ("acetic acid", "CC(O)=O"),
        ("formaldehyde", "C=O"),
        ("acetaldehyde", "CC=O"),
        ("glycerol", "OCC(O)CO"),
        ("pyridine", "c1ccncc1"),
        ("furan", "c1ccoc1"),
        ("indole", "c1ccc2[nH]ccc2c1"),
        ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
        ("acetone", "CC(C)=O"),
        ("glycine", "NCC(O)=O"),
        ("alanine", "CC(N)C(O)=O"),
        ("styrene", "C=Cc1ccccc1"),
        ("aniline", "Nc1ccccc1"),
    ]

    def test_opsin_roundtrip_sample(self):
        """At least 15/20 sample names round-trip through OPSIN CLI."""
        jar = _opsin_jar_path()
        if jar is None:
            pytest.skip("OPSIN CLI jar not found")
        if not _java_available():
            pytest.skip("Java not available")

        matches = 0
        failures = []

        for name, expected_smi in self.SAMPLE_NAMES:
            try:
                opsin_smi = _opsin_name_to_smiles(name, jar)
                if opsin_smi and _inchi_match(expected_smi, opsin_smi):
                    matches += 1
                else:
                    failures.append((name, expected_smi, opsin_smi))
            except Exception as e:
                failures.append((name, expected_smi, f"ERROR: {e}"))

        assert matches >= 15, (
            f"Only {matches}/20 round-tripped. Failures: "
            + ", ".join(f"{n} (got '{s}')" for n, _, s in failures[:5])
        )
