#!/usr/bin/env python3
"""CIP Validation Suite integration test.

Tests RDKit rdCIPLabeler.AssignCIPLabels() against 300 molecules from
Hanson et al. 2018 (CIP Validation Suite). Results document known
RDKit CIP limitations for Phase 140 stereo pipeline planning.

Source: https://github.com/CIPValidationSuite/ValidationSuite
Paper: Hanson et al., "Algorithmic Analysis of Cahn-Ingold-Prelog Rules
       of Stereochemistry" (2018), J. Chem. Inf. Model. 58(9), 1755-1765.
"""

import datetime
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from rdkit import Chem
from rdkit import __version__ as RDKIT_VERSION
from rdkit.Chem import rdCIPLabeler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CIP_DATA = Path(__file__).parent.parent / "data" / "cip_validation" / "compounds.smi"
RESULTS_FILE = CIP_DATA.parent / "rdkit_cip_results.txt"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_expected_labels(label_str: str) -> Dict[int, str]:
    """Parse CIP Validation Suite label strings into {1-based-atom-idx: descriptor}.

    Expected format: space-separated tokens like "2R 5S 13S", "3Z 4E 6E 8Z",
    or "19P 26P" for helicene descriptors.

    Descriptors: R, S, r, s (pseudoasymmetric), E, Z, M, P, m, p
    Lowercase r/s and m/p indicate pseudoasymmetric centers.

    Returns empty dict for empty or whitespace-only input.
    """
    if not label_str or not label_str.strip():
        return {}

    result = {}
    for token in label_str.strip().split():
        match = re.match(r"(\d+)([RSrsEZMPmpz])", token)
        if match:
            idx = int(match.group(1))
            descriptor = match.group(2)
            result[idx] = descriptor
    return result


def get_rdkit_cip_labels(mol) -> Dict[int, str]:
    """Extract CIP labels assigned by RDKit rdCIPLabeler.

    Returns dict of {1-based-atom-idx: descriptor} to match CIP Validation
    Suite's 1-based numbering convention. RDKit uses 0-based internally,
    so we add 1 to convert.

    Includes ALL descriptor types: R, S, r, s, E, Z, M, P.
    """
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        return {}

    labels = {}

    # Atom stereochemistry: R, S, r, s
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            code = atom.GetProp("_CIPCode")
            # RDKit uses R, S, r, s (lowercase for pseudoasymmetric)
            if code in ("R", "S", "r", "s"):
                # Convert 0-based RDKit index to 1-based CIP Suite numbering
                labels[atom.GetIdx() + 1] = code

    # Bond stereochemistry: E, Z
    for bond in mol.GetBonds():
        if bond.HasProp("_CIPCode"):
            code = bond.GetProp("_CIPCode")
            if code in ("E", "Z"):
                # Use the lower-numbered atom (1-based) as the key
                begin_idx = bond.GetBeginAtomIdx() + 1
                end_idx = bond.GetEndAtomIdx() + 1
                key = min(begin_idx, end_idx)
                labels[key] = code

    return labels


def load_cip_data() -> List[Dict]:
    """Load CIP Validation Suite compounds from the .smi file.

    Format: tab-separated columns:
      0: SMILES
      1: ID (VS001-VS300)
      2: Expected labels (e.g., "2R 5S 13S") -- may be empty
      3: IUPAC reference (e.g., "P-93.5.3.3") -- may be empty
      4: Stereo type (TH, CT, AT, HE, TH3, CT4, TH5) -- may be empty
      5: CIP rules required (e.g., "4c,6") -- may be empty

    Returns list of dicts with keys: smiles, id, expected_labels, iupac_ref,
    stereo_type, rules.
    """
    entries = []
    with open(CIP_DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # Skip comment lines and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            entry = {
                "smiles": parts[0] if len(parts) > 0 else "",
                "id": parts[1] if len(parts) > 1 else "",
                "expected_labels": parts[2] if len(parts) > 2 else "",
                "iupac_ref": parts[3] if len(parts) > 3 else "",
                "stereo_type": parts[4] if len(parts) > 4 else "",
                "rules": parts[5] if len(parts) > 5 else "",
            }
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_labels(
    expected: Dict[int, str], actual: Dict[int, str]
) -> Tuple[bool, List[str]]:
    """Compare expected CIP labels against RDKit's actual assignments.

    Returns (all_match, list_of_mismatch_descriptions).

    Comparison rules:
    - For each expected label, check if actual has the same descriptor
      at the same 1-based atom index.
    - Lowercase 'z' in expected (used in CIP Suite for some CT descriptors)
      matches either 'E' or 'Z' from RDKit -- these are "undetermined" markers
      in the suite and should not count as failures.
    - M/P (axial chirality) and m/p (pseudoasymmetric axial) are compared
      but RDKit may not assign these -- record as mismatch if missing.
    """
    if not expected:
        # No expected labels -- pass if molecule has no expected stereocenters
        return True, []

    mismatches = []
    for idx, exp_desc in expected.items():
        # Skip 'z' (undetermined/generic cis-trans in CIP Suite)
        if exp_desc == "z":
            continue

        actual_desc = actual.get(idx)
        if actual_desc is None:
            mismatches.append(f"{idx}{exp_desc}: expected {exp_desc}, got <none>")
        elif actual_desc != exp_desc:
            mismatches.append(
                f"{idx}{exp_desc}: expected {exp_desc}, got {actual_desc}"
            )

    return len(mismatches) == 0, mismatches


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------


def write_results_file(
    total: int,
    pass_count: int,
    fail_count: int,
    invalid_count: int,
    no_expected_count: int,
    failures_by_type: Dict[str, int],
    failure_details: List[str],
):
    """Write CIP validation results to a text file for Phase 140 reference."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    lines = [
        f"CIP Validation Suite Results -- {timestamp}",
        f"RDKit rdCIPLabeler.AssignCIPLabels() (RDKit {RDKIT_VERSION})",
        "",
        f"  Total compounds:       {total}",
        f"  With expected labels:  {total - no_expected_count}",
        f"  No expected labels:    {no_expected_count}",
        "",
        f"  Pass:                  {pass_count}",
        f"  Fail:                  {fail_count}",
        f"  Invalid SMILES:        {invalid_count}",
        "",
        "Failures by stereo type:",
    ]

    type_order = ["TH", "CT", "AT", "HE", "CT,TH", "Other"]
    for stype in type_order:
        if stype in failures_by_type:
            label_map = {
                "TH": "TH (tetrahedral)",
                "CT": "CT (cis-trans)",
                "AT": "AT (atropisomeric)",
                "HE": "HE (helicene)",
                "CT,TH": "CT,TH (mixed)",
                "Other": "Other",
            }
            label = label_map.get(stype, stype)
            lines.append(f"  {label:30s} {failures_by_type[stype]}")

    # Add any types not in the predefined order
    for stype, count in sorted(failures_by_type.items()):
        if stype not in type_order:
            lines.append(f"  {stype:30s} {count}")

    lines.append("")
    lines.append("Failure details:")
    for detail in failure_details:
        lines.append(f"  {detail}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCIPValidationSuite:
    """Integration tests for the CIP Validation Suite (Hanson et al. 2018)."""

    @pytest.fixture(scope="class")
    def cip_data(self):
        if not CIP_DATA.exists():
            pytest.skip(
                "CIP Validation Suite not downloaded. Run: "
                "curl -L -o tests/data/cip_validation/compounds.smi "
                "'https://raw.githubusercontent.com/CIPValidationSuite/"
                "ValidationSuite/master/compounds.smi'"
            )
        return load_cip_data()

    def test_suite_loaded(self, cip_data):
        """Verify the CIP Validation Suite loaded with approximately 300 compounds."""
        count = len(cip_data)
        print(f"\nCIP Validation Suite: loaded {count} compounds")
        assert count >= 280, (
            f"Expected >= 280 compounds in CIP Validation Suite, got {count}"
        )

    def test_cip_assignments(self, cip_data):
        """Test RDKit CIP assignments against expected labels for all 300 molecules.

        This test documents RDKit CIP accuracy rather than failing on mismatches.
        CIP failures are known RDKit limitations, not Orthonym bugs.
        Results are written to rdkit_cip_results.txt for Phase 140 reference.
        """
        pass_count = 0
        fail_count = 0
        invalid_count = 0
        no_expected_count = 0
        failures_by_type: Dict[str, int] = defaultdict(int)
        failure_details: List[str] = []

        for entry in cip_data:
            smiles = entry["smiles"]
            entry_id = entry["id"]
            stereo_type = entry["stereo_type"] or "Unknown"

            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                failure_details.append(f"{entry_id}: invalid SMILES: {smiles}")
                continue

            # Parse expected labels
            expected = parse_expected_labels(entry["expected_labels"])

            if not expected:
                # No expected labels for this compound
                no_expected_count += 1
                continue

            # Get RDKit CIP labels
            actual = get_rdkit_cip_labels(mol)

            # Compare
            all_match, mismatches = compare_labels(expected, actual)

            if all_match:
                pass_count += 1
            else:
                fail_count += 1
                # Categorize failure by stereo type
                # stereo_type may be comma-separated (e.g., "CT,TH")
                failures_by_type[stereo_type] += 1

                rules_str = f", rules: {entry['rules']}" if entry["rules"] else ""
                for mm in mismatches:
                    failure_details.append(
                        f"{entry_id}: {mm} ({stereo_type}{rules_str})"
                    )

        # Print summary
        total = len(cip_data)
        tested = pass_count + fail_count
        print(f"\nCIP Validation Suite Results:")
        print(f"  Total:              {total}")
        print(f"  With expected:      {total - no_expected_count}")
        print(f"  No expected labels: {no_expected_count}")
        print(f"  Pass:               {pass_count}")
        print(f"  Fail:               {fail_count}")
        print(f"  Invalid SMILES:     {invalid_count}")
        print()
        print("Failures by stereo type:")
        for stype, count in sorted(failures_by_type.items()):
            print(f"  {stype:25s} {count}")
        print()
        print("Failure details (first 50):")
        for detail in failure_details[:50]:
            print(f"  {detail}")
        if len(failure_details) > 50:
            print(f"  ... and {len(failure_details) - 50} more")

        # Write results file for Phase 140 reference (CIP-03)
        write_results_file(
            total=total,
            pass_count=pass_count,
            fail_count=fail_count,
            invalid_count=invalid_count,
            no_expected_count=no_expected_count,
            failures_by_type=dict(failures_by_type),
            failure_details=failure_details,
        )
        print(f"\nResults written to: {RESULTS_FILE}")

        # Soft assertion: we expect most compounds to pass
        # Do NOT fail on CIP mismatches -- they are RDKit limitations
        if tested > 0:
            pass_pct = pass_count / tested * 100
            print(f"\nPass rate: {pass_count}/{tested} ({pass_pct:.1f}%)")
            assert pass_count > 150, (
                f"Expected at least 150 CIP passes, got {pass_count}/{tested}. "
                f"This may indicate a fundamental issue with the test harness."
            )

    def test_stereo_type_coverage(self, cip_data):
        """Verify the CIP Validation Suite covers multiple stereo types."""
        type_counts: Dict[str, int] = defaultdict(int)
        for entry in cip_data:
            stype = entry.get("stereo_type", "")
            if stype:
                # Handle comma-separated types (e.g., "CT,TH")
                for t in stype.split(","):
                    type_counts[t.strip()] += 1

        print("\nStereo type coverage:")
        for stype, count in sorted(type_counts.items()):
            print(f"  {stype}: {count}")

        unique_types = set(type_counts.keys())
        assert len(unique_types) >= 3, (
            f"Expected at least 3 different stereo types, got {len(unique_types)}: "
            f"{sorted(unique_types)}"
        )

        # We specifically expect TH and CT at minimum
        assert "TH" in unique_types, "Missing TH (tetrahedral) stereo type"
        assert "CT" in unique_types, "Missing CT (cis-trans) stereo type"
