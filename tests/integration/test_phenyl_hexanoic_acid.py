"""Phase 146 P-44.1.1 canonical chain-wins test (RESEARCH §7 Dimension 2).

SMILES: c1ccc(CCCCCC(=O)O)cc1 (6-phenylhexanoic acid)
  - Chain: hexanoic acid (6 carbons + 1 PCG = COOH)
  - Ring: benzene (6 carbons + 0 PCGs)

Per P-44.1.1: max PCG count wins → chain wins. V18 must produce a name
whose parent stem is `hexanoic acid`.

V17 may produce a ring-biased name (e.g., an acyl-benzene variant) —
that is the bug Phase 146 aims to fix. This test documents V17 baseline
without asserting the specific string, so calibration drift does not
break the test; only the V18 expectation is strict.

Both modes are exercised through subprocess isolation so module-import-
time env-var reads take effect cleanly, following the pattern adopted
in Plan 03's test_feature_flag.py (Rule-1 Auto-fix: avoids
importlib.reload class-identity leakage into downstream tests).

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1
"""

import os
import subprocess
import sys
import textwrap

import pytest


PHENYL_HEXANOIC_ACID_SMILES = "c1ccc(CCCCCC(=O)O)cc1"


def _run_in_mode(use_v18: str, sel_mode: str) -> str:
    """Invoke name_compound(PHENYL_HEXANOIC_ACID_SMILES) in a subprocess
    with ORTHONYM_USE_V18_WEIGHTS=<use_v18> and
    ORTHONYM_SELECTION_MODE=<sel_mode>. Returns the last line of stdout
    (the produced name), or '' on subprocess failure.
    """
    script = textwrap.dedent(
        f"""
        from orthonym import name_compound
        name = name_compound({PHENYL_HEXANOIC_ACID_SMILES!r})
        print(name)
        """
    ).strip()
    env = {
        "ORTHONYM_USE_V18_WEIGHTS": use_v18,
        "ORTHONYM_SELECTION_MODE": sel_mode,
        "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
    }
    # Preserve PYTHONPATH if set (test runners often need it for source tree).
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return ""
    out = result.stdout.strip()
    return out.splitlines()[-1] if out else ""


@pytest.mark.integration
def test_v18_picks_chain_parent():
    """V18 P-44.1.1: chain has 1 PCG, ring has 0 → chain (hexanoic acid) wins.

    The Tier-1 cascade step 1 (max PCG count) picks the chain over the
    ring. The produced name must contain 'hexanoic' (e.g.,
    '6-phenylhexanoic acid' in P-44.1.1-compliant IUPAC PIN style).
    """
    name = _run_in_mode(use_v18="true", sel_mode="score_based")
    assert name, (
        f"V18 produced empty name for {PHENYL_HEXANOIC_ACID_SMILES!r}"
    )
    assert "hexanoic" in name.lower(), (
        f"P-44.1.1 chain-wins FAILURE: V18 expected chain (hexanoic acid) "
        f"parent for {PHENYL_HEXANOIC_ACID_SMILES!r}; got {name!r}"
    )


@pytest.mark.integration
def test_v17_documents_baseline(capsys):
    """V17 baseline: produces SOMETHING (no assertion on specific string).

    Documents V17 behavior for comparison with V18 in
    test_v18_picks_chain_parent. V17 may produce a ring-biased name
    (e.g., 'hexanoylbenzene' or similar) — that is the v17 bug Phase
    146's two-tier selector fixes. This test exists to ensure the V17
    path doesn't regress to empty.
    """
    name = _run_in_mode(use_v18="false", sel_mode="first_applicable")
    assert name, (
        f"V17 produced empty name for {PHENYL_HEXANOIC_ACID_SMILES!r}"
    )
    # Print to stdout so CI logs capture the V17 baseline for diff tracking.
    print(f"V17 baseline name for {PHENYL_HEXANOIC_ACID_SMILES!r}: {name!r}")
