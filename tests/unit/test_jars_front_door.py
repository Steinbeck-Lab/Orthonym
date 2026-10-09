"""A missing pinned jar stops naming loudly at the front door (fresh processes).

Without the OPSIN / centres jars Orthonym would silently name in a weaker mode
(no OPSIN round-trip check, a different CIP labeller). It must refuse instead,
unless ORTHONYM_ALLOW_REDUCED=1 is set on purpose.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import orthonym.jars as jars

ROOT = Path(__file__).resolve().parents[2]
ORTHONYM_CLI = Path(sys.executable).with_name("orthonym")


def _no_jar_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ORTHONYM_") or k == "ORTHONYM_JVM_BUDGET"}
    env.update(ORTHONYM_JAR_DIR=str(tmp_path / "empty-jar-dir"), ORTHONYM_NO_DOWNLOAD="1",
               ORTHONYM_JVM_BUDGET="off")
    env.update(extra)
    return env


def _run(args, env, cwd):
    return subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True, timeout=300)


def test_name_compound_refuses_without_jars(tmp_path):
    r = _run([sys.executable, "-c", "from orthonym import name_compound; print(name_compound('CCO'))"],
             _no_jar_env(tmp_path), tmp_path)
    assert r.returncode != 0
    assert "JarUnavailable" in r.stderr and "fetch-jars" in r.stderr
    assert "ethanol" not in r.stdout


@pytest.mark.skipif(not ORTHONYM_CLI.exists(), reason="console script not installed in this venv")
def test_cli_without_arguments_prints_help_without_the_jars(tmp_path):
    """The help needs no jar: with none in the jar directory (and no download allowed)
    a bare `orthonym` still prints the usage at once, rather than refusing."""
    r = _run([str(ORTHONYM_CLI)], _no_jar_env(tmp_path), tmp_path)
    assert r.returncode == 1 and r.stdout.startswith("usage: orthonym"), (r.stdout, r.stderr)
    assert "fetch-jars" not in r.stderr and not (tmp_path / "empty-jar-dir").exists()


@pytest.mark.skipif(not ORTHONYM_CLI.exists(), reason="console script not installed in this venv")
def test_cli_exits_2_with_the_fix(tmp_path):
    r = _run([str(ORTHONYM_CLI), "CCO"], _no_jar_env(tmp_path), tmp_path)
    assert r.returncode == 2
    assert "orthonym --fetch-jars" in r.stderr and "ORTHONYM_ALLOW_REDUCED=1" in r.stderr
    assert r.stdout.strip() == ""


def test_reduced_mode_is_opt_in(tmp_path):
    r = _run([sys.executable, "-c", "from orthonym import name_compound; print(name_compound('CCO'))"],
             _no_jar_env(tmp_path, ORTHONYM_ALLOW_REDUCED="1"), tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ethanol"


def test_grammar_import_needs_no_jar_and_downloads_nothing(tmp_path):
    env = _no_jar_env(tmp_path)
    r = _run([sys.executable, "-c", "import orthonym.validation.opsin_grammar as g; print(len(g._TOKEN_REGEX) > 0)"],
             env, tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"
    assert not (tmp_path / "empty-jar-dir").exists()  # nothing was fetched


@pytest.mark.skipif(not ORTHONYM_CLI.exists(), reason="console script not installed in this venv")
def test_fetch_jars_with_overrides_reports_paths(tmp_path):
    try:
        opsin, centres = jars.find_jar("opsin"), jars.find_jar("centres")
    except jars.JarUnavailable:
        pytest.skip("pinned jars not available on this machine")
    env = _no_jar_env(tmp_path, ORTHONYM_OPSIN_JAR=opsin, ORTHONYM_CENTRES_JAR=centres)
    r = _run([str(ORTHONYM_CLI), "--fetch-jars"], env, tmp_path)
    assert r.returncode == 0, r.stderr
    assert opsin in r.stderr and centres in r.stderr
