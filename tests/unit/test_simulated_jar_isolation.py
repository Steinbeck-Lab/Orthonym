"""A missing OPSIN jar, simulated by a test or real at the start of a process, must not
leave the rest of the process without OPSIN.

The validity gate's oracle (`namer._VALIDITY_ORACLE`) is built at its first use for
the jar `_find_opsin_jar` names at that moment and is not built again. A naming
without the jar that reaches it (an N-acyl ring amino acid, through
`substituent_naming.fragment_acid_name_verified`) builds it with jar=None. Once the
jar is back the gate runs again, that oracle answers 'unavailable' to every parse, and
the gate withholds every name as 'unknown organic compound' (TRIAGE 'Unit suite -- the
unknown-organic-compound order leak').

* In the test suite a jar-absent test did this and monkeypatch put the resolver back:
  every later name on that pytest worker was withheld. `tests/conftest.py` now puts
  such an oracle back after the test (first test below).
* Outside the tests, `orthonym.jars.fetch_all` in a process that started in reduced
  mode did the same (second test): the validity gate's oracle
  (`namer._validity_oracle`), the split gate's oracle
  (`group_splitting._get_default_oracle`) and the centres jar lookup
  (`centres_bridge._find_centres_jar`) now follow the jar resolution, so an oracle
  built without a jar is built again once the jar is found.

Both run in a subprocess, because each needs a process in which nothing has built the
oracle yet (in the running session it usually has been built, and under xdist two
tests of one file can land on different workers).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import orthonym
from tests.support.jars import jar_or_skip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The subprocesses import the same orthonym as this test process.
_SRC = str(Path(orthonym.__file__).resolve().parents[1])

_PROBE = '''
import orthonym.namer as namer
import orthonym.validation.opsin_roundtrip as rt


def test_first_use_with_the_jar_simulated_absent(monkeypatch):
    monkeypatch.setattr(rt, "_find_opsin_jar", lambda *a, **k: None)
    assert namer._VALIDITY_ORACLE is None, "the probe must be the first use of the oracle"
    assert namer._validity_gate_status("ethanol") == "unavailable"


def test_the_next_test_reaches_opsin_again():
    assert namer._validity_gate_status("ethanol") == "parsed"
'''


def _env(**extra):
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_SRC, os.environ.get("PYTHONPATH")) if p)
    return env


def test_an_oracle_built_for_a_simulated_missing_jar_is_put_back():
    jar_or_skip("opsin")
    # Under tests/unit so that tests/conftest.py applies; the name does not match
    # `test_*.py`, so no other collection picks the file up.
    probe = PROJECT_ROOT / "tests" / "unit" / f"_simulated_jar_probe_{os.getpid()}.py"
    probe.write_text(_PROBE)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-q", "--no-header",
             "-p", "no:cacheprovider", "-p", "no:xdist", "-o", "addopts="],
            cwd=PROJECT_ROOT, env=_env(), capture_output=True, text=True, timeout=300,
        )
        combined = r.stdout + r.stderr
        assert "2 passed" in combined, (
            "after a test that simulated a missing OPSIN jar, the next test in the "
            f"process could not reach OPSIN:\n{combined[-3000:]}")
    finally:
        probe.unlink(missing_ok=True)


_AFTER_FETCH = '''
import os, sys
from orthonym import Orthonym
import orthonym.jars as jars
import orthonym.namer as namer
from orthonym.perception.centres_bridge import _find_centres_jar

# Reduced mode (no jar in reach). An N-acyl ring amino acid reaches the validity
# gate's oracle, which is then built without a jar.
first = Orthonym(style="pin").name("CC(=O)N1CCCC1C(=O)O")
oracle = namer._VALIDITY_ORACLE
built_without_jar = oracle is not None and oracle._jar is None
# The jars become reachable and are fetched explicitly in the same process.
os.environ["ORTHONYM_OPSIN_JAR"], os.environ["ORTHONYM_CENTRES_JAR"] = sys.argv[1], sys.argv[2]
jars.fetch_all(verbose=False)
print(repr((first, built_without_jar, Orthonym(style="pin").name("CCO"), _find_centres_jar())))
'''


def test_jars_fetched_into_a_reduced_mode_process_are_used(tmp_path):
    opsin, centres = jar_or_skip("opsin"), jar_or_skip("centres")
    env = _env(ORTHONYM_ALLOW_REDUCED="1", ORTHONYM_NO_DOWNLOAD="1",
               ORTHONYM_JAR_DIR=str(tmp_path))  # an empty jar directory
    env.pop("ORTHONYM_OPSIN_JAR", None)
    env.pop("ORTHONYM_CENTRES_JAR", None)
    r = subprocess.run([sys.executable, "-c", _AFTER_FETCH, opsin, centres], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not r.stdout.strip():
        pytest.fail(f"the probe process did not run:\n{r.stderr[-3000:]}", pytrace=False)
    first, built_without_jar, after, centres_after = ast.literal_eval(r.stdout.strip().splitlines()[-1])
    # Precondition, not the defect: the reduced-mode naming built the oracle without a
    # jar. (Its name is not the precondition: the default tier declines the
    # 'N-acetylproline' of the reduced mode, a name the code records as not the PIN,
    # with NO_VERIFIED_PIN; the oracle is built while the name is made.)
    if not built_without_jar:
        pytest.fail(f"the reduced-mode naming no longer builds the oracle without a jar "
                    f"(name {first!r}); the probe needs another first molecule", pytrace=False)
    assert after == "ethanol", f"after fetch_all, CCO was named {after!r}"
    assert centres_after and os.path.realpath(centres_after) == os.path.realpath(centres), (
        f"after fetch_all, the centres jar lookup gave {centres_after!r}")
