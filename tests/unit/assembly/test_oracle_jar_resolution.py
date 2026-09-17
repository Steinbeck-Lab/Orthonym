"""R1 (a performance pass): every OpsinOracle must hold the SAME absolute jar path the
in-process JVM was started with; otherwise jvm_bridge.opsin_stdout refuses to serve and
rt_safe falls back to a fresh `java -jar` subprocess (0.7 s each), and from a non-root cwd
the relative helper finds no jar at all (splits fail closed, charged gates fail open)."""
import os, subprocess, sys
import pytest

from orthonym import jvm_bridge
from orthonym.assembly import group_splitting
from orthonym.validation.opsin_roundtrip import _find_opsin_jar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def fresh_default_oracle():
    saved = group_splitting._DEFAULT_ORACLE
    group_splitting._DEFAULT_ORACLE = None
    yield
    group_splitting._DEFAULT_ORACLE = saved


def test_default_oracle_jar_is_absolute_and_matches_resolver(fresh_default_oracle):
    jar = group_splitting._get_default_oracle()._jar
    assert jar is not None
    assert os.path.isabs(jar)
    assert jar == _find_opsin_jar()


@pytest.mark.skipif(not jvm_bridge.opsin_available(), reason="needs the in-process JVM")
def test_default_oracle_jar_equals_jvm_jar(fresh_default_oracle):
    assert jvm_bridge._ensure_jvm()
    assert group_splitting._get_default_oracle()._jar == jvm_bridge._OPSIN_JAR


def test_default_oracle_finds_jar_from_foreign_cwd(fresh_default_oracle, tmp_path):
    code = ("from orthonym.assembly.group_splitting import _get_default_oracle;"
            "print(_get_default_oracle()._jar)")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() not in ("", "None")
    assert os.path.isabs(out.stdout.strip())


@pytest.mark.skipif(not jvm_bridge.opsin_available(), reason="needs the in-process JVM")
def test_bridge_serves_relative_spelling_of_the_same_jar():
    assert jvm_bridge._ensure_jvm()
    rel = os.path.relpath(jvm_bridge._OPSIN_JAR, os.getcwd())
    text, served = jvm_bridge.opsin_stdout("acetic acid", allow_radicals=True, jar_path=rel)
    assert served is True
    assert text.strip() != ""
