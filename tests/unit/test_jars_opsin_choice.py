"""ORTHONYM_OPSIN selects the OPSIN every check uses: the release jar by default, a jar built from
OPSIN's main branch as an opt-in. A failed build is an error, never a fall back to the release jar."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orthonym import jars, opsin_build
from tests.support.jars import jar_or_skip

SHA = "e9f62eb38a6f2519056549cc16385a4257c1d55f"
RELEASE = "/fake/opsin-cli-2.9.0-jar-with-dependencies.jar"


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(jars, "_CACHE", {})
    monkeypatch.setattr(jars, "_resolve_release", lambda spec, download: RELEASE)
    # the build's step finds the release jar together with the checked jar directory (the main-branch build is made in it)
    monkeypatch.setattr(jars, "_resolve_release_in",
                        lambda spec, download: (RELEASE, Path(os.environ["ORTHONYM_JAR_DIR"])))
    for var in ("ORTHONYM_OPSIN", "ORTHONYM_OPSIN_JAR", "ORTHONYM_NO_DOWNLOAD", "ORTHONYM_ALLOW_REDUCED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(tmp_path / "jars"))


@pytest.fixture
def built(monkeypatch, tmp_path):
    """``ensure_jar`` replaced by one that hands back a jar and records its calls."""
    jar = tmp_path / f"opsin-main-{SHA[:12]}.jar"
    jar.write_bytes(b"jar")
    calls = []

    def ensure(ref, directory, base, download=True):
        calls.append((ref, directory, base, download))
        return opsin_build.BuiltJar(jar, SHA, ref)
    monkeypatch.setattr(opsin_build, "ensure_jar", ensure)
    return jar, calls


@pytest.mark.parametrize("value", [None, "", "2.9.0"])
def test_default_is_the_release_jar(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("ORTHONYM_OPSIN", value)
    monkeypatch.setattr(jars, "_resolve_built_opsin", lambda *a: pytest.fail("the opt-in ran by default"))
    assert jars.find_jar("opsin") == RELEASE
    assert jars.active_opsin() == {"label": "2.9.0", "commit": None, "path": RELEASE}


def test_main_choice_returns_the_built_jar_for_every_caller(monkeypatch, built):
    jar, calls = built
    monkeypatch.setenv("ORTHONYM_OPSIN", f"main@{SHA}")
    assert jars.find_jar("opsin") == str(jar)
    assert jars.find_jar("opsin", "2.9.0") == str(jar)
    assert jars.find_jar("opsin", "2.8.0") is None
    assert jars.active_opsin() == {"label": "main", "commit": SHA, "path": str(jar)}
    assert [c[0] for c in calls] == [SHA]  # resolved once; the release jar is the build's library donor
    assert calls[0][2] == Path(RELEASE) and calls[0][1] == Path(os.environ["ORTHONYM_JAR_DIR"])


def test_main_is_the_master_branch(monkeypatch, built):
    _, calls = built
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    jars.find_jar("opsin")
    assert calls[0][0] == "master"


def test_explicit_jar_path_still_wins(monkeypatch, tmp_path, built):
    own = tmp_path / "own.jar"
    own.write_bytes(b"jar")
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setenv("ORTHONYM_OPSIN_JAR", str(own))
    assert jars.find_jar("opsin") == str(own.resolve())
    assert jars.active_opsin() == {"label": "custom", "commit": None, "path": str(own.resolve())}
    assert built[1] == []


def test_bad_choice_is_a_jar_error(monkeypatch):
    monkeypatch.setenv("ORTHONYM_OPSIN", "3.0")
    with pytest.raises(jars.JarUnavailable, match="2.9.0, main or main@<commit>"):
        jars.find_jar("opsin")


def test_a_malformed_choice_is_an_error_even_in_reduced_mode(monkeypatch):
    monkeypatch.setenv("ORTHONYM_OPSIN", "main@@")
    monkeypatch.setenv("ORTHONYM_ALLOW_REDUCED", "1")
    with pytest.raises(jars.OpsinChoiceError, match="2.9.0, main or main@<commit>"):
        jars.find_jar("opsin")
    with pytest.raises(jars.JarUnavailable):
        jars.active_opsin()
    with pytest.raises(jars.JarUnavailable):
        jars.fetch_opsin_choice(verbose=False)


def test_a_failed_build_is_an_error_and_never_the_release_jar(monkeypatch):
    def broken(*a, **k):
        raise opsin_build.OpsinBuildError("javac failed")
    monkeypatch.setattr(opsin_build, "ensure_jar", broken)
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    for _ in range(2):  # the outcome is kept: a run never switches OPSIN half-way
        with pytest.raises(jars.JarUnavailable, match=r"ORTHONYM_OPSIN=main: javac failed"):
            jars.find_jar("opsin")


def test_a_platform_without_getuid_is_a_jar_error_that_says_so(monkeypatch):
    class OsWithoutGetuid:
        def __getattr__(self, name):
            if name == "getuid":
                raise AttributeError(name)
            return getattr(os, name)
    monkeypatch.setattr(opsin_build, "os", OsWithoutGetuid())
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    with pytest.raises(jars.JarUnavailable, match=r"ORTHONYM_OPSIN=main: the OPSIN main opt-in needs a POSIX system"):
        jars.find_jar("opsin")


def test_any_failure_of_the_build_is_a_jar_error(monkeypatch):
    def unreachable(*a, **k):
        raise PermissionError("jar directory")
    monkeypatch.setattr(opsin_build, "ensure_jar", unreachable)
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    with pytest.raises(jars.JarUnavailable, match="PermissionError"):
        jars.find_jar("opsin")


def test_reduced_mode_applies_to_a_failed_build_like_to_a_missing_jar(monkeypatch):
    monkeypatch.setattr(opsin_build, "ensure_jar", lambda *a, **k: (_ for _ in ()).throw(opsin_build.OpsinBuildError("x")))
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setenv("ORTHONYM_ALLOW_REDUCED", "1")
    assert jars.find_jar("opsin") is None


def test_no_download_reaches_the_build(monkeypatch, built):
    _, calls = built
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    jars.find_jar("opsin")
    assert calls[0][3] is False
    jars._reset_cache()
    jars.find_jar("opsin", download=False)
    assert calls[1][3] is False


def test_fetch_builds_ahead_and_ignores_no_download(monkeypatch, built):
    jar, calls = built
    assert jars.fetch_opsin_choice(verbose=False) is None and calls == []
    monkeypatch.setenv("ORTHONYM_OPSIN", f"main@{SHA}")
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    assert jars.fetch_opsin_choice(verbose=False) == str(jar)
    assert calls[0][3] is True
    monkeypatch.setattr(opsin_build, "ensure_jar", lambda *a, **k: pytest.fail("built twice"))
    assert jars.find_jar("opsin") == str(jar)  # the explicit fetch is the resolution of this process


def test_fetch_leaves_a_given_jar_alone(monkeypatch, tmp_path, built):
    own = tmp_path / "own.jar"
    own.write_bytes(b"jar")
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setenv("ORTHONYM_OPSIN_JAR", str(own))
    assert jars.fetch_opsin_choice(verbose=False) is None and built[1] == []


def test_a_build_that_fails_to_fetch_is_an_error(monkeypatch):
    monkeypatch.setattr(opsin_build, "ensure_jar", lambda *a, **k: (_ for _ in ()).throw(opsin_build.OpsinBuildError("no JDK")))
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    with pytest.raises(jars.JarUnavailable, match="no JDK"):
        jars.fetch_opsin_choice(verbose=False)


# --- capabilities of the OPSIN in use -------------------------------------------------------------

def test_release_jar_has_no_phane_grammar(monkeypatch):
    from orthonym import opsin_capabilities as caps
    caps._reset_cache()
    monkeypatch.setattr(caps, "_probe", lambda path: pytest.fail("the release jar is not probed"))
    assert caps.opsin_capabilities() == {"phane": False, "halogen_uide": False}
    assert caps.phane_enabled() is False


def test_phane_enabled_follows_the_probe(monkeypatch):
    from orthonym import opsin_capabilities as caps
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": True, "halogen_uide": False})
    assert caps.phane_enabled() is True


def test_an_unserved_probe_is_not_cached(monkeypatch):
    # a transient "OPSIN cannot answer" must not switch the producers off for the rest of the process
    from orthonym import opsin_capabilities as caps
    from orthonym.jvm_bridge import OpsinUnavailable
    calls = []

    def probe(path):
        calls.append(path)
        if len(calls) == 1:
            raise OpsinUnavailable("in-process OPSIN busy")
        return {"phane": True, "halogen_uide": True}

    monkeypatch.setattr(caps, "_probe", probe)
    monkeypatch.setattr(jars, "active_opsin", lambda: {"label": "custom", "commit": None, "path": "/x.jar"})
    caps._reset_cache()
    assert caps.opsin_capabilities() == {"phane": False, "halogen_uide": False}
    assert caps.opsin_capabilities() == {"phane": True, "halogen_uide": True}
    assert caps.opsin_capabilities() == {"phane": True, "halogen_uide": True}
    assert calls == ["/x.jar", "/x.jar"]


def test_a_built_jar_keeps_its_answer_in_its_sidecar(monkeypatch, tmp_path):
    from orthonym import opsin_capabilities as caps
    jar = tmp_path / "opsin-main-abc.jar"
    jar.write_bytes(b"jar")
    side = jar.with_suffix(".json")
    side.write_text(json.dumps({"commit": SHA}))
    side.chmod(0o644)  # as ensure_jar leaves it
    probes = []
    monkeypatch.setattr(caps, "_probe", lambda path: probes.append(path) or {"phane": True, "halogen_uide": False})
    monkeypatch.setattr(jars, "active_opsin", lambda: {"label": "main", "commit": SHA, "path": str(jar)})
    caps._reset_cache()
    assert caps.opsin_capabilities() == {"phane": True, "halogen_uide": False}
    assert json.loads(side.read_text()) == {"commit": SHA, "capabilities": {"phane": True, "halogen_uide": False}}
    caps._reset_cache()  # a new process: the sidecar answers, no probe
    assert caps.opsin_capabilities() == {"phane": True, "halogen_uide": False}
    assert probes == [str(jar)]


def test_a_malformed_sidecar_answer_is_probed_again(monkeypatch, tmp_path):
    from orthonym import opsin_capabilities as caps
    jar = tmp_path / "opsin-main-abc.jar"
    jar.write_bytes(b"jar")
    jar.with_suffix(".json").write_text(json.dumps({"commit": SHA, "capabilities": {"phane": "yes"}}))
    jar.with_suffix(".json").chmod(0o644)
    monkeypatch.setattr(caps, "_probe", lambda path: {"phane": False, "halogen_uide": True})
    monkeypatch.setattr(jars, "active_opsin", lambda: {"label": "main", "commit": SHA, "path": str(jar)})
    caps._reset_cache()
    assert caps.opsin_capabilities() == {"phane": False, "halogen_uide": True}


def test_a_given_jar_is_probed_and_leaves_no_file(monkeypatch, tmp_path):
    from orthonym import opsin_capabilities as caps
    jar = tmp_path / "own.jar"
    jar.write_bytes(b"jar")
    monkeypatch.setattr(caps, "_probe", lambda path: {"phane": True, "halogen_uide": True})
    monkeypatch.setattr(jars, "active_opsin", lambda: {"label": "custom", "commit": None, "path": str(jar)})
    caps._reset_cache()
    assert caps.opsin_capabilities() == {"phane": True, "halogen_uide": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["own.jar"]


def test_no_jar_means_no_capability(monkeypatch):
    from orthonym import opsin_capabilities as caps

    def missing():
        raise jars.JarUnavailable("no jar")
    monkeypatch.setattr(jars, "active_opsin", missing)
    caps._reset_cache()
    assert caps.opsin_capabilities() == {"phane": False, "halogen_uide": False}


def test_the_probe_reads_the_release_jar_as_unable(tmp_path):
    """The probe end to end, in a fresh process (the in-process JVM takes its jar once): the
    release jar given as a custom jar reads neither probe name."""
    jar = jar_or_skip("opsin")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ORTHONYM_") or k == "ORTHONYM_JVM_BUDGET"}
    env.update(ORTHONYM_OPSIN_JAR=jar, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="test_jars_opsin_choice"):
        r = subprocess.run([sys.executable, "-c",
                            "from orthonym.opsin_capabilities import opsin_capabilities as c; print(c())"],
                           env=env, cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.stdout.strip() == "{'phane': False, 'halogen_uide': False}", (r.stdout, r.stderr[-500:])
