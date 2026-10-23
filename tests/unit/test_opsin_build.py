"""orthonym.opsin_build: fetch, compile, check and cache an OPSIN jar for ORTHONYM_OPSIN=main[@ref].

Nothing here needs the network. The build test needs a JDK and is skipped without one; the test
that fetches the real sources from GitHub runs only with ORTHONYM_NETWORK_TESTS=1.
"""
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest

from orthonym import opsin_build as ob
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_flags import java_cmd

SHA = "e9f62eb38a6f2519056549cc16385a4257c1d55f"
GOOD_OUTPUT = "".join(smiles + "\n" for _, smiles in ob._SELFTEST)


def plant_build(cache, commit=SHA, payload=b"jar", **sidecar):
    """A built jar and its sidecar in ``cache`` as ``ensure_jar`` leaves them (a private directory)."""
    jar = cache / f"opsin-main-{commit[:12]}.jar"
    jar.write_bytes(payload)
    jar.chmod(0o644)
    meta = jar.with_suffix(".json")
    meta.write_text(json.dumps({"commit": commit, "jar_sha256": hashlib.sha256(payload).hexdigest(), **sidecar}))
    meta.chmod(0o644)
    return jar


def files(directory):
    """The names in a cache directory, without the lock file that a build leaves there."""
    return sorted(p.name for p in directory.iterdir() if p.name != ".lock")


def private(tmp_path):
    """``(jar_dir, cache)``: a jar directory and its private cache subdirectory, made as Orthonym makes it."""
    return tmp_path / "jars", ob.private_dir(tmp_path / "jars", create=True)


@pytest.mark.parametrize("value,ref", [(None, None), ("", None), ("2.9.0", None), ("main", "master"),
                                       ("main@e9f62eb38a6f", "e9f62eb38a6f"), (" main ", "master"),
                                       ("main@release/3.0", "release/3.0")])
def test_parse_choice(value, ref):
    assert ob.parse_choice(value) == ref


@pytest.mark.parametrize("value", ["3.0", "master", "main@", "main@a b", "main@../x", "main@x?y=1", "main@-rf"])
def test_parse_choice_rejects_other_values(value):
    with pytest.raises(ob.OpsinBuildError, match="2.9.0, main or main@<commit>"):
        ob.parse_choice(value)


def test_write_jar_is_reproducible(tmp_path):
    files = {"b/x.class": b"\xca\xfe", "a/y.xml": b"<y/>"}
    one = ob.write_jar(files, b"Manifest-Version: 1.0\n", tmp_path / "one.jar")
    two = ob.write_jar(dict(reversed(list(files.items()))), b"Manifest-Version: 1.0\n", tmp_path / "two.jar")
    assert one.read_bytes() == two.read_bytes()
    with zipfile.ZipFile(one) as z:
        assert z.namelist() == ["META-INF/MANIFEST.MF", "a/y.xml", "b/x.class"]
        assert {i.date_time for i in z.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert {i.create_system for i in z.infolist()} == {3}
    assert not list(tmp_path.glob("*.part"))


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            elif kind == "link":
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tf.addfile(info)
    return buf.getvalue()


def test_extract_keeps_regular_files_inside_the_target(tmp_path):
    dest = tmp_path / "out"
    ob._extract(_tar([("root/ok.txt", "file", b"ok"), ("../escaped.txt", "file", b"no"),
                      ("/abs-escaped.txt", "file", b"no"), ("root/link", "link", "/etc/passwd")]), dest)
    assert (dest / "root" / "ok.txt").read_bytes() == b"ok"
    assert not (tmp_path / "escaped.txt").exists()
    assert not (dest / "root" / "link").exists() and not (dest / "root" / "link").is_symlink()
    assert sorted(p.name for p in dest.rglob("*") if p.is_file()) == ["ok.txt"]


def test_missing_javac_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(ob.shutil, "which", lambda name: None)
    with pytest.raises(ob.OpsinBuildError, match="JDK"):
        ob._javac()


def test_a_missing_jdk_stops_the_build_before_any_download(tmp_path, monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(ob.shutil, "which", lambda name: None)
    monkeypatch.setattr(ob, "_get", lambda *a, **k: pytest.fail("fetched the sources without a JDK"))
    monkeypatch.setattr(ob, "require_on_main", lambda commit, timeout=30: None)
    with pytest.raises(ob.OpsinBuildError, match="JDK"):
        ob.ensure_jar(SHA, tmp_path / "jars", tmp_path / "base.jar")
    assert files(tmp_path / "jars" / ob.PRIVATE_DIR) == []


class _OsWithoutGetuid:
    """``os`` as seen on a platform without ``os.getuid`` (Windows), for ``opsin_build`` only."""

    def __getattr__(self, name):
        if name == "getuid":
            raise AttributeError(name)
        return getattr(os, name)


def test_a_platform_without_getuid_is_told_so_at_the_first_use(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "os", _OsWithoutGetuid())
    for download in (True, False):
        with pytest.raises(ob.OpsinBuildError, match="needs a POSIX system") as err:
            ob.ensure_jar(SHA, tmp_path / "jars", tmp_path / "base.jar", download=download)
        assert "ORTHONYM_OPSIN" in str(err.value) and not isinstance(err.value.__cause__, AttributeError)
    assert not (tmp_path / "jars").exists()


def test_cached_commit_is_used_offline(tmp_path, monkeypatch):
    jar_dir, cache = private(tmp_path)
    (cache / "opsin-main-refs.json").write_text(json.dumps({"master": {"commit": SHA, "checked": 1}}))
    (cache / "opsin-main-refs.json").chmod(0o644)
    jar = plant_build(cache)

    def offline(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(ob, "resolve_commit", offline)
    got = ob.ensure_jar("master", jar_dir, tmp_path / "base.jar")
    assert got.commit == SHA and got.path == jar


def test_an_abbreviated_commit_names_the_build_that_is_here(tmp_path, monkeypatch):
    jar_dir, cache = private(tmp_path)
    jar = plant_build(cache)
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: pytest.fail("no network for a built commit"))
    assert ob.ensure_jar(SHA[:12], jar_dir, tmp_path / "base.jar").path == jar


def test_an_abbreviation_of_two_built_commits_is_refused(tmp_path, monkeypatch):
    jar_dir, cache = private(tmp_path)
    other = SHA[:7] + "0" * 33
    plant_build(cache)
    plant_build(cache, commit=other, payload=b"other jar")
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: pytest.fail("no guess by the network"))
    with pytest.raises(ob.OpsinBuildError, match="start of 2 OPSIN builds"):
        ob.ensure_jar(SHA[:7], jar_dir, tmp_path / "base.jar")
    assert ob.ensure_jar(SHA, jar_dir, tmp_path / "base.jar").commit == SHA  # the whole hash is not ambiguous


def test_a_fresh_ref_answer_is_not_asked_again(tmp_path, monkeypatch):
    jar_dir, cache = private(tmp_path)
    calls = []
    monkeypatch.setattr(ob, "resolve_commit", lambda ref, timeout=30: calls.append(ref) or SHA)
    assert ob._commit_for("master", cache, True) == SHA
    assert ob._commit_for("master", cache, True) == SHA
    assert calls == ["master"]
    assert json.loads((cache / "opsin-main-refs.json").read_text())["master"]["commit"] == SHA


def test_offline_with_nothing_cached_says_what_to_do(tmp_path, monkeypatch):
    def offline(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(ob, "resolve_commit", offline)
    with pytest.raises(ob.OpsinBuildError, match="unset ORTHONYM_OPSIN") as err:
        ob.ensure_jar("master", tmp_path / "jars", tmp_path / "base.jar")
    assert "URLError" in str(err.value)
    assert not list((tmp_path / "jars" / ob.PRIVATE_DIR).glob("opsin-main-*"))


def test_a_used_up_rate_limit_is_named(tmp_path, monkeypatch):
    def limited(*a, **k):
        raise urllib.error.HTTPError("https://api.github.com", 403, "rate limit exceeded", {}, None)
    monkeypatch.setattr(ob, "resolve_commit", limited)
    with pytest.raises(ob.OpsinBuildError, match="rate limit"):
        ob.ensure_jar("master", tmp_path / "jars", tmp_path / "base.jar")


def _run_returning(stdout):
    return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")


def test_selftest_accepts_the_expected_structures(tmp_path, monkeypatch):
    jar = tmp_path / "x.jar"
    jar.write_bytes(b"jar")
    monkeypatch.setattr(ob.subprocess, "run", _run_returning(GOOD_OUTPUT))
    ob.self_test(jar)


def test_selftest_rejects_a_wrong_jar(tmp_path, monkeypatch):
    jar = tmp_path / "x.jar"
    jar.write_bytes(b"jar")
    monkeypatch.setattr(ob.subprocess, "run", _run_returning("C\n" * len(ob._SELFTEST)))
    with pytest.raises(ob.OpsinBuildError, match="self-test"):
        ob.self_test(jar)


def test_selftest_rejects_a_jar_that_prints_the_wrong_number_of_lines(tmp_path, monkeypatch):
    jar = tmp_path / "x.jar"
    jar.write_bytes(b"jar")
    monkeypatch.setattr(ob.subprocess, "run", _run_returning("[warning] noise\n" + GOOD_OUTPUT))
    with pytest.raises(ob.OpsinBuildError, match="self-test"):
        ob.self_test(jar)


def test_a_jar_that_fails_its_selftest_is_never_left_in_the_jar_directory(tmp_path, monkeypatch):
    base = tmp_path / "base.jar"
    base.write_bytes(b"base")
    jar_dir = tmp_path / "jars"
    monkeypatch.setattr(ob, "build_jar", lambda commit, b, out: out.write_bytes(b"built") and out)

    def refuse(jar):
        raise ob.OpsinBuildError("self-test failed for " + jar.name)
    monkeypatch.setattr(ob, "self_test", refuse)
    monkeypatch.setattr(ob, "require_on_main", lambda commit, timeout=30: None)
    with pytest.raises(ob.OpsinBuildError, match="self-test"):
        ob.ensure_jar(SHA, jar_dir, base)
    assert files(jar_dir / ob.PRIVATE_DIR) == []


def test_a_checked_build_is_cached_with_its_sidecar(tmp_path, monkeypatch):
    base = tmp_path / "base.jar"
    base.write_bytes(b"base")
    jar_dir = tmp_path / "jars"
    built = []
    monkeypatch.setattr(ob, "build_jar", lambda commit, b, out: built.append(commit) or out.write_bytes(b"built") and out)
    monkeypatch.setattr(ob, "self_test", lambda jar: None)
    monkeypatch.setattr(ob, "require_on_main", lambda commit, timeout=30: None)
    got = ob.ensure_jar(SHA, jar_dir, base)
    cache = jar_dir / ob.PRIVATE_DIR
    assert got.path == cache / f"opsin-main-{SHA[:12]}.jar" and got.commit == SHA
    meta = json.loads(got.path.with_suffix(".json").read_text())
    assert meta["commit"] == SHA and meta["selftest"] == "pass"
    assert meta["jar_sha256"] == hashlib.sha256(got.path.read_bytes()).hexdigest()
    assert files(cache) == [got.path.name, got.path.with_suffix(".json").name]
    assert stat.S_IMODE(os.lstat(cache).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(got.path).st_mode) == 0o644  # not the umask of the user
    assert ob.ensure_jar(SHA, jar_dir, base).path == got.path and built == [SHA]


_HELLO = """package uk.ac.cam.ch.wwmm.opsin;
public class Hello {
    public static void main(String[] args) throws Exception {
        for (String r : new String[] {"opsinbuild.props", "/dep/data.txt"}) {
            java.io.BufferedReader in = new java.io.BufferedReader(
                new java.io.InputStreamReader(Hello.class.getResourceAsStream(r), "UTF-8"));
            System.out.println(in.readLine());
        }
    }
}
"""


def _fake_sources(commit):
    return _tar([
        (f"opsin-{commit}/opsin-core/src/main/java/uk/ac/cam/ch/wwmm/opsin/Hello.java", "file", _HELLO.encode()),
        (f"opsin-{commit}/opsin-core/src/main/resources/uk/ac/cam/ch/wwmm/opsin/opsinbuild.props", "file",
         b"version=${project.version}"),
    ])


def _fake_release_jar(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("META-INF/MANIFEST.MF",
                   "Manifest-Version: 1.0\r\nMain-Class: uk.ac.cam.ch.wwmm.opsin.Hello\r\n\r\n")
        z.writestr("dep/data.txt", "library")
        z.writestr("uk/ac/cam/ch/wwmm/opsin/Removed.txt", "a file of the release that main no longer has")
        z.writestr("META-INF/maven/uk.ac.cam.ch.opsin/opsin-core/pom.properties", "version=2.9.0")
    return path


@pytest.mark.skipif(shutil.which("javac") is None, reason="needs a JDK")
def test_build_jar_compiles_the_sources_over_the_libraries_of_the_release(tmp_path, monkeypatch):
    commit = SHA
    monkeypatch.setattr(ob, "_get", lambda url, timeout, accept=None, limit=0: _fake_sources(commit))
    base = _fake_release_jar(tmp_path / "release.jar")
    one = ob.build_jar(commit, base, tmp_path / "one.jar")
    two = ob.build_jar(commit, base, tmp_path / "two.jar")
    assert one.read_bytes() == two.read_bytes()
    with zipfile.ZipFile(one) as z:
        names = set(z.namelist())
        assert {"uk/ac/cam/ch/wwmm/opsin/Hello.class", "dep/data.txt"} <= names
        assert not [n for n in names if n.endswith("Removed.txt") or n.startswith("META-INF/maven/")]
        assert b"Main-Class: uk.ac.cam.ch.wwmm.opsin.Hello" in z.read("META-INF/MANIFEST.MF")
    with jvm_slots(1, purpose="test_opsin_build"):
        run = subprocess.run(java_cmd("-jar", str(one)), capture_output=True, text=True, timeout=60)
    assert run.stdout.splitlines() == [f"version=3.0-SNAPSHOT+{commit[:12]}", "library"]


@pytest.mark.skipif(shutil.which("javac") is None, reason="needs a JDK")
def test_a_source_error_is_a_build_error(tmp_path, monkeypatch):
    broken = _tar([(f"opsin-{SHA}/opsin-core/src/main/java/uk/ac/cam/ch/wwmm/opsin/Bad.java", "file",
                    b"package uk.ac.cam.ch.wwmm.opsin; class Bad { int x = ; }")])
    monkeypatch.setattr(ob, "_get", lambda url, timeout, accept=None, limit=0: broken)
    with pytest.raises(ob.OpsinBuildError, match="javac failed"):
        ob.build_jar(SHA, _fake_release_jar(tmp_path / "release.jar"), tmp_path / "x.jar")
    assert not (tmp_path / "x.jar").exists()


def test_an_archive_without_sources_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "_get", lambda url, timeout, accept=None, limit=0: _tar([(f"opsin-{SHA}/README.md", "file", b"x")]))
    with pytest.raises(ob.OpsinBuildError, match="no Java sources"):
        ob.build_jar(SHA, _fake_release_jar(tmp_path / "release.jar"), tmp_path / "x.jar")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("ORTHONYM_NETWORK_TESTS") != "1", reason="needs GitHub and a JDK")
def test_build_the_measured_commit(tmp_path):
    from orthonym.jars import find_jar
    base = Path(find_jar("opsin", "2.9.0"))
    sha = "9f30df5cc22c6f9e7f993a497cb3f65fdf4e5fb8"
    a = ob.ensure_jar(sha, tmp_path / "a", base)
    b = ob.ensure_jar(sha, tmp_path / "b", base)
    assert a.path.read_bytes() == b.path.read_bytes()
