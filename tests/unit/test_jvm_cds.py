"""The class-data archive of the in-process JVM (``orthonym.jvm_cds``).

A run of the command line that started the JVM leaves an archive of the JVM's classes in the jar directory, and
the JVM of a later process starts from it. What must hold: a missing, foreign or damaged archive changes nothing
on standard output (the JVM reports such an archive through its unified logging, on stdout, unless told not to);
a JDK before 19 gets no archive option; only a process that started its own JVM makes an archive, so the workers
of ``--jobs`` never write one.
"""
import os
import subprocess
import sys

import pytest

from orthonym import jvm_cds

pytestmark = pytest.mark.unit


def _jdk(tmp_path, version):
    home = tmp_path / "jdk"
    (home / "lib" / "server").mkdir(parents=True, exist_ok=True)
    (home / "lib" / "server" / "libjvm.so").write_bytes(b"")
    (home / "release").write_text(f'IMPLEMENTOR="x"\nJAVA_VERSION="{version}"\n')
    return str(home / "lib" / "server" / "libjvm.so")


@pytest.mark.parametrize("version, major", [("21.0.12", 21), ("19", 19), ("11.0.2", 11), ("1.8.0_392", 8)])
def test_the_jdk_version_is_read_from_its_release_file(tmp_path, version, major):
    assert jvm_cds.jdk_major(_jdk(tmp_path, version)) == major


def test_no_release_file_no_version(tmp_path):
    lib = tmp_path / "libjvm.so"
    lib.write_bytes(b"")
    assert jvm_cds.jdk_major(str(lib)) is None
    assert jvm_cds.jdk_major(None) is None


def test_a_jdk_before_19_gets_no_archive_option(tmp_path, monkeypatch):
    monkeypatch.setattr(jvm_cds, "_jvm_library", lambda: _jdk(tmp_path, "11.0.2"))
    assert jvm_cds.archive_path(["a.jar"]) is None
    assert jvm_cds.start_options(["a.jar"]) == []


def test_off_turns_it_off(monkeypatch):
    monkeypatch.setenv("ORTHONYM_JVM_CDS", "off")
    assert jvm_cds.archive_path(["a.jar"]) is None and jvm_cds.start_options(["a.jar"]) == []


def test_an_archive_is_used_with_the_jvm_messages_off(tmp_path, monkeypatch):
    archive = tmp_path / "x.jsa"
    monkeypatch.setattr(jvm_cds, "archive_path", lambda classpath: archive)
    assert jvm_cds.start_options(["a.jar"]) == []          # none made yet
    archive.write_bytes(b"x")
    assert jvm_cds.start_options(["a.jar"]) == [f"-XX:SharedArchiveFile={archive}", "-Xlog:cds*=off",
                                                "-Xlog:class+path*=off"]
    monkeypatch.setenv("ORTHONYM_JVM_CDS_DUMP", str(tmp_path / "new.part"))
    assert jvm_cds.start_options(["a.jar"])[0] == f"-XX:ArchiveClassesAtExit={tmp_path / 'new.part'}"
    assert "-Xlog:cds*=off" in jvm_cds.start_options(["a.jar"])


def test_a_process_that_did_not_start_the_jvm_makes_no_archive(monkeypatch):
    from orthonym import jvm_bridge
    monkeypatch.setattr(jvm_bridge, "_STARTED_PID", None)
    assert jvm_cds.build_if_missing() is None


# ---------------------------------------------------------------------------------------------------------------
# whole processes, with a jar directory of their own (the jars are given by path, as the user may give them)
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def own_jar_dir(tmp_path, monkeypatch):
    """The environment of a process whose archives go to a jar directory of this test, and the class path its
    JVM gets. Skipped where this JVM takes no archive (a JDK before 19, no JVM)."""
    from orthonym.perception.centres_bridge import _find_centres_jar
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    opsin, centres = _find_opsin_jar(), _find_centres_jar()
    jars = tmp_path / "jars"
    jars.mkdir(mode=0o700)
    env = dict(os.environ, ORTHONYM_JAR_DIR=str(jars), ORTHONYM_OPSIN_JAR=opsin, ORTHONYM_CENTRES_JAR=centres)
    env.pop("ORTHONYM_JVM_CDS", None)
    for key in ("ORTHONYM_JAR_DIR", "ORTHONYM_OPSIN_JAR", "ORTHONYM_CENTRES_JAR"):
        monkeypatch.setenv(key, env[key])
    monkeypatch.delenv("ORTHONYM_JVM_CDS", raising=False)
    path = jvm_cds.archive_path([opsin, centres])
    if path is None:
        pytest.skip("this JVM takes no class-data archive (JDK before 19, or no JVM)")
    return env, path


def jvm_cds_classpath(env):
    return [env["ORTHONYM_OPSIN_JAR"], env["ORTHONYM_CENTRES_JAR"]]


def _orthonym(env, *args):
    return subprocess.run([sys.executable, "-m", "orthonym", *args], env=env, capture_output=True, text=True,
                          timeout=300)


def test_a_run_leaves_the_archive_and_the_next_run_starts_from_it(own_jar_dir):
    env, path = own_jar_dir
    assert not path.exists()
    first = _orthonym(env, "CCO")
    assert (first.returncode, first.stdout) == (0, "ethanol\n")
    assert path.is_file() and path.stat().st_size > 0
    assert not list(path.parent.glob("*.part"))             # the file of the child was moved into place
    assert jvm_cds.start_options(jvm_cds_classpath(env))[0] == f"-XX:SharedArchiveFile={path}"   # the next JVM takes it
    second = _orthonym(env, "CC(=O)O")
    assert (second.returncode, second.stdout) == (0, "acetic acid\n")


def test_a_damaged_archive_changes_nothing_on_standard_output(own_jar_dir):
    """The JVM starts without it and says nothing: without ``-Xlog:cds*=off`` it prints its warning on stdout."""
    env, path = own_jar_dir
    path.parent.mkdir(mode=0o700)
    path.write_bytes(os.urandom(5000))
    done = _orthonym(env, "CCO")
    assert (done.returncode, done.stdout) == (0, "ethanol\n")
    assert "cds" not in done.stderr
    batch = path.parent.parent / "in.smi"
    batch.write_text("CCO\nCC(=O)O\n")
    done = _orthonym(env, "--batch", str(batch))
    assert done.stdout == "CCO\tethanol\nCC(=O)O\tacetic acid\n"


def test_the_workers_of_jobs_make_no_archive(own_jar_dir):
    env, path = own_jar_dir
    batch = path.parent.parent / "in.smi"
    batch.write_text("CCO\nCC(=O)O\nc1ccccc1\n")
    done = _orthonym(env, "--batch", str(batch), "--jobs", "2")
    assert done.stdout == "CCO\tethanol\nCC(=O)O\tacetic acid\nc1ccccc1\tbenzene\n"
    assert not path.parent.exists() or not list(path.parent.iterdir())


# ---------------------------------------------------------------------------------------------------------------
# the builder: once per key, in its own directory, and nothing it cannot account for
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def started_here(tmp_path, monkeypatch):
    """A process that started its JVM itself, whose archive would go to a private directory of this test."""
    from orthonym import jvm_bridge
    cds = tmp_path / "cds"
    archive = cds / "k.jsa"
    monkeypatch.setattr(jvm_bridge, "_STARTED_PID", os.getpid())
    monkeypatch.setattr(jvm_cds, "archive_path", lambda classpath: archive)
    monkeypatch.delenv("ORTHONYM_JVM_CDS_DUMP", raising=False)
    return archive


def test_a_build_that_leaves_no_archive_is_not_tried_again(started_here, monkeypatch):
    runs = []

    def run(*args, **kwargs):
        runs.append(kwargs)
        return subprocess.CompletedProcess(args, 1)
    monkeypatch.setattr(jvm_cds.subprocess, "run", run)
    assert jvm_cds.build_if_missing() is None
    assert jvm_cds.build_if_missing() is None
    assert len(runs) == 1 and started_here.with_suffix(".failed").exists()


def test_the_child_runs_in_the_archive_directory(started_here, monkeypatch):
    """With ``-m`` the child puts its working directory first on its path: the user's directory must not be it."""
    seen = {}

    def run(args, **kwargs):
        seen.update(kwargs)
        part = kwargs["env"]["ORTHONYM_JVM_CDS_DUMP"]
        with open(part, "wb") as f:
            f.write(b"archive")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(jvm_cds.subprocess, "run", run)
    assert jvm_cds.build_if_missing() == started_here
    assert seen["cwd"] == started_here.parent and started_here.read_bytes() == b"archive"
    assert not list(started_here.parent.glob("*.part"))


def test_ctrl_c_during_the_build_ends_quietly_and_leaves_no_part(started_here, monkeypatch):
    def run(args, **kwargs):
        open(kwargs["env"]["ORTHONYM_JVM_CDS_DUMP"], "wb").close()
        raise KeyboardInterrupt
    monkeypatch.setattr(jvm_cds.subprocess, "run", run)
    assert jvm_cds.build_if_missing() is None
    assert not list(started_here.parent.glob("*.part"))


def test_only_the_four_newest_archives_are_kept(tmp_path):
    for i in range(6):
        p = tmp_path / f"{i}.jsa"
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
    old_part = tmp_path / "x.1.part"
    old_part.write_bytes(b"x")
    os.utime(old_part, (1000, 1000))
    jvm_cds._prune(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2.jsa", "3.jsa", "4.jsa", "5.jsa"]


def test_an_archive_that_fails_the_jar_checks_is_not_used(tmp_path, monkeypatch):
    from orthonym import jars
    archive = tmp_path / "x.jsa"
    archive.write_bytes(b"x")
    monkeypatch.setattr(jvm_cds, "archive_path", lambda classpath: archive)
    monkeypatch.delenv("ORTHONYM_JVM_CDS_DUMP", raising=False)
    assert jvm_cds.start_options(["a.jar"])[0].startswith("-XX:SharedArchiveFile=")

    def unsafe(*a, **k):
        raise jars.JarDirUnsafe("another user can write it")
    monkeypatch.setattr(jars, "_check_jar_file", unsafe)
    assert jvm_cds.start_options(["a.jar"]) == []


def test_the_key_follows_the_whole_jdk_version_and_the_heap(tmp_path, monkeypatch):
    jar = tmp_path / "a.jar"
    jar.write_bytes(b"x")
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(tmp_path))
    lib = _jdk(tmp_path, "21.0.4")
    monkeypatch.setattr(jvm_cds, "_jvm_library", lambda: lib)
    first = jvm_cds.archive_path([str(jar)])
    (tmp_path / "jdk" / "release").write_text('JAVA_VERSION="21.0.5"\n')
    second = jvm_cds.archive_path([str(jar)])
    monkeypatch.setenv("ORTHONYM_JVM_XMX", "40g")
    third = jvm_cds.archive_path([str(jar)])
    assert first is not None and len({first, second, third}) == 3
