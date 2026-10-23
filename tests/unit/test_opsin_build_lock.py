"""The lock that serialises builds of the OPSIN main-branch jar: inside the private cache directory, never
followed through a symlink, and taken only when a jar has to be built. Offline; a build is a function that
writes a file."""
import fcntl
import hashlib
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from orthonym import opsin_build as ob

SHA = "e9f62eb38a6f2519056549cc16385a4257c1d55f"


def make_cache(tmp_path):
    jar_dir = tmp_path / "jars"
    return jar_dir, ob.private_dir(jar_dir, create=True)


def base_jar(tmp_path):
    base = tmp_path / "base.jar"
    base.write_bytes(b"base")
    return base


def plant(cache, payload=b"jar bytes"):
    jar = cache / f"opsin-main-{SHA[:12]}.jar"
    jar.write_bytes(payload)
    jar.chmod(0o644)
    meta = jar.with_suffix(".json")
    meta.write_text(json.dumps({"commit": SHA, "jar_sha256": hashlib.sha256(payload).hexdigest()}))
    meta.chmod(0o644)
    return jar


@pytest.fixture
def builds(monkeypatch):
    built = []

    def build(commit, base, out):
        built.append(commit)
        out.write_bytes(b"built by the test")
        return out
    monkeypatch.setattr(ob, "build_jar", build)
    monkeypatch.setattr(ob, "self_test", lambda jar: None)
    monkeypatch.setattr(ob, "require_on_main", lambda commit, timeout=30: None)
    return built


def hold(path):
    """Take the lock file the way a competing process would; the returned descriptor releases it on close."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def in_thread(fn):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # reported by the test
            box["error"] = exc
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def test_the_lock_is_in_the_private_directory_and_made_0600(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))
    assert stat.S_IMODE(os.lstat(cache / ".lock").st_mode) == 0o600
    assert not (jar_dir / ".lock").exists()


def test_a_symlinked_lock_is_refused_and_not_followed(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    (cache / ".lock").symlink_to(tmp_path / "created-through-the-lock")
    with pytest.raises(ob.OpsinBuildError, match="build lock"):
        ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))
    assert not (tmp_path / "created-through-the-lock").exists() and builds == []
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    (cache / ".lock").unlink()
    (cache / ".lock").symlink_to(victim)
    with pytest.raises(ob.OpsinBuildError, match="build lock"):
        ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))
    assert victim.read_text() == "precious" and builds == []


@pytest.mark.parametrize("what", ["group-writable", "world-writable", "another user", "a directory"])
def test_a_lock_that_is_not_the_users_own_regular_file_is_refused(tmp_path, builds, monkeypatch, what):
    jar_dir, cache = make_cache(tmp_path)
    lock = cache / ".lock"
    if what == "a directory":
        lock.mkdir()
    else:
        lock.write_text("")
        lock.chmod({"group-writable": 0o660, "world-writable": 0o602, "another user": 0o600}[what])
    if what == "another user":
        monkeypatch.setattr(ob, "_uid", lambda: os.getuid() + 1)
        monkeypatch.setattr(ob, "private_dir", lambda d, create: cache)  # the directory check is not the subject
    with pytest.raises(ob.OpsinBuildError, match="build lock"):
        ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))
    assert builds == []


def test_a_cache_hit_takes_no_lock(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    jar = plant(cache)
    fd = hold(cache / ".lock")
    try:
        thread, box = in_thread(lambda: ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path)))
        thread.join(10)
        assert not thread.is_alive(), "a cache hit waited for the lock"
    finally:
        os.close(fd)
    assert box["value"].path == jar and builds == []


def test_a_lock_held_by_a_reader_of_the_file_does_not_block_a_hit_either(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    plant(cache)
    (cache / ".lock").write_text("")
    fd = os.open(cache / ".lock", os.O_RDONLY)  # a read-only descriptor can hold a flock
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        thread, box = in_thread(lambda: ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path), download=False))
        thread.join(10)
        assert not thread.is_alive() and "value" in box
    finally:
        os.close(fd)


def test_a_miss_waits_for_the_lock_and_then_builds(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    fd = hold(cache / ".lock")
    thread, box = in_thread(lambda: ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path)))
    thread.join(0.5)
    assert thread.is_alive() and builds == [], "a miss did not wait for the lock"
    os.close(fd)
    thread.join(10)
    assert not thread.is_alive() and builds == [SHA]
    assert box["value"].commit == SHA


def test_a_process_that_waited_uses_the_jar_the_other_one_built(tmp_path, builds, monkeypatch):
    jar_dir, cache = make_cache(tmp_path)
    real = ob._build_lock

    @ob.contextlib.contextmanager
    def other_process_finishes_first(c):
        with real(c):
            plant(c)
            yield
    monkeypatch.setattr(ob, "_build_lock", other_process_finishes_first)
    monkeypatch.setattr(ob, "require_on_main", lambda *a, **k: pytest.fail("checked a commit that was built"))
    got = ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))
    assert got.path == cache / f"opsin-main-{SHA[:12]}.jar" and builds == []


def test_no_lock_is_made_when_downloads_are_off_and_nothing_is_built(tmp_path):
    jar_dir = tmp_path / "jars"
    with pytest.raises(ob.OpsinBuildError, match="not built"):
        ob.ensure_jar(SHA, jar_dir, Path("/nonexistent"), download=False)
    assert not jar_dir.exists()
