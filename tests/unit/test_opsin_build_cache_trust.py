"""The cache of built OPSIN jars is a trust root: a jar from it runs with the user's rights.

Every test here is offline. A build is replaced by a function that writes a file, and GitHub by
fakes, so no JDK is needed.
"""
import hashlib
import json
import os
import stat
import urllib.error
from pathlib import Path

import pytest

from orthonym import opsin_build as ob

SHA = "e9f62eb38a6f2519056549cc16385a4257c1d55f"


def make_cache(tmp_path):
    jar_dir = tmp_path / "jars"
    return jar_dir, ob.private_dir(jar_dir, create=True)


def plant(cache, commit=SHA, payload=b"jar bytes", mode=0o644, meta_mode=0o644, **sidecar):
    jar = cache / f"opsin-main-{commit[:12]}.jar"
    jar.write_bytes(payload)
    jar.chmod(mode)
    meta = jar.with_suffix(".json")
    body = {"commit": commit, "jar_sha256": hashlib.sha256(payload).hexdigest()}
    body.update(sidecar)
    meta.write_text(json.dumps(body))
    meta.chmod(meta_mode)
    return jar, meta


@pytest.fixture
def builds(monkeypatch):
    """``build_jar``, ``self_test`` and ``require_on_main`` replaced; the list holds each commit built."""
    built = []

    def build(commit, base, out):
        built.append(commit)
        out.write_bytes(b"built by the test")
        return out
    monkeypatch.setattr(ob, "build_jar", build)
    monkeypatch.setattr(ob, "self_test", lambda jar: None)
    monkeypatch.setattr(ob, "require_on_main", lambda commit, timeout=30: None)
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: pytest.fail("the network was asked"))
    return built


BASE = Path("/nonexistent/base.jar")


def base_jar(tmp_path):
    base = tmp_path / "base.jar"
    base.write_bytes(b"base")
    return base


# --- the cached jar is checked on every use --------------------------------------------------------

def test_a_good_cached_build_is_used_without_a_build(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    jar, _ = plant(cache)
    assert ob.ensure_jar(SHA, jar_dir, BASE, download=False).path == jar
    assert ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path)).path == jar and builds == []


@pytest.mark.parametrize("what", ["group-writable jar", "world-writable jar", "group-writable sidecar",
                                  "symlinked jar", "symlinked sidecar", "hash mismatch", "other commit",
                                  "no hash", "unreadable sidecar"])
def test_a_cached_build_that_fails_a_check_is_not_used(tmp_path, builds, what):
    jar_dir, cache = make_cache(tmp_path)
    jar, meta = plant(cache)
    if what == "group-writable jar":
        jar.chmod(0o664)
    elif what == "world-writable jar":
        jar.chmod(0o646)
    elif what == "group-writable sidecar":
        meta.chmod(0o664)
    elif what == "symlinked jar":
        real = tmp_path / "real.jar"
        real.write_bytes(jar.read_bytes())
        jar.unlink()
        jar.symlink_to(real)
    elif what == "symlinked sidecar":
        real = tmp_path / "real.json"
        real.write_bytes(meta.read_bytes())
        meta.unlink()
        meta.symlink_to(real)
    elif what == "hash mismatch":
        meta.write_text(json.dumps({"commit": SHA, "jar_sha256": "0" * 64}))
    elif what == "other commit":
        meta.write_text(json.dumps({"commit": "f" * 40, "jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest()}))
    elif what == "no hash":
        meta.write_text(json.dumps({"commit": SHA}))
    else:
        meta.write_text("{not json")
    with pytest.raises(ob.OpsinBuildError, match="not built"):
        ob.ensure_jar(SHA, jar_dir, BASE, download=False)
    assert builds == []
    got = ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))  # downloads allowed: built again, over the bad one
    assert builds == [SHA] and got.path == jar
    assert not jar.is_symlink() and stat.S_IMODE(os.lstat(jar).st_mode) == 0o644
    assert ob.ensure_jar(SHA, jar_dir, BASE, download=False).path == jar  # and the new one checks out


def test_the_digest_of_a_new_build_is_that_of_the_jar_in_the_cache(tmp_path, builds):
    jar_dir, cache = make_cache(tmp_path)
    got = ob.ensure_jar(SHA, jar_dir, base_jar(tmp_path))
    meta = json.loads(got.path.with_suffix(".json").read_text())
    assert meta["jar_sha256"] == hashlib.sha256(b"built by the test").hexdigest()
    assert stat.S_IMODE(os.lstat(got.path.with_suffix(".json")).st_mode) == 0o644


# --- strings read from files ------------------------------------------------------------------------

@pytest.mark.parametrize("commit", ["/../../x", "../x", SHA.upper(), SHA[:39], SHA + "0", "g" * 40, "", None, 7,
                                    ["a" * 40]])
def test_a_commit_that_is_not_40_lowercase_hex_in_the_refs_file_is_ignored(tmp_path, monkeypatch, commit):
    jar_dir, cache = make_cache(tmp_path)
    (tmp_path / "x.jar").write_bytes(b"outside the jar directory")
    (tmp_path / "x.json").write_text("{}")
    refs = cache / "opsin-main-refs.json"
    refs.write_text(json.dumps({"master": {"commit": commit, "checked": ob.time.time()}}))
    refs.chmod(0o644)

    def offline(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(ob, "resolve_commit", offline)
    with pytest.raises(ob.OpsinBuildError, match="cannot tell which OPSIN commit"):
        ob.ensure_jar("master", jar_dir, BASE, download=True)


@pytest.mark.parametrize("checked", [1e12, float("nan"), float("inf"), -1, "now", None, True, [1]])
def test_a_checked_time_that_is_not_in_the_past_is_ignored(tmp_path, monkeypatch, checked):
    jar_dir, cache = make_cache(tmp_path)
    plant(cache)
    refs = cache / "opsin-main-refs.json"
    refs.write_text(json.dumps({"master": {"commit": SHA, "checked": checked}}))
    refs.chmod(0o644)
    asked = []
    monkeypatch.setattr(ob, "resolve_commit", lambda ref, timeout=30: asked.append(ref) or "a" * 40)
    monkeypatch.setattr(ob, "require_on_main", lambda commit, timeout=30: None)
    assert ob._commit_for("master", cache, download=True) == "a" * 40  # asked GitHub: the entry was not "fresh"
    assert asked == ["master"]


def test_an_old_but_well_formed_entry_still_serves_offline(tmp_path, monkeypatch):
    jar_dir, cache = make_cache(tmp_path)
    refs = cache / "opsin-main-refs.json"
    refs.write_text(json.dumps({"master": {"commit": SHA, "checked": ob.time.time() - 10 * ob._REF_TTL_S}}))
    refs.chmod(0o644)
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    assert ob._commit_for("master", cache, download=True) == SHA


def test_a_refs_file_that_is_a_symlink_or_group_writable_is_not_read(tmp_path):
    jar_dir, cache = make_cache(tmp_path)
    entry = {"master": {"commit": SHA, "checked": ob.time.time()}}
    real = tmp_path / "real-refs.json"
    real.write_text(json.dumps(entry))
    (cache / "opsin-main-refs.json").symlink_to(real)
    assert ob._known_ref(cache, "master") is None
    (cache / "opsin-main-refs.json").unlink()
    refs = cache / "opsin-main-refs.json"
    refs.write_text(json.dumps(entry))
    refs.chmod(0o664)
    assert ob._known_ref(cache, "master") is None
    refs.chmod(0o644)
    assert ob._known_ref(cache, "master")[0] == SHA


def test_a_sidecar_with_a_commit_that_is_not_hex_or_not_its_name_is_not_a_build(tmp_path):
    jar_dir, cache = make_cache(tmp_path)
    for name, commit in (("opsin-main-aaaaaaaaaaaa.json", "/../.."), ("opsin-main-bbbbbbbbbbbb.json", "c" * 40),
                         ("opsin-main-DDDDDDDDDDDD.json", "d" * 40), ("opsin-main-refs.json", "e" * 40),
                         ("opsin-main-cccccccccccc.json", "c" * 40)):
        (cache / name).write_text(json.dumps({"commit": commit}))
        (cache / name).chmod(0o644)
    assert ob._built_commits(cache) == ["c" * 40]


# --- the private directory --------------------------------------------------------------------------

def test_the_cache_directory_is_made_private(tmp_path):
    jar_dir = tmp_path / "jars"
    assert ob.private_dir(jar_dir, create=False) is None and not jar_dir.exists()
    cache = ob.private_dir(jar_dir, create=True)
    assert cache == jar_dir / "opsin-main" and stat.S_IMODE(os.lstat(cache).st_mode) == 0o700
    assert ob.private_dir(jar_dir, create=False) == cache


def test_a_cache_directory_that_others_can_write_is_refused_and_named(tmp_path):
    jar_dir, cache = make_cache(tmp_path)
    cache.chmod(0o770)
    with pytest.raises(ob.OpsinBuildError, match=str(cache)):
        ob.private_dir(jar_dir, create=True)
    with pytest.raises(ob.OpsinBuildError, match="not private"):
        ob.ensure_jar(SHA, jar_dir, BASE, download=False)


def test_a_symlinked_cache_directory_is_refused(tmp_path):
    jar_dir = tmp_path / "jars"
    jar_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (jar_dir / "opsin-main").symlink_to(elsewhere)
    with pytest.raises(ob.OpsinBuildError, match="not private"):
        ob.private_dir(jar_dir, create=True)


def test_a_cache_directory_of_another_user_is_refused(tmp_path, monkeypatch):
    jar_dir, cache = make_cache(tmp_path)
    monkeypatch.setattr(ob, "_uid", lambda: os.getuid() + 1)
    with pytest.raises(ob.OpsinBuildError, match="not private"):
        ob.private_dir(jar_dir, create=False)


# --- no fixed temporary name ------------------------------------------------------------------------

def test_a_symlink_at_the_old_temporary_name_is_not_followed(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    target = tmp_path / "out.json"
    (tmp_path / "out.json.part").symlink_to(victim)
    ob._write_json(target, {"a": 1})
    assert victim.read_text() == "precious" and json.loads(target.read_text()) == {"a": 1}
    assert stat.S_IMODE(os.lstat(target).st_mode) == 0o644


def test_a_symlink_at_the_target_is_replaced_not_followed(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    target = tmp_path / "out.json"
    target.symlink_to(victim)
    ob._write_json(target, {"a": 1})
    assert victim.read_text() == "precious" and not target.is_symlink()


def test_a_symlink_at_the_old_name_of_a_written_jar_is_not_followed(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    (tmp_path / "x.jar.part").symlink_to(victim)
    ob.write_jar({"a.txt": b"a"}, b"Manifest-Version: 1.0\n", tmp_path / "x.jar")
    assert victim.read_text() == "precious"


def test_the_capability_sidecar_is_written_through_a_new_file_and_only_if_trusted(tmp_path):
    from orthonym import opsin_capabilities as caps
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    jar = tmp_path / "opsin-main-abc.jar"
    jar.write_bytes(b"jar")
    side = jar.with_suffix(".json")
    side.write_text(json.dumps({"commit": SHA}))
    side.chmod(0o644)
    (tmp_path / "opsin-main-abc.json.part").symlink_to(victim)
    caps._store(str(jar), {"phane": True, "halogen_uide": False})
    assert victim.read_text() == "precious"
    assert json.loads(side.read_text())["capabilities"] == {"phane": True, "halogen_uide": False}
    assert caps._stored(str(jar)) == {"phane": True, "halogen_uide": False}
    side.chmod(0o664)  # group-writable: not read, not rewritten
    assert caps._stored(str(jar)) is None
    caps._store(str(jar), {"phane": False, "halogen_uide": False})
    assert json.loads(side.read_text())["capabilities"] == {"phane": True, "halogen_uide": False}


# --- the reviewer's attacks, offline ----------------------------------------------------------------

def test_a_jar_planted_in_the_jar_directory_is_not_found(tmp_path, monkeypatch):
    """A jar, a sidecar with a wrong hash and a refs file with a time in the far future, placed in the jar
    directory itself (the old place), then the same three in the private cache."""
    jar_dir = tmp_path / "jars"
    jar_dir.mkdir(mode=0o775)
    commit = "f" * 40
    jar, meta = plant(jar_dir, commit=commit, payload=b"attacker bytes", mode=0o664, meta_mode=0o664)
    meta.write_text(json.dumps({"commit": commit, "jar_sha256": "0" * 64}))
    (jar_dir / "opsin-main-refs.json").write_text(json.dumps({"master": {"commit": commit, "checked": 1e12}}))
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(ob.OpsinBuildError):
        ob.ensure_jar("master", jar_dir, BASE, download=False)
    cache = ob.private_dir(jar_dir, create=True)
    jar2, meta2 = plant(cache, commit=commit, payload=b"attacker bytes", mode=0o664)
    meta2.write_text(json.dumps({"commit": commit, "jar_sha256": "0" * 64}))
    (cache / "opsin-main-refs.json").write_text(json.dumps({"master": {"commit": commit, "checked": 1e12}}))
    with pytest.raises(ob.OpsinBuildError):
        ob.ensure_jar("master", jar_dir, BASE, download=False)


def test_a_commit_that_climbs_out_of_the_cache_returns_nothing(tmp_path, monkeypatch):
    jar_dir, cache = make_cache(tmp_path)
    (tmp_path / "x.jar").write_bytes(b"outside the jar dir")
    (tmp_path / "x.json").write_text("{}")
    (cache / "opsin-main-").mkdir()
    (cache / "opsin-main-refs.json").write_text(json.dumps({"master": {"commit": "/../../x", "checked": 1e12}}))
    monkeypatch.setattr(ob, "resolve_commit", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(ob.OpsinBuildError):
        ob.ensure_jar("master", jar_dir, BASE, download=False)
