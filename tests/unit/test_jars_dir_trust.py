"""orthonym.jars: a jar is loaded from the jar directory only when no other user can reach it.

The JVM is given the jar's path, not the file that was hashed, so a directory that another user can
write (or rename) lets them swap the jar after the check. ``check_jar_dir`` proves the whole chain of
directories out of reach, like OpenSSH's StrictModes. Every test is offline.

The directories above the scratch tree are replaced by synthetic root-owned 0755 ones (``world``), so
the verdict does not depend on where the temporary directory lives (a private ``/var/folders`` on macOS
would otherwise hide every refusal behind its own 0700); everything below is real, with real modes.
"""
import errno
import hashlib
import os
import stat
import sys
import types
import urllib.request
from pathlib import Path

import pytest

import orthonym.jars as jars
from orthonym import opsin_build

PAYLOAD = b"pinned bytes, not really a jar" * 50
SHA = hashlib.sha256(PAYLOAD).hexdigest()
ME = os.getuid()


def make(path, mode):
    """A directory with exactly ``mode`` (the umask does not narrow it); the parents it has to make are 0755."""
    path = Path(path)
    missing, entry = [], path
    while not entry.exists():
        missing.append(entry)
        entry = entry.parent
    for entry in reversed(missing):
        entry.mkdir()
        entry.chmod(0o755)
    path.chmod(mode)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """``tmp_path`` is a scratch root whose own ancestors look like ``/`` and ``/home``: root-owned, 0755."""
    scratch = tmp_path.resolve()
    ancestors = {str(scratch), *(str(p) for p in scratch.parents)}
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if os.fspath(path) in ancestors:
            return os.stat_result((stat.S_IFDIR | 0o755, 1, 1, 2, 0, 0, 4096, 0, 0, 0))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    for var in ("ORTHONYM_OPSIN_JAR", "ORTHONYM_CENTRES_JAR", "ORTHONYM_NO_DOWNLOAD", "ORTHONYM_ALLOW_REDUCED",
                "ORTHONYM_CACHE_DIR", "ORTHONYM_JAR_DIR", "ORTHONYM_OPSIN", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    private = jars._group_is_private  # a test may replace it; the cache to clear is this one's
    jars._reset_cache()
    private.cache_clear()
    yield scratch
    jars._reset_cache()
    private.cache_clear()


@pytest.fixture
def group(monkeypatch):
    """Decide whether a group counts as private (no member but the user), whatever the account databases say."""
    def decide(private):
        monkeypatch.setattr(jars, "_group_is_private", lambda gid, uid: private)
    return decide


def refused(directory, *needles):
    with pytest.raises(jars.JarUnavailable) as caught:
        jars.check_jar_dir(Path(directory))
    for needle in needles:
        assert needle in str(caught.value), str(caught.value)
    return str(caught.value)


# --- the ordinary layouts keep working --------------------------------------------------------------

@pytest.mark.parametrize("mode", [0o700, 0o755, 0o750, 0o555])
def test_a_jar_directory_that_others_cannot_write_is_accepted(world, mode):
    jars.check_jar_dir(make(world / "cache" / "orthonym" / "jars", mode))


def test_the_default_layout_of_a_shared_account_is_accepted(world, group):
    """Umask 002: a 0775 jar directory in 0775 ``orthonym``, in a 0700 ``.cache`` of a 0750 home. Nobody else
    can reach it, so it does not matter that the group (here assumed to have other members) can write it."""
    group(False)
    make(world / "home", 0o750)
    make(world / "home" / ".cache", 0o700)
    make(world / "home" / ".cache" / "orthonym", 0o775)
    jars.check_jar_dir(make(world / "home" / ".cache" / "orthonym" / "jars", 0o775))


def test_a_group_writable_chain_is_accepted_when_the_group_is_the_users_own(world, group):
    group(True)
    for part in ("home", "home/.cache", "home/.cache/orthonym", "home/.cache/orthonym/jars"):
        make(world / part, 0o775)
    jars.check_jar_dir(world / "home" / ".cache" / "orthonym" / "jars")


def test_a_jar_directory_that_does_not_exist_yet_is_accepted_below_a_safe_parent(world):
    make(world / "home", 0o755)
    jars.check_jar_dir(world / "home" / ".cache" / "orthonym" / "jars")


# --- what is refused ----------------------------------------------------------------------------------

def test_a_shared_non_sticky_jar_directory_is_refused(world):
    shared = make(world / "shared" / "jars", 0o777)
    message = refused(shared, str(shared), "writable by group or others", "chmod go-w", "ORTHONYM_JAR_DIR")
    assert "not loading a jar" in message


def test_every_directory_at_fault_is_named_in_one_error_with_one_chmod(world):
    """A shared account may have several: the user should not have to fix them one refusal at a time."""
    jar_dir = make(world / "x" / "y" / "jars", 0o755)
    make(world / "x" / "y", 0o777)
    make(world / "x", 0o777)
    with pytest.raises(jars.JarDirUnsafe) as caught:
        jars.check_jar_dir(jar_dir)
    message = str(caught.value)
    assert f"{world / 'x'} is writable" in message and f"{world / 'x' / 'y'} is writable" in message
    assert message.count("chmod") == 1 and f"chmod go-w {world / 'x'} {world / 'x' / 'y'}`" in message
    assert "ORTHONYM_JAR_DIR" in message
    assert "ORTHONYM_OPSIN_JAR" not in message  # nothing is another user's: chmod is the user's fix


def test_a_directory_of_another_user_names_the_site_jars_as_a_way_out(world, monkeypatch):
    jar_dir = make(world / "cache" / "jars", 0o700)
    monkeypatch.setattr(jars, "_uid", lambda: ME + 4242)
    with pytest.raises(jars.JarDirUnsafe, match="ORTHONYM_OPSIN_JAR / ORTHONYM_CENTRES_JAR to jars your site provides") as caught:
        jars.check_jar_dir(jar_dir)
    assert f"{world / 'cache'} is owned" in str(caught.value) and f"{jar_dir} is owned" in str(caught.value)
    assert "chmod" not in str(caught.value)  # no mode would help: the directories are not the user's


def test_a_loose_jar_in_an_unsafe_directory_is_named_with_the_directory_in_one_chmod(world):
    directory = make(world / "shared", 0o777)
    jar = plant(directory, 0o666)
    with pytest.raises(jars.JarDirUnsafe) as caught:
        jars._check_jar_file(jar, directory)
    assert f"chmod go-w {directory} {jar}`" in str(caught.value)


def test_a_group_writable_jar_directory_is_refused_when_another_user_is_in_the_group(world, group):
    group(False)
    shared = make(world / "shared" / "jars", 0o775)
    refused(shared, str(shared), "writable by group or others")


def test_a_group_writable_chain_is_refused_when_it_can_be_reached(world, group):
    """Not sealed: ``orthonym`` is 0755, so another user can walk to the 0775 jar directory."""
    group(False)
    make(world / "home", 0o755)
    make(world / "home" / ".cache", 0o755)
    jar_dir = make(world / "home" / ".cache" / "orthonym" / "jars", 0o775)
    refused(jar_dir, str(jar_dir))


def test_a_safe_directory_in_a_directory_that_others_can_write_is_refused(world):
    """The first safe-looking directory is not enough: ``x`` is 0777, so another user can rename ``x/jars``
    away and put their own in its place."""
    jar_dir = make(world / "x" / "jars", 0o700)
    make(world / "x", 0o777)
    refused(jar_dir, str(world / "x"), "writable by group or others")


def test_a_directory_of_another_user_is_refused(world, monkeypatch):
    jar_dir = make(world / "cache" / "jars", 0o700)
    monkeypatch.setattr(jars, "_uid", lambda: ME + 4242)
    refused(jar_dir, "owned by another user", str(world / "cache"))


def test_a_jar_directory_in_a_directory_of_another_user_is_refused_even_when_it_is_sticky(world, monkeypatch):
    sticky = make(world / "tmp", 0o1777)
    jar_dir = make(sticky / "jars", 0o700)
    jars.check_jar_dir(jar_dir)  # the user's own, in a sticky directory of their own
    monkeypatch.setattr(jars, "_uid", lambda: ME + 4242)  # now every directory of the scratch tree is another user's
    refused(jar_dir, "owned by another user")


def test_a_directory_that_is_not_a_directory_is_refused(world):
    (world / "file").write_bytes(b"x")
    refused(world / "file" / "jars", "cannot be inspected")


# --- sticky directories -------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [0o700, 0o755])
def test_a_sticky_parent_is_accepted_when_the_jar_directory_is_the_users_and_closed(world, mode):
    sticky = make(world / "tmp", 0o1777)
    jars.check_jar_dir(make(sticky / "jars", mode))


def test_a_jar_directory_that_others_can_write_is_refused_in_a_sticky_parent(world):
    sticky = make(world / "tmp", 0o1777)
    refused(make(sticky / "jars", 0o777), "writable by group or others")


def test_the_jar_directory_itself_may_not_be_a_sticky_shared_directory(world):
    refused(make(world / "tmp", 0o1777), "writable by group or others")


# --- symlinks -----------------------------------------------------------------------------------------

def test_a_symlinked_cache_is_accepted(world):
    real = make(world / "elsewhere" / "cache", 0o700)
    make(real / "orthonym" / "jars", 0o775)
    (world / "home").mkdir()
    (world / "home" / ".cache").symlink_to(real)
    jars.check_jar_dir(world / "home" / ".cache" / "orthonym" / "jars")


def test_a_symlink_to_an_unsafe_directory_is_refused_where_it_leads(world):
    unsafe = make(world / "shared", 0o777)
    make(unsafe / "jars", 0o755)
    (world / "home").mkdir()
    (world / "home" / ".cache").symlink_to(unsafe)
    refused(world / "home" / ".cache" / "jars", str(unsafe))


# --- the jar file -------------------------------------------------------------------------------------

def plant(directory, mode=0o644, payload=PAYLOAD):
    jar = Path(directory) / "pinned.jar"
    jar.write_bytes(payload)
    jar.chmod(mode)
    return jar


def refused_file(jar, directory, *needles):
    with pytest.raises(jars.JarUnavailable) as caught:
        jars._check_jar_file(jar, directory)
    for needle in needles:
        assert needle in str(caught.value), str(caught.value)


def test_a_jar_that_others_can_write_is_refused_in_a_reachable_directory(world):
    directory = make(world / "jars", 0o755)
    refused_file(plant(directory, 0o666), directory, "writable by group or others", "chmod go-w")
    jars._check_jar_file(plant(directory, 0o644), directory)


def test_a_group_writable_jar_follows_the_group_and_the_reach(world, group):
    directory = make(world / "jars", 0o755)
    jar = plant(directory, 0o664)
    group(True)
    jars._check_jar_file(jar, directory)
    group(False)
    refused_file(jar, directory, "writable by group or others")
    sealed = make(world / "private" / "jars", 0o755)
    make(world / "private", 0o700)
    jars._check_jar_file(plant(sealed, 0o664), sealed)  # out of reach: no other user can open the directory


def test_a_jar_of_another_user_is_refused(world, monkeypatch):
    directory = make(world / "jars", 0o755)
    jar = plant(directory)
    monkeypatch.setattr(jars, "_uid", lambda: ME + 4242)
    with pytest.raises(jars.JarUnavailable, match="owned by another user"):
        jars._check_jar_file(jar, directory)


def test_a_missing_jar_is_not_an_error_of_the_check(world):
    directory = make(world / "jars", 0o755)
    jars._check_jar_file(directory / "absent.jar", directory)


def test_a_jar_linked_in_from_an_unsafe_directory_is_refused(world):
    unsafe = make(world / "shared", 0o777)
    target = plant(unsafe, 0o644)
    directory = make(world / "jars", 0o755)
    (directory / "pinned.jar").symlink_to(target)
    refused_file(directory / "pinned.jar", directory, str(unsafe))


# --- the whole resolution -----------------------------------------------------------------------------

@pytest.fixture
def spec(monkeypatch):
    spec = jars.JarSpec(kind="opsin", version="9.9.9", filename="pinned.jar", url="http://127.0.0.1:9/never",
                        sha256=SHA, env_var="ORTHONYM_OPSIN_JAR")
    monkeypatch.setitem(jars.JARS, "opsin", spec)
    return spec


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was asked")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_a_jar_with_the_pinned_hash_is_not_used_from_an_unsafe_directory(world, spec, no_network, monkeypatch):
    directory = make(world / "shared" / "jars", 0o777)
    plant(directory)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    with pytest.raises(jars.JarUnavailable, match="not loading a jar"):
        jars.find_jar("opsin")


def test_reduced_mode_does_not_turn_an_unsafe_directory_into_a_silent_none(world, spec, no_network, monkeypatch):
    """``ORTHONYM_ALLOW_REDUCED=1`` is for a jar that cannot be found. An unsafe directory is a setup to fix, and
    a missing jar in a safe one still gives ``None``."""
    directory = make(world / "shared" / "jars", 0o777)
    plant(directory)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    monkeypatch.setenv("ORTHONYM_ALLOW_REDUCED", "1")
    jars._reset_cache()
    with pytest.raises(jars.JarDirUnsafe, match="not loading a jar"):
        jars.find_jar("opsin")
    with pytest.raises(jars.JarDirUnsafe):
        jars.require_all()
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(make(world / "cache" / "jars", 0o755)))
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    jars._reset_cache()
    assert jars.find_jar("opsin") is None


def test_the_same_jar_is_used_from_a_safe_directory(world, spec, no_network, monkeypatch):
    directory = make(world / "cache" / "jars", 0o755)
    plant(directory)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    assert jars.find_jar("opsin") == str((directory / spec.filename).resolve())


def test_a_jar_with_a_loose_mode_is_not_used_from_a_safe_directory(world, spec, no_network, monkeypatch):
    """The jar file check is wired into the resolution, not only correct on its own."""
    directory = make(world / "cache" / "jars", 0o755)
    plant(directory, 0o666)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    with pytest.raises(jars.JarUnavailable, match="writable by group or others") as caught:
        jars.find_jar("opsin")
    assert str(directory / spec.filename) in str(caught.value)


def test_the_jvm_gets_the_resolved_path_of_a_symlinked_cache_in_an_unsafe_home(world, spec, no_network, monkeypatch):
    """The home is 0777, so another user can repoint its ``.cache`` link after the check. The chain that was
    checked is the one the link led to, and that is the path the jar is used by."""
    real = make(world / "real" / "cache", 0o700)
    plant(make(real / "orthonym" / "jars", 0o755))
    home = make(world / "home", 0o777)
    (home / ".cache").symlink_to(real)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(home / ".cache" / "orthonym" / "jars"))
    path = jars.find_jar("opsin")
    assert path == str(real / "orthonym" / "jars" / spec.filename)
    assert os.path.realpath(path) == path and str(home) not in path
    assert jars.check_jar_dir(home / ".cache" / "orthonym" / "jars") == real / "orthonym" / "jars"


def test_a_download_through_a_symlinked_cache_lands_in_and_returns_the_resolved_directory(world, spec, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())
    real = make(world / "real" / "cache", 0o700)
    home = make(world / "home", 0o777)
    (home / ".cache").symlink_to(real)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(home / ".cache" / "orthonym" / "jars"))
    path = jars.find_jar("opsin")
    assert path == str(real / "orthonym" / "jars" / spec.filename) and Path(path).read_bytes() == PAYLOAD


def test_a_jar_linked_into_a_safe_directory_is_used_by_where_it_leads(world, spec, no_network, monkeypatch):
    elsewhere = make(world / "site" / "jars", 0o755)
    target = plant(elsewhere)
    directory = make(world / "cache" / "jars", 0o755)
    (directory / spec.filename).symlink_to(target)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    assert jars.find_jar("opsin") == str(target)


def test_a_symlinked_cache_that_leads_to_an_unsafe_directory_is_refused_not_resolved_away(world, spec, no_network,
                                                                                         monkeypatch):
    unsafe = make(world / "shared", 0o777)
    plant(make(unsafe / "jars", 0o755))
    home = make(world / "home", 0o755)
    (home / ".cache").symlink_to(unsafe)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(home / ".cache" / "jars"))
    with pytest.raises(jars.JarUnavailable, match=str(unsafe)):
        jars.find_jar("opsin")


def test_a_jar_given_by_path_keeps_its_behaviour(world, spec, no_network, monkeypatch):
    """``ORTHONYM_OPSIN_JAR`` names the exact file; no directory is searched and none is checked."""
    unsafe = make(world / "shared", 0o777)
    jar = plant(unsafe)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(unsafe))
    monkeypatch.setenv("ORTHONYM_OPSIN_JAR", str(jar))
    assert jars.find_jar("opsin") == str(jar.resolve())


def test_fetch_all_refuses_an_unsafe_directory(world, spec, no_network, monkeypatch):
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(make(world / "shared" / "jars", 0o777)))
    with pytest.raises(jars.JarUnavailable, match="not loading a jar"):
        jars.fetch_all(verbose=False)


def test_no_download_into_an_unsafe_directory(world, spec, no_network, monkeypatch):
    """The directory is judged before the first request, not after the file is written."""
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(make(world / "shared" / "jars", 0o777)))
    with pytest.raises(jars.JarUnavailable, match="not loading a jar"):
        jars.find_jar("opsin")


def test_the_download_judges_its_own_target_directory(world, spec, no_network):
    """``_download`` does not rely on its caller having checked: nothing is requested or written."""
    unsafe = make(world / "shared" / "jars", 0o777)
    with pytest.raises(jars.JarUnavailable, match="not loading a jar"):
        jars._download(spec, unsafe / spec.filename)
    assert list(unsafe.iterdir()) == []


class _Response:
    headers = {"Content-Length": str(len(PAYLOAD))}

    def __init__(self):
        self._data = PAYLOAD

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        data, self._data = self._data[:size], self._data[size:]
        return data


def test_a_download_makes_its_directories_closed_whatever_the_umask(world, spec, monkeypatch):
    """Python's ``mkdir(parents=True)`` gives the parents ``0777 & ~umask``; under umask 002 that is group-writable
    and the directory would be refused at the next start."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())
    directory = make(world / "home", 0o755) / ".cache" / "orthonym" / "jars"
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    old = os.umask(0o000)
    try:
        path = jars.find_jar("opsin")
    finally:
        os.umask(old)
    assert Path(path).read_bytes() == PAYLOAD
    for part in (".cache", ".cache/orthonym", ".cache/orthonym/jars"):
        assert stat.S_IMODE((world / "home" / part).stat().st_mode) == 0o755, part
    jars._reset_cache()
    assert jars.find_jar("opsin") == path  # and the next start accepts what this one made


# --- the lock file ------------------------------------------------------------------------------------

def test_the_lock_file_is_made_0600_whatever_the_umask(tmp_path):
    old = os.umask(0o000)
    try:
        with jars._DirLock(tmp_path):
            assert stat.S_IMODE((tmp_path / ".lock").stat().st_mode) == 0o600
    finally:
        os.umask(old)


def test_the_lock_file_is_not_followed_when_it_is_a_link(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    directory = make(tmp_path / "jars", 0o755)
    (directory / ".lock").symlink_to(elsewhere)
    with jars._DirLock(directory):
        pass  # the lock is skipped, not taken through the link
    assert not elsewhere.exists()


def test_the_lock_file_descriptor_is_closed_on_exit(tmp_path):
    with jars._DirLock(tmp_path) as lock:
        fd = lock._fd
        os.fstat(fd)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_the_lock_is_exclusive_and_released(tmp_path):
    import fcntl
    with jars._DirLock(tmp_path):
        probe = os.open(tmp_path / ".lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)
    probe = os.open(tmp_path / ".lock", os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free again
    finally:
        os.close(probe)


# --- a jar directory that did not exist at the first look ---------------------------------------------------

def test_a_link_planted_where_the_jar_directory_is_made_is_refused(world, spec, no_network, monkeypatch):
    """The first look saw only the part of the chain that existed. Another user plants ``new`` in the sticky
    directory before it is made, and the jar would land, and be used, through their link."""
    sticky = make(world / "tmp", 0o1777)
    victim = make(world / "victim", 0o755)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(sticky / "new" / "jars"))
    real_make_dirs = jars.make_dirs

    def plant_the_link(path):
        if not os.path.lexists(sticky / "new"):
            (sticky / "new").symlink_to(victim)
        real_make_dirs(path)

    monkeypatch.setattr(jars, "make_dirs", plant_the_link)
    with pytest.raises(jars.JarDirUnsafe, match="put a link in its place"):
        jars._resolve_release(spec, True)
    assert list((victim / "jars").iterdir()) == []  # nothing was requested, nothing written


def test_a_jar_directory_made_loose_after_the_first_look_is_refused(world, spec, monkeypatch):
    """With downloads off: the directory appears (0777, with a jar of the pinned hash) between the first look and
    the checksum, and the second look, once the jar is found, refuses it."""
    sticky = make(world / "tmp", 0o1777)
    new = sticky / "new" / "jars"
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(new))
    monkeypatch.setenv("ORTHONYM_NO_DOWNLOAD", "1")
    real_verified = jars._verified

    def racing(path, jar_spec):
        if not new.exists():
            make(new, 0o777)
            make(new.parent, 0o777)
            plant(new, 0o666)
        return real_verified(path, jar_spec)

    monkeypatch.setattr(jars, "_verified", racing)
    with pytest.raises(jars.JarDirUnsafe, match="writable by group or others"):
        jars._resolve_release(spec, False)


def test_a_link_repointed_to_another_safe_directory_after_the_first_look_is_refused(world, spec, monkeypatch):
    """Not unsafe, only not the place that was checked: the second look sees a different directory."""
    first = make(world / "first" / "jars", 0o755)
    second = make(world / "second" / "jars", 0o755)
    plant(first)
    plant(second)
    home = make(world / "home", 0o777)
    link = home / "cache"
    link.symlink_to(first.parent)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(link / "jars"))
    real_verified = jars._verified

    def repoint(path, jar_spec):
        found = real_verified(path, jar_spec)
        link.unlink()
        link.symlink_to(second.parent)
        return found

    monkeypatch.setattr(jars, "_verified", repoint)
    with pytest.raises(jars.JarDirUnsafe, match="its place changed while it was being found"):
        jars._resolve_release(spec, False)


def test_the_place_is_looked_at_again_after_a_download(world, spec, monkeypatch):
    directory = make(world / "cache" / "jars", 0o755)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))

    def download_then_open_up(jar_spec, target, **kwargs):
        plant(target.parent)
        target.parent.chmod(0o777)

    monkeypatch.setattr(jars, "_download", download_then_open_up)
    with pytest.raises(jars.JarDirUnsafe, match="writable by group or others"):
        jars._resolve_release(spec, True)


def test_fetch_all_looks_at_the_place_again_after_a_download(world, spec, monkeypatch):
    directory = make(world / "cache" / "jars", 0o755)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    monkeypatch.setattr(jars, "JARS", {"opsin": spec})

    def download_then_open_up(jar_spec, target, **kwargs):
        plant(target.parent)
        target.parent.chmod(0o777)

    monkeypatch.setattr(jars, "_download", download_then_open_up)
    with pytest.raises(jars.JarDirUnsafe, match="writable by group or others"):
        jars.fetch_all(verbose=False)


def test_fetch_all_looks_at_the_place_again_when_the_jar_was_found(world, spec, monkeypatch):
    sticky = make(world / "tmp", 0o1777)
    new = sticky / "new" / "jars"
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(new))
    monkeypatch.setattr(jars, "JARS", {"opsin": spec})
    real_verified = jars._verified

    def racing(path, jar_spec):
        if not new.exists():
            make(new, 0o777)
            make(new.parent, 0o777)
            plant(new, 0o666)
        return real_verified(path, jar_spec)

    monkeypatch.setattr(jars, "_verified", racing)
    with pytest.raises(jars.JarDirUnsafe, match="writable by group or others"):
        jars.fetch_all(verbose=False)


def test_fetch_all_returns_the_path_of_the_second_look(world, spec, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())
    real = make(world / "real" / "cache", 0o700)
    home = make(world / "home", 0o777)
    (home / ".cache").symlink_to(real)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(home / ".cache" / "orthonym" / "jars"))
    monkeypatch.setattr(jars, "JARS", {"opsin": spec})
    assert jars.fetch_all(verbose=False) == {"opsin": str(real / "orthonym" / "jars" / spec.filename)}


# --- the build of OPSIN's main branch ----------------------------------------------------------------

def test_the_main_branch_build_is_not_started_for_an_unsafe_directory(world, spec, monkeypatch):
    directory = make(world / "shared" / "jars", 0o777)
    plant(directory)
    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setattr(opsin_build, "ensure_jar", lambda *a, **k: pytest.fail("a jar was built into it"))
    with pytest.raises(jars.JarUnavailable, match="not loading a jar"):
        jars.find_jar("opsin")


def test_the_main_branch_build_runs_for_a_safe_directory(world, spec, monkeypatch):
    directory = make(world / "cache" / "jars", 0o755)
    plant(directory)
    built = directory / opsin_build.PRIVATE_DIR / "opsin-main-0123456789ab.jar"
    seen = []

    def ensure_jar(ref, jar_dir, base_jar, download=True):
        seen.append(Path(jar_dir))
        return opsin_build.BuiltJar(built, "0123456789ab" + "0" * 28, ref)

    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(directory))
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setattr(opsin_build, "ensure_jar", ensure_jar)
    assert jars.find_jar("opsin") == str(built) and seen == [directory]


def test_the_main_branch_build_gets_the_resolved_directory(world, spec, monkeypatch):
    real = make(world / "real" / "cache", 0o700)
    directory = make(real / "orthonym" / "jars", 0o755)
    plant(directory)
    home = make(world / "home", 0o777)
    (home / ".cache").symlink_to(real)
    seen = []

    def ensure_jar(ref, jar_dir, base_jar, download=True):
        seen.append((Path(jar_dir), Path(base_jar)))
        return opsin_build.BuiltJar(Path(jar_dir) / opsin_build.PRIVATE_DIR / "opsin-main-0123456789ab.jar",
                                    "0123456789ab" + "0" * 28, ref)

    monkeypatch.setenv("ORTHONYM_JAR_DIR", str(home / ".cache" / "orthonym" / "jars"))
    monkeypatch.setenv("ORTHONYM_OPSIN", "main")
    monkeypatch.setattr(opsin_build, "ensure_jar", ensure_jar)
    path = jars.find_jar("opsin")
    assert seen == [(directory, directory / spec.filename)] and os.path.realpath(path) == path


# --- no home directory --------------------------------------------------------------------------------

@pytest.fixture
def no_home(world, monkeypatch):
    """No HOME and no passwd entry: the jar directory falls back to ``<tmp>/orthonym-<uid>/jars``."""
    def refuse():
        raise RuntimeError("Could not determine home directory.")

    temp = make(world / "tmp", 0o1777)
    monkeypatch.setattr(Path, "home", staticmethod(refuse))
    monkeypatch.setattr(jars.tempfile, "gettempdir", lambda: str(temp))
    return temp / f"orthonym-{ME}"


def test_the_fallback_directory_is_made_private(no_home):
    assert jars.jar_dir() == no_home / "jars"
    assert not no_home.exists()
    jars.check_jar_dir(jars.jar_dir())
    assert stat.S_IMODE(no_home.stat().st_mode) == 0o700 and no_home.stat().st_uid == ME
    jars.make_dirs(jars.jar_dir())
    jars.check_jar_dir(jars.jar_dir())  # what _download makes next is accepted
    assert stat.S_IMODE(jars.jar_dir().stat().st_mode) & 0o077 in (0o055, 0o000)  # 0755, or narrower by the umask


def test_a_fallback_directory_made_in_advance_by_another_user_is_refused(no_home, monkeypatch):
    """Anyone can make ``/tmp/orthonym-<uid>`` first. Here the directory is the user's and the user is patched."""
    make(no_home / "jars", 0o700)
    make(no_home, 0o700)
    monkeypatch.setattr(jars, "_uid", lambda: ME + 4242)
    monkeypatch.setattr(jars, "_fallback_root", lambda: no_home)
    with pytest.raises(jars.JarUnavailable, match="not a private directory of yours") as caught:
        jars.check_jar_dir(no_home / "jars")
    assert str(no_home) in str(caught.value) and "another user may have made it" in str(caught.value)


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o770, 0o777])
def test_a_fallback_directory_that_is_not_closed_is_refused(no_home, mode):
    make(no_home / "jars", 0o700)
    make(no_home, mode)
    with pytest.raises(jars.JarUnavailable, match="not a private directory of yours"):
        jars.check_jar_dir(no_home / "jars")


def test_a_fallback_directory_that_is_a_symlink_is_refused(no_home, world):
    elsewhere = make(world / "elsewhere", 0o700)
    make(elsewhere / "jars", 0o700)
    no_home.symlink_to(elsewhere)
    with pytest.raises(jars.JarUnavailable, match="not a real directory"):
        jars.check_jar_dir(no_home / "jars")


def test_the_jar_is_found_in_the_fallback_directory(no_home, spec, no_network, monkeypatch):
    jars.check_jar_dir(jars.jar_dir())
    jars.make_dirs(jars.jar_dir())
    plant(jars.jar_dir())
    assert jars.find_jar("opsin") == str(jars.jar_dir() / spec.filename)


# --- platforms without user ids -------------------------------------------------------------------------

def test_the_check_is_skipped_where_there_are_no_user_ids(world, monkeypatch):
    unsafe = make(world / "shared" / "jars", 0o777)
    monkeypatch.delattr(os, "getuid")
    assert jars._uid() is None
    jars.check_jar_dir(unsafe)  # skipped, not an error
    jars._check_jar_file(plant(unsafe, 0o666), unsafe)
    assert jars.jar_dir().name == "jars"


# --- when a group is the user's own ---------------------------------------------------------------------

def fake_accounts(monkeypatch, users, groups):
    """Fake ``pwd`` and ``grp``. ``users``: uid -> (name, primary gid); ``groups``: gid -> (name, members)."""
    def entry(name, gid):
        return types.SimpleNamespace(pw_name=name, pw_gid=gid)

    def getpwuid(uid):
        return entry(*users[uid])

    def getgrgid(gid):
        name, members = groups[gid]
        return types.SimpleNamespace(gr_name=name, gr_mem=list(members))

    pwd = types.SimpleNamespace(getpwuid=getpwuid, getpwall=lambda: [entry(*v) for v in users.values()])
    monkeypatch.setitem(sys.modules, "pwd", pwd)
    monkeypatch.setitem(sys.modules, "grp", types.SimpleNamespace(getgrgid=getgrgid))
    jars._group_is_private.cache_clear()
    return pwd


@pytest.fixture
def accounts(monkeypatch):
    """User 1000 ``me`` with the private group ``me`` (gid 1000), root, another user ``other`` and a shared
    group ``staff`` (gid 50)."""
    users = {0: ("root", 0), 1000: ("me", 1000), 2000: ("other", 2000)}
    groups = {0: ("root", []), 1000: ("me", []), 2000: ("other", []), 50: ("staff", ["me", "other"])}
    fake_accounts(monkeypatch, users, groups)
    yield types.SimpleNamespace(users=users, groups=groups, monkeypatch=monkeypatch)
    jars._group_is_private.cache_clear()


def test_the_users_private_group_is_private(accounts):
    assert jars._group_is_private(1000, 1000)


@pytest.mark.parametrize("gid", [0, 50, 2000])
def test_any_other_group_is_not_private(accounts, gid):
    assert not jars._group_is_private(gid, 1000)


def test_a_group_with_the_users_gid_but_another_name_is_not_private(accounts):
    accounts.groups[1000] = ("users", [])
    jars._group_is_private.cache_clear()
    assert not jars._group_is_private(1000, 1000)


def test_a_group_named_like_the_user_but_not_the_users_primary_group_is_not_private(accounts):
    accounts.groups[51] = ("me", [])
    assert not jars._group_is_private(51, 1000)


def test_a_group_that_lists_another_member_is_not_private(accounts):
    accounts.groups[1000] = ("me", ["other"])
    jars._group_is_private.cache_clear()
    assert not jars._group_is_private(1000, 1000)


def test_a_group_that_is_another_accounts_primary_group_is_not_private(accounts):
    """Primary membership is not listed in ``gr_mem``; the account search is the extra reason to refuse."""
    accounts.users[3000] = ("pal", 1000)
    jars._group_is_private.cache_clear()
    assert not jars._group_is_private(1000, 1000)


def test_the_account_search_is_not_made_when_the_convention_already_fails(accounts, monkeypatch):
    """It can take seconds on a big directory, and it would decide nothing."""
    def refuse():
        raise AssertionError("the account database was enumerated")
    monkeypatch.setattr(sys.modules["pwd"], "getpwall", refuse)
    jars._group_is_private.cache_clear()
    assert not jars._group_is_private(50, 1000) and not jars._group_is_private(0, 1000)


def test_an_account_search_that_fails_is_not_taken_as_private(accounts, monkeypatch):
    def fail():
        raise OSError("directory service unreachable")
    monkeypatch.setattr(sys.modules["pwd"], "getpwall", fail)
    jars._group_is_private.cache_clear()
    assert not jars._group_is_private(1000, 1000)


def test_a_user_without_a_password_entry_has_no_private_group(accounts):
    """``docker run -u 1234``: no passwd entry, so every group is shared."""
    assert not jars._group_is_private(1000, 1234)


def test_a_group_that_cannot_be_looked_up_is_not_private(accounts):
    assert not jars._group_is_private(4242, 1000)


# --- ACLs -----------------------------------------------------------------------------------------------

@pytest.fixture
def own_group(world, monkeypatch):
    """The group of everything made here is the user's private group (as on Ubuntu), so a group-writable
    directory passes unless an ACL says otherwise."""
    gid = world.stat().st_gid
    fake_accounts(monkeypatch, {ME: ("me", gid), 0: ("root", 0)}, {gid: ("me", []), 0: ("root", [])})
    yield
    jars._group_is_private.cache_clear()


def with_acl(monkeypatch, *paths, error=None):
    """``os.listxattr`` says the given paths have a POSIX access ACL (or raises ``error``)."""
    acl = {os.fspath(p) for p in paths}
    real = os.listxattr

    def listxattr(path, *args, **kwargs):
        if error is not None:
            raise error
        return ["system.posix_acl_access"] if os.fspath(path) in acl else real(path, *args, **kwargs)

    monkeypatch.setattr(os, "listxattr", listxattr, raising=False)


def test_a_group_writable_directory_of_the_users_own_group_is_accepted_without_an_acl(world, own_group):
    jars.check_jar_dir(make(world / "cache" / "jars", 0o775))


def test_a_directory_with_an_acl_is_refused_whatever_its_group(world, own_group, monkeypatch):
    """A named-user ACL shows up as the group-write bit (the mask): the group is the user's own, and still
    someone else can write."""
    jar_dir = make(world / "cache" / "jars", 0o775)
    with_acl(monkeypatch, jar_dir)
    refused(jar_dir, "writable by group or others", str(jar_dir))


def test_a_jar_with_an_acl_is_refused(world, own_group, monkeypatch):
    directory = make(world / "jars", 0o755)
    jar = plant(directory, 0o664)
    jars._check_jar_file(jar, directory)
    with_acl(monkeypatch, jar)
    refused_file(jar, directory, "writable by group or others")


def test_the_acl_is_only_asked_for_when_a_group_bit_is_set(world, own_group, monkeypatch):
    def refuse(path, *args, **kwargs):
        raise AssertionError("listxattr was asked")
    monkeypatch.setattr(os, "listxattr", refuse, raising=False)
    jars.check_jar_dir(make(world / "cache" / "jars", 0o700))
    jars.check_jar_dir(make(world / "cache2" / "jars", 0o755))  # group bits r-x: read and search, no write


def test_a_filesystem_without_extended_attributes_has_no_acl(world, own_group, monkeypatch):
    jar_dir = make(world / "cache" / "jars", 0o775)
    with_acl(monkeypatch, error=OSError(errno.ENOTSUP, "Operation not supported"))
    jars.check_jar_dir(jar_dir)


def test_a_failure_to_read_the_acl_counts_as_one(world, own_group, monkeypatch):
    jar_dir = make(world / "cache" / "jars", 0o775)
    with_acl(monkeypatch, error=PermissionError(errno.EACCES, "Permission denied"))
    refused(jar_dir, "writable by group or others")


def test_acls_are_not_looked_at_where_the_platform_has_no_listxattr(world, own_group, monkeypatch):
    jar_dir = make(world / "cache" / "jars", 0o775)
    monkeypatch.delattr(os, "listxattr", raising=False)
    jars.check_jar_dir(jar_dir)
