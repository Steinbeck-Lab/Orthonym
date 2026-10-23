"""The two Java jars Orthonym uses, and where to find them.

Orthonym does not ship any jar. It uses two, pinned to one version and one
 checksum each, and downloads them from their official releases:

* OPSIN 2.9.0 (name -> structure), for the round-trip check of names.
* centres 1.2.1, for CIP stereo descriptors.

Resolution order for each jar:

1. ``ORTHONYM_OPSIN_JAR`` / ``ORTHONYM_CENTRES_JAR`` -- a path to the jar. It is used as
   given: the user named this exact file, so there is no search for it in a place others may
   write, and no check of the places along its path (the checks of 2. are for a jar that
   Orthonym finds or fetches by itself).
2. The jar directory (``ORTHONYM_JAR_DIR``, else ``<cache>/jars`` where
   ``<cache>`` is ``ORTHONYM_CACHE_DIR``, ``$XDG_CACHE_HOME/orthonym`` or
   ``~/.cache/orthonym``), if the file there has the pinned checksum.
3. A download into the jar directory, unless ``ORTHONYM_NO_DOWNLOAD=1``.

A jar runs with the user's rights, and the JVM is given the jar's path, not the file that was
hashed. The checksum therefore holds only while no other user can replace the jar, or rename a
directory above it, between the check and any later start of the JVM. Before a jar of the jar
directory is used, the directory and every directory above it must be proven out of other users'
reach (:func:`check_jar_dir`), as OpenSSH's StrictModes does for the files it trusts; otherwise
:class:`JarUnavailable` names the path and the fix.

``ORTHONYM_OPSIN=main`` (or ``main@<commit>``) is an opt-in to OPSIN's main branch: instead of
the release jar, step 2 and 3 give a jar built from that commit with the local JDK and cached
in the jar directory (``opsin_build``). Everything that asks for the OPSIN jar gets that one jar,
so one OPSIN serves a whole process. ``ORTHONYM_OPSIN_JAR`` still wins over it, and with the
variable unset or ``2.9.0`` nothing of the opt-in runs.

A jar that cannot be found is an error (:class:`JarUnavailable`): naming
without it would silently change names (no OPSIN check, a different CIP
labeller). ``ORTHONYM_ALLOW_REDUCED=1`` opts in to that reduced mode, in which
the finders return ``None`` as before. It does not apply to a jar directory that fails the check
above (:class:`JarDirUnsafe`), nor to a mistyped ``ORTHONYM_OPSIN``: those are setups to fix.

This module uses only the standard library, so the install-time build hook
can load it by file path before Orthonym itself is installed.
"""

import errno
import functools
import hashlib
import os
import shlex
import stat
import sys
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class JarSpec:
    kind: str
    version: str
    filename: str
    url: str
    sha256: str
    env_var: str


JARS: Dict[str, JarSpec] = {
    "opsin": JarSpec(
        kind="opsin",
        version="2.9.0",
        filename="opsin-cli-2.9.0-jar-with-dependencies.jar",
        url="https://github.com/dan2097/opsin/releases/download/2.9.0/"
            "opsin-cli-2.9.0-jar-with-dependencies.jar",
        sha256="c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f",
        env_var="ORTHONYM_OPSIN_JAR",
    ),
    "centres": JarSpec(
        kind="centres",
        version="1.2.1",
        filename="centres-cli-1.2.1.jar",
        url="https://github.com/SiMolecule/centres/releases/download/1.2.1/centres.jar",
        sha256="6c5d5cafa9393ed7b667a2340222348cd15a9bbdd0d1b92518229f26cc5e87fb",
        env_var="ORTHONYM_CENTRES_JAR",
    ),
}

FETCH_HINT = "run `orthonym --fetch-jars` (or set ORTHONYM_OPSIN_JAR / ORTHONYM_CENTRES_JAR)"
_DOWNLOAD_TIMEOUT_S = 120


class JarUnavailable(RuntimeError):
    """A pinned jar is missing and reduced mode was not requested.

    Deliberately NOT an ``OrthonymLimitError``: ``Orthonym.name`` turns those
    into 'unknown organic compound', and a missing jar must stop the run.
    """


class JarDirUnsafe(JarUnavailable):
    """The jar directory, a directory above it, or the jar is not out of other users' reach, so the jar is
    not used. Unlike a missing jar this is not met by ``ORTHONYM_ALLOW_REDUCED=1``: it is raised in reduced mode
    too (a setup to fix, and a mode that quietly drops the OPSIN check is no answer to a refusal about trust),
    and the command line does not suggest that mode."""


class OpsinChoiceError(JarUnavailable):
    """``ORTHONYM_OPSIN`` holds a value that is none of 2.9.0, main or main@<commit>. Always an error:
    reduced mode (``ORTHONYM_ALLOW_REDUCED=1``) is for a jar that cannot be found, not for a mistyped choice."""


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def allow_reduced() -> bool:
    """True when the user opted in to naming without the jars."""
    return _truthy(os.environ.get("ORTHONYM_ALLOW_REDUCED"))


def jar_dir() -> Path:
    """Directory where downloaded jars are kept (not created here)."""
    explicit = os.environ.get("ORTHONYM_JAR_DIR")
    if explicit:
        return Path(explicit).expanduser()
    cache = os.environ.get("ORTHONYM_CACHE_DIR")
    if cache:
        return Path(cache).expanduser() / "jars"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "orthonym" / "jars"
    try:
        return Path.home() / ".cache" / "orthonym" / "jars"
    except (RuntimeError, KeyError):  # no HOME and no passwd entry (containers)
        return _fallback_root() / "jars"


def _uid() -> Optional[int]:
    """The user id, or None where the platform has none (Windows)."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid else None


def _fallback_root() -> Path:
    """The directory above the jar directory when no home directory can be found: a per-user name
    in the shared temporary directory.:func:`check_jar_dir` makes it (mode 0700) and refuses one
    that is not the user's own."""
    uid = _uid()
    return Path(tempfile.gettempdir()) / f"orthonym-{'user' if uid is None else uid}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verified(path: Path, spec: JarSpec) -> bool:
    try:
        return path.is_file() and _sha256(path) == spec.sha256
    except OSError:
        return False


@functools.lru_cache(maxsize=None)
def _group_is_private(gid: int, uid: int) -> bool:
    """Group ``gid`` is the private group of user ``uid``, so that its write bit is the user's own.

    That is the user-private-group convention (Debian, Ubuntu, Fedora): the group is the user's primary
    group, is named like the user, and lists no member but the user. A group is called private only on
    that evidence, never because a search found nobody else: membership by primary group is not listed in
    the group database, and an account enumeration may be partial (LDAP) or slow. The password database is
    then searched for another account whose primary group this is, as one more reason to refuse, and only
    when the convention already holds. A user without a password entry (a container run with
    ``docker -u 1234``) has no private group: every group counts as shared. So does a group that cannot be
    looked up."""
    try:
        import grp
        import pwd
        user = pwd.getpwuid(uid)
        group = grp.getgrgid(gid)
    except (ImportError, KeyError, OSError):
        return False
    if gid != user.pw_gid or group.gr_name != user.pw_name or set(group.gr_mem) - {user.pw_name}:
        return False
    try:
        return all(entry.pw_gid != gid or entry.pw_name == user.pw_name for entry in pwd.getpwall())
    except (KeyError, OSError):
        return False


def _has_posix_acl(path: Path) -> bool:
    """``path`` has a POSIX access ACL (Linux, through ``os.listxattr``). With an ACL the group bits of the
    mode are the ACL's mask, so a named user's write permission shows up as the group-write bit, and even the
    user's private group can no longer be taken to be the only one with it. A filesystem without extended
    attributes has no ACL; any other failure to look counts as one. Known limit: where ``os.listxattr`` does not
    exist (macOS, Windows) ACLs are not looked at, so on macOS an entry added with ``chmod +a``, which does not
    show in the mode bits, is not seen, and a group-writable directory of the user's own group passes."""
    listxattr = getattr(os, "listxattr", None)
    if listxattr is None:
        return False
    try:
        return "system.posix_acl_access" in listxattr(path, follow_symlinks=False)
    except OSError as exc:
        return exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS)


def _others_have(path: Path, st: os.stat_result, uid: int, other_bit: int, group_bit: int) -> bool:
    """Some user besides the owner and root has a permission: the ``other_bit`` of the world class, or the
    ``group_bit`` of a group that is not the user's own private group, or of one that has an ACL."""
    if st.st_mode & other_bit:
        return True
    return bool(st.st_mode & group_bit) and (not _group_is_private(st.st_gid, uid) or _has_posix_acl(path))


class _Findings:
    """What is wrong with a jar directory chain (and jar): every directory at once, so that one error names
    them all and one ``chmod`` repairs them."""

    def __init__(self) -> None:
        self.problems: List[Tuple[Path, str]] = []
        self.writable: List[Path] = []  # what ``chmod go-w`` would repair
        self.foreign = False  # something belongs to another user: chmod is not the user's fix

    def other_owner(self, path: Path, st: os.stat_result) -> None:
        self.problems.append((path, f"is owned by another user (uid {st.st_uid})"))
        self.foreign = True

    def writable_by_others(self, path: Path, st: os.stat_result) -> None:
        self.problems.append((path, f"is writable by group or others (mode {stat.S_IMODE(st.st_mode):04o}, "
                                    f"gid {st.st_gid})"))
        self.writable.append(path)

    def raise_any(self, where: Path) -> None:
        if not self.problems:
            return
        fixes = []
        if self.writable:
            fixes.append("run `chmod go-w " + " ".join(shlex.quote(str(p)) for p in self.writable)
                         + "` (as their owner)")
        fixes.append("set ORTHONYM_JAR_DIR to a directory that only you can write")
        if self.foreign:
            fixes.append("set ORTHONYM_OPSIN_JAR / ORTHONYM_CENTRES_JAR to jars your site provides")
        raise JarDirUnsafe(
            f"not loading a jar from {where}: " + "; ".join(f"{p} {why}" for p, why in self.problems)
            + ". Another user could replace the jar after it has been checked, and the JVM would run their "
              "code with your rights. Fix: " + ", or ".join(fixes) + ".")


def _cannot_judge(where: Path, path: Path, why: str) -> JarDirUnsafe:
    return JarDirUnsafe(f"not loading a jar from {where}: {path} {why}, so it cannot be shown that no other "
                        f"user can change it. Set ORTHONYM_JAR_DIR to a directory of your own.")


def _walk(real: Path, where: Path, uid: int, found: _Findings) -> bool:
    """Add to ``found`` every directory of the chain of ``real`` that another user could modify, rename or
    have renamed; return whether other users can walk down into ``real``.

    ``real`` is a path with its symlinks already resolved (``os.path.realpath``): a symlinked
    ``~/.cache`` is fine, what counts is where it leads, and the caller must then use the resolved
    path, never the one with the link in it, which another user may be able to repoint. The chain
    is checked from ``/`` down to ``real``, which may not exist yet (the part that is there is
    checked). Each directory must belong to this user or root, and must not be writable
    by anyone else -- group or world, the group not counting when it is the user's own private group
    (``_group_is_private``) and has no ACL -- unless
    other users cannot reach it: a directory without search permission for others (a 0700
    ``~/.cache``) puts everything below it out of reach, writable or not. A sticky directory such as
    ``/tmp`` is passed (others cannot rename or delete the entries of this user or of root), but the
    jar directory itself may not be one. The walk goes to ``/`` and does not stop at the first
    directory that looks safe: a safe directory inside one that others can write can be renamed away
    and replaced, and the jar's path is used for as long as the process runs."""
    chain = []
    for entry in (real, *real.parents):
        try:
            st = os.lstat(entry)
        except FileNotFoundError:
            continue  # still to be created
        except OSError as exc:
            raise _cannot_judge(where, entry, f"cannot be inspected ({exc.strerror or exc})") from exc
        if not stat.S_ISDIR(st.st_mode):
            raise _cannot_judge(where, entry, "is not a directory")
        chain.append((entry, st))
    reachable = True  # other users can walk down to the directory being looked at
    for entry, st in reversed(chain):
        if st.st_uid not in (0, uid):
            found.other_owner(entry, st)
        elif (reachable and not (st.st_mode & stat.S_ISVTX and entry != real)
                and _others_have(entry, st, uid, stat.S_IWOTH, stat.S_IWGRP)
                and _others_have(entry, st, uid, stat.S_IXOTH, stat.S_IXGRP)):
            found.writable_by_others(entry, st)
        reachable = reachable and _others_have(entry, st, uid, stat.S_IXOTH, stat.S_IXGRP)
    return reachable


def _check_chain(real: Path, where: Path, uid: int) -> bool:
    """Raise:class:`JarDirUnsafe`, naming every directory at fault, unless no other user can change the
    chain of ``real`` (see:func:`_walk`); return whether other users can walk down into it."""
    found = _Findings()
    reachable = _walk(real, where, uid, found)
    found.raise_any(where)
    return reachable


def _private_root(root: Path, uid: int) -> None:
    """Make ``root`` (the per-user directory in the shared temporary directory) with mode 0700, or
    refuse what is there: it must be a real directory of this user that group and others cannot
    enter. Another user can make ``/tmp/orthonym-<uid>`` first, since the name is predictable."""
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise JarUnavailable(f"cannot make {root} for the jar directory ({exc.strerror or exc}); "
                             f"set ORTHONYM_JAR_DIR to a directory of your own.") from exc
    try:
        st = os.lstat(root)
    except OSError as exc:
        raise JarUnavailable(f"cannot inspect {root} ({exc.strerror or exc}); "
                             f"set ORTHONYM_JAR_DIR to a directory of your own.") from exc
    if not (stat.S_ISDIR(st.st_mode) and st.st_uid == uid and not st.st_mode & 0o077):
        kind = "a directory" if stat.S_ISDIR(st.st_mode) else "not a real directory (a symlink or a file)"
        raise JarDirUnsafe(
            f"not loading a jar from {root / 'jars'}: {root} is not a private directory of yours ({kind}, owner "
            f"uid {st.st_uid}, mode {stat.S_IMODE(st.st_mode):04o}); another user may have made it first. Remove "
            f"it (as its owner or root) and try again, or set HOME or ORTHONYM_JAR_DIR so that the jars go to a "
            f"directory of your own.")


def check_jar_dir(directory: Path) -> Path:
    """Raise:class:`JarDirUnsafe` unless jars may be loaded from ``directory`` (see:func:`_walk`),
    and return the directory with its symlinks resolved: the path that was checked, and the only one to
    hand to a JVM or to build a jar path from.

    Called before any jar of the jar directory is hashed, used or written, for the pinned jars and
    for the built OPSIN (whose own cache ``opsin-main`` is checked by ``opsin_build``). With no home
    directory the default ``<tmp>/orthonym-<uid>/jars`` is made here, mode 0700, and a ``<tmp>/orthonym-<uid>``
    that is not the user's own is refused. Skipped where the platform has no user ids (Windows: access
    follows ACLs rather than mode bits, and the profile directory that holds the default is private to the
    user); the opt-in build of OPSIN's main branch refuses such a platform on its own."""
    uid = _uid()
    if uid is None:
        return Path(directory)
    if directory == _fallback_root() / "jars":
        _private_root(_fallback_root(), uid)
    real = Path(os.path.realpath(directory))
    _check_chain(real, directory, uid)
    return real


def _check_jar_file(path: Path, where: Path) -> Optional[Path]:
    """Raise:class:`JarDirUnsafe` if the jar ``path`` is there but another user could change it in
    place: it belongs to neither this user nor root, or others can write it. A jar that is a symlink is
    judged where it leads, directory chain included, and that resolved path is returned. ``None`` for a
    jar that is absent (or not a regular file): the checksum step says so."""
    uid = _uid()
    if uid is None:
        return None
    real = Path(os.path.realpath(path))
    try:
        st = os.stat(real)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    found = _Findings()
    reachable = _walk(real.parent, where, uid, found)
    if st.st_uid not in (0, uid):
        found.other_owner(real, st)
    elif reachable and _others_have(real, st, uid, stat.S_IWOTH, stat.S_IWGRP):
        found.writable_by_others(real, st)
    found.raise_any(where)
    return real


def _release_target(spec: JarSpec) -> Tuple[Path, Path]:
    """``(directory, jar)``: the checked jar directory and where the pinned jar of ``spec`` lives in it, after
    the directory (and the jar, if it is there) passed the checks. Neither path has a symlink in it: the
    directory is the one that was checked, not ``jar_dir`` as spelled (a link in an unsafe place could be
    repointed after the check)."""
    directory = check_jar_dir(jar_dir())
    target = directory / spec.filename
    return directory, _check_jar_file(target, directory) or target


def _confirmed(spec: JarSpec, placed: Tuple[Path, Path]) -> Tuple[Path, Path]:
    """Look at the place of the jar again, now that the jar is in it (found, or downloaded), and return what the
    new look gives. A jar directory that did not exist at the first look is made afterwards, by this process or
    by someone else, and the first look only saw the part that was there; this one sees it all, and a directory
    or jar that is not the one first checked (a link put in its place, a directory made with loose modes) is
    refused with:class:`JarDirUnsafe`, never used."""
    again = _release_target(spec)
    if again != placed:
        raise JarDirUnsafe(
            f"not loading the {spec.kind} jar: its place changed while it was being found ({placed[1]} is now "
            f"{again[1]}), so another user may be changing it. Run it again, or set ORTHONYM_JAR_DIR to a "
            f"directory that only you can write.")
    return again


def make_dirs(path: Path) -> None:
    """Make ``path`` and any missing parent with mode 0755 (0700 for the per-user directory of the
    no-home fallback). ``Path.mkdir(parents=True)`` gives the parents ``0777 & ~umask``, which is
    group-writable under the umask 002 of a shared account, and such a directory is then refused."""
    missing = []
    entry = Path(path)
    while not os.path.lexists(entry) and entry != entry.parent:
        missing.append(entry)
        entry = entry.parent
    for entry in reversed(missing):
        try:
            os.mkdir(entry, 0o700 if entry == _fallback_root() else 0o755)
        except FileExistsError:
            pass


class _DirLock:
    """Best-effort inter-process lock on the jar directory (POSIX flock).

    Only an optimisation against N workers downloading the same file at once:
    correctness comes from temp-file + checksum + atomic replace. The lock file is opened without
    following a symlink and made with mode 0600 (``open(..., "a")`` follows a link and takes its mode
    from the umask, 0666 under umask 000); a lock that cannot be taken that way is skipped.
    """

    def __init__(self, directory: Path):
        self._path = directory / ".lock"
        self._fd: Optional[int] = None

    def __enter__(self):
        try:
            import fcntl
            self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            if self._fd is not None:
                os.close(self._fd)
            self._fd = None
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            os.close(self._fd)
            self._fd = None
        return False


def _download(spec: JarSpec, target: Path, verbose: bool = False, first_start: bool = False) -> None:
    """Download ``spec`` to ``target`` (temp file, checksum, atomic replace).

    ``first_start``: the download was not asked for (a namer found the jar missing), so
    say on stderr why this call is slow."""
    make_dirs(target.parent)
    checked = check_jar_dir(target.parent)  # also what was just made, and what another user put in its way
    if checked != target.parent:
        raise JarDirUnsafe(
            f"not loading a jar from {target.parent}: it leads to {checked} now, so another user may have put "
            f"a link in its place. Remove the link, or set ORTHONYM_JAR_DIR to a directory that only you can write.")
    with _DirLock(checked):
        if _verified(target, spec):  # another process finished while we waited
            return
        if target.exists():
            raise JarUnavailable(
                f"{target} exists but its SHA-256 does not match the pinned {spec.kind} "
                f"{spec.version} jar; remove it and {FETCH_HINT}")
        if verbose:
            print(f"[orthonym] downloading {spec.kind} {spec.version} from {spec.url}", file=sys.stderr)
        fd, tmp = tempfile.mkstemp(prefix=f".{spec.filename}.", dir=str(checked))
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(spec.url, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                if first_start:
                    # once: later starts find the verified jar in the jar directory
                    print(_first_start_message(spec, target, resp.headers.get("Content-Length")),
                          file=sys.stderr, flush=True)
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            got = _sha256(Path(tmp))
            if got != spec.sha256:
                raise JarUnavailable(
                    f"downloaded {spec.kind} jar has SHA-256 {got}, expected {spec.sha256}; not using it")
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def _first_start_message(spec: JarSpec, target: Path, length: Optional[str]) -> str:
    size = ""
    if length and length.isdigit():
        size = f" ({int(length) / 1e6:.1f} MB)"
    return (f"[orthonym] first start: downloading the {spec.kind} {spec.version} jar{size} to "
            f"{target.parent}. This happens once; later starts are fast. To fetch the jars "
            f"in advance, run `orthonym --fetch-jars`.")


_CACHE: Dict[str, object] = {}
_CACHE_LOCK = threading.Lock()


def _opsin_build():
    """The module that builds a main-branch jar. Imported when the opt-in is used, never at
    import: the install hook loads this file by path, before the package is installed."""
    import importlib
    return importlib.import_module("orthonym.opsin_build")


def opsin_choice() -> Optional[str]:
    """The git ref of OPSIN's main branch that ``ORTHONYM_OPSIN`` asks for; ``None`` for the
    release jar (the variable unset, empty or ``2.9.0``).

    Raises:class:`OpsinChoiceError` (a:class:`JarUnavailable`) for any other value."""
    if os.environ.get("ORTHONYM_OPSIN", "").strip() in ("", "2.9.0"):
        return None
    opsin_build = _opsin_build()
    try:
        return opsin_build.parse_choice(os.environ.get("ORTHONYM_OPSIN"))
    except opsin_build.OpsinBuildError as exc:
        raise OpsinChoiceError(str(exc)) from exc


#: path of a jar built from OPSIN's main branch -> the commit it was built from
_BUILT_COMMITS: Dict[str, str] = {}


def _resolve_built_opsin(spec: JarSpec, ref: str, download: bool, explicit: bool = False) -> str:
    """The jar of OPSIN ``ref`` (built and checked on first use), or:class:`JarUnavailable`.

    The release jar is resolved first: it supplies the libraries of the build. A failed fetch,
    build or self-test is an error here, never a fall back to the release jar. ``explicit``: the
    user asked for the build (``--fetch-jars``), so ``ORTHONYM_NO_DOWNLOAD`` does not apply."""
    opsin_build = _opsin_build()
    release, directory = _resolve_release_in(spec, download)  # the checked directory is what the cache is made in
    allowed = download and (explicit or not _truthy(os.environ.get("ORTHONYM_NO_DOWNLOAD")))
    try:  # ensure_jar locks (inside its private directory) only to build; the release jar's _DirLock is separate
        built = opsin_build.ensure_jar(ref, directory, Path(release), allowed)
    except opsin_build.OpsinBuildError as exc:
        raise JarUnavailable(f"ORTHONYM_OPSIN={os.environ.get('ORTHONYM_OPSIN')}: {exc}") from exc
    except Exception as exc:  # network, permissions, disk, a JDK that does not start
        raise JarUnavailable(f"ORTHONYM_OPSIN={os.environ.get('ORTHONYM_OPSIN')}: could not build OPSIN "
                             f"({type(exc).__name__}: {exc}); unset ORTHONYM_OPSIN to use the release jar") from exc
    _BUILT_COMMITS[str(built.path)] = built.commit
    return str(built.path)


def _resolve(spec: JarSpec, download: bool) -> str:
    override = os.environ.get(spec.env_var)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise JarUnavailable(f"{spec.env_var}={override} is not a file")
        return str(p.resolve())
    if spec.kind == "opsin":
        ref = opsin_choice()
        if ref is not None:
            return _resolve_built_opsin(spec, ref, download)
    return _resolve_release(spec, download)


def _resolve_release(spec: JarSpec, download: bool) -> str:
    """The pinned, checksummed release jar of ``spec``: in the jar directory, else downloaded."""
    return _resolve_release_in(spec, download)[0]


def _resolve_release_in(spec: JarSpec, download: bool) -> Tuple[str, Path]:
    """:func:`_resolve_release` and the checked jar directory it found the jar in. The place is looked at again
    once the jar is in it (:func:`_confirmed`), and the path returned is the one of that second look."""
    placed = _release_target(spec)
    target = placed[1]
    if _verified(target, spec):
        directory, jar = _confirmed(spec, placed)
        return str(jar), directory
    if not download or _truthy(os.environ.get("ORTHONYM_NO_DOWNLOAD")):
        raise JarUnavailable(f"the {spec.kind} {spec.version} jar is not in {target.parent}; {FETCH_HINT}")
    try:
        _download(spec, target, first_start=True)
    except JarUnavailable:
        raise
    except Exception as exc:  # network, permissions, disk
        raise JarUnavailable(
            f"could not download the {spec.kind} {spec.version} jar ({type(exc).__name__}: {exc}); "
            f"{FETCH_HINT}") from exc
    directory, jar = _confirmed(spec, placed)
    return str(jar), directory


def _cache_key(kind: str, download: bool) -> str:
    """Key of a resolved outcome. For OPSIN the choice is part of the key: the command line sets
    ``ORTHONYM_OPSIN`` after the package was imported (and may have looked for a jar), and an
    outcome found under another choice must not be reused."""
    key = f"{kind}:{int(download)}"
    if kind == "opsin":
        key += f":{os.environ.get('ORTHONYM_OPSIN', '').strip()}:{os.environ.get(JARS[kind].env_var, '')}"
    return key


def find_jar(kind: str, version: Optional[str] = None, download: bool = True) -> Optional[str]:
    """Absolute path to the pinned ``kind`` jar ('opsin' or 'centres').

    A ``version`` other than the pinned one returns ``None`` (only pinned,
    checksummed jars are ever used). The outcome -- a path or a failure -- is
    resolved once for the settings in force and then reused. For OPSIN the settings
    ``ORTHONYM_OPSIN`` and ``ORTHONYM_OPSIN_JAR`` are part of what the outcome is kept
    for (the command line sets ``ORTHONYM_OPSIN`` after the package was imported), so
    a process that changes them gets a new outcome; the in-process JVM takes its jar
    once, so choose before the first name. On failure: raises:class:`JarUnavailable`,
    or returns ``None`` when ``ORTHONYM_ALLOW_REDUCED=1``; a malformed ``ORTHONYM_OPSIN``
    (:class:`OpsinChoiceError`) and a jar directory that others can write (:class:`JarDirUnsafe`) are
    always raised: both are setups to fix, not a jar that is missing.

    With ``ORTHONYM_OPSIN`` set, ``find_jar("opsin")`` is the jar built from OPSIN's
    main branch, for a caller that names the pinned version too.
    """
    spec = JARS[kind]
    if version is not None and version != spec.version:
        return None
    key = _cache_key(kind, download)
    with _CACHE_LOCK:
        if key not in _CACHE:
            try:
                _CACHE[key] = _resolve(spec, download)
            except JarUnavailable as exc:
                _CACHE[key] = exc
        outcome = _CACHE[key]
    if isinstance(outcome, JarUnavailable):
        if allow_reduced() and not isinstance(outcome, (OpsinChoiceError, JarDirUnsafe)):
            return None
        raise outcome
    return outcome  # type: ignore[return-value]


def active_opsin(download: bool = True) -> Dict[str, Optional[str]]:
    """Which OPSIN this process uses: ``label`` (``"2.9.0"``, ``"main"`` or ``"custom"``, a jar
    given by ``ORTHONYM_OPSIN_JAR``), the ``commit`` it was built from (``None`` unless
    ``"main"``) and its ``path`` (``None`` in reduced mode).

    Resolves the jar like:func:`find_jar`, so it can raise:class:`JarUnavailable`."""
    path = find_jar("opsin", download=download)
    if path is None:
        return {"label": "2.9.0", "commit": None, "path": None}
    commit = _BUILT_COMMITS.get(path)
    if commit is not None:
        return {"label": "main", "commit": commit, "path": path}
    return {"label": "custom" if os.environ.get(JARS["opsin"].env_var) else "2.9.0", "commit": None, "path": path}


def require_all() -> None:
    """Raise:class:`JarUnavailable` unless every pinned jar is available
    (or reduced mode was requested). Called once when a namer is created."""
    for kind in JARS:
        find_jar(kind)


def fetch_all(verbose: bool = True) -> Dict[str, str]:
    """Download (if needed) and verify every pinned jar; return kind -> path.

    Ignores ``ORTHONYM_NO_DOWNLOAD`` and reduced mode: this is the explicit
    fetch. Raises:class:`JarUnavailable` on the first failure.
    """
    paths: Dict[str, str] = {}
    for kind, spec in JARS.items():
        override = os.environ.get(spec.env_var)
        if override:
            target = Path(override).expanduser()
            if not target.is_file():
                raise JarUnavailable(f"{spec.env_var}={override} is not a file")
            paths[kind] = str(target.resolve())
        else:
            placed = _release_target(spec)
            if not _verified(placed[1], spec):
                try:
                    _download(spec, placed[1], verbose=verbose)
                except JarUnavailable:
                    raise
                except Exception as exc:
                    raise JarUnavailable(
                        f"could not download the {kind} {spec.version} jar ({type(exc).__name__}: {exc})") from exc
            paths[kind] = str(_confirmed(spec, placed)[1])  # the second look, not the first
        if verbose:
            print(f"[orthonym] {kind} {spec.version}: {paths[kind]}", file=sys.stderr)
    _reset_cache()
    return paths


def fetch_opsin_choice(verbose: bool = True) -> Optional[str]:
    """Build (if needed) and check the jar that ``ORTHONYM_OPSIN`` asks for, ahead of use.

    Like:func:`fetch_all` it ignores ``ORTHONYM_NO_DOWNLOAD`` and reduced mode: this is the
    explicit request. Returns the jar's path, or ``None`` when the release jar is in use (the
    variable unset or ``2.9.0``, or a jar given by ``ORTHONYM_OPSIN_JAR``, which wins). Raises
    :class:`JarUnavailable` on failure."""
    if os.environ.get(JARS["opsin"].env_var):
        return None
    ref = opsin_choice()
    if ref is None:
        return None
    with _CACHE_LOCK:
        path = _resolve_built_opsin(JARS["opsin"], ref, True, explicit=True)
        _CACHE[_cache_key("opsin", True)] = path
    if verbose:
        print(f"[orthonym] opsin main {_BUILT_COMMITS[path][:12]}: {path}", file=sys.stderr)
    return path


def _reset_cache() -> None:
    """Forget resolved outcomes (tests, and after an explicit fetch)."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _BUILT_COMMITS.clear()
