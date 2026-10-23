"""Build an OPSIN jar from a commit of github.com/dan2097/opsin, for ``ORTHONYM_OPSIN=main[@ref]``.

Standard library only, like ``jars.py``. OPSIN's main branch uses the same third-party
libraries as the 2.9.0 release, so the pinned release jar supplies them and only OPSIN's
own sources are compiled (``javac``; a JDK is needed for the opt-in only). Jar entries get
a fixed time and a fixed creator, so one commit, one release jar and one JDK always give the
same bytes. A jar is built in a private scratch directory, checked and hashed there, and
moved into the cache only when the check passes; the sidecar file ``opsin-main-<commit12>.json``
is written last and is what marks a build as complete, so a half-built jar is never used.

The cache is a trust root: a jar from it runs with the user's rights. It is a private
subdirectory of the jar directory (``opsin-main``, mode 0700, a real directory owned by the user
and not writable by group or others; anything else is refused with an error naming the path).
A cached jar and its sidecar are used only when both are regular files of the user that no one
else can write, the sidecar names the commit, and its is the jar's. A commit string read
from a file must be 40 lowercase hex digits. Downloads and the unpacked archive have size limits.
A commit other than the tip of OPSIN's main branch is built only when GitHub's compare API says it
is an ancestor of that branch (the source archive service also serves commits that exist only in
forks); when GitHub cannot say, the build is refused.

Nothing here runs at import time and nothing is fetched unless a jar is asked for.
"""
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO = "dan2097/opsin"
DEFAULT_BRANCH = "master"
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_UNIX = 3  # ZipInfo.create_system of a Unix zip, set explicitly so every platform writes the same bytes
_REF_TTL_S = 24 * 3600
PRIVATE_DIR = "opsin-main"
_REFS_FILE = "opsin-main-refs.json"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX_ABBREV = re.compile(r"[0-9a-f]{7,39}")
_SIDECAR_RE = re.compile(r"opsin-main-([0-9a-f]{12})\.json")
#: Limits, about thirty times what a commit needs today (2.2 MB archive, 30 MB unpacked, 397 entries).
MAX_ARCHIVE_BYTES = 64 << 20
MAX_UNPACKED_BYTES = 256 << 20
MAX_ENTRIES = 20_000
_MAX_API_BYTES = 8 << 20
_MODULES = ("opsin-core", "opsin-inchi", "opsin-cli")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_STALE_METADATA = "META-INF/maven/uk.ac.cam.ch.opsin/"
_VERSION_PROPS = "uk/ac/cam/ch/wwmm/opsin/opsinbuild.props"
#: (name, OPSIN SMILES): names every OPSIN since 2.x reads (plain, stereo, ionic). What the active
#: OPSIN reads beyond these, phane names for one, is the capability probe's question
#: (``opsin_capabilities``), not a reason to refuse a commit.
_SELFTEST = (("2-chloropyridine", "ClC1=NC=CC=C1"),
             ("(2S)-2-aminopropanoic acid", "N[C@H](C(=O)O)C"),
             ("sodium acetate", "C(C)(=O)[O-].[Na+]"))
_CHOICE_HELP = "use 2.9.0, main or main@<commit>"


class OpsinBuildError(RuntimeError):
    """The opt-in OPSIN jar could not be built, checked or found."""


@dataclass(frozen=True)
class BuiltJar:
    path: Path
    commit: str
    ref: str


def parse_choice(value: Optional[str]) -> Optional[str]:
    """``None`` for the release jar (unset, empty or ``2.9.0``), else the git ref to build."""
    v = (value or "").strip()
    if v in ("", "2.9.0"):
        return None
    if v == "main":
        return DEFAULT_BRANCH
    if v.startswith("main@") and len(v) > len("main@"):
        ref = v[len("main@"):]
        if _REF_RE.match(ref) and ".." not in ref:
            return ref
    raise OpsinBuildError(f"ORTHONYM_OPSIN={value!r}: {_CHOICE_HELP}")


def _get(url: str, timeout: int, accept: Optional[str] = None, limit: int = _MAX_API_BYTES) -> bytes:
    """The body of ``url``; an answer larger than ``limit`` bytes is an error, not read on."""
    headers = {"User-Agent": "orthonym"}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        declared = r.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise OpsinBuildError(f"{url} is {int(declared)} bytes, more than the {limit} Orthonym accepts")
        blocks, total = [], 0
        while True:
            block = r.read(1 << 20)
            if not block:
                return b"".join(blocks)
            total += len(block)
            if total > limit:
                raise OpsinBuildError(f"{url} sent more than the {limit} bytes Orthonym accepts")
            blocks.append(block)


def _explain(exc: BaseException) -> str:
    """The cause of a failed GitHub request, in words the user can act on."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (403, 429):
            return f"GitHub refused the request (HTTP {exc.code}; its rate limit for this address may be used up)"
        if exc.code == 404:
            return "GitHub has no such commit or branch in dan2097/opsin (HTTP 404)"
        return f"GitHub answered HTTP {exc.code}"
    return f"{type(exc).__name__}: {exc}"


def resolve_commit(ref: str, timeout: int = 30) -> str:
    sha = _get(f"https://api.github.com/repos/{REPO}/commits/{urllib.parse.quote(ref, safe='/')}", timeout,
               accept="application/vnd.github.sha", limit=4096).decode(errors="replace").strip()
    if not _HEX40.fullmatch(sha):
        raise OpsinBuildError(f"GitHub returned no commit for {ref!r}")
    return sha


def require_on_main(commit: str, timeout: int = 30) -> None:
    """Raise unless ``commit`` is the tip of OPSIN's main branch or one of its ancestors.

    The source archive service builds any commit of the repository network, forks included, so a
    commit named by hash is checked with GitHub's compare API (``status`` ``identical`` or
    ``behind``: the commit is in the history of the branch). Fails closed: an answer that cannot
    be had or read is a refusal."""
    try:
        data = json.loads(_get(f"https://api.github.com/repos/{REPO}/compare/{DEFAULT_BRANCH}...{commit}?per_page=1",
                               timeout, accept="application/vnd.github+json"))
    except (urllib.error.URLError, OSError, ValueError, OpsinBuildError) as exc:
        raise OpsinBuildError(
            f"cannot check that OPSIN commit {commit[:12]} is on OPSIN's main branch ({_explain(exc)}); a "
            f"commit is built only after that check. Connect to GitHub, or use a build that is already in the "
            f"jar directory.") from exc
    if not isinstance(data, dict) or data.get("status") not in ("behind", "identical"):
        raise OpsinBuildError(f"OPSIN commit {commit[:12]} is not on OPSIN's main branch (GitHub reports it as "
                              f"{data.get('status') if isinstance(data, dict) else None!r} to {DEFAULT_BRANCH}); "
                              f"only commits of the main branch are built.")


def _javac() -> str:
    candidates = []
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]) / "bin" / "javac")
    found = shutil.which("javac")
    if found:
        candidates.append(Path(found))
    for c in candidates:
        if c.is_file():
            return str(c)
    raise OpsinBuildError("ORTHONYM_OPSIN=main needs a Java JDK (javac, version 11 or newer); only a JRE was "
                          "found. Install a JDK (conda: openjdk), use a build already in the jar directory, "
                          "or unset ORTHONYM_OPSIN.")


def write_jar(files: Dict[str, bytes], manifest: bytes, out: Path) -> Path:
    fd, part = tempfile.mkstemp(dir=out.parent, prefix=".jar-", suffix=".tmp")  # a new file: no name to plant
    try:
        with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w") as z:
            for name, data in (("META-INF/MANIFEST.MF", manifest), *((n, files[n]) for n in sorted(files))):
                info = zipfile.ZipInfo(name, _FIXED_TIME)
                info.create_system = _UNIX
                info.external_attr = 0o644 << 16
                z.writestr(info, data, zipfile.ZIP_DEFLATED)
        os.replace(part, out)
    finally:
        Path(part).unlink(missing_ok=True)
    return out


def _gunzip(data: bytes, limit: int) -> bytes:
    """``data`` gunzipped; more than ``limit`` bytes is an error (a compressed bomb is stopped, not unpacked)."""
    out, total = io.BytesIO(), 0
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        while True:
            block = gz.read(1 << 20)
            if not block:
                return out.getvalue()
            total += len(block)
            if total > limit:
                raise OpsinBuildError(f"the OPSIN source archive unpacks to more than {limit} bytes")
            out.write(block)


def _extract(data: bytes, dest: Path) -> None:
    """Unpack a source archive: regular files and directories only, nothing outside ``dest``, at most
    ``MAX_ENTRIES`` entries and ``MAX_UNPACKED_BYTES`` bytes."""
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(_gunzip(data, MAX_UNPACKED_BYTES)), mode="r:") as tf:
        members, size = [], 0
        for m in tf:  # one entry at a time: a flood of entries is stopped at the count, not collected
            if len(members) >= MAX_ENTRIES:
                raise OpsinBuildError(f"the OPSIN source archive has more than {MAX_ENTRIES} entries")
            members.append(m)
            size += m.size if m.isreg() else 0
            if size > MAX_UNPACKED_BYTES:
                raise OpsinBuildError(f"the OPSIN source archive holds more than {MAX_UNPACKED_BYTES} bytes of files")
        safe = [m for m in members if (m.isreg() or m.isdir()) and (root / m.name).resolve().is_relative_to(root)]
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(dest, members=safe, **kwargs)


def _javac_run(javac: str, sources: Path, deps: Path, classes: Path) -> None:
    from .jvm_budget import jvm_slots
    from .jvm_flags import JVM_HYGIENE_FLAGS
    cmd = [javac, *(f"-J{f}" for f in JVM_HYGIENE_FLAGS), "-J-Xmx1g", "-nowarn", "-encoding", "UTF-8",
           "--release", "8", "-cp", str(deps), "-d", str(classes), f"@{sources}"]
    with jvm_slots(1, purpose="opsin-main-javac"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise OpsinBuildError("javac failed:\n" + proc.stderr[-2000:])


def build_jar(commit: str, base_jar: Path, out: Path) -> Path:
    """Compile the sources of ``commit`` against the libraries of ``base_jar`` and write the jar to ``out``."""
    javac = _javac()  # before any download: no JDK is the commonest reason for a failed build
    print(f"[orthonym] building OPSIN {commit[:12]} from github.com/{REPO} with the local JDK; this happens "
          f"once per commit.", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="orthonym-opsin-") as tmp:
        tmp = Path(tmp)
        archive = _get(f"https://codeload.github.com/{REPO}/tar.gz/{commit}", 300, limit=MAX_ARCHIVE_BYTES)
        try:
            _extract(archive, tmp / "src")
        except (tarfile.TarError, EOFError, zlib.error, OSError) as exc:
            raise OpsinBuildError(f"the OPSIN source archive of {commit[:12]} is unreadable: {exc}") from exc
        roots = [p for p in (tmp / "src").iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise OpsinBuildError("unexpected layout of the OPSIN source archive")
        src, deps, classes = roots[0], tmp / "deps", tmp / "classes"
        deps.mkdir()
        classes.mkdir()
        files: Dict[str, bytes] = {}
        try:
            with zipfile.ZipFile(base_jar) as z:
                manifest = z.read("META-INF/MANIFEST.MF")
                for info in z.infolist():
                    name = info.filename
                    if (info.is_dir() or name.startswith("uk/") or name == "META-INF/MANIFEST.MF"
                            or name.startswith(_STALE_METADATA)):
                        continue
                    files[name] = z.read(info)
                    target = deps / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(files[name])
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise OpsinBuildError(f"the OPSIN 2.9.0 jar {base_jar} cannot supply the libraries: {exc}") from exc
        sources = sorted(str(p) for m in _MODULES for p in (src / m / "src/main/java").rglob("*.java"))
        if not sources:
            raise OpsinBuildError("the OPSIN source archive holds no Java sources")
        (tmp / "sources.txt").write_text("\n".join(sources))
        _javac_run(javac, tmp / "sources.txt", deps, classes)
        for m in _MODULES:
            res = src / m / "src/main/resources"
            if res.is_dir():
                shutil.copytree(res, classes, dirs_exist_ok=True)
        for p in classes.rglob("*"):
            if p.is_file():
                files[p.relative_to(classes).as_posix()] = p.read_bytes()
        if _VERSION_PROPS in files:  # Maven fills this in; without it the jar would say '${project.version}'
            files[_VERSION_PROPS] = files[_VERSION_PROPS].replace(
                b"${project.version}", f"3.0-SNAPSHOT+{commit[:12]}".encode())
        return write_jar(files, manifest, out)


def _same(got: str, want: str) -> bool:
    try:
        from rdkit import Chem
        g, w = Chem.MolFromSmiles(got), Chem.MolFromSmiles(want)
        return g is not None and w is not None and Chem.MolToInchiKey(g) == Chem.MolToInchiKey(w)
    except ImportError:
        return got == want


def self_test(jar: Path) -> None:
    """Read the names of ``_SELFTEST`` with ``jar``; raise unless every structure is the expected one."""
    from .jvm_budget import jvm_slots
    from .jvm_flags import aligned_batch_lines, java_cmd
    with jvm_slots(1, purpose="opsin-main-selftest"):
        proc = subprocess.run(java_cmd("-jar", str(jar), "-osmi", heap="512m"),
                              input="".join(n + "\n" for n, _ in _SELFTEST),
                              capture_output=True, text=True, timeout=180)
    got = aligned_batch_lines(proc.stdout, len(_SELFTEST), "opsin self-test")
    if got is None or not all(_same(g, w) for g, (_, w) in zip(got, _SELFTEST)):
        raise OpsinBuildError(f"self-test failed for {jar.name}: {proc.stdout.strip().splitlines()!r}")


def _uid() -> int:
    try:
        return os.getuid()
    except AttributeError:
        raise OpsinBuildError(
            "the OPSIN main opt-in needs a POSIX system: it checks that the jar cache belongs to you with "
            "os.getuid, which this platform does not have. Use the release jar (unset ORTHONYM_OPSIN, or "
            "--opsin 2.9.0), or give a jar with ORTHONYM_OPSIN_JAR.") from None


def _trusted(path: Path, directory: bool) -> bool:
    """``path`` is a real directory (``directory``) or a regular file of this user that no one else can write:
    no symlink (``os.lstat``), owned by ``_uid``, no group or other write bit."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    kind = stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)
    return kind and st.st_uid == _uid() and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def trusted_file(path: Path) -> bool:
    """``path`` is a regular file of this user (no symlink) that group and others cannot write."""
    return _trusted(Path(path), False)


def private_dir(jar_dir: Path, create: bool) -> Optional[Path]:
    """The private cache ``<jar_dir>/opsin-main``: made with mode 0700 when ``create`` and absent; None when it is
    absent and not asked for. A directory that is there but is not a real directory of this user without group
    or other write is refused (:class:`OpsinBuildError` naming it), never used."""
    directory = Path(jar_dir) / PRIVATE_DIR
    if create:
        Path(jar_dir).mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
    try:
        os.lstat(directory)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OpsinBuildError(f"cannot use {directory}: {exc}") from exc
    if not _trusted(directory, True):
        raise OpsinBuildError(
            f"{directory} is not private to this user (it must be a real directory, not a symlink, owned by you, "
            f"that group and others cannot write), so no jar from it is used. Remove it, or fix its owner and mode "
            f"(chmod 700), or point ORTHONYM_JAR_DIR at a directory of your own.")
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` through a new file of a unique name (``mkstemp``) in the same directory and
    ``os.replace``: no fixed temporary name that a planted symlink could take over, and a planted file or
    symlink at ``path`` is replaced, not followed."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".json-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=1))
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    """The JSON object in ``path``, or {} (not a trusted regular file, unreadable or not an object)."""
    if not trusted_file(path):
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _built_commits(cache: Optional[Path]) -> List[str]:
    """The commits that have a sidecar in ``cache`` (a trusted file of the right name that names a 40-hex
    commit starting with the 12 characters of its name), sorted."""
    found = set()
    for meta in (cache.glob("opsin-main-*.json") if cache is not None else ()):
        named = _SIDECAR_RE.fullmatch(meta.name)
        commit = _read_json(meta).get("commit") if named else None
        if isinstance(commit, str) and _HEX40.fullmatch(commit) and commit[:12] == named.group(1):
            found.add(commit)
    return sorted(found)


def _known_ref(cache: Optional[Path], ref: str):
    """``(commit, age in seconds)`` that the refs file of ``cache`` holds for ``ref``, or None when it holds
    none or an entry that is not well formed: a commit that is not 40 lowercase hex digits, a ``checked`` that is
    not a finite number or lies in the future."""
    entry = _read_json(cache / _REFS_FILE).get(ref) if cache is not None else None
    if not isinstance(entry, dict):
        return None
    commit, checked = entry.get("commit"), entry.get("checked")
    if not (isinstance(commit, str) and _HEX40.fullmatch(commit)):
        return None
    if isinstance(checked, bool) or not isinstance(checked, (int, float)):
        return None
    age = time.time() - checked
    return (commit, age) if math.isfinite(age) and age >= 0 else None


def _commit_for(ref: str, cache: Optional[Path], download: bool) -> str:
    if _HEX40.fullmatch(ref):
        return ref
    if _HEX_ABBREV.fullmatch(ref):  # an abbreviated commit names a build that is already here
        matches = [c for c in _built_commits(cache) if c.startswith(ref)]
        if len(matches) > 1:
            raise OpsinBuildError(f"{ref!r} is the start of {len(matches)} OPSIN builds in the jar directory "
                                  f"({', '.join(c[:12] for c in matches)}); give more of the commit.")
        if matches:
            return matches[0]
    known = _known_ref(cache, ref)
    if known and known[1] < _REF_TTL_S:
        return known[0]
    cause = "downloads are off"
    if download:
        try:
            commit = resolve_commit(ref)
        except (urllib.error.URLError, OSError, OpsinBuildError) as exc:
            cause = _explain(exc)
        else:
            refs = cache / _REFS_FILE
            table = _read_json(refs)
            table[ref] = {"commit": commit, "checked": time.time()}
            _write_json(refs, table)
            return commit
    if known:  # the last answer is better than none: a cached build for this ref still works offline
        return known[0]
    raise OpsinBuildError(f"cannot tell which OPSIN commit {ref!r} is ({cause}), and no build of it is known in "
                          f"the jar directory. Connect to GitHub, name a commit that is already built "
                          f"(ORTHONYM_OPSIN=main@<commit>), or unset ORTHONYM_OPSIN.")


@contextlib.contextmanager
def _build_lock(cache: Path):
    """The exclusive lock on building into ``cache``: the file ``.lock`` in the private directory. It is opened
    without following a symlink (mode 0600 when created) and must be a regular file of this user that group
    and others cannot write, as every other file of the cache must; anything else is refused. Only a cache miss
    takes it (a hit needs no lock: a jar is moved into place whole, and its sidecar says it is complete), and no
    other user can hold it, since no other user can open a file in the private directory."""
    import fcntl
    path = cache / ".lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise OpsinBuildError(f"cannot open the build lock {path} ({exc.strerror or exc}): it must be a regular "
                              f"file of yours, not a symlink. Remove it and try again.") from exc
    try:
        st = os.fstat(fd)
        if not (stat.S_ISREG(st.st_mode) and st.st_uid == _uid() and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise OpsinBuildError(f"the build lock {path} is not a regular file of yours that group and others "
                                  f"cannot write. Remove it and try again.")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # releases the lock


def _cached(jar: Path, meta: Path, commit: str) -> bool:
    """The jar and its sidecar in the cache are the build of ``commit``: both are trusted regular files, the
    sidecar names ``commit``, and its ``jar_sha256`` is the of the jar."""
    if not (trusted_file(jar) and trusted_file(meta)):
        return False
    data = _read_json(meta)
    try:
        return data.get("commit") == commit and data.get("jar_sha256") == _sha256(jar)
    except OSError:
        return False


def ensure_jar(ref: str, jar_dir: Path, base_jar: Path, download: bool = True) -> BuiltJar:
    """The jar of OPSIN ``ref`` in the private cache of ``jar_dir``: the cached build when it checks out, else
    (``download``) a new, checked one. A cached jar that fails its checks counts as not built."""
    _uid()  # the first use: a platform without os.getuid is refused here, clearly
    cache = private_dir(jar_dir, create=download)
    commit = _commit_for(ref, cache, download)
    jar = (cache or Path(jar_dir) / PRIVATE_DIR) / f"opsin-main-{commit[:12]}.jar"
    meta = jar.with_suffix(".json")
    if cache is not None and _cached(jar, meta, commit):
        return BuiltJar(jar, commit, ref)
    if not download:
        raise OpsinBuildError(f"OPSIN {commit[:12]} is not built in {jar.parent} (or the build there fails its "
                              f"checks) and downloads are off; build it once with "
                              f"`orthonym --opsin main@{commit[:12]} --fetch-jars`.")
    with _build_lock(cache):  # one process builds; a process that waited finds the jar when it gets the lock
        if _cached(jar, meta, commit):
            return BuiltJar(jar, commit, ref)
        if ref != DEFAULT_BRANCH:
            require_on_main(commit)
        scratch = Path(tempfile.mkdtemp(prefix=".build-", dir=cache))  # mode 0700
        try:
            try:
                built = build_jar(commit, base_jar, scratch / jar.name)
            except (urllib.error.URLError, OSError) as exc:
                raise OpsinBuildError(f"could not fetch the OPSIN sources of {commit[:12]}: {_explain(exc)}") from exc
            os.chmod(built, 0o644)  # not the umask of the user: a jar that group or others can write is refused later
            digest = _sha256(built)  # taken here, in the private scratch directory, before the jar is moved
            self_test(built)
            os.replace(built, jar)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        _write_json(meta, {"ref": ref, "commit": commit, "jar_sha256": digest,
                           "base_jar_sha256": _sha256(base_jar),
                           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "selftest": "pass"})
    return BuiltJar(jar, commit, ref)
