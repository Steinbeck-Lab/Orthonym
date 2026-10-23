"""The two Java jars Orthonym uses, and where to find them.

Orthonym does not ship any jar. It uses two, pinned to one version and one
 checksum each, and downloads them from their official releases:

* OPSIN 2.9.0 (name -> structure), for the round-trip check of names.
* centres 1.2.1, for CIP stereo descriptors.

Resolution order for each jar:

1. ``ORTHONYM_OPSIN_JAR`` / ``ORTHONYM_CENTRES_JAR`` -- a path to the jar.
2. The jar directory (``ORTHONYM_JAR_DIR``, else ``<cache>/jars`` where
   ``<cache>`` is ``ORTHONYM_CACHE_DIR``, ``$XDG_CACHE_HOME/orthonym`` or
   ``~/.cache/orthonym``), if the file there has the pinned checksum.
3. A download into the jar directory, unless ``ORTHONYM_NO_DOWNLOAD=1``.

``ORTHONYM_OPSIN=main`` (or ``main@<commit>``) is an opt-in to OPSIN's main branch: instead of
the release jar, step 2 and 3 give a jar built from that commit with the local JDK and cached
in the jar directory (``opsin_build``). Everything that asks for the OPSIN jar gets that one jar,
so one OPSIN serves a whole process. ``ORTHONYM_OPSIN_JAR`` still wins over it, and with the
variable unset or ``2.9.0`` nothing of the opt-in runs.

A jar that cannot be found is an error (:class:`JarUnavailable`): naming
without it would silently change names (no OPSIN check, a different CIP
labeller). ``ORTHONYM_ALLOW_REDUCED=1`` opts in to that reduced mode, in which
the finders return ``None`` as before.

This module uses only the standard library, so the install-time build hook
can load it by file path before Orthonym itself is installed.
"""

import hashlib
import os
import sys
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


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
        uid = os.getuid() if hasattr(os, "getuid") else "user"
        return Path(tempfile.gettempdir()) / f"orthonym-{uid}" / "jars"


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


class _DirLock:
    """Best-effort inter-process lock on the jar directory (POSIX flock).

    Only an optimisation against N workers downloading the same file at once:
    correctness comes from temp-file + checksum + atomic replace.
    """

    def __init__(self, directory: Path):
        self._path = directory / ".lock"
        self._fh = None

    def __enter__(self):
        try:
            import fcntl
            self._fh = open(self._path, "a")
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        except (ImportError, OSError):
            if self._fh is not None:
                self._fh.close()
            self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            self._fh.close()
        return False


def _download(spec: JarSpec, target: Path, verbose: bool = False, first_start: bool = False) -> None:
    """Download ``spec`` to ``target`` (temp file, checksum, atomic replace).

    ``first_start``: the download was not asked for (a namer found the jar missing), so
    say on stderr why this call is slow."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with _DirLock(target.parent):
        if _verified(target, spec):  # another process finished while we waited
            return
        if target.exists():
            raise JarUnavailable(
                f"{target} exists but its SHA-256 does not match the pinned {spec.kind} "
                f"{spec.version} jar; remove it and {FETCH_HINT}")
        if verbose:
            print(f"[orthonym] downloading {spec.kind} {spec.version} from {spec.url}", file=sys.stderr)
        fd, tmp = tempfile.mkstemp(prefix=f".{spec.filename}.", dir=str(target.parent))
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
    release = _resolve_release(spec, download)
    allowed = download and (explicit or not _truthy(os.environ.get("ORTHONYM_NO_DOWNLOAD")))
    try:  # ensure_jar locks (inside its private directory) only to build; the release jar's _DirLock is separate
        built = opsin_build.ensure_jar(ref, jar_dir(), Path(release), allowed)
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
    target = jar_dir() / spec.filename
    if _verified(target, spec):
        return str(target)
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
    return str(target)


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
    (:class:`OpsinChoiceError`) is always raised.

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
        if allow_reduced() and not isinstance(outcome, OpsinChoiceError):
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
        target = Path(override).expanduser() if override else jar_dir() / spec.filename
        if override:
            if not target.is_file():
                raise JarUnavailable(f"{spec.env_var}={override} is not a file")
        elif not _verified(target, spec):
            try:
                _download(spec, target, verbose=verbose)
            except JarUnavailable:
                raise
            except Exception as exc:
                raise JarUnavailable(
                    f"could not download the {kind} {spec.version} jar ({type(exc).__name__}: {exc})") from exc
        paths[kind] = str(target.resolve())
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
