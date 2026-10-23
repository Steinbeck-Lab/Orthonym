"""An independent reader of SMILES for the read-back of lone-pair stereocentres.

``IndepOracle.java`` (this directory) reads a SMILES string with CDK's SmilesParser (the OpenSMILES reading) and
gives it to the InChI library, and reads a name with OPSIN's own InChI output; the keys are compared by the
caller. It shares no code with Orthonym or with RDKit, which reads a lone-pair stereocentre by its position in
the string (``orthonym.perception.lone_pair_reading``). The source is compiled on first use into a private
cache directory of the user keyed by its hash, or into a private temporary directory of the process (``_compiled``:
a class is never run from a directory another user made or can write); the process it runs in stays up for the life
of the Python process.
"""
import atexit
import hashlib
import os
import stat
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

_HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()
_STATE = {"proc": None, "dead": False}


def _classpath() -> Optional[str]:
    from orthonym.jars import find_jar
    try:
        centres, opsin = find_jar("centres"), find_jar("opsin")
    except Exception:
        return None
    return os.pathsep.join(p for p in (centres, opsin) if p) if centres and opsin else None


def _uid() -> int:
    return os.getuid()


def _trusted(path: Path, directory: bool) -> bool:
    """``path`` is a real directory (``directory``) or regular file of this user that no one else can write: no
    symlink (``os.lstat``), owned by ``_uid``, not writable by group or others."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    kind = stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)
    return kind and st.st_uid == _uid() and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _owned_directory(path: Path) -> bool:
    """``path`` is a real directory (no symlink) owned by ``_uid``."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == _uid()


def _trusted_classes(directory: Path) -> bool:
    """``directory`` holds IndepOracle.class and only files this user wrote and alone can write."""
    if not _trusted(directory, True) or not _trusted(directory / "IndepOracle.class", False):
        return False
    try:
        return all(_trusted(entry, False) for entry in directory.iterdir())
    except OSError:
        return False


def _cache_parent() -> Optional[Path]:
    """The per-user cache directory ``<cache>/orthonym-indep-oracle`` (``$XDG_CACHE_HOME`` or ``~/.cache``), made
    private (0700) when it does not exist; None unless it and the cache directory above it are this user's and not
    writable by group or others. A class is run from it, so a directory another user could have made or can write
    (or rename away) is not used."""
    try:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        base.mkdir(parents=True, exist_ok=True)
        directory = base / "orthonym-indep-oracle"
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        return directory if _trusted(base, True) and _trusted(directory, True) else None
    except OSError:
        return None


def _compiled(cp: str) -> Optional[Path]:
    """The directory holding IndepOracle.class compiled from this source against ``cp``, or None (no javac).

    A compiled class is only ever run from a directory this user alone owns and can write. It is kept in the
    private per-user cache (``_cache_parent``) under the hash of the source and the class path, and used again
    only after ``_trusted_classes`` has checked every file of it (owner, no symlink, no group or world write).
    Otherwise it is compiled again into a fresh ``tempfile.mkdtemp`` (0700, a random name nobody can have made
    beforehand) that serves this process alone and is removed at exit; a directory that is in the way and is
    not this user's is left as it is."""
    import shutil
    source = _HERE / "IndepOracle.java"
    digest = hashlib.sha256(source.read_bytes() + cp.encode()).hexdigest()[:16]
    parent = _cache_parent()
    cached = parent / digest if parent is not None else None
    if cached is not None and _trusted_classes(cached):
        return cached
    scratch = Path(tempfile.mkdtemp(prefix="orthonym_indep_oracle_", dir=parent))   # 0700
    try:
        done = subprocess.run(["javac", "-nowarn", "-cp", cp, "-d", str(scratch), str(source)],
                              capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(scratch, ignore_errors=True)
        return None
    if done.returncode != 0 or not (scratch / "IndepOracle.class").exists():
        shutil.rmtree(scratch, ignore_errors=True)
        return None
    try:
        for entry in scratch.iterdir():
            os.chmod(entry, 0o644)              # javac writes with the umask of the user (0664 under 002)
    except OSError:
        shutil.rmtree(scratch, ignore_errors=True)
        return None
    if cached is not None:
        try:
            if _owned_directory(cached):         # an earlier copy of this user's own that did not pass the check
                shutil.rmtree(cached)
            os.replace(scratch, cached)         # fails where a non-empty directory is in the way
            if _trusted_classes(cached):
                return cached
        except OSError:
            pass
    if (scratch.exists() and _trusted_classes(scratch)):
        atexit.register(shutil.rmtree, scratch, True)
        return scratch
    return None


def _start():
    cp = _classpath()
    if cp is None:
        return None
    classes = _compiled(cp)
    if classes is None:
        return None
    try:
        proc = subprocess.Popen(["java", "-XX:-UsePerfData", "-cp", os.pathsep.join([str(classes), cp]),
                                 "IndepOracle"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except OSError:
        return None
    atexit.register(proc.kill)
    return proc


def keys(smiles: str, name: str = "", radicals: bool = False) -> Optional[Tuple[str, str]]:
    """``(StdInChIKey of the SMILES as CDK reads it, OPSIN's own StdInChIKey of the name)``; the first is
    ``ERR:...`` for a structure the bridge cannot hand over, the second is empty without a name and
    ``NOPARSE`` for a name OPSIN does not read (with ``radicals`` OPSIN reads a name with a free valence, as
    for an input with radical electrons). None when the oracle is unavailable (no jar, no javac)."""
    if "\t" in smiles or "\n" in smiles or "\t" in name or "\n" in name:
        return None
    with _LOCK:
        if _STATE["dead"]:
            return None
        if _STATE["proc"] is None:
            _STATE["proc"] = _start()
            if _STATE["proc"] is None:
                _STATE["dead"] = True
                return None
        proc = _STATE["proc"]
        try:
            proc.stdin.write(f"{smiles}\t{name}\t{'r' if radicals else ''}\n")
            proc.stdin.flush()
            for _ in range(1000):            # the JVM may print a line of its own first
                line = proc.stdout.readline()
                if not line:
                    break
                if line.startswith("KEYS\t"):
                    _tag, k1, k2 = line.rstrip("\n").split("\t")
                    return k1, k2
        except (OSError, ValueError):
            pass
        _STATE["dead"] = True
        return None
