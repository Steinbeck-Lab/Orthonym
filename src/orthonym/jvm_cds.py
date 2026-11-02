"""A class-data archive (AppCDS) for the in-process JVM: a new process starts its JVM about 0.2 s sooner.

The JVM of a new process spends most of its first second loading and checking the classes of OPSIN and centres.
A dynamic class-data archive (JDK 19 and later) holds them ready to map. Measured on JDK 21: start, OPSIN's
set-up and the first parse 0.75 s without the archive, 0.56 s with it (four runs each, no overlap); a one-shot
``orthonym CCO`` takes about 1.9 s.

Where it lives. ``<jar directory>/cds/<key>.jsa``, in the directory of the jars and under the same checks
(``jars.check_jar_dir``): the JVM maps the archive as it loads a jar, so no other user may be able to put one
there. The key is the JVM library, the JDK version, the JPype version and each jar of the class path (path, size,
time), so a new JDK or jar gets an archive of its own.

Who makes it. Only a process that started the JVM itself and asked for it (:func:`build_if_missing`, called at
the end of a command-line run): a child process starts the JVM with ``-XX:ArchiveClassesAtExit`` on a file of its
own, names one structure and exits; the parent moves the finished file into place (``os.replace``, atomic). A
reader therefore never sees a half-written archive, and the workers of ``--jobs`` (whose parent starts no JVM)
only read. The child runs in the archive's directory, so that a module in the user's working directory is never
imported by it. A build that leaves no archive (a JDK without a base archive, a JVM that ignores the option, a
crash) leaves ``<key>.failed`` instead, and no later run tries that key again; the four newest archives are kept.

What a JVM does with a missing, foreign or damaged archive: it starts without it. It would say so on STANDARD
OUTPUT through its unified logging (``[warning][cds] shared class paths mismatch``), which is the class of fault
that shifted the rows of a batch parse (``jvm_flags``); the archive is therefore always passed with
``-Xlog:cds*=off`` and ``-Xlog:class+path*=off``. ``ORTHONYM_JVM_CDS=off`` turns all of it off.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = ["MIN_JDK", "LOG_FLAGS", "archive_path", "build_if_missing", "jdk_major", "start_options"]

#: The first JDK whose dynamic archives this module uses (``-XX:+AutoCreateSharedArchive`` came with 19).
MIN_JDK = 19

#: The JVM's own messages about the archive, which go to standard output, switched off.
LOG_FLAGS = ("-Xlog:cds*=off", "-Xlog:class+path*=off")

#: The child that makes the archive is given this file through the environment.
_DUMP_ENV = "ORTHONYM_JVM_CDS_DUMP"

#: The structure the child names: it loads the classes of OPSIN (the read-back) and of centres (a stereocentre).
_WARM_UP = "C[C@H](N)C(=O)O"


def _enabled() -> bool:
    return os.environ.get("ORTHONYM_JVM_CDS", "on").strip().lower() not in ("off", "0", "false", "no")


def _jvm_library() -> Optional[str]:
    try:
        import jpype
        return jpype.getDefaultJVMPath()
    except Exception:  # noqa: BLE001 - no JVM found: no archive
        return None


def _java_version(jvm_library: Optional[str]) -> Optional[str]:
    """``JAVA_VERSION`` of the JDK that holds ``jvm_library``, read from its ``release`` file (no process is
    started), or None when there is none to read."""
    if not jvm_library:
        return None
    for home in list(Path(jvm_library).resolve().parents)[:4]:
        release = home / "release"
        if release.is_file():
            try:
                text = release.read_text(errors="replace")
            except OSError:
                return None
            m = re.search(r'^JAVA_VERSION="?([^"\n]+)', text, re.M)
            return m.group(1).strip() if m else None
    return None


def jdk_major(jvm_library: Optional[str]) -> Optional[int]:
    """The major version of the JDK that holds ``jvm_library``, or None when it cannot be read."""
    m = re.match(r"(\d+)(?:\.(\d+))?", _java_version(jvm_library) or "")
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2) or 0) if major == 1 else major       # "1.8.0" is Java 8


def archive_path(classpath: Sequence[str]) -> Optional[Path]:
    """The archive for a JVM with ``classpath`` (it may not exist yet), or None when none is used here: turned off,
    a JDK before:data:`MIN_JDK`, or no jar directory that passes the checks."""
    if not _enabled() or not classpath:
        return None
    library = _jvm_library()
    major = jdk_major(library)
    if major is None or major < MIN_JDK:
        return None
    try:
        import jpype

        from .jars import check_jar_dir, jar_dir
        base = check_jar_dir(jar_dir())
        lib = os.stat(library)
        # the whole JDK version and the library itself (an update in place keeps the path), and the heap: a heap
        # above 32 GB makes the JVM take another base archive, against which this one does not load
        parts = [library, str(lib.st_size), str(lib.st_mtime_ns), _java_version(library), jpype.__version__,
                 os.environ.get("ORTHONYM_JVM_XMX", "")]
        for jar in classpath:
            st = os.stat(jar)
            parts += [os.path.realpath(jar), str(st.st_size), str(st.st_mtime_ns)]
    except Exception:  # noqa: BLE001 - a jar directory that fails its checks, a jar gone: no archive
        return None
    key = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]
    return base / "cds" / f"{key}.jsa"


def start_options(classpath: Sequence[str]) -> List[str]:
    """The options that make a JVM with ``classpath`` use its archive ( when there is none to use), or, in the
    child of:func:`build_if_missing`, write it."""
    dump = os.environ.get(_DUMP_ENV)
    if dump:
        return [f"-XX:ArchiveClassesAtExit={dump}", *LOG_FLAGS]
    path = archive_path(classpath)
    if path is None or not path.is_file() or not _trusted(path):
        return []
    return [f"-XX:SharedArchiveFile={path}", *LOG_FLAGS]


def _trusted(path: Path) -> bool:
    """The archive and its directory pass the checks a jar of the jar directory passes: no other user can
    change them."""
    try:
        from .jars import _check_jar_file, check_jar_dir
        check_jar_dir(path.parent)
        return _check_jar_file(path, path.parent) is not None
    except Exception:  # noqa: BLE001 - JarDirUnsafe, or anything that stops the check: do not use it
        return False


def _prune(directory: Path, keep: int = 4) -> None:
    """Keep the ``keep`` newest archives; drop the part files of builds that died (older than an hour)."""
    import time
    try:
        archives = sorted(directory.glob("*.jsa"), key=lambda p: p.stat().st_mtime, reverse=True)
        stale = [p for p in directory.glob("*.part") if time.time() - p.stat().st_mtime > 3600]
        for p in archives[keep:] + stale:
            p.unlink()
    except OSError:
        pass


def build_if_missing(timeout: float = 60.0) -> Optional[Path]:
    """Make the archive of this process's JVM when it has none, and return its path (None when there is nothing to
    make). Only after this process started the JVM itself: a process that named nothing makes nothing."""
    from . import jvm_bridge
    if jvm_bridge._STARTED_PID != os.getpid() or os.environ.get(_DUMP_ENV):
        return None
    classpath = [j for j in (jvm_bridge._OPSIN_JAR, jvm_bridge._CENTRES_JAR) if j]
    path = archive_path(classpath)
    failed = None if path is None else path.with_suffix(".failed")
    if path is None or path.is_file() or failed.exists():
        return None
    try:
        path.parent.mkdir(mode=0o700, exist_ok=True)
        from .jars import check_jar_dir
        check_jar_dir(path.parent)
        if not os.access(path.parent, os.W_OK | os.X_OK):
            return None
    except Exception:  # noqa: BLE001 - no place to put it: start without it
        return None
    part = path.with_name(f"{path.stem}.{os.getpid()}.part")
    env = dict(os.environ, **{_DUMP_ENV: str(part)})
    made = False
    try:
        done = subprocess.run([sys.executable, "-m", "orthonym.jvm_cds"], env=env, cwd=path.parent,
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout)
        if done.returncode == 0 and part.is_file() and part.stat().st_size > 0:
            os.replace(part, path)
            made = True
    except (OSError, subprocess.SubprocessError):
        pass
    except KeyboardInterrupt:     # Ctrl-C while the archive is made: the run's output is out; end without it
        return None
    finally:
        try:
            part.unlink()
        except OSError:
            pass
    if not made:
        try:
            failed.touch()
        except OSError:
            pass
        return None
    _prune(path.parent)
    return path


def _warm_up() -> int:
    """The child of:func:`build_if_missing`: name one structure, so that the JVM loads its classes, and exit
    (the JVM writes the archive as it ends)."""
    from .namer import Orthonym
    Orthonym().name(_WARM_UP)
    return 0


if __name__ == "__main__":
    sys.exit(_warm_up())
