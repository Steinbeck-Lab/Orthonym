"""ONE lazily-started in-process JVM (via JPype) shared by the OPSIN and centres bridges.

WHY
---
Both JVM bridges shelled out ``java -jar ...`` **per call**. Measured 2026-07-30 by
counting ``subprocess`` invocations whose argv contains ``java`` over 15 molecules
spanning all naming classes: 65 spawns (4.33/molecule) before any fix, 37 (2.47/molecule)
after the ``_java_available`` probe was cached in ``91a1a51b``. Of those 37, **30 came
from ``centres_label_batch``** — a function named "batch" that production calls twice per
molecule with ONE molecule each, so its batching never amortised anything. Each spawn is
~130 ms of pure process launch.

JPype starts one JVM inside this process and hands back live Java objects, so
``parseChemicalName`` becomes a JNI call rather than a process launch. Measured on the
real single-molecule production pattern (60 calls): **216 ms/call -> 0.8 ms/call**.

BYTE-IDENTITY IS THE CONTRACT
-----------------------------
This is a pure performance change: it must not alter one character of one name. So this
module does **not** reimplement either tool — it drives the *same* code the CLI drives,
and it reproduces the CLI's own input/output framing exactly:

* **OPSIN** — ``opsin/opsin-cli/.../Cli.java:interactiveSmilesOutput`` calls
  ``nts.parseChemicalName(name, cfg)`` and writes ``result.getSmiles()`` (nothing at all
  when it is null) followed by a newline. ``generateOpsinConfigObjectFromCmd`` sets all
  five config flags explicitly from ``cmd.hasOption(...)``, so with no flags every one is
  ``false`` and with ``-r`` only ``allowRadicals`` is ``true``. We set all five explicitly
  rather than trusting ``NameToStructureConfig``'s defaults, because the CLI does.
  Two CLI framing details are reproduced deliberately:
    - the CLI splits each input line at the first TAB and parses only the part before it
      (``line.indexOf('\\t')``), so we truncate identically;
    - the CLI reads **line by line**, so a name containing a newline is two inputs to it
      and one to us. Those are NOT equivalent, so such names are refused here and the
      caller's subprocess path handles them.
* **centres** — ``com.simolecule.centres.LabelCip`` exposes *only* ``main(String[])``
  (verified with ``javap``); there is no programmatic API to call, and reimplementing its
  logic would risk changing labels. So we invoke ``LabelCip.main`` with the identical argv
  and the identical temp file, capturing ``System.out`` into a ``ByteArrayOutputStream``.
  Verified with ``javap -c``: ``LabelCip`` contains **zero ``System.exit`` calls**, so
  calling ``main`` in-process cannot terminate the interpreter.

Rather than re-parse anything, the OPSIN entry point returns the exact bytes the CLI would
have written to stdout, so each caller keeps its own existing output handling unchanged and
merely receives it from a cheaper source.

Validation ( harnesses, denominators asserted): 217 distinct real names x both
shipped configs = 434 pairs, **0 mismatches** vs the real CLI; centres 508 SMILES as one
batch plus 60 single calls, **0 mismatches**.

SAFETY
------
* **Lazy.** Importing ``orthonym`` starts no JVM and touches no network. The JVM boots on
  first actual use. Nothing is ever downloaded: jars are resolved by the existing
  ``_find_opsin_jar`` / ``_find_centres_jar`` helpers, which glob the vendored jars at
  PROJECT_ROOT (OPSIN 2.9.0 — the version `` records in its baseline
  provenance — and centres 1.5).
* **Fallback, never failure.** Every entry point returns a sentinel meaning "I could not
  do this; use your subprocess path" when jpype is absent, the jars are missing, or the
  JVM will not start. A missing JVM must never hard-fail a name (``centres_bridge`` D-13).
* **fork-safe.** A JVM does not survive ``fork()``, and `` uses
  ``mp.Pool`` (processes). A child that inherited a parent's JVM would see
  ``isJVMStarted() == True`` while the JVM's threads no longer exist — and a JNI call into
  that is liable to crash the worker outright rather than raise something catchable. So we
  record the pid that actually called ``startJVM`` and refuse to touch a JVM started by any
  other pid. In the normal flow this never triggers (laziness means the parent starts no
  JVM and each worker starts its own); it exists so that ordering cannot become a
  segfault.
* **Heap.** ``-Xmx`` defaults to 512 MB (verified sufficient for OPSIN + CDK), overridable
  via ``ORTHONYM_JVM_XMX``. This matters because the worker model is *processes*: the
  reference snippet's ``-Xmx4096M`` times 12 workers would reserve 48 GB.
* ``ORTHONYM_DISABLE_JPYPE=1`` forces every caller back onto the subprocess path — the
  escape hatch for A/B measurement and for reproducing a subprocess-only result.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Sequence, Tuple

from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

logger = logging.getLogger(__name__)

# One re-entrant lock guards BOTH the start-up decision and every call into Java.
# It is re-entrant because the centres path holds it while redirecting System.out
# (a JVM-global) and the OPSIN path may be entered from the same thread.
_LOCK = threading.RLock()

# Availability decision, scoped to the pid that made it (see "fork-safe" above).
_STATE: Optional[bool] = None
_STATE_PID: Optional[int] = None
# The pid that actually called startJVM. Only that pid may use the JVM.
_STARTED_PID: Optional[int] = None
_LOGGED_UNAVAILABLE = False

# Live Java handles (populated by _start).
_N2S_CLS = None        # uk.ac.cam.ch.wwmm.opsin.NameToStructure (the class)
_N2S = None            # ...and its singleton, built on FIRST OPSIN call (see _n2s)
_CFG_CLS = None        # uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig
_CFG_BY_RADICALS: dict = {}
_LABELCIP = None       # com.simolecule.centres.LabelCip
_JSYSTEM = None
_BAOS = None
_PRINTSTREAM = None
_JSTRARR = None
_OPSIN_JAR: Optional[str] = None
_CENTRES_JAR: Optional[str] = None

DEFAULT_XMX = "512m"


def _log_unavailable_once(reason: str) -> None:
    """Log the fallback reason exactly once, at DEBUG (non-negotiable #2)."""
    global _LOGGED_UNAVAILABLE
    if not _LOGGED_UNAVAILABLE:
        _LOGGED_UNAVAILABLE = True
        logger.debug("in-process JVM unavailable (%s); using subprocess fallback", reason)


def _reset_handles() -> None:
    global _N2S, _N2S_CLS, _CFG_CLS, _LABELCIP, _JSYSTEM, _BAOS, _PRINTSTREAM, _JSTRARR
    global _OPSIN_JAR, _CENTRES_JAR
    _N2S = _N2S_CLS = _CFG_CLS = _LABELCIP = None
    _JSYSTEM = _BAOS = _PRINTSTREAM = _JSTRARR = None
    _OPSIN_JAR = _CENTRES_JAR = None
    _CFG_BY_RADICALS.clear()


def _start(pid: int) -> bool:
    """Resolve jars, boot the JVM, bind the classes. Caller holds _LOCK."""
    global _STARTED_PID, _N2S_CLS, _CFG_CLS, _LABELCIP
    global _JSYSTEM, _BAOS, _PRINTSTREAM, _JSTRARR, _OPSIN_JAR, _CENTRES_JAR

    if os.environ.get("ORTHONYM_DISABLE_JPYPE"):
        _log_unavailable_once("ORTHONYM_DISABLE_JPYPE set")
        return False

    try:
        import jpype
    except ImportError:
        _log_unavailable_once("jpype not installed")
        return False

    # Jars come from the EXISTING resolvers -- vendored only, never downloaded.
    # Imported lazily so this module can be imported from either bridge without
    # a circular import.
    try:
        from .perception.centres_bridge import _find_centres_jar
        from .validation.opsin_roundtrip import _find_opsin_jar
        opsin_jar = _find_opsin_jar()
        centres_jar = _find_centres_jar()
    except Exception as exc:  # pragma: no cover - import guard
        _log_unavailable_once(f"jar resolution failed: {exc}")
        return False

    classpath = [j for j in (opsin_jar, centres_jar) if j]
    if not classpath:
        _log_unavailable_once("no vendored jar found at PROJECT_ROOT")
        return False

    if jpype.isJVMStarted():
        # Started by someone else, or inherited across fork() -- see "fork-safe".
        # We cannot change a running JVM's classpath and cannot verify a
        # post-fork JVM without risking a crash, so refuse.
        if _STARTED_PID != pid:
            _log_unavailable_once(
                f"JVM already started by pid {_STARTED_PID}, not {pid}")
            return False
    else:
        xmx = os.environ.get("ORTHONYM_JVM_XMX", DEFAULT_XMX)
        # JVM_HYGIENE_FLAGS carries -Djava.awt.headless plus -XX:-UsePerfData.
        # The perf-data flag matters most HERE: this is the highest-churn JVM in
        # the tree (12 pool workers x maxtasksperchild=25 means a fresh JVM every
        # 25 molecules per worker), and a JVM killed by a SIGALRM guard leaks its
        # /tmp/hsperfdata_<user>/<pid> file, which is what lets a later JVM print
        # a warning onto stdout. See orthonym/jvm_flags.py for the mechanism.
        base = [f"-Xmx{xmx}", *JVM_HYGIENE_FLAGS]
        # JDK 24+ prints a 4-line "restricted method ... System::load" warning to
        # stderr for JPype's own native load. Harmless, but every worker process
        # would emit it, so silence it where supported. The flag is UNKNOWN to
        # JDK < 24 and would abort start-up there, hence the retry without it --
        # a cosmetic flag must never be the reason the JVM fails to come up.
        for opts in ([*base, "--enable-native-access=ALL-UNNAMED"], base):
            try:
                jpype.startJVM(*opts, classpath=classpath)
                break
            except Exception as exc:  # JVMNotFound / OSError / bad option
                # Deliberately NOT BaseException: a KeyboardInterrupt or SystemExit
                # during start-up must propagate, not be recorded as "no JVM".
                last = exc
        else:
            _log_unavailable_once(f"startJVM failed: {last}")
            return False
        _STARTED_PID = pid

    try:
        if opsin_jar:
            # Bind the CLASS only. `getInstance()` builds OPSIN's parse automaton
            # (~0.8 s) and is deferred to the first OPSIN call by `_n2s()`, so a
            # molecule that only needs centres never pays for it. This is worth
            # doing because the worker model is processes with
            # maxtasksperchild=25: every worker recycle re-pays whatever start-up
            # happens here.
            _N2S_CLS = jpype.JClass("uk.ac.cam.ch.wwmm.opsin.NameToStructure")
            _CFG_CLS = jpype.JClass("uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig")
            _OPSIN_JAR = opsin_jar
        if centres_jar:
            _LABELCIP = jpype.JClass("com.simolecule.centres.LabelCip")
            _JSYSTEM = jpype.JClass("java.lang.System")
            _BAOS = jpype.JClass("java.io.ByteArrayOutputStream")
            _PRINTSTREAM = jpype.JClass("java.io.PrintStream")
            _JSTRARR = jpype.JArray(jpype.JString)
            _CENTRES_JAR = centres_jar
    except Exception as exc:
        _log_unavailable_once(f"class binding failed: {exc}")
        _reset_handles()
        return False

    return _N2S_CLS is not None or _LABELCIP is not None


def _n2s():
    """The NameToStructure singleton, built on first use. Caller holds _LOCK."""
    global _N2S
    if _N2S is None and _N2S_CLS is not None:
        _N2S = _N2S_CLS.getInstance()
    return _N2S


def _ensure_jvm() -> bool:
    """True if the in-process JVM is usable **from this process**. Never raises."""
    global _STATE, _STATE_PID
    pid = os.getpid()
    with _LOCK:
        if _STATE is not None and _STATE_PID == pid:
            return _STATE
        # First call in this process, or the decision was inherited across a
        # fork -- re-decide from scratch.
        _reset_handles()
        _STATE_PID = pid
        try:
            _STATE = _start(pid)
        except Exception as exc:  # pragma: no cover - belt and braces
            _log_unavailable_once(f"unexpected start failure: {exc}")
            _STATE = False
        return _STATE


def _attach_thread() -> None:
    """Attach the calling thread to the JVM if it is not already (JPype req.)."""
    try:
        import jpype
        if not jpype.java.lang.Thread.isAttached():
            jpype.attachThreadToJVM()
    except Exception:  # pragma: no cover - auto-attach covers modern JPype
        pass


def _cfg(allow_radicals: bool):
    """The NameToStructureConfig the OPSIN CLI would build. Caller holds _LOCK.

    Mirrors ``Cli.generateOpsinConfigObjectFromCmd``, which sets every flag
    explicitly from ``cmd.hasOption(...)``: all false with no flags, and only
    ``allowRadicals`` true for ``-r``. Set explicitly, not left to the class
    defaults, precisely because the CLI sets them explicitly.
    """
    cfg = _CFG_BY_RADICALS.get(allow_radicals)
    if cfg is None:
        cfg = _CFG_CLS()
        cfg.setInterpretAcidsWithoutTheWordAcid(False)
        cfg.setDetailedFailureAnalysis(False)
        cfg.setAllowRadicals(bool(allow_radicals))
        cfg.setWarnRatherThanFailOnUninterpretableStereochemistry(False)
        cfg.setOutputRadicalsAsWildCardAtoms(False)
        _CFG_BY_RADICALS[allow_radicals] = cfg
    return cfg


def opsin_available() -> bool:
    """True if the in-process OPSIN path can serve calls."""
    return _ensure_jvm() and _N2S_CLS is not None


def centres_available() -> bool:
    """True if the in-process centres path can serve calls."""
    return _ensure_jvm() and _LABELCIP is not None


def opsin_stdout(name: str, allow_radicals: bool,
                 jar_path: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """Exactly what ``java -jar opsin [-r] -osmi`` would write to stdout for ONE name.

    Returns ``(stdout_text, True)`` on success — ``"<smiles>\\n"``, or ``"\\n"`` when
    OPSIN definitively rejects the name, matching the CLI byte for byte — or
    ``(None, False)`` when the in-process path cannot serve this call and the caller
    MUST fall back to its subprocess path. Never raises.

    Returning the CLI's raw bytes (rather than a parsed result) is deliberate: each
    caller keeps its own existing stdout handling verbatim and only changes where the
    bytes come from, so no caller's parsing can drift.

    ``jar_path``, when given, must be the jar the JVM was started with; a request for
    any other OPSIN version falls back, since a running JVM's classpath is fixed.
    """
    if not name:
        return None, False
    # The CLI reads line-by-line: an embedded newline is several inputs to it and
    # one to us. Not equivalent -> refuse, let the subprocess path handle it.
    if "\n" in name or "\r" in name:
        return None, False
    if not _ensure_jvm() or _N2S_CLS is None:
        return None, False
    if jar_path is not None and jar_path != _OPSIN_JAR:
        return None, False
    # Faithful to Cli.interactiveSmilesOutput: parse only up to the first TAB.
    tab = name.find("\t")
    if tab >= 0:
        name = name[:tab]
    try:
        with _LOCK:
            _attach_thread()
            result = _n2s().parseChemicalName(name, _cfg(allow_radicals))
            smiles = result.getSmiles()
    except Exception as exc:  # a Java-side failure is transient, like a crashed CLI
        logger.debug("in-process OPSIN failed for %r: %s", name, exc)
        return None, False
    # The CLI writes the SMILES only when non-null, then always a newline.
    return ("\n" if smiles is None else str(smiles) + "\n"), True


def opsin_extended_smiles(name: str,
                          jar_path: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """Exactly what ``java -jar opsin -o extendedsmi`` writes for ONE name.

    Returns ``(extended_smiles, True)`` on success — the ``"<smiles> |$_AV:...$|"``
    line with per-atom locant annotations — or ``(None, False)`` when the in-process
    path cannot serve this call (jpype/jar absent, wrong jar version, embedded
    newline, definitive OPSIN rejection, or a Java-side error), so the caller MUST
    fall back to its subprocess path. Never raises. Mirrors ``opsin_stdout`` but for
    the extended-SMILES output mode; used by the stereo-locant re-anchor
    (``validation.opsin_roundtrip.opsin_atom_locant_map``)."""
    if not name:
        return None, False
    if "\n" in name or "\r" in name:
        return None, False
    if not _ensure_jvm() or _N2S_CLS is None:
        return None, False
    if jar_path is not None and jar_path != _OPSIN_JAR:
        return None, False
    tab = name.find("\t")
    if tab >= 0:
        name = name[:tab]
    try:
        with _LOCK:
            _attach_thread()
            result = _n2s().parseChemicalName(name, _cfg(False))
            ext = result.getExtendedSmiles()
    except Exception as exc:  # a Java-side failure is transient, like a crashed CLI
        logger.debug("in-process OPSIN extendedsmi failed for %r: %s", name, exc)
        return None, False
    if ext is None:
        return None, False  # definitive rejection -> no locants to serve
    return str(ext), True


def centres_stdout(argv: Sequence[str]) -> Optional[str]:
    """Run ``LabelCip.main(argv)`` in-process; return its captured stdout.

    ``argv`` is exactly what would follow ``java -jar centres-cli.jar`` (e.g.
    ``["-i", "smi", "/tmp/xxx.smi"]``), so this drives the identical code path with
    the identical arguments — the output is byte-identical by construction.

    Returns None when the in-process path is unavailable or Java raised, meaning the
    caller must fall back to its subprocess path. Never raises.

    ``System.out`` is a JVM-global, so the redirect is held under ``_LOCK`` and
    restored in a ``finally`` on every path.
    """
    if not _ensure_jvm() or _LABELCIP is None:
        return None
    try:
        with _LOCK:
            _attach_thread()
            saved_out = _JSYSTEM.out
            buf = _BAOS()
            stream = _PRINTSTREAM(buf, True, "UTF-8")
            try:
                _JSYSTEM.setOut(stream)
                _LABELCIP.main(_JSTRARR(list(argv)))
                stream.flush()
                return str(buf.toString("UTF-8"))
            finally:
                _JSYSTEM.setOut(saved_out)
    except Exception as exc:
        logger.debug("in-process centres failed for %r: %s", list(argv), exc)
        return None
