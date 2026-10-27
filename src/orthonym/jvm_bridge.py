"""ONE lazily-started in-process JVM (via JPype) shared by the OPSIN and centres bridges.

WHY
---
Both JVM bridges shelled out ``java -jar...`` **per call**. Measured 2026-07-30 by
counting ``subprocess`` invocations whose argv contains ``java`` over 15 molecules
spanning all naming classes: 65 spawns (4.33/molecule) before any fix, 37 (2.47/molecule)
after the ``_java_available`` probe was cached in ``. Of those 37, **30 came
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
  ``nts.parseChemicalName(name, cfg)`` and writes ``result.getSmiles`` (nothing at all
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
* **centres** — ``com.simolecule.centres.LabelCip`` exposes *only* ``main(String)``
  (verified with ``javap``); there is no programmatic API to call, and reimplementing its
  logic would risk changing labels. So we invoke ``LabelCip.main`` with the identical argv
  and the identical temp file, capturing ``System.out`` into a ``ByteArrayOutputStream``.
  Verified with ``javap -c``: ``LabelCip`` contains **zero ``System.exit`` calls**, so
  calling ``main`` in-process cannot terminate the interpreter.

Rather than re-parse anything, the OPSIN entry point returns the exact bytes the CLI would
have written to stdout, so each caller keeps its own existing output handling unchanged and
merely receives it from a cheaper source.

Validation (a temp dir harnesses, denominators asserted): 217 distinct real names x both
shipped configs = 434 pairs, **0 mismatches** vs the real CLI; centres 508 SMILES as one
batch plus 60 single calls, **0 mismatches**.

SAFETY
------
* **Lazy.** Importing ``orthonym`` starts no JVM and touches no network. The JVM boots on
  first actual use. Jars are resolved by the existing ``_find_opsin_jar`` /
  ``_find_centres_jar`` helpers through ``orthonym.jars.find_jar``, which uses a local jar
  and downloads a missing one on first use (OPSIN 2.9.0 — the version ``eval/goals.json``
  records in its baseline provenance — and centres 1.2.1).
* **Fallback, never failure.** Every entry point returns a sentinel meaning "I could not
  do this; use your subprocess path" when jpype is absent, the jars are missing, or the
  JVM will not start. A missing JVM must never hard-fail a name (``centres_bridge``).
* **fork-safe.** A JVM does not survive ``fork``, and ``eval/harness.py`` uses
  ``mp.Pool`` (processes). A child that inherited a parent's JVM would see
  ``isJVMStarted == True`` while the JVM's threads no longer exist — and a JNI call into
  that is liable to crash the worker outright rather than raise something catchable. So we
  record the pid that actually called ``startJVM`` and refuse to touch a JVM started by any
  other pid. In the normal flow this never triggers (laziness means the parent starts no
  JVM and each worker starts its own); it exists so that ordering cannot become a
  segfault. The same dead JVM would hang a forked child in its first garbage collection
  through JPype's ``gc.callbacks`` entry; ``_drop_jpype_gc_callback_in_child`` removes it
  (see there).
* **Heap.** ``-Xmx`` defaults to 512 MB (verified sufficient for OPSIN + CDK), overridable
  via ``ORTHONYM_JVM_XMX``. This matters because the worker model is *processes*: the
  reference snippet's ``-Xmx4096M`` times 12 workers would reserve 48 GB.
* ``ORTHONYM_DISABLE_JPYPE=1`` forces every caller back onto the subprocess path — the
  escape hatch for A/B measurement and for reproducing a subprocess-only result.
"""
from __future__ import annotations

import collections
import contextlib
import gc
import logging
import os
import re
import signal
import threading
from typing import Optional, Sequence, Tuple

from orthonym.jvm_flags import JVM_CORE_HYGIENE_FLAGS, JVM_HYGIENE_FLAGS

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
# True once the after-fork hook below is registered (os.register_at_fork cannot undo it).
_FORK_HOOK_REGISTERED = False

# Live Java handles (populated by _start).
_N2S_CLS = None        # uk.ac.cam.ch.wwmm.opsin.NameToStructure (the class)
_N2S = None            #...and its singleton, built on FIRST OPSIN call (see _n2s)
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


def _drop_jpype_gc_callback_in_child() -> None:
    """Runs in every ``fork`` child of a process that started the JVM.

    JPype's ``startJVM`` adds ``_jpype._collect`` to ``gc.callbacks``; it ends every Python
    collection by calling ``System.gc`` and waiting for the JVM's VM thread. A fork child
    has no JVM (only the forking thread survives), so its first garbage collection would
    wait forever on a thread that does not exist (audit 2026-10-13, defect 2). Dropping the
    callback in the child is correct for the child's own code too: it never has a JVM, and
    ``_ensure_jvm`` refuses a JVM that another pid started.

    JPype's ``atexit`` hook ``_JTerminate`` calls ``_jpype.shutdown`` into the same dead JVM
    when ``jpype.config.onexit`` is true (the default), so a child that ends through
    ``sys.exit`` or by reaching the end of its script hangs too. ``multiprocessing`` children
    end with ``os._exit`` and skip it; a plain ``os.fork`` child does not. The hook reads
    ``onexit`` at exit time, so clearing it here turns the shutdown off in the child only.
    Never raises.
    """
    try:
        import _jpype
        collect = getattr(_jpype, "_collect", None)
        if collect is not None:
            gc.callbacks[:] = [cb for cb in gc.callbacks if cb is not collect]
    except Exception:  # pragma: no cover - a failing hook must not break fork
        pass
    try:
        import jpype.config
        jpype.config.onexit = False
    except Exception:  # pragma: no cover - a failing hook must not break fork
        pass


def _register_fork_hook_once() -> None:
    """Register:func:`_drop_jpype_gc_callback_in_child` (once per process). Caller holds _LOCK."""
    global _FORK_HOOK_REGISTERED
    if _FORK_HOOK_REGISTERED:
        return
    _FORK_HOOK_REGISTERED = True
    if hasattr(os, "register_at_fork"):  # POSIX only; Windows has no fork
        os.register_at_fork(after_in_child=_drop_jpype_gc_callback_in_child)


@contextlib.contextmanager
def _alarm_held():
    """Hold a ``SIGALRM`` that comes in during the block back, and raise it when the block is left.

    ``jpype.startJVM`` is the one call of this module that cannot be interrupted at its end. The alarm of
    a wall-clock limit (``orthonym.wallclock``) that comes in while it is inside the JVM is raised as soon
    as it returns, and that is in JPype's own Python code, which finishes the set-up of its classes: the
    exception tears that set-up, ``_start`` then fails to bind the classes, and the in-process JVM is off
    for the rest of the run, so that every later call spawns ``java`` (measured: a limit of 0.3 s or 0.5 s
    on the first structure of a process). Held back, the alarm is raised when the block is left, with the
    JVM up and its pid recorded, and the next call binds the classes. Every other call into the JVM is
    safe to interrupt: a retry repeats it.

    The handler is the Python one installed for ``SIGALRM`` (none: nothing to hold). A signal mask would not
    do: the kernel may hand the signal to a thread of the JVM, and the Python handler then runs on the
    main thread all the same.

    The alarm is handed to the handler as the signal gave it (the frame that was running when it came in).
    An exception that ends the block (a Ctrl-C, ``SystemExit``) wins over the held alarm, which is dropped:
    the limit raises its alarm again at its refire interval, and the interrupt must not be replaced by a
    timeout that a batch then carries on from."""
    try:
        handler = signal.getsignal(signal.SIGALRM)
    except (AttributeError, ValueError):    # no SIGALRM on this platform
        handler = None
    held = []
    completed = False
    try:
        if callable(handler) and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGALRM, lambda signum, frame: held.append((signum, frame)))
        else:
            handler = None
        yield
        completed = True
    finally:
        if handler is not None:
            signal.signal(signal.SIGALRM, handler)
            if held and completed:
                handler(*held[0])


def _start(pid: int) -> bool:
    """Resolve jars, boot the JVM, bind the classes. Caller holds _LOCK."""
    global _STARTED_PID, _STATE, _N2S_CLS, _CFG_CLS, _LABELCIP
    global _JSYSTEM, _BAOS, _PRINTSTREAM, _JSTRARR, _OPSIN_JAR, _CENTRES_JAR

    if os.environ.get("ORTHONYM_DISABLE_JPYPE"):
        _log_unavailable_once("ORTHONYM_DISABLE_JPYPE set")
        return False

    try:
        import jpype
    except ImportError:
        _log_unavailable_once("jpype not installed")
        return False

    # Jars come from the EXISTING resolvers (orthonym.jars.find_jar: local, else downloaded once).
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
        # Started by someone else, or inherited across fork -- see "fork-safe".
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
        # Perf lever A2: JVM_HYGIENE_FLAGS now ends with JVM_PERF_FLAGS (SerialGC, C1-only).
        # Both are accepted by every supported JDK, but a start-up failure must never be
        # caused by a performance flag, so the ladder retries without them last.
        base_core = [f"-Xmx{xmx}", *JVM_CORE_HYGIENE_FLAGS]
        # JDK 24+ prints a 4-line "restricted method... System::load" warning to
        # stderr for JPype's own native load. Harmless, but every worker process
        # would emit it, so silence it where supported. The flag is UNKNOWN to
        # JDK < 24 and would abort start-up there, hence the retry without it --
        # a cosmetic flag must never be the reason the JVM fails to come up.
        started = False
        with _alarm_held():
            for opts in ([*base, "--enable-native-access=ALL-UNNAMED"], base,
                         [*base_core, "--enable-native-access=ALL-UNNAMED"], base_core):
                try:
                    jpype.startJVM(*opts, classpath=classpath)
                    started = True
                    break
                except Exception as exc:  # JVMNotFound / OSError / bad option
                    # Deliberately NOT BaseException: a KeyboardInterrupt or SystemExit
                    # during start-up must propagate, not be recorded as "no JVM".
                    last = exc
            if started:
                _STARTED_PID = pid
                _register_fork_hook_once()
            else:
                # Recorded here, inside the block: a held alarm is raised when the block is left, and
                # ``_ensure_jvm`` would not get to record the failed start (it would be tried again by
                # every call) as it does for a start that no alarm met.
                _log_unavailable_once(f"startJVM failed: {last}")
                _STATE = False
        if not started:
            return False
        # The thread that calls startJVM becomes the JVM's creating thread, a
        # USER thread. When that is a worker thread (first use from a
        # per-molecule ThreadPoolExecutor timeout thread), it exits without
        # detaching, and DestroyJavaVM at interpreter exit waits for it forever
        # (see _attach_thread). Re-attach it as a daemon like every other
        # thread. The Python main thread keeps its attachment: it is the thread
        # that runs DestroyJavaVM at exit.
        if threading.current_thread() is not threading.main_thread():
            try:
                jpype.java.lang.Thread.detach()
                jpype.java.lang.Thread.attachAsDaemon()
            except Exception as exc:  # pragma: no cover - never block start-up
                logger.debug("could not re-attach the JVM creating thread as daemon: %s", exc)

    try:
        if opsin_jar:
            # Bind the CLASS only. `getInstance` builds OPSIN's parse automaton
            # (~0.8 s) and is deferred to the first OPSIN call by `_n2s`, so a
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
    """Attach the calling thread to the JVM as a DAEMON if it is not attached yet.

    Daemon, never a user thread. JPype does not detach a Python thread when it
    ends, so a worker thread (e.g. the per-molecule ``ThreadPoolExecutor``
    timeout thread in ``scripts/benchmark_multi_corpus.py``) attached as a USER
    thread stays alive in the JVM's view after it has exited. At interpreter exit
    JPype's atexit hook (``_JTerminate``) destroys the JVM, and ``DestroyJavaVM``
    waits for every user thread to finish, so the process hung forever there.
    Measured: jstack showed the dead workers as non-daemon ``Thread-1..4`` with
    ``DestroyJavaVM`` waiting on them. JPype's own automatic attach is daemon
    (``java.lang.Thread.attachAsDaemon`` docstring), so this matches it.
    """
    try:
        import jpype
        if not jpype.java.lang.Thread.isAttached():
            jpype.java.lang.Thread.attachAsDaemon()
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
    """Memoising front of:func:`_opsin_stdout_uncached` (Lever I, 2026-09-12): within one
    naming scope the same (name, allow_radicals, jar_path) is parsed once; a served result is
    a pure function of the name, so the memo is exact. Unserved results are never cached, and
    nothing is cached outside a scope or in ORTHONYM_MEMO=verify mode."""
    from .assembly.memo import _MODE as _memo_mode, _cache_var, _process_cache, _PROCESS_MAX
    if _memo_mode != "on":
        return _opsin_stdout_uncached(name, allow_radicals, jar_path)
    ck = ("opsin_stdout", (name, allow_radicals, jar_path))
    # Perf lever A8 (2026-09-13): the same candidate names recur ACROSS molecules
    # (26 % of the parses in one scope were already repeats), and a served parse is a
    # pure function of the name, so the entry outlives the scope. Unserved results
    # (the JVM could not answer) are still never cached.
    if _PROCESS_MAX > 0:
        hit = _process_cache.get(ck)
        if hit is not None:
            _process_cache.move_to_end(ck)
            return hit
    cache = _cache_var.get()
    if cache is not None:
        hit = cache.get(ck)
        if hit is not None:
            return hit
    res = _opsin_stdout_uncached(name, allow_radicals, jar_path)
    if res[1]:
        if cache is not None:
            cache[ck] = res
        if _PROCESS_MAX > 0:
            _process_cache[ck] = res
            if len(_process_cache) > _PROCESS_MAX:
                _process_cache.popitem(last=False)
    return res


def _same_jar(jar_path: str) -> bool:
    """True when ``jar_path`` names the jar the JVM was started with, in any spelling
    (relative, symlinked,...). a performance pass R1: a relative spelling used to be
    refused, sending every oracle check to a fresh ``java -jar`` subprocess."""
    if _OPSIN_JAR is None:
        return False
    if jar_path == _OPSIN_JAR:
        return True
    try:
        return os.path.realpath(jar_path) == os.path.realpath(_OPSIN_JAR)
    except (OSError, ValueError):
        return False


def _opsin_stdout_uncached(name: str, allow_radicals: bool,
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
    if jar_path is not None and not _same_jar(jar_path):
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
            if smiles is None:
                _note_refusal(name, allow_radicals, result)
    except Exception as exc:  # a Java-side failure is transient, like a crashed CLI
        logger.debug("in-process OPSIN failed for %r: %s", name, exc)
        return None, False
    # The CLI writes the SMILES only when non-null, then always a newline.
    return ("\n" if smiles is None else str(smiles) + "\n"), True


#: How OPSIN refused a name, by ``(name, allow_radicals)``: ``(kind, locant)`` (:func:`refusal_kind`,
#::func:`refusal_locant`). Filled by:func:`_opsin_stdout_uncached` from the message of the parse it makes,
#: so a caller that asks after a read-back that found no structure costs no second parse.
_REFUSALS: "collections.OrderedDict[tuple, Tuple[str, Optional[str]]]" = collections.OrderedDict()
_REFUSALS_MAX = 20000

#: ``Fragment.getAtomByLocantOrThrow``: "Could not find the atom with locant X." The polycyclic spiro
#: message of ``ComponentProcessor`` starts with the same words and goes on ("... for use in polycyclic
#: spiro system"), which is a refusal of another part of the name.
_LOCANT_REFUSAL_RE = re.compile(r"Could not find the atom with locant (\S+)\.")


def refusal_locant(message: Optional[str]) -> Optional[str]:
    """The locant ``X`` of an OPSIN 2.9.0 refusal "Could not find the atom with locant X." (the first one
    the structure builder did not find, as it spells it, primes and letters included); None for any other
    message."""
    found = _LOCANT_REFUSAL_RE.fullmatch(message or "")
    return found.group(1) if found else None


def refusal_kind(message: Optional[str]) -> str:
    """The kind of an OPSIN 2.9.0 refusal, from the message of ``OpsinResult.getMessage``.

    ``"grammar"``: no tokenisation of the name exists (``Parser.parse``: "<name> is unparsable due to the
    following being uninterpretable:..."); ``"locant"``: the name tokenised and the structure builder asked
    for a locant the fragment does not have (``Fragment.getAtomByLocantOrThrow``: "Could not find the atom
    with locant X.",:func:`refusal_locant`); ``"other"``: any other message (a valency, an assignment of
    double bonds, a polycyclic spiro system's locant, a name that is a substituent,...), which says nothing
    about where the name was refused."""
    text = message or ""
    if " is unparsable due to the following being uninterpretable" in text:
        return "grammar"
    return "locant" if refusal_locant(text) is not None else "other"


def _note_refusal(name: str, allow_radicals: bool, result) -> None:
    """Remember how OPSIN refused ``name``, from its ``OpsinResult`` (never raises: it runs inside the
    parse of a caller)."""
    try:
        raw = result.getMessage()
        message = str(raw) if raw is not None else ""
        _REFUSALS[(name, bool(allow_radicals))] = (refusal_kind(message), refusal_locant(message))
        if len(_REFUSALS) > _REFUSALS_MAX:
            _REFUSALS.popitem(last=False)
    except Exception:  # pragma: no cover - a bookkeeping failure must not cost a parse
        pass


def _read_back_jar_is_the_jvms() -> bool:
    """Is the jar the read-backs use (``validation.opsin_roundtrip.opsin_parse``: ``find_jar("opsin", "2.9.0")``)
    the jar this process's JVM started with? Only then does the in-process parse here say how that read-back
    was refused."""
    try:
        from .jars import find_jar
        jar = find_jar("opsin", "2.9.0")
    except Exception:  # a missing jar: the read-back had none either
        return False
    return bool(jar) and _same_jar(jar)


def opsin_refusal(name: str, allow_radicals: bool = False) -> Optional[Tuple[str, Optional[str]]]:
    """How the in-process OPSIN refused ``name``: ``(kind, locant)`` (:func:`refusal_kind`,
    :func:`refusal_locant`); None when OPSIN reads the name, when the in-process path cannot answer (no
    JVM, an embedded newline), or when the jar of the JVM is not the one the read-backs use. A name that
    :func:`opsin_stdout` parsed already is answered from the record of that parse; otherwise it is parsed
    once here."""
    if not name or "\n" in name or "\r" in name:
        return None
    tab = name.find("\t")
    if tab >= 0:
        name = name[:tab]
    if not _ensure_jvm() or _N2S_CLS is None or not _read_back_jar_is_the_jvms():
        return None
    key = (name, bool(allow_radicals))
    hit = _REFUSALS.get(key)
    if hit is not None:
        return hit
    try:
        with _LOCK:
            _attach_thread()
            result = _n2s().parseChemicalName(name, _cfg(allow_radicals))
            if result.getSmiles() is not None:
                return None
            _note_refusal(name, allow_radicals, result)
    except Exception as exc:  # a Java-side failure is transient
        logger.debug("in-process OPSIN failed for %r: %s", name, exc)
        return None
    return _REFUSALS.get(key)


def opsin_refusal_kind(name: str, allow_radicals: bool = False) -> Optional[str]:
    """The kind of:func:`opsin_refusal`: ``"grammar"``, ``"locant"`` or ``"other"``; None where that is None."""
    refusal = opsin_refusal(name, allow_radicals)
    return refusal[0] if refusal else None


class OpsinUnavailable(RuntimeError):
    """The in-process OPSIN path cannot answer. The command line prints warnings on
    stderr without the name, so a subprocess cannot give a per-name answer."""


def opsin_warnings(name: str, allow_radicals: bool = True) -> Tuple[Tuple[str, str], ...]:
    """OPSIN's warnings for ONE name as ``(type, message)`` pairs, for example
    ``(("APPEARS_AMBIGUOUS", "Positioning of radical on: aziridin"),)``. Empty when OPSIN
    raises no warning or rejects the name. Same parse and configuration as
    :func:`opsin_stdout`; memoised the same way (a pure function of the name).

    Raises:class:`OpsinUnavailable` when the in-process path cannot serve the name (no
    JVM or jar, an embedded newline, a Java-side failure); an unserved answer is never
    cached."""
    from .assembly.memo import _MODE as _memo_mode, _process_cache, _PROCESS_MAX
    ck = ("opsin_warnings", (name, allow_radicals))
    if _memo_mode == "on" and _PROCESS_MAX > 0:
        hit = _process_cache.get(ck)
        if hit is not None:
            _process_cache.move_to_end(ck)
            return hit
    if not name or "\n" in name or "\r" in name:
        raise OpsinUnavailable("name cannot be served in-process")
    if not _ensure_jvm() or _N2S_CLS is None:
        raise OpsinUnavailable("in-process OPSIN not available")
    # Faithful to Cli.interactiveSmilesOutput: parse only up to the first TAB.
    tab = name.find("\t")
    if tab >= 0:
        name = name[:tab]
    try:
        with _LOCK:
            _attach_thread()
            result = _n2s().parseChemicalName(name, _cfg(allow_radicals))
            out = tuple((str(w.getType()), str(w.getMessage())) for w in result.getWarnings())
    except Exception as exc:  # a Java-side failure is transient, like a crashed CLI
        raise OpsinUnavailable(f"in-process OPSIN failed: {exc}") from exc
    if _memo_mode == "on" and _PROCESS_MAX > 0:
        _process_cache[ck] = out
        if len(_process_cache) > _PROCESS_MAX:
            _process_cache.popitem(last=False)
    return out


REJECTED = "__opsin_rejected__"   # sentinel: OPSIN parsed the request and definitively rejected the name (Lever A, 2026-09-12)


def opsin_extended_smiles(name: str,
                          jar_path: Optional[str] = None,
                          allow_radicals: bool = False) -> Tuple[Optional[str], bool]:
    """Exactly what ``java -jar opsin -o extendedsmi`` writes for ONE name.

    Returns ``(extended_smiles, True)`` on success — the ``"<smiles> |$_AV:...$|"``
    line with per-atom locant annotations — ``(REJECTED, True)`` when OPSIN parsed
    the request and definitively rejected the name (the CLI would print an empty
    line, so there is nothing a subprocess could add), or ``(None, False)`` only when
    the in-process path cannot serve this call (jpype/jar absent, wrong jar version,
    embedded newline, or a Java-side error), so the caller MUST fall back to its
    subprocess path. Never raises. Mirrors ``opsin_stdout`` but for
    the extended-SMILES output mode; used by the stereo-locant re-anchor
    (``validation.opsin_roundtrip.opsin_atom_locant_map``). ``allow_radicals`` is OPSIN's ``-r``, for the
    name of a structure that has radical electrons (``validation/read_back_policy.py``)."""
    if not name:
        return None, False
    if "\n" in name or "\r" in name:
        return None, False
    if not _ensure_jvm() or _N2S_CLS is None:
        return None, False
    if jar_path is not None and not _same_jar(jar_path):
        return None, False
    tab = name.find("\t")
    if tab >= 0:
        name = name[:tab]
    try:
        with _LOCK:
            _attach_thread()
            result = _n2s().parseChemicalName(name, _cfg(allow_radicals))
            ext = result.getExtendedSmiles()
    except Exception as exc:  # a Java-side failure is transient, like a crashed CLI
        logger.debug("in-process OPSIN extendedsmi failed for %r: %s", name, exc)
        return None, False
    if ext is None:
        return REJECTED, True  # definitive rejection: served, and there are no locants
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
