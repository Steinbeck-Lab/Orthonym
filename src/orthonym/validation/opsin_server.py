"""Persistent OPSIN process — amortize the ~1.7s JVM startup across every call.

The validity gate (namer.py -> OpsinOracle) shells out ``java -jar opsin -r
-osmi`` once per distinct name; on a full gate that is ~2000 fresh JVM boots
(~1.7s each ≈ 50+ min of pure startup). OPSIN's CLI reads names line-by-line
from stdin and emits exactly ONE line per input (the SMILES, or a BLANK line
when it rejects the name) and flushes each line interactively — so a single
long-lived JVM fed over a pipe gives one startup for the whole run.

Contract preserved verbatim for the caller (``OpsinOracle._invoke_opsin``):
``invoke(name) -> (raw_smiles_or_None, ran)``
  * ``(smiles, True)`` — OPSIN parsed the name;
  * ``(None, True)`` — OPSIN ran and DEFINITIVELY rejected it (blank line);
  * ``(None, False)`` — the parse could not be performed (server absent /
    dead / read timeout / protocol anomaly). The caller treats this as
    transient and MUST fall back (to a one-shot ``subprocess.run``), so
    correctness NEVER depends on this optimization — it only removes latency.

Safety: a per-read timeout + auto-restart means a hung/crashed JVM degrades to
``(None, False)`` (caller falls back) rather than blocking the gate. A single
lock serializes access (the pipe is one ordered stream; the gate is
single-threaded per the historical [99Tc] ThreadPoolExecutor-hang lesson).
"""
from __future__ import annotations

import atexit
import logging
import queue
import subprocess
import threading
from typing import Dict, List, Optional, Tuple
from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

logger = logging.getLogger(__name__)

_EOF = object()  # sentinel pushed by the reader when OPSIN's stdout closes


class PersistentOpsin:
    """One long-lived ``java -jar opsin...`` process fed names over stdin."""

    def __init__(self, jar: str, args: Tuple[str, ...] = ("-r", "-osmi"),
                 read_timeout: float = 15.0):
        self._jar = jar
        self._args = tuple(args)
        self._read_timeout = read_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._q: Optional[queue.Queue] = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    # -- lifecycle -----------------------------------------------------------
    def _start(self) -> bool:
        """(Re)start the JVM + a fresh reader thread/queue. Caller holds _lock."""
        self._kill()
        try:
            self._proc = subprocess.Popen(
                ["java", *JVM_HYGIENE_FLAGS, "-jar", self._jar, *self._args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except OSError as exc:  # java missing / fork failure under memory pressure
            logger.debug("PersistentOpsin start failed: %s", exc)
            self._proc = None
            return False
        self._q = queue.Queue()
        proc, q = self._proc, self._q
        t = threading.Thread(target=self._read_loop, args=(proc, q), daemon=True)
        t.start()
        return True

    @staticmethod
    def _read_loop(proc: subprocess.Popen, q: queue.Queue) -> None:
        try:
            for line in proc.stdout:          # one line per input name
                q.put(line.rstrip("\n"))
        except Exception:                      # pragma: no cover - defensive
            pass
        q.put(_EOF)                            # stdout closed -> proc gone

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.terminate()
            except OSError:
                pass
            self._proc = None
        self._q = None

    def close(self) -> None:
        with self._lock:
            self._kill()

    # -- the hot path --------------------------------------------------------
    def invoke(self, name: str) -> Tuple[Optional[str], bool]:
        """Feed one name; return (raw_smiles_or_None, ran). Never raises."""
        # Protocol guard: a name with an embedded newline would desync the
        # one-line-per-input stream. IUPAC names never contain one; fail to the
        # caller's subprocess fallback if somehow present.
        if not name or "\n" in name or "\r" in name:
            return None, False
        with self._lock:
            if not self._alive() and not self._start():
                return None, False
            try:
                self._proc.stdin.write(name + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                logger.debug("PersistentOpsin write failed (%s); restarting", exc)
                self._kill()
                return None, False
            try:
                line = self._q.get(timeout=self._read_timeout)
            except queue.Empty:
                logger.debug("PersistentOpsin read timeout for %r; restarting", name)
                self._kill()
                return None, False
            if line is _EOF:
                self._kill()
                return None, False
            # blank line == OPSIN's clean rejection (ran=True, raw=None), matching
            # the one-shot path's "empty stdout, exit 0" rejection.
            return (line or None), True


# Module-level singletons keyed by (jar, args) so one JVM is shared per config
# across all OpsinOracle instances in a process.
_SERVERS: Dict[Tuple[str, Tuple[str, ...]], PersistentOpsin] = {}
_SERVERS_LOCK = threading.Lock()


def get_persistent_opsin(jar: Optional[str],
                         args: Tuple[str, ...] = ("-r", "-osmi")
                         ) -> Optional[PersistentOpsin]:
    """Shared PersistentOpsin for ``jar``/``args``; None if no jar."""
    if not jar:
        return None
    key = (jar, tuple(args))
    with _SERVERS_LOCK:
        srv = _SERVERS.get(key)
        if srv is None:
            srv = PersistentOpsin(jar, args)
            _SERVERS[key] = srv
        return srv
