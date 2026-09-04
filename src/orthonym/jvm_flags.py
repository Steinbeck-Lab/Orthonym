"""One place that decides how a JVM is launched. Import this, never hand-roll argv.

Every ``java`` invocation in this tree -- OPSIN CLI, the centres CIP jar, the
persistent OPSIN child, and the in-process JPype JVM -- goes through
:data:`JVM_HYGIENE_FLAGS` so a single edit reaches all of them.

Why this exists: the hsperfdata stdout-contamination class
----------------------------------------------------------
A JVM writes a perf-data file at ``/tmp/hsperfdata_<user>/<pid>``. When it cannot
use that path it prints, **to stdout**::

    [0.003s][warning][perf,memops] Cannot use file /tmp/hsperfdata_kohulan/3428897
    because it is locked by another process (errno = 11)

Anything mapping OPSIN output positionally onto its inputs then shifts every
later row by one, so correct names silently acquire the *previous* name's SMILES.
Measured once at 12 concurrent JVMs: 34 of 400 results shifted. That is a wrong
molecule, not a parse failure -- the single worst failure mode this project has.

Mechanism, verified in all four legs 2026-08-03:

===========================  ==========================================
a live JVM                   creates ``/tmp/hsperfdata_<user>/<pid>``
**SIGKILL**                  **the file LEAKS (survives the process)**
a clean exit                 removes it
``-XX:-UsePerfData``         **never creates it at all**
===========================  ==========================================

This project kills JVMs routinely -- SIGALRM guards in ``pin_conformance_eval``,
``measure_breadth`` slice timeouts, hang-recovery ``pkill`` -- so it manufactures
the stale files that make the collision possible. Disabling perf data removes the
cause rather than coping with the symptom. It is also rare (``pid_max`` is
4,194,304, so PID reuse is uncommon), which makes it the dangerous kind:
intermittent, silent wrongness that a green test run cannot see.

Cost: ``jps``/``jstat``/``jcmd`` can no longer discover these JVMs. Nothing in
this tree uses them; ``orthonym.jvm_budget.status()`` is the supported way to
see what is running.

Belt and braces
---------------
Removing the cause is not sufficient on its own -- a JVM can print to stdout for
other reasons (a GC log, an agent, a future JDK warning). So every positional
batch parse must ALSO fail closed on a line-count mismatch, the way
``eval/harness.py:opsin_batch`` already does. Use :func:`aligned_batch_lines`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

#: Flags applied to every JVM this project launches.
#:
#: ``-XX:-UsePerfData``       no /tmp/hsperfdata file -> the contamination class
#:                            above becomes structurally impossible.
#: ``-Djava.awt.headless``    no display is ever available; avoids an AWT probe.
JVM_HYGIENE_FLAGS: List[str] = [
    "-XX:-UsePerfData",
    "-Djava.awt.headless=true",
]


def java_cmd(*args: str, heap: Optional[str] = None) -> List[str]:
    """Build a ``java`` argv with the hygiene flags applied.

    >>> java_cmd("-jar", "opsin.jar", "-osmi")[:3]
    ['java', '-XX:-UsePerfData', '-Djava.awt.headless=true']

    Args:
        *args: everything after the JVM options -- typically
            ``"-jar", jar_path, ...`` or ``"-version"``.
        heap: optional ``-Xmx`` value, e.g. ``"512m"``. Worth passing on any
            path that may run many JVMs at once: the ~40 ``java -jar`` sites
            historically carried NO heap flag, so each inherited the JDK
            ergonomic default of 25% of RAM (15.8 GB on this host) as *reserved*
            address space, while the JPype path was capped at 512m. Actual
            measured footprint is 196-395 MB, so a cap costs nothing.
    """
    cmd = ["java", *JVM_HYGIENE_FLAGS]
    if heap:
        cmd.append(f"-Xmx{heap}")
    cmd.extend(args)
    return cmd


def aligned_batch_lines(stdout: str, expected: int,
                        where: str = "opsin batch") -> Optional[List[str]]:
    """Split ``stdout`` into exactly ``expected`` lines, or return ``None``.

    ``None`` means "do not trust this output" -- the caller must treat the whole
    batch as unavailable rather than map it positionally. Returning ``None``
    instead of raising keeps this usable on the paths that must degrade to
    "unknown" rather than abort.

    Any line the JVM injected (a warning, an info line, a banner) changes the
    count, so this catches the contamination class even when its cause is
    something ``-XX:-UsePerfData`` does not cover.

    >>> aligned_batch_lines("C(C)O\\nC1=CC=CC=C1\\n", 2)
    ['C(C)O', 'C1=CC=CC=C1']
    >>> aligned_batch_lines("[warning] oops\\nC(C)O\\n", 1) is None
    True
    """
    lines = (stdout or "").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != expected:
        import logging
        logging.getLogger(__name__).warning(
            "%s returned %d lines for %d inputs; refusing to map them "
            "positionally (a stray JVM line would silently shift every later "
            "row onto the previous input's result)", where, len(lines), expected)
        return None
    return [ln.strip() for ln in lines]


def strip_jvm_noise(lines: Sequence[str]) -> List[str]:
    """Drop lines a JVM prints that are not payload.

    A last-resort helper for callers that cannot fail closed. Prefer
    :func:`aligned_batch_lines` -- dropping lines can only ever guess, whereas a
    count mismatch is proof something is wrong.
    """
    out = []
    for ln in lines:
        s = ln.strip()
        if (s.startswith("[") and "]" in s
                and any(t in s for t in ("[warning]", "[info]", "[error]",
                                         "[debug]", "[trace]"))):
            continue
        if s.startswith("Picked up JAVA_TOOL_OPTIONS") or s.startswith("Picked up _JAVA_OPTIONS"):
            continue
        if s.startswith("Run the jar using the -h flag"):
            continue
        out.append(ln)
    return out
