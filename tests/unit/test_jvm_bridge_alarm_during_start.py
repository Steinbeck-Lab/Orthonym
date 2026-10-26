"""An alarm that arrives while the JVM is starting does not turn the in-process JVM off.

The alarm of a wall-clock limit (``orthonym.wallclock``) that comes in while ``jpype.startJVM`` is inside
the JVM is raised as soon as the call returns, in JPype's own Python code, which finishes the set-up of its
classes. The exception tore that set-up: ``jvm_bridge._start`` then could not bind the classes, and the
in-process JVM was off for the rest of the process (every later OPSIN and centres call spawned ``java``).
Measured before the fix: a limit of 0.3 s or 0.5 s on the first structure of a process.
``jvm_bridge._alarm_held`` holds the alarm back for the length of the start and raises it when the start is
over, with the JVM up.
"""
import os
import signal
import subprocess
import sys
import textwrap
import threading
import types

import pytest

from orthonym import jvm_bridge

pytestmark = pytest.mark.unit

# The start-up in a process of its own (the JVM of this process may be up already). The alarm is armed by a
# wrapper around ``jpype.startJVM``, so that it comes in while the JVM is starting, not before or after.
SCRIPT = textwrap.dedent("""
    import os, signal
    import jpype
    from orthonym import jvm_bridge
    from orthonym.wallclock import WallClockTimeout, wall_clock_limit

    real_start = jpype.startJVM
    def start_with_an_alarm(*args, **kwargs):
        signal.setitimer(signal.ITIMER_REAL, 0.01)      # arrives 10 ms into the start (it takes ~400 ms)
        return real_start(*args, **kwargs)
    jpype.startJVM = start_with_an_alarm

    outcome = "no timeout"
    try:
        with wall_clock_limit(600):
            jvm_bridge._ensure_jvm()
    except WallClockTimeout:
        outcome = "timeout"
    jpype.startJVM = real_start
    print(outcome, jvm_bridge._STARTED_PID == os.getpid(), jvm_bridge._ensure_jvm(), jvm_bridge.opsin_available())
""")


# Every option of ``startJVM`` fails (no JVM on the host), and the alarm comes in during the first attempt.
FAILING_SCRIPT = textwrap.dedent("""
    import signal, time
    import jpype
    from orthonym import jvm_bridge
    from orthonym.wallclock import WallClockTimeout, wall_clock_limit

    calls = []
    def failing_start(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            signal.setitimer(signal.ITIMER_REAL, 0.01)
            time.sleep(0.1)                 # the alarm comes in here, and is held
        raise OSError("no JVM here")
    jpype.startJVM = failing_start

    outcome = "no timeout"
    try:
        with wall_clock_limit(600):
            jvm_bridge._ensure_jvm()
    except WallClockTimeout:
        outcome = "timeout"
    recorded, attempts = jvm_bridge._STATE, len(calls)
    print(outcome, recorded, jvm_bridge._ensure_jvm(), attempts, len(calls))
""")


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM")
def test_a_failed_start_is_recorded_before_the_held_alarm_is_raised(opsin_available):
    """The alarm is raised, and the failed start is cached as it is without an alarm: the next call does
    not try to start the JVM again (it would, four more attempts, every call, if the state stayed unset)."""
    pytest.importorskip("jpype")
    if not opsin_available:
        pytest.skip("no JVM or OPSIN jar")
    done = subprocess.run([sys.executable, "-c", FAILING_SCRIPT], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "ORTHONYM_JVM_BUDGET": "off"})
    assert done.stdout.split("\n")[-2] == "timeout False False 4 4", done.stdout + done.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM")
def test_an_alarm_during_the_start_of_the_jvm_leaves_the_in_process_jvm_usable(opsin_available):
    pytest.importorskip("jpype")
    if not opsin_available:
        pytest.skip("no JVM or OPSIN jar")
    done = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "ORTHONYM_JVM_BUDGET": "off"})
    assert done.stdout.split("\n")[-2] == "timeout True True True", done.stdout + done.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM")
def test_alarm_held_raises_the_alarm_when_the_block_is_left_and_puts_the_handler_back():
    class Alarm(BaseException):
        pass

    frames = []

    def handler(signum, frame):
        frames.append(frame)
        raise Alarm()

    before = signal.signal(signal.SIGALRM, handler)
    try:
        reached = []
        with pytest.raises(Alarm):
            with jvm_bridge._alarm_held():
                signal.setitimer(signal.ITIMER_REAL, 0.01)
                end = threading.Event()
                end.wait(0.1)                  # the alarm comes in here, and is held
                reached.append("end of the block")
        assert reached == ["end of the block"]
        assert signal.getsignal(signal.SIGALRM) is handler
        # the handler gets the frame the signal came in on, as it does from the signal module
        assert len(frames) == 1 and isinstance(frames[0], types.FrameType)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM")
def test_an_interrupt_that_ends_the_block_wins_over_the_held_alarm():
    """A Ctrl-C that lands while an alarm is held is not replaced by the timeout (a batch carries on from a
    timeout and would not stop): the interrupt propagates, the alarm is dropped, the handler is back."""
    class Alarm(BaseException):
        pass

    def handler(signum, frame):
        raise Alarm()

    before = signal.signal(signal.SIGALRM, handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            with jvm_bridge._alarm_held():
                signal.setitimer(signal.ITIMER_REAL, 0.01)
                threading.Event().wait(0.1)         # the alarm comes in here, and is held
                raise KeyboardInterrupt
        assert signal.getsignal(signal.SIGALRM) is handler
        signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM")
def test_alarm_held_does_nothing_without_a_python_handler_and_off_the_main_thread():
    previous = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    try:
        with jvm_bridge._alarm_held():
            assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGALRM, previous)

    seen = []

    def in_a_thread():
        with jvm_bridge._alarm_held():
            seen.append("body ran")

    worker = threading.Thread(target=in_a_thread)
    worker.start()
    worker.join()
    assert seen == ["body ran"]
