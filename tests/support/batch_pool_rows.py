"""Module-level row functions for ``orthonym.batch_pool`` tests: spawn workers import them by name."""
import os

ERROR_LINE = "ERROR"


def noop_init() -> None:
    pass


def upper(item: str) -> str:
    return item.upper()


def exit_on_marker(item: str) -> str:
    """The row of ``item``; the worker dies without a word on the marker 'EXIT' (a crash mid-row)."""
    if item == "EXIT":
        os._exit(1)
    return item


def error_line(item: str) -> str:
    return ERROR_LINE


def report_pid_and_sleep(item: str) -> str:
    """Write this worker's process id to the file ``item``, then sleep a minute (a row that outlives its parent)."""
    import time
    with open(item, "w") as fh:
        fh.write(str(os.getpid()))
    time.sleep(60)
    return item
