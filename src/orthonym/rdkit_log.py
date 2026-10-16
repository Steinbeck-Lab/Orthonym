"""Switch RDKit's own log streams off (or on) for a block of code and put them back as they were.

RDKit writes its warnings (``Omitted undefined stereo``, ``Proton(s) added/removed``) and
errors (``Can't kekulize mol``, ``SMILES Parse Error``) to the error stream from C++, outside
Python's ``logging``. ``RDLogger.EnableLog`` afterwards turns a stream ON whatever state it had
before, so a block that changed a stream must restore the previous state, not enable it.
"""
from contextlib import contextmanager
from typing import Iterator

from rdkit import RDLogger, rdBase

_STREAMS = ("rdApp.debug", "rdApp.info", "rdApp.warning", "rdApp.error")


def _enabled() -> set:
    """The RDKit log streams that are on now, from ``rdBase.LogStatus`` lines such as
    ``rdApp.warning:enabled``."""
    return {line.split(":", 1)[0] for line in rdBase.LogStatus().splitlines()
            if line.endswith(":enabled")}


@contextmanager
def _streams_set(streams: tuple, turn_on: bool) -> Iterator[None]:
    named = _STREAMS if "rdApp.*" in streams else streams
    before = _enabled()
    for stream in named:
        (RDLogger.EnableLog if turn_on else RDLogger.DisableLog)(stream)
    try:
        yield
    finally:
        for stream in named:
            (RDLogger.EnableLog if stream in before else RDLogger.DisableLog)(stream)


def rdkit_logs_off(*streams: str):
    """Disable the named RDKit streams (``"rdApp.warning"``, or ``"rdApp.*"`` for all) inside
    the block; on exit each stream is back in the state it had before."""
    return _streams_set(streams, False)


def rdkit_logs_on(*streams: str):
    """Enable the named RDKit streams inside the block (for the one RDKit message that is about
    the user's own input); on exit each stream is back in the state it had before."""
    return _streams_set(streams, True)
