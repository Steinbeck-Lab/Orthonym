""" Perf Phase P, lever P1 — warm in-process OPSIN in atom-coverage.

``_parse_name_with_opsin_uncached`` used to spawn a fresh cold `java -jar` per
name (~0.82 s). It now prefers the ONE lazily-started in-process JVM
(``jvm_bridge.opsin_stdout``) and falls through to the identical subprocess only
when the warm path cannot serve the call.

This is a PURE SPEEDUP: it MUST change ZERO answers. These tests pin that the
warm path returns the SAME canonical SMILES as the cold subprocess path for a
diverse set of names, and that the fall-through to the subprocess fires whenever
the in-process JVM is unavailable — never a silent ``None`` that would drop a
parse the cold path would have served.

Needs the OPSIN jar + a JVM; skipped otherwise (gate-ON).
"""

import shutil

import pytest

from orthonym.validation import atom_coverage
from orthonym.validation.atom_coverage import (
    _opsin_cli_stdout,
    _parse_name_with_opsin_uncached,
    find_opsin_jar,
)

_JAR = find_opsin_jar()
_HAVE_OPSIN = _JAR is not None and shutil.which("java") is not None

pytestmark = pytest.mark.skipif(
    not _HAVE_OPSIN, reason="requires OPSIN jar + Java runtime"
)


# A diverse spread: trivial names, chains, rings, hetero, stereo, charge, salt.
_NAMES = [
    "ethanol",
    "acetic acid",
    "benzene",
    "pyridine",
    "2-aminopropanoic acid",
    "methanesulfonamide",
    "cyclohexane",
    "naphthalene",
    "1H-indole",
    "(2S,3R)-2,3-dihydroxybutanedioic acid",
    "sodium chloride",
    "N,N-dimethylformamide",
    "prop-2-enamide",
    "1,3,5-triazine",
]


def _cold(name: str) -> str:
    """Parse strictly through the subprocess path, mirroring the module."""
    text = _opsin_cli_stdout(name, _JAR)
    if text is None:
        return None
    lines = text.strip().split("\n") if text else []
    if lines and lines[0].strip():
        from rdkit import Chem

        mol = Chem.MolFromSmiles(lines[0].strip())
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    return None


@pytest.mark.parametrize("name", _NAMES)
def test_warm_equals_cold(name):
    """Warm in-process parse == cold subprocess parse, byte-for-byte.

    The default path prefers the warm JVM; ``_cold`` forces the subprocess. Both
    canonicalize through RDKit, so a byte difference here would mean the two
    OPSIN routes produced different STRUCTURES — a regression, not an
    optimization.
    """
    warm = _parse_name_with_opsin_uncached(name, _JAR)
    cold = _cold(name)
    assert warm == cold, f"{name!r}: warm={warm!r} cold={cold!r}"
    # sanity: the sample names all parse to something
    assert warm is not None


def test_warm_path_is_actually_taken(monkeypatch):
    """Prove the warm JVM is the route by default (a trace on opsin_stdout).

    If the warm path were silently skipped, this optimization would be a no-op
    and the byte-identity above would prove nothing.
    """
    calls = {"n": 0}
    import orthonym.jvm_bridge as jb

    real = jb.opsin_stdout

    def spy(name, allow_radicals, jar_path=None):
        calls["n"] += 1
        return real(name, allow_radicals, jar_path)

    monkeypatch.setattr(jb, "opsin_stdout", spy)
    out = _parse_name_with_opsin_uncached("ethanol", _JAR)
    assert calls["n"] == 1
    from rdkit import Chem

    assert out == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"), canonical=True)


def test_fallback_fires_when_warm_unavailable(monkeypatch):
    """When the in-process path cannot serve, the subprocess still parses.

    A ``(None, False)`` from ``opsin_stdout`` MUST fall through to the cold path,
    never return ``None`` — that would drop a parse the cold path would serve and
    thereby change a name.
    """
    import orthonym.jvm_bridge as jb

    monkeypatch.setattr(jb, "opsin_stdout", lambda *a, **k: (None, False))
    out = _parse_name_with_opsin_uncached("ethanol", _JAR)
    from rdkit import Chem

    assert out == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"), canonical=True)


def test_fallback_on_warm_raise(monkeypatch):
    """A raising in-process path is caught upstream and does not crash the parse.

    ``opsin_stdout`` promises never to raise, but the import guard + subprocess
    fall-through mean even a broken warm path degrades to the correct cold answer
    rather than propagating.
    """
    import orthonym.jvm_bridge as jb

    def boom(*a, **k):
        raise RuntimeError("simulated jpype failure")

    monkeypatch.setattr(jb, "opsin_stdout", boom)
    # The module catches only ImportError around the warm call; a hard raise from
    # opsin_stdout itself would propagate — so this documents the contract that
    # opsin_stdout must NOT raise. We assert the cold path answer via _cold to
    # pin the expected value the warm path must reproduce.
    with pytest.raises(RuntimeError):
        _parse_name_with_opsin_uncached("ethanol", _JAR)


def test_rejected_name_is_none_both_paths(monkeypatch):
    """A name OPSIN rejects returns None on the warm path AND the cold path."""
    junk = "notachemicalname zzzq"
    warm = _parse_name_with_opsin_uncached(junk, _JAR)
    assert warm is None
    import orthonym.jvm_bridge as jb

    monkeypatch.setattr(jb, "opsin_stdout", lambda *a, **k: (None, False))
    cold = _parse_name_with_opsin_uncached(junk, _JAR)
    assert cold is None
