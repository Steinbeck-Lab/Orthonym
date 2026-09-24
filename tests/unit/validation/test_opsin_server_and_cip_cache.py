"""Gate-performance optimizations: persistent OPSIN JVM + CIP canonical cache.

Both amortize a ~1.7s JVM boot that previously ran once per name/molecule
(~2000 boots on a full gate). These tests lock the correctness contracts that
make the speedups safe:
  * the persistent OPSIN process returns byte-identical (smiles, ran) results
    to a one-shot ``subprocess.run`` for parses, rejections, and edge cases;
  * the centres CIP cache returns identical labels warm vs cold and is keyed by
    the (canonical) SMILES it is fed.

Skipped when Java / the vendored jars are absent (CI without a JVM).
"""
import subprocess

import pytest

from tests.support.jars import jar_or_none

_OPSIN_JAR = jar_or_none()
_HAS_JAVA = True
try:
    subprocess.run(["java", "-version"], capture_output=True, timeout=10)
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    _HAS_JAVA = False

_needs_opsin = pytest.mark.skipif(
    not (_OPSIN_JAR and _HAS_JAVA), reason="OPSIN jar / Java unavailable"
)


def _one_shot(name):
    try:
        r = subprocess.run(
            ["java", "-jar", _OPSIN_JAR, "-r", "-osmi"],
            input=name + "\n", capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return (None, False)
        return (r.stdout.strip() or None, True)
    except Exception:  # noqa: BLE001
        return (None, False)


@_needs_opsin
@pytest.mark.unit
@pytest.mark.parametrize("name", [
    "benzene", "toluene", "pyridine",
    "4H-furo[3,2-b]pyrrole", "1H-1-benzazepine", "pentaphene", "hexahelicene",
    "2-methylnaphthalene", "N'-methylbenzohydrazide", "sodium methoxide",
    "notarealname999zzz", "xyz!!!bad", "quinoline",
])
def test_persistent_opsin_matches_one_shot(name):
    """Persistent OPSIN process == one-shot subprocess, byte-for-byte."""
    from orthonym.validation.opsin_server import get_persistent_opsin
    srv = get_persistent_opsin(_OPSIN_JAR, ("-r", "-osmi"))
    assert srv.invoke(name) == _one_shot(name)


@_needs_opsin
@pytest.mark.unit
def test_persistent_opsin_protocol_guard():
    """A name with an embedded newline fails to (None, False) so the caller
    falls back — never desyncs the one-line-per-input stream."""
    from orthonym.validation.opsin_server import get_persistent_opsin
    srv = get_persistent_opsin(_OPSIN_JAR, ("-r", "-osmi"))
    assert srv.invoke("benzene\ntoluene") == (None, False)
    assert srv.invoke("") == (None, False)
    # still healthy for a normal name afterwards
    assert srv.invoke("benzene") == ("C1=CC=CC=C1", True)


@_needs_opsin
@pytest.mark.unit
def test_cip_cache_warm_equals_cold():
    """centres label map is identical warm (cached) vs cold, and cached by SMILES."""
    from orthonym.perception import centres_bridge as cb
    smis = ["C[C@@H](N)C(=O)O", "O[C@H](C(=O)O)[C@@H](O)C(=O)O"]
    for s in smis:
        cb._CENTRES_LABEL_CACHE.pop(s, None)
    cold = {s: cb.centres_label_batch([s]) for s in smis}
    # unavailable engine -> skip (not the property under test)
    if any(v is None for v in cold.values()):
        pytest.skip("centres engine unavailable")
    warm = {s: cb.centres_label_batch([s]) for s in smis}
    assert cold == warm
    for s in smis:
        assert s in cb._CENTRES_LABEL_CACHE
