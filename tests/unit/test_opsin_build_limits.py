"""Size limits on what the build downloads and unpacks. Offline: GitHub and the archives are fakes."""
import gzip
import io
import tarfile

import pytest

from orthonym import opsin_build as ob


class FakeResponse:
    def __init__(self, body, declared=None):
        self._io = io.BytesIO(body)
        self.headers = {"Content-Length": str(declared)} if declared is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self._io.read(n)


def test_get_stops_an_answer_that_declares_more_than_the_limit(monkeypatch):
    monkeypatch.setattr(ob.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"x", declared=10_000))
    with pytest.raises(ob.OpsinBuildError, match="more than the 100"):
        ob._get("https://example.org/x", 5, limit=100)


def test_get_stops_an_answer_that_is_longer_than_it_declared(monkeypatch):
    monkeypatch.setattr(ob.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"x" * (3 << 20), declared=5))
    with pytest.raises(ob.OpsinBuildError, match="sent more than"):
        ob._get("https://example.org/x", 5, limit=2 << 20)


def test_get_returns_an_answer_within_the_limit(monkeypatch):
    monkeypatch.setattr(ob.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"abc" * 1000))
    assert ob._get("https://example.org/x", 5, limit=1 << 20) == b"abc" * 1000


def _targz(members, payload=b"x"):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:") as tf:
        for name in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return gzip.compress(raw.getvalue())


def test_a_flood_of_entries_is_stopped_at_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "MAX_ENTRIES", 5)
    with pytest.raises(ob.OpsinBuildError, match="more than 5 entries"):
        ob._extract(_targz([f"root/f{i}.txt" for i in range(6)]), tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*.txt"))
    ob._extract(_targz([f"root/f{i}.txt" for i in range(5)]), tmp_path / "ok")
    assert len(list((tmp_path / "ok").rglob("*.txt"))) == 5


def test_an_archive_of_more_than_the_limit_in_total_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "MAX_UNPACKED_BYTES", 10_000)
    with pytest.raises(ob.OpsinBuildError, match="unpacks to more than 10000 bytes"):
        ob._extract(_targz(["root/a.txt", "root/b.txt"], payload=b"y" * 6000), tmp_path / "out")


def test_a_compressed_bomb_is_stopped_before_it_is_unpacked(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "MAX_UNPACKED_BYTES", 1 << 20)
    bomb = gzip.compress(b"\0" * (8 << 20))
    assert len(bomb) < 100_000
    with pytest.raises(ob.OpsinBuildError, match="unpacks to more than"):
        ob._extract(bomb, tmp_path / "out")
