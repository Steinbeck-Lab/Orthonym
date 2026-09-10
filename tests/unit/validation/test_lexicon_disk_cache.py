"""S3 (audit 2026-09-03): the morpheme lexicon is identical for every process of
the same code, yet each process rebuilt it (530 ms; pickle load is 3 ms). The
build is now cached on disk, keyed by the code that produces it.

The cache must never change WHAT the lexicon contains: a hit equals a fresh
build, a corrupt or stale file is ignored, and the cache can be switched off.
"""
import pickle

import pytest

from orthonym.validation import name_morphemes as nm


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORTHONYM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ORTHONYM_LEXICON_CACHE", raising=False)
    nm._lexicon.cache_clear()
    nm._by_first_char.cache_clear()
    yield tmp_path
    nm._lexicon.cache_clear()
    nm._by_first_char.cache_clear()


def test_first_build_writes_one_cache_file(cache_dir):
    lex = nm._lexicon()
    files = list(cache_dir.iterdir())
    assert len(files) == 1 and files[0].suffix == ".pkl", files
    assert len(lex) > 1000


def test_cache_hit_equals_fresh_build_and_skips_the_build(cache_dir, monkeypatch):
    fresh = nm._lexicon()
    nm._lexicon.cache_clear()

    def boom():
        raise AssertionError("_build_lexicon must not run on a cache hit")

    monkeypatch.setattr(nm, "_build_lexicon", boom)
    from_disk = nm._lexicon()
    assert from_disk == fresh
    assert from_disk is not fresh


def test_corrupt_cache_file_is_ignored_and_rebuilt(cache_dir):
    fresh = nm._lexicon()
    (f,) = list(cache_dir.iterdir())
    f.write_bytes(b"not a pickle")
    nm._lexicon.cache_clear()
    assert nm._lexicon() == fresh
    assert pickle.loads(f.read_bytes()) == fresh  # rewritten with a good copy


def test_wrong_shape_in_cache_is_ignored(cache_dir):
    fresh = nm._lexicon()
    (f,) = list(cache_dir.iterdir())
    f.write_bytes(pickle.dumps({"ethan": "not a _Morph"}))
    nm._lexicon.cache_clear()
    assert nm._lexicon() == fresh


def test_cache_key_changes_when_the_source_changes(cache_dir, monkeypatch):
    k1 = nm.lexicon_cache_key()
    monkeypatch.setattr(nm, "_LEXICON_SOURCES", nm._LEXICON_SOURCES + ("_add_nothing",))
    assert nm.lexicon_cache_key() != k1


def test_cache_can_be_switched_off(cache_dir, monkeypatch):
    monkeypatch.setenv("ORTHONYM_LEXICON_CACHE", "off")
    nm._lexicon()
    assert list(cache_dir.iterdir()) == []


def test_unwritable_cache_dir_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ORTHONYM_CACHE_DIR", str(tmp_path / "missing" / "file.txt" / "x"))
    (tmp_path / "missing").write_text("a file, not a directory")
    nm._lexicon.cache_clear()
    assert len(nm._lexicon()) > 1000
    nm._lexicon.cache_clear()
