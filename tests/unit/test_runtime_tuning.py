"""Perf lever A1/A11 (2026-09-13): the opt-in batch-worker tuning is idempotent, honours
ORTHONYM_GC_TUNE=off, and only the env-driven path reacts to ORTHONYM_GC_TUNE=on."""
import gc
import importlib
import os

import pytest


@pytest.fixture
def rt(monkeypatch):
    import orthonym.runtime_tuning as m
    saved = gc.get_threshold()
    m._state["tuned"] = False
    yield m
    gc.set_threshold(*saved)
    gc.unfreeze()
    m._state["tuned"] = False


def test_off_env_disables_explicit_call(rt, monkeypatch):
    monkeypatch.setenv("ORTHONYM_GC_TUNE", "off")
    before = gc.get_threshold()
    assert rt.tune_batch_process() is False
    assert gc.get_threshold() == before
    assert rt.is_tuned() is False


def test_explicit_call_sets_threshold_once(rt, monkeypatch):
    monkeypatch.delenv("ORTHONYM_GC_TUNE", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    assert rt.tune_batch_process((12345, 20, 20)) is True
    assert gc.get_threshold() == (12345, 20, 20)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    # second call is a no-op (does not re-apply a different threshold)
    assert rt.tune_batch_process((999, 9, 9)) is True
    assert gc.get_threshold() == (12345, 20, 20)


def test_env_path_only_when_on(rt, monkeypatch):
    monkeypatch.delenv("ORTHONYM_GC_TUNE", raising=False)
    assert rt.maybe_tune_from_env() is False and rt.is_tuned() is False
    monkeypatch.setenv("ORTHONYM_GC_TUNE", "on")
    assert rt.maybe_tune_from_env() is True
    assert gc.get_threshold() == rt.DEFAULT_THRESHOLD
