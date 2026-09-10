"""R4 (audit 2026-09-03): the tree has 592 broad ``except`` blocks; the two at
the top of ``name`` / ``name_tiered`` turn ANY escaped exception into an
abstention. ``raise_on_limit=True`` already re-raises there, but batch
harnesses cannot pass it. ``ORTHONYM_STRICT=1`` is the environment switch for
the same behaviour, so a run can be made to fail loudly on programming errors.
Default behaviour is unchanged.
"""
import pytest

from orthonym import Orthonym
from orthonym import namer as namer_mod
from orthonym.errors import is_failure_name


@pytest.fixture
def boom(monkeypatch):
    def _boom(self, smiles):
        raise RuntimeError("injected programming error")
    monkeypatch.setattr(namer_mod.Orthonym, "_name_impl", _boom)


def test_default_swallows_into_an_abstention(boom, monkeypatch):
    monkeypatch.delenv("ORTHONYM_STRICT", raising=False)
    eng = Orthonym(style="pin")
    assert is_failure_name(eng.name("CCO"))
    assert eng.name_tiered("CCO")["tier"] == "abstain"


def test_strict_env_reraises_from_name(boom, monkeypatch):
    monkeypatch.setenv("ORTHONYM_STRICT", "1")
    with pytest.raises(RuntimeError, match="injected"):
        Orthonym(style="pin").name("CCO")


def test_strict_env_reraises_from_name_tiered(boom, monkeypatch):
    monkeypatch.setenv("ORTHONYM_STRICT", "1")
    with pytest.raises(RuntimeError, match="injected"):
        Orthonym(style="pin").name_tiered("CCO")


def test_strict_off_values_are_not_strict(boom, monkeypatch):
    monkeypatch.setenv("ORTHONYM_STRICT", "0")
    assert is_failure_name(Orthonym(style="pin").name("CCO"))
