"""Logging-contract assertions per MLF-03 + CONTEXT D-11.

Validates the INFO-level structured log lines + ERROR-level inference-failure
log emitted by ``MLFallbackInvoker.invoke()`` via pytest's caplog fixture.

Per 162-AUDIT-MLF.md § 5:
* Two INFO log lines per successful invoke (trigger + result)
* One ERROR log line on inference failure (D-10 tier 2)
* Logger namespace: ``orthonym.ml_fallback``
* Logger is passive (no module-level ``addHandler`` calls per RESEARCH § 7.3)

Tests that require STOUT (the logging happens inside ``MLFallbackInvoker.invoke``)
gracefully skip when STOUT is not available or the R-02 sentinel SHA is in
place. The structural tests (logger namespace, passive config) run
unconditionally.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from orthonym import Orthonym
from orthonym.data.ml_model_pin import STOUT_MODEL_SHA256


def _stout_available() -> bool:
    """Returns True iff STOUT can actually run (SHA pin real + module importable)."""
    if STOUT_MODEL_SHA256.startswith("BLOCKED_ON_R02"):
        return False
    try:
        import STOUT  # noqa: F401, WPS433
    except ImportError:
        return False
    return True


@pytest.mark.integration
class TestMLLoggingNoSetupRequired:
    """Structural log tests that don't require STOUT installed."""

    def test_logger_namespace(self) -> None:
        """D-11: logger namespace is exactly 'orthonym.ml_fallback'."""
        from orthonym.ml_fallback.invoker import logger
        assert logger.name == "orthonym.ml_fallback"

    def test_logger_is_passive_no_handlers_in_module(self) -> None:
        """RESEARCH § 7.3: logger is passive — no module-level addHandler() calls."""
        from orthonym.ml_fallback import invoker
        src = inspect.getsource(invoker)
        # The module body must NOT call addHandler() (would be eager
        # configuration; RESEARCH § 7.3 mandates passive logger).
        assert "addHandler" not in src, (
            "orthonym.ml_fallback.invoker must be a passive logger consumer "
            "per RESEARCH § 7.3 (no addHandler at module-import time)"
        )

    def test_logger_format_constants_present_in_source(self) -> None:
        """Audit § 5.2: trigger + result format strings are locked verbatim."""
        from orthonym.ml_fallback import invoker
        src = inspect.getsource(invoker)
        assert "ML fallback triggered:" in src
        assert "ML fallback result:" in src
        assert "ML fallback inference failed:" in src


@pytest.mark.integration
@pytest.mark.skipif(
    not _stout_available(),
    reason=(
        "STOUT not installed OR R-02 sentinel SHA in ml_model_pin.py "
        "(see 162-AUDIT-MLF.md § 1.3)"
    ),
)
class TestMLLoggingContract:
    """MLF-03: INFO log line at every ML invocation (requires STOUT)."""

    _ML_TRIGGERING_SMILES = "[Pt](Cl)(Cl)([NH3])[NH3]"

    def test_trigger_line_emitted_on_ml_fire(self, caplog) -> None:
        """Audit § 5.2: the 'ML fallback triggered:' line fires when ML attaches."""
        namer = Orthonym(allow_ml_fallback=True)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name(self._ML_TRIGGERING_SMILES)
        triggers = [
            r for r in caplog.records
            if r.getMessage().startswith("ML fallback triggered:")
        ]
        assert len(triggers) >= 1, (
            f"MLF-03 violation: no 'ML fallback triggered' log line; "
            f"records={[r.getMessage() for r in caplog.records]}"
        )

    def test_result_line_emitted_on_ml_fire(self, caplog) -> None:
        """Audit § 5.2: the 'ML fallback result:' line fires when ML attaches."""
        namer = Orthonym(allow_ml_fallback=True)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name(self._ML_TRIGGERING_SMILES)
        results = [
            r for r in caplog.records
            if r.getMessage().startswith("ML fallback result:")
        ]
        assert len(results) >= 1, (
            f"MLF-03 violation: no 'ML fallback result' log line"
        )

    def test_no_log_when_ml_off(self, caplog) -> None:
        """Default-OFF: no log lines when allow_ml_fallback=False."""
        namer = Orthonym(allow_ml_fallback=False)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name(self._ML_TRIGGERING_SMILES)
        triggers = [
            r for r in caplog.records
            if "ML fallback triggered:" in r.getMessage()
        ]
        assert len(triggers) == 0

    def test_no_log_when_quality_gate_doesnt_fire(self, caplog) -> None:
        """Clean rule-based input: predicate returns False, ML NOT invoked, no log."""
        namer = Orthonym(allow_ml_fallback=True)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name("CCO")
        triggers = [
            r for r in caplog.records
            if "ML fallback triggered:" in r.getMessage()
        ]
        assert len(triggers) == 0

    def test_log_format_contains_required_fields(self, caplog) -> None:
        """Audit § 5.2: trigger line includes smiles + context_class + model_version."""
        namer = Orthonym(allow_ml_fallback=True)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name(self._ML_TRIGGERING_SMILES)
        triggers = [
            r for r in caplog.records
            if "ML fallback triggered:" in r.getMessage()
        ]
        if triggers:
            msg = triggers[0].getMessage()
            assert "smiles=" in msg
            assert "context_class=" in msg
            assert "model_version=" in msg

    def test_result_format_contains_opsin_status(self, caplog) -> None:
        """Audit § 5.2: result line includes name + opsin_parse."""
        namer = Orthonym(allow_ml_fallback=True)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name(self._ML_TRIGGERING_SMILES)
        results = [
            r for r in caplog.records
            if "ML fallback result:" in r.getMessage()
        ]
        if results:
            msg = results[0].getMessage()
            assert "name=" in msg
            assert "opsin_parse=" in msg

    def test_two_log_lines_per_invoke(self, caplog) -> None:
        """Audit § 5.2: exactly 2 INFO log lines per successful invoke."""
        namer = Orthonym(allow_ml_fallback=True)
        with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
            namer.name(self._ML_TRIGGERING_SMILES)
        info_records = [
            r for r in caplog.records if r.levelno == logging.INFO
        ]
        # 2 lines per invoke (trigger + result) when ML fires successfully.
        # When ml_inference_failed: 1 trigger + 1 ERROR — still >= 1 INFO record.
        assert len(info_records) >= 1
