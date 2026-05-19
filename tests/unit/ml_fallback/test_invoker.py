"""Unit tests for ``orthonym.ml_fallback.invoker`` per 162-AUDIT-MLF.md
sections 1.7 + 5 + 6.

Organized into class-per-concern:

* :class:`TestMLFallbackResultDataclass` — dataclass shape + frozen
* :class:`TestComputeModelSha256` — algorithm + determinism + ordering
* :class:`TestInvoke` — translate_forward + 3-tier error policy
* :class:`TestLoggingContract` — caplog INFO + ERROR per audit § 5
* :class:`TestSecurityMitigationOrder` — SHA-check BEFORE STOUT import
  per audit § 1.7 (pickle.load RCE mitigation)

Uses mocks for STOUT/pystow (TF is heavyweight and may not be installed
in CI; the R-02 sentinel SHA also forces hard-fail in real
``get_instance()`` calls).
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from unittest.mock import MagicMock, patch

import pytest

from orthonym.data.ml_model_pin import (
    STOUT_MODEL_SHA256,
    STOUT_MODEL_VERSION,
    MLModelPinViolation,
    pin_violation_error,
)
from orthonym.ml_fallback.invoker import (
    MLFallbackInvoker,
    MLFallbackResult,
    _classify_opsin_parse_status,
    _compute_model_sha256,
)


@pytest.mark.unit
class TestMLFallbackResultDataclass:
    """MLFallbackResult is frozen + has 3 documented fields."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(MLFallbackResult)

    def test_is_frozen(self):
        result = MLFallbackResult(
            name="ethanol",
            model_version="ab" * 32,
            opsin_parse_status="parse_ok",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.name = "changed"  # type: ignore[misc]

    def test_three_fields_required(self):
        fnames = {f.name for f in dataclasses.fields(MLFallbackResult)}
        assert fnames == {"name", "model_version", "opsin_parse_status"}

    def test_name_optional(self):
        # MLFallbackResult.name is Optional[str] per CONTEXT D-10 tier 2
        r = MLFallbackResult(
            name=None,
            model_version="cd" * 32,
            opsin_parse_status="ml_inference_failed",
        )
        assert r.name is None

    def test_equality(self):
        a = MLFallbackResult(name="x", model_version="y", opsin_parse_status="parse_ok")
        b = MLFallbackResult(name="x", model_version="y", opsin_parse_status="parse_ok")
        assert a == b


@pytest.mark.unit
class TestComputeModelSha256:
    """Manifest-hash algorithm per CONTEXT D-05 + audit § 6.1."""

    def test_deterministic(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"alpha")
        (tmp_path / "b.bin").write_bytes(b"beta")
        manifest = ("a.bin", "b.bin")
        sha1 = _compute_model_sha256(tmp_path, manifest)
        sha2 = _compute_model_sha256(tmp_path, manifest)
        assert sha1 == sha2
        assert len(sha1) == 64

    def test_manifest_order_independent(self, tmp_path):
        # Algorithm sorts manifest internally per audit § 6.1
        (tmp_path / "a.bin").write_bytes(b"alpha")
        (tmp_path / "b.bin").write_bytes(b"beta")
        sha_ab = _compute_model_sha256(tmp_path, ("a.bin", "b.bin"))
        sha_ba = _compute_model_sha256(tmp_path, ("b.bin", "a.bin"))
        assert sha_ab == sha_ba

    def test_content_change_detected(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"alpha")
        (tmp_path / "b.bin").write_bytes(b"beta")
        sha_orig = _compute_model_sha256(tmp_path, ("a.bin", "b.bin"))
        # Mutate one file
        (tmp_path / "a.bin").write_bytes(b"gamma")
        sha_mut = _compute_model_sha256(tmp_path, ("a.bin", "b.bin"))
        assert sha_orig != sha_mut

    def test_missing_file_raises(self, tmp_path):
        manifest = ("a.bin", "missing.bin")
        (tmp_path / "a.bin").write_bytes(b"data")
        with pytest.raises(FileNotFoundError):
            _compute_model_sha256(tmp_path, manifest)

    def test_classify_opsin_parse_status_none(self):
        assert _classify_opsin_parse_status(None) == "ml_inference_failed"


@pytest.mark.unit
class TestInvoke:
    """invoke() runs translate_forward + handles errors per CONTEXT D-10."""

    def test_invoke_returns_result_with_name(self):
        mock_translate = MagicMock(return_value="ethanol")
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        with patch(
            "orthonym.ml_fallback.invoker._classify_opsin_parse_status",
            return_value="parse_ok",
        ):
            result = invoker.invoke("CCO")
        assert result.name == "ethanol"
        assert result.opsin_parse_status == "parse_ok"
        assert result.model_version == "de" * 32

    def test_invoke_inference_exception_returns_none(self):
        mock_translate = MagicMock(side_effect=RuntimeError("TF crashed"))
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        result = invoker.invoke("CCO")
        assert result.name is None
        assert result.opsin_parse_status == "ml_inference_failed"
        assert result.model_version == "de" * 32

    def test_invoke_with_context_class(self):
        mock_translate = MagicMock(return_value="some name")
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        with patch(
            "orthonym.ml_fallback.invoker._classify_opsin_parse_status",
            return_value="parse_ok",
        ):
            result = invoker.invoke("CCO", context_class="general")
        assert result.name == "some name"

    def test_invoke_no_caching(self):
        # CONTEXT D-13: no caching; each invoke calls translate_forward fresh
        mock_translate = MagicMock(return_value="ethanol")
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        with patch(
            "orthonym.ml_fallback.invoker._classify_opsin_parse_status",
            return_value="parse_ok",
        ):
            invoker.invoke("CCO")
            invoker.invoke("CCO")
            invoker.invoke("CCO")
        assert mock_translate.call_count == 3


@pytest.mark.unit
class TestLoggingContract:
    """CONTEXT D-11 + audit § 5: structured INFO + ERROR log lines."""

    def test_trigger_log_emitted(self, caplog):
        mock_translate = MagicMock(return_value="ethanol")
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        with patch(
            "orthonym.ml_fallback.invoker._classify_opsin_parse_status",
            return_value="parse_ok",
        ):
            with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
                invoker.invoke("CCO")
        triggers = [
            r for r in caplog.records
            if "ML fallback triggered:" in r.getMessage()
        ]
        assert len(triggers) == 1

    def test_result_log_emitted(self, caplog):
        mock_translate = MagicMock(return_value="ethanol")
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        with patch(
            "orthonym.ml_fallback.invoker._classify_opsin_parse_status",
            return_value="parse_ok",
        ):
            with caplog.at_level(logging.INFO, logger="orthonym.ml_fallback"):
                invoker.invoke("CCO")
        results = [
            r for r in caplog.records
            if "ML fallback result:" in r.getMessage()
        ]
        assert len(results) == 1

    def test_error_log_on_inference_exception(self, caplog):
        mock_translate = MagicMock(side_effect=RuntimeError("TF crashed"))
        invoker = MLFallbackInvoker(mock_translate, "de" * 32)
        with caplog.at_level(logging.ERROR, logger="orthonym.ml_fallback"):
            invoker.invoke("CCO")
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "ML fallback inference failed" in r.getMessage() for r in errors
        )

    def test_log_format_uses_printf_not_fstring(self):
        # Audit § 5.3 + RESEARCH § 7.5: printf-style lazy formatting
        from orthonym.ml_fallback import invoker as inv_mod
        src = inspect.getsource(inv_mod)
        # f-string in logger call would be `logger.info(f"..."`
        assert "logger.info(f\"" not in src
        assert "logger.info(f'" not in src


@pytest.mark.unit
class TestSecurityMitigationOrder:
    """Audit § 1.7: SHA-check BEFORE ``from STOUT import`` (pickle RCE mitigation)."""

    def test_sha_check_runs_before_stout_import(self):
        """The get_instance() factory MUST call _compute_model_sha256 BEFORE any STOUT import.

        Inspects the CODE body of ``get_instance()`` (skipping the
        docstring, which references ``from STOUT import`` in its
        explanation paragraph) by looking for the assignment expression
        and the runtime import statement.
        """
        src = inspect.getsource(MLFallbackInvoker.get_instance)
        # Locate the assignment that actually performs the SHA computation
        # (the expression appears in the code body, NOT the docstring).
        sha_call_idx = src.find("observed_sha = _compute_model_sha256")
        # Locate the runtime import statement (not the docstring mention,
        # which uses backticks-quoted form).
        stout_import_idx = src.find("from STOUT import translate_forward")
        assert sha_call_idx >= 0, (
            "get_instance() must assign observed_sha = _compute_model_sha256(...)"
        )
        assert stout_import_idx >= 0, (
            "get_instance() must execute `from STOUT import translate_forward`"
        )
        assert sha_call_idx < stout_import_idx, (
            f"SHA-check MUST occur BEFORE `from STOUT import` (audit § 1.7 "
            f"pickle.load RCE mitigation). Observed: sha_idx={sha_call_idx}, "
            f"stout_idx={stout_import_idx}"
        )

    def test_pin_violation_factory_includes_observed_sha(self):
        err = pin_violation_error("wrong_sha_value")
        assert "wrong_sha_value" in str(err)

    def test_pin_violation_includes_remediation(self):
        err = pin_violation_error("any_sha")
        msg = str(err)
        assert "pip install --force-reinstall" in msg
        assert STOUT_MODEL_VERSION in msg

    def test_pin_violation_includes_expected_sha(self):
        err = pin_violation_error("any_sha")
        # Expected SHA is the pinned constant (may be the R-02 sentinel)
        assert STOUT_MODEL_SHA256 in str(err)
