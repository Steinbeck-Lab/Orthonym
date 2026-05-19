"""Unit tests for ``orthonym.data.ml_model_pin`` per CONTEXT D-05 +
162-AUDIT-MLF.md § 1.4 + § 6.

>=10 tests covering:

* :class:`TestPinConstants` — STOUT_MODEL_VERSION, manifest tuple shape,
  SHA-256 string format
* :class:`TestPinViolationError` — RuntimeError subclass + remediation
  text per RESEARCH § 9.3
* :class:`TestComputeModelSha256` — algorithm correctness + ordering +
  determinism (cross-validates with invoker._compute_model_sha256)
"""

from __future__ import annotations

import pytest

from orthonym.data.ml_model_pin import (
    STOUT_MODEL_FILES_MANIFEST,
    STOUT_MODEL_SHA256,
    STOUT_MODEL_VERSION,
    MLModelPinViolation,
    pin_violation_error,
)
from orthonym.ml_fallback.invoker import _compute_model_sha256


@pytest.mark.unit
class TestPinConstants:
    """STOUT model pin constants per CONTEXT D-05 + audit § 1.4."""

    def test_stout_model_version_is_2_0_5(self):
        assert STOUT_MODEL_VERSION == "2.0.5"

    def test_manifest_is_tuple_of_8_strings(self):
        assert isinstance(STOUT_MODEL_FILES_MANIFEST, tuple)
        assert len(STOUT_MODEL_FILES_MANIFEST) == 8
        for entry in STOUT_MODEL_FILES_MANIFEST:
            assert isinstance(entry, str)

    def test_manifest_contains_expected_files(self):
        expected = {
            "translator_forward/saved_model.pb",
            "translator_forward/variables/variables.data-00000-of-00001",
            "translator_forward/variables/variables.index",
            "translator_reverse/saved_model.pb",
            "translator_reverse/variables/variables.data-00000-of-00001",
            "translator_reverse/variables/variables.index",
            "assets/tokenizer_input.pkl",
            "assets/tokenizer_target.pkl",
        }
        assert set(STOUT_MODEL_FILES_MANIFEST) == expected

    def test_sha256_is_hex_string_or_sentinel(self):
        # Audit § 1.3: empirical 64-char hex OR sentinel
        # "BLOCKED_ON_R02_AWAITING_ZENODO_REPUBLISH"
        assert isinstance(STOUT_MODEL_SHA256, str)
        is_hex = (
            len(STOUT_MODEL_SHA256) == 64
            and all(c in "0123456789abcdef" for c in STOUT_MODEL_SHA256)
        )
        is_sentinel = (
            STOUT_MODEL_SHA256 == "BLOCKED_ON_R02_AWAITING_ZENODO_REPUBLISH"
        )
        assert is_hex or is_sentinel, (
            f"SHA must be 64-char hex OR sentinel; got {STOUT_MODEL_SHA256!r}"
        )


@pytest.mark.unit
class TestPinViolationError:
    """pin_violation_error factory per CONTEXT D-10 tier 3 + audit § 6.5."""

    def test_returns_mlmodel_pin_violation_subclass(self):
        err = pin_violation_error("deadbeef")
        assert isinstance(err, MLModelPinViolation)
        assert isinstance(err, RuntimeError)

    def test_message_contains_observed_sha(self):
        err = pin_violation_error("observed_xxxx")
        assert "observed_xxxx" in str(err)

    def test_message_contains_expected_version(self):
        err = pin_violation_error("xxx")
        assert STOUT_MODEL_VERSION in str(err)

    def test_message_contains_remediation_command(self):
        err = pin_violation_error("xxx")
        assert "pip install --force-reinstall" in str(err)

    def test_message_contains_audit_reference(self):
        # RESEARCH § 9.3: remediation text cites audit doc path
        err = pin_violation_error("xxx")
        assert "162-AUDIT-MLF.md" in str(err)


@pytest.mark.unit
class TestComputeModelSha256:
    """compute_model_sha256 algorithm per CONTEXT D-05 + RESEARCH § 2.1.

    Cross-validates :func:`orthonym.ml_fallback.invoker._compute_model_sha256`
    matches the algorithm specified in audit § 6.1.
    """

    def test_deterministic_on_synthetic_set(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"alpha")
        (tmp_path / "b.bin").write_bytes(b"beta")
        manifest = ("a.bin", "b.bin")
        sha1 = _compute_model_sha256(tmp_path, manifest)
        sha2 = _compute_model_sha256(tmp_path, manifest)
        assert sha1 == sha2

    def test_manifest_order_independent(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"alpha")
        (tmp_path / "b.bin").write_bytes(b"beta")
        sha1 = _compute_model_sha256(tmp_path, ("a.bin", "b.bin"))
        sha2 = _compute_model_sha256(tmp_path, ("b.bin", "a.bin"))
        assert sha1 == sha2

    def test_different_files_different_sha(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"alpha")
        (tmp_path / "b.bin").write_bytes(b"beta")
        sha_both = _compute_model_sha256(tmp_path, ("a.bin", "b.bin"))

        # Mutate one file
        (tmp_path / "a.bin").write_bytes(b"gamma")
        sha_diff = _compute_model_sha256(tmp_path, ("a.bin", "b.bin"))
        assert sha_both != sha_diff

    def test_missing_file_raises_filenotfound(self, tmp_path):
        manifest = ("missing.bin",)
        with pytest.raises(FileNotFoundError):
            _compute_model_sha256(tmp_path, manifest)

    def test_returns_64_char_hex_string(self, tmp_path):
        (tmp_path / "x.bin").write_bytes(b"data")
        sha = _compute_model_sha256(tmp_path, ("x.bin",))
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)
