"""ML Fallback Invoker for Phase 162.

Encapsulates the STOUT 2.0.5 model wrapper per CONTEXT D-01 + D-09 +
D-10 + D-11. Lazy-loads STOUT on FIRST invocation (default-OFF
construction cost unchanged per audit § 10.1). SHA-256 verification
fires BEFORE ``pickle.load()`` per audit § 1.7 security mitigation order.
INFO-level structured logging per audit § 5 / CONTEXT D-11.

Audit reference:
:doc:`162-AUDIT-MLF` § 1.1 (STOUT API), § 1.7 (pickle ordering), § 5
(logging), § 6 (SHA-256 pinning).

Public surface:

* :class:`MLFallbackInvoker` — process-level singleton via
  :func:`functools.cache`; constructed via :meth:`get_instance`
* :class:`MLFallbackResult` — frozen dataclass returned by
  :meth:`MLFallbackInvoker.invoke`

Plan-03 T01 ``namer.py:1248`` wrapper calls ``MLFallbackInvoker.get_instance().invoke(...)``.
Plan-04 T02 ``test_invoker.py`` asserts the get_instance, invoke,
SHA-pin, and 3-tier error policy branches.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from orthonym.data.ml_model_pin import (
    MLModelPinViolation,
    STOUT_MODEL_FILES_MANIFEST,
    STOUT_MODEL_SHA256,
    STOUT_MODEL_VERSION,
    pin_violation_error,
)


__all__ = [
    "MLFallbackInvoker",
    "MLFallbackResult",
]


logger = logging.getLogger("orthonym.ml_fallback")


@dataclass(frozen=True)
class MLFallbackResult:
    """Result of an ML fallback invocation per CONTEXT D-11 + Phase 158 D-03 inheritance.

    Frozen + structurally-trivial per Phase 161 D-04 pattern. The wrapper
    at ``namer.py:1248`` (Plan-03 T01) stores this on
    ``self._last_ml_result`` so ``name_with_confidence`` can surface
    ``ml_fallback_used`` + ``ml_model_version`` in its return dict.
    """

    name: Optional[str]
    """ML-emitted IUPAC name, or ``None`` on inference failure (D-10 tier 2)."""

    model_version: str
    """64-char SHA-256 manifest hash of the STOUT model artifact (MLF-05)."""

    opsin_parse_status: str
    """One of: ``parse_ok`` | ``parse_unsupported`` | ``parse_lossy`` |
    ``parse_fail`` | ``ml_inference_failed``.
    """


def _compute_model_sha256(model_root: Path, manifest: Tuple[str, ...]) -> str:
    """Compute the manifest-tuple SHA-256 per CONTEXT D-05.

    Algorithm matches `` + audit § 6.1
    verbatim. Re-implemented here to keep production code free of any
    `the project tooling` import.

    :param model_root: directory containing the manifest entries
    :param manifest: tuple of relative paths
    :returns: 64-char lowercase hex digest of the manifest concat-hash
    :raises FileNotFoundError: if any manifest entry is missing on disk
    """
    per_file: list[str] = []
    for rel in sorted(manifest):
        path = model_root / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"STOUT manifest file missing: {path}. "
                f"Likely cause: R-02 unresolved (model artifact not yet "
                f"downloaded; see 162-AUDIT-MLF.md § 1.3)."
            )
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        per_file.append(h.hexdigest())
    concat = ":".join(per_file).encode("utf-8")
    return hashlib.sha256(concat).hexdigest()


def _classify_opsin_parse_status(name: Optional[str]) -> str:
    """Map a name through OPSIN-parse to one of the standard status strings.

    Per audit § 5 + Phase 161 D-09 convention. Lazy-imports ``opsin_parse``
    so this module remains independent of ``validation/`` until needed.

    :param name: the ML-emitted IUPAC name (or ``None`` on failure)
    :returns: ``parse_ok`` / ``parse_fail`` / ``parse_unsupported`` /
              ``ml_inference_failed``
    """
    if name is None:
        return "ml_inference_failed"
    try:
        from orthonym.validation.opsin_roundtrip import opsin_parse
        result = opsin_parse(name)
        if result is None:
            return "parse_fail"
        return "parse_ok"
    except Exception:
        # OPSIN itself errored — record as unsupported rather than parse_fail
        # (parse_fail means OPSIN ran but rejected; unsupported means OPSIN
        # could not be invoked at all)
        return "parse_unsupported"


class MLFallbackInvoker:
    """STOUT 2.0.5 model wrapper with SHA-pinning + 3-tier error policy.

    Lazy-loaded via process-level :func:`functools.cache` singleton per
    CONTEXT D-09. First invocation: ~5-15 s (TF init + model load + SHA
    check). Subsequent invocations: ~1-3 s STOUT inference + log overhead.

    Per CONTEXT D-13: NO caching of predictions.

    **Security note (audit § 1.7):** SHA verification fires BEFORE
    ``from STOUT import translate_forward`` to prevent ``pickle.load()``
    RCE on a tampered cache. Plan-04 T02
    ``test_sha_check_runs_before_stout_import`` asserts this ordering.
    """

    __slots__ = ("_translate_forward", "_model_sha256")

    def __init__(self, translate_forward: Callable[..., str], model_sha256: str) -> None:
        """Private constructor — callers use :meth:`get_instance` factory."""
        self._translate_forward = translate_forward
        self._model_sha256 = model_sha256

    @classmethod
    @functools.cache
    def get_instance(cls) -> "MLFallbackInvoker":
        """Process-level singleton factory per CONTEXT D-09.

        :raises RuntimeError: if STOUT not installed (D-10 tier 1; helpful
                              pip install hint) OR if pystow / model
                              artifact missing
        :raises MLModelPinViolation: if SHA-256 manifest hash mismatch
                                      (D-10 tier 3; hard-fail)
        :returns: :class:`MLFallbackInvoker` singleton ready for
                  :meth:`invoke` calls

        Implementation order is LOCKED per audit § 1.7 — SHA verification
        runs BEFORE ``from STOUT import`` so that a tampered
        ``tokenizer_*.pkl`` cannot execute ``pickle.load()`` arbitrary
        code before being detected.
        """
        # STEP 1: Locate model root via pystow (or env-var override)
        env_root = os.environ.get("ORTHONYM_STOUT_MODEL_ROOT")
        if env_root:
            model_root = Path(env_root)
        else:
            try:
                import pystow
            except ImportError as e:
                raise RuntimeError(
                    "ML fallback requires pystow. Install with: "
                    "pip install orthonym[ml]"
                ) from e
            model_root = Path(pystow.join("STOUT-V2", "models"))

        # STEP 2: Compute observed SHA-256 manifest hash
        try:
            observed_sha = _compute_model_sha256(
                model_root, STOUT_MODEL_FILES_MANIFEST,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"STOUT model artifact missing at {model_root}.\n"
                f"  Per Phase 162 162-AUDIT-MLF.md § 1.3 R-02: the upstream "
                f"Zenodo record was taken down 2025-02-05.\n"
                f"  Resolution path (b) requires the user (STOUT "
                f"maintainer) to publish a new Zenodo record.\n"
                f"  Alternative resolution (a) is to vendor the model via "
                f"git-LFS under models/stout-v2-2.0.5/ and set "
                f"ORTHONYM_STOUT_MODEL_ROOT={model_root}.\n"
                f"  Original error: {e}"
            ) from e

        # STEP 3: SHA verification — HARD-FAIL on mismatch (D-10 tier 3)
        # This MUST run BEFORE step 4 (the STOUT import). Per audit § 1.7,
        # `from STOUT import ...` triggers `pickle.load()` on the
        # tokenizer files; a tampered pickle could execute arbitrary code
        # at unpickle time. SHA-check first.
        if observed_sha != STOUT_MODEL_SHA256:
            raise pin_violation_error(observed_sha)

        # STEP 4: Only NOW import STOUT (triggers pickle.load on verified files)
        try:
            from STOUT import translate_forward  # noqa: WPS433  (lazy import)
        except ImportError as e:
            raise RuntimeError(
                "ML fallback requires STOUT. Install with: "
                "pip install orthonym[ml]"
            ) from e

        return cls(translate_forward, observed_sha)

    def invoke(
        self,
        smiles: str,
        *,
        context_class: Optional[str] = None,
    ) -> MLFallbackResult:
        """Run STOUT inference + emit logging contract per CONTEXT D-11.

        :param smiles: SMILES to translate
        :param context_class: optional CFR class that triggered the
                              fallback (for log clarity; defaults to
                              ``"unknown"``)
        :returns: :class:`MLFallbackResult` with name + model_version +
                  opsin_parse_status. On inference failure: returns
                  ``MLFallbackResult(name=None, ..., opsin_parse_status="ml_inference_failed")``
                  per D-10 tier 2.

        Per CONTEXT D-13: NO caching.
        """
        # Trigger line per audit § 5.2 (plain printf-style; D-11 lazy formatting)
        logger.info(
            "ML fallback triggered: smiles=%r context_class=%s model_version=%s",
            smiles,
            context_class or "unknown",
            self._model_sha256[:12],
        )

        # Inference per audit § 1.1 (translate_forward, NOT predict_IUPACname)
        try:
            name = self._translate_forward(smiles)
        except Exception as e:  # noqa: BLE001  (D-10 tier 2: log + return None)
            logger.error(
                "ML fallback inference failed: smiles=%r error=%s",
                smiles,
                f"{type(e).__name__}: {e}",
            )
            return MLFallbackResult(
                name=None,
                model_version=self._model_sha256,
                opsin_parse_status="ml_inference_failed",
            )

        # OPSIN parse status classification per audit § 5
        opsin_status = _classify_opsin_parse_status(name)

        # Result line per audit § 5.2
        logger.info(
            "ML fallback result: name=%r opsin_parse=%s",
            name,
            opsin_status,
        )

        return MLFallbackResult(
            name=name,
            model_version=self._model_sha256,
            opsin_parse_status=opsin_status,
        )
