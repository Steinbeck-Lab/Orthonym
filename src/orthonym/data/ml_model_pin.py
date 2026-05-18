"""STOUT 2.0.5 ML-model artifact reproducibility-pin constants.

Locks the upstream model contract for Phase 162 ML fallback gate:

* ``STOUT_MODEL_VERSION``        — pinned PyPI version of ``STOUT-pypi``
* ``STOUT_MODEL_FILES_MANIFEST`` — 8-tuple of relative file paths inside the
  cached model directory (see :doc:`162-AUDIT-MLF` § 1.4)
* ``STOUT_MODEL_SHA256``         — 64-char hex digest of the manifest-tuple
  concat hash, OR the literal sentinel
  ``"BLOCKED_ON_R02_AWAITING_ZENODO_REPUBLISH"`` while the upstream Zenodo
  record (R-02) is unresolved
* ``MLModelPinViolation``        — ``RuntimeError`` subclass raised by the
  Plan-02 invoker when the cached artifact does not match the pinned hash
* ``pin_violation_error``        — factory returning an ``MLModelPinViolation``
  with observed/expected SHA + remediation text

References:

* CONTEXT D-05 (SHA-256 of file-manifest tuple)
* CONTEXT D-10 tier 3 (pin-mismatch = hard-fail)
* MLF-05 (reproducibility pin)
* RESEARCH § 2.1 (compute_model_sha256 algorithm) and § 9.3 (pin-violation
  template, locked verbatim below)

The sentinel SHA exists because Zenodo record 12542360 returned HTTP 410
GONE on 2025-02-05 (RESEARCH § 2.2). Plan-01 T03 cannot empirically compute
the manifest hash until the R-02 user-action republishes the artifact under
a new Zenodo DOI. Per CONTEXT D-07 audit-first cadence + project-memory
rule 4 (no band-aid relaxation), the sentinel forces a hard-fail at
:class:`MLFallbackInvoker.__init__ <orthonym.ml_fallback.invoker.MLFallbackInvoker>`
until an amendment commit replaces it with the real digest.
"""

from __future__ import annotations

from typing import Tuple

__all__ = [
    "STOUT_MODEL_VERSION",
    "STOUT_MODEL_FILES_MANIFEST",
    "STOUT_MODEL_SHA256",
    "MLModelPinViolation",
    "pin_violation_error",
]


STOUT_MODEL_VERSION: str = "2.0.5"
"""Pinned PyPI version of ``STOUT-pypi`` (see CONTEXT D-01)."""


STOUT_MODEL_FILES_MANIFEST: Tuple[str, ...] = (
    "translator_forward/saved_model.pb",
    "translator_forward/variables/variables.data-00000-of-00001",
    "translator_forward/variables/variables.index",
    "translator_reverse/saved_model.pb",
    "translator_reverse/variables/variables.data-00000-of-00001",
    "translator_reverse/variables/variables.index",
    "assets/tokenizer_input.pkl",
    "assets/tokenizer_target.pkl",
)
"""Frozen 8-file manifest for STOUT V2 model artifact.

Verbatim from 162-AUDIT-MLF.md § 1.4. Adding or removing a file invalidates
the pin via manifest-cardinality mismatch before the SHA compare runs.
"""


STOUT_MODEL_SHA256: str = "BLOCKED_ON_R02_AWAITING_ZENODO_REPUBLISH"
"""SHA-256 manifest hash for the pinned STOUT V2 model artifact.

While R-02 (Zenodo 410 GONE; see RESEARCH § 2.2) is unresolved, this is the
literal sentinel string. Plan-02 invoker compares against this value on
every construction; mismatch (which is guaranteed while the sentinel is in
place) raises :class:`MLModelPinViolation`.

Plan-01 T-final amendment commit replaces this with the real 64-char hex
digest via ``python  --write`` once R-02 is
resolved by a new Zenodo publication.
"""


class MLModelPinViolation(RuntimeError):
    """Raised when the cached STOUT V2 model artifact does not match the
    pinned SHA-256 manifest hash.

    Per CONTEXT D-10 tier 3, this is a HARD-FAIL — reproducibility is
    non-negotiable (MLF-05). The Plan-02 invoker computes the SHA before
    importing STOUT so that a tampered ``tokenizer_*.pkl`` cannot execute
    ``pickle.load()`` arbitrary code before the mismatch is detected
    (see 162-AUDIT-MLF.md § 1.7).
    """


def pin_violation_error(observed_sha: str) -> MLModelPinViolation:
    """Build an :class:`MLModelPinViolation` with observed/expected SHA and
    a concrete remediation procedure.

    Template locked verbatim from RESEARCH § 9.3 (Plan-02 T06 implements
    this signature 1:1).

    :param observed_sha: the 64-char hex digest computed from the cache
    :returns: an :class:`MLModelPinViolation` ready to ``raise``
    """
    return MLModelPinViolation(
        f"STOUT model SHA-256 mismatch (reproducibility violation).\n"
        f"  Observed: {observed_sha}\n"
        f"  Expected: {STOUT_MODEL_SHA256}\n"
        f"\n"
        f"  Remediation:\n"
        f"  1. Run: pip install --force-reinstall STOUT-pypi=={STOUT_MODEL_VERSION}\n"
        f"  2. If the issue persists, the upstream artifact may have been updated.\n"
        f"     Bump STOUT_MODEL_SHA256 in src/orthonym/data/ml_model_pin.py and\n"
        f"     re-run the MLF-04 measurement per Phase 164 audit cadence.\n"
        f"  3. See 162-AUDIT-MLF.md § 6 for the canonical SHA-256 verification\n"
        f"     procedure and audit-amendment commit protocol."
    )
