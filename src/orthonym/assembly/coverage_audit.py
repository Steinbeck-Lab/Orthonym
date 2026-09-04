"""Producer-agnostic coverage verdict for the ``_finish`` choke.

Two carrier-specific proofs feed ONE verdict:
  * ``GeneralEngineResult`` (has ``.bindings``) -> ``certify_general_result``
    (E1 + binding spine, Java-free atom->token partition).
  * bare ``str`` (PIN handlers) -> reuse the SELF-01 verdict ALREADY computed
    by ``_final_opsin_validity_gate`` for this exact name (the CARRIED RULING
    in ``task-L0-brief.md``: SELF-01 already OPSIN-parses every PIN name, so a
    second OPSIN call per name would ~2x default-path cost). Only when no
    SELF-01 result is available for this name (e.g. a tier/path that skipped
    the gate) does this fall back to a fresh OPSIN re-anchor
    (``validate_atom_coverage``, the (mol, name) InChIKey-skeleton compare).

This is a coverage AUDIT; L0 runs it SHADOW (telemetry only, never changes the
returned name), L1 (a later task) turns it into a veto.

HONEST LIMIT: the bare-str path depends on OPSIN (not Java-free); it is the
de-facto PIN coverage proof SELF-01 already applies. A Java-free PIN
certificate is out of scope (phase4b Gap-1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageVerdict:
    """The producer-agnostic coverage verdict for one emitted name.

    Attributes:
        complete: True iff the winner's proof shows it accounts for every
            heavy atom (and, on the GER path, bond/charge) of the input.
        method: which proof produced this verdict -- one of
            ``"e1_spine"`` (GeneralEngineResult, certify_general_result),
            ``"self01"`` (bare str, reused SELF-01 verdict -- no extra OPSIN
            call), ``"reanchor"`` (bare str, a fresh OPSIN re-anchor because
            no SELF-01 verdict was available), ``"unavailable"`` (no proof
            could be made -- OPSIN absent/erroring; fails OPEN in SHADOW).
        detail: free-text diagnostic, empty on a clean pass.
    """
    complete: bool
    method: str  # "e1_spine" | "self01" | "reanchor" | "unavailable"
    detail: str = ""


def audit_coverage(mol, name: str, result_obj,
                    self01_complete: Optional[bool] = None,
                    skip_reanchor: bool = False,
                    skip_detail: str = "") -> CoverageVerdict:
    """Return the coverage verdict for the winning ``name`` of ``mol``.

    Args:
        mol: RDKit Mol for the input structure.
        name: the name string about to ship (or that shipped).
        result_obj: the ``GeneralEngineResult`` that produced ``name``, if the
            winner came from the general engine (has a ``.bindings``
            attribute); ``None`` for every bare-str (PIN/T4/etc.) winner.
        self01_complete: the SELF-01 verdict ALREADY computed for this exact
            ``name`` by ``_final_opsin_validity_gate`` (True = OPSIN re-parsed
            it to the same constitution, False = a verified mismatch), or
            ``None`` when no such verdict exists for this name (a path that
            skipped the gate, or the gate's outcome belonged to a different
            string -- see ``metrics.provenance.resolve_gate_outcome``).
            IGNORED when ``result_obj`` is not None (the GER path never needs
            it -- E1 is Java-free and strictly more informative: it covers
            bonds/charge, not merely constitution).
        skip_reanchor:. When ``True`` (and
            ``self01_complete`` is ``None``), skip the ``validate_atom_coverage``
            re-anchor ENTIRELY -- no OPSIN subprocess is spawned -- and return
            ``complete=True, method="unavailable"`` directly. Set by the caller
            when a fresh re-anchor is GUARANTEED useless: a carve-out PIN name
            is OPSIN-unparseable BY DESIGN (thioperoxol/inositol/np_stereoparent/
            dianhydride/.../halogen_uide), and a gate outcome of
            disabled/unavailable/not_run means no real gate decision exists to
            reuse or repeat. Ignored when ``self01_complete`` is not ``None``
            (a real verdict always wins) and ignored on the GER path.
        skip_detail: the :class:`CoverageVerdict` ``detail`` to use when
            ``skip_reanchor`` fires; defaults to a generic message.

    Returns:
        A :class:`CoverageVerdict`. Fail-closed (``complete=False``) on an
        exception in the GER path; fail-OPEN (``complete=True,
        method="unavailable"``) on an exception, OPSIN-unavailable, or
        ``skip_reanchor`` in the bare-str path -- SHADOW must never break a
        name, and must never spawn a guaranteed-useless OPSIN subprocess.
    """
    if result_obj is not None and hasattr(result_obj, "bindings"):
        try:
            from orthonym.validation.coverage_gate import certify_general_result
            ok = certify_general_result(mol, result_obj, allow_charged=True)
        except Exception as exc:  # pragma: no cover - defensive, audit-only
            logger.info("audit_coverage e1_spine raised: %s", exc)
            return CoverageVerdict(False, "e1_spine", f"certify raised: {exc}")
        return CoverageVerdict(bool(ok), "e1_spine", "" if ok else "certify failed")

    # bare-str path -- CARRIED RULING: prefer the already-computed SELF-01
    # verdict over a second OPSIN call.
    if self01_complete is not None:
        return CoverageVerdict(
            bool(self01_complete), "self01",
            "" if self01_complete else "self01 mismatch")

    # C1/C2 fix: some resolved gate outcomes make a fresh re-anchor guaranteed
    # useless (carve-out PIN families are OPSIN-unparseable BY DESIGN; a
    # disabled/unavailable/not-run gate means no real decision exists at all).
    # Do NOT import or call `validate_atom_coverage` in that case -- no OPSIN
    # subprocess spawn, ever, for these.
    if skip_reanchor:
        return CoverageVerdict(True, "unavailable",
                               skip_detail or "reanchor skipped")

    try:
        from orthonym.validation.atom_coverage import validate_atom_coverage
        cov = validate_atom_coverage(mol, name)
    except Exception as exc:  # pragma: no cover - defensive, audit-only
        logger.info("audit_coverage reanchor raised: %s", exc)
        return CoverageVerdict(True, "unavailable", f"reanchor error: {exc}")
    if getattr(cov, "method", None) == "unavailable":
        return CoverageVerdict(True, "unavailable", "opsin unavailable")
    return CoverageVerdict(
        bool(cov.is_complete), "reanchor",
        "" if cov.is_complete else "constitution mismatch")
