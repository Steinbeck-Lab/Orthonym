"""The emit tiers and the default tier: one home for the table that maps a tier to the engine switches.

``pin`` is the PIN-only tier: a name only when the strict path for the Preferred IUPAC Name built it and OPSIN
read it back (the default tier of releases up to 1.0.x). ``valid``, ``complete``, ``best-effort`` and
``full-coverage`` name more molecules with producers that are not the strict path.

The default of ``name_compound``, of the ``orthonym`` command and of ``--batch`` is the ``best-effort`` tier,
as it runs when it is asked for by name: one pass, the strict PIN path first among its producers and the
other producers after it. Without OPSIN to read the names back (the reduced mode of a host without Java, and
the raw-output switches) the default stays the PIN-only tier, because a wider tier ships only names that read
back. ``--emit-tier pin`` and ``emit_tier="pin"`` give the PIN-only tier.
"""
from typing import Optional

#: the tiers a caller can ask for by name, narrowest first
EMIT_TIERS = ("pin", "valid", "complete", "best-effort", "full-coverage")

#: the tier of the default when OPSIN reads the names back
DEFAULT_TIER = "best-effort"


def check_emit_tier(emit_tier: Optional[str]) -> Optional[str]:
    """``emit_tier`` itself, or ValueError when it is not ``None`` or one of:data:`EMIT_TIERS`."""
    if emit_tier is not None and emit_tier not in EMIT_TIERS:
        raise ValueError(f"emit_tier must be None or one of {', '.join(EMIT_TIERS)}, got {emit_tier!r}")
    return emit_tier


def emit_tier_flags(emit_tier: str) -> dict:
    """Map an ``--emit-tier`` value to the engine switches (``Orthonym`` keyword arguments).

    Single source of truth for the tier -> switch contract; ``cli._emit_tier_flags`` and ``name_compound``
    read it, and ``tests/unit/test_emit_tier_flags.py`` pins it.

    Invariant table (0-wrong-critical):

      pin -> gf=F, gfu=F, aag=F (PIN-or-abstain: the default tier of releases up to 1.0.x)
      valid -> gf=T, gfu=F, aag=F (round-trip-verified general-engine names)
      complete -> gf=T, gfu=F, aag=T (round-trip-verified aggressive aromatic/hetero names)
      best-effort -> gf=T, gfu=T, aag=T (complete's production plus the OPSIN-unverified opt-in)

    ``general_fallback_unverified`` (gfu) is the UNIQUE best-effort discriminator; ``allow_aromatic_general``
    (aag) is True for BOTH complete and best-effort, so best-effort candidate production is a superset of
    complete's. ``full-coverage`` is best-effort's switches plus its own ``full_coverage`` marker bit (the
    opt-in that arms the general coordination-additive namer); the key is present (False) on every other
    tier, so a consumer can always read it.
    """
    check_emit_tier(emit_tier)
    be_or_fc = emit_tier in ("best-effort", "full-coverage")
    return {
        "general_fallback": emit_tier != "pin",
        "general_fallback_unverified": be_or_fc,
        "allow_aromatic_general": emit_tier in ("complete", "best-effort", "full-coverage"),
        "full_coverage": emit_tier == "full-coverage",
    }


def verification_active() -> bool:
    """Whether the names of a call are read back by OPSIN: the jar is there and the validity gate is not
    switched off (the raw-output switches, and the reduced mode of a host without Java)."""
    from . import namer
    return bool(namer._validity_gate_jar_present()) and not namer._DISABLE_VALIDITY_GATE


def default_tier(style: str = "pin") -> str:
    """The tier the default names: ``best-effort`` for the PIN style when OPSIN reads the names back, else
    ``pin`` (``--style general`` and ``cas`` keep the names they gave)."""
    return DEFAULT_TIER if style == "pin" and verification_active() else "pin"


def default_flags(style: str = "pin") -> dict:
    """The engine switches of the default (:func:`default_tier`)."""
    return emit_tier_flags(default_tier(style))


def resolve_switches(style: str, emit_tier: Optional[str], general_fallback=None, general_fallback_unverified=None,
                     allow_aromatic_general=None, full_coverage=None, *, own_call: bool = False):
    """The four engine switches ``(general_fallback, general_fallback_unverified, allow_aromatic_general,
    full_coverage)`` for a call of ``name_compound``.

    ``emit_tier`` None leaves the switches as given, except for the caller's own call (``own_call``: no
    ``name`` runs above it) with none of them given, which is the default (:func:`default_flags`). A tier and
    a switch given together is a ``ValueError``."""
    given = (general_fallback, general_fallback_unverified, allow_aromatic_general, full_coverage)
    if emit_tier is None:
        if not own_call or any(v is not None for v in given):
            return given
        flags = default_flags(style)
    else:
        check_emit_tier(emit_tier)
        if any(v is not None for v in given):
            raise ValueError("emit_tier cannot be combined with general_fallback, general_fallback_unverified, "
                             "allow_aromatic_general or full_coverage: the tier sets them")
        flags = emit_tier_flags(emit_tier)
    return (flags["general_fallback"], flags["general_fallback_unverified"],
            flags["allow_aromatic_general"], flags["full_coverage"])


def tier_label_of(namer, name, smiles: str) -> Optional[str]:
    """The tier label (``pin_verified``...) of the name a call of ``namer`` returned, or None for no name:
    the row derived from the provenance of the call as ``name_tiered`` derives it right after ``name``."""
    from .errors import is_failure_name
    if not name or is_failure_name(name):
        return None
    try:
        from .metrics.provenance import get_provenance
        return namer._tier_row_and_pin_form(smiles, name, get_provenance())[0].get("tier")
    except Exception:                       # the label is information, never a reason to lose the name
        return None


def is_wider(namer) -> bool:
    """Whether ``namer`` runs a tier above the PIN-only one."""
    return bool(namer._general_fallback or namer._general_fallback_unverified or namer._allow_aromatic_general
                or namer._full_coverage)
