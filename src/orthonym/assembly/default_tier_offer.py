"""The best-effort tier does not name less than the default tier names.

The validity gate withdraws, at the best-effort tier, a name OPSIN 2.9.0 cannot read: that tier ships a
name only when OPSIN reads it back to the input's full InChIKey. The default tier ships one class of
such names, the strict path's PIN spelling whose stereo layer OPSIN rejects (the lowercase ``r``/``s``
of ``(1r,4r)-cyclohexane-1,4-diol``, "Achiral cyclic compounds"; a name of a polycycle
with many stereocentres): the gate reads the stereo-stripped name, compares its constitution to the
input's, and the descriptors are checked against the input's CIP labels (the gate outcome
``self_consistency_constitution_only``, label ``pin_unverified``). At the best-effort tier the same name
was withdrawn, so a molecule the default tier names was left without a name by the tier above it.

User decision 2026-10-20 ("ship it too"): a molecule the best-effort tier leaves unnamed gets the row
the default tier gives it, when that row is a ``pin_unverified`` name of this class and is a name the
best-effort gate withdrew for the molecule. It is an offer, not a return (a project rule): it is made
only when every other path of the best-effort tier has left the molecule without a name, so no name
that ships changes, and a verified name of another producer (the cis/trans name of
``rules.relative_stereo``, the general engine's name) is never replaced by it.

Parity, not a new trust path. The row is the default tier's own: the molecule is named by a default-tier
engine (:func:`default_tier_engine`, with the construction options of the best-effort engine) and its
row, label and gate outcome are returned as they are (``opsin`` ``verified_constitution_only``,
``verified`` ``opsin_constitution``). Whether the default tier ships the name is decided by the default
tier's code (the grammar carve-outs, the stereo-layer carve-out, the emission rule of
``namer._default_tier_emits``), not by a copy of its conditions here. A row of any other default-tier
exception stays withheld at best-effort: the grammar carve-outs, the organometallic additive carve-out
(it has a known atom-drop hole, see ``namer._final_opsin_validity_gate``) and the twin ships keep the
gate outcome that names them, and only the stereo-layer one is offered.

Every entry point agrees: ``name_tiered`` (:func:`offer_default_tier_row`, the last step of
``spelling_offer.name_tiered_with_offer``) and the caller's own ``Orthonym.name`` call
(:func:`default_tier_name_scope`, the decorator of ``name``, which ``name_compound`` and
``name_with_confidence`` call).:func:`hold_withdrawn` is called by the gate when it withdraws such a
name and:func:`clear_held` starts every outermost naming.
"""
import contextvars
import functools
from typing import Dict, Optional, Tuple

#: The names the best-effort gate withdrew for the molecule being named, in the order withdrawn. The bound
#: is a memory guard: one naming withdraws a handful of such names.
_MAX_HELD = 16

_HELD: contextvars.ContextVar = contextvars.ContextVar("orthonym_default_tier_held", default=())
#: the name the caller's own ``name`` call was offered (:func:`default_tier_name_scope`)
_OFFERED: contextvars.ContextVar = contextvars.ContextVar("orthonym_default_tier_offered", default=None)


def clear_held() -> None:
    """Forget the held names and the name offered. Called at the start of every outermost naming (``name``
    or ``name_tiered``), so that an offer reads only the names its own run withdrew."""
    _HELD.set(())
    _OFFERED.set(None)


def is_offered(namer, name) -> bool:
    """Whether ``name`` is the name the caller's own ``name`` call of the best-effort engine ``namer`` was
    offered. The exit check of ``name_with_confidence`` reads it: the default tier's own checks, which the
    round trip of a general tier cannot make for a name OPSIN cannot read, have passed it."""
    return (bool(name) and name == _OFFERED.get() and getattr(namer, "style", None) == "pin"
            and bool(getattr(namer, "_general_fallback_unverified", False)))


def held_names() -> Tuple[str, ...]:
    """The names with a stereo layer that the gate withdrew for the whole molecule, in the order withdrawn."""
    return _HELD.get()


def hold_withdrawn(name: str, smiles: str, caller_input) -> None:
    """Hold ``name``, which the best-effort gate withdrew for ``smiles`` because OPSIN cannot read it.

    Only a WHOLE-MOLECULE call is held: ``smiles`` is the string the outermost ``name`` was given
    (``caller_input``); the names of fragments and of other molecules a producer probes are not. Only a name
    with a stereo layer is held: the stereo-layer carve-out is licensed by OPSIN reading the stereo-stripped
    name, so a name without one cannot be a name the default tier ships through it (a narrowing of what is
    held; the default tier decides every offer). Never raises: it is called inside the gate."""
    try:
        if not name or not smiles or smiles != caller_input:
            return
        from ..rules.stereochemistry import strip_stereo
        if strip_stereo(name) == name:
            return
        held = held_names()
        if name not in held and len(held) < _MAX_HELD:
            _HELD.set(held + (name,))
    except Exception:                       # pragma: no cover - a hold must never break naming
        pass


def default_tier_engine(namer):
    """A default-tier engine with the construction options of ``namer`` and every breadth flag off.

    The one place that builds it: the strict twin of a wider tier (``namer.Orthonym._strict_pin_twin_name``,
    which decides the honest PIN status of a label) and the engine of the offer below. It inherits the
    OPSIN-gate setting, so its acceptance bar matches ``namer``'s."""
    from ..namer import Orthonym
    return Orthonym(
        style=namer.style,
        _disable_grammar_validation=(namer._grammar is None),
        _disable_opsin_validity_gate=namer._disable_opsin_validity_gate,
        enable_triviality_controller=namer._enable_triviality_controller,
        enable_group_splitting=namer._enable_group_splitting,
        trivial_fallback=namer._trivial_fallback,
        general_fallback=False,
        general_fallback_unverified=False,
        allow_aromatic_general=False,
        full_coverage=False,
        _seed_excluded_dispatch_classes=namer._seed_excluded_dispatch_classes,
        binding_proof=namer._binding_proof,
    )


def default_tier_row_for(namer, smiles) -> Optional[Dict]:
    """The default tier's row for ``smiles`` when ``namer`` is a best-effort engine of the PIN style, the
    gate withdrew names for the molecule, and the default tier ships one of them through its stereo-layer
    carve-out; else None. ``smiles`` is a SMILES string or an RDKit molecule, as for ``name_tiered``."""
    held = held_names()
    if not held or getattr(namer, "style", None) != "pin" or not getattr(namer, "_general_fallback_unverified", False):
        return None
    try:
        from ..errors import is_failure_name
        from ..metrics import provenance as pv
        alt = default_tier_engine(namer).name_tiered(smiles)
    except Exception:                       # the offer must never break a naming: the abstention stands
        return None
    name = alt.get("name")
    if (name and not is_failure_name(name) and name in held and alt.get("tier") != "abstain"
            and alt.get("gate_outcome") == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY):
        return alt
    return None


def offer_default_tier_row(namer, smiles: str, row: Dict) -> Dict:
    """``row``, or the default tier's row when ``row`` is an abstention of the best-effort tier and the
    default tier ships, through its stereo-layer carve-out, a name the best-effort gate withdrew."""
    if row.get("tier") != "abstain":
        return row
    return default_tier_row_for(namer, smiles) or row


def default_tier_name_scope(fn):
    """The decorator of ``Orthonym.name`` that gives the caller's own best-effort call the name of
    :func:`default_tier_row_for` when the call leaves the molecule without a name, as ``name_tiered`` does.

    Directly under ``failure_mode_scope``: it sees the caller's input before ``on_failure`` turns a failure
    into ``""`` or an error (the ``"raise"`` mode raises ``OrthonymLimitError`` from ``fn``; the molecule is
    then offered the same way). A nested call, a call inside the retry that ``name_tiered`` owns
    (``namer._BOOK_RETRY_OWNED``; ``name_tiered`` makes its own offer) and every other tier run ``fn``
    directly."""
    @functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        from ..namer import _BOOK_RETRY_OWNED
        from .fragment_naming import name_scope_depth
        if (not getattr(self, "_general_fallback_unverified", False) or getattr(self, "style", None) != "pin"
                or _BOOK_RETRY_OWNED.get() or name_scope_depth() > 0):
            return fn(self, *args, **kwargs)
        from ..errors import OrthonymLimitError, is_failure_name
        clear_held()
        try:
            produced = fn(self, *args, **kwargs)
        except OrthonymLimitError:
            offered = _offered_name(self, args, kwargs)
            if offered is None:
                raise
            return offered
        if produced and not is_failure_name(produced):
            return produced
        offered = _offered_name(self, args, kwargs)
        return produced if offered is None else offered
    return _wrapper


def _offered_name(namer, args, kwargs) -> Optional[str]:
    row = default_tier_row_for(namer, args[0] if args else kwargs.get("smiles"))
    if row is None:
        return None
    _OFFERED.set(row["name"])
    return row["name"]
