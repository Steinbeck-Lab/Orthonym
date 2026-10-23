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
gate outcome that names them, and only the stereo-layer one is offered as the default tier's row (the twin
of a thioperoxol is the second class, below, a pass of the tier's own engine).

A third class, the adduct of a drawing of several fragments whose delocalised cation OPSIN draws on the other
resonance atom than the input has it (:func:`relaxed_adduct_row`, ``Orthonym._candidate_rt_match``). The recovery of
the best-effort tier judges such an adduct by the exit's rule (the full InChIKey, the radical graph and the
protonation site) only in the pass of this offer; in the first pass it keeps the exact comparison of canonical
SMILES and notes that it rejected such a candidate. A molecule that pass leaves without a name is named once more
with the exit's rule admitted, so a name that ships changes in no case (a cation drawn as a cyanine ships the name
its later retries give it, as before) and a molecule that had none gets the first candidate that reads back.

Every entry point agrees: ``name_tiered`` (:func:`offer_default_tier_row`, the last step of
``spelling_offer.name_tiered_with_offer``) and the caller's own ``Orthonym.name`` call
(:func:`default_tier_name_scope`, the decorator of ``name``, which ``name_compound`` and
``name_with_confidence`` call).:func:`hold_withdrawn` is called by the gate when it withdraws such a
name and:func:`clear_held` starts every outermost naming.

A second class, the PIN spelling of a thioperoxol (user decision 2026-10-22, "Best-effort ships readable"). A
general tier (``valid``, ``complete``, ``best-effort``) prefers the name OPSIN reads, the name by prefixes
'(hydroxysulfanyl)methane', to the PIN spelling 'methane-SO-thioperoxol', which only a twin spelling can verify
(``validation.thioperoxol_twin``), so the gate withdraws the neutral PIN spelling there. Such a name is held too, and
a molecule the tier leaves without a name is named once more with the twin admitted (:func:`twin_last_resort_row`).
The anion of -S-OH is not of this class: the engine has no other spelling for it, so the gate ships it through its
twin in the one pass (``twin_spellings.general_tier_twin_round_trip``). The second pass is a safety net, kept so that a
general tier never names less than it named when the twin shipped the PIN spelling at every tier: a molecule that
only the PIN path names (the valid and complete tiers have no systematic floor, and the general engine has size caps)
would otherwise be named by the default tier and declined by a wider one. No molecule of the measured sets needs it
(141 rows with an S-O group, three tiers, none lost without it). The pass is the tier's own engine, so the row, its
label and its gate outcome are the ones the twin gave before.
"""
import contextvars
import functools
from typing import Dict, Optional, Tuple

#: The names a general-tier gate withdrew for the molecule being named, in the order withdrawn: the names with a
#: stereo layer (every general tier holds them, the best-effort tier offers them) and the neutral thioperoxol names
#: that have a twin (every general tier holds and offers them), in the order withdrawn. The bound is a memory guard,
#: one naming withdraws a handful of such names, and it counts each class apart (:func:`hold_withdrawn`), so a name of
#: one class can never crowd a name of the other out.
_MAX_HELD = 16

_HELD: contextvars.ContextVar = contextvars.ContextVar("orthonym_default_tier_held", default=())
#: the name the caller's own ``name`` call was offered (:func:`default_tier_name_scope`)
_OFFERED: contextvars.ContextVar = contextvars.ContextVar("orthonym_default_tier_offered", default=None)


#: True once the recovery of the running naming has rejected, by the exact comparison of canonical SMILES, an
#: adduct of the whole molecule that the exit's rule accepts (:func:`note_relaxed_adduct`); read by
#::func:`relaxed_adduct_row`, cleared with the held names
_RELAXED_SEEN: contextvars.ContextVar = contextvars.ContextVar("orthonym_relaxed_adduct_seen", default=False)
#: True inside the pass of:func:`relaxed_adduct_row`, where the recovery accepts such an adduct
_RELAXED_ON: contextvars.ContextVar = contextvars.ContextVar("orthonym_relaxed_adduct_on", default=False)


def clear_held() -> None:
    """Forget the held names and the name offered. Called at the start of every outermost naming (``name``
    or ``name_tiered``), so that an offer reads only the names its own run withdrew."""
    _HELD.set(())
    _OFFERED.set(None)
    _RELAXED_SEEN.set(False)


def note_relaxed_adduct() -> None:
    """The recovery rejected an adduct of the whole molecule that the exit's rule accepts: the molecule is
    offered the pass of:func:`relaxed_adduct_row` if no other path names it."""
    _RELAXED_SEEN.set(True)


def relaxed_adduct_active() -> bool:
    """Whether the running naming is the pass of:func:`relaxed_adduct_row`."""
    return _RELAXED_ON.get()


def is_offered(namer, name) -> bool:
    """Whether ``name`` is the name the caller's own ``name`` call of the best-effort engine ``namer`` was
    offered. The exit check of ``name_with_confidence`` reads it: the default tier's own checks, which the
    round trip of a general tier cannot make for a name OPSIN cannot read, have passed it."""
    return (bool(name) and name == _OFFERED.get() and getattr(namer, "style", None) == "pin"
            and bool(getattr(namer, "_general_fallback_unverified", False)))


def held_names() -> Tuple[str, ...]:
    """The names a general-tier gate withdrew for the whole molecule and that an offer may give back: a name with a
    stereo layer, or a neutral thioperoxol name that has a twin, in the order withdrawn."""
    return _HELD.get()


def _has_stereo_layer(name: str) -> bool:
    from ..rules.stereochemistry import strip_stereo
    return strip_stereo(name) != name


def _is_neutral_thioperoxol(name: str) -> bool:
    """A thioperoxol name with a twin and no charge: the name a general tier gives way to the name OPSIN reads."""
    from ..validation.thioperoxol_twin import thioperoxol_twin
    found = thioperoxol_twin(name)
    return found is not None and not found.anion


def hold_withdrawn(name: str, smiles: str, caller_input) -> None:
    """Hold ``name``, which a general-tier gate withdrew for ``smiles`` because OPSIN cannot read it.

    Only a WHOLE-MOLECULE call is held: ``smiles`` is the string the outermost ``name`` was given
    (``caller_input``); the names of fragments and of other molecules a producer probes are not. Only a name
    with a stereo layer or a neutral thioperoxol name that has a twin is held: the stereo-layer carve-out is
    licensed by OPSIN reading the stereo-stripped name, so a name without one cannot be a name the default tier
    ships through it, and the twin of a neutral thioperoxol name is the other class of the module docstring (a
    narrowing of what is held; the default tier and the twin decide every offer). A thioperoxolate anion is not
    held: the gate has already tried its twin. Each class has its own bound ``_MAX_HELD``. Never raises: it is
    called inside the gate."""
    try:
        if not name or not smiles or smiles != caller_input:
            return
        classes = tuple(is_class for is_class in (_has_stereo_layer, _is_neutral_thioperoxol) if is_class(name))
        held = held_names()
        if classes and name not in held and any(sum(map(is_class, held)) < _MAX_HELD for is_class in classes):
            _HELD.set(held + (name,))
    except Exception:                       # pragma: no cover - a hold must never break naming
        pass


def _engine_like(namer, **tier_flags):
    """An engine with the construction options of ``namer`` and the tier flags ``tier_flags``."""
    from ..namer import Orthonym
    return Orthonym(
        style=namer.style,
        _disable_grammar_validation=(namer._grammar is None),
        _disable_opsin_validity_gate=namer._disable_opsin_validity_gate,
        enable_triviality_controller=namer._enable_triviality_controller,
        enable_group_splitting=namer._enable_group_splitting,
        trivial_fallback=namer._trivial_fallback,
        _seed_excluded_dispatch_classes=namer._seed_excluded_dispatch_classes,
        binding_proof=namer._binding_proof,
        **tier_flags,
    )


def default_tier_engine(namer):
    """A default-tier engine with the construction options of ``namer`` and every breadth flag off.

    The one place that builds it: the strict twin of a wider tier (``namer.Orthonym._strict_pin_twin_name``,
    which decides the honest PIN status of a label) and the engine of the offer below. It inherits the
    OPSIN-gate setting, so its acceptance bar matches ``namer``'s."""
    return _engine_like(namer, general_fallback=False, general_fallback_unverified=False,
                        allow_aromatic_general=False, full_coverage=False)


def general_tier_engine(namer):
    """An engine at the tier of ``namer`` (the same breadth flags) with its construction options: the engine of
    :func:`twin_last_resort_row`, which names the molecule once more."""
    return _engine_like(namer, general_fallback=namer._general_fallback,
                        general_fallback_unverified=namer._general_fallback_unverified,
                        allow_aromatic_general=namer._allow_aromatic_general,
                        full_coverage=namer._full_coverage)


def default_tier_row_for(namer, smiles) -> Optional[Dict]:
    """The row ``namer`` offers a molecule it left without a name, or None. ``smiles`` is a SMILES string or an
    RDKit molecule, as for ``name_tiered``.

    The default tier's row when ``namer`` is a best-effort engine of the PIN style, the gate withdrew names for
    the molecule, and the default tier ships one of them through its stereo-layer carve-out
    (:func:`stereo_layer_row`); else, at any general tier, the row of the twin last resort
    (:func:`twin_last_resort_row`)."""
    if getattr(namer, "style", None) != "pin":
        return None
    held = held_names()
    row = (stereo_layer_row(namer, smiles, held) or twin_last_resort_row(namer, smiles, held)) if held else None
    return row or relaxed_adduct_row(namer, smiles)


def relaxed_adduct_row(namer, smiles) -> Optional[Dict]:
    """The row of ``namer`` for ``smiles`` named once more with the exit's rule admitted for the adduct of the
    whole molecule, when the first pass left the molecule without a name after rejecting such an adduct
    (:func:`note_relaxed_adduct`); else None.

     (Organic adducts): "Names are formed by citing the names of individual compounds in the order of
    the formula connected by long (em) dashes". OPSIN draws a delocalised cation on either resonance atom, so
    an adduct whose component reads back to the same species on the other atom differs from the input in its
    canonical SMILES. The pass is the tier's own engine (:func:`general_tier_engine`), so the row, its label
    and its gate outcome are the tier's; the row is returned only for a name that joins components by long
    dashes. No pass inside the pass. Best-effort tier only."""
    if not _RELAXED_SEEN.get() or _RELAXED_ON.get() or not getattr(namer, "_general_fallback_unverified", False):
        return None
    if getattr(namer, "_forced_parent_rank", 0):
        return None
    held = held_names()
    token = _RELAXED_ON.set(True)
    try:
        from ..errors import is_failure_name
        from ..rules.adducts import EM_DASH
        alt = general_tier_engine(namer).name_tiered(smiles)
    except Exception:                       # the offer must never break a naming: the abstention stands
        return None
    finally:
        _RELAXED_ON.reset(token)
        _HELD.set(held)                     # the pass named the molecule again, and cleared what it held
    name = alt.get("name")
    if name and not is_failure_name(name) and alt.get("tier") != "abstain" and EM_DASH in name:
        return alt
    return None


def stereo_layer_row(namer, smiles, held) -> Optional[Dict]:
    """The default tier's row for ``smiles`` when ``namer`` is a best-effort engine and the default tier ships,
    through its stereo-layer carve-out, one of the names ``held`` that the best-effort gate withdrew; else
    None."""
    held = tuple(name for name in held if _has_stereo_layer(name))
    if not held or not getattr(namer, "_general_fallback_unverified", False):
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


def twin_last_resort_row(namer, smiles, held) -> Optional[Dict]:
    """The row of ``namer`` for ``smiles`` named once more with the thioperoxol twin admitted, when the gate
    withdrew a neutral thioperoxol name that has a twin (``held``) and no other path left the molecule with a name;
    else None.

    The PIN spelling 'methane-SO-thioperoxol' is withdrawn at a general tier in favour of the name OPSIN reads
    ('(hydroxysulfanyl)methane'); a molecule that has no such name keeps the PIN spelling the twin read back before,
    with its row as the tier gave it: ``validation.thioperoxol_twin.last_resort_scope`` admits the twin in the gate,
    and the pass is the engine of the tier (:func:`general_tier_engine`), so no label is decided here. The row is
    returned only when it is a name the gate withdrew in the first pass and a twin read back (the gate outcome
    ``twin_verified:thioperoxol_...``). No second pass inside the pass, none for a junior parent of the pool
    (``spelling_offer``): the name is the whole molecule's. The anion of -S-OH never comes here: the gate ships it
    in the first pass."""
    from ..validation.thioperoxol_twin import last_resort_active, last_resort_scope, thioperoxol_twin_outcome
    if (last_resort_active() or not getattr(namer, "_general_fallback", False)
            or getattr(namer, "_forced_parent_rank", 0) or not any(map(_is_neutral_thioperoxol, held))):
        return None
    try:
        from ..errors import is_failure_name
        with last_resort_scope():
            alt = general_tier_engine(namer).name_tiered(smiles)
    except Exception:                       # the offer must never break a naming: the abstention stands
        return None
    finally:
        _HELD.set(held)                     # the pass named the molecule again, and cleared what it held
    name = alt.get("name")
    if (name and not is_failure_name(name) and name in held and alt.get("tier") != "abstain"
            and thioperoxol_twin_outcome(alt.get("gate_outcome"))):
        return alt
    return None


def offer_default_tier_row(namer, smiles: str, row: Dict) -> Dict:
    """``row``, or the row ``default_tier_row_for`` offers when ``row`` is an abstention: the default tier's
    row when the default tier ships, through its stereo-layer carve-out, a name the best-effort gate withdrew,
    else the row of the thioperoxol twin last resort of a general tier."""
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
        if (not getattr(self, "_general_fallback", False) or getattr(self, "style", None) != "pin"
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
