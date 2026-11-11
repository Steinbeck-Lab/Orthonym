"""The memos of the isotope decorator (``rules/isotopes.py``): one read-back per candidate and one skeleton naming per
engine and state, within one top-level naming (``assembly.memo``, ``assembly.nested_memo``; ``ORTHONYM_MEMO=verify``
computes again and compares)."""
from rdkit import Chem


def round_trips_memoised(candidate_name: str, original_mol: Chem.Mol, stereo_blind: bool = False) -> bool:
    """:func:`isotopes._isotope_round_trips_once`, once per top-level naming for the same candidate, structure and reading
    state. The placement search of the decorator reads one candidate again and again (a labelled mixture of
    2026-11-11: 27,512 read-backs, 5,354 distinct, 132 s). The key holds everything the check reads besides its
    arguments: the radical reading, the tier's spelling (the twin parse) and the tautomer pass; a hit replays the
    two records the check makes (a bond-shift isomer refused, a tautomer refused). ``ORTHONYM_MEMO=verify``
    checks again and compares."""
    from ..assembly.memo import cache_or_compute
    from . import isotopes as _iso
    from .isotope_compare import TAUTOMER_READING
    try:
        state = TAUTOMER_READING.get()
        key = (candidate_name, Chem.MolToSmiles(original_mol), stereo_blind, _iso._RADICAL_READING.get(),
               _iso._pin_spellings(), None if state is None else (state["possible"], state["accept"]))
    except Exception:  # noqa: BLE001 - a state that cannot be keyed is checked afresh
        return _iso._isotope_round_trips_once(candidate_name, original_mol, stereo_blind)

    def fresh():
        shift_token = _iso._BOND_SHIFT_PARSE_REFUSED.set(False)
        refused_before = state.get("refused") if state is not None else None
        if state is not None:
            state["refused"] = False
        try:
            result = _iso._isotope_round_trips_once(candidate_name, original_mol, stereo_blind)
            return result, _iso._BOND_SHIFT_PARSE_REFUSED.get(), bool(state is not None and state.get("refused"))
        finally:
            _iso._BOND_SHIFT_PARSE_REFUSED.reset(shift_token)
            if state is not None:
                state["refused"] = refused_before

    result, shift_refused, tautomer_refused = cache_or_compute("isotope_round_trip", key, fresh)
    if shift_refused:
        _iso._BOND_SHIFT_PARSE_REFUSED.set(True)
    if tautomer_refused and state is not None:
        state["refused"] = True
    return result


#: the context variables a skeleton naming writes and no naming reads back as an input: the memo caches, the
#: provenance (replayed by ``nested_memo.cached_nested_call``), the scoped tie flag and the last floor flags
_SKELETON_KEY_OUTPUT_VARS = ("orthonym_memo_", "orthonym_prov_", "orthonym_p45_tie", "t4_last_", "inner_dispatch_stats")


def skeleton_naming_key(naming, smiles: str) -> tuple:
    """The key of one skeleton naming: the SMILES, the engine's style and switches, the ambient naming state of
    ``nested_memo.renaming_state`` / ``ambient_naming_state`` and of ``adducts._naming_context``, and the value of
    every other context variable that is not an output (``_SKELETON_KEY_OUTPUT_VARS``; a molecule as its isomeric
    SMILES with isotopes), so a hit is served only for a naming in the same state. The engine enters by its scalar
    attributes only: its one set-valued switch, ``_excluded_dispatch_classes``, is set again by every ``.name``
    (``namer.py``), so it carries nothing from one naming to the next. A value without a hash enters by its ``repr``
    (the tautomer pass names molecules by address), which costs hits, never a wrong one."""
    import contextvars
    from ..assembly.nested_memo import ambient_naming_state
    from .adducts import _naming_context
    engine = (type(naming).__name__, getattr(naming, "style", None),
              tuple(sorted((k, v) for k, v in vars(naming).items() if isinstance(v, (bool, int, str, type(None))))))
    ambient = []
    for var, value in contextvars.copy_context().items():
        name = getattr(var, "name", "")
        if name.startswith(_SKELETON_KEY_OUTPUT_VARS):
            continue
        if isinstance(value, Chem.Mol):
            value = ("mol", Chem.MolToSmiles(value))
        else:
            try:
                hash(value)
            except TypeError:
                value = ("repr", repr(value))
        ambient.append((name, value))
    return (smiles, engine, ambient_naming_state(), _naming_context(), tuple(sorted(ambient, key=lambda x: (x[0], repr(x[1])))))


def skeleton_naming_memoised(naming, smiles: str) -> str:
    """``naming.name(smiles)``, once per top-level naming for the same skeleton, engine and state
    (:func:`skeleton_naming_key`): the passes of the decorator (as drawn, the induced centres unspecified, each with
    a tautomer reading, a scope composer, then the phane and hydro retries) name one skeleton up to a dozen times
    in one state (a labelled mixture of 2026-11-11: 12 namings, 4 states, 150 s of 238 s in the repeats). A hit
    returns the stored name and replays the provenance the naming recorded (``nested_memo.cached_nested_call``);
    a trip is never stored; ``ORTHONYM_MEMO=verify`` names again and compares."""
    from ..assembly.nested_memo import cached_nested_call
    try:
        key = skeleton_naming_key(naming, smiles)
        hash(key)
    except Exception:  # noqa: BLE001 - a state that cannot be keyed is named afresh
        return naming.name(smiles)
    return cached_nested_call("isotope_skeleton", key, lambda: naming.name(smiles), replay_budgets=False)
