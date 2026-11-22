"""The memos of the isotope decorator (``rules/isotopes.py``): one read-back per candidate and one skeleton naming per
engine and state, within one top-level naming (``assembly.memo``, ``assembly.nested_memo``; ``ORTHONYM_MEMO=verify``
computes again and compares)."""
import contextlib
import contextvars
import re
from typing import Optional

from rdkit import Chem


#: ``id(mol) -> (mol, its SMILES)`` for the molecules the placement search reads against: one search reads one
#: ``original_mol`` up to 75,000 times, and its SMILES (the memo key) cost 0.5 ms each time on a large labelled molecule
#: (an ISO row of the 2026-11-12 time-out audit: 14 of 49 s). The entry holds the molecule, so its id cannot be reused
#: while the entry lives, and a hit is checked by identity. The decorator never edits ``original_mol`` in place (every
#: label move works on a ``Chem.Mol``/``RWMol`` copy). At most ``_SMILES_SLOTS`` entries, the oldest dropped first.
_SMILES_OF: dict = {}
_SMILES_SLOTS = 64


def _smiles_of(mol: Chem.Mol) -> str:
    hit = _SMILES_OF.get(id(mol))
    if hit is not None and hit[0] is mol:
        return hit[1]
    smiles = Chem.MolToSmiles(mol)
    if len(_SMILES_OF) >= _SMILES_SLOTS:
        _SMILES_OF.pop(next(iter(_SMILES_OF)))
    _SMILES_OF[id(mol)] = (mol, smiles)
    return smiles


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
        key = (candidate_name, _smiles_of(original_mol), stereo_blind, _iso._RADICAL_READING.get(),
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
        key = None
    name = (naming.name(smiles) if key is None
            else cached_nested_call("isotope_skeleton", key, lambda: naming.name(smiles), replay_budgets=False))
    noted = _NOTED_SKELETONS.get()
    if noted is not None and name:
        noted.append(name)
    return name


#: The skeleton names one pass of the decorator built (:func:`skeleton_names_noted`), None outside a pass.
_NOTED_SKELETONS: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "noted_isotope_skeletons", default=None)


@contextlib.contextmanager
def skeleton_names_noted():
    """Note, in the list it yields, every skeleton name:func:`skeleton_naming_memoised` returns inside the block."""
    token = _NOTED_SKELETONS.set([])
    try:
        yield _NOTED_SKELETONS.get()
    finally:
        _NOTED_SKELETONS.reset(token)


#: An isotopic descriptor of as the decorator splices it ('(13C)', '(2-2H)', '(1,1,2,3,3-2H5)', '(N1,N1-2H2)'),
#: not after a digit (indicated hydrogen, 'naphthalen-1(2H)-one');:func:`_is_descriptor` checks each nuclide.
_DESCRIPTOR = re.compile(r"(?<![0-9])\((?:[0-9A-Za-z,'′″]+-)?(\d+[A-Z][a-z]?\d*(?:,\d+[A-Z][a-z]?\d*)*)\)")
_NUCLIDE = re.compile(r"(\d+)([A-Z][a-z]?)\d*")


def _is_descriptor(match) -> bool:
    """True when every nuclide of ``match`` is a nuclide: a known element whose mass number is not below its atomic
    number, not protium (so '(2S)', '(2R)', '(1H)' are no descriptors; '(34S)' is)."""
    if not _ATOMIC_NUMBER:
        table = Chem.GetPeriodicTable()
        _ATOMIC_NUMBER.update({table.GetElementSymbol(z): z for z in range(1, 119)})
    for mass, symbol in _NUCLIDE.findall(match.group(1)):
        z = _ATOMIC_NUMBER.get(symbol)
        if z is None or int(mass) < max(z, 2) or int(mass) > 3 * z + 30:
            return False
    return True


_ATOMIC_NUMBER: dict = {}


def _without_descriptors(name: str) -> str:
    return _DESCRIPTOR.sub(lambda m: "" if _is_descriptor(m) else m.group(0), name)


def carry_skeleton_labels(name: Optional[str], skeletons) -> Optional[str]:
    """Record ``name`` below the PIN when the skeleton name it is built on holds a non-PIN record (a label-only
    record, a non-PIN fragment or an uncertified PIN-form name, ``metrics.provenance``), and return ``name``.
    (the Blue Book, "The name of an isotopically substituted compound is formed by adding or inserting the
    nuclide symbol(s)... before the part of the compound that is isotopically substituted"): the name of the labelled
    compound is the name of the unlabelled one with the descriptor, so it is the PIN only when that name is; the
    recorded substring does not survive the descriptor ('2-[(4-oxo-4H-1-benzopyran-2-yl)methyl]-2,3-dihydro(3,3-2H2)-
    4H-1-benzopyran-4-one'). Only a skeleton that ``name`` reads as once its descriptors are taken out counts (a
    skeleton the pass named and did not build on carries nothing over); none matching, nothing is recorded."""
    from ..metrics.provenance import (record_derived_non_pin_fragment, record_derived_non_pin_label,
                                      record_derived_uncertified_pin_name)
    if not name:
        return name
    bare = _without_descriptors(name)
    for skeleton in skeletons or ():
        if skeleton == bare:
            record_derived_non_pin_label(skeleton, name)
            record_derived_non_pin_fragment(skeleton, name)
            record_derived_uncertified_pin_name(skeleton, name)
    return name
