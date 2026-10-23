"""The best-effort tier does not ship a chain-parent name that breaks a spelling rule while another
parent of the same molecule gives a name that breaks none.

The spelling checks (``validation.pin_spelling``) run on a name that would be labelled ``pin_verified``;
a name already below the PIN is never read by them, so a chain-parent name of a secondary or tertiary
amine that was malformed, and labelled ``systematic_verified`` for another reason, shipped as it was:
``3-(5-chlorofuran-2-yl)-N-2-methoxyethylprop-2-en-1-amine`` (a compound prefix without its enclosing
marks,, the Blue Book), ``N,N-dimethyl(diphenyl)methanamine`` (the multiplicative prefix
inside the parentheses,,:7272), a name with the stereodescriptors of a substituent at its front
,:44643). The parent of such an amine is the chain since counts the nitrogen for every
chain it is bonded to (``parent_selection.bridging_group_reaches``); before, the ring was chosen by the
order of the atoms and the ring-parent producers gave a clean name.

The pool of parents is an offer (``p44_scorer.select_parent_unified``, ``_offer_rank``): the chain is
``ranked[0]``, the rings are the junior members. When the name of ``ranked[0]`` breaks a spelling rule, the
junior members are named in turn, and the first whose name breaks none, reads back to the input, and has a label
not below the label of the chain-parent name replaces it (never ``pin_verified``: a junior parent is not the PIN
while the PIN tier declines it). No such name: the chain-parent name ships, as before, and
the rules it breaks are listed in the row's ``spelling_failures``. The PIN tier is not touched (it declines a name
that breaks a rule), nor is any molecule whose parent is not a chain parent of an amine.
"""
from __future__ import annotations

import contextvars
from typing import Optional

#: canonical SMILES of the molecule whose top-level naming added a chain-parent candidate of a secondary
#: or tertiary amine to the pool (:func:`note_amine_chain_parent`); another molecule's flag never matches
_AMINE_CHAIN_PARENT: contextvars.ContextVar = contextvars.ContextVar("orthonym_amine_chain_parent", default=None)

#: at most this many junior parents are named
_MAX_OFFERS = 4


def note_amine_chain_parent(features) -> None:
    """Note that the name added to the pool for ``features`` is built on the chain parent of a secondary or
    tertiary amine (cyclic molecule: the rings are the other members of the pool). Only for the molecule
    being named, not for a substituent named inside it."""
    from ..rules.parent_selection import BRIDGING_HETEROATOM_PGS
    from .fragment_naming import is_top_level_naming
    if (getattr(features, 'principal_group', None) not in BRIDGING_HETEROATOM_PGS
            or not getattr(features, 'is_cyclic', False) or not is_top_level_naming()):
        return
    mol = getattr(features, 'mol', None)
    if mol is not None:
        from rdkit import Chem
        _AMINE_CHAIN_PARENT.set(Chem.MolToSmiles(mol))


def name_tiered_with_offer(namer, smiles: str, run) -> dict:
    """``run(smiles)``, the tiered naming of the molecule, then:func:`prefer_clean_name` on its row and, for a
    best-effort abstention, ``default_tier_offer.offer_default_tier_row``. The flag and the held names of an
    earlier naming are cleared first, so what the row is offered never depends on what was named before it."""
    from .default_tier_offer import clear_held, offer_default_tier_row
    from .fragment_naming import hang_budget_trips
    from .memo import recording_scopes
    _AMINE_CHAIN_PARENT.set(None)
    clear_held()
    trips = hang_budget_trips()
    with recording_scopes() as passes:
        row = run(smiles)
    if hang_budget_trips() != trips:
        passes = ()                         # a naming that ran out of a hang budget hands on nothing
    return offer_default_tier_row(namer, smiles, prefer_clean_name(namer, smiles, row, passes))


def prefer_clean_name(namer, smiles: str, row: dict, passes=()) -> dict:
    """``row`` of the best-effort tier, or the row of the junior parent whose name breaks no spelling rule
    when the chain-parent name of ``row`` breaks one (see the module docstring). ``passes`` are the memo
    scopes of the namings that made ``row`` (``memo.recording_scopes``): the junior namings start with the
    names of the nested calls those namings made."""
    try:
        return _prefer_clean_name(namer, smiles, row, passes)
    except Exception:                       # the offer must never break a naming: the row stands
        return row


def _prefer_clean_name(namer, smiles: str, row: dict, passes=()) -> dict:
    from rdkit import Chem
    from ..errors import is_failure_name
    from ..namer import _LABEL_RANK, Orthonym
    from ..validation.pin_spelling import check_pin_spelling
    from .fragment_naming import hang_budget_trips
    from .memo import recording_scopes, seeded_scopes, shareable_entries
    if (_AMINE_CHAIN_PARENT.get() is None or row.get('tier') in ('abstain', 'pin_verified')
            or not getattr(namer, '_general_fallback_unverified', False)
            or getattr(namer, '_forced_parent_rank', 0) != 0
            or getattr(namer, '_top_parent_pool_size', 1) < 2):
        return row
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.MolToSmiles(mol) != _AMINE_CHAIN_PARENT.get():
        return row
    failures = check_pin_spelling(mol, row.get('name'))
    if not failures:
        return row
    pool = min(getattr(namer, '_top_parent_pool_size', 1), _MAX_OFFERS + 1)
    # Every naming of the molecule names the same parts of it again (a junior parent changes the parent
    # of the molecule, not the nested names of the groups on it): the names the namings so far gave to
    # nested calls are handed to the next one, which keeps a memo scope of its own.
    given = shareable_entries(passes)
    for rank in range(1, pool):
        inner = Orthonym(style=namer.style, general_fallback=True, general_fallback_unverified=True,
                         allow_aromatic_general=getattr(namer, '_allow_aromatic_general', False),
                         _forced_parent_rank=rank)
        trips = hang_budget_trips()
        with seeded_scopes(given), recording_scopes() as made:
            alt = inner.name_tiered(smiles)
        if hang_budget_trips() == trips:    # the names of a naming that ran out of a hang budget are cut short
            shareable_entries(made, into=given)
        name = alt.get('name')
        if (alt.get('tier') in ('abstain', 'pin_verified') or not name or is_failure_name(name)
                or _LABEL_RANK.get(alt.get('tier'), 0) < _LABEL_RANK.get(row.get('tier'), 0)
                or alt.get('verified') != 'opsin' or alt.get('spelling_failures')
                or check_pin_spelling(mol, name)):
            continue
        return alt
    row['spelling_failures'] = [{"rule": f.rule, "detail": f.detail} for f in failures]
    return row
