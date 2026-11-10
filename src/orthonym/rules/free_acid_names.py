"""The name of the free acid of an ester side, once per naming.

The ester producers name an ester side through its free acid (``rules.esters._free_acid_name``, the hub-group and polyol
esters' ``hub_group_esters.anion_of_free_acid``). The polyol-ester routes ask the free acid of one side again and again:
22 namings of one hemi-carbonate acid, about 25 s, in one spiro-oxindole carbonate (program 22, 2026-11-10). The name
is kept once per top-level naming for the same acid and ambient state (``assembly.nested_memo``, with the key of
``rules.adducts._name_component``). The kinds of the book spellings the naming gave are kept with the name and noted
again on every call, a hit too: the enclosing ``watch_book_forms`` box decides whether the molecule is named again
with the mechanical spellings (``namer._name_with_book_retry``, ``assembly.radical_floor``).
"""
from typing import Optional


def free_acid_name(acid_smi: str) -> Optional[str]:
    """The name of the free acid of ``acid_smi`` through the general pipeline, None when it has none."""
    from ..assembly.book_prefixes import note_book_form, watch_book_forms
    from ..assembly.nested_memo import cached_nested_call, renaming_state
    from .adducts import _naming_context

    def fresh():
        with watch_book_forms() as fired:
            name = _free_acid_name_once(acid_smi)
        return name, tuple(sorted(fired[1]))

    name, kinds = cached_nested_call("free_acid_name", (acid_smi, _naming_context(), renaming_state()), fresh,
                                     replay_budgets=False)
    for kind in kinds:
        note_book_form(kind)
    return name


def _free_acid_name_once(acid_smi: str) -> Optional[str]:
    """:func:`free_acid_name` without the memo."""
    from orthonym import name_compound  # lazy: re-entrant on the free-acid fragment
    try:
        acid_name = name_compound(acid_smi)
    except Exception:
        return None
    if (not acid_name
            or "unknown" in acid_name.lower()
            or "not supported" in acid_name.lower()):
        return None
    return acid_name
