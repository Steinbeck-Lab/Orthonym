"""The names of the neutralized skeleton of an anion, computed once (``rules.ions.name_anion``)."""
from __future__ import annotations

from rdkit import Chem


def neutral_skeleton_name_once(namer, mol) -> str:
    """``namer(mol)``, the name of the neutralized skeleton of the anion ``mol``, computed once per molecule and naming
    state. The multi-anion branch of ``name_anion`` is reached for every call of ``name_anion`` on the molecule (20
    per row on the large lipid anions of the dev sets), and the skeleton's name does not depend on which call asks (the
    scope-bound replay memo of nested namings, keyed like its other users by the skeleton and the ambient state the nested
    naming reads, ``nested_memo.renaming_state``)."""
    from ..assembly.nested_memo import cached_nested_call, renaming_state
    return cached_nested_call(
        "anion_neutral_skeleton_name",
        (getattr(namer, "__name__", repr(namer)), Chem.MolToSmiles(mol), renaming_state()),
        lambda: namer(mol))
