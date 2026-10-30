"""Substructure matches of a query whose atom order is not the caller's choice.

RDKit's matcher extends a partial match in the atom order of the query. A molecule parsed from an OPSIN name comes
in OPSIN's order, which can start at a leaf of a highly symmetric molecule (a methyl of one of fifteen tert-butyl
groups); the matcher then tries the automorphisms of every leaf group before it reaches an atom that fails, and on
such a molecule it does not return (measured: a 134-atom boron-carbazole emitter, no return in 130 s). The same
query renumbered into its canonical SMILES order, which walks the molecule from a canonical start along bonds,
matches in 0.08 s with the same matches. ``ordered_matches`` searches that renumbered query and returns each match
indexed by the query's own atoms, so callers read it as before.
"""
from typing import List, Tuple

import numpy as np
from rdkit import Chem


def ordered_matches(target: Chem.Mol, query: Chem.Mol, *, use_chirality: bool = False, uniquify: bool = True,
                    max_matches: int = 1000) -> List[Tuple[int, ...]]:
    """:func:`ordered_match_array` as a list of tuples."""
    return [tuple(row) for row in ordered_match_array(target, query, use_chirality=use_chirality,
                                                       uniquify=uniquify, max_matches=max_matches).tolist()]


def ordered_match_array(target: Chem.Mol, query: Chem.Mol, *, use_chirality: bool = False, uniquify: bool = True,
                        max_matches: int = 1000) -> np.ndarray:
    """``target.GetSubstructMatches(query,...)`` searched in the canonical atom order of ``query``.

    One row per match, indexed by the atoms of ``query`` (as RDKit returns a match); an array, because a caller
    that ranks 100,000 matches of a large molecule spent seconds per call on the tuples alone. Which matches are
    found first can
    differ from a search in the query's own order: uncapped and with ``uniquify=False`` the set of matches is the
    same; with ``uniquify=True`` the representative of a set of atoms can differ, and under ``max_matches`` the
    subset can.
    """
    work = Chem.Mol(query)
    try:
        Chem.MolToSmiles(work)
        order = [int(i) for i in work.GetPropsAsDict(True, True).get("_smilesAtomOutputOrder", ())]
    except Exception:  # noqa: BLE001 -- no canonical order: the query is searched as given
        order = []
    if sorted(order) != list(range(query.GetNumAtoms())):
        found = target.GetSubstructMatches(query, useChirality=use_chirality, uniquify=uniquify,
                                           maxMatches=max_matches)
        return np.asarray(found, dtype=np.intp).reshape(-1, query.GetNumAtoms())
    renumbered = Chem.RenumberAtoms(query, order)
    found = target.GetSubstructMatches(renumbered, useChirality=use_chirality, uniquify=uniquify,
                                       maxMatches=max_matches)
    if not found:
        return np.empty((0, len(order)), dtype=np.intp)
    # match[new] is the target atom of query atom order[new]: index the columns by the inverse order
    inverse = np.empty(len(order), dtype=np.intp)
    inverse[np.asarray(order, dtype=np.intp)] = np.arange(len(order), dtype=np.intp)
    return np.asarray(found, dtype=np.intp)[:, inverse]
