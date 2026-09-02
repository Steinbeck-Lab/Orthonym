"""W3-P02 hydroximic acid handler — PIN = 'N-hydroxy<imidic acid>' (P-65.1.3.3.1).

A hydroximic acid (R-C(=N-OH)-OH) is, per P-65.1.3.3.1, named as the
*N-hydroxy derivative of the corresponding imidic acid* — NOT with the
general-only '-hydroximic acid' suffix (P-65.1.3.3.1 Note). This mirrors the
hydroxamic-acid handler ('N-hydroxy<stem>amide') exactly one class up.

Examples:
- CC(O)=NO -> 'N-hydroxyethanimidic acid' (acetohydroximic acid)
- OC=NO -> 'N-hydroxymethanimidic acid' (formohydroximic acid)

The imidic-acid core is named by the full production pipeline: we remove the
-OH from the imino nitrogen (=N-OH -> =NH), name the resulting imidic acid,
then prepend 'N-hydroxy'. This inherits chain/ring/substituent handling from
the imidic path and FAILS CLOSED (returns None) whenever the core cannot be
named (so a wrong name is never emitted).

IUPAC cite: P-65.1.3.3.1.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_hydroximic_acid(features: Any) -> bool:
    """Gates on principal_group == 'hydroximic_acid'."""
    return getattr(features, 'principal_group', None) == 'hydroximic_acid'


def name_hydroximic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return hydroximic acid handler (W3-P02, P-65.1.3.3.1).

    Names the imidic-acid core (=N-OH stripped to =NH) via the full pipeline,
    then prepends 'N-hydroxy'. Returns None on any failure so the cascade
    continues (fail-safe per).
    """
    from rdkit import Chem

    from ..candidate_pool import get_current_pool
    from ...errors import is_failure_name
    from ...namer import name_compound

    _mol = mol if mol is not None else getattr(features, 'mol', None)
    if _mol is None:
        return None

    pg_atoms = getattr(features, 'principal_group_atoms', None)
    if not pg_atoms:
        return None

    # SMARTS match tuple: (C, imino-N, O-on-N, hydroxyl-O-on-C). Only ONE
    # hydroximic group is supported as the principal group here; a molecule
    # with two would need multi-suffix handling — fail closed.
    if len(pg_atoms) != 1:
        return None
    match = pg_atoms[0]
    if len(match) < 4:
        return None
    n_idx = match[1]

    # Locate the -OH oxygen bonded to the imino N by graph walk (robust to
    # tuple order): the O neighbour of the imino nitrogen.
    n_atom = _mol.GetAtomWithIdx(n_idx)
    o_on_n = None
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 8:
            bond = _mol.GetBondBetweenAtoms(n_idx, nb.GetIdx())
            if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                o_on_n = nb.GetIdx()
                break
    if o_on_n is None:
        return None
    # The =N-OH oxygen must be terminal (degree 1); an O-substituted
    # hydroximate (=N-O-R) is a different class (fail closed).
    if _mol.GetAtomWithIdx(o_on_n).GetDegree() != 1:
        return None

    # Build the imidic-acid core: strip the imino-N -OH (=N-OH -> =NH).
    try:
        rw = Chem.RWMol(_mol)
        rw.RemoveAtom(o_on_n)
        core = rw.GetMol()
        Chem.SanitizeMol(core)
        core_smiles = Chem.MolToSmiles(core)
    except Exception:
        return None

    core_name = name_compound(core_smiles, style=style)
    if not core_name or is_failure_name(core_name):
        return None

    result_name = f"N-hydroxy{core_name}"

    pool = get_current_pool()
    pool.add(result_name, "hydroximic_acid", features)
    best = pool.best()
    if best is None:
        return None
    return NamingResult(
        name=best.name,
        tree=NameTreeNode(
            parent_stem=result_name,
            class_id="hydroximic_acid",
            iupac_section_cite="P-65.1.3.3.1",
            fragment_legacy=result_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_hydroximic_acid", "_is_hydroximic_acid"]
