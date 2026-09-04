"""W3-P04 N-hydroxy-sulfonimidic-acid handler — PIN = 'N-hydroxy<sulfonimidic acid>'
(P-65.3.1.5).

A hydroximic acid derived from a sulfonic acid, R-S(=O)(=N-OH)-OH, is named per
P-65.3.1.5 as the *N-hydroxy derivative of the corresponding sulfonimidic acid*
(P-65.3.1.5 example: 'N-hydroxymethanesulfonimidic acid (PIN)'). This mirrors the
hydroximic-acid handler ('N-hydroxy<imidic acid>', P-65.1.3.3.1) exactly one
sulfur-acid class over.

Example:
- CS(=O)(O)=NO -> 'N-hydroxymethanesulfonimidic acid'

The sulfonimidic-acid core is named by the full production pipeline: we remove
the -OH from the imino nitrogen (=N-OH -> =NH), name the resulting sulfonimidic
acid, then prepend 'N-hydroxy'. FAILS CLOSED (returns None) whenever the core
cannot be named, so a wrong name is never emitted.

Gating: fires ONLY when the perceived sulfonimidic_acid principal group carries a
terminal -OH on its imino nitrogen (the hydroximic / N-hydroxy case). The plain
=NH parent (CS(=O)(=N)O -> 'methanesulfonimidic acid') has no such -OH, so the
predicate returns False and the ordinary suffix path names it.

IUPAC cite: P-65.3.1.5.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _imino_n_oh_oxygen(mol: Any, pg_atoms: Any) -> Optional[int]:
    """Return the atom index of a terminal -OH bonded to the imino N of a
    single sulfonimidic_acid match, or None.

    SMARTS match tuple is (S, =O, imino-N, -OH-on-S); match[2] is the imino N.
    The N-hydroxy (hydroximic) variant carries an extra terminal O single-bonded
    to that nitrogen. Returns None (no fire) for the plain =NH parent, for
    O-substituted (=N-O-R) forms, and for any multi-group case.
    """
    from rdkit import Chem

    if not pg_atoms or len(pg_atoms) != 1:
        return None
    match = pg_atoms[0]
    if len(match) < 3:
        return None
    n_idx = match[2]
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetAtomicNum() != 7:
        return None
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 8:
            bond = mol.GetBondBetweenAtoms(n_idx, nb.GetIdx())
            if (bond is not None
                    and bond.GetBondType() == Chem.BondType.SINGLE
                    and nb.GetDegree() == 1):
                return nb.GetIdx()
    return None


def _is_sulfonimidic_n_hydroxy(features: Any) -> bool:
    """Gate: principal_group == 'sulfonimidic_acid' AND the imino N bears a
    terminal -OH (the P-65.3.1.5 N-hydroxy / hydroximic case)."""
    if getattr(features, 'principal_group', None) != 'sulfonimidic_acid':
        return False
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    return _imino_n_oh_oxygen(mol, getattr(features, 'principal_group_atoms', None)) is not None


def name_sulfonimidic_n_hydroxy(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return N-hydroxy-sulfonimidic-acid handler (W3-P04, P-65.3.1.5).

    Names the sulfonimidic-acid core (=N-OH stripped to =NH) via the full
    pipeline, then prepends 'N-hydroxy'. Returns None on any failure so the
    cascade continues (fail-safe).
    """
    from rdkit import Chem

    from ...errors import is_failure_name
    from ...namer import name_compound
    from ..candidate_pool import get_current_pool

    _mol = mol if mol is not None else getattr(features, 'mol', None)
    if _mol is None:
        return None

    o_on_n = _imino_n_oh_oxygen(_mol, getattr(features, 'principal_group_atoms', None))
    if o_on_n is None:
        return None

    # Build the sulfonimidic-acid core: strip the imino-N -OH (=N-OH -> =NH).
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
    # Only accept a genuine sulfonimidic-acid core (guard against the core
    # re-routing to some other class) — the PIN suffix must be present.
    if "sulfonimidic acid" not in core_name:
        return None

    result_name = f"N-hydroxy{core_name}"

    pool = get_current_pool()
    pool.add(result_name, "sulfonimidic_acid", features)
    best = pool.best()
    if best is None:
        return None
    return NamingResult(
        name=best.name,
        tree=NameTreeNode(
            parent_stem=result_name,
            class_id="sulfonimidic_acid",
            iupac_section_cite="P-65.3.1.5",
            fragment_legacy=result_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_sulfonimidic_n_hydroxy", "_is_sulfonimidic_n_hydroxy"]
