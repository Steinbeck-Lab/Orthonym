"""W3-P02 hydroximic acid handler — PIN = 'N-hydroxy<imidic acid>'.

A hydroximic acid (R-C(=N-OH)-OH) is, per, named as the
*N-hydroxy derivative of the corresponding imidic acid* — NOT with the
general-only '-hydroximic acid' suffix Note). This mirrors the
hydroxamic-acid handler ('N-hydroxy<stem>amide') exactly one class up.

Examples:
- CC(O)=NO -> 'N-hydroxyethanimidic acid' (acetohydroximic acid)
- OC=NO -> 'N-hydroxymethanimidic acid' (formohydroximic acid)

The imidic-acid core is named by the full production pipeline: we remove the
-OH from the imino nitrogen (=N-OH -> =NH), name the resulting imidic acid,
then prepend 'N-hydroxy'. This inherits chain/ring/substituent handling from
the imidic path and FAILS CLOSED (returns None) whenever the core cannot be
named (so a wrong name is never emitted).

IUPAC cite:.
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
    """Direct-return hydroximic acid handler (W3-P02,.

    Names the imidic-acid core (=N-OH stripped to =NH) via the full pipeline,
    then prepends 'N-hydroxy'. Returns None on any failure so the cascade
    continues (fail-safe per -04).
    """
    from rdkit import Chem

    from ...errors import is_failure_name
    from ...namer import name_compound
    from ..candidate_pool import get_current_pool

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

    # An -SH / -SeH / -TeH acid ('N-hydroxycyclohexanecarboximidoselenoic acid
    # (PIN)',, the Blue Book): 'N-hydroxy' is put in front of the
    # core name, which is right only when the core cites no prefix of its own
    # (otherwise 'N-hydroxy' belongs among them,. So the acyl group must
    # be a bare unbranched carbon chain or a bare monocycle; else decline.
    acid_x = _mol.GetAtomWithIdx(match[3])
    if acid_x.GetSymbol() != 'O' and not _bare_acyl_group(_mol, match[0], n_idx,
                                                          match[3]):
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


def _bare_acyl_group(mol: Any, c_idx: int, n_idx: int, x_idx: int) -> bool:
    """True when the group on the acid carbon (all atoms reached from it without
    passing its =N- and -XH) is nothing, an unbranched acyclic carbon chain bonded
    by its end, or the atoms of one ring and nothing else -- a parent that cites no
    substituent prefix."""
    seen, stack = set(), [nb.GetIdx() for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                          if nb.GetIdx() not in (n_idx, x_idx)]
    while stack:
        i = stack.pop()
        if i in seen or i in (c_idx, n_idx, x_idx):
            continue
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
    if not seen:
        return True
    ring_info = mol.GetRingInfo()
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in seen):
        return any(set(r) == seen for r in ring_info.AtomRings())
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in seen):
        return False
    deg = {i: sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in seen)
           for i in seen}
    attach = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors()
              if nb.GetIdx() in seen]
    return (len(attach) == 1 and deg[attach[0]] <= 1
            and all(d <= 2 for d in deg.values()))


__all__ = ["name_hydroximic_acid", "_is_hydroximic_acid"]
