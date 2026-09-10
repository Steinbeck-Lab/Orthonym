""": sulfur-oxoacid acyl-oxy / -amino substituent prefixes.

When a sulfur oxoacid group is attached BY OXYGEN (chalcogen) OR NITROGEN to a
compound that also carries a characteristic group senior to the sulfur acid for
citation as principal group (e.g. a carboxylic acid), the sulfur group is cited
as a substituent PREFIX, not as the parent
, ``the Blue Book Blue Book``):

    3-(sulfooxy)propanoic acid (PIN):36488
    3-[(methoxysulfinyl)oxy]propanoic acid (PIN):36490
    3-[(chlorosulfonyl)oxy]propanoic acid (PIN):36492
    3-(sulfamoyloxy)propanoic acid (PIN):36494
    3-[(aminosulfinyl)oxy]propanoic acid (PIN):36500 (NOT sulfinamoyloxy)
    3-[(methoxysulfonyl)amino]propanoic acid (PIN):36502

The generic path names this tail by skeletal ("a") replacement
(``…-1,3-dioxa-2λ6-thiapropyl``) — a valid, round-tripping, but NON-PIN form.

The S-oxoacid acyl group is ``{X}sulfonyl`` (n S=O double bonds = 2) or
``{X}sulfinyl`` (n = 1), where X is the single S ligand that is neither the
linking O/N nor an oxo. Two Blue Book contractions apply / the
substituent-prefix tables at:56857,:56859): ``hydroxysulfonyl`` -> ``sulfo``
and ``aminosulfonyl`` -> ``sulfamoyl``. ``aminosulfinyl`` is NOT contracted
(explicitly ``[not …sulfinamoyloxy…]`` at:36500).

Every function fails closed (returns ``None``) on any S outside the neutral
mono/di-oxo acyl class (charged/radical S, ring S, an unexpected ligand set,
a carbon-R sulfonyl, a plain thioether); the top-level /OPSIN round-trip
gate keeps 0-wrong on whatever is emitted.
"""
from typing import List, Optional

from rdkit import Chem

# X ligand -> substituent-prefix stem ligand set)
_HALOGEN = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}


def _collect_subtree(mol, start: int, exclude: int) -> List[int]:
    """Atom indices reachable from ``start`` without crossing ``exclude``."""
    seen = {start}
    stack = [start]
    while stack:
        i = stack.pop()
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j == exclude or j in seen:
                continue
            seen.add(j)
            stack.append(j)
    return list(seen)


def _ligand_prefix(mol, x_idx: int, s_idx: int) -> Optional[str]:
    """Name the single non-oxo, non-linker S ligand X as a substituent stem.

    ``-OH`` -> ``hydroxy``, ``-Cl`` -> ``chloro``, ``-O-CH3`` -> ``methoxy``
    (alkoxy via the ordinary substituent cascade), ``-NH2`` -> ``amino``.
    Returns ``None`` (fail closed) for anything else (a carbon-R sulfonyl, a
    substituted amino, a charged ligand,...).
    """
    a = mol.GetAtomWithIdx(x_idx)
    if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons():
        return None
    z = a.GetAtomicNum()
    if z in _HALOGEN and a.GetDegree() == 1:
        return _HALOGEN[z]
    if a.GetSymbol() == "O":
        if a.GetDegree() == 1 and a.GetTotalNumHs() == 1:
            return "hydroxy"
        if a.GetTotalNumHs() == 0:
            others = [n.GetIdx() for n in a.GetNeighbors()
                      if n.GetIdx() != s_idx]
            if len(others) == 1 and mol.GetAtomWithIdx(others[0]).GetSymbol() == "C":
                # -O-alkyl: name the O-alkyl subtree as an oxy prefix
                # (methoxy / ethoxy / …) via the ordinary cascade.
                from ..assembly.substituent_enumerator import name_substituent
                sub = _collect_subtree(mol, x_idx, s_idx)
                nm = name_substituent(mol, sub, x_idx, True)
                if (nm and nm.endswith("oxy")
                        and "(" not in nm and " " not in nm):
                    return nm
        return None
    if a.GetSymbol() == "N":
        if a.GetDegree() == 1 and a.GetTotalNumHs() == 2:
            return "amino"
        return None
    return None


def _sulfur_oxoacid_acyl(mol, s_idx: int, link_idx: int) -> Optional[str]:
    """Return the S-oxoacid acyl prefix for an S bonded to ``link_idx``.

    ``link_idx`` (the O or N linker) is excluded from the ligand walk. The S
    must be neutral, non-radical, acyclic, carry exactly ``n in (1, 2)`` terminal
    ``=O`` groups and exactly ONE further single-bonded ligand X. Returns e.g.
    ``sulfo`` / ``chlorosulfonyl`` / ``sulfamoyl`` / ``aminosulfinyl`` /
    ``methoxysulfonyl`` / ``methoxysulfinyl``; ``None`` if out of class.
    """
    s = mol.GetAtomWithIdx(s_idx)
    if (s.GetSymbol() != "S" or s.GetFormalCharge() != 0
            or s.GetNumRadicalElectrons() or s.IsInRing()):
        return None
    n_oxo = 0
    x_ligands: List[int] = []
    for nb in s.GetNeighbors():
        if nb.GetIdx() == link_idx:
            continue
        bond = mol.GetBondBetweenAtoms(s_idx, nb.GetIdx())
        bt = bond.GetBondType()
        if (nb.GetSymbol() == "O" and bt == Chem.BondType.DOUBLE
                and nb.GetDegree() == 1 and nb.GetFormalCharge() == 0):
            n_oxo += 1
            continue
        if bt == Chem.BondType.SINGLE:
            x_ligands.append(nb.GetIdx())
            continue
        return None  # any other bond (S=N, S#…) -> out of class
    if n_oxo not in (1, 2) or len(x_ligands) != 1:
        return None
    xp = _ligand_prefix(mol, x_ligands[0], s_idx)
    if xp is None:
        return None
    acyl = xp + ("sulfonyl" if n_oxo == 2 else "sulfinyl")
    # contractions (the -sulfinyl forms are NOT contracted::36500).
    if acyl == "hydroxysulfonyl":
        return "sulfo"
    if acyl == "aminosulfonyl":
        return "sulfamoyl"
    return acyl


def name_sulfur_oxoacid_oxy_substituent(
        mol, o_idx: int, from_idx: int) -> Optional[str]:
    """O-linked ``-O-S(oxoacid)`` as an oxy substituent prefix.

    ``o_idx`` is the ester/bridging oxygen (the fragment attach atom);
    ``from_idx`` is its parent-side neighbour. Returns ``sulfooxy`` /
    ``sulfamoyloxy`` (single compound tokens, or ``(chlorosulfonyl)oxy``
    / ``(aminosulfinyl)oxy`` / ``(methoxysulfinyl)oxy`` (an internal paren the
    caller's enclosure escalates to brackets), or ``None`` (fail closed).
    """
    o = mol.GetAtomWithIdx(o_idx)
    if (o.GetSymbol() != "O" or o.GetFormalCharge() != 0
            or o.GetTotalNumHs() != 0 or o.IsInRing() or o.GetDegree() != 2):
        return None
    s_nbrs = [n.GetIdx() for n in o.GetNeighbors()
              if n.GetIdx() != from_idx and n.GetSymbol() == "S"]
    if len(s_nbrs) != 1:
        return None
    acyl = _sulfur_oxoacid_acyl(mol, s_nbrs[0], o_idx)
    if acyl is None:
        return None
    if acyl in ("sulfo", "sulfamoyl"):
        return acyl + "oxy"          # sulfooxy / sulfamoyloxy
    return f"({acyl})oxy"            # (chlorosulfonyl)oxy, …


def name_sulfur_oxoacid_acyl_for_amino(
        mol, s_idx: int, frag_atoms: set) -> Optional[str]:
    """N-linked ``-NH-S(oxoacid)``: the acyl prefix for the amino wrapper.

    Reached when the S-oxoacid attaches through an amine N (the amino-branch
    caller has already split off the N as the linker and recursed into the S
    fragment, so ``s_idx`` is the fragment attach atom whose single EXTERNAL
    neighbour, outside ``frag_atoms``, is that N). Returns e.g.
    ``methoxysulfonyl`` for the caller to wrap as ``(methoxysulfonyl)amino``,
    or ``None`` (fail closed) if the external linker is not a neutral N or the S
    is out of class.
    """
    s = mol.GetAtomWithIdx(s_idx)
    ext = [n.GetIdx() for n in s.GetNeighbors() if n.GetIdx() not in frag_atoms]
    if len(ext) != 1:
        return None
    n_atom = mol.GetAtomWithIdx(ext[0])
    if n_atom.GetSymbol() != "N" or n_atom.GetFormalCharge() != 0:
        return None
    return _sulfur_oxoacid_acyl(mol, s_idx, ext[0])
