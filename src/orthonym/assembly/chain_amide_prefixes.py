"""Detachable amide-type prefixes of an acyclic chain substituent (lane/l17 item 12).

``assembly.substituent_naming`` names a chain substituent from its bonds: it enumerates the groups on the chain,
numbers the chain from the free valence and cites each group as a prefix,. Two of its passes for a
pendant amide-type group live here, each a function of the molecule and the atoms handed in:

*:func:`consume_acylated_nitrogen` -- a pendant acylated nitrogen -N(R')-CO-R or -N(R')-SO2-R on a chain carbon is
  the method-(1) amido prefix of (the Blue Book), like the same group on a ring or a parent chain;
*:func:`consume_sulfamoyl` -- a sulfonamide -SO2-NR2 on a chain carbon, moved from the producer unchanged.
"""
from __future__ import annotations

from typing import Set

from rdkit import Chem


def consume_acylated_nitrogen(mol, sub_set, attach_idx, consumed, add_prefix) -> None:
    """Consume every pendant acylated nitrogen of the chain substituent ``sub_set`` as an amido prefix on its host
    carbon (``n_acyl_amido.chain_acylated_n_prefix``): the group's atoms join ``consumed`` and the prefix, enclosed
    when it is compound, the Blue Book; marks escalate by,:7444: '(N-methylacetamido)',
    but the simple 'acetamido' bare, '(acetamidomethyl)'), is handed to ``add_prefix(host, prefix)``. None is consumed in
    a molecule with a dipeptide unit, nor in a fragment the engine cut from one (``n_acyl_amido.in_peptide``, which says what
    the scoping is and what it was measured to change)."""
    from .n_acyl_amido import chain_acylated_n_prefix, in_peptide
    from .naming_utils import apply_enclosing_marks, needs_brackets
    attach = mol.GetAtomWithIdx(attach_idx)
    if attach.GetAtomicNum() != 6:
        return  # a chain substituent has its free valence on a carbon
    if any(b.GetBondTypeAsDouble() == 2.0 for b in attach.GetBonds()) and not _is_ketone_acyl(mol, sub_set, attach):
        return  # an acyl group (a C=O) has its own prefix tiers; only a ketone's acyl cites the amido prefix on its chain
    if in_peptide(mol):
        return
    for n_idx in sorted(sub_set):
        found = chain_acylated_n_prefix(mol, sub_set, attach_idx, n_idx, consumed)
        if found is None:
            continue
        prefix, group, host = found
        consumed.update(group)
        add_prefix(host, apply_enclosing_marks(prefix, -1) if needs_brackets(prefix) else prefix)


def _is_ketone_acyl(mol, sub_set, attach) -> bool:
    """True when ``attach``, the free-valence carbon of a chain substituent, is the carbonyl carbon of an acyl group
    on a carbon (a ketone,; its parent-side neighbour is a carbon) whose chain continues on a carbon:
    '(N-methylacetamido)acetyl' "substitution on acyl groups is identical to that of acids",
    "Substituents of the types -NH-CO-R and -NH-SO2-R", ``acyl_prefix.acyl_prefix_text`` writes the acyl name from the
    prefixes the chain holds). An acyl group on N or O (an amide, an ester) keeps its own prefix tiers."""
    outside = [n for n in attach.GetNeighbors() if n.GetIdx() not in sub_set]
    inside = [n for n in attach.GetNeighbors() if n.GetIdx() in sub_set]
    oxo = [n for n in inside if n.GetAtomicNum() == 8 and n.GetDegree() == 1
           and mol.GetBondBetweenAtoms(attach.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2.0]
    chain = [n for n in inside if n.GetAtomicNum() == 6 and not n.IsInRing()]
    return (len(outside) == 1 and outside[0].GetAtomicNum() == 6 and len(oxo) == 1 and len(chain) == 1
            and len(inside) == 2 and attach.GetDegree() == 3)


def consume_sulfamoyl(mol, sub_set, attach_idx, consumed, _add_prefix, _ring_info_pf,
                     carbon_free_valence_prefix, composed_prefix_organyl_name) -> None:
    """Moved unchanged from ``substituent_naming._name_polyfunctional_acyclic_substituent_impl`` (its pass 1f); the
    producer's own helpers are handed in, so this module imports nothing from it.

    Pass 1f (a phase E2c, /, WRONG-MOLECULE
    RISK): a sulfonamide -S(=O)(=O)-NH2/-NHR/-NR2 hanging off a backbone
    carbon is the retained 'sulfamoyl' prefix, or its N-substituted
    '(R-sulfamoyl)'/'(dialkylsulfamoyl)' form -- '(methylsulfamoyl)methyl'
    for -CH2-SO2-NHCH3. Consumed HERE so it can never reach the generic
    recursive Tier-4 path + `parent_to_prefix`'s '-amide' string transform,
    which reads 'methanesulfonamide' as if it ended in the CARBOXAMIDE
    suffix and COLLAPSES THE SULFONYL TO 'carbamoyl' -- a WRONG
    CONSTITUTION (measured: '-CH2-SO2-NHCH3' -> the wrong 'carbamoylmethyl',
    a different molecule must catch). Reuses
    `rules.benzene._build_n_substituted_sulfamoyl_prefix` (the ring-side
    '{N-subs}sulfamoyl' builder) rather than re-deriving it, for the
    CHAIN-side attachment. Fail-closed (skip this S) for: a ring-borne
    sulfonyl, a charged/hypervalent/isotope-labelled S, a ring amide N,
    DISTINCT N,N-disubstituents (the shared builder's own conservative
    envelope -- nested 'ethyl(methyl)sulfamoyl' is not built), or any shape
    outside the plain sulfonamide envelope (a bridging N, a branch this
    project's organyl namer cannot express).
    """
    for idx in list(sub_set):
        if idx in consumed:
            continue
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() != 'S' or a.GetFormalCharge() != 0 or a.GetIsotope():
            continue
        if _ring_info_pf.NumAtomRings(idx) > 0:
            continue
        nbrs = list(a.GetNeighbors())
        if len(nbrs) != 4 or any(n.GetIdx() not in sub_set for n in nbrs):
            continue
        dbl_o = []
        host_c = None
        n_idx = None
        ok = True
        for n in nbrs:
            ni = n.GetIdx()
            b = mol.GetBondBetweenAtoms(idx, ni)
            if (n.GetSymbol() == 'O' and b.GetBondType() == Chem.BondType.DOUBLE
                    and n.GetFormalCharge() == 0 and n.GetDegree() == 1):
                dbl_o.append(ni)
            elif (n.GetSymbol() == 'C' and b.GetBondType() == Chem.BondType.SINGLE
                    and host_c is None):
                host_c = ni
            elif (n.GetSymbol() == 'N' and b.GetBondType() == Chem.BondType.SINGLE
                    and n_idx is None):
                n_idx = ni
            else:
                ok = False
                break
        if not ok or len(dbl_o) != 2 or host_c is None or n_idx is None:
            continue
        if host_c in consumed:
            continue
        n_atom = mol.GetAtomWithIdx(n_idx)
        if (_ring_info_pf.NumAtomRings(n_idx) > 0
                or n_atom.GetFormalCharge() != 0 or n_atom.GetIsotope()):
            continue
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in n_atom.GetBonds()):
            continue
        n_branches = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                      if nb.GetIdx() != idx]
        if any(b not in sub_set for b in n_branches):
            continue

        def _s_branch_component(seed, _block=None):
            _blocked = _block or {idx, n_idx}
            comp: Set[int] = set()
            st = [seed]
            while st:
                x = st.pop()
                if x in comp or x in _blocked or x not in sub_set:
                    continue
                comp.add(x)
                for n in mol.GetAtomWithIdx(x).GetNeighbors():
                    nj = n.GetIdx()
                    if nj not in _blocked and nj not in comp:
                        st.append(nj)
            return comp

        names = []
        branch_atoms: Set[int] = set()
        ok2 = True
        for b0 in n_branches:
            comp = _s_branch_component(b0)
            if host_c in comp or attach_idx in comp:
                ok2 = False
                break
            if any(mol.GetAtomWithIdx(v).GetFormalCharge() != 0 for v in comp):
                ok2 = False
                break
            _fv = carbon_free_valence_prefix(mol, comp, b0)
            nm = _fv.prefix
            if nm is None:
                if _fv.must_fail_closed:
                    ok2 = False
                    break
                nm = composed_prefix_organyl_name(mol, comp, b0)
            if nm is None:
                ok2 = False
                break
            names.append(nm)
            branch_atoms |= comp
        if not ok2:
            continue
        if not names:
            sulfamoyl_pfx = 'sulfamoyl'
        else:
            from ..rules.benzene import _build_n_substituted_sulfamoyl_prefix
            sulfamoyl_pfx = _build_n_substituted_sulfamoyl_prefix(names)
        if not sulfamoyl_pfx:
            continue
        consumed.add(idx)
        consumed.update(dbl_o)
        consumed.add(n_idx)
        consumed.update(branch_atoms)
        _add_prefix(host_c, sulfamoyl_pfx)
