"""Thioimide namer (P-66.1.4.2, BB 33172).

A thioimide is the di-thioacyl analogue of an imide: two C(=S) groups bonded
to a common nitrogen, ``R-C(=S)-NH-C(=S)-R'``. The Blue Book PIN names one
thioacyl group as the parent alkanethioamide suffix and cites the other as an
``N-(alkanethioyl)`` acyl substituent prefix (BB verbatim example)::

    CC(=S)NC(C)=S  ->  N-(ethanethioyl)ethanethioamide   (PIN)

This mirrors the shipped O-imide behaviour, which the decomposition engine
produces (``CCC(=O)NC=O -> N-propanoylformamide``): the LESS-senior (shorter)
thioacyl chain becomes the parent thioamide and the MORE-senior (longer) chain
becomes the ``N-(alkanethioyl)`` substituent prefix. The decomposition engine
never reaches the thio case because its amide-bond SMARTS requires ``C(=O)``,
so the C(=S)-N bond is not recognised as a cleavable amide bond; this dedicated
graph classifier fills exactly that gap.

SCOPE (fail-closed, accuracy-first): emit ONLY when
  * exactly two thiocarbonyl carbons C(=S) share one nitrogen (the thioimide N);
  * that nitrogen carries nothing else but the two thioacyl carbons plus H
    (i.e. an N-H thioimide — N-substituted thioimides are NOT built here);
  * each thioacyl carbon's remaining branch is a PLAIN, saturated, acyclic,
    all-carbon alkyl group (no rings, no heteroatoms, no unsaturation, no
    charges/radicals/isotopes) — the ``alkanethioyl`` retained acyl form only
    covers unbranched alkanethioyl chains here.
Anything else returns ``None`` and the dispatch cascade continues; never a
wrong name.

Graph/atom classifier (no SMARTS broadening); pure — no mol mutation.
"""

from collections import deque
from typing import List, Optional

from rdkit import Chem

from .amides import _get_chain_prefix, get_amide_parent_name


def _linear_alkyl_carbon_count(mol, start_c: int, blocked: int) -> Optional[int]:
    """Return the carbon count of a plain unbranched saturated all-carbon alkyl
    branch rooted at ``start_c`` (the thioacyl carbon), walking away from
    ``blocked`` (the thioimide nitrogen). Returns ``None`` if the branch
    contains a ring, a heteroatom, unsaturation, a branch point, or any
    charge/radical/isotope — i.e. anything that is not a plain -C_nH_(2n+1)."""
    # start_c is the thioacyl carbon; its =S and the N are excluded. The chain
    # is the carbons hanging off start_c (start_c itself is carbon 1 of the
    # thioacyl group, e.g. the C of CH3-C(=S)-).
    seen = {start_c, blocked}
    # Exclude the double-bonded sulfur leaf.
    for nb in mol.GetAtomWithIdx(start_c).GetNeighbors():
        if nb.GetSymbol() == 'S':
            bond = mol.GetBondBetweenAtoms(start_c, nb.GetIdx())
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                seen.add(nb.GetIdx())

    count = 1  # start_c is carbon 1
    current = start_c
    while True:
        atom = mol.GetAtomWithIdx(current)
        nexts = [
            nb for nb in atom.GetNeighbors()
            if nb.GetIdx() not in seen and nb.GetAtomicNum() > 1
        ]
        if not nexts:
            break
        if len(nexts) != 1:
            return None  # branch point -> not a plain alkyl chain
        nxt = nexts[0]
        if nxt.GetSymbol() != 'C':
            return None
        if nxt.IsInRing():
            return None
        if nxt.GetFormalCharge() != 0 or nxt.GetNumRadicalElectrons() != 0:
            return None
        if nxt.GetIsotope() != 0:
            return None
        bond = mol.GetBondBetweenAtoms(current, nxt.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return None  # unsaturation -> not alkyl
        count += 1
        seen.add(nxt.GetIdx())
        current = nxt.GetIdx()
    return count


def _covered_atoms(mol, n_idx: int, acyl_cs: List[int]) -> set:
    """All heavy atoms in the R-C(=S)-N-C(=S)-R' unit reachable from N without
    re-crossing N except via the two thioacyl carbons."""
    seen = {n_idx}
    queue = deque(acyl_cs)
    seen.update(acyl_cs)
    while queue:
        idx = queue.popleft()
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nb.GetIdx()
            if j == n_idx or j in seen:
                continue
            if nb.GetAtomicNum() <= 1:
                continue
            seen.add(j)
            queue.append(j)
    return seen


def name_thioimide(mol) -> Optional[str]:
    """Return the ``N-(alkanethioyl)alkanethioamide`` PIN for an acyclic N-H
    thioimide, else ``None`` (fail-closed cascade continuation)."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None

    # No charged / radical / isotopic species anywhere.
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
        if atom.GetIsotope() != 0:
            return None

    # Locate the thioimide nitrogen: an acyclic N bonded to exactly two
    # thiocarbonyl carbons C(=S) and otherwise only H.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'N' or atom.IsInRing():
            continue
        if atom.GetFormalCharge() != 0:
            continue
        carbon_nbrs = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() == 'C']
        # N must carry exactly two carbon neighbours (both thioacyl) and no
        # other heavy atom; the remaining valence is H (N-H thioimide only).
        heavy_nbrs = [nb for nb in atom.GetNeighbors()
                      if nb.GetAtomicNum() > 1]
        if len(carbon_nbrs) != 2 or len(heavy_nbrs) != 2:
            continue

        acyl_cs = []
        ok = True
        for c in carbon_nbrs:
            if c.IsInRing():
                ok = False
                break
            # c must be a thiocarbonyl carbon: exactly one =S plus the
            # thioimide N, and AT MOST one further (alkyl) carbon neighbour.
            # other_c == 0 is the methanethioyl case (H-C(=S)-, e.g.
            # methanethioamide parent); other_c == 1 is every longer chain.
            # No other heteroatom may hang off the thioacyl carbon.
            s_double = 0
            other_c = 0
            bad = False
            for nb in c.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
                if (nb.GetSymbol() == 'S'
                        and bond.GetBondType() == Chem.BondType.DOUBLE):
                    if nb.GetDegree() != 1:  # =S must be a terminal leaf
                        bad = True
                    s_double += 1
                elif nb.GetIdx() == atom.GetIdx():
                    continue  # the thioimide N
                elif nb.GetSymbol() == 'C':
                    other_c += 1
                elif nb.GetAtomicNum() > 1:
                    bad = True  # any other heteroatom on the thioacyl C
            if bad or s_double != 1 or other_c > 1:
                ok = False
                break
            acyl_cs.append(c.GetIdx())
        if not ok:
            continue

        # Each thioacyl branch must be a plain unbranched saturated alkyl.
        lengths = []
        for c_idx in acyl_cs:
            n_carbons = _linear_alkyl_carbon_count(mol, c_idx, atom.GetIdx())
            if n_carbons is None:
                lengths = None
                break
            lengths.append(n_carbons)
        if lengths is None:
            continue

        # Coverage: the whole molecule must be exactly this thioimide unit.
        covered = _covered_atoms(mol, atom.GetIdx(), acyl_cs)
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        if covered != heavy:
            continue

        # Parent = the LESS-senior (shorter) thioacyl chain -> alkanethioamide;
        # N-acyl substituent = the MORE-senior (longer) chain -> alkanethioyl.
        # Mirrors the shipped O-imide decomposition convention
        # (CCC(=O)NC=O -> N-propanoylformamide: formamide parent, propanoyl sub).
        len_a, len_b = lengths
        parent_len = min(len_a, len_b)
        sub_len = max(len_a, len_b)
        parent = get_amide_parent_name(parent_len, suffix_form="thioamide")
        acyl = f"{_get_chain_prefix(sub_len)}anethioyl"
        return f"N-({acyl}){parent}"

    return None
