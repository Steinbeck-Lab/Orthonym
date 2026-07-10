"""BBR-PERC (Phase 169.7) hydroxylamine handler.

Names substituted hydroxylamines on the retained parent hydride ``hydroxylamine``
(H2N-OH, P-68.3.1.1). Substituents on the nitrogen take the ``N-`` locant; those
on the oxygen take ``O-`` (P-68.3.1.2.1):

    CCCNO        -> N-propylhydroxylamine
    CCCN(O)C     -> N-methyl-N-propylhydroxylamine
    CON          -> O-methylhydroxylamine
    CN(C)O       -> N,N-dimethylhydroxylamine

Without this handler the perceived ``hydroxylamine`` FG (added to
``functional_groups.py`` in 169.7) is dropped and the molecule names as the bare
carbon chain (``CCCNO`` -> ``propane``), silently losing both heteroatoms.

The substituent names come from the shared ``name_substituent_fragment`` namer;
multiplying affixes + alphanumeric ordering reuse the standard prefix helpers.

IUPAC cite: P-68.3.1.2.1.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, List, Optional, Set, Tuple

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_hydroxylamine(features: Any) -> bool:
    """Fire when hydroxylamine is the principal characteristic group, OR when the
    ONLY functional group is 'aminooxy' (H2N-O-R) and the whole molecule is a
    hydroxylamine derivative with no more-senior parent (P-68.3.1.1.1.2: the
    O-substituted-N-bare form's PIN is O-substituted hydroxylamine, e.g.
    CON -> O-methylhydroxylamine, NOT the prefix 'aminooxy...' which applies only
    when a senior parent is present)."""
    if getattr(features, "principal_group", None) == "hydroxylamine":
        return True
    return _is_pure_aminooxy_hydroxylamine(features)


def _is_pure_aminooxy_hydroxylamine(features: Any) -> bool:
    """True when the molecule is exactly a hydroxylamine derivative perceived
    only as 'aminooxy' (H2N-O-R): a single N-O bond, N and O neutral acyclic,
    every other heavy atom reachable through a carbon substituent, and NO other
    functional group / no more-senior PCG. Fail-closed for anything richer."""
    if getattr(features, "principal_group", None) is not None:
        return False
    fgs = getattr(features, "functional_groups", None) or {}
    active = [k for k, v in fgs.items() if v]
    if active != ["aminooxy"]:
        return False
    mol = getattr(features, "mol", None)
    if mol is None:
        return False
    # Exactly one N and one O forming the hydroxylamine core, both neutral,
    # acyclic, single-bonded to each other; no ring anywhere; no other hetero.
    n_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    o_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "O"]
    if len(n_atoms) != 1 or len(o_atoms) != 1:
        return False
    for a in mol.GetAtoms():
        if a.GetSymbol() not in ("C", "N", "O"):
            return False
        if a.IsInRing() or a.GetFormalCharge() != 0:
            return False
    n_idx, o_idx = n_atoms[0].GetIdx(), o_atoms[0].GetIdx()
    bond = mol.GetBondBetweenAtoms(n_idx, o_idx)
    from rdkit import Chem
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    # No C=O / C=N (would be a senior oxime/amide territory).
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            syms = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
            if syms & {"N", "O"}:
                return False
    return True


def _collect_substituent_fragment(mol, start_idx: int, block_idx: int) -> List[int]:
    """BFS the substituent fragment rooted at ``start_idx``, never crossing
    ``block_idx`` (the hydroxylamine N or O)."""
    seen: Set[int] = set()
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        if idx in seen or idx == block_idx:
            continue
        seen.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx != block_idx and nidx not in seen:
                queue.append(nidx)
    return sorted(seen)


def _named_substituents(mol, center_idx: int, other_center_idx: int) -> List[str]:
    """Name every carbon-rooted substituent on the hydroxylamine center atom
    ``center_idx`` (N or O), excluding the bond to ``other_center_idx``."""
    from ..substituent_naming import name_substituent_fragment

    names: List[str] = []
    center = mol.GetAtomWithIdx(center_idx)
    for nbr in center.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == other_center_idx:
            continue
        if nbr.GetSymbol() != "C":
            continue  # only carbon substituents take a locant; H is the parent
        frag = _collect_substituent_fragment(mol, nidx, center_idx)
        sub_name = name_substituent_fragment(mol, frag, nidx, [center_idx])
        if sub_name:
            names.append(sub_name)
    return names


def _format_locant_block(names: List[str], locant: str) -> List[Tuple[str, str]]:
    """Group identical substituents into ``(sort_key, "locant,locant-prefixname")``
    tuples with the simple multiplying affix (di/tri/...)."""
    from ..naming_utils import SIMPLE_MULTIPLIERS  # {2: 'di', 3: 'tri', ...}

    out: List[Tuple[str, str]] = []
    # Count duplicates preserving first-seen order.
    counts: dict = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    for sub_name, count in counts.items():
        locs = ",".join([locant] * count)
        if count > 1:
            mult = SIMPLE_MULTIPLIERS.get(count, "")
            term = f"{locs}-{mult}{sub_name}"
        else:
            term = f"{locant}-{sub_name}"
        out.append((sub_name, term))
    return out


def name_hydroxylamine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Name a substituted hydroxylamine on the ``hydroxylamine`` parent."""
    m = features.mol
    matches = features.functional_groups.get("hydroxylamine", [])
    if matches:
        # SMARTS [OX2H1][NX3...][#6] -> match[0]=O, match[1]=N.
        match = matches[0]
        o_idx, n_idx = match[0], match[1]
    else:
        # Pure aminooxy whole-molecule case (H2N-O-R -> O-substituted
        # hydroxylamine, P-68.3.1.1.1.2): locate the single N-O core.
        amx = features.functional_groups.get("aminooxy", [])
        if not amx or not _is_pure_aminooxy_hydroxylamine(features):
            return None
        # SMARTS [NX3H2][OX2][#6] -> match[0]=N, match[1]=O.
        match = amx[0]
        n_idx, o_idx = match[0], match[1]
    if m.GetAtomWithIdx(o_idx).GetSymbol() != "O" or m.GetAtomWithIdx(n_idx).GetSymbol() != "N":
        return None

    n_subs = _named_substituents(m, n_idx, o_idx)   # N-locant
    o_subs = _named_substituents(m, o_idx, n_idx)   # O-locant
    if not n_subs and not o_subs:
        return None  # bare hydroxylamine has no carbon substituents — leave to default

    # Alphanumeric order on the substituent name (P-14.5.2); each term already
    # carries its leading O-/N- locant, so adjacent terms hyphen-join and the parent
    # attaches directly (e.g. "N-ethyl-N-methylhydroxylamine", "N-propylhydroxylamine").
    terms = _format_locant_block(o_subs, "O") + _format_locant_block(n_subs, "N")
    terms.sort(key=lambda t: t[0].lstrip("([{").lower())
    name = "-".join(t[1] for t in terms) + "hydroxylamine"

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "HANDLER_COVERAGE: handler=hydroxylamine coverage=NA accounted=NA/%d name=%s",
            m.GetNumHeavyAtoms(), name[:60],
        )

    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    pool = get_current_pool()
    pool.add(name, "hydroxylamine", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem="hydroxylamine", fragment_legacy=final_name,
            class_id="hydroxylamine", iupac_section_cite="P-68.3.1.2.1",
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_hydroxylamine", "_is_hydroxylamine"]
