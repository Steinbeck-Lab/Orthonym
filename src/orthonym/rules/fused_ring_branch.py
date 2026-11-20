"""An acyclic branch on a fused ring core that no element branch of ``fused_rings._identify_fused_substituent`` knows.

The identifier knew the sulfanyl and alkylsulfanyl branches on a sulfur root and sent every ring-bearing branch to the
ring delegate; any other branch returned None, and its caller skipped it (``continue``), so the name lost the branch and
only the read-back stopped it: '1-(methanesulfonyl)-N-methyl-1H-indole-5-carboxamide' got no name at the PIN tier.
The branch is named here by the shared substituent namer (``substituent_enumerator.name_substituent``: 'methanesulfonyl',
'tert-butoxycarbonyl', acyl groups of sulfonic acids,:31285 '2-(methanesulfonyl)ethan-1-ol (PIN)' form);
its failure placeholder ('substituent', ``errors.CASCADE_PLACEHOLDER``) is no name. At the PIN tier the name is kept only
through ``pin_vocabulary.promote_at_pin_tier``, as for the ring delegate."""
from typing import Dict, Optional, Set


def acyclic_branch_record(mol, start_idx: int, excluded: Set[int]) -> Optional[Dict]:
    """``{'name', 'type': 'functionalized', 'atoms'}`` for the acyclic branch rooted at ``start_idx`` (every atom outside
    a ring, at most 25 atoms), or None."""
    # only a branch on a ring NITROGEN (an N-sulfonyl or N-acyl group, cited as a prefix with its ring locant): on a
    # ring carbon the branch may hold the principal group, which this prefix would demote (panel 2026-11-20:
    # '3-(methylcarbamoyl)-...-1-benzofuran' for '...-1-benzofuran-3-carboxamide')
    if not any(nb.GetIdx() in excluded and nb.GetSymbol() == 'N' for nb in mol.GetAtomWithIdx(start_idx).GetNeighbors()):
        return None
    from .fused_rings import _bfs_collect_all
    frag = _bfs_collect_all(mol, start_idx, excluded)
    if not frag or len(frag) > 25 or any(mol.GetAtomWithIdx(i).IsInRing() for i in frag):
        return None
    from ..assembly.substituent_enumerator import name_substituent
    from ..errors import CASCADE_PLACEHOLDER
    from ..metrics.provenance import best_effort_ctx
    from .pin_vocabulary import promote_at_pin_tier

    def named():
        name = name_substituent(mol, sorted(frag), start_idx)
        return None if not name or name == CASCADE_PLACEHOLDER or ' ' in name else name
    name = named() if best_effort_ctx.get() else promote_at_pin_tier(named)
    return {'name': name, 'type': 'functionalized', 'atoms': sorted(frag)} if name else None
