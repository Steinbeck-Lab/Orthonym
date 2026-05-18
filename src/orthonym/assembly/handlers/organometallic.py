"""Phase 161 organometallic handler — P-69 + IR-10 + Salzer 1999.

SECOND tree-emitting handler in Orthonym (first was simple_molecule.py
from Phase 160.2 Plan-10). Mirrors that file's shape per CONTEXT D-04 +
D-11 + 161-AUDIT-ORGM.md § 5.

The handler is OUTER-CFR-only (per ORGM-03 + Phase 158 CFR-02): it does
NOT add an inner-dispatch entry; the CFR-level ORGANOMETALLIC entry at
priority 50 routes here BEFORE SALT@100.

Byte-identical contract per CONTEXT D-11: name_tree_to_string(result.tree)
== result.name for ALL non-None returns.

Cascade-continuation on None preserved per CONTEXT D-02: any failure
(mol is None; metal_complex is None; multimetal compound; ValueError
from hapticity; result is None) returns None so CFR cascade falls
through to SALT@100 → ... → GENERAL@99999.

Anti-patterns to avoid (PATTERNS lines 499-503):
- NEVER mutate features or mol inside the handler — predicate + handler
  purity (D-12 hard invariant).
- NEVER catch ALL exceptions; only catch ValueError from compute_hapticity
  per RESEARCH §3.2 lines 540-544 to signal cascade-continuation.
- NEVER emit a tree whose name_tree_to_string(tree) does NOT byte-equal
  result.name — CONTEXT D-11 + RESEARCH §4.3 line 757 byte-identical contract.
"""

from __future__ import annotations
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def name_organometallic(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 161 ORGM handler; CFR-routed at priority 50.

    STUB BODY (Plan-02): returns None always — cascade falls through to
    SALT@100 → existing pipeline. Plan-03 implements per-tier branches
    per RESEARCH §4.3 lines 765-815.

    Per CONTEXT D-11: returns Optional[NamingResult] with tree populated
    when result is non-None. The CFR shim _handle_organometallic in
    routing/dispatch_table.py extracts result.name as Optional[str].
    """
    # Plan-03 implementation per RESEARCH §4.3 + AUDIT § 5:
    # 1. Lazy-import detect_metal_complex, RETAINED_METALLOCENES,
    #    assemble_organometallic_name.
    # 2. Guard: if mol is None: return None.
    # 3. metal_complex = detect_metal_complex(mol); if None: return None.
    # 4. Risk R-08 guard: if metal_complex.is_multimetal: return None
    #    (Phase 161.3 territory).
    # 5. Tier-1 fast path: canon_smi = Chem.MolToSmiles(mol);
    #    if style=='pin' and canon_smi in RETAINED_METALLOCENES:
    #      return NamingResult(name=retained, tree=NameTreeNode(
    #        parent_stem=retained, class_id='organometallic',
    #        iupac_section_cite='P-69 / Salzer §5.4'),
    #        atom_to_locant_hint=None).
    # 6. Systematic path: try assemble_organometallic_name(metal_complex,
    #    mol, style=style); on ValueError or None result, return None.
    # 7. Build NameTreeNode(parent_stem=metal_name_part,
    #    prefixes=tuple(ligand_tree_nodes), class_id='organometallic',
    #    iupac_section_cite='P-69 / IR-10 / Salzer 1999').
    # 8. Return NamingResult(name=name, tree=tree, atom_to_locant_hint=None).
    return None


__all__ = ["name_organometallic"]
