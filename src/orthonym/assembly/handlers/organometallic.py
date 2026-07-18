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

    Per CONTEXT D-11: returns Optional[NamingResult] with tree populated
    when result is non-None. The CFR shim _handle_organometallic in
    routing/dispatch_table.py extracts result.name as Optional[str].

    Tier-1 fast path: retained-PIN lookup for style="pin" — ferrocene,
    ruthenocene, etc. Tier-1 systematic + Tier-2/3/4 forms go through
    the systematic-assembly path.

    Cascade-continuation on None per CONTEXT D-02: any failure (mol is
    None; metal_complex is None; multimetal; ValueError from hapticity;
    result is None) returns None so CFR cascade falls through to
    SALT@100 → ... → GENERAL@99999.
    """
    from rdkit import Chem

    if mol is None:
        return None

    # Lazy imports avoid circular dependency at module import time.
    from ...perception.metals import detect_metal_complex, detect_metallacycle
    from ...rules.organometallics import (
        assemble_organometallic_name, _assemble_metallacycle,
    )
    from ...data.organometallics import RETAINED_METALLOCENES

    # W8-P9 Task 9.5 (P-69.4): a metal RING atom (metallacycle) is checked
    # FIRST — detect_metal_complex's Tier-3 sigma-ligand walker would
    # otherwise try to fold the whole ring backbone into one "ligand"
    # fragment (a topology _ligand_name_from_atoms cannot name), returning
    # None and cascading past this handler entirely (the Pt-metallacycle
    # atom-drop leak; backstopped by the Task 9.2 veto, but the metallacycle
    # namer below now supplies the correct name instead of just failing
    # closed).
    metallacycle_info = detect_metallacycle(mol)
    if metallacycle_info is not None:
        try:
            mc_result = _assemble_metallacycle(metallacycle_info, mol, style=style)
        except ValueError:
            return None
        if mc_result is None:
            return None  # narrow builder declined -- cascade (Task 9.2 backstops)
        full_name, _metal_part, _tree_nodes = mc_result
        tree = NameTreeNode(
            parent_stem=full_name,
            class_id='organometallic',
            iupac_section_cite='P-69.4 (skeletal replacement)',
        )
        return NamingResult(name=full_name, tree=tree, atom_to_locant_hint=None)

    metal_complex = detect_metal_complex(mol)
    if metal_complex is None:
        return None
    if metal_complex.is_multimetal:
        return None  # Risk R-08: Phase 161.3 territory

    # Tier-1 fast path: retained PIN lookup for style="pin"
    canon_smi = Chem.MolToSmiles(mol)
    if style == "pin" and canon_smi in RETAINED_METALLOCENES:
        retained_name = RETAINED_METALLOCENES[canon_smi]
        tree = NameTreeNode(
            parent_stem=retained_name,
            class_id='organometallic',
            iupac_section_cite='P-69 / Salzer §5.4',
        )
        return NamingResult(name=retained_name, tree=tree, atom_to_locant_hint=None)

    # Systematic assembly path (also covers style="pin" for compounds
    # without retained PIN, e.g. Tier-2/3/4 + Tier-1 systematic style)
    try:
        result = assemble_organometallic_name(metal_complex, mol, style=style)
    except ValueError:
        # Per CONTEXT D-12 honest-fail-on-data: hapticity/naming failure
        # cascades to next CFR entry.
        return None

    if result is None:
        return None  # cascade to SALT@100

    full_name, _metal_name_part, _ligand_tree_nodes = result
    # Use flat tree with full name as parent_stem to satisfy the byte-identical
    # round-trip contract (name_tree_to_string(tree) == result.name).
    # CONTEXT D-11 allows the metal_name_part + prefixes structure too; the
    # flat representation is the minimal compliant form. Sub-tree structure
    # can be refined in Phase 161.1+ if downstream consumers need it.
    tree = NameTreeNode(
        parent_stem=full_name,
        class_id='organometallic',
        iupac_section_cite='P-69 / IR-10 / Salzer 1999',
    )
    return NamingResult(name=full_name, tree=tree, atom_to_locant_hint=None)


__all__ = ["name_organometallic"]
