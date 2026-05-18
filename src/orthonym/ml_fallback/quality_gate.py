"""Quality-gate predicate for Phase 162 ML Fallback Gate.

Implements ``is_name_quality_inadequate(name, mol, canonical_smiles, *,
opsin_parse_required=True) -> bool`` per CONTEXT D-04 + audit section 2
verbatim. Returns ``True`` iff the rule-based pipeline output is degraded
enough to justify ML attachment per MLF-06.

The 5 patterns from CONTEXT D-04 + 1 added per RESEARCH R-03 (cheap
fast-path over the OPSIN subprocess):

* **P1** ``name is None``                                            (~1 µs)
* **P2** ``name == canonical_smiles``                                (~1 µs)
* **P3** any token in ``_GARBLED_TOKENS`` substring of ``name``     (<5 µs)
* **P4** ``len(name) < 6 AND mol.GetNumHeavyAtoms() > 15``          (<10 µs)
* **P6** ``name in _DESCRIPTIVE_FALLBACK_NAMES`` (ADD per R-03)      (<1 µs)
* **P5** ``opsin_parse_required AND not opsin_parse_ok(name)``    (~1269 ms)

Pattern ordering is cheap-first, P5 always last and gated by
``opsin_parse_required`` so the expensive OPSIN subprocess only runs when
needed.

References:

* CONTEXT D-04 (5 patterns), D-08 (opsin_parse_required toggle),
  D-12 (predicate purity hard invariant)
* 162-AUDIT-MLF.md § 2 (full predicate spec + Pattern 6 ADD rationale)
* namer.py:1180 (source for ``_GARBLED_TOKENS``)
* namer.py:1940-1998 (source for ``_DESCRIPTIVE_FALLBACK_NAMES``)

Plan-04 ``tests/unit/ml_fallback/test_quality_gate.py`` asserts every
pattern with explicit fixtures + purity (no mol mutation, deterministic
re-call).
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem  # noqa: F401  (typing only)


__all__ = [
    "is_name_quality_inadequate",
    "_GARBLED_TOKENS",
    "_DESCRIPTIVE_FALLBACK_NAMES",
]


# Per CONTEXT D-04 pattern 3 + namer.py:1180 — kept in sync as a tuple constant.
# Mutating this would violate CONTEXT D-12 (no module-global state R/W).
_GARBLED_TOKENS: Tuple[str, ...] = ("cycloane", "anedicarboxamide", "aneyl")


def _build_descriptive_fallback_names() -> FrozenSet[str]:
    """Build the descriptive-fallback name set at module-import time.

    Reads ``_METAL_NAMES`` dict from ``orthonym.namer`` to dynamically
    extend the base 3-string set with per-metal fallback strings (e.g.
    ``"platinum compound (not supported)"``).

    Per RESEARCH R-03 + 162-AUDIT-MLF.md § 2.2: P6 is purely additive over
    P5 OPSIN-parse (P6 set lookup <1 µs; P5 OPSIN subprocess ~1269 ms).
    Adding P6 saves ~1.3 s per match when running with
    ``opsin_parse_required=False``.
    """
    base = {
        "unknown organic compound",
        "compound with wildcard atoms (not supported)",
        "inorganic compound (not supported)",
    }
    # Lazy-import namer to avoid circular dependency at module-import time.
    try:
        from orthonym.namer import _METAL_NAMES
        for metal_name in _METAL_NAMES.values():
            base.add(f"{metal_name} compound (not supported)")
    except (ImportError, AttributeError):
        # Defensive fallback: if _METAL_NAMES not yet importable (e.g. at
        # very early boot ordering), use a conservative hard-coded subset.
        for metal in ("platinum", "uranium", "thorium", "gold", "iron",
                      "copper", "silver", "nickel", "palladium", "ruthenium"):
            base.add(f"{metal} compound (not supported)")
    return frozenset(base)


_DESCRIPTIVE_FALLBACK_NAMES: FrozenSet[str] = _build_descriptive_fallback_names()


def is_name_quality_inadequate(
    name: Optional[str],
    mol: "Chem.Mol",
    canonical_smiles: str,
    *,
    opsin_parse_required: bool = True,
) -> bool:
    """Predicate gating ML attachment per MLF-06.

    Returns ``True`` iff the rule-based pipeline output is degraded.

    See 162-AUDIT-MLF.md § 2 for the locked spec + per-pattern rationale.

    :param name: output of the rule-based pipeline (may be ``None``)
    :param mol: parsed RDKit molecule — used ONLY for ``GetNumHeavyAtoms()``
    :param canonical_smiles: RDKit-canonical SMILES (compared in P2)
    :param opsin_parse_required: gates the expensive P5 OPSIN backstop
                                  (keyword-only, default ``True``)
    :returns: ``True`` iff any of P1..P5 or P6 fires; ``False`` otherwise

    Purity contract per CONTEXT D-12:

    * Reads only the args (name, mol, canonical_smiles, opsin_parse_required)
    * NO ``mol`` mutation (no ``Chem.SanitizeMol``, no
      ``UpdatePropertyCache``, no atom-property assignment)
    * NO module-global state read/write — ``_GARBLED_TOKENS`` (tuple) and
      ``_DESCRIPTIVE_FALLBACK_NAMES`` (frozenset) are immutable constants
    * NO exception swallowing — OPSIN-parse exceptions propagate
    """
    # P1 — cheap O(1)
    if name is None:
        return True
    # P2 — cheap O(1)
    if name == canonical_smiles:
        return True
    # P3 — cheap O(len(_GARBLED_TOKENS))
    name_lower = name.lower()
    if any(t in name_lower for t in _GARBLED_TOKENS):
        return True
    # P4 — cheap O(1)
    if len(name) < 6 and mol.GetNumHeavyAtoms() > 15:
        return True
    # P6 — cheap O(1) frozenset membership (ADD per audit § 2.2 / R-03)
    if name in _DESCRIPTIVE_FALLBACK_NAMES:
        return True
    # P5 — expensive ~1269 ms OPSIN subprocess; gated by flag
    if opsin_parse_required:
        from orthonym.validation.opsin_roundtrip import opsin_parse
        return opsin_parse(name) is None
    return False
