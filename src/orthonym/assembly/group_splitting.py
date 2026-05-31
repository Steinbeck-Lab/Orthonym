"""Phase 169 Plan-02: Group-splitting polyfunctional rescue (POLY-01/02).

When a non-principal **composite** functional group (today only the
empirically-firing ``ester`` / ``thioester`` — see ``data/group_split_rules``)
has no clean strict-IUPAC prefix and would otherwise be **dropped** at the
``polyfunctional.py:get_fg_prefix_form()`` / ``DROP-23`` site (CONTEXT D-01/F2),
``split_composite_fg`` decomposes it into its ordered sub-group prefix
**components** instead of dropping it:

* ``ester``  ``-C(=O)-O-R``  →  ``oxo`` (the ``=O`` chalcogen) + ``R-oxy`` (the
  ``-O-R`` linker, e.g. ``ethoxy``)
* ``thioester`` ``-C(=O)-S-R`` → ``oxo`` + ``R-sulfanyl`` (e.g. ``ethylsulfanyl``)

The carbonyl carbon stays in the chain and carries the locant; both components
share it (CONTEXT worked example ``4-(ethylsulfanyl)-4-oxobutanoic acid``). The
caller appends each component to ``all_prefixes`` so they re-enter the EXISTING
``format_fg_prefix`` + ``alpha_sort_key`` pipeline (POLY-02 native).

**Single source of truth (CONTEXT D-02):** every component's prefix STRING is
resolved through the EXISTING authority — ``oxo`` via ``seniority.get_prefix``,
the alkoxy/sulfanyl forms via ``assembly.substituent_prefix_forms`` — NEVER
hardcoded and NEVER a string-rewrite (no regex substitution, no string-replace
call, no postprocessor pass; ``./skills/fix-methodology.md``). The
decomposition acts at the FG-prefix-resolution layer and returns structured
components.

**RT safety (CONTEXT D-05, FAIL-CLOSED):** when an ``OpsinOracle`` is supplied
(flag-ON only), the FULL assembled split name is OPSIN-round-trip-checked; a
split that does not round-trip (incl. the FAIL-CLOSED ``oracle._jar is None``
case) is rejected and the caller behaves exactly as today's ``DROP-23``
``continue`` — never emits a worse name (favorable asymmetry: the status-quo
dropped-FG name already fails RT, so a rejected split only preserves it).

Source: 169-CONTEXT.md D-01..D-05, F2; 169-RESEARCH.md "Architecture Patterns" +
"Code Examples"; 169-PATTERNS.md "group_splitting.py".
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

# Recursion guard: the FAIL-CLOSED RT gate re-names the whole molecule (flag ON)
# to obtain the actual assembled split name. That inner naming re-enters
# split_composite_fg; the guard makes the inner call skip the gate (base case) so
# there is no infinite recursion.
_IN_SPLIT_PROBE = threading.local()

# Per-process cache of the probe (full split name) keyed by canonical SMILES, so a
# repeated molecule does not pay a second full re-naming. The OpsinOracle separately
# caches the (candidate, target) -> RT-result.
_PROBE_NAME_CACHE: dict = {}


@dataclass(frozen=True)
class SplitComponent:
    """One resolved sub-group prefix of a split composite (TOPOLOGY → STRING).

    ``prefix_form`` is the RESOLVED string from the existing authority
    (``seniority.get_prefix`` / ``substituent_prefix_forms``), never from the
    JSON table. ``locants=None`` means "use the caller's central-carbon locants"
    so both components share the carbonyl-carbon locant.
    """

    role: str            # "chalcogen" | "linker"
    prefix_form: str     # resolved string, e.g. "oxo" / "ethoxy" / "ethylsulfanyl"
    locants: Optional[List[int]] = None
    count: int = 1


@dataclass(frozen=True)
class SplitEvent:
    """Phase 169 D-06 diagnostic event (the reach-report data source)."""

    kind: str            # "split_emit" | "split_reject_rt_unsafe" | "passthrough_not_in_table"
    fg_name: str
    encoding: Optional[str] = None
    rt_oracle_result: Optional[bool] = None
    p_section_cite: Optional[str] = None


def _resolve_oxo() -> Optional[str]:
    """The ``=O`` chalcogen prefix, resolved through the existing PREFIX_FORMS authority."""
    from ..rules.seniority import get_prefix  # Pattern-S3 lazy import
    return get_prefix("ketone")  # == "oxo" (P-66.6.1) — single source of truth, NOT a literal


def _decompose_carbonyl_ester(
    mol: Any, match: tuple, principal_chain: Optional[List[int]], *, sulfur: bool
) -> Optional[List[SplitComponent]]:
    """Decompose ``-C(=O)-X-R`` (X=O ester / X=S thioester) into oxo + linker.

    SMARTS match order (confirmed): (carbonyl_C, =O, linker_X, alkyl_C).
    The carbonyl carbon stays in the chain (carries the locant); the chalcogen
    ``=O`` resolves to ``oxo`` and the ``-X-R`` linker resolves to the
    alkoxy/sulfanyl prefix via the existing substituent_prefix_forms dispatcher.
    """
    if mol is None or match is None or len(match) < 4:
        return None
    carbonyl_c, _double_o, linker_x, alkyl_c = match[0], match[1], match[2], match[3]

    oxo = _resolve_oxo()
    if not oxo:
        return None

    from ..assembly.substituent_prefix_forms import (  # Pattern-S3 lazy import
        get_alkoxy_prefix,
        get_sulfanyl_prefix,
    )
    # The linker hetero-atom + the two carbons it bridges, in the (X, C, C) order the
    # dispatcher expects; it names the substituent (alkyl) side.
    linker_atoms = (linker_x, carbonyl_c, alkyl_c)
    if sulfur:
        linker_prefix = get_sulfanyl_prefix(mol, linker_atoms, principal_chain)
    else:
        linker_prefix = get_alkoxy_prefix(mol, linker_atoms, principal_chain)
    if not linker_prefix:
        return None

    # Both components share the caller's central-carbon (carbonyl) locant (locants=None).
    return [
        SplitComponent(role="chalcogen", prefix_form=oxo, locants=None, count=1),
        SplitComponent(role="linker", prefix_form=linker_prefix, locants=None, count=1),
    ]


_DECOMPOSERS = {
    "ester": lambda mol, m, pc: _decompose_carbonyl_ester(mol, m, pc, sulfur=False),
    "thioester": lambda mol, m, pc: _decompose_carbonyl_ester(mol, m, pc, sulfur=True),
}


def _probe_full_split_name(mol: Any) -> Optional[str]:
    """Re-name the whole molecule with group-splitting ON to obtain the actual
    assembled split name (the RT-gate candidate). Recursion-guarded + cached."""
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        return None
    if smi in _PROBE_NAME_CACHE:
        return _PROBE_NAME_CACHE[smi]
    _IN_SPLIT_PROBE.active = True
    name = None
    try:
        from ..namer import name_compound  # Pattern-S3 lazy import (avoids import cycle)
        name = name_compound(smi, enable_group_splitting=True)
    except Exception as exc:  # naming failure -> no candidate -> FAIL-CLOSED at caller
        logger.debug("group-split RT probe naming failed for %r: %s", smi, exc)
        name = None
    finally:
        _IN_SPLIT_PROBE.active = False
    _PROBE_NAME_CACHE[smi] = name
    return name


def split_composite_fg(
    fg_name: str,
    mol: Any,
    atoms: tuple,
    principal_chain: Optional[List[int]],
    oracle: Any = None,
) -> Optional[List[SplitComponent]]:
    """Decompose a table-listed composite loser into ordered sub-group prefix components.

    Returns ``None`` (the caller then behaves exactly as today's ``DROP-23``
    ``continue``) when:
      * ``fg_name`` is not in the split table (deny-path — functional-class FGs
        stay dropped, CONTEXT D-03), OR
      * the structural decomposition cannot resolve a clean component string, OR
      * the per-split OPSIN-RT gate rejects the assembled split name
        (FAIL-CLOSED, CONTEXT D-05 — never a worse name).

    Otherwise returns the ordered ``[SplitComponent, ...]`` list (e.g.
    ``[oxo, ethoxy]``) the caller appends to ``all_prefixes`` (POLY-02 native).
    """
    # Deny-path first (short-circuits before touching mol — so a functional-class
    # FG returns None even with null args; CONTEXT D-03 bound).
    from ..data.group_split_rules import SPLIT_RULES  # Pattern-S3 lazy import
    if fg_name not in SPLIT_RULES:
        return None

    decompose = _DECOMPOSERS.get(fg_name)
    if decompose is None:  # in the table but no decomposer wired -> stay dropped
        return None

    components = decompose(mol, atoms, principal_chain)
    if not components:
        return None

    rule = SPLIT_RULES[fg_name]

    # FAIL-CLOSED per-split RT gate (CONTEXT D-05). Skipped inside the probe re-naming
    # (base case) and when no oracle is supplied (flag-OFF never reaches here; unit
    # tests may pass oracle=None to exercise the pure decomposition).
    if oracle is not None and not getattr(_IN_SPLIT_PROBE, "active", False):
        try:
            target_canon = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        except Exception:
            target_canon = None
        candidate = _probe_full_split_name(mol)
        rt_ok = bool(
            target_canon and candidate and oracle.rt_safe(target_canon, candidate)
        )
        if not rt_ok:
            logger.debug(
                "group-split RT-reject fg=%s encoding=%s (candidate=%r)",
                fg_name, rule.encoding, candidate,
            )
            return None

    logger.debug("group-split emit fg=%s encoding=%s components=%d",
                 fg_name, rule.encoding, len(components))
    return components


__all__ = ["split_composite_fg", "SplitComponent", "SplitEvent"]
