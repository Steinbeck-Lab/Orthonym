"""Phase 160 shared handler helpers (DECOMP-01 + CONTEXT D-08).

Substrate commit 02-00: lazy re-export wrappers around the canonical
implementations in ``composer.py``. Per CONTEXT D-24 incremental-migration
discipline, composer.py STILL OWNS:

* ``_name_iso_x_cyanate`` (composer.py:2204; 27 LOC) — shared between
  isocyanate (commit 02-04) and isothiocyanate (commit 02-05) handlers.
* ``_name_r_group`` (composer.py:2233; 217 LOC) — shared between urea
  (commit 02-08), guanidine (commit 02-09), and the general_acyclic
  catch-all (Plan-03 commit 03-09).

The substrate ships THIS module so handler files can write the
forward-looking import path::

    from ._handler_shared import name_iso_x_cyanate
    from ._handler_shared import name_r_group

while internally the symbols delegate (via lazy import inside each
function body) to composer.py. When Plan-03 commit 03-10 lands, the
function BODIES move here verbatim and composer.py's
``_name_iso_x_cyanate`` + ``_name_r_group`` definitions delete. The
re-export shape ensures handler files do NOT need to change import paths
at thinning time — only this delegation layer flips.

Per CONTEXT D-08 catch-all helper convention: shared logic between
multiple handlers MUST live in this module (not duplicated across
handlers/). The "≥ 2 handler" threshold is per CONTEXT D-03 +
160-AUDIT-DECOMP.md § 1.2 in-file-handler-body inventory.

Anti-pattern hygiene:
- AP-160-02 banned: do NOT group two handlers into one extraction commit
  because they share a body. The handlers are separate files; the SHARED
  helper lives HERE; each handler imports the helper but keeps its own
  file + own atomic commit.
- AP-160-12 banned: handler logic in shim (must be 1-3-line wrapper).
  This module's helpers ARE the multi-line logic; handlers import them.

References:
- composer.py:2204-2230 (``_name_iso_x_cyanate``) — verbatim source.
- composer.py:2233-2449 (``_name_r_group``) — verbatim source.
- 160-PATTERNS.md § "Common Conventions" + line 458.
- 160-CONTEXT.md D-08 + D-24 — catch-all helper convention + migration.

PHASE 160.2 EXTENSION (Plan-02-01; CONTEXT D-02 + D-03):
Six NEW public functions lifted verbatim from composer.py per the
two-step "helpers-first" extraction:
  - _generate_chain_parent  (composer.py:3785-3818, 33 LOC)
  - _generate_ring_parent   (composer.py:3821-3893, 72 LOC)
  - _generate_suffix        (composer.py:3934-4076, 142 LOC)
  - _generate_prefixes      (composer.py:4079-4282, 203 LOC)
  - _generate_stereodescriptors (composer.py:6634-6704, 70 LOC)
  - _assemble_fragments     (composer.py:6799-6954, 155 LOC)

Each function body is COPIED VERBATIM from composer.py per Phase 145.1 D-09
mechanical-translation discipline. composer.py keeps thin re-export shims
(``from .handlers._handler_shared import _generate_chain_parent`` at module
top) so all in-file call sites in _name_oxime_or_hydrazone,
_assemble_amide_name, _assemble_amine_name, _assemble_ring_with_ester_prefixes,
_assemble_complex_ring_name remain byte-identical per CONTEXT D-13
forbidden-boundary preservation.

Per CONTEXT D-09 honest-fail-on-data: any byte-identical canary regression
at commit 02-01 reverts the commit; remediation lands in a follow-up.

Module-level dependencies resolve via LAZY imports inside each function
body to avoid the circular dependency
``_handler_shared.py -> composer.py -> _handler_shared.py`` that the
composer-side shim re-export creates at module-load time. composer.py
imports this module at its top; this module deferring composer.py imports
to first call breaks the cycle while preserving byte-identical behavior.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..naming_utils import (
    BRANCH_HANDLED_FGS,
    SIMPLE_MULTIPLIERS,
    alpha_sort_key,
    format_suffix_with_locants,
    get_multiplier_prefix,
    get_suffix_multiplier_prefix,
    should_omit_locant_one,
)
from ...data.chain_names import get_chain_prefix
from ...rules.locants import get_bond_locants, get_functional_group_locants
from ...rules.seniority import get_prefix, get_suffix


logger = logging.getLogger(__name__)


def name_iso_x_cyanate(
    features: Any, fg_key: str, suffix_word: str,
) -> Optional[str]:
    """Common implementation for isocyanate and isothiocyanate naming.

    Lazy delegate to ``composer.py:_name_iso_x_cyanate`` (composer.py:2204).
    Per CONTEXT D-24 + PATTERNS § 5 first-wave guidance, composer.py owns
    the canonical body at this commit; this wrapper provides the
    forward-looking import path ``handlers._handler_shared.name_iso_x_cyanate``
    for handler files that want stable paths now.

    SMARTS pattern: ``[#6][NX2]=[CX2]=[OX1]`` (isocyanate) or
    ``[#6][NX2]=[CX2]=[SX1]`` (isothiocyanate). Match tuple:
    ``(R_carbon, N, C, O/S)``.

    Args:
        features: MolecularFeatures object.
        fg_key: Either ``'isocyanate'`` or ``'isothiocyanate'``.
        suffix_word: The functional-class suffix word
            (``'isocyanate'`` / ``'isothiocyanate'``).

    Returns:
        Functional class name like ``'methyl isocyanate'``, or None.

    See Also:
        composer.py:_name_iso_x_cyanate — canonical implementation.
        composer.py:_name_isocyanate / _name_isothiocyanate — single-line
            callers; both move to handlers/{isocyanate,isothiocyanate}.py.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _name_iso_x_cyanate
    return _name_iso_x_cyanate(features, fg_key, suffix_word)


def name_r_group(
    mol: Any, start_idx: int, exclude_atoms: set,
) -> Optional[str]:
    """Name an R group (substituent fragment) starting from start_idx.

    Lazy delegate to ``composer.py:_name_r_group`` (composer.py:2233).
    Per CONTEXT D-24 + PATTERNS § 5 first-wave guidance.

    Args:
        mol: RDKit Mol object.
        start_idx: Atom index of the R-group attachment point.
        exclude_atoms: Set of atom indices to exclude from the R-group walk
            (e.g., the functional group atoms already named).

    Returns:
        Substituent name as IUPAC P-29 substituent prefix (e.g., ``'methyl'``,
        ``'phenyl'``, ``'4-chlorophenyl'``), or None on failure.

    See Also:
        composer.py:_name_r_group — canonical implementation; consumed by
            urea, guanidine, isocyanate, isothiocyanate, and the
            general_acyclic catch-all.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _name_r_group
    return _name_r_group(mol, start_idx, exclude_atoms)


def cached_is_complex_ring_system(features: Any) -> bool:
    """WR-02: per-features memoization of composer._is_complex_ring_system.

    The SMARTS-based complex-ring check is heavy; predicates that call it
    inside the dispatch loop violate the spirit of CONTEXT D-25 (predicates
    are pure read-only over already-perceived state). Cache the result on
    the features object as a private attribute so partial_sat / polycyclic /
    ring_ester predicates share a single SMARTS evaluation per features
    instance instead of running it three times per dispatch.

    The cache is per-features-instance state owned by features itself; the
    predicate remains pure with respect to shared/global state.

    Args:
        features: MolecularFeatures-like object with a ``mol`` attribute.

    Returns:
        True iff the molecule is a complex ring system per the SMARTS check.
    """
    cached = getattr(features, "_cached_complex_ring_system_result", None)
    if cached is not None:
        return cached
    from ..composer import _is_complex_ring_system
    result = bool(_is_complex_ring_system(features.mol))
    try:
        features._cached_complex_ring_system_result = result
    except (AttributeError, TypeError):
        # Frozen / immutable features: fall back to per-call computation.
        pass
    return result


# =============================================================================
# Phase 160.2 Plan-02-01: 6 _generate_* helpers lifted from composer.py
# verbatim per CONTEXT D-02 + D-03 + Phase 145.1 D-09 mechanical-translation
# discipline. Each function body is byte-identical to its composer.py
# counterpart prior to this commit. Composer-side symbols (NameFragment,
# TERMINAL_GROUPS, _generate_alkyl_prefixes, etc.) resolve via lazy imports
# inside each function body to avoid the import cycle that the
# composer.py shim re-export creates at module-load time.
# =============================================================================


def _generate_chain_parent(features: Any) -> "NameFragment":
    """Generate parent name for acyclic chains.

    Returns a NameFragment where:
    - text: stem + unsaturation_base (e.g., "but", "prop")
    - locants: tuple of (double_bond_locants, triple_bond_locants)
    """
    from ..composer import NameFragment

    chain_length = len(features.principal_chain)

    # Get chain prefix (stem)
    stem = get_chain_prefix(chain_length)

    # WS-A task 9 / P-15.4: a skeletal chain with embedded heteroatoms
    # (the P-44.3 replacement-nomenclature parent) cites them as 'oxa'/
    # 'aza'/'thia' prefixes with their locants — the bare carbon stem
    # silently described a DIFFERENT molecule ('dodecane' for a chain
    # with 4 O). Heteroatom locants come from the oriented chain.
    _het_positions = {}
    for _pos, _aidx in enumerate(features.principal_chain, start=1):
        _sym = features.mol.GetAtomWithIdx(_aidx).GetSymbol()
        if _sym != 'C':
            _het_positions.setdefault(_sym, []).append(_pos)
    if _het_positions:
        _REPL = {'O': 'oxa', 'S': 'thia', 'Se': 'selena', 'Te': 'tellura',
                 'N': 'aza', 'P': 'phospha', 'Si': 'sila', 'B': 'bora'}
        if not all(_sym in _REPL for _sym in _het_positions):
            _het_positions = {}
    if _het_positions:
        from ...rules.locants import compare_locant_sets
        # P-31.1.4.2.3: heteroatom replacement terms get LOWEST locants
        # before detachable prefixes — re-orient (reverse) when reversal
        # lowers the heteroatom locant set.
        _n = chain_length
        _all_locs = sorted(
            loc for locs in _het_positions.values() for loc in locs
        )
        _rev_locs = sorted(_n + 1 - loc for loc in _all_locs)
        if compare_locant_sets(_rev_locs, _all_locs) < 0:
            features.principal_chain = list(reversed(features.principal_chain))
            from ...rules.locants import build_atom_to_locant
            features.atom_to_locant = build_atom_to_locant(
                features.principal_chain
            )
            _het_positions = {
                _sym: sorted(_n + 1 - loc for loc in locs)
                for _sym, locs in _het_positions.items()
            }
        # Citation: seniority order O > S > Se > Te > N > P > Si > B
        # (P-15.4.3.1), each term with its locants and multiplier.
        _SENIORITY = ['O', 'S', 'Se', 'Te', 'N', 'P', 'Si', 'B']
        _MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta',
                 6: 'hexa', 7: 'hepta', 8: 'octa'}
        _terms = []
        for _sym in _SENIORITY:
            if _sym not in _het_positions:
                continue
            _locs = sorted(_het_positions[_sym])
            _mult = _MULT.get(len(_locs), str(len(_locs)))
            _terms.append(
                f"{','.join(str(x) for x in _locs)}-{_mult}{_REPL[_sym]}"
            )
        stem = ''.join(_terms) + stem

    # Get bond locants if we have atom_to_locant mapping
    double_locants = []
    triple_locants = []
    if features.atom_to_locant:
        double_locants = get_bond_locants(
            features.principal_chain,
            features.double_bonds,
            features.atom_to_locant
        )
        triple_locants = get_bond_locants(
            features.principal_chain,
            features.triple_bonds,
            features.atom_to_locant
        )

    # Store stem as text, bond locants as locants tuple
    # We'll use a nested tuple: ((double_locants), (triple_locants))
    return NameFragment(
        text=stem,
        locants=(tuple(double_locants), tuple(triple_locants)),
        fragment_type="parent"
    )


def _generate_ring_parent(features: Any) -> "NameFragment":
    """
    Generate parent name for cyclic compounds.

    For cycloalkanes: "cyclo" + chain prefix + "an" (e.g., "cyclohexan")
    The final 'e' is added during assembly if no suffix follows.

    For cycloalkenes: "cyclo" + chain prefix + double bond info
    - Mono-cycloalkenes: cyclo + stem + "en" (cyclohexene) - no locant
    - Cycloalkadienes: cyclo + stem + "a" + locants + "dien" (cyclohexa-1,3-diene)

    For other ring types, returns placeholder for now (to be implemented
    in subsequent plans).

    Args:
        features: MolecularFeatures object with ring_type and principal_ring

    Returns:
        NameFragment with parent text and bond locants
    """
    from ..composer import NameFragment

    ring_type = getattr(features, 'ring_type', None)
    principal_ring = getattr(features, 'principal_ring', None)

    if not principal_ring:
        # Fallback: no ring identified -- return empty parent to avoid
        # generating garbled 'cycloane' (cyclo + ane with no stem)
        return NameFragment(text="", fragment_type="parent")

    ring_size = len(principal_ring)

    # Guard: ring_size must produce a valid stem; otherwise return empty parent
    try:
        stem = get_chain_prefix(ring_size)
    except (ValueError, KeyError):
        # Invalid ring size (0, negative, etc.) -- return empty parent
        return NameFragment(text="", fragment_type="parent")

    # Guard: stem must be non-empty to avoid generating 'cycloane'
    if not stem:
        return NameFragment(text="", fragment_type="parent")

    if ring_type == 'cycloalkane':
        # Cycloalkane naming: cyclo + stem + an (e.g., cyclohexan)
        # Return stem - the "ane" will be added in assembly
        return NameFragment(
            text=f"cyclo{stem}",
            locants=((), ()),  # No bond locants for saturated rings
            fragment_type="parent"
        )

    elif ring_type == 'cycloalkene':
        # Get double bond locants from features
        ring_double_bond_locants = getattr(features, 'ring_double_bond_locants', [])

        return NameFragment(
            text=f"cyclo{stem}",
            locants=(tuple(ring_double_bond_locants), ()),  # (double_bond_locants, triple_bond_locants)
            fragment_type="parent"
        )

    elif ring_type == 'aromatic':
        # Aromatic ring type not handled by cyclic naming -- return empty
        # parent to signal that this ring needs a specialized handler
        # (benzene retained name, fused ring dictionary, etc.)
        return NameFragment(text="", fragment_type="parent")

    elif ring_type and ring_type.startswith('heterocyclic'):
        # Heterocyclic ring type not handled by cyclic naming -- return
        # empty parent to signal need for specialized handler
        return NameFragment(text="", fragment_type="parent")

    # Unknown ring type -- return empty parent rather than garbled 'cyclo'
    return NameFragment(text="", fragment_type="parent")


# P-14.3.4.2(c) licence: suffix classes with Blue Book evidence that the trivial
# mono-suffix ring locant is withheld. NOT a spelling table -- a scope restriction on
# an otherwise structural predicate. `imine` is excluded on purpose; see the docstring.
_P14_3_4_RING_SUFFIX_CLASSES = frozenset({
    'ketone',
    'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol', 'alcohol',
    'thiol', 'selenol', 'tellurol',
    'primary_amine', 'amine',
})


def _ring_suffix_locant_is_trivial(features, oriented_ring, ring_idx_to_locant,
                                   suffix_ring_atoms=None) -> bool:
    """P-14.3.4.2(c): may a single non-terminal ring suffix drop its locant?

    v29 Phase C tranche A. **DENY BY DEFAULT** — P-14.3.3 "Citation of locants"
    (``BlueBookV2.md:2869``) says *"if any locants are essential … then all locants
    must be cited"*, so this returns True only for the case where the Blue Book's own
    ``(PIN)`` rows show the locant withheld, and False for everything else including
    anything it cannot positively establish.

    Licensed (all must hold):
      * the parent is ONE ring (monocyclic) — a fused system numbers
        non-equivalently, so ``naphthalen-1-ol`` keeps its locant;
      * every ring atom is carbon — a heterocycle keeps it
        (``piperidine-1-carbonitrile``, ``:34730``);
      * every ring bond is single — unsaturation makes positions distinct, so
        ``cyclohex-2-en-1-ol`` keeps its locant;
      * every ring atom other than the suffix-bearing one is an unsubstituted CH2 —
        any other cited substituent restores the locant
        (``4-methylcyclohexan-1-one``).

    Under those conditions every substitutable position is equivalent
    (P-14.3.4.3 *"only one kind of substitutable hydrogen"*), so ``1`` carries no
    information: ``cyclohexanone``, ``cyclopentanol``, ``cyclohexanethiol``.

    ⚠ **The four conditions are deliberately OVER-DETERMINED, and mutation testing
    proved it — do not "simplify" them.** Measured 2026-07-28: disabling the
    monocyclic check, the all-carbon check, the H-count check, the single-bond check
    or the exocyclic-neighbour loop *individually* breaks NO test, because every
    witness is rejected by two or more of them:

      * no ring heteroatom carries 2 H (``N``-H has 1, ring ``O`` has 0), so the
        CH2 test already excludes heterocycles — the all-carbon test is redundant;
      * a fused system's bridgehead atoms carry ≤1 H, so the CH2 test already
        excludes them — the monocyclic test is redundant;
      * an sp2 ring CH carries 1 H, so the CH2 test and the single-bond test cover
        ring unsaturation *jointly* — disabling BOTH does break
        ``cyclohex-3-en-1-ol`` (verified);
      * a substituted ring carbon has both an exocyclic neighbour and ≠2 H, so
        those two cover substitution jointly.

    Removing the whole predicate breaks 4 tests. Each condition is kept because it
    states one clause of P-14.3.4.2(c) explicitly and defends the licence if another
    clause is ever loosened — not because it is independently exercised.
    """
    mol = getattr(features, 'mol', None)
    if mol is None or not oriented_ring:
        return False
    # P-14.3.3 (BB:2869) as an AMBIENT scope. Some essential-locant facts are known
    # only OUTSIDE this call: rules/isotopes.py strips every label and names the
    # isotope-FREE skeleton, so `features.mol` here has GetIsotope()==0 everywhere and
    # every isotope test we could write is structurally False. MEASURED: `OC1CCCC[13CH2]1`
    # shipped `(2-13C1)cyclohexanol` where P-82.6.1.1 (BB:44180) requires
    # `(2-13C)cyclohexan-1-ol` -- BB:44186 prints the elided form as "[not
    # (2-13C)ethanol]". The decorator now declares the scope; we honour it.
    from ..locant_omission import locants_are_forced
    if locants_are_forced():
        return False


    # Suffix-class allowlist. P-14.3.3 is deny-by-default, so a class is licensed only
    # where the Blue Book actually shows the locant withheld:
    #   ketone   -- verbatim "'cyclohexanone' (PIN)"  BB:14916 (+ BB:2917, BB:28394)
    #   alcohol  -- verbatim "(1) cyclopentanol (PIN)" BB:26854
    #   thiol    -- P-14.3.4.2(c) worked example `cyclohexanethiol`
    #   amine    -- same shape, single monovalent heteroatom suffix
    #
    # `imine` is DELIBERATELY EXCLUDED. There is no verbatim bare `-imine` row (the
    # Phase C derivation flagged it as the lowest-confidence of its five gold
    # conflicts), and licensing it shipped a real defect: for an oxime the OH is
    # STRIPPED from `features.mol` before this point and re-added later as an
    # `N-hydroxy` prefix, so the N-substituent is INVISIBLE here -- both
    # `features.mol` and `features.canonical_smiles` carry the reduced 7-heavy-atom
    # form of `ON=C1CCCCC1`. The licence therefore fired and emitted
    # `N-hydroxycyclohexanimine`, but `N` is an ESSENTIAL locant in the same scope and
    # P-14.3.3 (BB:2869) says one essential locant restores every locant in that
    # scope, so `N-hydroxycyclohexan-1-imine` is correct. The GATE caught it as a
    # target regression.
    #
    # Deciding this properly needs the whole scope -- i.e. the assembler, where the
    # prefix fragments exist -- which is tranche B's per-scope machinery. Until then
    # the honest answer for a class with no verbatim evidence is: do not licence it.
    if getattr(features, 'principal_group', None) not in _P14_3_4_RING_SUFFIX_CLASSES:
        return False
    ring = list(oriented_ring)
    ring_set = set(ring)

    ring_info = mol.GetRingInfo()
    # Monocyclic only: every ring atom must belong to exactly one ring, and the
    # perceived parent must be the whole of that ring system.
    for idx in ring:
        try:
            if ring_info.NumAtomRings(idx) != 1:
                return False
        except Exception:
            return False
    if not any(set(r) == ring_set for r in ring_info.AtomRings()):
        return False

    # All-carbon, and every ring bond single.
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
            return False
        if atom.GetFormalCharge() != 0:
            return False
    from rdkit import Chem as _Chem
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in ring_set and b in ring_set:
            if bond.GetBondType() != _Chem.BondType.SINGLE:
                return False

    # ── The whole-molecule condition ─────────────────────────────────────────
    # DENY-BY-DEFAULT, and deliberately made OBVIOUSLY correct rather than cleverly
    # correct: the molecule must be NOTHING BUT the ring plus one suffix heteroatom.
    # Then, trivially, every other ring position is an unsubstituted CH2, nothing
    # else in the name can carry a locant, and `1` cannot be distinctive.
    #
    # Two earlier, cleverer versions each shipped a defect:
    #   (1) exempting every atom of `features.principal_group_atoms` from a per-atom
    #       CH2 test -- but a ring ketone's SMARTS match spans BOTH ALPHA CARBONS
    #       (`O=C1CCCCC1CCCCC` -> `(2, 1, 0, 6)`, atom 6 bearing the pentyl), so an
    #       alpha substituent was invisible and `2-pentylcyclohexanone` shipped. The
    #       GATE caught it as a protect regression; `4-methylcyclohexan-1-one` had
    #       passed only because position 4 is not an alpha carbon.
    #   (2) a per-ring-atom test alone still licensed `ON=C1CCCCC1` ->
    #       `N-hydroxycyclohexanimine`, because the N-hydroxy hangs off the SUFFIX
    #       heteroatom, not off a ring atom. But `N` is an ESSENTIAL locant in the
    #       same scope, and P-14.3.3 (BB:2869) says one essential locant in a scope
    #       restores every locant in that scope -- so `N-hydroxycyclohexan-1-imine`
    #       is right. The GATE caught that one too, as a target regression.
    #
    # Hence: count heavy atoms. Anything beyond ring + 1 -- a ring substituent, an
    # N-substituent on the suffix, a multi-atom suffix such as -C(=O)OOH -- denies.
    # That leaves `-one` / `-ol` / `-thiol` / `-amine` / `-imine` on a bare
    # cycloalkane, which is exactly the evidence base (BB:14916, BB:26854,
    # P-14.3.4.2(c)). Widening beyond it needs the real per-scope locant machinery
    # and belongs in tranche B, not in a looser predicate here.
    if mol.GetNumHeavyAtoms() != len(ring) + 1:
        return False

    # ⚠⚠ CORRECTED 2026-07-28 BY MEASUREMENT. This block used to be documented as the
    # structural catch for the oxime: "Re-check against the ORIGINAL input, so anything
    # the perception layer moved out of `mol` still denies the licence."
    # **THAT CLAIM IS FALSE, and the guard is a NO-OP for the molecule it names.**
    # Measured with a call-spy validated on two known positives (`cyclohexanone`,
    # `cyclopentanol`), `ON=C1CCCCC1` reaches this predicate with:
    #     principal_group='imine'  len(ring)=6  mol.GetNumHeavyAtoms()=7
    #     features.canonical_smiles='N=C1CCCCC1'  -> re-parsed heavy atoms = 7
    # `canonical_smiles` is NOT the original input -- the perception layer strips the
    # oxime OH from it too -- so `7 == 6+1` PASSES here exactly as it passes above.
    # The oxime and the plain imine `N=C1CCCCC1` arrive at this function with
    # BYTE-IDENTICAL inputs on every field the predicate can read. The only thing that
    # actually denied the oxime was the `_P14_3_4_RING_SUFFIX_CLASSES` allowlist.
    #
    # ⇒ Two consequences, both load-bearing:
    #   1. The `imine` exclusion above is NOT redundant belt-and-braces; it is the sole
    #      guard. Do not remove it on the reasoning that this re-check backs it up.
    #   2. Re-admitting `imine` (Phase C tranche B Task 4) is IMPOSSIBLE at this site,
    #      not merely awkward: no predicate here can distinguish the two molecules. It
    #      must be decided where the `N-hydroxy` prefix fragment exists (the assembler),
    #      or left excluded.
    # This block is retained because it still denies a scope whose canonical_smiles
    # genuinely carries extra atoms, but what it guards has NOT been re-derived --
    # treat its coverage as unknown rather than as documented.
    _canon = getattr(features, 'canonical_smiles', None)
    if not _canon:
        return False
    _input = _Chem.MolFromSmiles(_canon)
    if _input is None or _input.GetNumHeavyAtoms() != len(ring) + 1:
        return False

    outside = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring_set]
    if len(outside) != 1:
        return False
    sub = mol.GetAtomWithIdx(outside[0])
    if sub.GetSymbol() not in ('O', 'N', 'S', 'Se', 'Te'):
        return False
    if sub.GetFormalCharge() != 0:
        return False
    # It must hang off exactly one ring atom, and that atom must be the one whose
    # locant is being decided (when the caller told us which that is).
    nbrs = [n.GetIdx() for n in sub.GetNeighbors()]
    if len(nbrs) != 1 or nbrs[0] not in ring_set:
        return False
    if suffix_ring_atoms and nbrs[0] not in set(suffix_ring_atoms):
        return False

    return True


def _generate_suffix(features: Any) -> Optional["NameFragment"]:
    """Generate suffix fragment for principal group.

    Handles locant assignment for the principal functional group:
    - Terminal groups (carboxylic acid, aldehyde): locant always 1, omitted from name
    - Non-terminal groups (alcohol, ketone): locant MUST be included in PIN style
    - Multiple instances: include multiplier (di, tri) and all locants
    """
    from ..composer import NameFragment, TERMINAL_GROUPS

    fg_name = features.principal_group
    if not fg_name:
        return None

    # Correct is_ring: use chain suffix when chain is parent or principal chain exists,
    # ring suffix only when ring is parent and PG is on a ring carbon.
    if getattr(features, 'chain_is_parent', False):
        is_ring = False
    elif features.principal_chain:
        is_ring = False
    else:
        is_ring = features.is_cyclic
    suffix_text = get_suffix(fg_name, is_ring=is_ring)

    if not suffix_text:
        return None

    # Get locants for functional group positions on the chain
    locants = ()
    fg_count = len(features.principal_group_atoms) if features.principal_group_atoms else 1

    # Validate suffix count against parent capacity:
    # The number of suffix groups cannot exceed the number of atoms in the parent
    # structure (chain length or ring size). E.g., ethane (2C) cannot have tetraol.
    if features.principal_chain:
        max_capacity = len(features.principal_chain)
    elif getattr(features, 'oriented_ring', None):
        max_capacity = len(features.oriented_ring)
    else:
        max_capacity = fg_count  # no constraint if we can't determine parent size

    if fg_count > max_capacity:
        fg_count = max_capacity

    if features.principal_chain and features.atom_to_locant and features.principal_group_atoms:
        fg_locants = get_functional_group_locants(
            features.principal_chain,
            features.principal_group_atoms,
            features.atom_to_locant,
            mol=features.mol
        )

        # Deduplicate locants (overlapping SMARTS can produce duplicates)
        fg_locants = sorted(set(fg_locants))

        # Further validate: count should match unique locants when locants exist
        if fg_locants:
            fg_count = len(fg_locants)

        # Terminal groups: locant is implicitly 1, do NOT include in name.
        # Pass chain_length + is_monosubstituted so P-14.3.4 elision also covers
        # methane (chain_length==1) and a single non-terminal suffix on ethane
        # (chain_length==2) -> 'methanamine' / 'N,N-dimethylmethanamine' /
        # 'ethanamine', not 'methan-1-amine' / 'ethan-1-amine'.
        # Wave2 T3a: is_monosubstituted means the parent bears ONLY this one
        # decoration. A C-substituent (chain, ring-as-substituent, or a
        # non-principal FG prefix) forces the suffix locant back, exactly like
        # the dedicated -ol handler ('2-chloroethan-1-ol' P-14.3.4 verbatim):
        # '1-cyclohexylethan-1-imine', not '1-cyclohexylethanimine'.
        # N-substituents live on the FG nitrogen, not the parent chain, and
        # are counted nowhere here (BB VERBATIM 'N-methylethanimine' stays).
        _suffix_chain_len = len(features.principal_chain)
        _other_subs = 0
        # Wave2 T3 determinism fix: the _other_subs tightening (cite the suffix
        # locant when a substituent is present) is scoped to NEUTRAL parents.
        # For a CHARGED principal group (aminium/-ium) the numbering follows
        # P-74 and HEAD deterministically ELIDED the suffix locant
        # ('2-hydroxy-N,N,N-trimethylethanaminium'); forcing the locant there
        # exposed an order-dependent charged-species numbering
        # ('1-hydroxy…ethan-2-aminium' vs '2-hydroxy…ethan-1-aminium'). The
        # motivating imine/selenol cases (1-cyclohexylethan-1-imine,
        # 2-chloroethane-1-selenol) are all neutral, so this scope keeps them.
        _is_charged = features.mol is not None and any(
            a.GetFormalCharge() != 0 for a in features.mol.GetAtoms()
        )
        _pg_match_atoms: set = set()
        for _m in (features.principal_group_atoms or []):
            _pg_match_atoms.update(_m)
        # Charged parents keep HEAD behavior (_other_subs stays 0 ->
        # is_monosubstituted = fg_count == 1); the tightening is neutral-only.
        if not _is_charged:
            if features.substituents:
                # The chain-finder records the principal FG's own heteroatom(s)
                # as a pseudo-substituent branch (imine =N on 'CC=N' shows up as
                # {1: [[N]]}); those are the suffix itself, not a decoration —
                # skip any branch wholly inside a principal-group match.
                for _subs_at_pos in features.substituents.values():
                    for _sub_atoms in _subs_at_pos:
                        _sa = set(
                            _sub_atoms
                            if isinstance(_sub_atoms, (list, tuple, set, frozenset))
                            else [_sub_atoms]
                        )
                        # Any overlap with the PG matches means the branch hangs
                        # off the suffix heteroatom (an N-substituent: the
                        # N-ethyl branch of CCNCC contains the amine N) — those
                        # do not decorate the parent chain and must not force
                        # the locant ('N-ethylethanamine' stays elided).
                        if _sa & _pg_match_atoms:
                            continue
                        _other_subs += 1
            if getattr(features, 'chain_is_parent', False):
                # Ring substituents attached through the suffix HETEROATOM are
                # N-substituents (N-phenylethanimine), not parent decorations;
                # rings attached at a parent carbon force the locant
                # (1-phenylethan-1-imine, parallel to 1-phenylethan-1-one).
                for _rg in (getattr(features, 'ring_substituents_as_groups', None)
                            or []):
                    _rg_set = set(_rg)
                    _on_het = False
                    for _ra in _rg_set:
                        for _nb in features.mol.GetAtomWithIdx(_ra).GetNeighbors():
                            _ni = _nb.GetIdx()
                            if (_ni not in _rg_set
                                    and _ni in _pg_match_atoms
                                    and features.mol.GetAtomWithIdx(_ni)
                                    .GetAtomicNum() != 6):
                                _on_het = True
                                break
                        if _on_het:
                            break
                    if not _on_het:
                        _other_subs += 1
            if getattr(features, 'non_principal_groups', None):
                _other_subs += len(features.non_principal_groups)
        if should_omit_locant_one(
            context="suffix",
            fg_type=fg_name,
            chain_length=_suffix_chain_len,
            is_monosubstituted=(fg_count == 1 and _other_subs == 0),
        ):
            locants = ()
        else:
            # For non-terminal groups (alcohol, ketone), include locants
            locants = tuple(fg_locants)

    elif (not features.principal_chain
          and getattr(features, 'oriented_ring', None)
          and features.principal_group_atoms
          and fg_name in TERMINAL_GROUPS):
        # v21 WS-A.1 S4: terminal groups appended to a RING parent
        # (-carbaldehyde / -carboxylic acid / -carbonitrile). Count and
        # locate only RING-ANCHORED matches (the exocyclic suffix carbon
        # bonded to a ring atom); a match wholly inside a demoted chain
        # substituent is expressed there (oxo/cyano prefix), never as a
        # second ring suffix ('cyclopentanedicarbaldehyde' bug).
        from ..composer import ring_anchored_pg_atoms
        oriented_ring = features.oriented_ring
        ring_set_local = set(oriented_ring)
        ring_idx_to_locant = {
            atom_idx: pos + 1
            for pos, atom_idx in enumerate(oriented_ring)
        }
        mol = features.mol
        anchored_locants = []
        for match in features.principal_group_atoms:
            match_set = set(match)
            anchor_locant = None
            for atom_idx in match:
                if atom_idx in ring_idx_to_locant:
                    anchor_locant = ring_idx_to_locant[atom_idx]
                    break
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() != 'C':
                    continue
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in ring_idx_to_locant:
                        anchor_locant = ring_idx_to_locant[nbr.GetIdx()]
                        break
                if anchor_locant is not None:
                    break
            if anchor_locant is not None:
                anchored_locants.append(anchor_locant)
        # AM-3 (P-66.4.1.4.2, plan P1AM Task 12): keep the MULTISET of anchor
        # locants for the appended ring-suffix family so a GEMINAL di-suffix
        # (two -carboxamide carbons on the SAME ring atom -> cyclohexane-1,1-
        # dicarboxamide) is not collapsed to a single suffix. features.
        # principal_group_atoms is already de-overlapped upstream, so each
        # entry is a DISTINCT group; only a set() would wrongly merge two
        # geminal groups. Sort the multiset (duplicates preserved).
        anchored_locants = sorted(anchored_locants)
        if anchored_locants:
            fg_count = len(anchored_locants)
            # Locant presentation: the bare mono case keeps the historical
            # locant-free form (cyclohexanecarbaldehyde); cite locants when
            # the suffix is multiplied or the ring carries other cited
            # substituents (2-(7-oxoheptyl)cyclopentane-1-carbaldehyde).
            other_subs_exist = False
            ring_subs = getattr(features, 'ring_substituents', None) or {}
            anchored_atoms = ring_anchored_pg_atoms(
                mol, features.principal_group_atoms, ring_set_local
            )
            for _pos, sub_list in ring_subs.items():
                for sub_atoms in sub_list:
                    if not (set(sub_atoms) <= anchored_atoms):
                        other_subs_exist = True
                        break
                if other_subs_exist:
                    break
            if fg_count > 1 or other_subs_exist:
                locants = tuple(anchored_locants)

    elif (not features.principal_chain
          and getattr(features, 'oriented_ring', None)
          and features.principal_group_atoms
          and fg_name not in TERMINAL_GROUPS):
        # Ring compounds: build idx_to_locant from oriented_ring and compute
        # suffix locants.  This mirrors the logic in _get_fg_locants() but
        # uses get_functional_group_locants for consistency.
        oriented_ring = features.oriented_ring
        ring_idx_to_locant = {
            atom_idx: pos + 1
            for pos, atom_idx in enumerate(oriented_ring)
        }
        # Map FG atoms to ring locants
        fg_locants = []
        mol = features.mol
        for match in features.principal_group_atoms:
            found = False
            match_set = set(match)
            ring_set_local = set(ring_idx_to_locant.keys())
            # First pass: prefer a ring C bonded to a non-ring atom in
            # the same FG match (the characteristic heteroatom, e.g.,
            # the C in C=O for ketone, C in C-OH for alcohol).
            for atom_idx in match:
                if atom_idx in ring_idx_to_locant:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        # Check if bonded to a non-ring FG atom
                        has_fg_hetero = any(
                            nbr.GetIdx() in match_set and nbr.GetIdx() not in ring_set_local
                            for nbr in atom.GetNeighbors()
                        )
                        if has_fg_hetero:
                            fg_locants.append(ring_idx_to_locant[atom_idx])
                            found = True
                            break
            if not found:
                # Second pass: any ring C in the match
                for atom_idx in match:
                    if atom_idx in ring_idx_to_locant:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.GetSymbol() == 'C':
                            fg_locants.append(ring_idx_to_locant[atom_idx])
                            found = True
                            break
            if not found:
                # Fallback: any atom in match on the ring
                for atom_idx in match:
                    if atom_idx in ring_idx_to_locant:
                        fg_locants.append(ring_idx_to_locant[atom_idx])
                        found = True
                        break
            if not found:
                # Try neighbors
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in ring_idx_to_locant:
                            fg_locants.append(ring_idx_to_locant[nbr.GetIdx()])
                            found = True
                            break
                    if found:
                        break

        fg_locants = sorted(set(fg_locants))
        if fg_locants:
            fg_count = len(fg_locants)
            locants = tuple(fg_locants)
            # P-14.3.4.2(c) (v29 Phase C tranche A). The TERMINAL ring branch
            # above already withholds a trivial mono-suffix locant -- which is why
            # `cyclohexanecarbaldehyde` is correct today -- while this
            # NON-terminal sibling cited it unconditionally, emitting
            # `cyclohexan-1-one` / `cyclopentan-1-ol` / `cyclohexane-1-thiol`
            # against the verbatim PINs `cyclohexanone` (BB:14916),
            # `cyclopentanol` (BB:26854) and `cyclohexanethiol`
            # (P-14.3.4.2(c) example).
            #
            # P-14.3.3 "Citation of locants" (BB:2869) is DENY-BY-DEFAULT, so this
            # is a narrow LICENCE, never a locant-stripping pass. It fires only
            # where the locant is provably not distinctive: a single suffix on a
            # MONOCYCLIC, fully saturated, all-carbon ring whose every other
            # position is an unsubstituted CH2. Then all substitutable positions
            # are equivalent (P-14.3.4.3 "only one kind of substitutable
            # hydrogen") and `1` carries no information.
            #
            # Everything else keeps its locant, deliberately:
            #  - heterocycles          -> `piperidine-1-carbonitrile` (BB:34730)
            #  - any ring unsaturation -> `cyclohex-2-en-1-ol` (positions differ)
            #  - any other substituent -> `4-methylcyclohexan-1-one`
            #  - multiplied suffixes   -> `cyclohexane-1,2-diol`
            #  - fused/bridged systems -> `naphthalen-1-ol` (Tranche B widens this)
            _wanted = set(fg_locants)
            _suffix_ring_atoms = {
                _a for _a, _loc in ring_idx_to_locant.items() if _loc in _wanted
            }
            if len(fg_locants) == 1 and _ring_suffix_locant_is_trivial(
                    features, oriented_ring, ring_idx_to_locant,
                    suffix_ring_atoms=_suffix_ring_atoms):
                locants = ()

    # Final safety: reconcile multiplier count with actual locants
    if locants:
        from ...rules.locant_validation import reconcile_multiplier_count
        fg_count = reconcile_multiplier_count(fg_count, list(locants))

    return NameFragment(
        text=suffix_text,
        locants=locants,
        fragment_type="suffix",
        count=fg_count,
    )


def _generate_prefixes(features: Any) -> List["NameFragment"]:
    """
    Generate prefix fragments for alkyl substituents and non-principal groups.

    For alkane/cycloalkane naming, this extracts alkyl substituents from
    features.substituents (for chains) or features.ring_substituents (for rings),
    groups them by name (methyl, ethyl, etc.), and formats with locants and
    multiplicative prefixes.

    When chain_is_parent=True (ring-chain compounds where chain won parent selection),
    rings become substituents and are named as prefixes (phenyl, cyclohexyl, etc.).
    """
    from ..composer import (
        NameFragment,
        _generate_alkyl_prefixes,
        _generate_ring_alkyl_prefixes,
        _generate_ring_substituent_prefixes,
        _get_fg_locants,
        _merge_duplicate_prefixes,
    )

    prefixes = []

    # --- Handle ring-as-substituent prefixes when chain is parent ---
    if getattr(features, 'chain_is_parent', False):
        ring_sub_prefixes = _generate_ring_substituent_prefixes(features)
        prefixes.extend(ring_sub_prefixes)

    # --- Handle alkyl substituents from features.substituents (chains) ---
    if features.substituents and features.mol:
        alkyl_prefixes = _generate_alkyl_prefixes(features)
        prefixes.extend(alkyl_prefixes)

    # --- Handle ring substituents from features.ring_substituents ---
    # Skip when chain_is_parent: ring is a substituent of the chain, not the parent.
    # Ring substituent data was populated for ring-as-substituent naming but should
    # not be used for prefix generation on the chain parent (Phase 139 ARCH-01).
    ring_substituents = getattr(features, 'ring_substituents', None)
    oriented_ring = getattr(features, 'oriented_ring', None)
    # DROP-07 fix: track FG atoms handled by ring alkyl prefixes to prevent
    # double-emission in the global FG loop below
    handled_ring_fg_atoms = frozenset()
    if ring_substituents and features.mol and oriented_ring and not getattr(features, 'chain_is_parent', False):
        ring_prefixes, handled_ring_fg_atoms = _generate_ring_alkyl_prefixes(features)
        prefixes.extend(ring_prefixes)

    # --- Handle non-principal functional groups as prefixes ---
    # When chain_is_parent, skip FGs on ring atoms (already in ring substituent name).
    # Also include non-chain atoms reachable from ring atoms: inner substituents on
    # fused heterocycles (CF3, NO2, CN, etc.) are named as part of the ring compound
    # prefix and must NOT leak as FG prefixes on the parent chain.
    ring_atom_set = set()
    if getattr(features, 'chain_is_parent', False):
        chain_set_for_expand = set(features.principal_chain)
        for rg in getattr(features, 'ring_substituents_as_groups', []):
            ring_atom_set.update(rg)
        # BFS expand: include all atoms reachable from ring atoms that are not
        # on the principal chain (captures inner substituents like CF3, NO2, CN)
        expand_queue = list(ring_atom_set)
        while expand_queue:
            aidx = expand_queue.pop()
            for nbr in features.mol.GetAtomWithIdx(aidx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_atom_set and nidx not in chain_set_for_expand:
                    ring_atom_set.add(nidx)
                    expand_queue.append(nidx)

    # BUG-B: Collect all substituent branch atom indices.
    # FG matches located entirely on a branch are handled by substituent naming,
    # so we skip them here to avoid double-counting (e.g., standalone "hydroxy"
    # when the branch is already named "(hydroxymethyl)").
    branch_atoms = set()
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                branch_atoms.update(sub_atoms)
    # Also collect ring substituent branch atoms -- the enumerator now names
    # compound substituents (trifluoromethyl, etc.) on rings, so their FG atoms
    # should not be double-counted as standalone FG prefixes.
    ring_substituents = getattr(features, 'ring_substituents', None)
    if ring_substituents:
        for _ring_idx, sub_list in ring_substituents.items():
            for sub_atoms in sub_list:
                branch_atoms.update(sub_atoms)

    # DEF-4 (P-14.3.4, Phase 171 BBR-ASM): the locant-1 omission decision must use
    # the MOLECULE-WIDE substituent count, not the per-FG-type count. Collect FG
    # prefix specs here, then emit them after the loop once the total is known
    # (composer.py:5044-5066 parity). Without this, a C1 substituent elides its
    # locant whenever its own type-count is 1 even though another substituent exists
    # (e.g. FCCCl -> '1-chloro-2-fluoroethane', not 'chloro-2-fluoroethane').
    fg_prefix_specs = []  # list of (prefix_text, fg_locants)

    # DD5 RC-4 (P-44.1.1): subtypes of the principal group's equal-seniority class
    # (e.g. secondary_alcohol when primary_alcohol is principal) are expressed
    # TOGETHER in the multiplied suffix (diol/triol) by _generate_suffix via the
    # get_principal_group class-union — they must NOT also leak here as a redundant
    # hydroxy prefix. SCOPED to the same classes the union covers (_RC4_UNION_CLASSES,
    # alcohols) so amine subtypes are untouched.
    from ...rules.seniority import _SENIORITY_PARENT as _SEN_PARENT, _RC4_UNION_CLASSES
    _principal_class = _SEN_PARENT.get(features.principal_group, features.principal_group)
    _skip_same_class = _principal_class in _RC4_UNION_CLASSES

    for fg_name, matches in features.functional_groups.items():
        if fg_name == features.principal_group:
            continue
        if _skip_same_class and _SEN_PARENT.get(fg_name, fg_name) == _principal_class:
            continue

        # Unsaturation indicators are NOT functional groups (IUPAC P-31.1).
        # They are handled as -ene/-yne infixes by _build_unsaturation_infix(),
        # not as prefixes.  Skip to avoid false DROP-16 noise.
        if fg_name in ('alkene', 'alkyne'):
            continue

        # DD2 Fix B (Phase D): peroxide / disulfide are DIVALENT chalcogen-chalcogen
        # linkages. Their substitutive substituent form ((R)peroxy / (R)disulfanyl)
        # is emitted by the substituent-branch namer, and the genuine divalent
        # bridge ('disulfanediyl', 'peroxy') is built by the multiplicative handler.
        # The monovalent FG-prefix loop must NOT also emit them, or the S-S/O-O is
        # double-counted (e.g. CSSC -> '(methyldisulfanyl)disulfanediylmethane').
        if fg_name in ('peroxide', 'disulfide'):
            continue

        # Filter out FGs on ring atoms when chain is parent
        if ring_atom_set:
            filtered = []
            for match in matches:
                # Skip FG if ANY atom in the match is part of the ring
                # substituent fragment (includes inner substituents like CF3)
                on_ring = any(a in ring_atom_set for a in match)
                if not on_ring:
                    filtered.append(match)
            matches = filtered

        # DROP-07 fix: skip FG matches already handled as ring substituents
        # by _generate_ring_alkyl_prefixes() to prevent double-emission.
        # FG SMARTS matches include anchor atoms (e.g., C-F match = (C_idx, F_idx)),
        # so check if ANY atom in the match is in the handled set.
        if handled_ring_fg_atoms:
            matches = [m for m in matches if not any(a in handled_ring_fg_atoms for a in m)]

        # BUG-B: Skip simple FG matches on small substituent branches (<=3 carbons)
        # that get named as compound substituents (hydroxymethyl, aminomethyl, etc.)
        # Uses shared BRANCH_HANDLED_FGS from naming_utils (unified in Phase 113).
        if branch_atoms and fg_name in BRANCH_HANDLED_FGS:
            filtered_branch = []
            for match in matches:
                if not all(a in branch_atoms for a in match):
                    filtered_branch.append(match)
                    continue
                # DD5 RC-4 corollary: a HALOGEN entirely within a substituent
                # branch is ALWAYS cited inside that substituent's name (the
                # haloalkyl / located / recursive namers emit it as a mandatory
                # detachable prefix), so it must NOT also leak as a standalone
                # halo prefix — regardless of branch size. (The ≤3-carbon limit
                # below is for COMPOUND FG substituents like 'hydroxymethyl'
                # whose name only includes the FG on small branches.) Fixes the
                # diol + halo-arm double-emit exposed by the new diol chain
                # selection: 3-(4-chlorobutyl)pentane-1,4-diol, NOT
                # chloro-3-(4-chlorobutyl)pentane-1,4-diol.
                if fg_name in ('fluoro', 'chloro', 'bromo', 'iodo'):
                    continue
                # Check if FG is on a small branch with carbons
                # Search both chain substituents and ring substituents
                on_small_branch = False
                # Chain substituents
                if features.substituents:
                    for _pos, sub_list in features.substituents.items():
                        for sub_atoms in sub_list:
                            sub_set = set(sub_atoms)
                            if all(a in sub_set for a in match):
                                c_count = sum(1 for a in sub_atoms
                                              if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                                if 1 <= c_count <= 3:
                                    on_small_branch = True
                                    break
                        if on_small_branch:
                            break
                # Ring substituents (for enumerator-handled compound substituents)
                if not on_small_branch and ring_substituents:
                    for _ring_idx, sub_list in ring_substituents.items():
                        for sub_atoms in sub_list:
                            sub_set = set(sub_atoms)
                            if all(a in sub_set for a in match):
                                c_count = sum(1 for a in sub_atoms
                                              if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                                if 1 <= c_count <= 3:
                                    on_small_branch = True
                                    break
                        if on_small_branch:
                            break
                if not on_small_branch:
                    filtered_branch.append(match)
            original_count = len(matches)
            matches = filtered_branch
            if not matches and original_count > 0:
                # DROP-17: BUG-B removed all matches. Per-branch containment
                # check ensures each match is genuinely on a small branch whose
                # compound name (e.g., "chloromethyl") already includes the FG.
                # Do NOT blindly restore -- that would cause double-emission.
                # Only log for diagnostic purposes.
                logger.debug(
                    "DROP-17 all_filtered: fg_name=%s original_count=%d (branch naming handles these)",
                    fg_name, original_count,
                )

        prefix_text = get_prefix(fg_name)
        if prefix_text and matches:
            count = len(matches)

            # Compute locants by mapping FG anchor atoms to chain positions
            fg_locants = _get_fg_locants(features, fg_name, matches)

            # Reconcile multiplier count with actual locants found
            if fg_locants:
                from ...rules.locant_validation import reconcile_multiplier_count
                count = reconcile_multiplier_count(count, fg_locants)

            if count > 1:
                # Add multiplier
                multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
                prefix_text = f"{multiplier}{prefix_text}"

            # DEF-4: defer the locant-1 omission decision to the post-loop block,
            # which knows the molecule-wide substituent count.
            fg_prefix_specs.append((prefix_text, fg_locants))
        elif not prefix_text and matches:
            # By-design: FGs using functional class naming (ether, sulfoxide, etc.)
            # don't have prefix forms — handled by specialized naming paths
            logger.debug(
                "DROP-16 substituent_skip: reason=no_fg_prefix_form fg_name=%s match_count=%d",
                fg_name, len(matches),
            )

    # DEF-4 (P-14.3.4, Phase 171 BBR-ASM): emit the collected FG prefixes using a
    # MOLECULE-WIDE substituent count for the locant-1 omission decision. 'prefixes'
    # already holds the alkyl/ring substituents; count their attachment positions
    # plus the FG positions. The omission only fires for a genuinely single-
    # substituent parent (total == 1); for >= 2 substituents every locant is cited
    # (mirrors composer.py:5044-5066; gold FCCCl -> '1-chloro-2-fluoroethane').
    chain_len = len(getattr(features, 'principal_chain', []))
    existing_sub_positions = sum(len(p.locants) if p.locants else 1 for p in prefixes)
    fg_sub_positions = sum(len(locs) if locs else 1 for _txt, locs in fg_prefix_specs)
    total_substituents = existing_sub_positions + fg_sub_positions
    for prefix_text, fg_locants in fg_prefix_specs:
        # NOTE: do NOT add an outer `and total_substituents == 1` guard. The
        # is_monosubstituted arg already encodes the single-substituent condition,
        # and should_omit_locant_one Rule 1 (chain_length == 1) must still omit for
        # methane regardless of substituent count (e.g. CBr4 -> 'tetrabromomethane',
        # NOT '1,1,1,1-tetrabromomethane').
        # Wave2 T2a: ring parents never reached Rule 4 here (is_ring was not
        # passed), so a lone FG prefix kept a spurious '1-' on a symmetric
        # carbocycle ('1-isocyanatocyclohexane'; BB VERBATIM PIN is
        # 'isocyanatocyclohexane', parallel to methylcyclohexane). Gate is
        # CONSERVATIVE: all-carbon ring, all ring bonds single (aromatic /
        # cycloalkene rings keep their locant — position is distinguishable).
        _sym_carbo_ring = False
        _oring = getattr(features, 'oriented_ring', None)
        if (_oring and not getattr(features, 'chain_is_parent', False)
                and features.mol is not None):
            from rdkit import Chem as _Chem
            _n_ring = len(_oring)
            _sym_carbo_ring = all(
                features.mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                for i in _oring
            ) and all(
                (lambda b: b is not None and b.GetBondType() == _Chem.BondType.SINGLE)(
                    features.mol.GetBondBetweenAtoms(
                        _oring[k], _oring[(k + 1) % _n_ring]))
                for k in range(_n_ring)
            )
        omit_locants = should_omit_locant_one(
            context="prefix",
            chain_length=chain_len,
            is_ring=_sym_carbo_ring,
            is_monosubstituted=(total_substituents == 1 and fg_locants == [1]
                                and features.principal_group is None),
        )
        prefixes.append(NameFragment(
            text=prefix_text,
            locants=tuple(sorted(fg_locants)) if (fg_locants and not omit_locants) else (),
            fragment_type="prefix"
        ))

    # --- Merge duplicate prefix names ---
    # If the same base prefix name appears multiple times (from different sources),
    # merge them into a single entry with combined count and appropriate multiplier.
    # This prevents stacking like "dihydroxyhydroxy" -> should be "trihydroxy".
    prefixes = _merge_duplicate_prefixes(prefixes)

    return prefixes


def _is_unsubstituted_monocyclic_cycloalkene(mol) -> bool:
    """True for an unsubstituted all-carbon monocyclic ring with ONE double bond.

    STER-01 (Phase H, P-93.5.1.4.1): this is exactly the Blue Book ``cyclooctene``
    / ``cyclononene`` class whose single cyclic double bond carries NO locant in
    the parent name ("cyclooctene", not "cyclooct-1-ene"). Used to decide whether
    the lone E/Z stereodescriptor's locant is likewise elided (``(E)-cyclooctene``).

    Deliberately NARROW so it can never misfire:
      * exactly one ring (NumRings == 1) -- fused/bridged systems keep locants;
      * every atom is carbon AND every atom is in the ring (no substituent and no
        heteroatom) -- a substituent or suffix (e.g. cyclooct-2-en-1-ol) can push
        the double bond off locant 1 and the parent name then CITES the locant, so
        those must keep "(2E)-"; restricting to the bare hydrocarbon avoids that
        entirely;
      * exactly one double bond, in the ring, and no triple bond -- a di-ene
        (cyclodeca-1,3-diene) keeps "(1Z,3E)" via the len(descriptors)>1 guard at
        the call site, and this is belt-and-suspenders.
    """
    ri = mol.GetRingInfo()
    if ri.NumRings() != 1:
        return False
    ring = ri.AtomRings()[0]
    if mol.GetNumAtoms() != len(ring):
        return False  # substituent atoms present -> not the bare cycloalkene
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in range(mol.GetNumAtoms())):
        return False  # heteroatom -> would carry a suffix locant
    double_bonds = [b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0]
    if len(double_bonds) != 1 or not double_bonds[0].IsInRing():
        return False
    if any(b.GetBondTypeAsDouble() == 3.0 for b in mol.GetBonds()):
        return False
    return True


def _generate_stereodescriptors(features: Any, atom_to_locant_override: Optional[Dict[int, int]] = None) -> Optional["NameFragment"]:
    """
    Generate stereodescriptor prefix using correct IUPAC locants.

    Uses the stereochemistry rules module to collect R/S and E/Z descriptors
    based on the atom_to_locant mapping (from chain/ring orientation).

    For different compound types:
    - Acyclic: uses features.atom_to_locant (from principal chain orientation)
    - Heterocycles: uses features.heterocycle_atom_to_locant
    - Cycloalkanes/cycloalkenes: builds from features.oriented_ring

    When atom_to_locant_override is provided, it takes priority over all
    features.* fields. This allows early-return handlers to pass the correct
    locant map for the structure they are naming (which may differ from
    features.principal_ring).

    Args:
        features: MolecularFeatures object with stereocenters and/or double_bond_stereo
        atom_to_locant_override: Optional explicit locant map. When provided,
            bypasses the features.* priority chain entirely.

    Returns:
        NameFragment with stereodescriptor prefix like "(2R)-" or "(2E,3R)-",
        or None if no stereodescriptors.
    """
    from ..composer import NameFragment
    from ...rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Need either stereocenters or double_bond_stereo
    if not features.stereocenters and not getattr(features, 'double_bond_stereo', None):
        return None

    mol = features.mol

    # When an explicit override is provided, use it directly instead of
    # reading from features.* fields. This allows early-return handlers
    # to pass the correct locant map for the structure they are naming
    # (which may differ from features.principal_ring).
    if atom_to_locant_override:
        atom_to_locant = atom_to_locant_override
    else:
        # Existing priority chain (unchanged):
        atom_to_locant = features.atom_to_locant

        # For heterocycles, use ring-specific mapping
        if getattr(features, 'heterocycle_atom_to_locant', None):
            atom_to_locant = features.heterocycle_atom_to_locant
        elif getattr(features, 'oriented_ring', None):
            # Build mapping from oriented_ring for cycloalkanes/cycloalkenes
            # oriented_ring is a list of atom indices in ring order starting at position 1
            # This creates ring_atom_to_locant: {atom_idx: ring_locant} where locants are 1-indexed
            atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(features.oriented_ring)}

    if not atom_to_locant:
        return None

    # Collect stereodescriptors with proper locants.
    # Enable near-parent E/Z detection for top-level naming only:
    # substituent E/Z bonds one hop from the parent ring/chain should
    # be included in the stereo block per IUPAC P-93.5.2.
    descriptors = collect_stereodescriptors(
        mol, atom_to_locant, include_near_parent_ez=True
    )

    if not descriptors:
        return None

    # STEREO-06 (Phase 177 WSB-02 D-10): mononuclear-parent locant omission.
    # When the parent hydride is a single skeletal atom (len(atom_to_locant)==1)
    # and the lone stereodescriptor's locant is that unique position, the locant
    # is omitted per P-91 (a unique position carries no locant) -> bare "(R)-".
    # Example: [C@H](Br)(Cl)F has parent methane {C_idx: 1}; the descriptor
    # (1,'R') must format as "(R)-", not the spurious "(1R)-".
    if (
        len(atom_to_locant) == 1
        and len(descriptors) == 1
        and isinstance(descriptors[0][0], int)
        and descriptors[0][0] == next(iter(atom_to_locant.values()))
    ):
        # Strip the locant: emit the bare descriptor "(R)-".
        cip = descriptors[0][1]
        return NameFragment(text=f"({cip})-", fragment_type="stereo")

    # STER-01 (Phase H, P-93.5.1.4.1): single cyclic double bond on an
    # UNSUBSTITUTED monocyclic cycloalkene. The parent name elides the
    # unsaturation locant ("cyclooctene", not "cyclooct-1-ene"), so the lone
    # E/Z stereodescriptor's locant is elided too -> the Blue Book PINs
    # "(E)-cyclooctene" / "(E)-cyclononene". Guarded by
    # _is_unsubstituted_monocyclic_cycloalkene to the exact BB class so it can
    # never touch a substituted/heteroatom ring (keeps "(2E)-cyclooct-2-en-1-ol")
    # or a multi-element block (cyclodeca-1,3-diene keeps "(1Z,3E)" via len==1).
    # The prior "(1Z)-cyclooctene" was over-specified (OPSIN-parseable, not wrong).
    if (
        len(descriptors) == 1
        and descriptors[0][1] in ('E', 'Z')
        and _is_unsubstituted_monocyclic_cycloalkene(mol)
    ):
        cip = descriptors[0][1]
        return NameFragment(text=f"({cip})-", fragment_type="stereo")

    # W4-S1 folded-fix #5 RECONCILE (P-93.4.2.1.1): the audit wanted the ethene
    # E/Z locant ELIDED (`F/C=C/F` -> `(E)-1,2-difluoroethene`). It is WRONG.
    # BB P-93.4.2.1.1 is explicit — "Locants ... are used before the
    # stereodescriptors E and Z" — and gives the verbatim ethene PIN
    # `(1Z)-1,2-dibromo-1-chloro-2-iodoethene (PIN)` WITH the `1` locant. HEAD's
    # `(1E)-1,2-difluoroethene` is therefore already the PIN; NO elision here.
    # (The STER-01 cyclooctene elision above is a DIFFERENT rule — P-93.5.1.4.1,
    # where the PARENT NAME itself elides the ring double-bond locant.)

    # Format as "(2R,3S)-" etc
    text = format_stereodescriptor_string(descriptors)

    return NameFragment(text=text, fragment_type="stereo")


def _l3_prefix_locant_omitted(features: Any, fragments: List["NameFragment"]) -> bool:
    """P-14.3.4.3 (BB:2939) "Omission of locants" -- the substituent-PREFIX case.

        "The locant is omitted in monosubstituted symmetrical parent hydrides or
         parent compounds where there is only one kind of substitutable hydrogen."

    Its example block prints `chloropropanedioic acid (PIN)` (BB:2951) and
    `methylurea (PIN)` (BB:2943): a PREFIX substitution, the mirror of the ring-suffix
    case wired in `rules/heterocycles.py`. Both delegate the decision to
    `assembly.locant_omission`, the one place the P-14.3.4 licences live.

    ★ Read together with P-14.3.4.1 (BB:2877), which is ALREADY live here and is a
    DIFFERENT rule: it withdraws only the TERMINAL (suffix) locants, which is what
    makes `HOOC-CH2-CH2-COOH` `butanedioic acid` rather than `butane-1,4-dioic acid`.
    It says nothing about substituent locants. Conflating the two would emit
    `chloropropanediamide` for the class of `2-methylpropanediamide (PIN)` (BB:2887),
    which sits in P-14.3.4.1's OWN example block WITH its locant. The two rows only
    look contradictory until the split is applied: L1 takes the suffix locants off
    both, and L3 then decides the substituent locant separately -- omitting it on the
    di-ACID (propanedioic acid's sole substitutable position is C2, the two acid O-H
    being chalcogen H excluded by BB:3007) and keeping it on the di-AMIDE (C2 *and*
    the amide N-H, which are substitutable -- proven by BB:2889
    `N1,N3-dimethylpropanediamide (PIN)`). Nothing about chain length, nothing about
    the substituent.

    Returns True only when every essential-locant escape has been positively
    excluded; anything unestablished retains the locant (P-14.3.3 is deny-by-default).
    """
    import re as _re

    if features is None or not fragments:
        return False

    prefixes = [f for f in fragments if f.fragment_type == "prefix"]
    suffixes = [f for f in fragments if f.fragment_type == "suffix"]
    parents = [f for f in fragments if f.fragment_type == "parent"]
    stereos = [f for f in fragments if f.fragment_type == "stereo"]

    # "monosubstituted": exactly ONE prefix, carrying exactly ONE locant, once.
    if len(prefixes) != 1 or len(parents) != 1 or len(suffixes) > 1:
        return False
    pf = prefixes[0]
    pf_locants = list(pf.locants or [])
    if len(pf_locants) != 1 or getattr(pf, "count", 1) != 1:
        return False
    # A prefix whose TEXT already begins with a digit names its own locants (a ring
    # or complex substituent); it is not the simple mono prefix this licence covers,
    # and the assembler's `already_has_locant` branch would leave them in place.
    if _re.match(r"^\d", pf.text or ""):
        return False

    # P-14.3.3: any other locant cited in the same scope restores them all.
    if stereos and str(stereos[0].text or "").strip():
        return False
    stem = parents[0].text or ""
    if not stem.isalpha():
        # A stem that is not purely alphabetic already cites something -- an
        # indicated hydrogen or a locant baked into the parent name.
        # ⚠ MUTATION-SURVIVING, DELIBERATELY KEPT (measured 2026-07-29): deleting this
        # breaks no test, because no molecule that reaches this assembler with a single
        # locant-bearing prefix has a non-alphabetic stem -- chain stems ('prop', 'but')
        # and 'cyclohex' are all alpha, and indicated-hydrogen parents are named by
        # other handlers. It states one clause of P-14.3.3 explicitly and defends the
        # licence if another clause is ever loosened. Do not "simplify" it; the same
        # reasoning is already recorded for the over-determined conditions of
        # `_ring_suffix_locant_is_trivial` above.
        return False
    for bond_locants in (parents[0].locants or ()):
        if bond_locants:
            return False                    # unsaturation locant in the parent scope
    if suffixes and list(suffixes[0].locants or []):
        # ⚠ ALSO MUTATION-SURVIVING AND DELIBERATELY KEPT. Unreachable today for the
        # same reason: the orbit test is restrictive enough that no parent compound
        # whose suffix still cites a locant also has just one kind of substitutable
        # hydrogen. It is the direct statement of P-14.3.3's *"then all locants must be
        # cited"* for the suffix half of the scope, and it is what would stop the
        # licence the moment L1 (P-14.3.4.1) stopped withdrawing the terminal locants.
        return False                        # suffix locant still cited in the scope

    # The parent COMPOUND is the parent skeleton PLUS its principal characteristic
    # group(s) -- for `chloropropanedioic acid` the three chain carbons and BOTH
    # -COOH. Everything else is the one substitution, and
    # `l3_locant_omitted_for_parent_atoms` proves structurally that it is exactly one.
    parent_atoms = set()
    for seq in (getattr(features, "principal_chain", None) or (),
                getattr(features, "principal_ring", None) or ()):
        for idx in seq:
            parent_atoms.add(int(idx))
    for match in (getattr(features, "principal_group_atoms", None) or ()):
        for idx in match:
            parent_atoms.add(int(idx))
    if not parent_atoms:
        return False

    from ..locant_omission import l3_locant_omitted_for_parent_atoms

    return l3_locant_omitted_for_parent_atoms(
        getattr(features, "mol", None),
        parent_atoms,
        prefix_locants=pf_locants,
        suffix_locants=[],
        parent_cites_locants=False,
        stereo_text="",
    )


def _naming_call_produces_a_name_component() -> bool:
    """True while this naming call's result will be spliced into a LARGER name.

    ``P-14.3.3`` (BB:2869) scopes locant citation to *"the parent structure or ... a
    unit of structure **as defined by its appropriate enclosing marks**"*. A licence
    that empties a scope of ALL its locants therefore may only fire when the scope's
    boundary is known -- and it is known only when the molecule being named IS the
    whole compound. It is NOT known when the "molecule" is a decomposition fragment
    standing in for a substituent, because the enclosing scope's own locants are
    invisible from inside.

    ★ MEASURED 2026-07-30, and it is the defect invariant 11 exists to catch. For
    ``F(CF2)7-CO-N(piperidine)`` the decomposition engine cuts the acyl bond, CAPS the
    fragment as the free acid, and names ``pentadecafluorooctanoic acid`` as a
    standalone molecule -- for which P-14.3.4.5 genuinely fires, 15 of 15 -- then
    rewrites ``oic acid`` -> ``oyl`` and splices it in as ``N-...oylpiperidine``. That
    scope cites ``N``, an essential letter locant, so P-14.3.3 restores every locant;
    and BB:29619 says so about this very molecule in as many words: *"(PIN, the locants
    for the fluoro substituents are required, see P-14.3.4.5)"*. Without this guard the
    licence emitted ``N-pentadecafluorooctanoylpiperidine``, contradicting the Blue
    Book's own explicit negative for the rule it was implementing.

    The signal is ``assembly.fragment_naming``'s visited-SMILES set, read-only. That
    module's own comment calls its recursive entry point *"THE chokepoint where a
    WHOLE-MOLECULE naming becomes a NAME COMPONENT"*; it adds the fragment's canonical
    SMILES before delegating to ``name_compound`` and discards it in a ``finally``, so
    a non-empty set means exactly "a nested fragment naming is in progress".

    Spy-validated on 3 known positives (``heptafluorobutanoic acid``,
    ``pentafluoropropanoic acid``, ``hexafluoroethane`` -- all top-level, all empty)
    and the known negative above (non-empty, holding precisely the capped surrogate
    ``O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F``).

    Kept HERE rather than in ``locant_omission`` so that module stays the pure
    "RDKit mols in, booleans out" leaf its docstring promises.
    """
    try:
        from ..fragment_naming import _get_visited
    except Exception:                          # noqa: BLE001 -- deny-by-default
        return True
    try:
        return bool(_get_visited())
    except Exception:                          # noqa: BLE001 -- deny-by-default
        return True


def _l5_prefix_locants_omitted(features: Any, fragments: List["NameFragment"]) -> bool:
    """§**P-14.3.4.5** (BB:3007) "Omission of locants" -- the PARENT-scope case.

        "All locants are omitted in compounds or substituent groups in which all
         substitutable positions are completely substituted or modified, for example,
         by hydro, in the same way. Except for hydrogen atoms attached to chalcogen
         atoms, such as in acids, alcohols, and to the carbon atoms of formyl groups
         (aldehydes), all hydrogen atoms are considered substitutable."

    and its counter-clause BB:3009: *"In case of partial substitution or modification,
    all numerical prefixes must be indicated. The prefix 'per-' is no longer
    recommended."*

    True => this scope's substituent prefixes cite NO locants:
    ``heptafluorobutanoic acid`` (BB:3017, verbatim ``(PIN)``).

    ★ **THE SHARPEST BOUNDARY IN THE CLASS**, and it falls straight out of BB:3007's
    chalcogen carve-out with no special case::

        F3C-CF2-COOH   ->  heptafluoro... / pentafluoropropanoic acid   OMITS
        F3C-CF2-CO-NH2 ->  2,2,3,3,3-pentafluoropropanamide            KEEPS

    Identical fluorination. The acid's own O-H sits on a CHALCOGEN and is excluded
    from the substitutable count, so C2/C3 are the whole substitutable set and
    fluorine exhausts it. Propanamide's set is C2 + C3 **+ the amide N-H**, which are
    neither chalcogen H nor formyl H -- so they count, they are unsubstituted, the
    substitution is *partial*, and BB:3009 restores every locant. That an amide N-H is
    substitutable is proven independently by BB:2889 ``N1,N3-dimethylpropanediamide
    (PIN)``. Any wiring that moves the amide row has implemented "fluorines everywhere
    => drop locants", not P-14.3.4.5.

    ⚠ **Read as a carve-out of the DENY-DEFAULT** P-14.3.3 (BB:2869). Everything this
    function cannot positively establish returns False, and the rule itself is NOT
    re-derived here: the decision is delegated to
    ``assembly.locant_omission.l5_uniform_complete``, the one place the P-14.3.4
    licences live. This function only proves the preconditions and marshals the scope.

    ★ **The class is OPEN.** BB:3009 retires the ``per-`` contraction, which *was*
    exactly a closed-list mechanism, and the 2013 recommendations replaced it with
    counting -- so a table keyed on ``"heptafluoro"`` would be wrong on its complement
    by construction (``nonafluoropentanoic acid``, ``pentachloropropanoic acid``,
    ``octafluoropropane``, ``hexafluoropentanedioic acid`` are all entailed and none is
    printed anywhere in the Blue Book). Structural predicate only.

    Sibling: :func:`_l3_prefix_locant_omitted` (P-14.3.4.3) above. The two are
    disjoint in practice -- L3 needs exactly ONE cited locant, L5 needs every
    substitutable hydrogen replaced -- and where both could hold they agree.

    **Why the fragments and not a flag.** The caller applies a True answer by
    REBUILDING the prefix fragments with empty ``locants``, not by threading a
    print-time suppression flag into :func:`_assemble_fragments` the way the older
    ``l3_omit_prefix_locant`` does. The fragment list is the single input to BOTH the
    legacy assembler and ``name_tree_builder.fragments_to_tree``, and the tree IS the
    production composition site for this handler -- ``general_acyclic`` is the sole
    member of ``name_tree_to_string.SERIALIZER_PRODUCTION_CLASSES``. A flag reaching
    only one of the two makes them disagree, and ``composer._serializer_flip_or_name``
    then silently falls back to the legacy string, so the flip becomes a no-op for
    exactly the rows the licence touched. Measured 2026-07-30: that is already the
    case for ``chloropropanedioic acid`` (legacy ``chloropropanedioic acid`` vs
    serialized ``2-chloropropanedioic acid`` -> DIVERGE), while all three L5 rows
    probed AGREE. Removing the locants from the NAME STRUCTURE is also what
    P-14.3.4.5 actually says.
    """
    from ..locant_omission import (
        l5_uniform_complete, locants_are_forced, scope_forces_locants,
        scope_has_isotopic_modification, substitutable_h_count,
    )

    # P-14.3.3 (BB:2869) as an AMBIENT scope. ``rules/isotopes.py`` names an
    # isotope-STRIPPED skeleton, so the structural ``has_isotope`` computed below is
    # blind by construction for a labelled input and cannot be the guard.
    if locants_are_forced():
        return False
    # ⚠ AND the weaker declaration: measured 2026-07-29 (Task 5a), the isotope
    # decorator enters ``forced_locant_scope`` only CONDITIONALLY, so
    # ``locants_are_forced()`` alone reads False for a molecule whose finished name
    # still carries ``(13C1)``. This licence empties its scope of ALL prefix locants,
    # which is exactly what **P-82.6.1.1** (BB:44180) forbids when an isotopic
    # modification needs a locant to state its position.
    if scope_has_isotopic_modification():
        return False
    # ⚠ AND the third P-14.3.3 scoping fact: this licence empties a scope completely,
    # so it may only fire when the scope's boundary IS the molecule being named. See
    # :func:`_naming_call_produces_a_name_component` -- without it, BB:29619's own
    # explicit negative lost its locants.
    if _naming_call_produces_a_name_component():
        return False

    if features is None or not fragments:
        return False
    mol = getattr(features, "mol", None)
    if mol is None:
        return False

    prefixes = [f for f in fragments if f.fragment_type == "prefix"]
    suffixes = [f for f in fragments if f.fragment_type == "suffix"]
    parents = [f for f in fragments if f.fragment_type == "parent"]
    stereos = [f for f in fragments if f.fragment_type == "stereo"]
    if not prefixes or len(parents) != 1 or len(suffixes) > 1:
        return False

    # ------------------------------------------------------------------ #
    # P-14.3.3: anything ELSE cited in this scope restores every locant.  #
    # ------------------------------------------------------------------ #
    if stereos and str(stereos[0].text or "").strip():
        return False
    stem = parents[0].text or ""
    if not stem.isalpha():
        # A stem that is not purely alphabetic already cites something -- an indicated
        # hydrogen or a locant baked into the parent name.
        return False
    for bond_locants in (parents[0].locants or ()):
        if bond_locants:
            # An unsaturation locant (``but-2-enoic acid``) is essential, so P-14.3.3
            # restores the substitution locants. This is also the deny-by-default side
            # of a boundary the Blue Book does not print: ``tetrafluoroethene``'s
            # parent name cites no locant of its own (P-14.3.4.2(d) omits it for
            # unsubstituted dinuclear alkenes), so the elided form is arguably
            # licensed -- but no printed example settles it, so we retain.
            return False
    if suffixes and list(suffixes[0].locants or []):
        # ★ LOAD-BEARING, and it is a rule not a convenience: the cited suffix locant
        # IS essential, so P-14.3.3's *"then all locants must be cited"* restores the
        # prefix locants. This is the only thing that keeps
        # ``1,1,2,2,3,3,3-heptafluoropropan-1-ol`` (whose propane skeleton IS
        # completely and uniformly fluorinated -- 7 of 7 -- yet whose ``-1-ol``
        # distinguishes it from propan-2-ol) and, for the same reason,
        # ``1,1,1,3,3,3-hexafluoropropan-2-one`` (BB:3005 lists ``propan-2-one``
        # among the four spellings that keep their locant even when unambiguous) and
        # ``1,1,2,2,2-pentafluoroethane-1-thiol``.
        return False
    for pf in prefixes:
        # A prefix whose TEXT already begins with a digit names its own locants (a
        # ring or complex substituent). Those are a different enclosing-mark scope,
        # and the assembler's ``already_has_locant`` branch would leave them in place,
        # so the scope would still cite a locant.
        if re.match(r"^\d", pf.text or ""):
            return False
        if not pf.text or not str(pf.text).strip():
            return False
        if not pf.locants:
            return False
        # ⚠ INVARIANT: the multiplier must survive the elision. Every prefix that
        # reaches this assembler bakes its multiplier INTO ``text``
        # (``heptafluoro``) and carries ``count == 1`` -- ``_assemble_fragments``
        # ignores ``count`` entirely, so a ``count > 1`` prefix relying on the
        # name-tree serializer's ``multiplicative_prefix`` to supply the multiplier
        # would silently LOSE it once the locants (and with them the only thing the
        # legacy path prints) go. Deny rather than emit ``fluorobutanoic acid`` for
        # seven fluorines. (Task 5a hit the same class from the other side: dropping
        # the locants there lost the ENCLOSING MARKS.)
        if getattr(pf, "count", 1) != 1:
            return False

    # ------------------------------------------------------------------ #
    # The scope must be an ACYCLIC ALL-CARBON SATURATED CHAIN parent.     #
    # ------------------------------------------------------------------ #
    # Independently re-established here (not taken from the handler's name) so a
    # second call site cannot be wired against a scope this marshalling does not fit.
    chain = [int(i) for i in (getattr(features, "principal_chain", None) or ())]
    if not chain or len(set(chain)) != len(chain):
        return False
    if getattr(features, "oriented_ring", None) or getattr(features, "principal_ring", None):
        # A ring parent's locants come from the ring orientation, and its L5 case is
        # ``benzenehexol`` -- already wired in ``rules/benzene.py`` against a ring
        # parent hydride. ``cyclohexanecarboxylic acid`` reaches this handler with an
        # EMPTY principal_chain, so it is denied here by construction.
        return False
    n_atoms = mol.GetNumAtoms()
    if any(i < 0 or i >= n_atoms for i in chain):
        return False
    ring_info = mol.GetRingInfo()
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        # ⚠ The all-carbon clause is MUTATION-SURVIVING and UNREACHABLE today (measured
        # 2026-07-30, mutation M12): no heteroatom-chain parent reaches this handler at
        # all -- ``F[Si](F)(F)[Si](F)(F)F``, ``Cl[Si](Cl)(Cl)[Si](Cl)(Cl)Cl``,
        # ``FN(F)N(F)F`` and ``F[P](F)[P](F)F`` all emit ``unknown organic compound``,
        # and the germanium analogue is refused upstream. It is a deliberate SCOPE
        # NARROWING, not a correctness guard: ``hexafluorodisilane`` would in fact be
        # licensed by P-14.3.4.5 the day disilane becomes nameable, and the count would
        # then have to be re-derived for a chain whose atoms are not all tetravalent.
        # Deny-by-default until that is done.
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            return False
        if ring_info.NumAtomRings(idx) > 0:
            return False
        if atom.GetIsotope():
            return False
    for a, b in zip(chain, chain[1:]):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return False

    # ``chain[pos]`` is locant ``pos + 1`` -- the same map ``_get_fg_locants``
    # (composer.py:6485) builds to ASSIGN these locants. It is not trusted: the
    # ``n_out`` cross-check below re-derives the decoration count at every locant
    # from the STRUCTURE and requires it to equal the count the name cites, so a
    # wrong or reversed mapping denies instead of silently mis-measuring.
    idx_to_locant = {a: pos + 1 for pos, a in enumerate(chain)}

    # The parent COMPOUND is the chain PLUS the principal characteristic group(s) --
    # for ``heptafluorobutanoic acid`` the four chain carbons and the whole -COOH.
    parent_set = set(chain)
    for match in (getattr(features, "principal_group_atoms", None) or ()):
        for idx in match:
            parent_set.add(int(idx))
    if any(i < 0 or i >= n_atoms for i in parent_set):
        return False
    for idx in parent_set:
        if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
            return False
        if mol.GetAtomWithIdx(idx).GetIsotope():
            return False

    # ------------------------------------------------------------------ #
    # Marshal the cited locants into a per-position decoration map.        #
    # ------------------------------------------------------------------ #
    # EVERY prefix is included, not only the halogens -- otherwise a scope whose
    # uniformity is broken by a non-halogen substituent could not be detected, which
    # is precisely BB:29619's ``...pentadecafluorooctan-1-one (PIN, the locants for
    # the fluoro substituents are required, see P-14.3.4.5)``: every carbon there has
    # zero hydrogens, but C1 is substituted by something that is not fluorine.
    kind_at: Dict[int, set] = {}
    count_at: Dict[int, int] = {}
    for pf in prefixes:
        for loc in pf.locants:
            if isinstance(loc, bool) or not isinstance(loc, int):
                return False          # a letter locant (N-, N1-) is always essential
            if not 1 <= loc <= len(chain):
                return False
            kind_at.setdefault(loc, set()).add(pf.text)
            count_at[loc] = count_at.get(loc, 0) + 1
    # Two kinds at one position is not "in the same way" -> BB:3009.
    # ⚠ MUTATION-SURVIVING (M8's sibling; measured 2026-07-30 as M5) and kept as the
    # explicit statement of the rule. Its removal is now provably harmless rather than
    # seed-dependent: the ``"|".join(sorted(...))`` key built below hands a mixed
    # position a composite kind, which ``l5_uniform_complete``'s own uniformity test
    # refuses on every hash seed. See the comment there -- that determinism is the
    # thing that had to be fixed, not this check.
    if not kind_at or any(len(kinds) != 1 for kinds in kind_at.values()):
        return False

    # ★ THE MAPPING IS A MEASUREMENT, NOT AN ASSUMPTION. For each chain atom, count
    # the bonds leaving the parent compound: that is how many decorations really sit
    # there, independent of any name string. Requiring it to equal the cited count at
    # that locant states, structurally, that (a) the locant->atom map is right, (b)
    # every decoration in the molecule is one of the prefixes counted above, and (c) no
    # decoration hangs off a principal-group atom (an N-substituent), whose locant is
    # a letter and whose position this marshalling cannot express.
    #
    # ⚠ MUTATION-SURVIVING, DELIBERATELY KEPT (measured 2026-07-30, mutation M4). It is
    # OVER-DETERMINED by ``l5_uniform_complete``'s own per-position test, and the reason
    # is a small proof worth recording: in the parent hydride the substitutable-H count
    # at an atom EQUALS the number of decorations removed from it (each displaced
    # exactly one H). So any locant->atom bijection error either moves a count onto an
    # atom whose H count differs -- which ``l5_uniform_complete`` rejects -- or is an
    # automorphism of the count profile, in which case the licence's answer is
    # unchanged. Purpose (b) is covered for the same reason: an uncounted decoration
    # makes the cited count strictly less than the H count. Kept because it is the
    # clause that makes P-14.3.3's "the map must be structural" explicit, and because
    # the redundancy is a property of TODAY's marshalling (all-carbon, one prefix kind
    # per position), not of the rule. Same reasoning already recorded for the
    # over-determined clauses of ``_l3_prefix_locant_omitted`` above.
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        n_out = sum(1 for nb in atom.GetNeighbors() if nb.GetIdx() not in parent_set)
        if count_at.get(idx_to_locant[idx], 0) != n_out:
            return False
    # ...and nothing may hang off a NON-chain parent atom either (the acid O-H, the
    # amide N-H): such a decoration is not in ``count_at`` at all, so the per-chain
    # check above cannot see it.
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in parent_set) == (b in parent_set):
            continue
        inside = a if a in parent_set else b
        if inside not in idx_to_locant:
            return False

    # P-14.3.3, the residual clauses. Fail-closed on anything unestablished.
    if scope_forces_locants(
        prefix_locants=[loc for pf in prefixes for loc in pf.locants],
        suffix_locants=[],
        stereo_text="",
        # Indicated hydrogen is a ring/tautomer device and the scope is verified
        # acyclic; multiplicative names and ring assemblies name their scope
        # elsewhere and never arrive here as one acyclic chain; skeletal replacement
        # needs a heteroatom IN the chain, verified all-carbon above.
        has_indicated_h=False,
        has_isotope=any(a.GetIsotope() for a in mol.GetAtoms()),
        is_multiplicative=False,
        is_ring_assembly=False,
        has_skeletal_replacement=False,
    ):
        return False

    # ------------------------------------------------------------------ #
    # Build the PARENT COMPOUND and hand the rule its own question.       #
    # ------------------------------------------------------------------ #
    # P-14.3.4.5 speaks of the substitutable positions of the PARENT, so the licence
    # must be measured against the UNDECORATED skeleton: on the input molecule a
    # fully substituted carbon has zero hydrogens and the count that DECIDES the
    # licence would be lost. Deleting the decorations lets RDKit restore the implicit
    # hydrogens they displaced. Same reason, and the same idiom, as
    # ``rules/benzene.py::_benzene_parent_hydride`` and
    # ``locant_omission.l3_locant_omitted_for_parent_atoms``.
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for idx in sorted(range(n_atoms), reverse=True):
        if idx not in parent_set:
            rw.RemoveAtom(idx)
    parent = rw.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except Exception:                          # noqa: BLE001 -- deny-by-default
        return False
    kept = sorted(parent_set)
    if parent.GetNumAtoms() != len(kept):
        return False
    # Deleting in DESCENDING index order preserves the survivors' relative order, so
    # the new index of a kept atom is its rank in ``kept``. Verified rather than
    # assumed -- a mismatch of element or degree denies.
    old_to_new = {}
    for new_i, old_i in enumerate(kept):
        if (parent.GetAtomWithIdx(new_i).GetAtomicNum()
                != mol.GetAtomWithIdx(old_i).GetAtomicNum()):
            return False
        old_to_new[old_i] = new_i
    for a, b in ((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()) for bd in mol.GetBonds()):
        if a in parent_set and b in parent_set:
            if parent.GetBondBetweenAtoms(old_to_new[a], old_to_new[b]) is None:
                return False

    # ⚠ ``"|".join(sorted(...))`` and NOT ``next(iter(kinds))``. A position carrying two
    # kinds is already refused above, so this join can only ever see one element today
    # -- but ``next(iter(set))`` would make the FALLBACK behaviour hash-order dependent
    # if that check were ever weakened: for ``2-chloro-2,3,3,3-tetrafluoropropanoic
    # acid`` C2 holds {chloro, tetrafluoro} and its cited count (2) equals its
    # substitutable-H count (2), so an arbitrary pick of ``tetrafluoro`` would present
    # ``l5_uniform_complete`` with ONE uniform kind and elide a locant the Blue Book
    # requires -- on some hash seeds only. Task 5a hit exactly this seed-dependent
    # escape. A sorted join is deterministic and, being unequal to the single-kind key
    # at the other positions, is refused by ``l5_uniform_complete``'s own uniformity
    # test on EVERY seed.
    decoration_of = {
        old_to_new[chain[loc - 1]]: "|".join(sorted(kinds))
        for loc, kinds in kind_at.items()
    }
    counts = {old_to_new[chain[loc - 1]]: n for loc, n in count_at.items()}

    # A decorated position whose parent hydride has NO substitutable hydrogen would make
    # the aggregate arithmetic unsound (it could offset a bare position), so it is
    # rejected outright rather than left to ``l5_uniform_complete``, whose docstring
    # deliberately tolerates such entries for ``decahydronaphthalene``'s bridgeheads.
    #
    # ⚠ MUTATION-SURVIVING, DELIBERATELY KEPT (measured 2026-07-30, mutation M8), and
    # UNREACHABLE for today's scope -- provably: a decorated chain atom's parent-hydride
    # H count is at least the number of decorations removed from it, so it is >= 1; the
    # only way to reach 0 is BB:3007's carve-outs, and neither applies (a chain atom is
    # verified all-carbon, so not a chalcogen; a formyl carbon has exactly one H and
    # would not be a decorated position). It becomes load-bearing the moment the scope
    # is widened to heteroatom chains, where a decorated chalcogen IS possible.
    for new_i in decoration_of:
        if substitutable_h_count(parent, new_i) <= 0:
            return False

    return l5_uniform_complete(parent, decoration_of=decoration_of, counts=counts)


def _assemble_fragments(
    fragments: List["NameFragment"],
    style: str,
    is_mononuclear_parent: bool = False,
    l3_omit_prefix_locant: bool = False,
) -> str:
    """
    Assemble fragments into final name string.

    Order: stereo + prefixes (alphabetized) + parent + suffix

    For functional group compounds, uses PIN-style infix locants:
    - propan-1-ol (not propanol or 1-propanol)
    - butan-2-one (not butanone or 2-butanone)

    For unsaturated hydrocarbons, includes bond locants:
    - but-1-ene (not butene or 1-butene)
    - pent-1-en-4-yne (enyne)

    Args:
        fragments: The NameFragment list (parent/suffix/prefix/stereo).
        style: Naming style ("pin").
        is_mononuclear_parent: True when the perceived parent skeleton has
            exactly ONE heavy (non-H) atom, of ANY element (C, Si, P, B, Ge,
            Sn, ...), regardless of whether a characteristic-group suffix is
            present. Computed structurally by the caller from `features` (parent
            atom count == 1) — NOT by string-matching the stem. Governs the
            P-16.5.1.3.1 mononuclear enclosing rule below. Default False so the
            other (multi-atom) callers are byte-identical.
        l3_omit_prefix_locant: the P-14.3.4.3 (BB:2939) licence, decided by
            `_l3_prefix_locant_omitted` in the caller — where `features` exists, as
            for `is_mononuclear_parent`. True means the scope's SINGLE substituent
            prefix cites no locant (`chloropropanedioic acid`, BB:2951). NOT a
            locant-stripping flag: it is only ever set by a positively-licensed
            structural predicate, and defaults False so every other caller of this
            shared assembler stays byte-identical.
    """
    from ..composer import NameFragment
    # Phase 179 (D-03): the composition-grammar primitives now live in the leaf
    # module composition_primitives.py (single source of truth shared with the
    # name-tree serializer). NameFragment stays in composer.
    from ..composition_primitives import (
        _build_hydrocarbon_name,
        _build_unsaturation_infix,
        _estimate_parent_size_from_name,
        _join_prefix_to_name,
        _join_prefixes,
    )

    stereo = ""
    prefixes = []
    parent_frag = None
    suffix_frag = None

    for frag in fragments:
        if frag.fragment_type == "stereo":
            stereo = frag.text
        elif frag.fragment_type == "prefix":
            prefixes.append(frag)
        elif frag.fragment_type == "parent":
            parent_frag = frag
        elif frag.fragment_type == "suffix":
            suffix_frag = frag

    # ----------------------------------------------------------------
    # Detect suffix-prefix locant collisions on ring systems.
    # A collision occurs when a suffix locant (e.g., ketone at position 3)
    # and a prefix locant (e.g., methyl at position 3) share the same
    # numeric value.  OPSIN interprets this as both groups on the same
    # carbon, producing an unphysical valency.
    # Resolution: remove the colliding prefix locant (suffix has priority
    # per IUPAC P-14.7).  Only applies to ring parents.
    # ----------------------------------------------------------------
    if suffix_frag and suffix_frag.locants and prefixes:
        # Determine if parent is a ring (collision only matters for rings)
        parent_text = parent_frag.text if parent_frag else ""
        is_ring_parent = any(
            kw in parent_text.lower()
            for kw in ('cyclo', 'benz', 'pyrid', 'pyrrol', 'furan',
                       'thiophen', 'imidazol', 'naphthal', 'indol',
                       'quinol', 'pyrimid', 'pyrazin', 'oxazol',
                       'thiazol', 'triazol', 'morpholin', 'piperidin',
                       'pyrrolidin', 'aziridin', 'oxiran', 'thiiran',
                       'oxetan', 'azetidin', 'thietan')
        )
        if is_ring_parent:
            from ...rules.locant_validation import detect_locant_collisions

            suffix_locants_list = list(suffix_frag.locants)
            prefix_locant_groups = [
                list(p.locants) for p in prefixes if p.locants
            ]
            parent_size = _estimate_parent_size_from_name(parent_text)

            if suffix_locants_list and prefix_locant_groups:
                collisions = detect_locant_collisions(
                    suffix_locants_list,
                    prefix_locant_groups,
                    parent_type="ring",
                    parent_size=parent_size,
                )
                if collisions:
                    import logging
                    _log = logging.getLogger(__name__)
                    collision_set = set(loc for _, loc in collisions)
                    adjusted = []
                    for pf in prefixes:
                        if pf.locants:
                            new_locants = tuple(
                                l for l in pf.locants if l not in collision_set
                            )
                            if new_locants != pf.locants:
                                _log.debug(
                                    "Collision resolved: removed prefix locant(s) %s "
                                    "for '%s' (suffix has priority per IUPAC P-14.7)",
                                    set(pf.locants) - set(new_locants),
                                    pf.text,
                                )
                                pf = NameFragment(
                                    text=pf.text,
                                    locants=new_locants,
                                    fragment_type=pf.fragment_type,
                                    count=len(new_locants) if new_locants else pf.count,
                                )
                        adjusted.append(pf)
                    prefixes = adjusted

    # Alkyl prefixes are already sorted by _generate_alkyl_prefixes.
    # For non-alkyl prefixes added later, sort all together.
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    # Build prefix strings with locants
    # IUPAC rule: hyphens separate locants from names, and are needed
    # between prefixes when one ends with a letter and the next starts with a digit
    # NOTE: Some prefixes already have locants baked in (ring substituent prefixes
    # like "4-phenyl"). Only add locants to those that don't already have them.
    import re
    # `l3_omit_prefix_locant` is the P-14.3.4.3 (BB:2939) licence decided by the
    # caller (`_l3_prefix_locant_omitted`). It only ever holds when there is exactly
    # ONE prefix carrying exactly ONE locant, so suppressing the locant here cannot
    # reach a second prefix: `chloropropanedioic acid`, `chlorobutanedioic acid`.
    prefix_texts = []
    for f in prefixes:
        text = f.text
        already_has_locant = bool(re.match(r'^\d', text))
        if f.locants and not already_has_locant and not l3_omit_prefix_locant:
            loc_str = ",".join(str(l) for l in f.locants)
            prefix_texts.append(f"{loc_str}-{text}")
        else:
            prefix_texts.append(text)

    # P-16.5.1.3.1 (BlueBookV2 7272, verbatim): "For mononuclear parent hydrides
    # with two or more substituents the FIRST cited substituent never has
    # enclosing marks unless it includes a locant. The SECOND AND FURTHER
    # substituents are EACH enclosed with parentheses even for simple
    # substituents. When the simple substituent groups are accompanied by
    # multiplicative prefixes such as 'di' and 'tri', the multiplicative prefixes
    # are not included in the parentheses."
    #
    # Worked PINs: bromo(chloro)(fluoro)methane, bromo(chloro)(fluoro)(iodo)-
    # methane, chloro(methyl)silane, butyl(ethyl)(methyl)(propyl)silane,
    # cyclopropyl(phenyl)methanol [SUFFIXED], methyl(phenyl)phosphinic acid
    # [SUFFIXED]. The rule governs ANY mononuclear parent (C, Si, P, B, Ge, Sn,
    # ...) REGARDLESS of whether a characteristic-group suffix is present — hence
    # `is_mononuclear_parent` is detected STRUCTURALLY by the caller (parent atom
    # count == 1), never by string-matching the stem or gating on suffix absence.
    #
    # Conservative scope (documented carry-forward): when ANY simple prefix is
    # multiplied (dichloro/trifluoro/...), ALL prefixes are left bare/unenclosed
    # — the common-PIN form `bromodichlorofluoromethane`. The `di(chloro)`
    # refinement (enclose the simple stem while excluding the multiplier from the
    # marks) is explicitly deferred; the PROTECT gold row `C(Br)(Cl)(Cl)F` ->
    # `bromodichlorofluoromethane` enforces this branch.
    # v29: this used to be a SECOND inline copy of the rule (and so a second copy
    # of the compound-hyphen test), differing from composition_primitives.
    # apply_mononuclear_enclosing only in two comment words. Both were live —
    # this legacy assembler and the lifted serializer — so the P-16.3.4
    # italicized-prefix carve-out had to be applied twice or not at all. It is now
    # applied ONCE, there.
    from ..composition_primitives import apply_mononuclear_enclosing
    prefix_texts = apply_mononuclear_enclosing(prefix_texts, is_mononuclear_parent)

    prefix_str = _join_prefixes(prefix_texts)

    # Extract parent info: stem is in text, bond locants in locants
    stem = parent_frag.text if parent_frag else ""
    double_locants = []
    triple_locants = []
    if parent_frag and parent_frag.locants:
        # locants is a tuple of (double_bond_locants, triple_bond_locants)
        double_locants = list(parent_frag.locants[0]) if parent_frag.locants[0] else []
        triple_locants = list(parent_frag.locants[1]) if len(parent_frag.locants) > 1 and parent_frag.locants[1] else []

    # P-65.1.1.1 + P-14.3.4.6: a SUBSTITUTED 2-carbon monocarboxylic acid uses
    # the RETAINED functional parent 'acetic acid' (itself a PIN, P-12.3), and
    # because its ONLY substitutable position is the alpha-carbon, ALL
    # substituent locants are OMITTED — 'chloroacetic acid', 'difluoroacetic
    # acid' (BB 3037: "not 2,2-difluoroacetic acid"), 'phenylacetic acid' (BB
    # 6694), 'cyanoacetic acid' (BB 30999), '(4-chlorophenoxy)acetic acid'.
    # Substitution is limited to prefixes of LOWER seniority than the acid
    # (P-15.1.8.2.1.1) — satisfied by construction here: the acid is the PCG
    # (suffix) and everything else is a prefix; a second acid would form a longer
    # diacid parent and never reach this 2-carbon monoacid branch. Bare CH3-COOH
    # is caught upstream by the exact-SMILES retained-name lookup. The systematic
    # 'ethanoic acid' form is kept (fall-through) only when a substituent carries
    # a stereo locant, so the omitted-locant contraction cannot corrupt it.
    _is_substituted_acetic = (
        stem == "eth" and prefix_str
        and suffix_frag and suffix_frag.text == "oic acid"
        and not double_locants and not triple_locants
        and max(len(list(suffix_frag.locants) if suffix_frag.locants else []),
                getattr(suffix_frag, 'count', 1)) == 1
        and not any(ch.isdigit() for ch in stereo)
    )
    if _is_substituted_acetic:
        from ..composition_primitives import retained_acetic_from_prefixes
        return retained_acetic_from_prefixes(prefix_texts, stereo)

    # Handle suffix attachment using PIN-style formatting
    if suffix_frag and suffix_frag.text:
        suffix_text = suffix_frag.text
        suffix_locants = list(suffix_frag.locants) if suffix_frag.locants else []

        # P-14.3.4.4 / P-16.5.1.3.1: on a MONONUCLEAR parent there is only one
        # skeletal position, so a single-group suffix locant (always "1") is
        # meaningless and is omitted — `methanol`, not `methan-1-ol`
        # (cf. cyclopropyl(phenyl)methanol). This applies to ANY mononuclear
        # parent element and is keyed off the structural `is_mononuclear_parent`,
        # not the stem string. Multi-group suffixes keep their locants.
        if is_mononuclear_parent and len(suffix_locants) == 1:
            suffix_locants = []

        # Determine multiplier for multiple functional groups
        # Use suffix_frag.count (set by _generate_suffix) which includes terminal
        # groups like diacids where locants are omitted but multiplier is needed
        count = max(len(suffix_locants), getattr(suffix_frag, 'count', 1))
        multiplier = get_suffix_multiplier_prefix(count, suffix_text) if count > 1 else ""

        # Build unsaturation infix with locants for compounds with functional groups
        unsaturation_infix = _build_unsaturation_infix(double_locants, triple_locants)

        name = format_suffix_with_locants(
            stem,
            unsaturation_infix,
            suffix_text,
            suffix_locants,
            multiplier
        )
    else:
        # No suffix = hydrocarbon, build name with unsaturation.
        # WSD-06 (NUM-01): a substituted cycloalkene must keep its ring ene-locant
        # (`3-bromocyclohex-1-ene`), so the bond-locant is omittable ONLY when the
        # ring is unsubstituted (no substituent prefixes). Acyclic / non-cyclo stems
        # pass None to preserve the existing count-proxy behavior.
        _ring_bond_omittable = (not prefix_str) if stem.startswith("cyclo") else None
        # Wave2 T6a (P-14.3.4.2(d)): an UNSUBSTITUTED propene/propyne omits
        # the bond locant; any prefix keeps it (3-chloroprop-1-ene).
        _chain_bond_omittable = (not prefix_str) if stem == "prop" else None
        name = _build_hydrocarbon_name(
            stem, double_locants, triple_locants,
            ring_bond_locant_omittable=_ring_bond_omittable,
            chain_bond_locant_omittable=_chain_bond_omittable,
        )

    # Add prefixes with proper hyphenation at boundary
    if prefix_str:
        name = _join_prefix_to_name(prefix_str, name)

    # Add stereodescriptors at the very start
    if stereo:
        name = f"{stereo}{name}"

    return name


__all__ = [
    "name_iso_x_cyanate",
    "name_r_group",
    "cached_is_complex_ring_system",
    "_generate_chain_parent",
    "_generate_ring_parent",
    "_generate_suffix",
    "_generate_prefixes",
    "_generate_stereodescriptors",
    "_assemble_fragments",
]
