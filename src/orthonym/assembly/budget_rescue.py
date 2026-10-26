"""The degraded namings of a part whose analysis ran its hang budgets out.

A part of a name that is a compound of its own (a component of a adduct, an ion of a
 salt) is named with the hang budgets of one compound (``fragment_naming.own_hang_budgets``).
When the nested ``name`` of such a part runs them out, ``PerfBudgetExceeded`` (a ``BaseException``)
unwinds to the frame that owns the budgets of the whole assembly, which abstains the whole drawing
although every other part was named. A compound named alone meets the same trip at its outermost
``name``, which offers a bounded rescue before it abstains. ``name_after_budget_trip`` is the one
rule for the part: the same offer, then one run under the PIN ceilings of the von Baeyer analyses;
what neither names is left to the caller, who re-raises the trip as before.
"""
import contextlib
from typing import Optional


@contextlib.contextmanager
def outermost_tier_contexts():
    """The tier context variables as they stand where a compound named alone runs its budget
    rescue: the outermost ``name`` has already unwound the ones it published, so they are at
    their defaults. Inside the naming of an assembly they are set to its tier, which the T4
    producer reads as the best-effort substituent vocabulary: its von Baeyer analyses then run
    the fresh budget of the rescue out for parts it names within it when named alone
    (``Orthonym._try_besteffort_clean_general_fallthrough`` resets the same four for the same
    reason). Restored on exit."""
    from ..metrics.provenance import (
        allow_aromatic_general_ctx,
        best_effort_ctx,
        full_coverage_ctx,
        general_fallback_ctx,
    )
    tokens = [(var, var.set(False)) for var in (
        general_fallback_ctx, best_effort_ctx, allow_aromatic_general_ctx, full_coverage_ctx)]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except Exception:  # noqa: BLE001
                pass


def t4_rescue_name(namer, smiles: str) -> Optional[str]:
    """The rescue name of one part whose main-path analysis ran its hang budgets out, or None.

    ``Orthonym._try_perf_budget_t4_rescue``: a fresh, bounded budget around one whole-molecule T4
    naming that ships only when OPSIN reads the name back to the part (best-effort tier only; None
    at every other tier, so the caller's trip unwinds as it always did), run in the context a compound
    named alone has at its outermost ``name`` (``outermost_tier_contexts``). The fragment memo, the
    naming session and the provenance are the ones of the enclosing naming. A name is a
    general-engine name, never a PIN (as for the other spellings of ``rules.adducts``), so the name
    that carries it is labelled below the PIN."""
    try:
        with outermost_tier_contexts():
            name = namer._try_perf_budget_t4_rescue(smiles)
    except Exception:  # noqa: BLE001 - an offer, never a failure of the assembly
        return None
    if name:
        from ..metrics.provenance import record_non_pin_fragment, record_source
        record_source("general_engine")
        record_non_pin_fragment(name)
    return name


def pin_ceiling_name(namer, smiles: str) -> Optional[str]:
    """The name of one part whose main-path analysis ran its hang budgets out and whose rescue
    found no name, built once more with the PIN-tier ceilings of the von Baeyer analyses
    (``rules.vonbaeyer_universal.pin_ceilings_scope``), or None.

    The best-effort ceilings (100 skeletal atoms, 15 rings) let the analysis of a large spiro-fused
    ring system run the budgets of one compound out; under the PIN ceilings such a system is refused
    at once and another producer names the part, as before they existed. The run has budgets of its
    own (``rearm_hang_budgets``: the rescue before it left them disarmed), a fresh memo and a depth-0
    session (``isolated_naming_session``), so it neither reads a name the exhausted run left nor
    leaves one. Best-effort tier only; the name is the one the part's own namer ships, so its labels
    are its own. Never raises."""
    if not namer._general_fallback_unverified:
        return None
    from ..rules.vonbaeyer_universal import pin_ceilings_scope
    from .fragment_naming import (
        PerfBudgetExceeded,
        isolated_naming_session,
        name_scope_depth,
        own_hang_budgets,
        rearm_hang_budgets,
    )
    try:
        with (pin_ceilings_scope(), own_hang_budgets(),
              isolated_naming_session(reset_cache=True)):
            if name_scope_depth() > 0:
                rearm_hang_budgets()
            return namer.name(smiles)
    except (Exception, PerfBudgetExceeded):
        return None


def name_after_budget_trip(namer, smiles: str) -> Optional[str]:
    """The name of one part (``smiles``) whose nested ``name`` by ``namer`` ran the hang budgets of
    one compound out: the rescue a compound named alone is offered (``t4_rescue_name``), else a
    run under the PIN ceilings (``pin_ceiling_name``), else None -- the caller then lets the trip
    unwind to the frame that owns the budgets, as it did before the part had this offer.

    The offers run with hang budgets of their own (``own_hang_budgets``), so the budgets of the
    assembly that enclosed the part are back as they were when the offers leave: the rescue
    disarms the budgets it ran under, and a caller that handles the trip outside the part's own
    ``own_hang_budgets`` (``rules.salts``) would otherwise leave the whole molecule unbounded."""
    from .fragment_naming import own_hang_budgets
    with own_hang_budgets():
        return t4_rescue_name(namer, smiles) or pin_ceiling_name(namer, smiles)


def recover_beyond_ring_ceiling(namer, smiles: str, origin: dict) -> Optional[str]:
    """The late engine recovery of ``namer`` (``_try_general_engine_recovery_impl``), run once more
    with the lifted ceilings of the von Baeyer analyses
    (``rules.vonbaeyer_universal.lifted_ring_ceiling_scope``) when it names nothing and the
    molecule holds a ring system that the ceilings of 100 atoms and 15 rings refuse
    (``ring_system_beyond_ceiling``).

    A fused aromatic macrocycle of 16 to 22 rings, or of 102 atoms, is analysed in a few seconds
    once the ceilings are lifted, but at the first attempt a larger ring system costs the budget that
    its parts need (``rules.vonbaeyer_universal``, a performance pass and 3), so the lift is an offer made after
    the ordinary attempt found no name: a name that exists is never replaced, and a molecule with no
    such ring system costs nothing. The offer has hang budgets of its own and is made once for the
    molecule that ``name`` was called with (``first_whole_molecule_offer``), not for each of the
    recoveries that run one after the other. The name is the one the recovery ships, its round trip
    decides. Best-effort tier only. ``origin`` is the recovery's own record of the name it ships."""
    rec = namer._try_general_engine_recovery_impl(smiles, origin)
    if rec is not None or not namer._general_fallback_unverified:
        return rec
    from rdkit import Chem

    from ..rules.vonbaeyer_universal import lifted_ring_ceiling_scope, ring_system_beyond_ceiling
    from .fragment_naming import PerfBudgetExceeded, first_whole_molecule_offer, own_hang_budgets
    try:
        mol = Chem.MolFromSmiles(smiles)
        if (mol is None or not ring_system_beyond_ceiling(mol)
                or not first_whole_molecule_offer("lifted_ring_ceiling", smiles)):
            return None
        # hang budgets of its own: the enclosing ones are back as they were when the offer leaves
        with lifted_ring_ceiling_scope(), own_hang_budgets():
            return namer._try_general_engine_recovery_impl(smiles, origin)
    except (Exception, PerfBudgetExceeded):  # an offer, never a failure of the molecule
        return None
