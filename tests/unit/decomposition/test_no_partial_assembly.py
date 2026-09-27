""" a phase (name-all compositional) -- decomposition must never ship a
partial assembly.

Spec: internal notes. The a trace found
`_try_multi_bond_decompose` and `_try_iterative_mixed_decompose` cut a large
molecule at linkages, name each fragment, and ship a name built from whichever
>= 2 fragments named -- SILENTLY DROPPING any fragment that failed. The
shipped name denotes a SMALLER molecule than the input, which (rightly)
suppresses, so the whole molecule abstains -- but a producer that dishonestly
offers a smaller-molecule candidate is not the same thing as an honest
abstention (a project rule: "0-wrong is delivered by E1 + rejecting a
candidate, never by a producer declining to build one" -- but PIN-tier fail-
closed is licensed by the SAME invariant's own carve-out, and this defect
sits squarely on the PIN-shared decomposition path).

The fix (`decomposition/engine.py`):
  1. `_name_fragment_with_fallback` gained a third rung, `_name_fragment_t4_rescue`,
     which retries a fragment that failed BOTH PIN-tier attempts through the
     best-effort (T4/general-engine) recovery lane
     (`Orthonym._try_general_engine_recovery`) inside an isolated naming
     session. This can only ever turn a fragment FAILURE into a fragment
     SUCCESS -- a fragment that already names at PIN tier never reaches it, so
     every currently-correct decomposition name is byte-identical.
  2. Both multi-fragment assemblers now FAIL CLOSED: if any input fragment
     still could not be named/covered adequately (even after the rescue
     rung), the whole assembly is declined (`return None`) instead of shipping
     a name built from the successfully-named fragments alone.

This file is the load-bearing regression test for that invariant. It is
deliberately built on REAL molecules (no fragment-naming mocks) so the
assertions exercise the genuine rescue + fail-closed interaction, not a
simulated one.
"""

import logging

import pytest
from rdkit import Chem

from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
from orthonym.decomposition.engine import (
    _name_fragment_t4_rescue,
    _name_fragment_with_fallback,
    _try_iterative_mixed_decompose,
)
from orthonym.assembly.fragment_naming import (
    end_naming_session,
    start_naming_session,
)
from orthonym.errors import is_failure_name
from orthonym.namer import Orthonym, name_compound
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.jars import jar_or_none

pytestmark = [
    pytest.mark.skipif(
        jar_or_none() is None,
        reason="round-trip assertions need the OPSIN jar; without it the "
               "invariant cannot be measured (it would be vacuously true)",
    ),
    # This file asserts PRODUCTION's invariant (name_compound with the
    # OPSIN validity gate ON). The suite's autouse fixture disables that gate
    # by default (see tests/conftest.py's `_opsin_validity_gate_state`), which
    # is a DIFFERENT, deliberately more permissive configuration used to probe
    # the raw generator -- under gate-OFF a malformed/partial candidate can
    # slip through unchecked, which is exactly the failure mode this file
    # exists to rule out. `opsin_gate` re-enables it for every test here.
    pytest.mark.opsin_gate,
]


# ---------------------------------------------------------------------------
# Traced molecules (internal notes +
# live corpus rows this session pulled from
# internal notes, terminal_code
# GATE_SUPPRESSED / self01_mismatch).
# ---------------------------------------------------------------------------

# GPI-type phosphoethanolamine-trimannoside with a thioalkyl-decorated GlcN
# unit. VERIFIED (this session, debug-logged trace): decomposes via
# `_try_iterative_mixed_decompose`; before this fix, one fragment (the
# GlcN-thioether unit) failed PIN-tier naming and the assembler shipped
# "3/4 fragments named (1 failed)" -- a name for a SMALLER molecule, which
# correctly suppressed (byte-matching the corpus's recorded
# self01_suppressed_name field for this exact row).
GPI_MANNOSIDE = (
    "NCCOP(=O)(O)OC[C@H]1O[C@H](O[C@@H]2[C@@H](OC[C@H]3O[C@H](O[C@H]4[C@H](O)"
    "[C@@H](N)[C@H](OCCCCCCS)O[C@@H]4CO)[C@@H](O)[C@@H](O)[C@@H]3O)O[C@H](CO)"
    "[C@@H](O)[C@@H]2O)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)"
    "[C@@H](O)[C@@H]1O"
)

# The isolated, previously-unnameable GlcN-thioether fragment cut from the
# GPI mannoside above (H-capped at both attachment points). VERIFIED
# (this session): `name_pipeline_only` and `name_fragment_recursively` both
# fail on it (PIN tier); `_name_fragment_t4_rescue` succeeds.
GPI_FAILING_FRAGMENT = (
    "N[C@H]1[C@H](OCCCCCCS)O[C@H](CO)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)"
    "[C@H](O)[C@@H]2O)[C@@H]1O"
)
# 2026-09-25: GPI_FAILING_FRAGMENT's name above is STALE -- both PIN rungs now
# name it (see test_fragment_fails_pin_tier_alone). This one still fails both:
# the PIN tier has no producer for a DECORATED SATURATED ring-yl prefix (here
# 2,2,5,5-tetramethyl-1,3-dioxolan-4-yl; `rules.ring_substituents.
# _decorated_heteroaryl_substituent_name` is aromatic-monocycle-only), while
# the rung names it '(3R)-3,7-dimethyl-9-(2,2,5,5-tetramethyl-1,3-dioxolan-
# 4-yl)nona-1,6-dien-3-ol', RT-exact (measured 2026-09-25). It is the canary
# call 171 molecule of TRIAGE.md " known cases".
# Suite fix j6 (TRIAGE g3 C17b): that producer now exists for saturated
# chalcogen heteromonocycles (Hantzsch-Widman stem,, so both PIN
# rungs name the dioxolane row RT-exact and it no longer anchors anything. The
# anchor is now the PAH_10 hexahydronaphthalene: its PIN (a hydro-naphthalene
# fusion name, has no producer, both PIN rungs decline, and the T4
# rung names it '2,6-dimethyl-8-(prop-1-en-2-yl)bicyclo[4.4.0]deca-1(10),2-
# diene', RT-exact (measured 2026-09-27).
PIN_TIER_FAILING_FRAGMENT = "C=C(C)C1CC=C2C(C)=CCCC2(C)C1"

# 74-heavy-atom lipopeptide (fatty-acyl N-cap + 6 amide-linked residues,
# non-standard/branched residues). VERIFIED (this session, monkeypatch trace):
# `_try_multi_bond_decompose` (amide, 7 bonds) fires for this molecule --
# confirms the a trace's ASSUMED call-site attribution for the peptide/lipid
# family. In THIS instance all fragments name successfully (failed_count==0
# even before this fix); the suppression here traces to a SEPARATE,
# out-of-scope defect in `_assemble_multi_amide`'s "N,N-di..." grouping, not
# a dropped fragment. It is included as a genuinely-decomposed, real-world
# molecule the no-partial-ship invariant must still hold for.
LIPOPEPTIDE = (
    "CCC(C)CCCCCCCC(O)CC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](CC(=O)N[C@H]"
    "(CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CC(=O)O)C(=O)N[C@H](CC(C)C)C(=O)"
    "N[C@@H](CC(C)C)C(=O)O)C(C)C)CC(C)C"
)

# Small ACC-dipeptide-like amide (cyclopropane amino acid N-acylated by a
# glutamate-like acid). Real GATE_SUPPRESSED/self01_mismatch corpus row.
SMALL_AMIDE = "NC(CCC(=O)NC1(C(=O)O)CC1)C(=O)O"

# Ordinary triester (glycerol triacetate) that must remain a full,
# non-abstaining, round-tripping name -- a plain sanity control proving the
# fail-closed guard does not newly break a case that was always fine (every
# fragment names cleanly at PIN tier, so `_name_fragment_t4_rescue` never
# even fires for it).
TRIACETIN = "CC(=O)OCC(OC(C)=O)COC(C)=O"


def _full_rt(smiles: str, name: str) -> bool:
    """True iff OPSIN parses `name` to something with the SAME full
    (isomeric, charge-aware) InChIKey as `smiles` -- a strict, whole-molecule
    round-trip, not a substructure/coverage check.
    """
    opsin_smiles = opsin_parse(name)
    if not opsin_smiles:
        return False
    m_in = Chem.MolFromSmiles(smiles)
    m_out = Chem.MolFromSmiles(opsin_smiles)
    if m_in is None or m_out is None:
        return False
    return Chem.MolToInchiKey(m_in) == Chem.MolToInchiKey(m_out)


def _is_strict_subset_name(smiles: str, name: str) -> bool:
    """True iff `name` parses (via OPSIN) to a molecule whose heavy-atom
    count is STRICTLY SMALLER than the input's -- the exact "shipped a name
    for a smaller molecule" shape this phase eliminates. Only meaningful
    when `name` is not itself a failure/abstention sentinel.
    """
    opsin_smiles = opsin_parse(name)
    if not opsin_smiles:
        return False
    m_in = Chem.MolFromSmiles(smiles)
    m_out = Chem.MolFromSmiles(opsin_smiles)
    if m_in is None or m_out is None:
        return False
    return m_out.GetNumHeavyAtoms() < m_in.GetNumHeavyAtoms()


class TestNoPartialShip:
    """Core invariant: `name_compound` NEVER emits a name that round-trips to
    a STRICT SUBSET of the input's atoms. It either round-trips FULLY or it
    abstains (`is_failure_name`) -- never a partial in between.
    """

    @pytest.mark.parametrize(
        "smiles",
        [GPI_MANNOSIDE, LIPOPEPTIDE, SMALL_AMIDE, TRIACETIN],
        ids=["gpi_mannoside", "lipopeptide", "small_amide", "triacetin"],
    )
    def test_never_ships_a_smaller_molecule_name(self, smiles):
        result = name_compound(smiles)
        assert result, f"name_compound returned falsy for {smiles}"

        if is_failure_name(result):
            # Honest abstention -- always acceptable.
            return

        # A non-abstaining emission must NEVER denote a strict subset of the
        # input's atoms (the partial-ship defect this phase fixes).
        assert not _is_strict_subset_name(smiles, result), (
            f"REGRESSION: name_compound shipped a name for a SMALLER "
            f"molecule than the input -- exactly the partial-assembly "
            f"defect v32 Phase 2 fixed. smiles={smiles!r} name={result!r}"
        )

    def test_gpi_mannoside_honestly_abstains_not_partial(self):
        """The GPI mannoside is the a trace's directly-traced positive: before
        this fix it shipped 'α-D-mannopyranosyloxy α-D-mannopyranosyl-
        (...' (3/4 fragments, dropping the failed GlcN-thioether unit's 27
        atoms) and suppressed it. After this fix the assembler either
        ships an ATOM-COMPLETE candidate or the molecule abstains cleanly --
        it must never again reach a partial-ship shape.
        """
        result = name_compound(GPI_MANNOSIDE)
        assert result
        assert is_failure_name(result) or not _is_strict_subset_name(
            GPI_MANNOSIDE, result
        )

    def test_lipopeptide_honestly_abstains_not_partial(self):
        """The lipopeptide is the a trace's call-site-confirmed positive for
        `_try_multi_bond_decompose` (amide). Must never ship a name for a
        subset of its 74 heavy atoms.
        """
        result = name_compound(LIPOPEPTIDE)
        assert result
        assert is_failure_name(result) or not _is_strict_subset_name(
            LIPOPEPTIDE, result
        )


class TestT4RescueMechanism:
    """Verifies the T4-rescue rung and the fail-closed guard actually fire
    and interact as designed, using REAL (non-mocked) fragment naming.
    """

    def test_fragment_fails_pin_tier_alone(self):
        """Sanity anchor: a fragment that genuinely fails BOTH PIN-tier rungs
        on its own, so the rescue rung is doing new work, not merely
        duplicating an existing success.

        Re-scoped 2026-09-25 (pre-existing-failures plan, Task 3; TRIAGE.csv
        row for this test, class D-spelling, owner T3). Its premise went STALE:
        both PIN rungs now name GPI_FAILING_FRAGMENT, and the name round-trips
        to the fragment's full InChIKey (TRIAGE got_rt=exact; re-measured
        2026-09-25). So the test now asserts (a) that measured truth for
        GPI_FAILING_FRAGMENT, and (b) the anchor's original claim on
        PIN_TIER_FAILING_FRAGMENT, which still fails both PIN rungs and which
        the rung names RT-exact. The GPI fragment's spelling
        '(6-sulfanylhexyloxy)' (an enclosing-mark defect) is Task 5's, so no
        spelling is pinned here -- only round-trip identity.
        """
        from orthonym.assembly.fragment_naming import name_fragment_recursively
        from orthonym.namer import name_pipeline_only

        # (a) the GPI fragment is now PIN-nameable on both rungs, RT-exact.
        for rung in (name_fragment_recursively, name_pipeline_only):
            got = rung(GPI_FAILING_FRAGMENT)
            assert got and not is_failure_name(got), (rung.__name__, got)
            assert _full_rt(GPI_FAILING_FRAGMENT, got), (rung.__name__, got)

        # (b) a fragment that fails both PIN rungs; the rung names it.
        assert not name_fragment_recursively(PIN_TIER_FAILING_FRAGMENT)
        pin_only = name_pipeline_only(PIN_TIER_FAILING_FRAGMENT)
        assert not pin_only or is_failure_name(pin_only), pin_only
        rescued = _name_fragment_t4_rescue(PIN_TIER_FAILING_FRAGMENT)
        assert rescued and _full_rt(PIN_TIER_FAILING_FRAGMENT, rescued), rescued

    def test_t4_rescue_names_the_previously_unnameable_fragment(self):
        """`_name_fragment_t4_rescue` succeeds where PIN tier failed. This
        is the BREADTH half of the fix: a fragment unnameable at PIN tier
        but completable by best-effort no longer inflates `failed_count`."""
        rescued = _name_fragment_t4_rescue(GPI_FAILING_FRAGMENT)
        assert rescued, "T4 rescue should name the GlcN-thioether fragment"
        assert "unknown" not in rescued.lower()

    def test_with_fallback_reaches_t4_rung_only_after_pin_fails(self):
        """`_name_fragment_with_fallback` must return the SAME rescued name
        for the failing fragment (proves the three rungs are wired
        correctly), and must be BYTE-IDENTICAL to the plain PIN name for a
        fragment that already succeeds at PIN tier (scoping invariant: T4
        never competes with or replaces a PIN-successful fragment name).
        """
        # Failing-at-PIN fragment: rescue rung fires and returns a name.
        result = _name_fragment_with_fallback(GPI_FAILING_FRAGMENT)
        assert result and "unknown" not in result.lower()

        # PIN-successful fragment: identical to the plain PIN pipeline
        # result -- the rung must not fire/alter it.
        from orthonym.namer import name_pipeline_only

        pin_fragment = "CC(=O)O"  # acetic acid -- trivially PIN-nameable
        assert (
            _name_fragment_with_fallback(pin_fragment)
            == name_pipeline_only(pin_fragment)
        )

    def test_iterative_mixed_decompose_achieves_atom_complete_assembly(self):
        """Direct call (real molecule, real fragment naming, no mocks):
        `_try_iterative_mixed_decompose` on the GPI mannoside must return
        either ONE name of the WHOLE input -- OPSIN full-InChIKey exact -- or
        None. It must never return a partial (3/4 fragments, the a phase
        defect) and never a glue of fragment names.

        Re-scoped 2026-09-25 (pre-existing-failures plan, Task 3; TRIAGE.csv
        row for this test, class D-wrong, owner T3). The old assertion wanted a
        non-None result carrying the rescued fragment's tokens, and the
        docstring said why that result was not one connected name: the flat
        weaver fell back to a "naive space-join". Measured: that string --
        '2-(phosphonooxy)ethan-1-amine (2R,3R,4R,5S,6R)-3-amino-...-oxan-4-ol
        α-D-mannopyranosyloxy α-D-mannopyranosyl-(1->2)-α-D-mannopyranose' --
        OPSIN parses to C38H75N2O31PS, the input is C38H71N2O28PS (TRIAGE
        got_rt=wrong). Blue Book: separate words in a name are the class term
        of a functional class name (GLOSSARY, "Functional class nomenclature",
        BB:1816: "the principal characteristic group is expressed as a class
        term... written as a separate word or words"), and even separate
        molecular entities are joined by em dashes with a ratio
        "Organic adducts", BB:4651). A blank-joined list of four component
        names is neither. The glue is gone (`fragment_assembly.
        _assemble_by_bond_type` and this function's two fallbacks now decline),
        so the flat assembler voids the candidate; weaving a ring-hub star into
        one name is out of `decomposition/weave.py`'s v1 scope, which says so.

        Voiding the candidate declines no molecule: the best-effort tier names
        the GPI mannoside RT-exact (asserted below; plan Global Constraints,
        "breadth never drops").
        """
        mol = Chem.MolFromSmiles(GPI_MANNOSIDE)
        bonds = find_cleavable_bonds(mol)
        start_naming_session()
        try:
            result = _try_iterative_mixed_decompose(mol, bonds, style="pin")
        finally:
            end_naming_session()

        if result is not None:
            assert _full_rt(GPI_MANNOSIDE, result), (
                f"the mixed decomposition returned a string that is not a "
                f"name of the whole input (a partial or a glue): {result!r}"
            )

        from orthonym.cli import _emit_tier_flags

        best_effort = Orthonym(
            style="pin", **_emit_tier_flags("best-effort")
        ).name_tiered(GPI_MANNOSIDE)["name"]
        assert best_effort and not is_failure_name(best_effort), best_effort
        assert _full_rt(GPI_MANNOSIDE, best_effort), best_effort

    def test_fail_closed_guard_present_in_both_assemblers(self):
        """Static confirmation the fail-closed `return None` guard (added
        this phase) exists ahead of the `named_fragments` assembly dispatch
        in BOTH multi-fragment assemblers -- a cheap, fast-running canary
        against an accidental revert, independent of the (slower) live
        molecule tests above.
        """
        import inspect

        from orthonym.decomposition import engine

        multi_src = inspect.getsource(engine._try_multi_bond_decompose)
        mixed_src = inspect.getsource(engine._try_iterative_mixed_decompose)

        for src, label in (
            (multi_src, "_try_multi_bond_decompose"),
            (mixed_src, "_try_iterative_mixed_decompose"),
        ):
            assert "failed_count > 0" in src and "return None" in src, (
                f"{label} no longer appears to fail closed on "
                f"failed_count > 0 -- has the guard been reverted?"
            )


class TestHonestAbstainWhenGenuinelyUnnameable:
    """At least one case must still abstain CLEANLY (never a partial) when a
    fragment is genuinely unnameable even with assistance.
    """

    def test_lipopeptide_abstains_cleanly(self):
        """The lipopeptide's own internal amide-fragment residues include a
        branched/non-standard fatty-acyl N-cap that neither PIN nor T4
        currently completes into the exact whole-molecule structure (traced
        this session: `name_compound` returns the failure sentinel, never a
        malformed or partial name).
        """
        result = name_compound(LIPOPEPTIDE)
        assert is_failure_name(result), (
            f"expected an honest abstention for the lipopeptide, got: "
            f"{result!r}"
        )


class TestPinSuccessUnaffected:
    """Byte-identity control: the fix must not perturb an ordinary molecule
    that was already correctly decomposed at PIN tier (rescue never even
    fires because every fragment already succeeds at PIN tier)."""

    def test_triacetin_still_names_and_round_trips(self):
        result = name_compound(TRIACETIN)
        assert result and not is_failure_name(result)
        assert _full_rt(TRIACETIN, result), (
            f"triacetin must still fully round-trip: {result!r}"
        )
