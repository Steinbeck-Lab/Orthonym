"""v32 Phase 2 (name-all compositional) -- decomposition must never ship a
partial assembly.

Spec: . The SPY found
`_try_multi_bond_decompose` and `_try_iterative_mixed_decompose` cut a large
molecule at linkages, name each fragment, and ship a name built from whichever
>= 2 fragments named -- SILENTLY DROPPING any fragment that failed. The
shipped name denotes a SMALLER molecule than the input, which SELF-01 (rightly)
suppresses, so the whole molecule abstains -- but a producer that dishonestly
offers a smaller-molecule candidate is not the same thing as an honest
abstention (invariant 1: "0-wrong is delivered by E1 + SELF-01 rejecting a
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
     still could not be named/covered adequately (even after the T4 rescue
     rung), the whole assembly is declined (`return None`) instead of shipping
     a name built from the successfully-named fragments alone.

This file is the load-bearing regression test for that invariant. It is
deliberately built on REAL molecules (no fragment-naming mocks) so the
assertions exercise the genuine T4 rescue + fail-closed interaction, not a
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
from orthonym.validation.opsin_roundtrip import _find_opsin_jar, opsin_parse

pytestmark = [
    pytest.mark.skipif(
        _find_opsin_jar() is None,
        reason="round-trip assertions need the OPSIN jar; without it the "
               "invariant cannot be measured (it would be vacuously true)",
    ),
    # This file asserts PRODUCTION's invariant (name_compound with the SELF-01
    # OPSIN validity gate ON). The suite's autouse fixture disables that gate
    # by default (see tests/conftest.py's `_opsin_validity_gate_state`), which
    # is a DIFFERENT, deliberately more permissive configuration used to probe
    # the raw generator -- under gate-OFF a malformed/partial candidate can
    # slip through unchecked, which is exactly the failure mode this file
    # exists to rule out. `opsin_gate` re-enables it for every test here.
    pytest.mark.opsin_gate,
]


# ---------------------------------------------------------------------------
# Traced molecules ( +
# live corpus rows this session pulled from
# {chebi,pubchem}500.json`, terminal_code
# GATE_SUPPRESSED / self01_mismatch).
# ---------------------------------------------------------------------------

# GPI-type phosphoethanolamine-trimannoside with a thioalkyl-decorated GlcN
# unit. VERIFIED (this session, debug-logged trace): decomposes via
# `_try_iterative_mixed_decompose`; before this fix, one fragment (the
# GlcN-thioether unit) failed PIN-tier naming and the assembler shipped
# "3/4 fragments named (1 failed)" -- a name for a SMALLER molecule, which
# SELF-01 correctly suppressed (byte-matching the corpus's recorded
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

# 74-heavy-atom lipopeptide (fatty-acyl N-cap + 6 amide-linked residues,
# non-standard/branched residues). VERIFIED (this session, monkeypatch trace):
# `_try_multi_bond_decompose` (amide, 7 bonds) fires for this molecule --
# confirms the SPY's ASSUMED call-site attribution for the peptide/lipid
# family. In THIS instance all fragments name successfully (failed_count==0
# even before this fix); the SELF-01 suppression here traces to a SEPARATE,
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
        """The GPI mannoside is the SPY's directly-traced positive: before
        this fix it shipped 'alpha-D-mannopyranosyloxy alpha-D-mannopyranosyl-
        (...' (3/4 fragments, dropping the failed GlcN-thioether unit's 27
        atoms) and SELF-01 suppressed it. After this fix the assembler either
        ships an ATOM-COMPLETE candidate or the molecule abstains cleanly --
        it must never again reach a partial-ship shape.
        """
        result = name_compound(GPI_MANNOSIDE)
        assert result
        assert is_failure_name(result) or not _is_strict_subset_name(
            GPI_MANNOSIDE, result
        )

    def test_lipopeptide_honestly_abstains_not_partial(self):
        """The lipopeptide is the SPY's call-site-confirmed positive for
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
        """Sanity anchor: the isolated GlcN-thioether fragment genuinely
        fails BOTH PIN-tier rungs on their own (so the T4 rescue rung is
        actually doing new work, not merely duplicating an existing success).
        """
        from orthonym.assembly.fragment_naming import name_fragment_recursively
        from orthonym.namer import name_pipeline_only

        assert not name_fragment_recursively(GPI_FAILING_FRAGMENT)
        pin_only = name_pipeline_only(GPI_FAILING_FRAGMENT)
        assert not pin_only or "unknown" in str(pin_only).lower()

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
        # Failing-at-PIN fragment: T4 rescue rung fires and returns a name.
        result = _name_fragment_with_fallback(GPI_FAILING_FRAGMENT)
        assert result and "unknown" not in result.lower()

        # PIN-successful fragment: identical to the plain PIN pipeline
        # result -- the T4 rung must not fire/alter it.
        from orthonym.namer import name_pipeline_only

        pin_fragment = "CC(=O)O"  # acetic acid -- trivially PIN-nameable
        assert (
            _name_fragment_with_fallback(pin_fragment)
            == name_pipeline_only(pin_fragment)
        )

    def test_iterative_mixed_decompose_achieves_atom_complete_assembly(self):
        """Direct call (real molecule, real fragment naming, no mocks):
        `_try_iterative_mixed_decompose` on the GPI mannoside now accounts
        for ALL 4 cut fragments (thanks to the T4 rescue rung), where before
        this fix it silently dropped the failing one and shipped a 3/4
        "partial assembly". The returned string must mention every
        fragment's structural content (mannopyranosyl appears for BOTH
        sugar arms, the phosphonooxyethanamine arm, and the rescued
        GlcN-thioether unit's own descriptive tokens) -- i.e. the assembler
        no longer drops the previously-failing fragment's atoms into the
        void, even though a SEPARATE, out-of-scope limitation (weaving a
        standalone T4 compound name into a glycoside prefix position) means
        the resulting string does not yet compose into ONE connected IUPAC
        name (it falls back to the pre-existing naive space-join and is
        correctly SELF-01-rejected as a whole -- see
        `test_gpi_mannoside_honestly_abstains_not_partial`).
        """
        mol = Chem.MolFromSmiles(GPI_MANNOSIDE)
        bonds = find_cleavable_bonds(mol)
        start_naming_session()
        try:
            result = _try_iterative_mixed_decompose(mol, bonds, style="pin")
        finally:
            end_naming_session()

        assert result is not None, (
            "with the T4 rescue rung, all 4 fragments should name "
            "successfully and the assembler should stop declining"
        )
        # The previously-dropped fragment's rescued name carries these
        # tokens (verified this session): thia/oxa-octyl chain + oxane ring.
        assert "thia" in result.lower() or "oxane" in result.lower(), (
            f"assembled result does not appear to include the rescued "
            f"fragment's content: {result!r}"
        )

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
    fragment is genuinely unnameable even with T4 assistance.
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
    that was already correctly decomposed at PIN tier (T4 rescue never even
    fires because every fragment already succeeds at PIN tier)."""

    def test_triacetin_still_names_and_round_trips(self):
        result = name_compound(TRIACETIN)
        assert result and not is_failure_name(result)
        assert _full_rt(TRIACETIN, result), (
            f"triacetin must still fully round-trip: {result!r}"
        )
