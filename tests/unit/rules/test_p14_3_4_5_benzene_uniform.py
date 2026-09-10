"""P-14.3.4.5 -- uniform complete substitution on benzene (v29 Phase C tranche B, Task 1).

Governing rule chain, verbatim from ``BlueBookV2/BlueBookV2.md``:

``P-14.3.3`` "Citation of locants" (``:2869``) is **DENY BY DEFAULT** --

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure as defined by its appropriate
     enclosing marks, then all locants must be cited for the parent structure or that
     structural unit."

``P-14.3.4.5`` (``:3007``) grants the licence exercised here --

    "All locants are omitted in compounds or substituent groups in which all
     substitutable positions are completely substituted or modified, for example, by
     hydro, in the same way."

-- and its own counter-clause (``:3009``) is the tripwire --

    "In case of partial substitution or modification, all numerical prefixes must be
     indicated."

Verbatim ``(PIN)`` witnesses: ``:7625`` ``benzenehexol (PIN, P-63.1.2) (not
benzenehexaol)``; the free-valence analogue ``:3021`` ``benzenehexayl``. The three
prefix targets (``hexamethyl-`` / ``hexafluoro-`` / ``hexachlorobenzene``) are
**DERIVED** from ``:3007``, not verbatim rows -- derivation F7.

★ THE BOUNDARY, ``:54823``: "Inositols, cyclohexane-1,2,3,4,5,6-hexols, are a specific
group of cyclitols." The saturated analogue RETAINS every locant, because a cyclohexane
ring carbon has TWO substitutable H and six OH is therefore only PARTIAL substitution.
If ``OC1C(O)C(O)C(O)C(O)C1O`` ever loses its locants, the predicate is counting
positions instead of hydrogens.

MEASURED CODE PATH (validated call-spy, 2026-07-28 -- see task-pcB1-report.md; line
numbers re-verified 2026-07-28 after drift, function names are the durable anchor):
  * the suffix target is joined at ``rules/benzene.py:3237``
    (``return f"benzene{multiplied}"``) inside ``_assemble_benzene_with_suffix`` --
    CONFIRMED, 8 line hits for benzenehexol against 4 for the known positive
    ``benzene-1,2-diol``;
  * the three prefix targets are joined INLINE in ``name_substituted_benzene``
    (``:3074`` ``format_substituent_prefix`` -> ``:3103`` ``_join_benzene_prefixes``),
    **not** in ``_build_prefix_string_with_locants`` as the task brief stated -- that
    function and its sibling ``_build_prefix_string`` record ZERO hits for
    ``hexamethylbenzene`` AND zero for the prefix known positive
    ``1,4-dibromobenzene``.

Invariant 11: removing a locant can unmask something worse (in v29 a fail-closed prefix
turned a fabrication into a silent atom drop, four separate times). Every guard below
asserts the FULL emitted name, never merely that a locant vanished.
"""
import re

import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _name(namer, smiles):
    return namer.name(smiles)


# --------------------------------------------------------------------------- #
# 1. The licence fires -- uniform complete substitution of the benzene ring    #
# --------------------------------------------------------------------------- #
class TestLicensedOmission:
    @pytest.mark.parametrize("smiles,expected,authority", [
        ("Oc1c(O)c(O)c(O)c(O)c1O", "benzenehexol", "verbatim BB:7625"),
        ("Cc1c(C)c(C)c(C)c(C)c1C", "hexamethylbenzene", "derived from BB:3007 (F7)"),
        ("Fc1c(F)c(F)c(F)c(F)c1F", "hexafluorobenzene", "derived from BB:3007 (F7)"),
        ("Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "hexachlorobenzene",
         "derived from BB:3007 (F7)"),
    ])
    def test_uniform_complete_benzene_omits_all_locants(
        self, namer, smiles, expected, authority
    ):
        assert _name(namer, smiles) == expected, authority


# --------------------------------------------------------------------------- #
# 2. The deny-default holds -- the mandatory tripwire set                      #
# --------------------------------------------------------------------------- #
class TestDenyByDefault:
    def test_the_boundary_cyclohexanehexol_keeps_every_locant(self, namer):
        """★ BB:54823. 12 substitutable H, 6 decorated => partial => BB:3009."""
        assert _name(namer, "OC1C(O)C(O)C(O)C(O)C1O") == \
            "cyclohexane-1,2,3,4,5,6-hexol"

    def test_partial_ring_substitution_keeps_the_pentol_locants(self, namer):
        """5 OH + 1 Cl on benzene: complete, but NOT 'in the same way' => BB:3009.

        Asserted structurally, NOT as an exact string: the current emission
        ``1-chlorobenzene-2,3,4,5,6-pentol`` is itself a non-PIN (defect N1 -- the
        principal characteristic group must get the lower locant set, so
        ``6-chlorobenzene-1,2,3,4,5-pentol`` is correct). Pinning the exact string
        would enshrine the non-PIN, which is the trap tranche A fell into.
        """
        got = _name(namer, "Oc1c(O)c(O)c(O)c(O)c1Cl")
        assert "pentol" in got, got
        assert re.search(r"\d(?:,\d){4}-pentol", got), \
            f"the five -ol locants must all be cited: {got!r}"
        assert "chloro" in got, f"the chlorine must not be dropped: {got!r}"

    def test_heterogeneous_complete_substitution_keeps_locants(self, namer):
        """All six positions substituted but not 'in the same way' (BB:3007)."""
        got = _name(namer, "Cc1c(C)c(C)c(C)c(C)c1Cl")
        assert got == "1-chloro-2,3,4,5,6-pentamethylbenzene", got

    @pytest.mark.parametrize("smiles,expected,why", [
        ("Oc1ccccc1O", "benzene-1,2-diol", "partial: 2 of 6"),
        ("Brc1ccc(Br)cc1", "1,4-dibromobenzene", "partial: 2 of 6"),
        ("Oc1ccccc1", "phenol", "retained name pre-empts (constraint 10)"),
        ("Oc1ccccc1Cl", "2-chlorophenol", "partial"),
        ("C1CCCc2ccccc12", "1,2,3,4-tetrahydronaphthalene", "partial hydro"),
        ("C1CCC2CCCCC2C1", "decahydronaphthalene",
         "complete hydro -- already correct, proves agreement with a shipped L5 case"),
        ("c1ccccc1-c1ccccc1", "1,1'-biphenyl", "ring assembly always cites"),
        ("c1ccccc1Oc1ccccc1", "1,1'-oxydibenzene", "multiplicative always cites"),
    ])
    def test_tripwire_names_are_unchanged(self, namer, smiles, expected, why):
        assert _name(namer, smiles) == expected, why


# --------------------------------------------------------------------------- #
# 3. Predicate level -- the benzene licence helper itself                      #
# --------------------------------------------------------------------------- #
class TestBenzeneLicenceHelper:
    """Direct tests of ``benzene._benzene_l5_uniform_licence``.

    The end-to-end tests above cannot distinguish "the licence declined" from "some
    other handler won", so the predicate is also pinned directly.
    """

    def _lic(self, smiles, suffixes, prefixes, stereo=()):
        from rdkit import Chem

        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()][:6]
        return _benzene_l5_uniform_licence(
            mol, ring, suffixes, prefixes, stereo)

    def test_six_ol_licensed(self):
        assert self._lic("Oc1c(O)c(O)c(O)c(O)c1O",
                         {"ol": [1, 2, 3, 4, 5, 6]}, {}) is True

    def test_six_methyl_licensed(self):
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 6]}) is True

    def test_mixed_kinds_denied(self):
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1Cl",
                         {}, {"methyl": [1, 2, 3, 4, 5], "chloro": [6]}) is False

    def test_suffix_plus_prefix_denied(self):
        assert self._lic("Oc1c(O)c(O)c(O)c(O)c1Cl",
                         {"ol": [1, 2, 3, 4, 5]}, {"chloro": [6]}) is False

    def test_partial_denied(self):
        assert self._lic("Oc1ccccc1O", {"ol": [1, 2]}, {}) is False

    def test_stereodescriptor_denies(self):
        """P-14.3.3: one essential locant in the scope restores every locant."""
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 6]},
                         stereo=[{"locant": 1, "descriptor": "E"}]) is False

    def test_non_aromatic_six_ring_denied(self):
        """★ The licence is evaluated against BENZENE's parent hydride, so it must
        first prove the parent IS a benzene ring.

        Found by mutation testing: disabling the aromatic/carbon confirmation loop
        broke NO test, and the missing witness was the most important molecule in the
        whole class. A cyclohexane ring carbon has TWO substitutable H, so six
        identical substituents are only PARTIAL (BB:3009, BB:54823) -- but measured
        against benzene's parent hydride they look complete. Without this guard the
        predicate licenses ``hexamethylcyclohexane``, which is wrong.
        """
        from rdkit import Chem

        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        mol = Chem.MolFromSmiles("CC1C(C)C(C)C(C)C(C)C1C")
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
        assert len(ring) == 6
        assert _benzene_l5_uniform_licence(
            mol, ring, {}, {"methyl": [1, 2, 3, 4, 5, 6]}, ()) is False

    def test_heteroatom_six_ring_denied(self):
        """Second witness for the same guard, for the OTHER reason it can fail:
        an aromatic six-ring that is not all-carbon. Pyridine's N is not a
        substitutable position at all, so benzene's parent hydride does not describe
        it."""
        from rdkit import Chem

        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        mol = Chem.MolFromSmiles("Cc1c(C)c(C)nc(C)c1C")
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
        assert len(ring) == 6
        assert any(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring)
        assert _benzene_l5_uniform_licence(
            mol, ring, {}, {"methyl": [1, 2, 3, 4, 5, 6]}, ()) is False

    def test_non_six_ring_denied(self):
        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert _benzene_l5_uniform_licence(
            mol, [0, 1, 2, 3, 4], {}, {"methyl": [1, 2, 3, 4, 5]}) is False

    def test_duplicate_locant_denied(self):
        """Two decorations claiming one ring position -- fail closed."""
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {"ol": [1]}, {"methyl": [1, 2, 3, 4, 5, 6]}) is False

    def test_out_of_range_locant_denied(self):
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 7]}) is False

    def test_isotope_denies(self):
        """Constraint 10 -- isotopic labels (BB:44180) always cite locants.

        ⚠⚠ THIS TEST IS GREEN AND STRUCTURALLY BLIND, and is retained ONLY to pin
        the predicate's own arithmetic. It calls the licence DIRECTLY on a mol built
        from the raw SMILES, so it exercises a path PRODUCTION NEVER TAKES.

        Measured 2026-07-28 (workflow wf_b94d5bc8-3c9, spy validated on 2 known
        positives + 1 known negative): on the real path
        ``rules/isotopes.py:decorate_isotopic_name`` calls ``strip_isotopes`` FIRST,
        names the isotope-FREE skeleton, and splices the descriptor into the finished
        string. So ``_benzene_l5_uniform_licence`` receives a mol with
        ``isotopes_seen_in_mol == []`` and computes ``has_isotope=False``: the guard
        this test asserts is **DEAD BY CONSTRUCTION** in production.

        Consequence, with a forced-False A/B proving the licence caused it:
        ``Cc1c(C)c(C)c(C)c(C)[13c]1C`` ships ``hexamethyl(13C1)benzene`` where the
        PIN is ``1,2,3,4,5,6-hexamethyl(13C1)benzene`` (P-82.6.1.1, BB:44180 --
        "if isotopic modification requires a locant to specify its position, then all
        locants must be specified and none are omitted"; BB:44186 prints the elided
        chain analogue as "[not (2-13C)ethanol]").

        The END-TO-END tripwire that actually binds is
        ``TestIsotopeEndToEnd::test_labelled_hexamethylbenzene_keeps_locants`` below.
        Mutation testing did not catch this: mutating the PREDICATE fails this test,
        which made it look load-bearing, but the predicate is not what is broken.
        See ``.planning/audit-v29/PHASEC-WORKFLOW-FINDINGS.md``.
        """
        assert self._lic("[13CH3]c1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 6]}) is False

    def test_none_mol_denied(self):
        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        assert _benzene_l5_uniform_licence(
            None, [0, 1, 2, 3, 4, 5], {}, {"methyl": [1, 2, 3, 4, 5, 6]}) is False


# --------------------------------------------------------------------------- #
class TestIsotopeEndToEnd:
    """END-TO-END isotope locant behaviour (v29 Phase C).

    THE DEFECT: ``rules/isotopes.py:decorate_isotopic_name`` strips every label, names the
    isotope-FREE skeleton, then splices the descriptor into the finished string. The
    P-14.3.4 licences therefore saw ``GetIsotope() == 0`` everywhere, every
    ``has_isotope`` guard was structurally unreachable-True (measured with a spy validated
    on 2 positives + 1 negative), and we shipped ``(2-13C1)cyclohexanol``.

    THE FIX: when the descriptor itself needs a locant, **P-82.6.1.1** (``:44180``)
    restores the parent's locants -- *"if isotopic modification requires a locant to
    specify its position, then all locants must be specified and none are omitted"* --
    and its own example prints the elided form as the rejected one (``:44186``):
    ``13CH3-CH2-OH (2-13C)ethan-1-ol [not (2-13C)ethanol]``. The decorator re-names the
    skeleton inside ``locant_omission.forced_locant_scope()``, an ambient P-14.3.3 scope
    the licences consult, so the parent's locants come back.

    ★ THE CONDITION IS "THE DESCRIPTOR NEEDS A LOCANT", NOT "AN ISOTOPE IS PRESENT".
    A blanket rule over-cites, and the tests below pin both sides. Two names the
    adversarial sweep reported as defects are in fact CORRECT, and are kept here as
    passing tests precisely because they are the near-miss neighbours of a real one --
    see ``test_hexamethylbenzene_scoped_descriptor_is_correct``.
    """

    # ---------------- FIXED: the descriptor carries a locant => parent cites --------- #

    def test_labelled_cyclohexanol_keeps_its_locant(self):
        from orthonym.namer import Orthonym
        assert Orthonym().name("OC1CCCC[13CH2]1") == "(2-13C)cyclohexan-1-ol"

    def test_labelled_cyclohexanone_keeps_its_locant(self):
        """The ketone class too -- the defect spanned the whole licence allowlist."""
        from orthonym.namer import Orthonym
        assert Orthonym().name("O=C1CCCC[13CH2]1") == "(2-13C)cyclohexan-1-one"

    def test_deuterium_keeps_its_locant(self):
        from orthonym.namer import Orthonym
        assert Orthonym().name("[2H]C1CCCCC1O") == "(2-2H1)cyclohexan-1-ol"

    def test_chain_analogue_matches_the_bb_example(self):
        """BB:44186 verbatim: ``13CH3-CH2-OH (2-13C)ethan-1-ol [not (2-13C)ethanol]``.
        The count subscript is omitted per P-82.2.1 (FIX-A), matching the BB
        verbatim ``(2-13C)`` exactly."""
        from orthonym.namer import Orthonym
        assert Orthonym().name("[13CH3]CO") == "(2-13C)ethan-1-ol"

    # ---------------- CORRECT ALREADY: omission is right, sweep over-claimed -------- #

    def test_hexamethylbenzene_scoped_descriptor_is_correct(self):
        """★ The adversarial sweep reported ``hexamethyl(13C1)benzene`` as a defect and
        it is NOT one. **P-82.2.1** (``:44182``, verbatim) says the descriptor is inserted
        *"before the part of the compound that is isotopically substituted"* -- so its
        POSITION carries its scope. Here it precedes ``benzene``, scoping the ring, whose
        six carbons are all equivalent; one isotopomer, no locant required, so
        P-82.6.1.1's condition is not met and the parent keeps its licensed omission.

        ``:7492`` ``1,2-di[(13C)methyl]benzene (PIN. P-82.2.1)`` is the direct witness for
        scoped descriptors carrying no locant of their own.

        Contrast ``:44206`` ``1-(79Br)bromo(2-13C)benzene (PIN)``: bromobenzene's ring
        carbons are NOT equivalent, so there the label does need a locant and the bromo
        locant is duly restored. That is the real discriminator."""
        from orthonym.namer import Orthonym
        assert Orthonym().name("Cc1c(C)c(C)c(C)c(C)[13c]1C") == "hexamethyl(13C)benzene"

    def test_hexafluorobenzene_correctly_omits(self):
        """Also reported by the sweep, also correct: hexafluorobenzene has exactly ONE
        carbon environment (fluorine is not carbon), so one 13C gives one isotopomer."""
        from orthonym.namer import Orthonym
        assert Orthonym().name("Fc1c(F)c(F)c(F)c(F)[13c]1F") == "hexafluoro(13C)benzene"

    def test_uniformly_equivalent_ring_correctly_omits(self):
        """P-82.6.1.3 (``:44202``) -- all six ring positions are one orbit, so labelling
        any of them gives the same compound. Must stay locant-free."""
        from orthonym.namer import Orthonym
        assert Orthonym().name("O[13c]1c(O)c(O)c(O)c(O)c1O") == "(13C)benzenehexol"

    def test_scoped_descriptor_on_an_ester_alkyl_is_correct(self):
        """★ THE WITNESS THAT REFUTED AN OVER-BROAD FIX. A whole-molecule
        isotopomer-ambiguity test was built, wired, and REGRESSED this name to
        ``(1-13C1)methyl acetate``. Methyl acetate has three distinct carbons, so a
        whole-molecule test calls it ambiguous -- but the descriptor precedes ``methyl``,
        whose scope is a single carbon, so the name is already unique (P-82.2.1). The
        over-broad test was withdrawn; this row guards against re-introducing it."""
        from orthonym.namer import Orthonym
        assert Orthonym().name("[13CH3]OC(C)=O") == "(13C)methyl acetate"

    # ---------------- STILL OPEN, with evidence ------------------------------------- #

    @pytest.mark.xfail(strict=True, reason=(
        "OPEN DEFECT, evidenced not guessed. Two 13C on benzenehexol: the 1,2- / 1,3- / "
        "1,4-isotopomers are three distinct compounds, so P-82.6.1.4 (BB:44198, "
        "'Locants are not omitted when there is a possibility of isomers') requires the "
        "label locants, and P-82.6.1.1 would then restore the six hydroxy locants. "
        "MEASURED: we emit `(13C2)benzenehexol` for the 1,2-isotopomer only because "
        "OPSIN's default parse lands there, while the 1,3- and 1,4-isotopomers emit "
        "`unknown organic compound`. Blocked on a MULTI-locant descriptor: "
        "`_descriptor(locant)` takes a single locant and cannot spell `(1,2-13C2)`. "
        "Not fixed by the ambient-scope mechanism, because the trigger is the "
        "descriptor's own locant, which is never reached here."))
    def test_double_labelled_benzenehexol_needs_locants(self):
        from orthonym.namer import Orthonym
        assert Orthonym().name("O[13c]1[13c](O)c(O)c(O)c(O)c1O") == \
            "(1,2-13C2)benzene-1,2,3,4,5,6-hexol"


class TestForcedLocantScope:
    """The ambient P-14.3.3 mechanism itself (``assembly/locant_omission.py``)."""

    def test_scope_is_inert_by_default(self):
        from orthonym.assembly.locant_omission import locants_are_forced
        assert locants_are_forced() is False

    def test_scope_activates_and_restores(self):
        from orthonym.assembly.locant_omission import (
            forced_locant_scope,
            locants_are_forced,
        )
        assert locants_are_forced() is False
        with forced_locant_scope("test"):
            assert locants_are_forced() is True
        assert locants_are_forced() is False, "the ContextVar token must be reset"

    def test_both_licences_decline_inside_the_scope(self):
        """Wired at two sites; if either stops consulting the scope, the isotope defect
        returns silently -- SELF-01 cannot see it (`namer.py` states verbatim that it
        'ignores isotopes')."""
        from rdkit import Chem

        from orthonym.assembly.locant_omission import forced_locant_scope
        from orthonym.rules.benzene import _benzene_l5_uniform_licence

        mol = Chem.MolFromSmiles("Cc1c(C)c(C)c(C)c(C)c1C")
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()][:6]
        args = (mol, ring, {}, {"methyl": [1, 2, 3, 4, 5, 6]}, ())
        assert _benzene_l5_uniform_licence(*args) is True, "licensed outside the scope"
        with forced_locant_scope("isotope"):
            assert _benzene_l5_uniform_licence(*args) is False, \
                "the benzene L5 licence must decline inside a forced-locant scope"
