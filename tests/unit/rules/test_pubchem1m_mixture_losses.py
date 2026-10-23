"""PubChem 1M -- mixtures the paper run named and the breadth branch abstained (lane M).

Two causes, both in the adduct assembly "Organic adducts": "Names are
formed by citing the names of individual compounds in the order of the formula connected
by long (em) dashes"). The rows are PubChem rows (none is in eval/splits/a holdout split.json);
every shipped name is read back by an independent OPSIN call to the input's full InChIKey
(tests/support/rt_assert).

  A A component is a compound of its own and has the hang budgets of one compound
     (``fragment_naming.own_hang_budgets``). The best-effort cage ceilings of 853298f0c (100
     skeletal atoms, 11 rings) let the von Baeyer analysis of a large spiro-fused component run
     those budgets out, and the ``PerfBudgetExceeded`` of the nested ``name`` unwound to the
     frame that owns the assembly's budgets, which abstained the whole drawing although every
     other component was named. A compound named alone meets the same trip at its outermost
     ``name``, which offers the bounded rescue first, in the tier context of an outermost
     call; the component now gets that offer (``adducts._budget_rescue_name``), then one run
     under the PIN ceilings of the von Baeyer analyses, where a ring system beyond them is
     refused at once (``adducts._pin_ceiling_name``). A spelling built for ``_readable_spellings``
     that runs its budgets out is left out instead of ending the adduct.
  C A drawing of several fragments whose ions all carry one sign (photoacid-generator sulfonates:
     net charge -2 to -16, no cation) is routed to the POLY_ANION handler, whose neutralise-and-name
     pipeline names the acids (other species) and runs the budgets of the whole molecule out in
     the nested namings of its drug-size components; the trip unwound to the outermost name,
     which abstained before the late recovery could offer the adduct of the ions as drawn. At the
     best-effort tier the pipeline now has the budgets of one compound, a trip is a producer that
     names nothing, and then the adduct of the CHARGED drawing is offered
     (``dispatch_table._handle_poly_anion``), read back by the exit's rule before it is returned.
  D A single compound whose main path and rescue both run the hang budgets out at the best-effort
     ceilings is named once more under the PIN ceilings (``Orthonym._try_pin_ceiling_rescue``, the
     shared rule of ``assembly.budget_rescue``), and ships only on an exact OPSIN read-back.
  B The late recovery built the adduct name, OPSIN read it back to the input's full InChIKey,
     and ``_rt_match`` withdrew it because the canonical SMILES differ: OPSIN draws a
     delocalised cation (imidazolium, azetium, benzothiazolium,...) on the other resonance
     atom than the input has it, and a component's name spells either form. The exit asks every
     general tier for ``_shipped_name_round_trip`` alone (the full key, the radical graph and
     the protonation site); the recovery's adduct, and the stereo-emit check that reads it, are
     now judged by that rule (``Orthonym._candidate_rt_match``): a hydron or a charge on another
     site of a base or an acid is another species to it (``protonation_site_verdict``).
"""
import contextlib

import pytest
from rdkit import Chem

from orthonym import namer as nm
from orthonym.assembly import fragment_naming as fn
from orthonym.rules import adducts
from orthonym.rules import vonbaeyer_universal as vbu
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

EM = "—"


# --------------------------------------------------------------------------
# A a component that runs its hang budgets out
# --------------------------------------------------------------------------

def _trips(*_a, **_k):
    raise fn.PerfBudgetExceeded()


def test_component_budget_trip_takes_the_t4_rescue(monkeypatch):
    """The nested ``name`` of the component trips; the rescue names it, as it does at the
    outermost ``name`` of a compound named alone."""
    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    seen = []

    def rescue(self, smiles):
        seen.append(smiles)
        return "ethanol"

    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", rescue)
    got = adducts._name_component("CCO", "pin", general_fallback_unverified=True)
    assert got == "ethanol" and seen == ["CCO"]


def test_component_budget_trip_without_a_rescue_unwinds_as_before(monkeypatch):
    """A component the rescue does not name (every tier but the best-effort one, or a
    re-explosion) leaves the trip to unwind to the frame that owns the budgets."""
    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", lambda self, smiles: None)
    with pytest.raises(fn.PerfBudgetExceeded):
        adducts._name_component("CCO", "pin", general_fallback_unverified=True)


def test_rescued_component_is_never_labelled_a_pin(monkeypatch):
    """A name is a general-engine name: the adduct that carries it is labelled below the PIN."""
    from orthonym.metrics import provenance as pv
    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", lambda self, smiles: "ethanol")
    pv.clear_provenance()
    adducts._name_component("CCO", "pin", general_fallback_unverified=True)
    assert pv.get_provenance()["source"] == "general_engine"
    assert "ethanol" in pv.get_provenance()["non_pin_fragments"]


def test_rescue_runs_in_the_context_of_an_outermost_call(monkeypatch):
    """The rescue of a compound named alone runs after the outermost ``name`` has unwound the
    tier contexts it published; inside a mixture they are set, and the producer then ran the
    budget of the rescue out for components it names alone."""
    from orthonym.metrics.provenance import best_effort_ctx, general_fallback_ctx
    seen = []

    def rescue(self, smiles):
        seen.append((best_effort_ctx.get(), general_fallback_ctx.get()))
        return "ethanol"

    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", rescue)
    tokens = [(best_effort_ctx, best_effort_ctx.set(True)),
              (general_fallback_ctx, general_fallback_ctx.set(True))]
    try:
        adducts._name_component("CCO", "pin", general_fallback_unverified=True)
        assert (best_effort_ctx.get(), general_fallback_ctx.get()) == (True, True)  # restored
    finally:
        for var, token in tokens:
            var.reset(token)
    assert seen == [(False, False)]


def test_component_the_rescue_cannot_name_is_named_under_the_pin_ceilings(monkeypatch):
    """Under the best-effort ceilings the analysis runs the budgets out and the rescue
    finds no name; under the PIN ceilings a ring system beyond them is refused at once and the
    component is named by another producer (as before the best-effort ceilings existed)."""
    ceilings = []

    def name(self, smiles):
        ceilings.append(vbu.cage_caps())
        if vbu.cage_caps() != (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS):
            raise fn.PerfBudgetExceeded()
        return "ethanol"

    monkeypatch.setattr(nm.Orthonym, "name", name)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", lambda self, smiles: None)
    from orthonym.metrics.provenance import best_effort_request_ctx
    token = best_effort_request_ctx.set(True)
    try:
        got = adducts._name_component("CCO", "pin", general_fallback_unverified=True)
        assert vbu.cage_caps() == (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    finally:
        best_effort_request_ctx.reset(token)
    assert got == "ethanol"
    assert ceilings == [(vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS),
                        (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS)]


def test_pin_ceilings_are_for_the_best_effort_tier_only(monkeypatch):
    """Where the rescue is not offered (every tier but the best-effort one) there is no second run
    either: the trip unwinds as it always did."""
    calls = []

    def name(self, smiles):
        calls.append(smiles)
        raise fn.PerfBudgetExceeded()

    monkeypatch.setattr(nm.Orthonym, "name", name)
    with pytest.raises(fn.PerfBudgetExceeded):
        adducts._name_component("CCO", "pin", general_fallback=True)
    assert calls == ["CCO"]


def test_pin_ceilings_scope_overrides_the_tier_and_restores_it():
    from orthonym.metrics.provenance import best_effort_request_ctx
    token = best_effort_request_ctx.set(True)
    try:
        best_effort = vbu.cage_caps()
        with vbu.pin_ceilings_scope():
            assert vbu.cage_caps() == (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS)
        assert vbu.cage_caps() == best_effort
        assert best_effort == (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    finally:
        best_effort_request_ctx.reset(token)


def test_salt_ion_budget_trip_takes_the_t4_rescue(monkeypatch):
    """An ion of a salt is a compound of its own with the hang budgets of one compound
    (``salts._name_organic_ion_best_effort``): a trip in its nested ``name`` is offered what a
    compound named alone is offered, instead of ending the whole salt."""
    from orthonym.rules import salts
    seen = []

    def rescue(self, smiles):
        seen.append(smiles)
        return "acetate"

    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", rescue)
    got = salts._name_organic_ion_best_effort("CC(=O)[O-]", general_fallback_unverified=True)
    assert got == "acetate" and seen == ["CC(=O)[O-]"]


def test_salt_ion_budget_trip_without_a_rescue_unwinds_as_before(monkeypatch):
    from orthonym.rules import salts
    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", lambda self, smiles: None)
    with pytest.raises(fn.PerfBudgetExceeded):
        salts._name_organic_ion_best_effort("CC(=O)[O-]", general_fallback_unverified=True)
    # every tier but the best-effort one: no ion word (the flags are what gate the widening)
    assert salts._name_organic_ion_best_effort("CC(=O)[O-]") == ""


def test_the_adduct_and_the_salt_share_one_rule_for_the_trip(monkeypatch):
    """One rule for a part that runs its budgets out: ``budget_rescue.name_after_budget_trip``."""
    from orthonym.assembly import budget_rescue
    from orthonym.rules import salts
    calls = []

    def rule(namer, smiles):
        calls.append(smiles)
        return {"CCO": "ethanol", "CC(=O)[O-]": "acetate"}[smiles]

    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(budget_rescue, "name_after_budget_trip", rule)
    assert adducts._name_component("CCO", "pin", general_fallback_unverified=True) == "ethanol"
    assert salts._name_organic_ion_best_effort("CC(=O)[O-]", general_fallback_unverified=True) == "acetate"
    assert calls == ["CCO", "CC(=O)[O-]"]


def _armed_scope():
    fn.enter_name_scope("X")                 # depth 1, the budgets armed as at the outermost name
    guard = fn._fragment_guard
    before = (guard.perf_budget, guard.analysis_budget)
    assert None not in before
    return guard, before


@pytest.mark.parametrize("part", ["ion", "component"])
def test_the_enclosing_budgets_survive_the_offers_to_a_part_that_trips(monkeypatch, part):
    """The rescue disarms the budgets it ran under. A salt handles the trip of an ion outside the
    ion's own ``own_hang_budgets``, so the offers have budgets of their own: the budgets of the
    molecule that encloses the part are back as they were, rescued or not."""
    from orthonym.rules import salts
    call = {
        "ion": lambda: salts._name_organic_ion_best_effort(
            "CC(=O)[O-]", general_fallback_unverified=True),
        "component": lambda: adducts._name_component(
            "CCCO", "pin", general_fallback_unverified=True),
    }[part]
    named = {"ion": "acetate", "component": "propan-1-ol"}[part]
    monkeypatch.setattr(nm.Orthonym, "name", _trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue",
                        lambda self, smiles: fn.disarm_hang_budgets() or named)
    guard, before = _armed_scope()
    try:
        assert call() == named
        assert (guard.perf_budget, guard.analysis_budget) == before
        monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue",
                            lambda self, smiles: fn.disarm_hang_budgets())     # no name: the trip stands
        with pytest.raises(fn.PerfBudgetExceeded):
            call()
        assert (guard.perf_budget, guard.analysis_budget) == before
    finally:
        fn.exit_name_scope()


def test_rescue_comes_before_the_pin_ceilings(monkeypatch):
    from orthonym.assembly import budget_rescue
    order = []
    monkeypatch.setattr(budget_rescue, "t4_rescue_name", lambda namer, smiles: order.append("t4") or None)
    monkeypatch.setattr(budget_rescue, "pin_ceiling_name", lambda namer, smiles: order.append("pin") or "named")
    assert budget_rescue.name_after_budget_trip(object(), "CCO") == "named"
    monkeypatch.setattr(budget_rescue, "t4_rescue_name", lambda namer, smiles: order.append("t4") or "rescued")
    assert budget_rescue.name_after_budget_trip(object(), "CCO") == "rescued"
    assert order == ["t4", "pin", "t4"]


def test_spelling_that_runs_its_budget_out_is_left_out(monkeypatch):
    """``_component_spellings`` builds optional spellings: a trip in either builder is
    'no other spelling', not the end of the adduct."""
    import orthonym.assembly.t4_coverage as t4
    monkeypatch.setattr(nm.Orthonym, "_engine_recovery_spelling", _trips)
    monkeypatch.setattr(t4, "_verified_universal_floor", _trips)
    assert adducts._component_spellings(
        "CCO", "ethanol", "pin", dict(general_fallback_unverified=True)) == []


# A mixture of ethane and a 69-heavy-atom dispiro component whose von Baeyer analysis runs
# the budgets of one compound out (PubChem 1M); named alone it is named by the rescue.
ETHANE_AND_SPIRO_ROW = (
    "CC.CCC1=C(C(=CC=C1)C2=CCC=C3C=CC4=C(C=CC5=C4C3=C2C=C5)C6=C7C=CC=C8C7=C9C(=C6)C=CC=C9C1=C8"
    "C2(C3C=CC=CC3C3=C2C=CCCC3)C2=CC=CC=C21)/C=C\\C")
# Three dispiro components of PubChem 1M (a thiapentacyclo-fluorene spiro system on a triazine and
# a pyrimidine), each of which trips the budgets of one compound.
SPIRO_TRIPLE_ROW = (
    "C1=CC=C(C=C1)C2=CC(=NC(=N2)C3=CC4=C(C=C3)C5=C(C46C7=CC=CC=C7C8(C9=CC=CC=C9C1=CC=CC=C18)"
    "C1=CC=CC=C61)C=CC1=C5SC2=CC=CC=C12)C1=CC=CC=C1."
    "C1=CC=C(C=C1)C2=NC(=NC(=N2)C3=CC=C(C=C3)C4=CC5=C(C=C4)C6=C(C57C8=CC=CC=C8C9(C1=CC=CC=C1"
    "C1=CC=CC=C19)C1=CC=CC=C71)C=CC1=C6C2=CC=CC=C2S1)C1=CC=CC=C1."
    "C1=CC=C(C=C1)C2=NC(=NC(=N2)C3=CC4=C(C=C3)C5=C(C46C7=CC=CC=C7C8(C9=CC=CC=C9C1=CC=CC=C18)"
    "C1=CC=CC=C61)C=CC1=C5SC2=CC=CC=C12)C1=CC=CC=C1")


@pytest.mark.parametrize("smiles, joins", [
    (SPIRO_TRIPLE_ROW, 2),
    (ETHANE_AND_SPIRO_ROW, 1),
])
def test_mixture_with_a_component_that_runs_its_budgets_out_is_named(smiles, joins):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name.count(EM) == joins, name
    assert res.get("tier") in ("systematic_verified", "best_effort"), res.get("tier")


# --------------------------------------------------------------------------
# B an adduct whose cation OPSIN draws on the other resonance atom
# --------------------------------------------------------------------------

#: A 1,3-substituted imidazolium (the ring cation of a PubChem 1M row of 13 components): the name
#: OPSIN reads gives the charge to the other nitrogen than the input has it.
IMIDAZOLIUM_ADDUCT = "CCC1=C[N+](=CN1)C.CC"


@contextlib.contextmanager
def _whole_molecule(smiles, relaxed=False):
    """The state of the recovery of the outermost naming of ``smiles``: its input, the notes of the
    first pass cleared, and (``relaxed``) the pass of ``default_tier_offer.relaxed_adduct_row``."""
    from orthonym.assembly import default_tier_offer as dto
    dto.clear_held()
    fn.enter_name_scope(smiles)
    token = dto._RELAXED_ON.set(relaxed)
    try:
        yield dto
    finally:
        dto._RELAXED_ON.reset(token)
        fn.exit_name_scope()
        dto.clear_held()


def _read(name):
    opsin = nm._validity_gate_name_to_smiles(name)
    assert opsin is not None
    return opsin


def test_first_pass_keeps_the_exact_comparison_and_notes_the_other_resonance_form():
    name = f"4-ethyl-1-methyl-1H-imidazol-3-ium{EM}ethane (1/1)"
    opsin = _read(name)
    # canonical SMILES differ (the charge sits on the other nitrogen), the species is one
    assert not nm.Orthonym._rt_match(IMIDAZOLIUM_ADDUCT, opsin, False)
    with _whole_molecule(IMIDAZOLIUM_ADDUCT) as dto:
        assert not nm.Orthonym._candidate_rt_match(name, IMIDAZOLIUM_ADDUCT, opsin, False)
        assert dto._RELAXED_SEEN.get()           # noted: the molecule may be offered the pass
    # an exact reading is accepted in every pass, and notes nothing
    exact = f"1,3-dimethyl-1H-imidazol-3-ium{EM}ethane (1/1)"
    with _whole_molecule("C[n+]1ccn(C)c1.CC") as dto:
        assert nm.Orthonym._candidate_rt_match(exact, "C[n+]1ccn(C)c1.CC", _read(exact), False)
        assert not dto._RELAXED_SEEN.get()


def test_relaxed_pass_accepts_the_other_resonance_form_of_an_adduct():
    name = f"4-ethyl-1-methyl-1H-imidazol-3-ium{EM}ethane (1/1)"
    opsin = _read(name)
    with _whole_molecule(IMIDAZOLIUM_ADDUCT, relaxed=True):
        assert nm.Orthonym._candidate_rt_match(name, IMIDAZOLIUM_ADDUCT, opsin, False)
        # a name that asserts less stereo is judged by _rt_match alone
        assert not nm.Orthonym._candidate_rt_match(name, IMIDAZOLIUM_ADDUCT, opsin, True)
        # a name that is not an adduct of components keeps the exact comparison
        assert not nm.Orthonym._candidate_rt_match(
            "4-ethyl-1-methyl-1H-imidazol-3-ium", IMIDAZOLIUM_ADDUCT, opsin, False)
    # a part of the molecule (another input than the outermost naming's) keeps it too
    with _whole_molecule("CCO.CC", relaxed=True) as dto:
        assert not nm.Orthonym._candidate_rt_match(name, IMIDAZOLIUM_ADDUCT, opsin, False)
        assert not dto._RELAXED_SEEN.get()


def test_relaxed_pass_still_rejects_another_species():
    """A reading with another full key is not the input."""
    other = f"1,3-dimethyl-1H-imidazol-3-ium{EM}ethane (1/1)"
    opsin = _read(other)
    with _whole_molecule(IMIDAZOLIUM_ADDUCT, relaxed=True):
        assert not nm.Orthonym._candidate_rt_match(other, IMIDAZOLIUM_ADDUCT, opsin, False)


def test_relaxed_pass_accepts_the_other_tautomer_of_a_guanidine_adduct():
    """OPSIN reads a guanidine with the double bond on the other nitrogen; a name places no mobile
    hydrogen, and the exit accepts it (the full key, no protonated atom)."""
    name = f"N-(3-phenylpropyl)guanidine{EM}ethane (1/1)"
    smiles = "NC(N)=NCCCc1ccccc1.CC"
    opsin = _read(name)
    assert not nm.Orthonym._rt_match(smiles, opsin, False)
    with _whole_molecule(smiles, relaxed=True):
        assert nm.Orthonym._candidate_rt_match(name, smiles, opsin, False)


def test_relaxed_pass_rejects_the_hydron_on_another_site():
    """The protonation site is the one thing the standard key cannot see: an adduct whose cation is
    read with the hydron on another atom than the input has it is another species."""
    name = f"N-methyl-3-(methylamino)propanamidium{EM}ethane (1/1)"
    smiles = "C[NH2+]CCC(=O)NC.CC"
    opsin = _read(name)
    with _whole_molecule(smiles, relaxed=True):
        assert not nm.Orthonym._candidate_rt_match(name, smiles, opsin, False)


def test_pass_is_offered_only_to_a_molecule_whose_first_pass_noted_a_candidate(monkeypatch):
    from orthonym.assembly import default_tier_offer as dto
    monkeypatch.setattr(dto, "general_tier_engine",
                        lambda namer: pytest.fail("a pass for a molecule that noted nothing"))
    namer = _best_effort_namer()
    dto.clear_held()
    assert dto.relaxed_adduct_row(namer, IMIDAZOLIUM_ADDUCT) is None
    assert dto.default_tier_row_for(namer, IMIDAZOLIUM_ADDUCT) is None
    dto.note_relaxed_adduct()
    assert dto.relaxed_adduct_row(nm.Orthonym(style="pin"), IMIDAZOLIUM_ADDUCT) is None  # not best-effort
    dto.clear_held()                                                                      # next naming
    assert not dto._RELAXED_SEEN.get()


def test_pass_gives_no_pass_inside_the_pass(monkeypatch):
    from orthonym.assembly import default_tier_offer as dto
    monkeypatch.setattr(dto, "general_tier_engine",
                        lambda namer: pytest.fail("a pass inside the pass"))
    dto.note_relaxed_adduct()
    token = dto._RELAXED_ON.set(True)
    try:
        assert dto.relaxed_adduct_row(_best_effort_namer(), IMIDAZOLIUM_ADDUCT) is None
    finally:
        dto._RELAXED_ON.reset(token)
        dto.clear_held()


# A cationic dye and ethane. The charge is drawn on the QUINOLINE nitrogen, so the cation parent is the quinolinium,
# the other ring a prefix with its ylidene bond (2Z) kept.
#
# Change-asserted-value record. The name this test pinned was
# '3-methyl-2-[(1-methyl-1,4-dihydroquinolin-4-ylidene)methyl]-1,3-benzothiazol-3-ium—ethane (1/1)': the charge on the
# benzothiazole nitrogen, the other resonance form, and the (2Z) of the drawn bond dropped.
# (a) Rule: the drawing as written is the structure to name (the contributor guide), and "General rule for
# systematically naming cationic centers in parent hydrides" (the Blue Book) puts the suffix 'ium' on the
# atom that carries the charge; the configuration of the drawn double bond is "Specification by
# stereodescriptors 'E' and 'Z'" (:48271), method (b) here, so the name is labelled below the PIN.
# (b) Independent check: OPSIN's own parse of the name (tests.support.rt_assert._independent_parse, not the engine)
# puts the [N+] in the ring that holds no sulfur, the quinoline ring, as the input does; the old name's parse
# puts it in the benzothiazole ring (``test_the_old_name_fails_the_charge_site_check``).
# (c) Mutation: with the ring-ylidene prefix of a stereo-flagged bond switched off (the wide-tier producer of lane S,
# ``ring_ylidene_carrier._ring_ylidene_bonds(..., allow_stereo=True)``) the tree gives the old name again
# (``test_without_the_stereo_ylidene_prefix_the_old_name_returns``).
CYANINE_ETHANE = "CN\\1C2=CC=CC=C2S/C1=C\\C3=CC=[N+](C)C4=CC=CC=C34.CC"
CYANINE_ETHANE_NAME = ("1-methyl-4-{[(2Z)-3-methyl-2,3-dihydro-1,3-benzothiazol-2-ylidene]methyl}quinolin-1-ium"
                       f"{EM}ethane (1/1)")
CYANINE_ETHANE_OLD_NAME = ("3-methyl-2-[(1-methyl-1,4-dihydroquinolin-4-ylidene)methyl]-1,3-benzothiazol-3-ium"
                           f"{EM}ethane (1/1)")


def _charged_nitrogen_ring_has_sulfur(smiles: str) -> bool:
    """Whether the ring of the cationic nitrogen of ``smiles`` (its smallest ring) holds a sulfur atom."""
    mol = Chem.MolFromSmiles(smiles)
    rings = mol.GetRingInfo().AtomRings()
    atom = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() == 1)
    ring = min((r for r in rings if atom.GetIdx() in r), key=len)
    return any(mol.GetAtomWithIdx(i).GetSymbol() == "S" for i in ring)


def test_a_name_that_ships_is_not_changed_by_the_pass(monkeypatch):
    from orthonym.assembly import default_tier_offer as dto
    from tests.support.rt_assert import _independent_parse
    res = name_best_effort(CYANINE_ETHANE)
    assert res.get("name") == CYANINE_ETHANE_NAME
    assert_full_rt(res.get("name"), CYANINE_ETHANE)
    # the charge is where the input draws it: the quinoline ring (no sulfur), by OPSIN's own parse of the name
    assert not _charged_nitrogen_ring_has_sulfur(CYANINE_ETHANE)
    assert not _charged_nitrogen_ring_has_sulfur(_independent_parse(res["name"], CYANINE_ETHANE))
    # the relaxed pass changes no name that ships: the tree without the pass gives the same name
    monkeypatch.setattr(dto, "relaxed_adduct_row", lambda *_a, **_k: None)
    assert name_best_effort(CYANINE_ETHANE).get("name") == CYANINE_ETHANE_NAME


def test_the_old_name_fails_the_charge_site_check():
    from tests.support.rt_assert import _independent_parse
    assert _charged_nitrogen_ring_has_sulfur(_independent_parse(CYANINE_ETHANE_OLD_NAME, CYANINE_ETHANE))


def test_without_the_stereo_ylidene_prefix_the_old_name_returns(monkeypatch):
    from orthonym.rules import ring_ylidene_carrier as carrier
    built = carrier._ring_ylidene_bonds
    monkeypatch.setattr(carrier, "_ring_ylidene_bonds",
                        lambda mol, frag, ring, allow_stereo=False: built(mol, frag, ring, False))
    assert name_best_effort(CYANINE_ETHANE).get("name") == CYANINE_ETHANE_OLD_NAME


@pytest.mark.parametrize("smiles", [
    # ethane and an azetium: the adduct is read with the charge on the other ring atom
    "CC.CC1=CC=C(C=C1)C2=C(C=[N+]2CCC#N)C3=C4C=C(OC4=NC=N3)C",
    # three imidazolium / triazolium cations of one drawing
    "CC(C)(C)CCCN1C=C[N+](=C1)C.CC(C)(C)CCCN1C=C[N+](=C1)C.CC(C)(C)CCCN1C=C[N+](=N1)C."
    "CC(C)(CCCN1C=C[N+](=C1)C)C2=CC=CC=C2",
])
def test_cationic_adduct_is_named(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert EM in name and name.endswith(")"), name


# --------------------------------------------------------------------------
# C an adduct of anions only (no cation in the drawing)
# --------------------------------------------------------------------------

def _poly_anion_handler(monkeypatch, *, pipeline, adduct, verdict="verified", flags=None):
    from orthonym.routing import dispatch_table as dt
    calls = {"pipeline": 0, "adduct": 0}

    def body(mol, smiles, canonical_smiles, features=None, *, style="pin", **kwargs):
        calls["pipeline"] += 1
        return pipeline()

    def offer(mol, canonical_smiles, style="pin", **kwargs):
        calls["adduct"] += 1
        calls["flags"] = kwargs
        return adduct

    monkeypatch.setattr(dt, "_name_poly_anion", body)
    monkeypatch.setattr("orthonym.rules.adducts.name_adduct", offer)
    monkeypatch.setattr(nm, "_shipped_name_round_trip", lambda name, smiles: verdict)
    smiles = "CC(F)(F)S(=O)(=O)[O-].CCC(F)(F)S(=O)(=O)[O-]"
    mol = Chem.MolFromSmiles(smiles)
    kwargs = {"general_fallback_unverified": True} if flags is None else flags
    return dt._handle_poly_anion(mol, smiles, Chem.MolToSmiles(mol), None, style="pin", **kwargs), calls


def _pipeline_trips():
    raise fn.PerfBudgetExceeded()


def test_anion_only_drawing_whose_pipeline_runs_its_budgets_out_is_offered_its_adduct(monkeypatch):
    got, calls = _poly_anion_handler(monkeypatch, pipeline=_pipeline_trips,
                                     adduct=f"1,1-difluoroethanesulfonate{EM}x (1/1)")
    assert got == f"1,1-difluoroethanesulfonate{EM}x (1/1)"
    assert calls["adduct"] == 1 and calls["flags"]["general_fallback_unverified"] is True


def test_anion_only_drawing_the_pipeline_names_keeps_that_name(monkeypatch):
    got, calls = _poly_anion_handler(monkeypatch, pipeline=lambda: "a name", adduct="never")
    assert got == "a name" and calls["adduct"] == 0


def test_adduct_that_does_not_read_back_is_not_offered(monkeypatch):
    got, calls = _poly_anion_handler(monkeypatch, pipeline=lambda: None, adduct=f"a{EM}b (1/1)",
                                     verdict="failed")
    assert got is None and calls["adduct"] == 1


def test_anion_only_offer_is_for_the_best_effort_tier_only(monkeypatch):
    # every other tier runs the pipeline as before: a trip is not caught, no adduct is offered
    with pytest.raises(fn.PerfBudgetExceeded):
        _poly_anion_handler(monkeypatch, pipeline=_pipeline_trips, adduct="never", flags={})
    got, calls = _poly_anion_handler(monkeypatch, pipeline=lambda: None, adduct="never", flags={})
    assert got is None and calls["adduct"] == 0


def test_single_fragment_poly_anion_is_not_offered_an_adduct(monkeypatch):
    from orthonym.routing import dispatch_table as dt
    monkeypatch.setattr(dt, "_name_poly_anion", lambda *a, **k: None)
    monkeypatch.setattr("orthonym.rules.adducts.name_adduct",
                        lambda *a, **k: pytest.fail("an adduct of one fragment"))
    mol = Chem.MolFromSmiles("[O-]S(=O)(=O)CCS(=O)(=O)[O-]")
    assert dt._handle_poly_anion(mol, "x", Chem.MolToSmiles(mol), None, style="pin",
                                 general_fallback_unverified=True) is None


# Two sulfonate anions of PubChem 1M (photoacid generators), 60 to 70 heavy atoms each: the nested
# namings of the neutralised drawing ran the hang budgets of the whole molecule out.
SULFONATE_PAIR_ROWS = [
    "CC(C(F)(F)S(=O)(=O)[O-])OC(=O)CCC(=O)OC1C2C(C3C(C1OC3=O)O2)C(=O)OC4(CCCCC4)C5CC6CCC5O6.CC(C(F)(F)S(=O)(=O)[O-])OC(=O)CCC(=O)OC1C2C(C3C(C1OC3=O)O2)C(=O)OC4(CCCC4)C5CC6CCC5O6",
    "CC(C)C1(CCCCC1)OC(=O)C2C3CC4C2C(=O)OC4C3OC(=O)COCC(=O)OC(C)C(F)(F)S(=O)(=O)[O-].CC(C(F)(F)S(=O)(=O)[O-])OC(=O)COCC(=O)OC1C2CC3C1OC(=O)C3C2C(=O)OC4(CCCCC4)C(C)(C)C",
]


@pytest.mark.parametrize("smiles", SULFONATE_PAIR_ROWS)
def test_adduct_of_two_sulfonate_anions_is_named(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name.count(EM) == 1 and "sulfonat" in name, name


# --------------------------------------------------------------------------
# S one naming per component and top-level naming (speed)
# --------------------------------------------------------------------------

@pytest.fixture
def memo_scope():
    from orthonym.assembly import memo
    token = memo.push_scope()
    yield
    memo.pop_scope(token)


def _counting_name(monkeypatch, name="propan-1-ol", record=None):
    calls = []

    def named(self, smiles):
        calls.append(smiles)
        if record:
            from orthonym.metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(record)
        return name

    monkeypatch.setattr(nm.Orthonym, "name", named)
    return calls


def test_component_is_named_once_per_scope_flags_and_context(monkeypatch, memo_scope):
    from orthonym.metrics.provenance import best_effort_ctx
    calls = _counting_name(monkeypatch)
    again = lambda **k: adducts._name_component("CCCO", "pin", general_fallback_unverified=True, **k)
    assert again() == again() == "propan-1-ol"
    assert len(calls) == 1                         # the second naming is a hit
    again(charged_ok=True)
    assert len(calls) == 2                         # other flags: another naming
    token = best_effort_ctx.set(True)
    try:
        again()
    finally:
        best_effort_ctx.reset(token)
    assert len(calls) == 3                         # another tier context: another naming


def test_component_memo_keeps_armed_and_disarmed_budgets_apart(monkeypatch, memo_scope):
    """A naming under disarmed hang budgets never trips, so it can give a name that a naming under
    armed budgets does not: the two are not each other's hits."""
    calls = _counting_name(monkeypatch)
    fn.enter_name_scope("X")
    try:
        again = lambda: adducts._name_component("CCCO", "pin", general_fallback_unverified=True)
        again(); again()
        assert len(calls) == 1
        fn.disarm_hang_budgets()
        again(); again()
        assert len(calls) == 2
    finally:
        fn.exit_name_scope()


def test_component_is_named_each_time_outside_a_scope(monkeypatch):
    calls = _counting_name(monkeypatch)
    for _ in range(2):
        assert adducts._name_component("CCCO", "pin", general_fallback_unverified=True) == "propan-1-ol"
    assert len(calls) == 2


def test_hit_replays_the_provenance_of_the_first_naming(monkeypatch, memo_scope):
    from orthonym.metrics import provenance as pv
    _counting_name(monkeypatch, record="propan-1-ol")
    pv.clear_provenance()
    adducts._name_component("CCCO", "pin", general_fallback_unverified=True)
    pv.clear_provenance()
    assert "propan-1-ol" not in pv.get_provenance()["non_pin_fragments"]
    adducts._name_component("CCCO", "pin", general_fallback_unverified=True)   # a hit
    assert "propan-1-ol" in pv.get_provenance()["non_pin_fragments"]


def test_trip_is_not_stored(monkeypatch, memo_scope):
    calls = []

    def trips(self, smiles):
        calls.append(smiles)
        raise fn.PerfBudgetExceeded()

    monkeypatch.setattr(nm.Orthonym, "name", trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue", lambda self, smiles: None)
    for _ in range(2):
        with pytest.raises(fn.PerfBudgetExceeded):
            adducts._name_component("CCCO", "pin", general_fallback_unverified=True)
    assert len(calls) >= 2


def test_spellings_and_readable_spellings_are_computed_once_per_scope(monkeypatch, memo_scope):
    import orthonym.assembly.t4_coverage as t4
    runs = []
    monkeypatch.setattr(nm.Orthonym, "_engine_recovery_spelling", lambda self, smiles: runs.append(smiles) or None)
    monkeypatch.setattr(t4, "_verified_universal_floor", lambda frag, smiles: None)
    flags = dict(general_fallback_unverified=True)
    assert adducts._component_spellings("CCO", "ethanol", "pin", flags) == []
    assert adducts._component_spellings("CCO", "ethanol", "pin", flags) == []
    assert runs == ["CCO"]
    assert adducts._component_spellings("CCO", "ethyl alcohol", "pin", flags) == []
    assert runs == ["CCO", "CCO"]


# --------------------------------------------------------------------------
# D a single compound that runs the hang budgets out at the best-effort ceilings
# --------------------------------------------------------------------------

def _best_effort_namer():
    from orthonym.cli import _emit_tier_flags
    return nm.Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def test_pin_ceiling_rescue_ships_a_name_that_reads_back(monkeypatch):
    from orthonym.assembly import budget_rescue
    seen = []
    monkeypatch.setattr(budget_rescue, "pin_ceiling_name",
                        lambda namer, smiles: seen.append(smiles) or "ethanol")
    assert _best_effort_namer()._try_pin_ceiling_rescue("CCO") == "ethanol"
    assert seen == ["CCO"]


@pytest.mark.parametrize("cand", ["methanol", "unknown organic compound", None])
def test_pin_ceiling_rescue_never_ships_another_molecule_or_a_failure(monkeypatch, cand):
    from orthonym.assembly import budget_rescue
    monkeypatch.setattr(budget_rescue, "pin_ceiling_name", lambda namer, smiles: cand)
    assert _best_effort_namer()._try_pin_ceiling_rescue("CCO") is None


def test_pin_ceiling_rescue_is_for_the_best_effort_tier_only(monkeypatch):
    from orthonym.assembly import budget_rescue
    monkeypatch.setattr(budget_rescue, "pin_ceiling_name", lambda namer, smiles: "ethanol")
    assert nm.Orthonym(style="pin")._try_pin_ceiling_rescue("CCO") is None


def test_pin_ceiling_name_is_labelled_below_the_pin(monkeypatch):
    from orthonym.assembly import budget_rescue
    from orthonym.metrics import provenance as pv
    monkeypatch.setattr(budget_rescue, "pin_ceiling_name", lambda namer, smiles: "ethanol")
    pv.clear_provenance()
    _best_effort_namer()._try_pin_ceiling_rescue("CCO")
    assert pv.get_provenance()["source"] == "general_engine"
    assert "ethanol" in pv.get_provenance()["non_pin_fragments"]


def test_boundary_offers_the_pin_ceiling_run_after_the_t4_rescue(monkeypatch):
    """The outermost name converts the trip of the main path into the offers of
    ``_try_perf_budget_t4_rescue`` and then ``_try_pin_ceiling_rescue``."""
    from orthonym.assembly import budget_rescue
    order = []

    def main_path_trips(self, smiles, *a, **k):
        raise fn.PerfBudgetExceeded()

    monkeypatch.setattr(nm.Orthonym, "_name_impl", main_path_trips)
    monkeypatch.setattr(nm.Orthonym, "_try_perf_budget_t4_rescue",
                        lambda self, smiles: order.append("t4") or None)
    monkeypatch.setattr(budget_rescue, "pin_ceiling_name",
                        lambda namer, smiles: order.append("pin") or "ethanol")
    assert _best_effort_namer().name("CCO") == "ethanol"
    assert order == ["t4", "pin"]


# Single PubChem 1M compounds (lane S, r045 r160 r417 r628): the main path and the rescue run the
# budgets out at the best-effort ceilings; under the PIN ceilings another producer names them.
BUDGET_FAMILY_SINGLES = [
    "CC(C)C(C(=O)N1C2CC2CC1C3=NC4=C(N3)C=C(C=C4)C5=C6CC7CCC7C6=C(C=C5)C8=CC=C(C=C8)C9=CN=C(N9)C1CC2CC2N1C(=O)C(C(C)C)NC(=O)OC)NC(=O)OC",
    "CC(C)[C@@H](C(=O)N1CCC[C@H]1C2=NC3=CCC(C=C3N2)C4CC[C@@H](N4C5=CC=C(C=C5)C67CC8CC(C6)CC(C8)C7)C9=CC1=C(C=C9)N=C(N1)[C@@H]1CCCN1C(=O)[C@H](C(C)C)NC(=O)OC)NC(=O)OC",
    "[2H]C1=CC(=C2C(=C1[2H])C3=C(C(=C(C(=C3N2C4=CC=CC=C4C5=NC(=NC(=N5)C6=CC7=C(C=C6)N=CC=C7)C8=CC=CC(=C8)C9=CC=CC1=C9OC2=C1C=C(C=C2)C1=CC=CC2=C1OC1=CC=CC=C21)[2H])[2H])[2H])[2H])[2H]",
    "CC(C)[C@@H](C(=O)N1CC2(CC2)C[C@H]1C3=NC=C(N3)C4=CC5=C(C=C4)C6=C(O5)C=C(C=C6)C7=CC8=C(C=C7)N=C(N8)C9[C@H]1CC[C@H](C1)N9C(=O)[C@H](C(C)C)NC(=O)OC)NC(=O)OC",
]


@pytest.mark.parametrize("smiles", BUDGET_FAMILY_SINGLES)
def test_single_compound_that_runs_its_budgets_out_is_named(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name and res.get("tier") == "systematic_verified", res.get("tier")
