"""Suite fix j12-verify-fixes (TRIAGE.md 'Suite fix -- j12-verify-fixes'): rows with
teeth for the findings of the cross-family verification panel. Names are asserted
exactly (gate on, PIN tier unless stated) and each name is checked by an independent
full-InChIKey OPSIN round trip (tests/support/rt_assert.py, not the engine's gate).
"""
import pytest

from orthonym.namer import Orthonym
from tests.support.rt_assert import name_is_rt_exact


# ---------------------------------------------------------------------------
# Finding 6: organometallic compounds of the Group 1-12 metals carry no PIN.
# INTRODUCTION (the Blue Book): "neither preferred IUPAC names or
# preselected names (see for organometallic compounds involving the
# transition elements (including the Group 3 elements) and Groups 1 and 2
# elements, except for 'ocene' compounds, are noted." (:40157) gives
# 'methyllithium' and 'methylmagnesium iodide' without '(PIN)'. The names ship
# unchanged, labelled below pin_verified; the ocenes,:40129), the
# Group-14 substitutive PINs, 'tetraethylplumbane (PIN)') and a salt of
# an organic ion (no metal-carbon bond) keep the PIN label.
# ---------------------------------------------------------------------------

NO_PIN_ORGANOMETALLICS = [
    ("CC(C)(C)[Li]", "tert-butyllithium"),
    ("CCCC[Li]", "butyllithium"),
    ("C[Mg]Br", "methylmagnesium bromide"),
    ("CC[Zn]CC", "diethylzinc"),
]

PIN_KEPT = [
    ("[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]", "ferrocene"),
    ("CC[Pb](CC)(CC)CC", "tetraethylplumbane"),
    ("CC(=O)[O-].[Na+]", "sodium acetate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", NO_PIN_ORGANOMETALLICS)
def test_group_1_to_12_organometallic_is_not_labelled_pin(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "systematic_verified", r
    assert r["is_pin"] is False, r
    assert name_is_rt_exact(expected, smiles), expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PIN_KEPT)
def test_ocene_group14_and_salt_keep_the_pin_label(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected


# ---------------------------------------------------------------------------
# Finding 1: a transient OPSIN outage ('unavailable') must not change the gate's
# verdict on a grammar-gap carve-out (a class OPSIN 2.9.0 has no grammar for, so a
# healthy call can only reject it and the name ships on its construction alone),
# while every name that needs an OPSIN observation still fails closed: an ordinary
# name, the organometallic additive carve-out (OPSIN parses some of its names), and
# any carve-out at the general-fallback tiers except the exact NP stereoparents.
# Gate level, both oracle probes stubbed (the state an OPSIN timeout leaves).
# ---------------------------------------------------------------------------

GRAMMAR_GAP_NAMES = [
    # 'methane-SO-thioperoxol (PIN)' (thioperoxol carve-out)
    ("methane-SO-thioperoxol", "CSO", "thioperoxol"),
    # inositol retained name (inositol carve-out)
    ("scyllo-inositol", "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
     "inositol"),
    # method (1) polyol polyester (regex carve-out)
    ("propane-1,2,3-triyl 1,3-diacetate 2-propanoate",
     "CC(=O)OCC(COC(=O)C)OC(=O)CC", "polyol_polyester"),
]


def _gate(monkeypatch, status, name, smiles, general_fallback_tier=False):
    from orthonym import namer
    from orthonym.metrics import provenance as pv
    monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
    monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda n: None)
    monkeypatch.setattr(namer, "_validity_gate_status", lambda n: status)
    pv.clear_provenance()
    out = namer._final_opsin_validity_gate(
        name, smiles, {}, general_fallback_tier=general_fallback_tier)
    return out, pv.get_provenance()["gate_outcome"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("name,smiles,slug", GRAMMAR_GAP_NAMES)
def test_grammar_gap_carveout_verdict_does_not_depend_on_opsin_reachability(
        monkeypatch, name, smiles, slug):
    from orthonym.metrics import provenance as pv
    rejected = _gate(monkeypatch, "rejected", name, smiles)
    unavailable = _gate(monkeypatch, "unavailable", name, smiles)
    assert rejected == (name, pv.carveout_outcome(slug)), rejected
    assert unavailable == rejected, (unavailable, rejected)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("name,smiles", [
    ("ethanol", "CCO"),
    ("trichlorido(methyl)titanium", "Cl[Ti](Cl)(Cl)C"),
])
def test_names_needing_an_opsin_observation_still_fail_closed(monkeypatch, name, smiles):
    from orthonym.metrics import provenance as pv
    from orthonym.namer import _descriptive_fallback
    out, outcome = _gate(monkeypatch, "unavailable", name, smiles)
    assert out == _descriptive_fallback(smiles), out
    assert outcome == pv.GATE_OUTCOME_SUPPRESSED, outcome


@pytest.mark.opsin_gate
def test_general_fallback_tier_keeps_suppressing_grammar_gap_names(monkeypatch):
    from orthonym.namer import _descriptive_fallback
    for status in ("rejected", "unavailable"):
        out, _ = _gate(monkeypatch, status, "methane-SO-thioperoxol", "CSO",
                       general_fallback_tier=True)
        assert out == _descriptive_fallback("CSO"), (status, out)


# ---------------------------------------------------------------------------
# Findings 2/4/7: a substituted acetic acid keeps the retained parent when its
# stereocentre is ON the 2-carbon chain; the alpha descriptor is cited bare.
# (the Blue Book,:29725) "only acetic acid, benzoic acid, and
# oxamic acid can be substituted", 'acetic acid (PIN) ethanoic acid';
# (:3031) locants omitted; (:44643) a descriptor takes a locant "when such
# locants are present"; '(R)-{bis[(1R)-1-hydroxyethyl]amino}{...}acetic acid (PIN)'
# (:45731), '(S)-cyclopropyl(hydroxy)acetaldehyde (PIN)' (:45259).
# (:7304) second and later simple substituents enclosed, 'bromo(chloro)acetic acid
# (PIN)' (:7312). The amido prefix is built on the retained 'acetamide', which
# REQUIRES its locants:32995;:7304 "Locants are
# required... for example acetamide"; '2-phenylacetamide':33364).
# ---------------------------------------------------------------------------

ACETIC_PINS = [
    ("O[C@@H](c1ccccc1)C(=O)O", "(S)-hydroxy(phenyl)acetic acid"),
    ("N[C@@H](c1ccccc1)C(=O)O", "(S)-amino(phenyl)acetic acid"),
    ("N[C@@H](C1CC1)C(=O)O", "(S)-amino(cyclopropyl)acetic acid"),
    ("Cl[C@H](Br)C(=O)O", "(R)-bromo(chloro)acetic acid"),
    ("ClC(Br)C(=O)O", "bromo(chloro)acetic acid"),
    ("COC(=O)[C@@H](O)c1ccccc1", "methyl (S)-hydroxy(phenyl)acetate"),
    ("OC(c1ccccc1)C(=O)NCC(=O)O", "(2-hydroxy-2-phenylacetamido)acetic acid"),
    ("O[C@@H](c1ccccc1)C(=O)NCC(=O)O",
     "[(2S)-2-hydroxy-2-phenylacetamido]acetic acid"),
    ("N[C@H](C1CC1)C(=O)NCC(=O)O",
     "[(2R)-2-amino-2-cyclopropylacetamido]acetic acid"),
    # a second prefix that merely starts with 'octa'/'deca' is one substituent,
    # not a multiplied one (the multiplier is read off the locants)
    ("CCCCCCCC(=O)N[C@@H](O)C(=O)[O-]", "(S)-hydroxy(octanamido)acetate"),
    ("CCCCCCCC(=O)NC(O)C(=O)[O-]", "hydroxy(octanamido)acetate"),
    ("CCCCCCCCCC(=O)NC(O)C(=O)O", "decanamido(hydroxy)acetic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ACETIC_PINS)
def test_substituted_acetic_acid_pin(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected


@pytest.mark.parametrize("acid,amido", [
    # a recorded located spelling is used; the fold records it when it builds
    # the retained name, so the acid is named first
    ("OC(=O)[C@@H](O)c1ccccc1", "(2S)-2-hydroxy-2-phenylacetamido"),
    ("OC(=O)C(O)c1ccccc1", "2-hydroxy-2-phenylacetamido"),
    ("OC(=O)C(Cl)Br", "2-bromo-2-chloroacetamido"),
    ("OC(=O)C(OCCO)OCCO", "2,2-bis(2-hydroxyethoxy)acetamido"),
    # single prefixes keep the '2-' insertion
    ("OC(=O)CCl", "2-chloroacetamido"),
    ("OC(=O)C(F)(F)F", "2,2,2-trifluoroacetamido"),
    ("OC(=O)CNC(=O)[C@@H](N)C(C)C", "2-[(2S)-2-amino-3-methylbutanamido]acetamido"),
])
def test_acetic_amido_prefix_keeps_the_locants(acid, amido):
    from orthonym.assembly.fragment_naming import name_fragment_recursively
    from orthonym.assembly.substituent_naming import acid_name_to_amido_prefix
    assert acid_name_to_amido_prefix(name_fragment_recursively(acid)) == amido


@pytest.mark.parametrize("acid,amido", [
    ("(2S)-2-hydroxy-2-phenylethanoic acid", "(2S)-2-hydroxy-2-phenylacetamido"),
    ("2-phenylethanoic acid", "2-phenylacetamido"),
    ("ethanoic acid", "acetamido"),
    # never a guess: an unrecorded bare-descriptor / multi-prefix retained name
    ("(S)-foo(bar)acetic acid", None),
    ("foo(bar)acetic acid", None),
    ("oxydiethanoic acid", None),
])
def test_acetic_amido_prefix_from_names(acid, amido):
    from orthonym.assembly.substituent_naming import acid_name_to_amido_prefix
    assert acid_name_to_amido_prefix(acid) == amido


def test_prefix_from_a_non_pin_acid_spelling_is_labelled_unless_it_is_acetamido():
    """record_prefix_from_acid_status: an acyl prefix still carries the systematic
    'ethanoyl' of a '...ethanoic acid' spelling (non-PIN,:29725), the
    amido prefix is the retained '...acetamido' and is not labelled."""
    from orthonym.assembly.substituent_naming import record_prefix_from_acid_status
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()
    record_prefix_from_acid_status("2-(ethenyloxy)ethanoic acid", "(ethenyloxy)ethanoyl")
    record_prefix_from_acid_status("2-phenylethanoic acid", "2-phenylacetamido")
    frags = pv.get_provenance()["non_pin_fragments"]
    assert "(ethenyloxy)ethanoyl" in frags
    assert "2-phenylacetamido" not in frags


# ---------------------------------------------------------------------------
# Finding 8: enclosing marks continue the "{[({})]}" order,
# the Blue Book) from a group escalated by (:7509, "consecutive
# enclosing marks of the same level, the next level of enclosing mark is used",
# '(3S)-2-[(2S)-2-{[(2S)-1-ethoxy-1-oxo-4-phenylbutan-2-yl]amino}propanoyl]-...',:7536).
# The group around '[(2S)-2-{...}propanamido]' is '{', not '[' again.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content,wrapped", [
    ("2-methylpropyl", "(2-methylpropyl)"),
    ("(R)-butan-2-yl", "[(R)-butan-2-yl]"),
    # the escalation itself: '(' would stand next to '(2S)'
    ("(2S)-2-{2-[(2R)-2-aminopropanamido]acetamido}propanamido",
     "[(2S)-2-{2-[(2R)-2-aminopropanamido]acetamido}propanamido]"),
    # the level above an escalated '[' is '{'
    ("(2S)-2-[(2S)-2-{2-[(2R)-2-aminopropanamido]acetamido}propanamido]"
     "-3-methylbutanamido",
     "{(2S)-2-[(2S)-2-{2-[(2R)-2-aminopropanamido]acetamido}propanamido]"
     "-3-methylbutanamido}"),
    # four raw levels without escalation keep the plain order
    ("2-fluoro-4-({6-[methyl(prop-2-en-1-yl)amino]hexyl}oxy)phenyl",
     "[2-fluoro-4-({6-[methyl(prop-2-en-1-yl)amino]hexyl}oxy)phenyl]"),
    # an integral bracket the scan does not strip (von Baeyer superscripts) is
    # counted at its natural level, not read as an escalation
    ("(23S,24R)-4,24-dimethyl-25,26-diazahexacyclo[16.6.1.1^3,6.1^8,11.1^13,16"
     ".0^19,24]octacos-1-en-9-yl",
     "[(23S,24R)-4,24-dimethyl-25,26-diazahexacyclo[16.6.1.1^3,6.1^8,11.1^13,16"
     ".0^19,24]octacos-1-en-9-yl]"),
])
def test_enclosing_mark_continues_the_order_after_an_escalation(content, wrapped):
    from orthonym.assembly.naming_utils import apply_enclosing_marks
    assert apply_enclosing_marks(content, -1) == wrapped


@pytest.mark.opsin_gate
def test_pentapeptide_marks_follow_p16_5_4():
    smiles = ("N[C@H](C1CC1)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C(C)C)C(=O)"
              "N[C@@H](CC(C)C)C(=O)O")
    expected = ("(2S)-2-{(2S)-2-[(2S)-2-{2-[(2R)-2-amino-2-cyclopropylacetamido]"
                "acetamido}propanamido]-3-methylbutanamido}-4-methylpentanoic acid")
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected


# ---------------------------------------------------------------------------
# Finding 5: a ring ketone on a parent with ONE indicated hydrogen takes that
# hydrogen at the ketone, and the hydro prefixes cite the other saturated
# positions. (the Blue Book) "When there are an equal number
# of indicated hydrogen atoms and principal characteristic groups..., the
# indicated hydrogen atoms are placed at peripheral atoms that will accommodate
# these principal characteristic groups... Locants for hydro prefixes are those
# of the saturated positions"; '2-(1,3,4,5-tetrahydro-2H-2-benzazepin-2-yl)
# ethan-1-ol (PIN)' (:24774), '7H-1-benzopyran-7-one (PIN)' (:24776), '3-imino-
# 2,3-dihydro-1H-isoindol-1-one (PIN)' (:29609). With fewer indicated hydrogens
# than suffixes (:24808) keeps the lowest one (the controls).
# ---------------------------------------------------------------------------

RING_KETONE_PINS = [
    ("O=C1Cc2ccccc2N1", "1,3-dihydro-2H-indol-2-one"),
    ("O=C1Cc2ccccc2C1", "1,3-dihydro-2H-inden-2-one"),
    ("CN1C(=O)Cc2ccccc21", "1-methyl-1,3-dihydro-2H-indol-2-one"),
    ("Cc1ccc2c(c1)CC(=O)N2", "5-methyl-1,3-dihydro-2H-indol-2-one"),
    ("O=C1Cc2cccnc2N1", "1,3-dihydro-2H-pyrrolo[2,3-b]pyridin-2-one"),
    # unchanged controls
    ("O=C1CNc2ccccc21", "1,2-dihydro-3H-indol-3-one"),
    ("O=C1NCc2ccccc21", "2,3-dihydro-1H-isoindol-1-one"),
    ("O=C1CCc2ccccc21", "2,3-dihydro-1H-inden-1-one"),
    ("O=C1CC(=O)c2ccccc21", "1H-indene-1,3(2H)-dione"),
    ("Cn1cnc2c1c(=O)n(c(=O)n2C)C", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", RING_KETONE_PINS)
def test_ring_ketone_indicated_hydrogen_at_the_suffix(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected


@pytest.mark.parametrize("prefixes,expected", [
    (["2-hydroxy", "2-octanamido"], "hydroxy(octanamido)acetic acid"),
    (["2-hydroxy", "2,2-diphenyl"], "hydroxydi(phenyl)acetic acid"),   # BB:29852
    (["2-bromo", "2-chloro"], "bromo(chloro)acetic acid"),             # BB:7312
    (["2-hydroxy", "2-heptanoyl"], "hydroxy(heptanoyl)acetic acid"),
    (["2-hydroxy", "2,2-bis(2-hydroxyethoxy)"], "hydroxybis(2-hydroxyethoxy)acetic acid"),
])
def test_retained_acetic_multiplier_is_read_off_the_locants(prefixes, expected):
    from orthonym.assembly.composition_primitives import retained_acetic_from_prefixes
    assert retained_acetic_from_prefixes(prefixes, enclose_subsequent=True) == expected
