""" composed-charge lever — control acceptance tripwire .

These two round-trip TODAY (verified) and must never regress across the
composed-charge workstreams. Do NOT edit the expected values to match a future
regression — that defeats the tripwire; if one of these breaks, STOP and report
the actual output so the failure gets diagnosed, not silenced.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import assert_full_rt

BE = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)

CONTROLS = [
    ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),   # betaine — Z3 prefix already works
    ("C(C(=O)[O-])[NH3+]", "glycine"),                           # amino-acid retained path pre-empts
]


@pytest.mark.parametrize("smi,expected", CONTROLS)
def test_composed_charge_controls_unchanged(smi, expected):
    assert Orthonym(**BE).name(smi) == expected


# --------------------------------------------------------------------------
# review a performance pass — synthetic branch coverage for v34_charged_probe.probe's
# RT classifier. The real charged corpus backlog measured B=0 (no wrong/
# unparseable rows), so the `wrong` and `unparseable` branches — and the
# falsy-InChIKey fail-closed guard the Critical finding was about — are
# otherwise never exercised by any test. `v34_charged_probe.py` lives outside
# the `orthonym` package (it is a measurement script, not shipped code). It
# used to sit in internal notes, which the public-release commit
# 0245191d1 untracked, so collection failed with ModuleNotFoundError; since
# 2026-09-03 the test's own copy lives in `tests/support/` and is imported via
# an explicit sys.path insert.
# --------------------------------------------------------------------------
_PROBE_DIR = str(Path(__file__).resolve().parents[2] / "support")
if _PROBE_DIR not in sys.path:
    sys.path.insert(0, _PROBE_DIR)
import v34_charged_probe as v34_probe  # noqa: E402 (path must be inserted first)


def test_probe_classifies_wrong_molecule_name_as_wrong(monkeypatch):
    """The namer emits a real, OPSIN-parseable name for the WRONG molecule.

    Input is ethanol (``CCO``); the (monkeypatched) namer lies and returns
    ``"methane"``, which OPSIN parses to ``C`` — a different, genuine molecule.
    Must classify ``wrong``, never ``full``/``block1``.
    """
    monkeypatch.setattr(Orthonym, "name", lambda self, smi: "methane")
    result = v34_probe.probe(["CCO"])
    assert result["CCO"]["rt"] == "wrong"
    assert result["CCO"]["name"] == "methane"


def test_probe_classifies_opsin_unparseable_name_as_unparseable(monkeypatch):
    """The namer emits a string OPSIN cannot parse into any structure at all."""
    monkeypatch.setattr(Orthonym, "name", lambda self, smi: "zzzznotarealchemicalnamexyz")
    result = v34_probe.probe(["CCO"])
    assert result["CCO"]["rt"] == "unparseable"
    assert result["CCO"]["name"] == "zzzznotarealchemicalnamexyz"


def test_probe_wildcard_input_never_classified_full(monkeypatch):
    """Exact regression test for the Critical finding (review a performance pass).

    ``Chem.MolToInchiKey(Chem.MolFromSmiles('*CC'))`` returns ``''`` — falsy,
    NOT an exception, NOT ``None`` — for any wildcard/dummy-atom structure. The
    reported false positive needs BOTH sides to key to ``''``: the input
    (``*CC``) is wildcard-bearing, and so — after monkeypatching OPSIN's parse
    — is the "round-tripped" structure. Pre-fix this hit ``have == want`` as
    ``'' == ''`` and returned ``full`` (reproduced directly against the OLD
    branch logic before writing this test — see the fix-report note).
    Post-fix, ``not have`` fires first and this must classify ``unparseable``,
    never ``full``.
    """
    import orthonym.validation.opsin_roundtrip as opsin_roundtrip_mod

    monkeypatch.setattr(Orthonym, "name", lambda self, smi: "some-name-for-a-wildcard-structure")
    monkeypatch.setattr(opsin_roundtrip_mod, "opsin_parse", lambda name, jar_version="2.9.0": "*CC")
    result = v34_probe.probe(["*CC"])
    assert result["*CC"]["rt"] != "full"
    assert result["*CC"]["rt"] == "unparseable"


def test_probe_falsy_want_with_genuine_have_classified_wrong_not_full(monkeypatch):
    """Companion case: input has a falsy key (wildcard) but the emitted name
    round-trips to a GENUINE, different molecule (``have`` truthy). Per the
    fix, ``not want`` (with ``have`` truthy) must fail closed to ``wrong`` —
    never ``full``/``block1`` — since there is nothing genuine on the input
    side to compare against.
    """
    monkeypatch.setattr(Orthonym, "name", lambda self, smi: "ethane")
    result = v34_probe.probe(["*CC"])
    assert result["*CC"]["rt"] != "full"
    assert result["*CC"]["rt"] == "wrong"


# --------------------------------------------------------------------------
# + (merged, 2026-08-22) — carboxylate/polyacid junior-prefix
# multiplicity fix.
#
# Diagnosis (verified, NOT the brief's original hypothesis): the "clean-PIN
# acid gate" in `_neutralize_carboxylate_to_acid` (ions.py) already carries a
# best-effort re-entry, and the multi-anion backbone namer
# (`_try_neutralize_and_name`) already inherits best-effort from
# `general_fallback_ctx` -- for every backlog witness probed, the BACKBONE
# itself already named correctly under best-effort. The actual blocker was
# ONE STEP LATER, in `_apply_anionic_substituent_prefixes`, "cite a
# JUNIOR anionic centre by its anionic prefix"): it computed
# `junior_carb = n_carb - 1`, silently assuming the parent suffix ALWAYS
# consumes exactly one carboxylate. A genuine POLYACID parent
# ('-dioate'/'-tricarboxylate'/...) consumes 2, 3,... -- so a real
# tricarboxylate anion (a '-dioate' parent + ONE junior 'carboxy' arm) was
# undercounted as TWO junior sites, tripped the (deliberately fail-closed)
# `total_junior != 1` guard, and shipped the junior COOH as the NEUTRAL
# 'carboxy' prefix -- a charge-dropping name for a DIFFERENT (less-anionic)
# molecule that correctly suppressed to an avoidable abstention.
#
# Fix: `_parent_acid_suffix_multiplicity` reads the multiplying prefix
# already embedded in the converted suffix ('-dioate' -> 2, '-tricarboxylate'
# -> 3, plain '-oate'/'-carboxylate' -> 1, an unmatched/retained suffix -> 1
# unchanged) so the junior count reflects only what the parent did NOT
# already claim; the exactly-1-junior scope is then widened to ANY junior
# count PROVIDED every junior site is the SAME class (a genuinely mixed set,
# e.g. one junior carboxylate riding alongside one junior alkoxide, still
# fails closed -- converting two different substituent words risks
# re-ordering the alphabetized prefix list).
#
# Witness (backlog, bucket A, pulled from
# internal notes): a tricarboxylate anion
# (net -3, no cation) whose backbone is a heptanedioate chain decorated with
# an amide-linked succinyl arm -- exactly the "carboxylate anion whose
# backbone is decorated" shape the brief asked for. Re-derived with
# `v34_charged_probe.probe`: rt flips abstain -> full, 0 new wrong/
# unparseable elsewhere (see the RT-safety sweep, ws1-sweep.json).
# --------------------------------------------------------------------------

_WS1_BE = dict(general_fallback=True, general_fallback_unverified=True,
               allow_aromatic_general=True)


def _full_ik_rt(smi, name):
    """OPSIN-parse ``name`` and require the FULL InChIKey (constitution +
    charge + stereo) to match ``smi`` -- the 0-wrong contract for a shipped
    name. Mirrors ``test_charged_completion.py``'s helper of the same name."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    assert name and "unknown" not in name, f"expected a real name, got {name!r}"
    got = opsin_parse(name)
    assert got, f"OPSIN could not parse {name!r}"
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(got)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi)), \
        f"full-InChIKey RT mismatch for {name!r} vs input {smi!r}"


def test_ws1_tricarboxylate_amide_junior_prefix_converts():
    """WS0 bucket-A witness (dev500): a carboxylate TRIANION whose parent
    chain (a heptanedioate) cannot express all three -COO- in its suffix, so
    one rides as a JUNIOR 'carboxylato' prefix. Pre-fix this abstained
    (SELF-01 suppressed the charge-dropping 'carboxy' spelling); post-fix it
    must emit and round-trip FULL. VERIFIED via py2opsin (see module docstring
    above) — do not hand-edit this expected string without re-deriving it."""
    smi = "O=C([O-])CCC(=O)N[C@@H](CCCC(=O)C(=O)[O-])C(=O)[O-]"
    name = Orthonym(**_WS1_BE).name(smi)
    assert name == "(2S)-2-[(3-carboxylato-1-oxopropyl)amino]-6-oxoheptanedioate"
    _full_ik_rt(smi, name)


def test_parent_acid_suffix_multiplicity_reads_the_embedded_multiplier():
    """Unit-level: ``_parent_acid_suffix_multiplicity`` must read the SAME
    multiplying-prefix word ``_acid_to_oate``/``_acid_name_to_carboxylate``
    already emit, not re-derive it from scratch."""
    from orthonym.rules.ions import _parent_acid_suffix_multiplicity as mult
    assert mult("propanoate") == 1
    assert mult("benzoate") == 1
    assert mult("hexanedioate") == 2
    assert mult("benzene-1,2-dicarboxylate") == 2
    assert mult("pentane-1,2,4,5-tetracarboxylate") == 4
    # Retained/irregular suffixes this producer does not model default to 1
    # (the ORIGINAL, pre- assumption) rather than guessing.
    assert mult("acetate") == 1
    assert mult("formate") == 1


def test_ws1_homogeneous_two_junior_carboxylates_convert():
    """Unit-level generalization proof: TWO junior carboxylate sites (a
    pentanedioate parent -- multiplicity 2 -- plus a direct 'carboxy' AND a
    non-identical '(2-carboxyethyl)' junior arm, so no 'bis(...)' grouping
    hides the count) both convert to 'carboxylato'. Built directly against
    ``_apply_anionic_substituent_prefixes`` (the producer) on a REAL,
    fully-ionized tetracarboxylate mol; the composed name is OPSIN-RT-verified
    against the exact structure (see impl report) rather than asserted blind.
    """
    from orthonym.rules.ions import (
        _apply_anionic_substituent_prefixes, _acid_name_to_carboxylate,
    )
    from orthonym.perception.ions import get_ion_sites

    smi = "[O-]C(=O)C(C(=O)[O-])CC(CCC(=O)[O-])C(=O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    sites = get_ion_sites(mol)
    anions = sites["anions"]
    assert len(anions) == 4

    acid_name = "2-carboxy-4-(2-carboxyethyl)pentanedioic acid"
    converted = _acid_name_to_carboxylate(acid_name, 4)
    out = _apply_anionic_substituent_prefixes(mol, anions, converted)
    expected = "2-carboxylato-4-(2-carboxylatoethyl)pentanedioate"
    assert out == expected
    _full_ik_rt(smi, out)


def test_ws1_mixed_junior_classes_still_fail_closed():
    """Regression guard: a genuinely MIXED junior set (one junior carboxylate
    AND one junior alkoxide on the SAME pentanedioate parent) must still
    decline -- the widening covers ONLY a homogeneous multi-junior set,
    never a mixed one (prefix re-ordering risk, per the function's own
    docstring).

    Declining means '' (j7, TRIAGE g2 G2-C7): the old "return the name
    unchanged" shipped the neutral prefixes, and '3-carboxy-3-hydroxy-
    pentanedioate' denotes the DIANION (OPSIN 2.9.0 full InChIKey differs from
    this tetra-anion's), a different species, BB:41197: the junior
    anionic centres "expressed as anionic substituent group(s)"). The full
    pipeline still names the molecule '2-oxidopropane-1,2,3-tricarboxylate'
    (RT exact) through route_charged."""
    from orthonym.rules.ions import _apply_anionic_substituent_prefixes
    from orthonym.perception.ions import get_ion_sites

    smi = "[O-]C(=O)CC(C(=O)[O-])([O-])CC(=O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    sites = get_ion_sites(mol)
    anions = sites["anions"]
    assert len(anions) == 4  # 3 carboxylate + 1 alkoxide

    name_in = "3-carboxy-3-hydroxypentanedioate"
    out = _apply_anionic_substituent_prefixes(mol, anions, name_in)
    assert out == ""  # fail-closed: never the charge-dropping neutral-prefix name


# --------------------------------------------------------------------------
# (2026-08-22) — the coverage-FLOOR no-abstain backstop for the charged
# residual.
#
# reachability trace (ws7-reachability.json) confirmed the universal floor
# (`assembly/universal_substituent.py`) is REACHED for 96/97 in-scope charged
# bucket-A abstains and `_resolve_spine_charge` VOIDED for 87 of them.
#
# Root cause (empirically established, NOT the brief's suffix hypothesis): the
# floor names by SKELETAL REPLACEMENT, threading a carboxylate/sulfonate/
# phosphonate carbonyl `=O` INTO the parent skeleton, which leaves the FG
# anion's `[O-]` as a bare, single-bonded TERMINAL branch (structurally an
# alkoxide O). So every FG anion the floor meets decomposes to a lone charged
# terminal atom, and the ONE fix for the whole class is `_charged_leaf_shortcut`
# -- a CHARGED substituent prefix (`oxido`/`sulfido`/`azaniumyl`) carrying the
# charge -- plus `_charge_suffix_text(parent_only=True)` so a SKELETAL cation in
# a zwitterion still gets its `-ium` while the paired anion rides as a leaf. All
# floor emissions are still gated by the caller's / verify_or_none net
# (0-wrong preserved: a stereo-CONFLICT or unverifiable name is suppressed).
#
# Witnesses below are REAL bucket-A rows; each expected string was OPSIN-RT
# re-derived (`_full_ik_rt`), never hand-edited.
# --------------------------------------------------------------------------

_WS7_BE = dict(general_fallback=True, general_fallback_unverified=True,
               allow_aromatic_general=True)


def _floor_name_and_verify(smi):
    """Name ``smi`` via the coverage FLOOR directly and 0-wrong-gate it through
    ``verify_or_none`` -- the SAME oracle the live pipeline applies to a floor
    emission, but deterministic and independent of the in-process OPSIN-validity
    gate's warm-up state (which is unreliable under pytest -- a documented
    project OPSIN/JVM harness hazard, not a code defect: the end-to-end
    fresh-process behaviour is proven by ``ws7-sweep.json``). Returns
    ``(name, verified_name_or_None)``."""
    from orthonym.assembly.universal_substituent import (
        name_universal_substitutive)
    from orthonym.validation.reconstruct import verify_or_none
    r = name_universal_substitutive(Chem.MolFromSmiles(smi))
    name = r.name if r is not None else None
    return name, (verify_or_none(name, smi) if name else None)


def test_ws7_net_anion_dithiocarbamate_carboxylate_names_and_full_rt():
    """ bucket-A (pubchem500): a net-1 carboxylate whose backbone also bears
    a dithiocarbamate. Pre- the floor VOIDED on the carboxylate `[O-]`;
    post- the charged-leaf `oxido` names it and ``verify_or_none`` CONFIRMs it
    on the FULL InChIKey (constitution + charge; no stereo to omit)."""
    smi = "CN(C)C(=S)SCCC(=O)[O-]"
    name, verified = _floor_name_and_verify(smi)
    # D3: plain C1 methyl -> retained 'methyl', not 'methan-1-yl'
    # (RT-identical InChIKey; pure spelling).
    assert name == ("7-methyl-2-oxido-6-sulfanylidene-"
                    "1-oxa-5-thia-7-azaoct-1-ene")
    assert verified == name


def test_ws7_diglycine_zwitterion_skeletal_ium_plus_oxido_leaves_full_rt():
    """ bucket-A (chebi500): a net-1 zwitterion -- two carboxylate anions and
    an internal (degree-2, SKELETAL) ammonium. The paired anions ride as `oxido`
    leaves while the skeletal `[NH2+]` gets its `-ium` suffix via
    `_charge_suffix_text(parent_only=True)` (the both-signs global gate no longer
    blocks a per-spine cation). ``verify_or_none`` CONFIRMs FULL."""
    smi = "O=C([O-])CNCC[NH2+]CC(=O)[O-]"
    name, verified = _floor_name_and_verify(smi)
    assert name == "2,9-dioxido-1,10-dioxa-4,7-diazadeca-1,9-dien-7-ium"
    assert verified == name


def test_ws7_net_dianion_triacetate_amine_names_and_full_rt():
    """ bucket-A (chebi500): a net-2 species (three carboxylates + one
    internal ammonium). Exercises multi-`oxido` + a recursively-named charged
    branch arm. ``verify_or_none`` CONFIRMs FULL."""
    smi = "O=C([O-])CNCC[NH+](CC(=O)[O-])CC(=O)[O-]"
    name, verified = _floor_name_and_verify(smi)
    assert name == ("2,9-dioxido-4-(2-oxido-3-oxaprop-2-en-1-yl)-"
                    "1,10-dioxa-4,7-diazadeca-1,9-dien-4-ium")
    assert verified == name


def test_ws7_stereo_carboxylate_now_ships_full_stereo_ws_stereo_win():
    """ bucket-A (a dev split): a STEREO carboxylate, all 5 defined stereocentres
    on the polycyclic RING SPINE itself. Pre-STEREO the floor emitted a
    constitution-only name and this test locked ``verified is None`` (a safe
    stereo-OMISSION superset, per the pipeline's `_rt_match` gate). STEREO
    now prepends the ring spine's OWN ``_stereo_prefix`` (mirroring
    general_engine's four parent engines) and 0-wrong-gates it through
    ``verify_or_none`` before shipping -- for THIS witness every stereocentre is
    on the spine (none buried in an off-spine branch), so the with-stereo
    candidate FULL-RT-verifies and block1 -> full. Re-verified independently via
    OPSIN's own parse-back + full-InChIKey compare (not just re-run of the
    generator): reverting the STEREO splice (``_stereo_prefix`` forced to
    ``''``) reproduces the exact former value (name without the ``(...)-``
    block, ``verified is None``) -- see task-STEREO-report.md's mutation
    check. Do NOT revert this value without re-deriving it the same way."""
    smi = ("C=C[C@]1(C)CC[C@@H]2C(=CC[C@@H]3[C@]2(C)CCC[C@]3(C)"
           "C(=O)[O-])C1")
    name, verified = _floor_name_and_verify(smi)
    # D3: plain C1 methyls -> retained 'trimethyl', not 'tri(methan-1-yl)'.
    assert name == ("(1R,2R,5R,10R,11S)-5-(eth-1-en-1-yl)-1,5,11-trimethyl-"
                     "11-(1-oxido-2-oxaeth-1-en-1-yl)tricyclo[8.4.0.0^2,7]tetradec-7-ene")
    assert verified == name  # full-InChIKey CONFIRMED (constitution+stereo+charge)
    got = opsin_parse(name)
    assert got, f"OPSIN could not parse {name!r}"
    # full isomeric match now (not merely a flat/charged superset)
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(got)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def test_ws_stereo_branch_buried_stereocentre_still_omits_not_wrong():
    """0-WRONG GUARD (task brief, CRITICAL): the sec-butyl substituent's own
    stereocentre is BRANCH-internal (off the ring spine), which this producer's
    STEREO splice intentionally does not capture (scope: top-level SPINE
    stereo only, mirroring general_engine's own parent-scope-only split). The
    ring spine's OWN stereocentre (position 5) DOES have a computable CIP
    ('(5S)-') -- so a naive splice would ship a name that specifies ONLY the
    ring stereocentre while leaving the branch one silently undefined, which
    would OPSIN-parse to a DIFFERENT (partially-specified) molecule than the
    input. The verify-or-fallback gate catches exactly this: the with-stereo
    candidate fails full-RT verification and the module falls back to the
    plain, stereo-omitted name -- never shipping the wrong/partial descriptor.
    """
    smi = "CC[C@H](C)[C@@H]1NC(=O)[C-](C(C)=O)C1=O"
    from orthonym.assembly.universal_substituent import (
        _build_ctx, _name_component)
    from orthonym.assembly.general_engine import _stereo_prefix
    mol = Chem.MolFromSmiles(smi)
    ctx, heavy = _build_ctx(mol, 20_000)
    comp = _name_component(ctx, heavy, attach_hint=None, is_top=True)
    # Sanity: the ring spine DOES carry a computable stereo block (position 5)
    # -- this is not a "no stereo at all" vacuous case.
    assert _stereo_prefix(ctx.mol, comp.spine_atom_to_locant) == "(5S)-"

    # fix a performance pass (wp6-tests), change-asserted-value (was the stereo-omitted
    # '5-[1-(methan-1-yl)propan-1-yl]-2,4-dioxo-3-(1-oxoethan-1-yl)-1-azacyclopentan-
    # 3-ide' with verified None, and a guard that the name must not start with
    # '(5S)-'). The floor now emits the COMPLETE descriptor set -- the branch centre
    # too ('(1S)-1-methylpropan-1-yl') -- and verify_or_none confirms it. The guard's
    # 0-wrong intent (never ship a PARTIAL descriptor) is met by a complete one:
    # OPSIN 2.9.0 (fresh call, outside the engine) gives the input's full InChIKey.
    # A producer-level best-effort name, not a PIN claim. The test id is kept.
    name, verified = _floor_name_and_verify(smi)
    assert name == ("(5S)-5-[(1S)-1-methylpropan-1-yl]-2,4-dioxo-3-(1-oxoethan-1-yl)-"
                     "1-azacyclopentan-3-ide")
    assert verified == name               # verified, with every stereocentre cited
    jar_or_skip()
    assert_full_rt(name, smi)


def test_ws7_control_betaine_still_prefers_route_charged_pin():
    """ must NOT regress the PIN charged path: betaine is still named by
    `route_charged` as `(trimethylazaniumyl)acetate`, NOT the floor's uglier
    von-Baeyer/replacement alternative (the floor is only reached AFTER the PIN
    charged path declines). End-to-end assertion (route_charged does not depend
    on the flaky recovery-lane path)."""
    assert Orthonym(**_WS7_BE).name("C[N+](C)(C)CC(=O)[O-]") == \
        "(trimethylazaniumyl)acetate"


# --------------------------------------------------------------------------
# fix a performance pass (dual review) — the offers-lane 0-wrong fail-open + the
# selanido/sulfido/azaniumyl cleanup.
# --------------------------------------------------------------------------

def test_ws7_offers_lane_fails_closed_on_transient_opsin_bypassed_offer():
    """CRITICAL (0-wrong): a FRESH floor offer whose exact string the gate never
    verified resolves to gate-outcome `bypassed` -> `_self01_lookup` returns
    `(None, False, "")`, which SKIPS the deny-by-default `verify_or_none` branch
    (that branch only guards `unavailable`/`not_run`). Pre-fix `_offer_rt_ok`
    then reached the `opsin_smiles is None` return and fail-OPENed on a transient
    `unavailable` -> the ungated floor offer shipped (production-reachable: an
    OPSIN subprocess timeout/OSError under load, with the jar PRESENT). The floor
    name for `F[B-](F)(F)F` is `2,2-difluoro-2-borapropan-2-uide`; healthy the
    pipeline ships the verified `tetrafluoroboranuide`, so the floor offer must
    NOT out-rank it via a transient blip. Post-fix the offers lane fails CLOSED
    (symmetric with the recovery-lane T4). MUST fail pre-fix (returned True)."""
    import orthonym.namer as N
    from orthonym.metrics import provenance as pv
    floor_name = "2,2-difluoro-2-borapropan-2-uide"
    smi = "F[B-](F)(F)F"
    try:
        # a REAL verdict recorded for a DIFFERENT (primary) string ->
        # resolve_gate_outcome(floor_name) == bypassed (the exact trigger).
        pv.record_gate_outcome(pv.GATE_OUTCOME_SELF01, "some-other-primary-name")
        assert pv.resolve_gate_outcome(
            pv.GATE_OUTCOME_SELF01, "some-other-primary-name", floor_name) \
            == pv.GATE_OUTCOME_BYPASSED
        assert N._self01_lookup(floor_name) == (None, False, "")
        # all OPSIN tiers transiently unavailable, jar PRESENT (not absent).
        _saved = (N._validity_gate_jar_present, N._validity_gate_status,
                  N._validity_gate_name_to_smiles)
        N._validity_gate_jar_present = lambda: True
        N._validity_gate_status = lambda name: "unavailable"
        N._validity_gate_name_to_smiles = lambda name: None
        try:
            assert N._offer_rt_ok(floor_name, smi) is False, \
                "offers lane fail-OPENed an ungated floor offer on transient OPSIN"
        finally:
            (N._validity_gate_jar_present, N._validity_gate_status,
             N._validity_gate_name_to_smiles) = _saved
    finally:
        pv.clear_provenance()


def test_ws7_offers_lane_healthy_verified_floor_offer_still_ships():
    """No over-tightening: a genuinely VERIFIED floor offer (jar present, OPSIN
    parses it, full-InChIKey matches the input) must still PASS `_offer_rt_ok`
    even though its provenance is `bypassed` (a fresh offer). Deterministic:
    `_validity_gate_name_to_smiles` is stubbed to OPSIN's real parse-back and the
    full-InChIKey compare (`_full_inchikey_offer_match`) is pure RDKit."""
    import orthonym.namer as N
    from orthonym.metrics import provenance as pv
    floor_name = "2-oxido-1-oxabut-1-ene"   # the floor name for propanoate
    smi = "CCC(=O)[O-]"
    try:
        pv.record_gate_outcome(pv.GATE_OUTCOME_SELF01, "some-other-primary-name")
        assert N._self01_lookup(floor_name) == (None, False, "")  # bypassed
        _saved = (N._validity_gate_jar_present, N._validity_gate_status,
                  N._validity_gate_name_to_smiles)
        N._validity_gate_jar_present = lambda: True
        N._validity_gate_status = lambda name: "parsed"
        N._validity_gate_name_to_smiles = lambda name: "CCC(=O)[O-]"  # OPSIN parse-back
        try:
            assert N._offer_rt_ok(floor_name, smi) is True, \
                "a genuinely-verified floor offer must still ship (no over-tighten)"
        finally:
            (N._validity_gate_jar_present, N._validity_gate_status,
             N._validity_gate_name_to_smiles) = _saved
    finally:
        pv.clear_provenance()


def test_ws7_selenido_leaf_roundtrips():
    """MEDIUM: the Se-anion leaf spelling is `selenido` (NOT the OPSIN-rejected
    `selanido`). The floor names `[Se-]CC` -> `1-selenidoethane` and
    `verify_or_none` CONFIRMs FULL, so the docstring's OPSIN-RT claim is true."""
    name, verified = _floor_name_and_verify("CC[Se-]")
    assert name == "1-selenidoethane"
    assert verified == name


def test_ws7_sulfido_leaf_roundtrips():
    """LOW: lock the S-anion leaf path so it is not silently uncovered."""
    name, verified = _floor_name_and_verify("CC[S-]")
    assert name == "1-sulfidoethane"
    assert verified == name


def test_ws7_azaniumyl_leaf_requires_three_h_not_a_nitrenium():
    """The `azaniumyl` (-NH3+) leaf fires only for a degree-1 N+ with EXACTLY 3 H
    (a true primary ammonium). A 2-H terminal N+ (aminylium/nitrenium) is a
    different species and must NOT be labelled `azaniumyl`."""
    from orthonym.assembly.universal_substituent import _charged_leaf_shortcut
    # -NH3+ (3 H) -> azaniumyl
    m1 = Chem.MolFromSmiles("C[NH3+]")
    n_idx = [a.GetIdx() for a in m1.GetAtoms() if a.GetSymbol() == "N"][0]
    assert _charged_leaf_shortcut(m1, frozenset({n_idx}), n_idx) == \
        ("azaniumyl", frozenset({n_idx}), frozenset({n_idx}))
    # R-NH2+ (2 H, nitrenium) -> the leaf declines (None)
    m2 = Chem.MolFromSmiles("C[NH2+]")
    n_idx2 = [a.GetIdx() for a in m2.GetAtoms() if a.GetSymbol() == "N"][0]
    assert _charged_leaf_shortcut(m2, frozenset({n_idx2}), n_idx2) is None


# --------------------------------------------------------------------------
# STEREO : the coverage floor now emits stereo descriptors. Root
# cause: `_build_ctx` already computes CIP via `assign_stereochemistry` but
# nothing downstream ever consulted `general_engine._stereo_prefix` /
# `collect_stereodescriptors`. Fix: prepend the TOP-LEVEL spine's OWN stereo
# block (mirroring general_engine's four parent engines,
# general_engine.py:893/1473/1930/2037), 0-wrong-gated through
# `verify_or_none` -- ships ONLY on a full-InChIKey confirm, else falls back
# to the plain (stereo-omitted) name unchanged. Witnesses below are REAL
# bucket-C (block1) rows from `v34_charged_backlog.json`; each expected string
# is OPSIN-RT re-derived via `_floor_name_and_verify`, never hand-edited.
# --------------------------------------------------------------------------

def test_ws_stereo_bicyclic_carnitine_ester_converts_block1_to_full():
    """ bucket-C (a dev split/chebi500): a quaternary-ammonium ester of a
    bicyclo[2.2.1]heptane alcohol. All 3 defined stereocentres sit on the
    polycyclic ring SPINE (`analyze_cage_universal`'s own atom_to_locant), so
    the with-stereo candidate full-RT-verifies and block1 -> full."""
    smi = "CCCCOC(=O)C[N+](C)(C)CCO[C@H]1C[C@H]2CC[C@@]1(C2(C)C)C"
    name, verified = _floor_name_and_verify(smi)
    # D3: plain C1 methyls -> 'dimethyl'/'trimethyl'; with the inner
    # parens gone the outer enclosure de-escalates  -> .
    # RT-verified identical InChIKey.
    assert name == ("(1R,2S,4R)-2-(4,4-dimethyl-6-oxo-1,7-dioxa-4-azaundecan-"
                     "4-ium-1-yl)-1,7,7-trimethylbicyclo[2.2.1]heptane")
    assert verified == name


def test_ws_stereo_cyclohexene_ammonium_ester_converts_block1_to_full():
    """ bucket-C (a dev split/chebi500): a triethyl/methyl-ammonium ester of a
    trimethylcyclohexenol. The 3 defined stereocentres are all monocyclic ring
    SPINE atoms (`_name_ring_spine`'s own atom_to_locant) -> block1 -> full."""
    smi = "CC[N+](C)(CC)CC(=O)OC[C@H]1[C@@H](CC(=C[C@@H]1C)C)C"
    name, verified = _floor_name_and_verify(smi)
    # D3: plain C1 ethyl/methyls -> 'ethyl'/'methyl'/'trimethyl'; the
    # outer enclosure de-escalates  -> . RT-verified identical.
    assert name == ("(3S,4S,5R)-4-(5-ethyl-5-methyl-3-oxo-2-oxa-5-"
                     "azaheptan-5-ium-1-yl)-1,3,5-trimethylcyclohex-1-ene")
    assert verified == name


def test_ws_stereo_long_chain_phosphocholine_thioester_converts_block1_to_full():
    """ bucket-C (a dev split/chebi500): a long acyclic replacement-nomenclature
    parent chain (thio/oxa-substituted, E/Z-laden). The ONE defined
    stereocentre (the glycerol-type carbon) is itself a CHAIN SPINE atom
    (`_name_chain_spine`'s own atom_to_locant), so it converts block1 -> full
    even though the parent is 40 atoms long with 4 defined double-bond
    descriptors already threaded through the unsaturation locants."""
    smi = ("CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)S[C@H](COCCCCCCCCCCCCCCCC)"
           "COP(=O)(O)OCC[N+](C)(C)C")
    name, verified = _floor_name_and_verify(smi)
    # D3: plain C1 methyls -> 'dimethyl'; outer enclosure de-escalates
    #  -> . RT-verified identical InChIKey.
    assert name == ("(19R,25Z,28Z,31Z,34Z)-19-(3-hydroxy-7,7-dimethyl-3-oxo-"
                     "2,4-dioxa-7-aza-3-phosphaoctan-7-ium-1-yl)-21-oxo-17-oxa-"
                     "20-thiatetraconta-25,28,31,34-tetraene")
    assert verified == name



# --------------------------------------------------------------------------
# NOABSTAIN  -- class 1: fused mancude/aromatic ring anion/cation
# (flavonoid/isoflavone phenolate + protonated heteroaromatic).
#
# Root cause (empirically established via direct probing of
# `analyze_cage_universal`, NOT the brief's original hypothesis): the floor's
# `_name_ring_spine` routes any >=2-SSSR ring system through
# `rules.vonbaeyer_universal.analyze_cage_universal`, passing `ctx.mol` --
# which `_build_ctx` has already `Chem.Kekulize(..., clearAromaticFlags=True)`d
# for its OWN bond-order bookkeeping. `analyze_cage_universal`'s internal
# self-consistency check round-trips its `mol` argument through a canonical
# SMILES and re-parses it to derive `orig_to_canon`; re-parsing NATURALLY
# re-aromatizes the fresh copy, so an already-Kekulized `mol` (aromatic flags
# cleared) vs. the freshly re-aromatized `canon` copy have INCOMPATIBLE bond
# representations for the SAME graph, and the bond-type-strict
# `GetSubstructMatch` reports NO match at all -- silently voiding the whole
# call for EVERY mancude/fused-aromatic ring system reaching this floor
# (measured: not just the charged witnesses here but plain naphthalene too).
# A second, independent bug rode along: `is_mancude` read the aromaticity
# flag off the SAME already-Kekulized `mol` (always False), so even a fixed
# match would have silently misclassified a real mancude cage as non-mancude.
#
# Fix (`rules/vonbaeyer_universal.py`): try the ORIGINAL self-consistency
# check UNCHANGED first (so every existing, already-working caller is
# byte-identical); only on failure, retry against a re-aromatized COPY with
# every atom's already-correct `GetTotalNumHs` re-pinned as EXPLICIT
# (`SetNoImplicit(True)` + `SetNumExplicitHs`) before serializing -- without
# this, `MolToSmiles` silently drops a pyrrole-type ring N-H (e.g. the
# protonated-purine witness below) that Kekulize had demoted from explicit to
# implicit bookkeeping, and the reparse then fails to kekulize at all. Then
# (`assembly/universal_substituent.py::_name_ring_spine`) opt into
# `allow_mancude=True` -- the SAME opt-in `ring_substituents.py`'s
# `_universal_cage_substituent_name` already uses -- since this module is the
# unconditional best-effort FLOOR (`t4_coverage.py`: "Reachable ONLY on the
# best-effort path... so PIN is untouched"), never reached at PIN tier.
#
# Witnesses are the REAL NOABSTAIN-brief rows (a dev split, in-scope charged
# abstains). Expected strings are OPSIN-RT re-derived via
# `_floor_name_and_verify` / `opsin_parse`, never hand-typed.
# --------------------------------------------------------------------------

def test_ws_noabstain_flavonoid_phenolate_mancude_cage_names_full_rt():
    """NOABSTAIN class 1, witness 1 (a dev split): a chromone/flavonoid
    phenolate anion. Pre-fix: floor VOIDED (Kekulize/re-aromatize mismatch);
    post-fix: emits a von-Baeyer polyene floor name that FULL-RT-verifies."""
    smi = "COc1cc(-c2cc(=O)c3c(O)cc([O-])cc3o2)ccc1O"
    name, verified = _floor_name_and_verify(smi)
    assert name == ("7-hydroxy-3-[4-hydroxy-3-(1-oxaethan-1-yl)cyclohexa-1,3,5-"
                     "trien-1-yl]-9-oxido-5-oxo-2-oxabicyclo[4.4.0]deca-"
                     "1(10),3,6,8-tetraene")
    assert verified == name  # full-InChIKey CONFIRMED


def test_ws_noabstain_isoflavone_phenolate_mancude_cage_names_full_rt():
    """NOABSTAIN class 1, witness 2 (a dev split): an isoflavone (methylenedioxy
    + phenolate) mancude bicyclic anion. Same fix as witness 1."""
    smi = "COc1cc2c(=O)c(-c3cc4c(cc3OC)OCO4)coc2cc1[O-]"
    name, verified = _floor_name_and_verify(smi)
    # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value: a prefix "is considered to begin with the first letter of its complete name" (the Blue Book): '(1-oxaethan-1-yl)' keys at 'oxaethanyl'
    # and '[4-(1-oxaethan-1-yl)-7,9-dioxabicyclo...-3-yl]' at 'oxaethanyldioxa...'
    # (it keyed at its inner locant '4', before every letter). OPSIN RT exact.
    assert name == ("8-(1-oxaethan-1-yl)-4-[4-(1-oxaethan-1-yl)-7,9-dioxabicyclo"
                     "[4.3.0]nona-1,3,5-trien-3-yl]-9-oxido-5-oxo-2-oxabicyclo"
                     "[4.4.0]deca-1(10),3,6,8-tetraene")
    assert verified == name  # full-InChIKey CONFIRMED


def test_ws_noabstain_protonated_purine_mancude_cage_block1_safe_omission():
    """NOABSTAIN class 1, witness 3 (a dev split): a protonated fused
    imidazo-pyrimidinium (purine-like) cation riding a piperidine amide arm
    with 2 branch-internal stereocentres. The mancude cage fix (this test's
    module docstring) is what stops the whole call from VOIDING; the 2
    stereocentres are branch-internal (off the ring spine STEREO
    captures), so the emitted name is constitution-correct but
    stereo-silent -- a SAFE OMISSION (`verify_or_none` returns None; block1,
    never `wrong`), not a wrong molecule. Mirrors
    `test_ws_stereo_branch_buried_stereocentre_still_omits_not_wrong`'s
    pattern."""
    smi = "C=CC(=O)N1C[C@H](Nc2[nH+]cnc3[nH]ccc23)CC[C@@H]1C"
    # fix a performance pass (wp6-tests), change-asserted-value (was the stereo-omitted name
    # with verified None and 'have != want'): the floor now cites both branch centres
    # ('(1R,4S)-') and verify_or_none confirms the name; OPSIN 2.9.0 (fresh call,
    # outside the engine) gives the input's full InChIKey (an InChIKey).
    # A producer-level best-effort name, not a PIN claim. The test id is kept.
    name, verified = _floor_name_and_verify(smi)
    assert name == ("5-{1-[(1R,4S)-4-methyl-3-(1-oxoprop-2-en-1-yl)-3-"
                     "azacyclohexan-1-yl]-1-azamethan-1-yl}-2,4,9-"
                     "triazabicyclo[4.3.0]nona-1,3,5,7-tetraen-4-ium")
    assert verified == name
    jar_or_skip()
    assert_full_rt(name, smi)


def test_ws_noabstain_naphthalene_mancude_cage_no_longer_voids():
    """Non-charged control: the SAME bug voided EVERY fused-aromatic ring
    system reaching this floor, not just charged ones. Naphthalene is a
    non-PIN control (the catalogs always win at PIN tier -- see
    `test_vonbaeyer_parent_fallback.py`'s own docstring) so this only checks
    the floor PRODUCER directly, never the end-to-end namer."""
    from orthonym.assembly.universal_substituent import (
        name_universal_substitutive)
    r = name_universal_substitutive(Chem.MolFromSmiles("c1ccc2ccccc2c1"))
    assert r is not None
    assert r.name == "bicyclo[4.4.0]deca-1,3,5,7,9-pentaene"


def test_ws_noabstain_achiral_saturated_bicycle_self_check_unchanged():
    """Regression guard for the try-original-first fallback strategy: a
    saturated (non-aromatic to begin with) bicyclic cage must take the
    UNCHANGED original code path (first attempt succeeds), never the
    aromaticity-repair fallback -- locks that the fallback only ever ADDS
    coverage and never alters an already-working case's numbering/spelling.
    """
    from orthonym.rules.vonbaeyer_universal import analyze_cage_universal
    smi = "C1CC2CCC1CC2"  # bicyclo[2.2.2]octane, fully saturated
    mol = Chem.MolFromSmiles(smi)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    cage = analyze_cage_universal(mol, cage_atoms=ring_atoms)
    assert cage is not None
    assert not cage.is_mancude



# --------------------------------------------------------------------------
# NOABSTAIN  -- class 2: cyclic phosphate diester (ring-embedded P,
# nucleotide).
#
# No dedicated code fix needed: this witness's abstain was a DOWNSTREAM
# consequence of the SAME class-1 mancude-cage bug above (its fused
# bromo-purine base ring hit the identical Kekulize/re-aromatize mismatch
# in `analyze_cage_universal`), not a separate ring-embedded-P defect as the
# brief's original hypothesis suggested. Once class 1's fix landed, this
# witness's ring-P phosphate `[O-]` composes exactly as `_charged_leaf_
# shortcut` and `_resolve_spine_charge` already intended -- confirmed here
# with its own dedicated regression test (not merely inferred from class 1's
# coverage) since it exercises a DIFFERENT ring system (a fused sugar-
# phosphate bicycle) than any class-1 witness.
# --------------------------------------------------------------------------

def test_ws_noabstain_ring_phosphate_bromopurine_nucleotide_full_rt():
    """NOABSTAIN class 2 (a dev split): a cyclic phosphate diester (ring-
    embedded P) on a fused bromo-purine nucleoside. FULL-RT-verifies once the
    fused purine ring's mancude-cage bug (class 1) is fixed."""
    smi = ("Nc1nc2c(nc(Br)n2[C@@H]2O[C@@H]3COP(=O)([O-])O[C@H]3[C@H]2O)"
           "c(=O)[nH]1")
    name, verified = _floor_name_and_verify(smi)
    assert name == ("(1S,6R,8R,9R)-8-[3-amino-8-bromo-5-oxo-2,4,7,9-"
                     "tetraazabicyclo[4.3.0]nona-1(6),2,7-trien-9-yl]-9-"
                     "hydroxy-3-oxido-3-oxo-2,4,7-trioxa-3-phosphabicyclo"
                     "[4.3.0]nonane")
    assert verified == name  # full-InChIKey CONFIRMED


# --------------------------------------------------------------------------
# NOABSTAIN  -- class 3: phosphinate P-H + poly-anion (3 independent
# charged leaves).
#
# Root cause (measured, NOT the brief's "net-charge/coverage composition in
# `_resolve_spine_charge` not yet validated for THREE simultaneous leaves"
# hypothesis -- the SAME defect reproduces on a single isolated phosphinate
# with ONE charged leaf, e.g. `C[PH](=O)[O-]`, so leaf MULTIPLICITY was never
# the blocker): the floor's generic chain/`_build_hetero_prefix` replacement-
# nomenclature machinery threads a phosphinate's P into the parent chain as a
# bare "phospha" position with its `[O-]` riding as an "oxido" branch
# substituent -- exactly the pattern that correctly spells a CARBON-hosted
# carboxylate (`-C(=O)[O-]` -> "...-oxido-1-oxaethene...", verified
# extensively by the class-1/ tests above). It silently mis-spells a
# PHOSPHORUS-hosted analogue: carbon's standard valence (4) leaves exactly 1
# spare valence unit for the "oxido" branch once the chain's own `=O` + `-C`
# bonds are counted (2+1=3, +1 for oxido = 4, self-consistent); phosphorus's
# standard valence (3) is ALREADY fully used by those same two chain bonds
# (2+1=3), so adding "oxido" pushes P to bonding number 4 with NO valence
# headroom and no lambda citation -- OPSIN then reads the extra "oxido" as
# ANOTHER double-bonded oxo, silently dropping the anion's charge (and the
# defining P-H) entirely: `C[PH](=O)[O-]` -> floor's old chain spelling
# "2-oxido-1-oxa-2-phosphaprop-1-ene" -> OPSIN parses to the NEUTRAL
# `O=P(=O)C`. Citing the correct lambda convention
# (`rules.lambda_convention`) on that SAME chain spelling does NOT fix it
# either (measured: `2-oxido-1-oxa-2λ5-phosphaprop-1-ene` still parses back
# to `O=P(=O)C`) -- OPSIN simply does not validate this shape via replacement
# nomenclature, so citing lambda inside `_build_hetero_prefix` was reverted
# (it also regressed multiple already-correct, already-shipping names that
# carry an unrelated non-standard-valence S/P WITHOUT a lambda citation,
# e.g. a sulfonic-acid-in-chain S or an ammonium-ester phosphate P that OPSIN
# already round-trips correctly without one).
#
# Fix: `_phosphinate_oxide_leaf_shortcut` (+ `_is_phosphinate_oxide_root` to
# keep `_tree_neighbors` from threading the P into chain continuation)
# recognises the whole `-P(=O)([O-])(H)` unit and renders it via
# SUBSTITUTIVE (not replacement) nomenclature as the `phosphanyl` mononuclear-
# hydride substituent group with `oxo`/`oxido` prefixes --
# OPSIN-verified: `(oxido(oxo)phosphanyl)ethane` -> `[O-]P(=O)CC`, RDKit
# filling the SAME 1-implicit-H valence-5 P the already-working
# `methylphosphinate` -> `CP([O-])=O` relies on.
#
# The 3-simultaneous-charged-leaf witness itself carries a genuine, separate
# E/Z (not R/S) geometric descriptor on its central C=C that this producer's
# STEREO splice does not cite (scoped to tetrahedral CIP only) -- so it
# converts abstain -> block1 (safe geometric-descriptor OMISSION, right
# constitution, never a wrong molecule), while the 2 simpler (no E/Z)
# phosphinate witnesses convert abstain -> full.
# --------------------------------------------------------------------------

def test_ws_noabstain_methylphosphinate_leaf_full_rt():
    """Unit witness: the simplest phosphinate leaf, isolated (no other
    charge). Confirms the shortcut alone, independent of the composite
    3-charged-leaf witness below."""
    name, verified = _floor_name_and_verify("C[PH](=O)[O-]")
    assert name == "1-[oxido(oxo)phosphanyl]methane"
    assert verified == name  # full-InChIKey CONFIRMED


def test_ws_noabstain_ethylphosphinate_leaf_full_rt():
    """Unit witness: same shortcut, one carbon longer -- generalization
    proof (not a single hard-coded molecule)."""
    name, verified = _floor_name_and_verify("CC[PH](=O)[O-]")
    assert name == "1-[oxido(oxo)phosphanyl]ethane"
    assert verified == name  # full-InChIKey CONFIRMED


def test_ws_noabstain_triple_charged_leaf_phosphinate_polyanion_block1():
    """NOABSTAIN class 3, the brief's own witness (chebi500): a net-3
    polyanion -- two independent carboxylates plus the phosphinate leaf above,
    all on one short unsaturated backbone. Converts abstain -> block1 (a
    genuine E/Z geometric-descriptor omission on the central C=C, verified
    via direct InChIKey compare below; never a wrong molecule)."""
    smi = "O=C([O-])C/C(=C/[PH](=O)[O-])C(=O)[O-]"
    name, verified = _floor_name_and_verify(smi)
    assert name == ("2,5-dioxido-3-{1-[oxido(oxo)phosphanyl]methan-1-ylidene}"
                     "-1,6-dioxahexa-1,5-diene")
    assert verified is None  # block1: safe E/Z omission, never shipped wrong
    got = opsin_parse(name)
    assert got, f"OPSIN could not parse {name!r}"
    have = Chem.MolToInchiKey(Chem.MolFromSmiles(got))
    want = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    assert have[:14] == want[:14]  # constitution matches (block1)
    assert have != want            # E/Z genuinely omitted, not wrong


def test_ws_noabstain_phosphinate_leaf_excluded_from_chain_continuation():
    """Unit-level: `_is_phosphinate_oxide_root` must keep the phosphinate P
    OUT of `_tree_neighbors`' chain-continuation candidates so it is always
    discovered as a branch (where the leaf shortcut can see it), never
    mis-threaded into the replacement-nomenclature chain as an ordinary
    standard-valence heteroatom."""
    from orthonym.assembly.universal_substituent import (
        _is_phosphinate_oxide_root)
    mol = Chem.MolFromSmiles("CC[PH](=O)[O-]")
    p_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P"][0]
    assert _is_phosphinate_oxide_root(mol, p_idx) is True
    # A phosphonate-style P (two carbon substituents, no P-H) is a DIFFERENT
    # species and must NOT match -- it is not degree-3 with exactly 1 H.
    mol2 = Chem.MolFromSmiles("CP(C)(=O)[O-]")
    p_idx2 = [a.GetIdx() for a in mol2.GetAtoms() if a.GetSymbol() == "P"][0]
    assert _is_phosphinate_oxide_root(mol2, p_idx2) is False


# --------------------------------------------------------------------------
# NOABSTAIN  -- class 4: reverse-prenyl substituent riding a
# zwitterion (a GENERAL chain/unsaturation defect, not a charge defect per
# the brief).
#
# No dedicated fix needed: this witness's abstain was ALSO a downstream
# consequence of the class-1 mancude-cage bug (its fused indole ring hit the
# identical Kekulize/re-aromatize mismatch), not the brief's originally-
# diagnosed "reverse-prenyl mis-spelled as saturated" general-chain defect.
# Once class 1's fix lands, the floor's OWN chain builder already spells the
# prenyl branch correctly and completely as the UNSATURATED
# "3-(methan-1-yl)but-2-en-1-yl" (3-methylbut-2-en-1-yl, i.e. genuine
# prenyl -- NOT the brief's feared saturated "pentyl" mis-spelling), so no
# general chain/unsaturation fix was needed at all. Converts abstain ->
# block1: the ONE defined stereocentre sits on the azaniumyl/carboxylate
# BRANCH (off the ring spine), which this producer's STEREO splice
# intentionally does not capture (spine-only scope, same as
# `test_ws_stereo_branch_buried_stereocentre_still_omits_not_wrong` above)
# -- confirmed a safe OMISSION, not a wrong molecule, via direct InChIKey
# compare.
# --------------------------------------------------------------------------

def test_ws_noabstain_reverse_prenyl_indole_zwitterion_block1_safe_omission():
    """NOABSTAIN class 4 (a dev split): a tryptophan-like zwitterion decorated
    with a prenyl substituent on the fused indole ring."""
    smi = "CC(C)=CCc1cccc2c(C[C@H]([NH3+])C(=O)[O-])c[nH]c12"
    # fix a performance pass (wp6-tests), change-asserted-value (was the stereo-omitted name
    # with verified None and 'have != want'): the floor now cites the branch centre
    # ('(2S)-') and verify_or_none confirms the name; OPSIN 2.9.0 (fresh call, outside
    # the engine) gives the input's full InChIKey (an InChIKey). A
    # producer-level best-effort name, not a PIN claim. The test id is kept.
    name, verified = _floor_name_and_verify(smi)
    assert name == ("9-[(2S)-2-azaniumyl-3-oxido-4-oxabut-3-en-1-yl]-5-"
                     "(3-methylbut-2-en-1-yl)-7-azabicyclo[4.3.0]"
                     "nona-1,3,5,8-tetraene")
    assert "pentyl" not in name  # NOT the brief-feared saturated mis-spelling
    assert verified == name
    jar_or_skip()
    assert_full_rt(name, smi)


def test_ws_stereo_achiral_floor_name_unaffected_no_stereo_block():
    """No regression on a NEUTRAL, achiral molecule that reaches the floor: no
    stereocentre exists, so `_stereo_prefix` returns '' and the plain name is
    used unchanged (the `if stereo_block:` short-circuit is never entered)."""
    from orthonym.assembly.universal_substituent import (
        name_universal_substitutive)
    smi = "CC1(C)CCCCC1"  # 1,1-dimethylcyclohexane, no stereocentre
    r = name_universal_substitutive(Chem.MolFromSmiles(smi))
    assert r is not None
    assert not r.name[:1] == "("   # no spurious stereo-descriptor parenthetical
    assert "R)" not in r.name and "S)" not in r.name


# --------------------------------------------------------------------------
# NOABSTAIN  -- class 5: fold-in, the skeletal-`ium`
# `_p74_bare_ring_stem` spurious `-2-yl` bug (name QUALITY, not an abstain --
# 's floor already masked this with a valid, full-RT name).
#
# Root cause: ``_p74_bare_ring_stem`` (``rules/ions.py``) strips every
# EXOCYCLIC substituent off the ring (`RWMol.RemoveAtom`) to get a bare stem
# to re-name, e.g. "pyridine". `RemoveAtom` does NOT recompute a REMAINING
# ring atom's implicit-H bookkeeping when a heavy neighbour is deleted -- the
# carboxylate-bearing ring carbon's own bracket atom (`[C@H]`) keeps
# its ORIGINAL `noImplicit=True` / explicit-H=1 from when it had 4
# connections (2 ring bonds + the now-deleted carboxyl branch + 1 H). After
# deletion it has only 3 (2 ring bonds + 1 explicit H) -- one short of
# carbon's valence 4 -- and with `noImplicit` still `True`, `SanitizeMol`
# cannot silently top it up, so it becomes an open-valence/radical-shaped
# atom (`[CH]1CCCN1`, not `C1CCCN1`). The namer then reads that open valence
# as a FREE VALENCE ("this is a substituent fragment") and returns the `-yl`
# form (`pyrrolidin-2-yl`) instead of the parent hydride (`pyrrolidine`) --
# exactly the malformed `1-methylpyrrolidin-2-yl-1-ium-2-carboxylate`
# `emit_zwitterion_ring_carboxylate` then glued a `-1-ium-2-carboxylate`
# cumulative suffix onto. Fix: reset `SetNoImplicit(False)` on EVERY ring
# atom (the existing reset already did this for the cation atom alone, for
# the same underlying reason), not just the cation -- a ring atom that never
# lost a neighbour is unaffected (clearing an already-False flag is a no-op).
#
# The overall PIPELINE OUTPUT for the witness is unchanged by this fix ('s
# floor + STEREO already ship the SAME full-RT name either way, since
# already suppressed the malformed pre-fix string and fell through to
# the floor) -- this is a pure internal-quality fix: `route_charged`'s own
# emitted string is no longer grammatically malformed/unparseable for this
# whole class (any skeletal-ium ring whose anion-attachment ring
# atom is a stereocentre), even though it still omits the stereo descriptor
# itself (a separate, larger gap -- threading stereo into this NEW
# cumulative-suffix builder is out of this fix's scope).
# --------------------------------------------------------------------------

def test_ws_noabstain_p74_skeletal_ium_stem_no_longer_emits_spurious_yl():
    """NOABSTAIN class 5 (a dev split): `emit_zwitterion_ring_carboxylate` on
    the brief's own witness must emit a VALID, OPSIN-parseable
    cumulative name -- no more spurious `-yl` glued onto the ring stem."""
    from orthonym.rules.ions import emit_zwitterion_ring_carboxylate
    smi = "C[NH+]1CCC[C@H]1C(=O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    cation_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1][0]
    anion_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == -1][0]
    name = emit_zwitterion_ring_carboxylate(mol, cation_idx, anion_idx)
    assert name == "1-methylpyrrolidin-1-ium-2-carboxylate"
    assert "-yl" not in name  # the spurious substituent-yl morpheme is gone
    got = opsin_parse(name)
    assert got, f"OPSIN could not parse {name!r} (pre-fix this was unparseable)"
    # constitution matches (this builder never claimed to thread stereo)
    have = Chem.MolToInchiKey(Chem.MolFromSmiles(got))
    want = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    assert have[:14] == want[:14]


def test_ws_noabstain_p74_bare_ring_stem_resets_every_ring_atom():
    """Unit-level: `_p74_bare_ring_stem` must return the PARENT HYDRIDE
    ('pyrrolidine'), never a substituent `-yl` form, once its exocyclic
    substituents are stripped -- generalization proof, not molecule-specific
    (the SAME shape with a different ring size / different anion-attachment
    position must also resolve)."""
    from orthonym.rules.ions import _p74_bare_ring_stem
    smi = "C[NH+]1CCC[C@H]1C(=O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    ring_system = frozenset(
        a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    cation_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1][0]
    stem = _p74_bare_ring_stem(mol, ring_system, cation_idx)
    assert stem == "pyrrolidine"
    assert not stem.endswith("-yl") and "yl" not in stem


def test_ws_noabstain_p74_end_to_end_still_ships_full_via_floor():
    """End-to-end: the witness's SHIPPED name is unchanged by this fix (both
    before and after, `route_charged`'s malformed/stereo-incomplete string is
    -suppressed and the STEREO-capable floor ships the same full-RT
    name) -- this locks that this is a pure internal-quality fix, not a
    behaviour change to what users see for THIS witness."""
    from orthonym.assembly.universal_substituent import (
        name_universal_substitutive)
    from orthonym.validation.reconstruct import verify_or_none
    smi = "C[NH+]1CCC[C@H]1C(=O)[O-]"
    r = name_universal_substitutive(Chem.MolFromSmiles(smi))
    assert r is not None
    # D3: plain C1 methyl -> retained 'methyl'.
    assert r.name == ("(2S)-1-methyl-2-(1-oxido-2-oxaeth-1-en-1-yl)"
                       "-1-azacyclopentan-1-ium")
    assert verify_or_none(r.name, smi) == r.name  # full-InChIKey CONFIRMED
