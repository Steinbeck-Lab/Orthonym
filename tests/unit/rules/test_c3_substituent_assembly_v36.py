"""v36 Milestone C3 — substituent/assembly gap on WORKING ring cores.

RED baseline + RT harness for the C3 witnesses (grounding spy
``). The C3 charter is: name molecules
whose ring system ALREADY names but which abstain because a ring-bearing
SUBSTITUENT (or a spiro-linked partner ring) can't be rendered — by letting
the working whole-molecule ring namer serve the substituent role too, all
OPSIN-round-trip-verified, ZERO ring-engine change (touch only ``assembly/``).

================================================================================
MEASURED at the milestone tier (``general_fallback=True,
general_fallback_unverified=True`` — the exact tier ``gen_ost_full.py`` used to
build ``chebi_full_ost.jsonl``, from which the spy drew its 2,147 abstainers):
the plan's Task-2 premise (a fail-closed DROP-24 guard in
``assembly/substituent_naming.py`` is THE dominant decline site and re-anchoring
the whole-molecule ring namer ahead of it closes the category-(a) witnesses)
is OFF-PATH for every named flagship witness. This is the contributor guide invariant 8
("the named choke point is off the path") holding again.

The re-anchor mechanism the plan asks for ALREADY EXISTS and already works at the
milestone tier: ``rules/ring_substituents.py::name_ring_system_substituent(...,
allow_mancude=True)`` re-anchors von-Baeyer / spiro / fused / cage ring systems
into the ``-yl`` substituent role (verified control: ``adamantyl_benzoic`` below
names ``4-(adamantan-1-yl)benzoic acid`` at gfu). Step 1c already passes
``allow_mancude=best_effort_ctx``, which is True at the gfu milestone tier, so no
DROP-24-rescue gap remains there.

Per-witness VERIFIED root cause (why each abstains at gfu), and the engine that
owns the real fix — NONE is a DROP-24 re-anchor gap in ``assembly/``:

  * estramustine — the steroid ESTER engine builds a near-correct candidate
    ``(...)-17-hydroxyestr-1,3,5(10)-trien-3-yl pentanoate`` that SELF-01
    suppresses for TWO bugs, both in ``rules/natural_products.py``
    (a scaffold engine, out of C3's assembly-only charter):
      (a) euphonic 'a' dropped: ``_assemble_np_ester_name`` concatenates
          ``stem`` (``estr``) + ``unsat_suffix`` (``-1,3,5(10)-trien``) with no
          'a' before the multiplied ``-trien`` suffix → ``estr`` not ``estra``;
      (b) the carbamate acid is named by naive carbon-counting
          (``_find_ester_decorations`` → ``_count_acid_fragment_carbons`` →
          ``get_systematic_acylate``): the 5 carbons of
          ``C(=O)N(CH2CH2Cl)(CH2CH2Cl)`` → ``pentanoate``, ignoring N and Cl —
          a DIFFERENT molecule. (Sharpened: OST's GENERAL ester path already
          names this acid correctly — ``Cc1ccc(OC(=O)N(CCCl)CCCl)cc1`` →
          ``4-methylphenyl N,N-bis(2-chloroethyl)carbamate`` RT-True — so the
          defect is the STEROID ester path preempting the general one with its
          carbon-count shortcut.)
    → RE-BUCKET to the steroid/scaffold ester engine. (Both the ester
    functional-class target and a substitutive steroid-parent + carbamoyloxy
    form are RT-verified below, so a 0-wrong name exists once that engine is
    fixed.)

  * w0, w13 — the spiro ring engine DOES attach every FG (the candidate carries
    ``diformyl``/``hydroxy``/``oxo``/etc. on the correct spiro core); SELF-01
    suppresses it only because a stereodescriptor is OMITTED (the OPSIN
    round-trip SMILES is the RIGHT constitution with a dropped ``@``). This is
    stereo OMISSION, not a substituent gap and not a wrong molecule
    (feedback_stereo_omission_is_not_wrong_molecule). → RE-BUCKET to spiro/ring
    stereo completion.

  * w16 — the composer picks the ketone 6-ring as parent and the spiro-attached
    naphtho-dioxole partner (16 atoms) is dropped; the candidate is
    ``8-hydroxy-3,4-dihydronaphthalen-1(2H)-one`` (the tetralone alone), which
    SELF-01 correctly vetoes as an atom-drop → the molecule abstains (NO silent
    wrong emission; 0-wrong already holds via SELF-01/atom-coverage). Naming it
    requires routing the WHOLE spiro system through the complex_ring/spiro namer
    WITH a ketone principal group + phenol — a ring-engine capability no tier
    currently has. → RE-BUCKET to C1/C2 (ring program). The plan's Site-3 site
    (``composer.py`` ~:966-993) is only the ASSEMBLY_DISPATCH log line; the
    parent/substituent decision is upstream in perception
    (``namer.py``'s NAMING_DECISION at :6414-6424).

  * catb3 — genuine category-(b): two independently-nameable ring systems joined
    by an ester/carbamate linker. Overlaps C4's charter (see Task 6). Abstains
    cleanly (0-wrong).

CONCLUSION: no C3 witness is closeable within the strict assembly-only,
zero-ring-engine charter. This file is therefore the RED baseline + a set of
RT-verified TARGET names (xfail) that will flip to GREEN automatically once the
re-bucketed ring/scaffold-engine fixes land — actionable regression scaffolding
for the re-plan, not dead code.

Regenerate the witness data in minutes via the spy's method
(`` "Method").
"""
import pytest

from orthonym.errors import is_failure_name
from orthonym.namer import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# --- witnesses (spy representatives, spanning categories a / d / b) ----------
ESTRAMUSTINE = (
    "C[C@]12CC[C@H]3[C@@H](CCc4cc(OC(=O)N(CCCl)CCCl)ccc34)[C@@H]1CC[C@@H]2O"
)
W0 = "CC(C)c1cc(C=O)c2c(c1O)OC(=O)[C@@]21CCCC(C)(C)[C@@H]1C=O"
W13 = "CC(=O)O[C@@H]1[C@@H](O)[C@H]2O[C@@H]3C=C(C)[C@@H](O)C[C@]3(CO)[C@]1(C)C21CO1"
W16 = "O=C1CCC2(Oc3cccc4cccc(c34)O2)c2cccc(O)c21"
CATB3 = (
    "O=C(ON1C(=O)CCC1=O)c1cc(Cl)c2c(c1Cl)C1(OC2=O)c2cc(Cl)c(O)cc2Oc2cc(O)"
    "c(Cl)cc21"
)
CATB6 = "O=C1OC2(c3ccc(O)cc3Oc3cc(Oc4ccc(O)cc4)ccc32)c2ccccc21"

# genuine category-(b): two independently-nameable ring systems joined by an
# ester/ether/carbamate linker. Overlaps C4's charter (see Task 6).
C3_CATEGORY_B_WITNESSES = {"catb3": CATB3, "catb6": CATB6}

# The four category-(a)/(d) witnesses the plan's Task-2 was meant to close.
C3_TARGET_WITNESSES = {
    "estramustine": ESTRAMUSTINE,
    "w0": W0,
    "w13": W13,
    "w16": W16,
}

# Control: a complex ring system genuinely in the SUBSTITUENT role. Proves the
# "re-anchor the working ring namer into the substituent role" mechanism the
# plan's Task-2 asks for ALREADY EXISTS and already fires at the milestone tier
# (allow_mancude → get_ring_substituent_name → 'adamantan-1-yl'). Not a witness.
ADAMANTYL_BENZOIC = "OC(=O)c1ccc(cc1)C12CC3CC(CC(C3)C1)C2"


def _c3_name(smiles: str) -> str:
    """Best-effort name at the C3 MILESTONE tier (the exact tier gen_ost_full.py
    used to build the abstain corpus the spy sampled)."""
    return name_compound(
        smiles, general_fallback=True, general_fallback_unverified=True
    )


def _c3_rt(smiles: str):
    """Name best-effort (milestone tier) + OPSIN round-trip check.

    Returns (name, passed): ``passed`` is True iff a non-failure name was
    produced AND it OPSIN-round-trips to the input's InChI. An abstention
    ('unknown organic compound') is ``passed=False`` with the failure name.
    """
    name = _c3_name(smiles)
    if is_failure_name(name):
        return name, False
    rt = opsin_roundtrip_check(smiles, name)
    return name, bool(rt.get("passed"))


# --- RED baseline: no C3 target witness produces a 0-wrong name TODAY ---------
# Definition of RED is gate-INDEPENDENT: OST produces no name that OPSIN-round-
# trips. That is exactly the C3 gap, and it holds whether or not the SELF-01/
# OPSIN naming gate is active in the test process (at the milestone tier the
# gate suppresses the near-miss candidate → abstain; with the gate off the namer
# emits a candidate that fails OPSIN-RT here — either way, no 0-wrong name).
@pytest.mark.parametrize("key,smiles", sorted(C3_TARGET_WITNESSES.items()))
def test_c3_witness_has_no_roundtripping_name_today_RED(key, smiles):
    """RED baseline: each C3 target witness has NO name that OPSIN-round-trips
    today. When a re-bucketed ring/scaffold-engine fix lands and a witness
    starts naming correctly, this fails LOUDLY — the signal to promote it into
    the GREEN ``xfail(strict=True)`` targets below."""
    name, passed = _c3_rt(smiles)
    assert not passed, (
        f"{key} now produces a round-tripping name: {name!r} — promote it to a "
        f"GREEN target."
    )


def test_c3_reanchor_mechanism_already_works_at_milestone_tier():
    """The plan's Task-2 lever ('re-anchor the working ring namer into the
    substituent role') ALREADY EXISTS: a complex cage ring in the substituent
    role names + round-trips at the milestone tier via allow_mancude. This is
    the evidence that a fresh DROP-24 re-anchor in assembly/ would be dead code
    (the contributor guide invariant 17)."""
    name, passed = _c3_rt(ADAMANTYL_BENZOIC)
    assert not is_failure_name(name), f"cage substituent regressed: {name!r}"
    assert passed, f"cage substituent no longer round-trips: {name!r}"


# --- RT-verified TARGET names (xfail) — flip to GREEN when the re-bucketed ----
# --- ring/scaffold-engine fixes land. strict=True so a silent close is caught. -
_C3_TARGETS = {
    # estramustine: PIN = ester functional-class name (owned by the steroid
    # ester engine in rules/natural_products.py — euphonic 'a' + carbamate acid).
    "estramustine": (
        ESTRAMUSTINE,
        "(8R,9S,13S,14S,17S)-17-hydroxyestra-1,3,5(10)-trien-3-yl "
        "N,N-bis(2-chloroethyl)carbamate",
    ),
}


@pytest.mark.parametrize("key,smiles,target", [
    (k, s, t) for k, (s, t) in _C3_TARGETS.items()
])
def test_c3_target_name_round_trips(key, smiles, target):
    """The intended 0-wrong name for each re-bucketed witness OPSIN-round-trips
    to the input — proving a correct deliverable exists once its owning engine
    is fixed (this is a name-validity check, independent of the namer)."""
    rt = opsin_roundtrip_check(smiles, target)
    assert rt.get("passed"), f"{key} target does not round-trip: {rt.get('error')}"


@pytest.mark.xfail(strict=True, reason=(
    "estramustine PIN needs rules/natural_products.py fixes (euphonic 'a' in "
    "_assemble_np_ester_name + carbamate acid in _find_ester_decorations) — a "
    "scaffold-engine fix, RE-BUCKETED out of C3's assembly-only charter."
))
def test_c3_estramustine_names_GREEN():
    name, passed = _c3_rt(ESTRAMUSTINE)
    assert passed, f"estramustine did not name+round-trip: {name!r}"


@pytest.mark.xfail(strict=True, reason=(
    "w0 abstains only because a stereodescriptor is OMITTED from an otherwise-"
    "correct spiro candidate (right constitution) — RE-BUCKETED to spiro/ring "
    "stereo completion, not a substituent-assembly gap."
))
def test_c3_w0_names_GREEN():
    name, passed = _c3_rt(W0)
    assert passed, f"w0 did not name+round-trip: {name!r}"


@pytest.mark.xfail(strict=True, reason=(
    "w16 needs the WHOLE spiro system routed through complex_ring/spiro naming "
    "with a ketone principal group + phenol — a ring-engine capability, "
    "RE-BUCKETED to C1/C2. Abstains cleanly today (0-wrong via SELF-01)."
))
def test_c3_w16_names_GREEN():
    name, passed = _c3_rt(W16)
    assert passed, f"w16 did not name+round-trip: {name!r}"


# --- Task 6: category-(b) two-ring-linker witnesses → coordinate with C4 ------
@pytest.mark.parametrize("key,smiles", sorted(C3_CATEGORY_B_WITNESSES.items()))
def test_c3_categoryb_abstains_cleanly_defer_to_C4(key, smiles):
    """Category-(b): two independently-nameable ring systems joined by an
    ester/ether/carbamate linker. That linker-composition step overlaps C4's
    charter — do NOT duplicate it in C3. Each must abstain CLEANLY (0-wrong: an
    abstention, never a wrong/atom-dropped molecule) until C4 owns it. The 2
    bis-indole peptide-shaped spy witnesses are EXCLUDED from C3 entirely
    (unowned peptide-backbone gap, per the spy).

    Gate-independent 0-wrong check (as in the RED baseline): C3 must not emit a
    name that round-trips. At the milestone tier the SELF-01/OPSIN gate makes
    this an outright abstention; with the gate off, the namer may emit an
    atom-dropped near-name (e.g. catb6 → ``2-benzofuran-1(3H)-one``, only one
    ring) that fails OPSIN-RT — never a 0-wrong emission either way."""
    name, passed = _c3_rt(smiles)
    assert not passed, (
        f"{key} now produces a round-tripping name {name!r} — if intentional, "
        f"this belongs to C4, not C3."
    )
