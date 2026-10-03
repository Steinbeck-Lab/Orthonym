""" Defect B: best-effort rescue for a purine ring substituent that
`_identify_fused_substituent` cannot (fully) identify.

`get_fused_heterocycle_substituents` silently drops any exocyclic branch it
cannot type, so `name_substituted_purine` / `name_purine_substituent` fail
closed (via `_exocyclic_atoms_accounted`) rather than name a different
molecule. That is correct for PIN/default, but it means a purine carrying one
GIANT arm -- e.g. a nucleotide's ribose-diphosphate-pantetheine chain on N9,
as in acetyl-CoA -- can never be named at all: the narrow identifier's
producers were never built to recognise a whole nucleotide tail.

`_best_effort_augment_purine_subs` (purine.py) retries any unaccounted branch
with the general recursive substituent namer, gated on `best_effort_ctx`, so
the PIN/default tier stays byte-identical and 0-wrong is preserved by the
downstream /OPSIN round-trip gate.

⚠ Acetyl-CoA's arm (41 heavy atoms) is the clean unit-level demonstration:
`_identify_fused_substituent`'s general fallback caps at 25 atoms, so this
branch is UNAMBIGUOUSLY unidentified (`sub_info is None` for every atom in
it), never partially-and-wrongly identified. A DIFFERENT, smaller molecule
(`OP(=O)(O)OCn1cnc2c(N)ncnc21`, a bare phosphate directly on N9) is deliberately
NOT used for a direct `name_substituted_purine` unit assertion here: probing
it surfaced a PRE-EXISTING, separate bug -- `_identify_fused_substituent`'s
functionalized-chain identifier mis-names the whole `-CH2-O-P(=O)(OH)2` branch
as bare `hydroxymethyl` (2 atoms) while claiming (via `sub_info['atoms']`) to
have consumed the SAME atom set as the full 6-atom branch, so
`_exocyclic_atoms_accounted` reports the branch as already fully accounted
and this fix's augmentation never runs -- WRONG on the PIN/default path too,
not something this fix introduces or can gate. It is caught downstream (the
top-level /OPSIN gate suppresses it: 0-wrong holds at the pipeline
level, confirmed by `test_...pipeline` below round-tripping via a DIFFERENT
producer's candidate), but it is a live latent defect in the shared
identifier, out of scope here (explicitly not to be touched by this task) and
reported separately.
"""
import pytest
from rdkit import Chem

from orthonym.rules.purine import name_substituted_purine
from orthonym.metrics.provenance import best_effort_ctx

pytestmark = pytest.mark.opsin_gate

# Full acetyl-CoA: purine N9 carries a ribose-diphosphate-pantetheine-thioester
# arm (41 heavy atoms) that `_identify_fused_substituent`'s narrow producers
# cannot type at all (its general fallback caps at 25 atoms).
_ACETYL_COA = (
    "CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)"
    "OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O"
)
# Team-lead acceptance case #2: a phosphate arm directly on N9. NOT used for a
# direct name_substituted_purine assertion (see module docstring); exercised
# only end-to-end, where the full pipeline's producer competition round-trips
# correctly regardless of which internal candidate wins.
_PHOSPHO_ARM_PURINE = "OP(=O)(O)OCn1cnc2c(N)ncnc21"


def _mol(smi):
    return Chem.MolFromSmiles(smi)


@pytest.fixture
def _best_effort():
    tok = best_effort_ctx.set(True)
    yield
    best_effort_ctx.reset(tok)


def test_giant_arm_declines_on_pin_default_path():
    # Byte-identical to pre-fix behaviour: best_effort_ctx is OFF by default,
    # so the giant/unidentifiable arm still fails closed -> None.
    assert name_substituted_purine(_mol(_ACETYL_COA)) is None


def test_acetyl_coa_names_under_best_effort(_best_effort):
    # The GOAL case: a whole ribose-diphosphate-pantetheine-thioester arm
    # named as a single ring substituent at N9, instead of the molecule
    # silently dropping to a malformed / unnameable fallback.
    name = name_substituted_purine(_mol(_ACETYL_COA))
    assert name is not None
    assert name.endswith("9H-purin-6-amine")
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="test-purine-best-effort"):
        rt = opsin_roundtrip_check(_ACETYL_COA, name)
    assert rt["passed"], rt


def test_boronic_acid_now_names_under_best_effort(_best_effort):
    # A genuine widening (not a regression): the general recursive namer CAN
    # express this fragment even though `_identify_fused_substituent`'s
    # narrow producers cannot. PIN/default decline for this same molecule is
    # covered by test_unclassifiable_substituent_fails_closed in
    # test_substituted_purine.py, which never sets best_effort_ctx.
    name = name_substituted_purine(_mol("OB(O)c1ncnc2[nH]cnc12"))
    assert name is not None
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="test-purine-best-effort"):
        rt = opsin_roundtrip_check("OB(O)c1ncnc2[nH]cnc12", name)
    assert rt["passed"], rt


def test_sentinel_leak_guard_direct():
    # errors.py's documented 310/10000-row leak class: a bare equality/space
    # check on the cascade placeholder MISSES its DECORATED forms (a locant or
    # italic-element prefix already welded on, e.g. 'N-substituentformamide').
    # `is_refusal_sentinel` is a substring test, so it catches both the bare
    # and the decorated shape; lock that contract directly.
    from orthonym.errors import is_refusal_sentinel
    assert is_refusal_sentinel('substituent')
    assert is_refusal_sentinel('N-substituentformamide')
    assert is_refusal_sentinel('N-substituenthydroxyphosphonooxytricosanamide')
    assert not is_refusal_sentinel('9H-purin-6-amine')


def test_decorated_sentinel_leak_fails_closed(_best_effort, monkeypatch):
    # If the general recursive namer ever returns a DECORATED sentinel for an
    # unaccounted branch -- no literal space, not equal to the bare
    # 'substituent' placeholder, so a naive equality/space check would have
    # let it through -- the augmentation must still fail closed rather than
    # weld it into the assembled purine name (never a wrong/garbage name).
    import orthonym.rules.ring_substituents as ring_substituents_mod
    monkeypatch.setattr(
        ring_substituents_mod, "name_ring_system_substituent",
        lambda *a, **k: "N-substituentformamide",
    )
    assert name_substituted_purine(_mol(_ACETYL_COA)) is None


def test_pin_default_engine_output_unchanged_end_to_end():
    # Byte-identity spot check on the default (PIN) engine tier: normal
    # molecules unaffected by this fix, incl. the purine family itself.
    from orthonym import Orthonym
    eng = Orthonym()
    assert eng.name("CCO") == "ethanol"
    assert eng.name("CC(=O)Oc1ccccc1C(=O)O") == "2-(acetyloxy)benzoic acid"  # aspirin
    # 'adenine' is not a Blue Book name (0 hits); "the PIN is 7H-purine" (the Blue Book)
    assert eng.name("Nc1ncnc2nc[nH]c12") == "7H-purin-6-amine"
    assert eng.name("Cn1cnc2c(N)ncnc21") == "9-methyl-9H-purin-6-amine"


def test_acetyl_coa_end_to_end_best_effort_pipeline():
    # The full pipeline, best-effort tier, exactly as specified in the task
    # acceptance criteria: the molecule must round-trip, regardless of which
    # internal producer's candidate the composer ultimately selects.
    from orthonym import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.jvm_budget import jvm_slots

    eng = Orthonym(
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )
    name = eng.name(_ACETYL_COA)
    assert name, "acetyl-CoA must not abstain under best-effort"
    with jvm_slots(1, purpose="test-acetyl-coa-e2e"):
        rt = opsin_roundtrip_check(_ACETYL_COA, name)
    assert rt["passed"], rt


def test_phospho_arm_purine_end_to_end_best_effort_pipeline():
    # Team-lead acceptance case #2, via the real pipeline (see module
    # docstring for why this is not also asserted as a direct
    # name_substituted_purine call).
    from orthonym import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.jvm_budget import jvm_slots

    eng = Orthonym(
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )
    name = eng.name(_PHOSPHO_ARM_PURINE)
    assert name
    with jvm_slots(1, purpose="test-phospho-arm-e2e"):
        rt = opsin_roundtrip_check(_PHOSPHO_ARM_PURINE, name)
    assert rt["passed"], rt


def test_adenosine_5_phosphate_no_regression_best_effort_pipeline():
    # Named via the purine-as-SUBSTITUENT path (oxolane parent); must remain
    # RT-exact with best-effort flags on (no regression from this fix).
    from orthonym import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.jvm_budget import jvm_slots

    smi = "OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1O"
    eng = Orthonym(
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )
    name = eng.name(smi)
    assert name
    with jvm_slots(1, purpose="test-amp-regression"):
        rt = opsin_roundtrip_check(smi, name)
    assert rt["passed"], rt
