"""Acylamino (-NH-C(=O)-R) SUBSTITUENT naming when a senior group makes the
ring/chain the parent (a phase residual / general-engine).

Root cause (measured): ``rules/polyfunctional.py::_name_ring_as_parent_polyfunctional``
Step 6 unconditionally added EVERY non-principal FG match's off-ring atoms to
``consumed_atoms`` -- including ``secondary_amide``/``tertiary_amide``, whose
``get_prefix`` is ``None`` (seniority.py:916-917, "Named via acylamino pathway
in universal pipeline") so Step 5 never actually emits a prefix for them. That
orphaned the acyl R-group carbon (its only path back to the ring runs through
the now-"parent" amide N/C=O atoms), so the universal substituent walker could
never discover the whole -NH-C(=O)-R branch; ``_verify_completeness``'s hard
assert on the orphaned atom was then silently swallowed by
``_integrate_universal_prefixes``'s bare ``except Exception: return ""``,
dropping the acetamido group from the assembled name entirely (an atom-drop
that then correctly vetoed, so the whole molecule abstained).

Fix: exclude ``secondary_amide``/``tertiary_amide`` from that unconditional
consumption so the universal pipeline (already wired to build
``acetamido``/``N-methylacetamido`` via ``_name_amino_branch`` ->
``linear_acyl_amido_prefix``) gets the chance to discover and name the branch.

RT-gated (`@pytest.mark.opsin_gate`); the top-level /OPSIN gate is the
0-wrong backstop.
"""
import pytest
from orthonym import Orthonym
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C[N+](C)(C)CCOS(=O)(=O)[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)
    return namer_obj.name_tiered(smiles)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- the goal molecules --------------------------------------------------
@pytest.mark.opsin_gate
def test_acetamido_on_ring_with_senior_acid(namer):
    # -NHC(=O)CH3 on a cyclohexane ring, carboxylic acid is the senior group
    # (parent = cyclohexane-1-carboxylic acid); the amido group must survive
    # as a '2-acetamido' substituent prefix, not be dropped.
    assert _dt_obj_name(namer, "CC(=O)NC1CCCCC1C(=O)O") == (
        "2-acetamidocyclohexane-1-carboxylic acid"
    )


@pytest.mark.opsin_gate
def test_acetamido_on_chain_with_senior_acid(namer):
    # -NHC(=O)CH3 on a chain, carboxylic acid is the senior group (parent =
    # acetic acid; the locant is omitted per -- acetic acid's C2 is
    # the only substitutable position). Already worked before this fix (the
    # chain path has its own correct consume-guard); asserted here as a
    # companion regression.
    assert _dt_obj_name(namer, "CC(=O)NCC(=O)O") == "acetamidoacetic acid"


# --- regression: amide-as-PRINCIPAL-group case must stay unchanged -------
@pytest.mark.opsin_gate
def test_amide_parent_unchanged(namer):
    # No senior group present -> the amide IS the principal group (parent),
    # NOT a substituent. Must stay 'N-cyclohexylacetamide', never become an
    # 'acetamido'-prefixed name.
    assert _dt_obj_name(namer, "CC(=O)NC1CCCCC1") == "N-cyclohexylacetamide"


# --- byte-identical regression list (unrelated substituent shapes) -------
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("OCCOC(=O)c1ccccc1", "2-hydroxyethyl benzoate"),
    ("Nc1ccccc1", "aniline"),
    ("NCCO", "2-aminoethan-1-ol"),
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(trimethylazaniumyl)ethyl sulfate"),
])
def test_byte_identical_regressions(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected
