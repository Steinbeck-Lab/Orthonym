"""Regression tests for the '_validate_anion_name'/'_validate_cation_name'
'methylidene' radical-leak guard (a phase review follow-up).

Root cause: the guard was a project-wide SUBSTRING check --

    if 'methylidene' in result:
        return ''

-- added defensively in Plan 17-06 (2026-02-05) against a radical name
"leaking" into ion naming. a trace (2026-08-17): the leak mechanism is real --
``_try_neutralize_and_name`` (the generic single-atom fallback both
``name_anion``/``name_cation`` call when no dedicated class matches) collapses
a genuine single-carbon ion down to a bare divalent-radical fragment and
names it via the general engine, which calls it 'methylidene' with NO ion
suffix attached at all (proven directly:
``_try_neutralize_and_name(Chem.MolFromSmiles('[CH3+]'))`` and
``_try_neutralize_and_name(Chem.MolFromSmiles('[CH-]'))`` both return the
bare string ``'methylidene'``). That bare, suffix-less descriptor is the
actual leak shape.

But the substring check ALSO fires on legitimate ylidene-owner names built by
producers such as the oxime-ether / glucosinolate '(...ylidene)amino'
substituent (e.g. 'sulfanylmethylideneamino sulfate') whenever the ylidene
owner happens to be exactly one backbone carbon ('methylidene', as opposed to
'ethylidene', 'propylidene',...). In every legitimate case 'methylidene' is
followed by MORE text (an 'amino'/'hydrazin...' linker word) because it names
a substituent attached to something else; in the leak case it is the bare
TERMINAL token of the whole name, with nothing after it.

Fix: reject only the bare-terminal shape (`_is_bare_methylidene_leak`), not
every occurrence of the substring.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.ions import (
    _validate_anion_name,
    _validate_cation_name,
    _try_neutralize_and_name,
)
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse
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
    "C[N+](C)(C)CCCC(=O)[O-]",
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



def _full_rt(smiles: str, name: str) -> bool:
    osmi = opsin_parse(name)
    if not osmi:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(osmi))


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# ---------------------------------------------------------------------------
# The fixed case: a legitimate 1-carbon ylidene-owner anion name must no
# longer be stripped to '' by the guard.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_one_carbon_ylidene_owner_anion_round_trips():
    smi = "SC=NOS(=O)(=O)[O-]"
    expected = "sulfanylmethylideneamino sulfate"
    # Verify RT validity independently before asserting the exact form.
    assert _full_rt(smi, expected), (
        f"expected name {expected!r} must OPSIN round-trip to {smi!r} "
        "before it is asserted as the emitted name"
    )


@pytest.mark.opsin_gate
def test_one_carbon_ylidene_owner_anion_integration(namer):
    # A compound prefix is enclosed, the Blue Book), as the two-carbon
    # sibling below already is.
    assert _dt_obj_name(namer, "SC=NOS(=O)(=O)[O-]") == "[(sulfanylmethylidene)amino] sulfate"


def test_validate_anion_name_accepts_one_carbon_ylidene_owner():
    mol = Chem.MolFromSmiles("SC=NOS(=O)(=O)[O-]")
    result = "sulfanylmethylideneamino sulfate"
    assert _validate_anion_name(mol, result) == result


# ---------------------------------------------------------------------------
# Regression: the 2-carbon ylidene owner never hit the substring guard in the
# first place ('ethylidene', not 'methylidene') -- must stay unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_two_carbon_ylidene_owner_unchanged(namer):
    assert _dt_obj_name(namer, "CC(S)=NOS(=O)(=O)[O-]") == "[(1-sulfanylethylidene)amino] sulfate"


# ---------------------------------------------------------------------------
# The radical leak the guard exists to protect against: a bare,
# suffix-less 'methylidene'/'...methylidene' descriptor must still be
# rejected as an ion name. The exact historical molecule that first
# triggered this (Plan 17-06, 2026-02-05) is not recoverable from the
# record, so this reproduces the mechanism directly against the real
# ``_try_neutralize_and_name`` helper (proven above to emit the bare string
# for a genuine one-carbon ion) and confirms the validators fail closed on
# it, exactly as the single-cation/-anion dispatch would pass it in.
# ---------------------------------------------------------------------------

def test_try_neutralize_and_name_can_produce_bare_methylidene_leak():
    # [CH3+] (methylium) and [CH-] both neutralize down to a lone:CH2
    # fragment, which the general engine names 'methylidene' with no ion
    # suffix -- this is the real shape name_cation/name_anion would
    # return to the validator if route_charged/retained-name lookup ever
    # declines a single-carbon ion of this shape.
    assert _try_neutralize_and_name(Chem.MolFromSmiles("[CH3+]")) == "methylidene"
    assert _try_neutralize_and_name(Chem.MolFromSmiles("[CH-]")) == "methylidene"


def test_bare_methylidene_leak_rejected_by_anion_guard():
    mol = Chem.MolFromSmiles("[CH-]")
    assert _validate_anion_name(mol, "methylidene") == ""


def test_bare_methylidene_leak_rejected_by_cation_guard():
    mol = Chem.MolFromSmiles("[CH3+]")
    assert _validate_cation_name(mol, "methylidene") == ""


def test_bare_substituted_methylidene_leak_rejected():
    # A substituted bare leak (e.g. a benzylic-type cation whose ylium
    # suffix was lost) -- 'methylidene' is still the TERMINAL token, so it
    # must still fail closed regardless of what precedes it.
    mol = Chem.MolFromSmiles("[CH3+]")
    assert _validate_anion_name(mol, "phenylmethylidene") == ""
    assert _validate_cation_name(mol, "phenylmethylidene") == ""
    assert _validate_anion_name(mol, "(phenyl)methylidene") == ""
    assert _validate_cation_name(mol, "(phenyl)methylidene") == ""


# ---------------------------------------------------------------------------
# The 7 named regressions: unrelated charged-name shapes, none containing
# 'methylidene', must be byte-identical to their pre-fix values.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
    ("[NH3+]C(CCC(=O)[O-])C(=O)[O-]", "2-azaniumylpentanedioate"),
    ("CCCCCCCCCCCCOS(=O)(=O)[O-]", "dodecyl sulfate"),
    ("C[N+]1=CC=CC=C1", "1-methylpyridin-1-ium"),
    ("CC(S)=NOS(=O)(=O)[O-]", "[(1-sulfanylethylidene)amino] sulfate"),
    # / substitutive '-bis(aminium)' PIN (the Blue Book,:42366)
    ("C[N+](C)(C)CCCCCC[N+](C)(C)C",
     "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)"),
    ("[O-]C(=O)CCC(=O)[O-]", "butanedioate"),
])
def test_seven_regressions_unchanged(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected
