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
    assert namer.name("SC=NOS(=O)(=O)[O-]") == "sulfanylmethylideneamino sulfate"


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
    assert namer.name("CC(S)=NOS(=O)(=O)[O-]") == "[(1-sulfanylethylidene)amino] sulfate"


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
    ("C[N+](C)(C)CCCCCC[N+](C)(C)C", "hexane-1,6-diylbis(trimethylazanium)"),
    ("[O-]C(=O)CCC(=O)[O-]", "butanedioate"),
])
def test_seven_regressions_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected
