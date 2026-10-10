""" — carbon-free P-oxo substituent fragment rendered as a prefix.

A phosphono / phosphonato group that must be cited as a NESTED substituent
prefix was DROPPED because the bare P-oxo fragment ``O=P(O)O`` has no carbon and
the whole-molecule namer rejects it as ``inorganic compound (not supported)``.
Two substituent chokepoints delegated the fragment to that reject:
``assembly/substituent_naming.py::name_substituent_fragment``  and
``assembly/substituent_enumerator.py::name_substituent`` (P-rooted block). Both
now dispatch a carbon-free P-oxo fragment to the RETAINED prefixes
(``phosphono`` / ``phosphonato``, both RT-proven) via
``rules/phosphorus.carbon_free_phospho_prefix`` BEFORE the inorganic reject.

Measured deliverable (HEAD A/B, gate-ON): a nicotinic-acid mononucleotide-class
molecule goes abstain -> RT-valid emit at the DEFAULT tier. The bis-nucleotide
and medronic-acid witnesses in the original brief need an ADDITIONAL composition
step (best-effort multiplicative / phospho-as-prefix backbone) that is a separate
lever; here they are pinned only under the 0-wrong contract (abstain OR RT-valid,
never a wrong molecule).

Every asserted name is RT-verified (name -> OPSIN -> InChIKey == input). The
OPSIN validity gate is OFF by default in tests (conftest documents this
"green-but-blind" trap): with it off a wrong-molecule candidate like
'methanephosphonic acid' leaks for medronic. These tests assert PRODUCTION
emission and the 0-wrong contract, so they MUST run gate-ON.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.rules.phosphorus import carbon_free_phospho_prefix
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
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
    "OCCOP(=O)(O)O",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


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


pytestmark = pytest.mark.opsin_gate

UNK = "unknown organic compound"


def _rt_ok(smiles: str, name: str) -> bool:
    return bool(opsin_roundtrip_check(smiles, name).get("passed"))


def _phospho_frag(smiles: str):
    """Return (mol, [P + its O neighbours], P_idx) for a bare P-oxo molecule."""
    mol = Chem.MolFromSmiles(smiles)
    p = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'P'][0]
    frag = [p] + [n.GetIdx() for n in mol.GetAtomWithIdx(p).GetNeighbors()
                  if n.GetSymbol() == 'O']
    return mol, frag, p


# ---------------------------------------------------------------------------
# Witnesses
# ---------------------------------------------------------------------------

# MEASURED default-tier breadth win (abstain -> RT-valid emit with the fix).
# Nicotinic acid mononucleotide-class (charged pyridinium nucleotide).
PYRIDINIUM_NT = "O=C(O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c1"
# 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value: the
# oxolanyl prefix holds its stereodescriptor parentheses (counted,,
# the Blue Book) AND its own '[(phosphonooxy)methyl]', so it is enclosed
# in braces by the order "{[({})]}" (:7446), as in
# '10-{[(3S)-1-phosphabicyclo[2.2.2]octan-3-yl]methyl}-10H-phenoxazine (PIN)'
# (:7489); the old '[' put a bracket directly around a bracket. OPSIN RT exact.
PYRIDINIUM_NT_NAME = (
    "3-carboxy-1-{(2R,3R,4S,5R)-3,4-dihydroxy-5-[(phosphonooxy)methyl]"
    "oxolan-2-yl}pyridin-1-ium"
)

# Original brief witnesses — need an additional composition step; pinned here
# only under the 0-wrong contract (abstain OR RT-valid, never wrong).
BIS_NUCLEOTIDE = (
    "N=c1c2ncn([C@@H]3O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]3O)c2ncn1"
    "[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O"
)
MEDRONIC = "OP(=O)(O)CP(=O)(O)O"


# ---------------------------------------------------------------------------
# 1. Measured default-tier breadth win: abstain -> RT-verified emit
# ---------------------------------------------------------------------------

@pytest.mark.roundtrip
def test_pyridinium_nucleotide_abstain_to_emit():
    out = _dt_name_compound(PYRIDINIUM_NT)
    assert out not in (None, UNK), f"pyridinium nucleotide still abstains: {out!r}"
    assert _rt_ok(PYRIDINIUM_NT, out), f"emitted name is not RT-valid: {out!r}"


@pytest.mark.roundtrip
def test_pyridinium_nucleotide_pinned_name():
    assert _rt_ok(PYRIDINIUM_NT, PYRIDINIUM_NT_NAME)
    assert _dt_name_compound(PYRIDINIUM_NT) == PYRIDINIUM_NT_NAME


# ---------------------------------------------------------------------------
# 2. Direct fragment rendering — the fix at both chokepoints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,prefix", [
    ("CP(=O)(O)O", "phosphono"),          # neutral -P(=O)(OH)2
    ("P(=O)([O-])([O-])C", "phosphonato"),  # di-anion -P(=O)(O-)2
])
def test_carbon_free_phospho_prefix_renders(smiles, prefix):
    mol, frag, p = _phospho_frag(smiles)
    assert carbon_free_phospho_prefix(mol, frag, p) == prefix
    #...and the same fragment routed through the enumerator chokepoint.
    assert name_substituent(mol, frag, p) == prefix


@pytest.mark.parametrize("smiles", [
    "CPC",          # phosphine (no P=O)
    "OP(O)O",       # phosphorous acid (no P=O)
    "CP(=O)(O)OC",  # phosphonate mono-ester (an -O-C bridge, not a bare acid)
    "[PH](=O)(O)O",  # P-H present (phosphonic acid parent, not a -yl prefix)
])
def test_carbon_free_phospho_prefix_fail_closed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        pytest.skip("unparseable probe SMILES")
    ps = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'P']
    if not ps:
        pytest.skip("no P")
    p = ps[0]
    frag = [p] + [n.GetIdx() for n in mol.GetAtomWithIdx(p).GetNeighbors()
                  if n.GetSymbol() == 'O']
    assert carbon_free_phospho_prefix(mol, frag, p) is None


# ---------------------------------------------------------------------------
# 3. 0-wrong contract: emit an RT-valid name OR abstain — never a wrong molecule
# ---------------------------------------------------------------------------

@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [
    PYRIDINIUM_NT,
    BIS_NUCLEOTIDE,          # brief witness — needs composition; must not go wrong
    MEDRONIC,                # brief witness — needs composition; must not go wrong
    "OCCCP(=O)(O)O",
    "OP(=O)(O)CCP(=O)(O)O",
    "OP(=O)(O)CCCP(=O)(O)O",
])
def test_never_wrong(smiles):
    out = _dt_name_compound(smiles)
    if out in (None, UNK):
        return  # clean abstain preserves 0-wrong
    assert _rt_ok(smiles, out), f"0-wrong violated: {out!r} for {smiles}"


# ---------------------------------------------------------------------------
# 4. Spelling authority: the target PINs describe their molecule (RT-verified)
# ---------------------------------------------------------------------------

@pytest.mark.roundtrip
def test_target_pins_are_rt_valid():
    # medronic acid PIN candidates diphosphonic / phosphono form).
    assert _rt_ok(MEDRONIC, "(phosphonomethyl)phosphonic acid")
    assert _rt_ok(MEDRONIC, "methylenediphosphonic acid")
    # bis-nucleotide best-effort target (needs general-fallback tier to emit).
    assert _rt_ok(
        BIS_NUCLEOTIDE,
        "1,9-di{(2R,3R,4S,5R)-3,4-dihydroxy-5-[(phosphonooxy)methyl]"
        "oxolan-2-yl}6-imino-9H-purine",
    )


# ---------------------------------------------------------------------------
# 5. Determinism (2+ SMILES orders yield one name)
# ---------------------------------------------------------------------------

@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [PYRIDINIUM_NT, MEDRONIC])
def test_determinism(smiles):
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    names = {_dt_name_compound(smiles), _dt_name_compound(canon)}
    assert len(names) == 1, f"non-deterministic across orders: {names}"


# ---------------------------------------------------------------------------
# 6. Byte-identity controls: currently-naming molecules MUST NOT regress
# ---------------------------------------------------------------------------

@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected", [
    ("COP(=O)(O)O", "methyl dihydrogen phosphate"),
    ("OP(=O)(O)OCC", "ethyl dihydrogen phosphate"),
    ("CP(=O)(O)O", "methylphosphonic acid"),
    # leads program L4, item 30b: (the Blue Book) 'ethylphosphonic acid (PIN)
    # (not ethanephosphonic acid)' (:35461): the acid is a functional parent, not a suffix; the
    # old string was '2-hydroxyethane-1-phosphonic acid' (OPSIN reads both to one InChIKey).
    ("OCCP(=O)(O)O", "(2-hydroxyethyl)phosphonic acid"),
    ("OCCOP(=O)(O)O", "2-(phosphonooxy)ethan-1-ol"),
    ("OC(=O)CP(=O)(O)O", "phosphonoacetic acid"),
    ("c1ccccc1CP(=O)(O)O", "benzylphosphonic acid"),
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
])
def test_byte_identity_controls(smiles, expected):
    assert _dt_name_compound(smiles) == expected
