"""
Wave2 Tier 5c — acyl branch as prefix on a senior benzene parent
(P-66.6.3: ketones expressed as acyl prefixes when a senior characteristic
group takes the suffix; retained acyl prefixes P-66.6.1).

Reproduce-first: 4-acetylbenzoic acid & co. were fail-closed 'unknown' at
HEAD (the benzene path's `_identify_substituent` had amido and acyloxy
recognizers but no ketone-acyl branch, and the generic fallback's carbonyl
guard correctly blocks carbonyl carbons).  benzanilide -> N-phenylbenzamide
was ALREADY CORRECT (stale plan claim).

Fix (scoped, fail-closed): a new acyl-branch recognizer in benzene's
`_identify_substituent` emitting retained/systematic acyl prefixes
(acetyl / propanoyl / butanoyl / benzoyl), gated on a senior suffix FG
existing on ANOTHER ring branch — so acetophenone-class molecules (ketone
would be the PCG) never see it and keep their chain-parent PINs.
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CC(=O)c1ccc(C(=O)O)cc1", "4-acetylbenzoic acid"),
        ("CC(=O)c1cccc(C(=O)O)c1", "3-acetylbenzoic acid"),
        ("CC(=O)c1ccc(C(N)=O)cc1", "4-acetylbenzamide"),
        ("CCC(=O)c1ccc(C(=O)O)cc1", "4-propanoylbenzoic acid"),
        ("O=C(c1ccccc1)c1ccc(C(=O)O)cc1", "4-benzoylbenzoic acid"),
        ("CC(=O)c1ccc(C#N)cc1", "4-acetylbenzonitrile"),
        ("CC(=O)c1ccc(C=O)cc1", "4-acetylbenzaldehyde"),
        ("CC(=O)c1ccc(C(=O)O)cc1C", "4-acetyl-3-methylbenzoic acid"),
    ],
)
def test_tier5c_acyl_prefix_on_senior_parent(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.fixture
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,why",
    [
        ("OC(=O)c1ccc(C(=O)c2ccc(C)cc2)cc1",
         "substituted-aryl acyl (4-methylbenzoyl) — fail closed, deferred"),
        ("CC(=O)c1ccc(S(=O)(=O)O)cc1",
         "S-suffix gate excluded: the sulfo assembly drops locants with an "
         "acyl prefix — stays on the HEAD fail-closed path"),
    ],
)
def test_tier5c_fail_closed(_validity_gate_on, smiles, why):
    # production (validity gate ON) must not emit a name for these shapes
    assert "unknown" in name_compound(smiles), why


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # the PCG-ketone class must keep its chain-parent PIN (T1d gold)
        ("CC(=O)c1ccccc1", "1-phenylethan-1-one"),
        # keto-acid chain controls (oxo prefix, not acetyl)
        ("CC(=O)CC(=O)O", "3-oxobutanoic acid"),
        ("CC(=O)CCC(=O)O", "4-oxopentanoic acid"),
        # amido path (T1c) untouched
        ("CC(=O)Nc1ccc(C(=O)O)cc1", "4-acetamidobenzoic acid"),
        # already-healed rows named in the master plan's success criteria
        ("COc1ccc(Nc2ccccc2)cc1", "4-methoxy-N-phenylaniline"),
        ("O=C(Nc1ccccc1)c1ccccc1", "N-phenylbenzamide"),
        # retained aryl ketone with its own name must not flip
        ("CC(=O)c1ccc(O)cc1", "4-hydroxyacetophenone"),
    ],
)
def test_tier5c_regression_controls(smiles, expected):
    assert name_compound(smiles) == expected
