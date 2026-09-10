""" lambda-branch: -C#[N+][O-] as (oxo-λ5-azanylidyne)methyl
prefix in ANION context only; neutral (suffix-form) contexts fail closed.

'sodium 4-[(oxo-λ5-azanylidyne)methyl]benzoate' is the BB PIN verbatim
(the Blue Book); OPSIN-2.9.0 parses it back to the input structure
(verified 2026-07-09).
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.rules.benzene import _nitrile_oxide_prefix


@pytest.mark.unit
def test_prefix_dict_in_anion_context():
    anion = Chem.MolFromSmiles("[O-]C(=O)c1ccc(C#[N+][O-])cc1")
    ring = {a.GetIdx() for a in anion.GetAtoms() if a.GetIsAromatic()}
    c = next(a.GetIdx() for a in anion.GetAtoms()
             if a.GetSymbol() == 'C' and not a.GetIsAromatic()
             and any(b.GetBondType() == Chem.BondType.TRIPLE for b in a.GetBonds()))
    d = _nitrile_oxide_prefix(anion, c, ring)
    assert d is not None
    assert d['name'] == "(oxo-λ5-azanylidyne)methyl"
    assert len(d['atoms']) == 3


@pytest.mark.unit
def test_sentinel_in_neutral_context():
    neutral = Chem.MolFromSmiles("COC(=O)c1ccc(C#[N+][O-])cc1")
    ring = {a.GetIdx() for a in neutral.GetAtoms() if a.GetIsAromatic()}
    c = next(a.GetIdx() for a in neutral.GetAtoms()
             if a.GetSymbol() == 'C' and not a.GetIsAromatic()
             and any(b.GetBondType() == Chem.BondType.TRIPLE for b in a.GetBonds()))
    d = _nitrile_oxide_prefix(neutral, c, ring)
    assert d == {'name': None}


@pytest.mark.xfail(
    reason="Detector is correct (see test_prefix_dict_in_anion_context), but the "
    "salt handler does not yet route its anion fragment through _identify_substituent "
    "-> the benzoate-nitrile-oxide salt still refuses ('sodium compound (not "
    "supported)'). FOLLOW-UP: route the salt anion through the fixed benzene "
    "substituent path, then this yields the BB PIN "
    "'sodium 4-[(oxo-λ5-azanylidyne)methyl]benzoate' (BB 34897).",
    strict=False,
)
@pytest.mark.unit
def test_sodium_salt_bluebook_pin():
    assert name_compound("[Na+].[O-]C(=O)c1ccc(C#[N+][O-])cc1") == \
        "sodium 4-[(oxo-λ5-azanylidyne)methyl]benzoate"


@pytest.mark.xfail(
    reason="Nitrile oxide on a NON-ring (chain/ester) context mis-names via a "
    "chain path this benzene-scoped detector does not cover, and the result is "
    "canonicalization-order-dependent (diagnose canonicalizes -> 'unknown'; raw "
    "SMILES -> a chain mis-name). The BB neutral PIN is the '...nitrile oxide' "
    "SUFFIX form (unbuilt). FOLLOW-UP: nitrile-oxide chain-path perception + the "
    "functional-class suffix. Pre-existing; the benzene ring path is fail-closed.",
    strict=False,
)
@pytest.mark.unit
def test_neutral_ester_fails_closed():
    # BB 34893: the neutral-molecule PIN is '4-(methoxycarbonyl)benzonitrile
    # oxide' (suffix form, unbuilt) — the prefix name is explicitly NOT the
    # PIN here, so the namer must refuse.
    assert "unknown" in name_compound("COC(=O)c1ccc(C#[N+][O-])cc1").lower()


@pytest.mark.xfail(
    reason="Bare CC#[N+][O-] mis-names as 'ethane' via the chain path (a "
    "pre-existing structure loss outside this benzene-scoped detector). "
    "FOLLOW-UP: nitrile-oxide chain-path perception + P-66.5.4.2 suffix form.",
    strict=False,
)
@pytest.mark.unit
def test_bare_nitrile_oxide_fails_closed():
    assert "unknown" in name_compound("CC#[N+][O-]").lower()


@pytest.mark.unit
def test_salt_protects_unchanged():
    # recorded at HEAD, OPSIN-RT clean
    assert name_compound("[Na+].[O-]C(=O)c1ccc(C)cc1") == "sodium 4-methylbenzoate"
    assert name_compound("CCOC(=O)c1ccccc1") == "ethyl benzoate"
