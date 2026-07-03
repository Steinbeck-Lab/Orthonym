"""D2: N-substituted acyclic diamines with primed italic-N locants.

Gap-fix D2 (P-62.2.2 / P-16.3.3): the acyclic secondary/tertiary amine
handler _assemble_amine_name was a SINGLE-nitrogen handler. For a diamine
it walked only the first N and dropped the second N's substituent, producing
a WRONG structure that SELF-01 suppressed to 'unknown'. Fix generalizes it to
multi-N with primed italic-N locants (N, N', N''), mirroring the aromatic
analog _name_substituted_benzenediamine.

All expected PINs are OPSIN round-trip verified (style='pin').
"""
import pytest

from orthonym import name_compound


# ---- ACCEPTANCE: N-substituted acyclic di/poly-amines (was 'unknown') ----

DIAMINE_ACCEPTANCE = [
    ("CNCCNC", "N,N'-dimethylethane-1,2-diamine"),
    ("CNCCNCC", "N-ethyl-N'-methylethane-1,2-diamine"),
    ("CNCCCNC", "N,N'-dimethylpropane-1,3-diamine"),
    ("CCNCCNCC", "N,N'-diethylethane-1,2-diamine"),
    ("CN(C)CCN(C)C", "N,N,N',N'-tetramethylethane-1,2-diamine"),
    ("CNCCCCNCCC", "N-methyl-N'-propylbutane-1,4-diamine"),
]


@pytest.mark.parametrize(
    "smiles, expected",
    DIAMINE_ACCEPTANCE,
    ids=[c[0] for c in DIAMINE_ACCEPTANCE],
)
def test_nsubst_diamine_acceptance(smiles, expected):
    """N-substituted acyclic diamines emit primed italic-N locant PINs."""
    name = name_compound(smiles, style="pin")
    assert name == expected, (
        f"Expected '{expected}' for {smiles}, got '{name}'"
    )


# ---- GUARDS: single-N path MUST stay byte-identical ----

SINGLE_N_GUARDS = [
    ("CCNCC", "N-ethylethanamine"),
    ("CN(C)C", "N,N-dimethylmethanamine"),
    ("CCCNCCC", "N-propylpropan-1-amine"),
]


@pytest.mark.parametrize(
    "smiles, expected",
    SINGLE_N_GUARDS,
    ids=[c[0] for c in SINGLE_N_GUARDS],
)
def test_single_n_amine_unchanged(smiles, expected):
    """Single-N secondary/tertiary amines are unchanged by the diamine fix."""
    name = name_compound(smiles, style="pin")
    assert name == expected, (
        f"Regression: expected '{expected}' for {smiles}, got '{name}'"
    )


# ---- GUARDS: primary diamines do NOT reach this handler ----

PRIMARY_DIAMINE_GUARDS = [
    ("NCCN", "ethane-1,2-diamine"),
    ("NCCCN", "propane-1,3-diamine"),
]


@pytest.mark.parametrize(
    "smiles, expected",
    PRIMARY_DIAMINE_GUARDS,
    ids=[c[0] for c in PRIMARY_DIAMINE_GUARDS],
)
def test_primary_diamine_unchanged(smiles, expected):
    """Primary diamines classify as primary_amine and are unaffected."""
    name = name_compound(smiles, style="pin")
    assert name == expected, (
        f"Regression: expected '{expected}' for {smiles}, got '{name}'"
    )
