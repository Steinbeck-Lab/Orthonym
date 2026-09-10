"""D2: N-substituted acyclic diamines with numeric-superscript italic-N locants.

Gap-fix D2 /: the acyclic secondary/tertiary amine
handler _assemble_amine_name was a SINGLE-nitrogen handler. For a diamine
it walked only the first N and dropped the second N's substituent, producing
a WRONG structure that suppressed to 'unknown'. Fix generalizes it to
multi-N, mirroring the aromatic analog _name_substituted_benzenediamine.

a phase Thread A, the Blue Book): the N-substituent locant of a
SIMPLE polyamine (all principal amine N on the parent chain) is the NUMERIC
SUPERSCRIPT = the parent-hydride locant of the carbon the nitrogen attaches to
(flattened N1, N2, N3), NOT a bare prime (N, N'). the Blue Book gives
'N1-ethyl-N3-methylpropane-1,3-diamine' verbatim; the Blue Book proves even
ethane-1,2-diamine uses N1/N2. The old bare-primed forms below were RIGHT
molecule but non-PIN and are updated to the superscript PINs (each still
OPSIN round-trips to the same structure — a pure spelling-conformance move).

All expected PINs are OPSIN round-trip verified (style='pin').
"""
import pytest

from orthonym import name_compound


# ---- ACCEPTANCE: N-substituted acyclic di/poly-amines (was 'unknown') ----
# (the Blue Book): numeric-superscript italic-N locants.

DIAMINE_ACCEPTANCE = [
    ("CNCCNC", "N1,N2-dimethylethane-1,2-diamine"),
    ("CNCCNCC", "N1-ethyl-N2-methylethane-1,2-diamine"),
    ("CNCCCNC", "N1,N3-dimethylpropane-1,3-diamine"),
    ("CCNCCNCC", "N1,N2-diethylethane-1,2-diamine"),
    ("CN(C)CCN(C)C", "N1,N1,N2,N2-tetramethylethane-1,2-diamine"),
    ("CNCCCCNCCC", "N1-methyl-N4-propylbutane-1,4-diamine"),
    # the Blue Book verbatim PIN example (superscript flattened).
    ("CCNCCCNC", "N1-ethyl-N3-methylpropane-1,3-diamine"),
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


# ---- GUARDS (a phase Thread A): the superscript flip must NOT regress the
# COMPLEX (demoted-N / branch-amine) path, which already used superscripts, nor
# the aromatic diamines, nor the bare-primed MONONUCLEAR-parent name
# (N,N'-dinitromethanediamine, the Blue Book) -- which is built by a
# DIFFERENT handler entirely (_assemble_polyamine_name is never reached for it),
# so this is a cross-handler regression guard, NOT a check of the (currently
# unreachable) bare-prime branch inside _assemble_polyamine_name. ----

SUPERSCRIPT_FLIP_GUARDS = [
    ("NCCNCN", "N1-(aminomethyl)ethane-1,2-diamine"),          # complex/demoted
    ("CN(C)CCN(C)CCN",
     "N1-(2-aminoethyl)-N1,N2,N2-trimethylethane-1,2-diamine"),  # complex/demoted
    ("Nc1ccc(-c2ccc(N)cc2)cc1", "[1,1'-biphenyl]-4,4'-diamine"),
    ("Cc1cc(-c2ccc(N)c(C)c2)ccc1N",
     "3,3'-dimethyl[1,1'-biphenyl]-4,4'-diamine"),
    # Mononuclear parent (methanediamine): methane has no cited skeletal locant,
    # so N-locants stay bare-primed. Produced by a separate handler, not the
    # bare-prime branch in _assemble_polyamine_name (verified: that assembler is
    # not called for this input).
    ("O=[N+]([O-])NCN[N+](=O)[O-]", "N,N'-dinitromethanediamine"),
]


@pytest.mark.parametrize(
    "smiles, expected",
    SUPERSCRIPT_FLIP_GUARDS,
    ids=[c[0] for c in SUPERSCRIPT_FLIP_GUARDS],
)
def test_superscript_flip_does_not_regress(smiles, expected):
    """The simple-polyamine superscript flip leaves the complex-N, aromatic,
    and mononuclear-parent (bare-primed) paths byte-identical."""
    name = name_compound(smiles, style="pin")
    assert name == expected, (
        f"Regression: expected '{expected}' for {smiles}, got '{name}'"
    )


# ---- GEMINAL polyamine (both amines on ONE carbon): the Blue Book wants
# 'N3-ethyl-N'3-methylhexane-3,3-diamine' (geminal prime disambiguates the two
# N sharing carbon locant 3). The tag flip now emits the geminal prime, but the
# molecule still ABSTAINS: (1) the shared suffix generator collapses the
# duplicate principal-group locants [3,3] -> [3] via sorted(set(...)) at
# assembly/handlers/_handler_shared.py (so the suffix is 'hexan-3-amine', not
# 'hexane-3,3-diamine'), and (2) the geminal-prime tie-break is by atom index,
# not by substituent alphanumerical order (our order gives N'3-ethyl-N3-methyl).
# Fixing both is shared-machinery work with a wide blast radius (every geminal
# diol/diketone), tracked as the a phase Thread A NAMED BLOCKER. Until then the
# molecule fails closed (abstains) rather than emit a wrong structure -- 0-wrong
# intact. This xfail(strict) flags the day the blocker is resolved. ----

@pytest.mark.xfail(
    strict=True,
    reason="geminal reach blocked: suffix dedup collapses [3,3]->[3] "
           "(_handler_shared.py) + prime tie-break by atom index not alpha order",
)
def test_geminal_diamine_pin_blocked():
    """the Blue Book geminal PIN -- currently abstains (fail closed)."""
    name = name_compound("CCCC(CC)(NC)NCC", style="pin")
    assert name == "N3-ethyl-N'3-methylhexane-3,3-diamine"
