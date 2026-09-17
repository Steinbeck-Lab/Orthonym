"""
Task R2 (v51) — sulfonic / sulfinic acid (and ring-N sulfonamide) SUFFIX on a
heterocyclic ring parent, ring-CARBON and ring-NITROGEN alike.

Before R2 the heterocycle namer wired only the carbon-anchored carb* suffixes
(``-carboxylic acid`` etc.) as ring suffixes. A sulfur-oxo-acid suffix is
S-anchored, so ``_identify_suffix_fg`` (which required a carbon anchor) dropped
it: even the ring-CARBON ``piperidine-4-sulfonic acid`` abstained, while the
carbocyclic ``cyclohexane-1-sulfonic acid`` named. The ring-NITROGEN forms
abstained additionally because the sulfonic/sulfinic perception SMARTS required
a carbon on sulfur (``$([SX4][#6])`` / ``$([SX3][#6])``), so a ring-N-attached
sulfonic/sulfinic acid was never even perceived as the acid.

IUPAC 2013 rule (verbatim):
  * (the Blue Book) "Substitutive nomenclature, suffix mode, for
    sulfonic, sulfinic, etc., acids": "Sulfonic, sulfinic, etc., acids are named
    substitutively by adding an appropriate suffix listed in Table 6.2 to the
    name of a parent hydride name." — the parent hydride may be a ring (C or a
    heteroatom); no carbon-attachment restriction appears in the rule text.
  * Table 6.2 (the Blue Book) "Suffixes and prefixes used to denote
    sulfur, selenium, and tellurium acids with chalcogen atoms directly linked
    to a parent": ``–SO2-OH → sulfonic acid`` (prefix ``sulfo``), ``–S(O)-OH →
    sulfinic acid`` (prefix ``sulfino``) — preselected suffixes.
  * The ring-N attachment is senior to the noncarbon-oxoacid re-framing exactly
    as the ring-N carboxylic acid is, the Blue Book; the Blue Book). The
    ``sulfamic acid`` counter-case is scoped to NONCARBON oxoacids, whose
    whole molecule has no carbon, so it never reaches a ring-N of a carbon
    parent hydride like piperidine — see the R2 report.
"""

import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# (smiles, expected PIN) — ring-CARBON sulfonic / sulfinic targets (G2).
RING_C_TARGETS = [
    ("OS(=O)(=O)C1CCNCC1", "piperidine-4-sulfonic acid"),
    ("OS(=O)C1CCNCC1", "piperidine-4-sulfinic acid"),
]

# (smiles, expected PIN) — ring-NITROGEN sulfonic / sulfinic / sulfonamide
# targets (G2 + the ring-N suffix path + the perception widening).
RING_N_TARGETS = [
    ("OS(=O)(=O)N1CCCCC1", "piperidine-1-sulfonic acid"),
    ("OS(=O)N1CCCCC1", "piperidine-1-sulfinic acid"),
    ("OS(=O)N1CCCC1", "pyrrolidine-1-sulfinic acid"),
    ("NS(=O)(=O)N1CCCCC1", "piperidine-1-sulfonamide"),
]

# Positive controls — the existing, correct paths MUST stay working.
CONTROLS = [
    # carbocyclic sulfonic (cycloalkane path, not the heterocycle namer)
    ("OS(=O)(=O)C1CCCCC1", "cyclohexane-1-sulfonic acid"),
    # chain sulfinic (C-anchored; the widening must not disturb it)
    ("CCS(=O)O", "ethanesulfinic acid"),
    # R1 ring-N carboxylic acid must be unaffected
    ("OC(=O)N1CCCC1", "pyrrolidine-1-carboxylic acid"),
    # ring-C carboxylic acid must be unaffected
    ("OC(=O)C1CCNCC1", "piperidine-4-carboxylic acid"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", RING_C_TARGETS + RING_N_TARGETS)
def test_ring_sulfur_oxoacid_suffix_name(smiles, expected):
    """A ring-attached sulfonic/sulfinic acid (or ring-N sulfonamide) is the
    ring principal-group suffix, Table 6.2)."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", CONTROLS)
def test_controls_unchanged(smiles, expected):
    """Carbocyclic / chain / R1 controls must not regress."""
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize(
    "smiles,expected", RING_C_TARGETS + RING_N_TARGETS + CONTROLS
)
def test_ring_sulfur_oxoacid_round_trip(smiles, expected):
    """Each emitted name OPSIN-parses back to the input structure (0-wrong)."""
    result = opsin_roundtrip_check(smiles, expected)
    assert result["passed"], (
        f"{expected!r} did not round-trip to {smiles!r}: {result}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Free NONCARBON oxoacids stay their own preselected names — the S has
        # only O (or an acyclic N) neighbours, so it is neither a C-acid nor a
        # ring-N acid. These must NOT become a sulfinic/sulfonic name.
        ("OS(=O)O", "sulfurous acid"),
        ("OS(=O)(=O)O", "sulfuric acid"),
        # sulfamic acid: acyclic H2N-SO2-OH — the N is not a ring atom, so the
        # sulfonic widening (ring-N only) leaves it excluded.
        ("NS(=O)(=O)O", "sulfamic acid"),
        # sulfinamides stay sulfinamides (no -OH on S; the [OX2H1] excludes them).
        ("CS(=O)N", "methanesulfinamide"),
    ],
)
def test_exclusions_hold(smiles, expected):
    """Free oxoacids / sulfamic / sulfinamide must NOT be reclaimed as a ring
    or chain sulfinic/sulfonic acid (0-wrong)."""
    assert name_compound(smiles) == expected
