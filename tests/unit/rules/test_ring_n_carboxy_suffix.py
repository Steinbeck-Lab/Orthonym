"""
Task R1 (v51) — principal-group carboxy / carboxamide SUFFIX on a ring NITROGEN.

The heterocycle namer already attaches a carboxylic-acid / carboxamide suffix to
a ring CARBON (``piperidine-4-carboxylic acid`` works), but a ring-N attachment
abstained: perception claimed the group as ``carbamic_acid`` / ``urea`` (which
open the ring), and ``get_heterocycle_substituents`` skipped every acyl group on
a ring nitrogen (task 9) so the suffix was dropped.

IUPAC 2013 rule (verbatim):
  * (the Blue Book) "Substitutive nomenclature: 'carboxylic
    acid'": "The carboxy group can be attached to any atom, carbon or heteroatom,
    of any parent hydride."
  * (the Blue Book): "Carboxy groups attached to cyclic parent
    hydrides or heteroacyclic parent hydrides are always named by using the suffix
    'carboxylic acid'." — verbatim example ``pyrrolidine-1-carboxylic acid (PIN)``
    (the carboxy sits on the ring N, locant 1).
  * Seniority over the carbonic-acid (carbamic / urea) re-framing —
    the Blue Book: ``tetraazane-1-carboxylic acid (PIN) [not
    (triazan-1-yl)carbamic acid; the carboxylic acid is senior to the carbonic
    acid derivative]``.
  * Ring-N carboxamide — the Blue Book: ``piperidine-1-carboxamide (PIN)``.
"""

import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# (smiles, expected PIN) — the ring-N carboxy / carboxamide targets.
RING_N_TARGETS = [
    # pyrrolidine-1-carboxylic acid (PIN) — verbatim, the Blue Book
    ("OC(=O)N1CCCC1", "pyrrolidine-1-carboxylic acid"),
    # piperidine-1-carboxamide (PIN) — verbatim example, the Blue Book
    ("NC(=O)N1CCCCC1", "piperidine-1-carboxamide"),
    # piperidine-1-carboxylic acid (parallel to pyrrolidine-1-carboxylic acid)
    ("OC(=O)N1CCCCC1", "piperidine-1-carboxylic acid"),
    # aromatic ring N — engine emits the 1H-indicated form (engine-consistent PIN:
    # 1H distinguishes the N-substituted pyrrole tautomer, as for 1-methyl-1H-pyrrole).
    # OPSIN round-trips both '1H-pyrrole-1-carboxylic acid' and the bare form to the
    # same structure, so the 1H- spelling is correct and RT-verified.
    ("OC(=O)n1cccc1", "1H-pyrrole-1-carboxylic acid"),
]

# Ring-CARBON positive controls — the existing, correct path MUST stay working.
RING_C_CONTROLS = [
    ("OC(=O)C1CCNCC1", "piperidine-4-carboxylic acid"),
    ("NC(=O)C1CCNCC1", "piperidine-4-carboxamide"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", RING_N_TARGETS)
def test_ring_n_carboxy_suffix_name(smiles, expected):
    """The ring-N carboxy / carboxamide is the ring-N principal-group suffix."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", RING_C_CONTROLS)
def test_ring_c_controls_unchanged(smiles, expected):
    """Ring-CARBON attachment must not regress."""
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", RING_N_TARGETS + RING_C_CONTROLS)
def test_ring_n_targets_round_trip(smiles, expected):
    """Each emitted name OPSIN-parses back to the input structure (0-wrong)."""
    result = opsin_roundtrip_check(smiles, expected)
    assert result["passed"], (
        f"{expected!r} did not round-trip to {smiles!r}: {result}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Acyclic carbamic acids stay carbamic (N NOT a ring atom) — the reframing
        # is correct here; only a RING nitrogen is senior to it (the Blue Book).
        ("NC(=O)O", "carbamic acid"),
        ("CNC(=O)O", "N-methylcarbamic acid"),
        ("CN(C)C(=O)O", "N,N-dimethylcarbamic acid"),
        # Acyclic ureas stay urea (neither N is a ring atom).
        ("NC(=O)N", "urea"),
        ("CNC(=O)NC", "N,N'-dimethylurea"),
        # phenylurea: the anilino N is exocyclic to benzene (not a ring atom).
        ("NC(=O)Nc1ccccc1", "phenylurea"),
        # task 9 control: a true N-acyl group on a ring N (acetyl, not a
        # carboxy/carboxamide) must NOT become a ring-N suffix — the amide/ketone
        # machinery still names it.
        ("CC(=O)N1CCCC1", "1-(pyrrolidin-1-yl)ethan-1-one"),
    ],
)
def test_ring_n_carboxy_does_not_disturb_carbamic_urea_or_acyl(smiles, expected):
    assert name_compound(smiles) == expected
