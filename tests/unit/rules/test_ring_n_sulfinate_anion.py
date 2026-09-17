"""
Task R3 (v51) — the ring-NITROGEN sulfinate / sulfonate ANION.

R2 made the NEUTRAL ring-N sulfinic / sulfonic acid nameable
(``piperidine-1-sulfinic acid``). The corresponding ANION should then fall out
of the charged pipeline exactly as the ring-CARBON anion already does
(``O=S([O-])C1CCNCC1`` -> ``piperidine-4-sulfinate``). It did not, because
``rules.ions.classify_anion`` gated the sulfonate / sulfinate acid-anion class
on a CARBON directly bonded to the sulfur (``nb_has_carbon``); a ring-N-anchored
S-oxoacid ``[O-]`` (S bonded to the ring N, no carbon on S) fell through to
``alkoxide`` and the whole molecule abstained.

R3 widens that classifier to admit an S-oxoacid ``[O-]`` whose sulfur bonds a
CARBON *or a RING NITROGEN* — mirroring R2's ring-N-only neutral widening — so
the ring-N anion reaches the same neutralize -> re-name -> re-suffix seam and is
named ``…-sulfinate`` / ``…-sulfonate``.

IUPAC 2013 rule (verbatim):
  * (the Blue Book) "Anions derived from acids": "The
    preferred IUPAC name of anions formed by the removal of a hydron from the
    chalcogen atom (O, S, Se, and Te) of an acid or peroxyacid characteristic
    group or functional parent compound is formed by replacing the 'ic acid' or
    'ous acid' ending of the acid name by 'ate' or 'ite', respectively."
    Example (the Blue Book): ``C6H5-SO2-O– benzenesulfonate (PIN)``.
  * The sulfonic / sulfinic acid suffix itself is / Table 3.3
    (the Blue Book) ``–SO2-OH sulfonic acid`` / ``–SO-OH sulfinic acid``,
    established for the ring-N parent by R2.
  * Acyclic sulfamic acid (H2N-SO2-OH, keeps its own preselected name;
    its N is NOT a ring atom, so the ring-N-only widening leaves it excluded
    (see ``test_scope_acyclic_sulfamate_excluded``).
"""

import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# (smiles, expected PIN) — the R3 ring-NITROGEN anion targets.
RING_N_ANION_TARGETS = [
    ("O=S([O-])N1CCCCC1", "piperidine-1-sulfinate"),
    ("O=S([O-])N1CCCC1", "pyrrolidine-1-sulfinate"),
    ("O=S(=O)([O-])N1CCCCC1", "piperidine-1-sulfonate"),
]

# Positive control — the ring-CARBON anion must stay working (unchanged path).
RING_C_ANION_CONTROL = [
    ("O=S([O-])C1CCNCC1", "piperidine-4-sulfinate"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", RING_N_ANION_TARGETS)
def test_ring_n_sulfur_oxoacid_anion_name(smiles, expected):
    """A ring-N sulfinate / sulfonate anion is named on its acid-anion suffix
    ."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", RING_C_ANION_CONTROL)
def test_ring_c_anion_control_unchanged(smiles, expected):
    """The ring-CARBON sulfinate anion (already-working path) must not regress."""
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize(
    "smiles,expected", RING_N_ANION_TARGETS + RING_C_ANION_CONTROL
)
def test_ring_n_anion_round_trip(smiles, expected):
    """Each emitted anion name OPSIN-parses back to the input ANION structure
    (full InChI, i.e. including the -1 charge layer) — 0-wrong."""
    result = opsin_roundtrip_check(smiles, expected)
    assert result["passed"], (
        f"{expected!r} did not round-trip to {smiles!r}: {result}"
    )


@pytest.mark.unit
def test_scope_acyclic_sulfamate_excluded():
    """The ring-N-only widening must NOT reclaim an ACYCLIC sulfamate
    (H2N-SO2-O-, P-67) as a ring/chain sulfonate — its N is not a ring atom.
    Whatever it emits, it is not a false ``…sulfonate`` / ``…sulfinate`` name."""
    out = name_compound("NS(=O)(=O)[O-]")
    assert "sulfonate" not in out and "sulfinate" not in out
