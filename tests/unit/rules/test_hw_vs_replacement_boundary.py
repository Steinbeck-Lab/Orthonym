"""The Hantzsch-Widman / skeletal-replacement ring-size boundary.

Blue Book "Heteromonocyclic hydrides named by skeletal replacement
('a') nomenclature" (``the Blue Book Blue Book``) opens with the
decisive sentence:

    "Mancude and saturated heteromonocyclic compounds with up to and
    including ten ring members are named by the extended Hantzsch-Widman
    system (see. For monocyclic rings with eleven and more ring
    members, skeletal replacement ('a') nomenclature (see is used
    for the fully saturated or fully unsaturated compounds ([n]annulenes)."

The boundary is therefore RING SIZE ALONE (<=10 -> HW, >=11 -> replacement).
It is *not* conditioned on saturation. Confirmed independently by
 (``:23442``) and (``:23682``).

Partially saturated <=10 rings are expressed as hydro prefixes on the HW
mancude parent -- "The prefix 'hydro'" (``:16906``): "'Hydro'
prefixes are used to modify the degree of hydrogenation of monocyclic
mancude compounds having retained or systematic names", whose own PIN
examples include ``4,5,6,7-tetrahydro-1,4-thiazepine`` and
``2,7-dihydro-1H-azepine``.

Regression guarded here: unsaturated 7- and 8-membered heteromonocycles
were routed to skeletal replacement (``1-azacyclohepta-2,4,6-triene``)
instead of Hantzsch-Widman (``1H-azepine``).

Assertions are made at the PRODUCER (``name_heterocycle``) and at the
ROUTER (``try_skeletal_replacement_name``) as well as end-to-end, so that a
green result cannot be manufactured by the OPSIN validity gate.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.heterocycles import name_heterocycle
from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES: {smiles}"
    return mol


def _first_ring(mol):
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) == 1
    return list(rings[0])


# ---------------------------------------------------------------------------
# The router must hand every <=10-membered heteromonocycle to the HW namer,
# saturated or not. This is the assertion that was RED before the fix.
# ---------------------------------------------------------------------------

# (smiles, why)
HW_TERRITORY = [
    # --- saturated 7-ring (already worked; controls) ---
    ("C1CCNCCC1", "saturated 7-ring, 1 N"),
    ("C1CCOCCC1", "saturated 7-ring, 1 O"),
    ("C1CCSCCC1", "saturated 7-ring, 1 S"),
    ("O1CCNCCC1", "saturated 7-ring, O+N"),
    ("S1CCNCCC1", "saturated 7-ring, S+N"),
    ("N1CCNCCC1", "saturated 7-ring, 2 N"),
    # --- mancude 7-ring (was broken) ---
    ("N1C=CC=CC=C1", "mancude 7-ring, 1 N"),
    ("O1C=CC=CC=C1", "mancude 7-ring, 1 O"),
    ("S1C=CC=CC=C1", "mancude 7-ring, 1 S"),
    ("S1C=CN=CC=C1", "mancude 7-ring, S+N"),
    # --- partially saturated 7-ring (was broken) ---
    ("N1CCC=CC=C1", "partly saturated 7-ring, 1 N"),
    ("O1CCC=CC=C1", "partly saturated 7-ring, 1 O"),
    ("S1C=CNCCC1", "partly saturated 7-ring, S+N"),
    ("N1CC=CC=CC1", "partly saturated 7-ring, 1 N (BB PIN example)"),
    ("N1=CCCCC=C1", "partly saturated 7-ring, 1 N (BB PIN example)"),
    # --- 8-ring, saturated and unsaturated (unsaturated was broken) ---
    ("C1CCCNCCC1", "saturated 8-ring, 1 N"),
    ("C1CCCOCCC1", "saturated 8-ring, 1 O"),
    ("N1=CC=CC=CC=C1", "mancude 8-ring, 1 N"),
    ("O1CC=CC=CC=C1", "mancude 8-ring, 1 O"),
    # --- 9- and 10-ring (these only worked by accident: RDKit called them
    # aromatic, so a separate has_aromatic gate rescued them) ---
    ("N1C=CC=CC=CC=C1", "mancude 9-ring, 1 N"),
    ("N1=CC=CC=CC=CC=C1", "mancude 10-ring, 1 N"),
    ("O1CC=CC=CC=CC=C1", "partly saturated 10-ring, 1 O (non-aromatic)"),
    ("C1CCCCNCCC1", "saturated 9-ring, 1 N"),
    ("C1CCCCNCCCC1", "saturated 10-ring, 1 N"),
]


@pytest.mark.parametrize("smiles,why", HW_TERRITORY)
def test_router_yields_ring_of_ten_or_fewer_to_hantzsch_widman(smiles, why):
    """: <=10 ring members belong to HW, saturated or not."""
    assert try_skeletal_replacement_name(_mol(smiles)) is None, (
        f"{smiles} ({why}) was claimed by skeletal replacement; "
        "P-22.2.3 reserves 'a' nomenclature for rings of eleven or more"
    )


# ---------------------------------------------------------------------------
# The other side of the boundary must NOT move. Without these, deleting the
# ring-size test entirely would leave the suite green.
# ---------------------------------------------------------------------------

REPLACEMENT_TERRITORY = [
    ("C1CCCCCCCCCCO1", "saturated 12-ring, 1 O"),
    ("N1=CC=CC=CC=CC=CC=C1", "mancude 12-ring, 1 N"),
    ("N1C=CC=CC=CC=CC=C1", "mancude 11-ring, 1 N"),
]


@pytest.mark.parametrize("smiles,why", REPLACEMENT_TERRITORY)
def test_router_keeps_rings_of_eleven_or_more(smiles, why):
    """ /: >10 ring atoms stay with 'a' nomenclature."""
    name = try_skeletal_replacement_name(_mol(smiles))
    assert name is not None, f"{smiles} ({why}) lost its replacement name"
    assert "cyclo" in name, f"{smiles} -> {name!r} is not a cyclic replacement name"


# ---------------------------------------------------------------------------
# Producer-level assertions: the exact PIN strings.
# ---------------------------------------------------------------------------

# (smiles, expected PIN, provenance)
PRODUCER_EXPECTATIONS = [
    # controls that already worked and must stay byte-identical
    ("C1CCNCCC1", "azepane", "HW Table 2.5 stem 'epane'"),
    ("C1CCOCCC1", "oxepane", "HW Table 2.5 stem 'epane'"),
    ("C1CCSCCC1", "thiepane", "HW Table 2.5 stem 'epane'"),
    ("O1CCNCCC1", "1,4-oxazepane", "P-22.2.2.1.3 citation order O before N"),
    ("S1CCNCCC1", "1,4-thiazepane", "BB PIN, :16962"),
    ("N1CCNCCC1", "1,4-diazepane", "P-22.2.2.1.2"),
    # the defect rows
    ("N1C=CC=CC=C1", "1H-azepine", "BB PIN, :7573 '1H-azepine (PIN, P-22.2.2.1.4)'"),
    ("O1C=CC=CC=C1", "oxepine", "HW 'epine'; locant omitted per P-22.2.2.1.7"),
    ("S1C=CC=CC=C1", "thiepine", "BB :8476 'thiepine (not 1-thiepine)'"),
    ("S1C=CN=CC=C1", "1,4-thiazepine", "BB PIN, :16960"),
    ("N1CCC=CC=C1", "2,3-dihydro-1H-azepine", "P-31.2.3.1 hydro on mancude parent"),
    ("O1CCC=CC=C1", "2,3-dihydrooxepine", "P-31.2.3.1 hydro on mancude parent"),
    ("S1C=CNCCC1", "4,5,6,7-tetrahydro-1,4-thiazepine", "BB PIN, :16918"),
    ("N1CC=CC=CC1", "2,7-dihydro-1H-azepine", "BB PIN, :16920"),
    ("N1=CCCCC=C1", "4,5-dihydro-3H-azepine", "BB PIN, :16888"),
    # 8-ring
    ("C1CCCNCCC1", "azocane", "HW Table 2.5 stem 'ocane'"),
    ("C1CCCOCCC1", "oxocane", "HW Table 2.5 stem 'ocane'"),
    ("N1=CC=CC=CC=C1", "azocine", "HW Table 2.5 stem 'ocine'; no indicated H"),
    ("O1CC=CC=CC=C1", "2H-oxocine", "P-22.2.2.1.4 indicated H at lowest locant"),
    # 9/10-ring
    ("N1C=CC=CC=CC=C1", "1H-azonine", "HW Table 2.5 stem 'onine'"),
    ("N1=CC=CC=CC=CC=C1", "azecine", "HW Table 2.5 stem 'ecine'"),
    ("O1CC=CC=CC=CC=C1", "2H-oxecine", "HW 'ecine' + P-22.2.2.1.4"),
]


@pytest.mark.parametrize("smiles,expected,provenance", PRODUCER_EXPECTATIONS)
def test_heterocycle_producer_emits_pin(smiles, expected, provenance):
    """Assert on the generator itself, not on the OPSIN gate's opinion."""
    mol = _mol(smiles)
    assert name_heterocycle(mol, _first_ring(mol)) == expected, provenance


# ---------------------------------------------------------------------------
# End-to-end: what the user actually gets.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles,expected,provenance", PRODUCER_EXPECTATIONS)
def test_whole_molecule_emits_pin(smiles, expected, provenance):
    assert name_compound(smiles) == expected, provenance


def test_no_seven_or_eight_ring_gets_a_replacement_name():
    """The defect's own signature: no 'aza/oxa/thia-cyclohepta|cycloocta'."""
    for smiles, _why in HW_TERRITORY:
        name = name_compound(smiles)
        assert name is not None, f"{smiles} abstained"
        for bad in ("cyclohepta", "cycloocta", "cyclonona", "cyclodeca",
                    "cycloheptan", "cyclooctan"):
            assert bad not in name, (
                f"{smiles} -> {name!r}: skeletal replacement leaked into "
                "Hantzsch-Widman territory (P-22.2.3)"
            )
