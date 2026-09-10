""": λ-convention on Hantzsch-Widman heteromonocycles.

All expected PINs OPSIN-2.9.0-verified 2026-07-09 (ASCII 'lambda' spelling,
RDKit-canonical round-trip equality with the input SMILES).
"""
import pytest
from orthonym.namer import name_compound

LAMBDA_HW_CASES = [
    # (SMILES, expected PIN) — BB examples verbatim
    ("[SH2]1C=CC=C1", "1H-1λ4-thiophene"),
    ("[IH]1CCCCC1", "1λ3-iodinane"),
    ("O1C=[PH2]C=C1", "1,3λ5-oxaphosphole"),
    ("S1=CN=CC=C1", "1λ4,3-thiazine"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", LAMBDA_HW_CASES)
def test_lambda_heteromonocycle_pin(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_standard_heteromonocycles_unchanged():
    # protects — recorded at HEAD 2026-07-09, all OPSIN-RT clean
    assert name_compound("c1ccsc1") == "thiophene"
    assert name_compound("C1CCOC1") == "oxolane"
    assert name_compound("c1ccncc1") == "pyridine"


@pytest.mark.unit
def test_partially_saturated_lambda_ring_is_named():
    """The hydro form of a lambda ring is NAMED, not refused.

    This assertion previously required an abstention, on the reading that
    "hydro-naming for lambda rings is unbuilt". That was a statement about
    the implementation, not about the Blue Book, which marks the construction
    (PIN): "Intramolecular amides of amino sulfinic acids."
    (the Blue Book) gives ``3,4,5,6-tetrahydro-1lambda4,2-thiazin-1-ol
    (PIN)`` at:33292. The producer was built; the expectation is stale.

    The value below is NOT taken from current output. Derived first:
    S is lambda-4 (two ring sigma bonds + 2 H), so all five ring positions are
    double-bond eligible and the mancude parent carries max-matching 2; the
    molecule has 1, hence 2*(2-1) = 2 hydro positions and 5 - 2*2 = 1
    indicated hydrogen. Sulfur takes locant 1, leaving two directions whose
    saturated sets are {1,2,3} and {1,4,5}; both put the indicated hydrogen at
    1, so (b) (:3246) ties and (e)(i) (:3288) decides on the lower
    hydro set, {2,3}. Order of citation is fixed by (:16880) -- "the
    'hydro' prefixes precede them".

    Both candidate spellings round-trip through OPSIN to this exact structure
    and both satisfy the hydro-count arithmetic, because they denote the SAME
    molecule -- so neither available oracle can choose between them and only
    the numbering rule can. That is precisely the spelling blind spot, which
    is why the derivation is written out here.
    """
    assert name_compound("[SH2]1CCC=C1") == "2,3-dihydro-1H-1λ4-thiophene"
