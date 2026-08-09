"""v30 breadth — partially-saturated (dihydro/tetrahydro) monocyclic heterocycle
bearing a principal-group SUFFIX.

Root cause (spy-verified + BB-derived): the parent stem string (saturation prefix +
indicated-H) was numbered PCG-BLIND, so its dihydro locants disagreed with the
suffix locant the substituted path appends — `3,6-dihydro-2H-1,4-thiazine` +
`-3-carboxylic acid` instead of the correct `5,6-dihydro-2H-1,4-thiazine-3-...`.
SELF-01 caught the mismatch and abstained (0-wrong held), so these molecules named
nothing.

Fix: add a principal-characteristic-group locant term to the two heterocyclic
hydro-name cascade keys, AFTER indicated hydrogen (b) and BEFORE hydro prefixes (e)
per P-14.4 (BlueBookV2.md:3219): (a) fixed -> (b) indicated-H -> (c) PCG/suffix ->
(e) hydro. Mirrors the carbocyclic sibling (partial_saturation.py:546). Additive:
`principal_group_atoms=None` (bare rings / substituents) leaves the key unchanged.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.unit


def _pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smi,expected", [
    # _mancude_hydro_name path (parents needing indicated hydrogen)
    ("O=C(O)C1=NCCSC1", "5,6-dihydro-2H-1,4-thiazine-3-carboxylic acid"),
    ("O=C(O)C1=NCCOC1", "5,6-dihydro-2H-1,4-oxazine-3-carboxylic acid"),
    # _aromatizable_hydro_name path (pyridine-type)
    ("O=C(O)C1=CCCCN1", "1,4,5,6-tetrahydropyridine-2-carboxylic acid"),
    ("O=C(O)C1=NCCCC1", "3,4,5,6-tetrahydropyridine-2-carboxylic acid"),
])
def test_dihydro_heterocycle_carboxylic_acid(smi, expected):
    assert _pin().name(smi) == expected


def test_dihydro_heterocycle_suffix_rt_exact():
    """Every emitted name round-trips (0-wrong): the fix converts abstentions to
    correct names, never to a wrong molecule."""
    import sys
    sys.path.insert(0, "scripts")
    from diagnose import diagnose
    smis = ["O=C(O)C1=NCCSC1", "O=C(O)C1=NCCOC1",
            "O=C(O)C1=CCCCN1", "O=C(O)C1=NCCCC1"]
    rows = diagnose(smis, style="pin", use_opsin=True)
    assert all(r.get("verdict") == "OK" for r in rows), [
        (r["smiles"], r.get("verdict"), r.get("name")) for r in rows]


@pytest.mark.parametrize("smi,expected", [
    # bare partially-saturated rings must stay byte-identical (PCG term is empty)
    ("C1=NCCSC1", "3,6-dihydro-2H-1,4-thiazine"),
    ("C1=NCCOC1", "3,6-dihydro-2H-1,4-oxazine"),
    ("C1C=CC=CN1", "1,2-dihydropyridine"),
    # saturated rings + their suffixes are on a different path, unaffected
    ("C1CSCCN1", "thiomorpholine"),
    ("O=C(O)C1CSCCN1", "thiomorpholine-3-carboxylic acid"),
    ("c1ccncc1", "pyridine"),
])
def test_no_regression_bare_and_saturated_rings(smi, expected):
    assert _pin().name(smi) == expected
