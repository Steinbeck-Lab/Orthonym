"""A multiplied fused-ring prefix is named the same whatever order the input SMILES lists
its atoms in. The (d) 'bis before an attached-component fusion name' feature was
taken out of lane L2 (a side record keyed by the name text made di/bis depend on input
order); these rows pin main's single, order-independent spelling until that feature is
rebuilt keyed by structure. Each molecule: 8 fixed re-spellings, one name and one tier
(best-effort flags, the tier the 5-methyl row is only reachable at)."""
import pytest

from orthonym import Orthonym

ROWS = [
    # systematic_verified: the name breaks the PIN spelling rule ('di' before the
    # substituted component '5-methyl-1-benzofuran-2-yl'; 'bis', 'tris'... are the numerical
    # prefixes there, the Blue Book), which ``check_pin_spelling`` reports. The label used
    # to read pin_unverified because the strict-twin demotion ran before the spelling check
    # and hid it; the name is unchanged.
    ("di(5-methyl-1-benzofuran-2-yl)methanol", "systematic_verified", [
        "OC(c1cc2cc(C)ccc2o1)c1cc2cc(C)ccc2o1",
        'c1(C)ccc2oc(cc2c1)C(c1cc2c(o1)ccc(C)c2)O',
        'c1c2cc(C)ccc2oc1C(c1cc2c(o1)ccc(c2)C)O',
        'c1c2c(oc1C(c1oc3c(c1)cc(C)cc3)O)ccc(c2)C',
        'c1c2oc(cc2cc(C)c1)C(c1cc2cc(C)ccc2o1)O',
        'o1c2ccc(cc2cc1C(c1oc2c(cc(cc2)C)c1)O)C',
        'c12oc(C(O)c3cc4cc(ccc4o3)C)cc2cc(C)cc1',
        'c12cc(oc2ccc(C)c1)C(c1oc2ccc(C)cc2c1)O',
    ]),
    ("di(imidazo[1,2-a]pyridin-3-yl)methanol", "pin_verified", [
        "OC(c1cnc2ccccn12)c1cnc2ccccn12",
        'C(c1n2c(cccc2)nc1)(O)c1n2ccccc2nc1',
        'n1c2n(cccc2)c(C(c2n3ccccc3nc2)O)c1',
        'c1c2ncc(C(O)c3n4ccccc4nc3)n2ccc1',
        'n12ccccc1ncc2C(O)c1cnc2n1cccc2',
        'c1n2c(ccc1)ncc2C(c1n2c(nc1)cccc2)O',
        'C(O)(c1n2ccccc2nc1)c1n2c(nc1)cccc2',
        'c1c2ncc(n2ccc1)C(O)c1n2c(nc1)cccc2',
    ]),
    ("tri(1-benzofuran-2-yl)methanol", "pin_verified", [
        "OC(c1cc2ccccc2o1)(c1cc2ccccc2o1)c1cc2ccccc2o1",
        'o1c(C(O)(c2oc3ccccc3c2)c2cc3c(cccc3)o2)cc2c1cccc2',
        'c1cccc2c1oc(C(O)(c1oc3c(cccc3)c1)c1cc3c(cccc3)o1)c2',
        'c12oc(C(O)(c3oc4c(c3)cccc4)c3cc4c(o3)cccc4)cc1cccc2',
        'C(c1cc2ccccc2o1)(c1oc2ccccc2c1)(c1oc2ccccc2c1)O',
        'c12c(cccc2)cc(C(c2oc3ccccc3c2)(O)c2cc3ccccc3o2)o1',
        'c1c2c(oc(C(c3oc4c(c3)cccc4)(c3oc4ccccc4c3)O)c2)ccc1',
        'c1c(C(c2cc3ccccc3o2)(c2cc3c(cccc3)o2)O)oc2c1cccc2',
    ]),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("expected,tier,spellings", ROWS, ids=[r[0] for r in ROWS])
def test_a_multiplied_fused_prefix_is_named_the_same_in_every_spelling(
        expected, tier, spellings):
    namer = Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)
    rows = [namer.name_tiered(s) for s in spellings]
    got = {(r.get("name"), r.get("tier")) for r in rows}
    assert got == {(expected, tier)}, got
