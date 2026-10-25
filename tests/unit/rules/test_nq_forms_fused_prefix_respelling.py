"""A multiplied fused-ring prefix is named the same whatever order the input SMILES lists
its atoms in. The record that a prefix is substituted (so it takes 'bis', (c)) is
written from the atoms of the prefix (``prefix_derivation``), not keyed by the name text (a side
record keyed by the name text made di/bis depend on input order and was taken out of lane L2);
these rows pin the single, order-independent spelling. Each molecule: 8 fixed re-spellings, one
name and one tier (best-effort flags, the tier the 5-methyl row is only reachable at)."""
import pytest

from orthonym import Orthonym

ROWS = [
    # pin_verified: "General methodology" (c) "any component which is substituted
    # automatically requires use of the multiplicative forms 'bis', 'tris', etc."
    # (the Blue Book), so the substituted prefix '5-methyl-1-benzofuran-2-yl' takes 'bis'
    # (the book's own PIN carries it: '(2,2-dimethylpropyl)bis(2-methylprop-2-en-1-yl)azane',
    # the Blue Book). The name used to read 'di(...)' at systematic_verified because the
    # record that a prefix is substituted was lost at the ring-yl writer; it is now written
    # there, from the atoms of the prefix, so the spelling does not depend on the input order
    # (OPSIN 2.9.0 reads the 'bis' name to the input InChIKey).
    ("bis(5-methyl-1-benzofuran-2-yl)methanol", "pin_verified", [
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
