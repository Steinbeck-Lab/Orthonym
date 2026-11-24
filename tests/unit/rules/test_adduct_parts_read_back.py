"""A mixture name whose parts read back one by one, a charge-separated or tautomeric drawing of one part aside.

 "Organic adducts" (the Blue Book) names each component as a compound of its own; "Mixed
organic – inorganic adducts" (:4666): "preferred IUPAC names cannot be assigned to mixed adducts". A part's name is
verified alone to its full standard InChIKey, so the same test applies to that part inside the adduct; a part with a
metal, a metalloid or one heavy atom must still be read back exactly as drawn, because OPSIN balances the charges of
such parts itself (``rules.adducts._parts_reproduce_depiction``).
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)O[P+](=O)[O-].[NH4+]", "4,4-dimethyl-2-oxo-1,3-dioxa-2-phosphapent-1-ene—azanium (1/1)"),
    ("C=O.C(#N)N=C(N)N.N.Cl", "N-cyanoguanidine—formaldehyde—hydrogen chloride—ammonia (1/1/1/1)"),
])
def test_a_part_drawn_charge_separated_or_as_another_tautomer(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("drawn,read,ok", [
    ("CC(C)(C)O[P+](=O)[O-].[NH4+]", "CC(C)(C)OP(=O)=O.[NH4+]", True),
    ("N.Cl", "[NH4+].[Cl-]", False),                                  # one-atom parts: exact
    ("CCC[P+](CCC)(CCC)CCC.[Cu]I", "CCC[P+](CCC)(CCC)CCC.[Cu+].[I-]", False),   # a metal part: exact
    ("C[Si](C)(C)[N-][Si](C)(C)C.[Ge]", "C[Si](C)(C)[N-][Si](C)(C)C.[Ge+]", False),
    ("CC(=O)O.CN", "CC(=O)[O-].C[NH3+]", False),                      # a proton moved between parts
])
def test_the_part_by_part_check(monkeypatch, drawn, read, ok):
    from rdkit import Chem
    import orthonym.rules.adducts as ad
    monkeypatch.setattr(ad, "_adduct_opsin_read", lambda name: read)
    assert ad._parts_reproduce_depiction("any name", Chem.MolFromSmiles(drawn)) is ok
