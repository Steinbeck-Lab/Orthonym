"""A free acid group is senior to the esters, the Blue Book; "Esters cited as prefixes",:31696;
,:31935; 'methyl hydrogen sulfate (PIN)',:35968). An ester name that cites the acid as a prefix is correct
but no PIN (validation/spelling/checks_acid_beside_ester.py).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_acid_beside_ester import acid_beside_ester_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCC",
     "(2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl dihexadecanoate"),
    ("CCCCCCCCC(CCCCCCCCC(=O)OCCCC)OS(=O)(=O)O", "butyl 10-(sulfooxy)octadecanoate"),
    ("CCCCCCCCCCOC(=O)C(C)OP(=O)(O)O", "decyl 2-(phosphonooxy)propanoate"),
    ("COC(=O)c1ccc(OS(=O)(=O)O)cc1", "methyl 4-(sulfooxy)benzoate"),
    ("COC(=O)CCC(=O)O", "methyl 3-carboxypropanoate"),
    ("CC(=O)OC(=O)c1ccc(OS(=O)(=O)O)cc1", "acetic 4-(sulfooxy)benzoic anhydride"),   # an acid outranks an anhydride
])
def test_an_acid_prefix_under_an_ester_name_is_flagged(smiles, name):
    failure = acid_beside_ester_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-65.6.3.2.3", name


@pytest.mark.parametrize("smiles,name", [
    ("CCCCCCCCC(CCCCCCCCC(=O)OCCCC)OS(=O)(=O)O", "1-butoxy-1-oxooctadecan-10-yl hydrogen sulfate"),
    ("COC(=O)c1ccc(OP(=O)(O)O)cc1", "4-(methoxycarbonyl)phenyl dihydrogen phosphate"),
    ("COC(=O)CCC(=O)O", "4-methoxy-4-oxobutanoic acid"),
    ("OC(=O)CCOC(=O)c1ccccc1", "3-(benzoyloxy)propanoic acid"),                     #:31696
    ("CC(=O)OCC[N+](C)(C)C.[O-]S(=O)(=O)O", "2-(acetyloxy)-N,N,N-trimethylethan-1-aminium hydrogen sulfate"),
    ("CC(=O)OC1COP(=O)(O)OC1", "2-hydroxy-2-oxo-1,3,2-dioxaphosphinan-5-yl acetate"),   # a ring P-OH
    ("CC(=O)NCCOC(=O)O", "1-acetamido-2-(carboxyoxy)ethane"),                     # a carbonic half-ester: one group
    ("COC(=O)CCOP(=O)(O)OP(=O)(O)O", "3-methoxy-3-oxopropyl trihydrogen diphosphate"),
])
def test_other_names_pass(smiles, name):
    assert acid_beside_ester_check(Chem.MolFromSmiles(smiles), name) is None, name


@pytest.mark.opsin_gate
def test_a_phospholipid_named_on_its_acyl_esters_loses_the_pin_label():
    """The lipid route reads this check too (validation/spelling/route_scope.py, _ROUTE_EXTRA_RULES); (:50943) gives
    no PIN for the name (gold row -01 now in default_tier_non_pin_rows.json)."""
    from orthonym.cli import _tiered_namer
    row = _tiered_namer("pin", "best-effort").name_tiered("O=C(OCC(OC(=O)CCCCCCCCCCCCC)COP(=O)(O)OCCN)CCCCCCCCCCCCC")
    assert row["name"] == "3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl ditetradecanoate", row   # kept
    assert row["tier"] != "pin_verified" and not row["is_pin"], row

