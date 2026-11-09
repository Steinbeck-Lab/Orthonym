"""A ring ketone or ester beside an anilinium is a prefix (program 17 lead, 2026-11-09).

 'SENIORITY ORDER FOR CLASSES' (the Blue Book, Table 4.1) ranks '6 Cations' (:18169) before '7 Acids' and
every class below, so the protonated amine is the principal characteristic group and the anilinium the parent
 :41415 'aniline -> anilinium'); a ketone or an ester of the ring is then cited as 'acetyl' or
'methoxycarbonyl', as beside a phenol ('4-acetylphenol'). The engine declined 'methyl 4-azaniumylbenzoate' at the PIN
tier: the benzene writer had no prefix form for the ring ester when the amine was the forced principal group.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("COC(=O)c1ccc([NH3+])cc1", "4-(methoxycarbonyl)anilinium"),
    ("COC(=O)c1ccc([NH3+])cc1.[Cl-]", "4-(methoxycarbonyl)anilinium chloride"),
    ("CC(=O)c1ccc([NH3+])cc1", "4-acetylanilinium"),
    ("OC(=O)c1ccc([NH3+])cc1", "4-carboxyanilinium"),
    # neutral: the ketone or ester is senior to the amine, unchanged
    ("CC(=O)c1ccc(N)cc1", "1-(4-aminophenyl)ethan-1-one"),
    ("COC(=O)c1ccc(N)cc1", "methyl 4-aminobenzoate"),
])
def test_the_cation_takes_the_ring_carbonyl_as_a_prefix(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
