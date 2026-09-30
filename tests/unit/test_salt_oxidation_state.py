"""a phase BBR-.6-caveats — salt cation Stock oxidation state .

The salt path dropped the Stock oxidation-state numeral from variable-valence metal
salt cations ('gold(I) chloride' -> 'gold chloride', the 169.6 regression). 169.7
appends the Stock numeral (.4.2.2 / for VARIABLE-valence metals
ONLY; FIXED-valence metals (group 1/2, Al, Zn, Ag,...) carry no numeral. For a
monatomic metal cation the oxidation state == the formal charge. All RT-verified.

NOTE (documented residual, deferred): quaternary-ammonium salt cations still drop
their cation WORD (name_cation('') -> only the anion ships, e.g.
CCCCCC[N+](C)(C)C.[Cl-] -> 'chloride'). That is a deeper name_cation gap (the
cation-as-parent '-aminium' naming), tracked as a follow-on — NOT this fix.
"""
import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("[Au+].[Cl-]", "gold(I) chloride"),               # variable-valence: Stock recovered
    # variable-valence; the stated charge fixes the ratio, so no stoichiometric
    # prefix: binary names cite "the name of the cation followed by that of the
    # anion", the Blue Book; a performance pass,)
    ("[Fe+2].[Cl-].[Cl-]", "iron(II) chloride"),
])
def test_variable_valence_metal_carries_stock(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("[Na+].[Cl-]", "sodium chloride"),                # fixed-valence: NO Stock
    ("[Ca+2].[Cl-].[Cl-]", "calcium dichloride"),      # fixed-valence
    ("[K+].CC(=O)[O-]", "potassium acetate"),          # fixed-valence + organic anion
])
def test_fixed_valence_metal_no_stock(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.unit
def test_gold_chloride_round_trips():
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    assert opsin_roundtrip_check("[Au+].[Cl-]", "gold(I) chloride")["passed"]
