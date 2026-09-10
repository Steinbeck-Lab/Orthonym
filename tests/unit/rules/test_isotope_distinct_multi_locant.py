"""P-45.4.1 / P-82.2.1: ONE nuclide labelled at MULTIPLE DISTINCT positions on
one parent -> a single ascending multi-locant descriptor (1,4-2H2 / 1,3,5-2H3 /
1,4-13C2). Before ``_decorate_distinct_multi_locant`` the placement search only
enumerated a SINGLE shared locant (2,2,2-2H3 -- all at one position) and the
attachment-group split handled only distinct PARTS, so a multiply-labelled ring
or cage fell through to abstention. Every candidate stays OPSIN-RT gated, so a
wrong locant set is never shipped.

Cite: the Blue Book P-82.2.1 (descriptor + locants), P-45.4.1 (lowest locants).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.rules.isotopes import decorate_isotopic_name
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def eng():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _ik(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return inchi.MolToInchiKey(m) if m is not None else None


@pytest.mark.unit
@pytest.mark.parametrize(
    "smi,expected",
    [
        ("[2H]C1CCC([2H])CC1", "(1,4-2H2)cyclohexane"),      # symmetric ring, distinct locants
        ("[2H]C1C([2H])CCCC1", "(1,2-2H2)cyclohexane"),
        ("[2H]C1CC([2H])CC([2H])C1", "(1,3,5-2H3)cyclohexane"),
        ("[13CH3]CC[13CH3]", "(1,4-13C2)butane"),            # heavy nuclide, distinct positions
        # PRIMED locants: a ring assembly (biphenyl) / multiparent parent numbers
        # across sub-units with primed locants, which a plain integer sweep cannot
        # reach -- the D on the second ring had no placeable locant.
        ("[2H]c1ccc([2H])cc1-c1ccc([2H])cc1", "(2,4',5-2H3)-1,1'-biphenyl"),
    ],
)
def test_distinct_multi_locant_name(eng, smi, expected):
    assert decorate_isotopic_name(smi, eng.style, eng) == expected


@pytest.mark.unit
def test_no_prime_parent_sweep_unchanged(eng):
    """A skeleton with NO prime must sweep the identical integer set as before
    (primed tokens are only added when the skeleton itself carries a prime), so
    a plain ring's descriptor is byte-identical -- guards against a regression
    from the primed-locant extension."""
    assert decorate_isotopic_name("[2H]C1CCC([2H])CC1", eng.style, eng) == "(1,4-2H2)cyclohexane"


@pytest.mark.unit
@pytest.mark.parametrize(
    "smi",
    ["[2H]C1CCC([2H])CC1", "[2H]C1CC([2H])CC([2H])C1", "[13CH3]CC[13CH3]",
     "[2H]c1ccc([2H])cc1-c1ccc([2H])cc1"],
)
def test_distinct_multi_locant_round_trips(eng, smi):
    """Every emitted multi-locant name must OPSIN-parse back to the input (0-wrong)."""
    name = decorate_isotopic_name(smi, eng.style, eng)
    assert name is not None
    parsed = opsin_parse(name)
    assert parsed is not None
    assert _ik(parsed) == _ik(smi)
