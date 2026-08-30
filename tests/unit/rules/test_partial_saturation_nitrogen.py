"""Task C increment 5: a SATURATED ring nitrogen is a HYDRO position.

`rules.fused_rings._try_partial_saturation_name` used to DEFER (return None) on
any saturated sp3 ring nitrogen, so a partially-saturated mancude azine
(4,5,6,7-tetrahydro-imidazo[4,5-c]pyridine and the pyrrolo/pyrazolo/pyrido
analogues) fell to the legacy indicated-H path and emitted the WRONG
`1H,2H,3H,4H-imidazo[5,4-c]pyridine` (OPSIN-parseable but a DIFFERENT
constitution — the saturated ring cited as four indicated-H instead of
tetrahydro). Since the pyridine-type =N- of the mancude parent becomes -NH-
under the ring saturation, a non-aromatic sp3 N belongs in the max-double-bond
partition exactly like a saturated carbon; the pyrrole-type NH is aromatic and
excluded. A fail-closed post-check defers when any N would be cited as
INDICATED hydrogen (the genuinely ambiguous case).

Governing rules: IUPAC 2013 P-31.1.4 / P-25.7.1.1 (hydro prefixes), P-58.2.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.fused_rings import name_fused_heterocycle


def _const(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


# (smiles, expected tetrahydro prefix substring)
SATURATED_N = [
    ("c1nc2c([nH]1)CCNC2", "tetrahydroimidazo"),          # imidazo[4,5-c]pyridine
    ("c1cc2c([nH]1)CCNC2", "tetrahydropyrrolo"),          # pyrrolo[3,2-c]pyridine
    ("c1n[nH]c2c1CCNC2", "tetrahydropyrazolo"),           # pyrazolo
    ("C1Cc2cncnc2CN1", "tetrahydropyrido"),               # pyrido[3,4-d]pyrimidine
]

# carbon-only (and O) partial saturation must be UNCHANGED (regression guard)
UNCHANGED = [
    ("c1ccc2c(c1)CCNC2", "1,2,3,4-tetrahydroisoquinoline"),
    ("c1ccc2c(c1)OCC2", "2,3-dihydro-1-benzofuran"),
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,frag", SATURATED_N)
def test_saturated_ring_nitrogen_is_tetrahydro_not_indicated_h(smi, frag):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(smi)
    res = name_fused_heterocycle(m)
    assert res is not None
    name = res[0]
    assert "hydro" in name and frag in name, name
    assert "H," not in name, f"still emits indicated-H run: {name}"
    parsed = opsin_parse(name)
    assert parsed is not None, f"OPSIN rejected: {name}"
    assert _const(parsed) == _const(smi), f"wrong constitution: {name} -> {parsed}"


@pytest.mark.parametrize("smi,expected", UNCHANGED)
def test_carbon_only_partial_saturation_unchanged(smi, expected):
    m = Chem.MolFromSmiles(smi)
    res = name_fused_heterocycle(m)
    assert res is not None
    assert res[0] == expected, res[0]
