"""v31 PIN conformance (P-33.3): a suffixable principal characteristic group on a
SPIRO parent must be cited as the principal SUFFIX, not demoted to a prefix.

Root cause (measured 2026-08-10): the `complex_ring` assembly named the acid/ol/
one/amine as a detachable prefix on a spiro parent (`9-carboxyspiro[5.5]undecane`),
where the PIN is `spiro[5.5]undecane-3-carboxylic acid`. von-Baeyer parents already
suffix correctly; only spiro demoted. Fix: the spiro handler prefers the general
engine's suffix-form name when it OPSIN-round-trips to the input (0-wrong).
"""
import pytest
from rdkit import Chem
from orthonym import Orthonym


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(O)C1CCC2(CCCCC2)CC1", "spiro[5.5]undecane-3-carboxylic acid"),
    ("OC1CCC2(CCCCC2)CC1",      "spiro[5.5]undecan-3-ol"),
    ("O=C1CCC2(CCCCC2)CC1",     "spiro[5.5]undecan-3-one"),
    ("NC1CCC2(CCCCC2)CC1",      "spiro[5.5]undecan-3-amine"),
])
def test_spiro_principal_group_is_suffix_not_prefix(opsin_gate, smiles, expected):
    out = Orthonym(style="pin").name(Chem.CanonSmiles(smiles))
    assert out == expected, f"expected suffix PIN {expected!r}, got {out!r}"


@pytest.mark.parametrize("smiles,expected", [
    # PG-free spiro and von-Baeyer parents must be byte-identical (fix is scoped).
    ("C1CCC2(CCCCC2)CC1",              "spiro[5.5]undecane"),
    ("OC(=O)C12CC3CC(CC(C3)C1)C2",     "tricyclo[3.3.1.1^3,7]decane-1-carboxylic acid"),
    ("OC(=O)C1CC2CCC1C2",              "bicyclo[2.2.1]heptane-2-carboxylic acid"),
])
def test_scope_unchanged(opsin_gate, smiles, expected):
    out = Orthonym(style="pin").name(Chem.CanonSmiles(smiles))
    assert out == expected, f"scope regression: expected {expected!r}, got {out!r}"
