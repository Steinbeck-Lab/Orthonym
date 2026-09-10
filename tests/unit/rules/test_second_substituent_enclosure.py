"""/.3.2 — first cited substituent bare, each subsequent one enclosed.

On a single-substitutable retained parent (acetic acid) or a mononuclear parent
hydride (silane), 2+ substituents are cited first-bare / second-and-further-enclosed;
a multiplicative prefix stays OUTSIDE the marks:

    anilino(oxo)acetic acid, the Blue Book)
    hydroxydi(phenyl)acetic acid (the Blue Book -- 'di' outside, not '(diphenyl)')
    (R)-methyl(propyl)silanol, cf. ethyl(methyl)(propyl)phosphane:7282)

Every emitted name OPSIN-round-trips.
"""
import pytest
from rdkit import Chem

_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(O)C(=O)Nc1ccccc1", "anilino(oxo)acetic acid"),
    ("OC(=O)C(O)(c1ccccc1)c1ccccc1", "hydroxydi(phenyl)acetic acid"),
    ("CCC[Si@@H](C)O", "(R)-methyl(propyl)silanol"),
])
def test_second_substituent_enclosed(smiles, expected):
    from orthonym import Orthonym
    got = Orthonym(**_FLAGS).name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(O)C(=O)Nc1ccccc1", "anilino(oxo)acetic acid"),
    ("OC(=O)C(O)(c1ccccc1)c1ccccc1", "hydroxydi(phenyl)acetic acid"),
    ("CCC[Si@@H](C)O", "(R)-methyl(propyl)silanol"),
])
def test_expected_pins_round_trip(smiles, expected):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    rt = opsin_parse(expected)
    got = Chem.MolFromSmiles(rt) if rt else None
    assert got is not None and Chem.MolToInchiKey(got) == _inchikey(smiles), \
        f"RT failed for {expected!r}"


@pytest.mark.parametrize("smiles,expected", [
    # single substituent -> unaffected (needs >= 2 to trigger the enclose pass);
    # the acetic parent omits the alpha locant
    ("OC(=O)CO", "hydroxyacetic acid"),
    # multiplied single ligand on silane stays as the plain multiplier form
    ("C[Si](C)(C)C", "tetramethylsilane"),
    ("C[SiH](C)O", "dimethylsilanol"),
])
def test_single_or_multiplied_unchanged(smiles, expected):
    from orthonym import Orthonym
    got = Orthonym(**_FLAGS).name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"
