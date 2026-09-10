"""P-66.6.1.2.1 — acetaldehyde as a SUBSTITUTABLE retained functional parent.

BB P-66.6.1.2.1: "The following names are preferred IUPAC names, with substitution
allowed for acetaldehyde and benzaldehyde"; P-66.6.1.2 "Substitution of aldehydes
parallels that of... carboxylic acids". Like acetic acid, acetaldehyde has a single
substitutable position (the alpha carbon), so substituent locants are omitted:

    O=CCOc1ccccc1 -> phenoxyacetaldehyde (BB 35076)
    O=C[C@@H](O)C1CC1 -> (S)-cyclopropyl(hydroxy)acetaldehyde (BB 45259)

The alpha carbon is the sole stereogenic-capable position on the 2-carbon parent, so
its descriptor is cited BARE ('(S)-', not '(2S)-'). Anything outside the narrow
byte-verified allowance degrades to the systematic '...ethanal' (never a wrong name).

Every emitted name here OPSIN-round-trips to the input (0-wrong).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

# The measure's flags — bare Orthonym() under-emits and would mislead.
_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)


@pytest.fixture(scope="module")
def engine():
    return Orthonym(**_FLAGS)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", [
    # bb_conformance targets (were RIGHT_MOL_NONPIN "…ethanal", now the retained PIN)
    ("O=CCOc1ccccc1", "phenoxyacetaldehyde"),                        # def_id 66.6.4, BB 35076
    ("O=C[C@@H](O)C1CC1", "(S)-cyclopropyl(hydroxy)acetaldehyde"),   # def_id 92.2.1.3, BB 45259
])
def test_substituted_acetaldehyde_retained_pin(engine, smiles, expected):
    got = engine.name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"
    # 0-wrong: the emitted PIN must OPSIN-round-trip to the input structure.
    rt = opsin_parse(got)
    assert rt and _inchikey(rt) == _inchikey(smiles), f"RT failed for {got!r}"


@pytest.mark.parametrize("smiles,expected", [
    ("O=CC", "acetaldehyde"),          # unsubstituted parent (no prefix -> arm never fires)
    ("O=CCC", "propanal"),             # 3-carbon chain -> systematic, not acetaldehyde
    ("O=CCCO", "3-hydroxypropanal"),   # substituent not on the 2-carbon alpha -> systematic
])
def test_acetaldehyde_arm_degrades_outside_allowance(engine, smiles, expected):
    """The retained arm must NOT over-fire: only a substituted 2-carbon aldehyde."""
    assert engine.name_tiered(smiles)["name"] == expected
