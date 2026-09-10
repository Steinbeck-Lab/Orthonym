"""Wave-2 completion Tier 2: benzene ring-FG walker root /.

Explicit recognizers for azido / isocyano / iodosyl / iodyl / oxophosphanyl ring
substituents that the plain-symbol walker branches mis-named or dropped. All
OPSIN-RT probed at build time.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound

pytestmark = pytest.mark.unit


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


@pytest.mark.parametrize("smiles,expected", [
    ("[N-]=[N+]=Nc1ccccc1", "azidobenzene"),               #
    ("O=Ic1ccccc1", "iodosylbenzene"),                     #
    ("O=I(=O)c1ccccc1", "iodylbenzene"),
    # Wave-2 C: BB PIN via the heterone namer (was the prefix form)
    ("O=Pc1ccccc1", "phenylphosphanone"),
    ("[C-]#[N+]c1ccccc1", "isocyanobenzene"),              #
    ("[N-]=[N+]=Nc1ccc(F)cc1", "1-azido-4-fluorobenzene"),
    ("[N-]=[N+]=Nc1ccc(Cl)cc1Cl", "1-azido-2,4-dichlorobenzene"),
])
def test_ring_fg_recognizers(smiles, expected):
    assert _name(smiles) == expected


def test_halogen_and_nitroso_unchanged():
    # Protect: plain halogens + nitroso keep their names (iodosyl must not
    # steal plain iodobenzene).
    assert _name("Ic1ccccc1") == "iodobenzene"
    assert _name("Clc1ccccc1") == "chlorobenzene"
    assert _name("O=Nc1ccccc1") == "nitrosobenzene"
    assert _name("Brc1ccccc1") == "bromobenzene"


def test_azido_nitro_coexistence_heals():
    # Wave-2 completion B4 BUILT the former charged-FG gap: the zwitterion
    # detector now masks internal-charge FG atoms, so the
    # pre-dispatch neutralisation no longer corrupts azide+nitro molecules.
    # OPSIN-RT verified.
    assert _name("[N-]=[N+]=Nc1ccccc1[N+](=O)[O-]") == "1-azido-2-nitrobenzene"
