""" a phase — added-indicated-H ring THIONE / SELONE suffix chalcogen
replacement). Extends the suffix-agnostic cyclic-oxo added-IH engine from =O/=NH
to =S/=Se, so ring thiones emit the PIN `-thione` suffix instead of the non-PIN
`sulfanylidene` substitutive prefix.

Also LOCKS the added-indicated-H coverage the default path already provides
(reproduce-first a phase finding: the added-IH monocycle class is fully covered;
these are regression guards) and the =O/=NH byte-identity.
"""
import pytest
from rdkit import Chem

from orthonym.rules.partial_saturation import name_cyclic_oxo_compound


def _canon(smi):
    return Chem.CanonSmiles(smi)


# ---- NEW: thione / selone (was `sulfanylidene`/`selanylidene` prefix) ----

@pytest.mark.unit
@pytest.mark.parametrize("smi,expected", [
    ("S=c1cccc[nH]1", "pyridine-2(1H)-thione"),
    ("S=c1cc[nH]cc1", "pyridine-4(1H)-thione"),
    ("S=c1cc[nH]c(=S)[nH]1", "pyrimidine-2,4(1H,3H)-dithione"),
    ("[Se]=c1cccc[nH]1", "pyridine-2(1H)-selone"),
    ("S=c1ccc2ccccc2[nH]1", "quinoline-2(1H)-thione"),
    ("CC1=CC(=S)NC=C1", "4-methylpyridine-2(1H)-thione"),
])
def test_ring_thione_selone_suffix(smi, expected):
    assert name_cyclic_oxo_compound(Chem.MolFromSmiles(smi)) == expected


# ---- seniority: O > S > Se > imine, fail-closed on senior substituents ----

@pytest.mark.unit
def test_oxo_senior_to_thione():
    # thiouracil-shaped: =O wins the -one suffix; =S becomes the sulfanylidene
    # prefix (name_cyclic_oxo_compound's thione branch is only reached when there
    # is NO ring =O).
    out = name_cyclic_oxo_compound(Chem.MolFromSmiles("S=c1[nH]c(=O)cc[nH]1"))
    assert out == "2-sulfanylidene-2,3-dihydropyrimidin-4(1H)-one"


@pytest.mark.unit
def test_acid_substituent_declines_thione():
    # a carboxylic-acid substituent is senior -> the thione engine fails closed
    # (returns None) so the acid takes the suffix downstream.
    assert name_cyclic_oxo_compound(
        Chem.MolFromSmiles("OC(=O)c1cc[nH]c(=S)c1")) is None


@pytest.mark.unit
def test_thiophene_not_a_thione():
    # a RING sulfur (thiophene) is not an exocyclic thione -> declines.
    assert name_cyclic_oxo_compound(Chem.MolFromSmiles("Cc1ccsc1")) is None


# ---- PARITY: the =O / =NH paths are byte-identical (regression guard) ----

@pytest.mark.unit
@pytest.mark.parametrize("smi,expected", [
    ("O=c1cccc[nH]1", "pyridin-2(1H)-one"),
    ("O=c1cc[nH]cc1", "pyridin-4(1H)-one"),
    ("O=C1C=COC=C1", "4H-pyran-4-one"),
    ("O=c1cc[nH]c(=O)[nH]1", "pyrimidine-2,4(1H,3H)-dione"),
    ("O=c1ccc2ccccc2[nH]1", "quinolin-2(1H)-one"),
])
def test_oxo_path_byte_identical(smi, expected):
    assert name_cyclic_oxo_compound(Chem.MolFromSmiles(smi)) == expected


@pytest.mark.unit
def test_saturated_ketone_declines():
    # a fully-saturated ring ketone is NOT an added-IH mancude parent -> declines
    assert name_cyclic_oxo_compound(Chem.MolFromSmiles("O=C1CCCCC1")) is None
