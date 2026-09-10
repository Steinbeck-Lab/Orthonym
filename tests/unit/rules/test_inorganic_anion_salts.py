""" abstain-recovery — common inorganic oxo/complex anions in onium salts.

A salt abstains iff one component is unnameable; perchlorate/BF4/PF6 were the biggest
residual salt-abstain cause. Each anion word OPSIN-round-trips to the anion (0-wrong).
"""
import pytest
from rdkit import Chem

_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)


def _ik(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m) if m else None


@pytest.mark.parametrize("smi,word", [
    ("[O-]Cl(=O)(=O)=O", "perchlorate"),
    ("[O-]Cl(=O)=O", "chlorate"),
    ("[O-]Br(=O)(=O)=O", "perbromate"),
    ("[O-]Br(=O)=O", "bromate"),
    ("[O-]I(=O)(=O)=O", "periodate"),
    ("[O-]I(=O)=O", "iodate"),
    ("[B-](F)(F)(F)F", "tetrafluoroborate"),
    ("F[P-](F)(F)(F)(F)F", "hexafluorophosphate"),
])
def test_anion_word_round_trips(smi, word):
    """Each added anion word parses back to the anion (0-wrong contract)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    rt = opsin_parse(word)
    assert rt and _ik(rt) == _ik(smi), f"{word} !RT {smi}"


@pytest.mark.parametrize("smi,expected", [
    ("[Na+].[O-]Cl(=O)(=O)=O", "sodium perchlorate"),
    ("[Na+].[O-]Cl(=O)=O", "sodium chlorate"),
])
def test_simple_perchlorate_salts_name(smi, expected):
    from orthonym import Orthonym
    assert Orthonym(**_FLAGS).name_tiered(smi)["name"] == expected


@pytest.mark.parametrize("smi", [
    "[O-]Cl(=O)(=O)=O.C[N+](C)(C)C",   # tetramethylammonium perchlorate
    "F[P-](F)(F)(F)(F)F.C[N+](C)(C)C",  #...hexafluorophosphate
    "[B-](F)(F)(F)F.C[N+](C)(C)C",      #...tetrafluoroborate
])
def test_onium_inorganic_salts_round_trip(smi):
    """The onium salt now names AND round-trips to the full multi-fragment InChIKey."""
    from orthonym import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse
    nm = Orthonym(**_FLAGS).name_tiered(smi)["name"]
    assert nm, f"abstained on {smi}"
    rt = opsin_parse(nm)
    assert rt and _ik(rt) == _ik(smi), f"{nm} !RT {smi}"
