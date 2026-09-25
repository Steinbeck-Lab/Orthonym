"""Salts of mononuclear noncarbon oxoacids: arsoric, stiboric and silicic acid.

the Blue Book "Salts of mononuclear noncarbon oxoacids" (:35898-35900):
"Names of anions are formed by changing the 'ic acid' ending to 'ate'"; the acids
are preselected names (arsoric acid:35413; stiboric acid and silicic
acid:18423); hydrogen words as in 'methyl dihydrogen phosphate (PIN)' (:35940).

The multi-anion fallback in rules/ions.py used to return the NEUTRAL acid name for
such an anion, so 'O=[As]([O-])([O-])O.[Pb+2]' shipped as 'lead(II) arsoric acid'
(a different molecule) -- exempts a neutral salt input from its net-charge
guard.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from orthonym.rules.ions import name_anion
from tests.support.jars import jar_or_skip


def _strict_rt(name: str, smiles: str) -> bool:
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=False, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


SALTS = [
    ("O=[As]([O-])([O-])O.[Pb+2]", "lead(II) hydrogen arsorate"),
    ("O=[As]([O-])([O-])O.[Sn+2]", "tin(II) hydrogen arsorate"),
    ("O=[As]([O-])([O-])O.[Ca+2]", "calcium hydrogen arsorate"),
    ("O=[As]([O-])([O-])[O-].[Na+].[Na+].[Na+]", "trisodium arsorate"),
    ("O=[As](O)(O)[O-].[Na+]", "sodium dihydrogen arsorate"),
    ("O=[As]([O-])([O-])O.[Na+].[Na+]", "disodium hydrogen arsorate"),
    ("O=[Sb]([O-])([O-])O.[Pb+2]", "lead(II) hydrogen stiborate"),
    ("O=[Sb]([O-])([O-])[O-].[Na+].[Na+].[Na+]", "trisodium stiborate"),
    ("O=[Sb]([O-])(O)O.[K+]", "potassium dihydrogen stiborate"),
    ("[O-][Si]([O-])([O-])[O-].[Na+].[Na+].[Na+].[Na+]", "tetrasodium silicate"),
    ("[O-][Si]([O-])(O)O.[Ca+2]", "calcium dihydrogen silicate"),
    ("[O-][Si](O)(O)O.[Na+]", "sodium trihydrogen silicate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", SALTS)
def test_oxoacid_salt_named(smiles, expected):
    with jvm_slots(1, purpose="test-oxoacid-salt"):
        name = Orthonym(style="pin").name(smiles)
    assert name == expected
    assert _strict_rt(expected, smiles)


# The fallback itself: an anion never gets a neutral-acid name. These four
# returned 'stiboric acid' / 'silicic acid' at the previous commit.
@pytest.mark.parametrize("smiles", [
    "O=[Sb]([O-])([O-])O",
    "[O-][Sb]([O-])([O-])=O",
    "[O-][Si]([O-])(O)O",
    "O=[As]([O-])([O-])O",
])
def test_name_anion_never_returns_a_neutral_acid(smiles):
    assert not name_anion(Chem.MolFromSmiles(smiles)).endswith("acid")
