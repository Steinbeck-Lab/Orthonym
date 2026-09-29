"""Breadth Job 2 (M17b): Group-12 organometallics with a functionalised ligand, and
the spelling of the producer that names them.

A functionalised organyl ligand on Zn/Cd/Hg ('C1CCC(=O)C(C1)[Hg]Cl', a milestone1500
best-effort gap row) is named at the best-effort tier only:
(the Blue Book) "Coordination nomenclature is the primary nomenclature method
used to name organometallic compounds containing elements of Groups 3 through 12", so
the metal is the central atom whatever its ligand carries, and (:39735) notes
no PIN for these metals. The name takes the producer's compositional form, as in
'methylmagnesium iodide',:40197).

The producer now encloses a compound or locant-bearing ligand prefix --
(the Blue Book) "Parentheses are used around compound... and complex...
prefixes"; (:7255) around simple prefixes with locants, as in
'(propan-2-yl)cyanamide (PIN)' (:33535) -- and multiplies identical ligands once,
as in '1,4-di(propan-2-yl)cyclohexane (PIN)' (:25719). It used to ship
'propan-2-ylmercury chloride' and '(propan-2-yl)(propan-2-yl)zinc'.

Every expected name is read back by a FRESH OPSIN call (tests.support.rt_assert).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import _independent_parse, name_is_rt_exact

pytestmark = [pytest.mark.opsin_gate]


def _best_effort_row(smiles):
    with jvm_slots(1, purpose="breadth-job2-test"):
        return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)


def _pin_row(smiles):
    with jvm_slots(1, purpose="breadth-job2-test"):
        return Orthonym().name_tiered(smiles)


def _canon(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _assert_reads_back_exactly(name, smiles):
    parsed = _independent_parse(name)
    assert parsed and _canon(parsed) == _canon(smiles), (
        f"{name!r} reads back as {parsed!r}, not as the drawn {smiles}")
    assert name_is_rt_exact(name, smiles), f"{name!r}: no full-key round trip"


# Group-12 spellings /, the Blue Book /:7255).
GROUP12 = [
    ("CC(C)[Hg]Cl", "(propan-2-yl)mercury chloride"),
    ("CC(C)[Zn]C(C)C", "di(propan-2-yl)zinc"),
    ("CC(C)(C)[Hg]Cl", "tert-butylmercury chloride"),
    ("CC[Mg]Br", "ethylmagnesium bromide"),
]


@pytest.mark.parametrize("smiles,expected", GROUP12)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_group12_ligand_spelling(smiles, expected, tier):
    row = _pin_row(smiles) if tier == "pin" else _best_effort_row(smiles)
    name = row.get("name")
    assert name == expected, f"{smiles} ({tier}): {name!r}"
    _assert_reads_back_exactly(name, smiles)
    assert row["is_pin"] is False  #, the Blue Book


@pytest.mark.parametrize("smiles,expected", [
    ("C1CCC(=O)C(C1)[Hg]Cl", "(2-oxocyclohexyl)mercury chloride"),
    ("OC(=O)CC[Hg]Cl", "(2-carboxyethyl)mercury chloride"),
    ("NCC[Zn]CCN", "bis(2-aminoethyl)zinc"),
])
def test_group12_functional_ligand_best_effort_only(smiles, expected):
    row = _best_effort_row(smiles)
    assert row.get("name") == expected, f"{smiles}: {row.get('name')!r}"
    _assert_reads_back_exactly(expected, smiles)
    assert _pin_row(smiles)["tier"] == "abstain"
