"""A ring-ester parent is a whole monocyclic ring system (pre-existing-failures
plan, Task 4 continuation, 2026-09-25; the tropisetron chain, TRIAGE row 36 /
canary call 182).

`assembly/composer.py::_assemble_ring_with_ester_prefixes` took the ring that
holds the ester's attachment atom as the parent even when that ring is one ring
of a bicycle, and named the rest as an open chain: '4-(acetyloxy)-2-(ethan-1-yl)-
1-methylpiperidine' for tropan-3-yl acetate, and, with the acid-word defect,
'2-(ethan-1-yl)-1-methyl-4-(nonanoyloxy)piperidine' for tropisetron.
(the Blue Book) names a monocycle only ("The names of saturated monocyclic
hydrocarbons are formed by attaching the nondetachable prefix 'cyclo'..."), so
the parent must be a whole monocyclic ring system, the guard
`_handler_shared._generate_ring_parent` already uses.
"""
from types import SimpleNamespace

import pytest
from rdkit import Chem

from orthonym.assembly.composer import _assemble_ring_with_ester_prefixes
from orthonym.rules.esters import detect_exocyclic_esters

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles", [
    # (tropisetron itself no longer reaches this assembler: its indole acid now
    # has no acid word, so detect_exocyclic_esters finds no nameable ester)
    "CN1C2CCC1CC(OC(C)=O)C2",      # tropan-3-yl acetate (was a piperidine)
    "CC(=O)OC1CC2CCC1C2",          # bicyclo[2.2.1]heptan-2-yl acetate (was a cyclopentane)
])
def test_ring_ester_parent_is_never_one_ring_of_a_bicycle(smiles):
    mol = Chem.MolFromSmiles(smiles)
    exo = detect_exocyclic_esters(mol)
    assert exo
    assert _assemble_ring_with_ester_prefixes(SimpleNamespace(mol=mol), exo) is None


def test_monocyclic_ring_ester_parent_unchanged():
    mol = Chem.MolFromSmiles("CC(=O)OC1CCCCC1")
    exo = detect_exocyclic_esters(mol)
    assert (_assemble_ring_with_ester_prefixes(SimpleNamespace(mol=mol), exo)
            == "(acetyloxy)cyclohexane")
