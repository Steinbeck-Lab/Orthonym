"""Shared helpers for the large-polycycle tests. Engines are built once per process.

Every data file lives under ``tests/data/large_polycycles/``, including the large target molecule.
A reader whose file is absent -- a tree that carries the tests but not their data, such
as the published one -- skips the calling test instead of failing its collection.
"""
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/large_polycycles"
_ENGINES = {}

#: Default seed of:func:`spellings`. 2 is the smallest seed whose first 14 spellings of
#: the ladder's dimethylcubane include BOTH PIN names that the spelling-dependent cage
#: numbering produced before it was fixed: index 2 gave 1,6-dimethyl..., the other 13
#: gave 1,2-dimethyl... Seeds 1, 4, 7 and 10 give 1,2- only, so a test on them could not
#: fail before the fix. Measured in a fresh process per spelling, at both tiers.
SEED = 2


def _engine(tier):
    if tier not in _ENGINES:
        _ENGINES[tier] = Orthonym(style="pin", **_emit_tier_flags(tier))
    return _ENGINES[tier]


def pin(smiles: str) -> dict:
    return _engine("pin").name_tiered(smiles)


def be(smiles: str) -> dict:
    return _engine("best-effort").name_tiered(smiles)


def _data(name: str) -> str:
    f = DATA / name
    if not f.is_file():
        # allow_module_level: some test modules read the data while pytest collects them
        pytest.skip(f"large-polycycle test data {f} is not in this tree", allow_module_level=True)
    return f.read_text()


def ladder() -> dict:
    out = {}
    for line in _data("class_ladder.tsv").splitlines():
        if line.strip() and not line.startswith("#"):
            smi, label = line.split("\t")[:2]
            out[label] = smi
    return out


def vb_test_smiles() -> list:
    return [s for s in _data("vb_test_smiles.txt").split() if s]


def full_molecule_smiles() -> str:
    """The large target molecule, as reported (explicit [H] atoms)."""
    return _data("mol.smi").split()[0]


def core_smiles() -> str:
    return ladder()["R05_core_21rings"]


def canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def spellings(smiles: str, n: int, seed: int = SEED):
    """n distinct random SMILES spellings of one molecule, identical in every process.

    Seeded through RDKit's own generator (``MolToRandomSmilesVect(randomSeed=)``):
    ``MolToSmiles(doRandom=True)`` ignores Python's ``random`` state and continues a
    process-global RDKit stream, so the same call returned different spellings
    depending on what ran before it in the process (measured).
    """
    mol = Chem.MolFromSmiles(smiles)
    out = []
    for s in Chem.MolToRandomSmilesVect(mol, 50 * n, randomSeed=seed):
        if s not in out:
            out.append(s)
            if len(out) == n:
                break
    return out
