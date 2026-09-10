""" Phase-A: determinism gate — same molecule => same PIN regardless of SMILES atom order.

P-14.3.5 / P-44.3.3 / P-63.1.1.2 require a UNIQUE PIN. If the name changes when the same molecule
is re-spelled, the selection cascade fell through to input order (a Blue-Book violation RT cannot see).

Each curated probe (benchmarks/the gold set/determinism_probes.json) is named from its input SMILES, its
canonical SMILES, and several SEEDED atom-permutation re-spellings (stereo-preserving). All names must be
byte-identical after normalization. Probes listed in known_nondeterministic.json are xfail(strict=True),
pinned to their DEF id — when the owning fix lands they flip to a hard pass and must STAY fixed.

See internal notes §B.
"""
import json
import random
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger

from orthonym import name_compound

RDLogger.DisableLog("rdApp.*")

_PIN_ORACLE = Path(__file__).resolve().parents[3] / "benchmarks" / "pin_oracle"
_PROBES = _PIN_ORACLE / "determinism_probes.json"
_QUARANTINE = _PIN_ORACLE / "known_nondeterministic.json"

_N_RAND = 6
_SEED = 0


def _norm(s):
    return " ".join(str(s).lower().strip().split())


def _rows(path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("rows", data) if isinstance(data, dict) else data


def _quarantined_smiles():
    return {(r.get("smiles") if isinstance(r, dict) else r) for r in _rows(_QUARANTINE)}


def _respell(mol, seed):
    rng = random.Random(seed)
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


def _distinct_names(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable probe SMILES: {smiles}"
    spellings = [smiles, Chem.CanonSmiles(smiles)] + [_respell(mol, _SEED + i) for i in range(_N_RAND)]
    return sorted({_norm(name_compound(s)) for s in spellings})


_PROBE_ROWS = _rows(_PROBES)
_QUAR = _quarantined_smiles()


def _param(r):
    smi = r["smiles"] if isinstance(r, dict) else r
    did = (r.get("def_id", "") if isinstance(r, dict) else "")
    marks = ()
    if smi in _QUAR:
        marks = (pytest.mark.xfail(strict=True,
                                   reason=f"known non-determinism [{did}] — quarantined until the owning fix lands"),)
    return pytest.param(smi, marks=marks, id=smi)


class TestDeterminismProbesExist:
    def test_probe_file_exists_and_nonempty(self):
        assert _PROBES.exists(), "v22 Phase-A must create benchmarks/pin_oracle/determinism_probes.json"
        assert _PROBE_ROWS, "determinism_probes.json has no rows"


@pytest.mark.skipif(not _PROBE_ROWS, reason="no determinism probes yet")
@pytest.mark.parametrize("smiles", [_param(r) for r in _PROBE_ROWS])
def test_probe_is_deterministic(smiles):
    """Quarantined probes are xfail(strict): they MUST still be non-deterministic until fixed,
    and a non-quarantined probe MUST be deterministic."""
    names = _distinct_names(smiles)
    assert len(names) == 1, f"{smiles} is order-dependent: produced {names}"
