"""v35 Track A #2: verify-before-commit in the T4 best-effort cascade.

`_best_effort_candidate` rung 0 (`_run_general_e1`) is E1-complete but
STEREO-BLIND -- it proves atom coverage, not stereo. For a molecule with
defined stereo it can emit a stereo-OMITTED name that E1-passes and so
short-circuits the cascade at `return candidate`, never reaching the final
`name_universal_substitutive` rung whose v35 branch-stereo produces a
FULL-InChIKey-round-tripping name. The stereo-blind name then wins the whole
pipeline (it full-RT-FAILS, but so does every other offer, so the pool falls
back to it) -- shadowing the correct floor name.

These assert the whole-namer outcome: the best-effort name for a
branch-stereo molecule FULL-InChIKey round-trips (constitution AND stereo).
Before the fix the pipeline ships the stereo-blind rung-0 name (full-RT FAIL).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _full_ik(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return inchi.MolToInchiKey(m)


def _best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _assert_full_rt(smiles: str):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    nm = _best_effort()
    name = nm.name(smiles)
    assert name and not str(name).startswith("unknown"), f"no name: {name!r}"
    opsin_smi = opsin_parse(str(name))
    if opsin_smi is None:
        pytest.skip("OPSIN jar unavailable")
    assert _full_ik(smiles) == _full_ik(opsin_smi), (
        f"emitted name does not full-RT-match (stereo shadowed): {name!r}")


def test_branch_stereo_amide_not_shadowed_by_stereoblind_primary():
    # rung-0 general_engine emits a (2,5-dihydro-1H-pyrrole) parent that omits
    # the chain (10E) stereo; the universal floor names it stereo-complete.
    _assert_full_rt(r"C#CCCCC(=C)CCC(C)/C=C/CCC(=O)NCC/C(=C\C(=O)N1C(=O)C=C[C@@H]1C)OC")


def test_hexapeptide_full_stereo_not_shadowed():
    # rung-0 drops most of the 6 backbone centres; the floor carries all of them.
    _assert_full_rt("C#CCCCC[C@@H](C)C(=O)N(C)[C@H](C(=O)N(C)[C@H](C(=O)N[C@H]"
                    "(C(=O)N(C)[C@H](C(=O)N[C@@H](C)C(N)=O)[C@H](C)CC)[C@H](C)CC)"
                    "C(C)C)C(C)C")
