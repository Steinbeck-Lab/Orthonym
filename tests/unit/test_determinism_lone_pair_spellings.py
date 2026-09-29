"""The determinism probe's spellings of a molecule with a lone-pair stereocentre
(scripts/determinism_eval.py; TRIAGE.md 'Lone-pair centre written first -- naming and the
gate probe').

The probe re-spells each input with RDKit's writer. RDKit writes a lone-pair stereocentre
that a string starts with (and some ring-closure positions at P(III)) unlike an implicit
hydrogen there, so such a spelling denotes the other stereoisomer for CDK, OpenSMILES and
OPSIN. At adee5416e the probe reported 'NEW NONDET: CC[S@](=O)C -> [(R)-(methanesulfinyl)
ethane,...]' for exactly those spellings. Every spelling of a lone-pair probe is now checked
with CDK against the input as written, re-spelt from the same atom order when CDK reads
another configuration, and dropped when that cannot be done; any other probe is untouched.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import determinism_eval as det  # noqa: E402

from orthonym import namer  # noqa: E402

pytestmark = pytest.mark.opsin_gate


def _spellings(smiles, n_rand=6, seed=0):
    mol = Chem.MolFromSmiles(smiles)
    out = {"input": smiles, "canonical": Chem.CanonSmiles(smiles)}
    for i in range(n_rand):
        out[f"rand{i}"] = det.seeded_respelling(mol, seed + i)
    return mol, out


def _key(smiles):
    from orthonym.perception.centres_bridge import centres_label_batch
    labels = centres_label_batch([smiles])[smiles]
    return namer._lp_annotated_smiles(namer._lp_heavy_mol(smiles), labels)


def test_sulfur_first_spellings_are_respelt_to_the_input_configuration():
    smiles = "CC[S@](=O)C"
    mol, spellings = _spellings(smiles)
    first = {k for k, v in spellings.items() if v.startswith("[S")}
    assert first, spellings  # seed 0 does write the sulfur first
    kept, respelled, dropped = det._lone_pair_standard_spellings(smiles, spellings, mol, 0)
    assert set(kept) == set(spellings) and not dropped
    assert set(respelled) == first
    ref = _key(smiles)
    for label, spelled in kept.items():
        assert _key(spelled) == ref, (label, spelled)
        # Same molecule constitution; the respelling keeps RDKit's atom order.
        assert Chem.MolToSmiles(Chem.MolFromSmiles(spelled), isomericSmiles=False) == "CCS(C)=O"
    for label in respelled:
        assert kept[label] != spellings[label] and kept[label].startswith("[S")
        assert _key(spellings[label]) != ref


def test_a_sulfur_first_input_drops_rdkits_canonical_spelling():
    smiles = "[S@](=O)(C)CC"
    mol, spellings = _spellings(smiles)
    kept, respelled, dropped = det._lone_pair_standard_spellings(smiles, spellings, mol, 0)
    assert dropped == {"canonical": spellings["canonical"]}
    ref = _key(smiles)
    assert all(_key(s) == ref for s in kept.values())


def test_p_ring_closure_spellings_are_checked_too():
    smiles = "CO[P@@]1OC[C@H](C)O1"
    mol, spellings = _spellings(smiles)
    kept, respelled, dropped = det._lone_pair_standard_spellings(smiles, spellings, mol, 0)
    assert respelled and not dropped
    ref = _key(smiles)
    assert all(_key(s) == ref for s in kept.values())


@pytest.mark.parametrize("smiles", ["C[C@H](O)CC", "CCO", "C/C=C/C", "N[C@@H](C)C(=O)O"])
def test_other_probes_are_untouched(smiles):
    mol, spellings = _spellings(smiles)
    kept, respelled, dropped = det._lone_pair_standard_spellings(smiles, spellings, mol, 0)
    assert kept is spellings and respelled == {} and dropped == {}


def test_the_gate_probe_is_deterministic_for_the_sulfoxide():
    from orthonym import Orthonym
    res = det.probe_smiles("CC[S@](=O)C", Orthonym().name, 6, 0, 120.0)
    assert res["deterministic"], res["names"]
    assert res["distinct_names"] == ["(r)-(methanesulfinyl)ethane"]
    assert res["lone_pair_respelled"] and not res["lone_pair_dropped"]


def test_without_centres_the_spellings_are_kept(monkeypatch):
    import orthonym.perception.centres_bridge as cb
    monkeypatch.setattr(cb, "centres_label_batch", lambda smiles, timeout=120.0: None)
    smiles = "CC[S@](=O)C"
    mol, spellings = _spellings(smiles)
    kept, respelled, dropped = det._lone_pair_standard_spellings(smiles, spellings, mol, 0)
    assert kept is spellings and respelled == {} and dropped == {}
