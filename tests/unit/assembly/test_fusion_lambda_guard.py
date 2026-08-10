"""v31: the P5 fusion-word upgrade in the general engine must NOT preempt the
von-Baeyer polyene when the fused parent carries a λ (non-standard-valence) ring
atom the bare fusion word cannot express. Regression: `c1ccc2c(c1)O[SH2]O2`
(λ4-sulfur) returned the catalog fusion word `[1,3,2]benzodioxathiole` — which
denotes the DIVALENT-S ring, a different molecule (SELF-01-suppressed → abstain),
LOSING the valid von-Baeyer name `7,9-dioxa-8lambda4-thiabicyclo[4.3.0]nona-1,3,5-triene`
that DOES round-trip. Mirrors the pre-existing stereo carve-out on the same
early-return. Normal fused aromatics (no λ) keep the fusion word.
"""
import pytest
from rdkit import Chem
from orthonym.assembly.general_engine import name_general
from orthonym.namer import _validity_gate_name_to_smiles
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

_BE = Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def _canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def _name_general(smi):
    mol = Chem.MolFromSmiles(smi)
    feats = _BE._perceive(mol, smi, Chem.MolToSmiles(mol, canonical=True))
    _BE._classify(feats)
    r = name_general(mol, feats, allow_aromatic_general=True, allow_suffix_free=True)
    return r.name if r else None


@pytest.mark.roundtrip
def test_lambda4_fused_parent_uses_von_baeyer_not_fusion_word():
    smi = "c1ccc2c(c1)O[SH2]O2"
    name = _name_general(smi)
    assert name is not None, "engine should build the von-Baeyer λ4 name"
    # it must NOT be the divalent-S fusion word, and it must round-trip
    assert "benzodioxathiole" not in name, f"still the non-RT fusion word: {name!r}"
    opsin = _validity_gate_name_to_smiles(name)
    assert opsin is not None and _canon(opsin) == _canon(smi), (
        f"NOT rt_exact: {name!r} -> {_canon(opsin)} != {_canon(smi)}")


@pytest.mark.roundtrip
def test_normal_fused_aromatic_keeps_fusion_word():
    # naphthalene (no λ atom) must still take the fusion-word upgrade.
    assert _name_general("c1ccc2ccccc2c1") == "naphthalene"
