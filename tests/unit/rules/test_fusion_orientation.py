"""Unit tests for the deterministic fusion-orientation engine (Stage 0/1).

Covers the coordinate-free hex-lattice embedding, the cata-fused all-6
carbocyclic classification gate, and the P-25.3.2.3.3 orientation scoring.
"""

import random

import pytest
from rdkit import Chem
from rdkit import RDLogger

from orthonym.rules.fusion_orientation import (
    best_orientations,
    classify_ring_system,
    embed_atoms_on_hex_lattice,
    score_orientation,
)

RDLogger.DisableLog("rdApp.*")


def _ring_atoms(mol):
    s = set()
    for r in mol.GetRingInfo().AtomRings():
        s.update(r)
    return s


# Canonical SMILES for the systems under test.
_PAH = {
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
    "tetracene": "c1ccc2cc3cc4ccccc4cc3cc2c1",
    "chrysene": "c1ccc2c(c1)ccc1c3ccccc3ccc21",
}


@pytest.mark.unit
class TestClassification:
    def test_all6_cata_carbocyclic_accepts_pahs(self):
        for name, smi in _PAH.items():
            m = Chem.MolFromSmiles(smi)
            info = classify_ring_system(m, _ring_atoms(m))
            assert info["all_six"], name
            assert info["carbocyclic"], name
            assert info["cata_fused"], name
            assert info["connected"], name

    def test_pyrene_is_peri_fused_rejected(self):
        # Pyrene has interior atoms (peri-fusion) -> cata_fused False.
        m = Chem.MolFromSmiles("c1cc2ccc3cccc4ccc(c1)c2c34")
        info = classify_ring_system(m, _ring_atoms(m))
        assert info["all_six"]
        assert info["carbocyclic"]
        assert not info["cata_fused"]

    def test_azulene_not_all_six_rejected(self):
        m = Chem.MolFromSmiles("c1ccc2cccc-2cc1")
        info = classify_ring_system(m, _ring_atoms(m))
        assert not info["all_six"]

    def test_heterocycle_not_carbocyclic(self):
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")  # quinoline-like
        info = classify_ring_system(m, _ring_atoms(m))
        assert not info["carbocyclic"]


@pytest.mark.unit
class TestEmbedding:
    def test_embedding_places_every_atom_on_integer_lattice(self):
        for name, smi in _PAH.items():
            m = Chem.MolFromSmiles(smi)
            info = classify_ring_system(m, _ring_atoms(m))
            coords = embed_atoms_on_hex_lattice(info["graph"], m)
            assert coords is not None, name
            assert set(coords) == _ring_atoms(m), name
            for (x, y) in coords.values():
                assert isinstance(x, int) and isinstance(y, int), name
                # Every lattice point has x+y even (triangular-lattice parity).
                assert (x + y) % 2 == 0, name

    def test_pyrene_peri_fused_orientation_none(self):
        # Pyrene is peri-fused: the classification (cata-fused) gate rejects it,
        # so the public entry point ``best_orientations`` returns None.
        m = Chem.MolFromSmiles("c1cc2ccc3cccc4ccc(c1)c2c34")
        info = classify_ring_system(m, _ring_atoms(m))
        assert not info["cata_fused"]
        assert best_orientations(m, _ring_atoms(m)) is None

    def test_helicene_embedding_rejected_overcrowded(self):
        # A [5]+ helicene cannot embed planar; the overcrowding/planarity gate in
        # ``embed_atoms_on_hex_lattice`` returns None (non-bonded atoms crammed
        # below 3*d^2 = 12).  [6]+ overlaps atoms outright; [5] crams to bond
        # distance.  Both decline -> best_orientations is None.
        for smi in (
            "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC=CC5=C4C3=C12",  # pentahelicene
            "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=CC=C6C5=C4C3=C12",  # hexahelicene
        ):
            m = Chem.MolFromSmiles(smi)
            info = classify_ring_system(m, _ring_atoms(m))
            assert info["cata_fused"], "helicene is cata-fused"
            assert embed_atoms_on_hex_lattice(info["graph"], m) is None
            assert best_orientations(m, _ring_atoms(m)) is None


@pytest.mark.unit
class TestOrientationScoring:
    def test_anthracene_three_in_a_row(self):
        # Linear acene -> best orientation has 3 rings in the horizontal row.
        m = Chem.MolFromSmiles(_PAH["anthracene"])
        info = classify_ring_system(m, _ring_atoms(m))
        best = best_orientations(m, _ring_atoms(m))
        assert best
        score = score_orientation(info["graph"], best[0])
        assert score[0] == -3  # -(max rings in a row) == -3

    def test_phenanthrene_two_in_a_row_angular(self):
        # Angular tricyclic -> 2 in the row, then 1.5 rings upper-right.
        m = Chem.MolFromSmiles(_PAH["phenanthrene"])
        info = classify_ring_system(m, _ring_atoms(m))
        best = best_orientations(m, _ring_atoms(m))
        assert best
        score = score_orientation(info["graph"], best[0])
        assert score[0] == -2  # 2 rings in the row (anthracene-senior rule)
        # upper-right quarters: phenanthrene = 1.5 rings = 6 quarters.
        assert score[1] == -6

    def test_naphthalene_two_in_a_row(self):
        m = Chem.MolFromSmiles(_PAH["naphthalene"])
        info = classify_ring_system(m, _ring_atoms(m))
        best = best_orientations(m, _ring_atoms(m))
        assert best
        assert score_orientation(info["graph"], best[0])[0] == -2

    def test_triphenylene_row_is_contiguous_not_spurious(self):
        # Branched D3h system: the main row is only 2 ortho-fused rings joined by
        # a vertical bond.  A naive "all rings at one y" count would wrongly find
        # 3 (a same-y ring is disconnected); the contiguous-run fix keeps it 2.
        m = Chem.MolFromSmiles("c1ccc2c(c1)c1ccccc1c1ccccc21")
        info = classify_ring_system(m, _ring_atoms(m))
        best = best_orientations(m, _ring_atoms(m))
        assert best
        assert score_orientation(info["graph"], best[0])[0] == -2

    def test_benzanthracene_three_in_a_row(self):
        # benz[a]anthracene has a genuine 3-ring (anthracene-like) horizontal row.
        m = Chem.MolFromSmiles("c1ccc2cc3c(ccc4ccccc43)cc2c1")
        info = classify_ring_system(m, _ring_atoms(m))
        best = best_orientations(m, _ring_atoms(m))
        assert best
        assert score_orientation(info["graph"], best[0])[0] == -3


@pytest.mark.unit
class TestOrientationDeterminism:
    @pytest.mark.parametrize("name", list(_PAH))
    def test_best_score_stable_across_random_spellings(self, name):
        base = Chem.MolFromSmiles(_PAH[name])
        random.seed(2024)
        scores = set()
        for _ in range(15):
            rsmi = Chem.MolToSmiles(base, doRandom=True, canonical=False)
            m = Chem.MolFromSmiles(rsmi)
            info = classify_ring_system(m, _ring_atoms(m))
            best = best_orientations(m, _ring_atoms(m))
            assert best, f"{name}: best_orientations returned empty"
            scores.add(score_orientation(info["graph"], best[0]))
        # The optimal orientation score is a structural invariant.
        assert len(scores) == 1, f"{name}: scores varied {scores}"
