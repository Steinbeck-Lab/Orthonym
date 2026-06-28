"""Unit tests for the deterministic fusion-numbering engine (Stage 1).

Covers:
* oracle match — engine numbering equals the stored ``iupac_numbering`` map
  (up to the molecule's own automorphism) for naphthalene / anthracene /
  phenanthrene;
* the systematic peripheral walk reproduces OPSIN numbering for tetracene /
  chrysene (systematically-numbered polyacenes/aphenes);
* fail-closed — peri-fused (pyrene) and non-all-6 (azulene) return ``None``;
* determinism across random SMILES spellings.
"""

import random
import re

import pytest
from rdkit import Chem
from rdkit import RDLogger

from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
from orthonym.rules.fusion_numbering import compute_fused_numbering

RDLogger.DisableLog("rdApp.*")


def _ring_atoms(mol):
    s = set()
    for r in mol.GetRingInfo().AtomRings():
        s.update(r)
    return s


def _norm(v):
    """Normalise a stored ('4a'/4) or engine ((4,'a')/4) locant to (int, str)."""
    if isinstance(v, str):
        m = re.match(r"^(\d+)([a-z]*)$", v)
        return (int(m.group(1)), m.group(2))
    if isinstance(v, int):
        return (v, "")
    return (v[0], v[1])


def _automorph_equal(mol, a_map, b_map):
    """True if a_map equals b_map under some molecular automorphism."""
    if sorted(a_map.values()) != sorted(b_map.values()):
        return False
    n = mol.GetNumAtoms()
    for phi in mol.GetSubstructMatches(mol, uniquify=False):
        if {phi[i]: a_map[i] for i in range(n)} == b_map:
            return True
    return False


@pytest.mark.unit
class TestOracleMatch:
    """compute_fused_numbering == stored iupac_numbering (up to automorphism)."""

    @pytest.mark.parametrize("name", ["naphthalene", "anthracene", "phenanthrene"])
    def test_matches_stored_map(self, name):
        entry = POLYCYCLIC_DATA[name]
        m = Chem.MolFromSmiles(entry["canonical_smiles"])
        stored = {k: _norm(v) for k, v in entry["iupac_numbering"].items()}
        computed = compute_fused_numbering(m, _ring_atoms(m))
        assert computed is not None, f"{name}: engine returned None"
        computed = {k: _norm(v) for k, v in computed.items()}
        # Symmetric systems (naphthalene) may pick a symmetry-equivalent atom
        # labelling — both are valid IUPAC and yield identical names.
        assert computed == stored or _automorph_equal(m, stored, computed), (
            f"{name}: computed {computed} != stored {stored}"
        )

    def test_anthracene_meso_positions_are_9_and_10(self):
        # The defining feature of anthracene's fixed numbering: meso carbons 9,10.
        entry = POLYCYCLIC_DATA["anthracene"]
        m = Chem.MolFromSmiles(entry["canonical_smiles"])
        computed = compute_fused_numbering(m, _ring_atoms(m))
        ints = sorted(v for v in computed.values() if isinstance(v, int))
        assert ints == list(range(1, 11))  # 1..10, no meso renumbered away

    def test_phenanthrene_has_4a_4b_fusion_labels(self):
        entry = POLYCYCLIC_DATA["phenanthrene"]
        m = Chem.MolFromSmiles(entry["canonical_smiles"])
        computed = {k: _norm(v) for k, v in
                    compute_fused_numbering(m, _ring_atoms(m)).items()}
        fusion = {v for v in computed.values() if v[1]}
        assert (4, "a") in fusion
        assert (4, "b") in fusion


@pytest.mark.unit
class TestSystematicNumbering:
    """The systematic peripheral walk reproduces OPSIN for non-special PAHs."""

    # OPSIN-derived ($_AV) atom-ordered locants for a fresh parse of the SMILES.
    _CASES = {
        "tetracene": (
            "C1=CC=CC2=CC3=CC4=CC=CC=C4C=C3C=C12",
            ["1", "2", "3", "4", "4a", "5", "5a", "6", "6a", "7", "8", "9",
             "10", "10a", "11", "11a", "12", "12a"],
        ),
        "chrysene": (
            "C1=CC=CC=2C3=CC=C4C=CC=CC4=C3C=CC12",
            ["1", "2", "3", "4", "4a", "4b", "5", "6", "6a", "7", "8", "9",
             "10", "10a", "10b", "11", "12", "12a"],
        ),
    }

    @pytest.mark.parametrize("name", list(_CASES))
    def test_systematic_walk_matches_opsin(self, name):
        smi, locs = self._CASES[name]
        m = Chem.MolFromSmiles(smi)
        target = {i: _norm(l) for i, l in enumerate(locs)}
        computed = {k: _norm(v) for k, v in
                    compute_fused_numbering(m, _ring_atoms(m)).items()}
        assert computed == target, f"{name}: {computed} != {target}"


@pytest.mark.unit
class TestFailClosed:
    def test_pyrene_peri_fused_returns_none(self):
        m = Chem.MolFromSmiles("c1cc2ccc3cccc4ccc(c1)c2c34")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    def test_azulene_not_all_six_returns_none(self):
        m = Chem.MolFromSmiles("c1ccc2cccc-2cc1")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    def test_quinoline_heterocycle_returns_none(self):
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    def test_benzene_single_ring_returns_none(self):
        m = Chem.MolFromSmiles("c1ccccc1")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None


@pytest.mark.unit
class TestDeterminism:
    _CASES = {
        "naphthalene": "c1ccc2ccccc2c1",
        "anthracene": "c1ccc2cc3ccccc3cc2c1",
        "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
        "tetracene": "c1ccc2cc3cc4ccccc4cc3cc2c1",
        "chrysene": "c1ccc2c(c1)ccc1c3ccccc3ccc21",
    }

    @pytest.mark.parametrize("name", list(_CASES))
    def test_numbering_stable_across_random_spellings(self, name):
        base = Chem.MolFromSmiles(self._CASES[name])
        random.seed(31337)
        fingerprints = set()
        for _ in range(15):
            rsmi = Chem.MolToSmiles(base, doRandom=True, canonical=False)
            m = Chem.MolFromSmiles(rsmi)
            nb = compute_fused_numbering(m, _ring_atoms(m))
            assert nb is not None, f"{name}: returned None on a random spelling"
            # SMILES-order-independent fingerprint: (canonical_rank, locant).
            cr = list(Chem.CanonicalRankAtoms(m, breakTies=False))
            fp = tuple(sorted((cr[a], _norm(l)) for a, l in nb.items()))
            fingerprints.add(fp)
        assert len(fingerprints) == 1, f"{name}: {len(fingerprints)} numberings"
