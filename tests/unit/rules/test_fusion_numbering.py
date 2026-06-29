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
class TestAngularBranchedNumbering:
    """Angular / branched cata-fused PAH match OPSIN up to automorphism.

    These are the cases the start-atom (P-25.3.3.1.1) and contiguous-row
    (P-25.3.2.3.3a) fixes added — the engine previously mis-numbered every one
    (fusion-carbon locants one position too low).  Compared up to the molecule's
    own automorphism because symmetric systems (D3h triphenylene) have several
    equivalent atom assignments with the same locant set.
    """

    # SMILES + OPSIN ($_AV) atom-ordered locants for a fresh parse.
    _CASES = {
        "triphenylene": (
            "C1=CC=CC=2C3=CC=CC=C3C3=CC=CC=C3C12",
            ["1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8", "8a", "8b",
             "9", "10", "11", "12", "12a", "12b"],
        ),
        "benz[a]anthracene": (
            "C1=CC=CC=2C1=C1C=C3C=CC=CC3=CC1=CC2",
            ["1", "2", "3", "4", "4a", "12b", "12a", "12", "11a", "11", "10",
             "9", "8", "7a", "7", "6a", "6", "5"],
        ),
        "picene": (
            "C1=CC=CC2=CC=C3C4=CC=C5C=CC=CC5=C4C=CC3=C21",
            ["1", "2", "3", "4", "4a", "5", "6", "6a", "6b", "7", "8", "8a",
             "9", "10", "11", "12", "12a", "12b", "13", "14", "14a", "14b"],
        ),
        "benzo[c]chrysene": (
            "C1=C2C=CC3=C4C5=C(C=CC4=CC=C3C2=CC=C1)C=CC=C5",
            ["1", "14a", "14", "13", "12c", "12b", "12a", "8a", "8", "7", "6a",
             "6", "5", "4b", "4a", "4", "3", "2", "9", "10", "11", "12"],
        ),
        "dibenz[a,c]anthracene": (
            "C1=CC=CC2=C1C1=CC3=CC=CC=C3C=C1C1=C2C=CC=C1",
            ["1", "2", "3", "4", "4a", "14b", "14a", "14", "13a", "13", "12",
             "11", "10", "9a", "9", "8b", "8a", "4b", "5", "6", "7", "8"],
        ),
    }

    @pytest.mark.parametrize("name", list(_CASES))
    def test_matches_opsin_up_to_automorphism(self, name):
        smi, locs = self._CASES[name]
        m = Chem.MolFromSmiles(smi)
        target = {i: _norm(l) for i, l in enumerate(locs)}
        computed = compute_fused_numbering(m, _ring_atoms(m))
        assert computed is not None, f"{name}: engine returned None"
        computed = {k: _norm(v) for k, v in computed.items()}
        assert _automorph_equal(m, target, computed), (
            f"{name}: {computed} not automorph-equal to OPSIN {target}"
        )


@pytest.mark.unit
class TestFailClosed:
    def test_pyrene_peri_fused_returns_none(self):
        m = Chem.MolFromSmiles("c1cc2ccc3cccc4ccc(c1)c2c34")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    def test_azulene_not_all_six_returns_none(self):
        m = Chem.MolFromSmiles("c1ccc2cccc-2cc1")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    def test_seven_membered_ring_returns_none(self):
        # v23 13B(a) S2b admits 5/6-membered mixed rings (indole etc.), but a
        # system containing a 7- (or 8-) membered ring still fails closed
        # (mixed >6-ring geometry = S2b.3).
        m = Chem.MolFromSmiles("C1=CC=CC2=CC=CC=CC2=C1")  # heptalene (7,7)
        assert m is not None
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    def test_benzene_single_ring_returns_none(self):
        m = Chem.MolFromSmiles("c1ccccc1")
        assert compute_fused_numbering(m, _ring_atoms(m)) is None

    @pytest.mark.parametrize("name,smi", [
        # Helicenes are a Blue-Book special class (P-25.3.3.1.1 note) and cannot
        # embed planar -> the overcrowding/planarity gate declines them.
        ("pentahelicene", "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC=CC5=C4C3=C12"),
        ("hexahelicene",
         "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=CC=C6C5=C4C3=C12"),
    ])
    def test_helicene_returns_none(self, name, smi):
        m = Chem.MolFromSmiles(smi)
        assert m is not None, f"{name}: bad SMILES"
        assert compute_fused_numbering(m, _ring_atoms(m)) is None, (
            f"{name}: should fail-close (non-planar special class)"
        )


@pytest.mark.unit
class TestHeterocycleNumbering:
    """v23 13B(a) S2a — all-6 fused HETEROCYCLES are numbered via the relaxed
    gate + heteroatom-lowest-locant cascade (P-25.3.3.1.2) and the Table-2.8
    anthracene-type "special numbering" fixed maps.  Locants verified against
    OPSIN ``-o extendedsmi``."""

    @pytest.mark.parametrize("smi,sym,expected", [
        ("c1ccc2ncccc2c1", "N", 1),    # quinoline    — N is position 1
        ("c1ccc2cnccc2c1", "N", 2),    # isoquinoline — N is position 2
        ("c1ccc2nc3ccccc3cc2c1", "N", 10),    # acridine  — meso N is 10
        ("c1ccc2c(c1)Cc1ccccc1O2", "O", 10),  # 9H-xanthene — meso O is 10
        ("c1ccc2c(c1)Cc1ccccc1S2", "S", 10),  # 9H-thioxanthene — meso S is 10
    ])
    def test_single_heteroatom_locant(self, smi, sym, expected):
        m = Chem.MolFromSmiles(smi)
        nb = compute_fused_numbering(m, _ring_atoms(m))
        assert nb is not None, f"{smi}: engine declined a valid all-6 heterocycle"
        het = [nb[a.GetIdx()] for a in m.GetAtoms() if a.GetSymbol() == sym]
        assert het == [expected], f"{smi}: {sym} locants {het} != [{expected}]"

    @pytest.mark.parametrize("smi,sym,expected", [
        ("c1ccc2[nH]ccc2c1", "N", 1),       # 1H-indole       — NH is 1
        ("c1ccc2occc2c1", "O", 1),          # 1-benzofuran    — O is 1
        ("c1ccc2[nH]cnc2c1", "N", [1, 3]),  # 1H-benzimidazole — NH=1, N=3
        ("c1ccn2cccc2c1", "N", 4),          # indolizine      — bridgehead N is 4
    ])
    def test_56_bicyclic_heteroatom(self, smi, sym, expected):
        # v23 13B(a) S2b — (5,6) bicyclic mixed-ring numbering via the general
        # regular-polygon embedding + cascade (incl. indicated-H tier).
        m = Chem.MolFromSmiles(smi)
        nb = compute_fused_numbering(m, _ring_atoms(m))
        assert nb is not None, f"{smi}: engine declined a valid (5,6) heterocycle"
        het = sorted(nb[a.GetIdx()] for a in m.GetAtoms() if a.GetSymbol() == sym)
        exp = sorted(expected) if isinstance(expected, list) else [expected]
        assert het == exp, f"{smi}: {sym} locants {het} != {exp}"

    def test_carbazole_nh_is_9_special_numbering(self):
        # 9H-carbazole (Blue Book Table 2.8 entry 6, "special numbering"): NH=9,
        # fusion 4a/4b/8a/9a — via _FIXED_NUMBERING_SYSTEMS (the systematic
        # walk/scorer mis-selects this carbazole-shape, NH not lowest-locant).
        m = Chem.MolFromSmiles("c1ccc2c(c1)[nH]c1ccccc12")
        nb = compute_fused_numbering(m, _ring_atoms(m))
        assert nb is not None
        nloc = [nb[a.GetIdx()] for a in m.GetAtoms() if a.GetSymbol() == "N"]
        assert nloc == [9], f"carbazole NH locant {nloc} != [9]"

    def test_pteridine_nitrogen_set_is_1_3_5_8(self):
        # The DATA-01 follow-on bug: stored had N-set {2,4,5,8}; correct is
        # {1,3,5,8} (Blue Book Table 2.8 entry 7).
        m = Chem.MolFromSmiles("c1cnc2ncncc2n1")
        nb = compute_fused_numbering(m, _ring_atoms(m))
        nset = sorted(nb[a.GetIdx()] for a in m.GetAtoms() if a.GetSymbol() == "N")
        assert nset == [1, 3, 5, 8], f"pteridine N-set {nset} != [1,3,5,8]"

    def test_acridine_meso_positions_are_9_and_10(self):
        # anthracene-type fixed numbering: the two central-ring meso atoms take
        # the highest locants 9 and 10 (the systematic walk would not).
        m = Chem.MolFromSmiles("c1ccc2nc3ccccc3cc2c1")
        nb = compute_fused_numbering(m, _ring_atoms(m))
        ints = sorted(v for v in nb.values() if isinstance(v, int))
        assert ints == list(range(1, 11))  # 1..10, both meso atoms numbered last


@pytest.mark.unit
class TestDeterminism:
    _CASES = {
        "naphthalene": "c1ccc2ccccc2c1",
        "anthracene": "c1ccc2cc3ccccc3cc2c1",
        "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
        "tetracene": "c1ccc2cc3cc4ccccc4cc3cc2c1",
        "chrysene": "c1ccc2c(c1)ccc1c3ccccc3ccc21",
        # angular / branched (the fixed cases) must also be spelling-stable
        "triphenylene": "c1ccc2c(c1)c1ccccc1c1ccccc21",
        "benz[a]anthracene": "c1ccc2cc3c(ccc4ccccc43)cc2c1",
        "picene": "c1ccc2c(c1)ccc1c2ccc2c1ccc1ccccc12",
        "benzo[c]chrysene": "C1=C2C=CC3=C4C5=C(C=CC4=CC=C3C2=CC=C1)C=CC=C5",
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
