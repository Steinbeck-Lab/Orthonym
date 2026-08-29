"""Task F (v39 CQ5/QM9 finding) -- ``analyze_spiro_universal`` dispiro
descriptor fix for adjacent-spiro small rings, PLUS round 2's determinism
canonicalization.

Round 1 bug: for a linear dispiro system whose middle ring's two spiro atoms
are directly bonded (one spiro-to-spiro arc has 0 linking atoms, the other
has 1+), ``_walk_ring_between_spiros`` (``rules/spiro.py``) picked the LONGER
unvisited arc as the first middle-ring segment while ``_compute_spiro_segments``
(the descriptor-string builder) always cites the SHORTER arc first. The two
computations silently disagreed whenever a heteroatom broke the arc-length
symmetry, so the emitted descriptor's heteroatom locant did not match what
the descriptor string itself denotes -- an OPSIN round-trip ``inchi_mismatch``
(confirmed at HEAD `87365346`, CQ5-SPY.md).

Round 1 fix: ``_walk_ring_between_spiros`` now selects the arc with FEWER
unvisited (linking) atoms first, matching ``_compute_spiro_segments``'s
``min(seg_a, seg_b)``-first convention (P-24.2.2 "Linear polyspiro alicyclic
ring systems", ``BlueBookV2/BlueBookV2.md:9977``: *"...proceeding
consecutively, always by the SHORTER path..."*). This alone made both
witnesses round-trip, but ROUTED them onto pre-existing non-canonical
heteroatom-locant numbering (``_build_ring_chain``'s ring-order tie-break and
``_walk_ring_from_spiro``'s traversal direction both iterated raw RDKit
neighbour/ring order, not a canonical rank) -- so the emitted locant (still
correct, still round-tripping) varied by input SMILES atom order. Determinism
is a hard gate independent of 0-wrong, so that is a real defect.

Round 2 fix: ``_dispiro_numbering_candidates`` (new) enumerates every
P-24.2.2-legal numbering of a 3-ring dispiro chain -- the three genuine free
choices the Blue Book's own construction rules leave unresolved (which
physical terminal ring is numbered first when the two tie in size; the
traversal direction within each terminal ring; which middle-ring arc goes
first when the two tie in length) -- and ``_get_polyspiro_numbering`` picks
among them DETERMINISTICALLY via P-24.2.4.1.1's lowest-heteroatom-locant rule
(then P-31.1.4.3.4 lowest free-valence locant for a substituent, then a
canonical-rank tiebreak), mirroring ``get_spiro_numbering``'s monospiro
sibling. Both witnesses now emit the single lowest-locant PIN
(``1-oxadispiro[3.0.3.1]nonane`` / ``1-oxadispiro[2.0.2.2]octane``)
deterministically, at both PIN and best-effort tiers.
"""
import pytest
from rdkit import Chem

from orthonym.rules.spiro import _get_alkane_name, name_spiro_system
from orthonym.rules.vonbaeyer_universal import analyze_spiro_universal
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


class TestAdjacentSpiroDescriptorFix:
    """The two CQ5-SPY witnesses: adjacent (directly-bonded) spiro atoms with
    a heteroatom in one terminal ring -- previously RT-FAIL, now RT-PASS at
    the deterministic lowest-locant PIN."""

    @pytest.mark.unit
    def test_witness1_oxadispiro_roundtrips(self):
        smi = 'C1C2(CCC2)C11CCO1'
        mol = Chem.MolFromSmiles(smi)
        result = name_spiro_system(mol)
        assert result is not None, "must not abstain (0-wrong via None is the fallback, not the goal)"
        name = result[0]
        assert name == '1-oxadispiro[3.0.3.1]nonane', name
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"

    @pytest.mark.unit
    def test_witness2_oxadispiro_roundtrips(self):
        smi = 'C1CC11CCC11CO1'
        mol = Chem.MolFromSmiles(smi)
        result = name_spiro_system(mol)
        assert result is not None
        name = result[0]
        assert name == '1-oxadispiro[2.0.2.2]octane', name
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"

    @pytest.mark.unit
    def test_witness1_via_vonbaeyer_universal(self):
        """analyze_spiro_universal (the T4/best-effort producer CQ5-SPY named)
        must ALSO emit an RT-verifying, lowest-locant descriptor -- it feeds
        the same generate_spiro_descriptor/_get_polyspiro_numbering pair.
        Stem computed from ``total_atoms`` (not hardcoded) so this stays
        correct if the descriptor's atom count ever changes."""
        smi = 'C1C2(CCC2)C11CCO1'
        mol = Chem.MolFromSmiles(smi)
        analysis = analyze_spiro_universal(mol)
        assert analysis is not None
        stem = _get_alkane_name(analysis.total_atoms)
        name = analysis.hetero_prefix + analysis.descriptor + stem
        assert name == '1-oxadispiro[3.0.3.1]nonane', name
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"

    @pytest.mark.unit
    def test_witness2_via_vonbaeyer_universal(self):
        smi = 'C1CC11CCC11CO1'
        mol = Chem.MolFromSmiles(smi)
        analysis = analyze_spiro_universal(mol)
        assert analysis is not None
        stem = _get_alkane_name(analysis.total_atoms)
        name = analysis.hetero_prefix + analysis.descriptor + stem
        assert name == '1-oxadispiro[2.0.2.2]octane', name
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"


class TestExistingSpiroFixturesUnchanged:
    """Non-regression: pre-existing dispiro fixtures (symmetric arcs, or
    all-carbon adjacent-spiro) must emit the SAME string as before either
    round of the fix -- the bug (and the round-2 canonicalization) is
    invisible to them by construction (isomorphism / symmetry / no
    heteroatom), so this locks that in rather than assuming it."""

    @pytest.mark.unit
    def test_dispiro_2_1_2_1_octane_unchanged(self):
        mol = Chem.MolFromSmiles('C1CC12CC1(CC1)C2')
        name = name_spiro_system(mol)[0]
        assert name == 'dispiro[2.1.2.1]octane'

    @pytest.mark.unit
    def test_dispiro_4_2_4_2_tetradecane_unchanged(self):
        mol = Chem.MolFromSmiles('C1CCCC12CCC1(CCCC1)CC2')
        name = name_spiro_system(mol)[0]
        assert name == 'dispiro[4.2.4.2]tetradecane'

    @pytest.mark.unit
    def test_dispiro_2_0_2_1_heptane_unchanged_and_now_verified_roundtrip(self):
        """This fixture is the SAME adjacent-spiro (arc lengths 0 and 1)
        topology as the two oxa witnesses, but all-carbon -- the arc swap is
        a graph automorphism here, so the emitted string is BYTE-IDENTICAL
        pre/post-fix. Also lock in that it actually round-trips (it always
        did; not previously asserted by the pre-existing loose test)."""
        smi = 'C1CC12C1(CC1)C2'
        mol = Chem.MolFromSmiles(smi)
        name = name_spiro_system(mol)[0]
        assert name == 'dispiro[2.0.2.1]heptane'
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], rt

    @pytest.mark.unit
    def test_polyspiro_substituent_free_valence_lowest_locant_unchanged(self):
        """Round 2 threaded ``suffix_ring_atoms`` (free-valence bias,
        P-31.1.4.3.4) through ``_get_polyspiro_numbering`` -- previously only
        the monospiro ``get_spiro_numbering`` had it. Locked in via the
        ``tests/unit/rules/test_v27_p3_spiro_engine.py`` PIN correction
        (dispiro[3.2.3.2]dodecan-5-yl, not the old arbitrary -12-yl -- both
        denote the identical molecule, InChIKey HCHGJBAWDHMYAZ-UHFFFAOYSA-N,
        confirmed via OPSIN); re-asserted here as the module-level regression
        guard for this file's scope."""
        from orthonym.rules.ring_substituents import _universal_spiro_substituent_name
        from orthonym.rules.ring_substituents import _extract_ring_submol
        smi = 'C1CCC12CCC1(CCC1)CC2CC(=O)O'
        mol = Chem.MolFromSmiles(smi)
        ring_atoms = tuple(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
        attach = next(
            a for a in ring_atoms
            for nb in mol.GetAtomWithIdx(a).GetNeighbors()
            if not nb.IsInRing() and nb.GetSymbol() == 'C')
        sub, attach_sub = _extract_ring_submol(mol, ring_atoms, attach)
        got = _universal_spiro_substituent_name(sub, attach_sub, allow_mancude=True)
        assert got == 'dispiro[3.2.3.2]dodecan-5-yl', got


class TestDeterminism:
    """Round 2: the emitted NAME STRING (not just RT-pass) must be identical
    across many randomized SMILES atom orders -- the lowest-locant selection
    in ``_get_polyspiro_numbering`` is a canonical-rank-keyed ``min()`` over
    an enumeration that does not depend on input atom order, so the same
    molecule always resolves to the same PIN regardless of how it was
    written. (Round 1 alone could only promise round-trip correctness here;
    see git history for that weaker version -- this supersedes it now that
    round 2 canonicalizes the locant choice itself.)"""

    N_ORDERS = 6

    @pytest.mark.unit
    def test_witness1_name_identical_under_every_atom_order(self):
        smi = 'C1C2(CCC2)C11CCO1'
        mol0 = Chem.MolFromSmiles(smi)
        names = set()
        for seed_atom in range(min(self.N_ORDERS, mol0.GetNumAtoms())):
            smi_r = Chem.MolToSmiles(mol0, canonical=False, doRandom=True,
                                      rootedAtAtom=seed_atom)
            mol_r = Chem.MolFromSmiles(smi_r)
            result = name_spiro_system(mol_r)
            assert result is not None, smi_r
            names.add(result[0])
        assert names == {'1-oxadispiro[3.0.3.1]nonane'}, names

    @pytest.mark.unit
    def test_witness2_name_identical_under_every_atom_order(self):
        smi = 'C1CC11CCC11CO1'
        mol0 = Chem.MolFromSmiles(smi)
        names = set()
        for seed_atom in range(min(self.N_ORDERS, mol0.GetNumAtoms())):
            smi_r = Chem.MolToSmiles(mol0, canonical=False, doRandom=True,
                                      rootedAtAtom=seed_atom)
            mol_r = Chem.MolFromSmiles(smi_r)
            result = name_spiro_system(mol_r)
            assert result is not None, smi_r
            names.add(result[0])
        assert names == {'1-oxadispiro[2.0.2.2]octane'}, names
