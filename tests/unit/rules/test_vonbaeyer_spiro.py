"""Task F (v39 CQ5/QM9 finding) -- ``analyze_spiro_universal`` dispiro
descriptor fix for adjacent-spiro small rings.

Bug: for a linear dispiro system whose middle ring's two spiro atoms are
directly bonded (one spiro-to-spiro arc has 0 linking atoms, the other has
1+), ``_walk_ring_between_spiros`` (``rules/spiro.py``) picked the LONGER
unvisited arc as the first middle-ring segment while ``_compute_spiro_segments``
(the descriptor-string builder) always cites the SHORTER arc first. The two
computations silently disagreed whenever a heteroatom broke the arc-length
symmetry, so the emitted descriptor's heteroatom locant did not match what
the descriptor string itself denotes -- an OPSIN round-trip ``inchi_mismatch``
(confirmed at HEAD `87365346`, CQ5-SPY.md).

Governing rule: **P-24.2.2 "Linear polyspiro alicyclic ring systems"**
(``BlueBookV2/BlueBookV2.md:9977``): *"...proceeding consecutively, always by
the SHORTER path, to the other terminal ring through each spiro atom and then
back to the first spiro atom..."*

Fix: ``_walk_ring_between_spiros`` now selects the arc with FEWER unvisited
(linking) atoms first, matching ``_compute_spiro_segments``'s
``min(seg_a, seg_b)``-first convention. Invisible for symmetric middle rings
and all-carbon skeletons (a hydrocarbon automorphism hides the swap) -- both
covered here as non-regression witnesses.
"""
import pytest
from rdkit import Chem

from orthonym.rules.spiro import name_spiro_system
from orthonym.rules.vonbaeyer_universal import analyze_spiro_universal
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


def _inchi(smi: str) -> str:
    from rdkit.Chem.inchi import MolToInchi
    return MolToInchi(Chem.MolFromSmiles(smi))


class TestAdjacentSpiroDescriptorFix:
    """The two CQ5-SPY witnesses: adjacent (directly-bonded) spiro atoms with
    a heteroatom in one terminal ring -- previously RT-FAIL, now RT-PASS."""

    @pytest.mark.unit
    def test_witness1_oxadispiro_roundtrips(self):
        smi = 'C1C2(CCC2)C11CCO1'
        mol = Chem.MolFromSmiles(smi)
        result = name_spiro_system(mol)
        assert result is not None, "must not abstain (0-wrong via None is the fallback, not the goal)"
        name = result[0]
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"

    @pytest.mark.unit
    def test_witness2_oxadispiro_roundtrips(self):
        smi = 'C1CC11CCC11CO1'
        mol = Chem.MolFromSmiles(smi)
        result = name_spiro_system(mol)
        assert result is not None
        name = result[0]
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"

    @pytest.mark.unit
    def test_witness1_via_vonbaeyer_universal(self):
        """analyze_spiro_universal (the T4/best-effort producer CQ5-SPY named)
        must ALSO emit an RT-verifying descriptor -- it feeds the same
        generate_spiro_descriptor/_get_polyspiro_numbering pair."""
        smi = 'C1C2(CCC2)C11CCO1'
        mol = Chem.MolFromSmiles(smi)
        analysis = analyze_spiro_universal(mol)
        assert analysis is not None
        name = analysis.hetero_prefix + analysis.descriptor + 'nonane'
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"

    @pytest.mark.unit
    def test_witness2_via_vonbaeyer_universal(self):
        smi = 'C1CC11CCC11CO1'
        mol = Chem.MolFromSmiles(smi)
        analysis = analyze_spiro_universal(mol)
        assert analysis is not None
        name = analysis.hetero_prefix + analysis.descriptor + 'octane'
        rt = opsin_roundtrip_check(smi, name)
        assert rt['passed'], f"{name} must OPSIN round-trip to the input: {rt}"


class TestExistingSpiroFixturesUnchanged:
    """Non-regression: pre-existing dispiro fixtures (symmetric arcs, or
    all-carbon adjacent-spiro) must emit the SAME string as before the fix --
    the bug is invisible to them by construction (isomorphism / symmetry), so
    this locks that in rather than assuming it."""

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


class TestDeterminism:
    """The arc-order fix itself must not depend on atom order: across several
    randomized SMILES atom orders of each witness, the emitted name must
    ALWAYS round-trip (0-wrong holds regardless of input atom order).

    NOTE -- measured, PRE-EXISTING, OUT-OF-SCOPE finding (not introduced by
    this fix, confirmed present with the unpatched arc-order logic too): the
    literal heteroatom LOCANT is not yet canonical-rank-keyed elsewhere in
    this module (``_build_ring_chain``'s ring1-vs-ring2 tie-break for
    equal-sized terminal rings, and ``_walk_ring_from_spiro``'s traversal
    direction, both iterate raw RDKit neighbour/ring order rather than a
    canonical rank), so the exact oxa-locant can vary by input atom order
    even though the STRUCTURE named never does. That is a separate,
    pre-existing determinism gap in the numbering direction/tie-break choices
    -- out of scope for this bounded fix (see task-F-report.md) -- so this
    test asserts what Task F actually guarantees (round-trip correctness is
    atom-order invariant), not full string identity."""

    @pytest.mark.unit
    def test_witness1_roundtrips_under_every_atom_order(self):
        smi = 'C1C2(CCC2)C11CCO1'
        mol0 = Chem.MolFromSmiles(smi)
        for seed_atom in range(mol0.GetNumAtoms()):
            smi_r = Chem.MolToSmiles(mol0, canonical=False, doRandom=True,
                                      rootedAtAtom=seed_atom)
            mol_r = Chem.MolFromSmiles(smi_r)
            result = name_spiro_system(mol_r)
            assert result is not None, smi_r
            rt = opsin_roundtrip_check(smi_r, result[0])
            assert rt['passed'], (smi_r, result[0], rt)

    @pytest.mark.unit
    def test_witness2_roundtrips_under_every_atom_order(self):
        smi = 'C1CC11CCC11CO1'
        mol0 = Chem.MolFromSmiles(smi)
        for seed_atom in range(mol0.GetNumAtoms()):
            smi_r = Chem.MolToSmiles(mol0, canonical=False, doRandom=True,
                                      rootedAtAtom=seed_atom)
            mol_r = Chem.MolFromSmiles(smi_r)
            result = name_spiro_system(mol_r)
            assert result is not None, smi_r
            rt = opsin_roundtrip_check(smi_r, result[0])
            assert rt['passed'], (smi_r, result[0], rt)
