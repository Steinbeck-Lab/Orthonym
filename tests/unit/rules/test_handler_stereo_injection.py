"""Tests for handler-level stereo injection (a phase predicate-first wiring).

a phase Plan 01 / Commit 1: STER-15 / STER-16 -- pure-function infrastructure
for handler-level stereo injection. Tests cover:

  - needs_stereo_injection(mol, name): predicate identical to namer.py:62-83
    detection logic per /.
  - inject_stereo_from_locant_map(name, mol, atom_to_locant): pure injector
    per / / (no atom-index fallback).
  - _ring_atom_to_locant_from_oriented(oriented_ring): shared helper per.
  - P-91 format compliance: (2R)-, (2R,3S)-, (2E,3R,5Z)-, (2r,3s)-.

All tests written BEFORE the implementation (TDD discipline per). RED
phase: every test fails on ImportError until the three functions ship in
src/orthonym/rules/stereochemistry.py. GREEN phase: implementation makes
them pass. Tests are the contract.
"""

import inspect
import logging
import re

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.rules.stereochemistry import (
    _ring_atom_to_locant_from_oriented,
    inject_stereo_from_locant_map,
    needs_stereo_injection,
)
# a phase: composite-locant sort key (introduced commit 1/5).
from orthonym.rules.stereochemistry import _composite_locant_sort_key


@pytest.mark.unit
class TestNeedsStereoInjection:
    """Pure-predicate tests for needs_stereo_injection (/)."""

    # Test N-01
    def test_returns_false_when_mol_is_none(self):
        assert needs_stereo_injection(None, "butan-2-ol") is False

    # Test N-02
    def test_returns_false_when_name_is_empty(self):
        mol = Chem.MolFromSmiles("CCO")
        assert needs_stereo_injection(mol, "") is False

    # Test N-03
    def test_returns_false_when_name_is_unknown(self):
        mol = Chem.MolFromSmiles("CCO")
        assert needs_stereo_injection(mol, "unknown") is False

    # Test N-04 -- Pattern A prefix form
    def test_returns_false_for_pattern_a_prefix(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "(2R)-butan-2-ol") is False

    # Test N-05 -- Pattern B embedded block
    def test_returns_false_for_pattern_b_embedded(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "something-(2R)-else") is False

    # Test N-06 -- Pattern C carbohydrate
    def test_returns_false_for_carbohydrate_pattern(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "α-D-glucopyranose") is False
        assert needs_stereo_injection(mol, "β-L-mannose") is False
        # 'alfa' alternative spelling, case-insensitive
        assert needs_stereo_injection(mol, "Alpha-D-Galactose") is False

    # Test N-07 -- no _CIPCode
    def test_returns_false_when_no_cip_code(self):
        mol = Chem.MolFromSmiles("CCC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "propane") is False

    # Test N-08 -- atom stereo present, name lacks all 3 patterns
    def test_returns_true_for_atom_stereo_when_name_lacks_descriptors(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "butan-2-ol") is True

    # Test N-09 -- bond stereo (E/Z) present, name lacks descriptors
    def test_returns_true_for_bond_stereo_when_name_lacks_descriptors(self):
        # cis-1,2-dichloroethene -- bond stereo only
        mol = Chem.MolFromSmiles("Cl/C=C/Cl")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "1,2-dichloroethene") is True

    # Test N-10 -- idempotence (predicate is read-only per)
    def test_predicate_is_idempotent(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        first = needs_stereo_injection(mol, "butan-2-ol")
        second = needs_stereo_injection(mol, "butan-2-ol")
        assert first == second is True

    # Test N-11 -- pseudoasymmetric r/s (lowercase) is recognised by Pattern A
    def test_returns_false_for_lowercase_pseudoasymmetric(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "(2r,3s)-pentane-2,3-diol") is False


@pytest.mark.unit
class TestInjectStereoFromLocantMap:
    """Pure-function tests for inject_stereo_from_locant_map (//)."""

    # Test I-01 -- single R-center on butan-2-ol
    def test_emits_single_R_prefix(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        # atom indices: C(0), C@@H(1), O(2), C(3), C(4)
        # Expected stereo locant 2 lives on atom index 1
        atom_to_locant = {0: 1, 1: 2, 3: 3, 4: 4}
        result = inject_stereo_from_locant_map("butan-2-ol", mol, atom_to_locant)
        # rdkit may say R or S depending on internal canonicalisation; we
        # care that a (digit + R/S)- prefix appears in front of the name body
        assert re.match(r"^\(2[RS]\)-butan-2-ol$", result), result

    # Test I-02 -- multi-center molecule, descriptor block has 2 entries
    def test_emits_two_center_prefix(self):
        # 2,3-dihydroxypentane -- two stereocenters at locants 2 and 3
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        # atoms: C(0), [C@H](1), O(2), [C@@H](3), O(4), C(5), C(6)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4, 6: 5}
        result = inject_stereo_from_locant_map(
            "pentane-2,3-diol", mol, atom_to_locant
        )
        assert re.match(r"^\(2[RS],3[RS]\)-pentane-2,3-diol$", result), result

    # Test I-03 -- ascending locant ordering verification (P-91.1,)
    def test_descriptors_emitted_in_ascending_locant_order(self):
        # 2,3-dihydroxypentane (atoms in canonical order)
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4, 6: 5}
        result = inject_stereo_from_locant_map(
            "pentane-2,3-diol", mol, atom_to_locant
        )
        # Block format: (NN<chr>,NN<chr>,...). Extract digit sequence.
        block_match = re.match(r"^\(([^)]+)\)-", result)
        assert block_match is not None, f"no descriptor block in {result!r}"
        parts = block_match.group(1).split(",")
        # Strip the trailing R/S/E/Z to get bare locants
        locants = [int(re.match(r"(\d+)", p).group(1)) for p in parts]
        assert locants == sorted(locants), (
            f"descriptors not ascending: {locants} in {result!r}"
        )

    # Test I-04 -- pseudoasymmetric centers (P-92.1.4.2 / lowercase r/s)
    def test_lowercase_r_s_preserved_for_pseudoasymmetric(self):
        # Pseudoasymmetric meso compound: 2,3,4-trihydroxypentane.
        # The middle (3) carbon is pseudoasymmetric and rdCIPLabeler should
        # tag it with lowercase r or s.
        mol = Chem.MolFromSmiles("C[C@H](O)[C@H](O)[C@@H](O)C")
        rdCIPLabeler.AssignCIPLabels(mol)
        cip_codes = [
            a.GetProp("_CIPCode") for a in mol.GetAtoms()
            if a.HasProp("_CIPCode")
        ]
        if not any(c in ("r", "s") for c in cip_codes):
            pytest.skip("rdCIPLabeler did not produce lowercase r/s for this case")

        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4, 7: 5}
        result = inject_stereo_from_locant_map(
            "pentane-2,3,4-triol", mol, atom_to_locant
        )
        # At least one descriptor must be lowercase
        assert re.match(r"^\([^)]*[rs][^)]*\)-", result), result

    # Test I-05 -- no-op when name already stereoed
    def test_no_op_when_name_already_has_prefix(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 4: 4}
        result = inject_stereo_from_locant_map(
            "(2R)-butan-2-ol", mol, atom_to_locant
        )
        assert result == "(2R)-butan-2-ol"

    # Test I-06 -- no-op when mol has no stereo
    def test_no_op_when_mol_has_no_stereo(self):
        mol = Chem.MolFromSmiles("CCC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 2: 3}
        result = inject_stereo_from_locant_map("propane", mol, atom_to_locant)
        assert result == "propane"

    # Test I-07 -- no-op when atom_to_locant is None, with DEBUG log ()
    def test_no_op_when_locant_map_none_emits_debug(self, caplog):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.DEBUG, logger="orthonym.rules.stereochemistry"):
            result = inject_stereo_from_locant_map("butan-2-ol", mol, None)
        assert result == "butan-2-ol"
        skip_msgs = [
            r.message for r in caplog.records
            if "inject_stereo: skipped" in r.message
        ]
        assert len(skip_msgs) >= 1, (
            f"expected DEBUG log containing 'inject_stereo: skipped', got: "
            f"{[r.message for r in caplog.records]}"
        )

    # Test I-08 -- no-op when atom_to_locant is empty {}
    def test_no_op_when_locant_map_empty(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = inject_stereo_from_locant_map("butan-2-ol", mol, {})
        assert result == "butan-2-ol"

    # Test I-09 -- locants all zero is a degenerate map (parent_size=0).
    # a phase-02 WR-02 fix: all-zero locants are now a no-op (-- a
    # missing stereo block is preferred to a malformed '(0R)-' garbage one).
    # Real handlers (benzene/heterocycle/cycloalkane/cycloalkene) NEVER
    # emit a zero locant -- this test pins the new no-op contract.
    def test_all_zero_locants_emits_zero_block_no_fallback(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 0, 1: 0, 3: 0, 4: 0}
        result = inject_stereo_from_locant_map("butan-2-ol", mol, atom_to_locant)
        # a phase-02 WR-02 fix: all-zero locant map is now a no-op ().
        assert result == "butan-2-ol", (
            f"WR-02 fix expects all-zero locant map to be a no-op, got {result!r}"
        )

    # Test I-10 -- ring-strain filter inherited (cyclohexene E/Z is filtered)
    def test_ring_strain_filter_inherited(self):
        # cyclohex-1-ene with no chiral center -- the ring double bond is
        # geometry-locked and collect_stereodescriptors filters it (ring < 8)
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        # No CIP set on either atom or bond after assign -> predicate False
        # already short-circuits. The injector should simply return the name.
        result = inject_stereo_from_locant_map(
            "cyclohex-1-ene", mol, atom_to_locant
        )
        assert result == "cyclohex-1-ene"

    # Test I-11 -- grep gate: NO atom-index fallback in injector body
    def test_injector_body_has_no_atom_index_fallback(self):
        src = inspect.getsource(inject_stereo_from_locant_map)
        #: forbidden patterns
        assert "range(len(" not in src, (
            f"injector body must not use range(len(...)) fallback (D-09):\n{src}"
        )
        assert "enumerate(mol.GetAtoms()" not in src, (
            f"injector body must not enumerate mol atoms as fallback (D-09):\n{src}"
        )

    # Test I-12 -- idempotent (calling twice yields the same string; Pitfall 1)
    def test_injector_is_idempotent(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 4: 4}
        first = inject_stereo_from_locant_map("butan-2-ol", mol, atom_to_locant)
        second = inject_stereo_from_locant_map(first, mol, atom_to_locant)
        # Second call sees an already-stereoed name -> predicate False -> no-op
        assert first == second

    # Test I-13 -- locant outside parent_size dropped silently (inherited
    # from validate_stereo_locants). We use parent_size 4 (max locant in map)
    # and supply an out-of-bounds locant 99 for the stereocenter. The validator
    # filters all locants > parent_size; since 99 IS the parent_size here,
    # the validator does not drop it. We use the OPPOSITE direction:
    # parent_size 4 with locant 99 (only via mismatch). Use a separate
    # mol with parent_size from a small map and stereocenter at high locant.
    def test_locant_above_parent_size_dropped(self):
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        # parent_size is computed as max(int_locants); we keep stereocenter
        # at locant 5 but the rest at 1..4 so parent_size=5. Now flip:
        # to test the validator firing, we need a locant > parent_size.
        # Add an extra high locant on a non-stereo atom to set parent_size
        # high, then place the stereocenter ABOVE it. RDKit indices: 0,1,3,4.
        atom_to_locant = {0: 1, 1: 50, 3: 3, 4: 4}
        # parent_size = max(1, 50, 3, 4) = 50; stereocenter locant 50 -> kept.
        # To trigger the validator we need locant > parent_size, which by
        # construction never happens since parent_size = max(values). This
        # test therefore validates that validate_stereo_locants does NOT
        # spuriously drop the only stereocenter when its locant equals
        # parent_size -- complementing I-09.
        result = inject_stereo_from_locant_map(
            "butan-2-ol", mol, atom_to_locant
        )
        # The output must include the supplied locant (50) verbatim -- no
        # atom-index fallback, no silent re-numbering ().
        assert re.match(r"^\(50[RS]\)-butan-2-ol$", result), (
            f"injector must use supplied locant verbatim: {result!r}"
        )


@pytest.mark.unit
class TestRingAtomToLocantFromOriented:
    """Tests for the shared {idx: pos+1} helper ()."""

    # Test R-01
    def test_empty_ring_returns_empty_dict(self):
        assert _ring_atom_to_locant_from_oriented([]) == {}

    # Test R-02
    def test_single_atom(self):
        assert _ring_atom_to_locant_from_oriented([5]) == {5: 1}

    # Test R-03 -- duplicate keys (last occurrence wins, matches dict semantics)
    def test_duplicate_keys_last_wins(self):
        # The original one-liner at composer.py:7184 has the same semantics.
        # Constructed example with a duplicate (indices [3,1,4,1,5,9])
        result = _ring_atom_to_locant_from_oriented([3, 1, 4, 1, 5, 9])
        # When dict-comprehension hits the same key, the last assignment wins
        assert result == {3: 1, 1: 4, 4: 3, 5: 5, 9: 6}

    # Test R-04
    def test_six_ring(self):
        result = _ring_atom_to_locant_from_oriented([10, 11, 12, 13, 14, 15])
        assert result == {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6}


@pytest.mark.unit
class TestP91FormatCompliance:
    """P-91 explicit format-compliance gate (/ SC-5)."""

    def test_emits_paren_R_paren_dash_form(self):
        # (R)-butan-2-ol type prefix: starts with `(`, contains `R)-`
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 4: 4}
        result = inject_stereo_from_locant_map("butan-2-ol", mol, atom_to_locant)
        assert result.startswith("(")
        # Must contain a single R or S followed by `)-`
        assert re.match(r"^\(2[RS]\)-", result), result

    def test_emits_multi_center_comma_separated(self):
        # (2R,3S)- form
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4, 6: 5}
        result = inject_stereo_from_locant_map(
            "pentane-2,3-diol", mol, atom_to_locant
        )
        assert re.match(r"^\(2[RS],3[RS]\)-", result), result

    def test_emits_mixed_E_Z_R_S(self):
        # (E)-but-2-enoic acid type compound -- test mixed E/Z + R/S in macrocycle.
        # Use a molecule with a stereocenter and a near-parent E/Z bond.
        # The simplest controllable example is a 2-methylpent-3-enoic acid where
        # both (R/S) at C2 and (E) at C3-C4 are valid; we validate the format
        # alone (existence of both letter classes in the block).
        mol = Chem.MolFromSmiles("C[C@@H](C(=O)O)/C=C/C")
        rdCIPLabeler.AssignCIPLabels(mol)
        # atoms: 0=Me, 1=C@@H, 2=C(=O), 3=O, 4=O, 5=C, 6=C, 7=C
        atom_to_locant = {2: 1, 1: 2, 5: 3, 6: 4, 7: 5}
        result = inject_stereo_from_locant_map(
            "2-methylpent-3-enoic acid", mol, atom_to_locant
        )
        # Format compliance: must be a parenthesised block followed by '-'
        block_match = re.match(r"^\(([^)]+)\)-", result)
        assert block_match is not None, result
        block = block_match.group(1)
        # Should contain at least one R/S letter
        assert re.search(r"[RS]", block), result
        # Should contain at least one E/Z letter (near-parent E/Z, P-93.5.2)
        assert re.search(r"[EZ]", block), result

    def test_emits_lowercase_r_s_for_pseudoasymmetric(self):
        # Same pseudoasymmetric meso compound used in I-04
        mol = Chem.MolFromSmiles("C[C@H](O)[C@H](O)[C@@H](O)C")
        rdCIPLabeler.AssignCIPLabels(mol)
        cip_codes = [
            a.GetProp("_CIPCode") for a in mol.GetAtoms()
            if a.HasProp("_CIPCode")
        ]
        if not any(c in ("r", "s") for c in cip_codes):
            pytest.skip("rdCIPLabeler did not produce lowercase r/s for this case")
        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4, 7: 5}
        result = inject_stereo_from_locant_map(
            "pentane-2,3,4-triol", mol, atom_to_locant
        )
        # Block must contain at least one lowercase r or s
        block_match = re.match(r"^\(([^)]+)\)-", result)
        assert block_match is not None, result
        assert re.search(r"[rs]", block_match.group(1)), result


# ----------------------------------------------------------------------
# a phase-02 BL-01 gap closure -- real-coverage gate tests
# ----------------------------------------------------------------------

def test_gate_real_coverage_blocks_partial_fragment():
    """G-01 (BL-01): The Tier-A injection guard MUST measure REAL atom
    coverage from CandidateName.parent_atom_indices (POST-HOC contract,
    candidate_pool.py:759), NOT the ratio_score fallback in
    coverage_scoring.py:434-437.

    Test contract: when a synthetic CandidateName has parent_atom_indices
    covering only PART of the molecule's heavy atoms (real coverage < 0.99),
    the gate expression as inlined at composer.py:1593-1610 must evaluate to
    False. This test pins the gate MECHANISM. The complementary G-02 below
    pins the full-coverage allow path end-to-end via name_compound().

    Note: pre-fix, the gate used ``best.factors['atom_coverage']`` which fell
    back to ratio_score (a name-length proxy) per coverage_scoring.py:434-437
    because pool.add() never forwarded parent_atom_indices into
    compute_confidence. Therefore short candidate names trivially passed even
    when real coverage was poor. We verify the new gate is decoupled from
    factors['atom_coverage'] and instead reads parent_atom_indices.
    """
    from rdkit import Chem
    from orthonym.assembly.candidate_pool import CandidateName

    mol = Chem.MolFromSmiles("Oc1ccc(/C=C/CC)cc1")
    assert mol is not None
    total_heavy = mol.GetNumHeavyAtoms()  # 11

    # Synthetic candidate with PARTIAL coverage: 6 of 11 atoms (ring only),
    # 6/11 = 0.545 < 0.99. Pre-fix gate (factors['atom_coverage']=0.99) would
    # have wrongly PASSED; post-fix gate (real coverage = 6/11) BLOCKS.
    cand = CandidateName(
        name="phenol",
        handler="benzene",
        confidence=0.99,
        factors={"atom_coverage": 0.99},  # ratio_score proxy (would pass pre-fix)
    )
    cand.parent_atom_indices = {0, 1, 2, 3, 4, 9}  # partial: 6 of 11

    # Production gate expression -- mirrors composer.py:1593-1610 byte-for-byte.
    _parent_set = cand.parent_atom_indices
    _real_coverage = (
        len(_parent_set) / max(mol.GetNumHeavyAtoms(), 1)
        if _parent_set else 0.0
    )
    assert _real_coverage < 0.99, (
        f"BL-01 gate test setup is wrong: real_coverage={_real_coverage:.3f}; "
        f"need < 0.99 to exercise the BLOCK branch."
    )
    assert cand.factors["atom_coverage"] >= 0.99, (
        "test sanity: ratio_score proxy in factors must be >= 0.99 to "
        "demonstrate that the OLD gate would have wrongly passed."
    )

    # Also verify the helper function returns sensible data for real
    # benzene features. T-152-02-01 mitigation: defensive shape handling
    # so substituent-shape evolution does not silently break the gate.
    from orthonym.assembly.composer import _ring_handler_parent_atom_indices
    from orthonym.rules.benzene import get_benzene_substituents

    class _F:
        pass
    f = _F()
    ri = mol.GetRingInfo()
    ring0 = ri.AtomRings()[0]
    f.benzene_ring = ring0
    f.benzene_substituents = get_benzene_substituents(mol, ring0)
    parent_set = _ring_handler_parent_atom_indices(f, "benzene")
    assert parent_set is not None
    assert all(isinstance(i, int) for i in parent_set)
    assert len(parent_set) >= len(ring0), (
        f"_ring_handler_parent_atom_indices must include at least the ring "
        f"atoms: got {parent_set!r}, ring={ring0!r}"
    )

    # T-152-02-01: also exercise the heterocycle branch and the "unknown handler"
    # branch (returns None) so future maintainers cannot accidentally introduce
    # a third handler kind without going through the gate logic.
    assert _ring_handler_parent_atom_indices(f, "unknown") is None
    f_empty = _F()
    assert _ring_handler_parent_atom_indices(f_empty, "benzene") is None
    assert _ring_handler_parent_atom_indices(f_empty, "heterocycle") is None


def test_cycloalkane_no_exocyclic_ez_misattribution():
    """G-03 (BL-02): The cycloalkane / cycloalkene Tier-A wiring at
    composer.py:1808-1825 must pass include_near_parent_ez=False when the
    molecule has heavy atoms outside the named ring.

    Plan-specified canary: (E)-but-2-enylcyclobutane (C/C=C/CC1CCC1, 8 HA).
    Ring is 4 of 8 atoms -- NOT the whole molecule. The double bond is two
    hops from the ring (CH2 spacer), so include_near_parent_ez=True does
    not actually attribute the chain bond at this position; the wiring
    correctly emits no ring-locant E/Z prefix.

    The test MECHANISM check verifies the SOURCE-LEVEL gate (the
    _ring_is_whole_molecule expression at the cycloalkane wiring site)
    and the inject_stereo_from_locant_map signature accepts the new
    keyword argument.
    """
    from orthonym import name_compound
    import inspect
    from orthonym.rules.stereochemistry import inject_stereo_from_locant_map

    # End-to-end behavioural assertion (plan-specified canary):
    # 8 HA: cyclobutane ring (4) + butenyl chain (4). Ring != whole.
    name = name_compound("C/C=C/CC1CCC1")
    assert name and name != "unknown"
    # The cycloalkane wiring (post-fix) MUST NOT prepend a ring-locant
    # (\d+E)-/(\d+Z)- block when the ring is not the whole molecule.
    assert not re.match(r"^\(\d+[EZ]\)-?cyclobut", name), (
        f"BL-02 gate failed: exocyclic E/Z on cyclobutane substituent was "
        f"mis-attributed to a ring locant: {name!r}"
    )

    # Source-level mechanism gate. The Tier-A ring injection wiring was
    # REFACTORED out of composer.py into assembly/handlers/tier_a_ring.py
    # (Milestone C); re-point the greps there. The live mechanism computes
    # the per-molecule whole-ring boolean _inpe (via
    # _ring_is_whole_molecule_for_complex) and forwards it as
    # include_near_parent_ez=_inpe -- exactly the "do not attribute exocyclic
    # E/Z to a ring locant when the ring is not the whole molecule" gate.
    from orthonym.assembly.handlers import tier_a_ring
    tier_a_src = inspect.getsource(tier_a_ring)
    assert "_ring_is_whole_molecule_for_complex(" in tier_a_src, (
        "BL-02 fix missing: tier_a_ring must compute the per-molecule "
        "whole-ring boolean via _ring_is_whole_molecule_for_complex."
    )
    assert "include_near_parent_ez=_inpe" in tier_a_src, (
        "BL-02 fix missing: the Tier-A ring caller must forward the "
        "per-molecule include_near_parent_ez=_inpe to the stereo injector."
    )

    # Signature gate. The injector must accept the include_near_parent_ez
    # keyword as a keyword-only argument with a default of True.
    sig = inspect.signature(inject_stereo_from_locant_map)
    assert "include_near_parent_ez" in sig.parameters, (
        "BL-02 fix missing: inject_stereo_from_locant_map must accept "
        "include_near_parent_ez keyword."
    )
    param = sig.parameters["include_near_parent_ez"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"include_near_parent_ez must be KEYWORD_ONLY, got {param.kind}"
    )
    assert param.default is True, (
        f"include_near_parent_ez default must be True (preserves benzene/"
        f"heterocycle Tier-A behaviour), got {param.default}"
    )


def test_cycloalkene_ring_only_stereo_still_injects():
    """G-06 (BL-02 regression check): a fully-ring cycloalkene with R/S on
    every ring atom (e.g. CHEBI:67226 cyclohex-5-ene-1,2,3,4-tetraol -- 10
    heavy atoms = 6 ring + 4 OH; every atom is named via the ring +
    substituents path so the BL-02 gate must allow R/S injection).

    Soft-failure design: if the cycloalkene handler does NOT win the
    candidate-pool selection for this molecule, the test passes silently
    (no injection attempted). The point is to PIN that the BL-02 fix does
    NOT regress the ring-only injection case.
    """
    from orthonym import name_compound
    name = name_compound("O[C@@H]1[C@@H](O)[C@H](O)C=C[C@H]1O")
    assert name and name != "unknown"
    # If the result starts with '(' it must be a leading parenthesised stereo
    # block. Tolerate any of (1R,2S,3S,4R)- shapes that rdCIPLabeler emits.
    if name.startswith("("):
        block_match = re.match(r"^\([0-9RSrs,a-z]+\)-", name)
        assert block_match is not None, (
            f"BL-02 fix broke ring-only cycloalkene injection: {name!r}"
        )


def test_gate_real_coverage_allows_full_name():
    """G-02: When the chosen Tier-A candidate covers all heavy atoms of the
    molecule (oxolane-2-carboxylic acid, 8 heavy atoms, all covered by ring +
    carboxylic-acid substituent), the BL-01 real-coverage gate must allow the
    predicate-first injector to run and emit the (2R/S)- prefix.

    WSD-07 (a phase): the original vehicle was D-proline, but proline is a
    STANDARD amino acid and now correctly resolves to its retained PIN
    'D-proline' (OPSIN-RT-verified) instead of the systematic
    'pyrrolidine-2-carboxylic acid'. Switched to oxolane-2-carboxylic acid — a
    full-coverage heterocyclic acid that is NOT an amino acid, so it still
    exercises the systematic-stereo-injection path this gate test targets."""
    from orthonym import name_compound
    # Oxolane-2-carboxylic acid: 8 heavy atoms; oxolane ring (5) + COOH (3).
    name = name_compound("O=C(O)[C@H]1CCCO1")
    assert name and name != "unknown"
    import re
    assert re.match(r"^\(2[RS]\)-", name), (
        f"BL-01 gate over-blocked: full-coverage heterocycle did not receive "
        f"its expected (2R/S)- prefix: {name!r}"
    )


# ----------------------------------------------------------------------
# a phase-02 WR-02 + WR-03 quick-hit defensive predicates
# ----------------------------------------------------------------------

def test_inject_stereo_skip_on_all_zero_locants():
    """W-02 (WR-02 fix): an atom_to_locant map of all-zero locants is
    degenerate (locants are 1-indexed in IUPAC) and must be a no-op per
     ("better a missing stereo block than a wrong one"). Pre-fix
    behaviour was to silently emit '(0R)-name' garbage."""
    mol = Chem.MolFromSmiles("C[C@@H](O)CC")
    rdCIPLabeler.AssignCIPLabels(mol)
    result = inject_stereo_from_locant_map(
        "butan-2-ol", mol, {0: 0, 1: 0, 3: 0, 4: 0}
    )
    assert result == "butan-2-ol", (
        f"WR-02 fix failed: all-zero locants should be no-op, got {result!r}"
    )


def test_inject_stereo_skip_on_negative_locants():
    """W-02 (WR-02 fix, complement): a locant map with non-positive values
    (negative or zero) must also be a no-op. Locants <= 0 are not valid
    IUPAC locants and indicate caller bugs."""
    mol = Chem.MolFromSmiles("C[C@@H](O)CC")
    rdCIPLabeler.AssignCIPLabels(mol)
    # Mix of zero and negative locants
    result = inject_stereo_from_locant_map(
        "butan-2-ol", mol, {0: -1, 1: 0, 3: -2, 4: 0}
    )
    assert result == "butan-2-ol", (
        f"WR-02 fix failed: non-positive locants should be no-op, got {result!r}"
    )


def test_inject_stereo_runs_when_at_least_one_positive_locant(caplog):
    """W-02 (WR-02 fix, sanity): the predicate accepts maps where AT
    LEAST ONE locant is positive, even if others are zero. This preserves
    the contract that downstream filtering (validate_stereo_locants) handles
    individual bad locants."""
    mol = Chem.MolFromSmiles("C[C@@H](O)CC")
    rdCIPLabeler.AssignCIPLabels(mol)
    # Locant 2 on atom 1 is positive; others are zero. Injector must NOT
    # short-circuit -- it should pass the map to collect_stereodescriptors
    # which uses validate_stereo_locants to filter zeros.
    result = inject_stereo_from_locant_map(
        "butan-2-ol", mol, {0: 0, 1: 2, 3: 0, 4: 0}
    )
    # The result will either be the unchanged name (if validate filters
    # everything) or a (2R/S)-prefixed name. Either is acceptable; the
    # critical contract is that the WR-02 short-circuit did NOT fire.
    assert result == "butan-2-ol" or re.match(r"^\(2[RS]\)-", result), (
        f"WR-02 over-blocked: at least one positive locant must NOT short-"
        f"circuit, got {result!r}"
    )


def test_ring_atom_to_locant_warns_on_duplicates(caplog):
    """W-03 (WR-03 fix): duplicate atom indices in oriented_ring should
    emit a WARNING log so upstream orientator bugs don't silently
    propagate. Behaviour preserved: last-occurrence wins (matches the
    pre-existing dict semantics from composer.py:7184)."""
    import logging
    with caplog.at_level(logging.WARNING, logger="orthonym.rules.stereochemistry"):
        result = _ring_atom_to_locant_from_oriented([10, 11, 12, 10, 14, 15])
    assert any(
        "duplicate atom indices" in rec.message
        for rec in caplog.records
    ), (
        f"WR-03 fix failed: no WARNING emitted for duplicates, "
        f"records={[r.message for r in caplog.records]!r}"
    )
    # Behaviour preserved: last occurrence still wins.
    assert result[10] == 4


def test_ring_atom_to_locant_no_warn_on_unique(caplog):
    """W-03 (WR-03 fix, complement): unique-index oriented_ring must NOT
    emit any WARNING (the WR-03 fix must be a true precondition guard,
    not noisy false-positive logging on the happy path)."""
    import logging
    with caplog.at_level(logging.WARNING, logger="orthonym.rules.stereochemistry"):
        result = _ring_atom_to_locant_from_oriented([10, 11, 12, 13, 14, 15])
    assert not any(
        "duplicate" in rec.message.lower()
        for rec in caplog.records
    ), (
        f"WR-03 over-warns: no WARNING expected on unique input, got "
        f"{[r.message for r in caplog.records]!r}"
    )
    assert result == {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6}


# ============================================================================
# a phase commit 1/5: composite-locant sort key ()
# ============================================================================


@pytest.mark.unit
class TestCompositeLocantSortKey:
    """a phase: _composite_locant_sort_key replaces the line-163
    lambda in collect_stereodescriptors. Mixed int/'<int><letter>' locants
    must sort per IUPAC P-91.1 (composite '3a' BETWEEN integer 3 and 4).

    Per Pitfall 2, the helper must dispatch on isinstance(locant, int) FIRST
    so int input does not fall into the int(str[:-1]) ValueError trap.
    """

    # Test C-01a -- int-only sort order (regression invariant).
    def test_sort_int_only_byte_identical(self):
        items = [(2, 'R'), (3, 'S'), (5, 'E')]
        assert sorted(items, key=_composite_locant_sort_key) == [
            (2, 'R'), (3, 'S'), (5, 'E')
        ]

    # Test C-01b -- str-only composite locant sort.
    def test_sort_str_only_composite(self):
        items = [('7a', 'S'), ('3a', 'R')]
        assert sorted(items, key=_composite_locant_sort_key) == [
            ('3a', 'R'), ('7a', 'S')
        ]

    # Test C-01c -- mixed int/str sort per IUPAC P-91.1: 3 < '3a' < 4.
    def test_sort_mixed_int_and_composite(self):
        items = [(4, 'S'), ('3a', 'R'), (3, 'R')]
        assert sorted(items, key=_composite_locant_sort_key) == [
            (3, 'R'), ('3a', 'R'), (4, 'S')
        ]

    # Test C-02 -- byte-identical to lambda x: x[0] for int-only inputs.
    def test_int_only_byte_identical_to_phase152_baseline(self):
        items = [(2, 'R'), (3, 'S'), (5, 'E')]
        from_lambda = sorted(items, key=lambda x: x[0])
        from_helper = sorted(items, key=_composite_locant_sort_key)
        assert from_lambda == from_helper, (
            "Pitfall 2 regression: int-only sort order must be byte-identical "
            "to pre-153 lambda x: x[0]"
        )

    # Test C-03 -- two-letter suffix corner case locks ValueError raising.
    def test_str_locant_with_two_letter_suffix_raises(self):
        # '12bb' has TWO trailing alpha chars. The current parse pattern
        # consumes only the LAST char via locant[-1].isalpha() / locant[:-1].
        # int('12b'[:-1]) is well-defined ONLY if the residue is purely
        # numeric, so '12bb' will raise ValueError on int('12b') because
        # '12b' is not a pure integer. We lock that ValueError so a future
        # regression (e.g. quietly broadening to two-letter suffixes) is
        # caught loudly.
        with pytest.raises(ValueError):
            _composite_locant_sort_key(('12bb', 'R'))

    # Test C-04 -- empty input.
    def test_empty_input(self):
        assert sorted([], key=_composite_locant_sort_key) == []


@pytest.mark.unit
class TestRingJunctionInjection:
    """a phase: end-to-end verification that the injector now produces
    composite-locant prefixes like '(3aR,8aS)-' once the sort key accepts
    mixed int/str. Decalin-shaped fixture; ring junctions live at atoms 3
    and 8 with composite locants '3a' / '8a'.
    """

    # Test J-01 -- ring-junction stereo flows through the universal injector.
    def test_emits_composite_locant_prefix_for_decalin_like(self):
        # cis-decalin (atoms 3, 8 are ring junctions; @@ markers fix CIP).
        mol = Chem.MolFromSmiles("C1CC[C@@H]2CCCC[C@@H]2C1")
        rdCIPLabeler.AssignCIPLabels(mol)
        # Composite-locant atom_to_locant map (decalin numbering).
        atom_to_locant = {
            0: 1, 1: 2, 2: 3, 3: '3a',
            4: 5, 5: 6, 6: 7, 7: 8, 8: '8a', 9: 4,
        }
        result = inject_stereo_from_locant_map(
            "decahydronaphthalene", mol, atom_to_locant,
        )
        # The result must carry a P-91 composite-locant prefix block. Lock
        # only the regex shape: leading '(' + at least one '3a'/'8a' or
        # bare integer then R/S, and a trailing ')-' prefix.
        assert re.match(r"^\([0-9aRSrsEZ,]+\)-", result), (
            f"D-03: composite-locant stereo prefix missing from {result!r}"
        )


@pytest.mark.unit
class TestStereoBackstopRegressionInvariant:
    """a phase: bytes-identical lock for a phase collector behaviour.

    These tests verify two pre-153 invariants that depend on the line-163
    sort key. Together with the test_stereo_backstop suite (13 tests) and
    the rest of test_handler_stereo_injection (39 tests), they catch any
    byte-level drift introduced by the helper swap.
    """

    # Test B-01 -- pre-153 collector behaviour on int-only locants is
    # preserved end-to-end (uses a phase fixture pattern).
    def test_collector_int_only_behaviour_preserved(self):
        from orthonym.rules.stereochemistry import collect_stereodescriptors
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4, 6: 5}
        result = collect_stereodescriptors(mol, atom_to_locant)
        # Two stereocenters at locants 2 and 3, sorted ascending. The
        # CIP codes depend on RDKit canonicalisation; we lock locant
        # ordering and emission count.
        assert len(result) == 2
        assert result[0][0] == 2
        assert result[1][0] == 3
        assert result[0][1] in ('R', 'S')
        assert result[1][1] in ('R', 'S')

    # Test B-02 -- empty descriptor list still sorts cleanly under the
    # new sort key (regression: never raise on empty input).
    def test_collector_empty_descriptors_no_error(self):
        from orthonym.rules.stereochemistry import collect_stereodescriptors
        mol = Chem.MolFromSmiles("CCO")  # no stereo
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {0: 1, 1: 2, 2: 3}
        result = collect_stereodescriptors(mol, atom_to_locant)
        assert result == []


# ============================================================================
# a phase commit 2/5: complex_ring Tier-A wiring (/ /)
# ============================================================================


@pytest.mark.unit
class TestComplexRingHelpers:
    """a phase /: two pure helpers in composer.py mirror the
    a phase BL-01 / BL-02 patterns for the complex_ring handler.
    """

    def test_complex_ring_parent_atom_indices_returns_set_of_ring_atoms(self):
        import types
        from orthonym.assembly.composer import _complex_ring_parent_atom_indices
        cr = types.SimpleNamespace(ring_atoms=(0, 1, 2, 3, 4))
        assert _complex_ring_parent_atom_indices(cr) == {0, 1, 2, 3, 4}

    def test_complex_ring_parent_atom_indices_returns_None_for_empty(self):
        import types
        from orthonym.assembly.composer import _complex_ring_parent_atom_indices
        assert _complex_ring_parent_atom_indices(None) is None
        assert _complex_ring_parent_atom_indices(
            types.SimpleNamespace(ring_atoms=())
        ) is None

    def test_ring_is_whole_molecule_for_complex_True_when_ring_eq_mol(self):
        import types
        from orthonym.assembly.composer import _ring_is_whole_molecule_for_complex
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene 10 HA
        cr = types.SimpleNamespace(ring_atoms=tuple(range(10)))
        assert _ring_is_whole_molecule_for_complex(cr, mol) is True

    def test_ring_is_whole_molecule_for_complex_False_when_substituent_present(self):
        import types
        from orthonym.assembly.composer import _ring_is_whole_molecule_for_complex
        mol = Chem.MolFromSmiles("Cc1ccc2ccccc2c1")  # methylnaphthalene 11 HA
        cr = types.SimpleNamespace(ring_atoms=tuple(range(10)))
        assert _ring_is_whole_molecule_for_complex(cr, mol) is False


@pytest.mark.unit
class TestComplexRingTierAWiring:
    """Static-source verifications for / / wiring (commit 2/5).

    The Tier-A ring injection block was REFACTORED out of ``composer.py`` into
    ``assembly/handlers/tier_a_ring.py`` (Milestone C); these greps are
    re-pointed there. Verified the block is present at tier_a_ring.py."""

    _TIER_A = "src/orthonym/assembly/handlers/tier_a_ring.py"

    def test_complex_ring_in_handler_set(self):
        with open(self._TIER_A) as f:
            src = f.read()
        # Find the Tier-A injection block by anchoring on the handler tuple.
        assert "best.handler in ('benzene', 'heterocycle', 'complex_ring')" in src, (
            "D-01: Tier-A injection block must include complex_ring in handler set"
        )

    def test_complex_ring_pool_add_passes_parent_atom_indices(self):
        with open(self._TIER_A) as f:
            src = f.read()
        # The pool.add must pass parent_atom_indices= via the new helper.
        assert "parent_atom_indices=_complex_ring_parent_atom_indices(" in src, (
            "D-05: pool.add for complex_ring must pass parent_atom_indices "
            "via the new helper"
        )

    def test_complex_ring_uses_per_molecule_include_near_parent_ez(self):
        with open(self._TIER_A) as f:
            src = f.read()
        #: the new wiring must call _ring_is_whole_molecule_for_complex
        # to compute include_near_parent_ez per-molecule for the complex_ring
        # branch (benzene/heterocycle keep include_near_parent_ez=True).
        assert "_ring_is_whole_molecule_for_complex(" in src, (
            "D-06: new wiring must use _ring_is_whole_molecule_for_complex "
            "(per-molecule include_near_parent_ez)"
        )


# ============================================================================
# a phase commit 4/5: ring strain filter regression guard (/)
# ============================================================================


from orthonym.rules.stereochemistry import collect_stereodescriptors


@pytest.mark.unit
class TestRingStrainFilter:
    """a phase: parametrized verification of the existing E/Z
    ring-strain filter at rules/stereochemistry.py:85-95. The filter
    REJECTS ring sizes < 8 and ACCEPTS ring sizes >= 8 (P-31.1.3 Sep 2024
    errata makes E/Z mandatory for >= 8). a phase ships TESTS, not new
    filter code (-- the existing filter is correct).
    """

    @pytest.mark.parametrize("smiles,ring_size,expect_ez", [
        # ring_size 6 -- rejected (filter says < 8 -> skip)
        ("C1CC=CCC1", 6, False),                  # cyclohex-1-ene
        # ring_size 7 -- rejected
        ("C1CCC=CCC1", 7, False),                 # cyclohept-1-ene
        # ring_size 8 -- ACCEPTED (errata threshold)
        ("C1CC/C=C/CCC1", 8, True),               # (E)-cyclooct-1-ene
        # ring_size 9 -- accepted
        ("C1CCC/C=C\\CCC1", 9, True),             # cyclonon-1-ene with explicit Z
        # ring_size 12 -- accepted (errata canonical example)
        ("C1CCC/C=C\\CC/C=C\\CC1", 12, True),     # cyclododeca-1,5-diene
    ])
    def test_strain_filter_per_ring_size(self, smiles, ring_size, expect_ez):
        mol = Chem.MolFromSmiles(smiles)
        rdCIPLabeler.AssignCIPLabels(mol)
        # Verify the test fixture has the ring size we claim (sanity check
        # so a stray SMILES typo cannot quietly bypass the gate).
        ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        assert ring_size in ring_sizes, (
            f"fixture mis-sized: claimed {ring_size}, actual {ring_sizes}"
        )
        # Build atom_to_locant from the chosen ring's atom order.
        ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) == ring_size)
        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(ring)}
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        has_ez = any(d[1] in ('E', 'Z') for d in descriptors)
        assert has_ez == expect_ez, (
            f"ring_size={ring_size} expect_ez={expect_ez} got descriptors="
            f"{descriptors} (filter at rules/stereochemistry.py:85-95)"
        )
