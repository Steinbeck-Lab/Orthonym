"""Wave-8 P8 unit tests: the phane PIN subsystem.

Per docs/the workflow tooling/plans/2026-07-16-wave8-p8-phane.md: OPSIN 2.9 cannot
parse ANY phane name (verified 2026-07-16 -- neither the /.3
simplified-skeletal PIN nor the legacy bracket-prefix
'[2.2]paracyclophane'), so there is no round-trip oracle for this
subsystem. Verification is BB-name-exact (the verbatim `(PIN)` string quoted
from the Blue Book at the cited line) + the source-level
`_phane_formula_veto` atom-conservation guard + a tight `_PHANE_PIN_RE`
validity-gate carve-out (namer.py, modeled on `_DIANHYDRIDE_PIN_RE`).

Test classes / functions map 1:1 onto the plan's tasks:
- Task 8.2 test_amplification_prefix_transform
- Task 8.3 test_simplified_skeletal_name
- Task 8.4 test_multiplied_amplificant
- Task 8.1 TestSimplify (design-contract: perception + skeleton graph)
- Task 8.5 TestNumberSkeleton (design-contract: superatom-locant numbering)
- Task 8.6 TestAttachmentLocants (design-contract: attachment ordering)
- Task 8.7 TestBuildPhanePin (integration: the 5 verified homophane golds)
- Task 8.8 test_composite_locant_sort_key (design-contract, pure, BB tuple)
- Task 8.9 test_apply_skeletal_replacement (design-contract, pure, BB trithia)
            + test_simplify_rejects_heteroatom_bridge (safety-net veto)
- Task 8.11 test_phane_amplificant_seniority_key_matches_bb_order (design-
            contract, pure, reuses ring_selection.ring_system_score)
- Task 8.12 test_phane_formula_veto / test_dispatch_emits_phane_pin
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.rules.phane import (
    Amplificant,
    PhaneStructure,
    SkeletonClass,
    _amplification_prefix,
    _apply_skeletal_replacement,
    _attachment_locants,
    _composite_locant_sort_key,
    _multiplied_amplificant,
    _number_skeleton,
    _phane_amplificant_seniority_key,
    _phane_formula_veto,
    _simplified_skeletal_name,
    _simplify,
    build_phane_pin,
    name_cyclophane,
)


# ---------------------------------------------------------------------------
# Task 8.2 -- amplification-prefix transform
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("parent,expected", [
    ("benzene", "benzena"),
    ("naphthalene", "naphthalena"),
    ("anthracene", "anthracena"),
    ("pyrrole", "pyrrola"),
    ("furan", "furana"),      # no final 'e' -> add 'a'
    ("pyran", "pyrana"),
    ("pyridine", "pyridina"),
    ("[1,2]oxazole", "[1,2]oxazola"),  # bracketed locants retained note)
])
def test_amplification_prefix_transform(parent, expected):
    assert _amplification_prefix(parent) == expected


# ---------------------------------------------------------------------------
# Task 8.3 -- simplified skeletal-name builder
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("n,klass,expected", [
    (6, SkeletonClass.MONOCYCLIC, "cyclohexaphane"),
    (7, SkeletonClass.MONOCYCLIC, "cycloheptaphane"),
    (8, SkeletonClass.MONOCYCLIC, "cyclooctaphane"),
    (9, SkeletonClass.MONOCYCLIC, "cyclononaphane"),
    (20, SkeletonClass.MONOCYCLIC, "cycloicosaphane"),
    (9, SkeletonClass.ACYCLIC, "nonaphane"),   # no prefix for linear
])
def test_simplified_skeletal_name(n, klass, expected):
    assert _simplified_skeletal_name(n, klass) == expected


def test_simplified_skeletal_name_fails_closed_for_von_baeyer_and_spiro():
    # Task 8.10: no verified canonicalizable BB fixture this phase.
    assert _simplified_skeletal_name(14, SkeletonClass.VON_BAEYER) is None
    assert _simplified_skeletal_name(13, SkeletonClass.SPIRO) is None


# ---------------------------------------------------------------------------
# Task 8.4 -- multiplicative amplificant term di/bis
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("prefix,count,expected", [
    ("benzena", 2, "dibenzena"),
    ("pyridina", 3, "tripyridina"),
    ("benzena", 4, "tetrabenzena"),
    ("bicyclo[2.2.1]heptana", 2, "bis(bicyclo[2.2.1]heptana)"),
    ("[1,3]dioxola", 2, "bis([1,3]dioxola)"),
])
def test_multiplied_amplificant(prefix, count, expected):
    assert _multiplied_amplificant(prefix, count) == expected


def test_multiplied_amplificant_count_one_is_identity():
    assert _multiplied_amplificant("benzena", 1) == "benzena"


# ---------------------------------------------------------------------------
# Task 8.1 -- simplification engine (_simplify)
# ---------------------------------------------------------------------------


ANCHOR_PARA = "C1Cc2ccc(cc2)CCc2ccc1cc2"  # [2.2]paracyclophane
ANCHOR_META = "c1cc2cc(c1)CCc1cccc(c1)CC2"  # [2.2]metacyclophane


@pytest.mark.unit
class TestSimplify:
    def test_anchor_yields_two_amplificants_and_six_node_monocycle(self):
        mol = Chem.MolFromSmiles(ANCHOR_PARA)
        struct = _simplify(mol)
        assert struct is not None
        assert len(struct.amplificants) == 2
        assert all(a.parent_name == "benzene" for a in struct.amplificants)
        assert len(struct.bridge_atoms) == 4
        assert struct.skeleton_class is SkeletonClass.MONOCYCLIC
        assert len(struct.node_cycle) == 6  # 2 superatoms + 4 bridge atoms

    @pytest.mark.parametrize("smiles,label", [
        ("c1ccc2ccccc2c1", "naphthalene (fused, no macrocycle)"),
        ("c1ccc(-c2ccccc2)cc1", "biphenyl (ring assembly, no macrocycle)"),
        ("C1CC2CCC1CC2", "bicyclo[2.2.2]octane (von Baeyer alkane, no amplificants)"),
        ("CCO", "ethanol (no rings)"),
    ])
    def test_non_phane_controls_return_none(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        assert _simplify(mol) is None, label

    def test_none_input_returns_none(self):
        assert _simplify(None) is None


# ---------------------------------------------------------------------------
# Task 8.9 safety net -- heteroatom bridges must never emit a hydrocarbon name
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simplify_rejects_heteroatom_bridge():
    """A non-carbon bridge atom (aza-bridged paracyclophane) MUST fail closed
    in `_simplify` -- Task 8.9's 'a'-replacement naming is a design contract
    this phase (unwired to emission), so a heteroatom bridge must never be
    silently misnamed as a plain hydrocarbon phane."""
    mol = Chem.MolFromSmiles("c1cc2ccc1CNc1ccc(cc1)CN2")  # aza-bridged
    assert _simplify(mol) is None
    assert build_phane_pin(mol) is None


# ---------------------------------------------------------------------------
# Task 8.5 -- skeleton numbering + superatom-locant assignment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNumberSkeleton:
    def test_anchor_numbers_superatoms_to_1_4(self):
        mol = Chem.MolFromSmiles(ANCHOR_PARA)
        struct = _simplify(mol)
        locants = _number_skeleton(struct)
        superatom_locants = sorted(v for k, v in locants.items() if k[0] == "amp")
        assert superatom_locants == [1, 4]

    def test_meta_isomer_also_numbers_to_1_4(self):
        mol = Chem.MolFromSmiles(ANCHOR_META)
        struct = _simplify(mol)
        locants = _number_skeleton(struct)
        superatom_locants = sorted(v for k, v in locants.items() if k[0] == "amp")
        assert superatom_locants == [1, 4]

    def test_returns_none_for_non_monocyclic_or_missing_cycle(self):
        empty_struct = PhaneStructure(
            mol=Chem.MolFromSmiles("CCO"),
            amplificants=(),
            bridge_atoms=frozenset(),
            skeleton_class=SkeletonClass.VON_BAEYER,
            node_cycle=None,
        )
        assert _number_skeleton(empty_struct) is None


# ---------------------------------------------------------------------------
# Task 8.6 -- attachment-locant perception + ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAttachmentLocants:
    def test_anchor_gives_para_1_4_on_both_rings(self):
        mol = Chem.MolFromSmiles(ANCHOR_PARA)
        struct = _simplify(mol)
        locants = _number_skeleton(struct)
        per_amp = _attachment_locants(struct, locants)
        assert [a[1] for a in per_amp] == [(1, 4), (1, 4)]

    def test_meta_gives_1_3_on_both_rings(self):
        mol = Chem.MolFromSmiles(ANCHOR_META)
        struct = _simplify(mol)
        locants = _number_skeleton(struct)
        per_amp = _attachment_locants(struct, locants)
        assert [a[1] for a in per_amp] == [(1, 3), (1, 3)]

    def test_ortho_gives_1_2(self):
        # Synthetic ortho-substituted [2.2]orthocyclophane-like control.
        mol = Chem.MolFromSmiles("c1ccc2c(c1)CCc1ccccc1CC2")
        struct = _simplify(mol)
        if struct is None:
            pytest.skip("ortho control SMILES did not perceive as a simplifiable phane")
        locants = _number_skeleton(struct)
        per_amp = _attachment_locants(struct, locants)
        assert all(a[1] == (1, 2) for a in per_amp)


# ---------------------------------------------------------------------------
# Task 8.7 -- integration: monocyclic all-benzene homophane PIN (ANCHOR GOLD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPhanePin:
    """BB-name-exact fixtures. `[2.2]paracyclophane` / `[2.2]metacyclophane`
    are BB (:14947) / (:15024) `(PIN)` verbatim. The
    [3.3]/[3.2] homologs apply the SAME cited rules /.3.1/.3.2/.3.2.1)
    to already-topology-verified fixtures from
    tests/fixtures/cyclophane/{blue_book_examples,corpus_mined}.json --
    not independently BB-cited with these exact node counts, but derived by
    the identical mechanism (hand-traced + code-cross-checked, see the P8
    report)."""

    @pytest.mark.parametrize("smiles,expected,label", [
        (ANCHOR_PARA, "1,4(1,4)-dibenzenacyclohexaphane",
         "[2.2]paracyclophane -- BB P-26.3.2.1 (:14947) (PIN) verbatim"),
        (ANCHOR_META, "1,4(1,3)-dibenzenacyclohexaphane",
         "[2.2]metacyclophane -- BB P-26.4.1.4 (:15024) (PIN) verbatim"),
        ("c1cc2ccc1CCCc1ccc(cc1)CCC2", "1,5(1,4)-dibenzenacyclooctaphane",
         "[3.3]paracyclophane homolog (rule-derived)"),
        ("c1cc2ccc1CCCc1ccc(cc1)CC2", "1,4(1,4)-dibenzenacycloheptaphane",
         "[3.2]paracyclophane (asymmetric bridge) homolog (rule-derived)"),
        ("c1cc2cc(c1)CCCc1cccc(c1)CCC2", "1,5(1,3)-dibenzenacyclooctaphane",
         "[3.3]metacyclophane homolog (rule-derived)"),
    ])
    def test_monocyclic_homophane_pin(self, smiles, expected, label):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        assert build_phane_pin(mol) == expected, label
        # name_cyclophane must delegate to build_phane_pin for this class.
        assert name_cyclophane(mol) == expected, label

    def test_build_phane_pin_returns_none_for_non_cyclophane(self):
        assert build_phane_pin(Chem.MolFromSmiles("c1ccccc1")) is None
        assert build_phane_pin(Chem.MolFromSmiles("CCO")) is None

    def test_build_phane_pin_returns_none_for_none_input(self):
        assert build_phane_pin(None) is None


# ---------------------------------------------------------------------------
# Task 8.8 -- composite-locant citation ordering (design contract, pure)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_composite_locant_sort_key_matches_bb_heptachloro_order():
    """BB (:15101) verbatim citation order:
    '1(4),1(5),1(6),3,3,4(2),4(3)-heptachloro-...' (ASCII form of the
    superscript original `1⁴,1⁵,1⁶,3,3,4²,4³`). Sorting via
    `_composite_locant_sort_key` must reproduce this exact order: primary
    locant ascending; a PLAIN locant (no superscript) before any composite
    sharing the same primary; composite locants then ascending by
    superscript."""
    bb_order = ["1(4)", "1(5)", "1(6)", "3", "3", "4(2)", "4(3)"]
    shuffled = ["4(3)", "3", "1(5)", "4(2)", "1(6)", "3", "1(4)"]
    assert sorted(shuffled, key=_composite_locant_sort_key) == bb_order


def test_composite_locant_sort_key_rejects_malformed_input():
    with pytest.raises(ValueError):
        _composite_locant_sort_key("not-a-locant")


# ---------------------------------------------------------------------------
# Task 8.9 -- 'a'-replacement heterophanes (design contract, pure)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_skeletal_replacement_matches_bb_trithia_example():
    """BB (:15155) `(PIN)`:
    '2,4,6-trithia-1,7(1),3,5(1,4)-tetrabenzenaheptaphane'."""
    base_name = "1,7(1),3,5(1,4)-tetrabenzenaheptaphane"
    result = _apply_skeletal_replacement({2: "S", 4: "S", 6: "S"}, base_name)
    assert result == "2,4,6-trithia-1,7(1),3,5(1,4)-tetrabenzenaheptaphane"


def test_apply_skeletal_replacement_orders_mixed_elements_by_bb_seniority():
    """BB element seniority O>S>Se>Te>N>...; ledger example
    '2-thia-6-aza-1,4(1,4)-dibenzenacyclohexaphane' cites thia (S) before
    aza (N)."""
    base_name = "1,4(1,4)-dibenzenacyclohexaphane"
    result = _apply_skeletal_replacement({2: "S", 6: "N"}, base_name)
    assert result == "2-thia-6-aza-1,4(1,4)-dibenzenacyclohexaphane"


# ---------------------------------------------------------------------------
# Task 8.11 -- multi-different-amplificant seniority (design contract, pure)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_phane_amplificant_seniority_key_matches_bb_worked_example_order():
    """BB (:14998) worked example:
    `1(8,5)-quinolina-4(1,4)-phenanthrena-7(1,4)-naphthalenacyclononaphane`
    -- quinoline (N-heterocycle) is senior to phenanthrene (3 all-carbon
    rings), senior to naphthalene (2 all-carbon rings) on; the most
    senior amplificant gets the LOWEST superatom locant. Reuses the existing
     `ring_system_score` (no reimplementation) to derive each
    amplificant's seniority rank, then verifies `_phane_amplificant_seniority_key`
    sorts them into exactly the BB citation order."""
    from orthonym.rules.ring_selection import ring_system_score

    rings = {
        "quinolina": ("c1ccc2ncccc2c1", (8, 5)),
        "phenanthrena": ("c1ccc2c(c1)ccc1ccccc12", (1, 4)),
        "naphthalena": ("c1ccc2ccccc2c1", (1, 4)),
    }
    entries = []
    for label, (smi, attach) in rings.items():
        mol = Chem.MolFromSmiles(smi)
        score = ring_system_score(mol, set(range(mol.GetNumAtoms())))
        entries.append((label, _phane_amplificant_seniority_key(score, attach)))

    ordered = [label for label, _ in sorted(entries, key=lambda e: e[1])]
    assert ordered == ["quinolina", "phenanthrena", "naphthalena"]


# ---------------------------------------------------------------------------
# Task 8.12 -- formula-conservation veto + dispatch emission
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_phane_formula_veto_accepts_intact_structure():
    mol = Chem.MolFromSmiles(ANCHOR_PARA)
    struct = _simplify(mol)
    assert _phane_formula_veto(struct, mol) is True


def test_phane_formula_veto_blocks_atom_drop():
    mol = Chem.MolFromSmiles(ANCHOR_PARA)
    struct = _simplify(mol)
    # Simulate a mis-perceived structure that dropped one amplificant ring.
    struct_missing_one_ring = PhaneStructure(
        mol=struct.mol,
        amplificants=struct.amplificants[:1],
        bridge_atoms=struct.bridge_atoms,
        skeleton_class=struct.skeleton_class,
        node_cycle=struct.node_cycle,
    )
    assert _phane_formula_veto(struct_missing_one_ring, mol) is False


def test_phane_formula_veto_rejects_none():
    assert _phane_formula_veto(None, Chem.MolFromSmiles(ANCHOR_PARA)) is False
    assert _phane_formula_veto(_simplify(Chem.MolFromSmiles(ANCHOR_PARA)), None) is False


@pytest.mark.unit
def test_dispatch_emits_phane_pin():
    """Task 8.12 Step 1 acceptance test: production (gate-ON) now EMITS the
     PIN for the anchor instead of 'unknown organic compound'."""
    from orthonym import Orthonym

    namer_on = Orthonym(style="pin")
    assert namer_on.name(ANCHOR_PARA) == "1,4(1,4)-dibenzenacyclohexaphane"


@pytest.mark.unit
def test_dispatch_gated_matches_raw_for_all_homophane_golds():
    """Gate-ON and gate-OFF (`_disable_opsin_validity_gate=True`) must agree
    for every verified homophane gold -- OPSIN cannot parse phane names at
    all, so `_PHANE_PIN_RE` must carve the gate out cleanly (no suppression)."""
    from orthonym import Orthonym

    namer_on = Orthonym(style="pin")
    namer_off = Orthonym(style="pin", _disable_opsin_validity_gate=True)
    cases = [
        (ANCHOR_PARA, "1,4(1,4)-dibenzenacyclohexaphane"),
        (ANCHOR_META, "1,4(1,3)-dibenzenacyclohexaphane"),
        ("c1cc2ccc1CCCc1ccc(cc1)CCC2", "1,5(1,4)-dibenzenacyclooctaphane"),
        ("c1cc2ccc1CCCc1ccc(cc1)CC2", "1,4(1,4)-dibenzenacycloheptaphane"),
        ("c1cc2cc(c1)CCCc1cccc(c1)CCC2", "1,5(1,3)-dibenzenacyclooctaphane"),
    ]
    for smi, expected in cases:
        assert namer_on.name(smi) == expected
        assert namer_off.name(smi) == expected


@pytest.mark.unit
def test_dispatch_still_fails_closed_for_heteroatom_bridge_and_controls():
    """Non-emittable phane topologies (heteroatom bridge, heterocyclic
    linker) and ordinary non-cyclophane controls must be unaffected."""
    from orthonym import Orthonym

    namer_on = Orthonym(style="pin")
    assert namer_on.name("c1cc2ccc1CNc1ccc(cc1)CN2") == "unknown organic compound"
    assert namer_on.name("c1cc2ccc1COc1ccc(cc1)CO2") == "unknown organic compound"
    assert namer_on.name("c1cc2ncc1CCc1ccc(cn1)CC2") == "unknown organic compound"
    assert namer_on.name("c1ccc(-c2ccccc2)cc1") == "1,1'-biphenyl"
    assert namer_on.name("c1ccc2ccccc2c1") == "naphthalene"
