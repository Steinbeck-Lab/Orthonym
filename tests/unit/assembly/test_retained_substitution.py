"""a phase Plan-02: triviality-controller unit tests.

Mirrors the a phase class-per-bug discipline (analog:
tests/unit/assembly/test_diaryl_methyl_substituent.py). Tests construct ``NameTreeNode``
directly and exercise the per-Type dispatch + the frozen-dataclass invariants WITHOUT the
perception layer / OPSIN / candidate_pool wiring — the per-Type checks and ``_build_rewrite``
are pure functions of (node, seed_entry), so they are deterministic without recovery.

Pins the reviews-iter-1 fixes: #2 (conditional fragment_legacy reset via
``_replace_preserving_or_resetting_legacy``) and #3 (the hint-based path removed; zero
``atom_to_locant_hint`` references; 3-path recovery).

Source: 168-internal notes /////; internal notes #2 + #3.
"""

import dataclasses
import inspect

import pytest
from rdkit import Chem

from orthonym.assembly.name_tree import NameTreeNode, is_coarse_node
from orthonym.assembly import retained_substitution as rs
from orthonym.assembly.retained_substitution import (
    apply_triviality_controller,
    _build_rewrite,
    _recompute_multiplicative_prefix,
    _replace_preserving_or_resetting_legacy,
    _type_1_check,
    _type_2a_check,
    _type_2c_check,
    _type_3_check,
)
from orthonym.data.triviality_controller_seed import SEED_TABLE


def _entry(smiles):
    return SEED_TABLE[Chem.CanonSmiles(smiles)]


class TestStageAInvariants:
    """enabled=False is a no-op; coarse nodes pass through unchanged (+)."""

    @pytest.mark.unit
    def test_enabled_false_returns_input_unchanged(self):
        n = NameTreeNode(parent_stem="benzene")
        out = apply_triviality_controller(n, None, None, enabled=False)
        assert out is n

    @pytest.mark.unit
    def test_coarse_node_returns_unchanged(self):
        n = NameTreeNode(parent_stem="benzenamine", fragment_legacy="benzenamine")
        assert is_coarse_node(n)
        out = apply_triviality_controller(n, None, None, enabled=True)
        assert out is n


class TestType1Branch:
    """Type 1: unconditional swap on a canonical-SMILES match."""

    @pytest.mark.unit
    def test_type_1_check_always_true(self):
        assert _type_1_check(NameTreeNode(parent_stem="x"), _entry("c1ccoc1")) is True

    @pytest.mark.unit
    def test_build_rewrite_to_furan(self):
        entry = _entry("c1ccoc1")
        node = NameTreeNode(parent_stem="1-oxacyclopenta-2,4-diene", fragment_legacy="legacy")
        out = _build_rewrite(node, entry, ())
        assert out.parent_stem == "furan"
        assert out.fragment_legacy is None
        assert out.iupac_section_cite == entry.iupac_p_section


def _real_principal_group(smiles):
    """The actual ``features.principal_group`` value the controller compares against — driven
    through the perception layer so a test cannot encode a stale literal ."""
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles(smiles)
    pg_name, _ = get_principal_group(mol, detect_functional_groups(mol))
    return pg_name


class TestType2aBranch:
    """Type 2a: principal-group-bound swap; refuses on PG mismatch (Pitfall 2).

     + (code review 2026-05-30): these drive ``_type_2a_check`` with the REAL
    ``get_principal_group`` output (NOT a hand-written literal). The pre-fix seed required
    "primary_alcohol"/"primary_amine", but perception returns "phenol"/"aromatic_amine", so the
    phenol/aniline swaps could NEVER fire — and the old ``test_phenol_swap_with_matching_pg``
    passed only because it fed the same wrong literal the seed held. Asserting the perceived PG
    value here fails the suite if the seed vocabulary and perception ever drift apart again."""

    @pytest.mark.unit
    def test_phenol_swap_with_real_pg(self):
        entry = _entry("Oc1ccccc1")
        real_pg = _real_principal_group("Oc1ccccc1")
        assert real_pg == "phenol", f"perception drift: get_principal_group(phenol)={real_pg!r}"
        assert _type_2a_check(NameTreeNode(parent_stem="benzenol"), real_pg, entry) is True

    @pytest.mark.unit
    def test_aniline_swap_with_real_pg(self):
        entry = _entry("Nc1ccccc1")
        real_pg = _real_principal_group("Nc1ccccc1")
        assert real_pg == "aromatic_amine", f"perception drift: get_principal_group(aniline)={real_pg!r}"
        assert _type_2a_check(NameTreeNode(parent_stem="benzenamine"), real_pg, entry) is True

    @pytest.mark.unit
    def test_phenol_refuses_swap_when_pg_mismatch(self):
        entry = _entry("Oc1ccccc1")
        assert _type_2a_check(NameTreeNode(parent_stem="benzenol"), "carboxylic_acid", entry) is False

    @pytest.mark.unit
    def test_phenol_refuses_swap_when_pg_none(self):
        entry = _entry("Oc1ccccc1")
        assert _type_2a_check(NameTreeNode(parent_stem="benzenol"), None, entry) is False


class TestType2bBranch:
    """Type 2b: closed-list; bare reduces to unconditional."""

    @pytest.mark.unit
    def test_formic_acid_present(self):
        assert Chem.CanonSmiles("OC=O") in SEED_TABLE
        assert _entry("OC=O").substitution_type.value == "type_2b"

    @pytest.mark.unit
    def test_formic_acid_bare_is_allowed(self):
        # No prefixes => Type 2b reduces to Type 1 unconditional.
        from orthonym.assembly.retained_substitution import _type_2b_check
        entry = _entry("OC=O")
        assert _type_2b_check(NameTreeNode(parent_stem="methanoic acid"),
                              Chem.MolFromSmiles("OC=O"), entry, None) is True


class TestType2cBranch:
    """Type 2c: default-to-Type-3 when no locus override (hydroxylamine).

    F- (DD6): anisole was removed from the seed table — it is no longer a PIN
    (the PIN is methoxybenzene,, so the triviality controller must NOT swap
    methoxybenzene -> anisole. The Type-2c bare/substituted dispatch is now exercised via
    hydroxylamine, the remaining Type-2c entry; ``test_anisole_removed_from_seed`` locks the
    F- policy.
    """

    @pytest.mark.unit
    def test_hydroxylamine_bare_swap_allowed(self):
        entry = _entry("NO")
        assert entry.locus_override_rule_id is None
        assert _type_2c_check(NameTreeNode(parent_stem="hydroxylamine"),
                              Chem.MolFromSmiles("NO"), entry) is True

    @pytest.mark.unit
    def test_hydroxylamine_refuses_swap_with_substituent(self):
        entry = _entry("NO")
        child = NameTreeNode(parent_stem="methyl")
        node = NameTreeNode(parent_stem="hydroxylamine", prefixes=(child,))
        assert _type_2c_check(node, Chem.MolFromSmiles("CNO"), entry) is False

    @pytest.mark.unit
    def test_anisole_removed_from_seed(self):
        # F- / DD6: 'anisole' is general-only (PIN methoxybenzene), denied in
        # iupac_2013_pin_list.json. It must NOT be a triviality-controller swap target
        # (the deny gate in load_seed_table would otherwise zero the whole table).
        assert Chem.CanonSmiles("COc1ccccc1") not in SEED_TABLE


class TestType3Branch:
    """Type 3: bare-only + locant_context (toluene/xylene; Pitfall 3)."""

    @pytest.mark.unit
    def test_toluene_bare_swap_allowed(self):
        assert _type_3_check(NameTreeNode(parent_stem="methylbenzene"), _entry("Cc1ccccc1")) is True

    @pytest.mark.unit
    def test_toluene_refuses_swap_with_chlorine(self):
        child = NameTreeNode(parent_stem="chloro")
        node = NameTreeNode(parent_stem="methylbenzene", prefixes=(child,))
        assert _type_3_check(node, _entry("Cc1ccccc1")) is False

    @pytest.mark.unit
    def test_xylene_1_2_locant_context_match(self):
        entry = _entry("Cc1ccccc1C")
        assert entry.locant_context == (1, 2)
        # Bare node with matching locants passes; mismatched locants fail.
        assert _type_3_check(NameTreeNode(parent_stem="o-xylene-stem", locants=(1, 2)), entry) is True
        assert _type_3_check(NameTreeNode(parent_stem="o-xylene-stem", locants=(1, 3)), entry) is False


class TestFrozenDataclassInvariants:
    """Rewrites never mutate the input; parent-changing rewrite resets fragment_legacy;
     multiplier is re-derived via the SSOT predicates (NOT statically None)."""

    @pytest.mark.unit
    def test_rewrite_does_not_mutate_input(self):
        entry = _entry("Oc1ccccc1")
        node = NameTreeNode(parent_stem="benzenol", fragment_legacy="legacy")
        _ = _build_rewrite(node, entry, ())
        assert node.parent_stem == "benzenol"  # original unchanged
        assert node.fragment_legacy == "legacy"

    @pytest.mark.unit
    def test_build_rewrite_resets_fragment_legacy(self):
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", fragment_legacy="x"), entry, ())
        assert out.fragment_legacy is None  # #2 / Pitfall 1

    @pytest.mark.unit
    def test_multiplier_none_stays_none(self):
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", multiplicative_prefix=None), entry, ())
        assert out.multiplicative_prefix is None

    @pytest.mark.unit
    def test_multiplier_recompute_simple_keeps_di(self):
        # phenol is a SIMPLE substituent name (no digits/hyphens) => di
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", multiplicative_prefix="di"), entry, ())
        assert out.multiplicative_prefix == "di"

    @pytest.mark.unit
    def test_multiplier_recompute_unsubstituted_stays_di(self):
        """: the recompute consults ``is_substituted_substituent``, so a name that
        is *complex* but **not substituted** keeps ``di``.

        ⚠ **RENAMED AND CORRECTED 2026-07-30.** This was
        ``test_multiplier_recompute_complex_flips_to_bis`` and expected ``"bis"``, on the
        premise that "1,2-xylene is COMPLEX (digit + hyphen) => di flips to bis". The
        premise names a predicate the code does not use: a trace measured
        ``is_complex_substituent`` at **zero** calls from ``get_multiplier_prefix``.
        Measured, ``1,2-xylene`` is complex=True but **substituted=False**, so ``di`` is
        both what the code produces and what the Blue Book requires — /
        , with ``:25719``'s ``di(propan-2-yl)`` the analogous unsubstituted case
        against ``:25811``'s ``1,2-bis(bromomethyl)benzene (PIN)``.

        Reachable scope: **0 shipped names.** ``_recompute_multiplicative_prefix`` is
        reached only through ``apply_triviality_controller``, whose callers pass an
        ``enabled`` flag defaulting FALSE. So this was a stale test over dormant code,
        not a live defect — but the docstrings it agreed with had already caused one real
        regression (``), which is why both were corrected too.
        """
        entry = _entry("Cc1ccccc1C")
        out = _build_rewrite(NameTreeNode(parent_stem="x", multiplicative_prefix="di"), entry, ())
        assert out.multiplicative_prefix == "di"

    def test_multiplier_recompute_substituted_flips_to_bis(self):
        """The other side of the boundary: a genuinely *substituted* prefix takes ``bis``.

        Added 2026-07-30 alongside the correction above, so the pair pins the real
        discriminator rather than only one side of it. Without this, re-pointing the
        predicate at ``is_complex_substituent`` would leave the suite green.
        """
        from orthonym.assembly.naming_utils import get_multiplier_prefix
        assert get_multiplier_prefix(2, "bromomethyl") == "bis"
        assert get_multiplier_prefix(2, "1,2-xylene") == "di"
        assert get_multiplier_prefix(2, "propan-2-yl") == "di"


class TestCompleteNameFieldReset:
    """ + regression (code review 2026-05-30): ``_build_rewrite`` resets the fields a
    COMPLETE retained name already subsumes (Type 2a/2b/2c/3: locants, suffix, indicated_h,
    unsaturation) and PRESERVES the principal-group suffix for a BARE Type-1 parent hydride.
    Each case serializes the rewritten node to prove the absence of the double-render bug
    (pre-fix: "acetic acidoic acid", "1,2-1,2-xylene")."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,suffix,expected", [
        ("CC(=O)O", "oic acid", "acetic acid"),        #
        ("O=C(O)c1ccccc1", "ic acid", "benzoic acid"),
        ("O=C(O)C(=O)O", "dioic acid", "oxalic acid"),
    ])
    def test_acid_swap_clears_redundant_suffix(self, smiles, suffix, expected):
        from orthonym.assembly.name_tree_to_string import name_tree_to_string
        entry = _entry(smiles)
        out = _build_rewrite(NameTreeNode(parent_stem="systematic-stem", suffix=suffix), entry, ())
        assert out.suffix is None, "a complete acid name must clear the redundant node.suffix"
        assert name_tree_to_string(out, style="pin") == expected

    @pytest.mark.unit
    def test_phenol_swap_clears_suffix_and_locants(self):
        from orthonym.assembly.name_tree_to_string import name_tree_to_string
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzen", suffix="ol", locants=(1,)), entry, ())
        assert out.suffix is None and out.locants == ()
        assert name_tree_to_string(out, style="pin") == "phenol"

    @pytest.mark.unit
    def test_type1_preserves_principal_group_suffix(self):
        # Type 1 retained name is a BARE parent hydride; the -ol suffix is NOT embedded in
        # "naphthalene", so it MUST survive (dropping it would lose the principal group).
        entry = _entry("c1ccc2ccccc2c1")  # naphthalene (Type 1)
        out = _build_rewrite(NameTreeNode(parent_stem="naphthalen", suffix="ol", locants=(2,)), entry, ())
        assert out.parent_stem == "naphthalene"
        assert out.suffix == "ol" and out.locants == (2,)

    @pytest.mark.unit
    def test_type1_resets_indicated_h_no_double(self):
        # "1H-pyrrole" embeds its own indicated H; a node carrying indicated_h must not double it.
        from orthonym.assembly.name_tree_to_string import name_tree_to_string
        entry = _entry("c1cc[nH]c1")
        out = _build_rewrite(NameTreeNode(parent_stem="azole-stem", indicated_h=(1,)), entry, ())
        assert out.indicated_h == ()
        assert name_tree_to_string(out, style="pin") == "1H-pyrrole"


class TestFragmentLegacyConditionalReset:
    """#2 FIX (reviews iter 1): _replace_preserving_or_resetting_legacy resets fragment_legacy
    when a child changed (the serializer short-circuit would otherwise discard the rewrite) and
    preserves it when nothing changed."""

    @pytest.mark.unit
    def test_no_child_change_preserves_fragment_legacy(self):
        c1 = NameTreeNode(parent_stem="bromo")
        node = NameTreeNode(parent_stem="benzene", prefixes=(c1,), fragment_legacy="X")
        # Pass the SAME (already-alphabetized single) prefixes => nothing changed => preserve.
        out = _replace_preserving_or_resetting_legacy(node, node.prefixes)
        assert out.fragment_legacy == "X"

    @pytest.mark.unit
    def test_child_changed_resets_fragment_legacy(self):
        c1 = NameTreeNode(parent_stem="bromo")
        node = NameTreeNode(parent_stem="benzene", prefixes=(c1,), fragment_legacy="X")
        new_child = NameTreeNode(parent_stem="chloro")  # a DIFFERENT child
        out = _replace_preserving_or_resetting_legacy(node, (new_child,))
        assert out.fragment_legacy is None  # child changed => reset (serializer-short-circuit closed)


class TestNoPathA:
    """#3 FIX (reviews iter 1): the hint-based recovery path was removed — the module source
    carries zero ``atom_to_locant_hint`` references; recovery is a 3-path B/C/D cascade."""

    @pytest.mark.unit
    def test_no_atom_to_locant_hint_reference(self):
        source = inspect.getsource(rs)
        assert "atom_to_locant_hint" not in source

    @pytest.mark.unit
    def test_recover_has_three_paths_not_path_a(self):
        source = inspect.getsource(rs)
        assert "Path B" in source and "Path C" in source and "Path D" in source
        assert "Path A" not in source
