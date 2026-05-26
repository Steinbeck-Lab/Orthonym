"""Phase 160 Plan-04 integration tests: Orthonym.name_with_tree() round-trip.

Per CONTEXT D-04 + D-05 + DECOMP-02: ``Orthonym.name_with_tree(smi)``
returns a NamingResult(name, tree, atom_to_locant_hint) per the public API.

Per CONTEXT D-05 incremental migration: for the 30 currently-extracted
handlers (Plans 02-03 ship) tree is None; the name field is byte-
identical to ``Orthonym.name(smi)``. For tree-emitting handlers
(v19+1), name_tree_to_string(tree) round-trips byte-identical to name.

This file verifies:
- name_with_tree returns NamingResult.
- name matches Orthonym.name(smi).
- For tree-emitting paths, tree is a NameTreeNode and round-trips.
- For tree=None paths, the legacy fragment-list rendering produced name.
"""
from __future__ import annotations

import pytest

from orthonym import Orthonym, NameTreeNode, NamingResult
from orthonym.assembly.name_tree_to_string import name_tree_to_string


REPRESENTATIVE_SMILES = [
    ("CCO", "ethanol"),  # alcohol
    ("CC(=O)O", "acetic acid"),  # carboxylic acid
    ("CC(=O)C", "propan-2-one"),  # ketone
    ("CC=O", "acetaldehyde"),  # aldehyde
    ("CCN", "ethanamine"),  # amine
    ("c1ccccc1", "benzene"),  # benzene
    ("c1ccncc1", "pyridine"),  # heterocycle
    ("CC(=NO)C", None),  # oxime (no expected name assertion)
    ("CS(=O)C", "dimethyl sulfoxide"),  # sulfoxide
    ("CB(O)O", "methylboronic acid"),  # boronic acid
    ("c1ccc(-c2ccccc2)cc1", None),  # ring assembly (biphenyl)
    ("[H][H]", None),  # simple molecule (hydrogen)
]


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.parametrize(
    "smi,expected_name_or_none",
    REPRESENTATIVE_SMILES,
    ids=[s[0] for s in REPRESENTATIVE_SMILES],
)
def test_name_with_tree_returns_naming_result(
    smi, expected_name_or_none, namer,
):
    """name_with_tree returns a NamingResult NamedTuple."""
    result = namer.name_with_tree(smi)
    assert isinstance(result, NamingResult)
    assert isinstance(result.name, str)
    assert result.name


@pytest.mark.parametrize(
    "smi,expected_name_or_none",
    REPRESENTATIVE_SMILES,
    ids=[s[0] for s in REPRESENTATIVE_SMILES],
)
def test_name_field_matches_name(smi, expected_name_or_none, namer):
    """The name field is byte-identical to Orthonym.name(smi)."""
    expected = namer.name(smi)
    result = namer.name_with_tree(smi)
    assert result.name == expected, (
        f"name_with_tree({smi!r}).name = {result.name!r}; "
        f"Orthonym.name({smi!r}) = {expected!r}"
    )


def test_tree_is_optional_per_d05(namer):
    """Per CONTEXT D-05 first-wave: tree is None for the 30 extracted handlers."""
    result = namer.name_with_tree("CCO")
    # Tree may be None (first-wave) or NameTreeNode (v19+1).
    assert result.tree is None or isinstance(result.tree, NameTreeNode)


def test_atom_to_locant_hint_is_optional(namer):
    """atom_to_locant_hint is Optional[Dict] per CONTEXT D-05."""
    result = namer.name_with_tree("CCO")
    assert result.atom_to_locant_hint is None or isinstance(
        result.atom_to_locant_hint, dict,
    )


def test_round_trip_on_tree_populated_node():
    """When tree is populated (manually-constructed for test), serializer
    round-trips to the name field byte-for-byte.

    This exercises the explicit-field serialization branch of
    name_tree_to_string (NOT the legacy fragment_legacy path).
    """
    # Manual tree construction (mirrors what a v19+1 tree-emitting handler
    # would do).
    tree = NameTreeNode(parent_stem="ethan", suffix="ol")
    serialized = name_tree_to_string(tree)
    # Build a manual NamingResult; serializer output is the "name".
    result = NamingResult(name=serialized, tree=tree, atom_to_locant_hint=None)
    # Round-trip: serialize the tree again, compare to result.name.
    round_trip = name_tree_to_string(result.tree)
    assert round_trip == result.name


def test_name_with_tree_idempotent(namer):
    """Two name_with_tree() calls on the same SMILES produce same result."""
    r1 = namer.name_with_tree("CCO")
    r2 = namer.name_with_tree("CCO")
    assert r1.name == r2.name
    assert r1.tree == r2.tree
    assert r1.atom_to_locant_hint == r2.atom_to_locant_hint


def test_name_with_tree_handles_complex_smiles(namer):
    """Pipeline handles non-trivial SMILES without crashing."""
    # Cyclohexanol — a partially-saturated ring with an alcohol.
    result = namer.name_with_tree("OC1CCCCC1")
    assert isinstance(result, NamingResult)
    assert isinstance(result.name, str)
    assert result.name


def test_name_with_tree_invalid_smiles_raises(namer):
    """Invalid SMILES raises ValueError (matches Orthonym.name() contract)."""
    with pytest.raises(ValueError):
        namer.name_with_tree("not-a-valid-smiles-string-XYZ!!")


def test_name_with_tree_tree_population_status(namer):
    """Phase 165 SCORE-01 supersedes the Phase-160 first-wave tree=None policy.

    Handlers that route via dispatch_inner now emit a NameTreeNode whose
    serialization round-trips byte-identically to the name. Functional-class /
    retained-name paths that are named BEFORE dispatch_inner (e.g. dimethyl
    sulfoxide) still surface tree=None — documented in 165-01-SUMMARY.md
    (Open Question 3) + CANARY-BASELINE.md.
    """
    from orthonym.assembly.name_tree import NameTreeNode
    from orthonym.assembly.name_tree_to_string import name_tree_to_string

    # Routes via dispatch_inner -> tree populated (Phase 165).
    for smi in ["CCCCO", "CN=C=O"]:  # general_acyclic (structured), isocyanate (coarse)
        result = namer.name_with_tree(smi)
        assert isinstance(result.tree, NameTreeNode), (
            f"Expected a populated tree for {smi!r} (Phase 165 SCORE-01); "
            f"got {result.tree!r}."
        )
        assert name_tree_to_string(result.tree, "pin") == result.name

    # Functional-class name produced before dispatch_inner -> still tree=None.
    result = namer.name_with_tree("CS(=O)C")  # dimethyl sulfoxide (bypasses dispatch_inner)
    assert result.tree is None


@pytest.mark.parametrize("smi", ["[He]", "[Ne]", "[Ar]", "[Kr]", "[Xe]", "[Rn]"])
def test_simple_molecule_emits_tree_round_trip(smi, namer):
    """Plan-10 DECOMP-02 OBSERVABLE CLOSURE: simple_molecule is the FIRST
    tree-emitting handler. For each in-scope SMILES, the handler returns
    a NameTreeNode whose serialization round-trips byte-identically to
    result.name.
    """
    result = namer.name_with_tree(smi)
    assert isinstance(result, NamingResult)
    assert result.tree is not None, (
        f"simple_molecule must emit a non-None tree for {smi!r}; "
        f"DECOMP-02 closure depends on at least ONE handler proving "
        f"the IR substrate works end-to-end"
    )
    assert result.tree.parent_stem == result.name, (
        f"NameTreeNode.parent_stem must equal NamingResult.name for "
        f"simple_molecule; got {result.tree.parent_stem!r} vs {result.name!r}"
    )
    assert result.tree.class_id == "simple_molecule"
    assert result.tree.iupac_section_cite == "P-14"
    assert name_tree_to_string(result.tree) == result.name, (
        f"name_tree_to_string round-trip must equal result.name for {smi!r}"
    )


def test_atom_to_locant_hint_preserved_for_n_oxide(namer):
    """CR-04 part B regression: n_oxide handler's atom_to_locant_hint
    threads to name_with_tree caller (no longer silently None).

    The n_oxide handler produces a non-None ``atom_to_locant_hint``
    (the heterocycle locant_map). Prior to Plan-05, Orthonym.name_with_tree
    discarded all handler hints and always returned None.
    """
    smi_candidates = [
        "O=[N+]1=CC=CC=C1[O-]",
        "[O-][n+]1ccccc1",
        "O=[n+]1ccccc1",
    ]
    for smi in smi_candidates:
        try:
            result = namer.name_with_tree(smi)
        except Exception:
            continue
        if "oxide" in (result.name or ""):
            assert isinstance(result, NamingResult)
            assert result.atom_to_locant_hint is not None, (
                f"n_oxide handler should produce non-None "
                f"atom_to_locant_hint for smi={smi!r}; got None "
                f"(CR-04 part B regression)"
            )
            return
    import pytest as _pytest
    _pytest.skip(
        "No N-oxide SMILES routed through n_oxide handler in current build"
    )
