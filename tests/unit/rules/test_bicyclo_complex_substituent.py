""" a phase (beta-lactam layer 2): the bicyclo composer's OWN substituent
builder (`_build_bicyclo_substituent_prefix`, composer.py) had no recursive
substituent namer -- any heteroatom-bearing substituent (acylamino,
acyloxymethyl) was silently mis-spelled as a same-carbon-count alkyl, because
`get_bicyclo_substituents` records only a raw carbon COUNT and the consumer
fell straight to `get_alkyl_name(carbon_count)` for anything that wasn't a
bare halogen/hydroxy/amino/pure-hydrocarbon.

Layer 1 (commit 8217684f) fixed `is_bicyclo_system`'s pendant-ring scope bug
so the bicyclo composer is REACHED for penicillin-G-shaped molecules; this
layer fixes what it does once reached. a trace (see
`.the workflow tooling/sdd/2026-08-17--phase3-acid-ester-anion/trace-betalactam-full.md`)
named `name_substituent`/`_enrich_complex_ring_with_subs` as the candidate
reuse target. Direct probing found a REFINEMENT: `name_substituent` alone
returns the `'substituent'` sentinel for a branched/ring-containing acyl in
an acylamino branch (phenylacetamido, penicillin G's own side chain) -- it
has no route to the ring-in-acyl-subtree fix that already lives in
`_check_for_acylamino` (composer.py:8851), the primitive
`rules/polyfunctional.py:3272` already uses for the identical problem on a
CHAIN parent. The fix therefore tries BOTH existing primitives in sequence
(`_check_for_acylamino` first, then `name_substituent`), reusing two already
-shipped, already-tested naming mechanisms -- no new substituent namer was
written. Neither succeeding fails the whole bicyclo parent closed (returns
None), matching the existing `unnameable` contract already in this
function.
"""
from unittest.mock import patch

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.namer import _validity_gate_name_to_smiles


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def _inchikey(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m else None


def _assert_rt_exact(smiles, name):
    assert name, f"no name produced for {smiles!r}"
    assert name != "unknown organic compound", (
        f"{smiles!r} abstained instead of emitting a name"
    )
    opsin_smiles = _validity_gate_name_to_smiles(name)
    assert opsin_smiles is not None, f"OPSIN could not parse {name!r}"
    assert _inchikey(opsin_smiles) == _inchikey(smiles), (
        f"NOT rt_exact: {name!r} -> {opsin_smiles!r} "
        f"({_inchikey(opsin_smiles)}) != input {smiles!r} ({_inchikey(smiles)})"
    )


# ============================================================================
# 1. Integration: penicillin G (fused bicyclic, needs layer 1 + layer 2)
# ============================================================================

@pytest.mark.opsin_gate
class TestPenicillinG:
    def test_penicillin_g_names_and_round_trips(self):
        """
        (2S,5R,6R)-3,3-dimethyl-7-oxo-6-{acylamino}-4-thia-1-azabicyclo[3.2.0]-
        heptane-2-carboxylic acid. The 6-substituent is the acylamino side
        chain -NHC(=O)CH2C6H5 (2-phenylacetamido); before this fix its 8
        carbons were mis-spelled '6-octyl' (correctly caught it as a
        different molecule -- 0-wrong held, but the molecule abstained).

        Per the task's own instruction, the EXACT substituent spelling is not
        forced here -- '(2-phenylacetamido)' method (1),
        the Blue Book contracted amido form, already verified elsewhere in
        this codebase) is accepted as well as the method-(2)
        '[(2-phenylacetyl)amino]' form; only round-trip-exactness to the
        input's InChIKey is asserted.
        """
        smiles = "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"
        name = name_compound(smiles)
        _assert_rt_exact(smiles, name)
        # The acylamino side chain must be present in SOME nameable form --
        # not silently dropped/replaced by a same-carbon-count alkyl.
        assert "octyl" not in name, f"atom-dropped alkyl substituent leaked: {name!r}"


# ============================================================================
# 2. Integration: simpler acylamino-on-bicyclo (bridged, layer-1-independent)
# ============================================================================

@pytest.mark.opsin_gate
class TestSimplerAcylaminoOnBicyclo:
    def test_acetamido_norbornane_carboxylic_acid(self):
        """
        2-acetamidobicyclo[2.2.1]heptane-1-carboxylic acid. A BRIDGED
        bicyclic (bridge lengths never 0), so this exercises layer 2 (the
        substituent builder) in isolation from layer 1's zero-bridge
        pendant-ring scope fix -- `is_bicyclo_system` already returned True
        for this shape before either fix. Before layer 2: the acetamido's 2
        carbons were mis-spelled '2-ethyl' (caught it; the molecule
        abstained).
        """
        smiles = "CC(=O)NC1CC2CCC1(C2)C(=O)O"
        name = name_compound(smiles)
        _assert_rt_exact(smiles, name)
        assert "ethyl" not in name, f"atom-dropped alkyl substituent leaked: {name!r}"

    def test_cephem_acetoxymethyl(self):
        """
        Cephalothin-like cephem core with a C3' acetoxymethyl substituent
        (-CH2-O-C(=O)-CH3): `is_bicyclo_system` already returns True for this
        exact shape without needing layer 1 (no pendant ring >= 5 anywhere),
        so this isolates layer 2 alone. Before the fix: the acetoxymethyl's
        3 carbons were mis-spelled '3-propyl' (caught it; the
        molecule abstained). `name_substituent` alone (no
        `_check_for_acylamino` needed here) spells this correctly as
        '(acetyloxy)methyl'.
        """
        smiles = "OC(=O)C1=C(COC(C)=O)CSC2CC(=O)N12"
        name = name_compound(smiles)
        _assert_rt_exact(smiles, name)
        assert "propyl" not in name, f"atom-dropped alkyl substituent leaked: {name!r}"


# ============================================================================
# 3. Regression: pure-hydrocarbon bicyclo substituents are UNCHANGED
# ============================================================================

class TestPureHydrocarbonSubstituentUnchanged:
    """The new is_pure_hydrocarbon branch-point must not reroute a plain
    alkyl substituent (no heteroatoms) through the recursive namer -- the
    pre-existing `get_alkyl_name(carbon_count)` path stays byte-identical."""

    def test_methylbicycloheptane_unchanged(self):
        smiles = "C1CC2CCC1(C)C2"
        assert name_compound(smiles) == "1-methylbicyclo[2.2.1]heptane"

    def test_norbornane_unchanged(self):
        smiles = "C1CC2CCC1C2"
        assert name_compound(smiles) == "norbornane"


# ============================================================================
# 4. Fail-closed: a substituent neither primitive can spell -> abstain
# ============================================================================

class TestFailClosedOnUnspellableSubstituent:
    def test_neither_primitive_available_declines_whole_parent(self):
        """
        White-box contract test, calling `_build_bicyclo_substituent_prefix`
        DIRECTLY (not through the full `name_compound` pipeline, whose other
        composers/fallback engines also call `_check_for_acylamino` /
        `name_substituent` and would be contaminated by a global patch,
        masking what this specific function does). When BOTH primitives
        decline (forced via monkeypatch to simulate a heteroatom-bearing
        substituent shape neither existing primitive can spell),
        `_build_bicyclo_substituent_prefix` must return None -- the whole
        bicyclo parent aborts (its caller, `_assemble_complete_bicyclo_name`,
        already declines cleanly on a `None` sub_prefix) -- rather than
        falling through to `get_alkyl_name(carbon_count)` and emitting a
        wrong alkyl that silently drops the heteroatom.
        """
        import orthonym.assembly.composer as composer_module

        # Real inputs, captured from the exact call `_assemble_complete_
        # bicyclo_name` makes for the acetamido-norbornane-carboxylic-acid
        # case above (traced, not hand-built, so the substituents/atom_to_
        # locant shapes are authentic).
        captured = {}
        original = composer_module._build_bicyclo_substituent_prefix

        def spy(mol, substituents, atom_to_locant, fg_prefixes=None):
            captured["args"] = (mol, substituents, atom_to_locant, fg_prefixes)
            return original(mol, substituents, atom_to_locant, fg_prefixes)

        composer_module._build_bicyclo_substituent_prefix = spy
        try:
            name_compound("CC(=O)NC1CC2CCC1(C2)C(=O)O")
        finally:
            composer_module._build_bicyclo_substituent_prefix = original

        mol, substituents, atom_to_locant, fg_prefixes = captured["args"]

        with patch(
            "orthonym.assembly.composer._check_for_acylamino",
            return_value=None,
        ), patch(
            "orthonym.assembly.substituent_enumerator.name_substituent",
            return_value="substituent",
        ):
            result = original(mol, substituents, atom_to_locant, fg_prefixes)

        assert result is None, (
            f"expected a fail-closed None when neither primitive can spell "
            f"the substituent, got {result!r} instead (must never be a "
            f"wrong alkyl)"
        )
