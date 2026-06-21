"""v22 Phase G4 — spiro polycyclic cluster (COV-04, P-24.2 / P-24.5).

Rule-family (A8) tests for the three shipped G4 sub-classes:

  Build 1  heteroatom multiplier cap   -> `hexaoxa` not the malformed `6-oxa`
  Build 2  lambda-convention (P-31.1.4.2) for non-standard-valence ring atoms
  Build 3  P-24.5.1 alphanumerical component order for mixed spiro-fused

plus the A10 honest-deferral guard: the spirobi polycyclic-component case
(DD7-spiro-1) must stay FAIL-CLOSED (`unknown`), never a wrong name.

All assertions test the OUTPUT of the rule family, not literal canaries, and
the expected PINs were OPSIN-2.9.0 round-trip verified before being written.
"""

import re

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.spiro import (
    _component_alpha_key,
    _nonstandard_bonding_number,
    _strip_consumed_indicated_h,
    name_mixed_spiro_fused,
    name_spiro_system,
)


def _spiro_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles!r}"
    result = name_spiro_system(mol)
    assert result is not None, f"name_spiro_system declined {smiles!r}"
    return result[0]


def _mixed_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles!r}"
    result = name_mixed_spiro_fused(mol)
    assert result is not None, f"name_mixed_spiro_fused declined {smiles!r}"
    return result[0]


@pytest.mark.unit
class TestSpiroHeteroMultiplier:
    """Build 1: the 'a'-prefix multiplier routes through the shared generator,
    so 6+ heteroatoms spell hexaoxa/heptaoxa/... — never the malformed
    `<n>-oxa` the old penta-capped dict produced."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("O1COCOC12OCOCO2", "1,3,5,7,9,11-hexaoxaspiro[5.5]undecane"),  # 6 (gold)
            ("C1COC2(CO1)COCO2", "1,3,6,9-tetraoxaspiro[4.5]decane"),       # 4
            ("O1CCOCC12OCCOC2", "1,4,7,10-tetraoxaspiro[5.5]undecane"),     # 4 (lowest-locant)
        ],
    )
    def test_multiplier_family(self, smiles, expected):
        assert _spiro_name(smiles) == expected

    def test_six_heteroatoms_use_hexa_not_digit_oxa(self):
        name = _spiro_name("O1COCOC12OCOCO2")
        assert "hexaoxa" in name
        # the old bug emitted a bare "<digit>-oxa" multiplier token
        assert not re.search(r"\b\d-oxa", name), name

    def test_lower_counts_unaffected(self):
        # di/tetra still correct after swapping in the shared generator
        assert _spiro_name("C1COC2(CO1)COCO2").count("tetraoxa") == 1


@pytest.mark.unit
class TestSpiroLambda:
    """Build 2: a ring skeletal atom whose valence differs from its IUPAC
    standard bonding number carries the lambda convention (P-31.1.4.2);
    standard-valence atoms get NO lambda (fail-closed)."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("C1CCS12CCCCC2", "4lambda4-thiaspiro[3.5]nonane"),    # gold (S val 4)
            ("C1CC[Se]12CCCCC2", "4lambda4-selenaspiro[3.5]nonane"),  # Se val 4
            ("C1CCS12CCCCCC2", "4lambda4-thiaspiro[3.6]decane"),   # other ring sizes
        ],
    )
    def test_lambda_family(self, smiles, expected):
        assert _spiro_name(smiles) == expected

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),  # divalent O -> no lambda
            ("S1CCCC12CCCCC2", "1-thiaspiro[4.5]decane"),  # divalent S -> no lambda
        ],
    )
    def test_standard_valence_no_lambda(self, smiles, expected):
        name = _spiro_name(smiles)
        assert name == expected
        assert "lambda" not in name


@pytest.mark.unit
class TestSpiroFusedAlphanumerical:
    """Build 3: P-24.5.1 cites the two ring components in ALPHANUMERICAL order
    of the component name (Note: NOT by seniority, NOT fused-first), priming
    the second; indicated H at the spiro locant is dropped."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("C1CCCC12C=Cc1ccccc12", "spiro[cyclopentane-1,1'-indene]"),  # gold
            ("C1CCCCC12C=Cc1ccccc12", "spiro[cyclohexane-1,1'-indene]"),
            ("C1CCC12C=Cc1ccccc12", "spiro[cyclobutane-1,1'-indene]"),
        ],
    )
    def test_alphanumerical_family(self, smiles, expected):
        assert _mixed_name(smiles) == expected

    def test_cyclo_component_cited_before_indene(self):
        # 'cyclopentane' (c) sorts before 'indene' (i) -> cited first / unprimed
        name = _mixed_name("C1CCCC12C=Cc1ccccc12")
        assert name.index("cyclopentane") < name.index("indene")

    def test_indicated_h_dropped_at_spiro_locant(self):
        # spiro at indene C1 (the indicated-H position) -> no '1H-'
        assert "1h-indene" not in _mixed_name("C1CCCC12C=Cc1ccccc12").lower()


@pytest.mark.unit
class TestSpirobiFailClosedA10:
    """A10 honest-deferral: the spirobi polycyclic-component class (DD7-spiro-1)
    is NOT built — it must stay FAIL-CLOSED ('unknown'), never a wrong name.
    Guards against a future change silently emitting a structurally-wrong
    spirobi/von-Baeyer name."""

    def test_spirobi_indane_is_fail_closed(self):
        name = name_compound("C1Cc2ccccc2C13Cc1ccccc1C3")
        assert name is not None and "unknown" in name.lower(), name

    def test_spirobi_not_claimed_by_spiro_handlers(self):
        from orthonym.rules.spiro import is_mixed_spiro_fused, is_spiro_system
        mol = Chem.MolFromSmiles("C1Cc2ccccc2C13Cc1ccccc1C3")
        # Both-sides-fused polycyclic spiro: the DETECTORS must NOT claim it
        # (is_spiro_system requires n_rings == n_spiro + 1; this is 4 rings /
        # 1 spiro). That keeps the wrong-name spiro handlers off the dispatch
        # path (name_spiro_system would otherwise emit a bogus spiro[4.4]nonane
        # from the spiro atom's two 5-rings, dropping both benzenes) and routes
        # the molecule to the G0 fail-closed backstop instead.
        assert is_spiro_system(mol) is False
        assert is_mixed_spiro_fused(mol) is False


@pytest.mark.unit
class TestSpiroNumberingDeterminism:
    """The G4 get_spiro_numbering rewrite must give SMILES-order-independent,
    LOWEST-locant heteroatom numbering — closing the pre-existing A9 trap where
    `1-oxaspiro[4.5]decane` non-deterministically flipped to `4-oxaspiro[4.5]`
    and symmetric acetals emitted several distinct locant sets per spelling."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),
            ("S1CCCC12CCCCC2", "1-thiaspiro[4.5]decane"),
            ("O1CCOCC12OCCOC2", "1,4,7,10-tetraoxaspiro[5.5]undecane"),
            ("O1COCOC12OCOCO2", "1,3,5,7,9,11-hexaoxaspiro[5.5]undecane"),
        ],
    )
    def test_atom_order_independent(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        # Re-spell the molecule rooted at every atom (deterministic, no RNG)
        # and re-parse — the same way the v22 determinism eval probes for
        # SMILES-order dependence — then assert one stable, correct name.
        names = set()
        for root in range(mol.GetNumAtoms()):
            spelling = Chem.MolToSmiles(mol, rootedAtAtom=root, canonical=False)
            rm = Chem.MolFromSmiles(spelling)
            if rm is None:
                continue
            result = name_spiro_system(rm)
            names.add(result[0] if result else None)
        assert names == {expected}, names


@pytest.mark.unit
class TestSpiroG4Helpers:
    """Unit coverage for the three G4 helpers."""

    def test_nonstandard_bonding_number_flags_tetravalent_sulfur(self):
        mol = Chem.MolFromSmiles("C1CCS12CCCCC2")
        s_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")
        assert _nonstandard_bonding_number(mol, s_idx) == 4

    def test_nonstandard_bonding_number_none_for_divalent_oxygen(self):
        mol = Chem.MolFromSmiles("O1CCCC12CCCCC2")
        o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
        assert _nonstandard_bonding_number(mol, o_idx) is None

    @pytest.mark.parametrize(
        "name,key",
        [
            ("1H-indene", "indene"),
            ("3H-indene", "indene"),
            ("cyclopentane", "cyclopentane"),
            ("indoline", "indoline"),
        ],
    )
    def test_component_alpha_key_strips_indicated_h(self, name, key):
        assert _component_alpha_key(name) == key

    def test_strip_indicated_h_only_at_matching_locant(self):
        assert _strip_consumed_indicated_h("1H-indene", 1) == "indene"
        # spiro at a different locant -> keep the indicated H (fail-safe)
        assert _strip_consumed_indicated_h("1H-indene", 2) == "1H-indene"
        assert _strip_consumed_indicated_h("cyclopentane", 1) == "cyclopentane"
