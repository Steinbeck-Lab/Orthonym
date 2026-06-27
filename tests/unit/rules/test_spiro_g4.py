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
class TestSpirobiBuiltP24_3:
    """v23 Phase 13: the spirobi polycyclic-component class (P-24.3.1) is now
    BUILT (`name_spirobi`). Two identical polycyclic components at one spiro
    atom -> ``<lo>,<hi>'-spirobi[component]``, RT-verified. (Previously this
    class was A10 honest-deferral / fail-closed; the build supersedes it.)"""

    def test_spirobi_indene_named(self):
        # 1H-indene mancude component; spiro at C1 of both -> 1,1'.
        assert name_compound("C1=Cc2ccccc2C13C=Cc1ccccc13") == "1,1'-spirobi[indene]"

    def test_spirobi_indane_named(self):
        # Spiro is benzylic (locant 1) in one indane, the middle carbon
        # (locant 2) in the other -> 1,2' (NOT 1,1'); RT-verified PIN.
        assert name_compound("C1Cc2ccccc2C13Cc1ccccc1C3") == "1,2'-spirobi[indane]"

    def test_spirobi_claimed_by_new_detector_only(self):
        from orthonym.rules.spiro import (
            is_mixed_spiro_fused, is_spiro_system, is_spirobi,
        )
        mol = Chem.MolFromSmiles("C1Cc2ccccc2C13Cc1ccccc1C3")
        # The legacy detectors still decline (is_spiro_system requires
        # n_rings == n_spiro + 1; this is 4 rings / 1 spiro; mixed-spiro-fused
        # requires a single side ring). The NEW P-24.3 detector claims it.
        assert is_spiro_system(mol) is False
        assert is_mixed_spiro_fused(mol) is False
        assert is_spirobi(mol) is True


@pytest.mark.unit
class TestSpiroVonBaeyerP24_5:
    """v23 Phase 13B(c): monospiro systems with >=1 von Baeyer (bridged) cage
    component (P-24.5 component-name form, or P-24.3.1 spirobi for two identical
    cages). Built by ``name_spiro_vonbaeyer`` via a robust atom-based
    separation-atom finder that handles a cage-bridge spiro atom sitting in >2
    SSSR rings (where ``get_spiro_atoms`` fails). The von Baeyer component is
    cited by its SYSTEMATIC ``bicyclo[...]`` name, not a retained name."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            # VB cage + monocycle (component-name form, alphanumerical order b<c).
            ("C1CCC2(CC1)CC1CCC2C1",
             "spiro[bicyclo[2.2.1]heptane-2,1'-cyclohexane]"),
            ("C1CCC2(CC1)CC1CCC2CC1",
             "spiro[bicyclo[2.2.2]octane-2,1'-cyclohexane]"),
            ("C1CCC2(C1)CC1CCC2C1",
             "spiro[bicyclo[2.2.1]heptane-2,1'-cyclopentane]"),
            # Spiro at the 1-atom bridge (cage position 7, in BOTH cage SSSR rings).
            ("C1CCC2(CC1)C1CCC2CC1",
             "spiro[bicyclo[2.2.1]heptane-7,1'-cyclohexane]"),
            # Two identical cages -> spirobi (P-24.3.1).
            ("C1CC2CC1CC21CC2CCC1C2", "2,2'-spirobi[bicyclo[2.2.1]heptane]"),
            ("C1CC2CCC1CC21CC2CCC1CC2", "2,2'-spirobi[bicyclo[2.2.2]octane]"),
        ],
    )
    def test_spiro_vonbaeyer_named(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_systematic_not_retained_component_name(self):
        # The cage MUST be cited systematically (bicyclo[2.2.1]heptane), never
        # the retained 'norbornane' — that also fixes the citation order (b<c).
        out = name_compound("C1CCC2(CC1)CC1CCC2C1")
        assert "bicyclo[2.2.1]heptane" in out
        assert "norbornane" not in out

    def test_claimed_by_new_detector_only(self):
        from orthonym.rules.spiro import (
            is_mixed_spiro_fused, is_spiro_system, is_spirobi,
            is_spiro_vonbaeyer,
        )
        # bicyclo[2.2.2]octane spiro cyclohexane: the cage-position-2 spiro atom
        # is in both cage SSSR rings, so all legacy detectors decline.
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1CCC2CC1")
        assert is_spiro_system(mol) is False
        assert is_mixed_spiro_fused(mol) is False
        assert is_spirobi(mol) is False
        assert is_spiro_vonbaeyer(mol) is True

    @pytest.mark.parametrize(
        "smiles",
        [
            "C1C2CC3CC1CC(C2)C3",      # adamantane (pure cage)
            "C1CC2CCC1C2",             # norbornane (pure bicyclo)
            "C1CCC2(CC1)CCCC2",        # spiro[4.5]decane (two monocycles)
            "C1CCC2CCCCC2C1",          # decalin (fused bicyclic)
            "C1CCC2(CC1)CC1(CCCCC1)C2",  # dispiro[5.1.5.1]tetradecane (polyspiro)
        ],
    )
    def test_fail_closed_declines(self, smiles):
        from orthonym.rules.spiro import is_spiro_vonbaeyer
        mol = Chem.MolFromSmiles(smiles)
        assert is_spiro_vonbaeyer(mol) is False

    def test_determinism_across_spellings(self):
        # The spiro locant + citation order must be SMILES-order-independent.
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1CCC2C1")
        names = set()
        for root in range(mol.GetNumAtoms()):
            spelling = Chem.MolToSmiles(mol, rootedAtAtom=root, canonical=False)
            rm = Chem.MolFromSmiles(spelling)
            if rm is not None:
                names.add(name_compound(Chem.MolToSmiles(rm)))
        assert names == {"spiro[bicyclo[2.2.1]heptane-2,1'-cyclohexane]"}, names


@pytest.mark.unit
class TestSpiroPAHFluoreneP24_5:
    """v23 Phase 13B(b): spiro systems with a fused CARBOCYCLIC-PAH component
    (fluorene), named via a numbered template. Headlined by 9,9'-spirobifluorene
    (the spiro-OLED core). The fluorene component is cited by its retained PAH
    name with the indicated H consumed by the C9 spiro atom."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc21",
             "9,9'-spirobi[fluorene]"),
            ("c1ccc2c(c1)-c1ccccc1C21CCCCC1",
             "spiro[cyclohexane-1,9'-fluorene]"),
            ("c1ccc2c(c1)-c1ccccc1C21CCCC1",
             "spiro[cyclopentane-1,9'-fluorene]"),
            # fluorene (carbo-PAH) + xanthene (fused heterocycle) co-component.
            ("c1ccc2c(c1)Oc1ccccc1C21c2ccccc2-c2ccccc21",
             "spiro[fluorene-9,9'-xanthene]"),
        ],
    )
    def test_spiro_pah_named(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_catalog_skip_guard_is_precise(self):
        # The _input_is_spiro_vb catalog-skip guard must NOT steal genuine fused
        # heterocycles that merely contain a catalog sub-core. Oxanthrene
        # (dibenzo-p-dioxin) has no spiro atom -> stays on the catalog path.
        assert name_compound("O1c2ccccc2Oc2ccccc21") == "oxanthrene"
        # Bare xanthene / fluorene keep their catalog names.
        assert name_compound("c1ccc2c(c1)Cc1ccccc1O2") == "9H-xanthene"
        assert name_compound("c1ccc2c(c1)Cc3ccccc3-2") == "fluorene"

    def test_spirobifluorene_determinism(self):
        mol = Chem.MolFromSmiles(
            "c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc21"
        )
        names = set()
        for root in range(0, mol.GetNumAtoms(), 3):
            spelling = Chem.MolToSmiles(mol, rootedAtAtom=root, canonical=False)
            rm = Chem.MolFromSmiles(spelling)
            if rm is not None:
                names.add(name_compound(Chem.MolToSmiles(rm)))
        assert names == {"9,9'-spirobi[fluorene]"}, names


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
