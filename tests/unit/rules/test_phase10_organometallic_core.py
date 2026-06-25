"""Unit tests for v23 Phase 10 — P-69 organometallic / catenated-hydride core (M-L).

Three clusters, all OPSIN-2.9.0 round-trip verified:
  * 10a ethenyl/vinyl sigma-ligand recognition (root-cause fix in
    rules.organometallics._ligand_name_from_atoms): a 2-carbon C=C sigma-ligand
    is 'ethenyl', not 'ethyl' (the double bond was dropped = structure loss).
  * 10b metal carbonyls in the disconnected (dot) representation.
  * 10c two-atom Group-14/Group-15 catenated parent hydride (P-69.5.3): a new
    rules.mononuclear_hydrides.name_dinuclear_hydride at DINUCLEAR_HYDRIDE@48.

These tests run with the OPSIN validity gate disabled (conftest autouse), so
name_compound returns the RAW production name.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.mononuclear_hydrides import name_dinuclear_hydride


def _dinuc(smiles):
    return name_dinuclear_hydride(Chem.MolFromSmiles(smiles))


class TestEthenylLigand:
    """10a: a clean 2-carbon C=C sigma-ligand on a metal is 'ethenyl' (PIN)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("C=C[Na]", "ethenylsodium"),
        ("C=C[Li]", "ethenyllithium"),
        ("C=C[K]", "ethenylpotassium"),
        ("C=C[Mg]Br", "ethenylmagnesium bromide"),
    ])
    def test_ethenyl_wins(self, smiles, expected):
        assert name_compound(smiles, style="pin") == expected

    @pytest.mark.parametrize("smiles,expected", [
        # The double bond must NOT be dropped: saturated alkyls stay alkyl.
        ("CC[Li]", "ethyllithium"),
        ("C[Li]", "methyllithium"),
        ("C[Na]", "methylsodium"),
        ("CCCC[Li]", "butyllithium"),
        ("CC[Pb](CC)(CC)CC", "tetraethylplumbane"),
    ])
    def test_saturated_alkyls_unchanged(self, smiles, expected):
        """Regression guard: the ethenyl branch only fires on a C=C double bond."""
        assert name_compound(smiles, style="pin") == expected


class TestMetalCarbonylDotForm:
    """10b: disconnected (dot) metal carbonyls name + round-trip."""

    @pytest.mark.parametrize("smiles,expected", [
        ("[Fe].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]", "pentacarbonyliron"),
        ("[Ni].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]", "tetracarbonylnickel"),
        ("[Cr].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]", "hexacarbonylchromium"),
        ("[Mo].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]", "hexacarbonylmolybdenum"),
        ("[W].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]", "hexacarbonyltungsten"),
    ])
    def test_carbonyl_wins(self, smiles, expected):
        assert name_compound(smiles, style="pin") == expected


class TestDinuclearHydrideWins:
    """10c: Group-14 substituent on senior Group-15 parent (P-69.5.3 / P-41)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("[SiH3][AsH2]", "silylarsane"),
        ("[SiH3][SbH2]", "silylstibane"),
        ("[SiH3][BiH2]", "silylbismuthane"),
        ("[GeH3][AsH2]", "germylarsane"),
        ("[GeH3][SbH2]", "germylstibane"),
        ("[GeH3][BiH2]", "germylbismuthane"),
        ("[SnH3][AsH2]", "stannylarsane"),
        ("[SnH3][SbH2]", "stannylstibane"),
        ("[SnH3][BiH2]", "stannylbismuthane"),
        ("[PbH3][AsH2]", "plumbylarsane"),
        ("[PbH3][SbH2]", "plumbylstibane"),
        ("[PbH3][BiH2]", "plumbylbismuthane"),
    ])
    def test_direct_namer(self, smiles, expected):
        assert _dinuc(smiles) == expected

    def test_routes_through_dispatch(self):
        """The DINUCLEAR_HYDRIDE@48 dispatch entry reaches the namer end-to-end."""
        assert name_compound("[GeH3][SbH2]", style="pin") == "germylstibane"
        assert name_compound("[PbH3][BiH2]", style="pin") == "plumbylbismuthane"

    def test_parent_is_always_the_group15_atom(self):
        """P-41: Group-15 outranks Group-14, so it is always the parent hydride."""
        # germyl (Ge, Group 14) on stibane (Sb, Group 15) — never the reverse.
        assert _dinuc("[GeH3][SbH2]") == "germylstibane"
        assert _dinuc("[SbH2][GeH3]") == "germylstibane"  # SMILES order independent


class TestDinuclearHydrideFailClosed:
    """10c is fail-closed: every non-target topology declines (None)."""

    @pytest.mark.parametrize("smiles,why", [
        ("[SiH3][SiH3]", "homo-Group-14 (disilane)"),
        ("[GeH3][GeH3]", "homo-Group-14 (digermane)"),
        ("[SbH2][BiH2]", "homo-Group-15"),
        ("[AsH2][AsH2]", "homo-Group-15 (diarsane)"),
        ("C[SiH2][SbH2]", "carbon present"),
        ("F[GeH2][SbH2]", "halide present"),
        ("[GeH3][PH2]", "P parent excluded (stays with name_phosphine)"),
        ("[GeH3][SbH2][GeH3]", "three hubs"),
        ("C[Si]([Si](C)(C)C)(C)C", "hexamethyldisilane (substituted)"),
        ("[GeH3][SnH3]", "two Group-14 atoms (no senior Group-15 parent)"),
    ])
    def test_declines(self, smiles, why):
        assert _dinuc(smiles) is None, f"should decline: {why}"
