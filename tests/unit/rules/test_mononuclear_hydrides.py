"""Unit tests for the mononuclear parent-hydride namer (a phase + 7).

 / /. Two substituent regimes:
  * all-halogen hub on S/Se/Te/P/As/Sb/Bi/I -> ``<halo>[-lambda<n>-]<stem>``
    (a phase added the λ forms; a phase dropped the "λ required" gate so
    standard-valence PCl3/SF2/AsCl3 name too);
  * organyl / bare Group-15 As/Sb/Bi hub -> ``<organyl><stem>`` (a phase).

Fail-closed: every guard narrows; oxoacids, oxo-halides, interhalogens,
di-nuclear species, branched/cyclic/unsaturated organyls, carbon-substituted
P/chalcogens, ions and ring-member hubs all decline (None) and cascade onward.
"""
import pytest
from rdkit import Chem

from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride


def _name(smiles):
    return name_mononuclear_hydride(Chem.MolFromSmiles(smiles))


class TestLambdaWins:
    @pytest.mark.parametrize("smiles,expected", [
        ("FS(F)(F)(F)(F)F", "hexafluoro-λ6-sulfane"),    # SF6
        ("FS(F)(F)F", "tetrafluoro-λ4-sulfane"),          # SF4
        ("FP(F)(F)(F)F", "pentafluoro-λ5-phosphane"),     # PF5
        ("ClP(Cl)(Cl)(Cl)Cl", "pentachloro-λ5-phosphane"),  # PCl5
        ("FI(F)(F)(F)F", "pentafluoro-λ5-iodane"),        # IF5
        ("FI(F)F", "trifluoro-λ3-iodane"),                # IF3
        ("F[Se](F)(F)(F)(F)F", "hexafluoro-λ6-selane"),   # SeF6
    ])
    def test_known_lambda_hydrides(self, smiles, expected):
        assert _name(smiles) == expected

    def test_mixed_halogens_alphabetical(self):
        # SF5Cl: chloro (c) before fluoro (f); multiplier ignored for ordering.
        assert _name("FS(F)(F)(F)(F)Cl") == "chloropentafluoro-λ6-sulfane"


class TestStandardValenceHalides:
    """Phase 7: dropped the λ-required gate -> standard-valence all-halogen
    element hydrides name with NO λ (every one round-trips through OPSIN)."""
    @pytest.mark.parametrize("smiles,expected", [
        ("ClP(Cl)Cl", "trichlorophosphane"),    # PCl3
        ("FP(F)F", "trifluorophosphane"),        # PF3
        ("FSF", "difluorosulfane"),              # SF2
        ("Cl[As](Cl)Cl", "trichloroarsane"),     # AsCl3
        ("F[As](F)F", "trifluoroarsane"),        # AsF3
        ("Br[Sb](Br)Br", "tribromostibane"),     # SbBr3
        ("Cl[Bi](Cl)Cl", "trichlorobismuthane"),  # BiCl3
    ])
    def test_standard_valence_halides(self, smiles, expected):
        assert _name(smiles) == expected


class TestGroup15OrganylAndBare:
    """a phase: organyl / bare Group-15 As/Sb/Bi parent hydrides."""
    @pytest.mark.parametrize("smiles,expected", [
        ("C[As](C)C", "trimethylarsane"),
        ("C[Sb](C)C", "trimethylstibane"),
        ("C[Bi](C)C", "trimethylbismuthane"),
        ("CC[As](CC)CC", "triethylarsane"),
        ("c1ccccc1[As](c1ccccc1)c1ccccc1", "triphenylarsane"),
        ("[AsH3]", "arsane"),
        ("[SbH3]", "stibane"),
        ("[BiH3]", "bismuthane"),
    ])
    def test_group15_organyl_and_bare(self, smiles, expected):
        assert _name(smiles) == expected


class TestGroup14Halides:
    """Phase 8: Group-14 Si/Ge tetrahalides via the all-halogen regime
    (P-68.2.1.1 / P-67.1.2.5.2). Standard valence -> no λ. Every one OPSIN-RT."""
    @pytest.mark.parametrize("smiles,expected", [
        ("F[Si](F)(F)F", "tetrafluorosilane"),     # SiF4
        ("Cl[Si](Cl)(Cl)Cl", "tetrachlorosilane"),  # SiCl4
        ("Br[Si](Br)(Br)Br", "tetrabromosilane"),   # SiBr4
        ("F[Ge](F)(F)F", "tetrafluorogermane"),     # GeF4
        ("Cl[Ge](Cl)(Cl)Cl", "tetrachlorogermane"),  # GeCl4
        # Wave-2 completion: Sn/Pb complete the Group-14 column (OPSIN-RT ok)
        ("Cl[Sn](Cl)(Cl)Cl", "tetrachlorostannane"),  # SnCl4
        ("Cl[Pb](Cl)(Cl)Cl", "tetrachloroplumbane"),  # PbCl4
    ])
    def test_group14_tetrahalides(self, smiles, expected):
        assert _name(smiles) == expected

    def test_group14_mixed_organyl_halide_names(self):
        """Wave-3: mixed organyl+halide Group-14 hub now NAMES.

        C[Si](F)(F)F -> trifluoro(methyl)silane (OPSIN-RT ok). Previously this
        was expected to fail-closed; the element-hydride namer now builds the
        mixed organyl+halide substituent set directly.
        """
        assert _name("C[Si](F)(F)F") == "trifluoro(methyl)silane"

    @pytest.mark.parametrize("smiles,why", [
        ("C[Si](C)(C)C", "tetramethylsilane: organyl Si -> P-69 namer owns it"),
        ("CO[Si](OC)(OC)OC", "tetramethoxysilane: alkoxy ligands, not halides"),
        ("c1cc[siH]c1", "silole: Si ring member, not a parent hydride"),
    ])
    def test_group14_declines(self, smiles, why):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip(f"RDKit rejects SMILES ({why})")
        assert name_mononuclear_hydride(mol) is None, why


class TestFailClosedDecline:
    @pytest.mark.parametrize("smiles,why", [
        ("O=S(=O)(O)O", "sulfuric acid: O neighbours are non-halogen"),
        ("O=P(O)(O)O", "phosphoric acid"),
        ("O=S(Cl)Cl", "thionyl chloride SOCl2: has an O"),
        ("C[As](C)(=O)O", "cacodylic acid: =O/OH neighbours are not organyl"),
        ("ClC(Cl)(Cl)Cl", "CCl4: carbon hub, not in set"),
        ("FC(F)(F)F", "CF4: carbon hub"),
        ("ClCl", "Cl2 diatomic: no degree>=2 hub"),
        ("ClI", "ICl interhalogen: no degree>=2 hub"),
        ("FS(F)(F)(F)(F)S(F)(F)(F)(F)F", "S2F10: two hubs (di-nuclear)"),
        ("CP(C)C", "trimethylphosphane: organyl P -> name_phosphine owns it"),
        ("CSC", "dimethyl sulfide: organyl chalcogen is not in the organyl regime"),
        ("c1cc[as]c1", "arsole: As is a ring member, not a parent hydride"),
    ])
    def test_declines(self, smiles, why):
        mol = Chem.MolFromSmiles(smiles)
        # Some hypervalent halides RDKit refuses outright; treat as N/A.
        if mol is None:
            pytest.skip(f"RDKit rejects SMILES ({why})")
        assert name_mononuclear_hydride(mol) is None, why

    @pytest.mark.parametrize("smiles,expected,fabricated", [
        ("CC(C)[As](C(C)C)C(C)C",            "tri(propan-2-yl)arsane",
         "tripropylarsane"),
        ("C1CCCCC1[As](C1CCCCC1)C1CCCCC1",   "tricyclohexylarsane",
         "trihexylarsane"),
        ("c1ccccc1C[As](Cc1ccccc1)Cc1ccccc1", "tribenzylarsane",
         "triheptylarsane"),
        ("C=C[As](C=C)C=C",                  "triethenylarsane",
         "triethylarsane"),
    ])
    def test_branched_cyclic_and_unsaturated_organyls_are_now_NAMED(
        self, smiles, expected, fabricated,
    ):
        """These four were `test_declines` rows until a phase.

        Their stated reasons -- "branched/internal attachment", "cyclic alkyl",
        "benzyl mislabel risk", "unsaturated alkyl" -- were artefacts of the
        private carbon-skeleton walker behind the old organyl guard, which
        miscounted a ring or branch as a linear chain (cyclohexyl -> 'hexyl',
        benzyl -> 'heptyl', propan-2-yl -> 'propyl'). Refusing was the correct
        response to THAT walker; the four refusals were never nomenclature.

        The guard now routes to the audited shared chokepoint
        (`rules.substituent_purity.organyl_prefix_name` ->
        `assembly.substituent_enumerator.name_substituent`), which spells all four
        classes correctly, so the class is NAMED instead of declined. The
        underlying safety property is unchanged and is asserted positively here:
        no fabricated linear chain may appear. All four names are OPSIN-exact
        against the input structure.
        """
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        name = name_mononuclear_hydride(mol)
        assert name == expected
        # the exact WRONG-CONSTITUTION name the old walker would have fabricated
        assert name != fabricated

    def test_charged_declines(self):
        assert _name("F[S+](F)(F)F") is None

    def test_none_input(self):
        assert name_mononuclear_hydride(None) is None
