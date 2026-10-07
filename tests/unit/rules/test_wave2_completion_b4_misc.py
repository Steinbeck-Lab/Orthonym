"""Wave-2 completion batch B4 — investigation-driven fixes across subsystems.

Every expected name OPSIN-RT verified against its evidence SMILES. Rows:
  * ring_chalcogen_oxide.py (NEW, dispatch 49.5) — / additive
    oxide on the intact ring parent (dibenzothiophene 5-oxide/5,5-dioxide).
  * PREFIX_FORMS sulfinimidamide → 'S-aminosulfinimidoyl'.
  * NEW peroxy_acid + imidic_acid FG classes /.
  * benzenesulfinohydrazonohydrazide, benzene + chain wiring).
  * selenazolo fusion coverage (f)).
  * 1,6-dihydropyrrolo[2,3-b]pyrrole catalog row, corrected
    PIN — the bare mancude name is a different molecule).
  * fluorene cata-fused skip + 9H indicated-H /.
  * didehydrobenzene, benzyne).
  * internal-charge zwitterion mask, azido+nitro coexistence).
  * ring-assembly acid suffix, [1,1'-biphenyl]-4,4'-dicarboxylic).
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound


@pytest.mark.unit
class TestRingChalcogenOxide:
    @pytest.mark.parametrize("smiles,expected", [
        ("O=S1c2ccccc2-c2ccccc21", "dibenzo[b,d]thiophene 5-oxide"),
        ("O=S1(=O)c2ccccc2-c2ccccc21", "dibenzo[b,d]thiophene 5,5-dioxide"),
        ("O=S1CCCC1", "thiolane 1-oxide"),
        ("O=S1(=O)CCCC1", "thiolane 1,1-dioxide"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CS(C)=O", "(methanesulfinyl)methane"),      # acyclic sulfoxide
        ("CS(C)(=O)=O", "(methanesulfonyl)methane"),  # acyclic sulfone
        ("c1ccc2c(c1)sc1ccccc12", "dibenzo[b,d]thiophene"),  # bare parent (PIN descriptor)
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_substituted_ring_fails_closed(self):
        from orthonym.rules.ring_chalcogen_oxide import (
            name_ring_chalcogen_oxide,
        )
        # A substituent on the ring is outside the bare-ring scope.
        assert name_ring_chalcogen_oxide(
            Chem.MolFromSmiles("Cc1ccc2c(c1)S(=O)c1ccccc12")) is None


@pytest.mark.unit
class TestPrefixAndFrnClasses:
    def test_s_aminosulfinimidoyl(self):
        assert (name_compound("N=S(N)CCC(=O)O")
                == "3-(S-aminosulfinimidoyl)propanoic acid")

    def test_sulfinimidamide_suffix_unchanged(self):
        assert name_compound("CS(=N)N") == "methanesulfinimidamide"

    def test_propaneperoxoic_acid(self):
        assert name_compound("CCC(=O)OO") == "propaneperoxoic acid"

    def test_cyclohexanecarboperoxoic_acid(self):
        # fix a performance pass (wp6-tests), change-asserted-value (was
        # 'cyclohexane-1-carboperoxoic acid'): (the Blue Book) and the
        # BB's own 'cyclohexanecarboperoxoic acid (PIN)' (:30184); a single suffix on a
        # ring of identical positions takes no locant. OPSIN 2.9.0 full-InChIKey exact.
        assert (name_compound("OOC(=O)C1CCCCC1")
                == "cyclohexanecarboperoxoic acid")

    def test_carbonoperoxoic_protected(self):
        # [#6]-guard: the inorganic exact-SMILES entry keeps ownership.
        assert name_compound("OOC(=O)O") == "carbonoperoxoic acid"

    def test_peroxol_protected(self):
        assert name_compound("COO") == "methaneperoxol"

    def test_ethanimidic_acid(self):
        assert name_compound("CC(=N)O") == "ethanimidic acid"

    def test_propanimidic_acid(self):
        assert name_compound("CCC(=N)O") == "propanimidic acid"

    @pytest.mark.parametrize("smiles,expected", [
        ("N=C(O)O", "carbonimidic acid"),
        ("N=C(N)O", "carbamimidic acid"),
        ("CC(=N)OC", "methyl ethanimidate"),
        ("CC(=N)N", "ethanimidamide"),
    ])
    def test_imidic_neighbours_protected(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_benzenesulfinohydrazonohydrazide(self):
        assert (name_compound("NN=S(NN)c1ccccc1")
                == "benzenesulfinohydrazonohydrazide")

    @pytest.mark.parametrize("smiles,expected", [
        ("NNS(=O)(=O)c1ccccc1", "benzenesulfonohydrazide"),
        ("NS(=O)(=O)c1ccccc1", "benzenesulfonamide"),
    ])
    def test_sulfonyl_siblings_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestFusedRingRows:
    def test_selenazolothiazole(self):
        # fix a performance pass (wp6-tests), change-asserted-value (was
        # 'selenazolo[5,4-d]thiazole'): 'Seniority criteria for selecting the
        # parent component' (the Blue Book) and the BB's own
        # '[1,3]selenazolo[5,4-d][1,3]thiazole (PIN) (S,N senior to Se,N)' (:12311).
        # OPSIN 2.9.0 full-InChIKey exact.
        assert (name_compound("c1nc2[se]cnc2s1")
                == "[1,3]selenazolo[5,4-d][1,3]thiazole")

    def test_thiazolothiazole(self):
        # S2c-1 (change-asserted-value, was 'thiazolo[5,4-d]thiazole'):
        # 'Heteromonocycles' (the Blue Book), the Hantzsch-Widman name 1,3-thiazole
        # with its locants in square brackets, as the sibling above
        # ('[1,3]selenazolo[5,4-d][1,3]thiazole (PIN)',:12311). OPSIN 2.9.0 full-InChIKey
        # exact.
        assert name_compound("c1nc2scnc2s1") == "[1,3]thiazolo[5,4-d][1,3]thiazole"

    def test_dihydropyrrolopyrrole(self):
        # LEDGER-STALE override: the bare mancude 'pyrrolo[2,3-b]pyrrole' is a
        # different molecule; the 2-NH compound is the 1,6-dihydro derivative.
        assert (name_compound("c1cc2cc[nH]c2[nH]1")
                == "1,6-dihydropyrrolo[2,3-b]pyrrole")

    def test_fluorene_9_carboxylic_acid(self):
        assert (name_compound("OC(=O)C1c2ccccc2-c2ccccc21")
                == "9H-fluorene-9-carboxylic acid")

    def test_9_methylfluorene(self):
        assert (name_compound("CC1c2ccccc2-c2ccccc21")
                == "9-methyl-9H-fluorene")

    def test_bare_fluorene_retained(self):
        # (the Blue Book-11400): '(9H-isomer shown; the PIN is 9H-fluorene)'
        assert name_compound("c1ccc2c(c1)Cc1ccccc12") == "9H-fluorene"

    @pytest.mark.parametrize("smiles,expected", [
        ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),  # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
        ("C12C3C4C1C5C2C3C45", "cubane"),
        ("C1Cc2cccc3cccc1c23", "acenaphthene"),
    ])
    def test_vb_cages_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestBenzeneRows:
    def test_didehydrobenzene(self):
        assert name_compound("C1=CC=CC#C1") == "1,2-didehydrobenzene"

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccccc1", "benzene"),
        ("Cc1ccccc1", "toluene"),
        ("Oc1ccccc1", "phenol"),
    ])
    def test_bare_benzene_protected(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_azido_nitro_coexistence(self):
        assert (name_compound("Fc1ccc(N=[N+]=[N-])cc1[N+](=O)[O-]")
                == "4-azido-1-fluoro-2-nitrobenzene")

    @pytest.mark.parametrize("smiles,expected", [
        ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),
        ("[O-][n+]1ccccc1", "pyridine 1-oxide"),
    ])
    def test_zwitterion_detection_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestRingAssemblyAcidSuffix:
    def test_biphenyl_dicarboxylic(self):
        assert (name_compound("OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1")
                == "[1,1'-biphenyl]-4,4'-dicarboxylic acid")

    def test_biphenyl_monocarboxylic(self):
        assert (name_compound("OC(=O)c1ccc(-c2ccccc2)cc1")
                == "[1,1'-biphenyl]-4-carboxylic acid")

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
        ("Clc1ccc(-c2ccccc2)cc1", "4-chloro-1,1'-biphenyl"),
        ("Cc1ccc(-c2ccc(C)cc2)cc1", "4,4'-dimethyl-1,1'-biphenyl"),
    ])
    def test_assembly_prefix_path_protected(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_biphenyl_diol_suffix(self):
        # OH is the PCG -> cited as the -diol suffix on the enclosed assembly
        # parent + /, never as a 'dihydroxy' prefix.
        assert (name_compound("Oc1ccc(-c2ccc(O)cc2)cc1")
                == "[1,1'-biphenyl]-4,4'-diol")

    def test_biphenyl_carbaldehyde_suffix(self):
        assert (name_compound("O=Cc1ccc(-c2ccccc2)cc1")
                == "[1,1'-biphenyl]-4-carbaldehyde")

    def test_biphenyl_carbonitrile_suffix(self):
        assert (name_compound("N#Cc1ccc(-c2ccccc2)cc1")
                == "[1,1'-biphenyl]-4-carbonitrile")

    def test_biphenyl_amine_suffix(self):
        assert (name_compound("Nc1ccc(-c2ccccc2)cc1")
                == "[1,1'-biphenyl]-4-amine")
