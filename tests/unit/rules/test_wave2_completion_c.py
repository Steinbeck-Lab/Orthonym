"""Wave-2 completion pass C — FAIL_CLOSED/NA research-pass builds.

All expected names OPSIN-RT verified. Covers:
  * 13 BB-cited multi-component fusion parents (catalog rows, P-25.3.4-.8).
  * Bridged-fused extensions: anthracene/acridine residuals, etheno bridge,
    hydro-before-bridge citation order (P-25.4.x / P-31.1.4.2.4).
  * Multiplicative composites: carbonothioyl + ring-N units, methylenebis(oxy),
    oxybis(azanylylidenemethanylylidene) (P-64.6.2 / P-15.3.1.2.2.x).
  * New namers @dispatch 48.3-48.6: azinic derivatives, heterones, sulfines,
    acyl-on-Si/Ge/P/As pseudoketones (P-61.5.3 / P-64.x).
  * Additives: carbonyl dicyanide retained row, diazo/imine collision fix,
    chalcogenylidene substituent prefix (P-66.5.3.1 / P-61.4 / P-64.7.3).
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestFusionCatalogParents:
    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc2nc3cc4nc5c(nc4cc3nc2c1)nc1ccccn15",
         "pyrido[1'',2'':1',2']imidazo[4',5':5,6]pyrazino[2,3-b]phenazine"),
        ("c1cc2ccc3cc4cc5ccc1c5c4c23",
         "cyclopenta[ij]pentaleno[2,1,6-cde]azulene"),
        ("c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12", "dibenzo[c,g]phenanthrene"),
        ("c1cc2c3ccc4cncc5ccc(c6ccc7cncc1c7c26)c3c45",
         "anthra[2,1,9-def:6,5,10-d'e'f']diisoquinoline"),
        ("C1=Cc2c(c3ccccc3c3ncoc23)C1",
         "8H-cyclopenta[3,4]naphtho[1,2-d][1,3]oxazole"),
        ("C1=CN2C=CC3=CNOC3=C2O1",
         "2H-[1,2]oxazolo[5,4-c][1,3]oxazolo[3,2-a]pyridine"),
        ("c1cc2cc3csnc3cc2s1", "thieno[3,2-f][2,1]benzothiazole"),
        ("c1cc2c(o1)oc1cc3cocc3nc12",
         "furo[3,4-b]furo[3',2':4,5]furo[2,3-e]pyridine"),
        ("C1=CC2=c3ccncc3=NC2=C1", "cyclopenta[4,5]pyrrolo[2,3-c]pyridine"),
        ("c1cc2ccc3ccnc4ccc(c1)c2c34", "naphtho[2,1,8-def]quinoline"),
    ])
    def test_catalog_heals(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestBridgedFusedExtensions:
    @pytest.mark.parametrize("smiles,expected", [
        ("C1CC2c3ccccc3C1c1ccccc21", "9,10-dihydro-9,10-ethanoanthracene"),
        ("C1=CC2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethenoanthracene"),
        ("c1ccc2c(c1)C1CCN2c2ccccc21", "9H-9,10-ethanoacridine"),
        ("C1=CC2CCC1c1cc3ccccc3cc12", "1,4-dihydro-1,4-ethanoanthracene"),
        ("C1CC2CCC1c1ccccc21", "1,2,3,4-tetrahydro-1,4-ethanonaphthalene"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # mancude path untouched
        ("C1=CC=CC2=C3C4=CC=CC=C4C(=C12)CC3", "9,10-ethanoanthracene"),
        ("C12=CC=C(C3=CC=CC=C13)O2", "1,4-epoxynaphthalene"),
        ("C1Cc2cccc3cccc1c23", "acenaphthene"),
        ("c1ccc2c(c1)Cc1ccccc12", "fluorene"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestMultiplicativeComposites:
    def test_carbonothioyl_ring_n_units(self):
        assert (name_compound("O=c1ccccn1C(=S)n1ccccc1=O")
                == "1,1'-carbonothioyldi(pyridin-2(1H)-one)")

    def test_methylenebis_oxy(self):
        assert (name_compound("Oc1ccc(OCOc2ccc(O)cc2)cc1")
                == "4,4'-[methylenebis(oxy)]diphenol")

    def test_oxybis_azanylylidenemethanylylidene(self):
        assert (name_compound("C(=NON=Cc1ccccc1)c1ccccc1")
                == "1,1'-[oxybis(azanylylidenemethanylylidene)]dibenzene")

    @pytest.mark.parametrize("smiles,expected", [
        # v24 W8-P1 R6: PIN is diphenylmethanone (BB 28326 verbatim
        # "benzophenone diphenylmethanone (PIN) (not 1,1′-carbonyldibenzene)").
        # The multiplicative-split protection still holds — the diaryl ketone must
        # NOT become 1,1'-carbonyldibenzene; only the retained→systematic form changed.
        ("O=C(c1ccccc1)c1ccccc1", "diphenylmethanone"),
        ("OC(=O)COCCOCC(=O)O",
         "2,2'-[ethane-1,2-diylbis(oxy)]diacetic acid"),
        ("c1ccccc1Oc1ccccc1", "1,1'-oxydibenzene"),
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestNewParentNamers:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(=O)[SiH3]", "1-silylethan-1-one"),
        ("CC(=O)[Si](C)(C)C", "1-(trimethylsilyl)ethan-1-one"),
        ("[PH2]C(=O)CCC", "1-phosphanylbutan-1-one"),
        ("C[Si](C)=O", "dimethylsilanone"),
        ("C[SiH]=O", "methylsilanone"),
        ("CCC=S=O", "propylidene-λ4-sulfanone"),
        ("CC=[N+]([O-])O", "ethylideneazinic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CS(C)=O", "(methanesulfinyl)methane"),   # sulfoxide untouched
        ("CC[N+](=O)[O-]", "nitroethane"),          # nitro tautomer untouched
        ("O=S1CCCC1", "thiolane 1-oxide"),          # ring S-oxide untouched
        ("CC(C)=O", "propan-2-one"),                # plain ketone untouched
    ])
    def test_protections(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestAdditives:
    def test_carbonyl_dicyanide(self):
        assert name_compound("O=C(C#N)C#N") == "carbonyl dicyanide"

    def test_ethyl_diazoacetate(self):
        # Engine acetate-policy form of BB 'ethyl diazoacetate' (RT-verified).
        assert (name_compound("[N-]=[N+]=CC(=O)OCC")
                == "ethyl 2-diazoethanoate")

    def test_diazomethane(self):
        assert name_compound("C=[N+]=[N-]") == "diazomethane"

    def test_sulfanylidene_prefix(self):
        # Engine paren form (pre-existing enclosure trait shared with =O twin).
        assert (name_compound("CCC(=S)CC1CCCC(CC(=O)CC)C1")
                == "1-(3-(2-sulfanylidenebutyl)cyclohexyl)butan-2-one")

    def test_oxo_twin_unchanged(self):
        assert (name_compound("CCC(=O)CC1CCCC(CC(=O)CC)C1")
                == "1-(3-(2-oxobutyl)cyclohexyl)butan-2-one")

    def test_thioketone_parent_unchanged(self):
        assert name_compound("CCC(=S)C") == "butane-2-thione"
