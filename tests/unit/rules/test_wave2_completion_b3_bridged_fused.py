"""Wave-2 completion batch B3 — mancude bridged-fused constructor (P-25.4).

The new `_try_mancude_bridged` path names bridged-fused systems whose residual
keeps its full aromatic system and whose bridgeheads stay sp2 (0 H) — so no
hydro prefix: 1,4-epoxynaphthalene / 1,4-ethanonaphthalene /
9,10-ethanoanthracene / 1,4-ethano-5,8-methanoanthracene (all BB verbatim,
all OPSIN-RT verified).

Two supporting root fixes are covered here too:
  * identify_polycyclic exact fused-component coverage veto (replaces the 0.6
    ratio that let an 11-ring-atom system claim a 10-atom naphthalene core,
    dropping the O bridge);
  * is_polycyclic_system's all-aromatic skip exempts a genuine chalcogen
    bridge that RDKit's extended aromaticity hides
    (has_aromatic_chalcogen_bridge).
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.bridged_fused import (
    has_aromatic_chalcogen_bridge,
    name_bridged_fused_pin,
)


def _pin(smiles):
    res = name_bridged_fused_pin(Chem.MolFromSmiles(smiles))
    return res[0] if res else None


@pytest.mark.unit
class TestMancudeBridgedConstructor:
    def test_ethano_methano_anthracene_bb_verbatim(self):
        # Two bridges on distinct terminal rings; ethano cited first
        # (alphanumerical) and takes 1,4 (P-25.4.3.3(b)).
        assert (_pin("C12=CC=C(C3=CC=4C5=CC=C(C4C=C13)C5)CC2")
                == "1,4-ethano-5,8-methanoanthracene")

    def test_epoxynaphthalene_bb_verbatim(self):
        assert _pin("C12=CC=C(C3=CC=CC=C13)O2") == "1,4-epoxynaphthalene"

    def test_ethanonaphthalene_bb_verbatim(self):
        assert _pin("C12=CC=C(C3=CC=CC=C13)CC2") == "1,4-ethanonaphthalene"

    def test_methanonaphthalene(self):
        assert _pin("C12=CC=C(C3=CC=CC=C13)C2") == "1,4-methanonaphthalene"

    def test_meso_ethanoanthracene_bb_verbatim(self):
        assert (_pin("C1=CC=CC2=C3C4=CC=CC=C4C(=C12)CC3")
                == "9,10-ethanoanthracene")

    def test_dihydro_path_hydro_first_order(self):
        # sp3 bridgeheads keep the dihydro/tetrahydro path; Wave-2 completion C
        # fixed the citation order to hydro-BEFORE-bridge (P-31.1.4.2.4 --
        # hydro sits between detachable and nondetachable prefixes).
        assert (_pin("C1CC2CCC1c1ccccc21")
                == "1,2,3,4-tetrahydro-1,4-ethanonaphthalene")

    def test_peri_fused_declines(self):
        # acenaphthene: peri attachment (bridgeheads in different rings) is
        # fusion territory, not a bridge — the constructor must decline.
        assert _pin("C1Cc2cccc3cccc1c23") is None

    def test_ortho_fused_declines(self):
        # fluorene: adjacent attachment = ortho-fusion.
        assert _pin("c1ccc2c(c1)Cc1ccccc12") is None


@pytest.mark.unit
class TestChalcogenBridgePredicate:
    def test_epoxy_bridge_detected(self):
        assert has_aromatic_chalcogen_bridge(
            Chem.MolFromSmiles("C12=CC=C(C3=CC=CC=C13)O2")) is True

    @pytest.mark.parametrize("smiles", [
        "c1ccc2c(c1)oc1ccccc12",   # dibenzofuran — fusion O
        "c1ccc2c(c1)sc1ccccc12",   # dibenzothiophene — fusion S
        "c1ccoc1",                  # furan — monocyclic
        "c1ccc2ccccc2c1",           # naphthalene — no chalcogen
    ])
    def test_fusion_chalcogens_not_flagged(self, smiles):
        assert has_aromatic_chalcogen_bridge(
            Chem.MolFromSmiles(smiles)) is False


@pytest.mark.unit
class TestProductionEndToEnd:
    @pytest.mark.parametrize("smiles,expected", [
        ("C12=CC=C(C3=CC=4C5=CC=C(C4C=C13)C5)CC2",
         "1,4-ethano-5,8-methanoanthracene"),
        ("C12=CC=C(C3=CC=CC=C13)O2", "1,4-epoxynaphthalene"),
        ("C12=CC=C(C3=CC=CC=C13)CC2", "1,4-ethanonaphthalene"),
        ("C1=CC=CC2=C3C4=CC=CC=C4C(=C12)CC3", "9,10-ethanoanthracene"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1ccc2ccccc2c1", "2-methylnaphthalene"),
        ("c1ccc(-c2cccc3ccccc23)cc1", "1-phenylnaphthalene"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("C1Cc2cccc3cccc1c23", "acenaphthene"),
        ("c1ccc2c(c1)Cc1ccccc12", "fluorene"),
        ("c1ccc2c(c1)oc1ccccc12", "dibenzofuran"),
        ("c1ccc2c(c1)sc1ccccc12", "dibenzothiophene"),
    ])
    def test_pah_and_heterocycle_protections(self, smiles, expected):
        # The coverage-veto + aromatic-skip changes must not disturb the
        # established PAH / fused-heterocycle names.
        assert name_compound(smiles) == expected
