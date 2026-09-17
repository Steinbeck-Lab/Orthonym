"""A hydroxy on a fused heteroaromatic parent stays a PREFIX when a SENIOR group is
present seniority guard for the ``-ol`` promotion).

The sibling fix ``test_p41_fused_heteroaryl_ol_suffix.py`` promotes a hydroxy that is
the SOLE principal characteristic group on a fused heteroaromatic parent to the
``-ol`` SUFFIX (``quinolin-8-ol``). Its gate was ``not suffix_groups`` -- but the
fused substituent collector never routes a directly-attached carbamoyl / cyano /
acyl group into ``suffix_groups`` (they arrive as ``c_substituents`` PREFIX strings
via the general substituent-namer). So the gate OVER-FIRED: it promoted hydroxy to
``-ol`` even when a group SENIOR to hydroxy was present, asserting the WRONG principal
characteristic group::

    NC(=O)c1ccc(O)c2ncccc12 -> 5-carbamoylquinolin-8-ol (WRONG principal group)
                             => 8-hydroxyquinoline-5-carboxamide (PIN)

RULE
----
 (the Blue Book "Table 4.1 General compound classes listed in
decreasing order of seniority") ranks, in DECREASING seniority:

    11 Amides (the Blue Book) -- carboxamide
    14 Nitriles (the Blue Book) -- carbonitrile
    15 Aldehydes (the Blue Book)
    16 Ketones (the Blue Book)
    17 Hydroxy compounds (the Blue Book) -- alcohols and phenols

Classes 11-16 are ALL senior to hydroxy (class 17). selects as the
principal characteristic group the SENIOR-most group; only it is expressed as a
suffix. So when any group senior to hydroxy is present, the hydroxy is NOT the
principal group and must stay a ``hydroxy-`` prefix -- the ``-ol`` suffix is
reserved for the carbamoyl / cyano / acyl group's parent, or the name degrades to
the all-prefix form. The guard reuses the shared seniority table via
``get_principal_group(mol, detect_functional_groups(mol))``: promote only when the
molecule's principal characteristic group is itself an alcohol-class group.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


class TestFusedHydroxySeniorityGuard:

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", [
        # a carboxamide (class 11) is senior to hydroxy (class 17)
        "NC(=O)c1ccc(O)c2ncccc12",
        "NC(=O)c1cc(O)c2ccccc2n1",
        "NC(=O)c1ccc2[nH]ccc2c1O",
        # a carbonitrile (class 14) is senior to hydroxy (class 17)
        "Oc1c(C#N)ccc2cccnc12",
    ])
    def test_senior_group_keeps_hydroxy_a_prefix(self, smiles):
        """A group senior to hydroxy is present -> the emitted name must NOT assert
        hydroxy as the principal group (no ``-ol`` suffix); hydroxy stays a prefix."""
        name = name_compound(smiles)
        assert name is not None, smiles
        assert not name.endswith("ol"), name
        assert "hydroxy" in name, name

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # A2's wins -- hydroxy is the SOLE PCG, still promotes to the -ol suffix.
        # Verbatim Blue Book PINs (:3942,:3250).
        ("Oc1cccc2cccnc12", "quinolin-8-ol"),
        ("Oc1ccc2cccc3c2c1C=CC3", "1H-phenalen-4-ol"),
    ])
    def test_sole_pcg_hydroxy_still_promotes(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", [
        "NC(=O)c1ccc(O)c2ncccc12",
        "Oc1c(C#N)ccc2cccnc12",
        "NC(=O)c1cc(O)c2ccccc2n1",
        "NC(=O)c1ccc2[nH]ccc2c1O",
    ])
    def test_declined_names_roundtrip(self, smiles, opsin_to_smiles):
        """0-wrong: the degraded (hydroxy-prefix) name must still parse back to the
        input structure."""
        name = name_compound(smiles)
        parsed = opsin_to_smiles(name)
        assert parsed is not None, f"OPSIN could not parse {name!r}"
        assert Chem.CanonSmiles(parsed) == Chem.CanonSmiles(smiles)
