"""-1a: neutral single-atom co-components (methane / hydrogen sulfide /
phosphane) name as adduct partners instead of forcing the whole
multi-component row to abstain.

Root cause (FABLE full-census): `SINGLE_ATOM_COMPONENT_NAMES` held only
O/F/Cl/Br/I, so a bare C/S/P fragment made `_name_component` return None and
`name_adduct` decline the entire row -- a table-miss-degrades-to-refusal
defect (CLAUDE.md a project rule). CH4/H2S/PH3 are neutral molecular species and
legitimate adduct components; each name is OPSIN-2.9.0-parseable in
em-dash adduct notation ("benzene—methane (1/1)").

Bare metals and bare N/ammonia stay excluded (organometallic routing / a bare
nitrogen is more often a perception artefact) -- see adducts.py.
"""
import pytest

pytestmark = pytest.mark.unit


class TestSingleAtomComponentTable:
    def test_nonmetal_hydrides_present(self):
        from orthonym.rules.adducts import SINGLE_ATOM_COMPONENT_NAMES as T
        assert T["C"] == "methane"
        assert T["S"] == "hydrogen sulfide"
        assert T["P"] == "phosphane"

    def test_metals_and_ammonia_stay_excluded(self):
        from orthonym.rules.adducts import SINGLE_ATOM_COMPONENT_NAMES as T
        assert "N" not in T        # ammonia deliberately excluded
        assert "Na" not in T and "Fe" not in T  # metals: organometallic routing

    def test_name_component_methane(self):
        from orthonym.rules.adducts import _name_component
        assert _name_component("C", "pin") == "methane"


class TestMethaneAdductNames:
    """End-to-end: the whole row names (was abstain) and the name is exact."""

    @pytest.mark.parametrize("smiles, expected", [
        ("C.c1ccccc1", "benzene—methane (1/1)"),
        ("S.c1ccccc1", "benzene—hydrogen sulfide (1/1)"),
        ("P.c1ccccc1", "benzene—phosphane (1/1)"),
        ("C.C1=CC=C(C=C1)C2=CC=C(C=C2)N",
         "[1,1'-biphenyl]-4-amine—methane (1/1)"),
    ])
    def test_adduct_name(self, smiles, expected):
        from orthonym import Orthonym
        namer = Orthonym(general_fallback=True,
                          general_fallback_unverified=True,
                          allow_aromatic_general=True)
        res = namer.name_tiered(smiles)
        assert res.get("name") == expected

    @pytest.mark.opsin_gate
    def test_three_component_methane_adduct_round_trips(self):
        # 3-component (1/1/1): methane + two organics; the reclaimed witness
        # from the PubChem-1M abstention census.
        from orthonym import Orthonym
        from orthonym.validation.opsin_roundtrip import opsin_parse
        from rdkit import Chem
        smi = "C.CC1=C2C=CCC2=CC=C1.CC1=C2C(=CC=C1)OCCO2"
        namer = Orthonym(general_fallback=True,
                          general_fallback_unverified=True,
                          allow_aromatic_general=True)
        name = namer.name_tiered(smi).get("name")
        assert name is not None and "methane" in name and "(1/1/1)" in name
        osmi = opsin_parse(name)
        assert osmi is not None
        assert (Chem.MolToInchiKey(Chem.MolFromSmiles(osmi))
                == Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
