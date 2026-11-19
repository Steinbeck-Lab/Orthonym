"""Ions and radicals of homogeneous parent-hydride chains (ChEBI C1, 2026-11-19).

 "Ionic centers in the same parent structure" (the Blue Book) cites the ionic suffixes in the order
'ium', 'ylium', 'ide', 'uide'; "Radical ions derived from parent hydrides" (:43430) cites the radical suffixes
after them ('ethan-2-id-1-yl', 'boranuidyl', 'methyliumyl'). Two 'ylium' suffixes are 'bis(ylium)',
:7114). Every name is carbon-free, so it ships at the best-effort tier as systematic_verified and the PIN tier
declines it."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.hydride_ions import name_hydride_ion
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

CASES = [
    ("[S-]S[S-]", "trisulfane-1,3-diide"),
    ("[S]S[S-]", "trisulfan-3-id-1-yl"),
    ("[S][S-]", "disulfan-2-id-1-yl"),
    ("[NH-][NH-]", "hydrazine-1,2-diide"),
    ("[H][N-][N-2]", "hydrazine-1,1,2-triide"),
    ("[N-]=N", "diazenide"),             # (b): one centre on two atoms, no locant
    ("[H][N+2]", "azanebis(ylium)"),
    ("[H][B-2]", "boranediide"),
    ("[H][Al+]([H])[H]", "alumaniumyl"),
    ("[H][F+]", "fluoraniumyl"),
    ("[H][O+2]([H])([H])[H]", "oxidanediium"),
    ("[H][Si+]([H])[Si]([H])([H])[H]", "disilanylium"),
    ("[S-]S[SH2-]", "trisulfan-3-id-1-uide"),   #: anions first, then 'uide'
    ("[SH2+]S[S+]", "trisulfan-3-ium-1-ylium"),  #: cations first, then 'ylium'
    ("[SiH2+2]", "silanebis(ylium)"),            # 'ane' is cut only before one 'yl' or 'ylium'
    ("[OH+]", "oxidanylium"),                    #: lone pairs, no radical
    ("[H][S]([H])[H]", "λ4-sulfanyl"),
    ("[H][Bi-]([H])[H]", "bismuthanuidyl"),      # bismuth: PARENT_HYDRIDE_IONS (data/metal_scope.py, 2026-11-19)
    ("[H][Bi+]([H])([H])[H]", "bismuthanium"),
]


@pytest.mark.parametrize("smiles,name", CASES)
def test_hydride_ion_names(smiles, name):
    assert name_hydride_ion(Chem.MolFromSmiles(smiles)) == name


@pytest.mark.parametrize("smiles,name", CASES)
def test_hydride_ion_names_ship_at_best_effort_only(smiles, name):
    with jvm_slots(1, purpose="hydride-ions"):
        be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
        pin = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
        assert (be["name"], be["tier"]) == (name, "systematic_verified")
        assert pin.get("tier") == "abstain"
        assert name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("smiles", [
    "[NH]",              # two radical centres on one atom: 'azanylidene' is the radical producer's
    "[SH]S([SH])S",      # a branch
    "S1SS1",             # a ring
    "[S-]O[S-]",         # two elements
    "C[S-]",             # carbon
    "[Li][H]",           # not a parent-hydride element
    "N",                 # a neutral hydride: no ionic or radical centre
    "[Cl]",              # a lone atom: rules/metal_atoms.lone_atom_name
    "[H][Po][H]",        # polonium is not admitted to PARENT_HYDRIDE_IONS (data/metal_scope.py)
])
def test_out_of_class_declines(smiles):
    assert name_hydride_ion(Chem.MolFromSmiles(smiles)) is None
