"""A fused or bridged ring system that holds a guanidine unit is named by fusion nomenclature.

 "Guanidine and its derivatives" (the Blue Book): "The preferred IUPAC name for
the 'amidine' related to carbonic acid, H2N-C(=NH)-NH2, is the retained name 'guanidine'", an acyclic
parent hydride. The carbon of a guanidine has three nitrogen neighbours and no other, so a ring that
runs through it holds two of those nitrogens as skeletal atoms: the unit is part of a heterocycle,
the ring system is the parent "Systems composed of rings and chains",:19336,:19340
"Within the same class, a ring or ring system has seniority over a chain"), and no guanidine stands
on it. Two producers read it as a guanidine anyway. The retained-name handler (priority 900 of
``assembly.inner_dispatch``) named the whole molecule N-substituted guanidine, a name no gate accepts
and one that kept every ring producer from running; and ``bridged_fused_pin.principal_suffix`` took
the unit for a group with no suffix and declined, as it had for the amidine of DBU
(``test_fused_ring_amidine.py``). The perception puts the unit in the class 'guanidine' when both
single-bonded nitrogens have three connections, in 'amidine' when one of them has two.

When the unit has one exocyclic nitrogen and it is a plain -NH2, that nitrogen is the amino group of an
amine: "Primary amines, R-NH2, are systematically named in the following ways" (:26168),
substitutive '-amine' ('1-benzofuran-2-amine (PIN)',:26193; 'quinolin-4-amine (PIN)',:26199), with
the 'amino' prefix when a senior group is present. A unit whose exocyclic nitrogen is the =NH of
an imine is named by the suffix 'imine',:26510 "All imines are named substitutively using
the suffix 'imine'"; the prefix 'imino' only in the presence of "characteristic groups having seniority
over imines",:26540): this builder spells no 'imine' suffix and declines it.
"""
import pytest
from rdkit import Chem

from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import (
    get_principal_group,
    guanidine_parent_matches,
    ring_skeleton_units_resolved,
)
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.unit

TBD = "C1CNC2=NCCCN2C1"                      # 1,5,7-triazabicyclo[4.4.0]dec-5-ene
AMINO_HEXAHYDROBENZIMIDAZOLE = "NC1=NC2CCCCC2N1"
AMINO_DIHYDROQUINOLINE = "NC1=Nc2ccccc2CC1"   # an amidine of the ring carbon, the =N in the ring


def _mol_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, detect_functional_groups(mol)


def _ring_atoms(mol):
    return {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}


@pytest.mark.parametrize("smiles", [TBD, "NC1=NCCN1", AMINO_HEXAHYDROBENZIMIDAZOLE, "NC1=NCc2ccccc2N1"])
def test_a_guanidine_whose_carbon_is_a_ring_atom_is_not_a_parent_of_the_guanidine_handler(smiles):
    """The red side: the perception reports the class 'guanidine' and the handler took it; the carbon
    of a ring guanidine is a ring atom, so no match is left for the retained parent."""
    mol, groups = _mol_groups(smiles)
    assert groups.get("guanidine"), smiles
    assert guanidine_parent_matches(mol, groups["guanidine"]) == []


@pytest.mark.parametrize("smiles", ["NC(=N)N", "CN=C(N)NC", "N[C@@H](CCCNC(=N)N)C(=O)O", "NC(=N)Nc1ccccn1"])
def test_an_acyclic_guanidine_stays_a_parent_of_the_guanidine_handler(smiles):
    mol, groups = _mol_groups(smiles)
    matches = groups["guanidine"]
    assert guanidine_parent_matches(mol, matches) == matches


def test_a_ring_guanidine_beside_an_acyclic_one_leaves_only_the_acyclic_match():
    mol, groups = _mol_groups("NC(=N)NCCCC1CNC2=NCCCN2C1")
    assert len(groups["guanidine"]) == 2
    kept = guanidine_parent_matches(mol, groups["guanidine"])
    assert len(kept) == 1 and not mol.GetAtomWithIdx(kept[0][1]).IsInRing()


def test_a_ring_guanidine_with_all_three_nitrogens_in_the_ring_is_set_aside():
    """TBD: no group is left, so the ring system is the parent (nothing senior hides behind it)."""
    mol, groups = _mol_groups(TBD)
    assert "guanidine" in groups
    rest = ring_skeleton_units_resolved(mol, groups, _ring_atoms(mol))
    assert not rest.get("guanidine")
    assert get_principal_group(mol, rest) == (None, [])


@pytest.mark.parametrize("smiles", [AMINO_HEXAHYDROBENZIMIDAZOLE, "NC1=NCc2c(ccc(Cl)c2Cl)N1"])
def test_the_exocyclic_nh2_of_a_ring_guanidine_is_an_amine(smiles):
    """Two ring nitrogens and a plain -NH2: the unit is replaced by a 'primary_amine' match (N, C)."""
    mol, groups = _mol_groups(smiles)
    assert get_principal_group(mol, groups)[0] is None
    rest = ring_skeleton_units_resolved(mol, groups, _ring_atoms(mol))
    assert not rest.get("guanidine")
    pg, matches = get_principal_group(mol, rest)
    assert pg == "primary_amine" and len(matches) == 1, (pg, matches)
    n_idx, c_idx = matches[0]
    assert mol.GetAtomWithIdx(n_idx).GetTotalNumHs() == 2 and mol.GetAtomWithIdx(c_idx).IsInRing()


def test_the_exocyclic_nh2_of_a_ring_amidine_is_an_amine():
    """The amidine class: the ring carbon, the =N in the ring and an exocyclic -NH2 (the pterin shape)."""
    mol, groups = _mol_groups(AMINO_DIHYDROQUINOLINE)
    assert get_principal_group(mol, groups)[0] == "amidine"
    rest = ring_skeleton_units_resolved(mol, groups, _ring_atoms(mol))
    assert not rest.get("amidine")
    assert get_principal_group(mol, rest)[0] == "primary_amine"


@pytest.mark.parametrize("smiles", [
    "N=C1NC2CCCCC2N1",            # the exocyclic nitrogen is the =NH of an imine
    "CNC1=NC2CCCCC2N1",           # a substituted amino
    "CC(=O)NC1=NC2CCCCC2N1",      # an acylated amino
])
def test_a_ring_guanidine_with_another_exocyclic_part_is_kept(smiles):
    """Not a plain amine: the dict is returned unchanged (no 'primary_amine' is invented)."""
    mol, groups = _mol_groups(smiles)
    rest = ring_skeleton_units_resolved(mol, groups, _ring_atoms(mol))
    assert rest.get("guanidine") == groups.get("guanidine")
    assert not rest.get("primary_amine")


def test_a_carboximidamide_on_a_ring_is_not_a_ring_unit():
    """Benzamidine: the amidine carbon is not an atom of the ring:34200)."""
    mol, groups = _mol_groups("NC(=N)c1ccccc1")
    benzene = set(range(2, 8))
    assert ring_skeleton_units_resolved(mol, groups, benzene) is groups


# the fusion names (hydro prefixes:16872, indicated hydrogen, numbering
#:3219) of ring guanidines whose three nitrogens, or two of them and an -NH2, are in the ring system
RING_GUANIDINES = [
    (TBD, "1,3,4,6,7,8-hexahydro-2H-pyrimido[1,2-a]pyrimidine"),
    ("CN1CCCN2CCCN=C12", "1-methyl-1,3,4,6,7,8-hexahydro-2H-pyrimido[1,2-a]pyrimidine"),
    ("C1CNC2=[NH+]CCCN2C1", "1,3,4,6,7,8-hexahydro-2H-pyrimido[1,2-a]pyrimidin-9-ium"),
    (AMINO_HEXAHYDROBENZIMIDAZOLE, "3a,4,5,6,7,7a-hexahydro-1H-1,3-benzimidazol-2-amine"),
    ("NC1=NC2CCCCC2N1C", "1-methyl-3a,4,5,6,7,7a-hexahydro-1H-1,3-benzimidazol-2-amine"),
    ("NC1=Nc2ccccc2CC1", "3,4-dihydroquinolin-2-amine"),
    ("NC1=NCc2c(ccc(Cl)c2Cl)N1", "5,6-dichloro-1,4-dihydroquinazolin-2-amine"),
]


@pytest.mark.opsin_gate
def test_the_builder_names_a_ring_guanidine_and_declines_an_imine_of_the_ring():
    jar_or_skip()
    from orthonym.rules.bridged_fused_pin import build_fused
    got = build_fused(Chem.MolFromSmiles(TBD))
    assert got is not None and got[0] == RING_GUANIDINES[0][1], got
    got = build_fused(Chem.MolFromSmiles("NC1=NCc2c(ccc(Cl)c2Cl)N1"))
    assert got is not None and got[0] == RING_GUANIDINES[6][1], got
    for smiles in ("N=C1NC2CCCCC2N1", "CNC1=NC2CCCCC2N1", "N=C1CCc2ccccc2N1"):
        assert build_fused(Chem.MolFromSmiles(smiles)) is None, smiles


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [r for r in RING_GUANIDINES if r[0] != "NC1=NCc2c(ccc(Cl)c2Cl)N1"])
def test_the_default_tier_names_the_ring_guanidine_and_the_name_reads_back(smiles, name):
    """pin_verified, and OPSIN reads the name back to the input's full InChIKey."""
    jar_or_skip()
    from orthonym import Orthonym
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified", row
    assert_full_rt(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [(TBD, RING_GUANIDINES[0][1]), (AMINO_HEXAHYDROBENZIMIDAZOLE, RING_GUANIDINES[3][1])])
def test_the_best_effort_tier_gives_the_same_name(smiles, name):
    jar_or_skip()
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name and row["tier"] == "pin_verified", row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("NC(=N)N", "guanidine"),
    ("CN=C(N)NC", "N,N'-dimethylguanidine"),
    ("CC(C)N=C(N)NC", "N-methyl-N'-(propan-2-yl)guanidine"),
])
def test_an_acyclic_guanidine_keeps_the_retained_name(smiles, name):
    """The handler still names the guanidines that are parents:34242)."""
    jar_or_skip()
    from orthonym import Orthonym
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified", row
