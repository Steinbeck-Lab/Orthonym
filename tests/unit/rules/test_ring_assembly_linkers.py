"""Ring assemblies as linkers: the polyvalent ring-assembly prefix, the central group of a multiplicative name, the linker
of an assembly of parent radicals.

``### ****`` "Substituent prefixes derived from ring assemblies" (the Blue Book),:16109: "Low locants are
assigned to ring junctions, then to free valences" ('[1,1'-biphenyl]-4-yl (preferred prefix)',:16118). A ring assembly
that holds no principal characteristic group lies in the linker of a multiplicative name, where it is the central
multiplicative substituent group,:5238): '10,10'-[[1,1'-biphenyl]-4,4'-diylbis(oxy)]di(decanoic acid)
(PIN)' (the Blue Book). An assembly that holds the principal characteristic groups is the parent
('[1,1'-biphenyl]-4,4'-dicarboxylic acid'). "ASSEMBLIES OF PARENT RADICALS" (the Blue Book) names radical
centres in identical parts "according to the principles for nomenclature of assemblies of identical units linked by
multivalent substituents (see ".
"""
import pytest
from rdkit import Chem

from orthonym.assembly.radical_assembly import radical_assembly_names
from orthonym.cli import _tiered_namer
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.ring_assemblies import (
    detect_ring_assembly,
    name_ring_assembly_polyvalent_prefix,
    name_ring_assembly_prefix,
)
from tests.support.rt_assert import name_best_effort, name_is_rt_exact


def _methyl_hosts(mol):
    """The ring atoms that carry a methyl group: the free valences of the prefix."""
    return [a.GetNeighbors()[0].GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetDegree() == 1]


PREFIXES = [
    ("Cc1ccc(cc1)-c1ccc(C)cc1", "[1,1'-biphenyl]-4,4'-diyl"),
    ("Cc1cccc(c1)-c1cccc(C)c1", "[1,1'-biphenyl]-3,3'-diyl"),
    # the lower set of free-valence locants: 3 unprimed, 4 primed (3,4' before 3',4)
    ("Cc1ccc(cc1)-c1cccc(C)c1", "[1,1'-biphenyl]-3,4'-diyl"),
    ("Cc1cccc(c1)-c1ccc(C)cc1", "[1,1'-biphenyl]-3,4'-diyl"),
    # the ring name keeps its final 'e' before 'diyl', as 'pyridine-2,6-diyl'
    ("Cc1ccc(nc1)-c1ccc(C)cn1", "[2,2'-bipyridine]-5,5'-diyl"),
    ("Cc1ccc(s1)-c1ccc(C)s1", "[2,2'-bithiophene]-5,5'-diyl"),
    ("CC1CCC(CC1)C1CCC(C)CC1", "[1,1'-bi(cyclohexane)]-4,4'-diyl"),
    ("Cc1ccc(cc1)-c1ccc(cc1)-c1ccc(C)cc1", "[1,1':4',1''-terphenyl]-4,4''-diyl"),
]


@pytest.mark.parametrize("smiles,expected", PREFIXES)
def test_polyvalent_prefix(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    info = detect_ring_assembly(mol, [set(s) for s in get_ring_systems(mol)])
    assert name_ring_assembly_polyvalent_prefix(mol, info, _methyl_hosts(mol)) == expected


def test_polyvalent_prefix_declines_two_valences_on_one_ring():
    mol = Chem.MolFromSmiles("Cc1cc(C)cc(c1)-c1ccccc1")
    info = detect_ring_assembly(mol, [set(s) for s in get_ring_systems(mol)])
    assert name_ring_assembly_polyvalent_prefix(mol, info, _methyl_hosts(mol)) is None


@pytest.mark.parametrize("smiles,expected", [
    ("Cc1ccc(cc1)-c1ccccc1", "[1,1'-biphenyl]-4-yl"),
    ("Cc1ccc(nc1)-c1ccccn1", "[2,2'-bipyridin]-5-yl"),
])
def test_monovalent_prefix_unchanged(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    info = detect_ring_assembly(mol, [set(s) for s in get_ring_systems(mol)])
    assert name_ring_assembly_prefix(mol, info, _methyl_hosts(mol)[0]) == expected


MULTIPLICATIVE = [
    # the book's own PIN (the Blue Book)
    ("OC(=O)CCCCCCCCCOc1ccc(cc1)-c1ccc(cc1)OCCCCCCCCCC(=O)O",
     "10,10'-[[1,1'-biphenyl]-4,4'-diylbis(oxy)]di(decanoic acid)"),
    ("OC(=O)CCOc1ccc(nc1)-c1ccc(OCCC(=O)O)cn1", "3,3'-[[2,2'-bipyridine]-5,5'-diylbis(oxy)]dipropanoic acid"),
    # an unsymmetrical central group is allowed,:6295)
    ("OC(=O)Cc1ccc(cc1)-c1cccc(CC(=O)O)c1", "2,2'-([1,1'-biphenyl]-3,4'-diyl)diacetic acid"),
    # the assembly holds the principal characteristic groups: it is the parent
    ("OC(=O)c1ccc(cc1)-c1ccc(cc1)C(=O)O", "[1,1'-biphenyl]-4,4'-dicarboxylic acid"),
    # rings bonded directly to a different ring are no assembly needs identical ring systems): the
    # general detector's central ring group, as for any other linker,:23178;,:23172: identical
    # parents "do not have to have a principal characteristic group")
    ("OC(=O)c1ccc(cc1)-c1cccc(n1)-c1ccc(cc1)C(=O)O", "4,4'-(pyridine-2,6-diyl)dibenzoic acid"),
    ("c1ccc(nc1)-c1ccc(cc1)-c1ccccn1", "2,2'-(1,4-phenylene)dipyridine"),
]


@pytest.mark.opsin_gate
def test_three_ring_assembly_is_no_central_pin():
    """A terphenyl central group would need the superscript locants prefers (:16134): no PIN label."""
    row = _tiered_namer("pin", "pin").name_tiered("OC(=O)CCOc1ccc(cc1)-c1ccc(cc1)-c1ccc(OCCC(=O)O)cc1")
    assert row["tier"] != "pin_verified" or "terphenyl]-4,4''-diyl" not in (row["name"] or ""), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", MULTIPLICATIVE)
def test_multiplicative_pin(smiles, expected):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row


RADICALS = [
    ("C[Si](C)c1cccc(c1)-c1cccc(c1)[Si](C)C", "([1,1'-biphenyl]-3,3'-diyl)bis(dimethylsilyl)"),
    ("C[B]c1ccc(cc1)-c1ccc([B]C)cc1", "([1,1'-biphenyl]-4,4'-diyl)bis(methylboranyl)"),
    # one-atom divalent groups between the assembly and the units are cited after it
    ("[O]Cc1ccc(cc1)-c1ccc(C[O])cc1", "[[1,1'-biphenyl]-4,4'-diylbis(methylene)]bis(oxyl)"),
]


@pytest.mark.parametrize("smiles,expected", RADICALS)
def test_radical_assembly_offer(smiles, expected):
    assert expected in list(radical_assembly_names(Chem.MolFromSmiles(smiles)))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", RADICALS)
def test_radical_assembly_ships(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res
