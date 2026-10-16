"""SF lane E, task 1: the bare ``-O-P=O`` prefix is ``(oxophosphanyl)oxy``, not ``(phosphoryl)oxy``.

``phosphoryl`` is ``-PO<`` with THREE free valences "Fundamental acyl groups for
substituents derived from mononuclear noncarbon oxoacids", the Blue Book; the list at
:18069 '-PO< phosphoryl (preselected prefix)'). In ``-O-P=O`` the phosphorus has one bond to the
parent through O and one =O, so ``[(phosphoryl)oxy]`` leaves two valences unfilled and the name
reads as a radical only (OPSIN reads it back only with ``-r``).

The group is named by "Substitutive names for groups derived from mononuclear
noncarbon oxoacids" (:36239; the list at:36297-36299): '-P=O oxophosphanyl (preselected prefix)'
(Appendix 2,:56495 'oxophosphanyl*...; '). A preselected prefix builds PINs
of organic derivatives: "PRESELECTED NAMES" (:2060), "noncarbon-containing characteristic
groups, prefixes, and suffixes used in PINs are designated as preselected prefixes or suffixes"
(:2064). So a carbon-containing parent that cites ``(oxophosphanyl)oxy`` keeps the PIN label.

Ligand counts of the producer ``name_phosphoanhydride_oxy_substituent`` (P bonded to the linking
O and one =O): two single-bond oxygen ligands -> 'phosphoryl' with its branches (unchanged);
none -> 'oxophosphanyl'; one (a P-H, '-O-PH(=O)OR') -> declined.

A ring-ketone parent that the PIN heterocycle collector used to decline on this group was named by
the older ring assembler from its retained table ('acridone', '2,3-dihydro-1H-isoindole-1,3-dione');
the PINs are 'acridin-9(10H)-one' and '1H-isoindole-1,3(2H)-dione' imides, the Blue Book:
33853 '2-phenyl-1H-isoindole-1,3(2H)-dione (PIN)'), which the collector now reaches.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.rules.phosphorus import (
    bare_oxophosphanyl_oxy_prefix,
    name_phosphoanhydride_oxy_substituent as name_pa,
)
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _o_and_parent(smi):
    """``(mol, bridging O idx, parent atom idx)`` for the first X-O-P linkage."""
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetSymbol() == 'O':
            nbrs = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(nbrs) == 2 and sorted(n.GetSymbol() for n in nbrs)[-1] == 'P':
                parent = [n for n in nbrs if n.GetSymbol() != 'P'][0]
                return m, a.GetIdx(), parent.GetIdx()
    raise AssertionError("no X-O-P linkage found")


# ---- the producer: ligand counts --------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "CCCCOP=O", "c1ccccc1OP=O", "COC(=O)C(CC1CCCCC1)COP=O", "OCCOP=O",
])
def test_bare_p_double_o_is_oxophosphanyl(smiles):
    m, o, c = _o_and_parent(smiles)
    assert name_pa(m, o, c) == '(oxophosphanyl)oxy'


@pytest.mark.parametrize("smiles,expected", [
    ("OCCOP(=O)(OC)OC", '(dimethoxyphosphoryl)oxy'),                 # two branches: unchanged
    ("OCCOP(=O)(OCC)OCC", '(diethoxyphosphoryl)oxy'),
    ("OCCOP(=O)(O)O", 'phosphonooxy'),                               # terminal phosphate
    ("OCCOP(=O)(O)OP(=O)(O)O", '[hydroxy(phosphonooxy)phosphoryl]oxy'),
])
def test_two_branches_keep_the_phosphoryl_forms(smiles, expected):
    m, o, c = _o_and_parent(smiles)
    assert name_pa(m, o, c) == expected


@pytest.mark.parametrize("smiles", [
    "CCOP(=O)O",            # -O-PH(=O)OH: one branch and a P-H; no prefix here spells it
    "CCOP(=O)OC",
])
def test_one_branch_p_h_is_declined(smiles):
    m, o, c = _o_and_parent(smiles)
    assert name_pa(m, o, c) is None


def test_bare_group_nested_in_a_bridge_is_oxophosphanyl_oxy_branch():
    m, o, c = _o_and_parent("OCCOP(=O)(O)OP=O")
    out = name_pa(m, o, c)
    assert out is not None and '(oxophosphanyl)oxy' in out and 'phosphoryl' in out, out


def _ring_collector_hook(smiles):
    """The hook ``heterocycles._identify_hetero_substituent`` ends with, on the ring substituent
    that starts at the O bonded to P (its atoms: that O, the P and the P's other ligands)."""
    m = Chem.MolFromSmiles(smiles)
    o = [a for a in m.GetAtoms() if a.GetSymbol() == 'O'
         and any(n.GetSymbol() == 'P' for n in a.GetNeighbors())
         and any(n.IsInRing() for n in a.GetNeighbors())][0]
    p = [n for n in o.GetNeighbors() if n.GetSymbol() == 'P'][0]
    ring = {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()}
    sub = [o.GetIdx(), p.GetIdx()] + [n.GetIdx() for n in p.GetNeighbors() if n.GetIdx() != o.GetIdx()]
    return bare_oxophosphanyl_oxy_prefix(m, sub, ring)


@pytest.mark.parametrize("smiles", ["c1ccccc1OP=O", "c1ccc2c(c1)C(=O)N(C2=O)OP=O"])
def test_ring_collector_hook_returns_the_bare_form(smiles):
    assert _ring_collector_hook(smiles) == '(oxophosphanyl)oxy'


@pytest.mark.parametrize("smiles", ["c1ccccc1OP(=O)(O)O", "c1ccccc1OP(=O)(OC)OC", "c1ccccc1OP(=O)O"])
def test_ring_collector_hook_declines_every_other_o_p_group(smiles):
    """An O-P ester of an oxoacid is senior to a ketone, so the ring parent must not absorb it."""
    assert _ring_collector_hook(smiles) is None


# ---- whole names: independent non-radical read-back ------------------------------------

def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _reads_back_closed_shell(name, smiles):
    """OPSIN WITHOUT ``-r`` (``opsin_parse``) reads ``name`` to the input's full InChIKey."""
    back = opsin_parse(name)
    return back is not None and inchi.MolToInchiKey(Chem.MolFromSmiles(back)) == _key(smiles)


_DEFAULT = Orthonym()
_BEST = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)

PIN_ROWS = [
    # PubChem 19878094: shipped as '(phosphoryl)oxy' pin_verified on main
    ("COC(=O)C(CC1CCCCC1)COP=O", "methyl 2-(cyclohexylmethyl)-3-[(oxophosphanyl)oxy]propanoate"),
    # PubChem 66724406 (an aryl ester-free carbon parent)
    ("CC(=CC(=O)O)c1ccc(OP=O)cc1", "3-{4-[(oxophosphanyl)oxy]phenyl}but-2-enoic acid"),
    # a ring ketone: the PIN heterocycle collector names it, not the retained table
    ("C1COCCN1C2=CC3=C(C(=C2)OP=O)C(=O)C4=CC=CC=C4N3",
     "3-(morpholin-4-yl)-1-[(oxophosphanyl)oxy]acridin-9(10H)-one"),
    ("C1=CC=C2C(=C1)C(=O)N(C2=O)OP=O", "2-[(oxophosphanyl)oxy]-1H-isoindole-1,3(2H)-dione"),
]


@pytest.mark.parametrize("smiles,name", PIN_ROWS)
def test_default_tier_ships_the_oxophosphanyl_pin_and_it_reads_back_without_r(smiles, name):
    d = _DEFAULT.name_tiered(smiles)
    assert (d.get("name"), d.get("tier")) == (name, "pin_verified"), d
    assert _reads_back_closed_shell(name, smiles), name


@pytest.mark.parametrize("smiles", [s for s, _ in PIN_ROWS] + ["CCCC(CC)CCCCOP=O", "OP=O"])
def test_best_effort_name_reads_back_without_r_and_never_names_phosphoryl_oxy(smiles):
    b = _BEST.name_tiered(smiles)
    assert b.get("name") and "(phosphoryl)oxy" not in b["name"], b
    assert _reads_back_closed_shell(b["name"], smiles), b


def test_phosphoryl_oxy_with_two_branches_is_still_pin():
    smiles = "COC(=O)c1ccc(OP(=O)(OC)OC)cc1"
    assert _DEFAULT.name_tiered(smiles).get("name") == "methyl 4-[(dimethoxyphosphoryl)oxy]benzoate"
