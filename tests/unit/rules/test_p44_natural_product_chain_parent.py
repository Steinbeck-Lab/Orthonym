"""DT4 F2: a chain that carries the principal characteristic group is the parent, also when the ring system is a
steroid or alkaloid skeleton (`internal notes`).

Governing text (the Blue Book): ``### **** SENIORITY ORDER FOR PARENT STRUCTURES``, ``****`` (:18875)
"The senior parent structure has the maximum number of substituents corresponding to the principal characteristic
group (suffix)...". A stereoparent keeps a side-chain suffix only because the side chain is part of the
stereoparent, and its names come from the natural-product producer; parent selection used to make every
natural-product ring system the parent (a shortcut ahead of the pool), which overruled and left the
systematic route with a ring parent it could not name.

``### **** ALPHANUMERICAL ORDER`` (:3446) leaves Greek letters out of alphanumerical order: the 'α' of a
steroid locant ('3α,7α-dihydroxy') is not a letter of the order.
"""
import pytest

from orthonym import Orthonym, name_compound
from orthonym.assembly.naming_utils import alpha_sort_key
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.p44_scorer import select_parent_unified
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit


def _pin(smiles):
    with jvm_slots(1, purpose="test-p44-np-chain"):
        return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    # the steroid skeleton (cyclopenta[a]phenanthrene) as a substituent of the acid chain
    ("OC(=O)CCC1CCC2C1CCC1C2CCC2CCCCC12", "3-(hexadecahydro-1H-cyclopenta[a]phenanthren-17-yl)propanoic acid"),
    ("CC(CCC(=O)O)C1CCC2C1CCC1C2CCC2CCCCC12", "4-(hexadecahydro-1H-cyclopenta[a]phenanthren-17-yl)pentanoic acid"),
    # the same split without the natural-product skeleton was named before
    ("OC(=O)CCC1CCC2C1CCC1CCCCC12", "3-(dodecahydro-1H-cyclopenta[a]naphthalen-3-yl)propanoic acid"),
])
def test_the_acid_chain_is_the_parent_of_a_steroid_skeleton(smiles, name):
    row = _pin(smiles)
    # the label of these names is the producer's (pin_unverified for the cyclopenta[a]naphthalene one on main too)
    assert row["name"] == name and row["tier"] in ("pin_verified", "pin_unverified")
    assert name_is_rt_exact(name, smiles)


def test_parent_selection_follows_p44_1_1_for_a_natural_product_skeleton():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("OC(=O)CCC1CCC2C1CCC1C2CCC2CCCCC12")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain = [a.GetIdx() for a in mol.GetAtoms() if not a.IsInRing() and a.GetSymbol() == "C"]
    acid = [tuple(m) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[OH]"))]
    sel = select_parent_unified(mol, [ring], chain, "carboxylic_acid", acid)
    assert sel.parent_type == "chain"


_DIOXOCHOLANOATE = "COC(=O)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC(=O)[C@@H]4CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C"


def test_a_failed_chain_assembly_leaves_no_locant_map_for_the_stereo_check(monkeypatch):
    # The chain route names nothing here and the decomposition fallback names the ester; the chain's map put a
    # '(4R)-' (the chain's C-4, which the stereoparent name implies) in front of it. The chain route fails only
    # without the decorated fused '-yl' prefix (DT4 F2b), so that prefix is withheld here.
    from orthonym.assembly import composer
    monkeypatch.setattr(composer, "_fused_decorated_prefix", lambda *a, **k: None)
    row = _pin(_DIOXOCHOLANOATE)
    assert row["name"] == "methyl 3,6-dioxo-5β-cholan-24-oate"
    assert name_is_rt_exact(row["name"], _DIOXOCHOLANOATE)


def test_the_chain_route_names_the_ester_with_a_decorated_steroid_prefix():
    # DT4 F2b: the steroid ring system with its oxo groups is the '-yl' prefix of the ester chain, its
    # descriptors cited at the front of the prefix, the Blue Book).
    row = _pin(_DIOXOCHOLANOATE)
    assert row["name"] == ("methyl (4R)-4-[(5R,8S,9S,10R,13R,14S,17R)-10,13-dimethyl-3,6-dioxohexadecahydro-"
                           "1H-cyclopenta[a]phenanthren-17-yl]pentanoate")
    assert name_is_rt_exact(row["name"], _DIOXOCHOLANOATE)


def test_greek_letters_of_a_locant_are_not_letters_of_the_order():
    assert alpha_sort_key("(3α,7α-dihydroxy-5β-cholan-24-amido)") < alpha_sort_key("sulfo")
    assert alpha_sort_key("(3,7-dihydroxycholan-24-amido)") < alpha_sort_key("sulfo")


def test_an_inner_lambda_is_not_a_letter_of_the_order():
    # 'dimethyl-λ4-sulfanylidene': the letters are 'dimethylsulfanylidene', so it is cited before
    # 'dimethylt...' (the λ used to sort after every Roman letter)
    assert alpha_sort_key("(dimethyl-λ4-sulfanylidene)amino") < alpha_sort_key("dimethylthiocarbamoyl")


def test_the_dihydroxy_amido_prefix_is_cited_before_sulfo():
    smiles = ("C[C@H](CCC(=O)NC(CCS(=O)(=O)O)C(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)"
              "[C@H]3CC[C@]12C")
    with jvm_slots(1, purpose="test-p44-np-chain"):
        name = name_compound(smiles)
    assert name == "2-(3α,7α-dihydroxy-5β-cholan-24-amido)-4-sulfobutanoic acid"
