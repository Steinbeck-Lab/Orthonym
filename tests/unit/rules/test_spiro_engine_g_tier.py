""" (g) in the numbering of the general engine's spiro parents (LO7).

``### **** NUMBERING`` (the Blue Book): "When several structural features appear in
cyclic and acyclic compounds, low locants are assigned to them in the following decreasing order of
seniority" (:3221), and criterion "(g) lowest locants for the substituent cited first as a prefix in
the name" (:3307); '1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)' (:3318).
The prefix that is cited first is the first one in alphanumerical order, ``### ****
ALPHANUMERICAL ORDER`` (:3436): "Alphanumerical order is used to establish the order of citation of
detachable substituent prefixes... and the numbering of a chain, ring, or ring system when a choice
is possible" (:3440), and ``****``: "The name of a prefix for a substituent is considered to
begin with the first letter of its complete name" (:3477). A spiro system is numbered by
``****`` / ``****`` (:10079,:10095, "(a) low locants are allocated to
heteroatoms as a set"); when those leave the numberings of a symmetric system tied, (g) decides.

The PIN path of the spiro parents has had the tier since the numbering of the whole molecule was given
to it (``rules/spiro._select_spiro_numbering``, ``rank_full_molecule``). The general engine
(``general_engine.name_general_spiro`` -> ``vonbaeyer_universal.analyze_spiro_universal``) numbers a
ring-only submolecule, and the canonical ranks decided every tie, so the locant of the prefix cited
first followed the SMILES spelling: ``9-(benzenesulfonyl)-3-[...]`` for one spelling of a molecule
and ``3-(benzenesulfonyl)-9-[...]`` for another.
"""
import pytest
from rdkit import Chem

from orthonym.rules.spiro import get_spiro_numbering
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate  # the names below are built with the validity gate on

THIOUREA = "C(=S)NCC8CCCO8"        # ({[(oxolan-2-yl)methyl]amino})(sulfanylidene)methyl: 'o'
BENZENESULFONYL = "S(=O)(=O)c8ccccc8"  # 'b'
THIOUREA_PREFIX = "({[(oxolan-2-yl)methyl]amino}(sulfanylidene)methyl)"


def _name(smiles):
    row = tier_row(smiles, "best-effort")
    return row["name"], row["tier"]


def _dz55(a, b):
    return f"C1CN({a})CCC12CCN({b})CC2"


def test_the_reproducer_cites_benzenesulfonyl_at_locant_3():
    # LO7b: 'benzenesulfonyl' (b) is cited before the thiocarbamoyl prefix (o), so (g) gives it the
    # lower of the two tied locants, 3
    name, tier = _name("S=C(NCC1CCCO1)N1CCC2(CC1)CCN(S(=O)(=O)c1ccccc1)CC2")
    assert name == (f"3-(benzenesulfonyl)-9-{THIOUREA_PREFIX}-3,9-diazaspiro[5.5]undecane")
    assert tier == "systematic_verified"


@pytest.mark.parametrize("first,second", [(BENZENESULFONYL, THIOUREA), (THIOUREA, BENZENESULFONYL)])
def test_the_name_does_not_depend_on_which_nitrogen_the_smiles_writes_first(first, second):
    name, _ = _name(_dz55(first, second))
    assert name == f"3-(benzenesulfonyl)-9-{THIOUREA_PREFIX}-3,9-diazaspiro[5.5]undecane"


@pytest.mark.parametrize("smiles,expected", [
    # 'benzenesulfonyl' (b) before 'phenyl' (p)
    (_dz55("c8ccccc8", BENZENESULFONYL), "3-(benzenesulfonyl)-9-phenyl-3,9-diazaspiro[5.5]undecane"),
    (_dz55(BENZENESULFONYL, "c8ccccc8"), "3-(benzenesulfonyl)-9-phenyl-3,9-diazaspiro[5.5]undecane"),
    # 'ethyl' (e) before '(4-methylbenzene-1-sulfonyl)' (m)
    (_dz55("S(=O)(=O)c8ccc(C)cc8", "CC"),
     "3-ethyl-9-(4-methylbenzene-1-sulfonyl)-3,9-diazaspiro[5.5]undecane"),
    # 'benzyl' (b) before 'phenyl' (p)
    (_dz55("c8ccccc8", "Cc8ccccc8"), "3-benzyl-9-phenyl-3,9-diazaspiro[5.5]undecane"),
    # the rings of a spiro[4.4] and a spiro[3.3] system are numbered in either order
    ("C1N(c8ccccc8)CCC12CCN(S(=O)(=O)c8ccccc8)C2",
     "2-(benzenesulfonyl)-7-phenyl-2,7-diazaspiro[4.4]nonane"),
    ("C1N(c8ccccc8)CC12CN(S(=O)(=O)c8ccccc8)C2",
     "2-(benzenesulfonyl)-6-phenyl-2,6-diazaspiro[3.3]heptane"),
    # a dispiro system is numbered by the same tier ('methyl' before 'phenyl')
    ("C1N(c8ccccc8)CC12CC23CN(C)C3", "2-methyl-7-phenyl-2,7-diazadispiro[3.0.3^5.1^4]nonane"),
])
def test_the_prefix_cited_first_takes_the_lower_of_the_tied_locants(smiles, expected):
    assert _name(smiles)[0] == expected


@pytest.mark.parametrize("smiles,expected", [
    # decided before (g): the heteroatom locants (a), (b)), 3-oxa-9-aza, not 9-oxa-3-aza
    ("O=S(=O)(c8ccccc8)N1CCC2(CC1)CCOCC2", "9-(benzenesulfonyl)-3-oxa-9-azaspiro[5.5]undecane"),
    # decided before (g): the rings of a spiro[3.5] system have an order, the locants are fixed
    ("C1N(c8ccccc8)CC12CCN(S(=O)(=O)c8ccccc8)CC2",
     "7-(benzenesulfonyl)-2-phenyl-2,7-diazaspiro[3.5]nonane"),
])
def test_a_numbering_decided_by_an_earlier_criterion_is_unchanged(smiles, expected):
    assert _name(smiles)[0] == expected


@pytest.mark.parametrize("first,second", [("c3ccccc3", "S(=O)(=O)c3ccccc3"),
                                          ("S(=O)(=O)c3ccccc3", "c3ccccc3")])
def test_the_numbering_function_breaks_a_tie_by_the_names_of_the_whole_molecule(first, second):
    # the engine numbers the ring-only submolecule, which has no prefixes: it is given the whole
    # molecule and the map of the submolecule atoms into it
    from orthonym.rules.vonbaeyer_universal import _extract_spiro_submol
    mol = Chem.MolFromSmiles(_dz55(first, second))
    cage = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and not a.GetIsAromatic()}
    sub, order = _extract_spiro_submol(mol, cage)
    ring_n = [k for k in range(sub.GetNumAtoms()) if sub.GetAtomWithIdx(k).GetSymbol() == "N"]
    spiro = next(k for k in range(sub.GetNumAtoms()) if sub.GetAtomWithIdx(k).GetDegree() == 4)
    sulfonyl_n = next(k for k in ring_n
                      if any(nb.GetSymbol() == "S"
                             for nb in mol.GetAtomWithIdx(order[k]).GetNeighbors()))
    # one prefix on each ring nitrogen; the prefix cited first ('benzenesulfonyl', b before p)
    # takes locant 3 whichever nitrogen the SMILES spelling writes first
    numbering = get_spiro_numbering(
        sub, spiro, prefix_ring_atoms=ring_n, whole_molecule=(mol, order))
    assert numbering[sulfonyl_n] == 3
