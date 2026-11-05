"""A chain substituent that holds an oxo, a hydroxy, a methylidene or a substituted imino group is named at the PIN tier
(lane/l17 item 3).

 (the R-oxy prefixes): the prefix of an R-oxy group is the name of R with 'oxy' for the free valence, so a
decorated R keeps its prefixes ('bis(2-hydroxyethoxy)acetic acid (PIN)', the Blue Book). The alkoxy namer refused any
R whose name held 'hydroxy' or 'oxo' (a guard for the spurious prefixes that an oxygen inside R makes), so every -O-R with a
terminal -OH or =O on R abstained at the PIN tier.

 "THE PRINCIPAL SUBSTITUENT CHAIN" (the Blue Book), criterion (d): the principal chain has the greater number of
multiple bonds, after (b) the longest chain. A C=C that no longest chain carries is a prefix of its off-chain atom with the
free valence of its bond: '2-methylidene-4-oxopentyl'. The branched unsaturated producer took only chains that
carried every C=C, so a chain with an exo-methylene and an oxo group declined.

 "Oximes" (heading:38300; the last sentence of its paragraph,:38460): "In the presence of a characteristic
group having priority for citation as a suffix, in substitutive nomenclature oximes are designated by the prefix
'hydroxyimino'" ('3-(hydroxyimino)butanal (PIN)',:38487; '4-[(ethoxyimino)methyl]benzene-1-sulfonic acid (PIN)',:38485;
'3-(methylimino)butan-2-ylidene (PIN)',:43352). The chain producer took =NH only, so a =N-OH, =N-OR or =N-R on a carbon
of the chain declined it.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.chain_decoration_prefixes import alkoxy_decoration_tokens_are_real
from orthonym.cli import _tiered_namer
from orthonym.errors import is_failure_name
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


def _abstains(row):
    return row["tier"] == "abstain" or not row["name"] or is_failure_name(row["name"])


# the target rows: an R-oxy or R prefix whose chain holds an exo-methylene, an oxo, a hydroxy or a substituted imino group
_TARGETS = [
    ("OC(=O)c1ccc(OCC(=C)CC(C)=O)cc1", "4-[(2-methylidene-4-oxopentyl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=NO)CC)cc1", "4-[2-(hydroxyimino)butoxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=NC)CC)cc1", "4-[2-(methylimino)butoxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=NOC)CC)cc1", "4-[2-(methoxyimino)butoxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=NN)CC)cc1", "4-(2-hydrazinylidenebutoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(=O)CC)cc1", "4-(2-oxobutoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(=O)C)cc1", "4-(2-oxopropoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC=O)cc1", "4-(2-oxoethoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCCO)cc1", "4-(2-hydroxyethoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(O)CC)cc1", "4-(2-hydroxybutoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(O)CO)cc1", "4-(2,3-dihydroxypropoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(=C)CO)cc1", "4-{[2-(hydroxymethyl)prop-2-en-1-yl]oxy}benzoic acid"),
    ("OC(=O)c1ccc(OCC(=CC)CC(C)=O)cc1", "4-[(2-ethylidene-4-oxopentyl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=C)C(=C)C(C)=O)cc1", "4-[(2,3-dimethylidene-4-oxopentyl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=C)C=CC(C)=O)cc1", "4-[(2-methylidene-5-oxohex-3-en-1-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=NO)C(=O)C)cc1", "4-[2-(hydroxyimino)-3-oxobutoxy]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=C)CC(=NO)C)cc1", "4-{[4-(hydroxyimino)-2-methylidenepentyl]oxy}benzoic acid"),
    ("NC(=O)c1ccc(OCC(=O)CC)cc1", "4-(2-oxobutoxy)benzamide"),
    ("OC(=O)COCC(=O)CC", "(2-oxobutoxy)acetic acid"),
    # the chain attached to the ring, not through an oxygen
    ("OC(=O)c1ccc(CC(=C)CC(C)=O)cc1", "4-(2-methylidene-4-oxopentyl)benzoic acid"),
    ("OC(=O)c1ccc(CCC(=C)CC(C)=O)cc1", "4-(3-methylidene-5-oxohexyl)benzoic acid"),
    ("OC(=O)c1ccc(CC(=NO)CC)cc1", "4-[2-(hydroxyimino)butyl]benzoic acid"),
    ("OC(=O)c1ccc(CC(=NC)CC)cc1", "4-[2-(methylimino)butyl]benzoic acid"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _TARGETS)
def test_a_decorated_chain_substituent_is_named_at_the_pin_tier(smiles, name, tier):
    """ (d) / / / (module docstring): the chain keeps its prefixes."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the rows that were named before: the chalcogen analogue of the oxo row, the retained exo-methylene and the oxo/ketone
# chains that are the parent, and the thioether and amine links
_CONTROLS = [
    ("OC(=O)c1ccc(OCC(=S)CC)cc1", "4-(2-sulfanylidenebutoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(=C)CC)cc1", "4-(2-methylidenebutoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(=C)C)cc1", "4-[(2-methylprop-2-en-1-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(CC(=O)CC)cc1", "4-(2-oxobutyl)benzoic acid"),
    ("OC(=O)c1ccc(SCC(=O)CC)cc1", "4-[(2-oxobutyl)sulfanyl]benzoic acid"),
    ("OC(=O)c1ccc(NCC(=O)CC)cc1", "4-[(2-oxobutyl)amino]benzoic acid"),
    ("OC(=O)c1ccc(OCC(=N)CC)cc1", "4-(2-iminobutoxy)benzoic acid"),
    ("OC(=O)CC(=C)CC(C)=O", "3-methylidene-5-oxohexanoic acid"),
    ("OC(=O)CC(=NO)CC", "3-(hydroxyimino)pentanoic acid"),
    # the group is the principal one: the chain that holds it is the parent, the ring a prefix
    ("OCCOc1ccccc1", "2-phenoxyethan-1-ol"),
    ("CCC(=O)COc1ccccc1", "1-phenoxybutan-2-one"),
    ("CC(=O)COc1ccccc1", "1-phenoxypropan-2-one"),
    ("CC(=O)c1ccc(OCC(=O)CC)cc1", "1-(4-acetylphenoxy)butan-2-one"),
    ("CCC(=O)COc1ccc(N)cc1", "1-(4-aminophenoxy)butan-2-one"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _CONTROLS)
def test_the_controls_keep_their_names(smiles, name, tier):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


# a C=N or C=C with a defined geometry has no locant to be cited at in a prefix of this producer, so the PIN tier declines
# it (never a name without the geometry); the best-effort tier cites the geometry. The wider tier's name is not the PIN (its
# parent is spelled 'benzene-1-carboxylic acid' or 'carboxy...benzene'), so the test pins what makes it correct, not its
# spelling: the full isomeric round trip (the E descriptor in the name is read back to the input's geometry) and the descriptor.
_GEOMETRY = [
    "OC(=O)c1ccc(OC/C(=N/O)CC)cc1",
    "OC(=O)c1ccc(OC/C(=C/C)CC(C)=O)cc1",
]


@pytest.mark.parametrize("smiles", _GEOMETRY)
def test_a_prefix_with_a_defined_geometry_is_not_named_without_it(smiles):
    assert _abstains(_row(smiles, "pin"))
    row = _row(smiles, "best-effort")
    assert row["name"], row
    assert "2E" in row["name"], row
    assert_full_rt(row["name"], smiles)


# the same compound prefix twice needs 'bis', which the simple multiplier does not spell: no PIN is shipped for it
def test_a_repeated_compound_prefix_is_not_named_at_the_pin_tier():
    smiles = "OC(=O)c1ccc(OCC(=NO)C(=NO)C)cc1"
    assert _abstains(_row(smiles, "pin"))
    row = _row(smiles, "best-effort")
    assert row["name"], row
    assert_full_rt(row["name"], smiles)


# an oxygen inside the R of an R-oxy group (an ether link) is what the 'hydroxy'/'oxo' guard stays closed for: the recursion
# cannot yet spell the prefixes of such an R, so the PIN tier abstains and the wider tier names the molecule
@pytest.mark.parametrize("smiles", ["OC(=O)c1ccc(OCCOCCO)cc1", "OC(=O)c1ccc(OCCOCC(C)=O)cc1"])
def test_an_ether_inside_the_chain_is_not_named_at_the_pin_tier_by_its_decoration(smiles):
    assert _abstains(_row(smiles, "pin"))
    row = _row(smiles, "best-effort")
    assert row["name"] and row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)


# The labels of the names that the chain-substituent namer opened at the best-effort tier and the PIN tier does not ship. The
# names are in preferred-name form (for the first, "Substitutive nomenclature, prefix mode" (2), the Blue Book:
# "a combination of the simple prefixes 'hydroxy' and 'imino' at the end of a carbon chain is used in preferred IUPAC names"),
# which a producer outside the strict path built, so they are 'pin_unverified' (not 'pin_verified', not 'systematic_verified')
# and the PIN-only tier abstains. The test states those labels; it does not claim the PIN-only tier should ship them.
_LABELLED = [
    ("OC(=O)c1ccc(OCC(O)=N)cc1", "4-(2-hydroxy-2-iminoethoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCC(=O)S)cc1", "4-(2-oxo-2-sulfanylethoxy)benzoic acid"),
    ("OC(=O)c1ccc(OCNCC=O)cc1", "4-{[(2-oxoethyl)amino]methoxy}benzoic acid"),
    ("OC(=O)c1ccc(OCC(=N[N+](=O)[O-])CC)cc1", "4-[2-(nitroimino)butoxy]benzoic acid"),
]


@pytest.mark.parametrize("smiles,name", _LABELLED)
def test_the_label_of_a_name_the_pin_tier_does_not_ship_is_stated(smiles, name):
    assert _abstains(_row(smiles, "pin"))
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "pin_unverified"), row
    assert_full_rt(row["name"], smiles)


def _rest(smiles, attach_o_index):
    """The atoms on the carbon side of the ether oxygen ``attach_o_index`` of ``smiles``."""
    mol = Chem.MolFromSmiles(smiles)
    oxygen = mol.GetAtomWithIdx(attach_o_index)
    seen = {oxygen.GetIdx()}
    stack = [n.GetIdx() for n in oxygen.GetNeighbors() if n.GetSymbol() == "C" and not n.GetIsAromatic()]
    attach = stack[0]
    rest = []
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        rest.append(cur)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors() if n.GetIdx() not in seen)
    return mol, rest, attach


@pytest.mark.parametrize(
    "smiles,oxygen,expected",
    [
        ("OC(=O)c1ccc(OCC(=O)CC)cc1", 7, True),  # a ketone on R, the acid in the parent
        ("OC(=O)c1ccc(OCCO)cc1", 7, True),  # an alcohol on R, the acid in the parent
        ("OC(=O)c1ccc(OCCOCCO)cc1", 7, False),  # an oxygen inside R
        ("Oc1ccc(OCCO)cc1", 5, False),  # the alcohol of R is of the principal class: the suffix of the parent
        ("CCC(=O)COc1ccccc1", 5, False),  # the ketone of R is the principal group
        ("OC(=O)c1ccc(OC(=O)CC(C)C(=O)Cl)cc1", 7, False),  # an acyloxy: the free-valence carbon is a carbonyl carbon
    ],
)
def test_the_decoration_tokens_of_an_alkoxy_are_real_only_for_terminal_groups_that_are_not_principal(
        smiles, oxygen, expected):
    """ /: the principal characteristic group is the suffix of the parent, never a prefix of an R-oxy group."""
    mol, rest, attach = _rest(smiles, oxygen)
    assert mol.GetAtomWithIdx(oxygen).GetSymbol() == "O"
    assert alkoxy_decoration_tokens_are_real(mol, rest, attach) is expected


def _plan(smiles, attach):
    """``unsaturated_chain_plan`` of the substituent ``smiles`` whose free valence is on atom ``attach``."""
    from orthonym.assembly.chain_decoration_prefixes import unsaturated_chain_plan
    from orthonym.assembly.substituent_naming import _longest_carbon_chains_from

    mol = Chem.MolFromSmiles(smiles)
    backbone = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"}
    double = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
              for b in mol.GetBonds()
              if b.GetBondType() == Chem.BondType.DOUBLE and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C"}}
    return unsaturated_chain_plan(mol, _longest_carbon_chains_from(mol, backbone, attach), attach, double)


def test_no_longest_chain_carries_the_c_c_so_the_substituent_is_named_as_a_saturated_one():
    """ (d): 2-methylidene-4-oxopentyl -- the chain of five has no C=C, the methylidene is a prefix."""
    assert _plan("CC(=O)CC(=C)C", 6) == []


def test_a_chain_that_carries_a_c_c_keeps_the_off_chain_one_as_a_prefix():
    """ (d): 2-methylidene-5-oxohex-3-en-1-yl -- the ene is on the principal chain, the methylidene is a prefix."""
    plan = _plan("CC(=O)C=CC(=C)C", 7)
    assert plan is not None and len(plan) == 1
    chain, on_chain = plan[0]
    assert chain == [7, 5, 4, 3, 1, 0] and on_chain == {frozenset((3, 4))}


def test_a_c_c_inside_a_branch_beside_a_chain_c_c_is_not_planned():
    """The producer builds no prefix for a C=C inside a branch while the chain carries another: it fails closed (never
    the saturated finish, which would drop the 'ene')."""
    assert _plan("CC(C)(C=C)CC=C", 0) is None
