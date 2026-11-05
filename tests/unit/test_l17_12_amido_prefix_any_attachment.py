"""The amido-type prefix of an acylated nitrogen on any attachment (lane/l17 item 12).

 Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference
for citation as a principal characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted
amide is named in two ways: (1) substitutively, by using a prefix formed by changing the final letter 'e' in the
complete name of the amide to 'o'... Method (1) generates preferred IUPAC names." The book's examples are
'4-formamidobenzoic acid (PIN)' (:33000), '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33040) and
'(N-cyclopropylmethanesulfonamido)acetic acid' (:33029). A urea group is the amino prefix that carries the carbamoyl
prefix, the Blue Book: "The prefixes 'ureido' and 'ureylene' are not recommended";
'2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid (PIN)',:33354).

The ring-attached acyl nitrogen had the method-(1) prefix (lane/l16 044a2cdb15); the same group on a carbon of a chain
substituent ('4-[(N-phenylacetamido)methyl]benzoic acid') was cited by the anilino or the 'acetyl(methyl)amino' form, a
method-(2) spelling that left the PIN tier without a name. The chain-substituent namer now consumes the group as one amido
prefix of its host carbon (``chain_amide_prefixes.consume_acylated_nitrogen``), the ring hook of ``rules.benzene``
serves the N-sulfonyl and urea nitrogens as well, and the amino-branch namer of an acyclic parent has the N-substituted
sulfonamido prefix.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.chain_amide_prefixes import consume_acylated_nitrogen
from orthonym.assembly.n_acyl_amido import in_peptide, parent_sulfonamide_is_senior
from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# (smiles, the PIN) -- every name read back whole by OPSIN to the input's full InChIKey, at both tiers
_TARGETS = [
    # the acyl nitrogen on a carbon of a chain substituent (the controller's probe pair)
    ("OC(=O)c1ccc(CN(c2ccccc2)C(C)=O)cc1", "4-[(N-phenylacetamido)methyl]benzoic acid"),
    ("OC(=O)c1ccc(CN(C)C(C)=O)cc1", "4-[(N-methylacetamido)methyl]benzoic acid"),
    ("OC(=O)c1ccc(CCN(C)C(C)=O)cc1", "4-[2-(N-methylacetamido)ethyl]benzoic acid"),
    ("OC(=O)c1ccc(CN(C)C(=O)c2ccccc2)cc1", "4-[(N-methylbenzamido)methyl]benzoic acid"),
    ("OC(=O)C1CCC(CN(C)C(C)=O)CC1", "4-[(N-methylacetamido)methyl]cyclohexane-1-carboxylic acid"),
    # the secondary amides of the same chain: the simple 'acetamido' is cited inside its chain prefix
    ("OC(=O)c1ccc(CNC(C)=O)cc1", "4-(acetamidomethyl)benzoic acid"),
    ("OC(=O)c1ccc(CNC(=O)c2ccccc2)cc1", "4-(benzamidomethyl)benzoic acid"),
    ("OC(=O)c1ccc(CNC(=O)CCl)cc1", "4-[(2-chloroacetamido)methyl]benzoic acid"),
    # the amide that is the principal group is the suffix, the acyl nitrogen on the chain is the prefix of the ring
    ("CC(=O)NCc1ccc(cc1)C(=O)NC", "4-(acetamidomethyl)-N-methylbenzamide"),
    # the sulfonyl nitrogen, N-substituted and not, on a chain substituent and on the ring
    ("OC(=O)c1ccc(CN(C)S(C)(=O)=O)cc1", "4-[(N-methylmethanesulfonamido)methyl]benzoic acid"),
    ("OC(=O)c1ccc(CNS(C)(=O)=O)cc1", "4-(methanesulfonamidomethyl)benzoic acid"),
    ("OC(=O)c1ccc(N(c2ccccc2)S(C)(=O)=O)cc1", "4-(N-phenylmethanesulfonamido)benzoic acid"),
    ("OC(=O)c1ccc(NS(C)(=O)=O)cc1", "4-(methanesulfonamido)benzoic acid"),
    ("OC(=O)c1ccc(NS(=O)(=O)c2ccccc2)cc1", "4-(benzenesulfonamido)benzoic acid"),
    # the urea group: the amino prefix that carries the carbamoyl prefix, never 'ureido'
    ("OC(=O)c1ccc(NC(=O)Nc2ccccc2)cc1", "4-[(phenylcarbamoyl)amino]benzoic acid"),
    ("OC(=O)c1ccc(CNC(=O)Nc2ccccc2)cc1", "4-{[(phenylcarbamoyl)amino]methyl}benzoic acid"),
    ("OC(=O)c1ccc(N(C)C(=O)NC)cc1", "4-[methyl(methylcarbamoyl)amino]benzoic acid"),
    ("OC(=O)c1ccc(NC(=O)N(C)C)cc1", "4-[(dimethylcarbamoyl)amino]benzoic acid"),
    # the N-substituted sulfonamido prefix on an acyclic acid parent: the book's own example (:33029) and its aryl analogue
    ("OC(=O)CCN(c1ccccc1)S(C)(=O)=O", "3-(N-phenylmethanesulfonamido)propanoic acid"),
    ("OC(=O)CN(C1CC1)S(C)(=O)=O", "(N-cyclopropylmethanesulfonamido)acetic acid"),  # the Blue Book, verbatim
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _TARGETS)
def test_an_acylated_nitrogen_is_the_amido_prefix_on_any_attachment(smiles, name, tier):
    """ (the Blue Book), method (1) 'generates preferred IUPAC names'."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# (smiles, the PIN) -- the neighbours of the class keep the names they had
_UNCHANGED = [
    # the ring-attached acyl nitrogen (lane/l16 044a2cdb15)
    ("OC(=O)c1ccc(N(c2ccccc2)C(C)=O)cc1", "4-(N-phenylacetamido)benzoic acid"),
    # an acid parent with the acyl nitrogen in its chain
    ("OC(=O)CN(C)C(C)=O", "(N-methylacetamido)acetic acid"),
    ("OC(=O)CCN(CC)C(C)=O", "3-(N-ethylacetamido)propanoic acid"),
    # the amide is the principal group there, so it is the suffix: no amido prefix of the senior group
    ("CC(=O)N(C)Cc1ccccc1", "N-benzyl-N-methylacetamide"),
    # a urea alone is the parent, not a prefix
    ("CNC(=O)Nc1ccc(NC(N)=O)cc1", "N-[4-(carbamoylamino)phenyl]-N'-methylurea"),
]


@pytest.mark.parametrize("smiles,name", _UNCHANGED)
def test_the_neighbours_of_the_class_keep_their_names(smiles, name):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_a_sulfonamide_senior_to_the_group_is_not_cited_as_its_prefix():
    """ "Maximum number of substituent groups cited as prefixes" (the Blue Book): two sulfonamides tie, so
    the group is no prefix of the other, and the PIN tier names neither as one."""
    smiles = "CS(=O)(=O)N(C)Cc1ccc(cc1)S(N)(=O)=O"
    row = _row(smiles, "pin")
    assert row["tier"] != "pin_verified", row
    assert "sulfonamido" not in (row["name"] or "").replace("benzene-1-sulfonamide", ""), row


def test_a_peptide_stays_out_of_the_pin_tier():
    """ INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in this
    Chapter": the tripeptide has no name of the PIN tier (its semisystematic residue name is no PIN, and the chain amido pass
    does not reach it, ``in_peptide``); the fragments the engine cuts from it are not asked. The systematic substitutive name
    of a dipeptide is a PIN of the project,:54247)."""
    smiles = "C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O"
    assert in_peptide(Chem.MolFromSmiles(smiles))
    assert _row(smiles, "pin")["tier"] != "pin_verified"


def test_a_group_behind_a_heteroatom_of_the_chain_keeps_the_name_it_had():
    """The acyl nitrogen on a nested branch of the chain is named by the prefix of that branch (the amine, the ester
    acyl), not consumed at the outer chain: the name keeps its 'acetic acid' parent."""
    smiles = "CC(=O)N(C)CC(C)(C)C(=O)OCC(CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O)O"
    name = ("[{2-[bis(carboxymethyl)amino]ethyl}(3-{[2,2-dimethyl-3-(N-methylacetamido)propanoyl]oxy}-2-"
            "hydroxypropyl)amino]acetic acid")
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


# ---- the producers, on their own atoms -------------------------------------------------------------------------


def _consume(smiles, attach_idx):
    """What ``consume_acylated_nitrogen`` consumes of the substituent of ``smiles`` that has its free valence on the atom
    ``attach_idx`` (every atom reached from it without entering a ring: the ring and the acid on it are the parent), and
    the prefixes it adds."""
    mol = Chem.MolFromSmiles(smiles)
    sub_set, stack = set(), [attach_idx]
    while stack:
        i = stack.pop()
        if i in sub_set or mol.GetAtomWithIdx(i).IsInRing():
            continue
        sub_set.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
    consumed, added = set(), []
    consume_acylated_nitrogen(mol, sub_set, attach_idx, consumed, lambda host, prefix: added.append((host, prefix)))
    return consumed, added


def test_the_consumer_takes_the_group_on_a_chain_carbon():
    # 4-(CH2-N(CH3)-CO-CH3)benzoic acid: atoms 7 (CH2), 8 (N), 9 (CH3), 10 (C=O), 11 (CH3), 12 (O)
    consumed, added = _consume("OC(=O)c1ccc(CN(C)C(C)=O)cc1", 7)
    assert consumed == {8, 9, 10, 11, 12} and added == [(7, "(N-methylacetamido)")]


def test_the_consumer_leaves_a_group_that_is_the_principal_group():
    """ (the Blue Book): the amide of the molecule is the suffix of the parent, never a prefix: the substituent
    of 'N-(3-phenylpropyl)formamide' has no amido prefix."""
    assert _consume("O=CNCCCc1ccccc1", 2) == (set(), [])


def test_the_consumer_leaves_a_group_behind_a_heteroatom_of_the_chain():
    """: the carbons of a substituent chain carry its prefixes; a carbon behind an oxygen belongs to the prefix of
    that group."""
    assert _consume("OC(=O)c1ccc(COCN(C)C(C)=O)cc1", 7) == (set(), [])


def test_the_consumer_takes_the_group_on_the_chain_of_a_ketone_acyl():
    """A free valence on the C=O of a ketone (the acyl group of a carbon, is an acyl group whose chain carries
    the amido prefix of (the Blue Book, 'Substituents of the types -NH-CO-R and -NH-SO2-R'):
    -CO-CH(R)-N(CH3)-CO-CH3 is '2-(N-methylacetamido)...acyl', the acyl name written from the prefixes of the chain
    (``acyl_prefix.acyl_prefix_text``; "substitution on acyl groups is identical to that of acids",:30556).

    change-asserted-value (lane/l17 fix a performance pass): this expectation was ``(set, )`` (the consumer left every acyl group to
    its own tiers). The name of this molecule is the same before and after ('4-[4-methyl-2-(N-methylacetamido)pentanoyl]benzoic
    acid', composed by the ring parent); the owner of an ester of sulfuric acid had no such tier and named the group
    '4-{2-[acetyl(methyl)amino]-1-oxoethyl}phenyl hydrogen sulfate', not the acyl name ``test_l17_12_fix_round2``. Independent
    check: a fresh OPSIN call reads '4-[(N-methylacetamido)acetyl]phenyl hydrogen sulfate' to the input's full key
    (``test_a_ketone_acyl_with_an_amido_prefix_is_named_by_the_acyl_name``). Mutation: with the guard of the consumer as it was
    (any C=O on the free-valence carbon returns), that name is the old spelling and this test fails."""
    # atoms: 7 C(=O), 9 CH, 14 N, 15 N-CH3, 16 C(=O), 17 CH3, 18 O
    assert _consume("OC(=O)c1ccc(C(=O)C(CC(C)C)N(C)C(C)=O)cc1", 7) == ({14, 15, 16, 17, 18}, [(9, "(N-methylacetamido)")])


def test_the_consumer_leaves_an_acyl_group_on_a_heteroatom():
    """An acyl group on a nitrogen or an oxygen (an amide, an ester) has its own prefix tiers: the amido prefix of an
    acylated nitrogen of its chain is not consumed beside the amide that carries it."""
    assert _consume("CN(C(=O)CN(C)C(C)=O)c1ccc(C(=O)O)cc1", 2) == (set(), [])


def test_the_consumer_leaves_a_group_on_a_heteroatom_attachment():
    assert _consume("OC(=O)c1ccc(N(C)CN(C)C(C)=O)cc1", 7) == (set(), [])


def _branch(mol, n):
    """The atoms of the group on the nitrogen ``n``: every atom reached from it without entering a ring."""
    seen, stack = set(), [n]
    while stack:
        i = stack.pop()
        atom = mol.GetAtomWithIdx(i)
        if i in seen or (i != n and atom.IsInRing()):
            continue
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in atom.GetNeighbors())
    return seen


@pytest.mark.parametrize("smiles,senior", [
    ("OC(=O)c1ccc(N(C)S(C)(=O)=O)cc1", True),   # an acid is senior to a sulfonamide, the Blue Book)
    ("NC(=O)c1ccc(N(C)S(C)(=O)=O)cc1", True),   # a carboxamide (class 16) is senior to a sulfonamide (class 19)
    ("NS(=O)(=O)c1ccc(N(C)S(C)(=O)=O)cc1", False),  # a second sulfonamide ties: decides
])
def test_a_sulfonamido_prefix_needs_a_senior_group(smiles, senior):
    mol = Chem.MolFromSmiles(smiles)
    n = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetDegree() == 3 and not a.IsInRing()][-1]
    assert parent_sulfonamide_is_senior(mol, _branch(mol, n)) is senior
