"""Wider tiers give the PIN for the item 11 classes.

The wider tiers (valid, complete, best-effort) name a molecule in one pass with the breadth producers in it, so a
class the PIN tier names by rule can reach the wider tiers in a less preferred spelling. Each section below pins one
class; the target names are read back whole (``assert_full_rt``) and the PIN tier's own row is asserted unchanged.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# --- A1: a carboacyl branch is spelled as an acyl, not '1-oxo...yl' ------------------------------------------------
#
# Definitions and name formation (the Blue Book): "Compound substitutive names for acyclic acyl
# groups, such as '1-oxopropyl' and '1-iminoethyl'... are included for use in general nomenclature." The acyl names
# are the preferred prefixes: "CH3-CO– acetyl (preferred prefix) ethanoyl 1-oxoethyl" (:30442).
# A ring-bearing chain carboacyl branch used to be assembled by the located-FG namer (Tier FG) into
# '2-(...)-1-oxoethyl'; the cascade now declines it there and the caller spells the acyl group, as the other tiers do.

_ACYL_CASES = [
    # (smiles, text the name holds, text it must not hold)
    ("COC(=O)c1ccc2c(c1)NC(=O)/C2=C(\\Nc1ccc(N(C)C(=O)CN2CCN(C)CC2)cc1)c1ccccc1",
     "{methyl[(4-methylpiperazin-1-yl)acetyl]amino}", "1-oxoethyl"),
    ("COC(=O)c1ccc(N(C)C(=O)Cc2ccccc2)cc1", "(N-methyl-2-phenylacetamido)", "1-oxo-2-phenylethyl"),
    ("COC(=O)c1ccc(N(C)C(=O)CN2CCOCC2)cc1", "[N-methyl-2-(morpholin-4-yl)acetamido]", "1-oxoethyl"),
]


# the first row's valid and complete names are a von Baeyer parent whose phenyl prefix the general namer spells (task A9
# does not reach it); the best-effort name takes the amido prefix of the anilino group
_AMIDO_AT_BEST_EFFORT = {_ACYL_CASES[0][0]: "{4-[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]anilino}"}


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,holds,not_holds", _ACYL_CASES)
def test_carboacyl_branch_takes_the_acyl_name(smiles, holds, not_holds, tier):
    """ (:30442): the acyl name 'acetyl' is the preferred prefix, '1-oxoethyl' the general-nomenclature one.
    The acyl of a tertiary nitrogen is cited as the N-substituted amido prefix (task A9;,:32991, method (1)
    "generates preferred IUPAC names", '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)',:33040): the acyl name
    stood here while the nitrogen's own amido prefix had no builder."""
    row = _row(smiles, tier)
    holds = _AMIDO_AT_BEST_EFFORT.get(smiles, holds) if tier == "best-effort" else holds
    assert holds in row["name"] and not_holds not in row["name"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_a_senior_acid_keeps_the_amide_pin(tier):
    """ (method (1)): the acid is the parent and the amide is cited as the prefix '2-phenylacetamido'."""
    smiles = "OC(=O)c1ccc(NC(=O)Cc2ccccc2)cc1"
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == ("4-(2-phenylacetamido)benzoic acid", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # a carboacyl branch with no ring atom: the guard is ring-restricted, and the nitrogen's own prefix is the amido one
    # (task A9,,:32991)
    ("COC(=O)c1ccc(N(C)C(=O)CC(=O)CC)cc1", "methyl 4-(N-methyl-3-oxopentanamido)benzoate"),
    # the head carries a ring directly (aroyl, cycloalkanecarbonyl): the located-FG writer spelled the acyl name here
    ("COC(=O)c1ccc(N(C)C(=O)c2ccccc2)cc1", "methyl 4-(N-methylbenzamido)benzoate"),
    ("COC(=O)c1ccc(N(C)C(=O)C2CCCCC2)cc1", "methyl 4-(N-methylcyclohexanecarboxamido)benzoate"),
])
def test_the_carboacyl_guard_leaves_other_branches_alone(smiles, name):
    """Only a chain carboacyl branch that holds a ring is taken off the located-FG writer,:30442); the
    other acyl branches of a tertiary nitrogen are cited as its N-substituted amido prefix,:32991)."""
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


def test_the_pin_tier_rows_are_unchanged_by_the_acyl_route():
    """The guard acts on the wider tiers only: the PIN tier's own rows stand. The two ester rows take the PIN of
     (task A9,:32991); the molecule with the N-methylpiperazine acyl on a ring that is not the parent
    stays an abstention, as its NH analog does."""
    assert _row("OC(=O)c1ccc(NC(=O)Cc2ccccc2)cc1", "pin")["name"] == "4-(2-phenylacetamido)benzoic acid"
    pins = {
        "COC(=O)c1ccc(N(C)C(=O)Cc2ccccc2)cc1": "methyl 4-(N-methyl-2-phenylacetamido)benzoate",
        "COC(=O)c1ccc(N(C)C(=O)CN2CCOCC2)cc1": "methyl 4-[N-methyl-2-(morpholin-4-yl)acetamido]benzoate",
    }
    for smiles, _holds, _not_holds in _ACYL_CASES:
        row = _row(smiles, "pin")
        want = (pins[smiles], "pin_verified") if smiles in pins else ("unknown organic compound", "abstain")
        assert (row["name"], row["tier"]) == want, row


# --- A2: a fused parent keeps its ring-bearing branch at the wider tiers ------------------------------------------
#
# Five-membered ring requirement (the Blue Book): "When fusion names are not allowed, unsaturated von
# Baeyer ring system names are preferred IUPAC names" (:23710). Here the fusion name is allowed, so the fused name is
# the PIN. A fused parent's ring-bearing exocyclic branch ('4-methylpiperazin-1-yl', '4-methoxyoxan-4-yl') used to get
# no name from the wider tiers' ring producer, the fused assembly failed closed, and the von Baeyer name won.

_FUSED_RING_BRANCH_CASES = [
    ("CN1CCN(c2ccc3c(c2)C(=O)c2nc(C#N)c(N)n2-3)CC1",
     "3-amino-7-(4-methylpiperazin-1-yl)-9-oxo-9H-imidazo[1,2-a]indole-2-carbonitrile"),
    ("CC(C)[Si]1(O[C@@H]2[C@H]([C@@H]([C@@H]([C@@H]([C@H]2O[Si](O1)(C(C)C)C(C)C)O)OC3(CCOCC3)OC)O)O)C(C)C",
     "(5aR,6S,7S,8S,9S,9aR)-8-[(4-methoxyoxan-4-yl)oxy]-2,2,4,4-tetra(propan-2-yl)hexahydro-2H,4H-1,3,5,2,4-"
     "benzotrioxadisilepine-6,7,9-triol"),
    ("CCCCCC[C@H](O)[C@@H]1/C(C(=O)O)=C(/C(=O)O)C[C@H]([C@@H](O)[C@H]2CC=CC(=O)C2)CC2=C(C(=O)OC2=O)[C@H]1O",
     "(4S,5S,6Z,9S)-4-hydroxy-5-[(1S)-1-hydroxyheptyl]-9-{(1R)-hydroxy[(1S)-5-oxocyclohex-3-en-1-yl]methyl}-1,3-dioxo-"
     "3,4,5,8,9,10-hexahydro-1H-cyclonona[c]furan-6,7-dicarboxylic acid"),
]

_ROW32 = "C[C@@H]1CC[C@@H]2[C@@H]1CC3(C4CC2([C@]3(C(=C4)C(C)C)C(=O)O)C=O)C(=O)C#CCC5CCCCC5"


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name", _FUSED_RING_BRANCH_CASES)
def test_fused_parent_with_ring_branch_keeps_the_fused_pin(smiles, pin_name, tier):
    """ (:23708-:23710): a fused parent is preferred to the von Baeyer name where a fusion name is allowed."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_fused_parent_with_a_ring_bearing_acyl_branch_names_as_the_pin_tier_does(tier):
    """Same rule,:23708): the wider tiers give the PIN tier's own name; its spelling ('1,4-methano-s-indacene') is checked in A10."""
    pin = _row(_ROW32, "pin")
    row = _row(_ROW32, tier)
    assert pin["tier"] == "pin_verified", pin
    assert (row["name"], row["tier"]) == (pin["name"], pin["tier"]), (pin, row)
    assert_full_rt(row["name"], _ROW32)


@pytest.mark.parametrize("smiles,pin_name", _FUSED_RING_BRANCH_CASES)
def test_the_pin_tier_rows_are_unchanged_by_the_strict_ring_fallback(smiles, pin_name):
    """The fallback acts on the wider tiers only: the PIN tier's own rows stand."""
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_a_fused_ring_prefix_cites_the_ring_fusion_centres(tier):
    """ Stereodescriptors used in substitutive nomenclature (:44624): "In preferred IUPAC names,
    stereodescriptors, preceded by a locant, must be cited to specify each stereogenic unit"; the fusion atoms of a
    fused ring are lettered "Numbering of peripheral skeletal atoms",:12501). The strict prefix of
    this branch cited (1R,2S) only, the gates rejected the route that took it, and a name with the hydroxy group as a
    prefix won. The carbamate is the parent word "SENIORITY ORDER FOR CLASSES", Table 4.1: class 9 "Esters",:18182,
    ranks above class 17 "Hydroxy compounds",:18190; the exemplar of is '2-hydroxypropyl (2-aminoethyl)carbamate
    (PIN)',:30766), and the fused ring prefix of its N group cites all four centres, 3aR and 7aS among them. (This row
    asserted '{(1R,2S,3aR,7aS)-2-[(tert-butoxycarbonyl)amino]octahydro-1H-inden-1-yl}methanol', the alcohol as the parent.)"""
    smiles = "CC(C)(C)OC(=O)N[C@H]1C[C@H]2CCCC[C@@H]2[C@H]1CO"
    row = _row(smiles, tier)
    assert row["name"] == "tert-butyl [(1R,2S,3aR,7aS)-1-(hydroxymethyl)octahydro-1H-inden-2-yl]carbamate", row
    assert_full_rt(row["name"], smiles)


_DECALIN_BRANCH = "CC[C@@H](C)[C@H]1O[C@]1(C)[C@]1(O)C(C)=C[C@@H]2CC(CO)=CC[C@H]2[C@@H]1/C=C/C=C/C=C/C(=O)O"


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
def test_a_fused_ring_branch_with_fusion_centres_gives_one_pin_at_every_tier(tier):
    """ (:44624) with (:12501): the bicyclic branch cites its fusion centres '4a' and '8a',
    the name reads back whole, and the default tier ships it as a PIN where it abstained (the prefix without them
    named another stereoisomer, which the gates reject). Every tier gives the same name."""
    row = _row(_DECALIN_BRANCH, tier)
    # the enclosing marks of the branch follow (the Blue Book): "When the nesting order... results in
    # consecutive enclosing marks of the same level, the next level of enclosing mark is used", so '{' inside '[' and '['
    # inside '{' where the level after '{' would be '(('
    assert (row["name"], row["tier"]) == (
        "(2E,4E,6E)-7-[(1S,2R,4aR,8aR)-2-{(2S,3R)-3-[(2R)-butan-2-yl]-2-methyloxiran-2-yl}-2-hydroxy-6-(hydroxymethyl)-3-"
        "methyl-1,2,4a,5,8,8a-hexahydronaphthalen-1-yl]hepta-2,4,6-trienoic acid", "pin_verified"), row
    assert row["name"] == _row(_DECALIN_BRANCH, "pin")["name"], row
    assert_full_rt(row["name"], _DECALIN_BRANCH)


# --- A3: a ring assembly in a split fragment keeps its assembly prefix ---------------------------------------------
#
# Substituent prefixes derived from ring assemblies (the Blue Book): "Names of substituent groups
# derived from ring assemblies are written in two ways: (1) the suffix 'yl' is added to the name biphenyl; (2) names
# denoted by locants are placed in brackets", with '[2,2'-bipyridin]-5-yl (preferred prefix)' (:16130);
# (:15567): "These locants must be cited in preferred IUPAC names". A branch that holds a ring assembly used to be
# named ring by ring ('5-(thiophen-2-yl)thiophen-2-yl', '4-phenylphenyl') by the terminal-fragment writer, which the
# decorated-chain composer (the PIN tier's own route) precedes when the branch holds an assembly.

_ASSEMBLY_BRANCH_CASES = [
    ("CC(=O)OCC(O)C#Cc1ccc(-c2cccs2)s1", "4-([2,2'-bithiophen]-5-yl)-2-hydroxybut-3-yn-1-yl acetate"),
    ("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)cc1", "4-([1,1'-biphenyl]-4-yl)-2-hydroxybut-3-yn-1-yl acetate"),
    # the branch keeps its stereodescriptors,:44624)
    ("CC(=O)OC[C@@H](O)C#Cc1ccc(-c2ccccc2)cc1", "(2S)-4-([1,1'-biphenyl]-4-yl)-2-hydroxybut-3-yn-1-yl acetate"),
    ("CC(=O)OC[C@H](O)/C=C/c1ccc(-c2cccs2)s1", "(2R,3E)-4-([2,2'-bithiophen]-5-yl)-2-hydroxybut-3-en-1-yl acetate"),
]


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name", _ASSEMBLY_BRANCH_CASES)
def test_ring_assembly_branch_keeps_the_assembly_prefix(smiles, pin_name, tier):
    """ (:16107) and (:15567): the branch is '[2,2'-bithiophen]-5-yl', not the ring-by-ring spelling."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,pin_name", _ASSEMBLY_BRANCH_CASES)
def test_the_pin_tier_rows_are_unchanged_by_the_assembly_branch_route(smiles, pin_name):
    """The route acts on the wider tiers only: the PIN tier's own rows stand."""
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,name", [
    # controls: the assembly is the substituent of the parent chain / ester alkyl group, named by the producers that
    # already spell it (not touched by the split-fragment route)
    ("OCCCc1ccc(-c2ccccc2)cc1", "3-([1,1'-biphenyl]-4-yl)propan-1-ol"),
    ("CC(=O)OCCc1ccc(-c2cccs2)s1", "2-([2,2'-bithiophen]-5-yl)ethyl acetate"),
    # a branch that holds two different ring systems is no assembly: identical rings): ring by ring
    ("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)s1", "2-hydroxy-4-(5-phenylthiophen-2-yl)but-3-yn-1-yl acetate"),
])
def test_other_branches_keep_their_names(smiles, name, tier):
    """ (:15567) names an assembly of identical rings only; every other branch is spelled as before."""
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_a_composer_name_that_cites_no_assembly_does_not_replace_the_writers(tier):
    """ (:16107): the composer is offered a branch that holds an assembly to spell it '[...]-yl'. Here the
    composer names the two oxolanes ring by ring as well, and its chain core ('1,3-dimethyl-4-oxoheptyl', counted from
    the free valence) is shorter than the writer's 'octan-2-yl': the longest chain is the parent): no gain,
    so the writer's name stands."""
    smiles = "CCC(C(=O)C(C)C(O)C(C)C1OC(O)(CC(=O)O)C(C)CC1C)C1OC(C)(C2(O)OC(CC)(C(C)O)CC2C)CC1C"
    row = _row(smiles, tier)
    assert "-3-hydroxy-4-methyl-5-oxooctan-2-yl)" in row["name"] and "heptyl" not in row["name"], row


@pytest.mark.parametrize("smiles,drop_ring,expected", [
    # two identical rings joined by a bond: an assembly; with one of them left out of the fragment, none
    ("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)cc1", None, True),
    ("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)cc1", -1, False),
    # nonidentical rings are no assembly: "composed of identical cyclic systems")
    ("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)s1", None, False),
    # a ring that no direct bond joins to the pair is no part of the assembly, and does not hide it
    ("c1ccc(-c2ccccc2)cc1CCc1ccccc1", None, True),
    ("Cc1ccc(-c2ccc(-c3ccccc3)cc2)cc1", None, True),
])
def test_fragment_holds_ring_assembly(smiles, drop_ring, expected):
    """ "DEFINITIONS" (:15540): rings 'directly joined to each other by single or double bonds' are a ring
    assembly when 'composed of identical cyclic systems'; a fragment with only one of the rings holds none."""
    from rdkit import Chem

    from orthonym.rules.ring_assemblies import fragment_holds_ring_assembly

    mol = Chem.MolFromSmiles(smiles)
    atoms = set(range(mol.GetNumAtoms()))
    if drop_ring is not None:
        atoms -= set(mol.GetRingInfo().AtomRings()[drop_ring])
    assert fragment_holds_ring_assembly(mol, atoms) is expected


# The composer counts the chain of a branch from the free valence. "THE PRINCIPAL SUBSTITUENT CHAIN" (b)
# (the Blue Book): "the greater number of skeletal atoms, i.e., the longest chain"; the example at:22744,
# '1-hydroxypropan-2-yl (preferred prefix)... 2-hydroxy-1-methylethyl... [not 1-(hydroxymethyl)ethyl]'. For the alcohol
# group of these esters the chain runs through the attachment carbon, so the PIN is '...-3-hydroxypent-4-yn-2-yl acetate'
# and '...-2-hydroxy-1-methylbut-3-yn-1-yl' is the shorter chain: a branch whose attachment atom lies inside its chain is
# left to the writer, and no tier labels the composer's chain a PIN.

_ASSEMBLY_INSIDE_CHAIN_CASES = [
    # (smiles, the (b) PIN, the name the tiers gave before the assembly route, its tier at the wider tiers)
    ("CC(=O)OC(C)C(O)C#Cc1ccc(-c2ccccc2)cc1",
     "5-([1,1'-biphenyl]-4-yl)-3-hydroxypent-4-yn-2-yl acetate",
     "2-hydroxy-1-methyl-4-(4-phenylphenyl)but-3-yn-1-yl acetate"),
    ("CC(=O)OC(C)C(O)/C=C/c1ccc(-c2cccs2)s1",
     "(4E)-5-([2,2'-bithiophen]-5-yl)-3-hydroxypent-4-en-2-yl acetate",
     "(3E)-2-hydroxy-1-methyl-4-[5-(thiophen-2-yl)thiophen-2-yl]but-3-en-1-yl acetate"),
]


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name,earlier_name", _ASSEMBLY_INSIDE_CHAIN_CASES)
def test_a_branch_attached_inside_its_chain_is_not_named_from_the_free_valence(smiles, pin_name, earlier_name, tier):
    """ (b) (:22632): no tier ships '1-methylbut-3-yn-1-yl' as a PIN. The PIN tier gives the book's chain or
    abstains; the wider tiers give the book's chain or the earlier systematic name (not a PIN)."""
    row = _row(smiles, tier)
    assert not (row["tier"] == "pin_verified" and "-1-methylbut-3-" in row["name"]), row
    if row["tier"] == "pin_verified":
        assert row["name"] == pin_name, row
    elif tier == "pin":
        assert row["tier"] == "abstain", row
    elif row["name"] == pin_name:
        # the book's chain, labelled below the PIN for the ring assembly's own spelling
        assert row["tier"] == "pin_unverified", row
        assert_full_rt(row["name"], smiles)
    else:
        assert (row["name"], row["tier"]) == (earlier_name, "systematic_verified"), row
        assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,at_attach,expected", [
    # (the fragment is every atom after the ester oxygen; the attachment atom, index 4, is the one it bonds to)
    ("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)cc1", 4, True),    # CH2: the chain starts here
    ("CC(=O)OC(C)C(O)C#Cc1ccc(-c2ccccc2)cc1", 4, False),    # CH(CH3)-CH(OH): the chain runs on both sides
    ("CC(=O)OC(C)c1ccc(-c2ccccc2)cc1", 4, True),    # CH(CH3)-ring: one acyclic carbon neighbour
    ("CC(=O)OC1CCCCC1", 4, True),    # a ring atom
])
def test_the_chain_runs_from_the_attachment_atom(smiles, at_attach, expected):
    """ (b) (:22632): the free valence ends the principal chain only when the attachment atom has one chain
    neighbour."""
    from rdkit import Chem

    from orthonym.rules.terminal_fragment import _chain_runs_from_attachment

    mol = Chem.MolFromSmiles(smiles)
    frag = set(range(4, mol.GetNumAtoms()))
    assert mol.GetAtomWithIdx(at_attach).GetSymbol() == "C"
    assert _chain_runs_from_attachment(mol, frag, at_attach) is expected


def test_the_assembly_branch_route_degrades_on_an_error(monkeypatch):
    """The neighbouring writer degrades on an unexpected error (17 call sites): so does the route."""
    from rdkit import Chem

    import orthonym.rules.ring_assemblies as ra
    from orthonym.metrics.provenance import general_fallback_ctx
    from orthonym.rules.terminal_fragment import ring_assembly_branch_name

    def boom(*a, **k):
        raise RuntimeError("probe")

    monkeypatch.setattr(ra, "fragment_holds_ring_assembly", boom)
    mol = Chem.MolFromSmiles("CC(=O)OCC(O)C#Cc1ccc(-c2ccccc2)cc1")
    token = general_fallback_ctx.set(True)
    try:
        assert ring_assembly_branch_name(mol, set(range(4, mol.GetNumAtoms())), 4) is None
    finally:
        general_fallback_ctx.reset(token)


# --- A2 follow-up: the strict ring producer numbers a monocycle by -----------------------------------------
#
# (the Blue Book): "The locant '1' is given to a heteroatom that occurs first in the seniority sequence
# used for citation of the skeletal replacement ('a') prefixes. The numbering is then chosen to give lowest locants to
# heteroatoms considered as a set in ascending numerical order." The enumerator's monocycle numbering ranked the lowest
# set first, so a 1,3,4-oxadiazole took the nitrogen numbering N1,N2,C3,O4 (the set {1,2,4}) and its ring carbon was
# cited '-3-yl'. The strict producer asks it at every wider tier since A2, and the ketone below lost its suffix to a
# name with the oxadiazole branch unnamed.

@pytest.mark.parametrize("smiles,attach,tail", [
    ("CSc1nnc(C)o1", 2, "1,3,4-oxadiazol-2-yl"),
    ("CSc1nnc(C)s1", 2, "1,3,4-thiadiazol-2-yl"),
    ("CSc1nnco1", 2, "1,3,4-oxadiazol-2-yl"),
    ("Cc1nnc(C)o1", 1, "1,3,4-oxadiazol-2-yl"),
])
def test_a_monocycle_core_gives_locant_1_to_the_senior_heteroatom(smiles, attach, tail):
    """ (:8284): O (or S) is locant 1 before the lowest set of heteroatom locants is looked at."""
    from rdkit import Chem

    from orthonym.assembly import substituent_enumerator as se

    mol = Chem.MolFromSmiles(smiles)
    ring = set(mol.GetRingInfo().AtomRings()[0])
    carriers = [a for a in ring if a != attach and any(nb.GetIdx() not in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors())]
    pos = se._monocycle_position_map(mol, ring, attach, carriers, mol.GetRingInfo())
    assert pos[attach] == 2, pos
    assert se._monocycle_core_tail(mol, ring, attach, pos, mol.GetRingInfo()) == tail


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_a_ketone_keeps_its_suffix_when_the_strict_producer_names_the_oxadiazole_branch(tier):
    """ (Table 4.1): the ketone is senior to the ether and the heteroarene; the ring branch is named
    '...-1,3,4-oxadiazol-2-yl', so the ketone stays the suffix of the 1-(4-methylphenyl)ethan-1-one parent at
    best-effort (valid and complete name the same molecule by the engine's systematic name, as at the lane start)."""
    smiles = "CC1=CC=C(C=C1)C(=O)CSC2=NN=C(O2)COC3=CC=CC4=C3N=CC=C4"
    row = _row(smiles, tier)
    assert "oxadiazol-3-yl" not in row["name"], row
    if tier == "best-effort":
        assert row["name"].endswith("ethan-1-one") and "1-(4-methylphenyl)" in row["name"], row
    assert_full_rt(row["name"], smiles)


_AMIDOXIME = "CC1=NC(=C(C=C1)/C(=N/O)/N(C)C2CCN(CC2)C)OC3=CC=CC(=C3)C(C)C"


def test_the_amidoxime_keeps_the_ring_cascades_pyridine_parent_at_best_effort():
    """ (heterocycles are senior to carbocycles) and Oximes (the Blue Book), the prefix
    '(hydroxyimino)': the name is built on the pyridine, as at the lane start, and not on the benzene with the pyridine as
    '(pyridin-2-yl)oxy'. Since A2 names the piperidin-4-yl group of the amidoxime fragment at the wider tiers, the located
    FG assembler (``substituent_naming._located_fg_assemble``) reached that fragment and named the branch =N-OH of its
    carbon as a single-bond prefix, '(hydroxyamino)[methyl(...)amino]methyl' (another molecule, rejected at the round
    trip) and the ring cascade's parent was lost to it; the assembler now declines a branch joined by a double bond that
    it has no ylidene prefix for."""
    row = _row(_AMIDOXIME, "best-effort")
    assert row["name"] == ("(1Z)-3-{(hydroxyimino)[methyl(1-methylpiperidin-4-yl)amino]methyl}-6-methyl-"
                           "2-[3-(propan-2-yl)phenoxy]pyridine"), row
    assert_full_rt(row["name"], _AMIDOXIME)


# --- A4: the inner amides of a peptide-like chain take the amido prefix -------------------------------------------
#
# Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): method (1) "changing the final
# letter 'e' in the complete name of the amide to 'o'" (:32995); "Method (1) generates preferred IUPAC names" (:32998).
# The acyl branch of a peptide-like chain is named as its acid; a retained peptide name ('acetylvalylvaline') has no
# '-oic acid' to turn into '-amido', so the branch is named substitutively first (the peptide handler's own route) and
# every inner amide is '...amido', not '[(...)-1-oxoethyl]amino'.

_PEPTIDE_CHAIN_CASES = [
    ("CC(=O)NCC(=O)NCC(=O)NCC(=O)NCCC(=O)O",
     "3-{2-[2-(2-acetamidoacetamido)acetamido]acetamido}propanoic acid"),
    ("CC(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)NCC(=O)NCCC(=O)O",
     "3-(2-{(2S)-2-[(2S)-2-acetamido-3-methylbutanamido]-3-methylbutanamido}acetamido)propanoic acid"),
    ("CC(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)[C@@H](O)CC(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CC(=O)O)"
     "C(C)C)C(C)C",
     "(3S,4S)-4-{(2S)-2-[(3S,4S)-4-{(2S)-2-[(2S)-2-acetamido-3-methylbutanamido]-3-methylbutanamido}-3-hydroxy-6-"
     "methylheptanamido]propanamido}-3-hydroxy-5-phenylpentanoic acid"),
]


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name", _PEPTIDE_CHAIN_CASES)
def test_inner_amides_take_the_amido_prefix(smiles, pin_name, tier):
    """ (:32995,:32998): every inner amide of the chain is cited as '...amido' (method (1))."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,pin_name", _PEPTIDE_CHAIN_CASES)
def test_the_pin_tier_rows_are_unchanged_by_the_peptide_acyl_route(smiles, pin_name):
    """The route acts on the wider tiers only: the PIN tier's own rows stand."""
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row


@pytest.mark.parametrize("smiles,pin_name", _PEPTIDE_CHAIN_CASES)
def test_every_entry_point_gives_the_same_peptide_chain_name(smiles, pin_name):
    """One name from ``name_compound``, ``name_tiered`` and ``name_with_confidence`` (the default tier)."""
    from orthonym import name_compound

    namer = _tiered_namer("pin", "best-effort")
    assert name_compound(smiles) == pin_name
    record = namer.name_with_confidence(smiles)
    assert (record["name"], record["tier"]) == (pin_name, "pin_verified"), record
    assert namer.name_tiered(smiles)["name"] == pin_name


# (the Blue Book), under "Guanidine and its derivatives": "In the presence of a
# characteristic group having seniority over guanidine... the following prefixes are used": 'carbamimidoylamino
# (preferred prefix)'. The chain composer skipped a guanidine branch as "named by the functional-group loop" when that
# loop had emitted no prefix for it, so a chain whose amide branch is now spelled '...amido' lost the 5-guanidino group
# ('...-3-phenylpropanamido}pentanoic acid'); the gates rejected that name and the peptide went unnamed. A branch is
# skipped only when the loop named it.

_ARGININE_CASES = [
    ("N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O",
     "(2S)-2-{(2S)-2-[(2S)-2-amino-3-hydroxypropanamido]-3-phenylpropanamido}-5-(carbamimidoylamino)pentanoic acid"),
    ("N[C@@H](C)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O",
     "(2S)-2-[(2S)-2-{(2S)-2-[(2S)-2-aminopropanamido]-3-hydroxypropanamido}-3-phenylpropanamido]-5-"
     "(carbamimidoylamino)pentanoic acid"),
]


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name", _ARGININE_CASES)
def test_a_guanidine_branch_is_skipped_only_when_the_functional_group_loop_named_it(smiles, pin_name, tier):
    """ (:34266): the 5-guanidino group of arginine is cited 'carbamimidoylamino' at every tier."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,pin_name", [
    # the same peptide, the guanidine drawn (H2N)2C=N-: '(diaminomethylidene)amino (preferred prefix)'
    ("N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O",
     "(2S)-2-{(2S)-2-[(2S)-2-amino-3-hydroxypropanamido]-3-phenylpropanamido}-5-[(diaminomethylidene)amino]pentanoic acid"),
    ("CC(C)[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O",
     "(2S)-2-{(2S)-2-[(2S)-2-amino-3-(4-hydroxyphenyl)propanamido]-3-methylbutanamido}-5-"
     "[(diaminomethylidene)amino]pentanoic acid"),
])
def test_the_guanidine_drawing_decides_the_prefix(smiles, pin_name, tier):
    """ "In the presence of a characteristic group having seniority over guanidine" (the Blue Book):
    the book prints a prefix for each drawing, 'carbamimidoylamino (preferred prefix)' for H2N-C(=NH)-NH- and
    '(diaminomethylidene)amino (preferred prefix)' for (H2N)2C=N-, '4-[(diaminomethylidene)amino]butanoic acid (PIN)'
    (:34282). A peptide drawn with the second form took the first one's prefix before the amido route of A4."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin_name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
