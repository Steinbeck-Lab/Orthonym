"""An amide or an ester is named by its class, not on a parent hydride that cites it in a prefix (lane/l17 item 10).

 "N-Substitution" (the Blue Book): R-CO-NHR' is named "by citing the substituents R' and R'' as
prefixes preceded by the locant N". SENIORITY ORDER FOR CLASSES, Table 4.1 (:18158): the amide (class 11) and the
ester (class 9) are principal characteristic groups of these molecules, so a name that holds them only in an 'acetamido'
or an 'acetyloxy' prefix is a name of a lower class. (:31663): "All preferred IUPAC names for esters are
named by functional class nomenclature", the alcoholic component by "seniority of rings, ring systems, and chains"
,:31858), a ring before a chain.

The decomposition engine splits a molecule only when the name the pipeline gives it without the split (the probe) is
poor, and it judges the probe by a character count. At a wider tier the probe held a name of another molecule that the
count took for a good one, so the split was never made and a general-engine name that cites the principal group in a
prefix shipped. Two producers made those names, and both are repaired where the name is made, so that no probe is read
back through OPSIN:

* ``CNC(=O)CNCC(=O)NC...`` (two N-substituted amides joined by an azanediyl group): the chain-suffix machinery of the
  polyfunctional producer and the general-acyclic fallback express the amides of the principal chain and no N-substituent
  of the amide that is the suffix, which is a name of another molecule ('1-(methylamino)-2-{...}acetamide',
  'aminopropanamide'). They decline when an acyl carbon of the principal group is off the principal chain
  (``rules.polyfunctional_parts.principal_amides_off_chain``).
* An ester whose alcohol fragment is named with its hydroxy group as a prefix (the fragment's own principal group is an
  ester or a ketone): the string rule turned the hydrocarbon name into a group word by cutting 'ane'
  ('5-[...]-9-hydroxy-1-methyl-4-oxobicyclo[4.3.0]nonyl acetate'), which kept the hydroxy group and had no locant for
  the free valence. A name that does not state the oxygen as '-ol' is not trusted: the group built from the structure
  is the first candidate and is read back (``_alcohol_group_candidates``).

The N-substituent of an amide built from an amine name keeps the free valence as the alcohol rule does,
the Blue Book; '2-methylcyclopentyl (preferred prefix)',:16204): '2-methylcyclohexyl', not '2-methylcyclohexan-1-yl'.

Two amides of one molecule that tie as parent structures (the same number of prefixes at the same locants) are told apart by
the names of their prefixes: "the lower locant set for substituent groups in order of citation in the name"
(:21791) and "the name that is earlier in alphanumerical order" (:22234; ``rules.amide_tie_names``,
``decomposition.amide_ties``). The split names the molecule on the amide those criteria choose where it has a name for it,
and labels the name of the other amide below the PIN where it has not.

The N-acyl float of the amide assembler is not built on a name that the best-effort rescue gave to the amine fragment (the
rescue names an ester as an oxo-alkane and cites the nitrogen inside a substituent): its locant N would designate no
nitrogen of the parent "Position of locants",:2857).
"""
from types import SimpleNamespace

import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_WIDER = ["valid", "complete", "best-effort"]

# the two target rows
_AMIDE = "C[C@@H]1CCCC[C@@H]1NC(=O)CNCC(=O)NC"
_AMIDE_NAME = "N-methyl-2-[(2-{[(1S,2R)-2-methylcyclohexyl]amino}-2-oxoethyl)amino]acetamide"
_ESTER = "CC(CCCC(=O)CCC1C2CCC(C2(CCC1=O)C)OC(=O)C)OC(=O)C"
_ESTER_NAME = "4-[7-(acetyloxy)-3-oxooctyl]-7a-methyl-5-oxooctahydro-1H-inden-1-yl acetate"


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


@pytest.mark.parametrize("tier", _WIDER)
def test_the_amide_is_the_suffix(tier):
    """The amide is the parent, the other amide a prefix of the acid side. The two amides tie as parents,:
    each cites two prefixes, 'N,2'), and the lower locant set in the order of citation decides, the Blue Book,
    " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS"): the amide that has the N-methyl group cites
    'methyl' before 'methylcyclohexyl...' at 'N,2', the other cites its other amide first, at '2,N',:3189: 'N' is
    lower than a numeral). The name is the first of them, with its N-block first as the order of citation puts it. It is
    not labelled the PIN (the nested prefix of the acid side is not certified), and the PIN tier gives no name."""
    row = _row(_AMIDE, tier)
    assert row["name"] == _AMIDE_NAME, row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], _AMIDE)


def test_the_pin_tier_gives_no_name_for_the_amide_with_a_substituted_ring():
    row = _row(_AMIDE, "pin")
    assert row["tier"] == "abstain" and not row["is_pin"], row


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
@pytest.mark.parametrize("smiles,name", [
    # positive controls: the same two amides with an N-substituent that was named, or abstained, before the repair
    ("CNC(=O)CNCC(=O)NC1CCCCC1", "N-cyclohexyl-2-{[2-(methylamino)-2-oxoethyl]amino}acetamide"),
    ("CNC(=O)CNCC(=O)NCC", "N-ethyl-2-{[2-(methylamino)-2-oxoethyl]amino}acetamide"),
])
def test_two_amides_with_an_azanediyl_group_are_named_on_the_amide_at_every_tier(smiles, name, tier):
    """'N-ethyl...' sorts before 'methylamino...', so the name is the PIN; before, the PIN tier gave no name for the
    ethyl amide ('2-{[2-(ethylamino)-2-oxoethyl]amino}-1-(methylamino)acetamide' stood in the way of the split)."""
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=O)CNCC(=O)NCCC", "N-methyl-2-{[2-oxo-2-(propylamino)ethyl]amino}acetamide"),
    ("CNC(=O)CNCC(=O)NCC=C", "N-methyl-2-({2-oxo-2-[(prop-2-en-1-yl)amino]ethyl}amino)acetamide"),
    ("CC(C)CNC(=O)CNCC(=O)NC", "N-methyl-2-({2-[(2-methylpropyl)amino]-2-oxoethyl}amino)acetamide"),
    ("CNC(=O)CNCC(=O)NC(C)C", "N-methyl-2-({2-oxo-2-[(propan-2-yl)amino]ethyl}amino)acetamide"),
])
@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
def test_the_amide_that_has_the_n_methyl_group_is_the_pin_where_its_locants_are_lower(smiles, name, tier):
    """The N-propyl amide would cite its other amide first ('methylamino...' before 'propyl'), at '2,N'; the N-methyl amide
    cites 'methyl' before 'oxo...propylamino...', at 'N,2', which is the lower locant set in the order of citation
    , the Blue Book, " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS"), so the N-methyl
    amide is the parent of the PIN; the split of the other amide ('N-propyl-2-{...}acetamide', its N-block out of the order of
     is no longer the name of the molecule."""
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # 'anilino...' sorts before 'methyl', so the PIN cites it first ('2-[...]-N-methylacetamide'); the name built
    # puts the N-block first, which the assembly records, so it is labelled below the PIN
    ("CNC(=O)CNCC(=O)Nc1ccccc1", "N-methyl-2-[(2-anilino-2-oxoethyl)amino]acetamide"),
    # 'benzyl' is the preferred prefix, the Blue Book) and the amine prefix is spelled 'phenylmethyl'
    ("CNC(=O)CNCC(=O)NCc1ccccc1", "N-(phenylmethyl)-2-{[2-(methylamino)-2-oxoethyl]amino}acetamide"),
])
@pytest.mark.parametrize("tier", _WIDER)
def test_the_siblings_whose_name_is_spelled_below_the_pin_are_named_and_labelled(smiles, name, tier):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # the two parents tie on and (each cites 'N,2'), and on ('N,2' both); (:22234) puts the
    # name that is earlier in alphanumerical order first: 'ethylaminooxoethylmethylamino' before 'ethylmethylaminoacetamido',
    # so the PIN is 'N-[2-(ethylamino)-2-oxoethyl]-2-(methylamino)acetamide', which no split of the engine names; the
    # name that is built cannot be shown to be the preferred one and is labelled below the PIN
    ("CCNC(=O)CNC(=O)CNC", "N-ethyl-2-[2-(methylamino)acetamido]acetamide"),
    ("CCNC(=O)CNC(=O)CO", "N-ethyl-2-(2-hydroxyacetamido)acetamide"),
])
@pytest.mark.parametrize("tier", _WIDER)
def test_amides_that_tie_on_every_locant_are_named_but_not_labelled_the_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", ["CCNC(=O)CNC(=O)CNC", "CCNC(=O)CNC(=O)CO"])
def test_the_pin_tier_gives_no_name_for_amides_that_tie_on_every_locant(smiles):
    assert _row(smiles, "pin")["tier"] == "abstain"


@pytest.mark.parametrize("tier", _WIDER)
def test_the_ester_of_the_ring_is_the_parent_and_the_ring_is_fused(tier):
    """Two acetates, a ring alcohol and a chain alcohol: the ring is the senior alcoholic component,
    the Blue Book), the other acetate a prefix. The ring is the fused hydrindane, as the strict ring namer names it
    : a fused name is senior to a von Baeyer name for an ortho-fused bicycle), not 'bicyclo[4.3.0]nonane'."""
    row = _row(_ESTER, tier)
    assert row["name"] == _ESTER_NAME, row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], _ESTER)


def test_the_pin_tier_gives_no_name_for_the_ester():
    assert _row(_ESTER, "pin")["tier"] == "abstain"


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)O[C@@H]1[C@H](C([C@H]([C@@H](C1OC(=O)C)OC(=O)C)OC(=O)C)F)OC(=O)C",
     "(1R,2S,4R,5S)-2,3,4,5-tetrakis(acetyloxy)-6-fluorocyclohexyl acetate"),
    ("CC(=O)OC[C@@H]1[C@H]([C@H]([C@H](O1)C#N)OC(=O)C)OC(=O)C",
     "(2R,3R,4S,5R)-4-(acetyloxy)-2-[(acetyloxy)methyl]-5-cyanooxolan-3-yl acetate"),
])
@pytest.mark.parametrize("tier", _WIDER)
def test_a_polyacetate_is_named_from_one_of_its_esters(smiles, name, tier):
    """The same cause on rows of the L16 scan: the alcohol fragment named '...(acetyloxy)...cyclohexane' lost its
    ester to the string rule, and the molecule was named '1,2,3,4,5-pentakis(acetyloxy)-6-fluorocyclohexane'."""
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
def test_an_ester_whose_alcohol_is_named_as_an_alcohol_keeps_its_name(tier):
    """Positive control: the alcohol fragment states its oxygen as '-ol', the string rule is trusted and nothing is read."""
    smiles = "CC(=O)OC1CCC2CCCC12"
    row = _row(smiles, tier)
    assert row["name"] == "octahydropentalen-1-yl acetate" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [_AMIDE, _ESTER])
def test_the_probe_holds_no_name_and_no_probe_is_read_back(smiles, monkeypatch):
    """The decomposition engine splits because the pipeline's name of the molecule is gone, not because a name of another
    molecule was read back: the probe of the whole molecule is empty, and no name is asked of ``name_constitution_verdict``."""
    from rdkit import Chem

    import orthonym.assembly.ester_prefix_probe as probe
    from orthonym.decomposition import engine
    wanted = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    seen, reads = [], []
    real_probe, real_verdict = probe.probe_name, engine.name_constitution_verdict
    monkeypatch.setattr(probe, "probe_name", lambda s, style: seen.append((s, real_probe(s, style))) or seen[-1][1])
    monkeypatch.setattr(engine, "name_constitution_verdict", lambda n, m: reads.append(n) or real_verdict(n, m))
    row = _row(smiles, "best-effort")
    assert row["name"] in (_AMIDE_NAME, _ESTER_NAME), row
    whole = [name for s, name in seen if Chem.MolToSmiles(Chem.MolFromSmiles(s)) == wanted]
    assert whole and not any(whole), seen
    assert not reads, reads


# ---- the sites, without a JVM -------------------------------------------------------------------------------------

def _features(pg, matches, chain):
    return SimpleNamespace(principal_group=pg, functional_groups={pg: matches}, principal_chain=chain)


@pytest.mark.parametrize("pg,matches,chain,expected", [
    # one acyl carbon (index 0 of a match) off the chain: the machinery cannot express both amides
    ("secondary_amide", [(3, 4, 2, 1), (8, 9, 10, 11)], [8, 7], True),
    ("tertiary_amide", [(3, 4, 2, 1), (8, 9, 10, 11)], [2, 1, 0], True),
    # every acyl carbon on the chain; a single amide; a principal group that is no N-substituted amide
    ("secondary_amide", [(3, 4, 2, 1), (8, 9, 10, 11)], [3, 8], False),
    ("secondary_amide", [(3, 4, 2, 1)], [8, 7], False),
    ("primary_amide", [(3, 4, 2), (8, 9, 10)], [8, 7], False),
    ("carboxylic_acid", [(3, 4, 2), (8, 9, 10)], [8, 7], False),
])
def test_principal_amides_off_chain(pg, matches, chain, expected):
    from orthonym.rules.polyfunctional_parts import principal_amides_off_chain
    assert principal_amides_off_chain(_features(pg, matches, chain)) is expected


@pytest.mark.parametrize("pg,matches,chain,expected", [
    ("ester", [(0, 1, 2)], [0, 1], True),                                      #: named by functional class
    ("secondary_amide", [(3, 4, 2, 1), (8, 9, 10, 11)], [8, 7], True),         # the amides off the chain
    ("secondary_amide", [(8, 9, 10, 11)], [8, 7], False),                      # one amide on the chain: named as before
    ("carboxylic_acid", [(3, 4, 2), (8, 9, 10)], [8, 7], False),
])
def test_the_chain_fallback_declines_an_ester_and_amides_off_its_chain(pg, matches, chain, expected):
    from orthonym.rules.polyfunctional_parts import chain_fallback_declines
    assert chain_fallback_declines(_features(pg, matches, chain)) is expected


@pytest.mark.parametrize("name,word,rule", [
    ("methanol", "methyl", "trivial"),
    ("heptan-1-ol", "heptyl", "trivial"),
    ("heptan-3-ol", "heptan-3-yl", "ol"),
    ("2-methylcyclohexan-1-ol", "2-methylcyclohexyl", "ol"),
    ("2-ethylhexyl", "2-ethylhexyl", "alkyl"),
    # the rule that reads no '-ol': the hydroxy group is cited as a prefix and stays in the word
    ("9-hydroxy-1-methyl-4-oxobicyclo[4.3.0]nonane", "9-hydroxy-1-methyl-4-oxobicyclo[4.3.0]nonyl", "alkane"),
])
def test_the_rule_that_built_the_group_word_is_named(name, word, rule):
    from orthonym.decomposition.fragment_assembly import _alcohol_to_alkyl, _alcohol_to_alkyl_by_rule
    assert _alcohol_to_alkyl_by_rule(name) == (word, rule)
    assert _alcohol_to_alkyl(name) == word


def test_a_name_that_states_no_oxygen_is_not_trusted_and_the_group_of_the_structure_comes_first():
    from orthonym.decomposition.fragment_assembly import _alcohol_group_candidates
    class_drop = "5-[7-(acetyloxy)-3-oxooctyl]-9-hydroxy-1-methyl-4-oxobicyclo[4.3.0]nonane"
    smiles = {"acid": "CC(=O)O", "alkyl": "CC(=O)OC(C)CCCC(=O)CCC1C(=O)CCC2(C)C(O)CCC12"}
    found = _alcohol_group_candidates(class_drop, smiles, _ESTER)
    assert found == [
        ("4-[7-(acetyloxy)-3-oxooctyl]-7a-methyl-5-oxooctahydro-1H-inden-1-yl", True),
        ("5-[7-(acetyloxy)-3-oxooctyl]-9-hydroxy-1-methyl-4-oxobicyclo[4.3.0]nonyl", True),
    ]


def test_a_name_that_states_its_oxygen_is_trusted_unread():
    from orthonym.decomposition.fragment_assembly import _alcohol_group_candidates
    assert _alcohol_group_candidates("cyclohexan-1-ol", {"acid": "CC(=O)O", "alkyl": "OC1CCCCC1"}, "CC(=O)OC1CCCCC1") == [
        ("cyclohexyl", False)]


@pytest.mark.parametrize("amine,prefix", [
    # (the Blue Book): the locant 1 is omitted for a saturated chain and a saturated monocycle
    ("2-methylcyclohexan-1-amine", "2-methylcyclohexyl"),
    ("(1S,2R)-2-methylcyclohexan-1-amine", "(1S,2R)-2-methylcyclohexyl"),
    ("cyclohexan-1-amine", "cyclohexyl"),
    ("2-methylpropan-1-amine", "2-methylpropyl"),
    ("4-chlorobutan-1-amine", "4-chlorobutyl"),
    ("propan-1-amine", "propyl"),
    # every other locant, and every other parent, keeps its free-valence locant
    ("propan-2-amine", "propan-2-yl"),
    ("butan-2-amine", "butan-2-yl"),
    ("prop-2-en-1-amine", "prop-2-en-1-yl"),
    ("naphthalen-1-amine", "naphthalen-1-yl"),
    ("adamantan-1-amine", "adamantan-1-yl"),
    ("bicyclo[2.2.1]heptan-2-amine", "bicyclo[2.2.1]heptan-2-yl"),
    # not one group
    ("ethane-1,2-diamine", None),
])
def test_the_n_substituent_of_an_amine_keeps_the_free_valence_as_the_alcohol_rule_does(amine, prefix):
    from orthonym.decomposition.fragment_assembly import _amine_to_prefix
    assert _amine_to_prefix(amine) == prefix


# ---- two amides that tie as parents ---------------------------------------------------------------------------------

def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _carbonyls(mol):
    from orthonym.rules.amide_parent import comparable_amide_units
    return sorted({m[0] for _, m in comparable_amide_units(mol) or ()})


@pytest.mark.parametrize("smiles,tied", [
    ("CCNC(=O)CNC(=O)CNC", True),                    # the two acetamides cite 'N,2' each
    ("C[C@@H]1CCCC[C@@H]1NC(=O)CNCC(=O)NC", True),
    ("CNC(=O)CNCC(=O)NCC", True),
    # not a tie: the keys differ (an amide with one more prefix; a longer chain)
    ("CC(=O)NCC(=O)NC", False),
    ("CCC(=O)NCC(=O)NC", False),
    # not a choice: a symmetric copy (one name), two amides on one chain, one amide
    ("CNC(=O)CNCC(=O)NC", False),
    ("CNC(=O)CCC(=O)NC", False),
    ("CC(=O)NC", False),
])
def test_two_amides_tie_as_parents_unless_the_keys_differ_or_the_molecule_maps_one_onto_the_other(smiles, tied):
    from orthonym.rules.amide_parent import tied_amide_parents
    mol = _mol(smiles)
    assert any(tied_amide_parents(mol, {c}) is not None for c in _carbonyls(mol)) is tied


@pytest.mark.parametrize("smiles,chosen_is_first", [
    #: the lower locant set in the order of citation
    ("CNC(=O)CNCC(=O)NCC", False),       # 'ethyl' before 'methylamino...' at N,2: the amide of the ethyl group has it
    ("CNC(=O)CNCC(=O)NCCC", True),       # 'methyl' before 'oxo...propylamino...' at N,2, where the other cites 2,N
    #: the same locants, the name that is earlier in alphanumerical order
    ("CCNC(=O)CNC(=O)CNC", False),       # 'ethylaminooxoethylmethylamino' is before 'ethylmethylaminoacetamido'
])
def test_the_names_of_the_prefixes_choose_the_parent(smiles, chosen_is_first):
    from orthonym.rules.amide_parent import tied_amide_parents
    from orthonym.rules.amide_tie_names import senior_amide_by_names
    mol = _mol(smiles)
    carbonyls = _carbonyls(mol)
    group = tied_amide_parents(mol, {carbonyls[0]})
    assert group is not None and len(group) == 2, group
    chosen = senior_amide_by_names(mol, [c for c, _ in group])
    assert chosen in carbonyls
    assert (chosen == group[0][0]) is chosen_is_first, (smiles, chosen, group)


def test_a_tie_that_the_prefix_names_cannot_decide_is_left_undecided():
    """A carboxamide on a ring is not worked out (its numbering decides the locants): None, so that the caller labels."""
    from orthonym.rules.amide_tie_names import senior_amide_by_names
    mol = _mol("O=C(NC)c1ccccc1C(=O)NCC")
    carbonyls = [m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))]
    assert len(carbonyls) == 2 and senior_amide_by_names(mol, carbonyls) is None


# ---- the N-acyl float on the best-effort rescue's name of the amine fragment -----------------------------------------

_LIPID = ("CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])"
          "OC[C@H](NC(=O)CCCCCCCCCCCCCCC)C(=O)[O-]")


def test_the_n_acyl_float_is_not_built_on_the_rescue_name_of_an_amine_fragment():
    """The amine fragment of the N-acyl serine phospholipid is named by the best-effort rescue as an oxo-alkane that cites its
    amino group inside a substituent ('(9Z)-1-{[...2-amino-...propoxy]...}-1-oxooctadec-9-ene'); '(9Z)-N-hexadecanoyl-1-{...}-1-
    oxooctadec-9-ene' put a locant N in front of a parent that has no nitrogen. The molecule keeps the name with the amide
    cited as the prefix 'hexadecanamido', the Blue Book)."""
    row = _row(_LIPID, "best-effort")
    assert "N-hexadecanoyl" not in row["name"], row
    assert "2-hexadecanamido-3-oxido-3-oxopropoxy" in row["name"], row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], _LIPID)
