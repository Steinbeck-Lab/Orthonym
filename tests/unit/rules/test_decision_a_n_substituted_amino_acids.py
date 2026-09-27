"""User decision A (2026-09-26), part 1: an amino acid substituted on its NITROGEN
takes the systematic substitutive name at the PIN tier; the retained amino-acid names
stay PIN-tier names only when the nitrogen is unsubstituted (decision D-a, narrowed).
Controller rulings: O-/S-substituted amino acids are not covered; carnitine loses the
PIN label; citrulline stays (Table 10.5, the Blue Book).

Blue Book basis:

* "Substituents of the types -NH-CO-R and -NH-SO2-R"
  (the Blue Book): -NH-CO-R is named "(1) substitutively, by using a prefix
  formed by changing the final letter 'e' in the complete name of the amide to 'o'"
  (:32995) or "(2)... 'acylamino' prefixes" (:32996); "Method (1) generates
  preferred IUPAC names." (:32998), "(1) 4-formamidobenzoic acid (PIN)" (:33000). An
  N-substituted glycine is printed on the acetic acid parent at:33213:
  "[(methanesulfinothioyl)amino]acetic acid (PIN)".
* "Systematic substitutive names" (:54247),:54251: "When not denoted by a
  retained name, amino acids receive systematic substitutive names constructed by
  applying the principles, rules and conventions of substitutive nomenclature."
* "INTRODUCTION" (:50939),:50943: "Preferred IUPAC names (PINs) are not
  identified for the compounds in this Chapter." A trivial name is a PIN-tier name only
  with Blue Book evidence; tricine, taurocyamine, the opines and carnitine
  have 0 Blue Book hits (control 'glycine' 30, 'citrulline' 1).
* "Derivatives formed by substitution" (:54478),:54480: "Retained names are
  used to indicate carbon, nitrogen, oxygen and sulfur substitution" -- general
  nomenclature. Where the engine cannot build the systematic name yet, the N-acyl name
  is kept, labelled below the PIN tier (honest demotion, never an abstention).
* (:7232): parentheses around compound prefixes, so the demoted
  'N-4-hydroxyoctanoylglycine' is spelled 'N-(4-hydroxyoctanoyl)glycine'.

Part 2 (2026-09-27) builds the systematic names part 1 could only demote:

* a DECORATED acyl on an acyclic parent is the method (1) amido prefix
  ('[(6E)-4-hydroxyoct-6-enamido]acetic acid'; substituent_enumerator._name_amino_branch
  through substituent_naming.acyl_amido_prefix_from_branch);
* an acyl on a ring N of a ring carrying the carboxylic-acid suffix is an acyl prefix:
  the 'hidden amide' is a pseudoketone,:33125), a ketone is "cited as
  prefixes" under a suffix group,:29585), and acyl names denote those
  substituents unchanged,:31376; '2-acetylbenzoic acid (PIN)':31378):
  '(2S)-1-acetylpyrrolidine-2-carboxylic acid' (rules.heterocycles);
* the N,N-disubstituted amino prefix with a decorated branch
  ('[carbamimidoyl(methyl)amino]acetic acid', creatine), with the guanidine FG prefix no
  longer double-citing a branch the walk names whole;
* a located descriptor in a free-valence-numbered substituent ('(1R)-1-carboxyethyl',
   :44624) and no extra mark level around a braced prefix:7444):
  lysopine '(2S)-6-amino-2-{[(1R)-1-carboxyethyl]amino}hexanoic acid';
* creatine and lysopine then get pin:false rows (the sarcosine precedent).

Carnitine stays as part 1 left it: its PIN is the '-aminiumyl' form
('(3R)-4-(N,N-dimethylmethanaminiumyl)-3-hydroxybutanoate', after:42473 and
 :42296/:42299), which OPSIN 2.9.0 cannot parse, so no carnitine name can be
verified as a PIN.

Every asserted name is checked by an independent OPSIN 2.9.0 full-InChIKey round trip
(``tests.support.rt_assert.assert_full_rt``), outside the engine.
"""
import json
from pathlib import Path

import pytest

from orthonym import Orthonym
from tests.support.rt_assert import assert_full_rt

DENIED = ["tricine", "taurocyamine", "hypotaurocyamine", "strombine", "alanopine",
          "beta-alanopine", "octopine", "octopinic acid", "nopaline", "tauropine"]


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


def _pin_rows():
    import orthonym.data as data_pkg
    path = Path(data_pkg.__file__).parent / "iupac_2013_pin_list.json"
    with open(path) as fh:
        return json.load(fh)["entries"]


# --- 1. the trivial table names leave the PIN tier ---------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("name", DENIED)
def test_deny_row_withdraws_the_table_name(name):
    from orthonym.data.amino_acids import (
        GENERAL_ONLY_AMINO_ACIDS,
        NON_STANDARD_AMINO_ACIDS,
        STANDARD_AMINO_ACIDS,
    )
    rows = [e for e in _pin_rows() if e["name"].lower() == name]
    assert len(rows) == 1 and rows[0]["pin"] is False
    assert rows[0]["citation"] == "P-103.1.1.3"
    assert name not in {n.lower() for n in STANDARD_AMINO_ACIDS.values()}
    assert name not in {n.lower() for n in NON_STANDARD_AMINO_ACIDS.values()}
    # demoted, not deleted (the sarcosine precedent)
    assert name in {n.lower() for n in GENERAL_ONLY_AMINO_ACIDS.values()}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # was 'tricine' (pin_verified)
    ("OCC(CO)(CO)NCC(=O)O",
     "{[1,3-dihydroxy-2-(hydroxymethyl)propan-2-yl]amino}acetic acid"),
    # was 'taurocyamine' (pin_verified); 'carbamimidoylamino (preferred prefix)':34268
    ("N=C(N)NCCS(=O)(=O)O", "2-(carbamimidoylamino)ethane-1-sulfonic acid"),
])
def test_systematic_name_at_the_pin_tier(namer, smiles, expected):
    res = namer.name_tiered(smiles)
    assert res["name"] == expected
    assert res["tier"] == "pin_verified" and res["is_pin"] is True
    assert_full_rt(res["name"], smiles)


# --- 2. the N-acyl float onto a non-suffix nitrogen --------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # was 'N-acetamidoacetylglycine' (pin_verified)
    ("CC(=O)NCC(=O)NCC(=O)O", "(2-acetamidoacetamido)acetic acid"),
    # was 'N-propanamidoacetylglycine' (pin_verified)
    ("CCC(=O)NCC(=O)NCC(=O)O", "(2-propanamidoacetamido)acetic acid"),
    # was 'N-[(2S)-2-acetamidopropanoyl](2S)-2-aminopropanoic acid' (pin_verified)
    ("CC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",
     "(2S)-2-[(2S)-2-acetamidopropanamido]propanoic acid"),
])
def test_float_refused_systematic_name_built(namer, smiles, expected):
    res = namer.name_tiered(smiles)
    assert res["name"] == expected
    assert res["tier"] == "pin_verified" and res["is_pin"] is True
    assert_full_rt(res["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    # the ion keeps the acid's status (ions._acid_name_to_carboxylate,
    # ions.apply_ion_suffix_to_name): the 'N-acyl' float on a charged amine parent
    # ('2-aminoethane-1-sulfonate', 'glycinate', '2-aminoethyl sulfate') has no
    # systematic replacement yet
    "CCCCC/C=C\\C[C@@H](/C=C/C=C\\C/C=C\\CCCC(=O)NCCS(=O)(=O)[O-])OO",
    "CCCCC[C@@H](/C=C/C=C\\C/C=C\\C/C=C\\CCCC(=O)NCC(=O)[O-])OO",
    "CCCCCCCCCCCCCC(=O)NCCOS(=O)(=O)[O-]",
])
def test_float_the_engine_cannot_replace_yet_is_demoted(namer, smiles):
    """Breadth never drops and the PIN tier never labels a non-PIN pin_verified: the
    name still ships (RT-exact) but below the PIN tier."""
    res = namer.name_tiered(smiles)
    assert res["is_pin"] is False and res["tier"] != "pin_verified", res
    assert_full_rt(res["name"], smiles)


@pytest.mark.unit
@pytest.mark.parametrize("amine_smiles,amine_name,is_suffix_n", [
    ("NC1CCCCC1", "cyclohexanamine", True),          # the -amine N
    ("Nc1ccccc1", "aniline", True),
    ("CC(N)=O", "acetamide", True),                  # the -amide N
    ("NS(=O)(=O)c1ccccc1", "benzenesulfonamide", True),
    ("CNC1CCCCC1", "N-methylcyclohexanamine", True),
    ("NC1CCCCC1", None, True),                       # structure alone
    ("NCC(=O)O", "glycine", False),                  # alpha-amino N, acid principal
    ("NCC(=O)O", None, False),
    ("N[C@@H](C)C(=O)O", "(2S)-2-aminopropanoic acid", False),
    ("NCCS(=O)(=O)O", "2-aminoethane-1-sulfonic acid", False),
    ("Nc1ccc(C(=O)O)cc1", "4-aminobenzoic acid", False),
    ("NCCO", "2-aminoethan-1-ol", False),            # the alcohol is principal
    ("OC(=O)C1CCCN1", "pyrrolidine-2-carboxylic acid", False),  # ring N (N-1)
    ("C1CCNC1", None, False),                        # ring N
    # the structural principal group misses these; the parent name cites 'amino'
    ("NCC(=O)[O-]", "glycinate", False),
    ("NCCOS(=O)(=O)[O-]", "2-aminoethyl sulfate", False),
])
def test_suffix_nitrogen_classification(amine_smiles, amine_name, is_suffix_n):
    from orthonym.decomposition.engine import _nacyl_float_n_is_suffix_nitrogen
    assert _nacyl_float_n_is_suffix_nitrogen(amine_smiles, amine_name) is is_suffix_n


@pytest.mark.unit
def test_gate_refuses_in_scope_and_demotes_outside():
    from orthonym.decomposition.engine import (
        _NACYL_FLOAT_REFUSALS,
        gate_nonsuffix_nacyl_float,
        nacyl_float_refusing,
    )
    from orthonym.metrics import provenance as pv
    # a suffix nitrogen: unchanged, nothing recorded
    pv.clear_provenance()
    assert gate_nonsuffix_nacyl_float("NC1CCCCC1", "N-acetylcyclohexanamine") == \
        "N-acetylcyclohexanamine"
    assert not pv.get_provenance()["non_pin_fragments"]
    # outside a scope: kept, recorded non-PIN
    assert gate_nonsuffix_nacyl_float("NCC(=O)O", "N-acetylglycine") == "N-acetylglycine"
    assert "N-acetylglycine" in pv.get_provenance()["non_pin_fragments"]
    # inside a scope: withheld, noted for the caller's fallback
    refused = []
    tok = _NACYL_FLOAT_REFUSALS.set(refused)
    try:
        assert nacyl_float_refusing() is True
        assert gate_nonsuffix_nacyl_float("NCC(=O)O", "N-propanoylglycine") is None
    finally:
        _NACYL_FLOAT_REFUSALS.reset(tok)
    assert refused == ["N-propanoylglycine"] and nacyl_float_refusing() is False
    pv.clear_provenance()


# --- 3. carnitine: kept, never a PIN-tier name --------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[N+](C)(C)C[C@H](O)CC(=O)[O-]", "L-carnitine"),
    ("C[N+](C)(C)CC(O)CC(=O)[O-]", "carnitine"),
])
def test_carnitine_is_labelled_below_the_pin_tier(namer, smiles, expected):
    """0 Blue Book hits; the PIN tier builds no RT-exact systematic name yet (the L
    zwitterion abstains without the retained name, the stereo-free one gives the
    malformed '4-(trimethylazaniumyl)3-hydroxybutanoate'), so the name is kept and
    labelled non-PIN rather than withdrawn."""
    res = namer.name_tiered(smiles)
    assert res["name"] == expected
    assert res["is_pin"] is False and res["tier"] != "pin_verified"
    assert_full_rt(res["name"], smiles)


@pytest.mark.opsin_gate
def test_d_carnitine_from_the_amino_acid_table_is_labelled_below_the_pin_tier(namer):
    smiles = "C[N+](C)(C)C[C@@H](O)CC(=O)[O-]"
    res = namer.name_tiered(smiles)
    assert res["is_pin"] is False and res["tier"] != "pin_verified", res
    assert_full_rt(res["name"], smiles)


# --- 4. part 2 (2026-09-27): the systematic names part 1 could only demote ----------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # (1) a DECORATED acyl on an acyclic parent: the method (1) amido prefix
    #:32995/:32998). Was 'N-[(6E)-4-hydroxyoct-6-enoyl]glycine'
    # (stereo_benchmark row 121), 'N-(4-hydroxyoctanoyl)glycine', 'N-[(6E)-4-
    # hydroxyoct-6-enoyl](2S)-2-aminopropanoic acid', 'N-[(2S)-2-acetamido-3-
    # hydroxypropanoyl]glycine' -- all demoted by part 1.
    ("C/C=C/CC(O)CCC(=O)NCC(=O)O", "[(6E)-4-hydroxyoct-6-enamido]acetic acid"),
    ("CCCCC(O)CCC(=O)NCC(=O)O", "(4-hydroxyoctanamido)acetic acid"),
    ("C/C=C/CC(O)CCC(=O)N[C@@H](C)C(=O)O",
     "(2S)-2-[(6E)-4-hydroxyoct-6-enamido]propanoic acid"),
    ("CC(=O)N[C@@H](CO)C(=O)NCC(=O)O",
     "[(2S)-2-acetamido-3-hydroxypropanamido]acetic acid"),
    ("CC(O)CC(=O)NCC(=O)O", "(3-hydroxybutanamido)acetic acid"),
    ("CCCCC(O)CCC(=O)NCCS(=O)(=O)O", "2-(4-hydroxyoctanamido)ethane-1-sulfonic acid"),
    # (2) an acyl on the ring N of a ring carrying the carboxylic-acid suffix: an
    # acyl PREFIX:33125,:29585,:31376). Was
    # 'N-acetyl(2S)-pyrrolidine-2-carboxylic acid' and 'N-[(2S)-2-methyl-3-
    # sulfanylpropanoyl](2S)-pyrrolidine-2-carboxylic acid' (demoted), and PIN
    # abstentions for the stereo-free forms.
    ("CC(=O)N1CCC[C@H]1C(=O)O", "(2S)-1-acetylpyrrolidine-2-carboxylic acid"),
    ("C[C@H](CS)C(=O)N1CCC[C@H]1C(=O)O",
     "(2S)-1-[(2S)-2-methyl-3-sulfanylpropanoyl]pyrrolidine-2-carboxylic acid"),
    ("CC(=O)N1CCCC1C(=O)O", "1-acetylpyrrolidine-2-carboxylic acid"),
    ("CC(CS)C(=O)N1CCCC1C(=O)O",
     "1-(2-methyl-3-sulfanylpropanoyl)pyrrolidine-2-carboxylic acid"),
    ("O=CN1CCCC1C(=O)O", "1-formylpyrrolidine-2-carboxylic acid"),
    ("NCC(=O)N1CCCC1C(=O)O", "1-(aminoacetyl)pyrrolidine-2-carboxylic acid"),
    ("CC(=O)N1CCNC(C1)C(=O)O", "4-acetylpiperazine-2-carboxylic acid"),
    ("CC(C)[C@H]1N([C@H](CS1)C(=O)[O-])C(=O)C2=CC(=CC=C2)F",
     "(2S,4S)-3-(3-fluorobenzoyl)-2-(propan-2-yl)-1,3-thiazolidine-4-carboxylate"),
    # (3) creatine: the N,N-disubstituted amino prefix with a decorated branch
    # ('carbamimidoylamino (preferred prefix)':34268;:7272). Was
    # 'creatine' (pin_verified).
    ("CN(CC(=O)O)C(=N)N", "[carbamimidoyl(methyl)amino]acetic acid"),
    ("CN(CC(=O)O)C(=N)N.Cl",
     "[carbamimidoyl(methyl)amino]acetic acid—hydrogen chloride (1/1)"),
    # (4) lysopine: the located substituent descriptor:44624,
    # '[(1R)-1-chloropropyl]benzene (PIN)':44668) and no extra mark level
    #:7444). Was 'lysopine' (pin_verified).
    ("C[C@@H](N[C@@H](CCCCN)C(=O)O)C(=O)O",
     "(2S)-6-amino-2-{[(1R)-1-carboxyethyl]amino}hexanoic acid"),
])
def test_part2_systematic_name_at_the_pin_tier(namer, smiles, expected):
    res = namer.name_tiered(smiles)
    assert res["name"] == expected, res
    assert res["tier"] == "pin_verified" and res["is_pin"] is True, res
    assert_full_rt(res["name"], smiles)


@pytest.mark.opsin_gate
def test_amido_prefix_of_a_substituted_acetic_acid_is_the_acetamido_pin(namer):
    """The acid producer used to spell this substituted acetic acid '...-2-phenyl
    ethanoic acid' (the general-nomenclature alternative, the Blue Book:
    29725), so the name carried '...ethanamido' and was labelled below the PIN
    tier. The acid now takes the retained parent with its alpha descriptor cited
    bare (TRIAGE j12 findings 2/4/7), and the amido prefix is built on the
    retained 'acetamide', which keeps its locants:32995;
    :7304 "Locants are required... for example acetamide"): the PIN ships
    pin_verified. The label rule for a prefix from a non-PIN acid spelling is
    covered in test_j12_verify_fixes.py."""
    smiles = ("CCOC(=O)C1=C(SC=C1C2=CC=CC=C2)NC(=O)[C@H](C3=CC=CC=C3)"
              "SC4=NN=NN4C5=C(C=C(C=C5)C)C")
    res = namer.name_tiered(smiles)
    assert res["name"] == (
        "ethyl 2-[(2S)-2-{[1-(2,4-dimethylphenyl)-1H-tetrazol-5-yl]sulfanyl}-2-"
        "phenylacetamido]-4-phenylthiophene-3-carboxylate"), res
    assert res["is_pin"] is True and res["tier"] == "pin_verified", res
    assert_full_rt(res["name"], smiles)


@pytest.mark.unit
@pytest.mark.parametrize("acid,acyl", [
    ("acetic acid", "acetyl"),                       #:30442
    ("formic acid", "formyl"),                       #:30444
    ("benzoic acid", "benzoyl"),                     #:30446
    ("propanoic acid", "propanoyl"),                 #:30608
    ("(2S)-2-methyl-3-sulfanylpropanoic acid", "(2S)-2-methyl-3-sulfanylpropanoyl"),
    ("cyclohexanecarboxylic acid", "cyclohexanecarbonyl"),  #:30624
    ("chloroacetic acid", "chloroacetyl"),           # '(chloroacetyl)oxyl (PIN)':40694
    # fail closed: a poly-acid needs a divalent / multiplied prefix; a retained
    # amino-acid name or a functional-replacement acid is not converted
    ("butanedioic acid", None),
    ("benzene-1,2-dicarboxylic acid", None),
    ("oxydiacetic acid", None),
    ("L-alanine", None),
    ("propanethioic S-acid", None),
    ("ethaneperoxoic acid", None),
    ("sodium acetate", None),
])
def test_acid_name_to_acyl_prefix(acid, acyl):
    from orthonym.assembly.substituent_naming import acid_name_to_acyl_prefix
    assert acid_name_to_acyl_prefix(acid) == acyl


@pytest.mark.unit
@pytest.mark.parametrize("smiles,carbonyl_c,attach,expected", [
    # the acyl group of CC(=O)N1CCCC1C(=O)O: atoms 0,1,2 on ring N 3
    ("CC(=O)N1CCCC1C(=O)O", 1, 3, "acetyl"),
    # a carbamate-type (O-rooted) carbonyl is not an acyl group
    ("COC(=O)N1CCCC1C(=O)O", 2, 4, None),
])
def test_acyl_prefix_from_branch(smiles, carbonyl_c, attach, expected):
    from rdkit import Chem
    from orthonym.assembly.substituent_naming import acyl_prefix_from_branch
    mol = Chem.MolFromSmiles(smiles)
    ring = set(mol.GetRingInfo().AtomRings()[0])
    acyl, stack = set(), [carbonyl_c]
    while stack:
        a = stack.pop()
        if a in acyl or a == attach or a in ring:
            continue
        acyl.add(a)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors())
    assert acyl_prefix_from_branch(mol, carbonyl_c, attach, acyl) == expected


@pytest.mark.unit
def test_acyl_prefix_from_branch_refuses_an_open_fragment():
    """Every acyl atom must stay inside the group: a fragment that leaves an atom
    out would be named as a different acid."""
    from rdkit import Chem
    from orthonym.assembly.substituent_naming import acyl_prefix_from_branch
    mol = Chem.MolFromSmiles("CCC(=O)N1CCCC1C(=O)O")
    assert acyl_prefix_from_branch(mol, 2, 4, {1, 2, 3}) is None   # C0 left out
    assert acyl_prefix_from_branch(mol, 2, 4, {0, 1, 2, 3}) == "propanoyl"


@pytest.mark.unit
@pytest.mark.parametrize("smiles,principal_group,expected", [
    ("CC(=O)N1CCCC1C(=O)O", "carboxylic_acid", "acetyl"),
    ("NCC(=O)N1CCCC1C(=O)O", "carboxylic_acid", "(aminoacetyl)"),
    # deny by default: no ring carboxylic acid, or another principal group
    ("CC(=O)N1CCC(O)CC1", "secondary_alcohol", None),
    ("CC(=O)N1CCC(C#N)CC1", "nitrile", None),
    ("CC(=O)N1CCC(CC1)CC(=O)O", "carboxylic_acid", None),   # the acid is on a chain
])
def test_ring_n_acyl_prefix_scope(smiles, principal_group, expected):
    from rdkit import Chem
    from orthonym.rules.heterocycles import _ring_n_acyl_prefix_under_ring_acid
    mol = Chem.MolFromSmiles(smiles)
    ring = set(mol.GetRingInfo().AtomRings()[0])
    n_idx = next(a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == 'N' and a.GetIdx() in ring)
    c_idx = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                 if nb.GetIdx() not in ring)
    acyl, stack = set(), [c_idx]
    while stack:
        a = stack.pop()
        if a in acyl or a == n_idx:
            continue
        acyl.add(a)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors())
    assert _ring_n_acyl_prefix_under_ring_acid(
        mol, n_idx, c_idx, sorted(acyl), ring, principal_group) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # the guanidine FG prefix still names a branch that IS the guanidine
    ("NC(=N)NCC(=O)O", "(carbamimidoylamino)acetic acid"),
    # a decorated second N-substituent: bare when first cited, enclosed after
    ("CCN(CC(=O)O)C(=N)N", "[carbamimidoyl(ethyl)amino]acetic acid"),
    ("CN(CCO)CC(=O)O", "[(2-hydroxyethyl)(methyl)amino]acetic acid"),
])
def test_n_n_disubstituted_amino_prefix(namer, smiles, expected):
    res = namer.name_tiered(smiles)
    assert res["name"] == expected, res
    assert res["tier"] == "pin_verified" and res["is_pin"] is True, res
    assert_full_rt(res["name"], smiles)


@pytest.mark.unit
def test_braced_prefix_takes_no_extra_mark_level():
    """ (the Blue Book): '{}' is already the third level; a prefix
    that arrives fully enclosed in braces is cited as it is."""
    from orthonym.assembly.naming_utils import format_substituent_prefix
    assert format_substituent_prefix(
        "{[(1R)-1-carboxyethyl]amino}", [2], 1) == "2-{[(1R)-1-carboxyethyl]amino}"
    assert format_substituent_prefix(
        "[(1R)-1-carboxyethyl]amino", [2], 1) == "2-{[(1R)-1-carboxyethyl]amino}"
    assert format_substituent_prefix("2-methylpropyl", [3], 1) == "3-(2-methylpropyl)"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["creatine", "lysopine"])
def test_part2_deny_rows(name):
    from orthonym.data.amino_acids import (
        GENERAL_ONLY_AMINO_ACIDS,
        NON_STANDARD_AMINO_ACIDS,
        STANDARD_AMINO_ACIDS,
    )
    rows = [e for e in _pin_rows() if e["name"].lower() == name]
    assert len(rows) == 1 and rows[0]["pin"] is False
    assert rows[0]["citation"] == "P-103.1.1.3"
    assert name not in {n.lower() for n in STANDARD_AMINO_ACIDS.values()}
    assert name not in {n.lower() for n in NON_STANDARD_AMINO_ACIDS.values()}
    assert name in {n.lower() for n in GENERAL_ONLY_AMINO_ACIDS.values()}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("acid_name,acid_smiles,verified", [
    ("acetic acid", "CC(=O)O", True),
    ("(2S)-2-methyl-3-sulfanylpropanoic acid", "C[C@H](CS)C(=O)O", True),
    # the wrong enantiomer, and an atom-dropping name a fragment namer returned at the
    # best-effort tier (the 3-methylbutyl group lost)
    ("(2R)-2-methyl-3-sulfanylpropanoic acid", "C[C@H](CS)C(=O)O", False),
    ("((4-chlorophenoxy)acetylamino)acetic acid",
     "OC(=O)CN(CCC(C)C)C(=O)COc1ccc(Cl)cc1", False),
    ("not a name", "CC(=O)O", False),
])
def test_fragment_acid_name_verified(acid_name, acid_smiles, verified):
    from orthonym.assembly.substituent_naming import fragment_acid_name_verified
    assert fragment_acid_name_verified(acid_name, acid_smiles) is verified


@pytest.mark.opsin_gate
def test_amido_prefix_refuses_an_unverified_acid_name(monkeypatch):
    """The decorated-acyl amido path trusts no acid name it has not verified: a
    fragment namer that returns an atom-dropping acid name yields no prefix."""
    from rdkit import Chem
    import orthonym.assembly.fragment_naming as fn
    from orthonym.assembly.substituent_naming import acyl_amido_prefix_from_branch
    mol = Chem.MolFromSmiles("CCCCC(O)CCC(=O)NCC(=O)O")
    branch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]     # the acyl and its N (atom 10)
    assert acyl_amido_prefix_from_branch(
        mol, 10, 8, branch, verify_acid=True) == "4-hydroxyoctanamido"
    monkeypatch.setattr(fn, "name_fragment_recursively",
                        lambda smi, *a, **k: "octanoic acid")   # the OH dropped
    assert acyl_amido_prefix_from_branch(mol, 10, 8, branch, verify_acid=True) is None
