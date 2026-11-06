"""An ester molecule is named as an ester, not on a parent hydride that cites the ester in a prefix (lane/l17 item 2).

 "General methodology" (the Blue Book): "All preferred IUPAC names for esters are named by functional
class nomenclature." SENIORITY ORDER FOR CLASSES, Table 4.1 (:18158): class 9 esters rank above class 11 amides,
class 17 hydroxy compounds and class 19 amines, so the ester is the principal characteristic group of these molecules and
the name that holds it only in an 'acyloxy' or 'alkoxy...oxo' prefix is a name of a lower class "Esters
cited as prefixes",:31696: for an ester beside a group senior to it).

Two causes, one chain. (1) ``rules.esters._build_ester_acid_word`` named the acid word of 'ethyl cyclohexyl
butanedioate' as 'ethyl 4-(cyclohexyloxy)butanoate': the =O of the second carboxyl carbon on the acid chain is neither the
suffix nor an 'oxo' prefix of that word, so the word named an ether,:31769: an ester of one polyacid is
named from the acid component). (2) The decomposition engine keeps the pipeline's name when it is "good enough" and
splits the molecule only when it is poor; at a wider tier that name was a name of another molecule (or an ester cited in a
prefix, which the probe already read as no name), and the engine kept it, so the ester was left to the ring producers that
cite it in a prefix ('2-({[4-(...oxy)-4-oxobutanoyl]oxy}methyl)-3,4,5-trihydroxypiperidine'). The probe now reads a name
that OPSIN reads as another molecule as no name, for an ester molecule at a wider tier; the PIN tier is unchanged.

Fix round of the review of that change:

* (I2) the acid word keeps the =O of every other carbon of its chain as a prefix ('oxo'), not only of an ester: 'ethyl
  4-amino-4-oxobutanoate', never 'ethyl 4-aminobutanoate'; an anhydride and a carbamate are no chain ester;
* (I1) a name the fragment cache holds was computed with no cycle met on the visited set, or is read only where the same
  SMILES are on it again; the wrong 'prop-2-enoic acid' name of an acid fragment came of ``polyfunctional_parts``
  keeping the ring's tertiary alcohol as a chain group ('3-hydroxyprop-2-enoic acid');
* (I3) the probe rejects a name only when OPSIN was asked and does not read it as the molecule;
* (I4) 'ethyl (2,3-dihydroxypropyl) butanedioate' is the PIN of an ester of one polyacid, the Blue Book:
  31769), not the glyceride word '2,3-dihydroxypropyl 4-ethoxy-4-oxobutanoate' of the lipid assembler.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_WIDER = ["valid", "best-effort"]

# the target row: a steroid ester of a butanedioic acid whose other carboxyl is esterified with a piperidine-triol
_S = ("C[C@H](CCCC(C)C)C1CCC2[C@@]1(CCC3C2CCC4[C@@]3(CC[C@@H](C4)OC(=O)CCC(=O)OC[C@@H]5[C@H]([C@@H]([C@H](CN5)O)O)O)C)C")
_S_NAME = ("(3S,10S,13R)-10,13-dimethyl-17-[(2R)-6-methylheptan-2-yl]hexadecahydro-1H-cyclopenta[a]phenanthren-3-yl "
           "4-oxo-4-{[(2R,3R,4R,5S)-3,4,5-trihydroxypiperidin-2-yl]methoxy}butanoate")


@pytest.mark.parametrize("tier", _WIDER)
def test_the_target_is_named_as_an_ester(tier):
    """ (:31663): the name is the functional class name of the ester, the acid word holding the second ester
    as 'oxo' + alkoxy; it is no PIN (an ester of one polyacid has the PIN of,:31769, which the engine
    does not build for these two alcohols), so its label is below the PIN."""
    row = tier_row(_S, tier)
    assert row["name"] == _S_NAME, row
    assert row["tier"] != "pin_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], _S)


def test_the_pin_tier_gives_no_name_for_the_target():
    row = tier_row(_S, "pin")
    assert row["tier"] == "abstain" and not row["is_pin"], row


# alcohol components with groups of their own: an ester of one polyacid with two different alcohols is named from its acid
# component,:31769: 'different organyl groups are cited in alphanumerical order'), the PIN, at every tier
_POLYACID_PINS = [
    ("CCOC(=O)CCC(=O)OCC1NCCCC1O", "ethyl (3-hydroxypiperidin-2-yl)methyl butanedioate"),
    ("CC(C)OC(=O)CCC(=O)OCC1NCCCC1O", "(3-hydroxypiperidin-2-yl)methyl propan-2-yl butanedioate"),
]


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
@pytest.mark.parametrize("smiles,name", _POLYACID_PINS)
def test_an_ester_of_one_polyacid_is_named_from_the_acid_component(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
def test_the_pin_tier_names_the_diester_of_one_acid_with_two_alcohols(tier):
    """Re-asserted 2026-11-05 (lane/l17 item 5, change-asserted-value): 'ethyl cyclohexyl butanedioate' was a no-ether
    abstention test of the PIN tier (it named nothing: the ring cascade's acyloxy name was suppressed). The ring cascade now leaves
    an ester molecule to the producers that name it by its class at the PIN tier, and the name is the PIN of
     "Fully esterified acids derived from a single acid" (the Blue Book): "different organyl groups
    are cited in alphanumerical order (see ", 'ethyl methyl butanedioate (PIN)' (:31779). 'cyclohexyl' is cited
    before 'ethyl', alphanumerical order). The name is read back by OPSIN to the full InChIKey; the same shape
    with 'methyl' for the cyclohexyl is a control below and passes at every tier."""
    smiles = "CCOC(=O)CCC(=O)OC1CCCCC1"
    row = tier_row(smiles, tier)
    assert row["name"] == "cyclohexyl ethyl butanedioate", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", ["CCOC(=O)CCC(=O)OC1CCCCC1", "CCOC(=O)CCC(=O)OCC1NCCCC1O"])
def test_a_chain_acid_word_declines_when_it_ends_in_a_second_ester(smiles):
    """The acid word of either ester of the diester would name the other as an alkoxy prefix without its =O."""
    from orthonym.rules.esters import name_ester
    mol = Chem.MolFromSmiles(smiles)
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))
    assert len(matches) == 2
    for match in matches:
        assert name_ester(mol, match) is None


# I2: the =O of a second chain carbonyl that is no ester (an amide, a thioester, a hydroxamic acid, a ketone) is a prefix
# of the acid word ('4-amino-4-oxobutanoate',: the ester, class 9, outranks the amide, class 11), and an
# anhydride, a carbamate or a carbonate is no chain ester: each word was a name of another molecule
_SECOND_CARBONYL = [
    ("CCOC(=O)CCC(=O)NC", "ethyl 4-(methylamino)-4-oxobutanoate"),
    ("CCOC(=O)CCC(=O)N", "ethyl 4-amino-4-oxobutanoate"),
    ("CCOC(=O)CCC(=O)SCC", "ethyl 4-(ethylsulfanyl)-4-oxobutanoate"),
    ("CCOC(=O)C(C)C(=O)C(C)C", "ethyl 2,4-dimethyl-3-oxopentanoate"),
    ("CCOC(=O)CC(=O)C", "ethyl 3-oxobutanoate"),
    ("CCOC(=O)CCC(=O)C1CCCCC1", "ethyl 4-cyclohexyl-4-oxobutanoate"),
]


@pytest.mark.parametrize("smiles,name", _SECOND_CARBONYL)
def test_the_second_chain_carbonyl_is_cited_as_oxo(smiles, name):
    """``name_ester`` itself, then the PIN tier (a PIN: the amide, thioester and ketone are junior to the ester)."""
    from orthonym.rules.esters import name_ester
    mol = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))[0]
    assert name_ester(mol, match) == name
    assert_full_rt(name, smiles)
    row = tier_row(smiles, "pin")
    assert row["name"] == name and row["tier"] == "pin_verified", row


@pytest.mark.parametrize("smiles", [
    "CCOC(=O)C(C)C(=O)N(C)O",      # a hydroxamic acid at the far end: the prefix 'hydroxy(methyl)amino' is not built
    "CCOC(=O)CCC(=O)OC(=O)C",      # the acetic anhydride of ethyl hydrogen succinate: 'hexyl acetate' named its alcohol word
    "CCOC(=O)N", "COC(=O)NC",      # a carbamate has no chain: 'methyl formate' counted its carbons
])
def test_name_ester_declines_where_the_acid_word_would_drop_an_atom(smiles):
    from orthonym.rules.esters import name_ester
    mol = Chem.MolFromSmiles(smiles)
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")):
        assert name_ester(mol, match) is None


def test_the_acyloxy_prefix_cites_the_second_carbonyl_and_never_drops_a_group():
    """``name_ester_as_prefix``: the =O of the other carboxyl is an 'oxo' prefix of the acyl word, the chain starts at the
    ester's own carbonyl carbon (the lower atom index is the other one here), and an acyl chain with a group of its own
    that the word cannot cite declines."""
    from orthonym.rules.esters import name_ester_as_prefix
    mol = Chem.MolFromSmiles("CCOC(=O)CCC(=O)OC1CCCCC1")
    own = [m for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")) if m[3] == 10][0]
    assert name_ester_as_prefix(mol, own) == "(4-ethoxy-4-oxobutanoyl)oxy"
    for smiles in ("NCCC(=O)OC", "ClCCC(=O)OC", "OCCC(=O)OC", "NC(=O)CCC(=O)OC"):
        mol = Chem.MolFromSmiles(smiles)
        assert name_ester_as_prefix(mol, mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))[0]) is None


# an ester whose best-effort name cited the ester in a prefix of a ring or chain parent (class sample of the 182)
_CLASS = [
    ("CC(=O)OCC1C(C(CO1)OC)OC(=O)C", "2-[(acetyloxy)methyl]-4-methoxyoxolan-3-yl acetate"),
    ("CC(=O)O[C@H]1C(=C)C=C(C=C1OC)OC", "(1S)-2,4-dimethoxy-6-methylidenecyclohexa-2,4-dien-1-yl acetate"),
    ("CCOC(=O)C(N)C(=O)OCC", "ethyl 2-amino-3-ethoxy-3-oxopropanoate"),
    ("CCOC(=O)C(C)C(=O)N(C)O", "ethyl 3-[hydroxy(methyl)amino]-2-methyl-3-oxopropanoate"),
    ("COC(=O)c1ccc([C@](C)(O)CCCC(C)(C)O)c(O)c1", "methyl 4-[(2R)-2,6-dihydroxy-6-methylheptan-2-yl]-3-hydroxybenzoate"),
    ("C=CC(=O)OC1CCC(=CC1)OC(=O)C=C", "4-[(prop-2-enoyl)oxy]cyclohex-1-en-1-yl prop-2-enoate"),
    ("CC(=O)OC/C=C/c1ccc(OC(C)=O)cc1", "4-[(1E)-3-(acetyloxy)prop-1-en-1-yl]phenyl acetate"),
]


# the one row of the class sample that is a PIN at the PIN tier: a single ester of a ring is named by its class there
# (``test_the_pin_tier_names_a_single_ester_of_a_ring_by_its_class``,, lane/l17 item 5)
_CLASS_PIN = "CC(=O)O[C@H]1C(=C)C=C(C=C1OC)OC"
# and the diester of a phenol and an allylic alcohol: the book's own pattern '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)'
# (``test_the_pin_tier_names_a_phenyl_acetate_with_an_acetyloxy_chain``)
_CLASS_PIN_DIESTER = "CC(=O)OC/C=C/c1ccc(OC(C)=O)cc1"
_CLASS_PINS = (_CLASS_PIN, _CLASS_PIN_DIESTER)


@pytest.mark.parametrize("smiles,name", _CLASS)
def test_an_ester_molecule_is_named_by_its_ester(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    if smiles not in _CLASS_PINS:
        assert row["tier"] != "pin_verified" and not row["is_pin"], row       # none of these is a PIN the PIN tier built
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [s for s, _n in _CLASS if s not in _CLASS_PINS])
def test_the_pin_tier_is_unchanged_for_the_class(smiles):
    """The class sample has no PIN at the PIN tier (it declined before the change and still does)."""
    row = tier_row(smiles, "pin")
    assert not row["is_pin"], row


def test_the_pin_tier_names_a_phenyl_acetate_with_an_acetyloxy_chain():
    """Re-asserted 2026-11-05 (lane/l17 final round, change-asserted-value): this row was in the class sample above as a
    name below the PIN. "Polyester names formed by using substitutive nomenclature, and/or multiplicative
    nomenclature, and functional class nomenclature" (the Blue Book): '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)
    (not 2-[2-(acetyloxy)phenyl]ethyl acetate; the ring is senior to the chain)': the ester of the ring is the anion and the
    ester of the chain a prefix of the organyl group. The name was below the PIN only because the probe of the decomposition
    engine at the PIN tier met the ring cascade's acyloxy name and set the one-way label of the call
    (``tier_a_ring._record_ester_cited_as_prefix``). Read back by OPSIN to the full InChIKey, the stereo layer included.
    Mutation: with the record of the cascade restored to ``probe_watched and (polycyclic or wider_tier_ctx.get)`` the row
    is ``pin_unverified`` at both tiers."""
    for tier in ("pin", "best-effort"):
        row = tier_row(_CLASS_PIN_DIESTER, tier)
        assert row["name"] == "4-[(1E)-3-(acetyloxy)prop-1-en-1-yl]phenyl acetate", row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
        assert_full_rt(row["name"], _CLASS_PIN_DIESTER)


def test_the_pin_tier_names_a_single_ester_of_a_ring_by_its_class():
    """Re-asserted 2026-11-05 (lane/l17 item 5, change-asserted-value): this row was in the class sample above and the
    PIN tier declined it (the ring cascade's acyloxy name was suppressed and nothing else named it). A single ester is
    named by functional class, "General methodology" (the Blue Book): "All preferred IUPAC names for
    esters are named by functional class nomenclature"; the organyl group is the substituent group of the ring,
    numbered from the free valence, the double bonds low ('cyclohexa-2,4-dien-1-yl', methoxy 2,4, methylidene 6: the
    locants {2,4} of the double bonds are lower than {3,5},. Read back by OPSIN to the full InChIKey, the
    stereo layer included."""
    row = tier_row(_CLASS_PIN, "pin")
    assert row["name"] == "(1S)-2,4-dimethoxy-6-methylidenecyclohexa-2,4-dien-1-yl acetate", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], _CLASS_PIN)


# the controls: the PIN of an ester of one polyacid with two alcohols the acid component names, unchanged
_CONTROLS = [
    ("CCOC(=O)CCC(=O)OC", "ethyl methyl butanedioate"),
    ("CC(C)OC(=O)CCC(=O)OCC1CCCCC1", "cyclohexylmethyl propan-2-yl butanedioate"),
    ("CCOC(=O)c1ccccc1C(=O)OC", "ethyl methyl benzene-1,2-dicarboxylate"),
]


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
@pytest.mark.parametrize("smiles,name", _CONTROLS)
def test_the_controls_keep_their_names(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


def test_the_probe_reads_a_name_of_another_molecule_as_no_name_for_an_ester():
    """``assembly.ester_prefix_probe.names_another_molecule`` (a wider tier only): True for an ester molecule and a name
    OPSIN reads as another constitution or does not read; False for a name that holds it, and for a molecule with no
    ester (an amide keeps its probe)."""
    from orthonym.assembly.ester_prefix_probe import names_another_molecule
    from orthonym.metrics.provenance import wider_tier_ctx
    diester = Chem.MolFromSmiles("CCOC(=O)CCC(=O)OC1CCCCC1")
    token = wider_tier_ctx.set(True)
    try:
        assert names_another_molecule("ethyl 4-(cyclohexyloxy)butanoate", diester)
        assert names_another_molecule("(acetyloxy) methyl butanoate", diester)            # OPSIN does not read it
        assert not names_another_molecule("cyclohexyl 4-ethoxy-4-oxobutanoate", diester)
        assert not names_another_molecule("N-ethylacetamide", Chem.MolFromSmiles("CC(=O)NC"))
    finally:
        wider_tier_ctx.reset(token)
    assert not names_another_molecule("ethyl 4-(cyclohexyloxy)butanoate", diester)       # the PIN tier: no probe rule


def test_the_verdict_of_the_constitution_oracle_has_three_answers(monkeypatch):
    """(I3) ``decomposition.engine.name_constitution_verdict``: True / False when OPSIN was asked, None when it was not
    (no name, the oracle off): the contract of ``_name_is_proven_complete`` ("only ever consulted to RESCUE, never to
    reject") holds for a caller that rejects only on False. With the oracle off the probe keeps even a name of another
    molecule, as before; and a correct ester name is never lost to it (``ethyl benzoate``)."""
    from orthonym.assembly.ester_prefix_probe import names_another_molecule
    from orthonym.decomposition import engine
    from orthonym.metrics.provenance import wider_tier_ctx
    mol = Chem.MolFromSmiles("CCOC(=O)c1ccccc1")
    assert engine.name_constitution_verdict("ethyl benzoate", mol) is True
    assert engine.name_constitution_verdict("methyl benzoate", mol) is False
    assert engine.name_constitution_verdict("", mol) is None
    monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "0")
    engine._PROVEN_COMPLETE_CACHE.clear()
    token = wider_tier_ctx.set(True)
    try:
        assert engine.name_constitution_verdict("ethyl benzoate", mol) is None
        assert engine.name_constitution_verdict("methyl benzoate", mol) is None
        assert engine._name_is_proven_complete("ethyl benzoate", mol) is False
        for name in ("ethyl benzoate", "methyl benzoate"):
            assert not names_another_molecule(name, mol)
    finally:
        wider_tier_ctx.reset(token)
        engine._PROVEN_COMPLETE_CACHE.clear()


def test_the_probe_runs_opsin_only_for_a_name_the_quality_gate_took(monkeypatch):
    """(M5) a name the gate rejects is not read by OPSIN: the engine asks ``names_another_molecule`` only after
    ``quality_ok`` (a name that passes is asked once)."""
    from orthonym.decomposition import engine
    from orthonym.metrics.provenance import wider_tier_ctx
    import orthonym.assembly.ester_prefix_probe as probe
    mol = Chem.MolFromSmiles("CCOC(=O)CCC(=O)OC1CCCCC1")
    real = probe.names_another_molecule

    def run(accepted):
        calls = []
        monkeypatch.setattr(probe, "names_another_molecule", lambda name, m: calls.append(name) or real(name, m))
        monkeypatch.setattr(engine, "_name_quality_is_acceptable", lambda name, m: accepted)
        token = wider_tier_ctx.set(True)
        try:
            engine.try_decompose(mol)
        finally:
            wider_tier_ctx.reset(token)
        return calls

    assert run(False) == []
    assert run(True) != []


# (I4) the glyceride assembler's mono-ester word of a diester of one polyacid is no PIN
_GLYCEROL_DIESTERS = [
    ("CCOC(=O)CCC(=O)OCC(O)CO", "2,3-dihydroxypropyl ethyl butanedioate"),
    ("COC(=O)CCC(=O)OCC(O)CO", "2,3-dihydroxypropyl methyl butanedioate"),
]


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
@pytest.mark.parametrize("smiles,name", _GLYCEROL_DIESTERS)
def test_the_glycerol_ester_of_a_polyacid_is_named_from_its_acid(smiles, name, tier):
    """ "Polyesters derived from a single acid component" (the Blue Book): "Fully esterified acids
    derived from a single acid are systematically named by placing the name(s) of the hydroxylic component [...] in front
    of the name of the acid component"; 'ethyl methyl butanedioate (PIN)'. The producer ``outer:lipid`` named it
    '2,3-dihydroxypropyl 4-ethoxy-4-oxobutanoate' at pin_verified."""
    row = tier_row(smiles, tier)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"], row
    assert row.get("producer") != "outer:lipid", row
    assert_full_rt(row["name"], smiles)


def test_the_polyester_builder_cites_different_organyl_groups_in_alphanumerical_order():
    """``name_polyacid_polyester`` with different alcohol components: "different organyl groups are cited
    in alphanumerical order (see "); an acid whose groups need locants of their own declines."""
    from orthonym.rules.esters import name_polyacid_polyester
    assert name_polyacid_polyester(Chem.MolFromSmiles("CCOC(=O)CCC(=O)OC1CCCCC1")) == "cyclohexyl ethyl butanedioate"
    assert name_polyacid_polyester(Chem.MolFromSmiles("COC(=O)CCC(=O)OCC(O)CO")) == "2,3-dihydroxypropyl methyl butanedioate"
    assert name_polyacid_polyester(Chem.MolFromSmiles("CCOC(=O)C(C)CC(=O)OC")) is None                 # locants needed
    assert name_polyacid_polyester(Chem.MolFromSmiles("CCOC(=O)C(CC(C)C(=O)OC)C(=O)OCC")) is None      # (M7: open)


# (M7) the triester of butane-1,1,3-tricarboxylic acid needs the locants of ('1,1-diethyl 3-methyl
# butane-1,1,3-tricarboxylate (PIN)',:31783), which the builder does not cite; the name stays an ester-prefixed one
def test_the_tricarboxylate_with_two_alcohols_keeps_its_valid_name():
    row = tier_row("CCOC(=O)C(CC(C)C(=O)OC)C(=O)OCC", "best-effort")
    assert not row["is_pin"] and row["tier"] != "abstain", row
    assert_full_rt(row["name"], "CCOC(=O)C(CC(C)C(=O)OC)C(=O)OCC")


# (I1) a name computed beside a cycle hit is no function of its SMILES: the fragment cache does not hold it under the bare key
_ACRYLATE_GLYCOSIDE = "O=C(/C=C/C1(O)C=CC(O)CC1=O)OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O"
_ACRYLIC_ACID = "O=C(O)/C=C/C1(O)C=CC(O)CC1=O"


def test_the_acid_fragment_is_named_without_a_hydroxy_it_does_not_have():
    """The acid fragment of the cinnamoyl-like glycoside ester was named '...-3-hydroxyprop-2-enoic acid' (another
    molecule) by ``polyfunctional_parts._group_on_chain_parent``: the tertiary alcohol of the ring is a group of the ring,
    the chain atom it hangs from is only a neighbour in its SMARTS match. (The standalone name was right either way; the
    wrong one came out in the nested naming, where the cache kept it: the two glycoside tests below fail without the fix.)"""
    from orthonym import name_compound
    assert name_compound(_ACRYLIC_ACID) == "(2E)-3-(1,4-dihydroxy-6-oxocyclohex-2-en-1-yl)prop-2-enoic acid"


@pytest.mark.parametrize("tier", _WIDER)
def test_the_glycoside_ester_keeps_its_ester_name_after_the_probe_rejected_a_name(tier):
    row = tier_row(_ACRYLATE_GLYCOSIDE, tier)
    assert row["name"] == ("[(2R,3S,4S,5R,6S)-3,4,5-trihydroxy-6-(4-hydroxyphenoxy)oxan-2-yl]methyl "
                           "(2E)-3-(1,4-dihydroxy-6-oxocyclohex-2-en-1-yl)prop-2-enoate"), row
    assert "(oxo)methyl" not in row["name"]
    assert_full_rt(row["name"], _ACRYLATE_GLYCOSIDE)


def test_the_fragment_cache_keeps_a_name_met_beside_a_cycle_under_its_context():
    """``fragment_naming``: a naming that met a SMILES of the visited set is cached with it, and served only where the same
    SMILES are on the set again; a naming that met none is cached as before."""
    from orthonym.assembly import fragment_naming as fn
    fn.start_naming_session()
    try:
        cache = fn._fragment_guard.cache
        fn._remember_fragment_name(cache, "A", "name-a", frozenset())
        fn._remember_fragment_name(cache, "B", "name-b", frozenset({"C"}))
        assert cache["A"] == "name-a" and "B" not in cache
        assert cache[(fn._CTX_KEY, "B")] == (frozenset({"C"}), "name-b")
        assert fn._cycle_hits_since(fn._cycle_hit_mark(), "X") == frozenset()
        mark = fn._cycle_hit_mark()
        fn._note_cycle_hit("C"); fn._note_cycle_hit("X")
        assert fn._cycle_hits_since(mark, "X") == frozenset({"C"})
    finally:
        fn.end_naming_session()


# (C1) an ester whose alcohol component carries another ester of a different anion is named in method (2) of
# "When anions are different, two methods are used" (the Blue Book: "Method (1) generates preferred
# IUPAC names but names formed by using method (2) are acceptable in general nomenclature"; '(1) 1,4-phenylene acetate
# dichloroacetate (PIN) (2) 4-(acetyloxy)phenyl dichloroacetate'): the decomposition of the ester bond names it so, below the PIN
_LIPID_A_TAIL = "CC(=O)N[C@H]1[C@H](C[C@@H]([C@H]([C@@H]1OC(=O)C[C@@H](CCCCCCCCCCC)O)O)CO)O"
_MIXED_ANION_ROWS = [
    # the phosphate, the 3-hydroxybutanamido / 3-hydroxyoctanamido analogues and the sulfate of a lipid A fragment
    "CCCCCCCCCCC[C@H](CC(=O)N[C@H]1[C@H](C[C@@H]([C@H]([C@@H]1OC(=O)C[C@@H](CCCCCCCCCCC)O)O)CO)OP(=O)(O)O)O",
    "CC[C@H](CC(=O)N[C@H]1[C@H](C[C@@H]([C@H]([C@@H]1OC(=O)C[C@@H](CCCCCCCCCCC)O)O)CO)OP(=O)(O)O)O",
    "CCCCC[C@H](CC(=O)N[C@H]1[C@H](C[C@@H]([C@H]([C@@H]1OC(=O)C[C@@H](CCCCCCCCCCC)O)O)CO)OP(=O)(O)O)O",
    "CCCCCCCCCCC[C@H](CC(=O)N[C@H]1[C@H](C[C@@H]([C@H]([C@@H]1OC(=O)C[C@@H](CCCCCCCCCCC)O)O)CO)OS(=O)(=O)O)O",
]


@pytest.mark.parametrize("smiles", _MIXED_ANION_ROWS)
def test_a_polyol_ester_with_two_different_anions_is_no_pin(smiles):
    """The default and PIN tiers do not ship the method (2) name as a PIN (the PIN tier gives no name, as before the
    item); the wider tier still names it, labelled below the PIN, and the name reads back."""
    row = tier_row(smiles, "pin")
    assert not row["is_pin"] and row["tier"] != "pin_verified", row
    row = tier_row(smiles, "best-effort")
    assert row["name"] and not row["is_pin"] and row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


def test_the_decomposition_labels_a_method_two_ester_name_below_the_pin():
    """``decomposition.engine._record_split_mixed_anion_non_pin`` (``rules.esters.record_split_polyol_mixed_anion_non_pin``):
    the label of the pipeline's ester route, asked of the alcohol fragment of the split. The ester oxygens of a split that
    names several ester bonds at once are named as ester groups ('ethane-1,2-diyl diacetate'); an ester far from the alcohol
    component (beyond an amide link, with its own alcohol) is no ester of it; identical anions keep their label (the polyol
    of, '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)', cannot be told from here)."""
    from orthonym.decomposition.engine import _record_split_mixed_anion_non_pin
    from orthonym.metrics import provenance as pv

    def labelled(smiles, name, n_bonds):
        mol = Chem.MolFromSmiles(smiles)
        matches = list(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")))[:n_bonds]
        bonds = [{"type": "ester", "match": m, "alkyl_atom": m[3]} for m in matches]
        pv.clear_provenance()
        _record_split_mixed_anion_non_pin(mol, bonds, name)
        return name in pv.get_provenance().get("non_pin_labels", ())

    assert labelled("CC(=O)OCCOP(=O)(O)O", "2-(phosphonooxy)ethyl acetate", 1)          # a phosphate cited as a prefix
    assert labelled("CC(=O)OCCOC(=O)CCl", "2-[(chloroacetyl)oxy]ethyl acetate", 1)      # another carboxylic anion
    assert not labelled("CC(=O)OCCOC(=O)CCl", "ethane-1,2-diyl acetate chloroacetate", 2)   # method (1): both named
    assert not labelled("CC(=O)OCCO", "2-hydroxyethyl acetate", 1)                      # no other ester
    assert not labelled("CC(=O)OCCOC(C)=O", "2-(acetyloxy)ethyl acetate", 1)            # identical anions
    assert not labelled("CC(=O)OC(C)CNC(=O)CC(=O)OC", "x acetate", 1)                   # the methyl ester is of another alcohol


# (C2) 'ethyl methyl (2R,3S)-2,3-dimethylbutanedioate' names the two enantiomers of the ethyl methyl ester of the meso acid
# alike: with a stereo layer the organyl groups need locants "When necessary, locants are cited at the front
# of the organyl groups", the Blue Book), which the builder does not cite
@pytest.mark.parametrize("smiles", [
    "CCOC(=O)[C@H](C)[C@H](C)C(=O)OC", "CCOC(=O)[C@@H](C)[C@@H](C)C(=O)OC",      # the meso acid, both enantiomers
    "CCOC(=O)C[C@H](C)CC(=O)OC", "CCOC(=O)C[C@@H](C)CC(=O)OC",                  # a centre that is none until the organyls differ
    "CCOC(=O)[C@H](O)[C@H](O)C(=O)OC",                                          # meso tartaric acid
])
def test_organyl_groups_that_need_locants_are_not_named_without_them(smiles):
    from orthonym.rules.esters import name_polyacid_polyester
    assert name_polyacid_polyester(Chem.MolFromSmiles(smiles)) is None
    for tier in ("pin", "valid", "best-effort"):
        row = tier_row(smiles, tier)
        assert not row["is_pin"] and row["tier"] != "pin_verified", row


@pytest.mark.parametrize("smiles,name", [
    ("CCOC(=O)[C@H](C)[C@@H](C)C(=O)OC", "ethyl methyl (2R,3R)-2,3-dimethylbutanedioate"),   # the groups swap by a rotation
    ("CCOC(=O)[C@@H](C)[C@H](C)C(=O)OC", "ethyl methyl (2S,3S)-2,3-dimethylbutanedioate"),
    ("CCOC(=O)C(C)C(C)C(=O)OC", "ethyl methyl 2,3-dimethylbutanedioate"),                   # no stereo layer: constitution
    ("CCOC(=O)CC(C)CC(=O)OC", "ethyl methyl 3-methylpentanedioate"),
    ("CCOC(=O)/C=C/C(=O)OC", "ethyl methyl (2E)-but-2-enedioate"),
])
def test_organyl_groups_that_swap_by_symmetry_need_no_locants(smiles, name):
    from orthonym.rules.esters import name_polyacid_polyester
    assert name_polyacid_polyester(Chem.MolFromSmiles(smiles)) == name
    assert_full_rt(name, smiles)


# (I1) the fragment cache, one level up: a name composed around a name served from a context entry is as context-bound
_CACHE_P, _CACHE_Q, _CACHE_S = "CC(C)(C)CCCCCCC", "CC(C)(C)CCCCCCCC", "CC(C)(C)CCCCCCCCC"


def _fake_names(monkeypatch, fn, calls_of, letter):
    """``name_compound`` stand-in: the name of each fragment cites the names of the fragments it asks for."""
    import orthonym.namer as namer

    def name_compound(smiles, style="pin"):
        parts = []
        for other in calls_of[smiles]:
            got = fn.name_fragment_recursively(other)
            parts.append("NONE" if got is None else got)
        return f"{letter[smiles]}[{','.join(parts)}]"

    monkeypatch.setattr(namer, "name_compound", name_compound)


def test_a_name_composed_around_a_context_bound_name_is_context_bound(monkeypatch):
    """P names Q then S; Q names P (a cycle: P is on the visited set, the answer is no name) and S names Q (Q is served
    from its context entry, the same SMILES being on the set). Both Q and S are degraded names of the set they were named on;
    S named alone, later, must not be served the degraded name 'S[Q[NONE]]' from the cache."""
    from orthonym.assembly import fragment_naming as fn
    from orthonym.perception.molcache import canon_smiles
    p, q, s = (canon_smiles(x) for x in (_CACHE_P, _CACHE_Q, _CACHE_S))
    assert not any(fn.FRAGMENT_NAME_CACHE.get(x) for x in (p, q, s))
    calls_of = {p: [q, s], q: [p], s: [q]}
    _fake_names(monkeypatch, fn, calls_of, {p: "P", q: "Q", s: "S"})

    fn.start_naming_session()
    try:
        clean = fn.name_fragment_recursively(s)
    finally:
        fn.end_naming_session()
    assert clean == "S[Q[P[NONE,NONE]]]", clean

    fn.start_naming_session()
    try:
        assert fn.name_fragment_recursively(p) == "P[Q[NONE],S[Q[NONE]]]"
        assert (fn._CTX_KEY, s) in fn._fragment_guard.cache and s not in fn._fragment_guard.cache
        # P is named, context-free, by now; S named alone is not served the name it had beside P's cycle
        assert fn.name_fragment_recursively(s) == "S[Q[P[Q[NONE],S[Q[NONE]]]]]"
    finally:
        fn.end_naming_session()


def test_the_depth_net_answer_is_cached_with_the_visited_set(monkeypatch):
    """The name of the depth net (``name_pipeline_only``) is a function of the visited set: it is not held under the bare
    SMILES, and the fragment named around it is logged as context-bound."""
    import orthonym.namer as namer
    from orthonym.assembly import fragment_naming as fn
    from orthonym.perception.molcache import canon_smiles
    p, q = canon_smiles(_CACHE_P), canon_smiles(_CACHE_Q)
    monkeypatch.setattr(fn, "_MAX_VISITED_SIZE", 1)
    monkeypatch.setattr(namer, "name_pipeline_only", lambda smiles, **kw: "net:" + smiles)
    monkeypatch.setattr(namer, "name_compound", lambda smiles, style="pin": (
        f"P[{fn.name_fragment_recursively(q)}]" if smiles == p else f"alone:{smiles}"))
    fn.start_naming_session()
    try:
        assert fn.name_fragment_recursively(p) == f"P[net:{q}]"
        cache = fn._fragment_guard.cache
        assert q not in cache and cache[p] == f"P[net:{q}]"
        assert cache[(fn._CTX_KEY, q)][0] == frozenset({p})
        assert fn.name_fragment_recursively(q) == f"alone:{q}"
    finally:
        fn.end_naming_session()


def test_an_unavailable_verdict_is_not_cached(monkeypatch):
    """``name_constitution_verdict``: a run that returned no structure (a JVM error, a timeout, a name OPSIN does not read)
    is no property of the name and gives no verdict (``None``: the caller that rejects on ``False`` keeps the name), so a
    host under load names a molecule as an idle one does (final round F1)."""
    from orthonym.decomposition import engine
    from orthonym.validation import atom_coverage
    mol = Chem.MolFromSmiles("CCOC(=O)c1ccccc1")
    engine._PROVEN_COMPLETE_CACHE.clear()
    real = atom_coverage.validate_atom_coverage
    state = {"down": True}

    def flaky(m, name, *args, **kwargs):
        if state["down"]:
            return atom_coverage.CoverageResult(
                total_heavy_atoms=m.GetNumHeavyAtoms(), claimed_atoms=0, unclaimed_atoms=m.GetNumHeavyAtoms(),
                coverage_ratio=0.0, is_complete=False, method="unavailable")
        return real(m, name, *args, **kwargs)

    monkeypatch.setattr(atom_coverage, "validate_atom_coverage", flaky)
    try:
        assert engine.name_constitution_verdict("ethyl benzoate", mol) is None
        assert not engine._PROVEN_COMPLETE_CACHE
        state["down"] = False
        assert engine.name_constitution_verdict("ethyl benzoate", mol) is True
    finally:
        engine._PROVEN_COMPLETE_CACHE.clear()


# (carbamate) a carbamate is an ester of carbamic acid, named by its own route: the rule of this item does not count it
_CARBAMATE = "O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(=O)OC)c2ccccc12"


def test_a_carbamate_is_not_counted_among_the_esters_of_the_probe():
    """ (the Blue Book) names the carbamate by its own route SENIORITY ORDER FOR CLASSES, Table 4.1):
    ``non_carbamate_ester_carbonyl_atoms`` leaves its carbonyl carbon out; a carbonate and a haloformate stay."""
    from orthonym.rules.esters import non_carbamate_ester_carbonyl_atoms
    for smiles, n in [("CCOC(=O)c1ccccc1", 1), ("CC(=O)OC", 1), ("O=COC", 1), ("COC(=O)NC", 0), (_CARBAMATE, 0),
                      ("COC(=O)OC", 1), ("COC(=O)Cl", 1), ("CCOC(=O)CCC(=O)OC", 2), ("COC(=O)NCC(=O)OC", 1)]:
        assert len(non_carbamate_ester_carbonyl_atoms(Chem.MolFromSmiles(smiles))) == n, smiles


@pytest.mark.parametrize("tier", ["valid", "complete"])
def test_the_carbamate_row_keeps_the_name_of_the_general_engine(tier):
    """The valid and complete tiers keep the general engine's name, the carbamate cited as a prefix. The best-effort tier
    names the carbamate ester as the parent (lane/l17 item 3, "9 Esters", the Blue Book:
    ``tests/unit/rules/test_leads_l5_item7.py::test_a_carbamate_ester_is_the_parent_and_the_ring_one_is_its_prefix_at_best_effort``),
    and the probe does not reject it (``non_carbamate_ester_carbonyl_atoms``)."""
    row = tier_row(_CARBAMATE, tier)
    assert row["tier"] == "systematic_verified", row
    assert "1-oxo-1,2-dihydrophthalazine" in row["name"] and "N-{" not in row["name"], row
    assert_full_rt(row["name"], _CARBAMATE)
