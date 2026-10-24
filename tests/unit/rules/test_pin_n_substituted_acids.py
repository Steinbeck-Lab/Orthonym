"""PIN class program, topic HXL: the substituent on the nitrogen of a hydroxamic, sulfonimidic or imidic acid.

 "Substitutive nomenclature, suffix mode" (the Blue Book,:30096): "Acids in which the
carbonyl oxygen atom of a carboxylic acid group has been replaced by =N-OH are named as *N*-hydroxy
derivatives of imidic acids". The N-hydroxy is one substituent on the N of the group, and the book prints
the others: 'N-sulfanylcyclopentanecarboximidic acid (PIN)' "Functional replacement in systematic
names of carboxylic acids",:30271), 'N-ethyl-N-methylcarbamimidic acid (PIN)',:30764),
'N-hydroxymethanesulfonimidic acid (PIN)' "Hydroximic and hydroxamic acids derived from sulfonic,
sulfinic, etc. acids",:31230,:31234); a hydroxamic acid is the N-hydroxy amide,:30144;
"Hydroxamic acids",:32821: 'N-hydroxypropanamide (PIN)'). The substituent is a prefix like the others, cited
with them in one alphanumerical series,:3436; 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)',:32879),
under an italic locant that is essential,:2869), and one that the parent also carries is one multiplied
prefix (b),:7067).

The groups were named by handlers that put 'N-hydroxy' in front of the finished name of the parent acid, which
is right only for a parent with no prefix of its own ('N-hydroxy1-chloromethanesulfonimidic acid', nothing for
'N-hydroxy-2-methylpropanamide'); an imidic acid with a carbon group on its N was named as an imine and a
thiol ('1-(methylimino)ethane-1-thiol'), which the acid outranks class 7a,:18172).

A row is the Blue Book's own where the comment quotes its line; the others are what the rules above give for
a compound the book does not print, and each is read back by OPSIN (``assert_pin_at_both_tiers``) to the input's
full InChIKey and, for the groups whose tautomer or C=N configuration that key does not see, to its isomeric
SMILES (``_isomeric_read_back``).
"""
import pytest

from orthonym import Orthonym
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)

pytestmark = pytest.mark.opsin_gate

# the Blue Book's own (PIN) rows
BB_ROWS = [
    ("OC(=NS)C1CCCC1", "N-sulfanylcyclopentanecarboximidic acid"),             #:30271
    ("CS(=O)(=NO)O", "N-hydroxymethanesulfonimidic acid"),                      #:31234
    ("ONC(=O)CC", "N-hydroxypropanamide"),                                     #:32825
    ("ONC(=O)C1CCCCC1", "N-hydroxycyclohexanecarboxamide"),                    #:32827
]

HYDROXAMIC_ROWS = [
    ("CC(C)C(=O)NO", "N-hydroxy-2-methylpropanamide"),
    ("ClCC(=O)NO", "2-chloro-N-hydroxyacetamide"),
    ("ONC=O", "N-hydroxyformamide"),                                           #,:32699
    ("ONC(=O)C(C)(C)C", "N-hydroxy-2,2-dimethylpropanamide"),
    ("ONC(=O)CCCl", "3-chloro-N-hydroxypropanamide"),
    ("ONC(=O)Cc1ccccc1", "N-hydroxy-2-phenylacetamide"),
    ("ONC(=O)CC(C)O", "N,3-dihydroxybutanamide"),
    ("ONC(=O)CCN", "3-amino-N-hydroxypropanamide"),
    ("ONC(=O)C1CCCCC1C", "N-hydroxy-2-methylcyclohexane-1-carboxamide"),
    ("CC(C)(CCS(C)(=O)=O)C(=O)NO", "N-hydroxy-4-(methanesulfonyl)-2,2-dimethylbutanamide"),
    ("CC(CCC=C)CC(=O)NO", "N-hydroxy-3-methylhept-6-enamide"),
    ("CC(C)(C(=O)NO)ON", "2-(aminooxy)-N-hydroxy-2-methylpropanamide"),
    ("ONC(=O)C=CC", "N-hydroxybut-2-enamide"),
    ("ONC(=O)/C=C/C", "(2E)-N-hydroxybut-2-enamide"),
    ("CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)NO", "(9Z,12Z)-N-hydroxyoctadeca-9,12-dienamide"),
]

SULFONIMIDIC_ROWS = [
    # the N locant is cited, so the carbon's '1' is too: an N-prefix that is not a carbon group could move to
    # it (``locant_omission.mononuclear_parent_locant_required``)
    ("ClCS(=O)(=NO)O", "1-chloro-N-hydroxymethanesulfonimidic acid"),
    ("CCS(=O)(=NO)O", "N-hydroxyethanesulfonimidic acid"),
    ("ClCCS(=O)(=NO)O", "2-chloro-N-hydroxyethane-1-sulfonimidic acid"),
    ("OCCS(=O)(=NO)O", "N,2-dihydroxyethane-1-sulfonimidic acid"),
    ("NCCS(=O)(=NO)O", "2-amino-N-hydroxyethane-1-sulfonimidic acid"),
    ("CS(=O)(=NC)O", "N-methylmethanesulfonimidic acid"),
    # a carbon group on the N would change the parent, so the '1' is omitted
    ("ClCS(=O)(=NC)O", "chloro-N-methylmethanesulfonimidic acid"),
    ("CCCS(=O)(=NCC)O", "N-ethylpropane-1-sulfonimidic acid"),
    ("CC(O)CS(=O)(=NC)O", "2-hydroxy-N-methylpropane-1-sulfonimidic acid"),
    # "The locant '1' is omitted:... (c) in monosubstituted homogeneous monocyclic rings" (:2891,:2913,
    # 'cyclohexanethiol (PIN)',:2917): the substituent on the N is cited with the italic N, outside the ring, as in
    # 'N-hydroxycyclohexanecarboxamide (PIN)' (:32827) and 'N-sulfanylcyclopentanecarboximidic acid (PIN)' (:30271)
    ("O=S(=NC)(O)C1CCCCC1", "N-methylcyclohexanesulfonimidic acid"),
    ("O=S(=NO)(O)C1CCCCC1", "N-hydroxycyclohexanesulfonimidic acid"),
    ("O=S(=NCC)(O)C1CCCCC1", "N-ethylcyclohexanesulfonimidic acid"),
    ("S(=NC)(O)C1CCCC1", "N-methylcyclopentanesulfinimidic acid"),
    ("S(=NO)(O)C1CCCCC1", "N-hydroxycyclohexanesulfinimidic acid"),
    # the suffix takes the lowest locant before the prefixes and the double bond (c),:3256)
    ("O=S(=NC)(O)C1CCC(Cl)CC1", "4-chloro-N-methylcyclohexane-1-sulfonimidic acid"),
    ("O=S(=NO)(O)C1CCC(Cl)CC1", "4-chloro-N-hydroxycyclohexane-1-sulfonimidic acid"),
    ("S(=NC)(O)C1CCC(Cl)CC1", "4-chloro-N-methylcyclohexane-1-sulfinimidic acid"),
    ("O=S(=NC)(O)C1CCC=CC1", "N-methylcyclohex-3-ene-1-sulfonimidic acid"),
    ("O=S(=N)(O)C1CCC(Cl)CC1", "4-chlorocyclohexane-1-sulfonimidic acid"),
    # a prefix of the ring that is also the N's is one multiplied prefix ('N,4-dimethyl-N-(3-methylphenyl)benzamide
    # (PIN)',:32879; 'N,N,2-trimethyl...propanamide (PIN)',:21624)
    ("O=S(=NO)(O)C1CCC(O)C1", "N,3-dihydroxycyclopentane-1-sulfonimidic acid"),
    ("O=S(=NO)(O)C1CCC(O)CC1", "N,4-dihydroxycyclohexane-1-sulfonimidic acid"),
    ("S(=NO)(O)C1CCC(O)C1", "N,3-dihydroxycyclopentane-1-sulfinimidic acid"),
    ("O=S(=NC)(O)C1CCC(C)C1", "N,3-dimethylcyclopentane-1-sulfonimidic acid"),
    ("CS(=O)(=NS)O", "N-sulfanylmethanesulfonimidic acid"),
    # the sulfinic acids: "sulfonic, sulfinic, etc."
    ("CS(=NO)O", "N-hydroxymethanesulfinimidic acid"),
    ("CS(=NC)O", "N-methylmethanesulfinimidic acid"),
    ("ClCCS(=NO)O", "2-chloro-N-hydroxyethane-1-sulfinimidic acid"),
]

N_IMIDIC_ROWS = [
    # the false PIN: the acid outranks the thiol class 7a,:18172,:18190)
    ("CC(=NC)S", "N-methylethanimidothioic acid"),
    ("CC(=NC)O", "N-methylethanimidic acid"),
    ("CC(=NCC)O", "N-ethylethanimidic acid"),
    ("CC(C)C(=NC)O", "N,2-dimethylpropanimidic acid"),
    ("CC(C)C(=NCC)O", "N-ethyl-2-methylpropanimidic acid"),
    ("SC(C(C)(C)C)=NC", "N,2,2-trimethylpropanimidothioic acid"),
    ("CC(=Nc1ccccc1)O", "N-phenylethanimidic acid"),
    ("CC(=Nc1ccccc1)S", "N-phenylethanimidothioic acid"),
    ("CC(=NC(C)C)O", "N-(propan-2-yl)ethanimidic acid"),
    ("CC(=NC1CCCCC1)O", "N-cyclohexylethanimidic acid"),
    ("CC(=NCc1ccccc1)O", "N-benzylethanimidic acid"),
    ("CC(O)=NC(C)(C)C", "N-tert-butylethanimidic acid"),
    ("OC=NC", "N-methylmethanimidic acid"),
    ("SC=NC", "N-methylmethanimidothioic acid"),
    ("[SeH]C(C)=NC", "N-methylethanimidoselenoic acid"),
    ("[TeH]C(C)=NC", "N-methylethanimidotelluroic acid"),
    ("ClCCC(=NC)O", "3-chloro-N-methylpropanimidic acid"),
    ("OCCC(=NC)O", "3-hydroxy-N-methylpropanimidic acid"),
    ("NCCC(=NC)O", "3-amino-N-methylpropanimidic acid"),
    ("OC(=NC)CC(=O)C", "N-methyl-3-oxobutanimidic acid"),
    ("NC(=O)CCC(=NC)O", "4-amino-N-methyl-4-oxobutanimidic acid"),
    ("OC(=NC)C=C", "N-methylprop-2-enimidic acid"),
    ("OC(=NC)C1CC1", "N-methylcyclopropanecarboximidic acid"),
    ("OC(=NC)C1CCCCC1", "N-methylcyclohexanecarboximidic acid"),
    ("SC(=NC)C1CCCCC1", "N-methylcyclohexanecarboximidothioic acid"),
    # benzene: the suffix keeps its locant beside a ring prefix, as it does for '4-methylbenzene-1-carboximidic acid'
    ("c1ccccc1C(=NC)O", "N-methylbenzenecarboximidic acid"),
    ("c1ccccc1C(=NC)S", "N-methylbenzenecarboximidothioic acid"),
    ("Cc1ccc(cc1)C(=NC)O", "N,4-dimethylbenzene-1-carboximidic acid"),
    ("Clc1ccccc1C(=NCC)O", "2-chloro-N-ethylbenzene-1-carboximidic acid"),
    ("OC(=Nc1ccccc1)c1ccccc1", "N-phenylbenzenecarboximidic acid"),
    ("Cc1ccccc1C(=Nc1ccccc1C)O", "2-methyl-N-(2-methylphenyl)benzene-1-carboximidic acid"),
    ("Oc1ccccc1C(=NC)O", "2-hydroxy-N-methylbenzene-1-carboximidic acid"),
    ("OC(=NS)c1ccccc1", "N-sulfanylbenzenecarboximidic acid"),
    ("OC(=N[SeH])C", "N-selanylethanimidic acid"),
    # named alike at both tiers: the reading of the group is a function of the structure (see
    # test_perception_reads_the_structure_only)
    ("CC(O)=NCC1(c2ccccc2)CCCC1", "N-[(1-phenylcyclopentyl)methyl]ethanimidic acid"),
    ("CC1CCCCC1C(=NC)O", "N,2-dimethylcyclohexane-1-carboximidic acid"),
    ("OC(=NC)C1CCC(C)CC1", "N,4-dimethylcyclohexane-1-carboximidic acid"),
    ("OC(=NCC1(c2ccccc2)CCCC1)c1ccccc1", "N-[(1-phenylcyclopentyl)methyl]benzenecarboximidic acid"),
]

STEREO_ROWS = [
    ("C/C(O)=N/C", "(1Z)-N-methylethanimidic acid"),
    ("C/C(O)=N\\C", "(1E)-N-methylethanimidic acid"),
    ("C/C(S)=N/C", "(1Z)-N-methylethanimidothioic acid"),
    ("C/C(S)=N\\C", "(1E)-N-methylethanimidothioic acid"),
    ("CC/C(S)=N\\C", "(1E)-N-methylpropanimidothioic acid"),
    ("C/C([SeH])=N/C", "(1Z)-N-methylethanimidoselenoic acid"),
    # a ring or benzene parent has no locant for the carbon of the group: the descriptor is bare
    ("c1ccccc1/C(O)=N/C", "(Z)-N-methylbenzenecarboximidic acid"),
    ("c1ccccc1/C(O)=N\\C", "(E)-N-methylbenzenecarboximidic acid"),
    ("COc1ccccc1/C(O)=N/CC1(c2ccccc2)CCCC1", "(Z)-2-methoxy-N-[(1-phenylcyclopentyl)methyl]benzene-1-carboximidic acid"),
]


@pytest.mark.parametrize("smiles,pin", BB_ROWS)
def test_blue_book_n_substituted_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", HYDROXAMIC_ROWS)
def test_hydroxamic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", SULFONIMIDIC_ROWS)
def test_sulfonimidic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _isomeric_read_back(name, smiles):
    """True iff OPSIN reads ``name`` to the input's ISOMERIC canonical SMILES.

    The full InChIKey the other read-backs compare sees neither the configuration of the C=N of -C(=N-R)-XH nor
    which tautomer is drawn: its mobile hydrogen puts the group in one layer, so 'C/C(O)=N/C', 'C/C(O)=N\\C' and
    the amide 'CC(=O)NC' share the key an InChIKey. That the name is the imidic acid's, with the
    configuration the input has, is proved here against OPSIN's own reading of the name."""
    from tests.support.rt_assert import _canon, _independent_parse
    read = _independent_parse(name, smiles)
    return bool(read) and _canon(read) == _canon(smiles)


@pytest.mark.parametrize("smiles,pin", N_IMIDIC_ROWS)
def test_n_substituted_imidic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert _isomeric_read_back(pin, smiles), (smiles, pin)


@pytest.mark.parametrize("smiles,pin", STEREO_ROWS)
def test_n_substituted_imidic_acid_states_the_cn_configuration(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert _isomeric_read_back(pin, smiles), (smiles, pin)
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res["stereo_unexpressed"] is False, res


@pytest.mark.parametrize("smiles,thiol", [
    ("CC(=NC)S", "1-(methylimino)ethane-1-thiol"),
    ("SC(=NC)c1ccccc1", "(methylimino)(phenyl)methanethiol"),
    ("SC(CC(O)C)=NC", "4-(methylimino)-4-sulfanylbutan-2-ol"),
    ("C/C(S)=N/C", "(1Z)-1-(methylimino)ethane-1-thiol"),
])
def test_the_thiol_name_of_an_imidothioic_acid_is_not_a_pin(smiles, thiol):
    assert_not_pin_labelled(smiles, thiol)


def test_the_acid_group_is_perceived_only_where_it_is_the_principal_group():
    """The book prints no prefix for a carbon group on the N of an imidic acid demoted by a senior group, so there
    the imine and the thiol keep their own prefixes, as before."""
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group

    def principal(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return get_principal_group(mol, detect_functional_groups(mol))[0]

    assert principal("CC(=NC)O") == "n_substituted_imidic_acid"
    assert principal("CC(=NC)S") == "n_substituted_imidic_acid"
    for demoted in ("OC(=O)CCC(=NC)S", "OC(=O)CCC(=NC)O",       # a senior acid
                    "OC(=NC)CCC(=N)O",                           # an imidic acid of its own rank
                    "OC(=NC)CCC(=NO)O",                          # a hydroximic acid of its rank
                    "OC(=NC)CCC(=NC)O"):                         # two of them: not one group
        assert principal(demoted) != "n_substituted_imidic_acid", demoted
        assert "n_substituted_imidic_acid" not in detect_functional_groups(Chem.MolFromSmiles(demoted)), demoted
    # a group on the N the producers do not build is not read as the acid either
    for unbuilt in ("CC(=NCCO)O", "CC(=NCC=C)O", "CC(=N[C@H](C)CC)O"):
        assert "n_substituted_imidic_acid" not in detect_functional_groups(Chem.MolFromSmiles(unbuilt)), unbuilt


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)CCC(=NC)S", "4-(methylimino)-4-sulfanylbutanoic acid"),
    ("OC(=O)CCC(=NC)O", "4-hydroxy-4-(methylimino)butanoic acid"),
    ("OC(=O)c1ccc(C(=NC)O)cc1", "1-carboxy-4-[hydroxy(methylimino)methyl]benzene"),
])
def test_a_demoted_group_keeps_the_name_it_had(smiles, name):
    """Demote, never delete: the senior acid's name is the one before this change, at the tier it was."""
    assert name_breadth(smiles).get("name") == name


@pytest.mark.parametrize("smiles", [
    "NC(=O)CC(=O)NO",                 # a hydroxamic acid beside an amide is a diamide,:18875)
    "ONC(=O)CCCCCCC(=O)NO",           # two hydroxamic acids: N1,N8 locants, not built
    "ONC(=O)CCC(=O)O",                # under a senior acid
    "ONC(=O)N",                       # a carbonic acid derivative
    "CC(=NCCO)O",                     # a group on the N that has a group of its own
    "CC(=NOC)O",                      # an alkoxy on the N
    "CS(=O)(=NOC)O",
    "OC(=NC)CCC(=NC)O",               # two N-substituted imidic acids
    "O=S(=NO)(O)CCS(=O)(=NO)O",       # two sulfonimidic acids with an N-substituent
])
def test_the_group_that_is_not_built_is_not_labelled_a_pin(smiles):
    assert_declined_at_default(smiles)


def test_the_two_dedicated_handlers_are_gone():
    """One home for the rule: the fragment assembly names the group, and the substituent on its N is a prefix."""
    from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
    assert "hydroxamic_acid" not in INNER_DISPATCH_TABLE
    assert "sulfonimidic_n_hydroxy" not in INNER_DISPATCH_TABLE
    assert Orthonym().name_tiered("ONC(=O)C")["name"] == "N-hydroxyacetamide"


def test_the_group_is_written_with_the_suffix_of_its_parent_acid():
    from orthonym.rules.n_substituted_acids import n_substituents, suffix_group

    def group(smiles, key):
        from orthonym.perception.functional_groups import detect_functional_groups
        mol = Chem.MolFromSmiles(smiles)
        return mol, detect_functional_groups(mol)[key]

    mol, matches = group("CC(=NC)S", "n_substituted_imidic_acid")
    assert suffix_group(mol, "n_substituted_imidic_acid", matches) == "imidothioic_acid"
    mol, matches = group("[SeH]C(C)=NC", "n_substituted_imidic_acid")
    assert suffix_group(mol, "n_substituted_imidic_acid", matches) == "imidoselenoic_acid"
    mol, matches = group("CC(C)C(=O)NO", "hydroxamic_acid")
    assert suffix_group(mol, "hydroxamic_acid", matches) == "primary_amide"
    assert [n for n, _ in n_substituents(mol, "hydroxamic_acid", matches[0])] == ["hydroxy"]
    mol, matches = group("ClCS(=O)(=NC)O", "sulfonimidic_acid")
    assert suffix_group(mol, "sulfonimidic_acid", matches) == "sulfonimidic_acid"
    assert [n for n, _ in n_substituents(mol, "sulfonimidic_acid", matches[0])] == ["methyl"]
    mol, matches = group("CS(=O)(=N)O", "sulfonimidic_acid")
    assert n_substituents(mol, "sulfonimidic_acid", matches[0]) == []        # the plain acid has none to cite


def test_a_ring_suffix_beside_the_group_on_the_n_cites_no_locant_of_its_own():
    """The group on the N is the suffix's, not a substituent of the ring: 'N-sulfanylcyclopentanecarboximidic acid
    (PIN)' (:30271) has no ring locant, and 'N-methylcyclohexanecarboximidic acid' does not either."""
    assert name_default("OC(=NC)C1CCCCC1")["name"] == "N-methylcyclohexanecarboximidic acid"
    assert name_default("CC1CCCCC1C(=NO)O")["name"] == "N-hydroxy-2-methylcyclohexane-1-carboximidic acid"


@pytest.mark.parametrize("smiles,reading,verdict", [
    ("C/C(O)=N/C", "C/C(O)=N/C", "ok"),
    ("C/C(O)=N/C", "C/C(O)=N\\C", "mismatch"),          # the flipped descriptor shares the key with the input
    ("C/C(O)=N/C", "CC(O)=NC", "mismatch"),             # no configuration where the input has one
    ("CC(O)=NC", "C/C(O)=N/C", "mismatch"),             # a configuration the input does not state
    ("C/C(S)=N\\C", "C/C(S)=N\\C", "ok"),
    ("C/C(O)=N/O", "C/C(O)=N\\O", "mismatch"),          # the N-hydroxy one, as before
    ("C/C(O)=N/S", "C/C(O)=N\\S", "mismatch"),
    ("CC(O)=NC", "CC(O)=NC", "ok"),
])
def test_the_cn_configuration_verdict_reads_the_n_substituted_group(smiles, reading, verdict):
    from orthonym.rules.hydroximic_acids import cn_configuration_verdict
    assert cn_configuration_verdict(smiles, reading) == verdict


def test_a_flipped_cn_descriptor_does_not_ship(monkeypatch):
    """Without the verdict the flipped name ships (the key sees nothing), with it the name does not: the test can fail."""
    import orthonym.assembly.handlers._handler_shared as hs
    smiles = "C/C(S)=N/C"
    real = hs._generate_stereodescriptors

    def flipped(features, *args, **kwargs):
        frag = real(features, *args, **kwargs)
        if frag is not None and "(1Z)" in frag.text:
            frag.text = frag.text.replace("(1Z)", "(1E)")
        return frag

    monkeypatch.setattr(hs, "_generate_stereodescriptors", flipped)
    res = Orthonym().name_tiered(smiles)
    assert res.get("name") != "(1E)-N-methylethanimidothioic acid", res
    import orthonym.rules.hydroximic_acids as hx
    monkeypatch.setattr(hx, "cn_configuration_verdict", lambda *args, **kwargs: "ok")
    assert Orthonym().name_tiered(smiles).get("name") == "(1E)-N-methylethanimidothioic acid"


def test_the_two_names_of_a_spelling_are_one_name():
    """Determinism: the same molecule written two ways is named alike."""
    pairs = [("CC(=NC)S", "SC(C)=NC"), ("ClCC(=O)NO", "ONC(=O)CCl"), ("ClCS(=O)(=NO)O", "ON=S(=O)(O)CCl"),
             ("CC(C)C(=NC)O", "OC(C(C)C)=NC"), ("c1ccccc1C(=NC)O", "CN=C(O)c1ccccc1")]
    for a, b in pairs:
        assert name_default(a)["name"] == name_default(b)["name"], (a, b)


def test_perception_reads_the_structure_only(monkeypatch):
    """Perception is cached as a pure function of the molecule (``assembly/memo.py``, ``pure_cache_or_compute``), so
    whether it reads the group as the acid may not ask the naming pipeline, whose answer depends on the naming
    context: the substituent on the N is tested by its structure, and the producer names it (or declines)."""
    from orthonym.perception.functional_groups import _detect_functional_groups_impl
    import orthonym.rules.amides as amides
    calls = []
    monkeypatch.setattr(amides, "_name_n_substituent", lambda *args, **kwargs: calls.append(args) or None)
    for smiles in ("CC(=NC)O", "CC(O)=NCC1(c2ccccc2)CCCC1", "OC(=NCC1(c2ccccc2)CCCC1)c1ccccc1"):
        found = _detect_functional_groups_impl(Chem.MolFromSmiles(smiles))
        assert "n_substituted_imidic_acid" in found, smiles
    assert calls == []


def test_the_name_does_not_depend_on_the_tier_named_before():
    """One process, both orders of the tiers: one name for each tier (the reviewer's row named the group as an
    imine after a default-tier call had cached its perception without it)."""
    smiles = "COc1ccccc1/C(O)=N/CC1(c2ccccc2)CCCC1"
    pin = "(Z)-2-methoxy-N-[(1-phenylcyclopentyl)methyl]benzene-1-carboximidic acid"
    for first, second in ((name_default, name_breadth), (name_breadth, name_default)):
        assert (first(smiles)["name"], second(smiles)["name"]) == (pin, pin)


#: one row per group and parent kind; each is named in many atom orders
_RESPELL_ROWS = [
    ("COc1ccccc1C(O)=NCC1(c2ccccc2)CCCC1", "2-methoxy-N-[(1-phenylcyclopentyl)methyl]benzene-1-carboximidic acid"),
    ("COc1ccccc1/C(O)=N/CC1(c2ccccc2)CCCC1",
     "(Z)-2-methoxy-N-[(1-phenylcyclopentyl)methyl]benzene-1-carboximidic acid"),
    ("CC(O)=NCC1(c2ccccc2)CCCC1", "N-[(1-phenylcyclopentyl)methyl]ethanimidic acid"),
    ("CC(C)C(=NC)O", "N,2-dimethylpropanimidic acid"),
    ("O=S(=NO)(O)C1CCC(O)C1", "N,3-dihydroxycyclopentane-1-sulfonimidic acid"),
    ("O=S(=NC)(O)C1CCC(Cl)CC1", "4-chloro-N-methylcyclohexane-1-sulfonimidic acid"),
    ("ClCS(=O)(=NO)O", "1-chloro-N-hydroxymethanesulfonimidic acid"),
    ("CC(C)(C(=O)NO)ON", "2-(aminooxy)-N-hydroxy-2-methylpropanamide"),
]


@pytest.mark.parametrize("smiles,pin", _RESPELL_ROWS)
def test_the_name_does_not_depend_on_the_atom_order(smiles, pin):
    """Determinism over the spellings of one molecule (random atom orders, both tiers): the reviewer's row declined at
    the default tier whenever the cyclopentane was written before the phenyl, because the N-substituent's name lost
    its CH2 (``amides._enrich_ring_n_substituent`` read the first ring of RDKit's ring list as the stem)."""
    mol = Chem.MolFromSmiles(smiles)
    spellings = sorted({smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(30)})[:8]
    for spelling in spellings:
        for res in (name_default(spelling), name_breadth(spelling)):
            assert (res["name"], res["tier"]) == (pin, "pin_verified"), (spelling, res)


@pytest.mark.parametrize("smiles,pin", [
    ("COc1ccccc1C(=O)NCC1(c2ccccc2)CCCC1", "2-methoxy-N-[(1-phenylcyclopentyl)methyl]benzamide"),
    ("CC(=O)NCC1CCC(C)CC1", "N-[(4-methylcyclohexyl)methyl]acetamide"),
    ("CC(=O)NC1CCC(C)CC1", "N-(4-methylcyclohexyl)acetamide"),
])
def test_the_amide_n_substituent_does_not_depend_on_the_atom_order(smiles, pin):
    """The same helper names an amide's N-substituent: the ring stem it enriches is the ring of the attachment atom,
    so a ring reached through a chain ('(4-methylcyclohexyl)methyl') is named whole, in every atom order."""
    from tests.support.rt_assert import name_is_rt_exact
    assert name_is_rt_exact(pin, smiles)
    mol = Chem.MolFromSmiles(smiles)
    for spelling in sorted({smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(30)})[:8]:
        for res in (name_default(spelling), name_breadth(spelling)):
            assert (res["name"], res["tier"]) == (pin, "pin_verified"), (spelling, res)
