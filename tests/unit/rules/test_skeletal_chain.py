"""The skeletal replacement ('a') PIN of an acyclic chain without a principal group (``rules/skeletal_chain.py``).

``## **** Skeletal replacement ('a') nomenclature in acyclic chains`` (the Blue Book),
(:23348): four or more heterounits in an unbranched chain, none of them part of the principal characteristic group,
give an 'a' PIN. The principal chain is chosen by ``## **** SENIORITY OF ACYCLIC CHAINS`` (:20916), the
numbering by ``### **** NUMBERING`` (:3219) and (:6476): the heteroatoms take the lowest locants
before anything else, and the name is built on that fixed numbering,:23352).
"""

import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.cli import _emit_tier_flags
from orthonym.rules import skeletal_chain
from orthonym.rules.skeletal_chain import name_a_chain
from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
from orthonym.validation.pin_spelling import check_pin_spelling
from orthonym.validation.spelling.checks_skeletal_chain import a_chain_applies
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.unit

CLASS_D = [  # the Task 3 rows of "Expected names" (no leaf group): SMILES -> name
    ("COCCOCCOCCOC", "2,5,8,11-tetraoxadodecane"), ("CCOCCOCCOCCOC", "2,5,8,11-tetraoxatridecane"),
    ("C=CCOCCOCCOCCOC", "2,5,8,11-tetraoxatetradec-13-ene"), ("COC=COCCOCCOC", "2,5,8,11-tetraoxadodec-3-ene"),
    ("C=COCCOCCOCCOCC", "3,6,9,12-tetraoxatetradec-1-ene"),
    ("C#CCOCCOCCOCCOCCOCCI", "1-iodo-3,6,9,12,15-pentaoxaoctadec-17-yne"),
    ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]CCCl", "10-chloro-2,4,6,8-tetrasiladecane"),
    ("ClCCOCCOCCOCCOCCCl", "1,14-dichloro-3,6,9,12-tetraoxatetradecane"),
    ("CSCSCSCC(C)SCSC", "8-methyl-2,4,6,9,11-pentathiadodecane"),
    ("CC(C)OCCOCCOCCOC", "12-methyl-2,5,8,11-tetraoxatridecane"),
    ("CCOC(CC)C(C)OCCOCCOC", "10-ethyl-9-methyl-2,5,8,11-tetraoxatridecane"),
    ("[SiH3]CCOCCOCCOCCOCC", "4,7,10,13-tetraoxa-1-silapentadecane"),
    ("CB(C)COCCOCB(C)C", "2,9-dimethyl-4,7-dioxa-2,9-diboradecane"),
    ("COCCOCC[SiH2]CSC", "7,10-dioxa-2-thia-4-silaundecane"), ("C[SiH2]C[SiH2]C[SiH2]CSCC", "8-thia-2,4,6-trisiladecane"),
]

#: what the builder leaves to the legacy path or to the substitutive namers (value None)
DECLINED = [
    ("C[C@H](O)COCCOCCOCCOC", "stereo"),
    ("[2H]OCCOCCOCCOCCOC", "isotope"),
    ("OCC" + "OCC" * 40 + "O", "124 heavy atoms, over the cap"),
    ("CCC(C(C)C)OCCOCCOCCOC", "tie of two chains with two names"),
    ("CCCOCCOCC(COCCO)OCCOCCO", "star: a branch carries heteroatoms"),
    ("CCNCCCNCCCCNCCCNCC", "secondary amines are the principal group"),
    ("NCCNCCNCCNCCNCC", "one NH2 end"),
    ("C=CCOCCOCCOCCOCCO", "unsaturation with a suffix"),
    ("COCCOOCCOCCOC", "adjacent heteroatoms: legacy path, 2,5,6,9,12-pentaoxatridecane"),
    ("CCCOCCOCC(=O)OCCOCCOC", "ester"),
    ("OC(=O)COCCOCCOCCOCC(O)=O", "acid: legacy path"),
    ("COCCOCCOCCOCCOC.O", "two fragments"),
    ("[NH3+]CCOCCOCCOCCOCCOC", "charge"),
    ("CCOCCOCCOCC", "3 heterounits"),
]


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


@pytest.mark.parametrize("smiles, name", CLASS_D, ids=[s for s, _ in CLASS_D])
def test_class_d_names(smiles, name):
    assert name_a_chain(_mol(smiles)) == name


@pytest.mark.parametrize("smiles, why", DECLINED, ids=[why for _, why in DECLINED])
def test_builder_declines(smiles, why):
    assert name_a_chain(_mol(smiles)) is None, why


def test_builder_declines_none():
    assert name_a_chain(None) is None


def test_legacy_alone(monkeypatch):
    """With the builder off, the legacy path still names a plain chain and an -ol chain (the hook changes nothing else)."""
    monkeypatch.setattr(skeletal_chain, "name_a_chain", lambda mol: None)
    assert try_skeletal_replacement_name(_mol("COCCOCCOCCOC")) == "2,5,8,11-tetraoxadodecane"
    assert try_skeletal_replacement_name(_mol("OCCOCCOCCOCCOC")) == "2,5,8,11-tetraoxatridecan-13-ol"


def test_the_legacy_function_returns_the_builder_name(monkeypatch):
    """The hook: a name from the builder is returned by the legacy function before its own gates, so a chain the legacy
    path declines (an iodo-yne ether chain) is named, and a ring never reaches the builder."""
    assert try_skeletal_replacement_name(_mol("C#CCOCCOCCOCCOCCOCCI")) == "1-iodo-3,6,9,12,15-pentaoxaoctadec-17-yne"
    monkeypatch.setattr(skeletal_chain, "name_a_chain", lambda mol: "from the builder")
    assert try_skeletal_replacement_name(_mol("COCCOCCOCCOC")) == "from the builder"

    def refuse(mol):
        raise AssertionError("a ring reached the builder")
    monkeypatch.setattr(skeletal_chain, "name_a_chain", refuse)
    try_skeletal_replacement_name(_mol("C1COCCOCCOCCOCCO1"))


#: rows that decide one numbering criterion each (a), (c), (e), (f), (g),:3219;,:6476)
NUMBERING = [
    # heteroatom sets tie ({2,4,6,8}); O is cited first, so O takes the lower locants {2,6}, not {4,8}
    ("COCSCOCSC", "2,6-dioxa-4,8-dithianonane"),
    # everything ties but the prefixes: {3,10} either way; chloro is cited first and takes 3 (g))
    ("COC(C)COCCOCC(Cl)OC", "3-chloro-10-methyl-2,5,8,11-tetraoxadodecane"),
    # the multiple bonds as a set {2,16} tie; the double bond takes the lower locant,:6502)
    ("CC#CCOCCOCCOCCOCC=CC", "5,8,11,14-tetraoxaoctadec-2-en-16-yne"),
    # two methyl groups on each of two atoms: the multiplier counts all four
    ("CC(C)(C)OCCOCCOCCOC(C)(C)C", "2,2,13,13-tetramethyl-3,6,9,12-tetraoxatetradecane"),
    # an allene: the double bonds are 12 and 13 (the locant of a bond is that of its lower-numbered atom,
    ("C=C=COCCOCCOCCOC", "2,5,8,11-tetraoxatetradeca-12,13-diene"),
]


@pytest.mark.parametrize("smiles, name", NUMBERING, ids=[s for s, _ in NUMBERING])
def test_numbering_criteria(smiles, name):
    assert name_a_chain(_mol(smiles)) == name


@pytest.mark.parametrize("smiles", [s for s, _ in CLASS_D + NUMBERING], ids=lambda s: s)
def test_the_atom_order_does_not_matter(smiles):
    """Determinism: the same molecule written in another atom order has the same name."""
    mol = _mol(smiles)
    expected = name_a_chain(mol)
    assert expected is not None
    order = list(range(mol.GetNumAtoms()))
    for permutation in (order[::-1], order[3:] + order[:3], order[::2] + order[1::2]):
        assert name_a_chain(Chem.RenumberAtoms(mol, permutation)) == expected


def test_the_principal_chain_is_the_longer_path():
    """ (:20938), criterion (b): the chain with more skeletal atoms; the iPr group is a methyl on the long chain."""
    mol = _mol("CC(C)OCCOCCOCCOC")
    cands = skeletal_chain._candidates(mol)
    best = max(skeletal_chain._rank(mol, c) for c in cands)
    assert best[:3] == (0, 4, 13)    # no principal group, four heteroatoms, thirteen skeletal atoms


@pytest.mark.parametrize("smiles, name", [
    ("COCCOCCOCCOC(CCC)C(C)(C)C", "12-tert-butyl-2,5,8,11-tetraoxapentadecane"),   #,:24412
    ("COCCOCCOCCOC(CCC)C(C)C", "12-(propan-2-yl)-2,5,8,11-tetraoxapentadecane"),   #,:15935
    ("COCCOCCOCCOC(CCC)CC", "12-ethyl-2,5,8,11-tetraoxapentadecane"),              # the unbranched alkyls, 'ethyl' here
])
def test_branched_pendants(smiles, name):
    """The chain is the longer path; the group left over is named 'tert-butyl' / 'propan-2-yl' / 'ethyl'."""
    assert name_a_chain(_mol(smiles)) == name


@pytest.mark.parametrize("smiles", [
    "COCCOCCOCCOC(CC(C)C)CC(C)C",              # 2-methylpropyl on every chain through the carbon
    "COCCOCCOCCOC(C=C)C=C",                    # a vinyl group is not saturated
    "COCCOCCOCCOC(C(Cl)(Cl)Cl)C(Cl)(Cl)Cl",   # trichloromethyl holds halogens in a part that is not a halogen
    "Cl[SiH2]C[SiH2]C[SiH2]C[SiH2]C",          # a halogen on a heteroatom
    # the longer chain holds a part outside the vocabulary,:20938, takes the longer chain): the shorter chain,
    # with that part as a prefix, is not the principal one, so no name; its 'a' PIN is e.g.
    # 13-(butan-2-yl)-2,5,8,11-tetraoxaheptadecane
    "COCCOCCOCCOCC(CCCC)C(C)CC",               # butan-2-yl
    "COCCOCCOCCOCC(CCCC)CCl",                  # chloromethyl
    "COCCOCCOCCOCC(CCCC)C=C",                  # ethenyl
    "COCCOCCOCCOCC(CC(C)C)CCCCC",              # 2-methylpropyl beside a longer pentyl
    "OCCOCCOCCOCCOCC(CN)CCCC",                 # aminomethyl, with -ol
    "OCCOCCOCCOCCOCC(CCCC)C(C)CC",             # butan-2-yl, with -ol
])
def test_pendants_are_a_closed_vocabulary(smiles):
    """Only a halogen on a carbon, an unbranched alkyl, propan-2-yl and tert-butyl hang on the chain, and a longer chain
    that holds anything else is not passed over for a shorter one."""
    assert name_a_chain(_mol(smiles)) is None


def test_principal_groups_come_before_the_length_of_the_chain():
    """ (:18875) comes before: the 14-atom chain that ends at the CH2OH group has the -ol group; the
    17-atom chain through the butyl group has none, so the shorter chain is the principal one."""
    assert name_a_chain(_mol("COCCOCCOCCOCC(CCCC)CO")) == "13-butyl-2,5,8,11-tetraoxatetradecan-14-ol"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["COCCOCCOCCOCC(CCCC)C(C)CC", "OCCOCCOCCOCCOCC(CN)CCCC"])
def test_a_declined_longer_chain_is_named_substitutively_not_on_a_shorter_chain(smiles):
    """What ships after the decline is a substitutive name that reads back, below the PIN (S5: the molecule has an 'a'
    PIN), not the name of a shorter chain."""
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["tier"] == "systematic_verified", row
    assert "tetraoxa" not in row["name"], row
    assert_full_rt(row["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [CLASS_D[0], CLASS_D[5], CLASS_D[12]], ids=["tetraoxadodecane", "iodo-yne", "boranes"])
def test_end_to_end(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert row["spelling_failures"] == [], row


# --- Task 4: -ol, -thiol, -amine and the N policy ----------------------------------------------------------------

SUFFIX = [  # the Task 4 rows of "Expected names": SMILES -> name
    # -ol: the heteroatom set decides before the suffix (a) before (c);,:23352), then -ol is lowest
    ("OCCOCCOCCOCCOC", "2,5,8,11-tetraoxatridecan-13-ol"),
    ("OCCOCCOCCOCCOCC", "3,6,9,12-tetraoxatetradecan-1-ol"),
    ("COCCOCCOCCOCCOCCOCCO", "2,5,8,11,14,17-hexaoxanonadecan-19-ol"),
    ("CC(O)COCCOCCOCCOC", "2,5,8,11-tetraoxatetradecan-13-ol"),
    ("OCCOCCOCCOCCOCCO", "3,6,9,12-tetraoxatetradecane-1,14-diol"),
    ("OCCOCCOCCOCCOCC(O)CO", "3,6,9,12-tetraoxapentadecane-1,14,15-triol"),
    ("OCCOCCOCCOCCOCCOCCOCCO", "3,6,9,12,15,18-hexaoxaicosane-1,20-diol"),
    ("CC(O)[SiH2]C[SiH2]C[SiH2]C[SiH2]CO", "2,4,6,8-tetrasiladecane-1,9-diol"),         # BB:21335
    ("OCC[SiH2]C[SiH2]C[SiH2]C[SiH2]CO", "2,4,6,8-tetrasiladecane-1,10-diol"),          # BB:21339
    # -thiol, -amine
    ("SCCOCCOCCOCCOCCS", "3,6,9,12-tetraoxatetradecane-1,14-dithiol"),
    ("SCCOCCOCCOCCOCC", "3,6,9,12-tetraoxatetradecane-1-thiol"),
    ("NCCOCCOCCOCCOCCN", "3,6,9,12-tetraoxatetradecane-1,14-diamine"),
    ("NCCOCCOCCOCCOCC", "3,6,9,12-tetraoxatetradecan-1-amine"),
    ("COCCOCCOCCOC(CN)C(C)(C)C", "12-tert-butyl-2,5,8,11-tetraoxatridecan-13-amine"),   # the chain runs through CH2NH2
    # the N policy: the polyamine of (BB:26400) has NH2 on both ends
    ("NCCNCCNCCNCCNCCN", "3,6,9,12-tetraazatetradecane-1,14-diamine"),
    ("NCCCNCCCCNCCCNCCCNCCCN", "4,8,12,17-tetraazaicosane-1,20-diamine"),
    # mixed groups: -ol is senior to -thiol, which is senior to -amine,:18162); the others are prefixes
    ("NCCOCCOCCOCCOCCO", "14-amino-3,6,9,12-tetraoxatetradecan-1-ol"),
    ("CCOCCOCC(O)CNCCOCC(C)C", "15-methyl-3,6,13-trioxa-10-azahexadecan-8-ol"),        # N is not principal: K is OH
    ("OCCOCCOCCOCCOCCF", "14-fluoro-3,6,9,12-tetraoxatetradecan-1-ol"),
    ("CSC[SiH2]CSC[SiH2]CCN", "2,6-dithia-4,8-disiladecan-10-amine"),
]


@pytest.mark.parametrize("smiles, name", SUFFIX, ids=[s for s, _ in SUFFIX])
def test_suffix_names(smiles, name):
    assert name_a_chain(_mol(smiles)) == name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    ("CPCOCCOCCOC", "4,7,10-trioxa-2-phosphaundecane"),
    ("CP(C)COCCOCCOCC", "2-methyl-4,7,10-trioxa-2-phosphadodecane"),
    ("CCOCCOCCOCCP(C)C", "2-methyl-5,8,11-trioxa-2-phosphatridecane"),
], ids=lambda smiles: smiles)
def test_a_phosphane_chain_has_its_a_pin(smiles, name):
    """ (:23352-23358) gives '3-phospha-2,5,7-trisilaoctane [PIN]' and calls the phosphane name of the same
    chain 'a substitutive name': the phosphane regime of ``mononuclear_hydrides`` leaves a chain that has an 'a' name
    to the skeletal route, at the PIN tier and at the default tier, and the name reads back."""
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert row["spelling_failures"] == [], row
    assert name_compound(smiles) == name
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles", [s for s, _ in CLASS_D + SUFFIX], ids=lambda s: s)
def test_builder_implies_check(smiles):
    """Independence (global constraints): wherever the builder names a molecule, the check, which shares none of its
    code, reads the molecule as an 'a' chain, so the label machinery agrees with the builder."""
    mol = _mol(smiles)
    assert name_a_chain(mol) is not None
    assert a_chain_applies(mol), smiles


@pytest.mark.parametrize("smiles, name", [
    ("SCCOCCOCCOCCOCCO", "14-sulfanyl-3,6,9,12-tetraoxatetradecan-1-ol"),            # -ol outranks -thiol
    ("NCCOCCOCCOCCOCCS", "14-amino-3,6,9,12-tetraoxatetradecane-1-thiol"),            # -thiol outranks -amine
    ("OCCOCCOCCOCCOCCCl", "14-chloro-3,6,9,12-tetraoxatetradecan-1-ol"),
    ("OCCOCCOCCOCCOC(C)(C)C", "13,13-dimethyl-3,6,9,12-tetraoxatetradecan-1-ol"),
], ids=lambda smiles: smiles)
def test_junior_groups_are_prefixes(smiles, name):
    """ / (:18162): every group of the senior kind is a suffix, a junior one is a prefix."""
    assert name_a_chain(_mol(smiles)) == name


#: a chain nitrogen is a skeletal atom only when another group is principal, or in the two-ends polyamine (S2:
#: Polyamines,:26324, '3,6,9,12-tetraazatetradecane-1,14-diamine (PIN)',:26400)
N_DECLINED = [
    ("CCNCCCNCCCCNCCCNCC", "secondary amines are the principal group (no NH2 at all)"),
    ("NCCNCCNCCNCCNCC", "one NH2 end: the other nitrogens are amine groups too"),
    ("NCCNCCNCC(N)CNCCNC", "an NH2 on a chain carbon that is not an end"),
    # (:20938) takes the longer chain, whose NH2 are not on its ends: a shorter chain that the policy accepts
    # (ending at the CH(NH2) carbon, the alkyl group as a prefix) is not the principal chain, so nothing is named
    ("CC(N)NCCNCCNCCNCCN", "the longer chain through the methyl group is refused by the N policy"),
    ("CCC(N)NCCNCCNCCNCCN", "the longer chain through the ethyl group is refused by the N policy"),
    ("NCCNCCNCCN(C)CCNCCN", "a chain nitrogen with a substituent (degree 3)"),
    ("CNCCNCCNCCNCCN", "one end is a methylamino group, not NH2"),
    ("NCC(N)OCCOCCOCCOCCN", "three NH2 groups"),
    ("NC(N)COCCOCCOCCOCCOC", "two NH2 groups on one carbon"),
]


@pytest.mark.parametrize("smiles, why", N_DECLINED, ids=[why for _, why in N_DECLINED])
def test_n_policy(smiles, why):
    assert name_a_chain(_mol(smiles)) is None, why


def test_n_policy_names_the_polyamine():
    assert name_a_chain(_mol("NCCNCCNCCNCCNCCN")) == "3,6,9,12-tetraazatetradecane-1,14-diamine"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [SUFFIX[4], SUFFIX[14], SUFFIX[16]], ids=["diol", "polyamine", "amino-ol"])
def test_suffix_end_to_end(smiles, name):
    """The default tier ships the builder's name at ``pin_verified``, and no spelling check fails on it."""
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles", [s for s, _ in SUFFIX], ids=lambda s: s)
def test_suffix_atom_order_does_not_matter(smiles):
    mol = _mol(smiles)
    expected = name_a_chain(mol)
    assert expected is not None
    order = list(range(mol.GetNumAtoms()))
    for permutation in (order[::-1], order[5:] + order[:5], order[::2] + order[1::2]):
        assert name_a_chain(Chem.RenumberAtoms(mol, permutation)) == expected


PANEL_A_ROWS = [  # the 34 rows of the fold panel that main already names with an 'a' block (SMILES)
    "CCOCCOCCOCCOCCOCCOCC", "CCCCCCCCOCCOCCOCCOCCO", "CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCO",
    "COCCO[C@@H](C)CO[C@H](C)CO[C@H](C)CO[C@H](C)CO[C@@H](C)CO[C@@H](C)CO[C@H](C)CO[C@H](C)COC[C@H](C)N",
    "CCOCCOCCOCCOCCOCC", "COCCOCCOCCOCCOCCOCCO", "COCCOCCOCCOC", "COCCOCCOCCOCCOCCOC", "COCCOCCOCCOCCOC",
    "COCCOCCOCCOCCOCCOCCOCCOCCOCCO", "CCCCCCCCCCCCOCCOCCOCCOCCOCCO", "CCOC(C)OCC(C)OC(C)OCC", "CCCCOCCOCC(C)OCCCOC",
    "CCCCCCCCCCCCOCCOCCOCCOCCO", "CCCCCCCCCCCCCCOCCOCCOCCOCCOCCOCCO", "CCCCCCCCCOCCOCCOCCOCCOCCO", "COCCOCCOCCOCCO",
    "CC(C)(C)CCOCCOCCOCCOCCOCCOCCOCCOC(C)(C)C", "CC(C)(C)OOC(=O)COCCOCCOCCOCC(=O)OOC(C)(C)C",
    "CN(C)C(OPOS(=O)(=O)OPOC(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C", "CC(C)(C)CCOCCOCCOCCOCCOCCOC(C)(C)C",
    "C=CCOCC(O)COCCC(C)(C)OCCOCC(C)OCCOCC(O)COCCOC=C", "CC(C)(C)OCCOCCOCCOCCOCCOC(C)(C)C",
    "CCCCCOCCOCCOCCOCCOCCOCCCC", "COCCCCOCCCCOCCCCOCCCCO", "CCCCCCCCCCCOCCOCCOCCOCCOCC(=O)O",
    "CCCCCCCCSCCCCSCCCCCCCCCSCCCCSCCCC", "O=C(CSCCS(=O)CSCSC(=O)CSCCOOCCSCSCO)OCCSCSCO", "COCCOCOCCOCCO",
    "C=COCOCC(O)(COCOC(=C)C(C)O)OCOC(=C)C(C)O", "C[SiH2]C[Si](C)(C)C[Si](C)(C)C[SiH2]C",
    "CCC(C)COCC(C)COCC(C)COCC(C)COCC(C)COCC(C)C", "CSCCOCCSCCOCCOCCSCCOCCSC", "C=COCCOCCOCCOC",
]


def test_parity_with_legacy(monkeypatch):
    """On the panel rows the builder names, the legacy path (the builder off) gives the same name, or its name breaks
     (a numbering that is not the lowest: the independent justification of the difference), or it has none."""
    built = {}
    for smiles in PANEL_A_ROWS:
        name = name_a_chain(_mol(smiles))
        if name is not None:
            built[smiles] = name
    assert len(built) >= 27, len(built)
    monkeypatch.setattr(skeletal_chain, "name_a_chain", lambda mol: None)
    for smiles, name in built.items():
        mol = _mol(smiles)
        legacy = try_skeletal_replacement_name(mol)
        if legacy is None or legacy == name:
            continue
        rules = [f.rule for f in check_pin_spelling(mol, legacy, strict=True)]
        assert "P-14.4" in rules, (smiles, name, legacy, rules)
