"""Leads program L4, item 41: a multiplicative PIN is never labelled on a substitutive name.

 "Preferred IUPAC multiplicative names" (the Blue Book): "Multiplicative
nomenclature is preferred to substitutive nomenclature for generating preferred IUPAC
names to express multiple occurrences of identical parent structures" when the linking
bonds are identical, the multiplicative groups other than the central one are
symmetrically substituted and the locants of all substituent groups, including suffix
groups, are identical (:23180-23184). (:23174) names skeletal replacement ('a')
and phane names ahead of a multiplicative name when their conditions are met.

``multiplicative_pin_expected`` claimed only ring units that own a principal group. Two
classes slipped through and shipped a substitutive name at ``pin_verified``:

(a) chain units joined through a ring or a heteroatom: '3,3'-(1,4-phenylene)di(propan-1-ol)
    (PIN)' "Selection of parent compounds",:25171), '1,1'-(ferrocene-1,1'-diyl)
    di(ethan-1-one) (PIN)' "Ocenes",:40133); '3,6,9,12-tetraoxatetradecane-1,14-
    dioic acid [PIN, a skeletal replacement ('a') name]' (:23212);
(b) ring units with no principal group joined by a symmetrical central group
    ,:23172: "identical parent structures do not have to have a principal
    characteristic group in order to construct a multiplicative name").

The label is withdrawn (the PIN tier declines, the wider tiers keep the correct
substitutive name); the multiplicative producer for these classes is a separate item.
"""
import inspect

import pytest
from rdkit import Chem

from orthonym.rules.pin_vocabulary import multiplicative_pin_expected
from tests.support.pin_tiers import assert_declined_at_default, name_default
from tests.support.rt_assert import name_is_rt_exact  # noqa: F401 (also used below)

pytestmark = pytest.mark.opsin_gate

# (smiles, the PIN) -- every PIN below was read back by OPSIN 2.9.0 to the full InChIKey
# of its SMILES while this test was written (see the read-back test).
CHAIN_UNIT_ROWS = [
    ("CC(=O)CC1CCC(CC1)CC(=O)C", "1,1'-(cyclohexane-1,4-diyl)di(propan-2-one)"),
    ("CC(=O)CCOCCC(C)=O", "4,4'-oxydi(butan-2-one)"),
    ("CC(=O)c1ccc(C(C)=O)cc1", "1,1'-(1,4-phenylene)di(ethan-1-one)"),
    ("CC(=O)C1=CC(=CC(=C1)N=C=O)C(=O)C", "1,1'-(5-isocyanato-1,3-phenylene)di(ethan-1-one)"),
    ("O=C(O)CCCCCCCCCOc1ccc(-c2ccc(OCCCCCCCCCC(=O)O)cc2)cc1",
     "10,10'-[[1,1'-biphenyl]-4,4'-diylbis(oxy)]di(decanoic acid)"),
    ("OCCOCCOCCOCCOCCOCCOCCO", "3,6,9,12,15,18-hexaoxaicosane-1,20-diol"),
]

# The molecules whose substitutive name is not the PIN: the label predicate is True.
EXPECTED_TRUE = [s for s, _ in CHAIN_UNIT_ROWS] + [
    "OC(=O)COCC(O)=O",                      # 2,2'-oxydiacetic acid (BB BB row, 2,2'-sulfanediyl analogue:23192)
    "N#CCCNCCC#N",                          # 3,3'-azanediyldipropanenitrile (PIN),:34831
    "NCCOCCN",                              # 2,2'-oxydi(ethan-1-amine) (PIN),:6124
    "OCCSCCO",                              # 2,2'-sulfanediyldi(ethan-1-ol) (PIN),:5813
    "OC(=O)CCc1ccc(CCC(O)=O)cc1",           # 3,3'-(1,4-phenylene)dipropanoic acid
    "OCCCc1ccc(CCCO)cc1",                   # 3,3'-(1,4-phenylene)di(propan-1-ol) (PIN):25171
    "OC(=O)CN(CCN(CC(O)=O)CC(O)=O)CC(O)=O",  # a nitrilo linker under four acid groups
    "c1cc(OCCOCCOCCOCCOCCOc2cocc2OCCOCCOCCOCCOCCOc2ccoc2)co1",  # BB furan row
    "c1ccc(Oc2ccccc2)cc1",                  # 1,1'-oxydibenzene (PIN), no principal group
    "c1ccccc1COCc1ccccc1",                  # 1,1'-[oxybis(methylene)]dibenzene
]

# No claim: a single chain carries the groups, the groups are on a ring, a linker atom is a
# second principal group, the units are not symmetrical, or the parent is something else.
EXPECTED_FALSE = [
    ("OC(=O)CCCC(O)=O", "one chain carries both acids (pentanedioic acid)"),
    ("OC(=O)CC(=O)O", "one chain carries both acids"),
    ("CC(=O)CC(C)=O", "one chain carries both ketones"),
    ("OC(=O)OC(O)=O", "dicarbonic acid: the acid carbon is bonded to oxygen"),
    ("O=C(O)NC(=O)O", "2-imidodicarbonic acid"),
    ("O=C(O)OC(=O)OC(=O)O", "tricarbonic acid"),
    ("NCCNCCN", "the NH linker is itself an amine: the polyamine takes the substitutive name (P-62.2.4, :26398)"),
    ("OC(=O)c1ccc(C(O)=O)cc1", "the acids are ring suffixes of one benzene ring"),
    ("CC(=O)OCCOC(C)=O", "esters are named by functional class"),
    ("c1ccccc1COc1ccccc1", "an unsymmetrical -O-CH2- linker (benzyl phenyl ether)"),
    ("c1ccc(-c2ccccc2)cc1", "a ring assembly, not units joined by a central group"),
    ("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "terphenyl: the units are bonded to an identical ring (P-28.2)"),
    ("C[Si](C)(C)c1ccc(Cc2ccc([Si](C)(C)C)cc2)cc1", "a silane is the senior parent hydride (P-44.1.2.2)"),
    ("c1ccccc1Cc1ccc(Cc2ccccc2)nc1", "the repeated benzene is not the senior ring (pyridine)"),
    ("c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "four rings on a seven-node chain: a linear phane (P-52.2.5.1, :23826)"),
    ("CCCC(=O)OCC", "one group only"),
    ("OC(=O)CC(O)(CC(O)=O)C(O)=O", "citric acid: the three acids are not one symmetry class"),
]


@pytest.mark.parametrize("smiles", EXPECTED_TRUE)
def test_substitutive_name_is_not_the_pin(smiles):
    assert multiplicative_pin_expected(Chem.MolFromSmiles(smiles)) is True


@pytest.mark.parametrize("smiles,why", EXPECTED_FALSE)
def test_no_multiplicative_claim(smiles, why):
    assert multiplicative_pin_expected(Chem.MolFromSmiles(smiles)) is False, why


@pytest.mark.parametrize("smiles,pin", CHAIN_UNIT_ROWS)
def test_expected_pin_reads_back(smiles, pin):
    """The multiplicative or 'a' PIN is the same molecule: OPSIN's full-key read-back."""
    assert name_is_rt_exact(pin, smiles), pin


@pytest.mark.parametrize("smiles", [s for s, _ in CHAIN_UNIT_ROWS])
def test_default_tier_declines_the_substitutive_name(smiles):
    """The PIN tier no longer ships the substitutive name as a PIN; the wider tier keeps a
    correct name (round-tripped by ``assert_declined_at_default``)."""
    assert_declined_at_default(smiles)


# The label rule must not lower the names of the multiplicative, skeletal-replacement,
# amino-acid and carbonic-acid routes: they are PINs by their own nomenclature. The
# homocystine, L-cystine and tetrathia-diacid rows need the dispatch-class skip in namer.py
# (``name_tiered``, the tuple before ``multiplicative_pin_expected``): without it the label
# rule reads their names too.
def _namer_skips_own_route_names() -> bool:
    import orthonym.namer as namer
    src = inspect.getsource(namer)
    i = src.index("multiplicative_pin_expected(_mol_mult)")
    window = src[max(0, i - 900):i]
    return all(c in window for c in ("SKELETAL_REPLACEMENT", "INORGANIC_ACID",
                                      "RETAINED_NAME", "AMINO_ACID"))


NEEDS_NAMER_SKIP = pytest.mark.xfail(
    not _namer_skips_own_route_names(), strict=True,
    reason="needs the dispatch-class skip of namer.py (patch L4-41-namer-dispatch-skip)")

KEEP_PIN_ROWS = [
    pytest.param("N[C@@H](CCSSCC[C@H](N)C(=O)O)C(=O)O", "homocystine", marks=NEEDS_NAMER_SKIP,
                 id="homocystine-amino-acid-route"),
    pytest.param("O=C(O)CSCCSCCCSCCSCC(=O)O", "3,6,10,13-tetrathiapentadecanedioic acid",
                 marks=NEEDS_NAMER_SKIP, id="tetrathia-diacid-skeletal-replacement"),
    # a retained name: the L-cystine PIN is a retained amino-acid name, not a substitutive diacid
    pytest.param("N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O", "L-cystine", marks=NEEDS_NAMER_SKIP,
                 id="cystine-retained-name"),
    pytest.param("O=C(O)OC(=O)O", "dicarbonic acid", id="dicarbonic-acid"),
    pytest.param("O=C(O)NC(=O)O", "2-imidodicarbonic acid", id="imidodicarbonic-acid"),
    pytest.param("OCCCc1ccc(CCCO)cc1", "3,3'-(1,4-phenylene)di(propan-1-ol)", id="multiplicative-diol"),
    pytest.param("O=C(O)COCC(=O)O", "2,2'-oxydiacetic acid", id="multiplicative-diacid"),
    pytest.param("c1ccc(Oc2ccccc2)cc1", "1,1'-oxydibenzene", id="multiplicative-no-group"),
]


@pytest.mark.parametrize("smiles,name", KEEP_PIN_ROWS)
def test_own_route_pins_keep_their_label(smiles, name):
    res = name_default(smiles)
    assert (res.get("name"), res.get("tier")) == (name, "pin_verified"), res
