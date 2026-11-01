"""Oligonucleotides are named with the prefix names of their nucleotides,.

 "DERIVATIVES OF NUCLEOTIDES" (the Blue Book; the paragraph at:55065): "Oligonucleotides are named using
the prefix names derived from the traditional names for the nucleotides", with the example
`2'-deoxyguanylyl-(3'->5')-2'-deoxyuridylyl-(3'->5')-2'-deoxyguanosine`. (:55055): the prefix names replace
'ic acid' by 'yl'; "the substituent prefix name from inosinic acid is an exception; it is named inosinylyl".
 (:55071): "When a phosphorothioic acid is used (HS-P in place of HO-P), the prefix P-thio is added at the
front of the name of the nucleotide", with `2'-deoxy-P-thioguanylyl-(3'->5')-2'-deoxy-P-thiouridylyl-(3'->5')-
2'-deoxyguanosine`. INTRODUCTION (:50939): "Preferred IUPAC names (PINs) are not identified for the compounds in
this Chapter": the name is a systematic one, `systematic_verified`, and the PIN tier does not ship it.

Every expected name is read back by an independent OPSIN call (``name_is_rt_exact``); the structures are built from
nucleosides (``tests/support/oligonucleotide_smiles.py``) or written out.
"""
import contextlib

import pytest
from rdkit import Chem

from orthonym.rules import nucleosides as nuc
from orthonym.rules.oligonucleotides import name_oligonucleotide
from tests.support.oligonucleotide_smiles import oligo
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit

# the book's two examples, written as the book draws them (OPSIN 2.9.0 reads the names to these structures)
_BOOK_DGUDG = ("[C@@H]1(C[C@H](OP(=O)(O)OC[C@@H]2[C@H](C[C@@H](O2)N2C(=O)NC(=O)C=C2)OP(=O)(O)OC[C@@H]2[C@H](C[C@@H]"
               "(O2)N2C=NC=3C(=O)NC(N)=NC23)O)[C@@H](CO)O1)N1C=NC=2C(=O)NC(N)=NC12")
_BOOK_DGSUSG = _BOOK_DGUDG.replace("P(=O)", "P(=S)")
_APU = oligo([("r", "A"), ("r", "U")])

_NAMED = [
    # (:55065) and (:55071): the book's examples, verbatim
    (_BOOK_DGUDG, "2'-deoxyguanylyl-(3'→5')-2'-deoxyuridylyl-(3'→5')-2'-deoxyguanosine"),
    (_BOOK_DGSUSG, "2'-deoxy-P-thioguanylyl-(3'→5')-2'-deoxy-P-thiouridylyl-(3'→5')-2'-deoxyguanosine"),
    # the same through the builder: the structure of the builder is the book's
    (oligo([("d", "G"), ("d", "U"), ("d", "G")]), "2'-deoxyguanylyl-(3'→5')-2'-deoxyuridylyl-(3'→5')-2'-deoxyguanosine"),
    (oligo([("d", "G"), ("d", "U"), ("d", "G")], thio=(0, 1)),
     "2'-deoxy-P-thioguanylyl-(3'→5')-2'-deoxy-P-thiouridylyl-(3'→5')-2'-deoxyguanosine"),
    (_APU, "adenylyl-(3'→5')-uridine"),
    (oligo([("d", "T"), ("d", "T")]), "thymidylyl-(3'→5')-thymidine"),         # thymidine states the 2'-deoxy itself
    (oligo([("r", "A")] * 3), "adenylyl-(3'→5')-adenylyl-(3'→5')-adenosine"),
    (oligo([("r", "C"), ("r", "G")]), "cytidylyl-(3'→5')-guanosine"),
    # (:55055): inosinylyl is the exception; xanthylyl follows xanthylic acid,:55003)
    (oligo([("r", "I"), ("r", "X")]), "inosinylyl-(3'→5')-xanthosine"),
    (oligo([("r", "X"), ("r", "I"), ("d", "T")]), "xanthylyl-(3'→5')-inosinylyl-(3'→5')-thymidine"),
    (oligo([("d", "I"), ("d", "I")]), "2'-deoxyinosinylyl-(3'→5')-2'-deoxyinosine"),
    # ribo and 2'-deoxy units in one chain: each prefix states its own sugar
    (oligo([("d", "A"), ("r", "U")]), "2'-deoxyadenylyl-(3'→5')-uridine"),
    # P-thio on one link: the prefix of the unit whose 3'-phosphate it is; 'deoxy' before 'thio'
    (oligo([("r", "A"), ("r", "U")], thio=(0,)), "P-thioadenylyl-(3'→5')-uridine"),
    (oligo([("r", "A")] * 3, thio=(1,)), "adenylyl-(3'→5')-P-thioadenylyl-(3'→5')-adenosine"),
    (oligo([("d", "A"), ("r", "I"), ("d", "C")], thio=(0, 1)),
     "2'-deoxy-P-thioadenylyl-(3'→5')-P-thioinosinylyl-(3'→5')-2'-deoxycytidine"),
    (oligo([("d", "T"), ("d", "C"), ("d", "A"), ("d", "G")], thio=(1,)),
     "thymidylyl-(3'→5')-2'-deoxy-P-thiocytidylyl-(3'→5')-2'-deoxyadenylyl-(3'→5')-2'-deoxyguanosine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, expected", _NAMED)
def test_the_producer_builds_the_book_name(smiles, expected):
    assert name_oligonucleotide(Chem.MolFromSmiles(smiles)) == expected
    assert name_is_rt_exact(expected, smiles), "OPSIN does not read the name back to the structure"


def test_the_thio_tautomer_is_the_same_compound():
    """P(=O)(SH) and P(=S)(OH) have one standard InChIKey (the mobile H sits on the P-S/P-O pair), and the P-thio of
    the book does not say which: both drawings get the name."""
    thione = oligo([("r", "A"), ("r", "U")], thio=(0,))
    thiol = thione.replace("P(O)(=S)", "P(S)(=O)").replace("P(=S)(O)", "P(=O)(S)")
    assert thiol != thione
    assert name_oligonucleotide(Chem.MolFromSmiles(thiol)) == "P-thioadenylyl-(3'→5')-uridine"


def test_a_defined_p_of_a_plain_phosphodiester_is_no_stereocentre():
    """The two non-bridging oxygens of P(=O)(OH)(OR)(OR') are one by proton exchange, and InChI sees no centre:
    a P tag of the input states nothing the name could leave out."""
    tagged = _APU.replace("P(=O)(O)", "[P@](=O)(O)")
    assert tagged != _APU and Chem.MolFromSmiles(tagged) is not None
    assert name_oligonucleotide(Chem.MolFromSmiles(tagged)) == "adenylyl-(3'→5')-uridine"


# ---------------------------------------------------------------------------------------------------------------
# what the name does not state: the producer returns None
# ---------------------------------------------------------------------------------------------------------------
_U_5P = "OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O"          # the second unit of ApU, from its 5'-O
_REFUSED = {
    "2'-5' link": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1OP(=O)(O)" + _U_5P,
    "a 3'-3' link": oligo([("r", "A"), ("r", "U")], ends=[("3'", "3'")]),
    "a 5'-5' link": oligo([("r", "A"), ("r", "U")], ends=[("5'", "5'")]),
    "a unit with a 3'- and a 2'-phosphodiester (branched)":
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](OP(=O)(O)" + _U_5P + ")[C@H]1OP(=O)(O)" + _U_5P,
    "a phosphotriester (three sugars on one P)":
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](OP(=O)(" + _U_5P + ")" + _U_5P + ")[C@H]1O",
    "a 5'-phosphate terminus (out of scope)":
        "P(=O)(O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(N)=NC=NC12)O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)"
        "N1C(=O)NC(=O)C=C1)O)O",
    "a 3'-phosphate terminus (out of scope)":
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2OP(=O)(O)O)"
        "[C@H]1O",
    "an undefined C1'": _APU.replace("[C@@H]1O[C@H](CO)", "C1O[C@H](CO)", 1),
    "an undefined C4'": _APU.replace("[C@H](CO)", "C(CO)", 1),
    "the alpha anomer of one unit": _APU.replace("[C@@H](n3ccc", "[C@H](n3ccc"),
    "N6-methyladenine": _APU.replace("Nc1ncnc2", "CNc1ncnc2", 1),
    "5-methyluridine (a ribo unit has no thymidylyl)": oligo([("r", "M"), ("r", "A")]),
    "5-methylcytosine": oligo([("d", "A"), ("d", "M")]),
    "8-bromoadenine": _APU.replace("Nc1ncnc2c1ncn2", "Nc1ncnc2c1nc(Br)n2", 1),
    "a charged phosphate": _APU.replace("P(=O)(O)", "P(=O)([O-])"),
    "a methyl phosphotriester": _APU.replace("P(=O)(O)", "P(=O)(OC)"),
    "a P-thio with a defined P": _APU.replace("P(=O)(O)", "[P@](=S)(O)"),
    "a dithio link": _APU.replace("P(=O)(O)", "P(=S)(S)"),
    "a pyrophosphate link": _APU.replace("OP(=O)(O)OC[C@H]2", "OP(=O)(O)OP(=O)(O)OC[C@H]2"),
    "a 2'-O-methyl": _APU.replace("[C@H](O)[C@@H]2O", "[C@H](OC)[C@@H]2O"),
    "a 5'-O-acetyl": _APU.replace("[C@H](CO)", "[C@H](COC(C)=O)", 1),
    "a 3'-O-acetyl": _APU.replace("[C@@H]2O", "[C@@H]2OC(C)=O"),
    "a 13C": _APU.replace("[C@H](CO)", "[C@H]([13CH2]O)", 1),
    "a salt (a second component)": _APU + ".[Na+].[Cl-]",
    "water of crystallisation": _APU + ".O",
    "a single nucleoside": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
    "a 5'-nucleotide": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
}


@pytest.mark.parametrize("label", sorted(_REFUSED))
def test_the_producer_refuses_what_its_name_does_not_state(label):
    smiles = _REFUSED[label]
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, label
    assert name_oligonucleotide(mol) is None, label


def test_a_circle_has_no_5_prime_end():
    """3',5'-cyclic ApU: both links are 3'->5' but the chain has no first unit."""
    mol = Chem.RWMol(Chem.MolFromSmiles(_APU))
    o5 = [m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1][CH2][CH]1[OX2][CH]([#7])[CH2,CH1][CH]1"))]
    o3 = [m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1][CH]1[CH2,CH1][CH]([#7])[OX2][CH]1[CH2]"))]
    assert len(o5) == 1 and len(o3) == 1                      # the free 5'-OH of A and the free 3'-OH of U
    p = mol.AddAtom(Chem.Atom(15))
    dbl, oh = mol.AddAtom(Chem.Atom(8)), mol.AddAtom(Chem.Atom(8))
    mol.AddBond(p, dbl, Chem.BondType.DOUBLE)
    mol.AddBond(p, oh, Chem.BondType.SINGLE)
    mol.AddBond(p, o5[0], Chem.BondType.SINGLE)
    mol.AddBond(p, o3[0], Chem.BondType.SINGLE)
    circle = mol.GetMol()
    Chem.SanitizeMol(circle)
    assert name_oligonucleotide(circle) is None


def test_the_nested_naming_of_the_offer_builds_no_oligonucleotide_name():
    """``systematic_nucleoside_name`` names the molecule once more with the nucleoside names left out
    (``nucleoside_names_off``); the producer builds nothing then."""
    token = nuc._NUCLEOSIDE_NAMES_OFF.set(True)
    try:
        assert name_oligonucleotide(Chem.MolFromSmiles(_APU)) is None
    finally:
        nuc._NUCLEOSIDE_NAMES_OFF.reset(token)
    assert name_oligonucleotide(Chem.MolFromSmiles(_APU)) == "adenylyl-(3'→5')-uridine"


# ---------------------------------------------------------------------------------------------------------------
# the route and the label
# ---------------------------------------------------------------------------------------------------------------
def _handle(smiles):
    from orthonym.routing.dispatch_table import _handle_nucleoside, _is_nucleoside
    mol = Chem.MolFromSmiles(smiles)
    return _is_nucleoside(mol, smiles, Chem.MolToSmiles(mol)), _handle_nucleoside(mol, smiles, Chem.MolToSmiles(mol))


def test_the_nucleoside_route_takes_an_oligonucleotide(monkeypatch):
    monkeypatch.setattr(nuc, "systematic_nucleoside_name", lambda mol: None)
    assert _handle(_APU) == (True, "adenylyl-(3'→5')-uridine")
    assert _handle(_REFUSED["2'-5' link"]) == (False, None)


def test_the_systematic_pin_is_offered_first(monkeypatch):
    """ INTRODUCTION (:50939): when the systematic name of the whole molecule reads as a PIN it stands before the
     name, which is the fallback (ruling 2026-11-01, internal notes item 1)."""
    monkeypatch.setattr(nuc, "systematic_nucleoside_name", lambda mol: "the systematic PIN")
    assert _handle(_APU) == (True, "the systematic PIN")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, expected", [_NAMED[0], _NAMED[1], _NAMED[4], _NAMED[5]])
def test_the_name_ships_at_best_effort_below_the_pin(smiles, expected):
    row = name_breadth(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (expected, "systematic_verified", False), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, expected", [_NAMED[0], _NAMED[1], _NAMED[4], _NAMED[5]])
def test_the_pin_tier_does_not_ship_it(smiles, expected):
    """No PIN is identified for an oligonucleotide INTRODUCTION,:50939): the PIN tier declines."""
    row = name_default(smiles)
    assert row.get("name") != expected and row.get("tier") != "pin_verified" and not row.get("is_pin"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("label", ["2'-5' link", "a 5'-phosphate terminus (out of scope)",
                                   "a 3'-phosphate terminus (out of scope)", "an undefined C1'",
                                   "a P-thio with a defined P", "a charged phosphate"])
def test_a_refused_chain_keeps_its_name(label):
    """A refusal is never a new decline: the molecule goes on to the names it had."""
    row = name_breadth(_REFUSED[label])
    assert row.get("name") and row.get("tier") != "abstain", row
    assert "(3'→5')" not in row["name"], row


# 5'-2H1 ApA (a census row): the pipeline hands the producer the isotope-stripped skeleton, and the name has no
# isotopic descriptor to give. The producer declines it (the outermost input carries a label), so the isotope
# decorator keeps the systematic skeleton it decorates; before this, the molecule had no name at all.
_APA_5D = ("[H][C@@]([2H])([C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)O)OP(=O)(O)O[C@@H]4[C@H](O[C@H]"
           "([C@@H]4O)N5C=NC6=C(N=CN=C65)N)CO")


@pytest.mark.opsin_gate
def test_an_isotope_labelled_chain_keeps_its_isotopic_name():
    row = name_breadth(_APA_5D)
    assert row.get("name") and row["tier"] != "abstain", row
    assert "(3'→5')" not in row["name"] and "2H" in row["name"], row
    assert name_is_rt_exact(row["name"], _APA_5D), row


def test_the_producer_names_only_the_whole_input():
    """Inside a ``name`` of another molecule (a capped fragment, a part of an adduct) the producer declines."""
    from orthonym.assembly import fragment_naming as fn
    guard = fn._fragment_guard
    saved = getattr(guard, "name_inputs", None)
    guard.name_inputs = ["Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O"]          # the outer input is another molecule
    try:
        assert name_oligonucleotide(Chem.MolFromSmiles(_APU)) is None
        guard.name_inputs = [_APU]
        assert name_oligonucleotide(Chem.MolFromSmiles(_APU)) == "adenylyl-(3'→5')-uridine"
        guard.name_inputs = [_APU.replace("[C@H](CO)", "[C@H]([13CH2]O)", 1)]    # the outer input is labelled
        assert name_oligonucleotide(Chem.MolFromSmiles(_APU)) is None
    finally:
        if saved is None:
            del guard.name_inputs
        else:
            guard.name_inputs = saved
