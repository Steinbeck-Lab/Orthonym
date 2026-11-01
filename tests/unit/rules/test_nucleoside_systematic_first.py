"""Nucleosides and nucleotides: the systematic name is offered first, the nucleoside name is the fallback (K2).

 INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in
this Chapter", and the same paragraph: "When the full structure is known, a 'systematic name' may be generated in
accordance with Rules described in Chapters through ". So 'adenosine' "RETAINED NAMES OF
NUCLEOSIDES",:54943) and '5-methyluridine',:54963) are no PINs; the systematic substitutive name of the
molecule is offered first, as for the amino acids (E-AA) and the natural products (NPL), and labelled a PIN only when
OPSIN reads it back with its stereodescriptors. Where no such name is built (guanosine, the nucleotides today), the
nucleoside name ships below the PIN (tests/unit/data/test_retained_names_kept_below_pin.py).

Every expected name below is read back by an independent OPSIN call (``assert_pin_at_both_tiers``).
"""
import pytest
from rdkit import Chem

from orthonym.rules.nucleosides import name_nucleoside
from tests.support.nucleoside_producer import SYSTEMATIC_FIRST
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


_ROWS = [
    ('Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O',
     'adenosine', '(2R,3R,4S,5R)-2-(6-amino-9H-purin-9-yl)-5-(hydroxymethyl)oxolane-3,4-diol'),
    ('Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1',
     'cytidine', '4-amino-1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidin-2(1H)-one'),
    ('Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O',
     'thymidine', '1-[(2R,4S,5R)-4-hydroxy-5-(hydroxymethyl)oxolan-2-yl]-5-methylpyrimidine-2,4(1H,3H)-dione'),
    ('O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1',
     'uridine', '1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidine-2,4(1H,3H)-dione'),
    ('Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1',
     "2'-deoxyadenosine", '(2R,3S,5R)-5-(6-amino-9H-purin-9-yl)-2-(hydroxymethyl)oxolan-3-ol'),
    ('Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1',
     "2'-deoxycytidine", '4-amino-1-[(2R,4S,5R)-4-hydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidin-2(1H)-one'),
    ('O=c1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1',
     "2'-deoxyuridine", '1-[(2R,4S,5R)-4-hydroxy-5-(hydroxymethyl)oxolan-2-yl]pyrimidine-2,4(1H,3H)-dione'),
    ('Cn1c(N)nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c1=O',
     '1-methylguanosine', '2-amino-9-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]-1-methyl-1,9-dihydro-6H-purin-6-one'),
    ('Cc1cn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]c1=O',
     '5-methyluridine', '1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]-5-methylpyrimidine-2,4(1H,3H)-dione'),
    ('Cc1cn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)nc1N',
     '5-methylcytidine', '4-amino-1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]-5-methylpyrimidin-2(1H)-one'),
    ('CC(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O',
     "adenosine 5'-acetate", '[(2R,3S,4R,5R)-5-(6-amino-9H-purin-9-yl)-3,4-dihydroxyoxolan-2-yl]methyl acetate'),
]


@pytest.mark.parametrize("smiles, nucleoside_name, pin", _ROWS)
def test_the_systematic_pin_comes_first(smiles, nucleoside_name, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles, nucleoside_name, pin", [r for r in _ROWS if r[1] not in SYSTEMATIC_FIRST])
def test_the_producer_still_builds_the_nucleoside_name(smiles, nucleoside_name, pin):
    """The fallback is unchanged: the producer builds the name; only the route offers the PIN first."""
    assert name_nucleoside(Chem.MolFromSmiles(smiles)) == nucleoside_name


def test_the_offer_is_strict_about_stereo(monkeypatch):
    """The amino-acid rule, not the natural-product relaxation: the offered name must read back with its
    stereodescriptors (ruling 2026-11-01, internal notes item 2)."""
    import orthonym.rules.systematic_offer as offer
    from orthonym.rules.nucleosides import systematic_nucleoside_name
    seen = {}
    monkeypatch.setattr(offer, "systematic_pin_offer", lambda mol, off, key, **kw: seen.update(kw))
    systematic_nucleoside_name(Chem.MolFromSmiles(_ROWS[0][0]))
    assert not seen.get("stereo_unread_ok", False)


def test_the_name_set_is_the_table():
    """RETAINED_NUCLEOSIDE_NAMES holds every nucleoside or nucleotide name of the retained-name table."""
    import re
    from orthonym.data import ALL_RETAINED_NAMES
    from orthonym.rules.nucleosides import RETAINED_NUCLEOSIDE_NAMES
    word = re.compile(r"(?:adenos|guanos|inos|xanthos|cytid|urid|thymid)ine$|(?:adenyl|guanyl|inosin|xanthyl|cytidyl|"
                      r"uridyl|thymidyl)ic acid$")
    assert {n for n in ALL_RETAINED_NAMES.values() if word.search(n)} == set(RETAINED_NUCLEOSIDE_NAMES)


def test_a_salt_component_keeps_its_nucleoside_name():
    """The offer is for the caller's whole molecule only (``outermost_name_input``): a component of an adduct keeps
    the name its producer builds, as glycine does in 'glycine—hydrogen chloride (1/1)'."""
    row = name_breadth("Cl." + _ROWS[0][0])
    assert row.get("name") == "adenosine\u2014hydrogen chloride (1/1)", row
