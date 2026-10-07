"""Lane L2 proper fix (R1): the writers spell the retained preferred prefixes from the
structure, and the benzyl radical keeps its PIN label.

* (the Blue Book): '-C(CH3)3 tert-butyl (preferred prefix)
  1,1-dimethylethyl' (:24412); 'C6H5-CH2- benzyl (preferred prefix) phenylmethyl'
  (:24414); 'benzylidene' (:24416); 'benzylidyne' (:24418). (:16272): "retained
  preferred prefixes, but are not to be substituted".
* (:24591) 'benzyloxy (preferred prefix)'; (:27679) 'tert-butoxy
  (preferred prefix) (no substitution)'.
* (1) (:40376): "the preferred IUPAC name for a radical may not be the same as the
  preferred prefix"; '2-methylpropan-2-yl (PIN)... tert-butyl' (:40453). The radical
  C6H5-CH2. is 'phenylmethyl' (gold row W2F-P5-09).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
from orthonym.cli import _emit_tier_flags
from orthonym.rules.pin_vocabulary import non_pin_vocabulary
from orthonym.rules.terminal_fragment import terminal_fragment_name
from tests.support.rt_assert import name_is_rt_exact


def _engine(tier):
    return Orthonym() if tier == "pin" else Orthonym(style="pin", **_emit_tier_flags(tier))


def _kekulized(smiles):
    m = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(m, clearAromaticFlags=True)
    return m


@pytest.mark.parametrize("mol,group,attach,order,expected", [
    (Chem.MolFromSmiles("OCc1ccccc1"), set(range(1, 8)), 1, 1, "benzyl"),
    (_kekulized("OCc1ccccc1"), set(range(1, 8)), 1, 1, "benzyl"),
    (Chem.MolFromSmiles("C=Cc1ccccc1"), set(range(1, 8)), 1, 2, "benzylidene"),
    (Chem.MolFromSmiles("C#Cc1ccccc1"), set(range(1, 8)), 1, 3, "benzylidyne"),
    (Chem.MolFromSmiles("OC(C)(C)C"), {1, 2, 3, 4}, 1, 1, "tert-butyl"),
    # not the retained group:16272; a labelled atom needs its descriptor)
    (Chem.MolFromSmiles("OCc1ccc(Cl)cc1"), set(range(1, 9)), 1, 1, None),
    (Chem.MolFromSmiles("O[13CH2]c1ccccc1"), set(range(1, 8)), 1, 1, None),
    (Chem.MolFromSmiles("OC([2H])c1ccccc1"), {1, 3, 4, 5, 6, 7, 8}, 1, 1, None),
    (Chem.MolFromSmiles("OCc1ccncc1"), set(range(1, 8)), 1, 1, None),
    (Chem.MolFromSmiles("OCC1CCCCC1"), set(range(1, 8)), 1, 1, None),
    (Chem.MolFromSmiles("OC(C)(C)CC"), {1, 2, 3, 4, 5}, 1, 1, None),
    (Chem.MolFromSmiles("OC(c1ccccc1)c1ccccc1"), set(range(1, 14)), 1, 1, None),
    (Chem.MolFromSmiles("OC(C)(C)C"), {1, 2, 3, 4}, 1, 2, None),
])
def test_retained_group_prefix(mol, group, attach, order, expected):
    from orthonym.assembly.book_prefixes import retained_group_prefix
    assert retained_group_prefix(mol, group, attach, order) == expected


@pytest.mark.parametrize("smiles,frag,attach,expected", [
    ("OCc1ccccc1", set(range(1, 8)), 1, "benzyl"),
    ("OC(C)(C)C", {1, 2, 3, 4}, 1, "tert-butyl"),
    ("OCc1ccc(Cl)cc1", set(range(1, 9)), 1, "(4-chlorophenyl)methyl"),   # substituted
    ("OC(c1ccccc1)c1ccccc1", set(range(1, 14)), 1, "diphenylmethyl"),
])
def test_both_writers_spell_the_retained_prefix_from_the_atoms(smiles, frag, attach, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert terminal_fragment_name(mol, frag, attach).name == expected
    assert name_universal_substituent_prefix(mol, sorted(frag), attach) == expected


def test_the_general_spelling_is_no_longer_a_label_pattern():
    # the label decision left with the writer; the radical's whole-name PIN is not flagged.
    # ('1,1-dimethylethyl' is still read by main's older longest-chain check,
    # '_METHYL_ON_ETHYL', which this lane did not add and does not change.)
    for name in ("phenylmethyl", "phenylmethoxy", "1,1-dimethylethoxy"):
        assert non_pin_vocabulary(name) is None, name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_benzyl_radical_is_the_pin_phenylmethyl(tier):
    row = _engine(tier).name_tiered("[CH2]c1ccccc1")
    assert (row.get("name"), row.get("tier")) == ("phenylmethyl", "pin_verified"), row
    assert name_is_rt_exact("phenylmethyl", "[CH2]c1ccccc1")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[C](C)C", "2-methylpropan-2-yl"),          #:40453
    ("CC(C)(C)[O]", "tert-butoxyl"),              #:40681
    ("CC(C)(C)S[S]", "tert-butyldisulfanyl"),     #:40737
    ("c1ccc(C[SiH2])cc1", "benzylsilyl"),
    ("c1ccc(Cc2ccncc2)cc1", "4-benzylpyridine"),  #:16276 pattern
    ("CC(C)(C)c1ccncc1", "4-tert-butylpyridine"),
])
def test_the_radical_and_prefix_controls_keep_their_pin_labels(smiles, expected):
    row = _engine("pin").name_tiered(smiles)
    assert (row.get("name"), row.get("tier")) == (expected, "pin_verified"), row


# rows whose name carried 'phenylmethyl' / '1,1-dimethylethyl' / '-ethoxy' forms on the
# lane (study-r1 r1-trace/jobs_o.tsv; study-audit trace/pv_out.jsonl)
GENERAL_FORM_ROWS = [
    ("B(C1CCCCC1)(C2CCCCC2)C(C)(C)C", "best-effort", "tert-butyl", "1,1-dimethylethyl"),
    ("CCCCCCC[C@H]1OC(=O)CC(=O)[C@H](Cc2ccccc2)N(C)C(=O)[C@H](C(C)C)OC(=O)[C@H]1C",
     "complete", "benzyl", "phenylmethyl"),
    ("C[C@H](COc1ccccc1Cc1ccccc1)N1CCCCC1", "complete", "benzylphenoxy", "phenylmethyl"),
    ("CC(C)(C)OC(=O)N1CCS[C@@H]1C(=O)O", "complete", "tert-butoxycarbonyl",
     "1,1-dimethylethoxy"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,book,general", GENERAL_FORM_ROWS)
def test_engine_rows_take_the_retained_prefix(smiles, tier, book, general):
    row = _engine(tier).name_tiered(smiles)
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    assert book in name and general not in name, name
