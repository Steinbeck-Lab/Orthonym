"""Roadmap N5e: acyl groups by their acyl names, at the writers.

* (2) (the Blue Book): "carbonyl groups in position 1 of a side chain, i.e.,
  -CO-R, are described by the appropriate acyl group name".
* (:32922-:32928): -CO-NH2 on a ring, a heterogeneous chain or a
  nonterminal carbon is 'carbamoyl', "substituted in the normal way"
  ('3-(dimethylcarbamoyl)pentanedioic acid (PIN)',:32934;
  '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid (PIN)',:33354) -- never
  '{[R]amino}-oxomethyl'.
* (:30624) with 'pyrrolidine-1-carboxylic acid (PIN)' (:29892): a ring-N
  carbonyl is '<ring>-1-carbonyl'; (2) (:30851) the concatenated
  '(azetidin-1-yl)carbonyl' when that acid name is not built.
* (:31698): an ester cited as a prefix, 'alkoxycarbonyl' ('3-(ethoxycarbonyl)
  phenoxy',:31813).
* (:30446): 'oxo(phenyl)methyl', the enclosed general form;
  (:7272) for every one-carbon group.

``assembly.book_prefixes.acyl_group_prefix`` is the producer; the writers --
``substituent_naming._located_fg_assemble`` (the '-oxomethyl' writer), the
terminal-fragment namer and the floor's acyl leaf -- call it with their own
recursion for the parts.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import acyl_group_prefix, mechanical_forms
from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
from orthonym.cli import _emit_tier_flags
from orthonym.rules.terminal_fragment import terminal_fragment_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402

#: acyl groups on a benzene ring, the ring written first: parent atom = 0, the carbonyl
#: carbon = 6 (the first atom after the ring)
GROUPS = [
    ("c1ccccc1C=O", "formyl"),
    ("c1ccccc1C(=O)O", "carboxy"),
    ("c1ccccc1C(N)=O", "carbamoyl"),
    ("c1ccccc1C(=O)NC", "methylcarbamoyl"),
    ("c1ccccc1C(=O)N(C)c1ccccc1", "methyl(phenyl)carbamoyl"),
    ("c1ccccc1C(=O)N1CCC1", "azetidine-1-carbonyl"),
    ("c1ccccc1C(=O)N1CCOCC1", "morpholine-4-carbonyl"),
    ("c1ccccc1C(=O)OC", "methoxycarbonyl"),
    ("c1ccccc1C(=O)OCc1ccccc1", "(benzyloxy)carbonyl"),   #:24591
    ("c1ccccc1C(=O)OC(C)(C)C", "tert-butoxycarbonyl"),    #:27679
    ("c1ccccc1C(C)=O", "acetyl"),
    ("c1ccccc1C(=O)c1ccccc1Cl", "2-chlorobenzoyl"),
]


def _tf_part(mol):
    def part(atoms, root):
        sub = terminal_fragment_name(mol, set(atoms), root)
        return sub.name if sub is not None else None
    return part


@pytest.mark.parametrize("smiles,expected", GROUPS)
def test_acyl_group_prefix(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    ring = {0, 1, 2, 3, 4, 5}
    acyl = {a.GetIdx() for a in mol.GetAtoms()} - ring
    parent = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(6).GetNeighbors()
                  if nb.GetIdx() in ring)
    assert acyl_group_prefix(mol, 6, parent, acyl, _tf_part(mol)) == expected


def test_a_ring_nitrogen_carbonyl_without_an_acid_name_is_concatenated():
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)N1CCC1")
    acyl = {6, 7, 8, 9, 10, 11}

    def part(atoms, root):
        return "azetidin-1-yl"
    # an acid name that cannot be built: the concatenated general form
    import orthonym.assembly.book_prefixes as bp
    from orthonym.metrics import provenance as prov
    orig = bp._ring_nitrogen_carbonyl
    bp._ring_nitrogen_carbonyl = lambda *a, **k: None
    try:
        prov.clear_provenance()
        assert acyl_group_prefix(mol, 6, 5, acyl, part) == "(azetidin-1-yl)carbonyl"
        # (the Blue Book,:30853): method (2), never part of a PIN
        assert "(azetidin-1-yl)carbonyl" in prov.get_provenance()["non_pin_fragments"]
    finally:
        bp._ring_nitrogen_carbonyl = orig


@pytest.mark.parametrize("smiles,frag,attach,expected,old", [
    ("c1ccccc1C(=O)NC", [6, 7, 8, 9], 6, "methylcarbamoyl", "1-oxo-2-azapropyl"),
    ("c1ccccc1C(=O)OC", [6, 7, 8, 9], 6, "methoxycarbonyl", "1-oxo-2-oxapropyl"),
])
def test_terminal_fragment_acyl(smiles, frag, attach, expected, old):
    mol = Chem.MolFromSmiles(smiles)
    assert terminal_fragment_name(mol, set(frag), attach).name == expected
    with mechanical_forms():
        assert terminal_fragment_name(mol, set(frag), attach).name == old


def test_floor_acyl_leaf():
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)N1CCOCC1")
    assert name_universal_substituent_prefix(
        mol, [6, 7, 8, 9, 10, 11, 12, 13], 6) == "morpholine-4-carbonyl"


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym().name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


NILOTINIB = ("CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)"
             "C=1C=NC=CC1")

#: rows whose name carried '-oxomethyl' (or an amide 'a' chain) at the base
E2E = [
    (NILOTINIB, "best-effort",
     "{[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]carbamoyl}"),
    (NILOTINIB, "valid",
     "{[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]carbamoyl}"),
    ("CC(=O)NC(=O)NC1CCS(=O)(=O)C1", "best-effort", "[(acetylcarbamoyl)amino]"),
    ("O=C(Nc1ccccc1C(=O)O)OCC1c2ccccc2-c2ccccc21", "best-effort",
     "{[(9H-fluoren-9-yl)methoxy]carbonyl}amino"),
    ("CC(C)(C)OC(=O)N[C@@H](Cc1ccc2ccccc2n1)C(=O)OC", "best-effort", "(tert-butoxycarbonyl)amino"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,text", E2E)
def test_acyl_groups_end_to_end(smiles, tier, text):
    row = _row(smiles, tier)
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    forms = F.detect(name)
    assert "OXOM_M" not in forms and "ACH" not in forms, (name, forms)
    assert text in name, name


# --- a substituted acyl prefix is a compound prefix (Review Focus 7) ------------------------
# (the Blue Book): "Parentheses are used around compound... and complex
#... prefixes"; '(chloroacetyl)oxyl (PIN)' (:40694), '4-(acetylamino)' (:33011). The
# unsubstituted acyl prefixes ('acetyl (preferred prefix)',:30442) stay bare.

# The marks come from the writer that built the acyl prefix: it records from the acyl
# group's atoms whether the prefix is substituted and enclosed (book_prefixes.acyl_derivation);
# no vocabulary test over the name decides them.

@pytest.mark.parametrize("smiles,cited", [
    ("c1ccccc1C(=O)CCl", "(chloroacetyl)"),
    ("c1ccccc1C(=O)CN", "(aminoacetyl)"),
    ("c1ccccc1C(=O)Cc1ccccc1", "(phenylacetyl)"),
    ("c1ccccc1C(=O)CO", "(hydroxyacetyl)"),
    ("c1ccccc1C(C)=O", "acetyl"),
    ("c1ccccc1C(=O)c1ccccc1", "benzoyl"),
    ("c1ccccc1C(=O)CC", "propanoyl"),
    ("c1ccccc1C=O", "formyl"),
])
def test_a_substituted_acyl_prefix_takes_enclosing_marks(smiles, cited):
    from orthonym.assembly.naming_utils import enclose_if_compound
    mol = Chem.MolFromSmiles(smiles)
    acyl = {a.GetIdx() for a in mol.GetAtoms()} - {0, 1, 2, 3, 4, 5}
    assert enclose_if_compound(acyl_group_prefix(mol, 6, 5, acyl, _tf_part(mol))) == cited


def test_an_acylamino_prefix_takes_enclosing_marks():
    # '-NH-CO-CH3' from the hetero-rooted writer: a compound prefix,:7232)
    from orthonym.assembly.naming_utils import enclose_if_compound
    from orthonym.assembly.substituent_naming import _located_fg_hetero_root
    mol = Chem.MolFromSmiles("c1ccccc1NC(C)=O")
    assert enclose_if_compound(_located_fg_hetero_root(mol, [6, 7, 8, 9], 6)) == \
        "(acetylamino)"


def test_the_contracted_amido_prefix_stays_bare():
    from orthonym.assembly.naming_utils import enclose_if_compound
    assert enclose_if_compound("acetamido") == "acetamido"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,part", [
    ("ClCC(=O)c1ccc(Cl)cc1", "4-(chloroacetyl)"),
    ("NCC(=O)c1ccc(C(=O)O)cc1", "4-(aminoacetyl)"),
])
def test_the_floor_encloses_a_substituted_acyl_prefix(smiles, part):
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    res = name_universal_substitutive(Chem.MolFromSmiles(smiles))
    assert res is not None and part in res.name, res
    assert name_is_rt_exact(res.name, smiles), res.name
