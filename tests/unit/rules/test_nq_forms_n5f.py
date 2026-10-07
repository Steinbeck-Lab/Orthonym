"""Roadmap N5f: no locant that no isomer needs, at the writers.

* (a) (the Blue Book-2893, 'The locant '1' is omitted: (a) in substituted
  mononuclear parent hydrides'): a methyl group cites no locant on its prefixes
  ('chloromethyl',:25805); (:7272) encloses the second and further
  substituents of a mononuclear parent ('chloro(methyl)silane (PIN)').
* (:3007): "All locants are omitted in compounds or substituent groups in which
  all substitutable positions are completely substituted or modified... in the same way"
  ('1-chloro-2-(pentafluoroethyl)benzene (PIN)',:3023); partial substitution keeps all
  locants (:3009).
* (1) (:15813): 'methyl', 'ethyl' -- never 'methan-1-yl' ('methyl (preferred
  prefix) methanyl',:15864).

The writers: the floor (``universal_substituent._name_component`` /
``_render_as_substituent``), the terminal-fragment chain namer
(``terminal_fragment._terminal_fragment_name``) and the carbamoylamino chain writer
(``substituent_naming._name_carbamoylamino_chain_substituent``), which wrote the one
pin_verified name with such a locant: 'N-[1-(carbamoylamino)methyl]urea'.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import (
    mechanical_forms, methyl_group_name, positions_alike_prefix)
from orthonym.assembly.naming_utils import format_substituent_prefix
from orthonym.assembly.substituent_naming import _name_carbamoylamino_chain_substituent
from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
from orthonym.cli import _emit_tier_flags
from orthonym.rules.terminal_fragment import terminal_fragment_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


@pytest.mark.parametrize("names,bond_order,expected", [
    (["fluoro"] * 3, 1, "trifluoromethyl"),
    (["chloro", "fluoro", "fluoro"], 1, "chlorodi(fluoro)methyl"),
    (["4-chlorophenyl"], 1, "(4-chlorophenyl)methyl"),
    (["hydroxy", "phenyl"], 1, "hydroxy(phenyl)methyl"),
    (["carbamoylamino"], 1, "(carbamoylamino)methyl"),
    (["phenyl"], 2, "phenylmethylidene"),
    (["chloro"] * 3, 0, "trichloromethane"),
])
def test_methyl_group_name(names, bond_order, expected):
    assert methyl_group_name(names, bond_order) == expected


def test_positions_alike_prefix_only_for_complete_substitution():
    cf2cf3 = Chem.MolFromSmiles("FC(F)(F)C(F)(F)C1CC1")
    spine = [1, 4]
    assert positions_alike_prefix(cf2cf3, spine, ["fluoro"] * 5,
                                  unit_cites_locant=False) == "pentafluoro"
    ch2cf3 = Chem.MolFromSmiles("FC(F)(F)CC1CC1")
    assert positions_alike_prefix(ch2cf3, [1, 4], ["fluoro"] * 3,
                                  unit_cites_locant=False) is None   #:3009


def test_a_compound_prefix_with_only_inner_marks_takes_bis():
    # (a) (the Blue Book); the writer records the substitution
    assert format_substituent_prefix(methyl_group_name(["hydroxy", "phenyl"]), [2, 4], 2) == \
        "2,4-bis[hydroxy(phenyl)methyl]"
    assert format_substituent_prefix("propan-2-yl", [2, 4], 2) == "2,4-di(propan-2-yl)"


def _frag(smiles, frag_smarts_atoms, attach):
    mol = Chem.MolFromSmiles(smiles)
    return mol, set(frag_smarts_atoms), attach


@pytest.mark.parametrize("smiles,frag,attach,expected,old", [
    # CF3 on a ring: the chain namer
    ("FC(F)(F)c1ccccc1", [0, 1, 2, 3], 1, "trifluoromethyl", "1,1,1-trifluoromethyl"),
    # CF2CF3: every position alike
    ("FC(F)(F)C(F)(F)c1ccccc1", [0, 1, 2, 3, 4, 5, 6], 4, "pentafluoroethyl",
     "1,1,2,2,2-pentafluoroethyl"),
])
def test_terminal_fragment_chain_drops_the_locants(smiles, frag, attach, expected, old):
    mol = Chem.MolFromSmiles(smiles)
    assert terminal_fragment_name(mol, frag, attach).name == expected
    with mechanical_forms():
        assert terminal_fragment_name(mol, frag, attach).name == old


def test_terminal_fragment_keeps_locants_under_partial_substitution():
    mol = Chem.MolFromSmiles("FC(F)(F)Cc1ccccc1")
    assert terminal_fragment_name(mol, {0, 1, 2, 3, 4}, 4).name == "2,2,2-trifluoroethyl"


@pytest.mark.parametrize("smiles,frag,attach,expected", [
    ("FC(F)(F)c1ccccc1", [0, 1, 2, 3], 1, "trifluoromethyl"),
    ("BrCc1ccccc1", [0, 1], 1, "bromomethyl"),
    ("ClCCc1ccccc1", [0, 1, 2], 2, "2-chloroethyl"),
])
def test_floor_branch_spells_methyl_and_alkyl_groups(smiles, frag, attach, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert name_universal_substituent_prefix(mol, frag, attach) == expected


def test_carbamoylamino_on_a_one_carbon_chain_has_no_locant():
    mol = Chem.MolFromSmiles("NC(=O)NCNC(N)=O")
    # fragment -CH2-NH-C(=O)-NH2 attached at C4 (the CH2)
    frag = [4, 5, 6, 7, 8]
    assert _name_carbamoylamino_chain_substituent(mol, frag, 4) == "(carbamoylamino)methyl"
    with mechanical_forms():
        assert _name_carbamoylamino_chain_substituent(mol, frag, 4) == \
            "1-(carbamoylamino)methyl"


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym().name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: names that carried an N5f form at the base (shared baseline, valid / pin tier). The
#: label is the base's for each row; for the first it is not a PIN conformance claim:
#: (the Blue Book) makes the multiplicative "N,N''-methylenediurea" the
#: PIN of two ureas joined by methylene (the analogue "N',N'''-methylenediacetohydrazide
#: (PIN)",:6147), a parent choice the strict path does not make (roadmap N8); this lane
#: re-spells the substitutive name and leaves its label as it was.
E2E = [
    ("NC(=O)NCNC(N)=O", "pin", "N-[(carbamoylamino)methyl]urea"),
    ("NC(=O)NCc1ccc(cc1)C(=O)O", "pin", "4-[(carbamoylamino)methyl]benzoic acid"),
    ("CC(=O)NCNC(N)=O", "pin", "N-[(carbamoylamino)methyl]acetamide"),
    ("CC(OC(=O)CBr)[Si](C)(C)C", "valid", None),
    ("FC(F)(F)c1ccccc1NC(=S)N1CCCC1", "valid", None),
    ("CSCCC(=C)C1C(=O)CCCC1=O", "valid", None),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,expected", E2E)
def test_no_redundant_locant_end_to_end(smiles, tier, expected):
    row = _row(smiles, tier)
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    forms = F.detect(name)
    assert "RLOC" not in forms and "ANYL_ME" not in forms, (name, forms)
    if expected is not None:
        assert (name, row["tier"]) == (expected, "pin_verified")


# (the Blue Book): "if any locants are essential for defining the structure
# of the parent structure or of a unit of structure... then all locants must be cited";
# '(1,1,1,3,3,3-hexafluoropropan-2-yl)oxy' (:46359). The omission of is for a
# unit whose own name cites no locant ('pentafluoroethyl',:3023).
def test_positions_alike_prefix_keeps_the_locants_of_a_unit_that_cites_one():
    mol = Chem.MolFromSmiles("FC(F)(F)C(F)(F)F")
    assert positions_alike_prefix(mol, [1, 4], ["fluoro"] * 5,
                                  unit_cites_locant=False) == "pentafluoro"
    assert positions_alike_prefix(mol, [1, 4], ["fluoro"] * 5,
                                  unit_cites_locant=True) is None
    assert name_universal_substituent_prefix(
        Chem.MolFromSmiles("FC(F)(F)C(F)(F)c1ccccc1"), [0, 1, 2, 3, 4, 5, 6], 4) \
        == "pentafluoroethyl"

# The book-forms switch is a context variable a naming path branches on, so every memo
# key carries it (``assembly.memo``): the terminal-fragment writer names a fragment
# mechanically first inside the same memo scope as the book spellings, and neither run
# may be served the other's value.
def test_the_memo_keys_carry_the_book_forms_switch():
    from orthonym.assembly import memo
    from orthonym.assembly.substituent_naming import _nsf_memo_key
    mol = Chem.MolFromSmiles("NC(=O)NCNC(N)=O")
    book_key = memo._ck("name_substituent", "k")
    book_nsf = _nsf_memo_key(mol, [4, 5, 6, 7, 8], 4, None)
    with mechanical_forms():
        assert memo._ck("name_substituent", "k") != book_key
        assert _nsf_memo_key(mol, [4, 5, 6, 7, 8], 4, None) != book_nsf
        # the pure structure namespaces stay shared
        assert memo._ck("fg_detect", "k") == ("fg_detect", "k")
    assert book_key == ("name_substituent", "k")
