"""PubChem best-effort losses (TRIAGE.md 'Breadth -- PubChem losses').

The paper's PubChem 500k run (code f67429619) named six molecules round-trip
exact on which the code of 2b1455817 / aa521bf23 abstains at best-effort. The
six rows are the witnesses (none is in eval/splits/a holdout split.json), plus a
small analogue for the universal floor itself. Every shipped name is checked by
an independent OPSIN full-InChIKey round trip (tests/support/rt_assert).

Classes (causing commit at the time -> root cause at HEAD):
  A the universal floor cites a '-ylidene' double bond twice. A branch attached
     to its host by a double bond is one stereogenic unit with an end in each
     scope; the host's block cites it with the host's locant (1)
     method (a), the Blue Book, "Method (a) generates preferred IUPAC
     names":48279) and the ylidene prefix cited it again (method (b),:48277),
     e.g. '3-{(1Z)-1-[(1Z)-2-azaethan-1-ylidene]-...}', which OPSIN cannot read,
     so the floor fell back to its stereo-free spelling and the round trip
     refused it. Until 49a0e838f the chain host did not cite the bond at all, and
     the rows were named only through a composed '(1Z)-' block at the front of
     the whole name whose locant belongs to no parent double bond (OPSIN bound it
     by luck); 5620d69f2 / 49a0e838f / 95126f4a1 changed which candidate reached
     that composer. Rows 1, 3, 4, 5.
  B the last rung ships the floor's first spelling without the verified
     retry the earlier rungs get (_prefer_verified_floor): for row 2 the fused
     spiro floor spelling 'spiro[piperidine-4,4'-tryptoline]' reads back as a
     DIFFERENT molecule, while its von Baeyer form verifies. d42290dd4 changed the
     prefix order so the general-engine rung (which carried the retry) no longer
     certified.
Row 6 (a P(III) stereocentre): its floor name, prefixes in order, round-trips
(OPSIN's SMILES read by RDKit) to the other P configuration, and to the input only
when the P-substituent is cited first (an order the rule does not give).
OPSIN's own StdInChIKey is the same for both orders; the difference is RDKit's
reading of a lone-pair stereocentre that carries a ring-closure digit (D3,
tests/unit/rules/test_prefix_order_fallback.py). A name that does not round-trip
never ships; since D3 (user decision 2026-09-28) the best-effort last resort ships
the out-of-order spelling, labelled best_effort.
"""
import re

import pytest
from rdkit import Chem

from orthonym.rules.stereochemistry import (
    count_defined_stereo_elements,
    count_expressed_stereo_descriptors,
)
from tests.support.rt_assert import (
    _full_inchikey,
    _independent_parse,
    assert_full_rt,
    name_best_effort,
)

pytestmark = pytest.mark.opsin_gate


# --------------------------------------------------------------------------
# A a '-ylidene' double bond is cited once, by its host (1)(a))
# --------------------------------------------------------------------------

YLIDENE_ROWS = [
    # PubChem row 1 (paper: '(1Z)-9-[...]-3-[...]-1,7-diazabicyclo[4.3.0]...')
    "N=C(/C(=C\\N)c1ccc2ncc(C(=N)NC(=O)C(F)F)n2c1)c1ccccc1F",
    # PubChem row 3 (paper: '(1E,2E,6E)-2-[...]-5-[...]-1-azacyclohexa-1,3,5-triene')
    "C/C=C(\\CC(=C\\C)/C(C)=C/Nc1ccc(-c2csc(C(=O)N(C)C)c2)cn1)OC",
    # PubChem row 4 (paper: '(1Z)-1-amino-4-(...)cyclohexa-1,3,5-triene')
    "CN=C/C(=C\\N)c1cc(Oc2ccc(N)cc2)c(Cl)cn1",
    # PubChem row 5 (paper: '(1E)-4-[...]-1-chlorocyclohexa-1,3,5-triene')
    "Cc1cc(N2CCCC2)ccc1/C=C(/Sc1nnc(-c2ccc(Cl)cc2)o1)C(=O)O",
]

# Every E/Z bond of rows 1, 3 and 4 lies in a substituent: the parent ring has
# none, so a descriptor block at the front of the whole name has no parent double
# bond to refer to, the Blue Book: "When they relate to substituent
# groups, they are cited at the front of the corresponding prefix").
_LEADING_EZ_BLOCK = re.compile(r"^\((?:\d+[EZ],?)+\)-")

# Row 5's carboxylic acid is the suffix since the drug lane's Task L3.4 (before it,
# the best-effort name cited no suffix and a ring was the parent), so its E/Z bond
# is C-2=C-3 of the parent chain 'prop-2-enoic acid' and its block stands at the
# front of the complete name, the Blue Book: "They are placed at the
# front of the complete name when related to the parent structure"). The block
# and the parent ending the name must agree.
PARENT_EZ_BLOCK = {
    YLIDENE_ROWS[3]: ("(2E)-", "prop-2-enoic acid"),
}


@pytest.mark.parametrize("smiles", YLIDENE_ROWS, ids=["row1", "row3", "row4", "row5"])
def test_ylidene_rows_name_rt_exact(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert res.get("tier") != "pin_verified"
    mol = Chem.MolFromSmiles(smiles)
    # each stereogenic unit cited exactly once (no double citation)
    assert (count_expressed_stereo_descriptors(name)
            == count_defined_stereo_elements(mol)), name
    block = _LEADING_EZ_BLOCK.match(name)
    if smiles in PARENT_EZ_BLOCK:
        # the block at the front is the parent chain's own E/Z unit
        lead, parent = PARENT_EZ_BLOCK[smiles]
        assert block and block.group(0) == lead and name.endswith(parent), name
    else:
        # the misplaced whole-name block of the paper names does not come back
        assert not block, name


def test_floor_cites_the_ylidene_bond_once():
    """The universal floor itself, on a small analogue: 3-amino-2-phenyl-
    acrylonitrile. Its floor spelling hangs the E/Z unit between the
    3-azaprop-2-yn-1-yl host and a 2-azaethan-1-ylidene branch."""
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    smiles = "N/C=C(\\C#N)c1ccccc1"
    uni = name_universal_substitutive(Chem.MolFromSmiles(smiles))
    assert uni is not None
    assert count_expressed_stereo_descriptors(uni.name) == 1, uni.name
    assert "[(1Z)-2-azaethan-1-ylidene]" not in uni.name
    assert_full_rt(uni.name, smiles)


def test_double_cited_form_is_unreadable():
    """The spelling the floor used to build: OPSIN cannot read one bond with two
    descriptors, so the floor dropped to its stereo-free name."""
    name = ("1-{(1Z)-1-[(1Z)-2-azaethan-1-ylidene]-3-azaprop-2-yn-1-yl}"
            "cyclohexa-1,3,5-triene")
    assert _independent_parse(name) is None


def test_ring_ylidene_branch_keeps_its_own_block():
    """The branch keeps the bond when its ylidene end is a ring atom (the host
    then does not cite it, _is_true_exocyclic): method (b), unchanged."""
    from orthonym.assembly.universal_substituent import _branch_stereo_block
    # 2-(2-methylcyclohexylidene)ethanol analogue: host C of the chain, the
    # ylidene end in the ring
    mol = Chem.MolFromSmiles("OC/C=C1/CCCCC1C")
    from orthonym.perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)
    ring_c = 3  # the ring atom doubly bonded to chain atom 2
    ring_map = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
    with_bond = _branch_stereo_block(mol, ring_map, host_atom=2, root=ring_c)
    assert with_bond.startswith("(1"), with_bond


# --------------------------------------------------------------------------
# B the last rung offers a verified floor spelling
# --------------------------------------------------------------------------

SPIRO_ROW = "COc1cccc2c3c([nH]c12)C1(CCN(CCCC#N)CC1)N(C=O)CC3"  # PubChem row 2
TRYPTOLINE_FORM = ("1-(5-azapent-4-yn-1-yl)-6'-(1-oxaethan-1-yl)-3'-(2-oxaeth-1-en-"
                   "1-yl)spiro[piperidine-4,4'-tryptoline]")


def test_fused_spiro_row_names_rt_exact():
    res = name_best_effort(SPIRO_ROW)
    name = assert_full_rt(res.get("name"), SPIRO_ROW)
    assert "tryptoline" not in name
    assert res.get("tier") != "pin_verified"


def test_first_floor_spelling_is_a_different_molecule():
    """Why the retry is needed: the first floor spelling reads back as another
    molecule (so it can never ship)."""
    back = _independent_parse(TRYPTOLINE_FORM)
    assert back is not None
    assert _full_inchikey(back)[:14] != _full_inchikey(SPIRO_ROW)[:14]


# --------------------------------------------------------------------------
# Row 6: the round trip of the P(III) descriptor depends on citation order; nothing that fails it ships
# --------------------------------------------------------------------------

P_ROW = ("Cc1cn([C@H]2CC(O[P@]3O[C@](C)(c4ccccc4)[C@@H]4CCCN43)[C@@H](CO)O2)"
         "c(=O)[nH]c1=O")


def test_p_stereocentre_row_ships_nothing_opsin_misreads():
    res = name_best_effort(P_ROW)
    name = res.get("name")
    if res.get("source") != "abstain":
        assert_full_rt(name, P_ROW)


# --------------------------------------------------------------------------
# C a verified floor stand-in does not pre-empt the clean-context engine name
# --------------------------------------------------------------------------

# milestone1500 row (not a holdout split). Inside name the rung builds a candidate
# that fails its round trip and the universal floor's verified spelling stands in
# for it; the clean fall-through (depth 0, fresh memo) builds the engine's own
# verified name. With the class-A fix the floor now verifies for this molecule,
# and without the clean-first offer it shipped the floor spelling instead:
# '2-amino-5-{(1Z,4S,11S)-...-2,5-dioxo-13-oxa-3,6-diazatridec-12-en-1-yl}-
# 3-thia-1-azacyclopenta-1,4-diene' (a 1,3-thiazole as a replacement name, the
# carboxylic acid as '12-hydroxy-...-13-oxa-...-12-en').
THIAZOLE_ROW = ("C[C@H](NS(=O)(=O)O)[C@H](NC(=O)/C(=N\\OC(C)(C)C(=O)O)c1csc(N)n1)"
                "C(=O)NCCCC[C@H](N)C(=O)O")


def test_clean_engine_name_outranks_a_floor_stand_in():
    res = name_best_effort(THIAZOLE_ROW)
    name = assert_full_rt(res.get("name"), THIAZOLE_ROW)
    assert "1,3-thiazole" in name, name
    assert "azacyclopenta" not in name, name


def test_acyclic_floor_stand_in_keeps_its_principal_chain():
    """An acyclic molecule keeps the floor's principal chain (the Blue Book
    :20922: "(a) contains the greater number of heteroatoms of any kind; (b) has
    the greater number of skeletal atoms"): the clean recovery's ring-first rung,
    with no ring to choose, gives '1-(3-methyl-5-oxa-1,3,4-triazapent-4-en-1-yl)-
    1-oxoethane' (an ethane parent), which must not replace it."""
    smiles = "CC(=O)NCN(C)N=O"  # milestone1500 row, not a holdout split
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert "oxoethane" not in name, name
