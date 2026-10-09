"""Roadmap item 12a (task t3): a detached fragment gets its leaf once its free valence is capped.

``name_substituent`` (best-effort) and the radical rung of ``t4_coverage`` hand a branch to
``name_universal_substituent_prefix`` as a separate molecule: its attachment atom carries the
free valence as a radical electron, so no leaf that reads the one outside neighbour fired
(``_nitro_shortcut``: ``len(outside) != 1``) and nitro came back as the 'a' chain
'1-oxido-2-oxa-1-azaeth-1-en-1-ium-1-yl', the Blue Book, section
General rules, "The chain must be terminated by a C atom or one of the following heteroatoms...";
the prefix is 'nitro',,:25935). ``cap_attachment`` closes the free valence by a
one-atom cap, which puts the fragment in the state it has inside the molecule; the mechanical
spelling (``mechanical_forms``) is untouched.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import mechanical_forms
from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
from orthonym.cli import _emit_tier_flags

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402

OLD_NITRO = "1-oxido-2-oxa-1-azaeth-1-en-1-ium-1-yl"


def test_a_detached_fragment_gets_its_leaf_once_capped():
    mol = Chem.MolFromSmiles("O=[N+][O-]")
    assert name_universal_substituent_prefix(mol, [0, 1, 2], 1, 1) == OLD_NITRO
    assert name_universal_substituent_prefix(mol, [0, 1, 2], 1, 1, cap_attachment=True) == "nitro"
    with mechanical_forms():
        assert name_universal_substituent_prefix(
            mol, [0, 1, 2], 1, 1, cap_attachment=True) == OLD_NITRO


def test_cap_attachment_leaves_other_shapes_alone():
    # an attachment atom in a ring, a fragment that is not detached
    mol = Chem.MolFromSmiles("c1ccccc1N(=O)=O")
    assert name_universal_substituent_prefix(mol, [6, 7, 8], 6, 1, cap_attachment=True) == \
        name_universal_substituent_prefix(mol, [6, 7, 8], 6, 1)


@pytest.mark.opsin_gate
def test_end_to_end_nitro_in_a_row_the_floor_names():
    smiles = "CC(=O)NC(COC(C)=O)C(O)c1ccc([N+](=O)[O-])cc1"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    # fix-all 2026-10-09: the ester is named by functional class, the alcohol component a
    # separate word 'Esters', the Blue Book); OPSIN 2.9.0 full InChIKey exact
    assert row["name"] == "2-acetamido-3-hydroxy-3-(4-nitrophenyl)propyl acetate"
    assert not F.ach_hits(row["name"])
    assert name_is_rt_exact(row["name"], smiles)
