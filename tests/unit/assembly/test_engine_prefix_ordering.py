"""The ordering name of a prefix is asked the way the general engine names that prefix.

 NUMBERING (the Blue Book), criterion (g): "lowest locants for the substituent cited
first as a prefix in the name" (:3307). The ring numbering decides a tie between orientations by
the alphanumerical key of each prefix on the ring (``substituent_enumerator.
name_substituent_for_ordering``), and it applies (g) only when every prefix has a key.

The general engine cites a prefix with ``name_substituent(..., allow_mancude=True)``. In the clean
context of the best-effort fall-through (the four propagation flags reset) the cascade does not name
an N-substituted thiocarbamoyl or carbamoyl group without that argument, so the key was missing, (g)
was skipped, and the canonical ranks decided which prefix of ``1-(benzenesulfonyl)-4-[...]piperazine``
took locant 1. ``engine_prefix_ordering`` marks the engine run (for the naming-session depth of the
engine call only), and the key is asked with the engine's argument.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming
from orthonym.assembly.engine_prefix_ordering import engine_prefix_ordering
from orthonym.assembly.substituent_enumerator import name_substituent_for_ordering
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate  # the names below are built with the validity gate on

PIPERAZINE_THIOUREA = "C1C[C@@H](OC1)CNC(=S)N2CCN(CC2)S(=O)(=O)C3=CC=CC=C3"
PREFIX = "({[(2R)-oxolan-2-yl]methyl}amino)(sulfanylidene)methyl"


def _thiocarbamoyl(smiles):
    mol = Chem.MolFromSmiles(smiles)
    carbon = next(a.GetIdx() for a in mol.GetAtoms()
                  if a.GetSymbol() == "C" and not a.IsInRing()
                  and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "S"
                          for b in a.GetBonds()))
    ring_n = next(n.GetIdx() for n in mol.GetAtomWithIdx(carbon).GetNeighbors() if n.IsInRing())
    edit = Chem.RWMol(mol)
    edit.RemoveBond(carbon, ring_n)
    frag = next(f for f in Chem.GetMolFrags(edit) if carbon in f)
    return mol, sorted(frag), carbon


def test_the_clean_context_has_no_key_for_the_thiocarbamoyl_prefix_outside_an_engine_run():
    mol, frag, attach = _thiocarbamoyl(PIPERAZINE_THIOUREA)
    assert name_substituent_for_ordering(mol, frag, attach) is None


def test_an_engine_run_asks_the_key_with_the_argument_the_engine_names_the_prefix_with():
    mol, frag, attach = _thiocarbamoyl(PIPERAZINE_THIOUREA)
    with engine_prefix_ordering(True):
        assert name_substituent_for_ordering(mol, frag, attach) == PREFIX
    # the mark is gone with the run, and the memo does not serve the engine's answer to a caller
    # that asks the cascade
    assert name_substituent_for_ordering(mol, frag, attach) is None


def test_an_engine_run_without_allow_mancude_marks_nothing():
    mol, frag, attach = _thiocarbamoyl(PIPERAZINE_THIOUREA)
    with engine_prefix_ordering(False):
        assert name_substituent_for_ordering(mol, frag, attach) is None


def test_the_mark_holds_for_the_session_depth_of_the_engine_call_only():
    # a substituent the engine names is named one session deeper, by a namer whose own prefixes
    # are asked the cascade as before
    mol, frag, attach = _thiocarbamoyl(PIPERAZINE_THIOUREA)
    with engine_prefix_ordering(True):
        fragment_naming.start_naming_session()
        try:
            assert name_substituent_for_ordering(mol, frag, attach) is None
        finally:
            fragment_naming.end_naming_session()
        assert name_substituent_for_ordering(mol, frag, attach) == PREFIX


@pytest.mark.parametrize("smiles,expected", [
    # (g): 'benzenesulfonyl' is cited before the thiocarbamoyl prefix (b before o,,
    #:3477) and takes locant 1, the one of the two orientations the heteroatom locants tie on
    (PIPERAZINE_THIOUREA,
     "1-(benzenesulfonyl)-4-[({[(2R)-oxolan-2-yl]methyl}amino)(sulfanylidene)methyl]piperazine"),
    ("S=C(NCc1ccccc1)N1CCN(S(=O)(=O)c2ccccc2)CC1",
     "1-(benzenesulfonyl)-4-[(benzylamino)(sulfanylidene)methyl]piperazine"),
    # the seven-membered ring is numbered by the same tier
    ("S=C(NCc1ccccc1)N1CCCN(S(=O)(=O)c2ccccc2)CC1",
     "1-(benzenesulfonyl)-4-[(benzylamino)(sulfanylidene)methyl]-1,4-diazepane"),
])
def test_the_prefix_cited_first_takes_locant_1_in_the_engines_clean_context(smiles, expected):
    name = tier_row(smiles, "best-effort")["name"]
    assert name == expected
