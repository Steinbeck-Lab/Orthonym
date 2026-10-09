"""Item 12a (task 2): a ring nitrogen cation by the book parent and the 'ium' suffix.

* (the Blue Book, "General rule for systematically naming cationic centers in
  parent hydrides"): '1-methylpyridin-1-ium (PIN)' (:41394); (:42290, "CATIONIC PREFIX
  NAMES"): 'pyridin-1-ium-1-yl (preferred prefix)' (:42324); (:42330, "CHOICE OF A PARENT
  STRUCTURE"): '2-(piperidin-1-ium-3-yl)propane-1,2-bis(aminium) (PIN)' (:42340).
* (:42219): "Where there is a choice, low locants for skeletal cationic centers are
  determined before considering locants for cationic suffixes."
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.cli import _emit_tier_flags
from orthonym.rules.monocycle_forms import monocycle_form
from orthonym.rules.terminal_ring import terminal_ring_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


def _ring_and_fv(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = next(r for r in mol.GetRingInfo().AtomRings()
                if any(mol.GetBondBetweenAtoms(1, a) for a in r))
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    return mol, list(ring), fv


def _form(smiles, **kw):
    mol, ring, fv = _ring_and_fv(smiles)
    branches = [a for a in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() not in ring and nb.GetIdx() != 1]
    return fv, monocycle_form(mol, ring, fv, branches, **kw)


@pytest.mark.parametrize("smiles,parent,prefix", [
    ("OC[n+]1ccccc1", "pyridine", "pyridin-1-yl"),
    ("OC[NH+]1CCCCC1", "piperidine", "piperidin-1-yl"),
    ("OCC1CC[NH2+]CC1", "piperidine", "piperidin-4-yl"),
    ("OC[N+]1(C)CCOCC1", "morpholine", "morpholin-4-yl"),
    ("OCn1cc[nH+]c1", "1H-imidazole", "1H-imidazol-1-yl"),
])
def test_a_ring_cation_is_spelled_only_for_a_caller_that_expresses_the_charge(smiles, parent, prefix):
    assert _form(smiles)[1] is None                      # the default: a charged ring declines
    fv, form = _form(smiles, allow_cation=True)
    assert (form.parent, form.prefix(fv)) == (parent, prefix)


@pytest.mark.parametrize("smiles,expected", [
    ("CCC1C[NH2+]CCN1", "3-ethylpiperazin-1-ium"),            #: the cationic N is 1
    ("CCC1COCC[NH2+]1", "3-ethylmorpholin-4-ium"),
    ("CCN1CC[NH+](C)CC1", "4-ethyl-1-methylpiperazin-1-ium"),
    ("CC1CC[NH+](C)CC1", "1,4-dimethylpiperidin-1-ium"),
    ("CCc1cc[n+](C)cc1C", "4-ethyl-1,3-dimethylpyridin-1-ium"),
    ("CCn1c[nH+]cc1", "1-ethyl-1H-imidazol-3-ium"),
    ("CCC1CCCC[N+]1(C)C", "2-ethyl-1,1-dimethylpiperidin-1-ium"),
])
def test_floor_names_a_ring_cation_by_its_book_parent(smiles, expected):
    name = name_universal_substitutive(Chem.MolFromSmiles(smiles)).name
    assert name == expected
    assert name_is_rt_exact(name, smiles)


# (1) (the Blue Book): "Where there is a choice for numbering, free valences
# receive lowest possible locants"; '4,4-dimethylpiperazin-4-ium-1-ylium (PIN)' (:42205): the
# free valence is cited at 1 and the cationic centre at 4, not the reverse.
def test_the_free_valence_is_numbered_before_the_cationic_centre():
    smiles = "CC(=O)Nc1ccc(N2CC[NH+](C)CC2)cc1"
    name = name_universal_substitutive(Chem.MolFromSmiles(smiles)).name
    assert "(4-methylpiperazin-4-ium-1-yl)" in name, name
    assert name_is_rt_exact(name, smiles)


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: molecules whose best-effort name was written by 'a' replacement at the base
E2E = [
    ("NC(=O)[C@@H]1CCC[NH2+]C1", "(3R)-3-carbamoylpiperidin-1-ium"),
    # the ring prefix and the round trip only: the numbering of the benzene parent is another
    # writer's (g) would number from the acetamido group)
    ("CC(=O)Nc1ccc(C[NH+]2CCOCC2)cc1", "(morpholin-4-ium-4-yl)"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", E2E)
def test_wider_tier_rings_end_to_end(smiles, expected):
    row = _row(smiles, "best-effort")
    name = row["name"]
    assert expected in name, row
    assert name_is_rt_exact(name, smiles)
    assert not {"MONO", "BENZ"} & set(F.detect(name))


def test_a_charged_ring_is_still_refused_by_the_terminal_ring_writer():
    mol = Chem.MolFromSmiles("CC(=O)c1cc[nH+]cc1")
    ring = list(mol.GetRingInfo().AtomRings()[0])
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    assert terminal_ring_name(mol, ring, fv) is None
