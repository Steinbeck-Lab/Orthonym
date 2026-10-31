"""Rows of the 2026-10-30 filtered PubChem run that ended in an exception or in a native call that does not return.

Slow3 census (best-effort tier, each row alone, 120 s): the 8 exception rows that are no refusal of RDKit's reader
(5 'Digraph generation failed', 2 IndexError in the isotope licence, 1 more) and the polyazene rows that hung RDKit's
resonance enumeration. Each test pins the cause, not the row.
"""
import subprocess
import sys
import textwrap

import pytest
from rdkit import Chem, RDLogger

from orthonym import namer
from orthonym.perception.charge_shift import has_mobile_cation
from orthonym.rules.isotope_licence import stands_before_a_one_hydrogen_group

pytestmark = pytest.mark.unit

#: two porphyrin-zinc dimers of bis-imide cages: RDKit's CIP labeller needs more than 100,000 digraph nodes for them
#: and raises 'Digraph generation failed' (PubChem CID 71733097; 28 specified tetrahedral centres, no stereo bond)
DIGRAPH_ROW = (
    "COC(=O)[C@]12[C@@H]3[C@@H]4C[C@H]([C@@H]3[C@](O1)([C@H]5[C@@H]2[C@H]6C[C@@H]5[C@@H]7[C@H]6C(=O)N(C7=O)"
    "C8=CC=C(C=C8)N9C(=O)[C@@H]1[C@H]2C[C@@H]([C@@H]1C9=O)[C@H]1[C@@H]2[C@@]2([C@H]3[C@H]5C[C@@H]([C@H]3[C@]1(O2)"
    "C(=O)OC)[C@H]1[C@@H]5C(=O)N(C1=O)C1=CC=C(C=C1)C1=C2C=CC(=N2)C(=C2C=CC(=C(C3=NC(=C(C5=CC=C1[N-]5)C1=CC=CC=C1)"
    "C=C3)C1=CC=CC=C1)[N-]2)C1=CC=CC=C1)C(=O)OC)C(=O)OC)[C@@H]1[C@H]4C(=O)N(C1=O)C1=CC=C(C=C1)C1=C2C=CC(=N2)"
    "C(=C2C=CC(=C(C3=NC(=C(C4=CC=C1[N-]4)C1=CC=CC=C1)C=C3)C1=CC=CC=C1)[N-]2)C1=CC=CC=C1.[Zn+2].[Zn+2]")


def test_the_stereo_count_does_not_ask_the_cip_labeller():
    """``_specified_stereo_count`` counts specified centres and stereo bonds; the count needs no CIP label, and the
    labeller raised RuntimeError ('Digraph generation failed: more than 100000nodes found') on a large symmetric
    molecule, which ended the naming of the row with that exception. FindPotentialStereo, which the count reads,
    gives the same centres with or without the labels."""
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(DIGRAPH_ROW)
    assert namer._specified_stereo_count(mol) == 28
    small = Chem.MolFromSmiles("C[C@H](F)C(Cl)[C@@H](Br)C/C=C/C")
    assert namer._specified_stereo_count(small) == len(
        Chem.FindMolChiralCenters(Chem.Mol(small), includeUnassigned=False, useLegacyImplementation=False)) + 1


def test_the_omission_check_reads_the_large_molecule_without_an_exception():
    assert namer._name_omits_input_stereo(DIGRAPH_ROW, DIGRAPH_ROW) is False


@pytest.mark.parametrize("smiles", [
    "[2H+].CC1=CC=CC=C1",                       # a lone hydrogen ion is the only labelled hydrogen
    "[2H+].[O-]S(=O)(=O)CC12CCC(CC1=O)C2",      # the camphorsulfonate rows of the run
])
def test_a_labelled_hydrogen_ion_is_no_one_hydrogen_group(smiles):
    """The licence for a count-less descriptor before 'methylidene' reads the one labelled hydrogen of the molecule
    and the carbon it is bonded to; a hydrogen ion is bonded to nothing, and ``GetNeighbors[0]`` raised
    IndexError for 2 rows of the run. The licence is not given then (the descriptor shows its count)."""
    RDLogger.DisableLog("rdApp.*")
    assert stands_before_a_one_hydrogen_group(Chem.MolFromSmiles(smiles), "methylidene", 0) is False


def test_a_labelled_methine_still_licenses_the_descriptor():
    """The reading the guard must not change: the one labelled hydrogen is on an acyclic carbon that bears no
    other hydrogen (the labelled hydrogen itself is the carbon's only one)."""
    RDLogger.DisableLog("rdApp.*")
    assert stands_before_a_one_hydrogen_group(Chem.MolFromSmiles("[2H]C(=O)c1ccccc1"), "methylidene", 0) is True
    assert stands_before_a_one_hydrogen_group(Chem.MolFromSmiles("[2H]CCC"), "methylidene", 0) is False   # CH2D
    assert stands_before_a_one_hydrogen_group(Chem.MolFromSmiles("[2H]C1CCCCC1"), "methylidene", 0) is False  # ring
    assert stands_before_a_one_hydrogen_group(Chem.MolFromSmiles("[2H]C(=O)c1ccccc1"), "ethylidene", 0) is False


def test_polyazene_chains_are_not_asked_to_the_resonance_enumeration():
    """RDKit's enumeration of the resonance structures of N=N-N=N-... costs 4 times more for each two nitrogens
    (14 s at 21 pi atoms) and its call cannot be interrupted; the +1 of the azido group at the end of the chain
    does not move (``charge_shift.azide_atoms``), so the molecule has no charge form to look for.
    Run in a child process: were the gate lost, the test would end at the time-out and not hang the suite."""
    chain = "N/N=N/" * 7 + "N/N=N/N=[N+]=[N-]"
    code = textwrap.dedent(f"""
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        from orthonym import namer
        print(namer._charge_forms({chain!r}))
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().endswith("None"), done.stdout + done.stderr


@pytest.mark.parametrize("smiles, expected", [
    ("CCN=[N+]=[N-]", False),                       # an azide: its +1 does not move
    ("[N-]-[N+]#N", False),                         # its charge-separated drawing
    ("CN(C)N=[N+]=[N-]", False),                    # the host nitrogen is part of the group
    ("C[n+]1ccccc1", True),                         # an N-methylpyridinium: a mobile nitrogen cation
    ("Cc1c[nH]c[nH+]1", True),
    ("C(N=[N+]=[N-])C.C[n+]1ccccc1", True),         # one outside the azide is enough
    ("C[N+](C)(C)C", False),                        # no pi bond to move the charge along
    ("CCO", False),
])
def test_a_mobile_nitrogen_cation_outside_an_azido_group(smiles, expected):
    RDLogger.DisableLog("rdApp.*")
    assert has_mobile_cation(Chem.MolFromSmiles(smiles)) is expected
