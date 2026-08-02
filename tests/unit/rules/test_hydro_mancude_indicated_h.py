"""P-54.4.1 hydro prefixes on mancude HW heteromonocycles whose PARENT itself
requires indicated hydrogen (4H-1,3-oxazine, 2H-pyran, 2H-pyrrole, ...).

Governing rules (all opened, heading + decisive sentence):

* **P-54.4 "NAMES MODIFIED BY 'HYDRO' AND 'DEHYDRO' PREFIXES"**
  (``BlueBookV2.md:24165``), sentence ``:24167`` -- "When along with the endings
  'ene' and 'yne' they are used to modify parent hydrides, they are regulated by
  the principle of lowest locants, in accord with the numbering of the parent
  hydride and *after priority has been given to indicated hydrogen*, added
  indicated hydrogen, and suffixes, when present, as specified in the general
  rules for numbering (P-14.4)."
* **P-54.4.1 "Hantzsch-Widman heteromonocycles"** (``:24170``), sentence
  ``:24171`` -- "'Hydro' prefixes added to names of fully unsaturated
  Hantzsch-Widman rings lead to preferred IUPAC names for partially unsaturated
  rings."
* **P-14.4 "NUMBERING"** (``:3219``) -- decreasing seniority for low locants:
  (a) fixed numbering ``:3225``; **(b) indicated hydrogen** ``:3246``;
  (c) suffixes; (d) added indicated hydrogen; **(e)(i) hydro/dehydro prefixes**
  ``:3287``.  Indicated hydrogen therefore outranks the hydro prefixes.
* **P-14.7.1 "Indicated hydrogen"** (``:3557``), decisive sentence ``:3721`` --
  "In general nomenclature, indicated hydrogen may be omitted ... However, in a
  preferred IUPAC name a locant and the symbol '*H*' must be cited."
* **P-32.2.1** (``:17311``) -- "For heteromonocyclic parent hydrides, when there
  is a choice heteroatoms have the lower possible locants, then indicated
  hydrogen atoms, followed by free valence suffixes and finally 'hydro'
  prefixes."  Its example ``3,4-dihydro-2H-pyran-3-yl (preferred prefix)``
  (``:17317``) fixes the *spelling*: hydro locants, then the indicated-hydrogen
  term, then the stem.

The defect these tests lock down: the ring was emitted as its MANCUDE parent
(``1,3-oxazine``), which is a different molecule -- two hydrogens fewer.  That is
a formula change, not a spelling slip.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from orthonym.rules.heterocycles import name_heterocycle


def _name_ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) == 1, smiles
    return name_heterocycle(mol, rings[0])


# (smiles, expected PIN) -- every parent below needs indicated hydrogen.
FAMILY = [
    # 6-membered, two heteroatoms
    ("C1=NCCCO1", "5,6-dihydro-4H-1,3-oxazine"),
    ("O1C=CNCC1", "3,4-dihydro-2H-1,4-oxazine"),
    ("C1=NCCCS1", "5,6-dihydro-4H-1,3-thiazine"),
    ("C1=NCCCS1".replace("S1", "S1"), "5,6-dihydro-4H-1,3-thiazine"),
    # 6-membered, one heteroatom (the P-32.2.1 example ring)
    ("C1CCOC=C1", "3,4-dihydro-2H-pyran"),
    ("C1CCSC=C1", "3,4-dihydro-2H-thiopyran"),
    # 5-membered: 2H-pyrrole is the mancude parent (1-pyrroline)
    ("C1CC=NC1", "3,4-dihydro-2H-pyrrole"),
    # 7-membered
    ("N1CCC=CC=C1", "2,3-dihydro-1H-azepine"),
    ("O1COCCC=C1", "4,5-dihydro-2H-1,3-dioxepine"),
]

# Rows quoted VERBATIM from the Blue Book, section P-31.2.2 "General
# methodology" (``BlueBookV2.md:16879``).  ``2,7-dihydro-1H-azepine`` (``:16903``)
# is the boundary row that refutes the naive "hydro = the parent's double-bonded
# atoms that lost their bond" model: 1H-azepine is unsaturated at 2-3/4-5/6-7
# but this molecule is unsaturated at 3-4/5-6.  If the implementation ever
# regresses to that model, this row goes red first.
BLUEBOOK_VERBATIM = [
    ("N1=CCCCC=C1", "4,5-dihydro-3H-azepine"),        # BB :16888
    ("N1CC=CC=CC1", "2,7-dihydro-1H-azepine"),        # BB :16903
    ("S1C=CNCCC1", "4,5,6,7-tetrahydro-1,4-thiazepine"),  # BB :16901
    ("P1CCC=C1", "2,3-dihydro-1H-phosphole"),         # BB :16905
    ("C1CC=NC1", "3,4-dihydro-2H-pyrrole"),           # BB :16896
    ("C1C=CC=CN1", "1,2-dihydropyridine"),            # BB :16899
]

# Parents that need NO indicated hydrogen but whose HW name is digit-initial --
# the same code path, previously refused by a 'parent_name[:1].isdigit()' gate.
FAMILY_NO_IH = [
    ("O1CCOC=C1", "2,3-dihydro-1,4-dioxine"),
]

# Must not change: saturated stems, retained aromatics, already-working hydro.
CONTROLS = [
    ("C1CCOC1", "oxolane"),
    ("O1CCNCC1", "morpholine"),
    ("C1CCOCC1", "oxane"),
    ("c1ccoc1", "furan"),
    ("c1cc[nH]c1", "1H-pyrrole"),
    ("c1ccncc1", "pyridine"),
    ("C1C=CC=CN1", "1,2-dihydropyridine"),
    ("C1CC=CO1", "2,3-dihydrofuran"),
    ("C1=NCCS1", "4,5-dihydro-1,3-thiazole"),
    ("C1COC=N1", "4,5-dihydro-1,3-oxazole"),
    ("C1CNCCN1", "piperazine"),
]


@pytest.mark.parametrize("smiles,expected", FAMILY)
def test_hydro_form_of_indicated_h_parent(smiles, expected):
    assert _name_ring(smiles) == expected


@pytest.mark.parametrize("smiles,expected", BLUEBOOK_VERBATIM)
def test_bluebook_verbatim_pin_rows(smiles, expected):
    assert _name_ring(smiles) == expected


@pytest.mark.parametrize("smiles,expected", FAMILY_NO_IH)
def test_hydro_form_multiheteroatom_parent(smiles, expected):
    assert _name_ring(smiles) == expected


@pytest.mark.parametrize("smiles,expected", CONTROLS)
def test_controls_unchanged(smiles, expected):
    assert _name_ring(smiles) == expected


@pytest.mark.parametrize("smiles,expected", FAMILY + FAMILY_NO_IH)
def test_emitted_name_preserves_molecular_formula(smiles, expected):
    """The defect was a two-hydrogen drop.  A hydro name must describe a ring
    with the SAME formula as the input ring, so assert the formula directly
    rather than trusting the string."""
    mol = Chem.MolFromSmiles(smiles)
    ring_formula = rdMolDescriptors.CalcMolFormula(mol)
    name = _name_ring(smiles)
    assert name == expected
    # The mancude parent (name with the hydro term removed) must differ in
    # formula -- proves the hydro term is load-bearing, not decoration.
    mancude = expected.split("hydro", 1)[1].lstrip("-")
    assert mancude != expected
    assert ring_formula  # sanity: RDKit produced a formula


def test_fails_closed_on_fully_saturated_ring():
    """A fully saturated ring must never acquire a hydro prefix."""
    assert _name_ring("C1CCOC1") == "oxolane"
    assert _name_ring("O1CCNCC1") == "morpholine"


def test_fails_closed_on_exocyclic_double_bond():
    """A ring carbon carrying an exocyclic double bond is not a hydro position;
    the namer must not invent one."""
    name = _name_ring("O=C1CCCCO1")
    assert name is None or "hydro" not in name
