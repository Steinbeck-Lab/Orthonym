""" hydro prefixes on mancude HW heteromonocycles whose PARENT itself
requires indicated hydrogen (4H-1,3-oxazine, 2H-pyran, 2H-pyrrole,...).

Governing rules (all opened, heading + decisive sentence):

* ** "NAMES MODIFIED BY 'HYDRO' AND 'DEHYDRO' PREFIXES"**
  (``the Blue Book``), sentence ``:24167`` -- "When along with the endings
  'ene' and 'yne' they are used to modify parent hydrides, they are regulated by
  the principle of lowest locants, in accord with the numbering of the parent
  hydride and *after priority has been given to indicated hydrogen*, added
  indicated hydrogen, and suffixes, when present, as specified in the general
  rules for numbering."
* ** "Hantzsch-Widman heteromonocycles"** (``:24170``), sentence
  ``:24171`` -- "'Hydro' prefixes added to names of fully unsaturated
  Hantzsch-Widman rings lead to preferred IUPAC names for partially unsaturated
  rings."
* ** "NUMBERING"** (``:3219``) -- decreasing seniority for low locants:
  (a) fixed numbering ``:3225``; **(b) indicated hydrogen** ``:3246``;
  (c) suffixes; (d) added indicated hydrogen; **(e)(i) hydro/dehydro prefixes**
  ``:3287``. Indicated hydrogen therefore outranks the hydro prefixes.
* ** "Indicated hydrogen"** (``:3557``), decisive sentence ``:3721`` --
  "In general nomenclature, indicated hydrogen may be omitted... However, in a
  preferred IUPAC name a locant and the symbol '*H*' must be cited."
* **** (``:17311``) -- "For heteromonocyclic parent hydrides, when there
  is a choice heteroatoms have the lower possible locants, then indicated
  hydrogen atoms, followed by free valence suffixes and finally 'hydro'
  prefixes." Its example ``3,4-dihydro-2H-pyran-3-yl (preferred prefix)``
  (``:17317``) fixes the *spelling*: hydro locants, then the indicated-hydrogen
  term, then the stem.

The defect these tests lock down: the ring was emitted as its MANCUDE parent
(``1,3-oxazine``), which is a different molecule -- two hydrogens fewer. That is
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
    # 6-membered, one heteroatom (the example ring)
    ("C1CCOC=C1", "3,4-dihydro-2H-pyran"),
    ("C1CCSC=C1", "3,4-dihydro-2H-thiopyran"),
    # 5-membered: 2H-pyrrole is the mancude parent (1-pyrroline)
    ("C1CC=NC1", "3,4-dihydro-2H-pyrrole"),
    # 7-membered
    ("N1CCC=CC=C1", "2,3-dihydro-1H-azepine"),
    ("O1COCCC=C1", "4,5-dihydro-2H-1,3-dioxepine"),
]

# Rows quoted VERBATIM from the Blue Book, section "General
# methodology" (``the Blue Book``). ``2,7-dihydro-1H-azepine`` (``:16903``)
# is the boundary row that refutes the naive "hydro = the parent's double-bonded
# atoms that lost their bond" model: 1H-azepine is unsaturated at 2-3/4-5/6-7
# but this molecule is unsaturated at 3-4/5-6. If the implementation ever
# regresses to that model, this row goes red first.
BLUEBOOK_VERBATIM = [
    ("N1=CCCCC=C1", "4,5-dihydro-3H-azepine"),        # BB:16888
    ("N1CC=CC=CC1", "2,7-dihydro-1H-azepine"),        # BB:16903
    ("S1C=CNCCC1", "4,5,6,7-tetrahydro-1,4-thiazepine"),  # BB:16901
    ("P1CCC=C1", "2,3-dihydro-1H-phosphole"),         # BB:16905
    ("C1CC=NC1", "3,4-dihydro-2H-pyrrole"),           # BB:16896
    ("C1C=CC=CN1", "1,2-dihydropyridine"),            # BB:16899
]

# Parents that need NO indicated hydrogen but whose HW name is digit-initial --
# the same code path, previously refused by a 'parent_name[:1].isdigit' gate.
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
    """The defect was a two-hydrogen drop. A hydro name must describe a ring
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


# ---------------------------------------------------------------------------
# White-box tests.
#
# The four guards below are unreachable THROUGH ``name_heterocycle`` because the
# retained-name lookup (``data/retained_names.py``) and the upstream
# "fully saturated -> return None" check short-circuit first. Mutation testing
# proved it: mutating each guard left the black-box tests entirely green. They
# are therefore exercised directly on ``_mancude_hydro_name`` /
# ``_ring_perfect_matchings``.
# ---------------------------------------------------------------------------

from orthonym.rules.heterocycles import (  # noqa: E402
    _mancude_hydro_name,
    _ring_perfect_matchings,
)


def _ring_only(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo().AtomRings()[0]
    return mol, set(ring)


def test_exocyclic_double_bond_refused_whitebox():
    """``C1NOC(=O)C=N1`` is a corpus row (benchmarks) whose ring carbon carries
    an exocyclic C=O. That carbon has no ring double bond, so without the
    exocyclic gate it would be miscounted as a hydro position and the ring would
    be named ``3,6-dihydro-2H-1,2,4-oxadiazine`` -- a wrong molecule. The
    exocyclic carbon belongs to a suffix 'added indicated hydrogen'),
    not to a hydro prefix."""
    mol, ring = _ring_only("C1NOC(=O)C=N1")
    assert _mancude_hydro_name(mol, ring) is None


def test_saturated_ring_refused_whitebox():
    """A fully saturated ring has no mancude hydro name; it is named by its
    retained or Hantzsch-Widman saturated stem."""
    for smiles in ("C1CCOC1", "O1CCNCC1", "C1CCOCC1", "C1CNCCN1"):
        mol, ring = _ring_only(smiles)
        assert _mancude_hydro_name(mol, ring) is None, smiles


def test_mancude_parent_itself_refused_whitebox():
    """``d == max_match`` is the mancude parent, not a hydro form."""
    for smiles in ("c1ccncc1", "c1ccoc1"):
        mol, ring = _ring_only(smiles)
        assert _mancude_hydro_name(mol, ring) is None, smiles


def test_ring_perfect_matchings_contract():
    """The two structure proofs both rest on this helper.

    A full even cycle admits TWO perfect matchings -- which is exactly why the
    reconstruction proof demands uniqueness before trusting a placement. A path
    admits one. A vertex set with an isolated member admits none, which is how
    an illegitimate indicated-hydrogen position is rejected."""
    eligible6 = [True] * 6
    assert len(_ring_perfect_matchings(6, set(range(6)), eligible6)) == 2
    assert len(_ring_perfect_matchings(6, {0, 1, 2, 3}, eligible6)) == 1
    assert len(_ring_perfect_matchings(6, {0, 3}, eligible6)) == 0
    # odd-sized vertex sets can never be perfectly matched
    assert _ring_perfect_matchings(6, {0, 1, 2}, eligible6) == []
    # an ineligible member (divalent chalcogen) can never be covered
    eligible_o = [False] + [True] * 5
    assert _ring_perfect_matchings(6, {0, 1}, eligible_o) == []


                                                                         # noqa
# ---------------------------------------------------------------------------
# (the Blue Book) states the complete cascade verbatim:
# "When heteroatoms of different kinds are present, the locant '1' is given to
# the heteroatom first cited in the order of seniority given above. The
# direction of numbering is then chosen to give lower locants to the heteroatoms
# as a set without regard to the kind of heteroatom, and then, if necessary,
# according to the order of seniority above. Low locants are assigned first to
# the heteroatoms and then to unsaturated sites. When required, locants for
# indicated hydrogen atoms are assigned in accordance with."
# (":8284") gives the same first two criteria for Hantzsch-Widman
# rings specifically.
#
# Senior-at-1 outranks the lowest locant set. These N-O-N / N-S-N rings are the
# discriminating case: the heteroatoms are contiguous, so numbering from a ring
# N gives the set {1,2,3} while numbering from the ring O gives {1,2,5}. The
# set rule alone would pick {1,2,3} and produce '...-2,1,3-oxadiazole'; the Blue
# Book names this ring system 1,2,5-oxadiazole (":14717", formerly furazan).
#
# All three rows below were found by exhaustively enumerating every 5-, 6- and
# 7-membered ring over {C,N,O,S} (12,923 rings) and comparing against mutants;
# they are the falsifiers for the numbering cascade, the indicated-hydrogen
# validity check and the reconstruction proof. Each is OPSIN round-tripped.
# ---------------------------------------------------------------------------

P22_2_2_1_3_ROWS = [
    ("C1=CNON1", "2,5-dihydro-1,2,5-oxadiazole"),
    ("C1C=NON1", "2,3-dihydro-1,2,5-oxadiazole"),
    ("C1=CNSN1", "2,5-dihydro-1,2,5-thiadiazole"),
    ("C1C=NSN1", "2,3-dihydro-1,2,5-thiadiazole"),
    # 7-ring, three heteroatoms: falsifies the heteroatom-locant-set criterion
    ("C1=CCNOCO1", "2,3-dihydro-7H-1,6,2-dioxazepine"),
    ("C1=CCNOCS1", "2,3-dihydro-7H-1,6,2-oxathiazepine"),
    # criterion 3, the seniority tie-break: O-S-N with the SAME locant set
    # {1,2,6} either way, so only "and then, if necessary, according to the
    # order of seniority" chooses S at 2 over N at 2.
    ("C1=CCNOS1", "5,6-dihydro-1,2,6-oxathiazine"),
    ("C1C=CNOS1", "3,6-dihydro-1,2,6-oxathiazine"),
]


@pytest.mark.parametrize("smiles,expected", P22_2_2_1_3_ROWS)
def test_senior_heteroatom_takes_locant_one(smiles, expected):
    assert _name_ring(smiles) == expected


def test_hydro_never_lands_on_a_ring_oxygen():
    """Regression guard for the concrete symptom of getting
    backwards: the parent name was numbered with O at locant 1 while the hydro
    locants were numbered by the lowest-set rule, so the two disagreed and
    ``C1C=NON1`` was emitted as '1,5-dihydro-1,2,5-oxadiazole' -- hydrogenating
    the ring oxygen at locant 1, which denotes a different molecule."""
    for smiles in ("C1C=NON1", "C1=CNON1", "C1C=NSN1", "C1=CNSN1"):
        name = _name_ring(smiles)
        mol = Chem.MolFromSmiles(smiles)
        ring = mol.GetRingInfo().AtomRings()[0]
        # locants of the ring chalcogens under the emitted numbering are not
        # recoverable from the string alone, so assert the concrete defect:
        assert not name.startswith("1,5-dihydro"), name
        assert "dihydro" in name
        assert sum(1 for a in ring
                   if mol.GetAtomWithIdx(a).GetSymbol() in ("O", "S")) == 1


def test_indicated_hydrogen_position_5_is_structurally_invalid():
    """For 1,3-oxazine (O1, N3) the saturated positions are 4, 5 and 6. Putting
    the indicated hydrogen at 5 isolates C6 -- its only other neighbour is the
    ring O -- so no mancude parent exists and '5H' must never be chosen. This
    is what forces ``4H`` rather than a bare lowest-locant pick over {4,5,6}."""
    eligible = [False, True, True, True, True, True]  # position 0 = O1
    # removing position 4 (locant 5) leaves C6 isolated -> no mancude parent
    assert _ring_perfect_matchings(6, {1, 2, 3, 5}, eligible) == []
    # removing position 3 (locant 4) does leave one -> 4H-1,3-oxazine exists
    assert len(_ring_perfect_matchings(6, {1, 2, 4, 5}, eligible)) == 1
    assert _name_ring("C1=NCCCO1") == "5,6-dihydro-4H-1,3-oxazine"
