"""von Baeyer numbering must use the remaining freedom to give LOW locants.

The von Baeyer descriptor (ring size, main bridge, secondary-bridge superscripts)
is fixed by P-23.2.1 - P-23.2.6.  Once it is fixed, several numberings of the SAME
skeleton usually remain legal -- they differ in which main bridgehead becomes
locant 1 and in the traversal direction.  The Blue Book does not leave that choice
free:

* **P-23.3.1** "HETEROGENEOUS HETEROCYCLIC VON BAEYER PARENT HYDRIDES"
  (BlueBookV2.md:9765) -- *"Numbering is determined first by the fixed numbering of
  the hydrocarbon system."*
* **P-23.3.2** (:9777) -- *"When there is a choice for numbering, the following
  criteria are applied in order until a decision can be made."*
  **P-23.3.2.1** -- *"Low locants are assigned to the heteroatoms considered
  together as a set compared in increasing numerical order. The preferred numbering
  is the lowest set at the first point of difference."*
* **P-14.4** "NUMBERING" (:3219) -- *"When several structural features appear in
  cyclic and acyclic compounds, low locants are assigned to them in the following
  decreasing order of seniority:"* ... *"(c) principal characteristic groups and
  free valences (suffixes)"* (:3256), then *"(e) saturation/unsaturation"*, then
  *"(f) detachable alphabetized prefixes"*.

P-31.1.4.1 - P-31.1.4.4 (:16619-16700), the von Baeyer numbering cascade, covers
only unsaturation and skeletal heteroatoms; it is silent on principal
characteristic groups, so P-14.4(c) is the governing tier for them.

Before this module, the orientation was chosen by an arbitrary atom-index /
canonical-rank backstop and NO locant criterion at all was applied to substituted
or heteroatom-bearing cages.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --------------------------------------------------------------------------
# P-14.4(c) -- principal characteristic group gets the lowest locant.
# --------------------------------------------------------------------------

def test_memantine_amine_takes_locant_one(namer):
    """Memantine: the amine is the principal characteristic group.

    Its established name is 3,5-dimethyladamantan-1-amine -- the suffix at C1 and
    the two methyl prefixes at C3/C5.  All three substituents sit on bridgeheads,
    so the four adamantane bridgeheads (1,3,5,7) can be permuted freely by the
    orientation choice; P-14.4(c) forces the suffix onto the lowest of them.
    """
    name = namer.name("CC12CC3CC(C)(C1)CC([NH3+])(C3)C2")
    assert name == "3,5-dimethyltricyclo[3.3.1.1^3,7]decan-1-aminium", name


def test_amino_dimethyl_adamantanol_suffix_takes_locant_one(namer):
    """-ol outranks the amine (P-41 suffix seniority), so the -ol is the suffix and
    P-14.4(c) puts it at C1.  The three prefixes then occupy the remaining
    bridgeheads {3,5,7}; P-14.4(g) gives the lowest of those to the prefix cited
    first alphabetically ('amino' < 'methyl').
    """
    name = namer.name("CC12CC3(C)CC(N)(C1)CC(O)(C2)C3")
    assert name == "3-amino-5,7-dimethyltricyclo[3.3.1.1^3,7]decan-1-ol", name


# --------------------------------------------------------------------------
# P-23.3.2.1 -- skeletal heteroatoms get the lowest locant SET.
# --------------------------------------------------------------------------

MESOIONIC = "CC1=N[N+]2=CC=CC=C2C(=N1)[O-]"


def test_von_baeyer_heteroatom_locants_are_the_lowest_set():
    """Four numberings are legal for this bicyclo[4.4.0] cage (both segments are
    4 atoms), giving heteroatom sets (1,2,4), (1,8,10), (3,5,6) and (6,7,9).
    P-23.3.2.1 requires the lowest, (1,2,4).
    """
    from orthonym.rules.vonbaeyer_universal import analyze_cage_universal

    mol = Chem.MolFromSmiles(MESOIONIC)
    cage = analyze_cage_universal(mol, allow_mancude=True)
    chosen = tuple(sorted(
        cage.atom_to_locant[i] for i in sorted(cage.cage_atoms)
        if mol.GetAtomWithIdx(i).GetSymbol() != "C"
    ))
    assert chosen == (1, 2, 4), chosen


# --------------------------------------------------------------------------
# Guard: the descriptor itself must NOT move.  The orientation pass may only
# re-letter a fixed skeleton -- if it can change bridge lengths or superscripts
# it is choosing a different ring analysis, which P-23.2 already decided.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C1C2CC3CC1CC(C2)C3", "adamantane"),   # retained name, P-25.7.1.3
    ("C1CC2CCC1CC2", "bicyclo[2.2.2]octane"),
    ("C1CC2CCC3CCC1C23", "tricyclo[5.2.1.0^4,10]decane"),
])
def test_unsubstituted_cage_descriptors_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected
