"""The numbering of a von Baeyer cage with several detachable prefixes (large-polycycle plan, Task 3a).

 NUMBERING (the Blue Book): (f) "detachable alphabetized prefixes, all considered together
in a series of increasing numerical order" (:3300) and then (g) "lowest locants for the substituent cited
first as a prefix in the name" (:3307), with ``1-methyl-4-nitronaphthalene (PIN) (not
4-methyl-1-nitronaphthalene)``. The prefixes are cited in alphanumerical order. The numbering
criteria of a cage stopped at (f), so a cage whose numberings tied there took the numbering the input
atom order produced: ``ClC1C2C(Br)C3C1C3C2`` was ``3-bromo-5-chloro...`` or ``5-bromo-3-chloro...``
by SMILES spelling, and ``5-ethyl-3-methyltricyclo[2.2.1.0^2,6]heptane`` for every spelling of an
ethyl/methyl cage (deterministic, but not the (g) numbering).

The expected names come from an oracle that does not use the numbering choice under test: every
numbering of a cage is the numbering of its bare skeleton composed with an automorphism of that skeleton
(an element-blind graph automorphism), so the oracle takes the lowest (c) suffix locants, then (f) all prefix
locants, then (g) the locants in citation order, over all of them.
"""
import pytest

from tests.support.cage_numbering_oracle import check
from tests.support.default_tier import strict_path_row
from tests.support.large_polycycles import be, spellings

pytestmark = pytest.mark.opsin_gate

SPELLINGS = 40


# two or more DIFFERENT prefixes on a cage (the reviewer's molecules), then two alkyl prefixes
DIFFERENT_PREFIXES = [
    "ClC1C2C(Br)C3C1C3C2",            # 3-bromo-5-chloro
    "ClC1C2C(F)C3C1C3C2",             # 3-chloro-5-fluoro
    "COC1C2C(Cl)C3C1C3C2",            # 3-chloro-5-methoxy
    "FC1C2C3CC1C(Br)C3C2",            # tricyclo[3.2.1.0^3,6]octane: 2-bromo-8-fluoro
    "ClC1C2C(I)C3C(Br)C1C3C2",        # three different prefixes
    "CCC1C2C(C)C3C1C3C2",             # 3-ethyl-5-methyl: ethyl is cited first
    "CCC1C2C3CC1C(C)C3C2",            # tricyclo[3.2.1.0^3,6]octane: 2-ethyl-8-methyl
]
# the suffix and two different prefixes; a gem-disubstituted atom; prefixes on a carbinol atom
SUFFIX_AND_PREFIXES = [
    "OC1C2C(Cl)C3C1C3C2Br",           # -ol with bromo and chloro
    "OC1C2C3C(F)C1C(Cl)(Cl)C23",      # -ol with fluoro and gem-dichloro
]
GEM_AND_CARBINOL = [
    "ClC1(Cl)C2CC3C(C2Br)C31",        # 3,3-dichloro and 5-bromo: {3,3,5} is lower than {3,5,5}
    "OC1C2C3CC1C(O)(Cl)C32",          # 3,5-diol with chloro on one carbinol atom
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles", DIFFERENT_PREFIXES + SUFFIX_AND_PREFIXES + GEM_AND_CARBINOL)
def test_the_numbering_is_the_lowest_and_never_depends_on_the_spelling(smiles):
    names = set()
    for spelling in spellings(smiles, SPELLINGS):
        row = strict_path_row(spelling)
        assert row["name"] and row["tier"] != "abstain", row
        names.add(row["name"])
    assert len(names) == 1, sorted(names)
    check(smiles, names.pop())


@pytest.mark.unit
@pytest.mark.parametrize("smiles", ["ClC1C2C(Br)C3C1C3C2", "CCC1C2C(C)C3C1C3C2", "OC1C2C(Cl)C3C1C3C2Br"])
def test_best_effort_names_the_same_numbering(smiles):
    names = {be(s)["name"] for s in spellings(smiles, SPELLINGS)}
    assert len(names) == 1, sorted(names)
    check(smiles, names.pop())


def _ring_system_of(mol, atom_idx):
    """The atoms of the ring system (rings joined by shared atoms) that holds ``atom_idx``."""
    system = set()
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx in ring:
            system |= set(ring)
    grown = True
    while grown:
        grown = False
        for ring in mol.GetRingInfo().AtomRings():
            if system & set(ring) and not set(ring) <= system:
                system |= set(ring)
                grown = True
    return system


@pytest.mark.unit
def test_the_free_valence_of_a_cage_named_as_a_substituent_takes_the_lowest_locant():
    """A cage named AS a substituent of the rest of the molecule: its neighbour outside the group is
    the parent, so the ring atom bonded to it takes the lowest locant together with the other suffixes
     (c), the Blue Book "principal characteristic groups and free valences (suffixes)"),
    and it is not a prefix cited at (g). 7,9-Dioxabicyclo[4.3.0]nona-1,3,5-triene with a methoxy group
    and the phenyl parent: the free valence is 3 and the methoxy 4 (the numbering that cites the methoxy
    prefix first at the lowest locant would give the free valence 4)."""
    from rdkit import Chem

    from orthonym.rules.polycyclic import VonBaeyerAnalyzer
    for spelling in spellings("COc1cc2c(cc1-c1ccccc1)OCO2", 12):
        mol = Chem.MolFromSmiles(spelling)
        biaryl = next(b for b in mol.GetBonds() if not b.IsInRing()
                      and b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic())
        ends = (biaryl.GetBeginAtomIdx(), biaryl.GetEndAtomIdx())
        cage_end = next(i for i in ends if any(a.GetSymbol() == "O" for a in
                                               (mol.GetAtomWithIdx(j) for j in _ring_system_of(mol, i))))
        cage = _ring_system_of(mol, cage_end)
        methoxy_o = next(n for i in cage for n in mol.GetAtomWithIdx(i).GetNeighbors()
                         if n.GetSymbol() == "O" and n.GetIdx() not in cage)
        methoxy_ring_atom = next(n.GetIdx() for n in methoxy_o.GetNeighbors() if n.GetIdx() in cage)
        group = cage | {methoxy_o.GetIdx()} | {n.GetIdx() for n in methoxy_o.GetNeighbors() if n.GetIdx() not in cage}
        numbering = VonBaeyerAnalyzer().analyze(mol, cage, substituent_atoms=group).numbering
        assert numbering[cage_end] < numbering[methoxy_ring_atom], (spelling, numbering)


@pytest.mark.unit
def test_symmetric_halves_keep_their_stereodescriptors_and_the_z_takes_the_lower_locant():
    """ (j) (the Blue Book): "the lower locant is assigned to CIP stereodescriptors Z, R, M, and
    r (pseudoasymmetry) that are preferred to E, S, P, and s", with ``(2Z,5E)-hepta-2,5-dienedioic acid
    (PIN)`` (the chain is numbered by assigning the low locant to the 'Z' double bond). The two halves of
    this bicyclo[3.3.0]octadiene are equivalent, one hydrazinylidene Z and one E, so the name is one of
    ``(3Z,7E)`` and ``(3E,7Z)``; the mapping of the input onto its canonical copy ignored double-bond
    stereo and could swap the halves, so the name followed the spelling."""
    smiles = "CC(=O)N/N=C/1\\CC2=C/C(=N\\NC(=O)C)/CC2=C1"
    names = {be(s)["name"] for s in spellings(smiles, SPELLINGS)}
    assert names == {"(3Z,7E)-3,7-bis(acetylhydrazinylidene)bicyclo[3.3.0]octa-1(8),4-diene"}, names


@pytest.mark.unit
def test_a_macrocyclic_cage_cites_the_ethyl_at_its_lowest_locant():
    """ (g) (the Blue Book, section `` NUMBERING``): of two numberings whose prefix locant
    sets tie, the one that gives the lowest locant to the prefix cited first. The tetrolide has one ethyl and
    seven methyls, the prefix locant sets of the two numberings are both {2,5,11,14,20,23,29,32}, and the
    ethyl is cited first, so it takes 5, not 32. Both names read back exactly (OPSIN 2.9.0), so only
    the rule chooses; this was the canary row 536 known defect (the numbering followed the input atom order)."""
    smiles = ("CCC1CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)"
              "C(C)C(=O)O1")
    expected = ("5-ethyl-2,11,14,20,23,29,32-heptamethyl-4,13,22,31,37,38,39,40-octaoxapentacyclo"
                "[32.2.1.1^7,10.1^16,19.1^25,28]tetracontane-3,12,21,30-tetrone")
    assert {be(s)["name"] for s in [smiles] + spellings(smiles, 10)} == {expected}


@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    ("[2H]C1C2CC3C1C3C2", "(3-2H1)tricyclo[2.2.1.0^2,6]heptane"),
    ("[2H]C1C2CC3C1C3C2Cl", "3-chloro(5-2H1)tricyclo[2.2.1.0^2,6]heptane"),
])
def test_an_isotopically_modified_atom_takes_the_lowest_locant_the_prefixes_leave(smiles, expected):
    """ (i) (the Blue Book, section `` NUMBERING``), after the prefixes (f), (g): isotopically
    modified atoms take the lowest locants, with ``(2S,4R)-(4-2H1,2-3H1)pentane (PIN)``. The two numberings
    of the deuterated cage tie on everything before (i); the chloro cage is settled by (g) first, so the
    deuterium takes 5. Both numberings of each name read back to the input (OPSIN 2.9.0). This pins the
    outcome and its spelling independence; it does not isolate (i), because the canonical atom ranks that
    follow it choose the same numbering on every cage measured (all two- and three-isotope placements on
    three cages: no name differs without the (i) key)."""
    assert {strict_path_row(s)["name"] for s in spellings(smiles, SPELLINGS)} == {expected}


@pytest.mark.unit
def test_a_nonstandard_valence_atom_takes_the_lower_locant_before_the_stereodescriptors():
    """ (h) (the Blue Book, section `` NUMBERING``): of the numberings that tie on the
    heteroatoms and the prefixes, "the one in a nonstandard valence state is assigned the lower locant". The
    two lambda3 tellurium atoms of this heptatellura cage are bonded to each other, so they take 1 and 2."""
    from rdkit import Chem

    from orthonym.rules.polycyclic import VonBaeyerAnalyzer, get_heteroatom_replacement_prefix
    for spelling in spellings("[TeH]1[Te][TeH]2[Te][Te][Te]1[Te]2", 12):
        mol = Chem.MolFromSmiles(spelling)
        ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
        numbering = VonBaeyerAnalyzer().analyze(mol, ring_atoms).numbering
        assert get_heteroatom_replacement_prefix(mol, numbering, ring_atoms) == "1λ3,2λ3,3,4,5,6,7-heptatellura", spelling


ZWITTERION = ("C1=Cc2nc1c(-c1ccccc1)c1ccc([n-]1)c(-c1ccccc1)c1[nH+]c(c(-c3ccccc3)c3ccc([nH]3)c2-c2ccccc2)C=C1")
DICATION = "C1=Cc2[nH+]c1c(-c1ccccc1)c1ccc([nH]1)c(-c1ccccc1)c1[nH+]c(c(-c3ccccc3)c3ccc([nH]3)c2-c2ccccc2)C=C1"


@pytest.mark.unit
@pytest.mark.parametrize("smiles, ending", [
    (ZWITTERION, "-22-ium-21-ide"),
    (DICATION, "-21,23-diium"),
])
def test_the_ionic_centres_take_the_lowest_locants_before_the_unsaturation(smiles, ending):
    """ Ionic centers in the same parent structure (the Blue Book): "Where there is a choice, lowest locants are given to
    the ionic centers in the following order, listed in decreasing order of seniority: 'uide' ('uida'), 'ide'
    ('ida'), 'ylium' ('ylia'), and 'ium' ('onia')"; the ionic suffixes are suffixes, which (c)
    (the Blue Book) ranks above the 'ene' endings of (e). The four nitrogen atoms of a porphyrin are 21 to
    24 in every numbering of its cage, and the anionic centre takes 21, the cationic one the lowest left
    (the numbering followed the unsaturation and the input atom order before, ``-24-ium-22-ide``). Both names
    read back to the input (OPSIN 2.9.0)."""
    names = {be(s)["name"] for s in spellings(smiles, 12)}
    assert len(names) == 1, sorted(names)
    assert names.pop().endswith(ending)


@pytest.mark.unit
def test_an_aromatic_branch_of_a_kekulized_cage_is_cited_under_its_own_name():
    """ (alphanumerical order is that of the complete prefix name): the cage is analysed on a kekulized
    copy, where a phenyl branch is a cyclohexatrienyl one, cited under ``c`` and ahead of an ``e`` prefix, so the
    numbering gave the lowest locant to the wrong one of phenyl and isocyanato (``p`` and ``i``) of a porphyrin
    (``tests/unit/routing/test_porphyrin_route.py``, the isocyanato row)."""
    from rdkit import Chem

    from orthonym.rules.cage_numbering import _name_keys, aromatic_view, cage_branches
    mol = Chem.MolFromSmiles("CC1CC2CC1C2c1ccccc1")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() if len(ring) != 6 for i in ring}
    kekulized = _name_keys(mol, cage_branches(mol, ring_atoms, set()))
    restored = _name_keys(aromatic_view(mol), cage_branches(aromatic_view(mol), ring_atoms, set()))
    assert "phenyl" in restored.values() and "phenyl" not in kekulized.values(), (kekulized, restored)


@pytest.mark.unit
def test_numberings_the_ranked_criteria_leave_apart_keep_the_analysis_choice():
    """ UNBRANCHED POLYSPIRO RING SYSTEMS WITH DIFFERENT RING COMPONENTS ONE BEING A POLYCYCLIC RING
    SYSTEM (the Blue Book): ``7'-azadispiro[fluorene-9,2'-bicyclo[2.2.1]heptane-5',1''-indene] (PIN)``,
    the spiro atom shared with the first cited component takes 2'. The two numberings of the bicycloheptane
    component put the two spiro atoms at {2,5} and {5,2}: they are not an automorphism apart, and none of
    (c) to (j) ranks them, so the canonical ranks must not decide between them (they chose 5',2'). The name
    is the heteroatom-in-the-component form, not the PIN, and it reads back (OPSIN 2.9.0)."""
    smiles = "C1=CC2(CC3NC2CC32c3ccccc3-c3ccccc32)c2ccccc21"
    names = {be(s)["name"] for s in [smiles] + spellings(smiles, 8)}
    assert names == {"dispiro[fluorene-9,2'-7-azabicyclo[2.2.1]heptane-5',1''-indene]"}, names
