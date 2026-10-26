"""Universal coverage-floor: a hypervalent Cl/Br/I atom is a one-atom parent hydride or a substituent
root with the lambda convention (UN lane H, ``assembly/universal_halane.py``).

The floor refused any spine with a non-skeletal atom (``_chain_spine_from_path``): a halogen with a
bonding number other than 1 matches no leaf, so a ``C=I`` ylidene or ``-I(OH)-`` unit used to void the
whole molecule (it must never be threaded into a chain and dropped: a phantom ``methane``). Those
molecules now have a name that OPSIN 2.9.0 reads back to the input's full InChIKey, and the spine guard
that fails closed for every shape this lane does not name is unchanged.

Rules (the Blue Book, quoted with their section headings):

* "Nomenclature based on halogen parent hydrides" (:39660): "For cyclic and acyclic parent
  hydrides the lambda convention is used to indicate nonstandard bonding numbers." 'CH3-ICl2
  dichloro(methyl)-lambda3-iodane (PIN)' (:39668), '(CH3-CO-O)2I- bis(acetyloxy)-lambda3-iodanyl
  (preferred prefix)' (:39670).
* "Seniority order for classes", Table 4.1 (:18158): class 39 "lambda7, lambda5 and lambda3 halogen
  compounds in the order F > Cl > Br > I" (:18215) is senior to class 40 "Carbon compounds: rings,
  chains": the only seniority applied. Classes 1-38 (a principal characteristic group, a heteroatom
  hydride N > P >... > Te, are not checked: the floor writes no suffix, so these names are
  systematic, never a PIN (a lead).
* "The locant '1' is omitted: (a) in substituted mononuclear parent hydrides" (:2891): the
  prefixes of the one-atom parent carry no locant.
* "Anions formed by addition of hydride ions" (:41089): 'diphenyliodanuide (PIN)
  diphenyl-lambda3-iodanide' (:41110), method (2) ("by adding the suffix 'ide' to a parent hydride in
  which the bonding number at the anionic site is higher than the standard bonding number").
* "General rule for systematically naming cationic centers in parent hydrides" (:41366): the
  cation is "named by replacing the final letter 'e' of the parent hydride name, if any, by the suffix
  'ium'"; 'diphenyliodanium (PIN)' (:41362).

Targeted-file run only:
    .venv/bin/python -m pytest tests/unit/assembly/test_universal_halane.py -q
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.assembly import universal_halane as uh
from orthonym.assembly import universal_substituent as us
from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.rules.lambda_convention import LAMBDA
from orthonym.validation.reconstruct import verify_or_none
from tests.support.rt_assert import assert_rt_exact, name_best_effort


def _heavy(smiles):
    m = Chem.MolFromSmiles(smiles)
    return frozenset(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1)


def _floor(smiles):
    r = name_universal_substitutive(Chem.MolFromSmiles(smiles))
    assert r is not None, f"expected a name for {smiles!r}, got None"
    assert r.covers == _heavy(smiles), f"coverage gap for {smiles!r}"
    return r.name


# ---------------------------------------------------------------------------
# The halogen is the parent hydride of the molecule class 39 > class 40)
# ---------------------------------------------------------------------------

PARENTS = [
    #: the one-atom parent, prefixes without locants (a)), the second and
    # further enclosed and cited in the order
    ("CIO", "hydroxy(methyl)-λ3-iodane"),
    ("CI(Cl)Cl", "dichloro(methyl)-λ3-iodane"),                       #:39668 (PIN)
    ("OI(O)c1ccccc1", "dihydroxy(phenyl)-λ3-iodane"),
    ("CC(=O)OI(OC(C)=O)c1ccccc1", "bis(acetyloxy)(phenyl)-λ3-iodane"),
    ("C[IH]C", "dimethyl-λ3-iodane"),
    ("c1ccccc1I=O", "oxo(phenyl)-λ3-iodane"),
    ("C=Ic1ccccc1", "methylidene(phenyl)-λ3-iodane"),                 # the ylide: a '-ylidene' prefix
    ("CI(=O)=O", "methyldi(oxo)-λ5-iodane"),
    ("FI(F)(F)(F)F", "pentafluoro-λ5-iodane"),
    ("CC[I](Br)Cl", "bromo(chloro)(ethyl)-λ3-iodane"),                # alphanumerical order
    # RDKit draws Cl and Br of bonding number 3, 5, 7 charge-separated ('Cl(=O)(=O)=O' is
    # '[Cl+3]([O-])([O-])[O-]'): the semipolar drawing of the same oxo groups
    ("CCCCOCl(=O)(=O)=O", "butoxytri(oxo)-λ7-chlorane"),
    ("CCCCOCl(=O)=O", "butoxydi(oxo)-λ5-chlorane"),
    ("CO[Br+2]([O-])[O-]", "methoxydi(oxo)-λ5-bromane"),
    ("OCl(=O)(=O)=O", "hydroxytri(oxo)-λ7-chlorane"),
]


@pytest.mark.parametrize("smiles,expected", PARENTS)
def test_halogen_parent_hydride_name_reads_back(smiles, expected):
    name = _floor(smiles)
    assert name == expected
    assert verify_or_none(name, smiles) == name, f"{name!r} did not round-trip for {smiles!r}"


# ---------------------------------------------------------------------------
# The halogen is the root of a substituent: '-lambda3-iodanyl', method (2) of
# ---------------------------------------------------------------------------

ROOTS = [
    # a second hypervalent halogen is the root of a prefix of the first (the senior one by the
    # canonical rank: they are alike, the choice only has to be the same for every numbering)
    ("ClI(Cl)c1ccc(cc1)I(Cl)Cl", "dichloro[4-(dichloro-λ3-iodanyl)phenyl]-λ3-iodane"),
    ("ClI(Cl)OI(Cl)Cl", "dichloro[(dichloro-λ3-iodanyl)oxy]-λ3-iodane"),
]


@pytest.mark.parametrize("smiles,expected", ROOTS)
def test_halogen_substituent_root_reads_back(smiles, expected):
    name = _floor(smiles)
    assert name == expected
    assert verify_or_none(name, smiles) == name


def test_the_senior_halogen_is_the_parent_whatever_the_atom_numbering():
    """ class 39 'in the order F > Cl > Br > I': chlorine before iodine; the same
    molecule drawn from either end gives the same name (determinism)."""
    names = {_floor(s) for s in ("ClI(Cl)OI(Br)Br", "BrI(Br)OI(Cl)Cl")}
    assert len(names) == 1


@pytest.mark.parametrize("pair", [("CIO", "OIC"), ("CI(Cl)Cl", "ClI(Cl)C"),
                                  ("CC(=O)OI(OC(C)=O)c1ccccc1", "c1ccccc1I(OC(C)=O)OC(C)=O")])
def test_name_is_the_same_for_every_atom_numbering(pair):
    assert _floor(pair[0]) == _floor(pair[1])


# ---------------------------------------------------------------------------
# Anions and cations of one-atom halogen hydrides
# ---------------------------------------------------------------------------

IONS = [
    # method (2): OPSIN 2.9.0 has no grammar for the substituted '-uide' of method (1)
    # ('diphenyliodanuide', validation.uide_twin), and reads 'diphenyl-lambda3-iodanide'
    ("CO[I-]C", "methoxy(methyl)-λ3-iodanide"),
    ("COC(=O)[I-]C1CCCCC1", "cyclohexyl(methoxycarbonyl)-λ3-iodanide"),
    # 'diphenyliodanium (PIN)': the standard hydride with its '-ium', no lambda
    ("C[I+]c1ccccc1", "methyl(phenyl)iodanium"),
]


@pytest.mark.parametrize("smiles,expected", IONS)
def test_halogen_ion_name_reads_back(smiles, expected):
    name = _floor(smiles)
    assert name == expected
    assert verify_or_none(name, smiles) == name


# ---------------------------------------------------------------------------
# The perception: which atoms are a unit
# ---------------------------------------------------------------------------

def _unit(smiles, idx):
    return uh.halane_unit(Chem.MolFromSmiles(smiles), idx)


def test_unit_of_neutral_and_charge_separated_halogens():
    u = _unit("CIO", 1)
    assert (u.stem, u.lam, u.ending, u.ionic, u.oxides) == ("iodane", 3, "e", False, frozenset())
    u = _unit("CCCCOCl(=O)(=O)=O", 5)           # [Cl+3] with three [O-]: the uncharged drawing has 7 bonds
    assert (u.stem, u.lam, len(u.oxides)) == ("chlorane", 7, 3)
    u = _unit("CO[I-]C", 2)
    assert (u.lam, u.ending, u.ionic) == (3, "ide", True)       # standard 1 + 2, method (2)
    u = _unit("C[I+]C", 1)
    assert (u.lam, u.ending, u.ionic) == (None, "ium", True)


@pytest.mark.parametrize("smiles,idx", [
    ("CI", 1),                    # a monovalent halogen is a 'iodo' prefix, never a unit
    ("C1CCC[IH]C1", 4),           # a ring halogen has no one-atom name here
    ("C[I+](C)(C)C", 1),          # a charge that is neither an oxide pair nor an -ium of bonding number 2
    ("C[125I](C)C", 1),           # an isotope
])
def test_atoms_that_are_not_a_unit(smiles, idx):
    assert uh.halane_unit(Chem.MolFromSmiles(smiles), idx) is None


@pytest.mark.parametrize("smiles", [
    "N->I(C)C",           # a dative bond to the halogen counts in its bonding number, not in the name
    "C[I](C)<-O",         # the same, drawn the other way round, with an oxygen donor
    "CI(C)->[Cl]",        # a dative bond from the halogen
])
def test_halogen_with_a_dative_bond_is_not_a_unit_and_the_rung_declines(smiles):
    """``GetTotalValence`` counts a dative bond, so 'N->I(C)C' read as the bonding number 3 and was
    named 'aminodi(methyl)-lambda3-iodane': one hydrogen fewer than the input, a different molecule. A
    halogen with a bond other than single, double or triple is no unit, and the whole call voids."""
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    halogen = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() in uh.HALANE_STEM)
    assert uh.halane_unit(m, halogen) is None
    assert any(b.GetBondType() == Chem.BondType.DATIVE for b in m.GetBonds())
    assert name_universal_substitutive(m) is None


def test_candidate_filter_is_cheap_and_exact():
    assert uh.has_halane_candidate(Chem.MolFromSmiles("CIO"))
    assert uh.has_halane_candidate(Chem.MolFromSmiles("C[I-]C"))
    assert not uh.has_halane_candidate(Chem.MolFromSmiles("CCCCl"))
    assert not uh.has_halane_candidate(Chem.MolFromSmiles("c1ccccc1Br"))


# ---------------------------------------------------------------------------
# The phantom-carbon guard of the spine builders is unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", ["CIO", "CI(Cl)Cl", "C=Ic1ccccc1", "CCCCOCl(=O)(=O)=O"])
def test_floor_without_the_halane_rung_still_voids(smiles):
    """The hypervalent halogen reaches the spine allow-list guard of ``_chain_spine_from_path``
    exactly as before when the lambda forms are off (every spelling the floor gave earlier): the
    whole call voids; no name with a phantom carbon, no dropped halogen."""
    assert us._name_universal_substitutive_unsafe(Chem.MolFromSmiles(smiles), 20000) is None


def test_spine_builder_refuses_a_path_through_a_halogen():
    m = Chem.MolFromSmiles("CI(Cl)Cl")
    ctx, heavy = us._build_ctx(m, 20000)
    iodine = next(a.GetIdx() for a in ctx.mol.GetAtoms() if a.GetSymbol() == "I")
    assert us._chain_spine_from_path(ctx, heavy, None, [0, iodine]) is None
    assert us._chain_spine_from_path(ctx, heavy, iodine, [iodine]) is None


@pytest.mark.parametrize("smiles", [
    "C1CCC[Br+]C1",       # a halogen in a ring that ``ring_replacement`` does not admit (bromine): voids
    "C[I+](C)(C)C",       # a charged halogen of bonding number 4: no -ium of bonding number 2
])
def test_halogen_shapes_not_named_still_void_never_a_phantom_carbon(smiles):
    r = name_universal_substitutive(Chem.MolFromSmiles(smiles))
    assert r is None, f"{smiles!r} was named {r.name!r}"


def test_a_ring_iodine_is_named_with_its_morpheme_and_never_as_a_phantom_carbocycle():
    """Lane L3 admitted ring iodine to the ring 'a' table (``ring_replacement``): the floor's ring spine spells
    it 'ioda' with its lambda, the Blue Book;:9765), so the name counts and
    spells the atom (this SMILES used to void)."""
    smiles = "C1CCC[IH]C1"
    name = _floor(smiles)
    assert "ioda" in name and "\u03bb3" in name, name
    assert verify_or_none(name, smiles) == name


def test_halane_rung_is_not_reached_by_a_molecule_without_a_halogen_unit(monkeypatch):
    """The rung runs only after every earlier floor spelling declined, so it is not even called for a
    molecule they name (an ordinary halide): the floor's own names are not changed."""
    def boom(*a, **k):
        raise AssertionError("the halane rung ran for a molecule an earlier spelling names")
    monkeypatch.setattr(us, "_halane_rung", boom)
    assert "chloro" in _floor("CCCCl")
    assert "bromo" in _floor("c1ccccc1Br")


# ---------------------------------------------------------------------------
# The rung names the molecule the outermost name was asked for, not a structure derived from it
# ---------------------------------------------------------------------------

def test_rung_is_off_for_a_structure_a_nested_name_derives_from_the_input():
    """A nested ``name`` (the charged router's re-entry on the neutral skeleton of a cation, a cap)
    hands its name to the outer frame, whose tier label reads what it was built from: a floor name of
    the derived structure turned the best-effort row of an iodanium cation from pin_verified into
    systematic_verified. The molecule of the outermost frame (also named again by its clean
    fall-through) and a direct caller get the rung."""
    from orthonym.assembly import fragment_naming as fn
    given, derived = Chem.MolFromSmiles("CIO"), Chem.MolFromSmiles("CI(C)O")
    assert uh.names_the_input(given) and uh.names_the_input(derived)      # no name frame: a direct call
    fn.enter_name_scope("CIO")
    try:
        assert uh.names_the_input(given)
        assert name_universal_substitutive(given) is not None
        fn.enter_name_scope("CI(C)O")                                     # a nested frame names a derived one
        try:
            assert not uh.names_the_input(derived)
            assert name_universal_substitutive(derived) is None
        finally:
            fn.exit_name_scope()
    finally:
        fn.exit_name_scope()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CC1=CC=C(C=C1)C#C[I+]C2=CC=CC=C2", "[2-(4-methylphenyl)ethynyl](phenyl)iodanium"),
    ("C1=CC=C(C=C1)[I+]C#CC2=CSC=C2", "phenyl[2-(thiophen-3-yl)ethynyl]iodanium"),
])
def test_best_effort_keeps_the_pin_label_of_a_name_the_pin_pass_builds(smiles, name):
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


# ---------------------------------------------------------------------------
# The best-effort tier end to end (UN lane H study rows): a name that reads back to the full key. The
# OPSIN validity gate is ON (the suite turns it off): with it off the PIN path ships '1-ethenyl-4-
# iodobenzene' for the second row and an earlier general-engine candidate the iodanide row, names the
# gate withdraws in production
# ---------------------------------------------------------------------------

STUDY = [
    "CIO",                                                           # CID155731853
    "CC1(C2CCC1(C(=O)C2)CS(=O)(=O)OI(C3=CC=CC=C3)O)C",                # CID14486159, a sulfonyloxy iodane
    "CC1=CC=C(C=C1)S(=O)(=O)OI(C2=CC=C(C=C2)C=C)O",                  # CID101135120
    "CCCCOCl(=O)(=O)=O",                                             # a perchlorate ester
    "CS(C)(C1=C(C=CS1)[N+](=O)[O-])OCl(=O)(=O)=O",                   # CID129786858
    "COC(=O)C(=O)NC1=C(C=C(C=C1)C2CCC(CC2)[I-]C(=O)OC)[N+](=O)[O-]",  # CID163925522, an iodanide
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", STUDY)
def test_best_effort_names_the_study_row_and_it_reads_back_exactly(smiles):
    name = assert_rt_exact(smiles)
    assert LAMBDA in name
    assert name_best_effort(smiles)["tier"] in ("best_effort", "systematic_verified")
