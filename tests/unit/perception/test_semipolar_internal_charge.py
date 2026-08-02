"""P-74.2.1 semipolar (dative) charges are INTERNAL, not ionic centres.

Blue Book, **P-74.2 "DIPOLAR COMPOUNDS"** (heading; ``BlueBookV2.md:42501``):

    "Dipolar compounds are electrically neutral molecules carrying a negative
    and a positive charge in at least one of their major canonical resonance
    structures. ... 1,2-Dipolar compounds have the opposite charges on adjacent
    atoms."

**P-74.2.1.1 "'Ylides'"** (heading, ``:42509``) states the depiction rule that
is the whole origin of this defect -- note the *last* clause, which is the
decisive one:

    "If 'X' is a saturated atom of an element from the second row of the
    periodic system, the 'ylide' is commonly represented by a charge-separated
    form; if 'X' is a third, fourth, etc. row element uncharged canonical forms
    are usually shown, RmX=YRn."

So a second-row cation (N) is *always* drawn charge-separated -- and the module's
hard-coded P-59 SMARTS cover exactly that case (nitro, N-oxide, azide, diazo).
A third/fourth-row cation (P, S, As, Se, Sb, Te, I ...) is *usually* drawn
uncharged, which is precisely why nobody wrote a SMARTS for it -- but nothing
stops an input from being drawn charge-separated, and when it is, the pair used
to read as a genuine ionic centre.

**P-74.2.1.4 "Phosphine oxides and chalcogen analogues"** (heading, ``:43041``)
settles what such a pair means:

    "Phosphine oxides have the generic formula R3P+ -O- <-> R3P=O. Chalcogen
    analogues are phosphine sulfides, phosphine selenides, and phosphine
    telluride (where O is replaced by S, Se, and Te, respectively)."
    "Method (3) leads to preferred IUPAC names."   [(3) = l5-phosphanone]

The Blue Book's own double-headed arrow says the two depictions are one
compound, and it makes the NEUTRAL name (a l5-heterone) the PIN. Same verdict
for amine/imine oxides, **P-74.2.1.2** (heading, ``:43008``): "Method (2) leads
to preferred IUPAC names when one amine oxide is present. ... Hence,
zwitterionic compounds are never PINs".

THE BOUNDARY -- the Blue Book draws it by the ANION, and both sides matter:

  * anion on CARBON = an ylide. **P-74.2.1.1** ``:42513``: "Method (1) is
    applicable to all 'ylides' and leads to preferred IUPAC names", where
    method (1) is "as zwitterionic compounds". So an ylide's PIN *is* the
    zwitterion name and its charges must stay VISIBLE.
  * anion on NITROGEN with a nitrogen cation = an amine imide. **P-74.2.1.3**
    (heading, ``:43022``): "Method (1) leads to preferred IUPAC names", method
    (1) being "as a zwitterion based on hydrazine". Again: stay visible.
  * anion on a CHALCOGEN (O/S/Se/Te) = the oxide / chalcogenide class above.
    Neutral PIN, so the charges are internal.

Hence the mask keys on the anion being a terminal chalcogen. It deliberately
does NOT enumerate cation elements: the cation side is settled by a structure
proof (build the uncharged multiple-bond form, require it to sanitise and to
carry the SAME standard InChIKey), because an element list cannot tell a
semipolar oxide from a genuine oxoanion.

Regression anchor: ``CC1CO[PH+](C1)[O-]`` and OPSIN's ``CC1CP(OC1)=O`` share
InChIKey ``DWXZAVJWXATXAG-UHFFFAOYSA-N``; treating that pair as an ionic centre
once destroyed the correct name ``4-methyl-2-oxo-1,2-oxaphospholane``.
"""
import pytest
from rdkit import Chem

from orthonym.perception.ions import (
    _get_internal_charge_atoms,
    detect_species_type,
    get_ion_sites,
)


def _n_sites(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"test fixture does not parse: {smiles}"
    sites = get_ion_sites(mol)
    return len(sites["cations"]) + len(sites["anions"])


# ---------------------------------------------------------------------------
# The class that must be masked: P-74.2.1.2 / P-74.2.1.4 semipolar chalcogenides
# ---------------------------------------------------------------------------

SEMIPOLAR = [
    # (smiles, uncharged depiction, label)
    ("CC1CO[PH+](C1)[O-]", "CC1CP(OC1)=O", "cyclic P-oxide (the regression anchor)"),
    ("C[P+](C)(C)[O-]", "CP(C)(C)=O", "trimethylphosphine oxide"),
    ("C[As+](C)(C)[O-]", "C[As](C)(C)=O", "trimethylarsine oxide"),
    ("C[Sb+](C)(C)[O-]", "C[Sb](C)(C)=O", "trimethylstibine oxide"),
    ("C[S+](C)[O-]", "CS(C)=O", "dimethyl sulfoxide"),
    ("C[Se+](C)[O-]", "C[Se](C)=O", "dimethyl selenoxide"),
    ("C[P+](C)(C)[S-]", "CP(C)(C)=S", "phosphine sulfide (chalcogen analogue)"),
    ("c1ccccc1[I+][O-]", "c1ccccc1I=O", "iodosylbenzene"),
    ("C[S+2](C)([O-])[O-]", "CS(C)(=O)=O", "charge-separated sulfone (cation +2)"),
]


@pytest.mark.parametrize("smiles,neutral,label", SEMIPOLAR, ids=[c[2] for c in SEMIPOLAR])
def test_semipolar_chalcogenide_reports_no_ionic_site(smiles, neutral, label):
    """A P-74.2.1.4 semipolar X(+)-A(-) pair is NOT an ionic centre."""
    assert _n_sites(smiles) == 0, f"{label}: semipolar pair leaked as an ionic site"


@pytest.mark.parametrize("smiles,neutral,label", SEMIPOLAR, ids=[c[2] for c in SEMIPOLAR])
def test_semipolar_chalcogenide_atoms_are_marked_internal(smiles, neutral, label):
    """Both atoms of the pair carry INTERNAL charges."""
    mol = Chem.MolFromSmiles(smiles)
    internal = _get_internal_charge_atoms(mol)
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0}
    assert charged, f"{label}: fixture carries no formal charge -- test is vacuous"
    assert charged <= internal, f"{label}: {sorted(charged - internal)} not masked"


@pytest.mark.parametrize("smiles,neutral,label", SEMIPOLAR, ids=[c[2] for c in SEMIPOLAR])
def test_semipolar_chalcogenide_is_the_same_species_as_its_uncharged_form(
    smiles, neutral, label
):
    """The premise of the mask, stated as an executable claim (P-74.2.1.4 '<->').

    If this ever fails, the mask is unsound for that row and must not fire.
    """
    a = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    b = Chem.MolToInchiKey(Chem.MolFromSmiles(neutral))
    assert a and a == b, f"{label}: {smiles} and {neutral} are different species"


@pytest.mark.parametrize("smiles,neutral,label", SEMIPOLAR, ids=[c[2] for c in SEMIPOLAR])
def test_semipolar_chalcogenide_is_a_neutral_species(smiles, neutral, label):
    mol = Chem.MolFromSmiles(smiles)
    assert detect_species_type(mol) == "neutral", label


# ---------------------------------------------------------------------------
# The boundary: these carry GENUINE charges and must keep them
# ---------------------------------------------------------------------------

GENUINE = [
    # P-74.2.1.1 ylides -- zwitterion name IS the PIN, charges must stay visible
    ("C[P+](C)(C)[CH2-]", "phosphorus ylide (P-74.2.1.1, zwitterion PIN)"),
    ("C[S+](C)[CH2-]", "sulfur ylide (P-74.2.1.1, zwitterion PIN)"),
    ("C[N+](C)(C)[CH2-]", "nitrogen ylide (P-74.2.1.1.1, zwitterion PIN)"),
    # P-74.2.1.3 amine imide -- zwitterion PIN
    ("C[N+](C)(C)[N-]C", "amine imide (P-74.2.1.3, zwitterion PIN)"),
    # genuine ions / zwitterions
    ("CC(=O)[O-]", "acetate anion"),
    ("[NH3+]CC(=O)[O-]", "glycine zwitterion"),
    ("C[N+](C)(C)CC(=O)[O-]", "betaine (1,3, charges not adjacent)"),
    ("[O-]C(=O)c1ccc[nH+]c1", "pyridinium-3-carboxylate"),
    ("C[N+](C)(C)C", "tetramethylazanium"),
    # oxoanions: adjacent X(+)-O(-) but the charges do NOT balance locally
    ("[O-][I+]([O-])(O)(O)(O)O", "iodine oxoanion: two O- on one +1 cation"),
    ("[O-][Cl+3]([O-])([O-])[O-]", "perchlorate: four O- on a +3 cation"),
]


@pytest.mark.parametrize("smiles,label", GENUINE, ids=[c[1] for c in GENUINE])
def test_genuine_charges_are_not_masked(smiles, label):
    """The mask must not swallow a real ionic centre (the contributor guide #9)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    internal = _get_internal_charge_atoms(mol)
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0}
    assert charged, f"{label}: fixture carries no charge -- test is vacuous"
    assert not charged <= internal, f"{label}: a genuine ionic centre was masked"


# ---------------------------------------------------------------------------
# The pre-existing P-59 Table 5.1 mask must keep working (no unmasking)
# ---------------------------------------------------------------------------

TABLE_5_1 = [
    ("C[N+](=O)[O-]", "nitro"),
    ("[O-][n+]1ccccc1", "aromatic N-oxide"),
    ("C[N+](C)(C)[O-]", "aliphatic amine oxide"),
    ("[N-]=[N+]=NC", "organic azide"),
    ("[CH-]=[N+]=[N-]", "diazo"),
]


@pytest.mark.parametrize("smiles,label", TABLE_5_1, ids=[c[1] for c in TABLE_5_1])
def test_table_5_1_groups_remain_internal(smiles, label):
    assert _n_sites(smiles) == 0, f"{label}: regressed to an ionic site"


# ---------------------------------------------------------------------------
# The retired general_engine workaround must not come back
# ---------------------------------------------------------------------------


def test_general_engine_reads_the_same_perception():
    """``_genuine_ion_sites`` must delegate, not re-derive.

    It used to subtract semipolar pairs itself from a cation ELEMENT LIST. That
    list cannot tell a semipolar oxide from a genuine oxoanion, and these two
    real corpus rows are what it got wrong: both are iodine oxoanions (two O-
    on a +1 iodine, net charge -1) whose charges it erased.
    """
    from orthonym.assembly.general_engine import _genuine_ion_sites

    for smiles in ("[O-][I+]([O-])(O)(O)(O)O", "[O-][I+]([O-])([O-])(O)(O)O"):
        mol = Chem.MolFromSmiles(smiles)
        assert Chem.GetFormalCharge(mol) != 0, f"fixture is not an ion: {smiles}"
        cations, anions = _genuine_ion_sites(mol)
        assert cations and anions, (
            f"{smiles}: a genuine oxoanion was flattened to 'no ionic centre'"
        )


def test_semipolar_pair_still_hidden_from_the_general_engine():
    """...while the pair the workaround existed for stays hidden (the contributor guide #9:
    removing a guard must not unmask what it was covering)."""
    from orthonym.assembly.general_engine import _genuine_ion_sites, _has_ionic_centres

    mol = Chem.MolFromSmiles("CC1CO[PH+](C1)[O-]")
    cations, anions = _genuine_ion_sites(mol)
    assert not cations and not anions
    assert not _has_ionic_centres(mol)


def test_chalcogen_analogues_the_workaround_missed_are_now_covered():
    """The old oxide-only list left these leaking as ionic centres.

    ``get_ion_sites`` measured 2 sites for the phosphine sulfide and 3 for the
    charge-separated sulfone before this fix (its cation is +2, and the list
    required exactly +1).
    """
    from orthonym.assembly.general_engine import _has_ionic_centres

    for smiles in ("C[P+](C)(C)[S-]", "C[S+2](C)([O-])[O-]", "C[Se+](C)[O-]"):
        mol = Chem.MolFromSmiles(smiles)
        assert not _has_ionic_centres(mol), smiles
