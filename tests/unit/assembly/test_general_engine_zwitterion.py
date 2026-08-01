"""P-74.1.1 / P-74.1.2 — zwitterionic ionic centres in the general-engine parent.

R7 (v29 residue). A mesoionic zwitterion was emitted by the best-effort tier as a
NEUTRAL name: the ring ``[N+]`` was spelled ``aza`` like any neutral ring N, and
the ``[O-]`` was spelled as a neutral ``oxo`` prefix. The result described a
molecule that cannot exist (OPSIN: "Atom is in unphysical valency state!
Element: C valency: 5") and was one of the remaining T6 emitted-but-unparseable
rows.

Blue Book, ``BlueBookV2/BlueBookV2.md``:

* **P-74.1.1 "Ionic centers in the same parent structure"** (heading ``:42415``),
  decisive sentence ``:42421``: *"For nomenclature purposes, zwitterionic
  compounds having the ionic centers in the same parent structure are not
  considered as neutral compounds."*  Construction, ``:42417``: *"…may be named
  by combining appropriate cumulative suffixes at the end of the name of a parent
  hydride in the order 'ium', 'ylium', 'ide', 'uide'. … anionic suffixes are
  cited after cationic suffixes … The final letter 'e' of the name of a parent
  hydride … is elided before the letter 'i' or 'y', or before a cumulative
  suffix beginning with a vowel."*
* **P-74.1.2** (heading ``:42462``), ``:42463``: *"In names, cationic suffixes are
  cited before anionic suffixes."*  Governing (PIN) example ``:42456``:
  ``1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate``.
* The Blue Book's own worked example of exactly this defect, ``:42441``:
  ``2-methyl-4-oxo-3,4-dihydro-1H-2-benzoselenopyran-2-ium-3-ide (PIN)``
  *(not ``2-methyl-3,4-dihydro-1H-2-benzoselenopyran-2-ium-3-id-4-one``)*.

The fix widens the EXISTING v26-P5 charge-suffix layer
(``general_engine._charge_suffix_text`` / ``_append_charge_suffix``), which was
scoped to NET molecular charge and therefore never saw a zwitterion (net 0).
"""

import pytest
from rdkit import Chem, RDLogger

from orthonym.assembly import general_engine as ge
from orthonym.namer import Orthonym

RDLogger.DisableLog("rdApp.*")


# The R7 row. Mesoionic: ring [N+] and an [O-] on a ring carbon.
MESOIONIC = "CC1=N[N+]2=CC=CC=C2C(=N1)[O-]"

# Coordinator-verified, re-verified in this session: fed to opsin-cli-2.9.0 the
# returned structure has InChIKey YZYLEZPSMVVLED-UHFFFAOYSA-N, identical to
# MESOIONIC's InChIKey.
TARGET = ("4-methyl-3,5,6-triazabicyclo[4.4.0]deca-1(10),2,4,6,8-"
          "pentaen-6-ium-2-olate")


def _name(smiles, tier="best-effort"):
    """Name ``smiles`` at ``tier`` using the CLI's own tier->flag contract."""
    from orthonym.cli import _emit_tier_flags

    return Orthonym(style="pin", **_emit_tier_flags(tier)).name(smiles)


# --------------------------------------------------------------------------
# The target row
# --------------------------------------------------------------------------

def test_mesoionic_zwitterion_emits_cumulative_ium_olate():
    """P-74.1.1: the cumulative ``-6-ium-2-olate``, not a neutral ``2-oxo``."""
    assert _name(MESOIONIC) == TARGET


def test_mesoionic_zwitterion_no_longer_emits_neutral_oxo():
    """The specific wrong rendering must be gone (P-74.1.1 :42421)."""
    name = _name(MESOIONIC)
    assert "oxo" not in name
    assert name.endswith("-olate")


def test_mesoionic_cationic_suffix_precedes_anionic():
    """P-74.1.2 :42463 — cationic suffixes are cited before anionic ones."""
    name = _name(MESOIONIC)
    assert name.index("-ium") < name.index("-olate")


def test_mesoionic_parent_hydride_e_is_elided():
    """P-74.1.1 :42417 — final 'e' elided before a suffix beginning 'i'."""
    name = _name(MESOIONIC)
    assert "pentaen-6-ium" in name
    assert "pentaene-6-ium" not in name


def test_mesoionic_pin_tier_unchanged():
    """No PIN byte may move: the PIN tier abstained before and must still."""
    assert _name(MESOIONIC, "pin") == "unknown organic compound"


# --------------------------------------------------------------------------
# The plan primitive, directly
# --------------------------------------------------------------------------

def _plan_for(smiles):
    """Build the parent numbering the ring tail would use, then plan."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def test_plan_declines_when_cation_is_off_parent():
    """A cation that is not a numbered parent atom is not a parent suffix."""
    mol = Chem.MolFromSmiles("C[N+](C)(C)CC(=O)[O-]")  # betaine
    # Pretend a parent numbering that excludes the cationic N entirely.
    atom_to_locant = {5: 1, 6: 2}
    assert ge._zwitterion_suffix_plan(mol, atom_to_locant) is None


def test_plan_declines_on_net_charged_molecule():
    """Net-charged species belong to the existing net-charge suffix path."""
    mol = Chem.MolFromSmiles("C[N+](C)(C)C")
    atom_to_locant = {1: 1}
    assert ge._zwitterion_suffix_plan(mol, atom_to_locant) is None


def test_plan_declines_on_internal_p59_charges():
    """P-59 nitro/azide/N-oxide charges are not ionic centres (get_ion_sites)."""
    for smi in ("C[N+](=O)[O-]", "[N-]=[N+]=NC", "C[N+]([O-])(C)C"):
        mol = Chem.MolFromSmiles(smi)
        a2l = {a.GetIdx(): a.GetIdx() + 1 for a in mol.GetAtoms()}
        assert ge._zwitterion_suffix_plan(mol, a2l) is None, smi


def test_plan_holds_out_the_anionic_oxygen():
    """The olate oxygen must be held out of substituent discovery so it is not
    ALSO emitted as a neutral ``oxo``/``hydroxy`` prefix (no double-count)."""
    mol = Chem.MolFromSmiles(MESOIONIC)
    # atom 3 = ring [N+], atom 9 = the ring C bearing the [O-] (atom 11).
    atom_to_locant = {i: i for i in range(len(mol.GetAtoms())) if i != 11}
    plan = ge._zwitterion_suffix_plan(mol, atom_to_locant)
    assert plan is not None
    held, text = plan
    assert 11 in held, "the anionic oxygen must be held out of discovery"
    assert text.endswith("olate")
    assert "ium" in text


# --------------------------------------------------------------------------
# Requirement 4 — probe EVERY category the new guard gates.
# A guard that silences a correct name is a regression even if R7 improves.
# --------------------------------------------------------------------------

# (smiles, expected name) captured at d193c723 BEFORE the fix, on both tiers.
UNCHANGED = [
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),   # quaternary ammonium
    ("CC(=O)[O-]", "acetate"),                          # carboxylate
    ("C[N+](=O)[O-]", "nitromethane"),                  # P-59 nitro
    ("C[N+]([O-])(C)C", "N,N-dimethylmethanamine N-oxide"),   # P-59 N-oxide
    ("[N-]=[N+]=NC", "azidomethane"),                   # P-59 azide
    ("c1ccccc1[N+]#N", "benzenediazonium"),             # diazonium
    ("CC(=O)O", "acetic acid"),
    ("c1ccccc1", "benzene"),
    ("CCO", "ethanol"),
    ("[NH3+]CC(=O)[O-]", "glycine"),                    # amino-acid zwitterion
    ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),      # betaine
    ("[O-]C(=O)c1ccc[nH+]c1", "pyridin-1-ium-3-carboxylate"),      # P-74.1.2
]


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_charged_species_unchanged_pin_tier(smiles, expected):
    assert _name(smiles, "pin") == expected


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_charged_species_unchanged_best_effort_tier(smiles, expected):
    assert _name(smiles, "best-effort") == expected


# --------------------------------------------------------------------------
# Fail closed: a charged molecule must never receive a NEUTRAL name.
# --------------------------------------------------------------------------

def test_semipolar_phosphoryl_oxide_keeps_its_neutral_name():
    """the contributor guide #9 regression guard, caught by measurement during R7.

    ``[PH+]...[O-]`` is the charge-separated depiction of a NEUTRAL ``P=O``.
    RDKit/InChI agree: ``CC1CO[PH+](C1)[O-]`` and OPSIN's parse of
    ``4-methyl-2-oxo-1,2-oxaphospholane`` (``CC1CP(OC1)=O``) share InChIKey
    ``DWXZAVJWXATXAG-UHFFFAOYSA-N``. A first cut of the fail-closed guard
    treated the pair as a zwitterion and destroyed this correct,
    round-tripping name.
    """
    assert _name("CC1CO[PH+](C1)[O-]") == "4-methyl-2-oxo-1,2-oxaphospholane"


def test_semipolar_pairs_are_not_ionic_centres():
    """The guard's own predicate, directly."""
    for smi in ("CC1CO[PH+](C1)[O-]", "C[N+](=O)[O-]", "C[N+]([O-])(C)C",
                "[O-][n+]1ccccc1", "CS(=O)C"):
        mol = Chem.MolFromSmiles(smi)
        assert ge._has_ionic_centres(mol) is False, smi


def test_genuine_zwitterion_is_an_ionic_centre():
    """The anion is on a ring CARBON, not on the cation -> not semipolar."""
    assert ge._has_ionic_centres(Chem.MolFromSmiles(MESOIONIC)) is True
    assert ge._has_ionic_centres(Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")) is True


def test_unexpressible_zwitterion_is_refused_not_neutralised():
    """the contributor guide #9 — verify what is EMITTED, not just that the bad path stopped.

    An ionic centre the parent cannot carry must abstain, never ship a neutral
    name that describes a different (uncharged) species.
    """
    # A sulfonium ylide: cation S+, anion carbanion, both off any ring parent.
    name = _name("C[S+](C)[CH2-]", "best-effort")
    assert "oxo" not in name
    # Whatever is emitted must not be the plain neutral hydride name.
    assert name not in ("dimethyl(methyl)sulfane", "trimethylsulfane")
