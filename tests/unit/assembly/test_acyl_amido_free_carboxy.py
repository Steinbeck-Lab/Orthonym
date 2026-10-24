"""PB lane S, task S3: the amido prefix of an acyl branch that carries a second COOH.

An N-terminal Asp or Glu on a peptide hangs on the C-terminal residue as ``-NH-CO-CH(NH2)-CH2-COOH``.
The ordinary route names the ACID ``HO-CO-R`` and turns '...oic acid' into '...amido'; for this acyl R
holds another COOH, the acid is a diacid ('2-aminobutanedioic acid') and ``acid_name_to_amido_prefix``
declines on purpose, so the peptide was declined at the default tier ('aspartylglutamic acid' at best
effort only). Rules cited:

* "Substituents of the types -NH-CO-R and -NH-SO2-R" (the Blue Book): "Method (1)
  generates preferred IUPAC names", the amide name with its final 'e' changed to 'o'.
* "Substituent groups with the free valence on a nitrogen atom" (the Blue Book), example
  ``-HN-CO-[CH2]2-CH(NH2)-COOH 4-amino-4-carboxybutanamido`` (:54470): the second acid is 'carboxy' on
  the chain that ends at the amide carbon.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly import acyl_amido_polyacid as polyacid
from orthonym.assembly.substituent_naming import acid_name_to_amido_prefix

ASP_GLU = "N[C@@H](CC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O"

# (smiles, PIN) -- each name read back by OPSIN 2.9.0 to the input's full InChIKey (java run outside
# the engine), the old answer being the retained peptide name or a decline.
ROWS = [
    (ASP_GLU, "(2S)-2-[(2S)-2-amino-3-carboxypropanamido]pentanedioic acid"),
    ("N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",
     "(2S)-2-[(2S)-2-amino-4-carboxybutanamido]butanedioic acid"),
    ("CC[C@H](C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](CCC(N)=O)C(=O)O",
     "(2S)-5-amino-2-{(2S,3S)-2-[(2S)-2-amino-3-carboxypropanamido]-3-methylpentanamido}-5-oxopentanoic acid"),
    # the descriptors come from the original molecule: D-Asp and D-Glu keep their own letters
    ("N[C@H](CC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
     "(2S)-2-[(2R)-2-amino-3-carboxypropanamido]pentanedioic acid"),
    ("N[C@@H](CC(=O)O)C(=O)N[C@H](CCC(=O)O)C(=O)O",
     "(2R)-2-[(2S)-2-amino-3-carboxypropanamido]pentanedioic acid"),
    # no stereo given: none cited
    ("OC(=O)CC(N)C(=O)NC(CCC(=O)O)C(=O)O",
     "2-(2-amino-3-carboxypropanamido)pentanedioic acid"),
]


@pytest.mark.unit
def test_the_ordinary_conversion_still_declines_a_diacid():
    """The decline is deliberate (one amido prefix cannot be read off a diacid name); S3 does not lift it."""
    assert acid_name_to_amido_prefix("(2S)-2-aminobutanedioic acid") is None
    assert acid_name_to_amido_prefix("pentanedioic acid") is None
    assert acid_name_to_amido_prefix("aminoacetic acid") == "2-aminoacetamido"


def _asp_acyl():
    mol = Chem.MolFromSmiles(ASP_GLU)
    # atoms: 0 N, 1 CH, 2 CH2, 3-5 COOH, 6 amide C, 7 O, 8 amide N
    return mol, 8, 6, set(range(0, 9))


@pytest.mark.unit
def test_polyacid_prefix_of_the_aspartyl_acyl():
    mol, n, c, atoms = _asp_acyl()
    assert polyacid.polyacid_acyl_amido_prefix(mol, n, c, atoms) == "(2S)-2-amino-3-carboxypropanamido"


@pytest.mark.unit
def test_an_acyl_without_a_second_acid_is_not_this_modules():
    mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)O")  # glycyl: 0 N, 1 CH2, 2 C(=O), 3 O, 4 N
    assert polyacid.polyacid_acyl_amido_prefix(mol, 4, 2, {0, 1, 2, 3, 4}) is None


@pytest.mark.unit
def test_the_acyl_names_non_pin_status_carries_to_the_prefix():
    """record_prefix_from_acid_status does this on the ordinary route; the polyacid route must too."""
    from orthonym.metrics import provenance as pv
    mol, n, c, atoms = _asp_acyl()
    token = pv._NON_PIN_FRAGMENTS.set(())
    try:
        assert polyacid.polyacid_acyl_amido_prefix(mol, n, c, atoms)
        assert pv._NON_PIN_FRAGMENTS.get() == ()      # nothing recorded: the prefix stays clean
        pv.record_non_pin_fragment("3-carboxypropanoyl")  # a non-PIN fragment of the acyl name
        amido = polyacid.polyacid_acyl_amido_prefix(mol, n, c, atoms)
        assert amido == "(2S)-2-amino-3-carboxypropanamido"
        assert amido in pv._NON_PIN_FRAGMENTS.get()
    finally:
        pv._NON_PIN_FRAGMENTS.reset(token)


@pytest.mark.unit
def test_cip_labels_come_from_the_project_cip_path(monkeypatch):
    """Raw rdCIPLabeler is off the naming path (the contributor guide.local): both molecules are labelled by
    perception.stereo.assign_stereochemistry."""
    from orthonym.perception import stereo
    seen = []
    real = stereo.assign_stereochemistry
    monkeypatch.setattr(stereo, "assign_stereochemistry", lambda m: (seen.append(m), real(m))[1])
    mol, n, c, atoms = _asp_acyl()
    assert polyacid.polyacid_acyl_amido_prefix(mol, n, c, atoms)
    assert len(seen) == 2  # the original (copy) and the stand-in (copy)


@pytest.mark.unit
def test_a_stand_in_that_changes_a_cip_label_is_refused(monkeypatch):
    mol, n, c, atoms = _asp_acyl()
    calls = []

    def fake_labels(m, atoms_):
        calls.append(m)
        return {i: ("S" if len(calls) == 1 else "R") for i in atoms_}

    monkeypatch.setattr(polyacid, "_cip_labels", fake_labels)
    assert polyacid.polyacid_acyl_amido_prefix(mol, n, c, atoms) is None


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,pin", ROWS)
def test_default_tier_names_the_peptide_by_its_pin(smiles, pin):
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] == "pin_verified", row
    assert row["name"] == pin
    assert "aspartyl" not in row["name"] and "glutamyl" not in row["name"]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [
    # ('CC(=S)c1ccc(C(=O)O)cc1' left this list: its PIN '4-(ethanethioyl)benzoic acid', the Blue Book, is
    # built by the strict path, test_pin_class_t12_acid_suffixes.py)
    "O=[N+]([O-])c1cnc2cc(Sc3ccc4c([N+](=O)[O-])c(Cl)cnc4c3)ccc2c1Cl",
    "N#CS(=O)(=O)C1CCCCC1C(=O)O",
    "CCC(=O)C1CCCCPC1=O",
])
def test_negative_controls_stay_below_pin_verified(smiles):
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"
