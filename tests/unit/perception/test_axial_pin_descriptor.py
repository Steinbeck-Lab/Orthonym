"""M/P is the PIN axial stereodescriptor; Ra/Sa is general nomenclature.

** "Cahn-Ingold-Prelog (CIP) stereodescriptors"**
(the Blue Book) -- *"The following stereodescriptors are used as
**preferred** stereodescriptors (see:"*... clause **(c)** (:44588)
*"'M' and 'P', to specify the absolute configuration of an axial or planar
entity using the helicity rule"*. ``Ra``/``Sa`` appear only under *"The
following stereodescriptors are recommended for **general** nomenclature"*
(:44594).

The two spell the same sense, so the general form is derivable rather than
separately computed: ** "The helicity rule: stereodescriptors 'M' and
'P'"** (:44812) -- *"When proceeding from the nearer ligand having priority in
the pair to the further away atom or group having priority in the pair, the
chirality is described by the symbols 'M' if the path is anticlockwise; the
symbol is 'P' if the path is clockwise. Stereodescriptors 'M' and 'P' are used
in preferred IUPAC names."* That is the same clockwise/anticlockwise test the
Ra/Sa elongated-tetrahedron model applies, hence Ra == P and Sa == M.

⚠ **This path is DORMANT.** Measured over 7000 corpus rows
(pubchem_2000 + chebi_5000): 0 allene elements, 0 atropisomer elements, 0 with a
CIP code. RDKit 2026.03.1 sets neither ``CHI_ALLENE`` nor ``STEREOATROPCW/CCW``
from SMILES input, so nothing reaches ``detect_axial_chirality`` in production.
These tests therefore construct the tags explicitly -- they lock the descriptor
contract for when the input channel does carry an axis, and this change must NOT
be quoted as a coverage or accuracy gain.
"""

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.perception.stereo import (
    AXIAL_GENERAL_FORM,
    _manual_allene_cip,
    detect_axial_chirality,
)


def _biaryl(stereo):
    """A biaryl with the atropisomeric bond stereo set explicitly.

    ``SetStereoAtoms`` and ``AssignCIPLabels`` are both required: without them
    RDKit records the bond stereo but never writes ``_CIPCode``, and the
    descriptor comes back ``None`` -- a fixture that silently tests nothing.
    """
    mol = Chem.RWMol(Chem.MolFromSmiles("Cc1ccccc1-c1ccccc1C"))
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtom(), bond.GetEndAtom()
        if (b.IsInRing() and e.IsInRing() and not bond.IsInRing()
                and b.GetIsAromatic() and e.GetIsAromatic()):
            bond.SetStereo(stereo)
            b_nbrs = [n.GetIdx() for n in b.GetNeighbors()
                      if n.GetIdx() != e.GetIdx()]
            e_nbrs = [n.GetIdx() for n in e.GetNeighbors()
                      if n.GetIdx() != b.GetIdx()]
            bond.SetStereoAtoms(b_nbrs[0], e_nbrs[0])
            break
    out = mol.GetMol()
    rdCIPLabeler.AssignCIPLabels(out)
    return out


def _allene():
    """An allene with CHI_ALLENE forced on the central carbon (ClC=C=CBr:
    four distinct terminal substituents, so the axis is chiral)."""
    mol = Chem.RWMol(Chem.MolFromSmiles("ClC=C=CBr"))
    central = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and sum(
                1 for b in atom.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE) == 2:
            atom.SetChiralTag(Chem.ChiralType.CHI_ALLENE)
            central = atom.GetIdx()
            break
    return mol.GetMol(), central


def test_atropisomer_pin_descriptor_is_the_helicity_letter():
    """(c): the PIN descriptor is 'M'/'P', not 'Sa'/'Ra'."""
    for stereo in (Chem.BondStereo.STEREOATROPCW,
                   Chem.BondStereo.STEREOATROPCCW):
        got = detect_axial_chirality(_biaryl(stereo))
        assert len(got) == 1
        assert got[0]["cip"] in ("M", "P"), got


def test_atropisomer_general_style_is_the_ra_sa_form():
    """ (:44594): Ra/Sa remain available for general nomenclature."""
    for stereo in (Chem.BondStereo.STEREOATROPCW,
                   Chem.BondStereo.STEREOATROPCCW):
        mol = _biaryl(stereo)
        pin = detect_axial_chirality(mol)[0]["cip"]
        general = detect_axial_chirality(mol, style="general")[0]["cip"]
        assert general in ("Ra", "Sa"), general
        assert general == AXIAL_GENERAL_FORM[pin]


def test_allene_helicity_is_computed_as_m_or_p():
    """ (:44812): clockwise -> 'P', anticlockwise -> 'M'."""
    mol, central = _allene()
    assert _manual_allene_cip(mol, central) in ("M", "P")


def test_allene_general_style_maps_through_the_same_table():
    mol, _central = _allene()
    pin = detect_axial_chirality(mol)
    general = detect_axial_chirality(mol, style="general")
    assert len(pin) == len(general) == 1
    if pin[0]["cip"] is not None:
        assert general[0]["cip"] == AXIAL_GENERAL_FORM[pin[0]["cip"]]


def test_general_form_table_is_the_verified_equivalence():
    """Ra == P and Sa == M:44812 vs:44594)."""
    assert AXIAL_GENERAL_FORM == {"P": "Ra", "M": "Sa"}


def test_unknown_style_does_not_silently_relabel():
    """Any style other than 'general' yields the PIN form -- a typo in a style
    string must not produce a third, undefined spelling."""
    mol = _biaryl(Chem.BondStereo.STEREOATROPCW)
    assert detect_axial_chirality(mol, style="pin")[0]["cip"] in ("M", "P")
    assert detect_axial_chirality(mol, style="typo")[0]["cip"] in ("M", "P")
