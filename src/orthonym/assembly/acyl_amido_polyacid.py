"""The amido prefix of an acyl branch that carries a second carboxylic acid (Asp, Glu).

 method (1) (the Blue Book-32998) writes ``-NH-CO-R`` as the amide name with
its final 'e' changed to 'o'. ``acyl_amido_prefix_from_branch`` gets there by naming the ACID
``HO-CO-R`` and turning '...oic acid' into '...amido'; for an acyl whose R holds another COOH
that acid is a diacid ('2-aminobutanedioic acid'), one amido prefix cannot be read off it
(``acid_name_to_amido_prefix`` declines on purpose), and the whole peptide was declined at the
default tier. The Blue Book spells such an acyl with the second acid as 'carboxy' on the chain
that ends at the amide carbon: "Substituent groups with the free valence on a
nitrogen atom" (:54456), ``-HN-CO-[CH2]2-CH(NH2)-COOH`` is '4-amino-4-carboxybutanamido' (:54470).

The acyl namer already writes that form when the acyl hangs on a carbon: ``-CO-CH(NH2)-CH2-COOH``
on a carbon is '(2S)-2-amino-3-carboxypropanoyl'. So the amide nitrogen is replaced by a carbon
in a COPY of the molecule (atom indices are kept), the acyl atoms are named there, and the
'...oyl' word becomes '...amido'. Three checks keep the stand-in honest:

* the copy changes one atom, so the CIP labels of every stereocentre of the acyl are computed on
  both molecules by the project's CIP path (``assign_stereochemistry``) and must agree, otherwise the
  descriptors in the name would not be the original's;
* the name must end in 'oyl' and cite 'carboxy' (the shape this module exists for);
* the acyl name's non-PIN status carries to the prefix (``record_derived_non_pin_fragment``), as
  ``record_prefix_from_acid_status`` does on the ordinary route;
* when the caller asked for a verified prefix, the amide ``R-CO-NH2`` the prefix stands for
  ('...amide') must read back from OPSIN as that amide on the full InChIKey, as
  ``fragment_acid_name_verified`` demands of the acid name on the ordinary route.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

_FREE_ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')


def _cip_labels(mol, atoms: Iterable[int]) -> dict:
    """The project's CIP labels (``perception.stereo.assign_stereochemistry``: centres, then
    rdCIPLabeler, then the legacy labeller) of ``atoms`` on a fresh copy of ``mol``. The copy loses
    the 'already assigned' marker and every stale label first, so the stand-in is labelled from its
    own constitution and the same path labels both sides."""
    from ..perception.stereo import _CIP_ASSIGNED_PROP, assign_stereochemistry
    copy = Chem.Mol(mol)
    if copy.HasProp(_CIP_ASSIGNED_PROP):
        copy.ClearProp(_CIP_ASSIGNED_PROP)
    for atom in copy.GetAtoms():
        if atom.HasProp('_CIPCode'):
            atom.ClearProp('_CIPCode')
    assign_stereochemistry(copy)
    return {i: copy.GetAtomWithIdx(i).GetPropsAsDict().get('_CIPCode') for i in atoms}


def _amide_of_acyl(mol, n_idx: int, acyl, halogen: Optional[int] = None) -> Optional[str]:
    """SMILES of the primary amide the amido prefix stands for: the acyl atoms plus the amide N
    as NH2, every other atom removed. With ``halogen`` (an atomic number), the acyl halide of that
    element: the nitrogen is replaced by it."""
    rw = Chem.RWMol(mol)
    keep = set(acyl) | {n_idx}
    # the atoms that lose a neighbour take their hydrogens again; an atom that keeps its neighbours keeps its own
    # (the hydrogen of an indole [nH] cannot be recomputed: the ring does not kekulize without it)
    cut = {n.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep for n in a.GetNeighbors()} & keep
    if halogen is not None:
        rw.GetAtomWithIdx(n_idx).SetAtomicNum(halogen)
    for idx in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep), reverse=True):
        rw.RemoveAtom(idx)
    try:
        amide = rw.GetMol()
        survivors = sorted(keep)
        for new_idx, old_idx in enumerate(survivors):
            if old_idx in cut or old_idx == n_idx:
                atom = amide.GetAtomWithIdx(new_idx)
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(amide)
        return Chem.MolToSmiles(amide)
    except Exception:  # noqa: BLE001 - a molecule that will not sanitize is a decline
        return None


def ordinary_amido_is_the_acyl(mol, n_idx: int, carbonyl_c: int, sub_atoms, amido: str) -> bool:
    """False when the ordinary route's ``amido`` prefix (the acid name of the acyl with '...oic acid' turned into
    '...amido', ``substituent_naming.acid_name_to_amido_prefix``) is not the amide of THIS acyl.

    An acyl that holds a second free COOH has an acid name whose '...oic acid' may be that other group:
    '3-amino-4-{...}-4-oxobutanoic acid' names the Asp side-chain acid of an Asp-Leu-Met acyl whose own
    carboxyl is cited as '1-hydroxy-...-1-oxobutan-2-yl', and '...-4-oxobutanamido' is a prefix of another
    molecule. The amide the prefix stands for ('...amide', method (1), the Blue Book) must read
    back from OPSIN as the amide of the acyl on the full InChIKey, as ``polyacid_acyl_amido_prefix`` demands of its own
    prefix when it is asked for a verified one. An acyl with no second acid is the ordinary route's, unchecked here."""
    sub_set = set(sub_atoms)
    if not any(m[0] != carbonyl_c and m[0] in sub_set for m in mol.GetSubstructMatches(_FREE_ACID)):
        return True
    acyl = sorted(sub_set - {n_idx})
    amide_smiles = _amide_of_acyl(mol, n_idx, acyl)
    if not amide_smiles or not amido.endswith('o'):
        return False
    from .substituent_naming import fragment_acid_name_verified
    return bool(fragment_acid_name_verified(amido[:-1] + 'e', amide_smiles))


def acyl_name_is_the_acyl(mol, n_idx: int, carbonyl_c: int, sub_atoms, acid_name: str, acyl_name: str) -> bool:
    """False when ``acyl_name``, the acyl prefix of ``acid_name`` ('...oic acid' turned into '...oyl',
    "changing the ending 'oic acid' to 'oyl'", the Blue Book), is the acyl of another group than the one at
    ``carbonyl_c`` of the N-attached acyl ``sub_atoms``.

    An acyl that holds a second free COOH has an acid name whose '...oic acid' may be that other group: the acid
    '2-({[...]amino}oxy)-2-methylpropanoic acid' of 'C(=O)C(=NOC(C)(C)C(=O)O)Ar' cites the carboxyl of the acyl as
    'carboxy', and '...-2-methylpropanoyl' is the acyl of the other. The acyl chloride the name stands for
    ('...oyl chloride', must read back from OPSIN as the acyl chloride of the acyl on the full InChIKey.
    Unchecked (True): an acyl with no second acid (the ordinary route's), and an acid name that is a poly-acid or a
    functional replacement acid (``acid_name_to_amido_prefix`` has no spelling for it: its '...oyl' is a divalent
    group, which this single-acyl read-back does not describe; the validity gate rejects such a name when it is wrong)."""
    sub_set = set(sub_atoms)
    if not any(m[0] != carbonyl_c and m[0] in sub_set for m in mol.GetSubstructMatches(_FREE_ACID)):
        return True
    from .substituent_naming import acid_name_to_amido_prefix, fragment_acid_name_verified
    if not acid_name_to_amido_prefix(acid_name):
        return True
    chloride = _amide_of_acyl(mol, n_idx, sorted(sub_set - {n_idx}), halogen=17)
    if not chloride or not acyl_name:
        return False
    return bool(fragment_acid_name_verified(f"{acyl_name} chloride", chloride))


def polyacid_acyl_amido_prefix(mol, n_idx: int, carbonyl_c: int, sub_atoms,
                               verify: bool = False) -> Optional[str]:
    """The bare amido prefix of the N-attached acyl branch ``sub_atoms`` when its acyl holds a
    second free COOH ('2-amino-3-carboxypropanamido'), else None (fail closed).

    ``sub_atoms`` is the ENTIRE substituent (the amide N plus the acyl); ``verify`` asks for the
    OPSIN full-InChIKey read-back of the amide the prefix stands for.
    """
    sub_set = set(sub_atoms)
    acyl = sorted(sub_set - {n_idx})
    if n_idx not in sub_set or carbonyl_c not in acyl:
        return None
    if not any(m[0] != carbonyl_c and m[0] in sub_set for m in mol.GetSubstructMatches(_FREE_ACID)):
        return None  # no second acid in the acyl: the ordinary route owns it
    try:
        stand_in = Chem.RWMol(mol)
        atom = stand_in.GetAtomWithIdx(n_idx)
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        stand_in = stand_in.GetMol()
        Chem.SanitizeMol(stand_in)
        stereo_atoms = [i for i in acyl if mol.GetAtomWithIdx(i).GetChiralTag()
                        != Chem.ChiralType.CHI_UNSPECIFIED]
        if stereo_atoms and _cip_labels(mol, stereo_atoms) != _cip_labels(stand_in, stereo_atoms):
            return None  # the stand-in changed a CIP rank: its descriptors are not the original's
        from .substituent_enumerator import name_substituent
        acyl_name = name_substituent(stand_in, acyl, carbonyl_c, allow_mancude=False)
    except Exception:  # noqa: BLE001 - a producer that raises is a decline
        return None
    if not acyl_name or not acyl_name.endswith('oyl') or 'carboxy' not in acyl_name:
        return None
    amido = acyl_name[:-len('oyl')] + 'amido'
    from ..metrics.provenance import record_derived_non_pin_fragment
    record_derived_non_pin_fragment(acyl_name, amido)  # the acyl's non-PIN status carries to the prefix
    if verify:
        from .substituent_naming import fragment_acid_name_verified
        amide_smiles = _amide_of_acyl(mol, n_idx, acyl)
        if not amide_smiles or not fragment_acid_name_verified(amido[:-1] + 'e', amide_smiles):
            return None
    return amido
