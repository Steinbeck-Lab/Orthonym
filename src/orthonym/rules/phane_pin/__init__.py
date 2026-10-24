"""Phane names for the structures names with a phane name.

The producers read the same structure test as the spelling check of the cyclic phane rule
(``orthonym.validation.spelling.cyclophane``): a cyclophane is what the check expects a phane name for.
"""
from typing import Optional

from rdkit import Chem

from ...validation.spelling.cyclophane import is_p52_cyclophane, p52_cyclophane_systems

__all__ = ["general_phane_name", "is_p52_cyclophane", "linear_phane_name", "p52_cyclophane_systems",
           "verified_or_none"]

_MEMO_MOL = None
_MEMO: dict = {}
_LINEAR_MEMO_MOL = None
_LINEAR_MEMO: dict = {}


def _plain(mol: Chem.Mol) -> bool:
    """One component, neutral, no radical, no isotope, no specified stereo (a name of such a structure is
    built from its constitution alone)."""
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            return False
    return all(b.GetStereo() == Chem.BondStereo.STEREONONE for b in mol.GetBonds())


def _kekule(mol: Chem.Mol) -> Chem.Mol:
    """A copy with Kekule bond types and the aromatic flags kept. RDKit can read a macrocycle that joins aromatic
    rings through lone-pair atoms (two N-aryl groups between two naphthalenes) as one aromatic system and give
    its bonds the aromatic type; a bond between an amplificant and an acyclic skeletal atom is single."""
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=False)
    except Exception:  # noqa: BLE001 -- RDKit raises several types for a molecule it cannot kekulize
        return mol
    return kekule


def _build(mol: Chem.Mol) -> Optional[str]:
    from .naming import phane_name
    from .skeleton import perceive_cyclic
    from .substituents import plan_substituents
    if not _plain(mol) or not is_p52_cyclophane(mol):
        return None
    mol = _kekule(mol)
    skel = perceive_cyclic(mol)
    if skel is None:
        return None
    plan = plan_substituents(skel)
    if plan is None:
        return None
    return phane_name(skel, plan)


def general_phane_name(mol: Optional[Chem.Mol]) -> Optional[str]:
    """The phane name of a cyclophane (1)) as writes it, or None when the structure is
    outside what is built. Memoised for the last molecule."""
    global _MEMO_MOL, _MEMO
    if mol is None:
        return None
    if _MEMO_MOL is not mol:
        _MEMO_MOL, _MEMO = mol, {}
    if "v" not in _MEMO:
        _MEMO["v"] = _build(mol)
    return _MEMO["v"]


def _build_linear(mol: Chem.Mol) -> Optional[str]:
    """: the phane name of the chain that makes the linear phane the parent (the reader's verdict),
    or None. A structure with a cyclophane is named by it: cyclic phane before linear phane)."""
    from ...validation.spelling.linear_phane import linear_phane_chains, linear_phane_pin_expected
    from .linear import perceive_linear
    from .naming import phane_candidate
    from .substituents import plan_substituents
    if not _plain(mol) or mol.GetRingInfo().NumRings() < 4:
        return None
    verdict = linear_phane_pin_expected(mol)
    if verdict is None or verdict.as_prefix or verdict.principal_class == "ion":
        return None
    if is_p52_cyclophane(mol):
        return None
    candidates = []
    chains = linear_phane_chains(mol)
    mol = _kekule(mol)
    for chain in chains:
        skel = perceive_linear(mol, chain)
        plan = plan_substituents(skel) if skel is not None else None
        got = phane_candidate(skel, plan) if plan is not None else None
        if got is not None:
            candidates.append(got)
    if not candidates:
        return None
    best = min(candidates)
    return best[1]


def linear_phane_name(mol: Optional[Chem.Mol]) -> Optional[str]:
    """The linear phane name of a structure whose PIN is one (2),, as writes it, or
    None when the structure is outside what is built. Memoised for the last molecule."""
    global _LINEAR_MEMO_MOL, _LINEAR_MEMO
    if mol is None:
        return None
    if _LINEAR_MEMO_MOL is not mol:
        _LINEAR_MEMO_MOL, _LINEAR_MEMO = mol, {}
    if "v" not in _LINEAR_MEMO:
        _LINEAR_MEMO["v"] = _build_linear(mol)
    return _LINEAR_MEMO["v"]


def verified_or_none(name: Optional[str], smiles: Optional[str]) -> Optional[str]:
    """``name`` when OPSIN reads it back to the full InChIKey of ``smiles`` (and agrees on radicals and on the place
    of a hydron), else None: the producers offer a name and the round trip keeps or voids it. The read is the engine's
    own (``namer._shipped_name_round_trip``, memoised, no fail-open branch), so an OPSIN that cannot be asked voids
    the name too."""
    if not name or not smiles:
        return None
    from ... import namer
    return name if namer._shipped_name_round_trip(name, smiles) == "verified" else None
