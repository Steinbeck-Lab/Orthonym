"""The class of a zinc input for the label audit, read from the structure alone.

``zinc_label_class(smiles)`` puts a drawing of a zinc compound into one class of section 3 of the
plan ``docs/the workflow tooling/plans/2026-10-12-zinc-and-organometallics.md`` (the tier and label policy)
or into one class that the audit of ZN11 found. The classes are the keys of ``CLASS_BASIS`` in
``tests/unit/namer/test_zinc_labels.py``. The function uses RDKit and nothing of the engine, so a
label that the engine sets by a predicate is audited by a reading that shares no code with it.

A PIN label belongs to a class only where the Blue Book defines a PIN for it (the user's rule of
2026-10-12, with two named exceptions). The classes that may carry ``pin_verified`` are listed in
``PIN_CLASSES``; every other class is ``systematic_verified`` or lower at every tier.
"""
from typing import List

from rdkit import Chem, RDLogger

#: the classes whose names the default tier may emit as pin_verified (CLASS_BASIS gives each its basis)
PIN_CLASSES = (
    "SALT_OF_ACID",
    "SALT_OF_ALCOHOL_OR_THIOL",
    "SALT_OF_AMIDE_ANION",
    "CYANIDE_SALT",
    "ORGANIC_CATION_SALT",
    "CARBON_FREE_SALT",
)

#: the classes whose names are never labelled a PIN
NON_PIN_CLASSES = (
    "ORGANOZINC",
    "CARBANION_SALT",
    "BONDED_DRAWING",
    "DATIVE_DRAWING",
    "ADDUCT",
    "MACROCYCLE_COMPLEX",
    "ATOM_OR_ION",
)

#: a drawing the classifier cannot place: a name of it must not be labelled a PIN until the audit places it
UNPLACED = "UNPLACED"


_ACID_ANION = [Chem.MolFromSmarts(p) for p in (
    "[O-]-[#6]=[O,S]", "[S-]-[#6]=[O,S]",                 # carboxylate, dithiocarboxylate, thiocarboxylate
    "[O-]-[S,P,N,B,Cl,Br,I,Se,As]~[O,S,N]",               # sulfonate, sulfate, phosphate, nitrate, borate...
    "[O-]-[#6](-[O-])=O",                                  # carbonate
)]
_CARBANION = Chem.MolFromSmarts("[#6;-1;!$([#6]#[#7])]")
_CYANIDE = Chem.MolFromSmarts("[#6-]#[#7]")
_ALKOXIDE_THIOLATE = Chem.MolFromSmarts("[O-,S-;X1]-[#6;!$([#6]=[O,S,N]);!$([#6]=[#6]-[#6]=[O,S])]")
_AZANIDE = Chem.MolFromSmarts("[N-;X2]([#6,#16])")


def _parse(smiles: str):
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def _fragments(mol) -> List[Chem.Mol]:
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def _has_carbon(mol) -> bool:
    return any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


def _net_charge(mol) -> int:
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


def _is_macrocyclic_ion(frag) -> bool:
    """A fragment with a ring of 16 or more atoms that holds four ring nitrogens: the porphyrin and
    phthalocyanine ions (the dianion of a tetrapyrrole)."""
    ring_info = frag.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) >= 16 and sum(1 for i in ring if frag.GetAtomWithIdx(i).GetSymbol() == "N") >= 4:
            return True
    # a macrocycle ring is fused from the five- and six-membered rings: count the nitrogens of the core
    # by the ring systems when no single ring is large enough
    nitrogens = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() == "N" and a.IsInRing()]
    heavy = frag.GetNumHeavyAtoms()
    return len(nitrogens) >= 4 and heavy >= 24 and frag.GetRingInfo().NumRings() >= 5


def zinc_label_class(smiles: str) -> str:
    """The class of the zinc drawing ``smiles`` (see the module docstring); ``'NO_ZINC'`` for a drawing
    without a zinc atom and ``UNPLACED`` for one the classifier cannot place."""
    mol = _parse(smiles)
    if mol is None:
        return UNPLACED
    zinc = [a for a in mol.GetAtoms() if a.GetSymbol() == "Zn"]
    if not zinc:
        return "NO_ZINC"
    bonds = [b for a in zinc for b in a.GetBonds()]
    if any(b.GetBondType() == Chem.BondType.DATIVE for b in bonds):
        return "DATIVE_DRAWING"
    if any(b.GetOtherAtom(a).GetSymbol() == "C" for a in zinc for b in a.GetBonds()):
        return "ORGANOZINC"
    if bonds or any(a.GetTotalNumHs() > 0 for a in zinc):
        return "BONDED_DRAWING"
    if mol.GetNumAtoms() == 1:
        return "ATOM_OR_ION"

    cations, anions, neutrals = [], [], []
    for frag in _fragments(mol):
        charge = _net_charge(frag)
        (cations if charge > 0 else anions if charge < 0 else neutrals).append(frag)
    if any(_is_macrocyclic_ion(f) for f in anions + cations + neutrals):
        return "MACROCYCLE_COMPLEX"
    # a molecule or a neutral metal atom beside the metal, or an assembly with no anion: an adduct
    # (a molecule of water is a hydrate of a salt)
    if any(not (f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() == "O") for f in neutrals):
        return "ADDUCT"
    if cations and not anions:
        return "ADDUCT"

    # a cation of more than one atom (a diazonium, an ammonium of carbon) is an organic cation
    organic_cation = any(f.GetNumAtoms() > 1 or _has_carbon(f) for f in cations)
    if any(f.HasSubstructMatch(_CARBANION) for f in anions):
        return "CARBANION_SALT"
    if not _has_carbon(mol):
        return "CARBON_FREE_SALT"
    if any(f.GetNumAtoms() == 2 and f.HasSubstructMatch(_CYANIDE) for f in anions):    # the ion itself, no isocyano group
        return "CYANIDE_SALT"
    if organic_cation:
        return "ORGANIC_CATION_SALT"
    kinds = set()
    for frag in anions:
        if not _has_carbon(frag):
            kinds.add("INORGANIC")
        elif any(frag.HasSubstructMatch(p) for p in _ACID_ANION):
            kinds.add("ACID")
        elif frag.HasSubstructMatch(_ALKOXIDE_THIOLATE):
            kinds.add("ALCOHOL")
        elif frag.HasSubstructMatch(_AZANIDE):
            kinds.add("AMIDE")
        else:
            kinds.add("OTHER")
    if "OTHER" in kinds:
        return UNPLACED
    if "AMIDE" in kinds:
        return "SALT_OF_AMIDE_ANION"
    if "ACID" in kinds:
        return "SALT_OF_ACID"
    if "ALCOHOL" in kinds:
        return "SALT_OF_ALCOHOL_OR_THIOL"
    return UNPLACED
