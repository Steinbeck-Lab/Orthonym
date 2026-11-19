"""Anions of the inorganic oxoacids, named from their acid and the 'hydrogen' anions of.

The anion of an oxoacid whose acid has a name in ``rules.inorganic_acids`` ('arsorous acid', 'diphosphonic acid') is
named by changing 'ic acid' to 'ate' and 'ous acid' to 'ite'; hydrogen atoms of hydroxy groups that
remain are cited as 'hydrogen', 'dihydrogen',... in front ('dihydrogen phosphate', 'hydrogen arsorite'). Hydrogen
on the central atom is not an acidic hydrogen and is not cited ('hydrogen arsonite' for HAs(OH)O-).

Two acids that the Blue Book does not name are used for their anions only: orthoperiodic acid H5IO6 and orthotelluric
acid Te(OH)6 (IUPAC Red Book, Table.1); OPSIN 2.9.0 reads 'orthoperiodate', 'tetrahydrogen
orthoperiodate', 'orthotellurate' and 'orthotelluric acid' back to the full InChIKey. The neutral H5IO6 keeps the name
the engine gives it; Te(OH)6, which had none, is named 'orthotelluric acid'.

Every name is carbon-free, so a preselected name at most; the caller's read-back decides whether it ships.
This producer runs after every class namer, so a molecule that has a name today keeps it.
"""
from typing import Optional

from rdkit import Chem

#: acids named here for their anions only (Red Book, Table.1), as canonical SMILES of the neutral acid
_ANION_ONLY_ACIDS = {"OI(O)(O)(O)(O)=O": "orthoperiodic acid", "O[Te](O)(O)(O)(O)O": "orthotelluric acid"}
#: the neutral acids among them that are named here as acids too (the engine names none of them otherwise)
_NEUTRAL_ACIDS = frozenset({"orthotelluric acid"})
_HYDROGEN = {1: "hydrogen", 2: "dihydrogen", 3: "trihydrogen", 4: "tetrahydrogen", 5: "pentahydrogen"}

_ACID_NAMES = None


def _double_bond_form(mol):
    """``mol`` with each charge-separated X(+)-O(-) pair of a central atom drawn as X=O (the engine and the acid
    table draw the oxo groups of iodine and chlorine either way)."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        while atom.GetFormalCharge() > 0 and atom.GetSymbol() != "O":
            oxide = next((n for n in atom.GetNeighbors() if n.GetSymbol() == "O" and n.GetFormalCharge() == -1
                          and n.GetDegree() == 1 and not n.GetTotalNumHs()), None)
            if oxide is None:
                break
            rw.GetBondBetweenAtoms(atom.GetIdx(), oxide.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            oxide.SetFormalCharge(0)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
    return rw


def _canonical(mol) -> Optional[str]:
    """Canonical SMILES of ``mol`` with the valence check left out (RDKit refuses the heptavalent iodine of H5IO6)."""
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        mol.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _acid_names():
    """Canonical SMILES -> acid name: the oxoacid table of rules.inorganic_acids and the anion-only acids."""
    global _ACID_NAMES
    if _ACID_NAMES is None:
        from .inorganic_acids import _INORGANIC_OXOACIDS
        table = {}
        for smi, value in {**_INORGANIC_OXOACIDS, **_ANION_ONLY_ACIDS}.items():
            name = value if isinstance(value, str) else (value[0] if value else None)
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            key = _canonical(_double_bond_form(mol)) if mol is not None else None
            if key and isinstance(name, str) and name.endswith(" acid"):
                table.setdefault(key, name)
        _ACID_NAMES = table
    return _ACID_NAMES


def _anion_word(acid: str) -> Optional[str]:
    """'arsorous acid' -> 'arsorite', 'diphosphonic acid' -> 'diphosphonate'; the stems 'sulfur' and
    'phosphor' shorten to 'sulf' and 'phosph' ('sulfuric acid' -> 'sulfate', 'phosphorous acid' -> 'phosphite',
    'diphosphoric acid' -> 'diphosphate')."""
    stem = acid[:-len(" acid")]
    for long, short in (("sulfuric", "sulfic"), ("sulfurous", "sulfous"),
                        ("phosphoric", "phosphic"), ("phosphorous", "phosphous")):
        if stem.endswith(long):
            stem = stem[:-len(long)] + short
    if stem.endswith("ous"):
        return stem[:-3] + "ite"
    if stem.endswith("ic"):
        return stem[:-2] + "ate"
    return None


def name_oxoacid_anion(mol) -> Optional[str]:
    """The name of an anion of an inorganic oxoacid (every negative charge on a terminal oxygen, no other charge), or
    of a neutral acid of ``_NEUTRAL_ACIDS``; else None."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)          # explicit [H] atoms count as hydrogen of their atom
    except Exception:
        return None
    mol = _double_bond_form(mol)
    anionic = []
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge == 0:
            continue
        if charge != -1 or atom.GetSymbol() != "O" or atom.GetDegree() != 1 or atom.GetTotalNumHs():
            return None
        anionic.append(atom.GetIdx())
    acid = Chem.RWMol(mol)
    for idx in anionic:
        atom = acid.GetAtomWithIdx(idx)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
    key = _canonical(acid)
    name = _acid_names().get(key) if key else None
    if name is None:
        return None
    if not anionic:
        return name if name in _NEUTRAL_ACIDS else None
    word = _anion_word(name)
    hydroxy_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    if word is None or (hydroxy_h and hydroxy_h not in _HYDROGEN):
        return None
    return f"{_HYDROGEN[hydroxy_h]} {word}" if hydroxy_h else word


def red_book_word(name: Optional[str]) -> Optional[str]:
    """The anion or acid word of ``name`` that comes from an acid the Blue Book does not name ('orthoperiodate',
    'orthotelluric acid'), else None."""
    if not name:
        return None
    for acid in _ANION_ONLY_ACIDS.values():
        for word in (acid, _anion_word(acid)):
            if word and word in name:
                return word
    return None


__all__ = ["name_oxoacid_anion", "red_book_word"]
