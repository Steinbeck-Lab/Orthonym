"""The decorator on a molecule whose carboxy groups are labelled in different ways: one suffix, the rest prefixes.

 (heading 'Isotopically modified compounds', the Blue Book): "When two characteristic
groups are isotopically modified in different ways so that they cannot be combined using multiplicative terms
such as 'di-', 'bis-', etc., the isotopically modified characteristic group with the greater number of
modifications is chosen as the principal characteristic group to be cited as a suffix; the other characteristic
is then cited as a prefix" ('1-(13C)carboxycyclohexane-1-(14C)carboxylic acid (PIN)',:43778).

The decorator of ``rules.isotopes`` names the isotope-STRIPPED molecule, where the groups are alike and are cited
as one multiplied suffix ('benzene-1,2-dicarboxylic acid'); a label on one of them has no place in that name. With
one group the suffix, the parent is numbered again by the usual rules (lowest locant to the suffix, then to the
prefixes, then the first prefix in alphanumerical order), which a split of the multiplied suffix cannot do: the
label on C-2 of '4-methylbenzene-1,2-dicarboxylic acid' gives '2-carboxy-5-methylbenzene-1-(13C)carboxylic acid'.
So the numbering is the engine's: every other carboxy group is drawn as a carboxamide, which the engine cites as
the prefix 'carbamoyl' with the one acid as the suffix, and that name is spelled with 'carboxy'. No prefix may
sort between the two words ('carbamoyl' < 'carboxy'), or the alphanumerical tie-break could differ.

The labelled group is the most modified one and carries every label of the molecule. The name is OPSIN-read-back
gated like every other candidate, and labelled below the PIN: the PIN status of the split is not checked here.
"""
import re
from typing import Optional

from rdkit import Chem

from . import isotopes as _iso

#: a carboxy group: an acyclic carbon with a double-bonded and a hydroxy oxygen (the H may be an explicit atom)
_CARBOXY = Chem.MolFromSmarts("[CX3;!R](=[OX1])[OX2;$([OX2H1]),$([OX2][#1])]")
#: a prefix word that begins with 'carb' and could sort between 'carbamoyl' and 'carboxy'
_CARB_WORD = re.compile(r"carb[a-z]*")
_BENZOIC = "benzoic acid"
#: the carboxy groups of the unmodified name cited as one multiplied suffix
_MULTIPLIED_SUFFIX = re.compile(r"(?:di|tri|tetra|penta|hexa)carboxylic acid")
_SUFFIX = "carboxylic acid"
#: an amide of the input: its 'carbamoyl' could not be told from a converted group
_AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]")


def _group_atoms(mol: Chem.Mol, match) -> set:
    """The atoms of a carboxy group: C, both O and an explicit H on the hydroxy oxygen."""
    atoms = set(match)
    atoms.update(n.GetIdx() for n in mol.GetAtomWithIdx(match[2]).GetNeighbors() if n.GetAtomicNum() == 1)
    return atoms


def surrogate_smiles(original: Chem.Mol) -> Optional[str]:
    """The isotope-stripped molecule with every carboxy group but the labelled one drawn as a carboxamide, or None
    when the labels do not all sit on one of two or more carboxy groups, or the input has an amide of its own."""
    matches = original.GetSubstructMatches(_CARBOXY)
    if len(matches) < 2 or original.HasSubstructMatch(_AMIDE):
        return None
    labelled = {a.GetIdx() for a in original.GetAtoms() if a.GetIsotope()}
    holders = [m for m in matches if labelled & _group_atoms(original, m)]
    if len(holders) != 1 or not labelled <= _group_atoms(original, holders[0]):
        return None
    rw = Chem.RWMol(original)
    drop = []
    for m in matches:
        if m is holders[0]:
            continue
        oxygen = rw.GetAtomWithIdx(m[2])
        drop.extend(n.GetIdx() for n in oxygen.GetNeighbors() if n.GetAtomicNum() == 1)
        oxygen.SetAtomicNum(7)
        oxygen.SetNoImplicit(False)
        oxygen.SetNumExplicitHs(0)
    for idx in sorted(drop, reverse=True):
        rw.RemoveAtom(idx)
    for atom in rw.GetAtoms():
        atom.SetIsotope(0)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:  # noqa: BLE001 - no surrogate, no name
        return None


def decorate_with_one_suffix(namer, original: Chem.Mol, skeleton_name: str) -> Optional[str]:
    """The name of ``original`` with its labelled carboxy group the suffix and the others 'carboxy' prefixes,
    numbered by the engine on the carboxamide surrogate (module docstring); None when there is no such name.
    Only for groups the unmodified name ``skeleton_name`` combines with a multiplier ('benzene-1,2-dicarboxylic
    acid'), the case is about; a 'carboxymethyl' and a 'benzoic acid' are different groups already."""
    if not _MULTIPLIED_SUFFIX.search(skeleton_name or ""):
        return None
    smiles = surrogate_smiles(original)
    if smiles is None:
        return None
    from ..metrics.provenance import get_provenance, record_non_pin_fragment, restore_provenance
    before = get_provenance()
    try:
        skeleton = _iso._name_skeleton(_iso._flagged_systematic_namer(namer), smiles)
    finally:
        restore_provenance(before)      # what the surrogate's naming recorded is about another molecule
    if not skeleton or "carbamoyl" not in skeleton:
        return None
    # the input has no amide (``surrogate_smiles``), so every 'carbamoyl' is a converted group; any other word
    # that begins with 'carb' could sort between 'carbamoyl' and 'carboxy'
    if any(not w.startswith(("carbamoyl", "carboxylic")) for w in _CARB_WORD.findall(skeleton)):
        return None
    skeleton = skeleton.replace("carbamoyl", "carboxy")
    # (:44251): 'benzoic acid' numbers no position of its carboxy carbon, so the systematic spelling
    # of the same parent is tried first ('benzene(13C)carboxylic acid (PIN)',:44255)
    spellings = [skeleton]
    if skeleton.endswith(_BENZOIC):
        spellings.insert(0, skeleton[:-len(_BENZOIC)] + "benzene-1-carboxylic acid")
    # the descriptor goes on the SUFFIX, the labelled group, and nowhere else: a label placed on a
    # 'carboxy' prefix can read back exact on a symmetric parent while it names the labelled group as the prefix
    keys = _iso._label_keys(original)
    for spelling in spellings:
        if not spelling.endswith(_SUFFIX):
            continue
        head = spelling[:-len(_SUFFIX)]
        for force in (False, True):
            desc = _iso.format_isotope_descriptor(
                [(None, mass, el, count, maxpos) for (mass, el), count, maxpos in keys], force_show=force)
            candidate = head + desc + _SUFFIX
            if _iso._isotope_round_trips(candidate, original):
                record_non_pin_fragment(candidate)
                return candidate
    return None
