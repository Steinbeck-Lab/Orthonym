"""The standard reading of a SMILES string at a lone-pair stereocentre, as an RDKit Mol.

A lone pair is an implicit hydrogen to the SMILES standard (OpenSMILES), which ``centres`` (CDK) and
OPSIN follow. RDKit reads a lone-pair stereocentre otherwise in some positions of a string (the atom
written first, a ring-closure digit at P(III) and at an aziridine N), and its InChI reads the Mol RDKit
made of the string, so RDKit's InChIKey of a string depends on where the centre stands in it. A round
trip compares RDKit's InChIKey of the input with that of OPSIN's SMILES of the name, each as RDKit reads
it, and the two strings put the centre in positions RDKit reads differently: a right name is rejected
and the mirror image accepted (OPSIN's own StdInChIKey of '(1R,2S)-1,2-diphenylphospholane' is that of
the input 'C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1'; RDKit's key of OPSIN's SMILES of that name,
'c1ccccc1[P@]1[C@@H](CCC1)c1ccccc1', is the key of the (1S,2S) isomer).

:func:`standard_reading_mol` is the Mol a round trip compares both sides as: RDKit's reading of a string
with the tag of each lone-pair stereocentre set so that ``rdCIPLabeler``, the labeller that reads the Mol
and not a string of it, gives the label the standard reading of the string gives (``centres`` on the
string as written). Its InChIKey is the key of the configuration the string means, whatever the position
of the centre in it, for the centres below. Without a lone-pair stereocentre it is RDKit's reading.

What it holds, and what it does not:

* ``centres`` labels N (in a three-membered ring), P, S and Se, and gives no label to As, Sb, Bi, Te or Po.
  The parity rule of a string does not depend on the element, so such an atom is read through a stand-in
  of the same valence, P for As, Sb and Bi and S for Te and Po (:func:`stand_in_string`, written into the
  string in place and audited: the stand-in must read as the same atoms, bonds and tags): both labellers
  read the stand-in, and the flip they decide is applied to the atom of the element. The labels of a
  stand-in are labels of the stand-in, to be compared with one another and with nothing else.
* RDKit drops the chirality of some lone-pair centres at parsing (N of an N-aryl aziridine, Sb, Bi, Te,
  Po) that ``centres`` and the standard InChI keep.:func:`raw_reading` is the reading that keeps them,
  for the checks that must see everything the string defines.
* A pseudoasymmetric (r/s) label is never flipped on, a label ``centres`` or ``rdCIPLabeler`` does not
  give leaves RDKit's own tag, and so does ``rdCIPLabeler`` past its bound: nothing is then shown, and
  every check that needs the label fails closed.
"""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from . import centres_bridge

logger = logging.getLogger(__name__)

#: The CIP labels two readings are compared by; pseudoasymmetric r/s are never flipped on.
_RS = ('R', 'S')

#: The elements ``centres`` gives no label at a lone-pair stereocentre, and the element of the same valence
#: it reads in their place (measured: As, Sb, Bi, Te and Po centres get no label from ``centres``; N, P, S
#: and Se do).
STAND_IN = {'As': 'P', 'Sb': 'P', 'Bi': 'P', 'Te': 'S', 'Po': 'S'}
#: The elements that carry a lone-pair stereocentre: N, P, As, Sb, Bi, S, Se, Te, Po (a stray tag on another
#: atom of three ligands, a carbonyl carbon, is no centre and stays dropped).
_LONE_PAIR_ELEMENTS = frozenset((7, 15, 33, 51, 83, 16, 34, 52, 84))
_STAND_IN_RE = re.compile(r'(\[\d*)(' + '|'.join(STAND_IN) + r')(?![a-z])')


class _NoStandardReading(Exception):
    """``centres`` gave no labels for the string (no jar or no JVM): nothing to compare. Raised, not
    returned, so that:func:`_flips` does not remember the absence."""


def lone_pair_centres(mol) -> List[int]:
    """Indices of the atoms of ``mol`` the lone-pair check compares: a tetrahedral stereocentre
    (``namer._lone_pair_compared_atom``) with three ligands in all, the fourth position being the lone
    pair. An atom with an unpaired electron is no such centre: the fourth position holds the radical
    electron ('C[C@@H]1CC[P@@](=O)C1' is a phospholanyl radical, whose P the standard InChI does not see)."""
    from .. import namer
    return [a.GetIdx() for a in mol.GetAtoms()
            if namer._lone_pair_compared_atom(a) and a.GetTotalDegree() == 3
            and not a.GetNumRadicalElectrons()]


def dropped_centres_invisible_to_inchi(smiles: Optional[str]) -> bool:
    """True iff ``smiles`` writes a lone-pair centre RDKit drops when it parses (``raw_reading``) and the
    standard InChI of the Mol with the centre's tag and without it is one InChI: no key, of any reader of the
    structure, tells the two configurations apart (Sb, Bi, Te, Po; not the N of an N-aryl aziridine, whose
    configuration the InChI holds). Such a configuration cannot be confirmed or refuted by a round trip."""
    from rdkit.Chem import inchi
    if not smiles or '@' not in smiles:
        return False
    try:
        raw = raw_reading(smiles)
        clean = Chem.MolFromSmiles(smiles)
        if raw is None or clean is None or len(lone_pair_centres(raw)) <= len(lone_pair_centres(clean)):
            return False
        return bool(inchi.MolToInchi(raw)) and inchi.MolToInchi(raw) == inchi.MolToInchi(clean)
    except Exception:
        return False


def raw_reading(smiles: Optional[str]):
    """RDKit's reading of ``smiles`` (hydrogens removed) with the tag of every lone-pair stereocentre the
    string writes, also of the ones RDKit drops when it parses: ``Chem.MolFromSmiles`` ends with a stereo
    clean-up that removes the tag of a lone-pair centre it does not take for a stereocentre (the N of an
    N-aryl aziridine, Sb, Bi, Te, Po), which ``centres`` (the N) and the standard InChI keep. The tag is
    restored there, for the elements of a lone-pair centre; every other atom is RDKit's own reading. Each atom carries ``_lp_pos``, its 1-based
    position in the string, which is the position ``centres`` labels. None when RDKit cannot read the
    string."""
    if not smiles:
        return None
    params = Chem.SmilesParserParams()
    params.removeHs = False
    clean = Chem.MolFromSmiles(smiles, params)
    raw = _unsanitised(smiles)
    if clean is None or raw is None or clean.GetNumAtoms() != raw.GetNumAtoms():
        return None
    from .. import namer
    written = {}
    for atom in clean.GetAtoms():
        atom.SetIntProp('_lp_pos', atom.GetIdx() + 1)
        tag = raw.GetAtomWithIdx(atom.GetIdx()).GetChiralTag()
        # a tag RDKit dropped, on an atom with no hydrogen written as an atom of its own (removing that
        # atom would permute the neighbours the tag counts)
        if (atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED and tag in namer._LP_TETRAHEDRAL
                and atom.GetAtomicNum() in _LONE_PAIR_ELEMENTS
                and not any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())):
            written[atom.GetIdx() + 1] = tag
    mol = Chem.RemoveHs(clean)
    for atom in mol.GetAtoms():
        tag = written.get(atom.GetIntProp('_lp_pos'))
        if tag is not None and atom.GetTotalDegree() == 3:
            atom.SetChiralTag(tag)
    return mol


def _graph_signature(mol):
    """What a stand-in must keep: every atom's tag, charge, isotope and hydrogen count, every bond."""
    atoms = tuple((int(a.GetChiralTag()), a.GetFormalCharge(), a.GetIsotope(), a.GetNumExplicitHs(),
                   a.GetNoImplicit()) for a in mol.GetAtoms())
    bonds = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType()),
                          int(b.GetBondDir())) for b in mol.GetBonds()))
    return atoms, bonds


def _unsanitised(smiles: str):
    params = Chem.SmilesParserParams()
    params.sanitize = False
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)


@lru_cache(maxsize=8192)
def stand_in_string(smiles: str) -> Optional[str]:
    """``smiles`` with each bracketed As, Sb and Bi written P and each Te and Po written S, in place, so
    that the position of every atom in the string stays what it was; ``smiles`` itself when it has none of
    them; None when the stand-in is not shown to read as the same atoms, bonds and tags."""
    if not _STAND_IN_RE.search(smiles):
        return smiles
    stand_in = _STAND_IN_RE.sub(lambda m: m.group(1) + STAND_IN[m.group(2)], smiles)
    try:
        a, b = _unsanitised(smiles), _unsanitised(stand_in)
        if a is None or b is None or a.GetNumAtoms() != b.GetNumAtoms():
            return None
        return stand_in if _graph_signature(a) == _graph_signature(b) else None
    except Exception:
        return None


def standard_centres_labels(smiles: str) -> Optional[Dict[int, str]]:
    """``centres``' labels of ``smiles`` as written, keyed by 1-based position in the string, with As, Sb,
    Bi, Te and Po read through their stand-ins (:func:`stand_in_string`); None when ``centres`` is
    unavailable or the stand-in cannot be shown. A label of a stand-in is a label of the stand-in."""
    stand_in = stand_in_string(smiles)
    if stand_in is None:
        return None
    batch = centres_bridge.centres_label_batch([stand_in])
    if batch is None or stand_in not in batch:
        return None
    return batch[stand_in]


def geometry_labels(mol, atoms: List[int]) -> Dict[int, str]:
    """``rdCIPLabeler``'s labels of ``atoms`` of ``mol`` (an atom of an element with a stand-in is labelled
    as its stand-in), keyed by atom index; an atom it gives none is left out."""
    from rdkit.Chem import rdCIPLabeler

    from .. import namer
    held = Chem.Mol(mol)
    for atom in held.GetAtoms():
        if atom.GetSymbol() in STAND_IN:
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(STAND_IN[atom.GetSymbol()]))
    rdCIPLabeler.AssignCIPLabels(held, atomsToLabel=atoms, bondsToLabel=[],
                                 maxRecursiveIterations=namer._LP_CIP_MAX_ITERATIONS)
    return {i: held.GetAtomWithIdx(i).GetProp('_CIPCode') for i in atoms
            if held.GetAtomWithIdx(i).HasProp('_CIPCode')}


def canonical_smiles_keeping_the_reading(smiles: str) -> Optional[str]:
    """RDKit's canonical SMILES of ``smiles``, None when RDKit cannot read it; a string with a lone-pair
    stereocentre comes back as it is. RDKit writes its canonical SMILES for its own reading, and the
    string it writes can be read as the other configuration by a standard reader (``centres``, OPSIN) at
    a lone-pair centre, so what a string with such a centre means is kept by keeping the string."""
    mol = raw_reading(smiles)
    if mol is None:
        return None
    return smiles if '@' in smiles and lone_pair_centres(mol) else Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@lru_cache(maxsize=8192)
def _flips(smiles: str) -> Tuple[int, ...]:
    """The atoms (indices of RDKit's reading of ``smiles``, hydrogens removed) whose tag is inverted
    to give the standard reading: the lone-pair stereocentres at which ``rdCIPLabeler``'s label of
    that reading is not the ``centres`` label of the string as written (both read the stand-in of an
    element ``centres`` does not label). Empty when there is none or when it cannot be shown."""
    from .. import namer
    try:
        stand_in = stand_in_string(smiles)
        if stand_in is None:
            return ()
        mol = namer._lp_heavy_mol(stand_in)
        if mol is None:
            return ()
        centres = lone_pair_centres(mol)
        if not centres:
            return ()
        batch = centres_bridge.centres_label_batch([stand_in])
        if batch is None or stand_in not in batch:
            raise _NoStandardReading()
        written = batch[stand_in]
        rdkit = geometry_labels(mol, centres)
        out = []
        for i in centres:
            standard = written.get(mol.GetAtomWithIdx(i).GetIntProp('_lp_pos'))
            label = rdkit.get(i)
            if standard in _RS and label in _RS and label != standard:
                out.append(i)
        return tuple(out)
    except _NoStandardReading:
        raise
    except Exception:  # RDKit's CIP labeller past its bound, or any other error: RDKit's reading stands
        logger.debug("standard reading of a lone-pair stereocentre failed; RDKit's reading kept")
        return ()


def standard_reading_flips(smiles: Optional[str]) -> Tuple[int, ...]:
    """:func:`_flips`, empty where nothing can be shown (no ``centres``)."""
    if not smiles or '@' not in smiles:
        return ()
    try:
        return _flips(smiles)
    except _NoStandardReading:
        return ()


def standard_reading_mol(smiles: Optional[str]):
    """RDKit's reading of ``smiles`` with the tag of each lone-pair stereocentre set to the standard
    reading (see the module docstring); None when RDKit cannot read ``smiles``."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or '@' not in smiles:
        return mol
    if not lone_pair_centres(mol):
        return mol
    flips = standard_reading_flips(smiles)
    if not flips:
        return mol
    from .. import namer
    heavy = namer._lp_heavy_mol(smiles)
    work = Chem.RWMol(heavy)
    for i in flips:
        work.GetAtomWithIdx(i).InvertChirality()
    return work.GetMol()


def standard_reading_inchikey(smiles: Optional[str]) -> Optional[str]:
    """The full standard InChIKey of:func:`standard_reading_mol`, or None when ``smiles`` has none."""
    from .molcache import inchikey_of
    mol = standard_reading_mol(smiles)
    return (inchikey_of(mol) or None) if mol is not None else None
