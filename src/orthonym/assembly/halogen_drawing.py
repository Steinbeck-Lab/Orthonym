"""A chlorine or bromine with a nonstandard bonding number, named through its iodine drawing.

RDKit gives chlorine and bromine the bonding number 1 only, and none to their anions, so it reads no SMILES of a
λ3-chlorane (``CC(C)(C)Cl=C``) or of a two-bond chloranuide (``C[Cl-]C``), and the engine has no molecule to name.
Iodine has the bonding numbers 1, 3, 5 and 7 in RDKit, and the producers of the λ-iodanes, the iodanuides and the
iodine rings name its drawing.:func:`iodine_drawing_row` names the molecule drawn with iodine in place of every such
atom and writes the chlorine or bromine stem for each iodine stem of that name; every iodine of the drawing is one of
those atoms, because an input with iodine of its own is not drawn again.

What the element changes, and how each is kept out of a shipped name:

* Parent and numbering: none. The halogens stand together in every seniority order these follow: "For
  naming and numbering purposes, the following decreasing order of element seniority is followed: F > Cl > Br > I >
  At > O >..." (the Blue Book), the classes of ("λ7, λ5 and λ3 halogen compounds in the order
  F > Cl > Br > I",:18215), the heteroatoms of rings and chains:8484,:9765).
* Alphanumerical order and the lowest locants it decides (g)): the stem is a letter of a prefix's
  name.:func:`stem_may_change_the_spelling` declines a name where the stem begins a prefix, or where another part of
  the name shares the letters of the stem's prefix up to the stem.
* Elision: a vowel elided before the 'i' of 'iod' ('benziodoxole', is not elided before 'chlor' or
  'brom'; a stem right after other letters is declined, except after 'yl' or a multiplying prefix.
* CIP rule 1a,:44850) ranks I above Br and Se, Cl below them: a descriptor can differ. The read-back
  compares the full standard InChIKey, stereo layer included, so such a name is declined, never shipped.

The name ships only when OPSIN reads it back to the full standard InChIKey of the input, with the label
systematic_verified: the input is drawn with chlorine or bromine, and the name of another drawing is never certified
as its PIN. The default tier declines it (NO_VERIFIED_PIN); a wider tier with no name to ship declines UNNAMEABLE.

Kept out of ``namer.py`` so that file stays within its size budget."""
import functools
import re
from typing import NamedTuple, Optional

from rdkit import Chem

#: the halogens drawn with iodine, and the stem written for 'iod' in the iodine drawing's name
_STEMS = {'Cl': 'chlor', 'Br': 'brom'}

_RELAXED = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES

_MULTIPLIER = r'(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|bis|tris|tetrakis)'
#: the iodine stem as (the start of) a substituent prefix: 'iodyl', 'iodosyl', 'iodonio', 'λ3-iodanyl', '...ylidene'
_SUBSTITUENT_FORM = re.compile(r'iod(?:yl|osyl|onio|an(?:uid|id)?yl)')
#: a locant ('2', "1,2'", 'N') right before the stem or its λ: the stem begins the prefix the locant belongs to
_LOCANT_END = re.compile(r"(?:[0-9][0-9,']*|[A-Z][0-9,'A-Z]*)$")


class HalogenDrawing(NamedTuple):
    iodine_smiles: str      # the input drawn with iodine in place of each halogen RDKit cannot read
    stem: str               # 'chlor' or 'brom'
    key: str                # the full standard InChIKey of the input
    radical: bool           # the input has a radical centre (OPSIN reads the name with -r)


def relaxed_mol(smiles: str) -> Optional[Chem.Mol]:
    """``smiles`` read without RDKit's valence check (every other sanitization step runs), or None."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, _RELAXED)
    except Exception:  # noqa: BLE001 -- a SMILES that fails another step is not this module's input
        return None
    return mol


def halogen_drawing(smiles) -> Optional[HalogenDrawing]:
    """The iodine drawing of ``smiles``, or None: not a string, no Cl or Br in it, RDKit reads it (nothing to draw
    again), an iodine of its own, unreadable atoms not all chlorine or all bromine, a charge other than -1, 0 or +1
    on one, or an iodine drawing RDKit does not read with the hydrogens and radical electrons of the input."""
    if not isinstance(smiles, str) or ('Cl' not in smiles and 'Br' not in smiles) \
            or Chem.MolFromSmiles(smiles) is not None:
        return None
    mol = relaxed_mol(smiles)
    if mol is None or any(a.GetAtomicNum() == 53 for a in mol.GetAtoms()):
        return None
    table = Chem.GetPeriodicTable()
    unread = [a for a in mol.GetAtoms() if a.GetSymbol() in _STEMS
              and a.GetTotalValence() not in table.GetValenceList(a.GetAtomicNum() - a.GetFormalCharge())]
    symbols = {a.GetSymbol() for a in unread}
    if len(symbols) != 1 or any(abs(a.GetFormalCharge()) > 1 for a in unread):
        return None
    work = Chem.RWMol(mol)
    for atom in unread:
        drawn = work.GetAtomWithIdx(atom.GetIdx())
        drawn.SetNumExplicitHs(atom.GetTotalNumHs())   # the hydrogens of the input, not those iodine would take
        drawn.SetNoImplicit(True)
        drawn.SetAtomicNum(53)
    try:
        Chem.SanitizeMol(work)
        iodine_smiles = Chem.MolToSmiles(work)
    except Exception:  # noqa: BLE001 -- an iodine drawing RDKit cannot read is not offered
        return None
    if any(work.GetAtomWithIdx(a.GetIdx()).GetNumRadicalElectrons() != a.GetNumRadicalElectrons() for a in unread):
        return None     # iodine would hold another bonding number: not the input's structure
    if Chem.MolFromSmiles(iodine_smiles) is None:
        return None
    key = Chem.MolToInchiKey(mol)
    if not key:
        return None
    return HalogenDrawing(iodine_smiles, _STEMS[symbols.pop()], key,
                          any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))


def _depth(text: str) -> int:
    return sum(ch in '([{' for ch in text) - sum(ch in ')]}' for ch in text)


def stem_may_change_the_spelling(iodine_name: str) -> bool:
    """True when writing 'chlor' or 'brom' for an 'iod' of ``iodine_name`` could need another spelling. Errs on the
    side of True.

    * A letter right before the stem, other than the 'yl' of a substituent or a multiplying prefix: a vowel may be
      elided before 'iod' ('benziodoxol') that is kept before a consonant.
    * The stem in a substituent prefix -- inside an enclosing mark, or a substituent form ('iodyl', 'iodanyl') --
      when it begins the prefix (after a locant, a multiplying prefix or an opening mark): the prefix sorts by the
      stem and may move.
    * The stem after other letters of its prefix ('difluoro-λ3-iodanyl') when another part of the name begins with
      the same letters ('difluoroethyl'): the comparison of the two reaches the stem.
    A stem in the parent (the hydride, the ring, an 'a' prefix of the parent) is in no prefix."""
    for found in re.finditer('iod', iodine_name):
        i = found.start()
        before = iodine_name[:i]
        if before[-1:].isalpha() and not re.search(r'(?:yl|' + _MULTIPLIER + r')$', before):
            return True
        word = re.match(r'[a-z]+', iodine_name[i:]).group(0)
        if _depth(before) == 0 and not _SUBSTITUENT_FORM.match(word):
            continue
        head = before[:-1] if before.endswith('-') else before
        lam = re.search(r'λ\d+$', head)
        if lam:
            head = head[:lam.start()]
        if _LOCANT_END.search(head) or re.search(_MULTIPLIER + r'$', head):
            return True
        if lam and head.endswith('-'):
            head = head[:-1]
        key = re.search(r'[a-z]*$', head).group(0)
        if not key:
            return True
        rest = iodine_name[:len(head) - len(key)] + ' ' + iodine_name[i + len(word):]
        if any(w.startswith(key) or re.sub('^' + _MULTIPLIER, '', w).startswith(key)
               for w in re.split(r'[^a-z]+', rest) if w):
            return True
    return False


def _declined(namer, smiles: str) -> dict:
    """The ``name_tiered`` row of a decline: NO_VERIFIED_PIN at the default tier, UNNAMEABLE at a wider one."""
    from ..errors import LIMIT_CATALOG
    from ..namer import _default_tier_declined_row
    row = _default_tier_declined_row(smiles)[1]
    if namer._default_tier_policy_applies():
        return row
    return {**row, 'name': LIMIT_CATALOG['UNNAMEABLE']['message'], 'limit_code': 'UNNAMEABLE'}


def iodine_drawing_row(namer, smiles) -> Optional[dict]:
    """The ``name_tiered`` row of an input RDKit cannot read because of a chlorine or bromine with a nonstandard
    bonding number, or None for any other input (the caller names it as before).

    The row is the iodine drawing's row with the chlorine or bromine name, read back by OPSIN to the input's full
    standard InChIKey, labelled systematic_verified, with the iodine drawing's name under ``iodine_name``; or a
    decline (:func:`_declined`)."""
    drawing = halogen_drawing(smiles)
    if drawing is None:
        return None
    if namer._default_tier_policy_applies():
        return _declined(namer, smiles)
    row = namer.name_tiered(drawing.iodine_smiles)
    iodine_name = row.get('name')
    if row.get('tier') == 'abstain' or not iodine_name or stem_may_change_the_spelling(iodine_name):
        return _declined(namer, smiles)
    name = iodine_name.replace('iod', drawing.stem)
    if not _reads_back(name, drawing):
        return _declined(namer, smiles)
    from rdkit.Chem import rdMolDescriptors
    return {**row, 'name': name, 'tier': 'systematic_verified', 'is_pin': False, 'source': 'iodine_drawing',
            'iodine_name': iodine_name, 'formula': rdMolDescriptors.CalcMolFormula(relaxed_mol(smiles)),
            'verified': 'opsin', 'spelling_failures': []}


def _reads_back(name: str, drawing: HalogenDrawing) -> bool:
    """OPSIN (the engine's jar) reads ``name`` to the full standard InChIKey of the input. The parse is read without
    RDKit's valence check, as the input is: the shared parser of ``validation/atom_coverage.py`` drops a parse RDKit
    cannot sanitize, and RDKit sanitizes no λ3-chlorane."""
    from ..jvm_bridge import opsin_stdout
    from ..validation.atom_coverage import _opsin_cli_stdout, find_opsin_jar
    jar = find_opsin_jar()
    if not jar:
        return False
    text, served = opsin_stdout(name, allow_radicals=drawing.radical, jar_path=jar)
    if not served:
        text = None if drawing.radical else _opsin_cli_stdout(name, jar)
    lines = (text or '').strip().split('\n')
    mol = relaxed_mol(lines[0].strip()) if lines and lines[0].strip() else None
    return mol is not None and Chem.MolToInchiKey(mol) == drawing.key


def halogen_drawing_scope(fn):
    """The decorator of ``Orthonym.name``, ``name_tiered`` and ``name_with_confidence`` that names a caller's input
    through:func:`iodine_drawing_row`; any other input, and a nested call, runs ``fn`` as before. ``name_tiered``
    returns the row, ``name_with_confidence`` its record, ``name`` its name; a decline gives the label (``name``
    raises the limit with ``raise_on_limit``)."""
    kind = fn.__name__

    @functools.wraps(fn)
    def _wrapper(self, smiles, *args, **kwargs):
        from .fragment_naming import is_top_level_naming
        row = iodine_drawing_row(self, smiles) if is_top_level_naming() else None
        if row is None:
            return fn(self, smiles, *args, **kwargs)
        if kind == 'name_tiered':
            return row
        from ..errors import _make
        limit = None if row['tier'] != 'abstain' else _make(row['limit_code'], row['name'], smiles)
        if kind == 'name_with_confidence':
            from .coverage_scoring import unmeasured_confidence
            record = unmeasured_confidence(name=row['name'], handler='iodine_drawing')
            return {**record, 'tier': row['tier'] if limit is None else None,
                    'limit': None if limit is None else limit.as_dict(), 'spelling_failures': []}
        if limit is not None:
            self._default_tier_row = (row['name'], row)
            if kwargs.get('raise_on_limit'):
                raise limit
        return row['name']
    return _wrapper
