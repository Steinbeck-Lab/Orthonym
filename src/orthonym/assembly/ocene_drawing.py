"""A substituted metallocene drawn as a metal(2+) and two cyclopentadienide anions, named with its ocenyl prefix.

The ocene table (``data.organometallics``) names the bare sandwich ('ferrocene'); a substituted one was named as a
salt whose anion spelling is not that of the compound ('iron(II) 1-({[4-(cyclopenta-1,3-dien-5-id-1-yl)phenyl]...'),
or not at all. OPSIN reads the ocenyl prefix names ('ferrocen-1-ylmethanol', '4-(ferrocen-1-yl)benzoic acid',
'2-(osmocen-1-yl)ethan-1-ol') to the same ionic drawing, so such a name can be read back to the input.

The ocenyl group has no chokepoint among the producers of substituent prefixes, so it is named through a drawing:
the input with the metal and the bare ring removed and the substituted ring saturated is a cyclopentyl compound; the
engine names it, and the one 'cyclopentyl' prefix of that name is written '(<ocene>-1-yl)'. That is a token rewrite
with a check of everything the rewrite can change:

* Alphanumerical order and the numbering it decides (g)): declined when any other word of the name
  sorts between 'cyclopentyl' and the ocenyl name.
* Enclosing marks: only a 'cyclopentyl' outside every mark, a whole prefix (not 'cyclopentyloxy', not
  'dicyclopentyl'), is rewritten; the ocenyl prefix has a locant, so it is enclosed.
* The structure: the name ships only when OPSIN reads it to the input's full standard InChIKey.

Scope (fail closed, None otherwise): three fragments, a metal of the ocene table with charge 2+, a bare
cyclopentadienide and a cyclopentadienide with one substituent on one ring carbon, no other charge; a wider tier
only (the default tier keeps its own answer). The label is systematic_verified: an ocene has no PIN here.
"""
import re
from typing import Optional

from rdkit import Chem

#: the metal(2+) -> the ocene stem of the bare sandwich's name ('ferrocene')
_OCENE_STEMS = {'Fe': 'ferrocen', 'Ru': 'ruthenocen', 'Os': 'osmocen', 'Co': 'cobaltocen', 'Ni': 'nickelocen',
                'Cr': 'chromocen', 'V': 'vanadocen', 'Mn': 'manganocen'}
_SURROGATE = 'cyclopentyl'
_MULTIPLIER = r'(?:di|tri|tetra|penta|hexa|bis|tris|tetrakis)'


def _cp_ring(frag):
    """The five ring atoms of a cyclopentadienide fragment (all-carbon five-ring, one ring carbanion, two ring double
    bonds) or None."""
    rings = [r for r in frag.GetRingInfo().AtomRings() if len(r) == 5]
    for r in rings:
        atoms = [frag.GetAtomWithIdx(i) for i in r]
        if any(a.GetSymbol() != 'C' for a in atoms):
            continue
        if [a.GetFormalCharge() for a in atoms].count(-1) != 1 or any(a.GetFormalCharge() not in (0, -1) for a in atoms):
            continue
        km = Chem.Mol(frag)
        try:
            Chem.Kekulize(km, clearAromaticFlags=True)
        except Exception:  # noqa: BLE001 -- not this shape
            continue
        doubles = sum(1 for b in km.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                      and b.GetBeginAtomIdx() in r and b.GetEndAtomIdx() in r)
        if doubles == 2:
            return r
    return None


def _surrogate(smiles: str):
    """``(ocene stem, surrogate SMILES)`` for an input in scope, else None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) != 3:
        return None
    metals = [f for f in frags if f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() in _OCENE_STEMS]
    if len(metals) != 1 or metals[0].GetAtomWithIdx(0).GetFormalCharge() != 2:
        return None
    rest = [f for f in frags if f is not metals[0]]
    bare = [f for f in rest if f.GetNumAtoms() == 5 and _cp_ring(f) is not None]
    if not bare:
        return None
    sub = rest[1] if rest[0] is bare[0] else rest[0]
    ring = _cp_ring(sub)
    if ring is None or sub.GetNumAtoms() <= 5:
        return None
    if any(a.GetFormalCharge() for a in sub.GetAtoms() if a.GetIdx() not in ring):
        return None
    out = [(i, nb.GetIdx()) for i in ring for nb in sub.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in ring]
    if len(out) != 1 or sub.GetBondBetweenAtoms(*out[0]).GetBondType() != Chem.BondType.SINGLE:
        return None
    if any(a.GetIsotope() or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in sub.GetAtoms() if a.GetIdx() in ring):
        return None
    rw = Chem.RWMol(sub)
    for b in rw.GetBonds():
        if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring:
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
    for i in ring:
        a = rw.GetAtomWithIdx(i)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        surrogate = rw.GetMol()
        Chem.SanitizeMol(surrogate)
    except Exception:  # noqa: BLE001 -- not this shape
        return None
    if sum(1 for r in surrogate.GetRingInfo().AtomRings()
           if len(r) == 5 and all(surrogate.GetAtomWithIdx(i).GetSymbol() == 'C' for i in r)) != 1:
        return None                     # another cyclopentane would make the 'cyclopentyl' token ambiguous
    return _OCENE_STEMS[metals[0].GetAtomWithIdx(0).GetSymbol()], Chem.MolToSmiles(surrogate)


def rewrite_surrogate_prefix(name: str, stem: str) -> Optional[str]:
    """``name`` with its one 'cyclopentyl' prefix written '(<stem>-1-yl)', or None when the rewrite could change
    anything but that prefix (see the module docstring)."""
    hits = [m for m in re.finditer(_SURROGATE, name)]
    if len(hits) != 1:
        return None
    i, j = hits[0].start(), hits[0].end()
    before, after = name[:i], name[j:]
    if (before[-1:].isalpha() and not before.endswith('-')) or after[:1].isalpha():
        return None                     # 'dicyclopentyl', 'cyclopentyloxy', 'cyclopentylidene'
    depth = sum(1 if c in '([{' else -1 if c in ')]}' else 0 for c in before)
    if depth:
        return None                     # inside a compound prefix: the marks would step up
    target = stem + 'yl'
    low, high = sorted((_SURROGATE, target))
    for w in re.findall(r'[a-z]+', before + ' ' + after):
        # a compound prefix sorts with its multiplier ('dimethylanilino'), a simple one without it: either decides
        if low < w < high or low < re.sub('^' + _MULTIPLIER, '', w) < high:
            return None                 #: another prefix would change places with it
    if (not before or before.endswith(' ')) and (not after or after.startswith(' ')):
        return f"{before}{stem}-1-yl{after}"   # a word of its own (an ester's organyl group) takes no marks
    return f"{before}({stem}-1-yl){after}"


def ocene_drawing_row(namer, smiles: str) -> Optional[dict]:
    """The ``name_tiered`` row of a substituted ocene in scope at a wider tier, or None (the caller names it)."""
    if not isinstance(smiles, str) or namer._default_tier_policy_applies():
        return None                     # an RDKit Mol input is named through its own SMILES by the caller
    found = _surrogate(smiles)
    if found is None:
        return None
    stem, surrogate = found
    row = namer.name_tiered(surrogate)
    if row.get('tier') == 'abstain' or not row.get('name'):
        return None
    name = rewrite_surrogate_prefix(row['name'], stem)
    if name is None:
        return None
    from ..validation.opsin_roundtrip import opsin_parse
    parsed = opsin_parse(name)
    back = Chem.MolFromSmiles(parsed) if parsed else None
    if back is None or Chem.MolToInchiKey(back) != Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)):
        return None
    from rdkit.Chem import rdMolDescriptors
    return {**row, 'name': name, 'tier': 'systematic_verified', 'is_pin': False, 'source': 'ocene_drawing',
            'formula': rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)), 'verified': 'opsin',
            'spelling_failures': [], 'limit_code': None}
