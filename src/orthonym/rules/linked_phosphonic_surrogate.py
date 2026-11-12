"""The surrogate of a linked phosphonic acid and the multiplicative name built from it
(``multiplicative._try_linked_phosphonic_acids``)."""
import re
from typing import Optional

from rdkit import Chem
from rdkit.Chem import RWMol

#: The surrogate's sulfonic-acid suffix: '<parent>[-<locants>]-<n>sulfonic acid'.
_SULFONIC_SUFFIX_RE = re.compile(
    r'^(?P<head>.+?)(?:-(?P<locs>\d+[a-z]?[\x27′]*(?:,\d+[a-z]?[\x27′]*)+))?-?(?P<mult>di|tri|tetra)sulfonic acid$')


def _principal_group_is_sulfonic(surrogate) -> bool:
    """ "SENIORITY ORDER FOR CLASSES", Table 4.1 (the Blue Book): "7a 'Suffix' acids in the order
    carboxylic (not including carbonic or polycarbonic acids...), sulfonic, sulfinic,...". The surrogate is named
    '...sulfonic acid' only when its principal group is the sulfonic acid; with a carboxylic acid (or any senior
    group) elsewhere its name ends otherwise. A surrogate whose principal group cannot be read is not decided
    (True)."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        group, _ = get_principal_group(surrogate, detect_functional_groups(surrogate))
    except Exception:
        return True
    return group is None or 'sulfon' in group


def name_sulfonic_surrogate(surrogate) -> Optional[str]:
    """The name of ``surrogate``, the linked molecule with every phosphonic acid group drawn as a sulfonic acid
    group; None, without naming it, when a sulfonic acid cannot be its principal group (the caller reads only a name
    that ends '(di|tri|tetra)sulfonic acid')."""
    if not _principal_group_is_sulfonic(surrogate):
        return None
    from ..namer import Orthonym
    return Orthonym(style='pin', _disable_opsin_validity_gate=True).name(Chem.MolToSmiles(surrogate))


def _carboxylic_surrogate_name(mol, matches) -> Optional[str]:
    """The name of ``mol`` with every phosphonic acid group drawn as a carboxylic acid group (-P(=O)(OH)2 ->
    -C(=O)OH), when every group sits on a ring carbon; None otherwise. On a ring both acids are suffixes that do
    not take part in the ring ('-carboxylic acid', '-sulfonic acid',, so the suffix locants -- the
    free valences of the linking group -- are the same; the carboxylic surrogate is named where the sulfonic
    one is not ('bicyclo[2.2.1]heptane-2,7-dicarboxylic acid'). The caller has checked that a sulfonic acid
    would be the principal group, so a carboxylic acid is too, 7a)."""
    if not all(mol.GetAtomWithIdx(m[0]).IsInRing() for m in matches):
        return None
    rw = RWMol(mol)
    for m in matches:
        rw.GetAtomWithIdx(m[1]).SetAtomicNum(6)
    for idx in sorted((m[4] for m in matches), reverse=True):
        rw.RemoveAtom(idx)
    surrogate = rw.GetMol()
    try:
        Chem.SanitizeMol(surrogate)
        from ..namer import Orthonym
        return Orthonym(style='pin', _disable_opsin_validity_gate=True).name(Chem.MolToSmiles(surrogate))
    except Exception:
        return None


#: The carboxylic surrogate's suffix: '<ring>-<locants>-<n>carboxylic acid'.
_CARBOXYLIC_SUFFIX_RE = re.compile(
    r'^(?P<head>.+?)-(?P<locs>\d+[a-z]?[\x27′]*(?:,\d+[a-z]?[\x27′]*)+)-(?P<mult>di|tri|tetra)carboxylic acid$')


def linked_phosphonic_acids_name(mol, matches) -> Optional[str]:
    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS
    rw = RWMol(mol)
    for m in matches:
        rw.GetAtomWithIdx(m[1]).SetAtomicNum(16)
        # S(=O)(=O)O: the second hydroxy oxygen becomes the second oxo
        rw.GetBondBetweenAtoms(m[1], m[4]).SetBondType(Chem.BondType.DOUBLE)
        o = rw.GetAtomWithIdx(m[4])
        o.SetNoImplicit(True)
        o.SetNumExplicitHs(0)
    surrogate = rw.GetMol()
    try:
        Chem.SanitizeMol(surrogate)
        acid = name_sulfonic_surrogate(surrogate)
    except Exception:
        return None
    mm = _SULFONIC_SUFFIX_RE.match(acid or '')
    if not mm and _principal_group_is_sulfonic(surrogate):
        mm = _CARBOXYLIC_SUFFIX_RE.match(_carboxylic_surrogate_name(mol, matches) or '')
    if not mm:
        return None
    n = {'di': 2, 'tri': 3, 'tetra': 4}[mm.group('mult')]
    if n != len(matches):
        return None
    head, locs = mm.group('head'), mm.group('locs')
    if locs is None and head.endswith('methane'):
        linker, enclose = head[:-len('methane')] + 'methylene', head != 'methane'
    elif locs and head.endswith('benzene'):
        linker, enclose = f"{head[:-len('benzene')]}{locs}-phenylene", True
    elif locs:
        linker, enclose = f"{head}-{locs}-{mm.group('mult')}yl", True
    else:
        return None
    mult = COMPLEX_MULTIPLIERS.get(n)
    if not mult:
        return None
    #: a linker that already holds parentheses (a stereo block) is enclosed in square brackets
    opening, closing = ("[", "]") if "(" in linker else ("(", ")")
    name = f"{opening}{linker}{closing}{mult}(phosphonic acid)" if enclose else f"{linker}{mult}(phosphonic acid)"
    return name if _linked_name_rt_ok(name, mol) else None


def _linked_name_rt_ok(name: str, mol) -> bool:
    """OPSIN reads ``name`` back to ``mol``'s full InChIKey; False when it cannot
    be checked."""
    try:
        from ..validation.opsin_roundtrip import opsin_parse
        from ..perception.molcache import inchikey_of
        parsed = opsin_parse(name)
        pm = Chem.MolFromSmiles(parsed) if parsed else None
        return pm is not None and Chem.MolToInchiKey(pm) == inchikey_of(mol)
    except Exception:
        return False
