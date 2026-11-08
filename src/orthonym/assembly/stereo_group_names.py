"""Names with 'rel' and 'rac' for a molecule whose whole configuration is one OR or AND stereo group.

An OR group of a CXSMILES string (``|o1:3,7|``) says: one enantiomer, which one is not known (the relative
configuration of its members is known). An AND group (``|&1:3,7|``) says: both enantiomers, a racemate. The Blue
Book names both with plain CIP descriptors and a prefix:

- "Relative configuration", (the Blue Book): "Relative configuration denoting
  diastereoisomers may be denoted by using the prefix 'rel' which is cited at the front of the name of one
  enantiomer... This prefix 'rel' with plain stereodescriptors is preferred in preferred IUPAC names";
  "rel-(1R,2S)-2-chlorocyclopentane-1-carboxylic acid (PIN)... [not rel-(1S,2R)-2-chlorocyclopentane-1-carboxylic
  acid]".
- "Racemates" (:45930): "Racemates may be denoted by using the prefix 'rac' cited at the front of the name
  of one enantiomer... This prefix 'rac' with plain stereodescriptors is preferred in preferred IUPAC names...
  When a racemate is described, the stereodescriptor, such as 'R' or 'RS', is cited for the chirality center having
  the lowest locant"; "rac-(2R,3R)-3-bromobutan-2-ol (PIN)", "rac-(2R)-2-bromobutane (PIN)".

So the name is the name of the enantiomer whose first-cited CIP descriptor is R, with 'rel-' or 'rac-' in front.
Both enantiomers are named by the ordinary, checked path (the drawn structure and its mirror image, each as a
string without the group), the one whose leading descriptor block starts with R is taken, and the prefix is added.
The prefix only widens what the name says (this enantiomer, or its mirror image), so the name is shipped only when
OPSIN reads the name without the prefix back to exactly that enantiomer and reads the whole name to the same
structure (OPSIN 2.9.0 reads 'rel'/'rac' and keeps the plain descriptors). The tier is the tier the enantiomer's
name earned.

Scope (anything else returns None, and the caller names the string as before, the group's configuration left out):
one fragment; exactly one AND or OR group and no other stereo element (every tetrahedral centre is in the group, no
stereo double bond); an OR group of at least two centres (a single centre has no relative configuration), an AND
group of at least one; no lone-pair centre (three-connected N, P, S: their SMILES reading differs between RDKit and
the standard one); not a group that inverts to itself (a meso structure, named as drawn already); a name that
starts with its descriptor block.

When the name does not start with its descriptor block ('tert-butyl (2R,5S)-...', 'N-[(1R,2S)-...]'), the book's
general form is used instead, at a call that is not PIN-only: every CIP descriptor of the name starred for an OR
group, 'R*'/'S*', "the 'starred' stereodescriptors 'R*', 'S*'... that may be" used), or doubled for an
AND group, 'RS'/'SR', "the stereodescriptors such as 'RS', 'SR', etc. that may be used in general
nomenclature"), the first of them R. Only the descriptor blocks the descriptor check reads are rewritten
(``rules.stereochemistry.locate_stereo_blocks``), and the name passes the same read-back. Such a name is not the
PIN: it is labelled systematic_verified.
"""
import re
import threading
from typing import Callable, Optional, Tuple

from rdkit import Chem

_ACTIVE = threading.local()

_TETRAHEDRAL = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW}

#: the leading descriptor block of a name: '(1R,2S)-...', '(2R)-...', '(R)-...'
_LEADING_BLOCK = re.compile(r"^\(([^()]+)\)-")
#: one item of a block: optional locant (digits, primes, letters such as 'N' or 'a'), then the descriptor
_ITEM = re.compile(r"^(?:[0-9]+[a-z]?['′]*|[A-Za-z]['′]*|)([A-Za-z*]+)$")


def _forms(smiles: str) -> Optional[Tuple[str, str, str]]:
    """(prefix, SMILES of the drawn enantiomer, SMILES of its mirror image) for a string in scope, else None."""
    if not isinstance(smiles, str) or "|" not in smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            return None
        groups = list(mol.GetStereoGroups())
        if len(groups) != 1:
            return None
        group = groups[0]
        kind = group.GetGroupType()
        if kind == Chem.StereoGroupType.STEREO_OR:
            prefix, least = "rel", 2
        elif kind == Chem.StereoGroupType.STEREO_AND:
            prefix, least = "rac", 1
        else:
            return None
        members = sorted(atom.GetIdx() for atom in group.GetAtoms())
        if list(group.GetBonds()) or len(members) < least:
            return None
        if any(a.GetChiralTag() not in _TETRAHEDRAL and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
               for a in mol.GetAtoms()):
            return None                                   # square-planar, octahedral...: inverting a tag is no mirror
        centres = sorted(a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() in _TETRAHEDRAL)
        if centres != members:
            return None
        if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
            return None
        if any(mol.GetAtomWithIdx(k).GetTotalDegree() == 3 for k in members):
            return None                                   # a lone-pair centre (N, P, S)
        drawn = Chem.RWMol(mol)
        drawn.SetStereoGroups([])
        mirror = Chem.RWMol(drawn)
        for k in members:
            atom = mirror.GetAtomWithIdx(k)
            atom.SetChiralTag(_TETRAHEDRAL[atom.GetChiralTag()])
        drawn_smiles = Chem.MolToSmiles(drawn.GetMol())
        mirror_smiles = Chem.MolToSmiles(mirror.GetMol())
        if drawn_smiles == mirror_smiles:
            return None                                   # meso: the drawn string is the compound
        return prefix, drawn_smiles, mirror_smiles
    except Exception:
        return None


def leading_descriptor(name: str) -> Optional[str]:
    """The first R or S of the name's leading descriptor block ('(1S,2R)-...' -> 'S'), or None when the name does not
    start with a block or the block's first CIP chirality descriptor is not R or S (an E/Z item before it is skipped;
    a lowercase pseudoasymmetric r/s, a starred or RS descriptor gives None)."""
    if not isinstance(name, str):
        return None
    match = _LEADING_BLOCK.match(name)
    if not match:
        return None
    for item in match.group(1).split(","):
        found = _ITEM.match(item.strip())
        if not found:
            return None
        code = found.group(1)
        if code in ("E", "Z"):
            continue
        return code if code in ("R", "S") else None
    return None


def _inchi(smiles: Optional[str]) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return Chem.MolToInchi(mol) if mol is not None else None
    except Exception:
        return None


def _below_pin(result):
    """A row labelled below the PIN (a starred or doubled name is not the PIN); other results unchanged."""
    if isinstance(result, dict) and result.get("tier") in ("pin_verified", "pin_unverified"):
        return {**result, "tier": "systematic_verified", "is_pin": False}
    return result


def _without_stereo(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _read_back_ok(name: str, prefixed: str, enantiomer_smiles: str,
                  to_smiles: Callable[[str], Optional[str]]) -> bool:
    """OPSIN reads ``name`` to exactly ``enantiomer_smiles``, and ``prefixed`` to the same structure or to its
    constitution alone (OPSIN 2.9.0 reads 'rac-(2R)-2-bromobutane' without a configuration)."""
    want = _inchi(enantiomer_smiles)
    plain = _inchi(to_smiles(name))
    whole = _inchi(to_smiles(prefixed))
    return bool(want) and plain == want and whole in (want, _inchi(_without_stereo(enantiomer_smiles)))


#: a plain R or S descriptor anywhere: an item of a parenthesised list, optionally locanted, not starred or doubled
_PLAIN_DESCRIPTOR = re.compile(r"(?<=[(,])(?:[0-9]+[a-z]?['\u2032]*|[A-Za-z]['\u2032]*)?[RS](?=[,)])")
_WORD_BLOCK = re.compile(r"(?<= )(\(([^()]+)\))-")
_BLOCK_ITEM = re.compile(r"^((?:[0-9]+[a-z]?['\u2032]*|[A-Za-z]['\u2032]*)?)([A-Za-z]+)$")


def general_form(name: str, prefix: str) -> Optional[str]:
    """``name`` with every CIP chirality descriptor of its descriptor blocks starred ('rel': R -> R*, S -> S*) or
    doubled ('rac': R -> RS, S -> SR), or None when a block cannot be rewritten in place, holds a descriptor of
    another kind (E/Z, r/s are kept), or the first rewritten descriptor is not R."""
    from ..rules.stereochemistry import locate_stereo_blocks
    if not isinstance(name, str):
        return None
    try:
        blocks = list(locate_stereo_blocks(name))
    except Exception:
        return None
    # the locator reads leading and nested blocks; a block after a space ('tert-butyl (2R,5S)-...', the second word
    # of an ester or a salt name) is read here with the same item grammar
    seen = {start for start, _, _ in blocks}
    for found in _WORD_BLOCK.finditer(name):
        start = found.start(1)
        if start not in seen and all(_BLOCK_ITEM.match(item.strip()) for item in found.group(2).split(",")):
            blocks.append((start, found.end(), found.group(1)))
    blocks.sort()
    swap = {"R": "R*", "S": "S*"} if prefix == "rel" else {"R": "RS", "S": "SR"}
    out, last, first = [], 0, None
    for start, end, text in blocks:
        if name[start:end] != text + "-" or not (text.startswith("(") and text.endswith(")")):
            return None
        items = []
        for item in text[1:-1].split(","):
            found = _BLOCK_ITEM.match(item.strip())
            if not found:
                return None
            locant, code = found.groups()
            if code in swap:
                first = first or code
                code = swap[code]
            elif code not in ("E", "Z", "r", "s"):
                return None
            items.append(locant + code)
        out.append(name[last:start] + "(" + ",".join(items) + ")-")
        last = end
    if first != "R":
        return None
    rewritten = "".join(out) + name[last:]
    # a block the locators do not read ('{(1R)-...', '(1'R)') would keep a plain, absolute descriptor, which OPSIN
    # 2.9.0 reads like the starred ones (so the read-back cannot see it): refuse any plain R or S left in the name
    if _PLAIN_DESCRIPTOR.search(rewritten):
        return None
    return rewritten


def stereo_group_name(smiles, name_one: Callable[[str], object], name_of: Callable[[object], Optional[str]],
                      with_name: Callable[[object, str], object],
                      to_smiles: Callable[[str], Optional[str]], general: bool = False):
    """The result of naming ``smiles`` with 'rel-'/'rac-' (see the module docstring), or None to name it as before.

    ``name_one(s)`` names a string without a group (the public call itself); ``name_of(r)`` is the name in a result
    (None for no name); ``with_name(r, n)`` is the result with its name replaced; ``to_smiles(n)`` is OPSIN's SMILES of
    a name; ``general`` allows the starred or doubled form (a call that is not PIN-only). A nested call (the namings of the two enantiomers) is not in scope."""
    if getattr(_ACTIVE, "on", False):
        return None
    forms = _forms(smiles)
    if forms is None:
        return None
    prefix, drawn, mirror = forms
    _ACTIVE.on = True
    try:
        found, named = [], []
        for form in (drawn, mirror):
            result = name_one(form)
            name = name_of(result)
            named.append((form, result, name))
            if leading_descriptor(name) != "R":
                continue
            prefixed = f"{prefix}-{name}"
            if _read_back_ok(name, prefixed, form, to_smiles):
                found.append((prefixed, result))
        if not found and general:
            for form, result, name in named:
                if leading_descriptor(name) is not None:
                    continue            # a leading block that failed above is not rewritten in the general form
                rewritten = general_form(name, prefix)
                if rewritten and _read_back_ok(name, rewritten, form, to_smiles):
                    found.append((rewritten, _below_pin(result)))
        if not found:
            return None
        # both enantiomers' names lead with R (another numbering of the mirror image): the same choice whichever of
        # the two the string drew (core principle 4, determinism)
        prefixed, result = min(found, key=lambda pair: pair[0])
        return with_name(result, prefixed)
    except Exception:
        return None
    finally:
        _ACTIVE.on = False


#: the public naming calls that take a SMILES string and return a name, a row or a NamingResult
_ENTRY_METHODS = ("name", "name_tiered", "name_with_confidence", "name_with_tree")


def stereo_group_scope(fn):
    """Wrap a public naming method: a string in scope is named by:func:`stereo_group_name`, any other string, and
    any case it does not build, as before (namer._enhanced_stereo_cleared_spelling leaves the group's configuration
    out)."""
    import functools

    @functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        given = args[0] if args else kwargs.get("smiles")
        if not isinstance(given, str) or "|" not in given:
            return fn(self, *args, **kwargs)
        from ..errors import is_failure_name
        from ..namer import _validity_gate_name_to_opsin_smiles
        rest = tuple(args[1:])
        extra = {k: v for k, v in kwargs.items() if k != "smiles"}

        def name_of(result):
            if isinstance(result, dict):
                name = result.get("name")
            elif hasattr(result, "_replace"):
                name = getattr(result, "name", None)
            else:
                name = result
            return name if isinstance(name, str) and name and not is_failure_name(name) else None

        def with_name(result, name):
            # a NamingResult keeps the enantiomer's tree: 'rel'/'rac' is outside every part of it
            if isinstance(result, dict):
                return {**result, "name": name}
            return result._replace(name=name) if hasattr(result, "_replace") else name

        try:
            general = not self._default_tier_policy_applies()
        except Exception:
            general = False
        built = stereo_group_name(given, lambda form: fn(self, form, *rest, **extra), name_of, with_name,
                                  _validity_gate_name_to_opsin_smiles, general)
        return built if built is not None else fn(self, *args, **kwargs)
    return _wrapper


def install(cls) -> None:
    """Wrap the public naming methods of ``cls`` (the Orthonym class) once, outermost (orthonym/__init__.py)."""
    for method in _ENTRY_METHODS:
        current = getattr(cls, method)
        if not getattr(current, "_stereo_group_scope", False):
            wrapped = stereo_group_scope(current)
            wrapped._stereo_group_scope = True
            setattr(cls, method, wrapped)
